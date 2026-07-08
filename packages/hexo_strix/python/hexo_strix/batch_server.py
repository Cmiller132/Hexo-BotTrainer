"""Cross-game leaf-batching inference server for the ported HeXONet.

Concurrent Strix games each run the Rust Gumbel MCTS in their own thread; when a
search needs its leaves scored it builds the round into a
:class:`~hexo_strix.batched_infer.RoundBatch` (one zero-copy Rust call) and calls
:meth:`StrixBatchServer.evaluate`, which enqueues it and blocks. A single
background batcher thread coalesces the pending rounds from ALL in-flight games
into one batched GPU forward (:func:`~hexo_strix.batched_infer.concat_rounds` +
:func:`~hexo_strix.batched_infer.batched_eval_round`) and scatters the per-request
results back.

Design invariant: **only the batcher thread touches the model / GPU.** The game
threads do CPU-side Rust search + graph building (both release the GIL) and then
block on a result event, so the GPU sees one stream of large batches and the CPU
work parallelises across cores. This is the same leaf-batching idea hexfield's
native multi-root session uses, bolted around Strix's single-game Rust search.

The coalesced forward is numerically equal to a per-game forward up to
floating-point reduction order (batched matmul vs single-graph) — acceptable for
an eval opponent (see :mod:`hexo_strix.batched_infer`).
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .batched_infer import (
    GraphTensors,
    RoundBatch,
    batched_eval_round,
    concat_rounds,
    round_from_graph_tensors,
)


@dataclass
class _Request:
    round: RoundBatch
    n: int  # number of graphs (leaves) in this request's round
    event: threading.Event = field(default_factory=threading.Event)
    result: tuple[list[list[float]], list[float]] | None = None
    error: BaseException | None = None


_STOP = object()


class StrixBatchServer:
    """Coalesce leaf-eval requests from concurrent games into one GPU forward.

    Parameters
    ----------
    model : the HeXONet (already ``.to(device).eval()``); owned by the server.
    device : the forward device (e.g. ``"cuda"``).
    max_batch : soft cap on graphs per coalesced forward; the batcher stops
        gathering once the pending graph count reaches this.
    linger_s : how long the batcher waits, after the first pending request, to
        gather more requests before running the forward. A small value (sub-ms)
        trades a little latency for much larger GPU batches when several games
        are in flight. 0 disables lingering (drain-what's-there only).
    """

    def __init__(
        self,
        model: Any,
        device: str = "cuda",
        *,
        max_batch: int = 512,
        linger_s: float = 0.0008,
    ) -> None:
        self._model = model.to(device).eval()
        self._device = device
        self._max_batch = int(max_batch)
        self._linger_s = float(linger_s)
        self._q: "queue.Queue[Any]" = queue.Queue()
        self._closed = False
        # Guards the accepting-flag + enqueue against the batcher's shutdown drain
        # so a late request can never be orphaned (block forever on its event).
        self._lock = threading.Lock()
        self._accepting = True
        # Stats (read after close): forwards run, total graphs, batch histogram.
        self.n_forwards = 0
        self.n_graphs = 0
        self.max_seen_batch = 0
        self._thread = threading.Thread(target=self._run, name="strix-batcher", daemon=True)
        self._thread.start()

    # --- called by game threads ---
    def evaluate(
        self, unit: RoundBatch | list[GraphTensors]
    ) -> tuple[list[list[float]], list[float]]:
        """Score a round (blocking); returns ``(logits_per_graph, values)``.

        ``unit`` is a :class:`RoundBatch` (the fast zero-copy path) or a legacy
        list of per-graph :data:`GraphTensors` (converted here). Thread-safe; may
        be called concurrently from many game threads. Raises whatever the
        batched forward raised (surfaced to the caller rather than hanging the
        game thread).
        """

        round_batch = unit if isinstance(unit, RoundBatch) else round_from_graph_tensors(unit)
        if round_batch.num_graphs == 0:
            return [], []
        req = _Request(round=round_batch, n=round_batch.num_graphs)
        # Enqueue under the lock so this request is either seen by the batcher's
        # shutdown drain (and errored) or rejected here — never orphaned. Without
        # this, a worker that puts AFTER the batcher consumed _STOP would block on
        # req.event forever, hanging pool.shutdown() in play_strix_match's finally.
        with self._lock:
            if not self._accepting:
                raise RuntimeError("StrixBatchServer is closed")
            self._q.put(req)
        req.event.wait()
        if req.error is not None:
            raise req.error
        assert req.result is not None
        return req.result

    # --- batcher thread ---
    def _run(self) -> None:
        while True:
            first = self._q.get()
            if first is _STOP:
                self._shutdown_drain()
                return
            batch: list[_Request] = [first]
            total = first.n
            # Linger to gather more in-flight games' requests into one forward.
            deadline = time.monotonic() + self._linger_s
            stop_after = False
            while total < self._max_batch:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    nxt = self._q.get(timeout=remaining)
                except queue.Empty:
                    break
                if nxt is _STOP:
                    stop_after = True
                    break
                batch.append(nxt)
                total += nxt.n

            self._dispatch(batch)
            if stop_after:
                self._shutdown_drain()
                return

    def _shutdown_drain(self) -> None:
        """On stop, refuse new requests and fail every one already queued.

        Sets ``_accepting = False`` under the lock (so any concurrent
        :meth:`evaluate` either already enqueued — and is drained below — or is
        rejected), then errors out every pending request so no game thread blocks
        on its event after the batcher has exited. This is what keeps
        play_strix_match's fail-open contract intact when the forward raises
        mid-round (e.g. CUDA OOM): each in-flight decide() raises instead of
        hanging, the edge is dropped, and the eval continues.
        """

        with self._lock:
            self._accepting = False
        err = RuntimeError("StrixBatchServer closed before this request was served")
        while True:
            try:
                pending = self._q.get_nowait()
            except queue.Empty:
                break
            if pending is _STOP:
                continue
            pending.error = err
            pending.event.set()

    def _dispatch(self, batch: list[_Request]) -> None:
        coalesced = concat_rounds([req.round for req in batch])
        n_graphs = coalesced.num_graphs
        try:
            logits_per, values = batched_eval_round(self._model, coalesced, self._device)
        except BaseException as exc:  # noqa: BLE001 — surface to every waiting game
            for req in batch:
                req.error = exc
                req.event.set()
            return
        self.n_forwards += 1
        self.n_graphs += n_graphs
        self.max_seen_batch = max(self.max_seen_batch, n_graphs)
        pos = 0
        for req in batch:
            k = req.n
            req.result = (logits_per[pos : pos + k], values[pos : pos + k])
            pos += k
            req.event.set()

    def close(self) -> None:
        """Stop the batcher thread (idempotent). Pending requests already queued
        before close are still served."""

        if self._closed:
            return
        self._closed = True
        self._q.put(_STOP)
        self._thread.join(timeout=30.0)

    def __enter__(self) -> "StrixBatchServer":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
