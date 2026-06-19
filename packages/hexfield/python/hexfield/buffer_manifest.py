"""Persisted, mtime-free shard manifest for the KataGo-style replay buffer.

Ordering shards by ``st_mtime`` is fragile: resume/checkpoint/``touch`` perturb
it. This manifest instead uses a **stable ordering by ``(generation,
game_key)``** persisted to ``<samples_dir>/.buffer_manifest.json`` and updated
**incrementally** so the per-epoch re-glob + ``stat()``-sort goes away (it does
NOT replace the per-epoch window *decode* — that is what the Rust path is for).

Two row totals are tracked:

* ``total_rows`` — the **live** sum over present entries. Used for *window*
  selection (taper / recent-window / keep_prob).
* ``cumulative_rows_ever`` — a **monotone** counter, ``max(prev, live_total)``,
  never decremented even when shards are pruned. Fed to the train-bucket
  *governor*; feeding it a non-monotone count would spuriously trip the
  ``elif total_rows < level_at_row`` reload branch and zero the reuse counter.

Generation tagging never uses mtime: the key-derived epoch (``game_key //
1_000_000``, structurally guaranteed by the selfplay shard writer) is
**authoritative**; the sidecar ``epoch`` is a cross-check that WARNs on mismatch;
the directory name is the last-resort fallback when the sidecar is absent.

Robustness rules:

* Shards lacking a JSON sidecar are SKIPPED — the live writer emits the ``.npz``
  before (or concurrently with) its ``.json`` sidecar, so a sidecar-less file may
  be half-written; the manifest must never open a torn ``.npz``.
* Entries whose ``.npz`` file has vanished are pruned.
* A missing / garbled / version-mismatched manifest triggers a full rebuild
  (self-healing) rather than a crash.
* The manifest is written atomically (tmp + ``os.replace``) so a crash mid-update
  never wedges a resume.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

# Bump when the persisted manifest schema changes incompatibly; a mismatch
# discards the file and rebuilds from the tree.
MANIFEST_VERSION = 1

# File name under ``samples_dir``. Leading dot keeps it out of the
# ``game_*.npz`` glob and visually distinct from shard data.
MANIFEST_NAME = ".buffer_manifest.json"

# Rows per epoch in a game key, mirroring the selfplay shard writer
# (``next_key = epoch * 1_000_000``). game_key = epoch * 1_000_000 + within-epoch
# index.
_KEYS_PER_EPOCH = 1_000_000


@dataclass(frozen=True)
class ShardEntry:
    """One self-play shard. ``rel_path`` is POSIX-relative to ``samples_dir``
    (portable + the stable key the md5 train/val split hashes). ``generation`` is
    the producing epoch; ``game_key`` is the within-run game id
    (``epoch * 1_000_000 + i``)."""

    rel_path: str
    rows: int
    generation: int
    game_key: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "rel_path": self.rel_path,
            "rows": int(self.rows),
            "generation": int(self.generation),
            "game_key": int(self.game_key),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ShardEntry":
        # Raises (KeyError/TypeError/ValueError) on a malformed entry; the caller
        # treats any such failure as a corrupt manifest and rebuilds.
        return cls(
            rel_path=str(raw["rel_path"]),
            rows=int(raw["rows"]),
            generation=int(raw["generation"]),
            game_key=int(raw["game_key"]),
        )


@dataclass
class BufferManifest:
    version: int = MANIFEST_VERSION
    # Sorted by (generation, game_key) — stable, mtime-free.
    entries: list[ShardEntry] = field(default_factory=list)
    # Live sum over present entries (drives WINDOW selection).
    total_rows: int = 0
    # MONOTONE, never decremented (drives the GOVERNOR).
    cumulative_rows_ever: int = 0

    # -- derived helpers -------------------------------------------------

    def resort(self) -> None:
        """Re-establish the canonical ``(generation, game_key)`` ordering."""
        self.entries.sort(key=lambda e: (e.generation, e.game_key))

    def recompute_totals(self) -> None:
        """Recompute ``total_rows`` from entries and advance the monotone
        ``cumulative_rows_ever`` (never decremented)."""
        self.total_rows = sum(int(e.rows) for e in self.entries)
        self.cumulative_rows_ever = max(int(self.cumulative_rows_ever), int(self.total_rows))

    # -- (de)serialization ----------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": int(self.version),
            "total_rows": int(self.total_rows),
            "cumulative_rows_ever": int(self.cumulative_rows_ever),
            "entries": [e.to_dict() for e in self.entries],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "BufferManifest":
        """Strict load — raises on version mismatch or malformed structure so
        the caller can self-heal with a full rebuild."""
        if not isinstance(raw, Mapping):
            raise ValueError("manifest is not a JSON object")
        version = int(raw.get("version", 0))
        if version != MANIFEST_VERSION:
            raise ValueError(f"manifest version {version} != {MANIFEST_VERSION}")
        raw_entries = raw.get("entries", [])
        if not isinstance(raw_entries, list):
            raise ValueError("manifest 'entries' is not a list")
        entries = [ShardEntry.from_dict(item) for item in raw_entries]
        man = cls(
            version=version,
            entries=entries,
            total_rows=int(raw.get("total_rows", 0)),
            cumulative_rows_ever=int(raw.get("cumulative_rows_ever", 0)),
        )
        return man


# ----------------------------------------------------------------------
# Shard metadata extraction (no mtime, never opens a torn .npz)
# ----------------------------------------------------------------------


def _game_key_and_generation(npz_path: Path, sidecar: Mapping[str, Any] | None) -> tuple[int, int]:
    """Return ``(game_key, generation)`` for a shard.

    Key-derived epoch is authoritative; the sidecar ``epoch`` is a cross-check
    that WARNs on mismatch; never uses mtime. ``game_key`` is the integer after
    the first ``_`` in the stem (``game_<key>``); ``generation = game_key //
    1_000_000``.
    """
    stem = npz_path.stem  # e.g. "game_1000000"
    game_key: int
    try:
        game_key = int(stem.split("_", 1)[1])
    except (IndexError, ValueError):
        # Non-conforming name: fall back to the sidecar/dir, key unknown -> -1
        # so it sorts before real keys but still loads. This should not happen
        # for live shards (the selfplay writer names them game_<key>.npz).
        game_key = -1

    key_generation = game_key // _KEYS_PER_EPOCH if game_key >= 0 else None

    side_epoch: int | None = None
    if sidecar is not None and "epoch" in sidecar:
        try:
            side_epoch = int(sidecar["epoch"])
        except (TypeError, ValueError):
            side_epoch = None

    if key_generation is not None:
        if side_epoch is not None and side_epoch != key_generation:
            warnings.warn(
                f"shard {npz_path.name}: sidecar epoch {side_epoch} != key-derived "
                f"generation {key_generation}; trusting key-derived",
                RuntimeWarning,
                stacklevel=2,
            )
        return game_key, key_generation

    # Key-derivation failed (malformed name). Fall back to sidecar epoch, then
    # the parent dir "epoch_NNNNNN".
    if side_epoch is not None:
        return game_key, side_epoch
    parent = npz_path.parent.name
    if parent.startswith("epoch_"):
        try:
            return game_key, int(parent.split("_", 1)[1])
        except (IndexError, ValueError):
            pass
    return game_key, 0


def _rows_from_sidecar(sidecar: Mapping[str, Any]) -> int | None:
    """Row count from a parsed sidecar. The live writer emits ``"rows"``;
    ``num_rows``/``effective_rows`` are tolerated for cross-lineage robustness
    (the dense lineage uses those). Returns ``None`` if no usable key is present
    so the caller can fall back to the ``.npz``.
    """
    for key in ("rows", "num_rows", "effective_rows"):
        if key in sidecar:
            try:
                return int(sidecar[key])
            except (TypeError, ValueError):
                continue
    return None


def _rows_from_npz(npz_path: Path) -> int:
    """Fallback row count read straight from the shard's ``num_rows`` array
    (only used when the sidecar is absent — but the scan SKIPS sidecar-less
    shards, so this is reached only via an explicit re-read path). Reads lazily
    via ``np.load`` without exploding rows."""
    import numpy as np  # local import: keep module import light

    with np.load(npz_path) as data:
        if "num_rows" in data.files:
            return int(data["num_rows"])
        # Legacy/alt key seen in some restnet shards.
        if "effective_rows" in data.files:
            return int(data["effective_rows"])
    return 0


def _build_entry(npz_path: Path, samples_dir: Path) -> ShardEntry | None:
    """Build a :class:`ShardEntry` for one ``.npz``. Returns ``None`` if the
    shard lacks a sidecar (half-written by the live writer): such shards are
    SKIPPED, never opened, so the scan never reads a torn ``.npz``.
    """
    sidecar_path = npz_path.with_suffix(".json")
    if not sidecar_path.exists():
        # Half-written / sidecar not yet flushed -> skip this round (picked up
        # next scan once its sidecar lands).
        return None

    sidecar: Mapping[str, Any] | None
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if not isinstance(sidecar, Mapping):
            sidecar = None
    except (OSError, ValueError, json.JSONDecodeError):
        # Torn / unreadable sidecar mid-flush -> skip this round.
        return None

    rows: int | None = _rows_from_sidecar(sidecar) if sidecar is not None else None
    if rows is None:
        # Sidecar present but no row key -> fall back to the shard itself.
        try:
            rows = _rows_from_npz(npz_path)
        except (OSError, ValueError, KeyError):
            return None

    game_key, generation = _game_key_and_generation(npz_path, sidecar)
    rel_path = npz_path.relative_to(samples_dir).as_posix()
    return ShardEntry(
        rel_path=rel_path,
        rows=int(rows),
        generation=int(generation),
        game_key=int(game_key),
    )


# ----------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------


def _load_manifest(manifest_path: Path) -> BufferManifest | None:
    """Load + validate the persisted manifest. Returns ``None`` on any
    parse/version/structure failure so the caller rebuilds from the tree
    (self-healing). Never raises for a bad file."""
    if not manifest_path.exists():
        return None
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        return BufferManifest.from_dict(raw)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        warnings.warn(
            f"{manifest_path.name}: unreadable/incompatible manifest; rebuilding from tree",
            RuntimeWarning,
            stacklevel=2,
        )
        return None


def _atomic_write_manifest(manifest_path: Path, manifest: BufferManifest) -> None:
    """Write the manifest atomically: a uniquely-named tmp in the same dir,
    fsync-free, then ``os.replace`` (atomic on the same filesystem). A crash
    mid-write leaves the old manifest (or none) intact, never a torn file."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = manifest_path.with_name(f"{manifest_path.name}.{os.getpid()}.tmp")
    payload = json.dumps(manifest.to_dict(), indent=2)
    try:
        tmp_path.write_text(payload, encoding="utf-8")
        os.replace(tmp_path, manifest_path)
    finally:
        # Defensive: if replace failed, don't leave the tmp lingering.
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------


def _iter_shard_npzs(samples_dir: Path) -> Iterable[Path]:
    """All ``epoch_*/game_*.npz`` under ``samples_dir`` (the self-play layout).
    Sorted for deterministic discovery order."""
    return sorted(samples_dir.glob("epoch_*/game_*.npz"))


def scan_or_update_manifest(samples_dir: str | os.PathLike[str]) -> BufferManifest:
    """Load, incrementally update, persist, and return the buffer manifest.

    Steps:

    1. Load ``.buffer_manifest.json`` if present and valid; otherwise start a
       fresh manifest (full rebuild — self-healing on parse/version mismatch).
    2. Drop entries whose ``.npz`` file has vanished.
    3. Incrementally ``glob`` ``epoch_*/game_*.npz`` and add ONLY shards not
       already present. SKIP shards lacking a sidecar (half-written).
       Row count from the sidecar (``rows``/``num_rows``/``effective_rows``);
       fall back to the shard's ``num_rows`` array only if the sidecar carries
       no row key.
    4. Re-sort by ``(generation, game_key)``.
    5. ``total_rows`` = live sum over present entries; advance the monotone
       ``cumulative_rows_ever = max(prev, total_rows)`` (never decremented).
    6. Persist atomically (tmp + ``os.replace``).

    The returned manifest is always consistent with the on-disk tree at scan
    time. Reading is non-destructive (no file moves, no mtime hazard).
    """
    samples_dir = Path(samples_dir)
    manifest_path = samples_dir / MANIFEST_NAME

    manifest = _load_manifest(manifest_path)
    if manifest is None:
        manifest = BufferManifest()

    if not samples_dir.exists():
        # Nothing to scan; persist an empty (or carried-forward) manifest so the
        # monotone counter survives an empty intermediate state.
        manifest.entries = []
        manifest.recompute_totals()
        _atomic_write_manifest(manifest_path, manifest)
        return manifest

    # (2) Prune vanished entries; index the survivors by rel_path.
    present: dict[str, ShardEntry] = {}
    for entry in manifest.entries:
        npz_path = samples_dir / entry.rel_path
        if npz_path.exists():
            present[entry.rel_path] = entry
        # else: file vanished -> drop silently (eviction is allowed).

    # (3) Incrementally add new shards (not already entries; sidecar required).
    for npz_path in _iter_shard_npzs(samples_dir):
        rel_path = npz_path.relative_to(samples_dir).as_posix()
        if rel_path in present:
            continue
        new_entry = _build_entry(npz_path, samples_dir)
        if new_entry is not None:
            present[rel_path] = new_entry
        # else: sidecar-less / torn -> skip this round; retried next scan.

    manifest.entries = list(present.values())

    # (4) canonical ordering, (5) totals + monotone bump.
    manifest.resort()
    manifest.recompute_totals()
    manifest.version = MANIFEST_VERSION

    # (6) atomic persist.
    _atomic_write_manifest(manifest_path, manifest)
    return manifest
