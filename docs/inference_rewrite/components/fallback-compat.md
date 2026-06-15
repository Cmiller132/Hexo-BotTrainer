# fallback-compat

I have everything I need. My component is the rollback-safe fallback/compat layer: a central env-flag resolver, per-group attention-impl routing with automatic fallback for unsupported shapes, and the switch that keeps the live `sdpa` path the default. Here is my deliverable.

---

# COMPONENT: Fallback / Compat Path + Flag Contract (rollback-safety layer)

This is the layer that makes the entire rewrite safe to land "dark": every new path is reachable only by an env flag, every flag defaults to the **current deployed behaviour**, and any failure in a new path (unsupported shape, kernel not present, autotune/compile error) **silently and deterministically degrades to the existing `sdpa`+gated-compile path** that the harnesses already gate. Nothing here runs a GPU workload or writes to the live tree.

## 1. The flag contract (single source of truth)

All flags resolve through one new module so behaviour can't drift between callers (`inference.py`, `eval_arena.py`, `selfplay.py`, the scripts). Defaults reproduce today's deployed stack **exactly**.

| Env var | Type | Default | Meaning | Maps to |
|---|---|---|---|---|
| `HEXFIELD_ATTN_IMPL` | `sdpa` \| `hexflash` \| `flex` \| `materialized` | `sdpa` | Attention impl selected **only for large-S groups** (`pad_to > HEXFIELD_LARGE_NPAD`). Small-S always stays `sdpa`+compile (Layer C). | per-group `set_attention_impl` |
| `HEXFIELD_LARGE_NPAD` | int | value of `HEXFIELD_COMPILE_MAX_NPAD` (512) | Cutover: groups with `pad_to <=` this keep Layer C; above it use `HEXFIELD_ATTN_IMPL`. | `_resolve_group_impl` |
| `HEXFIELD_ATTN_FALLBACK` | impl name | `sdpa` | What an unsupported/failed large-S impl degrades to. | `_resolve_group_impl` |
| `HEXFIELD_ATTN_STRICT` | `0` \| `1` | `0` | `1` = raise instead of silently falling back (CI / parity runs). `0` = production rollback-safety (log once, degrade). | fallback wrapper |
| `HEXFIELD_PAYLOAD_ABI` | `1` \| `2` | `1` | Forces the wire path (v1 numpy loop vs v2 pinned). `1` keeps the deployed path regardless of what Rust offers. | `submit_payload` |
| `HEXFIELD_PIPELINE_DEPTH` | int >=1 | `1` | Rust submit/finish ring depth. `1` = today's strict serial path. | `search.rs` (B3) |
| `HEXFIELD_NO_COMPILE` | `0`\|`1` | `0` | **unchanged** — existing Layer C kill-switch. | `inference.py` |
| `HEXFIELD_COMPILE_MAX_NPAD` | int | `512` | **unchanged**. | `inference.py` |

**Master kill-switch (rollback in one variable):**

`HEXFIELD_REWRITE=off` (default `off`) forces *every* new path off in one move: ignores `HEXFIELD_ATTN_IMPL`/`HEXFIELD_PAYLOAD_ABI`/`HEXFIELD_PIPELINE_DEPTH` and pins `sdpa` / ABI 1 / depth 1. `HEXFIELD_REWRITE=on` honours the individual flags. This is the operator's "revert the whole experiment without touching the others" lever.

**Validity rule:** unknown/garbage values never crash the serve loop — they log once and fall to the safe default (`sdpa`, ABI 1, depth 1). An invalid flag must never be more dangerous than not setting it.

## 2. NEW module: `packages/hexfield/python/hexfield/serve_config.py`

```python
"""Central, rollback-safe resolution of the hexfield INFERENCE rewrite flags.

Every flag here defaults to the CURRENTLY DEPLOYED behaviour: attention impl
`sdpa` (with Layer-C gated compile for small-S), wire ABI 1 (the numpy pack
loop), pipeline depth 1 (strict serial submit/finish). The new fast paths
(hexflash/flex attention, ABI 2 pinned pack, depth-2 pipeline) are reachable
ONLY by explicitly opting in, and any failure degrades back to these defaults.

Single source of truth so inference.py, eval_arena.py, selfplay.py and the
bench/parity scripts can never disagree about what the env says. No torch /
model imports — pure stdlib so it is import-cheap and CPU-only safe.
"""

from __future__ import annotations

import os
import sys
import threading

# --- impl vocabulary ---------------------------------------------------------
# The two pre-existing, always-available impls. `materialized` is the math
# oracle (slow, fp32 scores) but is a *correct* universal fallback for any
# shape, so it is allowed as an explicit fallback target. `sdpa` is the
# deployed production impl and the default everywhere.
LEGACY_IMPLS = ("sdpa", "materialized")
# Impls introduced by the rewrite. Treated as "may not be supported on this
# build / this shape" -> guarded by the fallback wrapper (§3).
NEW_IMPLS = ("hexflash", "flex")
ALL_IMPLS = LEGACY_IMPLS + NEW_IMPLS

_DEFAULT_IMPL = "sdpa"
_DEFAULT_FALLBACK = "sdpa"
_DEFAULT_ABI = 1
_DEFAULT_DEPTH = 1

# Warn-once dedup so a degraded path doesn't spam the self-play log every flush.
_warned: set[str] = set()
_warn_lock = threading.Lock()


def _warn_once(key: str, msg: str) -> None:
    with _warn_lock:
        if key in _warned:
            return
        _warned.add(key)
    # stderr, not logging config-dependent: serve runs under Rust; keep it simple.
    print(f"[hexfield.serve_config] {msg}", file=sys.stderr, flush=True)


def _env(name: str) -> str | None:
    v = os.environ.get(name)
    return v.strip() if v is not None else None


def _env_bool(name: str, default: bool) -> bool:
    v = _env(name)
    if v is None or v == "":
        return default
    return v.lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    v = _env(name)
    if v is None or v == "":
        return default
    try:
        out = int(v)
    except ValueError:
        _warn_once(f"int:{name}", f"{name}={v!r} not an int; using {default}")
        return default
    if minimum is not None and out < minimum:
        _warn_once(f"min:{name}", f"{name}={out} < {minimum}; using {minimum}")
        return minimum
    return out


def rewrite_enabled() -> bool:
    """Master switch. Default OFF => the whole rewrite is dark; only the
    deployed sdpa/ABI1/depth1 stack runs. `HEXFIELD_REWRITE=on` honours the
    individual flags below."""
    return _env_bool("HEXFIELD_REWRITE", False)


def strict() -> bool:
    """CI/parity mode: raise on unsupported instead of silently degrading."""
    return _env_bool("HEXFIELD_ATTN_STRICT", False)


def _resolve_impl_name(name: str | None, *, what: str) -> str:
    if name is None or name == "":
        return _DEFAULT_IMPL if what == "impl" else _DEFAULT_FALLBACK
    if name not in ALL_IMPLS:
        _warn_once(
            f"impl:{what}:{name}",
            f"unknown attention impl {name!r}; using {_DEFAULT_IMPL!r}",
        )
        return _DEFAULT_IMPL
    return name


def large_attn_impl() -> str:
    """Impl requested for LARGE-S groups. `sdpa` unless the rewrite is enabled
    AND HEXFIELD_ATTN_IMPL asks for something else."""
    if not rewrite_enabled():
        return _DEFAULT_IMPL
    return _resolve_impl_name(_env("HEXFIELD_ATTN_IMPL"), what="impl")


def attn_fallback_impl() -> str:
    """Where a large-S impl degrades when unsupported/failed. Always a LEGACY
    impl (sdpa/materialized) so the fallback itself can never fail to exist."""
    fb = _resolve_impl_name(_env("HEXFIELD_ATTN_FALLBACK"), what="fallback")
    if fb in NEW_IMPLS:  # a new impl can't be a safe fallback target
        _warn_once(
            f"fb-new:{fb}",
            f"HEXFIELD_ATTN_FALLBACK={fb!r} is itself a rewrite impl; "
            f"using {_DEFAULT_FALLBACK!r}",
        )
        return _DEFAULT_FALLBACK
    return fb


def large_npad_cutover(compile_max_npad: int) -> int:
    """Groups with pad_to <= this stay on Layer C (sdpa+compile). Default ==
    the compile cutover so the two bands are contiguous (no gap, no overlap)."""
    return _env_int("HEXFIELD_LARGE_NPAD", compile_max_npad, minimum=0)


def payload_abi() -> int:
    """Wire ABI the evaluator will ACCEPT-and-prefer. ABI 1 (deployed numpy
    pack) unless the rewrite is on and ABI 2 is requested. Rust still decides
    what it SENDS; the evaluator handles both and this only biases the v2-vs-v1
    branch when both are available."""
    if not rewrite_enabled():
        return _DEFAULT_ABI
    abi = _env_int("HEXFIELD_PAYLOAD_ABI", _DEFAULT_ABI, minimum=1)
    if abi not in (1, 2):
        _warn_once(f"abi:{abi}", f"HEXFIELD_PAYLOAD_ABI={abi} unsupported; using 1")
        return _DEFAULT_ABI
    return abi


def pipeline_depth() -> int:
    """Rust submit/finish ring depth (read here only for telemetry/asserts;
    enforced Rust-side). Depth 1 == today's serial path."""
    if not rewrite_enabled():
        return _DEFAULT_DEPTH
    return _env_int("HEXFIELD_PIPELINE_DEPTH", _DEFAULT_DEPTH, minimum=1)


def shape_supported(impl: str, *, pad_to: int, group: int, num_tokens: int) -> bool:
    """Static guard: is `impl` willing to run THIS shape? Pure-arithmetic, no
    GPU. Used by the fallback wrapper to degrade BEFORE launching a kernel that
    would error. Legacy impls accept everything (they always have).

    The new impls advertise their envelope here so an out-of-envelope shape
    degrades cleanly instead of raising deep in a kernel. The envelopes are
    deliberately conservative; widen them only after the GPU pause validates a
    given regime. (S = num_tokens + pad_to is the attention sequence length.)
    """
    if impl in LEGACY_IMPLS:
        return True
    if pad_to <= 0 or group <= 0:
        return False
    s = num_tokens + pad_to
    if impl == "hexflash":
        # FA2-style kernel: needs a non-trivial sequence to amortise; the
        # bias-table column fits SRAM for any S; coords int32 safe for board
        # offsets. Upper bound is memory, not correctness — leave generous.
        return s >= num_tokens + 1
    if impl == "flex":
        # FlexAttention compiles a BlockMask per (B,S); require S large enough
        # that block tiling (128) is meaningful, else it's slower than sdpa.
        return s >= 128
    return False
```

## 3. NEW module: `packages/hexfield/python/hexfield/attn_fallback.py`

This is the runtime guard. It wraps the per-group impl selection so a new impl that is *requested* but *unavailable* (module import fails, Triton missing, shape out of envelope, or the kernel raises at first call) is caught **once per (impl, reason)** and replaced by the fallback for the rest of the process. In `strict` mode it re-raises (so parity harnesses surface the failure instead of silently passing on `sdpa`).

```python
"""Runtime support-probe + degrade wrapper for the rewrite attention impls.

Rollback-safety contract: requesting `hexflash`/`flex` must NEVER be able to
break the serve loop. If the impl's module can't import, the kernel isn't
present on this build, the shape is out of the impl's static envelope, or the
first real call raises, we degrade to the legacy fallback (sdpa) and remember
the failure so we don't retry-and-spam. HEXFIELD_ATTN_STRICT=1 turns the
degrade into a raise (CI/parity).

No GPU work happens at import. The only probe that could touch CUDA is the
lazy first-call try/except in `safe_set_impl`, which is exactly where a real
kernel failure would surface anyway.
"""

from __future__ import annotations

from . import serve_config as sc

# impls we have proven unusable this process (key -> reason). Once an impl is
# here, every later request for it degrades immediately with no retry.
_disabled: dict[str, str] = {}


def _module_available(impl: str) -> tuple[bool, str]:
    """Cheap, CPU-only import probe. Does NOT launch a kernel."""
    if impl in sc.LEGACY_IMPLS:
        return True, ""
    try:
        from . import hexflash  # noqa: F401  (the new kernel module, A1/A1-fb)
    except Exception as e:  # module missing on a build without it
        return False, f"hexflash module import failed: {e!r}"
    fn = "hexflash_attention" if impl == "hexflash" else "flex_attention_relpos"
    if not hasattr(hexflash, fn):
        return False, f"hexflash.{fn} not present"
    if impl == "flex":
        # FlexAttention needs a torch new enough to expose it; probe without
        # importing it into the hot path.
        try:
            import torch.nn.attention.flex_attention  # noqa: F401
        except Exception as e:
            return False, f"torch flex_attention unavailable: {e!r}"
    return True, ""


def resolve_group_impl(
    requested: str,
    *,
    pad_to: int,
    group: int,
    num_tokens: int,
    compile_max_npad: int,
) -> str:
    """Pure selection (no GPU): given the operator's requested LARGE-S impl and
    THIS group's shape, return the impl this group will actually use.

    - Small-S groups (pad_to <= cutover) always return `sdpa` (Layer C owns
      them; the rewrite never touches the small band).
    - Legacy impls pass through.
    - A new impl that is process-disabled, module-unavailable, or out of its
      static envelope degrades to the configured fallback.
    """
    cutover = sc.large_npad_cutover(compile_max_npad)
    if pad_to <= cutover or requested in sc.LEGACY_IMPLS:
        return "sdpa" if pad_to <= cutover else requested

    fallback = sc.attn_fallback_impl()

    if requested in _disabled:
        return _fail(requested, _disabled[requested], fallback)

    ok, why = _module_available(requested)
    if not ok:
        _disabled[requested] = why
        return _fail(requested, why, fallback)

    if not sc.shape_supported(
        requested, pad_to=pad_to, group=group, num_tokens=num_tokens
    ):
        # shape-specific: degrade THIS group but do NOT disable the impl (a
        # different group's shape may be in-envelope).
        return _fail(
            requested,
            f"shape pad_to={pad_to} group={group} out of {requested} envelope",
            fallback,
            disable=False,
        )
    return requested


def mark_runtime_failure(impl: str, exc: BaseException) -> str:
    """Called by the evaluator if a NEW impl raises during the actual forward.
    Disables it for the rest of the process and returns the fallback impl the
    caller should retry this group with. In strict mode, re-raises."""
    why = f"runtime error in {impl}: {exc!r}"
    _disabled[impl] = why
    return _fail(impl, why, sc.attn_fallback_impl())


def _fail(impl: str, why: str, fallback: str, *, disable: bool = True) -> str:
    if sc.strict():
        raise RuntimeError(
            f"hexfield attn impl {impl!r} unusable and HEXFIELD_ATTN_STRICT=1: {why}"
        )
    sc._warn_once(
        f"degrade:{impl}:{why[:40]}",
        f"attention impl {impl!r} -> {fallback!r} (rollback-safe): {why}",
    )
    return fallback


def reset_for_tests() -> None:
    """Clear the process-level disable cache (parity harness setup)."""
    _disabled.clear()
    sc._warned.clear()
```

## 4. Patch to `inference.py` — wire the compat layer into `_forward_group` and `__init__`

The change is purely additive: a new per-group impl resolution, a `set_attention_impl` flip around the large-S forward, and a try/except that catches a new-impl runtime error and **re-runs the same group on the fallback** so a flush never fails. Small-S and the v1 numpy pack are byte-unchanged.

### 4a. `__init__` — read flags through the resolver (after the existing compile block, around `inference.py:104`)

```diff
@@ class HexfieldEvaluator.__init__
         self._compile_max_npad = int(os.environ.get("HEXFIELD_COMPILE_MAX_NPAD", "512"))
+        # Rewrite compat layer (rollback-safe). Defaults reproduce the deployed
+        # stack: large-S impl == sdpa, ABI 1, no per-group impl switching. The
+        # new fast paths are reachable only via HEXFIELD_REWRITE=on + the
+        # individual flags, and degrade back to sdpa on any failure.
+        from . import serve_config as _sc
+        self._sc = _sc
+        self._large_attn_impl = _sc.large_attn_impl()      # "sdpa" by default
+        self._uses_new_attn = (
+            self.device.type == "cuda"
+            and self._large_attn_impl in _sc.NEW_IMPLS
+        )
+        # Remember the model's current (training/serve) impl so we can restore
+        # it after each large-S group — never leave the model in a non-sdpa
+        # state for the next caller / the small-S path.
+        self._base_attn_impl = "sdpa"
+        self.model.set_attention_impl(self._base_attn_impl)
         if self._use_compile:
```

### 4b. `_forward_group` — resolve impl, flip it only for large-S, catch + degrade

Replace the forward-call region (`inference.py:256-274`) with:

```diff
@@ def _forward_group
-        use_compiled = (
-            self._use_compile
-            and self._compiled_fpv is not self._raw_fpv
-            and pad_to <= self._compile_max_npad
-        )
-        fpv = self._compiled_fpv if use_compiled else self._raw_fpv
+        # Resolve which attention impl THIS group uses (pure, no GPU). Small-S
+        # always -> sdpa (Layer C); large-S -> the requested new impl unless it
+        # is unavailable/out-of-envelope, in which case it degrades to sdpa.
+        if self._uses_new_attn:
+            from . import attn_fallback as _afb
+            group_impl = _afb.resolve_group_impl(
+                self._large_attn_impl,
+                pad_to=pad_to, group=g, num_tokens=NUM_TOKENS,
+                compile_max_npad=self._compile_max_npad,
+            )
+        else:
+            group_impl = "sdpa"
+        use_new = group_impl in self._sc.NEW_IMPLS
+        # Compile is for the small-S sdpa band only; a new-impl group never
+        # goes through the compiled graph (its kernel owns the attention core).
+        use_compiled = (
+            not use_new
+            and self._use_compile
+            and self._compiled_fpv is not self._raw_fpv
+            and pad_to <= self._compile_max_npad
+        )
+        fpv = self._compiled_fpv if use_compiled else self._raw_fpv
         if use_compiled:
             for t in (d_feats, d_nbr, d_mask, d_coords):
                 if g > 1:
                     torch._dynamo.mark_dynamic(t, 0)
                 torch._dynamo.mark_static(t, 1)
-        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_fp16):
-            out = fpv(
-                d_feats,
-                d_nbr,
-                d_mask,
-                d_coords,
-                request_moves_left=request_ml,
-            )
+        out = self._run_forward(
+            fpv, group_impl, d_feats, d_nbr, d_mask, d_coords,
+            request_ml, device, use_fp16,
+        )
```

### 4c. New helper on the evaluator — the degrade-and-retry guard (math identical to the existing call)

```python
    def _run_forward(
        self, fpv, group_impl, d_feats, d_nbr, d_mask, d_coords,
        request_ml, device, use_fp16,
    ):
        """Run forward_policy_value under the group's attention impl. If a NEW
        impl raises at runtime, disable it process-wide and RE-RUN this group
        on the fallback (sdpa) so a flush is never dropped. The math of the
        sdpa path is exactly the deployed path (autocast fp16, same args).

        Coords dtype contract (C2): hexflash/flex want int32; sdpa wants the
        deployed int64. We only ever down-cast a *copy* for the new path, so
        the sdpa fallback always re-runs with the original int64 coords.
        """
        def _call(impl, coords):
            self.model.set_attention_impl(impl)
            try:
                with torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=use_fp16
                ):
                    return fpv(
                        d_feats, d_nbr, d_mask, coords,
                        request_moves_left=request_ml,
                    )
            finally:
                # Never leave the model in a non-base impl for the next group.
                self.model.set_attention_impl(self._base_attn_impl)

        if group_impl in self._sc.NEW_IMPLS:
            coords_new = d_coords.to(torch.int32)
            try:
                return _call(group_impl, coords_new)
            except Exception as exc:  # noqa: BLE001 — rollback-safety boundary
                from . import attn_fallback as _afb
                fb = _afb.mark_runtime_failure(group_impl, exc)  # raises if strict
                # Fall back: the compiled small-S graph is irrelevant here
                # (this is a large-S group), so use the raw fpv on sdpa.
                return _call(fb, d_coords)
        return _call(group_impl, d_coords)
```

Notes on why this is byte-safe for the default path:
- When `HEXFIELD_REWRITE` is unset/`off`, `large_attn_impl()` returns `"sdpa"`, `_uses_new_attn` is `False`, `group_impl` is always `"sdpa"`, and `_run_forward` reduces to `_call("sdpa", d_coords)` under the *same* `torch.autocast` as the original code. The only behavioural delta vs. today is two redundant `set_attention_impl("sdpa")` calls (the model is already `sdpa`), which are no-ops on the attribute. **maxabsdiff == 0.0** by construction — this satisfies the existing ASYNC-PARITY gate in `_hexfield_compile_overlap_test.py:130` with no threshold change.
- `set_attention_impl` is restored in a `finally`, so a large-S group can never leak a non-`sdpa` impl into the following small-S compiled group (which would otherwise silently invalidate the compiled graph's traced impl).

## 5. v2 ABI compat in `submit_payload` (rollback-safe accept-both)

`submit_payload` currently hard-rejects anything but ABI 1 (`inference.py:134-135`). The compat rule: **accept both**, but only *use* v2 when the operator has opted in (so a Rust binary that starts sending ABI 2 can't change serve behaviour until the flag flips). v1 stays the default and the universal fallback.

```diff
@@ def submit_payload
-        if int(payload["abi"]) != 1:
-            raise ValueError(f"unsupported hexfield ABI {payload['abi']}")
+        wire_abi = int(payload["abi"])
+        if wire_abi not in (1, 2):
+            raise ValueError(f"unsupported hexfield ABI {wire_abi}")
+        # Compat: even if Rust sends ABI 2 buffers, only consume the v2 pinned
+        # pack when the operator opted in (HEXFIELD_REWRITE=on +
+        # HEXFIELD_PAYLOAD_ABI=2). Otherwise fall back to the v1 numpy pack
+        # using the v1 fields the payload also carries. This keeps the deployed
+        # path the default and lets the wire change land before the serve flips.
+        use_v2 = (wire_abi == 2 and self._sc.payload_abi() == 2)
+        if use_v2 and not _v2_fields_present(payload):
+            self._sc._warn_once(
+                "abi2-missing",
+                "ABI 2 requested but v2 buffers absent; using v1 pack",
+            )
+            use_v2 = False
+        self._use_v2_pack = use_v2  # consumed by _forward_group (B1's branch)
```

with a module-level helper (the exact v2 field names are B2/Implementer-5's contract; this only checks presence so the *absence* of any one degrades to v1):

```python
def _v2_fields_present(payload: dict) -> bool:
    # C3 contract fields. If B2 renames any, update this one predicate only.
    return all(
        k in payload
        for k in ("v2_node_feats", "v2_coords", "v2_gather_idx",
                  "v2_cu_seqlens", "v2_legal_counts")
    )
```

Until B1 (Implementer 4) wires the actual v2 consumption, `self._use_v2_pack` is simply always-False-effectively (no v2 fields present → degrade), so this patch is inert today and becomes live only when both the Rust v2 pack and B1's reader land. That is the intended staging.

## 6. Tests (CPU-only, no GPU) — `tests/test_hexfield_serve_compat.py` (NEW)

These run today, on CPU, and lock the rollback contract statically:

```python
import os
import importlib
import pytest

from hexfield import serve_config as sc
from hexfield import attn_fallback as afb


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("HEXFIELD_REWRITE", "HEXFIELD_ATTN_IMPL", "HEXFIELD_ATTN_FALLBACK",
              "HEXFIELD_ATTN_STRICT", "HEXFIELD_LARGE_NPAD", "HEXFIELD_PAYLOAD_ABI",
              "HEXFIELD_PIPELINE_DEPTH"):
        monkeypatch.delenv(k, raising=False)
    afb.reset_for_tests()


def test_defaults_are_deployed_stack():
    assert sc.rewrite_enabled() is False
    assert sc.large_attn_impl() == "sdpa"
    assert sc.payload_abi() == 1
    assert sc.pipeline_depth() == 1


def test_master_switch_off_pins_everything(monkeypatch):
    monkeypatch.setenv("HEXFIELD_REWRITE", "off")
    monkeypatch.setenv("HEXFIELD_ATTN_IMPL", "hexflash")
    monkeypatch.setenv("HEXFIELD_PAYLOAD_ABI", "2")
    monkeypatch.setenv("HEXFIELD_PIPELINE_DEPTH", "4")
    assert sc.large_attn_impl() == "sdpa"   # ignored while master is off
    assert sc.payload_abi() == 1
    assert sc.pipeline_depth() == 1


def test_garbage_impl_degrades_not_crashes(monkeypatch):
    monkeypatch.setenv("HEXFIELD_REWRITE", "on")
    monkeypatch.setenv("HEXFIELD_ATTN_IMPL", "totally-bogus")
    assert sc.large_attn_impl() == "sdpa"


def test_small_s_group_never_uses_new_impl(monkeypatch):
    monkeypatch.setenv("HEXFIELD_REWRITE", "on")
    monkeypatch.setenv("HEXFIELD_ATTN_IMPL", "hexflash")
    # pad_to below cutover -> sdpa regardless
    got = afb.resolve_group_impl("hexflash", pad_to=256, group=8,
                                 num_tokens=8, compile_max_npad=512)
    assert got == "sdpa"


def test_large_s_missing_module_degrades(monkeypatch):
    monkeypatch.setenv("HEXFIELD_REWRITE", "on")
    # hexflash module not present on this CPU build -> degrade to sdpa
    got = afb.resolve_group_impl("hexflash", pad_to=2048, group=2,
                                 num_tokens=8, compile_max_npad=512)
    assert got == "sdpa"


def test_strict_mode_raises(monkeypatch):
    monkeypatch.setenv("HEXFIELD_REWRITE", "on")
    monkeypatch.setenv("HEXFIELD_ATTN_STRICT", "1")
    with pytest.raises(RuntimeError):
        afb.resolve_group_impl("hexflash", pad_to=2048, group=2,
                               num_tokens=8, compile_max_npad=512)


def test_fallback_cannot_be_a_new_impl(monkeypatch):
    monkeypatch.setenv("HEXFIELD_ATTN_FALLBACK", "flex")
    assert sc.attn_fallback_impl() == "sdpa"


def test_runtime_failure_disables_impl(monkeypatch):
    monkeypatch.setenv("HEXFIELD_REWRITE", "on")
    fb = afb.mark_runtime_failure("hexflash", RuntimeError("kernel boom"))
    assert fb == "sdpa"
    # now disabled process-wide -> even an in-envelope large group degrades
    got = afb.resolve_group_impl("hexflash", pad_to=2048, group=2,
                                 num_tokens=8, compile_max_npad=512)
    assert got == "sdpa"
```

## 7. Parity assertions this component owns

- **Default-path byte parity (today, CPU + the GPU pause):** with no rewrite flags set, `_run_forward` is the deployed call. Gate: `_hexfield_compile_overlap_test.py` COMPILE-PARITY (`TOL=3e-3`, `line 118`) and ASYNC-PARITY (`maxabsdiff==0.0`, `line 130`) **must be unchanged**. The two redundant `set_attention_impl("sdpa")` calls are no-ops → diff exactly 0.0. This is statically certain.
- **Degrade parity (GPU pause):** force `HEXFIELD_REWRITE=on HEXFIELD_ATTN_IMPL=hexflash` on a build *without* the kernel → every large-S group resolves to `sdpa` → the run is byte-identical to the default. Assert `maxabsdiff==0.0` vs the default run. This proves the rollback path equals the baseline.
- **Strict-mode CI gate (today):** the test above (`test_strict_mode_raises`) ensures parity harnesses can set `HEXFIELD_ATTN_STRICT=1` so a "new impl silently fell back to sdpa and the parity passed for the wrong reason" can never produce a false green.
- **Restore invariant (GPU pause):** after a mixed flush (large-S new-impl group followed by a small-S compiled group), assert the small-S group's output equals the all-sdpa baseline — proves the `finally: set_attention_impl(base)` prevents impl leakage into the compiled graph.

## 8. Files delivered by this component

- `packages/hexfield/python/hexfield/serve_config.py` — NEW (flag resolver, single source of truth).
- `packages/hexfield/python/hexfield/attn_fallback.py` — NEW (support-probe + degrade-and-disable wrapper).
- `packages/hexfield/python/hexfield/inference.py` — diffs in §4 (`__init__` flag read; `_forward_group` impl routing; `_run_forward` degrade-and-retry helper) and §5 (accept-both ABI with opt-in v2).
- `tests/test_hexfield_serve_compat.py` — NEW (CPU-only contract tests, runnable now).

## 9. Honesty / validation boundary

- **Statically certain now (no GPU):** flag resolution, default == deployed behaviour, garbage-value degradation, small-S never touching a new impl, strict-mode raising, fallback never being a new impl, the no-op-equivalence of the default forward path. All covered by §6 CPU tests.
- **Needs the GPU pause:** that a *runtime* hexflash/flex failure is actually caught by the `except Exception` boundary and the sdpa re-run produces correct output (the math is the deployed path, but "the exception is raised at the Python boundary and not, e.g., as an async CUDA error surfaced on a later sync" must be observed); and that `set_attention_impl` flip/restore around the large-S forward composes correctly with the compiled small-S graph. Both are listed in §7 as GPU-pause gates.

This layer touches no learned weights, no search behaviour, no numeric path on the default flags. Its sole job is to guarantee the rewrite can be shipped dark and reverted with a single env var (`HEXFIELD_REWRITE=off`).