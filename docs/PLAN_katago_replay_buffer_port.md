# PLAN — Port hexfield's replay buffer to the exact KataGo / dense_cnn_restnet design (multithreaded, no corners cut)

Status: AUTHORITATIVE IMPLEMENTATION PLAN. Read-only synthesis; no source was modified to produce it.
Date: 2026-06-17. Target run: a FRESH run (`hexfield_main_3`), NOT a hot edit to the live `hexfield_main_2`.
Reference implementation (the parity oracle): `packages/dense_cnn_restnet/python/dense_cnn_restnet/` (a KataGo-style port).

All file:line citations are verified against current source. Where a study cited a docstring, the verdict here follows the *code*, not the docstring.

---

## 1. Goal & non-goals

### Goal
Replace hexfield's v1 replay window (mtime-ordered glob + hard `shuffle_keep_target_rows` cutoff + eager per-row decode into `list[HexfieldSampleData]` + serial per-row Python `expand_sample`) with the **exact KataGo / dense_cnn_restnet mechanism**:

1. **Power-law taper window** (`compute_katago_window_rows`) keyed on cumulative rows, clamped at `min_rows`.
2. **Recent-window cutoff** (`_select_recent_window`): newest→oldest whole-shard accumulation until `used_rows ≥ desired_rows`.
3. **keep_prob uniform Bernoulli subsample** toward `keep_target_rows`.
4. **Train-bucket reuse governor** (`+max_train_bucket_per_new_data` accrual, `max_train_bucket_size` cap, debit-by-effective-rows, `train_bucket_limited` skip).
5. **Single pass, no within-epoch repeat**, capped at `train_samples_per_epoch` via whole-shard overshoot-skip selection.
6. **md5 path-keyed train/val split** (real, not a stub).
7. **Persisted train-state** (bucket + `data_files_used`) across checkpoint resume.
8. **Packed-columnar in-RAM window** (kills the ~1–2 GB `HexfieldSampleData` heap and the per-epoch re-glob).
9. **Multithreaded expansion**: extend the existing Rust+rayon GIL-free kernel for train-read, with a serial-Python parity oracle and a spawn-ProcessPool fallback.

"Almost exactly like KataGo where possible, don't cut corners" — the window mathematics, the governor, the split, and the determinism rules are ported faithfully. The few divergences are explicitly enumerated in §8 and each is justified by a *structural* hexfield difference (per-row variable expansion, no fixed batch size, single-game shards), not by expedience.

### Non-goals
- Do **not** disturb the live `hexfield_main_2` run. All work is READ-ONLY against its `samples/` tree; the feature ships as a fresh run.
- Do **not** change the on-disk compact shard schema (`hexfield_compact_v1`, `shards.py:147-174`) or the self-play write path (`selfplay.py:285-289`).
- Do **not** change the expansion math (`expand_sample`, `build_support`, `transform_facts`, `_legal_slot`), the loss/optimizer/AMP block (`trainer.py:124-160`), or the micro-bucket VRAM batching (`pair_budget_microbuckets`).
- Do **not** re-shard the window to disk (dense's `shuffleddata/<gen>/data*.npz` materialization) — see §8 (justified divergence) and §3.
- Do **not** change the framework dispatch contract (`select_training_samples` / `train_passes` signatures, `uses_shared_sample_store=False`, the `sample_symmetries`-ignored / opaque-window rule).

---

## 2. Current vs target architecture

| Concern | Current hexfield (v1) | Target (KataGo / dense port) |
|---|---|---|
| Shard ordering | `st_mtime` glob, reverse (`trainer.py:43-48`) — fragile (resume/cp/touch perturb it) | Persisted manifest, sorted by `(generation, game_key)` — mtime-free |
| Window size | Hard `shuffle_keep_target_rows` row cutoff (`trainer.py:68,71`) | Power-law taper `compute_katago_window_rows`, clamp `max(_, min_rows)` |
| Subsample | None | keep_prob Bernoulli toward `keep_target_rows` |
| In-RAM rep | `list[HexfieldSampleData]` (tuples of boxed scalars; ~1–2 GB @500k) | `PackedWindow` (concatenated columnar numpy + global CSR offsets) |
| Per-epoch IO | re-glob + re-decode EVERY shard EVERY epoch | manifest avoids re-glob/sort; window decode retained (see §8/M6) |
| Rows trained/epoch | the WHOLE window, `passes` times (`trainer.py:105-108`) | single pass, capped at `train_samples_per_epoch`, overshoot-skip selection |
| Reuse control | none | train-bucket governor (`+8×/new-row`, 500k cap, `train_bucket_limited` skip) + `no_repeat_files`-decision (§8/M2) |
| Train/val split | none | md5 path-keyed per-file split (`validation_fraction`) |
| Expansion | serial pure-Python per-row `expand_sample` (`trainer.py:115-121`) | Rust+rayon GIL-free shard expand; serial-Python oracle; spawn-pool fallback |
| Determinism | `random.Random`, `rng.shuffle` (`:107`) + in-loop `rng.randrange(12)` (`:117`) | `np.random.default_rng((run_seed)+epoch)`; ALL randomness pre-drawn main-thread |
| State persistence | none (checkpoint = `{meta,model,optimizer}` only, `checkpoints.py:16-20`) | `train_state` persisted in checkpoint `meta`, restored on resume only |

---

## 3. The exact mechanism being ported

All references are `dense_cnn_restnet/python/dense_cnn_restnet/`. The hexfield port mirrors these verbatim except where §8 marks a justified divergence.

### 3.1 Power-law taper window — `replay.py:620-632`
Ported **verbatim** (same float order, same `int()` truncation) into `hexfield/window.py::compute_katago_window_rows`:
```
offset      = taper_window_scale if taper_window_scale is not None else min_rows
power_law_x = usable_rows - min_rows + offset
unscaled    = power_law_x**e - offset**e
scaled      = unscaled / (e * offset**(e-1))
window      = int(scaled * expand_window_per_row + min_rows)        # e = taper_window_exponent
```
Caller clamps `desired_rows = max(window, min_rows)` (mirrors `replay.py:396`). As `usable_rows→min_rows`, `window→min_rows`. `e<1` ⇒ sublinear taper.

**Defaults (RECONCILED — see §7 and §8/S1):** hexfield-tuned, NOT dense's literal numbers, because hexfield produces ~7k Full rows/epoch (~28 Full rows/game; `selfplay.py:283` PCR filter). dense's `min_rows=100_000` would stall training ~14 epochs at that rate. The dimensionless knobs (`exponent=0.65`, `expand=0.4`) are kept unchanged.

| Knob | hexfield default | dense default | Rationale |
|---|---|---|---|
| `shuffle_taper_window_exponent` | 0.65 | 0.65 | dimensionless KataGo constant — unchanged |
| `shuffle_expand_window_per_row` | 0.4 | 0.4 | dimensionless derivative-at-floor — unchanged |
| `shuffle_taper_window_scale` (offset) | 20_000.0 | 50_000.0 | smaller row stream ⇒ taper turns on sooner |
| `shuffle_min_rows` | 20_000 | 100_000 | 100k stalls ~14 epochs at 7k rows/epoch; 20k ≈ 3-epoch bootstrap |
| `shuffle_keep_target_rows` | 300_000 | 600_000 | hexfield package target; raised from current 32k live value |

### 3.2 Recent-window cutoff — `replay.py:681-690`
`select_recent_window(entries, desired_rows)`: walk manifest entries newest→oldest (`reversed`, since entries are generation-ascending), accumulate WHOLE shards until `used_rows ≥ desired_rows`, then stop; re-sort ascending. Whole-shard granularity — overshoots `desired_rows` by < one shard. `window_start = max(0, total_rows - used_rows)` (mirrors `replay.py:423`), recorded for the governor and diagnostics.

### 3.3 keep_prob — `replay.py:404` + `771-778`
```
keep_prob = min(float(keep_target_rows), float(used_rows)) / float(used_rows)
```
`1.0` when `used_rows ≤ keep_target_rows`; else the down-sample ratio. Applied per-row as independent `Bernoulli(keep_prob)` via the **single shared** `np.random.default_rng(seed)`, consumed in deterministic `(generation, game_key)` shard/row order (mirrors `replay.py:777`). Realized kept count is stochastic.

### 3.4 Batch-aligned output — `replay.py:782-789` (ADAPTED, see §8/M3)
dense permutes, then `aligned = (len(rows)//batch_size)*batch_size; rows = rows[:aligned]` (drops the partial final batch), then chunks to a batch-multiple and writes `data*.npz`. hexfield has **no fixed batch size** — `train_passes` micro-buckets by a pad-quantized VRAM budget (`pair_budget_microbuckets`, `trainer.py:126`), so there is nothing to "align to" and no disk re-shard. The faithful equivalent that IS ported: **single pass, no within-epoch repeat, capped at exactly `effective_rows`** — the permuted survivor index is **truncated to `effective_rows`** before the micro-bucket loop (this is the load-bearing fidelity point, not the disk layout).

### 3.5 Train-bucket reuse governor — `replay.py:481-499` + `trainer.py:199-218`
Accrual (`_update_train_bucket`, called with cumulative `total_rows` and `window_start`):
```
cap = max(max_train_bucket_size, train_samples_per_epoch)
if total_rows > level_at_row:
    bucket = min(cap, bucket + (total_rows - level_at_row) * max_train_bucket_per_new_data)
    level_at_row = total_rows
elif total_rows < level_at_row:                  # window regenerated/shrank
    level_at_row = total_rows; steps_since_reload = 0; bucket = min(bucket, cap)
```
Each fresh self-play row credits the bucket by `max_train_bucket_per_new_data` (8.0). Consumption (`select_training_samples`):
```
effective_rows = min(requested_rows, selected_rows)
if bucket + 1e-9 < effective_rows: return {"status": "train_bucket_limited", ...}   # the throttle
bucket = max(0.0, bucket - effective_rows); steps_since_reload += 1                  # debit at SELECTION time
```
The bucket is debited by `effective_rows` at selection, before training runs — a later short pass does not refund (dense semantics, `trainer.py:217`). See §8/M4 for the hexfield off-legal-skip interaction.

**CRITICAL (M2):** `total_rows` MUST be a **monotone cumulative counter**, never decremented. dense gets this free because it never deletes selfplay shards; hexfield's manifest prunes vanished entries (§5), so the manifest must track a separate monotone `cumulative_rows_ever` (max of itself and the live sum, persisted) and feed *that* to the governor, while window selection uses the live total. Feeding a non-monotone count would spuriously trip the `elif total_rows < level_at_row` branch and zero the reuse counter.

### 3.6 md5 split — `replay.py:693-709` + `_md5_path_fraction` `:908-910`
```
def _md5_path_fraction(value): return int("0x"+hashlib.md5(value.encode()).hexdigest()[:13],16)/float(2**52)
def _split_by_md5(selected, *, validation_fraction):
    if validation_fraction <= 0.0: return list(selected), []
    train_upper = 1.0 - validation_fraction
    train = [e for e in selected if _md5_path_fraction(str(e.rel_path)) <  train_upper]
    val   = [e for e in selected if _md5_path_fraction(str(e.rel_path)) >= train_upper]
```
Per-file, stable across epochs (keyed on path). Default `validation_fraction=0.0` ⇒ all-train. **This is implemented for real (M5), not stubbed** — `select_training_samples` calls `_split_by_md5(selected, validation_fraction=cfg.validation_fraction)` between `select_recent_window` and `build_window_split`; a `validation_fraction>0` builds a val `PackedWindow` written to the diagnostics for the eval head. The `_md5_path_fraction` helper also serves as the optional `md5_lbound`/`md5_ubound` set filter (`replay.py:383-384`), ported but defaulting to `[0,1)` no-op.

### 3.7 Single-pass, no-repeat selection — `trainer.py:604-631`
`_select_files_for_rows(files, requested_rows, rng)`: shuffle candidate shards (with row counts from the manifest), greedily accumulate; a shard that would overshoot is skipped with probability `overshoot/row_count` and deferred; deferred shards are added back if short. Unbiasedly lands near (not far past) `requested_rows`. Per-epoch RNG `np.random.default_rng(seed + epoch*65537)` (mirrors `trainer.py:180`). `no_repeat_files` filtering is applied first (see §8/M2 for the hexfield decision).

### 3.8 Determinism / seeding — `trainer.py:134`, `replay.py:422`
Seed = `(ctx.config.run.seed or 0) + epoch`. A single `np.random.default_rng(seed)` per epoch drives keep_prob + the permutation; a separate `np.random.default_rng(seed + epoch*65537)` drives the file selection; the per-row D6 vector is drawn from a seed folding `(run_seed, epoch)` (mirror dense `_aug_seed`, `trainer.py:545-559`). **ALL randomness is drawn on the main thread BEFORE any parallel dispatch** and passed positionally; workers are pure functions. md5 splits/filters are seed-independent (hash of path).

---

## 4. Multithreading architecture

### 4.1 Per-stage concurrency decision

| Stage | Cost class | Concurrency | Justification (GIL) |
|---|---|---|---|
| Manifest scan / row counts | IO (sidecar reads) | single-thread | trivial; sidecar JSON only in steady state |
| keep_prob mask | numpy vectorized | single-thread | `rng.random(n) < p` releases GIL internally; no benefit to threading |
| permute | numpy vectorized | single-thread | `rng.permutation` is one call; main-thread per the determinism rule |
| D6 + Support BFS + features + policy projection (`expand_sample`) | **CPU-bound, pure-Python** (DOMINANT) | **Rust + rayon + `py.detach`** | the ONLY stage worth parallelizing; pure-Python ⇒ ThreadPool gives ZERO speedup (GIL); the kernel already exists GIL-free in Rust |
| collate / loss / optimizer step | GPU + small host | single-thread (main) | stays on the main process exactly as today (`trainer.py:124-160`) |

**ThreadPool is rejected everywhere** for the expand/decode loop: it is pure-Python CPU-bound and the GIL serializes it (zero parallel speedup). Threads would only overlap the GIL-releasing `np.load`, which the Rust path subsumes.

### 4.2 Primary: extend the Rust crate (do NOT copy dense's spawn ProcessPool)
The dominant work already exists rayon-parallel and GIL-free in Rust: `payload.rs:52-106` `featurize_and_sort` runs `build_support` (depth-9 BFS) + `build_features` across `states.par_iter()` under `py.detach`, order-preserving via `collect`. `serve_pack.rs:288` runs the parallel pad under `py.detach`. D6 cardinality is verified **12** (`geometry.py:63-81`), matching the current `rng.randrange(12)`.

New Rust entry point `replay_expand.rs::expand_shard_train(columns, d6, horizons, support_radius, tolerate_off_legal)`:
- Input: one shard's packed columns (the npz byte buffers / numpy arrays), a pre-drawn per-row `d6: i32[n]` vector, the config horizons `(2,6,16)`, the support radius, and the tolerate-off-legal flag.
- Per row, under rayon `par_iter`: apply the D6 symmetry to all stored coords (the `transform_facts` twin), `build_support` + `build_features`, project self/opp policy onto legal slots (the `_legal_slot` twin, `samples.py:199-224`).
- Returns: stacked numpy/byte buffers (feats/coords/dist/nbr/policy/opp/stvalue/...) **plus a per-row `valid: bool[n]` validity mask** (off-legal rows marked invalid, NOT silently dropped — see §4.5). Zero-copy back to Python like `serve_pack.rs:290-299` (`F16Buf`/`I32Buf`). No pickling, no per-worker torch RSS, no spawn cost; threads share the loaded shard buffers zero-copy.

Why Rust over dense's spawn ProcessPool on THIS host: spawn is mandatory on Windows/WSL (no fork). dense's pool pays full torch re-import (~250–400 MB RSS/worker, `dense/trainer.py:577`) and pickles the dense expanded arrays back every shard. hexfield's expanded output is ragged per-row support graphs — *more* expensive and fragile to pickle than dense's fixed N×C×41×41 planes. Rust+rayon sidesteps all of it.

### 4.3 Fallback ladder (so the port is not blocked on the Rust kernel — M3)
The port ships in phases (§10) so the memory + window + governor wins land FIRST with the **existing serial Python expand**, with the parallel path added behind a flag:
1. **Serial Python** (`HEXFIELD_EXPAND=serial`): current `expand_sample` loop. The parity oracle and the safe default for Phase 1.
2. **Spawn ProcessPool** (`HEXFIELD_EXPAND=pool`): dense's exact pattern — persistent `ProcessPoolExecutor(mp_context=spawn)`, unit = one whole shard via a picklable `expand_shard_to_arrays(path, syms, horizons)`, `workers+2` inflight, `wait(FIRST_COMPLETED)`, cancel-on-target, `W=min(8, max(1, cpu//4))`, `HEXFIELD_EXPAND_WORKERS` override, serial below `_PARALLEL_MIN_ROWS=2048`. Strictly inferior on this host but proven and low-risk.
3. **Rust + rayon** (`HEXFIELD_EXPAND=rust`, eventual default): §4.2. The production path once parity gates pass.

### 4.4 Deterministic per-worker seeding scheme
- Main thread draws, from `np.random.default_rng((run_seed)+epoch)`: the keep_prob mask (per shard, in `(generation, game_key)` order) and the window permutation.
- Main thread draws the per-row `d6: i32[n]` vector per shard, in deterministic file order, from `np.random.default_rng(_aug_seed(run_seed, epoch))` (mirror `dense/trainer.py:284,295-300`).
- These vectors are passed **positionally** into the Rust/process workers. No `rng` call ever happens inside a worker.
- `par_iter().collect()` preserves input order (relied on at `payload.rs:55-57`), so output is byte-identical regardless of worker scheduling and regardless of `workers=1` vs `workers=N`.
- **Two MUST-FIX determinism violations in the current code** (both move to pre-drawn vectors): `rng.shuffle(order)` (`trainer.py:107`) and in-loop `rng.randrange(12)` (`trainer.py:117`). Note: switching `random.Random`→`np.random.default_rng` makes the new pipeline NOT bit-identical to the current run by construction (different RNG streams). This is acceptable (D6 augmentation is exchangeable) but means the A/B test (§9) is **statistical, not bitwise**.

### 4.5 Off-legal skip under parallelism (M4) — the canonical order
The off-legal skip (`trainer.py:104,118-121`; raise at `samples.py:204-207`) is data-dependent, so it must be sequenced precisely so the serial oracle and the parallel path agree:
1. **Expand all selected rows** (skips applied deterministically inside the worker; off-legal rows flagged invalid in the `valid` mask, NOT dropped in-worker).
2. **Filter** to the survivor set on the main thread using the returned `valid` mask.
3. **Permute the SURVIVOR index** with the seeded RNG (permutation is drawn over the post-skip set, never the pre-skip set).
4. **Truncate to `effective_rows`** (§3.4 / M3).
5. **Micro-bucket** (`pair_budget_microbuckets`) the survivors — unchanged.

The surviving set is deterministic given `(row, d6, radius)`, so reproducibility holds; the design explicitly states the permutation is over survivors so batch composition matches between oracle and Rust path.

---

## 5. Buffer storage design

### 5.1 Packed columnar in-RAM window — NEW `hexfield/window.py`
The on-disk schema (`shards.py:147-174`) is already flat columnar (per-row scalar arrays + CSR `data`/`off` pairs). The defect is `read_compact_shard` (`shards.py:192-252`) eagerly exploding each row into a frozen `HexfieldSampleData` (tuples of boxed scalars; ~1–2 GB heap @500k). The replacement keeps every column packed.

```python
@dataclass
class PackedWindow:
    n: int
    cols: dict[str, np.ndarray]      # all scalar/block arrays + CSR data + CSR off (int64[n+1] per group)
    horizons: tuple[int, ...]        # UNION across shards; expand uses CONFIG horizons (2,6,16) (S4)
    generation: np.ndarray           # int32[n] producing-epoch per row (§5.4)
    row_shard_id: np.ndarray         # int32[n] source-shard index (diagnostics)
    @classmethod
    def empty(cls) -> "PackedWindow": ...     # n==0; train_passes already handles empty (trainer.py:81)
    def row_view(self, i) -> "PackedRowView": ...   # zero-copy slices feeding ONE expand_sample call
```
Columns kept packed (the exact `hexfield_compact_v1` set): scalars `turn_index, current_player, phase, value, moves_left, first_q, first_r, first_present`; blocks `stvalue, stvalue_mask (n,H)`; CSR groups `hist (hist_qr 2·ΣL / hist_owner / hist_pidx / hist_off)`, `pol (pol_act/pol_w/pol_off)`, `opp (opp_act/opp_w/opp_off)`, and the four `own_hot/opp_hot/own_win/opp_win` qr-CSR groups (`_pack_qr`, `shards.py:45-51`).

```python
def load_packed_shard(path) -> PackedWindow:   # np.load, KEEP packed; validate schema_version;
                                               # legacy restnet shards defer to read_legacy_restnet_shard (compat island)
def concat_packed(parts) -> PackedWindow:      # concat scalars/blocks/CSR-data; REBASE CSR offsets
```
`read_compact_shard` is **demoted to a parity/CI oracle** (no longer on the hot path) but kept intact.

### 5.2 Incremental append / evict
- **Append:** self-play already writes one compact shard per game (`selfplay.py:285-289`, UNCHANGED). "Append each epoch" = the manifest picks up new `epoch_NNNNNN/game_*.npz`; no row mutation.
- **Evict:** no destructive file deletion (matches dense — it never deletes selfplay shards). Old generations fall out of the taper window and the recent-window cut naturally. Optional disk GC is out of scope (the live run must not lose files).

### 5.3 Manifest / index — NEW `hexfield/buffer_manifest.py`
Persisted JSON `<output_dir>/samples/.buffer_manifest.json`, incrementally updated, mtime-free:
```python
@dataclass
class ShardEntry: rel_path: str; rows: int; generation: int; game_key: int
@dataclass
class BufferManifest:
    version: int
    entries: list[ShardEntry]        # sorted by (generation, game_key) — stable, mtime-free
    total_rows: int                  # live sum over present entries (used for WINDOW selection)
    cumulative_rows_ever: int        # MONOTONE, never decremented (used for the GOVERNOR — M2)
def scan_or_update_manifest(samples_dir) -> BufferManifest:
    # load if present; rglob ONLY for game_*.npz not already an entry (incremental);
    # SKIP shards lacking a sidecar (half-written by the live writer — S3);
    # drop entries whose file vanished; re-sort; update cumulative_rows_ever = max(prev, live_total);
    # atomic write (tmp + rename). Missing/garbled manifest -> full rebuild (self-healing).
    # version mismatch -> discard + rebuild.
```
Row count comes from the sidecar `num_rows` (`selfplay.py:288`), fallback `compact_row_count` only if the sidecar is absent. The manifest kills the per-epoch re-glob + `stat()`-sort; **it does NOT kill the per-epoch window decode** (M6 — that is what the Rust path is for).

### 5.4 Age / generation tagging
Generation = producing epoch, recoverable three consistent ways, **none using mtime**:
1. dir `samples/epoch_NNNNNN/` (`selfplay.py:337`);
2. game key: `epoch = game_key // 1_000_000`, `game_key = int(stem.split("_",1)[1])` (`selfplay.py:77,285,379`);
3. sidecar `{"epoch": ...}` (`selfplay.py:288`).

**Tie-break (S3):** the key-derived epoch is **authoritative** (structurally guaranteed by `selfplay.py:77`); the sidecar `epoch` is a cross-check that WARNS on mismatch; the dir name is the last-resort fallback when the sidecar is absent. `load_packed_shard` stamps `PackedWindow.generation` as `int32[n]` filled with the shard's generation; `concat_packed` concatenates. Every row carries its producing epoch for any generation-weighted sampling and for diagnostics (min/max/mean generation in window — replacing the meaningless mtime ordering).

### 5.5 Build-time memory ceiling (M5)
`build_window_split` must **stream-concat**: pre-size the output arrays from the manifest's per-shard row counts (and the CSR data lengths read once), then fill in place and free each shard's `PackedWindow` immediately after copy — instead of holding `parts[]` + `np.concatenate` (which peaks at ~2× the final window). The plan states a real byte budget per run config: at 300k rows the dominant terms are the scalar columns (~a few MB) and the ragged CSR `hist_qr int16[2·ΣL]` (ΣL = total history length; bounded by `max_game_plies`). The implementer MUST compute and log the realized packed bytes at build; the streaming build keeps the transient at ~1× plus one shard, which is safe alongside the live run (~4.8 GB RSS observed).

---

## 6. FILE-BY-FILE CHANGE LIST

### NEW `packages/hexfield/python/hexfield/window.py`
- `PackedWindow`, `PackedRowView`, `PackedWindow.empty()`
- `load_packed_shard(path) -> PackedWindow`
- `concat_packed(parts) -> PackedWindow` (streaming variant, §5.5)
- `compute_katago_window_rows(usable_rows, *, min_rows, expand_window_per_row, taper_window_exponent, taper_window_scale) -> int`
- `select_recent_window(entries, desired_rows) -> tuple[list[ShardEntry], int]`
- `build_window_split(selected, *, keep_prob, rng, samples_dir) -> PackedWindow`
- `_md5_path_fraction(value) -> float`, `_split_by_md5(selected, *, validation_fraction)`
- `_select_files_for_rows(entries, requested_rows, rng) -> tuple[list[ShardEntry], int]` (overshoot-skip, §3.7)

### NEW `packages/hexfield/python/hexfield/buffer_manifest.py`
- `ShardEntry`, `BufferManifest`, `scan_or_update_manifest(samples_dir) -> BufferManifest`

### NEW `packages/hexfield/python/hexfield/train_state.py`
- `HexfieldTrainState` (mirror `DenseTrainState`, `replay.py:108-149`): fields `train_bucket_level: float`, `train_bucket_level_at_row: int`, `train_steps_since_last_reload: int`, `data_files_used: set[str]`, `total_num_data_rows: int`, `window_start_data_row_idx: int`, `shuffle_dirs: list[str]`, `version: int`.
- `to_dict()` / `from_dict()` with a `version` field and a missing-key→fresh-state fallback (M1 sub-issue: old checkpoints must not KeyError).

### CHANGE `packages/hexfield/python/hexfield/trainer.py`
- REMOVE `_window_paths` (`:43-48`) and `_window` (`:65-73`).
- REPLACE `select_training_samples` (`:50-63`): scan/update manifest → governor accrual (`_update_train_bucket(cumulative_rows_ever, window_start)`) → taper → recent-window → md5 split → keep_prob → overshoot-skip selection (capped at `train_samples_per_epoch`, with `no_repeat_files` per §8/M2) → `effective_rows = min(requested, selected)` → bucket-limited skip / debit → build `PackedWindow` → set `components.shared.sample_window`. Returns the dict in §6-signatures.
- REPLACE the expand loop in `train_passes` (`:105-123`): iterate `window.row_view(i)` over a **pre-drawn permuted survivor index truncated to `effective_rows`**, dispatch expansion via the selected backend (`HEXFIELD_EXPAND`), keep the off-legal validity-mask filter (§4.5). The loss/optimizer/AMP/grad-clip block (`:124-160`) and the diagnostics write (`:162-177`) stay UNCHANGED except for the new diagnostic fields (§7/§9).
- ADD `self.train_state: HexfieldTrainState` to `__init__`; ADD `_update_train_bucket(self, total_rows, window_start)` (port `replay.py:481-499`) and `_record_shuffle_dir`-equivalent (see §8/M2 for whether it survives).

New signatures (framework kwargs preserved exactly):
```python
def select_training_samples(self, *, ctx, components, epoch: int) -> dict[str, Any]: ...
def train_passes(self, *, passes, sample_window, sample_symmetries, ctx, components, epoch) -> dict[str, Any]: ...
def _update_train_bucket(self, total_rows: int, window_start: int) -> None: ...
```

### CHANGE `packages/hexfield/python/hexfield/checkpoints.py` (M1 — the highest-severity wiring)
- `HexfieldCheckpointSaver.save` (`:85-96`): change `extra={"run": ctx.config.run.name}` →
  `extra={"run": ctx.config.run.name, "train_state": components.model.trainer.train_state.to_dict()}`.
  VERIFIED reachable: `components.model.trainer` is set at `components.py:155` and already used at `epoch/training.py:31` and `epoch/samples.py:96`. (Guard with `getattr(components.model, "trainer", None)` so a trainer without `train_state` — e.g. tests — does not crash.)
- `HexfieldCheckpointLoader.load` (`:62-79`): on the **resume branch only** (`resume` is True at `:72-75`), after `load_into`, call `components.model.trainer.train_state = HexfieldTrainState.from_dict(meta.get("train_state"))`. Do NOT touch the `initialize_from` warm-start branch (`:76`) — a BC-prefit warm start must start with a fresh governor, never inherit a stale bucket from an unrelated run. Read the key as `meta["train_state"]` (it lands in `meta` because `save_checkpoint` merges `extra` into `meta`, `checkpoints.py:17`). A missing key ⇒ `from_dict(None)` ⇒ fresh state (old-format checkpoints resume cleanly).

### CHANGE `packages/hexfield/python/hexfield/config.py`
- ADD 12 fields to `TrainingSection` (`:65-72`) — see §7. `_merge` (`:296-301`) is a strict flat merge that already accepts new scalar fields and tolerates unset ones (verified — raises only on UNKNOWN toml keys), so **no parse-code change is needed**. The live `main_2.toml` sets only fields that remain, so nothing breaks; the fresh run's toml sets the new block.

### CHANGE NEW Rust `packages/hexfield/rust/src/replay_expand.rs` (+ register in `lib.rs`)
- `expand_shard_train(columns, d6, horizons, support_radius, tolerate_off_legal) -> (stacked buffers, valid mask)` — §4.2. Reuses `support.rs::build_support` and the `features.rs` build already present; adds the per-row D6 application and the `_legal_slot` policy projection. Returns a per-row `valid` mask for the off-legal skip. Parity-pinned by fixtures like `support.rs:5-9`.
- `payload.rs`, `serve_pack.rs`, `support.rs`, `features.rs`: UNCHANGED (the kernel is reused, not modified).

### CHANGE the fresh-run toml (NEW `configs/hexfield_main_3.toml`)
- Copy `hexfield_main_1.toml`, set `run.name="hexfield_main_3"`, and add the `[model.config.training]` block in §7. The live `hexfield_main_2.toml` is NOT edited.

### UNTOUCHED (contract-preserving — must NOT regress)
- `selfplay.py` write path (`:285-289`), generation key (`:77,:379`).
- `samples.py` `expand_sample`/`_legal_slot`/`finalize_game_samples`/STV build — expansion math frozen.
- `support.py` / `features.py` — BFS/feature math frozen (Python stays the oracle; Rust the twin).
- `shards.py` schema (`:147-174`); `read_compact_shard` (demoted to oracle); `read_legacy_restnet_shard` (`:255-342`, Phase-B import island).
- `batching.py` `pair_budget_microbuckets` / `collate_training` — VRAM micro-bucketing frozen.
- `plugin.py` `uses_shared_sample_store=False` (`:54`) and trainer wiring.
- `hexo_train/epoch/{loop,samples,training,symmetry}.py` and `hexo_train/{components,context,config,symmetry}.py` — framework dispatch; signatures preserved.

**Contract subtleties preserved:**
- `select_training_samples(ctx=, components=, epoch=)` keeps exact kwargs, returns a dict, and **still sets `components.shared.sample_window`**.
- **Opaque-window regression guard (VERIFIED):** `PackedWindow` must expose neither `window_size` nor `index.sample_count`; otherwise `D6SymmetrySelector.select_for_window` (`symmetry.py:60-83`, count via `_sample_count` `:86-96`) would blake2b-hash every row each epoch. With only `n`/`cols`, `_sample_count` returns 0 and the empty-tuple path is taken — cheap, as today. `train_passes` keeps `_ = sample_symmetries` (`trainer.py:76`) and self-draws D6.
- `train_passes` keeps all six kwargs and returns a dict.

### `select_training_samples` reference return dict
```python
{"status": "completed"|"skipped"|"train_bucket_limited", "epoch": epoch,
 "total_rows": cumulative_rows_ever, "live_total_rows": total_rows,
 "desired_rows": desired, "used_rows": used, "keep_prob": keep_prob,
 "effective_rows": effective_rows, "window_rows": window.n,
 "window_start": max(0, total_rows - used), "train_bucket_level": self.train_state.train_bucket_level,
 "reuse_ratio": effective_rows / max(1, new_rows_this_epoch)}
```

---

## 7. Config schema

All 12 fields added to `TrainingSection` (`config.py:65-72`), accessed as `self.config.training.*`. Types/semantics mirror dense; hexfield-tuned defaults per §3.1.

| Field | Type | hexfield default | Notes |
|---|---|---|---|
| `shuffle_min_rows` | int | 20_000 | taper floor (dense 100_000) |
| `shuffle_keep_target_rows` | int | 300_000 | EXISTS today (currently the hard cutoff); reused as keep_prob target (dense 600_000) |
| `shuffle_taper_window_exponent` | float | 0.65 | dimensionless — = dense |
| `shuffle_expand_window_per_row` | float | 0.4 | dimensionless — = dense |
| `shuffle_taper_window_scale` | float | 20_000.0 | offset (dense 50_000.0) |
| `validation_fraction` | float | 0.0 | md5 split (implemented, §3.6) |
| `train_samples_per_epoch` | int | 100_000 | single-pass row cap |
| `max_train_bucket_per_new_data` | float | 8.0 | reuse accrual per new row |
| `max_train_bucket_size` | float | 500_000.0 | bucket cap |
| `no_repeat_files` | bool | **False** | DECISION per §8/M2 — default OFF for hexfield (single-game shards) |
| `expand_backend` | str | "serial" | "serial"\|"pool"\|"rust" (also `HEXFIELD_EXPAND` env) |
| `expand_workers` | int | 0 | 0 ⇒ auto `min(8, cpu//4)`; pool/rust only |

`[model.config.training]` block for `configs/hexfield_main_3.toml`:
```toml
[model.config.training]
batch_rows = 32
learning_rate = 1e-3
weight_decay = 1e-4
grad_clip = 1.0
warmup_steps = 0
shuffle_min_rows = 20000
shuffle_keep_target_rows = 300000
shuffle_taper_window_exponent = 0.65
shuffle_expand_window_per_row = 0.4
shuffle_taper_window_scale = 20000.0
validation_fraction = 0.0
train_samples_per_epoch = 100000
max_train_bucket_per_new_data = 8.0
max_train_bucket_size = 500000.0
no_repeat_files = false
expand_backend = "serial"   # flip to "rust" after parity gates (§10)
expand_workers = 0
```

---

## 8. Must-fix issues from critique and resolutions

- **M1 (CRITICAL) — train_state not persisted ⇒ governor resets every resume.** RESOLVED: saver writes `train_state` into `meta` via the `extra` hook; `components.model.trainer` reachability is **verified** (`components.py:155`, `epoch/training.py:31`). Loader restores `meta["train_state"]` on the **resume branch only** (`checkpoints.py:72-75`), never on `initialize_from`. `from_dict` is version-gated with a missing-key→fresh fallback so old checkpoints don't KeyError. A save→load round-trip test of non-empty `data_files_used`/`train_bucket_level` is a Phase-3 gate (§9/§10).
- **M2 (CRITICAL) — non-monotone `total_rows` breaks bucket accrual, and `no_repeat_files` + single-game shards starve the pool.** RESOLVED (two parts): (a) the manifest tracks a separate **monotone `cumulative_rows_ever`** fed to the governor; window selection uses the live total. (b) `no_repeat_files` defaults **OFF** for hexfield — its shards are single games (~28 rows, ~18k files), so marking a game permanently-used after one epoch touches ≥1 of its rows would collapse the eligible pool to the newest epoch within 1–2 epochs, defeating the taper window. The **bucket governor alone throttles reuse** (its design purpose); `no_repeat_files` was a secondary anti-staleness lever in dense that assumed coarse ~70k-row output shards we deliberately do not create. The dense `_record_shuffle_dir` last-20-prune is therefore dropped (no shuffle output dirs exist in the in-RAM design). The knob remains in config for completeness but is documented OFF.
- **M3 (CUT CORNER) — dropped batch-align + "train whole window" vs "cap at effective_rows" contradiction.** RESOLVED: not re-sharding to disk is the justified part (compact rows are small; dense itself shuffles in RAM, `replay.py:722-728`). The faithful part that IS enforced: **single pass, no within-epoch repeat, capped at exactly `effective_rows`** by truncating the permuted survivor index before the micro-bucket loop (§3.4/§4.5). The bucket debit-by-`effective_rows` is then honest.
- **M4 (BUG) — off-legal skip + Rust path + bucket debit.** RESOLVED: Rust returns a per-row `valid` mask; the main thread filters survivors, then permutes survivors, then truncates, then micro-buckets (§4.5). The bucket is debited by `effective_rows` (dense semantics — debit at selection); the realized over-debit during a radius transition is documented and surfaced as a `rows_skipped_off_legal` diagnostic (S2) rather than refunded.
- **M5 (CUT CORNER) — `validation_fraction` accepted but split unimplemented.** RESOLVED: `_split_by_md5` + `_md5_path_fraction` are ported verbatim and CALLED in `select_training_samples`; `validation_fraction>0` builds a val `PackedWindow`. Not a lying knob.
- **M5-mem (HIGH) — build-time 2× spike.** RESOLVED: streaming concat (§5.5) — pre-size from manifest counts, fill in place, free per shard; log realized packed bytes.
- **M6 (MEDIUM) — manifest oversells "no re-decode".** RESOLVED by honesty: the manifest kills the re-glob/`stat()`-sort only; the per-epoch window DECODE is retained (dense does the same) and is exactly what the Rust/parallel path accelerates. Documented in §2/§5.3.
- **S1 — conflicting taper defaults across design parts.** RESOLVED: the §3.1/§7 hexfield-tuned set (`min_rows=20_000`, `scale=20_000`, `keep_target=300_000`) is canonical; the dimensionless `0.65`/`0.4` stay. Part-3's dense-literal numbers are superseded.
- **S2 — off-legal under-count.** RESOLVED: `rows_skipped_off_legal` diagnostic added; over-debit documented as accepted.
- **S3 — generation tie-break + half-written shard race.** RESOLVED: key-derived epoch authoritative (sidecar warns, dir last resort); manifest scan SKIPS shards lacking a sidecar so it never opens a half-written npz.
- **S4 — STV horizon union.** RESOLVED: `PackedWindow.horizons = union`; expand passes the CONFIG horizons `(2,6,16)` (matching dense "trainer requests its config horizons", `replay.py:754-755`); the stored `stvalue`/`stvalue_mask` columns are preserved verbatim.
- **S5 — `window_start` threading.** RESOLVED: `window_start = max(0, total_rows - used)` computed and passed into `_update_train_bucket(total_rows, window_start)` (dense `trainer.py:481,499`), not dropped.

---

## 9. Parity + testing strategy (must NOT disturb the live run)

The live `hexfield_main_2` (a running supervised process) must not be touched. The new pipeline cannot be bitwise-compared to the old (RNG stream changed by construction, §4.4), so parity is **layered and offline**:

1. **Window-math unit parity (pure, no GPU, no live data).** Assert `compute_katago_window_rows`, `select_recent_window`, `keep_prob`, `_md5_path_fraction` byte-equal the dense functions (`replay.py:620-632`, `:681-690`, `:404`, `:908-910`) on synthetic row-count / path vectors. Fully deterministic — the cheapest high-value gate.
2. **Decode parity oracle (read-only on a COPY of live `samples/`).** `cp -p`/`rsync` a snapshot of `samples/epoch_*/` to scratch (never read the live dir under write-load for this). Assert `load_packed_shard(path).row_view(i)` reconstructs field-identical values to `read_compact_shard(path)[i]` (the existing oracle) for every row of a sample of shards, including CSR/ragged columns via `_unpack_qr`. Zero model involvement.
3. **Expansion parity (Rust vs Python), FIXED symmetry.** With a fixed `d6` vector (not RNG-drawn), assert `expand_shard_train` (Rust) == per-row `expand_sample` (Python) element-wise on coords/dist/nbr/features/policy/opp/stvalue across all 12 D6 values, mirroring the existing parity fixtures (`support.rs:5-9`). Include rows that trip the off-legal skip under `HEXFIELD_SUPPORT_RADIUS<8` to pin the validity-mask drop behavior (M4).
4. **Statistical training parity (NOT bitwise).** Run OLD and NEW pipelines as two SEPARATE offline processes (separate output dirs, CPU or a time-shared/second GPU — never the live GPU under the live run), from the same checkpoint + same `samples/` snapshot, for ≥10 epochs. Compare DISTRIBUTIONS: per-epoch loss components, grad-norm percentiles, `window_rows`, `keep_prob`, `reuse_ratio`. Equivalence = loss curves within a noise band over ≥10 epochs.
5. **Resume/restart failure-mode tests (M1).** `to_dict`/`from_dict` round-trip; resume from a checkpoint LACKING `train_state` (old format) ⇒ fresh state, no crash; resume mid-window ⇒ bucket/`data_files_used` restored equal; `initialize_from` warm start does NOT inherit a stale bucket.
6. **Memory/throughput bench.** On a snapshot, measure peak RSS of `build_window_split` (assert streaming keeps it near 1× + one shard, §5.5) and rows/s of each `expand_backend` (serial vs pool vs rust). Confirm Rust ≥ serial and pool's torch-RSS cost. Never on the live GPU.
7. **Concurrency determinism.** Fixed seed: run the parallel path twice ⇒ identical output (post-skip survivor permutation, M4); `workers=1` vs `workers=N` ⇒ identical (commutativity under `collect()`).

---

## 10. Rollout plan

### Fresh run vs migration
Ship as a **fresh run `hexfield_main_3`** with `expand_backend="serial"` initially. Let `hexfield_main_2` finish undisturbed. No hot edit to the live process, no default flips on it. There is no cross-format migration: hexfield reads its own `samples/epoch_*/game_*.npz`; the first `scan_or_update_manifest` builds the manifest additively (read-only, no file moves, no `cp -p` mtime hazard since we key on epoch/key).

### Resume/restart safety for the train-bucket state
- Saver persists `train_state` in `meta` (M1); loader restores on resume only; `from_dict(None)` ⇒ fresh state ⇒ old checkpoints resume cleanly.
- The monotone `cumulative_rows_ever` lives in the manifest (rebuilt/self-healed from the tree if lost) AND is cross-checked against the persisted `train_bucket_level_at_row`; on a window regeneration the governor's `elif total_rows < level_at_row` branch handles the reset safely (§3.5).
- The manifest write is atomic (tmp+rename) and self-heals on parse/version error, so a crash mid-update never wedges a resume.

### Phased implementation checklist (ordered; each step independently testable)
1. **Config + train_state plumbing.** Add the 12 `TrainingSection` fields; add `HexfieldTrainState` + `to_dict`/`from_dict`; wire saver/loader (M1). Gate: §9 test 5 (round-trip + old-format resume) passes. No behavior change yet (defaults reproduce v1-ish selection if window code not yet swapped — keep v1 path until step 4).
2. **Manifest.** Implement `buffer_manifest.py`; build from a snapshot; assert ordering and `cumulative_rows_ever` monotonicity. Gate: read-only against a `samples/` copy, no GPU.
3. **Packed window + decode oracle.** Implement `window.py` storage (`PackedWindow`, `load_packed_shard`, streaming `concat_packed`). Gate: §9 tests 2 + 6 (decode parity, build-time RSS).
4. **Window mathematics + selection + governor.** Implement taper/recent-window/keep_prob/md5-split/overshoot-skip; replace `select_training_samples`; wire `_update_train_bucket` (monotone counter, M2). Keep `expand_backend="serial"` using `PackedWindow.row_view` → existing `expand_sample`. Gate: §9 test 1 (window math) + a dry-run `select_training_samples` on a snapshot asserting the bucket/skip dict.
5. **Consumer rewrite (serial).** Replace the `train_passes` expand loop: pre-drawn survivor permutation truncated to `effective_rows`, off-legal validity filter, micro-bucket unchanged. Move `rng.shuffle`/`rng.randrange` to pre-drawn numpy vectors (determinism fix). Gate: §9 test 4 (statistical training parity, serial) over ≥10 offline epochs.
6. **Spawn ProcessPool fallback.** Add `expand_backend="pool"`. Gate: §9 test 7 (determinism) + throughput bench; correctness == serial.
7. **Rust train-read kernel.** Implement `replay_expand.rs::expand_shard_train` + `lib.rs` registration; add `expand_backend="rust"`. Gate: §9 test 3 (Rust↔Python expansion parity across all 12 D6 + off-legal rows) + test 7 (determinism, workers=1 vs N) + throughput.
8. **Flip default + launch the fresh run.** Once gates 1–7 pass, set `expand_backend="rust"` in `configs/hexfield_main_3.toml` and launch `hexfield_main_3`. `hexfield_main_2` finishes on its own.

Each step is independently testable offline and none requires touching the live run or its GPU until step 8, which is a brand-new run.
