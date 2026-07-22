# Light spec: weight averaging (EMA/SWA) for the hexfield_eq run

Status: DRAFT (2026-07-22). Target: main_6, not main_5. Motivation: KataGo's
stochastic weight averaging is a larger contributor to its smooth strength
curve than its LR schedule; the hexfield_eq **run** trainer currently keeps no
averaged weight copy (only a grad-norm EMA for the adaptive clip). This adds an
EMA of the weights, evaluates/gates on it, and A/B-tests whether self-play
should use it too.

## Why this is cheap here
- **No BatchNorm.** hexfield_eq uses LayerNorm/GroupAffineNorm, so the classic
  SWA headache — recomputing normalization statistics over the averaged weights
  — does not apply. The averaged weights are directly usable.
- **Reference impl exists.** `prefit._ema_update` already does exactly this
  (lerp params toward the live net, copy constant-LUT buffers). Copy it.
- **Memory is trivial.** The net is ~0.95M params (~4 MB fp32); a second copy
  is negligible in VRAM and on disk.

## Design

### 1. EMA maintenance (trainer.py) — mechanical
- `__init__`: when `training.ema_enabled`, build `self.ema_model =
  copy.deepcopy(self.model).requires_grad_(False).eval()` (eager module, never
  the compiled wrapper).
- After each successful `optimizer.step()` (post-`scaler.step`): call the ported
  `_ema_update(self.ema_model, self.model, decay=training.ema_decay)`.
  Update from the EAGER `self.model` (the compiled forward shares params, but
  update the eager handle — mirrors the prefit note).
- Skip the update on non-finite steps (reuse the existing `torch.isfinite(norm)`
  gate that already guards the grad-norm EMA).

### 2. Persist + resume (checkpoints.py) — the one correctness-critical bit
- `save_checkpoint` / `HexfieldCheckpointSaver.save`: add
  `payload["ema_model"] = ema.state_dict()` when present.
- Loader: on `resume_from`, restore `ema_model` from the checkpoint. On a fresh
  `initialize_from` (new run / new arch), initialize EMA = the raw warm-started
  weights (NOT absent — else the first eval sees a half-formed average).
- This must survive supervisor restarts exactly like `_grad_norm_ema` and
  `train_state` already do — follow that persistence pattern; it is the only
  place a bug would silently corrupt results (EMA resetting to raw mid-run
  erases the averaging benefit without any error).

### 3. Consumption — where it actually earns its keep (the real work)
Saving an EMA that nothing reads is a no-op. Two consumers, two decisions:
- **Eval / gating (DO from the start):** the multistage eval should load the
  candidate from `payload["ema_model"]` instead of `payload["model"]`. This is
  the KataGo pattern (the averaged net is the exported/gated net). Touches the
  eval/serve checkpoint loader (`eval_arena` / `infer_net_kwargs` path), which
  has **parity gates** — re-green them. Keep a config/CLI switch to eval raw vs
  EMA so the A/B is a flag flip.
- **Self-play generation (A/B, do NOT assume):** raw = latest, more exploratory;
  EMA = smoother, stronger-on-average but lagged. Genuine strength tradeoff —
  MEASURE, don't guess. Default self-play to RAW for the first main_6 segment,
  eval on EMA, and compare the fixed-anchor descriptive Elo curve of {raw
  self-play + EMA eval} vs a later segment of {EMA self-play + EMA eval}.

### 4. Config (config.py TrainingSection)
```
ema_enabled: bool = False      # default OFF -> byte-identical to today
ema_decay: float = 0.9995      # prefit's value; ~2000-step effective window
```
Both ride nothing arch-level (weights only), so no checkpoint-meta / arch-env
plumbing — unlike cell_q / reg_lane.

## Decisions to pin before building
- `ema_decay`: start at the prefit's 0.9995. KataGo-style SWA uses a longer
  window; 0.999-0.9999 is the band to sweep if the curve is over/under-smoothed.
- Eval-on-EMA from day one; self-play-on-EMA gated behind the measured A/B.
- Whether `save_name="latest"` (the pointer selfplay reloads) should point at
  raw or EMA weights — this IS the self-play A/B in practice.

## Tests
- EMA update numeric correctness (one step: `ema = decay*ema + (1-decay)*live`).
- Checkpoint round-trip carries `ema_model`; strict-load bitwise.
- Resume restores EMA (not reset to raw); fresh `initialize_from` seeds EMA=raw.
- Eval loader selects EMA vs raw per the switch.

## Effort / risk
- Minimum testable (maintain + persist + resume + eval-on-EMA + tests):
  **~half a day**, low risk (mechanism exists; no BN recalibration; trivial mem).
- Proper (self-play A/B + measurement pass + parity-gate re-green):
  **~1-2 days**, gated on one descriptive-Elo comparison.
- Only real hazard: resume-restore correctness (§2). Everything else is additive
  and default-OFF.
