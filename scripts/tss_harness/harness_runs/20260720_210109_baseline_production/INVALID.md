# INVALID AS BASELINE — do not diff against this run

Two arm-identity defects, found in the 2026-07-20 shakedown (fixed in
e28f6915), make this archive internally inconsistent:

1. The bench subprocess received config `{}` and silently resolved the
   production toml's engine-default **h16**, while the coverage sweep ran
   unbounded and `manifest_baseline_production.json` claims
   `semantic_horizon = u32::MAX`. The scorecard's 139 moves/min measures a
   DIFFERENT arm than the records (V2 h2h: those arms differ by ~60 Elo).
2. The sweep ran `goal=win`, which filters loss facts at the root — the
   `loss: 0` coverage is by construction, not measurement. Production
   parity is `goal=both`.

Superseded by `baseline_production_v2` (bench-identity gate now enforces
echo-to-echo config match; default goal=both). Kept for provenance only.
