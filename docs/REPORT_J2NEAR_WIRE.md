# J2near production wiring report

Date: 2026-07-21  
Lane: `claude/j2near-wire`  
Base production tip: `55f3c2b6`

## Result

J2near is now a first-class rollout configuration lever and production is
configured for node cap 1,000 with J2near enabled. The new lever is default
off everywhere except the explicit production TOML, so an absent or false key
retains the historical `WidthOptions::vcf_pair_complete()` profile.

## Wiring

- `SelfplayConfig` now declares `tss_solver_j2near: bool = False` and
  `build_divergence_overrides` emits the concrete boolean in both the base and
  Fast-class rollout maps.
- Rust's strict divergence-key whitelist accepts `tss_solver_j2near`, the
  shared resolver stores it in `Divergences`, and the default in
  `Divergences::parity()` is false (`production()` inherits that default).
- `Divergences::solver_j2near_enabled()` is the single mode gate: J2near is
  effective only when the key is true and `tss_solver_mode > 0`.
- The selected width reaches all production deep-solver paths: the persistent
  inline solver, the per-move root-guard solver, and each async worker request.
  A profile change uses the existing width setter, so persistent proof caches
  retain the established profile-isolation behavior.
- The new Rust rollout seam test covers true, false, absent, and true-with-mode-0
  cases and inspects the constructed solver's resolved
  `free_tempo_j2near` value.

## Environment-path removal

The `TSS_VCF_J2NEAR` runtime override was removed. Repository inspection found
no harness dependency on it: `tss_j2near_ab` and the witness tests select
`WidthOptions::vcf_pair_complete()` / `WidthOptions::vcf_pair_j2near()`
directly. The env-specific unit test was coherently replaced by
`j2near_named_profile_resolves_default_off_profile`, which verifies false is
the complete-pair profile and true is the named J2near profile. There is no
remaining `TSS_VCF_J2NEAR` reference under `packages/`, `tests/`, or `configs/`.

## Production config diff

```diff
-# cap 750 ...
-tss_solver_node_cap = 750
+# Cap 1,000 + J2near ON (owner completeness ruling 2026-07-21; wall budget
+# consciously relaxed). REPORT_J2NEAR_CAP.md measured both cap-1,000 arms as
+# decision-safe: 1,362 decided rows, zero archive downgrades, and zero W/L
+# flips. First-epoch validation gates at relaunch still apply.
+tss_solver_node_cap = 1000
 ...
+# Free-tempo J2near width is always ON for production under the owner ruling.
+tss_solver_j2near = true
 tss_solver_group2 = false
```

This applies the owner completeness ruling while citing the measured cap-1,000
rows in `docs/REPORT_J2NEAR_CAP.md`: both arms decided 1,362 rows with zero
downgrades and zero W/L flips.

## Validation

The required physical-memory gate passed before Cargo compilation. Windows CIM
was unavailable to this restricted session, so the native Memory performance
counter was used. It reported 14.16 GiB free initially, 14.17 GiB before the
full Python-feature suite, and 14.09 GiB before the release suite (all above
the 8 GiB floor).

All Cargo commands ran from this worktree with:

```powershell
$env:CARGO_TARGET_DIR = (Join-Path (Get-Location) '.cargo-target')
$env:RUST_MIN_STACK = '33554432'
```

Focused rollout seam:

```powershell
cargo test -p hexfield_eq --features python --target x86_64-pc-windows-msvc rollout_config_resolves_j2near_width_profile -- --test-threads=1
```

Result: **1 passed, 0 failed**.

Focused J2near unit set:

```powershell
cargo test -p hexfield_eq --features python --target x86_64-pc-windows-msvc j2near_ -- --test-threads=1
```

Result: **3 passed, 0 failed, 4 ignored**. The passed tests include
`j2near_widens_only_the_three_preregistered_root_shapes`, the replacement
named-profile default-off test, and the rollout seam. The four measurement /
witness campaigns remained intentionally ignored.

Serialized Windows Python-feature Rust suite:

```powershell
cargo test -p hexfield_eq --features python --target x86_64-pc-windows-msvc -- --test-threads=1
```

Result: **221 passed, 0 failed, 43 ignored**; doc tests also passed (0 tests).
This is the current baseline shape plus the new rollout seam test.

Required serialized release library suite:

```powershell
cargo test -p hexfield_eq --lib --release --target x86_64-pc-windows-msvc -- --test-threads=1
```

Result: **136 passed, 0 failed, 42 ignored**.

Python config smoke (the host lacks `pytest` and the package-level optional
NumPy dependency, so `config.py` was loaded directly with the installed
interpreter):

```powershell
python -c "import importlib.util, pathlib, sys; p=pathlib.Path('packages/hexfield_eq/python/hexfield_eq/config.py'); s=importlib.util.spec_from_file_location('hexfield_eq_config', p); m=importlib.util.module_from_spec(s); sys.modules[s.name]=m; s.loader.exec_module(m); off=m.build_divergence_overrides(m.SelfplayConfig()); on=m.build_divergence_overrides(m.SelfplayConfig(tss_solver_j2near=True)); assert off['tss_solver_j2near'] is False; assert on['tss_solver_j2near'] is True; print('python_j2near_config_plumbing: PASS')"
```

Result: **PASS**.

Final static checks: `git diff --check` passed, `tss_verify.rs` has no diff,
and the pre-existing untracked `CODEX_BRIEF.md` was left untouched. No commit
was created.
