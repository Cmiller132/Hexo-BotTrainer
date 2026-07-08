# 04 — Sanitization Checklist

Run against the **staged repo** (`E:\hexo-bot`) immediately before the initial
commit, and again on the fresh clone before pushing. The fresh-single-commit
strategy means history can't leak — these sweeps cover the working tree.

## Grep sweeps (all must come back empty or explained)

| Sweep | Pattern | Expected |
|---|---|---|
| Private mounts / drives | `/mnt/e/`, `E:\\`, `/mnt/c/` | 0 hits (all parameterized in Phase 3) |
| Private venvs / home | `/root/`, `\.venvs`, `hexgt-build`, `hexfield-dev`, `epicm`, `$HOME/hexfield` | 0 hits |
| Personal identity | `colton`, `miller`, `coltonmilleredu` | 0 hits outside LICENSE (name in LICENSE is intentional) |
| Secrets | `api_key`, `secret`, `password`, `Bearer`, `hf_[A-Za-z0-9]`, `ghp_`, `sk-`, `AKIA`, `ANTHROPIC` | 0 hits (attention "tokens" in ML code are fine — review each hit) |
| Private repo names | `Hexo-BotTrainer`, `hexgt`, `gumbel` (worktree name), `SealBot` local path | Only the public SealBot GitHub URL survives |
| Legacy lineages | `dense_cnn`, `restnet`, `hexgt`, `hexgnn` | Only intentional survivors: vendored-comment mentions in hexfield (`features.py`, `shards.py` — rewrite these comments too), UI defensive branches in `app.js` |
| Internal run history | `main_1`–`main_6`, `main4`, `ep75`, private run-dir names | Only main_7 provenance in the model card |
| Claude/AI tooling | `.claude`, `CLAUDE.md`, `claude` | 0 hits |
| Stripped paths (06 §8) | `PIPELINE_DEPTH2`, `GATE_COMPLETE`, `NO_PREFETCH`, `SERIAL_BACKUP`, `SERIAL_COMPLETE`, `CONV_FP8`, `forced_playout`, `dirichlet`, `visit_scaled`, `fpu_zero`, `ml_auto_disabled`, `run_batch`, `RunnerConfig`, `hexo-rl`, `build_selfplay_request`, `uses_shared_sample_store`, `legacy_model_v2`, `puct_eval_overrides` | Only intentional survivors (parity-test comments, docs explaining the Gumbel-vs-PUCT design choice) |

## Structural checks

- [ ] `find . -type l` → no symlinks anywhere.
- [ ] No `__pycache__/`, `.pytest_cache/`, `target/`, `*.pyc`, `*.so`, `*.log`.
- [ ] No files with `_` scratch prefix at repo root or in `scripts/`.
- [ ] `git log --oneline` → exactly one commit; `git log --format='%an %ae'` →
      the identity you want public (set `user.name`/`user.email` in the staged
      repo before committing — the global git identity may differ).
- [ ] `git lfs ls-files` → exactly the intended `models/*.pt`; repo size sans
      LFS < ~20 MB.
- [ ] `.pt` files: load each with `torch.load(..., map_location='cpu')` and
      inspect top-level keys — confirm no absolute paths, run-dir strings, or
      unexpected metadata pickled inside (training checkpoints often embed the
      config, which embeds paths — scrub or re-save if so).
- [ ] `configs/*.toml` headers read as public documentation, not private diary.
- [ ] systemd templates contain placeholder paths only.
- [ ] Binary sweep: `git ls-files | file -f -` (or equivalent) — no unexpected
      binaries besides the LFS weights.

## Judgment-call reviews (read, don't grep)

- [ ] Root `README.md` — rewritten, no private environment section.
- [ ] `docs/ARCHITECTURE.md` — trimmed of legacy lineages and private workflow.
- [ ] `docs/specs/*.md` — path-scrubbed; specs may reference old screens/runs
      in prose, decide keep/trim per file.
- [ ] `packages/*/README.md` — each reviewed (engine README mentions the
      workspace; utils README mentions consumers; frontend README mentions
      launch scripts — all must match the public layout).
- [ ] `tests/conftest.py` + `tests/eval_dashboard/` — no fixtures pointing at
      private paths or SealBot's local checkout.
- [ ] Model card numbers (epoch, eval strength) match the actually-shipped
      checkpoint.
