# hexfield_main_3 supervisor — install / operate

A **single self-contained, version-tracked** systemd unit (`hexfield-supervisor-3.service`).
Unlike main_2 — whose real config lived in an *untracked* `scratch/kbuf.conf` drop-in, so the
experiment silently depended on a file outside git — everything main_3 needs is inline in the
committed unit. Installing the file is the whole deployment.

## Prerequisites (before `systemctl start`)
1. The tree at `ROOT=/mnt/e/Hexo-BotTrainer-hexgt` is checked out to branch
   `claude/hexfield-main3-v3` (or v3 is merged to `main`). The supervisor puts
   `$ROOT/packages/hexfield/python` on `PYTHONPATH`, so v3 hexfield must be the code there.
2. The Rust extension is rebuilt for v3 (the new `visit_policy_q_bytes` search export):
   `bash scripts/_rebuild_hexfield.sh`.
3. The v3 BC prefit exists and `[checkpoint].initialize_from` in
   `configs/hexfield_main_3.toml` points at it (v3-arch, radius-4-native).
4. `main_2` epoch_000045.pt exists at
   `/mnt/e/Hexo-BotTrainer/runs/hexfield_main_2/checkpoints/` (the eval anchor; loaded via the
   frozen `legacy_model_v2` fallback in `eval_arena._load_hexfield_net`).

## Install
```bash
sudo cp scripts/systemd/hexfield-supervisor-3.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now hexfield-supervisor-3.service
journalctl -u hexfield-supervisor-3.service -f
```

## Notes
- Breaker: uptime <300s ×3 consecutive OR >8 crashes/hour → writes `supervisor_halted.flag`
  and stops relaunching. Clean completion writes `supervisor_completed.flag`.
- Do **not** relaunch via `wsl.exe` background tasks — run it as the systemd service
  (a detached `wsl.exe` SessionLeader teardown previously killed the trainer with code 143).
- Stop: `sudo systemctl stop hexfield-supervisor-3.service`.
