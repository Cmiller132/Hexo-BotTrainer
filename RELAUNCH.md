# One-step relaunch — dense_cnn Model 1 (scratch_64)

The live run (PID 10672, started 2026-05-28 23:18:54) is **untouched**. This note
is the single thing to do before relaunching so the run **advances** instead of
restarting at epoch 9.

## The problem

`configs/dense_cnn_model1_scratch_64.toml` `[checkpoint]` only sets
`initialize_from` (the fixed bootstrap epoch-8 prefit, `epoch: 8` in its
metadata). The loader uses `resume_from OR initialize_from`
(`checkpoints.py:29`) and start epoch is `loaded_epoch + 1` (`loop.py:155`).
With no `resume_from`, every relaunch reloads the bootstrap → starts at **epoch 9**,
discarding all saved epoch checkpoints.

## The one-step fix (apply ONLY at relaunch, not to the live run)

1. Find the newest epoch checkpoint (its `metadata.epoch` drives the resume point):

   ```powershell
   Get-Content E:\Hexo-BotTrainer\data\checkpoints\dense_cnn_model1_scratch_64_latest.txt
   # or, equivalently:
   Get-ChildItem E:\Hexo-BotTrainer\runs\dense_cnn_model1_scratch_64\checkpoints\epoch_*.pt |
     sort LastWriteTime -desc | select -First 1 -Expand FullName
   ```

2. Add a `resume_from` line to `[checkpoint]` pointing at that file. `resume_from`
   takes precedence over `initialize_from`; leave `initialize_from` as-is.

   As of this writing the newest is `epoch_000010.pt` (→ resumes at **epoch 11**):

   ```toml
   [checkpoint]
   resume_from = "../runs/dense_cnn_model1_scratch_64/checkpoints/epoch_000010.pt"
   initialize_from = "../runs/dense_cnn_model1/checkpoints/bootstrap_sealbot_050000_prefit_epoch_000008_remote_replay_converted.pt"
   save_name = "latest"
   ```

   Paths in the config are resolved relative to the config dir (`configs/`), which
   is why the `../runs/...` prefix is correct.

3. Relaunch (sets diagnostic env + watchdog + trainer):

   ```powershell
   $env:PYTHONFAULTHANDLER=1; $env:PYTHONUNBUFFERED=1; $env:RUST_BACKTRACE='full'
   .\scripts\start_model1_training.ps1 -ConfigPath configs\dense_cnn_model1_scratch_64.toml -SealBotPath E:\SealBot
   ```

> **Caveat:** `resume_from` is a fixed path — there is no auto-resume-from-latest.
> Re-run step 1 and bump the path on every future relaunch. (The
> `dense_cnn_model1_scratch_64_latest.txt` pointer file always holds the newest
> path, so step 1 is a one-liner.)

After relaunch, update the crash monitor (and any cron) to the new PID and the new
`trainer.<stamp>.err.log` path.
