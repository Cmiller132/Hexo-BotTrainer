# Archive Manifest — 2026-06-16

Workspace cleanup: stale one-off scripts, dead-end analysis probes, and superseded docs moved here from their original locations. All files are git-tracked moves (history preserved via `git mv`).

**To restore any file:** `git mv archive/2026-06-16/<path> <path>` (the original path is the part after `archive/2026-06-16/`).

Protected paths (runs/, data/, target/, packages/, configs/, tests/, .git/, current hexfield session tooling, protected docs) were NOT touched. The live training run is unaffected.

## Archived files (221 total)

| Original path | Category / reason |
|---|---|
| `scripts/_authstatus.sh` | Stale one-off status/monitor/wait/verify shell helper; not referenced by active run, configs, or systemd units. |
| `scripts/_bc_eval.py` | One-off behavior-cloning experiment script; lineage parked. |
| `scripts/_bc_smoke.py` | One-off behavior-cloning experiment script; lineage parked. |
| `scripts/_bc_train.py` | One-off behavior-cloning experiment script; lineage parked. |
| `scripts/_bounce_trainer.sh` | Stale one-off status/monitor/wait/verify shell helper; not referenced by active run, configs, or systemd units. |
| `scripts/_check_clean.sh` | Stale one-off status/monitor/wait/verify shell helper; not referenced by active run, configs, or systemd units. |
| `scripts/_check_gpu_state.sh` | Stale one-off status/monitor/wait/verify shell helper; not referenced by active run, configs, or systemd units. |
| `scripts/_coldstart_check.sh` | Stale one-off status/monitor/wait/verify shell helper; not referenced by active run, configs, or systemd units. |
| `scripts/_deep_status.sh` | Stale one-off status/monitor/wait/verify shell helper; not referenced by active run, configs, or systemd units. |
| `scripts/_disambig_trt.sh` | Stale one-off status/monitor/wait/verify shell helper; not referenced by active run, configs, or systemd units. |
| `scripts/_equiv_check.py` | Stale one-off benchmark/probe/analysis Python script; not referenced by active run or configs. |
| `scripts/_eval_pathbench.py` | Stale one-off benchmark/probe/analysis Python script; not referenced by active run or configs. |
| `scripts/_fp16_input_gate.py` | Stale one-off benchmark/probe/analysis Python script; not referenced by active run or configs. |
| `scripts/_gate_gpu_free.sh` | Stale one-off status/monitor/wait/verify shell helper; not referenced by active run, configs, or systemd units. |
| `scripts/_gpu_apps.sh` | Stale one-off status/monitor/wait/verify shell helper; not referenced by active run, configs, or systemd units. |
| `scripts/_leak_probe.py` | Stale one-off benchmark/probe/analysis Python script; not referenced by active run or configs. |
| `scripts/_lean_status.sh` | Stale one-off status/monitor/wait/verify shell helper; not referenced by active run, configs, or systemd units. |
| `scripts/_loss_decomp.py` | Stale one-off benchmark/probe/analysis Python script; not referenced by active run or configs. |
| `scripts/_mcts_selfplay_probe.py` | Stale one-off benchmark/probe/analysis Python script; not referenced by active run or configs. |
| `scripts/_mem_breakdown.sh` | Stale one-off status/monitor/wait/verify shell helper; not referenced by active run, configs, or systemd units. |
| `scripts/_oneshot_status.sh` | Stale one-off status/monitor/wait/verify shell helper; not referenced by active run, configs, or systemd units. |
| `scripts/_overnight_monitor.sh` | Stale one-off status/monitor/wait/verify shell helper; not referenced by active run, configs, or systemd units. |
| `scripts/_ram_check.sh` | Stale one-off status/monitor/wait/verify shell helper; not referenced by active run, configs, or systemd units. |
| `scripts/_ram_monitor.sh` | Stale one-off status/monitor/wait/verify shell helper; not referenced by active run, configs, or systemd units. |
| `scripts/_ram_trend.sh` | Stale one-off status/monitor/wait/verify shell helper; not referenced by active run, configs, or systemd units. |
| `scripts/_rebuild_hexo_models.sh` | Stale one-off status/monitor/wait/verify shell helper; not referenced by active run, configs, or systemd units. |
| `scripts/_rebuild_hexo_models_clean.sh` | Stale one-off status/monitor/wait/verify shell helper; not referenced by active run, configs, or systemd units. |
| `scripts/_restart_supervisor.sh` | Stale one-off status/monitor/wait/verify shell helper; not referenced by active run, configs, or systemd units. |
| `scripts/_run_dense_cnn_tests.sh` | Stale one-off status/monitor/wait/verify shell helper; not referenced by active run, configs, or systemd units. |
| `scripts/_run_equiv.sh` | Stale one-off status/monitor/wait/verify shell helper; not referenced by active run, configs, or systemd units. |
| `scripts/_run_gate.sh` | Stale one-off status/monitor/wait/verify shell helper; not referenced by active run, configs, or systemd units. |
| `scripts/_run_leak_probe.sh` | Stale one-off status/monitor/wait/verify shell helper; not referenced by active run, configs, or systemd units. |
| `scripts/_run_sample_probe.sh` | Stale one-off status/monitor/wait/verify shell helper; not referenced by active run, configs, or systemd units. |
| `scripts/_run_tree_probe.sh` | Stale one-off status/monitor/wait/verify shell helper; not referenced by active run, configs, or systemd units. |
| `scripts/_sample_size_probe.py` | Stale one-off benchmark/probe/analysis Python script; not referenced by active run or configs. |
| `scripts/_smoke_tempdecay.sh` | Stale one-off status/monitor/wait/verify shell helper; not referenced by active run, configs, or systemd units. |
| `scripts/_state_report.sh` | Stale one-off status/monitor/wait/verify shell helper; not referenced by active run, configs, or systemd units. |
| `scripts/_status_all.sh` | Stale one-off status/monitor/wait/verify shell helper; not referenced by active run, configs, or systemd units. |
| `scripts/_status_tight.sh` | Stale one-off status/monitor/wait/verify shell helper; not referenced by active run, configs, or systemd units. |
| `scripts/_target_trend.py` | Stale one-off benchmark/probe/analysis Python script; not referenced by active run or configs. |
| `scripts/_tree_growth_probe.py` | Stale one-off benchmark/probe/analysis Python script; not referenced by active run or configs. |
| `scripts/_validate_hexgt_candidates.py` | Stale one-off benchmark/probe/analysis Python script; not referenced by active run or configs. |
| `scripts/_validate_hexgt_expand.py` | Stale one-off benchmark/probe/analysis Python script; not referenced by active run or configs. |
| `scripts/_verify_changes.sh` | Stale one-off status/monitor/wait/verify shell helper; not referenced by active run, configs, or systemd units. |
| `scripts/_verify_live.sh` | Stale one-off status/monitor/wait/verify shell helper; not referenced by active run, configs, or systemd units. |
| `scripts/_verify_posps.sh` | Stale one-off status/monitor/wait/verify shell helper; not referenced by active run, configs, or systemd units. |
| `scripts/_wait_epoch1_boundary.sh` | Stale one-off status/monitor/wait/verify shell helper; not referenced by active run, configs, or systemd units. |
| `scripts/_wait_k2_games.sh` | Stale one-off status/monitor/wait/verify shell helper; not referenced by active run, configs, or systemd units. |
| `scripts/_wait_trt_adopt.sh` | Stale one-off status/monitor/wait/verify shell helper; not referenced by active run, configs, or systemd units. |
| `scripts/_watch_fp_epoch2.sh` | Stale one-off status/monitor/wait/verify shell helper; not referenced by active run, configs, or systemd units. |
| `scripts/goal_benchmark.py` | Stale one-off benchmark/probe/analysis Python script; not referenced by active run or configs. |
| `scripts/run_model1_wsl_smoke.sh` | Stale one-off status/monitor/wait/verify shell helper; not referenced by active run, configs, or systemd units. |
| `scripts/start_model1_training.ps1` | Stale Windows-side launch/watch helper for a superseded model run. |
| `scripts/watch_model1_resources.ps1` | Stale Windows-side launch/watch helper for a superseded model run. |
| `scripts/_feat_probe.py` | Stale one-off benchmark/probe/analysis Python script; not referenced by active run or configs. |
| `scripts/_game_analysis.py` | Stale one-off benchmark/probe/analysis Python script; not referenced by active run or configs. |
| `scripts/_perf_512.py` | Stale one-off benchmark/probe/analysis Python script; not referenced by active run or configs. |
| `scripts/_sweep_active.py` | Stale one-off benchmark/probe/analysis Python script; not referenced by active run or configs. |
| `scripts/_truelen.py` | Stale one-off benchmark/probe/analysis Python script; not referenced by active run or configs. |
| `scripts/_perf_tss_ab.py` | Stale one-off benchmark/probe/analysis Python script; not referenced by active run or configs. |
| `scripts/_pretrain_model3.py` | Stale one-off benchmark/probe/analysis Python script; not referenced by active run or configs. |
| `scripts/_tss_smoke.py` | One-off TSS verification/smoke probe; superseded. |
| `scripts/_tss_verify.py` | One-off TSS verification/smoke probe; superseded. |
| `scripts/_tss_verify2.py` | One-off TSS verification/smoke probe; superseded. |
| `scripts/_tss_verify_c.py` | One-off TSS verification/smoke probe; superseded. |
| `scripts/_vram_deep.py` | One-off VRAM diagnostic probe; superseded by current session tooling. |
| `scripts/_vram_recompile_probe.py` | One-off VRAM diagnostic probe; superseded by current session tooling. |
| `scripts/_vram_snapshot_attr.py` | One-off VRAM diagnostic probe; superseded by current session tooling. |
| `scripts/_rl_supervise.sh` | Superseded RESTNET / dense_cnn / RL launch+probe script from a parked model lineage; not referenced by active run or configs. |
| `scripts/_rl_launch_main3.sh` | Superseded RESTNET / dense_cnn / RL launch+probe script from a parked model lineage; not referenced by active run or configs. |
| `scripts/_dc_launch_main1.sh` | Superseded RESTNET / dense_cnn / RL launch+probe script from a parked model lineage; not referenced by active run or configs. |
| `scripts/_dc_supervise_main1.sh` | Superseded RESTNET / dense_cnn / RL launch+probe script from a parked model lineage; not referenced by active run or configs. |
| `scripts/_count_rows.sh` | Stale one-off status/monitor/wait/verify shell helper; not referenced by active run, configs, or systemd units. |
| `scripts/_dc_discover.sh` | Superseded RESTNET / dense_cnn / RL launch+probe script from a parked model lineage; not referenced by active run or configs. |
| `scripts/_dc_restnet_launch_main1.sh` | Superseded RESTNET / dense_cnn / RL launch+probe script from a parked model lineage; not referenced by active run or configs. |
| `scripts/_dc_restnet_supervise_main1.sh` | Superseded RESTNET / dense_cnn / RL launch+probe script from a parked model lineage; not referenced by active run or configs. |
| `scripts/_dc_stop_main1.sh` | Superseded RESTNET / dense_cnn / RL launch+probe script from a parked model lineage; not referenced by active run or configs. |
| `scripts/_hf_download.sh` | Stale one-off status/monitor/wait/verify shell helper; not referenced by active run, configs, or systemd units. |
| `scripts/_hf_probe.sh` | Stale one-off status/monitor/wait/verify shell helper; not referenced by active run, configs, or systemd units. |
| `scripts/_kv_gather_bench.py` | Stale one-off benchmark/probe/analysis Python script; not referenced by active run or configs. |
| `scripts/_overnight_check.sh` | Stale one-off status/monitor/wait/verify shell helper; not referenced by active run, configs, or systemd units. |
| `scripts/_restnet_attention_scope_perf.py` | Superseded RESTNET / dense_cnn / RL launch+probe script from a parked model lineage; not referenced by active run or configs. |
| `scripts/_restnet_bias_bench.py` | Superseded RESTNET / dense_cnn / RL launch+probe script from a parked model lineage; not referenced by active run or configs. |
| `scripts/_restnet_bounce.sh` | Superseded RESTNET / dense_cnn / RL launch+probe script from a parked model lineage; not referenced by active run or configs. |
| `scripts/_restnet_crop_coverage.py` | Superseded RESTNET / dense_cnn / RL launch+probe script from a parked model lineage; not referenced by active run or configs. |
| `scripts/_restnet_epoch2_diag.sh` | Superseded RESTNET / dense_cnn / RL launch+probe script from a parked model lineage; not referenced by active run or configs. |
| `scripts/_restnet_epoch_poll.sh` | Superseded RESTNET / dense_cnn / RL launch+probe script from a parked model lineage; not referenced by active run or configs. |
| `scripts/_restnet_gpe_verify.sh` | Superseded RESTNET / dense_cnn / RL launch+probe script from a parked model lineage; not referenced by active run or configs. |
| `scripts/_restnet_gpu_sanity.py` | Superseded RESTNET / dense_cnn / RL launch+probe script from a parked model lineage; not referenced by active run or configs. |
| `scripts/_restnet_loss_check.sh` | Superseded RESTNET / dense_cnn / RL launch+probe script from a parked model lineage; not referenced by active run or configs. |
| `scripts/_restnet_migrate_heads_v2.py` | Superseded RESTNET / dense_cnn / RL launch+probe script from a parked model lineage; not referenced by active run or configs. |
| `scripts/_restnet_prefit.sh` | Superseded RESTNET / dense_cnn / RL launch+probe script from a parked model lineage; not referenced by active run or configs. |
| `scripts/_restnet_prefit_monitor.sh` | Superseded RESTNET / dense_cnn / RL launch+probe script from a parked model lineage; not referenced by active run or configs. |
| `scripts/_restnet_prefit_poll.sh` | Superseded RESTNET / dense_cnn / RL launch+probe script from a parked model lineage; not referenced by active run or configs. |
| `scripts/_restnet_prefit_run.sh` | Superseded RESTNET / dense_cnn / RL launch+probe script from a parked model lineage; not referenced by active run or configs. |
| `scripts/_restnet_resume_check.sh` | Superseded RESTNET / dense_cnn / RL launch+probe script from a parked model lineage; not referenced by active run or configs. |
| `scripts/_restnet_run_status.sh` | Superseded RESTNET / dense_cnn / RL launch+probe script from a parked model lineage; not referenced by active run or configs. |
| `scripts/_restnet_run_verify.sh` | Superseded RESTNET / dense_cnn / RL launch+probe script from a parked model lineage; not referenced by active run or configs. |
| `scripts/_restnet_train_poll.sh` | Superseded RESTNET / dense_cnn / RL launch+probe script from a parked model lineage; not referenced by active run or configs. |
| `scripts/_restnet_train_verify.sh` | Superseded RESTNET / dense_cnn / RL launch+probe script from a parked model lineage; not referenced by active run or configs. |
| `scripts/_restnet_validate.sh` | Superseded RESTNET / dense_cnn / RL launch+probe script from a parked model lineage; not referenced by active run or configs. |
| `scripts/_run_ab_gate.sh` | Stale one-off status/monitor/wait/verify shell helper; not referenced by active run, configs, or systemd units. |
| `scripts/_best_config_probe.py` | Stale one-off benchmark/probe/analysis Python script; not referenced by active run or configs. |
| `scripts/_bounce_restnet_overlap.sh` | Stale one-off status/monitor/wait/verify shell helper; not referenced by active run, configs, or systemd units. |
| `scripts/_cachelimit_check.py` | Stale one-off benchmark/probe/analysis Python script; not referenced by active run or configs. |
| `scripts/_compile_trace_check.py` | Stale one-off benchmark/probe/analysis Python script; not referenced by active run or configs. |
| `scripts/_epoch_trend.py` | Stale one-off benchmark/probe/analysis Python script; not referenced by active run or configs. |
| `scripts/_explore_run.sh` | Stale one-off status/monitor/wait/verify shell helper; not referenced by active run, configs, or systemd units. |
| `scripts/_explore_stage1.py` | Stale one-off benchmark/probe/analysis Python script; not referenced by active run or configs. |
| `scripts/_explore_stage2.py` | Stale one-off benchmark/probe/analysis Python script; not referenced by active run or configs. |
| `scripts/_forward_lever_bench.py` | Stale one-off benchmark/probe/analysis Python script; not referenced by active run or configs. |
| `scripts/_pretrain_hexgnn.py` | Stale one-off benchmark/probe/analysis Python script; not referenced by active run or configs. |
| `scripts/_rl_launch_hexgnn.sh` | Superseded RESTNET / dense_cnn / RL launch+probe script from a parked model lineage; not referenced by active run or configs. |
| `scripts/_rl_supervise_hexgnn.sh` | Superseded RESTNET / dense_cnn / RL launch+probe script from a parked model lineage; not referenced by active run or configs. |
| `docs/analysis/RESTNET_EXPLORATION_KNOBS.md` | Stale RESTNET exploration analysis doc; superseded, references parked lineage. |
| `docs/analysis/RESTNET_OPENING_DIVERSITY.md` | Stale RESTNET exploration analysis doc; superseded, references parked lineage. |
| `analysis/01_baseline.py` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/02_bf16.py` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/03_evaluator_attribution.py` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/04_wsl_compile.py` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/05_selfplay_attribution.py` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/06_native_batchsweep.py` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/07_trt.py` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/08_verify_wsl.py` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/09_callback_attr_smallbatch.py` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/_bench_common.py` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/_deep_model_analysis.py` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/_deep_sample_review.py` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/_diag_exploration.py` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/_eval_game_peek.py` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/_health_watch_96x8.sh` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/_measure_train_96x8.sh` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/_mon_96x8.sh` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/_partial_spill_check.py` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/_quality_extract.py` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/_results_attribution.json` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/_results_baseline.json` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/_results_bf16.json` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/_results_callback_attr_small.json` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/_results_native_sweep.json` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/_results_selfplay_attribution.json` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/_results_trt.json` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/_results_verify.json` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/_results_wsl_compile.json` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/_setup_portproxy.ps1` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/_sim_resolution_results.json` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/_sim_resolution_test.py` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/_tmp_opening_check.py` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/_verify_frontend_efficiency.py` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/_verify_readvalue.py` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/_wait_to_epoch10.sh` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/a1_throughput_bench.py` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/a7_autotune_bench.py` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/aligned_a1.json` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/aligned_serial.json` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/epoch_timeline_summary.json` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/epoch_timings.json` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/evaluator_microbench.py` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/evaluator_microbench_summary.json` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/forward_fp_bench.py` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/gpu_microbench.py` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/gpu_microbench_summary.json` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/inference_backends/__init__.py` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/inference_backends/bench_harness.py` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/inference_backends/compile_variant.py` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/mcts_aligned_diff.py` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/mcts_aligned_harness.py` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/mcts_baseline_pre_a1.json` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/mcts_equiv_harness.py` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/mcts_microbench.py` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/mcts_microbench_summary.json` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/mcts_post_a1_fixed.json` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/p0_loadonce_microbench.py` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/parse_epoch_timings.py` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/parse_selfplay_diag.py` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/phase1_logged_entropy.py` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/phase1_summary.json` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/phase2_raw_head_inference.py` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/phase2_summary.json` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/phase3_target_vs_pred.py` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/phase4_sim_budget.py` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/phase4_vram_probe.py` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/reconstruct_epoch_timeline.py` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/root_parallel_bench.py` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/shuffle_mem_probe.py` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/shuffle_mem_summary.json` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/single_game_latency_bench.py` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/throughput_understanding/_prod_trt_check.py` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/throughput_understanding/_trt_failloud_test.py` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/throughput_understanding/_trt_gate_check.py` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/throughput_understanding/_trt_reliability_repro.py` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/throughput_understanding/_tu1_full_epoch.json` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/throughput_understanding/_tu2_vbatch_sweep.json` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/throughput_understanding/_tu3_gpu_util.json` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/throughput_understanding/_tu8_config_posps.json` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/throughput_understanding/_tv1_trt_move_agreement.json` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/throughput_understanding/_tv1_trt_move_agreement_bf16.json` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/throughput_understanding/_tv1_trt_move_agreement_fp16.json` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/throughput_understanding/_tv4_sealbot_ab.json` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/throughput_understanding/_tv5_regret.json` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/throughput_understanding/_vbatch_quality_ab.json` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/throughput_understanding/tu1_full_epoch.py` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/throughput_understanding/tu2_vbatch_sweep.py` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/throughput_understanding/tu3_gpu_util.py` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/throughput_understanding/tu5_concurrency.py` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/throughput_understanding/tu7_posps_table.py` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/throughput_understanding/tu8_config_posps.py` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/throughput_understanding/tv1_trt_move_agreement.py` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/throughput_understanding/tv2_pinpoint_overflow.py` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/throughput_understanding/tv4_sealbot_ab.py` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/throughput_understanding/tv5_regret.py` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/throughput_understanding/vbatch_quality_ab.py` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/throughput_understanding/verify_bucketing.py` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/train_microbench.py` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/train_microbench_summary.json` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/train_step_components.py` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/train_step_components_summary.json` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/train_step_pipeline.py` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/train_step_pipeline_summary.json` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/train_step_reconcile.py` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |
| `analysis/train_step_reconcile_summary.json` | Dead-end / completed throughput investigation probe, microbench, or result JSON; not referenced by active source (exploration_diversity.py kept). |

## Removed junk (regenerable scratch, not archived)

These were deleted (untracked scratch outputs / bytecode caches), except _tmp_t.txt which was git-tracked and removed via `git rm`. .gitignore updated with a `_tmp_*` pattern.

- `scripts/_value_head_review.log`
- `scripts/_grid_runner.log`
- `scripts/_m2_probe_m1_ep11.log`
- `scripts/_m2_probe_m1_ep35.log`
- `scripts/_m2_probe_m2_ep11.log`
- `scripts/_m2_probe_m2_ep5.log`
- `scripts/_policy_diffuseness_probe.log`
- `scripts/_smoke_8901.log`
- `scripts/_value_head_review_ep35.log`
- `scripts/_wf_m3_v_m3-ep6-winner-skew_probe_ep6.log`
- `_dc_gate.out`
- `_hexgnn_dirichlet.out`
- `_hexgnn_explore.out`
- `_prof.out`
- `_spy.err`
- `_tmp_check2.out`
- `recomp.err.log`
- `recomp.out.log`
- `_tmp_t.txt (git rm)`
- `__pycache__/ (dir)`
- `_expl/ (dir)`
