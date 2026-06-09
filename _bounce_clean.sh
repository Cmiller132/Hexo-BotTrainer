echo "=== kill stale benchmark group 29620 (pid 37763 _posps_old_main) ==="
kill -9 -29620 2>/dev/null; kill -9 37763 2>/dev/null; echo "rc=$?"
sleep 2
echo "=== any leftover hexgnn/posps python? ==="
ps -eo pid,etime,cmd | grep -iE "_posps|_hexgnn_posps|_rl_train_hexgnn|_fwdbench|_i32_" | grep -v grep || echo "none"
echo "=== GPU compute apps now (should be empty) ==="
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null || echo "(none)"
echo "=== rm stale supervisor.lock; confirm no halt flag ==="
rm -f runs/hexgnn_rl_main1/supervisor.lock
ls runs/hexgnn_rl_main1/supervisor_halted.flag 2>/dev/null && echo "HALT FLAG PRESENT (would block relaunch)" || echo "no halt flag (good)"
echo "=== latest checkpoint (resume target) ==="
ls -t runs/hexgnn_rl_main1/checkpoints/hexgnn_rl_*.pt 2>/dev/null | head -3 | xargs -n1 basename 2>/dev/null
