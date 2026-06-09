echo "=== kill supervisor group 62380 (supervisor+driver); bridge 397 kept ==="
kill -9 -62380 2>&1; echo "rc=$?"
sleep 4
echo "=== survivors? ==="
ps -eo pid,pgid,cmd | grep -E "_rl_supervise_hexgnn|_rl_train_hexgnn" | grep -v grep || echo "supervisor+driver GONE"
echo "--- bridge up? ---"; ps -o pid,etime -p 397 2>/dev/null | tail -1
echo "=== GPU compute apps (wait to clear) ==="
for i in 1 2 3 4 5; do
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
  echo "try $i: [$apps]"
  [ -z "$apps" ] && { echo "GPU CLEAR"; break; }
  sleep 3
done
echo "=== rm stale lock; halt flag? ==="
rm -f runs/hexgnn_rl_main1/supervisor.lock
ls runs/hexgnn_rl_main1/supervisor_halted.flag 2>/dev/null && echo "HALT FLAG PRESENT" || echo "no halt flag (good)"
echo "latest.pt epoch: $(ls -t runs/hexgnn_rl_main1/checkpoints/hexgnn_rl_epoch*.pt 2>/dev/null | head -1 | xargs -n1 basename)"
