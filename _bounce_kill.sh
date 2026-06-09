echo "=== identify mystery GPU pid 37763 ==="
ps -o pid,pgid,etime,cmd -p 37763 2>/dev/null || echo "(37763 not a host process — may be a transient)"
echo ""
echo "=== KILL supervisor group (pgid 398 = supervisor 398 + driver 431); bridge 397 kept ==="
kill -9 -398 2>&1; echo "kill -9 -398 rc=$?"
sleep 3
echo "=== survivors check ==="
ps -eo pid,pgid,etime,cmd | grep -iE "_rl_supervise_hexgnn|_rl_train_hexgnn" | grep -v grep || echo "supervisor+driver: GONE"
echo "--- bridge still up? ---"
ps -o pid,etime,cmd -p 397 2>/dev/null | tail -1
echo "=== nvidia-smi compute apps after kill ==="
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null
