for i in 1 2 3 4 5; do
  alive=$(ps -p 37763 -o pid= 2>/dev/null)
  apps=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null)
  echo "try $i: ps37763=[$alive] gpu_apps=[$apps]"
  if [ -z "$alive" ] && { [ -z "$apps" ] || ! echo "$apps" | grep -q 37763; }; then echo "GPU CLEAR"; break; fi
  sleep 3
done
echo "=== final GPU mem ==="
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null
