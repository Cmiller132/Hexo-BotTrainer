echo "nproc=$(nproc)  (cores/threads)"
pid=$(pgrep -f 'python.*_rl_train.py' | head -1)
echo "driver_pid=$pid"
for k in 1 2 3 4 5; do
  cpu=$(ps -o %cpu= -p "$pid" | tr -d ' ')
  rss=$(ps -o rss= -p "$pid" | tr -d ' ')
  la=$(cut -d' ' -f1-3 /proc/loadavg)
  mem=$(free -m | awk '/Mem:/{printf "used=%dMB avail=%dMB", $3, $7}')
  gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | tr ',' ' ')
  echo "  driverCPU=${cpu}%  driverRSS=$((rss/1024))MB  load=[$la]  $mem  gpu_util/mem=${gpu}"
  sleep 4
done
