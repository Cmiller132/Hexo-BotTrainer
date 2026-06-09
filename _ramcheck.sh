#!/usr/bin/env bash
python3 - <<'PY'
import resource
from pathlib import Path
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0
base = rss()
# worst case at the 500k cap: ~28 epochs x 256 shards = ~7168 shard paths + weights
shards = []; weights = []
for e in range(28):
    w = 0.9 ** (27 - e)
    for i in range(256):
        shards.append(Path(f'/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main2/selfplay/epoch_{e:06d}_game_{i:04d}.npz'))
        weights.append(w)
after = rss()
print(f'window index: {len(shards)} shard paths + {len(weights)} weights')
print(f'added RSS for the FULL ~500k-pool window index = {after-base:.1f} MB (this is ALL the new RAM; pool stays on disk, read one group at a time)')
PY
echo "=== live free RAM now ==="
awk '/MemAvailable/{printf "RAM free %.2f GB / 27.4\n", $2/1048576}' /proc/meminfo
echo "=== peak RAM driver process (RSS) ==="
p=$(pgrep -f _rl_train.py | head -1)
[ -n "$p" ] && awk '/VmRSS|VmHWM/{print}' /proc/$p/status
