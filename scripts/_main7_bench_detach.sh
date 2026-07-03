#!/usr/bin/env bash
# Detached bench-matrix launch: survives the caller. Kills a prior matrix run
# first (exact cmdline match), then setsid-nohups the matrix.
set -u
OLD=$(pgrep -f '_main7_bench_matrix.sh' || true)
if [ -n "$OLD" ]; then kill $OLD 2>/dev/null; sleep 2; fi
rm -f /tmp/main7_bench_matrix.log /tmp/main7_bench_matrix.done
setsid nohup bash -c 'bash /mnt/e/Hexo-BotTrainer-hexgt/scripts/_main7_bench_matrix.sh > /tmp/main7_bench_matrix.log 2>&1; echo $? > /tmp/main7_bench_matrix.done' >/dev/null 2>&1 < /dev/null &
sleep 2
pgrep -f '_main7_bench_matrix.sh' >/dev/null && echo LAUNCHED || echo FAILED
