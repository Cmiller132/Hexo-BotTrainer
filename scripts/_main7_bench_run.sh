#!/usr/bin/env bash
# Bench-matrix runner: full log to /tmp/main7_bench_matrix.log, condensed to stdout.
set -u
bash /mnt/e/Hexo-BotTrainer-hexgt/scripts/_main7_bench_matrix.sh > /tmp/main7_bench_matrix.log 2>&1
rc=$?
grep -E '^=====|^N=|^  \[|ms/fwd|^MATRIX DONE|Error|error|Traceback' /tmp/main7_bench_matrix.log | head -150
exit $rc
