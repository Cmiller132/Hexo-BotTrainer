#!/bin/bash
cd /mnt/e/Hexo-BotTrainer-hexgt
bash scripts/_rebuild_hexfield.sh > /dev/null 2>&1
echo "=== variant A with prefetch:"
/root/.venvs/hexgt-build/bin/python scripts/_hexfield_cont_diff.py A 2>&1 | sed -n '2,3p'
echo "=== variant A without prefetch:"
HEXFIELD_NO_PREFETCH=1 /root/.venvs/hexgt-build/bin/python scripts/_hexfield_cont_diff.py A 2>&1 | sed -n '2,3p'
