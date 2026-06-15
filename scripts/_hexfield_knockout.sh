#!/bin/bash
cd /mnt/e/Hexo-BotTrainer-hexgt
bash scripts/_rebuild_hexfield.sh > /dev/null 2>&1
echo "baseline (fpu 0.2, vloss 1.0):"
/root/.venvs/hexgt-build/bin/python scripts/_hexfield_reuse_diff.py 1 notss 0.2 1.0 2>&1 | head -3
echo "fpu=0:"
/root/.venvs/hexgt-build/bin/python scripts/_hexfield_reuse_diff.py 1 notss 0.0 1.0 2>&1 | head -3
echo "vloss=0:"
/root/.venvs/hexgt-build/bin/python scripts/_hexfield_reuse_diff.py 1 notss 0.2 0.0 2>&1 | head -3
echo "both 0:"
/root/.venvs/hexgt-build/bin/python scripts/_hexfield_reuse_diff.py 1 notss 0.0 0.0 2>&1 | head -3
