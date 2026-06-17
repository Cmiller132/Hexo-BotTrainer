#!/bin/bash
cd /mnt/e/Hexo-BotTrainer-hexgt
for v in A B D C; do
  /root/.venvs/hexgt-build/bin/python scripts/_hexfield_cont_diff.py "$v" 2>&1 | head -4
  echo ---
done
