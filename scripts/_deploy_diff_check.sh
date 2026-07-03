#!/usr/bin/env bash
# Compare dev-repo HEAD copies of model.py/inference.py against the live
# gumbel worktree files (expected identical or docstring-only drift).
set -u
cd /mnt/e
for f in model.py inference.py; do
  echo "--- $f ---"
  diff <(git -C Hexo-BotTrainer-hexgt show "HEAD:packages/hexfield/python/hexfield/$f") \
       "Hexo-BotTrainer-gumbel/packages/hexfield/python/hexfield/$f" > "/tmp/diff_$f.txt"
  echo "diff lines: $(wc -l < /tmp/diff_$f.txt)"
  head -30 "/tmp/diff_$f.txt"
done
