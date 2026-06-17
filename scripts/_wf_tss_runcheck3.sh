#!/usr/bin/env bash
# Scratch: why did the main_4 driver + supervisor die ~18:26Z?
set -u
out=/mnt/e/Hexo-BotTrainer-hexgt/scripts/_wf_tss_runcheck3.txt
cd /mnt/e/Hexo-BotTrainer/runs/dense_cnn_restnet_main_4
{
  echo "=== $(date -u) ==="
  echo "--- all train logs in run dir ---"
  ls -la train.*.log 2>/dev/null
  echo "--- tail of the 17:10 driver log ---"
  tail -50 train.20260612_171048.out.log 2>/dev/null || echo "(missing)"
  echo "--- kernel OOM / kill traces ---"
  dmesg -T 2>/dev/null | grep -iE "oom|killed process|out of memory" | tail -15 || echo "(dmesg unavailable)"
  echo "--- memory now ---"
  free -h
  echo "--- recent journal kills (if systemd) ---"
  journalctl --since "2026-06-12 17:00" 2>/dev/null | grep -iE "oom|kill" | tail -10 || echo "(no journal)"
} > "$out" 2>&1
echo written
