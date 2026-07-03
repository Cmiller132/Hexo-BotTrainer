#!/usr/bin/env bash
# Install (not start) hexfield-supervisor-7 into systemd.
set -eu
cp /mnt/e/Hexo-BotTrainer-hexgt/scripts/systemd/hexfield-supervisor-7.service /etc/systemd/system/
systemctl daemon-reload
systemctl status hexfield-supervisor-7 --no-pager -l | head -4 || true
echo INSTALLED
