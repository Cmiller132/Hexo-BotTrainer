#!/bin/bash
export PATH="$HOME/.cargo/bin:$PATH"; cd /mnt/e/Hexo-BotTrainer-hexgt || exit 11
source /root/.venvs/hexgt-build/bin/activate
python _dumpcfg.py 2>&1 | grep -vE "UserWarning|frombuffer|writable|warnings.warn"
