#!/bin/bash
cd /mnt/e/Hexo-BotTrainer-hexgt
export PATH="/root/.cargo/bin:$PATH"
cargo check -p hexfield --features python 2>&1 | tail -40
