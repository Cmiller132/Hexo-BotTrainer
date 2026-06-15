#!/usr/bin/env bash
# Scratch: compile-check the TSS/FPU config gating in the dense_cnn Rust crate.
# cargo check only — never builds/installs the python .so (live-run safe).
set -u
cd /mnt/e/Hexo-BotTrainer-hexgt
export PATH="$HOME/.cargo/bin:$PATH"
cargo check -p hexo_models --features python 2>&1 | tail -30
