#!/usr/bin/env bash
# Round-2 deploy: copy-stream + train-compile + malloc tunables into the unit;
# virtual_batch_size 16 -> 48 in the live toml. Idempotent. Then restart.
set -eu
UNIT=/etc/systemd/system/hexfield-supervisor-6.service
TOML=/mnt/e/Hexo-BotTrainer-gumbel/configs/hexfield_main_6.toml

if grep -q HEXFIELD_COPY_STREAM "$UNIT"; then
  echo "round-2 flags already present"
else
  sed -i '/^Environment=HEXFIELD_RUST_PACK=1/a \
# Round-2 speedup (2026-07-03, same doc): pinned copy-stream H2D (submit no\
# longer serializes with the GPU), compiled training step (1.30x), glibc\
# malloc tunables (allocator churn was ~40% of serve host time).\
Environment=HEXFIELD_COPY_STREAM=1\
Environment=HEXFIELD_TRAIN_COMPILE=1\
Environment=MALLOC_TRIM_THRESHOLD_=536870912\
Environment=MALLOC_MMAP_THRESHOLD_=536870912\
Environment=MALLOC_TOP_PAD_=134217728' "$UNIT"
  echo "round-2 flags added"
fi

if grep -q '^virtual_batch_size = 48' "$TOML"; then
  echo "vbs already 48"
else
  sed -i 's/^virtual_batch_size = 16$/# 16 -> 48 (2026-07-03): +6% pos\/s; SH round quotas unchanged, only per-slot\n# in-flight depth (staleness within a round grows; barriers stall less).\nvirtual_batch_size = 48/' "$TOML"
  grep -n 'virtual_batch_size' "$TOML"
fi

systemctl daemon-reload
systemctl start hexfield-supervisor-6
sleep 5
systemctl is-active hexfield-supervisor-6
