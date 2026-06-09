#!/bin/bash
export PATH="$HOME/.cargo/bin:$PATH"
cd /mnt/e/Hexo-BotTrainer-hexgt || exit 11
source /root/.venvs/hexgt-build/bin/activate
CKPT=${CKPT:-/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_bc/hexgt_bc_step006009.pt}
python -u scripts/_head_to_head.py --hexgt-ckpt "$CKPT" \
  --games "${GAMES:-40}" --visits "${VISITS:-200}" --max-actions "${MAXA:-512}" \
  ${SEALBOT:+--sealbot} 2>&1 | grep -vE "UserWarning|frombuffer|warnings summary|W0601|torch/_dynamo|torch/_inductor|capture_scalar|Graph break|consider setting|TORCHDYNAMO|node_type is not|return self|self\._validate|self\._heads|^$" | tail -40
