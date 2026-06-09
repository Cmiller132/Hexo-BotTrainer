#!/usr/bin/env bash
set -euo pipefail
export VIRTUAL_ENV=/root/.venvs/hexgt-build
export PATH="$VIRTUAL_ENV/bin:$PATH"
cd /mnt/e/Hexo-BotTrainer-hexgt
export PYTHONPATH="$PWD/packages/hexgnn/python"
python3 - <<'PY'
from hexgnn.constants import NODE_FEATURE_DIM, EDGE_ATTR_DIM
tn=109343; te=573486; tc=105144
nf=tn*NODE_FEATURE_DIM*4; ea=te*EDGE_ATTR_DIM*4
idx32=(2*tn+2*te+2*te+2*tc)*4; idx64=idx32*2
print('NODE_FEATURE_DIM',NODE_FEATURE_DIM,'EDGE_ATTR_DIM',EDGE_ATTR_DIM)
print('node_feat MB %.2f'%(nf/1e6),'edge_attr MB %.2f'%(ea/1e6))
print('idx32 MB %.2f'%(idx32/1e6),'idx64 MB %.2f'%(idx64/1e6))
print('total wire BEFORE(int64) MB %.2f'%((nf+ea+idx64)/1e6))
print('total wire AFTER (int32) MB %.2f'%((nf+ea+idx32)/1e6))
PY
