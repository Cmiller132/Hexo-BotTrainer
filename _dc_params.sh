export VIRTUAL_ENV=/root/.venvs/hexgt-build; export PATH="$VIRTUAL_ENV/bin:$PATH"; export CUDA_VISIBLE_DEVICES=
R=/mnt/e/Hexo-BotTrainer-hexgt
export PYTHONPATH="$R/packages/hexo_engine/python:$R/packages/hexo_utils/python:$R/packages/hexo_runner/python:$R/packages/hexo_train/python:$R/packages/hexo_models/python"
$VIRTUAL_ENV/bin/python - <<'PY' 2>/dev/null
import json, torch
from hexo_models.dense_cnn.architecture import build_network if False else None
try:
    from hexo_models.dense_cnn import architecture as A
    # build a 96x8 net the same way the plugin does
    net = None
    for fn in ("build_network","DenseCNN","DenseCNNNetwork","Model","build_model"):
        if hasattr(A, fn):
            obj=getattr(A,fn)
            try:
                net = obj(channels=96, residual_blocks=8) if callable(obj) else None
            except Exception: 
                try: net=obj(96,8)
                except Exception: net=None
            if net is not None and hasattr(net,"parameters"): break
    if net is not None and hasattr(net,"parameters"):
        print("dense_cnn 96x8 params:", f"{sum(p.numel() for p in net.parameters()):,}")
    else:
        print("could not auto-build; arch fns:", [x for x in dir(A) if not x.startswith('_')][:12])
except Exception as e:
    print("param compute failed:", repr(e))
PY
