#!/bin/bash
set -e
python3 -m venv /root/.venvs/order-dev
/root/.venvs/order-dev/bin/pip install -q /root/order-wheels/hexfield_eq-0.1.0-cp312-cp312-manylinux_2_34_x86_64.whl numpy
SP=$(ls -d /root/.venvs/order-dev/lib/python3*/site-packages)
printf '/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_engine/python\n' > "$SP/hexo_engine.pth"
printf '/mnt/e/hexo-bot/packages/hexo_utils/python\n' > "$SP/hexo_utils.pth"
/root/.venvs/order-dev/bin/python -c 'import hexo_engine; from hexfield_eq import _rust; m=_rust.hexfield_eq_solver_manifest(500,0,False,False,True,False,"prior"); print("ordering echo:", m.get("ordering"))'
