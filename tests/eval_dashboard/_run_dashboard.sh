#!/usr/bin/env bash
cd /mnt/e/hexgt-evaldash || exit 1
export PYTHONPATH=packages/hexo_frontend/python:packages/hexfield/python:packages/dense_cnn_restnet/python
export HEXFIELD_SUPPORT_RADIUS=4
PY=/root/.venvs/hexgt-build/bin/python
ok=1
for t in test_eval_dashboard_fixes verify_dashboard_rootcauses verify_d2d3_render_logic _smoke_payload test_app_js_balanced; do
  echo "### $t"
  "$PY" "tests/eval_dashboard/$t.py" || { ok=0; echo "FAIL $t"; }
done
echo "DASH_ALL_OK=$ok"
