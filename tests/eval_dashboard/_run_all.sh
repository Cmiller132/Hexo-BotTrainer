#!/usr/bin/env bash
cd /mnt/e/hexgt-evaldash || exit 1
PP=packages/hexfield/python:packages/dense_cnn_restnet/python
PY=/root/.venvs/hexgt-build/bin/python
ok=1
for t in test_e1_eval_hxr test_e2_anchor_drop test_e3_sealbot_substitution test_e4_radius_annotation _verify_signatures _smoke_live_read; do
  echo "### $t"
  HEXFIELD_SUPPORT_RADIUS=4 PYTHONPATH=$PP $PY "tests/eval_dashboard/$t.py" || { ok=0; echo "FAIL $t"; }
done
echo "ALL_OK=$ok"
