#!/bin/bash
# 25s py-spy sample of the prefit main process; prints top functions.
MAIN=$(ps aux | grep '[_]hexfield_prefit.py' | grep -v grep | awk '{print $2}' | head -1)
echo "profiling pid $MAIN"
/root/.venvs/hexfield-dev/bin/py-spy record --pid "$MAIN" --duration 25 --rate 50 \
  --format speedscope --output /tmp/prefit_profile.json 2>&1 | tail -2
/root/.venvs/hexfield-dev/bin/py-spy top --pid "$MAIN" --duration 12 2>/dev/null | head -25 || true
