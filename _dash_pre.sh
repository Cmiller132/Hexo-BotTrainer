echo "=== current :8080 dashboard process (cmdline/env) ==="
ps -eo pid,cmd | grep "[h]exo_frontend.web" | head -1
echo "--- its PYTHONPATH/env (from /proc) ---"
PID=$(ps -eo pid,cmd | grep "[h]exo_frontend.web" | grep -v grep | awk '{print $1}' | head -1)
[ -n "$PID" ] && tr '\0' '\n' < /proc/$PID/environ 2>/dev/null | grep -E "PYTHONPATH|HEXO_DEBUG_RUN_ROOT|CUDA_VISIBLE" | sed 's/^/  /'
echo "--- its cwd ---"; [ -n "$PID" ] && ls -l /proc/$PID/cwd 2>/dev/null | sed 's/^/  /'
echo "=== PREFLIGHT: does the edited web.py import cleanly? ==="
PP=$([ -n "$PID" ] && tr '\0' '\n' < /proc/$PID/environ 2>/dev/null | grep '^PYTHONPATH=' | cut -d= -f2-)
PYTHONPATH="$PP" /root/.venvs/hexgt-build/bin/python -c "import hexo_frontend.web; print('web.py import OK; _loss_buffer_from_training present:', hasattr(hexo_frontend.web,'_loss_buffer_from_training'))" 2>&1 | tail -5
echo "=== UNIT-VERIFY the synthesis (synthetic dense_cnn training block) ==="
PYTHONPATH="$PP" /root/.venvs/hexgt-build/bin/python -c "
import hexo_frontend.web as w
tr={'status':'completed','loss':5.12,'loss_components':{'policy':3.1,'value':1.0,'opp_policy':0.5,'stvalue_1':0.3,'stvalue_4':0.25,'stvalue_8':0.2}}
print('trained ->', w._loss_buffer_from_training(tr))
print('skipped ->', w._loss_buffer_from_training({'status':'skipped','loss':None}))
" 2>&1 | tail -3
