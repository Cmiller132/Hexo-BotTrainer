#!/usr/bin/env bash
# Overnight watcher for the dense_cnn target_96x6 run. Emits ONE line per event:
#  - supervisor lifecycle (RELAUNCH/EXIT/halt/breaker-near-trip/completed)
#  - new epoch checkpoint (PROGRESS — so silence != hidden stall)
#  - trainer crash signatures (Traceback/CUDA/OOM/TRT-not-adopted) from the
#    NEWEST trainer .err/.out log, following rotation across relaunches
#  - halt / completed flag files
# Read via the Windows E: path (same files WSL writes under /mnt/e).
D="E:/Hexo-BotTrainer/runs/dense_cnn_model1_target_96x6"
DI="$D/diagnostics"
SUPLOG="$DI/supervisor_wsl.log"
SUP_PAT='RELAUNCH|EXIT pid|halt:|HALT|consecFast=[3-9]|noProgress=[4-9]|completed flag|SUPERVISOR start'
ERR_PAT='Traceback|CUDA error|CUDA out of memory|OutOfMemory|torch.*OutOfMemory|Killed|NOT ADOPTED|TRTAdoptError|FATAL|RuntimeError|AssertionError'

ckpt_epoch() { ls "$D"/checkpoints/epoch_*.pt 2>/dev/null | sed -E 's/.*epoch_0*([0-9]+)\.pt/\1/' | sort -n | tail -1; }

sup=$(wc -l < "$SUPLOG" 2>/dev/null || echo 0)
lastck=$(ckpt_epoch); lastck=${lastck:-0}
errf=""; errn=0; outf=""; outn=0
echo "[monitor armed $(date -u +%FT%TZ)] sup_lines=$sup last_ckpt_epoch=$lastck — watching supervisor+trainer logs+checkpoints"

while true; do
  # 1) supervisor lifecycle: new matching lines
  cur=$(wc -l < "$SUPLOG" 2>/dev/null || echo 0)
  if [ "$cur" -gt "$sup" ]; then
    sed -n "$((sup+1)),${cur}p" "$SUPLOG" 2>/dev/null | grep -E "$SUP_PAT" | sed 's/^/[SUP] /'
    sup=$cur
  fi
  # 2) progress: new epoch checkpoint
  ck=$(ckpt_epoch); ck=${ck:-0}
  if [ "$ck" -gt "$lastck" ]; then echo "[PROGRESS $(date -u +%H:%M:%SZ)] epoch $ck checkpoint written"; lastck=$ck; fi
  # 3) trainer crash signatures from the NEWEST err log (reset on rotation)
  nf=$(ls -t "$DI"/trainer.*.err.log 2>/dev/null | head -1)
  if [ "$nf" != "$errf" ]; then errf="$nf"; errn=0; fi
  if [ -n "$errf" ]; then
    en=$(wc -l < "$errf" 2>/dev/null || echo 0)
    if [ "$en" -gt "$errn" ]; then
      sed -n "$((errn+1)),${en}p" "$errf" 2>/dev/null | grep -E "$ERR_PAT" | sed 's/^/[ERR] /'
      errn=$en
    fi
  fi
  # 4) TRT fail-loud lines land on stdout (out log)
  of=$(ls -t "$DI"/trainer.*.out.log 2>/dev/null | head -1)
  if [ "$of" != "$outf" ]; then outf="$of"; outn=0; fi
  if [ -n "$outf" ]; then
    on=$(wc -l < "$outf" 2>/dev/null || echo 0)
    if [ "$on" -gt "$outn" ]; then
      sed -n "$((outn+1)),${on}p" "$outf" 2>/dev/null | grep -E 'NOT ADOPTED|TRTAdoptError|FATAL' | sed 's/^/[TRT] /'
      outn=$on
    fi
  fi
  # 5) terminal flags
  [ -f "$DI/supervisor_halted.flag" ]    && echo "[HALT $(date -u +%H:%M:%SZ)] $(cat "$DI/supervisor_halted.flag" 2>/dev/null | head -1)"
  [ -f "$DI/supervisor_completed.flag" ] && echo "[DONE $(date -u +%H:%M:%SZ)] run completed flag present"
  sleep 60
done
