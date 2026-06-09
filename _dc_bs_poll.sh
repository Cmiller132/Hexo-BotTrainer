OUTDIR=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_rl_main1_prefit
BLOG="$OUTDIR/bootstrap.log"
for i in $(seq 1 45); do
  if [ -f "$OUTDIR/bootstrap_sealbot_prefit.pt" ] || grep -qiE "saved|verify|strict|done|prefit.*complete|Traceback|Error" "$BLOG" 2>/dev/null; then break; fi
  sleep 12
done
echo "=== bootstrap log tail ==="
grep -ivE "^\s*$" "$BLOG" 2>/dev/null | tail -25
echo "=== prefit checkpoint ==="
ls -la "$OUTDIR/bootstrap_sealbot_prefit.pt" 2>/dev/null || echo "(not saved yet)"
echo "=== errors? ==="
grep -iE "Traceback|Error|Exception|FAILED" "$BLOG" 2>/dev/null | head -4 || echo "(none)"
