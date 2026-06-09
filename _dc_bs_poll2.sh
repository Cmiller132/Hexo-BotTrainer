OUTDIR=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_rl_main1_prefit
BLOG="$OUTDIR/bootstrap.log"
for i in $(seq 1 40); do
  if [ -f "$OUTDIR/bootstrap_sealbot_prefit.pt" ] || grep -qiE "saved|verified|strict load|Traceback|Error" "$BLOG" 2>/dev/null; then break; fi
  sleep 12
done
echo "=== prefit / save / loss lines ==="
grep -iE "prefit|epoch|loss|policy|value|saved|verif|strict|param" "$BLOG" 2>/dev/null | tail -20
echo "=== checkpoint ==="
ls -la "$OUTDIR/bootstrap_sealbot_prefit.pt" 2>/dev/null || echo "(not saved yet)"
echo "=== errors ==="; grep -iE "Traceback|Error|Exception|FAILED" "$BLOG" 2>/dev/null | head -4 || echo "(none)"
