RD=/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgnn_rl_main1
echo "=== recompile / guard / dynamic warnings in log ==="
grep -icE "recompil|guard|dynamo|cudagraph|falling back|graph break" "$RD/rl_train.log" 2>/dev/null
grep -iE "recompil|graph break|falling back" "$RD/rl_train.log" 2>/dev/null | tail -5
echo "=== any HEXGT_EVAL_PIPELINE / trace env in supervise boot ==="
grep -iE "PIPELINE|TRACE|compile" "$RD/rl_train.log" 2>/dev/null | head -3
echo "=== rl_train.log very tail (live) ==="
tail -4 "$RD/rl_train.log"
