RD=/mnt/e/Hexo-BotTrainer/runs/dense_cnn_rl_main1
echo "=== params (grep diagnostics/logs/manifest) ==="
grep -rhoiE "parameter[s]?[\": ]+[0-9,]+|param_count[\": ]+[0-9,]+|[0-9,]+ parameters|num_parameters[\": ]+[0-9]+" "$RD"/diagnostics/*.json "$RD"/train.*.out.log "$RD"/manifest.json 2>/dev/null | head -5
grep -rhoiE "\"channels\": *[0-9]+|\"residual_blocks\": *[0-9]+|\"search_visits\": *[0-9]+" "$RD"/diagnostics/config.normalized.json 2>/dev/null | head
echo "=== DASHBOARD: does /api/training/runs show dense_cnn_rl_main1? ==="
