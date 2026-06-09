echo "--- dense_cnn run dirs ---"; ls -d /mnt/e/Hexo-BotTrainer/runs/dense_cnn* 2>/dev/null
echo "--- existing 96x8 prefit (96x8 shape, can't load 64x10 but confirms bootstrap ran) ---"
ls -la /mnt/e/Hexo-BotTrainer/runs/dense_cnn_model1_target_96x8/checkpoints/bootstrap_sealbot_prefit.pt 2>/dev/null || echo "(no 96x8 prefit on disk)"
echo "--- any reusable dense_cnn self-play NPZ shards (96x8 run)? ---"
for d in /mnt/e/Hexo-BotTrainer/runs/dense_cnn_model1_target_96x8; do
  echo "$d/selfplay: $(ls "$d/selfplay"/*.npz 2>/dev/null | wc -l) npz | shuffleddata: $(du -sh "$d/shuffleddata" 2>/dev/null | cut -f1)"
done
echo "--- SealBot present? ---"; ls -d /mnt/e/SealBot /mnt/e/SealBot/best 2>/dev/null | head
