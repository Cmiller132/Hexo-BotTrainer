V=/root/.venvs/hexgt-build/bin/python
$V - <<'PY'
import urllib.request,json
for r in ("dense_cnn_rl_main1","hexgnn_rl_main1"):
    d=json.loads(urllib.request.urlopen(f"http://127.0.0.1:8080/api/training/run?name={r}",timeout=10).read())
    eh=d.get('epoch_history',[])
    sp=(eh[-1].get('selfplay') if eh else {}) or {}
    keys=[k for k in ('game_length_mean','game_length_median','game_length_max','win_p0_fraction','win_p1_fraction','draw_fraction','decisive_fraction') if k in sp and sp[k] is not None]
    print(f"{r}: epoch_history[{len(eh)}] last-selfplay game-stat fields present={keys}")
PY
