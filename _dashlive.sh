V=/root/.venvs/hexgt-build/bin/python
echo "localhost:8080 -> HTTP $(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/ 2>/dev/null)"
$V -c "import urllib.request,json;d=json.loads(urllib.request.urlopen('http://127.0.0.1:8080/api/training/run?name=dense_cnn_rl_main1',timeout=10).read());s=d.get('status',{});print('dense_cnn_rl_main1: stage=',s.get('stage'),'stage_status=',s.get('stage_status'),'current_epoch=',s.get('current_epoch'),'| selfplay_live=',(s.get('selfplay_live') or {}).get('status'))" 2>&1 | tail -2
echo "in runs list: $(curl -s 'http://127.0.0.1:8080/api/training/runs' 2>/dev/null | grep -o dense_cnn_rl_main1 | head -1)"
