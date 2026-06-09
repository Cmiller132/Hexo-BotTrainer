#!/usr/bin/env bash
for run in dense_cnn_model1_target_96x8 hexgt_rl_main2; do
  curl -s "localhost:8080/api/training/run?name=$run" > "/tmp/$run.json"
done
python3 - <<'PY'
import json
def summ(f):
    d=json.load(open(f)); out={}
    for k,v in d.items():
        if isinstance(v,list): out[k]='list[%d]'%len(v)
        elif isinstance(v,dict): out[k]='dict{%s}'%','.join(list(v.keys())[:6])
        else: out[k]=repr(v)[:46]
    return out
a=summ('/tmp/dense_cnn_model1_target_96x8.json'); b=summ('/tmp/hexgt_rl_main2.json')
keys=sorted(set(a)|set(b))
print('%-22s | %-34s | %-34s'%('FIELD','dense_96x8 (working)','hexgt_rl_main2'))
print('-'*96)
for k in keys:
    print('%-22s | %-34s | %-34s'%(k, a.get(k,'(absent)'), b.get(k,'(absent)')))
PY
