import sys, tomllib, time
sys.path.insert(0,'/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_engine/python')
for p in ('hexo_models','hexo_utils','hexo_runner','hexo_train'):
    sys.path.insert(0, f'/mnt/e/Hexo-BotTrainer-hexgt/packages/{p}/python')
import torch, tempfile
from pathlib import Path
from hexo_models.hexgt.architecture import HexgtNetwork
from hexo_models.hexgt.config import parse_hexgt_config
from hexo_models.hexgt.selfplay import run_selfplay_games
ROOT=Path('/mnt/e/Hexo-BotTrainer-hexgt')
cfg=parse_hexgt_config(tomllib.loads((ROOT/'configs/hexgt_model3.toml').read_text())['model']['config'])
a=cfg.architecture; dev='cuda'
ck=torch.load(ROOT/'runs/hexgt_rl_main3/pretrain/hexgt_model3_pretrain.pt', map_location=dev, weights_only=False)
m=HexgtNetwork(node_feature_dim=a.node_feature_dim,token_dim=a.token_dim,gnn_layers=a.gnn_layers,ctx_layers=a.ctx_layers,
               attention_heads=a.attention_heads,ffn_dim=a.ffn_dim,short_term_value_horizons=a.short_term_value_horizons,
               value_pma_seeds=a.value_pma_seeds).to(dev).eval()
m.load_state_dict(ck['model']); m.forward_policy_value=torch.compile(m.forward_policy_value, dynamic=True)
tmp=Path(tempfile.mkdtemp(prefix='posps_'))
print('running model3 self-play @ visits=512 active=64 (8 games)...', flush=True)
t0=time.time()
res=run_selfplay_games(m,cfg,num_games=8,output_dir=tmp,epoch=0,device=dev,fp16=True,base_seed=7,active_games=64,virtual_batch_size=64,collect_examples=0)
print(f'DONE {time.time()-t0:.0f}s | searched_positions={res.searched_positions} '
      f'search_pos/s={res.search_positions_per_second:.2f} overall_pos/s={res.positions_per_second:.2f} '
      f'completed={res.completed_games}', flush=True)
import shutil; shutil.rmtree(tmp, ignore_errors=True)
