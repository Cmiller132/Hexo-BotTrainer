import torch
from hexo_models.hexgt.config import parse_hexgt_config
from hexo_models.hexgt.constants import NODE_FEATURE_DIM
from hexo_models.hexgt.losses import value_bins
# Replicate EXACTLY what _rl_train.py builds from the launcher args:
cfg = parse_hexgt_config({
    "device": "cuda",
    "architecture": {"candidate_radius": 3},
    "training": {"learning_rate": 2e-4, "warmup_steps": 200, "batch_size": 128},
    "selfplay": {
        "search_visits": 128, "max_actions": 240,
        "c_puct": 1.5, "root_policy_temperature": 1.0,
        "root_dirichlet_noise_enabled": True, "root_dirichlet_total_alpha": 6.6,
        "root_dirichlet_noise_fraction": 0.25,
        "temperature": 1.0, "final_temperature": 0.2,
        "temperature_decay_moves": 30, "temperature_floor": 0.1,
        "forced_playout_k": 2.0,
    },
})
sp, tr, ar = cfg.selfplay, cfg.training, cfg.architecture
print("SELF-PLAY: search_visits=%d  active(cli)=64  vbatch(cli)=64  candidate_radius=%d  max_actions=%d  mcts_active_root_limit=%d  cache_max_states=%d"
      % (sp.search_visits, ar.candidate_radius, sp.max_actions, sp.mcts_active_root_limit, sp.mcts_session_cache_max_states))
print("EXPLORATION: total_alpha=%.2f  eps=%.2f  noise_enabled=%s  root_policy_temp=%.2f  c_puct=%.2f  fpu_reduction=%.2f  virtual_loss=%.2f"
      % (sp.root_dirichlet_total_alpha, sp.root_dirichlet_noise_fraction, sp.root_dirichlet_noise_enabled, sp.root_policy_temperature, sp.c_puct, sp.fpu_reduction, sp.virtual_loss))
print("  temperature: init=%.2f final=%.2f decay_moves=%d floor=%.2f schedule=%s" % (sp.temperature, sp.final_temperature, sp.temperature_decay_moves, sp.temperature_floor, sp.temperature_schedule))
print("  widening: policy_mass=%.2f  max_children=%d  min_children=%d" % (sp.widening_policy_mass, sp.widening_max_children, sp.widening_min_children))
print("TRAINING: batch=%d  lr=%g  warmup=%d  weight_decay=%g  max_grad_norm=%.2f  amp=%s" % (tr.batch_size, tr.learning_rate, tr.warmup_steps, tr.weight_decay, tr.max_grad_norm, tr.amp))
print("  loss_weights: policy=%.2f  value=%.2f  opp_policy=%.2f  short_term_value=%.2f" % (tr.policy_weight, tr.value_weight, tr.opp_policy_weight, tr.short_term_value_weight))
print("ARCH(defaults): token_dim=%d gnn_layers=%d ctx_layers=%d heads=%d ffn=%d node_feat_dim=%d st_value_horizons=%s value_bins=%d"
      % (ar.token_dim, ar.gnn_layers, ar.ctx_layers, ar.attention_heads, ar.ffn_dim, NODE_FEATURE_DIM, ar.short_term_value_horizons, value_bins().numel()))
# the actual model arch from the epoch-8 checkpoint:
ck = torch.load("/mnt/e/Hexo-BotTrainer-hexgt/runs/hexgt_rl_main/checkpoints/hexgt_rl_epoch000008.pt", map_location="cpu", weights_only=False)
print("CHECKPOINT arch:", ck["arch"])
