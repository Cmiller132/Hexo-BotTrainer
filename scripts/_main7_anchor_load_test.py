"""Eval-arena anchor loading under the main_7 process env: every permanent
anchor in configs/hexfield_main_7.toml plus the candidate itself must
strict-load via eval_arena._load_hexfield_net (foreign archs included)."""

import sys

sys.path.insert(0, "/mnt/e/Hexo-BotTrainer-hexgt/packages/hexfield/python")

from hexfield.eval_arena import _load_hexfield_net  # noqa: E402

CKPTS = [
    ("main4_ep60", "/mnt/e/Hexo-BotTrainer/runs/hexfield_main_4/checkpoints/epoch_000060.pt"),
    ("main5_ep105", "/mnt/e/Hexo-BotTrainer/runs/hexfield_main_5/checkpoints/epoch_000105.pt"),
    ("main6_ep73", "/mnt/e/Hexo-BotTrainer/runs/hexfield_main_6/checkpoints/epoch_000073.pt"),
    ("main7_ep4", "/mnt/e/Hexo-BotTrainer/runs/hexfield_main_7/checkpoints/epoch_000004.pt"),
]

ok = True
for label, path in CKPTS:
    try:
        net = _load_hexfield_net(path)
        c = net.stem.out_channels
        heads = net.attn_blocks[0].attn.heads
        print(f"{label}: OK c={c} heads={heads} layout={net._trunk_layout} "
              f"conv={len(net.conv_blocks)} attn={len(net.attn_blocks)}")
    except Exception as e:
        print(f"{label}: FAIL {type(e).__name__}: {str(e)[:160]}")
        ok = False
print("ANCHOR LOAD TEST " + ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
