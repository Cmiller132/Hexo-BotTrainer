"""Unit test: hexfield::attn_pair Triton kernel vs fp32 reference math.

Reference reproduces the flex-pair semantics exactly: scores over ALL S keys,
bias = table2[pair] (pad-key columns carry the -3e4 pad row), softmax in fp32.
The kernel bounds its key loop by seq_lens; pad keys beyond the bound underflow
to exactly 0 in the reference softmax, so results must match to fp16 rounding.
Only q rows < seq_len are compared (pad q rows are discarded downstream).
"""

import sys

import torch

sys.path.insert(0, "/mnt/e/Hexo-BotTrainer-hexgt/packages/hexfield/python")
from hexfield._triton_attn import attn_pair  # noqa: E402

NUM_TOKENS = 8
ROWS = 238  # 237 bias rows + appended pad row
PAD_ROW = ROWS - 1
PAD_VAL = -3.0e4

torch.manual_seed(0)
dev = "cuda"


def make_case(b, h, s, d, n_valid_list):
    q = torch.randn(b, h, s, d, device=dev, dtype=torch.float16) * 0.5
    k = torch.randn(b, h, s, d, device=dev, dtype=torch.float16) * 0.5
    v = torch.randn(b, h, s, d, device=dev, dtype=torch.float16) * 0.5
    pair = torch.randint(0, PAD_ROW, (b, s, s), device=dev, dtype=torch.uint8)
    table2 = torch.cat(
        [
            torch.randn(PAD_ROW, h, device=dev, dtype=torch.float16) * 0.2,
            torch.full((1, h), PAD_VAL, device=dev, dtype=torch.float16),
        ]
    )
    seq_lens = torch.empty(b, dtype=torch.int32, device=dev)
    for i, nv in enumerate(n_valid_list):
        sl = NUM_TOKENS + nv
        seq_lens[i] = sl
        pair[i, :, sl:] = PAD_ROW  # pad-KEY columns -> pad row
    return q, k, v, pair, table2, seq_lens


def reference(q, k, v, pair, table2, scale):
    bias = table2[pair.long()].permute(0, 3, 1, 2).float()  # (B,H,S,S)
    scores = torch.einsum("bhqd,bhkd->bhqk", q.float(), k.float()) * scale + bias
    p = torch.softmax(scores, dim=-1)
    return torch.einsum("bhqk,bhkd->bhqd", p, v.float())


def run_case(name, b, h, s, d, n_valid_list):
    q, k, v, pair, table2, seq_lens = make_case(b, h, s, d, n_valid_list)
    with torch.no_grad():
        out = attn_pair(q, k, v, pair, table2, seq_lens)
    ref = reference(q, k, v, pair, table2, 1.0 / (d**0.5))
    worst = 0.0
    for i in range(b):
        sl = int(seq_lens[i])
        diff = (out[i, :, :sl].float() - ref[i, :, :sl]).abs().max().item()
        worst = max(worst, diff)
        # tile-skipped pad q rows must be finite (zeros)
        assert torch.isfinite(out[i]).all().item(), f"{name}: non-finite output"
    status = "PASS" if worst < 3e-2 else "FAIL"
    print(f"{name}: B={b} H={h} S={s} D={d} valid={n_valid_list} "
          f"max|diff|={worst:.2e} {status}")
    return worst < 3e-2


ok = True
ok &= run_case("main7-shape", 3, 3, 396, 64, [388, 200, 17])
ok &= run_case("main7-large", 2, 3, 681, 64, [673, 512])
ok &= run_case("main6-shape", 2, 4, 229, 32, [221, 100])
ok &= run_case("small", 1, 3, 71, 64, [63])
ok &= run_case("no-pad", 2, 3, 264, 64, [256, 256])
ok &= run_case("tiny-valid", 2, 3, 264, 64, [1, 8])
print("UNIT " + ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
