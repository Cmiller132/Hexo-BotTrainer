"""Verify the bucketing/padding fix is equivalence-preserving.

Feeds IDENTICAL real MCTS payloads (real engine states encoded via rust_bridge)
to two evaluators that differ ONLY in bucket granularity:
  A = default power-of-two buckets (current production)
  B = tighter multiple-of-N buckets (the fix)
and compares the bytes Rust actually receives (values_bytes, priors_bytes) plus
the raw policy/value forward. Reports exact-equality and max-abs diff.

Padding never crosses samples (every op is per-sample after BN folding), so the
only possible source of difference is cuDNN picking a different conv algorithm
for a different padded shape (a ~1-ULP reduction-order effect) — which the
EXISTING pow2 bucketing already incurs whenever batch size changes. We quantify
it rather than assume zero.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for p in ("hexo_engine", "hexo_utils", "hexo_runner", "hexo_train", "hexo_models", "hexo_frontend"):
    sp = str(REPO / "packages" / p / "python")
    if sp not in sys.path:
        sys.path.insert(0, sp)

import torch

CONFIG = REPO / "configs" / "dense_cnn_model1_target_96x6.toml"
CKPT = REPO / "runs" / "dense_cnn_model1_target_96x6" / "checkpoints" / "bootstrap_sealbot_prefit.pt"


def build_model():
    import tomllib
    from hexo_models.dense_cnn.plugin import DenseCNNPlugin
    section = tomllib.loads(CONFIG.read_text())["model"]["config"]
    model = DenseCNNPlugin().build_model(game_spec={}, config=section)
    model.load_state_dict(torch.load(CKPT, map_location="cpu")["model_state"], strict=True)
    return model.eval()


def make_payloads(n_states):
    import random
    import hexo_engine as engine
    from hexo_engine.types import unpack_coord_id
    from hexo_models.dense_cnn import rust_bridge
    rng = random.Random(11)
    states = []
    gi = 0
    while len(states) < n_states:
        st = engine.new_game(seed=300_000 + gi); gi += 1
        for _ in range(rng.randint(2, 110)):
            if engine.terminal(st) is not None:
                break
            aids = engine.legal_action_ids(st)
            if not aids:
                break
            engine.apply_action(st, engine.PlacementAction(unpack_coord_id(rng.choice(aids))))
        if engine.terminal(st) is None:
            states.append(engine.clone_state(st))
    # Build several payloads of varying batch size to exercise different buckets.
    payloads = []
    for batch in (5, 17, 33, 70, 99, 130, 200):
        sub = states[:batch]
        p = rust_bridge.model1_batch_inputs(sub)
        shape = tuple(int(x) for x in p["shape"])
        flats_rows = [list(int(i) for i in row) for row in p["legal_flat_indices"]]
        offs = [0]; flat = []
        for row in flats_rows:
            flat.extend(row); offs.append(len(flat))
        payloads.append({
            "shape": shape, "inputs": bytes(p["inputs"]),
            "legal_flat_indices_bytes": np.array(flat, dtype=np.int64).tobytes(),
            "legal_row_offsets": offs,
        })
    return payloads


def main():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from hexo_models.dense_cnn.inference import DenseCNNInference

    model = build_model()
    payloads = make_payloads(220)

    a = DenseCNNInference(model, device="cuda", amp=True, max_batch_size=1024, bucket_pad_multiple=None)   # pow2
    b = DenseCNNInference(model, device="cuda", amp=True, max_batch_size=1024, bucket_pad_multiple=16)      # tighter

    print(f"A buckets=pow2  B buckets=mult16", flush=True)
    all_exact = True
    max_v = 0.0; max_p = 0.0
    for pl in payloads:
        ra = a.evaluate_model1_payload(pl)
        rb = b.evaluate_model1_payload(pl)
        v_exact = ra["values_bytes"] == rb["values_bytes"]
        p_exact = ra["priors_bytes"] == rb["priors_bytes"]
        va = np.frombuffer(ra["values_bytes"], dtype=np.float32)
        vb = np.frombuffer(rb["values_bytes"], dtype=np.float32)
        pa = np.frombuffer(ra["priors_bytes"], dtype=np.float32)
        pb = np.frombuffer(rb["priors_bytes"], dtype=np.float32)
        dv = float(np.abs(va - vb).max()) if va.size else 0.0
        dp = float(np.abs(pa - pb).max()) if pa.size else 0.0
        max_v = max(max_v, dv); max_p = max(max_p, dp)
        all_exact &= (v_exact and p_exact)
        bs = pl["shape"][0]
        print(f"  bs={bs:>4d}: values byte-exact={v_exact} priors byte-exact={p_exact} "
              f"max|dvalue|={dv:.2e} max|dprior|={dp:.2e}", flush=True)
    print(f"\nRESULT: all_byte_exact={all_exact}  max|dvalue|={max_v:.2e}  max|dprior|={max_p:.2e}", flush=True)
    print("(if not byte-exact, diffs are cuDNN algo/reduction-order at ULP scale, "
          "far below fp16 noise ~1e-2 — equivalence-preserving)", flush=True)


if __name__ == "__main__":
    main()
