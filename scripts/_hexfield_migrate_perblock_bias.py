"""Migrate a hexfield checkpoint from the single SHARED relative-position bias
table to the per-block (3 tables) architecture — a WARM RESUME, not a fresh run.

What it does (see the per-block bias CONTRACT):
  * Model state:  pop  `bias_table` (237,4)  ->  write  `bias_tables.0/1/2`
                  each = an exact `.clone()` of the old shared table.  The seam
                  is therefore BIT-IDENTICAL: all 3 blocks reproduce the bias
                  they produced pre-migration.
  * Optimizer state: AdamW state is keyed by POSITIONAL flat ids. The only
                  structural change is a +2 insertion right after the old
                  `bias_table` id (no_decay id 50). So:
                    decay ids 0..48           -> unchanged
                    id 49 (tokens)            -> unchanged
                    id 50 (shared bias_table) -> CLONED into new ids 50,51,52
                                                 (bias_tables.0/1/2 each get a
                                                  copy of the old moments)
                    old ids >=51 (tail)       -> shift +2
                  param_groups[0].params -> [0..48]; param_groups[1].params ->
                  [49..49+no_decay_count-1].  lr/betas/eps/wd preserved verbatim
                  (resume re-applies the config lr after load).
  * `meta` (epoch, train_state, run, ...) preserved verbatim so resume
    fast-forward and the train-bucket governor stay intact.

The script operates CPU-only (no CUDA), writes a NEW file, and SELF-VERIFIES by
building a fresh HexfieldNet() + the plugin's AdamW and round-tripping the new
payload through model.load_state_dict(strict=True) + optimizer.load_state_dict.

NEVER point --in at a live checkpoint you intend to keep pristine and --out at
the same path; always copy the live ckpt to /tmp first and migrate the copy.

Usage:
    python scripts/_hexfield_migrate_perblock_bias.py --in OLD.pt --out NEW.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "packages" / "hexfield" / "python"))

import torch  # noqa: E402

from hexfield.constants import ATTENTION_HEADS as ATTN_HEADS_  # noqa: E402
from hexfield.constants import BIAS_ROWS as BIAS_ROWS_  # noqa: E402

torch.set_grad_enabled(False)


# --- structural constants (verified against model.py / live AdamW dump) -------
N_BLOCKS = 3                  # len(attn_blocks) in the live checkpoint
OLD_BIAS_TABLE_KEY = "bias_table"
NEW_BIAS_TABLE_KEYS = tuple(f"bias_tables.{i}" for i in range(N_BLOCKS))


def _clone(v):
    return v.clone() if torch.is_tensor(v) else v


def _flat_param_order(named, *, old: bool):
    """Reproduce the AdamW flat param-id order plugin.py builds: decay group
    (matrix weights, excluding bias tables + tokens) THEN no_decay group, each
    in `named_parameters()` traversal order.

    `named` is a list of (name, shape) in `named_parameters()` order for the NEW
    model. For `old=True` we synthesize the OLD model's order: the shared
    `bias_table` was a DIRECT nn.Parameter (so it appeared right after the other
    direct param `tokens`, at raw index 1), whereas the NEW `bias_tables` is a
    nn.ParameterList SUBMODULE (traversed after attn_blocks). This positional
    difference — not a simple +2 insertion — is what the id remap must follow.
    Returns the ordered list of names for that model's flat optimizer ids.
    """

    if old:
        # Direct params (tokens) come first in named_parameters; the OLD direct
        # bias_table immediately followed tokens. Drop the NEW ParameterList keys
        # and splice the single OLD key in right after `tokens`.
        names = [n for n, _ in named if not n.startswith("bias_tables.")]
        shapes = {n: s for n, s in named if not n.startswith("bias_tables.")}
        shapes[OLD_BIAS_TABLE_KEY] = (BIAS_ROWS_, ATTN_HEADS_)
        out = []
        for n in names:
            out.append(n)
            if n == "tokens":
                out.append(OLD_BIAS_TABLE_KEY)
        named_seq = [(n, shapes[n]) for n in out]
    else:
        named_seq = list(named)

    decay, no_decay = [], []
    for n, s in named_seq:
        if len(s) >= 2 and "bias_table" not in n and n != "tokens":
            decay.append(n)
        else:
            no_decay.append(n)
    return decay + no_decay


def migrate_model_state(old_model: dict) -> dict:
    """B.1 — pop shared `bias_table`, write 3 clones as bias_tables.{0,1,2}."""
    if OLD_BIAS_TABLE_KEY not in old_model:
        raise SystemExit(
            f"checkpoint model state has no '{OLD_BIAS_TABLE_KEY}' key — already "
            "migrated (or unexpected shape); aborting."
        )
    new_model = dict(old_model)
    shared = new_model.pop(OLD_BIAS_TABLE_KEY)  # (237,4)
    for key in NEW_BIAS_TABLE_KEYS:
        new_model[key] = shared.clone()
    return new_model


def migrate_optimizer_state(old_opt: dict) -> dict:
    """B.2 — remap positional AdamW state by a STRUCTURAL old->new id map.

    NOTE: the old shared `bias_table` was a DIRECT nn.Parameter (raw index 1,
    right after `tokens`), so it occupied no_decay flat id 50; the NEW
    `bias_tables` is a nn.ParameterList SUBMODULE traversed AFTER attn_blocks, so
    the 3 tables land at no_decay flat ids 145/146/147 (NOT a simple +2 insertion
    after id 50). An earlier version assumed contiguity at id 50 and corrupted
    every no_decay slot from 50 on (shape-mismatch crash at the first
    optimizer.step). We now derive both orderings from a fresh HexfieldNet and map
    each NEW flat id back to its OLD source id (the 3 bias tables all map to the
    OLD shared id), so the remap is correct by construction regardless of magic
    offsets.

    All 3 per-block tables receive a COPY of the old shared bias_table moments
    (step/exp_avg/exp_avg_sq): the seam gradients are identical at step 0, so a
    copy keeps the first post-migration AdamW step well-scaled and identical
    across the 3 tables (they diverge only once their gradients diverge).
    """
    from hexfield.model import HexfieldNet

    old_state = old_opt["state"]            # {int_id: {step, exp_avg, exp_avg_sq}}
    old_pg = old_opt["param_groups"]        # 2 groups: decay, no_decay
    if len(old_pg) != 2:
        raise SystemExit(f"expected 2 optimizer param_groups, got {len(old_pg)}")

    model = HexfieldNet()
    named = [(n, tuple(p.shape)) for n, p in model.named_parameters() if p.requires_grad]
    new_flat = _flat_param_order(named, old=False)   # names @ new flat ids
    old_flat = _flat_param_order(named, old=True)     # names @ old flat ids
    old_pos = {n: i for i, n in enumerate(old_flat)}
    old_shared_id = old_pos[OLD_BIAS_TABLE_KEY]       # the shared bias_table id

    # Sanity: the OLD flat order must agree with the checkpoint's actual state
    # shapes (catches any drift between this script's model and the live ckpt).
    for old_id, name in enumerate(old_flat):
        st = old_state.get(old_id)
        if not st or "exp_avg" not in st:
            continue
        want = (BIAS_ROWS_, ATTN_HEADS_) if name == OLD_BIAS_TABLE_KEY else dict(named)[name]
        got = tuple(st["exp_avg"].shape)
        if got != want:
            raise SystemExit(
                f"OLD optimizer id {old_id} ({name}) exp_avg shape {got} != expected "
                f"{want} — the script's reconstructed OLD param order does not match "
                "the checkpoint; aborting (would corrupt optimizer state)."
            )

    new_state: dict[int, dict] = {}
    for new_id, name in enumerate(new_flat):
        src_id = old_shared_id if name.startswith("bias_tables.") else old_pos[name]
        src = old_state.get(src_id)
        if src is None:
            continue
        # Per-block tables (3 NEW ids) each map to the SAME old shared id -> clone
        # so the three states are independent objects.
        if name.startswith("bias_tables."):
            new_state[new_id] = {k: _clone(v) for k, v in src.items()}
        else:
            new_state[new_id] = src

    # param_groups: preserve lr/betas/eps/wd verbatim; rewrite the id lists from
    # the freshly-derived split (decay = matrix weights; no_decay = the rest).
    n_decay = sum(1 for n, s in named if len(s) >= 2 and "bias_table" not in n and n != "tokens")
    n_total = len(new_flat)
    new_pg = [dict(old_pg[0]), dict(old_pg[1])]
    new_pg[0]["params"] = list(range(n_decay))
    new_pg[1]["params"] = list(range(n_decay, n_total))

    return {"state": new_state, "param_groups": new_pg}


def self_verify(new_payload: dict, old_model: dict, old_opt: dict | None) -> None:
    """Build a fresh model + the plugin's optimizer and round-trip the payload."""
    from hexfield.model import HexfieldNet

    model = HexfieldNet()  # CPU
    new_model_state = new_payload["model"]

    # 1) exact key-set match (mirrors checkpoints.load_into strict check).
    expected = set(model.state_dict().keys())
    got = set(new_model_state.keys())
    if expected != got:
        raise SystemExit(
            "key mismatch vs fresh HexfieldNet(): "
            f"missing={sorted(expected - got)[:5]} unexpected={sorted(got - expected)[:5]}"
        )

    # 2) build the optimizer EXACTLY as plugin.py does (new prefix guard).
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2 and not name.startswith("bias_table") and name != "tokens":
            decay.append(param)
        else:
            no_decay.append(param)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 0.01},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=1e-3,
    )

    # round-trip — must not raise.
    model.load_state_dict(new_model_state, strict=True)
    if new_payload.get("optimizer"):
        optimizer.load_state_dict(new_payload["optimizer"])

    # 3) bit-identity at the seam.
    shared = old_model[OLD_BIAS_TABLE_KEY]
    t0, t1, t2 = (new_model_state[k] for k in NEW_BIAS_TABLE_KEYS)
    if not (torch.equal(t0, t1) and torch.equal(t1, t2)):
        raise SystemExit("per-block bias tables are not all equal")
    for k, t in zip(NEW_BIAS_TABLE_KEYS, (t0, t1, t2)):
        if not torch.equal(t, shared):
            raise SystemExit(f"{k} != old shared bias_table")

    # 4) optimizer state STRUCTURAL integrity (the load-bearing check — an earlier
    #    bug mis-mapped these and crashed at the first optimizer.step). For every
    #    flat id, the loaded exp_avg/exp_avg_sq shape MUST equal the param it is
    #    attached to; and the 3 per-block tables must carry the OLD shared moments.
    if old_opt is not None and old_opt.get("state"):
        flat = decay + no_decay                       # NEW flat param order (objects)
        ns = optimizer.state                          # {param_tensor: {exp_avg,...}}
        for i, param in enumerate(flat):
            st = ns.get(param)
            if not st:
                continue
            for mk in ("exp_avg", "exp_avg_sq"):
                if mk in st and tuple(st[mk].shape) != tuple(param.shape):
                    raise SystemExit(
                        f"optimizer state shape mismatch at flat id {i}: "
                        f"{mk}{tuple(st[mk].shape)} attached to param{tuple(param.shape)}"
                    )
        # per-block tables carry a COPY of the old shared moments.
        old_state = old_opt["state"]
        # locate the old shared id via the reconstructed old order.
        named = [(n, tuple(p.shape)) for n, p in model.named_parameters() if p.requires_grad]
        old_flat = _flat_param_order(named, old=True)
        old_shared_id = old_flat.index(OLD_BIAS_TABLE_KEY)
        if old_state.get(old_shared_id):
            old_shared = old_state[old_shared_id]["exp_avg"]
            for bt in NEW_BIAS_TABLE_KEYS:
                param = dict(model.named_parameters())[bt]
                if not torch.equal(ns[param]["exp_avg"], old_shared):
                    raise SystemExit(f"new optimizer moments for {bt} != old shared moments")

        # 5) the definitive resume guard: one real optimizer.step() must not raise.
        model.zero_grad(set_to_none=True)
        for p in model.parameters():
            if p.requires_grad:
                p.grad = torch.zeros_like(p)
        optimizer.step()

    return model, optimizer, decay, no_decay


def main() -> None:
    ap = argparse.ArgumentParser(description="Migrate hexfield ckpt to per-block bias tables.")
    ap.add_argument("--in", dest="inp", required=True, help="old (shared-bias) checkpoint .pt")
    ap.add_argument("--out", dest="out", required=True, help="new (per-block) checkpoint .pt")
    args = ap.parse_args()

    in_path = Path(args.inp)
    out_path = Path(args.out)
    if not in_path.exists():
        raise SystemExit(f"input checkpoint not found: {in_path}")
    if out_path.resolve() == in_path.resolve():
        raise SystemExit("--out must differ from --in (never migrate in place)")

    payload = torch.load(in_path, map_location="cpu", weights_only=False)
    if "meta" not in payload:
        raise SystemExit("not a pipeline checkpoint (no 'meta') — refusing to migrate")
    old_model = payload["model"]
    old_opt = payload.get("optimizer")

    new_payload = dict(payload)                       # preserves 'meta' verbatim
    new_payload["model"] = migrate_model_state(old_model)
    if old_opt:
        new_payload["optimizer"] = migrate_optimizer_state(old_opt)
    else:
        new_payload["optimizer"] = old_opt            # None / falsy -> passthrough

    # verify on a COPY of state, BEFORE writing, so a bad transform never hits disk.
    _, _, decay, no_decay = self_verify(new_payload, old_model, old_opt)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_payload, out_path)

    # ---- summary --------------------------------------------------------------
    n_model_params = len(new_payload["model"])
    epoch = new_payload.get("meta", {}).get("epoch")
    print("=" * 64)
    print("hexfield per-block bias migration — OK")
    print("=" * 64)
    print(f"  in        : {in_path}")
    print(f"  out       : {out_path}")
    print(f"  meta.epoch: {epoch}")
    print(f"  model keys: {n_model_params - N_BLOCKS} (old) -> {n_model_params} (new, +{N_BLOCKS - 1})")
    print(f"  removed   : '{OLD_BIAS_TABLE_KEY}'")
    print(f"  added     : {', '.join(NEW_BIAS_TABLE_KEYS)} (each = clone of shared)")
    print(f"  optimizer : {'remapped' if old_opt else 'absent (passthrough)'}")
    if old_opt:
        print(f"              decay group params : {len(new_payload['optimizer']['param_groups'][0]['params'])}")
        print(f"              no_decay group params: {len(new_payload['optimizer']['param_groups'][1]['params'])}")
        print(f"              state ids          : {len(new_payload['optimizer']['state'])}")
    print(f"  fresh AdamW: decay={len(decay)} no_decay={len(no_decay)} (built like plugin.py)")
    # bit-identity restatement (already asserted in self_verify)
    print("  seam      : bias_tables.0 == .1 == .2 == old shared bias_table (verified)")
    print("  round-trip: model.load_state_dict(strict=True) + optimizer.load_state_dict OK")
    print("=" * 64)


if __name__ == "__main__":
    main()
