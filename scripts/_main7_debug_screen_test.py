"""Dashboard debug/match backend test: load + analyze + as-trained search +
attention for BOTH a main_6 (c=128/4-head/3-attn) and main_7
(c=192/3-head/5-attn) checkpoint, in a worker-like env (CPU, no HEXFIELD_*
arch env). Verifies the state-dict arch inference, the manifest-driven gumbel
search profile, and the dynamic attention constants."""

import sys

sys.path.insert(0, "/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_frontend/python")
sys.path.insert(0, "/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_engine/python")
sys.path.insert(0, "/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_utils/python")
sys.path.insert(0, "/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_runner/python")

from hexo_frontend import debug_infer as di  # noqa: E402

CASES = [
    ("main_6", "/mnt/e/Hexo-BotTrainer/runs/hexfield_main_6/checkpoints/epoch_000073.pt", 3, 4),
    ("main_7", "/mnt/e/Hexo-BotTrainer/runs/hexfield_main_7/checkpoints/epoch_000004.pt", 5, 3),
]

ok = True
for name, ckpt, want_blocks, want_heads in CASES:
    loaded = di.load_checkpoint(ckpt)
    arch = loaded.arch
    print(f"{name}: lineage={loaded.lineage} warnings={loaded.load_warnings} "
          f"c={arch.get('channels')} heads={arch.get('attention_heads')} "
          f"layout={arch.get('trunk_layout')} run_dir={loaded.run_dir is not None}")
    ok &= not loaded.load_warnings

    a = di.analyze_position(loaded, [])
    print(f"  analyze: value={a['value']:.4f} top_p={a['policy'][0]['p']:.4f} "
          f"candidates={a['candidate_count']}")
    ok &= abs(a["value"]) <= 1.0 and a["candidate_count"] > 0

    s = di.search_position(loaded, [], visits=48, c_puct=1.5, seed=7)
    prof = s.get("search_profile") or {}
    print(f"  search: visits={s['visits']} root_v={s['root_value']:.4f} "
          f"best={s['best']} in_search={s.get('selection_in_search')} "
          f"profile={prof}")
    ok &= bool(s.get("selection_in_search"))
    ok &= prof.get("source") == "manifest"
    ok &= prof.get("gumbel_root") is True  # both runs are gumbel-trained

    # tempered opening selection must also work (in-search sampling)
    s1 = di.search_position(loaded, [], visits=48, c_puct=1.5, seed=11, temperature=1.0)
    print(f"  search(temp=1): best={s1['best']}")

    att = di.attention_position(
        loaded, [], block=want_blocks - 1, head=None,
        query={"type": "token", "id": 0},
    )
    print(f"  attention: found={att['found']} num_blocks={att['num_blocks']} "
          f"num_heads={att['num_heads']} block={att['block']} cells={att['num_cells']}")
    ok &= att["found"] and att["num_blocks"] == want_blocks and att["num_heads"] == want_heads
    ok &= att["block"] == want_blocks - 1

print("DEBUG SCREEN TEST " + ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
