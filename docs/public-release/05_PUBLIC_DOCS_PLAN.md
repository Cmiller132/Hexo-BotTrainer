# 05 — Public Docs Plan

What the public repo's documentation should say, tuned for the stated audience:
**people studying the system**.

## Root `README.md` (rewrite from scratch)

Structure:

1. **What this is** (3–4 sentences): an AlphaZero-style self-play RL system for
   Hexo (Connect6-style game on an unbounded hex grid), with an authoritative
   Rust rules engine, a PyTorch/Triton model ("hexfield"), a config-driven
   trainer, a match runner, and a web dashboard. Trained weights included —
   you can play the bot in your browser in ~10 minutes.
2. **Quick start — play the bot**: clone (with LFS), venv, `pip install`,
   `scripts/build_native.sh`, `scripts/dashboard.sh`, open `:8080`, Match arena
   vs `models/hexfield_main7_infer.pt`. This is the hook; it must be short and
   it must work.
3. **Quick start — train**: `configs/hexfield_smoke_tiny.toml` first (minutes,
   proves the loop), then the real recipe `configs/hexfield_main_7.toml` with
   honest hardware expectations (12 GB GPU; what a day of training buys).
   Optional warm start via the HF corpus + prefit scripts.
4. **Package map**: the six-package table (engine / utils / runner / train /
   frontend / hexfield) — one row each, roles only, no legacy status columns.
5. **How it works** (short, link-heavy): self-play MCTS (Gumbel) → replay
   buffer → supervised updates → gated eval; pointer to ARCHITECTURE.md,
   hexfield_blueprint.md, and the model spec for depth. Include a short
   "why Gumbel, not classic PUCT+Dirichlet" paragraph — the repo deliberately
   ships only the Gumbel path (classic AlphaZero exploration knobs were
   stripped), and readers coming from the AlphaZero paper will look for them.
6. **The cross-package contracts** paragraph from the current README survives
   in spirit — it's exactly what a student wants: tensor byte protocol,
   `.npz` shard format, `.hxr` records, packed action IDs, diagnostics JSON.
7. **Optional: SealBot baseline** — link <https://github.com/Ramora0/SealBot>,
   build it, `export SEALBOT_PATH=...`, flip `sealbot_enabled` for eval.
8. **License** (MIT) + a one-line provenance note ("extracted from a private
   research repo; single-commit history is intentional").

Explicitly **absent**: WSL/Windows dual-environment narrative, live-run
warnings, supervisor babysitting lore, private venv paths. Document one clean
Linux/WSL path; note Windows-native is untested.

## `docs/` layout

```
docs/
  intro_to_hexo.md          # as-is (game rules, coordinates, terminology)
  ARCHITECTURE.md           # rewritten: hexfield-only system diagram & data flow
  hexfield_blueprint.md     # path-scrubbed
  specs/
    hexfield_model_spec.md  # path-scrubbed
    hexfield_eval_v2_spec.md
    hexfield_v2_fixes.md / hexfield_v2_synthesis.md   # keep if they read as design docs, cut if they read as internal patch logs — review
    match_screen_v2_spec.md / debug_screen_v2_spec.md / history_screen_v2_spec*.md
```

`ARCHITECTURE.md` rewrite guidance: keep the layered structure (engine →
featurization → search → training loop → eval → dashboard) but delete the
lineage-history sections and the environment appendix; every code path named
must exist in the public tree.

## `models/MODEL_CARD.md`

- **Files**: `hexfield_main7_infer.pt` (weights-only, for play/study) and
  `hexfield_main7_full.pt` (full checkpoint, resume training).
- **Architecture**: channels 192, 3 attention heads, trunk `CCACCACCACCACCA`
  (~8.1 M params) — and the exact env vars that must be set to instantiate it,
  since arch is env-driven: `HEXFIELD_CHANNELS=192 HEXFIELD_ATTENTION_HEADS=3
  HEXFIELD_TRUNK=CCACCACCACCACCA`.
- **Training provenance**: warm-started via behavioral cloning on the public
  corpus ([timmyburn/hexo-bootstrap-corpus](https://huggingface.co/datasets/timmyburn/hexo-bootstrap-corpus)),
  then N epochs of self-play RL on a single RTX 4070 Ti; epoch and date filled
  in at publish time.
- **Strength**: eval numbers vs the shipped anchors / SealBot at publish time,
  with the honest caveat that the run was live when snapshotted.
- **How to load**: three-line snippet (env vars + `HexfieldNet` + state dict),
  plus "or just open the dashboard".

## Package READMEs

Review each kept `packages/*/README.md` against the public layout; the main
edits are removing `hexo_models` build references (engine/utils) and legacy
lineage mentions (frontend's debug-workbench section). hexfield's README (if
absent) gets a short one: what the package is, the Rust/Python split, the
Triton kernels, pointer to the blueprint/spec.

## Tone rule

The private repo's annotated-config / evidence-dossier style is a genuine
teaching asset — keep the *why* annotations in configs and specs. Cut only the
*private history*: run post-mortems, machine paths, supervision lore, and
references to artifacts the public can't see.
