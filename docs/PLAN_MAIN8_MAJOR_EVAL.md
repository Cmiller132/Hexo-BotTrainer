# PLAN: main_8 major eval — automated 10-epoch powered gate (DEFERRED)

Status: DESIGN, NOT BUILT (2026-07-06). Deferred by user decision — main_8
runs with the existing every-5 multistage eval for now. This captures the
major-eval design for when it's built later. The scheduling and robustness
choices below are DECIDED (user, 2026-07-06); the opponent choice is open.

## Why a major eval at all

The every-5 multistage eval is a pooled-Elo tripwire against SealBot + an anchor
pool. It misled us three times on main_7 (ep35 "−188" was 4× exaggerated, ep55
"+191" ~7×, and seeding damage was invisible in it entirely). The fix is a
**powered paired match** on a fixed cadence as the *primary* strength read, with
multistage demoted to trigger-only. This is §5.2 of
`docs/PLAN_MAIN8_HYBRID_SEARCH.md`.

## Decided design (build to this)

- **Cadence**: every 10 epochs (the "major eval"). The every-5 multistage stays
  as the "mini eval," unchanged.
- **Games**: 200 paired games at the eval profile (512 visits).
- **Scheduling — INLINE in the epoch eval slot.** Run it in the existing
  between-epochs eval phase (where multistage already runs), not a separate
  process. No mid-generation interruption; it just extends the eval phase by
  ~55 min every 10 epochs (~10% throughput hit at the 10-epoch granularity).
  Cleanest signal, no GPU contention with self-play.
- **Robustness — paired openings + pentanomial.** CRN paired openings (each
  opening played from both seats), pentanomial pair-level SE, fixed 200 games.
  This matches the manual `h2h_match.py` flow already validated on main_7
  (ep40/50/55/67/77/91 gates). SPRT machinery exists (`multi_stage_eval.sprt`,
  currently off) and could be layered later to early-stop lopsided matches, but
  the decided baseline is fixed-N for a stable, comparable Elo each gate.
- **Search regime — PUCT, both sides** (main_8's native searcher; see the eval
  verification in `PLAN_MAIN8_HYBRID_SEARCH.md` §2.3 and the memory note). The
  candidate defaults to the self-play profile (`build_divergence_overrides(sp)`
  = Gumbel-off = PUCT); the eval path is the non-PCR batched session search, so
  no Fast/Gumbel moves ever appear.

## Open question — opponent (choose at build time)

1. **epN vs epN−10 (progress slope).** Both sides are main_8/PUCT — a clean,
   self-consistent measurement of the training slope over the last 10 epochs.
   This is the §5.2 proposal and the recommended PRIMARY. Doesn't give absolute
   position vs the bar.
2. **epN vs main7_ep67 (the bar to beat).** Absolute position vs the checkpoint
   main_8 warm-started from and must clear. CAVEAT: ep67 was Gumbel-trained; see
   the anchor-searcher note below.
3. **Both each major gate** (~400 games / ~2 hrs of eval). Fullest signal.

Recommendation: primary = epN vs epN−10 (clean slope); add epN vs main7_ep67 at
a coarser cadence (e.g. every 30 epochs) for an absolute-position check without
paying 400 games every gate.

## Anchor-searcher mismatch (applies to both mini and major eval)

`multistage_eval._foreign_opponent_overrides` forces the **PUCT** profile onto
every FOREIGN anchor (any checkpoint outside the run's own checkpoints dir),
with a docstring assuming foreign = "pre-Gumbel PUCT lineage (main4/main5)."
main_8's anchor pool includes `main6_ep73` and `main7_ep67`, which were
**Gumbel-trained**. So those two anchors are evaluated under a searcher they
were not tuned for → they play somewhat below their native strength → main_8's
Elo *vs those two edges* is mildly overstated (direction: flatters main_8).

- `main5_ep105` (genuinely PUCT-trained) and SealBot (heuristic, searcher-free)
  are unaffected.
- main_8's OWN eval is correct PUCT regardless — this only touches the two
  Gumbel-lineage anchor edges.
- The self-lineage edges (`ep5`, `ep30`) are in-run, so they keep the
  candidate's PUCT profile — clean.

Options if we want a fair `main7_ep67` bar (for the major eval especially):
(a) leave as-is — a fixed, consistent (if slightly loose) reference; the trend
    stays monotone; (b) play `main6/main7` anchors under their NATIVE Gumbel
    profile (per-opponent override = the run-of-origin's self-play profile
    instead of forced PUCT) so "did we beat the old regime's peak" is
    apples-to-apples. (b) needs `_foreign_opponent_overrides` to key the profile
    on the anchor's training lineage, not just "is it foreign."

## Implementation sketch (when built)

- Bake `scratchpad/h2h_match.py` into `scripts/_h2h_gate.sh` (paired 200 @ 512v,
  pentanomial SE, PUCT via the main_8 config).
- Trigger inside the epoch eval phase when `epoch % 10 == 0`, gated by a config
  flag (e.g. `[model.config.major_eval] enabled / every_n_epochs / n_games /
  opponent`) so it's opt-in and doesn't touch the shared supervisor for other
  runs.
- Append each result to `diagnostics/h2h_gates.jsonl`
  (`{epoch, opponent, w, l, d, elo, se, n, wall_s}`) and surface on the
  dashboard next to the multistage verdicts.
- Demote multistage to trigger-only: a large pooled swing *arms* a powered
  match; it never decides on its own.

## Kill-gate hook (from the hybrid-search plan §7)

Feed the major-eval slope into the kill-gate: pivot only if by ep50 the powered
slope is < +1 Elo/epoch over two consecutive major gates, OR main_8 tracks
> 100 Elo behind main_7's same-epoch trajectory. No pivot before ep50 (user's
"looser / more patience" decision).
