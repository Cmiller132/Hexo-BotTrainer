"""Render the hexfield in-run eval as nice, honest, COMPARABLE learning-curve
graphs from the persisted Bradley-Terry pool. PURE READ — touches only the pool
+ per-epoch JSONs and writes only under --out.

Two figures (both show 95% CIs):
  (a) Pooled Bradley-Terry Elo per candidate epoch, REFIT AS-OF-EPOCH (for each
      epoch E the fit uses only edges with epoch<=E), anchor pinned to bc_prefit
      (SealBot unavailable). The headline "is it still learning?" curve.
  (b) Candidate win-rate vs EACH fixed anchor (bc_prefit, ep5) over epochs — a
      drift-immune, frozen-opponent view. Pair-level CI when paired, else Wilson
      on the PHYSICAL integer counts.

A UNIFORMITY BADGE asserts every checkpoint edge was played at 512 visits / vbs16
(comparable); it reads GREEN only when that holds, else RED listing offenders.
matplotlib is intentionally NOT used (not installed) — output is self-contained
HTML + inline SVG, dark-mode friendly via CSS variables.

Usage:
  python scripts/_hexfield_eval_graphs.py [--run-dir DIR] [--pool POOL.json]
      [--out OUTDIR] [--anchors bc_prefit,ep5]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "packages" / "hexfield" / "python"))

from hexfield import eval_stats  # noqa: E402
from hexfield import multistage_eval as mse  # noqa: E402

LOCKED_VISITS = 512
LOCKED_VBS = 16


def _cand_epoch(label: str) -> int | None:
    if not label.startswith("cand_ep"):
        return None
    try:
        return int(label[len("cand_ep"):])
    except ValueError:
        return None


def _filter_doc(doc: dict, max_epoch: int) -> dict:
    return {**doc, "edges": [e for e in doc.get("edges", []) if int(e.get("epoch", -1)) <= max_epoch]}


def _fit_asof(doc: dict, epoch: int):
    """BT fit over edges with epoch<=`epoch`; anchor preferred bc_prefit."""
    sub = _filter_doc(doc, epoch)
    edges = mse._bt_edges_from_pool(sub)
    if not edges:
        return None, None
    labels = {l for e in edges for l in (e.a, e.b)}
    anchor = "bc_prefit" if "bc_prefit" in labels else (
        "sealbot" if "sealbot" in labels else sorted(labels)[0])
    if not any(anchor in (e.a, e.b) for e in edges):
        return None, anchor
    try:
        fit = eval_stats.bradley_terry(edges, anchor=anchor, grad_tol=1e-6, max_iter=200)
    except Exception:
        return None, anchor
    return fit, anchor


def _winrate_ci(raw: dict) -> tuple[float, float, float, int, str]:
    """(winrate, lo, hi, decided, interval_kind) from a pool row's raw block."""
    n_pairs = raw.get("n_pairs")
    wa = int(round(float(raw.get("physical_wins_a", 0) or 0)))
    wb = int(round(float(raw.get("physical_wins_b", 0) or 0)))
    decided = wa + wb
    if n_pairs and raw.get("pair_winrate") is not None and raw.get("pair_se") is not None \
            and math.isfinite(float(raw["pair_se"])):
        wr = float(raw["pair_winrate"])
        se = float(raw["pair_se"])
        return wr, max(0.0, wr - 1.96 * se), min(1.0, wr + 1.96 * se), decided, "pair-level"
    if decided <= 0:
        return float("nan"), 0.0, 1.0, 0, "none"
    lo, hi = eval_stats.wilson_ci(wa, decided)
    return wa / decided, lo, hi, decided, "Wilson"


# ----------------------------- tiny SVG helpers ----------------------------- #
def _svg_line_chart(series, *, width=720, height=320, y_domain, y_fmt, title,
                    y_label, ref=None):
    """series: list of dicts {name,color,points:[(x,y,lo,hi)]}. x is epoch int."""
    pad_l, pad_r, pad_t, pad_b = 56, 16, 36, 34
    xs = [p[0] for s in series for p in s["points"]]
    if not xs:
        return f'<svg viewBox="0 0 {width} {height}"><text x="20" y="40" fill="var(--muted)">no data</text></svg>'
    xmin, xmax = min(xs), max(xs)
    if xmin == xmax:
        xmin -= 1; xmax += 1
    ymin, ymax = y_domain
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b

    def X(x): return pad_l + (x - xmin) / (xmax - xmin) * pw
    def Y(y): return pad_t + (1 - (y - ymin) / (ymax - ymin)) * ph

    out = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" font-family="system-ui,sans-serif">']
    out.append(f'<text x="{pad_l}" y="20" fill="var(--fg)" font-size="14" font-weight="600">{title}</text>')
    # axes
    out.append(f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+ph}" stroke="var(--grid)"/>')
    out.append(f'<line x1="{pad_l}" y1="{pad_t+ph}" x2="{pad_l+pw}" y2="{pad_t+ph}" stroke="var(--grid)"/>')
    for i in range(5):
        yv = ymin + (ymax - ymin) * i / 4
        yy = Y(yv)
        out.append(f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{pad_l+pw}" y2="{yy:.1f}" stroke="var(--grid)" stroke-dasharray="2,4" opacity="0.5"/>')
        out.append(f'<text x="{pad_l-6}" y="{yy+4:.1f}" fill="var(--muted)" font-size="11" text-anchor="end">{y_fmt(yv)}</text>')
    for xv in sorted(set(int(x) for x in xs)):
        out.append(f'<text x="{X(xv):.1f}" y="{pad_t+ph+18:.1f}" fill="var(--muted)" font-size="11" text-anchor="middle">{xv}</text>')
    out.append(f'<text x="{pad_l+pw/2:.0f}" y="{height-4}" fill="var(--muted)" font-size="11" text-anchor="middle">candidate epoch</text>')
    out.append(f'<text x="14" y="{pad_t+ph/2:.0f}" fill="var(--muted)" font-size="11" text-anchor="middle" transform="rotate(-90 14 {pad_t+ph/2:.0f})">{y_label}</text>')
    if ref is not None:
        ry = Y(ref)
        out.append(f'<line x1="{pad_l}" y1="{ry:.1f}" x2="{pad_l+pw}" y2="{ry:.1f}" stroke="var(--muted)" stroke-dasharray="4,3" opacity="0.7"/>')
    for s in series:
        pts = [p for p in s["points"] if p[1] == p[1]]  # drop NaN
        if not pts:
            continue
        # CI band
        band = " ".join(f'{X(p[0]):.1f},{Y(p[3]):.1f}' for p in pts)
        band += " " + " ".join(f'{X(p[0]):.1f},{Y(p[2]):.1f}' for p in reversed(pts))
        out.append(f'<polygon points="{band}" fill="{s["color"]}" opacity="0.15"/>')
        line = " ".join(f'{X(p[0]):.1f},{Y(p[1]):.1f}' for p in pts)
        out.append(f'<polyline points="{line}" fill="none" stroke="{s["color"]}" stroke-width="2"/>')
        for p in pts:
            out.append(f'<circle cx="{X(p[0]):.1f}" cy="{Y(p[1]):.1f}" r="3" fill="{s["color"]}"/>')
    # legend
    lx = pad_l + 8
    for s in series:
        out.append(f'<rect x="{lx}" y="{pad_t+2}" width="10" height="10" fill="{s["color"]}"/>')
        out.append(f'<text x="{lx+14}" y="{pad_t+11}" fill="var(--fg)" font-size="11">{s["name"]}</text>')
        lx += 16 + 8 * len(s["name"])
    out.append("</svg>")
    return "".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="Render hexfield eval learning-curve graphs (PURE READ).")
    ap.add_argument("--run-dir", default="/mnt/e/Hexo-BotTrainer/runs/hexfield_main_1")
    ap.add_argument("--pool", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--anchors", default="bc_prefit,ep5")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    pool_path = Path(args.pool) if args.pool else (run_dir / "diagnostics" / "eval_pool.json")
    out_dir = Path(args.out) if args.out else (run_dir / "diagnostics" / "eval_graphs")
    out_dir.mkdir(parents=True, exist_ok=True)
    anchors = [a.strip() for a in args.anchors.split(",") if a.strip()]

    doc = mse._load_pool(pool_path)
    edges = doc.get("edges", [])
    cand_epochs = sorted({_cand_epoch(e["a"]) for e in edges if _cand_epoch(e["a"]) is not None})

    # Uniformity badge.
    offenders = []
    for e in edges:
        if e.get("kind") == "sealbot":
            continue
        raw = e.get("raw") or {}
        if int(raw.get("eval_visits") or 0) != LOCKED_VISITS or int(raw.get("virtual_batch_size") or 0) != LOCKED_VBS:
            offenders.append(f'ep{e.get("epoch")} {e.get("a")}-vs-{e.get("b")} '
                             f'(v={raw.get("eval_visits")},vbs={raw.get("virtual_batch_size")})')
    uniform = (len(offenders) == 0 and len(edges) > 0)

    # Fig (a): pooled BT Elo per epoch, refit as-of-epoch.
    elo_curve = []
    anchor_used = "bc_prefit"
    for ep in cand_epochs:
        fit, anchor = _fit_asof(doc, ep)
        if anchor:
            anchor_used = anchor
        lbl = f"cand_ep{ep}"
        if fit is None or lbl not in fit.players:
            continue
        lo, hi = fit.elo_ci(lbl)
        elo_curve.append({"epoch": ep, "elo": round(fit.rating(lbl), 1),
                          "lo": round(lo, 1), "hi": round(hi, 1), "se": round(fit.se(lbl), 1)})

    # Fig (b): winrate vs each fixed anchor.
    winrate_curves = {a: [] for a in anchors}
    for e in edges:
        ce = _cand_epoch(e["a"]) ; opp = e.get("b")
        if ce is None or opp not in winrate_curves:
            continue
        wr, lo, hi, decided, kind = _winrate_ci(e.get("raw") or {})
        if decided <= 0:
            continue
        winrate_curves[opp].append({"epoch": ce, "winrate": round(wr, 4),
                                    "lo": round(lo, 4), "hi": round(hi, 4),
                                    "decided": decided, "interval": kind})
    for a in winrate_curves:
        winrate_curves[a].sort(key=lambda r: r["epoch"])

    # Honest caption from the newest per-epoch JSON.
    note = "Single-epoch resolution is coarse (~100-150 Elo); tight resolution is the MULTI-EPOCH pooled asymptote."
    diag = sorted(run_dir.glob("diagnostics/hexfield.multistage_eval.epoch_*.json"))
    if diag:
        try:
            note = (json.loads(diag[-1].read_text(encoding="utf-8")).get("meta") or {}).get(
                "single_epoch_se_elo_note", note)
        except Exception:
            pass

    # SVGs.
    palette = {"bc_prefit": "#4f9dff", "ep5": "#ff9f43", "elo": "#42d392"}
    fig_a = _svg_line_chart(
        [{"name": f"pooled BT Elo (anchor={anchor_used})", "color": palette["elo"],
          "points": [(r["epoch"], r["elo"], r["lo"], r["hi"]) for r in elo_curve]}],
        y_domain=(min([r["lo"] for r in elo_curve] + [0]) - 30 if elo_curve else 0,
                  max([r["hi"] for r in elo_curve] + [0]) + 30 if elo_curve else 100),
        y_fmt=lambda v: f"{v:.0f}", title="Pooled Bradley-Terry Elo per epoch (refit as-of-epoch)",
        y_label="Elo (anchor=0)")
    fig_b_cards = []
    for a in anchors:
        pts = [(r["epoch"], r["winrate"], r["lo"], r["hi"]) for r in winrate_curves.get(a, [])]
        fig_b_cards.append(_svg_line_chart(
            [{"name": f"win-rate vs {a}", "color": palette.get(a, "#a0a0a0"), "points": pts}],
            y_domain=(0.0, 1.0), y_fmt=lambda v: f"{v*100:.0f}%",
            title=f"Candidate win-rate vs {a} (frozen anchor)", y_label="win rate", ref=0.5))

    badge = (f'<span class="badge ok">512v / vbs16 uniform ✓</span>' if uniform
             else f'<span class="badge bad">NOT uniform ✗ — {len(offenders)} off-config edge(s)</span>')
    offender_html = "" if uniform else "<details><summary>off-config edges</summary><ul>" + \
        "".join(f"<li>{o}</li>" for o in offenders) + "</ul></details>"

    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>hexfield eval — learning curve</title>
<style>
:root{{--bg:#0d1117;--fg:#e6edf3;--muted:#8b949e;--grid:#30363d;--card:#161b22}}
@media(prefers-color-scheme:light){{:root{{--bg:#fff;--fg:#1f2328;--muted:#656d76;--grid:#d0d7de;--card:#f6f8fa}}}}
body{{background:var(--bg);color:var(--fg);font-family:system-ui,sans-serif;margin:0;padding:24px}}
h1{{font-size:18px;margin:0 0 4px}} .sub{{color:var(--muted);font-size:12px;margin-bottom:16px}}
.card{{background:var(--card);border:1px solid var(--grid);border-radius:10px;padding:12px;margin-bottom:16px}}
.badge{{display:inline-block;padding:3px 10px;border-radius:999px;font-size:12px;font-weight:600}}
.badge.ok{{background:#15391f;color:#42d392}} .badge.bad{{background:#3a1518;color:#ff6b6b}}
.note{{color:var(--muted);font-size:11px;margin-top:8px;line-height:1.4}}
svg{{width:100%;height:auto}}
</style></head><body>
<h1>hexfield_main_1 — in-run eval learning curve</h1>
<div class="sub">512 sims / eval-vbs 16, fixed roster, Bradley-Terry pool · scale pinned to {anchor_used} (SealBot unavailable)</div>
<div>{badge} &nbsp; <span class="sub">{len(edges)} pooled edges · candidate epochs {cand_epochs}</span>{offender_html}</div>
<div class="card">{fig_a}<div class="note">Overlapping marginal CI bands here are NOT the difference test — the BT difference-CI (with +Cov) is the only significance claim. Per-epoch verdict is INCONCLUSIVE by design; the trend is the signal.</div></div>
{"".join(f'<div class="card">{c}</div>' for c in fig_b_cards)}
<div class="note">{note}</div>
</body></html>"""

    (out_dir / "eval_graphs.html").write_text(html, encoding="utf-8")
    machine = {
        "anchor_used": anchor_used,
        "uniform_512_vbs16": uniform,
        "off_config_edges": offenders,
        "n_pooled_edges": len(edges),
        "candidate_epochs": cand_epochs,
        "curve_elo": elo_curve,
        "winrate_curves": winrate_curves,
    }
    (out_dir / "eval_graphs.json").write_text(json.dumps(machine, indent=2), encoding="utf-8")
    print(f"wrote {out_dir/'eval_graphs.html'}")
    print(f"wrote {out_dir/'eval_graphs.json'}")
    print(f"uniform_512_vbs16={uniform}  anchor={anchor_used}  epochs={cand_epochs}")
    if not uniform:
        print(f"  WARNING off-config edges: {offenders}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
