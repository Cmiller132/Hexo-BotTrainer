/* mini-board.js — compact static hex-board thumbnail for atlas browse rows.
 *
 * Pure string builder: renders the placed stones (owner = Hexo turn order) at
 * their true axial positions, fit to a square viewBox over a faint board of
 * per-stone cells. No interactivity, no imports — safe to call lazily per row.
 * Reuses board.js axial->pixel geometry (S is local; viewBox fit cancels it).
 */
const SQ3 = Math.sqrt(3);
const S = 10, COL_W = S * SQ3, ROW_H = S * 1.5;      // same ratios as board.js
const ax = (q, r) => COL_W * (q + r / 2);
const ay = r => ROW_H * r;

// Hexo owner: move0=P0, then P1,P1,P0,P0,... (byte-identical to d6.js ownerAt)
const ownerAt = i => (i === 0 ? 0 : (Math.floor((i - 1) / 2) % 2 === 0 ? 1 : 0));

function hexPts(cx, cy, rad) {
  let p = "";
  for (let i = 0; i < 6; i++) {
    const a = Math.PI / 180 * (60 * i - 30);
    p += (cx + rad * Math.cos(a)).toFixed(1) + "," + (cy + rad * Math.sin(a)).toFixed(1) + " ";
  }
  return p.trim();
}

/* moves: [[q,r],...] in placement order. px: rendered square size. */
export function miniBoardSVG(moves, px = 44) {
  // Empty root: faint single cell + tengen dot, framed on origin.
  if (!moves || !moves.length) {
    const p = S * 1.6;
    const vb = `${(-p).toFixed(1)} ${(-p).toFixed(1)} ${(2 * p).toFixed(1)} ${(2 * p).toFixed(1)}`;
    return `<svg class="mini-board" width="${px}" height="${px}" viewBox="${vb}" ` +
      `preserveAspectRatio="xMidYMid meet" aria-hidden="true">` +
      `<g class="mini-cells"><polygon points="${hexPts(0, 0, S * 0.96)}"/></g>` +
      `<circle class="mini-tengen" cx="0" cy="0" r="${(S * 0.28).toFixed(1)}"/></svg>`;
  }

  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const [q, r] of moves) {
    const x = ax(q, r), y = ay(r);
    if (x < x0) x0 = x; if (x > x1) x1 = x;
    if (y < y0) y0 = y; if (y > y1) y1 = y;
  }
  const pad = S * 1.25;
  const vx = x0 - pad, vy = y0 - pad;
  const vw = (x1 - x0) + 2 * pad, vh = (y1 - y0) + 2 * pad;

  let cells = "", stones = "";
  const rStone = (S * 0.62).toFixed(1);
  for (let i = 0; i < moves.length; i++) {
    const [q, r] = moves[i];
    const x = ax(q, r), y = ay(r);
    cells += `<polygon points="${hexPts(x, y, S * 0.96)}"/>`;
    stones += `<circle class="${ownerAt(i) === 0 ? "m0" : "m1"}" ` +
      `cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${rStone}"/>`;
  }

  const vb = `${vx.toFixed(1)} ${vy.toFixed(1)} ${vw.toFixed(1)} ${vh.toFixed(1)}`;
  return `<svg class="mini-board" width="${px}" height="${px}" viewBox="${vb}" ` +
    `preserveAspectRatio="xMidYMid meet" aria-hidden="true">` +
    `<g class="mini-cells">${cells}</g>` +
    `<g class="mini-stones">${stones}</g></svg>`;
}
