/* d6.js — pure geometry + D6-canonical FNV-1a-64 id for Hexo root positions.
 *
 * Byte-for-byte the identity the certificates were minted against:
 *   root_position_key  ->  Rust {:?} (Debug)  ->  FNV-1a-64  ->  "oa-<16hex>"
 * (tss_opening_atlas.rs root_position_key/fnv1a64, tss_verify.rs d6_transform_coord).
 * Validated to reproduce all 122 stored ids in data/atlas.json.
 *
 * No DOM here — importable from the browser AND from Node for the self-check.
 */

// Apply one of the 12 D6 symmetries to an axial coordinate. sym in [0,12).
export function transform(q, r, sym) {
  if (sym >= 6) r = -q - r;              // reflection first
  for (let i = 0; i < sym % 6; i++) {    // then rotate sym%6 times
    const nq = -r, nr = q + r;
    q = nq; r = nr;
  }
  return [q, r];
}

// FNV-1a-64 over UTF-8 bytes -> 16-char lowercase hex (Rust {:016x}).
const _FNV_MASK = (1n << 64n) - 1n;
const _FNV_PRIME = 0x100000001b3n;
export function fnv1a64(str) {
  let h = 0xcbf29ce484222325n;
  for (const b of new TextEncoder().encode(str)) {
    h ^= BigInt(b);
    h = (h * _FNV_PRIME) & _FNV_MASK;
  }
  return h.toString(16).padStart(16, "0");
}

/* Owner of the stone at placement index i, from Hexo turn order:
 * P0 plays the single opening stone (i=0), then players alternate two-per-turn
 * (P1 places i=1,2 · P0 places i=3,4 · P1 places i=5,6 · …). */
export function ownerAt(i) {
  if (i === 0) return 0;
  return (Math.floor((i - 1) / 2) % 2 === 0) ? 1 : 0;
}

/* Derive (side-to-move, phase, first-witness) for a legal Hexo sequence of n
 * placed stones — the exact state whose root_position_key was hashed. */
export function deriveBinding(moves) {
  const n = moves.length;
  const side = ownerAt(n);                 // owner of the next stone = side to move
  if (n === 0) return { side, phase: "Opening", first: null };
  const within = (n - 1) % 2;              // 0 -> starting a pair, 1 -> mid-pair
  if (within === 0) return { side, phase: "FirstStone", first: null };
  return { side, phase: "SecondStone", first: moves[n - 1] };
}

function phaseTriple(phase, first, sym) {
  if (phase === "Opening") return [0, 0, 0];
  if (phase === "FirstStone") return [1, 0, 0];
  const [q, r] = transform(first[0], first[1], sym);   // SecondStone
  return [2, q, r];
}

// root_position_key for ONE symmetry. moves: [[q,r],...] in placement order.
function positionKeyForSym(moves, side, phase, first, sym) {
  const ph = phaseTriple(phase, first, sym);
  const st = moves
    .map(([q, r], i) => { const [tq, tr] = transform(q, r, sym); return [tq, tr, ownerAt(i)]; })
    .sort((a, b) => (a[0] - b[0]) || (a[1] - b[1]));
  return [side, ph, st];
}

// Rust tuple/Vec lexicographic Ord.
function cmpKey(a, b) {
  if (a[0] !== b[0]) return a[0] - b[0];
  for (let i = 0; i < 3; i++) if (a[1][i] !== b[1][i]) return a[1][i] - b[1][i];
  const A = a[2], B = b[2];
  for (let i = 0; i < Math.min(A.length, B.length); i++)
    for (let j = 0; j < 3; j++) if (A[i][j] !== B[i][j]) return A[i][j] - B[i][j];
  return A.length - B.length;
}

// Render exactly like Rust {:?} for (u8,(u8,i16,i16),Vec<(i16,i16,u8)>).
function debugKey(key) {
  const [player, ph, st] = key;
  const stones = st.map(([q, r, o]) => `(${q}, ${r}, ${o})`).join(", ");
  return `(${player}, (${ph[0]}, ${ph[1]}, ${ph[2]}), [${stones}])`;
}

// Canonical id: least root_position_key over the 12 D6 images, hashed.
export function canonicalId(moves) {
  const { side, phase, first } = deriveBinding(moves);
  let best = null;
  for (let sym = 0; sym < 12; sym++) {
    const k = positionKeyForSym(moves, side, phase, first, sym);
    if (best === null || cmpKey(k, best) < 0) best = k;
  }
  return { id: "oa-" + fnv1a64(debugKey(best)), debug: debugKey(best), binding: { side, phase, first } };
}
