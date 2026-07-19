/* build_frequencies.mjs — human-corpus usage counts per canonical opening.
 *
 * For every game in the bootstrap human corpus, replay each stone-placement
 * prefix of length 1..MAX_DEPTH, canonicalize the resulting root position with the
 * SITE'S OWN d6.js canonicalId (the exact identity the atlas rows were minted
 * against), and tally how many DISTINCT games pass through each canonical id.
 *
 * Because every D6 image of a prefix collapses to one canonical id, mirror /
 * rotation-distinct human move-orders that reach the same position sum into a
 * single counter (the "reduce duplicated work via symmetry" requirement,
 * applied to counting).
 *
 * Output: data/frequencies.json
 *   { schema, total_games, generated_from, counts:{id:games}, by_depth:{id:{depth:games}} }
 * PLUS the slim, pre-gzipped web derivative the site actually fetches:
 *   data/frequencies-web.{json,json.gz,jsonp.js}  (schema/total_games/
 *   generated_from/counts only — no by_depth). build_atlas_json.py regenerates
 *   this trio from frequencies.json ONLY on a from-scratch (non-frozen) build;
 *   once the atlas carries a frozen expansion base it preserves whatever exists
 *   byte-for-byte, so emitting it HERE is what actually carries an updated depth
 *   (and every loss's new deep count) into the browser. Format matches
 *   build_atlas_json.py's write_split exactly.
 *
 * Run:  node build_frequencies.mjs
 */
import { readFileSync, writeFileSync } from "node:fs";
import { gzipSync } from "node:zlib";
import { canonicalId } from "./d6.js";

const CORPUS =
  "E:/Hexo-BotTrainer-hexgt/data/hexo-bootstrap-corpus/hexo_human_corpus.jsonl";
const OUT = new URL("./data/frequencies.json", import.meta.url);
// Cover the atlas's full opening depth. The certified LOSS rows (and the deepest
// certified WINs) all sit at ply 9-11; a 7-ply cap gave every one of them NO
// usage count. Counting through 11 gives those deep rows an honest "used N×"
// badge. A prefix longer than a game simply isn't counted (depthLimit clamps to
// moves.length), so the badge stays ABSENT — not "0" — for openings the corpus
// never reaches at that depth.
const MAX_DEPTH = 11;

const raw = readFileSync(CORPUS, "utf8");
const lines = raw.split("\n").filter((l) => l.trim().length > 0);

// counts[id]      = # distinct games that pass through id at ANY depth.
// byDepth[id][d]  = # distinct games that pass through id at exactly depth d.
// idDepth[id]     = the (unique) depth an id belongs to (sanity check).
const counts = new Map();
const byDepth = new Map();
const idDepth = new Map();

let total = 0;
let skipped = 0;

for (const line of lines) {
  let game;
  try {
    game = JSON.parse(line);
  } catch {
    skipped++;
    continue;
  }
  const moves = game.moves;
  if (!Array.isArray(moves) || moves.length < 1) {
    skipped++;
    continue;
  }
  total++;

  // Distinct-id guard: one game contributes at most once to any (id) / (id,depth).
  // (A prefix length is unique within a game, and each canonical id belongs to a
  //  single depth, so this is belt-and-suspenders — but it makes "DISTINCT games"
  //  exact regardless.)
  const seenOverall = new Set();
  const depthLimit = Math.min(MAX_DEPTH, moves.length);

  // depth 0 = the empty root (position before any stone). Every game passes
  // through it, so its counter equals total_games — a fixed verification anchor.
  for (let depth = 0; depth <= depthLimit; depth++) {
    const prefix = moves.slice(0, depth);
    const id = canonicalId(prefix).id;

    // per-depth distinct
    let dm = byDepth.get(id);
    if (!dm) {
      dm = new Map();
      byDepth.set(id, dm);
    }
    const seenKey = depth;
    // within one game a given depth yields exactly one prefix -> one id, so no
    // intra-game dedup needed per depth; still track to be provably correct.
    dm.set(seenKey, (dm.get(seenKey) || 0) + 1);

    // overall distinct (dedup within the game)
    if (!seenOverall.has(id)) {
      seenOverall.add(id);
      counts.set(id, (counts.get(id) || 0) + 1);
    }

    // depth-uniqueness sanity
    const known = idDepth.get(id);
    if (known === undefined) idDepth.set(id, depth);
    else if (known !== depth) {
      console.error(
        `WARN id ${id} seen at depth ${known} AND ${depth} (hash collision across depths?)`
      );
    }
  }
}

// Fold by_depth into a plain object; also verify overall == the id's single depth.
const countsObj = {};
for (const [id, c] of counts) countsObj[id] = c;

const byDepthObj = {};
for (const [id, dm] of byDepth) {
  const o = {};
  for (const [d, c] of dm) o[d] = c;
  byDepthObj[id] = o;
}

const out = {
  schema: 1,
  total_games: total,
  generated_from: `corpus first-${MAX_DEPTH}-ply, D6-collapsed`,
  counts: countsObj,
  by_depth: byDepthObj,
};

writeFileSync(OUT, JSON.stringify(out));

// Slim, pre-gzipped web derivative (counts + totals only; by_depth dropped),
// emitted as the three forms loadDoc() understands: plain .json, gzip fast path
// .json.gz (inflated in-browser via DecompressionStream), and a window-global
// .jsonp.js file:// shim. Byte-for-byte the shape build_atlas_json.py's
// write_split(FREQ_WEB_BASE, ...) produces, so the two agree on either build path.
const webObj = {
  schema: out.schema,
  total_games: out.total_games,
  generated_from: out.generated_from,
  counts: out.counts,
};
const webTxt = JSON.stringify(webObj);
const webRaw = Buffer.from(webTxt, "utf8");
const webUrl = (ext) => new URL("./data/frequencies-web" + ext, import.meta.url);
writeFileSync(webUrl(".json"), webRaw);
writeFileSync(webUrl(".json.gz"), gzipSync(webRaw, { level: 9 }));
writeFileSync(webUrl(".jsonp.js"), "window.__ATLAS_FREQ__ = " + webTxt + ";\n");

console.error(
  `games=${total} skipped=${skipped} distinct_ids=${counts.size} ` +
    `wrote ${new URL(OUT).pathname} + frequencies-web.{json,json.gz,jsonp.js}`
);
