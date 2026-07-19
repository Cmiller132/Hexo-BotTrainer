/* selfcheck.mjs — offline verification of the atlas site's data + canonicalizer.
 * Run:  node selfcheck.mjs
 * Verifies: atlas.json loads; row count matches summary; the browser's exact
 * canonicalId() (from d6.js) maps every stored row to its own id.
 */
import { readFileSync } from "node:fs";
import { canonicalId } from "./d6.js";

const atlas = JSON.parse(readFileSync(new URL("./data/atlas.json", import.meta.url)));
let fail = 0;
const ok = (name, cond) => { console.log(`${cond ? "PASS" : "FAIL"}  ${name}`); if (!cond) fail++; };

// 1. structure
ok("atlas.json parsed", !!atlas && Array.isArray(atlas.rows));
ok(`row count (${atlas.rows.length}) == summary.total (${atlas.summary.total})`,
   atlas.rows.length === atlas.summary.total);

// 2. summary counts match the actual rows
const cnt = { WIN: 0, LOSS: 0, UNKNOWN: 0 };
let certified = 0;
for (const r of atlas.rows) { cnt[r.status]++; if (r.certified === 1) certified++; }
ok(`win ${cnt.WIN} == summary.win ${atlas.summary.win}`, cnt.WIN === atlas.summary.win);
ok(`loss ${cnt.LOSS} == summary.loss ${atlas.summary.loss}`, cnt.LOSS === atlas.summary.loss);
ok(`unknown ${cnt.UNKNOWN} == summary.unknown ${atlas.summary.unknown}`, cnt.UNKNOWN === atlas.summary.unknown);
ok(`certified ${certified} == summary.certified ${atlas.summary.certified}`, certified === atlas.summary.certified);
ok("certified == WIN+LOSS", certified === cnt.WIN + cnt.LOSS);

// 3. census constants present
ok("census ply2_d6 == 24", atlas.census.ply2_d6 === 24);
ok("census ply3_d6 == 3684", atlas.census.ply3_d6 === 3684);

// 4. THE canonicalizer maps every row to its own id (browser code, verbatim)
let idOK = 0, idBad = 0;
const badSamples = [];
for (const r of atlas.rows) {
  const got = canonicalId(r.moves).id;
  if (got === r.id) idOK++;
  else { idBad++; if (badSamples.length < 5) badSamples.push(`${r.id} != ${got} (plc ${r.placements})`); }
}
ok(`canonicalId round-trips all rows (${idOK}/${atlas.rows.length})`, idBad === 0);
for (const b of badSamples) console.log("      " + b);

// 5. spot-check: >= 3 certified rows map to themselves (explicit requirement)
const certRows = atlas.rows.filter(r => r.certified === 1).slice(0, 5);
let cOK = 0;
for (const r of certRows) if (canonicalId(r.moves).id === r.id) cOK++;
ok(`>=3 certified rows canonicalize to themselves (${cOK}/${certRows.length})`, cOK >= 3);

// 6. empty root parity
const emptyRow = atlas.rows.find(r => r.moves.length === 0);
if (emptyRow) ok(`empty-root id ${emptyRow.id}`, canonicalId([]).id === emptyRow.id);

console.log(fail === 0 ? "\nALL CHECKS PASSED" : `\n${fail} CHECK(S) FAILED`);
process.exit(fail === 0 ? 0 : 1);
