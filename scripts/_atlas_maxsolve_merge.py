"""Merge strict-verified opening-atlas shard output into a decisive-only raw.

The Rust harness emits a row only after its canonical certificate passes the
normative TssVerifier.  This merger adds schema/frozen-root checks and never
turns an already decisive atlas row into an incoming upgrade.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re


def read_text(path: str) -> str:
    with open(path, "rb") as fh:
        data = fh.read()
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return data.decode("utf-16")
    return data.decode("utf-8-sig")


def fields(line: str) -> dict[str, str]:
    return dict(token.split("=", 1) for token in line.split() if "=" in token)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", action="append", required=True, dest="patterns")
    parser.add_argument("--atlas", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--attempted-out")
    parser.add_argument("--method", required=True)
    parser.add_argument("--expected", type=int)
    args = parser.parse_args()

    paths = sorted({path for pattern in args.patterns for path in glob.glob(pattern)})
    assert paths, "no shard files matched"
    with open(args.atlas, encoding="utf-8") as fh:
        atlas = json.load(fh)
    atlas_by_id = {row["id"]: row for row in atlas["rows"]}
    assert len(atlas_by_id) == len(atlas["rows"]), "duplicate atlas ids"

    setups: list[str] = []
    done: list[str] = []
    stats_by_id: dict[str, dict[str, str]] = {}
    decisive: dict[str, str] = {}
    attempted_rows: dict[str, str] = {}
    rows_seen = 0
    verifier_reject_markers = 0
    for path in paths:
        for line in read_text(path).splitlines():
            line = line.rstrip("\r")
            if "ATLAS_SETUP " in line:
                setups.append(line[line.index("ATLAS_SETUP "):])
            elif "ATLAS_DONE " in line:
                done.append(line[line.index("ATLAS_DONE "):])
            elif "ATLAS_D6_REMAP_REJECT " in line:
                # These are symmetry-remap diagnostics, not canonical verifier
                # rejects.  The atlas schema records their accepted D6 mask.
                verifier_reject_markers += 1
            elif line.startswith("ATLAS_STATS "):
                rec = fields(line)
                stats_by_id[rec["id"]] = rec
            elif line.startswith("ATLAS_ROW "):
                rows_seen += 1
                rec = fields(line)
                previous_attempt = attempted_rows.get(rec["id"])
                if previous_attempt is not None:
                    assert previous_attempt == line, f"attempted row drift for {rec['id']}"
                attempted_rows[rec["id"]] = line
                if rec["status"] not in ("WIN", "LOSS"):
                    continue
                assert rec["certified"] == "1", f"decisive row not certified: {rec['id']}"
                assert int(rec["d6_verified"]) >= 1, f"identity verifier missing: {rec['id']}"
                if rec["status"] == "WIN":
                    assert rec["win_line_terminal"] == "1", (
                        f"WIN lacks concrete terminal line: {rec['id']}"
                    )
                    assert int(rec["win_line_len"]) > 0
                old = atlas_by_id.get(rec["id"])
                assert old is not None, f"incoming id absent from frozen atlas: {rec['id']}"
                assert old["status"] == "UNKNOWN" and old["certified"] == 0, (
                    f"incoming row is not UNKNOWN->decisive: {rec['id']}"
                )
                assert rec["moves"] == ";".join(f"{q},{r}" for q, r in old["moves"])
                assert int(rec["placements"]) == old["placements"]
                assert int(rec["source_prefix"]) == old["source_prefix"]
                assert rec["side"] == old["side"]
                assert rec["phase"] == old["phase"]
                expected_claimant = (
                    old["side"]
                    if rec["status"] == "WIN"
                    else ("P1" if old["side"] == "P0" else "P0")
                )
                assert rec["claimant"] == expected_claimant
                previous = decisive.get(rec["id"])
                if previous is not None:
                    assert fields(previous)["status"] == rec["status"], (
                        f"decisive conflict for {rec['id']}"
                    )
                    # Keep the smaller accepted certificate/search result.
                    if int(fields(previous)["cert_nodes"]) <= int(rec["cert_nodes"]):
                        continue
                decisive[rec["id"]] = line

    shard_attempted = residual = 0
    for marker in done:
        match = re.search(r"attempted=(\d+) residual=(\d+)", marker)
        assert match, marker
        shard_attempted += int(match.group(1))
        residual += int(match.group(2))
    attempted = len(attempted_rows)
    if args.expected is not None:
        assert attempted <= args.expected
        residual = args.expected - attempted
    status = {"WIN": 0, "LOSS": 0}
    nodes = []
    fragment = {
        "lookups": sum(int(stat["fragment_lookups"]) for stat in stats_by_id.values()),
        "hits": sum(int(stat["fragment_hits"]) for stat in stats_by_id.values()),
        "imports": sum(int(stat["fragment_imports"]) for stat in stats_by_id.values()),
    }
    for row in decisive.values():
        rec = fields(row)
        status[rec["status"]] += 1
        nodes.append(int(rec["nodes"]))
    nodes.sort()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(
            "ATLAS_MAXSOLVE_SETUP schema=1 "
            f"method={args.method} shards={len(paths)} rows_seen={rows_seen} "
            f"attempted={attempted} residual={residual}\n"
        )
        for row in sorted(decisive.values(), key=lambda value: fields(value)["id"]):
            fh.write(row + "\n")
        fh.write(
            "ATLAS_MAXSOLVE_DONE "
            f"method={args.method} win={status['WIN']} loss={status['LOSS']} "
            f"decisive={len(decisive)} attempted={attempted} residual={residual} "
            "canonical_verifier_rejects=0 "
            f"d6_remap_reject_diagnostics={verifier_reject_markers} "
            f"fragment_lookups={fragment['lookups']} fragment_hits={fragment['hits']} "
            f"fragment_imports={fragment['imports']}\n"
        )

    if args.attempted_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.attempted_out)), exist_ok=True)
        with open(args.attempted_out, "w", encoding="utf-8", newline="\n") as fh:
            for row in sorted(attempted_rows.values(), key=lambda value: fields(value)["id"]):
                fh.write(row + "\n")
            fh.write(
                f"ATLAS_ATTEMPTED_DONE method={args.method} rows={len(attempted_rows)}\n"
            )

    median = nodes[len(nodes) // 2] if nodes else 0
    maximum = nodes[-1] if nodes else 0
    print(
        json.dumps(
            {
                "method": args.method,
                "shards": len(paths),
                "setup_markers": len(setups),
                "done_markers": len(done),
                "rows_seen": rows_seen,
                "attempted": attempted,
                "shard_attempted_markers_sum": shard_attempted,
                "residual": residual,
                "win": status["WIN"],
                "loss": status["LOSS"],
                "decisive": len(decisive),
                "nodes_median": median,
                "nodes_max": maximum,
                "d6_remap_reject_diagnostics": verifier_reject_markers,
                "fragment": fragment,
                "out": os.path.abspath(args.out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
