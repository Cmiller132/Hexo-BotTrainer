#!/usr/bin/env python3
"""Source and LLVM-callgraph firewall for RefuteLeafExact/V1.

Run after `cargo rustc ... -- --emit=llvm-ir`. With no argument, source-only
checks run. With one or more `.ll` paths, the script walks direct LLVM call
edges from every public refute-verifier root and rejects reachable forbidden
module symbols. Optimizer inlining is conservative: inlined forbidden names
remain in the root definition and are checked as text as well.
"""

from __future__ import annotations

import pathlib
import re
import sys
import hashlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "packages/hexfield_eq/rust/src/tss_refute_verify.rs"
FORBIDDEN = (
    "tss_solver", "tss_verify17", "tss_verify::", "threats_shared",
    "WindowStore", "WidePnSearch", "WidthOptions",
)


def source_check():
    text = SOURCE.read_text(encoding="utf-8")
    imports = "\n".join(line for line in text.splitlines() if line.lstrip().startswith("use "))
    for name in FORBIDDEN:
        if name in imports:
            raise SystemExit(f"firewall source failure: {name}")


def llvm_check(path: pathlib.Path):
    text = path.read_text(encoding="utf-8", errors="replace")
    definitions = {}
    current = None
    body = []
    for line in text.splitlines():
        match = re.match(r"define\b.*?@([^ (]+)\(", line)
        if match:
            current, body = match.group(1), [line]
            continue
        if current is not None:
            body.append(line)
            if line == "}":
                definitions[current] = "\n".join(body)
                current = None
    roots = [name for name in definitions if "tss_refute_verify" in name and "verify_refute_leaf_exact_v1" in name]
    if not roots:
        raise SystemExit(f"firewall callgraph failure: verifier root absent in {path}")
    calls = {name: set(re.findall(r"\bcall\b.*?@([^ (]+)\(", body)) for name, body in definitions.items()}
    reached, pending = set(), list(roots)
    while pending:
        name = pending.pop()
        if name in reached:
            continue
        reached.add(name)
        pending.extend(calls.get(name, set()) - reached)
    reached_text = "\n".join(name + "\n" + definitions.get(name, "") for name in reached)
    for forbidden in FORBIDDEN:
        if forbidden in reached_text:
            raise SystemExit(f"firewall callgraph failure: reachable {forbidden}")
    print(f"FIREWALL_LL_OK file={path} roots={len(roots)} reachable={len(reached)}")


def main():
    source_check()
    allowlist_hash = hashlib.sha256("\0".join(FORBIDDEN).encode()).hexdigest()
    print(f"FIREWALL_SOURCE_OK denylist_sha256={allowlist_hash}")
    for item in sys.argv[1:]:
        llvm_check(pathlib.Path(item))


if __name__ == "__main__":
    main()
