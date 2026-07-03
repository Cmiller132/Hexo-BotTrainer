"""Aggregate a py-spy raw (collapsed) profile: leaf self-time + keyword buckets."""
import re
import sys
from collections import Counter

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/pyspy_profile.txt"
leaf = Counter()
buckets = Counter()
total = 0
KEYS = [
    "submit_payload", "_run_forward", "_submit_rust_pack", "frombuffer",
    "cudaMemcpy", "LaunchKernel", "cuda", "guard", "featurize", "plan_groups",
    "_decode_group", "result", "hex_conv", "flex", "sympy", "shape_env",
    "recompil", "inductor", "_dynamo", "backup", "select", "pin_memory",
    "cudaStreamSynchronize", "cudaHostAlloc", "acquire", "GIL",
]
with open(path) as f:
    for line in f:
        line = line.rstrip()
        m = re.match(r"^(.*) (\d+)$", line)
        if not m:
            continue
        stack, n = m.group(1), int(m.group(2))
        total += n
        frames = stack.split(";")
        leaf[frames[-1][:120]] += n
        seen = set()
        for fr in frames:
            for k in KEYS:
                if k.lower() in fr.lower() and k not in seen:
                    buckets[k] += n
                    seen.add(k)

print(f"total samples: {total}")
print("\n-- keyword buckets (any frame) --")
for k, n in buckets.most_common(25):
    print(f"{100*n/total:5.1f}%  {k}")
print("\n-- top leaves (self) --")
for fr, n in leaf.most_common(25):
    print(f"{100*n/total:5.1f}%  {fr}")
