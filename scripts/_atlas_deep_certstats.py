import re, os

BASE = r"E:/Hexo-BotTrainer-hexgt/.claude/worktrees/opening-atlas"
RAW = os.path.join(BASE, "OPENING_ATLAS_CORPUS7_DEEP_RAW.txt")


def field(line, key):
    m = re.search(rf" {key}=([^\s]+)", line)
    return m.group(1) if m else None


cert = []
with open(RAW, encoding="utf-8") as f:
    for line in f:
        if not line.startswith("ATLAS_ROW "):
            continue
        if field(line, "certified") == "1":
            cert.append(line.rstrip("\n"))

by_depth = {}
by_claim = {}
for r in cert:
    d = int(field(r, "source_prefix"))
    by_depth[d] = by_depth.get(d, 0) + 1
    c = field(r, "claimant")
    by_claim[c] = by_claim.get(c, 0) + 1

print("certified total:", len(cert))
print("certified by depth:", {k: by_depth[k] for k in sorted(by_depth)})
print("certified by claimant:", by_claim)
print("all certified status WIN:", all(field(r, "status") == "WIN" for r in cert))

# shallowest certified wins (fewest stones) — the most striking openings
cert.sort(key=lambda r: (int(field(r, "source_prefix")), int(field(r, "nodes"))))
print("\n-- shallowest / example certified openings --")
for r in cert[:8]:
    print(f"depth={field(r,'source_prefix')} side={field(r,'side')} phase={field(r,'phase')} "
          f"claimant={field(r,'claimant')} nodes={field(r,'nodes')} cert_nodes={field(r,'cert_nodes')} "
          f"derived_horizon={field(r,'derived_horizon')} d6={field(r,'d6_verified')}/12 "
          f"moves={field(r,'moves')}")
