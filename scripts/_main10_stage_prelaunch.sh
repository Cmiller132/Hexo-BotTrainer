#!/usr/bin/env bash
# main_10 pre-launch staging (run from WSL): release engine build into the
# hexgt-build venv + ep25 weights repackage. Idempotent; does NOT launch.
set -euo pipefail
ROOT="${ROOT:-/mnt/e/Hexo-BotTrainer-hexgt/.claude/worktrees/consolidate-main}"

echo "=== maturin release build (stale-.pth trap guard) ==="
cd "$ROOT/packages/hexfield_eq"
source /root/.venvs/hexgt-build/bin/activate
maturin develop --release --features python 2>&1 | tail -4

echo "=== import sanity ==="
python - <<'EOF'
import hexfield_eq._rust as r
names = dir(r)
print("native module:", r.__file__ if hasattr(r, "__file__") else "builtin")
print("deep solve seam present:", any("deep" in n for n in names))
EOF

echo "=== stale-.pth / duplicate _rust check ==="
python - <<'EOF'
import glob, sysconfig
sp = sysconfig.get_paths()["purelib"]
dups = [p for p in glob.glob(sp + "/**/_rust*", recursive=True) if "hexfield_eq" not in p]
print("foreign _rust artifacts in site-packages:", dups if dups else "none")
EOF

echo "=== repackage main_4 ep25 -> soak_init ==="
ROOT="$ROOT" bash "$ROOT/scripts/_main10_repackage.sh"

echo "MAIN10_STAGING_EXIT=0"
