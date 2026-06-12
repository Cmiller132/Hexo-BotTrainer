"""Small CLI placeholder for programmatic runner use.

Registered as the `hexo-rl` console script in pyproject.toml, but the runner
has no CLI surface: `main()` exists only to point callers at the programmatic
API (`hexo_runner.modes.run_match` / `run_batch`).
"""

from __future__ import annotations


# UNUSED(2026-06-12): only reference is the `hexo-rl` script entry in
# packages/hexo_runner/pyproject.toml; no caller in packages/tests/scripts
# (excl. scripts/archive). Deliberate error-message placeholder, not a CLI.
def main() -> int:
    raise SystemExit(
        "hexo_runner is a programmatic game runner. Build RunnerPlayer "
        "instances in Python and call hexo_runner.modes.run_match or run_batch."
    )
