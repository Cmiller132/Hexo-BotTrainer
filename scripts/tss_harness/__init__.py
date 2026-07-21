"""TSS solver comparison harness — docs/PLAN_TSS_HARNESS.md.

Core is solver-agnostic (contract.py, gates.py, archive, diff); solvers plug
in as adapters. Runs in the harness-dev WSL venv (/root/.venvs/harness-dev,
wheel-installed from /root/harness-wheels — NEVER the live hexfield-dev
editable install, whose in-tree .so a running eval may have mapped).
"""
