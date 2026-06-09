#!/bin/bash
echo "=== any hexgt RL / bridge / supervisor procs? ==="
ps -eo pid,etimes,args | grep -E "_rl_train.py|_rl_supervise|_dashboard_bridge|_rl_run_fg" | grep -v grep
echo "(blank above = none running)"
