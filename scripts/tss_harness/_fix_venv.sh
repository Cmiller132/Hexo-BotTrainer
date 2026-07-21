#!/bin/bash
# Repair twopass-dev venv .pth entries (shell-nesting CR mangling workaround).
SP=/root/.venvs/twopass-dev/lib/python3.12/site-packages
ls -la "$SP" | cat -A | grep -i pth
rm -f "$SP"/hexo_engine.pth "$SP"/hexo_utils.pth
find "$SP" -maxdepth 1 -name "*pth*" -print
printf '/mnt/e/Hexo-BotTrainer-hexgt/packages/hexo_engine/python\n' > "$SP/hexo_engine.pth"
printf '/mnt/e/hexo-bot/packages/hexo_utils/python\n' > "$SP/hexo_utils.pth"
/root/.venvs/twopass-dev/bin/python -c 'import hexo_engine, hexo_utils; print("imports OK:", hexo_engine.__file__)'
