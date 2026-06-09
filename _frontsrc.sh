#!/usr/bin/env bash
p=$(pgrep -f 'hexo_frontend.web' | head -1)
echo "8080 pid=$p"
echo "cwd: $(readlink /proc/$p/cwd 2>/dev/null)"
echo "--- PYTHONPATH / VENV ---"
tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -iE 'PYTHONPATH|VIRTUAL_ENV'
echo "--- which hexo_frontend module would this venv import ---"
ve=$(tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -i VIRTUAL_ENV | cut -d= -f2)
"$ve/bin/python" -c "import hexo_frontend, hexo_frontend.web as w; print('hexo_frontend.web file:', w.__file__)" 2>/dev/null || echo "(could not import)"
