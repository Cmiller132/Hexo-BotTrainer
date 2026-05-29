# WSL2 CUDA + torch environment for inference-backend benchmarks

Verified working on this box (2026-05-29). This is the env the `torch.compile`
variant runs in (native-Windows torch 2.10 has no working Inductor/Triton path,
so compile is exercised under WSL2 per the user).

## Verified state

| component | value |
|---|---|
| WSL distro | Ubuntu-24.04 (WSL2), `wsl -l -v` → Version 2 |
| GPU visible in WSL | `nvidia-smi -L` → NVIDIA GeForce RTX 4070 Ti (Ada, sm_89) |
| host driver | 595.71 (CUDA 13-capable; WSL passes the GPU through) |
| venv | `/root/.venvs/hexo-bottrainer-wsl` |
| Python | 3.12.3 |
| torch | **2.11.0+cu128** (CUDA build 12.8) |
| triton | 3.6.0 (ships with torch; required by torch.compile/Inductor) |
| numpy | 2.4.6 |
| toolchain | `/usr/bin/gcc`, `/usr/bin/g++` present (Inductor C++ codegen) |

Note: WSL torch (2.11+cu128) is a **newer** build than native Windows
(2.10+cu126). So a WSL-vs-native comparison conflates the runtime/OS AND the
torch version — always measure a WSL **eager fp16** baseline next to the WSL
compile number to isolate the compiler's contribution (the harness does this).

## Reproduce from scratch

```bash
# from Windows (PowerShell) — ensure the distro exists and sees the GPU
wsl -l -v
wsl -d Ubuntu-24.04 -- nvidia-smi -L

# inside WSL: create the venv (system has python3.12 + python3.12-venv)
wsl -d Ubuntu-24.04 -- bash -lc '
  python3 -m venv /root/.venvs/hexo-bottrainer-wsl
  /root/.venvs/hexo-bottrainer-wsl/bin/pip install --upgrade pip
  # torch with the cu128 wheels (Triton comes bundled on Linux):
  /root/.venvs/hexo-bottrainer-wsl/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
  /root/.venvs/hexo-bottrainer-wsl/bin/pip install numpy
'

# sanity
wsl -d Ubuntu-24.04 -- /root/.venvs/hexo-bottrainer-wsl/bin/python -c \
  "import torch,triton; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), triton.__version__)"
```

The repo's `scripts/run_model1_wsl_smoke.sh` expects this same venv
(`VENV=/root/.venvs/hexo-bottrainer-wsl`) and a CUDA-capable torch inside it; it
runs a one-epoch resume smoke through the real training CLI.

## Running the dense_cnn code under WSL (no rebuild needed for pure-Python paths)

The benchmark harness + torch.compile variant are pure PyTorch and the model is
pure-Python (`Model1Network`), so they run under WSL **without** building the
Rust extension: importing `hexo_models.dense_cnn` is safe because `rust_bridge`
guards its `_rust` import in a try/except. PYTHONPATH must point at the worktree
package `python/` dirs (the repo is visible at `/mnt/e/Hexo-BotTrainer`):

```bash
cd /mnt/e/Hexo-BotTrainer
# the harness inserts the package python dirs itself via _REPO_ROOT, so just:
/root/.venvs/hexo-bottrainer-wsl/bin/python -m analysis.inference_backends.compile_variant --smoke
```

Caveat: anything that calls the native MCTS / Rust encoder (`rust_bridge.*`,
`Model1MctsSession`, real self-play) DOES need the Linux `.so` built
(`cargo build --release ... --features python`), which is NOT done in this venv.
The compile/forward harness deliberately avoids that path (synthetic or
pre-cached inputs), so it needs only torch. Full real-self-play throughput
measurement is therefore done on **native Windows** (which has the built
`_rust.cp314-win_amd64.pyd`).
```
