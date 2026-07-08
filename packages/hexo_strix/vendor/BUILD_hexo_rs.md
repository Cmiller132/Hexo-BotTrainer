# Building / installing `hexo_rs` (the real Gumbel MCTS)

The faithful eval path (`hexo_strix.StrixMctsPlayer`) runs hexo-strix's **actual**
Rust Gumbel-AlphaZero MCTS, exposed as the Python extension module `hexo_rs`
(built from the `hexo-mcts`/`hexo-py` crates of
https://github.com/SootyOwl/hexo-strix). It does NOT link libtorch — the network
forward is a Python callback — so it builds with just a Rust toolchain + maturin.

## Fast path: install the vendored wheel

A prebuilt wheel for this machine (CPython 3.14, win_amd64) is checked in:

```bash
pip install --no-deps packages/hexo_strix/vendor/wheels/hexo_rs-0.1.0-cp314-cp314-win_amd64.whl
```

## Rebuild from source (other platforms / Python versions)

Requires: Rust (cargo) + `maturin` (`pip install maturin`).

```bash
git clone https://github.com/SootyOwl/hexo-strix.git
cd hexo-strix
# Windows only: the self-play inference-subprocess module uses a Unix-only
# pipe-resize (as_raw_fd) that doesn't compile on Windows and is irrelevant to
# eval. Apply the cfg(unix) guard patch:
git apply ../packages/hexo_strix/vendor/hexo-rs-windows-build.patch   # adjust path
cd hexo-rs
python -m maturin build --release -o dist
pip install --no-deps dist/hexo_rs-*.whl
```

On Linux/macOS the patch is unnecessary (the code already compiles there); build
straight from a clean clone.

## Why a callback, not their model

`StrixMctsPlayer` feeds the Rust MCTS a network callback backed by this repo's
pure-PyTorch port (`hexo_strix.model` + `hexo_strix.graph`), which was verified
to produce logits/values numerically identical to hexo-strix's own
torch_geometric model + Rust graph builder (max |Δlogit| = 0, max |Δvalue| ≈
6e-8). So the search is bit-for-bit their algorithm over identical priors/values,
with no `torch_geometric` / `hexo_a0` runtime dependency.
