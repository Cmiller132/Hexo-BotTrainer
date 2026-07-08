# 02 — Deployment

## Domain

**blueshrimp.uk** (owned). Public hostname: `hexo.blueshrimp.uk`. The zone must
be on Cloudflare DNS for the tunnel (move nameservers if not already there).

## LXC (on pve-homecolton, 192.168.68.200)

- New unprivileged LXC, Debian 12 or Ubuntu 24.04 template. Suggested:
  VMID 121, hostname `hexo-showcase`, 8 cores, 8 GB RAM, 2 GB swap, 40 GB disk
  on local-lvm (image + DB + ladder checkpoints + logs fit in a fraction of
  that; local-lvm is at 76% — 40 GB keeps us honest).
- Features: `nesting=1,keyctl=1` (required for Docker inside LXC).
- GPU (inert until phase 2, cheap to wire now): bind `/dev/dri` into the LXC
  (`dev0: /dev/dri/cardX`, `dev1: /dev/dri/renderDXXX` with gid mapping to the
  container's `render` group — identify which card is the A310 via
  `ls -la /dev/dri/by-path` on the host; the AMD iGPU is the other one).
  Sharing with Jellyfin is fine — media engines and compute coexist.
- Docker via the official convenience script; compose plugin.

## Compose stack (lives in the public repo `apps/showcase/`; `.env` does not)

```yaml
services:
  app:
    build: .            # or image: ghcr.io/<user>/hexo-showcase once published
    restart: unless-stopped
    env_file: .env      # NOT committed: rate caps, worker count
    volumes:
      - showcase-db:/data          # SQLite lives here
      - ./deploy/models:/models:ro # ladder checkpoints + bots.toml
    # phase 2 adds: devices: ["/dev/dri:/dev/dri"]
  cloudflared:
    image: cloudflare/cloudflared:latest
    restart: unless-stopped
    command: tunnel run
    environment:
      - TUNNEL_TOKEN=${TUNNEL_TOKEN}   # from .env, never committed
volumes:
  showcase-db:
```

- **Dockerfile** (multi-stage): stage 1 = rust:slim + maturin builds the three
  crates into wheels; stage 2 = python:3.12-slim, installs the wheels +
  CPU-torch + fastapi/uvicorn, copies `apps/showcase` + the three pure-python
  packages, non-root user, `HEXFIELD_*` env baked to the main_7 arch,
  read-only rootfs except /data and /tmp.
- The image contains NO checkpoints (they're a mounted volume) — keeps the
  public image small and lets the ladder refresh without rebuilds.

## Cloudflare setup

1. Zero Trust → Tunnels → create `hexo-showcase`, copy the token into `.env`.
2. Public hostname `hexo.blueshrimp.uk` → `http://app:8000`.
3. Recommended edge rules: rate limiting rule on `/api/*` (e.g. 60 req/min/IP
   — the app also enforces its own), Bot Fight Mode on, caching for `/static/*`.

## Checkpoint ladder refresh (main_7 is still training)

Script in the dev repo (private), run from Windows/WSL when a new epoch gates
well: exports inference-only weights (`export_weights.py`), scps them to the
LXC's `deploy/models/ladder/`, updates `bots.toml` label, `docker compose
restart app`. The "latest" ladder entry is a stable filename
(`main7_latest.pt`) so `bots.toml` rarely changes. Past-epoch entries are
immutable. Add rows to the DB's `bots` table on change so old games keep
their true bot identity (see 03).

## Security hardening checklist

- LXC is single-purpose; nothing else listens; Proxmox firewall on the CT:
  allow outbound only (the tunnel is outbound-initiated) + SSH from LAN.
- Container: non-root, read-only fs, no capabilities, resource limits
  (cpus: 7, mem: 6g) so a runaway search can't starve the LXC.
- App: session cookies httpOnly+SameSite=Lax, CSP (self only), no debug
  routes, nickname sanitization (length cap, charset allowlist, profanity
  list at write time), request body size caps.
- Secrets: only `TUNNEL_TOKEN` — in `.env` on the LXC, chmod 600.
- Backups: nightly `sqlite3 .backup` to the mounted volume + weekly copy to
  Unraid NFS (it's already mounted on the host); vzdump the LXC weekly.
- Logging: uvicorn access log with client IP from CF headers
  (`CF-Connecting-IP`), rotated; store only a salted hash of IP in the DB.

## Phase 2 — XPU (Arc A310)

Deliberately after launch. Work items:
1. Device plumbing in the showcase server (not in hexfield core): load the
   net with `.to("xpu")` when available; hexfield's fast kernels are
   `x.is_cuda`-gated so XPU tensors take the eager paths automatically —
   correctness expected, verify with the parity tests on XPU.
2. Image variant: `intel-extension-for-pytorch`/torch-xpu wheels + level-zero
   + compute runtime (Intel's `intel/intel-extension-for-pytorch` base image
   is the shortcut), `devices: /dev/dri`.
3. The batched evaluator matters more than raw speed: route all workers'
   evals through one XPU process (the continuous scheduler already batches).
4. Benchmark honestly vs 8 CPU cores at showcase batch sizes; ship only if it
   wins. If it wins big, raise MAX_ACTIVE_GAMES.

## Ops runbook (goes in apps/showcase/README, generic form)

- Deploy/update: `git pull && docker compose up -d --build`
- Logs: `docker compose logs -f app`
- DB shell: `docker compose exec app sqlite3 /data/showcase.db`
- Ladder refresh: (private script, above)
- Kill switch: `docker compose down` (tunnel goes down, site 502s at the edge)
