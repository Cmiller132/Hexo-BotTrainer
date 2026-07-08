# Hexo Bot Showcase — Design Plan

Public-facing site where anyone can play Hexo against hexfield checkpoints at
several strengths, review finished games in an analysis tab (policy heatmap,
value graph, top-k candidates), with every game saved to a compact database
for statistics.

This folder is PRIVATE planning (deployment specifics live here). The showcase
code itself is built in the public repo at `E:\hexo-bot\apps\showcase`.

## Locked decisions

| Decision | Choice |
|---|---|
| Deployment | Docker Compose inside a new dedicated Proxmox LXC on pve-homecolton (192.168.68.200); `/dev/dri` bind-mounted so the Arc A310 stays shared with Jellyfin |
| GPU | CPU inference at launch; PyTorch XPU (Arc) as phase 2 — hexfield's Triton kernels are CUDA-only, so Arc uses eager paths regardless |
| Frontend | New purpose-built public web app (play tab + analysis tab); the dev dashboard is NOT exposed |
| Bots | Epoch ladder × strength: a few main_7 epochs (early / mid / latest-good, refreshed periodically) × search budgets (16 / 64 / 256 / 1024 sims) |
| Ingress | Cloudflare Tunnel (`cloudflared` container); no router ports opened; domain on Cloudflare (prerequisite — see 02) |
| Identity | Anonymous by default + per-IP rate limits; optional nickname stored with the game record |
| Code home | Public repo `E:\hexo-bot` under `apps/showcase/` — Dockerfile + compose ship publicly, secrets/deploy config stay server-side in an uncommitted `.env` |
| Analysis | Post-game review tab: step through plies, policy heatmap, value/win-prob graph, top-k table; on-demand inference, cached in the DB |
| Database | SQLite, single file on a mounted volume (my call from "simple compact database" — see 03 for schema and the reuse of the `.hxr` codec for move storage) |

## Plan documents

| Doc | Contents |
|---|---|
| [01_ARCHITECTURE.md](01_ARCHITECTURE.md) | Services, the game/inference server design, API surface, frontend, bot ladder mechanics, capacity model |
| [02_DEPLOYMENT.md](02_DEPLOYMENT.md) | LXC spec, compose stack, Cloudflare Tunnel, checkpoint refresh, security hardening, XPU phase-2 path |
| [03_DATABASE.md](03_DATABASE.md) | SQLite schema, what gets stored per game, the statistics queries, analysis cache |

## Build phases

1. **Game server core** (public repo, `apps/showcase/server`): FastAPI service
   reusing `hexo_engine` + hexfield inference; game session lifecycle; bot
   ladder config; SQLite persistence. Testable locally on Windows/WSL without
   any deployment. Gate: pytest suite + a scripted full game via HTTP.
2. **Web frontend** (`apps/showcase/web`): static SPA (vanilla JS + SVG board,
   no framework build step) — play tab and analysis tab. Gate: full game +
   review in a browser against the local server.
3. **Containerization**: Dockerfile (multi-stage: maturin build → slim runtime),
   compose with the SQLite/models volumes. Gate: fresh `docker compose up`
   plays a game.
4. **Deployment**: LXC creation on the Proxmox host, Docker install, tunnel
   wiring, domain, rate rules. Gate: end-to-end game over the public URL.
5. **Stats page + polish**: public statistics endpoint/page from the DB views;
   nickname moderation pass; load test at target concurrency.
6. **Phase 2 (post-launch): XPU** — device plumbing for Arc, level-zero
   runtime in the image, benchmark vs CPU, flip a config flag if it wins.

## Capacity target (launch)

LXC: 8 cores / 8 GB RAM. Model is 8.1M params (~35 MB weights, fp32); the
whole 3-epoch ladder stays resident (~100 MB). Search at 256 sims ≈ 1–3 s/move
on one core (measured: warm CPU analyze ≈ 0.13 s/eval; a 256-visit search is a
few hundred evals batched). Cap: 8 concurrent games, queue beyond that,
per-IP cap 2 concurrent games. That is comfortable for a hobby-public site;
XPU raises the ceiling later.
