# Docker — Conventions & How To

> Single source of truth for everything Docker in this project. Read this before
> editing any `Dockerfile` or `docker-compose.yml`.

## TL;DR

```powershell
# Production: build and run the whole stack
docker compose up -d --build
# → UI on http://localhost:3000
# → (backend services join in Phase 6)

# Tear it down
docker compose down
```

```powershell
# Dev: run the React app with hot-reload OUTSIDE Docker
cd services\ui
npm run dev
# → http://localhost:5173 (Vite dev server, proxies /api → :8000)
```

## The two modes

| Mode | When | What runs in Docker | What runs on host |
|------|------|---------------------|-------------------|
| **Dev**  | Day-to-day coding | _nothing_ (only optional, see below) | Python venv (`uvicorn …`) + `npm run dev` |
| **Prod** | Defense demo, "one command to run" | All services (UI in Phase 0, +backend in Phase 6) | `docker compose up` |

The default assumption: **dev is on the host, prod is in Docker.** This keeps the
edit-build-run cycle fast (no rebuilds for every code change) and matches what
most Python teams do.

If you ever want a "containerised dev" mode (everything in Docker with hot-reload),
that's a Phase 10 polish item. Not built now.

## What ships in Phase 0

| File | Purpose |
|------|---------|
| `.dockerignore` (root) | Excludes `.venv/`, `node_modules/`, `data/`, `dist/`, `__pycache__/`, secrets, OS cruft from every future `docker build` context. |
| `docker-compose.yml` (root) | One service: **`ui`** (production build via nginx on `:3000`). Backend services are commented placeholders. |
| `services/ui/Dockerfile` | Multi-stage: `node:20-alpine` (build) → `nginx:1.27-alpine` (serve). Lockfile-pinned via `npm ci`. |
| `services/ui/nginx.conf` | SPA fallback (`try_files … /index.html`) + static-asset caching + security headers. The `/api/` reverse-proxy is **commented out** until Phase 6. |

## What gets added in Phase 6

- One `Dockerfile` per service under `services/<name>/Dockerfile` (all based on
  `python:3.12-slim`).
- Backend services joined to `docker-compose.yml` under an `ir_net` bridge network.
- A `data:/data` named volume shared by `indexing`, `retrieval`, and any service
  that needs to read the persisted indexes.
- The `/api/` reverse-proxy in `services/ui/nginx.conf` is uncommented and starts
  forwarding browser traffic to the gateway service inside the Docker network.
- Each service gets a `healthcheck: … wget /health` so `docker compose ps` shows
  green/yellow for each.

## Conventions

### Image naming
```
ir-project-2026/<service>:<tag>
```
- `ir-project-2026/ui:phase0` (or `:latest` in Phase 6+).
- `<service>` matches the directory name: `ui`, `gateway`, `preprocessing`, …

### Build context
- Always the **repo root** (`context: .`).
- Service `Dockerfile` lives inside its own directory; `COPY services/ui/ ./` from
  the root context.
- This lets every service share the same `requirements.txt` (when applicable) and
  the same `.env.example` without per-service duplication.

### Layer caching
- Copy lockfiles (`package-lock.json`, `requirements.txt`) **before** the source.
- For Python services, use `pip install --no-cache-dir -r requirements.txt` to keep
  images small.
- `.dockerignore` is the single biggest lever — if you skip it, every `COPY .` will
  drag in `.venv/`, `node_modules/`, `data/`, etc.

### Security
- **No `COPY .env` ever.** Use `env_file: .env` in `docker-compose.yml` and add
  `.env` to `.dockerignore` (already done).
- Use `python:3.12-slim` (not `python:3.12`) for backend services to reduce attack
  surface.
- Run containers as non-root (`USER appuser`) where it doesn't break things —
  Phase 6 item.

## Common commands

```powershell
# Build everything
docker compose build

# Build a single service (faster)
docker compose build ui

# Build from scratch (no cache)
docker compose build --no-cache ui

# Start in background
docker compose up -d

# Follow logs
docker compose logs -f ui

# List running containers
docker compose ps

# Restart a single service
docker compose restart ui

# Open a shell inside the UI container
docker compose exec ui sh

# Stop & remove containers, networks, but KEEP images
docker compose down

# Stop & remove containers, networks, AND images
docker compose down --rmi all
```

## Windows + WSL2 tips

- Docker Desktop 28.x is using the **WSL2** backend (Linux containers, fast).
- The `F:\` drive is on the Windows side; Docker Desktop's WSL2 VM sees it as
  `/mnt/f/`. Build context reads are fast for small contexts; for large ones,
  consider keeping the repo inside the WSL2 filesystem (e.g. `~/ir-project-2026`)
  and access it from VS Code via the WSL extension.
- We **do not** mount source into the UI container (artefacts are baked in),
  so file-watching issues (the famous Vite-in-Docker HMR bug) don't apply.
- If `docker compose up` fails with `bind: address already in use`, change the
  host port in `docker-compose.yml` (e.g. `"3001:80"` instead of `"3000:80"`).

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `bind: address already in use … 0.0.0.0:3000` | Port 3000 occupied | Change `"3000:80"` → `"3001:80"` in `docker-compose.yml` |
| `ERROR: failed to solve … exceeded timeout` | Slow Docker Hub pull | Retry; or `docker pull node:20-alpine` separately first |
| Image > 1 GB | `.dockerignore` missing/broken | Verify with `docker compose config` and `du -sh` on context |
| Container exits immediately | Bad CMD or missing static files | `docker compose logs ui` for the error |
| `localhost:3000` shows `502` | `/api/` proxy enabled but gateway not up | Make sure the block is commented out (Phase 0) |

## Future work (Phase 6+)

- [ ] Backend service `Dockerfile`s (one per service in `services/<name>/`).
- [ ] `ir_net` bridge network + `data:/data` named volume in `docker-compose.yml`.
- [ ] Uncomment `/api/` reverse-proxy in `services/ui/nginx.conf`.
- [ ] `depends_on: { condition: service_healthy }` between backend services.
- [ ] Non-root `USER` directive in every service.
- [ ] Multi-platform builds (`--platform linux/amd64,linux/arm64`) for portability.
- [ ] Optional: GitHub Actions workflow that builds and pushes images on every tag.
