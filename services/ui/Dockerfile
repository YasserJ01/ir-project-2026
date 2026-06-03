# ─── Stage 1: build the React app ────────────────────────
FROM node:20-alpine AS build
WORKDIR /app

# Copy only lockfiles first for better layer caching
COPY services/ui/package.json services/ui/package-lock.json ./
RUN npm ci

# Copy the rest of the UI source
COPY services/ui/ ./
RUN npm run build

# ─── Stage 2: serve via nginx (production only) ──────────
FROM nginx:1.27-alpine
LABEL org.opencontainers.image.title="ir-project-2026-ui"
LABEL org.opencontainers.image.description="React frontend for the IR Search Engine (production build, served by nginx)"

COPY --from=build /app/dist /usr/share/nginx/html
COPY services/ui/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=3s --retries=3 --start-period=5s \
  CMD wget -q -O - http://127.0.0.1/ > /dev/null || exit 1

CMD ["nginx", "-g", "daemon off;"]
