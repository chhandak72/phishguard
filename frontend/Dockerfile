# ============================================================
# PhishGuard Frontend – Dockerfile
# ============================================================
# Multi-stage build:
#   Stage 1 (builder) – Node.js to build the Vite static assets
#   Stage 2 (runtime) – nginx to serve the static build
#
# Build:
#   docker build -t phishguard-frontend ./frontend
#
# Run:
#   docker run -p 3000:80 phishguard-frontend
#
# Environment variables (set at BUILD time for Vite):
#   VITE_API_URL=http://localhost:8000
# ============================================================

# ---------- Stage 1: Node builder ----------
FROM node:20-alpine AS builder

WORKDIR /app

# Copy dependency files first for better layer caching
COPY package.json package-lock.json* ./

RUN npm ci --prefer-offline

# Copy source
COPY . .

# Build – VITE_API_URL can be overridden with --build-arg
ARG VITE_API_URL=http://localhost:8000
ENV VITE_API_URL=$VITE_API_URL

RUN npm run build

# ---------- Stage 2: nginx runtime ----------
FROM nginx:1.27-alpine AS runtime

LABEL maintainer="Aayush Paudel, Chhandak Mukherjee, Arnav Makhija"
LABEL version="1.0.0"
LABEL description="PhishGuard – phishing email classifier frontend"

# Copy the production build
COPY --from=builder /app/dist /usr/share/nginx/html

# Custom nginx config:
#  - Single-page app: all non-file requests → index.html (React router)
#  - Gzip compression
#  - Cache static assets aggressively
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

# Graceful shutdown
STOPSIGNAL SIGQUIT

CMD ["nginx", "-g", "daemon off;"]
