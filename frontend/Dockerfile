# ==============================================================
# Dockerfile для frontend (React + Vite + Nginx)
# ==============================================================
# Двухстадийная сборка:
#   1. build — npm ci + vite build
#   2. runtime — nginx alpine со статикой

# ---- 1. Сборка ----
FROM node:22-alpine AS builder

WORKDIR /app

# Сначала копируем только package.json — чтобы Docker кешировал
# npm install и не переустанавливал зависимости при каждом
# изменении исходников.
COPY package.json package-lock.json ./
RUN npm ci

# Теперь копируем исходники и собираем
COPY . .
RUN npm run build

# ---- 2. Runtime (nginx) ----
FROM nginx:alpine AS runtime

# Копируем готовую статику
COPY --from=builder /app/dist /usr/share/nginx/html

# Копируем конфиг nginx (см. nginx.conf ниже)
COPY nginx.conf /etc/nginx/conf.d/default.conf

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD wget --quiet --tries=1 --spider http://localhost:80/ || exit 1

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
