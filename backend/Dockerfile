# ==============================================================
# Dockerfile для backend (FastAPI + Uvicorn)
# ==============================================================
#
# Двухстадийная сборка:
#   1. build — устанавливаем зависимости в виртуальное окружение
#   2. runtime — копируем только venv + код, без слоя сборки
# 🚲 Многостадийная сборка — стандарт для Python-сервисов.
# Если не нужна минимальная дистрибуция (single-stage проще),
# можно объединить этапы. Но двухстадийная экономит ~100 MB
# финального образа за счёт отсутствия build-зависимостей.
# ==============================================================

# ---- 1. Сборка ----
FROM python:3.12-slim AS builder

WORKDIR /app

# Устанавливаем системные зависимости для numpy/scikit-learn.
# slim-образ минимален — без gcc и заголовков Python не собрать
# wheel'ы для numpy.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ---- 2. Runtime ----
FROM python:3.12-slim AS runtime

WORKDIR /app

# Копируем установленные пакеты из builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Копируем код приложения
COPY app/ app/
COPY ml_models/ ml_models/

# Healthcheck: uvicorn слушает 8000, /health возвращает 200
# 🚲 curl нет в slim-образе по умолчанию. Устанавливаем
# минимальный curl или используем wget.
# Альтернатива — HEALTHCHECK с python -c.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl --fail http://localhost:8000/api/v1/health || exit 1

EXPOSE 8000

# Запуск: uvicorn с одним воркером.
# workers=1 для простоты: модель загружается в память один раз.
# Для production — Gunicorn с UvicornWorker (много воркеров - много
# копий модели в RAM) или Uvicorn с --workers=N.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
