# Предсказание цен на квартиры




Вариант 1. Локальный запуск без Docker
Установка зависимостей
```bash

cd backend
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows PowerShell
pip install --upgrade pip
pip install -r requirements.txt
```
Запуск сервера
```bash
# из директории backend/
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
После старта:
API: http://localhost:8000
Swagger UI: http://localhost:8000/docs
Healthcheck: http://localhost:8000/api/v1/health
Запуск тестов
```bash
# из директории backend/
pytest
```
или с подробным выводом:
```bash
pytest -v
```
Вариант 2. Запуск в Docker
Только backend
```bash
# из директории backend/
docker build -t diploma-backend .
docker run --rm -p 8000:8000 --name diploma-backend diploma-backend
```
Проверка:
```bash
curl http://localhost:8000/api/v1/health
```


Остановить: Ctrl+C (флаг --rm уберёт контейнер автоматически).
Backend + frontend через docker compose
```bash
# из корня репозитория (где лежит docker-compose.yml)
docker compose up --build
```

После сборки:
Frontend: http://localhost
Backend API: http://localhost:8000
Swagger: http://localhost:8000/docs
Запуск в фоне:
```bash
docker compose up --build -d
```


Логи в фоновом режиме:
```bash
docker compose logs -f backend
docker compose logs -f frontend
```

Остановка:
```bash
docker compose down
```

Пересборка с нуля:
```bash
docker compose build --no-cache
docker compose up
```



