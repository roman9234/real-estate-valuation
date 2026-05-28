# Предсказание цен на квартиры

Быстрый старт

```bash
# Клонировать репозиторий и положить файлы моделей в backend/ml_models/

# Запустить всё одной командой
docker compose up --build

# Открыть браузер
open http://localhost:80
Разработка без Docker
```

Разработка без Docker
```bash
# Backend
cd backend
pip install -r requirements.txt
python -m app.main
# http://localhost:8000/docs

# Frontend
cd frontend
npm install
npm run dev
# http://localhost:5173

```

Тесты
```bash
cd backend
pytest                          # все тесты
pytest tests/test_enrichment.py # только enrichment (без моделей)
pytest -n auto                  # параллельно (нужен pytest-xdist)
pytest --cov=app.ml             # с покрытием (нужен pytest-cov)

```