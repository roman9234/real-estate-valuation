# Все тесты
pytest

Подробный вывод
pytest -v

# Только конкретный файл
pytest tests/test_enrichment.py -v

pytest tests/test_inference.py -v
pytest tests/test_edge_cases.py -v

# Параллельно (если установлен pytest-xdist)
pytest -n auto

# С покрытием (если установлен pytest-cov)
pytest --cov=app.ml --cov-report=term-missing
