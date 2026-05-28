"""
Общие фикстуры для всех тестов ML-слоя.

conftest.py — стандартное место pytest для определения фикстур,
которые автоматически подгружаются во все тесты в этом каталоге
и подкаталогах. Не нужно импортировать вручную.
"""

import json
from pathlib import Path

import pytest

from backend.app.core.config import settings
from backend.app.ml.registry import load_models
from backend.app.ml.base import BaseMLModel

# Путь к фикстурам
FIXTURES_DIR = Path(__file__).parent / "fixtures"

@pytest.fixture(scope="session")
def sample_features() -> dict:
    """
    Эталонный набор признаков (медианая квартира из датасета).

    scope="session" — фикстура читается один раз за весь прогон тестов.
    Это безопасно, потому что тесты не должны мутировать словарь
    (FeatureEnricher.enrich возвращает копию).

    Если бы хотелось полной защиты от случайной мутации,
    стоило бы возвращать types.MappingProxyType (immutable view).
    """
    with open(FIXTURES_DIR / "sample_features.json", "r", encoding="utf-8") as f:
        return json.load(f)

@pytest.fixture(scope="session")
def loaded_models() -> dict[str, BaseMLModel]:
    """
    Загружает все доступные модели один раз за сессию.

    scope="session" критичен: загрузка CatBoost/XGBoost — это
    сотни миллисекунд, перезагружать на каждый тест неэффективно.
    Тесты только читают модели (predict), не мутируют их состояние.
    """
    models = load_models()
    return models

@pytest.fixture
def models_available(loaded_models: dict[str, BaseMLModel]) -> bool:
    """
    Флаг наличия моделей. Используется через skipif для тестов,
    которые требуют файлы артефактов в ml_models/.
    """
    return len(loaded_models) > 0


# Список имён моделей для параметризации.
ALL_MODEL_NAMES = ["CatBoost", "XGBoost", "RandomForest", "LinearRegression"]
