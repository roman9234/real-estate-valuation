"""
Общие фикстуры тестов ML-слоя.

Если модели не загружаются — pytest падает сразу (fail-fast),
а не пропускает половину тестов. Это сознательное решение:
тесты должны проверять реальное поведение.
"""

import json
from pathlib import Path

import pytest

from app.ml.base import BaseMLModel
from app.ml.constants import ALL_MODEL_NAMES
from app.ml.registry import load_models

FIXTURES_DIR = Path(__file__).parent / "fixtures"

@pytest.fixture(scope="session")
def sample_features() -> dict:
    with open(FIXTURES_DIR / "sample_features.json", "r", encoding="utf-8") as f:
        return json.load(f)

@pytest.fixture(scope="session")
def loaded_models() -> dict[str, BaseMLModel]:
    models = load_models()
    missing = set(ALL_MODEL_NAMES) - set(models.keys())
    if missing:
        pytest.exit(
            f"Не загружены модели: {missing}. "
            f"Проверь содержимое ml_models/ и metrics.json",
            returncode=1,
        )
    return models
