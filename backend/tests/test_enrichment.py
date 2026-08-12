"""
Юнит-тесты для FeatureEnricher.

Не требуют файлов моделей, не требуют сети. Чистая математика.
"""

import math

import numpy as np
import pytest

from app.ml.enrichment import FeatureEnricher

class TestEnrichBasic:
    """
    Базовые проверки преобразования датасета
    """

    def test_enrich_ads_derived_features(self, sample_features):
        """Все 4 производных поля должны появиться в результате."""
        result = FeatureEnricher.enrich(sample_features)

        assert "floor_ratio" in result
        assert "is_first_floor" in result
        assert "is_last_floor" in result
        assert "log_area" in result

    def test_enrich_preserves_original_fields(self, sample_features):
        """Исходные поля должны сохраниться без изменений."""
        result = FeatureEnricher.enrich(sample_features)

        for key, value in sample_features.items():
            assert result[key] == value, f"Поле {key} изменено"

    def test_enrich_does_not_mutate_input(self, sample_features):
        """
        Критично: исходный словарь не должен мутироваться.
        Иначе повторные вызовы с одним и тем же dict дадут разный результат.
        """
        original_keys = set(sample_features.keys())
        FeatureEnricher.enrich(sample_features)
        assert set(sample_features.keys()) == original_keys

class TestFloorRatio:
    """Проверка floor_ratio = floor / total_floors."""

    def test_middle_floor(self):
        features = _make_features(floor=6, total_floors=15)
        result = FeatureEnricher.enrich(features)
        assert result["floor_ratio"] == pytest.approx(6 / 15)

    def test_first_floor(self):
        features = _make_features(floor=1, total_floors=10)
        result = FeatureEnricher.enrich(features)
        assert result["floor_ratio"] == pytest.approx(0.1)

    def test_top_floor(self):
        features = _make_features(floor=20, total_floors=20)
        result = FeatureEnricher.enrich(features)
        assert result["floor_ratio"] == pytest.approx(1.0)

    def test_zero_total_floors_no_exception(self):
        """
        Защита от деления на ноль. Хотя в реальных данных
        total_floors >= 1 (валидируется Pydantic), enricher
        должен быть безопасен даже при ошибочном воде.
        """
        features = _make_features(floor=1, total_floors=0)
        result = FeatureEnricher.enrich(features)
        assert result["floor_ratio"] == 0.0

class TestFirstLastFloor:
    """Проверка флагов is_first_floor / is_last_floor."""

    def test_first_floor_flag(self):
        features = _make_features(floor=1, total_floors=10)
        result = FeatureEnricher.enrich(features)
        assert result["is_first_floor"] == 1
        assert result["is_last_floor"] == 0

    def test_last_floor_flag(self):
        features = _make_features(floor=10, total_floors=10)
        result = FeatureEnricher.enrich(features)
        assert result["is_first_floor"] == 0
        assert result["is_last_floor"] == 1

    def test_middle_floor_flags(self):
        features = _make_features(floor=5, total_floors=10)
        result = FeatureEnricher.enrich(features)
        assert result["is_first_floor"] == 0
        assert result["is_last_floor"] == 0

    def test_one_floor_building(self):
        """
        Граничный случай: одноэтажный дом.
        floor=1, total_floors=1 → оба флага = 1.
        Это семантически страно, но математически корректно.
        Модель сама решит, как с этим работать.
        """
        features = _make_features(floor=1, total_floors=1)
        result = FeatureEnricher.enrich(features)
        assert result["is_first_floor"] == 1
        assert result["is_last_floor"] == 1

class TestLogArea:
    """Проверка log_area = log1p(area)."""

    def test_log_area_value(self):
        features = _make_features(area=60.0)
        result = FeatureEnricher.enrich(features)
        assert result["log_area"] == pytest.approx(np.log1p(60.0))

    def test_log_area_small(self):
        features = _make_features(area=1.0)
        result = FeatureEnricher.enrich(features)
        assert result["log_area"] == pytest.approx(np.log(2.0))  # log1p(1) = log(2)

    def test_log_area_zero(self):
        """log1p(0) = 0 — не падает."""
        features = _make_features(area=0.0)
        result = FeatureEnricher.enrich(features)
        assert result["log_area"] == 0.0

class TestTypes:
    """Типы производных признаков должны сответствовать обучению."""

    def test_floor_ratio_is_float(self, sample_features):
        result = FeatureEnricher.enrich(sample_features)
        assert isinstance(result["floor_ratio"], float)

    def test_is_first_floor_is_int(self, sample_features):
        """
        is_first_floor — int, не bool.
        Это важно: pandas.DataFrame различает int и bool колонки,
        а CatBoost при предсказании ожидает тот же тип, что был
        при обучении (int64).
        """
        result = FeatureEnricher.enrich(sample_features)
        assert isinstance(result["is_first_floor"], int)
        assert not isinstance(result["is_first_floor"], bool)
        # bool — подклас int в Python, поэтому проверяем явно.

    def test_log_area_is_float(self, sample_features):
        result = FeatureEnricher.enrich(sample_features)
        assert isinstance(result["log_area"], float)

# ─── helpers ────────────────────────────────

def _make_features(**overrides) -> dict:
    """
    Создаёт словарь признаков с дефолтными значениями,
    переопределяя только нужные поля.
    """
    base = {
        "area": 60.0,
        "rooms": 2,
        "floor": 6,
        "total_floors": 15,
        "minutes_to_metro": 11,
        "is_studio": 0,
        "metro_station": "Авиамоторная",
        "renovation": "Cosmetic",
    }
    base.update(overrides)
    return base
