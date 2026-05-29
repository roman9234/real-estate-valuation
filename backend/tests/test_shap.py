"""
Тесты SHAP-объяснений.

Проверяем триключевых свойства:
1. Аддитивность: base + sum(shap) ≈ prediction
2. Полнота: все исходные признаки присутствуют в выдаче
3. Структура ответа коректна
"""

import numpy as np
import pandas as pd
import pytest

from backend.app.ml.shap_explainer import (
    _aggregate_onehot_shap,
    _split_pipeline,
    explain_prediction,
    get_explainer,
)

ORIGINAL_FEATURES = [
    "area", "rooms", "floor", "total_floors",
    "minutes_to_metro", "is_studio",
    "metro_station", "renovation",
]

SAMPLE = pd.DataFrame([{
    "area": 60.0,
    "rooms": 2,
    "floor": 6,
    "total_floors": 15,
    "minutes_to_metro": 11,
    "is_studio": 0,
    "metro_station": "Авиамоторная",
    "renovation": "Cosmetic",
}])


@pytest.fixture(scope="module")
def models_registry():
    """Загружаем рестр один раз для всех тестов."""
    from backend.app.ml.registry import load_models
    return load_models()


@pytest.fixture(scope="module")
def background_data(models_registry):
    """Фоновая выборка из любой модели — она у всех одна и та же."""
    first = next(iter(models_registry.values()))
    return first.background


# ── Юнит-тесты helper-функций ───────────────────

class TestAggregateOnehotShap:
    """Агрегация one-hot колонок обратно к исходным признакам."""

    def test_numeric_passes_through(self):
        shap_vals = np.array([0.5, -0.3])
        transformed = ["num__area", "num__rooms"]
        original = ["area", "rooms", "metro_station"]

        result = _aggregate_onehot_shap(shap_vals, transformed, original)

        assert result["area"] == pytest.approx(0.5)
        assert result["rooms"] == pytest.approx(-0.3)
        assert result["metro_station"] == 0.0  # не было колонок

    def test_onehot_columns_are_summed(self):
        """Колонки 'metro_station_X' должны сложиться в 'metro_station'."""
        shap_vals = np.array([0.1, 0.2, -0.05])
        transformed = [
            "cat__metro_station_Авиамоторная",
            "cat__metro_station_ЗИЛ",
            "cat__metro_station_Аминьевская",
        ]
        original = ["metro_station"]

        result = _aggregate_onehot_shap(shap_vals, transformed, original)

        assert result["metro_station"] == pytest.approx(0.25)

    def test_mixed_features(self):
        shap_vals = np.array([1.0, 0.5, -0.3])
        transformed = ["num__area", "cat__renovation_Euro", "cat__renovation_Cosmetic"]
        original = ["area", "renovation"]

        result = _aggregate_onehot_shap(shap_vals, transformed, original)

        assert result["area"] == pytest.approx(1.0)
        assert result["renovation"] == pytest.approx(0.2)

    def test_2d_input_uses_first_row(self):
        shap_vals = np.array([0.5, -0.3])
        transformed = ["num__area", "num__rooms"]
        original = ["area", "rooms"]

        result = _aggregate_onehot_shap(shap_vals, transformed, original)

        assert result["area"] == pytest.approx(0.5)


class TestSplitPipeline:
    """Разделение Pipeline на препроцессор и модель."""

    def test_pipeline_with_preprocessor(self, models_registry):
        bundle = next(iter(models_registry.values()))
        pre, est = _split_pipeline(bundle.pipeline)
        assert pre is not None
        assert est is not None
        assert hasattr(est, "predict")


# ─── Интеграционные тесты ──────────────────────

class TestExplainPrediction:
    """Полная проверка для каждой модели в реестре."""

    @pytest.fixture(autouse=True)
    def _setup(self, models_registry, background_data):
        self.registry = models_registry
        self.background = background_data

    @pytest.mark.parametrize(
        "model_name",
        ["CatBoost", "RandomForest", "GradientBoosting", "LinearRegression"],
    )

    def test_explain_returns_required_fields(self, model_name):
        if model_name not in self.registry:
            pytest.skip(f"Модель {model_name} не загружена")
        bundle = self.registry[model_name]

        result = explain_prediction(
            pipeline=bundle.pipeline,
            sample=SAMPLE,
            background=self.background,
            original_feature_names=ORIGINAL_FEATURES,
        )

        assert "base_value" in result
        assert "prediction" in result
        assert "shap_values" in result
        assert isinstance(result["base_value"], float)
        assert isinstance(result["prediction"], float)
        assert isinstance(result["shap_values"], list)

    @pytest.mark.parametrize(
        "model_name",
        ["CatBoost", "RandomForest", "GradientBoosting", "LinearRegression"],
    )

    def test_additivity_property(self, model_name):
        """Главное свойство SHAP: base_value + sum(shap_values) == prediction.
        Допуск 1% от prediction — на численые шумы и агрегацию OHE.
        """
        if model_name not in self.registry:
            pytest.skip(f"Модель {model_name} не загружена")
        bundle = self.registry[model_name]

        result = explain_prediction(
            pipeline=bundle.pipeline,
            sample=SAMPLE,
            background=self.background,
            original_feature_names=ORIGINAL_FEATURES,
        )

        sum_shap = sum(item["value"] for item in result["shap_values"])
        reconstructed = result["base_value"] + sum_shap
        tolerance = abs(result["prediction"]) * 0.01

        assert reconstructed == pytest.approx(result["prediction"], abs=tolerance), (
            f"Аддитивность нарушена: base={result['base_value']:.2f} + "
            f"sum_shap={sum_shap:.2f} = {reconstructed:.2f}, "
            f"но prediction={result['prediction']:.2f}"
        )

    @pytest.mark.parametrize(
        "model_name",
        ["CatBoost", "RandomForest", "GradientBoosting", "LinearRegression"],
    )

    def test_all_features_present(self, model_name):
        """В выдаче должны быть все исходные признаки."""
        if model_name not in self.registry:
            pytest.skip(f"Модель {model_name} не загружена")
        bundle = self.registry[model_name]

        result = explain_prediction(
            pipeline=bundle.pipeline,
            sample=SAMPLE,
            background=self.background,
            original_feature_names=ORIGINAL_FEATURES,
        )

        returned_features = {item["feature"] for item in result["shap_values"]}
        assert returned_features == set(ORIGINAL_FEATURES)

    def test_shap_values_sorted_by_importance(self, models_registry):
        if "CatBoost" not in models_registry:
            pytest.skip("CatBoost не загружен")
        bundle = models_registry["CatBoost"]

        result = explain_prediction(
            pipeline=bundle.pipeline,
            sample=SAMPLE,
            background=bundle.background,
            original_feature_names=ORIGINAL_FEATURES,
        )

        abs_values = [abs(item["value"]) for item in result["shap_values"]]
        assert abs_values == sorted(abs_values, reverse=True), (
            "shap_values должны быть отсортированы по убыванию |value|"
        )

    def test_different_samples_give_different_explanations(self, models_registry):
        """Санити-чек: для разных объектов SHAP-значения должны различаться."""
        if "CatBoost" not in models_registry:
            pytest.skip("CatBoost не загружен")
        bundle = models_registry["CatBoost"]

        sample2 = SAMPLE.copy()
        sample2.loc[0, "area"] = 120.0
        sample2.loc[0, "rooms"] = 4

        r1 = explain_prediction(bundle.pipeline, SAMPLE, bundle.background, ORIGINAL_FEATURES)
        r2 = explain_prediction(bundle.pipeline, sample2, bundle.background, ORIGINAL_FEATURES)

        assert r1["prediction"] != r2["prediction"]
        # Хотя бы один SHAP должен заметно отличаться
        v1 = {x["feature"]: x["value"] for x in r1["shap_values"]}
        v2 = {x["feature"]: x["value"] for x in r2["shap_values"]}
        diffs = [abs(v1[f] - v2[f]) for f in ORIGINAL_FEATURES]
        assert max(diffs) > 1.0  # хоть какой-то сдвиг
