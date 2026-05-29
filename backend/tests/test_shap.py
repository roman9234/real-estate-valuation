import numpy as np
import pandas as pd
import pytest

from app.ml.constants import ALL_MODEL_NAMES, ORIGINAL_FEATURE_NAMES
from app.ml.shap_explainer import (
    _split_pipeline,
    aggregate_to_original,
)

SAMPLE_FEATURES = {
    "area": 60.0,
    "rooms": 2,
    "floor": 6,
    "total_floors": 15,
    "minutes_to_metro": 11,
    "is_studio": 0,
    "metro_station": "Авиамоторная",
    "renovation": "Cosmetic",
}

class TestAggregateToOriginal:
    def test_numeric_passes_through(self):
        result = aggregate_to_original(
            np.array([0.5, -0.3]),
            ["num__area", "num__rooms"],
            ["area", "rooms", "metro_station"],
        )
        assert result["area"] == pytest.approx(0.5)
        assert result["rooms"] == pytest.approx(-0.3)
        assert result["metro_station"] == 0.0

    def test_onehot_columns_are_summed(self):
        result = aggregate_to_original(
            np.array([0.1, 0.2, -0.05]),
            [
                "cat__metro_station_Авиамоторная",
                "cat__metro_station_ЗИЛ",
                "cat__metro_station_Аминьевская",
            ],
            ["metro_station"],
        )
        assert result["metro_station"] == pytest.approx(0.25)

    def test_mixed_features(self):
        result = aggregate_to_original(
            np.array([1.0, 0.5, -0.3]),
            ["num__area", "cat__renovation_Euro", "cat__renovation_Cosmetic"],
            ["area", "renovation"],
        )
        assert result["area"] == pytest.approx(1.0)
        assert result["renovation"] == pytest.approx(0.2)

    def test_2d_input_uses_first_row(self):
        result = aggregate_to_original(
            np.array([[0.5, -0.3]]),
            ["num__area", "num__rooms"],
            ["area", "rooms"],
        )
        assert result["area"] == pytest.approx(0.5)

class TestSplitPipeline:
    def test_pipeline_with_preprocessor(self, loaded_models):
        # Берём любую модель с pipeline (не CatBoost — у него нет _pipeline).
        from app.ml.constants import MODEL_RANDOM_FOREST
        rf = loaded_models[MODEL_RANDOM_FOREST]
        pre, est = _split_pipeline(rf._pipeline)
        assert pre is not None
        assert hasattr(est, "predict")

class TestGetShapValues:
    """Тестируем публичный интерфейс BaseMLModel.get_shap_values."""

    @pytest.mark.parametrize("model_name", ALL_MODEL_NAMES)
    def test_returns_shap_explanation(self, loaded_models, model_name):
        model = loaded_models[model_name]
        explanation = model.get_shap_values(SAMPLE_FEATURES)

        assert isinstance(explanation.base_value, float)
        assert isinstance(explanation.prediction, float)
        assert len(explanation.shap_values) > 0

    @pytest.mark.parametrize("model_name", ALL_MODEL_NAMES)
    def test_all_original_features_present(self, loaded_models, model_name):
        model = loaded_models[model_name]
        explanation = model.get_shap_values(SAMPLE_FEATURES)
        returned = {v.feature_name for v in explanation.shap_values}
        assert returned == set(ORIGINAL_FEATURE_NAMES)

    @pytest.mark.parametrize("model_name", ALL_MODEL_NAMES)
    def test_additivity(self, loaded_models, model_name):
        """
        base_value + sum(shap) ≈ prediction.
        Для LinearModel допуск выше: SHAP пересчитывается из log-шкалы
        в рубли поправкой, что вносит небольшую погрешность.
        """
        model = loaded_models[model_name]
        explanation = model.get_shap_values(SAMPLE_FEATURES)

        sum_shap = sum(v.value for v in explanation.shap_values)
        reconstructed = explanation.base_value + sum_shap

        # 2% — учитывает linear log→rub и численные шумы tree explainer.
        tolerance = abs(explanation.prediction) * 0.02

        assert reconstructed == pytest.approx(explanation.prediction, abs=tolerance), (
            f"[{model_name}] Аддитивность нарушена: "
            f"base={explanation.base_value:,.0f} + "
            f"sum_shap={sum_shap:,.0f} = {reconstructed:,.0f}, "
            f"но prediction={explanation.prediction:,.0f}"
        )

    @pytest.mark.parametrize("model_name", ALL_MODEL_NAMES)
    def test_shap_values_sorted_by_abs(self, loaded_models, model_name):
        """SHAP-значения отсортированы по убыванию модуля вклада."""
        model = loaded_models[model_name]
        explanation = model.get_shap_values(SAMPLE_FEATURES)

        abs_values = [abs(v.value) for v in explanation.shap_values]
        assert abs_values == sorted(abs_values, reverse=True), (
            f"[{model_name}] SHAP-значения не отсортированы по |value|"
        )

    @pytest.mark.parametrize("model_name", ALL_MODEL_NAMES)
    def test_different_samples_give_different_explanations(self, loaded_models, model_name):
        """
        Санити-чек: для существенно разных квартир SHAP должен отличаться.
        Если этот тест падает — модель либо не учитывает признаки,
        либо SHAP возвращает константу (ошибка в explainer).
        """
        model = loaded_models[model_name]

        cheap = {**SAMPLE_FEATURES, "area": 35.0, "rooms": 1, "floor": 1}
        expensive = {**SAMPLE_FEATURES, "area": 120.0, "rooms": 4, "floor": 10}

        e_cheap = model.get_shap_values(cheap)
        e_expensive = model.get_shap_values(expensive)

        # Сами прогнозы должны различаться
        assert e_cheap.prediction != e_expensive.prediction

        # И хотя бы один SHAP-вклад должен заметно сместиться
        v_cheap = {v.feature_name: v.value for v in e_cheap.shap_values}
        v_expensive = {v.feature_name: v.value for v in e_expensive.shap_values}
        max_diff = max(
            abs(v_cheap[f] - v_expensive[f]) for f in ORIGINAL_FEATURE_NAMES
        )
        assert max_diff > 1.0, (
            f"[{model_name}] SHAP практически не изменился между разными квартирами"
        )

class TestGetFeatureImportance:
    """Глобальная важность признаков для каждой модели."""

    @pytest.mark.parametrize("model_name", ALL_MODEL_NAMES)
    def test_returns_all_original_features(self, loaded_models, model_name):
        model = loaded_models[model_name]
        items = model.get_feature_importance()

        returned = {fi.feature_name for fi in items}
        assert returned == set(ORIGINAL_FEATURE_NAMES), (
            f"[{model_name}] Ожидались признаки {set(ORIGINAL_FEATURE_NAMES)}, "
            f"получены {returned}"
        )

    @pytest.mark.parametrize("model_name", ALL_MODEL_NAMES)
    def test_importance_non_negative(self, loaded_models, model_name):
        model = loaded_models[model_name]
        items = model.get_feature_importance()
        for fi in items:
            assert fi.importance >= 0, (
                f"[{model_name}] Важность {fi.feature_name} < 0: {fi.importance}"
            )

    @pytest.mark.parametrize("model_name", ALL_MODEL_NAMES)
    def test_importance_sorted_desc(self, loaded_models, model_name):
        model = loaded_models[model_name]
        items = model.get_feature_importance()
        values = [fi.importance for fi in items]
        assert values == sorted(values, reverse=True), (
            f"[{model_name}] Feature importance не отсортирована по убыванию"
        )

    @pytest.mark.parametrize("model_name", ALL_MODEL_NAMES)
    def test_at_least_one_feature_significant(self, loaded_models, model_name):
        """
        Сумма важностей > 0 — защита от ситуации, когда у модели
        все важности нули (часто означает, что модель не обучена).
        """
        model = loaded_models[model_name]
        items = model.get_feature_importance()
        total = sum(fi.importance for fi in items)
        assert total > 0, f"[{model_name}] Все важности равны нулю"
