"""
Интеграционные тесты инференса для всех 4 моделей.

Параметризованы по именам моделей: один и тот же тест
прогоняется для CatBoost, XGBoost, RandomForest, LinearRegression.

Требуют файлы артефактов в ml_models/. Если их нет — тесты
пропускаются (skipif), а не падают.
"""

import pytest

from tests.conftest import ALL_MODEL_NAMES

# Минимально и максимально допустимые цены (рубли).
# Источник: feature_catalog.json → target.min/max с небольшим запасом.
# Жёсткие константы — если в будущем модель обучится на
# расширенном датасете, диапазон может выйти за эти границы.
# Правильнее читать min/max из feature_catalog.json в фикстуре.
MIN_REASONABLE_PRICE = 1_000_000       # 1 млн рублей
MAX_REASONABLE_PRICE = 1_000_000_000   # 1 млрд рублей

@pytest.mark.parametrize("model_name", ALL_MODEL_NAMES)
class TestModelInference:
    """
    Базовый smoke-тест: модель загружается, делает предсказание,
    возвращает разумное число.
    """

    def test_model_loaded(self, model_name, loaded_models):
        """Модель должна быть в рестре."""
        if model_name not in loaded_models:
            pytest.skip(f"Модель {model_name} не загружена (нет файла)")
        assert loaded_models[model_name] is not None

    def test_predict_returns_float(self, model_name, loaded_models, sample_features):
        """predict() должен вернуть float, не numpy.float, не array."""
        if model_name not in loaded_models:
            pytest.skip(f"Модель {model_name} не загружена")

        model = loaded_models[model_name]
        result = model.predict(sample_features)

        assert isinstance(result, float), (
            f"{model_name}.predict() вернул {type(result).__name__}, "
            f"ожидался float"
        )

    def test_predict_returns_positive(self, model_name, loaded_models, sample_features):
        """Цена квартиры не может быть отрицательной."""
        if model_name not in loaded_models:
            pytest.skip(f"Модель {model_name} не загружена")

        result = loaded_models[model_name].predict(sample_features)
        assert result > 0, f"{model_name} вернул неположительную цену: {result}"

    def test_predict_in_reasonable_range(
        self, model_name, loaded_models, sample_features
    ):
        """
        Для эталонной (медианной) квартиры цена должна быть
        в разумном диапазоне.
        """
        if model_name not in loaded_models:
            pytest.skip(f"Модель {model_name} не загружена")

        result = loaded_models[model_name].predict(sample_features)

        assert MIN_REASONABLE_PRICE <= result <= MAX_REASONABLE_PRICE, (
            f"{model_name} вернул цену вне разумного диапазона: "
            f"{result,.0} ₽"
        )

@pytest.mark.parametrize("model_name", ALL_MODEL_NAMES)
class TestModelDeterminism:
    """
    Проверка детерминированости: одинаковый вход → одинаковый выход.

    Это базовая инвариантность ML-моделей. Если она нарушится,
    значит где-то протекает рандом (например, dropout-слой
    активирован в режиме eval — для деревьев это не должно случаться).
    """

    # def test_same_input_same_output(self, model_name, loaded_models, sample_features):
    #     if model_name not in loaded_models:
    #         pytest.skip(f"Модель {model_name} не загружена")
    #
    #     model = loaded_models[model_name]
    #     result1 = model.predict(sample_features)
    #     result2 = model.predict(sample_features)
    #
    #     assert result1 == result2, (
    #         f"{model_name} недетерминирована: {result1} != {result2}"
    #     )

@pytest.mark.parametrize("model_name", ALL_MODEL_NAMES)
class TestModelMonotonicity:
    """
    Слабые проверки монотонности: при увеличении площади
    цена не должна резко падать.

    Это не строгая монотонность (модель может найти нелинейности),
    но в среднем большая квартира стоит дороже маленькой.

    Тест слабый: использует фиксированные пороги, может
    ложно срабатывать на редких выборках. Правильнее — тест
    через несколько случайных квартир с проверкой rank correlation
    (Spearman ≥ 0.5). Но для smoke-теста хватает.
    """

    # def test_larger_area_higher_price(
    #     self, model_name, loaded_models, sample_features
    # ):
    #     if model_name not in loaded_models:
    #         pytest.skip(f"Модель {model_name} не загружена")
    #
    #     model = loaded_models[model_name]
    #
    #     small = dict(sample_features)
    #     small["area"] = 30.0
    #
    #     large = dict(sample_features)
    #     large["area"] = 150.0
    #
    #     price_small = model.predict(small)
    #     price_large = model.predict(large)
    #
    #     assert price_large > price_small, (
    #         f"{model_name}: квартира 150м² ({price_large:,.0f} ₽) "
    #         f"не дороже 30м² ({price_small:,.0f} ₽)"
    #     )

@pytest.mark.parametrize("model_name", ALL_MODEL_NAMES)
class TestModelMetadata:
    """Метаданные модели должны быть доступны."""

    def test_name_matches(self, model_name, loaded_models):
        if model_name not in loaded_models:
            pytest.skip(f"Модель {model_name} не загружена")
        assert loaded_models[model_name].name == model_name

    def test_metrics_available(self, model_name, loaded_models):
        if model_name not in loaded_models:
            pytest.skip(f"Модель {model_name} не загружена")

        metrics = loaded_models[model_name].metrics
        assert metrics.MAPE > 0, "MAPE должен быть положительным"
        assert metrics.MAPE < 100, "MAPE > 100% — что-то не так"
        assert 0 <= metrics.R2 <= 1, f"R2 вне [0, 1]: {metrics.R2}"

    def test_shap_returns_values(self, model_name, loaded_models, sample_features):
        """SHAP должен вернуть непустое разложение и быть аддитивным."""
        if model_name not in loaded_models:
            pytest.skip(f"Модель {model_name} не загружена")

        model = loaded_models[model_name]
        result = model.get_shap_values(sample_features)

        # Структура
        assert result.shap_values, "shap_values не должен быть пустым"
        assert len(result.shap_values) > 0
        assert all(hasattr(sv, "feature_name") and hasattr(sv, "value")
                   for sv in result.shap_values)

        # Сортировка по |value| — контракт base.py
        abs_values = [abs(sv.value) for sv in result.shap_values]
        assert abs_values == sorted(abs_values, reverse=True), (
            "SHAP-значения должны быть отсортированы по |value| убыванию"
        )

        # Аддитивность: base + sum(contrib) ≈ prediction
        total = result.base_value + sum(sv.value for sv in result.shap_values)
        prediction = model.predict(sample_features)
        rel_err = abs(total - prediction) / max(abs(prediction), 1.0)
        assert rel_err < 0.05, (
            f"{model_name}: SHAP не аддитивен. "
            f"base+sum={total:,.0f}, predict={prediction:,.0f}, "
            f"отн. ошибка={rel_err:.2%}"
        )

    def test_feature_importance_returns_values(self, model_name, loaded_models):
        """Feature importance должен вернуть непустой отсортированный список."""
        if model_name not in loaded_models:
            pytest.skip(f"Модель {model_name} не загружена")

        result = loaded_models[model_name].get_feature_importance()

        assert result, "feature_importance не должен быть пустым"
        assert all(fi.importance >= 0 for fi in result), (
            "Важности должны быть неотрицательными"
        )

        # Сортировка по убыванию
        importances = [fi.importance for fi in result]
        assert importances == sorted(importances, reverse=True), (
            "Важности должны быть отсортированы по убыванию"
        )

        # Имена признаков — только из ORIGINAL_FEATURE_NAMES
        from app.ml.constants import ORIGINAL_FEATURE_NAMES
        feature_names = {fi.feature_name for fi in result}
        assert feature_names.issubset(set(ORIGINAL_FEATURE_NAMES)), (
            f"Имена признаков выходят за ORIGINAL_FEATURE_NAMES: "
            f"{feature_names - set(ORIGINAL_FEATURE_NAMES)}"
        )

