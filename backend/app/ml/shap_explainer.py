"""
SHAP-объяснения прогнозов.

Поддерживает все четыре модели проекта:
- CatBoostRegressor → TreeExplainer
- RandomForestRegressor → TreeExplainer
- GradientBoostingRegressor → TreeExplainer
- LinearRegression / Ridge → LinearExplainer

Особенность: модели обёрнуты в Pipeline(preprocessor → estimator).
SHAP считается на трансформированных фичах, потом one-hot колонки
агрегируются обратно к исходным именам признаков.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
import shap
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline

# Размер фоновой выборки для TreeExplainer (interventional perturbation).
# Для LinearExplainer тоже используется как референс.
_BACKGROUND_SIZE = 100

_TREE_MODELS = (CatBoostRegressor, RandomForestRegressor, GradientBostingRegressor)
_LINEAR_MODELS = (LinearRegression, Ridge, Lasso)

def _split_pipeline(pipeline: Pipeline) -> tuple[Any, Any]:
    """
    Разбивает Pipeline на (препроцессор, итоговая модель).
    Препроцессор может быть None, если модель обучена на сырых фичах.
    """
    if not isinstance(pipeline, Pipeline):
        return None, pipeline

    final_estimator = pipeline.steps[-1][1]
    if len(pipeline.steps) == 1:
        return None, final_estimator

    # Pipeline без последнего шага = препроцессинг
    preprocessor = Pipeline(pipeline.steps[:-1])
    return preprocessor, final_estimator

def _get_transformed_feature_names(preprocessor: Any, raw_columns: list[str]) -> list[str]:
    """
    Получает имена колонок после препроцессинга.
    Для ColumnTransformer с OHE имена будут вида 'cat__metro_station_Авиамоторная'.
    """
    if preprocessor is None:
        return raw_columns

    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        # Фоллбек: нумеруем по порядку
        return [f"f{i}" for i in range(len(raw_columns))]

def _aggregate_onehot_shap(
    shap_values: np.ndarray,
    transformed_names: list[str],
    original_names: list[str],
) -> dict[str, float]:
    """
    Агрегирует SHAP-значения one-hot колонок обратно к исходным признакам.

    Пример: 'cat__metro_station_Авиамоторная' и 'cat__metro_station_ЗИЛ' →
    суммируются в один вклад 'metro_station'.

    Логика поска: для каждого исходного признака берём колонки, в имени
    которых он встречается как подстрока после '__' или как точное совпадение.
    """
    # shap_values shape: (n_features,) для одного объекта
    if shap_values.ndim == 2:
        shap_values = shap_values[0]

    aggregated = {name: 0.0 for name in original_names}

    for i, transformed_name in enumerate(transformed_names):
        # Убираем префикс трансформера: 'num__area' → 'area'
        clean = transformed_name.split("__", 1)[-1]

        matched_feature = None
        # Сначала ищем точное совпадение (числовые признаки)
        if clean in aggregated:
            matched_feature = clean
        else:
            # Ищем категориальный суффиксом значения: 'metro_station_ЗИЛ'
            for orig in original_names:
                if clean.startswith(f"{orig}_"):
                    matched_feature = orig
                    break

        if matched_feature is not None:
            aggregated[matched_feature] += float(shap_values[i])

    return aggregated

@lru_cache(maxsize=8)
def _build_explainer(model_id: int, model_type: str) -> shap.Explainer:
    """
    Заглушка кэша по id модели — не используется напрямую,
    т.к. shap.Explainer не сериализуем через lru_cache по объекту модели.
    Реальный кэш делаем на уровне ModelRegistry.
    """
    raise NotImplementedError("Use get_explainer instead")

def get_explainer(pipeline: Pipeline, background: pd.DataFrame) -> shap.Explainer:
    """
    Создаёт подходящий SHAP-explainer для модели.

    background: репрезентативная выборка из обучающих данных,
    нужна для interventional perturbation (TreeExplainer) и как
    референс для LinearExplainer.
    """
    preprocessor, estimator = _split_pipeline(pipeline)

    # Сэмплируем фон, если он слишком большой
    if len(background) > _BACKGROUND_SIZE:
        background = background.sample(n=_BACKGROUND_SIZE, random_state=42)

    transformed_bg = (
        preprocessor.transform(background) if preprocessor is not None else background.values
    )

    if isinstance(estimator, _TREE_MODELS):
        return shap.TreeExplainer(
            estimator,
            data=transformed_bg,
            feature_perturbation="interventional",
        )

    if isinstance(estimator, _LINEAR_MODELS):
        return shap.LinearExplainer(estimator, transformed_bg)

    # Универсальный фоллбек — медленный, но работает на всём
    return shap.KernelExplainer(estimator.predict, transformed_bg)

def explain_prediction(
    pipeline: Pipeline,
    sample: pd.DataFrame,
    background: pd.DataFrame,
    original_feature_names: list[str],
) -> dict[str, Any]:
    """
    Главная функция: считает SHAP для одного объекта.

    Возвращает:
        {
            "base_value": float,
            "prediction": float,
            "shap_values": [{"feature": str, "value": float}, ...]
        }
    """
    if len(sample) != 1:
        raise ValueError(f"Ожидается ровно одна строка, получено {len(sample)}")

    preprocessor, _ = _split_pipeline(pipeline)
    explainer = get_explainer(pipeline, background)

    transformed_sample = (
        preprocessor.transform(sample) if preprocessor is not None else sample.values
    )

    # У TreeExplainer.shap_values возвращает np.ndarray shape (n_samples, n_features)
    shap_values = explainer.shap_values(transformed_sample)
    if isinstance(shap_values, list):  # для мульти-выхода
        shap_values = shap_values[0]

    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(np.asarray(base_value).flatten()[0])
    else:
        base_value = float(base_value)

    prediction = float(pipeline.predict(sample)[0])

    transformed_names = _get_transformed_feature_names(
        preprocessor, list(sample.columns)
    )
    aggregated = _aggregate_onehot_shap(
        shap_values, transformed_names, original_feature_names
    )

    # Сортируем по абсолютной величине вклада — самые важные сверху
    items = sorted(aggregated.items(), key=lambda kv: abs(kv[1]), reverse=True)

    return {
        "base_value": base_value,
        "prediction": prediction,
        "shap_values": [{"feature": name, "value": value} for name, value in items],
    }
