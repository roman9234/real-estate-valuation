"""
SHAP-объяснения прогнозов.

Поддерживает все четыре модели проекта:
- CatBoostRegressor → TreeExplainer
- RandomForestRegressor → TreeExplainer
- GradientBoostingRegressor → TreeExplainer
- LinearRegression / Ridge / Lasso → LinearExplainer

Особенность: модели обёрнуты в Pipeline(preprocessor → estimator).
SHAP считается на трансформированных фичах, потом one-hot колонки
агрегируются обратно к исходным именам признаков.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
import shap
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline

# Размер фоновой выборки для TreeExplainer.
_BACKGROUND_SIZE = 100

_TREE_MODELS = (CatBoostRegressor, RandomForestRegressor, GradientBoostingRegressor)
_LINEAR_MODELS = (LinearRegression, Ridge, Lasso)

def _split_pipeline(pipeline: Pipeline) -> tuple[Optional[Pipeline], Any]:
    """
    Разбивает Pipeline на (препроцессор, итоговая модель).
    """
    if not isinstance(pipeline, Pipeline):
        return None, pipeline

    final_estimator = pipeline.steps[-1][1]
    if len(pipeline.steps) == 1:
        return None, final_estimator

    preprocessor = Pipeline(pipeline.steps[:-1])
    return preprocessor, final_estimator

def _get_transformed_feature_names(
    preprocessor: Any, raw_columns: list[str]
) -> list[str]:
    """
    Получает имена колонок после препроцессинга.
    Для ColumnTransformer с OHE имена будут вида 'cat__metro_station_Авиамоторная'.
    """
    if preprocessor is None:
        return raw_columns

    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return [f"f{i}" for i in range(len(raw_columns))]

def _aggregate_onehot_shap(
    shap_values: np.ndarray,
    transformed_names: list[str],
    original_names: list[str],
) -> dict[str, float]:
    """
    Агрегирует SHAP-значения one-hot колонок обратно к исходным признакам.
    """
    if shap_values.ndim == 2:
        shap_values = shap_values[0]

    aggregated = {name: 0.0 for name in original_names}

    for i, transformed_name in enumerate(transformed_names):
        clean = transformed_name.split("__", 1)[-1]

        matched_feature = None
        if clean in aggregated:
            matched_feature = clean
        else:
            for orig in original_names:
                if clean.startswith(f"{orig}_"):
                    matched_feature = orig
                    break

        if matched_feature is not None:
            aggregated[matched_feature] += float(shap_values[i])

    return aggregated

def get_explainer(pipeline: Pipeline, background: pd.DataFrame) -> shap.Explainer:
    """
    Создаёт подходящий SHAP-explainer для модели.
    """
    preprocessor, estimator = _split_pipeline(pipeline)

    if len(background) > _BACKGROUND_SIZE:
        background = background.sample(n=_BACKGROUND_SIZE, random_state=42)

    if preprocessor is not None:
        transformed_bg = preprocessor.transform(background)
    else:
        transformed_bg = background.values

    if isinstance(estimator, _TREE_MODELS):
        return shap.TreeExplainer(
            estimator,
            data=transformed_bg,
            feature_perturbation="interventional",
        )

    if isinstance(estimator, _LINEAR_MODELS):
        return shap.LinearExplainer(estimator, transformed_bg)

    # Универсальный фоллбек
    return shap.KernelExplainer(estimator.predict, transformed_bg)

def explain_prediction(
    pipeline: Pipeline,
    sample: pd.DataFrame,
    background: pd.DataFrame,
    original_feature_names: list[str],
) -> dict[str, Any]:
    """
    Главная функция: считает SHAP для одного объекта.
    """
    if len(sample) != 1:
        raise ValueError(f"Ожидается ровно одна строка, получено {len(sample)}")

    preprocessor, _ = _split_pipeline(pipeline)
    explainer = get_explainer(pipeline, background)

    if preprocessor is not None:
        transformed_sample = preprocessor.transform(sample)
    else:
        transformed_sample = sample.values

    shap_values = explainer.shap_values(transformed_sample)
    if isinstance(shap_values, list):
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

    items = sorted(aggregated.items(), key=lambda kv: abs(kv[1]), reverse=True)

    return {
        "base_value": base_value,
        "prediction": prediction,
        "shap_values": [{"feature": name, "value": value} for name, value in items],
    }
