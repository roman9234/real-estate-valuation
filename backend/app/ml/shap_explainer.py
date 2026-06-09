"""
SHAP-объяснения прогнозов: внутренние утилиты ML-слоя.

Эти функции вызываются из конкретных классов моделей внутри
их _compute_shap. Сервисный слой и роуты этот модуль НЕ импортируют.

Поддерживаемые сценарии:
- sklearn.Pipeline(preprocessor → RandomForestRegressor / XGBRegressor)
- sklearn.Pipeline(preprocessor → Ridge / Lasso / LinearRegression)
- CatBoostRegressor без Pipeline (категории нативные)
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
import shap
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from app.ml.constants import SHAP_BACKGROUND_SIZE

_TREE_MODELS = (RandomForestRegressor, XGBRegressor)
_LINEAR_MODELS = (LinearRegression, Ridge, Lasso)

def _split_pipeline(pipeline: Pipeline) -> tuple[Optional[Pipeline], Any]:
    """Разделяет Pipeline на (препроцессор, итоговый estimator)."""
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
    """Имена колонок после препроцессинга (с префиксами OHE)."""
    if preprocessor is None:
        return raw_columns
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return [f"f{i}" for i in range(len(raw_columns))]

def aggregate_to_original(
    values: np.ndarray,
    transformed_names: list[str],
    original_names: list[str],
) -> dict[str, float]:
    """
    Сворачивает значения по трансформированным колонкам
    обратно к исходным признакам.

    Используется и для SHAP (со знаком), и для Feature Importance
    (значения неотрицательны, сложение по группам).
    """
    if values.ndim == 2:
        values = values[0]

    aggregated = {name: 0.0 for name in original_names}

    for i, transformed_name in enumerate(transformed_names):
        clean = transformed_name.split("__", 1)[-1]

        if clean in aggregated:
            aggregated[clean] += float(values[i])
            continue

        for orig in original_names:
            if clean.startswith(f"{orig}_"):
                aggregated[orig] += float(values[i])
                break

    return aggregated

def _build_explainer(
    estimator: Any,
    transformed_background: np.ndarray,
) -> shap.Explainer:
    """Подбирает explainer под тип estimator."""
    if isinstance(estimator, _TREE_MODELS):
        return shap.TreeExplainer(
            estimator,
            data=transformed_background,
            feature_perturbation="interventional",
        )
    if isinstance(estimator, _LINEAR_MODELS):
        return shap.LinearExplainer(estimator, transformed_background)
    return shap.KernelExplainer(estimator.predict, transformed_background)

def _sample_background(background: pd.DataFrame) -> pd.DataFrame:
    if len(background) > SHAP_BACKGROUND_SIZE:
        return background.sample(n=SHAP_BACKGROUND_SIZE, random_state=42)
    return background

def explain_pipeline(
    pipeline: Pipeline,
    sample: pd.DataFrame,
    background: pd.DataFrame,
    original_feature_names: list[str],
) -> tuple[float, dict[str, float]]:
    """
    SHAP для модели, обёрнутой в sklearn.Pipeline.
    Возвращает (base_value, {feature_name: contribution}).
    """
    if len(sample) != 1:
        raise ValueError(f"Ожидается одна строка, получено {len(sample)}")

    preprocessor, estimator = _split_pipeline(pipeline)
    bg = _sample_background(background)

    if preprocessor is not None:
        transformed_bg = preprocessor.transform(bg)
        transformed_sample = preprocessor.transform(sample)
    else:
        transformed_bg = bg.values
        transformed_sample = sample.values

    explainer = _build_explainer(estimator, transformed_bg)
    shap_values = explainer.shap_values(transformed_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(np.asarray(base_value).flatten()[0])
    else:
        base_value = float(base_value)

    transformed_names = _get_transformed_feature_names(
        preprocessor, list(sample.columns)
    )
    contributions = aggregate_to_original(
        np.asarray(shap_values), transformed_names, list(original_feature_names)
    )
    return base_value, contributions

def explain_catboost(
    model: CatBoostRegressor,
    sample: pd.DataFrame,
    original_feature_names: list[str],
) -> tuple[float, dict[str, float]]:
    """
    SHAP для CatBoost без Pipeline.

    Используется встроенный get_feature_importance(type='ShapValues') —
    он быстрее, чем shap.TreeExplainer, и нативно учитывает
    категориальные признаки без OHE-разворачивания.
    Возвращает массив shape=(n_samples, n_features + 1):
    последняя колонка — bias (base_value).
    """
    from catboost import Pool

    feature_names = list(model.feature_names_)
    cat_features = model.get_cat_feature_indices()
    pool = Pool(
        data=sample[feature_names],
        cat_features=cat_features,
    )
    shap_matrix = model.get_feature_importance(data=pool, type="ShapValues")

    row = shap_matrix[0]
    base_value = float(row[-1])
    shap_values = row[:-1]
    contributions = aggregate_to_original(
        np.asarray(shap_values), feature_names, list(original_feature_names)
    )
    return base_value, contributions
