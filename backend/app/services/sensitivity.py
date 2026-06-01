"""
Сервис анализа чувствительности (Sensitivity / Partial Dependence).

Зафиксировать все признаки кроме одного, варьировать его
по диапазону, построить кривую цены.

Это более простая версия Partial Dependence Plot (PDP) — без
маргинализации по обучающей выборке. Для одной квартиры это
показывает «как изменится моя цена, если я повышу этаж/площадь».
"""

import numpy as np

from app.core.config import settings
from app.core.exceptions import ModelNotFoundError, FeaturesMetaNotFoundError
from app.core.features_meta import get_features_meta
from app.ml.base import BaseMLModel
from app.schemas.response import SensitivityPoint, SensitivityResponse

N_POINTS_NUMERIC = 30  # fallback, если клиент не прислал диапазон

def get_sensitivity(
    model_name: str,
    feature_name: str,
    base_features: dict,
    models: dict[str, BaseMLModel],
    range_min: float | None = None,
    range_max: float | None = None,
    step: float | None = None,
) -> SensitivityResponse:
    if model_name not in models:
        raise ModelNotFoundError(f"Модель '{model_name}' не загружена")

    model = models[model_name]
    meta = get_features_meta()

    numeric_meta = {f["name"]: f for f in meta["numeric"]}
    categorical_meta = {f["name"]: f for f in meta["categorical"]}

    if feature_name in numeric_meta:
        values = _numeric_grid(
            numeric_meta[feature_name],
            range_min=range_min,
            range_max=range_max,
            step=step,
        )
    elif feature_name in categorical_meta:
        values = categorical_meta[feature_name].get("values", [])
    else:
        raise ModelNotFoundError(
            f"Признак '{feature_name}' не найден в feature_catalog",
            details={"feature": feature_name},
        )

    points: list[SensitivityPoint] = []
    for v in values:
        modified = dict(base_features)
        modified[feature_name] = v
        try:
            price = model.predict(modified)
            points.append(SensitivityPoint(value=v, price=price))
        except Exception as e:
            print(f"[WARN] sensitivity skip {feature_name}={v}: {e}")

    return SensitivityResponse(
        model_name=model_name,
        feature_name=feature_name,
        points=points,
    )

def _numeric_grid(
    feature_meta: dict,
    range_min: float | None = None,
    range_max: float | None = None,
    step: float | None = None,
) -> list[float]:
    """
    Строит сетку значений для числового признака.

    Приоритет:
    1. Если клиент передал range_min, range_max, step — используем np.arange.
    2. Иначе — линейное разбиение от min до max из метаданных, N_POINTS_NUMERIC точек.
    """
    # Клиентский диапазон
    if range_min is not None and range_max is not None and step is not None:
        lo = float(range_min)
        hi = float(range_max)
        step_val = float(step)
        # np.arange не включает hi, поэтому + small epsilon для включения
        grid = np.arange(lo, hi + step_val * 0.5, step_val)
        grid = grid[grid <= hi + 1e-9]  # защита от float rounding
    else:
        lo = float(feature_meta.get("min", 0))
        hi = float(feature_meta.get("max", 100))
        grid = np.linspace(lo, hi, N_POINTS_NUMERIC)

    # Для целочисленных признаков (rooms, floor) округляем и оставляем уникальные
    if feature_meta.get("dtype") in {"int", "int64", "integer"}:
        grid = np.unique(np.round(grid).astype(int))

    return [float(x) for x in grid]
