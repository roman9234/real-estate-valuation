"""
Сервис анализа чувствительности (Sensitivity / Partial Dependence).

Зафиксировать все признаки кроме одного, варьировать его
по диапазону, построить кривую цены.

Это более простая версия Partial Dependence Plot (PDP) — без
маргинализации по обучающей выборке. Для одной квартиры это
показывает «как изменится моя цена, если я повышу этаж/площадь».
"""

import numpy as np

from backend.app.core.config import settings
from backend.app.core.exceptions import ModelNotFoundError, FeaturesMetaNotFoundError
from backend.app.core.features_meta import get_features_meta
from backend.app.ml.base import BaseMLModel
from backend.app.schemas.response import SensitivityPoint, SensitivityResponse

# Сколько точек строить для числовых признаков.
# Хардкод: 30 точек — компромисс между гладкостью кривой
# и временем расчёта (30 предсказаний на запрос).
N_POINTS_NUMERIC = 30

def get_sensitivity(
    model_name: str,
    feature_name: str,
    base_features: dict,
    models: dict[str, BaseMLModel],
) -> SensitivityResponse:
    """
    Строит кривую чувствительности по одному признаку.

    Args:
        model_name: имя модели из реестра
        feature_name: имя признака для варьирования
        base_features: фиксированные значения остальных признаков
        models: реестр моделей

    Returns:
        SensitivityResponse с массивом точек (значение → цена).
    """
    if model_name not in models:
        raise ModelNotFoundError(f"Модель '{model_name}' не загружена")

    model = models[model_name]
    meta = get_features_meta()

    # Ищем признак в metadata (числовой или категориальный)
    numeric_meta = {f["name"]: f for f in meta["numeric"]}
    categorical_meta = {f["name"]: f for f in meta["categorical"]}

    if feature_name in numeric_meta:
        values = _numeric_grid(numeric_meta[feature_name])
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
            # Пропускаем точки, на которых модель упала
            # (например, неизвестная категория). Кривая получится
            # с пропусками, но не сломается целиком.
            print(f"[WARN] sensitivity skip {feature_name}={v}: {e}")

    return SensitivityResponse(
        model_name=model_name,
        feature_name=feature_name,
        points=points,
    )

def _numeric_grid(feature_meta: dict) -> list[float]:
    """
    Сетка значений для числового признака:
    линейное разбиение от min до max, N_POINTS_NUMERIC точек.

    Линейная сетка — самая простая. Для признаков с длинным
    хвостом распределения (price_per_sqm) лучше работала бы
    логарифмическая сетка или квантильная (по перцентилям).
    Но для area/floor/minutes_to_metro линейная адекватна.
    """
    lo = float(feature_meta.get("min", 0))
    hi = float(feature_meta.get("max", 100))
    grid = np.linspace(lo, hi, N_POINTS_NUMERIC)

    # Для целочисленных признаков (rooms, floor) округляем
    # и оставляем уникальные значения.
    if feature_meta.get("dtype") in {"int", "int64", "integer"}:
        grid = np.unique(np.round(grid).astype(int))

    return [float(x) for x in grid]
