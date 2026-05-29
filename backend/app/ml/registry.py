"""
Фабрика и реестр ML-моделей.
"""

import json
from pathlib import Path

import pandas as pd

from app.core.config import settings
from app.ml.base import BaseMLModel, ModelMetrics
from app.ml.catboost_model import CatBoostModel
from app.ml.constants import (
    FILE_BACKGROUND,
    MODEL_CATBOOST,
    MODEL_LINEAR,
    MODEL_RANDOM_FOREST,
    MODEL_XGBOOST,
)
from app.ml.linear_model import LinearModel
from app.ml.random_forest_model import RandomForestModel
from app.ml.xgboost_model import XGBoostModel

def _load_metrics() -> dict[str, ModelMetrics]:
    metrics_path: Path = settings.METRICS_PATH
    if not metrics_path.exists():
        return {}

    with open(metrics_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    metrics_map: dict[str, ModelMetrics] = {}
    for model_name, data in raw.items():
        m = data.get("metrics", {})
        metrics_map[model_name] = ModelMetrics(
            MAE=m.get("MAE", 0.0),
            RMSE=m.get("RMSE", 0.0),
            MAPE=m.get("MAPE", 0.0),
            R2=m.get("R2", 0.0),
            best_params=data.get("best_params"),
        )
    return metrics_map

def _load_background(models_dir: Path) -> pd.DataFrame:
    """
    Фоновая выборка для SHAP-explainer (interventional baseline).
    Если файла нет — возвращаем пустой DataFrame; SHAP всё равно
    отработает (на пустом TreeExplainer fallback к tree_path_dependent),
    но рекомендуется всегда иметь background.parquet.
    """
    bg_path = models_dir / FILE_BACKGROUND
    if not bg_path.exists():
        print(f"[WARN] Background-файл не найден: {bg_path}")
        raise Exception("background.parquet не найден")

    return pd.read_csv(bg_path)

def load_models() -> dict[str, BaseMLModel]:
    models_dir: Path = settings.MODELS_DIR
    if not models_dir.exists():
        print(f"[WARN] Директория {models_dir} не найдена. Модели не загружены.")
        return {}

    metrics_map = _load_metrics()
    background = _load_background(models_dir)

    model_classes = {
        MODEL_CATBOOST: CatBoostModel,
        MODEL_XGBOOST: XGBoostModel,
        MODEL_RANDOM_FOREST: RandomForestModel,
        MODEL_LINEAR: LinearModel,
    }

    registry: dict[str, BaseMLModel] = {}
    for name, cls in model_classes.items():
        try:
            metrics = metrics_map.get(name, ModelMetrics(MAE=0, RMSE=0, MAPE=0, R2=0))
            registry[name] = cls(
                models_dir=models_dir,
                metrics=metrics,
                background=background,
            )
            print(f"[INFO] Загружена модель: {name} (MAE={metrics.MAE:,.0f})")
        except Exception as e:
            print(f"[WARN] Не удалось загрузить {name}: {e}")

    return registry
