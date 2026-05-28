"""
Фабрика и реестр ML-моделей.

Позволяет:
- Загрузить все доступные модели при старте приложения.
- Получить модель по имени для эндпоинтов.
- Загрузить метрики из metrics.json.

Паттерн: Registry + простая фабрика.
"""

from pathlib import Path
import json

from backend.app.core.config import settings
from backend.app.ml.base import ModelMetrics, BaseMLModel
from backend.app.ml.catboost_model import CatBoostModel
from backend.app.ml.xgboost_model import XGBoostModel
from backend.app.ml.random_forest_model import RandomForestModel
from backend.app.ml.linear_model import LinearModel

def _load_metrics() -> dict[str, ModelMetrics]:
    """
    Загружает метрики из ml_models/metrics.json.

    """
    metrics_path: Path = settings.METRICS_PATH
    if not metrics_path.exists():
        return {}  # модели не обучены — возвращаем пустой словарь

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

def load_models() -> dict[str, BaseMLModel]:
    """
    Загружает и возвращает словарь {имя_модели: экземпляр модели}.

    Модели, которые не удалось загрузить, пропускаются с логом в консоль.
    Приложение стартует, даже если не загрузилась ни одна модель
    (это позволяет разрабатывать фронт без обученных артефактов).
    """
    models_dir: Path = settings.MODELS_DIR
    metrics_map: dict[str, ModelMetrics] = _load_metrics()
    registry: dict[str, BaseMLModel] = {}

    # Проверяем, есть ли директория с моделями
    if not models_dir.exists():
        print(f"[WARN] Директория {models_dir} не найдена. Модели не загружены.")
        return registry


    model_classes = {
        "CatBoost": CatBoostModel,
        "XGBoost": XGBoostModel,
        "RandomForest": RandomForestModel,
        "LinearRegression": LinearModel,
    }

    for name, cls in model_classes.items():
        try:
            metrics = metrics_map.get(name, ModelMetrics(MAE=0, RMSE=0, MAPE=0, R2=0))
            model = cls(models_dir=models_dir, metrics=metrics)
            registry[name] = model
            print(f"[INFO] Загружена модель: {name} (MAE={metrics.MAE:,.0f})")
        except Exception as e:
            print(f"[WARN] Не удалось загрузить {name}: {e}")

    return registry
