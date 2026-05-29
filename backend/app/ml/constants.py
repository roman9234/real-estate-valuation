"""
Константы ML-слоя: имена моделей, имена файлов артефактов,
список исходных признаков для SHAP-агрегации.

Единый источник истины. Любая ссылка на «CatBoost» как строку
в коде или тестах должна идти через эти константы.
"""

from typing import Final

# ── Имена моделей ───────────────────────────────────────────
# Используются как ключи в registry и в URL эндпоинтов.

MODEL_CATBOOST: Final[str] = "CatBoost"
MODEL_XGBOOST: Final[str] = "XGBoost"
MODEL_RANDOM_FOREST: Final[str] = "RandomForest"
MODEL_LINEAR: Final[str] = "LinearRegression"

ALL_MODEL_NAMES: Final[tuple[str, ...]] = (
    MODEL_CATBOOST,
    MODEL_XGBOOST,
    MODEL_RANDOM_FOREST,
    MODEL_LINEAR,
)

# ── Имена файлов артефактов в MODELS_DIR ────────────────────

FILE_CATBOOST: Final[str] = "catboost.cbm"
FILE_XGBOOST: Final[str] = "xgboost.pkl"
FILE_RANDOM_FOREST: Final[str] = "random_forest.pkl"
FILE_LINEAR: Final[str] = "linear_regression.pkl"
FILE_BACKGROUND: Final[str] = "background.parquet"
FILE_METRICS: Final[str] = "metrics.json"
FILE_FEATURE_CATALOG: Final[str] = "feature_catalog.json"

# ── Исходные признаки (до FeatureEnricher) ──────────────────
# Соответствуют полям ApartmentRequest.
# SHAP-агрегация сворачивает one-hot и производные обратно
# в эти имена. Если добавляется новый признак в форме —
# обновлять здесь и в ApartmentRequest одновременно.

ORIGINAL_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "area",
    "rooms",
    "floor",
    "total_floors",
    "minutes_to_metro",
    "is_studio",
    "metro_station",
    "renovation",
)

# ── Параметры SHAP-explainer ────────────────────────────────

# Размер фоновой выборки для interventional TreeExplainer
# и для LinearExplainer. 100 — компромисс между точностью
# вклада и временем построения explainer (≈1–2 сек на модель).
SHAP_BACKGROUND_SIZE: Final[int] = 100
