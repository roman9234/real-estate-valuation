"""
Примитивный инференс обученных моделей.
Назначение: проверить предсказания вне ноутбуков обучения,
использовать в дальнейших этапах до того, как будет написан полноценный backend.

Использование:
    from inference import predict_catboost
    features = {
        "area": 55.0, "rooms": 2.0, "floor": 5.0, "total_floors": 9,
        "minutes_to_metro": 10.0, "is_studio": 0,
        "metro_station": "Профсоюзная", "renovation": "Cosmetic",
    }
    price = predict_catboost(features)
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "ml_models"

# === Вспомогательное: достраиваем производные признаки ===
def _enrich_features(features: dict) -> dict:
    """Дополняет словарь признаков производными полями, если их нет."""
    f = dict(features)
    floor = float(f["floor"])
    total_floors = float(f["total_floors"])

    f.setdefault("floor_ratio", floor / total_floors if total_floors else 0.0)
    f.setdefault("is_first_floor", int(floor == 1))
    f.setdefault("is_last_floor", int(floor == total_floors))
    f.setdefault("log_area", float(np.log1p(float(f["area"]))))
    return f

# === Линейная регрессия (обучена на log-таргете) ===
def predict_linear(features: dict) -> float:
    pipeline = joblib.load(MODELS_DIR / "linear_regression.pkl")
    df = pd.DataFrame([_enrich_features(features)])
    y_log_pred = pipeline.predict(df)[0]
    return float(np.expm1(y_log_pred))

# === Random Forest ===
def predict_random_forest(features: dict) -> float:
    pipeline = joblib.load(MODELS_DIR / "random_forest.pkl")
    df = pd.DataFrame([_enrich_features(features)])
    return float(pipeline.predict(df)[0])

# === XGBoost ===
def predict_xgboost(features: dict) -> float:
    pipeline = joblib.load(MODELS_DIR / "xgboost.pkl")
    df = pd.DataFrame([_enrich_features(features)])
    return float(pipeline.predict(df)[0])

# === CatBoost (категории нативные) ===
def predict_catboost(features: dict) -> float:
    from catboost import CatBoostRegressor
    model = CatBoostRegressor()
    model.load_model(str(MODELS_DIR / "catboost.cbm"))
    df = pd.DataFrame([_enrich_features(features)])
    # Берём ровно те колонки, на которых модель обучалась, в правильном порядке
    df = df[model.feature_names_]
    return float(model.predict(df)[0])

# === Единая точка входа ===
PREDICTORS = {
    "LinearRegression": predict_linear,
    "RandomForest": predict_random_forest,
    "XGBoost": predict_xgboost,
    "CatBoost": predict_catboost,
}

def predict(model_name: str, features: dict) -> float:
    if model_name not in PREDICTORS:
        raise ValueError(f"Неизвестная модель: {model_name}. Доступны: {list(PREDICTORS)}")
    return PREDICTORS[model_name](features)

if __name__ == "__main__":
    # Минимальный smoke-test
    example = {
        "area": 78.9, "rooms": 3.0, "floor": 4.0, "total_floors": 5,
        "minutes_to_metro": 18.0, "is_studio": 0,
        "metro_station": "Профсоюзная", "renovation": "Cosmetic",
    }
    for name in PREDICTORS:
        try:
            price = predict(name, example)
            print(f"{name:>18}: {price:>15,.0f} руб.")
        except FileNotFoundError:
            print(f"{name:>18}: модель не обучена")
