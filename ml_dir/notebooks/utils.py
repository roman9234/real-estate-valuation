"""
Утилиты для обучения и оценки
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === Пути проекта ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "ml_models"
REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_PATH = MODELS_DIR / "metrics.json"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

import json

# === Валидация категориальных признаков ===

CATALOG_PATH = MODELS_DIR / "feature_catalog.json"

def load_catalog() -> dict:
    """Загружает справочник признаков (если он построен)."""
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(
            f"Справочник не найден: {CATALOG_PATH}. "
            f"Запустите ноутбук 00_build_feature_catalog.ipynb."
        )
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def validate_features(features: dict, strict: bool = True) -> list[str]:
    """
    Проверяет признаки на соответствие справочнику.
    Возвращает список предупреждений. При strict=True кидает ValueError.
    """
    catalog = load_catalog()
    warnings = []

    # Категориальные: значение должно быть в справочнике
    for col, info in catalog["categorical_features"].items():
        if col not in features:
            continue
        value = str(features[col])
        if value not in info["values"]:
            msg = (f"[{col}] значение '{value}' отсутствует в обучающей выборке. "
                   f"Доступно {info['n_unique']} вариантов.")
            warnings.append(msg)

    # Числовые: значение должно попадать в [q01, q99] (мягкая проверка)
    for col, info in catalog["numeric_features"].items():
        if col not in features:
            continue
        try:
            v = float(features[col])
        except (TypeError, ValueError):
            warnings.append(f"[{col}] не число: {features[col]}")
            continue
        if v < info["min"] or v > info["max"]:
            warnings.append(
                f"[{col}] значение {v} вне диапазона обучающей выборки "
                f"[{info['min']}, {info['max']}]."
            )

    if strict and warnings:
        raise ValueError("Невалидные признаки:\n  - " + "\n  - ".join(warnings))
    return warnings



# === Загрузка данных ===
def load_data():
    """Загружает X_train, X_test, y_train, y_test, y_log_train, y_log_test."""
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_test  = pd.read_csv(DATA_DIR / "X_test.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze("columns")
    y_test  = pd.read_csv(DATA_DIR / "y_test.csv").squeeze("columns")
    y_log_train = pd.read_csv(DATA_DIR / "y_log_train.csv").squeeze("columns")
    y_log_test  = pd.read_csv(DATA_DIR / "y_log_test.csv").squeeze("columns")
    return X_train, X_test, y_train, y_test, y_log_train, y_log_test

# === Метрики ===
def evaluate_model(y_true, y_pred, model_name: str, verbose: bool = True) -> dict:
    """Считает метрики в исходном масштабе рублей."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    # R2 без зависимости от sklearn (для прозрачности)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot

    metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}

    if verbose:
        print(f"\n=== {model_name} ===")
        print(f"  MAE  = {mae:>15,.0f} руб.")
        print(f"  RMSE = {rmse:>15,.0f} руб.")
        print(f"  MAPE = {mape:>14.2f} %")
        print(f"  R2   = {r2:>14.4f}")

    return metrics

def save_metrics(model_name: str, metrics: dict, best_params: dict | None = None):
    """Дописывает метрики модели в общий metrics.json."""
    all_metrics = {}
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            all_metrics = json.load(f)
    all_metrics[model_name] = {"metrics": metrics, "best_params": best_params or {}}
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)
    print(f"Метрики сохранены в {METRICS_PATH}")

# === Визуализация ===
def plot_feature_importance(feature_names, importances, model_name: str,
                            top_n: int = 15, save: bool = True):
    """Горизонтальная диаграмма важности признаков. Сохраняет PNG в reports/."""
    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df = df.sort_values("importance", ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(df["feature"], df["importance"], color="#4f86c6")
    ax.set_xlabel("Важность")
    ax.set_title(f"Важность признаков — {model_name}")
    plt.tight_layout()

    if save:
        path = REPORTS_DIR / f"feature_importance_{model_name.lower()}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"График сохранён: {path}")

    plt.show()
    return df

# === Колонки по группам ===
NUMERIC_FEATURES_FULL = [
    "area", "rooms", "floor", "total_floors", "minutes_to_metro",
    "floor_ratio", "is_first_floor", "is_last_floor", "is_studio", "log_area",
]
# Для древесных моделей log_area исключаем
NUMERIC_FEATURES_TREES = [f for f in NUMERIC_FEATURES_FULL if f != "log_area"]
CATEGORICAL_FEATURES = ["metro_station", "renovation"]
