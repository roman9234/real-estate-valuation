"""
LinearModel — конкретная реализация BaseMLModel для линейной регрессии.

Особенность:
- Модель обучалась на логарифмированной целевой переменной (log_price).
- Pipeline включает StandardScaler + Ridge.
- Результат предсказания нужно преобразовать обратно: expm1().

При добавлении SHAP-разложения для этой модели
потребуется LinearExplainer из shap, которому нужны уже
преобразованные (закодированные/масштабированные) признаки.
Это значит, что explainer должен работать ПОСЛЕ pipeline,
а не вокруг него — в отличие от древесных моделей.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from backend.app.ml.base import BaseMLModel, ModelMetrics
from backend.app.ml.enrichment import FeatureEnricher

class LinearModel(BaseMLModel):
    def __init__(self, models_dir: Path, metrics: ModelMetrics):
        super().__init__(name="LinearRegression", metrics=metrics)
        model_path = models_dir / "linear_regression.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Файл модели не найден: {model_path}")

        self._pipeline = joblib.load(str(model_path))

    def predict(self, features: dict) -> float:
        enriched = FeatureEnricher.enrich(features)
        df = pd.DataFrame([enriched])
        y_log = self._pipeline.predict(df)[0]
        # expm1 вместо exp: модель обучалась на log(price + 1),
        # а не на log(price). Если бы использовался log(price),
        # здесь был бы np.exp(y_log). Разница мала (~1 рубль),
        # но неправильный выбор даст систематическое смещение
        # для дешёвых квартир.
        return float(np.expm1(y_log))
