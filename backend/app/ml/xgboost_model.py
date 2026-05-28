"""
XGBoostModel — конкретная реализация BaseMLModel для XGBoost.

XGBoost был обучен внутри sklearn.Pipeline с кодировщиками
(OneHotEncoder + OrdinalEncoder). Поэтому загружается полный pipeline.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from backend.app.ml.base import BaseMLModel, ModelMetrics
from backend.app.ml.enrichment import FeatureEnricher

class XGBoostModel(BaseMLModel):
    def __init__(self, models_dir: Path, metrics: ModelMetrics):
        super().__init__(name="XGBoost", metrics=metrics)
        model_path = models_dir / "xgboost.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Файл модели не найден: {model_path}")

        self._pipeline = joblib.load(str(model_path))
        # joblib.load() принимает str, не Path.
        # В Python 3.8+ pathlib поддерживается joblib,
        # но для единообразия с CatBoost оставляем str.

    def predict(self, features: dict) -> float:
        enriched = FeatureEnricher.enrich(features)
        df = pd.DataFrame([enriched])
        return float(self._pipeline.predict(df)[0])
