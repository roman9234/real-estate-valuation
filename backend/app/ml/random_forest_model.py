"""
RandomForestModel — конкретная реализация BaseMLModel для Random Forest.
Полностью аналогична XGBoostModel: загружает sklearn.Pipeline
с кодировщиками, обогащает, предсказывает.
"""

from pathlib import Path

import joblib
import pandas as pd

from backend.app.ml.base import BaseMLModel, ModelMetrics
from backend.app.ml.enrichment import FeatureEnricher

class RandomForestModel(BaseMLModel):
    def __init__(self, models_dir: Path, metrics: ModelMetrics):
        super().__init__(name="RandomForest", metrics=metrics)
        model_path = models_dir / "random_forest.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Файл модели не найден: {model_path}")

        self._pipeline = joblib.load(str(model_path))

    def predict(self, features: dict) -> float:
        enriched = FeatureEnricher.enrich(features)
        df = pd.DataFrame([enriched])
        return float(self._pipeline.predict(df)[0])
