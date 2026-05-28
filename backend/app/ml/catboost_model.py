"""
CatBoostModel — конкретная реализация BaseMLModel для CatBoost.

CatBoost — единственная модель, которая работает с категориальными
признаками нативно (через cat_features). Остальные модели используют
кодировщики (OneHotEncoder, TargetEncoder, OrdinalEncoder) внутри
своих sklearn.Pipeline.

Поэтому загрузка CatBoost отличается:
- Сама модель загружается через CatBoostRegressor.load_model()
- Порядок колонок берётся из model.feature_names_
- Не требуется pipeline — только модель.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from backend.app.ml.base import BaseMLModel, ModelMetrics
from backend.app.ml.enrichment import FeatureEnricher

class CatBoostModel(BaseMLModel):
    def __init__(self, models_dir: Path, metrics: ModelMetrics):
        super().__init__(name="CatBoost", metrics=metrics)
        model_path = models_dir / "catboost.cbm"

        if not model_path.exists():
            raise FileNotFoundError(f"Файл модели не найден: {model_path}")

        self._model = CatBoostRegressor()
        # load_model() в catboost принимает только строку str,
        # Path не поддерживается. Приведение явное — костыль,
        # но неизбежный. Обёртка не нужна — одна строка.
        self._model.load_model(str(model_path))
        self._feature_names = self._model.feature_names_

    def predict(self, features: dict) -> float:
        enriched = FeatureEnricher.enrich(features)
        df = pd.DataFrame([enriched])

        # CatBoost сам знает, какие колонки категориальные.
        # Но нужно убедиться, что порядок колонок совпадает
        # с порядком при обучении.
        df = df[self._feature_names]

        raw = self._model.predict(df)
        return float(raw[0])
