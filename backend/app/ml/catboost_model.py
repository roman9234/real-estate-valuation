"""
CatBoostModel — конкретная реализация BaseMLModel для CatBoost.
SHAP считается через нативный get_feature_importance(type='ShapValues'),
который понимает категориальные признаки без OHE.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from backend.app.ml.base import BaseMLModel, FeatureImportance, ModelMetrics
from backend.app.ml.constants import (
    FILE_CATBOOST,
    MODEL_CATBOOST,
    ORIGINAL_FEATURE_NAMES,
)
from backend.app.ml.enrichment import FeatureEnricher
from backend.app.ml.shap_explainer import aggregate_to_original, explain_catboost

class CatBoostModel(BaseMLModel):
    def __init__(self, models_dir: Path, metrics: ModelMetrics, background: pd.DataFrame):
        super().__init__(name=MODEL_CATBOOST, metrics=metrics, background=background)
        model_path = models_dir / FILE_CATBOOST
        if not model_path.exists():
            raise FileNotFoundError(f"Файл модели не найден: {model_path}")

        self._model = CatBoostRegressor()
        self._model.load_model(str(model_path))
        self._feature_names = list(self._model.feature_names_)

    def predict(self, features: dict) -> float:
        enriched = FeatureEnricher.enrich(features)
        df = pd.DataFrame([enriched])[self._feature_names]
        return float(self._model.predict(df)[0])

    def _compute_shap(self, sample: pd.DataFrame) -> tuple[float, dict[str, float]]:
        sample = sample[self._feature_names]
        return explain_catboost(self._model, sample, list(ORIGINAL_FEATURE_NAMES))

    def get_feature_importance(self) -> list[FeatureImportance]:
        raw = self._model.get_feature_importance()
        aggregated = aggregate_to_original(
            np.asarray(raw),
            self._feature_names,
            list(ORIGINAL_FEATURE_NAMES),
        )
        items = sorted(aggregated.items(), key=lambda kv: kv[1], reverse=True)
        return [FeatureImportance(feature_name=n, importance=float(v)) for n, v in items]
