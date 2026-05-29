"""
XGBoostModel — конкретная реализация BaseMLModel для XGBoost.
Pipeline(preprocessor → XGBRegressor). SHAP через TreeExplainer.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app.ml.base import BaseMLModel, FeatureImportance, ModelMetrics
from app.ml.constants import (
    FILE_XGBOOST,
    MODEL_XGBOOST,
    ORIGINAL_FEATURE_NAMES,
)
from app.ml.enrichment import FeatureEnricher
from app.ml.shap_explainer import aggregate_to_original, explain_pipeline

class XGBoostModel(BaseMLModel):
    def __init__(self, models_dir: Path, metrics: ModelMetrics, background: pd.DataFrame):
        super().__init__(name=MODEL_XGBOOST, metrics=metrics, background=background)
        model_path = models_dir / FILE_XGBOOST
        if not model_path.exists():
            raise FileNotFoundError(f"Файл модели не найден: {model_path}")

        self._pipeline = joblib.load(str(model_path))

    def predict(self, features: dict) -> float:
        enriched = FeatureEnricher.enrich(features)
        df = pd.DataFrame([enriched])
        return float(self._pipeline.predict(df)[0])

    def _compute_shap(self, sample: pd.DataFrame) -> tuple[float, dict[str, float]]:
        return explain_pipeline(
            self._pipeline, sample, self._background, list(ORIGINAL_FEATURE_NAMES)
        )

    def get_feature_importance(self) -> list[FeatureImportance]:
        preprocessor = self._pipeline[:-1] if len(self._pipeline.steps) > 1 else None
        estimator = self._pipeline.steps[-1][1]

        if preprocessor is not None:
            transformed_names = list(preprocessor.get_feature_names_out())
        else:
            transformed_names = list(self._pipeline.feature_names_in_)

        raw = estimator.feature_importances_
        aggregated = aggregate_to_original(
            np.asarray(raw), transformed_names, list(ORIGINAL_FEATURE_NAMES)
        )
        items = sorted(aggregated.items(), key=lambda kv: kv[1], reverse=True)
        return [FeatureImportance(feature_name=n, importance=float(v)) for n, v in items]
