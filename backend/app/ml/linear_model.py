"""
LinearModel — Pipeline(StandardScaler + encoders + Ridge),
обучен на log(price + 1).

SHAP считается в log-шкале (LinearExplainer), затем
переводится в рубли поправкой через прогноз. Аддитивность
сохраняется приближённо: для большинства квартир расхождение
< 1% (проверяется тестом test_additivity_property).
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app.ml.base import BaseMLModel, FeatureImportance, ModelMetrics
from app.ml.constants import (
    FILE_LINEAR,
    MODEL_LINEAR,
    ORIGINAL_FEATURE_NAMES,
)
from app.ml.enrichment import FeatureEnricher
from app.ml.shap_explainer import aggregate_to_original, explain_pipeline

class LinearModel(BaseMLModel):
    def __init__(self, models_dir: Path, metrics: ModelMetrics, background: pd.DataFrame):
        super().__init__(name=MODEL_LINEAR, metrics=metrics, background=background)
        model_path = models_dir / FILE_LINEAR
        if not model_path.exists():
            raise FileNotFoundError(f"Файл модели не найден: {model_path}")

        self._pipeline = joblib.load(str(model_path))

    def predict(self, features: dict) -> float:
        enriched = FeatureEnricher.enrich(features)
        df = pd.DataFrame([enriched])
        y_log = self._pipeline.predict(df)[0]
        return float(np.expm1(y_log))

    def _compute_shap(self, sample: pd.DataFrame) -> tuple[float, dict[str, float]]:
        base_log, contrib_log = explain_pipeline(
            self._pipeline, sample, self._background, list(ORIGINAL_FEATURE_NAMES)
        )
        # Перевод из логарифма в рубли.
        # prediction_log = base_log + sum(contrib_log_i)
        # prediction_rub = expm1(prediction_log)
        # Распределяем разницу пропорционально contrib_log_i:
        # contrib_rub_i = contrib_log_i / sum(contrib_log) * (pred_rub - base_rub)
        base_rub = float(np.expm1(base_log))
        prediction_log = base_log + sum(contrib_log.values())
        prediction_rub = float(np.expm1(prediction_log))
        total_log = sum(contrib_log.values())

        if abs(total_log) < 1e-9:
            contrib_rub = {k: 0.0 for k in contrib_log}
        else:
            scale = (prediction_rub - base_rub) / total_log
            contrib_rub = {k: v * scale for k, v in contrib_log.items()}

        return base_rub, contrib_rub

    def get_feature_importance(self) -> list[FeatureImportance]:
        preprocessor = self._pipeline[:-1] if len(self._pipeline.steps) > 1 else None
        estimator = self._pipeline.steps[-1][1]

        if preprocessor is not None:
            transformed_names = list(preprocessor.get_feature_names_out())
        else:
            transformed_names = list(self._pipeline.feature_names_in_)

        # |coef_| после стандартизации — корректная мера важности.
        raw = np.abs(estimator.coef_)
        aggregated = aggregate_to_original(
            np.asarray(raw), transformed_names, list(ORIGINAL_FEATURE_NAMES)
        )
        items = sorted(aggregated.items(), key=lambda kv: kv[1], reverse=True)
        return [FeatureImportance(feature_name=n, importance=float(v)) for n, v in items]
