"""
Сервис интерпретации: Feature Importance + SHAP.
Тонкая обёртка над BaseMLModel — вся логика в доменном слое.
"""

from backend.app.core.exceptions import ModelNotFoundError
from backend.app.ml.base import BaseMLModel
from backend.app.schemas.response import (
    FeatureImportanceItem,
    FeatureImportanceResponse,
    ShapResponse,
    ShapValueItem,
)

def _get_model(model_name: str, models: dict[str, BaseMLModel]) -> BaseMLModel:
    if model_name not in models:
        raise ModelNotFoundError(
            f"Модель '{model_name}' не загружена",
            details={"available": list(models.keys())},
        )
    return models[model_name]

def get_feature_importance(
    model_name: str,
    models: dict[str, BaseMLModel],
) -> FeatureImportanceResponse:
    model = _get_model(model_name, models)
    raw = model.get_feature_importance()
    return FeatureImportanceResponse(
        model_name=model_name,
        features=[
            FeatureImportanceItem(feature_name=f.feature_name, importance=f.importance)
            for f in raw
        ],
    )

def get_shap_explanation(
    model_name: str,
    features: dict,
    models: dict[str, BaseMLModel],
) -> ShapResponse:
    model = _get_model(model_name, models)
    explanation = model.get_shap_values(features)
    return ShapResponse(
        model_name=model_name,
        base_value=explanation.base_value,
        prediction=explanation.prediction,
        shap_values=[
            ShapValueItem(feature_name=v.feature_name, value=v.value)
            for v in explanation.shap_values
        ],
    )
