"""
Pydantic-схемы исходящих ответов.

Используются как response_model в FastAPI — это включает
автоматическую валидацию ответа и генерацию OpenAPI-схемы.
"""

from typing import Optional

from pydantic import BaseModel, Field

class MetricsResponse(BaseModel):
    MAE: float
    RMSE: float
    MAPE: float
    R2: float
    best_params: Optional[dict] = None

class PredictionItem(BaseModel):
    """Результат одной модели в /predict."""

    model_name: str
    price: float = Field(..., description="Цена в рублях")
    ci_lower: float
    ci_upper: float
    price_per_sqm: float
    metrics: MetricsResponse

class PredictionResponse(BaseModel):
    """Ответ /predict — список результатов всех моделей."""

    predictions: list[PredictionItem]

class FeatureImportanceItem(BaseModel):
    feature_name: str
    importance: float

class FeatureImportanceResponse(BaseModel):
    model_name: str
    features: list[FeatureImportanceItem]

class ShapValueItem(BaseModel):
    feature_name: str
    value: float

class ShapResponse(BaseModel):
    model_name: str
    base_value: float
    prediction: float
    shap_values: list[ShapValueItem]

class ModelInfo(BaseModel):
    name: str
    metrics: MetricsResponse

class ModelsListResponse(BaseModel):
    models: list[ModelInfo]

class SensitivityPoint(BaseModel):
    """Одна точка кривой чувствительности"""

    # числовое или категориальное значение
    value: float | str
    price: float

class SensitivityResponse(BaseModel):
    model_name: str
    feature_name: str
    points: list[SensitivityPoint]
