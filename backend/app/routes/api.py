"""
HTTP-эндпоинты приложения. Префикс /api/v1.
"""

from fastapi import APIRouter, Depends, Request

from backend.app.core.features_meta import get_features_meta
from backend.app.ml.base import BaseMLModel
from backend.app.schemas.request import ApartmentRequest
from backend.app.schemas.response import (
    FeatureImportanceResponse,
    MetricsResponse,
    ModelInfo,
    ModelsListResponse,
    PredictionResponse,
    SensitivityResponse,
    ShapResponse,
)
from backend.app.services.explanation import (
    get_feature_importance,
    get_shap_explanation,
)
from backend.app.services.prediction import predict_all
from backend.app.services.sensitivity import get_sensitivity

router = APIRouter(prefix="/api/v1")

def get_models(request: Request) -> dict[str, BaseMLModel]:
    return request.app.state.models

router.get("/health")
async def health() -> dict:
    return {"status": "ok"}

router.get("/features")
async def features() -> dict:
    return get_features_meta()

router.get("/models", response_model=ModelsListResponse)
async def models_list(
    models: dict[str, BaseMLModel] = Depends(get_models),
) -> ModelsListResponse:
    return ModelsListResponse(
        models=[
            ModelInfo(name=m.name, metrics=MetricsResponse(**m.metrics.__dict__))
            for m in models.values()
        ]
    )

router.post("/predict", response_model=PredictionResponse)
async def predict(
    payload: ApartmentRequest,
    models: dict[str, BaseMLModel] = Depends(get_models),
) -> PredictionResponse:
    return predict_all(payload.to_features(), models)

router.get(
    "/feature-importance/{model_name}",
    response_model=FeatureImportanceResponse,
)
async def feature_importance(
    model_name: str,
    models: dict[str, BaseMLModel] = Depends(get_models),
) -> FeatureImportanceResponse:
    return get_feature_importance(model_name, models)

router.post("/explain/{model_name}", response_model=ShapResponse)
async def explain(
    model_name: str,
    payload: ApartmentRequest,
    models: dict[str, BaseMLModel] = Depends(get_models),
) -> ShapResponse:
    return get_shap_explanation(model_name, payload.to_features(), models)

router.post(
    "/sensitivity/{model_name}/{feature_name}",
    response_model=SensitivityResponse,
)
async def sensitivity(
    model_name: str,
    feature_name: str,
    payload: ApartmentRequest,
    models: dict[str, BaseMLModel] = Depends(get_models),
) -> SensitivityResponse:
    return get_sensitivity(model_name, feature_name, payload.to_features(), models)
