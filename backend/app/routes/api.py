"""
HTTP-эндпоинты приложения.

Все роуты под префиксом /api/v1. Версионирование через URL —
самый простой способ; при breaking changes делается /api/v2.
"""

from fastapi import APIRouter, Depends, Request

from backend.app.core.features_meta import get_features_meta
from backend.app.ml.base import BaseMLModel
from backend.app.schemas.request import ApartmentRequest
from backend.app.schemas.response import (
    FeatureImportanceResponse,
    ModelInfo,
    ModelsListResponse,
    MetricsResponse,
    PredictionResponse,
    SensitivityResponse,
    ShapResponse,
)
from backend.app.services.explanation import get_feature_importance, get_shap_explanation
from backend.app.services.prediction import predict_all
from backend.app.services.sensitivity import get_sensitivity

router = APIRouter(prefix="/api/v1")

def get_models(request: Request) -> dict[str, BaseMLModel]:
    """
    Зависимость FastAPI: достаёт рестр моделей из app.state

    Глобальный рестр в app.state — самый простой DI для
    одного процесса. Для масштабирования (несколько воркеров
    Uvicorn) каждый воркер загрузит модели в свою память —
    оверхед по RAM, но никаких race condition.
    Альтернатива — вынести модели в отдельный сервис (gRPC)
    """
    return request.app.state.models

@router.get("/health")
async def health() -> dict:
    """Простой healthcheck для Docker/K8s."""
    return {"status": "ok"}

@router.get("/features")
async def features() -> dict:
    """
    Метаданные признаков для построения формы на фронте.
    Возвращает обогащённый feature_catalog.
    """
    return get_features_meta()

@router.get("/models", response_model=ModelsListResponse)
async def models_list(
    models: dict[str, BaseMLModel] = Depends(get_models),
) -> ModelsListResponse:
    """Список загруженных моделей с их метриками."""
    return ModelsListResponse(
        models=[
            ModelInfo(
                name=m.name,
                metrics=MetricsResponse(**m.metrics.__dict__),
            )
            for m in models.values()
        ]
    )

@router.post("/predict", response_model=PredictionResponse)
async def predict(
    payload: ApartmentRequest,
    models: dict[str, BaseMLModel] = Depends(get_models),
) -> PredictionResponse:
    """Предсказание цены всеми моделями."""
    return predict_all(payload.to_features(), models)

@router.get(
    "/feature-importance/{model_name}",
    response_model=FeatureImportanceResponse,
)
async def feature_importance(
    model_name: str,
    models: dict[str, BaseMLModel] = Depends(get_models),
) -> FeatureImportanceResponse:
    """
    Важность признаков для конкретной модели.
    🚲 Заглушка до главы 3.
    """
    return get_feature_importance(model_name, models)

@router.post("/explain/{model_name}", response_model=ShapResponse)
async def explain(
    model_name: str,
    payload: ApartmentRequest,
    models: dict[str, BaseMLModel] = Depends(get_models),
) -> ShapResponse:
    """
    SHAP-разложение для конкретной квартиры.
    🚲 Заглушка до главы 3.
    """
    return get_shap_explanation(model_name, payload.to_features(), models)

@router.post(
    "/sensitivity/{model_name}/{feature_name}",
    response_model=SensitivityResponse,
)

async def sensitivity(
    model_name: str,
    feature_name: str,
    payload: ApartmentRequest,
    models: dict[str, BaseMLModel] = Depends(get_models),
) -> SensitivityResponse:
    """
    Кривая чувствительности по одному признаку.

    Почему POST, а не GET, хотя ничего не создаём?
    Тело запроса содержит фиксированные значения остальных
    признаков (8 полей), которые передавать в URL неудобно.
    POST с body — пагматичный выбор.
    """
    return get_sensitivity(model_name, feature_name, payload.to_features(), models)
