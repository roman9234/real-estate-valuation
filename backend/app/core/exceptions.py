"""
Кастомные исключения их обработчики.

Зачем выносить в отдельный модуль:
- Любой слой (service, ml) может бросить доменную ошибку,
  не зная про FastAPI.
- Преобразование в HTTP-ответ происходит централизованно
  в main.py через app.add_exception_handler().
"""

from fastapi import Request
from fastapi.responses import JSONResponse


class DomainError(Exception):
    """Базовый класс для всех бизнес-ошибок приложения."""

    status_code: int = 500
    error_code: str = "internal_error"

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ModelNotFoundError(DomainError):
    """Запрошена модель, которой нет в рестре."""
    status_code = 404
    error_code = "model_not_found"


class ModelNotLoadedError(DomainError):
    """Рестр пуст — приложение запущено без артефактов."""
    status_code = 503
    error_code = "models_not_loaded"


class FeaturesMetaNotFoundError(DomainError):
    """feature_catalog.json отсутствует."""
    status_code = 503
    error_code = "features_meta_not_found"


class PredictionError(DomainError):
    """Модель упала на инференсе (например, неизвестная категория)."""
    status_code = 422
    error_code = "prediction_failed"


async def domain_error_handler(request: Request, exc: DomainError) -> JSONResponse:
    """
    Универсальный хендлер для всех DomainError.

    Возвращает единообразный JSON: {error_code, message, details}.
    Подключается в main.py через app.add_exception_handler.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error_code": exc.error_code,
            "message": exc.message,
            "details": exc.details,
        },
    )
