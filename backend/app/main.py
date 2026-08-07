"""
Точка входа FastAPI-приложения.

Здесь:
- Создаётся приложение FastAPI.
- В lifespan загружаются модели один раз при старте.
- Подключаются CORS, exception handlers, роуты.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.exceptions import DomainError, domain_error_handler
from app.ml.registry import load_models
from app.routes.api import router as api_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Загружаем модели один раз при старте приложения.

    Почему lifespan, а не on_event("startup")
    on_event объявлен deprecated в FastAPI 0.93+. Lifespan —
    рекомендованый способ через async context manager.
    """
    print("[INFO] Загрузка ML-моделей...")
    app.state.models = load_models()
    print(f"[INFO] Загружено моделей: {len(app.state.models)}")
    yield
    # На выходе ничего не освобождаем: GC заберёт сам.
    # Если бы были коннекты к БД/Redis — закрывали бы здесь.

app = FastAPI(
    title="Apartment Price Predictor",
    description="Backend дипломного проекта: предсказание цен на квартиры",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS: фронт будет на 5173 (Vite dev) или 80 (nginx в проде).
# allow_origins=["*"] было бы проще, но небезопасно. Указываем явные origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev
        "http://localhost:3000",   # Альтернативный порт
        "http://localhost",        # nginx в docker
        "http://localhost:80",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Все доменые ошибки - JSON через единый хендлер
app.add_exception_handler(DomainError, domain_error_handler)

app.include_router(api_router, prefix="/api")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
    )
