"""
Конфигурация приложения.
Читает пути к артефактам ML и базовые настройки из окружения.
"""

from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Использование pydantic_settings вместо os.getenv
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
    MODELS_DIR: Path = PROJECT_ROOT / "ml_models"
    FEATURES_META_PATH: Path = MODELS_DIR / "feature_catalog.json"
    METRICS_PATH: Path = MODELS_DIR / "metrics.json"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

settings = Settings()
