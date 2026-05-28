"""
Pydantic-схемы входящих запросов.

Pydantic V2 валидирует типы и диапазоны автоматически — если
прилетит что-то не то, FastAPI вернёт 422 автоматически
"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

# Допустимые значения для категориальных полей.
# Дублирование с feature_catalog.json: значения захардкожены
# и здесь, и в каталоге. Если в датасете появится новый тип
# ремонта — нужно будет править оба места.
# Правильнее — динамически собирать Literal из feature_catalog
# при старте, но Literal в pydantic должен быть статическим.

RenovationType = Literal[
    "Cosmetic",
    "Designer",
    "European-style renovation",
    "Without renovation",
]


class ApartmentRequest(BaseModel):
    """
    Признаки квартиры для предсказания.

    Диапазоны значений сответствуют min/max из обучающего датасета.
    Выход за пределы —422 от Pydantic.
    """

    area: float = Field(..., ge = 10, le = 500.0, description = "Площадь, м²")
    rooms: int = Field(..., ge=0, le=15, description="Число комнат (0 = студия)")
    floor: int = Field(..., ge=1, le=100, description="Этаж")
    total_floors: int = Field(..., ge=1, le=100, description="Этажей в доме")
    minutes_to_metro: int = Field(..., ge=0, le=120, description="Минут до метро")
    is_studio: int = Field(..., ge=0, le=1, description="0 или 1")
    metro_station: str = Field(..., min_length = 1, max_length = 100)
    renovation: RenovationType

    @model_validator(mode="after")
    def floor_not_above_total(self) -> "ApartmentRequest":
        """
        Бизнес-правило: этаж не может быть больше количества этажей в доме.
        Pydantic не покрывает зависимости между полями
        """
        if self.floor > self.total_floors:
            raise ValueError(
                f"floor ({self.floor}) не может быть больше "
                f"total_floors ({self.total_floors})"
            )
        return self

    def to_features(self) -> dict:
        """
        Преобразует в плоский dict для передачи в ML-слой.
        FeatureEnricher добавит производные поля внутри predict().
        """
        return self.model_dump()
