"""
Абстрактный базовый класс для всех ML-моделей.
Реализует паттерн Strategy: каждая модель — отдельный класс
с единым интерфейсом predict / get_feature_importance / get_shap_values.

Методы get_feature_importance и get_shap_values — заглушки.
"""
# TODO get_feature_importance и get_shap_values

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelMetrics:
    """Метрики качества одной модели."""
    MAE: float
    RMSE: float
    MAPE: float
    R2: float
    best_params: Optional[dict] = None


@dataclass
class FeatureImportance:
    """Одна строка важности признака."""
    feature_name: str
    importance: float


@dataclass
class ShapValue:
    """Одно значение SHAP для одного признака."""
    feature_name: str
    value: float

@dataclass
class ShapExplanation:
    """Полное SHAP-разложение одного предсказания."""
    base_value: float
    shap_values: list[ShapValue] = field(default_factory=list)
    prediction: float = 0.0


@dataclass
class PredictionResult:
    """Результат предсказания одной модели."""
    model_name: str
    price: float
    ci_lower: float
    ci_upper: float
    price_per_sqm: float
    metrics: ModelMetrics


class BaseMLModel(ABC):
    """
    Абстрактная ML-модель.

    Все конкретные модели (CatBoost, XGBoost, RandomForest, Linear)
    наследуют этот класс и реализуют 4 обязательных метода.
    """

    def __init__(self, name: str, metrics: ModelMetrics):
        self._name = name
        self._metrics = metrics
        self._explainer = None  # будет установлен в главе 3

    @property
    def name(self) -> str:
        return self._name

    @property
    def metrics(self) -> ModelMetrics:
        return self._metrics

    @abstractmethod
    def predict(self, features: dict) -> float:
        """
        Предсказать цену для одного объекта.

        Args:
            features: словарь признаков (сырые, без производных —
                      FeatureEnricher применяется до вызова predict).

        Returns:
            Цена в рублях (float).
        """
        ...

    def get_feature_importance(self) -> list[FeatureImportance]:
        # TODO доделать
        """
        Возвращает важность признаков (Feature Importance).

        Заглушка
        """
        return []

    def get_shap_values(self, features: dict) -> ShapExplanation:
        # TODO доделать
        """
        Возвращает SHAP-разложение для одного объекта.

        Сейчас возвращает ShapExplanation с пустым списком значений.

        Args:
            features: словарь признаков (после FeatureEnricher).

        Returns:
            ShapExplanation с base_value = 0 и пустым списком.
        """
        return ShapExplanation(base_value=0.0)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self._name}>"
