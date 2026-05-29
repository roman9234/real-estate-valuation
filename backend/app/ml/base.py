"""
Абстрактный базовый класс для всех ML-моделей.
Паттерн Strategy: каждая модель реализует predict, _compute_shap,
get_feature_importance с единой сигнатурой.

Аддитивная сборка ShapExplanation вынесена сюда (template method) —
конкретные классы возвращают только «сырые» SHAP-значения
по исходным признакам, обёртку в ShapExplanation делает база.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from backend.app.ml.enrichment import FeatureEnricher

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
    value: float  # вклад в рублях

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

    Все конкретные модели наследуют этот класс и реализуют:
    - predict
    - _compute_shap (вычисление сырых SHAP по исходным признакам)
    - get_feature_importance
    """

    def __init__(
        self,
        name: str,
        metrics: ModelMetrics,
        background: pd.DataFrame,
    ):
        self._name = name
        self._metrics = metrics
        self._background = background

    @property
    def name(self) -> str:
        return self._name

    @property
    def metrics(self) -> ModelMetrics:
        return self._metrics

    @property
    def background(self) -> pd.DataFrame:
        return self._background

    # ── Обязательные методы ─────────────────────────────────

    @abstractmethod
    def predict(self, features: dict) -> float:
        """Предсказать цену для одного объекта (в рублях)."""
        ...

    @abstractmethod
    def _compute_shap(self, sample: pd.DataFrame) -> tuple[float, dict[str, float]]:
        """
        Возвращает (base_value, {feature_name: contribution}).

        Контракт:
        - base_value в той же шкале, что и итоговая цена в рублях.
        - Ключи словаря — только имена из ORIGINAL_FEATURE_NAMES.
        - Сумма contributions + base_value ≈ predict(features).

        Конкретные классы внутри решают: использовать
        TreeExplainer, LinearExplainer или CatBoost native.
        """
        ...

    @abstractmethod
    def get_feature_importance(self) -> list[FeatureImportance]:
        """Глобальная важность признаков, отсортированная по убыванию."""
        ...

    # ── Template method для SHAP ────────────────────────────

    def get_shap_values(self, features: dict) -> ShapExplanation:
        """
        Сборка SHAP-разложения для одного объекта.

        Общая часть для всех моделей:
        - enrich сырых признаков → DataFrame.
        - Вызов _compute_shap (специфика модели).
        - Сборка ShapExplanation с сортировкой по |value|.
        """
        enriched = FeatureEnricher.enrich(features)
        sample = pd.DataFrame([enriched])

        base_value, contributions = self._compute_shap(sample)
        prediction = self.predict(features)

        sorted_items = sorted(
            contributions.items(),
            key=lambda kv: abs(kv[1]),
            reverse=True,
        )
        shap_values = [
            ShapValue(feature_name=name, value=value)
            for name, value in sorted_items
        ]

        return ShapExplanation(
            base_value=base_value,
            shap_values=shap_values,
            prediction=prediction,
        )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self._name}>"
