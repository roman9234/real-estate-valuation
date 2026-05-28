"""
Сервис предсказания цены.

Слой между routes и ml/. Отвечает за:
- Прогон признаков через все 4 модели.
- Подсчёт доверительного интервала (CI).
- Подсчёт цены за квадратный метр.
- Преобразование в DTO ответа.
"""

from backend.app.core.exceptions import ModelNotLoadedError, PredictionError
from backend.app.ml.base import BaseMLModel
from backend.app.schemas.response import (
    MetricsResponse,
    PredictionItem,
    PredictionResponse,
)

def _confidence_interval(price: float, mape: float) -> tuple[float, float]:
    """
    Доверительный интервал на основе MAPE модели.

    Альтернатива:
    - Для бустингов: квантильная регрессия; обучить отдельные модели на квантили 0.05 и 0.95)
    - Для линейной: использовать дисперсию остатков
    однако MAPE-интервал также достаточен и легко объясним
    """
    margin = price * (mape / 100.0)
    return price - margin, price + margin

def predict_all(
    features: dict,
    models: dict[str, BaseMLModel],
) -> PredictionResponse:
    """
    Прогоняет признаки через все загруженные модели.

    Если хотя бы одна модель не справилась — её результат
    отсутствует в ответе, остальные продолжают работать.
    Это лучше, чем падать целиком: пользователь увидит хотя бы
    часть результатов.
    """
    if not models:
        raise ModelNotLoadedError("Ни одна модель не загружена")

    items: list[PredictionItem] = []
    errors: list[str] = []

    for name, model in models.items():
        try:
            price = model.predict(features)
            ci_lower, ci_upper = _confidence_interval(price, model.metrics.MAPE)
            area = float(features["area"])
            price_per_sqm = price / area if area > 0 else 0.0

            items.append(
                PredictionItem(
                    model_name=name,
                    price=price,
                    ci_lower=ci_lower,
                    ci_upper=ci_upper,
                    price_per_sqm=price_per_sqm,
                    metrics=MetricsResponse(**model.metrics.__dict__),
                )
            )
        except Exception as e:
            # Логируем, но не прерываем остальные модели.
            errors.append(f"{name}: {e}")
            print(f"[ERROR] Предсказание {name} провалилось: {e}")

    if not items:
        # Все модели упали — это уже ошибка для клиента
        raise PredictionError(
            "Ни одна модель не смогла сделать предсказание",
            details={"errors": errors},
        )

    return PredictionResponse(predictions=items)
