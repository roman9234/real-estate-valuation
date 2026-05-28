"""
Сервис интерпретации (Feature Importance + SHAP).

"""
# TODO исправить заглушки
from backend.app.core.exceptions import ModelNotFoundError
from backend.app.ml.base import BaseMLModel
from backend.app.schemas.response import (
    FeatureImportanceItem,
    FeatureImportanceResponse,
    ShapResponse,
    ShapValueItem,
)

def get_feature_importance(
    model_name: str,
    models: dict[str, BaseMLModel],
) -> FeatureImportanceResponse:
    """
    Возвращает топ-N признаков по важности для указанной модели.

    список заглушек в BaseMLModel
    - CatBoost: model.get_feature_importance()
    - XGBoost: model.feature_importances_
    - RandomForest: model.feature_importances_
    - Linear: |coef_| (после стандартизации)
    """
    if model_name not in models:
        raise ModelNotFoundError(
            f"Модель '{model_name}' не загружена",
            details={"available": list(models.keys())},
        )

    raw = models[model_name].get_feature_importance()
    return FeatureImportanceResponse(
        model_name=model_name,
        features=[FeatureImportanceItem(feature_name=f.feature_name, importance=f.importance) for f in raw ]
    )

def get_shap_explanation(
    model_name: str,
    features: dict,
    models: dict[str, BaseMLModel],
) -> ShapResponse:
    """
    SHAP-разложение для конкретного объекта.

    # TODO нужный explainer:
    - CatBoost: TreeExplainer (нативный)
    - XGBoost: TreeExplainer
    - RandomForest: TreeExplainer
    - Linear: LinearExplainer (требует преобразованые признаки!)
    """
    if model_name not in models:
        raise ModelNotFoundError(f"Модель '{model_name}' не загружена")

    model = models[model_name]
    explanation = model.get_shap_values(features)
    prediction = model.predict(features)

    return ShapResponse(
        model_name=model_name,
        base_value=explanation.base_value,
        prediction=prediction,
        shap_values=[
            ShapValueItem(feature_name=v.feature_name, value=v.value)
            for v in explanation.shap_values
        ],
    )
