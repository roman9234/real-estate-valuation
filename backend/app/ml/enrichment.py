"""
FeatureEnricher: вычисляет производные признаки из сырых полей формы.

Эти поля явно не вводятся пользователем — они получаются
автоматически из area, floor, total_floors.
"""

import numpy as np

class FeatureEnricher:
    """
    Обогащает словарь признаков производными полями.

    """

    @staticmethod
    def enrich(features: dict) -> dict:
        """
        Возвращает новый словарь с добавленными производными признаками.

        Исходный словарь не мутируется — возвращается копия.
        """
        result = dict(features)

        floor = float(features["floor"])
        total_floors = float(features["total_floors"])
        area = float(features["area"])

        result["floor_ratio"] = floor / total_floors if total_floors > 0 else 0.0
        result["is_first_floor"] = int(floor == 1)
        result["is_last_floor"] = int(floor == total_floors)
        result["log_area"] = float(np.log1p(area))

        return result
