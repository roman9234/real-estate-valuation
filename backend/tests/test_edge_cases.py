"""
Тесты граничных случаев для всех моделей.

Проверяют, что модели не падают на необычных, но технически
валидных входных данных. Цели предсказания при этом могут быть
бессмысленными — это нормально,ловить такое должна валидация
на уровне Pydantic (этап routes), а не на уровне ML.
"""

import pytest

from tests.conftest import ALL_MODEL_NAMES

@pytest.mark.parametrize("model_name", ALL_MODEL_NAMES)
class TestExtremeValues:
    """Экстремальные, но допустимые значения числовых полей."""

    def test_minimal_apartment(self, model_name, loaded_models):
        """Самая маленькая возможная квартира из обучающего датасета."""
        if model_name not in loaded_models:
            pytest.skip(f"Модель {model_name} не загружена")

        features = _baseline()
        features["area"] = 11.0           # минимум из feature_catalog
        features["rooms"] = 0             # студия без перегородок
        features["is_studio"] = 1
        features["floor"] = 1
        features["total_floors"] = 1

        # Должна не упасть, цену не проверяем строго
        result = loaded_models[model_name].predict(features)
        assert result > 0

    def test_maximal_apartment(self, model_name, loaded_models):
        """Максимально большая квартира."""
        if model_name not in loaded_models:
            pytest.skip(f"Модель {model_name} не загружена")

        features = _baseline()
        features["area"] = 396.5
        features["rooms"] = 12
        features["floor"] = 50
        features["total_floors"] = 92

        result = loaded_models[model_name].predict(features)
        assert result > 0

    def test_high_floor_in_low_building(self, model_name, loaded_models):
        """
        Невозможная ситуация: floor=15, total_floors=5.
        floor_ratio будет = 3.0 — вне обучающего диапазона [0, 1].

        Модель не должна падать, но цена будет мусорной.
        Перехватывать это должна валидация Pydantic (floor <= total_floors),
        не ML.
        """
        if model_name not in loaded_models:
            pytest.skip(f"Модель {model_name} не загружена")

        features = _baseline()
        features["floor"] = 15
        features["total_floors"] = 5

        # Главное — не упасть
        result = loaded_models[model_name].predict(features)
        assert isinstance(result, float)

    def test_very_far_metro(self, model_name, loaded_models):
        """До метро 60 минут — максимум из датасета."""
        if model_name not in loaded_models:
            pytest.skip(f"Модель {model_name} не загружена")

        features = _baseline()
        features["minutes_to_metro"] = 60

        result = loaded_models[model_name].predict(features)
        assert result > 0

@pytest.mark.parametrize("model_name", ALL_MODEL_NAMES)
class TestCategoricalEdge:
    """Граничные случаи для категориальных признаков."""

    def test_all_renovation_types(self, model_name, loaded_models):
        """Каждый из 4 типов ремонта должен обабатываться."""
        if model_name not in loaded_models:
            pytest.skip(f"Модель {model_name} не загружена")

        renovations = [
            "Cosmetic",
            "Designer",
            "European-style renovation",
            "Without renovation",
        ]

        results = []
        for renov in renovations:
            features = _baseline()
            features["renovation"] = renov
            results.append(loaded_models[model_name].predict(features))

        # Все 4 значения дали валидные предсказания
        assert all(r > 0 for r in results)
        assert len(set(results)) > 1, (
            f"{model_name}: все 4 типа ремонта дали одинаковую цену"
        )

    def test_unknown_metro_station_handling(self, model_name, loaded_models):
        """
        Несуществующая станция метро.

        Поведение зависит от модели:
        - CatBoost: обработает как "редкое значение", не упадёт
        - XGBoost/RF: OneHotEncoder с handle_unknown='ignore'
          вернёт нулевой вектор — не упадёт
        - Linear: аналогично

        Если хотя бы одна из моделей обучалась с handle_unknown='error',
        этот тест покажет, что нужна нормализация на уровне API.
        """
        if model_name not in loaded_models:
            pytest.skip(f"Модель {model_name} не загружена")

        features = _baseline()
        features["metro_station"] = "Несуществующая станция XYZ"

        try:
            result = loaded_models[model_name].predict(features)
            # Если не упало — отлично
            assert isinstance(result, float)
        except (ValueError, KeyError) as e:
            # Зафиксируем как известное ограничение модели.
            # На уровне API это нужно перехватить и вернуть 422.
            pytest.xfail(
                f"{model_name} не обрабатывает неизвестную станцию: {e}. "
                f"Нужна валидация на уровне Pydantic."
            )

# class TestPopularStations:
#     """
#     Дополнительный smoke-тест: 5 самых популярных станций
#     из value_counts должны давать предсказания у всех моделей.
#     """
#
#     POPULAR_STATIONS = [
#         "ЗИЛ",
#         "Аминьевская",
#         "Минская",
#         "Коммунарка",
#         "Улица 1905 года",
#     ]
#
#     pytest.mark.parametrize("model_name", ALL_MODEL_NAMES)
#     pytest.mark.parametrize("station", POPULAR_STATIONS)
#     def test_popular_station_predicts(
#         self, model_name, station, loaded_models
#     ):
#         if model_name not in loaded_models:
#             pytest.skip(f"Модель {model_name} не загружена")
#
#         features = _baseline()
#         features["metro_station"] = station
#
#         result = loaded_models[model_name].predict(features)
#         assert result > 0

# ─── helpers ────────────────────────────────

def _baseline() -> dict:
    """Базовый словарь признаков для модификации в тестах."""
    return {
        "area": 60.0,
        "rooms": 2,
        "floor": 6,
        "total_floors": 15,
        "minutes_to_metro": 11,
        "is_studio": 0,
        "metro_station": "Авиамоторная",
        "renovation": "Cosmetic",
    }
