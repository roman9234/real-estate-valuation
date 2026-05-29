"""
Адаптер для feature_catalog.json.

feature_catalog.json — это сырой артефакт этапа EDA: статистика
по колонкам датасета (min, max, value_counts и т.п.). У него нет
полей для UI: нет человекочитаемых подписей, нет типов в формате,
понятном фронту, нет указания «производный/нет».

Этот адаптер обогащает каталог UI-метаданными и приводит его к
формату, который ожидает фронт.

Альтернатива править feature_catalog.json
вручную и добавлять туда UI-поля. Это смешало бы артефакт ML-этапа
с UI-конфигом — изменение интерфейса требовало бы переобучения
модели или, как минимум, перевыпуска артефакта.
"""

import json
from functools import lru_cache
from typing import Any

from app.core.config import settings
from app.core.exceptions import FeaturesMetaNotFoundError

# UI-метаданные, которые добавляются поверх сырого каталога.
_UI_META: dict[str, dict[str, Any]] = {
    "area": {
        "label": "Площадь, м²",
        "input_type": "number",
        "step": 0.1,
        "unit": "м²",
        "derived": False,
    },
    "rooms": {
        "label": "Комнат",
        "input_type": "number",
        "step": 1,
        "derived": False,
    },
    "floor": {
        "label": "Этаж",
        "input_type": "number",
        "step": 1,
        "derived": False,
    },
    "total_floors": {
        "label": "Этажей в доме",
        "input_type": "number",
        "step": 1,
        "derived": False,
    },
    "minutes_to_metro": {
        "label": "Минут до метро",
        "input_type": "number",
        "step": 1,
        "unit": "мин",
        "derived": False,
    },
    "is_studio": {
        "label": "Студия",
        "input_type": "checkbox",
        "derived": False,
    },
    "metro_station": {
        "label": "Станция метро",
        "input_type": "select",
        "derived": False,
    },
    "renovation": {
        "label": "Тип ремонта",
        "input_type": "select",
        "derived": False,
    },
    # Производные — не показываются в форме, но известны фронту
    # для отображения в Feature Importance / SHAP.
    "floor_ratio": {"label": "Этаж / всего этажей", "derived": True},
    "is_first_floor": {"label": "Первый этаж", "derived": True},
    "is_last_floor": {"label": "Последний этаж", "derived": True},
    "log_area": {"label": "log(площадь+1)", "derived": True},
}

@lru_cache(maxsize=1)
def get_features_meta() -> dict:
    """
    Загружает и возвращает обогащённый каталог признаков.

    @lru_cache: файл читается один раз за всё время жизни приложения.
    Если изменить feature_catalog.json — нужен перезапуск backend.
    """
    path = settings.FEATURES_META_PATH
    if not path.exists():
        raise FeaturesMetaNotFoundError(
            f"Файл {path} не найден",
            details={"path": str(path)},
        )

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # categorical_features и numeric_features — это dict[имя -> объект].
    # Итерируемся через .items(), name попадает как ключ.
    raw_numeric = raw.get("numeric_features", {})
    raw_categorical = raw.get("categorical_features", {})

    numeric = []
    for name, feat in raw_numeric.items():
        ui = _UI_META.get(name, {})
        if ui.get("derived"):
            continue
        numeric.append({
            "name": name,
            **feat,           # min, max, median, dtype и т.п.
            **ui,             # label, input_type, unit, step
        })

    categorical = []
    for name, feat in raw_categorical.items():
        ui = _UI_META.get(name, {})

        # Сортируем values по value_counts (популярные сверху).
        # Если value_counts нет — оставляем values как есть.
        values = feat.get("values", [])
        vc = feat.get("value_counts")
        if isinstance(vc, dict) and vc:
            values = [v for v, _ in sorted(vc.items(), key=lambda kv: kv[1], reverse=True)]

        categorical.append({
            "name": name,
            "_unique": feat.get("n_unique"),
            "values": values,
            **ui,             # label, input_type
        })

    return {
        "numeric": numeric,
        "categorical": categorical,
        "target": raw.get("target", {}),
    }

