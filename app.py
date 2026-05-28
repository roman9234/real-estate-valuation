from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

from inference import PREDICTORS, predict

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app = FastAPI(title="Apartment Price Estimator — MVP")

METRO_STATIONS = [
    "Коломенская", "Профсоюзная", "Академическая", "Беляево",
    "Тёплый Стан", "Юго-Западная", "Университет", "Парк культуры",
    "Тверская", "Чистые пруды", "Сокольники", "ВДНХ",
    "Алтуфьево", "Митино", "Строгино", "Кунцевская",
]
RENOVATION_TYPES = ["Without", "Cosmetic", "Euro", "Designer"]
MODEL_NAMES = list(PREDICTORS.keys())

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {
        "models": MODEL_NAMES,
        "metro_stations": sorted(METRO_STATIONS),
        "renovation_types": RENOVATION_TYPES,
        "result": None,
        "form": None,
        "error": None,
    })

@app.post("/", response_class=HTMLResponse)
async def estimate(
    request: Request,
    model_name: str = Form(...),
    area: float = Form(...),
    rooms: float = Form(...),
    floor: float = Form(...),
    total_floors: int = Form(...),
    minutes_to_metro: float = Form(...),
    is_studio: int = Form(0),
    metro_station: str = Form(...),
    renovation: str = Form(...),
):
    features = {
        "area": area,
        "rooms": rooms,
        "floor": floor,
        "total_floors": total_floors,
        "minutes_to_metro": minutes_to_metro,
        "is_studio": is_studio,
        "metro_station": metro_station,
        "renovation": renovation,
    }

    result = None
    error = None
    try:
        price = predict(model_name, features)
        result = {
            "model_name": model_name,
            "price": price,
            "price_formatted": f"{price:,.0f}".replace(",", " "),
            "ci_lower": f"{price * 0.90:,.0f}".replace(",", " "),
            "ci_upper": f"{price * 1.10:,.0f}".replace(",", " "),
            "price_per_sqm": f"{price / area:,.0f}".replace(",", " "),
        }
    except FileNotFoundError:
        error = f"Модель «{model_name}» не найдена в каталоге ml_models/."
    except Exception as e:
        error = f"Ошибка при инференсе: {type(e).__name__}: {e}"

    return templates.TemplateResponse(request, "index.html", {
        "models": MODEL_NAMES,
        "metro_stations": sorted(METRO_STATIONS),
        "renovation_types": RENOVATION_TYPES,
        "result": result,
        "form": features | {"model_name": model_name},
        "error": error,
    })
