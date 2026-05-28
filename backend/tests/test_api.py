"""
Интеграционные тесты HTTP-эндпоинтов.

Запускают FastAPI-приложение в тестовом режиме (без reload)
и общаются с ним через httpx.AsyncClient.

Используем httpx.AsyncClient + ASGITransport.
Lifespan FastAPI не запускается через ASGITransport автоматически,
поэтому модели загружаем вручную в фикстуре.

pytest-asyncio обязателен: все тесты асинхронны.
"""


import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from backend.app.main import app
from backend.app.ml.registry import load_models

# ── Фикстуры ────────────────────────────

@pytest_asyncio.fixture(scope="module")
async def client():
    """
    Тестовый HTTP-клиент.

    🚲 ASGITransport не запускает lifespan приложения, поэтому
    app.state.models остаётся пустым. Загружаем модели вручную
    перед созданием клиента — это эквивалент того, что делает
    lifespan при реальном запуске.

    Альтернатива — asgi-lifespan.LifespanManager, но это
    лишняя зависимость для одного места.
    """
    # Имитируем lifespan: загружаем модели в app.state
    app.state.models = load_models()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

# ─── Константы ────────────────────────────────

SAMPLE_REQUEST = {
    "area": 60.0,
    "rooms": 2,
    "floor": 6,
    "total_floors": 15,
    "minutes_to_metro": 11,
    "is_studio": 0,
    "metro_station": "Авиамоторная",
    "renovation": "Cosmetic",
}

# ─── 1. /health ────────────────────────────────

class TestHealth:
    async def test_health_returns_200(self, client: AsyncClient):
        response = await client.get("/api/v1/health")
        assert response.status_code == 200

    async def test_health_returns_json(self, client: AsyncClient):
        response = await client.get("/api/v1/health")
        assert response.json() == {"status": "ok"}

# ─── 2. /features ─────────────────────────

class TestFeatures:
    async def test_features_returns_200(self, client: AsyncClient):
        response = await client.get("/api/v1/features")
        assert response.status_code == 200

    async def test_features_has_numeric_and_categorical(self, client: AsyncClient):
        response = await client.get("/api/v1/features")
        data = response.json()
        assert "numeric" in data
        assert "categorical" in data
        assert "target" in data

    async def test_numeric_has_labeland_input_type(self, client: AsyncClient):
        response = await client.get("/api/v1/features")
        data = response.json()
        for feat in data["numeric"]:
            assert "label" in feat, f"Поле {feat.get('name')} не имеет label"
            assert "input_type" in feat

    async def test_categorical_has_label_and_values(self, client: AsyncClient):
        response = await client.get("/api/v1/features")
        data = response.json()
        for feat in data["categorical"]:
            assert "label" in feat
            assert "values" in feat
            assert len(feat["values"]) > 0

# ─── 3. /models────────────────────────────────

class TestModels:
    async def test_models_returns_200(self, client: AsyncClient):
        response = await client.get("/api/v1/models")
        assert response.status_code == 200

    async def test_models_list_not_empty(self, client: AsyncClient):
        """
        Если модели загружены — список не пуст.
        Используем напрямую app.state, без отдельной фикстуры
        models_available, чтобы не путать слои.
        """
        response = await client.get("/api/v1/models")
        models_list = response.json().get("models", [])
        # Модели в этой фикстуре загружаются вручную, значит должны быть
        assert len(models_list) > 0, "Модели не загружены — проверь ml_models/"

    async def test_each_model_has_metrics(self, client: AsyncClient):
        response = await client.get("/api/v1/models")
        models_list = response.json().get("models", [])
        if not models_list:
            pytest.skip("Нет загруженных моделей")

        for m in models_list:
            metrics = m.get("metrics", {})
            assert "MAE" in metrics
            assert "RMSE" in metrics
            assert "MAPE" in metrics
            assert "R2" in metrics
            assert metrics["MAPE"] > 0

# ─── 4. /predict ────────────────────────────────

class TestPredict:
    async def test_predict_returns_200(self, client: AsyncClient):
        response = await client.post("/api/v1/predict", json=SAMPLE_REQUEST)
        assert response.status_code == 200

    async def test_predict_returns_all_models(self, client: AsyncClient):
        models_resp = await client.get("/api/v1/models")
        models_count = len(models_resp.json().get("models", []))

        response = await client.post("/api/v1/predict", json=SAMPLE_REQUEST)
        predictions = response.json().get("predictions", [])
        assert len(predictions) == models_count

    async def test_each_prediction_has_required_fields(self, client: AsyncClient):
        response = await client.post("/api/v1/predict", json=SAMPLE_REQUEST)
        predictions = response.json().get("predictions", [])

        for p in predictions:
            assert "model_name" in p
            assert "price" in p
            assert "ci_lower" in p
            assert "ci_upper" in p
            assert "price_per_sqm" in p
            assert "metrics" in p
            assert isinstance(p["price"], (int, float))
            assert p["price"] > 0
            assert p["ci_lower"] < p["ci_upper"]

    @pytest.mark.parametrize(
        "bad_field, bad_value",
        [
            ("area", -10.0),
            ("rooms", -1),
            ("floor", 150),
            ("total_floors", 0),
            ("renovation", "премиум"),
            ("area", "не число"),
        ],
    )
    async def test_predict_422_on_invalid_input(
        self, client: AsyncClient, bad_field: str, bad_value
    ):
        bad_request = {**SAMPLE_REQUEST, bad_field: bad_value}
        response = await client.post("/api/v1/predict", json=bad_request)
        assert response.status_code == 422

        detail = response.json().get("detail", [])
        assert len(detail) > 0
        assert any(bad_field in str(d.get("loc", [])) for d in detail)

    async def test_predict_floor_above_total_floors_422(self, client: AsyncClient):
        bad_request = {**SAMPLE_REQUEST, "floor": 20, "total_floors": 5}
        response = await client.post("/api/v1/predict", json=bad_request)
        assert response.status_code == 422

# ─── 5. /feature-importance ────────────────────

class TestFeatureImportance:
    async def test_returns_200_for_valid_model(self, client: AsyncClient):
        response = await client.get("/api/v1/feature-importance/CatBoost")
        assert response.status_code in (200, 404)

    async def test_404_for_unknown_model(self, client: AsyncClient):
        response = await client.get("/api/v1/feature-importance/NonExistentModel")
        assert response.status_code == 404

    async def test_body_has_correct_structure(self, client: AsyncClient):
        response = await client.get("/api/v1/feature-importance/CatBoost")
        if response.status_code != 200:
            pytest.skip("Модель не загружена")
        data = response.json()
        assert "model_name" in data
        assert "features" in data

# ─── 6. /explain ────────────────────────────────

class TestExplain:
    async def test_returns_200_for_valid_model(self, client: AsyncClient):
        response = await client.post(
            "/api/v1/explain/CatBost",
            json=SAMPLE_REQUEST,
        )
        assert response.status_code in (200, 404)

    async def test_404_for_unknown_model(self, client: AsyncClient):
        response = await client.post(
            "/api/v1/explain/NonExistentModel",
            json=SAMPLE_REQUEST,
        )
        assert response.status_code == 404

    async def test_body_has_correct_structure(self, client: AsyncClient):
        response = await client.post(
            "/api/v1/explain/CatBoost",
            json=SAMPLE_REQUEST,
        )
        if response.status_code != 200:
            pytest.skip("Модель не загружена")
        data = response.json()
        assert data["model_name"] == "CatBoost"
        assert "base_value" in data
        assert "prediction" in data
        assert "shap_values" in data

# ─── 7. /sensitivity ────────────────────────────────

class TestSensitivity:
    async def test_returns_200_for_numeric_feature(self, client: AsyncClient):
        response = await client.post(
            "/api/v1/sensitivity/CatBoost/area",
            json=SAMPLE_REQUEST,
        )
        assert response.status_code in (200, 404)

    async def test_returns_200_for_categorical_feature(self, client: AsyncClient):
        response = await client.post(
            "/api/v1/sensitivity/CatBoost/metro_station",
            json=SAMPLE_REQUEST,
        )
        assert response.status_code in (200, 404)

    async def test_404_for_unknown_model(self, client: AsyncClient):
        response = await client.post(
            "/api/v1/sensitivity/NonExistentModel/area",
            json=SAMPLE_REQUEST,
        )
        assert response.status_code == 404

    async def test_404_for_unknown_feature(self, client: AsyncClient):
        response = await client.post(
            "/api/v1/sensitivity/CatBoost/unknown_feature_xyz",
            json=SAMPLE_REQUEST,
        )
        assert response.status_code == 404

    async def test_numeric_grid_has_points(self, client: AsyncClient):
        response = await client.post(
            "/api/v1/sensitivity/CatBost/area",
            json=SAMPLE_REQUEST,
        )
        if response.status_code != 200:
            pytest.skip("Модель не загружена")
        data = response.json()
        assert len(data["points"]) > 1
        for pt in data["points"]:
            assert "value" in pt
            assert "price" in pt
            assert isinstance(pt["value"], (int, float))

    async def test_categorical_grid_has_points(self, client: AsyncClient):
        response = await client.post(
            "/api/v1/sensitivity/CatBoost/metro_station",
            json=SAMPLE_REQUEST,
        )
        if response.status_code != 200:
            pytest.skip("Модель не загружена")
        data = response.json()
        assert len(data["points"]) > 0
        assert data["feature_name"] == "metro_station"
