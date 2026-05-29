import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from backend.app.main import app
from backend.app.ml.constants import MODEL_CATBOOST
from backend.app.ml.registry import load_models

@pytest_asyncio.fixture(scope="module")
async def client():
    app.state.models = load_models()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

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

class TestHealth:
    async def test_health_returns_200(self, client):
        r = await client.get("/api/v1/health")
        assert r.status_code == 200

    async def test_health_returns_json(self, client):
        r = await client.get("/api/v1/health")
        assert r.json() == {"status": "ok"}

class TestFeatures:
    async def test_features_returns_200(self, client):
        r = await client.get("/api/v1/features")
        assert r.status_code == 200

    async def test_features_has_numeric_and_categorical(self, client):
        data = (await client.get("/api/v1/features")).json()
        assert {"numeric", "categorical", "target"} <= data.keys()

    async def test_numeric_has_label_and_input_type(self, client):
        data = (await client.get("/api/v1/features")).json()
        for feat in data["numeric"]:
            assert "label" in feat
            assert "input_type" in feat

    async def test_categorical_has_label_and_values(self, client):
        data = (await client.get("/api/v1/features")).json()
        for feat in data["categorical"]:
            assert "label" in feat
            assert "values" in feat
            assert len(feat["values"]) > 0

class TestModels:
    async def test_models_returns_200(self, client):
        r = await client.get("/api/v1/models")
        assert r.status_code == 200

    async def test_models_list_not_empty(self, client):
        models_list = (await client.get("/api/v1/models")).json()["models"]
        assert len(models_list) > 0

    async def test_each_model_has_metrics(self, client):
        models_list = (await client.get("/api/v1/models")).json()["models"]
        for m in models_list:
            metrics = m["metrics"]
            assert {"MAE", "RMSE", "MAPE", "R2"} <= metrics.keys()
            assert metrics["MAPE"] > 0

class TestPredict:
    async def test_predict_returns_200(self, client):
        r = await client.post("/api/v1/predict", json=SAMPLE_REQUEST)
        assert r.status_code == 200

    async def test_predict_returns_all_models(self, client):
        models_count = len((await client.get("/api/v1/models")).json()["models"])
        predictions = (await client.post("/api/v1/predict", json=SAMPLE_REQUEST)).json()["predictions"]
        assert len(predictions) == models_count

    async def test_each_prediction_has_required_fields(self, client):
        predictions = (await client.post("/api/v1/predict", json=SAMPLE_REQUEST)).json()["predictions"]
        for p in predictions:
            assert {"model_name", "price", "ci_lower", "ci_upper", "price_per_sqm", "metrics"} <= p.keys()
            assert p["price"] > 0
            assert p["ci_lower"] < p["ci_upper"]

    @pytest.mark.parametrizeparametrize(
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
    async def test_predict_422_on_invalid_input(self, client, bad_field, bad_value):
        bad = {**SAMPLE_REQUEST, bad_field: bad_value}
        r = await client.post("/api/v1/predict", json=bad)
        assert r.status_code == 422

    async def test_predict_floor_above_total_floors_422(self, client):
        bad = {**SAMPLE_REQUEST, "floor": 20, "total_floors": 5}
        r = await client.post("/api/v1/predict", json=bad)
        assert r.status_code == 422

class TestFeatureImportance:
    async def test_returns_200_for_valid_model(self, client):
        r = await client.get(f"/api/v1/feature-importance/{MODEL_CATBOOST}")
        assert r.status_code == 200

    async def test_404_for_unknown_model(self, client):
        r = await client.get("/api/v1/feature-importance/NonExistentModel")
        assert r.status_code == 404

    async def test_body_has_correct_structure(self, client):
        data = (await client.get(f"/api/v1/feature-importance/{MODEL_CATBOOST}")).json()
        assert data["model_name"] == MODEL_CATBOOST
        assert "features" in data
        assert len(data["features"]) > 0

class TestExplain:
    async def test_returns_200_for_valid_model(self, client):
        r = await client.post(f"/api/v1/explain/{MODEL_CATBOOST}", json=SAMPLE_REQUEST)
        assert r.status_code == 200

    async def test_404_for_unknown_model(self, client):
        r = await client.post("/api/v1/explain/NonExistentModel", json=SAMPLE_REQUEST)
        assert r.status_code == 404

    async def test_body_has_correct_structure(self, client):
        data = (await client.post(f"/api/v1/explain/{MODEL_CATBOOST}", json=SAMPLE_REQUEST)).json()
        assert data["model_name"] == MODEL_CATBOOST
        assert {"base_value", "prediction", "shap_values"} <= data.keys()
        for item in data["shap_values"]:
            assert "feature_name" in item
            assert "value" in item

class TestSensitivity:
    async def test_returns_200_for_numeric_feature(self, client):
        r = await client.post(f"/api/v1/sensitivity/{MODEL_CATBOOST}/area", json=SAMPLE_REQUEST)
        assert r.status_code == 200

    async def test_returns_200_for_categorical_feature(self, client):
        r = await client.post(f"/api/v1/sensitivity/{MODEL_CATBOOST}/metro_station", json=SAMPLE_REQUEST)
        assert r.status_code == 200

    async def test_404_for_unknown_model(self, client):
        r = await client.post("/api/v1/sensitivity/NonExistentModel/area", json=SAMPLE_REQUEST)
        assert r.status_code == 404

    async def test_404_for_unknown_feature(self, client):
        r = await client.post(f"/api/v1/sensitivity/{MODEL_CATBOOST}/unknown_feature_xyz", json=SAMPLE_REQUEST)
        assert r.status_code == 404

    async def test_numeric_grid_has_points(self, client):
        data = (await client.post(f"/api/v1/sensitivity/{MODEL_CATBOOST}/area", json=SAMPLE_REQUEST)).json()
        assert len(data["points"]) > 1
        for pt in data["points"]:
            assert {"value", "price"} <= pt.keys()
            assert isinstance(pt["value"], (int, float))

    async def test_categorical_grid_has_points(self, client):
        data = (await client.post(f"/api/v1/sensitivity/{MODEL_CATBOOST}/metro_station", json=SAMPLE_REQUEST)).json()
        assert len(data["points"]) > 0
        assert data["feature_name"] == "metro_station"
