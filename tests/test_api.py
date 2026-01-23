from fastapi.testclient import TestClient
from src.api.main import app
import numpy as np

client = TestClient(app)
NUM_FEATURES = 14
SEQ_LEN = 30

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "API is running."}

def test_predict_success():
    dummy_payload = {
        "data": np.random.randn(SEQ_LEN + 5, NUM_FEATURES).tolist()
    }
    response = client.post("/predict", json=dummy_payload)
    assert response.status_code == 200
    response_json = response.json()
    assert "rul_prediction" in response_json
    assert isinstance(response_json["rul_prediction"], float)

def test_predict_insufficient_data():
    '''Test the error handling for data that is too short (less than SEQ_LEN)'''
    dummy_payload = {
        'data': np.random.rand(SEQ_LEN - 5, NUM_FEATURES).tolist()
    }
    response = client.post("/predict", json=dummy_payload)
    assert response.status_code == 400
    assert f"must have at least {SEQ_LEN}" in response.json()["detail"]

def test_predict_malformed_data():
    # Test the error handling for incorrect number of features
    dummy_payload = {
        "data": np.random.rand(SEQ_LEN, NUM_FEATURES + 1).tolist() # Only 15 features instead of 17
    }
    response = client.post("/predict", json=dummy_payload)
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "columns passed" in detail