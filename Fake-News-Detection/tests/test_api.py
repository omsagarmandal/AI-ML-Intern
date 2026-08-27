from fastapi.testclient import TestClient
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "api"))
from main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200

def test_predict_real():
    response = client.post("/predict", json={"text": "WASHINGTON (Reuters) - The Senate voted on Tuesday to advance the bill."})
    assert response.status_code == 200
    assert "prediction" in response.json()

def test_predict_fake():
    response = client.post("/predict", json={"text": "SHOCKING secret government coverup mainstream media won't tell you!!!"})
    assert response.status_code == 200
    assert response.json()["prediction"] in ["Fake", "Real"]

def test_predict_empty():
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 200