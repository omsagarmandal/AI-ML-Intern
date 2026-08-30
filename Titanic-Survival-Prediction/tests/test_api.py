from fastapi.testclient import TestClient
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "api"))
from main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200

def test_predict_female_first_class():
    response = client.post("/predict", json={
        "Pclass": 1, "Sex": "female", "Age": 28, "SibSp": 0, "Parch": 0,
        "Fare": 80, "Embarked": "S", "Title": "Miss"
    })
    assert response.status_code == 200
    assert "prediction" in response.json()

def test_predict_male_third_class():
    response = client.post("/predict", json={
        "Pclass": 3, "Sex": "male", "Age": 30, "SibSp": 0, "Parch": 0,
        "Fare": 8, "Embarked": "S", "Title": "Mr"
    })
    assert response.status_code == 200
    assert response.json()["prediction"] in ["Survived", "Did Not Survive"]

def test_predict_missing_age():
    response = client.post("/predict", json={
        "Pclass": 2, "Sex": "male", "SibSp": 0, "Parch": 0,
        "Fare": 15, "Embarked": "Q", "Title": "Mr"
    })
    assert response.status_code == 200