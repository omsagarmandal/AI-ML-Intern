from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="Titanic Survival Prediction API")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "..", "models", "titanic_svm_model.pkl"))
encoders = joblib.load(os.path.join(BASE_DIR, "..", "models", "titanic_encoders.pkl"))
age_medians = joblib.load(os.path.join(BASE_DIR, "..", "models", "age_medians.pkl"))

class PassengerInput(BaseModel):
    Pclass: int
    Sex: str
    Age: float | None = None
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str
    Title: str

@app.get("/")
def root():
    return {"message": "Titanic Survival Prediction API is running"}

@app.post("/predict")
def predict(input: PassengerInput):
    age = input.Age
    if age is None:
        age = age_medians.get(input.Title, age_medians.median())

    family_size = input.SibSp + input.Parch + 1
    is_alone = 1 if family_size == 1 else 0

    row = pd.DataFrame([{
        "Pclass": input.Pclass,
        "Sex": encoders["Sex"].transform([input.Sex])[0],
        "Age": age,
        "Fare": input.Fare,
        "Embarked": encoders["Embarked"].transform([input.Embarked])[0],
        "FamilySize": family_size,
        "IsAlone": is_alone,
        "Title": encoders["Title"].transform([input.Title])[0],
        "Deck": encoders["Deck"].transform(["U"])[0]
    }])

    pred = model.predict(row)[0]
    proba = model.predict_proba(row)[0][1]

    label = "Survived" if pred == 1 else "Did Not Survive"
    return {"prediction": label, "survival_probability": float(proba)}