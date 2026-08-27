from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

app = FastAPI(title="Fake News Detection API")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "..", "models", "fake_news_svm_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "..", "models", "tfidf_vectorizer.pkl"))

class NewsInput(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "Fake News Detection API is running"}

@app.post("/predict")
def predict(input: NewsInput):
    cleaned = input.text
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    score = model.decision_function(vec)[0]
    label = "Real" if pred == 1 else "Fake"
    return {"prediction": label, "confidence_score": float(score)}