from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

app = FastAPI(title="Sentiment Analysis API")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

model = joblib.load('/models/sentiment_model.pkl')
tfidf = joblib.load('/models/tfidf_vectorizer.pkl')

class ReviewRequest(BaseModel):
    text: str

def clean_text(text):
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

@app.get("/")
def root():
    return {"status": "Sentiment Analysis API is running"}

@app.post("/predict")
def predict(request: ReviewRequest):
    cleaned = clean_text(request.text)
    lemmatized = ' '.join([lemmatizer.lemmatize(w) for w in cleaned.split()])
    vec = tfidf.transform([lemmatized])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    return {
        "sentiment": "Positive" if pred == 1 else "Negative",
        "confidence": round(float(max(prob)) * 100, 2)
    }