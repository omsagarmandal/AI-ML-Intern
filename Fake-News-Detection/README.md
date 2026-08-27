# 📰 Fake News Detection

A Machine Learning + NLP system that classifies news articles as Fake or Real, built as a 3-month capstone project during my AI/ML internship.

## Overview
- **Type:** Text Classification (NLP)
- **Dataset:** Kaggle Fake and Real News Dataset (~44K articles)
- **Best Model:** Linear SVM (TF-IDF, 1-2 grams) — 98.94% test accuracy
- **Deployment:** FastAPI backend + Streamlit frontend, containerized with Docker

## Pipeline
1. Data cleaning & deduplication (209 duplicates removed)
2. Leakage detection & removal (Reuters source tags, subject column)
3. Text preprocessing (lowercase, stopword removal, lemmatization)
4. TF-IDF vectorization (5000 features, unigrams + bigrams)
5. Model comparison (Logistic Regression, Naive Bayes, SVM, Random Forest, XGBoost, MLP)
6. Hyperparameter tuning (GridSearchCV)
7. Explainability (LIME)
8. Generalization testing (temporal train/test split — 98.02% accuracy)
9. REST API (FastAPI) + Streamlit UI
10. Test suite (pytest) + Docker containerization

## Project Structure

Fake-News-Detection/
├── api/ # FastAPI backend
├── app/ # Streamlit frontend
├── data/ # Raw datasets
├── models/ # Saved model + vectorizer
├── notebooks/ # EDA + training notebook
├── tests/ # pytest test suite
├── images/ # Charts/plots
├── Dockerfile
└── README.md


## Running Locally
```bash
# Backend
cd api
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Frontend (new terminal)
cd app
pip install streamlit requests
streamlit run streamlit_app.py
```

## Running with Docker
```bash
docker build -t fake-news-api .
docker run -p 8000:8000 fake-news-api
```

## Known Limitations
- Trained on 2015–2018 US political/world news (agency-report style)
- On out-of-domain text (different topic, region, or era), predictions show lower confidence since TF-IDF vocabulary is domain-specific
- On in-domain text, the model is highly confident and accurate (98%+ accuracy, confirmed via temporal split, not just random split)
- A production system would need periodic retraining on fresh, diverse data

## Results Summary
| Model | Accuracy |
|---|---|
| Logistic Regression | 98.19% |
| Naive Bayes | 93.98% |
| **Linear SVM** | **98.94%** |
| Random Forest | 98.60% |
| XGBoost | 98.81% |
| MLP (Neural Net) | 98.76% |

---

**Om Sagar Mandal**
<<<<<<< HEAD
AI/ML Intern
=======
AI/ML Intern
>>>>>>> edaa2c4087b11ea5e5609c2936c5faa2af6ef983
