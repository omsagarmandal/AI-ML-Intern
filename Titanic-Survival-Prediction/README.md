# 🚢 Titanic Survival Prediction

A Machine Learning project that predicts whether a passenger would have survived the Titanic disaster, based on features like class, sex, age, and family size. Built as a capstone-style project during my AI/ML internship.

## Overview
- **Type:** Classification (Survived / Did Not Survive)
- **Dataset:** Kaggle Titanic dataset (891 train, 418 test)
- **Best Model:** SVM (RBF kernel, tuned) — 83.16% cross-validated accuracy
- **Deployment:** FastAPI backend + Streamlit frontend, containerized with Docker

## Pipeline
1. EDA (survival rate by class, sex, age, family size)
2. Feature engineering — `FamilySize`, `IsAlone`, `Title` (extracted from Name), `Deck` (from Cabin)
3. Missing value handling — Age imputed via Title-wise median, Embarked via mode
4. Encoding categorical features (Sex, Embarked, Title, Deck)
5. Model comparison (Logistic Regression, KNN, SVM, Random Forest, XGBoost)
6. Cross-validation + hyperparameter tuning (GridSearchCV)
7. Explainability (permutation feature importance)
8. REST API (FastAPI) + Streamlit UI
9. Test suite (pytest) + Docker containerization

## Key Insight
**"Women and children first"** is clearly reflected in the data:
- Female survival rate: **74.2%** vs Male: **18.9%**
- 1st class survival rate: **63.0%** vs 3rd class: **24.2%**
- `Sex` is by far the strongest predictor (permutation importance ~0.25, next closest feature ~0.04)

## App Features
- **🧍 Single Passenger tab** — enter passenger details and get an instant survival prediction with a probability gauge
- **📂 Batch CSV tab** — upload a CSV of multiple passengers and download predictions for all of them at once
- **🕒 History tab** — view the last 50 predictions made in the app
- Custom dark, animated UI (gradient background, glowing buttons, live probability gauge)

## Project Structure
```
Titanic-Survival-Prediction/
├── api/            # FastAPI backend
├── app/            # Streamlit frontend
├── data/           # Raw datasets
├── models/         # Saved model, encoders, age medians
├── notebooks/      # EDA + training notebook
├── tests/          # pytest test suite
├── images/         # Charts/plots
├── Dockerfile
└── README.md
```

## Running Locally
```bash
# Backend
cd api
pip install -r requirements.txt
uvicorn main:app --reload --port 8001

# Frontend (new terminal)
cd app
pip install streamlit requests plotly pandas
streamlit run streamlit_app.py
```

## Running with Docker
```bash
docker build -t titanic-api .
docker run -p 8001:8001 titanic-api
```

## Known Limitations
- Trained on a historical dataset from a single event (1912) — not applicable beyond this specific context
- Small dataset (891 rows) means the model can be sensitive to how missing values are imputed
- `Deck` feature has very high missingness (77%) in the original data, limiting its real predictive value despite being engineered
- A production system would need a much larger, more diverse dataset to generalize to other scenarios

## Results Summary
| Model | Cross-Validated Accuracy |
|---|---|
| Logistic Regression | 79.46% |
| KNN | 81.37% |
| **SVM (tuned)** | **83.16%** |
| Random Forest | 80.81% |
| XGBoost | 82.16% |

---

**Om Sagar Mandal**
AI/ML Intern