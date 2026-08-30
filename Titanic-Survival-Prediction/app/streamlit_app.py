import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import json
import os
from datetime import datetime

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap');
* { font-family: 'Poppins', sans-serif; }

.stApp {
    background: linear-gradient(-45deg, #0a1a2f, #12283f, #0d1f33, #1a2b45);
    background-size: 400% 400%;
    animation: gradientShift 15s ease infinite;
}
@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

h1 { color: #ffffff !important; text-shadow: 0 0 20px rgba(80, 160, 255, 0.6); }
p, label, .stMarkdown { color: #dceaff !important; }

.stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
    background: rgba(30, 40, 60, 0.6) !important;
    border: 1px solid rgba(80, 160, 255, 0.3) !important;
    border-radius: 10px !important;
    color: #ffffff !important;
}

.stButton button {
    background: linear-gradient(135deg, #1e3a8a, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 30px !important;
    padding: 10px 28px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 20px rgba(37, 99, 235, 0.4);
    transition: all 0.3s ease !important;
}
.stButton button:hover {
    transform: translateY(-3px) scale(1.03);
    box-shadow: 0 8px 30px rgba(37, 99, 235, 0.6);
}

div[data-testid="stTabs"] button { color: #dceaff !important; font-weight: 600 !important; }
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #3b82f6 !important;
    border-bottom: 3px solid #3b82f6 !important;
}

.stAlert { border-radius: 14px !important; backdrop-filter: blur(10px); }
div[data-testid="stDataFrame"] { background: rgba(255,255,255,0.05); border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

API_URL = "http://127.0.0.1:8001/predict"
HISTORY_FILE = "history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_history(entry):
    history = load_history()
    history.insert(0, entry)
    history = history[:50]
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f)

def predict_passenger(payload):
    response = requests.post(API_URL, json=payload)
    return response.json()

def probability_gauge(prob, label):
    color = "#22c55e" if label == "Survived" else "#ef4444"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        title={'text': f"Survival Probability", 'font': {'color': '#dceaff'}},
        number={'suffix': '%', 'font': {'color': '#ffffff'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#dceaff'},
            'bar': {'color': color},
            'bgcolor': 'rgba(255,255,255,0.05)',
        }
    ))
    fig.update_layout(height=280, margin=dict(l=20,r=20,t=50,b=20),
                       paper_bgcolor='rgba(0,0,0,0)', font={'color': '#dceaff'})
    return fig

st.title("🚢 Titanic Survival Prediction")

tab1, tab2, tab3 = st.tabs(["🧍 Single Passenger", "📂 Batch CSV", "🕒 History"])

with tab1:
    st.write("Enter passenger details to predict survival likelihood.")
    col1, col2, col3 = st.columns(3)

    with col1:
        pclass = st.selectbox("Passenger Class", [1, 2, 3], index=2)
        sex = st.selectbox("Sex", ["male", "female"])
        age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0)

    with col2:
        sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
        parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0)
        fare = st.number_input("Fare Paid", min_value=0.0, max_value=600.0, value=32.0)

    with col3:
        embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])
        title = st.selectbox("Title", ["Mr", "Mrs", "Miss", "Master", "Rare"])

    if st.button("Predict Survival"):
        try:
            with st.spinner("🔍 Analyzing passenger..."):
                payload = {
                    "Pclass": pclass, "Sex": sex, "Age": age,
                    "SibSp": sibsp, "Parch": parch, "Fare": fare,
                    "Embarked": embarked, "Title": title
                }
                result = predict_passenger(payload)

            label = result["prediction"]
            prob = result["survival_probability"]

            if label == "Survived":
                st.success(f"✅ Prediction: {label}")
            else:
                st.error(f"🚩 Prediction: {label}")

            st.plotly_chart(probability_gauge(prob, label), use_container_width=True)

            save_history({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "class": pclass, "sex": sex, "age": age,
                "prediction": label, "probability": prob
            })
        except Exception as e:
            st.error(f"Error connecting to API: {e}")

with tab2:
    st.write("Upload a CSV with passenger columns (Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, Title).")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        required_cols = {"Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Title"}
        if not required_cols.issubset(df.columns):
            st.error(f"CSV must have columns: {required_cols}")
        else:
            if st.button("Run Batch Prediction"):
                results = []
                progress = st.progress(0)
                for i, row in df.iterrows():
                    try:
                        payload = row[list(required_cols)].to_dict()
                        res = predict_passenger(payload)
                        results.append({"prediction": res["prediction"], "survival_probability": res["survival_probability"]})
                    except Exception:
                        results.append({"prediction": "Error", "survival_probability": None})
                    progress.progress((i + 1) / len(df))

                result_df = pd.concat([df, pd.DataFrame(results)], axis=1)
                st.dataframe(result_df)

                csv_out = result_df.to_csv(index=False).encode("utf-8")
                st.download_button("Download Results CSV", csv_out, "predictions.csv", "text/csv")

with tab3:
    st.write("Last 50 predictions made in this app.")
    history = load_history()
    if len(history) == 0:
        st.info("No predictions yet.")
    else:
        st.dataframe(pd.DataFrame(history), use_container_width=True)
        if st.button("Clear History"):
            if os.path.exists(HISTORY_FILE):
                os.remove(HISTORY_FILE)
            st.rerun()