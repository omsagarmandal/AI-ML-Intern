import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import plotly.graph_objects as go
import json
import os
import time
from datetime import datetime

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap');

* { font-family: 'Poppins', sans-serif; }

.stApp {
    background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e, #1a1a2e);
    background-size: 400% 400%;
    animation: gradientShift 15s ease infinite;
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

h1 {
    color: #ffffff !important;
    text-shadow: 0 0 20px rgba(150, 100, 255, 0.6);
    animation: fadeInDown 0.8s ease;
}

@keyframes fadeInDown {
    from { opacity: 0; transform: translateY(-20px); }
    to { opacity: 1; transform: translateY(0); }
}

p, label, .stMarkdown { color: #dcdcff !important; }

.stTextArea textarea, .stTextInput input {
    background: rgba(30, 30, 50, 0.6) !important;
    border: 1px solid rgba(100, 130, 255, 0.3) !important;
    border-radius: 12px !important;
    color: #ffffff !important;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
}

.stTextArea textarea:focus, .stTextInput input:focus {
    border: 1px solid rgba(100, 130, 255, 0.9) !important;
    box-shadow: 0 0 15px rgba(100, 130, 255, 0.4) !important;
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

div[data-testid="stTabs"] button {
    color: #dcdcff !important;
    font-weight: 600 !important;
}

div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #3b82f6 !important;
    border-bottom: 3px solid #3b82f6 !important;
}

.stAlert {
    border-radius: 14px !important;
    backdrop-filter: blur(10px);
    animation: popIn 0.5s ease;
}

@keyframes popIn {
    from { opacity: 0; transform: scale(0.9); }
    to { opacity: 1; transform: scale(1); }
}

div[data-testid="stDataFrame"] {
    background: rgba(255,255,255,0.05);
    border-radius: 12px;
    backdrop-filter: blur(10px);
}
</style>
""", unsafe_allow_html=True)

API_URL = "http://127.0.0.1:8000/predict"
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

def predict_text(text):
    response = requests.post(API_URL, json={"text": text})
    return response.json()

def extract_article_text(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(resp.text, "html.parser")
    paragraphs = soup.find_all("p")
    text = " ".join([p.get_text() for p in paragraphs])
    return text.strip()

def confidence_gauge(score, label):
    color = "#22c55e" if label == "Real" else "#ef4444"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=abs(score),
        title={'text': f"Confidence ({label})", 'font': {'color': '#dcdcff'}},
        number={'font': {'color': '#ffffff'}},
        gauge={
            'axis': {'range': [0, 3], 'tickcolor': '#dcdcff'},
            'bar': {'color': color},
            'bgcolor': 'rgba(255,255,255,0.05)',
            'borderwidth': 0,
        }
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#dcdcff'}
    )
    return fig

st.title("📰 Fake News Detection")

tab1, tab2, tab3, tab4 = st.tabs(["📝 Text", "🔗 URL", "📂 Batch CSV", "🕒 History"])

with tab1:
    st.write("Paste a news article below to check if it's likely Fake or Real.")
    text_input = st.text_area("News Article Text", height=250, key="text_tab")

    if st.button("Check News", key="btn_text"):
        if text_input.strip() == "":
            st.warning("Please paste some text first.")
        else:
            try:
                with st.spinner("🔍 Analyzing article..."):
                    time.sleep(0.8)
                    result = predict_text(text_input)
                label = result["prediction"]
                score = result["confidence_score"]

                if label == "Real":
                    st.success(f"✅ Prediction: {label}")
                else:
                    st.error(f"🚩 Prediction: {label}")

                st.plotly_chart(confidence_gauge(score, label), use_container_width=True)

                save_history({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "source": "Text",
                    "snippet": text_input[:100],
                    "prediction": label,
                    "confidence": score
                })
            except Exception as e:
                st.error(f"Error connecting to API: {e}")

with tab2:
    st.write("Paste a news article URL to auto-fetch and classify.")
    url_input = st.text_input("Article URL", key="url_tab")

    if st.button("Fetch & Check", key="btn_url"):
        if url_input.strip() == "":
            st.warning("Please paste a URL first.")
        else:
            try:
                with st.spinner("🌐 Fetching article..."):
                    article_text = extract_article_text(url_input)

                if len(article_text) < 50:
                    st.warning("Couldn't extract enough text from this URL. Try pasting the text directly.")
                else:
                    st.text_area("Extracted Text", article_text, height=150)
                    with st.spinner("🔍 Analyzing article..."):
                        time.sleep(0.8)
                        result = predict_text(article_text)
                    label = result["prediction"]
                    score = result["confidence_score"]

                    if label == "Real":
                        st.success(f"✅ Prediction: {label}")
                    else:
                        st.error(f"🚩 Prediction: {label}")

                    st.plotly_chart(confidence_gauge(score, label), use_container_width=True)

                    save_history({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "source": "URL",
                        "snippet": url_input,
                        "prediction": label,
                        "confidence": score
                    })
            except Exception as e:
                st.error(f"Error fetching or predicting: {e}")

with tab3:
    st.write("Upload a CSV with a 'text' column to classify multiple articles at once.")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if "text" not in df.columns:
            st.error("CSV must have a column named 'text'.")
        else:
            if st.button("Run Batch Prediction"):
                results = []
                progress = st.progress(0)
                for i, row in df.iterrows():
                    try:
                        res = predict_text(str(row["text"]))
                        results.append({"prediction": res["prediction"], "confidence_score": res["confidence_score"]})
                    except Exception:
                        results.append({"prediction": "Error", "confidence_score": None})
                    progress.progress((i + 1) / len(df))

                result_df = pd.concat([df, pd.DataFrame(results)], axis=1)
                st.dataframe(result_df)

                csv_out = result_df.to_csv(index=False).encode("utf-8")
                st.download_button("Download Results CSV", csv_out, "predictions.csv", "text/csv")

with tab4:
    st.write("Last 50 predictions made in this app.")
    history = load_history()
    if len(history) == 0:
        st.info("No predictions yet.")
    else:
        hist_df = pd.DataFrame(history)
        st.dataframe(hist_df, use_container_width=True)
        if st.button("Clear History"):
            if os.path.exists(HISTORY_FILE):
                os.remove(HISTORY_FILE)
            st.rerun()