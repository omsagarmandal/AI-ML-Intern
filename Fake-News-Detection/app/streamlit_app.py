import streamlit as st
import requests

st.set_page_config(page_title="Fake News Detector", page_icon="📰")

st.title("📰 Fake News Detection")
st.write("Paste a news article below to check if it's likely Fake or Real.")

API_URL = "http://127.0.0.1:8000/predict"

text_input = st.text_area("News Article Text", height=250)

if st.button("Check News"):
    if text_input.strip() == "":
        st.warning("Please paste some text first.")
    else:
        try:
            response = requests.post(API_URL, json={"text": text_input})
            result = response.json()
            label = result["prediction"]
            score = result["confidence_score"]

            if label == "Real":
                st.success(f"✅ Prediction: {label}")
            else:
                st.error(f"🚩 Prediction: {label}")

            st.write(f"Confidence score: {score:.4f}")
        except Exception as e:
            st.error(f"Error connecting to API: {e}")