import streamlit as st
import streamlit.components.v1 as components
import joblib
import numpy as np
import time

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="wide")

model = joblib.load('../models/disease_model.pkl')
scaler = joblib.load('../models/scaler.pkl')

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

@keyframes bgMove {
    0%   { background-position: 0% 0%, 100% 100%, 50% 50%; }
    50%  { background-position: 100% 50%, 0% 50%, 100% 0%; }
    100% { background-position: 0% 0%, 100% 100%, 50% 50%; }
}
.stApp {
    background:
        radial-gradient(circle at 20% 20%, rgba(56,189,248,0.10), transparent 40%),
        radial-gradient(circle at 80% 75%, rgba(99,102,241,0.12), transparent 45%),
        radial-gradient(circle at 50% 0%, #162032 0%, #0a0e1a 60%);
    background-size: 200% 200%, 200% 200%, 100% 100%;
    animation: bgMove 20s ease-in-out infinite;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 1100px; }

.hero { text-align: center; padding: 10px 0 30px 0; }
.hero-badge {
    display: inline-block;
    background: rgba(56,189,248,0.1);
    border: 1px solid rgba(56,189,248,0.3);
    color: #38bdf8;
    padding: 6px 18px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.5px;
    margin-bottom: 18px;
}
.hero-title {
    font-family: 'Poppins', sans-serif;
    font-size: 44px;
    font-weight: 800;
    background: linear-gradient(90deg, #ffffff, #94a3b8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.hero-sub { color: #64748b; font-size: 14px; margin-top: 10px; font-weight: 500; }

.stats-row { display: flex; justify-content: center; gap: 14px; margin-top: 26px; flex-wrap: wrap; }
.stat-pill {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 12px 22px;
    text-align: center;
    min-width: 110px;
    transition: transform 0.25s ease, background 0.25s ease;
}
.stat-pill:hover { transform: translateY(-3px); background: rgba(56,189,248,0.07); }
.stat-value { font-family: 'Poppins', sans-serif; font-size: 20px; font-weight: 800; color: #38bdf8; }
.stat-label { font-size: 10px; color: #64748b; letter-spacing: 1px; text-transform: uppercase; margin-top: 2px; font-weight: 600; }

.glass-card {
    background: rgba(255,255,255,0.025);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 22px;
    padding: 32px 36px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.25);
    margin-top: 24px;
    height: 100%;
    position: relative;
    overflow: hidden;
    transition: transform 0.35s ease, box-shadow 0.35s ease;
}
.glass-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 14px 40px rgba(56,189,248,0.18);
}
.glass-card::before {
    content: "";
    position: absolute; top: 0; left: -150%;
    width: 100%; height: 100%;
    background: linear-gradient(120deg, transparent, rgba(255,255,255,0.08), transparent);
    transition: left 0.7s ease;
}
.glass-card:hover::before { left: 150%; }

.section-label {
    font-family: 'Poppins', sans-serif;
    color: #e2e8f0; font-size: 12px; font-weight: 700;
    letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 6px;
}
.section-desc { color: #64748b; font-size: 12.5px; margin-bottom: 22px; }

div[data-testid="stNumberInput"] label { color: #94a3b8 !important; font-weight: 600 !important; font-size: 12.5px !important; }
div[data-testid="stNumberInput"] input {
    background-color: rgba(255,255,255,0.035) !important;
    color: white !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    padding: 10px 14px !important;
    font-weight: 600 !important;
}
div[data-testid="stNumberInput"] input:focus { border: 1px solid #38bdf8 !important; }
div[data-testid="stNumberInput"] button { background-color: rgba(255,255,255,0.04) !important; border: 1px solid rgba(255,255,255,0.08) !important; }

.stButton button {
    width: 100%;
    background: linear-gradient(90deg, #38bdf8, #6366f1);
    color: white;
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    font-size: 14.5px;
    padding: 14px 0;
    border-radius: 14px;
    border: none;
    margin-top: 10px;
    letter-spacing: 0.5px;
    box-shadow: 0 4px 20px rgba(56,189,248,0.22);
    transition: all 0.25s ease;
}
.stButton button:hover { transform: translateY(-2px); box-shadow: 0 8px 28px rgba(56,189,248,0.4); }

.placeholder-box {
    height: 100%; min-height: 320px;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    text-align: center; color: #475569;
}
.placeholder-icon { font-size: 34px; margin-bottom: 12px; opacity: 0.5; }
.placeholder-text { font-size: 13px; max-width: 220px; line-height: 1.6; }

.result-card { border-radius: 18px; padding: 30px 20px; text-align: center; animation: fadeIn 0.4s ease; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
.high-risk { background: linear-gradient(160deg, rgba(239,68,68,0.14), rgba(239,68,68,0.02)); border: 1px solid rgba(239,68,68,0.28); }
.low-risk { background: linear-gradient(160deg, rgba(34,197,94,0.14), rgba(34,197,94,0.02)); border: 1px solid rgba(34,197,94,0.28); }

.result-icon-badge {
    width: 56px; height: 56px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 26px; margin: 0 auto 14px auto;
}
.high-risk .result-icon-badge { background: rgba(239,68,68,0.15); }
.low-risk .result-icon-badge { background: rgba(34,197,94,0.15); }

.result-title { font-family: 'Poppins', sans-serif; font-size: 22px; font-weight: 800; margin-bottom: 4px; }
.high-risk .result-title { color: #f87171; }
.low-risk .result-title { color: #4ade80; }
.result-sub { color: #94a3b8; font-size: 12.5px; margin-bottom: 20px; }

.confidence-track { background: rgba(255,255,255,0.06); height: 8px; border-radius: 999px; overflow: hidden; }
.confidence-fill { height: 100%; border-radius: 999px; }
.high-risk .confidence-fill { background: linear-gradient(90deg, #f87171, #ef4444); }
.low-risk .confidence-fill { background: linear-gradient(90deg, #4ade80, #22c55e); }
.confidence-label { color: #64748b; font-size: 11.5px; margin-top: 10px; font-weight: 600; letter-spacing: 0.5px; }

.top-factors { margin-top: 22px; text-align: left; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 18px; }
.factor-title { font-size: 10.5px; color: #64748b; letter-spacing: 1px; text-transform: uppercase; font-weight: 700; margin-bottom: 10px; }
.factor-item { display: flex; justify-content: space-between; font-size: 12.5px; color: #cbd5e1; padding: 5px 0; }
.factor-item span:last-child { color: #64748b; }

.footer-note { text-align: center; color: #475569; font-size: 11px; margin-top: 36px; padding-bottom: 20px; }

.loading-box { text-align: center; padding: 60px 0; color: #38bdf8; font-family: 'Poppins', sans-serif; font-weight: 600; font-size: 14px; }
.loading-dot {
    display: inline-block; width: 8px; height: 8px; margin: 0 4px;
    background: #38bdf8; border-radius: 50%;
    animation: dotPulse 1s infinite ease-in-out;
}
.loading-dot:nth-child(2) { animation-delay: 0.15s; }
.loading-dot:nth-child(3) { animation-delay: 0.3s; }
@keyframes dotPulse { 0%, 80%, 100% { opacity: 0.25; transform: scale(0.8); } 40% { opacity: 1; transform: scale(1.1); } }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <div class="hero-badge">AI HEALTH DIAGNOSTICS</div>
    <p class="hero-title">Diabetes Risk Predictor</p>
    <p class="hero-sub">Machine learning model trained on 768 patient records · Logistic Regression</p>
    <div class="stats-row">
        <div class="stat-pill"><div class="stat-value">75.3%</div><div class="stat-label">Accuracy</div></div>
        <div class="stat-pill"><div class="stat-value">0.82</div><div class="stat-label">AUC Score</div></div>
        <div class="stat-pill"><div class="stat-value">8</div><div class="stat-label">Bio-markers</div></div>
    </div>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1.3, 1], gap="medium")

with left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">📋 Patient Health Metrics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Enter diagnostic measurements below</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=1)
        bp = st.number_input('Blood Pressure (mm Hg)', min_value=0, max_value=150, value=70)
        insulin = st.number_input('Insulin (mu U/ml)', min_value=0, max_value=900, value=80)
        dpf = st.number_input('Pedigree Function', min_value=0.0, max_value=3.0, value=0.5)
    with c2:
        glucose = st.number_input('Glucose (mg/dL)', min_value=0, max_value=250, value=120)
        skin = st.number_input('Skin Thickness (mm)', min_value=0, max_value=100, value=20)
        bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, value=25.0)
        age = st.number_input('Age (years)', min_value=1, max_value=120, value=30)

    predict_clicked = st.button('🔍  Analyze Risk')
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">🎯 Prediction Result</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Model output based on entered metrics</div>', unsafe_allow_html=True)

    if predict_clicked:
        loading_spot = st.empty()
        loading_spot.markdown("""
        <div class="loading-box">
            Analyzing patient data
            <span class="loading-dot"></span><span class="loading-dot"></span><span class="loading-dot"></span>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(1.1)
        loading_spot.empty()

        input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        risk_class = "high-risk" if prediction == 1 else "low-risk"
        icon = "⚠️" if prediction == 1 else "✅"
        title = "High Risk" if prediction == 1 else "Low Risk"
        sub = "Elevated likelihood of diabetes" if prediction == 1 else "No significant diabetes indicators"
        conf = prob * 100 if prediction == 1 else (1 - prob) * 100

        st.markdown(f"""
        <div class="result-card {risk_class}">
            <div class="result-icon-badge">{icon}</div>
            <div class="result-title">{title}</div>
            <div class="result-sub">{sub}</div>
            <div class="confidence-track"><div class="confidence-fill" style="width:{conf}%;"></div></div>
            <div class="confidence-label">{conf:.1f}% MODEL CONFIDENCE</div>
            <div class="top-factors">
                <div class="factor-title">Key Input Values</div>
                <div class="factor-item"><span>Glucose</span><span>{glucose} mg/dL</span></div>
                <div class="factor-item"><span>BMI</span><span>{bmi}</span></div>
                <div class="factor-item"><span>Age</span><span>{age} yrs</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="placeholder-box">
            <div class="placeholder-icon">🩺</div>
            <div class="placeholder-text">Fill in the patient details and click <b>Analyze Risk</b> to generate a prediction.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="footer-note">Built with Streamlit &amp; Scikit-learn · For educational purposes only, not a medical diagnosis</div>
""", unsafe_allow_html=True)

# ================= CLICK PARTICLE BURST (border) =================
components.html("""
<script>
const doc = window.parent.document;

function attachBurst() {
    const btns = doc.querySelectorAll('div.stButton > button');
    btns.forEach(btn => {
        if (btn.dataset.burstAttached) return;
        btn.dataset.burstAttached = "true";
        btn.addEventListener('click', function() {
            const rect = btn.getBoundingClientRect();
            let canvas = doc.getElementById('burstCanvas');
            if (!canvas) {
                canvas = doc.createElement('canvas');
                canvas.id = 'burstCanvas';
                canvas.style.position = 'fixed';
                canvas.style.top = '0';
                canvas.style.left = '0';
                canvas.style.width = '100vw';
                canvas.style.height = '100vh';
                canvas.style.pointerEvents = 'none';
                canvas.style.zIndex = '99999';
                doc.body.appendChild(canvas);
            }
            canvas.width = window.parent.innerWidth;
            canvas.height = window.parent.innerHeight;
            const ctx = canvas.getContext('2d');

            let particles = [];
            const steps = 40;
            for (let i = 0; i < steps; i++) {
                let t = i / steps, x, y;
                if (t < 0.25) { x = rect.left + (t/0.25)*rect.width; y = rect.top; }
                else if (t < 0.5) { x = rect.right; y = rect.top + ((t-0.25)/0.25)*rect.height; }
                else if (t < 0.75) { x = rect.right - ((t-0.5)/0.25)*rect.width; y = rect.bottom; }
                else { x = rect.left; y = rect.bottom - ((t-0.75)/0.25)*rect.height; }
                for (let j = 0; j < 2; j++) {
                    particles.push({
                        x, y,
                        vx: (Math.random()-0.5)*6,
                        vy: (Math.random()-0.5)*6,
                        rot: Math.random()*360,
                        vrot: (Math.random()-0.5)*15,
                        size: Math.random()*5+3,
                        life: 1
                    });
                }
            }

            function animate() {
                ctx.clearRect(0,0,canvas.width,canvas.height);
                particles.forEach(p => {
                    p.x += p.vx; p.y += p.vy; p.vy += 0.15;
                    p.rot += p.vrot; p.life -= 0.02;
                    if (p.life > 0) {
                        ctx.save();
                        ctx.translate(p.x, p.y);
                        ctx.rotate(p.rot * Math.PI/180);
                        ctx.fillStyle = `rgba(56,189,248,${p.life})`;
                        ctx.fillRect(-p.size/2, -p.size/2, p.size, p.size*0.6);
                        ctx.restore();
                    }
                });
                particles = particles.filter(p => p.life > 0);
                if (particles.length > 0) requestAnimationFrame(animate);
                else ctx.clearRect(0,0,canvas.width,canvas.height);
            }
            animate();
        });
    });
}
attachBurst();
setInterval(attachBurst, 800);
</script>
""", height=0)