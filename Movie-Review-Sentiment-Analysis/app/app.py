import streamlit as st
import streamlit.components.v1 as components
import joblib
import re
import time
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

st.set_page_config(page_title="CineSense", page_icon="🎬", layout="wide")

model = joblib.load('../models/sentiment_model.pkl')
tfidf = joblib.load('../models/tfidf_vectorizer.pkl')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return ' '.join(words)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

@keyframes bgMove {
    0%   { background-position: 0% 0%, 100% 100%; }
    50%  { background-position: 100% 50%, 0% 50%; }
    100% { background-position: 0% 0%, 100% 100%; }
}
.stApp {
    background:
        radial-gradient(circle at 15% 15%, rgba(240,180,41,0.08), transparent 42%),
        radial-gradient(circle at 85% 85%, rgba(220,38,38,0.10), transparent 45%),
        #0c0806;
    background-size: 200% 200%, 200% 200%, 100% 100%;
    animation: bgMove 22s ease-in-out infinite;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2.5rem; max-width: 100%; padding-left: 4rem; padding-right: 4rem; }

.hero { text-align: center; padding-bottom: 30px; }
.hero-badge {
    display: inline-block;
    background: rgba(240,180,41,0.1);
    border: 1px solid rgba(240,180,41,0.35);
    color: #f0b429;
    padding: 8px 22px;
    border-radius: 999px;
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 1.8px;
    margin-bottom: 20px;
}
.hero-title {
    font-family: 'Poppins', sans-serif;
    font-size: 64px;
    font-weight: 800;
    color: #f5efe6;
    margin: 0;
}
.hero-sub { color: #b3a794; font-size: 19px; margin-top: 12px; font-weight: 500; }

.stats-row { display: flex; justify-content: center; gap: 18px; margin-top: 28px; flex-wrap: wrap; }
.stat-pill {
    background: rgba(240,180,41,0.05);
    border: 1px solid rgba(240,180,41,0.18);
    border-radius: 14px;
    padding: 16px 30px;
    text-align: center;
    min-width: 130px;
    transition: transform 0.2s ease;
}
.stat-pill:hover { transform: translateY(-2px); background: rgba(240,180,41,0.09); }
.stat-value { font-family: 'Poppins', sans-serif; font-size: 26px; font-weight: 800; color: #f0b429; }
.stat-label { font-size: 12px; color: #b3a794; letter-spacing: 1px; text-transform: uppercase; margin-top: 4px; font-weight: 600; }

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255,255,255,0.02);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(240,180,41,0.15) !important;
    border-radius: 18px !important;
    box-shadow: 0 8px 28px rgba(0,0,0,0.35);
}

.section-label {
    font-family: 'Poppins', sans-serif;
    color: #f5efe6; font-size: 20px; font-weight: 700;
    letter-spacing: 0.5px; margin-bottom: 4px;
}
.section-desc { color: #b3a794; font-size: 15px; margin-bottom: 18px; }

div[data-testid="stTextArea"] textarea {
    background-color: rgba(255,255,255,0.03) !important;
    color: #f5efe6 !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    padding: 16px !important;
    font-size: 18px !important;
}
div[data-testid="stTextArea"] textarea:focus { border: 1px solid #f0b429 !important; }

.stButton button {
    width: 100%;
    background: linear-gradient(90deg, #dc2626, #f0b429);
    color: #0c0806;
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    font-size: 19px;
    padding: 16px 0;
    border-radius: 12px;
    border: none;
    margin-top: 10px;
    letter-spacing: 0.4px;
    box-shadow: 0 4px 16px rgba(220,38,38,0.25);
    transition: all 0.2s ease;
}
.stButton button:hover { transform: translateY(-2px); box-shadow: 0 8px 22px rgba(240,180,41,0.35); }

.result-title { font-family: 'Poppins', sans-serif; font-size: 30px; font-weight: 800; margin: 8px 0 4px 0; text-align: center; }
.positive-text { color: #4ade80; }
.negative-text { color: #f87171; }
.result-sub { color: #b3a794; font-size: 15px; margin-bottom: 18px; text-align: center; }
.result-icon { font-size: 42px; text-align: center; display: block; margin-bottom: 6px; }

.confidence-track { background: rgba(255,255,255,0.06); height: 10px; border-radius: 999px; overflow: hidden; }
.confidence-fill-pos { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #4ade80, #22c55e); }
.confidence-fill-neg { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #f87171, #ef4444); }
.confidence-label { color: #b3a794; font-size: 13px; margin-top: 10px; font-weight: 600; letter-spacing: 0.4px; text-align: center; }

.loading-box { text-align: center; padding: 20px 0; color: #f0b429; font-family: 'Poppins', sans-serif; font-weight: 600; font-size: 16px; }
.loading-dot {
    display: inline-block; width: 7px; height: 7px; margin: 0 3px;
    background: #f0b429; border-radius: 50%;
    animation: dotPulse 1s infinite ease-in-out;
}
.loading-dot:nth-child(2) { animation-delay: 0.15s; }
.loading-dot:nth-child(3) { animation-delay: 0.3s; }
@keyframes dotPulse { 0%, 80%, 100% { opacity: 0.25; transform: scale(0.8); } 40% { opacity: 1; transform: scale(1.1); } }

.footer-note { text-align: center; color: #5c5347; font-size: 10.5px; margin-top: 24px; padding-bottom: 16px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <div class="hero-badge">NLP SENTIMENT ANALYSIS</div>
    <p class="hero-title">🎬 CineSense</p>
    <p class="hero-sub">Movie review sentiment predictor · trained on 50,000 IMDB reviews</p>
    <div class="stats-row">
        <div class="stat-pill"><div class="stat-value">88.8%</div><div class="stat-label">Accuracy</div></div>
        <div class="stat-pill"><div class="stat-value">0.889</div><div class="stat-label">F1 Score</div></div>
        <div class="stat-pill"><div class="stat-value">50K</div><div class="stat-label">Reviews</div></div>
    </div>
</div>
""", unsafe_allow_html=True)

with st.container(border=True):
    st.markdown('<div class="section-label">✍️ Enter a Movie Review</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-desc">Paste or type any movie review below</div>', unsafe_allow_html=True)
    review_text = st.text_area("Review", height=140, placeholder="e.g. This movie completely blew me away, the acting was phenomenal...", label_visibility="collapsed")
    predict_clicked = st.button("🔍  Analyze Sentiment")

if predict_clicked:
    if not review_text.strip():
        st.warning("Please enter a review first.")
    else:
        loading_spot = st.empty()
        loading_spot.markdown("""
        <div class="loading-box">
            Analyzing sentiment
            <span class="loading-dot"></span><span class="loading-dot"></span><span class="loading-dot"></span>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(1.0)
        loading_spot.empty()

        cleaned = clean_text(review_text)
        vectorized = tfidf.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        prob = model.predict_proba(vectorized)[0]

        is_positive = prediction == 1
        icon = "🎉" if is_positive else "💔"
        title_class = "positive-text" if is_positive else "negative-text"
        title = "Positive Review" if is_positive else "Negative Review"
        fill_class = "confidence-fill-pos" if is_positive else "confidence-fill-neg"
        conf = prob[1] * 100 if is_positive else prob[0] * 100

        with st.container(border=True):
            st.markdown(f"""
            <span class="result-icon">{icon}</span>
            <div class="result-title {title_class}">{title}</div>
            <div class="result-sub">Model's read on this review's sentiment</div>
            <div class="confidence-track"><div class="{fill_class}" style="width:{conf}%;"></div></div>
            <div class="confidence-label">{conf:.1f}% MODEL CONFIDENCE</div>
            """, unsafe_allow_html=True)

st.markdown("""
<div class="footer-note">Built with Streamlit &amp; Scikit-learn · Trained on the IMDB 50K dataset · For educational purposes</div>
""", unsafe_allow_html=True)

# ================= CLICK PARTICLE BURST =================
components.html("""
<script>
const doc = window.parent.document;

function burst(rect) {
    let canvas = doc.getElementById('burstCanvas');
    if (!canvas) {
        canvas = doc.createElement('canvas');
        canvas.id = 'burstCanvas';
        canvas.style.position = 'fixed';
        canvas.style.top = '0'; canvas.style.left = '0';
        canvas.style.width = '100vw'; canvas.style.height = '100vh';
        canvas.style.pointerEvents = 'none'; canvas.style.zIndex = '99999';
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
            particles.push({ x, y, vx: (Math.random()-0.5)*6, vy: (Math.random()-0.5)*6,
                rot: Math.random()*360, vrot: (Math.random()-0.5)*15, size: Math.random()*5+3, life: 1 });
        }
    }
    let raf = null;
    function animate() {
        ctx.clearRect(0,0,canvas.width,canvas.height);
        particles.forEach(p => {
            p.x += p.vx; p.y += p.vy; p.vy += 0.15; p.rot += p.vrot; p.life -= 0.02;
            if (p.life > 0) {
                ctx.save(); ctx.translate(p.x, p.y); ctx.rotate(p.rot * Math.PI/180);
                ctx.fillStyle = `rgba(240,180,41,${p.life})`;
                ctx.fillRect(-p.size/2, -p.size/2, p.size, p.size*0.6);
                ctx.restore();
            }
        });
        particles = particles.filter(p => p.life > 0);
        if (particles.length > 0) raf = requestAnimationFrame(animate);
        else ctx.clearRect(0,0,canvas.width,canvas.height);
    }
    animate();
}

// Event delegation on body: survives Streamlit re-rendering the button on every rerun,
// and only needs to be attached once, ever, so it fires on every single click.
if (!doc.body.dataset.burstDelegated) {
    doc.body.dataset.burstDelegated = "true";
    doc.body.addEventListener('click', function(e) {
        const btn = e.target.closest('div.stButton > button');
        if (!btn) return;
        burst(btn.getBoundingClientRect());
    });
}
</script>
""", height=0)