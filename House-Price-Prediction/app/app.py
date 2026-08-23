import streamlit as st
import streamlit.components.v1 as components
import joblib
import pandas as pd
import numpy as np
import time

st.set_page_config(page_title="Estate Valuator", page_icon="🏛️", layout="wide")

model = joblib.load('../models/house_price_model.pkl')
model_columns = joblib.load('../models/model_columns.pkl')

USD_TO_NPR = 152.5   # live rate, update if needed

# ================= PREMIUM CSS =================
st.markdown("""
<style>
@keyframes bgMove {
    0%   { background-position: 0% 0%, 100% 100%, 50% 50%; }
    50%  { background-position: 100% 50%, 0% 50%, 100% 0%; }
    100% { background-position: 0% 0%, 100% 100%, 50% 50%; }
}
.stApp {
    background:
        radial-gradient(circle at 20% 30%, rgba(212,175,55,0.10), transparent 40%),
        radial-gradient(circle at 80% 70%, rgba(120,80,200,0.12), transparent 45%),
        radial-gradient(circle at 50% 50%, rgba(20,25,35,1), rgba(8,10,14,1) 70%);
    background-size: 200% 200%, 200% 200%, 100% 100%;
    animation: bgMove 18s ease-in-out infinite;
}

.title-text {
    font-size: 46px; font-weight: 800; color: #D4AF37;
    text-align: center; padding: 10px 0 0 0; letter-spacing: 1.5px;
    text-shadow: 0 0 20px rgba(212,175,55,0.5);
}
.subtitle-text {
    text-align: center; color: #9a9a9a; font-size: 15px; margin-bottom: 35px;
}

/* Glass card */
.glass-card {
    background: rgba(26, 29, 36, 0.55);
    backdrop-filter: blur(14px);
    border: 1px solid rgba(212,175,55,0.25);
    border-radius: 16px;
    padding: 22px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.45);
    position: relative;
    overflow: hidden;
    transition: transform 0.35s ease, box-shadow 0.35s ease;
}
.glass-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 14px 40px rgba(212,175,55,0.25);
}
.glass-card::before {
    content: "";
    position: absolute; top: 0; left: -150%;
    width: 100%; height: 100%;
    background: linear-gradient(120deg, transparent, rgba(255,255,255,0.15), transparent);
    transition: left 0.7s ease;
}
.glass-card:hover::before { left: 150%; }

.card-icon {
    font-size: 30px;
    filter: drop-shadow(0 0 6px rgba(212,175,55,0.6));
    transition: transform 0.3s ease, filter 0.3s ease;
}
.glass-card:hover .card-icon {
    transform: scale(1.15) rotate(-3deg);
    filter: drop-shadow(0 0 14px rgba(212,175,55,0.9));
}

.result-card {
    background: linear-gradient(135deg, rgba(26,29,36,0.85), rgba(38,43,54,0.85));
    border: 1px solid #D4AF37;
    border-radius: 18px;
    padding: 35px;
    text-align: center;
    margin-top: 25px;
    box-shadow: 0 0 40px rgba(212,175,55,0.25);
}
.result-price {
    font-size: 48px; color: #D4AF37; font-weight: 800;
    text-shadow: 0 0 25px rgba(212,175,55,0.6);
}

.stat-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(212,175,55,0.2);
    border-radius: 12px;
    padding: 14px;
    text-align: center;
    transition: transform 0.25s ease, background 0.25s ease;
}
.stat-card:hover {
    transform: translateY(-4px);
    background: rgba(212,175,55,0.08);
}
.stat-val { font-size: 20px; color: #D4AF37; font-weight: 700; }
.stat-label { font-size: 12px; color: #999; margin-top: 4px; }

div.stButton > button {
    background: linear-gradient(135deg, #D4AF37, #F0D77B);
    color: #0E1117; font-weight: 800; letter-spacing: 1px;
    border-radius: 10px; height: 3.2em; width: 100%; border: none;
    box-shadow: 0 6px 20px rgba(212,175,55,0.35);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    position: relative; overflow: hidden;
}
div.stButton > button:hover {
    transform: translateY(-3px) scale(1.01);
    box-shadow: 0 10px 30px rgba(212,175,55,0.55);
}
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown('<div class="title-text">🏛️ ESTATE VALUATOR</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">AI-Powered Property Price Estimation</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="glass-card"><span class="card-icon">🏠</span> <b>Structure</b></div>', unsafe_allow_html=True)
    overall_qual = st.slider("Overall Quality", 1, 10, 5)
    gr_liv_area = st.number_input("Living Area (sqft)", 500, 6000, 1500)
    total_bsmt_sf = st.number_input("Basement Area (sqft)", 0, 3000, 800)

with col2:
    st.markdown('<div class="glass-card"><span class="card-icon">📐</span> <b>Layout</b></div>', unsafe_allow_html=True)
    first_flr_sf = st.number_input("1st Floor (sqft)", 300, 3000, 1000)
    second_flr_sf = st.number_input("2nd Floor (sqft)", 0, 2000, 0)
    full_bath = st.slider("Full Bathrooms", 0, 4, 2)

with col3:
    st.markdown('<div class="glass-card"><span class="card-icon">🚗</span> <b>Extras</b></div>', unsafe_allow_html=True)
    garage_cars = st.slider("Garage Capacity", 0, 4, 2)
    year_built = st.number_input("Year Built", 1900, 2024, 2000)
    year_sold = st.number_input("Year Sold", 2006, 2024, 2010)

st.markdown("<br>", unsafe_allow_html=True)
predict_clicked = st.button("VALUATE PROPERTY")

# ================= BORDER PARTICLE BURST (click effect) =================
components.html("""
<script>
const doc = window.parent.document;

function attachBurst() {
    const btns = doc.querySelectorAll('div.stButton > button');
    btns.forEach(btn => {
        if (btn.dataset.burstAttached) return;
        btn.dataset.burstAttached = "true";
        btn.addEventListener('click', function(e) {
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
            const perimeterPoints = [];
            const steps = 40;
            for (let i = 0; i < steps; i++) {
                let t = i / steps;
                let x, y;
                if (t < 0.25) { x = rect.left + (t/0.25)*rect.width; y = rect.top; }
                else if (t < 0.5) { x = rect.right; y = rect.top + ((t-0.25)/0.25)*rect.height; }
                else if (t < 0.75) { x = rect.right - ((t-0.5)/0.25)*rect.width; y = rect.bottom; }
                else { x = rect.left; y = rect.bottom - ((t-0.75)/0.25)*rect.height; }
                perimeterPoints.push([x, y]);
            }

            perimeterPoints.forEach(([x, y]) => {
                for (let i = 0; i < 2; i++) {
                    particles.push({
                        x: x, y: y,
                        vx: (Math.random()-0.5)*6,
                        vy: (Math.random()-0.5)*6,
                        rot: Math.random()*360,
                        vrot: (Math.random()-0.5)*15,
                        size: Math.random()*6+3,
                        life: 1
                    });
                }
            });

            function animate() {
                ctx.clearRect(0,0,canvas.width,canvas.height);
                particles.forEach(p => {
                    p.x += p.vx; p.y += p.vy; p.vy += 0.15;
                    p.rot += p.vrot; p.life -= 0.02;
                    if (p.life > 0) {
                        ctx.save();
                        ctx.translate(p.x, p.y);
                        ctx.rotate(p.rot * Math.PI/180);
                        ctx.fillStyle = `rgba(212,175,55,${p.life})`;
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

# ================= PREDICTION =================
if predict_clicked:
    with st.spinner(""):
        placeholder = st.empty()
        placeholder.markdown("""
        <div style="text-align:center; padding:30px;">
            <div style="font-size:16px; color:#D4AF37;">Calculating Property Value...</div>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(1.2)
        placeholder.empty()

    input_df = pd.DataFrame(np.zeros((1, len(model_columns))), columns=model_columns)
    input_df['OverallQual'] = overall_qual
    input_df['GrLivArea'] = gr_liv_area
    input_df['TotalBsmtSF'] = total_bsmt_sf
    input_df['1stFlrSF'] = first_flr_sf
    input_df['2ndFlrSF'] = second_flr_sf
    input_df['GarageCars'] = garage_cars
    input_df['FullBath'] = full_bath
    input_df['TotalSF'] = total_bsmt_sf + first_flr_sf + second_flr_sf
    input_df['HouseAge'] = year_sold - year_built
    input_df['RemodAge'] = year_sold - year_built
    input_df['TotalBath'] = full_bath

    pred_log = model.predict(input_df)[0]
    pred_price_usd = np.expm1(pred_log)
    pred_price_npr = pred_price_usd * USD_TO_NPR
    total_sf = total_bsmt_sf + first_flr_sf + second_flr_sf
    price_per_sqft_npr = pred_price_npr / total_sf if total_sf > 0 else 0

    st.markdown(f"""
    <div class="result-card">
        <div style="color:#999; font-size:15px; letter-spacing:1px;">ESTIMATED MARKET VALUE</div>
        <div class="result-price">रू {pred_price_npr:,.0f}</div>
        <div style="color:#777; font-size:13px; margin-top:6px;">(~${pred_price_usd:,.0f} USD)</div>
    </div>
    """, unsafe_allow_html=True)

    s1, s2, s3, s4 = st.columns(4)
    stats = [
        (s1, "📏", f"{total_sf:,.0f} sqft", "Total Area"),
        (s2, "💰", f"रू {price_per_sqft_npr:,.0f}", "Price / sqft"),
        (s3, "🏗️", f"{year_sold - year_built} yrs", "Property Age"),
        (s4, "⭐", f"{overall_qual}/10", "Quality Score"),
    ]
    for col, icon, val, label in stats:
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div class="card-icon">{icon}</div>
                <div class="stat-val">{val}</div>
                <div class="stat-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)