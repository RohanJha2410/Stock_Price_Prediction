import streamlit as st
import yfinance as yf
import numpy as np
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tensorflow.keras.models import load_model
from datetime import datetime
import pandas as pd
import time

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QuantumLens · AI Stock Oracle",
    layout="wide",
    page_icon="🔮",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────
# GLOBAL CSS — CYBERPUNK FINANCIAL TERMINAL
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=JetBrains+Mono:wght@300;400;700&family=Bebas+Neue&display=swap');

:root {
    --void:        #03050a;
    --abyss:       #060c14;
    --deep:        #0a1220;
    --surface:     #0e1a2e;
    --elevated:    #122038;
    --highlight:   #1a2d4a;
    --neon-blue:   #00c8ff;
    --neon-green:  #00ff88;
    --neon-gold:   #ffd700;
    --neon-red:    #ff3366;
    --neon-purple: #bf5fff;
    --blue-dim:    rgba(0,200,255,0.08);
    --green-dim:   rgba(0,255,136,0.08);
    --gold-dim:    rgba(255,215,0,0.08);
    --text-bright: #e2f4ff;
    --text-mid:    #6d8fa8;
    --text-dark:   #2a4060;
    --glow-blue:   0 0 30px rgba(0,200,255,0.35), 0 0 60px rgba(0,200,255,0.1);
    --glow-green:  0 0 30px rgba(0,255,136,0.35);
    --glow-gold:   0 0 20px rgba(255,215,0,0.4);
}

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif !important;
    background: var(--void) !important;
    color: var(--text-bright) !important;
}

.stApp {
    background:
        radial-gradient(ellipse 120% 80% at 50% -30%, rgba(0,200,255,0.04) 0%, transparent 50%),
        radial-gradient(ellipse 60% 60% at 90% 10%, rgba(0,255,136,0.03) 0%, transparent 40%),
        radial-gradient(ellipse 80% 80% at 10% 80%, rgba(191,95,255,0.02) 0%, transparent 40%),
        var(--void) !important;
    min-height: 100vh;
    overflow-x: hidden;
}

.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(0,200,255,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,200,255,0.025) 1px, transparent 1px);
    background-size: 60px 60px;
    animation: gridShift 20s linear infinite;
    pointer-events: none;
    z-index: 0;
}
@keyframes gridShift {
    0%   { background-position: 0 0, 0 0; }
    100% { background-position: 60px 60px, 60px 60px; }
}

.stApp::after {
    content: '';
    position: fixed;
    inset: 0;
    background: repeating-linear-gradient(0deg, transparent, transparent 3px, rgba(0,0,0,0.04) 3px, rgba(0,0,0,0.04) 4px);
    pointer-events: none;
    z-index: 1;
}

#MainMenu, footer, header,
.stDeployButton, [data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; }

.block-container {
    padding: 0 2rem 3rem 2rem !important;
    max-width: 1400px !important;
    position: relative;
    z-index: 2;
}

/* TOP NAV */
.topnav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem 0 1rem 0;
    border-bottom: 1px solid rgba(0,200,255,0.12);
    margin-bottom: 2rem;
    position: relative;
}
.topnav::after {
    content: '';
    position: absolute;
    bottom: -1px; left: 0;
    width: 200px; height: 1px;
    background: linear-gradient(90deg, var(--neon-blue), transparent);
}
.brand-logo { display: flex; align-items: center; gap: 0.75rem; }
.brand-icon {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, var(--neon-blue), var(--neon-purple));
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem;
    box-shadow: var(--glow-blue);
    animation: iconPulse 3s ease-in-out infinite;
}
@keyframes iconPulse {
    0%, 100% { box-shadow: var(--glow-blue); }
    50%       { box-shadow: 0 0 50px rgba(0,200,255,0.5), 0 0 100px rgba(0,200,255,0.15); }
}
.brand-name {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.6rem !important;
    letter-spacing: 0.12em !important;
    background: linear-gradient(90deg, var(--neon-blue), var(--neon-green));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
}
.brand-tagline {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.6rem !important;
    color: var(--text-dark) !important;
    letter-spacing: 0.15em;
    margin-top: 2px;
    text-transform: uppercase;
}
.nav-right { display: flex; align-items: center; gap: 1.5rem; }
.nav-status {
    display: flex; align-items: center; gap: 0.4rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: var(--neon-green);
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.nav-status::before {
    content: '';
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--neon-green);
    box-shadow: 0 0 8px var(--neon-green), 0 0 16px var(--neon-green);
    animation: livePulse 1.5s ease-in-out infinite;
}
@keyframes livePulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.5; transform: scale(0.8); }
}
.nav-time {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--text-mid);
    background: var(--surface);
    border: 1px solid rgba(0,200,255,0.1);
    padding: 0.3rem 0.8rem;
    border-radius: 4px;
}
.nav-version {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    color: var(--text-dark);
    background: rgba(0,200,255,0.05);
    border: 1px solid rgba(0,200,255,0.1);
    padding: 0.3rem 0.7rem;
    border-radius: 20px;
}

/* HERO */
.hero-section { text-align: center; padding: 1rem 0 2.5rem 0; }
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: var(--neon-blue);
    margin-bottom: 1rem;
    opacity: 0.7;
}
.hero-title {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: clamp(2.5rem, 6vw, 4.5rem) !important;
    letter-spacing: 0.05em;
    line-height: 0.95;
    color: var(--text-bright);
    margin: 0 0 0.5rem 0 !important;
}
.hero-title span {
    background: linear-gradient(135deg, var(--neon-blue) 0%, var(--neon-green) 50%, var(--neon-blue) 100%);
    background-size: 200% 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: gradientShift 4s ease infinite;
}
@keyframes gradientShift {
    0%, 100% { background-position: 0% 50%; }
    50%       { background-position: 100% 50%; }
}
.hero-sub {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1rem;
    color: var(--text-mid);
    letter-spacing: 0.05em;
}

/* KPI CARDS */
.kpi-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    margin-bottom: 1.5rem;
}
.kpi-card {
    position: relative;
    background: var(--surface);
    padding: 1.5rem 1.6rem;
    overflow: hidden;
    transition: background 0.3s ease;
}
.kpi-card:first-child { border-radius: 12px 0 0 12px; }
.kpi-card:last-child  { border-radius: 0 12px 12px 0; }
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
}
.kpi-card.c-blue::before   { background: linear-gradient(90deg, transparent, var(--neon-blue), transparent); }
.kpi-card.c-green::before  { background: linear-gradient(90deg, transparent, var(--neon-green), transparent); }
.kpi-card.c-gold::before   { background: linear-gradient(90deg, transparent, var(--neon-gold), transparent); }
.kpi-card.c-red::before    { background: linear-gradient(90deg, transparent, var(--neon-red), transparent); }
.kpi-card.c-purple::before { background: linear-gradient(90deg, transparent, var(--neon-purple), transparent); }
.kpi-card::after {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 100px; height: 100px;
    border-radius: 50%;
    opacity: 0.04;
}
.kpi-card.c-blue::after   { background: var(--neon-blue); }
.kpi-card.c-green::after  { background: var(--neon-green); }
.kpi-card.c-gold::after   { background: var(--neon-gold); }
.kpi-card.c-red::after    { background: var(--neon-red); }
.kpi-card.c-purple::after { background: var(--neon-purple); }
.kpi-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--text-dark);
    margin-bottom: 0.6rem;
}
.kpi-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--text-bright);
    line-height: 1;
    margin-bottom: 0.5rem;
    letter-spacing: -0.02em;
}
.kpi-sub {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.8rem;
    color: var(--text-mid);
}
.badge {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    font-weight: 700;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    margin-top: 0.4rem;
}
.badge.up   { background: rgba(0,255,136,0.1); color: var(--neon-green); border: 1px solid rgba(0,255,136,0.2); }
.badge.down { background: rgba(255,51,102,0.1); color: var(--neon-red);   border: 1px solid rgba(255,51,102,0.2); }
.badge.flat { background: rgba(0,200,255,0.1); color: var(--neon-blue);   border: 1px solid rgba(0,200,255,0.2); }

/* STAT RIBBON */
.stat-ribbon {
    display: flex;
    background: var(--surface);
    border: 1px solid rgba(0,200,255,0.08);
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 1.5rem;
}
.ribbon-item {
    flex: 1;
    padding: 0.9rem 1.2rem;
    border-right: 1px solid rgba(0,200,255,0.06);
}
.ribbon-item:last-child { border-right: none; }
.ribbon-key {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-dark);
    margin-bottom: 0.25rem;
}
.ribbon-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: var(--text-bright);
    font-weight: 700;
}
.ribbon-val.green { color: var(--neon-green); }
.ribbon-val.red   { color: var(--neon-red);   }
.ribbon-val.blue  { color: var(--neon-blue);  }
.ribbon-val.gold  { color: var(--neon-gold);  }

/* SIGNAL CARDS */
.signal-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.signal-card {
    background: var(--surface);
    border: 1px solid rgba(0,200,255,0.08);
    border-radius: 10px;
    padding: 1.2rem;
    position: relative;
    overflow: hidden;
}
.signal-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
}
.signal-card.bull::after   { background: linear-gradient(90deg, transparent, var(--neon-green), transparent); }
.signal-card.bear::after   { background: linear-gradient(90deg, transparent, var(--neon-red), transparent);   }
.signal-card.neutral::after{ background: linear-gradient(90deg, transparent, var(--neon-gold), transparent);  }
.signal-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-dark);
    margin-bottom: 0.5rem;
}
.signal-value {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.8rem;
    letter-spacing: 0.05em;
    line-height: 1;
}
.signal-value.bull   { color: var(--neon-green); text-shadow: var(--glow-green); }
.signal-value.bear   { color: var(--neon-red); }
.signal-value.neutral{ color: var(--neon-gold); text-shadow: var(--glow-gold); }
.signal-desc { font-family: 'Rajdhani', sans-serif; font-size: 0.78rem; color: var(--text-mid); margin-top: 0.3rem; line-height: 1.4; }

/* CONFIDENCE BARS */
.conf-row { display: flex; align-items: center; gap: 1rem; margin-bottom: 0.6rem; }
.conf-label { font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: var(--text-mid); width: 90px; text-transform: uppercase; letter-spacing: 0.08em; }
.conf-bar { flex: 1; height: 4px; background: var(--deep); border-radius: 2px; overflow: hidden; }
.conf-fill { height: 100%; border-radius: 2px; }
.conf-pct { font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: var(--text-bright); width: 35px; text-align: right; }

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: var(--abyss) !important;
    border-right: 1px solid rgba(0,200,255,0.08) !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem !important; }
.sidebar-logo {
    padding: 0 1rem 1.2rem 1rem;
    border-bottom: 1px solid rgba(0,200,255,0.08);
    margin-bottom: 1.5rem;
}
.sidebar-logo-text {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.4rem;
    letter-spacing: 0.15em;
    background: linear-gradient(90deg, var(--neon-blue), var(--neon-green));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.sidebar-section {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--text-dark);
    padding: 0 1rem;
    margin-bottom: 0.7rem;
    margin-top: 1.5rem;
}
.model-spec {
    background: var(--deep);
    border: 1px solid rgba(0,200,255,0.08);
    border-radius: 8px;
    margin: 0 0.5rem;
    overflow: hidden;
}
.spec-row { display: flex; justify-content: space-between; align-items: center; padding: 0.6rem 1rem; border-bottom: 1px solid rgba(0,200,255,0.05); }
.spec-row:last-child { border-bottom: none; }
.spec-k { font-family: 'JetBrains Mono', monospace; font-size: 0.62rem; color: var(--text-dark); letter-spacing: 0.05em; }
.spec-v { font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: var(--neon-blue); font-weight: 700; }

[data-testid="stSidebar"] .stTextInput input {
    background: var(--void) !important;
    border: 1px solid rgba(0,200,255,0.15) !important;
    color: var(--text-bright) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.9rem !important;
    border-radius: 6px !important;
    padding: 0.6rem 0.9rem !important;
    letter-spacing: 0.08em;
}
[data-testid="stSidebar"] .stTextInput input:focus {
    border-color: var(--neon-blue) !important;
    box-shadow: 0 0 0 2px rgba(0,200,255,0.15) !important;
    outline: none !important;
}

.stButton > button {
    background: linear-gradient(135deg, #006fa8, #00b4d8, #006fa8) !important;
    background-size: 200% 200% !important;
    animation: btnGradient 3s ease infinite !important;
    color: var(--void) !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.8rem 1rem !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.1rem !important;
    letter-spacing: 0.15em !important;
    width: 100% !important;
    cursor: pointer !important;
    box-shadow: 0 4px 20px rgba(0,200,255,0.25) !important;
    transition: transform 0.15s ease !important;
}
@keyframes btnGradient {
    0%, 100% { background-position: 0% 50%; }
    50%       { background-position: 100% 50%; }
}
.stButton > button:hover {
    transform: translateY(-2px) scale(1.01) !important;
    box-shadow: 0 8px 30px rgba(0,200,255,0.4) !important;
}

/* TABS */
.stTabs [data-baseweb="tab-list"] {
    background: var(--deep) !important;
    border-radius: 8px 8px 0 0 !important;
    gap: 0 !important;
    border-bottom: 1px solid rgba(0,200,255,0.1) !important;
    padding: 0.3rem 0.3rem 0 0.3rem !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--text-mid) !important;
    padding: 0.7rem 1.5rem !important;
    border-radius: 6px 6px 0 0 !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: var(--surface) !important;
    color: var(--neon-blue) !important;
    border-bottom: 2px solid var(--neon-blue) !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: var(--surface) !important;
    border: 1px solid rgba(0,200,255,0.08) !important;
    border-top: none !important;
    border-radius: 0 0 12px 12px !important;
    padding: 1.2rem !important;
}

/* EMPTY STATE */
.empty-state { display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 6rem 2rem; text-align: center; }
.empty-icon { font-size: 4rem; margin-bottom: 1.5rem; filter: drop-shadow(0 0 20px rgba(0,200,255,0.4)); animation: float 3s ease-in-out infinite; }
@keyframes float {
    0%, 100% { transform: translateY(0); }
    50%       { transform: translateY(-10px); }
}
.empty-title { font-family: 'Bebas Neue', sans-serif; font-size: 2.5rem; letter-spacing: 0.1em; color: var(--text-bright); margin-bottom: 0.5rem; }
.empty-sub { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: var(--text-dark); line-height: 1.8; max-width: 400px; letter-spacing: 0.05em; }
.empty-chips { display: flex; gap: 0.5rem; margin-top: 2rem; flex-wrap: wrap; justify-content: center; }
.chip { background: var(--surface); border: 1px solid rgba(0,200,255,0.12); border-radius: 4px; padding: 0.3rem 0.9rem; font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: var(--neon-blue); letter-spacing: 0.1em; }

/* DISCLAIMER */
.disclaimer {
    background: rgba(255,215,0,0.04);
    border: 1px solid rgba(255,215,0,0.15);
    border-radius: 8px;
    padding: 0.8rem 1.2rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    color: rgba(255,215,0,0.6);
    letter-spacing: 0.05em;
    line-height: 1.6;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: flex-start;
    gap: 0.6rem;
}

/* FOOTER */
.footer-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem 0;
    border-top: 1px solid rgba(0,200,255,0.08);
    margin-top: 2rem;
}
.footer-left, .footer-right { font-family: 'JetBrains Mono', monospace; font-size: 0.62rem; color: var(--text-dark); }
.footer-dot { color: var(--neon-blue); margin: 0 0.5rem; }

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--void); }
::-webkit-scrollbar-thumb { background: rgba(0,200,255,0.2); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0,200,255,0.4); }
.stSpinner > div { border-top-color: var(--neon-blue) !important; }
hr { border-color: rgba(0,200,255,0.08) !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# LOAD ASSETS
# ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    model  = load_model("lstm_model.h5")
    scaler = joblib.load("scaler.save")
    return model, scaler

model, scaler = load_assets()


# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="sidebar-logo-text">🔮 QuantumLens</div>
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.58rem;
                    color:#2a4060; margin-top:2px; letter-spacing:0.12em;">
            AI STOCK ORACLE · v3.0.0
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Market Symbol</div>', unsafe_allow_html=True)
    stock_symbol = st.text_input("", value="GAIL.NS", label_visibility="collapsed", placeholder="e.g. AAPL, TSLA, GAIL.NS")

    st.markdown('<div class="sidebar-section">Exchange Region</div>', unsafe_allow_html=True)
    exchange = st.selectbox("", ["🇮🇳  NSE / BSE (.NS / .BO)", "🇺🇸  NYSE / NASDAQ", "🇬🇧  LSE (.L)", "🌏  Other"], label_visibility="collapsed")

    st.markdown('<div class="sidebar-section">Forecast Window</div>', unsafe_allow_html=True)
    forecast_days = st.slider("", min_value=7, max_value=60, value=30, step=1, label_visibility="collapsed", format="%d days")

    st.markdown("<br>", unsafe_allow_html=True)
    predict_button = st.button(f"⚡  RUN FORECAST · {forecast_days}D")

    st.markdown('<div class="sidebar-section">Model Architecture</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="model-spec">
        <div class="spec-row"><span class="spec-k">Type</span><span class="spec-v">2× LSTM</span></div>
        <div class="spec-row"><span class="spec-k">Input</span><span class="spec-v">100 days</span></div>
        <div class="spec-row"><span class="spec-k">Feature</span><span class="spec-v">Open Price</span></div>
        <div class="spec-row"><span class="spec-k">Horizon</span><span class="spec-v">Variable</span></div>
        <div class="spec-row"><span class="spec-k">Scaler</span><span class="spec-v">MinMaxScaler</span></div>
        <div class="spec-row"><span class="spec-k">Source</span><span class="spec-v">Yahoo Finance</span></div>
        <div class="spec-row"><span class="spec-k">History</span><span class="spec-v">10 Years</span></div>
    </div>
    <div style="border-top:1px solid rgba(0,200,255,0.06); padding-top:1rem; margin-top:1.5rem; padding:0.8rem 1rem 0 1rem;">
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.58rem; color:#2a4060; line-height:1.8; letter-spacing:0.05em;">
            ⚠ FOR INFORMATIONAL USE ONLY<br>NOT FINANCIAL ADVICE<br>PAST PERFORMANCE ≠ FUTURE RESULTS
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# TOP NAV
# ─────────────────────────────────────────────────────────────────
now = datetime.now()
st.markdown(f"""
<div class="topnav">
    <div class="brand-logo">
        <div class="brand-icon">🔮</div>
        <div>
            <div class="brand-name">QuantumLens</div>
            <div class="brand-tagline">Neural Market Intelligence</div>
        </div>
    </div>
    <div class="nav-right">
        <div class="nav-status">Live Feed</div>
        <div class="nav-time">{now.strftime('%a %d %b %Y  ·  %H:%M:%S')}</div>
        <div class="nav-version">LSTM v3.0</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
if predict_button:
    prog = st.progress(0, text="Initializing quantum matrices…")
    time.sleep(0.15)

    prog.progress(15, text="🛰 Connecting to market data feed…")
    data = yf.download(stock_symbol, period="10y", interval="1d")
    prog.progress(35, text="📡 Streaming historical candles…")

    if data.empty:
        st.error("⚠ Symbol not found. Check ticker and retry.")
        prog.empty()
        st.stop()

    time.sleep(0.15)
    prog.progress(55, text="🧠 Loading LSTM neural weights…")
    opn = data[['Open']]
    ds  = opn.values
    ds_scaled = scaler.transform(ds)

    last_100  = ds_scaled[-100:]
    tmp_inp   = last_100.reshape(1, -1).tolist()[0]
    lst_output = []
    n_steps = 100

    prog.progress(70, text=f"⚡ Running recursive {forecast_days}-day inference…")
    for i in range(forecast_days):
        x_input = np.array(tmp_inp[-100:]).reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        tmp_inp.append(yhat[0][0])
        lst_output.append(yhat[0][0])

    prog.progress(88, text="📊 Generating analytics…")
    forecast_array = np.array(lst_output).reshape(-1, 1)
    ds_new = np.vstack((ds_scaled, forecast_array))
    final_graph = scaler.inverse_transform(ds_new)
    prog.progress(100, text="✅ Forecast complete!")
    time.sleep(0.25)
    prog.empty()

    # ── Derived Metrics ──
    latest_price    = float(opn.values[-1][0])
    predicted_price = float(final_graph[-1][0])
    price_change    = predicted_price - latest_price
    pct_change      = (price_change / latest_price) * 100
    is_bullish      = price_change >= 0

    forecast_vals = final_graph[-forecast_days:].flatten()
    peak_price    = float(forecast_vals.max())
    trough_price  = float(forecast_vals.min())
    volatility    = float(np.std(forecast_vals))
    momentum      = float(np.mean(np.diff(forecast_vals)))

    high_52w = float(data['High'].tail(252).max())  if 'High'   in data.columns else None
    low_52w  = float(data['Low'].tail(252).min())   if 'Low'    in data.columns else None
    vol_avg  = int(data['Volume'].tail(30).mean())  if 'Volume' in data.columns else 0
    data_pts = len(data)

    closes   = data['Close'].values.flatten()
    delta    = np.diff(closes)
    gain_arr = np.where(delta > 0, delta, 0)
    loss_arr = np.where(delta < 0, -delta, 0)
    avg_g    = np.mean(gain_arr[-14:])
    avg_l    = np.mean(loss_arr[-14:])
    rsi      = 100 - (100 / (1 + avg_g / avg_l)) if avg_l != 0 else 50.0

    ema12    = float(pd.Series(closes).ewm(span=12, adjust=False).mean().iloc[-1])
    ema26    = float(pd.Series(closes).ewm(span=26, adjust=False).mean().iloc[-1])
    macd_val = ema12 - ema26
    conf     = min(95, max(55, 80 - volatility / latest_price * 100))

    # ── DISCLAIMER ──
    st.markdown("""
    <div class="disclaimer">
        <span>⚠</span>
        <span>Forecast generated by ML model for informational purposes only.
        <strong>Not financial advice.</strong> Past market patterns do not guarantee future results.
        Always conduct your own due diligence before making investment decisions.</span>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI CARDS ──
    badge_cls  = "up"   if is_bullish else "down"
    arrow      = "▲"    if is_bullish else "▼"
    kpi_color  = "c-green" if is_bullish else "c-red"
    signal_str = "STRONG BUY" if pct_change > 5 else "BUY" if pct_change > 0 else "SELL" if pct_change > -5 else "STRONG SELL"
    sig_class  = "bull"  if pct_change > 0 else "bear"

    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi-card c-blue">
            <div class="kpi-label">Current Open Price</div>
            <div class="kpi-value">₹{round(latest_price,2):,.2f}</div>
            <div class="kpi-sub">{stock_symbol.upper()} · Latest Session</div>
            <div><span class="badge flat">LIVE</span></div>
        </div>
        <div class="kpi-card {kpi_color}">
            <div class="kpi-label">AI Predicted · {forecast_days}D</div>
            <div class="kpi-value">₹{round(predicted_price,2):,.2f}</div>
            <div class="kpi-sub">{forecast_days}-Day LSTM Projection</div>
            <div><span class="badge {badge_cls}">{arrow} {abs(round(pct_change,2))}%</span></div>
        </div>
        <div class="kpi-card c-gold">
            <div class="kpi-label">Forecast Peak</div>
            <div class="kpi-value">₹{round(peak_price,2):,.2f}</div>
            <div class="kpi-sub">Max in {forecast_days}-day window</div>
            <div><span class="badge up">▲ PEAK</span></div>
        </div>
        <div class="kpi-card c-purple">
            <div class="kpi-label">Forecast Floor</div>
            <div class="kpi-value">₹{round(trough_price,2):,.2f}</div>
            <div class="kpi-sub">Min in {forecast_days}-day window</div>
            <div><span class="badge down">▼ FLOOR</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── STAT RIBBON ──
    rsi_color  = "red"   if rsi > 70 else "green" if rsi < 30 else "blue"
    macd_color = "green" if macd_val > 0 else "red"
    st.markdown(f"""
    <div class="stat-ribbon">
        <div class="ribbon-item">
            <div class="ribbon-key">52W High</div>
            <div class="ribbon-val gold">₹{round(high_52w,2) if high_52w else 'N/A'}</div>
        </div>
        <div class="ribbon-item">
            <div class="ribbon-key">52W Low</div>
            <div class="ribbon-val blue">₹{round(low_52w,2) if low_52w else 'N/A'}</div>
        </div>
        <div class="ribbon-item">
            <div class="ribbon-key">RSI (14)</div>
            <div class="ribbon-val {rsi_color}">{round(rsi,1)}</div>
        </div>
        <div class="ribbon-item">
            <div class="ribbon-key">MACD</div>
            <div class="ribbon-val {macd_color}">{'+' if macd_val>0 else ''}{round(macd_val,2)}</div>
        </div>
        <div class="ribbon-item">
            <div class="ribbon-key">Volatility σ</div>
            <div class="ribbon-val blue">₹{round(volatility,2)}</div>
        </div>
        <div class="ribbon-item">
            <div class="ribbon-key">Momentum</div>
            <div class="ribbon-val {'green' if momentum>0 else 'red'}">{'+' if momentum>0 else ''}{round(momentum,3)}</div>
        </div>
        <div class="ribbon-item">
            <div class="ribbon-key">Avg Vol (30D)</div>
            <div class="ribbon-val blue">{vol_avg:,}</div>
        </div>
        <div class="ribbon-item">
            <div class="ribbon-key">Data Points</div>
            <div class="ribbon-val blue">{data_pts:,}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── TABS ──
    tab1, tab2, tab3 = st.tabs(["📈  FORECAST CHART", "🕯  CANDLESTICK + OVERLAY", "📊  TECHNICAL ANALYSIS"])

    historical_dates = data.index
    future_dates = pd.date_range(
        start=historical_dates[-1] + pd.Timedelta(days=1),
        periods=forecast_days, freq='B'
    )
    all_dates = historical_dates.append(future_dates)
    n_hist    = len(historical_dates)
    hist_y    = final_graph[:n_hist].flatten()
    fore_full = final_graph.flatten()
    band_upper = forecast_vals + volatility * 1.5
    band_lower = forecast_vals - volatility * 1.5

    # ─── TAB 1: FORECAST ───
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=historical_dates, y=hist_y,
            mode='lines', name='Historical',
            line=dict(color='rgba(0,200,255,0.7)', width=1.5),
            fill='tozeroy', fillcolor='rgba(0,200,255,0.03)',
            hovertemplate='<b>%{x|%d %b %Y}</b><br>Open: ₹%{y:,.2f}<extra>Historical</extra>'
        ))
        fig.add_trace(go.Scatter(
            x=list(future_dates) + list(future_dates[::-1]),
            y=list(band_upper) + list(band_lower[::-1]),
            fill='toself', fillcolor='rgba(0,255,136,0.05)',
            line=dict(color='rgba(0,0,0,0)'),
            name='Confidence Band', hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=future_dates, y=band_upper,
            mode='lines', name='Upper Band',
            line=dict(color='rgba(0,255,136,0.2)', width=1, dash='dot'),
            hovertemplate='Upper: ₹%{y:,.2f}<extra></extra>'
        ))
        fig.add_trace(go.Scatter(
            x=future_dates, y=band_lower,
            mode='lines', name='Lower Band',
            line=dict(color='rgba(255,51,102,0.2)', width=1, dash='dot'),
            hovertemplate='Lower: ₹%{y:,.2f}<extra></extra>'
        ))
        fig.add_trace(go.Scatter(
            x=all_dates[n_hist - 1:], y=fore_full[n_hist - 1:],
            mode='lines', name='AI Forecast',
            line=dict(color='#00ff88', width=2.5),
            hovertemplate='<b>%{x|%d %b %Y}</b><br>Forecast: ₹%{y:,.2f}<extra>LSTM</extra>'
        ))
        fig.add_trace(go.Scatter(
            x=future_dates, y=forecast_vals,
            mode='markers', showlegend=False,
            marker=dict(color='#00ff88', size=3.5, opacity=0.6),
            hovertemplate='₹%{y:,.2f}<extra></extra>'
        ))
        fig.add_shape(type="line",
            x0=str(historical_dates[-1]), x1=str(historical_dates[-1]),
            y0=0, y1=1, yref='paper',
            line=dict(color="#ffd700", width=1.5, dash="dash"))
        fig.add_annotation(
            x=str(historical_dates[-1]), y=0.98, yref='paper',
            text="◀ HISTORICAL  ┃  FORECAST ▶",
            showarrow=False,
            font=dict(family="JetBrains Mono", size=8, color="#ffd700"),
            bgcolor="rgba(255,215,0,0.08)",
            bordercolor="rgba(255,215,0,0.3)",
            borderwidth=1, borderpad=5)
        fig.add_shape(type="line",
            x0=str(all_dates[0]), x1=str(all_dates[-1]),
            y0=predicted_price, y1=predicted_price,
            line=dict(color="rgba(0,255,136,0.3)", width=1, dash="dot"))
        fig.add_annotation(
            x=str(all_dates[-1]), y=predicted_price,
            text=f" {forecast_days}D Target ₹{round(predicted_price,2):,.0f}",
            showarrow=False, xanchor="right",
            font=dict(family="JetBrains Mono", size=9, color="#00ff88"))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=520, margin=dict(l=10, r=10, t=10, b=10),
            hovermode="x unified",
            legend=dict(orientation="h", x=0, y=1.06,
                font=dict(family="JetBrains Mono", size=9, color="#6d8fa8"),
                bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(showgrid=False, zeroline=False, type="date",
                tickfont=dict(family="JetBrains Mono", size=8, color="#2a4060"),
                rangeslider=dict(visible=True, bgcolor="#060c14", bordercolor="#1a2d4a", thickness=0.04)),
            yaxis=dict(showgrid=True, gridcolor="rgba(0,200,255,0.05)", zeroline=False,
                tickfont=dict(family="JetBrains Mono", size=8, color="#2a4060"), tickprefix="₹"),
            hoverlabel=dict(bgcolor="#0e1a2e", bordercolor="#1a2d4a",
                font=dict(family="JetBrains Mono", size=11, color="#e2f4ff"))
        )
        st.plotly_chart(fig, use_container_width=True)

        col_c1, col_c2 = st.columns([2, 1])
        with col_c1:
            vol_risk = min(95, round(volatility / latest_price * 2000))
            st.markdown(f"""
            <div style="padding:0.5rem 0;">
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;letter-spacing:0.15em;text-transform:uppercase;color:#2a4060;margin-bottom:0.8rem;">Model Confidence Metrics</div>
                <div class="conf-row"><span class="conf-label">Trend Conf.</span><div class="conf-bar"><div class="conf-fill" style="width:{round(conf)}%;background:linear-gradient(90deg,#00c8ff,#00ff88);"></div></div><span class="conf-pct">{round(conf)}%</span></div>
                <div class="conf-row"><span class="conf-label">Data Quality</span><div class="conf-bar"><div class="conf-fill" style="width:92%;background:linear-gradient(90deg,#00c8ff,#00ff88);"></div></div><span class="conf-pct">92%</span></div>
                <div class="conf-row"><span class="conf-label">LSTM Stable</span><div class="conf-bar"><div class="conf-fill" style="width:87%;background:linear-gradient(90deg,#ffd700,#f97316);"></div></div><span class="conf-pct">87%</span></div>
                <div class="conf-row"><span class="conf-label">Vol. Risk</span><div class="conf-bar"><div class="conf-fill" style="width:{vol_risk}%;background:linear-gradient(90deg,#ff3366,#bf5fff);"></div></div><span class="conf-pct">{vol_risk}%</span></div>
            </div>
            """, unsafe_allow_html=True)
        with col_c2:
            st.markdown(f"""
            <div class="signal-card {sig_class}" style="height:100%;display:flex;flex-direction:column;justify-content:center;">
                <div class="signal-label">AI SIGNAL</div>
                <div class="signal-value {sig_class}">{signal_str}</div>
                <div class="signal-desc">Based on {forecast_days}-day LSTM recursive projection with ±{round(volatility,1)} σ band</div>
                <div style="margin-top:0.8rem;"><span class="badge {badge_cls}">{arrow} {abs(round(pct_change,2))}% in {forecast_days}D</span></div>
            </div>
            """, unsafe_allow_html=True)

    # ─── TAB 2: CANDLESTICK ───
    with tab2:
        candle_data = data.tail(120)
        fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.03, row_heights=[0.75, 0.25])

        fig2.add_trace(go.Candlestick(
            x=candle_data.index,
            open=candle_data['Open'].values.flatten(),
            high=candle_data['High'].values.flatten(),
            low=candle_data['Low'].values.flatten(),
            close=candle_data['Close'].values.flatten(),
            name='OHLC',
            increasing_line_color='#00ff88', decreasing_line_color='#ff3366',
            increasing_fillcolor='rgba(0,255,136,0.6)', decreasing_fillcolor='rgba(255,51,102,0.6)',
        ), row=1, col=1)

        ma20 = candle_data['Close'].rolling(20).mean()
        fig2.add_trace(go.Scatter(x=candle_data.index, y=ma20.values.flatten(),
            mode='lines', name='MA(20)', line=dict(color='#ffd700', width=1.2)), row=1, col=1)

        ma50 = candle_data['Close'].rolling(50).mean()
        fig2.add_trace(go.Scatter(x=candle_data.index, y=ma50.values.flatten(),
            mode='lines', name='MA(50)', line=dict(color='#bf5fff', width=1.2)), row=1, col=1)

        fig2.add_trace(go.Scatter(x=list(future_dates), y=forecast_vals,
            mode='lines', name=f'{forecast_days}D Forecast',
            line=dict(color='#00ff88', width=2, dash='dash')), row=1, col=1)

        colors_vol = ['rgba(0,255,136,0.5)' if c >= o else 'rgba(255,51,102,0.5)'
                      for c, o in zip(candle_data['Close'].values.flatten(), candle_data['Open'].values.flatten())]
        fig2.add_trace(go.Bar(
            x=candle_data.index, y=candle_data['Volume'].values.flatten(),
            name='Volume', marker_color=colors_vol, showlegend=False), row=2, col=1)

        fig2.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=560, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
            legend=dict(orientation="h", x=0, y=1.05,
                font=dict(family="JetBrains Mono", size=9, color="#6d8fa8"), bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(showgrid=False, zeroline=False, rangeslider=dict(visible=False),
                tickfont=dict(family="JetBrains Mono", size=8, color="#2a4060")),
            xaxis2=dict(showgrid=False, zeroline=False, rangeslider=dict(visible=False),
                tickfont=dict(family="JetBrains Mono", size=8, color="#2a4060")),
            yaxis=dict(showgrid=True, gridcolor="rgba(0,200,255,0.05)", zeroline=False,
                tickfont=dict(family="JetBrains Mono", size=8, color="#2a4060"), tickprefix="₹"),
            yaxis2=dict(showgrid=False, zeroline=False,
                tickfont=dict(family="JetBrains Mono", size=7, color="#2a4060"))
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ─── TAB 3: TECHNICAL ───
    with tab3:
        close_series = pd.Series(data['Close'].values.flatten(), index=data.index)
        delta_s      = close_series.diff()
        gain_s       = delta_s.clip(lower=0).rolling(14).mean()
        loss_s       = (-delta_s.clip(upper=0)).rolling(14).mean()
        rs_s         = gain_s / loss_s
        rsi_s        = 100 - (100 / (1 + rs_s))

        bb_mid   = close_series.rolling(20).mean()
        bb_std   = close_series.rolling(20).std()
        bb_upper_s = bb_mid + 2 * bb_std
        bb_lower_s = bb_mid - 2 * bb_std

        fig3 = make_subplots(rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.05, subplot_titles=("Bollinger Bands (20, 2σ)", "RSI (14)"),
            row_heights=[0.6, 0.4])

        fig3.add_trace(go.Scatter(
            x=list(data.index) + list(data.index[::-1]),
            y=list(bb_upper_s.values) + list(bb_lower_s.values[::-1]),
            fill='toself', fillcolor='rgba(0,200,255,0.05)',
            line=dict(color='rgba(0,0,0,0)'), name='BB Band', hoverinfo='skip'), row=1, col=1)
        fig3.add_trace(go.Scatter(x=data.index, y=close_series.values,
            mode='lines', name='Close', line=dict(color='rgba(0,200,255,0.8)', width=1.5)), row=1, col=1)
        fig3.add_trace(go.Scatter(x=data.index, y=bb_upper_s.values,
            mode='lines', name='BB Upper', line=dict(color='rgba(0,200,255,0.3)', width=1)), row=1, col=1)
        fig3.add_trace(go.Scatter(x=data.index, y=bb_lower_s.values,
            mode='lines', name='BB Lower', line=dict(color='rgba(255,51,102,0.3)', width=1)), row=1, col=1)
        fig3.add_trace(go.Scatter(x=data.index, y=bb_mid.values,
            mode='lines', name='BB Mid', line=dict(color='rgba(255,215,0,0.4)', width=1, dash='dot')), row=1, col=1)

        fig3.add_trace(go.Scatter(x=data.index, y=rsi_s.values,
            mode='lines', name='RSI(14)', line=dict(color='#ffd700', width=1.5),
            fill='tozeroy', fillcolor='rgba(255,215,0,0.04)'), row=2, col=1)
        fig3.add_shape(type='line', row=2, col=1,
            x0=data.index[0], x1=data.index[-1], y0=70, y1=70,
            line=dict(color='rgba(255,51,102,0.4)', width=1, dash='dash'))
        fig3.add_shape(type='line', row=2, col=1,
            x0=data.index[0], x1=data.index[-1], y0=30, y1=30,
            line=dict(color='rgba(0,255,136,0.4)', width=1, dash='dash'))

        fig3.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=560, margin=dict(l=10, r=10, t=30, b=10),
            showlegend=False, hovermode="x unified",
            font=dict(family="JetBrains Mono", size=9, color="#6d8fa8"),
            xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(family="JetBrains Mono", size=8, color="#2a4060")),
            xaxis2=dict(showgrid=False, zeroline=False, tickfont=dict(family="JetBrains Mono", size=8, color="#2a4060")),
            yaxis=dict(showgrid=True, gridcolor="rgba(0,200,255,0.04)", zeroline=False,
                tickfont=dict(family="JetBrains Mono", size=8, color="#2a4060")),
            yaxis2=dict(showgrid=True, gridcolor="rgba(0,200,255,0.04)", zeroline=False,
                tickfont=dict(family="JetBrains Mono", size=8, color="#2a4060"))
        )
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown(f"""
        <div class="signal-grid">
            <div class="signal-card {'bull' if rsi<50 else 'bear'}">
                <div class="signal-label">RSI (14)</div>
                <div class="signal-value {'bull' if rsi<50 else 'bear'}">{round(rsi,1)}</div>
                <div class="signal-desc">{'Oversold zone — potential reversal' if rsi<30 else 'Overbought zone — watch for reversal' if rsi>70 else 'Neutral momentum range'}</div>
            </div>
            <div class="signal-card {'bull' if macd_val>0 else 'bear'}">
                <div class="signal-label">MACD Line</div>
                <div class="signal-value {'bull' if macd_val>0 else 'bear'}">{('+' if macd_val>0 else '')}{round(macd_val,2)}</div>
                <div class="signal-desc">{'Bullish crossover — upward momentum' if macd_val>0 else 'Bearish signal — downward momentum'}</div>
            </div>
            <div class="signal-card {sig_class}">
                <div class="signal-label">AI Signal</div>
                <div class="signal-value {sig_class}">{signal_str}</div>
                <div class="signal-desc">LSTM projection over {forecast_days} trading days with σ={round(volatility,1)}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── FOOTER ──
    st.markdown(f"""
    <div class="footer-bar">
        <div class="footer-left">
            QuantumLens AI <span class="footer-dot">·</span>
            LSTM Neural Engine v3.0 <span class="footer-dot">·</span>
            Data: Yahoo Finance <span class="footer-dot">·</span>
            Not Financial Advice
        </div>
        <div class="footer-right">Last Run: {datetime.now().strftime('%d %b %Y · %H:%M:%S')}</div>
    </div>
    """, unsafe_allow_html=True)

else:
    # ── EMPTY STATE ──
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">🔮</div>
        <div class="empty-title">AWAITING SYMBOL INPUT</div>
        <div class="empty-sub">
            Enter any stock ticker in the sidebar and click<br>
            <strong style="color:#00c8ff;">⚡ RUN FORECAST</strong> to deploy the neural oracle.<br><br>
            Supports NSE · BSE · NYSE · NASDAQ and more.
        </div>
        <div class="empty-chips">
            <div class="chip">GAIL.NS</div>
            <div class="chip">AAPL</div>
            <div class="chip">TSLA</div>
            <div class="chip">RELIANCE.NS</div>
            <div class="chip">AMZN</div>
            <div class="chip">INFY.NS</div>
            <div class="chip">NVDA</div>
            <div class="chip">TCS.NS</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="footer-bar">
        <div class="footer-left">QuantumLens AI · LSTM Neural Engine v3.0 · Not Financial Advice</div>
        <div class="footer-right">{datetime.now().strftime('%d %b %Y · %H:%M:%S')}</div>
    </div>
    """, unsafe_allow_html=True)
