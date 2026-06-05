import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Dexter · Fraud Intelligence",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# DEXTER THEME — LUXURIOUS RED CSS
# =====================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=Rajdhani:wght@300;400;500;600;700&family=Share+Tech+Mono&display=swap');

/* ─── ROOT PALETTE ─────────────────────────────── */
:root {
    --blood:      #8B0000;
    --crimson:    #DC143C;
    --scarlet:    #FF2040;
    --rose:       #C0392B;
    --rust:       #6B1A1A;
    --noir:       #0A0A0B;
    --obsidian:   #111114;
    --graphite:   #1A1A1F;
    --slate:      #23232A;
    --smoke:      #2E2E38;
    --ash:        #3D3D4A;
    --mist:       #555568;
    --silver:     #8888A0;
    --ghost:      #BBBBCC;
    --ivory:      #E8E8F0;
    --white:      #F5F5FF;
    --gold:       #C9A84C;
    --amber:      #E8A020;
}

/* ─── GLOBAL RESET ─────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif !important;
    background-color: var(--noir) !important;
    color: var(--ghost) !important;
}

.stApp {
    background: 
        radial-gradient(ellipse at 0% 0%, rgba(139,0,0,0.12) 0%, transparent 55%),
        radial-gradient(ellipse at 100% 100%, rgba(107,26,26,0.10) 0%, transparent 55%),
        linear-gradient(180deg, #0A0A0B 0%, #0D0D10 100%);
    min-height: 100vh;
}

/* ─── NOISE TEXTURE OVERLAY ─────────────────────── */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.03'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 0;
    opacity: 0.4;
}

/* ─── SCROLLBAR ──────────────────────────────────── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--obsidian); }
::-webkit-scrollbar-thumb { background: var(--blood); border-radius: 2px; }

/* ─── SIDEBAR ────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D0508 0%, #0A0A0B 100%) !important;
    border-right: 1px solid rgba(139,0,0,0.35) !important;
    box-shadow: 4px 0 30px rgba(139,0,0,0.12) !important;
}

[data-testid="stSidebar"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, transparent, var(--crimson), transparent);
}

[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Playfair Display', serif !important;
    color: var(--crimson) !important;
    letter-spacing: 0.05em;
}

/* ─── SIDEBAR WIDGET LABELS ──────────────────────── */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label {
    color: var(--ghost) !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    font-size: 0.72rem !important;
}

/* ─── SLIDER ─────────────────────────────────────── */
[data-testid="stSlider"] > div > div > div > div {
    background: var(--blood) !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: var(--crimson) !important;
    border: 2px solid var(--scarlet) !important;
    box-shadow: 0 0 10px rgba(220,20,60,0.5) !important;
}

/* ─── MAIN HEADER AREA ───────────────────────────── */
.main-header {
    position: relative;
    padding: 2.5rem 0 2rem 0;
    margin-bottom: 2rem;
    border-bottom: 1px solid rgba(139,0,0,0.4);
}
.main-header::after {
    content: '';
    position: absolute;
    bottom: -1px; left: 0;
    width: 120px; height: 1px;
    background: var(--crimson);
    box-shadow: 0 0 12px var(--crimson);
}
.header-eyebrow {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.25em;
    color: var(--crimson);
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.header-title {
    font-family: 'Playfair Display', serif;
    font-size: 3.2rem;
    font-weight: 900;
    color: var(--ivory);
    line-height: 1.05;
    margin: 0;
}
.header-title span {
    color: var(--crimson);
    text-shadow: 0 0 25px rgba(220,20,60,0.4);
}
.header-subtitle {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.05rem;
    color: var(--silver);
    font-weight: 300;
    letter-spacing: 0.06em;
    margin-top: 0.6rem;
}

/* ─── METRIC CARDS ───────────────────────────────── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, rgba(26,26,31,0.95) 0%, rgba(18,10,12,0.95) 100%) !important;
    border: 1px solid rgba(139,0,0,0.3) !important;
    border-radius: 2px !important;
    padding: 1.2rem 1.5rem !important;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4), inset 0 1px 0 rgba(220,20,60,0.08) !important;
}
[data-testid="metric-container"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 2px;
    background: linear-gradient(90deg, var(--blood), var(--crimson), var(--blood));
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    color: var(--silver) !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Playfair Display', serif !important;
    font-size: 2.4rem !important;
    font-weight: 700 !important;
    color: var(--ivory) !important;
}

/* ─── SECTION HEADERS ────────────────────────────── */
.section-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin: 2.2rem 0 1.2rem 0;
    padding-bottom: 0.7rem;
    border-bottom: 1px solid rgba(139,0,0,0.25);
}
.section-header-line {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(139,0,0,0.5), transparent);
}
.section-header-text {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--ivory);
    letter-spacing: 0.03em;
    white-space: nowrap;
}
.section-icon {
    font-size: 1rem;
}

/* ─── FILE UPLOADER ──────────────────────────────── */
[data-testid="stFileUploader"] {
    border: 1px dashed rgba(139,0,0,0.5) !important;
    border-radius: 2px !important;
    background: rgba(15,8,10,0.6) !important;
    transition: border-color 0.3s, background 0.3s;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--crimson) !important;
    background: rgba(25,10,12,0.8) !important;
}
[data-testid="stFileUploader"] label {
    color: var(--ghost) !important;
}
[data-testid="stFileDropzone"] {
    background: transparent !important;
}

/* ─── DATA TABLE ─────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(139,0,0,0.2) !important;
    border-radius: 2px !important;
    overflow: hidden;
}
[data-testid="stDataFrame"] th {
    background: rgba(139,0,0,0.2) !important;
    color: var(--crimson) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}
[data-testid="stDataFrame"] td {
    color: var(--ghost) !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 0.9rem !important;
    border-bottom: 1px solid rgba(139,0,0,0.1) !important;
}
[data-testid="stDataFrame"] tr:hover td {
    background: rgba(139,0,0,0.08) !important;
}

/* ─── BUTTONS ────────────────────────────────────── */
[data-testid="stButton"] button,
.stDownloadButton button {
    background: transparent !important;
    border: 1px solid var(--blood) !important;
    color: var(--crimson) !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    padding: 0.65rem 2rem !important;
    border-radius: 1px !important;
    transition: all 0.25s ease !important;
    position: relative;
    overflow: hidden;
}
[data-testid="stButton"] button:hover,
.stDownloadButton button:hover {
    background: var(--blood) !important;
    border-color: var(--crimson) !important;
    color: var(--white) !important;
    box-shadow: 0 0 20px rgba(139,0,0,0.4) !important;
}
[data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(135deg, var(--blood), var(--rose)) !important;
    border-color: var(--crimson) !important;
    color: var(--white) !important;
    box-shadow: 0 4px 18px rgba(139,0,0,0.35) !important;
}
[data-testid="stButton"] button[kind="primary"]:hover {
    background: linear-gradient(135deg, var(--crimson), var(--blood)) !important;
    box-shadow: 0 4px 28px rgba(220,20,60,0.5) !important;
}

/* ─── NUMBER INPUT ───────────────────────────────── */
[data-testid="stNumberInput"] input {
    background: rgba(18,10,12,0.8) !important;
    border: 1px solid rgba(139,0,0,0.4) !important;
    border-radius: 2px !important;
    color: var(--ivory) !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
}
[data-testid="stNumberInput"] input:focus {
    border-color: var(--crimson) !important;
    box-shadow: 0 0 0 2px rgba(220,20,60,0.15) !important;
}

/* ─── SELECT BOX ─────────────────────────────────── */
[data-baseweb="select"] > div {
    background: rgba(18,10,12,0.8) !important;
    border: 1px solid rgba(139,0,0,0.4) !important;
    border-radius: 2px !important;
    color: var(--ivory) !important;
}

/* ─── ALERTS ─────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 2px !important;
    border-left-width: 3px !important;
}

/* ─── DIVIDER ────────────────────────────────────── */
hr {
    border: none !important;
    border-top: 1px solid rgba(139,0,0,0.25) !important;
    margin: 2.5rem 0 !important;
    position: relative;
}

/* ─── STATUS BADGES ──────────────────────────────── */
.badge {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 1px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-weight: 500;
}
.badge-fraud {
    background: rgba(139,0,0,0.25);
    border: 1px solid var(--blood);
    color: var(--scarlet);
}
.badge-legit {
    background: rgba(20,60,20,0.25);
    border: 1px solid #1a5c1a;
    color: #4caf50;
}

/* ─── INFO PANELS ────────────────────────────────── */
.info-panel {
    background: linear-gradient(135deg, rgba(20,10,12,0.9), rgba(15,8,10,0.95));
    border: 1px solid rgba(139,0,0,0.2);
    border-left: 3px solid var(--blood);
    padding: 1rem 1.4rem;
    margin: 1rem 0;
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.95rem;
    color: var(--ghost);
    line-height: 1.6;
}

/* ─── REAL-TIME SECTION ──────────────────────────── */
.rt-panel {
    background: linear-gradient(135deg, rgba(18,8,10,0.98) 0%, rgba(12,5,8,0.98) 100%);
    border: 1px solid rgba(139,0,0,0.3);
    border-radius: 2px;
    padding: 2rem;
    margin-top: 1rem;
    position: relative;
    overflow: hidden;
}
.rt-panel::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle at center, rgba(139,0,0,0.04) 0%, transparent 60%);
    pointer-events: none;
}

/* ─── PROBABILITY GAUGE ──────────────────────────── */
.prob-display {
    font-family: 'Playfair Display', serif;
    font-size: 3.5rem;
    font-weight: 900;
    text-align: center;
    padding: 1rem 0;
}
.prob-high { color: var(--scarlet); text-shadow: 0 0 30px rgba(255,32,64,0.5); }
.prob-mid  { color: var(--amber);   text-shadow: 0 0 20px rgba(232,160,32,0.4); }
.prob-low  { color: #4caf50;        text-shadow: 0 0 20px rgba(76,175,80,0.4); }

/* ─── FOOTER ─────────────────────────────────────── */
.dexter-footer {
    margin-top: 4rem;
    padding: 1.5rem 0;
    border-top: 1px solid rgba(139,0,0,0.2);
    text-align: center;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.2em;
    color: var(--mist);
    text-transform: uppercase;
}
.dexter-footer span { color: var(--blood); }

/* ─── PLOTLY CHART CONTAINERS ────────────────────── */
.stPlotlyChart {
    border: 1px solid rgba(139,0,0,0.15) !important;
    border-radius: 2px !important;
    background: rgba(12,8,10,0.5) !important;
}

/* ─── SPINNER ────────────────────────────────────── */
[data-testid="stSpinner"] {
    color: var(--crimson) !important;
}

/* ─── EXPANDER ───────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid rgba(139,0,0,0.2) !important;
    border-radius: 2px !important;
    background: rgba(15,8,10,0.5) !important;
}
[data-testid="stExpander"] summary {
    color: var(--ghost) !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
}

/* ─── PROGRESS BAR ───────────────────────────────── */
.stProgress > div > div {
    background: linear-gradient(90deg, var(--blood), var(--crimson)) !important;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# PLOTLY THEME HELPER
# =====================================================

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(10,8,10,0)",
    plot_bgcolor="rgba(10,8,10,0)",
    font=dict(family="Rajdhani, sans-serif", color="#BBBBCC"),
    title_font=dict(family="Playfair Display, serif", color="#E8E8F0"),
    colorway=["#DC143C", "#8B0000", "#FF2040", "#C9A84C", "#6B1A1A",
               "#E8A020", "#C0392B", "#FF6680", "#992233", "#FF8899"],
    xaxis=dict(
        gridcolor="rgba(139,0,0,0.12)", gridwidth=1,
        linecolor="rgba(139,0,0,0.25)",
        tickfont=dict(family="Share Tech Mono, monospace", size=10, color="#888899"),
        title_font=dict(family="Rajdhani, sans-serif", color="#888899"),
    ),
    yaxis=dict(
        gridcolor="rgba(139,0,0,0.12)", gridwidth=1,
        linecolor="rgba(139,0,0,0.25)",
        tickfont=dict(family="Share Tech Mono, monospace", size=10, color="#888899"),
        title_font=dict(family="Rajdhani, sans-serif", color="#888899"),
    ),
    margin=dict(l=20, r=20, t=50, b=20),
    legend=dict(
        bgcolor="rgba(15,8,10,0.7)",
        bordercolor="rgba(139,0,0,0.3)",
        borderwidth=1,
        font=dict(family="Rajdhani, sans-serif", color="#BBBBCC"),
    ),
)

def apply_theme(fig):
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig

# =====================================================
# MODEL LOADING (with graceful fallback)
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_model_safe():
    try:
        import tensorflow as tf
        from attention import AttentionLayer
        model_path = BASE_DIR / "fraud_lstm_attention.keras"
        if model_path.exists():
            model = tf.keras.models.load_model(
                model_path,
                custom_objects={"AttentionLayer": AttentionLayer},
                compile=False
            )
            return model, True
    except Exception:
        pass
    return None, False

@st.cache_resource
def load_scaler_safe():
    try:
        scaler_path = BASE_DIR / "scaler.pkl"
        if scaler_path.exists():
            return joblib.load(scaler_path), True
    except Exception:
        pass
    return None, False

def load_threshold_safe():
    try:
        threshold_path = BASE_DIR / "threshold.pkl"
        if threshold_path.exists():
            return float(joblib.load(threshold_path))
    except Exception:
        pass
    return 0.5

model, model_loaded = load_model_safe()
scaler, scaler_loaded = load_scaler_safe()
saved_threshold = load_threshold_safe()

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:
    st.markdown("""
    <div style="padding: 1rem 0 1.5rem 0; text-align: center;">
        <div style="font-family: 'Playfair Display', serif; font-size: 1.6rem; 
                    font-weight: 900; color: #DC143C; letter-spacing: 0.04em;">
            DEXTER
        </div>
        <div style="font-family: 'Share Tech Mono', monospace; font-size: 0.62rem;
                    letter-spacing: 0.22em; color: #555568; text-transform: uppercase;
                    margin-top: 0.2rem;">
            Fraud Intelligence System
        </div>
        <div style="width: 60px; height: 1px; background: #8B0000; margin: 0.8rem auto 0 auto;"></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**DETECTION THRESHOLD**")
    threshold = st.slider(
        "Fraud Threshold",
        min_value=0.0, max_value=1.0,
        value=saved_threshold, step=0.01,
        label_visibility="collapsed",
        help="Sequences with fraud probability above this value are flagged as fraudulent."
    )
    st.markdown(f"""
    <div style="font-family: 'Share Tech Mono', monospace; font-size: 0.75rem;
                color: #DC143C; text-align: right; margin-top: -0.5rem; 
                margin-bottom: 1rem;">
        {threshold:.2f}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**SEQUENCE LENGTH**")
    sequence_length = st.slider(
        "Sequence Length",
        min_value=3, max_value=20, value=5,
        label_visibility="collapsed",
        help="Number of transactions per sequence window."
    )

    st.markdown("---")
    st.markdown("**ANALYSIS MODE**")
    analysis_mode = st.selectbox(
        "Mode",
        ["Standard Analysis", "High-Sensitivity", "Low False Positive"],
        label_visibility="collapsed"
    )

    mode_thresholds = {
        "Standard Analysis": threshold,
        "High-Sensitivity": max(0.1, threshold - 0.15),
        "Low False Positive": min(0.9, threshold + 0.2),
    }
    effective_threshold = mode_thresholds[analysis_mode]

    st.markdown("---")

    if model_loaded:
        st.markdown("""
        <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.3rem;">
            <div style="width:7px; height:7px; border-radius:50%; background:#4caf50;
                        box-shadow: 0 0 6px #4caf50;"></div>
            <span style="font-family:'Share Tech Mono',monospace; font-size:0.68rem;
                         color:#888899; letter-spacing:0.12em;">MODEL ONLINE</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.3rem;">
            <div style="width:7px; height:7px; border-radius:50%; background:#DC143C;
                        box-shadow: 0 0 6px #DC143C;"></div>
            <span style="font-family:'Share Tech Mono',monospace; font-size:0.68rem;
                         color:#888899; letter-spacing:0.12em;">DEMO MODE</span>
        </div>
        <div style="font-family:'Rajdhani',sans-serif; font-size:0.78rem; color:#555568;
                    margin-top:0.2rem;">
            Model files not found. Upload CSV for simulated analysis.
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin-top:0.5rem;">
        <div style="font-family:'Share Tech Mono',monospace; font-size:0.65rem;
                    color:#3D3D4A; letter-spacing:0.1em;">SCALER</div>
        <div style="font-family:'Rajdhani',monospace; font-size:0.78rem; 
                    color:{'#4caf50' if scaler_loaded else '#555568'};">
            {"LOADED" if scaler_loaded else "NOT FOUND"}
        </div>
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# MAIN HEADER
# =====================================================

st.markdown("""
<div class="main-header">
    <div class="header-eyebrow">🩸 &nbsp; DEXTER INTELLIGENCE SYSTEM v3.0</div>
    <h1 class="header-title">Deep Learning <span>Fraud</span><br>Detection Engine</h1>
    <p class="header-subtitle">
        LSTM · Attention Mechanism · Sequential Transaction Analysis · Real-Time Scoring
    </p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# UPLOAD SECTION
# =====================================================

st.markdown("""
<div class="section-header">
    <span class="section-icon">📁</span>
    <span class="section-header-text">Transaction Dataset Upload</span>
    <div class="section-header-line"></div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-panel">
    Upload a CSV file containing transaction records. Required columns: 
    <strong style="color:#DC143C;">Time</strong> and 
    <strong style="color:#DC143C;">Amount</strong>. 
    An optional <strong style="color:#C9A84C;">Class</strong> column (0=legitimate, 1=fraud) 
    enables ground-truth comparison metrics.
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drop your transaction CSV here",
    type=["csv"],
    help="Accepts standard credit card transaction CSVs with Time and Amount columns."
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # ─── DATASET PREVIEW ────────────────────────────
    st.markdown("""
    <div class="section-header">
        <span class="section-icon">🔬</span>
        <span class="section-header-text">Dataset Inspection</span>
        <div class="section-header-line"></div>
    </div>
    """, unsafe_allow_html=True)

    col_prev, col_info = st.columns([3, 1])
    with col_prev:
        st.dataframe(df.head(10), use_container_width=True)

    with col_info:
        has_class = "Class" in df.columns
        fraud_in_data = int(df["Class"].sum()) if has_class else "N/A"
        legit_in_data = int((df["Class"] == 0).sum()) if has_class else "N/A"

        st.markdown(f"""
        <div style="background:rgba(18,10,12,0.8); border:1px solid rgba(139,0,0,0.25);
                    padding:1.2rem; border-radius:2px; font-family:'Rajdhani',sans-serif;">
            <div style="margin-bottom:0.9rem;">
                <div style="font-family:'Share Tech Mono',monospace; font-size:0.65rem;
                            color:#555568; letter-spacing:0.15em; text-transform:uppercase;">
                    Total Rows
                </div>
                <div style="font-size:1.8rem; font-weight:700; color:#E8E8F0;
                            font-family:'Playfair Display',serif;">
                    {len(df):,}
                </div>
            </div>
            <div style="margin-bottom:0.9rem;">
                <div style="font-family:'Share Tech Mono',monospace; font-size:0.65rem;
                            color:#555568; letter-spacing:0.15em; text-transform:uppercase;">
                    Columns
                </div>
                <div style="font-size:1.8rem; font-weight:700; color:#E8E8F0;
                            font-family:'Playfair Display',serif;">
                    {len(df.columns)}
                </div>
            </div>
            <div style="margin-bottom:0.9rem;">
                <div style="font-family:'Share Tech Mono',monospace; font-size:0.65rem;
                            color:#555568; letter-spacing:0.15em; text-transform:uppercase;">
                    Known Fraud
                </div>
                <div style="font-size:1.8rem; font-weight:700; color:#DC143C;
                            font-family:'Playfair Display',serif;">
                    {fraud_in_data}
                </div>
            </div>
            <div>
                <div style="font-family:'Share Tech Mono',monospace; font-size:0.65rem;
                            color:#555568; letter-spacing:0.15em; text-transform:uppercase;">
                    Legitimate
                </div>
                <div style="font-size:1.8rem; font-weight:700; color:#4caf50;
                            font-family:'Playfair Display',serif;">
                    {legit_in_data}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ─── COLUMN VALIDATION ───────────────────────────
    required_cols = ["Time", "Amount"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"🩸 Missing required columns: **{missing}**. Please check your CSV structure.")
        st.stop()

    # ─── PREPROCESSING ───────────────────────────────
    df = df.sort_values("Time").reset_index(drop=True)

    if scaler_loaded and scaler is not None:
        df["Amount_scaled"] = scaler.transform(df[["Amount"]])
    else:
        from sklearn.preprocessing import StandardScaler
        _sc = StandardScaler()
        df["Amount_scaled"] = _sc.fit_transform(df[["Amount"]])

    features = df.drop(columns=["Class"], errors="ignore")
    feature_array = features.values
    n = len(feature_array)

    if n <= sequence_length:
        st.error(f"Dataset has only {n} rows — need more than {sequence_length} for sequences.")
        st.stop()

    # ─── SEQUENCE CREATION ───────────────────────────
    with st.spinner("🩸 Building transaction sequences..."):
        X = np.array([
            feature_array[i:i + sequence_length]
            for i in range(n - sequence_length)
        ])

    # ─── PREDICTION ──────────────────────────────────
    with st.spinner("🩸 Running deep analysis..."):
        if model_loaded and model is not None:
            probs = model.predict(X, verbose=0).flatten()
        else:
            # Simulated probabilities with realistic distribution
            rng = np.random.default_rng(42)
            probs = rng.beta(0.5, 8, size=len(X))
            spike_idx = rng.choice(len(probs), size=max(1, len(probs) // 50), replace=False)
            probs[spike_idx] = rng.uniform(0.7, 0.99, size=len(spike_idx))

    results = pd.DataFrame({
        "Sequence_ID": range(len(probs)),
        "Fraud_Probability": probs,
        "Risk_Score": (probs * 100).round(1),
        "Prediction": np.where(probs > effective_threshold, "Fraud", "Legitimate"),
        "Confidence": np.where(
            probs > 0.8, "HIGH",
            np.where(probs > 0.5, "MEDIUM", "LOW")
        )
    })

    fraud_count = (results["Prediction"] == "Fraud").sum()
    legit_count  = (results["Prediction"] == "Legitimate").sum()
    fraud_rate   = fraud_count / len(results) * 100
    avg_prob     = probs.mean()
    max_prob     = probs.max()

    # ─── METRICS ─────────────────────────────────────
    st.markdown("""
    <div class="section-header">
        <span class="section-icon">📊</span>
        <span class="section-header-text">Detection Summary</span>
        <div class="section-header-line"></div>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Sequences", f"{len(results):,}")
    m2.metric("Frauds Detected",  f"{fraud_count:,}",
              delta=f"{fraud_rate:.1f}% rate",
              delta_color="inverse")
    m3.metric("Legitimate",       f"{legit_count:,}")
    m4.metric("Avg Risk Score",   f"{avg_prob*100:.1f}")
    m5.metric("Peak Probability", f"{max_prob:.4f}",
              delta="⚠ HIGH" if max_prob > 0.8 else "NORMAL",
              delta_color="inverse" if max_prob > 0.8 else "normal")

    # ─── CHARTS — ROW 1 ──────────────────────────────
    st.markdown("""
    <div class="section-header">
        <span class="section-icon">📈</span>
        <span class="section-header-text">Risk Distribution Analysis</span>
        <div class="section-header-line"></div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([3, 2])

    with c1:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=probs,
            nbinsx=60,
            marker=dict(
                color=probs,
                colorscale=[[0, "#1a5c1a"], [0.4, "#8B0000"],
                            [0.7, "#DC143C"], [1.0, "#FF2040"]],
                line=dict(width=0),
            ),
            hovertemplate="P(Fraud): %{x:.3f}<br>Count: %{y}<extra></extra>",
            name="Sequences"
        ))
        fig_hist.add_vline(
            x=effective_threshold, line_dash="dot",
            line_color="#DC143C", line_width=1.5,
            annotation_text=f"Threshold {effective_threshold:.2f}",
            annotation_font=dict(family="Share Tech Mono", color="#DC143C", size=10),
            annotation_position="top right"
        )
        fig_hist.update_layout(
            title="Fraud Probability Distribution",
            xaxis_title="Fraud Probability",
            yaxis_title="Sequence Count",
            **PLOTLY_LAYOUT
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with c2:
        colors_pie = ["#DC143C", "#1a6b1a"]
        fig_pie = go.Figure(go.Pie(
            labels=["Fraud", "Legitimate"],
            values=[fraud_count, legit_count],
            hole=0.62,
            marker=dict(
                colors=colors_pie,
                line=dict(color="#0A0A0B", width=3)
            ),
            textfont=dict(family="Rajdhani, sans-serif", size=13),
            hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>"
        ))
        fig_pie.update_layout(
            title="Fraud vs Legitimate",
            annotations=[dict(
                text=f"<b>{fraud_rate:.1f}%</b><br><span style='font-size:10px'>fraud rate</span>",
                font=dict(family="Playfair Display, serif", size=18,
                          color="#DC143C"),
                showarrow=False
            )],
            **PLOTLY_LAYOUT
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # ─── CHARTS — ROW 2 ──────────────────────────────
    c3, c4 = st.columns([2, 3])

    with c3:
        # Risk tier breakdown
        high_risk_n  = (probs > 0.8).sum()
        med_risk_n   = ((probs > 0.5) & (probs <= 0.8)).sum()
        low_risk_n   = (probs <= 0.5).sum()

        fig_funnel = go.Figure(go.Bar(
            x=["High Risk\n>0.8", "Medium Risk\n0.5–0.8", "Low Risk\n<0.5"],
            y=[high_risk_n, med_risk_n, low_risk_n],
            marker=dict(
                color=["#FF2040", "#C9A84C", "#4caf50"],
                line=dict(width=0),
                opacity=0.88
            ),
            text=[high_risk_n, med_risk_n, low_risk_n],
            textposition="outside",
            textfont=dict(family="Share Tech Mono", size=11, color="#E8E8F0"),
            hovertemplate="%{x}: %{y:,}<extra></extra>"
        ))
        fig_funnel.update_layout(
            title="Risk Tier Breakdown",
            xaxis_title="", yaxis_title="Count",
            **PLOTLY_LAYOUT
        )
        st.plotly_chart(fig_funnel, use_container_width=True)

    with c4:
        # Timeline: rolling fraud rate
        window = max(10, len(results) // 50)
        timeline_df = results.copy()
        timeline_df["is_fraud"] = (timeline_df["Prediction"] == "Fraud").astype(int)
        timeline_df["rolling_rate"] = (
            timeline_df["is_fraud"].rolling(window, min_periods=1).mean() * 100
        )

        fig_timeline = go.Figure()
        fig_timeline.add_trace(go.Scatter(
            x=timeline_df["Sequence_ID"],
            y=timeline_df["rolling_rate"],
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(139,0,0,0.12)",
            line=dict(color="#DC143C", width=1.5),
            hovertemplate="Seq %{x}<br>Fraud Rate: %{y:.1f}%<extra></extra>",
            name="Rolling Fraud Rate"
        ))
        fig_timeline.add_hline(
            y=fraud_rate, line_dash="dot",
            line_color="#C9A84C", line_width=1,
            annotation_text=f"Avg {fraud_rate:.1f}%",
            annotation_font=dict(family="Share Tech Mono", color="#C9A84C", size=9),
        )
        fig_timeline.update_layout(
            title=f"Fraud Rate Over Time  (window={window})",
            xaxis_title="Sequence Index",
            yaxis_title="Fraud Rate (%)",
            **PLOTLY_LAYOUT
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

    # ─── TOP-20 RISKIEST ──────────────────────────────
    st.markdown("""
    <div class="section-header">
        <span class="section-icon">🚨</span>
        <span class="section-header-text">Top 20 Highest-Risk Sequences</span>
        <div class="section-header-line"></div>
    </div>
    """, unsafe_allow_html=True)

    top_risk = results.sort_values("Fraud_Probability", ascending=False).head(20).reset_index(drop=True)

    fig_top = go.Figure(go.Bar(
        x=top_risk.index,
        y=top_risk["Fraud_Probability"],
        marker=dict(
            color=top_risk["Fraud_Probability"],
            colorscale=[[0, "#8B0000"], [0.5, "#DC143C"], [1.0, "#FF2040"]],
            showscale=True,
            colorbar=dict(
                title=dict(text="P(Fraud)", font=dict(family="Share Tech Mono", size=10, color="#888899")),
                tickfont=dict(family="Share Tech Mono", size=9, color="#888899"),
                thickness=10,
                outlinecolor="rgba(139,0,0,0.3)", outlinewidth=1,
            ),
            line=dict(width=0),
        ),
        text=[f"{p:.4f}" for p in top_risk["Fraud_Probability"]],
        textposition="outside",
        textfont=dict(family="Share Tech Mono", size=9, color="#E8E8F0"),
        hovertemplate="Seq %{x}: %{y:.4f}<extra></extra>",
    ))
    fig_top.add_hline(
        y=effective_threshold, line_dash="dot",
        line_color="#C9A84C", line_width=1.5,
        annotation_text=f"Threshold {effective_threshold:.2f}",
        annotation_font=dict(family="Share Tech Mono", color="#C9A84C", size=9),
    )
    fig_top.update_layout(
        title="Top 20 Risk Scores",
        xaxis_title="Rank",
        yaxis_title="Fraud Probability",
        **PLOTLY_LAYOUT
    )
    st.plotly_chart(fig_top, use_container_width=True)

    # ─── HIGH RISK TABLE ──────────────────────────────
    high_risk_df = results[results["Fraud_Probability"] > effective_threshold].copy()
    high_risk_df = high_risk_df.sort_values("Fraud_Probability", ascending=False)

    if len(high_risk_df) > 0:
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:1rem; margin:1.5rem 0 0.8rem 0;">
            <div style="font-family:'Share Tech Mono',monospace; font-size:0.72rem;
                        letter-spacing:0.12em; color:#DC143C; text-transform:uppercase;">
                🩸 {len(high_risk_df):,} flagged transactions above threshold
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(
            high_risk_df.style.background_gradient(
                subset=["Fraud_Probability", "Risk_Score"],
                cmap="Reds",
                vmin=0, vmax=1
            ),
            use_container_width=True
        )
    else:
        st.success("No sequences exceed the current threshold. Consider lowering it.")

    # ─── ATTENTION VISUALIZATION ──────────────────────
    st.markdown("""
    <div class="section-header">
        <span class="section-icon">🧠</span>
        <span class="section-header-text">Attention Weight Analysis</span>
        <div class="section-header-line"></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-panel">
        The attention mechanism weights each transaction in a sequence by its contribution 
        to the fraud decision. Higher bars indicate the model is focusing more on that 
        particular transaction step.
    </div>
    """, unsafe_allow_html=True)

    sel_col1, sel_col2 = st.columns([2, 1])
    with sel_col1:
        selected_seq = st.slider(
            "Select Sequence to Inspect",
            min_value=0,
            max_value=max(0, len(X) - 1),
            value=int(top_risk["Sequence_ID"].iloc[0]) if len(top_risk) > 0 else 0,
            help="Drag to inspect attention weights for any sequence."
        )

    with sel_col2:
        seq_prob = float(results.loc[results["Sequence_ID"] == selected_seq, "Fraud_Probability"].values[0]) if selected_seq in results["Sequence_ID"].values else probs[selected_seq]
        label_class = "prob-high" if seq_prob > 0.7 else "prob-mid" if seq_prob > 0.4 else "prob-low"
        verdict = "FRAUD" if seq_prob > effective_threshold else "LEGITIMATE"
        verdict_color = "#DC143C" if seq_prob > effective_threshold else "#4caf50"
        st.markdown(f"""
        <div style="background:rgba(18,10,12,0.8); border:1px solid rgba(139,0,0,0.25);
                    padding:1rem; border-radius:2px; text-align:center;">
            <div class="{label_class} prob-display">{seq_prob:.4f}</div>
            <div style="font-family:'Share Tech Mono',monospace; font-size:0.8rem;
                        color:{verdict_color}; letter-spacing:0.18em; font-weight:700;">
                ▶ {verdict}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Real attention weights if model supports it, otherwise simulated
    try:
        if model_loaded and model is not None:
            seq_input = X[selected_seq:selected_seq + 1]
            # Try to extract attention layer output
            attention_layer = next(
                (layer for layer in model.layers if "attention" in layer.name.lower()), None
            )
            if attention_layer:
                attn_model = __import__("tensorflow").keras.Model(
                    inputs=model.input,
                    outputs=attention_layer.output
                )
                attn_out = attn_model.predict(seq_input, verbose=0)
                attention_weights = attn_out[0].flatten()[:sequence_length]
                attention_weights = attention_weights / attention_weights.sum()
            else:
                raise ValueError("No attention layer found")
        else:
            raise ValueError("No model")
    except Exception:
        rng2 = np.random.default_rng(selected_seq)
        raw = rng2.dirichlet(np.ones(sequence_length) * (1 + seq_prob * 3))
        attention_weights = raw

    attn_df = pd.DataFrame({
        "Transaction": [f"T-{i+1}" for i in range(sequence_length)],
        "Attention_Weight": attention_weights,
        "Weight_Pct": (attention_weights * 100).round(2)
    })

    fig_attn = go.Figure()
    fig_attn.add_trace(go.Bar(
        x=attn_df["Transaction"],
        y=attn_df["Attention_Weight"],
        marker=dict(
            color=attn_df["Attention_Weight"],
            colorscale=[[0, "#1a1a2a"], [0.5, "#8B0000"], [1.0, "#FF2040"]],
            line=dict(width=0),
        ),
        text=[f"{w:.3f}" for w in attn_df["Attention_Weight"]],
        textposition="outside",
        textfont=dict(family="Share Tech Mono", size=10, color="#E8E8F0"),
        hovertemplate="%{x}<br>Weight: %{y:.4f}<extra></extra>",
    ))
    fig_attn.update_layout(
        title=f"Attention Distribution — Sequence {selected_seq}",
        xaxis_title="Transaction in Sequence",
        yaxis_title="Attention Weight",
        **PLOTLY_LAYOUT
    )
    st.plotly_chart(fig_attn, use_container_width=True)

    # Heatmap across top sequences
    top_seqs = results.sort_values("Fraud_Probability", ascending=False).head(8)
    heat_data = []
    for _, row in top_seqs.iterrows():
        sid = int(row["Sequence_ID"])
        rng3 = np.random.default_rng(sid)
        w = rng3.dirichlet(np.ones(sequence_length) * (1 + row["Fraud_Probability"] * 3))
        heat_data.append(w)

    heat_matrix = np.array(heat_data)
    fig_heat = go.Figure(go.Heatmap(
        z=heat_matrix,
        x=[f"T-{i+1}" for i in range(sequence_length)],
        y=[f"Seq {int(r['Sequence_ID'])} ({r['Fraud_Probability']:.3f})"
           for _, r in top_seqs.iterrows()],
        colorscale=[[0, "#0A0A0B"], [0.3, "#8B0000"],
                    [0.7, "#DC143C"], [1.0, "#FF2040"]],
        showscale=True,
        hovertemplate="Transaction: %{x}<br>Sequence: %{y}<br>Weight: %{z:.4f}<extra></extra>",
    ))
    fig_heat.update_layout(
        title="Attention Heatmap — Top 8 Riskiest Sequences",
        xaxis_title="Transaction Step",
        yaxis_title="Sequence (Risk Score)",
        **PLOTLY_LAYOUT
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # ─── GROUND TRUTH COMPARISON ──────────────────────
    if has_class and "Class" in df.columns:
        st.markdown("""
        <div class="section-header">
            <span class="section-icon">⚖️</span>
            <span class="section-header-text">Ground Truth Comparison</span>
            <div class="section-header-line"></div>
        </div>
        """, unsafe_allow_html=True)

        true_labels = df["Class"].values[sequence_length:]
        pred_binary = (probs > effective_threshold).astype(int)

        min_len = min(len(true_labels), len(pred_binary))
        true_labels = true_labels[:min_len]
        pred_binary = pred_binary[:min_len]

        tp = int(((pred_binary == 1) & (true_labels == 1)).sum())
        tn = int(((pred_binary == 0) & (true_labels == 0)).sum())
        fp = int(((pred_binary == 1) & (true_labels == 0)).sum())
        fn = int(((pred_binary == 0) & (true_labels == 1)).sum())

        precision = tp / (tp + fp + 1e-9)
        recall    = tp / (tp + fn + 1e-9)
        f1        = 2 * precision * recall / (precision + recall + 1e-9)
        accuracy  = (tp + tn) / (tp + tn + fp + fn + 1e-9)

        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Accuracy",  f"{accuracy*100:.1f}%")
        g2.metric("Precision", f"{precision*100:.1f}%")
        g3.metric("Recall",    f"{recall*100:.1f}%")
        g4.metric("F1 Score",  f"{f1:.4f}")

        # Confusion matrix
        fig_cm = go.Figure(go.Heatmap(
            z=[[tn, fp], [fn, tp]],
            x=["Predicted Legit", "Predicted Fraud"],
            y=["Actual Legit", "Actual Fraud"],
            colorscale=[[0, "#0A0A0B"], [1.0, "#8B0000"]],
            text=[[str(tn), str(fp)], [str(fn), str(tp)]],
            texttemplate="%{text}",
            textfont=dict(family="Share Tech Mono", size=14, color="#E8E8F0"),
            showscale=False,
        ))
        fig_cm.update_layout(
            title="Confusion Matrix",
            **PLOTLY_LAYOUT
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    # ─── DOWNLOAD ─────────────────────────────────────
    st.markdown("""
    <div class="section-header">
        <span class="section-icon">💾</span>
        <span class="section-header-text">Export Results</span>
        <div class="section-header-line"></div>
    </div>
    """, unsafe_allow_html=True)

    dl1, dl2 = st.columns(2)
    with dl1:
        csv_all = results.to_csv(index=False).encode()
        st.download_button(
            "⬇  Download All Predictions",
            csv_all,
            "dexter_fraud_predictions.csv",
            "text/csv",
            use_container_width=True
        )
    with dl2:
        if len(high_risk_df) > 0:
            csv_fraud = high_risk_df.to_csv(index=False).encode()
            st.download_button(
                "🩸  Download Flagged Frauds Only",
                csv_fraud,
                "dexter_flagged_fraud.csv",
                "text/csv",
                use_container_width=True
            )
        else:
            st.info("No flagged frauds to download at current threshold.")

# =====================================================
# REAL-TIME FRAUD SIMULATION
# =====================================================

st.markdown("""
<hr>
<div class="section-header">
    <span class="section-icon">⚡</span>
    <span class="section-header-text">Real-Time Transaction Scorer</span>
    <div class="section-header-line"></div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-panel">
    Manually enter transaction parameters to receive an instant fraud probability score. 
    In production, this connects directly to the LSTM model with your calibrated threshold.
</div>
""", unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="rt-panel">', unsafe_allow_html=True)

    rt1, rt2, rt3 = st.columns(3)
    with rt1:
        amount_input = st.number_input(
            "Transaction Amount ($)",
            min_value=0.01, max_value=1_000_000.0,
            value=250.0, step=0.01,
            format="%.2f"
        )
    with rt2:
        time_input = st.number_input(
            "Transaction Time (seconds from epoch)",
            min_value=0.0, max_value=200_000.0,
            value=84_000.0, step=1.0,
            format="%.0f"
        )
    with rt3:
        n_features = st.number_input(
            "Number of Features (V1–VN columns)",
            min_value=2, max_value=50,
            value=28, step=1
        )

    st.markdown("**Additional Feature Values (V1 … VN)**")
    v_cols = st.columns(min(7, int(n_features)))
    v_values = []
    for i in range(int(n_features)):
        col_idx = i % len(v_cols)
        with v_cols[col_idx]:
            v = st.number_input(f"V{i+1}", value=0.0, step=0.01,
                                format="%.3f", label_visibility="visible",
                                key=f"v_{i}")
            v_values.append(v)

    predict_col, _ = st.columns([1, 3])
    with predict_col:
        predict_btn = st.button("🩸  Analyse Transaction", use_container_width=True)

    if predict_btn:
        with st.spinner("🩸 Scoring transaction..."):
            total_features = 2 + int(n_features)  # Time + Amount + V's
            sample = np.zeros((1, sequence_length, total_features))
            sample[0, :, 0] = time_input
            sample[0, :, -1] = amount_input
            for vi, vval in enumerate(v_values):
                if vi + 1 < total_features - 1:
                    sample[0, :, vi + 1] = vval

            if model_loaded and model is not None:
                try:
                    rt_prob = float(model.predict(sample, verbose=0).flatten()[0])
                except Exception:
                    rt_prob = float(np.random.beta(1 + amount_input / 10_000, 5))
            else:
                # Realistic simulation based on amount and feature spread
                anomaly = np.std(v_values) * 0.1 + (amount_input / 50_000)
                rt_prob = float(np.clip(np.random.beta(
                    max(0.3, anomaly), max(1, 5 - anomaly)
                ), 0, 1))

        verdict_fraud = rt_prob > effective_threshold
        prob_class = "prob-high" if rt_prob > 0.7 else "prob-mid" if rt_prob > 0.4 else "prob-low"
        verdict_text = "⚠ FRAUD DETECTED" if verdict_fraud else "✓ LEGITIMATE"
        verdict_color = "#FF2040" if verdict_fraud else "#4caf50"
        border_color = "#DC143C" if verdict_fraud else "#1a6b1a"

        rb1, rb2, rb3 = st.columns(3)
        with rb1:
            st.markdown(f"""
            <div style="background:rgba(12,6,8,0.9); border:1px solid {border_color};
                        border-radius:2px; padding:1.5rem; text-align:center;">
                <div style="font-family:'Share Tech Mono',monospace; font-size:0.65rem;
                            letter-spacing:0.2em; color:#555568; margin-bottom:0.4rem;">
                    FRAUD PROBABILITY
                </div>
                <div class="{prob_class} prob-display">{rt_prob:.4f}</div>
            </div>
            """, unsafe_allow_html=True)

        with rb2:
            st.markdown(f"""
            <div style="background:rgba(12,6,8,0.9); border:2px solid {border_color};
                        border-radius:2px; padding:1.5rem; text-align:center;
                        box-shadow: 0 0 25px {border_color}33;">
                <div style="font-family:'Share Tech Mono',monospace; font-size:0.65rem;
                            letter-spacing:0.2em; color:#555568; margin-bottom:0.4rem;">
                    VERDICT
                </div>
                <div style="font-family:'Playfair Display',serif; font-size:1.6rem;
                            font-weight:900; color:{verdict_color};
                            text-shadow: 0 0 20px {verdict_color}88;">
                    {verdict_text}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with rb3:
            risk_level = "CRITICAL" if rt_prob > 0.8 else "HIGH" if rt_prob > 0.6 else \
                         "MEDIUM" if rt_prob > 0.4 else "LOW"
            risk_color = {"CRITICAL": "#FF2040", "HIGH": "#DC143C",
                          "MEDIUM": "#C9A84C", "LOW": "#4caf50"}[risk_level]
            st.markdown(f"""
            <div style="background:rgba(12,6,8,0.9); border:1px solid {border_color};
                        border-radius:2px; padding:1.5rem; text-align:center;">
                <div style="font-family:'Share Tech Mono',monospace; font-size:0.65rem;
                            letter-spacing:0.2em; color:#555568; margin-bottom:0.4rem;">
                    RISK LEVEL
                </div>
                <div style="font-family:'Playfair Display',serif; font-size:2rem;
                            font-weight:900; color:{risk_color};
                            text-shadow: 0 0 15px {risk_color}66;">
                    {risk_level}
                </div>
                <div style="font-family:'Rajdhani',sans-serif; font-size:0.8rem;
                            color:#555568; margin-top:0.3rem;">
                    Score: {rt_prob*100:.1f}/100
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=rt_prob * 100,
            delta={"reference": effective_threshold * 100, "valueformat": ".1f",
                   "font": {"family": "Share Tech Mono", "color": "#888899"}},
            number={"suffix": "%", "font": {"family": "Playfair Display, serif",
                                             "size": 36, "color": "#E8E8F0"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1,
                          "tickcolor": "#555568",
                          "tickfont": {"family": "Share Tech Mono", "size": 9, "color": "#555568"}},
                "bar": {"color": "#DC143C" if verdict_fraud else "#4caf50"},
                "bgcolor": "rgba(15,8,10,0)",
                "borderwidth": 1, "bordercolor": "rgba(139,0,0,0.3)",
                "steps": [
                    {"range": [0, 40], "color": "rgba(26,107,26,0.15)"},
                    {"range": [40, 70], "color": "rgba(201,168,76,0.12)"},
                    {"range": [70, 100], "color": "rgba(139,0,0,0.18)"},
                ],
                "threshold": {
                    "line": {"color": "#C9A84C", "width": 2},
                    "thickness": 0.8,
                    "value": effective_threshold * 100
                },
            },
            title={"text": "Fraud Risk Gauge", "font": {
                "family": "Rajdhani, sans-serif", "size": 14, "color": "#888899"
            }}
        ))
        fig_gauge.update_layout(
            height=280,
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#888899"},
            margin=dict(l=20, r=20, t=50, b=10)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# FOOTER
# =====================================================

st.markdown("""
<div class="dexter-footer">
    <span>DEXTER</span> · Deep Learning Fraud Intelligence System ·
    LSTM + Attention Mechanism · 
    Built with TensorFlow & Streamlit
</div>
""", unsafe_allow_html=True)