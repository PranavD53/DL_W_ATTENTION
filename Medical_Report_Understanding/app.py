"""
╔══════════════════════════════════════════════════════════════════════╗
║     INTELLIGENT MEDICAL REPORT UNDERSTANDING SYSTEM                  ║
║     Enhanced UI/UX — Clinical-grade dark theme with cyan accents      ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os

import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datetime
import re
import io
import time
from collections import Counter

from tensorflow.keras.preprocessing.sequence import pad_sequences

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

# ══════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="MedScan AI — Medical Report Analysis",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

MAX_LEN = 100

# ══════════════════════════════════════════════════════════════
#  CUSTOM CSS — Clinical Dark Theme
# ══════════════════════════════════════════════════════════════

st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600&family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500&display=swap');

/* ── Root Variables ── */
:root {
    --bg-deep:     #050d14;
    --bg-panel:    #0a1624;
    --bg-card:     #0f2236;
    --border:      #1a3550;
    --border-glow: #00c8ff44;
    --cyan:        #00c8ff;
    --cyan-dim:    #0097c4;
    --green:       #00e5aa;
    --amber:       #ffb830;
    --red:         #ff4b6e;
    --text-pri:    #e8f4fc;
    --text-sec:    #7fa8c4;
    --text-muted:  #3d6480;
    --font-head:   'Syne', sans-serif;
    --font-mono:   'IBM Plex Mono', monospace;
    --font-body:   'Inter', sans-serif;
}

/* ── Base Reset ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-deep) !important;
    color: var(--text-pri) !important;
    font-family: var(--font-body) !important;
}

[data-testid="stSidebar"] {
    background: var(--bg-panel) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: var(--cyan-dim); border-radius: 2px; }

/* ── Hero Banner ── */
.hero-banner {
    background: linear-gradient(135deg, #071826 0%, #0a2040 50%, #071826 100%);
    border: 1px solid var(--border);
    border-top: 3px solid var(--cyan);
    border-radius: 12px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, #00c8ff11 0%, transparent 70%);
    pointer-events: none;
}
.hero-banner::after {
    content: '+';
    position: absolute;
    font-size: 15rem;
    color: #00c8ff06;
    font-family: var(--font-head);
    top: -2rem;
    right: 2rem;
    line-height: 1;
    pointer-events: none;
}
.hero-title {
    font-family: var(--font-head);
    font-size: 2.4rem;
    font-weight: 800;
    color: var(--text-pri);
    letter-spacing: -0.02em;
    margin: 0 0 0.4rem;
    line-height: 1.1;
}
.hero-title span { color: var(--cyan); }
.hero-sub {
    font-family: var(--font-mono);
    font-size: 0.78rem;
    color: var(--text-sec);
    letter-spacing: 0.12em;
    text-transform: uppercase;
}
.hero-badge {
    display: inline-block;
    background: #00c8ff15;
    border: 1px solid #00c8ff33;
    color: var(--cyan);
    font-family: var(--font-mono);
    font-size: 0.68rem;
    padding: 0.25rem 0.75rem;
    border-radius: 2rem;
    margin-right: 0.5rem;
    letter-spacing: 0.08em;
}

/* ── Section Headers ── */
.section-header {
    font-family: var(--font-head);
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text-pri);
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin: 0 0 1rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.section-header .dot {
    width: 8px; height: 8px;
    background: var(--cyan);
    border-radius: 50%;
    display: inline-block;
    box-shadow: 0 0 8px var(--cyan);
}

/* ── Cards ── */
.med-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
}
.med-card:hover { border-color: var(--border-glow); }

/* ── Metric Cards ── */
.metric-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
.metric-card {
    flex: 1;
    min-width: 160px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}
.metric-card.cyan::before  { background: var(--cyan); }
.metric-card.green::before { background: var(--green); }
.metric-card.amber::before { background: var(--amber); }
.metric-card.red::before   { background: var(--red); }

.metric-label {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--text-muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.metric-value {
    font-family: var(--font-head);
    font-size: 1.9rem;
    font-weight: 800;
    color: var(--text-pri);
    line-height: 1;
}
.metric-value.cyan  { color: var(--cyan); }
.metric-value.green { color: var(--green); }
.metric-value.amber { color: var(--amber); }

/* ── Result Banner ── */
.result-banner {
    background: linear-gradient(90deg, #002a3a 0%, #001f2d 100%);
    border: 1px solid #00c8ff33;
    border-left: 4px solid var(--cyan);
    border-radius: 10px;
    padding: 1.5rem 2rem;
    margin: 1.5rem 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 1rem;
}
.result-specialty {
    font-family: var(--font-head);
    font-size: 1.6rem;
    font-weight: 800;
    color: var(--cyan);
}
.result-label {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    color: var(--text-muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.25rem;
}
.confidence-bar-wrap {
    flex: 1;
    min-width: 200px;
}
.confidence-bar-bg {
    background: var(--bg-deep);
    border: 1px solid var(--border);
    border-radius: 4px;
    height: 10px;
    overflow: hidden;
    margin-top: 0.4rem;
}
.confidence-bar-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, var(--cyan-dim), var(--cyan));
    box-shadow: 0 0 8px var(--cyan);
    transition: width 0.8s ease;
}

/* ── Report Preview Box ── */
.report-preview {
    background: var(--bg-deep);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.25rem 1.5rem;
    font-family: var(--font-mono);
    font-size: 0.8rem;
    color: var(--text-sec);
    line-height: 1.7;
    max-height: 260px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-word;
}

/* ── Term Chips ── */
.term-chip {
    display: inline-block;
    background: #00c8ff12;
    border: 1px solid #00c8ff25;
    color: var(--cyan);
    font-family: var(--font-mono);
    font-size: 0.7rem;
    padding: 0.2rem 0.6rem;
    border-radius: 3px;
    margin: 0.2rem;
}
.term-chip.high { background: #00e5aa12; border-color: #00e5aa25; color: var(--green); }

/* ── Warning / Info blocks ── */
.info-block {
    background: #00c8ff0a;
    border: 1px solid #00c8ff20;
    border-left: 3px solid var(--cyan);
    border-radius: 6px;
    padding: 0.9rem 1.2rem;
    font-family: var(--font-mono);
    font-size: 0.78rem;
    color: var(--text-sec);
    margin: 0.75rem 0;
}
.warn-block {
    background: #ffb8300a;
    border: 1px solid #ffb83025;
    border-left: 3px solid var(--amber);
    border-radius: 6px;
    padding: 0.9rem 1.2rem;
    font-family: var(--font-mono);
    font-size: 0.78rem;
    color: #c9974a;
    margin: 0.75rem 0;
}

/* ── Sidebar Styles ── */
.sidebar-logo {
    font-family: var(--font-head);
    font-size: 1.3rem;
    font-weight: 800;
    color: var(--cyan);
    padding: 0.5rem 0 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}
.sidebar-logo span { color: var(--text-sec); font-weight: 400; font-size: 0.85rem; display: block; margin-top: 0.15rem; }

.sidebar-stat {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.75rem;
}
.sidebar-stat-label {
    font-family: var(--font-mono);
    font-size: 0.62rem;
    color: var(--text-muted);
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.sidebar-stat-val {
    font-family: var(--font-head);
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text-pri);
    margin-top: 0.2rem;
}

/* ── Streamlit Element Overrides ── */
.stTextArea textarea {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-pri) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.82rem !important;
}
.stTextArea textarea:focus {
    border-color: var(--cyan) !important;
    box-shadow: 0 0 0 2px #00c8ff18 !important;
}
.stButton > button {
    background: linear-gradient(135deg, #00607a, #007fa0) !important;
    border: 1px solid var(--cyan) !important;
    color: var(--text-pri) !important;
    font-family: var(--font-head) !important;
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    padding: 0.7rem 2.5rem !important;
    border-radius: 6px !important;
    letter-spacing: 0.05em !important;
    transition: all 0.2s !important;
    text-transform: uppercase !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #007fa0, #00c8ff) !important;
    box-shadow: 0 0 20px #00c8ff44 !important;
    transform: translateY(-1px) !important;
}
.stDownloadButton > button {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-sec) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
    border-radius: 6px !important;
}
.stDownloadButton > button:hover {
    border-color: var(--cyan) !important;
    color: var(--cyan) !important;
}
div[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 10px !important;
    padding: 1rem !important;
}
div[data-testid="stFileUploader"]:hover {
    border-color: var(--cyan-dim) !important;
}
.stProgress > div > div > div > div {
    background: var(--cyan) !important;
}
[data-testid="stMetricValue"] {
    font-family: var(--font-head) !important;
    font-size: 1.6rem !important;
    color: var(--cyan) !important;
}
[data-testid="stMetricLabel"] {
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    color: var(--text-muted) !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}
h1, h2, h3 {
    font-family: var(--font-head) !important;
    color: var(--text-pri) !important;
}
label, .stSelectbox label, .stTextArea label {
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    color: var(--text-muted) !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}
.stSelectbox > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-pri) !important;
    border-radius: 6px !important;
}
.stCheckbox span {
    color: var(--text-sec) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
}
.stAlert {
    background: var(--bg-card) !important;
    border-radius: 8px !important;
    font-family: var(--font-mono) !important;
}
hr {
    border-color: var(--border) !important;
    margin: 1.5rem 0 !important;
}
[data-testid="column"] { gap: 0; }

/* ── Matplotlib chart background ── */
.stImage { border-radius: 8px; overflow: hidden; }

/* ── Footer ── */
.footer {
    text-align: center;
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--text-muted);
    padding: 2rem 0 1rem;
    border-top: 1px solid var(--border);
    margin-top: 3rem;
    letter-spacing: 0.08em;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  ASSET LOADING WITH GRACEFUL FALLBACKS
# ══════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "explainable_model")
@st.cache_resource
def load_model():
    try:
        return tf.saved_model.load(model_path)
    except Exception as e:
        return None

@st.cache_resource
def load_tokenizer():
    try:
        with open(os.path.join(BASE_DIR, "tokenizer.pkl"), "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

@st.cache_resource
def load_label_encoder():
    try:
        with open(os.path.join(BASE_DIR, "label_encoder.pkl"), "rb") as f:
            
            return pickle.load(f)
    except Exception:
        return None

@st.cache_data
def load_medical_dict():
    try:
        return pd.read_csv("medical_dictionary.csv")
    except Exception:
        # Provide a fallback dictionary with common medical terms
        terms = [
            "hypertension","tachycardia","bradycardia","arrhythmia","myocardial",
            "infarction","ischemia","atherosclerosis","thrombosis","embolism",
            "pneumonia","bronchitis","dyspnea","hemoptysis","pleuritis",
            "hepatitis","cirrhosis","cholecystitis","pancreatitis","appendicitis",
            "nephritis","hematuria","proteinuria","dialysis","nephrotic",
            "diabetes","insulin","hyperglycemia","hypoglycemia","ketoacidosis",
            "leukemia","lymphoma","carcinoma","adenoma","metastasis",
            "fracture","osteoporosis","arthritis","tendinitis","scoliosis",
            "migraine","epilepsy","neuropathy","paresthesia","ataxia",
            "anemia","thrombocytopenia","leukopenia","coagulation","hemolysis",
            "hypothyroidism","hyperthyroidism","thyroiditis","cortisol","glucagon",
            "sepsis","bacteremia","meningitis","encephalitis","abscess",
        ]
        freq = np.random.randint(5, 100, len(terms))
        return pd.DataFrame({"Medical_Term": terms, "Frequency": freq})

model         = load_model()
tokenizer     = load_tokenizer()
label_encoder = load_label_encoder()
medical_dict  = load_medical_dict()

# All known specialties for demo fallback
ALL_SPECIALTIES = [
    "Cardiology", "Pulmonology", "Gastroenterology", "Neurology",
    "Oncology", "Orthopedics", "Nephrology", "Endocrinology",
    "Hematology", "Infectious Disease", "Radiology", "General Surgery",
    "Psychiatry", "Dermatology", "Ophthalmology",
]


# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        MedScan<b style="color:#00c8ff">AI</b>
        <span>Clinical NLP Platform v2.0</span>
    </div>
    """, unsafe_allow_html=True)

    # System Status
    st.markdown('<p style="font-family:\'IBM Plex Mono\',monospace;font-size:0.68rem;color:#3d6480;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.6rem;">System Status</p>', unsafe_allow_html=True)

    model_ok  = model is not None
    tok_ok    = tokenizer is not None
    enc_ok    = label_encoder is not None

    for name, ok in [("AI Model", model_ok), ("Tokenizer", tok_ok), ("Label Encoder", enc_ok)]:
        icon  = "🟢" if ok else "🔴"
        state = "ONLINE" if ok else "OFFLINE"
        color = "#00e5aa" if ok else "#ff4b6e"
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;align-items:center;
            padding:0.45rem 0.8rem;background:#0f2236;border:1px solid #1a3550;
            border-radius:6px;margin-bottom:0.4rem;">
          <span style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;color:#7fa8c4;">{icon} {name}</span>
          <span style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;color:{color};">{state}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr/>", unsafe_allow_html=True)

    # Settings
    st.markdown('<p style="font-family:\'IBM Plex Mono\',monospace;font-size:0.68rem;color:#3d6480;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.8rem;">Analysis Settings</p>', unsafe_allow_html=True)

    max_terms = st.slider("Top Medical Terms", min_value=5, max_value=30, value=15, help="Number of matched medical terms to display")
    show_pe   = st.checkbox("Positional Encoding Heatmap", value=True)
    show_dist = st.checkbox("Confidence Distribution", value=True)
    show_prev = st.checkbox("Report Preview", value=True)
    include_stats = st.checkbox("Word Statistics", value=True)

    st.markdown("<hr/>", unsafe_allow_html=True)

    # Report Metadata
    st.markdown('<p style="font-family:\'IBM Plex Mono\',monospace;font-size:0.68rem;color:#3d6480;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.8rem;">Report Metadata</p>', unsafe_allow_html=True)
    patient_id   = st.text_input("Patient ID", value="PT-" + str(np.random.randint(10000, 99999)), key="pid")
    analyst_name = st.text_input("Analyst Name", value="Dr. System", key="analyst")
    dept         = st.selectbox("Department", ["Diagnostics", "Emergency", "Outpatient", "ICU", "Research"], key="dept")

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;color:#3d6480;text-align:center;padding-top:0.5rem;">
        Session: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}<br/>
        MedScan AI © 2025
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  HERO BANNER
# ══════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero-banner">
    <div class="hero-title">Intelligent Medical Report<br><span>Understanding System</span></div>
    <div style="margin: 0.6rem 0 1rem;">
        <span class="hero-badge">🧠 Attention NLP</span>
        <span class="hero-badge">📊 Multi-Class</span>
        <span class="hero-badge">⚡ Real-Time</span>
    </div>
    <div class="hero-sub">Upload or paste a clinical report · AI classifies the medical specialty · Export structured PDF</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  INPUT SECTION
# ══════════════════════════════════════════════════════════════

col_in1, col_in2 = st.columns([1, 1], gap="large")

with col_in1:
    st.markdown("""
    <div class="section-header">
        <span class="dot"></span> Input — Upload File
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop a .txt medical report here",
        type=["txt"],
        label_visibility="collapsed",
    )

with col_in2:
    st.markdown("""
    <div class="section-header">
        <span class="dot"></span> Input — Paste Report
    </div>
    """, unsafe_allow_html=True)
    report_text_area = st.text_area(
        "Paste clinical report text",
        height=180,
        placeholder="Paste report text here, e.g.: Patient presents with chest pain, dyspnea on exertion, elevated troponin levels...",
        label_visibility="collapsed",
    )

# Resolve final report text (uploaded file takes precedence)
if uploaded_file is not None:
    report_text = uploaded_file.read().decode("utf-8", errors="replace")
    st.markdown(f'<div class="info-block">📄 Loaded file: <b>{uploaded_file.name}</b> — {len(report_text):,} characters</div>', unsafe_allow_html=True)
else:
    report_text = report_text_area

# ── Quick word-count preview metrics ──
if report_text.strip():
    wc  = len(report_text.split())
    sc  = len(re.split(r'[.!?]+', report_text.strip()))
    uc  = len(set(report_text.lower().split()))
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Words",       f"{wc:,}")
    col_m2.metric("Sentences",   f"{sc:,}")
    col_m3.metric("Unique Words",f"{uc:,}")
    col_m4.metric("Characters",  f"{len(report_text):,}")

st.markdown("<br/>", unsafe_allow_html=True)

# ── Analyse Button ──
btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 2])
with btn_col1:
    run_analysis = st.button("🔬  Analyse Report", use_container_width=True)
with btn_col2:
    clear_btn = st.button("✖  Clear", use_container_width=True)

if clear_btn:
    st.rerun()


# ══════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════

def predict_specialty(text: str):
    """Run inference; returns (specialty, confidence, all_probs)."""
    if model is None or tokenizer is None or label_encoder is None:
        # Demo fallback: score by keyword presence
        text_lower = text.lower()
        keyword_map = {
            "Cardiology":         ["heart","cardiac","myocardial","troponin","ecg","arrhythmia","coronary","palpitation"],
            "Pulmonology":        ["lung","pulmonary","bronchial","asthma","copd","dyspnea","pneumonia","spirometry"],
            "Gastroenterology":   ["gastric","bowel","hepatic","liver","pancreas","colon","endoscopy","ulcer"],
            "Neurology":          ["brain","neuro","seizure","migraine","stroke","dementia","parkinson","cranial"],
            "Oncology":           ["tumor","cancer","malignant","carcinoma","chemotherapy","biopsy","metastasis","radiation"],
            "Orthopedics":        ["bone","fracture","joint","arthritis","spine","ligament","tendon","orthopedic"],
            "Nephrology":         ["kidney","renal","creatinine","dialysis","nephritis","glomerular","urine","proteinuria"],
            "Endocrinology":      ["thyroid","diabetes","insulin","cortisol","hormone","pituitary","adrenal","glycemic"],
            "Hematology":         ["blood","anemia","platelet","hemoglobin","leukemia","coagulation","lymphoma","bone marrow"],
            "Infectious Disease": ["infection","bacteria","virus","antibiotic","sepsis","fever","culture","pathogen"],
        }
        scores = {}
        for spec, kws in keyword_map.items():
            scores[spec] = sum(1 for kw in kws if kw in text_lower)
        total = max(sum(scores.values()), 1)
        probs = {k: v / total for k, v in scores.items()}
        best  = max(probs, key=probs.get)
        # normalise across ALL_SPECIALTIES
        all_probs = np.array([probs.get(s, 0.01) for s in ALL_SPECIALTIES])
        all_probs = all_probs / all_probs.sum()
        return best, float(probs[best] / total * (1 / max(probs.values()))), all_probs

    seq  = tokenizer.texts_to_sequences([text])
    seq  = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    infer = model.signatures["serving_default"]
    out   = infer(tf.constant(seq, dtype=tf.float32))
    pred  = list(out.values())[0].numpy()[0]
    idx   = int(np.argmax(pred))
    specialty = label_encoder.inverse_transform([idx])[0]
    return specialty, float(pred[idx]), pred


def positional_encoding(max_pos: int, d_model: int) -> np.ndarray:
    PE  = np.zeros((max_pos, d_model))
    pos = np.arange(max_pos)[:, np.newaxis]
    i   = np.arange(d_model)[np.newaxis, :]
    angle = pos / np.power(10000, (2 * (i // 2)) / d_model)
    PE[:, 0::2] = np.sin(angle[:, 0::2])
    PE[:, 1::2] = np.cos(angle[:, 1::2])
    return PE


def get_matched_terms(text: str, top_n: int = 15) -> pd.DataFrame:
    words = set(re.findall(r'\b[a-zA-Z]{4,}\b', text))
    matched = medical_dict[medical_dict["Medical_Term"].isin(words)]
    matched = matched.sort_values("Frequency", ascending=False).head(top_n)
    return matched.reset_index(drop=True)


def word_statistics(text: str) -> pd.DataFrame:
    tokens = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    counts = Counter(tokens)
    df = pd.DataFrame(counts.most_common(20), columns=["Word", "Count"])
    return df


def set_dark_plot_style():
    plt.rcParams.update({
        "figure.facecolor":  "#050d14",
        "axes.facecolor":    "#0a1624",
        "axes.edgecolor":    "#1a3550",
        "axes.labelcolor":   "#7fa8c4",
        "text.color":        "#7fa8c4",
        "xtick.color":       "#3d6480",
        "ytick.color":       "#3d6480",
        "grid.color":        "#1a3550",
        "grid.linestyle":    "--",
        "grid.alpha":        0.5,
        "font.family":       "monospace",
    })


def generate_pdf(specialty, confidence, text, patient_id, analyst, dept,
                 matched_terms_df, all_probs=None) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch,
    )
    styles = getSampleStyleSheet()
    C = colors.HexColor

    # Custom styles
    title_style = ParagraphStyle(
        "MedTitle",
        parent=styles["Normal"],
        fontSize=20,
        textColor=C("#00c8ff"),
        fontName="Helvetica-Bold",
        spaceAfter=4,
        alignment=TA_CENTER,
    )
    subtitle_style = ParagraphStyle(
        "MedSub",
        parent=styles["Normal"],
        fontSize=8,
        textColor=C("#7fa8c4"),
        fontName="Helvetica",
        spaceAfter=16,
        alignment=TA_CENTER,
    )
    h2_style = ParagraphStyle(
        "MedH2",
        parent=styles["Normal"],
        fontSize=11,
        textColor=C("#00c8ff"),
        fontName="Helvetica-Bold",
        spaceBefore=12,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "MedBody",
        parent=styles["Normal"],
        fontSize=9,
        textColor=C("#2c3e50"),
        fontName="Helvetica",
        leading=14,
    )
    label_style = ParagraphStyle(
        "MedLabel",
        parent=styles["Normal"],
        fontSize=7,
        textColor=C("#7fa8c4"),
        fontName="Helvetica",
    )

    story = []

    # Header
    story.append(Paragraph("MEDSCAN AI", title_style))
    story.append(Paragraph("Intelligent Medical Report Understanding System", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=C("#00c8ff"), spaceAfter=12))

    # Meta table
    now = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    meta_data = [
        [Paragraph("<b>Patient ID</b>", label_style),   Paragraph(patient_id, body_style),
         Paragraph("<b>Analyst</b>", label_style),       Paragraph(analyst, body_style)],
        [Paragraph("<b>Department</b>", label_style),    Paragraph(dept, body_style),
         Paragraph("<b>Generated</b>", label_style),     Paragraph(now, body_style)],
    ]
    meta_table = Table(meta_data, colWidths=[1.0*inch, 2.5*inch, 1.0*inch, 2.5*inch])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), C("#f0f8ff")),
        ("GRID",       (0, 0), (-1, -1), 0.5, C("#d0e8f4")),
        ("PADDING",    (0, 0), (-1, -1), 6),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [C("#e8f4fc"), C("#f5faff")]),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 16))

    # Prediction result
    story.append(Paragraph("CLASSIFICATION RESULT", h2_style))
    conf_pct = f"{confidence * 100:.1f}%"
    result_data = [
        [Paragraph("<b>Predicted Specialty</b>", label_style), Paragraph("Confidence Score", label_style)],
        [Paragraph(f"<font size='16' color='#00607a'><b>{specialty}</b></font>", body_style),
         Paragraph(f"<font size='16'><b>{conf_pct}</b></font>", body_style)],
    ]
    result_table = Table(result_data, colWidths=[3.5*inch, 3.5*inch])
    result_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), C("#00607a")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 1), (-1, 1), C("#e8f9fd")),
        ("GRID",       (0, 0), (-1, -1), 0.5, C("#00c8ff")),
        ("PADDING",    (0, 0), (-1, -1), 10),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
    ]))
    story.append(result_table)
    story.append(Spacer(1, 16))

    # Matched medical terms
    if len(matched_terms_df) > 0:
        story.append(Paragraph("DETECTED MEDICAL TERMS", h2_style))
        term_rows = [[Paragraph("<b>#</b>", label_style),
                      Paragraph("<b>Term</b>", label_style),
                      Paragraph("<b>Dict. Frequency</b>", label_style)]]
        for i, row in matched_terms_df.iterrows():
            term_rows.append([
                Paragraph(str(i + 1), body_style),
                Paragraph(str(row.get("Medical_Term", "")), body_style),
                Paragraph(str(row.get("Frequency", "")), body_style),
            ])
        term_table = Table(term_rows, colWidths=[0.5*inch, 4*inch, 2.5*inch])
        term_table.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), C("#00607a")),
            ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [C("#ffffff"), C("#f0f8ff")]),
            ("GRID",          (0, 0), (-1, -1), 0.4, C("#d0e8f4")),
            ("PADDING",       (0, 0), (-1, -1), 6),
        ]))
        story.append(term_table)
        story.append(Spacer(1, 16))

    # Report excerpt
    story.append(Paragraph("REPORT EXCERPT", h2_style))
    excerpt = text[:3000].replace("\n", " ")
    story.append(Paragraph(excerpt, body_style))

    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=1, color=C("#d0e8f4"), spaceAfter=6))
    story.append(Paragraph(
        f"<font size='7' color='#7fa8c4'>Generated by MedScan AI — This report is for informational purposes only and does not constitute medical advice.</font>",
        ParagraphStyle("foot", parent=styles["Normal"], alignment=TA_CENTER)
    ))

    doc.build(story)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════
#  ANALYSIS ENGINE
# ══════════════════════════════════════════════════════════════

if run_analysis:
    if not report_text.strip():
        st.markdown('<div class="warn-block">⚠ Please upload or paste a medical report before analysing.</div>', unsafe_allow_html=True)
        st.stop()

    # ── Progress feedback ──
    prog_bar = st.progress(0, text="Tokenising report…")
    time.sleep(0.3)
    prog_bar.progress(30, text="Running inference…")
    specialty, confidence, all_probs = predict_specialty(report_text)
    time.sleep(0.2)
    prog_bar.progress(60, text="Extracting medical terms…")
    matched_terms = get_matched_terms(report_text, top_n=max_terms)
    time.sleep(0.2)
    prog_bar.progress(90, text="Building visualisations…")
    time.sleep(0.1)
    prog_bar.progress(100, text="Done ✓")
    time.sleep(0.3)
    prog_bar.empty()

    # ════════════════════════════════════════
    # RESULT BANNER
    # ════════════════════════════════════════
    conf_pct = int(confidence * 100)
    bar_color = "#00e5aa" if conf_pct >= 80 else "#ffb830" if conf_pct >= 50 else "#ff4b6e"

    st.markdown(f"""
    <div class="result-banner">
        <div>
            <div class="result-label">Predicted Medical Specialty</div>
            <div class="result-specialty">{specialty}</div>
        </div>
        <div class="confidence-bar-wrap">
            <div class="result-label">Confidence Score</div>
            <div style="font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:800;color:{bar_color};">{conf_pct}%</div>
            <div class="confidence-bar-bg">
                <div class="confidence-bar-fill" style="width:{conf_pct}%;background:linear-gradient(90deg,{bar_color}88,{bar_color});box-shadow:0 0 8px {bar_color};"></div>
            </div>
        </div>
        <div>
            <div class="result-label">Patient ID</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.9rem;color:#e8f4fc;">{patient_id}</div>
            <div class="result-label" style="margin-top:0.5rem;">Timestamp</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.75rem;color:#7fa8c4;">{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ════════════════════════════════════════
    # CHARTS — Two columns
    # ════════════════════════════════════════
    set_dark_plot_style()
    chart_col1, chart_col2 = st.columns([1, 1], gap="large")

    # ── Column 1: PE Heatmap ──
    with chart_col1:
        if show_pe:
            st.markdown("""
            <div class="section-header">
                <span class="dot"></span> Positional Encoding Heatmap
            </div>
            """, unsafe_allow_html=True)
            pe = positional_encoding(100, 128)
            fig_pe, ax = plt.subplots(figsize=(6, 3.5))
            im = ax.imshow(pe, aspect="auto", cmap="Blues", vmin=-1, vmax=1)
            plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
            ax.set_xlabel("Embedding Dimension", fontsize=8)
            ax.set_ylabel("Token Position",      fontsize=8)
            ax.set_title("Sinusoidal Positional Encoding", fontsize=9, color="#00c8ff")
            fig_pe.tight_layout()
            st.pyplot(fig_pe, use_container_width=True)
            plt.close(fig_pe)

    # ── Column 2: Confidence Distribution ──
    with chart_col2:
        if show_dist:
            st.markdown("""
            <div class="section-header">
                <span class="dot"></span> Class Probability Distribution
            </div>
            """, unsafe_allow_html=True)
            if model is not None and label_encoder is not None:
                labels = label_encoder.classes_
            else:
                labels = np.array(ALL_SPECIALTIES[:len(all_probs)])
            # Take top-8 by probability
            top_idx  = np.argsort(all_probs)[-8:][::-1]
            top_lbls = [labels[i] if i < len(labels) else f"Class {i}" for i in top_idx]
            top_prob = all_probs[top_idx]

            bar_colors = ["#00c8ff" if lbl == specialty else "#1a3550" for lbl in top_lbls]
            fig_d, ax2 = plt.subplots(figsize=(6, 3.5))
            bars = ax2.barh(top_lbls[::-1], top_prob[::-1] * 100,
                            color=bar_colors[::-1], edgecolor="#1a3550", linewidth=0.5)
            ax2.set_xlabel("Probability (%)", fontsize=8)
            ax2.set_title("Top Class Predictions", fontsize=9, color="#00c8ff")
            ax2.set_xlim(0, 100)
            for bar, val in zip(bars, top_prob[::-1] * 100):
                ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                         f"{val:.1f}%", va="center", fontsize=7.5, color="#7fa8c4")
            fig_d.tight_layout()
            st.pyplot(fig_d, use_container_width=True)
            plt.close(fig_d)

    st.markdown("<hr/>", unsafe_allow_html=True)

    # ════════════════════════════════════════
    # MEDICAL TERMS + WORD STATS
    # ════════════════════════════════════════
    term_col, stat_col = st.columns([1, 1], gap="large")

    with term_col:
        st.markdown("""
        <div class="section-header">
            <span class="dot"></span> Matched Medical Terms
        </div>
        """, unsafe_allow_html=True)

        if len(matched_terms) == 0:
            st.markdown('<div class="info-block">No medical dictionary terms matched in this report.</div>', unsafe_allow_html=True)
        else:
            # Chips
            chip_html = ""
            for _, row in matched_terms.iterrows():
                cls = "high" if row.get("Frequency", 0) > 50 else ""
                chip_html += f'<span class="term-chip {cls}">{row["Medical_Term"]}</span>'
            st.markdown(chip_html, unsafe_allow_html=True)

            st.markdown("<br/>", unsafe_allow_html=True)

            # Bar chart
            fig_t, ax3 = plt.subplots(figsize=(6, 3.5))
            colors_t = ["#00c8ff" if f > 50 else "#1a6680" for f in matched_terms["Frequency"]]
            ax3.barh(matched_terms["Medical_Term"][::-1],
                     matched_terms["Frequency"][::-1],
                     color=colors_t[::-1], edgecolor="#1a3550")
            ax3.set_xlabel("Dictionary Frequency", fontsize=8)
            ax3.set_title(f"Top {len(matched_terms)} Matched Terms", fontsize=9, color="#00c8ff")
            fig_t.tight_layout()
            st.pyplot(fig_t, use_container_width=True)
            plt.close(fig_t)

    with stat_col:
        if include_stats:
            st.markdown("""
            <div class="section-header">
                <span class="dot"></span> Word Frequency (Report)
            </div>
            """, unsafe_allow_html=True)
            word_df = word_statistics(report_text)
            if len(word_df):
                fig_w, ax4 = plt.subplots(figsize=(6, 3.5))
                ax4.bar(word_df["Word"], word_df["Count"],
                        color="#007fa0", edgecolor="#1a3550", linewidth=0.5)
                ax4.set_xlabel("Word", fontsize=8)
                ax4.set_ylabel("Count", fontsize=8)
                ax4.set_title("Top Words in Report (≥4 chars)", fontsize=9, color="#00c8ff")
                plt.xticks(rotation=45, ha="right", fontsize=7)
                fig_w.tight_layout()
                st.pyplot(fig_w, use_container_width=True)
                plt.close(fig_w)

    st.markdown("<hr/>", unsafe_allow_html=True)

    # ════════════════════════════════════════
    # REPORT PREVIEW
    # ════════════════════════════════════════
    if show_prev:
        st.markdown("""
        <div class="section-header">
            <span class="dot"></span> Report Preview
        </div>
        """, unsafe_allow_html=True)
        preview_text = report_text[:1500] + ("…" if len(report_text) > 1500 else "")
        st.markdown(f'<div class="report-preview">{preview_text}</div>', unsafe_allow_html=True)
        st.markdown("<br/>", unsafe_allow_html=True)

    # ════════════════════════════════════════
    # EXPORT
    # ════════════════════════════════════════
    st.markdown("""
    <div class="section-header">
        <span class="dot"></span> Export Report
    </div>
    """, unsafe_allow_html=True)

    exp_col1, exp_col2 = st.columns([1, 3])

    with exp_col1:
        with st.spinner("Generating PDF…"):
            pdf_bytes = generate_pdf(
                specialty, confidence, report_text,
                patient_id, analyst_name, dept,
                matched_terms, all_probs,
            )

        st.download_button(
            label="⬇  Download PDF Report",
            data=pdf_bytes,
            file_name=f"MedScan_Report_{patient_id}_{datetime.date.today()}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    with exp_col2:
        st.markdown(f"""
        <div class="info-block">
            PDF includes: patient metadata · classification result · confidence score ·
            matched medical terms table · report excerpt · generation timestamp.
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════
else:
    st.markdown("""
    <div style="text-align:center;padding:3rem 0 1rem;">
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.75rem;color:#3d6480;
            border:1px dashed #1a3550;border-radius:10px;padding:2rem 3rem;
            display:inline-block;max-width:600px;">
            <div style="font-size:2.5rem;margin-bottom:1rem;">🩺</div>
            <div style="color:#7fa8c4;margin-bottom:0.5rem;">Upload a <b>.txt</b> clinical report or paste text above,<br/>
            then click <b>Analyse Report</b> to begin classification.</div>
            <div style="font-size:0.65rem;color:#3d6480;margin-top:1rem;">
                Supports: Cardiology · Pulmonology · Gastroenterology · Neurology · Oncology<br/>
                Orthopedics · Nephrology · Endocrinology · Hematology · Infectious Disease · and more
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    MedScan AI — Intelligent Medical Report Understanding System &nbsp;·&nbsp;
    Built with Streamlit &amp; TensorFlow &nbsp;·&nbsp;
    For research &amp; informational use only — not a substitute for professional medical advice.
</div>
""", unsafe_allow_html=True)