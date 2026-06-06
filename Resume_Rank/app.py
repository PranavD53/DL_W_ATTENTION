import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import re

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(
    page_title="Aurum Recruit — AI Hiring Intelligence",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -------------------------------------------------------
# LUXURY THEME  (deep navy + champagne gold + ivory)
# -------------------------------------------------------
LUXURY_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&family=Jost:wght@300;400;500&display=swap');

  /* ---- Global reset ---- */
  html, body, [class*="css"] {
    font-family: 'Jost', sans-serif;
    background-color: #0B0F1A !important;
    color: #E8DEC8 !important;
  }

  /* ---- Scrollbar ---- */
  ::-webkit-scrollbar { width: 6px; background: #0B0F1A; }
  ::-webkit-scrollbar-thumb { background: #C9A84C; border-radius: 3px; }

  /* ---- Main container ---- */
  .main .block-container {
    padding: 2.5rem 3.5rem 4rem;
    max-width: 1260px;
  }

  /* ---- Header ---- */
  .luxury-header {
    text-align: center;
    padding: 3.5rem 0 2rem;
    border-bottom: 1px solid rgba(201,168,76,0.25);
    margin-bottom: 2.5rem;
  }
  .luxury-header .wordmark {
    font-family: 'Cormorant Garamond', serif;
    font-size: 3rem;
    font-weight: 300;
    letter-spacing: 0.22em;
    color: #C9A84C;
    text-transform: uppercase;
  }
  .luxury-header .tagline {
    font-size: 0.78rem;
    letter-spacing: 0.3em;
    color: rgba(232,222,200,0.5);
    text-transform: uppercase;
    margin-top: 0.35rem;
  }
  .gold-rule {
    width: 60px;
    height: 1px;
    background: linear-gradient(90deg, transparent, #C9A84C, transparent);
    margin: 0.85rem auto;
  }

  /* ---- Section labels ---- */
  .section-label {
    font-size: 0.68rem;
    letter-spacing: 0.28em;
    text-transform: uppercase;
    color: #C9A84C;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(201,168,76,0.18);
  }

  /* ---- Inputs ---- */
  textarea, .stTextArea textarea {
    background: #111626 !important;
    border: 1px solid rgba(201,168,76,0.3) !important;
    border-radius: 4px !important;
    color: #E8DEC8 !important;
    font-family: 'Jost', sans-serif !important;
    font-size: 0.9rem !important;
    padding: 0.85rem 1rem !important;
    transition: border-color 0.2s ease;
  }
  textarea:focus, .stTextArea textarea:focus {
    border-color: #C9A84C !important;
    box-shadow: 0 0 0 2px rgba(201,168,76,0.12) !important;
    outline: none !important;
  }

  /* ---- File uploader ---- */
  .stFileUploader {
    background: #111626 !important;
    border: 1px dashed rgba(201,168,76,0.35) !important;
    border-radius: 4px !important;
    padding: 1.2rem !important;
  }
  .stFileUploader label, .stFileUploader span, .stFileUploader p {
    color: rgba(232,222,200,0.6) !important;
    font-size: 0.85rem !important;
  }
  [data-testid="stFileUploadDropzone"] {
    background: #111626 !important;
  }

  /* ---- Primary button ---- */
  .stButton > button {
    background: linear-gradient(135deg, #C9A84C, #A87C2A) !important;
    color: #0B0F1A !important;
    border: none !important;
    border-radius: 3px !important;
    font-family: 'Jost', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.22em !important;
    text-transform: uppercase !important;
    padding: 0.75rem 2.5rem !important;
    cursor: pointer !important;
    transition: all 0.25s ease !important;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, #D9BC66, #C9A84C) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 22px rgba(201,168,76,0.25) !important;
  }

  /* ---- Download button ---- */
  .stDownloadButton > button {
    background: transparent !important;
    border: 1px solid rgba(201,168,76,0.45) !important;
    color: #C9A84C !important;
    border-radius: 3px !important;
    font-family: 'Jost', sans-serif !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    padding: 0.6rem 1.8rem !important;
    transition: all 0.2s ease !important;
  }
  .stDownloadButton > button:hover {
    background: rgba(201,168,76,0.1) !important;
    border-color: #C9A84C !important;
  }

  /* ---- Dataframe / table ---- */
  .stDataFrame {
    border: 1px solid rgba(201,168,76,0.2) !important;
    border-radius: 4px !important;
    overflow: hidden;
  }
  .stDataFrame thead tr th {
    background: #161C2D !important;
    color: #C9A84C !important;
    font-family: 'Jost', sans-serif !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid rgba(201,168,76,0.25) !important;
  }
  .stDataFrame tbody tr td {
    color: #E8DEC8 !important;
    font-size: 0.88rem !important;
    border-bottom: 1px solid rgba(255,255,255,0.04) !important;
  }
  .stDataFrame tbody tr:hover td {
    background: rgba(201,168,76,0.06) !important;
  }

  /* ---- Metric cards ---- */
  .metric-row { display: flex; gap: 1rem; margin-bottom: 2rem; }
  .metric-card {
    flex: 1;
    background: #111626;
    border: 1px solid rgba(201,168,76,0.18);
    border-radius: 4px;
    padding: 1.2rem 1.5rem;
  }
  .metric-card .m-label {
    font-size: 0.68rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: rgba(232,222,200,0.45);
    margin-bottom: 0.5rem;
  }
  .metric-card .m-value {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.2rem;
    font-weight: 300;
    color: #C9A84C;
    line-height: 1;
  }
  .metric-card .m-sub {
    font-size: 0.75rem;
    color: rgba(232,222,200,0.4);
    margin-top: 0.3rem;
  }

  /* ---- Skill pill tags ---- */
  .skills-container { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 0.5rem; }
  .skill-pill {
    background: rgba(201,168,76,0.1);
    border: 1px solid rgba(201,168,76,0.3);
    color: #C9A84C;
    font-size: 0.73rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 2px;
  }

  /* ---- Rank badge ---- */
  .rank-badge {
    display: inline-block;
    width: 28px; height: 28px;
    line-height: 28px;
    text-align: center;
    border-radius: 50%;
    font-size: 0.8rem;
    font-weight: 500;
  }
  .rank-1 { background: #C9A84C; color: #0B0F1A; }
  .rank-2 { background: rgba(201,168,76,0.35); color: #C9A84C; }
  .rank-3 { background: rgba(201,168,76,0.18); color: #C9A84C; }

  /* ---- Progress bar (score bar) ---- */
  .score-bar-wrap {
    background: rgba(255,255,255,0.06);
    border-radius: 2px;
    height: 5px;
    overflow: hidden;
    margin-top: 4px;
  }
  .score-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #A87C2A, #C9A84C);
    border-radius: 2px;
    transition: width 0.6s ease;
  }

  /* ---- Alerts ---- */
  .stWarning, .stError, .stSuccess, .stInfo {
    border-radius: 3px !important;
    font-size: 0.85rem !important;
  }

  /* ---- Sidebar (collapsed) ---- */
  [data-testid="stSidebar"] { background: #0D1120 !important; }

  /* ---- Divider ---- */
  hr { border-color: rgba(201,168,76,0.12) !important; margin: 2rem 0 !important; }

  /* ---- Chart backgrounds ---- */
  .element-container .stImage { border-radius: 4px; }

  /* ---- Subheader overrides ---- */
  h1, h2, h3 {
    font-family: 'Cormorant Garamond', serif !important;
    font-weight: 300 !important;
    color: #E8DEC8 !important;
    letter-spacing: 0.04em;
  }

  /* ---- Spinner ---- */
  .stSpinner > div { border-top-color: #C9A84C !important; }

  /* ---- Tooltip helper ---- */
  .tooltip-text {
    font-size: 0.78rem;
    color: rgba(232,222,200,0.4);
    margin-top: 0.25rem;
  }
</style>
"""

st.markdown(LUXURY_CSS, unsafe_allow_html=True)

# -------------------------------------------------------
# HEADER
# -------------------------------------------------------
st.markdown("""
<div class="luxury-header">
  <div class="wordmark">Aurum Recruit</div>
  <div class="gold-rule"></div>
  <div class="tagline">AI-Powered Hiring Intelligence &nbsp;·&nbsp; Precision Candidate Ranking</div>
</div>
""", unsafe_allow_html=True)


# -------------------------------------------------------
# PDF TEXT EXTRACTION
# -------------------------------------------------------
def extract_text(file):
    try:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + " "
        return text.lower().strip()
    except Exception as e:
        return ""


# -------------------------------------------------------
# POSITIONAL ENCODING
# -------------------------------------------------------
def positional_encoding(max_len, d_model=16):
    PE = np.zeros((max_len, d_model))
    for pos in range(max_len):
        for i in range(0, d_model, 2):
            angle = pos / np.power(10000, (2 * (i // 2)) / d_model)
            PE[pos, i] = np.sin(angle)
            if i + 1 < d_model:
                PE[pos, i + 1] = np.cos(angle)
    return PE


# -------------------------------------------------------
# CLEAN CANDIDATE NAME  (strip .pdf extension etc.)
# -------------------------------------------------------
def clean_name(filename):
    name = os.path.splitext(filename)[0]
    name = re.sub(r'[_\-]+', ' ', name)
    return name.title()


# -------------------------------------------------------
# LUXURY MATPLOTLIB STYLE
# -------------------------------------------------------
def apply_luxury_style():
    plt.rcParams.update({
        'figure.facecolor': '#111626',
        'axes.facecolor':   '#111626',
        'axes.edgecolor': (201/255, 168/255, 76/255, 0.25),
        'axes.labelcolor':  '#C9A84C',
        'xtick.color':      (232/255,222/255,200/255,0.45),
        'ytick.color':      (232/255,222/255,200/255,0.45),
        'text.color':       '#E8DEC8',
        'grid.color':       (201/255,168/255,76/255,0.08),
        'grid.linestyle':   '--',
        'font.family':      'sans-serif',
        'axes.spines.top':  False,
        'axes.spines.right':False,
    })


# -------------------------------------------------------
# TWO-COLUMN INPUT LAYOUT
# -------------------------------------------------------
col_jd, col_up = st.columns([1.1, 0.9], gap="large")

with col_jd:
    st.markdown('<div class="section-label">Job Description</div>', unsafe_allow_html=True)
    jd = st.text_area(
        label="jd_input",
        label_visibility="collapsed",
        placeholder="Paste the full job description here — required skills, responsibilities, qualifications…",
        height=220,
    )

with col_up:
    st.markdown('<div class="section-label">Candidate Résumés</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        label="resumes_upload",
        label_visibility="collapsed",
        type=["pdf"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        st.markdown(
            f'<p class="tooltip-text">✦ {len(uploaded_files)} résumé{"s" if len(uploaded_files)>1 else ""} queued for analysis</p>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<p class="tooltip-text">Upload one or more PDF résumés</p>',
            unsafe_allow_html=True,
        )

# ---- Centered CTA ----
st.markdown("<br>", unsafe_allow_html=True)
cta_col = st.columns([2, 1, 2])[1]
with cta_col:
    run = st.button("⟡  Analyse Candidates")


# -------------------------------------------------------
# MAIN ANALYSIS
# -------------------------------------------------------
if run:

    if not jd.strip() or not uploaded_files:
        st.warning("Please provide both a job description and at least one résumé PDF.")
        st.stop()

    with st.spinner("Evaluating candidates…"):

        resumes, names, raw_names = [], [], []

        for file in uploaded_files:
            text = extract_text(file)
            if text:
                resumes.append(text)
                names.append(clean_name(file.name))
                raw_names.append(file.name)
            else:
                st.warning(f"Could not extract text from **{file.name}** — skipping.")

        if not resumes:
            st.error("No readable résumés found. Please check your PDFs.")
            st.stop()

        # ---- TF-IDF ranking ----
        vectorizer = TfidfVectorizer(stop_words="english", min_df=1, ngram_range=(1, 2))
        docs = resumes + [jd.lower()]
        vectors = vectorizer.fit_transform(docs)

        resume_vecs = vectors[:-1]
        jd_vec = vectors[-1]
        scores = cosine_similarity(jd_vec, resume_vecs).flatten()

        df = pd.DataFrame({
            "Rank": range(1, len(names) + 1),
            "Candidate": names,
            "Match Score": (scores * 100).round(1),
        }).sort_values("Match Score", ascending=False).reset_index(drop=True)
        df["Rank"] = range(1, len(df) + 1)

        feature_names = vectorizer.get_feature_names_out()

    # ---- Divider ----
    st.markdown("<hr>", unsafe_allow_html=True)

    # ---- Summary metrics ----
    top_score  = df["Match Score"].iloc[0]
    avg_score  = df["Match Score"].mean()
    spread     = df["Match Score"].max() - df["Match Score"].min()

    metric_html = f"""
    <div class="metric-row">
      <div class="metric-card">
        <div class="m-label">Candidates Evaluated</div>
        <div class="m-value">{len(df)}</div>
        <div class="m-sub">résumés processed</div>
      </div>
      <div class="metric-card">
        <div class="m-label">Top Match Score</div>
        <div class="m-value">{top_score:.1f}<span style="font-size:1rem; color:rgba(201,168,76,0.6)">%</span></div>
        <div class="m-sub">{df["Candidate"].iloc[0]}</div>
      </div>
      <div class="metric-card">
        <div class="m-label">Average Score</div>
        <div class="m-value">{avg_score:.1f}<span style="font-size:1rem; color:rgba(201,168,76,0.6)">%</span></div>
        <div class="m-sub">across all résumés</div>
      </div>
      <div class="metric-card">
        <div class="m-label">Score Spread</div>
        <div class="m-value">{spread:.1f}<span style="font-size:1rem; color:rgba(201,168,76,0.6)">%</span></div>
        <div class="m-sub">max − min</div>
      </div>
    </div>
    """
    st.markdown(metric_html, unsafe_allow_html=True)

    # ---- Ranked candidates table (custom HTML) ----
    st.markdown('<div class="section-label">Ranked Candidates</div>', unsafe_allow_html=True)

    rows_html = ""
    for _, row in df.iterrows():
        rank = int(row["Rank"])
        badge_cls = f"rank-{min(rank, 3)}"
        score_pct = row["Match Score"]
        rows_html += f"""
        <tr>
          <td style="padding:12px 16px;"><span class="rank-badge {badge_cls}">{rank}</span></td>
          <td style="padding:12px 16px; font-size:0.92rem;">{row["Candidate"]}</td>
          <td style="padding:12px 16px; min-width:200px;">
            <span style="font-family:'Cormorant Garamond',serif;font-size:1.15rem;color:#C9A84C;">{score_pct:.1f}%</span>
            <div class="score-bar-wrap">
              <div class="score-bar-fill" style="width:{score_pct}%;"></div>
            </div>
          </td>
        </tr>
        """

    table_html = f"""
    <table style="width:100%;border-collapse:collapse;background:#111626;border:1px solid rgba(201,168,76,0.18);border-radius:4px;overflow:hidden;margin-bottom:1.5rem;">
      <thead>
        <tr style="background:#0D1220;">
          <th style="padding:10px 16px;text-align:left;font-size:0.67rem;letter-spacing:0.22em;text-transform:uppercase;color:rgba(201,168,76,0.7);font-weight:500;border-bottom:1px solid rgba(201,168,76,0.18);">Rank</th>
          <th style="padding:10px 16px;text-align:left;font-size:0.67rem;letter-spacing:0.22em;text-transform:uppercase;color:rgba(201,168,76,0.7);font-weight:500;border-bottom:1px solid rgba(201,168,76,0.18);">Candidate</th>
          <th style="padding:10px 16px;text-align:left;font-size:0.67rem;letter-spacing:0.22em;text-transform:uppercase;color:rgba(201,168,76,0.7);font-weight:500;border-bottom:1px solid rgba(201,168,76,0.18);">Match Score</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
    """
    st.markdown(table_html, unsafe_allow_html=True)

    # ---- Download ----
    csv = df.to_csv(index=False).encode("utf-8")
    dl_col = st.columns([1, 2, 1])[1]
    with dl_col:
        st.download_button("↓  Export Rankings CSV", csv, "aurum_rankings.csv", "text/csv")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ---- Top candidate insights ----
    st.markdown('<div class="section-label">Top Candidate Insights</div>', unsafe_allow_html=True)

    top_idx_in_orig = list(df.index)
    # Map back to original resumes list
    top_rank_candidate = df["Candidate"].iloc[0]
    # Find the index by matching sorted order
    sorted_scores_idx = np.argsort(scores)[::-1]
    top_orig_idx = sorted_scores_idx[0]
    top_resume = resumes[top_orig_idx]
    top_vector = resume_vecs[top_orig_idx].toarray().flatten()

    top_indices = top_vector.argsort()[-18:][::-1]
    top_words = [feature_names[i] for i in top_indices if top_vector[i] > 0]

    st.markdown(
        f'<p style="font-family:\'Cormorant Garamond\',serif;font-size:1.35rem;font-weight:300;color:#E8DEC8;margin-bottom:0.65rem;">'
        f'{top_rank_candidate} <span style="color:rgba(201,168,76,0.7);font-size:1rem;">— #1 ranked candidate</span></p>',
        unsafe_allow_html=True,
    )

    # Skills pills
    pills_html = '<div class="skills-container">'
    for w in top_words:
        pills_html += f'<span class="skill-pill">{w}</span>'
    pills_html += '</div>'
    st.markdown('<p style="font-size:0.73rem;letter-spacing:0.15em;text-transform:uppercase;color:rgba(232,222,200,0.45);margin-bottom:0.4rem;">Extracted Keywords & Skills</p>', unsafe_allow_html=True)
    st.markdown(pills_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Charts ----
    apply_luxury_style()
    chart_col1, chart_col2 = st.columns(2, gap="large")

    GOLD   = "#C9A84C"
    GOLD2  = "#A87C2A"
    BG     = "#111626"
    SURFACE= "#0D1220"
    TEXT   = "#E8DEC8"

    with chart_col1:
        st.markdown('<div class="section-label">Score Distribution</div>', unsafe_allow_html=True)

        fig1, ax1 = plt.subplots(figsize=(6, 3.5))
        cands = df["Candidate"].tolist()
        sc    = df["Match Score"].tolist()
        bars  = ax1.barh(
            cands[::-1], sc[::-1],
            color=[GOLD if i == 0 else f"rgba(201,168,76,{0.25 + 0.12*i})" for i in range(len(cands))[::-1]],
            height=0.55,
        )
        ax1.set_xlim(0, 100)
        ax1.set_xlabel("Match Score (%)", fontsize=9, color=GOLD, labelpad=8)
        ax1.tick_params(axis='y', labelsize=9, colors=TEXT)
        ax1.tick_params(axis='x', labelsize=8)
        for bar, val in zip(bars, sc[::-1]):
            ax1.text(val + 1, bar.get_y() + bar.get_height()/2,
                     f"{val:.1f}%", va='center', ha='left',
                     fontsize=8, color=GOLD)
        ax1.grid(axis='x')
        fig1.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

    with chart_col2:
        st.markdown('<div class="section-label">Positional Encoding</div>', unsafe_allow_html=True)

        tokens = top_resume.split()[:40]
        if len(tokens) < 2:
            tokens = ["token"] * 20
        pe = positional_encoding(len(tokens), 16)

        gold_cmap = mcolors.LinearSegmentedColormap.from_list(
            "aurum", [SURFACE, "#6B4F1A", GOLD, "#F0D98A"]
        )
        fig2, ax2 = plt.subplots(figsize=(6, 3.5))
        im = ax2.imshow(pe, aspect="auto", cmap=gold_cmap, interpolation='nearest')
        ax2.set_xlabel("Embedding Dimension", fontsize=9, color=GOLD, labelpad=8)
        ax2.set_ylabel("Token Position",       fontsize=9, color=GOLD, labelpad=8)
        ax2.tick_params(labelsize=8)
        cbar = fig2.colorbar(im, ax=ax2, fraction=0.03, pad=0.02)
        cbar.ax.tick_params(labelsize=7, colors=TEXT)
        cbar.outline.set_edgecolor(GOLD + "55")
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    st.markdown("<br>", unsafe_allow_html=True)
    chart_col3, chart_col4 = st.columns(2, gap="large")

    with chart_col3:
        st.markdown('<div class="section-label">Self-Attention Pattern (Simulated)</div>', unsafe_allow_html=True)

        attn = np.random.rand(len(tokens), len(tokens))
        # Make it more realistic — diagonal + spread
        for i in range(len(tokens)):
            attn[i, i] = 0.8 + 0.2 * np.random.rand()
            if i > 0:
                attn[i, i-1] = 0.5 + 0.2 * np.random.rand()

        gold_attn = mcolors.LinearSegmentedColormap.from_list(
            "attn", ["#0B0F1A", "#3A2800", GOLD2, GOLD, "#F5E4A0"]
        )
        fig3, ax3 = plt.subplots(figsize=(6, 3.5))
        im3 = ax3.imshow(attn, aspect="auto", cmap=gold_attn, interpolation='nearest')
        ax3.set_xlabel("Key Tokens",   fontsize=9, color=GOLD, labelpad=8)
        ax3.set_ylabel("Query Tokens", fontsize=9, color=GOLD, labelpad=8)
        ax3.tick_params(labelsize=8)
        cbar3 = fig3.colorbar(im3, ax=ax3, fraction=0.03, pad=0.02)
        cbar3.ax.tick_params(labelsize=7, colors=TEXT)
        cbar3.outline.set_edgecolor(GOLD + "55")
        fig3.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

    with chart_col4:
        st.markdown('<div class="section-label">Top Keywords — TF-IDF Weight</div>', unsafe_allow_html=True)

        top8_words = top_words[:8]
        top8_vals  = [top_vector[np.where(feature_names == w)[0][0]] for w in top8_words if w in feature_names]
        # Align lengths
        paired = [(w, v) for w, v in zip(top8_words, top8_vals)]
        paired.sort(key=lambda x: x[1], reverse=True)
        kw, kv = zip(*paired) if paired else ([], [])

        fig4, ax4 = plt.subplots(figsize=(6, 3.5))
        colors4 = [GOLD if i == 0 else GOLD2 for i in range(len(kw))]
        ax4.bar(range(len(kw)), kv, color=colors4, width=0.6)
        ax4.set_xticks(range(len(kw)))
        ax4.set_xticklabels(kw, rotation=35, ha='right', fontsize=8, color=TEXT)
        ax4.set_ylabel("TF-IDF Weight", fontsize=9, color=GOLD, labelpad=8)
        ax4.tick_params(axis='y', labelsize=8)
        ax4.grid(axis='y')
        fig4.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)

    # ---- Footer ----
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;padding:1.5rem 0 0.5rem;">
      <span style="font-size:0.7rem;letter-spacing:0.22em;text-transform:uppercase;color:rgba(201,168,76,0.35);">
        Aurum Recruit &nbsp;·&nbsp; Analysis Complete &nbsp;·&nbsp; ✦
      </span>
    </div>
    """, unsafe_allow_html=True)