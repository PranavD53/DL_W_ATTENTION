# DL_WITH_ATTENTION PROJECTS

---

## 🩸 Dexter · Fraud Intelligence System

A deep learning fraud detection dashboard built with Streamlit, TensorFlow (LSTM + Attention), and Plotly — styled in a luxurious dark-red "Dexter" theme.
Upload a transaction CSV (`Time`, `Amount`, optional `Class`) to get sequence-level fraud probabilities, risk-tier breakdowns, attention heatmaps, and a ground-truth confusion matrix.
Use the real-time scorer to manually input transaction features and receive an instant fraud verdict with a risk gauge.
Runs in **Demo Mode** automatically if model files (`fraud_lstm_attention.keras`, `scaler.pkl`, `threshold.pkl`) are absent — no crashes, just simulated analysis.
To launch: `pip install streamlit tensorflow plotly pandas numpy scikit-learn joblib` → `streamlit run app.py`

**Link**: https://dexter-fi.streamlit.app/

---

## MedScan AI — Medical Report Understanding System
An attention-based NLP Streamlit app that classifies medical reports into specialties (Cardiology, Neurology, Oncology, etc.) using a TensorFlow model with positional encoding visualization.
Upload a `.txt` clinical report or paste text directly, and the system predicts the medical specialty with a confidence score, matched medical terms, and word-frequency charts.
All results export as a styled PDF containing patient metadata, classification results, detected terms table, and a report excerpt.
Run locally with `pip install streamlit tensorflow pandas reportlab matplotlib` then `streamlit run medical_analysis_app.py`.

**Link**: https://medscan-dl.streamlit.app/

---

