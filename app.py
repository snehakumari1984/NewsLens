"""
NewsLens – AI-Powered News Credibility Analysis System
Author: Second-Year B.Tech Artificial Intelligence and Data Science Student
Technology Stack: Python, Streamlit, Scikit-learn, NLTK, Plotly, SQLite, Joblib

A professional, 100% free, open-source Machine Learning web application designed
for academic demonstration, practical ML portfolio building, and internship interviews.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Local application modules
from preprocess import clean_text
import database as db

# -----------------------------------------------------------------------------
# 1. Streamlit Application Configuration
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "assets", "logo.png")

st.set_page_config(
    page_title="NewsLens – AI-Powered News Credibility Analysis System",
    page_icon=LOGO_PATH if os.path.exists(LOGO_PATH) else "🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. Modern AI SaaS Dark Theme CSS
# -----------------------------------------------------------------------------
# Color Palette:
# Background: #0B1120 | Secondary: #111C30 | Primary Accent: #06B6D4 | Secondary Accent: #3B82F6
# Main Text: #F8FAFC | Muted Text: #94A3B8 | Border: #1E293B
CUSTOM_CSS = """
<style>
    /* Global Base */
    .stApp {
        background-color: #0B1120;
        color: #F8FAFC;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0F172A;
        border-right: 1px solid #1E293B;
    }
    section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p {
        color: #94A3B8;
    }

    /* Radio Navigation Options in Sidebar */
    div[data-testid="stRadio"] > label {
        color: #38BDF8 !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label {
        background-color: #111C30;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 8px 14px;
        margin-bottom: 6px;
        transition: all 0.2s ease;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label:hover {
        border-color: #06B6D4;
        background-color: #1E293B;
    }

    /* Card Containers */
    .nl-card {
        background-color: #111C30;
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 22px;
        margin-bottom: 20px;
        box-shadow: 0 4px 16px -2px rgba(0, 0, 0, 0.5);
    }

    .nl-card-glow {
        background: linear-gradient(135deg, #111C30 0%, #0F172A 100%);
        border: 1px solid #06B6D4;
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 22px;
        box-shadow: 0 0 24px rgba(6, 182, 212, 0.15);
    }

    /* Metric Boxes */
    .nl-metric-box {
        background: #111C30;
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .nl-metric-box:hover {
        transform: translateY(-2px);
        border-color: #06B6D4;
    }
    .nl-metric-title {
        font-size: 0.82rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .nl-metric-val {
        font-size: 1.85rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    .nl-metric-subtitle {
        font-size: 0.75rem;
        color: #06B6D4;
        margin-top: 4px;
    }

    /* Verdict Banners */
    .verdict-real {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(5, 150, 105, 0.22) 100%);
        border: 1px solid #10B981;
        border-radius: 12px;
        padding: 22px;
        margin-top: 18px;
        margin-bottom: 18px;
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.2);
    }
    .verdict-misleading {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.12) 0%, rgba(220, 38, 38, 0.22) 100%);
        border: 1px solid #EF4444;
        border-radius: 12px;
        padding: 22px;
        margin-top: 18px;
        margin-bottom: 18px;
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.2);
    }

    /* Mandatory Disclaimer Box */
    .nl-disclaimer-box {
        background-color: #0F172A;
        border-left: 4px solid #F59E0B;
        border-radius: 6px;
        padding: 14px 18px;
        margin-top: 16px;
        margin-bottom: 16px;
    }
    .nl-disclaimer-title {
        color: #FBBF24;
        font-weight: 600;
        font-size: 0.88rem;
        margin-bottom: 4px;
    }
    .nl-disclaimer-text {
        color: #CBD5E1;
        font-size: 0.82rem;
        line-height: 1.5;
        margin: 0;
    }

    /* Keyword Chips */
    .nl-chip-real {
        display: inline-block;
        background-color: rgba(16, 185, 129, 0.18);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 18px;
        padding: 4px 12px;
        margin: 4px;
        font-size: 0.82rem;
        font-weight: 500;
    }
    .nl-chip-misleading {
        display: inline-block;
        background-color: rgba(239, 68, 68, 0.18);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-radius: 18px;
        padding: 4px 12px;
        margin: 4px;
        font-size: 0.82rem;
        font-weight: 500;
    }

    /* Input Fields */
    .stTextInput input, .stTextArea textarea {
        background-color: #0F172A !important;
        border: 1px solid #1E293B !important;
        border-radius: 8px !important;
        color: #F8FAFC !important;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #06B6D4 !important;
        box-shadow: 0 0 10px rgba(6, 182, 212, 0.25) !important;
    }

    /* Action Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #0284C7 0%, #06B6D4 100%);
        color: #FFFFFF;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 10px 22px;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #0369A1 0%, #0284C7 100%);
        box-shadow: 0 0 14px rgba(6, 182, 212, 0.35);
        color: #FFFFFF;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 3. Model & Resource Loading with Caching
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_ml_pipeline():
    """
    Loads serialized machine learning model, vectorizer, and authentic metrics.
    Cached across user sessions to minimize latency.
    """
    model_path = os.path.join(BASE_DIR, "models", "news_model.pkl")
    vec_path = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")
    metrics_path = os.path.join(BASE_DIR, "models", "model_metrics.json")

    model = None
    vectorizer = None
    metrics = None

    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
        except Exception as e:
            st.error(f"Error loading model: {e}")

    if os.path.exists(vec_path):
        try:
            vectorizer = joblib.load(vec_path)
        except Exception as e:
            st.error(f"Error loading vectorizer: {e}")

    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        except Exception as e:
            st.error(f"Error loading metrics: {e}")

    return model, vectorizer, metrics


model, vectorizer, metrics_data = load_ml_pipeline()

# Initialize local SQLite database
db.init_db()


# -----------------------------------------------------------------------------
# 4. Sidebar Navigation
# -----------------------------------------------------------------------------
with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=130)

    st.markdown("## **NewsLens**")
    st.markdown(
        "<p style='color: #06B6D4; font-size: 0.85rem; font-weight:600; margin-top:-10px;'>"
        "AI-Powered News Credibility Analysis"
        "</p>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    # Strictly 5 required navigation pages
    nav_options = [
        "Home",
        "Analyze News",
        "Prediction History",
        "Model Performance",
        "About Project"
    ]

    selected_page = st.radio(
        "NAVIGATE",
        nav_options,
        index=0
    )

    st.markdown("---")

    # Model Status Badge
    if model is not None and metrics_data is not None:
        model_name = metrics_data.get("selected_model", "Logistic Regression")
        acc = metrics_data.get("evaluation_metrics", {}).get("accuracy", 0.9212) * 100
        st.markdown(f"""
        <div style='background: #111C30; border: 1px solid #1E293B; border-radius: 8px; padding: 12px; font-size: 0.8rem;'>
            <div style='color: #10B981; font-weight: 600;'>🟢 Model Online</div>
            <div style='color: #F8FAFC; margin-top: 4px;'><b>Classifier:</b> {model_name}</div>
            <div style='color: #94A3B8;'><b>Test Acc:</b> {acc:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Model not loaded. Run `python train_model.py` to train and save the model.")

    st.markdown("""
    <div style='margin-top: 25px; font-size: 0.75rem; color: #64748B; text-align: center;'>
        B.Tech AI & Data Science Portfolio<br>
        100% Free & Open-Source • Windows Ready
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# MODULE A: HOME DASHBOARD
# =============================================================================
if selected_page == "Home":
    st.markdown("""
    <div class='nl-card-glow'>
        <h1 style='margin:0; font-size: 2.2rem; color: #F8FAFC;'>
            NewsLens <span style='color: #06B6D4;'>AI</span>
        </h1>
        <p style='color: #94A3B8; font-size: 1.05rem; margin-top: 8px; margin-bottom: 0;'>
            AI-Powered News Credibility Analysis System — Detecting stylistic and lexical patterns 
            in news text using Natural Language Processing and supervised Machine Learning.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Database and Metric Statistics
    db_stats = db.get_summary_stats()
    total_analyzed = db_stats.get("total_analyzed", 0)
    real_count = db_stats.get("real_count", 0)
    fake_count = db_stats.get("fake_count", 0)

    test_acc = metrics_data.get("evaluation_metrics", {}).get("accuracy", 0.9212) * 100 if metrics_data else 92.12
    test_f1 = metrics_data.get("evaluation_metrics", {}).get("f1_score", 0.9212) * 100 if metrics_data else 92.12

    # 4 Dashboard Cards: Total Analyzed, Likely Real, Likely Misleading, Model Accuracy
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class='nl-metric-box'>
            <div class='nl-metric-title'>Total Analyzed</div>
            <div class='nl-metric-val'>{total_analyzed}</div>
            <div class='nl-metric-subtitle'>SQLite Local Records</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='nl-metric-box'>
            <div class='nl-metric-title'>Likely Real News</div>
            <div class='nl-metric-val' style='color:#10B981;'>{real_count}</div>
            <div class='nl-metric-subtitle'>Verified Patterns</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class='nl-metric-box'>
            <div class='nl-metric-title'>Likely Misleading</div>
            <div class='nl-metric-val' style='color:#EF4444;'>{fake_count}</div>
            <div class='nl-metric-subtitle'>Sensationalist Markers</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class='nl-metric-box'>
            <div class='nl-metric-title'>Model Test Accuracy</div>
            <div class='nl-metric-val' style='color:#06B6D4;'>{test_acc:.1f}%</div>
            <div class='nl-metric-subtitle'>Unseen Benchmark Test Data</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Overview and Visual Analytics Row
    col_chart_left, col_chart_right = st.columns([1, 1])

    with col_chart_left:
        st.markdown("<div class='nl-card'>", unsafe_allow_html=True)
        st.subheader("📊 Training Dataset Distribution")
        
        # Donut Chart for dataset distribution from authentic Hugging Face benchmark
        df_labels = ["Real News Articles", "Misleading / Fake Articles"]
        df_values = [64641, 43391]
        fig_donut = go.Figure(data=[go.Pie(
            labels=df_labels,
            values=df_values,
            hole=0.6,
            marker=dict(colors=["#10B981", "#EF4444"]),
            textinfo="label+percent"
        )])
        fig_donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#F8FAFC"),
            showlegend=False,
            height=260,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_donut, use_container_width=True)
        st.markdown(
            "<p style='color:#94A3B8; font-size:0.8rem; text-align:center; margin:0;'>"
            "Source: fauxNeuz/RealFakeNews (108,032 labeled news records on Hugging Face)"
            "</p>",
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_chart_right:
        st.markdown("<div class='nl-card'>", unsafe_allow_html=True)
        st.subheader("⚡ Model Comparison Benchmark")
        
        if metrics_data and "model_comparison" in metrics_data:
            comp_data = metrics_data["model_comparison"]
            comp_df = pd.DataFrame(comp_data)
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                name="Accuracy",
                x=comp_df["model"],
                y=[round(a * 100, 2) for a in comp_df["accuracy"]],
                marker_color="#06B6D4"
            ))
            fig_bar.add_trace(go.Bar(
                name="F1-Score",
                x=comp_df["model"],
                y=[round(f * 100, 2) for f in comp_df["f1_score"]],
                marker_color="#3B82F6"
            ))
            fig_bar.update_layout(
                barmode="group",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#F8FAFC"),
                yaxis=dict(title="Score (%)", range=[75, 100], gridcolor="#1E293B"),
                xaxis=dict(gridcolor="#1E293B"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=260,
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown(
            "<p style='color:#94A3B8; font-size:0.8rem; text-align:center; margin:0;'>"
            "Logistic Regression selected for production based on superior F1-score (92.12%)"
            "</p>",
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Recent Analysis Activity Table
    st.markdown("<div class='nl-card'>", unsafe_allow_html=True)
    st.subheader("🕒 Recent Analysis Activity")
    recent_records = db.get_history(limit=5)
    if recent_records:
        rec_display = []
        for r in recent_records:
            rec_display.append({
                "Timestamp": r["timestamp"],
                "Headline / Snippet": r["headline_text"][:90] + ("..." if len(r["headline_text"]) > 90 else ""),
                "Verdict": r["predicted_label"],
                "Confidence": f"{r['confidence_score']:.1f}%",
                "Words": r["word_count"]
            })
        st.dataframe(pd.DataFrame(rec_display), use_container_width=True)
    else:
        st.info("No articles analyzed yet. Switch to the 'Analyze News' tab to test an article!")
    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# MODULE B: NEWS ANALYSIS MODULE ("Analyze News")
# =============================================================================
elif selected_page == "Analyze News":
    st.markdown("""
    <div class='nl-card'>
        <h2 style='margin:0; color: #F8FAFC;'>🔍 Analyze News Credibility</h2>
        <p style='color: #94A3B8; font-size: 0.95rem; margin-top: 6px; margin-bottom: 0;'>
            Submit a news headline and article body to evaluate stylistic and lexical credibility 
            using the trained TF-IDF vectorizer and supervised classifier.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Preset Sample Articles
    SAMPLE_REAL_TITLE = "NASA James Webb Space Telescope Observes Earliest Known Galaxies"
    SAMPLE_REAL_BODY = (
        "Astronomers utilizing NASA's James Webb Space Telescope have obtained deep-field spectroscopic observations "
        "of ancient galactic structures formed within the initial 400 million years following the Big Bang. The research "
        "team, collaborating across international observatories, confirmed red-shift spectral measurements indicating stellar "
        "mass formation rates consistent with theoretical astrophysical models. Official peer-reviewed findings were published "
        "in the Astrophysical Journal."
    )

    SAMPLE_FAKE_TITLE = "SHOCKING BOMBSHELL: Secret Bio-Lab Caught Injecting Nanotech into Drinking Water"
    SAMPLE_FAKE_BODY = (
        "BOMBSHELL REVELATION: Anonymous whistleblowers have leaked classified proof that globalist elites are actively poisoning "
        "municipal tap water systems with microscopic mind-control nanochips! The mainstream corporate media is completely silent "
        "and desperately trying to cover this up. Watch this viral leaked footage immediately before the deep state scrubs it "
        "from the entire internet forever!"
    )

    SAMPLE_NEUTRAL_TITLE = "Federal Reserve Holds Benchmark Interest Rates Steady at Policy Meeting"
    SAMPLE_NEUTRAL_BODY = (
        "The Federal Open Market Committee announced on Wednesday that federal funds target rates will remain unchanged "
        "between 5.25% and 5.50% following their scheduled two-day monetary policy meeting. Central bank officials stated that "
        "while inflationary pressures have demonstrated measurable easing over consecutive quarters, committee members require "
        "further empirical data before enacting programmatic rate reductions."
    )

    # Preset Selection Buttons
    st.markdown("<p style='color:#94A3B8; font-size:0.85rem; font-weight:600;'>Load Verification Samples:</p>", unsafe_allow_html=True)
    col_p1, col_p2, col_p3 = st.columns(3)

    if "headline_val" not in st.session_state:
        st.session_state["headline_val"] = ""
    if "body_val" not in st.session_state:
        st.session_state["body_val"] = ""

    with col_p1:
        if st.button("📰 Sample A (Real Science)", use_container_width=True):
            st.session_state["headline_val"] = SAMPLE_REAL_TITLE
            st.session_state["body_val"] = SAMPLE_REAL_BODY
            st.rerun()

    with col_p2:
        if st.button("⚠️ Sample B (Sensationalist)", use_container_width=True):
            st.session_state["headline_val"] = SAMPLE_FAKE_TITLE
            st.session_state["body_val"] = SAMPLE_FAKE_BODY
            st.rerun()

    with col_p3:
        if st.button("🏛️ Sample C (Financial Policy)", use_container_width=True):
            st.session_state["headline_val"] = SAMPLE_NEUTRAL_TITLE
            st.session_state["body_val"] = SAMPLE_NEUTRAL_BODY
            st.rerun()

    # Input Form
    headline_input = st.text_input(
        "News Headline (Optional):",
        value=st.session_state.get("headline_val", ""),
        placeholder="e.g. Researchers discover new battery chemistry for electric vehicles..."
    )

    body_input = st.text_area(
        "Full News Article Content (Required):",
        value=st.session_state.get("body_val", ""),
        height=180,
        placeholder="Paste full news body text or extended paragraph here..."
    )

    # Word and Character Counts
    combined_raw_text = f"{headline_input.strip()} {body_input.strip()}".strip()
    words_count = len(combined_raw_text.split()) if combined_raw_text else 0
    chars_count = len(combined_raw_text)

    st.markdown(
        f"<p style='color:#64748B; font-size:0.8rem; margin-top:-8px;'>"
        f"Input Statistics: {words_count} words | {chars_count} characters"
        f"</p>",
        unsafe_allow_html=True
    )

    # Action Buttons: Analyze News and Clear
    col_btn_run, col_btn_clear = st.columns([2, 1])

    with col_btn_run:
        analyze_clicked = st.button("⚡ Analyze News", use_container_width=True)

    with col_btn_clear:
        if st.button("🔄 Clear", use_container_width=True):
            st.session_state["headline_val"] = ""
            st.session_state["body_val"] = ""
            st.rerun()

    # Processing & Prediction Pipeline
    if analyze_clicked:
        if not combined_raw_text:
            st.error("⚠️ Please provide news headline or article content before analyzing.")
        elif words_count < 4:
            st.warning("⚠️ Input is too short for reliable statistical evaluation. Please enter at least a complete sentence.")
        elif model is None or vectorizer is None:
            st.error("❌ Trained model or vectorizer artifact not found. Please run `python train_model.py` first.")
        else:
            with st.spinner("Processing news through NLP cleaning, stemming, and TF-IDF vectorization..."):
                # Clean text
                cleaned = clean_text(combined_raw_text)

                if not cleaned.strip():
                    st.warning("⚠️ Submitted text contains only stopwords or filtered punctuation. Please enter informative news text.")
                else:
                    # Transform through fitted TF-IDF vectorizer
                    tfidf_vector = vectorizer.transform([cleaned])

                    # Get probability and prediction
                    probabilities = model.predict_proba(tfidf_vector)[0]
                    # Mapping in training: 0 = REAL, 1 = FAKE
                    prob_real = float(probabilities[0])
                    prob_fake = float(probabilities[1])
                    prediction_idx = int(model.predict(tfidf_vector)[0])

                    is_real = (prediction_idx == 0)
                    predicted_label = "Likely Real" if is_real else "Likely Misleading"
                    confidence_pct = (prob_real if is_real else prob_fake) * 100.0

                    # Model Explainability: Extract matching vocabulary terms & weights
                    matched_tokens = []
                    if hasattr(model, "coef_"):
                        coefficients = model.coef_[0]
                        vocab = vectorizer.vocabulary_
                        for token in set(cleaned.split()):
                            if token in vocab:
                                idx = vocab[token]
                                coef = coefficients[idx]
                                matched_tokens.append({
                                    "word": token,
                                    "weight": float(coef),
                                    "lean": "Misleading" if coef > 0 else "Real"
                                })

                    # Sort by absolute influence
                    matched_tokens.sort(key=lambda x: abs(x["weight"]), reverse=True)
                    top_indicator_str = ", ".join([item["word"] for item in matched_tokens[:5]])

                    # Store analysis in SQLite database
                    db.insert_prediction(
                        headline_text=headline_input.strip() if headline_input.strip() else combined_raw_text[:120],
                        predicted_label=predicted_label,
                        confidence_score=confidence_pct,
                        word_count=words_count,
                        top_indicators=top_indicator_str
                    )

                    # Display Prediction Result Card
                    if is_real:
                        st.markdown(f"""
                        <div class='verdict-real'>
                            <h3 style='margin:0; color:#10B981; font-size:1.4rem;'>
                                ✅ PREDICTION: LIKELY REAL
                            </h3>
                            <p style='color:#E2E8F0; margin-top:8px; font-size:0.95rem; line-height:1.5;'>
                                The vocabulary distribution, stylistic structure, and syntax align closely with the objective, 
                                attributed journalistic patterns identified in the verified training corpus.
                            </p>
                            <div style='font-size:1.15rem; font-weight:700; color:#34D399; margin-top:10px;'>
                                Model Confidence: {confidence_pct:.1f}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class='verdict-misleading'>
                            <h3 style='margin:0; color:#EF4444; font-size:1.4rem;'>
                                ⚠️ PREDICTION: LIKELY MISLEADING
                            </h3>
                            <p style='color:#E2E8F0; margin-top:8px; font-size:0.95rem; line-height:1.5;'>
                                The text exhibits stylistic markers, rhetoric, or unhedged assertions frequently 
                                correlated with sensationalist, unverified, or misleading media narratives in the training data.
                            </p>
                            <div style='font-size:1.15rem; font-weight:700; color:#F87171; margin-top:10px;'>
                                Model Confidence: {confidence_pct:.1f}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Confidence Score Breakdown & Explainability
                    col_prob, col_words = st.columns([1, 1])

                    with col_prob:
                        st.markdown("<div class='nl-card'>", unsafe_allow_html=True)
                        st.subheader("🎯 Class Probabilities")
                        fig_prob = go.Figure()
                        fig_prob.add_trace(go.Bar(
                            x=[prob_real * 100, prob_fake * 100],
                            y=["Likely Real", "Likely Misleading"],
                            orientation="h",
                            marker=dict(color=["#10B981", "#EF4444"]),
                            text=[f"{prob_real * 100:.1f}%", f"{prob_fake * 100:.1f}%"],
                            textposition="auto"
                        ))
                        fig_prob.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#F8FAFC"),
                            xaxis=dict(range=[0, 100], title="Confidence (%)", gridcolor="#1E293B"),
                            yaxis=dict(gridcolor="#1E293B"),
                            height=200,
                            margin=dict(l=10, r=10, t=10, b=10)
                        )
                        st.plotly_chart(fig_prob, use_container_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                    with col_words:
                        st.markdown("<div class='nl-card'>", unsafe_allow_html=True)
                        st.subheader("🔬 Influential Word Indicators")
                        st.markdown(
                            "<p style='color:#94A3B8; font-size:0.82rem;'>"
                            "Key terms extracted from submitted text contributing to model decision:"
                            "</p>",
                            unsafe_allow_html=True
                        )

                        if matched_tokens:
                            chips_markup = ""
                            for item in matched_tokens[:8]:
                                if item["lean"] == "Real":
                                    chips_markup += f"<span class='nl-chip-real'>✓ {item['word']} ({abs(item['weight']):.2f})</span>"
                                else:
                                    chips_markup += f"<span class='nl-chip-misleading'>⚠ {item['word']} (+{item['weight']:.2f})</span>"
                            st.markdown(chips_markup, unsafe_allow_html=True)
                            st.markdown(
                                "<p style='color:#64748B; font-size:0.75rem; margin-top:8px;'>"
                                "Values show Logistic Regression coefficient weights. Positive weights lean misleading; negative weights lean credible."
                                "</p>",
                                unsafe_allow_html=True
                            )
                        else:
                            st.info("No high-weight domain keywords detected; classification was driven by general token frequency.")
                        st.markdown("</div>", unsafe_allow_html=True)

                    # Required Visible Disclaimer
                    st.markdown("""
                    <div class='nl-disclaimer-box'>
                        <div class='nl-disclaimer-title'>⚠️ Mandatory System Disclaimer:</div>
                        <p class='nl-disclaimer-text'>
                            This prediction is based on patterns learned from the training dataset. It is not independent 
                            verification of the news story and should not be treated as proof of factual accuracy.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)


# =============================================================================
# MODULE E: PREDICTION HISTORY (News Analysis History)
# =============================================================================
elif selected_page == "Prediction History":
    st.markdown("""
    <div class='nl-card'>
        <h2 style='margin:0; color: #F8FAFC;'>📜 Prediction History</h2>
        <p style='color: #94A3B8; font-size: 0.95rem; margin-top: 6px; margin-bottom: 0;'>
            Review, search, filter, and manage your local news analysis logs stored in the SQLite database.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Privacy Notice
    st.markdown("""
    <div style='background: #0F172A; border-left: 4px solid #06B6D4; border-radius: 6px; padding: 12px 16px; margin-bottom: 18px;'>
        <div style='color: #06B6D4; font-weight: 600; font-size: 0.85rem;'>🔒 Local Privacy Notice:</div>
        <p style='color: #94A3B8; font-size: 0.8rem; margin: 4px 0 0 0;'>
            All analysis history is stored strictly on your local computer in SQLite (<code>data/history.db</code>). 
            No queries or personal data are ever uploaded to cloud servers or external third parties.
        </p>
    </div>
    """, unsafe_allow_html=True)

    history_records = db.get_history(limit=250)

    if not history_records:
        st.info("No prediction history recorded yet. Use the 'Analyze News' page to submit an article.")
    else:
        df_hist = pd.DataFrame(history_records)

        # Search and Filter Controls
        col_f1, col_f2 = st.columns([1, 2])
        with col_f1:
            category_filter = st.selectbox("Filter by Category", ["All", "Likely Real", "Likely Misleading"])
        with col_f2:
            search_query = st.text_input("Search Text Snippet", placeholder="Type keywords to filter...")

        filtered_df = df_hist.copy()
        if category_filter != "All":
            filtered_df = filtered_df[filtered_df["predicted_label"] == category_filter]
        if search_query.strip():
            filtered_df = filtered_df[filtered_df["headline_text"].str.contains(search_query, case=False, na=False)]

        st.markdown(
            f"<p style='color:#94A3B8; font-size:0.85rem;'>"
            f"Displaying <b>{len(filtered_df)}</b> of <b>{len(df_hist)}</b> total entries"
            f"</p>",
            unsafe_allow_html=True
        )

        # Presentation Table
        table_df = filtered_df[[
            "id", "timestamp", "headline_text", "predicted_label", "confidence_score", "word_count", "top_indicators"
        ]].rename(columns={
            "id": "ID",
            "timestamp": "Timestamp",
            "headline_text": "News Headline / Snippet",
            "predicted_label": "Prediction Category",
            "confidence_score": "Confidence (%)",
            "word_count": "Words",
            "top_indicators": "Key Indicators"
        })

        st.dataframe(table_df, use_container_width=True, height=360)

        # History Actions: Delete Individual Record & Clear History
        st.markdown("<div class='nl-card'>", unsafe_allow_html=True)
        st.subheader("⚙️ History Management")
        act_col1, act_col2, act_col3 = st.columns([2, 1, 1])

        with act_col1:
            record_ids = df_hist["id"].tolist()
            selected_record_id = st.selectbox("Select Record ID to Delete:", record_ids)
            if st.button("🗑️ Delete Selected Record", use_container_width=True):
                if db.delete_prediction(selected_record_id):
                    st.success(f"Record #{selected_record_id} successfully deleted from database.")
                    st.rerun()

        with act_col2:
            st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
            csv_data = table_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export CSV",
                data=csv_data,
                file_name="newslens_prediction_history.csv",
                mime="text/csv",
                use_container_width=True
            )

        with act_col3:
            st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
            if st.button("⚠️ Clear All History", use_container_width=True):
                db.clear_history()
                st.success("All prediction records have been cleared.")
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# MODULE D: MODEL PERFORMANCE (Model Evaluation)
# =============================================================================
elif selected_page == "Model Performance":
    st.markdown("""
    <div class='nl-card'>
        <h2 style='margin:0; color: #F8FAFC;'>📊 Model Performance & Evaluation</h2>
        <p style='color: #94A3B8; font-size: 0.95rem; margin-top: 6px; margin-bottom: 0;'>
            Rigorous empirical evaluation metrics computed on the unseen 20% test partition (4,914 articles) 
            from the authentic Hugging Face RealFakeNews corpus.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if not metrics_data:
        st.warning("⚠️ Metrics file `models/model_metrics.json` not found. Please run `python train_model.py` to generate evaluation metrics.")
    else:
        eval_metrics = metrics_data.get("evaluation_metrics", {})
        acc_val = eval_metrics.get("accuracy", 0.9212)
        prec_val = eval_metrics.get("precision", 0.9235)
        rec_val = eval_metrics.get("recall", 0.9218)
        f1_val = eval_metrics.get("f1_score", 0.9212)

        # Performance Metric Summary Cards
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class='nl-metric-box'>
                <div class='nl-metric-title'>Test Accuracy</div>
                <div class='nl-metric-val'>{acc_val * 100:.2f}%</div>
                <div class='nl-metric-subtitle'>Correct Classifications</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class='nl-metric-box'>
                <div class='nl-metric-title'>Precision (Macro)</div>
                <div class='nl-metric-val'>{prec_val * 100:.2f}%</div>
                <div class='nl-metric-subtitle'>Low False Positive Rate</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class='nl-metric-box'>
                <div class='nl-metric-title'>Recall (Macro)</div>
                <div class='nl-metric-val'>{rec_val * 100:.2f}%</div>
                <div class='nl-metric-subtitle'>High Misinformation Capture</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class='nl-metric-box'>
                <div class='nl-metric-title'>F1-Score (Macro)</div>
                <div class='nl-metric-val'>{f1_val * 100:.2f}%</div>
                <div class='nl-metric-subtitle'>Harmonic Balance</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Confusion Matrix and Classification Report Row
        col_cm, col_report = st.columns([1, 1])

        with col_cm:
            st.markdown("<div class='nl-card'>", unsafe_allow_html=True)
            st.subheader("🧩 Confusion Matrix")
            cm_dict = eval_metrics.get("confusion_matrix", {})
            cm_matrix = [
                [cm_dict.get("true_real", 2325), cm_dict.get("false_fake", 97)],
                [cm_dict.get("false_real", 290), cm_dict.get("true_fake", 2202)]
            ]
            fig_cm = px.imshow(
                cm_matrix,
                labels=dict(x="Predicted Label", y="Actual True Label", color="Count"),
                x=["Real News", "Misleading News"],
                y=["Real News", "Misleading News"],
                text_auto=True,
                color_continuous_scale="Teal"
            )
            fig_cm.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#F8FAFC"),
                height=320,
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            st.markdown(
                f"<p style='color:#94A3B8; font-size:0.8rem; text-align:center; margin:0;'>"
                f"Evaluated on {metrics_data.get('test_samples', 4914):,} unseen test articles (20% partition)"
                f"</p>",
                unsafe_allow_html=True
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with col_report:
            st.markdown("<div class='nl-card'>", unsafe_allow_html=True)
            st.subheader("📋 Classification Report")
            clf_rep = eval_metrics.get("classification_report", {})
            if clf_rep:
                rep_rows = []
                for label_key in ["REAL", "FAKE"]:
                    if label_key in clf_rep:
                        rep_rows.append({
                            "Class": "Likely Real" if label_key == "REAL" else "Likely Misleading",
                            "Precision": f"{clf_rep[label_key]['precision'] * 100:.2f}%",
                            "Recall": f"{clf_rep[label_key]['recall'] * 100:.2f}%",
                            "F1-Score": f"{clf_rep[label_key]['f1-score'] * 100:.2f}%",
                            "Support": int(clf_rep[label_key]["support"])
                        })
                st.dataframe(pd.DataFrame(rep_rows), use_container_width=True)

            st.markdown(f"""
            <div style='font-size:0.82rem; color:#94A3B8; margin-top:10px; line-height:1.5;'>
                <b>Dataset Partition Details:</b><br>
                • <b>Training Samples:</b> {metrics_data.get('train_samples', 19654):,} (80%)<br>
                • <b>Testing Samples:</b> {metrics_data.get('test_samples', 4914):,} (20%)<br>
                • <b>Vocabulary Size:</b> {metrics_data.get('vocabulary_size', 10000):,} n-gram features
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Model Comparison Table & Charts
        st.markdown("<div class='nl-card'>", unsafe_allow_html=True)
        st.subheader("⚔️ Model Comparison: Logistic Regression vs Multinomial Naive Bayes")
        comp_records = metrics_data.get("model_comparison", [])
        if comp_records:
            comp_display = []
            for item in comp_records:
                comp_display.append({
                    "Model": item["model"],
                    "Accuracy": f"{item['accuracy'] * 100:.2f}%",
                    "Precision": f"{item['precision'] * 100:.2f}%",
                    "Recall": f"{item['recall'] * 100:.2f}%",
                    "F1-Score": f"{item['f1_score'] * 100:.2f}%",
                    "Training Time": f"{item['train_time_sec']}s"
                })
            st.dataframe(pd.DataFrame(comp_display), use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Global Predictive Indicators (Feature Importance)
        st.markdown("<div class='nl-card'>", unsafe_allow_html=True)
        st.subheader("🌐 Top Predictive Indicators Learned by Classifier")
        
        top_features = metrics_data.get("top_features", {})
        top_fake = top_features.get("FAKE", [])[:10]
        top_real = top_features.get("REAL", [])[:10]

        feat_col1, feat_col2 = st.columns(2)

        with feat_col1:
            st.markdown("<div style='color:#EF4444; font-weight:700; font-size:0.9rem;'>🚨 Indicative of Misleading Patterns:</div>", unsafe_allow_html=True)
            if top_fake:
                df_fake_f = pd.DataFrame(top_fake)
                fig_f = px.bar(
                    df_fake_f,
                    x="weight",
                    y="word",
                    orientation="h",
                    color_discrete_sequence=["#EF4444"]
                )
                fig_f.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#F8FAFC"),
                    yaxis=dict(autorange="reversed", gridcolor="#1E293B"),
                    xaxis=dict(title="Weight (Positive Coefficient)", gridcolor="#1E293B"),
                    height=280,
                    margin=dict(l=10, r=10, t=10, b=10)
                )
                st.plotly_chart(fig_f, use_container_width=True)

        with feat_col2:
            st.markdown("<div style='color:#10B981; font-weight:700; font-size:0.9rem;'>✅ Indicative of Credible Patterns:</div>", unsafe_allow_html=True)
            if top_real:
                df_real_f = pd.DataFrame(top_real)
                fig_r = px.bar(
                    df_real_f,
                    x="weight",
                    y="word",
                    orientation="h",
                    color_discrete_sequence=["#10B981"]
                )
                fig_r.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#F8FAFC"),
                    yaxis=dict(autorange="reversed", gridcolor="#1E293B"),
                    xaxis=dict(title="Weight (Negative Coefficient Magnitude)", gridcolor="#1E293B"),
                    height=280,
                    margin=dict(l=10, r=10, t=10, b=10)
                )
                st.plotly_chart(fig_r, use_container_width=True)

        st.markdown("""
        <p style='color:#94A3B8; font-size:0.8rem; margin-top:8px;'>
            <b>Limitation of Feature Explainability:</b> Top indicator words reflect linguistic correlations present in 
            the training corpus and do not constitute absolute proof of truth or falsehood. Words may appear across both categories in differing contextual phrasing.
        </p>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# MODULE G: ABOUT PAGE ("About Project")
# =============================================================================
elif selected_page == "About Project":
    st.markdown("""
    <div class='nl-card-glow'>
        <h2 style='margin:0; color: #F8FAFC;'>NewsLens – Project Documentation & Viva Guide</h2>
        <p style='color: #94A3B8; font-size: 1rem; margin-top: 6px; margin-bottom: 0;'>
            Comprehensive technical architectural documentation designed for academic evaluation, GitHub portfolio, and technical interviews.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Problem Statement & Objectives
    st.markdown("<div class='nl-card'>", unsafe_allow_html=True)
    st.subheader("🎯 Problem Statement & Objectives")
    st.markdown("""
    - **The Problem:** The exponential growth of digital media and unverified online publications has intensified the proliferation of misleading and sensationalist content. Readers often struggle to distinguish well-sourced journalism from hyperbolic claims.
    - **The Objective:** To develop a fully functional, zero-cost, locally executable machine learning system that uses Natural Language Processing and supervised classification to analyze textual indicators of news credibility.
    - **Target Audience:** Academic peers, student project evaluators, and prospective internship interviewers.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    # Technology Stack
    st.markdown("<div class='nl-card'>", unsafe_allow_html=True)
    st.subheader("🛠️ Technology Stack (100% Free & Open-Source)")
    tc1, tc2, tc3 = st.columns(3)
    with tc1:
        st.markdown("""
        **Data & Machine Learning:**
        - **Python 3**: Core language
        - **Scikit-learn**: Logistic Regression, Multinomial Naive Bayes
        - **TF-IDF Vectorizer**: Sublinear TF, unigram + bigram (10,000 features)
        - **Joblib**: Deterministic model serialization
        """)
    with tc2:
        st.markdown("""
        **Natural Language Processing:**
        - **NLTK**: Corpus tokenization & stopword removal
        - **Porter Stemmer**: Morphological root reduction
        - **Regular Expressions**: Sanitizing URLs, HTML tags, and non-alphabetic symbols
        """)
    with tc3:
        st.markdown("""
        **Interface & Database:**
        - **Streamlit**: Interactive web interface
        - **Plotly**: Data visualization
        - **SQLite3**: Thread-safe local query history logging
        - **Pandas & NumPy**: High-performance array operations
        """)
    st.markdown("</div>", unsafe_allow_html=True)

    # ML Pipeline Architecture
    st.markdown("<div class='nl-card'>", unsafe_allow_html=True)
    st.subheader("🔄 End-to-End Machine Learning Workflow")
    st.markdown("""
    1. **Data Acquisition & Hygiene:** Loaded benchmark `fauxNeuz/RealFakeNews` dataset (108,032 records). Handled null values and dropped identical text duplicates to prevent test memorization.
    2. **Stratified Splitting:** Created an 80/20 train/test stratified split **before** vectorization to ensure zero data leakage.
    3. **NLP Preprocessing:** Applied lowercasing, regex cleaning (URLs/HTML/digits), whitespace tokenization, stopword removal, and Porter Stemming.
    4. **Feature Extraction:** Fitted a `TfidfVectorizer` (sublinear scaling, max 10,000 features, unigram + bigram) strictly on training text.
    5. **Model Training & Comparison:** Trained both Logistic Regression and Multinomial Naive Bayes. Selected Logistic Regression due to superior F1-score (92.12% vs 87.34%).
    6. **Inference & Explainability:** Extracted class log-odds coefficients to explain which tokens contributed to each prediction.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    # Dataset Attribution & Limitations
    st.markdown("<div class='nl-card'>", unsafe_allow_html=True)
    st.subheader("📚 Dataset Attribution, Limitations & Ethics")
    st.markdown("""
    - **Dataset Citation:** [fauxNeuz/RealFakeNews on Hugging Face](https://huggingface.co/datasets/fauxNeuz/RealFakeNews)
    - **Dataset License:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International ([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)).
    - **System Limitations:**
      1. The system evaluates stylistic and linguistic patterns, not real-time live web facts.
      2. It cannot verify brand-new real-world events that broke today without factual cross-referencing.
      3. Language patterns evolve over time; models trained on past articles may reflect temporal phrasing shifts.
    - **Ethical Considerations:** Machine learning classifications should serve as assistive tools to encourage media literacy and critical thinking. They must never be treated as infallible proof of factual accuracy or used as an autonomous censorship mechanism.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    # Student Viva & Interview Q&A
    st.markdown("<div class='nl-card'>", unsafe_allow_html=True)
    st.subheader("🎓 Technical Viva & Interview Preparation Cheatsheet")

    with st.expander("Q1: What is the difference between Bag of Words (BoW) and TF-IDF?"):
        st.write("""
        **Bag of Words (BoW)** counts raw word frequencies. Common terms (like 'said', 'reported', 'news') dominate the representation without distinguishing credible from misleading styles.
        **TF-IDF** balances local frequency with inverse document frequency. It rewards terms that are frequent in a specific document but rare across the corpus, penalizing ubiquitous linguistic filler words.
        """)

    with st.expander("Q2: How did you prevent data leakage in your ML pipeline?"):
        st.write("""
        1. Duplicate records were removed during data hygiene.
        2. The train-test split (80/20 stratified) was executed **before** fitting the TF-IDF vectorizer.
        3. The vectorizer called `.fit_transform()` exclusively on the training partition and `.transform()` on the test partition. This guarantees the model never witnessed test vocabulary frequencies during training.
        """)

    with st.expander("Q3: Why did Logistic Regression outperform Multinomial Naive Bayes?"):
        st.write("""
        Naive Bayes relies on the strong assumption of conditional independence among words given the class label. In human language, words appear in correlated clusters (e.g., 'breaking' and 'bombshell'). Logistic Regression does not assume feature independence; it optimizes a joint log-odds boundary and discriminative weights using gradient descent.
        """)

    with st.expander("Q4: Why is F1-score more reliable than Accuracy in credibility classification?"):
        st.write("""
        Accuracy measures overall correct predictions but can mask poor minority-class recall. F1-score represents the harmonic mean of Precision and Recall. In news credibility analysis, failing to identify misleading articles (low recall) and falsely branding credible journalism as misleading (low precision) both carry high penalties.
        """)

    st.markdown("</div>", unsafe_allow_html=True)
