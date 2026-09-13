"""
Steam Game Reception & Popularity Predictor
============================================
A Streamlit product built on the cleaned Steam Store dataset (games through
May 2019), offering two experiences from the same trained models:

  - Developer Studio: forecast reception/popularity for a game you're
    planning, explore price sensitivity, see what actually drives success.
  - Player Corner: find under-the-radar "hidden gem" games, spot over/under
    rated titles, and browse the catalog with model-backed quality signals.

Run:
    streamlit run app.py
"""

import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Page config & style
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Steam Reception Predictor",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    /* ================= Fonts ================= */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;700&display=swap');

    /* ================= Palette ================= */
    :root {
        --bg-0:      #070a11;
        --bg-1:      #0d121c;
        --bg-2:      #141b2a;
        --bg-3:      #1b2334;
        --line:      #2a3348;
        --line-soft: #212a3c;
        --text:      #eef1f8;
        --muted:     #97a2b8;
        --muted-2:   #6b7688;
        --violet:    #8b5cf6;
        --indigo:    #6366f1;
        --cyan:      #22d3ee;
        --green:     #4ade80;
        --amber:     #fbbf24;
        --red:       #f87171;
        --pink:      #f472b6;

        --font-display: 'Space Grotesk', 'Inter', system-ui, sans-serif;
        --font-body:    'Inter', system-ui, -apple-system, sans-serif;
        --font-mono:    'JetBrains Mono', ui-monospace, 'SF Mono', Menlo, monospace;
    }

    /* ================= Base ================= */
    html, body, [class*="css"], .stApp {
        font-family: var(--font-body);
        font-feature-settings: "cv02","cv03","cv04","cv11";
        -webkit-font-smoothing: antialiased;
        text-rendering: optimizeLegibility;
    }

    .stApp {
        background:
            radial-gradient(1200px 700px at 12% -10%, rgba(139, 92, 246, 0.22), transparent 60%),
            radial-gradient(1000px 600px at 88% -5%,  rgba(34, 211, 238, 0.15), transparent 58%),
            radial-gradient(1000px 750px at 50% 115%,rgba(244, 114, 182, 0.10), transparent 62%),
            linear-gradient(180deg, var(--bg-0) 0%, var(--bg-1) 45%, #090d15 100%);
        background-attachment: fixed;
        color: var(--text);
    }
    .main > div { padding-top: 1.75rem; }

    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"] { right: 1rem; }

    /* ================= Typography ================= */
    h1, h2, h3, h4, h5, h6 {
        font-family: var(--font-display);
        color: var(--text);
        letter-spacing: -0.022em;
        line-height: 1.15;
    }
    h1 { font-weight: 700; font-size: clamp(1.9rem, 3vw, 2.6rem); }
    h2 {
        font-weight: 600;
        font-size: clamp(1.3rem, 2vw, 1.65rem);
        border-bottom: 1px solid var(--line-soft);
        padding-bottom: .4rem;
        margin-top: 1.2rem;
    }
    h3 {
        font-weight: 600;
        font-size: 1.2rem;
        letter-spacing: -0.015em;
        color: #dfe5f2;
    }
    h3::before {
        content: "";
        display: inline-block;
        width: 6px; height: 6px;
        border-radius: 50%;
        background: linear-gradient(135deg, var(--violet), var(--cyan));
        margin-right: 10px;
        vertical-align: middle;
        box-shadow: 0 0 10px rgba(139, 92, 246, 0.9);
    }
    h4 { font-weight: 600; font-size: 1.02rem; color: #dfe5f2; }

    p, li, .stMarkdown p, .stMarkdown li {
        color: #c8d0e0;
        line-height: 1.65;
        font-size: 0.955rem;
    }
    strong, b { color: var(--text); font-weight: 600; }
    a { color: var(--cyan); text-decoration: none; transition: color .15s ease; }
    a:hover { color: #67e8f9; text-decoration: underline; }

    code {
        font-family: var(--font-mono);
        font-size: 0.82em;
        background: rgba(139, 92, 246, 0.14);
        color: #c4b5fd;
        padding: 2px 7px;
        border-radius: 6px;
        border: 1px solid rgba(139, 92, 246, 0.28);
    }
    ::selection { background: rgba(34, 211, 238, 0.35); color: #fff; }

    /* ================= Sidebar ================= */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #101624 0%, #0b0f19 55%, #080b12 100%);
        border-right: 1px solid var(--line-soft);
    }
    [data-testid="stSidebar"] h2 {
        border: none;
        background: linear-gradient(90deg, #a78bfa, #22d3ee, #4ade80);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        font-family: var(--font-display);
        letter-spacing: -0.02em;
    }
    [data-testid="stSidebar"] .stRadio > label {
        color: var(--muted);
        font-family: var(--font-body);
        font-weight: 500;
        letter-spacing: .02em;
    }
    [data-testid="stSidebar"] hr { border-color: var(--line-soft); }
    [data-testid="stSidebar"] .stCaption { color: var(--muted-2); font-size: .82rem; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        font-size: .875rem;
        color: #b7c0d2;
    }

    /* ================= Metrics ================= */
    .stMetric, [data-testid="stMetric"] {
        background:
            linear-gradient(160deg, rgba(139, 92, 246, 0.12), rgba(34, 211, 238, 0.05) 55%, rgba(0,0,0,0)),
            var(--bg-2);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 8px 26px -18px rgba(139, 92, 246, 0.9),
                    inset 0 1px 0 rgba(255,255,255,0.03);
        transition: border-color .2s ease, box-shadow .2s ease, transform .2s ease;
    }
    .stMetric:hover, [data-testid="stMetric"]:hover {
        border-color: rgba(34, 211, 238, 0.55);
        box-shadow: 0 12px 32px -16px rgba(34, 211, 238, 0.8),
                    inset 0 1px 0 rgba(255,255,255,0.05);
        transform: translateY(-2px);
    }
    div[data-testid="stMetricValue"] {
        font-family: var(--font-mono);
        font-size: 1.55rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: var(--text);
    }
    div[data-testid="stMetricLabel"] p {
        font-family: var(--font-body);
        color: var(--muted) !important;
        text-transform: uppercase;
        letter-spacing: .1em;
        font-size: .7rem !important;
        font-weight: 600 !important;
    }

    /* ================= Tier badges ================= */
    .tier-badge {
        display: inline-block;
        padding: 7px 20px;
        border-radius: 999px;
        font-family: var(--font-display);
        font-weight: 700;
        font-size: 1.02rem;
        letter-spacing: .04em;
        text-transform: uppercase;
    }
    .tier-great {
        background: linear-gradient(135deg, #064e3b, #14532d 60%, #166534);
        color: #6ee7b7;
        border: 1px solid #22c55e;
        box-shadow: 0 0 22px -4px rgba(34, 197, 94, 0.9);
    }
    .tier-average {
        background: linear-gradient(135deg, #422006, #4a3208 60%, #533f0a);
        color: #fcd34d;
        border: 1px solid #f59e0b;
        box-shadow: 0 0 22px -4px rgba(245, 158, 11, 0.85);
    }
    .tier-poor {
        background: linear-gradient(135deg, #450a0a, #4c1010 60%, #551414);
        color: #fca5a5;
        border: 1px solid #ef4444;
        box-shadow: 0 0 22px -4px rgba(239, 68, 68, 0.85);
    }

    /* ================= Hero ================= */
    .hero {
        position: relative;
        padding: 1.75rem 2rem 1.6rem;
        border-radius: 18px;
        background:
            radial-gradient(700px 220px at 0% 0%, rgba(139, 92, 246, 0.30), transparent 70%),
            radial-gradient(700px 240px at 100% 100%, rgba(34, 211, 238, 0.20), transparent 70%),
            linear-gradient(135deg, #141c2e 0%, #0f1622 100%);
        border: 1px solid rgba(139, 92, 246, 0.35);
        box-shadow:
            0 22px 60px -28px rgba(139, 92, 246, 0.95),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
        margin-bottom: 1.4rem;
        overflow: hidden;
    }
    .hero::after {
        content: "";
        position: absolute;
        inset: 0;
        background: linear-gradient(120deg, transparent 40%, rgba(255,255,255,0.04) 50%, transparent 60%);
        pointer-events: none;
    }
    .hero h1 {
        font-family: var(--font-display);
        font-weight: 700;
        letter-spacing: -0.03em;
        margin: 0 0 .5rem 0;
        background: linear-gradient(90deg, #c4b5fd 0%, #22d3ee 48%, #4ade80 100%);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }
    .hero p {
        margin: 0;
        color: #b6bfd2;
        font-size: 1.06rem;
        line-height: 1.55;
        max-width: 62ch;
    }

    /* ================= Caveat box ================= */
    .caveat-box {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.13), rgba(245, 158, 11, 0.03));
        border: 1px solid rgba(245, 158, 11, 0.30);
        border-left: 4px solid var(--amber);
        border-radius: 10px;
        padding: 12px 16px;
        font-size: 0.86rem;
        line-height: 1.5;
        color: #e6d9b6;
    }

    /* ================= Game cards ================= */
    .game-card {
        background:
            linear-gradient(135deg, rgba(139, 92, 246, 0.10), rgba(34, 211, 238, 0.05) 55%, rgba(0,0,0,0)),
            var(--bg-2);
        border: 1px solid var(--line);
        border-left: 3px solid var(--violet);
        border-radius: 12px;
        padding: 15px 20px;
        margin-bottom: 11px;
        box-shadow: 0 10px 28px -22px rgba(0, 0, 0, 0.95);
        transition: transform .18s ease, border-left-color .18s ease, box-shadow .18s ease;
    }
    .game-card:hover {
        transform: translateX(5px);
        border-left-color: var(--cyan);
        box-shadow: 0 14px 34px -18px rgba(34, 211, 238, 0.6);
    }
    .game-card b {
        font-family: var(--font-display);
        letter-spacing: -0.01em;
        color: var(--text);
    }

    /* ================= Tabs ================= */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid var(--line-soft);
        padding: 6px;
        border-radius: 14px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 9px;
        padding: 9px 20px;
        color: var(--muted);
        font-family: var(--font-body);
        font-weight: 600;
        font-size: 0.92rem;
        letter-spacing: .01em;
        transition: color .16s ease, background .16s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text);
        background: rgba(139, 92, 246, 0.12);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #7c3aed, #0891b2) !important;
        color: #ffffff !important;
        box-shadow: 0 8px 22px -12px rgba(124, 58, 237, 1);
    }
    .stTabs [data-baseweb="tab-highlight"] { background: transparent; }

    /* ================= Buttons ================= */
    .stButton > button {
        font-family: var(--font-body);
        font-weight: 600;
        font-size: 0.94rem;
        letter-spacing: .015em;
        border-radius: 11px;
        border: 1px solid var(--line);
        background: var(--bg-3);
        color: var(--text);
        padding: 0.6rem 1rem;
        transition: transform .16s ease, box-shadow .16s ease, border-color .16s ease, filter .16s ease;
    }
    .stButton > button:hover {
        border-color: rgba(139, 92, 246, 0.75);
        box-shadow: 0 10px 26px -16px rgba(139, 92, 246, 1);
        transform: translateY(-1px);
    }
    .stButton > button:active { transform: translateY(0); }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #7c3aed 0%, #4f46e5 45%, #0891b2 100%);
        border: none;
        color: #ffffff;
        box-shadow: 0 12px 30px -16px rgba(99, 102, 241, 1);
    }
    .stButton > button[kind="primary"]:hover {
        filter: brightness(1.1);
        box-shadow: 0 16px 38px -16px rgba(34, 211, 238, 1);
    }

    /* ================= Inputs ================= */
    [data-baseweb="select"] > div,
    .stNumberInput input,
    .stTextInput input {
        background-color: var(--bg-2) !important;
        border-color: var(--line) !important;
        color: var(--text) !important;
        font-family: var(--font-body);
        border-radius: 9px !important;
    }
    [data-baseweb="select"] > div:hover,
    .stNumberInput input:focus,
    .stTextInput input:focus {
        border-color: rgba(34, 211, 238, 0.7) !important;
        box-shadow: 0 0 0 3px rgba(34, 211, 238, 0.12) !important;
    }
    [data-baseweb="tag"] {
        background: linear-gradient(135deg, #6d28d9, #0e7490) !important;
        border: none !important;
        color: #fff !important;
        font-family: var(--font-body);
        font-weight: 500;
        letter-spacing: .01em;
    }
    [data-testid="stSlider"] [role="slider"] {
        box-shadow: 0 0 0 4px rgba(34, 211, 238, 0.18);
    }
    .stSelectbox label, .stMultiSelect label, .stSlider label,
    .stNumberInput label, .stRadio label, .stCheckbox label {
        font-family: var(--font-body);
        font-weight: 500;
        color: #b7c0d2;
        letter-spacing: .01em;
        font-size: 0.88rem;
    }

    /* ================= Expander ================= */
    [data-testid="stExpander"] {
        border: 1px solid var(--line-soft);
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.02);
        overflow: hidden;
    }
    [data-testid="stExpander"] summary {
        font-family: var(--font-body);
        font-weight: 600;
        letter-spacing: .01em;
    }
    [data-testid="stExpander"] summary:hover { color: var(--cyan); }

    /* ================= Alerts ================= */
    [data-testid="stAlert"] {
        border-radius: 12px;
        border-left-width: 4px;
        font-family: var(--font-body);
    }

    /* ================= Dataframe ================= */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 12px;
        overflow: hidden;
    }

    /* ================= Divider ================= */
    hr { border-color: var(--line-soft); }

    /* ================= Scrollbar ================= */
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: var(--bg-1); }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #6d28d9, #0e7490);
        border-radius: 999px;
        border: 2px solid var(--bg-1);
    }
    ::-webkit-scrollbar-thumb:hover { background: linear-gradient(180deg, #7c3aed, #22d3ee); }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Load models & data (cached so this only happens once per session)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_models():
    return {
        "clf": joblib.load("models/clf_well_received.joblib"),
        "tier": joblib.load("models/clf_reception_tier.joblib"),
        "ratio": joblib.load("models/reg_positive_ratio.joblib"),
        "owners": joblib.load("models/reg_owners.joblib"),
    }


@st.cache_resource
def load_meta():
    return json.load(open("models/meta.json"))


@st.cache_data
def load_catalog():
    return pd.read_csv("models/steam_clean.csv")


models = load_models()
meta = load_meta()
catalog = load_catalog()
FEATURE_COLS = meta["feature_cols"]


# ---------------------------------------------------------------------------
# Core prediction logic (mirrors train_models.py feature engineering exactly)
# ---------------------------------------------------------------------------
def build_feature_row(price, achievements, platforms, genres, categories, tags, required_age=0):
    row = pd.DataFrame(0, index=[0], columns=FEATURE_COLS)

    row["price"] = price
    row["price_log"] = np.log1p(price)
    row["is_free"] = int(price == 0)
    row["achievements"] = achievements
    row["achievements_log"] = np.log1p(achievements)
    row["achievements_per_dollar"] = achievements / (price + 1)
    row["required_age"] = required_age
    row["num_genres"] = len(genres)

    row["dev_avg_rating"] = meta["global_mean_ratio"]
    row["dev_game_count"] = 0
    row["pub_avg_rating"] = meta["global_mean_ratio"]
    row["pub_game_count"] = 0

    d = meta["defaults"]
    row["release_month"] = d["release_month"]
    row["release_year_filled"] = d["release_year_filled"]
    row["name_len"] = d["name_len"]
    row["name_word_count"] = d["name_word_count"]

    for p in platforms:
        col = f"platform_{p.lower()}"
        if col in FEATURE_COLS:
            row[col] = 1
    row["num_platforms"] = len(platforms)

    for g in genres:
        col = "genre_" + g.replace(" ", "_").replace("-", "_")
        if col in FEATURE_COLS:
            row[col] = 1

    for c in categories:
        col = "cat_" + c
        if col in FEATURE_COLS:
            row[col] = 1

    for t in tags:
        col = "tag_" + t.replace(" ", "_").replace("-", "_")
        if col in FEATURE_COLS:
            row[col] = 1

    return row[FEATURE_COLS]


def predict_all(row):
    tier_idx = models["tier"].predict(row)[0]
    tier = meta["tier_classes"][tier_idx]
    tier_proba = models["tier"].predict_proba(row)[0]
    well_proba = float(models["clf"].predict_proba(row)[0][1])
    ratio = float(np.clip(models["ratio"].predict(row)[0], 0, 1))
    owners = float(np.expm1(models["owners"].predict(row)[0]))
    return {
        "tier": tier,
        "tier_confidence": float(tier_proba.max()),
        "well_received_proba": well_proba,
        "positive_ratio": ratio,
        "owners": owners,
    }


def tier_badge_html(tier):
    cls = {"Great": "tier-great", "Average": "tier-average", "Poor": "tier-poor"}[tier]
    return f'<span class="tier-badge {cls}">{tier}</span>'


def format_owners(n):
    if n >= 1_000_000:
        return f"~{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"~{n / 1_000:.0f}K"
    return f"~{n:.0f}"


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.markdown("## 🎮 Steam Reception Predictor")
st.sidebar.caption("Trained on 26,564 Steam games (through May 2019)")

audience = st.sidebar.radio(
    "I am a...",
    ["🛠️ Developer / Publisher", "🕹️ Player looking for games", "ℹ️ About this product"],
    label_visibility="collapsed",
)

st.sidebar.divider()
with st.sidebar.expander("Model accuracy (honest numbers)"):
    m = meta["metrics"]
    st.write(f"**Well-received (yes/no):** ROC-AUC {m['well_received_classifier']['roc_auc']}")
    st.write(f"**Reception tier (3-class):** Accuracy {m['reception_tier_classifier']['accuracy']}")
    st.write(f"**Positive ratio (regression):** R² {m['positive_ratio_regressor']['r2']}")
    st.write(f"**Owners (regression):** R² {m['owners_regressor']['r2']}")
    st.caption(
        "These are moderate, honest numbers — pre-release metadata alone can't fully "
        "predict reception. Treat every prediction as a directional estimate, not a guarantee."
    )

st.sidebar.markdown(
    '<div class="caveat-box">⚠️ Trained on data through May 2019. Genre trends, pricing '
    "norms, and player expectations have shifted since — treat this as a historical "
    "pattern-matcher, not a live market oracle.</div>",
    unsafe_allow_html=True,
)


# ===========================================================================
# ABOUT PAGE
# ===========================================================================
if audience == "ℹ️ About this product":
    st.markdown(
        '<div class="hero"><h1>🎮 Steam Reception &amp; Popularity Predictor</h1>'
        "<p>One set of trained models, two products: a launch-planning tool for developers, "
        "and a discovery tool for players.</p></div>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🛠️ For Developers")
        st.markdown(
            "- **Reception Predictor** — forecast how a planned game will land, before you build it\n"
            "- **Price Sensitivity Explorer** — see how reception & reach shift across price points\n"
            "- **What Drives Success** — the features that actually move the needle, ranked"
        )
    with col2:
        st.subheader("🕹️ For Players")
        st.markdown(
            "- **Hidden Gems Finder** — well-made games the crowd hasn't found yet\n"
            "- **Over/Underrated Explorer** — where reality and the model disagree, and why that's interesting\n"
            "- **Browse & Compare** — filter the whole catalog with model-backed quality signals"
        )

    st.divider()
    st.subheader("How the models were built")
    st.markdown(
        """
1. **Data**: 26,564 English-language Steam games with at least one review (from the original
   27,075-row catalog), covering 1997–2019.
2. **Targets**: positive-review ratio (0–100%), a well-received yes/no flag (>70% positive),
   a 3-tier reception label (Poor/Average/Great), and estimated owners (SteamSpy's bucketed
   ranges, midpoint-estimated).
3. **Features**: platform, genre, category, and tag flags; price and achievement stats;
   **leave-one-out, Bayesian-shrunk developer and publisher reputation** (a game's own outcome
   is never used to compute its own reputation score — avoids the single biggest way this kind
   of model can cheat); release seasonality; title-shape stats. 53 features total.
4. **Models**: XGBoost for all four tasks, tuned via cross-validated random search.
"""
    )
    st.info(
        "Full methodology, every decision and why it was made, and the honest accuracy "
        "numbers are in the accompanying README.md and analysis notebook."
    )


# ===========================================================================
# DEVELOPER STUDIO
# ===========================================================================
elif audience == "🛠️ Developer / Publisher":
    st.markdown(
        '<div class="hero"><h1>🛠️ Developer Studio</h1>'
        "<p>Forecast reception and reach for a game you're planning, and see which choices "
        "actually move the needle.</p></div>",
        unsafe_allow_html=True,
    )

    dev_tab = st.tabs(["🔮 Reception Predictor", "💲 Price Sensitivity", "📊 What Drives Success"])

    # --- Tab 1: Reception Predictor ---------------------------------------
    with dev_tab[0]:
        st.markdown("### Describe your game")
        c1, c2, c3 = st.columns(3)
        with c1:
            price = st.number_input("Price ($)", min_value=0.0, max_value=200.0, value=14.99, step=1.0)
            achievements = st.number_input("Number of achievements", min_value=0, max_value=500, value=20)
            required_age = st.selectbox("Required age", [0, 13, 16, 18], index=0)
        with c2:
            platforms = st.multiselect(
                "Platforms", ["Windows", "Mac", "Linux"], default=["Windows"]
            )
            genres = st.multiselect(
                "Genres (pick up to 3)", meta["top_genres"], default=["Indie", "Action"], max_selections=5
            )
        with c3:
            selected_cat_labels = st.multiselect(
                "Key features", list(meta["key_categories"].values()),
                default=["Single-player", "Steam Achievements"],
            )
            categories = [k for k, v in meta["key_categories"].items() if v in selected_cat_labels]
            categories = [c.replace("cat_", "") for c in categories]
            tags = st.multiselect("Community tags", meta["top_tags"], default=genres[:2] if genres else [])

        predict_clicked = st.button("🔮 Predict Reception", type="primary", use_container_width=True)

        if predict_clicked:
            if not platforms:
                st.warning("Pick at least one platform.")
            elif not genres:
                st.warning("Pick at least one genre.")
            else:
                row = build_feature_row(
                    price, achievements, platforms, genres, categories, tags or genres, required_age
                )
                result = predict_all(row)

                st.divider()
                r1, r2, r3, r4 = st.columns(4)
                with r1:
                    st.markdown("**Predicted tier**")
                    st.markdown(tier_badge_html(result["tier"]), unsafe_allow_html=True)
                    st.caption(f"{result['tier_confidence']:.0%} model confidence")
                with r2:
                    st.metric("Predicted positive ratio", f"{result['positive_ratio']:.0%}")
                with r3:
                    st.metric("Well-received probability", f"{result['well_received_proba']:.0%}")
                with r4:
                    st.metric("Estimated owners", format_owners(result["owners"]))

                gauge_color = (
                    "#4ade80" if result["positive_ratio"] > 0.7
                    else "#fbbf24" if result["positive_ratio"] > 0.5
                    else "#f87171"
                )
                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=result["positive_ratio"] * 100,
                        number={"suffix": "%", "font": {"color": gauge_color, "family": "JetBrains Mono"}},
                        title={"text": "Predicted Positive Review Ratio",
                               "font": {"color": "#97a2b8", "size": 14, "family": "Inter"}},
                        gauge={
                            "axis": {"range": [0, 100], "tickcolor": "#97a2b8"},
                            "bar": {"color": gauge_color, "thickness": 0.32},
                            "bgcolor": "rgba(0,0,0,0)",
                            "borderwidth": 1,
                            "bordercolor": "#2a3348",
                            "steps": [
                                {"range": [0, 50], "color": "rgba(248, 113, 113, 0.18)"},
                                {"range": [50, 75], "color": "rgba(251, 191, 36, 0.18)"},
                                {"range": [75, 100], "color": "rgba(74, 222, 128, 0.18)"},
                            ],
                            "threshold": {
                                "line": {"color": "#22d3ee", "width": 3},
                                "thickness": 0.8,
                                "value": 70,
                            },
                        },
                    )
                )
                fig.update_layout(height=280, margin=dict(l=30, r=30, t=50, b=10),
                                   paper_bgcolor="rgba(0,0,0,0)", font_color="#eef1f8",
                                   font_family="Inter")
                st.plotly_chart(fig, use_container_width=True)

                st.caption(
                    "Remember: R² for the ratio model is ~0.27 — this is a directional estimate "
                    "built from launch metadata, not a guarantee of how players will actually feel."
                )

    # --- Tab 2: Price Sensitivity ------------------------------------------
    with dev_tab[1]:
        st.markdown("### See how price alone shifts predicted outcomes")
        st.caption("Everything else about your game (from the Reception Predictor tab) stays fixed.")

        if "price" not in st.session_state:
            st.session_state["price"] = 14.99

        pc1, pc2 = st.columns(2)
        with pc1:
            ps_platforms = st.multiselect(
                "Platforms", ["Windows", "Mac", "Linux"], default=["Windows"], key="ps_platforms"
            )
        with pc2:
            ps_genres = st.multiselect(
                "Genres", meta["top_genres"], default=["Indie", "Action"], key="ps_genres", max_selections=5
            )
        ps_achievements = st.slider("Achievements", 0, 200, 20, key="ps_ach")

        if ps_platforms and ps_genres:
            price_points = list(np.arange(0, 61, 2.5))
            rows = []
            for p in price_points:
                row = build_feature_row(p, ps_achievements, ps_platforms, ps_genres, [], ps_genres)
                res = predict_all(row)
                rows.append({"price": p, "positive_ratio": res["positive_ratio"] * 100,
                             "owners": res["owners"]})
            sweep_df = pd.DataFrame(rows)

            fig1 = px.line(sweep_df, x="price", y="positive_ratio", markers=True,
                            title="Predicted Positive Ratio vs. Price",
                            labels={"price": "Price ($)", "positive_ratio": "Predicted positive ratio (%)"})
            fig1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.02)",
                                font_color="#eef1f8", height=380, font_family="Inter",
                                title_font_color="#a78bfa", title_font_family="Space Grotesk",
                                xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="#2a3348"),
                                yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="#2a3348"))
            fig1.update_traces(line=dict(color="#4ade80", width=3),
                               marker=dict(color="#22d3ee", size=8, line=dict(color="#0e131d", width=1)))

            fig2 = px.line(sweep_df, x="price", y="owners", markers=True,
                            title="Predicted Owners vs. Price",
                            labels={"price": "Price ($)", "owners": "Predicted owners"})
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.02)",
                                font_color="#eef1f8", height=380, font_family="Inter",
                                title_font_color="#60a5fa", title_font_family="Space Grotesk",
                                xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="#2a3348"),
                                yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="#2a3348"))
            fig2.update_traces(line=dict(color="#6366f1", width=3),
                               marker=dict(color="#f472b6", size=8, line=dict(color="#0e131d", width=1)))

            cc1, cc2 = st.columns(2)
            cc1.plotly_chart(fig1, use_container_width=True)
            cc2.plotly_chart(fig2, use_container_width=True)

            best_ratio_row = sweep_df.loc[sweep_df["positive_ratio"].idxmax()]
            best_owners_row = sweep_df.loc[sweep_df["owners"].idxmax()]
            st.info(
                f"For this configuration, the model's best predicted **reception** is around "
                f"**${best_ratio_row['price']:.2f}** ({best_ratio_row['positive_ratio']:.0f}% positive), "
                f"and its best predicted **reach** is around **${best_owners_row['price']:.2f}** "
                f"({format_owners(best_owners_row['owners'])} owners). These often don't line up — "
                "cheaper/free usually reaches more people even when it doesn't maximize review scores."
            )
        else:
            st.warning("Pick at least one platform and one genre to see the sweep.")

    # --- Tab 3: What Drives Success -----------------------------------------
    with dev_tab[2]:
        st.markdown("### What actually predicts good reception?")

        importances = pd.DataFrame({
            "feature": FEATURE_COLS,
            "importance": models["clf"].feature_importances_,
        }).sort_values("importance", ascending=False).head(15)
        importances["feature"] = importances["feature"].str.replace("_", " ").str.title()

        fig = px.bar(
            importances.sort_values("importance"), x="importance", y="feature", orientation="h",
            title="Top 15 Features — Well-Received Classifier",
            color="importance", color_continuous_scale=["#22d3ee", "#8b5cf6", "#f472b6"],
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.02)",
                           font_color="#eef1f8", height=480, font_family="Inter",
                           xaxis_title="Importance", yaxis_title="",
                           title_font_color="#a78bfa", title_font_family="Space Grotesk",
                           coloraxis_showscale=False,
                           xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="#2a3348"),
                           yaxis=dict(gridcolor="rgba(255,255,255,0.02)"))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Genre benchmarks")
        st.caption("How each genre has historically performed on Steam (avg. positive ratio & avg. owners).")
        genre_stats = []
        for g in meta["top_genres"]:
            col = "genre_" + g.replace(" ", "_").replace("-", "_")
            if col in catalog.columns:
                subset = catalog[catalog[col] == 1]
                genre_stats.append({
                    "Genre": g, "Avg positive ratio": subset["positive_ratio"].mean(),
                    "Avg owners": subset["owners_avg"].mean(), "Games in catalog": len(subset),
                })
        genre_df = pd.DataFrame(genre_stats).sort_values("Avg positive ratio", ascending=False)

        fig2 = px.scatter(
            genre_df, x="Avg owners", y="Avg positive ratio", size="Games in catalog", text="Genre",
            log_x=True, title="Genre Landscape: Reach vs. Reception",
            color="Avg positive ratio", color_continuous_scale=["#f87171", "#fbbf24", "#4ade80"],
        )
        fig2.update_traces(textposition="top center", textfont_color="#c9d1e4",
                           textfont_family="Inter",
                           marker=dict(line=dict(color="#0e131d", width=1)))
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.02)",
                            font_color="#eef1f8", height=460, font_family="Inter",
                            title_font_color="#22d3ee", title_font_family="Space Grotesk",
                            coloraxis_showscale=False,
                            yaxis_tickformat=".0%",
                            xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="#2a3348"),
                            yaxis_gridcolor="rgba(255,255,255,0.06)")
        st.plotly_chart(fig2, use_container_width=True)


# ===========================================================================
# PLAYER CORNER
# ===========================================================================
else:
    st.markdown(
        '<div class="hero"><h1>🕹️ Player Corner</h1>'
        "<p>Find games the crowd hasn't caught up to yet, and see where hype and quality "
        "don't quite match.</p></div>",
        unsafe_allow_html=True,
    )

    player_tab = st.tabs(["💎 Hidden Gems", "⚖️ Over/Underrated", "🔍 Browse & Compare"])

    # --- Tab 1: Hidden Gems --------------------------------------------------
    with player_tab[0]:
        st.markdown("### Games the model rates highly that haven't found their audience")
        st.caption(
            "Sorted by predicted positive ratio among games with **below-median owners** — "
            "well-made games that Steam's own popularity signals may have buried."
        )

        median_owners = catalog["owners_avg"].median()
        gems = catalog[
            (catalog["owners_avg"] <= median_owners) & (catalog["positive_ratio"] > 0.8)
        ].sort_values("positive_ratio", ascending=False)

        max_price = st.slider("Max price ($)", 0.0, 60.0, 20.0, key="gems_price")
        gems = gems[gems["price"] <= max_price].head(20)

        for _, g in gems.iterrows():
            st.markdown(
                f"""<div class="game-card">
                <b style="font-size:1.05rem;">{g['name']}</b>
                <span style="color:#22d3ee;font-weight:600;">— ${g['price']:.2f}</span><br>
                <span style="color:#97a2b8;">{g['genres']}</span><br>
                <span style="color:#4ade80;font-weight:600;">{g['positive_ratio']:.0%} positive</span>
                &nbsp;·&nbsp; <span style="color:#a78bfa;">{int(g['total_ratings'])} reviews</span>
                &nbsp;·&nbsp; <span style="color:#f472b6;">~{format_owners(g['owners_avg'])} owners</span>
                </div>""",
                unsafe_allow_html=True,
            )

    # --- Tab 2: Over/Underrated -----------------------------------------------
    with player_tab[1]:
        st.markdown("### Where reality and the model disagree")
        st.caption(
            "`reception_gap` = actual positive ratio − model-predicted positive ratio. "
            "Positive = the game did **better** than its price/genre/platform profile alone "
            "would suggest (often word-of-mouth or a great execution). Negative = it "
            "**underperformed** what similar games usually get."
        )

        direction = st.radio(
            "Show me", ["Overperformers (better than expected)", "Underperformers (worse than expected)"],
            horizontal=True,
        )
        min_reviews = st.slider("Minimum number of reviews", 10, 5000, 100, key="gap_reviews")

        pool = catalog[catalog["total_ratings"] >= min_reviews]
        if direction.startswith("Over"):
            top = pool.sort_values("reception_gap", ascending=False).head(20)
            gap_color = "#4ade80"
        else:
            top = pool.sort_values("reception_gap", ascending=True).head(20)
            gap_color = "#f87171"

        for _, g in top.iterrows():
            st.markdown(
                f"""<div class="game-card">
                <b style="font-size:1.05rem;">{g['name']}</b>
                <span style="color:#97a2b8;">— {g['genres']}</span><br>
                Actual: <b style="color:#eef1f8;">{g['positive_ratio']:.0%}</b> positive &nbsp;·&nbsp;
                Model expected: <b style="color:#a78bfa;">{g['pred_positive_ratio']:.0%}</b> &nbsp;·&nbsp;
                <span style="color:{gap_color};font-weight:700;">Gap: {g['reception_gap']:+.0%}</span>
                </div>""",
                unsafe_allow_html=True,
            )

    # --- Tab 3: Browse & Compare -----------------------------------------------
    with player_tab[2]:
        st.markdown("### Filter the catalog")
        f1, f2, f3 = st.columns(3)
        with f1:
            genre_filter = st.multiselect("Genre", meta["top_genres"])
        with f2:
            price_range = st.slider("Price range ($)", 0.0, 60.0, (0.0, 30.0))
        with f3:
            min_tier = st.selectbox("Minimum predicted tier", ["Any", "Average", "Great"])

        filtered = catalog[
            (catalog["price"] >= price_range[0]) & (catalog["price"] <= price_range[1])
        ]
        for g in genre_filter:
            col = "genre_" + g.replace(" ", "_").replace("-", "_")
            if col in filtered.columns:
                filtered = filtered[filtered[col] == 1]
        if min_tier == "Great":
            filtered = filtered[filtered["pred_reception_tier"] == "Great"]
        elif min_tier == "Average":
            filtered = filtered[filtered["pred_reception_tier"].isin(["Average", "Great"])]

        filtered = filtered.sort_values("positive_ratio", ascending=False)
        st.caption(f"{len(filtered):,} games match your filters — showing top 25 by positive ratio")

        display_cols = ["name", "genres", "price", "positive_ratio", "owners", "pred_reception_tier"]
        show = filtered[display_cols].head(25).rename(columns={
            "name": "Name", "genres": "Genres", "price": "Price ($)",
            "positive_ratio": "Positive Ratio", "owners": "Owners (est.)",
            "pred_reception_tier": "Model Tier",
        })
        show["Positive Ratio"] = (show["Positive Ratio"] * 100).round(1).astype(str) + "%"
        st.dataframe(show, use_container_width=True, hide_index=True)