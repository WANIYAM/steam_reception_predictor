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
# Page config & Funky Cyberpunk/Arcade Theme
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Steam Reception Predictor ⚡",
    page_icon="👾",
    layout="wide",
    initial_sidebar_state="expanded",
)

FUNKY_CSS = """
<style>
    /* ================= Funky Neon Imports ================= */
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=Space+Grotesk:wght@500;700&family=JetBrains+Mono:wght@700&display=swap');

    /* ================= Neo-Cyber Palette ================= */
    :root {
        --bg-main:       #080412;
        --bg-card:       rgba(23, 14, 43, 0.75);
        --neon-pink:     #ff2a85;
        --neon-cyan:     #00f0ff;
        --neon-purple:   #9d4edf;
        --neon-yellow:   #ffe600;
        --neon-green:    #00ff66;
        --text-bright:   #ffffff;
        --text-subtle:   #a7a4c7;

        --font-display:  'Syne', sans-serif;
        --font-body:     'Space Grotesk', sans-serif;
        --font-mono:     'JetBrains Mono', monospace;
    }

    /* ================= Main App Container ================= */
    html, body, [class*="css"], .stApp {
        font-family: var(--font-body) !important;
        -webkit-font-smoothing: antialiased;
    }

    .stApp {
        background: 
            radial-gradient(circle at 10% 15%, rgba(255, 42, 133, 0.25), transparent 45%),
            radial-gradient(circle at 90% 10%, rgba(0, 240, 255, 0.2), transparent 40%),
            radial-gradient(circle at 50% 85%, rgba(157, 78, 223, 0.25), transparent 50%),
            linear-gradient(180deg, #05020a 0%, #0d071e 100%) !important;
        background-attachment: fixed !important;
        color: var(--text-subtle);
    }

    /* ================= Funky Typography ================= */
    h1, h2, h3, h4, h5, h6 {
        font-family: var(--font-display) !important;
        color: var(--text-bright);
        letter-spacing: -0.03em;
        text-transform: uppercase;
    }

    h1 { font-weight: 800; font-size: 2.6rem; }
    h2 { font-size: 1.6rem; }
    h3 { font-size: 1.25rem; }

    code {
        font-family: var(--font-mono);
        background: rgba(255, 42, 133, 0.15);
        color: var(--neon-pink);
        padding: 3px 8px;
        border-radius: 6px;
        border: 1px solid rgba(255, 42, 133, 0.4);
    }

    /* ================= Neo-Brutalist Hero Container ================= */
    .hero-box {
        position: relative;
        padding: 2.5rem;
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(32, 16, 61, 0.8) 0%, rgba(14, 7, 29, 0.9) 100%);
        border: 2px solid rgba(0, 240, 255, 0.4);
        box-shadow: 0 0 30px rgba(0, 240, 255, 0.25), inset 0 0 15px rgba(255, 42, 133, 0.15);
        backdrop-filter: blur(16px);
        margin-bottom: 2rem;
    }

    .hero-box h1 {
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, var(--neon-pink) 0%, var(--neon-purple) 50%, var(--neon-cyan) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        text-shadow: 0 0 20px rgba(255, 42, 133, 0.3);
    }

    .hero-box p {
        color: #d1cbe8;
        font-size: 1.15rem;
        font-weight: 500;
        margin: 0;
    }

    /* ================= Sidebar Styling ================= */
    [data-testid="stSidebar"] {
        background: rgba(10, 5, 22, 0.95);
        border-right: 2px solid rgba(157, 78, 223, 0.3);
    }

    [data-testid="stSidebar"] h2 {
        background: linear-gradient(90deg, var(--neon-cyan), var(--neon-pink));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* ================= Glowing Card Containers ================= */
    [data-testid="stVerticalBlock"] > div[data-testid="stBlock"] {
        border-radius: 18px;
    }
    
    .stContainer {
        border-radius: 20px !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        background: var(--bg-card) !important;
        backdrop-filter: blur(10px);
    }

    /* ================= Metric Displays ================= */
    [data-testid="stMetric"] {
        background: rgba(28, 14, 54, 0.7);
        border: 1px solid rgba(0, 240, 255, 0.3);
        border-radius: 18px;
        padding: 1.25rem;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    [data-testid="stMetric"]:hover {
        transform: translateY(-4px) rotate(-0.5deg);
        border-color: var(--neon-pink);
        box-shadow: 0 0 25px rgba(255, 42, 133, 0.3);
    }

    [data-testid="stMetricValue"] {
        font-family: var(--font-mono);
        font-weight: 700;
        color: var(--neon-cyan) !important;
        font-size: 1.8rem !important;
    }

    [data-testid="stMetricLabel"] p {
        color: var(--neon-pink) !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    /* ================= Arcade Tier Badges ================= */
    .tier-badge {
        display: inline-block;
        padding: 8px 22px;
        border-radius: 12px;
        font-family: var(--font-display);
        font-weight: 800;
        font-size: 1.1rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .tier-great {
        background: rgba(0, 255, 102, 0.15);
        color: var(--neon-green);
        border: 2px solid var(--neon-green);
        box-shadow: 0 0 20px rgba(0, 255, 102, 0.4);
    }
    .tier-average {
        background: rgba(255, 230, 0, 0.15);
        color: var(--neon-yellow);
        border: 2px solid var(--neon-yellow);
        box-shadow: 0 0 20px rgba(255, 230, 0, 0.4);
    }
    .tier-poor {
        background: rgba(255, 42, 133, 0.15);
        color: var(--neon-pink);
        border: 2px solid var(--neon-pink);
        box-shadow: 0 0 20px rgba(255, 42, 133, 0.4);
    }

    /* ================= Funky Pill Buttons & Tabs ================= */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: rgba(15, 7, 32, 0.8);
        border: 2px solid rgba(157, 78, 223, 0.4);
        padding: 8px;
        border-radius: 18px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 12px;
        padding: 10px 24px;
        color: var(--text-subtle);
        font-family: var(--font-display);
        font-weight: 700;
        text-transform: uppercase;
        transition: all 0.25s ease;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--neon-pink) 0%, var(--neon-purple) 100%) !important;
        color: #ffffff !important;
        box-shadow: 0 0 20px rgba(255, 42, 133, 0.5);
    }

    /* Buttons */
    .stButton > button {
        border-radius: 14px;
        font-family: var(--font-display);
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 0.75rem 1.5rem;
        border: 2px solid rgba(0, 240, 255, 0.4);
        background: rgba(20, 10, 40, 0.8);
        color: var(--neon-cyan);
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        border-color: var(--neon-pink);
        color: var(--neon-pink);
        transform: scale(1.02);
        box-shadow: 0 0 25px rgba(255, 42, 133, 0.4);
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, var(--neon-pink), var(--neon-purple));
        border: none;
        color: #ffffff;
        box-shadow: 0 0 25px rgba(255, 42, 133, 0.5);
    }

    .stButton > button[kind="primary"]:hover {
        transform: scale(1.03);
        box-shadow: 0 0 35px rgba(0, 240, 255, 0.7);
    }

    /* Tags */
    .custom-tag {
        display: inline-block;
        background: rgba(0, 240, 255, 0.12);
        border: 1px solid rgba(0, 240, 255, 0.4);
        color: var(--neon-cyan);
        padding: 4px 10px;
        border-radius: 8px;
        font-family: var(--font-mono);
        font-size: 0.8rem;
        margin-right: 6px;
    }

    .caveat-box {
        background: rgba(255, 230, 0, 0.08);
        border: 2px solid var(--neon-yellow);
        border-radius: 16px;
        padding: 14px 18px;
        font-size: 0.9rem;
        color: #fffae0;
        box-shadow: 0 0 15px rgba(255, 230, 0, 0.15);
    }
</style>
"""
st.markdown(FUNKY_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Load models & data
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
# Prediction logic
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
# Sidebar Navigation
# ---------------------------------------------------------------------------
st.sidebar.markdown("## 👾 STEAM ENGINE")
st.sidebar.caption("⚡ Model-driven Game Analytics ⚡")

audience = st.sidebar.radio(
    "Choose Mode:",
    ["🛠️ Developer Studio", "🕹️ Player Corner", "ℹ️ Product Info"],
    label_visibility="collapsed",
)

st.sidebar.divider()
with st.sidebar.expander("🎯 Honest Model Metrics"):
    m = meta["metrics"]
    st.write(f"**Well-Received (ROC-AUC):** `{m['well_received_classifier']['roc_auc']}`")
    st.write(f"**Tier Accuracy:** `{m['reception_tier_classifier']['accuracy']}`")
    st.write(f"**Positive Ratio (R²):** `{m['positive_ratio_regressor']['r2']}`")
    st.write(f"**Owners Prediction (R²):** `{m['owners_regressor']['r2']}`")

st.sidebar.markdown(
    '<div class="caveat-box">⚠️ <b>TIME VAULT WARNING</b><br>Trained on data up to May 2019. Historical patterns only!</div>',
    unsafe_allow_html=True,
)


# ===========================================================================
# ABOUT PAGE
# ===========================================================================
if audience == "ℹ️ Product Info":
    st.markdown(
        '<div class="hero-box"><h1>⚡ STEAM GAME PREDICTOR ⚡</h1>'
        "<p>Dual-purpose AI engine designed for indie devs launching games and players hunting top-tier titles.</p></div>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.subheader("🛠️ DEV MODE")
            st.markdown(
                "- 🔮 **Reception Forecaster**: Test pre-launch game parameters\n"
                "- 💲 **Price Simulator**: Fine-tune optimization vs volume\n"
                "- 📊 **Success Drivers**: Feature impact rankings"
            )
    with col2:
        with st.container(border=True):
            st.subheader("🕹️ PLAYER MODE")
            st.markdown(
                "- 💎 **Hidden Gems**: High potential, low visibility titles\n"
                "- ⚖️ **Reality Gap**: Underrated & Overhyped games\n"
                "- 🔍 **Catalog Matrix**: Model-backed catalog filtering"
            )

    st.divider()
    with st.container(border=True):
        st.subheader("🧪 Under The Hood")
        st.markdown(
            """
1. **Catalog Base**: 26,564 English Steam titles (1997–2019).
2. **Targets**: Review ratios, 3-tier reception classifications, & owner buckets.
3. **Leakage Control**: Bayesian-shrunk reputation scores computed without single-game self-influence.
4. **Engine**: XGBoost hyper-tuned regressors & classifiers.
"""
        )


# ===========================================================================
# DEVELOPER STUDIO
# ===========================================================================
elif audience == "🛠️ Developer Studio":
    st.markdown(
        '<div class="hero-box"><h1>🛠️ DEV STUDIO</h1>'
        "<p>Simulate launch parameters, gauge market reception, and tweak pricing dynamics.</p></div>",
        unsafe_allow_html=True,
    )

    dev_tab = st.tabs(["🔮 RECEPTION FORECASTER", "💲 PRICE SIMULATOR", "📊 SUCCESS DRIVERS"])

    # --- Tab 1: Reception Predictor ---
    with dev_tab[0]:
        left_input, right_output = st.columns([1.1, 1])

        with left_input:
            with st.container(border=True):
                st.markdown("### 🎛️ Game Parameters")
                price = st.number_input("Price ($)", min_value=0.0, max_value=200.0, value=14.99, step=1.0)
                achievements = st.number_input("Achievements Count", min_value=0, max_value=500, value=20)
                required_age = st.selectbox("Age Gate", [0, 13, 16, 18], index=0)
                platforms = st.multiselect("Platforms", ["Windows", "Mac", "Linux"], default=["Windows"])
                genres = st.multiselect("Genres (Up to 3)", meta["top_genres"], default=["Indie", "Action"], max_selections=5)
                selected_cat_labels = st.multiselect("Key Features", list(meta["key_categories"].values()), default=["Single-player", "Steam Achievements"])
                categories = [k.replace("cat_", "") for k, v in meta["key_categories"].items() if v in selected_cat_labels]
                tags = st.multiselect("Community Tags", meta["top_tags"], default=genres[:2] if genres else [])

                predict_clicked = st.button("🚀 LAUNCH PREDICTION ENGINE", type="primary", use_container_width=True)

        with right_output:
            if predict_clicked:
                if not platforms or not genres:
                    st.warning("⚠️ Please select at least one platform and genre!")
                else:
                    row = build_feature_row(price, achievements, platforms, genres, categories, tags or genres, required_age)
                    result = predict_all(row)

                    with st.container(border=True):
                        st.markdown("### 📊 Predicted Reception")
                        
                        m1, m2 = st.columns(2)
                        with m1:
                            st.markdown(tier_badge_html(result["tier"]), unsafe_allow_html=True)
                            st.caption(f"{result['tier_confidence']:.0%} Tier Confidence")
                        with m2:
                            st.metric("Positive Ratio", f"{result['positive_ratio']:.0%}")

                        m3, m4 = st.columns(2)
                        with m3:
                            st.metric("Success Proba", f"{result['well_received_proba']:.0%}")
                        with m4:
                            st.metric("Est. Owners", format_owners(result["owners"]))

                    # Funky Gauge Chart
                    gauge_color = "#00ff66" if result["positive_ratio"] > 0.7 else "#ffe600" if result["positive_ratio"] > 0.5 else "#ff2a85"
                    fig = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=result["positive_ratio"] * 100,
                            number={"suffix": "%", "font": {"color": gauge_color, "family": "JetBrains Mono", "size": 42}},
                            gauge={
                                "axis": {"range": [0, 100], "tickcolor": "#9d4edf"},
                                "bar": {"color": gauge_color, "thickness": 0.3},
                                "bgcolor": "rgba(0,0,0,0)",
                                "bordercolor": "rgba(0,240,255,0.3)",
                                "steps": [
                                    {"range": [0, 50], "color": "rgba(255, 42, 133, 0.15)"},
                                    {"range": [50, 75], "color": "rgba(255, 230, 0, 0.15)"},
                                    {"range": [75, 100], "color": "rgba(0, 255, 102, 0.15)"},
                                ],
                            },
                        )
                    )
                    fig.update_layout(height=230, margin=dict(l=20, r=20, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", font_color="#fff")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                with st.container(border=True):
                    st.info("👈 Set your game's parameters on the left panel and hit **LAUNCH PREDICTION ENGINE** to execute models!")

    # --- Tab 2: Price Sensitivity ---
    with dev_tab[1]:
        st.markdown("### 💲 Price Optimization Sweep")
        with st.container(border=True):
            pc1, pc2, pc3 = st.columns([1, 1, 1])
            with pc1:
                ps_platforms = st.multiselect("Platforms", ["Windows", "Mac", "Linux"], default=["Windows"], key="ps_platforms")
            with pc2:
                ps_genres = st.multiselect("Genres", meta["top_genres"], default=["Indie", "Action"], key="ps_genres", max_selections=5)
            with pc3:
                ps_achievements = st.slider("Achievements", 0, 200, 20, key="ps_ach")

        if ps_platforms and ps_genres:
            price_points = list(np.arange(0, 61, 2.5))
            rows = []
            for p in price_points:
                row = build_feature_row(p, ps_achievements, ps_platforms, ps_genres, [], ps_genres)
                res = predict_all(row)
                rows.append({"price": p, "positive_ratio": res["positive_ratio"] * 100, "owners": res["owners"]})
            sweep_df = pd.DataFrame(rows)

            cc1, cc2 = st.columns(2)
            with cc1:
                fig1 = px.line(sweep_df, x="price", y="positive_ratio", markers=True, title="Positive Ratio vs Price ($)")
                fig1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(23, 14, 43, 0.5)", font_color="#fff", title_font_color="#00f0ff")
                fig1.update_traces(line=dict(color="#00ff66", width=3), marker=dict(color="#00f0ff", size=8))
                st.plotly_chart(fig1, use_container_width=True)
            with cc2:
                fig2 = px.line(sweep_df, x="price", y="owners", markers=True, title="Projected Reach (Owners) vs Price ($)")
                fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(23, 14, 43, 0.5)", font_color="#fff", title_font_color="#ff2a85")
                fig2.update_traces(line=dict(color="#ff2a85", width=3), marker=dict(color="#ffe600", size=8))
                st.plotly_chart(fig2, use_container_width=True)

    # --- Tab 3: What Drives Success ---
    with dev_tab[2]:
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("### 🎯 Key Feature Weights")
            importances = pd.DataFrame({
                "feature": FEATURE_COLS,
                "importance": models["clf"].feature_importances_,
            }).sort_values("importance", ascending=False).head(12)
            importances["feature"] = importances["feature"].str.replace("_", " ").str.title()

            fig = px.bar(importances.sort_values("importance"), x="importance", y="feature", orientation="h", color="importance", color_continuous_scale=["#00f0ff", "#ff2a85"])
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(23, 14, 43, 0.5)", font_color="#fff", coloraxis_showscale=False, height=450)
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.markdown("### 🌌 Genre Landscape")
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

            fig2 = px.scatter(genre_df, x="Avg owners", y="Avg positive ratio", size="Games in catalog", text="Genre", log_x=True, color="Avg positive ratio", color_continuous_scale=["#ff2a85", "#ffe600", "#00ff66"])
            fig2.update_traces(textposition="top center")
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(23, 14, 43, 0.5)", font_color="#fff", coloraxis_showscale=False, yaxis_tickformat=".0%", height=450)
            st.plotly_chart(fig2, use_container_width=True)


# ===========================================================================
# PLAYER CORNER
# ===========================================================================
else:
    st.markdown(
        '<div class="hero-box"><h1>🕹️ PLAYER CORNER</h1>'
        "<p>Unearth secret high-quality games & compare actual market performance against prediction models.</p></div>",
        unsafe_allow_html=True,
    )

    player_tab = st.tabs(["💎 HIDDEN GEMS", "⚖️ OVER / UNDERRATED", "🔍 CATALOG MATRIX"])

    # --- Tab 1: Hidden Gems ---
    with player_tab[0]:
        st.markdown("### 💎 Low-Volume High-Quality Gems")
        max_price = st.slider("Max Budget ($)", 0.0, 60.0, 20.0, key="gems_price")

        median_owners = catalog["owners_avg"].median()
        gems = catalog[(catalog["owners_avg"] <= median_owners) & (catalog["positive_ratio"] > 0.8) & (catalog["price"] <= max_price)].sort_values("positive_ratio", ascending=False).head(10)

        cols = st.columns(2)
        for i, (_, g) in enumerate(gems.iterrows()):
            with cols[i % 2]:
                with st.container(border=True):
                    st.markdown(f"### {g['name']}")
                    st.markdown(f"<span class='custom-tag'>{g['genres']}</span>", unsafe_allow_html=True)
                    st.markdown(f"<h3 style='color: var(--neon-green); margin-top:10px;'>{g['positive_ratio']:.0%} Positive</h3>", unsafe_allow_html=True)
                    st.caption(f"Price: **${g['price']:.2f}** | Total Reviews: **{int(g['total_ratings'])}** | Est. Owners: **{format_owners(g['owners_avg'])}**")

    # --- Tab 2: Over/Underrated ---
    with player_tab[1]:
        st.markdown("### ⚖️ Performance Gap Analysis")
        with st.container(border=True):
            f1, f2 = st.columns(2)
            with f1:
                direction = st.radio("Performance Metric", ["Overperformers (Better than expected)", "Underperformers (Worse than expected)"], horizontal=True)
            with f2:
                min_reviews = st.slider("Min Review Count", 10, 5000, 100, key="gap_reviews")

        pool = catalog[catalog["total_ratings"] >= min_reviews]
        if direction.startswith("Over"):
            top = pool.sort_values("reception_gap", ascending=False).head(10)
            gap_color = "var(--neon-green)"
        else:
            top = pool.sort_values("reception_gap", ascending=True).head(10)
            gap_color = "var(--neon-pink)"

        cols = st.columns(2)
        for i, (_, g) in enumerate(top.iterrows()):
            with cols[i % 2]:
                with st.container(border=True):
                    st.markdown(f"### {g['name']}")
                    st.caption(f"Genre: {g['genres']}")
                    st.markdown(f"<h3 style='color: {gap_color};'>Gap: {g['reception_gap']:+.0%}</h3>", unsafe_allow_html=True)
                    st.write(f"Actual Ratio: **{g['positive_ratio']:.0%}** | Model Predicted: **{g['pred_positive_ratio']:.0%}**")

    # --- Tab 3: Catalog Matrix ---
    with player_tab[2]:
        st.markdown("### 🔍 Full Catalog Explorer")
        with st.container(border=True):
            f1, f2, f3 = st.columns(3)
            with f1:
                genre_filter = st.multiselect("Genre Filter", meta["top_genres"])
            with f2:
                price_range = st.slider("Price Window ($)", 0.0, 60.0, (0.0, 30.0))
            with f3:
                min_tier = st.selectbox("Min Tier Classification", ["Any", "Average", "Great"])

        filtered = catalog[(catalog["price"] >= price_range[0]) & (catalog["price"] <= price_range[1])]
        for g in genre_filter:
            col = "genre_" + g.replace(" ", "_").replace("-", "_")
            if col in filtered.columns:
                filtered = filtered[filtered[col] == 1]
        if min_tier == "Great":
            filtered = filtered[filtered["pred_reception_tier"] == "Great"]
        elif min_tier == "Average":
            filtered = filtered[filtered["pred_reception_tier"].isin(["Average", "Great"])]

        filtered = filtered.sort_values("positive_ratio", ascending=False)
        st.caption(f"⚡ Showing top 25 of {len(filtered):,} matching games")

        display_cols = ["name", "genres", "price", "positive_ratio", "owners", "pred_reception_tier"]
        show = filtered[display_cols].head(25).rename(columns={
            "name": "Game", "genres": "Genres", "price": "Price ($)",
            "positive_ratio": "Positive Ratio", "owners": "Owners (Est.)",
            "pred_reception_tier": "Model Tier",
        })
        show["Positive Ratio"] = (show["Positive Ratio"] * 100).round(1).astype(str) + "%"
        st.dataframe(show, use_container_width=True, hide_index=True)