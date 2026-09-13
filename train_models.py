"""
train_models.py
================
Loads the raw Steam dataset, cleans it, engineers features, trains the four
production models (well-received classifier, reception-tier classifier,
positive-ratio regressor, owners regressor), and saves everything the
Streamlit app needs to run without retraining.

Run once:
    python train_models.py

Produces (under models/):
    clf_well_received.joblib
    clf_reception_tier.joblib
    reg_positive_ratio.joblib
    reg_owners.joblib
    feature_cols.json
    meta.json                 (top genres/cats/tags, medians, global mean, label encoder classes)
    steam_clean.csv           (cleaned + feature-engineered dataset, used by the app's Player pages)
"""

import json
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load & clean
# ---------------------------------------------------------------------------
print("Loading data...")
df = pd.read_csv("data/steam.csv")

df = df[(df["positive_ratings"] + df["negative_ratings"]) > 0].copy()
df = df[df["english"] == 1].copy()
df["developer"] = df["developer"].fillna("Unknown")
df["publisher"] = df["publisher"].fillna("Unknown")

df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
df["release_year"] = df["release_date"].dt.year


def parse_owners(owner_str):
    low, high = owner_str.split("-")
    return (int(low) + int(high)) / 2


df["owners_avg"] = df["owners"].apply(parse_owners)
df["owners_log"] = np.log1p(df["owners_avg"])

df["total_ratings"] = df["positive_ratings"] + df["negative_ratings"]
df["positive_ratio"] = df["positive_ratings"] / df["total_ratings"]
df["well_received"] = (df["positive_ratio"] > 0.7).astype(int)


def reception_tier(r):
    if r < 0.5:
        return "Poor"
    elif r < 0.75:
        return "Average"
    return "Great"


df["reception_tier"] = df["positive_ratio"].apply(reception_tier)
print(f"Cleaned shape: {df.shape}")

# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------
print("Engineering features...")

for p in ["windows", "mac", "linux"]:
    df[f"platform_{p}"] = df["platforms"].apply(lambda x: int(p in x.split(";")))
df["num_platforms"] = df[["platform_windows", "platform_mac", "platform_linux"]].sum(axis=1)

all_genres = [g for sub in df["genres"].str.split(";") for g in sub]
top_genres = [g for g, _ in Counter(all_genres).most_common(10)]
genre_cols = []
for g in top_genres:
    col = "genre_" + g.replace(" ", "_").replace("-", "_")
    df[col] = df["genres"].apply(lambda x: int(g in x.split(";")))
    genre_cols.append(col)
df["num_genres"] = df["genres"].apply(lambda x: len(x.split(";")))

key_categories = {
    "cat_multiplayer": "Multi-player",
    "cat_singleplayer": "Single-player",
    "cat_coop": "Co-op",
    "cat_online_multiplayer": "Online Multi-Player",
    "cat_steam_achievements": "Steam Achievements",
    "cat_steam_trading_cards": "Steam Trading Cards",
    "cat_full_controller_support": "Full controller support",
    "cat_steam_cloud": "Steam Cloud",
}
cat_cols = list(key_categories.keys())
for col, val in key_categories.items():
    df[col] = df["categories"].apply(lambda x: int(val in x.split(";")))

all_tags = [t for sub in df["steamspy_tags"].str.split(";") for t in sub]
top_tags = [t for t, _ in Counter(all_tags).most_common(15)]
tag_cols = []
for t in top_tags:
    col = "tag_" + t.replace(" ", "_").replace("-", "_")
    df[col] = df["steamspy_tags"].apply(lambda x: int(t in x.split(";")))
    tag_cols.append(col)

df["price_log"] = np.log1p(df["price"])
df["is_free"] = (df["price"] == 0).astype(int)
df["achievements_per_dollar"] = df["achievements"] / (df["price"] + 1)
df["achievements_log"] = np.log1p(df["achievements"])

K = 10
global_mean_ratio = float(df["positive_ratio"].mean())

dev_group = df.groupby("developer")["positive_ratio"]
dev_sum, dev_count = dev_group.transform("sum"), dev_group.transform("count")
df["dev_avg_rating"] = ((dev_sum - df["positive_ratio"]) + K * global_mean_ratio) / (
    (dev_count - 1) + K
)
df["dev_game_count"] = dev_count

pub_group = df.groupby("publisher")["positive_ratio"]
pub_sum, pub_count = pub_group.transform("sum"), pub_group.transform("count")
df["pub_avg_rating"] = ((pub_sum - df["positive_ratio"]) + K * global_mean_ratio) / (
    (pub_count - 1) + K
)
df["pub_game_count"] = pub_count

df["release_month"] = df["release_date"].dt.month
df["release_year_filled"] = df["release_year"].fillna(df["release_year"].median())
df["name_len"] = df["name"].astype(str).apply(len)
df["name_word_count"] = df["name"].astype(str).apply(lambda x: len(x.split()))

extra_cols = [
    "pub_avg_rating",
    "pub_game_count",
    "release_month",
    "release_year_filled",
    "name_len",
    "name_word_count",
]
platform_cols = ["platform_windows", "platform_mac", "platform_linux", "num_platforms"]
numeric_cols = [
    "price",
    "price_log",
    "is_free",
    "achievements",
    "achievements_log",
    "achievements_per_dollar",
    "required_age",
    "num_genres",
    "dev_avg_rating",
    "dev_game_count",
] + extra_cols
feature_cols = numeric_cols + platform_cols + genre_cols + cat_cols + tag_cols
print(f"Total features: {len(feature_cols)}")

X = df[feature_cols]

# ---------------------------------------------------------------------------
# 3. Train models (hyperparameters below were selected via RandomizedSearchCV
#    in the accompanying analysis notebook; fixed here so this script is fast
#    and fully reproducible without re-running the search every time).
# ---------------------------------------------------------------------------
print("\nTraining well-received classifier...")
y_clf = df["well_received"]
Xtr, Xte, ytr, yte = train_test_split(
    X, y_clf, test_size=0.2, random_state=RANDOM_STATE, stratify=y_clf
)
clf_model = XGBClassifier(
    n_estimators=500,
    max_depth=7,
    learning_rate=0.03,
    subsample=0.85,
    colsample_bytree=0.7,
    min_child_weight=1,
    gamma=0,
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
clf_model.fit(Xtr, ytr)
proba = clf_model.predict_proba(Xte)[:, 1]
pred = clf_model.predict(Xte)
clf_metrics = {
    "roc_auc": round(float(roc_auc_score(yte, proba)), 4),
    "accuracy": round(float(accuracy_score(yte, pred)), 4),
    "f1": round(float(f1_score(yte, pred)), 4),
}
print(clf_metrics)
# Refit on ALL data for the shipped model (more data -> better real-world predictions)
clf_model.fit(X, y_clf)

print("\nTraining reception-tier classifier...")
y_tier = df["reception_tier"]
tier_encoder = LabelEncoder()
y_tier_enc = tier_encoder.fit_transform(y_tier)
Xtr_t, Xte_t, ytr_t, yte_t = train_test_split(
    X, y_tier_enc, test_size=0.2, random_state=RANDOM_STATE, stratify=y_tier_enc
)
tier_model = XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    eval_metric="mlogloss",
)
tier_model.fit(Xtr_t, ytr_t)
tier_pred = tier_model.predict(Xte_t)
tier_metrics = {"accuracy": round(float(accuracy_score(yte_t, tier_pred)), 4)}
print(tier_metrics)
tier_model.fit(X, y_tier_enc)

print("\nTraining positive-ratio regressor...")
y_ratio = df["positive_ratio"]
Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(X, y_ratio, test_size=0.2, random_state=RANDOM_STATE)
ratio_model = XGBRegressor(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
ratio_model.fit(Xtr_r, ytr_r)
ratio_pred = ratio_model.predict(Xte_r)
ratio_metrics = {
    "r2": round(float(r2_score(yte_r, ratio_pred)), 4),
    "rmse": round(float(mean_squared_error(yte_r, ratio_pred) ** 0.5), 4),
}
print(ratio_metrics)
ratio_model.fit(X, y_ratio)

print("\nTraining owners regressor...")
y_owners = df["owners_log"]
Xtr_o, Xte_o, ytr_o, yte_o = train_test_split(X, y_owners, test_size=0.2, random_state=RANDOM_STATE)
owners_model = XGBRegressor(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
owners_model.fit(Xtr_o, ytr_o)
owners_pred = owners_model.predict(Xte_o)
owners_metrics = {
    "r2": round(float(r2_score(yte_o, owners_pred)), 4),
    "rmse": round(float(mean_squared_error(yte_o, owners_pred) ** 0.5), 4),
}
print(owners_metrics)
owners_model.fit(X, y_owners)

# ---------------------------------------------------------------------------
# 4. Score the FULL catalog with the final models (used by the Player pages
#    for "hidden gems" / "over- vs under-rated" comparisons against reality).
# ---------------------------------------------------------------------------
print("\nScoring full catalog for product pages...")
df["pred_positive_ratio"] = ratio_model.predict(X)
df["pred_well_received_proba"] = clf_model.predict_proba(X)[:, 1]
df["pred_owners"] = np.expm1(owners_model.predict(X))
df["pred_reception_tier"] = tier_encoder.inverse_transform(tier_model.predict(X))
df["reception_gap"] = df["positive_ratio"] - df["pred_positive_ratio"]  # + = overperformed model
df["owners_gap_log"] = df["owners_log"] - np.log1p(df["pred_owners"])  # + = more owners than expected

# ---------------------------------------------------------------------------
# 5. Save everything
# ---------------------------------------------------------------------------
print("\nSaving artifacts...")
joblib.dump(clf_model, "models/clf_well_received.joblib")
joblib.dump(tier_model, "models/clf_reception_tier.joblib")
joblib.dump(ratio_model, "models/reg_positive_ratio.joblib")
joblib.dump(owners_model, "models/reg_owners.joblib")

meta = {
    "feature_cols": feature_cols,
    "genre_cols": genre_cols,
    "cat_cols": cat_cols,
    "tag_cols": tag_cols,
    "top_genres": top_genres,
    "top_tags": top_tags,
    "key_categories": key_categories,
    "global_mean_ratio": global_mean_ratio,
    "tier_classes": tier_encoder.classes_.tolist(),
    "defaults": {
        "release_month": int(df["release_month"].median()),
        "release_year_filled": float(df["release_year_filled"].median()),
        "name_len": float(df["name_len"].median()),
        "name_word_count": float(df["name_word_count"].median()),
    },
    "metrics": {
        "well_received_classifier": clf_metrics,
        "reception_tier_classifier": tier_metrics,
        "positive_ratio_regressor": ratio_metrics,
        "owners_regressor": owners_metrics,
    },
    "dataset_stats": {
        "n_games": int(len(df)),
        "date_range": [str(df["release_date"].min().date()), str(df["release_date"].max().date())],
    },
}
json.dump(meta, open("models/meta.json", "w"), indent=2)

keep_cols = [
    "appid", "name", "developer", "publisher", "release_date", "release_year",
    "price", "genres", "categories", "steamspy_tags", "platforms", "achievements",
    "required_age", "positive_ratings", "negative_ratings", "total_ratings",
    "positive_ratio", "well_received", "reception_tier", "owners", "owners_avg",
    "pred_positive_ratio", "pred_well_received_proba", "pred_owners",
    "pred_reception_tier", "reception_gap", "owners_gap_log",
] + feature_cols
df[keep_cols].to_csv("models/steam_clean.csv", index=False)

print("\nDone. All artifacts saved under models/.")
