"""
verify_artifacts.py
===================
Post-training sanity checker. Runs independently of Streamlit.
Exits 0 on full pass, 1 on any failure.

Usage:
    python verify_artifacts.py
"""

import json
import sys
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PASS = "\033[32m[PASS]\033[0m"
FAIL = "\033[31m[FAIL]\033[0m"
INFO = "\033[34m[INFO]\033[0m"
SEP  = "=" * 65

ROOT   = Path(__file__).resolve().parent
MODELS = ROOT / "models"

errors = 0

def ok(msg):
    print(f"  {PASS}  {msg}")

def fail(msg):
    global errors
    errors += 1
    print(f"  {FAIL}  {msg}")

def info(msg):
    print(f"  {INFO}  {msg}")


# ---------------------------------------------------------------------------
# 1. All expected files exist and are non-empty
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("1. FILE EXISTENCE & SIZE")
print(SEP)

EXPECTED = {
    "clf_well_received.joblib":  1_000,      # >1 KB
    "clf_reception_tier.joblib": 1_000,
    "reg_positive_ratio.joblib": 1_000,
    "reg_owners.joblib":         1_000,
    "meta.json":                 500,
    "steam_clean.csv":           1_000_000,  # >1 MB
}

for fname, min_bytes in EXPECTED.items():
    p = MODELS / fname
    if not p.exists():
        fail(f"{fname}  — MISSING")
    elif p.stat().st_size < min_bytes:
        fail(f"{fname}  — suspiciously small ({p.stat().st_size:,} bytes, expected >{min_bytes:,})")
    else:
        ok(f"{fname}  ({p.stat().st_size:,} bytes)")


# ---------------------------------------------------------------------------
# 2. steam_clean.csv — no duplicate columns
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("2. steam_clean.csv — DUPLICATE COLUMN CHECK")
print(SEP)

csv_path = MODELS / "steam_clean.csv"
try:
    catalog = pd.read_csv(csv_path)
    info(f"Shape: {catalog.shape[0]:,} rows × {catalog.shape[1]} columns")

    cols = list(catalog.columns)
    seen = {}
    dupes = []
    for i, c in enumerate(cols):
        if c in seen:
            dupes.append((c, seen[c], i))
        else:
            seen[c] = i

    if dupes:
        for name, first_idx, second_idx in dupes:
            fail(f"Duplicate column '{name}' at positions {first_idx} and {second_idx}")
    else:
        ok(f"No duplicate columns ({len(cols)} unique column names)")

    # Spot-check: columns that were duplicated in the original bug
    PREVIOUSLY_DUPED = ["price", "achievements", "required_age"]
    for col in PREVIOUSLY_DUPED:
        count = cols.count(col)
        if count == 1:
            ok(f"  '{col}' appears exactly once")
        elif count == 0:
            fail(f"  '{col}' is completely absent from steam_clean.csv")
        else:
            fail(f"  '{col}' still appears {count} times — duplicate bug not fixed")

    # Check all values in key numeric columns are actually scalar (not DataFrame)
    for col in PREVIOUSLY_DUPED:
        if col in catalog.columns:
            sample = catalog[col].iloc[0]
            if isinstance(sample, (int, float, np.integer, np.floating)):
                ok(f"  catalog['{col}'].iloc[0] is scalar: {sample}")
            else:
                fail(f"  catalog['{col}'].iloc[0] is {type(sample).__name__} — still a DataFrame column?")

except Exception:
    fail("Could not load steam_clean.csv")
    traceback.print_exc()
    catalog = None


# ---------------------------------------------------------------------------
# 3. meta.json — schema check
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("3. meta.json — SCHEMA & CONSISTENCY")
print(SEP)

meta = None
try:
    meta = json.load(open(MODELS / "meta.json", encoding="utf-8"))

    REQUIRED_KEYS = [
        "feature_cols", "genre_cols", "cat_cols", "tag_cols",
        "top_genres", "top_tags", "key_categories",
        "global_mean_ratio", "tier_classes", "defaults", "metrics",
        "dataset_stats",
    ]
    for key in REQUIRED_KEYS:
        if key in meta:
            ok(f"Key '{key}' present")
        else:
            fail(f"Key '{key}' MISSING from meta.json")

    info(f"feature_cols count : {len(meta.get('feature_cols', []))}")
    info(f"top_genres         : {meta.get('top_genres')}")
    info(f"top_tags (first 5) : {meta.get('top_tags', [])[:5]}")
    info(f"tier_classes       : {meta.get('tier_classes')}")
    info(f"global_mean_ratio  : {meta.get('global_mean_ratio'):.4f}")
    info(f"defaults           : {meta.get('defaults')}")
    info(f"dataset n_games    : {meta.get('dataset_stats', {}).get('n_games')}")

    # Defaults should all be finite numbers
    defaults = meta.get("defaults", {})
    for k, v in defaults.items():
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            fail(f"defaults['{k}'] = {v!r} — not a finite number")
        else:
            ok(f"defaults['{k}'] = {v}")

except Exception:
    fail("Could not load / parse meta.json")
    traceback.print_exc()


# ---------------------------------------------------------------------------
# 4. All four joblib models load and produce predictions
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("4. JOBLIB MODELS — LOAD & SMOKE-TEST PREDICT")
print(SEP)

models = {}
MODEL_FILES = {
    "clf":    "clf_well_received.joblib",
    "tier":   "clf_reception_tier.joblib",
    "ratio":  "reg_positive_ratio.joblib",
    "owners": "reg_owners.joblib",
}
for key, fname in MODEL_FILES.items():
    try:
        m = joblib.load(MODELS / fname)
        models[key] = m
        ok(f"Loaded '{fname}' → {type(m).__name__}")
    except Exception:
        fail(f"Failed to load '{fname}'")
        traceback.print_exc()

if meta and len(models) == 4:
    try:
        FEATURE_COLS = meta["feature_cols"]
        row = pd.DataFrame(0.0, index=[0], columns=FEATURE_COLS)
        # Fill a plausible row
        row["price"] = 14.99
        row["price_log"] = np.log1p(14.99)
        row["is_free"] = 0
        row["achievements"] = 20
        row["achievements_log"] = np.log1p(20)
        row["achievements_per_dollar"] = 20 / 15.99
        row["dev_avg_rating"] = meta["global_mean_ratio"]
        row["pub_avg_rating"] = meta["global_mean_ratio"]
        row["num_genres"] = 2
        row["num_platforms"] = 1
        row["release_month"] = meta["defaults"]["release_month"]
        row["release_year_filled"] = meta["defaults"]["release_year_filled"]
        row["name_len"] = meta["defaults"]["name_len"]
        row["name_word_count"] = meta["defaults"]["name_word_count"]
        if "platform_windows" in FEATURE_COLS:
            row["platform_windows"] = 1

        # clf
        proba = float(models["clf"].predict_proba(row)[0][1])
        ok(f"clf predict_proba → well_received_proba = {proba:.4f}  ({'OK' if 0<=proba<=1 else 'OUT OF RANGE'})")

        # tier
        tier_idx   = int(models["tier"].predict(row)[0])
        tier_label = meta["tier_classes"][tier_idx]
        tier_proba = models["tier"].predict_proba(row)[0]
        ok(f"tier predict → {tier_label} (idx={tier_idx}, confidence={tier_proba.max():.4f})")

        # ratio
        ratio = float(np.clip(models["ratio"].predict(row)[0], 0, 1))
        ok(f"ratio predict → positive_ratio = {ratio:.4f}")

        # owners
        owners_raw = float(models["owners"].predict(row)[0])
        owners = max(0.0, float(np.expm1(owners_raw)))
        ok(f"owners predict → {owners:,.0f} owners  (raw log-space: {owners_raw:.4f})")

    except Exception:
        fail("Smoke-test predict() failed")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# 5. Feature column consistency: meta.json ↔ steam_clean.csv
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("5. FEATURE COLUMN CONSISTENCY (meta.json ↔ steam_clean.csv)")
print(SEP)

if meta and catalog is not None:
    feature_cols = set(meta.get("feature_cols", []))
    csv_cols = set(catalog.columns)

    in_meta_not_csv = feature_cols - csv_cols
    if in_meta_not_csv:
        fail(f"{len(in_meta_not_csv)} feature_cols from meta.json NOT in steam_clean.csv: {sorted(in_meta_not_csv)[:5]}...")
    else:
        ok(f"All {len(feature_cols)} feature_cols from meta.json are present in steam_clean.csv")

    # Genre / tag / cat cols
    for group_key in ["genre_cols", "cat_cols", "tag_cols"]:
        group = meta.get(group_key, [])
        missing_from_csv = [c for c in group if c not in csv_cols]
        if missing_from_csv:
            fail(f"{group_key}: {len(missing_from_csv)} cols missing from CSV: {missing_from_csv[:3]}")
        else:
            ok(f"{group_key}: all {len(group)} cols present in CSV")
else:
    info("Skipped (meta or catalog failed to load)")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
if errors == 0:
    print(f"  {PASS}  ALL CHECKS PASSED — artifacts are consistent and ready.")
else:
    print(f"  {FAIL}  {errors} CHECK(S) FAILED — see details above.")
print(SEP)

sys.exit(0 if errors == 0 else 1)
