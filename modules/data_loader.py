"""
Data Loader — Centralized data loading with Streamlit caching.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from modules.ui_components import ROOM_JP
from modules.ml_models import compute_risk_scores

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _read_csv(stem, **kw):
    """Read <stem>.csv.gz if present (deploy build), else <stem>.csv."""
    gz = DATA_DIR / f"{stem}.csv.gz"
    csv = DATA_DIR / f"{stem}.csv"
    return pd.read_csv(gz if gz.exists() else csv, **kw)


def _to_price(s: pd.Series) -> pd.Series:
    """價格欄轉數值:相容 "$1,238.43" 這類含貨幣符號與千分位的字串。

    原本直接 pd.to_numeric 會全部變 NaN,導致整份資料被過濾為空(已修)。
    """
    if pd.api.types.is_numeric_dtype(s):
        return s
    return pd.to_numeric(
        s.astype(str).str.replace(r"[^\d.\-]", "", regex=True), errors="coerce")


@st.cache_data(show_spinner=False)
def load_listings():
    """Load and preprocess listings data."""
    df = _read_csv("listings_cleaned", low_memory=False)
    df["price"] = _to_price(df["price"])
    df = df[df["price"].between(200, 80000)].copy()
    df["room_type_zh"] = df["room_type"].map(ROOM_JP).fillna(df["room_type"])
    df["reviews_per_month"] = df["reviews_per_month"].fillna(0)
    df["number_of_reviews_ltm"] = df["number_of_reviews_ltm"].fillna(0)
    df["number_of_reviews_l30d"] = df.get("number_of_reviews_l30d", pd.Series(0)).fillna(0)
    df["occupancy_pct"] = (
        (365 - df["availability_365"]) / 365 * 100
    ).clip(0, 100).round(1)

    # Review scores
    for col in ["review_scores_rating", "review_scores_cleanliness",
                "review_scores_location", "review_scores_value",
                "review_scores_communication", "review_scores_checkin",
                "review_scores_accuracy"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Dates
    df["last_review"] = pd.to_datetime(df["last_review"], errors="coerce")
    df["first_review"] = pd.to_datetime(df["first_review"], errors="coerce")

    # Numeric columns
    for col in ["bedrooms", "beds", "bathrooms_count", "accommodates"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Compute risk scores
    df = compute_risk_scores(df)

    return df.reset_index(drop=True)


def _guess_lang(s: str) -> str:
    """粗略語言判定:含 CJK 字元即視為中文,否則英文(僅供 NLP 模組分組用)。"""
    txt = str(s)
    has_cjk = any("一" <= ch <= "鿿" for ch in txt[:80])
    has_en = any(ch.isascii() and ch.isalpha() for ch in txt[:80])
    if has_cjk and has_en:
        return "mixed_zh_en"
    if has_cjk:
        return "zh"
    return "en" if has_en else "other"


@st.cache_data(show_spinner=False)
def load_reviews():
    """Load reviews (Parquet preferred — robust dtypes; falls back to .csv.gz/.csv).

    注意:parquet 版本缺少 cleaned_comments / language_type 兩欄(csv.gz 版才有),
    此處統一補齊,避免 NLP 模組因欄位不存在而崩潰。
    """
    pq = DATA_DIR / "reviews_cleaned.parquet"
    if pq.exists():
        df = pd.read_parquet(pq)
    else:
        df = _read_csv("reviews_cleaned", low_memory=False)
    # Guarantee listing_id is a clean int64 so it always matches listing ids
    df["listing_id"] = pd.to_numeric(df["listing_id"], errors="coerce")
    df = df.dropna(subset=["listing_id"])
    df["listing_id"] = df["listing_id"].astype("int64")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # ── 欄位相容層 ──
    if "cleaned_comments" not in df.columns:
        df["cleaned_comments"] = df.get("comments", pd.Series("", index=df.index))
    if "comments" not in df.columns:
        df["comments"] = df["cleaned_comments"]
    if "language_type" not in df.columns:
        df["language_type"] = df["cleaned_comments"].map(_guess_lang)
    return df


@st.cache_data(show_spinner=False)
def load_neighbourhoods():
    """Load neighbourhood list."""
    try:
        df = pd.read_csv(DATA_DIR / "neighbourhoods.csv", encoding="utf-8")
    except Exception:
        df = pd.read_csv(DATA_DIR / "neighbourhoods.csv", encoding="big5")
    r