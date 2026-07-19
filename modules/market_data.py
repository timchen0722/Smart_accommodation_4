# -*- coding: utf-8 -*-
"""跨平台市場資料層:載入並清理 Airbnb / 591 / Booking / ddroom 四平台房源。

統一輸出 schema(每列一房源):
  platform, title, district, lat, lon, capacity, bracket,
  price_raw(原始價格), price_unit('day'|'month'),
  price_day_eq(每晚等效價), price_pp_day(每人每晚等效價),
  rating(0~10, 可為 NaN), amenities(set), url, note
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"

# 跨平台共同設施字典(比對關鍵字 → 標準名)
AMENITY_CANON = {
    "冷氣": ["冷氣", "air conditioning", "ac unit"],
    "洗衣機": ["洗衣機", "washer", "laundry"],
    "冰箱": ["冰箱", "refrigerator", "fridge"],
    "電視": ["電視", "第四台", "tv", "hdtv"],
    "WiFi": ["wifi", "網路", "wireless", "internet"],
    "電梯": ["電梯", "elevator"],
    "車位": ["車位", "parking", "停車"],
    "陽台": ["陽台", "balcony", "patio"],
    "浴缸": ["浴缸", "bathtub", "bath tub"],
    "熱水器": ["熱水器", "hot water"],
    "廚房/可開伙": ["可開伙", "廚房", "kitchen", "天然瓦斯"],
    "自助入住": ["self check-in", "keypad", "lockbox", "smart lock", "自助入住"],
    "飲水機": ["飲水機", "water dispenser"],
    "工作空間": ["workspace", "桌椅", "dedicated workspace"],
}


def _canon_amenities(text: str) -> set:
    """從自由文字比對出標準設施集合。"""
    if not isinstance(text, str) or not text:
        return set()
    low = text.lower()
    return {k for k, kws in AMENITY_CANON.items() if any(w in low for w in kws)}


def capacity_bracket(c) -> str:
    if pd.isna(c):
        return "未知"
    c = float(c)
    if c <= 2:
        return "1-2人"
    if c <= 4:
        return "3-4人"
    return "5人以上"


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["lat", "lon", "price_day_eq"]).copy()
    df = df[(df["price_day_eq"] > 50) & (df["price_day_eq"] < 100_000)]
    df["capacity"] = df["capacity"].fillna(2).clip(1, 16)
    df["price_pp_day"] = df["price_day_eq"] / df["capacity"]
    df["bracket"] = df["capacity"].map(capacity_bracket)
    # 台北市經緯度粗略框(排除座標異常)
    df = df[(df["lat"].between(24.9, 25.3)) & (df["lon"].between(121.3, 121.7))]
    return df.reset_index(drop=True)


# ---------------- 591(月租) ----------------
_591_AMEN_COLS = ["冰箱", "洗衣機", "電視", "冷氣", "熱水器", "床", "衣櫃", "第四台",
                  "網路WiFi", "天然瓦斯", "沙發", "桌椅", "陽台", "電梯", "車位", "浴缸"]


def load_591() -> pd.DataFrame:
    raw = pd.read_csv(DATA / "591_taipei_20260718_124800_rooms.csv", encoding="utf-8-sig")
    raw = raw[~raw["房屋類型"].isin(["車位", "其他"])]
    price_m = pd.to_numeric(raw["價格"], errors="coerce")
    cap = raw["可住人數"].astype(str).str.extract(r"(\d+)")[0].astype(float)
    district = raw["地址"].astype(str).str.extract(r"^([^\-]{1,3}區)")[0]

    def amens(row):
        """將 591 之 16 個「有/無」設施欄轉為標準設施集合。"""
        owned = [c for c in _591_AMEN_COLS if str(row.get(c, "")) == "有"]
        return _canon_amenities("、".join(owned))

    df = pd.DataFrame({
        "platform": "591",
        "title": raw["標題"],
        "district": district,
        "lat": pd.to_numeric(raw["緯度"], errors="coerce"),
        "lon": pd.to_numeric(raw["經度"], errors="coerce"),
        "capacity": cap,
        "price_raw": price_m,
        "price_unit": "month",
        "price_day_eq": price_m / 30.0,
        "rating": np.nan,
        "amenities": raw.apply(amens, axis=1),
        "url": raw["連結"],
        "note": raw["房屋類型"].astype(str) + "|" + raw["坪數"].astype(str),
    })
    return _finalize(df)


# ---------------- ddroom / 租租網(月租) ----------------
def load_ddroom() -> pd.DataFrame:
    raw = pd.read_csv(DATA / "ddroom_taipei_by_rooms.csv", encoding="utf-8-sig")
    price_m = pd.to_numeric(raw["價格(元/月)"], errors="coerce")
    cap = raw["可住人數"].astype(str).str.extract(r"(\d+)\s*$")[0].astype(float)  # "1~2" 取上限
    df = pd.DataFrame({
        "platform": "ddroom",
        "title": raw["標題"],
        "district": raw["行政區"],
        "lat": pd.to_numeric(raw["緯度"], errors="coerce"),
        "lon": pd.to_numeric(raw["經度"], errors="coerce"),
        "capacity": cap,
        "price_raw": price_m,
        "price_unit": "month",
        "price_day_eq": price_m / 30.0,
        "rating": np.nan,
        "amenities": raw["特色標籤"].astype(str).map(_canon_amenities),
        "url": raw["url"],
        "note": raw["房型"].astype(str) + "|" + raw["坪數"].astype(str) + "坪|最短租期" +
                raw["最短租期(月)"].astype(str) + "月",
    })
    return _finalize(df)


# ---------------- Booking.com(日租) ----------------
def load_booking() -> pd.DataFrame:
    raw = pd.read_csv(DATA / "taipei_rooms_only_v14_20260718.csv", encoding="utf-8-sig")
    price_d = pd.to_numeric(raw["價格"].astype(str).str.replace(",", ""), errors="coerce")
    cap = pd.to_numeric(raw["可住宿人數"], errors="coerce")
    rating = pd.to_numeric(raw["評論分數"], errors="coerce")  # 0~10
    amen_text = raw["房間設施"].astype(str) + "、" + raw["館內熱門設施"].astype(str)
    df = pd.DataFrame({
        "platform": "Booking",
        "title": raw["飯店名稱"].astype(str) + " - " + raw["房型名稱"].astype(str),
        "district": raw["行政區"],
        "lat": pd.to_numeric(raw["緯度"], errors="coerce"),
        "lon": pd.to_numeric(raw["經度"], errors="coerce"),
        "capacity": cap,
        "price_raw": price_d,
        "price_unit": "day",
        "price_day_eq": price_d,
        "rating": rating,
        "amenities": amen_text.map(_canon_amenities),
        "url": raw["飯店連結"],
        "note": raw["房型名稱"].astype(str),
    })
    return _finalize(df)


# ---------------- Airbnb(日租, 主體) ----------------
def load_airbnb() -> pd.DataFrame:
    raw = pd.read_csv(
        DATA / "listings_cleaned.csv.gz",
        usecols=["id", "name", "latitude", "longitude", "price", "accommodates",
                 "room_type", "neighbourhood_cleansed", "amenities", "listing_url",
                 "review_scores_rating"],
    )
    price_d = pd.to_numeric(
        raw["price"].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")
    df = pd.DataFrame({
        "platform": "Airbnb",
        "listing_id": raw["id"],
        "title": raw["name"],
        "district": raw["neighbourhood_cleansed"],
        "lat": raw["latitude"],
        "lon": raw["longitude"],
        "capacity": pd.to_numeric(raw["accommodates"], errors="coerce"),
        "price_raw": price_d,
        "price_unit": "day",
        "price_day_eq": price_d,
        "rating": pd.to_numeric(raw["review_scores_rating"], errors="coerce") * 2,  # 0~5 → 0~10
        "amenities": raw["amenities"].astype(str).map(_canon_amenities),
        "url": raw["listing_url"],
        "note": raw["room_type"],
    })
    return _finalize(df)


def load_all_market() -> pd.DataFrame:
    """四平台合併(租客入口與競品索引共用)。"""
    frames = [load_airbnb(), load_booking(), load_591(), load_ddroom()]
    return pd.concat(frames, ignore_index=True)
