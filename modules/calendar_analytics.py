# -*- coding: utf-8 -*-
"""calendar_analytics.py — 未來檔期分析模組

資料來源:scripts/build_calendar_features.py 產出的輕量檔案
(不直接讀 234 萬列的 calendar.csv.gz,以符合 Streamlit Cloud 記憶體限制)。

主要能力
--------
1. 逐日訂房遮罩 → 日曆熱度資料
2. 未來各月已訂率 vs 同商圈基準
3. 連續空檔警示(未來 90 天)
4. 營收最適定價:以同商圈同房型的「真實已訂天數」建立營收曲線

重要限制(見 doc/04):Inside Airbnb 的 available='f' 同時包含「已被預訂」與
「房東主動封鎖」,故一律排除全年封鎖與全年全空的房源,並以未來 0~90 天為主要判讀窗口。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    import streamlit as st
    cache_data = st.cache_data
except Exception:                       # 允許無 Streamlit 環境測試
    def cache_data(*a, **k):
        def deco(f):
            return f
        return deco if not a else a[0]

DATA = Path(__file__).resolve().parent.parent / "data"
METRICS_CSV = DATA / "_calendar_metrics.csv"
MARKET_CSV = DATA / "_calendar_market.csv"


def available() -> bool:
    """檔期產物是否就緒。"""
    return METRICS_CSV.exists() and MARKET_CSV.exists()


@cache_data(show_spinner="載入未來檔期資料 …")
def load_metrics() -> pd.DataFrame:
    """每房源檔期指標(含 365 天訂房遮罩字串)。"""
    df = pd.read_csv(METRICS_CSV, dtype={"booked_mask": str})
    df["cal_start"] = pd.to_datetime(df["cal_start"], errors="coerce")
    return df


@cache_data(show_spinner=False)
def load_market() -> pd.DataFrame:
    """行政區 × 房型 × 月份 的市場已訂率基準。"""
    return pd.read_csv(MARKET_CSV)


@cache_data(show_spinner=False)
def healthy_metrics() -> pd.DataFrame:
    """排除全年封鎖 / 全年全空之異常房源(資料陷阱防護)。"""
    m = load_metrics()
    return m[(m["is_all_blocked"] == 0) & (m["is_all_open"] == 0)]


def get_listing(listing_id: int):
    """取單一房源檔期指標;找不到回傳 None。"""
    m = load_metrics()
    hit = m[m["listing_id"] == int(listing_id)]
    return None if hit.empty else hit.iloc[0]


def daily_frame(row) -> pd.DataFrame:
    """把訂房遮罩展開為逐日 DataFrame(date / booked / 週序 / 星期)。"""
    mask = str(row["booked_mask"])
    dates = pd.date_range(row["cal_start"], periods=len(mask), freq="D")
    d = pd.DataFrame({
        "date": dates,
        "booked": [int(ch) for ch in mask],
    })
    d["dow"] = d["date"].dt.dayofweek
    d["week"] = ((d["date"] - d["date"].min()).dt.days // 7).astype(int)
    d["month"] = d["date"].dt.to_period("M").astype(str)
    d["horizon"] = (d["date"] - d["date"].min()).dt.days
    return d


def monthly_vs_market(row, district: str, room_type: str) -> pd.DataFrame:
    """未來 12 個月:本房源已訂率 vs 同商圈同房型基準。"""
    mine = [(i, row.get(f"m{i}_rate")) for i in range(1, 13)]
    mkt = load_market()
    sub = mkt[(mkt["neighbourhood_cleansed"] == district)
              & (mkt["room_type"] == room_type)]
    # 樣本不足時退回同行政區(不分房型)
    if sub["n_days"].sum() < 3000:
        sub = (mkt[mkt["neighbourhood_cleansed"] == district]
               .groupby("mi", as_index=False)
               .apply(lambda g: pd.Series({
                   "mkt_rate": np.average(g["mkt_rate"], weights=g["n_days"]),
                   "n_days": g["n_days"].sum()}), include_groups=False)
               .reset_index(drop=True))
        sub["mi"] = range(1, len(sub) + 1)
    mkt_map = dict(zip(sub["mi"], sub["mkt_rate"]))
    start = pd.Timestamp(row["cal_start"])
    rows = []
    for i, v in mine:
        label = (start + pd.DateOffset(months=i - 1)).strftime("%Y-%m")
        rows.append({"月份": label, "本房源": v,
                     "同商圈基準": mkt_map.get(i, np.nan)})
    out = pd.DataFrame(rows).dropna(subset=["本房源"])
    out["差距"] = out["本房源"] - out["同商圈基準"]
    return out


def gap_segments(row, min_len: int = 5, horizon: int = 90) -> pd.DataFrame:
    """未來 N 天內的連續空檔區段(可訂且無訂單)。"""
    d = daily_frame(row)
    d = d[d["horizon"] <= horizon]
    segs, run_start, run = [], None, 0
    for _, r in d.iterrows():
        if r["booked"] == 0:
            if run == 0:
                run_start = r["date"]
            run += 1
        else:
            if run >= min_len:
                segs.append({"起日": run_start,
                             "迄日": run_start + pd.Timedelta(days=run - 1),
                             "連續天數": run})
            run = 0
    if run >= min_len:
        segs.append({"起日": run_start,
                     "迄日": run_start + pd.Timedelta(days=run - 1),
                     "連續天數": run})
    return pd.DataFrame(segs)


def peer_revenue_curve(listings: pd.DataFrame, district: str, room_type: str,
                       n_bands: int = 8) -> pd.DataFrame:
    """營收最適定價:同商圈同房型的「價格帶 × 真實已訂天數 × 年營收估算」。

    營收估算 = 每晚價格 × 真實已訂天數(已訂天數來自 calendar,與價格為
    獨立資料源,故非 doc/03 §3.3 所述的循環恆等式)。
    """
    ok = healthy_metrics()[["listing_id", "booked_days", "booked_rate"]]
    peer = listings[(listings["neighbourhood_cleansed"] == district)
                    & (listings["room_type"] == room_type)]
    d = peer.merge(ok, left_on="id", right_on="listing_id", how="inner")
    if len(d) < 30:      # 樣本不足 → 放寬為同房型全市
        peer = listings[listings["room_type"] == room_type]
        d = peer.merge(ok, left_on="id", right_on="listing_id", how="inner")
    # price 可能為含符號字串(未經 data_loader 清理),此處自行轉數值
    d = d.copy()
    d["price"] = pd.to_numeric(
        d["price"].astype(str).str.replace(r"[$,]", "", regex=True),
        errors="coerce")
    d = d.dropna(subset=["price"])
    d = d[(d["price"] > 0) & (d["price"] < d["price"].quantile(.98))]
    if len(d) < 20:
        return pd.DataFrame()
    d["band"] = pd.qcut(d["price"], min(n_bands, d["price"].nunique()),
                        duplicates="drop")
    g = (d.groupby("band", observed=True)
         .agg(價格中位=("price", "median"),
              已訂天數=("booked_days", "mean"),
              已訂率=("booked_rate", "mean"),
              樣本數=("price", "size")).reset_index(drop=True))
    g["年營收估算"] = (g["價格中位"] * g["已訂天數"]).round(0)
    return g


def optimal_price(curve: pd.DataFrame) -> dict | None:
    """由營收曲線取最適價格帶。"""
    if curve is None or curve.empty:
        return None
    i = int(curve["年營收估算"].idxmax())
    r = curve.loc[i]
    return {"price": float(r["價格中位"]), "revenue": float(r["年營收估算"]),
            "booked_days": float(r["已訂天數"]), "n": int(r["樣本數"])}


def portfolio_summary(listings: pd.DataFrame) -> pd.DataFrame:
    """房型獲利分析:房型 × 行政區 的真實已訂率與營收估算。"""
    ok = healthy_metrics()[["listing_id", "booked_days", "booked_rate"]]
    d = listings.merge(ok, left_on="id", right_on="listing_id", how="inner").copy()
    d["price"] = pd.to_numeric(
        d["price"].astype(str).str.replace(r"[$,]", "", regex=True),
        errors="coerce")
    d = d.dropna(subset=["price"])
    d["年營收估算"] = d["price"] * d["booked_days"]
    return d
