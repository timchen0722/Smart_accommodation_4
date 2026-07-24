# -*- coding: utf-8 -*-
"""platform_analytics.py — Airbnb 平台方後台的純計算層。

刻意不 import streamlit:所有函式皆為 DataFrame in / DataFrame(或 dict) out,
可離線用 pytest 驗證;快取由呼叫端(platform_sections)負責。

營收口徑(doc/07 + 2026-07-23 v90 雙輸出):
  預估年營收 = price x (1 - vac_pred_365) x 365   ← 年營收用 365 天空屋率
  平台收入   = 預估年營收 x 抽成率
  風險/空屋率顯示則用 vac_pred(90 天)。vac_pred_365 缺檔時回退 vac_pred。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COMMISSION_DEFAULT = 0.15
COMMISSION_MIN = 0.03
COMMISSION_MAX = 0.20

DAYS_PER_YEAR = 365


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0.0)


def add_revenue_columns(df: pd.DataFrame, commission: float) -> pd.DataFrame:
    """回傳新 DataFrame,附加 est_annual_revenue 與 platform_revenue。

    年營收採 365 天空屋率(vac_pred_365);舊資料無該欄時回退 vac_pred。
    """
    out = df.copy()
    price = _num(out["price"])
    vac_col = "vac_pred_365" if "vac_pred_365" in out.columns else "vac_pred"
    occ = (1.0 - _num(out[vac_col])).clip(0.0, 1.0)
    out["est_annual_revenue"] = price * occ * DAYS_PER_YEAR
    out["platform_revenue"] = out["est_annual_revenue"] * float(commission)
    return out


def market_kpis(df: pd.DataFrame, commission: float) -> dict:
    """全市(或篩選範圍)KPI;空母體回傳全零而非例外。"""
    n = int(len(df))
    if n == 0:
        return {"n_listings": 0, "n_hosts": 0, "avg_vacancy": 0.0,
                "red_ratio": 0.0, "yellow_ratio": 0.0,
                "total_revenue": 0.0, "platform_revenue": 0.0}
    d = add_revenue_columns(df, commission)
    tier = d["tier"].astype(str)
    total = float(d["est_annual_revenue"].sum())
    return {
        "n_listings": n,
        "n_hosts": int(d["host_id"].nunique()),
        "avg_vacancy": float(_num(d["vac_pred"]).mean()),
        "red_ratio": float((tier == "red").mean()),
        "yellow_ratio": float((tier == "yellow").mean()),
        "total_revenue": total,
        "platform_revenue": total * float(commission),
    }


def district_health(df: pd.DataFrame, commission: float) -> pd.DataFrame:
    """行政區健康度:房源數、平均空屋率、高風險占比、平台收入、vs 全市差異。"""
    cols = ["行政區", "房源數", "平均空屋率", "高風險占比",
            "預估平台收入", "空屋率vs全市"]
    if len(df) == 0:
        return pd.DataFrame(columns=cols)
    d = add_revenue_columns(df, commission)
    d["_vac"] = _num(d["vac_pred"])
    d["_red"] = (d["tier"].astype(str) == "red").astype(int)
    g = (d.groupby("neighbourhood_cleansed")
         .agg(房源數=("id", "size"),
              平均空屋率=("_vac", "mean"),
              高風險占比=("_red", "mean"),
              預估平台收入=("platform_revenue", "sum"))
         .reset_index()
         .rename(columns={"neighbourhood_cleansed": "行政區"}))
    g["空屋率vs全市"] = g["平均空屋率"] - float(d["_vac"].mean())
    return (g[cols].sort_values("高風險占比", ascending=False)
            .reset_index(drop=True))


def host_risk_summary(df: pd.DataFrame, commission: float) -> pd.DataFrame:
    """房東層級彙總:找出整批房源都在惡化的房東(高風險間數 → 占比 排序)。"""
    cols = ["host_id", "房源數", "高風險間數", "高風險占比",
            "平均風險分數", "預估年營收"]
    if len(df) == 0:
        return pd.DataFrame(columns=cols)
    d = add_revenue_columns(df, commission)
    d["_red"] = (d["tier"].astype(str) == "red").astype(int)
    d["_prob"] = _num(d["prob"])
    g = (d.groupby("host_id")
         .agg(房源數=("id", "size"),
              高風險間數=("_red", "sum"),
              高風險占比=("_red", "mean"),
              平均風險分數=("_prob", "mean"),
              預估年營收=("est_annual_revenue", "sum"))
         .reset_index())
    g["host_id"] = g["host_id"].astype(int)
    g["高風險間數"] = g["高風險間數"].astype(int)
    return (g[cols].sort_values(["高風險間數", "高風險占比"],
                                ascending=[False, False])
            .reset_index(drop=True))


def filter_scope(df: pd.DataFrame, districts=None, room_types=None) -> pd.DataFrame:
    """全域篩選;None 或空 list 代表該維度不篩選。"""
    out = df
    if districts:
        out = out[out["neighbourhood_cleansed"].isin(districts)]
    if room_types:
        out = out[out["room_type"].isin(room_types)]
    return out


def supply_demand_matrix(df: pd.DataFrame, min_listings: int = 15) -> pd.DataFrame:
    """行政區 x 房型 供需矩陣:需求強(空屋率低)且供給薄(房源少)= 招募缺口。"""
    cols = ["行政區", "房型", "房源數", "平均空屋率", "中位價格", "機會標籤"]
    if len(df) == 0:
        return pd.DataFrame(columns=cols)
    d = df.copy()
    d["_vac"] = _num(d["vac_pred"])
    d["_price"] = _num(d["price"])
    g = (d.groupby(["neighbourhood_cleansed", "room_type"])
         .agg(房源數=("id", "size"),
              平均空屋率=("_vac", "mean"),
              中位價格=("_price", "median"))
         .reset_index()
         .rename(columns={"neighbourhood_cleansed": "行政區",
                          "room_type": "房型"}))
    g = g[g["房源數"] >= int(min_listings)]
    if len(g) == 0:
        return pd.DataFrame(columns=cols)
    vac_mid = float(g["平均空屋率"].median())
    n_mid = float(g["房源數"].median())
    g["機會標籤"] = np.select(
        [(g["平均空屋率"] < vac_mid) & (g["房源數"] < n_mid),
         (g["平均空屋率"] > vac_mid) & (g["房源數"] > n_mid)],
        ["🟢 招募缺口", "🔴 供給飽和"], default="⚪ 一般")
    return g[cols].sort_values("平均空屋率").reset_index(drop=True)
