# -*- coding: utf-8 -*-
"""platform_analytics.py — Airbnb 平台方後台的純計算層。

刻意不 import streamlit:所有函式皆為 DataFrame in / DataFrame(或 dict) out,
可離線用 pytest 驗證;快取由呼叫端(platform_sections)負責。

營收口徑(doc/07):
  預估年營收 = price x (1 - vac_pred) x 365
  平台收入   = 預估年營收 x 抽成率
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
    """回傳新 DataFrame,附加 est_annual_revenue 與 platform_revenue。"""
    out = df.copy()
    price = _num(out["price"])
    occ = (1.0 - _num(out["vac_pred"])).clip(0.0, 1.0)
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
