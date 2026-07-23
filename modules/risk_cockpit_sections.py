# -*- coding: utf-8 -*-
"""risk_cockpit_sections.py — 後台「🚨 風險管理」雙檢視渲染層。

房東檢視(排行榜/模糊搜尋)⇄ 房源檢視(獨立 checkbox 派信),麵包屑導覽。
純計算委由 platform_analytics;信件組裝/寄送/紀錄沿用 notify_center 公開介面。
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from modules import platform_analytics as pa
from modules.ui_components import P, html_table, note, sec

ROOM_ZH = {"Entire home/apt": "整棟出租", "Private room": "私人套房",
           "Shared room": "共用套房", "Hotel room": "飯店客房"}
TIER_ZH = {"red": "🔴 高風險", "yellow": "🟡 觀察", "green": "🟢 安全"}
HOST_ALL = "不限"                 # 房源檢視「房東ID」selectbox 的不限哨兵
LEADERBOARD_LIMIT = 100          # 房東檢視排行榜顯示上限
LISTING_LIMIT_DEFAULT = 100      # 房源檢視預設顯示筆數


def _money(v: float) -> str:
    """金額縮寫:億 / 萬(與 platform_sections 一致的顯示規則)。"""
    if abs(v) >= 1e8:
        return f"${v / 1e8:,.2f} 億"
    if abs(v) >= 1e4:
        return f"${v / 1e4:,.1f} 萬"
    return f"${v:,.0f}"


# ── 純邏輯(可 pytest,不依賴 Streamlit runtime)──────────────────
def resolve_host_filter(val, valid_ids) -> int | None:
    """把 rm_host_filter 的值正規化為 int 房東ID 或 None(哨兵/非法/不在母體)。"""
    if val == HOST_ALL or val is None:
        return None
    try:
        hid = int(val)
    except (ValueError, TypeError):
        return None
    return hid if hid in {int(x) for x in valid_ids} else None


def search_hosts(host_df: pd.DataFrame, query: str,
                 limit: int = LEADERBOARD_LIMIT) -> pd.DataFrame:
    """房東檢視:host_id 子字串模糊過濾(query 空=全部),取前 limit 位。
    host_df 需已依風險排序(host_risk_summary 輸出)。"""
    d = host_df
    q = (query or "").strip()
    if q:
        d = d[d["host_id"].astype(str).str.contains(q, regex=False)]
    return d.head(limit)


def filter_listings(df: pd.DataFrame, tiers, prob_lo: float, prob_hi: float,
                    host_filter: int | None) -> pd.DataFrame:
    """房源檢視:套用房東鎖定 + 警報層級 + 風險分數區間,依 prob 降序。"""
    d = df if host_filter is None else df[df["host_id"] == host_filter]
    chosen = tiers or ["red", "yellow", "green"]
    d = d[d["tier"].astype(str).isin(chosen)]
    prob = pd.to_numeric(d["prob"], errors="coerce").fillna(0)
    d = d[(prob >= prob_lo) & (prob <= prob_hi)]
    return d.sort_values("prob", ascending=False)
