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


# ── 導覽狀態 callback(在 rerun 前寫 session_state,合法)──────────
def _clear_selection():
    for k in [k for k in st.session_state if str(k).startswith("rm_sel_")]:
        st.session_state[k] = False


def _go_hosts():
    st.session_state["rm_view"] = "hosts"
    st.session_state["rm_host_filter"] = HOST_ALL
    st.session_state["rm_expanded_id"] = None
    _clear_selection()


def _go_listings(host_id):
    st.session_state["rm_view"] = "listings"
    st.session_state["rm_host_filter"] = int(host_id)
    st.session_state["rm_expanded_id"] = None
    _clear_selection()


def _toggle_expand(lid):
    cur = st.session_state.get("rm_expanded_id")
    st.session_state["rm_expanded_id"] = None if cur == int(lid) else int(lid)


def _breadcrumb(view: str):
    """麵包屑;房源檢視顯示可點回的『房東檢視 › 房源檢視』。"""
    if view != "listings":
        st.markdown(
            f"<span style='color:{P['ink2']};font-weight:700;'>房東檢視</span>",
            unsafe_allow_html=True)
        return
    c = st.columns([1.2, 8])
    with c[0]:
        st.button("房東檢視", key="rm_bc_hosts", type="tertiary",
                  on_click=_go_hosts)
    with c[1]:
        st.markdown(
            f"<span style='color:{P['muted']};'>› </span>"
            f"<span style='color:{P['ink2']};font-weight:700;'>房源檢視</span>",
            unsafe_allow_html=True)


def render():
    """後台「🚨 風險管理」入口:依 rm_view 分流房東/房源檢視。"""
    from modules.platform_sections import guard_scope, commission
    df = guard_scope()
    if df is None:
        return
    cm = commission()

    sec("高風險房源與房東管理")
    view = st.session_state.setdefault("rm_view", "hosts")
    _breadcrumb(view)
    st.divider()
    if view == "listings":
        _render_listings(df, cm)
    else:
        _render_hosts(df, cm)


def _render_listings(df: pd.DataFrame, cm: float):
    st.info("(房源檢視於 Task 4 實作)")


def _render_hosts(df: pd.DataFrame, cm: float):
    """房東檢視:模糊查詢 + 可點房東ID排行榜(無勾選、無浮動列)。"""
    h = pa.host_risk_summary(df, cm)
    q = st.text_input("🔍 房東ID 模糊查詢", key="rm_host_search",
                      placeholder="輸入房東ID片段,如 123;留空看全部")
    res = search_hosts(h, q)
    _capped = "(僅顯示前 %d 位)" % LEADERBOARD_LIMIT \
        if len(h) > LEADERBOARD_LIMIT and len(res) >= LEADERBOARD_LIMIT else ""
    st.caption(f"搜尋結果:{len(res):,} 位房東 · "
               f"依「高風險間數 → 高風險占比」排序{_capped}")
    note("點<b>房東ID</b>(藍色連結)即可下鑽該房東名下房源清單並派信。")
    if not len(res):
        st.info("查無符合的房東,請調整查詢條件。")
        return

    widths = [0.6, 1.4, 0.9, 1.2, 1.0, 1.1, 1.3]
    hdr = st.columns(widths)
    for col, t in zip(hdr, ["排名", "房東ID", "房源數", "🔴高風險間數",
                            "高風險占比", "平均風險分數", "預估年營收"]):
        col.markdown(f"<span style='color:{P['muted']};font-size:.72rem;"
                     f"font-weight:700;'>{t}</span>", unsafe_allow_html=True)
    for rank, (_, r) in enumerate(res.iterrows(), 1):
        hid = int(r["host_id"])
        c = st.columns(widths)
        c[0].markdown(f"**{rank}**")
        c[1].button(f"#{hid} ▸", key=f"rm_host_{hid}", type="tertiary",
                    on_click=_go_listings, args=(hid,))
        c[2].markdown(f"{int(r['房源數'])}")
        c[3].markdown(f"{int(r['高風險間數'])}")
        c[4].markdown(f"{float(r['高風險占比']):.0%}")
        c[5].markdown(f"{float(r['平均風險分數']):.0%}")
        c[6].markdown(_money(float(r['預估年營收'])))
