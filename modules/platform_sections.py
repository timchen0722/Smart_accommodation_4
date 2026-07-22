# -*- coding: utf-8 -*-
"""platform_sections.py — Airbnb 平台方後台的渲染層。

計算全部委由 modules/platform_analytics.py(純 pandas,可測);
本檔只負責 Streamlit 版面、圖表與文案。
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from modules import platform_analytics as pa
from modules.feature_engineering import load_predictions
from modules.ui_components import P, apply_theme, html_table, mb, note, sec

ROOM_ZH = {"Entire home/apt": "整棟出租", "Private room": "私人套房",
           "Shared room": "共用套房", "Hotel room": "飯店客房"}


def commission() -> float:
    return float(st.session_state.get("pf_commission", pa.COMMISSION_DEFAULT))


@st.cache_data(show_spinner="載入平台母體 …")
def _base() -> pd.DataFrame:
    """全平台預測母體;load_predictions() 缺檔時回傳 None,此處轉為空表。"""
    d = load_predictions()
    return d if d is not None else pd.DataFrame()


def load_scope() -> pd.DataFrame:
    """全平台母體套用側欄全域篩選後的範圍。"""
    d = _base()
    if len(d) == 0:
        return d
    return pa.filter_scope(d,
                           st.session_state.get("pf_districts"),
                           st.session_state.get("pf_rooms"))


def guard_scope():
    """各分頁的前置守門:缺檔或篩選為空時渲染提示並回傳 None。"""
    if len(_base()) == 0:
        st.warning("尚未產生模型預測結果 data/_predictions.csv,請先執行:")
        st.code("python -X utf8 scripts/train_backend_models.py")
        return None
    df = load_scope()
    if len(df) == 0:
        st.warning("目前篩選條件下沒有房源,請放寬側欄的行政區/房型篩選。")
        return None
    return df


def _money(v: float) -> str:
    """金額縮寫:億 / 萬。"""
    if abs(v) >= 1e8:
        return f"${v / 1e8:,.2f} 億"
    if abs(v) >= 1e4:
        return f"${v / 1e4:,.1f} 萬"
    return f"${v:,.0f}"


# ════════════════════════════════════════════════════════════════
# 分頁:市場總覽
# ════════════════════════════════════════════════════════════════
def render_market_overview():
    df = guard_scope()
    if df is None:
        return
    cm = commission()

    k = pa.market_kpis(df, cm)
    sec("台北市 Airbnb 市場總覽")
    mb(f"母體 {k['n_listings']:,} 間房源 · 抽成率設定 {cm:.0%} · "
       f"所有金額皆為<b>模型預估</b>,非實際金流")

    c = st.columns(6)
    c[0].metric("總房源數", f"{k['n_listings']:,} 間")
    c[1].metric("活躍房東數", f"{k['n_hosts']:,} 位")
    c[2].metric("平均預估空屋率", f"{k['avg_vacancy']:.1%}")
    c[3].metric("🔴 高風險占比", f"{k['red_ratio']:.1%}",
                f"🟡 觀察層 {k['yellow_ratio']:.1%}", delta_color="off")
    c[4].metric("預估房東年營收總額", _money(k["total_revenue"]))
    c[5].metric("預估平台年收入", _money(k["platform_revenue"]),
                f"抽成 {cm:.0%}", delta_color="off")

    st.divider()
    sec("行政區健康度")
    dh = pa.district_health(df, cm)
    c1, c2 = st.columns([1.15, 1])
    with c1:
        show = dh.assign(
            平均空屋率=dh["平均空屋率"].map("{:.1%}".format),
            高風險占比=dh["高風險占比"].map("{:.1%}".format),
            預估平台收入=dh["預估平台收入"].map(_money),
            空屋率vs全市=dh["空屋率vs全市"].map(
                lambda v: f"{'▲' if v > 0 else '▼'} {abs(v):.1%}"))
        html_table(show, height=430)
        note("「空屋率vs全市」▲ 代表該區比全市平均更差、需優先關注;"
             "▼ 代表優於全市。點欄位標題可排序。")
    with c2:
        fig = px.bar(dh.head(12), x="高風險占比", y="行政區", orientation="h",
                     color="高風險占比",
                     color_continuous_scale=["#5B9E73", "#C49A4A", "#C4645A"],
                     text=dh.head(12)["高風險占比"].map("{:.0%}".format))
        apply_theme(fig, h=430).update_layout(
            title="各行政區高風險房源占比", xaxis_title="高風險占比",
            yaxis_title="", yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    sec("風險分布地圖")
    tier_zh = {"red": "🔴 高風險", "yellow": "🟡 觀察", "green": "🟢 安全"}
    m = df.copy()
    m["風險等級"] = m["tier"].astype(str).map(tier_zh).fillna("🟢 安全")
    c3, c4 = st.columns([1.6, 1])
    with c3:
        fig = px.scatter_map(
            m, lat="latitude", lon="longitude", color="風險等級",
            color_discrete_map={"🔴 高風險": P["high"], "🟡 觀察": P["medium"],
                                "🟢 安全": P["low"]},
            hover_name="neighbourhood_cleansed",
            hover_data={"price": ":,.0f", "vac_pred": ":.0%"},
            zoom=10.5, height=460, opacity=0.65)
        fig.update_layout(map_style="carto-positron",
                          margin=dict(l=0, r=0, t=0, b=0),
                          legend=dict(orientation="h", y=1.02))
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        cnt = (m["風險等級"].value_counts()
               .reindex(["🔴 高風險", "🟡 觀察", "🟢 安全"]).fillna(0)
               .reset_index())
        cnt.columns = ["風險等級", "房源數"]
        fig = px.bar(cnt, x="風險等級", y="房源數", color="風險等級",
                     color_discrete_map={"🔴 高風險": P["high"],
                                         "🟡 觀察": P["medium"],
                                         "🟢 安全": P["low"]},
                     text="房源數")
        apply_theme(fig, h=460, legend=False).update_layout(
            title="三層警報房源數", xaxis_title="", yaxis_title="房源數")
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# 分頁:風險管理
# ════════════════════════════════════════════════════════════════
TIER_ZH = {"red": "🔴 高風險", "yellow": "🟡 觀察", "green": "🟢 安全"}


def _lime_reasons(listing_id: int, top: int = 3) -> list:
    """單一房源的 Top-N 風險原因;回傳 [(中文特徵名, 百分點)]。"""
    from modules.vacancy_model import contributions, get_row
    row = get_row(int(listing_id))
    if row is None:
        return []
    return [(zh, dpp) for _f, zh, dpp in contributions(row, top=top)]


def render_risk_management():
    df = guard_scope()
    if df is None:
        return
    cm = commission()

    sec("高風險房源與房東管理")
    mb("由『發現高風險』到『介入輔導』一條龍:風險名單 → 房東彙總 → 派發輔導通知")

    t1, t2, t3 = st.tabs(["📋 房源風險名單", "👤 房東層級彙總", "✉️ 派發輔導通知"])

    # ── 房源風險名單 ──
    with t1:
        f1, f2, f3 = st.columns([1, 1, 1.4])
        tiers = f1.multiselect("警報層級", ["red", "yellow", "green"],
                               default=["red"],
                               format_func=lambda t: TIER_ZH[t],
                               key="rm_tiers")
        topn = f2.slider("顯示筆數", 20, 300, 100, 20, key="rm_topn")
        lo, hi = f3.slider("風險分數區間", 0.0, 1.0, (0.0, 1.0), 0.05,
                           key="rm_prob")
        d = df[df["tier"].astype(str).isin(tiers or ["red", "yellow", "green"])]
        d = d[(pd.to_numeric(d["prob"], errors="coerce").fillna(0) >= lo) &
              (pd.to_numeric(d["prob"], errors="coerce").fillna(0) <= hi)]
        d = pa.add_revenue_columns(d, cm).sort_values("prob", ascending=False)

        st.caption(f"符合條件 {len(d):,} 間,以下顯示風險分數最高的 {min(topn, len(d)):,} 間")
        show = d.head(topn)
        tbl = pd.DataFrame({
            "房源 ID": show["id"].astype(int),
            "行政區": show["neighbourhood_cleansed"],
            "房型": show["room_type"].map(ROOM_ZH).fillna(show["room_type"]),
            "每晚房價": pd.to_numeric(show["price"], errors="coerce")
                        .map("${:,.0f}".format),
            "風險分數": pd.to_numeric(show["prob"], errors="coerce")
                        .map("{:.0%}".format),
            "警報層級": show["tier"].astype(str).map(TIER_ZH),
            "預估空屋率": pd.to_numeric(show["vac_pred"], errors="coerce")
                          .map("{:.0%}".format),
            "房東 ID": show["host_id"].astype(int),
        })
        html_table(tbl, height=420)

        st.divider()
        sec("單一房源風險歸因")
        ids = show["id"].astype(int).tolist()
        if ids:
            pick = st.selectbox("選擇房源查看風險原因", ids,
                                format_func=lambda i: f"房源 #{i}",
                                key="rm_pick")
            with st.spinner("計算風險歸因 …"):
                reasons = _lime_reasons(int(pick), top=3)
            if reasons:
                for zh, dpp in reasons:
                    color = P["high"] if dpp > 0 else P["low"]
                    sign = "推高" if dpp > 0 else "降低"
                    st.markdown(
                        f"<div style='border-left:4px solid {color};"
                        f"background:{P['surface']};border-radius:0 8px 8px 0;"
                        f"padding:9px 14px;margin:6px 0;'>"
                        f"<b>{zh}</b> — {sign}空屋風險 "
                        f"<span style='color:{color};font-weight:700;'>"
                        f"{dpp:+.2f} 個百分點</span></div>",
                        unsafe_allow_html=True)
                note("以上為<b>基準對照邊際貢獻</b>:該特徵目前值相對全體中位數,"
                     "對高風險機率的加減分。可作為平台輔導房東時的溝通依據。")
            else:
                st.caption("此房源無足夠特徵可解釋。")

    # ── 房東層級彙總 ──
    with t2:
        h = pa.host_risk_summary(df, cm)
        k = st.columns(3)
        k[0].metric("涵蓋房東數", f"{len(h):,} 位")
        _bad = h[(h["高風險占比"] >= 0.5) & (h["房源數"] >= 2)]
        k[1].metric("整批惡化房東", f"{len(_bad):,} 位",
                    "名下 ≥2 間且過半高風險", delta_color="off")
        k[2].metric("這些房東的預估年營收", _money(float(_bad["預估年營收"].sum())))
        note("排序邏輯:<b>高風險間數</b>優先、其次<b>高風險占比</b> —— "
             "把「整批房源都在惡化」的房東排到最前面,平台應優先輔導。")

        top_h = h.head(50)
        html_table(top_h.assign(
            高風險占比=top_h["高風險占比"].map("{:.0%}".format),
            平均風險分數=top_h["平均風險分數"].map("{:.0%}".format),
            預估年營收=top_h["預估年營收"].map(_money)), height=400)

        st.divider()
        sec("房東下鑽:名下房源風險明細")
        hids = top_h["host_id"].astype(int).tolist()
        if hids:
            hpick = st.selectbox("選擇房東", hids,
                                 format_func=lambda i: f"房東 #{i}",
                                 key="rm_host")
            sub = pa.add_revenue_columns(
                df[df["host_id"] == hpick], cm).sort_values(
                "prob", ascending=False)
            html_table(pd.DataFrame({
                "房源 ID": sub["id"].astype(int),
                "行政區": sub["neighbourhood_cleansed"],
                "房型": sub["room_type"].map(ROOM_ZH).fillna(sub["room_type"]),
                "風險分數": pd.to_numeric(sub["prob"], errors="coerce")
                            .map("{:.0%}".format),
                "警報層級": sub["tier"].astype(str).map(TIER_ZH),
                "預估年營收": sub["est_annual_revenue"].map(_money),
            }), height=320)

    # ── 派發輔導通知 ──
    with t3:
        from modules.notify_center import render_notify_center
        note("以下為<b>平台營運方視角</b>的輔導通知作業:"
             "設定觸發門檻後,可批次自動寄送或逐筆手動寄出關懷輔導信件。")
        render_notify_center(host_id=None, key="pf_nc", platform_view=True)
