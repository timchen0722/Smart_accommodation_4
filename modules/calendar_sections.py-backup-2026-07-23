# -*- coding: utf-8 -*-
"""calendar_sections.py — 未來檔期分頁的 UI 區塊(房東入口用)

獨立模組:pages/1_🏠_房東入口.py 只需兩行接入,不影響既有四視圖與通知中心。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modules import calendar_analytics as ca
from modules.ui_components import P, apply_theme, html_table, mb, note, sec

DOW_ZH = ["一", "二", "三", "四", "五", "六", "日"]


def _heatmap(d: pd.DataFrame, title: str):
    """逐日訂房狀態日曆熱度圖(橫軸=星期、縱軸=週)。"""
    piv = d.pivot_table(index="week", columns="dow", values="booked",
                        aggfunc="first")
    txt = d.pivot_table(index="week", columns="dow", values="date",
                        aggfunc="first").map(
        lambda x: x.strftime("%m/%d") if pd.notna(x) else "")
    fig = go.Figure(go.Heatmap(
        z=piv.values, x=[f"週{DOW_ZH[c]}" for c in piv.columns],
        y=[f"第{int(i)+1}週" for i in piv.index],
        text=txt.values, texttemplate="%{text}",
        textfont=dict(size=9),
        colorscale=[[0, "#EAF5EE"], [0.5, "#EAF5EE"],
                    [0.5, P["high"]], [1, P["high"]]],
        showscale=False, xgap=2, ygap=2,
        hovertemplate="%{text}<br>%{z}<extra></extra>"))
    apply_theme(fig, h=max(260, 26 * len(piv))).update_layout(
        title=title, yaxis=dict(autorange="reversed"),
        margin=dict(l=60, r=10, t=40, b=10))
    return fig


def render_calendar_tab(listing_id: int, listing_row, listings_df):
    """房東入口「📅 未來檔期」分頁主體。

    參數:listing_id 房源 id、listing_row 該房源基本資料(Series)、
          listings_df 全量 listings(供同儕營收曲線)。
    """
    if not ca.available():
        st.warning("尚未產生檔期資料,請先執行:")
        st.code("python -X utf8 scripts/build_calendar_features.py")
        return

    row = ca.get_listing(listing_id)
    if row is None:
        st.info("此房源不在 calendar 資料範圍內(calendar 與 listings 為不同批次爬取,"
                "重疊約 4,940 間)。請改選其他房源。")
        return
    if row["is_all_blocked"]:
        note("⚠️ 此房源未來 365 天<b>全部不可訂</b> —— 可能已停業、轉長租或房東封鎖日曆,"
             "以下檔期分析不具參考價值。")
    if row["is_all_open"]:
        note("⚠️ 此房源未來 365 天<b>完全沒有訂單</b>,請優先檢視右側風險診斷與定價建議。")

    district = listing_row["neighbourhood_cleansed"]
    room_type = listing_row["room_type"]

    # ── KPI ──
    sec("未來檔期總覽")
    mb("資料來源:Inside Airbnb calendar(2026-06-30 爬取,未來 365 天真實訂房狀態)")
    k = st.columns(5)
    k[0].metric("未來 365 天已訂率", f"{row['booked_rate']:.0%}",
                f"{int(row['booked_days'])} 天", delta_color="off")
    for i, (tag, lbl) in enumerate([("d30", "未來 30 天"), ("d60", "未來 60 天"),
                                    ("d90", "未來 90 天")], start=1):
        v = row.get(f"booked_rate_{tag}")
        k[i].metric(f"{lbl}已訂率", "—" if pd.isna(v) else f"{v:.0%}")
    k[4].metric("90 天內空檔", f"{int(row['gap_days_90d'])} 天",
                f"{int(row['gap_count_90d'])} 段·最長 {int(row['gap_longest_90d'])} 天",
                delta_color="off")

    c1, c2 = st.columns([1.35, 1], gap="medium")

    # ── 日曆熱度 ──
    with c1:
        sec("逐日訂房狀態(紅=已訂/不可訂,綠=空房可訂)")
        d = ca.daily_frame(row)
        rng = st.radio("顯示範圍", ["未來 90 天", "未來 180 天", "全年 365 天"],
                       horizontal=True, key="cal_rng")
        n = {"未來 90 天": 90, "未來 180 天": 180, "全年 365 天": 365}[rng]
        st.plotly_chart(_heatmap(d[d["horizon"] < n], ""),
                        use_container_width=True)

    # ── 月度 vs 同商圈 ──
    with c2:
        sec("未來 12 個月:本房源 vs 同商圈")
        mv = ca.monthly_vs_market(row, district, room_type)
        if len(mv):
            fig = go.Figure()
            fig.add_trace(go.Bar(x=mv["月份"], y=mv["本房源"] * 100,
                                 name="本房源", marker_color=P["primary"]))
            fig.add_trace(go.Scatter(x=mv["月份"], y=mv["同商圈基準"] * 100,
                                     name=f"{district}·同房型基準", mode="lines+markers",
                                     line=dict(color=P["accent"], width=2,
                                               dash="dot")))
            apply_theme(fig, h=300).update_layout(
                yaxis_title="已訂率 (%)", margin=dict(l=40, r=10, t=10, b=60),
                legend=dict(orientation="h", y=-0.35))
            st.plotly_chart(fig, use_container_width=True)
            worst = mv.nsmallest(1, "差距").iloc[0]
            if worst["差距"] < -0.05:
                note(f"📉 <b>{worst['月份']}</b> 訂房率 {worst['本房源']:.0%},"
                     f"低於同商圈基準 {abs(worst['差距'])*100:.0f} 個百分點 —— "
                     f"建議優先針對此月促銷或調價。")
        else:
            st.caption("無足夠同商圈樣本可比較。")

    # ── 空檔警示 ──
    sec("空檔警示(未來 90 天內連續 5 天以上無訂單)")
    gaps = ca.gap_segments(row, min_len=5, horizon=90)
    if len(gaps):
        g = gaps.copy()
        g["起日"] = g["起日"].dt.strftime("%Y-%m-%d")
        g["迄日"] = g["迄日"].dt.strftime("%Y-%m-%d")
        g["建議"] = g["連續天數"].map(
            lambda n: ("🔴 長空檔:建議大幅折扣或開放長租" if n >= 21
                       else ("🟡 中空檔:建議限時折扣 10~15%" if n >= 10
                             else "🟢 短空檔:可設最後一分鐘折扣")))
        html_table(g[["起日", "迄日", "連續天數", "建議"]], height=240)
        note(f"共 <b>{int(row['gap_days_90d'])}</b> 天空檔待填補;"
             f"若以每晚 ${float(listing_row['price']):,.0f} 計,"
             f"填滿可增加約 <b>${float(listing_row['price']) * row['gap_days_90d']:,.0f}</b> 營收。")
    else:
        st.success("未來 90 天內無連續 5 天以上的空檔 🎉")

    # ── 營收最適定價 ──
    sec("💰 營收最適定價(以同商圈同房型的真實已訂天數估算)")
    mb("營收估算 = 每晚價格 × 真實已訂天數 · 已訂天數取自 calendar,與價格為獨立資料源")
    curve = ca.peer_revenue_curve(listings_df, district, room_type)
    if curve.empty:
        st.caption("同商圈同房型樣本不足,無法建立營收曲線。")
    else:
        opt = ca.optimal_price(curve)
        cur_price = float(pd.to_numeric(
            str(listing_row["price"]).replace("$", "").replace(",", ""),
            errors="coerce") or 0)
        cc1, cc2 = st.columns([1.4, 1])
        with cc1:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=curve["價格中位"], y=curve["年營收估算"],
                                 name="年營收估算", marker_color=P["primary"],
                                 opacity=.75))
            fig.add_trace(go.Scatter(x=curve["價格中位"],
                                     y=curve["已訂率"] * curve["年營收估算"].max(),
                                     name="已訂率(右軸比例)", mode="lines+markers",
                                     line=dict(color=P["medium"], width=2,
                                               dash="dot")))
            fig.add_vline(x=cur_price, line_dash="dot", line_color=P["high"],
                          annotation_text=f"目前 ${cur_price:,.0f}")
            if opt:
                fig.add_vline(x=opt["price"], line_dash="dash",
                              line_color=P["low"],
                              annotation_text=f"最適 ${opt['price']:,.0f}")
            apply_theme(fig, h=300).update_layout(
                xaxis_title="每晚價格 (NT$)", yaxis_title="年營收估算 (NT$)",
                legend=dict(orientation="h", y=-0.3))
            st.plotly_chart(fig, use_container_width=True)
        with cc2:
            if opt:
                _gap = opt["price"] - cur_price
                _dir = "調高" if _gap > 0 else "調降"
                st.metric("建議價格帶", f"${opt['price']:,.0f}",
                          f"{_dir} ${abs(_gap):,.0f}" if abs(_gap) > 50 else "已接近最適",
                          delta_color="off")
                st.metric("該價格帶年營收估算", f"${opt['revenue']:,.0f}",
                          f"平均已訂 {opt['booked_days']:.0f} 天 · {opt['n']} 筆同儕",
                          delta_color="off")
            html_table(curve.assign(
                價格中位=curve["價格中位"].map("${:,.0f}".format),
                已訂天數=curve["已訂天數"].round(0),
                已訂率=curve["已訂率"].map("{:.0%}".format),
                年營收估算=curve["年營收估算"].map("${:,.0f}".format))[
                ["價格中位", "已訂天數", "已訂率", "年營收估算", "樣本數"]],
                height=230)
        note("⚠️ <b>誠實限制</b>:此為<b>橫斷面</b>推論(同商圈不同房源在不同價位的實際表現),"
             "非同一間房調價的因果效應;真正的價格彈性需 A/B 調價實驗。"
             "另 Inside Airbnb 的『不可訂』同時包含已預訂與房東封鎖,已排除全年封鎖/全空房源。")
