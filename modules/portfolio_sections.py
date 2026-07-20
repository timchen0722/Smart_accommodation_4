# -*- coding: utf-8 -*-
"""portfolio_sections.py — 後台分析新增分頁:房型獲利分析 / 前瞻驗證

獨立模組:pages/3_📊_後台分析.py 兩行接入,不影響既有沙盒與 SHAP 分頁。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from modules import calendar_analytics as ca
from modules.data_loader import load_listings
from modules.ui_components import P, apply_theme, html_table, mb, note, sec

MODELS = Path(__file__).resolve().parent.parent / "models"
ROOM_ZH = {"Entire home/apt": "整棟出租", "Private room": "私人套房",
           "Shared room": "共用套房", "Hotel room": "飯店客房"}


@st.cache_data(show_spinner="計算房型獲利 …")
def _portfolio() -> pd.DataFrame:
    d = ca.portfolio_summary(load_listings())
    d["房型"] = d["room_type"].map(ROOM_ZH).fillna(d["room_type"])
    return d


# ════════════════════════════════════════════════════════════════
# 分頁:房型獲利分析
# ════════════════════════════════════════════════════════════════
def render_portfolio_tab():
    if not ca.available():
        st.warning("尚未產生檔期資料,請先執行:")
        st.code("python -X utf8 scripts/build_calendar_features.py")
        return
    d = _portfolio()

    sec("房型獲利分析(以 calendar 真實已訂天數估算)")
    mb("年營收估算 = 每晚價格 × 未來 365 天真實已訂天數 · "
       "已排除全年封鎖與全年全空之異常房源")

    k = st.columns(4)
    k[0].metric("可分析房源", f"{len(d):,} 間")
    k[1].metric("平均已訂率", f"{d['booked_rate'].mean():.0%}")
    k[2].metric("年營收估算中位", f"${d['年營收估算'].median():,.0f}")
    _best = d.groupby("房型")["年營收估算"].median().idxmax()
    k[3].metric("最高獲利房型", _best, delta_color="off")

    c1, c2 = st.columns(2)
    with c1:
        g = (d.groupby("房型")
             .agg(中位年營收=("年營收估算", "median"),
                  平均已訂率=("booked_rate", "mean"),
                  中位價格=("price", "median"),
                  房源數=("id", "size")).reset_index()
             .sort_values("中位年營收", ascending=False))
        fig = px.bar(g, x="房型", y="中位年營收", color="平均已訂率",
                     color_continuous_scale=["#C4645A", "#C49A4A", "#5B9E73"],
                     text=g["中位年營收"].map("${:,.0f}".format))
        apply_theme(fig, h=330).update_layout(
            title="各房型中位年營收估算", yaxis_title="年營收估算 (NT$)")
        st.plotly_chart(fig, use_container_width=True)
        html_table(g.assign(
            中位年營收=g["中位年營收"].map("${:,.0f}".format),
            平均已訂率=g["平均已訂率"].map("{:.0%}".format),
            中位價格=g["中位價格"].map("${:,.0f}".format)), height=190)
    with c2:
        piv = (d.pivot_table(index="neighbourhood_cleansed", columns="房型",
                             values="年營收估算", aggfunc="median")
               .round(0))
        fig = px.imshow(piv, text_auto=".2s", aspect="auto",
                        color_continuous_scale=["#F5F1EA", "#C49A4A", "#5B9E73"],
                        labels={"color": "中位年營收"})
        apply_theme(fig, h=480).update_layout(
            title="行政區 × 房型 中位年營收估算矩陣",
            xaxis_title="", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    sec("投資決策視角:哪一區、哪種房型最值得投入")
    dd = (d.groupby(["neighbourhood_cleansed", "房型"])
          .agg(中位年營收=("年營收估算", "median"),
               平均已訂率=("booked_rate", "mean"),
               中位價格=("price", "median"),
               房源數=("id", "size")).reset_index())
    dd = dd[dd["房源數"] >= 15]
    fig = px.scatter(dd, x="中位價格", y="平均已訂率", size="房源數",
                     color="中位年營收", hover_name="neighbourhood_cleansed",
                     hover_data={"房型": True, "房源數": True},
                     color_continuous_scale=["#C4645A", "#C49A4A", "#5B9E73"],
                     size_max=38)
    fig.add_hline(y=dd["平均已訂率"].median(), line_dash="dot",
                  line_color=P["muted"], annotation_text="已訂率中位")
    fig.add_vline(x=dd["中位價格"].median(), line_dash="dot",
                  line_color=P["muted"], annotation_text="價格中位")
    apply_theme(fig, h=430).update_layout(
        title="價格 × 已訂率 定位圖(泡泡大小=房源數,顏色=年營收)",
        xaxis_title="中位每晚價格 (NT$)", yaxis_title="平均已訂率")
    st.plotly_chart(fig, use_container_width=True)
    note("<b>右上象限</b>(高價且高訂房率)= 最佳投資標的;<b>左上</b>(低價高訂房)= 薄利多銷型;"
         "<b>右下</b>(高價低訂房)= 定價過高或需求不足。"
         "泡泡越大代表該組合的房源越多、市場越成熟。")

    top = dd.nlargest(10, "中位年營收")
    html_table(top.assign(
        中位年營收=top["中位年營收"].map("${:,.0f}".format),
        平均已訂率=top["平均已訂率"].map("{:.0%}".format),
        中位價格=top["中位價格"].map("${:,.0f}".format)).rename(
        columns={"neighbourhood_cleansed": "行政區"}), height=290)
    note("⚠️ 此為<b>橫斷面估算</b>:不同房源的實際成本(租金、清潔、平台抽成)未納入,"
         "且『不可訂』含房東封鎖成分;請作為相對比較而非絕對獲利保證。")

    # ── 長短租策略分析 ──
    st.divider()
    render_tenure_strategy(d)


def render_tenure_strategy(d: pd.DataFrame):
    """長短租策略分析:依 calendar 的最低入住天數分群比較填充率與營收。"""
    m = ca.healthy_metrics()[["listing_id", "min_nights_median",
                              "min_nights_varies", "booked_rate",
                              "booked_days", "gap_longest_30d"]]
    j = d.merge(m, on="listing_id", how="inner", suffixes=("", "_m"))
    if j.empty:
        return

    def _seg(v):
        if pd.isna(v):
            return "未知"
        if v <= 1:
            return "① 單晚起租(1 晚)"
        if v <= 3:
            return "② 短租(2~3 晚)"
        if v <= 6:
            return "③ 中短租(4~6 晚)"
        if v < 28:
            return "④ 週租型(7~27 晚)"
        return "⑤ 長租型(≥28 晚)"

    j["租期策略"] = j["min_nights_median"].map(_seg)
    order = ["① 單晚起租(1 晚)", "② 短租(2~3 晚)", "③ 中短租(4~6 晚)",
             "④ 週租型(7~27 晚)", "⑤ 長租型(≥28 晚)"]
    g = (j[j["租期策略"].isin(order)]
         .groupby("租期策略")
         .agg(房源數=("listing_id", "size"),
              平均已訂率=("booked_rate", "mean"),
              平均已訂天數=("booked_days", "mean"),
              中位每晚價=("price", "median"),
              中位年營收=("年營收估算", "median"))
         .reindex(order).dropna(how="all").reset_index())

    sec("🗓 長短租策略分析(以 calendar 每日最低入住天數判定)")
    mb("市場實測:34% 房源的最低入住天數逐日變動;30 晚設定達 32.8 萬筆日資料,是短租長租化的訊號")

    k = st.columns(4)
    _lt = j[j["min_nights_median"] >= 28]
    k[0].metric("長租型房源(≥28 晚)", f"{len(_lt):,} 間",
                f"占 {len(_lt)/len(j):.0%}", delta_color="off")
    k[1].metric("長租型平均已訂率", f"{_lt['booked_rate'].mean():.0%}"
                if len(_lt) else "—")
    _st = j[j["min_nights_median"] <= 3]
    k[2].metric("短租型(≤3 晚)平均已訂率",
                f"{_st['booked_rate'].mean():.0%}" if len(_st) else "—")
    k[3].metric("採動態天數策略",
                f"{int(j['min_nights_varies'].sum()):,} 間",
                f"占 {j['min_nights_varies'].mean():.0%}", delta_color="off")

    c1, c2 = st.columns([1.25, 1])
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=g["租期策略"], y=g["平均已訂率"] * 100,
                             name="平均已訂率 (%)", marker_color=P["primary"],
                             text=(g["平均已訂率"] * 100).round(0),
                             textposition="outside"))
        fig.add_trace(go.Scatter(x=g["租期策略"], y=g["中位年營收"] /
                                 max(g["中位年營收"].max(), 1) * 100,
                                 name="中位年營收(相對比例)",
                                 mode="lines+markers",
                                 line=dict(color=P["accent"], width=2,
                                           dash="dot")))
        apply_theme(fig, h=350).update_layout(
            title="不同租期策略的檔期填充表現",
            yaxis_title="已訂率 (%) / 營收相對比例",
            legend=dict(orientation="h", y=-0.25),
            margin=dict(l=40, r=20, t=50, b=90))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        html_table(g.assign(
            平均已訂率=g["平均已訂率"].map("{:.0%}".format),
            平均已訂天數=g["平均已訂天數"].round(0),
            中位每晚價=g["中位每晚價"].map("${:,.0f}".format),
            中位年營收=g["中位年營收"].map("${:,.0f}".format)), height=250)

    if len(g) >= 2:
        best_rate = g.loc[g["平均已訂率"].idxmax()]
        best_rev = g.loc[g["中位年營收"].idxmax()]
        note(f"<b>填充率最高</b>:{best_rate['租期策略']}"
             f"(平均已訂率 {best_rate['平均已訂率']:.0%});"
             f"<b>營收最高</b>:{best_rev['租期策略']}"
             f"(中位年營收 ${best_rev['中位年營收']:,.0f})。"
             f"若兩者不同組,代表「訂得滿」與「賺得多」是不同策略 —— "
             f"高填充率常來自低價長住,單價與周轉需權衡。")
    note("⚠️ 長租型(≥28 晚)在台灣多為<b>規避短租法規</b>或轉向月租市場;"
         "其 calendar『不可訂』比例高不必然代表訂滿,也可能是房東封鎖短租日期。"
         "此分析呈現市場策略分布,不作為法規建議。")


# ════════════════════════════════════════════════════════════════
# 分頁:前瞻驗證(用真實未來資料檢驗模型)
# ════════════════════════════════════════════════════════════════
def render_forward_validation_tab():
    path = MODELS / "forward_validation.json"
    if not path.exists():
        st.warning("尚未產生前瞻驗證結果,請先執行:")
        st.code("python -X utf8 scripts/build_calendar_features.py")
        return
    fv = json.loads(path.read_text(encoding="utf-8"))
    if "error" in fv:
        st.warning(fv["error"])
        return

    sec("前瞻驗證:用真實未來資料檢驗模型")
    mb(f"特徵快照 {fv['listings_scraped']} → 真實結果 {fv['calendar_scraped']}"
       f"(相隔 {fv['gap_months']} 個月)· 可對照 {fv['n_matched']:,} 間房源")
    note("這不是交叉驗證,而是<b>真正的時間外推驗證</b>:模型只看得到 2025 年 9 月的特徵,"
         "而答案來自 9 個月後才爬取的 calendar。一般專題只能報交叉驗證分數,"
         "本平台能用真實未來資料檢驗 —— 誠實呈現衰退,比宣稱高分更有價值。")

    # ── 指標對照 ──
    rows = [
        {"指標": "模型 A 迴歸 R²", "交叉驗證(GroupKFold OOF)": 0.243,
         "真實未來(前瞻)": round(fv["reg_r2"], 3)},
        {"指標": "模型 B 分類 AUC", "交叉驗證(GroupKFold OOF)": 0.716,
         "真實未來(前瞻)": round(fv["clf_auc"], 3)},
        {"指標": "🔴 紅色門檻 Precision", "交叉驗證(GroupKFold OOF)": 0.69,
         "真實未來(前瞻)": round(fv["red"]["precision"], 2)},
        {"指標": "🔴 紅色門檻 Recall", "交叉驗證(GroupKFold OOF)": 0.27,
         "真實未來(前瞻)": round(fv["red"]["recall"], 2)},
        {"指標": "🟡 黃色以上 Recall", "交叉驗證(GroupKFold OOF)": 0.70,
         "真實未來(前瞻)": round(fv["yellow"]["recall"], 2)},
    ]
    df = pd.DataFrame(rows)
    df["衰退"] = (df["真實未來(前瞻)"] - df["交叉驗證(GroupKFold OOF)"]).round(3)

    c1, c2 = st.columns([1.1, 1])
    with c1:
        html_table(df, height=240)
        m1, m2, m3 = st.columns(3)
        m1.metric("真實高風險率", f"{fv['real_high_risk_rate']:.0%}")
        m2.metric("真實平均空屋率", f"{fv['real_vacancy_mean']:.0%}",
                  f"模型預測 {fv['pred_vacancy_mean']:.0%}", delta_color="off")
        m3.metric("前瞻 AUC", f"{fv['clf_auc']:.3f}",
                  f"vs OOF 0.716", delta_color="off")
    with c2:
        fig = go.Figure()
        cats = ["迴歸 R²", "分類 AUC", "紅層 Precision"]
        fig.add_trace(go.Bar(name="交叉驗證(OOF)", x=cats,
                             y=[0.243, 0.716, 0.69], marker_color=P["medium"],
                             text=["0.243", "0.716", "0.69"],
                             textposition="outside"))
        fig.add_trace(go.Bar(name="真實未來(前瞻)", x=cats,
                             y=[fv["reg_r2"], fv["clf_auc"],
                                fv["red"]["precision"]],
                             marker_color=P["primary"],
                             text=[f"{fv['reg_r2']:.3f}", f"{fv['clf_auc']:.3f}",
                                   f"{fv['red']['precision']:.2f}"],
                             textposition="outside"))
        apply_theme(fig, h=330).update_layout(
            barmode="group", yaxis_range=[0, 0.95],
            title="交叉驗證 vs 真實未來", margin=dict(l=40, r=20, t=50, b=30))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    sec("衰退原因分析與改善方向")
    cause = pd.DataFrame([
        {"原因": "① 時間跨度過長(9 個月)",
         "說明": "特徵取自 2025-09,答案是 2026-06 起的檔期;期間市場、定價、經營者皆已變動",
         "改善方向": "縮短預測視野至 1~3 個月,或改用滾動更新的特徵"},
        {"原因": "② 標籤定義被汙染",
         "說明": "calendar 的『不可訂』同時包含已預訂與房東主動封鎖,"
                 "使『真實空屋率』並非純粹的市場需求結果",
         "改善方向": "取得多期 calendar 快照,以狀態轉變(可訂→不可訂)辨識真實訂單"},
        {"原因": "③ 房源母體不同",
         "說明": "兩批爬取的房源集合有差異,僅 4,940 間重疊,"
                 "存續下來的房源本身即帶有存活偏誤",
         "改善方向": "以固定追蹤群組(panel)重新評估"},
        {"原因": "④ 迴歸目標本身難以外推",
         "說明": "連續空屋率受季節、事件影響大;分類(是否高風險)較穩健,"
                 "故 AUC 衰退幅度遠小於 R²",
         "改善方向": "對外仍以分類雙層警報為主,迴歸值只作參考"},
    ])
    html_table(cause, wrap=True, scroll=False)
    note("<b>結論</b>:分類模型(AUC 0.632)在 9 個月後仍具實用鑑別力,"
         "雙層警報的黃色層仍抓到 63% 的真實高風險房源;但迴歸的連續空屋率預測"
         "不應被當作精確數值使用。平台現行設計(以分類機率決定等級、迴歸僅作輔助顯示)"
         "與此驗證結果一致。")
