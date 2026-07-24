# -*- coding: utf-8 -*-
"""portfolio_sections.py — 後台分析新增分頁:房型獲利分析 / 前瞻驗證

獨立模組:pages/3_📊_後台分析.py 兩行接入,不影響既有沙盒與 SHAP 分頁。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modules import calendar_analytics as ca
from modules.data_loader import load_listings
from modules.ui_components import (P, ROOM_JP, RTC, apply_theme, html_table,
                                   mb, note, overview_metric_card, sec)

MODELS = Path(__file__).resolve().parent.parent / "models"
ROOM_ZH = ROOM_JP          # 房型中譯改吃全站唯一來源(原本本檔複製一份)
# 圖表房型固定順序(照附圖:整棟→飯店→私人套房→共用)
ROOM_ORDER = ["整棟出租", "飯店客房", "私人套房", "共用套房"]


@st.cache_data(show_spinner="計算房型獲利 …")
def _portfolio() -> pd.DataFrame:
    d = ca.portfolio_summary(load_listings())
    d["房型"] = d["room_type"].map(ROOM_ZH).fillna(d["room_type"])
    return d


# ════════════════════════════════════════════════════════════════
# 分頁:房型獲利分析
# ════════════════════════════════════════════════════════════════
def _scoped_portfolio() -> pd.DataFrame:
    """calendar 房型獲利母體套用側欄『行政區 / 房型』篩選。"""
    d = _portfolio()
    dist = st.session_state.get("pf_districts")
    rooms = st.session_state.get("pf_rooms")
    if dist:
        d = d[d["neighbourhood_cleansed"].isin(dist)]
    if rooms:
        d = d[d["room_type"].isin(rooms)]
    return d


def _fmt_k(v) -> str:
    """金額縮寫成 $520k;NaN 回空字串。"""
    if pd.isna(v):
        return ""
    return f"${v / 1000:,.0f}k"


def _district_order(d: pd.DataFrame, col: str = "neighbourhood_cleansed"):
    """依房源數由多到少排序行政區(熱門在上)。"""
    return (d.groupby(col).size().sort_values(ascending=False).index.tolist())


# 區塊標題語意色(僅本分頁用;高對比、不改全站 .sec)
#   破題=藍 / 現況=琥珀 / 行動=綠,對應三段敘事節奏
SEC3_TONE = {
    "break": (P["primary"], "#EEF4FB"),   # 破題
    "now":   (P["medium"], "#F7F1E4"),    # 現況
    "act":   (P["low"], "#EAF3EE"),       # 行動
}


def _sec3(num: str, title: str, tag: str, tone: str = "break"):
    """精簡版專用的高對比區塊標題:序號徽章 + 粗體深色標題 + 語意 pill。

    以 inline style 產生單行 HTML(無空白行),不觸碰全站 `.sec` 樣式。
    """
    c, bg = SEC3_TONE.get(tone, SEC3_TONE["break"])
    st.markdown(
        "<div style='display:flex;align-items:center;flex-wrap:wrap;gap:9px;"
        "margin:26px 0 12px;'>"
        "<span style='display:inline-flex;align-items:center;justify-content:"
        f"center;width:26px;height:26px;border-radius:7px;background:{c};"
        f"color:#fff;font-weight:800;font-size:.9rem;flex:none;'>{num}</span>"
        f"<span style='font-size:1.08rem;font-weight:800;color:{P['ink']};"
        f"letter-spacing:.01em;'>{title}</span>"
        f"<span style='background:{bg};color:{c};font-size:.72rem;"
        "font-weight:700;padding:3px 11px;border-radius:999px;"
        f"border:1px solid {c}40;'>{tag}</span></div>",
        unsafe_allow_html=True)


def _tab_title(icon: str, title: str, subtitle: str):
    """分頁主標題:粗體深色 + 說明,取代 muted 小灰 sec()。"""
    st.markdown(
        f"<div style='margin:2px 0 2px;font-size:1.34rem;font-weight:800;"
        f"color:{P['ink']};letter-spacing:.01em;'>{icon} {title}</div>"
        f"<div style='color:{P['ink2']};font-size:.86rem;margin-bottom:10px;'>"
        f"{subtitle}</div>",
        unsafe_allow_html=True)


def render_portfolio_tab():
    if not ca.available():
        st.warning("尚未產生檔期資料,請先執行:")
        st.code("python -X utf8 scripts/build_calendar_features.py")
        return
    d = _scoped_portfolio()
    if len(d) == 0:
        st.warning("目前側欄篩選條件下沒有可分析房源,請放寬行政區 / 房型篩選。")
        return

    from modules.platform_sections import commission
    cm = commission()

    _tab_title("💰", "營收與成長(精簡版・3 圖)",
               "房型排行 → 熱力矩陣 → 成長機會,三種視覺語法講完整個故事;"
               "年營收估算 = 每晚價格 × 未來 365 天真實已訂天數")

    _rev_med = float(d["年營收估算"].median())
    k = st.columns(5)
    _best = d.groupby("房型")["年營收估算"].median().idxmax()
    with k[0]:
        overview_metric_card("可分析房源", f"{len(d):,} 間")
    with k[1]:
        overview_metric_card("平均已訂率", f"{d['booked_rate'].mean():.0%}")
    with k[2]:
        overview_metric_card("房東年營收中位", f"${_rev_med:,.0f}")
    with k[3]:
        overview_metric_card("平台預估年收入(中位)",
                             f"${_rev_med * cm:,.0f}",
                             f"抽成率 {cm:.0%}", accent_note=True)
    with k[4]:
        overview_metric_card("最高獲利房型", _best)

    # ── 圖1:房型獲利排行(破題) ──────────────────────────────
    _sec3("①", "房型獲利排行", "破題・哪個房型最賺", "break")
    g1 = (d.groupby("房型")["年營收估算"].median()
          .reindex(ROOM_ORDER).dropna().sort_values(ascending=True))
    fig1 = go.Figure(go.Bar(
        x=g1.values, y=g1.index, orientation="h",
        marker_color=[RTC.get(r, P["primary"]) for r in g1.index],
        text=[_fmt_k(v) for v in g1.values], textposition="outside",
        cliponaxis=False,
        hovertemplate="%{y}<br>中位年營收 %{x:$,.0f}<extra></extra>"))
    apply_theme(fig1, h=300, legend=False).update_layout(
        xaxis_title="中位年營收 (NT$)", yaxis_title="",
        margin=dict(l=20, r=70, t=20, b=30))
    st.plotly_chart(fig1, use_container_width=True)

    # ── 圖2:行政區 × 房型 獲利熱力矩陣(現況) ─────────────────
    _sec3("②", "行政區 × 房型 獲利熱力矩陣", "現況・錢在哪裡", "now")
    scope = st.radio("顯示口徑", ["房東端", "平台端"], horizontal=True,
                     key="rev_heat_scope",
                     help="房東端 = 中位年營收;平台端 = 中位年營收 × 抽成率")
    factor = cm if scope == "平台端" else 1.0
    piv = (d.pivot_table(index="neighbourhood_cleansed", columns="房型",
                         values="年營收估算", aggfunc="median")
           .reindex(index=_district_order(d),
                    columns=[c for c in ROOM_ORDER if c in d["房型"].unique()]))
    z = (piv * factor).values
    txt = [[_fmt_k(v) for v in row] for row in z]
    fig2 = go.Figure(go.Heatmap(
        z=z, x=list(piv.columns), y=list(piv.index),
        text=txt, texttemplate="%{text}",
        textfont=dict(size=13, color="#2A2A2A"),
        colorscale=["#F5F1EA", "#C49A4A", "#5B9E73"],
        hoverongaps=False, xgap=3, ygap=3,
        hovertemplate=("%{y} · %{x}<br>"
                       + ("平台" if scope == "平台端" else "房東")
                       + "中位年營收 %{z:$,.0f}<extra></extra>"),
        colorbar=dict(title=scope + "中位年營收")))
    apply_theme(fig2, h=90 + 48 * len(piv.index), legend=False).update_layout(
        xaxis_title="", yaxis=dict(autorange="reversed"),
        margin=dict(l=20, r=20, t=10, b=20))
    st.plotly_chart(fig2, use_container_width=True)

    # ── 圖3:成長機會供需矩陣(行動) ──────────────────────────
    render_growth_opportunity()

    # ── 底部:長短租一句話結論(不另做圖) ──────────────────────
    _tenure_oneliner(d)
    note("所有金額皆為<b>模型 / calendar 估算</b>,非實際金流;"
         "『不可訂』含房東封鎖成分,請作為相對比較而非絕對獲利保證。")


def _tenure_oneliner(d: pd.DataFrame):
    """用真實 calendar 資料算短租 vs 長租已訂率,輸出一行結論(不做圖)。"""
    try:
        m = ca.healthy_metrics()[["listing_id", "min_nights_median",
                                  "booked_rate"]]
        j = d.merge(m, on="listing_id", how="inner", suffixes=("", "_m"))
        _st = j[j["min_nights_median"] <= 3]["booked_rate"]
        _lt = j[j["min_nights_median"] >= 28]["booked_rate"]
        if len(_st) and len(_lt):
            note(f"📌 <b>長短租策略一句話結論</b>:短租型(≤3 晚)平均已訂率 "
                 f"<b>{_st.mean():.0%}</b>、周轉快;長租型(≥28 晚)平均已訂率 "
                 f"<b>{_lt.mean():.0%}</b>,單價穩但周轉慢,且需留意規避短租"
                 f"法規的訊號(詳見『數據分析』旁的合規提示)。不另做圖,"
                 f"避免雙軸圖表誤導。")
            return
    except Exception:
        pass
    note("📌 <b>長短租策略一句話結論</b>:短租型(≤3 晚)周轉快、已訂率較高;"
         "長租型(≥28 晚)單價穩但周轉慢,且需留意規避短租法規的訊號。"
         "不另做圖,避免雙軸圖表誤導。")


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


# ════════════════════════════════════════════════════════════════
# 區塊:成長機會(行政區 x 房型 供需缺口)
# ════════════════════════════════════════════════════════════════
# 供需狀態三級:現有兩級標籤 → 附圖用語 +(顏色碼,語意色)
STATUS_MAP = {
    "🟢 招募缺口": ("缺口市場（建議招募）", 2, "#7FB98E"),
    "⚪ 一般":     ("觀察中", 1, "#E3D5B0"),
    "🔴 供給飽和": ("已飽和", 0, "#D8D3CB"),
}
# 3 段離散色階(灰→米黃→綠),對應狀態碼 0 / 1 / 2
STATUS_SCALE = [[0.0, "#D8D3CB"], [0.333, "#D8D3CB"],
                [0.334, "#E3D5B0"], [0.666, "#E3D5B0"],
                [0.667, "#7FB98E"], [1.0, "#7FB98E"]]


def render_growth_opportunity():
    """圖3:行政區 × 房型 成長機會供需矩陣(缺口市場 / 觀察中 / 已飽和)。"""
    from modules import platform_analytics as pa
    from modules.platform_sections import ROOM_ZH as _RZ
    from modules.platform_sections import _money, commission, guard_scope

    _sec3("③", "成長機會供需矩陣", "行動・接下來去哪賺", "act")
    st.caption("需求強(空屋率低於中位)且供給薄(房源數低於中位)= 建議招募的缺口市場;"
               "反之為已飽和、不宜再增供給;點格子看 hover 明細")

    df = guard_scope()
    if df is None:
        return
    cm = commission()

    g = pa.supply_demand_matrix(df, min_listings=15)
    if len(g) == 0:
        st.info("篩選範圍內沒有房源數 ≥ 15 的『行政區 × 房型』組合,無法評估供需。")
        return
    g["房型中文"] = g["房型"].map(_RZ).fillna(g["房型"])
    g["狀態"] = g["機會標籤"].map(lambda s: STATUS_MAP.get(s, ("觀察中", 1, ""))[0])
    g["_code"] = g["機會標籤"].map(lambda s: STATUS_MAP.get(s, ("", 1, ""))[1])

    gap = g[g["機會標籤"] == "🟢 招募缺口"]
    sat = g[g["機會標籤"] == "🔴 供給飽和"]
    k = st.columns(3)
    k[0].metric("可評估組合", f"{len(g):,} 組")
    k[1].metric("🟢 缺口市場組合", f"{len(gap):,} 組")
    k[2].metric("⬜ 已飽和組合", f"{len(sat):,} 組")

    rows = _district_order(g, "行政區")
    cols = [c for c in ROOM_ORDER if c in g["房型中文"].unique()]

    def _piv(val):
        return (g.pivot_table(index="行政區", columns="房型中文", values=val,
                              aggfunc="first")
                .reindex(index=rows, columns=cols))

    z = _piv("_code").values
    p_vac, p_n, p_pr = _piv("平均空屋率"), _piv("房源數"), _piv("中位價格")
    p_lab = _piv("狀態")

    # customdata 全部預格式化為字串,避免 str/數值混型被 numpy 轉型後
    # 導致 hovertemplate 的數值格式碼失效
    txt, cd = [], []
    for i in range(len(rows)):
        trow, crow = [], []
        for j in range(len(cols)):
            v, n = p_vac.values[i][j], p_n.values[i][j]
            if pd.isna(v):
                trow.append("")
                crow.append(["—", "—", "—", "—"])
            else:
                trow.append(f"空屋 {v:.0%}<br>{int(n)} 間")
                crow.append([str(p_lab.values[i][j]), f"{v:.0%}",
                             f"{int(n):,}", f"${p_pr.values[i][j]:,.0f}"])
        txt.append(trow)
        cd.append(crow)

    fig = go.Figure(go.Heatmap(
        z=z, x=cols, y=rows, text=txt, texttemplate="%{text}",
        textfont=dict(size=12, color="#3A3A3A"),
        customdata=cd, zmin=0, zmax=2, colorscale=STATUS_SCALE,
        showscale=False, hoverongaps=False, xgap=3, ygap=3,
        hovertemplate=("%{y} · %{x}<br>狀態:%{customdata[0]}<br>"
                       "平均空屋率 %{customdata[1]}<br>"
                       "房源數 %{customdata[2]} 間<br>"
                       "中位價 %{customdata[3]}<extra></extra>")))
    apply_theme(fig, h=90 + 52 * len(rows), legend=False).update_layout(
        xaxis_title="", yaxis=dict(autorange="reversed"),
        margin=dict(l=20, r=20, t=10, b=20))
    st.plotly_chart(fig, use_container_width=True)

    _sq = ("<span style='display:inline-block;width:12px;height:12px;"
           "border-radius:2px;margin:0 4px 0 12px;background:{c};'></span>")
    st.markdown(
        _sq.format(c="#7FB98E") + "缺口市場(建議招募)"
        + _sq.format(c="#E3D5B0") + "觀察中"
        + _sq.format(c="#D8D3CB") + "已飽和",
        unsafe_allow_html=True)

    _rev = (pa.add_revenue_columns(df, cm)["platform_revenue"].sum())
    note(f"以目前抽成率 {cm:.0%} 估算,篩選範圍內平台年收入約 <b>{_money(_rev)}</b>;"
         f"缺口市場組合若各增加 10 間房源、以該組合中位表現計,可帶來的增量收入"
         f"應以此口徑另行試算。")
