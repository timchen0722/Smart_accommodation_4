# -*- coding: utf-8 -*-
"""absa_sections.py — 評論面向情感分析 UI 區塊

資料來源:scripts/build_absa.py 離線產出的三個小檔
(不在線上跑 NLP,以符合 Streamlit Cloud 記憶體限制)。

提供
----
render_listing_absa(listing_id)  房東入口:單一房源的面向強弱項
render_market_absa()             後台:全市/行政區面向痛點排行
listing_pain_points(listing_id)  供 LLM 建議引用的痛點清單(純資料,不繪圖)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from modules.ui_components import P, apply_theme, html_table, mb, note, sec

DATA = Path(__file__).resolve().parent.parent / "data"
LISTING_CSV = DATA / "_absa_listing.csv"
MARKET_CSV = DATA / "_absa_market.csv"
EXAMPLES_CSV = DATA / "_absa_examples.csv"

MIN_MENTIONS = 3          # 面向提及數低於此不下結論
ASPECT_TIP = {
    "清潔": "加強清潔 SOP 或改用專業清潔服務",
    "位置交通": "文案補充交通指引與接駁建議,或於定價讓利",
    "隔音噪音": "加裝氣密窗、門縫條;提供耳塞備品",
    "網路WiFi": "升級網速或加裝 Mesh 分享器,並在房源標示實測速度",
    "空調冷氣": "檢查冷氣效能與清洗濾網;夏季加強通風說明",
    "床與睡眠": "更換床墊/枕頭,提供軟硬枕選擇",
    "衛浴熱水": "檢修熱水器與水壓,加強浴室除霉",
    "房東服務": "縮短回覆時間,建立入住前主動關懷訊息範本",
    "性價比": "調整定價或增加超值備品(早餐包、飲用水)",
    "空間大小": "照片與文案誠實標示坪數,避免期待落差;優化收納",
    "設備廚房": "補齊常用家電與廚具,並在設施清單完整列出",
    "入住流程": "改用自助入住(密碼鎖),提供圖文入住指引",
    "早餐餐飲": "提供簡易早餐包或咖啡機;文案標示周邊早餐店與便利商店",
    "停車": "洽談鄰近停車場合作優惠;文案標示機車停放與收費停車場位置",
}


def available() -> bool:
    return LISTING_CSV.exists() and MARKET_CSV.exists()


@st.cache_data(show_spinner=False)
def load_listing_absa() -> pd.DataFrame:
    return pd.read_csv(LISTING_CSV)


@st.cache_data(show_spinner=False)
def load_market_absa() -> pd.DataFrame:
    return pd.read_csv(MARKET_CSV)


@st.cache_data(show_spinner=False)
def load_examples() -> pd.DataFrame:
    return pd.read_csv(EXAMPLES_CSV) if EXAMPLES_CSV.exists() else pd.DataFrame()


def listing_pain_points(listing_id: int, top_k: int = 3) -> list[dict]:
    """回傳該房源負評比例最高的前 K 個面向(供 LLM 建議與月報引用)。"""
    if not available():
        return []
    d = load_listing_absa()
    d = d[(d["listing_id"] == int(listing_id)) & (d["mentions"] >= MIN_MENTIONS)]
    if d.empty:
        return []
    d = d.sort_values("neg_ratio", ascending=False).head(top_k)
    return [{"aspect": r["aspect"], "neg_ratio": float(r["neg_ratio"]),
             "mentions": int(r["mentions"]),
             "tip": ASPECT_TIP.get(r["aspect"], "")}
            for _, r in d.iterrows() if r["neg_ratio"] > 0]


def render_listing_absa(listing_id: int, district: str | None = None):
    """房東入口:此房源住客在意什麼(強項 / 痛點)。"""
    if not available():
        st.info("尚未產生評論面向資料,請執行:python -X utf8 scripts/build_absa.py")
        return
    d = load_listing_absa()
    mine = d[d["listing_id"] == int(listing_id)]
    if mine.empty:
        st.info("此房源評論數不足,無法進行面向分析。")
        return
    mine = mine[mine["mentions"] >= MIN_MENTIONS].copy()
    if mine.empty:
        st.info("此房源各面向提及次數過少(<3),不下結論。")
        return

    # 市場基準(同行政區優先,否則全市)
    mk = load_market_absa()
    base = mk[(mk["scope"] == "行政區") & (mk["group"] == district)] \
        if district is not None else pd.DataFrame()
    if base.empty:
        base = mk[mk["scope"] == "全市"]
    base_map = dict(zip(base["aspect"], base["neg_ratio"]))

    mine["市場負評率"] = mine["aspect"].map(base_map)
    mine["差距"] = mine["neg_ratio"] - mine["市場負評率"]
    mine = mine.sort_values("neg_ratio", ascending=False)

    sec("💬 住客評論面向分析(ABSA)")
    mb("14 面向詞典 + 局部情感窗口 · 離線預計算 · 提及數 <3 不列入")

    # ── 評論自動摘要(規則版即時顯示;LLM 版手動觸發)──
    _render_summary(listing_id, district)

    c1, c2 = st.columns([1.3, 1])
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=mine["aspect"], x=mine["neg_ratio"] * 100, orientation="h",
            name="本房源負評率", marker_color=P["high"]))
        fig.add_trace(go.Scatter(
            y=mine["aspect"], x=mine["市場負評率"] * 100, mode="markers",
            name="市場基準", marker=dict(color=P["accent"], size=10,
                                         symbol="diamond")))
        apply_theme(fig, h=max(300, 30 * len(mine))).update_layout(
            xaxis_title="負評比例 (%)", yaxis=dict(autorange="reversed"),
            legend=dict(orientation="h", y=-0.18),
            margin=dict(l=90, r=20, t=10, b=50))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        worst = mine.nlargest(3, "差距")
        best = mine.nsmallest(2, "差距")
        for _, r in worst.iterrows():
            if r["差距"] > 0.01:
                note(f"⚠️ <b>{r['aspect']}</b>:負評率 {r['neg_ratio']:.0%}"
                     f"(市場 {r['市場負評率']:.0%},高 "
                     f"{r['差距']*100:.0f} 個百分點)<br>"
                     f"<span style='font-size:.78rem;'>建議:"
                     f"{ASPECT_TIP.get(r['aspect'], '針對此面向改善')}</span>")
        for _, r in best.iterrows():
            if r["差距"] < -0.01:
                note(f"✅ <b>{r['aspect']}</b>:負評率 {r['neg_ratio']:.0%},"
                     f"優於市場 {abs(r['差距'])*100:.0f} 個百分點 —— "
                     f"建議寫進房源標題與文案強化賣點。")

    show = mine.rename(columns={"aspect": "面向", "mentions": "提及則數",
                                "pos": "正評", "neg": "負評"})
    show["負評率"] = show["neg_ratio"].map("{:.0%}".format)
    show["市場基準"] = show["市場負評率"].map(
        lambda v: "—" if pd.isna(v) else f"{v:.0%}")
    html_table(show[["面向", "提及則數", "正評", "負評", "負評率", "市場基準"]],
               height=260)


def _render_summary(listing_id: int, district: str | None = None,
                    name: str = ""):
    """評論自動摘要區塊:規則版立即顯示,LLM 版由按鈕手動觸發。"""
    from modules import review_summary as rs

    reviews = None
    try:                                   # 關鍵字需要原始評論,取不到就只用 ABSA
        from modules.data_loader import load_reviews
        reviews = load_reviews()
    except Exception:
        pass

    @st.cache_data(show_spinner=False)
    def _facts(lid: int):
        return rs.collect_facts(lid, reviews)

    f = _facts(int(listing_id))
    store = st.session_state.setdefault("rev_sum_store", {})
    hit = store.get(int(listing_id))

    st.markdown(
        f"<div style='background:{P['mbg']};border:1px solid #C8DCF0;"
        f"border-radius:10px;padding:12px 16px;margin:6px 0 10px;"
        f"font-size:.87rem;line-height:1.85;color:{P['ink2']};'>"
        f"<b>📝 評論自動摘要</b>({'LLM 生成' if hit else '規則版'})<br>"
        f"{hit['text'] if hit else rs.rule_summary(f)}</div>",
        unsafe_allow_html=True)

    b1, b2 = st.columns([1, 3])
    if b1.button("🧠 用 LLM 重寫摘要", key=f"rsum_{listing_id}",
                 use_container_width=True):
        with st.spinner("LLM 生成評論摘要中 …"):
            try:
                prov, txt = rs.llm_summary(f, name, district or "")
                store[int(listing_id)] = {"prov": prov, "text": txt}
                st.rerun()
            except Exception as e:
                st.error(f"LLM 摘要失敗:{type(e).__name__} — {e}")
    if hit:
        b2.caption(f"由 {hit['prov']} 生成 · 再按一次可重新生成 · "
                   f"切換房源會回到規則版")


def render_market_absa():
    """後台:全市與行政區的面向痛點排行。"""
    if not available():
        st.warning("尚未產生評論面向資料,請先執行:")
        st.code("python -X utf8 scripts/build_absa.py")
        return
    mk = load_market_absa()
    city = mk[mk["scope"] == "全市"].sort_values("neg_ratio", ascending=False)

    sec("平台服務品質監管:全市評論面向分析")
    mb("以 24 萬則住客評論的面向情感分析,監控平台整體服務品質;"
       "負評率高且提及量大的面向,即為平台應優先要求改善的品質風險 · "
       "面向詞典 + 局部情感窗口(非深度模型,可離線重現)")
    k = st.columns(4)
    k[0].metric("分析評論則數", f"{int(city['mentions'].sum()):,} 次提及")
    k[1].metric("最大痛點", city.iloc[0]["aspect"],
                f"負評率 {city.iloc[0]['neg_ratio']:.0%}", delta_color="off")
    best = city.nsmallest(1, "neg_ratio").iloc[0]
    k[2].metric("最強優勢", best["aspect"],
                f"負評率 {best['neg_ratio']:.0%}", delta_color="off")
    k[3].metric("最常被談到", city.nlargest(1, "mentions").iloc[0]["aspect"],
                f"{int(city['mentions'].max()):,} 次", delta_color="off")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(city, x="neg_ratio", y="aspect", orientation="h",
                     color="neg_ratio",
                     color_continuous_scale=["#5B9E73", "#C49A4A", "#C4645A"],
                     text=city["neg_ratio"].map("{:.1%}".format))
        apply_theme(fig, h=430).update_layout(
            title="全市各面向負評率(越右越需改善)",
            xaxis_title="負評比例", yaxis_title="",
            yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.scatter(city, x="mentions", y="neg_ratio", size="mentions",
                         color="neg_ratio", text="aspect",
                         color_continuous_scale=["#5B9E73", "#C49A4A", "#C4645A"],
                         size_max=45)
        fig.update_traces(textposition="top center", textfont_size=10)
        fig.add_hline(y=city["neg_ratio"].median(), line_dash="dot",
                      line_color=P["muted"], annotation_text="負評率中位")
        apply_theme(fig, h=430).update_layout(
            title="提及熱度 × 負評率(右上=最該優先改善)",
            xaxis_title="提及次數", yaxis_title="負評比例",
            coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    note("<b>判讀</b>:右上象限(被大量談到、且負評率高)是全市共通痛點,"
         "代表市場普遍未解決的問題 —— 誰先改善誰就有差異化優勢。"
         "左上(少被談到但負評率高)則是特定房源的個別問題。")

    st.divider()
    sec("行政區面向熱力圖")
    dist = mk[mk["scope"] == "行政區"]
    piv = dist.pivot_table(index="group", columns="aspect", values="neg_ratio")
    fig = px.imshow(piv, text_auto=".0%", aspect="auto",
                    color_continuous_scale=["#5B9E73", "#F5F1EA", "#C4645A"],
                    labels={"color": "負評率"})
    apply_theme(fig, h=470).update_layout(
        title="行政區 × 面向 負評率(紅=該區此面向問題較多)",
        xaxis_title="", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    ex = load_examples()
    if len(ex):
        st.divider()
        sec("代表性負評例句(佐證用)")
        pick = st.selectbox("選擇面向", sorted(ex["aspect"].unique()))
        sub = ex[ex["aspect"] == pick].dropna(subset=["listing_id", "text"])
        sub = sub.nsmallest(6, "score")
        for _, r in sub.iterrows():
            st.markdown(
                f"<div style='background:{P['surface']};border-left:3px solid "
                f"{P['high']};padding:8px 12px;margin:6px 0;border-radius:0 8px 8px 0;"
                f"font-size:.82rem;color:{P['ink2']};'>"
                f"「{str(r['text'])}」<span style='color:{P['muted']};font-size:.72rem;'>"
                f" — 房源 #{int(r['listing_id'])}</span></div>",
                unsafe_allow_html=True)

    st.divider()
    note("<b>平台監管建議</b>:對負評率位居前段的面向,建議平台"
         "①針對該面向負評率前 10% 的房源發出品質改善要求並限期複查;"
         "②於房源刊登頁強制揭露該面向的實際條件(如是否有獨立空調、隔音狀況),"
         "降低期待落差;③將面向負評率納入排序權重,讓改善有實質誘因。"
         "可至「🚨 風險管理」分頁交叉比對這些房源是否同時具備空屋風險。")
