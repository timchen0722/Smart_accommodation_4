# -*- coding: utf-8 -*-
"""data_analysis_sections.py — 後台分析「數據分析」分頁（平鋪四段版）

一頁到底、不收合。四段回答「風險評分模型可不可信」：
  ① 信任成績單    ② 誠實雙軌對照    ③ 特徵怎麼篩選    ④ AI 在看什麼

資料唯一來源：models/eval_vacancy_90.json
  （scripts/04_eval/build_data_analysis_json.py 產出，App 不重算）
模型：HistGradientBoosting · 37 核心特徵 · 目標 Y_vacancy_90（雙輸出：90天風險 / 365天營收）
"""
import json
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go

from modules.ui_components import P, sec, note, apply_theme

EVAL_JSON = Path(__file__).resolve().parent.parent / "models" / "eval_vacancy_90.json"

# 由「頭部特徵」導出的可行動建議（key = 特徵欄名）；未列者以結構性訊號帶過
_ADVICE = {
    "host_acceptance_rate": ("房東接受率 · 可行動",
        "接受率越低、空屋風險越高。可提醒房東提高接受率與回覆速度，直接改善預訂轉換。"),
    "host_response_rate": ("房東回覆率 · 可行動",
        "回覆率是經營用心度訊號，偏低者可催促房東即時回覆訊息。"),
    "response_speed": ("房東回覆速度 · 可行動",
        "回覆越慢、轉換越差，適合作為「加快回覆」的輔導建議。"),
    "price_pctl_nbhd": ("同區價格百分位 · 定價",
        "定價落在商圈相對高位會推升空屋風險，可生成「對齊同區中位數」的定價建議。"),
    "price": ("每晚價格 · 定價",
        "價格是可直接調整的槓桿，配合同區分佈可做定價 what-if。"),
    "min_nights_avg_ntm": ("近期平均最短入住 · 入住策略",
        "近期設定的最短入住越長、周轉越慢，適合「調整最短入住」建議。"),
    "minimum_nights": ("最短入住晚數 · 入住策略",
        "最短入住過長會拉高空屋率，可建議下修門檻。"),
    "photo_design_sense": ("封面照設計感 · 素材",
        "封面照美感對點閱與訂房有實質效益，可建議更換高質感封面照。"),
    "self_checkin": ("自助入住 · 便利性",
        "自助入住提升便利與可訂性，可建議導入智慧鎖。"),
}
_STRUCTURAL = ("結構性訊號", "屬房型/規模等結構條件，較難短期調整，供市場定位與供需分析參考。")
_CARD_COLORS = None  # 於 render 時取用 P


@st.cache_data(show_spinner=False)
def _load():
    if not EVAL_JSON.exists():
        return None
    return json.loads(EVAL_JSON.read_text(encoding="utf-8"))


def _pct(x):
    return f"{x * 100:.1f}%"


def _model_card(tag, title, desc, color):
    return (
        f"<div style='background:{P['surface']};border:1px solid {P['border']};"
        f"border-left:4px solid {color};border-radius:12px;padding:14px 18px;"
        f"height:104px;box-sizing:border-box;display:flex;flex-direction:column;"
        f"justify-content:center;'>"
        f"<div style='font-size:.66rem;font-weight:700;letter-spacing:.08em;"
        f"color:{P['muted']};text-transform:uppercase;margin-bottom:4px;'>{tag}</div>"
        f"<div style='font-size:1rem;font-weight:800;color:{color};"
        f"margin-bottom:3px;'>{title}</div>"
        f"<div style='font-size:.74rem;color:{P['ink2']};line-height:1.4;'>{desc}</div>"
        f"</div>")


def _kpi(col, label, value, sub):
    col.markdown(
        f"<div class='overview-metric'>"
        f"<div class='overview-metric-label'>{label}</div>"
        f"<div class='overview-metric-value'>{value}</div>"
        f"<div style='font-size:.67rem;color:{P['muted']};margin-top:4px;"
        f"line-height:1.3;'>{sub}</div></div>", unsafe_allow_html=True)


def _funnel_box(big, small, sub, color, tint):
    return (
        f"<div style='flex:1 1 0;min-width:0;background:{tint};"
        f"border:1px solid {P['border']};border-top:3px solid {color};"
        f"border-radius:10px;padding:12px 10px;text-align:center;'>"
        f"<div style='font-size:1.5rem;font-weight:800;color:{color};"
        f"line-height:1.1;'>{big}</div>"
        f"<div style='font-size:.74rem;font-weight:700;color:{P['ink']};"
        f"margin-top:3px;'>{small}</div>"
        f"<div style='font-size:.64rem;color:{P['muted']};margin-top:2px;"
        f"line-height:1.3;'>{sub}</div></div>")


def _arrow():
    return (f"<div style='align-self:center;color:{P['muted']};font-size:1.1rem;"
            f"padding:0 2px;'>→</div>")


def _insight_card(title, body, color):
    return (
        f"<div style='background:{P['card']};border:1px solid {P['border']};"
        f"border-left:3px solid {color};border-radius:10px;padding:11px 14px;"
        f"height:100%;box-sizing:border-box;'>"
        f"<div style='font-size:.82rem;font-weight:800;color:{P['ink']};"
        f"margin-bottom:4px;'>{title}</div>"
        f"<div style='font-size:.75rem;color:{P['ink2']};line-height:1.5;'>{body}</div>"
        f"</div>")


def render_data_analysis():
    """後台分析『數據分析』分頁主渲染（平鋪四段）。"""
    d = _load()
    if d is None:
        st.warning("尚未產出 models/eval_vacancy_90.json，請先執行："
                   "`python -X utf8 scripts/04_eval/build_data_analysis_json.py`")
        return

    ss, gk = d["single_split"], d["groupkfold"]

    # ── 標題 ──
    st.markdown(
        f"<div style='font-size:1.18rem;font-weight:800;color:{P['ink']};"
        f"margin:4px 0 1px;'>📈 數據分析（{d['n_features']} 特徵最終版）</div>"
        f"<div style='color:{P['muted']};font-size:.84rem;margin-bottom:8px;'>"
        f"平台風險評分模型做過哪些數據分析、為什麼可信 —— 一頁到底看完</div>",
        unsafe_allow_html=True)

    # ════ ① 信任成績單 ════
    sec("① 信任成績單：模型是什麼、誠實表現如何")
    mc = st.columns(2)
    mc[0].markdown(_model_card(
        "模型 A · 迴歸", d["model"]["reg"],
        "輸出連續空屋率分數（雙輸出：90天→風險、365天→營收）",
        P["primary"]), unsafe_allow_html=True)
    mc[1].markdown(_model_card(
        "模型 B · 分類", d["model"]["clf"],
        f"判斷是否觸發高風險警報（{d['label_def_90']}）",
        P["accent"]), unsafe_allow_html=True)

    st.write("")
    k = st.columns(4)
    _kpi(k[0], "採用特徵數", str(d["n_features"]),
         f"從原始欄位精簡而來 · 共 {d['n_samples']:,} 筆")
    _kpi(k[1], "誠實 AUC（GroupKFold）", f"{gk['clf']['auc_90']:.3f}",
         f"± {gk['clf']['auc_90_std']:.3f} · 面對全新房東")
    _kpi(k[2], "誠實 R²（GroupKFold）", f"{gk['reg']['r2_90']:.3f}",
         f"± {gk['reg']['r2_90_std']:.3f} · 90 天空屋率")
    _kpi(k[3], "高風險基準率", _pct(d["base_rate_90"]),
         "vacancy_90 > 0.70（空置>63天）")

    st.markdown(
        f"<div style='background:#EAF5EE;border:1px solid #BFDCC9;border-radius:10px;"
        f"padding:10px 16px;margin:14px 0 2px;font-size:.82rem;color:#2F6B49;"
        f"line-height:1.5;'>✅ 這是平台正式採用的版本，所有風險評分都基於這 "
        f"{d['n_features']} 個特徵，模型為 <b>HistGradientBoosting</b>、"
        f"目標為 <b>未來 90 天空屋率</b>。知道極限在哪、也知道為什麼——"
        f"這是本專案最大的嚴謹度優勢。</div>", unsafe_allow_html=True)

    st.divider()

    # ════ ② 誠實雙軌對照 ════
    sec("② 誠實評估雙軌對照：主動抓出自己的漏洞")
    st.caption("用兩種切分方法驗證同一個模型，主動找出分數虛高的原因")
    cb1, cb2 = st.columns([1.25, 1])
    with cb1:
        cats = ["迴歸 R²", "分類 AUC"]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="單次切分（樂觀）", x=cats,
            y=[ss["reg"]["r2_90"], ss["clf"]["auc_90"]],
            marker_color=P["medium"],
            text=[f"{ss['reg']['r2_90']:.3f}", f"{ss['clf']['auc_90']:.3f}"],
            textposition="outside"))
        fig.add_trace(go.Bar(
            name="GroupKFold（誠實）", x=cats,
            y=[gk["reg"]["r2_90"], gk["clf"]["auc_90"]],
            error_y=dict(type="data",
                         array=[gk["reg"]["r2_90_std"], gk["clf"]["auc_90_std"]]),
            marker_color=P["primary"],
            text=[f"{gk['reg']['r2_90']:.3f}", f"{gk['clf']['auc_90']:.3f}"],
            textposition="outside"))
        apply_theme(fig, h=310).update_layout(
            barmode="group", yaxis_range=[0, 1.05],
            margin=dict(l=40, r=16, t=16, b=28))
        st.plotly_chart(fig, use_container_width=True)
    with cb2:
        note("同一房東名下常有多筆高度相似房源。<b>隨機切分</b>時，同一房東的房源會"
             "同時落在訓練與測試集，模型等於偷看了答案，分數虛高（AUC "
             f"{ss['clf']['auc_90']:.3f}、R² {ss['reg']['r2_90']:.3f}）。"
             "<b>GroupKFold 依 host_id 分組</b>，確保測試房東完全沒出現過——"
             f"誠實 AUC 掉到 {gk['clf']['auc_90']:.3f}、R² {gk['reg']['r2_90']:.3f}，"
             "這才是正式採用的基準。")
        note(f"<b>90天 vs 365天 雙輸出</b>：同一組 {d['n_features']} 特徵另訓一個"
             f"365天空屋率回歸器供營收估算。誠實對照——90天 AUC "
             f"{gk['clf']['auc_90']:.3f} / 365天 {gk['clf']['auc_365']:.3f}；"
             f"90天 R² {gk['reg']['r2_90']:.3f} / 365天 {gk['reg']['r2_365']:.3f}。"
             "90天更難預測但更即時可行動，365天較平滑、適合年營收換算。")

    st.divider()

    # ════ ③ 特徵怎麼篩選 ════
    sec("③ 特徵怎麼篩選出來的：不是隨便湊的")
    st.caption("從原始欄位一路測試、排除、驗證，最後留下 37 個核心特徵")
    st.markdown(
        f"<div style='display:flex;gap:6px;margin:6px 0 8px;'>"
        f"{_funnel_box('81', '原始欄位', 'Inside Airbnb 全欄', P['muted'], P['tag_bg'])}"
        f"{_arrow()}"
        f"{_funnel_box('−44', '排除', '洩漏／無效／未測試', P['ink2'], P['surface'])}"
        f"{_arrow()}"
        f"{_funnel_box('＋衍生', '特徵工程', '百分位／密度／經營天數', P['medium'], '#FBF6EA')}"
        f"{_arrow()}"
        f"{_funnel_box('37', '最終採用', '正式進模型', P['low'], '#EAF5EE')}"
        f"</div>", unsafe_allow_html=True)
    note("每一個留下來的特徵都經過測試驗證，不是原始資料有什麼就全部塞進去。"
         "特徵選擇實驗（前向選擇，以 365 天目標）顯示：約 <b>11~13 個特徵</b>即逼近"
         "誠實 R² 天花板（≈0.26），第 27 個特徵才達峰值 0.2627——增益早已是噪音。"
         "採用 37 個是兼顧穩健性與冷啟動可維護性下的合理選擇，不是隨便湊的數字。")
    # 37 特徵一覽（chips）
    chips = "".join(
        f"<span style='display:inline-block;background:{P['tag_bg']};"
        f"border:1px solid {P['border']};border-radius:6px;padding:2px 9px;"
        f"margin:3px 4px 0 0;font-size:.72rem;color:{P['ink2']};'>{f['zh']}</span>"
        for f in d["features"])
    st.markdown(
        f"<div style='margin-top:4px;'><span style='font-size:.7rem;font-weight:700;"
        f"color:{P['muted']};letter-spacing:.06em;'>37 特徵一覽</span><br>{chips}</div>",
        unsafe_allow_html=True)

    st.divider()

    # ════ ④ AI 在看什麼 ════
    sec("④ AI 在看什麼、能給什麼建議：不是黑箱")
    st.caption("每個判斷都能拆解成人看得懂的理由（Permutation Importance · 90 天模型）")
    imp = d["importance_90"]
    ci1, ci2 = st.columns([1.15, 1])
    with ci1:
        ys = [x["zh"] for x in imp][::-1]
        xs = [x["value"] for x in imp][::-1]
        fig = go.Figure(go.Bar(
            x=xs, y=ys, orientation="h", marker_color=P["primary"],
            text=[f"{v:+.3f}" for v in xs], textposition="outside"))
        apply_theme(fig, h=360, legend=False).update_layout(
            margin=dict(l=8, r=40, t=10, b=24),
            xaxis_title="重要度（打亂該特徵造成的 R² 損失）")
        st.plotly_chart(fig, use_container_width=True)
    with ci2:
        # 依「實際頭部特徵」動態產生可行動建議，確保與左側重要度圖一致
        palette = [P["primary"], P["accent"], P["low"]]
        cards, used = [], 0
        for x in imp:
            if used >= 3:
                break
            title, body = _ADVICE.get(x["key"], (f"{x['zh']} · {_STRUCTURAL[0]}",
                                                  _STRUCTURAL[1]))
            cards.append(_insight_card(f"{title}（重要度 {x['value']:+.3f}）",
                                       body, palette[used]))
            used += 1
        st.markdown("<div style='display:grid;gap:8px;'>" + "".join(cards)
                    + "</div>", unsafe_allow_html=True)
    note("以上為打亂單一特徵後模型 R² 的損失（permutation importance），"
         "值越大代表模型越依賴該特徵。這讓每筆風險評分都能回溯到人看得懂的原因，"
         "而非黑箱輸出。")
