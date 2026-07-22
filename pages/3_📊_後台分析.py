# -*- coding: utf-8 -*-
"""後台分析 — Airbnb 平台營運後台

使用者角色:Airbnb 平台營運方(非房東)。五分頁對應四大工作情境 + 模型稽核:
  🏙 市場總覽 / 🚨 風險管理 / 💰 營收與成長 / 💬 品質監管 / 🧪 模型監控
設計依據:doc/07_平台後台改造設計.md
"""
import streamlit as st

from modules import platform_analytics as pa
from modules.platform_sections import (render_market_overview,
                                       render_risk_management)
from modules.ui_components import ROOM_JP, inject_css, sidebar_nav

st.set_page_config(page_title="後台分析 — Airbnb 平台營運後台", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")
inject_css()
st.markdown(
    "<style>.block-container,[data-testid='stMainBlockContainer']"
    "{padding-top:3.5rem !important;}</style>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _filter_options():
    """側欄篩選選項(行政區、房型)。"""
    from modules.feature_engineering import load_predictions
    d = load_predictions()
    if d is None:
        return [], []
    return (sorted(d["neighbourhood_cleansed"].dropna().unique().tolist()),
            sorted(d["room_type"].dropna().unique().tolist()))


DISTRICTS, ROOMS = _filter_options()

with st.sidebar:
    sidebar_nav()
    st.markdown("#### 🏢 Airbnb 平台營運後台")
    st.caption("全域篩選會套用到所有分頁")
    st.multiselect("行政區", DISTRICTS, default=[], key="pf_districts",
                   placeholder="不選＝全部行政區")
    st.multiselect("房型", ROOMS, default=[], key="pf_rooms",
                   format_func=lambda r: ROOM_JP.get(r, r),
                   placeholder="不選＝全部房型")
    st.divider()
    st.markdown("#### 💰 平台抽成率")
    st.slider("抽成率", pa.COMMISSION_MIN, pa.COMMISSION_MAX,
              pa.COMMISSION_DEFAULT, 0.01, format="%.0f%%",
              key="pf_commission",
              help="用於把房東端預估年營收換算為平台收入;調整後所有分頁金額即時連動")
    st.divider()
    st.caption("技術堆疊:LightGBM 主力 + XGBoost 對照(Isotonic 校準)· "
               "標籤 Y≥0.6 · 雙層警報(紅 0.60 / 黃 0.35)· "
               "GroupKFold(host_id) 5 折誠實驗證 · 59 特徵多模態")

st.markdown(
    "<div style='font-size:1.6rem;font-weight:800;margin-bottom:2px;'>"
    "📊 Airbnb 平台營運後台</div>"
    "<div style='color:#9A9490;font-size:.86rem;margin-bottom:14px;'>"
    "市場健康度監控 · 高風險房源與房東管理 · 平台營收與成長 · 服務品質監管"
    "</div>", unsafe_allow_html=True)

t_ov, t_risk, t_rev, t_qa, t_model = st.tabs(
    ["🏙 市場總覽", "🚨 風險管理", "💰 營收與成長", "💬 品質監管", "🧪 模型監控"])

with t_ov:
    render_market_overview()

with t_risk:
    render_risk_management()

with t_rev:
    from modules.portfolio_sections import render_portfolio_tab
    render_portfolio_tab()

with t_qa:
    from modules.absa_sections import render_market_absa
    render_market_absa()

with t_model:
    from modules.backend_v2_sections import (render_model_tab_v2,
                                             render_shap_tab_v2)
    from modules.portfolio_sections import render_forward_validation_tab
    from modules.ui_components import mb, note, sec
    from modules.vacancy_model import get_metrics

    sec("模型稽核:平台方如何確認 AI 風險評分可信")
    mb("本頁供平台方稽核風險評分的可信度;所有指標皆為 "
       "<b>GroupKFold(host_id) 5 折誠實驗證</b>(測試房東不出現在訓練集)")
    _m = get_metrics()
    _k = st.columns(4)
    _k[0].metric("分類 AUC", f"{_m['AUC']:.3f}")
    _k[1].metric("🔴 紅層 Precision", "0.69", "門檻 0.60", delta_color="off")
    _k[2].metric("🟡 黃層以上 Recall", "0.70", "門檻 0.35", delta_color="off")
    _k[3].metric("迴歸 R²", f"{_m['R2']:.3f}")
    note("解讀:<b>紅色警報</b>精確率約 69% —— 平台派專人輔導時,約七成是真的高風險,"
         "介入成本可控;<b>黃色觀察層</b>召回率約 70% —— 適合用於批次自動通知,"
         "能覆蓋多數高風險房源。迴歸 R² 偏低是空屋率本質難以精確外推,"
         "故平台決策一律以<b>分類分層</b>為準,連續空屋率僅作輔助顯示。")
    st.divider()

    m1, m2, m3 = st.tabs(["模型評估(LightGBM vs XGBoost)",
                          "前瞻驗證(真實未來)", "SHAP 可解釋性"])
    with m1:
        render_model_tab_v2()
    with m2:
        render_forward_validation_tab()
    with m3:
        render_shap_tab_v2()
