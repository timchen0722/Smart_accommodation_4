# -*- coding: utf-8 -*-
"""房東入口 — 房東營運面板

分頁順序即使用順序,四項主打在前、模型診斷在後:
  房源總表   多房源優先序(體質 × 檔期四象限)· 空檔天數與機會成本
  定價情報   1km 內四平台每人每晚落點 · 同商圈同房型真實已訂率基準 · 周邊地圖
  口碑情報   評論 14 面向負評率 vs 同區基準 · 優先改善清單
  未來檔期   逐日訂房熱度 · 月度 vs 同商圈 · 空檔警示 · 營收最適定價
  風險診斷   模型分數 · LIME 原因 · What-if(前瞻 AUC 0.632,僅供輔助)
  通知中心   風險或空檔觸發 · 自動/手動寄送 · 已處理狀態
  月報       彙整以上為可下載報告
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from google import genai
from google.genai.errors import APIError

from modules.ui_components import (inject_css, P, ROOM_JP, sec, mb, note,
                                   html_table, apply_theme, sidebar_nav)
from modules.data_loader import load_listings
from modules import feature_engineering as fe
from modules.feature_engineering import (predict_risk_v2, simulate_price_change,
                                         load_predictions)
from modules.image_analysis import fake_host_email
from modules.market_data import capacity_bracket, _canon_amenities
from modules.pkl_store import load_module as _load_pkl

st.set_page_config(page_title="房東入口 — 智慧旅宿", page_icon="🏠",
                   layout="wide", initial_sidebar_state="expanded")
inject_css()

TIER_ZH = {"red": ("🔴 高風險", P["high"]), "yellow": ("🟡 觀察", P["medium"]),
           "green": ("🟢 安全", P["low"])}
_PLAT_COLORS = {"Airbnb": "#FF5A5F", "Booking": "#4E7FB0",
                "591": "#C49A4A", "ddroom": "#8B7BA8"}


# ─── 資料載入(快取) ─────────────────────────────────────────
@st.cache_data(show_spinner="載入房源與預測資料 …")
def load_all():
    df = load_listings()
    preds = load_predictions()
    ds = fe.load_dataset_final()
    meta = df[["id", "name", "picture_url", "host_name", "listing_url",
               "description", "amenities", "minimum_nights"]].copy()
    preds = preds.merge(meta, on="id", how="left")
    dist_med = preds.groupby("neighbourhood_cleansed")["vac_pred"].median()
    preds["dist_med"] = preds["neighbourhood_cleansed"].map(dist_med)
    return preds, ds, df


@st.cache_resource(show_spinner=False)
def get_bundle():
    return fe.load_bundle()


@st.cache_resource(show_spinner="載入跨平台競品索引 …")
def comp_index():
    return _load_pkl("competitor_index")


@st.cache_resource(show_spinner=False)
def sugg_engine():
    return _load_pkl("suggestion_engine")


PREDS, DS, DF_RAW = load_all()
# ── 體質(模型) × 檔期(calendar)四象限分類 ──
from modules import quadrant as QD
PREDS = QD.annotate(QD.attach_calendar(PREDS), tier_col="tier")
BUNDLE = get_bundle()
DS_IDX = DS.set_index("id")

# ─── 側邊欄 ───────────────────────────────────────────────────
with st.sidebar:
    sidebar_nav()
    st.markdown("#### 🎯 請選擇登入的房東")
    _hc = (PREDS.groupby(["host_id"])
           .agg(n=("id", "size"), host_name=("host_name", "first"),
                red=("tier", lambda s: int((s == "red").sum())))
           .sort_values("n", ascending=False).reset_index())
    _hlab = _hc.apply(lambda r: f"{r['host_name'] or '房東'}"
                                f"(ID {int(r['host_id'])}·{int(r['n'])} 間"
                                f"{'·🔴' + str(r['red']) if r['red'] else ''})",
                      axis=1)
    _hi = st.selectbox("host", range(len(_hc)), format_func=lambda i: _hlab[i],
                       label_visibility="collapsed")
    host_id = int(_hc.iloc[_hi]["host_id"])
    MY = PREDS[PREDS["host_id"] == host_id].reset_index(drop=True)

    st.markdown("#### 🧠 風險分類模型")
    _ap = st.radio("algo", ["LightGBM（主力）", "XGBoost（對照）"],
                   label_visibility="collapsed")
    ALGO = "xgb" if "XGBoost" in _ap else "lgbm"
    # 全站統一口徑:所有風險值皆為所選模型之 GroupKFold OOF 誠實預測
    PROB_COL = "prob_xgb" if (ALGO == "xgb" and "prob_xgb" in PREDS.columns) else "prob"
    TIER_COL = "tier_xgb" if (ALGO == "xgb" and "tier_xgb" in PREDS.columns) else "tier"
    if TIER_COL != "tier":
        PREDS = QD.annotate(PREDS, tier_col=TIER_COL)
        MY = PREDS[PREDS["host_id"] == host_id].reset_index(drop=True)

    radius = st.slider("📏 附近比較半徑 (公尺)", 500, 2000, 1000, step=100)
    st.divider()
    st.caption("LightGBM+XGBoost(Isotonic 校準)· 標籤 Y≥0.6 · "
               "LIME 可解釋 · GroupKFold 誠實驗證 · 59 特徵(含負評比例)")

st.markdown(f"""
<div style="padding:6px 0 10px;">
  <h1 style="font-size:1.4rem;font-weight:700;color:{P['ink']};margin:0;">
  房東營運面板</h1>
  <p style="font-size:.78rem;color:{P['muted']};margin:4px 0 0;">
  多房源優先序 · 跨平台定價落點 · 評論面向拆解 · 真實檔期基準</p>
</div><hr style="margin:0 0 12px;">""", unsafe_allow_html=True)

# 分頁順序 = 使用順序:先看該處理哪間,再看定價、口碑、檔期;
# 模型與 LIME 診斷退居後排(前瞻驗證 AUC 0.632,僅作輔助排序)。
TB1, TB2P, TB2R, TB5, TB2, TB4, TB6 = st.tabs(
    ["房源總表", "定價情報", "口碑情報", "未來檔期",
     "風險診斷", "通知中心", "月報"])
TB3 = TB2P          # 「周邊比較」內容併入定價情報分頁


def risk_ring(vac, color, size=92):
    """SVG 風險分數環(donut):百分比 = 高風險機率 P(空屋率≥60%),環色 = 等級色。"""
    r, sw = 38, 9
    c = 2 * np.pi * r
    filled = max(0.02, min(1.0, vac)) * c
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 100 100">
<circle cx="50" cy="50" r="{r}" fill="none" stroke="{P['tag_bg']}" stroke-width="{sw}"/>
<circle cx="50" cy="50" r="{r}" fill="none" stroke="{color}" stroke-width="{sw}"
 stroke-dasharray="{filled:.1f} {c:.1f}" stroke-linecap="round"
 transform="rotate(-90 50 50)"/>
<text x="50" y="55" text-anchor="middle" font-size="22" font-weight="800"
 fill="{color}">{vac*100:.0f}%</text></svg>"""


def trend_arrow(row):
    """趨勢箭頭:與同商圈(行政區)中位風險相比(↑高於商圈=紅,↓低於=綠)。"""
    d = float(row["vac_pred"]) - float(row["dist_med"])
    if d > 0.02:
        return f"<span style='color:{P['high']};font-weight:800;'>▲ 高於商圈 {d*100:.0f}pp</span>"
    if d < -0.02:
        return f"<span style='color:{P['low']};font-weight:800;'>▼ 低於商圈 {abs(d)*100:.0f}pp</span>"
    return f"<span style='color:{P['muted']};font-weight:700;'>◆ 與商圈持平</span>"


# ══════════════════════════════════════════════════════════════
# TB1 房東總覽
# ══════════════════════════════════════════════════════════════
with TB1:
    _gapd = MY["gap_days_30d"].fillna(0)
    _gapv = (MY["price"].fillna(0) * _gapd).sum()
    _alarm = int((MY["quadrant"] == "alarm").sum())
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("名下房源", f"{len(MY)} 間")
    k2.metric("近 30 天訂房率",
              "—" if MY["booked_rate_d30"].isna().all()
              else f"{MY['booked_rate_d30'].mean()*100:.0f}%",
              "真實日曆", delta_color="off")
    k3.metric("30 天空檔", f"{int(_gapd.sum())} 天",
              f"約 NT$ {_gapv:,.0f}", delta_color="off")
    k4.metric("需優先處理", f"{_alarm} 間", "檔期空且體質偏弱", delta_color="off")
    k5.metric("平均預測空屋率", f"{MY['vac_pred'].mean()*100:.0f}%",
              "模型推估·僅供參考", delta_color="off")
    # ── 體質 × 檔期 四象限 ──
    sec("體質(模型)× 檔期(真實已訂率)四象限")
    mb("模型 AUC 0.632 為體質推估;calendar 已訂率為 100% 真實觀測 · "
       "兩者衝突時以檔期為準(近期行動看檔期,長期投資看模型)")
    _qs = QD.summary(MY)
    _qcols = st.columns(len(_qs) if len(_qs) else 1)
    for _col, (_, _qr) in zip(_qcols, _qs.iterrows()):
        _col.metric(_qr["象限"], f"{_qr['房源數']} 間")
    html_table(_qs, wrap=True, scroll=False)

    _qopts = ["全部"] + _qs["象限"].tolist()
    _qpick = st.radio("篩選象限", _qopts, horizontal=True, key="q_filter")
    _cards = MY.copy()
    if _qpick != "全部":
        _cards = _cards[_cards["quadrant_label"] == _qpick]
    sec(f"房源卡片({len(_cards)} 間)· 風險環 = 高風險機率;箭頭 = 空屋率與同商圈中位比較")
    _cards = _cards.sort_values(["quadrant_priority", PROB_COL],
                                ascending=[True, False]).reset_index(drop=True)
    for _s in range(0, len(_cards), 3):
        cols = st.columns(3)
        for _c, (_, r) in zip(cols, _cards.iloc[_s:_s + 3].iterrows()):
            t_zh, t_c = TIER_ZH[r[TIER_COL]]
            with _c:
                st.markdown(f"""
<div style="background:{P['surface']};border:1px solid {P['border']};
     border-top:4px solid {t_c};border-radius:12px;padding:14px 16px;
     margin-bottom:12px;">
 <div style="display:flex;gap:12px;align-items:center;">
  <div>{risk_ring(float(r[PROB_COL]), t_c)}</div>
  <div style="min-width:0;">
   <div style="font-weight:700;font-size:.85rem;color:{P['ink']};
        white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:180px;"
        title="{str(r['name'])}">#{int(r['id'])} {str(r['name'])[:22]}</div>
   <div style="font-size:.74rem;color:{P['muted']};margin:2px 0;">
    {r['neighbourhood_cleansed']}·{ROOM_JP.get(r['room_type'], r['room_type'])}
    ·${r['price']:,.0f}/晚</div>
   <div style="font-size:.76rem;margin:3px 0;">
    <span style="background:{t_c};color:#fff;border-radius:12px;
     padding:2px 10px;font-weight:700;font-size:.72rem;">{t_zh}</span>
    <span style="font-size:.72rem;color:{P['muted']};">
     預測空屋率 {r['vac_pred']*100:.0f}%</span></div>
   <div style="font-size:.73rem;margin:2px 0;">
    <span style="background:{P[QD.QUADRANTS[r['quadrant']]['color']]};color:#fff;
     border-radius:10px;padding:1px 8px;font-weight:700;font-size:.68rem;">
     {r['quadrant_label']}</span>
    <span style="color:{P['muted']};margin-left:5px;">
     {'90天實訂 ' + format(r['booked_rate_d90']*100, '.0f') + '%' if r['booked_rate_d90'] == r['booked_rate_d90'] else '無檔期資料'}</span></div>
   <div style="font-size:.74rem;">{trend_arrow(r)}</div>
  </div></div></div>""", unsafe_allow_html=True)

# ── 詳情頁共用:選擇房源 ──
_opts = MY.sort_values(PROB_COL, ascending=False)
_opt_lab = {f"#{int(r['id'])}|{str(r['name'])[:26]}|"
            f"{TIER_ZH[r[TIER_COL]][0]} {r[PROB_COL]*100:.0f}%": int(r["id"])
            for _, r in _opts.iterrows()}

# ══════════════════════════════════════════════════════════════
# TB2 房源詳情(大分數 + LIME Top3 + 建議 + 趨勢線)
# ══════════════════════════════════════════════════════════════


with TB2:
    _sel = st.selectbox("選擇房源", list(_opt_lab.keys()))
    sel_id = _opt_lab[_sel]
    R = MY[MY["id"] == sel_id].iloc[0]
    ROW = DS_IDX.loc[sel_id] if sel_id in DS_IDX.index else None

    cA, cB = st.columns([1, 1.5], gap="large")
    with cA:
        # ── 基準值(OOF 誠實預測):與總覽/下拉/通知中心口徑一致 ──
        _vac0 = float(R["vac_pred"])
        _prob0 = float(R[PROB_COL])
        _tier0 = R[TIER_COL]
        _variant = R.get("variant", "full")

        # ── What-if 控制項(先操作、再看結果) ──
        _price0 = float(R["price"])
        _mn_now = pd.to_numeric(R["minimum_nights"], errors="coerce")
        _mn0 = 1 if pd.isna(_mn_now) else int(np.clip(_mn_now, 1, 30))
        if ROW is not None:
            sec("⚡ What-if 模擬(拖動後上方風險環即時重算)")
            _np_ = st.slider("每晚房價 (NT$)", 500, 50000,
                             int(np.clip(_price0, 500, 50000)), 100)
            _nm = st.number_input("最低入住天數(晚)", 1, 30, _mn0)
            _changed = (abs(_np_ - _price0) > 1) or (_nm != _mn0)
        else:
            _np_, _nm, _changed = _price0, _mn0, False

        # ── 依是否調整,決定環要顯示「基準」或「模擬後」 ──
        if _changed:
            _row2 = ROW.copy()
            _row2["minimum_nights"] = _nm
            _base_fit = predict_risk_v2(ROW, BUNDLE, algo=ALGO)
            _sim = simulate_price_change(_row2, BUNDLE, float(_np_), algo=ALGO)
            # 模擬以「相對變化」套回 OOF 基準,維持與全站一致的口徑
            _d_vac = _sim["risk_score"] - _base_fit["risk_score"]
            _d_prob = _sim["notify_prob"] - _base_fit["notify_prob"]
            _vac = float(np.clip(_vac0 + _d_vac, 0, 1))
            _prob = float(np.clip(_prob0 + _d_prob, 0, 1))
            _tier = ("red" if _prob >= .6 else
                     ("yellow" if _prob >= .35 else "green"))
        else:
            _vac, _prob, _tier = _vac0, _prob0, _tier0
            _d_vac = _d_prob = 0.0
        t_zh, t_c = TIER_ZH[_tier]

        _delta_html = ""
        if _changed:
            _dc = P["high"] if _d_prob > 0 else (P["low"] if _d_prob < 0 else P["muted"])
            _tier_moved = ("　等級 " + TIER_ZH[_tier0][0] + " → " + t_zh
                           if _tier != _tier0 else "")
            _delta_html = (
                f"<div style='margin-top:6px;font-size:.86rem;color:{_dc};"
                f"font-weight:700;'>模擬變化:機率 {_d_prob*100:+.1f} pp·"
                f"空屋率 {_d_vac*100:+.1f} pp{_tier_moved}</div>"
                f"<div style='font-size:.74rem;color:{P['muted']};'>"
                f"基準(現況):機率 {_prob0*100:.0f}%·空屋率 {_vac0*100:.0f}%</div>")

        st.markdown(
            f"<div style='text-align:center;background:{P['surface']};"
            f"border:1px solid {P['border']};border-radius:14px;padding:18px;"
            f"{'border:2px dashed ' + P['accent'] + ';' if _changed else ''}'>"
            f"<div style='font-size:.78rem;color:{P['muted']};letter-spacing:.08em;'>"
            f"{'⚡ 模擬後風險' if _changed else 'GroupKFold OOF 誠實預測'}</div>"
            f"{risk_ring(_prob, t_c, size=170)}"
            f"<div style='margin-top:6px;'><span style='background:{t_c};color:#fff;"
            f"border-radius:16px;padding:4px 18px;font-weight:800;'>{t_zh}</span></div>"
            f"{_delta_html}"
            f"<div style='color:{P['muted']};font-size:.8rem;margin-top:8px;'>"
            f"環 = 高風險機率 P(空屋率≥60%),紅≥60%·黃≥35%·綠<35%<br>"
            f"預測空屋率(模型A):<b>{_vac*100:.0f}%</b><br>"
            f"模型:{'XGBoost' if ALGO == 'xgb' else 'LightGBM'}·"
            f"{'冷啟動' if _variant == 'cold' else '完整'}變體</div></div>",
            unsafe_allow_html=True)

        if _changed and abs(_d_prob) < 0.005 and abs(_d_vac) < 0.005:
            note("⚠️ <b>此調整幅度下模型預測未變動</b>,原因有二:"
                 "①樹模型與 Isotonic 校準皆為<b>階梯函數</b>,微調常落在同一階;"
                 "②在本資料集中,<b>價格與最低入住天數並非主要風險驅動因子</b>"
                 "(前向選擇中分居第 19、15 名,屬噪音帶)。"
                 "真正的強訊號是<b>房東接受率、回覆速度、周邊口碑排名</b> —— "
                 "見右側 LIME 原因與改善建議。可試更大幅度調整觀察階梯跳動。")

    with cB:
        sec("LIME 原因 Top 3(為什麼有風險)")
        mb("LIME 局部線性近似 · 解釋模型 B 的 P(空屋率≥60%) · 正值=推高風險")
        if ROW is None:
            st.info("此房源不在訓練協定內(經營未滿一年),無法提供 LIME 解釋。")
            _lime_up = []
        else:
            try:
                from modules.lime_explainer import lime_reasons
                _ov = {"price": _np_, "minimum_nights": _nm}
                _lime = lime_reasons(ROW, _variant, ALGO,
                                     overrides=_ov, k=3)
                _lime_up = [x for x in _lime if x["direction"] == "up"][:3]
                _lime_dn = [x for x in _lime if x["direction"] == "down"][:2]
                _mx = max((abs(x["weight_pp"]) for x in _lime), default=1) or 1
                for i, x in enumerate(_lime_up, 1):
                    _w = abs(x["weight_pp"]) / _mx * 100
                    st.markdown(
                        f"<div style='margin:8px 0;'>"
                        f"<div style='font-size:.85rem;font-weight:700;'>"
                        f"{i}. {x['zh']}"
                        f"<span style='color:{P['high']};float:right;'>"
                        f"+{x['weight_pp']:.1f} pp</span></div>"
                        f"<div style='background:{P['tag_bg']};border-radius:6px;"
                        f"height:10px;margin-top:3px;'>"
                        f"<div style='width:{_w:.0f}%;background:{P['high']};"
                        f"height:10px;border-radius:6px;'></div></div></div>",
                        unsafe_allow_html=True)
                if _lime_dn:
                    note("✅ 加分項:" + "; ".join(
                        f"{x['zh']}({x['weight_pp']:.1f}pp)" for x in _lime_dn))
            except ImportError:
                st.warning("尚未安裝 lime 套件:pip install lime")
                _lime_up = []

        sec("💡 改善建議")
        _cs = None
        try:
            _cap = max(float(R["accommodates"] or 2), 1)
            _cs = comp_index().stats(
                float(R["latitude"]), float(R["longitude"]),
                listing_pp_day=float(R["price"]) / _cap,
                bracket=capacity_bracket(_cap), radius_m=float(radius),
                exclude_listing_id=int(sel_id))
        except FileNotFoundError:
            pass
        _own_am = _canon_amenities(str(R.get("amenities", "")))
        _gaps = ([k for k, v in _cs["amenity_coverage"].items()
                  if v >= .5 and k not in _own_am][:3] if _cs else [])

        # LLM 僅於「紅色高風險」提供,且一律「手動按鈕」觸發 ——
        # 調整房價/最低入住天數時不自動呼叫,避免每拖一次滑桿就打一次 API。
        _use_rules = True
        if _tier == "red":
            from modules.llm_advisor import llm_available, generate_advice
            _prov_name = llm_available()
            if _prov_name:
                mb(f"LLM 智慧建議({_prov_name})· 🔴 高風險觸發 · 手動產生",
                   warning=True)
                # 參數指紋:房價、最低天數、模型、房源任一改變即視為「已過期」
                _sig = f"{sel_id}|{ALGO}|{_np_ if ROW is not None else 0}|" \
                       f"{_nm if ROW is not None else 0}"
                _store = st.session_state.setdefault("llm_advice_store", {})
                _hit = _store.get(sel_id)

                _b1, _b2 = st.columns([1, 1])
                _clicked = _b1.button("🧠 產生 LLM 智慧建議", key=f"llm_go_{sel_id}",
                                      use_container_width=True)
                if _hit and _b2.button("🗑 清除", key=f"llm_clr_{sel_id}",
                                       use_container_width=True):
                    _store.pop(sel_id, None)
                    _hit = None
                    st.rerun()

                if _clicked:
                    with st.spinner("🧠 LLM 生成個人化建議中 …"):
                        try:
                            _prov, _md = generate_advice({
                                "name": str(R["name"]),
                                "district": R["neighbourhood_cleansed"],
                                "room_type": ROOM_JP.get(R["room_type"], R["room_type"]),
                                "price": float(_np_ if ROW is not None else R["price"]),
                                "vac_pred": _vac, "prob": _prob,
                                "tier": TIER_ZH[_tier][0],
                                "lime_reasons": _lime_up,
                                "comp_summary": (
                                    f"1km 內 {_cs['n_total']} 筆競品,"
                                    f"同容量層貴於 {_cs['pp_percentile']:.0%}"
                                    if _cs and _cs.get("pp_percentile") is not None
                                    else "無資料"),
                                "amenity_gaps": _gaps})
                            _store[sel_id] = {"sig": _sig, "prov": _prov,
                                              "md": _md,
                                              "at": pd.Timestamp.now().strftime("%H:%M")}
                            _hit = _store[sel_id]
                        except Exception as e:
                            st.error(f"LLM 呼叫失敗:{type(e).__name__} — {e}")
                            st.caption("常見原因:金鑰無效或未啟用 API、模型名稱已淘汰、"
                                       "配額用盡、或雲端網路限制。下方仍提供規則引擎建議。")

                if _hit:
                    if _hit["sig"] != _sig:
                        note("⚠️ 此建議是在<b>調整參數前</b>產生的"
                             f"(生成於 {_hit['at']})。若要依目前的房價與"
                             "最低入住天數重新生成,請再按一次上方按鈕。")
                    st.markdown(_hit["md"])
                    st.caption(f"由 {_hit['prov']} 於 {_hit['at']} 生成;"
                               f"建議僅供參考,請依實際經營狀況判斷。")
                    _use_rules = False
                else:
                    note("👆 按上方按鈕即可產生 LLM 個人化建議"
                         "(調整房價或最低入住天數<b>不會</b>自動觸發,避免重複呼叫 API)。"
                         "以下先提供規則引擎建議。")
            else:
                note("未設定 LLM 金鑰(ANTHROPIC_API_KEY / GEMINI_API_KEY),"
                     "以下為規則引擎建議;設定金鑰後即可手動產生 LLM 個人化建議。")
        elif _tier == "yellow":
            note("🟡 觀察層:以下為規則引擎建議;LLM 個人化建議於"
                 "🔴 紅色高風險(機率 ≥ 60%)時才可手動產生。")
        else:
            note("🟢 綠色安全層:暫無需調整,以下為保持競爭力的常規檢查。")

        if _use_rules and _cs is not None:
            _shap_items = [(x["rule"].split(" ")[0], x["weight_pp"] / 100)
                           for x in _lime_up]
            _sugs = sugg_engine().suggest(
                shap_items=_shap_items, comp_stats=_cs,
                features={"accommodates": _cap,
                          "minimum_nights": float(np.nan_to_num(pd.to_numeric(
                              R["minimum_nights"], errors="coerce"))),
                          "desc_len": float(len(str(R.get("description") or "")))},
                own_amenities=_own_am)
            for i, s in enumerate(_sugs[:4], 1):
                note(f"<b>{i}. {s['title']}</b>:{s['detail']}"
                     f"<br><span style='font-size:.72rem;'>依據:{s['evidence']}</span>")

    # ── 趨勢線(價格 What-if 模擬 × 真實市場對照) ──
    if ROW is not None:
        sec("趨勢線:價格 What-if 模擬 × 真實市場對照(非歷史時序)")
        _lo = max(500, int(float(R["price"]) * .5))
        _hi = int(float(R["price"]) * 1.8) + 500
        _xs = np.linspace(_lo, _hi, 15)
        _sims = [simulate_price_change(ROW, BUNDLE, float(p), algo=ALGO)
                 for p in _xs]
        _ys = [s["risk_score"] * 100 for s in _sims]
        _ps = [s["notify_prob"] * 100 for s in _sims]

        # 真實數據對照:同區同房型房源在各價格帶的「實際」平均空屋率
        _peer = DS[(DS["neighbourhood_code"] == ROW["neighbourhood_code"]) &
                   (DS["room_type_code"] == ROW["room_type_code"])]
        _emp_x, _emp_y, _emp_n = [], [], []
        if len(_peer) >= 20:
            _edges = np.linspace(_lo, _hi, 9)
            for _a, _b_ in zip(_edges[:-1], _edges[1:]):
                _g = _peer[(_peer["price"] >= _a) & (_peer["price"] < _b_)]
                if len(_g) >= 5:  # 每帶至少 5 筆真實樣本才畫
                    _emp_x.append((_a + _b_) / 2)
                    _emp_y.append(_g["Y_vacancy"].mean() * 100)
                    _emp_n.append(len(_g))

        _figT = go.Figure()
        _figT.add_trace(go.Scatter(x=_xs, y=_ys, mode="lines+markers",
                                   line=dict(color=P["primary"], width=3),
                                   name="模型預測空屋率(What-if)"))
        _figT.add_trace(go.Scatter(x=_xs, y=_ps, mode="lines",
                                   line=dict(color=P["medium"], width=2,
                                             dash="dash"),
                                   name="P(高風險) 機率"))
        if _emp_x:
            _figT.add_trace(go.Scatter(
                x=_emp_x, y=_emp_y, mode="markers+lines",
                line=dict(color=P["accent"], width=2, dash="dot"),
                marker=dict(size=[max(8, min(20, n / 2)) for n in _emp_n],
                            symbol="diamond"),
                name=f"真實市場:同區同房型實際空屋率(n={sum(_emp_n)})",
                hovertext=[f"價格帶均值·{n} 筆真實房源" for n in _emp_n]))
        _figT.add_vline(x=float(R["price"]), line_dash="dot",
                        line_color=P["accent"],
                        annotation_text=f"目前 ${float(R['price']):,.0f}")
        _figT.add_hline(y=60, line_dash="dot", line_color=P["high"],
                        annotation_text="60% 高風險線")
        apply_theme(_figT, h=320).update_layout(
            xaxis_title="每晚房價 (NT$)", yaxis_title="空屋率 / 機率 (%)")
        st.plotly_chart(_figT, use_container_width=True)
        st.caption("藍線 = 模型對「調價後」的 What-if 預測(price_pctl 已依同區同房型真實價格分佈"
                   "動態重排);菱形虛線 = **真實市場數據**:同區同房型房源在各價格帶的實際平均空屋率"
                   "(菱形越大樣本越多)。兩者趨勢一致代表模擬可信;樹模型對價格呈階梯狀反應屬正常。")

    st.caption("住客評論的面向拆解已移至「口碑情報」分頁。")

# ══════════════════════════════════════════════════════════════
# 口碑情報:評論拆成 14 個面向,與同區基準對照
# ══════════════════════════════════════════════════════════════
with TB2R:
    from modules.absa_sections import render_listing_absa
    _selR = st.selectbox("選擇房源", list(_opt_lab.keys()), key="rep_tab_sel")
    _rid2 = _opt_lab[_selR]
    _rrow2 = MY[MY["id"] == _rid2].iloc[0]
    render_listing_absa(_rid2, _rrow2["neighbourhood_cleansed"])
    st.caption("面向以中英關鍵詞比對,情感取關鍵詞前後 30 字的局部窗口計分;"
               "提及少於 3 次者不列入。全市負評率最高:空調冷氣 15.7%、"
               "隔音噪音 13.6%、空間大小 13.4%。")


# ══════════════════════════════════════════════════════════════
# TB3 附近比較(熱力圖 + 風險比較 + 同商圈排名 + 跨平台)
# ══════════════════════════════════════════════════════════════
with TB3:
    _sel3 = st.selectbox("基準房源", list(_opt_lab.keys()), key="nb_sel")
    bid = _opt_lab[_sel3]
    B = MY[MY["id"] == bid].iloc[0]
    _blat, _blon = float(B["latitude"]), float(B["longitude"])

    # ── 跨平台定價落點(主打:Airbnb 後台看不到的資訊)──
    sec(f"跨平台定價落點({radius}m 內 · 591 / Booking / 租租網 / Airbnb)")
    mb("月租平台掛牌價 ÷30 換算為每晚等效價,再除以可住人數 —— "
       "統一為「每人每晚」才能跨平台比較(未計押金、管理費與最短租期)")
    try:
        _cs0 = comp_index().stats(
            _blat, _blon,
            listing_pp_day=float(B["price"]) / max(float(B["accommodates"]), 1),
            bracket=capacity_bracket(B["accommodates"]),
            radius_m=float(radius), exclude_listing_id=int(bid))
        _my_pp = float(B["price"]) / max(float(B["accommodates"]), 1)
        _pc = st.columns(5)
        with _pc[0]:
            # 指標 + hover 摘要 + 查看詳情,收在同一個區塊內
            with st.container(border=True):
                st.metric("我的每人每晚", f"${_my_pp:,.0f}",
                          f"每晚 ${float(B['price']):,.0f}", delta_color="off")
                from modules import listing_detail as LD
                _BD = DF_RAW[DF_RAW["id"] == int(bid)]
                if len(_BD):
                    _BD = _BD.iloc[0]
                    # 滑鼠停留顯示摘要;完整內容(與租客入口同一個彈窗)由按鈕開啟
                    st.markdown(LD.HOVER_CSS + LD.hover_card_html(
                        _BD, extra_lines=[
                            f"📊 每人每晚 <b>${_my_pp:,.0f}</b>"
                            f"(可住 {int(float(B['accommodates']))} 人)"]),
                        unsafe_allow_html=True)
                    if st.button("🔍 查看詳情", key=f"detail_price_{bid}",
                                 width="stretch"):
                        LD.open_detail(_BD)
        for _i, _pl in enumerate(["Airbnb", "Booking", "591", "ddroom"], start=1):
            _v = _cs0["platforms"].get(_pl)
            with _pc[_i]:
                with st.container(border=True):
                    st.metric({"ddroom": "租租網"}.get(_pl, _pl),
                              f"{_v['count']} 筆" if _v else "0 筆",
                              f"中位 ${_v['pp_median']:,.0f}" if _v else None,
                              delta_color="off")
        _pp0 = _cs0.get("pp_percentile")
        if _pp0 is not None:
            _pcol = (P["high"] if _pp0 >= .6 else
                     (P["medium"] if _pp0 >= .4 else P["low"]))
            note(f"同容量層({B['bracket'] if 'bracket' in B else capacity_bracket(B['accommodates'])})"
                 f"共 {_cs0['n_same_bracket']} 筆競品,你的每人每晚 "
                 f"<b style='color:{_pcol}'>貴於 {_pp0:.0%}</b>,"
                 f"同層中位為 <b>${_cs0['bracket_pp_median']:,.0f}</b>。")
        else:
            note(f"同容量層競品不足 5 筆,暫不計算價格落點。")

        # 設施缺口(跨平台競品過半具備、本房源沒有)
        _own0 = _canon_amenities(str(B.get("amenities", "")))
        _gap0 = [(k, v) for k, v in _cs0.get("amenity_coverage", {}).items()
                 if v >= .5 and k not in _own0][:4]
        if _gap0:
            note("設施缺口:" + "、".join(
                f"<b>{k}</b>(周邊 {v:.0%} 具備)" for k, v in _gap0)
                + " —— 這些是同商圈的標配。")
    except FileNotFoundError:
        st.caption("競品索引未建置,請執行 scripts/train_backend_models.py。")

    # ── 同商圈同房型的真實已訂率基準 ──
    sec("同商圈同房型 真實已訂率基準")
    mb("取自 Inside Airbnb 日曆的實際訂房狀態,非模型推估")
    _peer = PREDS[(PREDS["neighbourhood_cleansed"] == B["neighbourhood_cleansed"])
                  & (PREDS["room_type"] == B["room_type"])]
    _peer = _peer[_peer["booked_rate_d90"].notna()]
    if len(_peer) >= 10:
        _b1, _b2, _b3, _b4 = st.columns(4)
        _mine90 = B.get("booked_rate_d90")
        _mine30 = B.get("booked_rate_d30")
        _b1.metric("我的 90 天訂房率",
                   "—" if pd.isna(_mine90) else f"{_mine90:.0%}")
        _b2.metric("同商圈同房型中位",
                   f"{_peer['booked_rate_d90'].median():.0%}",
                   f"{len(_peer)} 間", delta_color="off")
        if pd.notna(_mine90):
            _rank = float((_peer["booked_rate_d90"] < _mine90).mean())
            _b3.metric("我的排名", f"贏過 {_rank:.0%}",
                       "同商圈同房型", delta_color="off")
        _b4.metric("我的 30 天訂房率",
                   "—" if pd.isna(_mine30) else f"{_mine30:.0%}")
        _figB = px.histogram(_peer, x="booked_rate_d90", nbins=20,
                             color_discrete_sequence=[P["primary"]],
                             labels={"booked_rate_d90": "未來 90 天訂房率"})
        if pd.notna(_mine90):
            _figB.add_vline(x=float(_mine90), line_dash="dot",
                            line_color=P["high"], annotation_text="我的位置")
        apply_theme(_figB, h=260).update_layout(
            title=f"{B['neighbourhood_cleansed']}·"
                  f"{ROOM_JP.get(B['room_type'], B['room_type'])} 訂房率分佈",
            yaxis_title="房源數")
        st.plotly_chart(_figB, width="stretch")
    else:
        st.caption("同商圈同房型樣本不足 10 間,不建立基準。")

    st.divider()

    # 附近 Airbnb(半徑內)
    _d = PREDS.copy()
    _d["dist_m"] = 6371000 * 2 * np.arcsin(np.sqrt(
        np.sin(np.radians(_d["latitude"] - _blat) / 2) ** 2 +
        np.cos(np.radians(_blat)) * np.cos(np.radians(_d["latitude"])) *
        np.sin(np.radians(_d["longitude"] - _blon) / 2) ** 2))
    NB = _d[(_d["dist_m"] <= radius)].sort_values("dist_m")
    NB_o = NB[NB["id"] != bid]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{radius}m 內 Airbnb", f"{len(NB_o)} 間")
    c2.metric("周邊高風險機率中位", f"{NB_o[PROB_COL].median()*100:.0f}%"
              if len(NB_o) else "—")
    _pctl_rk = float((NB_o[PROB_COL] < float(B[PROB_COL])).mean()) \
        if len(NB_o) else None
    c3.metric("我的高風險機率 vs 周邊",
              f"{B[PROB_COL]*100:.0f}%",
              f"高於 {_pctl_rk:.0%} 的鄰居" if _pctl_rk is not None else None,
              delta_color="off")
    _rank = (int((NB[PROB_COL] < float(B[PROB_COL])).sum()) + 1) \
        if len(NB) else 1
    c4.metric("同商圈排名(低風險優先)", f"第 {_rank} / {len(NB)} 名")

    sec("周邊房源分佈圖")
    mb("點位顏色 = 預測空屋率(綠 <40%・黃 40–69%・紅 ≥70%)· "
       "虛線圓為比對半徑 · 閃爍點為本房源 · 滑過任一點會同步標示右側列表")
    from modules.geo_utils import nearest_address as _addr_fn
    from modules import map_view as MV
    try:
        _xc = comp_index().query(_blat, _blon, radius_m=float(radius))
        _xc = _xc[_xc["platform"] != "Airbnb"].copy()
    except FileNotFoundError:
        _xc = pd.DataFrame(columns=["platform", "lat", "lon", "title",
                                    "price_raw", "price_pp_day", "dist_m",
                                    "capacity", "price_unit"])
        st.caption("競品索引未建置,僅顯示 Airbnb 房源。")
    MV.render(own=B, nearby=NB_o.head(180), comp=_xc,
              radius_m=radius, addr_fn=_addr_fn, height=520)
    st.caption("跨平台圖層可於地圖右上角分別勾選;點擊列表項目會將地圖移至該房源。")

    st.divider()
    _mcol, _rcol = st.columns([1.5, 1], gap="medium")
    with _mcol:
        sec("同商圈風險分佈")
        _figS = px.scatter(
            NB.assign(空屋率=(NB["vac_pred"] * 100).round(0)),
            x="price", y="vac_pred", color="vac_pred", size="accommodates",
            color_continuous_scale=["#5B9E73", "#F7D774", "#C4645A"],
            labels={"price": "每晚價格 (NT$)", "vac_pred": "預測空屋率",
                    "accommodates": "可住人數"},
            hover_data={"id": True, "空屋率": True, "price": ":,.0f"})
        _figS.add_vline(x=float(B["price"]), line_dash="dot",
                        line_color="#2A2A2A", annotation_text="本房源定價")
        apply_theme(_figS, h=430).update_layout(
            coloraxis_colorbar_title="空屋率")
        st.plotly_chart(_figS, width="stretch")
    with _rcol:
        sec("同商圈排名表(依高風險機率,低者優先)")
        if len(NB):
            _tb = NB.head(60)[["id", "name", "price", "vac_pred",
                               PROB_COL, TIER_COL, "dist_m"]].copy()
            _tb["房源"] = _tb.apply(
                lambda r: ("👉 " if int(r["id"]) == bid else "")
                + f"#{int(r['id'])} {str(r['name'])[:12]}", axis=1)
            _tb["每晚"] = _tb["price"].map("${:,.0f}".format)
            _tb["機率"] = _tb[PROB_COL].map("{:.0%}".format)
            _tb["空屋率"] = _tb["vac_pred"].map("{:.0%}".format)
            _tb["等級"] = _tb[TIER_COL].map(lambda t: TIER_ZH[t][0])
            _tb["距離"] = _tb["dist_m"].map("{:.0f}m".format)
            _tb = _tb.sort_values(PROB_COL)
            html_table(_tb[["房源", "每晚", "機率", "等級", "空屋率", "距離"]],
                       height=470)
        else:
            st.info("半徑內無其他房源。")

    # (跨平台落點已移至本分頁最上方)
    try:
        _cs3 = comp_index().stats(
            _blat, _blon,
            listing_pp_day=float(B["price"]) / max(float(B["accommodates"]), 1),
            bracket=capacity_bracket(B["accommodates"]),
            radius_m=float(radius), exclude_listing_id=int(bid))
        _pl_txt = " · ".join(
            f"{p} {v['count']} 筆(每人每晚中位 ${v['pp_median']:,.0f})"
            for p, v in _cs3["platforms"].items())
        _pp = _cs3.get("pp_percentile")
        note(f"🌐 <b>跨平台競品({radius}m)</b>:{_pl_txt or '無'}"
             + (f"<br>價格落點:同容量層每人每晚貴於 <b>{_pp:.0%}</b> 的競品"
                f"(月租÷30 換算,樣本 {_cs3['n_same_bracket']} 筆)"
                if _pp is not None else ""))
    except FileNotFoundError:
        pass

# ══════════════════════════════════════════════════════════════
# TB4 通知中心(共用模組:自動寄信 + 智慧建議 + 手動補寄)
# ══════════════════════════════════════════════════════════════
with TB4:
    from modules.notify_center import render_notify_center
    render_notify_center(host_id=host_id, prob_col=PROB_COL,
                         tier_col=TIER_COL, key="host_nc")


# ══════════════════════════════════════════════════════════════
# TB5 未來檔期(calendar.csv.gz · 獨立模組,不影響既有分頁)
# ══════════════════════════════════════════════════════════════
with TB5:
    from modules.calendar_sections import render_calendar_tab
    _sel5 = st.selectbox("選擇房源", list(_opt_lab.keys()), key="cal_sel")
    _cid = _opt_lab[_sel5]
    _crow = MY[MY["id"] == _cid].iloc[0]
    render_calendar_tab(_cid, _crow, DF_RAW)


# ══════════════════════════════════════════════════════════════
# TB6 月報自動生成(純讀取既有分析,不影響其他分頁)
# ══════════════════════════════════════════════════════════════
with TB6:
    from modules import report_builder as rb
    sec("📄 房源經營月報自動生成")
    mb("彙整風險等級 · 檔期進度 · 空檔明細 · 營收最適定價 · 評論面向 · AI 摘要")
    _sel6 = st.selectbox("選擇房源", list(_opt_lab.keys()), key="rep_sel")
    _rid = _opt_lab[_sel6]
    _rrow = MY[MY["id"] == _rid].iloc[0]
    _use_llm = st.checkbox("使用 LLM 生成 AI 摘要(需設定金鑰,較慢)", value=False,
                           key="rep_llm")
    if st.button("🧾 產生月報", key="rep_go"):
        with st.spinner("彙整資料並生成月報 …"):
            _d = rb.collect(_rrow, _rrow, DF_RAW, PROB_COL, TIER_COL)
            if _use_llm:
                _src, _sum = rb.ai_summary(_d)
            else:
                _src, _sum = "規則摘要", rb._rule_summary(_d)
            _md = rb.to_markdown(_d, _src, _sum)
        st.session_state["report_md"] = _md
        st.session_state["report_name"] = f"月報_{_rid}_{_d['generated'][:10]}"
        st.toast("月報已生成")
    if st.session_state.get("report_md"):
        _md = st.session_state["report_md"]
        _nm = st.session_state.get("report_name", "月報")
        _c1, _c2 = st.columns(2)
        _c1.download_button("⬇️ 下載 Markdown", _md, file_name=f"{_nm}.md",
                            mime="text/markdown", use_container_width=True)
        _c2.download_button("⬇️ 下載 HTML(可列印/存 PDF)",
                            rb.to_html(_md, _nm), file_name=f"{_nm}.html",
                            mime="text/html", use_container_width=True)
        with st.container(border=True):
            st.markdown(_md)
