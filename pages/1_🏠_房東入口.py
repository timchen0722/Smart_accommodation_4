# -*- coding: utf-8 -*-
"""房東入口 — 房東 Dashboard(依專題 Workflow docx §四/§五 重新製作)

四視圖:
  🏠 房東總覽   房源卡片 · 風險分數環 · 等級色 · 趨勢箭頭
  📋 房源詳情   大風險分數 · LIME 原因 Top 3 · 改善建議(LLM/規則) · 趨勢線
  🗺 附近比較   地圖熱力圖 · 自己 vs 周邊風險 · 同商圈排名表 · 跨平台競品
  🔔 通知中心   60% 門檻高風險房源 · 通知紀錄 · 已處理狀態
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
    return preds, ds


@st.cache_resource(show_spinner=False)
def get_bundle():
    return fe.load_bundle()


@st.cache_resource(show_spinner="載入跨平台競品索引 …")
def comp_index():
    return _load_pkl("competitor_index")


@st.cache_resource(show_spinner=False)
def sugg_engine():
    return _load_pkl("suggestion_engine")


PREDS, DS = load_all()
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

    radius = st.slider("📏 附近比較半徑 (公尺)", 500, 2000, 1000, step=100)
    st.divider()
    st.caption("LightGBM+XGBoost(Isotonic 校準)· 標籤 Y≥0.6 · "
               "LIME 可解釋 · GroupKFold 誠實驗證 · 59 特徵(含負評比例)")

st.markdown(f"""
<div style="padding:6px 0 10px;">
  <h1 style="font-size:1.4rem;font-weight:700;color:{P['ink']};margin:0;">
  🏠 房東 Dashboard</h1>
  <p style="font-size:.78rem;color:{P['muted']};margin:4px 0 0;">
  房東總覽 → 房源詳情(LIME)→ 附近比較 → 60% 通知中心</p>
</div><hr style="margin:0 0 12px;">""", unsafe_allow_html=True)

TB1, TB2, TB3, TB4 = st.tabs(["🏠 房東總覽", "📋 房源詳情", "🗺 附近比較", "🔔 通知中心"])


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
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("名下房源", f"{len(MY)} 間")
    k2.metric("平均預測空屋率", f"{MY['vac_pred'].mean()*100:.0f}%")
    k3.metric("🔴 高風險", f"{int((MY[TIER_COL] == 'red').sum())} 間",
              f"🟡 觀察 {int((MY[TIER_COL] == 'yellow').sum())} 間", delta_color="off")
    k4.metric("最需優先處理",
              f"#{int(MY.sort_values(PROB_COL, ascending=False).iloc[0]['id'])}"
              if len(MY) else "—",
              "依高風險機率排序", delta_color="off")
    sec("房源卡片(風險分數環 = 高風險機率 P(空屋率≥60%),環色 = 等級;箭頭 = 空屋率與同商圈中位比較)")
    _cards = MY.sort_values(PROB_COL, ascending=False).reset_index(drop=True)
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
        sec("風險評估(GroupKFold OOF 誠實預測)")
        # 主指標一律採 OOF 誠實值 —— 與總覽卡片/下拉選單/通知中心完全一致
        _vac = float(R["vac_pred"])
        _prob = float(R[PROB_COL])
        _tier = R[TIER_COL]
        _variant = R.get("variant", "full")
        t_zh, t_c = TIER_ZH[_tier]
        st.markdown(
            f"<div style='text-align:center;background:{P['surface']};"
            f"border:1px solid {P['border']};border-radius:14px;padding:18px;'>"
            f"{risk_ring(_prob, t_c, size=170)}"
            f"<div style='margin-top:6px;'><span style='background:{t_c};color:#fff;"
            f"border-radius:16px;padding:4px 18px;font-weight:800;'>{t_zh}</span></div>"
            f"<div style='color:{P['muted']};font-size:.82rem;margin-top:8px;'>"
            f"環 = 高風險機率 P(空屋率≥60%),紅≥60%·黃≥35%·綠<35%<br>"f"OOF 預測空屋率(模型A):<b>{_vac*100:.0f}%</b><br>"
            f"模型:{'XGBoost' if ALGO == 'xgb' else 'LightGBM'}·"
            f"{'冷啟動' if _variant == 'cold' else '完整'}變體·"
            f"OOF 誠實評估(模型未看過此房東)</div></div>",
            unsafe_allow_html=True)
        if ROW is not None:
            st.markdown("**⚡ What-if 模擬**")
            _np_ = st.slider("每晚房價 (NT$)", 500, 50000,
                             int(np.clip(float(R["price"]), 500, 50000)), 100)
            _mn_now = pd.to_numeric(R["minimum_nights"], errors="coerce")
            _mn_now = 1 if pd.isna(_mn_now) else _mn_now
            _nm = st.number_input("最低入住天數(晚)", 1, 30,
                                  int(np.clip(_mn_now, 1, 30)))
            _row2 = ROW.copy()
            _row2["minimum_nights"] = _nm
            # What-if 用全量擬合模型:只看「相對變化」,基準亦為同一模型口徑
            _base_fit = predict_risk_v2(ROW, BUNDLE, algo=ALGO)["risk_score"]
            _sim = simulate_price_change(_row2, BUNDLE, float(_np_), algo=ALGO)
            st.metric("What-if 相對變化(模擬口徑)",
                      f"{(_sim['risk_score'] - _base_fit)*100:+.1f} pp",
                      f"模擬空屋率 {_sim['risk_score']*100:.1f}%",
                      delta_color="off")
            st.caption("模擬使用全量擬合模型,數值與上方 OOF 誠實評估口徑不同;"
                       "請以「相對變化」判讀調整方向與幅度。")

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

        # LLM 僅於「紅色高風險」觸發(依需求:高風險才由 LLM 給建議);
        # 黃色觀察層與綠色安全層使用規則引擎。
        if _tier == "red":
            from modules.llm_advisor import llm_available, generate_advice
            if llm_available():
                mb(f"LLM 智慧建議({llm_available()})· 🔴 高風險觸發", warning=True)
                try:
                    @st.cache_data(show_spinner="🧠 LLM 生成個人化建議中 …", ttl=3600)
                    def _llm_advice(_key: str, ctx: dict):
                        return generate_advice(ctx)

                    _prov, _md = _llm_advice(
                        f"{sel_id}-{ALGO}-{_tier}-{_np_ if ROW is not None else 0}",
                        {"name": str(R["name"]), "district": R["neighbourhood_cleansed"],
                         "room_type": ROOM_JP.get(R["room_type"], R["room_type"]),
                         "price": float(R["price"]), "vac_pred": _vac,
                         "prob": _prob, "tier": TIER_ZH[_tier][0],
                         "lime_reasons": _lime_up,
                         "comp_summary": (f"1km 內 {_cs['n_total']} 筆競品,"
                                          f"同容量層貴於 {_cs['pp_percentile']:.0%}"
                                          if _cs and _cs.get("pp_percentile") is not None
                                          else "無資料"),
                         "amenity_gaps": _gaps})
                    st.markdown(_md)
                    st.caption(f"由 {_prov} 生成;建議僅供參考,請依實際經營狀況判斷。")
                except Exception as e:
                    st.warning(f"LLM 呼叫失敗({type(e).__name__}),改用規則引擎建議。")
                    _use_rules = True
                else:
                    _use_rules = False
            else:
                note("未設定 LLM 金鑰(ANTHROPIC_API_KEY / GEMINI_API_KEY),"
                     "以下為規則引擎建議;設定金鑰後即自動改由 LLM 生成個人化建議。")
                _use_rules = True
        elif _tier == "yellow":
            note("🟡 觀察層:以下為規則引擎建議;LLM 個人化建議於"
                 "🔴 紅色高風險(機率 ≥ 60%)時才會觸發。")
            _use_rules = True
        else:
            note("🟢 綠色安全層:暫無需調整,以下為保持競爭力的常規檢查。")
            _use_rules = True

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

    # ── 趨勢線(價格 What-if 曲線) ──
    if ROW is not None:
        sec("趨勢線:價格 What-if 模擬(非歷史時序)")
        _lo = max(500, int(float(R["price"]) * .5))
        _hi = int(float(R["price"]) * 1.6) + 500
        _xs = np.linspace(_lo, _hi, 15)
        _ys = [simulate_price_change(ROW, BUNDLE, float(p),
                                     algo=ALGO)["risk_score"] * 100 for p in _xs]
        _figT = go.Figure()
        _figT.add_trace(go.Scatter(x=_xs, y=_ys, mode="lines+markers",
                                   line=dict(color=P["primary"], width=3),
                                   name="預測空屋率"))
        _figT.add_vline(x=float(R["price"]), line_dash="dot",
                        line_color=P["accent"],
                        annotation_text=f"目前 ${float(R['price']):,.0f}")
        _figT.add_hline(y=60, line_dash="dot", line_color=P["high"],
                        annotation_text="60% 高風險線")
        apply_theme(_figT, h=290).update_layout(
            xaxis_title="每晚房價 (NT$)", yaxis_title="預測空屋率 (%)")
        st.plotly_chart(_figT, use_container_width=True)
        st.caption("模型 A 對「調價後空屋率」的直接回答;樹模型對價格呈階梯狀反應屬正常現象。"
                   "價格百分位為市場相對排名,單筆模擬不重排整個市場。")

# ══════════════════════════════════════════════════════════════
# TB3 附近比較(熱力圖 + 風險比較 + 同商圈排名 + 跨平台)
# ══════════════════════════════════════════════════════════════
with TB3:
    _sel3 = st.selectbox("基準房源", list(_opt_lab.keys()), key="nb_sel")
    bid = _opt_lab[_sel3]
    B = MY[MY["id"] == bid].iloc[0]
    _blat, _blon = float(B["latitude"]), float(B["longitude"])

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

    _mcol, _rcol = st.columns([1.5, 1], gap="medium")
    with _mcol:
        sec("風險地圖熱力圖(顏色 = 預測空屋率)")
        _show_x = st.checkbox("疊加跨平台競品點位(Booking/591/ddroom)", value=False)
        _figH = px.density_mapbox(
            NB, lat="latitude", lon="longitude", z="vac_pred",
            radius=26, zoom=13.8, height=470,
            color_continuous_scale=["#5B9E73", "#F7D774", "#C4645A"],
            hover_data={"id": True, "vac_pred": ":.0%", "price": ":,.0f"})
        _figH.add_trace(go.Scattermapbox(
            lat=[_blat], lon=[_blon], mode="markers",
            marker=dict(size=17, color="#2A2A2A"),
            name="我的房源", hovertext=[f"#{bid}"]))
        if _show_x:
            try:
                _xc = comp_index().query(_blat, _blon, radius_m=float(radius))
                _xc = _xc[_xc["platform"] != "Airbnb"]
                for _pl, _g in _xc.groupby("platform"):
                    _figH.add_trace(go.Scattermapbox(
                        lat=_g["lat"], lon=_g["lon"], mode="markers",
                        marker=dict(size=8, color=_PLAT_COLORS[_pl]),
                        name=_pl,
                        hovertext=_g["title"].astype(str).str.slice(0, 26)))
            except FileNotFoundError:
                st.caption("競品索引未建置。")
        _figH.update_layout(mapbox_style="carto-positron",
                            margin=dict(l=0, r=0, t=0, b=0),
                            coloraxis_colorbar=dict(title="空屋率"))
        st.plotly_chart(_figH, use_container_width=True)
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

    # 跨平台補充
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
