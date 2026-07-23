# -*- coding: utf-8 -*-
"""房東入口 — 房東營運面板

分頁順序即使用順序,核心營運功能在前、模型診斷在後:
  房源總表   多房源優先序(體質 × 檔期四象限)· 空檔天數與機會成本
  房源定價情報   1km 內四平台每人每晚落點 · 同商圈同房型真實已訂率基準 · 周邊地圖
  未來檔期   逐日訂房熱度 · 月度 vs 同商圈 · 空檔警示 · 營收最適定價
  風險診斷   模型分數 · LIME 原因 · What-if(前瞻 AUC 0.632,僅供輔助)
  月報       彙整以上為可下載報告
"""
import html as _html

import numpy as np
import pandas as pd
import streamlit as st
from google import genai
from google.genai.errors import APIError

from modules.ui_components import (inject_css, P, ROOM_JP, sec, mb, note,
                                   sidebar_nav)
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
    # 選單僅顯示房東名稱與 ID;房源間數與高風險紅點不再出現在此。
    _hc = (PREDS.groupby(["host_id"])
           .agg(n=("id", "size"), host_name=("host_name", "first"))
           .sort_values("n", ascending=False).reset_index())
    _hlab = _hc.apply(lambda r: f"{r['host_name'] or '房東'}"
                                f"(ID {int(r['host_id'])})", axis=1)
    _hi = st.selectbox("host", range(len(_hc)), format_func=lambda i: _hlab[i],
                       label_visibility="collapsed")
    host_id = int(_hc.iloc[_hi]["host_id"])
    MY = PREDS[PREDS["host_id"] == host_id].reset_index(drop=True)

    # 全站統一口徑:所有風險值皆為 LightGBM(主力)之 GroupKFold OOF 誠實預測
    ALGO = "lgbm"
    PROB_COL = "prob"
    TIER_COL = "tier"

    # 房源總表卡片的篩選條件(僅影響 TB1 房源卡,不改變上方摘要與象限統計)
    # 篩選介面與租客入口一致:多選 multiselect,預設全選。
    st.markdown("#### 🔍 房源篩選")
    _dist_opts = sorted(MY["neighbourhood_cleansed"].dropna().astype(str).unique())
    DIST_PICK = st.multiselect("🗺 行政區（可複選）", _dist_opts,
                               default=_dist_opts, key="card_district")
    _room_opts = sorted(MY["room_type"].dropna().astype(str).unique())
    ROOM_PICK = st.multiselect("🛏 房型（可複選）", _room_opts, default=_room_opts,
                               key="card_room",
                               format_func=lambda v: ROOM_JP.get(v, v))

    st.divider()
    st.caption("LightGBM+XGBoost(Isotonic 校準)· 標籤 Y≥0.6 · "
               "LIME 可解釋 · GroupKFold 誠實驗證 · 59 特徵(含負評比例)")

st.markdown(f"""
<div style="padding:6px 0 10px;">
  <h1 style="font-size:1.4rem;font-weight:700;color:{P['ink']};margin:0;">
  房東營運面板</h1>
</div><hr style="margin:0 0 12px;">""", unsafe_allow_html=True)

# 側邊欄的行政區／房型篩選為全頁範圍:房源總表的統計與卡片、以及定價情報、
# 未來檔期、風險診斷、月報的房源選單皆以 SCOPE 為準(月報彙整同一範圍)。
SCOPE = MY[MY["neighbourhood_cleansed"].astype(str).isin(DIST_PICK)
           & MY["room_type"].astype(str).isin(ROOM_PICK)].reset_index(drop=True)
_filtered = len(SCOPE) != len(MY)
if SCOPE.empty:
    st.info("目前篩選條件下沒有符合的房源,請調整側邊欄的行政區或房型。")
    st.stop()

# 分頁順序 = 使用順序:先看該處理哪間,再看定價、檔期;
# 模型與 LIME 診斷退居後排(前瞻驗證 AUC 0.632,僅作輔助排序)。
TB1, TB2P, TB5, TB2, TB6 = st.tabs(
    ["🗂️ 房源總表", "💰 房源定價情報", "🗓️ 未來檔期",
     "⚠️ 風險診斷", "🧾 月報"])
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
    """顯示預估空房率與所屬行政區中位數的差距。"""
    district = _html.escape(str(row.get("neighbourhood_cleansed") or "所屬行政區"))
    try:
        d = float(row["vac_pred"]) - float(row["dist_med"])
    except (TypeError, ValueError):
        d = np.nan
    if not np.isfinite(d):
        return ("<span class='listing-card-comparison listing-card-comparison-flat'>"
                "暫無行政區空房率基準</span>")
    if d > 0.02:
        return ("<span class='listing-card-comparison listing-card-comparison-high'>"
                f"<span>高於{district}中位數</span>"
                f"<span>空房率 {d*100:.0f} 百分比</span></span>")
    if d < -0.02:
        return ("<span class='listing-card-comparison listing-card-comparison-low'>"
                f"<span>低於{district}中位數</span>"
                f"<span>空房率 {abs(d)*100:.0f} 百分比</span></span>")
    return ("<span class='listing-card-comparison listing-card-comparison-flat'>"
            f"<span>與{district}中位數</span><span>空房率持平</span></span>")


def platform_card_html(platform, stats, sub, my_pp, radius_m):
    """跨平台價格卡:平台中位每人每晚 + 與本房源的價差 + 對比長條。

    長條以「我的」與「平台中位」共用同一基準(兩者較大值)呈現,
    四張卡的長條因此可橫向互相比較。

    四張卡刻意不做平台代表色區分,配色一律沿用 .pf-* 的統一樣式;
    唯一保留的色彩訊號是「我的便宜／我的貴」徽章(綠/紅)。
    """
    from modules import platform_detail as PD
    head = (f'<div class="pf-head"><span class="pf-dot"></span>'
            f'{_html.escape(PD.label(platform))}'
            f'<span class="pf-count">{int(stats["count"]) if stats else 0} 筆</span>'
            f'</div>')
    if not stats or not stats.get("count"):
        return (f'<div class="pf-card">{head}'
                f'<div class="pf-empty">此半徑內無掛牌資料</div></div>')
    med = float(stats["pp_median"])
    diff = my_pp / med - 1 if med else np.nan
    if not np.isfinite(diff) or abs(diff) < .02:
        delta = '<span class="pf-delta pf-delta-flat">與中位持平</span>'
    elif diff < 0:
        delta = (f'<span class="pf-delta pf-delta-low">我的便宜 '
                 f'{abs(diff)*100:.0f}%</span>')
    else:
        delta = (f'<span class="pf-delta pf-delta-high">我的貴 '
                 f'{diff*100:.0f}%</span>')
    base = max(my_pp, med, 1)
    bars = (
        f'<div class="pf-bars">'
        f'<div class="pf-bar-row"><span>我的</span>'
        f'<span class="pf-bar"><i style="width:{my_pp/base*100:.0f}%;'
        f'background:{P["ink2"]};"></i></span>'
        f'<span class="pf-bar-val">${my_pp:,.0f}</span></div>'
        f'<div class="pf-bar-row"><span>中位</span>'
        f'<span class="pf-bar"><i style="width:{med/base*100:.0f}%;'
        f'background:{P["primary"]};"></i></span>'
        f'<span class="pf-bar-val">${med:,.0f}</span></div></div>')
    return (f'<div class="pf-card">{head}'
            f'<div class="pf-value">${med:,.0f}'
            f'<span class="pf-unit">中位每人每晚</span></div>{delta}{bars}'
            f'{PD.hover_card_html(platform, sub, radius_m, my_pp)}</div>')


def quadrant_summary_table(summary_df):
    """以狀態色、清楚欄寬與大字級呈現象限決策摘要。"""
    rows = []
    for quadrant, meta in sorted(
            QD.QUADRANTS.items(), key=lambda item: item[1]["priority"]):
        matched = summary_df[summary_df["象限"] == meta["label"]]
        if matched.empty:
            continue
        row = matched.iloc[0]
        status = meta["label"].split(" ", 1)[-1]
        color = P[meta["color"]]
        rows.append(f"""
        <tr style="--quadrant-color:{color};">
          <td><span class="quadrant-status"><span class="quadrant-status-dot"
              aria-hidden="true"></span>{_html.escape(status)}</span></td>
          <td><span class="quadrant-count">{int(row['房源數'])}</span>
              <span class="quadrant-count-unit">間</span></td>
          <td>{_html.escape(str(row['說明']))}</td>
          <td><span class="quadrant-action">{_html.escape(str(row['建議行動']))}</span></td>
        </tr>""")
    st.markdown(f"""
    <div class="quadrant-table-wrap">
      <table class="quadrant-table" aria-label="房源營運象限摘要">
        <colgroup><col style="width:17%"><col style="width:10%">
          <col style="width:36%"><col style="width:37%"></colgroup>
        <thead><tr><th scope="col">營運狀態</th><th scope="col">房源數</th>
          <th scope="col">狀況判讀</th><th scope="col">建議行動</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TB1 房東總覽
# ══════════════════════════════════════════════════════════════
with TB1:
    if _filtered:
        _dist_txt = "、".join(DIST_PICK) if DIST_PICK else "未選行政區"
        _room_txt = ("、".join(ROOM_JP.get(v, v) for v in ROOM_PICK)
                     if ROOM_PICK else "未選房型")
        st.caption(f"目前篩選:{_dist_txt} · {_room_txt}"
                   f"({len(SCOPE)}／{len(MY)} 間)")
    _gapd = SCOPE["gap_days_30d"].fillna(0)
    _alarm = int((SCOPE["quadrant"] == "alarm").sum())
    _alarm_all = int((MY["quadrant"] == "alarm").sum())
    k1, k2, k3, k4, k5 = st.columns(5)
    _overview = [
        ("篩選後房源" if _filtered else "名下房源",
         f"{len(SCOPE)}／{len(MY)} 間" if _filtered else f"{len(MY)} 間"),
        ("近 30 天訂房率", "—" if SCOPE["booked_rate_d30"].isna().all()
         else f"{SCOPE['booked_rate_d30'].mean()*100:.0f}%"),
        ("30 天空檔", f"{int(_gapd.sum())} 天"),
        ("需優先處理", f"{_alarm}／{_alarm_all} 間" if _filtered else f"{_alarm} 間"),
        ("平均預測空屋率", "—" if SCOPE["vac_pred"].isna().all()
         else f"{SCOPE['vac_pred'].mean()*100:.0f}%"),
    ]
    for _col, (_label, _value) in zip((k1, k2, k3, k4, k5), _overview):
        _col.markdown(
            f'<div class="overview-metric"><div class="overview-metric-label">'
            f'{_label}</div><div class="overview-metric-value">{_value}</div></div>',
            unsafe_allow_html=True)
    st.markdown(f"""
    <div style="margin:20px 0 8px;font-size:1.15rem;font-weight:700;
         color:{P['ink']};letter-spacing:.01em;">
      模型預估與90實際訂房分析
    </div>""", unsafe_allow_html=True)
    _qs = QD.summary(SCOPE)
    if not SCOPE.empty:
        quadrant_summary_table(_qs)

    _qopts = ["全部"] + _qs["象限"].tolist()
    _qpick = st.radio("篩選象限", _qopts, horizontal=True, key="q_filter")
    _cards = SCOPE.copy()
    if _qpick != "全部":
        _cards = _cards[_cards["quadrant_label"] == _qpick]
    sec(f"房源卡片({len(_cards)} 間)")
    if _cards.empty:
        st.info("目前篩選條件下沒有符合的房源,請調整側邊欄的行政區或房型。")
    _cards = _cards.sort_values(["quadrant_priority", PROB_COL],
                                ascending=[True, False]).reset_index(drop=True)
    from modules.geo_utils import nearest_address
    from modules import listing_detail as LD
    for _s in range(0, len(_cards), 3):
        cols = st.columns(3, gap="medium")
        for _c, (_, r) in zip(cols, _cards.iloc[_s:_s + 3].iterrows()):
            _, t_c = TIER_ZH[r[TIER_COL]]
            with _c:
                _name = _html.escape(str(r["name"]))
                _district = _html.escape(str(r.get("neighbourhood_cleansed") or "—"))
                _room = _html.escape(ROOM_JP.get(r.get("room_type"), r.get("room_type", "—")))
                _photo_url = str(r.get("picture_url", "") or "").strip()
                _photo = (
                    f'<img class="listing-card-photo" src="{_html.escape(_photo_url, quote=True)}" '
                    f'alt="{_name} 房源封面照片" loading="lazy" referrerpolicy="no-referrer">'
                    if _photo_url.startswith("http") else
                    '<div class="listing-card-photo-empty">暫無房源照片</div>')
                _addr = nearest_address(float(r["latitude"]), float(r["longitude"])) or "—"
                _addr = _html.escape(_addr)
                _capacity = pd.to_numeric(r.get("accommodates"), errors="coerce")
                _capacity_txt = "—" if pd.isna(_capacity) else f"可住 {int(_capacity)} 人"
                _booked90 = pd.to_numeric(r.get("booked_rate_d90"), errors="coerce")
                _unbooked90 = "—" if pd.isna(_booked90) else f"{(1 - float(_booked90)):.0%}"
                with st.container(border=True):
                    st.markdown(
                        f'<div class="listing-card-accent" style="background:{t_c};"></div>'
                        f'<div class="listing-card-id">房源 #{int(r["id"])}</div>'
                        f'<div class="listing-card-title" title="{_name}">{_name}</div>',
                        unsafe_allow_html=True)
                    _photo_col, _risk_col = st.columns([1.12, .88], gap="medium",
                                                       vertical_alignment="top")
                    with _photo_col:
                        st.markdown(f"""
{_photo}
<div class="listing-card-meta">
  <div class="listing-card-meta-row"><span class="listing-card-meta-key">行政區</span>
    <span class="listing-card-meta-value">{_district}</span></div>
  <div class="listing-card-meta-row"><span class="listing-card-meta-key">推估地址</span>
    <span class="listing-card-meta-value" title="{_addr}">{_addr}</span></div>
  <div class="listing-card-meta-row"><span class="listing-card-meta-key">每晚價格</span>
    <span class="listing-card-meta-value listing-card-price">NT$ {float(r['price']):,.0f}</span></div>
  <div class="listing-card-meta-row"><span class="listing-card-meta-key">房型</span>
    <span class="listing-card-meta-value">{_room} · {_capacity_txt}</span></div>
</div>""", unsafe_allow_html=True)
                    with _risk_col:
                        st.markdown(f"""
<div class="listing-card-risk" aria-label="預估空房率與訂房狀態">
  <div class="listing-card-risk-label">預估空房率</div>
  <div class="listing-card-ring">{risk_ring(float(r['vac_pred']), t_c, size=122)}</div>
  {trend_arrow(r)}
  <div class="listing-card-calendar">90 天實際未訂房率<strong>{_unbooked90}</strong></div>
</div>""", unsafe_allow_html=True)
                    _detail = DF_RAW[DF_RAW["id"] == int(r["id"])]
                    if st.button("查看詳情", key=f"listing_detail_{int(r['id'])}",
                                 width="stretch"):
                        if len(_detail):
                            LD.open_detail(_detail.iloc[0], show_actions=False)
                        else:
                            st.warning("此房源的詳細資料暫時無法載入。")

# ── 詳情頁共用:選擇房源(僅列出符合側邊欄篩選的房源) ──
_opts = SCOPE.sort_values(PROB_COL, ascending=False)
# 選單只顯示房源名稱;同名房源才補上 ID 以便區分。
_opt_lab = {}
for _, _r in _opts.iterrows():
    _nm = "" if pd.isna(_r["name"]) else str(_r["name"]).strip()
    _lbl = _nm or f"未命名房源 #{int(_r['id'])}"
    if _lbl in _opt_lab:
        _lbl = f"{_lbl}(ID {int(_r['id'])})"
    _opt_lab[_lbl] = int(_r["id"])

# ── 定價情報頁首:選擇房源 + 附近比較半徑 ──
# 這段先於其他分頁執行,讓 radius 在所有引用它的分頁(定價情報、風險診斷的
# 跨平台比較)都取得同一個使用者設定值;版面仍渲染在「房源定價情報」分頁最上方。
with TB2P:
    # 第三欄為留白,讓半徑滑桿往中間靠而不是貼齊右緣
    _c_sel, _c_rad, _ = st.columns([1.5, 1, .6], vertical_alignment="bottom")
    with _c_sel:
        _sel3 = st.selectbox("選擇房源", list(_opt_lab.keys()), key="nb_sel")
    with _c_rad:
        radius = st.slider("📏 附近比較半徑 (公尺)", 500, 2000, 1000, step=100,
                           key="cmp_radius")
bid = _opt_lab[_sel3]
B = MY[MY["id"] == bid].iloc[0]
_blat, _blon = float(B["latitude"]), float(B["longitude"])

# ══════════════════════════════════════════════════════════════
# TB2 房源詳情(大分數 + LIME Top3 + 建議 + 趨勢線)
# ══════════════════════════════════════════════════════════════


with TB2:
    _sel = st.selectbox("選擇房源", list(_opt_lab.keys()))
    sel_id = _opt_lab[_sel]
    R = MY[MY["id"] == sel_id].iloc[0]
    ROW = DS_IDX.loc[sel_id] if sel_id in DS_IDX.index else None

    # ── 基準值(OOF 誠實預測):與總覽/下拉/通知中心口徑一致 ──
    _vac0 = float(R["vac_pred"])
    _prob0 = float(R[PROB_COL])
    _tier0 = R[TIER_COL]
    _variant = R.get("variant", "full")

    # ── What-if 現值 ──
    # LIME 區塊排在 What-if 控制項「上方」,但要用到滑桿的當前值,因此改以
    # session_state key 綁定 widget:本次 rerun 開頭先讀值給 LIME,滑桿本身
    # 仍渲染在下方左欄;任何拖動都會觸發 rerun,上方 LIME 隨即同步。
    _price0 = float(R["price"])
    _mn_now = pd.to_numeric(R["minimum_nights"], errors="coerce")
    _mn0 = 1 if pd.isna(_mn_now) else int(np.clip(_mn_now, 1, 30))
    _k_price, _k_mn = f"wi_price_{sel_id}", f"wi_mn_{sel_id}"
    st.session_state.setdefault(_k_price, int(np.clip(_price0, 500, 50000)))
    st.session_state.setdefault(_k_mn, _mn0)
    if ROW is not None:
        _np_ = int(st.session_state[_k_price])
        _nm = int(st.session_state[_k_mn])
        _changed = (abs(_np_ - _price0) > 1) or (_nm != _mn0)
    else:
        _np_, _nm, _changed = _price0, _mn0, False

    # ── LIME 原因 Top 3(全寬置頂) ──
    sec("LIME 原因 Top 3(為什麼有風險)")
    mb("LIME 局部線性近似 · 解釋模型 B 的 P(空屋率≥60%) · 正值=推高風險")
    if ROW is None:
        st.info("此房源不在訓練協定內(經營未滿一年),無法提供 LIME 解釋。")
        _lime_up = []
    else:
        try:
            from modules.lime_explainer import lime_reasons
            _ov = {"price": _np_, "minimum_nights": _nm}
            _lime = lime_reasons(ROW, _variant, ALGO, overrides=_ov, k=3)
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

    cA, cB = st.columns([1, 1.5], gap="large")
    with cA:
        # ── What-if 控制項(先操作、再看結果) ──
        if ROW is not None:
            sec("⚡ What-if 模擬(拖動後下方風險環與上方 LIME 即時重算)")
            st.slider("每晚房價 (NT$)", 500, 50000, step=100, key=_k_price)
            st.number_input("最低入住天數(晚)", 1, 30, key=_k_mn)

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

# ══════════════════════════════════════════════════════════════
# TB3 附近比較(熱力圖 + 風險比較 + 同商圈排名 + 跨平台)
# ══════════════════════════════════════════════════════════════
with TB3:
    # 選擇房源與比較半徑已於頁首渲染(見 _opt_lab 之後的 TB2P 區塊)
    # ── 跨平台定價落點(主打:Airbnb 後台看不到的資訊)──
    # 依使用者指示不再顯示區塊標題與換算方式說明,畫面直接進入左右雙欄。
    try:
        _cs0 = comp_index().stats(
            _blat, _blon,
            listing_pp_day=float(B["price"]) / max(float(B["accommodates"]), 1),
            bracket=capacity_bracket(B["accommodates"]),
            radius_m=float(radius), exclude_listing_id=int(bid))
        _my_pp = float(B["price"]) / max(float(B["accommodates"]), 1)
        from modules import listing_detail as LD
        from modules import platform_detail as PD
        _allc = _cs0.get("competitors")
        _plats = []
        for _pl in ["Airbnb", "Booking", "591", "ddroom"]:
            _plats.append((_pl, _cs0["platforms"].get(_pl),
                           _allc[_allc["platform"] == _pl]
                           if _allc is not None and len(_allc) else pd.DataFrame()))
        # 左「房源」右「跨平台價格」雙欄:左欄是固定高 HTML 區塊(--pricing-h)
        # ＋一顆「查看完整詳情」;右欄是錨點列 ＋ 2×2 的「卡片＋該平台按鈕」。
        # 樣式見 ui_components 的 .pricing-* / .pf-*;--pricing-h 尚未依新版
        # 右欄高度重新量測校正,兩欄底部目前不保證切齊。
        _c_photo, _c_price = st.columns([1.02, .98], gap="medium")
        with _c_photo:
            _BD = DF_RAW[DF_RAW["id"] == int(bid)]
            _BD = _BD.iloc[0] if len(_BD) else None
            _purl = str((_BD.get("picture_url", "") if _BD is not None else "") or "")
            _pimg = (
                f'<img class="pricing-photo" src="{_html.escape(_purl, quote=True)}" '
                f'alt="房源封面照片" loading="lazy" referrerpolicy="no-referrer">'
                if _purl.startswith("http") else
                '<div class="pricing-photo pricing-photo-empty">暫無房源照片</div>')
            st.markdown(
                f'<div class="pricing-pane pricing-left">{_pimg}'
                f'{LD.summary_html(_BD, show_name=False) if _BD is not None else ""}'
                f'</div>', unsafe_allow_html=True)
            if _BD is None:
                st.caption("此房源的詳細資料暫時無法載入。")
            elif st.button("查看完整詳情", key=f"detail_price_{bid}",
                           width="stretch"):
                # 房東視角不需要「立即租房 / 加入收藏」(那是租客動作)
                LD.open_detail(_BD, show_actions=False)
        with _c_price:
            # hover 樣式由 listing_detail 提供,整段只需注入一次
            st.markdown(LD.HOVER_CSS + f"""
<div class="pricing-anchor">
  <span class="pricing-anchor-label">我的每人每晚</span>
  <span class="pricing-anchor-value">${_my_pp:,.0f}</span>
  <span class="pricing-anchor-sub">每晚 ${float(B['price']):,.0f} ·
    可住 {int(float(B['accommodates']))} 人</span>
</div>""", unsafe_allow_html=True)
            # 每個平台自成一格(卡片 + 該平台的查詢按鈕),2×2 排列。
            # Streamlit 按鈕無法寫進 st.markdown 的 HTML,只能緊接在卡片下方。
            for _row in (_plats[:2], _plats[2:]):
                for _col, (_pl, _v, _sub) in zip(
                        st.columns(2, gap="small"), _row):
                    with _col:
                        st.markdown(
                            platform_card_html(_pl, _v, _sub, _my_pp, radius),
                            unsafe_allow_html=True)
                        if st.button(f"🔍 {PD.label(_pl)}",
                                     key=f"pf_{_pl}_{bid}", width="stretch"):
                            PD.open_platform(_pl, _sub, radius, _my_pp)
        # ── 定價建議(單一區塊,三點:競品概況 → 價格落點 → 設施缺口) ──
        # 原本散在雙欄下方與分頁最末的三段說明,依使用者指示合併於此。
        sec("定價建議")
        _pts = []

        # 1) 跨平台競品概況
        _pl_txt = " · ".join(
            f"{p} {v['count']} 筆(每人每晚中位 ${v['pp_median']:,.0f})"
            for p, v in _cs0["platforms"].items())
        _pts.append(f"<b>跨平台競品({radius}m)</b>:{_pl_txt or '無'}")

        # 2) 價格落點
        _pp0 = _cs0.get("pp_percentile")
        _brk = B['bracket'] if 'bracket' in B else capacity_bracket(B['accommodates'])
        if _pp0 is not None:
            _pcol = (P["high"] if _pp0 >= .6 else
                     (P["medium"] if _pp0 >= .4 else P["low"]))
            _pts.append(
                f"<b>價格落點</b>:同容量層({_brk})共 {_cs0['n_same_bracket']} 筆競品,"
                f"你的每人每晚 <b style='color:{_pcol}'>貴於 {_pp0:.0%}</b>,"
                f"同層中位為 <b>${_cs0['bracket_pp_median']:,.0f}</b>"
                f"(月租÷30 換算)。")
        else:
            _pts.append("<b>價格落點</b>:同容量層競品不足 5 筆,暫不計算。")

        # 3) 設施缺口(跨平台競品過半具備、本房源沒有)
        _own0 = _canon_amenities(str(B.get("amenities", "")))
        _gap0 = [(k, v) for k, v in _cs0.get("amenity_coverage", {}).items()
                 if v >= .5 and k not in _own0][:4]
        if _gap0:
            _pts.append("<b>設施缺口</b>:" + "、".join(
                f"<b>{k}</b>(周邊 {v:.0%} 具備)" for k, v in _gap0)
                + " —— 這些是同商圈的標配,建議優先補齊。")
        else:
            _pts.append("<b>設施缺口</b>:周邊過半競品具備的設施你都有,無明顯缺口。")

        note("<br>".join(f"{i}. {t}" for i, t in enumerate(_pts, 1)))
    except FileNotFoundError:
        st.caption("競品索引未建置,請執行 scripts/train_backend_models.py。")

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
