# -*- coding: utf-8 -*-
"""platform_detail.py — 跨平台掛牌詳情(平台卡的 hover 預覽 + 查看詳情彈窗)

與 listing_detail.py 的差別:
  listing_detail  對象是「單一房源」,有照片、AI 清晰度、評論、周遭 POI
  platform_detail 對象是「半徑內某平台的 N 筆掛牌」,是集合視角

資料限制(已實測):競品資料沒有可用的照片網址與逐筆評論,
故本模組不呈現照片與評論,改以價格分佈、設施覆蓋率與掛牌明細為主。
"""
from __future__ import annotations

import html as _html

import numpy as np
import pandas as pd
import streamlit as st

from modules.geo_utils import nearest_address
from modules.ui_components import P, html_table, sec

_DIALOG = getattr(st, "dialog", None) or getattr(st, "experimental_dialog", None)

# 平台顯示名稱與代表色
PLATFORM = {
    "Airbnb": {"name": "Airbnb", "color": "#C4645A", "unit": "晚"},
    "Booking": {"name": "Booking.com", "color": "#2563EB", "unit": "晚"},
    "591": {"name": "591租屋網", "color": "#8B5CF6", "unit": "月"},
    "ddroom": {"name": "DD租租網", "color": "#4B4B4B", "unit": "月"},
}


def label(platform: str) -> str:
    return PLATFORM.get(platform, {}).get("name", platform)


def color(platform: str) -> str:
    return PLATFORM.get(platform, {}).get("color", P["muted"])


def _fmt(v, digits=0):
    try:
        f = float(v)
        return "—" if pd.isna(f) else f"{f:,.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def stats(sub: pd.DataFrame) -> dict:
    """單一平台在半徑內的統計。"""
    if sub.empty:
        return {"n": 0}
    pp = sub["price_pp_day"].dropna()
    return {
        "n": len(sub),
        "pp_median": float(pp.median()) if len(pp) else np.nan,
        "pp_q25": float(pp.quantile(.25)) if len(pp) else np.nan,
        "pp_q75": float(pp.quantile(.75)) if len(pp) else np.nan,
        "raw_median": float(sub["price_raw"].median()),
        "cap_median": float(sub["capacity"].median()),
        "dist_min": float(sub["dist_m"].min()),
        "dist_median": float(sub["dist_m"].median()),
        "rating": (float(sub["rating"].mean())
                   if sub["rating"].notna().any() else np.nan),
    }


def amenity_coverage(sub: pd.DataFrame, top: int = 8) -> pd.DataFrame:
    """該平台掛牌的設施覆蓋率。"""
    if sub.empty:
        return pd.DataFrame()
    keys = set()
    for s in sub["amenities"]:
        if isinstance(s, set):
            keys |= s
    rows = [{"設施": k,
             "具備比例": float(sub["amenities"].map(
                 lambda s: k in s if isinstance(s, set) else False).mean())}
            for k in keys]
    d = pd.DataFrame(rows).sort_values("具備比例", ascending=False).head(top)
    d["具備比例"] = d["具備比例"].map("{:.0%}".format)
    return d.reset_index(drop=True)


def render_platform(platform: str, sub: pd.DataFrame, radius_m: float,
                    my_pp: float | None = None):
    """平台掛牌詳情內容。"""
    info = PLATFORM.get(platform, {})
    name, col, unit = label(platform), color(platform), info.get("unit", "晚")
    s = stats(sub)

    st.markdown(
        f'<div style="font-size:1.05rem;font-weight:700;color:{col};'
        f'margin-bottom:4px;">{name}</div>'
        f'<div style="font-size:.8rem;color:{P["muted"]};line-height:1.9;">'
        f'📍 <b>比對範圍:</b>本房源半徑 {int(radius_m):,} 公尺內<br>'
        f'🗂 <b>掛牌筆數:</b>{s["n"]:,} 筆'
        f'{"｜最近 " + _fmt(s.get("dist_min")) + " 公尺" if s["n"] else ""}<br>'
        f'💰 <b>掛牌價中位:</b>${_fmt(s.get("raw_median"))} / {unit}'
        f'{"｜換算每人每晚 $" + _fmt(s.get("pp_median")) if s["n"] else ""}'
        f'</div>', unsafe_allow_html=True)

    if not s["n"]:
        st.info(f"此半徑內沒有 {name} 的掛牌資料,可放大比對半徑再試。")
        return

    # ── 指標 ──
    m = st.columns(4)
    m[0].metric("每人每晚中位", f"${_fmt(s['pp_median'])}",
                f"IQR ${_fmt(s['pp_q25'])}–${_fmt(s['pp_q75'])}",
                delta_color="off")
    if my_pp is not None and not np.isnan(s["pp_median"]):
        diff = my_pp - s["pp_median"]
        m[1].metric("與本房源差距", f"{'+' if diff >= 0 else ''}${_fmt(diff)}",
                    f"本房源 ${_fmt(my_pp)}", delta_color="off")
    else:
        m[1].metric("與本房源差距", "—")
    m[2].metric("可住人數中位", f"{_fmt(s['cap_median'])} 人")
    m[3].metric("平均評分",
                "—(此平台無評分)" if np.isnan(s["rating"]) else f"{s['rating']:.1f}",
                "0–10 分制" if not np.isnan(s["rating"]) else None,
                delta_color="off")

    # ── 價格分佈 ──
    sec("每人每晚價格分佈")
    import plotly.express as px
    from modules.ui_components import apply_theme
    fig = px.histogram(sub, x="price_pp_day", nbins=24,
                       color_discrete_sequence=[col],
                       labels={"price_pp_day": "每人每晚 (NT$)"})
    if my_pp is not None:
        fig.add_vline(x=float(my_pp), line_dash="dot", line_color=P["ink"],
                      annotation_text="本房源")
    apply_theme(fig, h=240).update_layout(yaxis_title="掛牌數")
    st.plotly_chart(fig, width="stretch")

    # ── 設施覆蓋率 ──
    cov = amenity_coverage(sub)
    if len(cov):
        sec("設施覆蓋率(此平台掛牌中具備該設施的比例)")
        html_table(cov, height=190, scroll=False)

    # ── 掛牌明細 ──
    sec(f"掛牌明細(依距離排序,共 {s['n']:,} 筆)")
    show = sub.sort_values("dist_m").head(60).copy()
    show["地址"] = [nearest_address(la, lo) or "—"
                    for la, lo in zip(show["lat"], show["lon"])]
    show["掛牌價"] = show.apply(
        lambda r: f"${r['price_raw']:,.0f}/"
                  f"{'月' if r['price_unit'] == 'month' else '晚'}", axis=1)
    show["每人每晚"] = show["price_pp_day"].map(lambda v: f"${v:,.0f}")
    show["距離"] = show["dist_m"].map(lambda v: f"{v:,.0f} m")
    show["可住"] = show["capacity"].map(lambda v: f"{v:,.0f} 人")
    show["房源"] = [
        f'<a href="{u}" target="_blank" style="color:{col};">'
        f'{_html.escape(str(t)[:30])} ↗</a>' if isinstance(u, str) and u.startswith("http")
        else _html.escape(str(t)[:30])
        for t, u in zip(show["title"], show["url"])]
    cols = ["房源", "地址", "掛牌價", "每人每晚", "可住", "距離"]
    if show["note"].notna().any():
        show["補充"] = show["note"].astype(str).str.slice(0, 24)
        cols.append("補充")
    html_table(show[cols], height=330, wrap=True)
    st.caption("補充欄位:Airbnb 為房型;Booking 為房型名稱;"
               "591 為房屋類型與坪數;租租網為房型、坪數與最短租期。"
               "月租平台的每人每晚為 ÷30 換算之等效價,未計押金與管理費。")


if _DIALOG:
    @_DIALOG("🗂 平台掛牌詳情")
    def _platform_dialog(platform, sub, radius_m, my_pp=None):
        render_platform(platform, sub, radius_m, my_pp)
else:
    def _platform_dialog(platform, sub, radius_m, my_pp=None):
        with st.expander("🗂 平台掛牌詳情", expanded=True):
            render_platform(platform, sub, radius_m, my_pp)


def open_platform(platform, sub, radius_m, my_pp=None):
    _platform_dialog(platform, sub, radius_m, my_pp)


def hover_card_html(platform: str, sub: pd.DataFrame, radius_m: float,
                    my_pp: float | None = None) -> str:
    """平台卡的滑鼠停留預覽(純 CSS)。"""
    name, col = label(platform), color(platform)
    s = stats(sub)
    if not s["n"]:
        body = f'<div style="font-size:.75rem;color:{P["muted"]};">' \
               f'此半徑內無 {name} 掛牌資料。</div>'
    else:
        near = sub.sort_values("dist_m").head(3)
        items = "".join(
            f'<div style="border-top:1px dashed {P["border"]};padding:4px 0;">'
            f'<div style="font-size:.72rem;color:{P["ink"]};white-space:nowrap;'
            f'overflow:hidden;text-overflow:ellipsis;">'
            f'{_html.escape(str(r["title"])[:26])}</div>'
            f'<div style="font-size:.68rem;color:{P["muted"]};">'
            f'${r["price_raw"]:,.0f}/{"月" if r["price_unit"] == "month" else "晚"}'
            f'・每人每晚 ${r["price_pp_day"]:,.0f}'
            f'・{r["dist_m"]:,.0f}m</div></div>'
            for _, r in near.iterrows())
        cmp_line = ""
        if my_pp is not None and not np.isnan(s["pp_median"]):
            diff = my_pp - s["pp_median"]
            cmp_line = (f'本房源 ${my_pp:,.0f},'
                        f'{"高於" if diff >= 0 else "低於"}此平台中位 '
                        f'<b style="color:{P["high"] if diff >= 0 else P["low"]}">'
                        f'${abs(diff):,.0f}</b><br>')
        body = (
            f'<div style="font-size:.73rem;color:{P["muted"]};line-height:1.8;">'
            f'🗂 半徑 {int(radius_m):,}m 內 <b>{s["n"]:,}</b> 筆掛牌<br>'
            f'💰 每人每晚中位 <b>${s["pp_median"]:,.0f}</b>'
            f'(IQR ${s["pp_q25"]:,.0f}–${s["pp_q75"]:,.0f})<br>'
            f'{cmp_line}'
            f'📍 最近 {s["dist_min"]:,.0f} 公尺</div>'
            f'<div style="margin-top:6px;font-size:.68rem;color:{P["muted"]};">'
            f'最近三筆</div>{items}')
    return f"""
<div class="hv-wrap">
  <span class="hv-anchor">ⓘ 滑鼠停留檢視摘要</span>
  <div class="hv-card">
    <div style="font-weight:700;font-size:.84rem;color:{col};
         margin-bottom:5px;">{name}</div>
    {body}
    <div style="font-size:.68rem;color:{P['muted']};margin-top:7px;
         border-top:1px dashed {P['border']};padding-top:5px;">
      完整掛牌明細、價格分佈與設施覆蓋率請按「查看詳情」</div>
  </div>
</div>"""
