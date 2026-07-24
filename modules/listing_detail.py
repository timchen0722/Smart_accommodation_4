# -*- coding: utf-8 -*-
"""listing_detail.py — 房源詳情共用元件

原本只存在於租客入口內部,現抽出為共用模組,讓房東入口的「查看詳情」
與租客入口顯示<b>完全相同</b>的內容(避免兩邊各自維護而內容分歧)。

提供
----
render_detail(L)       詳情內容(彈窗或展開區塊皆可用)
open_detail(L, key)    以按鈕開啟彈窗(自動相容舊版 Streamlit 的 expander)
hover_card_html(L)     滑鼠停留即顯示的預覽卡(純 CSS,無需 rerun)
"""
from __future__ import annotations

import ast
import html as _html

import pandas as pd
import streamlit as st

from modules.data_loader import load_reviews
from modules.geo_utils import (POI_NAMES, load_all_poi, nearest_address,
                               poi_points_within)
from modules.image_analysis import analyze, listing_photos, zh_amenity
from modules.nlp_analysis import recent_review_snippets
from modules.ui_components import P, html_table, sec

_DIALOG = getattr(st, "dialog", None) or getattr(st, "experimental_dialog", None)


@st.cache_data(show_spinner=False)
def _img(url: str):
    return analyze(url)


@st.cache_resource(show_spinner=False)
def _poi():
    return load_all_poi()


@st.cache_data(show_spinner=False)
def _reviews():
    return load_reviews()


def _amenities(raw) -> list:
    """解析 amenities 欄位(JSON 樣式字串)為清單。"""
    try:
        items = ast.literal_eval(raw) if isinstance(raw, str) else []
        return [str(a) for a in items if str(a).strip()]
    except (ValueError, SyntaxError):
        return []


def _gi(v, d=0) -> int:
    try:
        f = float(v)
        return d if pd.isna(f) else int(f)
    except (TypeError, ValueError):
        return d


def summary_html(L, show_name: bool = True) -> str:
    """房源基本資訊區塊(名稱、位置、推估地址、房型與價格)。

    詳情彈窗與其他頁面(如房源定價情報)共用同一份文案與樣式。
    """
    lat, lon = float(L["latitude"]), float(L["longitude"])
    rating = L.get("review_scores_rating")
    rating_s = f"{float(rating):.2f}" if pd.notna(rating) else "N/A"
    room_zh = L.get("room_type_zh") or L.get("room_type") or ""
    addr = nearest_address(lat, lon)
    hood = L.get("neighbourhood", "")
    hood = str(hood) if pd.notna(hood) else ""
<<<<<<< HEAD
    name = (f'<div style="font-size:var(--sa-text-card-title);font-weight:700;color:{P["ink"]};'
=======
    name = (f'<div style="font-size:1.05rem;font-weight:700;color:{P["ink"]};'
>>>>>>> a7faf33a8d684411d0b07457665e38d0f6d4d906
            f'margin-bottom:6px;">{_html.escape(str(L["name"]))}</div>'
            if show_name else "")
    # 注意:回傳字串不可有縮排或空行 —— 內嵌到其他 HTML 區塊時,
    # 縮排 4 個空白會被 Markdown 當成程式碼區塊而把原始碼直接印出來。
    return (
        f'{name}'
<<<<<<< HEAD
        f'<div style="font-size:var(--sa-text-caption);color:{P["muted"]};line-height:1.9;">'
=======
        f'<div style="font-size:.8rem;color:{P["muted"]};line-height:1.9;">'
>>>>>>> a7faf33a8d684411d0b07457665e38d0f6d4d906
        f'📍 <b>區域位置：</b>{L.get("neighbourhood_cleansed", "")}'
        f'{("｜" + hood) if hood else ""}<br>'
        f'🏠 <b>推估地址：</b>{addr or "—"}<br>'
        f'🧭 座標：{lat:.5f}, {lon:.5f}<br>'
        f'🛏 {room_zh} ｜ 👥 可住 {_gi(L.get("accommodates"))} 人 ｜ '
        f'🛁 {_gi(L.get("bathrooms_count"))} 衛浴 ｜ '
        f'🛏 {_gi(L.get("beds"))} 床<br>'
<<<<<<< HEAD
        f'💰 <b style="color:{P["tenant"]};font-size:var(--sa-text-card-title);">'
=======
        f'💰 <b style="color:{P["tenant"]};font-size:1.05rem;">'
>>>>>>> a7faf33a8d684411d0b07457665e38d0f6d4d906
        f'${float(L["price"]):,.0f}</b> / 晚 ｜ ⭐ {rating_s} ｜ '
        f'💬 {_gi(L.get("number_of_reviews"))} 則評論'
        f'</div>')


def render_summary(L, show_name: bool = True):
    """封面照片 + 基本資訊(不含設施、周遭機能與評論),供頁面內嵌重用。"""
    url = str(L.get("picture_url", "") or "")
    if url.startswith("http"):
        st.image(url, width="stretch")
    else:
        st.caption("暫無房源照片")
    st.markdown(summary_html(L, show_name=show_name), unsafe_allow_html=True)


def render_detail(L, show_actions: bool = True):
    """房源詳情內容(與租客入口一致)。"""
    lat, lon = float(L["latitude"]), float(L["longitude"])
    st.markdown(summary_html(L), unsafe_allow_html=True)

    # ── 封面照片 + AI 清晰度 ──
    url = str(L.get("picture_url", "") or "")
    if url.startswith("http"):
        c1, c2 = st.columns([1, 1])
        with c1:
            st.image(url, width="stretch", caption="房源封面照片")
            photos = listing_photos(L)
            links = (" ｜ ".join(
                f'<a href="{u}" target="_blank" style="color:{P["tenant"]};">'
                f'圖{i + 1} ↗</a>' for i, u in enumerate(photos))
                if len(photos) > 1 else
                f'<a href="{url}" target="_blank" style="color:{P["tenant"]};'
                f'font-weight:700;">🖼 開新視窗檢視原圖 ↗</a>')
            st.markdown(links, unsafe_allow_html=True)
        with c2:
            im = _img(url)
            if im.get("ok"):
                lab = im["label"]
                col = (P["low"] if lab == "清晰"
                       else (P["medium"] if lab == "尚可" else P["high"]))
                st.markdown(
                    f'<div style="background:{P["surface"]};border:1px solid '
                    f'{P["border"]};border-top:3px solid {col};border-radius:var(--sa-radius-sm);'
                    f'padding:12px 14px;">'
                    f'<div style="font-size:var(--sa-text-label);color:{P["muted"]};">'
                    f'AI 照片清晰度</div>'
                    f'<div style="font-size:var(--sa-text-metric);font-weight:800;color:{col};">'
                    f'{lab}</div>'
                    f'<div style="font-size:var(--sa-text-caption);color:{P["muted"]};">清晰機率 '
                    f'{im["prob"] * 100:.0f}%</div></div>', unsafe_allow_html=True)
                m = st.columns(2)
                m[0].metric("Laplacian", f"{im['raw']['laplacian_var']:,.0f}")
                m[1].metric("解析度", f"{im['raw']['megapixels']} MP")
                if lab == "模糊":
                    st.caption("⚠️ 封面照片偏模糊,實際看房請多加留意。")
            else:
                st.caption("(無法下載照片進行分析)")

    if show_actions:
        a = st.columns(2)
        if a[0].button("🏠 立即租房", key=f"rent_dlg_{L['id']}",
                       width="stretch"):
            st.toast(f"✅ 已送出「{str(L['name'])[:16]}」租房申請!", icon="🏠")
        if a[1].button("❤️ 加入收藏", key=f"fav_dlg_{L['id']}",
                       width="stretch"):
            st.toast("❤️ 已加入收藏清單!", icon="❤️")

    # ── 設施 ──
    ams = _amenities(L.get("amenities", "[]"))
    sec(f"房源設施(共 {len(ams)} 項)")
    if ams:
        chips = "".join(
            f'<span style="display:inline-block;background:{P["tag_bg"]};'
            f'border:1px solid {P["border"]};border-radius:var(--sa-radius-md);padding:3px 11px;'
            f'margin:3px 4px 0 0;font-size:var(--sa-text-caption);color:{P["ink2"]};">'
            f'{_html.escape(zh_amenity(a))}</span>' for a in ams)
        st.markdown(f"<div style='line-height:2.1;'>{chips}</div>",
                    unsafe_allow_html=True)
    else:
        st.caption("此房源未列出設施資料。")

    # ── 周遭設施 ──
    sec("周遭附近設施(1KM 範圍)")
    rows = []
    for t, pdf in _poi().items():
        pts = poi_points_within(lat, lon, pdf, 1000)
        if len(pts):
            n = pts.iloc[0]
            rows.append({"設施類型": POI_NAMES[t], "1KM 數量": len(pts),
                         "最近地點": n["poi_name"],
                         "最近距離": f"{n['distance_m']:.0f} m"})
        else:
            rows.append({"設施類型": POI_NAMES[t], "1KM 數量": 0,
                         "最近地點": "—", "最近距離": "—"})
    html_table(pd.DataFrame(rows), fmt={"1KM 數量": "{:,.0f}"}, height=280)

    # ── 近期評論 ──
    sec("近期評論")
    snips = recent_review_snippets(_reviews(), L["id"], n=8)
    if snips:
        items = "".join(f'<div class="rv-item">{_html.escape(str(x))}</div>'
                        for x in snips)
        st.markdown(
            f'<div style="max-height:220px;overflow-y:auto;border:1px solid '
            f'{P["border"]};border-radius:var(--sa-radius-sm);padding:8px 12px;font-size:var(--sa-text-caption);'
            f'color:{P["ink2"]};line-height:1.6;">{items}</div>',
            unsafe_allow_html=True)
    else:
        st.caption("此房源尚無評論。")


if _DIALOG:
    @_DIALOG("🏠 房源詳情")          # 不指定寬度 = 與租客入口相同
    def _detail_dialog(L, show_actions=True):
        render_detail(L, show_actions)
else:                                    # 舊版 Streamlit 後備
    def _detail_dialog(L, show_actions=True):
        with st.expander("🏠 房源詳情", expanded=True):
            render_detail(L, show_actions)


def open_detail(L, show_actions: bool = True):
    """開啟房源詳情彈窗。"""
    _detail_dialog(L, show_actions)


def hover_card_html(L, extra_lines: list[str] | None = None) -> str:
    """滑鼠停留即顯示的預覽卡(純 CSS,不觸發 rerun)。

    Streamlit 的彈窗必須由按鈕點擊觸發(無 hover 事件),
    因此 hover 顯示此預覽卡,完整內容仍由「查看詳情」按鈕開啟。
    """
    lat, lon = float(L["latitude"]), float(L["longitude"])
    rating = L.get("review_scores_rating")
    rating_s = f"{float(rating):.2f}" if pd.notna(rating) else "N/A"
    addr = nearest_address(lat, lon) or "—"
    url = str(L.get("picture_url", "") or "")
    img = (f'<img src="{url}" referrerpolicy="no-referrer" '
           f'style="width:100%;height:104px;object-fit:cover;border-radius:var(--sa-radius-sm);'
           f'margin-bottom:8px;background:{P["tag_bg"]};" '
           f'onerror="this.style.display=\'none\'">'
           if url.startswith("http") else "")
    ams = _amenities(L.get("amenities", "[]"))
    chips = "".join(
        f'<span style="display:inline-block;background:{P["tag_bg"]};'
        f'border-radius:var(--sa-radius-sm);padding:1px 7px;margin:2px 3px 0 0;'
        f'font-size:var(--sa-text-label);color:{P["ink2"]};">{_html.escape(zh_amenity(a))}</span>'
        for a in ams[:6])
    extra = "".join(f"<div>{x}</div>" for x in (extra_lines or []))
    return f"""
<div class="hv-wrap">
  <span class="hv-anchor">ⓘ 滑鼠停留檢視摘要</span>
  <div class="hv-card">
    {img}
    <div style="font-weight:700;font-size:var(--sa-text-body);color:{P['ink']};
         line-height:1.4;margin-bottom:5px;">{_html.escape(str(L['name']))[:44]}</div>
    <div style="font-size:var(--sa-text-caption);color:{P['muted']};line-height:1.8;">
      📍 {L.get('neighbourhood_cleansed', '')}｜{L.get('room_type_zh') or L.get('room_type', '')}<br>
      🏠 {addr}<br>
      🧭 {lat:.5f}, {lon:.5f}<br>
      👥 可住 {_gi(L.get('accommodates'))} 人 ｜ 🛁 {_gi(L.get('bathrooms_count'))} 衛浴
        ｜ 🛏 {_gi(L.get('beds'))} 床<br>
      💰 <b style="color:{P['tenant']};">${float(L['price']):,.0f}</b> / 晚
        ｜ ⭐ {rating_s} ｜ 💬 {_gi(L.get('number_of_reviews'))} 則
      {extra}
    </div>
    <div style="margin-top:6px;">{chips}</div>
    <div style="font-size:var(--sa-text-label);color:{P['muted']};margin-top:7px;
         border-top:1px dashed {P['border']};padding-top:5px;">
      完整設施、周遭 1KM 機能與近期評論請按「查看詳情」</div>
  </div>
</div>"""


HOVER_CSS = f"""
<style>
.hv-wrap{{position:relative;display:inline-block;}}
.hv-anchor{{font-size:var(--sa-text-label);color:{P['primary']};cursor:help;
  border-bottom:1px dashed {P['primary']};}}
.hv-card{{visibility:hidden;opacity:0;position:absolute;z-index:9999;
  left:0;top:150%;width:300px;background:{P['surface']};
  border:1px solid {P['border2']};border-radius:var(--sa-radius-md);padding:11px 13px;
  box-shadow:0 12px 34px rgba(0,0,0,.18);transition:opacity .15s ease;
  text-align:left;}}
.hv-wrap:hover .hv-card{{visibility:visible;opacity:1;}}
</style>"""
