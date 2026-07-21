# -*- coding: utf-8 -*-
"""map_view.py — 周邊房源互動地圖(Leaflet 自足元件)

為什麼不用 plotly:需要 hover 連動右側列表、虛線半徑圓、本房源閃爍、
平台分層勾選 —— 這些在 Streamlit 的 plotly 靜態渲染下做不到
(plotly 只支援 click 選取事件,且無法回傳 hover)。

做法:把地圖與列表包成同一段 HTML 放進 components.html,
所有互動在瀏覽器端完成,零 Streamlit 往返、即時反應。
"""
from __future__ import annotations

import html
import json

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# 平台識別色(依需求指定)
PLATFORM_STYLE = {
    "591": {"label": "591租屋網", "color": "#8B5CF6"},          # 紫
    "Booking": {"label": "Booking.com", "color": "#2563EB"},     # 藍
    "ddroom": {"label": "DD租租網", "color": "#4B4B4B"},         # 深灰
}
# 空屋率分級色(周邊 Airbnb 房源)
RISK_COLOR = [(0.40, "#5B9E73"), (0.70, "#C49A4A"), (1.01, "#C4645A")]


def _risk_color(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "#9A9490"
    for hi, c in RISK_COLOR:
        if float(v) < hi:
            return c
    return RISK_COLOR[-1][1]


def _fmt(v, unit="", digits=0):
    if v is None or (isinstance(v, float) and (np.isnan(v))):
        return "—"
    return f"{float(v):,.{digits}f}{unit}"


def build_points(own, nearby: pd.DataFrame, addr_fn) -> tuple[dict, list]:
    """整理本房源與周邊 Airbnb 房源為前端資料。"""
    own_pt = {
        "id": int(own["id"]),
        "name": str(own.get("name") or f"房源 {own['id']}")[:40],
        "lat": float(own["latitude"]), "lon": float(own["longitude"]),
        "addr": addr_fn(own["latitude"], own["longitude"]) or "—",
        "price": _fmt(own.get("price"), " 元/晚"),
        "vac": _fmt((own.get("vac_pred") or 0) * 100, "%"),
        "room": str(own.get("room_type") or ""),
        "dist": 0,
    }
    pts = []
    for _, r in nearby.iterrows():
        pts.append({
            "id": int(r["id"]),
            "name": str(r.get("name") or f"房源 {r['id']}")[:40],
            "lat": float(r["latitude"]), "lon": float(r["longitude"]),
            "addr": addr_fn(r["latitude"], r["longitude"]) or "—",
            "price": _fmt(r.get("price"), " 元/晚"),
            "vac": _fmt((r.get("vac_pred") or 0) * 100, "%"),
            "vac_raw": float(r.get("vac_pred") or 0),
            "room": str(r.get("room_type") or ""),
            "dist": int(r.get("dist_m") or 0),
            "color": _risk_color(r.get("vac_pred")),
        })
    pts.sort(key=lambda x: x["dist"])
    return own_pt, pts


def build_competitors(comp: pd.DataFrame, own_lat: float, own_lon: float,
                      addr_fn=None) -> dict:
    """依平台分組整理跨平台競品。"""
    out = {}
    for key, style in PLATFORM_STYLE.items():
        g = comp[comp["platform"] == key]
        rows = []
        for _, r in g.iterrows():
            unit = "元/月" if r.get("price_unit") == "month" else "元/晚"
            rows.append({
                "name": str(r.get("title") or "")[:38],
                "lat": float(r["lat"]), "lon": float(r["lon"]),
                "price": _fmt(r.get("price_raw"), f" {unit}"),
                "pp": _fmt(r.get("price_pp_day"), " 元"),
                "cap": _fmt(r.get("capacity"), " 人"),
                "dist": int(r.get("dist_m") or 0),
                "addr": (addr_fn(r["lat"], r["lon"]) if addr_fn else "") or "—",
            })
        rows.sort(key=lambda x: x["dist"])
        out[key] = {"label": style["label"], "color": style["color"],
                    "points": rows}
    return out


_HTML = """
<!DOCTYPE html><html lang="zh-Hant"><head><meta charset="utf-8">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html,body{margin:0;padding:0;font-family:"Noto Sans TC",-apple-system,sans-serif;
    background:#F8F7F5;color:#2A2A2A;}
  .wrap{display:flex;gap:10px;height:__H__px;}
  #map{flex:1 1 auto;border:1px solid #E8E4DE;border-radius:10px;}
  .side{width:308px;display:flex;flex-direction:column;
    border:1px solid #E8E4DE;border-radius:10px;background:#fff;overflow:hidden;}
  .side h4{margin:0;padding:9px 13px;font-size:.79rem;letter-spacing:.06em;
    color:#9A9490;background:#F2F0EC;border-bottom:1px solid #E8E4DE;font-weight:700;}
  .list{overflow-y:auto;flex:1;}
  .item{padding:8px 13px;border-bottom:1px solid #F0EDE8;cursor:pointer;
    transition:background .12s,border-left-color .12s;border-left:3px solid transparent;}
  .item:hover{background:#F8F7F5;}
  .item.on{background:#FFF6E8;border-left-color:#C4645A;}
  .t{font-size:.8rem;font-weight:600;color:#2A2A2A;white-space:nowrap;
    overflow:hidden;text-overflow:ellipsis;}
  .s{font-size:.72rem;color:#9A9490;margin-top:2px;line-height:1.5;}
  .dot{display:inline-block;width:8px;height:8px;border-radius:50%;
    margin-right:5px;vertical-align:middle;}
  .empty{padding:16px 13px;font-size:.78rem;color:#9A9490;}
  /* 本房源閃爍標記 */
  .me-wrap{position:relative;width:20px;height:20px;}
  .me-core{position:absolute;left:4px;top:4px;width:12px;height:12px;
    border-radius:50%;background:#2A2A2A;border:2px solid #fff;
    box-shadow:0 0 0 1px #2A2A2A;}
  .me-ring{position:absolute;left:0;top:0;width:20px;height:20px;border-radius:50%;
    background:rgba(196,100,90,.55);animation:pulse 1.6s ease-out infinite;}
  @keyframes pulse{0%{transform:scale(.5);opacity:.9;}
    70%{transform:scale(2.2);opacity:0;}100%{opacity:0;}}
  .leaflet-tooltip.zh{font-family:"Noto Sans TC",sans-serif;font-size:.76rem;
    line-height:1.65;padding:7px 10px;border-radius:7px;border:1px solid #D4CFC8;
    box-shadow:0 4px 14px rgba(0,0,0,.12);}
  .tt-t{font-weight:700;color:#2A2A2A;margin-bottom:3px;}
  .tt-r{color:#54514E;}
  .tt-k{color:#9A9490;}
</style></head><body>
<div class="wrap">
  <div id="map"></div>
  <div class="side">
    <h4 id="side-title">周邊房源</h4>
    <div class="list" id="list"></div>
  </div>
</div>
<script>
const OWN = __OWN__, PTS = __PTS__, COMP = __COMP__, RADIUS = __RADIUS__;

const map = L.map('map', {zoomControl:true, scrollWheelZoom:true})
  .setView([OWN.lat, OWN.lon], 15);
L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
  {attribution:'&copy; OpenStreetMap &copy; CARTO', maxZoom:19}).addTo(map);

// 半徑虛線圓
const ring = L.circle([OWN.lat, OWN.lon], {
  radius: RADIUS, color:'#4E7FB0', weight:1.6, opacity:.85,
  dashArray:'7 7', fill:true, fillColor:'#4E7FB0', fillOpacity:.045
}).addTo(map);
map.fitBounds(ring.getBounds(), {padding:[18,18]});

// 本房源(閃爍)
const meIcon = L.divIcon({className:'', iconSize:[20,20], iconAnchor:[10,10],
  html:'<div class="me-wrap"><div class="me-ring"></div><div class="me-core"></div></div>'});
L.marker([OWN.lat, OWN.lon], {icon:meIcon, zIndexOffset:1000})
  .bindTooltip(`<div class="tt-t">★ 本房源｜${OWN.name}</div>
    <div class="tt-r"><span class="tt-k">地址</span> ${OWN.addr}</div>
    <div class="tt-r"><span class="tt-k">定價</span> ${OWN.price}
    <span class="tt-k">預測空屋率</span> ${OWN.vac}</div>`,
    {className:'zh', direction:'top', offset:[0,-10]}).addTo(map);

// ── 周邊 Airbnb 房源 ──
const airbnbLayer = L.layerGroup().addTo(map);
const markers = {};
PTS.forEach((p, i) => {
  const m = L.circleMarker([p.lat, p.lon], {
    radius:6, color:'#fff', weight:1.4, fillColor:p.color, fillOpacity:.9
  }).bindTooltip(`<div class="tt-t">${p.name}</div>
    <div class="tt-r"><span class="tt-k">地址</span> ${p.addr}</div>
    <div class="tt-r"><span class="tt-k">距本房源</span> ${p.dist} 公尺
      <span class="tt-k">定價</span> ${p.price}</div>
    <div class="tt-r"><span class="tt-k">預測空屋率</span> ${p.vac}</div>`,
    {className:'zh', direction:'top', offset:[0,-6]});
  m.on('mouseover', () => focusItem(i, true));
  m.on('mouseout',  () => focusItem(i, false));
  m.addTo(airbnbLayer);
  markers[i] = m;
});

// ── 跨平台競品(分層)──
const overlays = {'周邊 Airbnb 房源': airbnbLayer};
Object.keys(COMP).forEach(k => {
  const g = COMP[k], layer = L.layerGroup();
  g.points.forEach(p => {
    L.circleMarker([p.lat, p.lon], {
      radius:5, color:'#fff', weight:1.2, fillColor:g.color, fillOpacity:.88
    }).bindTooltip(`<div class="tt-t" style="color:${g.color}">${g.label}｜${p.name}</div>
      <div class="tt-r"><span class="tt-k">地址</span> ${p.addr}</div>
      <div class="tt-r"><span class="tt-k">距本房源</span> ${p.dist} 公尺
        <span class="tt-k">掛牌價</span> ${p.price}</div>
      <div class="tt-r"><span class="tt-k">每人每晚等效</span> ${p.pp}
        <span class="tt-k">可住</span> ${p.cap}</div>`,
      {className:'zh', direction:'top', offset:[0,-5]}).addTo(layer);
  });
  overlays[`<span style="color:${g.color};font-weight:700">●</span> ${g.label}` +
           ` (${g.points.length})`] = layer;
});
L.control.layers(null, overlays, {collapsed:false, position:'topright'}).addTo(map);

// ── 右側列表(與地圖 hover 雙向連動)──
const list = document.getElementById('list');
document.getElementById('side-title').textContent =
  `周邊房源（${PTS.length} 間・依距離排序）`;
if (!PTS.length) {
  list.innerHTML = '<div class="empty">此半徑內沒有其他 Airbnb 房源。</div>';
}
PTS.forEach((p, i) => {
  const d = document.createElement('div');
  d.className = 'item'; d.id = 'it' + i;
  d.innerHTML = `<div class="t"><span class="dot" style="background:${p.color}"></span>${p.name}</div>
    <div class="s">${p.addr}</div>
    <div class="s">距離 ${p.dist} 公尺・${p.price}・空屋率 ${p.vac}</div>`;
  d.onmouseenter = () => { highlight(i, true);  markers[i].openTooltip(); };
  d.onmouseleave = () => { highlight(i, false); markers[i].closeTooltip(); };
  d.onclick = () => map.setView([p.lat, p.lon], 17);
  list.appendChild(d);
});

function highlight(i, on) {
  const el = document.getElementById('it' + i);
  if (el) el.classList.toggle('on', on);
  const m = markers[i];
  if (m) m.setStyle(on ? {radius:10, weight:2.6, color:'#C4645A', fillOpacity:1}
                       : {radius:6, weight:1.4, color:'#fff', fillOpacity:.9});
}
function focusItem(i, on) {
  highlight(i, on);
  if (on) {
    const el = document.getElementById('it' + i);
    if (el) el.scrollIntoView({block:'nearest', behavior:'smooth'});
  }
}
</script></body></html>
"""


def render(own, nearby: pd.DataFrame, comp: pd.DataFrame, radius_m: float,
           addr_fn, height: int = 520):
    """渲染互動地圖 + 連動列表。

    own      本房源(Series,需含 id/name/latitude/longitude/price/vac_pred)
    nearby   周邊 Airbnb 房源(需含 dist_m)
    comp     跨平台競品(需含 platform/lat/lon/title/price_raw/price_pp_day/dist_m)
    addr_fn  逆地理函式 (lat, lon) -> 地址字串
    """
    own_pt, pts = build_points(own, nearby, addr_fn)
    comps = build_competitors(comp, float(own["latitude"]),
                              float(own["longitude"]), addr_fn)
    page = (_HTML
            .replace("__OWN__", json.dumps(own_pt, ensure_ascii=False))
            .replace("__PTS__", json.dumps(pts, ensure_ascii=False))
            .replace("__COMP__", json.dumps(comps, ensure_ascii=False))
            .replace("__RADIUS__", str(int(radius_m)))
            .replace("__H__", str(int(height))))
    components.html(page, height=height + 12, scrolling=False)
