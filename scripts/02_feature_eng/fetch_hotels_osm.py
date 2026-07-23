# -*- coding: utf-8 -*-
"""
A1 資料源：從 OpenStreetMap Overpass API 抓取台北地區旅宿業 POI（飯店/旅館/青旅/民宿）。
- 依 dataset_final.csv 的房源經緯度範圍自動決定查詢 bounding box（外擴 margin）。
- 只取座標（lat/lon）與類型；OSM 無房價、房間數多半缺，故 A1 以「數量密度」為主。
- 結果快取到 hotels_taipei_osm.csv，避免重複打 API。
"""
import time
import json
import urllib.request
import urllib.parse
import pandas as pd

OUT = "../../hotels_taipei_osm.csv"
ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

df = pd.read_csv("../../dataset_final.csv", usecols=["latitude", "longitude"])
margin = 0.02  # 約 2km 外擴，涵蓋房源周邊競爭旅宿
south, north = df["latitude"].min() - margin, df["latitude"].max() + margin
west, east = df["longitude"].min() - margin, df["longitude"].max() + margin
bbox = "{:.4f},{:.4f},{:.4f},{:.4f}".format(south, west, north, east)
print("查詢 bbox (S,W,N,E):", bbox)

# tourism 標籤涵蓋主要旅宿型態；node 與 way(取中心點) 皆抓
query = """
[out:json][timeout:120];
(
  node["tourism"~"hotel|hostel|guest_house|motel|apartment"]({bbox});
  way["tourism"~"hotel|hostel|guest_house|motel|apartment"]({bbox});
  node["building"="hotel"]({bbox});
  way["building"="hotel"]({bbox});
);
out center tags;
""".replace("{bbox}", bbox)

data = None
for ep in ENDPOINTS:
    try:
        print("嘗試 Overpass endpoint:", ep)
        body = urllib.parse.urlencode({"data": query}).encode("utf-8")
        req = urllib.request.Request(ep, data=body, headers={"User-Agent": "airbnb-vacancy-research/1.0"})
        raw = urllib.request.urlopen(req, timeout=150).read()
        data = json.loads(raw)
        break
    except Exception as e:
        print("  失敗:", repr(e))
        time.sleep(3)

if data is None:
    raise SystemExit("所有 Overpass endpoint 皆失敗，請改用觀光署開放資料或稍後重試。")

rows = []
for el in data.get("elements", []):
    if el["type"] == "node":
        lat, lon = el.get("lat"), el.get("lon")
    else:  # way -> center
        c = el.get("center", {})
        lat, lon = c.get("lat"), c.get("lon")
    if lat is None or lon is None:
        continue
    tags = el.get("tags", {})
    rooms = tags.get("rooms") or tags.get("capacity:persons") or ""
    rows.append({
        "osm_id": el["id"],
        "osm_type": el["type"],
        "lat": lat,
        "lon": lon,
        "tourism": tags.get("tourism", tags.get("building", "")),
        "name": tags.get("name", ""),
        "rooms": rooms,
    })

hotels = pd.DataFrame(rows).drop_duplicates(subset=["lat", "lon"])
hotels.to_csv(OUT, index=False, encoding="utf-8-sig")
print("抓到旅宿 POI:", len(hotels), "筆 -> 存至", OUT)
print(hotels["tourism"].value_counts().to_dict())
