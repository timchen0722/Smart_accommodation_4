# -*- coding: utf-8 -*-
"""
B（需求側/生活機能）資料源：從 OSM Overpass 抓台北地區 POI，供密度特徵使用。
三類：
  attraction 觀光景點：tourism=attraction|museum|artwork|viewpoint|gallery|theme_park|zoo
  food       餐飲    ：amenity=restaurant|cafe|bar|fast_food
  conv       超商便利：shop=convenience|supermarket
結果快取到 poi_taipei_osm.csv（欄位：lat, lon, cat）。
"""
import time, json, urllib.request, urllib.parse
import pandas as pd

OUT = "../../poi_taipei_osm.csv"
ENDPOINTS = ["https://overpass-api.de/api/interpreter",
             "https://overpass.kumi.systems/api/interpreter"]

df = pd.read_csv("../../dataset_final.csv", usecols=["latitude", "longitude"])
margin = 0.02
s, n = df["latitude"].min() - margin, df["latitude"].max() + margin
w, e = df["longitude"].min() - margin, df["longitude"].max() + margin
bbox = "{:.4f},{:.4f},{:.4f},{:.4f}".format(s, w, n, e)
print("bbox:", bbox)

cats = {
    "attraction": 'node["tourism"~"attraction|museum|artwork|viewpoint|gallery|theme_park|zoo"]({bbox});'
                  'way["tourism"~"attraction|museum|artwork|viewpoint|gallery|theme_park|zoo"]({bbox});',
    "food": 'node["amenity"~"restaurant|cafe|bar|fast_food"]({bbox});'
            'way["amenity"~"restaurant|cafe|bar|fast_food"]({bbox});',
    "conv": 'node["shop"~"convenience|supermarket"]({bbox});'
            'way["shop"~"convenience|supermarket"]({bbox});',
}

all_rows = []
for cat, body in cats.items():
    query = "[out:json][timeout:180];(" + body.replace("{bbox}", bbox) + ");out center;"
    data = None
    for ep in ENDPOINTS:
        try:
            print("抓 {} @ {}".format(cat, ep))
            req = urllib.request.Request(ep, data=urllib.parse.urlencode({"data": query}).encode(),
                                         headers={"User-Agent": "airbnb-vacancy-research/1.0"})
            data = json.loads(urllib.request.urlopen(req, timeout=200).read())
            break
        except Exception as ex:
            print("  失敗:", repr(ex)); time.sleep(3)
    if data is None:
        print("  !! {} 全部 endpoint 失敗，略過".format(cat)); continue
    cnt = 0
    for el in data.get("elements", []):
        if el["type"] == "node":
            lat, lon = el.get("lat"), el.get("lon")
        else:
            c = el.get("center", {}); lat, lon = c.get("lat"), c.get("lon")
        if lat is not None and lon is not None:
            all_rows.append({"lat": lat, "lon": lon, "cat": cat}); cnt += 1
    print("  {} -> {} 筆".format(cat, cnt))
    time.sleep(2)

poi = pd.DataFrame(all_rows).drop_duplicates()
poi.to_csv(OUT, index=False, encoding="utf-8-sig")
print("總計 {} 筆 POI -> {}".format(len(poi), OUT))
print(poi["cat"].value_counts().to_dict())
