# -*- coding: utf-8 -*-
"""載入 Data/ 7 個台北官方 POI 來源，回傳 {key: ndarray[[lat,lon],...]}，已過 bbox/NaN。"""
import json
import numpy as np
import pandas as pd

DATA = "../../Data"
LAT_MIN, LAT_MAX = 24.9, 25.3
LON_MIN, LON_MAX = 121.4, 121.7


def _bbox_filter(lat, lon):
    lat = pd.to_numeric(lat, errors="coerce").to_numpy(dtype="float64")
    lon = pd.to_numeric(lon, errors="coerce").to_numpy(dtype="float64")
    ok = (~np.isnan(lat)) & (~np.isnan(lon)) & \
         (lat >= LAT_MIN) & (lat <= LAT_MAX) & (lon >= LON_MIN) & (lon <= LON_MAX)
    return np.column_stack([lat[ok], lon[ok]])


def _from_csv(path, lat_col, lon_col, encoding):
    df = pd.read_csv(f"{DATA}/{path}", encoding=encoding)
    return _bbox_filter(df[lat_col], df[lon_col])


def _mrt():
    # Big5、無可靠表頭 → 用位置：index 3=經度(lon), 4=緯度(lat)
    df = pd.read_csv(f"{DATA}/臺北捷運車站出入口座標.csv", encoding="big5")
    return _bbox_filter(df.iloc[:, 4], df.iloc[:, 3])


def _from_json(path, lat_key, lon_key):
    with open(f"{DATA}/{path}", encoding="utf-8") as f:
        rows = json.load(f)
    lat = pd.Series([r.get(lat_key) for r in rows])
    lon = pd.Series([r.get(lon_key) for r in rows])
    return _bbox_filter(lat, lon)


def load_all_poi():
    return {
        "mrt": _mrt(),
        "bus": _from_csv("台北市公車站牌.csv", "latitude", "longitude", "utf-8-sig"),
        "rest": _from_csv("taipei_restaurants.csv", "Latitude", "Longitude", "utf-8-sig"),
        "cvs": _from_csv("台北市超商資料集_含經緯度_門牌級.csv", "緯度", "經度", "utf-8-sig"),
        "park": _from_json("臺北市公園基本資料.json", "pm_Latitude", "pm_Longitude"),
        "school": _from_csv("taipei_school.csv", "緯度", "經度", "utf-8-sig"),
        "pharm": _from_json("taipei_clinics.json", "lat", "lon"),
    }


if __name__ == "__main__":
    for k, arr in load_all_poi().items():
        print(f"  {k:7s} {arr.shape[0]:6d} 筆  lat[{arr[:,0].min():.3f},{arr[:,0].max():.3f}] "
              f"lon[{arr[:,1].min():.3f},{arr[:,1].max():.3f}]")
