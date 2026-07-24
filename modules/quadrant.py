# -*- coding: utf-8 -*-
"""quadrant.py — 體質 × 檔期 四象限分類

解決的問題:模型等級(體質推估)與 calendar 已訂率(真實觀測)口徑不同,
並列呈現會出現「高風險卻訂滿」「安全卻空著」的表面矛盾。
實測 corr(模型機率, 實際90天空屋率) 僅 0.063,兩者本就不該互相取代。

判斷原則
--------
  近期行動看檔期(100% 真實觀測);長期投資看模型(AUC 0.632 的體質推估)。
  兩者衝突時,以檔期為準。

四象限
------
  🚨 真警報    體質差 + 檔期空  → 最高優先
  👻 隱形危機  體質佳 + 檔期空  → 模型沒抓到,但檔期在流血
  ⚠️ 靠降價撐住 體質差 + 檔期滿  → 短期無虞,查是否賠本衝量
  ✅ 健康      體質佳 + 檔期滿  → 維持現狀
  ❔ 檔期資料不足  無檔期資料(calendar 與 listings 為不同批次,約 1,600 間無對照)

文案注意:第五類的名稱一律是「檔期資料不足」(design_tokens.STATUS_NO_DATA)。
原本 docstring 寫「資料不足」、label 寫「檔期資料不足」,兩者不一致。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from modules import design_tokens as T

# 檔期門檻(未來 90 天已訂率)
BOOKED_HIGH = 0.50
BOOKED_LOW = 0.20


QUADRANTS = {
    "alarm": {"label": "🚨 真警報", "color": "high", "priority": 1,
              "desc": "房子本身條件不好,未來也幾乎沒人訂——兩邊都亮紅燈,最該優先處理",
              "action": "立即檢視定價與 LIME 痛點,同步啟動空檔促銷"},
    "hidden": {"label": "👻 隱形危機", "color": "medium", "priority": 2,
               "desc": "房子條件明明不錯,但接下來的日子幾乎沒訂單——問題不在房子,是最近出了狀況",
               "action": "優先查近期變動:競品降價、季節性淡季、日曆設定或照片失效"},
    "discount": {"label": "⚠️ 靠降價撐住", "color": "accent", "priority": 3,
                 "desc": "房間幾乎都訂滿,但房子本身條件其實不算好——住滿不是因為受歡迎,而是價格壓得夠低。看起來生意好,實際可能不太賺錢",
                 "action": "檢查單價與 RevPAR(而非入住率),確認未賠本衝量"},
    "healthy": {"label": "✅ 健康", "color": "low", "priority": 4,
                "desc": "房子條件好,訂單也滿,一切正常",
                "action": "維持現狀,持續觀察同商圈行情"},
    "unknown": {"label": f"❔ {T.STATUS_NO_DATA}", "color": "muted", "priority": 5,
                "desc": "這批房源沒抓到訂房日曆資料(兩份資料爬取時間不同),無法判斷未來訂況",
                "action": "僅依模型體質評估判讀"},
}


def classify_row(tier: str, booked_rate_d90) -> str:
    """單筆分類。tier 為模型等級(red/yellow/green),booked_rate_d90 為真實已訂率。"""
    if booked_rate_d90 is None or (isinstance(booked_rate_d90, float)
                                   and np.isnan(booked_rate_d90)):
        return "unknown"
    weak = tier in ("red", "yellow")          # 體質差 = 紅或黃
    if booked_rate_d90 < BOOKED_LOW:
        return "alarm" if weak else "hidden"
    if booked_rate_d90 >= BOOKED_HIGH:
        return "discount" if weak else "healthy"
    # 中間帶(20~50%):偏向體質判斷,但不視為警報
    return "discount" if weak else "healthy"


def annotate(df: pd.DataFrame, tier_col: str = "tier") -> pd.DataFrame:
    """為 DataFrame 加上象限欄位。需已含 booked_rate_d90(可為 NaN)。"""
    d = df.copy()
    if "booked_rate_d90" not in d.columns:
        d["booked_rate_d90"] = np.nan
    d["quadrant"] = [classify_row(t, b)
                     for t, b in zip(d[tier_col], d["booked_rate_d90"])]
    d["quadrant_label"] = d["quadrant"].map(lambda q: QUADRANTS[q]["label"])
    d["quadrant_priority"] = d["quadrant"].map(lambda q: QUADRANTS[q]["priority"])
    return d


def attach_calendar(df: pd.DataFrame, id_col: str = "id") -> pd.DataFrame:
    """併入 calendar 的真實已訂率與空檔指標(缺產物時回傳原表 + NaN 欄)。"""
    d = df.copy()
    try:
        from modules import calendar_analytics as ca
        if ca.available():
            cal = ca.healthy_metrics()[
                ["listing_id", "booked_rate", "booked_rate_d30",
                 "booked_rate_d90", "gap_days_30d", "gap_longest_30d"]]
            d = d.merge(cal, left_on=id_col, right_on="listing_id", how="left",
                        suffixes=("", "_cal"))
    except Exception:
        pass
    for c in ["booked_rate", "booked_rate_d30", "booked_rate_d90",
              "gap_days_30d", "gap_longest_30d"]:
        if c not in d.columns:
            d[c] = np.nan
    return d


def summary(df: pd.DataFrame) -> pd.DataFrame:
    """象限統計(依優先序排列)。"""
    g = (df.groupby("quadrant").size().rename("房源數").reset_index())
    g["象限"] = g["quadrant"].map(lambda q: QUADRANTS[q]["label"])
    g["優先序"] = g["quadrant"].map(lambda q: QUADRANTS[q]["priority"])
    g["說明"] = g["quadrant"].map(lambda q: QUADRANTS[q]["desc"])
    g["建議行動"] = g["quadrant"].map(lambda q: QUADRANTS[q]["action"])
    return g.sort_values("優先序")[["象限", "房源數", "說明", "建議行動"]]
