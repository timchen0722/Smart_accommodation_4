# -*- coding: utf-8 -*-
"""build_absa.py — 評論面向情感分析(Aspect-Based Sentiment Analysis)離線預計算

為什麼離線做:21 萬則評論 × 12 面向的比對成本高,且 Streamlit Cloud 約 1GB RAM
無法載入 BERT/XLM-R(400MB~1.1GB)。本腳本以「面向詞典 + 局部情感窗口」實作,
不需深度模型即可產出可用的面向情感,結果存成小檔供線上讀取。

方法
----
1. 12 個面向各有中英關鍵詞;命中即視為該則評論談到此面向
2. 取關鍵詞前後 ±WINDOW 字元的「局部窗口」,以正負情感詞計分
   (比整篇評論計分準確:「早餐好吃但浴室很小」能拆成兩種情感)
3. 依房源彙總:各面向的提及數、正評數、負評數、淨情感

產出
----
data/_absa_listing.csv   每房源 × 每面向 的統計
data/_absa_market.csv    全市 / 行政區層級的面向統計
data/_absa_examples.csv  每面向的代表性負評例句(供 UI 佐證與 LLM 建議引用)

執行
----
    python -X utf8 scripts/build_absa.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data"

WINDOW = 30          # 局部情感窗口(關鍵詞前後字元數)
MAX_EXAMPLES = 3     # 每面向保留幾則代表負評

# ── 12 個面向詞典(繁中 / 英文)──
ASPECTS = {
    "清潔": ["乾淨", "整潔", "髒", "灰塵", "打掃", "清潔", "clean", "dirty",
             "dust", "tidy", "spotless", "filthy"],
    "位置交通": ["位置", "地點", "交通", "捷運", "地鐵", "車站", "方便",
                 "location", "mrt", "metro", "station", "convenient", "walk"],
    "隔音噪音": ["安靜", "吵", "噪音", "隔音", "吵雜", "quiet", "noisy",
                 "noise", "loud", "soundproof"],
    "網路WiFi": ["網路", "wifi", "無線", "訊號", "internet", "wi-fi",
                 "connection"],
    "空調冷氣": ["冷氣", "空調", "暖氣", "悶熱", "通風", "air con", "aircon",
                 "ac ", "heater", "ventilation", "stuffy"],
    "床與睡眠": ["床", "床墊", "枕頭", "睡", "棉被", "bed", "mattress",
                 "pillow", "sleep", "blanket"],
    "衛浴熱水": ["浴室", "熱水", "馬桶", "淋浴", "廁所", "bathroom", "shower",
                 "hot water", "toilet"],
    "房東服務": ["房東", "服務", "溝通", "回覆", "親切", "host", "service",
                 "communication", "responsive", "helpful", "friendly"],
    "性價比": ["價格", "便宜", "划算", "值得", "性價比", "貴", "price",
               "value", "cheap", "worth", "expensive", "affordable"],
    "空間大小": ["空間", "坪數", "寬敞", "狹小", "小", "大", "space",
                 "spacious", "small", "tiny", "cramped", "roomy"],
    "設備廚房": ["廚房", "冰箱", "洗衣機", "微波爐", "設備", "家電",
                 "kitchen", "fridge", "washer", "microwave", "equipment"],
    "入住流程": ["入住", "check in", "checkin", "check-in", "鑰匙", "密碼",
                 "退房", "key", "keypad", "self check", "checkout"],
}

POS_WORDS = ["乾淨", "整潔", "舒適", "方便", "安靜", "親切", "推薦", "很棒",
             "不錯", "滿意", "貼心", "值得", "划算", "寬敞", "溫馨", "好",
             "clean", "comfortable", "convenient", "quiet", "friendly",
             "great", "good", "nice", "excellent", "perfect", "recommend",
             "spacious", "helpful", "amazing", "lovely", "worth"]
NEG_WORDS = ["髒", "吵", "噪音", "壞", "差", "失望", "狹小", "悶", "臭",
             "冷", "慢", "貴", "不便", "問題", "無法", "沒有熱水", "不乾淨",
             "dirty", "noisy", "broken", "bad", "terrible", "disappointing",
             "small", "cramped", "smell", "poor", "slow", "expensive",
             "uncomfortable", "worst", "issue", "problem", "not clean"]


def log(m):
    print(f"[absa] {m}", flush=True)


def _sent_score(text: str) -> int:
    """局部窗口情感:正詞數 − 負詞數。"""
    t = text.lower()
    pos = sum(1 for w in POS_WORDS if w in t)
    neg = sum(1 for w in NEG_WORDS if w in t)
    return pos - neg


def _windows(text: str, kws: list[str]) -> list[str]:
    """取所有關鍵詞的局部窗口(±WINDOW 字元)。"""
    t = text.lower()
    out = []
    for kw in kws:
        start = 0
        while True:
            i = t.find(kw, start)
            if i < 0:
                break
            out.append(t[max(0, i - WINDOW): i + len(kw) + WINDOW])
            start = i + len(kw)
            if len(out) >= 6:      # 單則評論同面向最多取 6 個窗口
                return out
    return out


def main():
    log("讀取評論 …")
    # 評論檔可能是 cleaned_comments 版或 comments 版,兩者皆相容
    head = pd.read_csv(DATA / "reviews_cleaned.csv.gz", nrows=2)
    text_col = "cleaned_comments" if "cleaned_comments" in head.columns else "comments"
    rv = pd.read_csv(DATA / "reviews_cleaned.csv.gz",
                     usecols=["listing_id", "date", text_col])
    rv = rv.rename(columns={text_col: "cleaned_comments"})
    rv["cleaned_comments"] = rv["cleaned_comments"].astype(str)
    rv["date"] = pd.to_datetime(rv["date"], errors="coerce")
    rv = rv.dropna(subset=["date"])
    log(f"{len(rv):,} 則評論 · {rv['listing_id'].nunique():,} 房源")

    txt_lower = rv["cleaned_comments"].str.lower()
    records, examples = [], []

    for aspect, kws in ASPECTS.items():
        pat = "|".join(re.escape(k) for k in kws)
        hit = txt_lower.str.contains(pat, regex=True, na=False)
        sub = rv[hit]
        if sub.empty:
            continue
        scores = np.fromiter(
            (sum(_sent_score(w) for w in _windows(t, kws))
             for t in sub["cleaned_comments"]),
            dtype=float, count=len(sub))
        d = pd.DataFrame({
            "listing_id": sub["listing_id"].to_numpy(),
            "aspect": aspect,
            "score": scores,
            "pos": (scores > 0).astype(int),
            "neg": (scores < 0).astype(int),
        })
        records.append(d)
        # 代表性負評(分數最低者)
        worst = sub.assign(score=scores).nsmallest(MAX_EXAMPLES * 40, "score")
        worst = worst.drop_duplicates("listing_id").head(MAX_EXAMPLES * 8)
        for _, r in worst.iterrows():
            examples.append({"aspect": aspect,
                             "listing_id": int(r["listing_id"]),
                             "score": float(r["score"]),
                             "text": str(r["cleaned_comments"])[:160]})
        log(f"  {aspect}: 提及 {len(sub):,} 則 "
            f"(正 {(scores > 0).sum():,} / 負 {(scores < 0).sum():,})")

    allr = pd.concat(records, ignore_index=True)

    # ── 每房源 × 面向 ──
    per = (allr.groupby(["listing_id", "aspect"])
           .agg(mentions=("score", "size"), pos=("pos", "sum"),
                neg=("neg", "sum"), avg_score=("score", "mean"))
           .reset_index())
    per["neg_ratio"] = (per["neg"] / per["mentions"]).round(4)
    per["pos_ratio"] = (per["pos"] / per["mentions"]).round(4)
    per["avg_score"] = per["avg_score"].round(3)
    per.to_csv(DATA / "_absa_listing.csv", index=False, encoding="utf-8")
    log(f"_absa_listing.csv: {len(per):,} 列 "
        f"({(DATA / '_absa_listing.csv').stat().st_size / 1e6:.1f} MB)")

    # ── 市場層級(全市 + 行政區)──
    li = pd.read_csv(DATA / "listings_cleaned.csv.gz",
                     usecols=["id", "neighbourhood_cleansed", "room_type"],
                     low_memory=False)
    j = per.merge(li, left_on="listing_id", right_on="id", how="left")
    city = (j.groupby("aspect")
            .agg(mentions=("mentions", "sum"), pos=("pos", "sum"),
                 neg=("neg", "sum")).reset_index())
    city["scope"] = "全市"
    city["group"] = "全市"
    dist = (j.dropna(subset=["neighbourhood_cleansed"])
            .groupby(["neighbourhood_cleansed", "aspect"])
            .agg(mentions=("mentions", "sum"), pos=("pos", "sum"),
                 neg=("neg", "sum")).reset_index()
            .rename(columns={"neighbourhood_cleansed": "group"}))
    dist["scope"] = "行政區"
    mkt = pd.concat([city, dist], ignore_index=True)
    mkt["neg_ratio"] = (mkt["neg"] / mkt["mentions"]).round(4)
    mkt["pos_ratio"] = (mkt["pos"] / mkt["mentions"]).round(4)
    mkt.to_csv(DATA / "_absa_market.csv", index=False, encoding="utf-8")
    log(f"_absa_market.csv: {len(mkt):,} 列")

    ex = pd.DataFrame(examples)
    ex.to_csv(DATA / "_absa_examples.csv", index=False, encoding="utf-8")
    log(f"_absa_examples.csv: {len(ex):,} 則代表負評")
    log("完成")


if __name__ == "__main__":
    main()
