# -*- coding: utf-8 -*-
"""
NLP 情緒分析 → 房源層級情緒特徵
- 英文/latin：VADER（compound → 0~1）
- 中文 zh：SnowNLP（0~1）
- 聚合每個 listing：情緒平均、負評比例、情緒標準差
輸出：nlp_features.csv
"""
import pandas as pd, numpy as np, time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from snownlp import SnowNLP

df = pd.read_csv("../../reviews_cleaned.csv.gz", compression="gzip", low_memory=False)
df = df.dropna(subset=["cleaned_comments"])
txt = df["cleaned_comments"].astype(str)
lang = df["language_type"].astype(str)

va = SentimentIntensityAnalyzer()
t0 = time.time()
scores = np.full(len(df), np.nan)

# 英文 / 混合 / other → VADER
mask_v = lang.isin(["en", "mixed_zh_en", "other"]).values
tv = txt[mask_v].tolist()
sv = np.array([(va.polarity_scores(t)["compound"] + 1) / 2 for t in tv])
scores[np.where(mask_v)[0]] = sv
print("VADER done: {} comments, {:.0f}s".format(mask_v.sum(), time.time()-t0))

# 中文 → SnowNLP
mask_z = (lang == "zh").values
tz = txt[mask_z].tolist()
sz = np.empty(len(tz))
for i, t in enumerate(tz):
    try:
        sz[i] = SnowNLP(t).sentiments
    except Exception:
        sz[i] = 0.5
scores[np.where(mask_z)[0]] = sz
print("SnowNLP done: {} comments, {:.0f}s total".format(mask_z.sum(), time.time()-t0))

df["senti"] = scores
df = df.dropna(subset=["senti"])

# 聚合到 listing（不用評論數，避免與入住量相關的洩漏）
agg = df.groupby("listing_id")["senti"].agg(
    senti_mean="mean",
    senti_std="std",
    senti_neg_ratio=lambda s: (s < 0.4).mean(),   # 負面評論比例
).reset_index()
agg["senti_std"] = agg["senti_std"].fillna(0)
agg.to_csv("../../nlp_features.csv", index=False, encoding="utf-8-sig")
print("saved nlp_features.csv | listings:", len(agg))
print("senti_mean 平均 {:.3f} | neg_ratio 平均 {:.3f}".format(agg["senti_mean"].mean(), agg["senti_neg_ratio"].mean()))
