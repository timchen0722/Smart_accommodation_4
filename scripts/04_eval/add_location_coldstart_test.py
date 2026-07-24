# -*- coding: utf-8 -*-
"""
針對性測試：地點/房間特徵對「冷啟動（新房東）」模型是否有幫助。
邏輯：這些特徵的價值主張是「當沒有房東歷史可用時，用地段與房間條件補訊號」。
故移除 7 個房東身分特徵後，比較 base_nohost vs +A1A2+A4，全用 GroupKFold(依 host)。
"""
import numpy as np, pandas as pd
from sklearn.neighbors import BallTree
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score

df = pd.read_csv("../../dataset_final.csv")
host = pd.read_csv("../../listings_cleaned.csv.gz", compression="gzip", low_memory=False)[["id", "host_id"]]
df = df.merge(host, left_on="listing_id", right_on="id", how="left").drop(columns=["id"])
hotels = pd.read_csv("../../hotels_taipei_osm.csv")

EARTH_KM = 6371.0088
htree = BallTree(np.radians(hotels[["lat", "lon"]].values), metric="haversine")
rad = np.radians(df[["latitude", "longitude"]].values)
df["hotel_count_1km"] = htree.query_radius(rad, r=1.0 / EARTH_KM, count_only=True)
df["hotel_count_500m"] = htree.query_radius(rad, r=0.5 / EARTH_KM, count_only=True)
acc = df["accommodates"].clip(lower=1)
df["price_per_person"] = df["price"] / acc
df["price_per_bedroom"] = df["price"] / df["bedrooms"].clip(lower=1)
df["beds_per_person"] = df["beds"] / acc
df["airbnb_hotel_supply_ratio"] = df["nbr_density_1km"] / (df["hotel_count_1km"] + 1)

NEW = ["price_per_person", "price_per_bedroom", "beds_per_person",
       "hotel_count_1km", "hotel_count_500m", "airbnb_hotel_supply_ratio"]
HOST_IDENTITY = ["host_is_superhost", "host_response_rate", "host_acceptance_rate",
                 "host_listings_count", "host_tenure_days", "response_speed", "host_about_len"]

full_base = [c for c in df.columns if c not in (["listing_id", "Y_vacancy", "host_id"] + NEW)
             and (not c.startswith("photo_") or c == "photo_design_sense")]
nohost_base = [c for c in full_base if c not in HOST_IDENTITY]

y = df["Y_vacancy"].values; yb = (y > 0.7).astype(int); groups = df["host_id"].values
REG = dict(max_iter=500, learning_rate=0.05, l2_regularization=1.0, random_state=42)
CLF = dict(max_iter=500, learning_rate=0.05, l2_regularization=1.0, random_state=42)

def group_cv(cols, tag):
    X = df[cols]; gkf = GroupKFold(n_splits=5); r2s, aucs = [], []
    for tr, te in gkf.split(X, y, groups):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        m = HistGradientBoostingRegressor(**REG).fit(Xtr, y[tr])
        r2s.append(r2_score(y[te], np.clip(m.predict(Xte), 0, 1)))
        if yb[tr].sum() >= 5 and yb[te].sum() >= 5:
            c = HistGradientBoostingClassifier(**CLF).fit(Xtr, yb[tr])
            aucs.append(roc_auc_score(yb[te], c.predict_proba(Xte)[:, 1]))
    r2s, aucs = np.array(r2s), np.array(aucs)
    print("  [{:28s}] ({:2d}特徵) R^2 {:.3f}±{:.3f} | AUC {:.3f}±{:.3f}".format(
        tag, len(cols), r2s.mean(), r2s.std(), aucs.mean(), aucs.std()))

print("GroupKFold 冷啟動測試（移除房東身分特徵）:")
group_cv(nohost_base, "冷啟動 base(無房東身分)")
group_cv(nohost_base + NEW, "冷啟動 +地點/房間特徵")
print("\n對照（含房東身分的完整模型）:")
group_cv(full_base, "完整 base")
group_cv(full_base + NEW, "完整 +地點/房間特徵")
