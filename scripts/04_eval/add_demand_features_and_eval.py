# -*- coding: utf-8 -*-
"""
B（需求側/生活機能）特徵並增量評估（同時看完整模型與冷啟動模型）。
特徵（OSM POI 密度，BallTree haversine）：
  attraction_1km / food_1km / conv_1km / attraction_500m
評估：完整 base vs +demand、冷啟動 base(無房東身分) vs +demand，全用 GroupKFold(依 host)。
另出新特徵與 Y 相關性、permutation 重要度。
"""
import numpy as np, pandas as pd
from sklearn.neighbors import BallTree
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, roc_auc_score

df = pd.read_csv("../../dataset_final.csv")
host = pd.read_csv("../../listings_cleaned.csv.gz", compression="gzip", low_memory=False)[["id", "host_id"]]
df = df.merge(host, left_on="listing_id", right_on="id", how="left").drop(columns=["id"])
poi = pd.read_csv("../../poi_taipei_osm.csv")

EARTH = 6371.0088
rad = np.radians(df[["latitude", "longitude"]].values)
def dens(cat, r_km, name):
    sub = poi[poi["cat"] == cat]
    if len(sub) == 0:
        df[name] = 0; return
    tree = BallTree(np.radians(sub[["lat", "lon"]].values), metric="haversine")
    df[name] = tree.query_radius(rad, r=r_km / EARTH, count_only=True)

dens("attraction", 1.0, "attraction_1km")
dens("attraction", 0.5, "attraction_500m")
dens("food", 1.0, "food_1km")
dens("conv", 1.0, "conv_1km")

NEW = ["attraction_1km", "attraction_500m", "food_1km", "conv_1km"]
HOST_ID = ["host_is_superhost", "host_response_rate", "host_acceptance_rate",
           "host_listings_count", "host_tenure_days", "response_speed", "host_about_len"]
full_base = [c for c in df.columns if c not in (["listing_id", "Y_vacancy", "host_id"] + NEW)
             and (not c.startswith("photo_") or c == "photo_design_sense")]
nohost_base = [c for c in full_base if c not in HOST_ID]

for f in NEW:
    print("  {:16s} 與 Y 相關 {:+.4f} | 中位數 {:.0f} 平均 {:.1f}".format(
        f, df[f].corr(df["Y_vacancy"]), df[f].median(), df[f].mean()))

y = df["Y_vacancy"].values; yb = (y > 0.7).astype(int); groups = df["host_id"].values
REG = dict(max_iter=500, learning_rate=0.05, l2_regularization=1.0, random_state=42)
CLF = dict(max_iter=500, learning_rate=0.05, l2_regularization=1.0, random_state=42)

def gcv(cols, tag):
    X = df[cols]; gkf = GroupKFold(5); r2s, aucs = [], []
    for tr, te in gkf.split(X, y, groups):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        m = HistGradientBoostingRegressor(**REG).fit(Xtr, y[tr])
        r2s.append(r2_score(y[te], np.clip(m.predict(Xte), 0, 1)))
        if yb[tr].sum() >= 5 and yb[te].sum() >= 5:
            c = HistGradientBoostingClassifier(**CLF).fit(Xtr, yb[tr])
            aucs.append(roc_auc_score(yb[te], c.predict_proba(Xte)[:, 1]))
    r2s, aucs = np.array(r2s), np.array(aucs)
    print("  [{:26s}] ({:2d}) R^2 {:.3f}±{:.3f} | AUC {:.3f}±{:.3f}".format(
        tag, len(cols), r2s.mean(), r2s.std(), aucs.mean(), aucs.std()))

print("\n完整模型:")
gcv(full_base, "base"); gcv(full_base + NEW, "+demand")
print("\n冷啟動模型（無房東身分）:")
gcv(nohost_base, "base"); gcv(nohost_base + NEW, "+demand")

print("\n+demand 完整模型 新特徵 permutation 重要度:")
cols = full_base + NEW
Xtr, Xte, ytr, yte = train_test_split(df[cols], y, test_size=0.2, random_state=42)
m = HistGradientBoostingRegressor(**REG).fit(Xtr, ytr)
imp = permutation_importance(m, Xte, yte, n_repeats=5, random_state=42, scoring="r2")
rank = list(np.argsort(imp.importances_mean)[::-1])
for f in NEW:
    i = cols.index(f)
    print("  {:16s} 第 {:2d}/{} 名 (貢獻 {:+.4f})".format(f, rank.index(i) + 1, len(cols), imp.importances_mean[i]))
