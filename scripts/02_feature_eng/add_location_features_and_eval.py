# -*- coding: utf-8 -*-
"""
新增地點級/房間級特徵並做增量評估（兩種評估都出）：
  A4 房間級衍生比值：price_per_person, price_per_bedroom, beds_per_person
  A1 飯店競爭密度   ：hotel_count_1km, hotel_count_500m（OSM 旅宿 POI，BallTree haversine）
  A2 供需競爭比     ：airbnb_hotel_supply_ratio = nbr_density_1km / (hotel_count_1km + 1)

評估：base / +A4 / +A1A2 / +ALL，各出
  ① 單次切分 80/20：回歸 MAE/MSE/R^2、分類 AUC/Recall/Precision/F1
  ② GroupKFold(依 host_id) 5 折：回歸 R^2/MAE、分類 AUC
最後對 +ALL 模型算新特徵 permutation 重要度排名。
"""
import numpy as np, pandas as pd
from sklearn.neighbors import BallTree
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.inspection import permutation_importance
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             roc_auc_score, recall_score, precision_score, f1_score)

# ---------- 載入資料 ----------
df = pd.read_csv("../../dataset_final.csv")
host = pd.read_csv("../../listings_cleaned.csv.gz", compression="gzip", low_memory=False)[["id", "host_id"]]
df = df.merge(host, left_on="listing_id", right_on="id", how="left").drop(columns=["id"])
hotels = pd.read_csv("../../hotels_taipei_osm.csv")

# ---------- A1 飯店密度（BallTree haversine，半徑內計數） ----------
EARTH_KM = 6371.0088
htree = BallTree(np.radians(hotels[["lat", "lon"]].values), metric="haversine")
listing_rad = np.radians(df[["latitude", "longitude"]].values)
df["hotel_count_1km"] = htree.query_radius(listing_rad, r=1.0 / EARTH_KM, count_only=True)
df["hotel_count_500m"] = htree.query_radius(listing_rad, r=0.5 / EARTH_KM, count_only=True)

# ---------- A4 房間級衍生比值 ----------
acc = df["accommodates"].clip(lower=1)
df["price_per_person"] = df["price"] / acc
df["price_per_bedroom"] = df["price"] / df["bedrooms"].clip(lower=1)
df["beds_per_person"] = df["beds"] / acc

# ---------- A2 供需競爭比 ----------
df["airbnb_hotel_supply_ratio"] = df["nbr_density_1km"] / (df["hotel_count_1km"] + 1)

A4 = ["price_per_person", "price_per_bedroom", "beds_per_person"]
A1 = ["hotel_count_1km", "hotel_count_500m"]
A2 = ["airbnb_hotel_supply_ratio"]
NEW = A4 + A1 + A2

# base = 既有最終特徵集（排除 id/Y/host_id 與 photo_ 但保留 photo_design_sense）
EXCLUDE = ["listing_id", "Y_vacancy", "host_id"] + NEW
base = [c for c in df.columns if c not in EXCLUDE
        and (not c.startswith("photo_") or c == "photo_design_sense")]
print("base 特徵數:", len(base))
print("飯店密度統計: 1km 中位數 {:.0f} / 平均 {:.1f} / 最大 {} ; 500m 中位數 {:.0f}".format(
    df["hotel_count_1km"].median(), df["hotel_count_1km"].mean(),
    df["hotel_count_1km"].max(), df["hotel_count_500m"].median()))
for f in NEW:
    print("  新特徵 {:26s} 與 Y_vacancy 相關 = {:+.4f}".format(f, df[f].corr(df["Y_vacancy"])))

y = df["Y_vacancy"].values
yb = (y > 0.7).astype(int)
groups = df["host_id"].values

REG = dict(max_iter=500, learning_rate=0.05, l2_regularization=1.0, random_state=42)
CLF = dict(max_iter=500, learning_rate=0.05, l2_regularization=1.0, random_state=42)

# ---------- 單次切分 ----------
def single_split(cols, tag):
    Xtr, Xte, ytr, yte = train_test_split(df[cols], y, test_size=0.2, random_state=42)
    m = HistGradientBoostingRegressor(**REG).fit(Xtr, ytr)
    p = np.clip(m.predict(Xte), 0, 1)
    mae, mse, r2 = mean_absolute_error(yte, p), mean_squared_error(yte, p), r2_score(yte, p)
    # 分類
    Xtr, Xte, ytr, yte = train_test_split(df[cols], yb, test_size=0.2, stratify=yb, random_state=42)
    Xtr2, Xv, ytr2, yv = train_test_split(Xtr, ytr, test_size=0.2, stratify=ytr, random_state=42)
    cl = HistGradientBoostingClassifier(**CLF).fit(Xtr2, ytr2)
    cal = CalibratedClassifierCV(FrozenEstimator(cl), method="isotonic").fit(Xv, yv)
    p = cal.predict_proba(Xte)[:, 1]; pv = cal.predict_proba(Xv)[:, 1]; best = None
    for t in np.round(np.arange(0.1, 0.9, 0.01), 2):
        pr = (pv >= t).astype(int); rc = recall_score(yv, pr); pc = precision_score(yv, pr, zero_division=0)
        if rc >= 0.80 and (best is None or pc > best[2]): best = (t, rc, pc)
    pred = (p >= best[0]).astype(int)
    auc, rec, pre, f1 = (roc_auc_score(yte, p), recall_score(yte, pred),
                         precision_score(yte, pred, zero_division=0), f1_score(yte, pred))
    print("  [{:9s}] 回歸 MAE {:.4f} MSE {:.4f} R^2 {:.3f} | 分類 AUC {:.3f} Recall {:.3f} Prec {:.3f} F1 {:.3f}".format(
        tag, mae, mse, r2, auc, rec, pre, f1))

# ---------- GroupKFold ----------
def group_cv(cols, tag):
    X = df[cols]; gkf = GroupKFold(n_splits=5)
    r2s, maes, aucs = [], [], []
    for tr, te in gkf.split(X, y, groups):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        m = HistGradientBoostingRegressor(**REG).fit(Xtr, y[tr])
        p = np.clip(m.predict(Xte), 0, 1)
        r2s.append(r2_score(y[te], p)); maes.append(mean_absolute_error(y[te], p))
        if yb[tr].sum() >= 5 and yb[te].sum() >= 5:
            c = HistGradientBoostingClassifier(**CLF).fit(Xtr, yb[tr])
            aucs.append(roc_auc_score(yb[te], c.predict_proba(Xte)[:, 1]))
    r2s, maes, aucs = np.array(r2s), np.array(maes), np.array(aucs)
    print("  [{:9s}] 回歸 R^2 {:.3f}±{:.3f} MAE {:.4f} | 分類 AUC {:.3f}±{:.3f}".format(
        tag, r2s.mean(), r2s.std(), maes.mean(), aucs.mean(), aucs.std()))

print("\n===== ① 單次切分 80/20 =====")
for cols, tag in [(base, "base"), (base + A4, "+A4"), (base + A1 + A2, "+A1A2"), (base + NEW, "+ALL")]:
    single_split(cols, tag)

print("\n===== ② GroupKFold(依 host) 5 折 — 誠實(新房東)評估 =====")
for cols, tag in [(base, "base"), (base + A4, "+A4"), (base + A1 + A2, "+A1A2"), (base + NEW, "+ALL")]:
    group_cv(cols, tag)

# ---------- 新特徵重要度 ----------
print("\n===== +ALL 模型：新特徵 permutation 重要度排名 =====")
cols = base + NEW
Xtr, Xte, ytr, yte = train_test_split(df[cols], y, test_size=0.2, random_state=42)
m = HistGradientBoostingRegressor(**REG).fit(Xtr, ytr)
imp = permutation_importance(m, Xte, yte, n_repeats=5, random_state=42, scoring="r2")
rank = list(np.argsort(imp.importances_mean)[::-1])
for f in NEW:
    i = cols.index(f)
    print("  {:26s} 第 {:2d}/{} 名  (貢獻 {:+.4f})".format(f, rank.index(i) + 1, len(cols), imp.importances_mean[i]))
