# -*- coding: utf-8 -*-
"""台北官方 POI → 密度+最近距離特徵，分組消融雙評估（完整＋冷啟動）。"""
import numpy as np, pandas as pd
from sklearn.neighbors import BallTree
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.inspection import permutation_importance
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             roc_auc_score, recall_score, precision_score, f1_score)
import sys as _sys; _sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..', '01_data_build'))
from load_taipei_poi import load_all_poi

EARTH_KM = 6371.0088

# ---- 載入資料與 host_id ----
df = pd.read_csv("../../dataset_final.csv")
host = pd.read_csv("../../listings_cleaned.csv.gz", compression="gzip", low_memory=False)[["id", "host_id"]]
df = df.merge(host, left_on="listing_id", right_on="id", how="left").drop(columns=["id"])
poi = load_all_poi()
listing_rad = np.radians(df[["latitude", "longitude"]].values)

# ---- 每來源建 density + nearest ----
SPEC = {
    "mrt":    ([],            True),
    "bus":    ([500],         True),
    "rest":   ([500, 1000],   True),
    "cvs":    ([500, 1000],   True),
    "park":   ([1000],        True),
    "school": ([1000],        True),
    "pharm":  ([1000],        True),
}
EXTRA_1KM = {"mrt", "park", "school", "pharm"}  # 這些要 *_count_1km


def radius_count(tree, r_m):
    return tree.query_radius(listing_rad, r=(r_m / 1000.0) / EARTH_KM, count_only=True)


ALL_NEW = []
for key, arr in poi.items():
    tree = BallTree(np.radians(arr), metric="haversine")
    radii, want_nearest = SPEC[key]
    for r_m in radii:
        col = f"{key}_count_{'500m' if r_m == 500 else '1km'}"
        df[col] = radius_count(tree, r_m); ALL_NEW.append(col)
    if key in EXTRA_1KM and f"{key}_count_1km" not in df.columns:
        col = f"{key}_count_1km"; df[col] = radius_count(tree, 1000); ALL_NEW.append(col)
    if want_nearest:
        dist, _ = tree.query(listing_rad, k=1)
        col = f"{key}_nearest_km"; df[col] = dist[:, 0] * EARTH_KM; ALL_NEW.append(col)

GROUPS = {
    "交通":     [c for c in ALL_NEW if c.startswith(("mrt_", "bus_"))],
    "餐飲超商": [c for c in ALL_NEW if c.startswith(("rest_", "cvs_"))],
    "社區":     [c for c in ALL_NEW if c.startswith(("park_", "school_", "pharm_"))],
}

print("新特徵數:", len(ALL_NEW))
assert not df[ALL_NEW].isna().any().any(), "新特徵有 NaN"
assert np.isfinite(df[ALL_NEW].to_numpy()).all(), "新特徵有 inf"
print("\n各新特徵與 Y_vacancy 相關性:")
for c in sorted(ALL_NEW, key=lambda c: -abs(df[c].corr(df["Y_vacancy"]))):
    print(f"  {c:20s} {df[c].corr(df['Y_vacancy']):+.4f}")

# ---- base 與目標 ----
full_base = [c for c in df.columns
             if c not in (["listing_id", "Y_vacancy", "host_id"] + ALL_NEW)
             and (not c.startswith("photo_") or c == "photo_design_sense")]
y = df["Y_vacancy"].values; yb = (y > 0.7).astype(int); groups = df["host_id"].values
REG = dict(max_iter=500, learning_rate=0.05, l2_regularization=1.0, random_state=42)
CLF = dict(max_iter=500, learning_rate=0.05, l2_regularization=1.0, random_state=42)


def single_split(cols, tag):
    Xtr, Xte, ytr, yte = train_test_split(df[cols], y, test_size=0.2, random_state=42)
    m = HistGradientBoostingRegressor(**REG).fit(Xtr, ytr)
    p = np.clip(m.predict(Xte), 0, 1)
    mae, mse, r2 = mean_absolute_error(yte, p), mean_squared_error(yte, p), r2_score(yte, p)
    Xtr, Xte, ytr, yte = train_test_split(df[cols], yb, test_size=0.2, stratify=yb, random_state=42)
    Xtr2, Xv, ytr2, yv = train_test_split(Xtr, ytr, test_size=0.2, stratify=ytr, random_state=42)
    cl = HistGradientBoostingClassifier(**CLF).fit(Xtr2, ytr2)
    cal = CalibratedClassifierCV(FrozenEstimator(cl), method="isotonic").fit(Xv, yv)
    p = cal.predict_proba(Xte)[:, 1]; pv = cal.predict_proba(Xv)[:, 1]; best = None
    for t in np.round(np.arange(0.1, 0.9, 0.01), 2):
        pr = (pv >= t).astype(int); rc = recall_score(yv, pr); pc = precision_score(yv, pr, zero_division=0)
        if rc >= 0.80 and (best is None or pc > best[2]): best = (t, rc, pc)
    pred = (p >= best[0]).astype(int)
    print("  [{:14s}] 回歸 MAE {:.4f} MSE {:.4f} R^2 {:.3f} | 分類 AUC {:.3f} Rec {:.3f} Prec {:.3f} F1 {:.3f}".format(
        tag, mae, mse, r2, roc_auc_score(yte, p), recall_score(yte, pred),
        precision_score(yte, pred, zero_division=0), f1_score(yte, pred)))


def group_cv(cols, tag):
    X = df[cols]; gkf = GroupKFold(n_splits=5); r2s, maes, aucs = [], [], []
    for tr, te in gkf.split(X, y, groups):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        m = HistGradientBoostingRegressor(**REG).fit(Xtr, y[tr])
        p = np.clip(m.predict(Xte), 0, 1)
        r2s.append(r2_score(y[te], p)); maes.append(mean_absolute_error(y[te], p))
        if yb[tr].sum() >= 5 and yb[te].sum() >= 5:
            c = HistGradientBoostingClassifier(**CLF).fit(Xtr, yb[tr])
            aucs.append(roc_auc_score(yb[te], c.predict_proba(Xte)[:, 1]))
    r2s, maes, aucs = np.array(r2s), np.array(maes), np.array(aucs)
    print("  [{:14s}] ({:2d}特徵) R^2 {:.3f}±{:.3f} MAE {:.4f} | AUC {:.3f}±{:.3f}".format(
        tag, len(cols), r2s.mean(), r2s.std(), maes.mean(), aucs.mean(), aucs.std()))


def ablation_sets(base):
    yield base, "base"
    for name, cols in GROUPS.items():
        yield base + cols, f"+{name}"
    yield base + ALL_NEW, "+全部"


print("\n===== 完整模型 ① 單次切分 80/20 =====")
for cols, tag in ablation_sets(full_base): single_split(cols, tag)
print("\n===== 完整模型 ② GroupKFold(依 host) 誠實評估 =====")
for cols, tag in ablation_sets(full_base): group_cv(cols, tag)

# ---- 冷啟動模型（移除 7 房東身分特徵）----
HOST_IDENTITY = ["host_is_superhost", "host_response_rate", "host_acceptance_rate",
                 "host_listings_count", "host_tenure_days", "response_speed", "host_about_len"]
nohost_base = [c for c in full_base if c not in HOST_IDENTITY]

print("\n===== 冷啟動模型（移除 7 房東身分特徵）② GroupKFold 誠實評估 =====")
for cols, tag in ablation_sets(nohost_base): group_cv(cols, tag)

# ---- 新特徵 permutation 重要度 ----
print("\n===== +全部 完整模型：新特徵 permutation 重要度排名 =====")
cols = full_base + ALL_NEW
Xtr, Xte, ytr, yte = train_test_split(df[cols], y, test_size=0.2, random_state=42)
m = HistGradientBoostingRegressor(**REG).fit(Xtr, ytr)
imp = permutation_importance(m, Xte, yte, n_repeats=5, random_state=42, scoring="r2")
rank = list(np.argsort(imp.importances_mean)[::-1])
for f in sorted(ALL_NEW, key=lambda f: rank.index(cols.index(f))):
    i = cols.index(f)
    print("  {:20s} 第 {:2d}/{} 名  (貢獻 {:+.4f})".format(f, rank.index(i) + 1, len(cols), imp.importances_mean[i]))
