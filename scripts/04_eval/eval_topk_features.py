# -*- coding: utf-8 -*-
"""找最小充分子集：依 permutation 重要度排名，實測 top-K 的 GroupKFold 誠實 R²/AUC，
看最少幾個特徵即追平 35 特徵版（R²≈0.2492）。"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np, pandas as pd
from sklearn.neighbors import BallTree
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score
import sys as _sys; _sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..', '01_data_build'))
from load_taipei_poi import load_all_poi

EARTH_KM = 6371.0088
REG = dict(max_iter=500, learning_rate=0.05, l2_regularization=1.0, random_state=42)
CLF = dict(max_iter=500, learning_rate=0.05, l2_regularization=1.0, random_state=42)
FULL35_R2 = 0.2492  # 35 特徵版誠實 R²（基準）

# ---- 載入資料與 host_id + 16 POI ----
df = pd.read_csv("../../dataset_final.csv")
host = pd.read_csv("../../listings_cleaned.csv.gz", compression="gzip", low_memory=False)[["id", "host_id"]]
df = df.merge(host, left_on="listing_id", right_on="id", how="left").drop(columns=["id"])
poi = load_all_poi()
listing_rad = np.radians(df[["latitude", "longitude"]].values)
SPEC = {"mrt": ([], True), "bus": ([500], True), "rest": ([500, 1000], True),
        "cvs": ([500, 1000], True), "park": ([1000], True), "school": ([1000], True), "pharm": ([1000], True)}
EXTRA_1KM = {"mrt", "park", "school", "pharm"}
rc = lambda tree, r_m: tree.query_radius(listing_rad, r=(r_m / 1000.0) / EARTH_KM, count_only=True)
for key, arr in poi.items():
    tree = BallTree(np.radians(arr), metric="haversine"); radii, wn = SPEC[key]
    for r_m in radii:
        df[f"{key}_count_{'500m' if r_m == 500 else '1km'}"] = rc(tree, r_m)
    if key in EXTRA_1KM and f"{key}_count_1km" not in df.columns:
        df[f"{key}_count_1km"] = rc(tree, 1000)
    if wn:
        dist, _ = tree.query(listing_rad, k=1); df[f"{key}_nearest_km"] = dist[:, 0] * EARTH_KM

y = df["Y_vacancy"].values; yb = (y > 0.7).astype(int); groups = df["host_id"].values
imp = pd.read_csv("../../selected_35_features_importance.csv")  # 已依重要度排序
RANK = imp["feature"].tolist()
ZH = dict(zip(imp["feature"], imp["中文名稱"]))

splits = list(GroupKFold(n_splits=5).split(df[RANK], y, groups))


def eval_cols(cols):
    r2s, maes, aucs = [], [], []
    for tr, te in splits:
        Xtr, Xte = df[cols].iloc[tr], df[cols].iloc[te]
        m = HistGradientBoostingRegressor(**REG).fit(Xtr, y[tr]); p = np.clip(m.predict(Xte), 0, 1)
        r2s.append(r2_score(y[te], p)); maes.append(mean_absolute_error(y[te], p))
        if yb[tr].sum() >= 5 and yb[te].sum() >= 5:
            c = HistGradientBoostingClassifier(**CLF).fit(Xtr, yb[tr])
            aucs.append(roc_auc_score(yb[te], c.predict_proba(Xte)[:, 1]))
    return np.mean(r2s), np.std(r2s), np.mean(maes), np.mean(aucs)


Ks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 18, 20, 25, 35]
print(f"基準：35 特徵版 GroupKFold R² = {FULL35_R2:.4f}（±0.0512）")
print(f"「追平」判準：top-K R² ≥ {FULL35_R2 - 0.0512:.4f}（落在基準一個折間 std 內即視為同水準）\n")
print(f"  {'K':>3} {'R²':>8} {'±std':>7} {'MAE':>7} {'AUC':>7}  新增特徵")
rows = []
prev = set()
first_match = None
for K in Ks:
    cols = RANK[:K]
    r2, sd, mae, auc = eval_cols(cols)
    added = [ZH[c] for c in cols if c not in prev]; prev = set(cols)
    match = "✓同水準" if r2 >= FULL35_R2 - 0.0512 else ""
    if first_match is None and r2 >= FULL35_R2 - 0.0512:
        first_match = K
    print(f"  {K:>3} {r2:>8.4f} {sd:>7.4f} {mae:>7.4f} {auc:>7.4f}  +{'、'.join(added)} {match}")
    rows.append(dict(K=K, r2=r2, r2_std=sd, mae=mae, auc=auc))

pd.DataFrame(rows).to_csv("../../topk_sufficiency.csv", index=False, encoding="utf-8-sig")
print(f"\n最小追平 K（R² 進入基準 std 內）= {first_match}")
print("→ topk_sufficiency.csv")
