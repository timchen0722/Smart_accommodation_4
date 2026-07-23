# -*- coding: utf-8 -*-
"""用前向選擇排出的 35 個特徵重訓完整模型，出標準雙評估指標 + permutation 重要度（繁中名稱+排名）。"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
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
REG = dict(max_iter=500, learning_rate=0.05, l2_regularization=1.0, random_state=42)
CLF = dict(max_iter=500, learning_rate=0.05, l2_regularization=1.0, random_state=42)

# ---- 特徵繁體中文名稱 ----
ZH = {
    "beds": "床位數", "self_checkin": "自助入住", "maximum_nights": "最長入住晚數",
    "instant_bookable": "可即時預訂", "host_is_superhost": "是否超讚房東", "price": "價格",
    "review_scores_communication": "溝通評分", "host_acceptance_rate": "房東接受率",
    "min_nights_avg_ntm": "未來30天平均最短入住", "latitude": "緯度",
    "score_pctl_nbhd": "鄰里內評分百分位", "park_count_1km": "1km內公園數",
    "desc_len": "房源描述長度", "bedrooms": "臥室數", "minimum_nights": "最短入住晚數",
    "pharm_nearest_km": "最近藥局距離(km)", "accommodates": "可容納人數",
    "host_response_rate": "房東回覆率", "price_pctl_nbhd": "鄰里內價格百分位",
    "amenities_count": "設施數量", "pharm_count_1km": "1km內藥局數",
    "photo_design_sense": "照片設計感", "review_scores_location": "地點評分",
    "bus_count_500m": "500m內公車站數", "neighbourhood_code": "行政區編碼",
    "review_scores_value": "性價比評分", "school_count_1km": "1km內學校數",
    "rest_count_1km": "1km內餐廳數", "nbr_density_same_type_1km": "1km內同類型房源密度",
    "rest_nearest_km": "最近餐廳距離(km)", "park_nearest_km": "最近公園距離(km)",
    "review_scores_checkin": "入住評分", "review_scores_accuracy": "準確度評分",
    "review_scores_rating": "綜合評分", "longitude": "經度",
}

# ---- 載入資料與 host_id ----
df = pd.read_csv("../../dataset_final.csv")
host = pd.read_csv("../../listings_cleaned.csv.gz", compression="gzip", low_memory=False)[["id", "host_id"]]
df = df.merge(host, left_on="listing_id", right_on="id", how="left").drop(columns=["id"])
poi = load_all_poi()
listing_rad = np.radians(df[["latitude", "longitude"]].values)

SPEC = {"mrt": ([], True), "bus": ([500], True), "rest": ([500, 1000], True),
        "cvs": ([500, 1000], True), "park": ([1000], True), "school": ([1000], True), "pharm": ([1000], True)}
EXTRA_1KM = {"mrt", "park", "school", "pharm"}


def radius_count(tree, r_m):
    return tree.query_radius(listing_rad, r=(r_m / 1000.0) / EARTH_KM, count_only=True)


for key, arr in poi.items():
    tree = BallTree(np.radians(arr), metric="haversine")
    radii, want_nearest = SPEC[key]
    for r_m in radii:
        col = f"{key}_count_{'500m' if r_m == 500 else '1km'}"; df[col] = radius_count(tree, r_m)
    if key in EXTRA_1KM and f"{key}_count_1km" not in df.columns:
        df[f"{key}_count_1km"] = radius_count(tree, 1000)
    if want_nearest:
        dist, _ = tree.query(listing_rad, k=1); df[f"{key}_nearest_km"] = dist[:, 0] * EARTH_KM

# ---- 讀前向選擇排序，取 35 個特徵（保留加入順序）----
order = pd.read_csv("../../forward_selection_order.csv")
FEATS = order["feature"].tolist()
assert len(FEATS) == 35 and all(f in df.columns for f in FEATS), "特徵缺漏"
y = df["Y_vacancy"].values; yb = (y > 0.7).astype(int); groups = df["host_id"].values
print(f"重訓特徵數: {len(FEATS)}  資料 {len(df)} 筆, 房東 {df['host_id'].nunique()} 個")

# ========== ① 單次隨機切分 80/20 ==========
print("\n===== ① 單次切分 80/20（含多房源房東洩漏，會虛高）=====")
Xtr, Xte, ytr, yte = train_test_split(df[FEATS], y, test_size=0.2, random_state=42)
m = HistGradientBoostingRegressor(**REG).fit(Xtr, ytr)
p = np.clip(m.predict(Xte), 0, 1)
mae, mse, r2 = mean_absolute_error(yte, p), mean_squared_error(yte, p), r2_score(yte, p)
Xtr, Xte, ytr, yte = train_test_split(df[FEATS], yb, test_size=0.2, stratify=yb, random_state=42)
Xtr2, Xv, ytr2, yv = train_test_split(Xtr, ytr, test_size=0.2, stratify=ytr, random_state=42)
cl = HistGradientBoostingClassifier(**CLF).fit(Xtr2, ytr2)
cal = CalibratedClassifierCV(FrozenEstimator(cl), method="isotonic").fit(Xv, yv)
pp = cal.predict_proba(Xte)[:, 1]; pv = cal.predict_proba(Xv)[:, 1]; best = None
for t in np.round(np.arange(0.1, 0.9, 0.01), 2):
    pr = (pv >= t).astype(int); rc = recall_score(yv, pr); pc = precision_score(yv, pr, zero_division=0)
    if rc >= 0.80 and (best is None or pc > best[2]): best = (t, rc, pc)
pred = (pp >= best[0]).astype(int)
print(f"  回歸: MAE {mae:.4f}  MSE {mse:.4f}  R^2 {r2:.4f}")
print(f"  分類: AUC {roc_auc_score(yte, pp):.4f}  Recall {recall_score(yte, pred):.4f}  "
      f"Precision {precision_score(yte, pred, zero_division=0):.4f}  F1 {f1_score(yte, pred):.4f} (門檻 {best[0]})")

# ========== ② GroupKFold(依 host) 誠實評估 ==========
print("\n===== ② GroupKFold(依 host_id 5 折) 誠實評估 =====")
gkf = GroupKFold(n_splits=5); r2s, maes, mses, aucs = [], [], [], []
for tr, te in gkf.split(df[FEATS], y, groups):
    Xtr, Xte = df[FEATS].iloc[tr], df[FEATS].iloc[te]
    m = HistGradientBoostingRegressor(**REG).fit(Xtr, y[tr]); p = np.clip(m.predict(Xte), 0, 1)
    r2s.append(r2_score(y[te], p)); maes.append(mean_absolute_error(y[te], p)); mses.append(mean_squared_error(y[te], p))
    if yb[tr].sum() >= 5 and yb[te].sum() >= 5:
        c = HistGradientBoostingClassifier(**CLF).fit(Xtr, yb[tr])
        aucs.append(roc_auc_score(yb[te], c.predict_proba(Xte)[:, 1]))
r2s, maes, mses, aucs = map(np.array, (r2s, maes, mses, aucs))
print(f"  回歸: R^2 {r2s.mean():.4f} ± {r2s.std():.4f}   MAE {maes.mean():.4f} ± {maes.std():.4f}   MSE {mses.mean():.4f}")
print(f"  分類: AUC {aucs.mean():.4f} ± {aucs.std():.4f}")
print(f"  各折 R^2: {', '.join(f'{v:.3f}' for v in r2s)}")

# ========== ③ Permutation 重要度（35 特徵排名）==========
print("\n===== ③ Permutation 重要度排名（held-out test, n_repeats=10, scoring=R^2）=====")
Xtr, Xte, ytr, yte = train_test_split(df[FEATS], y, test_size=0.2, random_state=42)
m = HistGradientBoostingRegressor(**REG).fit(Xtr, ytr)
imp = permutation_importance(m, Xte, yte, n_repeats=10, random_state=42, scoring="r2")
tbl = pd.DataFrame({
    "feature": FEATS,
    "中文名稱": [ZH[f] for f in FEATS],
    "重要度(R²下降)": imp.importances_mean,
    "std": imp.importances_std,
    "SFS加入序": order["step"].values,
}).sort_values("重要度(R²下降)", ascending=False).reset_index(drop=True)
tbl.insert(0, "排名", tbl.index + 1)
tot = tbl["重要度(R²下降)"].clip(lower=0).sum()
tbl["相對佔比%"] = (tbl["重要度(R²下降)"].clip(lower=0) / tot * 100).round(1)
tbl.to_csv("../../selected_35_features_importance.csv", index=False, encoding="utf-8-sig")

print(f"  {'排名':>3} {'中文名稱':<14} {'重要度':>10} {'±std':>8} {'佔比%':>6} {'SFS序':>5}")
for _, r in tbl.iterrows():
    print(f"  {int(r['排名']):>3}. {r['中文名稱']:<14} {r['重要度(R²下降)']:>+10.5f} {r['std']:>8.5f} "
          f"{r['相對佔比%']:>6.1f} {int(r['SFS加入序']):>5}")
print("\n→ selected_35_features_importance.csv")
