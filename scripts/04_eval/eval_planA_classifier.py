# -*- coding: utf-8 -*-
"""方案A（18特徵）分類模型完整評估：高空房(Y_vacancy>0.7)二元分類。
主：GroupKFold(依 host) 誠實；附：單次切分+校準 對照。"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np, pandas as pd
from sklearn.neighbors import BallTree
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                             precision_score, recall_score, f1_score, confusion_matrix,
                             brier_score_loss)
import sys as _sys; _sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..', '01_data_build'))
from load_taipei_poi import load_all_poi

EARTH_KM = 6371.0088
CLF = dict(max_iter=500, learning_rate=0.05, l2_regularization=1.0, random_state=42)

# ---- 載入資料 + 16 POI ----
df = pd.read_csv("../../dataset_final.csv")
host = pd.read_csv("../../listings_cleaned.csv.gz", compression="gzip", low_memory=False)[["id", "host_id"]]
df = df.merge(host, left_on="listing_id", right_on="id", how="left").drop(columns=["id"])
poi = load_all_poi(); listing_rad = np.radians(df[["latitude", "longitude"]].values)
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

# ---- 方案A：重要度前 18 ----
imp = pd.read_csv("../../selected_35_features_importance.csv")
FEATS = imp["feature"].tolist()[:18]
yb = (df["Y_vacancy"].values > 0.7).astype(int); groups = df["host_id"].values
print("方案A 18 特徵:", "、".join(imp["中文名稱"].tolist()[:18]))
print(f"樣本 {len(df)}，高空房(正類) {yb.sum()} ({yb.mean()*100:.1f}%)，房東 {df['host_id'].nunique()}\n")

# ========== 主：GroupKFold(依 host) OOF 誠實評估 ==========
gkf = GroupKFold(n_splits=5)
oof_p = np.full(len(df), np.nan); aucs, aps = [], []
for tr, te in gkf.split(df[FEATS], yb, groups):
    c = HistGradientBoostingClassifier(**CLF).fit(df[FEATS].iloc[tr], yb[tr])
    pr = c.predict_proba(df[FEATS].iloc[te])[:, 1]; oof_p[te] = pr
    aucs.append(roc_auc_score(yb[te], pr)); aps.append(average_precision_score(yb[te], pr))
aucs, aps = np.array(aucs), np.array(aps)


def at_threshold(t):
    pred = (oof_p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(yb, pred).ravel()
    return dict(t=t, acc=accuracy_score(yb, pred), prec=precision_score(yb, pred, zero_division=0),
                rec=recall_score(yb, pred), f1=f1_score(yb, pred), tn=tn, fp=fp, fn=fn, tp=tp)


print("===== 主結果：GroupKFold(依 host_id 5 折) 誠實分類評估 =====")
print(f"  ROC-AUC        : {aucs.mean():.4f} ± {aucs.std():.4f}   (各折 {', '.join(f'{v:.3f}' for v in aucs)})")
print(f"  PR-AUC(AP)     : {aps.mean():.4f} ± {aps.std():.4f}   (正類基準率 {yb.mean():.3f})")
print(f"  Brier score    : {brier_score_loss(yb, oof_p):.4f}   (越低越準)")

# 找達 recall>=0.80 的最高精確度門檻（產品：盡量抓出高空房）
best_hr = None
for t in np.round(np.arange(0.05, 0.95, 0.01), 2):
    r = at_threshold(t)
    if r["rec"] >= 0.80 and (best_hr is None or r["prec"] > best_hr["prec"]):
        best_hr = r
for tag, r in [("門檻 0.50（預設）", at_threshold(0.50)),
               (f"門檻 {best_hr['t']}（高召回營運點, Recall≥0.80）", best_hr)]:
    print(f"\n  【{tag}】")
    print(f"    Accuracy {r['acc']:.4f} | Precision {r['prec']:.4f} | Recall {r['rec']:.4f} | F1 {r['f1']:.4f}")
    print(f"    混淆矩陣: TP {r['tp']}  FP {r['fp']}  FN {r['fn']}  TN {r['tn']}")

# ========== 附：單次切分 + isotonic 校準（沿用專案協定，會虛高）==========
print("\n===== 附：單次切分 80/20 + isotonic 校準（含房東洩漏，僅供對照）=====")
Xtr, Xte, ytr, yte = train_test_split(df[FEATS], yb, test_size=0.2, stratify=yb, random_state=42)
Xtr2, Xv, ytr2, yv = train_test_split(Xtr, ytr, test_size=0.2, stratify=ytr, random_state=42)
cl = HistGradientBoostingClassifier(**CLF).fit(Xtr2, ytr2)
cal = CalibratedClassifierCV(FrozenEstimator(cl), method="isotonic").fit(Xv, yv)
p = cal.predict_proba(Xte)[:, 1]; pv = cal.predict_proba(Xv)[:, 1]; b = None
for t in np.round(np.arange(0.1, 0.9, 0.01), 2):
    pr = (pv >= t).astype(int); r = recall_score(yv, pr); pc = precision_score(yv, pr, zero_division=0)
    if r >= 0.80 and (b is None or pc > b[2]): b = (t, r, pc)
pred = (p >= b[0]).astype(int)
print(f"  ROC-AUC {roc_auc_score(yte, p):.4f} | PR-AUC {average_precision_score(yte, p):.4f} | "
      f"Brier {brier_score_loss(yte, p):.4f}")
print(f"  門檻 {b[0]}: Accuracy {accuracy_score(yte, pred):.4f} | Precision {precision_score(yte, pred, zero_division=0):.4f} "
      f"| Recall {recall_score(yte, pred):.4f} | F1 {f1_score(yte, pred):.4f}")
