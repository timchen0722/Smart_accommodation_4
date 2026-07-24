# -*- coding: utf-8 -*-
"""方案A(18特徵)分類：不分 group 的一般 StratifiedKFold(5) 評估（含房東洩漏，會虛高）。
與 GroupKFold 誠實版同格式對照。"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np, pandas as pd
from sklearn.neighbors import BallTree
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                             precision_score, recall_score, f1_score, confusion_matrix, brier_score_loss)
import sys as _sys; _sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..', '01_data_build'))
from load_taipei_poi import load_all_poi

EARTH_KM = 6371.0088
CLF = dict(max_iter=500, learning_rate=0.05, l2_regularization=1.0, random_state=42)

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

imp = pd.read_csv("../../selected_35_features_importance.csv")
FEATS = imp["feature"].tolist()[:18]
yb = (df["Y_vacancy"].values > 0.7).astype(int)
print(f"方案A 18 特徵｜不分 group｜StratifiedKFold(5, shuffle, seed=42)")
print(f"樣本 {len(df)}，高空房(正類) {yb.sum()} ({yb.mean()*100:.1f}%)\n")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_p = np.full(len(df), np.nan); aucs, aps = [], []
for tr, te in skf.split(df[FEATS], yb):
    c = HistGradientBoostingClassifier(**CLF).fit(df[FEATS].iloc[tr], yb[tr])
    pr = c.predict_proba(df[FEATS].iloc[te])[:, 1]; oof_p[te] = pr
    aucs.append(roc_auc_score(yb[te], pr)); aps.append(average_precision_score(yb[te], pr))
aucs, aps = np.array(aucs), np.array(aps)


def at_t(t):
    pred = (oof_p >= t).astype(int); tn, fp, fn, tp = confusion_matrix(yb, pred).ravel()
    return dict(t=t, acc=accuracy_score(yb, pred), prec=precision_score(yb, pred, zero_division=0),
                rec=recall_score(yb, pred), f1=f1_score(yb, pred), tn=tn, fp=fp, fn=fn, tp=tp)


print("===== 不分 group：StratifiedKFold(5) 評估（含多房源房東洩漏，會虛高）=====")
print(f"  ROC-AUC     : {aucs.mean():.4f} ± {aucs.std():.4f}   (各折 {', '.join(f'{v:.3f}' for v in aucs)})")
print(f"  PR-AUC(AP)  : {aps.mean():.4f} ± {aps.std():.4f}   (正類基準率 {yb.mean():.3f})")
print(f"  Brier score : {brier_score_loss(yb, oof_p):.4f}")
best = None
for t in np.round(np.arange(0.05, 0.95, 0.01), 2):
    r = at_t(t)
    if r["rec"] >= 0.80 and (best is None or r["prec"] > best["prec"]): best = r
for tag, r in [("門檻 0.50（預設）", at_t(0.50)),
               (f"門檻 {best['t']}（高召回, Recall≥0.80）", best)]:
    print(f"\n  【{tag}】")
    print(f"    Accuracy {r['acc']:.4f} | Precision {r['prec']:.4f} | Recall {r['rec']:.4f} | F1 {r['f1']:.4f}")
    print(f"    混淆矩陣: TP {r['tp']}  FP {r['fp']}  FN {r['fn']}  TN {r['tn']}")
