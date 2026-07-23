# -*- coding: utf-8 -*-
"""
用「誠實」的 GroupKFold(by host_id) 做超參數搜尋，
避免調參調到被房東洩漏虛高的分數上（那樣調出來的參數在真實新房東身上沒用）。
比較：預設參數 vs 調參後參數，在GroupKFold下的R^2/AUC。
"""
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score, make_scorer

df = pd.read_csv("../../dataset_final.csv")
L = pd.read_csv("../../listings_cleaned.csv.gz", compression="gzip", low_memory=False)[["id", "host_id"]]
df = df.merge(L, left_on="listing_id", right_on="id", how="left")

FINAL = [c for c in df.columns if c not in ["listing_id", "Y_vacancy", "id", "host_id"]
         and (not c.startswith("photo_") or c == "photo_design_sense")]
X = df[FINAL]
y = df["Y_vacancy"].values
yb = (y > 0.7).astype(int)
groups = df["host_id"].values

gkf_splits = list(GroupKFold(n_splits=5).split(X, y, groups))

# ============================================================
# 回歸：超參數搜尋（在GroupKFold下）
# ============================================================
print("="*60); print("回歸模型超參數搜尋（RandomizedSearchCV, GroupKFold評分）"); print("="*60)

param_dist = {
    "max_iter": [100, 200, 300, 500, 800],
    "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1, 0.15],
    "max_depth": [None, 3, 4, 6, 8],
    "max_leaf_nodes": [7, 15, 31, 63],
    "l2_regularization": [0, 0.1, 0.5, 1.0, 2.0, 5.0],
    "min_samples_leaf": [5, 10, 20, 40, 80],
}
base_r2 = []
for tr, te in gkf_splits:
    m = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, l2_regularization=1.0, random_state=42).fit(X.iloc[tr], y[tr])
    base_r2.append(r2_score(y[te], np.clip(m.predict(X.iloc[te]), 0, 1)))
print("調參前(預設參數)  GroupKFold R^2 = {:.3f} ± {:.3f}".format(np.mean(base_r2), np.std(base_r2)))

search = RandomizedSearchCV(
    HistGradientBoostingRegressor(random_state=42), param_dist, n_iter=25,
    cv=gkf_splits, scoring="r2", random_state=42, n_jobs=-1)
search.fit(X, y)
print("調參後最佳參數:", search.best_params_)
print("調參後  GroupKFold R^2 = {:.3f}".format(search.best_score_))

# ============================================================
# 分類：超參數搜尋（在GroupKFold下）
# ============================================================
print("\n" + "="*60); print("分類模型超參數搜尋（RandomizedSearchCV, GroupKFold評分）"); print("="*60)

base_auc = []
for tr, te in gkf_splits:
    ybtr, ybte = yb[tr], yb[te]
    if ybtr.sum() < 5 or ybte.sum() < 5: continue
    c = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, l2_regularization=1.0, random_state=42).fit(X.iloc[tr], ybtr)
    base_auc.append(roc_auc_score(ybte, c.predict_proba(X.iloc[te])[:, 1]))
print("調參前(預設參數)  GroupKFold AUC = {:.3f} ± {:.3f}".format(np.mean(base_auc), np.std(base_auc)))

search_c = RandomizedSearchCV(
    HistGradientBoostingClassifier(random_state=42), param_dist, n_iter=25,
    cv=gkf_splits, scoring="roc_auc", random_state=42, n_jobs=-1)
search_c.fit(X, yb)
print("調參後最佳參數:", search_c.best_params_)
print("調參後  GroupKFold AUC = {:.3f}".format(search_c.best_score_))

print("\n" + "="*60)
print("摘要：調參是否有效提升「誠實」的泛化分數")
print("  回歸  R^2   {:.3f} -> {:.3f}".format(np.mean(base_r2), search.best_score_))
print("  分類  AUC   {:.3f} -> {:.3f}".format(np.mean(base_auc), search_c.best_score_))
