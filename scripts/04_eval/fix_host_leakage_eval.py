# -*- coding: utf-8 -*-
"""
解決「房東層級特徵造成 GroupKFold 分數大幅下滑」問題 — 三方案對照實驗
A) 現狀（37特徵，含7個房東身分特徵）
B) 移除房東身分特徵（30特徵，只留房源本身條件）
C) 現狀特徵 + 加強正則化（限制模型記憶單一房東的能力）
全部用 GroupKFold(by host_id) 評估，避免同房東跨 train/test 洩漏。
"""
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score, average_precision_score

df = pd.read_csv("../../dataset_final.csv")
L = pd.read_csv("../../listings_cleaned.csv.gz", compression="gzip", low_memory=False)[["id", "host_id"]]
df = df.merge(L, left_on="listing_id", right_on="id", how="left")

HOST_IDENTITY = ["host_is_superhost", "host_response_rate", "host_acceptance_rate",
                  "host_listings_count", "host_tenure_days", "response_speed", "host_about_len"]

FINAL = [c for c in df.columns if c not in
         ["listing_id", "Y_vacancy", "id", "host_id"]
         and (not c.startswith("photo_") or c == "photo_design_sense")]
NO_HOST = [c for c in FINAL if c not in HOST_IDENTITY]

print("A) 現狀特徵集: {} 個".format(len(FINAL)))
print("B) 移除房東身分特徵後: {} 個 (拿掉 {})".format(len(NO_HOST), HOST_IDENTITY))

y = df["Y_vacancy"].values
yb = (y > 0.7).astype(int)
groups = df["host_id"].values

def group_cv(cols, tag, reg_kwargs=None, clf_kwargs=None):
    reg_kwargs = reg_kwargs or dict(max_iter=500, learning_rate=0.05, l2_regularization=1.0)
    clf_kwargs = clf_kwargs or dict(max_iter=500, learning_rate=0.05, l2_regularization=1.0)
    X = df[cols]
    gkf = GroupKFold(n_splits=5)
    r2s, maes, aucs, praucs = [], [], [], []
    for tr, te in gkf.split(X, y, groups):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        m = HistGradientBoostingRegressor(random_state=42, **reg_kwargs).fit(Xtr, y[tr])
        p = np.clip(m.predict(Xte), 0, 1)
        r2s.append(r2_score(y[te], p)); maes.append(mean_absolute_error(y[te], p))
        ybtr, ybte = yb[tr], yb[te]
        if ybtr.sum() >= 5 and ybte.sum() >= 5:
            c = HistGradientBoostingClassifier(random_state=42, **clf_kwargs).fit(Xtr, ybtr)
            p2 = c.predict_proba(Xte)[:, 1]
            aucs.append(roc_auc_score(ybte, p2)); praucs.append(average_precision_score(ybte, p2))
    r2s, maes = np.array(r2s), np.array(maes)
    aucs, praucs = np.array(aucs), np.array(praucs)
    print("[{}] ({}特徵)".format(tag, len(cols)))
    print("   回歸  R^2 = {:.3f} ± {:.3f}   MAE = {:.3f} ± {:.3f}".format(r2s.mean(), r2s.std(), maes.mean(), maes.std()))
    print("   分類  AUC = {:.3f} ± {:.3f}   PR-AUC = {:.3f} ± {:.3f}".format(aucs.mean(), aucs.std(), praucs.mean(), praucs.std()))
    return r2s.mean(), aucs.mean()

print("\n" + "="*60)
r2_a, auc_a = group_cv(FINAL, "A) 現狀（含房東身分特徵）")
print()
r2_b, auc_b = group_cv(NO_HOST, "B) 移除房東身分特徵")
print()
# C) 加強正則化：降低樹複雜度、提高L2、限制迭代
reg_reg = dict(max_iter=200, learning_rate=0.05, l2_regularization=5.0, max_depth=4, max_leaf_nodes=15)
clf_reg = dict(max_iter=200, learning_rate=0.05, l2_regularization=5.0, max_depth=4, max_leaf_nodes=15)
r2_c, auc_c = group_cv(FINAL, "C) 現狀特徵+加強正則化", reg_kwargs=reg_reg, clf_kwargs=clf_reg)

print("\n" + "="*60)
print("三方案 GroupKFold 摘要比較:")
print("  A) 現狀              回歸R^2={:.3f}  分類AUC={:.3f}".format(r2_a, auc_a))
print("  B) 移除房東身分特徵    回歸R^2={:.3f}  分類AUC={:.3f}".format(r2_b, auc_b))
print("  C) 加強正則化          回歸R^2={:.3f}  分類AUC={:.3f}".format(r2_c, auc_c))
