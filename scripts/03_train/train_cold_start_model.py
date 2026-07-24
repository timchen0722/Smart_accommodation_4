# -*- coding: utf-8 -*-
"""
「真正冷啟動」模型（24特徵：排除7個房東身分特徵 + 7個評分特徵 + score_pctl_nbhd）
用 GroupKFold(by host_id) 驗證，並與先前「30特徵版」(僅排除房東身分特徵)對照，
確認多排除review_scores_*後，分數差多少。
"""
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score, average_precision_score

df = pd.read_csv("../../dataset_final.csv")
L = pd.read_csv("../../listings_cleaned.csv.gz", compression="gzip", low_memory=False)[["id", "host_id"]]
df = df.merge(L, left_on="listing_id", right_on="id", how="left")

FINAL = [c for c in df.columns if c not in ["listing_id", "Y_vacancy", "id", "host_id"]
         and (not c.startswith("photo_") or c == "photo_design_sense")]

HOST_HISTORY = ["host_is_superhost", "host_response_rate", "host_acceptance_rate",
                 "host_tenure_days", "response_speed"]
REVIEW_DEPENDENT = ["review_scores_rating", "review_scores_cleanliness", "review_scores_location",
                     "review_scores_value", "review_scores_communication", "review_scores_checkin",
                     "review_scores_accuracy", "score_pctl_nbhd"]

COLD24 = [c for c in FINAL if c not in HOST_HISTORY + REVIEW_DEPENDENT]
PREV30 = [c for c in FINAL if c not in HOST_HISTORY]  # 先前版本：只排除房東身分特徵

print("目前37特徵版 -> 冷啟動24特徵版（排除 {} 個房東身分 + {} 個評分依賴）".format(
      len(HOST_HISTORY), len(REVIEW_DEPENDENT)))
print("冷啟動24特徵清單:", COLD24)

y = df["Y_vacancy"].values
yb = (y > 0.7).astype(int)
groups = df["host_id"].values

def group_cv(cols, tag):
    X = df[cols]
    gkf = GroupKFold(n_splits=5)
    r2s, maes, aucs, praucs = [], [], [], []
    for tr, te in gkf.split(X, y, groups):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        m = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, l2_regularization=1.0, random_state=42).fit(Xtr, y[tr])
        p = np.clip(m.predict(Xte), 0, 1)
        r2s.append(r2_score(y[te], p)); maes.append(mean_absolute_error(y[te], p))
        ybtr, ybte = yb[tr], yb[te]
        if ybtr.sum() >= 5 and ybte.sum() >= 5:
            c = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, l2_regularization=1.0, random_state=42).fit(Xtr, ybtr)
            p2 = c.predict_proba(Xte)[:, 1]
            aucs.append(roc_auc_score(ybte, p2)); praucs.append(average_precision_score(ybte, p2))
    r2s, maes, aucs, praucs = map(np.array, [r2s, maes, aucs, praucs])
    print("\n[{}] ({}特徵)".format(tag, len(cols)))
    print("   回歸  R^2 = {:.3f} ± {:.3f}   MAE = {:.3f} ± {:.3f}".format(r2s.mean(), r2s.std(), maes.mean(), maes.std()))
    print("   分類  AUC = {:.3f} ± {:.3f}   PR-AUC = {:.3f} ± {:.3f}".format(aucs.mean(), aucs.std(), praucs.mean(), praucs.std()))
    return r2s.mean(), aucs.mean()

print("\n" + "="*60)
r2_full, auc_full = group_cv(FINAL, "A) 現狀完整版")
r2_30, auc_30 = group_cv(PREV30, "B) 先前版(僅排除房東身分,30特徵)")
r2_24, auc_24 = group_cv(COLD24, "C) 真正冷啟動版(排除房東身分+評分,24特徵)")

print("\n" + "="*60)
print("摘要比較:")
print("  A) 完整版(37)         回歸R^2={:.3f}  分類AUC={:.3f}".format(r2_full, auc_full))
print("  B) 先前冷啟動(30)     回歸R^2={:.3f}  分類AUC={:.3f}".format(r2_30, auc_30))
print("  C) 真正冷啟動(24)     回歸R^2={:.3f}  分類AUC={:.3f}".format(r2_24, auc_24))
