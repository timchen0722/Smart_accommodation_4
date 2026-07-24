# -*- coding: utf-8 -*-
"""
最終模型驗收 — 補齊剩餘兩項標準：
④ 交叉驗證穩定度（GroupKFold by host_id，避免同房東房源跨train/test洩漏）
⑧ 子群體公平性（依 room_type、行政區拆開看表現）
最終特徵集：36原特徵 + photo_design_sense（37特徵，統一版）
"""
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (mean_absolute_error, r2_score, roc_auc_score,
                             average_precision_score, recall_score, precision_score)

df = pd.read_csv("../../dataset_final.csv")
L = pd.read_csv("../../listings_cleaned.csv.gz", compression="gzip", low_memory=False)[["id","host_id","room_type","neighbourhood_cleansed"]]
df = df.merge(L, left_on="listing_id", right_on="id", how="left")

FINAL = [c for c in df.columns if c not in
         ["listing_id","Y_vacancy","id","host_id","room_type","neighbourhood_cleansed"]
         and (not c.startswith("photo_") or c == "photo_design_sense")]
print("最終統一特徵集：{} 個".format(len(FINAL)))
X = df[FINAL]; y = df["Y_vacancy"].values
groups = df["host_id"].values

# ============================================================
# ④ GroupKFold 交叉驗證穩定度（5折，依房東分組）
# ============================================================
print("\n" + "="*60); print("④ 交叉驗證穩定度（GroupKFold, 5折, 依 host_id 分組）"); print("="*60)

gkf = GroupKFold(n_splits=5)
reg_r2, reg_mae = [], []
clf_auc, clf_prauc = [], []
yb_full = (y > 0.7).astype(int)

for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups), 1):
    Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
    ytr, yte = y[tr_idx], y[te_idx]

    # 回歸
    m = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, l2_regularization=1.0, random_state=42).fit(Xtr, ytr)
    p = np.clip(m.predict(Xte), 0, 1)
    r2, mae = r2_score(yte, p), mean_absolute_error(yte, p)
    reg_r2.append(r2); reg_mae.append(mae)

    # 分類
    ybtr, ybte = yb_full[tr_idx], yb_full[te_idx]
    if ybtr.sum() < 5 or ybte.sum() < 5:
        auc, prauc = np.nan, np.nan
    else:
        c = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, l2_regularization=1.0, random_state=42).fit(Xtr, ybtr)
        p2 = c.predict_proba(Xte)[:, 1]
        auc, prauc = roc_auc_score(ybte, p2), average_precision_score(ybte, p2)
    clf_auc.append(auc); clf_prauc.append(prauc)

    print("  第{}折: 回歸 R^2={:.3f} MAE={:.3f}  |  分類 AUC={:.3f} PR-AUC={:.3f}".format(fold, r2, mae, auc, prauc))

reg_r2, reg_mae = np.array(reg_r2), np.array(reg_mae)
clf_auc, clf_prauc = np.array(clf_auc), np.array(clf_prauc)
print("\n>> 回歸  R^2 = {:.3f} ± {:.3f}   MAE = {:.3f} ± {:.3f}".format(reg_r2.mean(), reg_r2.std(), reg_mae.mean(), reg_mae.std()))
print(">> 分類  AUC = {:.3f} ± {:.3f}   PR-AUC = {:.3f} ± {:.3f}".format(clf_auc.mean(), clf_auc.std(), clf_prauc.mean(), clf_prauc.std()))
print("（與單次split結果對照：回歸R^2=0.587、分類AUC=0.900 — 檢查是否在CV範圍內）")

# ============================================================
# ⑧ 子群體公平性（單次 80/20 split，訓練一個模型，依子群體切開評估）
# ============================================================
print("\n" + "="*60); print("⑧ 子群體公平性（依房型 / 行政區拆開看表現）"); print("="*60)

idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)
Xtr, Xte = X.iloc[idx_tr], X.iloc[idx_te]
ytr, yte = y[idx_tr], y[idx_te]
meta_te = df.iloc[idx_te][["room_type","neighbourhood_cleansed"]].reset_index(drop=True)

reg_model = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, l2_regularization=1.0, random_state=42).fit(Xtr, ytr)
pred_reg = np.clip(reg_model.predict(Xte), 0, 1)

ybtr_all, ybte_all = yb_full[idx_tr], yb_full[idx_te]
clf_model = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, l2_regularization=1.0, random_state=42).fit(Xtr, ybtr_all)
pred_clf = clf_model.predict_proba(Xte)[:, 1]

print("\n--- 依房型 room_type ---")
for rt, grp in meta_te.groupby("room_type"):
    m = grp.index.values
    if len(m) < 20: continue
    r2 = r2_score(yte[m], pred_reg[m]); mae = mean_absolute_error(yte[m], pred_reg[m])
    ybm = ybte_all[m]
    auc = roc_auc_score(ybm, pred_clf[m]) if ybm.sum() >= 5 and ybm.sum() < len(ybm) else float("nan")
    print("  {:18s} n={:4d}  回歸 R^2={:.3f} MAE={:.3f}  |  分類 AUC={:.3f}  高風險占比={:.1f}%".format(
        rt, len(m), r2, mae, auc, ybm.mean()*100))

print("\n--- 依行政區（樣本數Top6）---")
top_nb = meta_te["neighbourhood_cleansed"].value_counts().head(6).index
for nb in top_nb:
    m = meta_te.index[meta_te["neighbourhood_cleansed"] == nb].values
    r2 = r2_score(yte[m], pred_reg[m]); mae = mean_absolute_error(yte[m], pred_reg[m])
    ybm = ybte_all[m]
    auc = roc_auc_score(ybm, pred_clf[m]) if ybm.sum() >= 5 and ybm.sum() < len(ybm) else float("nan")
    print("  {:10s} n={:4d}  回歸 R^2={:.3f} MAE={:.3f}  |  分類 AUC={:.3f}  高風險占比={:.1f}%".format(
        nb, len(m), r2, mae, auc, ybm.mean()*100))
