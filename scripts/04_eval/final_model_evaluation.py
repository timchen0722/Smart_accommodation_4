# -*- coding: utf-8 -*-
"""
最終模型評估：dataset_final.csv（45特徵，含9個影像多模態特徵）
模型A：風險分數預估（回歸）
模型B：是否高風險（分類，Y_vacancy > 0.7）+ 機率校準 + 門檻選擇
並與「無影像特徵版」對照，確認影像特徵是否真的提升最終模型。
"""
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.inspection import permutation_importance
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             roc_auc_score, average_precision_score, recall_score,
                             precision_score, f1_score, confusion_matrix, brier_score_loss)

df = pd.read_csv("../../dataset_final.csv")
y = df["Y_vacancy"].values
X = df.drop(columns=["listing_id", "Y_vacancy"])
feat = X.columns.tolist()
photo_feat = [c for c in feat if c.startswith("photo_")]
base_feat = [c for c in feat if c not in photo_feat]
print("最終特徵集：{} 個（其中影像特徵 {} 個）\n".format(len(feat), len(photo_feat)))

# ============================================================
# 模型 A：風險分數預估（回歸）
# ============================================================
print("="*60); print("模型 A：風險分數預估（回歸，風險分數=預測空屋率）"); print("="*60)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

# 基準 Logistic/Linear 對照
sc = StandardScaler().fit(Xtr)
lin = LinearRegression().fit(sc.transform(Xtr), ytr)
p_lin = np.clip(lin.predict(sc.transform(Xte)), 0, 1)
print("[Baseline] LinearRegression      MAE {:.3f}  RMSE {:.3f}  R^2 {:.3f}".format(
      mean_absolute_error(yte, p_lin), np.sqrt(mean_squared_error(yte, p_lin)), r2_score(yte, p_lin)))

# 主力：不含影像特徵
Xtr_b, Xte_b = Xtr[base_feat], Xte[base_feat]
gb_b = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, l2_regularization=1.0, random_state=42).fit(Xtr_b, ytr)
p_b = np.clip(gb_b.predict(Xte_b), 0, 1)
print("[無影像特徵] HistGB ({}特徵)  MAE {:.3f}  RMSE {:.3f}  R^2 {:.3f}".format(
      len(base_feat), mean_absolute_error(yte, p_b), np.sqrt(mean_squared_error(yte, p_b)), r2_score(yte, p_b)))

# 主力：含影像特徵（最終版）
gb = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, l2_regularization=1.0, random_state=42).fit(Xtr, ytr)
p_gb = np.clip(gb.predict(Xte), 0, 1)
print("[最終版] HistGB ({}特徵，含影像)  MAE {:.3f}  RMSE {:.3f}  R^2 {:.3f}".format(
      len(feat), mean_absolute_error(yte, p_gb), np.sqrt(mean_squared_error(yte, p_gb)), r2_score(yte, p_gb)))

# 影像特徵在最終模型的重要度
imp = permutation_importance(gb, Xte, yte, n_repeats=5, random_state=42, scoring="r2")
rank = pd.Series(imp.importances_mean, index=feat).sort_values(ascending=False)
print("\n影像特徵在最終回歸模型的重要度排名（共{}個特徵）:".format(len(feat)))
for pf in photo_feat:
    r = list(rank.index).index(pf) + 1
    print("  {:20s} 第 {:2d} 名 (貢獻 {:.4f})".format(pf, r, rank[pf]))
print("\n回歸模型 Top10 特徵:")
for i, (f, v) in enumerate(rank.head(10).items(), 1):
    print("  {:2d}. {:22s} {:.4f}".format(i, f, v))

# ============================================================
# 模型 B：是否高風險（分類）+ 校準 + 門檻選擇
# ============================================================
print("\n" + "="*60); print("模型 B：是否高風險（分類，Y_vacancy > 0.7）"); print("="*60)
HI = 0.70
yb = (y > HI).astype(int)
print("高風險占比: {:.1f}%".format(yb.mean()*100))

def run_clf(cols, tag):
    Xtr2, Xte2, ytr2, yte2 = train_test_split(df[cols], yb, test_size=0.2, stratify=yb, random_state=42)
    Xtr3, Xv, ytr3, yv = train_test_split(Xtr2, ytr2, test_size=0.2, stratify=ytr2, random_state=42)
    clf = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, l2_regularization=1.0, random_state=42).fit(Xtr3, ytr3)
    cal = CalibratedClassifierCV(FrozenEstimator(clf), method="isotonic").fit(Xv, yv)
    p = cal.predict_proba(Xte2)[:, 1]
    pv = cal.predict_proba(Xv)[:, 1]
    best = None
    for t in np.round(np.arange(0.1, 0.9, 0.01), 2):
        pr = (pv >= t).astype(int)
        rc = recall_score(yv, pr); pc = precision_score(yv, pr, zero_division=0)
        if rc >= 0.80 and (best is None or pc > best[2]):
            best = (t, rc, pc)
    thr = best[0]
    pred = (p >= thr).astype(int)
    auc = roc_auc_score(yte2, p)
    pr_auc = average_precision_score(yte2, p)
    brier = brier_score_loss(yte2, p)
    print("[{}] ({}特徵) AUC {:.3f}  PR-AUC {:.3f}  Brier {:.4f}  |  門檻{:.2f} → Recall {:.3f} Precision {:.3f} F1 {:.3f}".format(
        tag, len(cols), auc, pr_auc, brier, thr, recall_score(yte2, pred), precision_score(yte2, pred), f1_score(yte2, pred)))
    cm = confusion_matrix(yte2, pred)
    print("   混淆矩陣 [[TN FP],[FN TP]] =", cm.tolist())
    return clf, Xte2, yte2

run_clf(base_feat, "無影像特徵")
clf_final, Xte_c, yte_c = run_clf(feat, "最終版(含影像)")

imp_c = permutation_importance(clf_final, Xte_c, yte_c, n_repeats=5, random_state=42, scoring="roc_auc")
rank_c = pd.Series(imp_c.importances_mean, index=feat).sort_values(ascending=False)
print("\n影像特徵在最終分類模型的重要度排名（共{}個特徵）:".format(len(feat)))
for pf in photo_feat:
    r = list(rank_c.index).index(pf) + 1
    print("  {:20s} 第 {:2d} 名 (貢獻 {:.4f})".format(pf, r, rank_c[pf]))
print("\n分類模型 Top10 特徵:")
for i, (f, v) in enumerate(rank_c.head(10).items(), 1):
    print("  {:2d}. {:22s} {:.4f}".format(i, f, v))
