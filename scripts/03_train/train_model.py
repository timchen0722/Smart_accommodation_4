# -*- coding: utf-8 -*-
"""
模型訓練示範：LR 基準 vs 樹模型(HistGradientBoosting) + 機率校準 + 門檻選擇 + 特徵重要度
輸入：dataset_train.csv
"""
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             recall_score, precision_score, f1_score,
                             confusion_matrix, brier_score_loss)

df = pd.read_csv("../../dataset_train.csv")
y = df["Y_stale"].values
X = df.drop(columns=["listing_id", "Y_stale"])
feat = X.columns.tolist()

# ① 切分：先切 test，再從 train 切 val（門檻校準用）
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_tr2, X_val, y_tr2, y_val = train_test_split(X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=42)
print("train {} / val {} / test {}  | stale rate test {:.1f}%".format(
      len(X_tr2), len(X_val), len(X_te), y_te.mean()*100))

def report(name, p, y_true, thr=0.5):
    pred = (p >= thr).astype(int)
    print("\n[{}] thr={:.2f}".format(name, thr))
    print("  AUC       {:.3f}".format(roc_auc_score(y_true, p)))
    print("  PR-AUC    {:.3f}".format(average_precision_score(y_true, p)))
    print("  Recall    {:.3f}".format(recall_score(y_true, pred)))
    print("  Precision {:.3f}".format(precision_score(y_true, pred)))
    print("  F1        {:.3f}".format(f1_score(y_true, pred)))

# ② 基準：Logistic Regression（需標準化）
sc = StandardScaler().fit(X_tr2)
lr = LogisticRegression(max_iter=1000, class_weight="balanced")
lr.fit(sc.transform(X_tr2), y_tr2)
p_lr = lr.predict_proba(sc.transform(X_te))[:, 1]
report("LogisticRegression (baseline)", p_lr, y_te)

# ③ 主力：樹模型 HistGradientBoosting（不需標準化）
gb = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
        max_depth=None, l2_regularization=1.0, random_state=42)
gb.fit(X_tr2, y_tr2)
p_gb = gb.predict_proba(X_te)[:, 1]
report("HistGradientBoosting (raw)", p_gb, y_te)

# ④ 機率校準（isotonic）→ 讓分數=真實機率
from sklearn.frozen import FrozenEstimator
cal = CalibratedClassifierCV(FrozenEstimator(gb), method="isotonic")
cal.fit(X_val, y_val)
p_cal = cal.predict_proba(X_te)[:, 1]
print("\nBrier score  raw {:.4f} -> calibrated {:.4f}  (越低越準)".format(
      brier_score_loss(y_te, p_gb), brier_score_loss(y_te, p_cal)))
report("HistGB + Calibrated", p_cal, y_te)

# ⑤ 用驗證集挑門檻：固定 Recall>=0.80 的最高精確率切點
p_val = cal.predict_proba(X_val)[:, 1]
best = None
for thr in np.round(np.arange(0.10, 0.90, 0.01), 2):
    pred = (p_val >= thr).astype(int)
    rec = recall_score(y_val, pred); prec = precision_score(y_val, pred, zero_division=0)
    if rec >= 0.80 and (best is None or prec > best[2]):
        best = (thr, rec, prec)
thr = best[0]
print("\n>> 校準門檻(驗證集 Recall>=0.80): thr={:.2f} (val recall {:.2f}, prec {:.2f})".format(*best))
report("最終模型 @ 校準門檻", p_cal, y_te, thr=thr)
cm = confusion_matrix(y_te, (p_cal >= thr).astype(int))
print("  混淆矩陣 [[TN FP],[FN TP]] =", cm.tolist())

# ⑥ 特徵重要度（permutation，前12名）
imp = permutation_importance(gb, X_te, y_te, n_repeats=5, random_state=42, scoring="roc_auc")
order = np.argsort(imp.importances_mean)[::-1][:12]
print("\n特徵重要度 Top12 (對 AUC 的貢獻):")
for i in order:
    print("  {:28s} {:.4f}".format(feat[i], imp.importances_mean[i]))

# ⑦ 洩漏稽核：拿掉口碑評分(需有評論才有) 再訓練一次
susp = [c for c in feat if c.startswith("review_scores")]
Xa_tr, Xa_te = X_tr2.drop(columns=susp), X_te.drop(columns=susp)
gb2 = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
        l2_regularization=1.0, random_state=42).fit(Xa_tr, y_tr2)
print("\n[洩漏稽核] 拿掉 review_scores_* 後 AUC: {:.3f} (原 {:.3f})".format(
      roc_auc_score(y_te, gb2.predict_proba(Xa_te)[:,1]), roc_auc_score(y_te, p_gb)))
