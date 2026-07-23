# -*- coding: utf-8 -*-
"""
兩個模型 + 一個實驗
實驗：全域模型 vs 「分房源型別(room_type)」分開建模 → 準確度有無提升
模型A：風險分數預估（回歸，預測空屋率0~1 當風險分數）
模型B：是否高風險（分類，空屋率>0.7 → 高風險=1）+ 門檻校準
輸入：dataset_vacancy.csv
"""
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (mean_absolute_error, r2_score, roc_auc_score,
                             recall_score, precision_score, f1_score, confusion_matrix)

df = pd.read_csv("../../dataset_vacancy.csv")
X = df.drop(columns=["listing_id", "Y_vacancy"])
y = df["Y_vacancy"].values
feat = X.columns.tolist()
HI = 0.70  # 高風險門檻(真實空屋率)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

print("="*60)
print("實驗：全域 vs 分房源型別建模（回歸，看 MAE / R^2）")
print("="*60)
# 全域
g = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, l2_regularization=1.0, random_state=42).fit(Xtr, ytr)
pg = np.clip(g.predict(Xte), 0, 1)
print("全域模型      : MAE {:.3f}  R^2 {:.3f}".format(mean_absolute_error(yte, pg), r2_score(yte, pg)))
# 分 room_type 分開建模（room_type_code 分群）
rt_tr, rt_te = Xtr["room_type_code"].values, Xte["room_type_code"].values
pred_seg = np.zeros(len(Xte))
for code in np.unique(rt_tr):
    m_tr, m_te = rt_tr == code, rt_te == code
    if m_te.sum() == 0: continue
    if m_tr.sum() < 50:  # 樣本太少 → 用全域模型預測
        pred_seg[m_te] = pg[m_te]; continue
    sub = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, l2_regularization=1.0, random_state=42)
    sub.fit(Xtr[m_tr], ytr[m_tr])
    pred_seg[m_te] = np.clip(sub.predict(Xte[m_te]), 0, 1)
print("分房源型別模型: MAE {:.3f}  R^2 {:.3f}".format(mean_absolute_error(yte, pred_seg), r2_score(yte, pred_seg)))

print("\n" + "="*60)
print("模型A：風險分數預估（回歸）— 風險分數 = 預測空屋率")
print("="*60)
print("MAE {:.3f} (±{:.1f}pt)  R^2 {:.3f}".format(mean_absolute_error(yte, pg), mean_absolute_error(yte, pg)*100, r2_score(yte, pg)))
print("範例(前5筆) 預測風險分數:", [f"{v*100:.0f}%" for v in pg[:5]], "| 實際:", [f"{v*100:.0f}%" for v in yte[:5]])

print("\n" + "="*60)
print("模型B：是否高風險（分類，空屋率>{:.0%} → 1）".format(HI))
print("="*60)
yb = (y > HI).astype(int)
Xtr2, Xte2, ybtr, ybte = train_test_split(X, yb, test_size=0.2, stratify=yb, random_state=42)
Xtr3, Xval, ybtr3, ybval = train_test_split(Xtr2, ybtr, test_size=0.2, stratify=ybtr, random_state=42)
clf = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, l2_regularization=1.0, random_state=42).fit(Xtr3, ybtr3)
cal = CalibratedClassifierCV(FrozenEstimator(clf), method="isotonic").fit(Xval, ybval)
p = cal.predict_proba(Xte2)[:, 1]
print("高風險占比: {:.1f}%  | AUC {:.3f}".format(ybte.mean()*100, roc_auc_score(ybte, p)))
# 用驗證集挑門檻(Recall>=0.80 最高精確率)
pval = cal.predict_proba(Xval)[:, 1]; best = None
for thr in np.round(np.arange(0.1, 0.9, 0.01), 2):
    pr = (pval >= thr).astype(int); rc = recall_score(ybval, pr); pc = precision_score(ybval, pr, zero_division=0)
    if rc >= 0.80 and (best is None or pc > best[2]): best = (thr, rc, pc)
thr = best[0]; pred = (p >= thr).astype(int)
print("校準門檻(val Recall>=0.80): {:.2f}".format(thr))
print("test → Recall {:.3f}  Precision {:.3f}  F1 {:.3f}".format(
      recall_score(ybte, pred), precision_score(ybte, pred), f1_score(ybte, pred)))
print("混淆矩陣[[TN FP][FN TP]] =", confusion_matrix(ybte, pred).tolist())
