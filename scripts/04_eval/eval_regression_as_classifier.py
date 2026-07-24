# -*- coding: utf-8 -*-
"""模型A（迴歸）二值化後計算分類指標：把預測空屋率門檻化為「高風險」。
與 final_model_evaluation.py 的模型A完全同一切分、同一模型，只是把輸出切門檻。
輸出寫檔（utf-8）避免終端機 Big5 亂碼。"""
import sys
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (recall_score, precision_score, f1_score,
                             confusion_matrix, roc_auc_score, average_precision_score)

df = pd.read_csv("../../dataset_final.csv")
y = df["Y_vacancy"].values
X = df.drop(columns=["listing_id", "Y_vacancy"])
feat = X.columns.tolist()
base_feat = [c for c in feat if not c.startswith("photo_")]

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

out = []
def line(s=""): out.append(s)

for cols, tag in [(base_feat, "無影像 36特徵"), (feat, "最終版 45特徵含影像")]:
    gb = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05,
                                       l2_regularization=1.0, random_state=42).fit(Xtr[cols], ytr)
    pred_cont = np.clip(gb.predict(Xte[cols]), 0, 1)   # 連續預測空屋率
    line("="*64)
    line(f"模型A（迴歸）當分類器用 — {tag}")
    line("="*64)
    # 迴歸輸出本身可當「分數」直接算排序型指標（不需門檻）
    yb_true = (yte > 0.70).astype(int)
    line(f"[排序型，不需門檻] 以預測空屋率為分數： "
         f"AUC {roc_auc_score(yb_true, pred_cont):.3f}  "
         f"PR-AUC {average_precision_score(yb_true, pred_cont):.3f}")
    line("")
    line("[門檻型] 預測空屋率 > 門檻 視為高風險，與真實(>0.70)比對：")
    line(f"{'門檻':>6} | {'Recall':>7} {'Precision':>10} {'F1':>7} | 混淆矩陣[[TN FP],[FN TP]]")
    for thr in [0.50, 0.60, 0.70, 0.80]:
        pred_b = (pred_cont > thr).astype(int)
        rc = recall_score(yb_true, pred_b, zero_division=0)
        pc = precision_score(yb_true, pred_b, zero_division=0)
        f1 = f1_score(yb_true, pred_b, zero_division=0)
        cm = confusion_matrix(yb_true, pred_b).tolist()
        mark = "  <- 門檻與真實同定義(0.70)" if thr == 0.70 else ""
        line(f"{thr:>6.2f} | {rc:>7.3f} {pc:>10.3f} {f1:>7.3f} | {cm}{mark}")
    line("")

txt = "\n".join(out)
with open("../../regression_as_classifier_output.txt", "w", encoding="utf-8") as f:
    f.write(txt)
print("done")
