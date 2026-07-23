# -*- coding: utf-8 -*-
"""
把「交通便利性」納入 X，重跑兩個模型並與原本(無交通特徵)比較。
交通特徵：到最近捷運站距離、500m/1km 內捷運站數（台北捷運站座標內建）
輸入：dataset_vacancy.csv（已含 latitude/longitude 與 Y_vacancy）
"""
import numpy as np, pandas as pd
from sklearn.neighbors import BallTree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (mean_absolute_error, r2_score, roc_auc_score,
                             recall_score, precision_score, f1_score)

# ---------- 台北捷運主要站點座標（代表性樣本，涵蓋各線）----------
MRT = [
 (25.0478,121.5170),(25.0333,121.5644),(25.0410,121.5679),(25.0417,121.5440),
 (25.0417,121.5514),(25.0424,121.5330),(25.0333,121.5436),(25.0331,121.5527),
 (25.0339,121.5290),(25.0263,121.5229),(25.0155,121.5342),(25.0421,121.5080),
 (25.0525,121.5203),(25.0578,121.5205),(25.0629,121.5153),(25.0713,121.5200),
 (25.0937,121.5262),(25.0847,121.5245),(25.1319,121.4986),(25.1677,121.4455),
 (25.0143,121.4628),(25.0084,121.4593),(25.0353,121.5000),(25.0300,121.4720),
 (25.0520,121.5442),(25.0521,121.5330),(25.0518,121.5637),(25.0497,121.5776),
 (25.0413,121.5573),(25.0530,121.6069),(25.0553,121.6178),(25.0508,121.5932),
 (25.0447,121.5824),(25.0407,121.5764),(25.0325,121.5185),(25.0353,121.5108),
 (25.0209,121.5285),(24.9927,121.5405),(24.9578,121.5378),(24.9825,121.5416),
 (24.9773,121.5426),(25.0607,121.5262),(25.0592,121.5334),(25.0632,121.5130),
 (25.0558,121.4845),(25.0912,121.4645),(25.0359,121.4520),(25.0430,121.4600),
 (24.9993,121.5115),(25.0155,121.5152),(25.0334,121.5350),(25.0328,121.5705),
 (25.0846,121.5556),(25.0796,121.5470),(25.0838,121.5945),(25.0821,121.5673),
 (25.0625,121.5518),(25.0608,121.5440),(24.9906,121.5090),(24.9937,121.5045),
]
mrt = np.radians(np.array(MRT))
tree = BallTree(mrt, metric="haversine")

df = pd.read_csv("../../dataset_vacancy.csv")
coords = np.radians(df[["latitude","longitude"]].values)
dist, _ = tree.query(coords, k=1)
df["mrt_dist_km"] = dist[:,0]*6371.0
df["mrt_within_500m"] = tree.query_radius(coords, r=0.5/6371.0, count_only=True)
df["mrt_within_1km"]  = tree.query_radius(coords, r=1.0/6371.0, count_only=True)
print("交通特徵敘述：")
print("  到最近捷運站(km)  中位數 {:.2f} | 平均 {:.2f}".format(df["mrt_dist_km"].median(), df["mrt_dist_km"].mean()))
print("  1km內捷運站數     中位數 {:.0f}".format(df["mrt_within_1km"].median()))
# 交通 vs 空屋率 相關
print("  與空屋率相關: mrt_dist_km corr {:+.3f} (正=越遠越空)".format(df["mrt_dist_km"].corr(df["Y_vacancy"])))

transport = ["mrt_dist_km","mrt_within_500m","mrt_within_1km"]
base_feat = [c for c in df.columns if c not in (["listing_id","Y_vacancy"]+transport)]
y = df["Y_vacancy"].values

def eval_reg(cols, tag):
    Xtr,Xte,ytr,yte = train_test_split(df[cols], y, test_size=0.2, random_state=42)
    m = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, l2_regularization=1.0, random_state=42).fit(Xtr,ytr)
    p = np.clip(m.predict(Xte),0,1)
    print("  [{}] MAE {:.3f}  R^2 {:.3f}".format(tag, mean_absolute_error(yte,p), r2_score(yte,p)))
    return m, Xte, yte

def eval_clf(cols, tag, HI=0.7):
    yb=(y>HI).astype(int)
    Xtr,Xte,ytr,yte = train_test_split(df[cols], yb, test_size=0.2, stratify=yb, random_state=42)
    Xtr2,Xv,ytr2,yv = train_test_split(Xtr,ytr,test_size=0.2,stratify=ytr,random_state=42)
    c=HistGradientBoostingClassifier(max_iter=500,learning_rate=0.05,l2_regularization=1.0,random_state=42).fit(Xtr2,ytr2)
    cal=CalibratedClassifierCV(FrozenEstimator(c),method="isotonic").fit(Xv,yv)
    p=cal.predict_proba(Xte)[:,1]; pv=cal.predict_proba(Xv)[:,1]
    best=None
    for t in np.round(np.arange(0.1,0.9,0.01),2):
        pr=(pv>=t).astype(int); rc=recall_score(yv,pr); pc=precision_score(yv,pr,zero_division=0)
        if rc>=0.80 and (best is None or pc>best[2]): best=(t,rc,pc)
    pred=(p>=best[0]).astype(int)
    print("  [{}] AUC {:.3f} | 門檻{:.2f} → Recall {:.3f} Precision {:.3f} F1 {:.3f}".format(
        tag, roc_auc_score(yte,p), best[0], recall_score(yte,pred), precision_score(yte,pred), f1_score(yte,pred)))

print("\n===== 模型A 風險分數(回歸) =====")
eval_reg(base_feat, "無交通特徵")
mA, XteA, yteA = eval_reg(base_feat+transport, "加交通特徵")

print("\n===== 模型B 高風險(分類) =====")
eval_clf(base_feat, "無交通特徵")
eval_clf(base_feat+transport, "加交通特徵")

# 交通特徵在回歸模型的重要度排名
from sklearn.inspection import permutation_importance
imp = permutation_importance(mA, XteA, yteA, n_repeats=5, random_state=42, scoring="r2")
cols = base_feat+transport
rank = np.argsort(imp.importances_mean)[::-1]
print("\n交通特徵在重要度中的排名(共{}個特徵):".format(len(cols)))
for t in transport:
    r = list(rank).index(cols.index(t))+1
    print("  {:18s} 第 {} 名  (貢獻 {:.4f})".format(t, r, imp.importances_mean[cols.index(t)]))
