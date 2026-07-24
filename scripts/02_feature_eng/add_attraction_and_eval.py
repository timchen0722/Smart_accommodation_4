# -*- coding: utf-8 -*-
"""
把「到景點/商圈距離」納入 X，重跑兩個模型並與原本(無此特徵)比較。
特徵：到最近景點/商圈距離、1km內景點數、到信義商圈距離、到最近夜市距離
輸入：dataset_vacancy.csv
"""
import numpy as np, pandas as pd
from sklearn.neighbors import BallTree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.inspection import permutation_importance
from sklearn.metrics import (mean_absolute_error, r2_score, roc_auc_score,
                             recall_score, precision_score, f1_score)

# 台北主要景點/商圈 (name, lat, lon)
POI = {
 "信義商圈/101":(25.0339,121.5645), "西門町":(25.0424,121.5074), "士林夜市":(25.0880,121.5240),
 "台北車站商圈":(25.0478,121.5170), "東區忠孝敦化":(25.0417,121.5514), "中山商圈":(25.0525,121.5203),
 "龍山寺艋舺":(25.0369,121.4999), "公館商圈":(25.0155,121.5342), "饒河夜市":(25.0510,121.5773),
 "大稻埕迪化街":(25.0560,121.5100), "故宮":(25.1024,121.5485), "北投溫泉":(25.1364,121.5065),
 "淡水老街":(25.1699,121.4389), "國父紀念館松菸":(25.0413,121.5573), "華山1914":(25.0442,121.5296),
 "南港展覽館":(25.0553,121.6178), "天母商圈":(25.1178,121.5320), "永康街":(25.0330,121.5300),
 "象山":(25.0270,121.5710), "圓山花博":(25.0713,121.5200),
}
NIGHTMARKETS = {"士林夜市":(25.0880,121.5240),"饒河夜市":(25.0510,121.5773),"公館":(25.0155,121.5342),
 "寧夏夜市":(25.0563,121.5155),"通化臨江":(25.0308,121.5533),"師大夜市":(25.0244,121.5290)}

def near(df, pts):
    tree = BallTree(np.radians(np.array(list(pts))), metric="haversine")
    c = np.radians(df[["latitude","longitude"]].values)
    d,_ = tree.query(c, k=1)
    return d[:,0]*6371.0, tree, c

df = pd.read_csv("../../dataset_vacancy.csv")
d_poi, tpoi, c = near(df, POI.values())
df["poi_dist_km"] = d_poi
df["poi_within_1km"] = tpoi.query_radius(c, r=1.0/6371.0, count_only=True)
sinyi = np.radians(np.array([[25.0339,121.5645]]))
df["dist_xinyi_km"] = BallTree(sinyi, metric="haversine").query(c,k=1)[0][:,0]*6371.0
d_nm = near(df, NIGHTMARKETS.values())[0]; df["nightmkt_dist_km"] = d_nm

print("景點/商圈特徵敘述：")
print("  到最近景點(km)   中位 {:.2f} | 平均 {:.2f} | 最大 {:.2f}".format(df["poi_dist_km"].median(), df["poi_dist_km"].mean(), df["poi_dist_km"].max()))
print("  到信義商圈(km)   中位 {:.2f} | 平均 {:.2f}".format(df["dist_xinyi_km"].median(), df["dist_xinyi_km"].mean()))
print("  與空屋率相關: poi_dist {:+.3f} | 信義距離 {:+.3f} | 夜市距離 {:+.3f}".format(
      df["poi_dist_km"].corr(df["Y_vacancy"]), df["dist_xinyi_km"].corr(df["Y_vacancy"]), df["nightmkt_dist_km"].corr(df["Y_vacancy"])))

poi_feat = ["poi_dist_km","poi_within_1km","dist_xinyi_km","nightmkt_dist_km"]
base = [c for c in df.columns if c not in (["listing_id","Y_vacancy"]+poi_feat)]
y = df["Y_vacancy"].values

def eval_reg(cols, tag):
    Xtr,Xte,ytr,yte = train_test_split(df[cols], y, test_size=0.2, random_state=42)
    m=HistGradientBoostingRegressor(max_iter=500,learning_rate=0.05,l2_regularization=1.0,random_state=42).fit(Xtr,ytr)
    p=np.clip(m.predict(Xte),0,1)
    print("  [{}] MAE {:.3f}  R^2 {:.3f}".format(tag, mean_absolute_error(yte,p), r2_score(yte,p)))
    return m,Xte,yte

def eval_clf(cols, tag, HI=0.7):
    yb=(y>HI).astype(int)
    Xtr,Xte,ytr,yte=train_test_split(df[cols],yb,test_size=0.2,stratify=yb,random_state=42)
    Xtr2,Xv,ytr2,yv=train_test_split(Xtr,ytr,test_size=0.2,stratify=ytr,random_state=42)
    cl=HistGradientBoostingClassifier(max_iter=500,learning_rate=0.05,l2_regularization=1.0,random_state=42).fit(Xtr2,ytr2)
    cal=CalibratedClassifierCV(FrozenEstimator(cl),method="isotonic").fit(Xv,yv)
    p=cal.predict_proba(Xte)[:,1]; pv=cal.predict_proba(Xv)[:,1]; best=None
    for t in np.round(np.arange(0.1,0.9,0.01),2):
        pr=(pv>=t).astype(int); rc=recall_score(yv,pr); pc=precision_score(yv,pr,zero_division=0)
        if rc>=0.80 and (best is None or pc>best[2]): best=(t,rc,pc)
    pred=(p>=best[0]).astype(int)
    print("  [{}] AUC {:.3f} | 門檻{:.2f} → Recall {:.3f} Precision {:.3f} F1 {:.3f}".format(
        tag, roc_auc_score(yte,p), best[0], recall_score(yte,pred), precision_score(yte,pred), f1_score(yte,pred)))

print("\n===== 模型A 風險分數(回歸) =====")
eval_reg(base,"無景點特徵")
mA,XteA,yteA=eval_reg(base+poi_feat,"加景點特徵")
print("\n===== 模型B 高風險(分類) =====")
eval_clf(base,"無景點特徵")
eval_clf(base+poi_feat,"加景點特徵")

imp=permutation_importance(mA,XteA,yteA,n_repeats=5,random_state=42,scoring="r2")
cols=base+poi_feat; rank=list(np.argsort(imp.importances_mean)[::-1])
print("\n景點特徵在重要度排名(共{}個):".format(len(cols)))
for t in poi_feat:
    print("  {:18s} 第 {} 名 (貢獻 {:.4f})".format(t, rank.index(cols.index(t))+1, imp.importances_mean[cols.index(t)]))
