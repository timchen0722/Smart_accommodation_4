# -*- coding: utf-8 -*-
"""
把「自助入住 self check-in」納入 X，重跑兩個模型並與原本比較。
特徵：self_checkin(自助入住/密碼鎖/智慧鎖) 、host_greets(房東親迎)
來源：listings.csv.gz 的 amenities，依 listing_id 併回 dataset_vacancy.csv
"""
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.inspection import permutation_importance
from sklearn.metrics import (mean_absolute_error, r2_score, roc_auc_score,
                             recall_score, precision_score, f1_score)

# 從 listings.csv.gz 萃取自助入住旗標
L = pd.read_csv("../../listings.csv.gz", compression="gzip", low_memory=False)
am = L["amenities"].fillna("").str.lower()
selfci = (am.str.contains("self check", regex=False) | am.str.contains("keypad", regex=False)
          | am.str.contains("lockbox", regex=False) | am.str.contains("smart lock", regex=False)
          | am.str.contains("keyless", regex=False)).astype(int)
greets = am.str.contains("host greets", regex=False).astype(int)
flags = pd.DataFrame({"listing_id": L["id"], "self_checkin": selfci.values, "host_greets": greets.values})

df = pd.read_csv("../../dataset_vacancy.csv").merge(flags, on="listing_id", how="left")
df[["self_checkin","host_greets"]] = df[["self_checkin","host_greets"]].fillna(0).astype(int)

newf = ["self_checkin","host_greets"]
base = [c for c in df.columns if c not in (["listing_id","Y_vacancy"]+newf)]
y = df["Y_vacancy"].values
print("自助入住占比 {:.1f}% | 房東親迎占比 {:.1f}%".format(df["self_checkin"].mean()*100, df["host_greets"].mean()*100))

def eval_reg(cols, tag):
    Xtr,Xte,ytr,yte=train_test_split(df[cols],y,test_size=0.2,random_state=42)
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
eval_reg(base,"無自助入住")
mA,XteA,yteA=eval_reg(base+newf,"加自助入住")
print("\n===== 模型B 高風險(分類) =====")
eval_clf(base,"無自助入住")
eval_clf(base+newf,"加自助入住")

imp=permutation_importance(mA,XteA,yteA,n_repeats=5,random_state=42,scoring="r2")
cols=base+newf; rank=list(np.argsort(imp.importances_mean)[::-1])
print("\n新特徵重要度排名(共{}個):".format(len(cols)))
for t in newf:
    print("  {:14s} 第 {} 名 (貢獻 {:.4f})".format(t, rank.index(cols.index(t))+1, imp.importances_mean[cols.index(t)]))
