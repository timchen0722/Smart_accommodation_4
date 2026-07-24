# -*- coding: utf-8 -*-
"""
把 NLP 情緒特徵納入 X，重跑兩個模型並與原本比較。
NLP 特徵：senti_mean、senti_std、senti_neg_ratio（來自 nlp_features.csv）
也測「NLP + self_checkin」全部一起。
"""
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.inspection import permutation_importance
from sklearn.metrics import (mean_absolute_error, r2_score, roc_auc_score,
                             recall_score, precision_score, f1_score)

df = pd.read_csv("../../dataset_vacancy.csv")
nlp = pd.read_csv("../../nlp_features.csv")
df = df.merge(nlp, on="listing_id", how="left")
nlp_feat = ["senti_mean","senti_std","senti_neg_ratio"]
# 無評論房源 → 用中位數補
for c in nlp_feat:
    df[c] = df[c].fillna(df[c].median())

# self_checkin（沿用前面結論的有效特徵）
L = pd.read_csv("../../listings_cleaned.csv.gz", compression="gzip", low_memory=False)
am = L["amenities"].fillna("").str.lower()
sc = (am.str.contains("self check",regex=False)|am.str.contains("keypad",regex=False)
      |am.str.contains("lockbox",regex=False)|am.str.contains("smart lock",regex=False)).astype(int)
df = df.merge(pd.DataFrame({"listing_id":L["id"],"self_checkin":sc.values}), on="listing_id", how="left")
df["self_checkin"] = df["self_checkin"].fillna(0).astype(int)

y = df["Y_vacancy"].values
base = [c for c in df.columns if c not in (["listing_id","Y_vacancy"]+nlp_feat+["self_checkin"])]
print("NLP 情緒 vs 空屋率相關: mean {:+.3f} | neg_ratio {:+.3f}".format(
      df["senti_mean"].corr(pd.Series(y)), df["senti_neg_ratio"].corr(pd.Series(y))))

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
    print("  [{}] AUC {:.3f} | Recall {:.3f} Precision {:.3f} F1 {:.3f}".format(
        tag, roc_auc_score(yte,p), recall_score(yte,pred), precision_score(yte,pred), f1_score(yte,pred)))

print("\n===== 模型A 風險分數(回歸) =====")
eval_reg(base,"原本30特徵")
eval_reg(base+nlp_feat,"+NLP情緒")
mA,XteA,yteA=eval_reg(base+nlp_feat+["self_checkin"],"+NLP+自助入住(全部)")
print("\n===== 模型B 高風險(分類) =====")
eval_clf(base,"原本30特徵")
eval_clf(base+nlp_feat,"+NLP情緒")
eval_clf(base+nlp_feat+["self_checkin"],"+NLP+自助入住(全部)")

imp=permutation_importance(mA,XteA,yteA,n_repeats=5,random_state=42,scoring="r2")
cols=base+nlp_feat+["self_checkin"]; rank=list(np.argsort(imp.importances_mean)[::-1])
print("\n新特徵重要度排名(共{}個):".format(len(cols)))
for t in nlp_feat+["self_checkin"]:
    print("  {:16s} 第 {} 名 (貢獻 {:.4f})".format(t, rank.index(cols.index(t))+1, imp.importances_mean[cols.index(t)]))
