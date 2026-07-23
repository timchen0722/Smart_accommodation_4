# -*- coding: utf-8 -*-
"""
驗證：只保留 Tier1 中「真正有效」的 5 個特徵，是否與全加(19個)效果相當。
最終特徵集 = baseline(30) + self_checkin + 5個精選 = 36個
"""
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score, recall_score, precision_score, f1_score

df = pd.read_csv("../../dataset_vacancy.csv")
L = pd.read_csv("../../listings_cleaned.csv.gz", compression="gzip", low_memory=False)
am = L["amenities"].fillna("").str.lower()

feat = pd.DataFrame({"listing_id": L["id"]})
feat["self_checkin"] = (am.str.contains("self check",regex=False)|am.str.contains("keypad",regex=False)
                        |am.str.contains("lockbox",regex=False)|am.str.contains("smart lock",regex=False)).astype(int)
resp_map = {"within an hour":4,"within a few hours":3,"within a day":2,"a few days or more":1}
feat["response_speed"] = L["host_response_time"].map(resp_map).fillna(0).astype(int)
feat["desc_len"] = L["description"].fillna("").astype(str).str.len()
feat["host_about_len"] = L["host_about"].fillna("").astype(str).str.len()
feat["maximum_nights"] = pd.to_numeric(L["maximum_nights"], errors="coerce")
feat["min_nights_avg_ntm"] = pd.to_numeric(L["minimum_nights_avg_ntm"], errors="coerce")

df = df.merge(feat, on="listing_id", how="left")
picked = ["self_checkin","response_speed","desc_len","host_about_len","maximum_nights","min_nights_avg_ntm"]
for c in picked:
    df[c] = df[c].fillna(df[c].median() if df[c].dtype!=int else 0)

y = df["Y_vacancy"].values
base = [c for c in df.columns if c not in (["listing_id","Y_vacancy"]+picked)]
final_cols = base + picked

def reg(cols, tag):
    Xtr,Xte,ytr,yte=train_test_split(df[cols],y,test_size=0.2,random_state=42)
    m=HistGradientBoostingRegressor(max_iter=500,learning_rate=0.05,l2_regularization=1.0,random_state=42).fit(Xtr,ytr)
    p=np.clip(m.predict(Xte),0,1)
    print("  [{}] MAE {:.3f}  R^2 {:.3f}  (n_features={})".format(tag, mean_absolute_error(yte,p), r2_score(yte,p), len(cols)))

def clf(cols, tag, HI=0.7):
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

print("===== 回歸 =====")
reg(base, "baseline(30)")
reg(final_cols, "最終36特徵(30+精選6)")
print("\n===== 分類 =====")
clf(base, "baseline(30)")
clf(final_cols, "最終36特徵(30+精選6)")

# 輸出最終資料集
out = df[["listing_id"]+final_cols+["Y_vacancy"]]
out.to_csv("../../dataset_final.csv", index=False, encoding="utf-8-sig")
print("\n已輸出 dataset_final.csv | {} 筆 x {} 特徵".format(len(out), len(final_cols)))
