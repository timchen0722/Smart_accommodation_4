# -*- coding: utf-8 -*-
"""
Tier 1 候選特徵批量測試：
  設施旗標(冷氣/電梯/停車/洗衣機/工作區/廚房/長租/寵物)
  房東回覆速度(host_response_time 編碼)
  共用衛浴(is_shared_bath)
  房東信任度(identity_verified / has_profile_pic / verifications數)
  掛牌用心度(description長度 / neighborhood_overview有無 / host_about長度)
  訂房彈性(maximum_nights / minimum_nights_avg_ntm)
一次全部加入，與 baseline(30特徵+self_checkin) 比較，並列出各新特徵重要度排名。
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
L = pd.read_csv("../../listings_cleaned.csv.gz", compression="gzip", low_memory=False)
am = L["amenities"].fillna("").str.lower()

def has(kws):
    return am.apply(lambda s: any(k in s for k in kws)).astype(int)

feat = pd.DataFrame({"listing_id": L["id"]})
feat["self_checkin"] = (am.str.contains("self check",regex=False)|am.str.contains("keypad",regex=False)
                        |am.str.contains("lockbox",regex=False)|am.str.contains("smart lock",regex=False)).astype(int)
feat["has_ac"] = has(["air conditioning","ac unit"])
feat["has_elevator"] = has(["elevator"])
feat["has_parking"] = has(["free parking"])
feat["has_washer"] = has(["washer"])
feat["has_workspace"] = has(["dedicated workspace"])
feat["has_kitchen"] = has(["kitchen"])
feat["long_term_ok"] = has(["long term stays allowed"])
feat["pets_ok"] = has(["pets allowed"])
feat["is_shared_bath"] = L["is_shared_bath"].astype(str).str.lower().eq("true").astype(int)
resp_map = {"within an hour":4, "within a few hours":3, "within a day":2, "a few days or more":1}
feat["response_speed"] = L["host_response_time"].map(resp_map).fillna(0).astype(int)
feat["host_id_verified"] = L["host_identity_verified"].astype(str).str.lower().eq("true").astype(int)
feat["host_has_pic"] = L["host_has_profile_pic"].astype(str).str.lower().eq("true").astype(int)
feat["host_verif_count"] = L["host_verifications"].fillna("[]").astype(str).str.count(",") + 1
feat["desc_len"] = L["description"].fillna("").astype(str).str.len()
feat["has_nbhd_overview"] = L["neighborhood_overview"].notna().astype(int)
feat["host_about_len"] = L["host_about"].fillna("").astype(str).str.len()
feat["maximum_nights"] = pd.to_numeric(L["maximum_nights"], errors="coerce")
feat["min_nights_avg_ntm"] = pd.to_numeric(L["minimum_nights_avg_ntm"], errors="coerce")

df = df.merge(feat, on="listing_id", how="left")
new_cols = [c for c in feat.columns if c != "listing_id"]
for c in new_cols:
    df[c] = df[c].fillna(df[c].median() if df[c].dtype != int else 0)

y = df["Y_vacancy"].values
base = [c for c in df.columns if c not in (["listing_id","Y_vacancy"]+new_cols)]
print("候選特徵占比/統計:")
for c in ["has_ac","has_elevator","has_parking","has_washer","has_workspace","has_kitchen",
          "long_term_ok","pets_ok","is_shared_bath","host_id_verified"]:
    print("  {:18s} {:.1f}%".format(c, df[c].mean()*100))

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
eval_reg(base+["self_checkin"], "baseline(30+self_checkin)")
mA,XteA,yteA = eval_reg(base+new_cols, "+Tier1全部")
print("\n===== 模型B 高風險(分類) =====")
eval_clf(base+["self_checkin"], "baseline(30+self_checkin)")
eval_clf(base+new_cols, "+Tier1全部")

imp=permutation_importance(mA,XteA,yteA,n_repeats=5,random_state=42,scoring="r2")
cols=base+new_cols; rank=list(np.argsort(imp.importances_mean)[::-1])
print("\nTier1 各特徵重要度排名(共{}個特徵，含baseline):".format(len(cols)))
for t in new_cols:
    print("  {:20s} 第 {:2d} 名 (貢獻 {:.4f})".format(t, rank.index(cols.index(t))+1, imp.importances_mean[cols.index(t)]))
