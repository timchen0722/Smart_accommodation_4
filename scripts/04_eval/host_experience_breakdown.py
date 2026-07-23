# -*- coding: utf-8 -*-
"""
用真實欄位(host_tenure_days / host_listings_count)定義新舊房東，
分開驗證兩件事：
(a) 一般隨機切分下，新舊房東表現是否有差
(b) GroupKFold(模型完全沒看過此房東)下，新舊房東表現是否有差
把「模型看沒看過這個房東」跟「這個房東是不是真的資淺」分開驗證。
"""
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score

df = pd.read_csv("../../dataset_final.csv")
L = pd.read_csv("../../listings_cleaned.csv.gz", compression="gzip", low_memory=False)[["id", "host_id"]]
df = df.merge(L, left_on="listing_id", right_on="id", how="left")

FINAL = [c for c in df.columns if c not in ["listing_id", "Y_vacancy", "id", "host_id"]
         and (not c.startswith("photo_") or c == "photo_design_sense")]
y = df["Y_vacancy"].values
yb = (y > 0.7).astype(int)
groups = df["host_id"].values

# 真實新舊房東定義
TENURE_Q25 = df["host_tenure_days"].quantile(0.25)
df["is_newcomer_by_tenure"] = df["host_tenure_days"] <= TENURE_Q25          # 相對資淺(後25%)
df["is_newcomer_by_count"] = df["host_listings_count"] == 1                 # 只有1筆房源=個人房東
print("資淺(經營天數後25%)門檻: {:.0f}天 | 人數占比 {:.1f}%".format(TENURE_Q25, df["is_newcomer_by_tenure"].mean()*100))
print("個人房東(僅1筆房源)人數占比: {:.1f}%\n".format(df["is_newcomer_by_count"].mean()*100))

def eval_subgroup(y_true, pred_reg, ybtrue, pred_clf, mask, label):
    if mask.sum() < 20:
        print("    {:14s} n={:4d}  (樣本太少，略過)".format(label, int(mask.sum())))
        return
    r2 = r2_score(y_true[mask], pred_reg[mask]); mae = mean_absolute_error(y_true[mask], pred_reg[mask])
    ybm = ybtrue[mask]
    auc = roc_auc_score(ybm, pred_clf[mask]) if 5 <= ybm.sum() < len(ybm) else float("nan")
    print("    {:14s} n={:4d}  回歸 R^2={:.3f} MAE={:.3f}  |  分類 AUC={:.3f}  高風險占比={:.1f}%".format(
        label, int(mask.sum()), r2, mae, auc, ybm.mean()*100))

# ============================================================
# (a) 一般隨機切分：新舊房東表現是否有差
# ============================================================
print("="*60); print("(a) 一般隨機切分（80/20，不分組）— 依真實新舊房東拆開看"); print("="*60)
idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)
Xtr, Xte = df[FINAL].iloc[idx_tr], df[FINAL].iloc[idx_te]
ytr, yte = y[idx_tr], y[idx_te]
ybtr, ybte = yb[idx_tr], yb[idx_te]
meta_te = df.iloc[idx_te].reset_index(drop=True)

reg = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, l2_regularization=1.0, random_state=42).fit(Xtr, ytr)
clf = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, l2_regularization=1.0, random_state=42).fit(Xtr, ybtr)
p_reg = np.clip(reg.predict(Xte), 0, 1); p_clf = clf.predict_proba(Xte)[:, 1]

print("\n[依經營天數]")
eval_subgroup(yte, p_reg, ybte, p_clf, meta_te["is_newcomer_by_tenure"].values, "資淺(後25%)")
eval_subgroup(yte, p_reg, ybte, p_clf, ~meta_te["is_newcomer_by_tenure"].values, "資深(前75%)")
print("[依房源數]")
eval_subgroup(yte, p_reg, ybte, p_clf, meta_te["is_newcomer_by_count"].values, "個人房東(1筆)")
eval_subgroup(yte, p_reg, ybte, p_clf, ~meta_te["is_newcomer_by_count"].values, "多房源房東")

# ============================================================
# (b) GroupKFold：模型完全沒看過此房東時，新舊房東表現是否有差
# ============================================================
print("\n" + "="*60); print("(b) GroupKFold（模型沒看過此房東）— 依真實新舊房東拆開看"); print("="*60)
gkf = GroupKFold(n_splits=5)
all_te_idx, all_pred_reg, all_pred_clf = [], [], []
for tr, te in gkf.split(df[FINAL], y, groups):
    Xtr2, Xte2 = df[FINAL].iloc[tr], df[FINAL].iloc[te]
    m = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, l2_regularization=1.0, random_state=42).fit(Xtr2, y[tr])
    c = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, l2_regularization=1.0, random_state=42).fit(Xtr2, yb[tr])
    all_te_idx.append(te)
    all_pred_reg.append(np.clip(m.predict(Xte2), 0, 1))
    all_pred_clf.append(c.predict_proba(Xte2)[:, 1])

te_idx = np.concatenate(all_te_idx)
pred_reg_all = np.concatenate(all_pred_reg)
pred_clf_all = np.concatenate(all_pred_clf)
y_all = y[te_idx]; yb_all = yb[te_idx]
meta_all = df.iloc[te_idx].reset_index(drop=True)

print("\n[依經營天數]")
eval_subgroup(y_all, pred_reg_all, yb_all, pred_clf_all, meta_all["is_newcomer_by_tenure"].values, "資淺(後25%)")
eval_subgroup(y_all, pred_reg_all, yb_all, pred_clf_all, ~meta_all["is_newcomer_by_tenure"].values, "資深(前75%)")
print("[依房源數]")
eval_subgroup(y_all, pred_reg_all, yb_all, pred_clf_all, meta_all["is_newcomer_by_count"].values, "個人房東(1筆)")
eval_subgroup(y_all, pred_reg_all, yb_all, pred_clf_all, ~meta_all["is_newcomer_by_count"].values, "多房源房東")
