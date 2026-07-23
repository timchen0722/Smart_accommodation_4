# -*- coding: utf-8 -*-
"""
90 天空屋率風險模型評估與 365 天對照分析腳本
- 特徵集：統一 37 個核心特徵（同 dataset_final.csv 結構）
- 目標變數：Y_vacancy_90 = (availability_90 / 90.0).clip(0, 1)
- 高風險定義：Y_vacancy_90 > 0.70 (未來 90 天空置 > 63 天)
- 評估架構：單次切分 (80/20) + GroupKFold (5 折, by host_id)
- 對照標的：同特徵集下的 365 天空屋率模型
"""
import sys
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             roc_auc_score, average_precision_score, accuracy_score,
                             recall_score, precision_score, f1_score)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---- 1. 載入資料與 37 特徵 ----
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
df_final = pd.read_csv(os.path.join(BASE_DIR, "dataset_final.csv"))
listings = pd.read_csv(os.path.join(BASE_DIR, "listings_cleaned.csv.gz"), compression="gzip", low_memory=False)

# 合併 availability_90 與 host_id
df = df_final.merge(
    listings[["id", "availability_90", "host_id"]],
    left_on="listing_id", right_on="id", how="left"
)

# 定義 37 個核心特徵 (排除 photo_design_sense 以外的影像特徵)
FEATURES = [c for c in df_final.columns if c not in ["listing_id", "Y_vacancy"] and (not c.startswith("photo_") or c == "photo_design_sense")]
print("使用 37 個核心特徵：len =", len(FEATURES))

# 定義 Y_90 與 Y_365
df["Y_vacancy_90"] = (df["availability_90"].astype(float) / 90.0).clip(0, 1)
df["Y_vacancy_365"] = df["Y_vacancy"].astype(float)

X = df[FEATURES]
y90 = df["Y_vacancy_90"].values
y365 = df["Y_vacancy_365"].values
groups = df["host_id"].values

high_risk_90 = (y90 > 0.70).astype(int)
high_risk_365 = (y365 > 0.70).astype(int)

print(f"總樣本數: {len(df)}")
print(f"90天天空屋率均值: {y90.mean():.4f}，高風險比例(>0.70): {high_risk_90.mean()*100:.2f}% ({high_risk_90.sum()}/{len(high_risk_90)})")
print(f"365天空屋率均值: {y365.mean():.4f}，高風險比例(>0.70): {high_risk_365.mean()*100:.2f}% ({high_risk_365.sum()}/{len(high_risk_365)})")

# ---- 2. 模型參數設定 ----
REG_PARAMS = dict(max_iter=500, learning_rate=0.05, l2_regularization=1.0, random_state=42)
CLF_PARAMS = dict(max_iter=500, learning_rate=0.05, l2_regularization=1.0, random_state=42)

# ---- 3. 單次切分 80/20 評估 (y90 vs y365) ----
print("\n" + "="*70)
print("一、 單次隨機切分 (80/20) 對比結果")
print("="*70)

X_tr, X_te, y90_tr, y90_te, y365_tr, y365_te, h90_tr, h90_te, h365_tr, h365_te = train_test_split(
    X, y90, y365, high_risk_90, high_risk_365, test_size=0.2, random_state=42
)

# 90 天回歸
reg90 = HistGradientBoostingRegressor(**REG_PARAMS).fit(X_tr, y90_tr)
p90_reg = np.clip(reg90.predict(X_te), 0, 1)
mae90 = mean_absolute_error(y90_te, p90_reg)
mse90 = mean_squared_error(y90_te, p90_reg)
rmse90 = np.sqrt(mse90)
r2_90 = r2_score(y90_te, p90_reg)

# 365 天回歸
reg365 = HistGradientBoostingRegressor(**REG_PARAMS).fit(X_tr, y365_tr)
p365_reg = np.clip(reg365.predict(X_te), 0, 1)
mae365 = mean_absolute_error(y365_te, p365_reg)
mse365 = mean_squared_error(y365_te, p365_reg)
rmse365 = np.sqrt(mse365)
r2_365 = r2_score(y365_te, p365_reg)

print("\n【模型 A：空屋率分數迴歸預測】")
print(f"指標             90天模型 (Y_90)      365天模型 (Y_365)")
print(f"MSE (均方誤差)   {mse90:.4f}             {mse365:.4f}")
print(f"MAE (平均絕對誤差){mae90:.4f}             {mae365:.4f}")
print(f"RMSE (均方根誤差){rmse90:.4f}             {rmse365:.4f}")
print(f"R^2 (解釋力)      {r2_90:.4f}             {r2_365:.4f}")

# 90 天分類
clf90 = HistGradientBoostingClassifier(**CLF_PARAMS).fit(X_tr, h90_tr)
prob90 = clf90.predict_proba(X_te)[:, 1]
pred90 = (prob90 > 0.5).astype(int)
auc90 = roc_auc_score(h90_te, prob90)
prauc90 = average_precision_score(h90_te, prob90)
acc90 = accuracy_score(h90_te, pred90)
rec90 = recall_score(h90_te, pred90)
prec90 = precision_score(h90_te, pred90)
f1_90 = f1_score(h90_te, pred90)

# 365 天分類
clf365 = HistGradientBoostingClassifier(**CLF_PARAMS).fit(X_tr, h365_tr)
prob365 = clf365.predict_proba(X_te)[:, 1]
pred365 = (prob365 > 0.5).astype(int)
auc365 = roc_auc_score(h365_te, prob365)
prauc365 = average_precision_score(h365_te, prob365)
acc365 = accuracy_score(h365_te, pred365)
rec365 = recall_score(h365_te, pred365)
prec365 = precision_score(h365_te, pred365)
f1_365 = f1_score(h365_te, pred365)

print("\n【模型 B：高風險二元分類 (Y > 0.70)】")
print(f"指標             90天模型 (Y_90)      365天模型 (Y_365)")
print(f"ROC-AUC          {auc90:.4f}             {auc365:.4f}")
print(f"PR-AUC           {prauc90:.4f}             {prauc365:.4f}")
print(f"Accuracy         {acc90:.4f}             {acc365:.4f}")
print(f"Precision        {prec90:.4f}             {prec365:.4f}")
print(f"Recall           {rec90:.4f}             {rec365:.4f}")
print(f"F1-score         {f1_90:.4f}             {f1_365:.4f}")


# ---- 4. GroupKFold 誠實交叉驗證 (5折, by host_id) ----
print("\n" + "="*70)
print("二、 GroupKFold 誠實交叉驗證 (5折, 依 host_id 分組)")
print("="*70)

gkf = GroupKFold(n_splits=5)

def eval_gkf(y_vals, high_risk_vals):
    r2_list, mae_list, mse_list = [], [], []
    auc_list, prauc_list = [], []
    
    for tr_idx, te_idx in gkf.split(X, y_vals, groups):
        Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
        ytr, yte = y_vals[tr_idx], y_vals[te_idx]
        htr, hte = high_risk_vals[tr_idx], high_risk_vals[te_idx]
        
        # 回歸
        rm = HistGradientBoostingRegressor(**REG_PARAMS).fit(Xtr, ytr)
        rp = np.clip(rm.predict(Xte), 0, 1)
        r2_list.append(r2_score(yte, rp))
        mae_list.append(mean_absolute_error(yte, rp))
        mse_list.append(mean_squared_error(yte, rp))
        
        # 分類
        if htr.sum() >= 5 and hte.sum() >= 5:
            cm = HistGradientBoostingClassifier(**CLF_PARAMS).fit(Xtr, htr)
            cp = cm.predict_proba(Xte)[:, 1]
            auc_list.append(roc_auc_score(hte, cp))
            prauc_list.append(average_precision_score(hte, cp))
            
    return np.mean(r2_list), np.mean(mae_list), np.mean(mse_list), np.mean(auc_list), np.mean(prauc_list)

gkf_r2_90, gkf_mae_90, gkf_mse_90, gkf_auc_90, gkf_prauc_90 = eval_gkf(y90, high_risk_90)
gkf_r2_365, gkf_mae_365, gkf_mse_365, gkf_auc_365, gkf_prauc_365 = eval_gkf(y365, high_risk_365)

print("\n【GroupKFold 誠實指標對比 (新房東情境)】")
print(f"指標                 90天模型 (Y_90)      365天模型 (Y_365)")
print(f"回歸 R^2             {gkf_r2_90:.4f}             {gkf_r2_365:.4f}")
print(f"回歸 MSE             {gkf_mse_90:.4f}             {gkf_mse_365:.4f}")
print(f"回歸 MAE             {gkf_mae_90:.4f}             {gkf_mae_365:.4f}")
print(f"分類 ROC-AUC         {gkf_auc_90:.4f}             {gkf_auc_365:.4f}")
print(f"分類 PR-AUC          {gkf_prauc_90:.4f}             {gkf_prauc_365:.4f}")

# ---- 5. Permutation Importance 特徵重要度 Top 10 對比 ----
print("\n" + "="*70)
print("三、 90 天模型 vs 365 天模型 特徵重要度 (Permutation Importance Top 10)")
print("="*70)

perm90 = permutation_importance(reg90, X_te, y90_te, n_repeats=5, random_state=42)
perm365 = permutation_importance(reg365, X_te, y365_te, n_repeats=5, random_state=42)

imp90 = pd.Series(perm90.importances_mean, index=FEATURES).sort_values(ascending=False)
imp365 = pd.Series(perm365.importances_mean, index=FEATURES).sort_values(ascending=False)

print("\n[90天模型 Top 10 關鍵特徵]")
for r, (col, val) in enumerate(imp90.head(10).items(), 1):
    print(f"  第 {r:2d} 名: {col:30s} (權重增益: {val:.4f})")

print("\n[365天模型 Top 10 關鍵特徵]")
for r, (col, val) in enumerate(imp365.head(10).items(), 1):
    print(f"  第 {r:2d} 名: {col:30s} (權重增益: {val:.4f})")

print("\n評估運算完畢！")
