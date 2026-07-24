# -*- coding: utf-8 -*-
"""
跨時間尺度 (30天 vs 60天 vs 90天 vs 365天) 空屋率風險模型全方位對比評估腳本
- 特徵集：精確 37 個核心特徵
- 目標變數：
  Y_30  = availability_30 / 30.0
  Y_60  = availability_60 / 60.0
  Y_90  = availability_90 / 90.0
  Y_365 = availability_365 / 365.0
- 高風險門檻：Y > 0.70 (空置率大於 70%)
- 評估維度：
  1. 基礎統計量 (均值、高風險比例)
  2. 單次切分 (80/20) 迴歸與分類指標
  3. GroupKFold (5折 by host_id) 誠實交叉驗證指標
  4. Permutation Importance 特徵重要度演變
"""
import sys
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             roc_auc_score, average_precision_score, accuracy_score,
                             recall_score, precision_score, f1_score)
from sklearn.inspection import permutation_importance

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---- 1. 載入資料 ----
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
df_final = pd.read_csv(os.path.join(BASE_DIR, "dataset_final.csv"))
listings = pd.read_csv(os.path.join(BASE_DIR, "listings_cleaned.csv.gz"), compression="gzip", low_memory=False)

# 合併多時間尺度欄位與 host_id
av_cols = ["availability_30", "availability_60", "availability_90", "availability_365"]
df = df_final.merge(
    listings[["id", "host_id"] + av_cols],
    left_on="listing_id", right_on="id", how="left"
)

# 定義 37 個核心特徵
FEATURES = [c for c in df_final.columns if c not in ["listing_id", "Y_vacancy"] and (not c.startswith("photo_") or c == "photo_design_sense")]
print(f"使用 37 個核心特徵（確認數量: {len(FEATURES)}）")

HORIZONS = [30, 60, 90, 365]
Y_DICT = {}
HR_DICT = {}

for h in HORIZONS:
    col = f"availability_{h}"
    y_val = (df[col].astype(float) / float(h)).clip(0, 1)
    Y_DICT[h] = y_val.values
    HR_DICT[h] = (y_val.values > 0.70).astype(int)

X = df[FEATURES]
groups = df["host_id"].values

print("\n" + "="*80)
print("一、 各時間尺度基礎統計數據")
print("="*80)
print(f"{'時間區間':12s} {'平均空屋率':12s} {'高風險門檻(>0.70)':20s} {'高風險樣本數':15s}")
for h in HORIZONS:
    y = Y_DICT[h]
    hr = HR_DICT[h]
    print(f"未來 {h:3d} 天    {y.mean()*100:8.2f}%       {hr.mean()*100:15.2f}%       {hr.sum():5d} / {len(hr)}")

# ---- 模型超參數 ----
REG_PARAMS = dict(max_iter=500, learning_rate=0.05, l2_regularization=1.0, random_state=42)
CLF_PARAMS = dict(max_iter=500, learning_rate=0.05, l2_regularization=1.0, random_state=42)

# ---- 2. 單次切分 80/20 評估 ----
print("\n" + "="*80)
print("二、 單次隨機切分 (80/20) 跨時間尺度比較")
print("="*80)

single_results = {}

for h in HORIZONS:
    y = Y_DICT[h]
    hr = HR_DICT[h]
    
    X_tr, X_te, y_tr, y_te, hr_tr, hr_te = train_test_split(X, y, hr, test_size=0.2, random_state=42)
    
    # 回歸
    reg = HistGradientBoostingRegressor(**REG_PARAMS).fit(X_tr, y_tr)
    p_reg = np.clip(reg.predict(X_te), 0, 1)
    mse = mean_squared_error(y_te, p_reg)
    mae = mean_absolute_error(y_te, p_reg)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_te, p_reg)
    
    # 分類
    clf = HistGradientBoostingClassifier(**CLF_PARAMS).fit(X_tr, hr_tr)
    prob = clf.predict_proba(X_te)[:, 1]
    pred = (prob > 0.5).astype(int)
    auc = roc_auc_score(hr_te, prob)
    prauc = average_precision_score(hr_te, prob)
    acc = accuracy_score(hr_te, pred)
    prec = precision_score(hr_te, pred)
    rec = recall_score(hr_te, pred)
    f1 = f1_score(hr_te, pred)
    
    single_results[h] = {
        "mse": mse, "mae": mae, "rmse": rmse, "r2": r2,
        "auc": auc, "prauc": prauc, "acc": acc, "prec": prec, "rec": rec, "f1": f1,
        "reg_model": reg, "X_te": X_te, "y_te": y_te
    }

print("\n【模型 A：迴歸指標】")
print(f"{'指標':15s} {'30天 (Y_30)':14s} {'60天 (Y_60)':14s} {'90天 (Y_90)':14s} {'365天 (Y_365)':14s}")
print(f"{'MSE (均方誤差)':15s} {single_results[30]['mse']:12.4f}   {single_results[60]['mse']:12.4f}   {single_results[90]['mse']:12.4f}   {single_results[365]['mse']:12.4f}")
print(f"{'MAE (平均絕對誤差)':15s}{single_results[30]['mae']:12.4f}   {single_results[60]['mae']:12.4f}   {single_results[90]['mae']:12.4f}   {single_results[365]['mae']:12.4f}")
print(f"{'RMSE (均方根誤差)':15s}{single_results[30]['rmse']:12.4f}   {single_results[60]['rmse']:12.4f}   {single_results[90]['rmse']:12.4f}   {single_results[365]['rmse']:12.4f}")
print(f"{'R^2 (解釋力)':15s} {single_results[30]['r2']:12.4f}   {single_results[60]['r2']:12.4f}   {single_results[90]['r2']:12.4f}   {single_results[365]['r2']:12.4f}")

print("\n【模型 B：分類指標 (Y > 0.70)】")
print(f"{'指標':15s} {'30天 (Y_30)':14s} {'60天 (Y_60)':14s} {'90天 (Y_90)':14s} {'365天 (Y_365)':14s}")
print(f"{'ROC-AUC':15s} {single_results[30]['auc']:12.4f}   {single_results[60]['auc']:12.4f}   {single_results[90]['auc']:12.4f}   {single_results[365]['auc']:12.4f}")
print(f"{'PR-AUC':15s} {single_results[30]['prauc']:12.4f}   {single_results[60]['prauc']:12.4f}   {single_results[90]['prauc']:12.4f}   {single_results[365]['prauc']:12.4f}")
print(f"{'Accuracy':15s} {single_results[30]['acc']:12.4f}   {single_results[60]['acc']:12.4f}   {single_results[90]['acc']:12.4f}   {single_results[365]['acc']:12.4f}")
print(f"{'Precision':15s} {single_results[30]['prec']:12.4f}   {single_results[60]['prec']:12.4f}   {single_results[90]['prec']:12.4f}   {single_results[365]['prec']:12.4f}")
print(f"{'Recall':15s} {single_results[30]['rec']:12.4f}   {single_results[60]['rec']:12.4f}   {single_results[90]['rec']:12.4f}   {single_results[365]['rec']:12.4f}")
print(f"{'F1-score':15s} {single_results[30]['f1']:12.4f}   {single_results[60]['f1']:12.4f}   {single_results[90]['f1']:12.4f}   {single_results[365]['f1']:12.4f}")


# ---- 3. GroupKFold 誠實交叉驗證 ----
print("\n" + "="*80)
print("三、 GroupKFold 誠實交叉驗證 (5折, 依 host_id 分組 — 模擬新房東)")
print("="*80)

gkf = GroupKFold(n_splits=5)
gkf_results = {}

for h in HORIZONS:
    y_vals = Y_DICT[h]
    hr_vals = HR_DICT[h]
    
    r2_list, mae_list, mse_list = [], [], []
    auc_list, prauc_list = [], []
    
    for tr_idx, te_idx in gkf.split(X, y_vals, groups):
        Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
        ytr, yte = y_vals[tr_idx], y_vals[te_idx]
        htr, hte = hr_vals[tr_idx], hr_vals[te_idx]
        
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
            
    gkf_results[h] = {
        "r2": np.mean(r2_list), "mae": np.mean(mae_list), "mse": np.mean(mse_list),
        "auc": np.mean(auc_list), "prauc": np.mean(prauc_list)
    }

print(f"{'指標':18s} {'30天 (Y_30)':14s} {'60天 (Y_60)':14s} {'90天 (Y_90)':14s} {'365天 (Y_365)':14s}")
print(f"{'回歸 R^2 (誠實)':18s} {gkf_results[30]['r2']:12.4f}   {gkf_results[60]['r2']:12.4f}   {gkf_results[90]['r2']:12.4f}   {gkf_results[365]['r2']:12.4f}")
print(f"{'回歸 MSE (誠實)':18s} {gkf_results[30]['mse']:12.4f}   {gkf_results[60]['mse']:12.4f}   {gkf_results[90]['mse']:12.4f}   {gkf_results[365]['mse']:12.4f}")
print(f"{'回歸 MAE (誠實)':18s} {gkf_results[30]['mae']:12.4f}   {gkf_results[60]['mae']:12.4f}   {gkf_results[90]['mae']:12.4f}   {gkf_results[365]['mae']:12.4f}")
print(f"{'分類 ROC-AUC (誠實)':18s}{gkf_results[30]['auc']:12.4f}   {gkf_results[60]['auc']:12.4f}   {gkf_results[90]['auc']:12.4f}   {gkf_results[365]['auc']:12.4f}")
print(f"{'分類 PR-AUC (誠實)':18s} {gkf_results[30]['prauc']:12.4f}   {gkf_results[60]['prauc']:12.4f}   {gkf_results[90]['prauc']:12.4f}   {gkf_results[365]['prauc']:12.4f}")


# ---- 4. Permutation Importance 特徵重要度演變 ----
print("\n" + "="*80)
print("四、 特徵重要度 Top 10 跨時間尺度演變 (Permutation Importance)")
print("="*80)

top_imp_df = pd.DataFrame(index=FEATURES)

for h in HORIZONS:
    res = single_results[h]
    perm = permutation_importance(res["reg_model"], res["X_te"], res["y_te"], n_repeats=5, random_state=42)
    top_imp_df[f"Y_{h}"] = perm.importances_mean

for h in HORIZONS:
    print(f"\n[未來 {h:3d} 天模型 (Y_{h}) Top 10 特徵]")
    s = top_imp_df[f"Y_{h}"].sort_values(ascending=False).head(10)
    for r, (col, val) in enumerate(s.items(), 1):
        print(f"  第 {r:2d} 名: {col:30s} (增益: {val:.4f})")

print("\n全方位跨時間尺度評估結束！")
