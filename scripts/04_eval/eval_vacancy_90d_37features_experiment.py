# -*- coding: utf-8 -*-
"""
eval_vacancy_90d_37features_experiment.py
================================================================
職責：
  1. 針對 2026 最新房源資料 (6,419 筆) 執行 host_since / host_tenure_days 回補診斷
  2. 匯出完全無法回補的 567 筆真實新房東清單 (data/unmatched_new_hosts_report.csv)
  3. 建立統一 37 個核心特徵的 90 天空屋風險模型 Baseline
  4. 採用 GroupKFold (5-fold, by host_id) 與 365 天模型進行誠實對比
  5. 評估固定機率門檻 (紅警報 >= 0.60, 黃警報 >= 0.35) 的預警表現
  6. 產出評估報告 JSON (models/eval_results_90d_vs_365d.json)
"""

import sys
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, brier_score_loss,
    mean_absolute_error, mean_squared_error, r2_score
)

# 確保 UTF-8 輸出
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("[Step 1] 載入資料與執行 host_since / host_tenure_days 跨版本回補診斷")
print("=" * 70)

# 1.1 載入資料檔
file_2026 = DATA_DIR / "listings_cleaned.csv.gz"
file_2025 = ROOT_DIR / "listings_cleaned.csv.gz"
file_final = ROOT_DIR / "dataset_final.csv"

df_2026 = pd.read_csv(file_2026, compression="gzip", low_memory=False)
df_2025 = pd.read_csv(file_2025, compression="gzip", low_memory=False)
df_final = pd.read_csv(file_final)

print(f"2026 最新房源資料筆數: {len(df_2026)}")
print(f"2025 歷史房源資料筆數: {len(df_2025)}")
print(f"現行 dataset_final.csv 筆數: {len(df_final)}")

# 1.2 建立歷史 host_since / tenure 映射字典 (優先依 id, 次依 host_id)
host_map_2025 = df_2025.dropna(subset=["host_since"]).drop_duplicates(subset=["host_id"]).set_index("host_id")["host_since"].to_dict()
listing_map_2025 = df_2025.dropna(subset=["host_since"]).drop_duplicates(subset=["id"]).set_index("id")["host_since"].to_dict()

# 1.3 執行回補
df_2026["host_since_backfilled"] = df_2026["id"].map(listing_map_2025).fillna(df_2026["host_id"].map(host_map_2025))

# 計算 last_scraped 與 host_since 的日數差 (host_tenure_days)
last_scraped_dt = pd.to_datetime(df_2026["last_scraped"], errors="coerce")
host_since_dt = pd.to_datetime(df_2026["host_since_backfilled"], errors="coerce")
df_2026["host_tenure_days_backfilled"] = (last_scraped_dt - host_since_dt).dt.days

backfilled_mask = df_2026["host_since_backfilled"].notna()
unfilled_mask = df_2026["host_since_backfilled"].isna()

filled_count = backfilled_mask.sum()
unfilled_count = unfilled_mask.sum()

print(f"成功回補 host_since 房源筆數: {filled_count} / {len(df_2026)} ({filled_count/len(df_2026)*100:.2f}%)")
print(f"無法回補 (真實新房東/新上架) 筆數: {unfilled_count} / {len(df_2026)} ({unfilled_count/len(df_2026)*100:.2f}%)")

# 1.4 輸出無匹配新房東報告
unmatched_df = df_2026[unfilled_mask][
    ["id", "host_id", "name", "neighbourhood_cleansed", "room_type", "price", "availability_90"]
].copy()
unmatched_df["unmatched_reason"] = "2026年6/7月新上架房源，歷史資料(2025)中查無此 host_id 之 host_since 紀錄 (真實新房東/冷啟動)"
unmatched_report_path = DATA_DIR / "unmatched_new_hosts_report.csv"
unmatched_df.to_csv(unmatched_report_path, index=False, encoding="utf-8-sig")
print(f"-> 已產出未匹配新房東清單報告: {unmatched_report_path} (共 {len(unmatched_df)} 筆)")

print("\n" + "=" * 70)
print("[Step 2] 構建 37 核心特徵集與目標標籤 (Y_vacancy_90 vs Y_vacancy_365)")
print("=" * 70)

# 2.1 取得 37 個核心特徵定義
FEATURES_37 = [c for c in df_final.columns if c not in ["listing_id", "Y_vacancy"] and (not c.startswith("photo_") or c == "photo_design_sense")]
print(f"核心特徵數量: {len(FEATURES_37)} 個")

# 2.2 實驗 A：重疊標竿集 (N=4,643) - 嚴格控制特徵完全相同，比較時間窗口
df_bench = df_final.merge(
    df_2026[["id", "availability_90", "host_id"]],
    left_on="listing_id", right_on="id", how="inner"
)

# 目標定義
df_bench["Y_vacancy_90"] = (df_bench["availability_90"].astype(float) / 90.0).clip(0, 1)
df_bench["Y_vacancy_365"] = df_bench["Y_vacancy"].astype(float)

df_bench["Y_high_risk_90"] = (df_bench["Y_vacancy_90"] > 0.70).astype(int)
df_bench["Y_high_risk_365"] = (df_bench["Y_vacancy_365"] > 0.70).astype(int)

X_bench = df_bench[FEATURES_37].copy()
groups_bench = df_bench["host_id"].values

print(f"重疊標竿樣本數 (N): {len(df_bench)}")
print(f" 90 天空屋率 (Y_vacancy_90)  均值: {df_bench['Y_vacancy_90'].mean():.4f}, 高風險(>0.70)比例: {df_bench['Y_high_risk_90'].mean()*100:.2f}% ({df_bench['Y_high_risk_90'].sum()}/{len(df_bench)})")
print(f"365 天空屋率 (Y_vacancy_365) 均值: {df_bench['Y_vacancy_365'].mean():.4f}, 高風險(>0.70)比例: {df_bench['Y_high_risk_365'].mean()*100:.2f}% ({df_bench['Y_high_risk_365'].sum()}/{len(df_bench)})")

# 確保無 availability_* 時間洩漏
leakage_cols = [c for c in X_bench.columns if "availability" in c]
assert len(leakage_cols) == 0, f"警告：特徵中含有洩漏欄位 {leakage_cols}"

print("\n" + "=" * 70)
print("[Step 3] 執行 GroupKFold (5-fold, by host_id) 機率校準與模型誠實評估")
print("=" * 70)

gkf = GroupKFold(n_splits=5)

def evaluate_models(X_mat, y_reg, y_clf, groups_arr):
    oof_pred_prob = np.zeros(len(X_mat))
    oof_pred_reg = np.zeros(len(X_mat))
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_mat, y_clf, groups=groups_arr)):
        X_tr, y_tr_clf, y_tr_reg = X_mat.iloc[train_idx], y_clf[train_idx], y_reg[train_idx]
        X_va, y_va_clf, y_va_reg = X_mat.iloc[val_idx], y_clf[val_idx], y_reg[val_idx]
        
        # 分類模型 + 校準
        base_clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=42)
        calibrated_clf = CalibratedClassifierCV(estimator=base_clf, method="isotonic", cv=3)
        calibrated_clf.fit(X_tr, y_tr_clf)
        oof_pred_prob[val_idx] = calibrated_clf.predict_proba(X_va)[:, 1]
        
        # 回歸模型
        reg_model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=42)
        reg_model.fit(X_tr, y_tr_reg)
        oof_pred_reg[val_idx] = reg_model.predict(X_va)
        
    # 分類指標
    auc_score = roc_auc_score(y_clf, oof_pred_prob)
    pr_auc = average_precision_score(y_clf, oof_pred_prob)
    brier = brier_score_loss(y_clf, oof_pred_prob)
    
    # 警報指標 (固定門檻: 紅 >= 0.60, 黃 >= 0.35)
    red_pred = (oof_pred_prob >= 0.60).astype(int)
    yellow_pred = (oof_pred_prob >= 0.35).astype(int)
    
    red_prec = precision_score(y_clf, red_pred, zero_division=0)
    red_cov = red_pred.mean()
    yellow_prec = precision_score(y_clf, yellow_pred, zero_division=0)
    yellow_cov = yellow_pred.mean()
    
    # 回歸指標
    mae = mean_absolute_error(y_reg, oof_pred_reg)
    rmse = np.sqrt(mean_squared_error(y_reg, oof_pred_reg))
    r2 = r2_score(y_reg, oof_pred_reg)
    
    return {
        "classification": {
            "roc_auc": round(float(auc_score), 4),
            "pr_auc": round(float(pr_auc), 4),
            "brier_score": round(float(brier), 4),
            "red_alert_precision": round(float(red_prec), 4),
            "red_alert_coverage": round(float(red_cov), 4),
            "yellow_alert_precision": round(float(yellow_prec), 4),
            "yellow_alert_coverage": round(float(yellow_cov), 4)
        },
        "regression": {
            "mae": round(float(mae), 4),
            "rmse": round(float(rmse), 4),
            "r2": round(float(r2), 4)
        }
    }

# 3.1 評估 90 天模型
results_90d = evaluate_models(
    X_bench,
    df_bench["Y_vacancy_90"].values,
    df_bench["Y_high_risk_90"].values,
    groups_bench
)

# 3.2 評估 365 天模型
results_365d = evaluate_models(
    X_bench,
    df_bench["Y_vacancy_365"].values,
    df_bench["Y_high_risk_365"].values,
    groups_bench
)

print("\n--- 【重疊標竿集 (N=4,643) 模型評估與對比結果】 ---")
print(f"90 天分類模型 (High Risk > 0.70):")
print(f"  ROC-AUC: {results_90d['classification']['roc_auc']} | PR-AUC: {results_90d['classification']['pr_auc']} | Brier Score: {results_90d['classification']['brier_score']}")
print(f"  紅警報(>=0.60): 精準度 {results_90d['classification']['red_alert_precision']*100:.1f}%, 觸發覆蓋率 {results_90d['classification']['red_alert_coverage']*100:.1f}%")
print(f"  黃警報(>=0.35): 精準度 {results_90d['classification']['yellow_alert_precision']*100:.1f}%, 觸發覆蓋率 {results_90d['classification']['yellow_alert_coverage']*100:.1f}%")
print(f"90 天回歸模型 (預測 vacancy_90): MAE = {results_90d['regression']['mae']}, R2 = {results_90d['regression']['r2']}")

print(f"\n365 天分類模型 (High Risk > 0.70):")
print(f"  ROC-AUC: {results_365d['classification']['roc_auc']} | PR-AUC: {results_365d['classification']['pr_auc']} | Brier Score: {results_365d['classification']['brier_score']}")
print(f"  紅警報(>=0.60): 精準度 {results_365d['classification']['red_alert_precision']*100:.1f}%, 觸發覆蓋率 {results_365d['classification']['red_alert_coverage']*100:.1f}%")
print(f"  黃警報(>=0.35): 精準度 {results_365d['classification']['yellow_alert_precision']*100:.1f}%, 觸發覆蓋率 {results_365d['classification']['yellow_alert_coverage']*100:.1f}%")
print(f"365 天回歸模型 (預測 vacancy_365): MAE = {results_365d['regression']['mae']}, R2 = {results_365d['regression']['r2']}")

# 匯出評估報告 JSON
summary_metrics = {
    "dataset_info": {
        "total_2026_listings": len(df_2026),
        "backfilled_host_since_count": int(filled_count),
        "unmatched_new_hosts_count": int(unfilled_count),
        "benchmark_overlapping_count": len(df_bench)
    },
    "model_90d_metrics": results_90d,
    "model_365d_metrics": results_365d
}

eval_json_path = MODEL_DIR / "eval_results_90d_vs_365d.json"
with open(eval_json_path, "w", encoding="utf-8") as f:
    json.dump(summary_metrics, f, ensure_ascii=False, indent=2)

print(f"\n-> 已成功匯出完整評估結果報告 JSON: {eval_json_path}")
print("=" * 70)
print("實驗與對比分析完成！")
print("=" * 70)
