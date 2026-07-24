# -*- coding: utf-8 -*-
"""
EBM (Explainable Boosting Machine) 訓練與非線性特徵曲線分析腳本 (修正版 - 解決對齊 bug)
1. 讀取 dataset_vacancy.csv。
2. 訓練 EBM 分類與回歸模型 (interactions=0)。
3. 評估模型指標 (MAE, R2, AUC, Recall, Precision, F1)。
4. 導出關鍵特徵 (minimum_nights, photo_design_sense, price) 的非線性影響曲線，並保存為圖檔。
5. 將圖檔複製到 Artifacts 目錄下。
"""
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 無 GUI 環境
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # 設定微軟正黑體
plt.rcParams['axes.unicode_minus'] = False  # 解決負號無法正常顯示的問題

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score, recall_score, precision_score, f1_score

from interpret.glassbox import ExplainableBoostingRegressor, ExplainableBoostingClassifier

# ---------- 參數與路徑配置 ----------
DATA_PATH = "../../dataset_vacancy.csv"
ARTIFACT_DIR = r"C:\Users\user\.gemini\antigravity-ide\brain\ce8657a1-8fd5-4180-8386-1af9abece3b7"
RANDOM_STATE = 42
HIGH_RISK_THRESHOLD = 0.70

# ---------- 1. 載入資料 ----------
print("載入資料集中...")
df = pd.read_csv(DATA_PATH)
y_reg = df["Y_vacancy"].astype(float).to_numpy()
X = df.drop(columns=["listing_id", "Y_vacancy"])
feature_names = X.columns.tolist()

print("總房源數: {} | 特徵數: {}".format(len(df), len(feature_names)))

# 用同一個 train_test_split 切分，確保 X_train 與所有 Y_train 對齊
X_train, X_test, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=0.20, random_state=RANDOM_STATE
)

# 分類標籤直接由回歸標籤衍生，保證 index 絕對對齊
y_train_cls = (y_train_reg > HIGH_RISK_THRESHOLD).astype(int)
y_test_cls = (y_test_reg > HIGH_RISK_THRESHOLD).astype(int)

# ---------- 2. 訓練 EBM 回歸模型 (預估空屋率) ----------
print("\n訓練 EBM 回歸模型 (ExplainableBoostingRegressor)...")
ebm_reg = ExplainableBoostingRegressor(interactions=0, random_state=RANDOM_STATE)
ebm_reg.fit(X_train, y_train_reg)

pred_reg = np.clip(ebm_reg.predict(X_test), 0, 1)
mae = mean_absolute_error(y_test_reg, pred_reg)
r2 = r2_score(y_test_reg, pred_reg)
print("EBM 回歸模型效能: MAE = {:.4f} (±{:.2f} 百分點) | R2 = {:.4f}".format(mae, mae * 100, r2))

# ---------- 3. 訓練 EBM 分類模型 (預估高空屋率風險) ----------
print("\n訓練 EBM 分類模型 (ExplainableBoostingClassifier)...")
ebm_cls = ExplainableBoostingClassifier(interactions=0, random_state=RANDOM_STATE)
ebm_cls.fit(X_train, y_train_cls)

prob_cls = ebm_cls.predict_proba(X_test)[:, 1]
prob_train = ebm_cls.predict_proba(X_train)[:, 1]

# 選擇最佳閾值 (使 Train Recall >= 0.80 下精率最高)
best_threshold = 0.50
best_precision = 0.0
for th in np.round(np.arange(0.05, 0.95, 0.01), 2):
    pred_train = (prob_train >= th).astype(int)
    rec = recall_score(y_train_cls, pred_train, zero_division=0)
    prec = precision_score(y_train_cls, pred_train, zero_division=0)
    if rec >= 0.80 and prec > best_precision:
        best_precision = prec
        best_threshold = th

pred_cls = (prob_cls >= best_threshold).astype(int)
auc = roc_auc_score(y_test_cls, prob_cls)
rec = recall_score(y_test_cls, pred_cls)
prec = precision_score(y_test_cls, pred_cls)
f1 = f1_score(y_test_cls, pred_cls)

print("EBM 分類模型效能 (選定預警閾值 = {:.2f}):".format(best_threshold))
print("  AUC       = {:.4f}".format(auc))
print("  Recall    = {:.4f}".format(rec))
print("  Precision = {:.4f}".format(prec))
print("  F1-score  = {:.4f}".format(f1))
    
# ---------- 4. 繪製並導出關鍵特徵的非線性影響曲線 ----------
print("\n正在生成特徵非線性影響曲線...")
key_features = ["minimum_nights", "photo_design_sense", "price"]
reg_global = ebm_reg.explain_global()

generated_plots = []

for feat in key_features:
    if feat not in feature_names:
        print("  警告: 特徵 {} 不在資料集欄位中，跳過。".format(feat))
        continue
    
    idx = feature_names.index(feat)
    data = reg_global.data(idx)
    
    names = data.get('names', [])
    scores = data.get('scores', [])
    
    if len(names) == 0 or len(scores) == 0:
        print("  無法取得特徵 {} 的數據，跳過。".format(feat))
        continue
    
    plt.figure(figsize=(8, 5))
    
    # 判斷特徵類型並處理 x, y 維度不一致的問題
    if data.get('type') == 'continuous' or all(isinstance(n, (int, float)) for n in names):
        # 連續特徵處理：names 是邊界，比 scores 長度多 1
        if len(names) == len(scores) + 1:
            # 計算區間中心點
            x_vals = [0.5 * (names[i] + names[i+1]) for i in range(len(scores))]
        else:
            x_vals = names[:len(scores)]
        
        feat_zh = {
            "minimum_nights": "最少入住晚數 (晚)",
            "photo_design_sense": "封面照設計感評分 (CLIP 美感模型)",
            "price": "每晚價格"
        }
        title_zh = {
            "minimum_nights": "最少入住晚數之 EBM 非線性效應",
            "photo_design_sense": "封面照設計感之 EBM 非線性效應",
            "price": "每晚價格之 EBM 非線性效應"
        }

        plt.plot(x_vals, scores, color='#1f77b4', linewidth=2.5, label='EBM 得分效應')
        plt.xlabel(feat_zh.get(feat, feat), fontsize=12)
    else:
        feat_zh = {
            "minimum_nights": "最少入住晚數 (晚)",
            "photo_design_sense": "封面照設計感評分 (CLIP 美感模型)",
            "price": "每晚價格"
        }
        # 類別特徵
        plt.bar(range(len(names)), scores, color='#2ca02c', alpha=0.8)
        plt.xticks(range(len(names)), names, rotation=45, ha='right')
        plt.xlabel(feat_zh.get(feat, feat), fontsize=12)
        
    plt.ylabel("得分效應 (空屋率變動百分點)", fontsize=12)
    plt.title(title_zh.get(feat, feat) if feat in title_zh else f"EBM {feat} 之非線性效應", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axhline(0, color='red', linestyle=':', alpha=0.7)
    
    # 在圖上印出特徵對應的最優範圍提示 (如果適用)
    if feat == "minimum_nights":
        plt.axvspan(0, 3, color='green', alpha=0.1, label='建議區間 (<=3 晚)')
        plt.legend()
    elif feat == "photo_design_sense":
        plt.axvspan(0.5, 1.0, color='green', alpha=0.1, label='建議區間 (>0.5 設計感)')
        plt.legend()
        
    plt.tight_layout()
    
    filename = "../../ebm_curve_{}.png".format(feat)
    plt.savefig(filename, dpi=150)
    plt.close()
    
    print("  成功繪製並保存: {}".format(filename))
    generated_plots.append(filename)

# ---------- 5. 複製圖檔至 Artifacts 目錄 ----------
print("\n自動複製生成的圖檔至 Artifacts 目錄...")
if not os.path.exists(ARTIFACT_DIR):
    print("  警告: 找不到 Artifacts 目錄 {}，未複製。".format(ARTIFACT_DIR))
else:
    copied_count = 0
    for plot in generated_plots:
        src = plot
        dst = os.path.join(ARTIFACT_DIR, plot)
        try:
            shutil.copy(src, dst)
            copied_count += 1
            print("  已複製 {} -> {}".format(plot, dst)) # 改成普通 ASCII 箭頭避開 CP950 編碼問題
        except Exception as e:
            print("  複製 {} 失敗: {}".format(plot, str(e)))
    print("共成功複製 {} 張特徵效應曲線圖。".format(copied_count))

print("\nEBM 訓練與特徵分析程序執行完畢！")
