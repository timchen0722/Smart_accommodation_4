# -*- coding: utf-8 -*-
"""
多模態 SHAP 分析：載入 multimodal_features_sample.csv 進行模型訓練與 SHAP 解釋，
驗證影像美感與畫質特徵在預估空屋率風險時的貢獻度，並輸出圖檔與複製到 Artifact 區。
"""
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 無 GUI 環境
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. 載入多模態特徵數據
print("載入多模態特徵融合數據集...")
try:
    df = pd.read_csv("../../multimodal_features_sample.csv")
except FileNotFoundError:
    print("錯誤：找不到 multimodal_features_sample.csv。請先執行 build_multimodal_features.py！")
    exit(1)

y = df["Y_vacancy"].values
X = df.drop(columns=["listing_id", "Y_vacancy"])
feat_names = X.columns.tolist()
print("數據載入成功，特徵維度 (包含影像特徵):", X.shape)

# 2. 切分訓練與測試集
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
print("訓練集樣本數:", len(Xtr), " 測試集樣本數:", len(Xte))

# 3. 訓練 HistGradientBoostingRegressor 模型
print("\n訓練 HistGradientBoostingRegressor 模型中...")
model = HistGradientBoostingRegressor(
    max_iter=300, 
    learning_rate=0.05,
    l2_regularization=1.0, 
    random_state=42
).fit(Xtr, ytr)

pred = np.clip(model.predict(Xte), 0, 1)
print("模型效能: MAE {:.3f}  R^2 {:.3f}".format(
    mean_absolute_error(yte, pred), r2_score(yte, pred)
))

# 4. 計算 SHAP 值
print("\n計算多模態 SHAP 值中...")
bg = Xtr.sample(min(50, len(Xtr)), random_state=1)
explainer = shap.Explainer(model.predict, bg)
shap_values = explainer(Xte)
print("SHAP 值計算完成。")

# 5. 輸出全域重要度 Top 15
mean_abs = np.abs(shap_values.values).mean(axis=0)
rank = pd.Series(mean_abs, index=feat_names).sort_values(ascending=False)
print("\n=== SHAP 多模態全域特徵重要度 Top15 ===")
for i, (f, v) in enumerate(rank.head(15).items(), 1):
    print("  {:2d}. {:28s} {:.4f}".format(i, f, v))

# 6. 圖表產生與保存
print("\n產生視覺化圖表中...")

# A. 蜂群摘要圖
plt.figure()
shap.plots.beeswarm(shap_values, max_display=15, show=False)
plt.tight_layout()
plt.savefig("../../shap_multimodal_beeswarm.png", dpi=150)
plt.close()
print("已存檔: shap_multimodal_beeswarm.png")

# B. 條形重要度圖
plt.figure()
shap.plots.bar(shap_values, max_display=15, show=False)
plt.tight_layout()
plt.savefig("../../shap_multimodal_bar.png", dpi=150)
plt.close()
print("已存檔: shap_multimodal_bar.png")

# C. 特徵散佈圖 (挑選前二大特徵)
top2 = rank.head(2).index.tolist()
for f in top2:
    plt.figure()
    shap.plots.scatter(shap_values[:, f], show=False)
    plt.tight_layout()
    fname = f"../../shap_multimodal_scatter_{f}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"已存檔: {fname}")

# D. 單一樣本瀑布圖 (挑選預測空屋率風險最高的樣本)
hi_idx = int(np.argmax(shap_values.values.sum(axis=1)))
plt.figure()
shap.plots.waterfall(shap_values[hi_idx], show=False)
plt.tight_layout()
plt.savefig("../../shap_multimodal_waterfall_sample.png", dpi=150)
plt.close()
print("已存檔: shap_multimodal_waterfall_sample.png (單一高風險房源診斷)")

# 7. 自動將所有產生的多模態圖檔複製到 Artifacts 區
print("\n自動拷貝多模態圖檔至 Artifact 目錄...")
artifact_dir = r"C:\Users\user\.gemini\antigravity-ide\brain\48555586-7d89-4905-8f8a-a85f663a53f4"
pics_to_copy = [
    "../../shap_multimodal_beeswarm.png",
    "../../shap_multimodal_bar.png",
    f"../../shap_multimodal_scatter_{top2[0]}.png",
    f"../../shap_multimodal_scatter_{top2[1]}.png",
    "../../shap_multimodal_waterfall_sample.png"
]

copied_count = 0
for pic in pics_to_copy:
    src_path = pic
    dst_path = f"{artifact_dir}\\{pic}"
    try:
        shutil.copy(src_path, dst_path)
        copied_count += 1
    except Exception as e:
        print("  拷貝 {} 失敗: {}".format(pic, str(e)))

print("圖檔複製完成！共成功複製 {} 張圖。".format(copied_count))
print("多模態 SHAP 完整流程結束。")
