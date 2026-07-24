# -*- coding: utf-8 -*-
"""
SHAP 分析：對 dataset_final.csv（36特徵最終版）的 HistGradientBoosting 模型
做真正的 SHAP 值計算，驗證特徵重要度排名，並輸出視覺化圖表。
"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")  # 無GUI環境存檔用
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("../../dataset_final.csv")
y = df["Y_vacancy"].values
X = df.drop(columns=["listing_id", "Y_vacancy"])
feat_names = X.columns.tolist()

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
model = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05,
        l2_regularization=1.0, random_state=42).fit(Xtr, ytr)
pred = np.clip(model.predict(Xte), 0, 1)
print("模型效能（與先前一致，作為對照）: MAE {:.3f}  R^2 {:.3f}".format(
      mean_absolute_error(yte, pred), r2_score(yte, pred)))

# ---------- SHAP：對 HistGradientBoosting 用 Explainer（tree-based自動路徑）----------
print("\n計算 SHAP 值中...")
# 用訓練集抽樣當背景資料，測試集(抽樣至多80筆)算SHAP值，加速運算
bg = Xtr.sample(min(50, len(Xtr)), random_state=1)
X_explain = Xte.sample(min(80, len(Xte)), random_state=1)
explainer = shap.Explainer(model.predict, bg)
shap_values = explainer(X_explain)
print("SHAP 值計算完成，樣本數:", len(X_explain))

# ---------- 全域重要度：依平均|SHAP|排序 ----------
mean_abs = np.abs(shap_values.values).mean(axis=0)
rank = pd.Series(mean_abs, index=feat_names).sort_values(ascending=False)
print("\n=== SHAP 全域特徵重要度 Top15 ===")
for i, (f, v) in enumerate(rank.head(15).items(), 1):
    print("  {:2d}. {:28s} {:.4f}".format(i, f, v))

# ---------- 圖表輸出 ----------
plt.figure()
shap.plots.beeswarm(shap_values, max_display=15, show=False)
plt.tight_layout()
plt.savefig("../../shap_beeswarm.png", dpi=150)
plt.close()
print("\n已存檔: shap_beeswarm.png")

plt.figure()
shap.plots.bar(shap_values, max_display=15, show=False)
plt.tight_layout()
plt.savefig("../../shap_bar.png", dpi=150)
plt.close()
print("已存檔: shap_bar.png")

# 前2大特徵的 scatter（看關係型態：線性/非線性）
top2 = rank.head(2).index.tolist()
for f in top2:
    plt.figure()
    shap.plots.scatter(shap_values[:, f], show=False)
    plt.tight_layout()
    fname = "../../shap_scatter_{}.png".format(f)
    plt.savefig(fname, dpi=150)
    plt.close()
    print("已存檔:", fname)

# 單一樣本 waterfall（示範風險原因解釋，挑 SHAP 貢獻總和最高=風險最高的樣本）
hi_idx = int(np.argmax(shap_values.values.sum(axis=1)))
plt.figure()
shap.plots.waterfall(shap_values[hi_idx], show=False)
plt.tight_layout()
plt.savefig("../../shap_waterfall_sample.png", dpi=150)
plt.close()
print("已存檔: shap_waterfall_sample.png (單一房源風險原因示範)")

# 與先前手動 permutation_importance 排名比較
print("\n=== 與先前 permutation_importance 排名對照(手動篩選前段特徵) ===")
prior_top = ["maximum_nights","min_nights_avg_ntm","host_about_len","response_speed","desc_len","self_checkin"]
for f in prior_top:
    if f in rank.index:
        r = list(rank.index).index(f) + 1
        print("  {:20s} SHAP排名第 {} 名".format(f, r))
