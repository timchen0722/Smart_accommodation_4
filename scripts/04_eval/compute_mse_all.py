# -*- coding: utf-8 -*-
"""補算報告中所有迴歸版本的真實 MSE（連同 MAE/RMSE/R² 一併輸出以供對照）。
與 final_model_evaluation.py 完全同一資料、同一切分(random_state=42)、同一超參數。
輸出寫 utf-8 檔避免終端機 Big5 亂碼。"""
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("../../dataset_final.csv")
y = df["Y_vacancy"].values
X = df.drop(columns=["listing_id", "Y_vacancy"])
feat = X.columns.tolist()
photo_feat = [c for c in feat if c.startswith("photo_")]
base_feat = [c for c in feat if c not in photo_feat]            # 36
final37 = base_feat + ["photo_design_sense"]                    # 37

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

out = []
def line(s=""): out.append(s)

def metrics(pred, tag, nfeat):
    mae = mean_absolute_error(yte, pred)
    mse = mean_squared_error(yte, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(yte, pred)
    line(f"{tag:36s} {nfeat:>4d}  MAE {mae:.4f}  MSE {mse:.4f}  RMSE {rmse:.4f}  R^2 {r2:.4f}")

line("單次隨機切分 80/20 (random_state=42)  ── 真實 MSE 補算")
line("="*92)
line(f"{'模型版本':36s} {'特徵':>4s}  {'MAE':>9s}  {'MSE':>9s}  {'RMSE':>10s}  {'R^2':>8s}")
line("-"*92)

# 1) Baseline LinearRegression（需標準化，與 final_model_evaluation.py 一致）
sc = StandardScaler().fit(Xtr)
lin = LinearRegression().fit(sc.transform(Xtr), ytr)
p_lin = np.clip(lin.predict(sc.transform(Xte)), 0, 1)
metrics(p_lin, "基準 LinearRegression", len(feat))

# 2) HistGB 無影像 36
gb36 = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, l2_regularization=1.0, random_state=42).fit(Xtr[base_feat], ytr)
metrics(np.clip(gb36.predict(Xte[base_feat]), 0, 1), "HistGB 無影像 (36 特徵)", 36)

# 3) HistGB 最終精簡 37 (base36 + photo_design_sense)
gb37 = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, l2_regularization=1.0, random_state=42).fit(Xtr[final37], ytr)
metrics(np.clip(gb37.predict(Xte[final37]), 0, 1), "HistGB 最終精簡版 (37 特徵)", 37)

# 4) HistGB 全影像 45
gb45 = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, l2_regularization=1.0, random_state=42).fit(Xtr, ytr)
metrics(np.clip(gb45.predict(Xte), 0, 1), "HistGB 全影像版 (45 特徵)", len(feat))

line("="*92)
line("備註：MSE = RMSE^2；MAE 與 MSE 不可互相換算，以上為同模型直接量測。")

txt = "\n".join(out)
with open("../../mse_all_output.txt", "w", encoding="utf-8") as f:
    f.write(txt)
print("done")
