# -*- coding: utf-8 -*-
"""
空屋率回歸訓練：線性回歸(基準) vs HistGradientBoostingRegressor(主力)
風險分數 = 預測空屋率(0~1)。高分=高風險。
輸入：dataset_vacancy.csv
"""
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, recall_score, precision_score

df = pd.read_csv("../../dataset_vacancy.csv")
y = df["Y_vacancy"].values
X = df.drop(columns=["listing_id", "Y_vacancy"])
feat = X.columns.tolist()

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
print("train {} / test {}  | Y mean {:.3f}".format(len(X_tr), len(X_te), y_te.mean()))

def report(name, pred, yt):
    print("\n[{}]".format(name))
    print("  MAE  {:.3f}  (平均誤差 ±{:.1f} 個百分點)".format(mean_absolute_error(yt, pred), mean_absolute_error(yt, pred)*100))
    print("  RMSE {:.3f}".format(np.sqrt(mean_squared_error(yt, pred))))
    print("  R^2  {:.3f}".format(r2_score(yt, pred)))

# ① 基準：線性回歸
sc = StandardScaler().fit(X_tr)
lin = LinearRegression().fit(sc.transform(X_tr), y_tr)
p_lin = np.clip(lin.predict(sc.transform(X_te)), 0, 1)
report("LinearRegression (baseline)", p_lin, y_te)

# ② 主力：HistGradientBoosting 回歸（Loss 可選 squared_error / absolute_error）
gb = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05,
        l2_regularization=1.0, loss="squared_error", random_state=42).fit(X_tr, y_tr)
p_gb = np.clip(gb.predict(X_te), 0, 1)
report("HistGradientBoosting (主力)", p_gb, y_te)

# ③ 把「預測空屋率」當風險分數 → 用門檻分級，看預警效果
print("\n>> 風險分數 = 預測空屋率；以門檻檢視預警品質（真實空屋率>門檻 視為高風險）")
for thr in [0.7, 0.8]:
    yt_hi = (y_te > thr).astype(int)
    pred_hi = (p_gb > thr).astype(int)
    if yt_hi.sum() > 0:
        print("  門檻{:.0%}: 真實高風險 {} 間 | 抓到 Recall {:.2f} | Precision {:.2f}".format(
            thr, int(yt_hi.sum()),
            recall_score(yt_hi, pred_hi, zero_division=0),
            precision_score(yt_hi, pred_hi, zero_division=0)))

# ④ 特徵重要度（對 R^2 貢獻）
imp = permutation_importance(gb, X_te, y_te, n_repeats=5, random_state=42, scoring="r2")
order = np.argsort(imp.importances_mean)[::-1][:12]
print("\n特徵重要度 Top12 (對空屋率預測的貢獻):")
for i in order:
    print("  {:28s} {:.4f}".format(feat[i], imp.importances_mean[i]))
