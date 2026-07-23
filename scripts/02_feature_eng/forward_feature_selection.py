# -*- coding: utf-8 -*-
"""前向特徵選擇（SFS）：從 base37+POI16=53 超集，貪婪逐一挑增益最大的特徵。

挑選依據：完整模型 GroupKFold(依 host_id, 5 折) R²（與 0.21 誠實基準同基）。
全排序 53 個、畫 R² vs 特徵數邊際曲線，不中途截斷。
對應設計：docs/superpowers/specs/2026-07-15-forward-feature-selection-design.md
"""
import sys, argparse
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
import sys as _sys; _sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..', '01_data_build'))
from load_taipei_poi import load_all_poi

EARTH_KM = 6371.0088
REG = dict(max_iter=500, learning_rate=0.05, l2_regularization=1.0, random_state=42)

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true", help="冒煙測試：只排前 3 步、max_iter 降到 50")
args = ap.parse_args()
if args.smoke:
    REG["max_iter"] = 50

# ---- 載入資料與 host_id（與 add_taipei_poi_features_and_eval.py 完全一致）----
df = pd.read_csv("../../dataset_final.csv")
host = pd.read_csv("../../listings_cleaned.csv.gz", compression="gzip", low_memory=False)[["id", "host_id"]]
df = df.merge(host, left_on="listing_id", right_on="id", how="left").drop(columns=["id"])
poi = load_all_poi()
listing_rad = np.radians(df[["latitude", "longitude"]].values)

SPEC = {
    "mrt":    ([],          True),
    "bus":    ([500],       True),
    "rest":   ([500, 1000], True),
    "cvs":    ([500, 1000], True),
    "park":   ([1000],      True),
    "school": ([1000],      True),
    "pharm":  ([1000],      True),
}
EXTRA_1KM = {"mrt", "park", "school", "pharm"}


def radius_count(tree, r_m):
    return tree.query_radius(listing_rad, r=(r_m / 1000.0) / EARTH_KM, count_only=True)


ALL_NEW = []
for key, arr in poi.items():
    tree = BallTree(np.radians(arr), metric="haversine")
    radii, want_nearest = SPEC[key]
    for r_m in radii:
        col = f"{key}_count_{'500m' if r_m == 500 else '1km'}"
        df[col] = radius_count(tree, r_m); ALL_NEW.append(col)
    if key in EXTRA_1KM and f"{key}_count_1km" not in df.columns:
        col = f"{key}_count_1km"; df[col] = radius_count(tree, 1000); ALL_NEW.append(col)
    if want_nearest:
        dist, _ = tree.query(listing_rad, k=1)
        col = f"{key}_nearest_km"; df[col] = dist[:, 0] * EARTH_KM; ALL_NEW.append(col)

# ---- 候選池：base37 + POI16 = 53 ----
full_base = [c for c in df.columns
             if c not in (["listing_id", "Y_vacancy", "host_id"] + ALL_NEW)
             and (not c.startswith("photo_") or c == "photo_design_sense")]
candidates = full_base + ALL_NEW
y = df["Y_vacancy"].values
groups = df["host_id"].values

assert len(candidates) == 53, f"候選數應為 53，實得 {len(candidates)}"
assert not df[candidates].isna().any().any(), "候選特徵有 NaN"
assert np.isfinite(df[candidates].to_numpy()).all(), "候選特徵有 inf"
print(f"候選特徵數: {len(candidates)}  (base {len(full_base)} + POI {len(ALL_NEW)})")
print(f"資料 {len(df)} 筆, 房東 {df['host_id'].nunique()} 個\n")

# ---- 固定一次 GroupKFold 切分，全程共用 ----
splits = list(GroupKFold(n_splits=5).split(df[candidates], y, groups))


def cv_r2(cols):
    """給定特徵集，回傳 5 折 R² 的 (平均, 標準差)。只跑迴歸。"""
    r2s = []
    for tr, te in splits:
        m = HistGradientBoostingRegressor(**REG).fit(df[cols].iloc[tr], y[tr])
        p = np.clip(m.predict(df[cols].iloc[te]), 0, 1)
        r2s.append(r2_score(y[te], p))
    return float(np.mean(r2s)), float(np.std(r2s))


# ---- 貪婪前向選擇 ----
selected, remaining = [], list(candidates)
records, prev_mean = [], 0.0
n_steps = 3 if args.smoke else len(candidates)

print("===== 前向特徵選擇（每步選 GroupKFold R² 增益最大者）=====")
for step in range(1, n_steps + 1):
    best = None  # (feature, mean, std)
    for c in remaining:
        mean, std = cv_r2(selected + [c])
        if best is None or mean > best[1]:
            best = (c, mean, std)
    feat, mean, std = best
    selected.append(feat); remaining.remove(feat)
    delta = mean - prev_mean; prev_mean = mean
    records.append(dict(step=step, feature=feat, cum_r2=mean, r2_std=std, delta=delta))
    flag = "  <- 增益 < 折間std" if 0 < delta < std else ""
    print(f"  step {step:2d}/{n_steps}: +{feat:22s} R^2 {mean:.4f} ±{std:.4f} (Δ{delta:+.4f}){flag}")
    sys.stdout.flush()

# ---- 輸出 ----
res = pd.DataFrame(records)
out_csv = "../../forward_selection_order_smoke.csv" if args.smoke else "../../forward_selection_order.csv"
res.to_csv(out_csv, index=False, encoding="utf-8-sig")
print(f"\n已寫出 {out_csv}")

if not args.smoke:
    plt.figure(figsize=(11, 6))
    xs = res["step"].values
    plt.plot(xs, res["cum_r2"], "-o", ms=4, color="#2166ac", label="前向選擇累積 GroupKFold R²")
    plt.fill_between(xs, res["cum_r2"] - res["r2_std"], res["cum_r2"] + res["r2_std"],
                     alpha=0.15, color="#2166ac", label="±折間 std")
    plt.axhline(0.209, ls="--", color="#b2182b", lw=1, label="base 37 全上 = 0.209 (參考)")
    best_i = int(res["cum_r2"].idxmax())
    plt.axvline(res["step"].iloc[best_i], ls=":", color="gray", lw=1)
    plt.annotate(f"峰值 {res['cum_r2'].iloc[best_i]:.4f}\n@{res['step'].iloc[best_i]}特徵",
                 (res["step"].iloc[best_i], res["cum_r2"].iloc[best_i]),
                 textcoords="offset points", xytext=(8, -28), fontsize=9)
    plt.xlabel("特徵數"); plt.ylabel("GroupKFold(host) R²")
    plt.title("前向特徵選擇：R² vs 特徵數 邊際曲線")
    plt.legend(loc="lower right", fontsize=9); plt.grid(alpha=0.3); plt.tight_layout()
    try:
        plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass
    plt.savefig("../../forward_selection_curve.png", dpi=130)
    print("已寫出 forward_selection_curve.png")

    peak = res.loc[res["cum_r2"].idxmax()]
    print(f"\n峰值 R² {peak['cum_r2']:.4f} @ 第 {int(peak['step'])} 個特徵")
    exhaust = res[(res["delta"] > 0) & (res["delta"] < res["r2_std"])]
    if len(exhaust):
        print(f"增益首度小於折間 std：第 {int(exhaust.iloc[0]['step'])} 步 "
              f"(+{exhaust.iloc[0]['feature']}, Δ{exhaust.iloc[0]['delta']:+.4f} < std{exhaust.iloc[0]['r2_std']:.4f})")
