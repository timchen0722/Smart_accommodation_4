# -*- coding: utf-8 -*-
"""從已完成的 SFS stdout（forward_selection_output.txt）解析 35 步，
產出 forward_selection_order.csv 與 forward_selection_curve.png。
（正式跑因 max_iter=500 過慢、且 R² 已在 0.26 平台，經使用者同意於 step 35 提前收尾。）"""
import re, sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAT = re.compile(
    r"step\s+(\d+)/\d+:\s+\+(\S+)\s+R\^2\s+([\d.]+)\s+±([\d.]+)\s+\(Δ([+-][\d.]+)\)")
rows = []
with open("../../forward_selection_output.txt", encoding="utf-8") as f:
    for line in f:
        m = PAT.search(line)
        if m:
            rows.append(dict(step=int(m.group(1)), feature=m.group(2),
                             cum_r2=float(m.group(3)), r2_std=float(m.group(4)),
                             delta=float(m.group(5))))
res = pd.DataFrame(rows).sort_values("step").reset_index(drop=True)
res.to_csv("../../forward_selection_order.csv", index=False, encoding="utf-8-sig")
print(f"解析 {len(res)} 步 → forward_selection_order.csv")

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.figure(figsize=(12, 6))
xs = res["step"].values
plt.plot(xs, res["cum_r2"], "-o", ms=4, color="#2166ac", label="前向選擇累積 GroupKFold R²")
plt.fill_between(xs, res["cum_r2"] - res["r2_std"], res["cum_r2"] + res["r2_std"],
                 alpha=0.15, color="#2166ac", label="±折間 std")
plt.axhline(0.209, ls="--", color="#b2182b", lw=1, label="base 37 全上 = 0.209 (參考)")
peak_i = int(res["cum_r2"].idxmax())
px, py = res["step"].iloc[peak_i], res["cum_r2"].iloc[peak_i]
plt.axvline(px, ls=":", color="gray", lw=1)
plt.annotate(f"峰值 {py:.4f}\n@{px}特徵 (+{res['feature'].iloc[peak_i]})",
             (px, py), textcoords="offset points", xytext=(8, -34), fontsize=9)
# 標記邊際增益耗盡點：delta 首度連續小於 std 且此後不再顯著上升
plt.xlabel("特徵數（前向選擇加入順序）"); plt.ylabel("GroupKFold(host) R²")
plt.title("前向特徵選擇：R² vs 特徵數 邊際曲線（step 35 提前收尾）")
plt.legend(loc="lower right", fontsize=9); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("../../forward_selection_curve.png", dpi=130)
print("→ forward_selection_curve.png")

# 摘要
print(f"\n峰值 R² {py:.4f} @ 第 {px} 個特徵")
first_cross = res[(res["cum_r2"] >= 0.209)].head(1)
if len(first_cross):
    print(f"追平 base(0.209) 於第 {int(first_cross['step'].iloc[0])} 個特徵 "
          f"(+{first_cross['feature'].iloc[0]}, R² {first_cross['cum_r2'].iloc[0]:.4f})")
# 邊際耗盡：從哪一步起後續再無淨提升超過噪音
print("\n前 15 步累積 R²：")
for _, r in res.head(15).iterrows():
    print(f"  {int(r['step']):2d}. +{r['feature']:24s} {r['cum_r2']:.4f} (Δ{r['delta']:+.4f})")
