# -*- coding: utf-8 -*-
"""
把 feature_comparison_table.csv 的 6 種狀態，合併重整為 3 大區塊：
  直接沿用 = 原「直接採用」
  衍生轉換 = 原「衍生採用」
  未採用   = 原4種「未採用-*」全部合併(細節原因保留在說明欄)
輸出時依區塊分組排列，並在每個區塊前插入標題列，方便閱讀。
"""
import pandas as pd

df = pd.read_csv("../../feature_comparison_table.csv")
n0 = len(df)

BLOCK_MAP = {
    "直接採用": "① 直接沿用",
    "衍生採用": "② 衍生轉換",
    "未採用-防洩漏": "③ 未採用",
    "未採用-已測試無效": "③ 未採用",
    "未採用-全空/無意義": "③ 未採用",
    "未採用-未測試": "③ 未採用",
}
df["區塊"] = df["採用狀態"].map(BLOCK_MAP)
assert df["區塊"].isna().sum() == 0, "有狀態未能對應到三大區塊"
assert len(df) == n0 == 81, "列數應維持81列"

order = {"① 直接沿用": 0, "② 衍生轉換": 1, "③ 未採用": 2}
df["_sort"] = df["區塊"].map(order)
df = df.sort_values(["_sort", "序號"]).drop(columns="_sort")
df = df[["區塊", "序號", "listings_cleaned.csv.gz 原始欄位", "採用狀態", "對應最終特徵", "說明"]]

df.to_csv("../../feature_comparison_table.csv", index=False, encoding="utf-8-sig")

print("驗證：三大區塊列數統計 (加總應=81)")
counts = df["區塊"].value_counts().sort_index()
print(counts.to_string())
print("加總:", counts.sum())
print("\n已更新: feature_comparison_table.csv（依 區塊 分組排列）")
