# -*- coding: utf-8 -*-
"""
補上目前表格中被「附註」在其他列、但沒有獨立列的特徵：
1. 4 個競爭特徵(多欄位組合衍生，非單一原始欄位1:1轉換)：
   score_pctl_nbhd, amenities_vs_median, nbr_density_1km, nbr_density_same_type_1km
   (price_pctl_nbhd 已有獨立標註，這裡一併補成獨立列)
2. 8 個已測試但排除的影像特徵(同樣源自 picture_url，只是CLIP/OpenCV算出來沒被採用)：
   photo_width, photo_height, photo_brightness, photo_contrast, photo_sharpness,
   photo_colorfulness, photo_coziness, photo_cleanliness
   (photo_design_sense 已有獨立列)
"""
import pandas as pd

df = pd.read_csv("../../feature_comparison_table.csv")
n0 = len(df)

# 修正 price 那列，拆成純 price 一列(移除bundle文字)
df.loc[df["listings_cleaned.csv.gz 原始欄位"] == "price", "對應最終特徵"] = "price"

new_rows = [
    # 區塊, 序號(沿用原序號+補.x), 原始欄位(標明組合來源), 採用狀態, 對應最終特徵, 說明
    ("② 衍生轉換", "29.1", "price + neighbourhood_cleansed + room_type（組合）", "衍生採用", "price_pctl_nbhd", "本房價格在「同行政區＋同房型」的百分位，屬第三方平台核心的競爭性特徵"),
    ("② 衍生轉換", "29.2", "review_scores_rating + neighbourhood_cleansed + room_type（組合）", "衍生採用", "score_pctl_nbhd", "本房評分在周邊同房型的百分位(情緒排名)，競爭性特徵"),
    ("② 衍生轉換", "29.3", "amenities + neighbourhood_cleansed + room_type（組合）", "衍生採用", "amenities_vs_median", "設施數量 ÷ 周邊同房型中位數，競爭性特徵"),
    ("② 衍生轉換", "31.1", "latitude + longitude（組合）", "衍生採用", "nbr_density_1km", "以經緯度用BallTree查詢周邊1km內房源數，供給密度競爭特徵"),
    ("② 衍生轉換", "31.2", "latitude + longitude + room_type（組合）", "衍生採用", "nbr_density_same_type_1km", "周邊1km內「同房型」房源數，供給密度競爭特徵"),
    ("③ 未採用", "9.1", "picture_url（同 photo_design_sense 來源）", "未採用-已測試無效", "photo_width", "OpenCV量測解析度寬度；與Y_vacancy全量相關係數僅-0.112，回歸模型中重要度排21/45"),
    ("③ 未採用", "9.2", "picture_url（同上）", "未採用-已測試無效", "photo_height", "OpenCV量測解析度高度；重要度排33/45"),
    ("③ 未採用", "9.3", "picture_url（同上）", "未採用-已測試無效", "photo_brightness", "OpenCV量測平均亮度；小樣本(n=500)相關係數-0.106看似最高，但全量(n=5849)驗算僅-0.023，訊號不穩健"),
    ("③ 未採用", "9.4", "picture_url（同上）", "未採用-已測試無效", "photo_contrast", "OpenCV量測對比度(灰階標準差)；重要度排37/45"),
    ("③ 未採用", "9.5", "picture_url（同上）", "未採用-已測試無效", "photo_sharpness", "OpenCV量測清晰度(Laplacian變異數)；重要度排44/45，貢獻近乎0"),
    ("③ 未採用", "9.6", "picture_url（同上）", "未採用-已測試無效", "photo_colorfulness", "OpenCV量測色彩豐富度；重要度排23/45"),
    ("③ 未採用", "9.7", "picture_url（同上）", "未採用-已測試無效", "photo_coziness", "CLIP zero-shot「溫馨感」機率；重要度排45/45(貢獻為負，等同雜訊)"),
    ("③ 未採用", "9.8", "picture_url（同上）", "未採用-已測試無效", "photo_cleanliness", "CLIP zero-shot「乾淨度」機率；平均值0.966、中位數0.991，判斷過度飽和，鑑別力弱"),
]
add_df = pd.DataFrame(new_rows, columns=df.columns)
df2 = pd.concat([df, add_df], ignore_index=True)

order = {"① 直接沿用": 0, "② 衍生轉換": 1, "③ 未採用": 2}
df2["_sort"] = df2["區塊"].map(order)
df2["_seq"] = df2["序號"].astype(str).str.split(".").str[0].astype(int)
df2["_sub"] = df2["序號"].astype(str).apply(lambda s: float(s) if "." in s else 0)
df2 = df2.sort_values(["_sort", "_seq", "_sub"]).drop(columns=["_sort", "_seq", "_sub"])

df2.to_csv("../../feature_comparison_table.csv", index=False, encoding="utf-8-sig")

print("原始列數: {} -> 新增 {} 列 -> 總列數: {}".format(n0, len(new_rows), len(df2)))
print("\n三大區塊列數統計:")
print(df2["區塊"].value_counts().sort_index().to_string())

# 驗證：37個最終特徵是否每個都至少出現一次在「對應最終特徵」欄
final_37 = ["price","accommodates","bedrooms","beds","bathrooms","minimum_nights","amenities_count",
    "instant_bookable","room_type_code","property_type_code","neighbourhood_code","latitude","longitude",
    "host_is_superhost","host_response_rate","host_acceptance_rate","host_listings_count","host_tenure_days",
    "review_scores_rating","review_scores_cleanliness","review_scores_location","review_scores_value",
    "review_scores_communication","review_scores_checkin","review_scores_accuracy","price_pctl_nbhd",
    "score_pctl_nbhd","amenities_vs_median","nbr_density_1km","nbr_density_same_type_1km","self_checkin",
    "response_speed","desc_len","host_about_len","maximum_nights","min_nights_avg_ntm","photo_design_sense"]
covered = df2["對應最終特徵"].tolist()
missing = [f for f in final_37 if not any(f == c or f in str(c).split(" / ") for c in covered)]
print("\n37個最終特徵逐一核對，尚未出現在表格中的:", missing if missing else "無，全數覆蓋")
