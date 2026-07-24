# -*- coding: utf-8 -*-
"""
比對 listings_cleaned.csv.gz 的 81 個原始欄位 與 目前模型採用的37個最終特徵，
生成完整對照表 feature_comparison_table.csv
狀態分類：
  直接採用   = 原始欄位未經轉換直接當特徵
  衍生採用   = 原始欄位經過轉換/編碼/組合運算後成為特徵
  未採用-防洩漏 = 與Y(空屋率/入住量)定義高度相關，為避免資料洩漏刻意排除
  未採用-已測試無效 = 曾納入候選並實測(加入前後比較)，證實無提升
  未採用-全空/無意義 = 欄位全空或屬中繼資訊(URL/ID/時間戳)，不具建模意義
  未採用-未測試 = 尚未嘗試過的欄位
"""
import pandas as pd

L = pd.read_csv("../../listings_cleaned.csv.gz", compression="gzip", low_memory=False, nrows=5)
raw_cols = list(L.columns)

# (原始欄位, 狀態, 對應最終特徵, 說明)
mapping = [
    ("id", "衍生採用", "listing_id(識別鍵)", "join各資料表的主鍵，非模型特徵本身"),
    ("listing_url", "未採用-全空/無意義", "-", "URL，無建模意義"),
    ("scrape_id", "未採用-全空/無意義", "-", "爬取批次ID，中繼資訊"),
    ("last_scraped", "衍生採用", "host_tenure_days", "與host_since相減，算房東經營天數"),
    ("source", "未採用-全空/無意義", "-", "資料來源標記，中繼資訊"),
    ("name", "未採用-未測試", "-", "房源標題文字，未做NLP特徵化"),
    ("description", "衍生採用", "desc_len", "取字數長度，代表經營用心度"),
    ("neighborhood_overview", "未採用-已測試無效", "-", "曾測試has_nbhd_overview(有無)，重要度排名30/49，無效"),
    ("picture_url", "衍生採用", "photo_design_sense", "下載封面照後用CLIP-ViT-B/32算設計感分數"),
    ("host_id", "衍生採用", "(GroupKFold分組依據)", "非模型特徵，用於交叉驗證時依房東分組，避免同房東房源跨train/test洩漏"),
    ("host_url", "未採用-全空/無意義", "-", "URL"),
    ("host_name", "未採用-全空/無意義", "-", "房東姓名文字，無建模意義"),
    ("host_since", "衍生採用", "host_tenure_days", "與last_scraped相減"),
    ("host_location", "未採用-未測試", "-", "房東所在地文字，未嘗試"),
    ("host_about", "衍生採用", "host_about_len", "取字數長度，代表房東經營用心度"),
    ("host_response_time", "衍生採用", "response_speed", "文字對應轉序數編碼(1~4)"),
    ("host_response_rate", "直接採用", "host_response_rate", "轉為0~1比例"),
    ("host_acceptance_rate", "直接採用", "host_acceptance_rate", "轉為0~1比例；模型中重要度最高特徵之一"),
    ("host_is_superhost", "直接採用", "host_is_superhost", "t/f轉0/1"),
    ("host_thumbnail_url", "未採用-全空/無意義", "-", "URL"),
    ("host_picture_url", "未採用-全空/無意義", "-", "URL"),
    ("host_neighbourhood", "未採用-未測試", "-", "缺值率高，未嘗試"),
    ("host_listings_count", "未採用-已測試無效", "-", "與calculated_host_listings_count重複，改用後者"),
    ("host_total_listings_count", "未採用-已測試無效", "-", "與calculated版本重複"),
    ("host_verifications", "未採用-已測試無效", "-", "曾測試host_verif_count(驗證項目數)，重要度排名36/49，無效"),
    ("host_has_profile_pic", "未採用-已測試無效", "-", "曾測試，重要度排名33/49，邊緣無效；t/f幾乎全為t，鑑別度低"),
    ("host_identity_verified", "未採用-已測試無效", "-", "曾測試，重要度排名40/49，99%皆為已驗證，鑑別度極低"),
    ("neighbourhood", "未採用-全空/無意義", "-", "原始未清理字串，改用neighbourhood_cleansed"),
    ("neighbourhood_cleansed", "衍生採用", "neighbourhood_code", "類別編碼；同時是price_pctl_nbhd/score_pctl_nbhd/amenities_vs_median的分組依據"),
    ("neighbourhood_group_cleansed", "未採用-全空/無意義", "-", "全欄位皆缺值"),
    ("latitude", "直接採用", "latitude", "同時用於算nbr_density_1km/nbr_density_same_type_1km"),
    ("longitude", "直接採用", "longitude", "同時用於算nbr_density_1km/nbr_density_same_type_1km"),
    ("property_type", "衍生採用", "property_type_code", "類別編碼"),
    ("room_type", "衍生採用", "room_type_code", "類別編碼；同時是競爭特徵的分組依據"),
    ("accommodates", "直接採用", "accommodates", "-"),
    ("bathrooms", "直接採用", "bathrooms", "-"),
    ("bathrooms_text", "未採用-已測試無效", "-", "與bathrooms/is_shared_bath重複"),
    ("bedrooms", "直接採用", "bedrooms", "-"),
    ("beds", "直接採用", "beds", "-"),
    ("amenities", "衍生採用", "amenities_count / self_checkin", "計數設施數量；並用關鍵字比對萃取「是否提供自助入住」旗標(模型中重要特徵)"),
    ("price", "直接採用", "price / price_pctl_nbhd", "去除貨幣符號轉數值；同時用於算周邊價格百分位"),
    ("minimum_nights", "直接採用", "minimum_nights", "-"),
    ("maximum_nights", "直接採用", "maximum_nights", "模型中重要度最高特徵之一(訂房彈性)"),
    ("minimum_minimum_nights", "未採用-未測試", "-", "與minimum_nights高度重複，未嘗試"),
    ("maximum_minimum_nights", "未採用-未測試", "-", "同上"),
    ("minimum_maximum_nights", "未採用-未測試", "-", "同上"),
    ("maximum_maximum_nights", "未採用-未測試", "-", "同上"),
    ("minimum_nights_avg_ntm", "直接採用", "min_nights_avg_ntm", "模型中重要度前段特徵之一"),
    ("maximum_nights_avg_ntm", "未採用-未測試", "-", "與maximum_nights高度重複，未嘗試"),
    ("calendar_updated", "未採用-全空/無意義", "-", "全欄位皆缺值"),
    ("has_availability", "未採用-全空/無意義", "-", "輔助判讀用旗標，非特徵"),
    ("availability_30", "未採用-防洩漏", "-", "與空屋率高度相關，防止資料洩漏"),
    ("availability_60", "未採用-防洩漏", "-", "同上"),
    ("availability_90", "未採用-防洩漏", "-", "同上"),
    ("availability_365", "未採用-防洩漏", "★用於定義Y(空屋率)", "Y_vacancy = availability_365/365，此欄位本身即標籤來源，不可當X"),
    ("calendar_last_scraped", "未採用-全空/無意義", "-", "中繼時間戳"),
    ("number_of_reviews", "未採用-防洩漏", "-", "與入住量高度相關，防止洩漏"),
    ("number_of_reviews_ltm", "未採用-防洩漏", "-", "同上"),
    ("number_of_reviews_l30d", "未採用-防洩漏", "-", "同上"),
    ("availability_eoy", "未採用-防洩漏", "-", "同上"),
    ("number_of_reviews_ly", "未採用-防洩漏", "-", "同上"),
    ("estimated_occupancy_l365d", "未採用-防洩漏", "-", "早期版本Y曾用此欄位定義，與現用Y高度相關，防止洩漏"),
    ("estimated_revenue_l365d", "未採用-防洩漏", "-", "與入住量高度相關，防止洩漏"),
    ("first_review", "未採用-未測試", "-", "可算房源年資，但已用host_tenure_days替代"),
    ("last_review", "未採用-防洩漏", "-", "與近期入住狀況高度相關，防止洩漏"),
    ("review_scores_rating", "直接採用", "review_scores_rating", "-"),
    ("review_scores_accuracy", "直接採用", "review_scores_accuracy", "-"),
    ("review_scores_cleanliness", "直接採用", "review_scores_cleanliness", "-"),
    ("review_scores_checkin", "直接採用", "review_scores_checkin", "-"),
    ("review_scores_communication", "直接採用", "review_scores_communication", "-"),
    ("review_scores_location", "直接採用", "review_scores_location", "同時用於算score_pctl_nbhd(周邊口碑排名)"),
    ("review_scores_value", "直接採用", "review_scores_value", "-"),
    ("license", "未採用-全空/無意義", "-", "全欄位皆缺值"),
    ("instant_bookable", "直接採用", "instant_bookable", "t/f轉0/1"),
    ("calculated_host_listings_count", "衍生採用", "host_listings_count", "更名採用，模型中重要度最高特徵之一"),
    ("calculated_host_listings_count_entire_homes", "未採用-未測試", "-", "與calculated_host_listings_count重複"),
    ("calculated_host_listings_count_private_rooms", "未採用-未測試", "-", "同上"),
    ("calculated_host_listings_count_shared_rooms", "未採用-未測試", "-", "同上"),
    ("reviews_per_month", "未採用-防洩漏", "-", "與入住量高度相關，防止洩漏"),
    ("bathrooms_count", "未採用-已測試無效", "-", "與bathrooms重複"),
    ("is_shared_bath", "未採用-已測試無效", "-", "曾測試，重要度排名38/49，無效"),
]

assert len(mapping) == len(raw_cols), "數量不符：mapping={} raw_cols={}".format(len(mapping), len(raw_cols))
missing = set(raw_cols) - set(m[0] for m in mapping)
extra = set(m[0] for m in mapping) - set(raw_cols)
assert not missing, "漏掉的欄位: {}".format(missing)
assert not extra, "多出不存在的欄位: {}".format(extra)

df_out = pd.DataFrame(mapping, columns=["listings_cleaned.csv.gz 原始欄位", "採用狀態", "對應最終特徵", "說明"])
df_out.insert(0, "序號", range(1, len(df_out) + 1))
df_out.to_csv("../../feature_comparison_table.csv", index=False, encoding="utf-8-sig")

print("驗證通過：81個原始欄位全部對應完成，無遺漏無多餘。")
print("\n採用狀態統計:")
print(df_out["採用狀態"].value_counts().to_string())
print("\n已存檔: feature_comparison_table.csv")
