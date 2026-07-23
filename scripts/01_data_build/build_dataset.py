# -*- coding: utf-8 -*-
"""
智慧旅宿滯銷風險預警平台 — 訓練資料集建置（v2：相對同儕門檻）
輸入：listings.csv.gz（Inside Airbnb 詳細版）
輸出：dataset_train.csv（Y + 結構化X + 面向情緒X + 競爭特徵X）

Y 定義（v2）：入住天數落在「同區同房型」後 30% → 滯銷（相對門檻，公平）
過濾：經營未滿1年、同儕群組太小(<5) → 資料不足，不納入訓練
防洩漏：入住/空房/評論數欄位只用來定 Y，不進 X
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

SRC = "../../listings.csv.gz"
OUT = "../../dataset_train.csv"
REL_Q = 0.30          # 同儕後30% → 滯銷
MIN_GROUP = 5         # 同區同房型至少5間才穩定
MIN_TENURE = 365      # 經營未滿1年剔除
RADIUS_KM = 1.0

df = pd.read_csv(SRC, compression="gzip", low_memory=False)
n0 = len(df)

def to_price(s): return pd.to_numeric(s.astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")
def to_pct(s):   return pd.to_numeric(s.astype(str).str.replace("%", "", regex=False), errors="coerce")/100.0
def tf(s):       return (s.astype(str).str.lower() == "t").astype(int)
def amen(s):     return s.fillna("[]").astype(str).str.count(",") + 1

# ---------- Y：相對同儕門檻 ----------
df["occ"] = pd.to_numeric(df["estimated_occupancy_l365d"], errors="coerce")
df = df[df["occ"].notna()].copy()
grp = df.groupby(["neighbourhood_cleansed", "room_type"])["occ"]
df["occ_pctl"] = grp.rank(pct=True)
df["grp_size"] = grp.transform("size")
df["Y_stale"] = (df["occ_pctl"] <= REL_Q).astype(int)

# ---------- 特徵 ----------
X = pd.DataFrame(index=df.index)
X["price"] = to_price(df["price"])
X["accommodates"] = pd.to_numeric(df["accommodates"], errors="coerce")
X["bedrooms"] = pd.to_numeric(df["bedrooms"], errors="coerce")
X["beds"] = pd.to_numeric(df["beds"], errors="coerce")
X["bathrooms"] = pd.to_numeric(df["bathrooms"], errors="coerce")
X["minimum_nights"] = pd.to_numeric(df["minimum_nights"], errors="coerce")
X["amenities_count"] = amen(df["amenities"])
X["instant_bookable"] = tf(df["instant_bookable"])
X["room_type"] = df["room_type"]
X["property_type"] = df["property_type"]
X["neighbourhood"] = df["neighbourhood_cleansed"]
X["latitude"] = df["latitude"]; X["longitude"] = df["longitude"]
X["host_is_superhost"] = tf(df["host_is_superhost"])
X["host_response_rate"] = to_pct(df["host_response_rate"])
X["host_acceptance_rate"] = to_pct(df["host_acceptance_rate"])
X["host_listings_count"] = pd.to_numeric(df["calculated_host_listings_count"], errors="coerce")
X["host_tenure_days"] = (pd.to_datetime(df["last_scraped"], errors="coerce")
                         - pd.to_datetime(df["host_since"], errors="coerce")).dt.days
for c in ["review_scores_rating","review_scores_cleanliness","review_scores_location",
          "review_scores_value","review_scores_communication","review_scores_checkin",
          "review_scores_accuracy"]:
    X[c] = pd.to_numeric(df[c], errors="coerce")

# 競爭特徵
g2 = [X["neighbourhood"], X["room_type"]]
X["price_pctl_nbhd"] = X.groupby(g2)["price"].rank(pct=True)
X["score_pctl_nbhd"] = X.groupby(g2)["review_scores_rating"].rank(pct=True)
X["amenities_vs_median"] = X["amenities_count"] / X.groupby(g2)["amenities_count"].transform("median").replace(0, np.nan)
coords = np.radians(df[["latitude","longitude"]].values)
tree = BallTree(coords, metric="haversine"); r = RADIUS_KM/6371.0
X["nbr_density_1km"] = tree.query_radius(coords, r=r, count_only=True) - 1
rt = df["room_type"].values; dens = np.zeros(len(df), int)
for i, nb in enumerate(tree.query_radius(coords, r=r)):
    dens[i] = int((rt[nb] == rt[i]).sum()) - 1
X["nbr_density_same_type_1km"] = dens
for c in ["room_type","property_type","neighbourhood"]:
    X[c+"_code"] = X[c].astype("category").cat.codes

num = X.select_dtypes(include=[np.number]).columns
X[num] = X[num].fillna(X[num].median())

feature_cols = ["price","accommodates","bedrooms","beds","bathrooms","minimum_nights",
    "amenities_count","instant_bookable","room_type_code","property_type_code",
    "neighbourhood_code","latitude","longitude","host_is_superhost","host_response_rate",
    "host_acceptance_rate","host_listings_count","host_tenure_days",
    "review_scores_rating","review_scores_cleanliness","review_scores_location",
    "review_scores_value","review_scores_communication","review_scores_checkin",
    "review_scores_accuracy","price_pctl_nbhd","score_pctl_nbhd","amenities_vs_median",
    "nbr_density_1km","nbr_density_same_type_1km"]

out = X[feature_cols].copy()
out["listing_id"] = df["id"].values
out["Y_stale"] = df["Y_stale"].values
out["_tenure"] = X["host_tenure_days"].values
out["_grp"] = df["grp_size"].values

# ---------- 過濾：未滿1年 / 群組太小 ----------
before = len(out)
mask = (out["_tenure"] >= MIN_TENURE) & (out["_grp"] >= MIN_GROUP)
out = out[mask].drop(columns=["_tenure","_grp"])
out = out[["listing_id"]+feature_cols+["Y_stale"]]
out.to_csv(OUT, index=False, encoding="utf-8-sig")

print("raw:", n0, "-> with occupancy:", before, "-> after filter:", len(out))
print("dropped (<1yr or small group):", before-len(out))
print("output:", OUT, "|", out.shape[0], "rows x", len(feature_cols), "features")
print("Y stale rate: {:.1f}%".format(out["Y_stale"].mean()*100),
      "| counts:", dict(out["Y_stale"].value_counts()))
