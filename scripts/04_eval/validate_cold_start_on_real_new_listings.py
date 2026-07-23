# -*- coding: utf-8 -*-
"""
用真實的392筆「經營未滿1年」房源(先前被MIN_TENURE過濾掉)，
實測24特徵冷啟動模型的預測表現。
⚠️ 重要但書：這些房源的 availability_365(=Y來源)本身可能因房源尚未滿一年而被低估，
   結果僅供方向性參考，非嚴謹的乾淨驗證。
"""
import numpy as np, pandas as pd
from sklearn.neighbors import BallTree
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score

L = pd.read_csv("../../listings_cleaned.csv.gz", compression="gzip", low_memory=False)

def to_price(s): return pd.to_numeric(s.astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")
def tf(s): return (s.astype(str).str.lower() == "t").astype(int)
def amen(s): return s.fillna("[]").astype(str).str.count(",") + 1

L["host_tenure_days"] = (pd.to_datetime(L["last_scraped"], errors="coerce") - pd.to_datetime(L["host_since"], errors="coerce")).dt.days
L["Y_vacancy"] = (pd.to_numeric(L["availability_365"], errors="coerce") / 365.0).clip(0, 1)
L["price_num"] = to_price(L["price"])
L["amenities_count"] = amen(L["amenities"])
am = L["amenities"].fillna("").str.lower()
L["self_checkin"] = (am.str.contains("self check", regex=False) | am.str.contains("keypad", regex=False)
                     | am.str.contains("lockbox", regex=False) | am.str.contains("smart lock", regex=False)).astype(int)
L["desc_len"] = L["description"].fillna("").astype(str).str.len()
L["host_about_len"] = L["host_about"].fillna("").astype(str).str.len()
L["instant_bookable"] = tf(L["instant_bookable"])
L["host_listings_count"] = pd.to_numeric(L["calculated_host_listings_count"], errors="coerce")
L["maximum_nights"] = pd.to_numeric(L["maximum_nights"], errors="coerce")
L["min_nights_avg_ntm"] = pd.to_numeric(L["minimum_nights_avg_ntm"], errors="coerce")
for c in ["room_type", "property_type", "neighbourhood_cleansed"]:
    L[c + "_code"] = L[c].astype("category").cat.codes

# 競爭特徵：以「全體6,241筆」為母體計算相對位置(新房源要跟既有市場比較，而非只跟自己這群比)
g2 = [L["neighbourhood_cleansed"], L["room_type"]]
L["price_pctl_nbhd"] = L.groupby(g2)["price_num"].rank(pct=True)
L["amenities_vs_median"] = L["amenities_count"] / L.groupby(g2)["amenities_count"].transform("median").replace(0, np.nan)
coords = np.radians(L[["latitude", "longitude"]].values)
tree = BallTree(coords, metric="haversine"); r = 1.0 / 6371.0
L["nbr_density_1km"] = tree.query_radius(coords, r=r, count_only=True) - 1
rt = L["room_type"].values; dens = np.zeros(len(L), int)
for i, nb in enumerate(tree.query_radius(coords, r=r)):
    dens[i] = int((rt[nb] == rt[i]).sum()) - 1
L["nbr_density_same_type_1km"] = dens

# photo_design_sense：從既有 dataset_final.csv 合併(已算好的CLIP分數)，若無則中位數補
photo = pd.read_csv("../../dataset_final.csv")[["listing_id", "photo_design_sense"]]
L = L.merge(photo, left_on="id", right_on="listing_id", how="left")
L["photo_design_sense"] = L["photo_design_sense"].fillna(L["photo_design_sense"].median())

COLD24 = ["price_num","accommodates","bedrooms","beds","bathrooms","minimum_nights","amenities_count",
    "instant_bookable","room_type_code","property_type_code","neighbourhood_cleansed_code","latitude","longitude",
    "host_listings_count","price_pctl_nbhd","amenities_vs_median","nbr_density_1km","nbr_density_same_type_1km",
    "self_checkin","desc_len","host_about_len","maximum_nights","min_nights_avg_ntm","photo_design_sense"]
for c in COLD24:
    if L[c].dtype != int:
        L[c] = pd.to_numeric(L[c], errors="coerce")
        L[c] = L[c].fillna(L[c].median())

# 訓練集：經營滿1年以上的 5,849 筆(可靠Y)；測試集：392筆未滿1年(Y有偏誤風險)
established = L[L["host_tenure_days"] >= 365].dropna(subset=["Y_vacancy"]).copy()
new_listings = L[L["host_tenure_days"] < 365].dropna(subset=["Y_vacancy"]).copy()
print("訓練集(established, tenure>=365天): {} 筆".format(len(established)))
print("測試集(new_listings, tenure<365天): {} 筆".format(len(new_listings)))

Xtr, ytr = established[COLD24], established["Y_vacancy"].values
Xte, yte = new_listings[COLD24], new_listings["Y_vacancy"].values
ybtr = (ytr > 0.7).astype(int)
ybte = (yte > 0.7).astype(int)

reg = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, l2_regularization=1.0, random_state=42).fit(Xtr, ytr)
clf = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, l2_regularization=1.0, random_state=42).fit(Xtr, ybtr)
pred_reg = np.clip(reg.predict(Xte), 0, 1)
pred_clf = clf.predict_proba(Xte)[:, 1]

print("\n" + "="*60)
print("[注意] 以下數字的Y(空屋率)可能因房源未滿1年而被低估，僅供方向參考")
print("="*60)
print("整體(392筆): 回歸 R^2={:.3f} MAE={:.3f}  |  分類 AUC={:.3f}".format(
      r2_score(yte, pred_reg), mean_absolute_error(yte, pred_reg),
      roc_auc_score(ybte, pred_clf) if 5 <= ybte.sum() < len(ybte) else float("nan")))

print("\n依經營天數分段(越新的段，Y偏誤風險越高):")
bins = [(0, 90, "0~90天(最新，Y最不可靠)"), (90, 180, "90~180天"), (180, 365, "180~365天(最接近訓練集)")]
for lo, hi, label in bins:
    m = (new_listings["host_tenure_days"] >= lo) & (new_listings["host_tenure_days"] < hi)
    n = m.sum()
    if n < 10:
        print("  {:28s} n={:3d} (樣本太少，略過)".format(label, n))
        continue
    yt, pr = yte[m.values], pred_reg[m.values]
    ybm, pcm = ybte[m.values], pred_clf[m.values]
    r2 = r2_score(yt, pr); mae = mean_absolute_error(yt, pr)
    auc = roc_auc_score(ybm, pcm) if 5 <= ybm.sum() < len(ybm) else float("nan")
    print("  {:28s} n={:3d}  回歸R^2={:.3f} MAE={:.3f}  分類AUC={:.3f}  平均實際Y={:.3f}  平均預測Y={:.3f}".format(
        label, n, r2, mae, auc, yt.mean(), pr.mean()))

# 真正零評論的子集(最接近真實冷啟動)
zero_review = new_listings["number_of_reviews"] == 0
print("\n真正零評論子集 (n={}):".format(zero_review.sum()))
if zero_review.sum() >= 10:
    m = zero_review.values
    yt, pr = yte[m], pred_reg[m]
    print("  平均實際Y(空屋率)={:.3f}  平均預測風險分數={:.3f}  MAE={:.3f}".format(
        yt.mean(), pr.mean(), mean_absolute_error(yt, pr)))
