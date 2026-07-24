"""
租客入口｜五科成績單計分引擎（依「新版房源評分模式規劃書 v1.0」）。

交通 / 生活 / 價格 / 口碑 / 設備，各 5 分、總分 25。
先以必要條件硬性篩選，再五科計分，最後依「最在意兩科」優先分排序。

設計原則：
  • 純計分函式（transit_score / life_score / price_score_from_D / reputation_score
    / amenity_score / priority_score / total_and_band …）不碰 I/O、可單元測試，
    直接重現規劃書中的計算例子。
  • 資料抽取 helper（compute_geo_facts / price_comparison /
    review_sentiment_breakdown）負責與 geo_utils、nlp_analysis、DataFrame 整合。
  • 資料不足時該科回傳 None（畫面標「資料不足」），總分以 0 計入但仍保留房源。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from modules.geo_utils import haversine, poi_points_within

SCORING_RULES_VERSION = "v1.0"

SUBJECTS = ["transit", "life", "price", "reputation", "amenity"]
SUBJECT_ZH = {
    "transit": "交通方便",
    "life": "生活便利",
    "price": "價格合理",
    "reputation": "住客口碑",
    "amenity": "房源設備",
}

# 設備標籤 → amenities 欄位英文關鍵字（小寫；任一命中即視為具備）
AMENITY_KEYWORDS = {
    "Wi-Fi": ["wifi"],
    "冷氣": ["air condition", "ac -", "cooling"],
    "洗衣機": ["washer"],
    "電視": ["tv", "hdtv"],
    "電冰箱": ["refrigerator", "fridge"],
    "廚房": ["kitchen"],
    "電梯": ["elevator"],
    "陽台": ["balcony", "patio"],
    "熱水": ["hot water"],
    "吹風機": ["hair dryer"],
    "停車": ["parking"],
}
# 側欄預設帶入的「希望設備」（確保 k≥1）
DEFAULT_WISH = ["Wi-Fi", "冷氣", "洗衣機"]


# ═══════════════════════════════════════════════════════════════
# 第一科：交通方便（T = M + B，上限 5）
# ═══════════════════════════════════════════════════════════════
def mrt_points(dist_m) -> int:
    """捷運出口最近直線距離 → 0~3 分。None/inf 視為無站（0 分）。"""
    if dist_m is None or not np.isfinite(dist_m):
        return 0
    if dist_m <= 400:
        return 3
    if dist_m <= 800:
        return 2
    if dist_m <= 1200:
        return 1
    return 0


def bus_points(dist_m) -> int:
    """公車站牌最近直線距離 → 0~2 分。"""
    if dist_m is None or not np.isfinite(dist_m):
        return 0
    if dist_m <= 300:
        return 2
    if dist_m <= 600:
        return 1
    return 0


def transit_score(mrt_dist_m, bus_dist_m):
    """回傳 (T, detail)。detail 含 M、B 與兩距離。"""
    M = mrt_points(mrt_dist_m)
    B = bus_points(bus_dist_m)
    return M + B, {"M": M, "B": B,
                   "mrt_dist_m": mrt_dist_m, "bus_dist_m": bus_dist_m}


# ═══════════════════════════════════════════════════════════════
# 第二科：生活便利（L = C + F + H + P，上限 5）
# ═══════════════════════════════════════════════════════════════
def life_score(conv_cnt, rest_cnt, clinic_cnt, park_cnt):
    """
    conv_cnt   : 500m 內超商/超市數
    rest_cnt   : 800m 內去重後餐飲地點數
    clinic_cnt : 1km 內診所數
    park_cnt   : 1km 內公園數
    """
    C = 2 if conv_cnt >= 1 else 0
    F = 1 if rest_cnt >= 5 else 0
    H = 1 if clinic_cnt >= 1 else 0
    P = 1 if park_cnt >= 1 else 0
    return C + F + H + P, {"C": C, "F": F, "H": H, "P": P,
                           "conv_cnt": conv_cnt, "rest_cnt": rest_cnt,
                           "clinic_cnt": clinic_cnt, "park_cnt": park_cnt}


# ═══════════════════════════════════════════════════════════════
# 第三科：價格合理（依差異率 D 分 5 段，上限 5）
# ═══════════════════════════════════════════════════════════════
def price_diff_pct(price, median):
    """D = (P - Md) / Md × 100%；median 無效回傳 None。"""
    if median is None or not np.isfinite(median) or median == 0:
        return None
    if price is None or not np.isfinite(price):
        return None
    return (price - median) / median * 100.0


def price_score_from_D(D):
    """差異率 D(%) → 1~5 分；D 為 None（資料不足）回傳 None。"""
    if D is None:
        return None
    if D <= -10:
        return 5
    if D <= 0:
        return 4
    if D <= 10:
        return 3
    if D <= 25:
        return 2
    return 1


def price_comparison(listing, market_df, occ_tol=1, min_group=10):
    """
    以「同行政區 + 同房型 + 可住人數差 ≤occ_tol」為比較組，取價格中位數。
    樣本 <min_group 時放寬人數限制（同區 + 同房型），並標示 relaxed=True。
    回傳 (median 或 None, group_size, relaxed)。market_df 為市場母體（未經預算篩選）。
    """
    dist = listing.get("neighbourhood_cleansed")
    rt = listing.get("room_type_zh")
    acc = listing.get("accommodates", np.nan)
    base = market_df[(market_df["neighbourhood_cleansed"] == dist)
                     & (market_df["room_type_zh"] == rt)]
    base = base[base["price"].notna()]
    relaxed = False
    grp = base
    if pd.notna(acc):
        narrow = base[(base["accommodates"] - acc).abs() <= occ_tol]
        if len(narrow) >= min_group:
            grp = narrow
        else:
            relaxed = len(base) != len(narrow)
            grp = base
    if len(grp) == 0:
        return None, 0, relaxed
    return float(grp["price"].median()), int(len(grp)), relaxed


def price_score(price, market_df, listing, **kw):
    """整合：算比較組中位數 → D → 分數。回傳 (分數 或 None, detail)。"""
    median, n, relaxed = price_comparison(listing, market_df, **kw)
    D = price_diff_pct(price, median)
    pts = price_score_from_D(D)
    return pts, {"median": median, "D": D, "group_size": n,
                 "relaxed": relaxed, "insufficient": median is None}


# ═══════════════════════════════════════════════════════════════
# 第四科：住客口碑（R = min(A + N, C)，上限 5）
# ═══════════════════════════════════════════════════════════════
def reputation_from_rating(rating):
    """Airbnb 平均評分 → A（0~3 分）；無評分回傳 None（資料不足）。"""
    if rating is None or pd.isna(rating) or not np.isfinite(rating):
        return None
    if rating >= 4.8:
        return 3.0
    if rating >= 4.5:
        return 2.5
    if rating >= 4.2:
        return 2.0
    if rating >= 4.0:
        return 1.0
    return 0.0


def nlp_content_score(p_pos, p_neg):
    """N = clamp(1 + P⁺ − P⁻, 0, 2)。"""
    return float(np.clip(1.0 + p_pos - p_neg, 0.0, 2.0))


def review_cap(n_effective):
    """有效評論總數 → 口碑最高可得分上限。"""
    if n_effective <= 0:
        return 0
    if n_effective <= 4:
        return 3
    if n_effective <= 19:
        return 4
    return 5


def reputation_score(rating, pos_n, neg_n, n_analyzable, n_total):
    """
    rating       : Airbnb 平均評分（無 → 資料不足）
    pos_n / neg_n: 近 20 則窗內正/負面則數
    n_analyzable : 近 20 則窗內可分析評論總數（P⁺/P⁻ 的分母）
    n_total      : 有效評論總數（決定上限 C）
    回傳 (R 或 None, detail)。無評分或無評論證據 → None（資料不足）。
    """
    A = reputation_from_rating(rating)
    cap = review_cap(n_total)
    if n_analyzable > 0:
        p_pos = pos_n / n_analyzable
        p_neg = neg_n / n_analyzable
        N = round(nlp_content_score(p_pos, p_neg), 2)
    else:
        p_pos = p_neg = 0.0
        N = None
    detail = {"A": A, "N": N, "cap": cap, "p_pos": round(p_pos, 3),
              "p_neg": round(p_neg, 3), "n_analyzable": n_analyzable,
              "n_total": n_total}
    if cap == 0:
        detail["insufficient"] = "無評論證據"
        return None, detail
    if A is None:
        detail["insufficient"] = "無評分"
        return None, detail
    Nval = N if N is not None else 1.0  # 有評分無可分析文字 → NLP 中性 1 分
    R = min(A + Nval, cap)
    detail["Rraw"] = round(A + Nval, 2)
    detail["insufficient"] = None
    return round(R, 2), detail


# ═══════════════════════════════════════════════════════════════
# 第五科：房源設備（E = m / k × 5，上限 5）
# ═══════════════════════════════════════════════════════════════
def match_amenities(amenities_text, labels, kw_map=AMENITY_KEYWORDS):
    """回傳 [bool]，對應 labels 是否在 amenities_text 中命中。"""
    low = str(amenities_text).lower()
    out = []
    for lbl in labels:
        kws = kw_map.get(lbl, [str(lbl).lower()])
        out.append(any(w in low for w in kws))
    return out


def amenity_score(present_flags):
    """present_flags: 各希望設備是否具備。E = m/k×5；k=0 回傳 None。"""
    k = len(present_flags)
    if k == 0:
        return None, {"m": 0, "k": 0}
    m = int(sum(1 for x in present_flags if x))
    return round(m / k * 5.0, 2), {"m": m, "k": k}


def has_all_amenities(amenities_text, must_labels, kw_map=AMENITY_KEYWORDS):
    """必備設備硬性篩選：全部命中才 True。must_labels 為空時 True。"""
    if not must_labels:
        return True
    return all(match_amenities(amenities_text, must_labels, kw_map))


# ═══════════════════════════════════════════════════════════════
# 排序、總分、推薦理由
# ═══════════════════════════════════════════════════════════════
def priority_score(scores, top2):
    """優先分 Q = 最在意兩科分數相加（資料不足科目以 0 計）。"""
    return float(sum((scores.get(s) or 0.0) for s in (top2 or [])))


def total_and_band(scores):
    """五科總分 S 與區間標籤。資料不足科目以 0 計入。"""
    S = round(sum((v if v is not None else 0.0) for v in scores.values()), 1)
    if S >= 20:
        band = "優先查看"
    elif S >= 15:
        band = "值得考慮"
    elif S >= 10:
        band = "普通"
    else:
        band = "建議多比較"
    return S, band


def sort_key(record, top2):
    """排序鍵：優先分↓ → 總分↓ → 評論數↓ → 價格↑。"""
    Q = priority_score(record["scores"], top2)
    S = record.get("total", total_and_band(record["scores"])[0])
    n_rev = record.get("n_reviews", 0) or 0
    price = record.get("price", float("inf")) or float("inf")
    return (-Q, -S, -n_rev, price)


def rank_records(records, top2):
    """依 sort_key 排序 records（list[dict]），回傳新的排序後 list。"""
    return sorted(records, key=lambda r: sort_key(r, top2))


def recommend_reason(scores, details, wish_labels=None):
    """依各科細節組出「✓ 加分 / △ 提醒」推薦理由（中文）。"""
    plus, minus = [], []
    td = details.get("transit", {})
    if td.get("mrt_dist_m") is not None and np.isfinite(td.get("mrt_dist_m", np.inf)) \
            and td["mrt_dist_m"] <= 800:
        plus.append(f"捷運出口約 {td['mrt_dist_m']:.0f}m")
    elif td.get("M", 0) == 0 and td.get("B", 0) == 0:
        minus.append("大眾運輸較遠")

    ld = details.get("life", {})
    if ld.get("C"):
        plus.append("500m 內有超商")
    if not ld.get("P"):
        minus.append("1km 內未找到公園")

    pd_ = details.get("price", {})
    if pd_.get("D") is not None:
        if pd_["D"] <= -5:
            plus.append(f"低於同類中位約 {abs(pd_['D']):.0f}%")
        elif pd_["D"] > 10:
            minus.append(f"高於同類中位約 {pd_['D']:.0f}%")

    rd = details.get("reputation", {})
    if scores.get("reputation") is not None and scores["reputation"] >= 4:
        plus.append(f"評分佳、評論證據充足（{rd.get('n_total', 0)} 則）")
    elif rd.get("insufficient"):
        minus.append("口碑資料不足")

    ad = details.get("amenity", {})
    if ad.get("k"):
        plus.append(f"符合 {ad.get('m', 0)}/{ad['k']} 項偏好設備")

    parts = []
    if plus:
        parts.append("　".join(f"✓ {p}" for p in plus[:4]))
    if minus:
        parts.append("　".join(f"△ {m}" for m in minus[:2]))
    return "；".join(parts) if parts else "各項表現平均。"


# ═══════════════════════════════════════════════════════════════
# 資料抽取 helper（與 geo_utils / nlp_analysis / DataFrame 整合）
# ═══════════════════════════════════════════════════════════════
def _nearest_dist(lat, lon, poi_df):
    if poi_df is None or poi_df.empty:
        return float("inf")
    d = haversine(lat, lon, poi_df["latitude"].values, poi_df["longitude"].values)
    return float(np.min(d)) if len(d) else float("inf")


def _dedup_count_within(lat, lon, poi_df, radius_m, round_dp=4):
    """半徑內去重地點數（依名稱＋四捨五入座標去重）。"""
    pts = poi_points_within(lat, lon, poi_df, radius_m)
    if pts.empty:
        return 0
    pts = pts.assign(_la=pts["latitude"].round(round_dp),
                     _lo=pts["longitude"].round(round_dp))
    return int(pts.drop_duplicates(subset=["poi_name", "_la", "_lo"]).shape[0])


def compute_geo_facts(lat, lon, poi):
    """
    由座標算出五科所需的地理事實：
      mrt_dist_m, bus_dist_m,
      conv_cnt(500m), rest_cnt(800m 去重), clinic_cnt(1km), park_cnt(1km)
    """
    return {
        "mrt_dist_m": _nearest_dist(lat, lon, poi.get("mrt")),
        "bus_dist_m": _nearest_dist(lat, lon, poi.get("bus")),
        "conv_cnt": _dedup_count_within(lat, lon, poi.get("convenience"), 500),
        "rest_cnt": _dedup_count_within(lat, lon, poi.get("restaurant"), 800),
        "clinic_cnt": _dedup_count_within(lat, lon, poi.get("clinic"), 1000),
        "park_cnt": _dedup_count_within(lat, lon, poi.get("park"), 1000),
    }


def review_sentiment_breakdown(reviews_df, listing_id, window=20):
    """
    近 window 則可分析評論的情感統計。
    回傳 dict：pos_n, neg_n, neu_n, n_analyzable, n_total。
    無法分析（空白/過短/不支援語言）者排除、不計入可分析總數（不當中立）。
    """
    from modules.nlp_analysis import analyze_sentiment
    r = reviews_df[reviews_df["listing_id"] == listing_id]
    n_total = int(len(r))
    if n_total == 0:
        return {"pos_n": 0, "neg_n": 0, "neu_n": 0,
                "n_analyzable": 0, "n_total": 0}
    if "date" in r.columns:
        r = r.sort_values("date", ascending=False)
    r = r.head(window)
    col = "cleaned_comments" if "cleaned_comments" in r.columns else "comments"
    pos = neg = neu = 0
    analyzable = 0
    for _, row in r.iterrows():
        txt = str(row.get(col, "") or "").strip()
        lang = str(row.get("language_type", "en"))
        if len(txt) < 3 or lang == "other":
            continue
        s = analyze_sentiment(txt, lang=lang)
        analyzable += 1
        if s["label"] == "正面":
            pos += 1
        elif s["label"] == "負面":
            neg += 1
        else:
            neu += 1
    return {"pos_n": pos, "neg_n": neg, "neu_n": neu,
            "n_analyzable": analyzable, "n_total": n_total}
