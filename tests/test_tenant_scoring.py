"""
tenant_scoring 單元測試 — 直接重現「新版房源評分模式規劃書 v1.0」的計算例子。
每個測試對應 PDF 的一段，確保計分引擎忠實於規劃書。
"""
import numpy as np
import pandas as pd
import pytest

from modules import tenant_scoring as ts


# ── 交通（§3.3）：捷運350m→3、公車520m→1、T=4 ──
def test_transit_pdf_example():
    T, d = ts.transit_score(350, 520)
    assert d["M"] == 3
    assert d["B"] == 1
    assert T == 4


@pytest.mark.parametrize("dist,exp", [(400, 3), (401, 2), (800, 2),
                                      (801, 1), (1200, 1), (1201, 0),
                                      (None, 0), (np.inf, 0)])
def test_mrt_boundaries(dist, exp):
    assert ts.mrt_points(dist) == exp


@pytest.mark.parametrize("dist,exp", [(300, 2), (301, 1), (600, 1),
                                      (601, 0), (None, 0)])
def test_bus_boundaries(dist, exp):
    assert ts.bus_points(dist) == exp


# ── 生活（§4.2）：超商2 + 餐飲8家(≥5)1 + 診所1 + 公園0 = 4 ──
def test_life_pdf_example():
    L, d = ts.life_score(conv_cnt=3, rest_cnt=8, clinic_cnt=1, park_cnt=0)
    assert (d["C"], d["F"], d["H"], d["P"]) == (2, 1, 1, 0)
    assert L == 4


def test_life_restaurant_threshold():
    # 餐飲需 ≥5 家才得分
    assert ts.life_score(1, 4, 1, 1)[0] == 2 + 0 + 1 + 1
    assert ts.life_score(1, 5, 1, 1)[0] == 2 + 1 + 1 + 1


# ── 價格（§5.4 / §10）：Md2500,P2250→D=-10%→5；P2700→D=8%→3 ──
def test_price_diff_and_score():
    assert ts.price_diff_pct(2250, 2500) == pytest.approx(-10.0)
    assert ts.price_score_from_D(-10.0) == 5
    assert ts.price_diff_pct(2700, 2500) == pytest.approx(8.0)
    assert ts.price_score_from_D(8.0) == 3


@pytest.mark.parametrize("D,exp", [(-15, 5), (-10, 5), (-5, 4), (0, 4),
                                   (5, 3), (10, 3), (20, 2), (25, 2),
                                   (30, 1), (None, None)])
def test_price_bands(D, exp):
    assert ts.price_score_from_D(D) == exp


def test_price_comparison_group():
    # 同區同房型人數±1 的中位數；樣本<10 放寬人數
    market = pd.DataFrame({
        "neighbourhood_cleansed": ["中山區"] * 12 + ["大安區"] * 3,
        "room_type_zh": ["私人套房"] * 15,
        "accommodates": [2] * 12 + [2] * 3,
        "price": list(range(2000, 2000 + 100 * 12, 100)) + [9000, 9000, 9000],
    })
    listing = {"neighbourhood_cleansed": "中山區",
               "room_type_zh": "私人套房", "accommodates": 2}
    median, n, relaxed = ts.price_comparison(listing, market)
    assert n == 12  # 只比中山區同房型
    assert median == pytest.approx(np.median(list(range(2000, 3200, 100))))


# ── 口碑（§6.6 / §10）──
def test_reputation_rating_bands():
    assert ts.reputation_from_rating(4.9) == 3.0
    assert ts.reputation_from_rating(4.7) == 2.5
    assert ts.reputation_from_rating(4.3) == 2.0
    assert ts.reputation_from_rating(4.05) == 1.0
    assert ts.reputation_from_rating(3.9) == 0.0
    assert ts.reputation_from_rating(None) is None


def test_nlp_content_score():
    # §6.3 例子：14 正、4 中、2 負 / 20 → N=1.60
    assert ts.nlp_content_score(14 / 20, 2 / 20) == pytest.approx(1.60)
    # §6.4：A 60%/40%中/0%負 → 1.60；B 60%/0%/40%負 → 1.20
    assert ts.nlp_content_score(0.60, 0.0) == pytest.approx(1.60)
    assert ts.nlp_content_score(0.60, 0.40) == pytest.approx(1.20)


@pytest.mark.parametrize("n,exp", [(0, 0), (1, 3), (4, 3), (5, 4),
                                   (19, 4), (20, 5), (85, 5)])
def test_review_cap(n, exp):
    assert ts.review_cap(n) == exp


def test_reputation_full_example_page10():
    # rating4.7→A2.5；14正2負/20→N1.6；總85則→cap5；R=min(4.1,5)=4.1
    R, d = ts.reputation_score(rating=4.7, pos_n=14, neg_n=2,
                               n_analyzable=20, n_total=85)
    assert d["A"] == 2.5
    assert d["N"] == pytest.approx(1.6)
    assert d["cap"] == 5
    assert R == pytest.approx(4.1)


def test_reputation_examples_page6():
    # 評論很多但很差：A=0, N=0.6, cap5 → 0.6
    R, _ = ts.reputation_score(3.5, pos_n=8, neg_n=12, n_analyzable=20, n_total=40)
    # N=clamp(1+0.4-0.6,0,2)=0.8 → 這裡驗證 min 行為，用可控輸入
    R2, d2 = ts.reputation_score(3.5, pos_n=0, neg_n=8, n_analyzable=20, n_total=40)
    assert d2["A"] == 0.0
    assert d2["N"] == pytest.approx(0.6)  # 1+0-0.4=0.6
    assert R2 == pytest.approx(0.6)
    # 只有1則滿分好評：A=3,N=2,cap3 → 3
    R3, d3 = ts.reputation_score(4.9, pos_n=1, neg_n=0, n_analyzable=1, n_total=1)
    assert d3["A"] == 3.0 and d3["N"] == pytest.approx(2.0) and d3["cap"] == 3
    assert R3 == pytest.approx(3.0)


def test_reputation_insufficient():
    # 無評分 → 資料不足（None）
    R, d = ts.reputation_score(None, 5, 0, 10, 30)
    assert R is None and d["insufficient"] == "無評分"
    # 無評論 → 資料不足
    R2, d2 = ts.reputation_score(4.9, 0, 0, 0, 0)
    assert R2 is None and d2["insufficient"] == "無評論證據"


# ── 設備（§7.2）：選5項符合4項 → 4 ──
def test_amenity_score_pdf_example():
    E, d = ts.amenity_score([True, True, True, True, False])
    assert d["m"] == 4 and d["k"] == 5
    assert E == pytest.approx(4.0)


def test_amenity_score_scales_to_5():
    # 只選2項全中 → 5 分（同樣換算到 5 分尺度）
    assert ts.amenity_score([True, True])[0] == pytest.approx(5.0)
    assert ts.amenity_score([])[0] is None


def test_match_and_must_have():
    amen = '["Wifi", "Air conditioning", "Elevator", "Kitchen"]'
    flags = ts.match_amenities(amen, ["Wi-Fi", "冷氣", "電梯", "陽台"])
    assert flags == [True, True, True, False]
    assert ts.has_all_amenities(amen, ["Wi-Fi", "電梯"]) is True
    assert ts.has_all_amenities(amen, ["陽台"]) is False


# ── 綜合例子（§10）：S=4+4+3+4.1+4=19.1，band「值得考慮」 ──
def test_total_and_band_page10():
    scores = {"transit": 4.0, "life": 4.0, "price": 3.0,
              "reputation": 4.1, "amenity": 4.0}
    S, band = ts.total_and_band(scores)
    assert S == pytest.approx(19.1)
    assert band == "值得考慮"


@pytest.mark.parametrize("S_scores,band", [
    ({"a": 22}, "優先查看"),
    ({"a": 20}, "優先查看"),
    ({"a": 15}, "值得考慮"),
    ({"a": 12}, "普通"),
    ({"a": 9}, "建議多比較"),
])
def test_band_boundaries(S_scores, band):
    assert ts.total_and_band(S_scores)[1] == band


def test_total_counts_insufficient_as_zero():
    scores = {"transit": 5.0, "life": 5.0, "price": None,
              "reputation": None, "amenity": 5.0}
    S, _ = ts.total_and_band(scores)
    assert S == pytest.approx(15.0)


# ── 排序（§8.2）：B,A,C（優先分同為9時比總分）──
def test_ranking_pdf_example():
    top2 = ["transit", "price"]
    records = [
        {"name": "A", "scores": {"transit": 5, "price": 4, "life": 3,
                                 "reputation": 3, "amenity": 3}, "total": 18,
         "n_reviews": 10, "price": 2000},
        {"name": "B", "scores": {"transit": 4, "price": 5, "life": 4,
                                 "reputation": 4, "amenity": 3}, "total": 20,
         "n_reviews": 10, "price": 2000},
        {"name": "C", "scores": {"transit": 5, "price": 3, "life": 5,
                                 "reputation": 5, "amenity": 4}, "total": 22,
         "n_reviews": 10, "price": 2000},
    ]
    ranked = ts.rank_records(records, top2)
    assert [r["name"] for r in ranked] == ["B", "A", "C"]
    # 優先分：A、B 皆 9，C 為 8
    assert ts.priority_score(records[0]["scores"], top2) == 9
    assert ts.priority_score(records[1]["scores"], top2) == 9
    assert ts.priority_score(records[2]["scores"], top2) == 8
