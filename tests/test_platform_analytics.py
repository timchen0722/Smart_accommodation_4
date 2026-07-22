# -*- coding: utf-8 -*-
"""platform_analytics 純計算層單元測試（不讀真實資料、不需 streamlit）。"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules import platform_analytics as pa


@pytest.fixture
def sample_df():
    """6 間房源 / 3 位房東 / 2 行政區 / 2 房型的合成母體。"""
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6],
        "host_id": [10, 10, 20, 20, 30, 30],
        "neighbourhood_cleansed": ["大安區", "大安區", "大安區",
                                   "信義區", "信義區", "信義區"],
        "room_type": ["Entire home/apt", "Entire home/apt", "Private room",
                      "Entire home/apt", "Private room", "Private room"],
        "price": [1000.0, 2000.0, 1000.0, 3000.0, 1000.0, 1000.0],
        "vac_pred": [0.0, 0.5, 0.8, 0.2, 0.9, 0.9],
        "prob": [0.10, 0.40, 0.70, 0.20, 0.80, 0.90],
        "tier": ["green", "yellow", "red", "green", "red", "red"],
    })


def test_add_revenue_columns_計算年營收與平台收入(sample_df):
    out = pa.add_revenue_columns(sample_df, commission=0.15)
    # 房源 1：1000 × (1-0.0) × 365 = 365000
    assert out.loc[0, "est_annual_revenue"] == pytest.approx(365000.0)
    assert out.loc[0, "platform_revenue"] == pytest.approx(54750.0)
    # 房源 2：2000 × (1-0.5) × 365 = 365000
    assert out.loc[1, "est_annual_revenue"] == pytest.approx(365000.0)


def test_add_revenue_columns_不改動輸入(sample_df):
    pa.add_revenue_columns(sample_df, commission=0.15)
    assert "est_annual_revenue" not in sample_df.columns


def test_add_revenue_columns_缺值視為零(sample_df):
    d = sample_df.copy()
    d.loc[0, "price"] = None
    out = pa.add_revenue_columns(d, commission=0.15)
    assert out.loc[0, "est_annual_revenue"] == pytest.approx(0.0)


def test_market_kpis_基本統計(sample_df):
    k = pa.market_kpis(sample_df, commission=0.15)
    assert k["n_listings"] == 6
    assert k["n_hosts"] == 3
    assert k["avg_vacancy"] == pytest.approx((0.0+0.5+0.8+0.2+0.9+0.9) / 6)
    assert k["red_ratio"] == pytest.approx(3 / 6)
    assert k["yellow_ratio"] == pytest.approx(1 / 6)


def test_market_kpis_平台收入等於營收乘抽成(sample_df):
    k = pa.market_kpis(sample_df, commission=0.15)
    assert k["platform_revenue"] == pytest.approx(k["total_revenue"] * 0.15)


def test_market_kpis_空母體回傳零而非例外():
    empty = pd.DataFrame(columns=["id", "host_id", "neighbourhood_cleansed",
                                  "room_type", "price", "vac_pred",
                                  "prob", "tier"])
    k = pa.market_kpis(empty, commission=0.15)
    assert k["n_listings"] == 0
    assert k["n_hosts"] == 0
    assert k["avg_vacancy"] == 0.0
    assert k["red_ratio"] == 0.0
    assert k["total_revenue"] == 0.0


def test_district_health_欄位與筆數(sample_df):
    d = pa.district_health(sample_df, commission=0.15)
    assert list(d.columns) == ["行政區", "房源數", "平均空屋率", "高風險占比",
                               "預估平台收入", "空屋率vs全市"]
    assert len(d) == 2


def test_district_health_依高風險占比降冪(sample_df):
    d = pa.district_health(sample_df, commission=0.15)
    # 信義區 3 間有 2 紅(0.667) > 大安區 3 間有 1 紅(0.333)
    assert d.iloc[0]["行政區"] == "信義區"
    assert d.iloc[0]["高風險占比"] == pytest.approx(2 / 3)
    assert d.iloc[1]["高風險占比"] == pytest.approx(1 / 3)


def test_district_health_vs全市差異正負號(sample_df):
    d = pa.district_health(sample_df, commission=0.15).set_index("行政區")
    # 大安區均空屋率 (0+0.5+0.8)/3 = 0.4333;全市 0.55 → 差值為負(優於全市)
    assert d.loc["大安區", "空屋率vs全市"] < 0
    assert d.loc["信義區", "空屋率vs全市"] > 0


def test_host_risk_summary_聚合正確(sample_df):
    h = pa.host_risk_summary(sample_df, commission=0.15).set_index("host_id")
    assert len(h) == 3
    assert h.loc[30, "房源數"] == 2
    assert h.loc[30, "高風險間數"] == 2
    assert h.loc[30, "高風險占比"] == pytest.approx(1.0)
    assert h.loc[30, "平均風險分數"] == pytest.approx(0.85)
    assert h.loc[10, "高風險間數"] == 0


def test_host_risk_summary_排序把整批惡化房東排最前(sample_df):
    h = pa.host_risk_summary(sample_df, commission=0.15)
    assert int(h.iloc[0]["host_id"]) == 30
    assert int(h.iloc[-1]["host_id"]) == 10


def test_filter_scope_行政區與房型皆篩選(sample_df):
    out = pa.filter_scope(sample_df, ["大安區"], ["Entire home/apt"])
    assert len(out) == 2
    assert set(out["id"]) == {1, 2}


def test_filter_scope_None代表不篩選(sample_df):
    assert len(pa.filter_scope(sample_df, None, None)) == 6
    assert len(pa.filter_scope(sample_df, [], [])) == 6
    assert len(pa.filter_scope(sample_df, ["信義區"], None)) == 3


def test_supply_demand_matrix_門檻過濾(sample_df):
    # 每個 行政區x房型 組合最多 2 間,門檻 15 應全部濾掉
    out = pa.supply_demand_matrix(sample_df, min_listings=15)
    assert len(out) == 0
    assert list(out.columns) == ["行政區", "房型", "房源數", "平均空屋率",
                                 "中位價格", "機會標籤"]


def test_supply_demand_matrix_標籤分類():
    # 兩組合:A 空屋率低且房源少 → 招募缺口;B 空屋率高且房源多 → 供給飽和
    rows = []
    rows += [{"id": i, "host_id": 1, "neighbourhood_cleansed": "A區",
              "room_type": "Entire home/apt", "price": 1000.0,
              "vac_pred": 0.1, "prob": 0.1, "tier": "green"}
             for i in range(2)]
    rows += [{"id": 100 + i, "host_id": 2, "neighbourhood_cleansed": "B區",
              "room_type": "Private room", "price": 1000.0,
              "vac_pred": 0.9, "prob": 0.9, "tier": "red"}
             for i in range(8)]
    d = pd.DataFrame(rows)
    out = pa.supply_demand_matrix(d, min_listings=1).set_index("行政區")
    assert out.loc["A區", "機會標籤"] == "🟢 招募缺口"
    assert out.loc["B區", "機會標籤"] == "🔴 供給飽和"
