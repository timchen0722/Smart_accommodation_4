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
