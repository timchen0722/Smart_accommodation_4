# -*- coding: utf-8 -*-
"""notify_center._compose 平台方/房東方文案分支測試。"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def row():
    return pd.Series({
        "id": 1, "host_id": 10, "host_name": "測試房東",
        "name": "大安區溫馨套房", "neighbourhood_cleansed": "大安區",
        "room_type": "Entire home/apt", "price": 1800.0,
        "vac_pred": 0.72, "prob": 0.81,
        "gap_days_30d": float("nan"), "gap_longest_30d": float("nan"),
        "gap_first_start_30d": "",
    })


def test_房東視角維持原本自稱系統(row):
    from modules.notify_center import _compose
    m = _compose(row, 0.81, 0.60, ["建議一"], "規則引擎")
    assert "系統偵測到您的房源" in m["body"]
    assert "Airbnb 平台營運團隊" not in m["body"]


def test_平台視角改為平台主動關懷署名(row):
    from modules.notify_center import _compose
    m = _compose(row, 0.81, 0.60, ["建議一"], "規則引擎", platform_view=True)
    assert "Airbnb 平台營運團隊" in m["body"]
    assert "平台營運團隊" in m["subject"] or "平台" in m["subject"]


def test_兩種視角都包含風險數據與建議(row):
    from modules.notify_center import _compose
    for pv in (False, True):
        m = _compose(row, 0.81, 0.60, ["調降房價 5%"], "規則引擎",
                     platform_view=pv)
        assert "81%" in m["body"]
        assert "調降房價 5%" in m["body"]
        assert m["to"]
