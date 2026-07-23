# -*- coding: utf-8 -*-
"""risk_cockpit_sections 純邏輯(不依賴 Streamlit runtime)。"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules import risk_cockpit_sections as rc


def _hosts():
    return pd.DataFrame({
        "host_id": [12345, 12399, 88, 700123],
        "房源數": [8, 3, 1, 5],
        "高風險間數": [6, 2, 1, 1],
        "高風險占比": [0.75, 0.66, 1.0, 0.2],
        "平均風險分數": [0.71, 0.60, 0.9, 0.3],
        "預估年營收": [1.24e7, 3.1e6, 8e5, 6e6],
    })


def _listings():
    return pd.DataFrame({
        "id": [1, 2, 3, 4],
        "host_id": [12345, 12345, 99, 99],
        "tier": ["red", "yellow", "red", "green"],
        "prob": [0.82, 0.40, 0.91, 0.10],
    })


def test_resolve_哨兵與非法回傳None():
    ids = [12345, 99]
    assert rc.resolve_host_filter(rc.HOST_ALL, ids) is None
    assert rc.resolve_host_filter(None, ids) is None
    assert rc.resolve_host_filter("abc", ids) is None
    assert rc.resolve_host_filter(777, ids) is None       # 不在母體
    assert rc.resolve_host_filter(12345, ids) == 12345
    assert rc.resolve_host_filter("99", ids) == 99         # 字串數字可轉


def test_search_hosts_模糊子字串與空查詢():
    h = _hosts()
    assert len(rc.search_hosts(h, "")) == 4                # 空=全部
    got = rc.search_hosts(h, "123")["host_id"].tolist()
    assert 12345 in got and 12399 in got and 700123 in got and 88 not in got
    assert rc.search_hosts(h, "123", limit=2).shape[0] == 2  # 上限


def test_filter_listings_房東鎖定與層級與區間():
    d = _listings()
    # 不鎖房東、只要 red、全區間 → id 1,3(prob 降序)
    r = rc.filter_listings(d, ["red"], 0.0, 1.0, None)
    assert r["id"].tolist() == [3, 1]
    # 鎖房東 12345、red+yellow → id 1,2
    r2 = rc.filter_listings(d, ["red", "yellow"], 0.0, 1.0, 12345)
    assert set(r2["id"]) == {1, 2}
    # 風險區間收斂
    r3 = rc.filter_listings(d, ["red", "yellow", "green"], 0.0, 0.5, None)
    assert set(r3["id"]) == {2, 4}
    # 空 tiers 視同全部
    assert len(rc.filter_listings(d, [], 0.0, 1.0, None)) == 4
