# -*- coding: utf-8 -*-
"""notify_center 公開組信介面 _advice_and_compose(不依賴 pkl / session_state)。"""
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


def test_規則引擎平台視角組信(monkeypatch, row):
    from modules import notify_center as nc
    monkeypatch.setattr(nc, "_rule_advice", lambda r: ["調降房價 5%"])
    mail, src, status = nc._advice_and_compose(
        row, 0.81, 0.60, platform_view=True, prefer_llm=False)
    assert src == "規則引擎"
    assert "調降房價 5%" in mail["body"]
    assert "Airbnb 平台營運團隊" in mail["body"]      # 平台視角署名
    assert "81%" in mail["body"]
    assert "成功" in status


def test_LLM失敗退回規則引擎不丟例外(monkeypatch, row):
    from modules import notify_center as nc

    def _boom(r, prob):
        raise RuntimeError("no api key")

    monkeypatch.setattr(nc, "_llm_advice", _boom)
    monkeypatch.setattr(nc, "_rule_advice", lambda r: ["規則後備建議"])
    # 強制走 LLM 分支(prefer_llm=True 且 prob>=0.6);llm_available 也 mock 成 True
    monkeypatch.setattr("modules.llm_advisor.llm_available", lambda: True)
    mail, src, status = nc._advice_and_compose(
        row, 0.81, 0.60, platform_view=True, prefer_llm=True)
    assert "規則後備建議" in mail["body"]
    assert "後備" in src
    assert "LLM 失敗" in status
