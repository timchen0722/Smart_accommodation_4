# -*- coding: utf-8 -*-
"""風險管理『雙檢視』AppTest 回歸。

無頭載入後台分析頁,驗證:
  1. 預設房東檢視:0 例外、rm_view=hosts、無任何 rm_sel_* checkbox、無批次發送鈕。
  2. 點某房東ID → rm_view=listings,出現『房東檢視』麵包屑回退鈕。
  3. 房源檢視點某房源ID → 出現該房源『產生此房源輔導通知』單筆鈕(LIME 展開)。
  4. 勾選某房源 → 出現『批次發送』鈕(底部浮動列現身)。

需 data/_predictions.csv;缺檔時該頁只渲染警告、房東搜尋框不會出現,對應測試 skip。
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest

APP = str(ROOT / "pages" / "3_📊_後台分析.py")


def _run():
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    return at


def _btn(at, pred):
    return next((b for b in at.button if pred(b)), None)


def _ready(at):
    """房東搜尋框在 = 有母體資料;否則 skip。"""
    return any(t.key == "rm_host_search" for t in at.text_input)


def test_預設房東檢視_無勾選無批次列():
    at = _run()
    assert not at.exception, at.exception
    if not _ready(at):
        pytest.skip("房東檢視未出現(可能缺 data/_predictions.csv)")
    assert at.session_state["rm_view"] == "hosts"
    assert not any(str(cb.key).startswith("rm_sel_") for cb in at.checkbox)
    assert _btn(at, lambda b: b.key == "rm_batch_send") is None


def test_點房東ID_切到房源檢視且有麵包屑():
    at = _run()
    if not _ready(at):
        pytest.skip("房東檢視未出現")
    hb = _btn(at, lambda b: str(b.key).startswith("rm_host_"))
    if hb is None:
        pytest.skip("排行榜無房東可點")
    hb.click().run()
    assert not at.exception, at.exception
    assert at.session_state["rm_view"] == "listings"
    assert _btn(at, lambda b: b.key == "rm_bc_hosts") is not None


def test_房源檢視點房源ID_展開單筆派信鈕():
    at = _run()
    if not _ready(at):
        pytest.skip("房東檢視未出現")
    hb = _btn(at, lambda b: str(b.key).startswith("rm_host_"))
    if hb is None:
        pytest.skip("排行榜無房東可點")
    hb.click().run()
    assert not at.exception, at.exception
    lb = _btn(at, lambda b: str(b.key).startswith("rm_lst_"))
    if lb is None:
        pytest.skip("該房東名下無房源列")
    lb.click().run()
    assert not at.exception, at.exception
    assert _btn(at, lambda b: str(b.key).startswith("rm_send1_")) is not None


def _checkbox(at, pred):
    return next((cb for cb in at.checkbox if pred(cb)), None)


def test_勾選房源_底部浮動批次列現身():
    at = _run()
    if not _ready(at):
        pytest.skip("房東檢視未出現")
    hb = _btn(at, lambda b: str(b.key).startswith("rm_host_"))
    if hb is None:
        pytest.skip("排行榜無房東可點")
    hb.click().run()
    cb = _checkbox(at, lambda c: str(c.key).startswith("rm_sel_"))
    if cb is None:
        pytest.skip("該房東名下無房源列")
    # 勾選前:無批次發送鈕
    assert _btn(at, lambda b: b.key == "rm_batch_send") is None
    cb.check().run()
    assert not at.exception, at.exception
    assert _btn(at, lambda b: b.key == "rm_batch_send") is not None
