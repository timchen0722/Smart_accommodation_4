# -*- coding: utf-8 -*-
"""design_tokens 契約測試 —— 純資料,不需要 Streamlit runtime。

重點守的兩件事:
1. token 值本身合法且不互相矛盾(色碼格式、風險等級三層齊全、別名可解析)。
2. `LEGACY_P` 與現行 `ui_components.P` **逐鍵完全相同** —— 這是階段 3 把
   `P` 指向 token 層時「零視覺變動」的保證,一旦有人改動任一邊就會紅燈。
"""
import re

import pytest

from modules import design_tokens as dt


HEX = re.compile(r"^#[0-9A-Fa-f]{6}$")


# ── 顏色 ────────────────────────────────────────────────────────
def test_color_values_are_valid_hex():
    for name, value in dt.COLOR.items():
        assert HEX.match(value), f"COLOR[{name}] = {value} 不是合法 6 碼色碼"


def test_tint_roles_cover_every_semantic_color():
    """每個語意狀態色都要有對應的淡底組合,否則 badge 會退回中性色。"""
    for role in ("danger", "warning", "success", "primary"):
        assert role in dt.TINT
        for slot in ("bg", "fg", "border"):
            assert HEX.match(dt.TINT[role][slot]), f"TINT[{role}][{slot}] 非法"


def test_no_duplicate_color_values():
    """語意不同的角色不該共用同一色碼(共用代表少了一個語意)。"""
    seen = {}
    for name, value in dt.COLOR.items():
        assert value not in seen, f"COLOR[{name}] 與 COLOR[{seen[value]}] 同色 {value}"
        seen[value] = name


# ── 字級 / 間距 / 圓角 ──────────────────────────────────────────
def test_type_scale_is_eight_steps():
    """字級刻意收斂成 8 階;要新增請先確認不是既有階級能表達的。"""
    assert len(dt.TYPE) == 8
    for name, spec in dt.TYPE.items():
        size, weight, _ls = spec
        assert size.endswith("rem"), f"TYPE[{name}] 字級應以 rem 表示"
        assert 300 <= weight <= 900


def test_radius_scale_is_small_and_ordered():
    assert set(dt.RADIUS) == {"sm", "md", "lg", "pill", "bar"}
    px = {k: int(v.rstrip("px")) for k, v in dt.RADIUS.items()}
    assert px["sm"] < px["md"] < px["lg"] < px["pill"]


def test_single_page_top_padding():
    """頁面上緣留白只能有一個值 —— 後台頁原本覆寫成 3.5rem 造成三頁不齊。"""
    assert dt.LAYOUT["page_top"] == "1.6rem"


# ── 風險等級 ────────────────────────────────────────────────────
def test_risk_tiers_complete():
    assert tuple(dt.RISK_TIERS) == dt.TIER_ORDER == ("red", "yellow", "green")
    for key, spec in dt.RISK_TIERS.items():
        assert spec["color"] in dt.COLOR
        assert spec["zh"] and spec["emoji"] and spec["rule"]


def test_tier_labels_are_the_agreed_wording():
    """2026-07-24 拍板:高風險 / 觀察 / 安全。改文案請同步更新本測試與計畫檔。"""
    assert dt.tier_label("red") == "🔴 高風險"
    assert dt.tier_label("yellow") == "🟡 觀察"
    assert dt.tier_label("green") == "🟢 安全"
    assert dt.tier_label("red", emoji=False) == "高風險"


@pytest.mark.parametrize("alias,expected", [
    ("高風險", "red"), ("🔴 高風險", "red"),
    ("中風險", "yellow"), ("中度風險", "yellow"), ("觀察", "yellow"),
    ("低風險", "green"), ("安全", "green"), ("🟢 安全", "green"),
    ("red", "red"), ("green", "green"),
])
def test_tier_alias_resolution(alias, expected):
    """舊文案必須解析得出來,否則收斂過程中既有呼叫端會顯示錯等級。"""
    assert dt.tier_key(alias) == expected


def test_tier_helpers_degrade_gracefully():
    """未知值不得丟例外 —— 渲染中途 raise 會讓整頁掛掉。"""
    assert dt.tier_key("莫名其妙") is None
    assert dt.tier_color("莫名其妙") == dt.COLOR["muted"]
    assert dt.tier_tint(None) == dt.TINT["neutral"]
    assert dt.tier_label("莫名其妙") == "莫名其妙"


def test_tier_color_matches_semantic_role():
    assert dt.tier_color("red") == dt.COLOR["danger"]
    assert dt.tier_color("yellow") == dt.COLOR["warning"]
    assert dt.tier_color("green") == dt.COLOR["success"]


# ── CSS 變數 ────────────────────────────────────────────────────
def test_css_variables_shape():
    css = dt.css_variables()
    assert css.startswith(":root{") and css.endswith("}")
    for probe in ("--sa-danger:#C4645A;", "--sa-radius-md:12px;",
                  "--sa-page-top:1.6rem;", "--sa-danger-bg:#FDECEA;"):
        assert probe in css, f"css_variables() 缺少 {probe}"


# ── 相容層(階段 3 的安全網)──────────────────────────────────────
def test_legacy_p_matches_ui_components_p_exactly():
    """LEGACY_P 必須與現行 P 逐鍵相同,階段 3 才能無痛換底。"""
    from modules.ui_components import P
    assert set(dt.LEGACY_P) == set(P), "鍵名不一致"
    diffs = {k: (dt.LEGACY_P[k], P[k]) for k in P if dt.LEGACY_P[k] != P[k]}
    assert not diffs, f"色值不一致:{diffs}"


def test_design_tokens_has_no_streamlit_dependency():
    """token 層要能被 pytest / report_builder 等非 Streamlit 情境使用。

    用 AST 檢查真正的 import 敘述,而不是掃字串 —— 註解裡提到 streamlit 不算依賴。
    """
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(dt))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "streamlit" not in imported, f"design_tokens 不該依賴 streamlit:{imported}"


# ── 分數帶(階段 7:租客入口五科成績單)──────────────────────────
def test_score_bands_are_descending_and_cover_zero():
    """門檻必須由高到低,且最後一帶為 0 —— 否則低分房源會落不到任何一帶。"""
    los = [lo for lo, _, _ in dt.SCORE_BANDS]
    assert los == sorted(los, reverse=True), f"門檻未由高到低:{los}"
    assert los[-1] == 0, "最低帶的門檻必須是 0"


def test_score_band_colors_are_valid_hex():
    for lo, name, color in dt.SCORE_BANDS:
        assert HEX.match(color), f"{name}({lo}) 色碼不合法:{color}"


def test_score_band_reuses_semantic_colors():
    """兩端與中段必須沿用語意色,不可另外調一組綠/黃/紅。"""
    by_name = {name: color for _, name, color in dt.SCORE_BANDS}
    assert by_name["優秀"] == dt.COLOR["success"]
    assert by_name["普通"] == dt.COLOR["warning"]
    assert by_name["最需比較"] == dt.COLOR["danger"]
    assert by_name["非常優秀"] == dt.TINT["success"]["fg"]


@pytest.mark.parametrize("total,expected", [
    (25, "非常優秀"), (22, "非常優秀"), (21.9, "優秀"), (18, "優秀"),
    (17, "普通"), (14, "普通"), (13, "較差"), (10, "較差"),
    (9, "最需比較"), (0, "最需比較"),
])
def test_score_band_boundaries(total, expected):
    assert dt.score_band(total)[0] == expected


def test_score_band_handles_unreadable_input():
    """None / 非數值不得丟例外,退回最低帶即可(頁面渲染不能因此中斷)。"""
    for bad in (None, "", "abc", float("nan")):
        name, color = dt.score_band(bad)
        assert name in {n for _, n, _ in dt.SCORE_BANDS}
        assert HEX.match(color)


def test_only_design_tokens_defines_score_band_colors():
    """頁面不得再自己寫一份 5 級色帶(階段 7 把它從租客入口搬進來)。"""
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    for f in list((root / "pages").glob("*.py")) + list((root / "modules").glob("*.py")):
        if f.name == "design_tokens.py":
            continue
        src = f.read_text(encoding="utf-8")
        assert "非常優秀" not in src or "SCORE_BANDS" in src, \
            f"{f.name} 疑似自帶一份分數帶文案,應改吃 design_tokens.SCORE_BANDS"
