# -*- coding: utf-8 -*-
"""介面一致性反迴歸測試 —— 掃原始碼,防止收斂完的東西又被寫回去。

對應 `docs/superpowers/plans/2026-07-24-前端介面一致性整理.md` 的階段 3 與階段 4。
這些測試刻意掃檔案文字而不是 import 後檢查:要擋的就是「有人又貼了一份字面值」。
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TOKENS = ROOT / "modules" / "design_tokens.py"

# 已完成收斂的檔案(階段 3 + 階段 4)。階段 5 起再逐批加入。
CONVERGED = [
    ROOT / "modules" / "platform_sections.py",
    ROOT / "modules" / "risk_cockpit_sections.py",
    ROOT / "pages" / "3_📊_後台分析.py",
]

HEX = re.compile(r"#[0-9A-Fa-f]{6}\b")


def _src(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ── 階段 3:風險等級文案只有一個來源 ────────────────────────────
def test_only_design_tokens_defines_tier_wording():
    """全站不得再出現「把 red/yellow/green 對到中文」的字面 dict。

    原本這種 dict 有 6 份(1_房東入口 / notify_center / platform_sections /
    report_builder ×2 / risk_cockpit_sections),改一次文案要改六個地方。
    """
    literal = re.compile(r'["\']red["\']\s*:\s*[\(\[]?\s*["\'][^"\']*'
                         r'(高風險|觀察|安全)')
    offenders = []
    for path in list((ROOT / "modules").glob("*.py")) + \
            list((ROOT / "pages").glob("*.py")):
        if path.name == "design_tokens.py" or "-backup-" in path.name:
            continue
        if literal.search(_src(path)):
            offenders.append(path.name)
    assert not offenders, f"這些檔案又自己寫了一份等級對照表:{offenders}"


def test_tier_wording_appears_once_in_tokens():
    """三個文案字面值只能在 design_tokens 的 RISK_TIERS 出現一次。"""
    src = _src(TOKENS)
    for zh in ("高風險", "觀察", "安全"):
        assert src.count(f'"zh": "{zh}"') == 1


def test_quadrant_no_data_label_is_single_sourced():
    """象限第五類的名稱來自 STATUS_NO_DATA,不再 docstring 與 label 各寫一套。"""
    src = _src(ROOT / "modules" / "quadrant.py")
    assert "T.STATUS_NO_DATA" in src
    assert '"❔ 檔期資料不足"' not in src


# ── 階段 4:已收斂檔案不得回頭寫死樣式 ──────────────────────────
@pytest.mark.parametrize("path", CONVERGED, ids=lambda p: p.name)
def test_converged_files_have_no_hardcoded_hex(path):
    found = HEX.findall(_src(path))
    assert not found, f"{path.name} 出現硬編碼色碼:{sorted(set(found))}"


@pytest.mark.parametrize("path", CONVERGED, ids=lambda p: p.name)
def test_converged_files_use_shared_components(path):
    """已收斂的檔案要嘛不碰標題,要嘛走 ui_kit;不得再自刻 h1/區塊標題 div。"""
    src = _src(path)
    assert "from modules import ui_kit" in src or "ui_kit." in src
    assert 'class="sec"' not in src


def test_no_page_overrides_page_top_padding():
    """頁面上緣留白由 design_tokens 統一;任何頁面都不得再覆寫 padding-top。

    原本 3_📊_後台分析.py 覆寫成 3.5rem,比另外兩頁多空一截。
    """
    offenders = []
    for path in (ROOT / "pages").glob("*.py"):
        if "-backup-" in path.name:
            continue
        src = _src(path)
        if re.search(r"block-container[^}]*padding-top", src):
            offenders.append(path.name)
    assert not offenders, f"這些頁面又覆寫了 padding-top:{offenders}"


# ── 房型中譯也只留一份 ──────────────────────────────────────────
def test_room_type_translation_single_source():
    """ROOM_JP 是唯一來源;各檔不得再複製一份 {"Entire home/apt": …}。"""
    literal = re.compile(r'["\']Entire home/apt["\']\s*:\s*["\']整棟出租')
    offenders = []
    for path in list((ROOT / "modules").glob("*.py")) + \
            list((ROOT / "pages").glob("*.py")):
        if path.name in ("ui_components.py", "design_tokens.py") or \
                "-backup-" in path.name:
            continue
        if literal.search(_src(path)):
            offenders.append(path.name)
    assert not offenders, f"這些檔案自己複製了房型中譯:{offenders}"
