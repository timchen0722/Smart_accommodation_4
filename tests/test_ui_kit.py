# -*- coding: utf-8 -*-
"""ui_kit 契約測試。

分兩層:
1. 純函式(`*_html`)—— 直接驗 HTML 輸出,不需要 Streamlit runtime。
2. 渲染函式 —— 用 AppTest 無頭跑一支把 10 個元件全叫過一遍的腳本,
   證據 = 「0 例外 + 真實 markdown 內容」(沿用本專案既有的 AppTest 驗證法)。
"""
import ast
import inspect
import re

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from modules import design_tokens as T
from modules import ui_kit as K


# ═══════════════════════════════════════════════════════════════
# 零硬編碼
# ═══════════════════════════════════════════════════════════════
def test_ui_kit_contains_no_hardcoded_hex_colors():
    """元件庫只准用 var(--sa-*);出現色碼代表又繞過 token 了。"""
    src = inspect.getsource(K)
    found = re.findall(r"#[0-9A-Fa-f]{6}\b", src)
    assert not found, f"ui_kit.py 出現硬編碼色碼:{sorted(set(found))}"


def test_ui_kit_contains_no_hardcoded_font_sizes():
    """字級一律走 var(--sa-text-*)。

    階段 8 之前還留著 .sa-empty-icon(1.6rem)與序號方塊(.9rem)兩個字面值,
    現已一併改吃 token,故這裡收緊為「一個都不准有」。
    """
    src = inspect.getsource(K)
    literal = re.findall(r"font-size:\s*[0-9.]+(?:rem|px|em)", src)
    assert literal == [], f"非預期的寫死字級:{literal}"


def test_component_css_defines_every_token_it_uses():
    """CSS 裡引用的每個 var(--sa-*) 都要在 design_tokens 產出的變數表裡有定義。"""
    css = K.component_css()
    declared = set(re.findall(r"(--sa-[a-z0-9-]+)\s*:", T.css_variables()))
    used = set(re.findall(r"var\((--sa-[a-z0-9-]+)\)", css))
    missing = used - declared
    assert not missing, f"CSS 用到未定義的 token:{sorted(missing)}"


# ═══════════════════════════════════════════════════════════════
# PageHeader / SectionHeader
# ═══════════════════════════════════════════════════════════════
def test_page_header_html_structure():
    out = K.page_header_html("房東營運面板", desc="說明文字", icon="🏠")
    assert 'class="sa-page-header"' in out
    assert 'class="sa-page-title"' in out and "🏠 房東營運面板" in out
    assert 'class="sa-page-desc"' in out and "說明文字" in out


def test_page_header_omits_desc_block_when_absent():
    assert "sa-page-desc" not in K.page_header_html("只有標題")


def test_page_header_escapes_html():
    """標題來源可能是資料欄位,必須跳脫,否則會把版面打壞或注入。"""
    out = K.page_header_html("<script>alert(1)</script>")
    assert "<script>" not in out and "&lt;script&gt;" in out


def test_section_header_number_note_desc():
    out = K.section_header_html("信任成績單", desc="副標", number=1, note="90 天")
    assert 'class="sa-section-num"' in out and ">1<" in out
    assert "（90 天）" in out
    assert 'class="sa-section-desc"' in out and "副標" in out


def test_section_header_minimal_has_no_number_chip():
    out = K.section_header_html("行政區健康度")
    assert "sa-section-num" not in out and "sa-section-desc" not in out


# ═══════════════════════════════════════════════════════════════
# StatCard
# ═══════════════════════════════════════════════════════════════
def test_stat_card_html_basic():
    out = K.stat_card_html("總房源數", "12,345 間")
    assert 'class="sa-stat"' in out
    assert "總房源數" in out and "12,345 間" in out
    assert "sa-stat-note" not in out


@pytest.mark.parametrize("tone,cls", [
    ("danger", "sa-stat-note-danger"),
    ("warning", "sa-stat-note-warning"),
    ("success", "sa-stat-note-success"),
    ("primary", "sa-stat-note-primary"),
])
def test_stat_card_note_tone(tone, cls):
    out = K.stat_card_html("預估平台年收入", "$1.2 億", note="抽成 15%", tone=tone)
    assert cls in out and "抽成 15%" in out


def test_stat_card_unknown_tone_falls_back_to_neutral():
    out = K.stat_card_html("A", "1", note="n", tone="莫名其妙")
    assert 'class="sa-stat-note"' in out
    assert "sa-stat-note-" not in out.split('class="sa-stat-note"')[1]


# ═══════════════════════════════════════════════════════════════
# RiskBadge —— 同一狀態同名同色的執行點
# ═══════════════════════════════════════════════════════════════
@pytest.mark.parametrize("tier,zh,cls", [
    ("red", "🔴 高風險", "sa-badge-danger"),
    ("yellow", "🟡 觀察", "sa-badge-warning"),
    ("green", "🟢 安全", "sa-badge-success"),
])
def test_risk_badge_wording_and_color(tier, zh, cls):
    out = K.risk_badge_html(tier)
    assert zh in out and cls in out


@pytest.mark.parametrize("legacy,cls", [
    ("高風險", "sa-badge-danger"),
    ("中風險", "sa-badge-warning"),
    ("中度風險", "sa-badge-warning"),
    ("低風險", "sa-badge-success"),
])
def test_risk_badge_accepts_legacy_wording(legacy, cls):
    """既有呼叫端傳中文,收斂期間不能壞掉,而且要被正規化成新文案。"""
    out = K.risk_badge_html(legacy)
    assert cls in out
    assert T.tier_label(legacy) in out


def test_risk_badge_unknown_value_is_neutral_not_crash():
    out = K.risk_badge_html("莫名其妙")
    assert "sa-badge-neutral" in out and "莫名其妙" in out


def test_risk_badge_without_emoji():
    assert K.risk_badge_html("red", emoji=False).count("🔴") == 0


def test_risk_legend_lists_all_three_tiers_with_rules():
    """門檻文字含 `<` `≥`,必須是跳脫後的形式(否則瀏覽器會把 `< 35%` 當標籤吃掉)。"""
    import html as _h
    out = K.risk_legend_html()
    for key in T.TIER_ORDER:
        spec = T.RISK_TIERS[key]
        assert spec["zh"] in out
        assert _h.escape(spec["rule"]) in out
    assert "&lt; 35%" in out


# ═══════════════════════════════════════════════════════════════
# DataTable
# ═══════════════════════════════════════════════════════════════
@pytest.fixture
def df():
    return pd.DataFrame({"行政區": ["大安區", "信義區"],
                         "房源數": [120, 98],
                         "高風險占比": [0.312, float("nan")]})


def test_data_table_renders_all_cells(df):
    out = K.data_table_html(df)
    assert 'class="sa-table"' in out
    for text in ("行政區", "大安區", "信義區", "120", "98"):
        assert text in out


def test_data_table_nan_becomes_dash(df):
    assert "–" in K.data_table_html(df)


def test_data_table_applies_fmt(df):
    out = K.data_table_html(df, fmt={"高風險占比": "{:.1%}"})
    assert "31.2%" in out


def test_data_table_cell_fn_failure_does_not_break_table(df):
    """單格上色函式炸掉時,整張表仍要渲染出來。"""
    out = K.data_table_html(df, cell_fn={"房源數": lambda v: 1 / 0})
    assert "大安區" in out and "信義區" in out


def test_data_table_widths_emit_colgroup(df):
    out = K.data_table_html(df, widths={"行政區": "40%"})
    assert "<colgroup>" in out and "width:40%" in out


def test_data_table_scroll_toggle(df):
    assert "max-height:200px" in K.data_table_html(df, height=200, scroll=True)
    assert "overflow:visible" in K.data_table_html(df, scroll=False)


def test_data_table_escapes_column_names():
    out = K.data_table_html(pd.DataFrame({"<b>欄</b>": [1]}))
    assert "&lt;b&gt;" in out


# ═══════════════════════════════════════════════════════════════
# EmptyState
# ═══════════════════════════════════════════════════════════════
def test_empty_state_html():
    out = K.empty_state_html("目前條件下沒有房源", hint="請放寬側欄篩選")
    assert 'class="sa-empty"' in out
    assert "目前條件下沒有房源" in out and "請放寬側欄篩選" in out


def test_empty_state_without_hint():
    assert "sa-empty-hint" not in K.empty_state_html("沒有資料")


# ═══════════════════════════════════════════════════════════════
# 十個元件都在(避免計畫寫了卻漏做)
# ═══════════════════════════════════════════════════════════════
def test_all_ten_components_exist():
    for name in ("page_header", "section_header", "stat_card", "risk_badge",
                 "filter_bar", "data_table", "primary_button",
                 "secondary_button", "empty_state", "loading"):
        assert callable(getattr(K, name)), f"缺少元件 {name}"


def test_ui_kit_only_depends_on_design_tokens_for_styling():
    """ui_kit 不得 import ui_components(避免循環),樣式來源只有 design_tokens。"""
    tree = ast.parse(inspect.getsource(K))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(a.name for a in node.names)
    assert "modules.ui_components" not in modules


# ═══════════════════════════════════════════════════════════════
# AppTest:渲染層無頭實跑,證據 = 0 例外 + 真實內容
# ═══════════════════════════════════════════════════════════════
_DEMO = """
import pandas as pd
import streamlit as st
from modules import ui_kit as K

K.inject()
K.page_header("房東營運面板", desc="一頁看完該處理哪一間", icon="🏠")
K.section_header("關鍵指標", desc="只放數字", number=1)
K.stat_card_row([
    ("總房源數", "12,345 間"),
    ("高風險占比", "31.2%", "門檻 60%", "danger"),
])
st.markdown(K.risk_badge("red") + K.risk_badge("中風險") + K.risk_badge("低風險"),
            unsafe_allow_html=True)
st.markdown(K.risk_legend_html(), unsafe_allow_html=True)

K.section_header("篩選條件", number=2)
with K.filter_bar():
    K.filter_group("房源篩選", desc="不選＝全部", icon="🔍")
    st.multiselect("行政區", ["大安區", "信義區"], default=["大安區"], key="d")

K.section_header("詳細資料列表", number=3)
K.data_table(pd.DataFrame({"行政區": ["大安區"], "房源數": [120]}))
cols = K.table_header_row(["選取", "房源ID"], [0.5, 1.3])
cols[0].checkbox("選取", key="c0", label_visibility="collapsed")
cols[1].button("#123 ▸", key="b0")

K.primary_button("批次發送", key="pb")
K.secondary_button("清除選取", key="sb")
K.empty_state("目前條件下沒有房源", hint="請放寬側欄的行政區或房型篩選")
with K.loading("計算風險歸因"):
    pass
"""


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_string(_DEMO, default_timeout=60)
    at.run()
    return at


def test_demo_app_has_no_exception(app):
    assert not app.exception, [str(e) for e in app.exception]


def test_demo_app_renders_page_and_section_headers(app):
    # 用 class="…" 比對,避免把注入的 CSS 規則本身也算進去
    body = "".join(m.value for m in app.markdown)
    assert 'class="sa-page-title"' in body and "房東營運面板" in body
    assert body.count('class="sa-section-title"') == 3


def test_demo_app_renders_stat_cards_and_badges(app):
    body = "".join(m.value for m in app.markdown)
    assert body.count('class="sa-stat"') == 2
    assert "12,345 間" in body and "sa-stat-note-danger" in body
    # 三顆 badge:紅/黃(由「中風險」正規化)/綠
    assert "sa-badge-danger" in body
    assert "sa-badge-warning" in body and "🟡 觀察" in body
    assert "sa-badge-success" in body and "🟢 安全" in body


def test_demo_app_renders_table_and_empty_state(app):
    body = "".join(m.value for m in app.markdown)
    assert 'class="sa-table"' in body and "大安區" in body
    assert 'class="sa-empty"' in body and "目前條件下沒有房源" in body


def test_demo_app_button_types_are_distinguished(app):
    """主要/次要動作必須在 UI 上分得出來,否則使用者不知道該按哪顆。"""
    types = {b.label: b.proto.type for b in app.button}
    assert types["批次發送"] == "primary"
    assert types["清除選取"] == "secondary"


def test_demo_app_injects_token_variables(app):
    body = "".join(m.value for m in app.markdown)
    assert "--sa-danger:" in body and "--sa-radius-md:" in body
