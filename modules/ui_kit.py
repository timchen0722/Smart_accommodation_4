# -*- coding: utf-8 -*-
"""ui_kit.py — 全站共用元件庫（外觀層唯一入口）。

盤點(2026-07-24)發現同一種東西被各自實作:區塊標題 6 種寫法、統計卡 4 種、
表格 4 套、空狀態 4 種。本檔把它們收斂成 10 個元件,頁面只准呼叫這裡,
不得再自行拼 HTML 或寫死顏色/字級。

設計約定
--------
1. **零硬編碼**:所有樣式一律走 `design_tokens` 的 CSS 變數 `var(--sa-*)`;
   本檔不得出現任何 6 碼色碼(有 `test_ui_kit.py` 把關)。
2. **純函式 / 渲染層分離**:`*_html()` 只回傳字串、可純 pytest;
   同名的無後綴函式才碰 `st.*`。這樣元件契約能被測,不必開 Streamlit runtime。
3. **樣式一次注入**:`inject()` 同時注入 token 變數與元件 CSS,由
   `ui_components.inject_css()` 統一呼叫,頁面不必知道它的存在。
"""
from __future__ import annotations

import html as _html
from contextlib import contextmanager

import streamlit as st

from modules import design_tokens as T

# ═══════════════════════════════════════════════════════════════
# 元件 CSS
# ═══════════════════════════════════════════════════════════════
_COMPONENT_CSS = """
/* ── PageHeader ── */
.sa-page-header{padding:6px 0 10px;margin-bottom:var(--sa-space-3);
  border-bottom:1px solid var(--sa-border);}
.sa-page-title{margin:0;color:var(--sa-ink);
  font-size:var(--sa-text-page-title);font-weight:var(--sa-text-page-title-weight);
  letter-spacing:var(--sa-text-page-title-ls);line-height:1.25;}
.sa-page-desc{margin:var(--sa-space-1) 0 0;color:var(--sa-muted);
  font-size:var(--sa-text-page-desc);line-height:1.55;max-width:78ch;}

/* ── SectionHeader ── */
.sa-section{display:flex;align-items:center;gap:var(--sa-space-2);
  margin:var(--sa-space-6) 0 var(--sa-title-gap);}
.sa-section:first-child{margin-top:0;}
.sa-section-num{display:inline-flex;align-items:center;justify-content:center;
  width:26px;height:26px;flex:none;border-radius:var(--sa-radius-sm);
  background:var(--sa-primary);color:var(--sa-surface);
  font-size:.9rem;font-weight:800;}
.sa-section-title{color:var(--sa-ink);font-size:var(--sa-text-section);
  font-weight:var(--sa-text-section-weight);
  letter-spacing:var(--sa-text-section-ls);line-height:1.35;}
.sa-section-note{color:var(--sa-muted);font-size:var(--sa-text-label);
  font-weight:var(--sa-text-label-weight);letter-spacing:var(--sa-text-label-ls);}
.sa-section-desc{margin:calc(-1 * var(--sa-space-1)) 0 var(--sa-desc-gap);
  color:var(--sa-muted);font-size:var(--sa-text-caption);line-height:1.55;}

/* ── StatCard ── */
.sa-stat{height:112px;box-sizing:border-box;display:flex;flex-direction:column;
  justify-content:flex-start;padding:var(--sa-card-pad);
  background:var(--sa-surface);border:1px solid var(--sa-border);
  border-radius:var(--sa-radius-md);box-shadow:var(--sa-shadow-sm);}
.sa-stat-label{margin-bottom:5px;color:var(--sa-muted);
  font-size:var(--sa-text-label);font-weight:var(--sa-text-label-weight);
  letter-spacing:var(--sa-text-label-ls);}
.sa-stat-value{color:var(--sa-ink);font-size:var(--sa-text-metric);
  font-weight:var(--sa-text-metric-weight);line-height:1.25;
  font-variant-numeric:tabular-nums;}
.sa-stat-note{align-self:flex-start;margin-top:auto;padding:4px 9px;
  border-radius:var(--sa-radius-pill);background:var(--sa-neutral-bg);
  color:var(--sa-neutral-fg);font-size:var(--sa-text-label);
  font-weight:var(--sa-text-label-weight);line-height:1;}
.sa-stat-note-danger{background:var(--sa-danger-bg);color:var(--sa-danger-fg);}
.sa-stat-note-warning{background:var(--sa-warning-bg);color:var(--sa-warning-fg);}
.sa-stat-note-success{background:var(--sa-success-bg);color:var(--sa-success-fg);}
.sa-stat-note-primary{background:var(--sa-primary-bg);color:var(--sa-primary-fg);}

/* ── RiskBadge ── */
.sa-badge{display:inline-block;padding:3px 12px;
  border-radius:var(--sa-radius-pill);white-space:nowrap;
  font-size:var(--sa-text-caption);font-weight:700;letter-spacing:.04em;}
.sa-badge-danger{background:var(--sa-danger-bg);color:var(--sa-danger-fg);}
.sa-badge-warning{background:var(--sa-warning-bg);color:var(--sa-warning-fg);}
.sa-badge-success{background:var(--sa-success-bg);color:var(--sa-success-fg);}
.sa-badge-neutral{background:var(--sa-neutral-bg);color:var(--sa-neutral-fg);}

/* ── FilterBar ── */
.sa-filter-title{display:flex;align-items:center;gap:6px;
  margin:var(--sa-space-2) 0 2px;color:var(--sa-ink);
  font-size:var(--sa-text-card-title);
  font-weight:var(--sa-text-card-title-weight);}
.sa-filter-desc{margin-bottom:var(--sa-space-2);color:var(--sa-muted);
  font-size:var(--sa-text-caption);line-height:1.45;}
.st-key-sa-filter-bar{padding:var(--sa-space-3) var(--sa-space-4);
  background:var(--sa-surface);border:1px solid var(--sa-border);
  border-radius:var(--sa-radius-md);margin-bottom:var(--sa-space-3);}

/* ── DataTable ── */
.sa-table-wrap{border:1px solid var(--sa-border);
  border-radius:var(--sa-radius-md);box-shadow:var(--sa-shadow-sm);}
.sa-table{width:100%;border-collapse:collapse;}
.sa-table th{position:sticky;top:0;z-index:1;text-align:left;
  padding:8px 13px;background:var(--sa-neutral-bg);color:var(--sa-muted);
  border-bottom:2px solid var(--sa-border2);
  font-size:var(--sa-text-label);font-weight:var(--sa-text-label-weight);
  letter-spacing:var(--sa-text-label-ls);}
.sa-table td{padding:7px 13px;color:var(--sa-ink);
  border-bottom:1px solid var(--sa-border);vertical-align:top;
  font-size:var(--sa-text-caption);}
.sa-table tbody tr:last-child td{border-bottom:0;}
.sa-table-head-cell{color:var(--sa-muted);font-size:var(--sa-text-label);
  font-weight:var(--sa-text-label-weight);letter-spacing:var(--sa-text-label-ls);}

/* ── EmptyState ── */
.sa-empty{display:flex;flex-direction:column;align-items:center;
  gap:var(--sa-space-2);padding:var(--sa-space-8) var(--sa-space-4);
  background:var(--sa-card);border:1px dashed var(--sa-border2);
  border-radius:var(--sa-radius-md);text-align:center;}
.sa-empty-icon{font-size:1.6rem;line-height:1;}
.sa-empty-msg{color:var(--sa-ink2);font-size:var(--sa-text-body);
  font-weight:700;line-height:1.5;}
.sa-empty-hint{color:var(--sa-muted);font-size:var(--sa-text-caption);
  line-height:1.55;max-width:60ch;}

@media(max-width:760px){
  .sa-stat{height:auto;min-height:112px;}
  .sa-section{margin-top:var(--sa-space-4);}
}
"""


def component_css() -> str:
    """token 變數 + 元件 CSS(不含 <style> 標籤)。"""
    return T.css_variables() + _COMPONENT_CSS


def inject() -> None:
    """注入 token 變數與元件樣式。由 ui_components.inject_css() 統一呼叫。"""
    st.markdown(f"<style>{component_css()}</style>", unsafe_allow_html=True)


def _esc(v) -> str:
    return _html.escape(str(v))


# ═══════════════════════════════════════════════════════════════
# 1. PageHeader
# ═══════════════════════════════════════════════════════════════
def page_header_html(title, desc=None, icon=None) -> str:
    head = f"{_esc(icon)} {_esc(title)}" if icon else _esc(title)
    desc_html = (f'<p class="sa-page-desc">{_esc(desc)}</p>' if desc else "")
    return (f'<div class="sa-page-header">'
            f'<h1 class="sa-page-title">{head}</h1>{desc_html}</div>')


def page_header(title, desc=None, icon=None) -> None:
    """頁面主標題 + 說明。每頁最上方只准出現一次,取代各頁自刻的 h1/hr。"""
    st.markdown(page_header_html(title, desc, icon), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# 2. SectionHeader
# ═══════════════════════════════════════════════════════════════
def section_header_html(title, desc=None, number=None, note=None) -> str:
    """number 給流程型頁面的段落序號;note 是標題後的小字補充(如單位、口徑)。"""
    num_html = (f'<span class="sa-section-num">{_esc(number)}</span>'
                if number is not None else "")
    note_html = (f'<span class="sa-section-note">（{_esc(note)}）</span>'
                 if note else "")
    desc_html = (f'<div class="sa-section-desc">{_esc(desc)}</div>'
                 if desc else "")
    return (f'<div class="sa-section">{num_html}'
            f'<span class="sa-section-title">{_esc(title)}{note_html}</span>'
            f'</div>{desc_html}')


def section_header(title, desc=None, number=None, note=None) -> None:
    """區塊標題。取代 sec() / numbered_section_title() / _tab_title() /
    各頁 inline 標題 / st.markdown("#### …") 五種寫法。"""
    st.markdown(section_header_html(title, desc, number, note),
                unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# 3. StatCard
# ═══════════════════════════════════════════════════════════════
_TONES = ("danger", "warning", "success", "primary", "neutral")


def stat_card_html(label, value, note=None, tone=None) -> str:
    """固定高度的關鍵指標卡;note 是釘在卡片底部的膠囊(單位、口徑、抽成率等)。

    tone 只影響 note 膠囊的顏色,不影響數值 —— 統計卡的規則是「只顯示關鍵數字」,
    語意判斷交給 RiskBadge。
    """
    note_html = ""
    if note is not None:
        cls = "sa-stat-note"
        if tone in _TONES and tone != "neutral":
            cls += f" sa-stat-note-{tone}"
        note_html = f'<div class="{cls}">{_esc(note)}</div>'
    return (f'<div class="sa-stat">'
            f'<div class="sa-stat-label">{_esc(label)}</div>'
            f'<div class="sa-stat-value">{_esc(value)}</div>'
            f'{note_html}</div>')


def stat_card(label, value, note=None, tone=None) -> None:
    """關鍵指標卡。取代 overview_metric_card() / 舊 stat_card() / _kpi() / 裸 st.metric。"""
    st.markdown(stat_card_html(label, value, note, tone), unsafe_allow_html=True)


def stat_card_row(items, columns=None) -> None:
    """一列統計卡。items = [(label, value)] 或 [(label, value, note)] 或
    [(label, value, note, tone)];columns 未給時等分。"""
    cols = st.columns(columns or len(items))
    for col, item in zip(cols, items):
        with col:
            stat_card(*item)


# ═══════════════════════════════════════════════════════════════
# 4. RiskBadge
# ═══════════════════════════════════════════════════════════════
def risk_badge_html(tier, emoji: bool = True) -> str:
    """風險等級標籤。tier 吃 red/yellow/green,也吃舊中文(高風險/中風險/低風險…)。

    文案與顏色全部來自 design_tokens.RISK_TIERS —— 這是「同一狀態在統計卡、
    圖表、列表、詳細頁必須同名同色」的執行點。
    """
    key = T.tier_key(tier)
    role = T.RISK_TIERS[key]["color"] if key else "neutral"
    return (f'<span class="sa-badge sa-badge-{role}">'
            f'{_esc(T.tier_label(tier, emoji=emoji))}</span>')


def risk_badge(tier, emoji: bool = True) -> str:
    """回傳 HTML 字串(可內嵌進表格儲存格),與既有 ui_components.risk_badge 同型。"""
    return risk_badge_html(tier, emoji)


def risk_legend_html() -> str:
    """三層警報的判讀圖例(高風險/觀察/安全 + 各自門檻),各頁共用同一份說明。"""
    parts = []
    for key in T.TIER_ORDER:
        spec = T.RISK_TIERS[key]
        parts.append(f'<span style="color:var(--sa-{spec["color"]});'
                     f'font-weight:700;">{spec["emoji"]} {spec["zh"]}'
                     f'　{_esc(spec["rule"])}</span>')
    return ('<div class="sa-section-desc">'
            + "　·　".join(parts) + "</div>")


# ═══════════════════════════════════════════════════════════════
# 5. FilterBar
# ═══════════════════════════════════════════════════════════════
def filter_group(title, desc=None, icon=None) -> None:
    """篩選器群組標題(側欄或主區塊皆可)。取代三頁三種寫法的篩選標題。"""
    head = f"{_esc(icon)} {_esc(title)}" if icon else _esc(title)
    st.markdown(f'<div class="sa-filter-title">{head}</div>'
                + (f'<div class="sa-filter-desc">{_esc(desc)}</div>'
                   if desc else ""),
                unsafe_allow_html=True)


@contextmanager
def filter_bar(key: str = "sa-filter-bar"):
    """主區塊用的橫向篩選列容器(白底、圓角、與卡片同一套邊框)。

    用法:
        with ui_kit.filter_bar():
            c1, c2 = st.columns(2)
            ...
    """
    with st.container(key=key):
        yield


# ═══════════════════════════════════════════════════════════════
# 6. DataTable
# ═══════════════════════════════════════════════════════════════
def data_table_html(df, fmt=None, cell_fn=None, height=360, wrap=False,
                    scroll=True, widths=None) -> str:
    """把 DataFrame 轉成統一樣式的 HTML 表格。

    參數沿用既有 html_table() 的語意,方便逐檔平移:
      fmt      {欄名: format 字串},如 {"價格": "${:,.0f}"}
      cell_fn  {欄名: v -> 額外 CSS},用於單格上色
      wrap     True 讓長文字換行(啟用 table-layout:fixed)
      scroll   False 時不套內捲動容器(彈窗/列印用)
      widths   {欄名: CSS 寬度},避免 fixed 佈局把長文字欄擠壓
    """
    import pandas as pd

    fmt = fmt or {}
    cell_fn = cell_fn or {}
    ws = "normal" if wrap else "nowrap"
    wb = "break-word" if wrap else "normal"

    hdr = "".join(f'<th style="white-space:{ws};">{_esc(c)}</th>'
                  for c in df.columns)
    rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        stripe = ("background:var(--sa-surface);" if i % 2 == 0
                  else "background:var(--sa-neutral-bg);")
        cells = []
        for col in df.columns:
            v = row[col]
            if pd.isna(v):
                disp = "–"
            elif col in fmt:
                disp = fmt[col].format(v)
            else:
                disp = str(v)
            css = f"white-space:{ws};word-break:{wb};{stripe}"
            if col in cell_fn:
                try:
                    css += cell_fn[col](v)
                except Exception:      # 單格上色失敗不該讓整張表掛掉
                    pass
            cells.append(f'<td style="{css}">{disp}</td>')
        rows.append(f'<tr>{"".join(cells)}</tr>')

    container = (f"overflow:auto;max-height:{int(height)}px;" if scroll
                 else "overflow:visible;")
    tstyle = "table-layout:fixed;" if wrap else ""
    cg = ""
    if widths:
        cg = "<colgroup>" + "".join(
            f'<col style="width:{widths[c]};">' if c in widths else "<col>"
            for c in df.columns) + "</colgroup>"
    return (f'<div class="sa-table-wrap" style="{container}">'
            f'<table class="sa-table" style="{tstyle}">{cg}'
            f'<thead><tr>{hdr}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def data_table(df, **kw) -> None:
    """靜態資料表。取代 html_table() 與 .quadrant-table 兩套樣式。"""
    st.markdown(data_table_html(df, **kw), unsafe_allow_html=True)


def table_header_row(labels, widths, gap="small"):
    """互動式表格(含 checkbox / 可點按鈕的列)的表頭。

    這類表格必須用 st.columns 手刻,無法走 data_table();但表頭樣式要一致,
    所以統一由本函式產生,並回傳欄物件供呼叫端接著放內容。
    """
    cols = st.columns(widths, gap=gap)
    for col, label in zip(cols, labels):
        col.markdown(f'<span class="sa-table-head-cell">{_esc(label)}</span>',
                     unsafe_allow_html=True)
    return cols


# ═══════════════════════════════════════════════════════════════
# 7 & 8. PrimaryButton / SecondaryButton
# ═══════════════════════════════════════════════════════════════
def primary_button(label, key=None, stretch=False, **kw):
    """主要動作(每個區塊至多一顆):送出、批次發送、產生報告。"""
    if stretch:
        kw.setdefault("width", "stretch")
    return st.button(label, key=key, type="primary", **kw)


def secondary_button(label, key=None, stretch=False, **kw):
    """次要動作:取消、清除、關閉、切換檢視。"""
    if stretch:
        kw.setdefault("width", "stretch")
    return st.button(label, key=key, type="secondary", **kw)


# ═══════════════════════════════════════════════════════════════
# 9. EmptyState
# ═══════════════════════════════════════════════════════════════
def empty_state_html(msg, hint=None, icon="🗂") -> str:
    hint_html = f'<div class="sa-empty-hint">{_esc(hint)}</div>' if hint else ""
    return (f'<div class="sa-empty">'
            f'<div class="sa-empty-icon">{_esc(icon)}</div>'
            f'<div class="sa-empty-msg">{_esc(msg)}</div>'
            f'{hint_html}</div>')


def empty_state(msg, hint=None, cmd=None, icon="🗂") -> None:
    """沒有資料可顯示時的統一畫面。取代 st.warning / st.info / st.caption 四種寫法。

    cmd 給「缺產物、需要先跑某個腳本」的情境,會附上可複製的指令。
    """
    st.markdown(empty_state_html(msg, hint, icon), unsafe_allow_html=True)
    if cmd:
        st.code(cmd, language="bash")


# ═══════════════════════════════════════════════════════════════
# 10. LoadingState
# ═══════════════════════════════════════════════════════════════
@contextmanager
def loading(msg: str = "載入中"):
    """統一的載入提示。文案一律「動詞 + …」,不要每處自己編。"""
    with st.spinner(f"{msg} …"):
        yield
