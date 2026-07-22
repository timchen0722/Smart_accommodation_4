# Listing Risk Results Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the duplicated quadrant statistics/table with clickable classification cards, deduplicated analysis summaries, and category-specific listing details while applying the approved classification copy everywhere.

**Architecture:** Keep the stable machine keys `alarm`, `hidden`, `discount`, `healthy`, and `unknown`, and make `modules/quadrant.py` the only source for presentation copy, priority, deduplication, counts, and partition validation. Put the Streamlit rendering in a focused `modules/quadrant_sections.py` module so the already-large host portal only supplies its deduplicated host slice and model-column names.

**Tech Stack:** Python 3, pandas, NumPy, Streamlit, Streamlit AppTest, pytest, existing `modules.ui_components` design tokens.

## Global Constraints

- Preserve the current warm neutral palette, rounded white cards, five-column desktop layout, and responsive behavior.
- Card content is limited to classification icon, classification name, and deduplicated listing count.
- Display `discount` as「預測風險與近期訂房不符」and `unknown` as「其他」everywhere.
- Keep classification priority exactly: 真警報 > 隱形危機 > 預測風險與近期訂房不符 > 健康 > 其他.
- Deduplicate by caller-supplied ID, otherwise `listing_id`, otherwise `id`; never use the DataFrame index as a listing identity.
- Card count sum, deduplicated total, and classification-detail total must be equal.
- Preserve existing uncommitted host-portal changes and do not include unrelated files in feature commits.
- Do not add a new dependency or a new external API.

---

### Task 1: Central classification metadata and unique-listing partition

**Files:**
- Create: `tests/test_quadrant.py`
- Modify: `modules/quadrant.py`

**Interfaces:**
- Consumes: pandas DataFrames containing `quadrant`/`quadrant_priority` and either `listing_id` or `id`.
- Produces: `QUADRANTS`, `QUADRANT_ORDER`, `deduplicate_listings(df, id_col=None)`, `category_counts(df, id_col=None)`, `analysis_summary(df)`, and `validate_partition(df, id_col=None)`.

- [ ] **Step 1: Write failing tests for approved copy, fixed order, priority deduplication, and count conservation**

```python
import numpy as np
import pandas as pd
import pytest

from modules import quadrant as q


def test_approved_display_copy_and_order():
    assert q.QUADRANT_ORDER == ["alarm", "hidden", "discount", "healthy", "unknown"]
    assert q.QUADRANTS["discount"]["name"] == "預測風險與近期訂房不符"
    assert q.QUADRANTS["discount"]["desc"] == (
        "模型預測結果與近期實際訂房表現存在落差，可能受到近期促銷、價格調整、"
        "臨時訂單或市場變化影響。"
    )
    assert q.QUADRANTS["discount"]["action"] == (
        "檢查近期訂單、價格變動、促銷活動及入住率，確認模型預測與實際營運狀況"
        "產生差異的原因。"
    )
    assert q.QUADRANTS["unknown"]["name"] == "其他"
    assert q.QUADRANTS["unknown"]["desc"] == (
        "因資料日期範圍、欄位完整性或分析條件不足，目前暫時無法歸入主要風險分類。"
    )
    assert q.QUADRANTS["unknown"]["action"] == (
        "補充或確認資料後重新分析，暫不納入主要風險判斷。"
    )


def test_classification_boundaries_stay_compatible():
    assert q.classify_row("red", 0.10) == "alarm"
    assert q.classify_row("green", 0.10) == "hidden"
    assert q.classify_row("yellow", 0.50) == "discount"
    assert q.classify_row("green", 0.50) == "healthy"
    assert q.classify_row("green", np.nan) == "unknown"


def test_duplicate_listing_keeps_highest_priority_category():
    raw = pd.DataFrame({
        "id": [101, 101, 102, 103],
        "quadrant": ["healthy", "alarm", "discount", "unknown"],
        "quadrant_priority": [4, 1, 3, 5],
    })
    unique = q.deduplicate_listings(raw)
    assert unique["id"].tolist() == [101, 102, 103]
    assert unique.set_index("id").loc[101, "quadrant"] == "alarm"


def test_counts_are_fixed_order_and_conserve_unique_listings():
    raw = pd.DataFrame({
        "id": [1, 1, 2, 3],
        "quadrant": ["healthy", "alarm", "discount", "unknown"],
        "quadrant_priority": [4, 1, 3, 5],
    })
    counts = q.category_counts(raw)
    assert counts["quadrant"].tolist() == q.QUADRANT_ORDER
    assert counts["房源數"].sum() == 3
    assert q.validate_partition(raw) == 3


def test_missing_listing_identifier_fails_loudly():
    with pytest.raises(KeyError, match="listing_id.*id"):
        q.deduplicate_listings(pd.DataFrame({"quadrant": ["healthy"]}))
```

- [ ] **Step 2: Run the tests and confirm they fail before implementation**

Run: `py -m pytest tests/test_quadrant.py -q`

Expected: FAIL because `QUADRANT_ORDER`, `deduplicate_listings`, `category_counts`, and `validate_partition` do not exist and the old labels remain.

- [ ] **Step 3: Replace display metadata and add deterministic partition helpers**

Replace the existing `QUADRANTS` dictionary entirely so deprecated wording is not left in runtime source. Every entry has `icon`, `name`, `label`, `color`, `priority`, `analysis`, `reason`, `desc`, and `action`:

```python
QUADRANT_ORDER = ["alarm", "hidden", "discount", "healthy", "unknown"]

QUADRANTS = {
    "alarm": {
        "icon": "🚨", "name": "真警報", "label": "🚨 真警報",
        "color": "high", "priority": 1,
        "analysis": "模型風險與近期空檔同步示警。",
        "reason": "模型體質評估偏弱，且未來 90 天已訂率低於 20%。",
        "desc": "體質差且檔期空——模型與近期檔期雙雙示警，最高處理優先序。",
        "action": "立即檢視定價與 LIME 痛點，同步啟動空檔促銷。",
    },
    "hidden": {
        "icon": "👻", "name": "隱形危機", "label": "👻 隱形危機",
        "color": "medium", "priority": 2,
        "analysis": "模型評估良好，但近期檔期表現偏弱。",
        "reason": "模型未顯示高風險，但未來 90 天已訂率低於 20%。",
        "desc": "體質評估良好，但未來檔期幾乎沒有訂單——模型未涵蓋的近期問題。",
        "action": "優先查近期變動：競品降價、季節性淡季、日曆設定或照片失效。",
    },
    "discount": {
        "icon": "⚠️", "name": "預測風險與近期訂房不符",
        "label": "⚠️ 預測風險與近期訂房不符", "color": "accent", "priority": 3,
        "analysis": "模型預測與近期訂房表現出現落差。",
        "reason": "模型風險等級偏高，但近期檔期未呈現同等程度的空置訊號。",
        "desc": "模型預測結果與近期實際訂房表現存在落差，可能受到近期促銷、價格調整、臨時訂單或市場變化影響。",
        "action": "檢查近期訂單、價格變動、促銷活動及入住率，確認模型預測與實際營運狀況產生差異的原因。",
    },
    "healthy": {
        "icon": "✅", "name": "健康", "label": "✅ 健康",
        "color": "low", "priority": 4,
        "analysis": "模型評估與近期訂房表現均處於良好狀態。",
        "reason": "模型風險等級較低，且近期檔期未出現嚴重空置訊號。",
        "desc": "體質與近期檔期皆良好。",
        "action": "維持現狀，持續觀察同商圈行情。",
    },
    "unknown": {
        "icon": "❔", "name": "其他", "label": "❔ 其他",
        "color": "muted", "priority": 5,
        "analysis": "現有資料尚不足以完成主要風險分類。",
        "reason": "缺少可用的近期檔期資料，或資料未通過主要分類所需的完整性條件。",
        "desc": "因資料日期範圍、欄位完整性或分析條件不足，目前暫時無法歸入主要風險分類。",
        "action": "補充或確認資料後重新分析，暫不納入主要風險判斷。",
    },
}
```

Implement the shared partition helpers:

```python
def _resolve_id_col(df: pd.DataFrame, id_col: str | None = None) -> str:
    if id_col is not None:
        if id_col not in df.columns:
            raise KeyError(f"找不到唯一識別欄位: {id_col}")
        return id_col
    for candidate in ("listing_id", "id"):
        if candidate in df.columns:
            return candidate
    raise KeyError("找不到唯一識別欄位，需提供 listing_id 或 id")


def deduplicate_listings(df: pd.DataFrame, id_col: str | None = None) -> pd.DataFrame:
    if df.empty:
        return df.copy().reset_index(drop=True)
    key = _resolve_id_col(df, id_col)
    d = df.copy()
    if "quadrant_priority" not in d.columns:
        d["quadrant_priority"] = d["quadrant"].map(
            lambda value: QUADRANTS[value]["priority"]
        )
    return (d.sort_values(["quadrant_priority", key], kind="stable")
             .drop_duplicates(subset=[key], keep="first")
             .reset_index(drop=True))


def category_counts(df: pd.DataFrame, id_col: str | None = None) -> pd.DataFrame:
    d = deduplicate_listings(df, id_col=id_col)
    observed = d["quadrant"].value_counts().to_dict()
    return pd.DataFrame([
        {
            "quadrant": key,
            "圖示": QUADRANTS[key]["icon"],
            "分類名稱": QUADRANTS[key]["name"],
            "房源數": int(observed.get(key, 0)),
            "優先序": QUADRANTS[key]["priority"],
        }
        for key in QUADRANT_ORDER
    ])


def analysis_summary(df: pd.DataFrame) -> pd.DataFrame:
    present = set(df["quadrant"].dropna())
    rows = [{
        "quadrant": key,
        "分析判斷": QUADRANTS[key]["analysis"],
        "主要原因": QUADRANTS[key]["reason"],
        "建議行動": QUADRANTS[key]["action"],
    } for key in QUADRANT_ORDER if key in present]
    return pd.DataFrame(rows).drop_duplicates(
        subset=["分析判斷", "主要原因", "建議行動"]
    )


def validate_partition(df: pd.DataFrame, id_col: str | None = None) -> int:
    d = deduplicate_listings(df, id_col=id_col)
    key = _resolve_id_col(d, id_col)
    counts = category_counts(d, id_col=key)
    total = int(d[key].nunique())
    detail_total = sum(int((d["quadrant"] == q).sum()) for q in QUADRANT_ORDER)
    if int(counts["房源數"].sum()) != total or detail_total != total:
        raise ValueError("分類卡片、房源總數與分類明細數量不一致")
    return total
```

Update `annotate()` to call `deduplicate_listings()` whenever an ID column exists, and update `attach_calendar()` to execute `cal.drop_duplicates("listing_id", keep="first")` before merging.

- [ ] **Step 4: Run focused tests and confirm they pass**

Run: `py -m pytest tests/test_quadrant.py -q`

Expected: `5 passed`.

- [ ] **Step 5: Commit the central data behavior**

```powershell
git add modules/quadrant.py tests/test_quadrant.py
git commit -m "feat: centralize unique risk classifications"
```

---

### Task 2: Clickable cards, deduplicated summaries, and listing-detail dialog

**Files:**
- Create: `modules/quadrant_sections.py`
- Create: `tests/test_quadrant_sections.py`
- Modify: `pages/1_🏠_房東入口.py:92-94`
- Modify: `pages/1_🏠_房東入口.py:170-231`

**Interfaces:**
- Consumes: `QD.deduplicate_listings()`, `QD.category_counts()`, `QD.analysis_summary()`, `QD.validate_partition()`, `QUADRANTS`, `QUADRANT_ORDER`, and a host-scoped DataFrame.
- Produces: `render_quadrant_results(df, prob_col, tier_col, tier_labels)` and `_listing_detail_payload(row, prob_col, tier_col, tier_labels)`.

- [ ] **Step 1: Write failing pure-data tests for listing detail presentation**

```python
import numpy as np
import pandas as pd

from modules.quadrant_sections import _listing_detail_payload


TIER_LABELS = {
    "red": ("🔴 高風險", "#C4645A"),
    "yellow": ("🟡 觀察", "#C49A4A"),
    "green": ("🟢 安全", "#5B9E73"),
}


def test_listing_detail_payload_formats_missing_booking_rate():
    row = pd.Series({
        "id": 88,
        "name": "測試房源",
        "neighbourhood_cleansed": "中山區",
        "prob": 0.712,
        "tier": "red",
        "booked_rate_d90": np.nan,
        "quadrant": "unknown",
    })
    payload = _listing_detail_payload(row, "prob", "tier", TIER_LABELS)
    assert payload["房源名稱"] == "測試房源"
    assert payload["模型風險"] == "🔴 高風險 71%"
    assert payload["90 天訂房率"] == "—"
    assert payload["主要原因"].startswith("缺少可用的近期檔期資料")
```

- [ ] **Step 2: Run the test and confirm the rendering module is absent**

Run: `py -m pytest tests/test_quadrant_sections.py -q`

Expected: FAIL with `ModuleNotFoundError: modules.quadrant_sections`.

- [ ] **Step 3: Create the focused Streamlit renderer**

Implement `_listing_detail_payload()` with a fallback name of `房源 #<id>`, percentage formatting, and metadata from `QUADRANTS[row["quadrant"]]`.

Implement `render_quadrant_results()` with this state flow:

```python
def render_quadrant_results(df, prob_col: str, tier_col: str, tier_labels: dict) -> None:
    unique = QD.deduplicate_listings(df, id_col="id")
    QD.validate_partition(unique, id_col="id")
    counts = QD.category_counts(unique, id_col="id")
    selected = st.session_state.get("quadrant_selected", "all")

    with st.container(key="quadrant-card-zone"):
        card_columns = st.columns(len(QD.QUADRANT_ORDER))
        for column, row in zip(card_columns, counts.to_dict("records")):
            key = row["quadrant"]
            with column:
                if st.button(
                    f"{row['圖示']}  {row['分類名稱']}\n\n{row['房源數']} 間",
                    key=f"quadrant_card_{key}",
                    use_container_width=True,
                    type="primary" if selected == key else "secondary",
                ):
                    st.session_state["quadrant_selected"] = key
                    st.rerun()

    if selected != "all" and st.button("← 顯示全部分析", key="quadrant_show_all"):
        st.session_state["quadrant_selected"] = "all"
        st.rerun()

    if selected == "all":
        _render_analysis_rows(QD.analysis_summary(unique), unique)
    else:
        selected_rows = unique[unique["quadrant"] == selected].copy()
        _render_selected_analysis(selected)
        _render_listing_rows(selected_rows, prob_col, tier_col, tier_labels)
```

Scope CSS under `.st-key-quadrant-card-zone` so it does not affect unrelated buttons. Use existing tokens `P['surface']`, `P['border']`, `P['ink']`, `P['muted']`, and `P['primary']`; use `min-height:128px`, `border-radius:14px`, visible `:focus-visible`, and a `@media(max-width:900px)` rule that permits natural wrapping.

Render default analysis rows with visible columns `分析判斷`, `主要原因`, and `建議行動`; keep `quadrant` internal and use it only to key each `查看對應房源` button. Do not render classification name or count in this lower summary.

Render selected listing rows with the six approved columns. Each row's `查看房源明細` button opens a `@st.dialog("房源風險明細", width="large")` containing the exact payload plus the selected category's full description and action.

- [ ] **Step 4: Replace the host-page quadrant table/radio/legacy cards with the renderer**

After `PREDS = QD.annotate(...)`, enforce the common source:

```python
PREDS = QD.deduplicate_listings(PREDS, id_col="id")
```

After selecting the host and after re-annotating for XGBoost, re-create `MY` from `PREDS` and call:

```python
from modules.quadrant_sections import render_quadrant_results

sec("體質（模型）× 檔期（近期訂房）分析")
mb("模型結果用於長期風險排序；近期訂房用於判斷當前營運狀態。")
render_quadrant_results(MY, PROB_COL, TIER_COL, TIER_ZH)
```

Remove `_qs`, the duplicated HTML summary table, the radio filter, the old passive listing-card loop, and the now-unused `risk_ring()`/`trend_arrow()` helpers. Keep the five top operational KPI metrics unchanged.

- [ ] **Step 5: Run data and renderer tests**

Run: `py -m pytest tests/test_quadrant.py tests/test_quadrant_sections.py -q`

Expected: all tests pass.

- [ ] **Step 6: Commit the interactive page**

```powershell
git add modules/quadrant_sections.py tests/test_quadrant_sections.py 'pages/1_🏠_房東入口.py'
git commit -m "feat: add clickable risk result cards"
```

---

### Task 3: Propagate approved labels to reports, notifications, and documentation

**Files:**
- Modify: `modules/report_builder.py`
- Modify: `modules/notify_center.py`
- Modify: `doc/06_v6新增功能報告.md`
- Create: `tests/test_quadrant_copy.py`

**Interfaces:**
- Consumes: the central `QUADRANTS` metadata created in Task 1.
- Produces: report/notification output that never hard-codes deprecated display labels.

- [ ] **Step 1: Write a failing source-copy regression test**

```python
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_FILES = [
    ROOT / "modules" / "quadrant.py",
    ROOT / "modules" / "report_builder.py",
    ROOT / "modules" / "notify_center.py",
    ROOT / "pages" / "1_🏠_房東入口.py",
]


def test_deprecated_labels_are_absent_from_runtime_sources():
    text = "\n".join(path.read_text(encoding="utf-8") for path in RUNTIME_FILES)
    assert "靠降價撐住" not in text
    assert "檔期資料不足" not in text


def test_reports_read_quadrant_copy_from_central_mapping():
    source = (ROOT / "modules" / "report_builder.py").read_text(encoding="utf-8")
    assert 'q["label"]' in source or "q['label']" in source
    assert 'q["desc"]' in source or "q['desc']" in source
    assert 'q["action"]' in source or "q['action']" in source
```

- [ ] **Step 2: Run the source-copy test and confirm the old notification comment fails it**

Run: `py -m pytest tests/test_quadrant_copy.py -q`

Expected: FAIL because runtime source still contains a deprecated name.

- [ ] **Step 3: Remove hard-coded old labels and align report content**

Update the notification priority comment to use stable keys or the approved new display name. Keep notification predicates, sorting columns, and priority values unchanged.

In `report_builder.py`, keep using `q['label']`, `q['desc']`, and `q['action']`; add `q['reason']` to the exported assessment section so the report contains the approved analytical reason without recreating copy locally:

```python
if q:
    L += [f"### {q['label']}", "",
          f"**分析判斷**：{q['analysis']}", "",
          f"**主要原因**：{q['reason']}", "",
          f"{q['desc']}", "",
          f"**建議行動**：{q['action']}", ""]
```

Update the current v6 feature report to use the approved display names and replace the unsupported causal discount statement with the approved model-versus-recent-booking explanation. Preserve the historical counts as historical documentation.

- [ ] **Step 4: Run copy and report tests**

Run: `py -m pytest tests/test_quadrant_copy.py tests/test_quadrant.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit propagated copy**

```powershell
git add modules/report_builder.py modules/notify_center.py 'doc/06_v6新增功能報告.md' tests/test_quadrant_copy.py
git commit -m "fix: align risk category copy across outputs"
```

---

### Task 4: End-to-end count and Streamlit regression verification

**Files:**
- Create: `tests/test_host_portal_quadrants.py`
- Modify only if verification exposes a defect: `modules/quadrant.py`, `modules/quadrant_sections.py`, or `pages/1_🏠_房東入口.py`

**Interfaces:**
- Consumes: the completed central partition and host portal renderer.
- Produces: executable evidence that card counts match unique detail rows and the page loads without Streamlit exceptions.

- [ ] **Step 1: Add a real-data count conservation test**

```python
from modules import quadrant as q
from modules.data_loader import load_predictions


def test_real_prediction_partition_conserves_unique_ids():
    predictions = load_predictions()
    annotated = q.annotate(q.attach_calendar(predictions), tier_col="tier")
    unique = q.deduplicate_listings(annotated, id_col="id")
    counts = q.category_counts(unique, id_col="id")

    assert unique["id"].is_unique
    assert int(counts["房源數"].sum()) == int(unique["id"].nunique())
    for key in q.QUADRANT_ORDER:
        card_count = int(counts.loc[counts["quadrant"] == key, "房源數"].iloc[0])
        detail_count = int(unique.loc[unique["quadrant"] == key, "id"].nunique())
        assert card_count == detail_count
```

- [ ] **Step 2: Run the real-data test**

Run: `py -m pytest tests/test_host_portal_quadrants.py -q`

Expected: PASS and no duplicated `id` values.

- [ ] **Step 3: Run the complete focused test suite**

Run: `py -m pytest tests/test_quadrant.py tests/test_quadrant_sections.py tests/test_quadrant_copy.py tests/test_host_portal_quadrants.py -q`

Expected: all tests pass.

- [ ] **Step 4: Load the Streamlit page with AppTest**

Run:

```powershell
py -X utf8 -c "from streamlit.testing.v1 import AppTest; at=AppTest.from_file(r'pages/1_🏠_房東入口.py').run(timeout=120); assert not at.exception, at.exception; labels=[b.label for b in at.button]; assert any('預測風險與近期訂房不符' in x for x in labels); assert any('其他' in x for x in labels); print('AppTest OK', len(labels))"
```

Expected: `AppTest OK <number>` and no exception output.

- [ ] **Step 5: Click a classification card through AppTest and confirm detail controls appear**

Run:

```powershell
py -X utf8 -c "from streamlit.testing.v1 import AppTest; at=AppTest.from_file(r'pages/1_🏠_房東入口.py').run(timeout=120); btn=next(b for b in at.button if '預測風險與近期訂房不符' in b.label); btn.click().run(timeout=120); assert not at.exception, at.exception; assert any('查看房源明細' in b.label for b in at.button); print('Card interaction OK')"
```

Expected: `Card interaction OK`.

- [ ] **Step 6: Verify deprecated runtime labels and working-tree scope**

Run:

```powershell
rg -n "靠降價撐住|檔期資料不足" modules pages
git diff --check
git status --short
```

Expected: `rg` returns no runtime matches; `git diff --check` prints nothing; status shows only intentional feature edits plus pre-existing unrelated user files.

- [ ] **Step 7: Commit verification coverage**

```powershell
git add tests/test_host_portal_quadrants.py
git commit -m "test: verify risk card and detail counts"
```
