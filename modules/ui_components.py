"""
UI Components — 日系簡約風（和風ミニマル · 淡色系 · Noto Sans TC）

本檔是外觀層的「對外門面」:全站 13 個檔案都從這裡 import。
真正的 token 定義已搬到 `modules/design_tokens.py`,共用元件在 `modules/ui_kit.py`;
本檔只負責「舊 API 相容 + 全域 CSS 注入」,不再自己定義色盤。

新程式請直接用 `ui_kit` 的元件(page_header / section_header / stat_card /
risk_badge / data_table / empty_state …),不要再擴充本檔的舊 helper。
"""
import html as _html
import streamlit as st

from modules import design_tokens as T
from modules import ui_kit
# 讓既有 `from modules.ui_components import …` 也能拿到新元件,頁面不必改 import 來源
from modules.ui_kit import (  # noqa: F401  (re-export)
    data_table, empty_state, filter_bar, filter_group, loading, page_header,
    primary_button, risk_legend_html, secondary_button, section_header,
    stat_card_row, table_header_row,
)

# ─── Design Tokens(相容層)───────────────────────────────────────
# 值全部來自 design_tokens.LEGACY_P;
# `tests/test_design_tokens.py::test_legacy_p_matches_ui_components_p_exactly`
# 逐鍵比對,確保這次換底沒有任何色值漂移。
P = dict(T.LEGACY_P)

# Risk color map —— 文案 2026-07-24 拍板為「高風險 / 觀察 / 安全」,
# 舊鍵(中風險/低風險)一併保留,避免既有呼叫端 KeyError。
RC = {T.RISK_TIERS[k]["zh"]: T.tier_color(k) for k in T.TIER_ORDER}
RC.update({"中風險": T.tier_color("yellow"), "低風險": T.tier_color("green")})
# Room type color map
RTC = {
    "整棟出租": P["primary"], "私人套房": P["accent"],
    "共用套房": P["medium"], "飯店客房": P["low"],
}
# Room type translation
ROOM_JP = {
    "Entire home/apt": "整棟出租", "Private room": "私人套房",
    "Shared room": "共用套房", "Hotel room": "飯店客房",
}
# Feature name translation
FEAT_ZH = {
    "availability_365":              "年度可訂天數",
    "number_of_reviews":             "評論總數",
    "number_of_reviews_ltm":         "近12月評論數",
    "reviews_per_month":             "月均評論數",
    "price":                         "每晚價格",
    "calculated_host_listings_count": "房東房源數",
    "minimum_nights":                "最少入住晚數",
    "rt_Entire home/apt":            "房型：整棟",
    "rt_Shared room":                "房型：共用",
    "rt_Private room":               "房型：私人",
    "rt_Hotel room":                 "房型：飯店",
    "review_scores_rating":          "評分",
    "accommodates":                  "可住人數",
    "bedrooms":                      "臥室數",
    "beds":                          "床數",
    "bathrooms_count":               "衛浴數",
}


def inject_css():
    """Inject the Japanese minimalist CSS theme."""
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@300;400;500;700&display=swap');
    html,body,.stApp{{background:{P['bg']};color:{P['ink']};
      font-family:'Noto Sans TC',sans-serif;}}
    section[data-testid="stSidebar"]{{background:{P['surface']};
      border-right:1px solid {P['border']};}}
    [data-testid="stMetric"]{{background:{P['surface']};border:1px solid {P['border']};
      border-radius:var(--sa-radius-md);padding:16px 18px;box-shadow:0 1px 4px rgba(0,0,0,.03);
      transition:box-shadow .2s ease;}}
    [data-testid="stMetric"]:hover{{box-shadow:0 3px 12px rgba(0,0,0,.07);}}
    [data-testid="stMetricLabel"]{{color:{P['muted']} !important;
      font-size:var(--sa-text-label) !important;letter-spacing:.08em;text-transform:uppercase;}}
    [data-testid="stMetricValue"]{{color:{P['ink']} !important;
<<<<<<< HEAD
      font-size:var(--sa-text-metric) !important;font-weight:700;
=======
      font-size:1.3rem !important;font-weight:700;
>>>>>>> a7faf33a8d684411d0b07457665e38d0f6d4d906
      font-variant-numeric:tabular-nums;
      white-space:normal !important;overflow:visible !important;
      text-overflow:clip !important;line-height:1.25;}}
    /* 等寬數字:同一列的 KPI 位數對得齊,數值更新時不會左右跳動 */
    [data-testid="stMetricDelta"]{{font-variant-numeric:tabular-nums;}}
    [data-testid="stMetricLabel"],[data-testid="stMetricLabel"] *{{
      white-space:normal !important;overflow:visible !important;
      text-overflow:clip !important;}}
    [data-testid="stMetricValue"] *{{white-space:normal !important;
      overflow:visible !important;text-overflow:clip !important;}}
    .stTabs [data-baseweb="tab-list"]{{background:transparent;
      border-bottom:2px solid {P['border']};gap:0;padding:0;}}
    .stTabs [data-baseweb="tab"]{{color:{P['muted']};border-radius:0;
      padding:9px 20px;border-bottom:2px solid transparent;margin-bottom:-2px;
      font-size:var(--sa-text-card-title);font-weight:700;letter-spacing:.01em;white-space:nowrap;
      transition:all .2s ease;}}
    .stTabs [aria-selected="true"]{{color:{P['primary']} !important;
      border-bottom:2px solid {P['primary']} !important;background:transparent !important;}}
    .overview-metric{{
      height:108px;box-sizing:border-box;display:flex;flex-direction:column;
      justify-content:center;background:{P['surface']};border:1px solid {P['border']};
      border-radius:var(--sa-radius-md);padding:16px 18px;box-shadow:0 1px 4px rgba(0,0,0,.03);
    }}
    .overview-metric-label{{
      color:{P['muted']};font-size:var(--sa-text-caption);font-weight:500;letter-spacing:.08em;
      margin-bottom:5px;
    }}
    .overview-metric-value{{
      color:{P['ink']};font-size:var(--sa-text-metric);font-weight:700;line-height:1.25;
    }}
    .overview-metric-uniform{{
      height:112px;justify-content:flex-start;
    }}
    .overview-metric-uniform .overview-metric-value{{
      font-variant-numeric:tabular-nums;
    }}
    .overview-metric-note{{
      align-self:flex-start;margin-top:auto;color:{P['muted']};font-size:var(--sa-text-label);
      font-weight:700;line-height:1;padding:4px 9px;border-radius:var(--sa-radius-pill);
      background:{P['tag_bg']};
    }}
    .overview-metric-note-accent{{
      color:{P['admin']};background:var(--sa-secondary-bg);border:1px solid var(--sa-secondary-border);
    }}
    @media(max-width:760px){{
      .overview-metric-uniform{{height:112px;}}
    }}
    .listing-card-accent{{height:4px;border-radius:var(--sa-radius-bar);margin-bottom:11px;}}
    .listing-card-id{{
      color:{P['muted']};font-size:var(--sa-text-label);font-weight:600;letter-spacing:.04em;
      white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-bottom:3px;
    }}
    .listing-card-title{{
      height:47px;color:{P['ink']};font-size:var(--sa-text-card-title);font-weight:700;line-height:1.42;
      margin:0 0 13px;overflow:hidden;text-overflow:ellipsis;
      display:-webkit-box;-webkit-box-orient:vertical;-webkit-line-clamp:2;
    }}
    .listing-card-photo{{
      width:100%;height:190px;display:block;object-fit:cover;
      background:{P['tag_bg']};border-radius:var(--sa-radius-sm);border:1px solid {P['border']};
    }}
    .listing-card-photo-empty{{
      width:100%;height:190px;display:flex;align-items:center;justify-content:center;
      background:{P['tag_bg']};border:1px dashed {P['border2']};border-radius:var(--sa-radius-sm);
      color:{P['muted']};font-size:var(--sa-text-caption);
    }}
    .listing-card-meta{{
      display:grid;gap:3px;margin-top:10px;color:{P['ink2']};
      font-size:var(--sa-text-caption);line-height:1.45;
    }}
    .listing-card-meta-row{{
      display:flex;align-items:center;gap:8px;min-width:0;height:22px;
    }}
    .listing-card-meta-key{{
      flex:0 0 56px;color:{P['muted']};font-size:var(--sa-text-label);font-weight:700;
    }}
    .listing-card-meta-value{{
      min-width:0;color:{P['ink2']};overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
    }}
    .listing-card-price{{color:{P['tenant']};font-weight:800;font-variant-numeric:tabular-nums;}}
    .listing-card-risk{{
      height:290px;box-sizing:border-box;min-width:0;display:flex;flex-direction:column;
      align-items:center;justify-content:flex-start;padding:16px 13px;
      background:{P['card']};border:1px solid {P['border']};border-radius:var(--sa-radius-sm);text-align:center;
    }}
    .listing-card-risk-label{{
      color:{P['muted']};font-size:var(--sa-text-caption);font-weight:700;letter-spacing:.06em;
    }}
    .listing-card-ring{{height:113px;display:flex;align-items:center;justify-content:center;}}
    .listing-card-comparison{{
      width:100%;min-height:54px;box-sizing:border-box;display:flex;
      flex-direction:column;align-items:center;
<<<<<<< HEAD
      justify-content:center;padding:7px 10px;border-radius:var(--sa-radius-sm);
      font-size:var(--sa-text-caption);font-weight:700;line-height:1.5;
=======
      justify-content:center;padding:7px 10px;border-radius:8px;
      font-size:.76rem;font-weight:700;line-height:1.5;
>>>>>>> a7faf33a8d684411d0b07457665e38d0f6d4d906
    }}
    .listing-card-comparison-high{{background:var(--sa-danger-bg);color:var(--sa-danger-fg);}}
    .listing-card-comparison-low{{background:var(--sa-success-bg);color:var(--sa-success-fg);}}
    .listing-card-comparison-flat{{background:{P['tag_bg']};color:{P['ink2']};}}
    .listing-card-calendar{{
      width:100%;box-sizing:border-box;margin-top:9px;padding:6px 10px 7px;
<<<<<<< HEAD
      background:{P['mbg']};border-radius:var(--sa-radius-sm);
      color:{P['mtxt']};font-size:var(--sa-text-label);font-weight:600;line-height:1.4;
    }}
    .listing-card-calendar strong{{
      display:block;margin-top:1px;color:{P['mtxt']};font-size:var(--sa-text-card-title);font-weight:800;
=======
      background:{P['mbg']};border-radius:8px;
      color:{P['mtxt']};font-size:.72rem;font-weight:600;line-height:1.4;
    }}
    .listing-card-calendar strong{{
      display:block;margin-top:1px;color:{P['mtxt']};font-size:1rem;font-weight:800;
>>>>>>> a7faf33a8d684411d0b07457665e38d0f6d4d906
      font-variant-numeric:tabular-nums;
    }}
    /* 房源總表營運狀態：桌面版標籤在左、篩選選項同列靠右。 */
    .st-key-q_filter [data-testid="stRadio"]{{
      width:100%;display:flex;align-items:center;gap:24px;
    }}
    .st-key-q_filter [data-testid="stRadio"]>label{{
      flex:0 0 auto;margin:0;
    }}
    .st-key-q_filter [role="radiogroup"]{{
      min-width:0;margin-left:auto;justify-content:flex-end;flex-wrap:wrap;
    }}
    @media(max-width:760px){{
      .listing-card-photo,.listing-card-photo-empty{{height:210px;}}
      .listing-card-meta{{font-size:var(--sa-text-caption);}}
      .st-key-q_filter [data-testid="stRadio"]{{
        flex-direction:column;align-items:flex-start;gap:8px;
      }}
      .st-key-q_filter [role="radiogroup"]{{
        margin-left:0;justify-content:flex-start;
      }}
    }}
    /* ── 定價情報:左「房源」右「跨平台價格」等高雙欄 ────────────── */
    .pricing-control-label{{
      height:40px;display:flex;align-items:center;white-space:nowrap;
      color:{P['ink2']};font-size:var(--sa-text-body);font-weight:600;
    }}
    .st-key-pricing-controls [data-testid="stSlider"]{{
      transform:translateY(8px);
    }}
    h2.pricing-section-title,
    h2.numbered-section-title{{
      margin:14px 0 9px !important;color:{P['ink']} !important;
      font-size:var(--sa-text-section) !important;font-weight:800 !important;
      line-height:1.35 !important;letter-spacing:-.01em !important;
    }}
    h2.numbered-section-title .section-title-note{{
      color:{P['muted']} !important;font-size:var(--sa-text-label) !important;
      font-weight:700 !important;letter-spacing:.12em !important;
      margin-left:4px;text-transform:none;vertical-align:baseline;
    }}
    @media (min-width:761px) and (max-width:1200px){{
      .st-key-pricing-radius-control > [data-testid="stLayoutWrapper"]
        > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child{{
        flex:0 0 150px;min-width:150px;
      }}
      .st-key-pricing-radius-control > [data-testid="stLayoutWrapper"]
        > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child{{
        flex:1 1 auto;min-width:0;
      }}
    }}
    /* 左欄是固定高 HTML 區塊(--pricing-h)＋一顆按鈕;右欄是錨點列＋
       2×2 的「卡片＋該平台按鈕」。--pricing-h 依右欄實測高度回推,
       兩欄底部因此切齊,改動卡高或按鈕時需重新量測。 */
    .pricing-pane{{--pricing-h:468px;height:var(--pricing-h);box-sizing:border-box;}}
    .pricing-left{{display:flex;flex-direction:column;gap:10px;}}
    .pricing-left>div{{flex:0 0 auto;}}
    .pricing-photo{{
      flex:1 1 auto;min-height:0;width:100%;object-fit:cover;border-radius:var(--sa-radius-md);
      border:1px solid {P['border']};background:{P['tag_bg']};
    }}
    .pricing-photo-empty{{
      display:flex;align-items:center;justify-content:center;
      border-style:dashed;border-color:{P['border2']};color:{P['muted']};font-size:var(--sa-text-caption);
    }}
    .pricing-anchor{{
      display:flex;align-items:baseline;gap:9px;padding:9px 14px;
      background:{P['mbg']};border:1px solid {P['border']};border-radius:var(--sa-radius-md);
    }}
    .pricing-anchor-label{{
      font-size:var(--sa-text-label);font-weight:700;color:{P['mtxt']};letter-spacing:.06em;
    }}
    .pricing-anchor-value{{
      font-size:var(--sa-text-metric);font-weight:800;color:{P['mtxt']};
      font-variant-numeric:tabular-nums;line-height:1.1;
    }}
    .pricing-anchor-sub{{margin-left:auto;font-size:var(--sa-text-label);color:{P['muted']};}}
    /* 四張平台卡不做代表色區分,一律同一組中性配色。 */
    .pf-card{{
      display:flex;flex-direction:column;gap:5px;height:134px;box-sizing:border-box;
      padding:10px 12px 11px;
      background:{P['card']};border:1px solid {P['border']};border-radius:var(--sa-radius-md);
      border-top:3px solid {P['primary']};
    }}
    .pf-head{{
      display:flex;align-items:center;gap:6px;font-size:var(--sa-text-label);font-weight:700;
      letter-spacing:.07em;color:{P['muted']};
    }}
    .pf-dot{{width:7px;height:7px;border-radius:50%;
      background:{P['primary']};flex:none;}}
    .pf-count{{margin-left:auto;letter-spacing:0;color:{P['ink2']};}}
    .pf-value{{
      font-size:var(--sa-text-metric);font-weight:800;color:{P['ink']};line-height:1.15;
      font-variant-numeric:tabular-nums;
    }}
    .pf-unit{{margin-left:5px;font-size:var(--sa-text-label);font-weight:600;color:{P['muted']};}}
    .pf-delta{{
      align-self:flex-start;font-size:var(--sa-text-label);font-weight:700;
      padding:2px 9px;border-radius:var(--sa-radius-pill);
    }}
    .pf-delta-low{{background:var(--sa-success-bg);color:var(--sa-success-fg);}}
    .pf-delta-high{{background:var(--sa-danger-bg);color:var(--sa-danger-fg);}}
    .pf-delta-flat{{background:{P['tag_bg']};color:{P['ink2']};}}
    .pf-empty{{margin:auto;font-size:var(--sa-text-label);color:{P['muted']};}}
    .pf-card .hv-wrap{{margin-top:auto;}}
    .pf-card .hv-anchor{{font-size:var(--sa-text-label);}}
    @media(max-width:760px){{
      .pricing-pane{{height:auto;}}
      .pricing-photo{{flex:0 0 auto;height:210px;}}
      .pf-card{{height:auto;min-height:134px;}}
    }}
    /* ── 定價情報:左「房源」右「跨平台價格」等高雙欄 ────────────── */
    /* 左欄是固定高 HTML 區塊(--pricing-h)＋一顆按鈕;右欄是錨點列＋
       2×2 的「卡片＋該平台按鈕」。--pricing-h 依右欄實測高度回推,
       兩欄底部因此切齊,改動卡高或按鈕時需重新量測。 */
    .pricing-pane{{--pricing-h:468px;height:var(--pricing-h);box-sizing:border-box;}}
    .pricing-left{{display:flex;flex-direction:column;gap:10px;}}
    .pricing-left>div{{flex:0 0 auto;}}
    .pricing-photo{{
      flex:1 1 auto;min-height:0;width:100%;object-fit:cover;border-radius:12px;
      border:1px solid {P['border']};background:{P['tag_bg']};
    }}
    .pricing-photo-empty{{
      display:flex;align-items:center;justify-content:center;
      border-style:dashed;border-color:{P['border2']};color:{P['muted']};font-size:.78rem;
    }}
    .pricing-anchor{{
      display:flex;align-items:baseline;gap:9px;padding:9px 14px;
      background:{P['mbg']};border:1px solid {P['border']};border-radius:12px;
    }}
    .pricing-anchor-label{{
      font-size:.71rem;font-weight:700;color:{P['mtxt']};letter-spacing:.06em;
    }}
    .pricing-anchor-value{{
      font-size:1.42rem;font-weight:800;color:{P['mtxt']};
      font-variant-numeric:tabular-nums;line-height:1.1;
    }}
    .pricing-anchor-sub{{margin-left:auto;font-size:.71rem;color:{P['muted']};}}
    /* 四張平台卡不做代表色區分,一律同一組中性配色。 */
    .pf-card{{
      display:flex;flex-direction:column;gap:5px;height:172px;box-sizing:border-box;
      padding:10px 12px 11px;
      background:{P['card']};border:1px solid {P['border']};border-radius:12px;
      border-top:3px solid {P['primary']};
    }}
    .pf-head{{
      display:flex;align-items:center;gap:6px;font-size:.68rem;font-weight:700;
      letter-spacing:.07em;color:{P['muted']};
    }}
    .pf-dot{{width:7px;height:7px;border-radius:50%;
      background:{P['primary']};flex:none;}}
    .pf-count{{margin-left:auto;letter-spacing:0;color:{P['ink2']};}}
    .pf-value{{
      font-size:1.32rem;font-weight:800;color:{P['ink']};line-height:1.15;
      font-variant-numeric:tabular-nums;
    }}
    .pf-unit{{margin-left:5px;font-size:.66rem;font-weight:600;color:{P['muted']};}}
    .pf-delta{{
      align-self:flex-start;font-size:.69rem;font-weight:700;
      padding:2px 9px;border-radius:999px;
    }}
    .pf-delta-low{{background:#EAF5EE;color:#3D7A55;}}
    .pf-delta-high{{background:#FEF2F0;color:#A03028;}}
    .pf-delta-flat{{background:{P['tag_bg']};color:{P['ink2']};}}
    .pf-bars{{margin-top:auto;display:grid;gap:5px;}}
    .pf-bar-row{{
      display:grid;grid-template-columns:26px 1fr auto;align-items:center;gap:7px;
      font-size:.65rem;color:{P['muted']};
    }}
    .pf-bar{{
      height:5px;border-radius:3px;background:{P['tag_bg']};overflow:hidden;display:block;
    }}
    .pf-bar>i{{display:block;height:100%;border-radius:3px;}}
    .pf-bar-val{{
      color:{P['ink2']};font-weight:700;font-variant-numeric:tabular-nums;
    }}
    .pf-empty{{margin:auto;font-size:.72rem;color:{P['muted']};}}
    .pf-card .hv-anchor{{font-size:.65rem;}}
    @media(max-width:760px){{
      .pricing-pane{{height:auto;}}
      .pricing-photo{{flex:0 0 auto;height:210px;}}
      .pf-card{{height:auto;min-height:172px;}}
    }}
    .quadrant-table-wrap{{
      overflow-x:auto;margin-top:8px;background:{P['surface']};
      border:1px solid {P['border']};border-radius:var(--sa-radius-md);
      box-shadow:0 4px 18px rgba(42,42,42,.055);
    }}
    .quadrant-table{{
      width:100%;min-width:880px;border-collapse:separate;border-spacing:0;
      table-layout:fixed;
    }}
    .quadrant-table th{{
      background:{P['mbg']};color:{P['mtxt']};padding:12px 18px;
      border-bottom:1px solid var(--sa-primary-border);font-size:var(--sa-text-caption);font-weight:700;
      letter-spacing:.06em;text-align:left;
    }}
    .quadrant-table td{{
      background:{P['surface']};color:{P['ink2']};padding:15px 18px;
      border-bottom:1px solid {P['border']};font-size:var(--sa-text-card-title);
      line-height:1.65;vertical-align:middle;
    }}
    .quadrant-table tbody tr:last-child td{{border-bottom:0;}}
    .quadrant-table tbody tr td:first-child{{
      border-left:4px solid var(--quadrant-color);
    }}
    .quadrant-table tbody tr:hover td{{background:var(--sa-primary-bg);}}
    .quadrant-status{{
      display:inline-flex;align-items:center;gap:9px;white-space:nowrap;
      color:var(--quadrant-color);font-size:var(--sa-text-body);font-weight:700;
    }}
    .quadrant-status-dot{{
      width:10px;height:10px;flex:0 0 10px;border-radius:50%;
      background:var(--quadrant-color);
      box-shadow:0 0 0 4px color-mix(in srgb,var(--quadrant-color) 14%,white);
    }}
    .quadrant-count{{
      color:{P['ink']};font-size:var(--sa-text-section);font-weight:800;
      font-variant-numeric:tabular-nums;
    }}
    .quadrant-count-unit{{
      margin-left:4px;color:{P['muted']};font-size:var(--sa-text-body);font-weight:600;
    }}
    .quadrant-action{{color:{P['ink']};font-weight:500;}}
    .quadrant-action::before{{
      content:'→';margin-right:8px;color:var(--quadrant-color);font-weight:800;
    }}
    section[data-testid="stSidebar"] label{{color:{P['ink2']} !important;font-size:var(--sa-text-caption);}}
    .sec{{font-size:var(--sa-text-card-title);font-weight:800;letter-spacing:.01em;
      color:{P['ink']};margin:20px 0 6px;padding-bottom:8px;
      border-bottom:1px solid {P['border']};}}
    .mb{{display:inline-flex;align-items:center;gap:4px;
      background:{P['mbg']};border:1px solid var(--sa-primary-border);border-radius:var(--sa-radius-bar);
      padding:3px 10px;font-size:var(--sa-text-label);font-weight:600;color:{P['mtxt']};
      letter-spacing:.03em;margin-bottom:7px;}}
    .mhigh{{background:var(--sa-danger-bg);border:1px solid var(--sa-danger-border);color:var(--sa-danger-fg);}}
    .note{{background:{P['tag_bg']};border-left:3px solid {P['primary']};
      padding:9px 14px;border-radius:0 6px 6px 0;
      font-size:var(--sa-text-caption);color:{P['ink2']};margin:8px 0;}}
    hr{{border:none;border-top:1px solid {P['border']} !important;margin:14px 0;}}
    ::-webkit-scrollbar{{width:4px;}}
    ::-webkit-scrollbar-thumb{{background:{P['border2']};border-radius:var(--sa-radius-bar);}}
    .portal-card{{
      background:{P['surface']};
      border:1px solid {P['border']};
      border-radius:var(--sa-radius-lg);
      padding:36px 28px;
      text-align:center;
      transition:all .3s cubic-bezier(.4,0,.2,1);
      cursor:pointer;
      box-shadow:0 1px 4px rgba(0,0,0,.03);
    }}
    .portal-card:hover{{
      box-shadow:0 8px 30px rgba(0,0,0,.08);
      transform:translateY(-4px);
      border-color:{P['primary']};
    }}
    .portal-icon{{font-size:2.8rem;margin-bottom:12px;}}
    .portal-title{{font-size:var(--sa-text-section);font-weight:700;color:{P['ink']};margin-bottom:6px;}}
    .portal-desc{{font-size:var(--sa-text-caption);color:{P['muted']};line-height:1.5;}}
    .risk-badge{{
      display:inline-block;padding:3px 12px;border-radius:var(--sa-radius-lg);
      font-size:var(--sa-text-label);font-weight:700;letter-spacing:.04em;
    }}
    .risk-high{{background:var(--sa-danger-bg);color:{P['high']};}}
    .risk-medium{{background:var(--sa-warning-bg);color:var(--sa-warning-fg);}}
    .risk-low{{background:var(--sa-success-bg);color:var(--sa-success-fg);}}
    .sentiment-pos{{color:{P['low']};font-weight:600;}}
    .sentiment-neg{{color:{P['high']};font-weight:600;}}
    .sentiment-neu{{color:{P['muted']};font-weight:500;}}
    .stat-card{{
      background:linear-gradient(135deg,{P['surface']},{P['tag_bg']});
      border:1px solid {P['border']};border-radius:var(--sa-radius-md);
      padding:16px 20px;text-align:center;
    }}
    .stat-value{{font-size:var(--sa-text-page-title);font-weight:700;color:{P['ink']};}}
    .stat-label{{font-size:var(--sa-text-label);color:{P['muted']};letter-spacing:.06em;margin-top:4px;}}
    .rv-wrap{{position:relative;display:inline-block;cursor:help;
      color:{P['primary']};font-weight:600;
      border-bottom:1px dashed {P['primary']};}}
    .rv-wrap .rv-tip{{visibility:hidden;opacity:0;position:absolute;z-index:9999;
      left:0;top:150%;width:330px;max-height:300px;overflow-y:auto;
      background:{P['surface']};border:1px solid {P['border2']};border-radius:var(--sa-radius-sm);
      padding:10px 13px;box-shadow:0 10px 34px rgba(0,0,0,.18);
      transition:opacity .15s ease;text-align:left;white-space:normal;
      font-size:var(--sa-text-label);line-height:1.55;color:{P['ink2']};font-weight:400;}}
    .rv-wrap:hover .rv-tip{{visibility:visible;opacity:1;}}
    .rv-tip-h{{font-size:var(--sa-text-label);font-weight:700;color:{P['muted']};
      letter-spacing:.08em;text-transform:uppercase;margin-bottom:6px;}}
    .rv-item{{padding:6px 0;border-bottom:1px dashed {P['border']};}}
    .rv-item:last-child{{border-bottom:none;}}
    .hero{{position:relative;display:flex;flex-wrap:wrap;border-radius:var(--sa-radius-lg);
      overflow:hidden;border:1px solid {P['border']};
      box-shadow:0 6px 26px rgba(0,0,0,.08);margin:4px 0 8px;}}
    .hero-half{{position:relative;flex:1 1 320px;min-height:196px;
      padding:22px 30px 0;overflow:hidden;}}
    .hero-txt{{position:relative;z-index:2;}}
    .hero-l{{background:linear-gradient(180deg,{T.HERO['paper_from']},{T.HERO['paper_to']});}}
    .hero-r{{background:linear-gradient(160deg,{T.HERO['sky']} 0%,{T.HERO['leaf']} 52%,{T.HERO['sun']} 100%);}}
    .hero-tag{{display:inline-block;font-size:var(--sa-text-label);font-weight:700;
      letter-spacing:.2em;color:{P['muted']};margin-bottom:8px;}}
    .hero-r .hero-tag{{color:{T.HERO['ink']};}}
    .hero-half h2{{font-size:var(--sa-text-page-title);line-height:1.2;font-weight:800;
      color:{P['ink']};margin:0 0 8px;letter-spacing:-.5px;}}
    .hero-half p{{font-size:var(--sa-text-caption);line-height:1.6;color:{P['muted']};margin:0;
      max-width:92%;}}
    .hero-r p{{color:{T.HERO['ink2']};}}
    .hero-cta{{margin-top:14px;display:inline-block;font-size:var(--sa-text-caption);
      font-weight:700;color:{P['landlord']};}}
    .hero-r .hero-cta{{color:{T.HERO['ink3']};}}
    .hero-sky{{position:absolute;left:0;bottom:0;width:100%;height:104px;
      z-index:1;display:block;}}
    .hero-seam{{position:absolute;top:-4%;left:calc(50% - 7px);width:14px;
      height:108%;background:{P['surface']};transform:rotate(4deg);z-index:3;
      box-shadow:0 0 12px rgba(0,0,0,.06);}}
    @media(max-width:660px){{.hero-half{{flex:1 1 100%;min-height:166px;}}
      .hero-seam{{display:none;}}}}
    [data-testid="stSidebarNav"]{{display:none;}}
    .block-container,[data-testid="stMainBlockContainer"],
    [data-testid="stAppViewBlockContainer"]{{padding-top:1.6rem !important;}}
    section[data-testid="stSidebar"] [data-testid="stPageLink"] a{{
      border-radius:var(--sa-radius-sm);padding:6px 10px;font-size:var(--sa-text-body);}}
    section[data-testid="stSidebar"] [data-testid="stPageLink"] a:hover{{
      background:{P['tag_bg']};}}
    </style>
    """, unsafe_allow_html=True)
    # token 變數(--sa-*)與共用元件樣式。放在最後注入,讓 ui_kit 的規則能覆蓋
    # 上面舊 CSS 的同名選擇器;各頁只要照舊呼叫 inject_css() 就會一併拿到。
    ui_kit.inject()


# ─── UI helper functions ────────────────────────────────────────
def sec(t):
    """Section header."""
    st.markdown(f'<div class="sec">{t}</div>', unsafe_allow_html=True)


def numbered_section_title(number, title, note_text=None):
    """Render a numbered major heading; an optional parenthetical keeps caption styling."""
    main_html = f"{_html.escape(str(number))}. {_html.escape(str(title))}"
    note_html = (
        f'<span class="section-title-note">({_html.escape(str(note_text))})</span>'
        if note_text else ""
    )
    st.markdown(
        f'<h2 class="numbered-section-title">{main_html}{note_html}</h2>',
        unsafe_allow_html=True)


def mb(text, warning=False):
    """Method badge."""
    cls = "mb mhigh" if warning else "mb"
    st.markdown(f'<span class="{cls}">📐 {text}</span>', unsafe_allow_html=True)


def note(t):
    """Info note block."""
    st.markdown(f'<div class="note">{t}</div>', unsafe_allow_html=True)


def risk_badge(level, emoji: bool = False):
    """風險等級標籤(HTML 字串)。

    改為委派給 ui_kit,文案與顏色一律走 design_tokens.RISK_TIERS ——
    舊實作會把「中風險」原樣印出,與全站上線文案「觀察」對不起來。
    emoji 預設 False 以維持舊呼叫端的純文字外觀。
    """
    return ui_kit.risk_badge_html(level, emoji=emoji)


def stat_card(value, label, color=None):
    """Render a stat card."""
    c = color or P["ink"]
    st.markdown(f'''
    <div class="stat-card">
      <div class="stat-value" style="color:{c};">{value}</div>
      <div class="stat-label">{label}</div>
    </div>''', unsafe_allow_html=True)


<<<<<<< HEAD
def overview_metric_card(label, value, note_text=None, accent_note=False, value_color=None):
    """Render a fixed-height KPI card; optional note stays anchored at the bottom."""
    label_html = _html.escape(str(label))
    value_html = _html.escape(str(value))
    note_html = ""
    if note_text is not None:
        note_cls = ("overview-metric-note overview-metric-note-accent"
                    if accent_note else "overview-metric-note")
        note_html = (
            f'<div class="{note_cls}">{_html.escape(str(note_text))}</div>'
        )
    val_style = f' style="color:{value_color};"' if value_color else ''
    st.markdown(
        f'<div class="overview-metric overview-metric-uniform">'
        f'<div class="overview-metric-label">{label_html}</div>'
        f'<div class="overview-metric-value"{val_style}>{value_html}</div>'
        f'{note_html}</div>',
        unsafe_allow_html=True)


=======
>>>>>>> a7faf33a8d684411d0b07457665e38d0f6d4d906
def html_table(df_in, fmt=None, cell_fn=None, height=360, wrap=False, scroll=True,
               widths=None):
    """Render a styled HTML table.

    wrap=True   lets cell text wrap so every column fits the width
                (no bottom/horizontal scrollbar).
    scroll=False renders the full table with no inner scroll container
                (the dialog/page provides its own vertical scroll).
    widths      optional {欄名: CSS 寬度},例如 {"地址": "26%"}。wrap=True 會啟用
                table-layout:fixed,預設把寬度平均分給每一欄,長文字欄(房源名稱、
                地址)因此被擠壓;給了 widths 就改用 <colgroup> 依重要性分配。
    """
    fmt = fmt or {}
    cell_fn = cell_fn or {}
    ws = "normal" if wrap else "nowrap"
    wb = "break-word" if wrap else "normal"
<<<<<<< HEAD
    th = (f"background:{P['tag_bg']};color:{P['muted']};font-size:var(--sa-text-label);"
=======
    th = (f"background:{P['tag_bg']};color:{P['muted']};font-size:.90rem;"
>>>>>>> a7faf33a8d684411d0b07457665e38d0f6d4d906
          f"letter-spacing:.07em;text-transform:uppercase;padding:8px 13px;"
          f"border-bottom:2px solid {P['border2']};white-space:{ws};"
          f"text-align:left;position:sticky;top:0;z-index:1;")
    td0 = (f"padding:7px 13px;font-size:var(--sa-text-caption);color:{P['ink']};"
           f"border-bottom:1px solid {P['border']};white-space:{ws};"
           f"word-break:{wb};vertical-align:top;")
    hdr = "".join(f'<th style="{th}">{c}</th>' for c in df_in.columns)
    rows = []
    import pandas as pd
    for i, (_, row) in enumerate(df_in.iterrows()):
        bg = P["surface"] if i % 2 == 0 else P["tag_bg"]
        cells = []
        for col in df_in.columns:
            v = row[col]
            disp = ("–" if pd.isna(v)
                    else (fmt[col].format(v) if col in fmt and pd.notna(v)
                          else str(v)))
            css = f"background:{bg};"
            if col in cell_fn:
                try:
                    css += cell_fn[col](v)
                except Exception:
                    pass
            cells.append(f'<td style="{td0}{css}">{disp}</td>')
        rows.append(f"<tr>{''.join(cells)}</tr>")
    container = (f'overflow:auto;max-height:{height}px;' if scroll
                 else 'overflow:visible;')
    tstyle = ("width:100%;border-collapse:collapse;"
              + ("table-layout:fixed;" if wrap else ""))
    cg = ("<colgroup>" + "".join(
        f'<col style="width:{widths[c]};">' if c in widths else "<col>"
        for c in df_in.columns) + "</colgroup>") if widths else ""
    st.markdown(
        f'<div style="{container}border:1px solid {P["border"]};'
<<<<<<< HEAD
        f'border-radius:var(--sa-radius-md);box-shadow:0 1px 4px rgba(0,0,0,.03);">'
=======
        f'border-radius:12px;box-shadow:0 1px 4px rgba(0,0,0,.03);">'
>>>>>>> a7faf33a8d684411d0b07457665e38d0f6d4d906
        f'<table style="{tstyle}">{cg}'
        f'<thead><tr>{hdr}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>',
        unsafe_allow_html=True)


# ─── Plotly chart theme ─────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=P["ink2"], family="Noto Sans TC,sans-serif", size=11),
    margin=dict(l=46, r=16, t=28, b=34),
    legend=dict(
        bgcolor="rgba(255,255,255,.8)", bordercolor=P["border"],
        borderwidth=1, font=dict(color=P["ink2"]),
    ),
    xaxis=dict(
        gridcolor=P["border"], linecolor=P["border"], zeroline=False,
        tickfont=dict(color=P["muted"]),
    ),
    yaxis=dict(
        gridcolor=P["border"], linecolor=P["border"], zeroline=False,
        tickfont=dict(color=P["muted"]),
    ),
)


def apply_theme(fig, h=None, legend=True):
    """Apply the Japanese minimalist theme to a Plotly figure."""
    kw = dict(**PLOTLY_LAYOUT)
    if h:
        kw["height"] = h
    if not legend:
        kw["showlegend"] = False
    return fig.update_layout(**kw)


def review_hover_html(count, snippets, label=None):
    """
    Return an inline HTML span that reveals recent reviews on hover.
    `snippets` is a list of plain-text review strings.
    """
    label = label or f"💬 {count} 則評論"
    if not snippets:
        return (f'<span style="color:var(--sa-muted);font-size:var(--sa-text-caption);">{label}'
                f'（尚無評論內容）</span>')
    items = "".join(
        f'<div class="rv-item">{_html.escape(str(s))}</div>' for s in snippets)
    return (f'<span class="rv-wrap">{label}'
            f'<span class="rv-tip"><div class="rv-tip-h">最新評論預覽</div>'
            f'{items}</span></span>')


def sidebar_nav():
    """Custom sidebar navigation: 回首頁(index.py 首頁) + 三入口。"""
    # 回首頁 → 首頁 index.py，於「原視窗」開啟（不開新分頁）
    def _link(path, label, home=False, full=False):
        try:
            st.page_link(path, label=label, use_container_width=full)
        except Exception:
            href = "./" if home else "./" + path.split("/")[-1].split("_", 1)[-1].replace(".py", "")
            st.markdown(f'<a href="{href}" target="_self">{label}</a>',
                        unsafe_allow_html=True)

    _link("index.py", "🏯 回首頁", home=True, full=True)
    for path, label in [
        ("pages/1_🏠_房東入口.py", "🏠 房東入口"),
        ("pages/2_🔍_租客入口.py", "🔍 租客入口"),
        ("pages/3_📊_後台分析.py", "📊 後台分析"),
    ]:
        _link(path, label)
    st.divider()
