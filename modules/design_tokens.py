# -*- coding: utf-8 -*-
"""design_tokens.py — 全站設計 token 的唯一來源（純資料，無 Streamlit 依賴）。

為什麼要有這一層
----------------
盤點(2026-07-24)發現 `pages/` + `modules/` 裡散落 **52 種字級、58 個色碼、15 種圓角**，
其中 `#C4645A`(高風險紅)被裸寫 10 次、`#5B9E73`(安全綠) 9 次 —— 明明 `ui_components.P`
已經有 token，各頁卻繞過去自己寫死，導致改一個顏色要改十幾個地方。
本檔把「顏色 / 字級 / 間距 / 圓角 / 陰影 / 斷點 / 風險等級文案」收斂成單一來源，
`ui_components.py` 與 `ui_kit.py` 一律從這裡取值,頁面不得再自行寫死。

刻意不 import streamlit:讓 token 可以被純 pytest 驗證,也能被 report_builder
這種產出靜態 HTML 的模組沿用。

配色沿用既有「日系簡約」色盤,**值完全沒變**,只是補上語意名稱與缺漏的
tint / on-tint 組合(原本 danger 有 #FDECEA 與 #FEF2F0 兩組底色在打架)。
"""
from __future__ import annotations

# ═══════════════════════════════════════════════════════════════
# 顏色
# ═══════════════════════════════════════════════════════════════
COLOR = {
    # 品牌 / 動作
    "primary":   "#4E7FB0",   # 主要動作、連結、選中態、房東身分
    "secondary": "#8B7BA8",   # 次要強調、平台/後台身分(舊名 accent)
    # 語意狀態 —— 跨分頁不得改義:紅=要處理
    "success":   "#5B9E73",   # 低風險/安全、正向差異、租客身分
    "warning":   "#C49A4A",   # 中度風險/觀察、需注意
    "danger":    "#C4645A",   # 高風險、要處理
    # 中性
    "ink":       "#2A2A2A",   # 主文
    "ink2":      "#505050",   # 次文
    "muted":     "#9A9490",   # 輔助文、標籤
    "bg":        "#F8F7F5",   # 頁底
    "surface":   "#FFFFFF",   # 面板
    "card":      "#FDFCFA",   # 卡片
    "border":    "#E8E4DE",   # 一般邊框
    "border2":   "#D4CFC8",   # 強調邊框、虛線
}

# 語意色的「淡底 + 深字」組合。badge / 差異標籤 / note 一律用這組,
# 不要再各自調 tint —— 原本 danger 有兩組(#FDECEA/#FEF2F0)造成同義不同色。
TINT = {
    "danger":  {"bg": "#FDECEA", "fg": "#A03028", "border": "#F0B8B4"},
    "warning": {"bg": "#FDF5E4", "fg": "#A07A20", "border": "#E3D5B0"},
    "success": {"bg": "#EAF5EE", "fg": "#3D7A55", "border": "#BFDCC9"},
    "primary": {"bg": "#EEF4FB", "fg": "#3D6B96", "border": "#C8DCF0"},
    # secondary(平台/後台身分)原本只有 ui_components 的 .overview-metric-note-accent
    # 自帶一組 #F2EDF7/#D9CDE6,補進來讓五個語意角色都有完整 tint 三件組。
    "secondary": {"bg": "#F2EDF7", "fg": "#6B5B88", "border": "#D9CDE6"},
    "neutral": {"bg": "#F2F0EC", "fg": "#505050", "border": "#E8E4DE"},
}

# 日曆「已訂」格的藍 —— primary 的淡階。刻意不用紅:紅在全站代表「要處理」,
# 留給空檔警示。經對比實測(日期文字 6.06、與空房格互比 2.08)後定案。
CAL_BOOKED_BLUE = "#8AACCD"

# 跨平台品牌識別色 —— 品牌資產,不是語意色(改這裡不會改變任何風險判讀)。
# 原本 map_view.PLATFORM_STYLE 與 platform_detail.PLATFORM 各寫一份,值相同、
# 用途相同,只有欄位名不同;統一放這裡,兩邊都從這裡取。
PLATFORM_COLOR = {
    "Airbnb":  COLOR["danger"],   # 本平台沿用主紅
    "Booking": "#2563EB",         # 藍
    "591":     "#8B5CF6",         # 紫
    "ddroom":  "#4B4B4B",         # 深灰
}

# 首頁 Hero 插畫用色 —— 不是 UI 狀態語意色,不參與 tint/badge/圖表配色。
# 放這裡只是為了不讓 ui_components 的 CSS 再裸寫一次(TYPE_EXEMPT 已把
# 「index.py 首頁 Hero」列為字級白名單,配色同理)。
HERO = {
    "sky": "#7ED6E8", "leaf": "#BFE08A", "sun": "#F7D774",   # 右半漸層
    "ink": "#256048", "ink2": "#20543E", "ink3": "#1C4A36",  # 漸層上的深綠字
    "paper_from": "#FBFAF8", "paper_to": "#EFEDE9",          # 左半米白漸層
}

# ═══════════════════════════════════════════════════════════════
# 字級(52 種 → 8 階)
# ═══════════════════════════════════════════════════════════════
# 每項 = (font-size, font-weight, letter-spacing)
TYPE = {
    "page_title": ("1.5rem",   800, "-.01em"),   # 頁面主標題
    "page_desc":  ("0.85rem",  400, "normal"),   # 頁面說明文字
    "section":    ("1.15rem",  800, ".01em"),    # 區塊標題
    "card_title": ("1rem",     700, "normal"),   # 卡片標題
    "metric":     ("1.35rem",  800, "normal"),   # 統計卡數值(配 tabular-nums)
    # 單一主角級數值(整段只放一個數字時用,如未來檔期的「90 天已訂率」)。
    # 原本 calendar_sections 自己寫 2.6rem;它是資料值而非裝飾,所以收成 token
    # 而不是丟進 TYPE_EXEMPT —— 代價是字級階數由 8 階變 9 階。
    "metric_hero": ("2.6rem",  800, "normal"),
    "body":       ("0.875rem", 400, "normal"),   # 內文
    "caption":    ("0.78rem",  400, "normal"),   # 輔助說明
    "label":      ("0.7rem",   700, ".07em"),    # 表頭、KPI 標籤、膠囊
}

# 不套用字級 token 的白名單(各有不可替代的理由,改動前先確認)
TYPE_EXEMPT = (
    "index.py 首頁 Hero",          # 獨立全版動畫,字級隨 iframe 縮放
    "risk_ring() SVG 內字",        # SVG viewBox 座標,非 CSS rem
    "portal-icon",                 # 入口大圖示,尺寸即視覺主體
)

FONT_FAMILY = "'Noto Sans TC',sans-serif"

# ═══════════════════════════════════════════════════════════════
# 間距(4px 基準)
# ═══════════════════════════════════════════════════════════════
SPACE = {"1": "4px", "2": "8px", "3": "12px", "4": "16px",
         "6": "24px", "8": "32px"}

LAYOUT = {
    "page_top":     "1.6rem",     # 頁面上緣留白(全站唯一值,各頁不得覆寫)
    "section_gap":  "24px",       # 區塊之間(st.divider / hr 的 margin)
    "card_pad":     "16px 18px",  # 卡片內距
    "card_gap":     "8px",        # 卡片內元素間距
    "title_gap":    "8px",        # 標題 → 內容
    "desc_gap":     "12px",       # 說明文字 → 內容
}

# ═══════════════════════════════════════════════════════════════
# 圓角 / 邊框 / 陰影 / 斷點
# ═══════════════════════════════════════════════════════════════
RADIUS = {
    "sm":   "8px",     # 小標籤、note、內嵌區塊
    "md":   "12px",    # 卡片、KPI、圖片、表格容器
    "lg":   "16px",    # 大型面板、portal card
    "pill": "999px",   # 狀態標籤、膠囊
    "bar":  "4px",     # 卡片頂部色條、scrollbar
}

BORDER = {
    "default": f"1px solid {COLOR['border']}",
    "strong":  f"1px solid {COLOR['border2']}",
    "dashed":  f"1px dashed {COLOR['border2']}",
}

SHADOW = {
    "sm": "0 1px 4px rgba(0,0,0,.03)",        # 靜態卡片
    "md": "0 4px 18px rgba(42,42,42,.055)",   # 表格容器、hover
    "lg": "0 10px 34px rgba(0,0,0,.18)",      # tooltip、彈窗
}

BREAKPOINT = {"mobile": "760px", "tablet": "1200px"}

# ═══════════════════════════════════════════════════════════════
# 風險等級 —— 全站唯一來源
# ═══════════════════════════════════════════════════════════════
# 文案於 2026-07-24 拍板為「高風險 / 觀察 / 安全」(沿用現行上線版)。
# 原本 TIER_ZH 在 6 個檔案各寫一份(1_房東入口:34、notify_center:27、
# platform_sections:116、report_builder:95、report_builder:176、
# risk_cockpit_sections:17),其中兩份回傳 tuple、三份回傳含 emoji 的 str、
# 一份不含 emoji —— 一律改成從這裡取。
RISK_TIERS = {
    "red":    {"zh": "高風險", "emoji": "🔴", "color": "danger",
               "rule": "風險機率 ≥ 60%"},
    "yellow": {"zh": "觀察",   "emoji": "🟡", "color": "warning",
               "rule": "風險機率 35% – 60%"},
    "green":  {"zh": "安全",   "emoji": "🟢", "color": "success",
               "rule": "風險機率 < 35%"},
}
TIER_ORDER = ("red", "yellow", "green")

# 舊文案 → tier key。ui_components.risk_badge() 與各頁既有呼叫端傳的是中文,
# 保留別名才不會在收斂過程中失效。
TIER_ALIAS = {
    "高風險": "red",  "🔴 高風險": "red",
    "觀察":   "yellow", "🟡 觀察": "yellow", "中風險": "yellow", "中度風險": "yellow",
    "安全":   "green",  "🟢 安全": "green",  "低風險": "green",
}

# 「檔期資料不足」在 quadrant.py 的 docstring 寫「資料不足」、label 寫
# 「檔期資料不足」—— 統一以 label 為準。
STATUS_NO_DATA = "檔期資料不足"

# ═══════════════════════════════════════════════════════════════
# 分數帶(租客入口五科成績單總分 0–25)
# ═══════════════════════════════════════════════════════════════
# 門檻與文案以「新版房源評分模式規劃書 v1.0」為準(2026-07-24 使用者裁示)。
#
# 這裡原本是另一套五級色帶(22/18/14/10,非常優秀/優秀/普通/較差/最需比較),
# 與 tenant_scoring.total_and_band() 的四級規格門檻不同、用詞也不同,
# 卻同時出現在租客入口的畫面上(房源卡寫「優先查看」、地圖圖例寫「優秀」)。
# 裁示為「以規則書四級為準」,故色帶跟著規格走,五級版本廢止。
#
# 為什麼不套 RISK_TIERS:那是三態的風險判讀,這是四級的分數排序,語意不同;
# 但顏色仍取自同一組語意色,不另調新色。
# (門檻, 名稱, 色碼) —— 由高分往低分比對,第一個 total >= 門檻者勝出。
SCORE_BANDS = (
    (20, "優先查看",   TINT["success"]["fg"]),   # 深綠
    (15, "值得考慮",   COLOR["success"]),
    (10, "普通",       COLOR["warning"]),
    (0,  "建議多比較", COLOR["danger"]),
)


def score_band(total) -> tuple:
    """回傳該總分所屬的 (名稱, 色碼);無法判讀時退回最低帶。"""
    try:
        v = float(total)
    except (TypeError, ValueError):
        return SCORE_BANDS[-1][1], SCORE_BANDS[-1][2]
    for lo, name, color in SCORE_BANDS:
        if v >= lo:
            return name, color
    return SCORE_BANDS[-1][1], SCORE_BANDS[-1][2]


def tier_key(value) -> str | None:
    """把 tier key 或任何舊中文文案正規化成 red/yellow/green;無法辨識回 None。"""
    if value is None:
        return None
    s = str(value).strip()
    if s in RISK_TIERS:
        return s
    return TIER_ALIAS.get(s)


def tier_label(value, emoji: bool = True, default=None) -> str:
    """回傳統一文案。

    emoji=False 用於 PDF / 純文字信件等不適合放 emoji 的場合。
    default 給「認不出來就顯示空字串」的呼叫端(月報原本用 `.get(tier, "")`);
    不給時沿用「原樣回傳」,避免把未知值悄悄吞掉。
    """
    k = tier_key(value)
    if k is None:
        return str(value) if default is None else default
    t = RISK_TIERS[k]
    return f"{t['emoji']} {t['zh']}" if emoji else t["zh"]


def tier_label_map(emoji: bool = True) -> dict:
    """{tier key: 文案} —— 給 pandas `.map()` / `format_func` 這類要吃 dict 的場合。"""
    return {k: tier_label(k, emoji=emoji) for k in TIER_ORDER}


def tier_color(value) -> str:
    """回傳該等級的語意色碼(找不到時退回 muted,不丟例外以免打斷渲染)。"""
    k = tier_key(value)
    return COLOR[RISK_TIERS[k]["color"]] if k else COLOR["muted"]


def tier_tint(value) -> dict:
    """回傳該等級的 {bg, fg, border} 淡底組合(badge/差異標籤用)。"""
    k = tier_key(value)
    return TINT[RISK_TIERS[k]["color"]] if k else TINT["neutral"]


# ═══════════════════════════════════════════════════════════════
# CSS 變數輸出(供 ui_components.inject_css 一次注入)
# ═══════════════════════════════════════════════════════════════
def css_variables() -> str:
    """把所有 token 攤成 :root 的 CSS 自訂屬性,前綴 --sa-。

    有了它,CSS 區塊就能寫 var(--sa-danger) 而不是把色碼複製一遍;
    Python 端則直接讀 COLOR/RADIUS 等 dict。
    """
    rows = []
    for k, v in COLOR.items():
        rows.append(f"--sa-{k}:{v};")
    for role, t in TINT.items():
        for slot, v in t.items():
            rows.append(f"--sa-{role}-{slot}:{v};")
    for k, (size, weight, ls) in TYPE.items():
        name = k.replace("_", "-")
        rows.append(f"--sa-text-{name}:{size};")
        rows.append(f"--sa-text-{name}-weight:{weight};")
        rows.append(f"--sa-text-{name}-ls:{ls};")
    for k, v in SPACE.items():
        rows.append(f"--sa-space-{k}:{v};")
    for k, v in LAYOUT.items():
        rows.append(f"--sa-{k.replace('_', '-')}:{v};")
    for k, v in RADIUS.items():
        rows.append(f"--sa-radius-{k}:{v};")
    for k, v in SHADOW.items():
        rows.append(f"--sa-shadow-{k}:{v};")
    rows.append(f"--sa-font:{FONT_FAMILY};")
    return ":root{" + "".join(rows) + "}"


# ═══════════════════════════════════════════════════════════════
# 相容層:舊的 ui_components.P 扁平色盤
# ═══════════════════════════════════════════════════════════════
# 全站 13 個檔案在用 `from modules.ui_components import P`,鍵名不能變。
# 這裡由新 token 反推出同樣的值,讓 ui_components.P 直接指過來即可,
# 既有呼叫端一行都不用改。
LEGACY_P = {
    "bg": COLOR["bg"], "surface": COLOR["surface"], "card": COLOR["card"],
    "border": COLOR["border"], "border2": COLOR["border2"],
    "ink": COLOR["ink"], "ink2": COLOR["ink2"], "muted": COLOR["muted"],
    "primary": COLOR["primary"], "accent": COLOR["secondary"],
    "high": COLOR["danger"], "medium": COLOR["warning"], "low": COLOR["success"],
    "tag_bg": TINT["neutral"]["bg"], "mbg": TINT["primary"]["bg"],
    "mtxt": TINT["primary"]["fg"],
    "landlord": COLOR["primary"], "tenant": COLOR["success"],
    "admin": COLOR["secondary"],
}
