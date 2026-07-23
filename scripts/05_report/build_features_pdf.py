# -*- coding: utf-8 -*-
"""產生「特徵總覽」PDF：列出專案用過的所有特徵，含繁中備註、來源、採用狀況。
資料來源：
  - 核心採用/影像/POI：本檔內結構化資料（取自各實驗報告與 dataset_final.csv 欄位）
  - 排除的 listings 原始欄位：直接讀 feature_comparison_table.csv（避免抄錯）
"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import os
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

pdfmetrics.registerFont(TTFont("JhengHei", "C:/Windows/Fonts/msjh.ttc", subfontIndex=0))
FONT = "JhengHei"
BLUE = colors.HexColor("#2166ac")
GREEN = colors.HexColor("#1a7f4b")
RED = colors.HexColor("#b2182b")
GRAY = colors.HexColor("#666666")

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontName=FONT, fontSize=17, leading=22, spaceAfter=4)
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontName=FONT, fontSize=12.5, leading=16,
                    spaceBefore=12, spaceAfter=4, textColor=BLUE)
H3 = ParagraphStyle("H3", parent=styles["Heading3"], fontName=FONT, fontSize=10.5, leading=14,
                    spaceBefore=7, spaceAfter=3, textColor=colors.HexColor("#333333"))
BODY = ParagraphStyle("BODY", parent=styles["Normal"], fontName=FONT, fontSize=9.5, leading=14)
CELL = ParagraphStyle("CELL", parent=styles["Normal"], fontName=FONT, fontSize=8, leading=10.5)
CODE = ParagraphStyle("CODE", parent=styles["Normal"], fontName=FONT, fontSize=7.6, leading=10.5,
                      textColor=colors.HexColor("#0b4a8a"))
SMALL = ParagraphStyle("SMALL", parent=styles["Normal"], fontName=FONT, fontSize=8, leading=11, textColor=GRAY)

story = []

# 先讀排除欄位（供摘要盒與 ⑤ 區塊共用同一數字，避免不一致）
_df = pd.read_csv("../../feature_comparison_table.csv")
_excl = _df[_df["採用狀態"].astype(str).str.startswith("未採用")].copy()
_excl = _excl[~_excl["對應最終特徵"].astype(str).str.startswith("photo_")].copy()
_excl["原因分類"] = _excl["採用狀態"].str.replace("未採用-", "", regex=False)
N_EXCL = len(_excl)


def p(text, style=BODY):
    story.append(Paragraph(text, style))


def feat_table(rows, header_bg=BLUE):
    """rows[0]=表頭；欄=[特徵名, 繁中備註, 來源, 採用狀況]。特徵名用 code 樣式、其餘自動換行。"""
    widths = [95, 168, 128, 132]
    data = [[Paragraph(f"<b>{c}</b>", CELL) for c in rows[0]]]
    for r in rows[1:]:
        data.append([
            Paragraph(r[0], CODE),
            Paragraph(r[1], CELL),
            Paragraph(r[2], CELL),
            Paragraph(r[3], CELL),
        ])
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#aaaaaa")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f8")]),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 6))


def excl_table(rows, header_bg=RED):
    """排除欄位表：欄=[原始欄位, 繁中說明, 排除原因]。"""
    widths = [130, 265, 128]
    data = [[Paragraph(f"<b>{c}</b>", CELL) for c in rows[0]]]
    for r in rows[1:]:
        data.append([Paragraph(r[0], CODE), Paragraph(r[1], CELL), Paragraph(r[2], CELL)])
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#aaaaaa")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#faeceb")]),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 6))


# ======================================================================
# 封面
# ======================================================================
p("智慧旅宿空屋率風險預警平台 — 特徵總覽", H1)
p("列出專案全程用過的所有特徵：繁體中文備註、資料來源、以及在什麼狀況下採用或排除。"
  "資料源 Inside Airbnb 台北（listings 81 原始欄）＋台北官方 POI＋OpenStreetMap POI＋封面照 CLIP 影像。", SMALL)
p("目標變數 Y_vacancy = availability_365 / 365（未來一年空屋率，0~1）。以下所有「採用/排除」皆以此為預測標的。", SMALL)
story.append(Spacer(1, 4))

# 摘要盒
summary = [
    [Paragraph("<b>特徵層級</b>", CELL), Paragraph("<b>數量</b>", CELL), Paragraph("<b>採用狀況</b>", CELL)],
    [Paragraph("① 核心採用（進最終雙模型）", CELL), "37", Paragraph("直接沿用 21＋衍生工程 16；寫入 dataset_final.csv", CELL)],
    [Paragraph("② 影像多模態", CELL), "9", Paragraph("僅 photo_design_sense 採用，其餘 8 個測試後排除（留資料集供研究）", CELL)],
    [Paragraph("③ 外部 POI（OSM 第一輪）", CELL), "9", Paragraph("實驗性；完整模型冗餘、冷啟動微幅有效，未寫回主資料集", CELL)],
    [Paragraph("④ 外部 POI（台北官方第二輪）", CELL), "16", Paragraph("實驗性；同上，交通組冷啟動最強，未寫回主資料集", CELL)],
    [Paragraph("⑤ 排除的 listings 原始欄位", CELL), str(N_EXCL), Paragraph("防洩漏／已測無效／未測試／無意義四類", CELL)],
    [Paragraph("⑥ 前向選擇精選（最後一輪）", CELL), "~13", Paragraph("從 53 候選重新篩序，11 個即超越 base-37；最小充分子集", CELL)],
]
t = Table(summary, colWidths=[190, 45, 288])
t.setStyle(TableStyle([
    ("FONTNAME", (0, 0), (-1, -1), FONT), ("FONTSIZE", (0, 0), (-1, -1), 8.5),
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#333333")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#aaaaaa")),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f0f0")]),
    ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
]))
story.append(t)
story.append(Spacer(1, 8))

# ======================================================================
# ① 核心採用 —— 直接沿用
# ======================================================================
p("① 核心採用特徵（37 個，進最終模型 A/B）", H2)
p("這 37 個特徵寫入 dataset_final.csv，是模型 A（迴歸）與模型 B（分類）實際使用的特徵集。分「直接沿用原始欄位」與「衍生工程特徵」兩類。", SMALL)

p("1-A　直接沿用原始欄位（21 個）", H3)
direct = [
    ["特徵名稱", "繁體中文備註", "來源", "採用狀況"],
    ["price", "每晚價格", "listings 原始欄（去貨幣符號轉數值）", "基礎結構化屬性，直接採用；另用於算周邊價格百分位"],
    ["accommodates", "可住人數", "listings 原始欄", "基礎屬性，直接採用"],
    ["bedrooms", "臥室數", "listings 原始欄", "基礎屬性；前向選擇第 14 步選入"],
    ["beds", "床位數", "listings 原始欄", "基礎屬性；前向選擇第 1 步即選入"],
    ["bathrooms", "衛浴數", "listings 原始欄", "基礎屬性，直接採用"],
    ["minimum_nights", "最少入住晚數", "listings 原始欄", "基礎屬性；EBM 發現 2~3 晚最優的非線性效應"],
    ["maximum_nights", "最多入住晚數", "listings 原始欄", "訂房彈性，重要度長期前 3 名；SHAP 呈非線性"],
    ["min_nights_avg_ntm", "近期平均最短入住", "minimum_nights_avg_ntm 原始欄", "重要度前 5 名；前向選擇第 9 步選入"],
    ["instant_bookable", "可即時預訂", "listings 原始欄（t/f→0/1）", "基礎屬性，直接採用"],
    ["latitude", "緯度", "listings 原始欄", "地點特徵；另用於 BallTree 算周邊密度"],
    ["longitude", "經度", "listings 原始欄", "地點特徵；另用於 BallTree 算周邊密度"],
    ["host_is_superhost", "是否超讚房東", "listings 原始欄（t/f→0/1）", "房東品質，直接採用"],
    ["host_response_rate", "房東回覆率", "listings 原始欄（轉 0~1）", "房東品質，直接採用"],
    ["host_acceptance_rate", "房東接受率", "listings 原始欄（轉 0~1）", "重要度最高特徵之一（SHAP Top1）"],
    ["review_scores_rating", "總體評分", "listings 原始欄", "口碑評分，直接採用"],
    ["review_scores_accuracy", "描述準確性評分", "listings 原始欄", "口碑評分，直接採用"],
    ["review_scores_cleanliness", "清潔度評分", "listings 原始欄", "口碑評分，直接採用"],
    ["review_scores_checkin", "入住體驗評分", "listings 原始欄", "口碑評分，直接採用"],
    ["review_scores_communication", "溝通評分", "listings 原始欄", "口碑評分；前向選擇五大核心訊號之一"],
    ["review_scores_location", "地點評分", "listings 原始欄", "口碑評分；另用於算周邊口碑排名"],
    ["review_scores_value", "性價比評分", "listings 原始欄", "口碑評分，直接採用"],
]
feat_table(direct)

p("1-B　衍生工程特徵（16 個）", H3)
derived = [
    ["特徵名稱", "繁體中文備註", "來源", "採用狀況"],
    ["host_listings_count", "房東房源數", "calculated_host_listings_count 更名", "重要度最高之一；但也是房東洩漏的主來源（見誠實評估）"],
    ["host_tenure_days", "房東經營天數", "host_since 與 last_scraped 相減", "經營資歷，衍生採用"],
    ["amenities_count", "設施數量", "amenities 欄逗號數＋1", "基礎屬性，衍生採用"],
    ["self_checkin", "是否提供自助入住", "amenities 關鍵字比對（self check-in / keypad / lockbox / smart lock）", "Tier1 批量測試驗證有效，重要度第 10；R² 0.561→0.572"],
    ["response_speed", "房東回覆速度", "host_response_time 文字序數化（1~4）", "比「回覆率」多一層快慢維度，採用"],
    ["desc_len", "房源描述字數", "description 取字數長度", "代表經營用心度，重要度前 10~15 名"],
    ["host_about_len", "房東自介字數", "host_about 取字數長度", "代表經營用心度，衍生採用"],
    ["room_type_code", "房型編碼", "room_type 類別編碼", "基礎屬性；另為競爭特徵的分組依據"],
    ["property_type_code", "物業類型編碼", "property_type 類別編碼", "基礎屬性，衍生採用"],
    ["neighbourhood_code", "行政區編碼", "neighbourhood_cleansed 類別編碼", "基礎屬性；另為競爭特徵的分組依據"],
    ["price_pctl_nbhd", "周邊價格百分位", "price ＋ 行政區 ＋ 房型 組合", "競爭特徵（第三方平台核心設計）：滯銷是相對而非絕對定價"],
    ["score_pctl_nbhd", "周邊口碑排名", "評分 ＋ 行政區 ＋ 房型 組合", "競爭特徵；前向選擇第 11 步即超越 base-37"],
    ["amenities_vs_median", "設施相對周邊", "設施數 ÷ 周邊同房型中位數", "競爭特徵：競爭力相對強弱"],
    ["nbr_density_1km", "周邊 1km 房源密度", "經緯度用 BallTree haversine 查詢", "競爭特徵：供給過剩程度"],
    ["nbr_density_same_type_1km", "周邊 1km 同房型密度", "經緯度 ＋ 房型用 BallTree", "競爭特徵：直接對手數量"],
    ["photo_design_sense", "封面照設計感", "picture_url 下載後用 CLIP-ViT-B/32 zero-shot 推論", "9 個影像特徵中唯一有效者，兩模型皆前 10~15 名"],
]
feat_table(derived, header_bg=GREEN)

story.append(PageBreak())

# ======================================================================
# ② 影像多模態
# ======================================================================
p("② 影像多模態特徵（9 個，僅 1 個採用）", H2)
p("從房源封面照（picture_url）擷取：OpenCV 客觀畫質 6 項＋CLIP-ViT-B/32 zero-shot 主觀美感 3 項。"
  "全量 6,241 張經 Airbnb CDN 限速下載後批次擷取。9 個特徵全部保留在 dataset_final.csv 供研究，"
  "但經全量相關性與加入前後比較後，<b>僅 photo_design_sense 通過採用</b>，其餘 8 個因訊號不穩健或無鑑別度而排除。", SMALL)
photo = [
    ["特徵名稱", "繁體中文備註", "來源／方法", "採用狀況"],
    ["photo_design_sense", "封面照設計感", "CLIP zero-shot 對比提示詞", "<b>採用</b>；重要度前 10~15；EBM：評分&lt;0.3 時 +9.8pp 風險"],
    ["photo_brightness", "平均亮度", "OpenCV", "排除：小樣本(n=500)相關 -0.106 看似最高，全量僅 -0.023，訊號不穩健"],
    ["photo_width", "解析度寬度", "OpenCV", "排除：全量相關僅 -0.112，重要度 21/45"],
    ["photo_height", "解析度高度", "OpenCV", "排除：重要度 33/45"],
    ["photo_contrast", "對比度（灰階標準差）", "OpenCV", "排除：重要度 37/45"],
    ["photo_sharpness", "清晰度（Laplacian 變異數）", "OpenCV", "排除：重要度 44/45，貢獻近乎 0"],
    ["photo_colorfulness", "色彩豐富度", "OpenCV", "排除：重要度 23/45"],
    ["photo_coziness", "溫馨感", "CLIP zero-shot", "排除：重要度 45/45，貢獻為負（等同雜訊）"],
    ["photo_cleanliness", "乾淨度", "CLIP zero-shot", "排除：均值 0.966、判斷過度飽和，鑑別力弱"],
]
feat_table(photo, header_bg=colors.HexColor("#8250c4"))

# ======================================================================
# ③ 外部 POI — OSM 第一輪
# ======================================================================
p("③ 外部特徵 — OpenStreetMap POI（第一輪，2026-07-09）", H2)
p("回應 Office Hours「補外部特徵（如飯店密度）」建議，用 OSM Overpass API 抓台北旅宿/景點/超商 POI。"
  "結論：完整模型冗餘、<b>新房東冷啟動小幅有效（R² +6%）</b>，<b>未寫回主資料集</b>。"
  "反直覺發現：飯店/景點密度與空屋率<b>負相關</b>——是「地段熱度」代理而非「競爭懲罰」。", SMALL)
osm = [
    ["特徵名稱", "繁體中文備註", "來源", "採用狀況"],
    ["hotel_count_1km", "周邊 1km 飯店數", "OSM 旅宿 POI（732 筆）", "實驗；與 Y 相關 -0.131（需求代理）；完整模型冗餘"],
    ["hotel_count_500m", "周邊 500m 飯店數", "OSM 旅宿 POI", "實驗；相關 -0.141（最強新訊號之一）；未寫回"],
    ["airbnb_hotel_supply_ratio", "短租/飯店供給比", "衍生（飯店數推估）", "實驗；相關僅 +0.028，弱；未寫回"],
    ["attraction_1km", "周邊 1km 景點數", "OSM 觀光 POI（941 筆）", "實驗；相關 -0.150（全批最強）；被鄰里特徵吃掉"],
    ["attraction_500m", "周邊 500m 景點數", "OSM 觀光 POI", "實驗；相關 -0.120；未寫回"],
    ["conv_1km", "周邊 1km 超商數", "OSM 超商 POI（3,740 筆）", "實驗；相關 -0.033，弱；未寫回"],
    ["price_per_person", "每人單價", "price ÷ accommodates", "實驗；相關 +0.036，弱；未寫回"],
    ["price_per_bedroom", "每臥室單價", "price ÷ bedrooms", "實驗；重要度 13（相對高）但無顯著提升；未寫回"],
    ["beds_per_person", "床位密度", "beds ÷ accommodates", "實驗；相關 -0.046，弱；未寫回"],
]
feat_table(osm, header_bg=colors.HexColor("#c47f1a"))

story.append(PageBreak())

# ======================================================================
# ④ 外部 POI — 台北官方第二輪
# ======================================================================
p("④ 外部特徵 — 台北市政府官方開放資料 POI（第二輪，2026-07-15）", H2)
p("用更豐富的官方資料（含上輪缺的餐廳/捷運/公車/公園）重測同一問題，7 來源建 16 特徵（密度＋最近距離兩型）。"
  "結論：完整模型仍無顯著提升（第三度撞天花板）；<b>冷啟動交通組最強（R² 0.181→0.196，+8%）</b>，<b>未寫回主資料集</b>。", SMALL)
poi = [
    ["特徵名稱", "繁體中文備註", "來源", "採用狀況"],
    ["mrt_nearest_km", "最近捷運出入口距離(km)", "官方捷運出入口（388 筆）", "實驗；冷啟動交通組（CP 值最高的一組）；未寫回"],
    ["mrt_count_1km", "周邊 1km 捷運出入口數", "官方捷運出入口", "實驗；同上交通組；未寫回"],
    ["bus_nearest_km", "最近公車站距離(km)", "官方公車站牌（3,407 筆）", "實驗；交通組；未寫回"],
    ["bus_count_500m", "周邊 500m 公車站數", "官方公車站牌", "實驗；重要度最高的新特徵（第 19）；相關 -0.123"],
    ["rest_count_500m", "周邊 500m 餐廳數", "官方餐廳（20,182 筆）", "實驗；餐飲組，訊號被鄰里吃掉；未寫回"],
    ["rest_count_1km", "周邊 1km 餐廳數", "官方餐廳", "實驗；重要度第 23；未寫回"],
    ["rest_nearest_km", "最近餐廳距離(km)", "官方餐廳", "實驗；冗餘；未寫回"],
    ["cvs_count_500m", "周邊 500m 超商數", "官方超商門牌（3,317 筆）", "實驗；相關 -0.078；未寫回"],
    ["cvs_count_1km", "周邊 1km 超商數", "官方超商門牌", "實驗；冗餘；未寫回"],
    ["cvs_nearest_km", "最近超商距離(km)", "官方超商門牌", "實驗；相關 +0.080；未寫回"],
    ["park_count_1km", "周邊 1km 公園數", "官方公園（830 筆）", "實驗；前向選擇第 12 步（POI 中最高 Δ0.010）；未寫回"],
    ["park_nearest_km", "最近公園距離(km)", "官方公園", "實驗；社區組；未寫回"],
    ["school_count_1km", "周邊 1km 學校數", "官方學校（283 筆）", "實驗；社區組；未寫回"],
    ["school_nearest_km", "最近學校距離(km)", "官方學校", "實驗；冗餘；未寫回"],
    ["pharm_count_1km", "周邊 1km 藥局數", "官方藥局（340 筆）", "實驗；相關 +0.112；未寫回"],
    ["pharm_nearest_km", "最近藥局距離(km)", "官方藥局", "實驗；前向選擇第 16 步；未寫回"],
]
feat_table(poi, header_bg=colors.HexColor("#1a7f7f"))

# ======================================================================
# ⑤ 排除的 listings 原始欄位（讀 CSV）
# ======================================================================
p(f"⑤ 排除的 listings 原始欄位（{N_EXCL} 個）", H2)
p("以下原始欄位經評估後未進模型，依「排除原因」分四類。此表直接由 feature_comparison_table.csv 生成"
  "（不含 8 個影像特徵，已於 ② 列出）。", SMALL)

excl = _excl

reason_order = [
    ("防洩漏", "防洩漏（用於定義 Y 或與入住量高度相關，放入即作弊）", RED),
    ("已測試無效", "已測試但無效（重要度後段或無鑑別度）", colors.HexColor("#b5651d")),
    ("未測試", "未測試（多與既有特徵高度重複，判斷冗餘未試）", GRAY),
    ("全空/無意義", "全空或無建模意義（URL／中繼資訊／全缺值欄）", colors.HexColor("#555555")),
]
for key, title, col in reason_order:
    sub = excl[excl["原因分類"] == key]
    if len(sub) == 0:
        continue
    p(f"5-{title}（{len(sub)} 個）", H3)
    rows = [["原始欄位", "繁體中文說明", "排除原因"]]
    for _, r in sub.iterrows():
        note = str(r["說明"]).strip()
        tgt = str(r["對應最終特徵"]).strip()
        if tgt and tgt not in ("-", "nan"):
            note = f"[{tgt}] {note}"
        rows.append([str(r["listings_cleaned.csv.gz 原始欄位"]), note, key])
    excl_table(rows, header_bg=col)

# ======================================================================
# ⑥ 前向特徵選擇精選特徵（35 步，GroupKFold 誠實協定）
# ======================================================================
story.append(PageBreak())
p("⑥ 前向特徵選擇「精選特徵」（最後一輪，2026-07-15）", H2)
p("這是專案最後做的一輪特徵篩選：從候選池 <b>53 個</b>（① 的 37 個 base 特徵＋④ 的 16 個台北官方 POI；不含 OSM）出發，"
  "以空集合起步，每一步試遍所有剩餘候選、選「加入後 GroupKFold(依房東分組) R² 增益最大」者納入，逐一排出重要性順序。"
  "全程實測約需 4.5~5 小時，跑至第 35 步 R² 已在平台穩定抖動，經同意提前收尾（尾段為已知下滑，不影響結論）。", SMALL)

CN = {
    "beds": "床位數", "self_checkin": "自助入住", "maximum_nights": "最多入住晚數",
    "instant_bookable": "可即時預訂", "host_is_superhost": "超讚房東", "price": "每晚價格",
    "review_scores_communication": "溝通評分", "host_acceptance_rate": "房東接受率",
    "min_nights_avg_ntm": "近期平均最短入住", "latitude": "緯度", "score_pctl_nbhd": "周邊口碑排名",
    "park_count_1km": "周邊1km公園數", "desc_len": "房源描述字數", "bedrooms": "臥室數",
    "minimum_nights": "最少入住晚數", "pharm_nearest_km": "最近藥局距離", "accommodates": "可住人數",
    "host_response_rate": "房東回覆率", "price_pctl_nbhd": "周邊價格百分位", "amenities_count": "設施數量",
    "pharm_count_1km": "周邊1km藥局數", "photo_design_sense": "封面照設計感",
    "review_scores_location": "地點評分", "bus_count_500m": "周邊500m公車站數",
    "neighbourhood_code": "行政區編碼", "review_scores_value": "性價比評分",
    "school_count_1km": "周邊1km學校數", "rest_count_1km": "周邊1km餐廳數",
    "nbr_density_same_type_1km": "周邊1km同房型密度", "rest_nearest_km": "最近餐廳距離",
    "park_nearest_km": "最近公園距離", "review_scores_checkin": "入住評分",
    "review_scores_accuracy": "準確性評分", "review_scores_rating": "總體評分", "longitude": "經度",
}
CORE5 = {"review_scores_communication", "host_acceptance_rate", "min_nights_avg_ntm",
         "latitude", "score_pctl_nbhd"}

sfs = pd.read_csv("../../forward_selection_order.csv")


def sfs_table(sub, header_bg, note_fn):
    widths = [30, 150, 108, 62, 55, 115]
    head = ["步", "特徵名稱", "繁體中文", "累積 R²", "ΔR²", "判讀"]
    data = [[Paragraph(f"<b>{c}</b>", CELL) for c in head]]
    for _, r in sub.iterrows():
        f = r["feature"]
        data.append([
            Paragraph(str(int(r["step"])), CELL),
            Paragraph(f, CODE),
            Paragraph(CN.get(f, "—"), CELL),
            Paragraph(f"{r['cum_r2']:.4f}", CELL),
            Paragraph(f"{r['delta']:+.4f}", CELL),
            Paragraph(note_fn(r), CELL),
        ])
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#aaaaaa")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef7f0")]),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 6))


def note_top13(r):
    s = int(r["step"]); f = r["feature"]
    if s <= 6:
        return "貪婪平坦區（R² 卡 0.06，前段名次不穩）"
    if f in CORE5:
        tag = "★ 追平並超越 base-37（0.209）" if s == 11 else "核心訊號"
        return tag
    return "轉平段（邊際效益漸盡）"


p("6-A　精選充分子集（前 13 步 ＝ 實務最小充分特徵集）", H3)
p("到第 <b>11 個</b>特徵（score_pctl_nbhd, R²=0.220）即追平並超越原本 37 個 base 特徵的 0.209；"
  "把 R² 從 0.06 拉到 0.22 的<b>五大核心訊號</b>集中在第 7~11 步（下表標「核心訊號」）。實務上可安全砍到 11~13 個特徵而不損誠實表現。", SMALL)
sfs_table(sfs[sfs["step"] <= 13], GREEN, note_top13)

p("6-B　噪音平台（第 14~35 步 ＝ 加入後無實質提升）", H3)
p("第 14 步之後每步淨增益 ΔR² 幾乎都小於折間標準差（±0.04~0.06），R² 在 0.24~0.26 間抖動，屬噪音而非訊號。"
  "峰值 R²=0.2627 出現在第 27 步（school_count_1km），但 13→27 之間增益全為噪音，不具統計意義。POI 特徵多落在此帶。", SMALL)
sfs_table(sfs[sfs["step"] >= 14], colors.HexColor("#888888"),
          lambda r: ("◆ 峰值 R²=0.2627" if int(r["step"]) == 27 else "噪音帶（Δ < 折間 std）"))

try:
    from reportlab.platypus import Image as RLImage
    from PIL import Image as PILImage
    _w, _h = PILImage.open("../../forward_selection_curve.png").size
    story.append(RLImage("../../forward_selection_curve.png", width=165 * mm, height=165 * mm * _h / _w))
    story.append(Spacer(1, 3))
    p("圖：前向選擇 R² vs 特徵數邊際曲線（陡升段 7→11、轉平 ~13、噪音平台 14→35；紅虛線為 base-37 的 0.209）", SMALL)
except Exception as _e:
    p(f"（曲線圖 forward_selection_curve.png 未能嵌入：{_e}）", SMALL)

p("6-C　結論（精選特徵怎麼用）", H3)
p("• <b>最小充分子集 ≈ 11~13 個</b>：若要精簡上線模型，可只留這批而非全部 37 個，誠實表現不變。", BODY)
p("• <b>五大核心訊號</b>：review_scores_communication（溝通評分）、host_acceptance_rate（房東接受率）、"
  "min_nights_avg_ntm（近期平均最短入住）、latitude（緯度）、score_pctl_nbhd（周邊口碑排名）"
  "—— 房東接受率與最短入住天數是最強的兩個非地段訊號。", BODY)
p("• <b>誠實界線</b>：貪婪法在平坦區（第 1~6 步）順序可能次優，可信的是「集合組成與平台高度」而非每一步精確排名；"
  "單一特徵重要度排名仍以 SHAP / permutation 報告為準。本輪的價值在揭露「最小充分子集規模」與「冗餘結構」。", BODY)

# ======================================================================
# 尾註
# ======================================================================
p("附註：採用原則與方法論", H2)
p("• <b>防洩漏鐵律</b>：凡用於定義 Y_vacancy 的欄位（availability_*、estimated_occupancy/revenue、"
  "number_of_reviews*、reviews_per_month、last_review）一律禁入特徵。", BODY)
p("• <b>精簡優於堆疊（Occam's Razor）</b>：Tier1 批量測試 19 候選後精簡至 6 個有效者，"
  "精簡版 R²(0.586) ≥ 全加版(0.581)；前向特徵選擇進一步證明只需 11~13 個特徵即達誠實天花板。", BODY)
p("• <b>外部 POI 全數未寫回主資料集</b>：兩輪實驗（OSM＋台北官方）證明對完整模型冗餘，"
  "僅新房東冷啟動情境有小幅增益；若上線冷啟動模型，建議只併入交通組 4 欄。", BODY)
p("• <b>採用門檻</b>：GroupKFold（依房東分組）誠實評估下，增益需超出折間標準差才算有效；"
  "多數外部特徵的增益小於一個標準差，故定位為「方向一致的小幅增益」而非決定性採用。", BODY)
p("詳細數字與出處見 docs/專案完整報告_2026-07-15.md 及各實驗報告。", SMALL)

doc = SimpleDocTemplate("特徵總覽_2026-07-15.pdf", pagesize=A4,
                        leftMargin=16 * mm, rightMargin=16 * mm,
                        topMargin=15 * mm, bottomMargin=15 * mm,
                        title="智慧旅宿空屋率風險預警平台 — 特徵總覽")
doc.build(story)
print("OK", os.path.getsize("特徵總覽_2026-07-15.pdf"), "bytes")
print("排除欄位數:", len(excl))
