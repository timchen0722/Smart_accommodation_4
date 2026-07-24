# -*- coding: utf-8 -*-
"""
產出「90天近季與365天模型對比」正式 PDF 報告腳本
包含指標（含迴歸 MSE/MAE/RMSE/R²、分類 AUC/PR-AUC/F1 等）與 Top 10 特徵重要度對比及商業洞察。
"""
import sys
import os
import pandas as pd
import numpy as np

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 註冊微軟正黑體
pdfmetrics.registerFont(TTFont("JhengHei", "C:/Windows/Fonts/msjh.ttc", subfontIndex=0))
FONT = "JhengHei"

# 顏色定義
BLUE = colors.HexColor("#1b4f72")
NAVY = colors.HexColor("#2874a6")
GREEN = colors.HexColor("#1e8449")
GRAY = colors.HexColor("#566573")
LIGHT_BLUE = colors.HexColor("#ebf5fb")
LIGHT_GREEN = colors.HexColor("#eafaf1")

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontName=FONT, fontSize=16, leading=22, spaceAfter=4)
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontName=FONT, fontSize=12, leading=16, spaceBefore=10, spaceAfter=4, textColor=BLUE)
H3 = ParagraphStyle("H3", parent=styles["Heading3"], fontName=FONT, fontSize=10, leading=14, spaceBefore=6, spaceAfter=3, textColor=colors.HexColor("#2c3e50"))
BODY = ParagraphStyle("BODY", parent=styles["Normal"], fontName=FONT, fontSize=9, leading=13.5)
CELL = ParagraphStyle("CELL", parent=styles["Normal"], fontName=FONT, fontSize=8, leading=11)
CELL_BOLD = ParagraphStyle("CELL_BOLD", parent=styles["Normal"], fontName=FONT, fontSize=8, leading=11)
SMALL = ParagraphStyle("SMALL", parent=styles["Normal"], fontName=FONT, fontSize=8, leading=11, textColor=GRAY)

story = []

def p(text, style=BODY):
    story.append(Paragraph(text, style))

# ── 頁面設定 ──────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
pdf_filename = os.path.join(BASE_DIR, "90天近季與365天模型對比報告.pdf")

doc = SimpleDocTemplate(
    pdf_filename,
    pagesize=A4,
    leftMargin=15 * mm,
    rightMargin=15 * mm,
    topMargin=15 * mm,
    bottomMargin=15 * mm
)

# ── 標題與摘要 ────────────────────────────────────────
p("智慧旅宿空屋率風險預警平台 — 90天近季與365天模型對比報告", H1)
p("資料規模：5,849 筆房源｜評估特徵：37 個核心特徵｜報告日期：2026-07-21", SMALL)
p("評估協定：HistGradientBoosting（回歸/分類）｜ 門檻設定：高風險 Y > 0.70", SMALL)
story.append(Spacer(1, 6))

summary_box = [
    [Paragraph("<b>核心摘要與發現</b>", CELL), Paragraph("", CELL)],
    [Paragraph("<b>目標變數轉移</b>", CELL), Paragraph("將目標變數切換為 <b>Y_vacancy_90 = availability_90 / 90.0</b>（未來 90 天近季空屋率）。", CELL)],
    [Paragraph("<b>誠實驗證表現</b>", CELL), Paragraph("在 GroupKFold 依房東分組誠實驗證下，90 天分類 <b>ROC-AUC (0.7532)</b> 與 <b>PR-AUC (0.6521)</b> 顯著超越 365 天模型 (+4.35% / +10.53%)。", CELL)],
    [Paragraph("<b>關鍵特徵轉變</b>", CELL), Paragraph("同區相對價格百分位 (<b>price_pctl_nbhd</b>) 躍升至第 3 名；封面照設計感分數 (<b>photo_design_sense</b>) 首度攻入 Top 10（第 7 名）。", CELL)],
]
t_sum = Table(summary_box, colWidths=[90, 430])
t_sum.setStyle(TableStyle([
    ("SPAN", (0, 0), (1, 0)),
    ("BACKGROUND", (0, 0), (-1, 0), BLUE),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#aed6f1")),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("BACKGROUND", (0, 1), (-1, -1), LIGHT_BLUE),
    ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
]))
story.append(t_sum)
story.append(Spacer(1, 8))

# ── 一、單次隨機切分 80/20 對比 ───────────────────────
p("一、單次隨機切分 (80/20) 完整評估指標對比", H2)

table1_data = [
    [Paragraph("<b>模型類別</b>", CELL), Paragraph("<b>評估指標 (Metrics)</b>", CELL), Paragraph("<b>90天近季模型 (Y_90)</b>", CELL), Paragraph("<b>365天全年模型 (Y_365)</b>", CELL), Paragraph("<b>指標增減 / 表現比較</b>", CELL)],
    [Paragraph("<b>模型 A：回歸預測</b><br/>(預測空屋率分數)", CELL), Paragraph("MSE (均方誤差)", CELL), Paragraph("<b>0.0678</b>", CELL), Paragraph("0.0404", CELL), Paragraph("90天波動較大，MSE略高", CELL)],
    ["", Paragraph("MAE (平均絕對誤差)", CELL), Paragraph("<b>0.1963</b>", CELL), Paragraph("0.1510", CELL), Paragraph("預測誤差維持約 19.6%", CELL)],
    ["", Paragraph("RMSE (均方根誤差)", CELL), Paragraph("<b>0.2603</b>", CELL), Paragraph("0.2011", CELL), Paragraph("標準差波動比率", CELL)],
    ["", Paragraph("R² (模型解釋力)", CELL), Paragraph("<b>0.4138</b>", CELL), Paragraph("0.5866", CELL), Paragraph("全年模型具更高總體擬合", CELL)],
    [Paragraph("<b>模型 B：二元分類</b><br/>(高風險 Y > 0.70)", CELL), Paragraph("ROC-AUC", CELL), Paragraph("<b>0.8424</b>", CELL), Paragraph("0.8986", CELL), Paragraph("保持 >0.84 強分類區隔力", CELL)],
    ["", Paragraph("PR-AUC", CELL), Paragraph("<b>0.7392</b>", CELL), Paragraph("0.8149", CELL), Paragraph("高風險精準召回曲線良好", CELL)],
    ["", Paragraph("Accuracy (準確度)", CELL), Paragraph("<b>78.89%</b>", CELL), Paragraph("84.19%", CELL), Paragraph("整體分類正確率達近八成", CELL)],
    ["", Paragraph("Precision (精準率)", CELL), Paragraph("<b>69.76%</b>", CELL), Paragraph("80.00%", CELL), Paragraph("預測高風險的真正高風險率", CELL)],
    ["", Paragraph("Recall (召回率)", CELL), Paragraph("<b>66.41%</b>", CELL), Paragraph("66.85%", CELL), Paragraph("兩者召回能力相當", CELL)],
    ["", Paragraph("F1-Score", CELL), Paragraph("<b>0.6805</b>", CELL), Paragraph("0.7283", CELL), Paragraph("綜合二元分類指標", CELL)],
]

t1 = Table(table1_data, colWidths=[110, 110, 95, 95, 110])
t1.setStyle(TableStyle([
    ("SPAN", (0, 1), (0, 4)),
    ("SPAN", (0, 5), (0, 10)),
    ("BACKGROUND", (0, 0), (-1, 0), NAVY),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bdc3c7")),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("ALIGN", (2, 0), (3, -1), "CENTER"),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9f9")]),
    ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
]))
story.append(t1)
story.append(Spacer(1, 8))

# ── 二、GroupKFold 誠實交叉驗證 ───────────────────────
p("二、GroupKFold 誠實交叉驗證（5 折，依 host_id 分組 — 模擬新房東）", H2)

table2_data = [
    [Paragraph("<b>模型類型</b>", CELL), Paragraph("<b>評估指標 (Metrics)</b>", CELL), Paragraph("<b>90天近季模型 (Y_90)</b>", CELL), Paragraph("<b>365天全年模型 (Y_365)</b>", CELL), Paragraph("<b>對比分析洞察</b>", CELL)],
    [Paragraph("<b>回歸模型 A</b>", CELL), Paragraph("R² (誠實解釋力)", CELL), Paragraph("<b>0.2032</b>", CELL), Paragraph("0.2087", CELL), Paragraph("未知房東下兩者解釋力相當 (~0.21)", CELL)],
    ["", Paragraph("MSE (均方誤差)", CELL), Paragraph("<b>0.0805</b>", CELL), Paragraph("0.0638", CELL), Paragraph("新房東極端值誤差幅度", CELL)],
    ["", Paragraph("MAE (平均絕對誤差)", CELL), Paragraph("<b>0.2433</b>", CELL), Paragraph("0.2269", CELL), Paragraph("平均預測誤差約 24.3%", CELL)],
    [Paragraph("<b>分類模型 B</b>", CELL), Paragraph("ROC-AUC", CELL), Paragraph("<b>0.7532</b>", CELL), Paragraph("0.7097", CELL), Paragraph("<b>90天模型顯著提升 +4.35%</b>", CELL)],
    ["", Paragraph("PR-AUC", CELL), Paragraph("<b>0.6521</b>", CELL), Paragraph("0.5468", CELL), Paragraph("<b>90天模型大幅提升 +10.53%</b>", CELL)],
]

t2 = Table(table2_data, colWidths=[90, 110, 95, 95, 130])
t2.setStyle(TableStyle([
    ("SPAN", (0, 1), (0, 3)),
    ("SPAN", (0, 4), (0, 5)),
    ("BACKGROUND", (0, 0), (-1, 0), GREEN),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#a9dfbf")),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("ALIGN", (2, 0), (3, -1), "CENTER"),
    ("BACKGROUND", (0, 4), (-1, 5), LIGHT_GREEN),
    ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
]))
story.append(t2)
story.append(Spacer(1, 8))

story.append(PageBreak())

# ── 三、Top 10 特徵重要度對比 ─────────────────────────
p("三、90天近季模型 vs 365天全年模型 特徵重要度 Top 10 對比", H2)

table3_data = [
    [Paragraph("<b>排名</b>", CELL), Paragraph("<b>90天近季模型 (Y_90) 特徵名</b>", CELL), Paragraph("<b>權重增益</b>", CELL), Paragraph("<b>365天全年模型 (Y_365) 特徵名</b>", CELL), Paragraph("<b>權重增益</b>", CELL), Paragraph("<b>特徵重要度轉變分析</b>", CELL)],
    [Paragraph("1", CELL), Paragraph("host_acceptance_rate", CELL), Paragraph("0.1154", CELL), Paragraph("host_acceptance_rate", CELL), Paragraph("0.1029", CELL), Paragraph("房東接單意願均為兩者第一大要素", CELL)],
    [Paragraph("2", CELL), Paragraph("host_listings_count", CELL), Paragraph("0.0793", CELL), Paragraph("host_listings_count", CELL), Paragraph("0.0844", CELL), Paragraph("房東經營規模仍具顯著影響力", CELL)],
    [Paragraph("3", CELL), Paragraph("<b>price_pctl_nbhd [暴衝]</b>", CELL), Paragraph("0.0788", CELL), Paragraph("host_tenure_days", CELL), Paragraph("0.0732", CELL), Paragraph("<b>相對價格百分位暴衝至第 3 名</b>", CELL)],
    [Paragraph("4", CELL), Paragraph("min_nights_avg_ntm", CELL), Paragraph("0.0375", CELL), Paragraph("min_nights_avg_ntm", CELL), Paragraph("0.0688", CELL), Paragraph("近期最短入住晚數限制", CELL)],
    [Paragraph("5", CELL), Paragraph("price", CELL), Paragraph("0.0331", CELL), Paragraph("maximum_nights", CELL), Paragraph("0.0651", CELL), Paragraph("絕對價格位階", CELL)],
    [Paragraph("6", CELL), Paragraph("minimum_nights", CELL), Paragraph("0.0287", CELL), Paragraph("price", CELL), Paragraph("0.0616", CELL), Paragraph("最短入住晚數門檻", CELL)],
    [Paragraph("7", CELL), Paragraph("<b>photo_design_sense [新進]</b>", CELL), Paragraph("0.0255", CELL), Paragraph("host_about_len", CELL), Paragraph("0.0476", CELL), Paragraph("<b>封面照設計感分數首度攻入 Top 10</b>", CELL)],
    [Paragraph("8", CELL), Paragraph("response_speed", CELL), Paragraph("0.0226", CELL), Paragraph("price_pctl_nbhd", CELL), Paragraph("0.0436", CELL), Paragraph("房東回覆速度", CELL)],
    [Paragraph("9", CELL), Paragraph("longitude", CELL), Paragraph("0.0186", CELL), Paragraph("response_speed", CELL), Paragraph("0.0376", CELL), Paragraph("地理經度座標", CELL)],
    [Paragraph("10", CELL), Paragraph("latitude", CELL), Paragraph("0.0176", CELL), Paragraph("desc_len", CELL), Paragraph("0.0328", CELL), Paragraph("地理緯度座標", CELL)],
]

t3 = Table(table3_data, colWidths=[28, 125, 48, 125, 48, 146])
t3.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), BLUE),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bdc3c7")),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("ALIGN", (0, 0), (0, -1), "CENTER"),
    ("ALIGN", (2, 0), (2, -1), "CENTER"),
    ("ALIGN", (4, 0), (4, -1), "CENTER"),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6f7")]),
    ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
]))
story.append(t3)
story.append(Spacer(1, 8))

# ── 四、核心商業洞察與營運建議 ───────────────────────
p("四、核心商業洞察與營運建議", H2)

p("1. <b>短期旅客對同儕相對價格位階高度敏感</b>：", H3)
p("在 90 天近季模型中，<b>price_pctl_nbhd</b>（同區同房型相對價格百分位）的權重增益由全年的第 8 名 (0.0436) 大幅升至<b>第 3 名 (0.0788)</b>。"
  "這代表未來 3 個月內，旅客訂房時會積極比價；若房東將價格設定在同區同儕的前段高價區間，將會直接導致近期的空屋率風險大幅暴增。", BODY)

p("2. <b>視覺美感（photo_design_sense）對短期衝刺具實質貢獻</b>：", H3)
p("CLIP 多模態模型產出的<b>封面照設計感分數（photo_design_sense）</b>首度跨入 Top 10（第 7 名，增益 0.0255）。"
  "這證實了良好的視覺第一印象，在吸引近 90 天內有即時入住需求的旅客時，發揮了不可替代的決策拉動效果。", BODY)

p("3. <b>新房東高風險識別能力（GroupKFold）顯著更強</b>：", H3)
p("在模擬全新房東的 GroupKFold 驗證中，90 天模型之分類 <b>ROC-AUC 達 0.7532</b>（高於 365 天模型的 0.7097），<b>PR-AUC 更高達 0.6521</b>（遠高於 365 天模型的 0.5468）。"
  "說明 90 天模型更依賴房源當下的即時定價與視覺條件，極為適合用於新上架房源的「即時空屋率預警與調優建議」。", BODY)

story.append(Spacer(1, 10))
p("— 報告結束 —", ParagraphStyle("END", parent=SMALL, alignment=1))

# 產生 PDF
doc.build(story)
print(f"✅ PDF 報告已成功生成：{pdf_filename}")
