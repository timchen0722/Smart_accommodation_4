# -*- coding: utf-8 -*-
"""把前向特徵選擇實驗結果（指標＋35特徵重要度＋曲線）產成正式 PDF（微軟正黑體）。"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                Image, PageBreak)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# ---- 註冊微軟正黑體（.ttc 用 subfontIndex）----
pdfmetrics.registerFont(TTFont("JhengHei", "C:/Windows/Fonts/msjh.ttc", subfontIndex=0))
FONT = "JhengHei"

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontName=FONT, fontSize=17, leading=22, spaceAfter=6)
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontName=FONT, fontSize=12.5, leading=16,
                    spaceBefore=10, spaceAfter=4, textColor=colors.HexColor("#2166ac"))
BODY = ParagraphStyle("BODY", parent=styles["Normal"], fontName=FONT, fontSize=9.5, leading=14)
SMALL = ParagraphStyle("SMALL", parent=styles["Normal"], fontName=FONT, fontSize=8, leading=11,
                       textColor=colors.HexColor("#666666"))

story = []


def p(text, style=BODY):
    story.append(Paragraph(text, style))


# ===== 標題 =====
p("前向特徵選擇實驗結果報告", H1)
p("空房率預測模型｜資料 5,849 筆・房東 1,296 位｜實驗日期 2026-07-15", SMALL)
p("評估協定：HistGradientBoosting（max_iter=500）＋ GroupKFold（依 host_id 5 折）誠實評估", SMALL)
story.append(Spacer(1, 8))

# ===== 一、模型指標 =====
p("一、模型指標（以 35 個篩選特徵重訓）", H2)
metric_rows = [
    ["評估法", "R²", "MAE", "MSE", "AUC", "說明"],
    ["① 單次切分 80/20", "0.5034", "0.1701", "0.0486", "0.8721", "含房東洩漏，虛高不可信"],
    ["② GroupKFold 誠實", "0.2492\n±0.0512", "0.2200\n±0.006", "0.0712", "0.7384\n±0.0223", "面對全新房東的真實表現"],
]
t = Table(metric_rows, colWidths=[70, 42, 42, 42, 46, 118])
t.setStyle(TableStyle([
    ("FONTNAME", (0, 0), (-1, -1), FONT), ("FONTSIZE", (0, 0), (-1, -1), 8.5),
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2166ac")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("BACKGROUND", (0, 2), (-1, 2), colors.HexColor("#eaf2fb")),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b0b0b0")),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (1, 0), (4, -1), "CENTER"),
    ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
]))
story.append(t)
story.append(Spacer(1, 4))
p("各折 R²：0.236 / 0.280 / 0.320 / 0.244 / 0.166（折間差異大，反映新老房東為兩種難度）。"
  "分類（空房率&gt;0.7）：Recall 0.877、Precision 0.574、F1 0.694。"
  "誠實 R²=0.2492 與前向選擇曲線第 35 步一致。", SMALL)

# ===== 二、曲線 =====
p("二、前向選擇邊際曲線（R² vs 特徵數）", H2)
img = Image("../../forward_selection_curve.png")
img._restrictSize(175 * mm, 95 * mm)
story.append(img)
p("三段結構：① 陡升段（7→11 個特徵，R² 0.06→0.22）→ ② 轉平（~13，0.244）→ "
  "③ 噪音平台（14→35，0.24–0.26 抖動）。第 11 個特徵即超越原 37 個 base 特徵（0.209）。", SMALL)

story.append(PageBreak())

# ===== 三、特徵重要度排名 =====
p("三、各特徵影響程度與排名（Permutation 重要度）", H2)
p("「影響程度」＝打亂該特徵後誠實 R² 的下降量；「佔比%」＝佔全部正貢獻比例；"
  "「SFS序」＝前向選擇加入順序。permutation 排名反映「完整模型中的獨特貢獻」，與加入順序不同。", SMALL)
story.append(Spacer(1, 4))

df = pd.read_csv("../../selected_35_features_importance.csv")
head = ["排名", "特徵（繁中）", "影響程度", "±std", "佔比%", "SFS序"]
rows = [head]
for _, r in df.iterrows():
    rows.append([str(int(r["排名"])), r["中文名稱"], f"{r['重要度(R²下降)']:+.4f}",
                 f"{r['std']:.4f}", f"{r['相對佔比%']:.1f}", str(int(r["SFS加入序"]))])
t = Table(rows, colWidths=[32, 150, 62, 50, 45, 42], repeatRows=1)
style = [
    ("FONTNAME", (0, 0), (-1, -1), FONT), ("FONTSIZE", (0, 0), (-1, -1), 8),
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2166ac")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
    ("ALIGN", (0, 0), (0, -1), "CENTER"), ("ALIGN", (2, 0), (-1, -1), "CENTER"),
    ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
]
# 前 6 名（累積佔比 ~60%）淡黃底
for i in range(1, 7):
    style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#fff5d6")))
# 佔比 ≤1% 的冗餘尾段淡灰
for i, (_, r) in enumerate(df.iterrows(), start=1):
    if r["相對佔比%"] <= 1.0:
        style.append(("TEXTCOLOR", (0, i), (-1, i), colors.HexColor("#999999")))
t.setStyle(TableStyle(style))
story.append(t)

# ===== 四、結論 =====
p("四、重點結論", H2)
for line in [
    "<b>1. 前 6 名吃掉約 60% 解釋力</b>：房東接受率(15.9%)、未來30天平均最短入住(12.1%)、"
    "最長入住晚數(10%)、鄰里內價格百分位、設施數量、價格。房東經營行為與入住規則比地段、評分更重要。",
    "<b>2. 天花板再確認</b>：誠實 R²≈0.25 / AUC≈0.74，與前幾輪一致；多加特徵無助突破。",
    "<b>3. POI 與 review 細項近乎冗餘</b>：全部 POI 特徵最高僅排第 13（500m內公車站數 2.6%）；"
    "可即時預訂、是否超讚房東影響度為 0。精簡到前 6–13 個特徵可保留幾乎全部效能。",
]:
    p(line, BODY)
    story.append(Spacer(1, 2))

# ---- 輸出 ----
doc = SimpleDocTemplate("前向特徵選擇實驗報告.pdf", pagesize=A4,
                        topMargin=15 * mm, bottomMargin=15 * mm,
                        leftMargin=16 * mm, rightMargin=16 * mm,
                        title="前向特徵選擇實驗報告")
doc.build(story)
print("→ 前向特徵選擇實驗報告.pdf")
