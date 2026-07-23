# -*- coding: utf-8 -*-
"""專案完整報告（07-01~07-15 全程）產成提報用 PDF（微軟正黑體）。
內容與 docs/專案完整報告_2026-07-15.md 同源，此為濃縮提報版。"""
import sys, os
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
from PIL import Image as PILImage
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                Image, PageBreak)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

pdfmetrics.registerFont(TTFont("JhengHei", "C:/Windows/Fonts/msjh.ttc", subfontIndex=0))
FONT = "JhengHei"
BLUE = colors.HexColor("#2166ac")
RED = colors.HexColor("#b2182b")

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontName=FONT, fontSize=17, leading=22, spaceAfter=6)
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontName=FONT, fontSize=12.5, leading=16,
                    spaceBefore=10, spaceAfter=4, textColor=BLUE)
BODY = ParagraphStyle("BODY", parent=styles["Normal"], fontName=FONT, fontSize=9.5, leading=14)
CELL = ParagraphStyle("CELL", parent=styles["Normal"], fontName=FONT, fontSize=8.5, leading=11.5)
SMALL = ParagraphStyle("SMALL", parent=styles["Normal"], fontName=FONT, fontSize=8, leading=11,
                       textColor=colors.HexColor("#666666"))
KEY = ParagraphStyle("KEY", parent=BODY, textColor=RED)

story = []


def p(text, style=BODY):
    story.append(Paragraph(text, style))


def tbl(rows, widths, header_bg="#2166ac", wrap_cols=None, fontsize=8.5):
    """rows[0] 為表頭；wrap_cols 指定哪些欄用 Paragraph 自動換行。"""
    wrap_cols = wrap_cols or []
    data = []
    for i, row in enumerate(rows):
        out = []
        for j, cell in enumerate(row):
            if i > 0 and j in wrap_cols:
                out.append(Paragraph(str(cell), CELL))
            else:
                out.append(str(cell))
        data.append(out)
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), FONT),
        ("FONTSIZE", (0, 0), (-1, -1), fontsize),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_bg)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f8")]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t)
    story.append(Spacer(1, 6))


def img(path, width_mm):
    w, h = PILImage.open(path).size
    story.append(Image(path, width=width_mm * mm, height=width_mm * h / w * mm))
    story.append(Spacer(1, 4))


# ===== 封面段 =====
p("智慧旅宿「空屋率風險」預警平台 — 專案完整報告", H1)
p("報告範圍 2026-07-01 ～ 07-15｜Inside Airbnb 台北 5,849 筆房源・1,296 位房東・21 萬則評論｜台北官方 POI 7 來源", SMALL)
story.append(Spacer(1, 8))

p("一、執行摘要", H2)
p("目標：預測房源未來一年空屋率（Y = availability_365/365），對高風險（&gt;70%）房源主動預警，"
  "並給出可解釋原因與優化建議。架構為雙模型：模型 A（迴歸，風險分數）＋ 模型 B（分類，通知決策），"
  "皆為 HistGradientBoosting（分類端加 isotonic 機率校準）。")
story.append(Spacer(1, 4))
tbl([
    ["評估方式", "迴歸 R²", "分類 AUC", "對應真實情境"],
    ["單次隨機切分 80/20", "0.587", "0.900", "平台上已有歷史的房東"],
    ["GroupKFold（依房東分組）", "0.209 ± 0.046", "0.710 ± 0.043", "全新房東的誠實泛化表現"],
], [125, 70, 70, 175], wrap_cols=[3])
p("本專案三大發現：① 單次切分高分部分來自「同房東多房源」的訓練/測試洩漏，GroupKFold 才是誠實基準；"
  "② 前向特徵選擇證明只需 11~13 個特徵即達誠實天花板 R²≈0.26，主特徵表大量冗餘；"
  "③ 兩輪外部 POI 實驗證明外部資料補不動完整模型，唯一價值在新房東冷啟動（交通特徵 R² +8%）。", KEY)

p("二、專案時間軸", H2)
tbl([
    ["日期", "階段", "主要產出"],
    ["07-01", "專案起始", "資料集建構"],
    ["07-06~07", "Y 定義收斂、特徵工程、影像多模態、雙模型、SHAP", "dataset_final.csv（37 特徵）、洩漏發現、8 項驗收"],
    ["07-07", "Office Hours 提報檢視", "4 項風險與修復優先序"],
    ["07-09", "外部特徵第一輪（OSM）", "冷啟動分組實驗"],
    ["07-15", "官方 POI 第二輪＋前向特徵選擇＋指標補充", "16 特徵消融、SFS 35 步、模型 A 分類指標"],
], [55, 205, 180], wrap_cols=[1, 2])

p("三、目標變數與防洩漏", H2)
p("Y 定義歷經三版演進：絕對雙門檻 → 同區同房型相對百分位（解決雅房 vs 整套房不公平）→ "
  "最終採連續空屋率 availability_365/365；分類任務衍生 Y&gt;0.7 為高風險（占比 31.4%）。"
  "凡用於定義 Y 的欄位一律禁入特徵（availability_*、reviews 系列等）；"
  "洩漏稽核：拿掉全部評分欄位重訓 AUC 僅 0.950→0.940，證明高分非洩漏。")

p("四、特徵工程（30 → 37 特徵）", H2)
tbl([
    ["結果", "特徵類別", "理由 / 效果"],
    ["排除", "交通/景點距離（初輪）、NLP 評論情緒（21 萬則）、具體設施旗標、房東認證、8/9 個影像特徵",
     "相關性過弱（情緒僅 +0.004）、台北普及率過高無鑑別度（冷氣 90%）、小樣本訊號全量不穩健"],
    ["採用", "自助入住、最長/最短入住晚數、描述/自介字數、回覆速度、競爭特徵 5 個、CLIP 設計感",
     "R² 0.561→0.587；photo_design_sense 為 9 個影像特徵中唯一有效者（前 10~15 名）"],
], [40, 220, 180], wrap_cols=[1, 2])
p("方法論：Tier1 批量測試 19 候選 → 精簡 6 個有效者，精簡版 R² 0.586 ≥ 全加版 0.581（Occam's Razor）。"
  "影像特徵以 OpenCV 客觀畫質 5 項＋CLIP-ViT-B/32 zero-shot 美感 3 項，全量 6,241 張封面照批次擷取。", SMALL)

story.append(PageBreak())

p("五、完整評估指標", H2)
p("5.1 單次隨機切分 80/20（老房東情境）", BODY)
tbl([
    ["迴歸模型", "MAE", "MSE", "RMSE", "R²"],
    ["基準 LinearRegression", "0.242", "0.0823", "0.287", "0.159"],
    ["HistGB 無影像（36 特徵）", "0.150", "0.0405", "0.201", "0.586"],
    ["HistGB 最終精簡版（37 特徵）", "0.151", "0.0405", "0.201", "0.587"],
    ["HistGB 全影像版（45 特徵）", "0.156", "0.0422", "0.206", "0.568"],
], [170, 55, 60, 60, 55])
p("MAE 與 MSE 不可互相換算（MSE 取平方、更放大離群值）；上表 MSE 為同模型直接量測，有 RMSE 者可驗證 MSE = RMSE²。", SMALL)
tbl([
    ["分類模型", "AUC", "PR-AUC", "Brier", "Recall", "Precision", "F1"],
    ["無影像（36 特徵）", "0.893", "0.776", "0.1204", "0.796", "0.662", "0.723"],
    ["最終精簡版（37 特徵）", "0.900", "—", "—", "0.804", "—", "0.737"],
    ["全影像版（45 特徵）", "0.905", "0.778", "0.1170", "0.828", "0.677", "0.745"],
], [140, 50, 50, 50, 50, 55, 45])
p("5.2 模型 A（迴歸）二值化分類指標（07-15 補充）", BODY)
p("迴歸預測值向平均收縮，若用真實定義 0.70 當門檻會漏抓一半高風險（Recall 僅 0.49~0.53）；"
  "實用門檻應設 0.55~0.60。", SMALL)
tbl([
    ["版本", "AUC", "PR-AUC", "門檻 0.60：Recall / Precision / F1"],
    ["36 特徵", "0.886", "0.808", "0.698 / 0.734 / 0.715"],
    ["45 特徵", "0.880", "0.807", "0.706 / 0.753 / 0.729"],
], [90, 55, 60, 200])
p("對照：模型 B 的 AUC（0.905）與 F1（0.745）較優，但迴歸切門檻的 PR-AUC 反而更高（0.808 vs 0.778）"
  "——若要精簡維運，可只留模型 A＋門檻 0.55~0.60 同時輸出分數與判定。", SMALL)

p("5.3 GroupKFold 誠實評估 — 關鍵方法論發現", BODY)
p("平均每位房東 4.48 筆房源，隨機切分讓同房東房源同落 train/test。個人房東（僅 1 筆）兩種評估幾乎不變"
  "（0.253 vs 0.210）；多房源房東由 0.626 崩落至 0.216 → 虛高確為洩漏所致。"
  "移除房東特徵（R² 反降至 0.181）、加強正則化（0.216）、RandomizedSearchCV 調參（+0.017）皆無法突破"
  "→ 瓶頸在資料資訊量與「新/老房東是兩個任務」的結構，非模型容量。", KEY)
p("5.4 公平性與驗收：房型 R² 0.578~0.743、六大行政區 AUC 0.865~0.947 無顯著劣化；"
  "8 項驗收標準 7 項通過，唯一例外（交叉驗證穩定度）即上述洩漏發現本身。", SMALL)

p("六、外部特徵兩輪實驗", H2)
tbl([
    ["實驗", "完整模型", "冷啟動模型（移除 7 個房東特徵）"],
    ["OSM（07-09）：飯店/景點/生活機能密度", "無顯著提升（0.209→0.204~0.217，誤差內）",
     "R² 0.181→0.192、AUC 0.715→0.720（小幅真實增益）"],
    ["台北官方 POI（07-15）：7 來源 16 特徵分組消融", "任何組合增益 ≤ +0.004（三度撞天花板）",
     "交通組最強：R² 0.181→0.196（+8%）；bus_count_500m 為最高新特徵（第 19 名）"],
], [135, 150, 155], wrap_cols=[0, 1, 2])
p("誠實界線：冷啟動 +0.015 仍小於折間 std（±0.047），屬「方向一致的小幅增益」而非突破；"
  "飯店/景點密度與空屋率負相關（地段熱度代理，非競爭懲罰）。決策：不寫回主資料集。", SMALL)

story.append(PageBreak())

p("七、前向特徵選擇（53 候選、GroupKFold 誠實協定）", H2)
p("每步試遍剩餘候選、選 R² 增益最大者。實測全程需 4.5~5 小時，跑至第 35 步時 R² 已在平台穩定抖動，"
  "經同意提前收尾（尾段為已知下滑，不影響結論）。")
img("../../forward_selection_curve.png", 165)
tbl([
    ["結論", "內容"],
    ["最小充分子集", "第 11 個特徵（score_pctl_nbhd, R²=0.220）即超越 base-37 的 0.209；實務充分規模 ≈ 11~13 個"],
    ["五大核心訊號", "review_scores_communication、host_acceptance_rate、min_nights_avg_ntm、latitude、score_pctl_nbhd（R² 0.06→0.22）"],
    ["誠實天花板", "R² ≈ 0.26（峰值 0.2627 @ 27 特徵，13 之後增益全為噪音）；POI 進榜但 Δ ≤ 0.01"],
    ["35 特徵重訓", "單次切分 R² 0.5034 / AUC 0.8721；GroupKFold R² 0.2492±0.0512 / AUC 0.7384±0.0223"],
], [90, 350], wrap_cols=[1])

p("八、可解釋性：SHAP 與 EBM", H2)
p("SHAP 全域 Top15 與 permutation importance 交叉一致（Top1：host_acceptance_rate）；"
  "maximum_nights 呈非線性（200 晚以下風險隨值降低而下降）。"
  "EBM（純加性 GAM）在完全可解釋前提下：迴歸 R² 0.3735、分類 AUC 0.8120（保有黑盒主力 0.8784 的 93%）。")
tbl([
    ["特徵", "EBM 非線性發現", "UI 動態調優建議"],
    ["minimum_nights", "2~3 晚最優（-3.5~-3.8 pp）；7 晚以上轉正（14 晚 +3.8 pp）", "「調到 3 晚預計降 7.35%」What-if 查表即得"],
    ["photo_design_sense", "&lt;0.3 時 +9.8 pp；&gt;0.5 階梯轉負", "封面照美感有實質預訂效益，建議重拍"],
    ["price", "突破商圈合理水平臨界點後風險躍升", "結合 price_pctl_nbhd 給相對定價建議"],
], [95, 190, 155], wrap_cols=[1, 2])

p("九、結論", H2)
tbl([
    ["情境", "建議模型", "誠實預期"],
    ["老房東（已有紀錄）", "37 特徵雙模型（或精簡 11~13 特徵版）", "R² 0.59 / AUC 0.90（單次切分口徑）"],
    ["新房東（冷啟動）", "30 特徵版＋交通 POI 4 欄", "R² ≈ 0.19~0.20 / AUC ≈ 0.72"],
    ["UI 呈現", "風險分數＋信心等級＋EBM What-if 建議", "提升可信度與可行動性"],
], [110, 180, 150], wrap_cols=[1, 2])
p("提報立場：主動揭露 GroupKFold 保守數字（0.21/0.71）並解釋其對應情境——三輪外部特徵實驗與前向選擇"
  "皆收斂到同一結構性天花板，「知道極限在哪、也知道為什麼」是本專案相對單一樂觀數字最大的嚴謹度優勢。", KEY)
p("後續方向（依潛力排序）：① 跨城市擴大資料規模；② 新房東替代訊號（首次定價、專業攝影）；"
  "③ 房東訪談驗證需求真實性；④ CLIP 提示詞工程。", SMALL)
p("完整詳細版：docs/專案完整報告_2026-07-15.md（含全部數字出處與檔案索引）", SMALL)

doc = SimpleDocTemplate("專案完整報告_2026-07-15.pdf", pagesize=A4,
                        leftMargin=18 * mm, rightMargin=18 * mm,
                        topMargin=16 * mm, bottomMargin=16 * mm,
                        title="智慧旅宿空屋率風險預警平台 — 專案完整報告")
doc.build(story)
print("OK", os.path.getsize("專案完整報告_2026-07-15.pdf"), "bytes")
