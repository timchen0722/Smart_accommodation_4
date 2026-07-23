# -*- coding: utf-8 -*-
"""
將 docs/專案完整報告_2026-07-15.md 轉換為 HTML 並透過 Headless Chrome 列印為 PDF 檔。
會自動將專案中的關鍵圖表（如前向選擇曲線、SHAP 影響圖、EBM 非線性曲線）動態嵌入至對應段落中。
"""
import os
import sys
import subprocess
from markdown_it import MarkdownIt

def main():
    # 確保輸出編碼為 UTF-8
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    # 使用絕對路徑以避免 Chrome 找不到路徑
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
    md_path = os.path.join(base_dir, "docs", "專案完整報告_2026-07-15.md")
    html_path = os.path.join(base_dir, "docs", "專案完整報告_2026-07-15.html")
    pdf_path = os.path.join(base_dir, "docs", "專案完整報告_2026-07-15.pdf")

    if not os.path.exists(md_path):
        print(f"錯誤: 找不到 Markdown 報告檔案: {md_path}")
        sys.exit(1)

    print(f"正在讀取 Markdown 報告: {md_path}...")
    with open(md_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 解析標題與主體
    lines = text.split("\n")
    title = "智慧旅宿「空屋率風險」預警平台 — 專案完整報告"
    if lines and lines[0].startswith("# "):
        title = lines[0].replace("# ", "").strip()
        lines = lines[1:]
    text_body = "\n".join(lines)

    # ==================== 動態嵌入圖表 ====================
    print("正在嵌入專案關鍵圖表...")
    
    # 1. 前向選擇邊際曲線
    curve_placeholder = "**曲線三段結構**："
    curve_replacement = (
        "**曲線三段結構**：\n\n"
        '<div style="text-align: center; margin: 20px 0; page-break-inside: avoid;">\n'
        '  <img src="../forward_selection_curve.png" alt="前向特徵選擇邊際曲線" style="width: 85%; max-width: 600px; border: 1px solid #e2e8f0; border-radius: 6px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);" />\n'
        '  <div style="font-size: 9pt; color: #718096; margin-top: 8px; font-weight: 500;">前向選擇邊際曲線（R² vs 特徵數）</div>\n'
        "</div>\n"
    )
    if curve_placeholder in text_body:
        text_body = text_body.replace(curve_placeholder, curve_replacement)
        print("已嵌入: 前向選擇邊際曲線")
    else:
        print("警告: 找不到前向選擇曲線的插入點，略過。")

    # 2. SHAP 影響圖
    shap_placeholder = "產出：beeswarm / bar / scatter / waterfall（單房源原因解釋範例）"
    shap_replacement = (
        "產出：beeswarm / bar / scatter / waterfall（單房源原因解釋範例）\n\n"
        '<div class="image-row">\n'
        '  <div class="image-col">\n'
        '    <img src="../shap_beeswarm.png" alt="SHAP Beeswarm" />\n'
        '    <div class="image-caption">SHAP Beeswarm 特徵分佈影響圖</div>\n'
        "  </div>\n"
        '  <div class="image-col">\n'
        '    <img src="../shap_bar.png" alt="SHAP Bar" />\n'
        '    <div class="image-caption">SHAP Bar 特徵重要度平均圖</div>\n'
        "  </div>\n"
        "</div>\n"
    )
    if shap_placeholder in text_body:
        text_body = text_body.replace(shap_placeholder, shap_replacement)
        print("已嵌入: SHAP 影響圖")
    else:
        print("警告: 找不到 SHAP 圖表的插入點，略過。")

    # 3. EBM 非線性曲線
    ebm_placeholder = "**非線性曲線的商業轉譯（UI「動態調優建議」）**："
    ebm_replacement = (
        "**非線性曲線的商業轉譯（UI「動態調優建議」）**：\n\n"
        '<div class="image-grid-3">\n'
        '  <div class="image-col">\n'
        '    <img src="../ebm_curve_minimum_nights.png" alt="EBM Minimum Nights" />\n'
        '    <div class="image-caption">minimum_nights EBM 曲線</div>\n'
        "  </div>\n"
        '  <div class="image-col">\n'
        '    <img src="../ebm_curve_photo_design_sense.png" alt="EBM Photo Design Sense" />\n'
        '    <div class="image-caption">photo_design_sense EBM 曲線</div>\n'
        "  </div>\n"
        '  <div class="image-col">\n'
        '    <img src="../ebm_curve_price.png" alt="EBM Price" />\n'
        '    <div class="image-caption">price EBM 曲線</div>\n'
        "  </div>\n"
        "</div>\n"
    )
    if ebm_placeholder in text_body:
        text_body = text_body.replace(ebm_placeholder, ebm_replacement)
        print("已嵌入: EBM 非線性曲線")
    else:
        print("警告: 找不到 EBM 曲線的插入點，略過。")

    # ==================== HTML 轉換與渲染 ====================
    print("正在轉換 Markdown 至 HTML...")
    md = MarkdownIt("commonmark", {"html": True}).enable("table")
    html_rendered = md.render(text_body)

    # 封裝為精美 HTML
    html_template = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
  <meta charset="UTF-8">
  <title>{title}</title>
  <style>
    @page {{
      size: A4;
      margin: 20mm 18mm 25mm 18mm;
    }}
    
    body {{
      font-family: "Microsoft JhengHei", "微軟正黑體", "Inter", system-ui, -apple-system, sans-serif;
      color: #2d3748;
      line-height: 1.7;
      font-size: 10.5pt;
      margin: 0;
      padding: 0;
      background-color: #ffffff;
      -webkit-print-color-adjust: exact;
      print-color-adjust: exact;
    }}
    
    /* ====== Cover Page (block layout, no flex) ====== */
    .cover-page {{
      page-break-after: always;
      padding: 40mm 25mm 20mm 25mm;
      background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
      position: relative;
      overflow: hidden;
    }}
    .cover-page::before {{
      content: "";
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      height: 8px;
      background: linear-gradient(90deg, #2b6cb0 0%, #c53030 100%);
    }}
    .cover-page::after {{
      content: "";
      position: absolute;
      bottom: 0;
      left: 0;
      right: 0;
      height: 3px;
      background: linear-gradient(90deg, #2b6cb0 0%, #c53030 100%);
    }}
    .cover-header {{
      margin-bottom: 35px;
    }}
    .cover-tag {{
      display: inline-block;
      background-color: #2b6cb0;
      color: white;
      padding: 6px 18px;
      font-size: 11pt;
      font-weight: bold;
      border-radius: 20px;
      letter-spacing: 2px;
    }}
    /* Override: cover h1 不繼承 page-break-before */
    h1.cover-title {{
      font-size: 28pt;
      color: #1a365d;
      margin: 0 0 15px 0;
      font-weight: 800;
      line-height: 1.35;
      letter-spacing: 0.5px;
      border-bottom: none;
      page-break-before: auto !important;
      page-break-after: auto;
    }}
    .cover-subtitle {{
      font-size: 15pt;
      color: #4a5568;
      margin: 0 0 35px 0;
      font-weight: 400;
    }}
    .cover-divider {{
      width: 70px;
      height: 5px;
      background-color: #c53030;
      margin-bottom: 40px;
      border-radius: 2px;
    }}
    .cover-meta {{
      background-color: #ffffff;
      border: 1px solid #e2e8f0;
      border-left: 5px solid #2b6cb0;
      border-radius: 6px;
      padding: 24px;
      text-align: left;
      max-width: 520px;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
    }}
    .meta-item {{
      font-size: 11pt;
      color: #4a5568;
      margin-bottom: 10px;
      line-height: 1.6;
    }}
    .meta-item:last-child {{
      margin-bottom: 0;
    }}
    .meta-item strong {{
      color: #1a365d;
    }}
    .cover-footer {{
      margin-top: 40px;
      font-size: 9pt;
      color: #a0aec0;
      text-align: right;
    }}
    
    /* ====== Main Content ====== */
    .content {{
      padding: 0;
    }}
    
    /* Headings */
    h1, h2, h3, h4 {{
      color: #1a365d;
      font-weight: 700;
      page-break-inside: avoid;
      page-break-after: avoid;
    }}
    h1 {{
      font-size: 17pt;
      border-bottom: 2.5px solid #2b6cb0;
      padding-bottom: 8px;
      margin-top: 30px;
      margin-bottom: 16px;
      page-break-before: always;
    }}
    /* 第一個 h2（= 0. 執行摘要）不需分頁 */
    .content > h2:first-child {{
      page-break-before: avoid;
      margin-top: 0;
    }}
    h2 {{
      font-size: 13pt;
      color: #2b6cb0;
      margin-top: 26px;
      margin-bottom: 12px;
      border-left: 4px solid #c53030;
      padding-left: 10px;
    }}
    h3 {{
      font-size: 11pt;
      color: #2d3748;
      margin-top: 20px;
      margin-bottom: 8px;
    }}
    
    p {{
      margin-top: 0;
      margin-bottom: 12px;
      text-align: justify;
    }}
    
    strong {{
      color: #1a365d;
    }}
    
    /* Lists */
    ul, ol {{
      margin-top: 0;
      margin-bottom: 12px;
      padding-left: 22px;
    }}
    li {{
      margin-bottom: 5px;
      text-align: justify;
    }}
    
    /* Blockquotes */
    blockquote {{
      border-left: 4px solid #2b6cb0;
      background-color: #ebf8ff;
      padding: 12px 18px;
      margin: 16px 0;
      color: #2d3748;
      border-radius: 0 6px 6px 0;
      font-size: 9.5pt;
    }}
    blockquote p {{
      margin: 4px 0;
    }}
    
    /* Tables */
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 16px 0;
      font-size: 9.5pt;
      page-break-inside: auto;
    }}
    tr {{
      page-break-inside: avoid;
      page-break-after: auto;
    }}
    th, td {{
      border: 1px solid #cbd5e0;
      padding: 7px 10px;
      text-align: left;
      vertical-align: middle;
    }}
    th {{
      background-color: #2b6cb0;
      color: white;
      font-weight: bold;
      font-size: 9pt;
    }}
    tr:nth-child(even) td {{
      background-color: #f7fafc;
    }}
    
    /* Code */
    code {{
      background-color: #edf2f7;
      color: #c53030;
      padding: 1px 4px;
      border-radius: 3px;
      font-family: Consolas, Monaco, "Courier New", monospace;
      font-size: 0.88em;
    }}
    pre {{
      background-color: #f7fafc;
      border: 1px solid #e2e8f0;
      padding: 10px 14px;
      border-radius: 6px;
      margin: 14px 0;
      page-break-inside: avoid;
    }}
    pre code {{
      background-color: transparent;
      color: inherit;
      padding: 0;
      border-radius: 0;
      font-size: 9pt;
      white-space: pre-wrap;
      word-wrap: break-word;
    }}
    
    /* Image Grid & Rows */
    .image-row {{
      display: flex;
      justify-content: space-between;
      gap: 15px;
      margin: 15px 0;
      page-break-inside: avoid;
    }}
    .image-col {{
      flex: 1;
      text-align: center;
      display: flex;
      flex-direction: column;
      align-items: center;
    }}
    .image-col img {{
      width: 100%;
      border: 1px solid #e2e8f0;
      border-radius: 6px;
      box-shadow: 0 4px 6px rgba(0, 0, 0, 0.03);
    }}
    .image-grid-3 {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      margin: 15px 0;
      page-break-inside: avoid;
    }}
    .image-grid-3 .image-col {{
      flex: 1;
    }}
    .image-caption {{
      font-size: 8.5pt;
      color: #718096;
      margin-top: 6px;
      font-weight: 500;
    }}
    
    hr {{
      border: 0;
      height: 1px;
      background: #e2e8f0;
      margin: 25px 0;
      page-break-after: avoid;
    }}
    
    /* Page footer (printed via CSS) */
    .page-footer {{
      position: fixed;
      bottom: 0;
      left: 0;
      right: 0;
      height: 15mm;
      display: flex;
      justify-content: space-between;
      align-items: flex-end;
      padding: 0 18mm 5mm 18mm;
      font-size: 8pt;
      color: #a0aec0;
      border-top: 1px solid #e2e8f0;
    }}
  </style>
</head>
<body>
  
  <div class="cover-page">
    <div class="cover-header">
      <span class="cover-tag">專案總結報告</span>
    </div>
    <div>
      <h1 class="cover-title">智慧旅宿「空屋率風險」預警平台</h1>
      <p class="cover-subtitle">專案完整報告 (07-01 ～ 07-15 全程)</p>
      <div class="cover-divider"></div>
    </div>
    <div class="cover-meta">
      <div class="meta-item"><strong>報告範圍：</strong>2026-07-01 ～ 2026-07-15</div>
      <div class="meta-item"><strong>資料來源：</strong>Inside Airbnb 台北資料集 &amp; 台北官方 POI (7 大來源)</div>
      <div class="meta-item"><strong>資料規模：</strong>5,849 筆房源、1,296 位房東、21 萬則評論</div>
      <div class="meta-item"><strong>開發架構：</strong>雙模型（HistGradientBoosting 迴歸＋機率校準分類）</div>
    </div>
    <div class="cover-footer">2026-07-15</div>
  </div>

  <div class="content">
    {html_rendered}
  </div>

</body>
</html>
"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_template)
    print(f"HTML 暫存檔已生成 (絕對路徑): {html_path}")

    # ==================== 呼叫 Chrome 進行列印 ====================
    chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    if not os.path.exists(chrome_path):
        print(f"錯誤: 找不到 Chrome 主程式於: {chrome_path}")
        sys.exit(1)

    print("正在啟動 Headless Chrome 進行 PDF 列印 (使用絕對路徑)...")
    # 使用 --no-pdf-header-footer 移除 Chrome 預設的日期/URL 頁首頁尾
    args = [
        chrome_path,
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        "--run-all-compositor-stages-before-draw",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_path}",
        html_path
    ]

    try:
        subprocess.run(args, check=True)
        print("==================================================")
        print(f"PDF 報告轉換成功!")
        print(f"輸出路徑: {pdf_path}")
        
        # 驗證 PDF 檔案是否生成且大小大於 0
        if os.path.exists(pdf_path):
            print(f"檔案大小: {os.path.getsize(pdf_path)} bytes")
            print("==================================================")
            # 清理 HTML 暫存檔
            if os.path.exists(html_path):
                os.remove(html_path)
                print(f"已自動清理 HTML 暫存檔。")
        else:
            print("錯誤: PDF 轉換命令已執行，但找不到生成的 PDF 檔案。")
            sys.exit(1)
            
    except subprocess.CalledProcessError as e:
        print(f"列印 PDF 失敗，Chrome 錯誤碼: {e.returncode}")
        sys.exit(1)
    except Exception as e:
        print(f"列印 PDF 失敗，發生未預期錯誤: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
