# 智慧旅宿「空屋率風險預警與策略沙盒平台」網頁應用程式規格書 (STREAMLIT_APP_SPEC.md)

本文件詳述了本專案 Streamlit 互動網頁前端應用程式（Web App）的 UI/UX 設計、佈局結構、CSS 樣式注入、前後端 API 整合、以及圖表渲染邏輯。

---

## 1. 專案環境與依賴 (Environment & Dependencies)
*   **前端框架**：Streamlit 1.30.0+ / 1.59.0+
*   **啟動檔案路徑**：`smartaccommodation_imp_new\app\main.py`
*   **後端依賴模組**：
    *   `src.inference_api`：負責執行沙盒模擬即時推論。
    *   `app.components_dashboard`：儀表板視覺化組件。
    *   `app.components_sandbox`：沙盒控制與 SHAP 渲染組件。
*   **啟動指令**：
    ```powershell
    streamlit run app/main.py
    ```

---

## 2. 頁面布局與視覺設計 (Page Layout & UI/UX Aesthetics)
為創造極具專業感且令人驚艷的第一印象，本平台採用了**深色系極簡高質感設計（Premium Dark Mode UI）**，搭配**玻璃擬物化卡片（Glassmorphism Cards）**及漸層效果。

### 2.1 CSS 樣式注入 (Custom CSS Injection)
在 `app/main.py` 中，我們藉由 `st.markdown(..., unsafe_allow_html=True)` 注入了自訂 CSS，覆寫 Streamlit 的預設樣式：
*   **字型載入**：引進 Google Fonts 的 `Inter`（用於一般內文與數字）與 `Outfit`（用於主要大標題，呈現高端感）。
*   **漸層標題**：主要標題統一採用漸層藍綠色（`linear-gradient(135deg, #3B82F6 0%, #10B981 100%)`）做為文字前景。
*   **玻璃卡片容器**：自訂 `.premium-card` class，設定背景色為 `#1F2937` (深灰)、邊框 `#374151`，並加入微距陰影與 `transform 0.2s ease` 的懸停（hover）微動畫，使頁面具有動態互動感。
*   **側邊欄暗化**：覆寫 `stSidebar` 背景為 `#111827` (黑灰色) 並加上與主欄的區隔線。

### 2.2 左右雙欄佈局 (Two-Column Layout)
主要操作區域採用 `st.columns([1, 1], gap="large")` 左右對稱排版：
*   **左側：策略沙盒與經營控制欄 (Sandbox Control)**
*   **右側：AI 診斷預警與指標評估欄 (AI Diagnosis & Evaluation)**

---

## 3. UI 元件與互動邏輯 (UI Components & Logic)

### 3.1 側邊欄控制中心 (Sidebar Control Center)
1.  **房東選擇器**：
    *   調用 `api.get_unique_hosts()`，取得所有房東 ID。
    *   下拉選單顯示：`房東 ID: {host_id} (名下 {N} 間房源)`，依房源多寡降序排列。
2.  **房源選擇器**：
    *   點選房東後，即時觸發 `api.get_host_properties(host_id)`。
    *   下拉選單列出該房東名下的所有房源，顯示：`房源 #{id} ({行政區} - {房型})`。
3.  **技術堆疊快照**：側邊欄底部顯示當前平台採用的算法與特徵狀態。

---

### 3.2 頂部資訊區與預測信心標籤 (Confidence Badges)
呈現當前選定房源的行政區、房型與詳細 GPS 經緯度座標。
*   **信心等級標籤判定邏輯** (由 `components_dashboard.py` 實現)：
    *   **極高信心（綠色漸層 Badge）**：若該房源之房東歷史經營房源總數 `host_listings_count > 1` 或歷史已有評分（即 `review_scores_rating` 非空）。
    *   **中等信心（黃色漸層 Badge）**：對於新加入平台、無任何評價的房源（冷啟動狀態），顯示「預測信心：中等 (基於大數據估算)」。

---

### 3.3 左欄：策略沙盒控制器 (Sandbox Controls)
1.  **鎖定唯讀歷史數據**：
    *   以 `st.text_input(..., disabled=True)` 呈現房源無法被房東任意改變的歷史客觀屬性（「所在行政區」、「清潔度評分」、「性價比評分」、「溝通評分」）。
    *   **新增機能快照 (Premium Snapshot)**：額外渲染一個 HTML 區塊，呈現本專案核心的**空間 POI 便利性數據**（如最近捷運公尺數、500公尺內超商數、公園綠地面積等）與 **NLP 評論情感得分**，增強房東對大數據機能的信賴度。
2.  **動態調整滑桿區**：
    *   **每晚房價 (NTD $)**：以 `st.slider` 控制，設有防呆下限 $500 與上限 $50,000。
    *   **最低入住天數限制 (晚)**：以 `st.number_input` 進行加減計數 (1 至 30 晚)。
    *   **客服回覆時間**：下拉選單（對應序數 1 到 4，轉換為 1小時內、數小時內等）。
    *   **描述文字模擬區**：以 `st.text_area` 供使用者輸入，前端自動計算字數 `desc_len`。

---

### 3.4 右欄：預警儀表板與 SHAP 解釋 (Outputs)

1.  **半圓形風險儀表板 (Risk Gauge)**：
    *   由 `components_dashboard.py` 之 `render_risk_gauge` 函數渲染。
    *   使用精美 SVG 繪製，利用 CSS 漸層色帶做背景。
    *   **動態進度弧**：計算空屋率與弧長的比例，動態繪製進度弧。
    *   **警報聯動**：
        *   空屋率 `< 40%`：綠色，顯示「低風險」。
        *   `40% ~ 69%`：黃色，顯示「中風險」。
        *   `>= 70%`：紅色，顯示「高風險」，且觸發 **模型 B** 的機率顯示（高空屋機率），並聯動 CSS 呼吸燈閃爍動畫。
2.  **AI 房源智能診斷報告 (Action Items)**：
    *   由 `components_sandbox.py` 之 `render_ai_diagnosis` 函數實現。
    *   後端篩選出 SHAP 貢獻值大於 0（$\phi_i > 0$，即推高空屋風險的因子）進行排序，取 **Top 2** 扣分項。
    *   **白話文映射**：整合 50 維特徵的規則引擎，針對價格百分位高、最低天數長、回覆速度慢、照片感低、評論情感負面、交通不便（捷運遠）等，輸出具備商務調整步驟的優化建議書。
3.  **SHAP 水瀑條形圖 (SHAP Contributions Chart)**：
    *   以自訂 HTML/CSS 繪製居中雙向對稱條形圖。
    *   以 `50%` 為零點中軸線，**綠色橫條向左** 代表降低風險（加分項），**紅色橫條向右** 代表提高風險（扣分項）。
    *   滑鼠懸浮在特徵名稱上可顯示完整中文名稱翻譯（如：`price_pctl_nbhd` 翻譯為 `同區同房型價格百分位排名`）。
