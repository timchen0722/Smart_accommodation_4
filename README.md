# 智慧旅宿空屋率風險預警平台 v4

以台北市 Airbnb 房源為主體(架構與色系沿用 Smart_accommodation_2 日系簡約風),
融合 591 / Booking.com / 租租網 1km 跨平台競品比對、LightGBM + XGBoost 雙模型、
雙層警報制與 SHAP 可解釋性。

## 部署到 Streamlit Community Cloud

1. **推上 GitHub**(在專案根目錄):

   ```powershell
   git init
   git add .
   git commit -m "smart accommodation v4"
   git remote add origin https://github.com/<你的帳號>/<repo>.git
   git push -u origin main
   ```

   `.gitignore` 已排除:`__pycache__`、訓練暫存檔、**`.streamlit/secrets.toml`(金鑰,絕不可上傳)**、`doc/*.mp4`(討論影片,體積大且不需部署)。
   `models/` 與 `data/` 的訓練產物**需要一併上傳**(雲端不重訓);單檔皆 < 100MB,GitHub 可直接收。

2. **建立 App**:到 [share.streamlit.io](https://share.streamlit.io) → New app → 選 repo/branch → **Main file path 填 `index.py`** → Advanced settings 選 Python 3.11+。

3. **設定金鑰**(LLM 智慧建議用):App settings → Secrets 貼上:

   ```toml
   GEMINI_API_KEY = "AIza..."
   # 或 ANTHROPIC_API_KEY = "sk-ant-..."
   ```

4. 部署相依已備妥:`requirements.txt`(**版本已釘選**,與 models/ 內 pkl 訓練版本一致,勿隨意升版)與 `packages.txt`(`fonts-noto-cjk`,讓雲端 Linux 的 SHAP/matplotlib 圖正確顯示繁中)。

> 注意:models/ 的 joblib 以 scikit-learn 1.7.2 / LightGBM 4.7.0 / XGBoost 3.2.0 訓練;
> 若改動 requirements 版本導致載入失敗,請本機重跑 `python -X utf8 scripts/train_backend_models.py` 後再 push。

## 快速開始

```powershell
pip install -r requirements.txt
python -X utf8 scripts/train_backend_models.py   # 重建 models/ 全部訓練產物(約 1 分鐘)
streamlit run index.py                            # 啟動平台(動畫首頁)
```

## 頁面

| 頁面 | 內容 |
|---|---|
| index.py 首頁 | 全版動畫 Hero(各行政區房源照片牆 + 即時統計) |
| 房東入口 | **房東 Dashboard 四視圖**:總覽(房源卡片/風險環/趨勢箭頭)、詳情(LIME Top3/LLM 智慧建議/價格趨勢線)、附近比較(熱力圖/同商圈排名/跨平台競品)、通知中心(60% 門檻/紀錄/已處理) |
| 租客入口 | 找房、地圖、比價(沿用 v2) |
| 後台分析 | 策略沙盒 + 模型誠實評估(LightGBM vs XGBoost)+ SHAP 可解釋性 |

## 模型口徑(2026-07-19 決議)

- **特徵**:59 特徵(44 核心 + POI 11 + NLP 4,含**負評比例 neg_review_ratio**);冷啟動變體 52 特徵
- **標籤**:高風險 = 未來一年空屋率 ≥ 0.6(基準率 37.4%)
- **模型**:LightGBM(主力)+ XGBoost(對照),皆 Isotonic 校準;房東入口側欄可切換
- **雙層警報**(GroupKFold 誠實 OOF 實測):
  - 🔴 紅色警報:校準機率 ≥ 0.60 → Precision ≈ 0.69 / Recall ≈ 0.27
  - 🟡 黃色觀察:校準機率 ≥ 0.35 → 整體 Recall ≈ 0.70
- **誠實評估**:GroupKFold(host_id, 5 折),AUC ≈ 0.716、R² ≈ 0.24;依據見
  `doc/01_資料分析報告_threshold06.md`

## 訓練產物(models/)

`backend_models_v2.joblib`(full/cold × LGBM/XGB + 雙門檻)· `eval_results.json` ·
`shap_cache.joblib` · `competitor_index.pkl`(四平台 BallTree)· `suggestion_engine.pkl`

另產出 `data/_predictions.csv`(全房源 OOF 誠實預測:熱力圖/排名/通知中心用)與
`data/_nlp_extra.csv`(負評比例特徵快取)。

跨平台競品與建議引擎為可抽換 PKL 模組(`modules/pkl_store.py`),可單獨重建替換。
591/Booking/ddroom 僅作 1km 競品比對,不進模型。

## 可解釋性與智慧建議

- **LIME**(房東端):`modules/lime_explainer.py` 解釋單一房源 P(空屋率≥60%) 的 Top-3 原因
- **SHAP**(研究端):後台分析頁全域蜂群/瀑布圖
- **LLM 智慧建議**:高風險(紅/黃)觸發時,由 LLM 依 LIME 原因 + 跨平台競品數據生成
  個人化建議。設定金鑰後自動啟用,無金鑰時退回規則引擎:
  - `.streamlit/secrets.toml` 加入 `ANTHROPIC_API_KEY = "sk-..."`(Claude)
  - 或 `GEMINI_API_KEY = "..."`(Gemini);亦可用同名環境變數
