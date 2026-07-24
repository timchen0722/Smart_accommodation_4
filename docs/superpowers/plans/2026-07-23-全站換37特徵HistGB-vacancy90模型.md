# 全站換模：37 特徵 · HistGradientBoosting · Y=vacancy_90

> 2026-07-23 · Smart_accommodation_4 · brainstorming 決議後定案
> 觸發：使用者要求「數據分析分頁重新排版」→ 擴大為「全站換成新模型」。

## 一、決議（使用者拍板）

| 項目 | 決議 |
|---|---|
| 模型 | HistGradientBoosting（回歸＋分類），Isotonic 校準，XGBoost 對照 |
| 特徵 | 37 核心特徵（root `dataset_final.csv`，`FEATURES=[c for c in cols if c not in {listing_id,Y_vacancy} and (not photo_* or photo_design_sense)]`） |
| 主目標 Y | `Y_vacancy_90 = availability_90/90`；高風險 = `vacancy_90 > 0.70`（基準率 37.4%） |
| 營收口徑 | **雙輸出**：同 37 特徵再訓一個 `vacancy_365` 回歸器，年營收 = 房價×(1−vac365)×365；風險用 vac90 |
| 冷啟動/對照 | **保留架構**：37 特徵冷啟動變體（移除 7 個 HOST_IDENTITY）＋ XGBoost 對照 |
| 警報門檻 | 紅 0.60 / 黃 0.35（校準機率，沿用） |

## 二、真實數字（已跑 `eval_vacancy_90_models.py` 佐證，N=5,849）

| 指標 | 單次切分(樂觀) | GroupKFold(誠實) |
|---|---|---|
| 90天 分類 AUC | 0.840 | **0.755** |
| 90天 迴歸 R² | 0.409 | **0.195** |
| 365天 分類 AUC | 0.898 | 0.717 |
| 365天 迴歸 R² | 0.574 | 0.190 |

90天 Top 特徵(permutation)：host_acceptance_rate .112 / price_pctl_nbhd .080 / host_listings_count .075 / min_nights_avg_ntm .040 / price .033 …

## 三、牽動範圍（唯讀盤點結果）

- `_predictions.csv` 讀者：feature_engineering, notify_center, platform_sections, 房東入口, 後台分析
- `backend_models_v2.joblib` 讀者：backend_v2_sections, feature_engineering, lime_explainer, vacancy_model, 房東入口
- 語意地雷：① `platform_analytics.py:32` 年營收×365；② 高風險定義 Y≥0.6(365) → vac90>0.70

## 四、分階段執行（每階段獨立驗收；破壞性前先備份 *-backup-2026-07-23）

### Phase 1 — 數據分析分頁重設計（低風險，先交付）✅ 完成 2026-07-23
- [x] `scripts/04_eval/build_data_analysis_json.py` → `models/eval_vacancy_90.json`（誠實 90天 AUC 0.755±0.015 / R² 0.195±0.054；單次 AUC 0.840 / R² 0.409；紅警報 P 0.723；importance）
- [x] `modules/data_analysis_sections.py`：平鋪四段讀 JSON
- [x] `pages/3_📊_後台分析.py` t_model 改呼叫 `render_data_analysis()`（移除舊三子分頁）
- **驗收 ✅**：AppTest 單段 0 例外＋內容含 37/HistGB/0.755/0.195；整頁 AppTest 0 例外（652 markdown，5 分頁全渲染）

### Phase 2 — 訓練新 bundle + 全站預測（核心，破壞性）✅ 完成
- [x] 備份 5 產物 → `*-backup-2026-07-23`
- [x] **重大發現**：root dataset_final ≠ 上線 dataset_multimodal（host_acceptance_rate 一致率僅 2%…）→ 決議「在上線資料上重訓」避免 train-serve skew；橋接 2 欄 → `data/_core_extra.csv`
- [x] `scripts/train_backend_models_v90.py`：HistGB、37特徵(bathrooms→bathrooms_count)、雙 reg(vac90+vac365)、xgb對照、cold=31、Isotonic
- [x] bundle 加 `reg_model_365`；`_predictions.csv` 加 `vac_pred_365`
- **驗收 ✅**：full 誠實 AUC 0.755±0.021/R² 0.200±0.032、cold AUC 0.736；bundle=HistGB 37/31；predictions 15 欄 0 NaN（red 968/yellow 1734）

### Phase 3 — 語意接線 rewire ✅ 完成
- [x] `platform_analytics.add_revenue_columns` 營收改用 `vac_pred_365`（缺欄回退 vac_pred）
- [x] `feature_engineering.load_dataset_final` + `vacancy_model.load_data` 併入 `_core_extra`（否則 predict_risk_v2/get_models KeyError）
- [x] 顯示字串：後台側欄、房東入口 caption、ALGO=histgb、docstrings → HistGB/37/vacancy_90>0.70
- **驗收 ✅**：predict_risk_v2 full/cold/xgb 皆正常、調價 what-if 有反應、營收確用 vac_pred_365；後台 caption 無殘留 59

### Phase 4 — SHAP 快取重算 ✅ 完成
- [x] 訓練腳本內以 `shap.TreeExplainer(HistGB reg_model)` 重算 full/cold → shap_cache.joblib（TreeExplainer 支援 HistGB，已測）
- **驗收 ✅**：房東入口 AppTest 0 例外（local SHAP 走 live TreeExplainer）

### Phase 5 — 全站驗證 ✅ 完成
- **驗收 ✅**：`pytest` 28 passed；AppTest 後台/房東/租客/首頁 全 0 例外（markdown 652/301/43/1）

## 五、狀態
- 2026-07-23 建檔並完成 Phase 1–5。全站已上線 37特徵/HistGB/vacancy_90（雙輸出）。
- 後續可選：doc/02、doc/07 文件更新（描述仍寫 LightGBM/59特徵，屬文件債）；backend_v2_sections.py 已無 live 呼叫者（死碼）。
