# models/ 目錄下四大核心權重檔 (.pkl) 詳盡白話解析說明書

在專案目錄下的 `models/` 資料夾中，有 4 個以 `.pkl` 為副檔名的二進位檔案。這些檔案是我們專案的**「AI 核心大腦資產」**。

本文件使用最淺顯易懂的白話文，為你解釋這些檔案分別代表什麼意義、是在哪一個執行步驟中被產生的，以及它們在網頁中的作用與具體的程式碼實現。

---

## 1. 什麼是 .pkl 檔案？
*   **白話解釋**：`.pkl` 是 Python 中用來**「把物件打包冷凍存檔」**的檔案格式（稱為 Pickle 序列化）。
*   **作用**：AI 訓練需要花時間去運算並記住規律。訓練完成後，我們用 `joblib.dump` 把 AI 的記憶和翻譯器打包存成實體檔案。網頁啟動時，只要花不到 0.5 秒用 `joblib.load` 把這些檔案「解凍」載入，網頁就能立刻具備即時預測的能力，不需要每次重新訓練。

---

## 2. 四大權重檔詳盡解析對照表

以下是 `models/` 下四個檔案的詳細說明：

| 檔案名稱 | 它代表的 AI 角色 | 它是哪一個步驟產生的？ | 它的工作原理與在網頁上的作用 |
| :--- | :--- | :--- | :--- |
| **`preprocessor.pkl`**<br>(資料翻譯器/防護鎖) | **「數據翻譯官與參考書」** | 執行 **`python src/model_trainer.py`** 過程中產生。 | *   **工作原理**：AI 只看得懂標準化後的數值。這個檔案記住了如何把房東輸入的文字「獨熱展開」、把運算評分補中位數、並將數值縮放標準化。<br>*   **網頁作用**：它內含一份歷史房源的「區域價格庫」。當你在沙盒拉動房價時，網頁就是靠它來即時重算價格在當地的相對排名百分位 (`price_pctl_nbhd`)。 |
| **`regressor_model.pkl`**<br>(模型 A：迴歸模型) | **「空屋率估算師」** | 執行 **`python src/model_trainer.py`** 過程中產生。 | *   **工作原理**：這是一個 `HistGradientBoostingRegressor` 迴歸模型，專門用來預測連續變數。<br>*   **網頁作用**：當你點選某間房源或調整沙盒時，網頁就是呼叫它來計算出**具體的預測空屋率數字**（如 58.7%），並將指針畫在半圓形儀表板上。 |
| **`classifier_model.pkl`**<br>(模型 B：分類模型) | **「高風險警報器」** | 執行 **`python src/model_trainer.py`** 過程中產生。 | *   **工作原理**：這是一個二元分類器，包裝了 Isotonic 概率校準器。專門回答是非題：「該房源空屋率是否會突破 70% 的紅線？」<br>*   **網頁作用**：當預估空屋率偏高時，網頁呼叫它來輸出**經物理校準、具真實信賴度的高風險發生概率（%）**，並觸發前端紅色呼吸燈閃爍。 |
| **`shap_explainer.pkl`**<br>(SHAP 解釋器) | **「加扣分診斷專家」** | 執行 **`python src/explainer.py`** 過程中產生。 | *   **工作原理**：這是一個基於模型 A 的 `TreeExplainer`。它能把迴歸大腦的預測結果，拆解為 50 個特徵對預測值的加分與扣分貢獻度。<br>*   **網頁作用**：當你調整參數後，網頁靠它畫出紅（扣分）綠（加分）相間的雙向橫條圖，並抓出扣分最多的 Top 2 項特徵產生白話智能診斷報告。 |

---

## 3. 這些 .pkl 檔案在後續程式碼中是如何被呼叫與使用的？

這四個大腦資產被保存在 `models/` 下之後，主要是在後端 API 模組 **`src/inference_api.py`** 與 **`src/explainer.py`** 中被載入並執行。以下對照實際程式碼，用最直白的方式解釋其運作：

### 3.1 步驟一：從硬碟中「解凍載入」大腦檔案
當網頁或後端 API 啟動時，程式會使用 `joblib.load` 來讀取這四個檔案，將它們載入到記憶體中：

*   **對應程式碼**（[inference_api.py L27-L29](file:///d:/files/smartaccommodation_imp_new/src/inference_api.py#L27-L29)）：
```python
# 從 models 資料夾中，載入翻譯器與兩個 AI 模型
preprocessor_pack = joblib.load('models/preprocessor.pkl')
reg_model = joblib.load('models/regressor_model.pkl')
clf_model = joblib.load('models/classifier_model.pkl')
```
*   **白話翻譯**：把我們存好的「翻譯官」、「空屋估算師」、「高風險警報器」三個大腦讀取出來，裝進程式變數中，準備隨時應付網頁端的預測請求。

---

### 3.2 步驟二：呼叫 preprocessor.pkl（將網頁參數翻譯給 AI）
當房東在網頁沙盒拉動房價或入住天數時，新輸入的參數還是「原始人類語言」。我們必須先用翻譯器將其標準化：

*   **對應程式碼**（[inference_api.py L112](file:///d:/files/smartaccommodation_imp_new/src/inference_api.py#L112)）：
```python
# 使用已訓練好的預處理管線，將新參數轉換為標準數值矩陣
X_prep = preprocessor.transform(X_sim_features)
```
*   **白話翻譯**：AI 不懂得什麼是「1晚、2晚或 NTD 1500元」。這行程式碼把這四個沙盒參數投進 `preprocessor` 翻譯機，轉換為小數點與標準化矩陣 `X_prep`，AI 才能讀懂。

---

### 3.3 步驟三：呼叫 regressor_model.pkl（計算空屋率）
將翻譯好的數據餵給模型 A（Regressor），預估具體的空屋率值：

*   **對應程式碼**（[inference_api.py L115-L116](file:///d:/files/smartaccommodation_imp_new/src/inference_api.py#L115-L116)）：
```python
# 使用迴歸模型，進行空屋率的實時預測
pred_vacancy = reg_model.predict(X_prep)[0]
pred_vacancy = np.clip(pred_vacancy, 0.0, 1.0)
```
*   **白話翻譯**：讓「空屋估算師」根據剛剛翻譯好的數據進行預測，得出一個空屋率小數（例如 0.587），並防呆限制在 0.0 ~ 1.0 之間。網頁會將其轉為 58.7% 顯示在 SVG 半圓形儀表板上。

---

### 3.4 步驟四：呼叫 classifier_model.pkl（計算高風險機率）
若空屋率偏高，則需要透過模型 B（Classifier）計算機率，以觸發警報：

*   **對應程式碼**（[inference_api.py L119](file:///d:/files/smartaccommodation_imp_new/src/inference_api.py#L119)）：
```python
# 使用等張校準後的分類模型，預測屬於「高空屋風險」的真實機率
pred_risk_prob = clf_model.predict_proba(X_prep)[0][1]
```
*   **白話翻譯**：讓「高風險警報器」預估該房源空屋率破 70% 的校準後實質概率（例如 82.5%）。網頁會利用這個百分比來聯動紅色呼吸燈閃爍，提醒房東。

---

### 3.5 步驟五：呼叫 shap_explainer.pkl（產出紅綠橫條圖）
呼叫 SHAP 解釋器，計算出這 50 個特徵在這次預測中各自貢獻了多少加分或扣分：

*   **對應程式碼**（[explainer.py L32 與 L43](file:///d:/files/smartaccommodation_imp_new/src/explainer.py#L32)）：
```python
# 1. 載入 SHAP 解釋器
_explainer = joblib.load('models/shap_explainer.pkl')

# 2. 計算特徵貢獻值 (SHAP values)
shap_values = _explainer.shap_values(X_prep)
```
*   **白話翻譯**：將翻譯好的數據 `X_prep` 餵給「加扣分診斷專家」。它會精確計算出例如：「*因為價格百分位偏高，所以推高了 12.3% 的空屋風險；因為回覆速度極快，所以降低了 5.2% 的空屋風險*」。網頁會接收這些數值，動態繪製成紅綠相間的條形圖，並在右側欄輸出 Top 2 行動改善指引。
