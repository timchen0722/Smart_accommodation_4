# -*- coding: utf-8 -*-
"""
多模態特徵提取：下載代表性樣本封面照片，計算客觀畫質特徵，
並使用 CLIP 模型提取主觀的「溫馨感」、「設計感」、「整潔感」特徵，
最後將其與結構化特徵融合，生成用於模型驗證的 multimodal_features_sample.csv。
"""
import time
import io
import urllib.request
import os
import subprocess
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------- 0. 自動安裝與導入依賴 ----------
print("檢查並導入機器學習依賴...")
try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
    print("PyTorch 與 Transformers 已安裝成功。")
except ImportError:
    print("未檢測到 torch 或 transformers，正在自動安裝輕量 CPU 版...")
    # 自動安裝 CPU 版本的 PyTorch 與 Transformers
    subprocess.run(["pip", "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cpu"], check=True)
    subprocess.run(["pip", "install", "transformers"], check=True)
    import torch
    from transformers import CLIPProcessor, CLIPModel
    print("自動安裝完成並成功導入！")

# ---------- 1. 加載原始數據與樣本篩選 ----------
print("\n[Step 1] 加載數據集...")
df_final = pd.read_csv("../../dataset_final.csv")
df_cleaned = pd.read_csv("../../listings_cleaned.csv.gz", compression="gzip", low_memory=False)

# 取出 final 中的 listing_id，並隨機抽取 500 個樣本做多模態 PoC 驗證
sample_ids = df_final["listing_id"].sample(500, random_state=42).tolist()
df_sample = df_final[df_final["listing_id"].isin(sample_ids)].copy()
print("已隨機篩選出 500 個代表性樣本房源進行驗證。")

# ---------- 2. 多線程下載封面照片 ----------
print("\n[Step 2] 開始多線程下載房源照片...")
df_urls = df_cleaned[df_cleaned["id"].isin(sample_ids)][["id", "picture_url"]]

def download_image(row):
    try:
        req = urllib.request.Request(row.picture_url, headers={"User-Agent": "Mozilla/5.0"})
        data = urllib.request.urlopen(req, timeout=10).read()
        pil_img = Image.open(io.BytesIO(data)).convert("RGB")
        cv_img = np.array(pil_img)
        return row.id, pil_img, cv_img
    except Exception:
        return row.id, None, None

downloaded_images = {}
t0 = time.time()
with ThreadPoolExecutor(max_workers=20) as executor:
    futures = [executor.submit(download_image, row) for row in df_urls.itertuples()]
    for i, f in enumerate(as_completed(futures), 1):
        lid, pil_img, cv_img = f.result()
        if pil_img is not None:
            downloaded_images[lid] = (pil_img, cv_img)
        if i % 100 == 0:
            print("  已處理下載進度: {}/500".format(i))

print("下載完成！成功取得 {}/500 張照片，耗時 {:.1f} 秒。".format(len(downloaded_images), time.time() - t0))

# ---------- 3. 客觀畫質特徵與 CLIP 主觀美學特徵提取 ----------
print("\n[Step 3] 初始化 CLIP 模型，開始提取影像特徵...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 定義主觀美學特徵的 CLIP Prompt 對 (正向對立於負向)
prompts = {
    "coziness": (
        "a cozy, warm, and inviting room with soft lighting", 
        "a cold, sterile, and uninviting room with harsh lighting"
    ),
    "design_sense": (
        "a beautifully designed room with stylish and modern interior decor", 
        "a poorly designed room with cheap, outdated, and ugly decor"
    ),
    "cleanliness": (
        "a clean, neat, and tidy room", 
        "a messy, dirty, and cluttered room with trash"
    )
}

def colorfulness(img):
    (B, G, R) = cv2.split(img.astype("float"))
    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)
    return np.sqrt(rg.std()**2 + yb.std()**2) + 0.3 * np.sqrt(rg.mean()**2 + yb.mean()**2)

rows = []
t1 = time.time()
for idx, (lid, (pil_img, cv_img)) in enumerate(downloaded_images.items(), 1):
    try:
        # A. 計算客觀特徵 (OpenCV)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
        brightness = gray.mean()
        contrast = gray.std()
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        color_val = colorfulness(cv_img)
        
        # B. 計算主觀特徵 (CLIP Softmax 相似度投影)
        feature_scores = {}
        with torch.no_grad():
            for feat_name, (pos_prompt, neg_prompt) in prompts.items():
                inputs = processor(text=[pos_prompt, neg_prompt], images=pil_img, return_tensors="pt", padding=True)
                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
                # 正向機率 (index 0) 作為此維度的主觀評分
                feature_scores[feat_name] = probs[0, 0].item()
        
        rows.append({
            "listing_id": lid,
            "photo_width": cv_img.shape[1],
            "photo_height": cv_img.shape[0],
            "photo_brightness": brightness,
            "photo_contrast": contrast,
            "photo_sharpness": sharpness,
            "photo_colorfulness": color_val,
            "photo_coziness": feature_scores["coziness"],
            "photo_design_sense": feature_scores["design_sense"],
            "photo_cleanliness": feature_scores["cleanliness"]
        })
        
        if idx % 50 == 0 or idx == len(downloaded_images):
            print("  特徵提取進度: {}/{}".format(idx, len(downloaded_images)))
            
    except Exception as e:
        print("  處理 listing_id {} 特徵時發生錯誤: {}".format(lid, str(e)))

df_img_features = pd.DataFrame(rows)
print("影像特徵提取完成，耗時 {:.1f} 秒。".format(time.time() - t1))

# ---------- 4. 特徵融合與存檔 ----------
print("\n[Step 4] 與原 36 維特徵進行多模態融合...")
df_multimodal = df_sample.merge(df_img_features, on="listing_id", how="inner")
df_multimodal.to_csv("../../multimodal_features_sample.csv", index=False, encoding="utf-8-sig")
print("成功生成多模態融合樣本集: multimodal_features_sample.csv，共計 {} 筆樣本。".format(len(df_multimodal)))

# ---------- 5. 輸出相關性分析 ----------
print("\n=== 新增影像特徵與空屋率 (Y_vacancy) 的相關係數 ===")
new_cols = [
    "photo_brightness", "photo_contrast", "photo_sharpness", "photo_colorfulness",
    "photo_coziness", "photo_design_sense", "photo_cleanliness"
]
for col in new_cols:
    if col in df_multimodal.columns:
        corr_val = df_multimodal[col].corr(df_multimodal["Y_vacancy"])
        print("  {:25s} : {:+.4f}".format(col, corr_val))
