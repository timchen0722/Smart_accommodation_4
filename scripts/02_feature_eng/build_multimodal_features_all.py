# -*- coding: utf-8 -*-
"""
全量多模態特徵提取腳本 (支援 Batching ＋ Checkpointing)
1. 自動讀取 dataset_final.csv 的 5849 筆房源。
2. 支援斷點續傳：檢查已計算的檔案，重啟時自動跳過已完成的 id。
3. 採用分批下載與 CLIP 批次推論（Batch Inference），大幅降低 CPU 推理時間與記憶體開銷。
4. 下載失敗的照片以中位數默認值填充，保證全量數據特徵矩陣之完整性。
5. 每批處理完畢即時 Append 存檔，防止中途意外中斷。
"""
import time
import io
import urllib.request
import os
import numpy as np
import pandas as pd
import cv2
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------- 參數配置 ----------
BATCH_SIZE = 64          # 每批下載與處理的房源數
CLIP_BATCH_SIZE = 32     # 送入 CLIP 推論的最大批次大小 (避免記憶體過載)
TIMEOUT = 5              # 網路下載逾時限制 (秒)
OUTPUT_FILE = "../../multimodal_features_all.csv"

# ---------- 0. 初始化資料庫與 Checkpoint 檢測 ----------
print("檢查並導入機器學習依賴...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("CLIP-ViT-B/32 模型加載成功。")

# 讀取全量 final 數據以獲取 5849 筆的 listing_id
df_final = pd.read_csv("../../dataset_final.csv")
all_lids = df_final["listing_id"].tolist()
print("全量數據集讀取完畢，總房源數: {}".format(len(all_lids)))

# 斷點檢測
processed_lids = set()
if os.path.exists(OUTPUT_FILE):
    try:
        df_existing = pd.read_csv(OUTPUT_FILE, usecols=["listing_id"])
        processed_lids = set(df_existing["listing_id"].tolist())
        print("檢測到歷史存檔！已完成 {}/{} 筆，將從斷點繼續。".format(len(processed_lids), len(all_lids)))
    except Exception:
        print("檢測到歷史存檔，但讀取失敗。將重新開始。")

todo_lids = [lid for lid in all_lids if lid not in processed_lids]
print("待處理房源數量: {}".format(len(todo_lids)))

if len(todo_lids) == 0:
    print("所有房源已提取完畢，無需執行。")
    exit(0)

# 讀取 url 映射表
df_cleaned = pd.read_csv("../../listings_cleaned.csv.gz", compression="gzip", low_memory=False)
id_to_url = dict(zip(df_cleaned["id"], df_cleaned["picture_url"]))

# ---------- 1. 定義影像計算函數 ----------
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

def download_single_image(lid):
    url = id_to_url.get(lid, None)
    if not url or not isinstance(url, str):
        return lid, None, None
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        data = urllib.request.urlopen(req, timeout=TIMEOUT).read()
        pil_img = Image.open(io.BytesIO(data)).convert("RGB")
        cv_img = np.array(pil_img)
        return lid, pil_img, cv_img
    except Exception:
        return lid, None, None

# 默認填充值 (當下載失敗或讀取錯誤時使用，保持特徵數據完整)
DEFAULT_FEATURES = {
    "photo_width": 1024,
    "photo_height": 683,
    "photo_brightness": 135.0,  # 均值亮度
    "photo_contrast": 60.0,      # 均值對比度
    "photo_sharpness": 400.0,    # 均值清晰度
    "photo_colorfulness": 25.0,  # 均值色彩豐富度
    "photo_coziness": 0.5,       # 溫馨感中位數
    "photo_design_sense": 0.5,   # 設計感中位數
    "photo_cleanliness": 0.95    # 乾淨度（通常偏高）
}

# ---------- 2. 分批循環提取 (Chunk Execution) ----------
t_start = time.time()
num_processed = len(processed_lids)
total_todo = len(todo_lids)

for i in range(0, total_todo, BATCH_SIZE):
    chunk_lids = todo_lids[i : i + BATCH_SIZE]
    chunk_results = {lid: {} for lid in chunk_lids}
    
    # A. 多線程下載圖片
    downloaded = {}
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(download_single_image, lid) for lid in chunk_lids]
        for f in as_completed(futures):
            lid, pil_img, cv_img = f.result()
            if pil_img is not None:
                downloaded[lid] = (pil_img, cv_img)
                
    # B. 計算 OpenCV 客觀畫質特徵
    success_lids = []
    success_pil_images = []
    
    for lid in chunk_lids:
        if lid in downloaded:
            pil_img, cv_img = downloaded[lid]
            success_lids.append(lid)
            success_pil_images.append(pil_img)
            
            # OpenCV 特徵
            gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
            chunk_results[lid]["photo_width"] = cv_img.shape[1]
            chunk_results[lid]["photo_height"] = cv_img.shape[0]
            chunk_results[lid]["photo_brightness"] = gray.mean()
            chunk_results[lid]["photo_contrast"] = gray.std()
            chunk_results[lid]["photo_sharpness"] = cv2.Laplacian(gray, cv2.CV_64F).var()
            chunk_results[lid]["photo_colorfulness"] = colorfulness(cv_img)
        else:
            # 下載失敗者填入預設值
            for k, v in DEFAULT_FEATURES.items():
                chunk_results[lid][k] = v
                
    # C. 使用 CLIP 進行批次推論 (Batch Inference) 提取主觀特徵
    if len(success_pil_images) > 0:
        # 將下載成功的圖片切分成 CLIP_BATCH_SIZE 大小進行批次推論
        for b_start in range(0, len(success_pil_images), CLIP_BATCH_SIZE):
            b_lids = success_lids[b_start : b_start + CLIP_BATCH_SIZE]
            b_imgs = success_pil_images[b_start : b_start + CLIP_BATCH_SIZE]
            
            with torch.no_grad():
                for feat_name, (pos_prompt, neg_prompt) in prompts.items():
                    inputs = processor(text=[pos_prompt, neg_prompt], images=b_imgs, return_tensors="pt", padding=True)
                    outputs = model(**inputs)
                    logits_per_image = outputs.logits_per_image  # Shape: (num_images, 2)
                    probs = logits_per_image.softmax(dim=1)
                    
                    # 寫入批次中每張圖片的該維度得分
                    for idx, lid in enumerate(b_lids):
                        chunk_results[lid][f"photo_{feat_name}"] = probs[idx, 0].item()

    # D. 轉換為 DataFrame 並寫入 CSV (Append 模式)
    rows_to_save = []
    for lid in chunk_lids:
        row = {"listing_id": lid}
        row.update(chunk_results[lid])
        rows_to_save.append(row)
        
    df_chunk = pd.DataFrame(rows_to_save)
    # 若檔案不存在則寫入標頭，否則追加
    header_needed = not os.path.exists(OUTPUT_FILE)
    df_chunk.to_csv(OUTPUT_FILE, mode="a", index=False, header=header_needed, encoding="utf-8-sig")
    
    num_processed += len(chunk_lids)
    elapsed_total = time.time() - t_start
    avg_speed = elapsed_total / num_processed if num_processed > 0 else 0
    est_remaining = avg_speed * (len(all_lids) - num_processed) / 60.0
    
    print("  進度: {}/{} ({:.1%}) | 下載成功: {}/{} | 均速: {:.2f}s/筆 | 預計剩餘: {:.1f}分鐘".format(
        num_processed, len(all_lids), num_processed / len(all_lids),
        len(success_lids), len(chunk_lids), avg_speed, est_remaining
    ))

print("\n全量影像多模態特徵提取完成！耗時 {:.1f} 分鐘，數據已安全寫入 {}。".format(
    (time.time() - t_start) / 60.0, OUTPUT_FILE
))
