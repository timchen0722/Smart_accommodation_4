# -*- coding: utf-8 -*-
"""
小樣本測試：下載30張房源封面照，計算「客觀畫質特徵」
(解析度、亮度、對比度、色彩豐富度、清晰度/模糊偵測)
不需GPU/深度學習，驗證可行性與速度後再決定是否擴大到全部6241筆。
"""
import time, io, urllib.request
import numpy as np, pandas as pd, cv2
from PIL import Image

L = pd.read_csv("../../listings_cleaned.csv.gz", compression="gzip", low_memory=False)
sample = L[["id","picture_url"]].dropna().sample(30, random_state=42)

def colorfulness(img):
    (B,G,R) = cv2.split(img.astype("float"))
    rg = np.abs(R-G); yb = np.abs(0.5*(R+G)-B)
    return np.sqrt(rg.std()**2+yb.std()**2) + 0.3*np.sqrt(rg.mean()**2+yb.mean()**2)

rows, t0, fail = [], time.time(), 0
for _, r in sample.iterrows():
    try:
        req = urllib.request.Request(r["picture_url"], headers={"User-Agent":"Mozilla/5.0"})
        data = urllib.request.urlopen(req, timeout=10).read()
        img = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        rows.append({
            "listing_id": r["id"],
            "width": img.shape[1], "height": img.shape[0],
            "brightness": gray.mean(),
            "contrast": gray.std(),
            "sharpness": cv2.Laplacian(gray, cv2.CV_64F).var(),
            "colorfulness": colorfulness(img),
        })
    except Exception as e:
        fail += 1
elapsed = time.time()-t0
df = pd.DataFrame(rows)
print("成功 {} / 失敗 {} | 耗時 {:.1f}s | 平均每張 {:.2f}s".format(len(df), fail, elapsed, elapsed/30))
print()
print(df.describe().round(1).to_string())
print()
print("推估：全部6241筆預估耗時 {:.0f} 分鐘".format(elapsed/30*6241/60))
