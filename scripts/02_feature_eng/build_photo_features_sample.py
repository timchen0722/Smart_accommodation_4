# -*- coding: utf-8 -*-
"""
500張代表性樣本：計算客觀畫質特徵，檢驗與空屋率的相關性
(先驗證有沒有訊號，值得才投資全量下載+CLIP美感模型)
"""
import time, io, urllib.request
import numpy as np, pandas as pd, cv2
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

L = pd.read_csv("../../listings_cleaned.csv.gz", compression="gzip", low_memory=False)
vac = pd.read_csv("../../dataset_vacancy.csv")[["listing_id","Y_vacancy"]]
sample = L[["id","picture_url"]].dropna().merge(vac, left_on="id", right_on="listing_id").sample(500, random_state=7)

def colorfulness(img):
    (B,G,R) = cv2.split(img.astype("float"))
    rg = np.abs(R-G); yb = np.abs(0.5*(R+G)-B)
    return np.sqrt(rg.std()**2+yb.std()**2) + 0.3*np.sqrt(rg.mean()**2+yb.mean()**2)

def fetch(row):
    try:
        req = urllib.request.Request(row.picture_url, headers={"User-Agent":"Mozilla/5.0"})
        data = urllib.request.urlopen(req, timeout=8).read()
        img = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return {"listing_id": row.id, "Y_vacancy": row.Y_vacancy,
                "photo_width": img.shape[1], "photo_height": img.shape[0],
                "photo_brightness": gray.mean(), "photo_contrast": gray.std(),
                "photo_sharpness": cv2.Laplacian(gray, cv2.CV_64F).var(),
                "photo_colorfulness": colorfulness(img)}
    except Exception:
        return None

t0 = time.time()
rows = []
with ThreadPoolExecutor(max_workers=30) as ex:
    futs = [ex.submit(fetch, row) for row in sample.itertuples()]
    for f in as_completed(futs):
        r = f.result()
        if r: rows.append(r)
print("成功 {} / {} | 耗時 {:.1f}分鐘".format(len(rows), len(sample), (time.time()-t0)/60))

df = pd.DataFrame(rows)
df.to_csv("../../photo_features_sample.csv", index=False, encoding="utf-8-sig")
print("\n畫質特徵敘述統計:")
print(df.describe().round(1).to_string())
print("\n與空屋率(Y_vacancy)的相關係數:")
for c in ["photo_width","photo_height","photo_brightness","photo_contrast","photo_sharpness","photo_colorfulness"]:
    print("  {:20s} {:+.3f}".format(c, df[c].corr(df["Y_vacancy"])))
