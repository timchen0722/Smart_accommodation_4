# -*- coding: utf-8 -*-
"""測試平行下載速度：50張樣本，30個執行緒"""
import time, io, urllib.request
import numpy as np, pandas as pd, cv2
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

L = pd.read_csv("../../listings_cleaned.csv.gz", compression="gzip", low_memory=False)
sample = L[["id","picture_url"]].dropna().sample(50, random_state=1)

def fetch(row):
    try:
        req = urllib.request.Request(row.picture_url, headers={"User-Agent":"Mozilla/5.0"})
        data = urllib.request.urlopen(req, timeout=8).read()
        img = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return {"id": row.id, "w": img.shape[1], "h": img.shape[0],
                "brightness": gray.mean(), "sharpness": cv2.Laplacian(gray, cv2.CV_64F).var()}
    except Exception:
        return None

t0 = time.time()
results = []
with ThreadPoolExecutor(max_workers=30) as ex:
    futs = [ex.submit(fetch, row) for row in sample.itertuples()]
    for f in as_completed(futs):
        r = f.result()
        if r: results.append(r)
elapsed = time.time()-t0
print("成功 {} / {} | 耗時 {:.1f}s | 平均每張 {:.2f}s".format(len(results), len(sample), elapsed, elapsed/len(sample)))
print("推估全部6241筆(30 threads)預估耗時: {:.1f} 分鐘".format(elapsed/len(sample)*6241/60))
