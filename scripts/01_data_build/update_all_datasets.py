# -*- coding: utf-8 -*-
"""
多模態數據集更新腳本
1. 載入全量提取的影像特徵 multimodal_features_all.csv (5849 筆)。
2. 自動備份原始 dataset_vacancy.csv 與 dataset_final.csv。
3. 將 7 個影像特徵合併入這兩個數據集中，覆寫存檔，完成模型特徵庫的實體更新。
"""
import shutil
import os
import pandas as pd

FEAT_FILE = "../../multimodal_features_all.csv"
VAC_FILE = "../../dataset_vacancy.csv"
FINAL_FILE = "../../dataset_final.csv"

# 1. 檢查影像特徵檔是否存在
if not os.path.exists(FEAT_FILE):
    print("錯誤：找不到 {}！請等特徵提取任務完成。".format(FEAT_FILE))
    exit(1)

print("加載影像多模態特徵...")
df_feats = pd.read_csv(FEAT_FILE)
print("載入成功，共計 {} 筆影像特徵。".format(len(df_feats)))

# 2. 備份與更新 dataset_vacancy.csv
if os.path.exists(VAC_FILE):
    shutil.copy(VAC_FILE, "../../dataset_vacancy_backup.csv")
    print("已備份 {} 至 dataset_vacancy_backup.csv。".format(VAC_FILE))
    
    df_vac = pd.read_csv(VAC_FILE)
    # 若原本已有影像欄位則刪除，重新合併以防重複
    cols_to_drop = [c for c in df_feats.columns if c in df_vac.columns and c != "listing_id"]
    if cols_to_drop:
        df_vac = df_vac.drop(columns=cols_to_drop)
        
    df_vac_new = df_vac.merge(df_feats, on="listing_id", how="inner")
    df_vac_new.to_csv(VAC_FILE, index=False, encoding="utf-8-sig")
    print("成功更新 {}，新維度: {}。".format(VAC_FILE, df_vac_new.shape))

# 3. 備份與更新 dataset_final.csv
if os.path.exists(FINAL_FILE):
    shutil.copy(FINAL_FILE, "../../dataset_final_backup.csv")
    print("已備份 {} 至 dataset_final_backup.csv。".format(FINAL_FILE))
    
    df_final = pd.read_csv(FINAL_FILE)
    cols_to_drop = [c for c in df_feats.columns if c in df_final.columns and c != "listing_id"]
    if cols_to_drop:
        df_final = df_final.drop(columns=cols_to_drop)
        
    df_final_new = df_final.merge(df_feats, on="listing_id", how="inner")
    df_final_new.to_csv(FINAL_FILE, index=False, encoding="utf-8-sig")
    print("成功更新 {}，新維度: {}。".format(FINAL_FILE, df_final_new.shape))

print("\n全量多模態數據集更新完畢！特徵欄位已成功對齊。")
