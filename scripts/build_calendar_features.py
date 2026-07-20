# -*- coding: utf-8 -*-
"""build_calendar_features.py — 未來檔期特徵預計算(calendar.csv.gz → 輕量產物)

設計考量:calendar.csv.gz 有 234 萬列,直接在 Streamlit Cloud(約 1GB RAM)載入會爆記憶體。
本腳本將其壓縮為兩份輕量產物,線上只讀取結果:

產出
----
data/_calendar_metrics.csv   每房源一列;含 365 天訂房遮罩(0/1 字串,僅 ~2MB)與彙總指標
data/_calendar_market.csv    市場層級:每日、每月 × 行政區/房型 的已訂率基準
models/forward_validation.json  前瞻驗證:現有模型 OOF 預測 vs calendar 真實未來結果

執行
----
    python -X utf8 scripts/build_calendar_features.py

資料陷阱(見 doc/04):available='f' 同時包含「已被預訂」與「房東主動封鎖」,
故本腳本一併輸出 is_all_blocked / is_all_open 旗標供前端排除異常房源。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data"
MODELS = ROOT / "models"
HORIZON_WINDOWS = [(0, 30, "d30"), (0, 60, "d60"), (0, 90, "d90")]
GAP_MIN_LEN = 5          # 連續空檔達幾天才列為警示
ANALYSIS_WINDOW = 90     # 空檔警示掃描範圍(天)


def log(m):
    print(f"[calendar] {m}", flush=True)


def load_calendar() -> pd.DataFrame:
    c = pd.read_csv(DATA / "calendar.csv.gz")
    c["date"] = pd.to_datetime(c["date"], errors="coerce")
    c = c.dropna(subset=["date"])
    c["booked"] = (c["available"].astype(str).str.lower() == "f").astype(np.int8)
    return c.sort_values(["listing_id", "date"])


def find_gaps(mask: str, dates: pd.DatetimeIndex, min_len: int = GAP_MIN_LEN):
    """從訂房遮罩找出連續空檔區段(可訂 = '0')。回傳 [(起日, 迄日, 天數)]。"""
    gaps, run = [], 0
    for i, ch in enumerate(mask):
        if ch == "0":
            run += 1
        else:
            if run >= min_len:
                gaps.append((dates[i - run], dates[i - 1], run))
            run = 0
    if run >= min_len:
        gaps.append((dates[len(mask) - run], dates[len(mask) - 1], run))
    return gaps


def build_listing_metrics(c: pd.DataFrame) -> pd.DataFrame:
    """每房源:365 天遮罩 + 各時窗已訂率 + 空檔統計 + 最低入住策略。"""
    start = c["date"].min()
    rows = []
    for lid, g in c.groupby("listing_id", sort=False):
        g = g.sort_values("date")
        dates = pd.DatetimeIndex(g["date"].values)
        mask = "".join(g["booked"].astype(str).tolist())
        booked = g["booked"].to_numpy()
        h = (dates - start).days.to_numpy()

        rec = {"listing_id": int(lid),
               "cal_start": dates[0].date().isoformat(),
               "n_days": len(g),
               "booked_mask": mask,
               "booked_rate": float(booked.mean()),
               "booked_days": int(booked.sum())}
        for lo, hi, tag in HORIZON_WINDOWS:
            sel = (h >= lo) & (h <= hi)
            rec[f"booked_rate_{tag}"] = (float(booked[sel].mean())
                                         if sel.sum() else np.nan)
        # 空檔警示:同時計算 90 天(檢視用)與 30 天(通知用,近期日曆已開放最可信)
        for win, tag in [(ANALYSIS_WINDOW, "90d"), (30, "30d")]:
            selw = h <= win
            gw = find_gaps("".join(booked[selw].astype(str)), dates[selw])
            rec[f"gap_count_{tag}"] = len(gw)
            rec[f"gap_days_{tag}"] = int(sum(x[2] for x in gw))
            rec[f"gap_longest_{tag}"] = int(max([x[2] for x in gw], default=0))
            if tag == "90d":
                rec["gap_first_start"] = (gw[0][0].date().isoformat()
                                          if gw else "")
            else:
                rec["gap_first_start_30d"] = (gw[0][0].date().isoformat()
                                              if gw else "")
        # 最低入住天數策略
        mn = g["minimum_nights"]
        rec["min_nights_median"] = float(mn.median())
        rec["min_nights_varies"] = int(mn.nunique() > 1)
        rec["is_longterm_only"] = int(mn.median() >= 28)
        # 異常旗標(見 doc/04 §2.2)
        rec["is_all_blocked"] = int(booked.mean() == 1.0)
        rec["is_all_open"] = int(booked.mean() == 0.0)
        rows.append(rec)
    df = pd.DataFrame(rows)
    # 逐月已訂率(寬表:m1~m12)
    c2 = c.copy()
    c2["mi"] = ((c2["date"].dt.year - start.year) * 12
                + c2["date"].dt.month - start.month) + 1
    piv = (c2[c2["mi"].between(1, 12)]
           .pivot_table(index="listing_id", columns="mi", values="booked",
                        aggfunc="mean"))
    piv.columns = [f"m{int(x)}_rate" for x in piv.columns]
    return df.merge(piv.round(4), left_on="listing_id", right_index=True,
                    how="left")


def build_market(c: pd.DataFrame, listings: pd.DataFrame) -> pd.DataFrame:
    """市場基準:行政區 × 房型 × 月份 的平均已訂率(同商圈比較用)。"""
    start = c["date"].min()
    m = c.merge(listings[["id", "neighbourhood_cleansed", "room_type"]],
                left_on="listing_id", right_on="id", how="inner")
    m["mi"] = ((m["date"].dt.year - start.year) * 12
               + m["date"].dt.month - start.month) + 1
    out = (m[m["mi"].between(1, 12)]
           .groupby(["neighbourhood_cleansed", "room_type", "mi"])["booked"]
           .agg(["mean", "size"]).reset_index()
           .rename(columns={"mean": "mkt_rate", "size": "n_days"}))
    out["mkt_rate"] = out["mkt_rate"].round(4)
    return out


def forward_validation(metrics: pd.DataFrame) -> dict:
    """以 calendar(2026-06 爬取)驗證模型(2025-09 特徵)的真實前瞻表現。"""
    pred_path = DATA / "_predictions.csv"
    if not pred_path.exists():
        return {"error": "缺 data/_predictions.csv,請先執行 train_backend_models.py"}
    from sklearn.metrics import (precision_score, r2_score, recall_score,
                                 roc_auc_score)
    p = pd.read_csv(pred_path)
    ok = metrics[(metrics["is_all_blocked"] == 0) & (metrics["is_all_open"] == 0)]
    j = p.merge(ok[["listing_id", "booked_rate"]], left_on="id",
                right_on="listing_id", how="inner")
    j["real_vacancy"] = 1 - j["booked_rate"]
    y = (j["real_vacancy"] >= 0.6).astype(int)
    out = {
        "n_matched": int(len(j)),
        "listings_scraped": "2025-09-30",
        "calendar_scraped": "2026-06-30",
        "gap_months": 9,
        "real_vacancy_mean": float(j["real_vacancy"].mean()),
        "pred_vacancy_mean": float(j["vac_pred"].mean()),
        "real_high_risk_rate": float(y.mean()),
        "reg_corr": float(np.corrcoef(j["vac_pred"], j["real_vacancy"])[0, 1]),
        "reg_r2": float(r2_score(j["real_vacancy"], j["vac_pred"])),
        "clf_auc": float(roc_auc_score(y, j["prob"])),
    }
    for th, tag in [(0.6, "red"), (0.35, "yellow")]:
        pred = (j["prob"] >= th).astype(int)
        out[tag] = {
            "threshold": th, "n_flag": int(pred.sum()),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
        }
    return out


def main():
    log("讀取 calendar.csv.gz …")
    c = load_calendar()
    log(f"{len(c):,} 列 · {c['listing_id'].nunique():,} 房源 · "
        f"{c['date'].min().date()} → {c['date'].max().date()}")

    metrics = build_listing_metrics(c)
    metrics.to_csv(DATA / "_calendar_metrics.csv", index=False, encoding="utf-8")
    log(f"_calendar_metrics.csv: {len(metrics):,} 房源 "
        f"({(DATA / '_calendar_metrics.csv').stat().st_size / 1e6:.1f} MB)")

    listings = pd.read_csv(DATA / "listings_cleaned.csv.gz",
                           usecols=["id", "neighbourhood_cleansed", "room_type"],
                           low_memory=False)
    market = build_market(c, listings)
    market.to_csv(DATA / "_calendar_market.csv", index=False, encoding="utf-8")
    log(f"_calendar_market.csv: {len(market):,} 組合")

    fv = forward_validation(metrics)
    MODELS.mkdir(exist_ok=True)
    (MODELS / "forward_validation.json").write_text(
        json.dumps(fv, ensure_ascii=False, indent=1), encoding="utf-8")
    if "error" not in fv:
        log(f"前瞻驗證: n={fv['n_matched']:,} AUC={fv['clf_auc']:.3f} "
            f"R2={fv['reg_r2']:.3f} 紅層P={fv['red']['precision']:.2f}")
    log("完成")


if __name__ == "__main__":
    main()
