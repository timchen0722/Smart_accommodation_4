# -*- coding: utf-8 -*-
"""build_data_analysis_json.py — 產出「數據分析」分頁所需的評估 JSON

單一可信來源：models/eval_vacancy_90.json
  · 模型：HistGradientBoosting（回歸＋分類）· Isotonic 校準警報 · 37 核心特徵
  · 主目標 Y_vacancy_90 = availability_90/90；高風險 = vacancy_90 > 0.70
  · 365 天同特徵對照（供營收雙輸出敘事）
  · 誠實 GroupKFold(host_id,5折) + 單次切分(樂觀) 雙軌
  · 校準機率警報 紅0.60/黃0.35 的 Precision/Coverage
  · Permutation Importance Top12（90 與 365）

參數與 scripts/04_eval/eval_vacancy_90_models.py 對齊，數字可互相佐證。
App 端不重算，只讀本 JSON。重跑：python -X utf8 scripts/04_eval/build_data_analysis_json.py
"""
import os
import sys
import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.ensemble import (HistGradientBoostingRegressor,
                              HistGradientBoostingClassifier)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             roc_auc_score, average_precision_score,
                             precision_score, recall_score, f1_score)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, BASE)
from modules.feature_engineering import FEAT_ZH_V2  # noqa: E402

# 少數缺漏的繁中標籤補丁
ZH_PATCH = {"bathrooms": "浴室數", "property_type_code": "房產類型",
            "photo_design_sense": "封面照設計感", "latitude": "緯度",
            "longitude": "經度"}
SEED = 42
RED_TH, YELLOW_TH = 0.60, 0.35
REG_PARAMS = dict(max_iter=500, learning_rate=0.05, l2_regularization=1.0,
                  random_state=SEED)
CLF_PARAMS = dict(max_iter=500, learning_rate=0.05, l2_regularization=1.0,
                  random_state=SEED)


def zh(key):
    return FEAT_ZH_V2.get(key) or ZH_PATCH.get(key, key)


def main():
    # v90：讀 App 上線的同一份資料（train=serve 一致），與部署 bundle 數字對齊
    mm = pd.read_csv(os.path.join(BASE, "data", "dataset_multimodal.csv"),
                     encoding="utf-8-sig")
    listings = pd.read_csv(os.path.join(BASE, "listings_cleaned.csv.gz"),
                           compression="gzip", low_memory=False)[["id", "availability_90"]]
    df = mm.merge(listings, on="id", how="left")
    extra = pd.read_csv(os.path.join(BASE, "data", "_core_extra.csv"))
    df = df.merge(extra, on="id", how="left")

    root_cols = pd.read_csv(os.path.join(BASE, "dataset_final.csv"), nrows=1).columns
    feats_root = [c for c in root_cols
                  if c not in ["listing_id", "Y_vacancy"]
                  and (not c.startswith("photo_") or c == "photo_design_sense")]
    FEATURES = ["bathrooms_count" if c == "bathrooms" else c for c in feats_root]

    df["Y_vacancy_90"] = (pd.to_numeric(df["availability_90"], errors="coerce")
                          / 90.0).clip(0, 1)
    df["Y_vacancy_365"] = pd.to_numeric(df["Y_vacancy"], errors="coerce").fillna(0)
    df = df.dropna(subset=["Y_vacancy_90"]).reset_index(drop=True)

    X = df[FEATURES]
    y90 = df["Y_vacancy_90"].values
    y365 = df["Y_vacancy_365"].values
    groups = df["host_id"].values
    h90 = (y90 > 0.70).astype(int)
    h365 = (y365 > 0.70).astype(int)

    # ── 1) 單次切分 80/20（樂觀，含房東洩漏）──
    (Xtr, Xte, y90tr, y90te, y365tr, y365te,
     h90tr, h90te, h365tr, h365te) = train_test_split(
        X, y90, y365, h90, h365, test_size=0.2, random_state=SEED)

    def _reg_single(ytr, yte):
        m = HistGradientBoostingRegressor(**REG_PARAMS).fit(Xtr, ytr)
        p = np.clip(m.predict(Xte), 0, 1)
        return dict(r2=r2_score(yte, p), mae=mean_absolute_error(yte, p),
                    rmse=float(np.sqrt(mean_squared_error(yte, p)))), m

    def _clf_single(htr, hte):
        m = HistGradientBoostingClassifier(**CLF_PARAMS).fit(Xtr, htr)
        p = m.predict_proba(Xte)[:, 1]
        pred = (p > 0.5).astype(int)
        return dict(auc=roc_auc_score(hte, p), prauc=average_precision_score(hte, p),
                    precision=precision_score(hte, pred, zero_division=0),
                    recall=recall_score(hte, pred, zero_division=0),
                    f1=f1_score(hte, pred, zero_division=0))

    reg90_s, reg90_model = _reg_single(y90tr, y90te)
    reg365_s, reg365_model = _reg_single(y365tr, y365te)
    clf90_s = _clf_single(h90tr, h90te)
    clf365_s = _clf_single(h365tr, h365te)

    # ── 2) GroupKFold 誠實（含每折 std）──
    gkf = GroupKFold(n_splits=5)

    def _gkf(y_vals, h_vals):
        r2s, maes, aucs, praucs = [], [], [], []
        for tr, te in gkf.split(X, y_vals, groups):
            rm = HistGradientBoostingRegressor(**REG_PARAMS).fit(X.iloc[tr], y_vals[tr])
            rp = np.clip(rm.predict(X.iloc[te]), 0, 1)
            r2s.append(r2_score(y_vals[te], rp))
            maes.append(mean_absolute_error(y_vals[te], rp))
            if h_vals[tr].sum() >= 5 and h_vals[te].sum() >= 5:
                cm = HistGradientBoostingClassifier(**CLF_PARAMS).fit(X.iloc[tr], h_vals[tr])
                cp = cm.predict_proba(X.iloc[te])[:, 1]
                aucs.append(roc_auc_score(h_vals[te], cp))
                praucs.append(average_precision_score(h_vals[te], cp))
        return dict(r2=float(np.mean(r2s)), r2_std=float(np.std(r2s)),
                    mae=float(np.mean(maes)),
                    auc=float(np.mean(aucs)), auc_std=float(np.std(aucs)),
                    prauc=float(np.mean(praucs)))

    gkf90 = _gkf(y90, h90)
    gkf365 = _gkf(y365, h365)

    # ── 3) 校準機率警報（GroupKFold OOF · 90 天）──
    oof = np.full(len(X), np.nan)
    for tr, te in gkf.split(X, h90, groups):
        cal = CalibratedClassifierCV(
            HistGradientBoostingClassifier(**CLF_PARAMS), method="isotonic", cv=3)
        cal.fit(X.iloc[tr], h90[tr])
        oof[te] = cal.predict_proba(X.iloc[te])[:, 1]

    def _alert(th):
        pred = (oof >= th).astype(int)
        return dict(precision=float(precision_score(h90, pred, zero_division=0)),
                    coverage=float(pred.mean()))

    alerts = {"red": _alert(RED_TH), "yellow": _alert(YELLOW_TH)}

    # ── 4) Permutation Importance Top12 ──
    def _imp(model, yte):
        pi = permutation_importance(model, Xte, yte, n_repeats=5, random_state=SEED)
        s = pd.Series(pi.importances_mean, index=FEATURES).sort_values(ascending=False)
        return [{"key": k, "zh": zh(k), "value": round(float(v), 4)}
                for k, v in s.head(12).items()]

    imp90 = _imp(reg90_model, y90te)
    imp365 = _imp(reg365_model, y365te)

    out = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_script": "scripts/04_eval/build_data_analysis_json.py",
        "model": {
            "reg": "HistGradientBoostingRegressor",
            "clf": "HistGradientBoostingClassifier + Isotonic 校準",
            "compare": "XGBoost（對照）",
        },
        "n_samples": int(len(df)),
        "n_features": len(FEATURES),
        "features": [{"key": k, "zh": zh(k)} for k in FEATURES],
        "label_def_90": "Y_vacancy_90 > 0.70（未來90天空置>63天）",
        "base_rate_90": round(float(h90.mean()), 4),
        "base_rate_365": round(float(h365.mean()), 4),
        "thresholds": {"red": RED_TH, "yellow": YELLOW_TH},
        "single_split": {
            "reg": {"r2_90": round(reg90_s["r2"], 4), "r2_365": round(reg365_s["r2"], 4),
                    "mae_90": round(reg90_s["mae"], 4), "mae_365": round(reg365_s["mae"], 4),
                    "rmse_90": round(reg90_s["rmse"], 4)},
            "clf": {"auc_90": round(clf90_s["auc"], 4), "auc_365": round(clf365_s["auc"], 4),
                    "prauc_90": round(clf90_s["prauc"], 4),
                    "precision_90": round(clf90_s["precision"], 4),
                    "recall_90": round(clf90_s["recall"], 4),
                    "f1_90": round(clf90_s["f1"], 4)},
        },
        "groupkfold": {
            "reg": {"r2_90": round(gkf90["r2"], 4), "r2_90_std": round(gkf90["r2_std"], 4),
                    "r2_365": round(gkf365["r2"], 4), "mae_90": round(gkf90["mae"], 4)},
            "clf": {"auc_90": round(gkf90["auc"], 4), "auc_90_std": round(gkf90["auc_std"], 4),
                    "auc_365": round(gkf365["auc"], 4),
                    "prauc_90": round(gkf90["prauc"], 4), "prauc_365": round(gkf365["prauc"], 4)},
        },
        "alerts_90": {
            "red": {"precision": round(alerts["red"]["precision"], 4),
                    "coverage": round(alerts["red"]["coverage"], 4)},
            "yellow": {"precision": round(alerts["yellow"]["precision"], 4),
                       "coverage": round(alerts["yellow"]["coverage"], 4)},
        },
        "importance_90": imp90,
        "importance_365": imp365,
    }

    path = os.path.join(BASE, "models", "eval_vacancy_90.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("== 數據分析 JSON 已輸出 ==", path)
    print(f"特徵數={out['n_features']} 樣本={out['n_samples']} "
          f"基準率90={out['base_rate_90']}")
    print(f"[誠實 GroupKFold] 90天 AUC={gkf90['auc']:.4f}±{gkf90['auc_std']:.3f} "
          f"R²={gkf90['r2']:.4f}±{gkf90['r2_std']:.3f} | "
          f"365天 AUC={gkf365['auc']:.4f} R²={gkf365['r2']:.4f}")
    print(f"[單次切分] 90天 AUC={clf90_s['auc']:.4f} R²={reg90_s['r2']:.4f}")
    print(f"[警報] 紅 P={alerts['red']['precision']:.3f}/cov={alerts['red']['coverage']:.3f} "
          f"黃 P={alerts['yellow']['precision']:.3f}/cov={alerts['yellow']['coverage']:.3f}")
    print("[90天 Top5]", [f"{d['zh']}={d['value']}" for d in imp90[:5]])


if __name__ == "__main__":
    main()
