# -*- coding: utf-8 -*-
"""train_backend_models_v90.py — 全站換模：37 特徵 · HistGradientBoosting · Y=vacancy_90

2026-07-23 決議（見 docs/superpowers/plans/2026-07-23-全站換37特徵HistGB-vacancy90模型.md）：
  • 資料：data/dataset_multimodal.csv（＝App 上線讀取的同一份，train=serve 無 skew）
          ＋ listings_cleaned.csv.gz（取 availability_90）＋ data/_core_extra.csv（橋接 2 欄）
  • 特徵：37 核心（bathrooms→bathrooms_count；property_type_code / photo_design_sense 橋接）
  • 模型：HistGradientBoosting（回歸＋分類）· Isotonic 校準 · XGBoost 對照
  • 目標：主 Y_vacancy_90 = availability_90/90；營收 Y_vacancy_365 = Y_vacancy（雙輸出）
  • 高風險：vacancy_90 > 0.70（基準率≈37%）· 雙層警報 紅0.60/黃0.35
  • 雙變體：完整（37）/ 冷啟動（移除 6 個房東身分特徵）
產出：models/backend_models_v2.joblib · eval_results.json · shap_cache.joblib
      data/_predictions.csv（含 vac_pred_365）· data/_model_metrics.json

執行：python -X utf8 scripts/train_backend_models_v90.py
"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (average_precision_score, precision_score,
                             r2_score, recall_score, mean_squared_error,
                             roc_auc_score)

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
SEED = 42
RED_TH, YELLOW_TH = 0.60, 0.35
HIGH_TH = 0.70                       # 高風險：vacancy_90 > 0.70
REG_PARAMS = dict(max_iter=500, learning_rate=0.05, l2_regularization=1.0,
                  random_state=SEED)
CLF_PARAMS = dict(max_iter=500, learning_rate=0.05, l2_regularization=1.0,
                  random_state=SEED)
# 37 個核心特徵（root dataset_final 定義，bathrooms→bathrooms_count 對映上線資料）
HOST_IDENTITY = ["host_acceptance_rate", "host_response_rate", "response_speed",
                 "host_is_superhost", "host_listings_count", "host_tenure_days"]


def log(m):
    print(f"[v90] {m}", flush=True)


def make_reg():
    return HistGradientBoostingRegressor(**REG_PARAMS)


def make_clf():
    return HistGradientBoostingClassifier(**CLF_PARAMS)


def make_xgb():
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=400, max_depth=5, learning_rate=.05,
                         subsample=.8, colsample_bytree=.8, reg_lambda=1.0,
                         n_jobs=-1, random_state=SEED, eval_metric="logloss")


def load_data():
    """組出訓練資料：上線多模態資料 ＋ availability_90 ＋ 橋接 2 欄。"""
    df = pd.read_csv(DATA_DIR / "dataset_multimodal.csv", encoding="utf-8-sig")
    lst = pd.read_csv(ROOT / "listings_cleaned.csv.gz", compression="gzip",
                      low_memory=False)[["id", "availability_90"]]
    df = df.merge(lst, on="id", how="left")
    extra = pd.read_csv(DATA_DIR / "_core_extra.csv")
    df = df.merge(extra, on="id", how="left")
    # 37 核心特徵（root 定義；bathrooms 以 bathrooms_count 對映）
    root_cols = pd.read_csv(ROOT / "dataset_final.csv", nrows=1).columns
    feats_root = [c for c in root_cols
                  if c not in ["listing_id", "Y_vacancy"]
                  and (not c.startswith("photo_") or c == "photo_design_sense")]
    feats = ["bathrooms_count" if c == "bathrooms" else c for c in feats_root]
    miss = [f for f in feats if f not in df.columns]
    assert not miss, f"上線資料缺特徵：{miss}"
    df["Y_vacancy_90"] = (pd.to_numeric(df["availability_90"], errors="coerce")
                          / 90.0).clip(0, 1)
    df["Y_vacancy_365"] = pd.to_numeric(df["Y_vacancy"], errors="coerce").fillna(0)
    df = df.dropna(subset=["Y_vacancy_90"]).reset_index(drop=True)
    return df, feats


def honest_oof(X, y90, y365, h90, groups):
    """GroupKFold(5) OOF：vac90 / vac365 迴歸 ＋ 校準分類機率（含每折指標）。"""
    n = len(X)
    oof90 = np.full(n, np.nan)
    oof365 = np.full(n, np.nan)
    oofp = np.full(n, np.nan)
    r2f, aucf = [], []
    for tr, va in GroupKFold(5).split(X, h90, groups):
        r90 = make_reg().fit(X.iloc[tr], y90[tr])
        oof90[va] = np.clip(r90.predict(X.iloc[va]), 0, 1)
        r2f.append(r2_score(y90[va], oof90[va]))
        r365 = make_reg().fit(X.iloc[tr], y365[tr])
        oof365[va] = np.clip(r365.predict(X.iloc[va]), 0, 1)
        cal = CalibratedClassifierCV(make_clf(), method="isotonic", cv=3)
        cal.fit(X.iloc[tr], h90[tr])
        oofp[va] = cal.predict_proba(X.iloc[va])[:, 1]
        aucf.append(roc_auc_score(h90[va], oofp[va]))
    return oof90, oof365, oofp, r2f, aucf


def honest_oof_xgb(X, h90, groups):
    n = len(X)
    oofx = np.full(n, np.nan)
    for tr, va in GroupKFold(5).split(X, h90, groups):
        cal = CalibratedClassifierCV(make_xgb(), method="isotonic", cv=3)
        cal.fit(X.iloc[tr], h90[tr])
        oofx[va] = cal.predict_proba(X.iloc[va])[:, 1]
    return oofx


def fit_variant(df, feats):
    """回傳單一變體的 OOF + 全量最終模型（reg90 / reg365 / clf / xgb）。"""
    X = df[feats].apply(pd.to_numeric, errors="coerce")
    y90 = df["Y_vacancy_90"].to_numpy()
    y365 = df["Y_vacancy_365"].to_numpy()
    h90 = (y90 > HIGH_TH).astype(int)
    g = df["host_id"].to_numpy()

    oof90, oof365, oofp, r2f, aucf = honest_oof(X, y90, y365, h90, g)
    oofx = honest_oof_xgb(X, h90, g)

    reg90 = make_reg().fit(X, y90)
    reg365 = make_reg().fit(X, y365)
    clf = CalibratedClassifierCV(make_clf(), method="isotonic", cv=3).fit(X, h90)
    xgb = CalibratedClassifierCV(make_xgb(), method="isotonic", cv=3).fit(X, h90)

    return {
        "feature_names": feats,
        "reg_model": reg90, "reg_model_365": reg365,
        "clf_model": clf, "clf_xgb": xgb, "threshold": RED_TH,
        "oof_reg90": oof90, "oof_reg365": oof365,
        "oof_prob": oofp, "oof_prob_xgb": oofx,
        "auc_oof": float(roc_auc_score(h90, oofp)),
        "auc_mean": float(np.mean(aucf)), "auc_std": float(np.std(aucf)),
        "r2_oof": float(r2_score(y90, oof90)),
        "r2_mean": float(np.mean(r2f)), "r2_std": float(np.std(r2f)),
        "mse_oof": float(mean_squared_error(y90, oof90)),
        "r2_folds": [round(float(x), 3) for x in r2f],
        "base_rate": float(h90.mean()),
    }


def _tier(p):
    return np.where(p >= RED_TH, "red", np.where(p >= YELLOW_TH, "yellow", "green"))


def main():
    t0 = time.time()
    MODEL_DIR.mkdir(exist_ok=True)
    df, feats = load_data()
    cold_feats = [f for f in feats if f not in HOST_IDENTITY]
    log(f"樣本={len(df)} 完整特徵={len(feats)} 冷啟動特徵={len(cold_feats)} "
        f"高風險基準率={float(((df['Y_vacancy_90']>HIGH_TH)).mean()):.3f}")

    log("訓練 full 變體…")
    full = fit_variant(df, feats)
    log(f"full 誠實 AUC={full['auc_mean']:.3f}±{full['auc_std']:.3f} "
        f"R²={full['r2_mean']:.3f}±{full['r2_std']:.3f}")
    log("訓練 cold 變體…")
    cold = fit_variant(df, cold_feats)
    log(f"cold 誠實 AUC={cold['auc_mean']:.3f}±{cold['auc_std']:.3f} "
        f"R²={cold['r2_mean']:.3f}±{cold['r2_std']:.3f}")

    # ── bundle ──
    keys = ["feature_names", "reg_model", "reg_model_365",
            "clf_model", "clf_xgb", "threshold"]
    bundle = {
        "label_def": f"Y_vacancy_90 > {HIGH_TH}",
        "red_th": RED_TH, "yellow_th": YELLOW_TH, "primary": "histgb",
        "full": {k: full[k] for k in keys},
        "cold": {k: cold[k] for k in keys},
    }
    joblib.dump(bundle, MODEL_DIR / "backend_models_v2.joblib")
    log("→ backend_models_v2.joblib")

    # ── _predictions.csv（依房東規模路由 full/cold；OOF 誠實機率）──
    ccl = pd.to_numeric(df.get("calculated_host_listings_count"), errors="coerce")
    cold_mask = ccl.isna() | (ccl <= 1)
    vac = np.where(cold_mask, cold["oof_reg90"], full["oof_reg90"])
    vac365 = np.where(cold_mask, cold["oof_reg365"], full["oof_reg365"])
    prob = np.where(cold_mask, cold["oof_prob"], full["oof_prob"])
    probx = np.where(cold_mask, cold["oof_prob_xgb"], full["oof_prob_xgb"])
    preds = df[["id", "host_id", "latitude", "longitude", "neighbourhood_cleansed",
                "room_type", "price", "accommodates"]].copy()
    preds["vac_pred"] = np.round(vac, 4)
    preds["vac_pred_365"] = np.round(vac365, 4)
    preds["prob"] = np.round(prob, 4)
    preds["tier"] = _tier(prob)
    preds["prob_xgb"] = np.round(probx, 4)
    preds["tier_xgb"] = _tier(probx)
    preds["variant"] = np.where(cold_mask, "cold", "full")
    preds.to_csv(DATA_DIR / "_predictions.csv", index=False, encoding="utf-8")
    log(f"→ _predictions.csv rows={len(preds)} "
        f"red={int((preds['tier']=='red').sum())} "
        f"yellow={int((preds['tier']=='yellow').sum())}")

    # ── eval_results.json（沿用 backend_v2 schema 供相容）──
    def _pr(h, p, th):
        pred = (p >= th).astype(int)
        if pred.sum() == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n_flag": 0}
        return {"precision": float(precision_score(h, pred, zero_division=0)),
                "recall": float(recall_score(h, pred, zero_division=0)),
                "f1": 0.0, "n_flag": int(pred.sum())}

    def _evblock(part, dfv):
        h = (dfv["Y_vacancy_90"].to_numpy() > HIGH_TH).astype(int)
        p = part["oof_prob"]
        return {
            "single_split": {"reg_r2": part["r2_oof"],
                             "clf": {"auc": part["auc_oof"]}},
            "groupkfold": {"r2": {"mean": part["r2_mean"], "std": part["r2_std"]},
                           "auc": {"mean": part["auc_mean"], "std": part["auc_std"]}},
            "dual_threshold": {"red_th": RED_TH, "yellow_th": YELLOW_TH,
                               "red": _pr(h, p, RED_TH), "yellow": _pr(h, p, YELLOW_TH),
                               "PR_AUC": float(average_precision_score(h, p))},
        }
    ev = {
        f"完整模型_{len(feats)}特徵": _evblock(full, df),
        f"冷啟動模型_{len(cold_feats)}特徵": _evblock(cold, df),
        "label_def": bundle["label_def"],
        "primary_model": "HistGradientBoosting(Isotonic 校準);XGBoost 對照",
    }
    (MODEL_DIR / "eval_results.json").write_text(
        json.dumps(ev, ensure_ascii=False, indent=1), encoding="utf-8")
    log("→ eval_results.json")

    # ── _model_metrics.json（get_metrics 相容）──
    m = {"n": int(len(df)), "n_features": len(feats),
         "high_risk_rate": full["base_rate"],
         "R2": full["r2_mean"], "MSE": full["mse_oof"], "AUC": full["auc_mean"],
         "Recall": _pr((df["Y_vacancy_90"].to_numpy() > HIGH_TH).astype(int),
                       full["oof_prob"], RED_TH)["recall"],
         "F1": 0.0, "R2_folds": full["r2_folds"], "label_def": bundle["label_def"]}
    (DATA_DIR / "_model_metrics.json").write_text(
        json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    log("→ _model_metrics.json")

    # ── shap_cache.joblib（TreeExplainer(HistGB 迴歸)）──
    import shap
    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(df), size=min(500, len(df)), replace=False)
    cache = {"meta": {"取樣筆數": int(len(idx))}}
    for name, part, fl in [("full", full, feats), ("cold", cold, cold_feats)]:
        Xs = df[fl].apply(pd.to_numeric, errors="coerce").iloc[idx].reset_index(drop=True)
        expl = shap.TreeExplainer(part["reg_model"])
        sv = expl.shap_values(Xs)
        pred = part["reg_model"].predict(Xs)
        cache[name] = {
            "shap_values": np.asarray(sv),
            "base_value": float(np.ravel(expl.expected_value)[0]),
            "X_sample": Xs, "feature_names": fl,
            "method": "TreeExplainer(HistGB 迴歸)",
            "listing_ids": df["id"].iloc[idx].tolist(),
            "risk_pred": pred,
            "example_high_idx": int(np.argmax(pred)),
            "example_low_idx": int(np.argmin(pred)),
        }
    joblib.dump(cache, MODEL_DIR / "shap_cache.joblib")
    log("→ shap_cache.joblib")
    log(f"完成，用時 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
