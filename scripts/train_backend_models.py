# -*- coding: utf-8 -*-
"""train_backend_models.py — v4 後台雙模型訓練(LightGBM 主力 + XGBoost 對照)

依 2026-07-19 決議:
  • 特徵集:dataset_multimodal 58 特徵(POI + NLP 全保留)
  • 標籤:Y_high_risk_06 = (Y_vacancy >= 0.6),基準率約 37%
  • 模型:LightGBM(主力)與 XGBoost(對照),皆 Isotonic 校準
  • 雙層警報:紅色 = 校準機率 >= 0.60(P約0.70)、黃色 = >= 0.35(整體R約0.70)
  • 誠實評估:GroupKFold(host_id, 5 折)OOF;另附單次隨機切分(樂觀)對照
  • 雙模型策略:完整(58 特徵)/ 冷啟動(移除 7 個房東身分特徵 = 51)

分段執行(單段皆 < 45 秒,可重複執行,已完成段自動跳過):
    python -X utf8 scripts/train_backend_models.py --part full
    python -X utf8 scripts/train_backend_models.py --part cold
    python -X utf8 scripts/train_backend_models.py --part final

產出:models/backend_models_v2.joblib · eval_results.json · shap_cache.joblib
      competitor_index.pkl · suggestion_engine.pkl · data/_model_metrics.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (auc, f1_score, mean_squared_error,
                             precision_recall_curve, precision_score, r2_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
SEED = 42
RED_TH, YELLOW_TH = 0.60, 0.35          # 雙層警報門檻(依 doc/01 分析報告)
LABEL_TH = 0.60                          # 高風險標籤:Y_vacancy >= 0.6

NON_FEATURES = {"id", "host_id", "Y_vacancy", "Y_high_risk",
                "neighbourhood_cleansed", "room_type"}
HOST_IDENTITY = ["host_acceptance_rate", "host_response_rate", "response_speed",
                 "host_is_superhost", "host_listings_count",
                 "calculated_host_listings_count", "host_tenure_days"]


def log(msg):
    print(f"[train-v4] {msg}", flush=True)


def make_lgbm_clf():
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=500, num_leaves=31, learning_rate=.05,
                          subsample=.8, colsample_bytree=.8, n_jobs=-1,
                          random_state=SEED, verbose=-1)


def make_xgb_clf():
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=400, max_depth=5, learning_rate=.05,
                         subsample=.8, colsample_bytree=.8, reg_lambda=1.0,
                         n_jobs=-1, random_state=SEED, eval_metric="logloss")


def make_lgbm_reg():
    from lightgbm import LGBMRegressor
    return LGBMRegressor(n_estimators=500, num_leaves=31, learning_rate=.05,
                         subsample=.8, colsample_bytree=.8, n_jobs=-1,
                         random_state=SEED, verbose=-1)


NEG_WORDS = ["dirty", "noisy", "bad", "rude", "smell", "broken", "terrible",
             "worst", "cockroach", "bug", "uncomfortable", "髒", "吵", "差",
             "臭", "爛", "糟", "壞", "不乾淨", "不舒服", "失望"]


def build_nlp_extra() -> pd.DataFrame:
    """負評比例特徵(docx To-Do:評論情緒與負評比例):
    每房源 neg_review_ratio = 含負向詞評論數 ÷ 總評論數;無評論者為 0。
    結果快取至 data/_nlp_extra.csv,重跑訓練不需重算 21 萬筆評論。
    """
    cache = DATA_DIR / "_nlp_extra.csv"
    if cache.exists():
        return pd.read_csv(cache)
    head = pd.read_csv(DATA_DIR / "reviews_cleaned.csv.gz", nrows=2)
    text_col = ("cleaned_comments" if "cleaned_comments" in head.columns
                else "comments")
    rv = pd.read_csv(DATA_DIR / "reviews_cleaned.csv.gz",
                     usecols=["listing_id", text_col])
    txt = rv[text_col].astype(str).str.lower()
    neg = pd.Series(False, index=rv.index)
    for w in NEG_WORDS:
        neg |= txt.str.contains(w, regex=False)
    rv["is_neg"] = neg.astype(int)
    agg = (rv.groupby("listing_id")["is_neg"]
           .agg(["mean", "count"]).reset_index()
           .rename(columns={"listing_id": "id", "mean": "neg_review_ratio",
                            "count": "review_cnt_nlp"}))
    out = agg[["id", "neg_review_ratio"]]
    out.to_csv(cache, index=False, encoding="utf-8")
    return out


def load_xy():
    df = pd.read_csv(DATA_DIR / "dataset_multimodal.csv", encoding="utf-8-sig")
    extra = build_nlp_extra()
    df = df.merge(extra, on="id", how="left")
    df["neg_review_ratio"] = df["neg_review_ratio"].fillna(0.0)  # 無評論 → 0
    feats_full = [c for c in df.columns if c not in NON_FEATURES]
    y_cls = (pd.to_numeric(df["Y_vacancy"], errors="coerce") >= LABEL_TH).astype(int)
    y_reg = pd.to_numeric(df["Y_vacancy"], errors="coerce").fillna(0)
    return df, feats_full, y_cls, y_reg


def pr_at(y, p, th):
    pred = (p >= th).astype(int)
    if pred.sum() == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n_flag": 0}
    return {"precision": float(precision_score(y, pred)),
            "recall": float(recall_score(y, pred)),
            "f1": float(f1_score(y, pred)), "n_flag": int(pred.sum())}


def honest_oof(X, y_cls, y_reg, groups, make_clf):
    """GroupKFold OOF:分類(校準機率)與迴歸(空屋率)。"""
    oof_p = np.full(len(X), np.nan)
    oof_r = np.full(len(X), np.nan)
    auc_folds, r2_folds = [], []
    for tr, va in GroupKFold(5).split(X, y_cls, groups):
        clf = CalibratedClassifierCV(make_clf(), method="isotonic", cv=3)
        clf.fit(X.iloc[tr], y_cls.iloc[tr])
        oof_p[va] = clf.predict_proba(X.iloc[va])[:, 1]
        auc_folds.append(roc_auc_score(y_cls.iloc[va], oof_p[va]))
        reg = make_lgbm_reg().fit(X.iloc[tr], y_reg.iloc[tr])
        oof_r[va] = reg.predict(X.iloc[va])
        r2_folds.append(r2_score(y_reg.iloc[va], oof_r[va]))
    prec, rec, _ = precision_recall_curve(y_cls, oof_p)
    step = max(1, len(prec) // 300)
    return {
        "oof_prob": oof_p, "oof_reg": oof_r,
        "auc": {"mean": float(np.mean(auc_folds)), "std": float(np.std(auc_folds))},
        "r2": {"mean": float(np.mean(r2_folds)), "std": float(np.std(r2_folds))},
        "AUC_oof": float(roc_auc_score(y_cls, oof_p)),
        "PR_AUC": float(auc(rec, prec)),
        "R2_oof": float(r2_score(y_reg, oof_r)),
        "MSE_oof": float(mean_squared_error(y_reg, oof_r)),
        "red": pr_at(y_cls, oof_p, RED_TH),
        "yellow": pr_at(y_cls, oof_p, YELLOW_TH),
        "pr_curve": {"precision": prec[::step].round(4).tolist(),
                     "recall": rec[::step].round(4).tolist()},
        "r2_folds": [round(float(x), 3) for x in r2_folds],
    }


def single_split_eval(X, y_cls, y_reg):
    """單次隨機切分(樂觀、含房東洩漏)+ 線性基準線 —— 對照展示用。"""
    Xtr, Xte, ytr, yte, rtr, rte = train_test_split(
        X, y_cls, y_reg, test_size=.2, random_state=SEED, stratify=y_cls)
    reg = make_lgbm_reg().fit(Xtr, rtr)
    clf = CalibratedClassifierCV(make_lgbm_clf(), method="isotonic", cv=3)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    lin = make_pipeline(SimpleImputer(strategy="median"),
                        LinearRegression()).fit(Xtr, rtr)
    logi = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(max_iter=2000)).fit(Xtr, ytr)
    return {
        "reg_r2": float(r2_score(rte, reg.predict(Xte))),
        "clf": {"auc": float(roc_auc_score(yte, p)),
                **pr_at(yte, p, RED_TH)},
        "baseline_reg_r2": float(r2_score(rte, lin.predict(Xte))),
        "baseline_clf_auc": float(
            roc_auc_score(yte, logi.predict_proba(Xte)[:, 1])),
    }


def train_variant(variant: str):
    """訓練單一變體(full/cold):誠實 OOF ×(LGBM+XGB)+ 單次切分 + 全量最終模型。"""
    t0 = time.time()
    df, feats_full, y_cls, y_reg = load_xy()
    feats = (feats_full if variant == "full"
             else [f for f in feats_full if f not in HOST_IDENTITY])
    X = df[feats].apply(pd.to_numeric, errors="coerce")
    g = df["host_id"]
    log(f"{variant}: {len(feats)} features, base_rate={y_cls.mean():.3f}")

    def _ckpt(name, fn):
        """細粒度 checkpoint:已完成的計算不重跑(45 秒工具逾時防護)。"""
        p = MODEL_DIR / f"_ck_{variant}_{name}.joblib"
        if p.exists():
            return joblib.load(p)
        out = fn()
        MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump(out, p)
        log(f"{variant}/{name} checkpointed")
        return out

    honest_lgbm = _ckpt("hlgbm", lambda: honest_oof(X, y_cls, y_reg, g, make_lgbm_clf))
    log(f"{variant} LGBM OOF AUC={honest_lgbm['AUC_oof']:.3f} "
        f"red P={honest_lgbm['red']['precision']:.2f}/R={honest_lgbm['red']['recall']:.2f} "
        f"yellow R={honest_lgbm['yellow']['recall']:.2f}")
    honest_xgb = _ckpt("hxgb", lambda: honest_oof(X, y_cls, y_reg, g, make_xgb_clf))
    log(f"{variant} XGB  OOF AUC={honest_xgb['AUC_oof']:.3f}")
    single = _ckpt("single", lambda: single_split_eval(X, y_cls, y_reg))

    # 全量最終模型(推論用)
    reg_model = make_lgbm_reg().fit(X, y_reg)
    clf_lgbm = CalibratedClassifierCV(make_lgbm_clf(), method="isotonic",
                                      cv=3).fit(X, y_cls)
    clf_xgb = CalibratedClassifierCV(make_xgb_clf(), method="isotonic",
                                     cv=3).fit(X, y_cls)

    part = {
        "feature_names": feats,
        "reg_model": reg_model, "clf_model": clf_lgbm, "clf_xgb": clf_xgb,
        "threshold": RED_TH,
        "honest": {"lgbm": {k: v for k, v in honest_lgbm.items()
                            if not k.startswith("oof")},
                   "xgb": {k: v for k, v in honest_xgb.items()
                           if not k.startswith("oof")}},
        "single_split": single,
        "oof_prob_lgbm": honest_lgbm["oof_prob"],
        "oof_reg_lgbm": honest_lgbm["oof_reg"],
        "oof_prob_xgb": honest_xgb["oof_prob"],
    }
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(part, MODEL_DIR / f"_part_{variant}.joblib")
    log(f"{variant} done in {time.time() - t0:.0f}s")


def _ev_block(part):
    """轉為 backend_v2_sections 期望的 eval schema。"""
    h = part["honest"]["lgbm"]
    return {
        "single_split": part["single_split"],
        "groupkfold": {"r2": h["r2"], "auc": h["auc"]},
        "dual_threshold": {"red_th": RED_TH, "yellow_th": YELLOW_TH,
                           "red": h["red"], "yellow": h["yellow"],
                           "PR_AUC": h["PR_AUC"]},
        "pr_curve": h["pr_curve"],
        "xgb_groupkfold": {"r2": part["honest"]["xgb"]["r2"],
                           "auc": part["honest"]["xgb"]["auc"],
                           "red": part["honest"]["xgb"]["red"],
                           "yellow": part["honest"]["xgb"]["yellow"],
                           "PR_AUC": part["honest"]["xgb"]["PR_AUC"]},
    }


def finalize():
    t0 = time.time()
    full = joblib.load(MODEL_DIR / "_part_full.joblib")
    cold = joblib.load(MODEL_DIR / "_part_cold.joblib")
    df, feats_full, y_cls, y_reg = load_xy()

    # ---------- bundle ----------
    bundle = {
        "label_def": f"Y_vacancy >= {LABEL_TH}",
        "red_th": RED_TH, "yellow_th": YELLOW_TH, "primary": "lgbm",
        "full": {k: full[k] for k in
                 ["feature_names", "reg_model", "clf_model", "clf_xgb", "threshold"]},
        "cold": {k: cold[k] for k in
                 ["feature_names", "reg_model", "clf_model", "clf_xgb", "threshold"]},
    }
    joblib.dump(bundle, MODEL_DIR / "backend_models_v2.joblib")

    # ---------- eval_results.json ----------
    ev = {
        f"完整模型_{len(full['feature_names'])}特徵": _ev_block(full),
        f"冷啟動模型_{len(cold['feature_names'])}特徵": _ev_block(cold),
        "label_def": bundle["label_def"],
        "primary_model": "LightGBM(Isotonic 校準);XGBoost 對照",
    }
    (MODEL_DIR / "eval_results.json").write_text(
        json.dumps(ev, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---------- shap_cache.joblib ----------
    import shap
    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(df), size=min(500, len(df)), replace=False)
    cache = {"meta": {"取樣筆數": int(len(idx))}}
    for name, part in [("full", full), ("cold", cold)]:
        feats = part["feature_names"]
        Xs = df[feats].apply(pd.to_numeric, errors="coerce").iloc[idx]
        expl = shap.TreeExplainer(part["reg_model"])
        sv = expl.shap_values(Xs)
        pred = part["reg_model"].predict(Xs)
        cache[name] = {
            "shap_values": sv,
            "base_value": float(np.ravel(expl.expected_value)[0]),
            "X_sample": Xs.reset_index(drop=True),
            "feature_names": feats,
            "method": "TreeExplainer(LightGBM 迴歸)",
            "listing_ids": df["id"].iloc[idx].tolist(),
            "risk_pred": pred,
            "example_high_idx": int(np.argmax(pred)),
            "example_low_idx": int(np.argmin(pred)),
        }
    joblib.dump(cache, MODEL_DIR / "shap_cache.joblib")

    # ---------- 跨平台競品索引 + 建議引擎(可抽換 PKL) ----------
    from modules.market_data import load_all_market
    from modules.competitor import CompetitorIndex
    from modules.suggestion import SuggestionEngine
    from modules.pkl_store import save_module
    market = load_all_market()
    log(f"market rows={len(market)} "
        f"{market['platform'].value_counts().to_dict()}")
    save_module("competitor_index", CompetitorIndex(market),
                schema="CompetitorIndex(BallTree, Airbnb/Booking/591/ddroom)")
    save_module("suggestion_engine", SuggestionEngine(),
                schema="SuggestionEngine(rules)")

    # ---------- 全量預測輸出(熱力圖 / 同商圈排名 / 60% 通知中心) ----------
    cold_mask = (df["calculated_host_listings_count"].isna()
                 | (pd.to_numeric(df["calculated_host_listings_count"],
                                  errors="coerce") <= 1))
    vac = np.full(len(df), np.nan)
    prob = np.full(len(df), np.nan)
    prob_x = np.full(len(df), np.nan)
    # 使用 GroupKFold OOF 誠實機率(非樣本內擬合值),通知中心不虛胖
    for name, part, mask in [("full", full, ~cold_mask),
                             ("cold", cold, cold_mask)]:
        if mask.sum() == 0:
            continue
        vac[mask.values] = np.clip(part["oof_reg_lgbm"][mask.values], 0, 1)
        prob[mask.values] = part["oof_prob_lgbm"][mask.values]
        prob_x[mask.values] = part["oof_prob_xgb"][mask.values]

    def _tier_of(p):
        return np.where(p >= RED_TH, "red",
                        np.where(p >= YELLOW_TH, "yellow", "green"))
    tier = _tier_of(prob)
    preds = df[["id", "host_id", "latitude", "longitude",
                "neighbourhood_cleansed", "room_type", "price",
                "accommodates"]].copy()
    preds["vac_pred"] = vac.round(4)
    preds["prob"] = prob.round(4)
    preds["tier"] = tier
    preds["prob_xgb"] = prob_x.round(4)
    preds["tier_xgb"] = _tier_of(prob_x)
    preds["variant"] = np.where(cold_mask, "cold", "full")
    preds.to_csv(DATA_DIR / "_predictions.csv", index=False, encoding="utf-8")
    log(f"predictions: {len(preds)} rows "
        f"(red={int((tier == 'red').sum())}, yellow={int((tier == 'yellow').sum())})")

    # ---------- data/_model_metrics.json(舊介面相容) ----------
    h = full["honest"]["lgbm"]
    m = {"n": int(len(df)), "n_features": len(full["feature_names"]),
         "high_risk_rate": float(y_cls.mean()),
         "R2": h["R2_oof"], "MSE": h["MSE_oof"],
         "AUC": h["AUC_oof"],
         "Recall": h["red"]["recall"], "F1": h["red"]["f1"],
         "R2_folds": h["r2_folds"],
         "label_def": bundle["label_def"]}
    (DATA_DIR / "_model_metrics.json").write_text(
        json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")

    for p in list(MODEL_DIR.glob("_part_*.joblib")) + \
             list(MODEL_DIR.glob("_ck_*.joblib")):
        try:
            p.unlink(missing_ok=True)
        except OSError as e:  # 沙盒掛載目錄禁止刪除時不中斷(僅提示)
            log(f"暫存檔待手動清理:{p.name}({e})")
    log(f"finalize done in {time.time() - t0:.0f}s → {MODEL_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["full", "cold", "final", "all"],
                    default="all")
    args = ap.parse_args()
    if args.part in ("full", "all") and not (MODEL_DIR / "_part_full.joblib").exists():
        train_variant("full")
    if args.part in ("cold", "all") and not (MODEL_DIR / "_part_cold.joblib").exists():
        train_variant("cold")
    if args.part in ("final", "all"):
        finalize()
