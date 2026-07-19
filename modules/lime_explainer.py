# -*- coding: utf-8 -*-
"""LIME 可解釋性模組(docx §四:LIME 可解釋性模組)。

對單一房源解釋「為何被判定高空屋風險」:以 LimeTabularExplainer 對
模型 B(校準分類器)的 P(空屋率≥60%) 做局部線性近似,輸出 Top-K 原因
(繁中規則 + 權重)。權重正值 = 推高風險,負值 = 降低風險。
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

try:
    import streamlit as st
    cache_resource = st.cache_resource
    cache_data = st.cache_data
except Exception:                       # 允許無 Streamlit 環境測試
    def cache_resource(*a, **k):
        def deco(f):
            return f
        return deco if not a else a[0]
    cache_data = cache_resource

from modules import feature_engineering as fe


@cache_resource(show_spinner="初始化 LIME 解釋器 …")
def _explainer(variant: str):
    """以訓練資料分佈建立 LimeTabularExplainer(每個變體各建一次)。"""
    from lime.lime_tabular import LimeTabularExplainer
    bundle = fe.load_bundle()
    feats = bundle[variant]["feature_names"]
    df = fe.load_dataset_final()
    X = df[feats].apply(pd.to_numeric, errors="coerce")
    med = X.median(numeric_only=True)
    Xf = X.fillna(med)
    expl = LimeTabularExplainer(
        Xf.to_numpy(), feature_names=feats,
        class_names=["正常", "高風險"], mode="classification",
        discretize_continuous=True, random_state=42)
    return expl, feats, med


def _zh_rule(rule: str, feats: list) -> str:
    """把 LIME 規則字串的英文特徵名換成繁中(保留條件式)。"""
    out = rule
    # 由長到短比對,避免子字串誤替換(如 price 誤傷 price_pctl_nbhd)
    for f in sorted(feats, key=len, reverse=True):
        if f in out:
            out = out.replace(f, fe.FEAT_ZH_V2.get(f, f))
    # 數字取兩位小數,規則更好讀
    out = re.sub(r"(\d+\.\d{3,})", lambda m: f"{float(m.group(1)):.2f}", out)
    return out


def lime_reasons(row: pd.Series, variant: str, algo: str = "lgbm",
                 overrides: dict | None = None, k: int = 3,
                 num_samples: int = 2000) -> list:
    """回傳該房源的 LIME Top-K 風險原因。

    參數:row 資料列、variant full/cold、algo lgbm/xgb、overrides 沙盒覆寫值。
    回傳:[{rule, zh, weight_pp, direction}],依 |weight| 排序;
          weight_pp 單位 = 高風險機率百分點。
    例外:lime 未安裝時丟出 ImportError(呼叫端顯示安裝指引)。
    """
    expl, feats, med = _explainer(variant)
    bundle = fe.load_bundle()
    m = bundle[variant]
    clf = m["clf_xgb"] if (algo == "xgb" and "clf_xgb" in m) else m["clf_model"]

    ov = dict(overrides or {})
    x = []
    for f in feats:
        v = ov.get(f, row.get(f))
        try:
            v = float(v)
            if np.isnan(v):
                v = float(med.get(f, 0.0))
        except (TypeError, ValueError):
            v = float(med.get(f, 0.0))
        x.append(v)
    x = np.asarray(x)

    def predict_fn(arr):
        return clf.predict_proba(pd.DataFrame(arr, columns=feats))

    exp = expl.explain_instance(x, predict_fn, num_features=max(k * 3, 8),
                                num_samples=num_samples, labels=(1,))
    items = exp.as_list(label=1)
    out = [{"rule": r, "zh": _zh_rule(r, feats),
            "weight_pp": round(w * 100, 2),
            "direction": "up" if w > 0 else "down"}
           for r, w in items]
    out.sort(key=lambda d: -abs(d["weight_pp"]))
    return out[:k * 2]  # 回傳 2K 個,前端取正向 Top-K 與加分項對照
