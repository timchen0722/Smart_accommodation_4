# -*- coding: utf-8 -*-
"""可抽換 PKL 模組之統一存取層。

每個 pkl 結構:{"meta": {name, version, built_at, schema}, "payload": <物件>}
任一模組可單獨重建替換,載入時檢查 meta。
"""
from __future__ import annotations

import pickle
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

MODULE_NAMES = [
    "preprocessor", "regressor", "classifier",
    "explainer", "competitor_index", "suggestion_engine",
]


def save_module(name: str, payload, schema: str = "", version: str = "1.0"):
    MODELS.mkdir(exist_ok=True)
    obj = {
        "meta": {
            "name": name,
            "version": version,
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "schema": schema,
        },
        "payload": payload,
    }
    with open(MODELS / f"{name}.pkl", "wb") as f:
        pickle.dump(obj, f)


def load_module(name: str, with_meta: bool = False):
    path = MODELS / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"找不到模組 {path.name},請先執行:python src/train.py")
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict) or "payload" not in obj:
        raise ValueError(f"{path.name} 非標準模組格式")
    return (obj["payload"], obj["meta"]) if with_meta else obj["payload"]


def modules_status() -> list:
    """回傳各模組檔案狀態(後台顯示用)。"""
    out = []
    for n in MODULE_NAMES:
        p = MODELS / f"{n}.pkl"
        if p.exists():
            try:
                with open(p, "rb") as f:
                    meta = pickle.load(f).get("meta", {})
            except Exception:
                meta = {}
            out.append({"module": n, "exists": True,
                        "size_mb": p.stat().st_size / 1e6, **meta})
        else:
            out.append({"module": n, "exists": False})
    return out
