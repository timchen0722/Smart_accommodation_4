# -*- coding: utf-8 -*-
"""LLM 智慧建議模組:高風險房源由 LLM 生成個人化經營建議。

金鑰來源(擇一,依序嘗試):
  1. st.secrets["ANTHROPIC_API_KEY"] / 環境變數 ANTHROPIC_API_KEY → Claude
  2. st.secrets["GEMINI_API_KEY"]    / 環境變數 GEMINI_API_KEY    → Gemini
無金鑰或呼叫失敗時,呼叫端應退回規則引擎建議(不擋頁面)。
"""
from __future__ import annotations

import json
import os

import requests

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = os.environ.get("LLM_MODEL", "claude-haiku-4-5-20251001")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
# 舊模型名會 404;依序嘗試,任一成功即回傳
GEMINI_FALLBACKS = ["gemini-2.5-flash", "gemini-flash-latest", "gemini-2.0-flash"]
TIMEOUT_S = 40


def _secret(name: str):
    """依序從 st.secrets 與環境變數取金鑰。"""
    try:
        import streamlit as st
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name)


def llm_available() -> str | None:
    """回傳可用供應商("claude"/"gemini")或 None。"""
    if _secret("ANTHROPIC_API_KEY"):
        return "claude"
    if _secret("GEMINI_API_KEY"):
        return "gemini"
    return None


def build_prompt(ctx: dict) -> str:
    """把模型輸出組合成 LLM 提示詞(繁中)。

    ctx 需含:name, district, room_type, price, vac_pred, prob, tier,
    lime_reasons(list[dict]), comp_summary(str), amenity_gaps(list[str])。
    """
    reasons = "\n".join(
        f"- {r['zh']}(影響 {r['weight_pp']:+.1f} 個百分點)"
        for r in ctx.get("lime_reasons", []) if r["direction"] == "up") or "- 無"
    gaps = "、".join(ctx.get("amenity_gaps", [])) or "無明顯缺口"
    return f"""你是台北市短租市場的資深營運顧問。以下是一間 Airbnb 房源的 AI 風險評估結果,請用繁體中文為房東寫出具體可執行的改善建議。

房源資訊:{ctx.get('name', '')}|{ctx.get('district', '')}|{ctx.get('room_type', '')}|每晚 NT$ {ctx.get('price', 0):,.0f}
模型預測:未來一年空屋率 {ctx.get('vac_pred', 0):.0%},高風險機率 {ctx.get('prob', 0):.0%}(警報層級:{ctx.get('tier', '')})
LIME 判定的主要風險原因:
{reasons}
1km 跨平台競品概況:{ctx.get('comp_summary', '無資料')}
設施缺口(周邊過半競品有、本房源沒有):{gaps}

要求:
1. output 3~4 條建議,每條含「做什麼、怎麼做、預期效果」,格式為 markdown 條列。
2. 建議必須對應上述風險原因與競品數據,給出具體數字(如目標價格區間)。
3. 語氣專業但親切,不用開場白與結尾,直接輸出條列。"""


def generate_advice(ctx: dict) -> tuple[str, str]:
    """呼叫 LLM 生成建議。回傳 (provider, markdown)。

    例外:無金鑰丟 RuntimeError;網路/API 錯誤丟 requests 例外,
    呼叫端捕捉後退回規則引擎。
    """
    provider = llm_available()
    if provider is None:
        raise RuntimeError("未設定 ANTHROPIC_API_KEY 或 GEMINI_API_KEY")
    prompt = build_prompt(ctx)

    if provider == "claude":
        r = requests.post(
            ANTHROPIC_URL, timeout=TIMEOUT_S,
            headers={"x-api-key": _secret("ANTHROPIC_API_KEY"),
                     "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": ANTHROPIC_MODEL, "max_tokens": 1024,
                  "messages": [{"role": "user", "content": prompt}]})
        r.raise_for_status()
        return "claude", r.json()["content"][0]["text"]

    # ── Gemini:優先走 google-genai SDK(端點/版本自動處理),多模型名容錯 ──
    key = _secret("GEMINI_API_KEY")
    models = [GEMINI_MODEL] + [m for m in GEMINI_FALLBACKS if m != GEMINI_MODEL]
    last_err: Exception | None = None
    try:
        from google import genai
        client = genai.Client(api_key=key)
        for model in models:
            try:
                resp = client.models.generate_content(model=model,
                                                      contents=prompt)
                return f"gemini({model})", resp.text
            except Exception as e:  # 404 模型名/配額等,換下一個模型名再試
                last_err = e
        raise RuntimeError(f"Gemini 各模型皆失敗:{last_err}")
    except ImportError:
        pass  # 未裝 SDK → 退回 REST

    for model in models:
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent",
            params={"key": key}, timeout=TIMEOUT_S,
            json={"contents": [{"parts": [{"text": prompt}]}]})
        if r.status_code == 404:      # 模型名不存在 → 試下一個
            last_err = RuntimeError(f"{model}: 404 {r.text[:120]}")
            continue
        r.raise_for_status()
        return f"gemini({model})", \
            r.json()["candidates"][0]["content"]["parts"][0]["text"]
    raise RuntimeError(f"Gemini REST 各模型皆失敗:{last_err}")
