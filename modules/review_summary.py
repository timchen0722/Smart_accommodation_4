# -*- coding: utf-8 -*-
"""review_summary.py — 房源評論自動摘要(Review Summarization)

把該房源的評論濃縮成一段話,例如:
「多數旅客稱讚地點便利、服務親切與房間整潔;負面意見主要集中於隔音與停車空間。」

兩種模式
--------
rule_summary()  規則版:由 ABSA 面向統計直接組句,零成本、即時、可離線重現
llm_summary()   LLM 版:把 ABSA 結果 + 正負關鍵字餵給 LLM 生成自然語句(需手動觸發)

設計:規則版永遠可用並立即顯示;LLM 版由使用者按鈕觸發,避免每次互動都打 API。
"""
from __future__ import annotations

import pandas as pd

MIN_MENTIONS = 3          # 面向提及數低於此不納入摘要
POS_CUT = 0.55            # 正評率高於此視為「稱讚」
NEG_CUT = 0.08            # 負評率高於此視為「抱怨」
TOP_N = 3


def collect_facts(listing_id: int, reviews_df=None) -> dict:
    """蒐集摘要素材:ABSA 面向統計 + 正負關鍵字 + 評論則數。"""
    out = {"listing_id": int(listing_id), "n_reviews": 0,
           "praise": [], "complain": [], "pos_kw": [], "neg_kw": [],
           "avg_sentiment": None}

    # ── ABSA 面向 ──
    try:
        from modules.absa_sections import available, load_listing_absa
        if available():
            d = load_listing_absa()
            d = d[(d["listing_id"] == int(listing_id))
                  & (d["mentions"] >= MIN_MENTIONS)]
            if not d.empty:
                out["n_mentions"] = int(d["mentions"].sum())
                praise = d[d["pos_ratio"] >= POS_CUT].nlargest(TOP_N, "pos_ratio")
                complain = d[d["neg_ratio"] >= NEG_CUT].nlargest(TOP_N, "neg_ratio")
                out["praise"] = [{"aspect": r["aspect"],
                                  "ratio": float(r["pos_ratio"]),
                                  "mentions": int(r["mentions"])}
                                 for _, r in praise.iterrows()]
                out["complain"] = [{"aspect": r["aspect"],
                                    "ratio": float(r["neg_ratio"]),
                                    "mentions": int(r["mentions"])}
                                   for _, r in complain.iterrows()]
    except Exception:
        pass

    # ── 關鍵字與情感(若提供 reviews_df)──
    if reviews_df is not None:
        try:
            from modules.nlp_analysis import listing_review_summary
            s = listing_review_summary(reviews_df, int(listing_id))
            out["n_reviews"] = int(s.get("total_reviews", 0))
            out["avg_sentiment"] = s.get("avg_sentiment")
            out["pos_kw"] = [w for w, _ in (s.get("pos_keywords") or [])[:6]]
            out["neg_kw"] = [w for w, _ in (s.get("neg_keywords") or [])[:6]]
        except Exception:
            pass
    return out


def rule_summary(f: dict) -> str:
    """規則版摘要:由面向統計組成一段中文敘述。"""
    if not f["praise"] and not f["complain"]:
        return "此房源評論則數不足,尚無法歸納出明確的稱讚或抱怨傾向。"

    parts = []
    n = f.get("n_reviews") or 0
    lead = f"綜合 {n:,} 則住客評論," if n else "綜合住客評論,"

    if f["praise"]:
        names = "、".join(p["aspect"] for p in f["praise"])
        best = f["praise"][0]
        parts.append(f"多數旅客稱讚{names}"
                     f"(其中「{best['aspect']}」正評率達 {best['ratio']:.0%})")
    if f["complain"]:
        names = "、".join(c["aspect"] for c in f["complain"])
        worst = f["complain"][0]
        parts.append(f"負面意見主要集中於{names}"
                     f"(「{worst['aspect']}」負評率 {worst['ratio']:.0%},"
                     f"共 {worst['mentions']} 則提及)")
    else:
        parts.append("未出現集中的負面意見")

    tail = ""
    if f.get("pos_kw"):
        tail += f" 高頻正面詞:{'、'.join(f['pos_kw'][:5])}。"
    if f.get("neg_kw"):
        tail += f" 高頻負面詞:{'、'.join(f['neg_kw'][:5])}。"
    return lead + ";".join(parts) + "。" + tail


def build_prompt(f: dict, name: str = "", district: str = "") -> str:
    """LLM 摘要提示詞。"""
    praise = "、".join(f"{p['aspect']}(正評率 {p['ratio']:.0%})"
                       for p in f["praise"]) or "無明顯強項"
    complain = "、".join(f"{c['aspect']}(負評率 {c['ratio']:.0%}"
                         f"、{c['mentions']} 則提及)"
                         for c in f["complain"]) or "無集中抱怨"
    return f"""你是旅宿業評論分析師。請用繁體中文寫出一段 3~5 句的評論摘要,給房東閱讀。

房源:{name}({district})
評論則數:{f.get('n_reviews', 0)}
面向情感分析結果:
- 住客稱讚的面向:{praise}
- 住客抱怨的面向:{complain}
- 高頻正面詞:{'、'.join(f.get('pos_kw', [])) or '無'}
- 高頻負面詞:{'、'.join(f.get('neg_kw', [])) or '無'}

要求:
1. 第一句先總結整體口碑傾向。
2. 接著具體說明旅客最常稱讚什麼、最常抱怨什麼(引用上面的面向與比例)。
3. 最後一句給一個最優先的改善方向。
4. 直接輸出段落,不要標題、不要條列、不要開場白。"""


def llm_summary(f: dict, name: str = "", district: str = "") -> tuple[str, str]:
    """LLM 版摘要。回傳 (來源, 文字);無金鑰或失敗時由呼叫端處理例外。"""
    from modules.llm_advisor import _secret, llm_available
    provider = llm_available()
    if not provider:
        raise RuntimeError("未設定 LLM 金鑰")
    prompt = build_prompt(f, name, district)

    if provider == "claude":
        import requests
        from modules.llm_advisor import ANTHROPIC_MODEL, ANTHROPIC_URL, TIMEOUT_S
        r = requests.post(
            ANTHROPIC_URL, timeout=TIMEOUT_S,
            headers={"x-api-key": _secret("ANTHROPIC_API_KEY"),
                     "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": ANTHROPIC_MODEL, "max_tokens": 600,
                  "messages": [{"role": "user", "content": prompt}]})
        r.raise_for_status()
        return "claude", r.json()["content"][0]["text"]

    from modules.llm_advisor import GEMINI_FALLBACKS, GEMINI_MODEL
    key = _secret("GEMINI_API_KEY")
    models = [GEMINI_MODEL] + [m for m in GEMINI_FALLBACKS if m != GEMINI_MODEL]
    errs = []
    from google import genai
    client = genai.Client(api_key=key)
    for m in models:
        try:
            return f"gemini({m})", client.models.generate_content(
                model=m, contents=prompt).text
        except Exception as e:
            errs.append(f"{m}: {type(e).__name__} {str(e)[:100]}")
    raise RuntimeError("Gemini 各模型皆失敗 → " + " | ".join(errs))
