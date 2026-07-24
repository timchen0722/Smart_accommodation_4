# -*- coding: utf-8 -*-
"""report_builder.py — 房東月報自動生成

彙整既有分析結果為一份可下載的 Markdown / HTML 月報:
  1. 本月摘要(風險等級、檔期進度、空檔損失估算)
  2. 未來檔期明細與空檔清單
  3. 同商圈比較與營收最適定價
  4. 住客評論面向(ABSA)強弱項
  5. AI 摘要(有 LLM 金鑰時生成,否則使用規則摘要)

不修改任何既有模組,純讀取。
"""
from __future__ import annotations

import html as _html
from datetime import datetime

import numpy as np
import pandas as pd

from modules import calendar_analytics as ca
from modules import design_tokens as T


def _fmt_money(v) -> str:
    try:
        return f"NT$ {float(v):,.0f}"
    except (TypeError, ValueError):
        return "—"


def collect(listing_row, pred_row, listings_df,
            prob_col: str = "prob", tier_col: str = "tier") -> dict:
    """蒐集月報所需的全部素材(純資料,不含排版)。"""
    lid = int(pred_row["id"])
    d = {
        "listing_id": lid,
        "name": str(pred_row.get("name") or f"房源 #{lid}"),
        "district": pred_row.get("neighbourhood_cleansed", ""),
        "room_type": pred_row.get("room_type", ""),
        "price": float(pd.to_numeric(pred_row.get("price"), errors="coerce") or 0),
        "vac_pred": float(pred_row.get("vac_pred") or 0),
        "prob": float(pred_row.get(prob_col) or 0),
        "tier": pred_row.get(tier_col, "green"),
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    # ── 檔期 ──
    cal = ca.get_listing(lid) if ca.available() else None
    if cal is not None:
        d["cal"] = {
            "booked_rate": float(cal["booked_rate"]),
            "booked_days": int(cal["booked_days"]),
            "d30": float(cal.get("booked_rate_d30") or np.nan),
            "d90": float(cal.get("booked_rate_d90") or np.nan),
            "gap_days_30d": int(cal.get("gap_days_30d") or 0),
            "gap_longest_30d": int(cal.get("gap_longest_30d") or 0),
            "gap_days_90d": int(cal.get("gap_days_90d") or 0),
            "min_nights_median": float(cal.get("min_nights_median") or 0),
            "is_all_blocked": int(cal.get("is_all_blocked") or 0),
            "is_all_open": int(cal.get("is_all_open") or 0),
            "is_longterm_only": int(cal.get("is_longterm_only") or 0),
        }
        # ── 體質(模型)× 檔期(真實)四象限 ──
        from modules.quadrant import QUADRANTS, classify_row
        q = classify_row(d["tier"], d["cal"]["d90"])
        d["quadrant"] = q
        d["quadrant_info"] = QUADRANTS[q]
        # 營運狀態辨識(房東暫停 / 轉長租 / 正常營運)
        if d["cal"]["is_all_blocked"]:
            d["op_status"] = ("疑似暫停營業",
                              "未來 365 天全部設為不可訂,可能已停業、轉自住或長期封鎖日曆")
        elif d["cal"]["is_longterm_only"]:
            d["op_status"] = ("已轉為長租經營",
                              f"最低入住天數中位 {d['cal']['min_nights_median']:.0f} 晚,"
                              f"已脫離短租市場,短租空屋率指標不適用")
        elif d["cal"]["is_all_open"]:
            d["op_status"] = ("完全無訂單",
                              "未來 365 天全部可訂,尚未取得任何預訂")
        else:
            d["op_status"] = ("正常營運", "")
        d["gaps"] = ca.gap_segments(cal, min_len=5, horizon=90)
        d["monthly"] = ca.monthly_vs_market(cal, d["district"], d["room_type"])
        curve = ca.peer_revenue_curve(listings_df, d["district"], d["room_type"])
        d["opt"] = ca.optimal_price(curve)
    # ── 評論面向 ──
    try:
        from modules.absa_sections import listing_pain_points
        d["pains"] = listing_pain_points(lid, top_k=3)
    except Exception:
        d["pains"] = []
    return d


def _rule_summary(d: dict) -> str:
    """無 LLM 金鑰時的規則式摘要(以真實檔期調和模型判斷)。"""
    # 純文字摘要不放 emoji;認不出來的等級沿用原本的空字串行為
    tier_zh = T.tier_label(d["tier"], emoji=False, default="")
    c = d.get("cal")
    q = d.get("quadrant_info")

    # 營運狀態異常時,先講狀態,模型指標降為次要
    op = d.get("op_status")
    if op and op[0] != "正常營運":
        return (f"⚠️ 此房源目前狀態為 **{op[0]}** —— {op[1]}。"
                f"模型的風險等級({tier_zh}、機率 {d['prob']:.0%})"
                f"是以短租經營為前提推估,在此狀態下僅供參考。")

    parts = []
    if q:
        parts.append(f"綜合評估:**{q['label']}** —— {q['desc']}。"
                     f"模型體質評估為{tier_zh}(機率 {d['prob']:.0%}),"
                     f"而未來 90 天真實已訂率為 {c['d90']:.0%}")
    else:
        parts.append(f"本房源目前風險等級為 **{tier_zh}**"
                     f"(高風險機率 {d['prob']:.0%}、預測空屋率 {d['vac_pred']:.0%})")
    if c:
        parts.append(f"未來 365 天已訂 {c['booked_days']} 天"
                     f"(訂房率 {c['booked_rate']:.0%});"
                     f"未來 30 天內有 {c['gap_days_30d']} 天空檔,"
                     f"最長連續 {c['gap_longest_30d']} 天")
        if d["price"] and c["gap_days_30d"]:
            parts.append(f"以現價估算,近 30 天空檔的機會成本約 "
                         f"{_fmt_money(d['price'] * c['gap_days_30d'])}")
    if d.get("opt") and d["price"]:
        gap = d["opt"]["price"] - d["price"]
        if abs(gap) > 100:
            parts.append(f"同商圈同房型的營收最適價格帶約 "
                         f"{_fmt_money(d['opt']['price'])},"
                         f"建議{'調高' if gap > 0 else '調降'} "
                         f"{_fmt_money(abs(gap))}")
    if d.get("pains"):
        p = d["pains"][0]
        parts.append(f"住客評論中最需改善的面向是「{p['aspect']}」"
                     f"(負評率 {p['neg_ratio']:.0%}),建議{p['tip']}")
    out = "。".join(parts) + "。"
    if q:
        out += f" 建議行動:{q['action']}。"
    return out


def ai_summary(d: dict) -> tuple[str, str]:
    """AI 摘要:有金鑰走 LLM,否則規則摘要。回傳 (來源, 文字)。"""
    try:
        from modules.llm_advisor import generate_advice, llm_available
        if not llm_available():
            return "規則摘要", _rule_summary(d)
        c = d.get("cal", {})
        q = d.get("quadrant_info")
        op = d.get("op_status")
        # 把「真實檔期」與「營運狀態」一併餵給 LLM,避免只依模型下判斷
        tier_ctx = d["tier"]
        if q:
            _v = c.get("d90", np.nan)
            _s = "—" if (_v is None or np.isnan(_v)) else f"{_v:.0%}"
            tier_ctx = (f"{d['tier']}(綜合分類:{q['label']} —— {q['desc']};"
                        f"未來90天真實已訂率 {_s})")
        if op and op[0] != "正常營運":
            tier_ctx += f";營運狀態:{op[0]} — {op[1]}"
        provider, md = generate_advice({
            "name": d["name"], "district": d["district"],
            "room_type": d["room_type"], "price": d["price"],
            "vac_pred": d["vac_pred"], "prob": d["prob"],
            "tier": tier_ctx,
            "lime_reasons": [{"zh": f"{p['aspect']}負評率 {p['neg_ratio']:.0%}",
                              "weight_pp": p["neg_ratio"] * 100,
                              "direction": "up"} for p in d.get("pains", [])],
            "comp_summary": (f"未來 30 天空檔 {c.get('gap_days_30d', 0)} 天、"
                             f"最長連續 {c.get('gap_longest_30d', 0)} 天;"
                             f"營收最適價 {_fmt_money((d.get('opt') or {}).get('price'))}"),
            "amenity_gaps": [p["aspect"] for p in d.get("pains", [])]})
        return provider, md
    except Exception:
        return "規則摘要(LLM 失敗後備)", _rule_summary(d)


def to_markdown(d: dict, summary_src: str, summary: str) -> str:
    """組出 Markdown 月報。"""
    tier_zh = T.tier_label(d["tier"], default="")
    c = d.get("cal")
    q = d.get("quadrant_info")
    op = d.get("op_status")

    L = [f"# 房源經營月報 — {d['name']}", "",
         f"> 房源 #{d['listing_id']} ｜ {d['district']} ｜ {d['room_type']} ｜ "
         f"每晚 {_fmt_money(d['price'])}",
         f"> 產生時間:{d['generated']}", "",
         "## 一、綜合評估(體質 × 檔期)", ""]

    if op and op[0] != "正常營運":
        L += [f"> ⚠️ **營運狀態:{op[0]}** —— {op[1]}",
              "> 下列模型指標以短租經營為前提推估,在此狀態下僅供參考。", ""]

    if q:
        L += [f"### {q['label']}", "",
              f"{q['desc']}", "",
              f"**建議行動**:{q['action']}", ""]

    L += ["| 面向 | 指標 | 數值 | 性質 |", "|---|---|---|---|",
          f"| 體質(模型推估) | 風險等級 | {tier_zh} | 依 2025-09 特徵推估 |",
          f"| 體質(模型推估) | 高風險機率 P(空屋率≥60%) | {d['prob']:.0%} | "
          f"GroupKFold OOF,前瞻 AUC 0.632 |",
          f"| 體質(模型推估) | 預測未來一年空屋率 | {d['vac_pred']:.0%} | "
          f"前瞻 R²≈0,僅供參考 |"]
    if c:
        s90 = "—" if np.isnan(c["d90"]) else f"{c['d90']:.0%}"
        s30 = "—" if np.isnan(c["d30"]) else f"{c['d30']:.0%}"
        L += [f"| **檔期(真實觀測)** | **未來 90 天已訂率** | **{s90}** | "
              f"Inside Airbnb calendar,100% 觀測值 |",
              f"| **檔期(真實觀測)** | **未來 30 天已訂率** | **{s30}** | "
              f"近期日曆已開放,最可信 |"]
    L += ["",
          "> **判讀原則**:近期行動看檔期(真實觀測),長期投資看模型(體質推估)。"
          "兩者衝突時以檔期為準 —— 模型特徵取自 2025-09 快照,"
          "與檔期資料相隔約 9 個月,期間房東可能已調價或改善房源。", ""]
    if c:
        d30 = "—" if np.isnan(c["d30"]) else f"{c['d30']:.0%}"
        d90 = "—" if np.isnan(c["d90"]) else f"{c['d90']:.0%}"
        L += ["## 二、未來檔期進度", "",
              "| 指標 | 數值 |", "|---|---|",
              f"| 未來 365 天已訂率 | {c['booked_rate']:.0%}"
              f"({c['booked_days']} 天) |",
              f"| 未來 30 天已訂率 | {d30} |",
              f"| 未來 90 天已訂率 | {d90} |",
              f"| 未來 30 天空檔 | {c['gap_days_30d']} 天"
              f"(最長連續 {c['gap_longest_30d']} 天) |",
              f"| 未來 90 天空檔 | {c['gap_days_90d']} 天 |", ""]
        if d["price"] and c["gap_days_30d"]:
            L += [f"**近 30 天空檔機會成本估算:"
                  f"{_fmt_money(d['price'] * c['gap_days_30d'])}**", ""]
        g = d.get("gaps")
        if g is not None and len(g):
            L += ["### 空檔明細(未來 90 天,連續 5 天以上)", "",
                  "| 起日 | 迄日 | 連續天數 |", "|---|---|---|"]
            for _, r in g.head(10).iterrows():
                L.append(f"| {r['起日'].strftime('%Y-%m-%d')} | "
                         f"{r['迄日'].strftime('%Y-%m-%d')} | {int(r['連續天數'])} |")
            L.append("")
        m = d.get("monthly")
        if m is not None and len(m):
            L += ["### 月度訂房率 vs 同商圈", "",
                  "| 月份 | 本房源 | 同商圈基準 | 差距 |", "|---|---|---|---|"]
            for _, r in m.iterrows():
                base = "—" if pd.isna(r["同商圈基準"]) else f"{r['同商圈基準']:.0%}"
                diff = "—" if pd.isna(r["差距"]) else f"{r['差距']*100:+.0f} pp"
                L.append(f"| {r['月份']} | {r['本房源']:.0%} | {base} | {diff} |")
            L.append("")

    if d.get("opt"):
        o = d["opt"]
        L += ["## 三、營收最適定價", "",
              f"- 目前每晚定價:{_fmt_money(d['price'])}",
              f"- 同商圈同房型營收最適價格帶:**{_fmt_money(o['price'])}**",
              f"- 該價格帶平均已訂 {o['booked_days']:.0f} 天,"
              f"年營收估算 {_fmt_money(o['revenue'])}(同儕樣本 {o['n']} 筆)",
              "",
              "> 註:此為橫斷面推論(同商圈不同房源在不同價位的實際表現),"
              "非同一房源調價的因果效應。", ""]

    if d.get("pains"):
        L += ["## 四、住客評論面向(ABSA)", "",
              "| 面向 | 負評率 | 提及則數 | 建議 |", "|---|---|---|---|"]
        for p in d["pains"]:
            L.append(f"| {p['aspect']} | {p['neg_ratio']:.0%} | "
                     f"{p['mentions']} | {p['tip']} |")
        L.append("")

    L += [f"## 五、AI 經營建議({summary_src})", "", summary, "",
          "---", "",
          "*本報告由智慧旅宿空屋率風險預警平台自動生成。"
          "檔期資料來源:Inside Airbnb calendar;"
          "風險預測為 GroupKFold OOF 誠實評估,"
          "長期外推能力有限(見後台『前瞻驗證』)。*"]
    return "\n".join(L)


def to_html(md_text: str, title: str) -> str:
    """把 Markdown 月報包成可離線開啟/列印的 HTML。"""
    body = _html.escape(md_text)
    return f"""<!DOCTYPE html><html lang="zh-Hant"><head><meta charset="utf-8">
<title>{_html.escape(title)}</title>
<style>
body{{font-family:"Noto Sans TC","Microsoft JhengHei",sans-serif;
 background:#F8F7F5;color:#2A2A2A;max-width:860px;margin:32px auto;padding:0 20px;
 line-height:1.75;}}
pre{{white-space:pre-wrap;word-break:break-word;background:#fff;border:1px solid #E8E4DE;
 border-radius:12px;padding:22px 26px;font-family:inherit;font-size:.92rem;}}
@media print{{body{{background:#fff;}} pre{{border:none;}}}}
</style></head><body><pre>{body}</pre></body></html>"""
