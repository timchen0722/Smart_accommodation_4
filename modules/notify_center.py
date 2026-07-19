# -*- coding: utf-8 -*-
"""共用通知中心模組(房東入口 TB4 與後台分析共用)。

功能(docx §四 60% 通知模組 + 2026-07-19 需求):
  • 門檻滑桿(作用於校準後高風險機率,預設 0.60)
  • 🤖 自動寄送:高風險房源自動產生「智慧建議」並模擬寄信;失敗列入待補寄
  • ✉️ 手動補寄:逐筆手動寄出(優先 LLM 建議,失敗退回規則引擎)
  • 通知紀錄(含建議內容預覽)與已處理狀態
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from modules.ui_components import P, sec, mb, note, html_table
from modules.data_loader import load_listings
from modules.feature_engineering import load_predictions
from modules.image_analysis import fake_host_email
from modules.market_data import capacity_bracket, _canon_amenities
from modules.pkl_store import load_module as _load_pkl

TIER_ZH = {"red": ("🔴 高風險", "high"), "yellow": ("🟡 觀察", "medium"),
           "green": ("🟢 安全", "low")}
AUTO_SEND_LIMIT = 30      # 單次自動寄送上限(避免一次跑太久)


@st.cache_data(show_spinner="載入通知資料 …")
def _notify_df() -> pd.DataFrame:
    preds = load_predictions()
    meta = load_listings()[["id", "name", "host_name", "amenities",
                            "description", "minimum_nights"]]
    return preds.merge(meta, on="id", how="left")


@st.cache_resource(show_spinner=False)
def _engine():
    return _load_pkl("suggestion_engine")


@st.cache_resource(show_spinner=False)
def _comp():
    try:
        return _load_pkl("competitor_index")
    except FileNotFoundError:
        return None


def _rule_advice(r: pd.Series) -> list:
    """規則引擎建議(自動寄送與 LLM 失敗時的後備)。"""
    comp = _comp()
    cs = {"amenity_coverage": {}, "n_total": 0}
    if comp is not None:
        cap = max(float(pd.to_numeric(r.get("accommodates"),
                                      errors="coerce") or 2), 1)
        cs = comp.stats(float(r["latitude"]), float(r["longitude"]),
                        listing_pp_day=float(r["price"]) / cap,
                        bracket=capacity_bracket(cap), radius_m=1000,
                        exclude_listing_id=int(r["id"]))
    mn = pd.to_numeric(r.get("minimum_nights"), errors="coerce")
    sugs = _engine().suggest(
        shap_items=[], comp_stats=cs,
        features={"accommodates": float(pd.to_numeric(
            r.get("accommodates"), errors="coerce") or 2),
            "minimum_nights": 0.0 if pd.isna(mn) else float(mn),
            "desc_len": float(len(str(r.get("description") or "")))},
        own_amenities=_canon_amenities(str(r.get("amenities", ""))))
    return [f"{s['title']}:{s['detail']}" for s in sugs[:4]] or \
        ["維持定價競爭力與服務品質,並持續觀察同商圈行情。"]


def _llm_advice(r: pd.Series, prob: float) -> tuple[str, list]:
    """優先 LLM 生成建議;失敗丟例外由呼叫端退回規則引擎。"""
    from modules.llm_advisor import generate_advice
    provider, md = generate_advice({
        "name": str(r["name"]), "district": r["neighbourhood_cleansed"],
        "room_type": r["room_type"], "price": float(r["price"]),
        "vac_pred": float(r["vac_pred"]), "prob": prob,
        "tier": "高風險", "lime_reasons": [],
        "comp_summary": "見平台附近比較頁", "amenity_gaps": []})
    lines = [ln.strip(" -*") for ln in md.splitlines() if ln.strip(" -*")]
    return provider, (lines or [md])


def _compose(r: pd.Series, prob: float, th: float, advice: list,
             source: str) -> dict:
    to = fake_host_email(r.get("host_name"), r.get("host_id", 0))
    tips = "\n".join(f"  {i}. {a}" for i, a in enumerate(advice, 1))
    body = (f"親愛的房東 {r.get('host_name') or ''} 您好,\n\n"
            f"系統偵測到您的房源「{str(r['name'])[:40]}」空屋風險偏高:\n"
            f"  ・高風險機率 {prob:.0%}(通知門檻 {th:.0%})\n"
            f"  ・預測未來一年空屋率 {float(r['vac_pred']):.0%}\n\n"
            f"為提升出租率,智慧建議如下({source}):\n{tips}\n\n"
            f"詳情請登入房東入口查看 LIME 原因分析與跨平台競品比較。\n"
            f"— 智慧旅宿空屋率風險預警平台(模擬信件)")
    return {"to": to, "subject": "【智慧旅宿平台】高空屋風險警示與改善建議",
            "body": body}


def _log(r, prob, th, status, source, mail=None):
    st.session_state["notify_log"].insert(0, {
        "房源": f"#{int(r['id'])}", "收件者": (mail or {}).get("to", "—"),
        "機率": f"{prob:.0%}", "門檻": f"{th:.0%}",
        "建議來源": source, "狀態": status,
        "時間": pd.Timestamp.now().strftime("%m-%d %H:%M"),
        "_body": (mail or {}).get("body", "")})


def render_notify_center(host_id=None, prob_col="prob", tier_col="tier",
                         key="nc"):
    """渲染通知中心。host_id=None 為全平台(後台);指定則僅該房東。"""
    df = _notify_df()
    sec("風險分級與通知(60% 通知模組)")
    mb("門檻作用於『校準後高風險機率 P(空屋率≥60%)』· OOF 誠實機率,紅色層精確率約 70%")

    for k, v in [("notify_log", []), ("processed", {}), ("auto_sent", set())]:
        if k not in st.session_state:
            st.session_state[k] = v

    c_th, c_auto = st.columns([2, 1])
    th = c_th.slider("通知門檻(機率 ≥)", 0.30, 0.90, 0.60, 0.05,
                     key=f"{key}_th")
    auto = c_auto.toggle("🤖 自動寄送通知(模擬)", key=f"{key}_auto",
                         help=f"開啟後,達門檻且未寄過的房源自動產生智慧建議並模擬寄信"
                              f"(單次上限 {AUTO_SEND_LIMIT} 間;失敗可手動補寄)")

    hits_all = df[df[prob_col] >= th]
    scope = hits_all if host_id is None else hits_all[hits_all["host_id"] == host_id]
    scope = scope.sort_values(prob_col, ascending=False)

    n1, n2, n3 = st.columns(3)
    n1.metric("全平台觸發" if host_id is None else "本房東觸發",
              f"{len(scope):,} 間")
    n2.metric("全平台觸發" if host_id is not None else "占比",
              f"{len(hits_all):,} 間" if host_id is not None
              else f"{len(hits_all)/len(df):.0%}",
              f"占比 {len(hits_all)/len(df):.0%}" if host_id is not None else None,
              delta_color="off")
    n3.metric("已處理", f"{sum(st.session_state['processed'].values())} 間")

    # ── 自動寄送 ──
    if auto:
        pending = [(_, r) for _, r in scope.iterrows()
                   if int(r["id"]) not in st.session_state["auto_sent"]]
        ok = fail = 0
        with st.spinner(f"自動寄送中(共 {min(len(pending), AUTO_SEND_LIMIT)} 間)…"):
            for _, r in pending[:AUTO_SEND_LIMIT]:
                prob = float(r[prob_col])
                try:
                    advice = _rule_advice(r)     # 批次自動寄送用規則引擎(快且穩)
                    mail = _compose(r, prob, th, advice, "規則引擎")
                    _log(r, prob, th, "自動寄送成功(模擬)", "規則引擎", mail)
                    ok += 1
                except Exception as e:           # 建議產生失敗 → 留待手動補寄
                    _log(r, prob, th, f"自動寄送失敗:{type(e).__name__}", "—")
                    fail += 1
                st.session_state["auto_sent"].add(int(r["id"]))
        if ok or fail:
            st.toast(f"自動寄送完成:成功 {ok} 件、失敗 {fail} 件(模擬)")
        if fail:
            st.warning(f"{fail} 件自動寄送失敗,請在下方清單以「✉️ 手動寄出」補寄。")

    # ── 清單(手動寄出 / 已處理) ──
    show = scope.head(40)
    if not len(show):
        st.success("目前無達到門檻之房源 🎉")
    for _, h in show.iterrows():
        hid = int(h["id"])
        prob = float(h[prob_col])
        done = st.session_state["processed"].get(hid, False)
        sent = hid in st.session_state["auto_sent"]
        t_zh, t_key = TIER_ZH.get(h[tier_col], TIER_ZH["yellow"])
        t_c = P[t_key]
        cc1, cc2, cc3 = st.columns([3, 1, 1])
        with cc1:
            st.markdown(
                f"<div style='background:{P['surface']};border:1px solid {P['border']};"
                f"border-left:4px solid {t_c};border-radius:0 10px 10px 0;"
                f"padding:9px 14px;{'opacity:.55;' if done else ''}'>"
                f"<b>#{hid}</b> {str(h['name'])[:32]}"
                f"<span style='background:{t_c};color:#fff;border-radius:10px;"
                f"padding:1px 9px;font-size:.7rem;font-weight:700;margin-left:8px;'>"
                f"{t_zh} {prob:.0%}</span>"
                f"<span style='color:{P['muted']};font-size:.74rem;margin-left:8px;'>"
                f"{h['neighbourhood_cleansed']}·空屋率 {h['vac_pred']:.0%}"
                f"·{h.get('host_name') or ''}"
                f"{'·📨 已寄' if sent else ''}{'·✅ 已處理' if done else ''}"
                f"</span></div>", unsafe_allow_html=True)
        with cc2:
            if st.button("✉️ 手動寄出", key=f"{key}m{hid}", disabled=done):
                with st.spinner("產生智慧建議並寄送(模擬)…"):
                    try:
                        from modules.llm_advisor import llm_available
                        # LLM 僅於紅色高風險(機率 ≥ 0.6)觸發,其餘用規則引擎
                        if llm_available() and prob >= 0.6:
                            src, advice = _llm_advice(h, prob)
                        else:
                            src, advice = "規則引擎", _rule_advice(h)
                        status = "手動寄送成功(模擬)"
                    except Exception as e:
                        src, advice = "規則引擎(LLM 失敗後備)", _rule_advice(h)
                        status = f"手動寄送成功(模擬,LLM 失敗:{type(e).__name__})"
                    mail = _compose(h, prob, th, advice, src)
                    _log(h, prob, th, status, src, mail)
                    st.session_state["auto_sent"].add(hid)
                st.toast(f"已模擬寄送至 {mail['to']}")
                st.rerun()
        with cc3:
            if st.button("✅ 已處理" if not done else "↩︎ 取消",
                         key=f"{key}d{hid}"):
                st.session_state["processed"][hid] = not done
                st.rerun()

    # ── 通知紀錄 ──
    sec("通知紀錄")
    log = st.session_state["notify_log"]
    if log:
        html_table(pd.DataFrame(log)[["房源", "收件者", "機率", "門檻",
                                      "建議來源", "狀態", "時間"]], height=230)
        with st.expander("📧 檢視最新一封信件內容(含智慧建議)"):
            st.code(log[0].get("_body") or "(無內容)")
        st.caption("模擬示範:實際部署可串接 SMTP / SendGrid / LINE Notify;"
                   "LLM 建議需設定 ANTHROPIC_API_KEY 或 GEMINI_API_KEY。")
    else:
        st.caption("尚無通知紀錄;開啟自動寄送或按「✉️ 手動寄出」產生。")
