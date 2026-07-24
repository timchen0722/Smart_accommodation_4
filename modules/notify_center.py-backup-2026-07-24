# -*- coding: utf-8 -*-
"""共用通知中心模組(房東入口 TB4 與後台分析共用)。

功能(docx §四 60% 通知模組 + 2026-07-19/20 需求):
  • 門檻滑桿(作用於校準後高風險機率,預設 0.60)
  • 📅 空檔警示觸發(2026-07-20 新增):未來 90 天內連續空檔達門檻亦發通知
  • 🤖 自動寄送:高風險房源自動產生「智慧建議」並模擬寄信;失敗列入待補寄
  • ✉️ 手動補寄:逐筆手動寄出(優先 LLM 建議,失敗退回規則引擎)
  • 通知紀錄(含建議內容預覽)與已處理狀態

觸發邏輯:風險門檻與空檔門檻為「或」關係,任一成立即進入通知清單;
清單會標示各筆是由哪個條件觸發(風險 / 空檔 / 兩者)。
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
    """通知母體:模型預測 + 房源資料 + 未來檔期空檔指標(若已產生)。"""
    preds = load_predictions()
    meta = load_listings()[["id", "name", "host_name", "amenities",
                            "description", "minimum_nights"]]
    df = preds.merge(meta, on="id", how="left")
    # ── 併入空檔指標(來自 build_calendar_features.py)──
    try:
        from modules import calendar_analytics as ca
        if ca.available():
            cal = ca.healthy_metrics()[
                ["listing_id", "gap_days_30d", "gap_longest_30d",
                 "gap_days_90d", "gap_longest_90d", "gap_first_start_30d",
                 "booked_rate_d30", "booked_rate_d90"]]
            df = df.merge(cal, left_on="id", right_on="listing_id", how="left")
    except Exception:
        pass
    for c in ["gap_days_30d", "gap_longest_30d",
              "gap_days_90d", "gap_longest_90d", "booked_rate_d90"]:
        if c not in df.columns:
            df[c] = np.nan
    return df


@st.cache_resource(show_spinner=False)
def _engine():
    return _load_pkl("suggestion_engine")


@st.cache_resource(show_spinner=False)
def _comp():
    try:
        return _load_pkl("competitor_index")
    except FileNotFoundError:
        return None


def _gap_advice(r: pd.Series) -> list:
    """空檔專屬建議(依最長連續空檔天數給促銷強度)。"""
    g = r.get("gap_days_30d")
    if pd.isna(g) or g <= 0:
        return []
    longest = int(r.get("gap_longest_30d") or 0)
    price = float(pd.to_numeric(r.get("price"), errors="coerce") or 0)
    out = [f"未來 30 天有 {int(g)} 天空檔待填補"
           + (f",以每晚 ${price:,.0f} 計約損失 ${price * g:,.0f} 營收" if price else "")]
    if longest >= 21:
        out.append(f"最長連續空檔達 {longest} 天:建議大幅折扣(20~30%)"
                   f"或開放月租/長住方案以整段承接")
    elif longest >= 10:
        out.append(f"最長連續空檔 {longest} 天:建議限時折扣 10~15%,"
                   f"並放寬最低入住天數至 1~2 晚")
    else:
        out.append(f"最長連續空檔 {longest} 天:建議設定最後一分鐘折扣"
                   f"(入住前 7 天內 8~9 折)填補零散空檔")
    # 評論痛點(若 ABSA 已產生)
    try:
        from modules.absa_sections import listing_pain_points
        for p in listing_pain_points(int(r["id"]), top_k=2):
            if p["neg_ratio"] >= 0.1:
                out.append(f"住客最常抱怨「{p['aspect']}」"
                           f"(負評率 {p['neg_ratio']:.0%}):{p['tip']}")
    except Exception:
        pass
    return out


def _rule_advice(r: pd.Series) -> list:
    """規則引擎建議(自動寄送與 LLM 失敗時的後備);空檔房源優先給檔期建議。"""
    gap_first = _gap_advice(r)
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
    rule = [f"{s['title']}:{s['detail']}" for s in sugs[:3]]
    out = gap_first + rule
    return out or ["維持定價競爭力與服務品質,並持續觀察同商圈行情。"]


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


def _gap_line(r: pd.Series) -> str:
    """信件與清單用的空檔說明字串(無資料時回傳空字串)。"""
    g = r.get("gap_days_30d")
    if pd.isna(g) or g <= 0:
        return ""
    longest = int(r.get("gap_longest_30d") or 0)
    start = str(r.get("gap_first_start_30d") or "")
    return (f"  ・未來 30 天內空檔 {int(g)} 天"
            f"(最長連續 {longest} 天{'、最近自 ' + start + ' 起' if start else ''})")


def _compose(r: pd.Series, prob: float, th: float, advice: list,
             source: str, reason: str = "風險",
             platform_view: bool = False) -> dict:
    to = fake_host_email(r.get("host_name"), r.get("host_id", 0))
    tips = "\n".join(f"  {i}. {a}" for i, a in enumerate(advice, 1))
    gap = _gap_line(r)
    head = {"風險": "空屋風險偏高", "空檔": "未來檔期出現長空檔",
            "風險+空檔": "空屋風險偏高,且未來檔期出現長空檔"}.get(reason, "需要注意")
    lines = [f"  ・高風險機率 {prob:.0%}(通知門檻 {th:.0%})",
             f"  ・預測未來一年空屋率 {float(r['vac_pred']):.0%}"]
    if gap:
        lines.append(gap)
    if platform_view:
        opening = (f"親愛的房東 {r.get('host_name') or ''} 您好,\n\n"
                   f"我們是 Airbnb 平台營運團隊。平台的空屋風險模型偵測到"
                   f"您的房源「{str(r['name'])[:40]}」{head},"
                   f"為協助您提升出租表現,主動與您聯繫:\n")
        closing = ("如需協助,歡迎回覆本信件由專人跟進。\n"
                   "— Airbnb 平台營運團隊 · 智慧旅宿風險預警系統(模擬信件)")
        subject = f"【Airbnb 平台營運團隊】{head} — 經營輔導與改善建議"
    else:
        opening = (f"親愛的房東 {r.get('host_name') or ''} 您好,\n\n"
                   f"系統偵測到您的房源「{str(r['name'])[:40]}」{head}:\n")
        closing = "— 智慧旅宿空屋率風險預警平台(模擬信件)"
        subject = f"【智慧旅宿平台】{head}警示與改善建議"
    body = (opening + "\n".join(lines) + "\n\n"
            f"為提升出租率,智慧建議如下({source}):\n{tips}\n\n"
            f"詳情請登入房東入口查看 LIME 原因分析、未來檔期空檔明細"
            f"與跨平台競品比較。\n"
            + closing)
    return {"to": to, "subject": subject, "body": body}


def _log(r, prob, th, status, source, mail=None):
    st.session_state["notify_log"].insert(0, {
        "房源": f"#{int(r['id'])}", "收件者": (mail or {}).get("to", "—"),
        "機率": f"{prob:.0%}", "門檻": f"{th:.0%}",
        "觸發原因": r.get("_reason") or "風險",
        "建議來源": source, "狀態": status,
        "時間": pd.Timestamp.now().strftime("%m-%d %H:%M"),
        "_body": (mail or {}).get("body", "")})


def render_notify_center(host_id=None, prob_col="prob", tier_col="tier",
                         key="nc", platform_view=False):
    """渲染通知中心。host_id=None 為全平台(後台);指定則僅該房東。"""
    df = _notify_df()
    sec("風險分級與通知(60% 通知模組)")
    mb("門檻作用於『校準後高風險機率 P(空屋率≥60%)』· OOF 誠實機率,紅色層精確率約 70%")

    for k, v in [("notify_log", []), ("processed", {}), ("auto_sent", set())]:
        if k not in st.session_state:
            st.session_state[k] = v

    has_gap = df["gap_days_90d"].notna().any()
    c_th, c_gap, c_auto = st.columns([1.5, 1.5, 1])
    th = c_th.slider("風險門檻(機率 ≥)", 0.30, 0.90, 0.60, 0.05,
                     key=f"{key}_th")
    if has_gap:
        gap_on = c_gap.checkbox("📅 同時納入空檔警示", value=True,
                                key=f"{key}_gapon",
                                help="未來 30 天內連續空檔達下列天數者,即使風險未達門檻也發通知。"
                                     "採 30 天窗口是因近期日曆已開放,空檔才是真實訊號;"
                                     "遠期多為房東尚未開放日曆")
        gap_th = c_gap.slider("連續空檔 ≥(天)", 7, 30, 14, 1,
                              key=f"{key}_gapth", disabled=not gap_on)
    else:
        gap_on, gap_th = False, 14
        c_gap.caption("📅 空檔警示未啟用:請先執行 "
                      "`python -X utf8 scripts/build_calendar_features.py`")
    auto = c_auto.toggle("🤖 自動寄送通知(模擬)", key=f"{key}_auto",
                         help=f"開啟後,達門檻且未寄過的房源自動產生智慧建議並模擬寄信"
                              f"(單次上限 {AUTO_SEND_LIMIT} 間;失敗可手動補寄)")

    # ── 觸發條件:風險 或 空檔(任一成立)──
    risk_hit = df[prob_col] >= th
    gap_hit = (df["gap_longest_30d"].fillna(0) >= gap_th) if gap_on \
        else pd.Series(False, index=df.index)
    df = df.assign(_risk_hit=risk_hit, _gap_hit=gap_hit)
    df["_reason"] = np.select(
        [risk_hit & gap_hit, risk_hit, gap_hit],
        ["風險+空檔", "風險", "空檔"], default="")
    hits_all = df[risk_hit | gap_hit]
    scope = hits_all if host_id is None else hits_all[hits_all["host_id"] == host_id]
    # 依中央分類優先序排序；顯示名稱統一由 modules.quadrant 提供。
    from modules import quadrant as QD
    scope = QD.annotate(scope, tier_col=tier_col)
    scope = scope.sort_values(["quadrant_priority", prob_col, "gap_longest_30d"],
                              ascending=[True, False, False])

    n1, n2, n3, n4 = st.columns(4)
    n1.metric("全平台觸發" if host_id is None else "本房東觸發",
              f"{len(scope):,} 間")
    n2.metric("觸發原因分布",
              f"風險 {int(scope['_risk_hit'].sum())}",
              f"空檔 {int(scope['_gap_hit'].sum())} · 兩者 "
              f"{int((scope['_risk_hit'] & scope['_gap_hit']).sum())}",
              delta_color="off")
    n3.metric("全平台觸發" if host_id is not None else "占比",
              f"{len(hits_all):,} 間" if host_id is not None
              else f"{len(hits_all)/len(df):.0%}",
              f"占比 {len(hits_all)/len(df):.0%}" if host_id is not None else None,
              delta_color="off")
    n4.metric("已處理", f"{sum(st.session_state['processed'].values())} 間")
    _qcnt = scope["quadrant_label"].value_counts()
    if len(_qcnt):
        note("📊 <b>處理優先序(體質 × 檔期象限)</b>:"
             + " ｜ ".join(f"{k} {v} 間" for k, v in _qcnt.items())
             + "。清單已依此排序,🚨 真警報(體質差且檔期空)排最前。")
    if gap_on:
        _only_gap = int((scope["_gap_hit"] & ~scope["_risk_hit"]).sum())
        if _only_gap:
            note(f"📅 其中 <b>{_only_gap}</b> 間是<b>僅由空檔條件</b>觸發 —— "
                 f"風險分數未達標,但未來 90 天有連續 {gap_th} 天以上無訂單,"
                 f"屬於「模型看不出來、但檔期已經在流血」的房源。")

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
                    mail = _compose(r, prob, th, advice, "規則引擎",
                                    reason=r.get("_reason") or "風險",
                                    platform_view=platform_view)
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
        reason = h.get("_reason") or "風險"
        r_badge = {"風險": ("⚠️ 風險", P["high"]),
                   "空檔": ("📅 空檔", P["primary"]),
                   "風險+空檔": ("⚠️📅 風險+空檔", P["accent"])}.get(
            reason, ("⚠️ 風險", P["high"]))
        gap_txt = ""
        if pd.notna(h.get("gap_longest_30d")) and h.get("gap_longest_30d", 0) > 0:
            gap_txt = (f"·📅 30天空檔 {int(h['gap_days_30d'])} 天"
                       f"(最長 {int(h['gap_longest_30d'])} 天)")
        cc1, cc2, cc3 = st.columns([3, 1, 1])
        with cc1:
            st.markdown(
                f"<div style='background:{P['surface']};border:1px solid {P['border']};"
                f"border-left:4px solid {t_c};border-radius:0 10px 10px 0;"
                f"padding:9px 14px;{'opacity:.55;' if done else ''}'>"
                f"<b>#{hid}</b> {str(h['name'])[:30]}"
                f"<span style='background:{t_c};color:#fff;border-radius:10px;"
                f"padding:1px 9px;font-size:.7rem;font-weight:700;margin-left:8px;'>"
                f"{t_zh} {prob:.0%}</span>"
                f"<span style='background:{r_badge[1]};color:#fff;border-radius:10px;"
                f"padding:1px 9px;font-size:.68rem;font-weight:700;margin-left:5px;'>"
                f"{r_badge[0]}</span>"
                f"<span style='background:{P[QD.QUADRANTS[h['quadrant']]['color']]};"
                f"color:#fff;border-radius:10px;padding:1px 9px;font-size:.68rem;"
                f"font-weight:700;margin-left:5px;'>{h['quadrant_label']}</span>"
                f"<span style='color:{P['muted']};font-size:.74rem;margin-left:8px;'>"
                f"{h['neighbourhood_cleansed']}·空屋率 {h['vac_pred']:.0%}{gap_txt}"
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
                    mail = _compose(h, prob, th, advice, src,
                                    reason=h.get("_reason") or "風險",
                                    platform_view=platform_view)
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
                                      "觸發原因", "建議來源", "狀態", "時間"]],
                   height=230)
        with st.expander("📧 檢視最新一封信件內容(含智慧建議)"):
            st.code(log[0].get("_body") or "(無內容)")
        st.caption("模擬示範:實際部署可串接 SMTP / SendGrid / LINE Notify;"
                   "LLM 建議需設定 ANTHROPIC_API_KEY 或 GEMINI_API_KEY。")
    else:
        st.caption("尚無通知紀錄;開啟自動寄送或按「✉️ 手動寄出」產生。")


# ════════════════════════════════════════════════════════════════
# 公開介面 — 供後台「風險管理雙檢視」沿用組信/寄送(不改上方任何行為)
# ════════════════════════════════════════════════════════════════
def notify_source_df() -> pd.DataFrame:
    """組信母體(預測 + 房源 meta + 空檔指標);快取沿用 _notify_df()。"""
    return _notify_df()


def _advice_and_compose(row, prob: float, th: float, *,
                        platform_view: bool = True,
                        prefer_llm: bool = True) -> tuple[dict, str, str]:
    """選建議(高風險優先 LLM,否則規則引擎;LLM 失敗退回規則)+ 組信。
    純函式:不碰 session_state。回傳 (mail, source, status)。"""
    try:
        from modules.llm_advisor import llm_available
        if prefer_llm and llm_available() and prob >= 0.6:
            src, advice = _llm_advice(row, prob)
        else:
            src, advice = "規則引擎", _rule_advice(row)
        status = "手動寄送成功(模擬)"
    except Exception as e:
        src, advice = "規則引擎(LLM 失敗後備)", _rule_advice(row)
        status = f"手動寄送成功(模擬,LLM 失敗:{type(e).__name__})"
    mail = _compose(row, prob, th, advice, src,
                    reason=row.get("_reason") or "風險",
                    platform_view=platform_view)
    return mail, src, status


def send_for_row(row, th: float = 0.60, *, platform_view: bool = True,
                 prefer_llm: bool = True) -> dict:
    """單筆寄送(模擬):組信 + 寫 notify_log + 標記已寄。回傳 mail。
    需在 Streamlit 執行環境(讀寫 session_state)。"""
    for k, v in [("notify_log", []), ("auto_sent", set()), ("processed", {})]:
        if k not in st.session_state:
            st.session_state[k] = v
    prob = float(pd.to_numeric(row.get("prob"), errors="coerce") or 0)
    mail, src, status = _advice_and_compose(
        row, prob, th, platform_view=platform_view, prefer_llm=prefer_llm)
    _log(row, prob, th, status, src, mail)
    st.session_state["auto_sent"].add(int(row["id"]))
    return mail
