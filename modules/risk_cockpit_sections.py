# -*- coding: utf-8 -*-
"""risk_cockpit_sections.py — 後台「🚨 風險管理」雙檢視渲染層。

房東檢視(排行榜/模糊搜尋)⇄ 房源檢視(獨立 checkbox 派信),麵包屑導覽。
純計算委由 platform_analytics;信件組裝/寄送/紀錄沿用 notify_center 公開介面。
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from modules import platform_analytics as pa
from modules.ui_components import P, html_table, note, sec

ROOM_ZH = {"Entire home/apt": "整棟出租", "Private room": "私人套房",
           "Shared room": "共用套房", "Hotel room": "飯店客房"}
TIER_ZH = {"red": "🔴 高風險", "yellow": "🟡 觀察", "green": "🟢 安全"}
HOST_ALL = "不限"                 # 房源檢視「房東ID」selectbox 的不限哨兵
LEADERBOARD_LIMIT = 100          # 房東檢視排行榜顯示上限
LISTING_LIMIT_DEFAULT = 100      # 房源檢視預設顯示筆數


def _money(v: float) -> str:
    """金額縮寫:億 / 萬(與 platform_sections 一致的顯示規則)。"""
    if abs(v) >= 1e8:
        return f"${v / 1e8:,.2f} 億"
    if abs(v) >= 1e4:
        return f"${v / 1e4:,.1f} 萬"
    return f"${v:,.0f}"


# ── 純邏輯(可 pytest,不依賴 Streamlit runtime)──────────────────
def resolve_host_filter(val, valid_ids) -> int | None:
    """把 rm_host_filter 的值正規化為 int 房東ID 或 None(哨兵/非法/不在母體)。"""
    if val == HOST_ALL or val is None:
        return None
    try:
        hid = int(val)
    except (ValueError, TypeError):
        return None
    return hid if hid in {int(x) for x in valid_ids} else None


def search_hosts(host_df: pd.DataFrame, query: str,
                 limit: int = LEADERBOARD_LIMIT) -> pd.DataFrame:
    """房東檢視:host_id 子字串模糊過濾(query 空=全部),取前 limit 位。
    host_df 需已依風險排序(host_risk_summary 輸出)。"""
    d = host_df
    q = (query or "").strip()
    if q:
        d = d[d["host_id"].astype(str).str.contains(q, regex=False)]
    return d.head(limit)


def filter_listings(df: pd.DataFrame, tiers, prob_lo: float, prob_hi: float,
                    host_filter: int | None) -> pd.DataFrame:
    """房源檢視:套用房東鎖定 + 警報層級 + 風險分數區間,依 prob 降序。"""
    d = df if host_filter is None else df[df["host_id"] == host_filter]
    chosen = tiers or ["red", "yellow", "green"]
    d = d[d["tier"].astype(str).isin(chosen)]
    prob = pd.to_numeric(d["prob"], errors="coerce").fillna(0)
    d = d[(prob >= prob_lo) & (prob <= prob_hi)]
    return d.sort_values("prob", ascending=False)


# ── 導覽狀態 callback(在 rerun 前寫 session_state,合法)──────────
def _clear_selection():
    for k in [k for k in st.session_state if str(k).startswith("rm_sel_")]:
        st.session_state[k] = False


def _go_hosts():
    st.session_state["rm_view"] = "hosts"
    st.session_state["rm_host_filter"] = HOST_ALL
    st.session_state["rm_expanded_id"] = None
    _clear_selection()


def _go_listings(host_id):
    st.session_state["rm_view"] = "listings"
    st.session_state["rm_host_filter"] = int(host_id)
    st.session_state["rm_expanded_id"] = None
    _clear_selection()


def _toggle_expand(lid):
    cur = st.session_state.get("rm_expanded_id")
    st.session_state["rm_expanded_id"] = None if cur == int(lid) else int(lid)


def _breadcrumb(view: str):
    """麵包屑;房源檢視顯示可點回的『房東檢視 › 房源檢視』。"""
    if view != "listings":
        st.markdown(
            f"<span style='color:{P['ink2']};font-weight:700;'>房東檢視</span>",
            unsafe_allow_html=True)
        return
    c = st.columns([1.2, 8])
    with c[0]:
        st.button("房東檢視", key="rm_bc_hosts", type="tertiary",
                  on_click=_go_hosts)
    with c[1]:
        st.markdown(
            f"<span style='color:{P['muted']};'>› </span>"
            f"<span style='color:{P['ink2']};font-weight:700;'>房源檢視</span>",
            unsafe_allow_html=True)


def render():
    """後台「🚨 風險管理」入口:依 rm_view 分流房東/房源檢視。"""
    from modules.platform_sections import guard_scope, commission
    df = guard_scope()
    if df is None:
        return
    cm = commission()

    sec("高風險房源與房東管理")
    view = st.session_state.setdefault("rm_view", "hosts")
    _breadcrumb(view)
    st.divider()
    if view == "listings":
        _render_listings(df, cm)
    else:
        _render_hosts(df, cm)


def _lime_reasons(listing_id: int, top: int = 3) -> list:
    """單一房源 Top-N 風險原因;回傳 [(中文特徵名, 百分點)](自 platform_sections 移入)。"""
    from modules.vacancy_model import contributions, get_row
    row = get_row(int(listing_id))
    if row is None:
        return []
    return [(zh, dpp) for _f, zh, dpp in contributions(row, top=top)]


def _send_single(lid):
    """LIME 面板『產生此房源輔導通知』:單筆組信+模擬寄送(平台視角,高風險優先 LLM)。"""
    from modules.notify_center import notify_source_df, send_for_row
    src = notify_source_df()
    hit = src[src["id"] == int(lid)]
    if len(hit):
        mail = send_for_row(hit.iloc[0], platform_view=True, prefer_llm=True)
        st.toast(f"已模擬寄送至 {mail['to']}")
    else:
        st.toast("查無此房源的通知資料")


def _lime_panel(row: pd.Series):
    """展開於房源列下方:Top-3 LIME 原因 + 單筆發送鈕。"""
    lid = int(row["id"])
    with st.spinner("計算風險歸因 …"):
        reasons = _lime_reasons(lid, top=3)
    if reasons:
        for zh, dpp in reasons:
            color = P["high"] if dpp > 0 else P["low"]
            sign = "推高" if dpp > 0 else "降低"
            st.markdown(
                f"<div style='border-left:4px solid {color};"
                f"background:{P['surface']};border-radius:0 8px 8px 0;"
                f"padding:9px 14px;margin:6px 0;'><b>{zh}</b> — {sign}空屋風險 "
                f"<span style='color:{color};font-weight:700;'>"
                f"{dpp:+.2f} 個百分點</span></div>", unsafe_allow_html=True)
    else:
        st.caption("此房源無足夠特徵可解釋。")
    st.button("✉️ 產生此房源輔導通知", key=f"rm_send1_{lid}",
              on_click=_send_single, args=(lid,))


def _listing_rows(shown: pd.DataFrame):
    """房源列:每列 = [checkbox][可點ID][其他欄位];點ID展開 LIME 面板。"""
    expanded = st.session_state.get("rm_expanded_id")
    widths = [0.5, 1.3, 1.0, 0.9, 1.0, 0.9, 1.0, 1.0]
    hdr = st.columns(widths)
    for col, t in zip(hdr, ["選取", "房源ID", "行政區", "房型", "每晚房價",
                            "風險分數", "警報層級", "房東ID"]):
        col.markdown(f"<span style='color:{P['muted']};font-size:.7rem;"
                     f"font-weight:700;'>{t}</span>", unsafe_allow_html=True)
    for _, r in shown.iterrows():
        lid = int(r["id"])
        c = st.columns(widths)
        c[0].checkbox("選取", key=f"rm_sel_{lid}", label_visibility="collapsed")
        c[1].button(f"#{lid} ▸", key=f"rm_lst_{lid}", type="tertiary",
                    on_click=_toggle_expand, args=(lid,))
        c[2].markdown(str(r["neighbourhood_cleansed"]))
        c[3].markdown(ROOM_ZH.get(r["room_type"], str(r["room_type"])))
        c[4].markdown(f"${pd.to_numeric(r['price'], errors='coerce'):,.0f}")
        c[5].markdown(f"{pd.to_numeric(r['prob'], errors='coerce'):.0%}")
        c[6].markdown(TIER_ZH.get(str(r["tier"]), str(r["tier"])))
        c[7].markdown(f"#{int(r['host_id'])}")
        if expanded == lid:
            _lime_panel(r)


_BATCH_BAR_CSS = f"""
<style>
.st-key-rm-batch-bar {{
  position: fixed; left: 0; right: 0; bottom: 0; z-index: 999;
  background: {P['surface']}; border-top: 2px solid {P['primary']};
  box-shadow: 0 -4px 18px rgba(0,0,0,.10); padding: 10px 26px;
}}
/* 浮動列出現時,墊高頁面底部避免遮住通知紀錄 */
[data-testid="stAppViewBlockContainer"] {{ padding-bottom: 96px; }}
</style>
"""


def _select_ids(ids):
    for i in ids:
        st.session_state[f"rm_sel_{int(i)}"] = True


def _selected_ids() -> list:
    out = []
    for k, v in st.session_state.items():
        if str(k).startswith("rm_sel_") and v:
            try:
                out.append(int(str(k)[len("rm_sel_"):]))
            except ValueError:
                pass
    return out


def _send_batch():
    """批次:對所有勾選房源逐筆組信+模擬寄送(規則引擎,快且穩),清空勾選。"""
    from modules.notify_center import notify_source_df, send_for_row
    ids = _selected_ids()
    src = notify_source_df()
    ok = 0
    for lid in ids:
        hit = src[src["id"] == lid]
        if len(hit):
            send_for_row(hit.iloc[0], platform_view=True, prefer_llm=False)
            ok += 1
    _clear_selection()
    st.toast(f"批次模擬寄送完成:{ok} 筆")


def _batch_bar():
    """底部浮動列:僅勾選數 ≥1 時渲染(房東檢視不呼叫本函式)。"""
    sel = _selected_ids()
    if not sel:
        return
    st.markdown(_BATCH_BAR_CSS, unsafe_allow_html=True)
    with st.container(key="rm-batch-bar"):
        c = st.columns([2.4, 1, 1])
        c[0].markdown(f"**已選 {len(sel)} 間**　將對這些房源產生平台輔導通知")
        c[1].button(f"✉️ 批次發送 {len(sel)} 筆", key="rm_batch_send",
                    type="primary", on_click=_send_batch)
        c[2].button("清除選取", key="rm_batch_clear", on_click=_clear_selection)


def _render_listings(df: pd.DataFrame, cm: float):
    """房源檢視:篩選 + 房源列(可勾選/可展開)+ 底部批次浮動列 + 通知紀錄。"""
    valid_ids = df["host_id"].astype(int).unique().tolist()
    f1, f2, f3, f4 = st.columns([1, 1, 1.4, 1.2])
    tiers = f1.multiselect("警報層級", ["red", "yellow", "green"],
                           default=["red"], format_func=lambda t: TIER_ZH[t],
                           key="rm_tiers")
    topn = f2.slider("顯示筆數", 20, 300, LISTING_LIMIT_DEFAULT, 20, key="rm_topn")
    lo, hi = f3.slider("風險分數區間", 0.0, 1.0, (0.0, 1.0), 0.05, key="rm_prob")
    opts = [HOST_ALL] + sorted(valid_ids)
    if st.session_state.get("rm_host_filter", HOST_ALL) not in opts:
        st.session_state["rm_host_filter"] = HOST_ALL     # 掉出母體→重置(合法,widget 前)
    f4.selectbox("房東ID(可打字搜尋)", opts, key="rm_host_filter",
                 format_func=lambda x: x if x == HOST_ALL else f"#{int(x)}")
    host_filter = resolve_host_filter(
        st.session_state.get("rm_host_filter", HOST_ALL), valid_ids)

    fdf = filter_listings(df, tiers, lo, hi, host_filter)
    shown = fdf.head(topn)
    shown_ids = shown["id"].astype(int).tolist()

    _scope = f"🎯 已鎖定房東 #{host_filter};" if host_filter is not None else ""
    st.caption(f"{_scope}符合 {len(fdf):,} 間,顯示風險分數最高的 {len(shown):,} 間")
    if not shown_ids:
        st.info("目前條件下沒有房源;請放寬篩選或改選房東。")
        return

    if host_filter is not None:
        st.button(f"☑ 全選目前篩選結果({len(shown_ids)} 間)",
                  key="rm_select_all", type="tertiary",
                  on_click=_select_ids, args=(shown_ids,))

    note("點<b>房源ID</b>(藍色連結)展開 LIME 風險原因與單筆派信;"
         "勾選左側方框後,底部會滑出批次派信列。")
    _listing_rows(shown)
    _batch_bar()
    st.divider()
    _notify_log_section()


def _notify_log_section():
    """通知紀錄(單筆/批次共用 st.session_state['notify_log'])。"""
    sec("通知紀錄")
    log = st.session_state.get("notify_log", [])
    if log:
        html_table(pd.DataFrame(log)[["房源", "收件者", "機率", "門檻",
                                      "觸發原因", "建議來源", "狀態", "時間"]],
                   height=230)
    else:
        st.caption("尚無通知紀錄;點房源展開後按「✉️ 產生此房源輔導通知」,"
                   "或勾選房源後批次發送。")


def _render_hosts(df: pd.DataFrame, cm: float):
    """房東檢視:模糊查詢 + 可點房東ID排行榜(無勾選、無浮動列)。"""
    h = pa.host_risk_summary(df, cm)
    q = st.text_input("🔍 房東ID 模糊查詢", key="rm_host_search",
                      placeholder="輸入房東ID片段,如 123;留空看全部")
    res = search_hosts(h, q)
    _capped = "(僅顯示前 %d 位)" % LEADERBOARD_LIMIT \
        if len(h) > LEADERBOARD_LIMIT and len(res) >= LEADERBOARD_LIMIT else ""
    st.caption(f"搜尋結果:{len(res):,} 位房東 · "
               f"依「高風險間數 → 高風險占比」排序{_capped}")
    note("點<b>房東ID</b>(藍色連結)即可下鑽該房東名下房源清單並派信。")
    if not len(res):
        st.info("查無符合的房東,請調整查詢條件。")
        return

    widths = [0.6, 1.4, 0.9, 1.2, 1.0, 1.1, 1.3]
    hdr = st.columns(widths)
    for col, t in zip(hdr, ["排名", "房東ID", "房源數", "🔴高風險間數",
                            "高風險占比", "平均風險分數", "預估年營收"]):
        col.markdown(f"<span style='color:{P['muted']};font-size:.72rem;"
                     f"font-weight:700;'>{t}</span>", unsafe_allow_html=True)
    for rank, (_, r) in enumerate(res.iterrows(), 1):
        hid = int(r["host_id"])
        c = st.columns(widths)
        c[0].markdown(f"**{rank}**")
        c[1].button(f"#{hid} ▸", key=f"rm_host_{hid}", type="tertiary",
                    on_click=_go_listings, args=(hid,))
        c[2].markdown(f"{int(r['房源數'])}")
        c[3].markdown(f"{int(r['高風險間數'])}")
        c[4].markdown(f"{float(r['高風險占比']):.0%}")
        c[5].markdown(f"{float(r['平均風險分數']):.0%}")
        c[6].markdown(_money(float(r['預估年營收'])))
