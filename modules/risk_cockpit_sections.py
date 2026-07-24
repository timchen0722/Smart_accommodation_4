# -*- coding: utf-8 -*-
"""risk_cockpit_sections.py — 後台「🚨 風險管理」雙檢視渲染層。

房東檢視(排行榜/模糊搜尋)⇄ 房源檢視(獨立 checkbox 派信),麵包屑導覽。
純計算委由 platform_analytics;信件組裝/寄送/紀錄沿用 notify_center 公開介面。
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from modules import design_tokens as T
from modules import platform_analytics as pa
from modules import ui_kit
from modules.ui_components import ROOM_JP, note

# 房型中譯與風險等級文案都改吃全站唯一來源(原本本檔各自複製一份)
ROOM_ZH = ROOM_JP
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
    """點房東ID:自動跳到『房源管理』頁簽並鎖定該房東。"""
    st.session_state["rm_view"] = "listings"
    st.session_state["rm_host_filter"] = int(host_id)
    st.session_state["rm_expanded_id"] = None
    _clear_selection()


def _go_listings_tab():
    """直接點『房源管理』頁簽:切到房源檢視,保留目前房東篩選(預設不限=全部)。"""
    st.session_state["rm_view"] = "listings"
    st.session_state["rm_expanded_id"] = None


def _toggle_expand(lid):
    cur = st.session_state.get("rm_expanded_id")
    st.session_state["rm_expanded_id"] = None if cur == int(lid) else int(lid)


def _tabs(view: str):
    """狀態驅動的兩頁簽列(房東管理 / 房源管理)。

    原生 st.tabs 無法由程式自動切換,故以兩顆按鈕當頁簽(綁 rm_view),
    才能同時支援『可自由切頁簽』與『點房東自動跳到房源管理』。
    """
    c = st.columns([1.1, 1.1, 4])
    c[0].button("🧑‍💼 房東管理", key="rm_tab_hosts", width="stretch",
                type="primary" if view == "hosts" else "secondary",
                on_click=_go_hosts)
    c[1].button("📋 房源管理", key="rm_tab_listings", width="stretch",
                type="primary" if view == "listings" else "secondary",
                on_click=_go_listings_tab)


def render():
    """後台「🚨 風險管理」入口:兩頁簽(房東管理 / 房源管理)依 rm_view 分流。"""
    from modules.platform_sections import guard_scope, commission
    df = guard_scope()
    if df is None:
        return
    cm = commission()

    ui_kit.section_header("高風險房源與房東管理",
                          desc="先在「房東管理」找到整批惡化的房東，再下鑽到「房源管理」逐間處理")
    view = st.session_state.setdefault("rm_view", "hosts")
    _tabs(view)
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
            # 推高風險=danger、降低風險=success,與全站「紅=要處理」語意一致
            role = "danger" if dpp > 0 else "success"
            sign = "推高" if dpp > 0 else "降低"
            st.markdown(
                f"<div style='border-left:4px solid var(--sa-{role});"
                f"background:var(--sa-surface);"
                f"border-radius:0 var(--sa-radius-sm) var(--sa-radius-sm) 0;"
                f"padding:9px 14px;margin:6px 0;'><b>{zh}</b> — {sign}空屋風險 "
                f"<span style='color:var(--sa-{role});font-weight:700;'>"
                f"{dpp:+.2f} 個百分點</span></div>", unsafe_allow_html=True)
    else:
        st.caption("此房源無足夠特徵可解釋。")
    ui_kit.primary_button("✉️ 產生此房源輔導通知", key=f"rm_send1_{lid}",
                          on_click=_send_single, args=(lid,))


def _listing_rows(shown: pd.DataFrame):
    """房源列:每列 = [checkbox][可點ID][其他欄位];點ID展開 LIME 面板。"""
    expanded = st.session_state.get("rm_expanded_id")
    widths = [0.5, 1.3, 1.0, 0.9, 1.0, 0.9, 1.0, 1.0]
    ui_kit.table_header_row(
        ["選取", "房源ID", "行政區", "房型", "每晚房價",
         "風險分數", "警報層級", "房東ID"], widths)
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
        # 等級改用 RiskBadge:與統計卡、圖表、詳細頁同名同色
        c[6].markdown(ui_kit.risk_badge(r["tier"]), unsafe_allow_html=True)
        c[7].markdown(f"#{int(r['host_id'])}")
        if expanded == lid:
            _lime_panel(r)


_BATCH_BAR_CSS = """
<style>
/* 批次派信列:置於房源表格上方的一般區塊
   (原為 position:fixed 底部浮動列,左側會被側邊欄遮住,故改置頂) */
.st-key-rm-batch-bar {
  background: var(--sa-surface); border: 1px solid var(--sa-border);
  border-left: 4px solid var(--sa-primary); border-radius: var(--sa-radius-md);
  padding: var(--sa-space-2) var(--sa-space-4); margin: 4px 0 10px;
}
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


def _batch_bar(shown_ids):
    """頂部批次派信列:全選 / 批次發送 / 清除選取(房東檢視不呼叫本函式)。

    置於房源表格「上方」,常駐顯示;未勾選時發送與清除鈕為 disabled。
    原本是底部 position:fixed 浮動列,左半部會被側邊欄蓋住,故改置頂。
    """
    sel = _selected_ids()
    st.markdown(_BATCH_BAR_CSS, unsafe_allow_html=True)
    with st.container(key="rm-batch-bar"):
        c = st.columns([2.3, 0.9, 1.25, 1])
        # D3(2026-07-24):筆數只在一個地方說 —— 「已選 N」在這行、
        # 「符合/顯示」在表格上方的 caption;按鈕標籤只寫動作,不再複述數字。
        c[0].markdown(f"**已選 {len(sel)} 間**　將對這些房源產生平台輔導通知"
                      if sel else "產生平台輔導通知　（先勾選左側方框）")
        with c[1]:
            ui_kit.secondary_button("☑ 全選", key="rm_select_all",
                                    on_click=_select_ids, args=(shown_ids,))
        with c[2]:
            ui_kit.primary_button("✉️ 批次發送", key="rm_batch_send",
                                  disabled=not sel, on_click=_send_batch)
        with c[3]:
            ui_kit.secondary_button("清除選取", key="rm_batch_clear",
                                    disabled=not sel,
                                    on_click=_clear_selection)


def render_sidebar_filters(df: pd.DataFrame):
    """風險管理側欄篩選(警報層級/顯示筆數/風險分數區間/房東ID)。

    由後台頁的側欄在「風險管理」分頁時呼叫;寫入 rm_tiers/rm_topn/
    rm_prob/rm_host_filter,供 _render_listings 直接讀 session_state 使用。
    """
    st.multiselect("警報層級", list(T.TIER_ORDER), default=["red"],
                   format_func=T.tier_label, key="rm_tiers")
    st.slider("顯示筆數", 20, 300, LISTING_LIMIT_DEFAULT, 20, key="rm_topn")
    st.slider("風險分數區間", 0.0, 1.0, (0.0, 1.0), 0.05, key="rm_prob")
    valid_ids = df["host_id"].astype(int).unique().tolist() if len(df) else []
    opts = [HOST_ALL] + sorted(valid_ids)
    if st.session_state.get("rm_host_filter", HOST_ALL) not in opts:
        st.session_state["rm_host_filter"] = HOST_ALL     # 掉出母體→重置(合法,widget 前)
    st.selectbox("房東ID(可打字搜尋)", opts, key="rm_host_filter",
                 format_func=lambda x: x if x == HOST_ALL else f"#{int(x)}")


def _render_listings(df: pd.DataFrame, cm: float):
    """房源檢視:讀側欄篩選 + 頂部批次派信列 + 房源列(可勾選/可展開)+ 通知紀錄。

    篩選 widget 已移至後台頁側欄(render_sidebar_filters),本函式僅讀取
    session_state 的 rm_tiers/rm_topn/rm_prob/rm_host_filter。
    """
    valid_ids = df["host_id"].astype(int).unique().tolist()
    tiers = st.session_state.get("rm_tiers", ["red"])
    topn = int(st.session_state.get("rm_topn", LISTING_LIMIT_DEFAULT))
    lo, hi = st.session_state.get("rm_prob", (0.0, 1.0))
    host_filter = resolve_host_filter(
        st.session_state.get("rm_host_filter", HOST_ALL), valid_ids)

    fdf = filter_listings(df, tiers, lo, hi, host_filter)
    shown = fdf.head(topn)
    shown_ids = shown["id"].astype(int).tolist()

    _scope = f"🎯 已鎖定房東 #{host_filter};" if host_filter is not None else ""
    st.caption(f"{_scope}符合 {len(fdf):,} 間,顯示風險分數最高的 {len(shown):,} 間")
    if not shown_ids:
        ui_kit.empty_state("目前條件下沒有房源",
                           hint="請放寬側欄的警報層級／風險分數區間，或改選房東。")
        return

    _batch_bar(shown_ids)
    # D4:派信流程由上方批次列自己說明,這裡只講「表格本身怎麼用」。
    note("點<b>房源ID</b>(藍色連結)可展開該房源的 LIME 風險原因。")
    _listing_rows(shown)
    st.divider()
    _notify_log_section()


def _notify_log_section():
    """通知紀錄(單筆/批次共用 st.session_state['notify_log'])。"""
    ui_kit.section_header("通知紀錄")
    log = st.session_state.get("notify_log", [])
    if log:
        ui_kit.data_table(pd.DataFrame(log)[["房源", "收件者", "機率", "門檻",
                                             "觸發原因", "建議來源", "狀態",
                                             "時間"]],
                          height=230)
    else:
        ui_kit.empty_state(
            "尚無通知紀錄",
            hint="點房源 ID 展開後按「✉️ 產生此房源輔導通知」，或勾選房源後批次發送。",
            icon="✉️")


# ── 房東排行榜:可點欄位標題排序(遞增/遞減切換)──────────────
HOST_SORT_COLS = ["房源數", "高風險間數", "高風險占比",
                  "平均風險分數", "預估年營收"]


def _set_host_sort(col):
    """點欄位標題:同欄再點切換升/降序,換欄則預設遞減(大→小)。"""
    if st.session_state.get("rm_host_sort_col") == col:
        st.session_state["rm_host_sort_asc"] = \
            not st.session_state.get("rm_host_sort_asc", False)
    else:
        st.session_state["rm_host_sort_col"] = col
        st.session_state["rm_host_sort_asc"] = False   # 首次點=遞減


def _host_sort_arrow(col):
    """回傳該欄目前排序箭頭:▼遞減 / ▲遞增 / ⇅未排序(可點)。"""
    if st.session_state.get("rm_host_sort_col") == col:
        return "▲" if st.session_state.get("rm_host_sort_asc") else "▼"
    return "⇅"


def _render_hosts(df: pd.DataFrame, cm: float):
    """房東檢視:模糊查詢 + 可點房東ID排行榜(無勾選、無浮動列)。"""
    h = pa.host_risk_summary(df, cm)
    q = st.text_input("🔍 房東ID 模糊查詢", key="rm_host_search",
                      placeholder="輸入房東ID片段,如 123;留空看全部")
    res = search_hosts(h, q)
    # 使用者若點過欄位標題排序,套用其排序;否則維持風險預設排序
    sort_col = st.session_state.get("rm_host_sort_col")
    sort_asc = bool(st.session_state.get("rm_host_sort_asc", False))
    if sort_col in res.columns:
        res = res.sort_values(sort_col, ascending=sort_asc, kind="mergesort")

    _capped = "(僅顯示前 %d 位)" % LEADERBOARD_LIMIT \
        if len(h) > LEADERBOARD_LIMIT and len(res) >= LEADERBOARD_LIMIT else ""
    _order = f"依「{sort_col}」{'遞增' if sort_asc else '遞減'}排序" if sort_col \
        else "依「高風險間數 → 高風險占比」排序"
    st.caption(f"搜尋結果:{len(res):,} 位房東 · {_order}{_capped}")
    note("點<b>房東ID</b>(藍色連結)即可下鑽該房東名下房源清單並派信;"
         "點欄位標題的箭頭可依該欄數字遞增/遞減排序。")
    if not len(res):
        ui_kit.empty_state("查無符合的房東", hint="請清空或縮短房東 ID 查詢字串。",
                           icon="🔍")
        return

    # 末欄為留白 spacer,讓資料欄靠左集中(欄距不再過寬)。
    # 表頭第一欄是純文字、其餘是可點排序按鈕,故不能整列交給 table_header_row,
    # 只有第一欄沿用它的樣式 class。
    widths = [1.4, 1.0, 1.5, 1.2, 1.5, 1.5, 2.0]
    hdr = st.columns(widths, gap="small")
    hdr[0].markdown('<span class="sa-table-head-cell">房東ID</span>',
                    unsafe_allow_html=True)
    for i, col in enumerate(HOST_SORT_COLS, start=1):
        label = ("🔴" if col == "高風險間數" else "") + col
        hdr[i].button(f"{label} {_host_sort_arrow(col)}",
                      key=f"rm_sort_{col}", type="tertiary",
                      on_click=_set_host_sort, args=(col,))
    for _, r in res.iterrows():
        hid = int(r["host_id"])
        c = st.columns(widths, gap="small")
        c[0].button(f"#{hid} ▸", key=f"rm_host_{hid}", type="tertiary",
                    on_click=_go_listings, args=(hid,))
        c[1].markdown(f"{int(r['房源數'])}")
        c[2].markdown(f"{int(r['高風險間數'])}")
        c[3].markdown(f"{float(r['高風險占比']):.0%}")
        c[4].markdown(f"{float(r['平均風險分數']):.0%}")
        c[5].markdown(_money(float(r['預估年營收'])))
