# -*- coding: utf-8 -*-
"""calendar_sections.py — 未來檔期分頁的 UI 區塊(房東入口用)

獨立模組:pages/1_🏠_房東入口.py 只需兩行接入,不影響既有四視圖與通知中心。
"""
from __future__ import annotations

import html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modules import calendar_analytics as ca
from modules import listing_detail as LD
<<<<<<< HEAD
from modules import design_tokens as T
from modules.ui_components import (P, apply_theme, html_table, mb, note,
                                   numbered_section_title, sec)
=======
from modules.ui_components import P, apply_theme, html_table, mb, note, sec
>>>>>>> a7faf33a8d684411d0b07457665e38d0f6d4d906

DOW_ZH = ["一", "二", "三", "四", "五", "六", "日"]

# ── 日曆配色語意 ────────────────────────────────────────────────
# 對房東而言「已訂」是收入、「空房」才是損失,所以已訂用站內主色藍
# (P["primary"] 的淡階)、空房退成背景米色 —— 藍越滿代表生意越好。
# 刻意不用紅色:P["high"] 在全站代表「高風險、要處理」,留給下方的
# 空檔警示,避免同一個紅在不同分頁有兩種意思。
<<<<<<< HEAD
# 這個藍是在「日期文字(ink)對比 ≥ 4.5」的前提下,與空房格辨識度最高的
# 主色淡階(實測文字對比 6.06、兩格互比 2.08)。Plotly 吃不到 CSS 變數,
# 故取 token 的 Python 值,不是 var(--sa-*)。
CAL_BOOKED = T.CAL_BOOKED_BLUE       # 已訂 / 不可訂
=======
# #8AACCD 是在「日期文字(P['ink'])對比 ≥ 4.5」的前提下,與空房格
# 辨識度最高的主色淡階(實測文字對比 6.06、兩格互比 2.08)。
CAL_BOOKED = "#8AACCD"     # 已訂 / 不可訂
>>>>>>> a7faf33a8d684411d0b07457665e38d0f6d4d906
CAL_OPEN = P["tag_bg"]     # 空房可訂

# 表格數字欄:等寬數字 + 右對齊,讓上下列的位數對得起來
_NUM_CSS = "font-variant-numeric:tabular-nums;text-align:right;"


def _cal_legend() -> str:
    """日曆圖例(KB Heat Map 檢查表要求 legend 必須可見)。"""
    def _sw(c, label, border=""):
        return (f'<span style="display:inline-flex;align-items:center;'
                f'gap:6px;margin-right:16px;">'
<<<<<<< HEAD
                f'<span style="width:13px;height:13px;border-radius:var(--sa-radius-bar);'
                f'background:{c};{border}"></span>'
                f'<span style="font-size:var(--sa-text-caption);color:{P["ink2"]};">{label}</span>'
=======
                f'<span style="width:13px;height:13px;border-radius:3px;'
                f'background:{c};{border}"></span>'
                f'<span style="font-size:.78rem;color:{P["ink2"]};">{label}</span>'
>>>>>>> a7faf33a8d684411d0b07457665e38d0f6d4d906
                f'</span>')
    return ('<div style="margin:2px 0 8px;">'
            + _sw(CAL_BOOKED, "已訂／不可訂")
            + _sw(CAL_OPEN, "空房可訂",
                  border=f"border:1px solid {P['border']};")
            + '</div>')


def _gap_badge(n: int) -> str:
    """空檔長度的等級 badge —— 與站內風險等級共用同一套色塊語言。"""
    c, t = ((P["high"], "長空檔") if n >= 21 else
            ((P["medium"], "中空檔") if n >= 10 else (P["low"], "短空檔")))
<<<<<<< HEAD
    return (f'<span style="background:{c};color:#fff;border-radius:var(--sa-radius-sm);'
            f'padding:2px 10px;font-size:var(--sa-text-label);font-weight:700;'
=======
    return (f'<span style="background:{c};color:#fff;border-radius:10px;'
            f'padding:2px 10px;font-size:.72rem;font-weight:700;'
>>>>>>> a7faf33a8d684411d0b07457665e38d0f6d4d906
            f'white-space:nowrap;">{t}</span>')


def _month_span(s: pd.DataFrame) -> tuple[int, int]:
    """該月資料實際佔用的列範圍(0-based,含頭含尾)。

    查詢範圍常從月中開始(例如 6/30 起算),此時該月前幾列完全沒有資料,
    直接畫會在月曆上方留下大片空白列 —— 取實際範圍把它們裁掉。
    """
    fd = s["date"].min().replace(day=1).dayofweek
    rows = (s["date"].dt.day - 1 + fd) // 7
    return int(rows.min()), int(rows.max())


def _month_calendar(s: pd.DataFrame):
    """單一月份的月曆(7 欄=週一~週日,列=當月第幾週,格子=該日訂房狀態)。

    改成真月曆而非「第 N 週」序列,日期才對得上使用者心中的月曆。
    zmin/zmax 必須釘死 0/1 —— 整月全訂滿或全空時 z 只有單一值,
    Plotly 會自動縮放色階,導致整月被塗成相反的顏色。
    """
    fd = s["date"].min().replace(day=1).dayofweek
    r0, r1 = _month_span(s)
    nrow = r1 - r0 + 1
    z = [[None] * 7 for _ in range(nrow)]
    tx = [[""] * 7 for _ in range(nrow)]
    cd = [[""] * 7 for _ in range(nrow)]
    for _, r in s.iterrows():
        i = int((r["date"].day - 1 + fd) // 7) - r0
        j = int(r["date"].dayofweek)
        b = int(r["booked"])
        z[i][j] = b
        tx[i][j] = str(int(r["date"].day))
        cd[i][j] = (r["date"].strftime("%m/%d") + "　"
                    + ("已訂／不可訂" if b == 1 else "空房可訂"))
    fig = go.Figure(go.Heatmap(
        z=z, x=DOW_ZH, y=[f"w{i}" for i in range(nrow)],
        text=tx, texttemplate="%{text}",
        textfont=dict(size=10, color=P["ink"]),
        customdata=cd, hoverongaps=False,
        colorscale=[[0, CAL_OPEN], [0.5, CAL_OPEN],
                    [0.5, CAL_BOOKED], [1, CAL_BOOKED]],
        zmin=0, zmax=1, showscale=False, xgap=3, ygap=3,
        hovertemplate="<b>%{customdata}</b><extra></extra>"))
    apply_theme(fig, h=34 * nrow + 44).update_layout(
        yaxis=dict(autorange="reversed", showticklabels=False),
        xaxis=dict(side="top", tickfont=dict(size=10)),
        margin=dict(l=4, r=4, t=24, b=4))
    return fig


def render_calendar_tab(listing_id: int, listing_row, listings_df):
    """房東入口「📅 未來檔期」分頁主體。

    參數:listing_id 房源 id、listing_row 該房源基本資料(Series)、
          listings_df 全量 listings(供同儕營收曲線)。
    """
    if not ca.available():
        st.warning("尚未產生檔期資料,請先執行:")
        st.code("python -X utf8 scripts/build_calendar_features.py")
        return

    row = ca.get_listing(listing_id)
    if row is None:
        st.info("此房源不在 calendar 資料範圍內(calendar 與 listings 為不同批次爬取,"
                "重疊約 4,940 間)。請改選其他房源。")
        return
    if row["is_all_blocked"]:
        note("⚠️ 此房源未來 365 天<b>全部不可訂</b> —— 可能已停業、轉長租或房東封鎖日曆,"
             "以下檔期分析不具參考價值。")
    if row["is_all_open"]:
        note("⚠️ 此房源未來 365 天<b>完全沒有訂單</b>,請優先檢視右側風險診斷與定價建議。")

    district = listing_row["neighbourhood_cleansed"]
    room_type = listing_row["room_type"]

    # ── 總覽:左側房源照片、右側未來訂房率(仿「房源定價情報」作法) ──
<<<<<<< HEAD
    numbered_section_title(1, "未來檔期總攬")
=======
    sec("未來檔期總覽")
>>>>>>> a7faf33a8d684411d0b07457665e38d0f6d4d906
    c_photo, c_kpi = st.columns([1.02, .98], gap="medium")
    with c_photo:
        _bd = listings_df[listings_df["id"] == int(listing_id)]
        _bd = _bd.iloc[0] if len(_bd) else None
        _purl = str((_bd.get("picture_url", "") if _bd is not None else "") or "")
        _pimg = (
            f'<img class="pricing-photo" src="{html.escape(_purl, quote=True)}" '
            f'alt="房源封面照片" loading="lazy" referrerpolicy="no-referrer">'
            if _purl.startswith("http") else
            '<div class="pricing-photo pricing-photo-empty">暫無房源照片</div>')
        st.markdown(
            f'<div class="pricing-pane pricing-left">{_pimg}'
            f'{LD.summary_html(_bd, show_name=False) if _bd is not None else ""}'
            f'</div>', unsafe_allow_html=True)
    with c_kpi:
        # 主指標放大、其餘 2×2 —— 五個數字等重時看不出該先看哪個,
        # 這頁真正要決策的是「未來 90 天訂得掉多少」。
        _v90 = row.get("booked_rate_d90")
        if pd.isna(_v90):
            _v90c, _v90t, _v90s = P["muted"], "—", "此房源無 90 天資料"
        else:
            _v90c = (P["low"] if _v90 >= .6 else
                     (P["medium"] if _v90 >= .35 else P["high"]))
            _v90t = f"{_v90:.0%}"
            _v90s = f"約 {round(float(_v90) * 90)} / 90 天已訂"
        st.markdown(
            f'<div style="background:{P["surface"]};border:1px solid {P["border"]};'
<<<<<<< HEAD
            f'border-radius:var(--sa-radius-md);padding:14px 18px;margin-bottom:10px;'
            f'font-variant-numeric:tabular-nums;">'
            f'<div style="font-size:var(--sa-text-caption);color:{P["muted"]};'
            f'letter-spacing:.06em;">未來 90 天已訂率</div>'
            f'<div style="font-size:var(--sa-text-metric-hero);'
            f'line-height:1.15;font-weight:800;'
            f'color:{_v90c};">{_v90t}</div>'
            f'<div style="font-size:var(--sa-text-caption);color:{P["ink2"]};">{_v90s}</div>'
=======
            f'border-radius:14px;padding:14px 18px;margin-bottom:10px;'
            f'font-variant-numeric:tabular-nums;">'
            f'<div style="font-size:.78rem;color:{P["muted"]};'
            f'letter-spacing:.06em;">未來 90 天已訂率</div>'
            f'<div style="font-size:2.6rem;line-height:1.15;font-weight:800;'
            f'color:{_v90c};">{_v90t}</div>'
            f'<div style="font-size:.78rem;color:{P["ink2"]};">{_v90s}</div>'
>>>>>>> a7faf33a8d684411d0b07457665e38d0f6d4d906
            f'</div>', unsafe_allow_html=True)
        k1 = st.columns(2)
        for i, (tag, lbl) in enumerate([("d30", "未來 30 天"),
                                        ("d60", "未來 60 天")]):
            v = row.get(f"booked_rate_{tag}")
            k1[i].metric(f"{lbl}已訂率", "—" if pd.isna(v) else f"{v:.0%}")
        k2 = st.columns(2)
        k2[0].metric("未來 365 天已訂率", f"{row['booked_rate']:.0%}",
                     f"{int(row['booked_days'])} 天", delta_color="off")
        k2[1].metric("90 天內空檔", f"{int(row['gap_days_90d'])} 天",
                     f"{int(row['gap_count_90d'])} 段·最長 {int(row['gap_longest_90d'])} 天",
                     delta_color="off")

    st.divider()

    # ── 細節:逐日訂房狀態(分月月曆) ──
<<<<<<< HEAD
    numbered_section_title(2, "逐日訂房狀態")
=======
    sec("逐日訂房狀態")
>>>>>>> a7faf33a8d684411d0b07457665e38d0f6d4d906
    st.markdown(_cal_legend(), unsafe_allow_html=True)
    d = ca.daily_frame(row)
    rng = st.radio("顯示範圍", ["未來 30 天", "未來 60 天", "未來 90 天"],
                   horizontal=True, index=2, key="cal_rng")
    n = {"未來 30 天": 30, "未來 60 天": 60, "未來 90 天": 90}[rng]
    dd = d[d["horizon"] < n].copy()
    dd["ym"] = dd["date"].dt.to_period("M")
    months = list(dd["ym"].unique())
    # 欄寬依各月天數加權:查詢範圍常從月中起算(cal_start 是 6/30),
    # 首月可能只有一兩天,平均分欄會讓它白佔一大格。下限 12 是為了
    # 讓殘月仍塞得下 7 欄格子,不至於擠成一條。
    # 各月依自己的列數決定高度、頂端對齊 —— 與實體月曆的並排方式一致。
    _w = [max(len(dd[dd["ym"] == m]), 12) for m in months]
    for _col, _m in zip(st.columns(_w, gap="small"), months):
        with _col:
            _s = dd[dd["ym"] == _m]
            _bk, _tt = int(_s["booked"].sum()), len(_s)
            st.markdown(
<<<<<<< HEAD
                f'<div style="font-size:var(--sa-text-body);font-weight:700;color:{P["ink"]};'
                f'margin-bottom:2px;">{_m.month} 月'
                f'<span style="font-weight:400;font-size:var(--sa-text-caption);'
=======
                f'<div style="font-size:.88rem;font-weight:700;color:{P["ink"]};'
                f'margin-bottom:2px;">{_m.month} 月'
                f'<span style="font-weight:400;font-size:.75rem;'
>>>>>>> a7faf33a8d684411d0b07457665e38d0f6d4d906
                f'color:{P["muted"]};margin-left:8px;'
                f'font-variant-numeric:tabular-nums;">'
                f'已訂 {_bk}/{_tt} 天 · {_bk / _tt:.0%}</span></div>',
                unsafe_allow_html=True)
            st.plotly_chart(_month_calendar(_s), use_container_width=True,
                            key=f"cal_{listing_id}_{_m}")

    st.divider()

    # ── 問題 → 行動:空檔警示與營收定價刻意不再用 divider 隔開,
    #    讓「哪裡有洞」緊接著「怎麼補」。 ──
<<<<<<< HEAD
    numbered_section_title(
        3, "空檔警示", "未來 90 天內連續 5 天以上無訂單")
=======
    sec("空檔警示(未來 90 天內連續 5 天以上無訂單)")
>>>>>>> a7faf33a8d684411d0b07457665e38d0f6d4d906
    gaps = ca.gap_segments(row, min_len=5, horizon=90)
    if len(gaps):
        g = gaps.copy()
        g["起日"] = g["起日"].dt.strftime("%Y-%m-%d")
        g["迄日"] = g["迄日"].dt.strftime("%Y-%m-%d")
        g["等級"] = g["連續天數"].map(_gap_badge)
        g["建議"] = g["連續天數"].map(
            lambda n: ("大幅折扣或開放長租" if n >= 21
                       else ("限時折扣 10~15%" if n >= 10
                             else "設最後一分鐘折扣")))
        html_table(g[["起日", "迄日", "連續天數", "等級", "建議"]], height=240,
                   cell_fn={"連續天數": lambda v: _NUM_CSS})
        note(f"共 <b>{int(row['gap_days_90d'])}</b> 天空檔待填補;"
             f"若以每晚 ${float(listing_row['price']):,.0f} 計,"
             f"填滿可增加約 <b>${float(listing_row['price']) * row['gap_days_90d']:,.0f}</b> 營收。")
    else:
        st.success("未來 90 天內無連續 5 天以上的空檔 🎉")

    # ── 營收最適定價 ──
    sec("💰 營收最適定價(以同商圈同房型的真實已訂天數估算)")
    mb("營收估算 = 每晚價格 × 真實已訂天數 · 已訂天數取自 calendar,與價格為獨立資料源")
    curve = ca.peer_revenue_curve(listings_df, district, room_type)
    if curve.empty:
        st.caption("同商圈同房型樣本不足,無法建立營收曲線。")
    else:
        opt = ca.optimal_price(curve)
        cur_price = float(pd.to_numeric(
            str(listing_row["price"]).replace("$", "").replace(",", ""),
            errors="coerce") or 0)
        cc1, cc2 = st.columns([1.4, 1])
        with cc1:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=curve["價格中位"], y=curve["年營收估算"],
                                 name="年營收估算", marker_color=P["primary"],
                                 opacity=.75))
            fig.add_trace(go.Scatter(x=curve["價格中位"],
                                     y=curve["已訂率"] * curve["年營收估算"].max(),
                                     name="已訂率(右軸比例)", mode="lines+markers",
                                     line=dict(color=P["medium"], width=2,
                                               dash="dot")))
            fig.add_vline(x=cur_price, line_dash="dot", line_color=P["high"],
                          annotation_text=f"目前 ${cur_price:,.0f}")
            if opt:
                fig.add_vline(x=opt["price"], line_dash="dash",
                              line_color=P["low"],
                              annotation_text=f"最適 ${opt['price']:,.0f}")
            apply_theme(fig, h=300).update_layout(
                xaxis_title="每晚價格 (NT$)", yaxis_title="年營收估算 (NT$)",
                legend=dict(orientation="h", y=-0.3))
            st.plotly_chart(fig, use_container_width=True)
        with cc2:
            if opt:
                _gap = opt["price"] - cur_price
                _dir = "調高" if _gap > 0 else "調降"
                st.metric("建議價格帶", f"${opt['price']:,.0f}",
                          f"{_dir} ${abs(_gap):,.0f}" if abs(_gap) > 50 else "已接近最適",
                          delta_color="off")
                st.metric("該價格帶年營收估算", f"${opt['revenue']:,.0f}",
                          f"平均已訂 {opt['booked_days']:.0f} 天 · {opt['n']} 筆同儕",
                          delta_color="off")
            html_table(curve.assign(
                價格中位=curve["價格中位"].map("${:,.0f}".format),
                已訂天數=curve["已訂天數"].round(0),
                已訂率=curve["已訂率"].map("{:.0%}".format),
                年營收估算=curve["年營收估算"].map("${:,.0f}".format))[
                ["價格中位", "已訂天數", "已訂率", "年營收估算", "樣本數"]],
                height=230,
                cell_fn={c: (lambda v: _NUM_CSS) for c in
                         ["價格中位", "已訂天數", "已訂率", "年營收估算", "樣本數"]})
        note("⚠️ <b>誠實限制</b>:此為<b>橫斷面</b>推論(同商圈不同房源在不同價位的實際表現),"
             "非同一間房調價的因果效應;真正的價格彈性需 A/B 調價實驗。"
             "另 Inside Airbnb 的『不可訂』同時包含已預訂與房東封鎖,已排除全年封鎖/全空房源。")
