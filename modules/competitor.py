# -*- coding: utf-8 -*-
"""跨平台競品索引模組(可抽換 PKL:competitor_index.pkl)。

以 BallTree(haversine)索引四平台房源;query() 回傳指定座標 1km 內競品,
stats() 產出價格落點(每人每晚等效價、同容量層)與設施覆蓋率。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

EARTH_R = 6_371_000.0  # m


class CompetitorIndex:
    def __init__(self, market: pd.DataFrame):
        """market: market_data.load_all_market() 之輸出。"""
        self.df = market.reset_index(drop=True)
        rad = np.radians(self.df[["lat", "lon"]].to_numpy())
        self.tree = BallTree(rad, metric="haversine")

    # ---- 查詢 ----
    def query(self, lat: float, lon: float, radius_m: float = 1000.0,
              exclude_listing_id=None) -> pd.DataFrame:
        idx = self.tree.query_radius(
            np.radians([[lat, lon]]), r=radius_m / EARTH_R)[0]
        sub = self.df.iloc[idx].copy()
        d = self._haversine_m(lat, lon, sub["lat"].to_numpy(), sub["lon"].to_numpy())
        sub["dist_m"] = d
        if exclude_listing_id is not None and "listing_id" in sub.columns:
            sub = sub[~(sub["listing_id"] == exclude_listing_id)]
        return sub.sort_values("dist_m").reset_index(drop=True)

    def stats(self, lat: float, lon: float, listing_pp_day: float,
              bracket: str, radius_m: float = 1000.0,
              exclude_listing_id=None) -> dict:
        comp = self.query(lat, lon, radius_m, exclude_listing_id)
        out = {"n_total": len(comp), "platforms": {}, "competitors": comp}

        # 各平台統計(不分層,呈現用)
        for p, g in comp.groupby("platform"):
            out["platforms"][p] = {
                "count": int(len(g)),
                "pp_median": float(g["price_pp_day"].median()),
                "pp_q25": float(g["price_pp_day"].quantile(.25)),
                "pp_q75": float(g["price_pp_day"].quantile(.75)),
            }

        # 落點百分位:只與同容量層競品比(規模經濟偏差防護)
        same = comp[comp["bracket"] == bracket]
        out["n_same_bracket"] = len(same)
        if len(same) >= 5 and np.isfinite(listing_pp_day):
            arr = same["price_pp_day"].to_numpy()
            out["pp_percentile"] = float((arr < listing_pp_day).mean())
            out["bracket_pp_median"] = float(np.median(arr))
        else:
            out["pp_percentile"] = None
            out["bracket_pp_median"] = None

        # 設施覆蓋率(該設施在 1km 競品中的擁有比例)
        cov = {}
        if len(comp) > 0:
            all_keys = set().union(*comp["amenities"].tolist()) if len(comp) else set()
            for k in all_keys:
                cov[k] = float(comp["amenities"].map(lambda s: k in s).mean())
        out["amenity_coverage"] = dict(sorted(cov.items(), key=lambda x: -x[1]))
        return out

    @staticmethod
    def amenity_gap(coverage: dict, own: set, min_cov: float = 0.5) -> list:
        """周邊過半競品有、而本房源沒有的設施(依覆蓋率排序)。"""
        return [(k, v) for k, v in coverage.items() if v >= min_cov and k not in own]

    @staticmethod
    def _haversine_m(lat, lon, lats, lons):
        p1, p2 = np.radians(lat), np.radians(lats)
        dp = np.radians(lats - lat)
        dl = np.radians(lons - lon)
        a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
        return 2 * EARTH_R * np.arcsin(np.sqrt(a))
