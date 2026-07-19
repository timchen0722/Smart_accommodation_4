# -*- coding: utf-8 -*-
"""智慧建議引擎模組(可抽換 PKL:suggestion_engine.pkl)。

綜合 SHAP 扣分項 + 跨平台價格落點 + 設施缺口,產出白話可執行建議。
"""
from __future__ import annotations

DEFAULT_CONFIG = {
    "price_pctl_high": 0.60,     # 跨平台同容量層落點高於此 → 建議降價
    "min_nights_long": 3,        # 最低入住 ≥ 此值 → 建議放寬
    "desc_len_short": 200,       # 描述字數低於此 → 建議充實文案
    "amenity_min_cov": 0.5,      # 設施覆蓋率門檻
    "amenity_top_k": 3,
    "shap_top_k": 2,
}

# SHAP 特徵 → 白話建議模板
SHAP_TEMPLATES = {
    "price_pctl_nbhd": ("定價偏高", "您的定價高於同區同房型多數競爭者,建議參考同區行情調降價格。"),
    "price": ("定價偏高", "價格是目前推高空屋風險的主因之一,建議調降至更具競爭力的區間。"),
    "minimum_nights": ("最低入住天數過長", "過嚴的入住天數限制會流失短住旅客,建議在法規允許內放寬至 1~2 晚。"),
    "min_nights_avg_ntm": ("入住天數限制偏嚴", "近期最短入住設定偏長,建議放寬以承接短住需求。"),
    "response_speed": ("回覆速度偏慢", "建議開啟即時預訂或設定自動回覆,將回覆速度提升至 1 小時內。"),
    "host_acceptance_rate": ("接受率偏低", "訂單接受率偏低會降低曝光,建議提高接單率或開啟即時預訂。"),
    "host_response_rate": ("回覆率偏低", "建議提高訊息回覆率,平台演算法與旅客都重視回覆表現。"),
    "desc_len": ("房源文案偏短", "建議補充周邊生活機能、交通與特色介紹,將描述充實至 200 字以上。"),
    "amenities_count": ("設施數偏少", "設施豐富度低於競品,建議優先補齊旅客最在意的基本設施。"),
    "review_scores_communication": ("溝通評分偏低", "建議入住前主動訊息確認需求、入住中即時回應,拉高溝通評分。"),
    "review_scores_cleanliness": ("清潔評分偏低", "建議強化清潔 SOP 或更換清潔服務,清潔是評論的首要痛點。"),
    "score_pctl_nbhd": ("口碑落後同區競品", "整體評分在周邊競品中排名偏後,建議從清潔與溝通兩項優先改善。"),
    "nbr_density_same_type_1km": ("同類型競爭激烈", "周邊同房型供給密集,建議以差異化(設施/照片/文案)突圍。"),
}


class SuggestionEngine:
    def __init__(self, config: dict | None = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.templates = dict(SHAP_TEMPLATES)

    def suggest(self, shap_items: list, comp_stats: dict, features: dict,
                own_amenities: set) -> list:
        """產出建議清單(依優先度排序)。

        shap_items: [(feature, phi)],已依 phi 由大到小排序、僅含 phi>0。
        comp_stats: CompetitorIndex.stats() 輸出。
        features: 該房源目前特徵值 dict。
        """
        cfg = self.config
        out = []

        # 1) SHAP Top-K 扣分項 → 模板建議
        used_titles = set()
        for feat, phi in shap_items[: cfg["shap_top_k"]]:
            t = self.templates.get(feat)
            if not t:
                continue
            title, detail = t
            if title in used_titles:
                continue
            used_titles.add(title)
            out.append({
                "type": "shap", "title": title, "detail": detail,
                "evidence": f"此因素推高空屋率 {phi * 100:+.1f} 個百分點(SHAP)",
                "priority": 1,
            })

        # 2) 跨平台價格落點 → 降價建議(含具體目標價)
        pctl = comp_stats.get("pp_percentile")
        med = comp_stats.get("bracket_pp_median")
        if pctl is not None and pctl >= cfg["price_pctl_high"] and med:
            cap = max(float(features.get("accommodates", 2) or 2), 1)
            target = med * cap
            out.append({
                "type": "price", "title": "價格高於周邊跨平台行情",
                "detail": (f"以每人每晚等效價計,您貴於 1km 內同容量層競品的 {pctl:.0%}。"
                           f"建議將每晚定價往 NT$ {target:,.0f} 附近調整(同層中位數)。"),
                "evidence": f"1km 內同容量層競品 {comp_stats.get('n_same_bracket', 0)} 筆(Airbnb/Booking/591/ddroom)",
                "priority": 2,
            })

        # 3) 設施缺口 → 增加設施建議
        gaps = [(k, v) for k, v in comp_stats.get("amenity_coverage", {}).items()
                if v >= cfg["amenity_min_cov"] and k not in own_amenities]
        if gaps:
            top = gaps[: cfg["amenity_top_k"]]
            names = "、".join(f"{k}(周邊 {v:.0%} 競品有)" for k, v in top)
            out.append({
                "type": "amenity", "title": "設施低於周邊競品標配",
                "detail": f"建議優先增加:{names}。",
                "evidence": f"與 1km 內 {comp_stats.get('n_total', 0)} 筆跨平台競品之設施覆蓋率比較",
                "priority": 3,
            })

        # 4) 規則兜底(SHAP 未覆蓋時的基本檢查)
        if features.get("minimum_nights", 0) >= cfg["min_nights_long"] and \
                not any(s["title"] == "最低入住天數過長" for s in out):
            out.append({
                "type": "rule", "title": "最低入住天數過長",
                "detail": "建議在法規允許內放寬至 1~2 晚,承接週末短住需求。",
                "evidence": f"目前最低入住 {int(features['minimum_nights'])} 晚",
                "priority": 4,
            })
        if features.get("desc_len", 999) < cfg["desc_len_short"] and \
                not any(s["title"] == "房源文案偏短" for s in out):
            out.append({
                "type": "rule", "title": "房源文案偏短",
                "detail": "建議補充周邊機能與特色介紹,將描述充實至 200 字以上。",
                "evidence": f"目前描述僅 {int(features.get('desc_len', 0))} 字",
                "priority": 4,
            })

        return sorted(out, key=lambda s: s["priority"])
