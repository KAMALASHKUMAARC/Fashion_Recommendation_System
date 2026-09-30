import numpy as np

from config import (
    W_VISUAL, W_CATEGORY, W_COLOR, W_PATTERN, W_SILHOUETTE, W_PRICE,
    RANK_SCORES, SIBLING_SCORE, NO_MATCH_SCORE,
    CATEGORY_SIBLINGS,
)


def _are_siblings(c1: str, c2: str) -> bool:
    for group in CATEGORY_SIBLINGS:
        if c1 in group and c2 in group:
            return True
    return False


class HybridScoreAndRank:
    def __init__(self, metadata_store):
        self.ms = metadata_store
        # Mutable weights — can be overridden from the app
        self.W_VISUAL = W_VISUAL
        self.W_CATEGORY = W_CATEGORY
        self.W_COLOR = W_COLOR
        self.W_PATTERN = W_PATTERN
        self.W_SILHOUETTE = W_SILHOUETTE
        self.W_PRICE = W_PRICE

    def _visual_score(self, faiss_distances) -> np.ndarray:
        return np.clip((faiss_distances + 1.0) / 2.0, 0.0, 1.0)

    def _category_score(self, query_ranked, candidate_id) -> float:
        cand_cat = self.ms.category(candidate_id)
        if cand_cat is None:
            return NO_MATCH_SCORE
        for rank, (qcat, _) in enumerate(query_ranked[:5]):
            if qcat == cand_cat:
                return RANK_SCORES.get(rank, NO_MATCH_SCORE)
            if _are_siblings(qcat, cand_cat):
                return SIBLING_SCORE
        return NO_MATCH_SCORE

    def _color_score(self, query_color, candidate_id) -> float:
        cand_color = self.ms.color(candidate_id)
        if cand_color is None or query_color is None:
            return 0.0
        return 1.0 if str(query_color).lower() == str(cand_color).lower() else 0.0

    def _pattern_score(self, query_pattern, candidate_id) -> float:
        cand = self.ms.pattern(candidate_id)
        if cand is None or query_pattern is None:
            return 0.0
        return 1.0 if str(query_pattern).lower() == str(cand).lower() else 0.0

    def _silhouette_score(self, query_sil, candidate_id) -> float:
        cand = self.ms.silhouette(candidate_id)
        if cand is None or query_sil is None:
            return 0.0
        return 1.0 if str(query_sil).lower() == str(cand).lower() else 0.0

    def _price_score(self, query_price, candidate_id) -> float:
        cand_price = self.ms.price(candidate_id)
        if query_price is None or cand_price is None or query_price <= 0:
            return 0.0
        return float(min(query_price, cand_price) / max(query_price, cand_price))

    def rank(self, query, candidate_ids, faiss_distances) -> list:
        visual_scores = self._visual_score(faiss_distances)
        results = []

        for i, cid in enumerate(candidate_ids):
            s_vis = float(visual_scores[i])
            s_cat = self._category_score(query["category_ranked"], cid)
            s_col = self._color_score(query["color"], cid)
            s_pat = self._pattern_score(query["pattern"], cid)
            s_sil = self._silhouette_score(query["silhouette"], cid)
            s_prc = self._price_score(query["price"], cid)

            final = (
                self.W_VISUAL * s_vis +
                self.W_CATEGORY * s_cat +
                self.W_COLOR * s_col +
                self.W_PATTERN * s_pat +
                self.W_SILHOUETTE * s_sil +
                self.W_PRICE * s_prc
            )

            results.append({
                "id": int(cid),
                "final_score": float(final),
                "visual": s_vis,
                "category": s_cat,
                "color": s_col,
                "pattern": s_pat,
                "silhouette": s_sil,
                "price": s_prc,
            })

        results.sort(key=lambda r: -r["final_score"])
        return results