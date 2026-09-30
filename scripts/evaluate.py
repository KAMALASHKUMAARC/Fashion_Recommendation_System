"""
Evaluate v2 retrieval: Recall@12 and mAP@12 on a sample of catalog images.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import random
import traceback
import numpy as np

from config import (
    ENRICHED_CSV, FAISS_INDEX_PATH, TOP_K_SEARCH, TOP_K_RETURN,
    QUERY_CATEGORIES, resolve_image_path,
)
from src.preprocess import load_image
from src.embedding import SigLIP2Encoder
from src.faiss_index import FAISSIndex
from src.metadata import MetadataStore
from src.attributes import enrich_attributes
from src.hybrid_ranker import HybridScoreAndRank
from src.utils import recall_at_k, average_precision_at_k
from src.clip_classifier import classify_category


def main(n_queries: int = 20, seed: int = 0, use_filter: bool = True):
    random.seed(seed)

    print("[1/3] Loading SigLIP-2 encoder...")
    encoder = SigLIP2Encoder()

    print("[2/3] Loading FAISS index...")
    index = FAISSIndex()
    index.load(FAISS_INDEX_PATH)
    print(f"     index.ntotal = {index.index.ntotal}")

    print("[3/3] Loading metadata...")
    ms = MetadataStore(ENRICHED_CSV)
    ranker = HybridScoreAndRank(ms)

    all_ids = ms.all_ids()
    print(f"     catalog size: {len(all_ids)}")
    print(f"     category filter: {use_filter}\n")

    queries = random.sample(all_ids, n_queries)
    print(f"Running {n_queries} queries...\n")

    recalls, aps = [], []

    for qi, qid in enumerate(queries, 1):
        try:
            img_path = resolve_image_path(ms.get(qid)["image_path"])
            img = load_image(img_path)

            q_vec = encoder.encode_image_multicrop(img)
            distances, cand_ids = index.search(q_vec, k=TOP_K_SEARCH)

            attrs = enrich_attributes(encoder, img)
            cat_result = classify_category(img, QUERY_CATEGORIES)

            # ---------- Category filter ----------
            if use_filter:
                top3_cats = set(cat_result["top5"][i][0] for i in range(3))
                filtered = [
                    (d, int(c))
                    for d, c in zip(distances, cand_ids)
                    if ms.category(int(c)) in top3_cats
                ]
                if len(filtered) >= TOP_K_RETURN:
                    distances = np.array([d for d, _ in filtered], dtype=np.float32)
                    cand_ids = np.array([c for _, c in filtered], dtype=np.int64)

            query = {
                "category_ranked": cat_result["top5"],
                "color": attrs["color"],
                "pattern": attrs["pattern"],
                "silhouette": attrs["silhouette"],
                "price": ms.price(qid),
            }
            ranked = ranker.rank(query, cand_ids, distances)
            retrieved = [r["id"] for r in ranked[:TOP_K_RETURN]]

            q_cat = ms.category(qid)
            relevant = [i for i in all_ids if ms.category(i) == q_cat and i != qid]

            r = recall_at_k(retrieved, relevant, TOP_K_RETURN)
            a = average_precision_at_k(retrieved, relevant, TOP_K_RETURN)
            recalls.append(r)
            aps.append(a)

            print(f"  [{qi}/{n_queries}] id={qid} cat={q_cat} "
                  f"pred={cat_result['top5'][0][0]} "
                  f"recall={r:.2f} ap={a:.2f}")

        except Exception as e:
            print(f"  [{qi}/{n_queries}] id={qid} FAILED: {e}")
            traceback.print_exc()
            recalls.append(0.0)
            aps.append(0.0)

    print(f"\n=== v2 Evaluation ({n_queries} queries) ===")
    print(f"Recall@{TOP_K_RETURN} : {np.mean(recalls):.4f}")
    print(f"mAP@{TOP_K_RETURN}    : {np.mean(aps):.4f}")


if __name__ == "__main__":
    main()