"""
Harder evaluation with multiple strict criteria.

Runs 4 metrics on the same set of queries:
  1. Recall@12   - any same-cat+color item in top-12 (baseline)
  2. Top1        - rank-1 must be same-cat+color
  3. >=3 matches - at least 3 same-cat+color items in top-12
  4. NDCG@12     - rank-weighted relevance

Auto-detects v1 or v2 project.
"""
import sys
import os
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def detect_version():
    v2_files = [
        PROJECT_ROOT / "data_v2" / "metadata_enriched.csv",
        PROJECT_ROOT / "models" / "product_embeddings.npy",
    ]
    return "v2" if all(f.exists() for f in v2_files) else "v1"


def load_v2():
    from config import (
        ENRICHED_CSV, FAISS_INDEX_PATH, EMBEDDINGS_PATH,
        TOP_K_SEARCH, TOP_K_RETURN, resolve_image_path,
    )
    from src.embedding import SigLIP2Encoder
    from src.faiss_index import FAISSIndex
    from src.metadata import MetadataStore

    print("[v2] Loading SigLIP-2...")
    encoder = SigLIP2Encoder()
    index = FAISSIndex()
    index.load(FAISS_INDEX_PATH)
    ms = MetadataStore(ENRICHED_CSV)
    df = ms.df.copy()
    embeddings = np.load(EMBEDDINGS_PATH)
    return {
        "version": "v2", "encoder": encoder, "index": index, "df": df,
        "embeddings": embeddings, "top_k": TOP_K_SEARCH, "top_n": TOP_K_RETURN,
        "resolve_path": resolve_image_path,
    }


def load_v1():
    from config import METADATA_PATH, TOP_K_RETRIEVAL, TOP_N_DISPLAY
    from src.embedding import CLIPEncoder
    from src.faiss_index import FaissIndex
    from src.metadata import MetadataStore
    from src.preprocess import preprocess_image

    print("[v1] Loading CLIP...")
    encoder = CLIPEncoder()
    index = FaissIndex()
    index.load()
    ms = MetadataStore(METADATA_PATH)
    ms.load()
    df = None
    for attr in ["df", "metadata", "data"]:
        if hasattr(ms, attr):
            obj = getattr(ms, attr)
            if hasattr(obj, "iloc"):
                df = obj.copy()
                break

    def resolve_path(p):
        p = str(p)
        return p if os.path.isabs(p) else str(PROJECT_ROOT / p)

    return {
        "version": "v1", "encoder": encoder, "index": index, "df": df,
        "embeddings": None, "top_k": TOP_K_RETRIEVAL, "top_n": TOP_N_DISPLAY,
        "resolve_path": resolve_path, "preprocess_image": preprocess_image,
    }


def compute_metrics(cands, relevant):
    """
    Compute 4 metrics for a single query.
    cands    : list of 12 candidate IDs (ordered by score, descending)
    relevant : set of relevant catalog IDs
    """
    # 1. Recall@12 (any match)
    any_match = 1.0 if set(cands) & relevant else 0.0

    # 2. Top-1 strict (rank-1 must be relevant)
    top1 = 1.0 if (len(cands) > 0 and cands[0] in relevant) else 0.0

    # 3. >=3 matches
    n_matches = len(set(cands) & relevant)
    three_plus = 1.0 if n_matches >= 3 else 0.0

    # 4. NDCG@12
    ndcg = 0.0
    for rank, cid in enumerate(cands, 1):
        if cid in relevant:
            ndcg += 1.0 / np.log2(rank + 1)
    ideal = sum(1.0 / np.log2(r + 1) for r in range(1, min(len(relevant), 12) + 1))
    ndcg = ndcg / ideal if ideal > 0 else 0.0

    # Also keep standard mAP
    hits, precisions = 0, []
    for rank, cid in enumerate(cands, 1):
        if cid in relevant:
            hits += 1
            precisions.append(hits / rank)
    ap = float(np.mean(precisions)) if precisions else 0.0

    return {
        "recall_any": any_match,
        "top1": top1,
        "three_plus": three_plus,
        "ndcg": ndcg,
        "ap": ap,
        "n_matches": n_matches,
    }


def main():
    version = detect_version()
    print(f"Detected project: {version}\n")
    ctx = load_v2() if version == "v2" else load_v1()

    df = ctx["df"]
    index = ctx["index"]
    top_k = ctx["top_k"]
    top_n = ctx["top_n"]
    resolve_path = ctx["resolve_path"]
    df["id"] = df["id"].astype(int)

    if "color" in df.columns:
        df["_color_norm"] = df["color"].fillna("").str.lower().str.strip()
    else:
        df["_color_norm"] = ""

    print(f"Catalog size: {len(df)}")
    print(f"Metric: same category AND same color (strict)\n")

    strict_groups = {}
    for (cat, col), group in df.groupby(["category", "_color_norm"]):
        strict_groups[(cat, col)] = set(int(x) for x in group["id"].tolist())

    # Accumulators
    metrics = {
        "recall_any": [], "top1": [], "three_plus": [],
        "ndcg": [], "ap": [],
    }
    skipped = 0

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        try:
            qid = int(row["id"])
            true_cat = row["category"]
            true_color = row["_color_norm"]

            # Encode
            if ctx["version"] == "v2":
                q_vec = ctx["embeddings"][i].astype(np.float32)
            else:
                img_path = resolve_path(row["image_path"])
                img = Image.open(img_path).convert("RGB")
                tensor = ctx["preprocess_image"](img)
                q_vec = ctx["encoder"].encode_image(tensor).cpu().numpy().squeeze(0).astype(np.float32)
            q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-8)

            # Search
            if ctx["version"] == "v2":
                scores, cands = index.search(q_vec, k=top_k)
            else:
                cands, scores = index.search(q_vec.reshape(1, -1), k=top_k)

            cands = [int(c) for c in cands]
            cands = [c for c in cands if c != qid][:top_n]

            relevant = strict_groups.get((true_cat, true_color), set()) - {qid}
            if len(relevant) == 0:
                skipped += 1
                continue

            m = compute_metrics(cands, relevant)
            for k in metrics:
                metrics[k].append(m[k])

        except Exception as e:
            print(f"\n  FAILED on row {i} (id={row.get('id')}): {e}")
            traceback.print_exc()

    print()
    print(f"=== {version.upper()} Hard Eval ({len(metrics['recall_any'])} queries) ===")
    print(f"Skipped (no relevant) : {skipped}")
    print()
    print(f"Recall@12 (any match) : {np.mean(metrics['recall_any']):.4f}")
    print(f"Top-1 strict          : {np.mean(metrics['top1']):.4f}")
    print(f">=3 matches in top-12 : {np.mean(metrics['three_plus']):.4f}")
    print(f"NDCG@12               : {np.mean(metrics['ndcg']):.4f}")
    print(f"mAP@12                : {np.mean(metrics['ap']):.4f}")


if __name__ == "__main__":
    main()