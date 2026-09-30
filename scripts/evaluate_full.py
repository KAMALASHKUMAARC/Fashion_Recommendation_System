"""
Strict evaluation: relevance = same category AND same color.
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

    # Normalize color/pattern strings for comparison
    if "color" in df.columns:
        df["_color_norm"] = df["color"].fillna("").str.lower().str.strip()
    else:
        df["_color_norm"] = ""
    if "pattern" in df.columns:
        df["_pattern_norm"] = df["pattern"].fillna("").str.lower().str.strip()
    else:
        df["_pattern_norm"] = ""

    print(f"Catalog size: {len(df)}")
    print(f"Strict metric: same category AND same color\n")

    # Build strict relevance sets: (category, color) -> set of ids
    strict_groups = {}
    for (cat, col), group in df.groupby(["category", "_color_norm"]):
        strict_groups[(cat, col)] = set(int(x) for x in group["id"].tolist())

    recalls, aps = [], []
    skipped = 0

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        try:
            qid = int(row["id"])
            true_cat = row["category"]
            true_color = row["_color_norm"]

            # Encode query
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

            # Strict relevance: same (category, color)
            relevant = strict_groups.get((true_cat, true_color), set()) - {qid}

            if len(relevant) == 0:
                skipped += 1
                recalls.append(0.0)
                aps.append(0.0)
                continue

            r = 1.0 if set(cands) & relevant else 0.0
            recalls.append(r)

            hits, precisions = 0, []
            for rank, cid in enumerate(cands, 1):
                if cid in relevant:
                    hits += 1
                    precisions.append(hits / rank)
            ap = float(np.mean(precisions)) if precisions else 0.0
            aps.append(ap)

        except Exception as e:
            print(f"\n  FAILED on row {i} (id={row.get('id')}): {e}")
            traceback.print_exc()
            recalls.append(0.0)
            aps.append(0.0)

    print()
    print(f"=== {ctx['version'].upper()} Strict Eval (same category + color) ===")
    print(f"Total queries : {len(recalls)}")
    print(f"Skipped (no relevant): {skipped}")
    print(f"Recall@{top_n} : {np.mean(recalls):.4f}")
    print(f"mAP@{top_n}    : {np.mean(aps):.4f}")


if __name__ == "__main__":
    main()