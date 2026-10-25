import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    ENRICHED_CSV, EMBEDDINGS_PATH, ID_MAP_PATH, FAISS_INDEX_PATH,
    EMBEDDING_DIM, resolve_image_path,
)
from src.preprocess import load_image
from src.embedding import SigLIP2Encoder
from src.faiss_index import FAISSIndex


CHECKPOINT_EVERY = 2000

#saves embeddings, id_map, faiss index before hand
def main():
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    df = pd.read_csv(ENRICHED_CSV, on_bad_lines="skip", engine="python")
    print(f"Catalog rows: {len(df)}")

    encoder = SigLIP2Encoder()

    embeddings = []
    ids = []

    ckpt_emb = EMBEDDINGS_PATH.with_suffix(".partial.npy")
    ckpt_ids = ID_MAP_PATH.with_suffix(".partial.npy")
    if ckpt_emb.exists() and ckpt_ids.exists():
        embeddings = list(np.load(ckpt_emb))
        ids = list(np.load(ckpt_ids))
        print(f"Resumed from checkpoint: {len(ids)} encoded")

    start = len(ids)

    for _, row in tqdm(df.iloc[start:].iterrows(), total=len(df) - start, desc="Encoding"):
        try:
            img_path = resolve_image_path(row["image_path"])
            img = load_image(img_path)
            vec = encoder.encode_image_multicrop(img)
            embeddings.append(vec)
            ids.append(int(row["id"]))
        except Exception as e:
            print(f"  [warn] skip {row.get('image_path')}: {e}")

        if (len(ids) % CHECKPOINT_EVERY) == 0:
            np.save(ckpt_emb, np.stack(embeddings))
            np.save(ckpt_ids, np.asarray(ids))
            print(f"  checkpoint @ {len(ids)}")

    embeddings = np.stack(embeddings).astype(np.float32)
    ids = np.asarray(ids)

    assert embeddings.shape[1] == EMBEDDING_DIM, embeddings.shape

    np.save(EMBEDDINGS_PATH, embeddings)
    np.save(ID_MAP_PATH, ids)
    print(f"Saved embeddings: {embeddings.shape}")
    print(f"Saved id_map: {ids.shape}")

    print("Building FAISS index...")
    index = FAISSIndex(dim=EMBEDDING_DIM)
    index.build(embeddings, ids)
    index.save(FAISS_INDEX_PATH)
    print(f"Saved FAISS index -> {FAISS_INDEX_PATH}")

    ckpt_emb.unlink(missing_ok=True)
    ckpt_ids.unlink(missing_ok=True)


if __name__ == "__main__":
    main()