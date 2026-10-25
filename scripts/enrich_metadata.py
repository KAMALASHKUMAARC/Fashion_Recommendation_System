import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import METADATA_CSV, ENRICHED_CSV, resolve_image_path
from src.preprocess import load_image
from src.embedding import SigLIP2Encoder
from src.attributes import predict_pattern, predict_silhouette

# will be needed to run before hand to save pattern and silhouettes for each image in dataset for faster recovery (one time)
def main():
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    print(f"\nLoading metadata from {METADATA_CSV}")
    df = pd.read_csv(METADATA_CSV, on_bad_lines="skip", engine="python")
    print(f"  rows: {len(df)}")

    print("Loading SigLIP-2 encoder...")
    encoder = SigLIP2Encoder()

    patterns, silhouettes = [], []
    fail_count = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Enriching"):
        try:
            img_path = resolve_image_path(row["image_path"])
            img = load_image(img_path)
            patterns.append(predict_pattern(encoder, img))
            silhouettes.append(predict_silhouette(encoder, img))
        except Exception as e:
            fail_count += 1
            if fail_count <= 5:
                print(f"  [warn] {row.get('image_path')}: {e}")
            patterns.append("unknown")
            silhouettes.append("unknown")

    print(f"\nTotal failures: {fail_count} / {len(df)}")

    df["pattern"] = patterns
    df["silhouette"] = silhouettes

    df.to_csv(ENRICHED_CSV, index=False)
    print(f"Wrote {ENRICHED_CSV} ({len(df)} rows)")


if __name__ == "__main__":
    main()