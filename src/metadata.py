import pandas as pd
from pathlib import Path

from config import ENRICHED_CSV, resolve_image_path


class MetadataStore:
    def __init__(self, csv_path=ENRICHED_CSV):
        self.df = pd.read_csv(csv_path, on_bad_lines="skip", engine="python")
        self.by_id = self.df.set_index("id").to_dict(orient="index")

    def get(self, product_id):
        return self.by_id.get(int(product_id))

    def image_path(self, product_id) -> Path:
        row = self.by_id[int(product_id)]
        return resolve_image_path(row["image_path"])

    def all_ids(self):
        return list(self.by_id.keys())

    def category(self, product_id):
        return self.by_id[int(product_id)].get("category")

    def color(self, product_id):
        return self.by_id[int(product_id)].get("color")

    def price(self, product_id):
        return float(self.by_id[int(product_id)].get("price", 0.0))

    def pattern(self, product_id):
        return self.by_id[int(product_id)].get("pattern")

    def silhouette(self, product_id):
        return self.by_id[int(product_id)].get("silhouette")