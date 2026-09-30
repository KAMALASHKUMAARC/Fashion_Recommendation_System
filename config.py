import os
from pathlib import Path

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_V2_DIR = ROOT / "data_v2"
MODELS_DIR = ROOT / "models"
CATALOG_DIR = DATA_DIR / "catalog"
METADATA_CSV = DATA_DIR / "metadata.csv"
ENRICHED_CSV = DATA_V2_DIR / "metadata_enriched.csv"

FAISS_INDEX_PATH = MODELS_DIR / "faiss_index.bin"
EMBEDDINGS_PATH = MODELS_DIR / "product_embeddings.npy"
ID_MAP_PATH = MODELS_DIR / "id_map.npy"

for d in [DATA_V2_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def resolve_image_path(image_path_str) -> Path:
    """Resolve image path from metadata.csv robustly."""
    p = Path(str(image_path_str))
    if p.is_absolute():
        return p
    if "catalog" in p.parts:
        return ROOT / p
    return CATALOG_DIR / p


# ---------- Model ----------
MODEL_ID = "google/siglip2-so400m-patch16-512"
IMAGE_SIZE = 512
EMBEDDING_DIM = 3456
BASE_EMBEDDING_DIM = 1152
BATCH_SIZE = 8
DEVICE = "cuda"
DTYPE = "float16"

CROP_REGIONS = {
    "full":   (0.00, 1.00),
    "top":    (0.00, 0.55),
    "bottom": (0.45, 1.00),
}

# ---------- FAISS ----------
NLIST = 100
NPROBE = 10
TOP_K_SEARCH = 500
TOP_K_RETURN = 12

# ---------- Hybrid Ranker Weights ----------
W_VISUAL     = 0.50
W_CATEGORY   = 0.20
W_COLOR      = 0.00
W_PATTERN    = 0.10
W_SILHOUETTE = 0.10
W_PRICE      = 0.10

# ---------- Category Scoring ----------
RANK_SCORES = {0: 1.00, 1: 0.90, 2: 0.80, 3: 0.70, 4: 0.60}
SIBLING_SCORE = 0.25
NO_MATCH_SCORE = 0.00

# ---------- Attribute Taxonomies ----------
PATTERNS = [
    "solid", "striped", "floral", "plaid",
    "polka dot", "geometric", "animal print", "abstract",
]

SILHOUETTES = [
    "fitted", "loose", "a-line", "bodycon",
    "oversized", "straight", "flared", "peplum",
]

# ---------- Attribute Prompts ----------
PATTERN_PROMPTS = {
    "solid":        "a photo of a solid colored garment with no pattern",
    "striped":      "a photo of a striped garment",
    "floral":       "a photo of a floral print garment",
    "plaid":        "a photo of a plaid or checkered garment",
    "polka dot":    "a photo of a polka dot garment",
    "geometric":    "a photo of a geometric pattern garment",
    "animal print": "a photo of an animal print garment",
    "abstract":     "a photo of an abstract print garment",
}

SILHOUETTE_PROMPTS = {
    "fitted":    "a photo of a fitted, tight garment",
    "loose":     "a photo of a loose, baggy garment",
    "a-line":    "a photo of an a-line silhouette garment",
    "bodycon":   "a photo of a bodycon dress",
    "oversized": "a photo of an oversized garment",
    "straight":  "a photo of a straight-cut garment",
    "flared":    "a photo of a flared garment",
    "peplum":    "a photo of a peplum top",
}

# ---------- 142 Categories (exact from catalog) ----------
QUERY_CATEGORIES = [
    "Accessory Gift Set", "Baby Dolls", "Backpacks", "Bangle", "Basketballs",
    "Bath Robe", "Beauty Accessory", "Belts", "Blazers", "Body Lotion",
    "Body Wash and Scrub", "Booties", "Boxers", "Bra", "Bracelet",
    "Briefs", "Camisoles", "Capris", "Caps", "Casual Shoes",
    "Churidar", "Clothing Set", "Clutches", "Compact", "Concealer",
    "Cufflinks", "Cushion Covers", "Deodorant", "Dresses", "Duffel Bag",
    "Dupatta", "Earrings", "Eye Cream", "Eyeshadow", "Face Moisturisers",
    "Face Scrub and Exfoliator", "Face Serum and Gel", "Face Wash and Cleanser",
    "Flats", "Flip Flops", "Footballs", "Formal Shoes", "Foundation and Primer",
    "Fragrance Gift Set", "Free Gifts", "Gloves", "Hair Accessory", "Hair Colour",
    "Handbags", "Hat", "Headband", "Heels", "Highlighter and Blush",
    "Innerwear Vests", "Ipad", "Jackets", "Jeans", "Jeggings",
    "Jewellery Set", "Jumpsuit", "Kajal and Eyeliner", "Key chain", "Kurta Sets",
    "Kurtas", "Kurtis", "Laptop Bag", "Leggings", "Lehenga Choli",
    "Lip Care", "Lip Gloss", "Lip Liner", "Lip Plumper", "Lipstick",
    "Lounge Pants", "Lounge Shorts", "Lounge Tshirts", "Makeup Remover",
    "Mascara", "Mask and Peel", "Mens Grooming Kit", "Messenger Bag",
    "Mobile Pouch", "Mufflers", "Nail Essentials", "Nail Polish",
    "Necklace and Chains", "Nehru Jackets", "Night suits", "Nightdress",
    "Patiala", "Pendant", "Perfume and Body Mist", "Rain Jacket",
    "Rain Trousers", "Ring", "Robe", "Rompers", "Rucksacks",
    "Salwar", "Salwar and Dupatta", "Sandals", "Sarees", "Scarves",
    "Shapewear", "Shirts", "Shoe Accessories", "Shoe Laces", "Shorts",
    "Shrug", "Skirts", "Socks", "Sports Sandals", "Sports Shoes",
    "Stockings", "Stoles", "Sunglasses", "Sunscreen", "Suspenders",
    "Sweaters", "Sweatshirts", "Swimwear", "Tablet Sleeve", "Ties",
    "Ties and Cufflinks", "Tights", "Toner", "Tops", "Track Pants",
    "Tracksuits", "Travel Accessory", "Trolley Bag", "Trousers",
    "Trunk", "Tshirts", "Tunics", "Umbrellas", "Waist Pouch",
    "Waistcoat", "Wallets", "Watches", "Water Bottle", "Wristbands",
]

# ---------- Sibling Groups ----------
CATEGORY_SIBLINGS = [
    {"Salwar", "Churidar", "Patiala", "Salwar and Dupatta"},
    {"Dresses", "Baby Dolls", "Nightdress", "Robe", "Bath Robe"},
    {"Kurtas", "Kurtis", "Kurta Sets", "Tunics"},
    {"Tops", "Lounge Tshirts", "Tshirts", "Shirts"},
    {"Jeans", "Jeggings", "Leggings", "Trousers", "Track Pants", "Capris",
     "Lounge Pants", "Rain Trousers", "Tights"},
    {"Shorts", "Lounge Shorts", "Boxers", "Briefs", "Trunk"},
    {"Skirts"},
    {"Casual Shoes", "Formal Shoes", "Sports Shoes", "Flats", "Heels",
     "Sandals", "Sports Sandals", "Flip Flops", "Booties"},
    {"Handbags", "Clutches", "Messenger Bag", "Laptop Bag", "Duffel Bag",
     "Rucksacks", "Backpacks", "Trolley Bag", "Waist Pouch", "Mobile Pouch",
     "Tablet Sleeve", "Travel Accessory"},
    {"Bangle", "Bracelet", "Ring", "Pendant", "Necklace and Chains",
     "Earrings", "Jewellery Set", "Cufflinks", "Ties and Cufflinks"},
    {"Bra", "Camisoles", "Shapewear", "Innerwear Vests", "Stockings",
     "Socks", "Swimwear"},
    {"Sweaters", "Sweatshirts", "Jackets", "Blazers", "Shrug", "Waistcoat",
     "Nehru Jackets", "Rain Jacket"},
    {"Watches", "Wristbands", "Bracelet"},
    {"Scarves", "Stoles", "Mufflers", "Dupatta", "Ties"},
    {"Caps", "Hat", "Headband", "Hair Accessory", "Hair Colour"},
]