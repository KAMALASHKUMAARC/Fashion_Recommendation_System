"""
AIML v2 — Multimodal Fashion Retrieval.

Supports:
  - Image-only queries
  - Text-only queries
  - Multimodal (image + text) queries with adjustable blend ratio
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

from config import (
    FAISS_INDEX_PATH, ENRICHED_CSV, TOP_K_SEARCH,
    QUERY_CATEGORIES, PATTERN_PROMPTS, SILHOUETTE_PROMPTS,
)
from src.embedding import SigLIP2Encoder
from src.faiss_index import FAISSIndex
from src.metadata import MetadataStore
from src.attributes import enrich_attributes
from src.hybrid_ranker import HybridScoreAndRank
from src.clip_classifier import (
    classify_category, classify_color,
    classify_category_from_text, classify_color_from_text,
)

st.set_page_config(page_title="AIML v2 — Fashion Retrieval", layout="wide")
st.title("AIML v2 — Fashion Retrieval (Image + Text)")

TOP_K_DISPLAY = 24


@st.cache_resource
def load_components():
    encoder = SigLIP2Encoder()
    index = FAISSIndex()
    index.load(FAISS_INDEX_PATH)
    ms = MetadataStore(ENRICHED_CSV)
    ranker = HybridScoreAndRank(ms)
    return encoder, index, ms, ranker


encoder, index, ms, ranker = load_components()

# ---------- Sidebar ----------
st.sidebar.header("Hybrid Ranker Weights")
w_visual = st.sidebar.slider("Visual", 0.0, 1.0, 0.35, 0.05)
w_category = st.sidebar.slider("Category", 0.0, 1.0, 0.20, 0.05)
w_color = st.sidebar.slider("Color", 0.0, 1.0, 0.15, 0.05)
w_pattern = st.sidebar.slider("Pattern", 0.0, 1.0, 0.10, 0.05)
w_silhouette = st.sidebar.slider("Silhouette", 0.0, 1.0, 0.10, 0.05)
w_price = st.sidebar.slider("Price", 0.0, 1.0, 0.10, 0.05)

total = w_visual + w_category + w_color + w_pattern + w_silhouette + w_price
if abs(total - 1.0) > 0.01:
    st.sidebar.warning(f"⚠️ Weights sum to {total:.2f}")
else:
    st.sidebar.success(f"✅ Weights sum to {total:.2f}")

st.sidebar.markdown("---")
st.sidebar.header("🔀 Multimodal Blend")
alpha = st.sidebar.slider(
    "Image ↔ Text weight",
    min_value=0.0, max_value=1.0, value=0.7, step=0.05,
    help="1.0 = image only, 0.0 = text only",
)

st.sidebar.markdown("**Category filter**")
use_filter = st.sidebar.checkbox("Enable category filter", value=True)


# ---------- Query inputs ----------
st.subheader("Query")

col_img, col_txt = st.columns(2)
with col_img:
    uploaded = st.file_uploader("Upload a fashion image (optional)",
                                 type=["jpg", "jpeg", "png"])
with col_txt:
    text_query = st.text_input("Text description (optional)",
                               placeholder="e.g., pink floral dress")

has_image = uploaded is not None
has_text = text_query.strip() != ""

if not has_image and not has_text:
    st.info("Provide an image, a text query, or both to start.")
    st.stop()

# Show query image
img = None
if has_image:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Query image", width=300)


# ---------- Encode ----------
with st.spinner("Encoding & searching..."):

    # ---- Embedding Fusion ----
    q_img = encoder.encode_image_multicrop(img) if has_image else None
    q_txt = encoder.encode_text_multicrop(text_query) if has_text else None

    if q_img is not None and q_txt is not None:
        q_vec = alpha * q_img + (1 - alpha) * q_txt
    elif q_img is not None:
        q_vec = q_img
    else:
        q_vec = q_txt

    q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-8)

    # ---- FAISS search ----
    distances, cand_ids = index.search(q_vec, k=TOP_K_SEARCH)

    # ---- Attribute & Category Prediction (SMART LOGIC) ----
    # If slider is near 0, use TEXT. If near 1, use IMAGE.
    if alpha < 0.2 and has_text:
        # Text-dominant mode
        cat_result = classify_category_from_text(text_query, QUERY_CATEGORIES)
        attrs = {
            "color": classify_color_from_text(text_query),
            "pattern": "unknown",
            "silhouette": "unknown",
        }
    elif has_image:
        # Image-dominant mode
        cat_result = classify_category(img, QUERY_CATEGORIES)
        attrs = enrich_attributes(encoder, img)
    else:
        # Text only fallback
        cat_result = classify_category_from_text(text_query, QUERY_CATEGORIES)
        attrs = {
            "color": classify_color_from_text(text_query),
            "pattern": "unknown",
            "silhouette": "unknown",
        }

    top5 = cat_result["top5"]

    # ---- Category filter ----
    if use_filter:
        top3_cats = set(top5[i][0] for i in range(3))
        filtered = [
            (d, int(c))
            for d, c in zip(distances, cand_ids)
            if ms.category(int(c)) in top3_cats
        ]
        if len(filtered) >= TOP_K_DISPLAY:
            distances = np.array([d for d, _ in filtered], dtype=np.float32)
            cand_ids = np.array([c for _, c in filtered], dtype=np.int64)

    # ---- Rank ----
    query = {
        "category_ranked": top5,
        "color": attrs["color"],
        "pattern": attrs["pattern"],
        "silhouette": attrs["silhouette"],
        "price": 0.0,
    }

    ranker.W_VISUAL = w_visual
    ranker.W_CATEGORY = w_category
    ranker.W_COLOR = w_color
    ranker.W_PATTERN = w_pattern
    ranker.W_SILHOUETTE = w_silhouette
    ranker.W_PRICE = w_price

    ranked = ranker.rank(query, cand_ids, distances)


# ---------- Display attributes ----------
st.subheader("Predicted attributes")

st.markdown("**Top-3 categories:**")
cat_data = {
    "Rank": [f"#{i+1}" for i in range(3)],
    "Category": [top5[i][0] for i in range(3)],
    "Probability": [f"{top5[i][1]*100:.2f}%" for i in range(3)],
}
st.table(cat_data)

st.markdown("**Other attributes:**")
st.write({
    "color": attrs["color"],
    "pattern": attrs["pattern"],
    "silhouette": attrs["silhouette"],
})

if has_image and has_text:
    st.info(f"🔀 Blending: **{alpha*100:.0f}% image + {(1-alpha)*100:.0f}% text**")

# ---------- Results ----------
st.subheader(f"Top {TOP_K_DISPLAY} results")
cols = st.columns(4)

for i, r in enumerate(ranked[:TOP_K_DISPLAY]):
    pid = r["id"]
    path = ms.image_path(pid)
    with cols[i % 4]:
        st.image(str(path), use_container_width=True)
        st.markdown(f"**{ms.category(pid)}** — `{r['final_score']:.3f}`")
        st.caption(
            f"vis={r['visual']:.2f} | cat={r['category']:.2f} | "
            f"col={r['color']:.2f} | pat={r['pattern']:.2f} | "
            f"sil={r['silhouette']:.2f} | price={r['price']:.2f}"
        )