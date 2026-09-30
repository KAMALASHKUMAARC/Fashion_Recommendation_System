import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

from config import (
    FAISS_INDEX_PATH, ENRICHED_CSV, TOP_K_SEARCH,
    QUERY_CATEGORIES,
)
from src.embedding import SigLIP2Encoder
from src.faiss_index import FAISSIndex
from src.metadata import MetadataStore
from src.attributes import enrich_attributes
from src.hybrid_ranker import HybridScoreAndRank
from src.clip_classifier import classify_category, classify_color

st.set_page_config(page_title="AIML v2 — Fashion Retrieval", layout="wide")
st.title("AIML v2 — SigLIP-2 Fashion Retrieval")

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

# ---------- Sidebar: Search Mode ----------
st.sidebar.header("Search Mode")
search_mode = st.sidebar.radio(
    "Choose input type:",
    ["Image upload", "Text query"],
    index=0,
)

# ---------- Sidebar: Weights ----------
st.sidebar.markdown("---")
st.sidebar.header("Hybrid Ranker Weights")

w_visual = st.sidebar.slider("Visual", 0.0, 1.0, 0.35, 0.05)
w_category = st.sidebar.slider("Category", 0.0, 1.0, 0.20, 0.05)
w_color = st.sidebar.slider("Color", 0.0, 1.0, 0.15, 0.05)
w_pattern = st.sidebar.slider("Pattern", 0.0, 1.0, 0.10, 0.05)
w_silhouette = st.sidebar.slider("Silhouette", 0.0, 1.0, 0.10, 0.05)
w_price = st.sidebar.slider("Price", 0.0, 1.0, 0.10, 0.05)

total_weight = w_visual + w_category + w_color + w_pattern + w_silhouette + w_price
if abs(total_weight - 1.0) > 0.01:
    st.sidebar.warning(f"⚠️ Weights sum to {total_weight:.2f} (should be 1.00)")
else:
    st.sidebar.success(f"✅ Weights sum to {total_weight:.2f}")

st.sidebar.markdown("**Category filter**")
use_filter = st.sidebar.checkbox("Enable category filter", value=True)


# ================================================================
# Query handling
# ================================================================

if search_mode == "Image upload":
    uploaded = st.file_uploader("Upload a fashion image", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("Upload an image to start.")
        st.stop()

    img = Image.open(uploaded).convert("RGB")
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image(img, caption="Query", use_container_width=True)

    with st.spinner("Encoding & searching..."):
        q_vec = encoder.encode_image_multicrop(img)
        distances, cand_ids = index.search(q_vec, k=TOP_K_SEARCH)
        attrs = enrich_attributes(encoder, img)
        cat_result = classify_category(img, QUERY_CATEGORIES)

else:  # Text query
    text_query = st.text_input(
        "Describe the item:",
        placeholder="e.g., blue floral dress",
    )
    if not text_query.strip():
        st.info("Type a description to start.")
        st.stop()

    with st.spinner("Encoding text & searching..."):
        # Encode text with CLIP (SigLIP-2 text is unreliable)
        from src.clip_classifier import _get_clip
        import torch

        clip_model, _, clip_tokenizer = _get_clip()

        # --- Text embedding (1152-dim from CLIP) ---
        text_tokens = clip_tokenizer([text_query]).cuda()
        with torch.no_grad():
            text_emb = clip_model.encode_text(text_tokens)
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
            text_emb = text_emb.squeeze(0).float().cpu().numpy()   # (512,)

        # CLIP ViT-B/32 is 512-dim. Triplicate to match 3456-dim index.
        # Pad/truncate to 1152 per crop by repeating.
        # Safer: encode into SigLIP-2 space via SigLIP-2's text encoder too.
        siglip_text = encoder.encode_text([text_query])[0]   # (1152,)
        q_vec = np.concatenate([siglip_text, siglip_text, siglip_text])  # (3456,)

        distances, cand_ids = index.search(q_vec, k=TOP_K_SEARCH)

        # --- Category prediction from text (using CLIP) ---
        cat_tokens = clip_tokenizer(
            [f"a photo of a {c}" for c in QUERY_CATEGORIES]
        ).cuda()
        with torch.no_grad():
            cat_embs = clip_model.encode_text(cat_tokens)
            cat_embs = cat_embs / cat_embs.norm(dim=-1, keepdim=True)
            text_emb_ = clip_model.encode_text(clip_tokenizer([text_query]).cuda())
            text_emb_ = text_emb_ / text_emb_.norm(dim=-1, keepdim=True)
            sims = (100.0 * text_emb_ @ cat_embs.T).softmax(dim=-1)
            probs = sims.squeeze(0).float().cpu().numpy()

        ranked_cats = sorted(zip(QUERY_CATEGORIES, probs), key=lambda x: -x[1])
        cat_result = {"top5": ranked_cats[:5], "all": dict(ranked_cats)}

        # Attributes from text aren't computed — mark as unknown
        attrs = {
            "color": "unknown",
            "pattern": "unknown",
            "silhouette": "unknown",
        }

# ================================================================
# Common downstream: category filter, rank, display
# ================================================================

top5 = cat_result["top5"]

# ---------- Category filter ----------
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

if search_mode == "Image upload":
    st.markdown("**Other attributes:**")
    st.write({
        "color": attrs["color"],
        "pattern": attrs["pattern"],
        "silhouette": attrs["silhouette"],
    })

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
            f"vis={r['visual']:.2f} | "
            f"cat={r['category']:.2f} | "
            f"col={r['color']:.2f} | "
            f"pat={r['pattern']:.2f} | "
            f"sil={r['silhouette']:.2f} | "
            f"price={r['price']:.2f}"
        )
        with st.expander("Weighted contributions"):
            st.write({
                "visual": round(w_visual * r["visual"], 3),
                "category": round(w_category * r["category"], 3),
                "color": round(w_color * r["color"], 3),
                "pattern": round(w_pattern * r["pattern"], 3),
                "silhouette": round(w_silhouette * r["silhouette"], 3),
                "price": round(w_price * r["price"], 3),
            })

# ---------- Stats ----------
st.markdown("---")
st.subheader("Result statistics")
categories_returned = [ms.category(r["id"]) for r in ranked[:TOP_K_DISPLAY]]
colors_returned = [ms.color(r["id"]) for r in ranked[:TOP_K_DISPLAY]]
patterns_returned = [ms.pattern(r["id"]) for r in ranked[:TOP_K_DISPLAY]]

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**Categories (top-24):**")
    st.write(pd.Series(categories_returned).value_counts().to_dict())
with col2:
    st.markdown("**Colors (top-24):**")
    st.write(pd.Series(colors_returned).value_counts().to_dict())
with col3:
    st.markdown("**Patterns (top-24):**")
    st.write(pd.Series(patterns_returned).value_counts().to_dict())