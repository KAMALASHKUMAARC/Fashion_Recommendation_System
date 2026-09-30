import torch
import numpy as np
import open_clip
from PIL import Image

_model = None
_preprocess = None
_tokenizer = None


def _get_clip():
    global _model, _preprocess, _tokenizer
    if _model is None:
        print("[ClipClassifier] Loading CLIP ViT-B/32...")
        _model, _, _preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="laion400m_e32",
            force_quick_gelu=True,
        )
        _model = _model.cuda().eval().half()
        _tokenizer = open_clip.get_tokenizer("ViT-B-32")
        print("[ClipClassifier] Ready.")
    return _model, _preprocess, _tokenizer


@torch.no_grad()
def classify_category(img: Image.Image, categories: list, top_k: int = 5) -> dict:
    model, preprocess, tokenizer = _get_clip()
    image = preprocess(img).unsqueeze(0).cuda().half()
    text = tokenizer([f"a photo of a {c}" for c in categories]).cuda()

    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    sims = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    probs = sims.squeeze(0).float().cpu().numpy()

    ranked = sorted(zip(categories, probs), key=lambda x: -x[1])
    return {"top5": ranked[:top_k], "all": dict(ranked)}


# ---------- Color ----------
COLOR_PROMPTS = [
    "a photo of a solid black garment",
    "a photo of a solid white garment",
    "a photo of a solid gray garment",
    "a photo of a solid grey garment",
    "a photo of a solid red garment",
    "a photo of a solid maroon garment",
    "a photo of a solid pink garment",
    "a photo of a solid rose garment",
    "a photo of a solid peach garment",
    "a photo of a solid orange garment",
    "a photo of a solid yellow garment",
    "a photo of a solid gold garment",
    "a photo of a solid green garment",
    "a photo of a solid olive garment",
    "a photo of a solid teal garment",
    "a photo of a solid blue garment",
    "a photo of a solid navy blue garment",
    "a photo of a solid sky blue garment",
    "a photo of a solid purple garment",
    "a photo of a solid lavender garment",
    "a photo of a solid brown garment",
    "a photo of a solid beige garment",
    "a photo of a solid cream garment",
    "a photo of a solid tan garment",
    "a photo of a solid silver garment",
]

COLOR_LABELS = [
    "black", "white", "gray", "gray",
    "red", "maroon", "pink", "rose",
    "peach", "orange", "yellow", "gold",
    "green", "olive", "teal", "blue",
    "navy", "sky blue", "purple", "lavender",
    "brown", "beige", "cream", "tan", "silver",
]


@torch.no_grad()
def classify_color(img: Image.Image) -> str:
    model, preprocess, tokenizer = _get_clip()
    image = preprocess(img).unsqueeze(0).cuda().half()
    text = tokenizer(COLOR_PROMPTS).cuda()

    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    sims = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    probs = sims.squeeze(0).float().cpu().numpy()

    label_probs = {}
    for label, p in zip(COLOR_LABELS, probs):
        label_probs[label] = label_probs.get(label, 0.0) + float(p)

    return max(label_probs, key=label_probs.get)

# ---------- Pattern classification ----------
PATTERN_PROMPTS_CLIP = [
    "a photo of a solid color garment with no pattern",
    "a photo of a striped garment",
    "a photo of a floral print garment with flowers",
    "a photo of a plaid checkered garment",
    "a photo of a polka dot garment",
    "a photo of a geometric pattern garment",
    "a photo of an animal print garment",
    "a photo of an abstract print garment",
]

PATTERN_LABELS = [
    "solid", "striped", "floral", "plaid",
    "polka dot", "geometric", "animal print", "abstract",
]


@torch.no_grad()
def classify_pattern(img: Image.Image) -> str:
    model, preprocess, tokenizer = _get_clip()
    image = preprocess(img).unsqueeze(0).cuda().half()
    text = tokenizer(PATTERN_PROMPTS_CLIP).cuda()

    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    sims = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    probs = sims.squeeze(0).float().cpu().numpy()

    return PATTERN_LABELS[probs.argmax()]

# ---------- Silhouette classification ----------
SILHOUETTE_PROMPTS_CLIP = [
    "a photo of a fitted tight garment showing body shape",
    "a photo of a loose baggy garment",
    "a photo of an a-line silhouette garment",
    "a photo of a bodycon dress hugging the body",
    "a photo of an oversized loose garment",
    "a photo of a straight cut garment",
    "a photo of a flared skirt garment",
    "a photo of a peplum top with a ruffle at waist",
]

SILHOUETTE_LABELS = [
    "fitted", "loose", "a-line", "bodycon",
    "oversized", "straight", "flared", "peplum",
]


@torch.no_grad()
def classify_silhouette(img: Image.Image) -> str:
    model, preprocess, tokenizer = _get_clip()
    image = preprocess(img).unsqueeze(0).cuda().half()
    text = tokenizer(SILHOUETTE_PROMPTS_CLIP).cuda()

    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    sims = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    probs = sims.squeeze(0).float().cpu().numpy()

    return SILHOUETTE_LABELS[probs.argmax()]