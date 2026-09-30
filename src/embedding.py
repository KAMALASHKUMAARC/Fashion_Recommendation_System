import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import torch
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoProcessor

from config import (
    MODEL_ID, DEVICE, DTYPE, CROP_REGIONS,
    BASE_EMBEDDING_DIM, EMBEDDING_DIM,
)
from src.preprocess import get_transform, crop_region, load_image


def _extract_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "text_embeds") and output.text_embeds is not None:
        return output.text_embeds
    if hasattr(output, "image_embeds") and output.image_embeds is not None:
        return output.image_embeds
    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        return output.last_hidden_state
    if isinstance(output, (tuple, list)) and len(output) > 0:
        return output[0]
    raise RuntimeError(f"Cannot extract tensor from {type(output)}")


class SigLIP2Encoder:
    def __init__(self, model_id: str = MODEL_ID, device: str = DEVICE):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available.")

        self.device = torch.device("cuda:0")
        self.dtype = torch.float16 if DTYPE == "float16" else torch.float32

        print(f"[SigLIP2Encoder] Loading processor from {model_id}")
        self.processor = AutoProcessor.from_pretrained(model_id)

        print(f"[SigLIP2Encoder] Loading model (dtype={self.dtype}, device_map=cuda:0)")
        self.model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            device_map={"": "cuda:0"},
        )
        self.model.eval()

        param_devices = {p.device for p in self.model.parameters()}
        print(f"[SigLIP2Encoder] Parameter devices: {param_devices}")
        if not all(d.type == "cuda" for d in param_devices):
            raise RuntimeError(f"Not all params on CUDA. Found: {param_devices}")

        if hasattr(self.model, "logit_scale"):
            self.logit_scale = float(self.model.logit_scale.exp().item())
        else:
            self.logit_scale = 100.0
        if hasattr(self.model, "logit_bias") and self.model.logit_bias is not None:
            self.logit_bias = float(self.model.logit_bias.item())
        else:
            self.logit_bias = 0.0
        print(f"[SigLIP2Encoder] logit_scale={self.logit_scale:.2f}, logit_bias={self.logit_bias:.4f}")

        dummy = torch.randn(1, 3, 512, 512, device=self.device, dtype=self.dtype)
        with torch.no_grad():
            _ = self.model.get_image_features(pixel_values=dummy)
        torch.cuda.synchronize()
        mem = torch.cuda.memory_allocated() / 1e9
        print(f"[SigLIP2Encoder] Test pass OK. GPU memory: {mem:.2f} GB")

        self.transform = get_transform()
        self._text_cache = {}
        self._text_cache_stats = {"hits": 0, "misses": 0}

        print(f"[SigLIP2Encoder] Ready on {self.device}")

    @torch.no_grad()
    def encode_image_full(self, img: Image.Image) -> np.ndarray:
        inputs = self.processor(images=img, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(self.device).to(self.dtype)

        out = self.model.get_image_features(pixel_values=pixel_values)
        feats = _extract_tensor(out)

        if feats.device.type != "cuda":
            raise RuntimeError(f"Image features on {feats.device}, expected cuda")
        if feats.dim() == 3:
            feats = feats[:, 0, :]

        vec = feats[0].float().cpu().numpy()
        return vec / (np.linalg.norm(vec) + 1e-8)

    @torch.no_grad()
    def encode_image_multicrop(self, img: Image.Image) -> np.ndarray:
        vecs = []
        for name, region in CROP_REGIONS.items():
            cropped = crop_region(img, *region)
            vecs.append(self.encode_image_full(cropped))
        vec = np.concatenate(vecs, axis=0)
        return vec / (np.linalg.norm(vec) + 1e-8)

    @torch.no_grad()
    def encode_batch_multicrop(self, images: list) -> np.ndarray:
        out = []
        for img in images:
            out.append(self.encode_image_multicrop(img))
        return np.stack(out, axis=0)

    @torch.no_grad()
    def encode_text(self, texts: list) -> np.ndarray:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        out = self.model.get_text_features(**inputs)
        feats = _extract_tensor(out)
        if feats.device.type != "cuda":
            raise RuntimeError(f"Text features on {feats.device}, expected cuda")
        if feats.dim() == 3:
            feats = feats.mean(dim=1)

        vecs = feats.float().cpu().numpy()
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
        return vecs / norms

    @torch.no_grad()
    def encode_text_cached(self, texts: tuple) -> np.ndarray:
        if texts in self._text_cache:
            self._text_cache_stats["hits"] += 1
            return self._text_cache[texts]
        self._text_cache_stats["misses"] += 1
        vecs = self.encode_text(list(texts))
        self._text_cache[texts] = vecs
        return vecs

    @torch.no_grad()
    def zero_shot_classify(self, img: Image.Image, prompts: dict) -> dict:
        """SigLIP-2 zero-shot (kept for pattern/silhouette)."""
        img_vec = self.encode_image_full(img)
        text_vecs = self.encode_text_cached(tuple(prompts.values()))
        sims = text_vecs @ img_vec
        sims = sims * 100.0
        probs = np.exp(sims - sims.max())
        probs = probs / probs.sum()
        return {label: float(p) for label, p in zip(prompts.keys(), probs)}

    @torch.no_grad()
    def category_ensemble(self, img, categories, templates=None):
        """Deprecated. Use src.clip_classifier.classify_category instead."""
        from src.clip_classifier import classify_category
        return classify_category(img, categories)

    def cache_stats(self) -> dict:
        return dict(self._text_cache_stats)

    @torch.no_grad()
    def category_from_text(self, text: str, categories: list) -> dict:
        """Classify a text query against category names using CLIP."""
        from src.clip_classifier import _get_clip

        model, _, tokenizer = _get_clip()
        text_input = tokenizer([text]).cuda()
        cat_inputs = tokenizer([f"a photo of a {c}" for c in categories]).cuda()

        text_features = model.encode_text(text_input)
        cat_features = model.encode_text(cat_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        cat_features = cat_features / cat_features.norm(dim=-1, keepdim=True)

        sims = (100.0 * text_features @ cat_features.T).softmax(dim=-1)
        probs = sims.squeeze(0).float().cpu().numpy()

        ranked = sorted(zip(categories, probs), key=lambda x: -x[1])
        return {"top5": ranked[:5], "all": dict(ranked)}