import numpy as np
from PIL import Image

from config import PATTERNS, SILHOUETTES
from src.clip_classifier import classify_color, classify_pattern, classify_silhouette


def predict_pattern(encoder, img: Image.Image) -> str:
    return classify_pattern(img)


def predict_silhouette(encoder, img: Image.Image) -> str:
    return classify_silhouette(img)


def enrich_attributes(encoder, img: Image.Image) -> dict:
    return {
        "color":      classify_color(img),
        "pattern":    classify_pattern(img),
        "silhouette": classify_silhouette(img),
    }