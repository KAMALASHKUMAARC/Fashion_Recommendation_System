from PIL import Image
import numpy as np
import torch
from torchvision import transforms

from config import IMAGE_SIZE

#forcing image to open in rgb format
def load_image(path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img

#image resizing, layout from hwc to chw, datatype change and normalization
def get_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

#used to crop image
def crop_region(img: Image.Image, y1_frac: float, y2_frac: float) -> Image.Image:
    w, h = img.size
    y1 = int(h * y1_frac)
    y2 = int(h * y2_frac)
    return img.crop((0, y1, w, y2))

#does all above function and add new dimension in front so that we can use batch processing
def preprocess_image(path_or_img, region: tuple = None) -> torch.Tensor:
    img = load_image(path_or_img) if isinstance(path_or_img, (str, bytes)) else path_or_img
    if region is not None:
        img = crop_region(img, *region)
    return get_transform()(img).unsqueeze(0)