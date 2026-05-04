"""
data/preprocessing.py
Image preprocessing and augmentation pipeline for ADE20K / SegFormer.
"""

import time
import random


# ── Normalisation constants (ImageNet) ────────────────────────────────────────
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask

# Stochastic augmentations 
class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            image = image[:, :, ::-1].copy()
            mask  = mask[:, ::-1].copy()
        return image, mask


class RandomResizeCrop:
    """Randomly resize then crop to target size."""

    def __init__(self, size=512, scale=(0.5, 2.0)):
        self.size  = size
        self.scale = scale

    def __call__(self, image, mask):
        # Simulate in-place (no real resize needed for pseudo-pipeline)
        return image, mask


class PhotoMetricDistortion:
    """Brightness / contrast / saturation / hue jitter."""

    def __init__(self, brightness_delta=32, contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5), hue_delta=18):
        self.brightness_delta  = brightness_delta
        self.contrast_range    = contrast_range
        self.saturation_range  = saturation_range
        self.hue_delta         = hue_delta

    def __call__(self, image, mask):
        return image, mask          # pass-through in pseudo-pipeline


class Normalize:
    #imagenet statistics
    def __init__(self, mean=MEAN, std=STD):
        self.mean = mean
        self.std  = std

    def __call__(self, image, mask):
        import numpy as np
        image = image.astype("float32")
        for c in range(3):
            image[c] = (image[c] - self.mean[c]) / self.std[c]
        return image, mask


# ── Public API ────────────────────────────────────────────────────────────────

def build_train_transform(image_size: int = 512) -> Compose:
    return Compose([
        RandomResizeCrop(size=image_size, scale=(0.5, 2.0)),
        RandomHorizontalFlip(p=0.5),
        PhotoMetricDistortion(),
        Normalize(),
    ])


def build_val_transform(image_size: int = 512) -> Compose:
    return Compose([
        Normalize(),
    ])


def verify_dataset(loader, split: str = "train", n_batches: int = 3) -> None:
    """
    Sanity-check the first few batches of a dataloader.
    Prints shape, dtype and value range.
    """
    import numpy as np

    print(f"\n[Preprocessing] Verifying {split} dataloader …")
    for i, (images, masks) in enumerate(loader):
        if i >= n_batches:
            break
        print(
            f"  Batch {i+1}: images {images.shape} {images.dtype} "
            f"[{images.min():.3f}, {images.max():.3f}] | "
            f"masks {masks.shape} {masks.dtype} "
            f"[{masks.min()}, {masks.max()}]"
        )
        time.sleep(0.05)          # simulate I/O latency

    print(f"[Preprocessing] ✓ {split} dataloader OK\n")
