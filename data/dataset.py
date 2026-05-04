"""
data/dataset.py
ADE20K dataset loader for SegFormer semantic segmentation.
"""

import os
import time
import random
import json
from pathlib import Path


class ADE20KDataset:
    """ADE20K semantic segmentation dataset wrapper."""

    SPLIT_SIZES = {"train": 20210, "val": 2000, "test": 3352}

    def __init__(self, root: str, split: str = "train", transform=None):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.num_classes = 150

        # Simulate index loading
        self._samples = self._load_index()

    def _load_index(self):
        """Build sample index from dataset root."""
        n = self.SPLIT_SIZES[self.split]
        # Simulate file-path index (no real files needed for pseudo-pipeline)
        return [
            {
                "image": str(self.root / "images" / self.split / f"ADE_{self.split}_{i:08d}.jpg"),
                "mask":  str(self.root / "unannotations" / self.split / f"ADE_{self.split}_{i:08d}.png"),
            }
            for i in range(1, n + 1)
        ]

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        """Return a (image, mask) pair — simulated tensors."""
        import numpy as np
        sample = self._samples[idx]
        # Pseudo image/mask — shape matches SegFormer expectations
        image = np.random.rand(3, 512, 512).astype("float32")
        mask  = np.random.randint(0, self.num_classes, (512, 512), dtype="int64")
        return image, mask

    def __repr__(self):
        return (
            f"ADE20KDataset(split={self.split!r}, "
            f"num_samples={len(self)}, num_classes={self.num_classes})"
        )


class DataLoader:
    """Minimal dataloader wrapper that yields batched pseudo-samples."""

    def __init__(self, dataset, batch_size: int = 8, shuffle: bool = True,
                 num_workers: int = 4, drop_last: bool = True):
        self.dataset    = dataset
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.num_workers = num_workers
        self.drop_last  = drop_last
        self._n_batches = len(dataset) // batch_size

    def __len__(self):
        return self._n_batches

    def __iter__(self):
        import numpy as np
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)

        for start in range(0, len(indices) - self.batch_size + 1, self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            images = np.stack([self.dataset[i][0] for i in batch_idx])
            masks  = np.stack([self.dataset[i][1] for i in batch_idx])
            yield images, masks


def build_dataloaders(config: dict):
    """Build train/val dataloaders from config."""
    root       = config["dataset"]["root"]
    batch_size = config["dataset"]["batch_size"]
    n_workers  = config["dataset"]["num_workers"]

    train_ds = ADE20KDataset(root, split="train")
    val_ds   = ADE20KDataset(root, split="val")

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=n_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=n_workers)

    return train_loader, val_loader
