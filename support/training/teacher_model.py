"""
models/teacher_model.py

Teacher Model: CLIPSeg (CIDAS)
Used for generating high-quality pseudo-labels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation


class TeacherModel(nn.Module):
    """
    CLIPSeg-based Teacher Model

    Responsibilities:
    - Load pretrained CLIPSeg
    - Generate segmentation masks from text prompts
    - Provide soft labels for distillation
    """

    def __init__(self, device="cuda"):
        super().__init__()

        self.device = device

        # Load CLIPSeg model (CIDAS)
        self.processor = CLIPSegProcessor.from_pretrained(
            "CIDAS/clipseg-rd64-refined"
        )

        self.model = CLIPSegForImageSegmentation.from_pretrained(
            "CIDAS/clipseg-rd64-refined"
        )

        self.model.to(self.device)
        self.model.eval()  # Teacher is always frozen

    # ─────────────────────────────────────────────
    # Forward Pass
    # ─────────────────────────────────────────────
    def forward(self, images, text_prompts):
        """
        Generate segmentation masks

        Args:
            images: list of PIL images OR tensor batch
            text_prompts: list of strings

        Returns:
            masks: tensor [B, 1, H, W]
        """

        inputs = self.processor(
            text=text_prompts,
            images=images,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits  # [B, H, W]

        masks = torch.sigmoid(logits)

        return masks.unsqueeze(1)  # [B, 1, H, W]

    # ─────────────────────────────────────────────
    # Generate Pseudo Labels
    # ─────────────────────────────────────────────
    def generate_pseudo_labels(self, images, prompts, threshold=0.5):
        """
        Convert soft masks → hard pseudo labels
        """

        masks = self.forward(images, prompts)

        pseudo_labels = (masks > threshold).float()

        return pseudo_labels

    # ─────────────────────────────────────────────
    # Confidence Map (for filtering bad labels)
    # ─────────────────────────────────────────────
    def get_confidence(self, masks):
        """
        Compute confidence scores for each mask
        """

        # Confidence = distance from 0.5
        confidence = torch.abs(masks - 0.5) * 2

        return confidence

    # ─────────────────────────────────────────────
    # Freeze Teacher
    # ─────────────────────────────────────────────
    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    # ─────────────────────────────────────────────
    # Utility: Resize output
    # ─────────────────────────────────────────────
    def resize_masks(self, masks, size):
        """
        Resize masks to match student resolution
        """

        return F.interpolate(
            masks,
            size=size,
            mode="bilinear",
            align_corners=False
        )