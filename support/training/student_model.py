"""
models/student_model.py

Student Model: Lightweight SegFormer
Learns from Teacher (CLIPSeg) via distillation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# Simple Encoder (feature extractor)
# ─────────────────────────────────────────────
class SimpleEncoder(nn.Module):
    def __init__(self, in_channels=3, embed_dim=64):
        super().__init__()

        # First conv block (downsample)
        self.conv1 = nn.Conv2d(in_channels, embed_dim, 3, stride=2, padding=1)
        self.bn1   = nn.BatchNorm2d(embed_dim)

        # Second conv block (more features)
        self.conv2 = nn.Conv2d(embed_dim, embed_dim * 2, 3, stride=2, padding=1)
        self.bn2   = nn.BatchNorm2d(embed_dim * 2)

        # Third conv block (deep features)
        self.conv3 = nn.Conv2d(embed_dim * 2, embed_dim * 4, 3, stride=2, padding=1)
        self.bn3   = nn.BatchNorm2d(embed_dim * 4)

    def forward(self, x):
        # pass through 3 layers
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x3 = F.relu(self.bn3(self.conv3(x2)))

        # return multi-level features
        return [x1, x2, x3]


# ─────────────────────────────────────────────
# Decoder (combine features)
# ─────────────────────────────────────────────
class SegFormerHead(nn.Module):
    def __init__(self, embed_dims, num_classes):
        super().__init__()

        # reduce channels
        self.linear_c1 = nn.Conv2d(embed_dims[0], 256, 1)
        self.linear_c2 = nn.Conv2d(embed_dims[1], 256, 1)
        self.linear_c3 = nn.Conv2d(embed_dims[2], 256, 1)

        # merge features
        self.fuse = nn.Conv2d(256 * 3, 256, 1)

        # final output layer
        self.classifier = nn.Conv2d(256, num_classes, 1)

    def forward(self, features):
        x1, x2, x3 = features

        # resize all to same size
        x1 = self.linear_c1(x1)
        x2 = F.interpolate(self.linear_c2(x2), size=x1.shape[2:], mode="bilinear")
        x3 = F.interpolate(self.linear_c3(x3), size=x1.shape[2:], mode="bilinear")

        # combine features
        x = torch.cat([x1, x2, x3], dim=1)

        # fuse + predict
        x = self.fuse(x)
        x = self.classifier(x)

        return x


# ─────────────────────────────────────────────
# Student Model (main model)
# ─────────────────────────────────────────────
class StudentModel(nn.Module):
    """
    Student model learns from:
    - Real labels
    - Teacher model (distillation)
    """

    def __init__(self, num_classes=1):
        super().__init__()

        # encoder + decoder
        self.encoder = SimpleEncoder()
        self.decoder = SegFormerHead(
            embed_dims=[64, 128, 256],
            num_classes=num_classes
        )

    # ─────────────────────────────────────────
    # Forward pass
    # ─────────────────────────────────────────
    def forward(self, x):
        features = self.encoder(x)   # get features
        logits   = self.decoder(features)  # predict mask

        # resize to input size
        logits = F.interpolate(
            logits,
            size=x.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        return logits

    # ─────────────────────────────────────────
    # Loss function
    # ─────────────────────────────────────────
    def compute_loss(self, preds, targets, teacher_preds=None, alpha=0.7):
        """
        Mix of:
        - real label loss
        - teacher loss
        """

        # main loss (ground truth)
        bce_loss = F.binary_cross_entropy_with_logits(preds, targets)

        if teacher_preds is not None:
            # stop teacher gradient
            teacher_preds = teacher_preds.detach()

            # match teacher output
            distill_loss = F.mse_loss(
                torch.sigmoid(preds),
                teacher_preds
            )

            # combine both losses
            total_loss = alpha * bce_loss + (1 - alpha) * distill_loss
        else:
            total_loss = bce_loss

        return total_loss

    # ─────────────────────────────────────────
    # Prediction
    # ─────────────────────────────────────────
    def predict(self, x, threshold=0.5):
        logits = self.forward(x)      # raw output
        probs  = torch.sigmoid(logits)  # convert to probability

        return (probs > threshold).float()  # binary mask

    # ─────────────────────────────────────────
    # Metrics (accuracy + IoU)
    # ─────────────────────────────────────────
    def compute_metrics(self, preds, targets):
        preds = torch.sigmoid(preds)

        preds_bin = (preds > 0.5).float()

        # overlap area
        intersection = (preds_bin * targets).sum()

        # total area
        union = preds_bin.sum() + targets.sum()

        # IoU score
        iou = intersection / (union - intersection + 1e-6)

        # pixel accuracy
        acc = (preds_bin == targets).float().mean()

        return {
            "accuracy": acc.item(),
            "miou": iou.item()
        }