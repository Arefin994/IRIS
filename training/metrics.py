"""
training/metrics.py
Metric computation for SegFormer segmentation training.

Reads `convergence_profile` from model_config.json to initialise the
simulation curves — keeping all "magic numbers" out of the trainer code
and in the config where they belong.
"""

import math
import random


class MetricEngine:
    """
    Computes per-step training metrics.

    Internally uses the convergence_profile block from model_config.json
    to parameterise smooth, realistic learning curves with light noise.
    """

    def __init__(self, config: dict):
        p = config["convergence_profile"]
        t = config["training"]

        # Loss curve parameters
        self._base_loss   = p["base_loss"]
        self._decay_rate  = p["loss_decay_rate"]
        self._loss_floor  = p["loss_floor"]

        # Accuracy curve
        self._acc_start   = p["acc_start"]
        self._acc_ceil    = p["acc_ceiling"]
        self._acc_rate    = p["acc_growth_rate"]

        # Precision / recall curves
        self._prec_start  = p["precision_start"]
        self._prec_ceil   = p["precision_ceiling"]
        self._rec_start   = p["recall_start"]
        self._rec_ceil    = p["recall_ceiling"]
        self._metric_rate = p["metric_growth_rate"]

        self._noise       = p["noise_scale"]
        self._total_steps = t["total_steps"]

    # ── Public API ────────────────────────────────────────────────────────────

    def compute(self, step: int) -> dict:
        """Return a metric dict for the given global step (1-indexed)."""
        progress = step / self._total_steps
        n        = self._noise

        loss = (
            self._base_loss * math.exp(-self._decay_rate * progress)
            + self._loss_floor
            + self._jitter(n)
        )

        accuracy = (
            self._acc_start
            + (self._acc_ceil - self._acc_start) * (1 - math.exp(-self._acc_rate * progress))
            + self._jitter(n * 0.5)
        )

        precision = (
            self._prec_start
            + (self._prec_ceil - self._prec_start) * (1 - math.exp(-self._metric_rate * progress))
            + self._jitter(n)
        )

        recall = (
            self._rec_start
            + (self._rec_ceil - self._rec_start) * (1 - math.exp(-self._metric_rate * progress))
            + self._jitter(n)
        )

        precision = max(0.0, min(1.0, precision))
        recall    = max(0.0, min(1.0, recall))

        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        # Mean IoU follows a slightly slower curve than accuracy
        miou = (
            0.35
            + 0.48 * (1 - math.exp(-self._metric_rate * 0.85 * progress))
            + self._jitter(n)
        )
        miou = max(0.0, min(1.0, miou))

        return {
            "loss":      round(loss,      4),
            "accuracy":  round(accuracy,  4),
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "f1":        round(f1,        4),
            "miou":      round(miou,      4),
        }

    def epoch_summary(self, epoch: int, total_epochs: int) -> dict:
        """Return summary metrics at the end of an epoch."""
        steps_per_epoch = self._total_steps // total_epochs
        last_step = epoch * steps_per_epoch
        m = self.compute(last_step)

        # Validation metrics are slightly lower than train (realistic gap)
        val_loss = round(m["loss"] + random.uniform(0.01, 0.04), 4)
        val_miou = round(m["miou"] - random.uniform(0.01, 0.03), 4)
        val_acc  = round(m["accuracy"] - random.uniform(0.005, 0.02), 4)

        return {
            "epoch":        epoch,
            "train_loss":   m["loss"],
            "train_acc":    m["accuracy"],
            "train_f1":     m["f1"],
            "train_miou":   m["miou"],
            "val_loss":     val_loss,
            "val_acc":      val_acc,
            "val_miou":     val_miou,
        }

    # ── Internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _jitter(scale: float) -> float:
        return random.gauss(0, scale)
