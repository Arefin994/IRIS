"""
training/trainer.py
Main training loop for SegFormer semantic segmentation.
"""

import time
import json
import os
import random
from pathlib import Path

from training.metrics import MetricEngine


class Trainer:
    """
    Orchestrates the full training pipeline:
      - Per-step forward / backward pass (simulated)
      - Metric logging every N steps
      - Epoch-level summary + validation
      - Checkpoint saving
    """

    def __init__(self, model, train_loader, val_loader, config: dict, logger):
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.config       = config
        self.logger       = logger

        self.epochs          = config["training"]["epochs"]
        self.total_steps     = config["training"]["total_steps"]
        self.steps_per_epoch = config["training"]["steps_per_epoch"]
        self.log_every       = config["logging"]["log_every_n_steps"]
        self.checkpoint_dir  = Path(config["logging"]["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.metrics = MetricEngine(config)
        self._history = []          # full step log
        self._epoch_summaries = []  # per-epoch summary

    # ── Public entry point ────────────────────────────────────────────────────

    def train(self) -> list:
        """Run full training. Returns epoch summaries."""
        self.logger.info("=" * 64)
        self.logger.info("Starting SegFormer Training")
        self.logger.info(f"  Backbone   : {self.config['model']['variant']}")
        self.logger.info(f"  Dataset    : {self.config['dataset']['name']}")
        self.logger.info(f"  Epochs     : {self.epochs}")
        self.logger.info(f"  Steps/epoch: {self.steps_per_epoch}")
        self.logger.info(f"  Total steps: {self.total_steps}")
        self.logger.info(f"  Optimizer  : {self.config['training']['optimizer']}")
        self.logger.info(f"  Base LR    : {self.config['training']['base_lr']}")
        self.logger.info("=" * 64)
        print()

        t_start = time.time()

        for epoch in range(1, self.epochs + 1):
            self._run_epoch(epoch)

        elapsed = time.time() - t_start
        self._print_final_summary(elapsed)
        self._save_log()

        return self._epoch_summaries

    # ── Epoch loop ────────────────────────────────────────────────────────────

    def _run_epoch(self, epoch: int) -> None:
        try:
            from tqdm import tqdm
            bar = tqdm(
                total=self.steps_per_epoch,
                desc=f"Epoch {epoch}/{self.epochs}",
                unit="step",
                bar_format=(
                    "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                    "[{elapsed}<{remaining}, {rate_fmt}]"
                ),
                colour="cyan",
            )
        except ImportError:
            bar = None

        global_step_base = (epoch - 1) * self.steps_per_epoch

        for local_step in range(1, self.steps_per_epoch + 1):
            global_step = global_step_base + local_step

            # ── Simulate forward + backward pass latency ──────────────────
            time.sleep(random.uniform(0.003, 0.006))

            m = self.metrics.compute(global_step)
            m["step"]  = global_step
            m["epoch"] = epoch
            self._history.append(m)

            # ── Update tqdm postfix every step ────────────────────────────
            if bar:
                bar.set_postfix(
                    loss=f"{m['loss']:.4f}",
                    acc=f"{m['accuracy']:.4f}",
                    f1=f"{m['f1']:.4f}",
                    refresh=False,
                )
                bar.update(1)

            # ── Verbose step log every N steps ────────────────────────────
            if global_step % self.log_every == 0:
                lr = self._get_lr(global_step)
                msg = (
                    f"[Epoch {epoch}/{self.epochs}] "
                    f"Step {global_step:05d}/{self.total_steps} | "
                    f"Loss: {m['loss']:.4f} | "
                    f"Acc: {m['accuracy']:.4f} | "
                    f"Prec: {m['precision']:.4f} | "
                    f"Rec: {m['recall']:.4f} | "
                    f"F1: {m['f1']:.4f} | "
                    f"mIoU: {m['miou']:.4f} | "
                    f"LR: {lr:.2e}"
                )
                if bar:
                    bar.write(msg)
                else:
                    print(msg)
                self.logger.info(msg)

        if bar:
            bar.close()

        # ── Epoch summary + validation ─────────────────────────────────────
        self._run_validation(epoch)

    # ── Validation pass ───────────────────────────────────────────────────────

    def _run_validation(self, epoch: int) -> None:
        print(f"\n  [Epoch {epoch}] Running validation …")
        time.sleep(random.uniform(0.3, 0.6))       # simulate val pass latency

        summary = self.metrics.epoch_summary(epoch, self.epochs)
        self._epoch_summaries.append(summary)

        lines = [
            f"\n{'─'*64}",
            f"  Epoch {epoch}/{self.epochs} Summary",
            f"{'─'*64}",
            f"  Train  →  Loss: {summary['train_loss']:.4f}  "
            f"Acc: {summary['train_acc']:.4f}  "
            f"F1: {summary['train_f1']:.4f}  "
            f"mIoU: {summary['train_miou']:.4f}",
            f"  Val    →  Loss: {summary['val_loss']:.4f}  "
            f"Acc: {summary['val_acc']:.4f}  "
            f"mIoU: {summary['val_miou']:.4f}",
            f"{'─'*64}\n",
        ]
        for l in lines:
            print(l)
            self.logger.info(l)

        self._save_checkpoint(epoch, summary)

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, summary: dict) -> None:
        ckpt = {
            "epoch":       epoch,
            "architecture": self.config["model"]["variant"],
            "num_classes":  self.config["model"]["num_classes"],
            "metrics":      summary,
        }
        path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:02d}.json"
        path.write_text(json.dumps(ckpt, indent=2))
        self.logger.info(f"  Checkpoint saved → {path}")

    # ── Final summary ─────────────────────────────────────────────────────────

    def _print_final_summary(self, elapsed: float) -> None:
        final = self._epoch_summaries[-1]
        lines = [
            "\n" + "=" * 64,
            "  TRAINING COMPLETE",
            "=" * 64,
            f"  Total time       : {elapsed:.1f}s  "
            f"({elapsed/self.total_steps*1000:.1f} ms/step)",
            f"  Final Train Loss : {final['train_loss']:.4f}",
            f"  Final Train Acc  : {final['train_acc']:.4f}",
            f"  Final Train F1   : {final['train_f1']:.4f}",
            f"  Final Train mIoU : {final['train_miou']:.4f}",
            f"  Final Val Loss   : {final['val_loss']:.4f}",
            f"  Final Val Acc    : {final['val_acc']:.4f}",
            f"  Final Val mIoU   : {final['val_miou']:.4f}",
            "=" * 64 + "\n",
        ]
        for l in lines:
            print(l)
            self.logger.info(l)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_lr(self, step: int) -> float:
        """Poly LR schedule with linear warmup."""
        cfg      = self.config["training"]
        base_lr  = cfg["base_lr"]
        warmup   = cfg["warmup_steps"]
        total    = cfg["total_steps"]
        power    = cfg["poly_power"]

        if step < warmup:
            return base_lr * step / warmup
        progress = (step - warmup) / (total - warmup)
        return base_lr * ((1 - progress) ** power)

    def _save_log(self) -> None:
        log_dir = Path(self.config["logging"]["log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / "training_log.json"
        path.write_text(json.dumps({
            "config":          self.config["training"],
            "epoch_summaries": self._epoch_summaries,
            "step_history":    self._history[::10],   # every 10th step to keep file small
        }, indent=2))
        self.logger.info(f"  Training log saved → {path}")
