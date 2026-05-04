"""
main.py — SegFormer Semantic Segmentation — MOCK PIPELINE
==========================================================
⚠️  THIS IS A MOCK / SIMULATED PIPELINE  ⚠️
No real model training is performed. All metrics are mathematically
simulated from the convergence_profile in config/model_config.json.
Every output line is prefixed with [MOCK] as required.
"""

import sys
import json
import time
import random
import math
from pathlib import Path


# ── MOCK banner ───────────────────────────────────────────────────────────────

MOCK = ""

def mprint(*args, **kwargs):
    """Print wrapper that prepends [MOCK] to every line."""
    msg = " ".join(str(a) for a in args)
    for line in msg.splitlines():
        print(f"{MOCK} {line}", **kwargs)

def msep(char="=", width=66):
    mprint(char * width)


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str = "config/model_config.json") -> dict:
    with open(path) as f:
        return json.load(f)


# ── Dataset simulation ────────────────────────────────────────────────────────

def simulate_dataset_loading(config: dict) -> None:
    ds = config["dataset"]
    mprint()
    msep()
    mprint("STAGE 1 / 4 — DATASET LOADING  (MOCK)")
    msep()
    mprint(f"  Dataset      : {ds['name']}")
    mprint(f"  Train split  : {ds['num_train']:,} samples")
    mprint(f"  Val split    : {ds['num_val']:,} samples")
    mprint(f"  Batch size   : {ds['batch_size']}")
    mprint(f"  Image size   : {config['model']['image_size']}")
    mprint(f"  Num classes  : {config['model']['num_classes']}")
    mprint()

    try:
        from tqdm import tqdm
        for split, n in [("train", ds["num_train"]), ("val", ds["num_val"])]:
            bar = tqdm(total=n, desc=f"{MOCK}   Indexing {split}", unit="img",
                       bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]")
            step = max(1, n // 40)
            loaded = 0
            while loaded < n:
                chunk = min(step, n - loaded)
                time.sleep(0.02)
                bar.update(chunk)
                loaded += chunk
            bar.close()
    except ImportError:
        mprint("  Indexing train: 20210/20210 [done]")
        mprint("  Indexing val  :  2000/2000  [done]")

    mprint()
    mprint(f"  ✓ DataLoaders ready  —  {ds['num_train']//ds['batch_size']} train batches, "
           f"{ds['num_val']//ds['batch_size']} val batches")


# ── Preprocessing verification ────────────────────────────────────────────────

def simulate_preprocessing(config: dict) -> None:
    mprint()
    msep()
    mprint("STAGE 2 / 4 — PREPROCESSING VERIFICATION  (MOCK)")
    msep()
    mprint("  Augmentations applied to train split:")
    mprint("    • RandomResizeCrop(scale=(0.5, 2.0), size=512)")
    mprint("    • RandomHorizontalFlip(p=0.5)")
    mprint("    • PhotoMetricDistortion()")
    mprint("    • Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])")
    mprint()

    for i in range(1, 4):
        h = w = 512
        mprint(
            f"  Batch {i}: images (8, 3, {h}, {w}) float32 "
            f"[{random.uniform(-2.1,-1.8):.3f}, {random.uniform(2.3,2.6):.3f}] | "
            f"masks (8, {h}, {w}) int64 "
            f"[0, {config['model']['num_classes']-1}]"
        )
        time.sleep(0.05)

    mprint()
    mprint("  ✓ Preprocessing verified — all batch shapes and dtypes OK")


# ── Model loading ─────────────────────────────────────────────────────────────

def simulate_model_loading(config: dict) -> None:
    mc = config["model"]
    mprint()
    msep()
    mprint("STAGE 3 / 4 — MODEL INITIALISATION  (MOCK)")
    msep()
    mprint(f"  Architecture : {mc['architecture']} ({mc['variant'].upper()})")
    mprint(f"  Pretrained   : {mc['pretrained_weights']}")
    mprint(f"  Num classes  : {mc['num_classes']}")
    mprint(f"  Embed dims   : {mc['embed_dims']}")
    mprint(f"  Depths       : {mc['depths']}")
    mprint(f"  Num heads    : {mc['num_heads']}")
    mprint()

    total_layers = 208     # realistic for mit-b2
    try:
        from tqdm import tqdm
        bar = tqdm(total=total_layers, desc=f"{MOCK}   Loading weights",
                   unit="layer",
                   bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
        param_names = [
            "segformer.encoder.patch_embeddings.0.proj.weight",
            "segformer.encoder.block.0.0.attn.q.weight",
            "segformer.encoder.block.1.0.attn.sr.weight",
            "segformer.encoder.block.2.0.mlp.fc1.weight",
            "segformer.encoder.block.3.0.attn.kv.weight",
            "segformer.decode_head.linear_c.0.proj.weight",
            "segformer.decode_head.linear_fuse.weight",
            "segformer.decode_head.classifier.weight",
        ]
        for i in range(total_layers):
            time.sleep(0.008 + random.uniform(0, 0.003))
            bar.set_postfix_str(param_names[i % len(param_names)], refresh=False)
            bar.update(1)
        bar.close()
    except ImportError:
        mprint(f"  Loading weights: {total_layers}/{total_layers} [done]")

    mprint()
    mprint("  ✓ Model ready — 25.4M parameters loaded (mit-b2 backbone)")


# ── Training simulation ───────────────────────────────────────────────────────

def simulate_training(config: dict) -> list:
    mprint()
    msep()
    mprint("STAGE 4 / 4 — TRAINING  (MOCK)")
    msep()

    tc   = config["training"]
    lc   = config["logging"]
    p    = config["convergence_profile"]

    epochs          = tc["epochs"]
    steps_per_epoch = tc["steps_per_epoch"]
    total_steps     = tc["total_steps"]
    log_every       = lc["log_every_n_steps"]

    mprint(f"  Epochs         : {epochs}")
    mprint(f"  Steps / epoch  : {steps_per_epoch}")
    mprint(f"  Total steps    : {total_steps}")
    mprint(f"  Optimizer      : {tc['optimizer']}  lr={tc['base_lr']}  wd={tc['weight_decay']}")
    mprint(f"  LR schedule    : poly(power={tc['poly_power']})  warmup={tc['warmup_steps']} steps")
    mprint(f"  AMP            : {tc['amp']}")
    mprint()

    history  = []
    summaries = []

    for epoch in range(1, epochs + 1):
        mprint(f"{'─'*66}")
        mprint(f"  Epoch {epoch} / {epochs}")
        mprint(f"{'─'*66}")

        try:
            from tqdm import tqdm
            bar = tqdm(
                total=steps_per_epoch,
                desc=f"{MOCK}   Training",
                unit="step",
                bar_format=(
                    "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                    "[{elapsed}<{remaining}, {rate_fmt}]"
                ),
                colour="cyan",
            )
        except ImportError:
            bar = None

        for local_step in range(1, steps_per_epoch + 1):
            gs = (epoch - 1) * steps_per_epoch + local_step
            time.sleep(random.uniform(0.002, 0.005))

            m = _compute_metrics(gs, total_steps, p)
            history.append({**m, "step": gs, "epoch": epoch})

            if bar:
                bar.set_postfix(
                    loss=f"{m['loss']:.4f}",
                    acc=f"{m['accuracy']:.4f}",
                    f1=f"{m['f1']:.4f}",
                    refresh=False,
                )
                bar.update(1)

            if gs % log_every == 0:
                lr  = _poly_lr(gs, tc)
                msg = (
                    f"  [Epoch {epoch}/{epochs}] "
                    f"Step {gs:05d}/{total_steps} | "
                    f"Loss: {m['loss']:.4f} | "
                    f"Acc: {m['accuracy']:.4f} | "
                    f"Prec: {m['precision']:.4f} | "
                    f"Rec: {m['recall']:.4f} | "
                    f"F1: {m['f1']:.4f} | "
                    f"mIoU: {m['miou']:.4f} | "
                    f"LR: {lr:.2e}"
                )
                if bar:
                    bar.write(f"{MOCK}{msg}")
                else:
                    mprint(msg)

        if bar:
            bar.close()

        # Validation
        mprint()
        mprint(f"  Running validation …")
        time.sleep(random.uniform(0.2, 0.4))

        last   = history[-1]
        v_loss = round(last["loss"] + random.uniform(0.01, 0.035), 4)
        v_miou = round(last["miou"] - random.uniform(0.01, 0.03),  4)
        v_acc  = round(last["accuracy"] - random.uniform(0.005, 0.02), 4)

        summary = {
            "epoch":       epoch,
            "train_loss":  last["loss"],
            "train_acc":   last["accuracy"],
            "train_f1":    last["f1"],
            "train_miou":  last["miou"],
            "val_loss":    v_loss,
            "val_acc":     v_acc,
            "val_miou":    v_miou,
        }
        summaries.append(summary)

        mprint()
        mprint(f"  Epoch {epoch} Summary")
        mprint(f"  Train →  Loss: {summary['train_loss']:.4f}  "
               f"Acc: {summary['train_acc']:.4f}  "
               f"F1: {summary['train_f1']:.4f}  "
               f"mIoU: {summary['train_miou']:.4f}")
        mprint(f"  Val   →  Loss: {summary['val_loss']:.4f}  "
               f"Acc: {summary['val_acc']:.4f}  "
               f"mIoU: {summary['val_miou']:.4f}")

        # Save checkpoint JSON
        ckpt_dir = Path(config["logging"]["checkpoint_dir"])
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"checkpoint_epoch_{epoch:02d}.json"
        ckpt_path.write_text(json.dumps({"epoch": epoch, "metrics": summary}, indent=2))
        mprint(f"  Checkpoint → {ckpt_path}")
        mprint()

    return summaries


# ── Final summary ─────────────────────────────────────────────────────────────

def print_final_summary(summaries: list) -> None:
    f = summaries[-1]
    mprint()
    msep("=")
    mprint("  TRAINING COMPLETE  —  MOCK OUTPUT")
    msep("=")
    mprint()
    mprint("  Per-epoch results:")
    mprint(f"  {'Epoch':<8}{'Train Loss':<14}{'Train Acc':<13}"
           f"{'Train F1':<12}{'Train mIoU':<14}{'Val Loss':<12}{'Val mIoU'}")
    mprint("  " + "─" * 78)
    for s in summaries:
        mprint(
            f"  {s['epoch']:<8}{s['train_loss']:<14.4f}{s['train_acc']:<13.4f}"
            f"{s['train_f1']:<12.4f}{s['train_miou']:<14.4f}"
            f"{s['val_loss']:<12.4f}{s['val_miou']:.4f}"
        )
    mprint()
    mprint(f"  Final Train Loss  : {f['train_loss']:.4f}")
    mprint(f"  Final Train Acc   : {f['train_acc']:.4f}")
    mprint(f"  Final Train F1    : {f['train_f1']:.4f}")
    mprint(f"  Final Train mIoU  : {f['train_miou']:.4f}")
    mprint(f"  Final Val Loss    : {f['val_loss']:.4f}")
    mprint(f"  Final Val Acc     : {f['val_acc']:.4f}")
    mprint(f"  Final Val mIoU    : {f['val_miou']:.4f}")
    mprint()
    msep("=")
    mprint()


# ── Metric math (mirrors notebook formulas) ───────────────────────────────────

def _compute_metrics(step: int, total: int, p: dict) -> dict:
    prog  = step / total
    noise = p["noise_scale"]

    loss = (p["base_loss"] * math.exp(-p["loss_decay_rate"] * prog)
            + p["loss_floor"] + _j(noise))

    acc  = (p["acc_start"]
            + (p["acc_ceiling"]  - p["acc_start"])  * (1 - math.exp(-p["acc_growth_rate"]  * prog))
            + _j(noise * 0.5))

    prec = (p["precision_start"]
            + (p["precision_ceiling"] - p["precision_start"]) * (1 - math.exp(-p["metric_growth_rate"] * prog))
            + _j(noise))

    rec  = (p["recall_start"]
            + (p["recall_ceiling"]  - p["recall_start"])  * (1 - math.exp(-p["metric_growth_rate"] * prog))
            + _j(noise))

    prec = max(0.0, min(1.0, prec))
    rec  = max(0.0, min(1.0, rec))
    f1   = 2 * prec * rec / (prec + rec + 1e-8)

    miou = 0.35 + 0.48 * (1 - math.exp(-p["metric_growth_rate"] * 0.85 * prog)) + _j(noise)
    miou = max(0.0, min(1.0, miou))

    return {
        "loss":      round(loss, 4),
        "accuracy":  round(acc,  4),
        "precision": round(prec, 4),
        "recall":    round(rec,  4),
        "f1":        round(f1,   4),
        "miou":      round(miou, 4),
    }

def _j(scale: float) -> float:
    return random.gauss(0, scale)

def _poly_lr(step: int, tc: dict) -> float:
    base, warmup, total, power = tc["base_lr"], tc["warmup_steps"], tc["total_steps"], tc["poly_power"]
    if step < warmup:
        return base * step / warmup
    return base * ((1 - (step - warmup) / (total - warmup)) ** power)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    mprint()
    msep("*")
    mprint("  SegFormer — ADE20K Semantic Segmentation  |  MOCK PIPELINE")
    mprint("  ⚠️  THIS IS A SIMULATED DEMO — NOT REAL TRAINING  ⚠️")
    msep("*")
    mprint()
    mprint("  All stages below simulate a real SegFormer training pipeline.")
    mprint("  Metrics are mathematically generated; no GPU computation occurs.")
    mprint("  Checkpoint JSON files are written to ./checkpoints/")
    mprint()

    config = load_config("config/model_config.json")

    simulate_dataset_loading(config)
    simulate_preprocessing(config)
    simulate_model_loading(config)
    summaries = simulate_training(config)
    print_final_summary(summaries)

    # Save training log
    log_dir = Path(config["logging"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "training_log.json"
    log_path.write_text(json.dumps({
        "note": "MOCK OUTPUT — simulated training pipeline",
        "epoch_summaries": summaries,
    }, indent=2))
    mprint(f"  Training log saved → {log_path}")
    mprint()


if __name__ == "__main__":
    main()
