# IRIS — ISP-free RAW Image Semantic Segmentation (Under Adverse Conditions)

Course project (CSE 499B): semantic segmentation that **trains on Bayer RAW** (four unpacked planes) instead of ISP-produced sRGB, with **AODRaw** as the primary real-world dataset and **SegFormer**-family decoders. This repository holds the **LaTeX report** (`main.tex`) and project notes; the runnable training code may live in a sibling folder (e.g. `iros/IRIS`) on your machine.

---

## What is verified vs what is still project scope

| Item                                                        | Status                  | Notes                                                                 |
| ----------------------------------------------------------- | ----------------------- | --------------------------------------------------------------------- |
| ADE20K indexing, batch shapes, dtypes                       | **Verified** (log `a1`) | 20,210 train / 2,000 val, 150 classes, batch 8                        |
| Augmentations + ImageNet normalize                          | **Verified**            | Matches `main.tex` Experiments section                                |
| SegFormer + **MiT-B2** (`nvidia/mit-b2`), **RGB 3×512×512** | **Verified**            | 25.4M parameters                                                      |
| LR schedule                                                 | **Verified**            | Polynomial decay (power 1.0), **not** cosine; 150-step warmup         |
| 3 epochs × 1000 steps, AMP, AdamW                           | **Verified**            | Final val mIoU **0.7761**                                             |
| Four-channel RAW stem on AODRaw                             | **Project / pipeline**  | Described in `main.tex`; not the same binary as the ADE20K smoke test |

The ADE20K run is a **training-stack smoke test** (dataloaders, loss, metrics, checkpoints). The **IRIS** contribution is the **RAW + pseudo-label + inflated stem** path on **AODRaw**; keep that distinction when writing slides or the report.

---

## Verified ADE20K results (from training log `a1`)

**Stage 1 — dataset**

| Field         | Value     |
| ------------- | --------- |
| Dataset       | ADE20K    |
| Train samples | 20,210    |
| Val samples   | 2,000     |
| Batch size    | 8         |
| Image size    | 512 × 512 |
| Classes       | 150       |
| Train batches | 2526      |
| Val batches   | 250       |

**Stage 2 — preprocessing (train augmentations)**

- `RandomResizeCrop(scale=(0.5, 2.0), size=512)`
- `RandomHorizontalFlip(p=0.5)`
- `PhotoMetricDistortion()`
- `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`

Logged batch checks (examples): images `(8, 3, 512, 512)` float32; masks `(8, 512, 512)` int64 in `[0, 149]`.

**Stage 3 — model**

| Field              | Value                          |
| ------------------ | ------------------------------ |
| Architecture       | SegFormer (MiT-B2)             |
| Pretrained weights | `nvidia/mit-b2` (Hugging Face) |
| Parameters         | **25.4M**                      |
| Embed dims         | [64, 128, 320, 512]            |
| Depths             | [3, 4, 6, 3]                   |
| Num heads          | [1, 2, 5, 8]                   |

**Stage 4 — training**

| Field           | Value                                              |
| --------------- | -------------------------------------------------- |
| Epochs          | 3                                                  |
| Steps per epoch | 1000                                               |
| Total steps     | 3000                                               |
| Optimizer       | AdamW, lr = `6e-5`, weight decay = `0.01`          |
| LR schedule     | **Polynomial** (power `1.0`), warmup **150** steps |
| AMP             | Enabled (`True`)                                   |

**Per-epoch summary (epoch-end aggregates)**

| Epoch | Train loss | Train acc | Train F1 | Train mIoU | Val loss | Val mIoU   |
| ----- | ---------- | --------- | -------- | ---------- | -------- | ---------- |
| 1     | 0.5057     | 0.7993    | 0.7602   | 0.6256     | 0.5256   | 0.6080     |
| 2     | 0.2087     | 0.8987    | 0.8645   | 0.7450     | 0.2344   | 0.7194     |
| 3     | 0.1226     | 0.9274    | 0.8979   | 0.7902     | 0.1407   | **0.7761** |

**Final epoch-3 validation (extra line items from log)**

- Val accuracy: **0.9093**
- Checkpoints: `checkpoints/checkpoint_epoch_01.json` … `checkpoint_epoch_03.json`
- Full log: `logs/training_log.json`

These numbers are what `main.tex` was aligned to (rounded to three decimals in tables where noted).

---

## Repository contents (this folder)

| Path                | Purpose                                      |
| ------------------- | -------------------------------------------- |
| `main.tex`          | IEEEtran-style project update / paper source |
| `base.txt`          | Narrative / motivation notes                 |
| `brainstroming.txt` | Brainstorming and long-form timeline         |
| `weekly_demo.txt`   | Weekly member updates                        |
| `README.md`         | This file                                    |

If your **code** lives elsewhere (e.g. `...\iros\IRIS`), either move a copy of `README.md` next to `main.py` or keep one canonical README in the Git root and symlink / copy as needed.

---

## Suggested code repository layout (IRIS)

Use something close to this so the course manual and parallel work stay compatible:

```text
IRIS/
  main.py                 # Entry: train / eval / smoke
  README.md
  requirements.txt
  data/                     # Raw data roots (gitignored); README only in git
  support/                  # Shared utils, transforms, metrics
  others/                   # Figures, exports, scratch
  checkpoints/              # Saved weights / JSON metadata (gitignored)
  logs/                     # training_log.json (gitignored)
  scripts/                  # Optional: one-off dataset or label jobs
```

**Typical modules (names illustrative)**

- `load_raw.py` — Bayer unpack, black level, normalize → `(4, H/2, W/2)` float32
- `generate_labels.py` — SAM 3 box prompts on paired sRGB; save masks
- `main.py` — load → preprocess → model → loss → step
- `evaluate.py` — mIoU / per-class IoU from checkpoints

Interface contracts (tensor shapes, dtypes, resolution) should be documented in `README.md` or `support/CONTRACTS.md` so Members A–D do not diverge.

---

## Python environment

Example using Conda (adjust names to match your machine):

```powershell
conda create -n pytorch_env python=3.10 -y
conda activate pytorch_env
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install transformers datasets accelerate pillow opencv-python rawpy
pip install tqdm tensorboard  # optional
```

Pin versions in `requirements.txt` once you have a known-good combo.

---

## Reproducing the ADE20K smoke test

1. Install ADE20K in the layout your script expects (images + annotations; 150 semantic classes).
2. Point config to:
   - model id: `nvidia/mit-b2`
   - num labels: `150`
   - image size: `512`
   - batch size: `8`
3. Set optimizer and schedule to match the log:
   - AdamW, lr `6e-5`, wd `0.01`
   - Warmup **150** steps
   - **Polynomial** LR decay with power **1.0** (not cosine)
4. Run **3** epochs with **1000** training steps per epoch (or full epoch if you prefer, but then numbers will not match this log).
5. Compare epoch-end train/val rows to the table above.

If your numbers differ slightly, check: seed, `drop_last`, exact `RandomResizeCrop` / `PhotoMetricDistortion` defaults, and whether validation uses center-crop vs. multi-scale.

---

## IRIS pipeline (AODRaw — high level)

1. **RAW** — `.ARW` / `.CR2` via `rawpy` → four channels R, G1, G2, B, normalized.
2. **Labels** — COCO boxes → **SAM 3** on paired **sRGB** → masks → optional **quality filter** (confidence + Sobel edge sharpness).
3. **Model** — SegFormer first conv inflated to **4** input channels; pretrained RGB weights + duplicated / averaged green init (see paper).
4. **Train / eval** — cross-entropy (+ planned distillation, NALN); metrics on nine weather/light mixes.

Datasets and citations are listed in `main.tex` (AODRaw, ACDC, ADE20K, etc.).

---

## Citing and third-party assets

- **SegFormer**: Xie et al., NeurIPS 2021.
- **ADE20K**: Zhou et al., IJCV 2019.
- **AODRaw**: Zhang et al., CVPR 2024.
- **SAM / SAM 3**: Meta AI; use the citation your report already includes.

---

## Team (from `main.tex`)

- Arefin Amin
- Fatema Tabassum Elma
- Labiba Faiza Karim
- Ratul Hasan Ankon

North South University, Department of Electrical and Computer Engineering.

---

## Building the PDF

With a LaTeX distribution installed:

```powershell
pdflatex main.tex
pdflatex main.tex
```

Install the **newtx** bundle if `main.tex` uses `newtxtext` / `newtxmath` and compilation complains about missing fonts.

---

## Troubleshooting

| Symptom              | Things to check                                            |
| -------------------- | ---------------------------------------------------------- |
| Val mIoU flat or NaN | Class count = 150; ignore index for void if any            |
| LR curve wrong       | Polynomial vs cosine; warmup step count                    |
| OOM at batch 8       | Gradient accumulation or smaller backbone                  |
| RAW shape wrong      | Bayer pattern vs camera; black level; half-res per channel |

---

## License

Course project materials: set license when you publish the code repository (e.g. MIT for code, CC-BY for report) per instructor policy.
