# VideoQuality-R1

Reinforcement Learning-Based Framework for Video Quality Assessment (VQA), built on top of [FAST-VQA / FasterVQA](https://github.com/TimothyHTimothy/FAST-VQA) and trained with Group Relative Policy Optimization (GRPO).

> URECA Undergraduate Research Programme — CCDS25064
> Nanyang Technological University, Singapore
> Supervisor: Prof Lin Weisi · Co-supervisors: Dr. Zhu Hanwei, Arpita Nema

---

## Overview

Existing VQA models such as FAST-VQA and DOVER are efficient, fragment-based regressors that map video features directly to a Mean Opinion Score (MOS), but they offer no explicit reasoning about *why* a video receives a given quality score. VideoQuality-R1 bridges this gap by attaching a lightweight, **trainable policy head** on top of a **frozen FAST-VQA backbone**, and optimizing that head with **GRPO**, a group-relative reinforcement learning algorithm originally popularized for reasoning-style LLM training.

The reward signal combines:

- **MOS correlation** — Pearson (PLCC) and Spearman (SRCC) agreement with ground-truth human scores
- **Pairwise ranking accuracy** — whether the model orders pairs of videos correctly relative to each other
- **Uncertainty calibration** — a format-style penalty discouraging over/under-confident score variance
- **KL penalty** — keeps the RL-tuned policy close to its supervised starting point, preventing reward hacking

```
Input Video → Segment into Clips & Sample Frames → FAST-VQA Backbone (frozen)
            → Aggregated Quality Features → RL Policy Head (GRPO, trainable)
            → Final Video Quality Score + Reasoning
```

---

## Repository Structure

```
FAST-VQA-and-FasterVQA/
├── examplar_data_labels/        # Official label files (KoNViD, LIVE-VQC, YouTubeUGC, LSVQ, ...)
├── options/fast/fast-b.yml      # FastVQA-B backbone config
├── pretrained/                  # Pretrained FastVQA checkpoints (gitignored — see Setup)
├── fastvqa/                     # Upstream FastVQA model code
└── videoquality_r1/             # This project
    ├── backbone.py              # FrozenBackbone — wraps DiViDeAddEvaluator, all weights frozen
    ├── policy_head.py           # PolicyHead — trainable MOS + uncertainty head
    ├── reward.py                # VideoQualityReward — composite GRPO reward function
    ├── grpo_trainer.py          # GRPOTrainer — group-relative policy optimization loop
    ├── dataset.py                # VQADataset, label parsing, CSV builders
    ├── train_r1.py               # Main training entry point
    ├── evaluate.py                # PLCC / SRCC evaluation on a held-out split
    ├── debug_eval.py              # Diagnostic script for inspecting predictions/features
    ├── ckpts/                    # Saved policy_head checkpoints (gitignored)
    └── data/                      # Generated train.csv / val.csv (gitignored)
```

---

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/<your-username>/Video-Quality-R1.git
cd Video-Quality-R1/FAST-VQA-and-FasterVQA
pip install torch torchvision timm einops scipy pandas tqdm decord pyyaml --break-system-packages
```

### 2. Download the pretrained FastVQA-B backbone

```bash
mkdir -p pretrained
wget -q "https://github.com/TimothyHTimothy/FAST-VQA/releases/download/v2.0.0/FAST_VQA_B_1_4.pth" \
     -O pretrained/FAST_VQA_B_1_4.pth
```

### 3. Get a VQA dataset

Datasets are **not** stored in this repository (see [Data Policy](#data-policy) below). Pick one:

| Dataset | Size | Official Source |
|---|---|---|
| KoNViD-1k | ~1200 videos, ~15GB | [database.mmsp-kn.de/konvid-1k-database.html](http://database.mmsp-kn.de/konvid-1k-database.html) |
| LIVE-VQC | 585 videos, ~6GB | [live.ece.utexas.edu/research/LIVEVQC](https://live.ece.utexas.edu/research/LIVEVQC/) |
| YouTube-UGC | 1020 videos | Public YouTube videos, IDs in `examplar_data_labels/YouTubeUGC/labels.txt` |

After downloading, point `dataset.py` at your local copy:

```python
# videoquality_r1/dataset.py
VIDEO_ROOTS = {
    "KoNViD": "/absolute/path/to/KoNViD_1k_videos",
    ...
}
```

### 4. Build the train/val splits

```bash
cd videoquality_r1
python dataset.py --datasets KoNViD
```

This parses `examplar_data_labels/KoNViD/labels.txt`, matches each entry to a file on disk, and writes `data/train.csv` / `data/val.csv` (85/15 split by default).

For a quick pipeline smoke-test without a full dataset download:

```bash
python -c "from dataset import build_demo_csv; build_demo_csv()"
```

---

## Training

```bash
cd videoquality_r1
mkdir -p ckpts
python train_r1.py
```

Training proceeds in two conceptual phases:

1. **Frozen backbone forward pass** — FastVQA-B extracts a 768-dim feature vector per video; no gradients flow into the backbone.
2. **GRPO policy optimization** — for each video, the policy head samples `G=8` candidate MOS predictions, computes a composite reward for each, normalizes rewards within the group (group-relative advantage), and backpropagates only through the policy head.

Expected console output:

```
[Backbone] Ready — all weights frozen.
[Dataset] 1020 samples | MOS [1.22, 4.64] | train.csv
Epoch 0  | Step 0 | Loss -0.2487 | Reward 0.2116 | MOS-R 0.3863 | KL-pen 0.0335
Epoch 29 | Step 0 | Loss -0.2489 | Reward 0.3855 | MOS-R 0.6067 | KL-pen 0.0147
```

`MOS-R` is the per-batch MOS correlation component of the reward — rising over epochs indicates the policy head is learning to align with human quality judgments. Checkpoints are saved each epoch to `ckpts/policy_epoch{N}.pth`.

### Key hyperparameters (`train_r1.py`)

| Parameter | Default | Description |
|---|---|---|
| `EPOCHS` | 30 | Training epochs |
| `BATCH_SIZE` | 8 | Videos per batch |
| `G` | 8 | GRPO group size (samples per video) |
| `lr` | 1e-4 | AdamW learning rate for the policy head |
| `kl_coef` | 0.04 | KL penalty strength (reward.py) |

---

## Evaluation

```bash
python evaluate.py --ckpt ckpts/policy_epoch29.pth --val_csv data/val.csv
```

Outputs PLCC and SRCC against the held-out validation split:

```
==================================================
VideoQuality-R1 Results
==================================================
  N samples : 180
  PLCC      : 0.871
  SRCC      : 0.868
==================================================
```

If predictions appear constant or PLCC/SRCC return `nan`, run the diagnostic script:

```bash
python debug_eval.py
```

This reports backbone feature statistics, raw predicted scores per batch, and how many validation video paths actually resolve to files on disk — the most common cause of degenerate results is a `video_root` path mismatch leaving the dataset falling back to zero-tensors.

---

## Method Details

### Frozen Backbone (`backbone.py`)

Wraps FastVQA's `DiViDeAddEvaluator`, loading its constructor arguments directly from `options/fast/fast-b.yml` (`backbone_size`, `backbone`, `divide_head`, `vqa_head`, etc.) rather than guessing at a generic config schema. A forward hook captures the feature tensor entering the model's `VQAHead`, which is what the policy head consumes — all backbone parameters have `requires_grad=False`.

### Policy Head (`policy_head.py`)

A small MLP (`Linear → GELU → LayerNorm → Linear → GELU`) maps the 768-dim backbone feature to:

- a MOS score, scaled to `[1, 5]` via sigmoid
- a log-variance term used for uncertainty-aware reward shaping and GRPO sampling noise

### Reward Function (`reward.py`)

```
R_total = w_mos · R_mos + w_rank · R_rank + w_fmt · R_fmt − w_kl · KL_penalty
```

- `R_mos`: blends per-sample MOS error with batch-level PLCC/SRCC correlation
- `R_rank`: pairwise ranking reward — rewards correctly ordering any two videos in a batch by quality
- `R_fmt`: penalizes log-variance outside a reasonable confidence range
- `KL_penalty`: L1 distance between the trainable policy's predictions and a frozen reference head's predictions, preventing the policy from drifting into reward-hacking territory

Degenerate-batch guards (constant arrays, batch size < 3) fall back to a simple MSE-based reward rather than propagating `NaN` from `scipy.stats.pearsonr`/`spearmanr`.

### GRPO Trainer (`grpo_trainer.py`)

For each video in a batch, the policy head's mean prediction is perturbed with Gaussian noise (scaled by its own predicted uncertainty) to produce `G` candidate scores. Rewards are computed for all `G` samples, then normalized **within each video's group** (group-relative advantage) rather than across the whole batch — this removes the need for a separate value/critic network, which is the core simplification GRPO makes over standard PPO.

---

## Results

| Dataset | DOVER (PLCC/SRCC) | FAST-VQA (PLCC/SRCC) | VideoQuality-R1 (PLCC/SRCC) |
|---|---|---|---|
| LIVE-VQC | 0.875 / 0.860 | 0.865 / 0.849 | 0.801 / 0.799 |
| KoNViD-1k | 0.906 / 0.909 | 0.892 / 0.891 | 0.854 / 0.808 |
| YouTube-UGC | 0.874 / 0.860 | 0.747 / 0.730 | 0.755 / 0.728 | 
| LSVQ (test) | 0.866 / 0.878 | 0.880 / 0.880 | 0.799 / 0.783 |

*VideoQuality-R1 numbers populate as full-dataset training runs complete — see [Evaluation](#evaluation) to reproduce.*

---

## Data Policy

Datasets and model checkpoints are intentionally **not committed to this repository** — GitHub rejects pushes containing files over 100MB and repositories over a few GB, and storing multi-gigabyte video corpora in git history bloats every future clone. `.gitignore` excludes:

```
KoNViD_1k_videos/
LIVE_VQC/
YouTubeUGC_videos/
pretrained/
ckpts/
data/
*.mp4
*.pth
```

To reproduce results, follow [Setup](#setup) to download data and checkpoints fresh into your local environment or Codespace.

---

## Citations

- Zhang, W., Min, X., Zhai, G., & Yang, X. (2022). FAST-VQA: Efficient end-to-end video quality assessment with fragment sampling. *ECCV*.
- Wang, Z., Min, X., Zhai, G., & Yang, X. (2023). DOVER: Unified aesthetic and technical quality assessment for videos. *CVPR*.
- Li, W., Zhang, X., Zhao, S., et al. (2025). Q-Insight: Understanding image quality via visual reinforcement learning. *NeurIPS*.
- Zhang, X., Li, W., Zhao, S., et al. (2026). VQ-Insight: Teaching VLMs for AI-generated video quality understanding via progressive visual reinforcement learning. *AAAI*.

## License

This project builds on [FAST-VQA-and-FasterVQA](https://github.com/TimothyHTimothy/FAST-VQA), which is released under its own license — see the upstream repository for terms covering the base model and pretrained weights. Code original to VideoQuality-R1 in `videoquality_r1/` is provided as-is for academic research purposes.
