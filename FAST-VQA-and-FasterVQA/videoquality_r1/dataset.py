"""
dataset.py — VQADataset for VideoQuality-R1

Label format in examplar_data_labels/:
    <relative_video_path>, <duration>, <fps>, <MOS>
    e.g.  KoNViD_1k_videos/4542323058.mp4, 8.008, 29.97, 3.22

MOS is always the LAST column (index -1).
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

warnings.filterwarnings("ignore", category=FutureWarning)

REPO_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LABEL_ROOT = os.path.join(REPO_ROOT, "examplar_data_labels")
DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")

# ---------------------------------------------------------------------------
# SET THESE to wherever your videos live on disk
# ---------------------------------------------------------------------------
VIDEO_ROOTS = {
    "KoNViD":     os.path.join(os.path.dirname(__file__), "KoNViD_1k_videos", "KoNViD"),
    "LIVE_VQC":   os.path.join(REPO_ROOT, "LIVE_VQC", "Video"),
    "YouTubeUGC": os.path.join(REPO_ROOT, "YouTubeUGC_videos"),
}

DATASET_CONFIGS = {
    "KoNViD": {
        "labels":     os.path.join(LABEL_ROOT, "KoNViD",     "labels.txt"),
        "video_root": VIDEO_ROOTS["KoNViD"],
    },
    "LIVE_VQC": {
        "labels":     os.path.join(LABEL_ROOT, "LIVE_VQC",   "labels.txt"),
        "video_root": VIDEO_ROOTS["LIVE_VQC"],
    },
    "YouTubeUGC": {
        "labels":     os.path.join(LABEL_ROOT, "YouTubeUGC", "labels.txt"),
        "video_root": VIDEO_ROOTS["YouTubeUGC"],
    },
    "CVD2014": {
        "labels":     os.path.join(LABEL_ROOT, "CVD2014",    "labels.txt"),
        "video_root": None,
    },
    "LSVQ": {
        "labels":     os.path.join(LABEL_ROOT, "LSVQ",       "labels.txt"),
        "video_root": None,
    },
    "LSVQ_test": {
        "labels":     os.path.join(LABEL_ROOT, "LSVQ",       "labels_test.txt"),
        "video_root": None,
    },
    "LSVQ_1080p": {
        "labels":     os.path.join(LABEL_ROOT, "LSVQ",       "labels_1080p.txt"),
        "video_root": None,
    },
    "LIVE_VQA": {
        "names":      os.path.join(LABEL_ROOT, "LIVE_VQA",   "names.txt"),
        "scores":     os.path.join(LABEL_ROOT, "LIVE_VQA",   "scores.txt"),
        "video_root": None,
        "format":     "split",
    },
    "KoNiQ10k_train": {
        "labels":     os.path.join(LABEL_ROOT, "KoNiQ10k",   "training_labels.txt"),
        "video_root": None,
    },
    "KoNiQ10k_val": {
        "labels":     os.path.join(LABEL_ROOT, "KoNiQ10k",   "validation_labels.txt"),
        "video_root": None,
    },
}


# ---------------------------------------------------------------------------
# Label parser — MOS is always the LAST column
# ---------------------------------------------------------------------------

def _parse_label_file(label_file: str, video_root: str = None) -> list:
    """
    Handles:
      KoNViD_1k_videos/4542323058.mp4, 8.008, 29.97, 3.22   ← MOS = last col
      Video/A001.mp4, 10.002, 30.0, 80.232                   ← MOS = last col
      Sports_2160P-4e9f.mp4, -1, -1, 4.405                   ← MOS = last col

    Path resolution:
      - if video_root set: video_root / basename(rel_path)
      - else:              REPO_ROOT  / rel_path  (preserves subdir like KoNViD_1k_videos/)
    """
    rows = []
    with open(label_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")] if "," in line else line.split()
            if len(parts) < 2:
                continue

            rel_path = parts[0]
            mos      = float(parts[-1])   # always last column

            if video_root:
                # Match by numeric ID prefix since downloaded filenames have
                # extra suffixes like _original_centercrop_960x540_8s.mp4
                base = os.path.basename(rel_path)
                vid_id = base.replace(".mp4", "")
                candidates = glob.glob(os.path.join(video_root, f"{vid_id}_*.mp4"))
                if not candidates:
                    candidates = glob.glob(os.path.join(video_root, f"{vid_id}.mp4"))
                path = candidates[0] if candidates else os.path.join(video_root, base)
            elif os.path.isabs(rel_path):
                path = rel_path
            else:
                # Preserve relative structure: REPO_ROOT/KoNViD_1k_videos/xxx.mp4
                path = os.path.join(REPO_ROOT, rel_path)

            rows.append({"video_path": path, "mos": mos})
    return rows


def _parse_split(names_file, scores_file, video_root=None):
    names  = [l.strip() for l in open(names_file)  if l.strip()]
    scores = [float(l)  for l in open(scores_file) if l.strip()]
    return [
        {"video_path": os.path.join(video_root, n) if video_root else n, "mos": s}
        for n, s in zip(names, scores)
    ]


# ---------------------------------------------------------------------------
# CSV builder
# ---------------------------------------------------------------------------

def build_csvs(
    datasets=("KoNViD",),
    val_ratio=0.15,
    seed=42,
    filter_missing=True,
    out_dir=DATA_DIR,
):
    os.makedirs(out_dir, exist_ok=True)
    all_rows = []

    for name in datasets:
        cfg = DATASET_CONFIGS.get(name)
        if not cfg:
            print(f"[Dataset] SKIP {name}: unknown dataset"); continue

        fmt = cfg.get("format", "AB")

        if fmt == "split":
            nf, sf = cfg["names"], cfg["scores"]
            if not os.path.exists(nf):
                print(f"[Dataset] SKIP {name}: {nf} not found"); continue
            rows = _parse_split(nf, sf, cfg.get("video_root"))
        else:
            lf = cfg["labels"]
            if not os.path.exists(lf):
                print(f"[Dataset] SKIP {name}: {lf} not found"); continue
            rows = _parse_label_file(lf, cfg.get("video_root"))

        print(f"[Dataset] {name}: {len(rows)} rows parsed")
        all_rows.extend(rows)

    if not all_rows:
        raise RuntimeError("No rows loaded from any dataset.")

    df = pd.DataFrame(all_rows)

    if filter_missing:
        before = len(df)
        df = df[df["video_path"].apply(os.path.exists)].reset_index(drop=True)
        print(f"[Dataset] {before} → {len(df)} rows after filtering missing files")

    if len(df) == 0:
        print("[Dataset] WARNING: 0 videos found on disk — check VIDEO_ROOTS paths.")
        print("[Dataset] Writing full CSV anyway for path debugging.")
        df = pd.DataFrame(all_rows)

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n_val   = max(1, int(len(df) * val_ratio))
    n_train = len(df) - n_val

    train_csv = os.path.join(out_dir, "train.csv")
    val_csv   = os.path.join(out_dir, "val.csv")
    df.iloc[:n_train].to_csv(train_csv, index=False)
    df.iloc[n_train:].to_csv(val_csv,   index=False)

    print(f"[Dataset] {n_train} train → {train_csv}")
    print(f"[Dataset] {n_val}   val   → {val_csv}")
    print(f"[Dataset] MOS range: {df['mos'].min():.3f} – {df['mos'].max():.3f}")
    return train_csv, val_csv


def build_demo_csv(out_dir=None):
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(__file__), "data")
    demo = os.path.join(REPO_ROOT, "demos", "10053703034.mp4")
    if not os.path.exists(demo):
        raise FileNotFoundError(f"Demo video not found: {demo}")
    os.makedirs(out_dir, exist_ok=True)
    rows = [{"video_path": demo, "mos": 1.0 + (i % 5)} for i in range(32)]
    df   = pd.DataFrame(rows)
    train_csv = os.path.join(out_dir, "train.csv")
    val_csv   = os.path.join(out_dir, "val.csv")
    df.iloc[:24].to_csv(train_csv, index=False)
    df.iloc[24:].to_csv(val_csv,   index=False)
    print(f"[Dataset] Demo CSV: 24 train / 8 val  ({demo})")
    return train_csv, val_csv


# ---------------------------------------------------------------------------
# Fragment sampler
# ---------------------------------------------------------------------------

def _sample_fragments(
    video_path, num_frames=8, fragment_size=32, num_fragments=7,
    keys=("fragments",),
):
    fallback = {k: torch.zeros(1, 3, num_frames, fragment_size, fragment_size)
                for k in keys}

    if not os.path.exists(video_path):
        return fallback   # silent — no decord error spam

    try:
        import decord
        decord.bridge.set_bridge("torch")
        vr     = decord.VideoReader(video_path, ctx=decord.cpu(0))
        total  = len(vr)
        idx    = np.linspace(0, total - 1, num_frames, dtype=int)
        frames = vr.get_batch(idx).permute(3, 0, 1, 2).float() / 255.0
    except Exception as e:
        print(f"[Dataset] WARNING: {os.path.basename(video_path)}: {e}")
        return fallback

    C, T, H, W = frames.shape
    result = {}

    if "resize" in keys:
        result["resize"] = torch.nn.functional.interpolate(
            frames.permute(1, 0, 2, 3),
            size=(fragment_size, fragment_size),
            mode="bilinear", align_corners=False,
        ).permute(1, 0, 2, 3).unsqueeze(0)

    if "fragments" in keys:
        patches = []
        for _ in range(num_fragments):
            if H > fragment_size and W > fragment_size:
                top  = np.random.randint(0, H - fragment_size)
                left = np.random.randint(0, W - fragment_size)
                p = frames[:, :, top:top+fragment_size, left:left+fragment_size]
            else:
                p = torch.nn.functional.interpolate(
                    frames.permute(1, 0, 2, 3),
                    size=(fragment_size, fragment_size),
                    mode="bilinear", align_corners=False,
                ).permute(1, 0, 2, 3)
            patches.append(p)
        result["fragments"] = torch.stack(patches).mean(0).unsqueeze(0)

    return result


# ---------------------------------------------------------------------------
# Dataset & collate
# ---------------------------------------------------------------------------

class VQADataset(Dataset):
    def __init__(
        self,
        csv_path,
        num_frames=8,
        fragment_size=32,
        num_fragments=7,
        fragment_keys=("fragments",),
        mos_min=None,
        mos_max=None,
    ):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"CSV not found: {csv_path}\n"
                "Run first:\n"
                "  python dataset.py --datasets KoNViD\n"
                "  python dataset.py --demo"
            )
        self.df            = pd.read_csv(csv_path)
        self.num_frames    = num_frames
        self.fragment_size = fragment_size
        self.num_fragments = num_fragments
        self.fragment_keys = tuple(fragment_keys)
        assert "video_path" in self.df.columns and "mos" in self.df.columns

        self.mos_min = mos_min if mos_min is not None else float(self.df["mos"].min())
        self.mos_max = mos_max if mos_max is not None else float(self.df["mos"].max())
        print(f"[Dataset] {len(self.df)} samples | MOS [{self.mos_min:.2f}, {self.mos_max:.2f}] | {os.path.basename(csv_path)}")

    def __len__(self):
        return len(self.df)

    def _norm(self, mos):
        if self.mos_max == self.mos_min:
            return 3.0
        return 1.0 + 4.0 * (mos - self.mos_min) / (self.mos_max - self.mos_min)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        frags = _sample_fragments(
            str(row["video_path"]),
            self.num_frames, self.fragment_size,
            self.num_fragments, self.fragment_keys,
        )
        mos = torch.tensor(self._norm(float(row["mos"])), dtype=torch.float32)
        return frags, mos


def vqa_collate_fn(batch):
    keys  = batch[0][0].keys()
    frags = {k: torch.cat([b[0][k] for b in batch], dim=0) for k in keys}
    mos   = torch.stack([b[1] for b in batch])
    return frags, mos


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["KoNViD"],
                   choices=list(DATASET_CONFIGS.keys()))
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--no_filter", action="store_true")
    p.add_argument("--demo",      action="store_true",
                   help="Build CSV from demos/10053703034.mp4")
    args = p.parse_args()

    if args.demo:
        build_demo_csv()
    else:
        build_csvs(args.datasets, args.val_ratio,
                   filter_missing=not args.no_filter)

        import pandas as _pd
        print("\nSample paths:")
        _df = _pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
        print(_df[["video_path","mos"]].head(3).to_string(index=False))