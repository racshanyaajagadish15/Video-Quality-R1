"""
dataset.py — VQADataset for VideoQuality-R1
Supports all label formats found in FastVQA's examplar_data_labels/:
  Format A:  <video_path> <mos>                      (space/tab separated)
  Format B:  <video_name>,<mos>,...                  (CSV, name only — needs prefix)
  Format C:  LIVE_VQA split files names.txt/scores.txt
Builds data/train.csv and data/val.csv automatically if they don't exist.
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Repo layout constants — all relative to REPO_ROOT
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LABEL_ROOT = os.path.join(REPO_ROOT, "examplar_data_labels")
DATA_DIR   = os.path.join(REPO_ROOT, "videoquality_r1", "data")

# Map dataset name → (label file, video root hint, format)
# format: "AB" = "path mos" per line, "csv" = comma-sep, "split" = names+scores
DATASET_CONFIGS = {
    "KoNViD": {
        "labels": os.path.join(LABEL_ROOT, "KoNViD", "labels.txt"),
        "video_root": None,   # lines contain full paths
        "format": "AB",
    },
    "LSVQ": {
        "labels": os.path.join(LABEL_ROOT, "LSVQ", "labels.txt"),
        "video_root": None,
        "format": "AB",
    },
    "LSVQ_test": {
        "labels": os.path.join(LABEL_ROOT, "LSVQ", "labels_test.txt"),
        "video_root": None,
        "format": "AB",
    },
    "LSVQ_1080p": {
        "labels": os.path.join(LABEL_ROOT, "LSVQ", "labels_1080p.txt"),
        "video_root": None,
        "format": "AB",
    },
    "LIVE_VQC": {
        "labels": os.path.join(LABEL_ROOT, "LIVE_VQC", "labels.txt"),
        "video_root": None,
        "format": "AB",
    },
    "YouTubeUGC": {
        "labels": os.path.join(LABEL_ROOT, "YouTubeUGC", "labels.txt"),
        "video_root": None,
        "format": "AB",
    },
    "CVD2014": {
        "labels": os.path.join(LABEL_ROOT, "CVD2014", "labels.txt"),
        "video_root": None,
        "format": "AB",
    },
    "LIVE_VQA": {
        "names":  os.path.join(LABEL_ROOT, "LIVE_VQA", "names.txt"),
        "scores": os.path.join(LABEL_ROOT, "LIVE_VQA", "scores.txt"),
        "video_root": None,
        "format": "split",
    },
    "KoNiQ10k_train": {
        "labels": os.path.join(LABEL_ROOT, "KoNiQ10k", "training_labels.txt"),
        "video_root": None,
        "format": "AB",
    },
    "KoNiQ10k_val": {
        "labels": os.path.join(LABEL_ROOT, "KoNiQ10k", "validation_labels.txt"),
        "video_root": None,
        "format": "AB",
    },
}


# ---------------------------------------------------------------------------
# Label parsers
# ---------------------------------------------------------------------------

def _parse_AB(label_file: str, video_root=None) -> list[dict]:
    """
    Parse lines of the form:
        /path/to/video.mp4 3.45
    or  video.mp4 3.45
    or  video.mp4,3.45
    """
    rows = []
    with open(label_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Support both whitespace and comma delimiters
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                continue
            path, mos = parts[0], float(parts[1])
            if video_root and not os.path.isabs(path):
                path = os.path.join(video_root, path)
            rows.append({"video_path": path, "mos": mos})
    return rows


def _parse_split(names_file: str, scores_file: str, video_root=None) -> list[dict]:
    """Parse LIVE_VQA-style split names.txt + scores.txt."""
    with open(names_file) as f:
        names = [l.strip() for l in f if l.strip()]
    with open(scores_file) as f:
        scores = [float(l.strip()) for l in f if l.strip()]
    rows = []
    for name, mos in zip(names, scores):
        path = os.path.join(video_root, name) if video_root else name
        rows.append({"video_path": path, "mos": mos})
    return rows


def load_dataset_rows(dataset_name: str) -> list[dict]:
    """Load rows for a named dataset from examplar_data_labels/."""
    cfg = DATASET_CONFIGS.get(dataset_name)
    if cfg is None:
        raise ValueError(
            f"Unknown dataset {dataset_name!r}. "
            f"Available: {list(DATASET_CONFIGS.keys())}"
        )
    fmt = cfg["format"]
    vr  = cfg.get("video_root")

    if fmt == "AB":
        lf = cfg["labels"]
        if not os.path.exists(lf):
            raise FileNotFoundError(f"Label file not found: {lf}")
        return _parse_AB(lf, vr)

    if fmt == "split":
        nf, sf = cfg["names"], cfg["scores"]
        if not os.path.exists(nf):
            raise FileNotFoundError(f"Names file not found: {nf}")
        if not os.path.exists(sf):
            raise FileNotFoundError(f"Scores file not found: {sf}")
        return _parse_split(nf, sf, vr)

    raise ValueError(f"Unknown format {fmt!r}")


# ---------------------------------------------------------------------------
# CSV builder — run once to create data/train.csv and data/val.csv
# ---------------------------------------------------------------------------

def build_csvs(
    datasets: list[str] = ("KoNViD", "LIVE_VQC", "YouTubeUGC"),
    val_ratio: float = 0.15,
    seed: int = 42,
    filter_missing: bool = True,
    out_dir: str = DATA_DIR,
) -> tuple[str, str]:
    """
    Merge rows from multiple datasets, split into train/val,
    write CSVs and return (train_csv_path, val_csv_path).

    Args:
        datasets      : list of dataset names from DATASET_CONFIGS
        val_ratio     : fraction of data held out for validation
        seed          : random seed for reproducible split
        filter_missing: drop rows whose video_path doesn't exist on disk
        out_dir       : directory to write train.csv / val.csv
    """
    os.makedirs(out_dir, exist_ok=True)
    all_rows = []

    for name in datasets:
        try:
            rows = load_dataset_rows(name)
            print(f"[Dataset] {name}: {len(rows)} samples loaded")
            all_rows.extend(rows)
        except FileNotFoundError as e:
            print(f"[Dataset] SKIP {name}: {e}")

    if not all_rows:
        raise RuntimeError(
            "No rows loaded. Check that examplar_data_labels/ exists and "
            "at least one dataset label file is readable."
        )

    df = pd.DataFrame(all_rows)
    original_len = len(df)

    if filter_missing:
        df = df[df["video_path"].apply(os.path.exists)].reset_index(drop=True)
        dropped = original_len - len(df)
        if dropped:
            print(f"[Dataset] Dropped {dropped} rows with missing video files")

    if len(df) == 0:
        print(
            "[Dataset] WARNING: All video paths are missing on disk.\n"
            "          Writing CSV with all rows anyway (useful for path debugging).\n"
            "          Set filter_missing=False or fix video paths."
        )
        df = pd.DataFrame(all_rows)

    # Shuffle and split
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n_val   = max(1, int(len(df) * val_ratio))
    n_train = len(df) - n_val

    train_df = df.iloc[:n_train]
    val_df   = df.iloc[n_train:]

    train_csv = os.path.join(out_dir, "train.csv")
    val_csv   = os.path.join(out_dir, "val.csv")
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv,   index=False)

    print(f"[Dataset] Written {n_train} train rows → {train_csv}")
    print(f"[Dataset] Written {n_val}   val   rows → {val_csv}")
    print(f"[Dataset] MOS range: {df['mos'].min():.3f} – {df['mos'].max():.3f}")
    return train_csv, val_csv


# ---------------------------------------------------------------------------
# Fragment sampler — matches FastVQA's spatial/temporal fragment strategy
# ---------------------------------------------------------------------------

def _sample_fragments(
    video_path: str,
    num_frames: int = 8,
    fragment_size: int = 32,
    num_fragments: int = 7,
    keys: tuple = ("fragments",),   # ← match backbone_preserve_keys
) -> dict:
    try:
        import decord
        decord.bridge.set_bridge("torch")
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        total = len(vr)
        idx = np.linspace(0, total - 1, num_frames, dtype=int)
        frames = vr.get_batch(idx).permute(3, 0, 1, 2).float() / 255.0  # (C,T,H,W)
    except Exception as e:
        print(f"[Dataset] WARNING: could not decode {video_path}: {e}")
        zeros = torch.zeros(1, 3, num_frames, fragment_size, fragment_size)
        return {k: zeros for k in keys}

    C, T, H, W = frames.shape
    result = {}

    if "resize" in keys:
        result["resize"] = torch.nn.functional.interpolate(
            frames.permute(1, 0, 2, 3),
            size=(fragment_size, fragment_size),
            mode="bilinear", align_corners=False,
        ).permute(1, 0, 2, 3).unsqueeze(0)   # (1,C,T,fs,fs)

    if "fragments" in keys:
        frag_list = []
        for _ in range(num_fragments):
            if H > fragment_size and W > fragment_size:
                top  = np.random.randint(0, H - fragment_size)
                left = np.random.randint(0, W - fragment_size)
                patch = frames[:, :, top:top+fragment_size, left:left+fragment_size]
            else:
                patch = torch.nn.functional.interpolate(
                    frames.permute(1, 0, 2, 3),
                    size=(fragment_size, fragment_size),
                    mode="bilinear", align_corners=False,
                ).permute(1, 0, 2, 3)
            frag_list.append(patch)
        result["fragments"] = torch.stack(frag_list).mean(0).unsqueeze(0)  # (1,C,T,fs,fs)

    return result


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class VQADataset(Dataset):
    """
    Video Quality Assessment dataset for VideoQuality-R1 training.

    Args:
        csv_path      : path to train.csv / val.csv (columns: video_path, mos)
        num_frames    : temporal frames to sample per video
        fragment_size : spatial size of each fragment patch (pixels)
        num_fragments : number of spatial crop fragments
        mos_min/max   : normalise MOS to [0,1] if set, else use raw values
    """

    def __init__(
        self,
        csv_path: str,
        num_frames: int = 8,
        fragment_size: int = 32,
        num_fragments: int = 7,
        mos_min: float = None,
        mos_max: float = None,
    ):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"CSV not found: {csv_path}\n"
                f"Run build_csvs() first:\n"
                f"  from dataset import build_csvs\n"
                f"  build_csvs(['KoNViD', 'LIVE_VQC', 'YouTubeUGC'])"
            )

        self.df = pd.read_csv(csv_path)
        assert "video_path" in self.df.columns, "CSV must have 'video_path' column"
        assert "mos" in self.df.columns,        "CSV must have 'mos' column"

        self.num_frames    = num_frames
        self.fragment_size = fragment_size
        self.num_fragments = num_fragments

        # MOS normalisation
        self.mos_min = mos_min if mos_min is not None else float(self.df["mos"].min())
        self.mos_max = mos_max if mos_max is not None else float(self.df["mos"].max())

        print(
            f"[Dataset] Loaded {len(self.df)} samples from {csv_path}\n"
            f"[Dataset] MOS range: {self.mos_min:.3f} – {self.mos_max:.3f}"
        )

    def __len__(self) -> int:
        return len(self.df)

    def _normalise_mos(self, mos: float) -> float:
        """Normalise MOS to [1, 5] scale (FastVQA convention)."""
        if self.mos_max == self.mos_min:
            return 3.0
        normed = (mos - self.mos_min) / (self.mos_max - self.mos_min)  # [0,1]
        return 1.0 + normed * 4.0  # [1,5]

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        video_path = str(row["video_path"])
        mos_raw    = float(row["mos"])

        fragments = _sample_fragments(
            video_path,
            num_frames=self.num_frames,
            fragment_size=self.fragment_size,
            num_fragments=self.num_fragments,
            keys=("fragments",),   # only what fast-b.yml builds
        )
        mos = torch.tensor(self._normalise_mos(mos_raw), dtype=torch.float32)
        return fragments, mos


# ---------------------------------------------------------------------------
# Collate function for DataLoader
# ---------------------------------------------------------------------------

def vqa_collate_fn(batch: list) -> tuple[dict, torch.Tensor]:
    """
    Custom collate: stacks fragment dicts along batch dim.
    batch: list of (fragment_dict, mos_tensor)
    """
    frag_keys = batch[0][0].keys()
    fragments = {
        k: torch.cat([item[0][k] for item in batch], dim=0)
        for k in frag_keys
    }
    mos = torch.stack([item[1] for item in batch])
    return fragments, mos


# ---------------------------------------------------------------------------
# CLI helper — run directly to build CSVs
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build train/val CSVs for VideoQuality-R1")
    parser.add_argument(
        "--datasets", nargs="+",
        default=["KoNViD", "LIVE_VQC", "YouTubeUGC"],
        choices=list(DATASET_CONFIGS.keys()),
        help="Which datasets to include",
    )
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument(
        "--no_filter", action="store_true",
        help="Don't filter rows with missing video files (useful for dry-run)",
    )
    args = parser.parse_args()

    print(f"Building CSVs from: {args.datasets}")
    train_csv, val_csv = build_csvs(
        datasets=args.datasets,
        val_ratio=args.val_ratio,
        filter_missing=not args.no_filter,
    )
    print(f"\nDone!\n  Train: {train_csv}\n  Val:   {val_csv}")

    # Quick sanity check
    print("\nFirst 3 train rows:")
    print(pd.read_csv(train_csv).head(3).to_string(index=False))

def build_demo_csv(out_dir=None):
    import os
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(__file__), "data")
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    demo = os.path.join(REPO_ROOT, "demos", "10053703034.mp4")
    if not os.path.exists(demo):
        raise FileNotFoundError(f"Demo video not found: {demo}")
    import pandas as pd
    os.makedirs(out_dir, exist_ok=True)
    rows = [{"video_path": demo, "mos": 1.0 + (i % 5)} for i in range(32)]
    df = pd.DataFrame(rows)
    train_csv = os.path.join(out_dir, "train.csv")
    val_csv   = os.path.join(out_dir, "val.csv")
    df.iloc[:24].to_csv(train_csv, index=False)
    df.iloc[24:].to_csv(val_csv,   index=False)
    print(f"[Dataset] Demo CSV: 24 train / 8 val  ({demo})")
    return train_csv, val_csv
