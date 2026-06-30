"""
evaluate.py — Compute PLCC/SRCC for VideoQuality-R1 on a held-out val set.
Run after training to get the numbers for the poster's Results table.

Usage:
    python evaluate.py --ckpt ckpts/policy_epoch29.pth --val_csv data/val.csv
"""

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backbone import FrozenBackbone
from policy_head import PolicyHead
from dataset import VQADataset, vqa_collate_fn

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def evaluate(
    ckpt_path: str,
    val_csv: str = "data/val.csv",
    options_path: str = None,
    pretrained_path: str = None,
    batch_size: int = 8,
    device: str = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if options_path is None:
        options_path = os.path.join(REPO_ROOT, "options", "fast", "fast-b.yml")
    if pretrained_path is None:
        pretrained_path = os.path.join(REPO_ROOT, "pretrained", "FAST_VQA_B_1_4.pth")

    print(f"[Eval] Device: {device}")
    print(f"[Eval] Loading backbone...")
    backbone = FrozenBackbone(
        pretrained_path=pretrained_path,
        options_path=options_path,
        device=device,
    )

    print(f"[Eval] Loading policy head from {ckpt_path}")
    policy = PolicyHead(feature_dim=768, hidden_dim=256).to(device)
    state = torch.load(ckpt_path, map_location=device)
    policy.load_state_dict(state)
    policy.eval()

    print(f"[Eval] Loading validation set from {val_csv}")
    val_dataset = VQADataset(val_csv, fragment_keys=("fragments",))
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=vqa_collate_fn,
    )

    preds, gts = [], []
    print(f"[Eval] Running inference on {len(val_dataset)} samples...")
    with torch.no_grad():
        for i, (fragments, gt_mos) in enumerate(val_loader):
            feats = backbone.extract_features(fragments)
            scores, _, _ = policy(feats)
            preds.extend(scores.cpu().tolist())
            gts.extend(gt_mos.tolist())
            if (i + 1) % 10 == 0:
                print(f"  batch {i+1}/{len(val_loader)}")

    plcc = pearsonr(preds, gts)[0]
    srcc = spearmanr(preds, gts)[0]

    print("\n" + "=" * 50)
    print(f"VideoQuality-R1 Results")
    print("=" * 50)
    print(f"  N samples : {len(preds)}")
    print(f"  PLCC      : {plcc:.3f}")
    print(f"  SRCC      : {srcc:.3f}")
    print("=" * 50)
    print(f"\nPoster table format:")
    print(f"  VideoQuality-RL-1   {plcc:.3f} / {srcc:.3f}")

    return {"plcc": plcc, "srcc": srcc, "n_samples": len(preds), "preds": preds, "gts": gts}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to trained policy_head .pth")
    parser.add_argument("--val_csv", default="data/val.csv")
    parser.add_argument("--options_path", default=None)
    parser.add_argument("--pretrained_path", default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    evaluate(
        ckpt_path=args.ckpt,
        val_csv=args.val_csv,
        options_path=args.options_path,
        pretrained_path=args.pretrained_path,
        batch_size=args.batch_size,
    )