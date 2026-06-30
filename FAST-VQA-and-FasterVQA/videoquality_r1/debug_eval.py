"""
debug_eval.py — Diagnose why predictions are constant.
"""
import os, sys, torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backbone import FrozenBackbone
from policy_head import PolicyHead
from dataset import VQADataset, vqa_collate_fn

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEVICE = "cpu"

backbone = FrozenBackbone(
    pretrained_path=os.path.join(REPO_ROOT, "pretrained", "FAST_VQA_B_1_4.pth"),
    options_path=os.path.join(REPO_ROOT, "options", "fast", "fast-b.yml"),
    device=DEVICE,
)
policy = PolicyHead(feature_dim=768, hidden_dim=256).to(DEVICE)
policy.load_state_dict(torch.load("ckpts/policy_epoch29.pth", map_location=DEVICE))
policy.eval()

val_dataset = VQADataset("data/val.csv", fragment_keys=("fragments",))
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=vqa_collate_fn)

print("\n--- Checking first 3 batches ---")
with torch.no_grad():
    for i, (fragments, gt_mos) in enumerate(val_loader):
        feats = backbone.extract_features(fragments)
        print(f"\nBatch {i}: feature stats — mean={feats.mean():.4f} std={feats.std():.4f} "
              f"min={feats.min():.4f} max={feats.max():.4f}")
        print(f"  feature is all-zero: {(feats == 0).all().item()}")

        scores, log_var, _ = policy(feats)
        print(f"  predicted scores: {scores.tolist()}")
        print(f"  gt MOS:           {gt_mos.tolist()}")

        if i >= 2:
            break

# Check how many videos actually exist on disk
import pandas as pd
df = pd.read_csv("data/val.csv")
exists = df["video_path"].apply(os.path.exists)
print(f"\n--- Val CSV check ---")
print(f"Total rows: {len(df)}")
print(f"Videos that exist on disk: {exists.sum()}")
print(f"Videos MISSING (will get zero-tensor fallback): {(~exists).sum()}")