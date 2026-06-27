# train_rl.py  — main entry point
import copy
import torch
from torch.utils.data import DataLoader
from backbone import FrozenBackbone
from policy_head import PolicyHead
from reward import VideoQualityReward
from grpo_trainer import GRPOTrainer
from dataset import VQADataset, vqa_collate_fn


def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # AFTER — absolute paths resolved from repo root
    import os
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # train_r1.py — fix the PRETRAINED path to match the real filename
    PRETRAINED = os.path.join(REPO_ROOT, "pretrained", "FAST_VQA_B_1_4.pth")
    OPTIONS_PATH = os.path.join(REPO_ROOT, "options", "fast", "fast-b.yml")
    TRAIN_CSV = os.path.join(os.path.dirname(__file__), "data", "train.csv")
    TRAIN_CSV = "data/train.csv"
    EPOCHS = 30
    BATCH_SIZE = 8
    G = 8  # group size for GRPO

    # --- Step 1: Pretrain policy head with SFT (supervised) first ---
    # (do a standard regression training pass before RL to warm-start)
    # ... (standard MSE training loop here, ~10 epochs) ...

    # --- Step 2: RL fine-tuning with GRPO ---
    backbone = FrozenBackbone(PRETRAINED, options_path=OPTIONS_PATH, device=DEVICE)
    policy_head = PolicyHead(feature_dim=768, hidden_dim=256)

    # Reference head: frozen copy of SFT-initialised policy head
    ref_head = copy.deepcopy(policy_head)
    for p in ref_head.parameters():
        p.requires_grad = False

    reward_fn = VideoQualityReward(
        w_mos=0.5, w_rank=0.3, w_fmt=0.1, w_kl=0.1, kl_coef=0.04
    )

    trainer = GRPOTrainer(
        backbone=backbone,
        policy_head=policy_head,
        ref_head=ref_head,
        reward_fn=reward_fn,
        G=G,
        lr=1e-4,
        device=DEVICE,
    )

    dataset = VQADataset(TRAIN_CSV)
    loader  = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        collate_fn=vqa_collate_fn,   # ← add this
    )

    for epoch in range(EPOCHS):
        epoch_loss, epoch_reward = 0, 0
        for step, (fragments, gt_mos) in enumerate(loader):
            # Move fragments to device
            for k in fragments:
                fragments[k] = fragments[k].to(DEVICE)

            metrics = trainer.train_step(fragments, gt_mos)
            epoch_loss += metrics["loss"]
            epoch_reward += metrics["mean_reward"]

            if step % 20 == 0:
                print(
                    f"Epoch {epoch} | Step {step} | "
                    f"Loss {metrics['loss']:.4f} | "
                    f"Reward {metrics['mean_reward']:.4f} | "
                    f"MOS-R {metrics['mean_mos_reward']:.4f} | "
                    f"KL-pen {metrics['kl_pen']:.4f}"
                )

        print(f"=== Epoch {epoch} | Avg Loss {epoch_loss/len(loader):.4f} ===")

        # Save checkpoint
        torch.save(policy_head.state_dict(), f"ckpts/policy_epoch{epoch}.pth")

if __name__ == "__main__":
    main()