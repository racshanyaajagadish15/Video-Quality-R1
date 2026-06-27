# reward.py
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr, pearsonr
import numpy as np

class VideoQualityReward:
    """
    Multi-component reward for VideoQuality-R1 GRPO training.

    R_total = w_mos * R_mos + w_rank * R_rank + w_fmt * R_fmt - w_kl * KL_penalty
    """

    def __init__(
        self,
        w_mos: float = 0.5,    # MOS correlation weight
        w_rank: float = 0.3,   # ranking accuracy weight
        w_fmt: float = 0.1,    # format/uncertainty penalty weight
        w_kl: float = 0.1,     # KL divergence penalty weight
        kl_coef: float = 0.04, # beta in GRPO — controls KL strength
    ):
        self.w_mos = w_mos
        self.w_rank = w_rank
        self.w_fmt = w_fmt
        self.w_kl = w_kl
        self.kl_coef = kl_coef

    def mos_reward(
        self,
        pred_scores: torch.Tensor,   # (B,) predicted MOS
        gt_scores: torch.Tensor,     # (B,) ground-truth MOS
    ) -> torch.Tensor:
        """PLCC-based reward: high Pearson correlation → high reward."""
        pred_np = pred_scores.detach().cpu().numpy()
        gt_np = gt_scores.cpu().numpy()

        if len(pred_np) < 3:
            # Fallback: negative MSE for tiny batches
            mse = F.mse_loss(pred_scores, gt_scores.to(pred_scores.device))
            return torch.ones(len(pred_np), device=pred_scores.device) * (1.0 - mse.item())

        plcc, _ = pearsonr(pred_np, gt_np)
        srcc, _ = spearmanr(pred_np, gt_np)
        correlation = 0.5 * (plcc + srcc)  # combined metric

        # Per-sample reward: combine correlation with local accuracy
        local_err = torch.abs(pred_scores - gt_scores.to(pred_scores.device))
        local_reward = torch.clamp(1.0 - local_err / 4.0, 0.0, 1.0)  # [0,1]

        # Scale: correlation boosts or penalises the whole batch
        reward = local_reward * (0.5 + 0.5 * max(correlation, 0.0))
        return reward  # (B,)

    def ranking_reward(
        self,
        pred_scores: torch.Tensor,   # (B,)
        gt_scores: torch.Tensor,     # (B,)
    ) -> torch.Tensor:
        """Pairwise ranking reward: correct ordering → +1, wrong → -1."""
        B = len(pred_scores)
        if B < 2:
            return torch.zeros(B, device=pred_scores.device)

        gt = gt_scores.to(pred_scores.device)
        pred_diff = pred_scores.unsqueeze(1) - pred_scores.unsqueeze(0)   # (B,B)
        gt_diff = gt.unsqueeze(1) - gt.unsqueeze(0)                        # (B,B)

        # correct pair if signs match (or both near-zero)
        correct = (torch.sign(pred_diff) == torch.sign(gt_diff)).float()
        mask = (gt_diff.abs() > 0.1).float()  # ignore near-ties
        pair_accuracy = (correct * mask).sum(1) / (mask.sum(1) + 1e-6)
        return 2.0 * pair_accuracy - 1.0  # scale to [-1, 1]

    def format_reward(
        self,
        log_var: torch.Tensor,  # (B,) log-variance from policy head
    ) -> torch.Tensor:
        """Penalise overconfidence or extreme uncertainty."""
        # Ideal log_var is in [-2, 2]; penalise outside that range
        penalty = torch.clamp(log_var.abs() - 2.0, min=0.0)
        return -penalty  # (B,)

    def kl_penalty(
        self,
        policy_scores: torch.Tensor,   # (B,) from trainable head
        ref_scores: torch.Tensor,       # (B,) from frozen reference head
    ) -> torch.Tensor:
        """
        Soft KL penalty between policy and reference outputs.
        Encourages the policy not to drift too far from supervised pretraining.
        """
        diff = (policy_scores - ref_scores.to(policy_scores.device)).abs()
        return self.kl_coef * diff  # (B,)

    def compute(
        self,
        pred_scores,
        gt_scores,
        log_var,
        ref_scores,
    ) -> dict:
        """Returns total reward + per-component breakdown."""
        r_mos = self.mos_reward(pred_scores, gt_scores)
        r_rank = self.ranking_reward(pred_scores, gt_scores)
        r_fmt = self.format_reward(log_var)
        r_kl = self.kl_penalty(pred_scores, ref_scores)

        total = (
            self.w_mos * r_mos
            + self.w_rank * r_rank
            + self.w_fmt * r_fmt
            - self.w_kl * r_kl
        )
        return {
            "total": total,         # (B,)
            "mos": r_mos,
            "rank": r_rank,
            "format": r_fmt,
            "kl_pen": r_kl,
        }