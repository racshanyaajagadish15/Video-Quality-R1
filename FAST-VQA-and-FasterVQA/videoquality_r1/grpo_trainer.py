# grpo_trainer.py
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

class GRPOTrainer:
    """
    Group Relative Policy Optimization for VideoQuality-R1.

    Key idea: for each video, generate G score proposals from the policy,
    compute rewards for all G, use group-relative advantage to update.
    """

    def __init__(
        self,
        backbone,           # FrozenBackbone
        policy_head,        # PolicyHead (trainable)
        ref_head,           # PolicyHead (frozen reference copy)
        reward_fn,          # VideoQualityReward
        G: int = 8,         # group size — samples per video
        lr: float = 1e-4,
        max_grad_norm: float = 1.0,
        device: str = "cuda",
    ):
        self.backbone = backbone.to(device)
        self.policy = policy_head.to(device)
        self.ref = ref_head.to(device)
        self.reward_fn = reward_fn
        self.G = G
        self.device = device
        self.max_grad_norm = max_grad_norm

        # Only the policy head is updated
        self.optimizer = AdamW(self.policy.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=1000, eta_min=1e-6)

        # Freeze reference head
        for p in self.ref.parameters():
            p.requires_grad = False

    def _sample_group(self, features: torch.Tensor):
        """
        For each video in the batch, draw G score samples by adding
        Gaussian noise to the deterministic policy output.

        features: (B, D)
        Returns:
            all_scores: (B, G) — noisy score samples
            log_var:    (B,)   — uncertainty estimate
            hidden:     (B, D_h)
        """
        scores, log_var, hidden = self.policy(features)  # (B,), (B,), (B,H)
        std = torch.exp(0.5 * log_var).unsqueeze(1)       # (B, 1)

        # Expand base scores and add noise for G samples
        scores_expanded = scores.unsqueeze(1).expand(-1, self.G)  # (B, G)
        noise = torch.randn_like(scores_expanded) * std
        samples = torch.clamp(scores_expanded + noise, 1.0, 5.0)   # (B, G)

        return samples, log_var, hidden

    def _compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        rewards: (B, G)
        Group-relative advantage: normalise within each video's group.
        A_i = (r_i - mean(r_group)) / (std(r_group) + eps)
        """
        mean = rewards.mean(dim=1, keepdim=True)   # (B, 1)
        std = rewards.std(dim=1, keepdim=True)     # (B, 1)
        return (rewards - mean) / (std + 1e-8)     # (B, G)

    def train_step(self, video_fragments: dict, gt_scores: torch.Tensor):
        """
        One GRPO training step.

        video_fragments: dict of fragment tensors (from dataset)
        gt_scores:       (B,) ground-truth MOS labels
        """
        gt_scores = gt_scores.to(self.device)

        # 1. Extract features (no grad — backbone is frozen)
        with torch.no_grad():
            features = self.backbone.extract_features(video_fragments)  # (B, D)

        # 2. Reference head scores (no grad)
        with torch.no_grad():
            ref_scores, _, _ = self.ref(features)  # (B,)

        # 3. Sample G score proposals from policy
        self.optimizer.zero_grad()
        sample_scores, log_var, _ = self._sample_group(features)  # (B, G), (B,)

        # 4. Compute per-sample rewards
        # Expand gt_scores and ref_scores to match (B*G,)
        B, G = sample_scores.shape
        flat_pred = sample_scores.reshape(B * G)                    # (B*G,)
        flat_gt = gt_scores.unsqueeze(1).expand(B, G).reshape(B * G)
        flat_ref = ref_scores.unsqueeze(1).expand(B, G).reshape(B * G)
        flat_logvar = log_var.unsqueeze(1).expand(B, G).reshape(B * G)

        reward_dict = self.reward_fn.compute(
            flat_pred, flat_gt, flat_logvar, flat_ref
        )
        rewards = reward_dict["total"].reshape(B, G)  # (B, G)

        # 5. GRPO advantage
        advantages = self._compute_advantages(rewards)  # (B, G)

        # 6. Policy gradient loss
        # L = -mean(advantage * log_prob(score))
        # Approximate log_prob via Gaussian log-likelihood
        mean_scores = sample_scores.mean(dim=1, keepdim=True)  # (B,1)
        std_scores = torch.exp(0.5 * log_var).unsqueeze(1)      # (B,1)
        log_probs = (
            -0.5 * ((sample_scores - mean_scores) / (std_scores + 1e-6)) ** 2
            - torch.log(std_scores + 1e-6)
            - 0.5 * torch.log(torch.tensor(2 * 3.14159))
        )  # (B, G)

        policy_loss = -(advantages.detach() * log_probs).mean()

        # 7. Backprop through policy head only
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()

        return {
            "loss": policy_loss.item(),
            "mean_reward": rewards.mean().item(),
            "mean_mos_reward": reward_dict["mos"].mean().item(),
            "mean_rank_reward": reward_dict["rank"].mean().item(),
            "kl_pen": reward_dict["kl_pen"].mean().item(),
        }