"""
backbone.py — FrozenBackbone for VideoQuality-R1
Wraps FastVQA's DiViDeAddEvaluator with all weights frozen.

DiViDeAddEvaluator real signature (from evaluator.py):
    def __init__(
        self,
        backbone_size="divided",
        backbone_preserve_keys='fragments,resize',
        multi=False,
        layer=-1,
        backbone=dict(resize={"window_size":(4,4,4)}, fragments={"window_size":(4,4,4)}),
        divide_head=False,
        vqa_head=dict(in_channels=768),
        var=False,
    )
When backbone_size=="divided", it reads hypers["type"] from each sub-dict in backbone.
"""

import os
import sys
import warnings
import torch
import torch.nn as nn
import yaml

warnings.filterwarnings("ignore", category=FutureWarning, module="timm")

# ---------------------------------------------------------------------------
# Repo root = one level above this file  (videoquality_r1/../)
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

# ---------------------------------------------------------------------------
# Valid backbone type strings accepted by DiViDeAddEvaluator
# ---------------------------------------------------------------------------
VALID_BACKBONE_TYPES = {
    "swin_tiny",
    "swin_tiny_grpb",   # FastVQA default
    "swin_tiny_grpb_m", # FastVQA-M
    "swin_small",
    "conv_tiny",
    "conv_small",
    "xclip",
}
DEFAULT_BACKBONE_TYPE = "swin_tiny_grpb"


def _parse_backbone_dict(yaml_backbone, default_type: str) -> dict:
    """
    Normalise the 'backbone' block from the YAML into the structure
    DiViDeAddEvaluator expects:

        {
          "resize":    {"type": "<backbone_type>", "window_size": (4,4,4), ...},
          "fragments": {"type": "<backbone_type>", "window_size": (4,4,4), ...},
        }

    Handles three YAML layouts:
      1. Already correct  — backbone: {resize: {type: ..., ...}, fragments: {...}}
      2. Flat with type   — backbone: {type: swin_tiny_grpb, window_size: ...}
      3. Missing entirely — build from scratch with defaults
    """
    default_sub = {"type": default_type, "window_size": (4, 4, 4)}

    if yaml_backbone is None:
        return {"resize": dict(default_sub), "fragments": dict(default_sub)}

    if not isinstance(yaml_backbone, dict):
        raise ValueError(f"backbone in YAML must be a dict, got {type(yaml_backbone)}")

    # Layout 1: already has sub-keys resize / fragments
    if "resize" in yaml_backbone or "fragments" in yaml_backbone:
        out = {}
        for key in ("resize", "fragments"):
            sub = yaml_backbone.get(key, dict(default_sub))
            if not isinstance(sub, dict):
                sub = dict(default_sub)
            if "type" not in sub:
                sub["type"] = default_type
            out[key] = sub
        return out

    # Layout 2: flat dict — treat it as shared config for both keys
    flat = dict(yaml_backbone)
    if "type" not in flat:
        flat["type"] = default_type
    return {"resize": dict(flat), "fragments": dict(flat)}


def _load_yaml_cfg(options_path: str) -> dict:
    """Load YAML and unwrap common nesting layers."""
    with open(options_path, "r") as f:
        opt = yaml.safe_load(f)

    # Some configs nest everything under 'model' or 'args'
    cfg = opt.get("model", opt)
    cfg = cfg.get("args", cfg)
    return cfg


def build_evaluator(options_path: str) -> nn.Module:
    """
    Construct DiViDeAddEvaluator with kwargs that match its real __init__.
    Reads the YAML but always passes explicit, validated arguments —
    never tries to pass a raw 'hypers' keyword that doesn't exist.
    """
    from fastvqa.models import DiViDeAddEvaluator

    cfg = _load_yaml_cfg(options_path)
    print(f"[Backbone] YAML cfg keys: {list(cfg.keys())}")

    # --- backbone_size ---
    # When "divided", __init__ reads hypers["type"] per sub-dict.
    # Anything else is used as a global type string.
    backbone_size = cfg.get("backbone_size", "divided")

    # --- resolve the per-key backbone type ---
    # The YAML might store it as backbone_size or inside backbone sub-dicts.
    # We resolve a single default_type to embed inside each sub-dict.
    yaml_backbone = cfg.get("backbone", None)

    if backbone_size != "divided":
        # backbone_size IS the type string; no per-key lookup needed
        default_type = backbone_size
    else:
        # Try to infer type from yaml_backbone or fall back to FastVQA default
        if isinstance(yaml_backbone, dict):
            # Could be flat {"type": "swin_tiny_grpb", ...} or nested
            if "type" in yaml_backbone:
                default_type = yaml_backbone["type"]
            elif "resize" in yaml_backbone and "type" in yaml_backbone["resize"]:
                default_type = yaml_backbone["resize"]["type"]
            elif "fragments" in yaml_backbone and "type" in yaml_backbone["fragments"]:
                default_type = yaml_backbone["fragments"]["type"]
            else:
                default_type = DEFAULT_BACKBONE_TYPE
        else:
            default_type = DEFAULT_BACKBONE_TYPE

    if default_type not in VALID_BACKBONE_TYPES:
        print(
            f"[Backbone] WARNING: backbone type {default_type!r} not in known list "
            f"{VALID_BACKBONE_TYPES}. Falling back to {DEFAULT_BACKBONE_TYPE!r}."
        )
        default_type = DEFAULT_BACKBONE_TYPE

    backbone_dict = _parse_backbone_dict(yaml_backbone, default_type)

    # --- other constructor params ---
    backbone_preserve_keys = cfg.get("backbone_preserve_keys", "fragments,resize")
    multi                  = cfg.get("multi", False)
    layer                  = cfg.get("layer", -1)
    divide_head            = cfg.get("divide_head", False)
    vqa_head               = cfg.get("vqa_head", {"in_channels": 768})
    var                    = cfg.get("var", False)

    print(f"[Backbone] backbone_size          = {backbone_size!r}")
    print(f"[Backbone] resolved backbone type  = {default_type!r}")
    print(f"[Backbone] backbone keys           = {list(backbone_dict.keys())}")
    for k, v in backbone_dict.items():
        print(f"[Backbone]   {k}: {v}")

    # Pass each arg by name — matches the real __init__ exactly
    model = DiViDeAddEvaluator(
        backbone_size=backbone_size,
        backbone_preserve_keys=backbone_preserve_keys,
        multi=multi,
        layer=layer,
        backbone=backbone_dict,
        divide_head=divide_head,
        vqa_head=vqa_head,
        var=var,
    )
    return model


def _attach_feature_hook(model: nn.Module, feature_dim: int):
    """
    Attach a forward hook to capture the feature vector just before
    the final VQAHead regression layer.

    Returns (hook_container, hook_handle) where hook_container is a
    one-element list so the closure can mutate it.
    """
    feature_store = [None]

    def _hook(module, inp, out):
        # Prefer the input to the head (richer features) over its output (scalar)
        if inp and isinstance(inp[0], torch.Tensor):
            feature_store[0] = inp[0].detach()
        else:
            feature_store[0] = out.detach()

    vqa_head = model.vqa_head

    if isinstance(vqa_head, nn.Sequential):
        # Hook the first child so we capture pre-relu / pre-linear features
        first = list(vqa_head.children())[0]
        handle = first.register_forward_hook(_hook)
        print(f"[Backbone] Hook attached to vqa_head[0]: {type(first).__name__}")
    elif hasattr(vqa_head, '__call__'):
        handle = vqa_head.register_forward_hook(_hook)
        print(f"[Backbone] Hook attached to vqa_head: {type(vqa_head).__name__}")
    else:
        raise RuntimeError(
            f"Cannot attach hook — vqa_head type {type(vqa_head)} not supported."
        )

    return feature_store, handle


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class FrozenBackbone(nn.Module):
    """
    FastVQA/DOVER backbone with all weights frozen.
    Only used as a feature extractor for VideoQuality-R1's policy head.

    Args:
        pretrained_path : path to FastVQA .pth checkpoint
        options_path    : path to fast-b.yml (or any DiViDeAdd options YAML)
                          defaults to <repo_root>/options/fast/fast-b.yml
        device          : 'cuda' or 'cpu'
    """

    def __init__(
        self,
        pretrained_path: str,
        options_path: str = None,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device

        # ── Resolve paths ──────────────────────────────────────────────────
        if options_path is None:
            options_path = os.path.join(REPO_ROOT, "options", "fast", "fast-b.yml")
        if not os.path.isabs(options_path):
            options_path = os.path.join(REPO_ROOT, options_path)

        print(f"[Backbone] options  : {options_path}")
        print(f"[Backbone] weights  : {pretrained_path}")

        if not os.path.exists(options_path):
            raise FileNotFoundError(
                f"Options file not found: {options_path}\n"
                f"Tip: pass options_path= explicitly, or check your repo structure."
            )
        if not os.path.exists(pretrained_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {pretrained_path}"
            )

        # ── Build model ────────────────────────────────────────────────────
        self.model = build_evaluator(options_path)

        # ── Load pretrained weights ────────────────────────────────────────
        ckpt = torch.load(pretrained_path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        print(f"[Backbone] Weights loaded — missing: {len(missing)}, unexpected: {len(unexpected)}")
        if missing:
            print(f"[Backbone] First 5 missing : {missing[:5]}")
        if unexpected:
            print(f"[Backbone] First 5 unexpected: {unexpected[:5]}")

        # ── Freeze all parameters ──────────────────────────────────────────
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        # ── Feature hook ───────────────────────────────────────────────────
        self.feature_dim = 768
        self._feature_store, self._hook_handle = _attach_feature_hook(
            self.model, self.feature_dim
        )

        self.model.to(device)
        print("[Backbone] Ready — all weights frozen.")

    # ── Forward ────────────────────────────────────────────────────────────
    @torch.no_grad()
    def extract_features(self, video_fragments: dict) -> torch.Tensor:
        """
        Only forward the keys that backbone_preserve_keys actually built.
        E.g. if YAML has backbone_preserve_keys='fragments', only pass 'fragments'.
        """
        self._feature_store[0] = None

        # Discover which backbone attributes were actually created
        built_keys = [
            k for k in ("fragments", "resize")
            if hasattr(self.model, f"{k}_backbone")
        ]

        if not built_keys:
            raise RuntimeError(
                "No backbone attributes found on model "
                "(expected fragments_backbone or resize_backbone)."
            )

        # Filter fragments dict to only the keys the model knows about
        fragments = {
            k: v.to(self.device)
            for k, v in video_fragments.items()
            if k in built_keys
        }

        _ = self.model(fragments)

        features = self._feature_store[0]
        if features is None:
            raise RuntimeError("Feature hook did not fire.")

        if features.dim() > 2:
            features = features.mean(dim=list(range(2, features.dim())))

        return features  # (B, feature_dim)

    def __del__(self):
        # Clean up hook to avoid memory leaks in long training runs
        if hasattr(self, "_hook_handle"):
            self._hook_handle.remove()