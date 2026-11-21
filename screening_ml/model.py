"""The network, device selection, determinism, and the decision rule used at inference."""
import os
import random

import numpy as np
import torch
from torch import nn

from .compare import CODE_DECISION, DECISIONS, REASON_CODES

# official cost matrix (BRIEF.md section 6): COST[true][predicted]
COST = torch.tensor([
    [0.0, 3.0, 3.0],     # true DIFFERENT_ENTITY
    [60.0, 0.0, 2.0],    # true SAME_ENTITY
    [60.0, 2.0, 0.0],    # true REVIEW
])
# which reason codes belong to which decision (mask for the reason head)
CODE_MASK = torch.tensor([[CODE_DECISION[c] == d for c in REASON_CODES] for d in DECISIONS])


def pick_device(name="auto"):
    """cuda (Windows/Linux NVIDIA) -> mps (Apple Silicon) -> cpu."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


class HitNet(nn.Module):
    """MLP backbone with two heads: decision (3 classes) and reason code (11 classes)."""

    def __init__(self, n_features, hidden=(128, 64), dropout=0.2):
        super().__init__()
        layers, width = [], n_features
        for h in hidden:
            layers += [nn.Linear(width, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            width = h
        self.backbone = nn.Sequential(*layers)
        self.decision_head = nn.Linear(width, len(DECISIONS))
        self.reason_head = nn.Linear(width, len(REASON_CODES))

    def forward(self, x):
        z = self.backbone(x)
        return self.decision_head(z), self.reason_head(z)


def loss_fn(dec_logits, code_logits, y_dec, y_code, cfg, class_weights=None):
    """Cross-entropy on both heads plus an expected-cost term that punishes unsafe clearances."""
    ce = nn.functional.cross_entropy(dec_logits, y_dec, weight=class_weights,
                                     label_smoothing=cfg.get("label_smoothing", 0.0))
    ce_code = nn.functional.cross_entropy(code_logits, y_code)
    probs = dec_logits.softmax(-1)
    expected_cost = (probs * COST.to(probs.device)[y_dec]).sum(-1).mean()
    return ce + cfg.get("reason_weight", 0.5) * ce_code + cfg.get("cost_weight", 0.05) * expected_cost


@torch.no_grad()
def decide(dec_logits, code_logits, cfg):
    """Turn logits into decisions: minimise expected cost, and only clear when confident enough."""
    probs = dec_logits.softmax(-1)
    if cfg.get("use_cost_matrix", True):
        exp_cost = probs @ COST.to(probs.device)          # cost of each possible prediction
        dec = exp_cost.argmin(-1)
    else:
        dec = probs.argmax(-1)
    diff = DECISIONS.index("DIFFERENT_ENTITY")
    unsafe = (dec == diff) & (probs[:, diff] < cfg.get("clear_min_prob", 0.9))
    dec = torch.where(unsafe, torch.full_like(dec, DECISIONS.index("REVIEW")), dec)
    mask = CODE_MASK.to(code_logits.device)[dec]
    code = code_logits.masked_fill(~mask, float("-inf")).argmax(-1)
    return dec, code, probs
