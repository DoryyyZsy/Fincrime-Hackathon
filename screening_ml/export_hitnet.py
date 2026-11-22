"""Export a default-architecture HitNet checkpoint for the standard-library runner."""
import argparse
import json
from pathlib import Path

from .common import load_checkpoint
from .compare import DECISIONS, REASON_CODES
from .features import FEATURE_NAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    network, checkpoint = load_checkpoint(args.checkpoint, "cpu")
    cfg = checkpoint["config"]
    if (list(cfg["model"]["hidden"]) != [128, 64] or cfg.get("drop_features")
            or checkpoint["feature_names"] != FEATURE_NAMES or cfg["inference"]["use_cost_matrix"]):
        parser.error("Checkpoint does not match the deployed HitNet architecture or decoder")
    payload = {"features": FEATURE_NAMES, "decisions": DECISIONS, "codes": REASON_CODES,
               "mean": checkpoint["mean"].tolist(), "std": checkpoint["std"].tolist(),
               "clear_min_prob": cfg["inference"]["clear_min_prob"],
               "weights": {k: v.tolist() for k, v in network.state_dict().items()}}
    args.output.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
