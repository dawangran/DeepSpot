#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepspot.stage1_data import load_stage1_arrays, split_indices_by_read_id  # noqa: E402
from deepspot.stage1_pca import fit_pca_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 1 baseline: train dependency-light PCA reconstruction model on unmodified windows."
    )
    parser.add_argument("--train_npz", nargs="+", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--n_components", type=int, default=32)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--raw_weight", type=float, default=1.0)
    parser.add_argument("--event_weight", type=float, default=0.35)
    parser.add_argument("--mask_weight", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arrays = load_stage1_arrays([Path(p) for p in args.train_npz])
    train_idx, val_idx = split_indices_by_read_id(arrays.read_ids, args.val_ratio, args.seed)

    model = fit_pca_model(
        arrays=arrays,
        train_idx=train_idx,
        val_idx=val_idx,
        n_components=args.n_components,
        raw_weight=args.raw_weight,
        event_weight=args.event_weight,
        mask_weight=args.mask_weight,
    )
    model_path = out_dir / "stage1_pca_model.npz"
    np.savez(
        model_path,
        feature_mean=model["feature_mean"],
        components=model["components"],
        singular_values=model["singular_values"],
        event_mean=model["event_mean"],
        event_std=model["event_std"],
        val_scores=model["val_scores"],
        config_json=np.asarray(
            [
                json.dumps(
                    {
                        "train_npz": args.train_npz,
                        "n_train_windows": int(len(train_idx)),
                        "n_val_windows": int(len(val_idx)),
                        "raw_weight": model["raw_weight"],
                        "event_weight": model["event_weight"],
                        "mask_weight": model["mask_weight"],
                        "window_len": model["window_len"],
                        "max_events": model["max_events"],
                        "event_dim": model["event_dim"],
                        "n_components": model["n_components"],
                        "val_score_mean": model["val_score_mean"],
                        "val_score_std": model["val_score_std"],
                        "val_score_p95": model["val_score_p95"],
                        "val_score_p99": model["val_score_p99"],
                        "seed": args.seed,
                    },
                    ensure_ascii=False,
                )
            ],
            dtype=object,
        ),
    )

    summary_path = out_dir / "stage1_pca_train_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "model_path": str(model_path),
                "train_npz": args.train_npz,
                "n_train_windows": int(len(train_idx)),
                "n_val_windows": int(len(val_idx)),
                "n_components": model["n_components"],
                "val_score_mean": model["val_score_mean"],
                "val_score_std": model["val_score_std"],
                "val_score_p95": model["val_score_p95"],
                "val_score_p99": model["val_score_p99"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("[DONE] trained Stage1 PCA baseline")
    print("  model:  ", model_path)
    print("  summary:", summary_path)
    print("  val p95:", model["val_score_p95"], "val p99:", model["val_score_p99"])


if __name__ == "__main__":
    main()
