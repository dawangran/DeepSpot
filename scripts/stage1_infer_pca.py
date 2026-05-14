#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepspot.stage1_data import load_stage1_arrays  # noqa: E402
from deepspot.stage1_pca import score_pca_model  # noqa: E402


def load_pca_model(path: str | Path) -> dict:
    data = np.load(path, allow_pickle=True)
    config = json.loads(str(data["config_json"][0]))
    return {
        "feature_mean": data["feature_mean"].astype(np.float32),
        "components": data["components"].astype(np.float32),
        "event_mean": data["event_mean"].astype(np.float32),
        "event_std": data["event_std"].astype(np.float32),
        "raw_weight": float(config["raw_weight"]),
        "event_weight": float(config["event_weight"]),
        "mask_weight": float(config["mask_weight"]),
        "window_len": int(config["window_len"]),
        "max_events": int(config["max_events"]),
        "event_dim": int(config["event_dim"]),
        "n_components": int(config["n_components"]),
        "val_score_mean": float(config["val_score_mean"]),
        "val_score_std": float(config["val_score_std"]),
        "val_score_p95": float(config["val_score_p95"]),
        "val_score_p99": float(config["val_score_p99"]),
    }


def write_tsv(path: Path, arrays, scores: dict) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "row_index",
                "sample_id",
                "condition",
                "read_id",
                "window_start",
                "num_events",
                "label",
                "mod_type",
                "total_score",
                "anomaly_z",
            ]
        )
        for i in range(len(scores["score"])):
            writer.writerow(
                [
                    i,
                    str(arrays.sample_ids[i]),
                    str(arrays.conditions[i]),
                    str(arrays.read_ids[i]),
                    int(arrays.window_starts[i]),
                    int(arrays.event_mask[i].sum()),
                    str(arrays.labels[i]),
                    str(arrays.mod_types[i]),
                    float(scores["score"][i]),
                    float(scores["anomaly_z"][i]),
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1 baseline: infer PCA reconstruction anomaly scores.")
    parser.add_argument("--input_npz", nargs="+", required=True)
    parser.add_argument("--model_npz", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--prefix", default="stage1_pca")
    parser.add_argument("--write_tsv", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arrays = load_stage1_arrays([Path(p) for p in args.input_npz])
    model = load_pca_model(args.model_npz)
    scores = score_pca_model(arrays, model)

    out_npz = out_dir / f"{args.prefix}.stage1_pca_scores.npz"
    np.savez(
        out_npz,
        total_score=scores["score"].astype(np.float32),
        anomaly_z=scores["anomaly_z"].astype(np.float32),
        latent=scores["latent"].astype(np.float32),
        raw_residual=scores["raw_residual"].astype(np.float32),
        read_ids=arrays.read_ids,
        sample_ids=arrays.sample_ids,
        conditions=arrays.conditions,
        labels=arrays.labels,
        mod_types=arrays.mod_types,
        window_starts=arrays.window_starts,
        num_events=arrays.event_mask.sum(axis=1).astype(np.int32),
    )

    summary = {
        "input_npz": args.input_npz,
        "model_npz": args.model_npz,
        "n_windows": int(len(scores["score"])),
        "score_mean": float(scores["score"].mean()),
        "score_median": float(np.median(scores["score"])),
        "score_p95": float(np.percentile(scores["score"], 95)),
        "score_p99": float(np.percentile(scores["score"], 99)),
        "anomaly_z_p95": float(np.percentile(scores["anomaly_z"], 95)),
        "anomaly_z_p99": float(np.percentile(scores["anomaly_z"], 99)),
        "calibration": {
            "val_score_mean": model["val_score_mean"],
            "val_score_std": model["val_score_std"],
            "val_score_p95": model["val_score_p95"],
            "val_score_p99": model["val_score_p99"],
        },
    }
    out_summary = out_dir / f"{args.prefix}.stage1_pca_infer_summary.json"
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.write_tsv:
        write_tsv(out_dir / f"{args.prefix}.stage1_pca_scores.tsv", arrays, scores)

    print("[DONE] Stage1 PCA inference saved")
    print("  scores: ", out_npz)
    print("  summary:", out_summary)
    if args.write_tsv:
        print("  tsv:    ", out_dir / f"{args.prefix}.stage1_pca_scores.tsv")


if __name__ == "__main__":
    main()
