#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepspot.stage1_data import load_stage1_arrays  # noqa: E402
from deepspot.stage1_model import (  # noqa: E402
    RawEventAutoencoder,
    Stage1WindowDataset,
    reconstruction_losses,
)


@torch.no_grad()
def infer(model, loader, device, event_weight: float):
    model.eval()
    all_index = []
    all_raw_mse = []
    all_event_mse = []
    all_total = []
    all_latent = []
    all_raw_residual = []

    for batch in loader:
        raw = batch["raw"].to(device)
        events = batch["events"].to(device)
        mask = batch["mask"].to(device)
        raw_hat, event_hat, z = model(raw, events, mask)
        raw_mse, event_mse = reconstruction_losses(raw, raw_hat, events, event_hat, mask)
        total = raw_mse + event_weight * event_mse

        all_index.append(batch["index"].cpu().numpy())
        all_raw_mse.append(raw_mse.cpu().numpy())
        all_event_mse.append(event_mse.cpu().numpy())
        all_total.append(total.cpu().numpy())
        all_latent.append(z.cpu().numpy())
        all_raw_residual.append((raw - raw_hat).cpu().numpy()[:, 0, :])

    return {
        "index": np.concatenate(all_index, axis=0).astype(np.int64),
        "raw_mse": np.concatenate(all_raw_mse, axis=0).astype(np.float32),
        "event_mse": np.concatenate(all_event_mse, axis=0).astype(np.float32),
        "total_score": np.concatenate(all_total, axis=0).astype(np.float32),
        "latent": np.concatenate(all_latent, axis=0).astype(np.float32),
        "raw_residual": np.concatenate(all_raw_residual, axis=0).astype(np.float32),
    }


def write_tsv(path: Path, arrays, scores, anomaly_z: np.ndarray) -> None:
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
                "raw_mse",
                "event_mse",
                "total_score",
                "anomaly_z",
            ]
        )
        for out_i, row_idx in enumerate(scores["index"]):
            n_events = int(arrays.event_mask[row_idx].sum())
            writer.writerow(
                [
                    int(row_idx),
                    str(arrays.sample_ids[row_idx]),
                    str(arrays.conditions[row_idx]),
                    str(arrays.read_ids[row_idx]),
                    int(arrays.window_starts[row_idx]),
                    n_events,
                    str(arrays.labels[row_idx]),
                    str(arrays.mod_types[row_idx]),
                    float(scores["raw_mse"][out_i]),
                    float(scores["event_mse"][out_i]),
                    float(scores["total_score"][out_i]),
                    float(anomaly_z[out_i]),
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 1: infer anomaly scores for raw/event windows."
    )
    parser.add_argument("--input_npz", nargs="+", required=True, help="One or more *.stage1_windows.npz files.")
    parser.add_argument("--model_ckpt", required=True, help="stage1_autoencoder_best.pt")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--prefix", default="stage1_infer")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--write_tsv", action="store_true", help="Also write a window-level TSV table.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.model_ckpt, map_location=device)
    config = ckpt["model_config"]
    event_mean = ckpt["event_mean"]
    event_std = ckpt["event_std"]
    event_weight = float(ckpt.get("event_weight", 0.35))

    arrays = load_stage1_arrays([Path(p) for p in args.input_npz])
    ds = Stage1WindowDataset(arrays, event_mean=event_mean, event_std=event_std)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = RawEventAutoencoder(**config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    scores = infer(model, loader, device, event_weight)
    val_stats = ckpt.get("val_score_stats", {})
    score_mean = float(val_stats.get("mean", scores["total_score"].mean()))
    score_std = float(max(val_stats.get("std", scores["total_score"].std()), 1e-8))
    anomaly_z = ((scores["total_score"] - score_mean) / score_std).astype(np.float32)

    out_npz = out_dir / f"{args.prefix}.stage1_scores.npz"
    np.savez(
        out_npz,
        row_index=scores["index"],
        raw_mse=scores["raw_mse"],
        event_mse=scores["event_mse"],
        total_score=scores["total_score"],
        anomaly_z=anomaly_z,
        latent=scores["latent"],
        raw_residual=scores["raw_residual"],
        read_ids=arrays.read_ids[scores["index"]],
        sample_ids=arrays.sample_ids[scores["index"]],
        conditions=arrays.conditions[scores["index"]],
        labels=arrays.labels[scores["index"]],
        mod_types=arrays.mod_types[scores["index"]],
        window_starts=arrays.window_starts[scores["index"]],
        num_events=arrays.event_mask[scores["index"]].sum(axis=1).astype(np.int32),
    )

    summary = {
        "input_npz": args.input_npz,
        "model_ckpt": args.model_ckpt,
        "n_windows": int(len(scores["total_score"])),
        "score_mean": float(scores["total_score"].mean()),
        "score_median": float(np.median(scores["total_score"])),
        "score_p95": float(np.percentile(scores["total_score"], 95)),
        "score_p99": float(np.percentile(scores["total_score"], 99)),
        "anomaly_z_p95": float(np.percentile(anomaly_z, 95)),
        "anomaly_z_p99": float(np.percentile(anomaly_z, 99)),
        "calibration_from_checkpoint": val_stats,
    }
    out_summary = out_dir / f"{args.prefix}.stage1_infer_summary.json"
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.write_tsv:
        write_tsv(out_dir / f"{args.prefix}.stage1_scores.tsv", arrays, scores, anomaly_z)

    print("[DONE] Stage1 inference saved")
    print("  scores: ", out_npz)
    print("  summary:", out_summary)
    if args.write_tsv:
        print("  tsv:    ", out_dir / f"{args.prefix}.stage1_scores.tsv")


if __name__ == "__main__":
    main()
