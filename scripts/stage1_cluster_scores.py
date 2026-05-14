#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def simple_kmeans(x: np.ndarray, n_clusters: int, max_iter: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if len(x) == 0:
        raise ValueError("No rows to cluster.")
    n_clusters = min(n_clusters, len(x))
    rng = np.random.default_rng(seed)
    centers = x[rng.choice(len(x), size=n_clusters, replace=False)].copy()
    labels = np.zeros(len(x), dtype=np.int32)

    for _ in range(max_iter):
        dist = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = np.argmin(dist, axis=1).astype(np.int32)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for k in range(n_clusters):
            mask = labels == k
            if np.any(mask):
                centers[k] = x[mask].mean(axis=0)
            else:
                centers[k] = x[rng.integers(0, len(x))]
    return labels, centers


def load_score_files(paths: list[str | Path]) -> dict:
    rows = []
    for source_id, path in enumerate(paths):
        data = np.load(path, allow_pickle=True)
        n = len(data["total_score"])
        for i in range(n):
            rows.append(
                {
                    "source_id": source_id,
                    "source_path": str(path),
                    "row_index": i,
                    "score": float(data["total_score"][i]),
                    "anomaly_z": float(data["anomaly_z"][i]),
                    "latent": data["latent"][i].astype(np.float32),
                    "read_id": str(data["read_ids"][i]) if "read_ids" in data else "",
                    "sample_id": str(data["sample_ids"][i]) if "sample_ids" in data else "",
                    "condition": str(data["conditions"][i]) if "conditions" in data else "",
                    "window_start": int(data["window_starts"][i]) if "window_starts" in data else -1,
                    "num_events": int(data["num_events"][i]) if "num_events" in data else -1,
                    "label": str(data["labels"][i]) if "labels" in data else "",
                    "mod_type": str(data["mod_types"][i]) if "mod_types" in data else "",
                }
            )
    return {"rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 1: cluster high-anomaly signal windows into open-set signal states."
    )
    parser.add_argument("--score_npz", nargs="+", required=True, help="Files from stage1_infer_pca.py or stage1_infer_autoencoder.py.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--prefix", default="stage1_clusters")
    parser.add_argument("--min_anomaly_z", type=float, default=3.0)
    parser.add_argument("--top_fraction", type=float, default=0.0, help="If >0, cluster top fraction by score instead of z threshold.")
    parser.add_argument("--n_clusters", type=int, default=8)
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_score_files([Path(p) for p in args.score_npz])
    rows = loaded["rows"]
    if len(rows) == 0:
        raise ValueError("No score rows loaded.")

    scores = np.asarray([r["score"] for r in rows], dtype=np.float32)
    anomaly_z = np.asarray([r["anomaly_z"] for r in rows], dtype=np.float32)
    latent = np.stack([r["latent"] for r in rows], axis=0).astype(np.float32)

    if args.top_fraction > 0:
        n_keep = max(1, int(round(len(rows) * args.top_fraction)))
        selected = np.argsort(scores)[-n_keep:]
    else:
        selected = np.where(anomaly_z >= args.min_anomaly_z)[0]

    if len(selected) == 0:
        raise ValueError("No windows passed the anomaly selection threshold.")

    x = latent[selected]
    x_mean = x.mean(axis=0)
    x_std = np.maximum(x.std(axis=0), 1e-6)
    x_stdzd = (x - x_mean) / x_std

    cluster_id, centers = simple_kmeans(
        x_stdzd,
        n_clusters=args.n_clusters,
        max_iter=args.max_iter,
        seed=args.seed,
    )

    out_npz = out_dir / f"{args.prefix}.npz"
    np.savez(
        out_npz,
        selected_global_index=selected.astype(np.int64),
        cluster_id=cluster_id.astype(np.int32),
        latent_mean=x_mean.astype(np.float32),
        latent_std=x_std.astype(np.float32),
        cluster_centers=centers.astype(np.float32),
        score=scores[selected].astype(np.float32),
        anomaly_z=anomaly_z[selected].astype(np.float32),
    )

    out_tsv = out_dir / f"{args.prefix}.tsv"
    with out_tsv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "cluster_id",
                "source_id",
                "row_index",
                "sample_id",
                "condition",
                "read_id",
                "window_start",
                "num_events",
                "label",
                "mod_type",
                "score",
                "anomaly_z",
                "source_path",
            ]
        )
        for out_i, global_i in enumerate(selected):
            r = rows[int(global_i)]
            writer.writerow(
                [
                    int(cluster_id[out_i]),
                    int(r["source_id"]),
                    int(r["row_index"]),
                    r["sample_id"],
                    r["condition"],
                    r["read_id"],
                    int(r["window_start"]),
                    int(r["num_events"]),
                    r["label"],
                    r["mod_type"],
                    float(r["score"]),
                    float(r["anomaly_z"]),
                    r["source_path"],
                ]
            )

    summary = {
        "score_npz": args.score_npz,
        "n_total_windows": int(len(rows)),
        "n_selected_windows": int(len(selected)),
        "selection": {
            "min_anomaly_z": args.min_anomaly_z,
            "top_fraction": args.top_fraction,
        },
        "n_clusters": int(min(args.n_clusters, len(selected))),
        "cluster_counts": np.bincount(cluster_id).astype(int).tolist(),
        "score_mean_selected": float(scores[selected].mean()),
        "score_p95_selected": float(np.percentile(scores[selected], 95)),
        "anomaly_z_mean_selected": float(anomaly_z[selected].mean()),
        "anomaly_z_p95_selected": float(np.percentile(anomaly_z[selected], 95)),
    }
    out_summary = out_dir / f"{args.prefix}.summary.json"
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[DONE] Stage1 clusters saved")
    print("  npz:    ", out_npz)
    print("  tsv:    ", out_tsv)
    print("  summary:", out_summary)


if __name__ == "__main__":
    main()

