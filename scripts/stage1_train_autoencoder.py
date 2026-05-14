#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepspot.stage1_data import (  # noqa: E402
    compute_event_standardization,
    load_stage1_arrays,
    split_indices_by_read_id,
)
from deepspot.stage1_model import (  # noqa: E402
    RawEventAutoencoder,
    Stage1WindowDataset,
    reconstruction_losses,
)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, optimizer, device, event_weight: float, train: bool) -> dict:
    model.train(train)
    total = 0
    raw_sum = 0.0
    event_sum = 0.0
    loss_sum = 0.0

    for batch in loader:
        raw = batch["raw"].to(device)
        events = batch["events"].to(device)
        mask = batch["mask"].to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            raw_hat, event_hat, _z = model(raw, events, mask)
            raw_mse, event_mse = reconstruction_losses(raw, raw_hat, events, event_hat, mask)
            loss = (raw_mse + event_weight * event_mse).mean()
            if train:
                loss.backward()
                optimizer.step()

        bs = raw.size(0)
        total += bs
        raw_sum += float(raw_mse.sum().item())
        event_sum += float(event_mse.sum().item())
        loss_sum += float(loss.item() * bs)

    return {
        "loss": loss_sum / max(total, 1),
        "raw_mse": raw_sum / max(total, 1),
        "event_mse": event_sum / max(total, 1),
        "n": total,
    }


@torch.no_grad()
def collect_scores(model, loader, device, event_weight: float) -> dict:
    model.eval()
    all_raw = []
    all_event = []
    all_total = []
    for batch in loader:
        raw = batch["raw"].to(device)
        events = batch["events"].to(device)
        mask = batch["mask"].to(device)
        raw_hat, event_hat, _z = model(raw, events, mask)
        raw_mse, event_mse = reconstruction_losses(raw, raw_hat, events, event_hat, mask)
        total = raw_mse + event_weight * event_mse
        all_raw.append(raw_mse.cpu().numpy())
        all_event.append(event_mse.cpu().numpy())
        all_total.append(total.cpu().numpy())

    raw_arr = np.concatenate(all_raw, axis=0)
    event_arr = np.concatenate(all_event, axis=0)
    total_arr = np.concatenate(all_total, axis=0)
    return {
        "raw_mse": raw_arr,
        "event_mse": event_arr,
        "total_score": total_arr,
        "mean": float(total_arr.mean()),
        "std": float(max(total_arr.std(), 1e-8)),
        "p95": float(np.percentile(total_arr, 95)),
        "p99": float(np.percentile(total_arr, 99)),
        "p999": float(np.percentile(total_arr, 99.9)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 1: train raw/event autoencoder on unmodified windows."
    )
    parser.add_argument("--train_npz", nargs="+", required=True, help="One or more *.stage1_windows.npz files from unmodified data.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--latent_dim", type=int, default=48)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--event_weight", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arrays = load_stage1_arrays([Path(p) for p in args.train_npz])
    if len(arrays.raw_signal) == 0:
        raise ValueError("No training windows found.")

    train_idx, val_idx = split_indices_by_read_id(arrays.read_ids, args.val_ratio, args.seed)
    event_mean, event_std = compute_event_standardization(
        arrays.event_features[train_idx],
        arrays.event_mask[train_idx],
    )

    train_ds = Stage1WindowDataset(arrays, train_idx, event_mean=event_mean, event_std=event_std)
    val_ds = Stage1WindowDataset(arrays, val_idx, event_mean=event_mean, event_std=event_std)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    window_len = int(arrays.raw_signal.shape[1])
    max_events = int(arrays.event_features.shape[1])
    event_dim = int(arrays.event_features.shape[2])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RawEventAutoencoder(
        window_len=window_len,
        max_events=max_events,
        event_dim=event_dim,
        latent_dim=args.latent_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
    best_val = float("inf")
    best_path = out_dir / "stage1_autoencoder_best.pt"

    print("[INFO] device:", device)
    print("[INFO] train windows:", len(train_ds), "val windows:", len(val_ds))
    print("[INFO] window_len:", window_len, "max_events:", max_events, "event_dim:", event_dim)

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, args.event_weight, train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, device, args.event_weight, train=False)

        item = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(item)
        print(
            f"epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.6f} | "
            f"val_loss={val_metrics['loss']:.6f} | "
            f"val_raw={val_metrics['raw_mse']:.6f} | "
            f"val_event={val_metrics['event_mse']:.6f}"
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": {
                        "window_len": window_len,
                        "max_events": max_events,
                        "event_dim": event_dim,
                        "latent_dim": args.latent_dim,
                    },
                    "event_mean": event_mean,
                    "event_std": event_std,
                    "event_weight": args.event_weight,
                    "train_npz": args.train_npz,
                    "seed": args.seed,
                    "best_val_loss": best_val,
                },
                best_path,
            )
            print("[INFO] saved best:", best_path)

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    val_scores = collect_scores(model, val_loader, device, args.event_weight)

    ckpt["val_score_stats"] = {
        "mean": val_scores["mean"],
        "std": val_scores["std"],
        "p95": val_scores["p95"],
        "p99": val_scores["p99"],
        "p999": val_scores["p999"],
    }
    torch.save(ckpt, best_path)

    np.savez(
        out_dir / "stage1_val_scores.npz",
        raw_mse=val_scores["raw_mse"].astype(np.float32),
        event_mse=val_scores["event_mse"].astype(np.float32),
        total_score=val_scores["total_score"].astype(np.float32),
    )
    (out_dir / "stage1_train_history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )
    (out_dir / "stage1_train_config.json").write_text(
        json.dumps(
            {
                "train_npz": args.train_npz,
                "n_train_windows": int(len(train_ds)),
                "n_val_windows": int(len(val_ds)),
                "window_len": window_len,
                "max_events": max_events,
                "event_dim": event_dim,
                "latent_dim": args.latent_dim,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "lr": args.lr,
                "val_ratio": args.val_ratio,
                "event_weight": args.event_weight,
                "seed": args.seed,
                "best_val_loss": best_val,
                "val_score_stats": ckpt["val_score_stats"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("[DONE] trained Stage1 autoencoder")
    print("  checkpoint:", best_path)
    print("  val score stats:", ckpt["val_score_stats"])


if __name__ == "__main__":
    main()
