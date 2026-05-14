from __future__ import annotations

import numpy as np

from deepspot.stage1_data import Stage1Arrays, compute_event_standardization


def build_stage1_feature_matrix(
    arrays: Stage1Arrays,
    event_mean: np.ndarray,
    event_std: np.ndarray,
    raw_weight: float = 1.0,
    event_weight: float = 0.35,
    mask_weight: float = 0.15,
) -> np.ndarray:
    """Flatten raw signal + padded event features into one PCA feature matrix."""
    raw = arrays.raw_signal.astype(np.float32) * float(raw_weight)
    mask = arrays.event_mask.astype(np.float32)
    events = arrays.event_features.astype(np.float32)
    events = (events - event_mean.reshape(1, 1, -1)) / event_std.reshape(1, 1, -1)
    events = events * mask[:, :, None] * float(event_weight)
    mask_part = mask * float(mask_weight)
    return np.concatenate(
        [
            raw,
            events.reshape(events.shape[0], -1),
            mask_part,
        ],
        axis=1,
    ).astype(np.float32)


def fit_pca_model(
    arrays: Stage1Arrays,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    n_components: int = 32,
    raw_weight: float = 1.0,
    event_weight: float = 0.35,
    mask_weight: float = 0.15,
) -> dict:
    event_mean, event_std = compute_event_standardization(
        arrays.event_features[train_idx],
        arrays.event_mask[train_idx],
    )
    x = build_stage1_feature_matrix(
        arrays,
        event_mean=event_mean,
        event_std=event_std,
        raw_weight=raw_weight,
        event_weight=event_weight,
        mask_weight=mask_weight,
    )

    x_train = x[train_idx]
    x_val = x[val_idx]
    feature_mean = x_train.mean(axis=0).astype(np.float32)
    x_center = x_train - feature_mean

    max_components = max(1, min(n_components, x_center.shape[0] - 1, x_center.shape[1]))
    _u, singular_values, vt = np.linalg.svd(x_center, full_matrices=False)
    components = vt[:max_components].astype(np.float32)
    singular_values = singular_values[:max_components].astype(np.float32)

    val_scores = score_pca_matrix(x_val, feature_mean, components)
    return {
        "feature_mean": feature_mean,
        "components": components,
        "singular_values": singular_values,
        "event_mean": event_mean,
        "event_std": event_std,
        "raw_weight": float(raw_weight),
        "event_weight": float(event_weight),
        "mask_weight": float(mask_weight),
        "window_len": int(arrays.raw_signal.shape[1]),
        "max_events": int(arrays.event_features.shape[1]),
        "event_dim": int(arrays.event_features.shape[2]),
        "n_components": int(max_components),
        "val_score_mean": float(val_scores["score"].mean()),
        "val_score_std": float(max(val_scores["score"].std(), 1e-8)),
        "val_score_p95": float(np.percentile(val_scores["score"], 95)),
        "val_score_p99": float(np.percentile(val_scores["score"], 99)),
        "val_scores": val_scores["score"].astype(np.float32),
    }


def score_pca_matrix(x: np.ndarray, feature_mean: np.ndarray, components: np.ndarray) -> dict:
    centered = x - feature_mean
    latent = centered @ components.T
    recon = latent @ components + feature_mean
    residual = x - recon
    score = np.mean(residual * residual, axis=1)
    return {
        "latent": latent.astype(np.float32),
        "recon": recon.astype(np.float32),
        "residual": residual.astype(np.float32),
        "score": score.astype(np.float32),
    }


def score_pca_model(arrays: Stage1Arrays, model: dict) -> dict:
    x = build_stage1_feature_matrix(
        arrays,
        event_mean=model["event_mean"],
        event_std=model["event_std"],
        raw_weight=float(model["raw_weight"]),
        event_weight=float(model["event_weight"]),
        mask_weight=float(model["mask_weight"]),
    )
    out = score_pca_matrix(x, model["feature_mean"], model["components"])

    window_len = int(model["window_len"])
    raw_weight = float(model["raw_weight"])
    raw_recon = out["recon"][:, :window_len] / max(raw_weight, 1e-8)
    raw_residual = arrays.raw_signal.astype(np.float32) - raw_recon.astype(np.float32)

    score = out["score"]
    z = (score - float(model["val_score_mean"])) / max(float(model["val_score_std"]), 1e-8)
    out["raw_recon"] = raw_recon.astype(np.float32)
    out["raw_residual"] = raw_residual.astype(np.float32)
    out["anomaly_z"] = z.astype(np.float32)
    return out
