from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Stage1Arrays:
    raw_signal: np.ndarray
    event_features: np.ndarray
    event_mask: np.ndarray
    read_ids: np.ndarray
    window_starts: np.ndarray
    labels: np.ndarray
    sample_ids: np.ndarray
    conditions: np.ndarray
    mod_types: np.ndarray
    event_starts: np.ndarray | None = None
    event_ends: np.ndarray | None = None


def load_stage1_arrays(paths: list[str | Path]) -> Stage1Arrays:
    raw = []
    events = []
    masks = []
    read_ids = []
    starts = []
    labels = []
    sample_ids = []
    conditions = []
    mod_types = []
    event_starts = []
    event_ends = []

    for path in paths:
        data = np.load(path, allow_pickle=True)
        raw.append(data["raw_signal"].astype(np.float32))
        events.append(data["event_features"].astype(np.float32))
        masks.append(data["event_mask"].astype(np.float32))
        read_ids.append(data["read_ids"].astype(object))
        starts.append(data["window_starts"].astype(np.int64))
        labels.append(data["labels"].astype(object) if "labels" in data else np.array([""] * len(data["raw_signal"]), dtype=object))
        sample_ids.append(data["sample_ids"].astype(object) if "sample_ids" in data else np.array([""] * len(data["raw_signal"]), dtype=object))
        conditions.append(data["conditions"].astype(object) if "conditions" in data else np.array([""] * len(data["raw_signal"]), dtype=object))
        mod_types.append(data["mod_types"].astype(object) if "mod_types" in data else np.array([""] * len(data["raw_signal"]), dtype=object))
        if "event_starts" in data:
            event_starts.append(data["event_starts"].astype(np.int32))
        if "event_ends" in data:
            event_ends.append(data["event_ends"].astype(np.int32))

    return Stage1Arrays(
        raw_signal=np.concatenate(raw, axis=0),
        event_features=np.concatenate(events, axis=0),
        event_mask=np.concatenate(masks, axis=0),
        read_ids=np.concatenate(read_ids, axis=0),
        window_starts=np.concatenate(starts, axis=0),
        labels=np.concatenate(labels, axis=0),
        sample_ids=np.concatenate(sample_ids, axis=0),
        conditions=np.concatenate(conditions, axis=0),
        mod_types=np.concatenate(mod_types, axis=0),
        event_starts=np.concatenate(event_starts, axis=0) if event_starts else None,
        event_ends=np.concatenate(event_ends, axis=0) if event_ends else None,
    )


def split_indices_by_read_id(
    read_ids: np.ndarray,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    unique_reads = np.unique(read_ids.astype(str))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_reads)

    n_val_reads = max(1, int(round(len(unique_reads) * val_ratio)))
    val_reads = set(unique_reads[:n_val_reads].tolist())
    is_val = np.asarray([str(r) in val_reads for r in read_ids], dtype=bool)

    train_idx = np.where(~is_val)[0]
    val_idx = np.where(is_val)[0]
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError("Read-level split produced an empty train or validation set.")
    return train_idx.astype(np.int64), val_idx.astype(np.int64)


def compute_event_standardization(event_features: np.ndarray, event_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = event_mask > 0.5
    if not np.any(valid):
        mean = np.zeros(event_features.shape[-1], dtype=np.float32)
        std = np.ones(event_features.shape[-1], dtype=np.float32)
        return mean, std

    values = event_features[valid]
    mean = values.mean(axis=0).astype(np.float32)
    std = values.std(axis=0).astype(np.float32)
    std = np.maximum(std, 1e-6).astype(np.float32)
    return mean, std

