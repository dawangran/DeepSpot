from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np


EVENT_FEATURE_NAMES = [
    "mean",
    "std",
    "median",
    "min",
    "max",
    "range",
    "dwell",
    "dwell_frac",
    "start_frac",
    "end_frac",
    "slope",
    "energy",
    "diff_abs_mean",
]


WINDOW_MAD_NORMALIZE_METHODS = {"window_mad", "chunk_mad"}


def canonical_normalize_method(method: str) -> str:
    if method == "chunk_mad":
        return "window_mad"
    return method


def robust_normalize(signal: np.ndarray, method: str = "read_mad") -> tuple[np.ndarray, dict]:
    """Normalize one read-level signal.

    `read_mad` is kept for backward compatibility. New Stage 1 chunk/window
    building should usually use `window_mad` via `normalize_one_window`.
    """
    x = np.asarray(signal, dtype=np.float32)
    if method == "none":
        return x, {"method": method, "median": 0.0, "mad": 1.0}
    if method != "read_mad":
        raise ValueError(f"Unsupported normalization method: {method}")

    if x.size == 0:
        return x, {"method": method, "median": 0.0, "mad": 1.0}

    med = float(np.median(x))
    mad = float(1.4826 * np.median(np.abs(x - med)))
    mad = max(mad, 1.0)
    return ((x - med) / mad).astype(np.float32), {
        "method": method,
        "median": med,
        "mad": mad,
    }


def normalize_one_window(signal: np.ndarray) -> tuple[np.ndarray, dict]:
    """Normalize one fixed-length window using the chunk-level MAD rule."""
    x = np.asarray(signal, dtype=np.float32)
    if x.size == 0:
        return x, {"method": "window_mad", "median": 0.0, "mad": 0.0, "valid": False}

    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    if mad < 1e-8:
        return np.array([], dtype=np.float32), {
            "method": "window_mad",
            "median": med,
            "mad": mad,
            "valid": False,
        }

    return ((x - med) / mad).astype(np.float32), {
        "method": "window_mad",
        "median": med,
        "mad": mad,
        "valid": True,
    }


def window_values_in_range(window: np.ndarray, value_min: float, value_max: float) -> bool:
    return bool(np.all((window >= value_min) & (window <= value_max)))


def sliding_window_starts(length: int, window_len: int, stride: int) -> np.ndarray:
    if length < window_len:
        return np.array([], dtype=np.int64)
    return np.arange(0, length - window_len + 1, stride, dtype=np.int64)


def _segment_sse(prefix: np.ndarray, prefix2: np.ndarray, start: int, end: int) -> float:
    n = end - start
    if n <= 0:
        return 0.0
    s = prefix[end] - prefix[start]
    s2 = prefix2[end] - prefix2[start]
    return float(max(s2 - (s * s / n), 0.0))


def _best_split(
    signal: np.ndarray,
    prefix: np.ndarray,
    prefix2: np.ndarray,
    start: int,
    end: int,
    min_event_len: int,
) -> tuple[int | None, float]:
    """Return the split with maximum reduction in piecewise-constant SSE."""
    if end - start < 2 * min_event_len:
        return None, 0.0

    candidates = np.arange(start + min_event_len, end - min_event_len + 1, dtype=np.int64)
    if candidates.size == 0:
        return None, 0.0

    parent_cost = _segment_sse(prefix, prefix2, start, end)

    n1 = candidates - start
    n2 = end - candidates
    s1 = prefix[candidates] - prefix[start]
    s2 = prefix[end] - prefix[candidates]
    q1 = prefix2[candidates] - prefix2[start]
    q2 = prefix2[end] - prefix2[candidates]

    left_cost = q1 - (s1 * s1 / n1)
    right_cost = q2 - (s2 * s2 / n2)
    costs = left_cost + right_cost

    best_i = int(np.argmin(costs))
    gain = float(parent_cost - costs[best_i])
    return int(candidates[best_i]), gain


def segment_piecewise_constant(
    signal: np.ndarray,
    penalty: float = 8.0,
    min_event_len: int = 5,
    max_events: int = 16,
) -> list[tuple[int, int]]:
    """Simple binary segmentation for one fixed-length raw signal window.

    This intentionally does not force a fixed number of events. The returned
    event count depends on the local signal, the penalty, and `min_event_len`.
    """
    x = np.asarray(signal, dtype=np.float64)
    n = int(x.size)
    if n == 0:
        return []
    if n <= min_event_len or max_events <= 1:
        return [(0, n)]

    prefix = np.concatenate([[0.0], np.cumsum(x)])
    prefix2 = np.concatenate([[0.0], np.cumsum(x * x)])

    segments: list[tuple[int, int]] = [(0, n)]
    while len(segments) < max_events:
        best = None
        best_gain = 0.0

        for seg_idx, (start, end) in enumerate(segments):
            split, gain = _best_split(x, prefix, prefix2, start, end, min_event_len)
            if split is not None and gain > best_gain:
                best = (seg_idx, start, split, end)
                best_gain = gain

        if best is None or best_gain <= penalty:
            break

        seg_idx, start, split, end = best
        segments[seg_idx : seg_idx + 1] = [(start, split), (split, end)]

    return sorted(segments)


def segment_piecewise_constant_ruptures(
    signal: np.ndarray,
    penalty: float = 8.0,
    min_event_len: int = 5,
    max_events: int = 16,
) -> list[tuple[int, int]]:
    """Segment one window with ruptures when the optional dependency exists."""
    try:
        import ruptures as rpt  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "event_backend='ruptures' requires the optional dependency `ruptures`."
        ) from exc

    x = np.asarray(signal, dtype=np.float64)
    n = int(x.size)
    if n == 0:
        return []
    if n <= min_event_len or max_events <= 1:
        return [(0, n)]

    algo = rpt.Binseg(model="l2", min_size=min_event_len, jump=1).fit(x.reshape(-1, 1))
    breakpoints = algo.predict(pen=penalty)
    if len(breakpoints) > max_events:
        breakpoints = algo.predict(n_bkps=max_events - 1)

    prev = 0
    segments = []
    for end in breakpoints:
        end = int(end)
        if end > prev:
            segments.append((prev, end))
        prev = end
    return segments or [(0, n)]


def segment_signal_events(
    signal: np.ndarray,
    penalty: float = 8.0,
    min_event_len: int = 5,
    max_events: int = 16,
    event_backend: str = "simple",
) -> list[tuple[int, int]]:
    if event_backend == "simple":
        return segment_piecewise_constant(
            signal,
            penalty=penalty,
            min_event_len=min_event_len,
            max_events=max_events,
        )
    if event_backend == "ruptures":
        return segment_piecewise_constant_ruptures(
            signal,
            penalty=penalty,
            min_event_len=min_event_len,
            max_events=max_events,
        )
    raise ValueError(f"Unsupported event backend: {event_backend}")


def _slope(y: np.ndarray) -> float:
    if y.size < 2:
        return 0.0
    x = np.arange(y.size, dtype=np.float64)
    x = x - x.mean()
    denom = float(np.sum(x * x))
    if denom == 0:
        return 0.0
    yy = y.astype(np.float64) - float(np.mean(y))
    return float(np.sum(x * yy) / denom)


def events_to_features(signal: np.ndarray, events: list[tuple[int, int]]) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float32)
    length = max(int(x.size), 1)
    rows = []
    for start, end in events:
        cur = x[start:end]
        if cur.size == 0:
            continue
        diffs = np.diff(cur)
        rows.append(
            [
                float(np.mean(cur)),
                float(np.std(cur)),
                float(np.median(cur)),
                float(np.min(cur)),
                float(np.max(cur)),
                float(np.max(cur) - np.min(cur)),
                float(end - start),
                float((end - start) / length),
                float(start / length),
                float(end / length),
                _slope(cur),
                float(np.mean(cur * cur)),
                float(np.mean(np.abs(diffs))) if diffs.size > 0 else 0.0,
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def encode_window_events(
    signal: np.ndarray,
    max_events: int,
    penalty: float,
    min_event_len: int,
    event_backend: str = "simple",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Segment a raw window and return padded event features plus metadata."""
    events = segment_signal_events(
        signal,
        penalty=penalty,
        min_event_len=min_event_len,
        max_events=max_events,
        event_backend=event_backend,
    )
    features = events_to_features(signal, events)
    event_dim = len(EVENT_FEATURE_NAMES)

    padded = np.zeros((max_events, event_dim), dtype=np.float32)
    mask = np.zeros((max_events,), dtype=np.float32)
    starts = np.full((max_events,), -1, dtype=np.int32)
    ends = np.full((max_events,), -1, dtype=np.int32)

    n = min(len(features), max_events)
    if n > 0:
        padded[:n] = features[:n]
        mask[:n] = 1.0
        starts[:n] = np.asarray([s for s, _ in events[:n]], dtype=np.int32)
        ends[:n] = np.asarray([e for _, e in events[:n]], dtype=np.int32)

    return padded, mask, starts, ends


def build_stage1_arrays_from_jsonl(
    input_jsonl: str | Path,
    window_len: int = 128,
    stride: int = 64,
    max_events: int = 16,
    penalty: float = 8.0,
    min_event_len: int = 5,
    min_signal_len: int = 128,
    normalize: str = "window_mad",
    value_min: float = -3.0,
    value_max: float = 3.0,
    sample_id: str = "",
    condition: str = "",
    max_reads: int = 0,
    max_windows: int = 0,
    event_backend: str = "simple",
) -> dict:
    """Build fixed raw windows plus variable-event representations from JSONL.

    Expected JSONL fields:
      - `signal`: list[float], required
      - `read_id`: optional
      - `label`, `pattern`, `mod_type`: optional metadata
    """
    input_jsonl = Path(input_jsonl)
    normalize = canonical_normalize_method(normalize)
    if value_min > value_max:
        raise ValueError("value_min must be <= value_max.")

    raw_windows = []
    event_features = []
    event_mask = []
    event_starts = []
    event_ends = []
    read_ids = []
    window_starts = []
    signal_lengths = []
    labels = []
    patterns = []
    mod_types = []
    medians = []
    mads = []

    n_records = 0
    n_kept_reads = 0
    n_skipped_missing_signal = 0
    n_skipped_short = 0
    n_window_candidates = 0
    n_windows_dropped_mad = 0
    n_windows_dropped_value_range = 0

    with input_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            n_records += 1
            if max_reads > 0 and n_records > max_reads:
                break

            record = json.loads(line)
            signal = record.get("signal")
            if signal is None:
                n_skipped_missing_signal += 1
                continue

            signal = np.asarray(signal, dtype=np.float32)
            if signal.size < max(min_signal_len, window_len):
                n_skipped_short += 1
                continue

            if normalize in WINDOW_MAD_NORMALIZE_METHODS:
                signal_for_windows = signal
                read_norm_meta = None
            else:
                signal_for_windows, read_norm_meta = robust_normalize(signal, method=normalize)

            starts = sliding_window_starts(signal_for_windows.size, window_len, stride)
            if starts.size == 0:
                n_skipped_short += 1
                continue

            read_id = str(record.get("read_id", f"read_{n_records}"))
            label = record.get("label", "")
            pattern = record.get("pattern", "")
            mod_type = record.get("mod_type", "")
            kept_for_read = 0

            for start in starts:
                n_window_candidates += 1
                window_source = signal_for_windows[start : start + window_len].astype(np.float32)
                if normalize in WINDOW_MAD_NORMALIZE_METHODS:
                    window, norm_meta = normalize_one_window(window_source)
                    if window.size == 0:
                        n_windows_dropped_mad += 1
                        continue
                    if not window_values_in_range(window, value_min=value_min, value_max=value_max):
                        n_windows_dropped_value_range += 1
                        continue
                else:
                    window = window_source
                    norm_meta = read_norm_meta

                feats, mask, ev_starts, ev_ends = encode_window_events(
                    window,
                    max_events=max_events,
                    penalty=penalty,
                    min_event_len=min_event_len,
                    event_backend=event_backend,
                )

                raw_windows.append(window)
                event_features.append(feats)
                event_mask.append(mask)
                event_starts.append(ev_starts)
                event_ends.append(ev_ends)
                read_ids.append(read_id)
                window_starts.append(int(start))
                signal_lengths.append(int(signal.size))
                labels.append("" if label is None else str(label))
                patterns.append("" if pattern is None else str(pattern))
                mod_types.append("" if mod_type is None else str(mod_type))
                medians.append(float(norm_meta["median"]))
                mads.append(float(norm_meta["mad"]))
                kept_for_read += 1

                if max_windows > 0 and len(raw_windows) >= max_windows:
                    break

            if kept_for_read > 0:
                n_kept_reads += 1
            if max_windows > 0 and len(raw_windows) >= max_windows:
                break

    if raw_windows:
        raw_windows_arr = np.stack(raw_windows, axis=0).astype(np.float32)
        event_features_arr = np.stack(event_features, axis=0).astype(np.float32)
        event_mask_arr = np.stack(event_mask, axis=0).astype(np.float32)
        event_starts_arr = np.stack(event_starts, axis=0).astype(np.int32)
        event_ends_arr = np.stack(event_ends, axis=0).astype(np.int32)
    else:
        raw_windows_arr = np.empty((0, window_len), dtype=np.float32)
        event_features_arr = np.empty((0, max_events, len(EVENT_FEATURE_NAMES)), dtype=np.float32)
        event_mask_arr = np.empty((0, max_events), dtype=np.float32)
        event_starts_arr = np.empty((0, max_events), dtype=np.int32)
        event_ends_arr = np.empty((0, max_events), dtype=np.int32)

    summary = {
        "input_jsonl": str(input_jsonl),
        "sample_id": sample_id,
        "condition": condition,
        "n_records": n_records,
        "n_kept_reads": n_kept_reads,
        "n_skipped_missing_signal": n_skipped_missing_signal,
        "n_skipped_short": n_skipped_short,
        "n_window_candidates": n_window_candidates,
        "n_windows_dropped_mad": n_windows_dropped_mad,
        "n_windows_dropped_value_range": n_windows_dropped_value_range,
        "n_windows": int(raw_windows_arr.shape[0]),
        "window_len": window_len,
        "stride": stride,
        "max_events": max_events,
        "penalty": penalty,
        "min_event_len": min_event_len,
        "event_backend": event_backend,
        "normalize": normalize,
        "value_min": value_min,
        "value_max": value_max,
    }

    return {
        "raw_signal": raw_windows_arr,
        "event_features": event_features_arr,
        "event_mask": event_mask_arr,
        "event_starts": event_starts_arr,
        "event_ends": event_ends_arr,
        "read_ids": np.asarray(read_ids, dtype=object),
        "window_starts": np.asarray(window_starts, dtype=np.int64),
        "signal_lengths": np.asarray(signal_lengths, dtype=np.int32),
        "labels": np.asarray(labels, dtype=object),
        "patterns": np.asarray(patterns, dtype=object),
        "mod_types": np.asarray(mod_types, dtype=object),
        "sample_ids": np.asarray([sample_id] * len(read_ids), dtype=object),
        "conditions": np.asarray([condition] * len(read_ids), dtype=object),
        "read_norm_median": np.asarray(medians, dtype=np.float32),
        "read_norm_mad": np.asarray(mads, dtype=np.float32),
        "norm_median": np.asarray(medians, dtype=np.float32),
        "norm_mad": np.asarray(mads, dtype=np.float32),
        "event_feature_names": np.asarray(EVENT_FEATURE_NAMES, dtype=object),
        "summary_json": np.asarray([json.dumps(summary, ensure_ascii=False)], dtype=object),
    }


def _safe_get_ccf5_read(s5, read_id):
    """Read one CCF5 record while avoiding fragile aux-field decoding paths."""
    last_err = None
    for kwargs in (
        {"pA": True},
        {"pA": True, "aux": None},
        {},
    ):
        try:
            return s5.get_read(read_id, **kwargs)
        except Exception as exc:  # pragma: no cover - depends on pyccf5 backend
            last_err = exc
    raise last_err


def _ccf5_signal_to_pa(read) -> np.ndarray:
    """Return the read signal in pA when CCF5 calibration fields are available."""
    if "signal" not in read:
        raise KeyError("signal")

    sig = np.asarray(read["signal"])
    if np.issubdtype(sig.dtype, np.floating):
        return sig.astype(np.float32)

    if "lvdsmid" in read and "unit" in read:
        return ((sig - read["lvdsmid"]) * read["unit"]).astype(np.float32)

    if all(k in read for k in ("K", "scale", "offset", "B")):
        return (
            read["K"] * read["scale"] * (sig.astype(np.uint16) + read["offset"]) + read["B"]
        ).astype(np.float32)

    return sig.astype(np.float32)


def _load_pyccf5():
    try:
        import pyccf5 as slow5  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "CCF5 input requires the optional dependency `pyccf5`. "
            "Install it before running --input_ccf5 or --input_ccf5_dir."
        ) from exc
    return slow5


def build_stage1_arrays_from_ccf5(
    ccf5_paths: Sequence[str | Path],
    window_len: int = 128,
    stride: int = 64,
    max_events: int = 16,
    penalty: float = 8.0,
    min_event_len: int = 5,
    min_signal_len: int = 128,
    normalize: str = "window_mad",
    value_min: float = -3.0,
    value_max: float = 3.0,
    sample_id: str = "",
    condition: str = "",
    max_reads: int = 0,
    max_windows: int = 0,
    trim_head: int = 0,
    trim_tail: int = 0,
    event_backend: str = "simple",
) -> dict:
    """Build Stage 1 arrays directly from one or more CCF5 files.

    The output schema intentionally matches `build_stage1_arrays_from_jsonl`
    so PCA/autoencoder training and inference can consume both input formats.
    """
    if trim_head < 0 or trim_tail < 0:
        raise ValueError("trim_head and trim_tail must be non-negative.")
    normalize = canonical_normalize_method(normalize)
    if value_min > value_max:
        raise ValueError("value_min must be <= value_max.")

    paths = [Path(p) for p in ccf5_paths]
    if not paths:
        raise ValueError("At least one CCF5 path is required.")

    slow5 = _load_pyccf5()

    raw_windows = []
    event_features = []
    event_mask = []
    event_starts = []
    event_ends = []
    read_ids = []
    window_starts = []
    signal_lengths = []
    labels = []
    patterns = []
    mod_types = []
    sample_ids = []
    conditions = []
    medians = []
    mads = []
    source_files = []
    trimmed_signal_lengths = []

    n_files = 0
    n_records = 0
    n_kept_reads = 0
    n_skipped_short = 0
    n_skipped_after_trim = 0
    n_skipped_read_fetch_error = 0
    n_skipped_signal_error = 0
    n_window_candidates = 0
    n_windows_dropped_mad = 0
    n_windows_dropped_value_range = 0

    for ccf5_path in paths:
        n_files += 1
        s5 = slow5.Open(str(ccf5_path), "r", DEBUG=0)
        try:
            ids_result = s5.get_read_ids()
            all_read_ids = ids_result[0] if isinstance(ids_result, tuple) else ids_result

            for read_id_obj in all_read_ids:
                n_records += 1
                if max_reads > 0 and n_records > max_reads:
                    break

                read_id = str(read_id_obj)
                try:
                    read = _safe_get_ccf5_read(s5, read_id_obj)
                except Exception:
                    n_skipped_read_fetch_error += 1
                    continue

                try:
                    signal = _ccf5_signal_to_pa(read)
                except Exception:
                    n_skipped_signal_error += 1
                    continue

                if signal.size < max(min_signal_len, window_len):
                    n_skipped_short += 1
                    continue

                if signal.size <= trim_head + trim_tail:
                    n_skipped_after_trim += 1
                    continue

                signal_trimmed = signal[trim_head : signal.size - trim_tail]
                if signal_trimmed.size < max(min_signal_len, window_len):
                    n_skipped_after_trim += 1
                    continue

                if normalize in WINDOW_MAD_NORMALIZE_METHODS:
                    signal_for_windows = signal_trimmed
                    read_norm_meta = None
                else:
                    signal_for_windows, read_norm_meta = robust_normalize(signal_trimmed, method=normalize)

                starts = sliding_window_starts(signal_for_windows.size, window_len, stride)
                if starts.size == 0:
                    n_skipped_after_trim += 1
                    continue

                kept_for_read = 0
                for start in starts:
                    n_window_candidates += 1
                    window_source = signal_for_windows[start : start + window_len].astype(np.float32)
                    if normalize in WINDOW_MAD_NORMALIZE_METHODS:
                        window, norm_meta = normalize_one_window(window_source)
                        if window.size == 0:
                            n_windows_dropped_mad += 1
                            continue
                        if not window_values_in_range(window, value_min=value_min, value_max=value_max):
                            n_windows_dropped_value_range += 1
                            continue
                    else:
                        window = window_source
                        norm_meta = read_norm_meta

                    feats, mask, ev_starts, ev_ends = encode_window_events(
                        window,
                        max_events=max_events,
                        penalty=penalty,
                        min_event_len=min_event_len,
                        event_backend=event_backend,
                    )

                    raw_windows.append(window)
                    event_features.append(feats)
                    event_mask.append(mask)
                    event_starts.append(ev_starts)
                    event_ends.append(ev_ends)
                    read_ids.append(read_id)
                    window_starts.append(int(start + trim_head))
                    signal_lengths.append(int(signal.size))
                    labels.append("")
                    patterns.append("")
                    mod_types.append("")
                    sample_ids.append(sample_id)
                    conditions.append(condition)
                    medians.append(float(norm_meta["median"]))
                    mads.append(float(norm_meta["mad"]))
                    source_files.append(str(ccf5_path))
                    trimmed_signal_lengths.append(int(signal_trimmed.size))
                    kept_for_read += 1

                    if max_windows > 0 and len(raw_windows) >= max_windows:
                        break

                if kept_for_read > 0:
                    n_kept_reads += 1
                if max_windows > 0 and len(raw_windows) >= max_windows:
                    break

        finally:
            s5.close()

        if max_reads > 0 and n_records >= max_reads:
            break
        if max_windows > 0 and len(raw_windows) >= max_windows:
            break

    if raw_windows:
        raw_windows_arr = np.stack(raw_windows, axis=0).astype(np.float32)
        event_features_arr = np.stack(event_features, axis=0).astype(np.float32)
        event_mask_arr = np.stack(event_mask, axis=0).astype(np.float32)
        event_starts_arr = np.stack(event_starts, axis=0).astype(np.int32)
        event_ends_arr = np.stack(event_ends, axis=0).astype(np.int32)
    else:
        raw_windows_arr = np.empty((0, window_len), dtype=np.float32)
        event_features_arr = np.empty((0, max_events, len(EVENT_FEATURE_NAMES)), dtype=np.float32)
        event_mask_arr = np.empty((0, max_events), dtype=np.float32)
        event_starts_arr = np.empty((0, max_events), dtype=np.int32)
        event_ends_arr = np.empty((0, max_events), dtype=np.int32)

    summary = {
        "input_format": "ccf5",
        "input_ccf5": [str(p) for p in paths],
        "sample_id": sample_id,
        "condition": condition,
        "n_files": n_files,
        "n_records": n_records,
        "n_kept_reads": n_kept_reads,
        "n_skipped_short": n_skipped_short,
        "n_skipped_after_trim": n_skipped_after_trim,
        "n_skipped_read_fetch_error": n_skipped_read_fetch_error,
        "n_skipped_signal_error": n_skipped_signal_error,
        "n_window_candidates": n_window_candidates,
        "n_windows_dropped_mad": n_windows_dropped_mad,
        "n_windows_dropped_value_range": n_windows_dropped_value_range,
        "n_windows": int(raw_windows_arr.shape[0]),
        "window_len": window_len,
        "stride": stride,
        "max_events": max_events,
        "penalty": penalty,
        "min_event_len": min_event_len,
        "event_backend": event_backend,
        "normalize": normalize,
        "value_min": value_min,
        "value_max": value_max,
        "trim_head": trim_head,
        "trim_tail": trim_tail,
    }

    return {
        "raw_signal": raw_windows_arr,
        "event_features": event_features_arr,
        "event_mask": event_mask_arr,
        "event_starts": event_starts_arr,
        "event_ends": event_ends_arr,
        "read_ids": np.asarray(read_ids, dtype=object),
        "window_starts": np.asarray(window_starts, dtype=np.int64),
        "signal_lengths": np.asarray(signal_lengths, dtype=np.int32),
        "labels": np.asarray(labels, dtype=object),
        "patterns": np.asarray(patterns, dtype=object),
        "mod_types": np.asarray(mod_types, dtype=object),
        "sample_ids": np.asarray(sample_ids, dtype=object),
        "conditions": np.asarray(conditions, dtype=object),
        "read_norm_median": np.asarray(medians, dtype=np.float32),
        "read_norm_mad": np.asarray(mads, dtype=np.float32),
        "norm_median": np.asarray(medians, dtype=np.float32),
        "norm_mad": np.asarray(mads, dtype=np.float32),
        "source_files": np.asarray(source_files, dtype=object),
        "trimmed_signal_lengths": np.asarray(trimmed_signal_lengths, dtype=np.int32),
        "event_feature_names": np.asarray(EVENT_FEATURE_NAMES, dtype=object),
        "summary_json": np.asarray([json.dumps(summary, ensure_ascii=False)], dtype=object),
    }


def save_stage1_npz(arrays: dict, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **arrays)


def load_summary_from_npz(npz_path: str | Path) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    if "summary_json" not in data:
        return {}
    return json.loads(str(data["summary_json"][0]))
