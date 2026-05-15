#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepspot.stage1_events import (  # noqa: E402
    EVENT_FEATURE_NAMES,
    build_stage1_arrays_from_chunk_corpus,
    build_stage1_arrays_from_ccf5,
    build_stage1_arrays_from_jsonl,
    save_stage1_npz,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 1: build 128-point raw windows plus variable event features from JSONL or CCF5 signal data."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input_jsonl", help="JSONL file with at least a `signal` field.")
    input_group.add_argument("--input_ccf5", nargs="+", help="One or more CCF5 files.")
    input_group.add_argument("--input_ccf5_dir", help="Directory containing CCF5 files.")
    input_group.add_argument(
        "--input_chunk_corpus",
        help="Directory or npz file produced by chunk-level corpus scripts, e.g. chunks.npy/chunk_meta.npz outputs.",
    )
    parser.add_argument("--ccf5_pattern", default="*.ccf5", help="Glob pattern used with --input_ccf5_dir.")
    parser.add_argument("--out_dir", required=True, help="Output directory.")
    parser.add_argument("--prefix", default="stage1", help="Output file prefix.")
    parser.add_argument("--sample_id", default="", help="Sample id stored in metadata.")
    parser.add_argument("--condition", default="", help="unmodified / natural / synthetic, stored in metadata.")
    parser.add_argument("--window_len", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--max_events", type=int, default=16)
    parser.add_argument("--penalty", type=float, default=8.0, help="Binary segmentation SSE gain threshold.")
    parser.add_argument("--min_event_len", type=int, default=5)
    parser.add_argument("--event_backend", choices=["simple", "ruptures"], default="simple")
    parser.add_argument("--min_signal_len", type=int, default=128)
    parser.add_argument("--normalize", choices=["window_mad", "chunk_mad", "read_mad", "none"], default="window_mad")
    parser.add_argument("--value_min", type=float, default=-3.0, help="Minimum allowed value after per-window normalization.")
    parser.add_argument("--value_max", type=float, default=3.0, help="Maximum allowed value after per-window normalization.")
    parser.add_argument("--trim_head", type=int, default=0, help="CCF5 only: raw samples trimmed from read head before windowing.")
    parser.add_argument("--trim_tail", type=int, default=0, help="CCF5 only: raw samples trimmed from read tail before windowing.")
    parser.add_argument("--max_reads", type=int, default=0, help="Debug limit. 0 means all reads.")
    parser.add_argument("--max_windows", type=int, default=0, help="Debug limit. 0 means all windows.")
    args = parser.parse_args()
    if args.value_min > args.value_max:
        parser.error("--value_min must be <= --value_max")
    if args.event_backend == "ruptures" and importlib.util.find_spec("ruptures") is None:
        parser.error("--event_backend ruptures requires installing the optional dependency `ruptures`")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.input_jsonl:
        arrays = build_stage1_arrays_from_jsonl(
            input_jsonl=args.input_jsonl,
            window_len=args.window_len,
            stride=args.stride,
            max_events=args.max_events,
            penalty=args.penalty,
            min_event_len=args.min_event_len,
            event_backend=args.event_backend,
            min_signal_len=args.min_signal_len,
            normalize=args.normalize,
            value_min=args.value_min,
            value_max=args.value_max,
            sample_id=args.sample_id,
            condition=args.condition,
            max_reads=args.max_reads,
            max_windows=args.max_windows,
        )
    elif args.input_chunk_corpus:
        arrays = build_stage1_arrays_from_chunk_corpus(
            input_corpus=args.input_chunk_corpus,
            window_len=args.window_len,
            max_events=args.max_events,
            penalty=args.penalty,
            min_event_len=args.min_event_len,
            event_backend=args.event_backend,
            sample_id=args.sample_id,
            condition=args.condition,
            max_windows=args.max_windows,
        )
    else:
        if args.input_ccf5_dir:
            ccf5_paths = sorted(glob.glob(str(Path(args.input_ccf5_dir) / args.ccf5_pattern)))
            if not ccf5_paths:
                parser.error(f"No CCF5 files matched: {Path(args.input_ccf5_dir) / args.ccf5_pattern}")
        else:
            ccf5_paths = args.input_ccf5
            missing = [p for p in ccf5_paths if not Path(p).exists()]
            if missing:
                parser.error(f"CCF5 file does not exist: {missing[0]}")

        try:
            arrays = build_stage1_arrays_from_ccf5(
                ccf5_paths=ccf5_paths,
                window_len=args.window_len,
                stride=args.stride,
                max_events=args.max_events,
                penalty=args.penalty,
                min_event_len=args.min_event_len,
                event_backend=args.event_backend,
                min_signal_len=args.min_signal_len,
                normalize=args.normalize,
                value_min=args.value_min,
                value_max=args.value_max,
                sample_id=args.sample_id,
                condition=args.condition,
                max_reads=args.max_reads,
                max_windows=args.max_windows,
                trim_head=args.trim_head,
                trim_tail=args.trim_tail,
            )
        except ImportError as exc:
            parser.error(str(exc))

    out_npz = out_dir / f"{args.prefix}.stage1_windows.npz"
    save_stage1_npz(arrays, out_npz)

    summary = json.loads(str(arrays["summary_json"][0]))
    summary["event_feature_names"] = EVENT_FEATURE_NAMES
    summary_path = out_dir / f"{args.prefix}.stage1_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[DONE] Stage1 windows saved")
    print("  npz:    ", out_npz)
    print("  summary:", summary_path)
    print("  raw_signal:    ", arrays["raw_signal"].shape)
    print("  event_features:", arrays["event_features"].shape)


if __name__ == "__main__":
    main()
