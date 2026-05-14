# DeepSpot

DeepSpot 的目标是面向纳米孔测序信号进行开放集修饰发现，重点不是把数据简单分成 `modified` / `unmodified`，而是先学习 canonical 信号分布，再发现稳定偏离的 signal-state，并进一步做 known-mod fingerprint matching 和 unknown candidate clustering。

## Stage 1 window builder

当前 Stage 1 的默认 window/chunk 处理规则是：

1. 先按 `window_len` 和 `stride` 切 window。
2. 每个 window 单独做 robust 标准化：
   `x' = (x - median) / MAD`
3. 仅保留标准化后所有值都落在 `[value_min, value_max]` 的 window。
4. 只要有任意值越界，整个 window 直接剔除。

对应脚本：

- [`scripts/stage1_build_windows.py`](scripts/stage1_build_windows.py)
- [`reference/jsonl_chunk.py`](reference/jsonl_chunk.py)
- [`reference/ccf5_chunk.py`](reference/ccf5_chunk.py)

默认参数：

- `window_len=128`
- `stride=64`
- `normalize=window_mad`
- `value_min=-3.0`
- `value_max=3.0`
- `event_backend=simple`

## JSONL 输入

每行至少包含：

```json
{"read_id": "read_001", "signal": [0.1, 0.2, 0.3]}
```

示例：

```bash
python3 scripts/stage1_build_windows.py \
  --input_jsonl data/unmodified.jsonl \
  --out_dir results/stage1/unmodified \
  --prefix unmodified \
  --window_len 128 \
  --stride 64 \
  --normalize window_mad \
  --value_min -3.0 \
  --value_max 3.0
```

## CCF5 输入

示例：

```bash
python3 scripts/stage1_build_windows.py \
  --input_ccf5_dir data/raw/unmodified \
  --ccf5_pattern "*.ccf5" \
  --out_dir results/stage1/unmodified \
  --prefix unmodified \
  --window_len 128 \
  --stride 64 \
  --normalize window_mad \
  --value_min -3.0 \
  --value_max 3.0
```

## Notes

- 这里的过滤是按“标准化后的整段 window/chunk”判断，不是按单点局部裁剪。
- `value_min` / `value_max` 过滤失败的 window/chunk 会整段丢弃。
- Stage 1 的后续窗口建模仍建议保持 read-level split，避免泄漏。
- event segmentation 默认使用内置 `simple` 后端；安装 `ruptures` 后可以用 `--event_backend ruptures`。
