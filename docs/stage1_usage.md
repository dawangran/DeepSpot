# Stage 1 Usage

第一阶段目标是做无位点的 open-set signal-state discovery：

```text
128 点 raw signal window
  -> 变点检测得到可变数量 events
  -> raw/event 双表示
  -> 用 unmodified 训练正常模型
  -> natural/synthetic 推理 anomaly score 和 residual fingerprint
```

## 1. 构建窗口

支持两类输入：

```text
JSONL: 每行一个 read，至少包含 read-level signal
CCF5: 一个或多个 .ccf5 文件，直接读取 read signal
```

JSONL 每行至少包含：

```json
{"read_id": "read_001", "signal": [0.1, 0.2, 0.3], "label": "", "mod_type": ""}
```

命令：

```bash
python3 scripts/stage1_build_windows.py \
  --input_jsonl data/unmodified.jsonl \
  --out_dir results/stage1/unmodified \
  --prefix unmodified \
  --sample_id U01 \
  --condition unmodified \
  --window_len 128 \
  --stride 64 \
  --max_events 16 \
  --penalty 8.0 \
  --min_event_len 5
```

CCF5 单文件或多文件：

```bash
python3 scripts/stage1_build_windows.py \
  --input_ccf5 data/raw/unmodified_01.ccf5 data/raw/unmodified_02.ccf5 \
  --out_dir results/stage1/unmodified \
  --prefix unmodified \
  --sample_id U01 \
  --condition unmodified \
  --window_len 128 \
  --stride 64 \
  --max_events 16 \
  --penalty 8.0 \
  --min_event_len 5 \
  --trim_head 2000 \
  --trim_tail 2000
```

CCF5 目录批量输入：

```bash
python3 scripts/stage1_build_windows.py \
  --input_ccf5_dir data/raw/unmodified \
  --ccf5_pattern "*.ccf5" \
  --out_dir results/stage1/unmodified \
  --prefix unmodified \
  --sample_id U01 \
  --condition unmodified
```

使用 CCF5 输入时需要额外安装 `pyccf5`。`trim_head` 和 `trim_tail` 只作用于 CCF5，用于去掉 read 首尾不稳定 raw signal；输出的 `window_start` 保持为原 read raw-signal 坐标。

输出：

```text
unmodified.stage1_windows.npz
unmodified.stage1_summary.json
```

## 2. 训练正常模型

推荐先用无深度学习依赖的 PCA baseline 跑通全流程：

```bash
python3 scripts/stage1_train_pca.py \
  --train_npz results/stage1/unmodified/unmodified.stage1_windows.npz \
  --out_dir results/stage1/pca_model \
  --n_components 32
```

如果环境里有 PyTorch，再训练 raw/event 双分支 autoencoder：

```bash
python3 scripts/stage1_train_autoencoder.py \
  --train_npz results/stage1/unmodified/unmodified.stage1_windows.npz \
  --out_dir results/stage1/model \
  --epochs 30 \
  --batch_size 512
```

训练时按 `read_id` 切分 train/val，避免相邻窗口泄漏到验证集。

## 3. 推理打分

对 natural 或 synthetic 使用同一个 PCA 模型：

```bash
python3 scripts/stage1_infer_pca.py \
  --input_npz results/stage1/natural/natural.stage1_windows.npz \
  --model_npz results/stage1/pca_model/stage1_pca_model.npz \
  --out_dir results/stage1/natural_scores \
  --prefix natural \
  --write_tsv
```

如果使用 PyTorch autoencoder：

```bash
python3 scripts/stage1_infer_autoencoder.py \
  --input_npz results/stage1/natural/natural.stage1_windows.npz \
  --model_ckpt results/stage1/model/stage1_autoencoder_best.pt \
  --out_dir results/stage1/natural_scores \
  --prefix natural \
  --write_tsv
```

PCA 核心输出：

```text
natural.stage1_pca_scores.npz
natural.stage1_pca_scores.tsv
natural.stage1_pca_infer_summary.json
```

`stage1_pca_scores.npz` 中包括：

```text
total_score
anomaly_z
latent
raw_residual
read_ids
window_starts
num_events
```

PyTorch autoencoder 输出名为 `natural.stage1_scores.npz`，并额外包含 `raw_mse` 和 `event_mse`。

## 4. 聚类高异常窗口

对高异常窗口做 open-set signal-state clustering：

```bash
python3 scripts/stage1_cluster_scores.py \
  --score_npz results/stage1/natural_scores/natural.stage1_pca_scores.npz \
  --out_dir results/stage1/natural_clusters \
  --prefix natural \
  --min_anomaly_z 3 \
  --n_clusters 8
```

输出：

```text
natural.tsv
natural.npz
natural.summary.json
```

这里的 cluster 是无位点的 signal-state cluster，不等同于已确认修饰类型。

## 重要边界

这一阶段的输出是：

```text
read_id + window_start + anomaly score + residual fingerprint
```

还不是：

```text
reference position + modification call
```

后续如果拿到 event/base/reference mapping，再把这些异常窗口锚定到位点层面。
