# AGENTS.md

本文件是 DeepSpot 项目的仓库级协作说明。后续 agent 在本目录内工作时必须优先遵守本文档。

## Project Goal

DeepSpot 的目标是设计和实现一个面向纳米孔测序信号的开放集修饰发现框架。

核心问题不是只识别少数已知修饰，而是：

1. 学习 canonical / unmodified 序列在纳米孔中的正常信号分布。
2. 在天然样本中发现稳定偏离 canonical 信号的位点或 read 群体。
3. 将偏离信号与人工合成修饰数据中的已知修饰指纹进行匹配。
4. 对不匹配已知指纹但具有稳定重复模式的信号，输出 unknown modification candidates。

不要把本项目简化成封闭类别分类任务，例如只训练 `modified` vs `unmodified` 或只训练若干已知修饰类别。

长期推荐的任务表述是：

```text
sequence-conditioned canonical signal modeling
+ open-set modification discovery
+ known modification fingerprint matching
+ unknown modification clustering
```

当前代码已经采用分阶段路线：

```text
Stage 1:
  no-reference open-set signal-state discovery
  128-point raw signal windows
  variable-number event segmentation
  unmodified-trained normal signal model
  anomaly score + residual fingerprint + signal-state clusters

Stage 2:
  event/read-window -> base/k-mer/reference anchoring
  read-site canonical residual
  site-level aggregation

Stage 3:
  synthetic known-mod fingerprint matching
  unknown modification candidate clustering
  replicate and batch-aware validation
```

Stage 1 的输出不是最终修饰位点识别，而是无位点的异常信号状态发现。后续若获得 event/base/reference mapping，再把 Stage 1 的异常窗口锚定到碱基或 reference position。

## Safety Rules

本仓库可能包含重要源码、原始测序数据、信号文件、实验结果和中间缓存。必须保守操作。

不要盲目删除文件。禁止运行以下破坏性命令，除非用户明确要求并确认具体范围：

```bash
rm -rf *
rm -rf .
rm -rf ./*
find . -delete
git clean -fdx
git reset --hard
```

不要擅自覆盖原始数据、参考基因组、索引文件、训练好的模型权重或实验结果目录。

如果需要清理临时文件，只能清理明确由当前任务新生成、且路径清楚的文件。

## Data Assumptions

项目围绕三类数据设计：

1. `unmodified`：不带修饰数据，用于学习 canonical signal model。
2. `natural_modified`：天然带修饰数据，用于开放集发现，允许同时包含已知修饰、未知修饰和技术噪声。
3. `synthetic_modified`：人工合成修饰数据，用作已知修饰指纹库、灵敏度评估和阈值校准，而不是完整标签空间。

天然修饰数据不能被默认视为干净正例。它可能包含：

- 已知修饰
- 未知修饰
- 多种修饰混合
- 部分修饰比例
- 测序和比对噪声
- batch effect
- basecalling error

因此，天然样本中的标签必须谨慎处理，优先使用无监督、半监督、开放集或弱监督方法。

## Modeling Principles

### Current Stage 1 Contract

当前第一阶段代码位于：

```text
src/deepspot/stage1_data.py
src/deepspot/stage1_events.py
src/deepspot/stage1_pca.py
src/deepspot/stage1_model.py
scripts/stage1_build_windows.py
scripts/stage1_train_pca.py
scripts/stage1_infer_pca.py
scripts/stage1_cluster_scores.py
scripts/stage1_train_autoencoder.py
scripts/stage1_infer_autoencoder.py
docs/stage1_usage.md
```

Stage 1 的输入假设是 JSONL，每行至少包含：

```json
{"read_id": "read_001", "signal": [0.1, 0.2, 0.3]}
```

可选字段包括：

```text
label
pattern
mod_type
sample_id
condition
```

Stage 1 的最小数据单元是：

```text
read_id + window_start + 128-point normalized raw signal + variable event sequence
```

变点检测得到的 event 数量必须允许变化。不要假设一个 128 点窗口一定对应 5 个 event，也不要因为 k-mer 是 5-mer 就强行把信号切成 5 个台阶。5-mer 是信号生成背景，不是窗口内 event 数量约束。

Stage 1 输出包括：

```text
total_score
anomaly_z
latent
raw_residual
read_ids
window_starts
num_events
cluster_id
```

这些输出只能解释为：

```text
open-set signal-state anomaly
candidate abnormal signal pattern
```

不能直接解释为：

```text
reference position modification call
confirmed chemical modification
```

### Stage 1 Model Preference

优先先跑无深度学习依赖的 PCA baseline：

```text
scripts/stage1_train_pca.py
scripts/stage1_infer_pca.py
scripts/stage1_cluster_scores.py
```

原因：

- 依赖少，当前默认 Python 环境可运行。
- 适合快速检查 unmodified / natural / synthetic 是否有可重复 signal-state 差异。
- 方便确认变点检测、窗口切分和 metadata 是否正确。

如果环境中有 PyTorch，再使用 raw/event 双分支 autoencoder：

```text
scripts/stage1_train_autoencoder.py
scripts/stage1_infer_autoencoder.py
```

不要在未验证数据表示和泄漏控制前直接堆复杂模型。

### Canonical Model First

长期目标中，优先从不带修饰数据学习：

```text
P(signal | sequence context, read quality, pore state, speed, batch)
```

模型应该理解不同 k-mer / sequence context 下的正常纳米孔信号，而不是直接记忆某个数据批次。

在 Stage 1 尚无 reference/k-mer mapping 时，临时降级为：

```text
P(raw/event window | unmodified signal-state distribution)
```

这只能用于信号状态异常发现，不能替代 reference-conditioned canonical model。

候选输入特征包括：

- raw signal window
- event-level mean current
- event-level standard deviation
- dwell time
- signal residual
- basecaller logits or probabilities
- alignment features
- k-mer context
- read quality
- batch / flowcell / chemistry metadata

### Open-Set Detection

天然样本预测时，长期优先输出：

- read-level anomaly score
- site-level anomaly score
- modified read fraction
- residual fingerprint
- known modification similarity
- unknown modification cluster id
- confidence score / FDR

不要只输出单一二分类标签。

在 Stage 1 中，优先输出窗口级：

- `read_id`
- `window_start`
- `num_events`
- `total_score`
- `anomaly_z`
- `latent`
- `raw_residual`
- `cluster_id`

后续拿到 mapping 后，再把窗口级异常聚合到 read-base 或 reference-site 层面。

### Known and Unknown Must Be Separated

人工合成修饰数据只能代表已知修饰指纹的一部分。

天然样本中的异常信号应分为：

```text
canonical-like
known-mod-like
unknown-stable-cluster
technical-noise-like
low-confidence
```

如果异常信号不匹配人工合成修饰，不要直接判为阴性。它可能是未知修饰候选。

### Site-Level Aggregation

单条 read 的异常不能直接等同于修饰位点。应在 site / position 层面聚合：

- coverage
- 支持异常的 read 数
- modified fraction
- replicate consistency
- residual cluster stability
- quality filters

低覆盖、低质量、高错配或低复杂度区域必须降权或过滤。

## Experimental Design Requirements

### Stage 1 Validation

Stage 1 的验证重点不是“准确叫出修饰位点”，而是确认无位点信号异常发现是否可靠：

1. unmodified held-out 窗口上的 anomaly score 分布。
2. natural 窗口是否出现高于 unmodified 背景的稳定异常模式。
3. synthetic known-mod 窗口是否在 anomaly score 或 latent fingerprint 上形成可重复模式。
4. 高异常 cluster 是否由少数低质量 read、单个 batch、极端 event 数或切分失败主导。
5. 同一套变点检测参数必须用于 unmodified / natural / synthetic。

Stage 1 不允许声称检测到具体 reference 位点修饰。报告时使用：

```text
abnormal signal window
open-set signal-state cluster
known-mod-like signal pattern
unknown signal-state candidate
```

### Batch Effect

纳米孔信号强烈受 batch、flowcell、chemistry、basecaller、library preparation 和 alignment pipeline 影响。

实现新方法时必须考虑：

- 同一 pipeline 处理三类数据
- 显式记录 batch metadata
- 对 batch effect 做可视化或统计检查
- 避免模型只学会区分数据来源
- 做跨 batch / 跨 replicate 验证

### Validation

验证应至少包含：

1. unmodified 数据上的 false positive rate。
2. synthetic modified 数据上已知修饰位点的 detection sensitivity。
3. natural modified 数据中的 replicate consistency。
4. unknown candidates 的 residual fingerprint 聚类稳定性。
5. 候选位点与 coverage、quality、mapping error 的关系检查。

对未知修饰候选，应明确写成 `candidate`，不能声称已确定化学身份，除非有额外实验支持。

## Recommended Repository Structure

如果需要创建项目结构，优先使用以下布局：

```text
DeepSpot/
  AGENTS.md
  README.md
  configs/
  data/
    raw/
    interim/
    processed/
  docs/
  notebooks/
  results/
  scripts/
  src/
    deepspot/
  tests/
```

原始数据应放在 `data/raw/`，并默认不被代码覆盖。

中间文件、特征文件和模型输出应写入 `data/interim/`、`data/processed/`、`results/` 或用户指定目录。

当前 Stage 1 相关代码和文档位置：

```text
docs/stage1_usage.md
scripts/stage1_build_windows.py
scripts/stage1_train_pca.py
scripts/stage1_infer_pca.py
scripts/stage1_cluster_scores.py
scripts/stage1_train_autoencoder.py
scripts/stage1_infer_autoencoder.py
src/deepspot/stage1_data.py
src/deepspot/stage1_events.py
src/deepspot/stage1_pca.py
src/deepspot/stage1_model.py
```

## Coding Guidelines

优先写可复现、可检查、可替换的代码。

关键要求：

- pipeline 参数放入配置文件或命令行参数。
- 不要在代码中硬编码用户机器上的绝对路径。
- 所有随机过程设置 seed，并记录 seed。
- 数据拆分必须避免 read / site / batch 泄漏。
- 模型训练输出必须保存配置、代码版本、数据版本、指标和日志。
- 长时间运行任务应支持 resume 或 checkpoint。
- 对数据格式、坐标系统和 strand 方向写清楚注释或文档。

## Preferred Implementation Order

新增能力时优先按以下顺序推进：

1. Stage 1 JSONL raw-signal window builder。
2. Stage 1 variable event segmentation and event-feature QA。
3. Stage 1 unmodified-trained PCA baseline。
4. Stage 1 natural/synthetic inference and anomaly score calibration。
5. Stage 1 high-anomaly signal-state clustering。
6. 数据 manifest 和 sample/batch metadata schema。
7. event/read-window -> base/k-mer/reference anchoring。
8. reference-conditioned canonical signal baseline。
9. read-site anomaly score 和 residual fingerprint。
10. site-level aggregation。
11. synthetic known-mod fingerprint matching。
12. unknown residual clustering。
13. 评估、可视化和报告输出。

## Reporting Standards

Stage 1 窗口级结果表建议至少包含：

```text
sample_id
condition
read_id
window_start
num_events
total_score
anomaly_z
cluster_id
label
mod_type
quality_flags
```

reference anchoring 之后，位点级结果表建议至少包含：

```text
sample_id
position
reference_base
kmer_context
coverage
modified_read_count
modified_fraction
anomaly_score
known_mod_similarity
unknown_cluster_id
confidence
fdr
quality_flags
```

对外报告时，区分以下术语：

- `abnormal signal window`
- `open-set signal-state cluster`
- `detected signal deviation`
- `known modification-like signal`
- `unknown modification candidate`
- `confirmed chemical modification`

不要把 `unknown modification candidate` 写成已确认修饰。

## Collaboration Notes

修改代码前先阅读相关文件和现有风格。

如果仓库中已有实现，优先保持现有接口和目录习惯。

如果用户询问方案设计，优先围绕开放集发现、canonical counterfactual、残差信号指纹、聚类稳定性和实验验证展开。

如果用户要求直接实现，先做最小可运行 baseline，再逐步增强模型复杂度。
