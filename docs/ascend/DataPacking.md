# 大模型指令微调模式全解析：从 Padding 到 Packing

## 概述

在大型语言模型（LLM）的指令微调（Instruction Tuning）任务中，**序列长度不一致**是数据处理面临的核心挑战。为了提高 GPU/NPU 的算力利用率并避免显存浪费，业界发展出了三种主要的数据对齐策略：**固定 Padding（Fixed Padding）**、**动态 Padding（Dynamic Padding）** 以及 **样本拼接（Packing）**。

本文将深入探讨这三种模式的实现原理、性能表现及生产环境下的最佳实践。

## 一、 固定 Padding 模式（Fixed Padding）

### 1.1 定义与实现

固定 Padding 是最基础的对齐方式。它将批次（Batch）内所有样本统一填充至一个预设的最大长度 $L_{max}$（如 2048）。

**实现步骤：**

1. 预定义全局最大序列长度 $seq\_len$。
2. **填充（Padding）：** 若样本长度 $L < seq\_len$，则在末尾补充 `padding_token`。
3. **截断（Truncation）：** 若样本长度 $L > seq\_len$，则强制截断。

### 1.2 优缺点分析

- **优点：**
  - 逻辑简单，对底层算子友好；
  - 所有序列长度一致，原生支持各种并行策略（如 Tensor Parallelism）。
- **缺点：**
  - 内存与效率低，大量填充导致内存利用率低  例如：若 $seq\_len=2048$ 而平均样本长度仅为 100，$95\%$ 内存都浪费了
  - 计算效率低，即使被 mask 掉，padding 部分仍需参与矩阵乘法等计算，造成不必要的 FLOPs 消耗

## 二、 动态 Padding 模式（Dynamic Padding）

在大模型训练中，尤其是**指令微调（Instruction Tuning）**任务中，输入文本长度差异较大（如文档分类、机器翻译等）。传统的固定 `padding` 会导致大量内存浪费和计算冗余。**动态 padding** 是一种更高效的处理方式，能够显著提升内存利用率与模型泛化能力。

### 2.1 核心机制

动态 Padding 摒弃了全局统一长度，转而根据当前 Batch 内的最长样本长度 $L_{batch\_max}$ 动态调整填充目标。

### 2.2 pad_to_multiple_of 优化

为了兼顾效率与硬件亲和性（如硬件指令通常要求向量长度对齐），通常配合 `pad_to_multiple_of` 参数使用。
$$
L_{final} = \lceil \frac{L_{batch\_max}}{N} \rceil \times N
$$
其中 $N$ 通常取值为 8、128 或 256。

### 2.3 实验数据分析（Qwen-7B）

在华为昇腾（Ascend）环境下的 Qwen-7B 实验表明，动态 Padding 能显著缓解显存压力：

| **并行策略**   | **模式**         | **单步迭代时间 (s)** | **显存占用 (MB)** |
| -------------- | ---------------- | -------------------- | ----------------- |
| TP=8, PP=11    | 动态 Padding     | 7.7                  | 25,051            |
| TP=8, PP=1     | 固定 Padding     | 25.8                 | 32,960            |
| TP=4, PP=2     | 动态 Padding     | 4.2                  | 26600             |
| TP=4, PP=2     | 固定 Padding     | 27.9                 | 41512             |
| TP=4, PP=1     | 动态 Padding     | 4.3                  | 45625             |
| TP=4, PP=1     | 固定 Padding     | /                    | **OOM**           |
| TP=2, PP=2     | 动态 Padding     | 2.7                  | 48800             |
| TP=2, PP=2     | 固定 Padding     | /                    | **OOM**           |
| TP=2, PP=4     | 动态 Padding     | 2.5                  | 31000             |
| TP=2, PP=4     | 固定 Padding     | /                    | **OOM**           |
| **TP=1, PP=8** | **动态 Padding** | **2.0**              | **42000**         |
| TP=1, PP=8     | 固定 Padding     | /                    | **OOM**           |
| TP=1, PP=4     | 动态 Padding     | 2.1                  | 59826             |
| TP=1, PP=4     | 固定 Padding     | /                    | **OOM**           |

> **结论：** 动态 Padding 可将显存占用降低 $30\% \sim 50\%$，单步迭代速度最高提升 7 倍，且完全不影响模型精度。

## 三、 样本拼接模式（Packing）

在大模型训练中，尤其是**指令微调（Instruction Tuning）\**任务中，输入序列长度差异较大。传统的 `padding` 方法会导致大量内存浪费和计算冗余。\*\*Packing\*\* 是一种更高效的处理方式，通过将多个短序列\**拼接成一个长序列**，显著提升内存与计算效率。

### 3.1 核心思想

Packing（或称样本打包）是目前长序列训练的最优解。它将多个短序列拼接成一条长度为 $seq\_len$ 的长序列，中间插入结束符（EOD），使每个 Batch 几乎填满有效数据。

### 3.2 关键技术点

- **注意力屏蔽（Attention Mask）：** 必须确保不同样本之间不会产生跨样本的注意力干扰。

- 位置编码重置（Reset Position IDs）： 在拼接处需重置位置索引，使每个子样本从 $0$ 开始计算位置。

  $$Pos(x_i) = i - \text{offset}(\text{sample})$$

### 3.3 性能优势

基于 `flan_20k` 数据集的测试显示：

- **显存利用率：** Packing 模式下显存增长随 Batch Size 增加表现极其平缓，有效避免 OOM。
- **吞吐量（Throughput）：** 在高负载场景下，Packing 的 TPS（Tokens Per Second）可达固定 Padding 的 2 倍以上。

## 四、 综合总结与选型指南

### 4.1 三大模式横向对比

| **维度**       | **固定 Padding** | **动态 Padding**  | **Packing**          |
| -------------- | ---------------- | ----------------- | -------------------- |
| **计算效率**   | 低               | 中                | **极高**             |
| **显存压力**   | 极大             | 较小              | **极小**             |
| **实现复杂度** | 极简             | 中等              | 较高（需预处理脚本） |
| **硬件扩展性** | 好               | 受限（不支持 CP） | **极好（支持 CP）**  |

### 4.2 最佳实践建议

1. **首选 Packing 模式：** 特别是进行大规模指令微调（SFT）时，Packing 能最大化硬件吞吐并支持上下文并行（Context Parallel）。
2. **退而求其次选动态 Padding：** 当数据分布极其不均且无法使用 Packing 脚本时，动态 Padding 是保底方案。
3. **慎用固定 Padding：** 仅推荐用于短序列的快速原型验证。

### 4.3 常用参数示例（以 Transformers 为例）

```python
collator = DataCollatorForSeq2Seq(
    tokenizer,
    pad_to_multiple_of=128, # 对硬件友好的动态对齐
    padding=True,
    pack_long_sequences=True # 若框架支持则开启 Packing
)
```
