# 精准预估：LLM 推理需要多少 GPU 显存？

许多开发者在部署 LLM 时会遇到“显存溢出（OOM）”的问题，即使他们的 GPU 显存看起来比模型文件大得多。这是因为 **模型权重（Weights）只是显存占用的一部分**。在实际运行中，KV Cache（键值缓存）和系统开销（Overhead）才是决定你能否支持长文本和高并发的关键。

## 1. 显存计算的“三大支柱”

部署一个 LLM，GPU 显存主要被以下三个部分瓜分：

### 1.1 模型权重（Static Weights）—— 静态占用

这是最容易计算的部分。它取决于模型的**参数量**和**数值精度**。

- 计算公式：

  $$
  M_{weights} \approx P \times \frac{Q}{8} \times (1 + \text{Overhead})
  $$

  其中：

  - $P$：参数量（单位：Billion/十亿）。
  - $Q$：位精度（如 FP16=16, INT8=8, INT4=4）。
  - $\text{Overhead}$：框架加载开销（通常估算为 10%~20%）。

> **速查表：以 70B 模型为例**
>
> - **FP16/BF16 (2 bytes)**：约 140 GB
> - **INT8 (1 byte)**：约 70 GB
> - **INT4 (0.5 bytes)**：约 35 GB

### 1.2 KV Cache —— 动态增长的核心

KV Cache 存储了注意力机制中已生成的键值对，避免重复计算。它的占用随**序列长度**和**并发请求数**线性增长。

- 计算公式（简化版）：

$$
M_{kv} \approx 2 \times \text{Layers} \times \text{Heads} \times \text{Dim} \times \text{SeqLen} \times \text{BatchSize} \times \text{BytesPerParam}
$$

为什么它很重要？

在一个 70B 模型中，如果上下文长度达到 32K，KV Cache 的占用可能超过 15-20 GB。如果你想支持 10 个并发用户，仅 KV Cache 就会吃掉几百 GB 显存。

### 1.3 激活值与系统缓冲区（Activations & Buffers）

这部分包括推理过程中的临时中间状态、显存对齐填充以及 CUDA 内核启动所需的缓冲区。通常预留 **1-2 GB** 即可，但在高并发下会略微增加。

## 2. 综合预估公式

为了在实践中快速决策，你可以使用这个综合公式：
$$
Total\_VRAM \approx \text{Model\_Size\_GB} + \text{KV\_Cache\_per\_Token} \times \text{Total\_Tokens} + \text{System\_Margin}
$$

- **Total Tokens** = 并发请求数 $\times$ (输入长度 + 输出长度)
- **System Margin**：建议预留 10% 的安全边际。

## 3. 典型配置推荐

基于 BentoML 的基准测试建议：

| **模型规模**     | **推荐精度** | **推荐 GPU 配置**               | **适用场景**       |
| ---------------- | ------------ | ------------------------------- | ------------------ |
| **8B (Llama 3)** | FP16         | 1x NVIDIA L4 (24GB) 或 A10)     | 个人助手、简单摘要 |
| **8B (Llama 3)** | INT4         | 1x NVIDIA T4 (16GB)             | 低成本边缘部署     |
| **70B (Qwen2)**  | FP16         | 4x NVIDIA A100 (80GB) 或 8x L40 | 企业级高并发对话   |
| **70B (Qwen2)**  | INT4         | 1x NVIDIA A100 (80GB)           | 兼顾性能与单机部署 |

## 4. 如何优化显存占用？

如果你发现显存不足，可以从以下三个维度“节流”：

1. **量化（Quantization）**：将 FP16 降至 INT4，显存直接压缩 75%。
2. **分页注意力（PagedAttention）**：如使用 vLLM 框架，它可以极大地减少 KV Cache 的碎片浪费。
3. **模型并行（Parallelism）**：通过 Tensor Parallelism (TP) 将模型分摊到多块显卡上。

## 5. 总结

在规划硬件时，**永远不要只看模型大小**。一个能“跑起来”的模型不代表它能提供“长文本支持”。务必为 KV Cache 留足空间，这才是决定你应用上限的关键。
