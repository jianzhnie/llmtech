# MindSpeed 长序列训练：算法演进与实战指南

## 1. 概述：从 8K 研发到 64K 的演进策略

在进行长序列（Context Length > 32K）优化前，必须确保 **短序列（如 8K）基础性能** 已达标。

### 核心调优原则

- **显存优先**：利用融合算子（`use-fused-swiglu`、`use-fused-rmsnorm`）和分布式优化器（`use-distributed-optimizer`）压低基础显存。
- **通信掩盖**：开启 `overlap-grad-reduce` 实现梯度同步与计算的重叠。
- **策略选择**：在 GQA（Grouped Query Attention）场景下，优先使用 **流水并行 (PP)** 或 **上下文并行 (CP)** 代替张量并行 (TP)，以避免由于 Head 数限制导致的算力利用率下降。

## 2. 三大上下文并行 (CP) 算法对比

为了突破单卡 KV Cache 的显存瓶颈，MindSpeed 提供了三种主流的上下文并行算法：

| **算法方案**      | **切分维度**        | **通信模式**        | **核心优势**               | **适用场景**                      |
| ----------------- | ------------------- | ------------------- | -------------------------- | --------------------------------- |
| **Ulysses**       | 注意力头 (Head/Dim) | All-to-All          | 通信效率高，计算密集度高   | 满足 `heads % CP == 0` 的标准模型 |
| **RingAttention** | 序列长度 (Sequence) | Ring P2P (SendRecv) | 显存优化极佳，逻辑简单     | 超长序列（64K+）、高带宽环境      |
| **Hybrid (CP)**   | 混合维度 (S + Dim)  | All-to-All + Ring   | 兼顾二者，突破 Head 数限制 | **GQA 模型、跨节点大规模集群**    |

## 3. Ulysses：基于注意力头切分的方案

**Ulysses** 是长序列训练的首选方案，尤其在高性能网络（如昇腾 HCCS）中表现优异。

### 3.1 核心逻辑

通过两次 **All-to-All** 转置，将数据在“序列维度切分”与“注意力头维度切分”之间转换。

1. **输入**：各卡持有序列的 $1/P$。
2. **通信 1**：All-to-All 转置，使得每张卡持有全量序列，但仅持有部分的 Head。
3. **计算**：在本地独立完成该 Head 的 Attention 计算。
4. **通信 2**：All-to-All 还原回序列切分状态。

### 3.2 开启方式与约束

```bash
--context-parallel-algo ulysses_cp_algo \
--context-parallel-size 8 \
--group-query-attention \
--num-query-groups 8  # 必须满足 num_query_groups >= cp_size
```

## 4. RingAttention：环形序列并行

**RingAttention** 采用分块计算（Blockwise）的思想。

### 4.1 性能关键：通算掩盖

为了让计算掩盖通信延迟，每个分块（Block）的序列长度 $c$ 必须满足：
$$
c \geq \frac{F \text{ (设备算力)}}{B \text{ (通信带宽)}}
$$

- **优化路径**：RSA (早期) $\rightarrow$ BPT (分块) $\rightarrow$ **MindSpeed RingAttention (并行环形 P2P)**。
- **开启方式**：`--context-parallel-algo megatron_cp_algo`。

## 5. Hybrid CP：MindSpeed 的最佳实践

**Hybrid** 方案是目前处理超长序列（如 64K/128K）且兼顾多节点扩展的最优解。

### 5.1 混合架构逻辑

它将 CP 维度细分为 **Ulysses 子组** 和 **Ring 子组**：

- **节点内 (Intra-node)**：使用 Ulysses 运行 All-to-All，利用高速内部带宽处理维度转置。
- **节点间 (Inter-node)**：使用 RingAttention 运行 P2P 通信，降低跨机带宽压力。

### 5.2 最佳实践配置

```bash
--context-parallel-size 8 \
--context-parallel-algo hybrid_cp_algo \
--ulysses-degree-in-cp 2  # 总CP为8，其中Ulysses占2，Ring占4
```

> 💡 **专家建议**：确保 `TP * ulysses_size <= 8`（单机卡数），将 All-to-All 限制在节点内，跨节点仅走 P2P 通信，此时系统效率最高。

## 6. 实战案例：ChatGLM3-6B 32K/64K 调优

在 ChatGLM3 的长序列预训练中，NPU A2 配合 MindSpeed 展现了卓越性能。

### 性能表现对比

| **硬件环境** | **序列长度** | **并行策略**            | **吞吐量 (Token/s/p)** |
| ------------ | ------------ | ----------------------- | ---------------------- |
| GPU A100     | 32K          | Megatron-LM             | 2887.84                |
| **NPU A2**   | **32K**      | **MindSpeed (Ulysses)** | **3006.24** (↑ 4%)     |

### 核心优化路径总结

1. **关闭 TP/PP**：针对 GQA=2 的模型，TP 容易导致算力浪费，优先切分序列维度。
2. **使能 CP=8**：使用 Ulysses 解决 KV Cache 爆炸。
3. **融合算子全开**：`use-fused-swiglu` 和 `use-fused-rmsnorm` 是长序列不 OOM 的基础。
4. **通信重叠**：开启 `overlap-grad-reduce` 隐藏反向传播的通信开销。
