# 大模型训练效率度量：MFU 计算深度解析

## 概述

在深度学习领域，**MFU (Model FLOPs Utilization，模型浮点运算利用率)** 是衡量大模型训练效率的核心指标。它描述了模型在实际训练过程中有效利用硬件理论计算能力的程度。对于 GPT 和 Llama 等参数量巨大的 Transformer 模型，监控并优化 MFU 是提升训练吞吐量、降低算力成本的关键。


## 1. MFU 的核心概念

### 1.1 定义

MFU 定义为模型在实际训练过程中每秒执行的有效浮点运算次数与硬件理论峰值算力的比值：

$$
MFU = \frac{\text{Actual FLOPs}}{\text{Theoretical Peak FLOPs}}
$$

### 1.2 关键要素

- **Actual FLOPs (实际达到的 FLOPs)**：指模型执行前向传播（Forward）和反向传播（Backward）化器更新等所需优所有计算。
- **Theoretical Peak FLOPs (硬件理论峰值)**：硬件（如 NVIDIA H100 或 Ascend 910）在特定精度（如 FP16/BF16）下的标称最大吞吐量。

### 1.3 核心意义

- **诊断瓶颈**：低 MFU 通常意味着存在通信带宽瓶颈（Communication Bound）、内存带宽限制（Memory Bound）或算子实现效率低下。
- **优化指导**：通过调整 Batch Size、并行策略（TP/PP/DP）或使用混合精度训练来逼近硬件极限。

## 2. Transformer 模型的 FLOPs 推导

以标准的 GPT 架构为例，详细推导其在训练过程中每步所需的浮点运算量（FLOPs），为评估大模型训练效率提供理论基础。

### 2.1 FLOPs 定义

- **FLOPs**：表示完成一次矩阵乘法所需的浮点操作次数。
- 对于一个 $ A_{m \times k} \times B_{k \times n} $ 的矩阵乘法：
  - 所需 FLOPs 数量为：
    $$ 2 \times m \times k \times n $$
  - 包含：
    - $ m \times k \times n $ 次乘法操作
    - $ m \times k \times (n - 1) $ 次加法操作
    - 总计约 $ 2mk n $

### 2.2 Transformer 层的 FLOPs 分析

#### 符号定义

| **符号** | **含义**                   | **符号** | **含义**                      |
| -------- | -------------------------- | -------- | ----------------------------- |
| $b$      | Batch Size (批大小)        | $h$      | Hidden Dimension (隐藏层维度) |
| $s$      | Sequence Length (序列长度) | $V$      | Vocabulary Size (词表大小)    |
| $L$      | Number of Layers (层数)    | $4h$     | MLP 中间层维度                |

#### 1. Self-Attention 模块 FLOPs

##### 公式：

```math
Q, K, V = xW_Q, xW_K, xW_V \\
\text{attn\_score} = \text{softmax}\left(\frac{QK^T}{\sqrt{h}}\right) \\
\text{context} = \text{attn\_score} * V \\
\text{proj} = \text{context} * W_O
```

##### FLOPs 计算：

| 操作                                     | 输入形状                 | 输出形状  | FLOPs                       |
| ---------------------------------------- | ------------------------ | --------- | --------------------------- |
| $ [b,s,h] \to [b,s,h] $（Q/K/V 投影）    | $[b,s,h] \times [h,h]$   | $[b,s,h]$ | $3 \times 2bs h^2 = 6bsh^2$ |
| $ [b,s,h] \to [b,s,h] $（QKᵀ）           | $[b,s,h] \times [b,h,s]$ | $[b,s,s]$ | $2bs^2h$                    |
| $ [b,s,s] \to [b,s,h] $（attention × V） | $[b,s,s] \times [b,s,h]$ | $[b,s,h]$ | $2bs^2h$                    |
| $ [b,s,h] \to [b,s,h] $（投影回 hidden） | $[b,s,h] \times [h,h]$   | $[b,s,h]$ | $2bsh^2$                    |

> 🔹 总计 Attention FLOPs：
> $$ 6bsh^2 + 2bs^2h + 2bs^2h + 2bsh^2 = 8bsh^2 + 4bs^2h $$

------

#### 2. MLP 模块 FLOPs

##### 结构：

- 升维：$ h \to 4h $
- GeLU 激活
- 降维：$4h \to h$

##### FLOPs 计算：

| 操作                     | 输入输出                 | FLOPs                           |
| ------------------------ | ------------------------ | ------------------------------- |
| $ [b,s,h] \to [b,s,4h] $ | $[b,s,h] \times [h,4h]$  | $2bs h \cdot 4h = 8bsh^2$       |
| $ [b,s,4h] \to [b,s,h] $ | $[b,s,4h] \times [4h,h]$ | $2bs \cdot 4h \cdot h = 8bsh^2$ |

> 🔹 总计 MLP FLOPs：
> $$ 8bsh^2 + 8bsh^2 = 16bsh^2 $$

### 2.3 单层 Transformer 总 FLOPs

忽略轻微的逐元素操作（如 LayerNorm, Softmax, GeLU），单层前向传播总 FLOPs 为：

$$
\text{FLOPs}_{\text{forward\_layer}} = 24bsh^2 + 4bs^2h
$$



### 2.4 Vocabulary Embedding FLOPs

#### 操作：

- 将 token ID 映射为 embedding 向量
- 形状变换：$[b,s] \to [b,s,h] \to [b,s,V]$

#### FLOPs：

$$ [b,s,h] \times [h,V] \Rightarrow 2bshV $$

> 🔹 注意：此步骤通常只在前向传播中执行一次，但在反向传播中也有类似计算。



### 2.5 模型总 FLOPs

假设GPT模型有 $L$ 层个Transformer层，则：

| 组件                | FLOPs 公式                             |
| ------------------- | -------------------------------------- |
| **Self-Attention**  | $8bsh^2 + 4bs^2h$                      |
| **MLP**             | $16bsh^2$                              |
| **Embedding**       | $2bshV$                                |
| **单个Transformer** | $24bsh^2 + 4bs^2h$                     |
| **整模型总 FLOPs**  | $L \times (24bsh^2 + 4bs^2hV)  + 2bsh$ |



## 3. 反向传播与全流程计算

### 3.1 梯度计算原理

在反向传播（Backward）中，对于每一个线性变换 $y = wx$，需要计算：

- 对权重的梯度: $\frac{\partial L}{\partial w} = \frac{\partial L}{\partial y} \cdot x$ → 一次矩阵乘法
- 对输入的梯度: $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot w$ → 一次矩阵乘法

**结论**：在训练中，反向传播的计算量约为前向传播的 **2 倍**。

### 3.2 训练总 FLOPs

包含前向和反向传播的单步总计算量约为：



$$
\text{Total FLOPs} \approx 3 \times \text{Forward FLOPs}
$$

(注：若考虑重计算/激活值检查点 Recompute，倍数会增至约 4 倍)。

### 3.3 经验简化公式

对于大模型，当 $h \gg s$ 时，公式可近似为：



$$
\text{Total FLOPs} \approx 72bsh^2L \left(1 + \frac{s}{6h} + \frac{V}{12hL}\right)
$$

## 4. Llama 系列模型的 MFU 特化计算

Llama 模型在标准 Transformer 基础上引入了 **GQA (Grouped Query Attention)** 和 **SwiGLU** 激活函数，其计算分布略有不同。

### 4.1 模型结构简要说明

- 使用 **Llama2 架构**：包含多层 Transformer 解码器
- 关键组件：
  - **Attention 模块**：含 RoPE、QKV 投影、Softmax、输出投影
  - **FeedForward 模块**：含两个线性层与 Silu 激活函数
  - **RMSNorm**：归一化操作
  - **lm_head**：最终输出层

> 📌 注意：`wq`, `wk`, `wv` 分别为 Q/K/V 的权重矩阵；`wo` 为输出投影；`w1`, `w2`, `w3` 为 FFN 中的权重。

#### 4.1.1 Llama 架构差异点

- **GQA 影响**：$K, V$ 的投影维度降为 $h/r$（$r$ 为 head 比例系数）。
- **SwiGLU 影响**：FFN 层由三个线性矩阵 $W_1, W_2, W_3$ 组成，中间维度为 $\hat{h}$（通常为 $\frac{8}{3}h$）。

#### 4.2.1 符号定义

| 符号        | 含义                                         |
| ----------- | -------------------------------------------- |
| $ b $       | batch_size                                   |
| $ L $       | num_layers（层数）                           |
| $ s $       | seq_length（序列长度）                       |
| $ h $       | hidden_size（隐藏维度）                      |
| $ n $       | num_heads（注意力头数）                      |
| $ d $       | head_dim（每个头的维度）                     |
| $ \hat{h} $ | intermediate_size（FFN 中间维度）            |
| $ v $       | vocab_size（词表大小）                       |
| $ r $       | repeat（重复因子，通常为 $ n / n_k,v $）     |
| $ m $       | ffn_dim_multiplier（FFN 扩展倍数，一般为 4） |

> 🔹 关系式：
>
> - $ h = n \times d $
> - $ \hat{h} \approx \frac{8}{3} h \times m $


### 4.2 各模块 FLOPs 详细分析

#### 1. Attention 模块（×L 层）

| 操作                                       | 输入 → 输出                  | FLOPs      |
| ------------------------------------------ | ---------------------------- | ---------- |
| $ W_q: (b,s,h) \to (b,s,h) $               | $[b,s,h] \times [h,h]$       | $2bsh^2$   |
| $ W_k: (b,s,h) \to (b,s,h/r) $             | $[b,s,h] \times [h,h/r]$     | $2bsh^2/r$ |
| $ W_v: (b,s,h) \to (b,s,h/r) $             | $[b,s,h] \times [h,h/r]$     | $2bsh^2/r$ |
| $ QK^T: (b,n,s,d) \to (b,n,s,s) $          | $[b,n,s,d] \times [b,n,d,s]$ | $2bs^2h$   |
| $ score \cdot V: (b,n,s,s) \to (b,n,s,d) $ | $[b,n,s,s] \times [b,n,d,s]$ | $2bs^2h$   |
| $ W_o: (b,s,h) \to (b,s,h) $               | $[b,s,h] \times [h,h]$       | $2bsh^2$   |

> ✅ 总计 Attention FLOPs（每层）： $$ 2bsh^2 + \frac{2bsh^2}{r} + \frac{2bsh^2}{r} + 2bs^2h + 2bs^2h + 2bsh^2 = 4bsh^2 + \frac{4bsh^2}{r} + 4bs^2h $$

#### 2. FeedForward 模块（×L 层）

| 操作                                     | 输入 → 输出                              | FLOPs         |
| ---------------------------------------- | ---------------------------------------- | ------------- |
| $ W_3: (b,s,h) \to (b,s,\hat{h}) $       | $[b,s,h] \times [h,\hat{h}]$             | $2bsh\hat{h}$ |
| $ W_1: (b,s,\hat{h}) \to (b,s,\hat{h}) $ | $[b,s,\hat{h}] \times [\hat{h},\hat{h}]$ | $2bsh\hat{h}$ |
| $ W_2: (b,s,\hat{h}) \to (b,s,h) $       | $[b,s,\hat{h}] \times [\hat{h},h]$       | $2bsh\hat{h}$ |

> ✅ 总计 FFN FLOPs（每层）： $$ 6bsh\hat{h} $$

#### 3. lm_head 模块

| 操作                              | 输入 → 输出            | FLOPs   |
| --------------------------------- | ---------------------- | ------- |
| $ W_{head}: (b,s,h) \to (b,s,v) $ | $[b,s,h] \times [h,v]$ | $2bshv$ |

> ✅ 注意：该操作仅在最后一步执行，但对总 FLOPs 贡献显著。

### **4.3 总 FLOPs 推导**

#### 4.3.1 前向传播总 FLOPs（单次）

$$ \text{FLOPs} = L \times \left[ 4bsh^2 + \frac{4bsh^2}{r} + 4bs^2h + 6bsh\hat{h} + 2bshv \right] $$

代入 $ \hat{h} \approx \frac{8}{3}hm $，并简化：

$$ = 12Lbsh^2 \left(1 + \frac{1}{r} + \frac{3\hat{h}}{2h} + \frac{s}{h} + \frac{v}{2Lh}\right) $$

进一步近似：

- $ \frac{3\hat{h}}{2h} \approx 4m $
- $ \frac{1}{r} \approx \frac{n}{n_k,v} $，常忽略或合并

> ✅ 最终简化公式： $$ \boxed{ \text{FLOPs} \approx 12Lbsh^2 \left(1 + \frac{1}{r} + 4m + \frac{s}{h} + \frac{v}{2Lh}\right) } $$

------

#### 4.3.2 反向传播 FLOPs

- 反向传播 FLOPs ≈ **前向的 2 倍**
- 因此，**总训练 FLOPs ≈ 3 × 前向 FLOPs**

> 💡 提示：MFU 计算时需使用总 FLOPs（前向 + 反向 + 优化器），但此处以前向为主进行估算。


### 4.4 实例分析：Llama2-70B

给定参数：

- $ s = 4096 $
- $ h = 8192 $
- $ L = 80 $
- $ r = 8 $
- $ \hat{h} = 28672 $
- $ v = 32000 $

代入公式：

$$ \text{FLOPs} \approx 12 \times 80 \times b \times 8192^2 \left(1 + \frac{1}{8} + 4 \times 4 + \frac{4096}{8192} + \frac{32000}{2 \times 80 \times 8192}\right) $$

> 🔸 近似可得： $$ \text{FLOPs} \approx 6bs \times (\text{总参数量} + 2Lsh) $$

对于 70B 模型，由于参数量极大，其 $6bs \times \text{Params}$ 的估算方法非常接近精确值。在计算 MFU 时，需精确代入 $h=8192$ 和 $\hat{h}=28672$ 等超参数，以获得准确的算力需求分析。

## 5. 总结与优化建议

| **优化方向** | **操作手段**        | **对 MFU 的影响**                |
| ------------ | ------------------- | -------------------------------- |
| **计算密度** | 增大 Batch Size     | 显著提升，减少算子调度和通信开销 |
| **通信优化** | 调整 TP/PP 比例     | 降低通信延迟，减少算力闲置       |
| **算子融合** | 使用 FlashAttention | 降低显存带宽压力，提高计算利用率 |
| **精度转换** | 使用 BF16/FP8       | 提升单位时间内的理论吞吐量       |

> **专家目标**：在千卡规模的集群训练中，将 MFU 稳定在 **55% - 70%** 是大模型训练达到工业级性能的标志。
