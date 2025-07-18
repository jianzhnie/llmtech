# Paradigms of Parallelism 并行范式

## Introduction

随着深度学习的发展，对并行训练的需求日益增长。这是因为模型和数据集变得越来越大，如果坚持使用单 GPU 训练，训练时间将变得难以忍受。本节我们将简要概述现有的并行训练方法。

## Data Parallel

数据并行因其简单性成为最常见的并行形式。在数据并行训练中，数据集被分割为若干分片，每个分片分配给一个设备。这相当于沿批次维度并行化训练过程。每个设备持有完整的模型副本，并在分配的数据分片上进行训练。反向传播后，模型梯度将进行全归约操作，确保不同设备上的模型参数保持同步。

<div align="center">
  <img src="https://s2.loli.net/2022/01/28/WSAensMqjwHdOlR.png" alt="img" style="zoom:33%;" />
</div>

>  数据并行示意图

数据并行通过并行处理数据显著减少训练时间，且可扩展性取决于可用 GPU 数量。但同步各 GPU 计算结果可能带来额外开销。

## Model Parallel

数据并行训练的一个显著特点是每个 GPU 都持有整个模型权重的副本，这带来了冗余问题。另一种并行范式是模型并行，即将模型拆分并分布到设备阵列上。主要存在两种并行类型：张量并行和流水线并行。张量并行是在矩阵乘法等运算内部实现并行计算，流水线并行则是在层间实现并行计算，因此从另一视角看，张量并行可视为层内并行，流水线并行可视为层间并行。

模型并行是训练超出单个 GPU 内存容量的大型模型的有效策略。但由于同一时间仅有一个 GPU 处于活跃状态，会导致 GPU 利用率不均衡。GPU 间的结果传递还会引入通信开销，可能成为性能瓶颈。

### Tensor Parallel

张量并行训练是将张量沿特定维度分割为 `N` 个分块，每个设备仅持有整个张量的 `1/N` 部分，同时不影响计算图的正确性。这需要额外的通信来确保结果正确。

以通用矩阵乘法 C=AB 为例，假设我们将 B 沿列维度分割为 `[B0 B1 B2 ... Bn]`，每个设备持有一列。然后在各设备上将 `A` 与 `B` 的对应列相乘，得到 `[AB0 AB1 AB2 ... ABn]`。此时每个设备仍持有部分结果，例如设备 0 持有 `AB0`。为确保结果正确，我们需要全收集这些部分结果并沿列维度拼接张量。通过这种方式，我们能在设备间分布张量的同时保证计算流程的正确性。

<div align="center">
  <img src="https://s2.loli.net/2022/01/28/2ZwyPDvXANW4tMG.png" alt="img" style="zoom:33%;" />
</div>

> 张量并行示意图



### Pipeline Parallel

<div align="center">
  <img src="https://s2.loli.net/2022/01/28/at3eDv7kKBusxbd.png" alt="img" style="zoom:33%;" />
</div>



> Pipeline parallel illustration
> 流水线并行示意图

流水线并行的核心思想是将模型按层分割为若干块，每块分配给不同设备。在前向传播过程中，每个设备将中间激活值传递给下一阶段；在反向传播时，各设备将输入张量的梯度回传给前一流水线阶段。这种机制使得设备能够并行计算，从而提升训练吞吐量。流水线并行训练的缺点在于会产生气泡时间——部分设备处于计算等待状态，导致计算资源浪费。

<div align="center">
  <img src="https://s2.loli.net/2022/01/28/sDNq51PS3Gxbw7F.png" alt="img" />
</div>



>  Source: [GPipe](https://arxiv.org/abs/1811.06965)

## Sequence Parallelism

序列并行是一种沿序列维度进行划分的并行策略，使其成为训练长文本序列的有效方法。成熟的序列并行方法包括 Megatron 序列并行、DeepSpeed-Ulysses 序列并行以及Ring-attention序列并行。

### Megatron SP

该序列并行方法构建于张量并行基础之上。在模型并行的每个 GPU 上，样本数据保持独立且被复制。对于无法应用张量并行的部分（如 LayerNorm 等非线性运算），可沿序列维度将样本数据分割为多个片段，由各 GPU 分别计算部分数据。而对于注意力机制和多层感知机等需要聚合激活值的线性运算部分，则采用张量并行处理。这种方案在模型分区时能进一步降低激活值内存占用。需特别注意，此序列并行方法必须与张量并行配合使用。

### DeepSpeed-Ulysses: DeepSpeed-Ulysses

在序列并行中，样本沿序列维度进行分割，并采用全交换通信操作（all-to-all），使每个 GPU 能获取完整序列但仅计算注意力头的非重叠子集，从而实现序列并行。该并行方法支持完全通用的注意力机制，可同时处理稠密与稀疏注意力。全交换操作是一种完整的数据交换操作，类似于分布式转置运算。在注意力计算前，样本沿序列维度分割，因此每个设备仅持有 N/P 长度的序列片段。但经过全交换操作后，qkv 子部分的形状转变为[N, d/p]，确保注意力计算时能考量整体序列信息。

### Ring Attention


环形注意力（Ring Attention）在概念上与Flash Attention（Flash Attention）类似。每个 GPU 仅计算局部注意力，最终通过归约运算汇总各注意力块以计算全局注意力。该算法沿序列维度将输入切分为多个分块，由不同 GPU 或处理器分别处理。其核心采用"环形通信"策略：通过点对点通信在 GPU 间传递键值子块进行迭代计算，从而支持超长文本的多 GPU 训练。该策略中，各处理器仅与前驱和后继节点交换信息，形成环形网络拓扑。这种设计无需全局同步即可高效传递中间计算结果，显著降低了通信开销。

## Optimizer-Level Parallel

另一种范式在优化器层面发挥作用，当前该范式最著名的方法是 ZeRO（即[零冗余优化器 ](https://arxiv.org/abs/1910.02054)）。ZeRO 通过将参数、梯度和优化器状态分区到不同数据并行进程，显著提升内存使用效率。包含三个优化阶段：

- 阶段1对优化器状态进行分区

- 阶段2对优化器及梯度状态进行分区
  -
  阶段3对优化器、梯度和参数全部分区

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/parallelism-zero.png" alt="img" />
</div>

## Parallelism on Heterogeneous System

上述方法通常需要大量 GPU 来训练大型模型。然而人们往往忽视了 CPU 内存容量远大于 GPU 这一事实——在典型服务器上，CPU 可轻松配备数百 GB 内存，而单个 GPU 通常仅有 16 或 32GB 内存。这一现状促使业界开始思考：为何不利用 CPU 内存进行分布式训练？

近期进展依赖于 CPU 甚至 NVMe 磁盘来训练大型模型。核心思路是在张量未被使用时将其卸载回 CPU 内存或 NVMe 磁盘。通过采用异构系统架构，可以在单台机器上容纳超大规模模型。

<div align="center">
  <img src="https://s2.loli.net/2022/01/28/qLHD5lk97hXQdbv.png" alt="img" />
</div>

> Heterogenous system illustration
> 异构系统示意图

### Hybrid parallelism 混合并行

可以组合多种并行方法来实现更大的内存节省，并更高效地训练具有数十亿参数的模型。
