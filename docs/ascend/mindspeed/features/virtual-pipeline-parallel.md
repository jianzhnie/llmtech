# 虚拟流水线并行

## 问题分析

Pipedream流水线并行切分粒度过大，运行过程中仍然有许多空泡(bubble)，计算资源利用率仍有提高空间。

## 解决方案

将计算进一步细分，减少空泡。

### 解决思路:

在设备数量不变的情况下，分出更多的流水线阶段，以更多的通信量，换取空泡比率降低。

![alt text](https://gitee.com/ascend/MindSpeed/raw/master/sources/images/virtual-pipeline.PNG)

[原文链接](https://people.eecs.berkeley.edu/~matei/papers/2021/sc_megatron_lm.pdf)

为了方便理解，举一个例子：假设模型层数为16，张量并行大小为1，流水线并行大小为4，虚拟流水线并行大小为2。模型会被分为 4 * 2 = 8 个阶段，每个阶段 16 / 8 = 2 个层。

    Device 0: [1, 2] [9, 10]
    Device 1: [3, 4] [11, 12]
    Device 2: [5, 6] [13, 14]
    Device 3: [7, 8] [15, 16]

前向的顺序为 Device 0 -> Device 1 -> Device 2 -> Device 3 -> Device 0 -> Device 1 -> Device 2 -> Device 3

## 使用场景

鉴于当前数据处理与模型训练过程中存在的性能瓶颈，特别是针对空泡比率（即无效或低效率计算周期占比）的优化需求，虚拟流水线并行技术展现出其独特的优势。该技术旨在通过创新的并行处理机制，有效减少空泡比率，显著提升模型训练的整体性能与效率。具体而言，它能够优化资源分配，加速数据处理流程，从而缩短训练周期，降低计算成本。

## 使用方法

设置`--num-layers-per-virtual-pipeline-stage N`。表示每个阶段的层数。要求模型的总层数 L % N == 0。要求 `--pipeline-model-parallel-size` >= 2。

## 使用效果

通过虚拟流水线并行策略，在流水线并行的基础上，成功降低了空泡比率，进一步提升了模型训练的性能和资源利用率。

## 注意事项

Megatron虚拟流水并行vpp影响权重切分方式，保存、加载权重时需保证vpp配置一致，才能正常加载；
