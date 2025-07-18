# 序列并行

## 问题分析

张量模型并行可以降低显存占用，加快训练速度，但是它要求将模型各层划分为独立的、可管理的块，所以不适用于 LayerNorm 和 Dropout 等操作。虽然 LayerNorm 和 Dropout 等操作的计算成本很低，但它们确实需要大量冗余内存来存储激活。为了分摊张量并行中无法切分的显存和计算，引入了序列并行的方法。

## 解决方案

在张量模型并行的基础上，进一步对 LayerNorm 和 Dropout 模块的序列维度进行切分。

### 解决思路:

将 LayerNorm 以及 Dropout 等操作的输入按序列维度进行了切分，使得各个设备上面只需要做一部分的 Dropout 和 LayerNorm 等操作即可。

为了方便理解，以下图为例：假设输入$X$的大小为$ s \times b \times h $，按照序列维度切分$X=[X_1^s,X_2^s]$，经过LayerNorm操作后的结果为$Y=[Y_1^s,Y_2^s]$，随后进行张量模型并行。

![image.png](https://gitee.com/ascend/MindSpeed/raw/master/sources/images/sequence-parallel.png)

[原文链接](https://arxiv.org/pdf/2205.05198)

## 使用场景

使用训练模型时，将模型加载到多卡，使用张量模型并行后显存依旧占用过高或超出了处理器显存限制，或者训练时间过长，可以开启序列并行来降低显存占用，加快训练速度。

## 使用方法

首先确保训练参数中加入`--tensor-model-parallel-size N`，设置张量模型并行。

同时添加`--sequence-parallel`，开启序列并行。

## 使用效果

利用多个设备，降低显存开销，加快训练速度。
