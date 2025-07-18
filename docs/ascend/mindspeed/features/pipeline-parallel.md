# 流水线并行

## 问题分析

在大模型时代，单一设备无法存储整个模型。模型并行可以在训练过程中将模型加载到多个设备上。在朴素的模型并行中，设备需要等待前一阶段的计算结果，导致计算资源的严重利用率不足。同时，设备需要储存计算的中间结果，存储开销大。

## 解决方案

采用流水线的思想，减少不同机器之间等待的时间。同时尽可能地缩短前向计算与反向计算之间的距离，以减少内存消耗

### 解决思路:

* 将整个网络分阶段（stage），不同阶段在不同的设备上，前后阶段流水线分批工作，通过一种“接力”的方式并行。
* 开始训练时，会先进行预热。预热完成后，每进行一个前向运算，就安排一个后向运算。最后进行冷却，完成剩余阶段。如下图所示

![alt text](https://gitee.com/ascend/MindSpeed/raw/master/sources/images/pipedream1F1B.png)


[原文链接](https://arxiv.org/pdf/1806.03377)
## 使用场景

在训练模型时，为了降低单个设备的存储开销，提升计算效率，将模型加载到多卡来进行流水线并行。

## 使用方法

设置`--pipeline_model_parallel_size`，默认为1，根据用户需求配置。

## 使用效果

提升计算效率，减少内存消耗
