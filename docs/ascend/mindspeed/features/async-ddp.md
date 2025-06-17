# 异步DDP

## 问题分析

大模型训练过程中，通常会使用数据并行。在进行梯度更新时，数据并行组中的通信（未开启分布式优化器时为AllReduce，开启时为ReduceScatter）要等反向计算完成后再进行。这样的串行执行顺序会造成计算和通信流存在一定的空闲等待时间，导致执行效率较低。

## 解决方案

通过将计算和通信任务分别拆分成更细粒度的子任务来实现相互的流水掩盖。并行原理如下图所示：
<p align="center"> <img src="https://gitee.com/ascend/MindSpeed/raw/master/sources/images/async_ddp.png" height="250px" width="680px"></p>

### 解决思路:

设置一个Bucket，存储反向计算的结果。每当Bucket存满时立刻执行桶中结果的通信任务，后续反向计算可以和这部分通信并行执行，从而增大计算和通信流的利用率，提高执行效率。

## 使用场景

使用该特性的前提是模型开启数据并行和虚拟流水并行，脚本中设置了`--num-layers-per-virtual-pipeline-stage N`。

## 使用方法

设置`--overlap-grad-reduce`即可调用该算法。

## 使用效果

开启该特性可以提升性能。
