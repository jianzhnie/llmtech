# Ascend MC2

## 使用前提
仅限于版本标识为`CANN 8.0.RC2`和`Ascend HDK 24.1.RC2`及其后续所有迭代版本的系统环境。

若在非指定版本中尝试采用该配置，可能触发系统级的异常行为，包括但不限于运行时错误。

## 问题分析

在开启了TP和SP的大模型训练场景下，存在matmul计算和all-reduce操作的强依赖关系（不开启SP），或存在matmul计算和all_gather/reduce_scatter操作的强依赖关系（开启SP）。当模型参数量较大时，此处通信量和计算量都较大，在串行执行时，会引入较长的等待闲置时间。

## 解决方案

针对存在强依赖关系的matmul计算和通信操作，昇腾自研MC2解决方案。

MC2通过融合算子的方式将matmul计算和集合通信操作进行融合，将较大的计算和通信任务切分成了较小的计算子任务和通信子任务，并通过流水的方式使得通信子任务和计算子任务可以互相掩盖，从而减少等待和闲置时间，提高利用率。

### 解决思路:

MindSpeed在python脚本侧将原本串行的matmul和all_gather/reduce_scatter操作通过MC2融合算子接口进行了融合。

具体实现参见[link](../../mindspeed/core/tensor_parallel/ascend_turbo/mc2_linears_seq_parallel.py)。

MC2算子接口参见[link](https://www.hiascend.com/document/detail/zh/Pytorch/60RC1/apiref/apilist/ptaoplist_000449.html)。

## 使用场景

当开启了TP和SP时，建议开启MC2进一步优化。

## 使用方法

设置--use-ascend-mc2即可使能MC2算子。


**同时需要确保开启**`--sequence-parallel`。

## 使用效果

在开启TP和SP的训练场景下，使用MC2可以减少内存开销并提高计算效率。

## 注意事项

mcore场景下（开启`--use-mcore-models`），MoE模型，MLP部分不使能MC2。
