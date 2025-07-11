## Ascend 确定性计算
## 问题分析

在训练过程中，各类随机因素会使得每次实验的训练过程并不完全一致，从而影响其LOSS曲线、性能曲线等无法完全重合。
然而，在重复实验与对比实验中有时需要确定性的计算结果，保证结果的可复现性。

## 解决方案

为满足上述需求，引入了“确定性计算”功能，允许用户通过昇腾(Ascend)芯片确保多次训练结果的一致性，从而帮助性能调优、对照实验等工作。

## 使用场景

需要进行性能对比、特定场景复现时。

## 使用方法

要启用此功能，在脚本中加入`--npu-deterministic`即可。

## 使用效果

通过确定性计算功能，可保证同参数下多次实验具有相同的实验结果。
