# Swap Optimizer

## 问题分析

在大模型训练中，通常会通过 BF16 格式进行前反向的计算，在梯度更新的时候使用 FP32 的格式，
因此导致优化器中需要保存一份 FP32 的权重以及两个 FP32 的动量，显存占用为 `参数量 * 12` Bytes。
这部分显存在前反向阶段并不会被使用，且会推高显存峰值，导致模型训练 OOM。
虽然可以通过分布式优化器等特性来减少这部分的显存占用，但无法完全消除，且减少比例过于依赖 DP 数。

## 解决思路

本特性通过在前反向期间，卸载优化器状态到 host 侧内存，device侧仅保留逻辑视图，
在 step 更新阶段再加载回 device 侧，来降低显存峰值。

## 解决方案

1. 在优化器初始化 `shard_fp32_from_float16_groups` 的时候，会从模型权重（bf16）复制权重到优化器权重（fp32），
为了不冲击显存峰值，需要每复制一份权重就将权重 swap 到 host 侧。权重加载的时候同理，每次加载一份权重就进行 swap 操作，
由于只在初始化阶段，因此对性能影响可忽略。
2. 在 step 阶段，为了 h2d 和 d2h 的并行，会先一次性下发大约 `numel(shard_fp32_from_float16_groups) // swap_optimizer_times`
大小参数的 h2d 操作，再做 adamw 计算以及 copy 到模型权重（bf16），最后再 d2h 释放显存。
3. 由于 d2h 与 h2d 是异步拷贝，为了保证时序正确，第二轮的 d2h 需要等前一轮的 h2d 操作结束之后再下发第二轮。

![img.png](https://gitee.com/ascend/MindSpeed/raw/master/sources/images/swap-optimizer.png)

## 使用场景

使用了分布式优化器`--use-distributed-optimizer`且`--optimizer-selection`为`fused_adamw`的模型训练场景。

## 使用方法

`--swap-optimizer`： 开启 swap optimizer 特性。

`--swap-optimizer-times`： 默认值为16，用于设置 step 更新阶段进行 swap 的次数，越大并行的越多，可减少性能劣化，但会提高显存峰值。

## 注意事项

1. 本特性仅适用于开启分布式优化器`--use-distributed-optimizer`且`--optimizer-selection`为`fused_adamw`的模型训练场景
2. 本特性与 `--reuse-fp32-param`、fused ema adamw优化器等其他优化器相关特性暂不兼容。
