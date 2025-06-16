【并行策略/加速算法/显存优化/融合算子】

|            |                                                                             场景                                                                              |
| :--------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  SPTD并行  |                                                   [张量并行](aigc/ascend/parallel_features/tensor-parallel.md)                                                    |
|  SPTD并行  |                                                 [流水线并行](aigc/ascend/parallel_features/pipeline-parallel.md)                                                  |
|  SPTD并行  |                                            [虚拟流水并行](aigc/ascend/parallel_features/virtual-pipeline-parallel.md)                                             |
|  SPTD并行  |                                                  [序列并行](aigc/ascend/parallel_features/sequence-parallel.md)                                                   |
| 长序列并行 |                               [Ascend Ring Attention 长序列并行](aigc/ascend/parallel_features/ring-attention-context-parallel.md)                                |
| 长序列并行 |                                          [Ulysses 长序列并行](aigc/ascend/parallel_features/ulysses-context-parallel.md)                                          |
| 长序列并行 |                                            [混合长序列并行](aigc/ascend/parallel_features/hybrid-context-parallel.md)                                             |
|    MOE     | [MOE 专家并行](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2FNVIDIA%2FMegatron-LM%2Fblob%2Fmain%2Fmegatron%2Fcore%2Ftransformer%2Fmoe%2FREADME.md) |
|    MOE     |                                [MOE 重排通信优化](aigc/ascend/parallel_features/megatron_moe/megatron-moe-allgather-dispatcher.md)                                |
|  显存优化  |                                                 [参数副本复用](aigc/ascend/parallel_features/reuse-fp32-param.md)                                                 |
|  显存优化  |                                              [分布式优化器](aigc/ascend/parallel_features/distributed-optimizer.md)                                               |
|  显存优化  |                                                 [Swap Attention](aigc/ascend/parallel_features/swap_attention.md)                                                 |
|  显存优化  |                                                   [重计算](aigc/ascend/parallel_features/recompute_relative.md)                                                   |
|  融合算子  |                                                [Flash attention](aigc/ascend/parallel_features/flash-attention.md)                                                |
|  融合算子  |                                                    [Fused rmsnorm](aigc/ascend/parallel_features/rms_norm.md)                                                     |
|  融合算子  |                                                      [Fused swiglu](aigc/ascend/parallel_features/swiglu.md)                                                      |
|  融合算子  |                                       [Fused rotary position embedding](aigc/ascend/parallel_features/rotary-embedding.md)                                        |
|  融合算子  |                                               [GMM](aigc/ascend/parallel_features/megatron_moe/megatron-moe-gmm.md)                                               |
|  融合算子  |                                                   [Matmul Add](aigc/ascend/parallel_features/npu_matmul_add.md)                                                   |
|  通信掩盖  |                                           [梯度reduce通算掩盖](aigc/ascend/parallel_features/async-ddp-param-gather.md)                                           |
|  通信掩盖  |                                     [Recompute in advance](aigc/ascend/parallel_features/recompute_independent_pipelining.md)                                     |
|  通信掩盖  |                                         [权重all-gather通算掩盖](aigc/ascend/parallel_features/async-ddp-param-gather.md)                                         |
|  通信掩盖  |                                                            [MC2](aigc/ascend/parallel_features/mc2.md)                                                            |
