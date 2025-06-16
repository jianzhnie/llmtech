【并行策略/加速算法/显存优化/融合算子】

|            |                                                                             场景                                                                              |
| :--------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  SPTD并行  |                                                   [张量并行](aigc/llm_recipes/features/tensor-parallel.md)                                                    |
|  SPTD并行  |                                                 [流水线并行](aigc/llm_recipes/features/pipeline-parallel.md)                                                  |
|  SPTD并行  |                                            [虚拟流水并行](aigc/llm_recipes/features/virtual-pipeline-parallel.md)                                             |
|  SPTD并行  |                                                  [序列并行](aigc/llm_recipes/features/sequence-parallel.md)                                                   |
| 长序列并行 |                               [Ascend Ring Attention 长序列并行](aigc/llm_recipes/features/ring-attention-context-parallel.md)                                |
| 长序列并行 |                                          [Ulysses 长序列并行](aigc/llm_recipes/features/ulysses-context-parallel.md)                                          |
| 长序列并行 |                                            [混合长序列并行](aigc/llm_recipes/features/hybrid-context-parallel.md)                                             |
|    MOE     | [MOE 专家并行](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2FNVIDIA%2FMegatron-LM%2Fblob%2Fmain%2Fmegatron%2Fcore%2Ftransformer%2Fmoe%2FREADME.md) |
|    MOE     |                                [MOE 重排通信优化](aigc/llm_recipes/features/megatron_moe/megatron-moe-allgather-dispatcher.md)                                |
|  显存优化  |                                                 [参数副本复用](aigc/llm_recipes/features/reuse-fp32-param.md)                                                 |
|  显存优化  |                                              [分布式优化器](aigc/llm_recipes/features/distributed-optimizer.md)                                               |
|  显存优化  |                                                 [Swap Attention](aigc/llm_recipes/features/swap_attention.md)                                                 |
|  显存优化  |                                                   [重计算](aigc/llm_recipes/features/recompute_relative.md)                                                   |
|  融合算子  |                                                [Flash attention](aigc/llm_recipes/features/flash-attention.md)                                                |
|  融合算子  |                                                    [Fused rmsnorm](aigc/llm_recipes/features/rms_norm.md)                                                     |
|  融合算子  |                                                      [Fused swiglu](aigc/llm_recipes/features/swiglu.md)                                                      |
|  融合算子  |                                       [Fused rotary position embedding](aigc/llm_recipes/features/rotary-embedding.md)                                        |
|  融合算子  |                                               [GMM](aigc/llm_recipes/features/megatron_moe/megatron-moe-gmm.md)                                               |
|  融合算子  |                                                   [Matmul Add](aigc/llm_recipes/features/npu_matmul_add.md)                                                   |
|  通信掩盖  |                                           [梯度reduce通算掩盖](aigc/llm_recipes/features/async-ddp-param-gather.md)                                           |
|  通信掩盖  |                                     [Recompute in advance](aigc/llm_recipes/features/recompute_independent_pipelining.md)                                     |
|  通信掩盖  |                                         [权重all-gather通算掩盖](aigc/llm_recipes/features/async-ddp-param-gather.md)                                         |
|  通信掩盖  |                                                            [MC2](aigc/llm_recipes/features/mc2.md)                                                            |
