# MindSpeed-LLM 参数指南



### 1. 基础训练参数 (Training)

| 参数名                              | 中文含义                    | 默认值        | 说明                                                         |
| ----------------------------------- | --------------------------- | ------------- | ------------------------------------------------------------ |
| transformer-impl                    | Transformer实现方式         | 'local'       | 可选: 'local', 'transformer_engine'                          |
| enable-recompute-layers-per-pp-rank | 启用每个PP rank的层重计算   | False         | 如果启用，recompute-num-layers将表示每个pp rank中重计算的层数 |
| pre-tockens                         | Flash attention预tokens数   | 65536         | 用于Flash attention                                          |
| next-tockens                        | Flash attention下一tokens数 | 0             | 用于Flash attention                                          |
| sparse-mode                         | Flash attention mask模式    | 0             | 不同的flash attention mask模式                               |
| shape-order                         | 输入shape顺序               | 'SBH'         | 可选: 'SBH', 'BSH', 'BSND', 'BNSD'                           |
| use-deter-comp                      | 启用确定性计算              | False         | 为NPU启用确定性计算                                          |
| jit-compile                         | JIT编译模式                 | False         | 设置JIT编译模式                                              |
| prompt-type                         | 提示模板类型                | None          | 用于训练/推理的提示模板类型                                  |
| prompt-type-path                    | 提示模板路径                | TEMPLATES_DIR | 模板json文件的路径                                           |
| pad-to-multiple-of                  | 填充倍数                    | 8             | 用于微调时的填充倍数                                         |
| scale-emb                           | 嵌入token缩放               | None          | 嵌入token的缩放因子                                          |
| dim-model-base                      | 模型基础维度                | None          | 模型基础维度                                                 |
| no-cut-token                        | 不切割token                 | False         | 用于微调时不切割token                                        |
| scale-depth                         | 深度缩放                    | None          | 深度缩放因子                                                 |
| swap-attention                      | 启用swap-attention          | False         | 开启swap-attention特性                                       |
| swap-modules                        | 交换模块                    | None          | 模型的交换模块，需与swap-attention一起使用                   |
| load-checkpoint-loosely             | 宽松加载检查点              | False         | 启用非严格加载检查点                                         |
| no-post-layer-norm                  | 禁用最终层归一化            | False         | 禁用最终层归一化                                             |
| return-document-ids                 | 返回文档ID                  | False         | 获取batch时返回文档ID                                        |
| reset-attention-mask                | 重置注意力掩码              | False         | 获取batch时返回文档ID                                        |
| swap-optimizer                      | 交换优化器到CPU             | False         | 将优化器交换到CPU                                            |
| swap-optimizer-times                | 优化器交换次数              | 16            | 每次交换将移动(len(shard_fp32_from_float16) // swap_optimizer_times)个元素 |

### 2. 分布式训练参数 (Distributed)

| 参数名                      | 中文含义             | 默认值 | 说明                        |
| --------------------------- | -------------------- | ------ | --------------------------- |
| local-rank                  | 本地rank             | None   | 分布式启动器传递的本地rank  |
| distributed-timeout-minutes | 分布式超时时间(分钟) | 45     | torch.distributed的超时时间 |

### 3. 算法相关参数 (Algorithm)

| 参数名                                   | 中文含义           | 默认值 | 说明                                                     |
| ---------------------------------------- | ------------------ | ------ | -------------------------------------------------------- |
| noop-layers                              | 空操作层           | None   | 指定空操作层                                             |
| reuse-fp32-param                         | 重用FP32参数       | False  | 分布式训练优化器释放FP32参数副本以节省内存               |
| recompute-activation-function            | 重计算激活函数     | False  | 在MLP层中重计算激活函数                                  |
| recompute-activation-function-num-layers | 重计算激活函数层数 | None   | 可与recompute-method block和recompute-num-layers一起使用 |
| recompute-in-advance                     | 提前重计算         | False  | 提前重计算以减少气泡并改善训练                           |
| recompute-norm                           | 重计算归一化       | False  | 在Transformer层中重计算归一化                            |
| recompute-norm-num-layers                | 重计算归一化层数   | None   | 可与激活函数重计算一起使用                               |
| o2-optimizer                             | 使用O2优化器       | False  | 使用bf16指数移动平均以大幅节省内存                       |
| o2-gradient                              | 使用O2梯度         | False  | 使用bf16梯度累积以大幅节省内存                           |
| share-kvstates                           | 共享KV状态         | False  | CLA共享kv状态                                            |

### 4. 网络架构参数 (Network)

| 参数名                    | 中文含义           | 默认值 | 说明                                                         |
| ------------------------- | ------------------ | ------ | ------------------------------------------------------------ |
| add-qkv-bias              | 添加QKV偏置        | False  | QKV偏置配置                                                  |
| add-dense-bias            | 添加Dense偏置      | False  | Dense偏置配置                                                |
| add-output-layer-bias     | 添加输出层偏置     | False  | 输出层偏置配置                                               |
| skip-bias-add             | 跳过偏置添加       | True   | 跳过偏置配置                                                 |
| add-rmsnorm-offset        | RMSNorm单元偏移    | False  | RMSNorm单元偏移                                              |
| geglu                     | 使用GEGLU激活函数  | False  | GEGLU激活函数                                                |
| input-embeds-norm         | 输入归一化         | False  | 输入归一化                                                   |
| gelu-tanh                 | Tanh GEGLU激活函数 | False  | Tanh GEGLU激活函数                                           |
| output-logit-softcapping  | 输出logit软上限    | None   | 输出logit软上限                                              |
| attn-logit-softcapping    | 注意力logit软上限  | None   | 注意力logit软上限                                            |
| query-pre-attn-scalar     | 注意力标量         | None   | 注意力标量                                                   |
| interleave-sliding-window | 交错滑动窗口大小   | None   | 使用交错滑动窗口注意力时的窗口大小                           |
| stage                     | 训练阶段           | None   | 可选: "sft", "dpo", "orm", "prm", "simpo", "ray_ppo", "ray_online_dpo", "ray_grpo", "trl_ppo" |
| cut-max-seqlen            | 切割最大序列长度   | False  | 确定训练模式                                                 |

### 5. 强化学习参数 (RL)

| 参数名                     | 中文含义                    | 默认值    | 说明                                                         |
| -------------------------- | --------------------------- | --------- | ------------------------------------------------------------ |
| dpo-beta                   | DPO损失beta参数             | 0.1       | DPO损失的beta参数                                            |
| simpo-beta                 | SimPO损失beta参数           | 2.5       | SimPO损失的beta参数                                          |
| gamma-beta-ratio           | SimPO损失gamma-beta比率     | 1.4       | SimPO损失的gamma-beta比率                                    |
| dpo-loss-type              | DPO损失类型                 | "sigmoid" | 可选: "sigmoid", "hinge", "ipo"                              |
| simpo-loss-type            | SimPO损失类型               | "sigmoid" | 可选: "sigmoid", "hinge", "ipo"                              |
| simpo-label-smoothing      | SimPO标签平滑               | 0.0       | 鲁棒SimPO标签平滑参数                                        |
| ref-model                  | 参考模型路径                | None      | 用于PPO或DPO训练的参考模型路径                               |
| refer-model-iter           | 参考模型迭代次数            | 1         | 用于PPO或DPO训练的参考模型迭代次数                           |
| dpo-label-smoothing        | DPO标签平滑                 | 0.0       | cDPO中的鲁棒DPO标签平滑参数，应在0到0.5之间                  |
| pref-ftx                   | DPO训练中的监督微调损失系数 | 0.0       | DPO训练中的监督微调损失系数                                  |
| is-pairwise-dataset        | 是否为成对数据集            | False     | 数据集是否为成对格式，包含选择序列和拒绝序列                 |
| placeholder-token          | 占位符token                 | 'ки'      | 标记每个步骤结束的特殊占位符token                            |
| reward-tokens              | 奖励token                   | []        | 表示整个推理过程中每个推理步骤正确性的标签                   |
| md5-validate               | 启用MD5验证                 | False     | 启用MD5验证                                                  |
| max-prompt-length          | PPO中最大提示长度           | 512       | PPO中的最大提示长度                                          |
| num-samples-per-step       | 生成时每步样本数            | 1         | 生成时每步的样本数                                           |
| rollout-batch-size         | actor rollout批次大小       | None      | actor rollout批次大小                                        |
| cliprange-value            | 裁剪范围值                  | 0.2       | 裁剪范围值                                                   |
| critic-mini-batch-size     | critic小批次大小            | 1         | critic小批次大小                                             |
| critic-update-epochs       | critic更新轮数              | 1         | critic更新轮数                                               |
| ppo-mini-batch-size        | PPO小批次大小               | 1         | PPO小批次大小                                                |
| clip-ratio                 | PPO损失裁剪比率             | 0.2       | PPO损失裁剪比率                                              |
| entropy-coeff              | 熵系数                      | 0.001     | 熵系数                                                       |
| ppo-epochs                 | PPO轮数                     | 1         | PPO轮数                                                      |
| shuffle-minibatch          | 启用小批次打乱              | False     | 在PPO中启用小批次打乱                                        |
| do-sample                  | 启用actor生成采样           | False     | 在actor生成中启用采样                                        |
| missing-eos-penalty        | EOS惩罚                     | 0.0       | EOS惩罚                                                      |
| n-samples-per-prompt       | GRPO中每个提示的样本数      | 1         | GRPO中每个提示的样本数                                       |
| reward-model               | 奖励模型路径                | False     | 用于PPO训练的参考模型路径                                    |
| verifier                   | 启用验证器                  | False     | 在计算分数时启用验证器                                       |
| kl-coef                    | PPO训练中的KL系数           | 0.3       | PPO训练中的KL系数                                            |
| gamma                      | PPO训练中的折扣因子         | 1.0       | PPO训练中的折扣因子                                          |
| lam                        | PPO训练中GAE的lambda值      | 0.95      | PPO训练中GAE的lambda值                                       |
| advantage-whiten           | GRPO中的优势白化            | True      | GRPO中的优势白化                                             |
| dataset-category           | 数据集类别                  | None      | 训练数据集类别的逗号分隔列表，0表示无准确答案的一般问题，1表示数学问题 |
| extract-content-for-reward | 为奖励提取内容              | False     | 为奖励模型判断提取答案标签中的内容                           |

### 6. 推理参数 (Inference)

| 参数名           | 中文含义                | 默认值 | 说明                    |
| ---------------- | ----------------------- | ------ | ----------------------- |
| task             | 任务ID                  | None   | 要运行的任务ID          |
| top-p            | Top-p采样               | 0.95   | Top-p采样参数           |
| top-k            | Top-k采样               | 50     | Top-k采样参数           |
| temperature      | 采样温度                | 0.7    | 采样温度参数            |
| max-length       | 文本总长度              | 256    | 生成文本的总长度        |
| max-new-tokens   | 新生成token数           | 128    | 生成文本的大小          |
| hf-chat-template | 使用Huggingface聊天模板 | False  | 使用Huggingface聊天模板 |
| add-eos-token    | 添加EOS token           | []     | 使用额外的EOS token     |
| use-kv-cache     | 使用KV缓存              | False  | 使用KV缓存加速推理      |
| history-turns    | 历史对话轮数            | 3      | 聊天历史轮数            |

### 7. MoE相关参数 (MoE)

| 参数名                                    | 中文含义                        | 默认值      | 说明                                                       |
| ----------------------------------------- | ------------------------------- | ----------- | ---------------------------------------------------------- |
| expert-interval                           | 专家间隔                        | 1           | 每隔多少层使用专家                                         |
| moe-train-capacity-factor                 | MoE专家训练容量因子             | 1.0         | MoE专家在训练时的容量因子                                  |
| use-fused-moe-token-permute-and-unpermute | 使用融合的moe token排列和反排列 | False       | 使用融合的moe排列和反排列                                  |
| gemm-gradient-accumulation-fusion         | 使用gemm中的梯度累积融合        | False       | 在gemm中使用梯度累积融合                                   |
| moe-token-dispatcher-type                 | MoE token分发器类型             | 'allgather' | 可选: 'allgather', 'alltoall'                              |
| noisy-gate-policy                         | 噪声门控策略                    | None        | 可选: 'Jitter', 'RSample', 'None'                          |
| enable-token-rearrange-opt                | 启用token重排优化               | False       | 启用token重排优化                                          |
| embedding-multiplier-scale                | 嵌入乘数缩放                    | 1.0         | 嵌入缩放因子                                               |
| input-jitter                              | 输入抖动                        | True        | 向输入张量添加噪声                                         |
| post-norm                                 | 后归一化                        | False       | 在注意力或MLP后进行归一化                                  |
| output-multiplier-scale                   | 输出乘数缩放                    | None        | 为logits输出添加缩放                                       |
| moe-permutation-async-comm                | 重叠moe排列3个all gather通信    | False       | 重叠moe排列3个all gather通信                               |
| shared-expert-gate                        | 共享专家门控                    | False       | moe模型具有共享专家门控                                    |
| shared-expert-gate-output-dimension       | 共享专家门控输出维度            | 1           | moe模型共享专家门控输出维度，仅可配置为1或hidden_state     |
| moe-alltoall-overlap-comm                 | moe alltoall重叠通信            | False       | moe alltoall重叠通信                                       |
| cla-share-factor                          | 跨层注意力共享因子              | 1           | 跨层注意力在cla-share-factor层之间共享kv                   |
| moe-tp-extend-ep                          | 使用tp组扩展专家并行            | False       | 使用tp组扩展专家并行而不是在tp组中分片专家权重张量         |
| moe-zero-memory                           | MoE零内存                       | 'disable'   | 在moe层中保存激活内存，可选: 'disable', 'level0', 'level1' |
| moe-zero-memory-num-layers                | MoE零内存层数                   | None        | 每个pp阶段中使用moe-zero-memory level1的层数               |
| moe-allgather-overlap-comm                | moe allgather重叠通信           | False       | moe allgather重叠通信                                      |

### 8. LoRA相关参数 (LoRA)

| 参数名                     | 中文含义          | 默认值                                 | 说明                                      |
| -------------------------- | ----------------- | -------------------------------------- | ----------------------------------------- |
| lora-target-modules        | LoRA目标模块      | []                                     | LoRA目标模块列表                          |
| lora-load                  | LoRA模型加载路径  | None                                   | 包含LoRA模型检查点的目录                  |
| lora-r                     | LoRA秩            | 16                                     | LoRA的秩                                  |
| lora-alpha                 | LoRA alpha值      | 32                                     | LoRA的alpha值                             |
| lora-modules-to-save       | LoRA保存模块      | None                                   | 要保存的LoRA模块                          |
| lora-register-forward-hook | LoRA注册前向钩子  | ['word_embeddings', 'input_layernorm'] | LoRA注册前向钩子                          |
| lora-fusion                | 使用LoRA融合      | False                                  | 使用融合加速LoRA                          |
| lora-ckpt-filter           | 仅保存LoRA检查点  | False                                  | 启用仅保存LoRA检查点                      |
| qlora                      | 启用QLoRA         | False                                  | 启用QLoRA以降低内存使用                   |
| qlora-save-dequantize      | QLoRA保存时反量化 | False                                  | 在QLoRA调优中保存时将权重反量化为原始精度 |

### 9. 数据相关参数 (Data)

| 参数名                           | 中文含义             | 默认值  | 说明                                                         |
| -------------------------------- | -------------------- | ------- | ------------------------------------------------------------ |
| is-instruction-dataset           | 使用指令数据集       | False   | 是否使用指令数据集                                           |
| full-shuffle-instruction-dataset | 完全打乱指令数据集   | False   | 是否完全打乱指令数据集                                       |
| variable-seq-lengths             | 使用可变序列长度     | False   | 是否使用可变序列长度                                         |
| tokenizer-kwargs                 | tokenizer参数        | None    | Huggingface tokenizer的参数                                  |
| tokenizer-padding-side           | tokenizer填充侧      | 'right' | tokenizer填充侧                                              |
| tokenizer-type                   | tokenizer类型        | None    | 可选: 'BertWordPieceLowerCase', 'BertWordPieceCase', 'GPT2BPETokenizer', 'SentencePieceTokenizer', 'GPTSentencePieceTokenizer', 'Llama2Tokenizer', 'PretrainedFromHF', 'NullTokenizer' |
| tokenizer-name-or-path           | tokenizer名称或路径  | None    | Huggingface tokenizer的名称或路径                            |
| tokenizer-not-use-fast           | 不使用快速tokenizer  | False   | Huggingface tokenizer不使用快速版本                          |
| input-layernorm-in-fp32          | 输入层归一化使用fp32 | False   | 将输入层归一化转换为fp32                                     |
| no-shuffle                       | 禁用数据打乱         | False   | 禁用数据打乱，主要用于损失比较                               |
| neat-pack                        | 使用zigzag注意力掩码 | False   | 使用zigzag注意力掩码                                         |
| padded-samples                   | 填充样本             | False   | 在epoch内填充缺失的样本，从索引0开始，与LlamaFatory对齐      |

### 10. 融合操作参数 (Fusion Operations)

| 参数名                          | 中文含义                      | 默认值 | 说明                          |
| ------------------------------- | ----------------------------- | ------ | ----------------------------- |
| use-fused-rmsnorm               | 使用融合的rmsnorm             | False  | 使用融合的rmsnorm             |
| use-fused-swiglu                | 使用融合的swiglu              | False  | 使用融合的swiglu              |
| use-fused-rotary-pos-emb        | 使用融合的rotary位置嵌入      | False  | 使用融合的rotary位置嵌入      |
| use-fused-ring-attention-update | 使用融合的ring注意力更新      | False  | 使用融合的ring注意力更新      |
| use-mc2                         | 在tp中使用mc2进行计算通信重叠 | False  | 在tp中使用mc2进行计算通信重叠 |
| use-fused-mlp                   | 使用融合的mlp                 | False  | 使用融合的mlp                 |

### 11. 网络大小参数 (Network Size)

| 参数名                 | 中文含义       | 默认值 | 说明                           |
| ---------------------- | -------------- | ------ | ------------------------------ |
| padded-vocab-size      | 填充词汇表大小 | None   | 设置填充词汇表大小             |
| embed-layernorm        | 嵌入层归一化   | False  | 设置填充词汇表大小             |
| use-glm-rope           | 使用GLM RoPE   | False  | 在GLM模型中使用自定义部分rope  |
| sliding-window         | 滑动窗口大小   | None   | 使用滑动窗口注意力时的窗口大小 |
| output-layer-slice-num | 输出层切片数   | 1      | 设置输出层权重的切片数         |

### 12. 上下文并行参数 (Context Parallel)

| 参数名                             | 中文含义                   | 默认值            | 说明                                                         |
| ---------------------------------- | -------------------------- | ----------------- | ------------------------------------------------------------ |
| context-parallel-algo              | 上下文并行算法             | 'ulysses_cp_algo' | 可选: 'ulysses_cp_algo', 'megatron_cp_algo', 'hybrid_cp_algo', 'adaptive_cp_algo', 'hybrid_adaptive_cp_algo' |
| ulysses-degree-in-cp               | Ulysses CP度               | None              | Ulysses CP度                                                 |
| attention-mask-type                | 注意力掩码类型             | 'causal'          | 可选: 'causal', 'general'                                    |
| cp-attention-mask-type             | CP注意力掩码类型           | 'causal'          | 可选: 'causal', 'general'                                    |
| use-cp-send-recv-overlap           | 使用CP发送接收重叠         | False             | 启用cp发送接收重叠                                           |
| cp-window-size                     | CP窗口大小                 | 1                 | 双环注意力的内部窗口大小                                     |
| attention-mask-on-cpu              | 在CPU上存储完整注意力掩码  | False             | 在CPU而不是NPU上存储完整注意力掩码                           |
| adaptive-cp-without-coarse         | 自适应CP不粗化             | False             | 在adaptive_cp特性中不粗化注意力掩码，仅当完整序列长度小于8K且动态注意力掩码不可行时推荐 |
| adaptive-cp-dynamic-attn-mask      | 自适应CP动态注意力掩码     | False             | 注意力掩码是否在批次之间动态变化                             |
| adaptive-cp-only-reschedule        | 自适应CP仅重新调度         | False             | 在adaptive-cp特性中不应用重映射，仅重新调度进程              |
| adaptive-cp-manually-set-mask-list | 自适应CP手动设置掩码列表   | False             | 手动设置预制的注意力掩码列表                                 |
| kv-head-repeat-before-uly-alltoall | 在uly alltoall之前重复kv头 | True              | 在使用GQA/MQA时为ulysses扩展key和value                       |

### 13. 2D张量并行参数 (2D Tensor Parallel)

| 参数名                                 | 中文含义                       | 默认值 | 说明                                     |
| -------------------------------------- | ------------------------------ | ------ | ---------------------------------------- |
| tp-2d                                  | 使用2D张量并行                 | False  | 使用2D张量并行替代megatron风格的张量并行 |
| tp-x                                   | 第一个维度张量并行大小         | 1      | Linear的第一个维度张量并行大小           |
| tp-y                                   | 第二个维度张量并行大小         | 1      | Linear的第二个维度张量并行大小           |
| enable-overlap-ag-with-matmul          | 启用all-gather与matmul重叠     | False  | 启用all-gather与matmul重叠               |
| enable-overlap-matmul-with-rs          | 启用matmul与reduce-scatter重叠 | False  | 启用matmul与reduce-scatter重叠           |
| enable-backward-overlap-ag-with-matmul | 启用反向all-gather与matmul重叠 | False  | 在反向中启用all-gather与matmul重叠       |

### 14. 通信重叠参数 (Communication Overlap)

| 参数名              | 中文含义          | 默认值 | 说明                                                         |
| ------------------- | ----------------- | ------ | ------------------------------------------------------------ |
| async-log-allreduce | 异步日志allreduce | False  | 将用于传输日志信息的AllReduce操作转换为异步操作以减少通信开销，在跨数据中心(DC)训练中很有用 |

### 15. 其他参数 (Others)

| 参数名                  | 中文含义              | 默认值    | 说明                                      |
| ----------------------- | --------------------- | --------- | ----------------------------------------- |
| ai-framework            | AI框架                | 'pytorch' | 可选: 'pytorch', 'mindspore'              |
| use-mcore-models        | 使用Megatron-Core模型 | False     | 使用Megatron-Core模型，将在未来版本中弃用 |
| hccl-group-buffer       | HCCL组缓冲区          | None      | 组的hccl缓冲区                            |
| no-shared-storage       | 无共享存储            | False     | 如果没有共享存储，设置它                  |
| dataset-additional-keys | 数据集附加键          | []        | 需要从数据集添加的附加键                  |



### 16. ALiBi (Attention with Linear Biases) 参数

| 参数名                  | 中文含义      | 默认值 | 说明                        |
| ----------------------- | ------------- | ------ | --------------------------- |
| position-embedding-type | 位置嵌入类型  | None   | 可选值包含 'alibi'          |
| square-alibi-mask       | 方形ALiBi掩码 | False  | ALiBi的注意力掩码是否为方形 |
| fill-neg-inf            | 填充负无穷    | False  | 是否用负无穷填充ALiBi       |

这些参数用于配置ALiBi（Attention with Linear Biases）位置编码。ALiBi是一种替代传统位置编码的方法，它通过向注意力分数添加线性偏置来编码位置信息。

主要特点：
1. `position-embedding-type` 需要设置为 'alibi' 来启用ALiBi
2. `square-alibi-mask` 控制是否使用方形注意力掩码
3. `fill-neg-inf` 控制是否用负无穷值填充ALiBi矩阵



### 17. NDMM (N-Dimensional Matrix Multiplication) 参数

| 参数名        | 中文含义                   | 默认值 | 说明                                                |
| ------------- | -------------------------- | ------ | --------------------------------------------------- |
| use-nd-matmul | 使用N维矩阵乘法            | False  | 使用N维矩阵乘法替代megatron风格的张量并行           |
| nd1-dim1-size | 第一个ND矩阵乘法的Dim1大小 | 1      | 当use-3d-matmul为True时，第一个ND矩阵乘法的Dim1大小 |
| nd2-dim1-size | 第二个ND矩阵乘法的Dim1大小 | 1      | 当use-3d-matmul为True时，第二个ND矩阵乘法的Dim1大小 |

这些参数用于配置N维矩阵乘法（NDMM）相关的功能：

1. `use-nd-matmul`：
   - 这是一个开关参数，用于决定是否使用N维矩阵乘法
   - 当设置为True时，将使用N维矩阵乘法来替代传统的megatron风格的张量并行

2. `nd1-dim1-size`：
   - 控制第一个ND矩阵乘法的第一个维度大小
   - 当启用3D矩阵乘法时使用
   - 默认值为1

3. `nd2-dim1-size`：
   - 控制第二个ND矩阵乘法的第一个维度大小
   - 当启用3D矩阵乘法时使用
   - 默认值为1

这些参数主要用于优化大规模模型训练中的矩阵乘法操作，特别是在分布式训练场景下。通过使用N维矩阵乘法，可以更灵活地处理不同维度的张量并行计算。

让我帮你整理 `_add_dualpipe_args` 相关的参数：



### 18. DualPipe (双管道) 参数

| 参数名                     | 中文含义                | 默认值 | 说明                                          |
| -------------------------- | ----------------------- | ------ | --------------------------------------------- |
| moe-fb-overlap             | MoE前向反向重叠         | False  | 启用MoE前向和反向计算的重叠                   |
| schedules-method           | 调度方法                | None   | 调度方法，目前仅支持 'dualpipev'              |
| dualpipev-dw-detach        | DualPipeV梯度分离       | False  | 在冷却阶段分离梯度以减少气泡                  |
| moe-unperm2-mem-optim      | MoE unperm2内存优化     | False  | 通过在激活函数后乘以概率来释放unperm2激活内存 |
| moe-unperm2-mem-optim-swap | MoE unperm2内存优化交换 | False  | 启用MoE unperm2内存优化交换                   |



### 19. MTP (Multi-Token Prediction) 参数

| 参数名                   | 中文含义          | 默认值 | 说明                                                         |
| ------------------------ | ----------------- | ------ | ------------------------------------------------------------ |
| mtp-num-layers           | MTP层数           | None   | 多令牌预测(MTP)的层数。MTP扩展了预测范围，在每个位置预测多个未来令牌。此MTP实现使用D个顺序模块预测D个额外令牌 |
| mtp-loss-scaling-factor  | MTP损失缩放因子   | 0.1    | 多令牌预测(MTP)损失的缩放因子。我们计算所有深度的MTP损失的平均值，并将其乘以缩放因子以获得总体MTP损失，作为额外的训练目标 |
| recompute-mtp-norm       | 重计算MTP归一化   | False  | 多令牌预测重计算归一化                                       |
| recompute-mtp-layer      | 重计算MTP层       | False  | 多令牌预测重计算层                                           |
| mtp-mem-efficient-logits | MTP内存高效logits | False  | 在使用mtp块时优化ce_loss内存                                 |

这些参数用于配置多令牌预测（Multi-Token Prediction，MTP）相关的功能：

1. `mtp-num-layers`：
   - 控制MTP的层数
   - MTP允许模型在每个位置预测多个未来令牌
   - 使用D个顺序模块来预测D个额外令牌

2. `mtp-loss-scaling-factor`：
   - 控制MTP损失的缩放因子
   - 默认值为0.1
   - 用于计算总体MTP损失，作为额外的训练目标

3. `recompute-mtp-norm`：
   - 控制是否重计算MTP的归一化层
   - 用于优化内存使用

4. `recompute-mtp-layer`：
   - 控制是否重计算整个MTP层
   - 用于优化内存使用

5. `mtp-mem-efficient-logits`：
   - 启用MTP块的内存优化
   - 特别用于优化交叉熵损失（ce_loss）的内存使用

这些参数主要用于优化模型的多令牌预测能力，同时平衡计算效率和内存使用。MTP是一种扩展模型预测能力的技术，允许模型同时预测多个未来令牌，而不是传统的单令牌预测。



### 20. 默认模型参数 (Default Model Arguments)

| 参数名           | 中文含义              | 默认值 | 说明                                      |
| ---------------- | --------------------- | ------ | ----------------------------------------- |
| use-mcore-models | 使用Megatron-Core模型 | False  | 使用Megatron-Core模型，将在未来版本中弃用 |

这个参数用于控制是否使用Megatron-Core模型：

1. `use-mcore-models`：
   - 这是一个布尔型参数
   - 当设置为True时，将使用Megatron-Core模型
   - 当设置为False时，将使用传统模型
   - 根据注释，这个参数将在未来版本中被弃用
   - 默认值为False

这个参数的主要作用是：
- 控制模型架构的选择
- 决定是否使用新的Megatron-Core模型实现
- 为未来版本迁移做准备

需要注意的是，这个参数被标记为将在未来版本中弃用，建议在新代码中避免使用这个参数，而是直接使用新的模型实现。



### 21. MLA (Multi-head Latent Attention) 参数

| 参数名                      | 中文含义              | 默认值 | 说明                                        |
| --------------------------- | --------------------- | ------ | ------------------------------------------- |
| multi-head-latent-attention | 使用多头潜在注意力    | False  | 使用多头潜在注意力(MLA)                     |
| padded-base-length          | 填充基础长度          | 128    | 将多头潜在注意力的Q K V填充到此参数的整数倍 |
| q-lora-rank                 | Q的LoRA秩             | None   | Q的低秩值                                   |
| kv-lora-rank                | K和V的LoRA秩          | None   | K和V的低秩值                                |
| v-head-dim                  | V的头维度             | None   | V的头维度                                   |
| qk-rope-head-dim            | QK的RoPE头维度        | None   | 用于rope的qk头维度                          |
| qk-nope-head-dim            | QK的Nope头维度        | None   | 仅用于自注意力的qk头维度                    |
| mla-fa-without-pad          | MLA FA不填充          | False  | 在MLA中不将v_head_dim填充到q_head_dim       |
| mla-mm-split                | MLA矩阵乘法分割       | False  | 在MLA中将2个上投影矩阵乘法分割为4个         |
| mla-zero-memory             | MLA零内存             | False  | 在多头潜在注意力中保存激活内存              |
| mla-up-proj-tp-overlap      | MLA上投影TP重叠       | False  | 重叠上投影TP通信                            |
| recompute-mla-up-proj       | 重计算MLA上投影       | False  | 在MLA中重计算上投影                         |
| mla-swap-core-attn-out      | 交换MLA核心注意力输出 | False  | 仅在MLA中交换core_attn_out                  |
| mla-fa-divide-qk            | MLA FA分离QK          | False  | Flash attn支持分离q和k的MLA                 |

这些参数用于配置多头潜在注意力（Multi-head Latent Attention，MLA）相关的功能：

1. 基础配置：
   - `multi-head-latent-attention`：启用MLA功能
   - `padded-base-length`：控制QKV的填充长度

2. 维度配置：
   - `q-lora-rank`：Q的LoRA秩
   - `kv-lora-rank`：K和V的LoRA秩
   - `v-head-dim`：V的头维度
   - `qk-rope-head-dim`：用于RoPE的QK头维度
   - `qk-nope-head-dim`：用于自注意力的QK头维度

3. 优化配置：
   - `mla-fa-without-pad`：控制是否在MLA中填充V头维度
   - `mla-mm-split`：控制是否分割上投影矩阵乘法
   - `mla-zero-memory`：控制是否在MLA中保存激活内存
   - `mla-up-proj-tp-overlap`：控制是否重叠上投影TP通信
   - `recompute-mla-up-proj`：控制是否重计算MLA上投影
   - `mla-swap-core-attn-out`：控制是否交换核心注意力输出
   - `mla-fa-divide-qk`：控制是否支持分离QK的Flash attention

这些参数主要用于：
1. 配置MLA的基本架构
2. 控制各个组件的维度
3. 优化内存使用和计算效率
4. 支持不同的注意力机制和优化策略

使用这些参数时需要注意：
1. 某些参数之间存在依赖关系
2. 部分参数需要配合其他功能（如Flash attention）使用
3. 参数的选择会影响模型的性能和内存使用

让我帮你整理 `_add_deepseek_moe_args` 相关的参数：

### 22. DeepSeek MoE 参数

| 参数名                | 中文含义         | 默认值 | 说明                                                         |
| --------------------- | ---------------- | ------ | ------------------------------------------------------------ |
| moe-intermediate-size | MoE层FFN隐藏大小 | None   | MoE层的FFN隐藏层大小                                         |
| n-shared-experts      | 共享专家数量     | None   | 共享专家的数量，等于共享专家的intermediate_size除以moe_intermediate_size |
| first-k-dense-replace | 前K层密集替换    | None   | 将前K层设置为密集层                                          |
| moe-layer-freq        | MoE层频率        | None   | 设置MoE层的出现频率                                          |

这些参数用于配置DeepSeek的MoE（Mixture of Experts）模型：

1. `moe-intermediate-size`：
   - 控制MoE层的FFN（前馈网络）隐藏层大小
   - 用于定义每个专家的内部维度

2. `n-shared-experts`：
   - 控制共享专家的数量
   - 这个值等于共享专家的intermediate_size除以moe_intermediate_size
   - 用于定义模型中共有多少个共享专家

3. `first-k-dense-replace`：
   - 控制将前K层设置为密集层
   - 用于在模型的前几层使用传统的密集层而不是MoE层

4. `moe-layer-freq`：
   - 控制MoE层的出现频率
   - 用于定义在模型中每隔多少层出现一个MoE层

这些参数主要用于：
1. 配置MoE模型的基本架构
2. 控制专家层的分布和大小
3. 优化模型的计算效率和性能

使用这些参数时需要注意：
1. 这些参数需要与MoE相关的其他参数配合使用
2. 参数的选择会影响模型的性能和计算效率
3. 某些参数之间存在依赖关系，需要合理配置



### 23. 性能分析器 (Profiler) 参数

| 参数名                      | 中文含义         | 默认值          | 说明                                                       |
| --------------------------- | ---------------- | --------------- | ---------------------------------------------------------- |
| profile-ranks               | 要分析的全局rank | [-1]            | 要分析的全局rank列表。默认值-1表示分析所有rank             |
| profile-export-type         | 分析导出类型     | 'text'          | 选择导出模式为text或db                                     |
| profile-level               | 分析级别         | 'level0'        | 分析级别，可选：'level_none', 'level0', 'level1', 'level2' |
| profile-data-simplification | 数据简化模式     | False           | 使用数据简化模式                                           |
| profile-with-stack          | 带堆栈信息分析   | False           | 分析时包含堆栈信息                                         |
| profile-with-memory         | 带内存信息分析   | False           | 分析时包含内存信息                                         |
| profile-record-shapes       | 带形状信息分析   | False           | 分析时包含形状信息                                         |
| profile-with-cpu            | 带CPU信息分析    | False           | 分析时包含CPU信息                                          |
| profile-save-path           | 分析文件保存路径 | './profile_dir' | 保存分析文件的路径                                         |

这些参数用于配置性能分析器：

1. 基本配置：
   - `profile-ranks`：指定要分析的进程rank
   - `profile-export-type`：选择分析结果的导出格式
   - `profile-level`：设置分析的详细程度
   - `profile-save-path`：指定分析结果的保存位置

2. 分析内容配置：
   - `profile-data-simplification`：是否使用数据简化模式
   - `profile-with-stack`：是否包含堆栈信息
   - `profile-with-memory`：是否包含内存使用信息
   - `profile-record-shapes`：是否记录张量形状信息
   - `profile-with-cpu`：是否包含CPU相关信息

这些参数主要用于：
1. 性能分析和调试
2. 内存使用分析
3. 计算效率分析
4. 系统资源使用分析

使用这些参数时需要注意：
1. 不同分析级别会收集不同详细程度的信息
2. 收集更多信息会增加分析开销
3. 需要合理选择要分析的rank，避免分析过多进程
4. 分析结果会占用存储空间，需要确保有足够的存储空间



### 24. YaRN (Yet another RoPE extension) 参数

| 参数名                                        | 中文含义                  | 默认值 | 说明                                |
| --------------------------------------------- | ------------------------- | ------ | ----------------------------------- |
| rope-scaling-beta-fast                        | YaRN rope beta fast       | 32     | YaRN rope: rope beta fast参数       |
| rope-scaling-beta-slow                        | YaRN rope beta slow       | 1      | YaRN rope: rope beta slow参数       |
| rope-scaling-factor                           | YaRN rope factor          | 1.0    | YaRN rope: rope factor参数          |
| rope-scaling-mscale                           | YaRN rope mscale          | 1.0    | YaRN rope: rope mscale参数          |
| rope-scaling-mscale-all-dim                   | YaRN rope mscale all dim  | 0.0    | YaRN rope: rope mscale all dim参数  |
| rope-scaling-original-max-position-embeddings | YaRN rope原始最大位置嵌入 | None   | YaRN rope: rope原始最大位置嵌入参数 |

这些参数用于配置YaRN（Yet another RoPE extension）位置编码：

1. 基础配置：
   - `rope-scaling-beta-fast`：YaRN的快速beta参数，默认值为32
   - `rope-scaling-beta-slow`：YaRN的慢速beta参数，默认值为1
   - `rope-scaling-factor`：YaRN的缩放因子，默认值为1.0

2. 高级配置：
   - `rope-scaling-mscale`：YaRN的mscale参数，默认值为1.0
   - `rope-scaling-mscale-all-dim`：YaRN的全局维度mscale参数，默认值为0.0
   - `rope-scaling-original-max-position-embeddings`：原始最大位置嵌入大小，默认值为None

这些参数主要用于：
1. 扩展RoPE（Rotary Position Embedding）的位置编码能力
2. 控制位置编码的缩放和调整
3. 优化模型对长序列的处理能力

使用这些参数时需要注意：
1. 这些参数需要与RoPE相关的其他参数配合使用
2. 参数的选择会影响模型对位置信息的编码能力
3. 某些参数之间存在依赖关系，需要合理配置
4. 这些参数主要用于优化模型处理长序列的能力



### 25. 上下文并行（CP）参数验证规则

1. **基础验证**：
   - 如果 `context_parallel_size <= 1`：
     - 禁用 `kv_head_repeat_before_uly_alltoall`
     - 禁用 `use_fused_ring_attention_update`
     - 直接返回

2. **强制启用 Flash Attention**：
   - 在上下文并行中强制使用 Flash Attention
   - 设置 `use_flash_attn = True`

3. **Mcore 模型验证**：
   - 必须使用 Mcore 模型
   - 如果 `use_mcore_models` 为 False，则抛出错误

4. **Ulysses CP 算法验证**：
   当 `context_parallel_algo == 'ulysses_cp_algo'` 时：
   - 序列长度必须能被 `context_parallel_size` 整除
   - 检查注意力头数配置：
     - 注意力头数必须能被 `ulysses_size * tensor_model_parallel_size` 整除
     - 如果使用 GQA，`num_query_groups` 必须能被 `tensor_model_parallel_size` 整除
     - 如果 `kv_head_repeat_before_uly_alltoall` 未启用，`num_query_groups` 必须能被 `ulysses_size * tensor_model_parallel_size` 整除

5. **Megatron CP 算法验证**：
   当 `context_parallel_algo == 'megatron_cp_algo'` 时：
   - 序列长度必须能被 `2 * context_parallel_size` 整除
   - `cp_window_size` 必须在 [1, context_parallel_size) 范围内
   - `context_parallel_size` 必须能被 `cp_window_size` 整除
   - 如果 `cp_attention_mask_type == 'general'`，则 `micro_batch_size` 必须为 1

6. **Hybrid CP 算法验证**：
   当 `context_parallel_algo == 'hybrid_cp_algo'` 时：
   - 必须指定 `ulysses_degree_in_cp`
   - `context_parallel_size` 必须能被 `ulysses_degree_in_cp` 整除
   - 序列长度必须能被 `2 * context_parallel_size` 整除
   - `cp_window_size` 必须在 [1, ring_degree) 范围内
   - `ring_degree` 必须能被 `cp_window_size` 整除
   - 检查注意力头数配置
   - 如果 `cp_attention_mask_type == 'general'`，则 `micro_batch_size` 必须为 1

7. **Adaptive CP 算法验证**：
   当 `context_parallel_size > 1` 且 `context_parallel_algo == 'adaptive_cp_algo'` 时：
   - 序列长度必须能被 `context_parallel_size` 整除
   - 如果 `cp_attention_mask_type == 'general'`，则 `micro_batch_size` 必须为 1

8. **Hybrid Adaptive CP 算法验证**：
   当 `context_parallel_size > 1` 且 `context_parallel_algo == 'hybrid_adaptive_cp_algo'` 时：
   - 必须指定 `ulysses_degree_in_cp`
   - `context_parallel_size` 必须能被 `ulysses_degree_in_cp` 整除
   - 注意力头数必须能被 `ulysse-degree-in-cp * tensor_model_parallel_size` 整除
   - 序列长度必须能被 `context_parallel_size` 整除
   - 如果 `cp_attention_mask_type == 'general'`，则 `micro_batch_size` 必须为 1

9. **滑动窗口验证**：
   - 如果启用滑动窗口，则不允许使用上下文并行

使用上下文并行时需要注意：
1. 必须使用 Mcore 模型
2. 必须使用 Flash Attention
3. 需要正确配置序列长度、注意力头数等参数
4. 不同算法有不同的参数要求
5. 某些配置组合可能不兼容



### 26. 重计算参数验证规则

1. **PP VPP 验证**：
```python
enable_pp_vpp = args.num_layers_per_virtual_pipeline_stage
enable_vanilla_recomputation = args.recompute_granularity is not None and args.recompute_method == 'block'
enable_swap = args.swap_attention
enable_recompute_activation = args.recompute_activation_function
enable_recomputation = enable_vanilla_recomputation or enable_swap or enable_recompute_activation

if args.enable_recompute_layers_per_pp_rank and not (enable_pp_vpp and enable_recomputation):
    raise AssertionError("enable-recompute-layers-per-pp-rank should be works with pipeline and virtual pipeline, when enabling re-computation.")
```
- 检查是否启用了管道并行和虚拟管道并行
- 检查是否启用了重计算
- 如果启用了按PP rank重计算层，则必须同时启用管道并行和重计算

2. **激活函数重计算验证**：
```python
if args.recompute_activation_function:
    if args.recompute_method == "uniform":
        raise AssertionError('uniform recomputation is not compatible with activation function recomputation.')
    if args.recompute_granularity == "selective":
        raise AssertionError('--recompute-activation-function is not compatible with selective recomputation.')
```
- 如果启用激活函数重计算：
  - 不允许使用均匀重计算
  - 不允许使用选择性重计算

3. **归一化重计算验证**：
```python
if args.recompute_norm:
    if args.recompute_method == "uniform":
        raise AssertionError('uniform recomputation is not compatible with norm recomputation.')
    if args.recompute_granularity == "selective":
        raise AssertionError('--recompute-norm is not compatible with selective recomputation')
    if not args.use_mcore_models:
        raise AssertionError('--recompute-norm is only supported with mcore models')
```
- 如果启用归一化重计算：
  - 不允许使用均匀重计算
  - 不允许使用选择性重计算
  - 必须使用 Mcore 模型

4. **Swap Attention 验证**：
```python
if args.swap_attention and args.swap_modules is None:
    if args.use_mcore_models:
        args.swap_modules = "input_layernorm,self_attention,pre_cross_attn_layernorm"
    else:
        args.swap_modules = "input_norm,self_attention,post_attention_norm"
```
- 如果启用 Swap Attention 但未指定交换模块：
  - 对于 Mcore 模型，设置默认交换模块
  - 对于其他模型，设置不同的默认交换模块

#### 参数依赖关系

1. **重计算类型**：
   - `recompute_granularity`：重计算粒度
   - `recompute_method`：重计算方法
   - `recompute_activation_function`：激活函数重计算
   - `recompute_norm`：归一化重计算

2. **管道并行**：
   - `num_layers_per_virtual_pipeline_stage`：虚拟管道并行层数
   - `enable_recompute_layers_per_pp_rank`：按PP rank重计算层

3. **模型类型**：
   - `use_mcore_models`：是否使用 Mcore 模型
   - `swap_attention`：是否使用交换注意力
   - `swap_modules`：交换模块列表

#### 验证规则总结

1. **功能兼容性**：
   - 重计算与管道并行的兼容性
   - 不同重计算方法的兼容性
   - 模型类型与重计算的兼容性

2. **配置依赖**：
   - 重计算需要特定的模型类型
   - 某些重计算功能需要特定的配置

3. **默认值设置**：
   - 根据模型类型设置默认交换模块
   - 确保配置的完整性

#### 使用注意事项

1. **配置顺序**：
   - 先设置基础重计算参数
   - 然后配置相关功能
   - 最后进行验证

2. **兼容性检查**：
   - 检查重计算与其他功能的兼容性
   - 检查模型类型与重计算的兼容性
   - 检查管道并行的配置

3. **性能考虑**：
   - 重计算对内存使用的影响
   - 不同重计算方法的效率
   - 管道并行的优化

4. **错误处理**：
   - 提供清晰的错误信息
   - 在配置冲突时及时报错

这个验证函数确保了重计算相关配置的正确性和兼容性，同时处理了各种特殊情况下的配置需求。
