#  MindSpeedRL 使用教程



##  简介

[Group Relative Policy Optimization (GRPO) ](https://gitee.com/link?target=https%3A%2F%2Farxiv.org%2Fpdf%2F2402.03300)是 Deepseek-Math中提出的训练方法，它移除了 PPO 中对 Critic 模型的依赖，而是通过计算同一prompt多次重复采样输出的相对奖励来估计优势函数，这一创新大大减少了显存占用，提高了算法在强化学习任务中的效率。

在 GRPO 方法中包含了三个关键模型：Actor，Reference，Reward。其中 Actor 和 Reference 模型是通过SFT后得到的策略模型，而 Reward 模型则是通过训练构建的奖励评估模型。GRPO 的核心训练目标是优化 Actor 模型的策略，使其在执行强化学习任务时能够产生更优的动作序列，更符合任务目标的预期。



## 环境配置

### 版本要求

本版本为**预览非正式发布**版本， 依赖配套如下表.

| 依赖软件                               | 版本     |
| -------------------------------------- | -------- |
| 昇腾NPU驱动                            | 25.0.RC1 |
| 昇腾NPU固件                            | 25.0.RC1 |
| Toolkit（开发套件）                    | 8.1.RC1  |
| Kernel（算子包）                       | 8.1.RC1  |
| NNAL（Ascend Transformer Boost加速库） | 8.1.RC1  |
|                                        |          |
| Python                                 | 3.10     |
| torch                                  | 2.5.1    |
| torch_npu                              | 2.5.1    |
| apex                                   | 0.1      |
| ray                                    | 2.42.1   |
| vllm                                   | 0.7.3    |

### 驱动固件安装

```shell
bash Ascend-hdk-*-npu-firmware_*.run --full
bash Ascend-hdk-*-npu-driver_*.run --full
```

### CANN安装

```shell
bash Ascend-cann-toolkit_8.1.RC1_linux-aarch64.run --install
bash Ascend-cann-kernels-*_8.1.RC1_linux-aarch64.run --install
bash Ascend-cann-nnal_8.1.RC1_linux-aarch64.run --install

# 设置环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/asdsip/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
```

### PTA安装

```shell
# 安装torch和torch_npu
pip install torch-2.5.1-cp310-cp310-*.whl
pip install torch_npu-2.5.1.*.manylinux2014_aarch64.whl

# apex for Ascend
# 构建参考 https://gitee.com/ascend/apex
pip install apex-0.1.dev*.whl
```

### vllm及相关依赖安装：

（注：环境中需要安装git，因为vllm的安装过程依赖git）

```shell
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout v0.7.3
pip install -r requirements-build.txt
VLLM_TARGET_DEVICE=empty pip install .
```

### vllm_ascend安装

```shell
git clone -b v0.7.3-dev https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git checkout 0713836e95fe993feefe334945b5b273e4add1f1
pip install -e .
```

### 准备源码

```shell
# Clone MindSpeed-RL repository
git clone https://gitee.com/ascend/MindSpeed-RL.git

# Clone and prepare MindSpeed
git clone https://gitee.com/ascend/MindSpeed.git
cd MindSpeed
git checkout 0dfa0035ec54d9a74b2f6ee2867367df897299df  # 参考MindSpeed-LLM依赖版本
pip install -r requirements.txt
cp -r mindspeed ../MindSpeed-RL/
cd ..

# Clone and prepare Megatron-LM
git clone https://github.com/NVIDIA/Megatron-LM.git  # Megatron从github下载，请确保网络能访问
cd Megatron-LM
git checkout core_r0.8.0
cp -r megatron ../MindSpeed-RL/
cd ..

# Clone and prepare MindSpeed-LLM
git clone https://gitee.com/ascend/MindSpeed-LLM.git
cd MindSpeed-LLM
git checkout 421ef7bcb83fb31844a1efb688cde71705c0526e
cp -r mindspeed_llm ../MindSpeed-RL/
cd ..

# Install MindSpeed-RL dependencies
cd ./MindSpeed-RL
pip install -r requirements.txt
pip install antlr4-python3-runtime==4.7.2 --no-deps
```

## 权重转换

### 模型选择

- Qwen2.5-7B该模型指令遵从度高，有一定概率能引导模型输出`<think>...</think><answer>...$\boxed{}</answer>`格式回复，训练曲线符合预期，在评测集上提升较大。

### 权重转换

在进行RL训练之前，模型需要从HuggingFace权重转换为megatron权重，可参考[**权重转换部分**](https://gitee.com/ascend/MindSpeed-RL/blob/master/docs/algorithms/grpo.md)

```shell
export CUDA_DEVICE_MAX_CONNECTIONS=1
# 修改 ascend-toolkit 路径
source set_env.sh
# conda
source /root/llmtuner/miniconda3/bin/activate mindspeed_rl

hf_model_path=/root/llmtuner/hfhub/models/Qwen/Qwen2.5-7B
mcore_model_path=/root/llmtuner/hfhub/mindspeed/models/Qwen/Qwen2.5-7B

tp=2
pp=4

# actor使用TP2PP4，将脚本里改成TP2PP4配置
# reference使用TP2PP2，将脚本里改成TP2PP2配置

# 设置需要的权重转换参数
python cli/convert_ckpt.py \
       --use-mcore-models \
       --model-type GPT \
       --load-model-type hf \
       --save-model-type mg \
       --target-tensor-parallel-size $tp \
       --target-pipeline-parallel-size $pp \
       --add-qkv-bias \
       --load-dir $hf_model_path \
       --save-dir $mcore_model_path/mcore_tp${tp}_pp${pp} \
       --tokenizer-model $hf_model_path/tokenizer.json \
       --model-type-hf llama2 \
       --params-dtype bf16
```



```shell
bash examples/ckpt/ckpt_convert_qwen25_hf2mcore.sh

# 训练完后如需要转回HF格式
bash examples/ckpt/ckpt_convert_qwen25_mcore2hf.sh
```

## 数据预处理

### 模板构造

- R1-Zero复现需要在数据处理时加上prompt模板激发`<think>...</think><answer>...$\boxed{}</answer>`

  ```
  <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nA conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>Put your final answer within \\boxed{}.\n{你真正的问题}<|im_end|>\n<|im_start|>assistant\n{模型真正的回答}
  ```

- 以上为默认的qwen_r1模板，根据模型和数据的不同，用户可以在`configs/model/templates.json`添加自己的**自定义模板**

### 选择数据集

对于7B模型应使用难度适中的数据集，所以我们使用Orz math 57K来训练

### 数据预处理

需要先配置数据处理的yaml文件(configs/datasets/r1_zero_qwen25_7b.yaml) 自定义数据集需要设置--map-keys映射，或重写自定义handler；具体参考[**数据集处理部分**](https://gitee.com/ascend/MindSpeed-RL/blob/master/docs/algorithms/grpo.md)

**Qwen2.5-7B**

- 处理的时候默认使用qwen_r1的模板

  ```
  # 启动转换
  bash examples/data/preprocess_data.sh r1_zero_qwen25_7b
  ```

## 配置文件

MindSpeed RL 通过将模型参数和训练配置解耦的层级化参数配置，来简化 GRPO 训练的参数配置过程。RLXF 训练涉及到的所有配置文件均存储在 configs/ 路径下，其中 model 文件夹下存储了模型结构相关的配置文件，GRPO 训练相关的模型参数文件以  `grpo_trainer_模型名_模型大小_机器型号.yaml` 方式命名。在每个 grpo_trainer 配置文件中，需要包含 defaults、megatron_training、actor_config、rl_config、generate_config 字段的参数配置。

1. defaults 负责引入模型配置文件，在 defaults 中应列举本配置文件中所需要用到的所有模型配置，模型配置可以在下方3个角色的具体配置中通过 model 字段进行选择。
2. megatron_training 字段设置的参数为所有 3 个角色通用的默认参数，这些参数可以在下方进一步被角色的单独配置所覆盖。
3. actor_config、ref_config 以及 reward_config：三个角色的训练配置。
4. rl_config: 在 GRPO 训练中的特性参数，以及 actor，reward，ref 模型的资源配置。
5. generate_config: 包含 tokenizer 相关配置、推理并行配置、vllm 模型相关设置以及样本采样参数配置。



### 配置文件字段解析

####  MegatronConfig 参数解析

这是一个用于配置 Megatron-LM 模型训练和推理的复杂配置类，继承自 `BaseConfig`。让我从几个主要方面来解释：

1. **模型架构配置**：
```python
# 基础模型参数
self.num_layers = None          # 模型层数
self.hidden_size = None         # 隐藏层大小
self.ffn_hidden_size = None     # 前馈网络隐藏层大小
self.num_attention_heads = None # 注意力头数
self.kv_channels = None         # KV投影维度
```

2. **注意力机制配置**：
```python
# 注意力相关配置
self.group_query_attention = False  # 是否使用分组查询注意力
self.num_query_groups = 1          # 查询组数量
self.attention_dropout = 0.1       # 注意力dropout率
self.position_embedding_type = 'learned_absolute'  # 位置编码类型
```

3. **并行计算配置**：
```python
# 并行计算参数
self.tensor_model_parallel_size = 1      # 张量并行大小
self.pipeline_model_parallel_size = 1    # 流水线并行大小
self.expert_model_parallel_size = 1      # 专家并行大小
self.context_parallel_size = 1           # 上下文并行大小
self.sequence_parallel = False           # 是否使用序列并行
```

4. **训练参数配置**：
```python
# 训练相关参数
self.global_batch_size = None           # 全局批次大小
self.micro_batch_size = None            # 微批次大小
self.train_iters = None                 # 训练迭代次数
self.lr = None                          # 学习率
self.weight_decay = 0.01                # 权重衰减
self.clip_grad = 1.0                    # 梯度裁剪
```

5. **优化器配置**：
```python
# 优化器参数
self.optimizer = 'adam'                 # 优化器类型
self.adam_beta1 = 0.9                  # Adam优化器beta1
self.adam_beta2 = 0.999                # Adam优化器beta2
self.lr_decay_style = 'linear'         # 学习率衰减方式
self.lr_warmup_fraction = None         # 学习率预热比例
```

6. **MoE (Mixture of Experts) 配置**：
```python
# MoE相关配置
self.moe_grouped_gemm = False          # 是否使用MoE分组矩阵乘法
self.moe_router_topk = 2               # MoE路由选择的专家数
self.num_experts = None                # 专家数量
self.moe_intermediate_size = None      # MoE中间层大小
self.moe_router_load_balancing_type = 'aux_loss'  # MoE负载均衡类型
```

7. **内存优化配置**：
```python
# 内存优化参数
self.recompute_granularity = None      # 重计算粒度
self.recompute_method = None           # 重计算方法
self.recompute_num_layers = None       # 重计算层数
self.swap_attention = False            # 是否使用注意力交换
```

8. **数据集配置**：
```python
# 数据集相关参数
self.data_path = None                  # 数据路径
self.split = None                      # 数据分割
self.is_instruction_dataset = False    # 是否为指令数据集
self.is_pairwise_dataset = False       # 是否为成对数据集
self.variable_seq_lengths = False      # 是否使用可变序列长度
```

9. **推理配置**：
```python
# 推理相关参数
self.use_kv_cache = False              # 是否使用KV缓存
self.do_sample = False                 # 是否进行采样
```

10. **特殊功能配置**：
```python
# 特殊功能参数
self.use_flash_attn = False            # 是否使用Flash Attention
self.use_rotary_position_embeddings = False  # 是否使用旋转位置编码
self.use_fused_rmsnorm = False         # 是否使用融合RMSNorm
self.use_fused_swiglu = False          # 是否使用融合SwiGLU
```

**使用场景**：
1. 大规模语言模型训练配置
2. 分布式训练环境设置
3. 模型架构参数调整
4. 训练优化策略配置
5. 内存和计算优化设置

#### GenerateConfig 参数解析

这是一个用于控制模型生成（推理）过程的配置类，继承自 `BaseConfig`。让我从几个关键方面来解释：

1. **基本配置参数**：
```python
# 基础配置
self.data_parallel_size = None  # 数据并行大小
self.tokenizer_name_or_path = "/path/to/tokenizer"  # tokenizer路径
self.trust_remote_code = True  # 是否信任远程代码（如自定义tokenizer）
```

2. **并行计算配置**：
```python
# 推理时的并行配置
self.infer_tensor_parallel_size = 8      # 张量并行大小
self.infer_pipeline_parallel_size = 1    # 流水线并行大小
self.infer_expert_parallel_size = 1      # 专家并行大小
```
这些参数控制模型在推理时的并行计算策略，用于优化大规模模型的推理性能。

3. **序列处理配置**：
```python
self.max_num_seqs = 1           # 最大可处理的序列数量
self.max_model_len = 2048       # 模型最大长度（token数）
self.max_num_batched_tokens = 2048  # 批处理的最大token数
```
这些参数控制模型处理序列的能力和限制。

4. **硬件资源配置**：
```python
self.dtype = "bfloat16"                # 模型权重数据类型
self.gpu_memory_utilization = 0.5      # GPU内存利用率
self.offload_train_optimizer = False   # 是否卸载优化器到CPU
self.offload_train_grad = False        # 是否卸载梯度到CPU
self.offload_train_param = False       # 是否卸载参数到CPU
```
这些参数控制模型在硬件资源上的使用策略，包括内存管理和计算精度。

5. **采样配置**：
```python
self.sampling_config = {
    "logprobs": 1,        # 返回的top token的对数概率数量
    "max_tokens": 128,    # 生成输出的最大token数量
    "top_p": 1.0,         # 核采样的累积概率阈值
    "top_k": 50,          # 采样时考虑的最高概率token数量
    "min_p": 0.0,         # token选择的最小概率阈值
    "temperature": 0.2,   # 控制预测随机性的温度参数
    "detokenize": False   # 是否将生成的token转换回可读字符串
}
```
这是最重要的配置部分，控制文本生成时的采样策略：
- `temperature`: 控制生成的随机性，值越低生成越确定
- `top_p` 和 `top_k`: 控制采样范围，用于平衡生成的多样性和质量
- `max_tokens`: 控制生成文本的最大长度
- `logprobs`: 控制是否返回token的概率信息

6. **其他优化配置**：
```python
self.enable_prefix_caching = False  # 是否启用前缀缓存
self.num_scheduler_steps = 1        # 调度器步数
```
这些是用于优化推理性能的配置项。

7. **配置更新机制**：

支持通过配置字典动态更新默认配置。

**使用场景**：
1. 在RLHF训练过程中，用于控制模型生成样本时的行为
2. 在推理阶段，用于控制模型输出文本的生成策略
3. 在分布式训练中，用于控制模型并行和资源使用

####  RLConfig 参数解析

这是用于配置强化学习训练过程的核心配置类，继承自 `BaseConfig`。我将从几个主要方面来解释：

1. **基础配置参数**：
```python
# 运行时环境配置
self.runtime_env_path = 'configs/envs/runtime_env.yaml'  # 运行时环境配置文件路径
self.use_integrated_worker = False  # 是否使用集成工作节点
```

2. **奖励相关配置**：
```python
# 奖励模型配置
self.rule_reward = True  # 是否使用基于规则的奖励
self.beta = 0.1  # 规则奖励和模型奖励的平衡系数
self.verifier_function = ["base_acc"]  # 验证器函数列表
self.verifier_weight = [1.0]  # 验证器权重列表
self.verifier_parallel = 1  # 验证器并行数
self.verifier_timeout = 30  # 验证器超时时间
```

3. **资源分配配置**：
```python
# 资源分配
self.actor_resource = None  # Actor模型资源分配
self.reference_resource = None  # Reference模型资源分配
self.reward_resource = None  # Reward模型资源分配
self.num_cpus_for_local_task = 1  # 本地任务CPU数量
self.num_cpus_for_placement_group = 8  # 放置组CPU数量
```

4. **训练参数配置**：
```python
# 训练参数
self.num_samples_per_step = 1  # 每步采样数
self.max_prompt_length = 512  # 最大提示长度
self.epochs = 1  # 训练轮数
self.clip_ratio = 0.2  # 裁剪比率
self.entropy_coeff = 0.0  # 熵系数
self.gamma = 1.0  # 折扣因子
self.lam = 0.95  # GAE lambda参数
```

5. **KL散度控制配置**：
```python
# KL散度控制
self.kl_penalty = "low_var_kl"  # KL惩罚类型
self.kl_ctrl_type = 'fixed'  # KL控制类型
self.init_kl_coef = 0.01  # 初始KL系数
self.kl_horizon = 1000  # KL控制时间范围
self.kl_target = 100.0  # KL目标值
```

6. **批次处理配置**：
```python
# 批次处理
self.shuffle_mini_batch = False  # 是否打乱小批次
self.n_samples_per_prompt = 1  # 每个提示的样本数
self.mini_batch_size = 1  # 小批次大小
```

7. **调度配置**：
```python
# 调度相关
self.actor_rollout_dispatch_size = None  # Actor生成调度大小
self.actor_logprob_dispatch_size = None  # Actor对数概率调度大小
self.ref_dispatch_size = None  # Reference调度大小
self.reward_dispatch_size = None  # Reward调度大小
self.adv_dispatch_size = None  # 优势函数调度大小
self.actor_update_dispatch_size = None  # Actor更新调度大小
```

8. **日志和监控配置**：
```python
# 日志和监控
self.use_tensorboard = False  # 是否使用TensorBoard
self.use_wandb = False  # 是否使用Weights & Biases
self.wandb_project = ""  # W&B项目名
self.wandb_exp_name = ""  # W&B实验名
self.wandb_save_dir = ""  # W&B保存目录
```

**使用场景**：
1. 配置RLHF训练过程
2. 管理分布式训练资源
3. 控制奖励计算和验证
4. 优化训练参数
5. 监控训练过程

#### 参数校验

我来详细解释 `validate_rl_args` 函数。这是一个用于验证RLHF训练配置参数的关键函数，它检查多个配置类之间的参数一致性和合理性。让我从几个主要方面来解释：

1. **集成工作节点模式验证**：
```python
# 检查集成工作节点模式下的参数设置
if rl_config.use_integrated_worker:
    # 集成模式下不应设置reference_resource
    if rl_config.reference_resource is not None:
        raise ValueError("reference_resource should not be set when use_integrated_worker mode is on.")
    rl_config.reference_resource = rl_config.actor_resource

    # 集成模式下不支持reward模型
    if rl_config.reward_resource is not None:
        raise ValueError("Reward model is not supported when use_integrated_worker mode is on.")
```

2. **序列长度验证**：
```python
# 检查序列长度是否超过模型最大长度限制
if generate_config.max_model_len < actor_config.seq_length:
    raise ValueError(
        f"Sequence length exceeds vLLM max_model_len! "
        f"Actor.seq_length={actor_config.seq_length} vs "
        f"GenerateConfig.max_model_len={generate_config.max_model_len}")
```

3. **资源分配验证**：
```python
def _validate_resource(resource, t_size, p_size, c_size, component):
    # 验证资源分配是否合理
    product = t_size * p_size * c_size
    if resource.num_npus % product != 0:
        raise ValueError(
            f"Invalid {component} resource allocation! "
            f"Resource={resource} must be divisible by (tensor_parallel * pipeline_parallel * context_parallel)")

# 验证各个组件的资源分配
_validate_resource(rl_config.actor_resource, ...)
_validate_resource(rl_config.reference_resource, ...)
_validate_resource(rl_config.reward_resource, ...)
```

4. **批次大小验证**：
```python
def _validate_batch_ratio(global_batch, micro_batch, n_samples, component):
    # 验证批次大小关系
    if (global_batch * n_samples) % micro_batch != 0:
        raise ValueError(
            f"Invalid {component} batch configuration! "
            f"(global_batch_size * n_samples) must be divisible by micro_batch_size")

# 验证各个组件的批次大小
_validate_batch_ratio(actor_config.global_batch_size, ...)
_validate_batch_ratio(ref_config.global_batch_size, ...)
_validate_batch_ratio(reward_config.global_batch_size, ...)
```

5. **数据并行验证**：
```python
def _validate_data_parallel(global_batch_size, data_parallel, micro_batch_size, n_samples, component):
    # 验证数据并行配置
    if global_batch_size % data_parallel != 0:
        raise ValueError(f"{component} global_batch_size must be divisible by data_parallel_size")

    if (global_batch_size // data_parallel * n_samples) % micro_batch_size != 0:
        raise ValueError(f"{component} batch configuration invalid")

# 计算并验证数据并行度
actor_data_parallel_size = rl_config.actor_resource.num_npus // (
    actor_config.tensor_model_parallel_size *
    actor_config.pipeline_model_parallel_size *
    actor_config.context_parallel_size)
```

6. **经验计数配置**：
```python
# 初始化各个组件的经验计数
rl_config.actor_logprob_dispatch_size = (
    rl_config.actor_logprob_dispatch_size or
    (actor_config.global_batch_size * rl_config.n_samples_per_prompt // actor_data_parallel_size)
)
rl_config.ref_dispatch_size = ...
rl_config.adv_dispatch_size = ...
rl_config.reward_dispatch_size = ...
```

7. **经验计数验证**：
```python
def _validate_experience_ratio(global_batch, experience_count, component):
    # 验证经验计数与全局批次的关系
    if global_batch * rl_config.n_samples_per_prompt % experience_count != 0:
        raise ValueError(
            f"{component} global_batch_size must be divisible by experience_count")

# 验证各个组件的经验计数
_validate_experience_ratio(actor_config.global_batch_size, ...)
_validate_experience_ratio(ref_config.global_batch_size, ...)
_validate_experience_ratio(reward_config.global_batch_size, ...)
```

8. **验证器参数验证**：
```python
# 检查验证器函数和权重的数量是否匹配
if len(rl_config.verifier_function) != len(rl_config.verifier_weight):
    raise ValueError(
        f"Verifier function and weight length mismatch: "
        f"{len(rl_config.verifier_function)} vs {len(rl_config.verifier_weight)}")
```



### 训练参数Yaml 文件解析

`defaults:`

引入模型配置(网络结构需要定义在model目录的yaml文件下)：

- `model`: qwen25_7b

#### `megatron_training:`

- `stage`：用于指定训练算法，使用 Ray GRPO 训练须设置为`ray_grpo`；
- `global_batch_size`:  经过多少样本后 actor-train 和 rollout 权重同步；
- `data_path`: 数据集路径配置，例如 /dataset/data，注意带前缀；
- `tokenizer_name_or_path`: 分词器路径配置，可以配置为 Hugging Face 权重文件的文件夹路径，例如 /ckpt/qwen2.5_7b_hf/ ;
- `其余参数`: 其余参数为Megatron训练中的特性配置；

#### `actor_config、ref_config 以及 reward_config：`

配置 GRPO 训练中 Actor 模型、Reference 模型和 Reward 模型的配置参数；当前支持不开启 Reward 模型，开启规则奖励进行打分，开启参数详见rl_config中的rule_reward参数。

- `micro_batch_size`：梯度累积的 mbs 大小;

- `tensor_model_parallel_size`：TP 并行策略数;

- `pipeline_model_parallel_size`：PP 并行策略数;

- `lr`：学习率；

- `lr_decay_style`：学习率衰减配置；

- `min_lr`：最小学习率；

- `weight_decay`：权重衰减，用于防止模型过拟合；

- `lr_warmup_fraction`：学习率预热比例，在训练初期逐渐增大学习率的比例；

- `clip_grad`：梯度裁剪系数；

- `load`：模型加载的路径；

- `save`：模型保存的路径；

- `no_load_optim`：续训加载优化器状态，默认为false；

- `no_load_rng`：续训加载数据随机数生成器，默认为false；

- `no_save_optim`：保存优化器状态，默认为false；

- `no_save_rng`：保存数据随机数生成器，默认为false；



#### `rl_config:`

- `use_integrated_worker`：是否开启全共卡模式，默认为 true;
- `blocking`：是否开启异步，默认为 true;
- `actor_forward_micro_batch_size`：actor model 前向计算 logp 的 mbs 大小;
- `ref_forward_micro_batch_size`：ref model 前向计算 logp 的 mbs 大小;
- `adv_estimator`：优势计算方法;
- `kl_ctrl_type`：kl loss 计算方法;
- `init_kl_coef`：kl loss 所占权重;
- `mini_batch_size`：每 mini batch size 之后 actor 会更新一次;
- `max_prompt_length`：GRPO 训练中最大 prompt 长度，默认为512;
- `clip_ratio`：Actor 模型训练计算损失函数时的 clip 比例，默认为0.2 一般取值范围 [0.1，0.3] 最大取值范围[0，1] 该数值越大允许策略更新的幅度越大，反之不然；
- `entropy_coeff`: entropy loss 所占权重;
- `n_samples_per_prompt`：每条prompt的重用次数，一条 prompt 输入能输出 n 条 responese;
- `guarantee_order`: 是否开启TransferDock保序，默认 False;
- `shuffle_mini_batch`：Actor 训练时是否对 minibatch 进行 shuffle，默认为 False;
- `actor_resource` ：分配给 Actor 、Reference模型的显卡数量;



##### 显卡资源配置

显卡资源配置格式为 :

```
actor_resource:
    num_npus: 8
```



##### 规则奖励配置

开启规则奖励开关后，不用分配资源给 reward_resource 参数，规则奖励参数配置如下：

- `rule_reward`: 开启后，使用规则奖励进行打分；

- `verifier_function`: 选择使用的规则奖励模型方法，例如["acc", "strict_format"] ；

- `verifier_weight`: 配置规则奖励模型权重，例如[1.0, 1.0]；



##### 日志配置

当前支持 wandb/tensorboard 日志输出：

tensorboard开关（若use_tensorboard和use_wandb同时为True，则tensorboard不生效）:

- `use_tensorboard`: 配置为 True 时打开 tensorboard；

wandb开关:

- `use_wandb`: 配置为 True 时打开 wandb；
- `wandb_project`: project 名称配置；
- `wandb_exp_name`: 实验名称配置；
- `wandb_save_dir`: 本地存储 wandb 路径；

#### generate_config:

##### 推理时的并行配置

- `infer_tensor_parallel_size`：TP并行策略数；
- `infer_pipeline_parallel_size`：PP并行策略数；
- `infer_expert_parallel_size`：EP并行策略数；

##### resharding 相关配置

- `offload_train_optimizer`：卸载训练节点优化器；
- `offload_train_grad`：卸载训练节点梯度；
- `offload_train_param`：卸载模型权重；

##### vllm 模型相关设置

vllm 模型参数 可以参照 [vllm官网参数介绍](https://gitee.com/link?target=https%3A%2F%2Fdocs.vllm.ai%2Fen%2Flatest%2Fserving%2Fengine_args.html)：

- `dtype`：vllm 推理所使用的数据类型；

- `max_num_seqs`：vllm 推理并发最大样本限制；
- `max_model_len`：vllm 能够处理的最大输入序列长度(prompt+response)；
- `max_num_batched_tokens`：vllm 推理并发最大token限制；
- `gpu_memory_utilization`：GPU 内存利用率，指定推理时使用 GPU 内存的比例；
- `num_scheduler_steps `：指的是在一个完整的调度周期内，调度器会将批处理请求分成多少个子步骤来执行；

##### 采样配置

- `logprobs`：是否生成logprobs；
- `max_tokens`：单条response最大生成token数量；
- `temperature`：采样时的随机性参数；
- `top_p`：vllm 筛选出概率累积和达到top_p的token集合，随后只在这个集合里进行采样；
- `top_k`：vllm 会先选出概率最高的 top_k 个 token，然后在这 top_k 个 token 范围内进行采样；
- `min_p`：vllm 过滤掉概率低于 min_p 的词元，不参与后续的采样过程；
- `detokenize`：是否将输出token重新转为文本；



###  runtime_env 环境变量

**（ 注：位于 configs/envs/runtime_env.yaml 中 ）**

- `RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES`：是否禁用 Ray 对 ASCEND_RT_VISIBLE_DEVICES 的自动设置，'true'为禁用
- `TOKENIZERS_PARALLELISM`：设置tokenizers是否支持并行，'true'为支持
- `NCCL_DEBUG`：NCCL Debug日志级别，VERSION、WARN、INFO、TRACE
- `PYTORCH_NPU_ALLOC_CONF`：设置缓存分配器行为
- `HCCL_CONNECT_TIMEOUT`：HCCL 连接超时时间
- `HCCL_EXEC_TIMEOUT`：HCCL 执行超时时间
- `HCCL_IF_BASE_PORT`：HCCL 通信端口
- `CUDA_DEVICE_MAX_CONNECTIONS`：设备最大连接数
- `HYDRA_FULL_ERROR`：设置 HYDRA 是否输出完整错误日志
- `VLLM_DP_SIZE`：vLLM数据并行度（Data Parallelism）大小，控制数据分片数量，MOE模型建议和EP一致，稠密模型设置为1
- `HCCL_BUFFSIZE`：HCCL通信层单次传输的最大缓冲区大小（单位MB），影响跨设备通信效率
- `VLLM_USE_V1`：使用vLLM的V1 engine API（v1接口），当前只支持 v1 ，需设置为 '1'。
- `VLLM_VERSION`：指定使用的vLLM版本号
- `VLLM_ENABLE_GRAPH_MODE`：启用昇腾torchair图模式优化（1=启用），提升执行效率
- `VLLM_ENABLE_TOPK_OPTIMZE`：使能vLLM TOPK性能优化
- `TASK_QUEUE_ENABLE`：控制开启task_queue算子下发队列优化的等级，推荐设置为 '2' 使能 Level 2 优化。
- `CPU_AFFINITY_CONF`：指定使用绑核优化，推荐设置为 '1'。
- `LCAL_COMM_ID`: 开启coc特性时配套启用，设置为'127.0.0.1:27001'。



## GRPO 训练

### 背景

传统的PPO中需要一个通过广义优势估计（Generalized Advantage Estimation）计算得到的advantage，并依赖于和reward model同结构的需要同步训练的critic model计算得到价值函数(V)

GRPO通过分组采样n个输出，利用组内的平均奖励作为基线计算每个输出在组内的相对奖励，并基于相对奖励计算优势值，从而避免了引入额外的价值网络（critic model）

![img](https://gitee.com/ascend/MindSpeed-RL/raw/master/sources/images/r1_zero/grpo.png)

DeepSeek-R1-Zero的训练过程使用GRPO算法，将ORM（结果奖励模型）替换为基于规则的打分器。

###  配置准备

模型结构的配置文件位于configs/model下，训练配置文件位于configs/目录下，我们以qwen2.5-7b为例[r1_zero_qwen25_7b.yaml]，该配置用到了8卡，为了进一步加速可以不断增加推理DP的数量。以下为参数配置：

```shell
defaults:
  - model:
      - qwen25_7b

megatron_training:
  model: qwen25_7b
  use_fused_rmsnorm: true
  use_mcore_models: true
  sequence_parallel: true
  use_flash_attn: true
  no_masked_softmax_fusion: true
  attention_softmax_in_fp32: true
  no_gradient_accumulation_fusion: true
  use_fused_swiglu: true
  use_fused_rotary_pos_emb: true
  bf16: true
  use_distributed_optimizer: true
  tokenizer_type: PretrainedFromHF
  tokenizer_name_or_path: /root/llmtuner/hfhub/models/Qwen/Qwen2.5-7B
  global_batch_size: 32
  seq_length: 4096
  save_interval: 1000
  train_iters: 200
  stage: ray_grpo
  attention_dropout: 0.0
  init_method_std: 0.01
  hidden_dropout: 0.0
  distributed_backend: nccl
  no_shared_storage: true
  variable_seq_lengths: true
  dataset_additional_keys: ['labels']
  data_path: /root/llmtuner/hfhub/mindspeed/datasets/qwen2.5-math-7b/math500_diffcult7-9_cot/math500_diffcult7-9_cot
  split: 100,0,0
  no_shuffle: false
  full_shuffle_instruction_dataset: false
  seed: 1234

actor_config:
  model: qwen25_7b
  micro_batch_size: 4
  tensor_model_parallel_size: 4
  pipeline_model_parallel_size: 2
  lr: 1e-6
  lr_decay_style: cosine
  min_lr: 1e-7
  weight_decay: 0.01
  lr_warmup_fraction: 0.03
  clip_grad: 1.0
  adam_beta1: 0.9
  adam_beta2: 0.95
  finetune: true
  load: /root/llmtuner/hfhub/mindspeed/models/Qwen/Qwen2.5-7B/mcore_tp4_pp2
  save: /root/llmtuner/hfhub/mindspeed/models/Qwen/Qwen2.5-7B/mcore_tp4_pp2/math500_diffcult7-9_cot
  no_load_optim: true
  no_load_rng: true

rl_config:
  blocking: true # 全共卡情况下应开启blocking。
  guarantee_order: false
  use_integrated_worker: true
  gamma: 1.0
  lam: 0.95
  actor_forward_micro_batch_size: 4
  adv_estimator: group_norm
  kl_penalty: kl
  kl_ctrl_type: fixed
  init_kl_coef: 0.0
  mini_batch_size: 32
  max_prompt_length: 1024
  epochs: 1
  clip_ratio: 0.2
  entropy_coeff: 0.00
  n_samples_per_prompt: 16
  rule_reward: true
  verifier_function: ["acc"]
  verifier_weight: [1.0]
  verifier_parallel: 1
  num_cpus_for_local_task: 1.0
  verifier_timeout: 120
  use_tensorboard: true
  tensorboard_log_dir: ./work_dir/qwen2.5-math-7b_32_4k/tensorboard_log2
  actor_resource:
    num_npus: 8

generate_config:
  trust_remote_code: true
  offload_train_optimizer: true
  offload_train_grad: true
  offload_train_param: true

  # 推理时的并行配置
  infer_tensor_parallel_size: 4
  infer_pipeline_parallel_size: 1
  infer_expert_parallel_size: 1

  # vllm 模型相关设置
  max_num_seqs: 1024
  max_model_len: 4096
  max_num_batched_tokens: 32768
  dtype: "bfloat16"
  gpu_memory_utilization: 0.8

  # 采样配置
  sampling_config:
    logprobs: 1
    max_tokens: 4096
    top_p: 1.0
    top_k: -1
    min_p: 0
    seed: 1234
    temperature: 1.0
    detokenize: false
```

### 训练启动方式

#### 单机

通过 --config-name 传递选取的 config 文件名（不添加.yaml后缀），可以通过下列命令直接启动训练（Qwen25 7B 模型可单机运行）。 目前已支持的配置文件放置在 configs/ 文件夹下。配置文件的具体说明见下文。

以 Qwen25 7B 模型为例,单机启动命令示例如下：

```shell
bash examples/grpo/grpo_trainer_qwen25_7b.sh
```

### 多机

以[ DeepSeekR1-ZERO-Qwen2.5-32B 复现](https://gitee.com/ascend/MindSpeed-RL/blob/master/docs/solutions/r1_zero_qwen25_32b.md) 为例，多机启动步骤如下：

#### 手动启动训练

与基于ray的其他强化训练一样，我们多机需要先在主节点初始化ray：

```shell
# 创建一个集群，端口6344，dashboard端口8260
ray start --head --port 6344 --dashboard-host=0.0.0.0 --dashboard-port=8260
```

随后，在其他节点加入主节点的集群：

```shell
# IP_ADDRESS 处填写主节点 IP 地址
ray start --address="IP_ADDRESS:6344"
```

最后，在主节点上启动训练：

```shell
export HCCL_CONNECT_TIMEOUT=1800
export CUDA_DEVICE_MAX_CONNECTIONS=1

python cli/train_grpo.py --config-name r1_zero_qwen25_7b.yaml | tee logs/r1_zero_qwen25_7b_full.log
```

#### 脚本启动训练

```shell
# 主节点
bash examples/r1/qwen25/r1_zero_qwen25_7b_master.sh r1_zero_qwen25_7b.yaml
# 其余子节点
bash examples/r1/qwen25/r1_zero_qwen25_7b_worker.sh
```
