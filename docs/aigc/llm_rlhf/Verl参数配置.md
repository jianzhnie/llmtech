# Verl 训练配置说明

## FSDP 后端的PPO 训练配置

### 数据部分（Data）

```yaml
data:
  tokenizer: null
  train_files: ~/data/rlhf/gsm8k/train.parquet
  val_files: ~/data/rlhf/gsm8k/test.parquet
  prompt_key: prompt
  max_prompt_length: 512
  max_response_length: 512
  train_batch_size: 1024
  return_raw_input_ids: False
  return_raw_chat: False
  shuffle: True
  filter_overlong_prompts: False
  truncation: error
  image_key: images
  custom_cls:
     path: null
     name: null
```

- `data.train_files`：训练集的 parquet 文件。可以是单个文件或文件列表。所有文件将在训练开始前加载到内存，因此文件总大小不应超过 100GB。支持本地路径或 HDFS 路径。对于 HDFS 路径，系统提供工具将其下载至 DRAM 并转换为本地路径。
- `data.val_files`：Evaluation集的 parquet 文件，支持单文件或文件列表。
- `data.prompt_key`：数据集中提示词所在字段的名称，默认值为 `prompt`。
- `data.max_prompt_length`：最大提示词长度。所有提示词将进行左侧填充至该长度。如果提示词过长将会报错。
- `data.max_response_length`：最大响应长度。在强化学习算法（如 PPO）中生成的响应长度上限。
- `data.train_batch_size`：每次训练迭代采样的 batch 大小。这个参数对应 openrlhf 中的 rollout_batch_size,  每次采样的 Prompt 数量
- `data.return_raw_input_ids`：是否返回未添加对话模板的原始 `input_ids`。当策略模型与奖励模型的模板不同，需设置为 True，并先解码后再加上 RM 模板。
- `data.return_raw_chat`：是否返回原始聊天内容。
- `data.shuffle`：是否在数据加载器中打乱数据顺序。
- `data.filter_overlong_prompts`：是否过滤过长提示词。默认关闭。对于小规模数据集可启用。大规模数据集建议关闭，并设置 `truncation='left'`。
- `data.truncation`：当 `input_ids` 或提示词长度超出最大值时的处理方式。默认值为 `error`。用户可设置为 `left` 或 `right` 实现自动截断。
- `data.image_key`：多模态数据集中图像字段的名称，默认值为 `images`。

### 自定义数据集（Customized Dataset）

```yaml
custom_cls:
  path: null
  name: null
```

- `data.custom_cls.path`：包含自定义数据集类的文件路径。如果未指定，则使用默认实现的数据集类。
- `data.custom_cls.name`：指定文件中的数据集类名。

### Actor / Rollout / Reference 配置

#### 通用设置

````yaml
actor_rollout_ref:
 hybrid_engine: True
 model:
   path: ~/models/deepseek-llm-7b-chat
   external_lib: null
   override_config: { }
   enable_gradient_checkpointing: False
   use_remove_padding: False
````



- `hybrid_engine`：是否启用混合引擎，目前仅支持混合引擎。
- `model.path`：Hugging Face模型路径，可以是本地路径或HDFS路径。对于HDFS路径，我们提供了工具将其下载到DRAM并转换为本地路径。
- `model.external_lib`：需要导入的额外Python包，用于将模型或分词器注册到Hugging Face系统中。
- `model.override_config`：用于覆盖模型原始配置（如 dropout）。
- `model.enable_gradient_checkpointing`：是否开启Actor Model梯度checkpoint。



#### Actor 模型设置

```yaml
actor_rollout_ref:

 actor:
   strategy: fsdp  # This is for backward-compatibility
   ppo_mini_batch_size: 256
   ppo_micro_batch_size: null # will be deprecated, use ppo_micro_batch_size_per_gpu
   ppo_micro_batch_size_per_gpu: 8
   use_dynamic_bsz: False
   ppo_max_token_len_per_gpu: 16384 # n * ${data.max_prompt_length} + ${data.max_response_length}
   grad_clip: 1.0
   clip_ratio: 0.2
   entropy_coeff: 0.001
   use_kl_loss: False # True for GRPO
   use_torch_compile: True # False to disable torch compile
   kl_loss_coef: 0.001 # for grpo
   kl_loss_type: low_var_kl # for grpo
   ppo_epochs: 1
   data_loader_seed: null
   shuffle: False
   ulysses_sequence_parallel_size: 1 # sp size
   optim:
     lr: 1e-6
     lr_warmup_steps: -1 # Prioritized. Negative values mean delegating to lr_warmup_steps_ratio.
     lr_warmup_steps_ratio: 0.  # the total steps will be injected during runtime
     min_lr_ratio: null   # only useful for warmup with cosine
     warmup_style: constant  # select from constant/cosine
     total_training_steps: -1  # must be override by program
   fsdp_config:
     wrap_policy:
       # transformer_layer_cls_to_wrap: None
       min_num_params: 0
     param_offload: False
     optimizer_offload: False
     fsdp_size: -1
   checkpoint:
     contents: ['model', 'optimizer', 'extra']
```

- `strategy`：并行策略，fsdp或megatron，当前为 `fsdp`。
- `ppo_mini_batch_size`：PPO更新时，Rollout 采样的数据被分割成多个子批量，批量大小为`ppo_mini_batch_size`。`ppo_mini_batch_size`是跨所有工作进程/GPU的全局数量。
- `ppo_micro_batch_size`：[将被废弃，使用`ppo_micro_batch_size_per_gpu`] 类似于梯度累积，每个前向传播的`micro_batch_size_per_gpu`，以速度换取GPU内存。该值表示全局视图。
- `ppo_micro_batch_size_per_gpu`：类似于梯度累积，每个前向传播的`micro_batch_size_per_gpu`，以速度换取GPU内存。该值表示每个GPU的局部数量。
- `grad_clip`：Actor 模型的梯度裁剪阈值。
- `clip_ratio`：PPO 中的裁剪比例。
- `entropy_coeff`：计算PPO损失时熵的权重。
- `use_kl_loss`：是否在Actor 中使用KL损失。当启用时，我们不在奖励函数中应用KL。
- `kl_loss_coef`：KL 损失系数，默认为0.001。
- `kl_loss_type`：KL 损失类型，可选项包括 `kl`、`abs`、`mse`、`low_var_kl`、`full`。用于计算Actor 和参考策略之间的KL散度。具体选项请参考`core_algos.py`中的`kl_penalty()`。

- `use_torch_compile`：是否启用 torch compile。

- `ppo_epochs`：对一组采样数据进行PPO更新的轮数。
- `data_loader_seed`: 从torch 2.6.0开始，Megatron后端可能会在cp等级之间生成错误的种子，并导致这些等级之间的数据对齐错误，因此我们需要手动设置种子以避免挂起问题。如果`actor_rollout_ref.actor.shuffle`不为null，则必须设置此值。
- `shuffle`：是否在多个 epoch 中打乱数据顺序。
- `ulysses_sequence_parallel_size`：序列并行大小。

- `optim`：Actor 的优化器参数设置，包括学习率、warmup 等。

- `fsdp_config`：FSDP 相关配置，包括参数 offload、optimizer offload、wrap_policy 等。


- `checkpoint`: Actor 的checkpoint配置。
  - `contents`: checkpoint中保存的内容。默认情况下，我们在checkpoint中保存模型、优化器和额外信息。额外信息包括当前的Rng状态、FSDP支持的lr_scheduler，Megatron的opt_param_scheduler即将推出。我们默认不在checkpoint中存储hf_model，但我们在`scripts/model_merge.py`中提供了一个工具，用于将checkpoint格式转换为hf格式。

#### Reference 模型设置

```yaml
actor_rollout_ref:

  ref:
     fsdp_config:
       param_offload: False
       wrap_policy:
         # transformer_layer_cls_to_wrap: None
         min_num_params: 0
     log_prob_micro_batch_size: null # will be deprecated, use log_prob_micro_batch_size_per_gpu
     log_prob_micro_batch_size_per_gpu: 16
     log_prob_use_dynamic_bsz: ${actor_rollout_ref.actor.use_dynamic_bsz}
     log_prob_max_token_len_per_gpu: ${actor_rollout_ref.actor.ppo_max_token_len_per_gpu}
     ulysses_sequence_parallel_size: ${actor_rollout_ref.actor.ulysses_sequence_parallel_size} #
```



当`actor.use_kl_loss`或`algorithm.use_kl_in_reward`为True时，将启用参考模型。

- `ref`: FSDP配置与Actor 相同。对于大于7B的模型，建议默认为参考模型启用卸载。
- `ref.log_prob_micro_batch_size`: [将被废弃，使用`log_prob_micro_batch_size_per_gpu`] 计算`ref_log_prob`时一个前向传播的批量大小。该值表示全局数量。
- `ref.log_prob_micro_batch_size_per_gpu`: 计算`ref_log_prob`时一个前向传播的批量大小。该值表示每个GPU的局部数量。

#### Rollout 模型设置

```yaml
 actor_rollout_ref:

   ref:
     fsdp_config:
       param_offload: False
       wrap_policy:
         # transformer_layer_cls_to_wrap: None
         min_num_params: 0
     log_prob_micro_batch_size: null # will be deprecated, use log_prob_micro_batch_size_per_gpu
     log_prob_micro_batch_size_per_gpu: 16
     log_prob_use_dynamic_bsz: ${actor_rollout_ref.actor.use_dynamic_bsz}
     log_prob_max_token_len_per_gpu: ${actor_rollout_ref.actor.ppo_max_token_len_per_gpu}
     ulysses_sequence_parallel_size: ${actor_rollout_ref.actor.ulysses_sequence_parallel_size} # sp size
   rollout:
     name: vllm
     temperature: 1.0
     top_k: -1 # 0 for hf rollout, -1 for vllm rollout
     top_p: 1
     prompt_length: ${data.max_prompt_length}  # not use for opensource
     response_length: ${data.max_response_length}
     # for vllm rollout
     dtype: bfloat16 # should align with FSDP
     gpu_memory_utilization: 0.5
     ignore_eos: False
     enforce_eager: True
     free_cache_engine: True
     load_format: dummy_dtensor
     tensor_model_parallel_size: 2
     max_num_batched_tokens: 8192
     max_num_seqs: 1024
     log_prob_micro_batch_size: null # will be deprecated, use log_prob_micro_batch_size_per_gpu
     log_prob_micro_batch_size_per_gpu: 16
     log_prob_use_dynamic_bsz: ${actor_rollout_ref.actor.use_dynamic_bsz}
     log_prob_max_token_len_per_gpu: ${actor_rollout_ref.actor.ppo_max_token_len_per_gpu}
     # for hf rollout
     do_sample: True
     engine_kwargs: # inference engine parameters
       swap_space: null # null means "use the engine default value" (usually 4 GB), setting it to, e.g., 32 means 32 GB
     # number of responses (i.e. num sample times)
     n: 1 # > 1 for grpo, rloo
```



- `rollout.name`：支持 `hf`、`vllm`、`sglang`。

- Rollout（自回归）参数。键应与vLLM的`SamplingParams`中的属性名一致。
  - `temperature`, `top_k`, `top_p`等：采样参数。
  - `dtype`: Rollout模型参数类型。应与FSDP/Megatron后端中的Actor 模型参数类型一致。
  - `gpu_memory_utilization`: 使用vllm时，在其他模型初始化后，分配给kv缓存的GPU内存比例。
  - `tensor_model_parallel_size`: Rollout的TP大小，仅对vllm有效。

- `actor_rollout_ref.ref.log_prob_micro_batch_size`: [将被废弃，使用`log_prob_micro_batch_size_per_gpu`] 重新计算`log_prob`时一个前向传播的批量大小。该值表示全局数量。
- `log_prob_micro_batch_size_per_gpu`: 每个GPU的微批量大小（一个前向传播的批量大小），用于重新计算`log_prob`。该值表示每个GPU的局部数量。
- `do_sample`: 是否进行采样。如果设置为False，则rollout模型将执行贪婪采样。在Evaluation阶段禁用`do_sample`。
- `actor_rollout_ref.rollout.engine_kwargs.swap_space`: 推理引擎使用的交换空间大小（以GB为单位）。- null：表示不设置并使用引擎默认值（例如，vLLM通常为4GB）- 正整数，例如`32`表示32GB。
- `actor_rollout_ref.rollout.ignore_eos`: 是否忽略EOS标记，并在生成EOS标记后继续生成标记。
- `actor_rollout_ref.rollout.free_cache_engine`: 在rollout生成阶段后卸载KV缓存。默认为True。当设置为True时，需要禁用CUDAGraph的使用（将`enforce_eager`设置为True）。
- `actor_rollout_ref.rollout.enforce_eager`: 是否在vLLM生成中使用CUDAGraph。默认设置为True以禁用CUDAGraph。
- `actor_rollout_ref.rollout.load_format`: 用于将Actor 模型权重加载到rollout模型的权重加载器。
  - `auto`: 使用Megatron权重加载器。
  - `megatron`: 使用Megatron权重加载器。与Megatron后端一起部署。输入模型的`state_dict()`已经沿TP维度分割，并且已经沿PP维度聚集。此权重加载器要求rollout模型和Actor 模型的参数形状和名称必须一致。
  - `dtensor`: 使用Hugging Face权重加载器时的默认解决方案。与FSDP后端一起部署，`state_dict_type`为`StateDictType.SHARDED_STATE_DICT`。推荐使用此权重加载器。
  - `hf`: 使用Hugging Face权重加载器。与FSDP后端一起部署，`state_dict_type`为`StateDictType.FULL_STATE_DICT`。此解决方案不需要为vLLM中实现的每个模型重写权重加载器，但会导致更大的峰值内存使用。
  - `dummy_hf`, `dummy_megatron`, `dummy_dtensor`: 随机初始化。

**注意**：在此配置字段中，用户只需从`dummy_megatron`、`dummy_dtensor`、`dummy_hf`中选择用于rollout初始化，我们的混合引擎将在Actor /rollout权重同步期间选择对应的权重加载器（即`megatron`、`dtensor`、`hf`）。

### Critic 模型

Critic 模型配置大致与 Actor 模型相同。

### 奖励模型（Reward Model）

```yaml
reward_model:
  enable: False
  model:
    input_tokenizer: ${actor_rollout_ref.model.path}
    path: ~/models/Anomy-RM-v0.1
    external_lib: ${actor_rollout_ref.model.external_lib}
    fsdp_config:
      min_num_params: 0
      param_offload: False
  micro_batch_size_per_gpu: 16
  max_length: null
  reward_manager: naive
```

- `reward_model.enable`: 是否启用奖励模型。如果为False，则仅使用用户定义的奖励函数计算奖励。在GSM8K和数学示例中，我们禁用了奖励模型。对于使用`full_hh_rlhf`的RLHF对齐示例，我们使用奖励模型来评估响应。如果为False，则以下参数无效。
- `reward_model.model`:
  - `input_tokenizer`: 输入分词器。如果奖励模型的聊天模板与策略不一致，则需要先解码为纯文本，然后应用奖励模型的聊天模板，再使用RM进行评分。如果聊天模板一致，则可以设置为null。
  - `path`: RM的HDFS路径或本地路径。注意，RM仅支持`AutoModelForSequenceClassification`。其他模型类型需要定义自己的`RewardModelWorker`并通过代码传递。
- `reward_model.reward_manager`: 奖励管理器。定义计算基于规则的奖励和处理不同奖励源的机制。默认为`naive`。如果所有验证函数都是多进程安全的，则可以将奖励管理器设置为`prime`以进行并行验证。

### 自定义奖励函数（Customized Reward Function）

```yaml
custom_reward_function:
  path: null
  name: compute_score
```

- `custom_reward_function.path`: 包含自定义奖励函数的文件路径。如果未指定，则使用预实现的奖励函数。
- `custom_reward_function.name`（可选）：指定文件中的奖励函数名称。默认为`compute_score`。



### (Algorithm) 算法配置

```yaml
algorithm:
  gamma: 1.0
  lam: 1.0
  adv_estimator: gae
  use_kl_in_reward: False
  kl_penalty: kl  # 用于估计KL散度的方法
  kl_ctrl:
    type: fixed
    kl_coef: 0.005
    horizon: 10000
    target_kl: 0.1
```

- `gamma`: 折扣因子（Discount Factor）。在强化学习中用于未来奖励的衰减系数，值越小越短视，值越大越看重长期回报。

- `lam`: GAE（广义优势估计）中的权衡参数，用于在偏差和方差之间取得平衡。

- `adv_estimator`: 优势估计器的类型，支持 gae、grpo、reinforce_plus_plus、reinforce_plus_plus_baseline、rloo。默认为 `gae`（广义优势估计）。可选项还包括 `reward`（仅使用奖励）、`none`（不使用优势估计）。

- `use_kl_in_reward`: 是否在奖励中加入KL惩罚项。默认为False。若设置为 True，则 KL 散度的惩罚项将用于奖励计算中。

- `kl_penalty`: 用于估计 KL 散度的方法。可选项包括 `kl`、`abs`、`mse`、`low_var_kl` 和 `full`。具体含义可参考 [`core_algos.py`](https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/core_algos.py) 中的 `kl_penalty()` 实现。

- `kl_ctrl`: 控制KL惩罚项的动态调整方式。
  - `type`: 调整方式类型，目前支持 `fixed`。
  - `kl_coef`: KL惩罚系数（常数类型下为固定值）。
  - `horizon`: 控制KL目标的周期长度。
  - `target_kl`: 目标KL散度。可在某些算法中用于动态调整KL惩罚力度。



### 训练器Trainer配置

```yaml
trainer:
  total_epochs: 30
  project_name: verl_examples
  experiment_name: gsm8k
  logger: ['console', 'wandb']
  log_val_generations: 0
  nnodes: 1
  n_gpus_per_node: 8
  save_freq: -1
  val_before_train: True
  test_freq: 2
  critic_warmup: 0
  default_hdfs_dir: ~/experiments/gsm8k/ppo/${trainer.experiment_name} # HDFScheckpoint路径
  default_local_dir: checkpoints/${trainer.project_name}/${trainer.experiment_name} # 本地checkpoint路径
  resume_mode: auto # 或disable或resume_path，如果设置了resume_from_path，则有效
  resume_from_path: null
  remove_previous_ckpt_in_save: False
  del_local_ckpt_after_load: False
  ray_wait_register_center_timeout: 300
```

- `total_epochs`: 训练的轮数。
- `project_name`: 用于wandb、swanlab、mlflow。
- `experiment_name`: 用于wandb、swanlab、mlflow。
- `logger`: 支持console、wandb、swanlab、mlflow、tensorboard。
- `log_val_generations`: Evaluation期间记录的生成数量（默认为0）。
- `nnodes`: 训练中使用的节点数。
- `n_gpus_per_node`: 每个节点的GPU数量。
- `save_freq`: 保存Actor 和 Critic模型checkpoint的频率（按迭代次数）。
- `val_before_train`: 是否在训练前运行Evaluation。
- `test_freq`: Evaluation的频率（按迭代次数）。
- `critic_warmup`: 在实际策略学习之前训练Critic模型的迭代次数。
- `resume_mode`: 恢复训练的模式。支持disable、auto和resume_path。如果设置为auto（默认），程序将自动从`default_hdfs_dir`中的最新checkpoint恢复。如果设置为resume_path，则程序将从`resume_from_path`中指定的路径恢复。
- `resume_from_path`: 恢复训练的路径。仅在`resume_mode`设置为resume_path时有效。
- `remove_previous_ckpt_in_save`: 是否在保存目录中删除之前的checkpoint，默认为False。
- `del_local_ckpt_after_load`: 是否在加载后删除本地checkpoint，默认为False。
- `ray_wait_register_center_timeout`: 等待ray注册中心准备就绪的超时时间，默认为300秒。



## 评估配置（evaluation.yaml）

### 数据配置

```yaml
data:
  path: /tmp/math_Qwen2-7B-Instruct.parquet
  prompt_key: prompt
  response_key: responses
  data_source_key: data_source
  reward_model_key: reward_model
```

- `data.path`: 数据集文件路径（Parquet格式）。
- `data.prompt_key`: 数据集中提示字段的名称，默认为`prompt`。
- `data.response_key`: 包含生成响应的键。这应该是一个字符串列表，表示响应。默认为`responses`。
- `data.data_source_key`: 用于分离不同数据源的指标计算，确保每个数据源的指标独立计算。
- `data.reward_model_key`: 包含参考答案的键。这些参考答案通常作为任务的基准或测试用例。

### 自定义奖励函数

```yaml
custom_reward_function:
  path: null
  name: compute_score
```

- `custom_reward_function.path`: 包含自定义奖励函数的文件路径。如果未指定，则使用预实现的奖励函数。
- `custom_reward_function.name`（可选）：指定文件中的奖励函数名称。默认为`compute_score`。



## SFT训练配置（FSDP后端）

```yaml
optim:
  lr: 1e-5
  weight_decay: 0.01
  warmup_steps_ratio: 0.1
  clip_grad: 1.0
  lr_scheduler: cosine
```

- `optim.lr`: 优化器的学习率。
- `optim.weight_decay`: 优化器的权重衰减。
- `optim.warmup_steps_ratio`: 预热步骤与总训练步骤的比例。
- `optim.clip_grad`: 梯度裁剪值。
- `optim.lr_scheduler`: 学习率调度器类型。选项包括：
  - `cosine`: 带有预热的余弦学习率调度器（默认）。
  - `wsd`: 预热-稳定-衰减调度器，在预热和衰减阶段之间提供一个稳定的

------

## 其他补充说明

- 所有路径字段（如模型路径、数据路径）均支持本地路径和HDFS路径。对于HDFS路径，框架提供工具支持下载到内存并自动转换为本地路径。
- 大部分配置项使用默认值即可满足常规需求，若需要更细致的控制，可以参考源代码中各配置项的具体使用方式。
- 在多卡大模型训练场景中，合理设置 `fsdp_config` 与 `offload` 选项有助于优化显存使用与训练性能。
- 所有带有 `*_micro_batch_size` 和 `*_per_gpu` 后缀的字段建议设置为每张GPU上可承载的最小batch size，以优化显存与吞吐。
