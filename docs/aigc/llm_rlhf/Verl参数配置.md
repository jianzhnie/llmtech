# 配置说明

## FSDP 后端的 `ppo_trainer.yaml` 配置说明

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
- `data.val_files`：验证集的 parquet 文件，支持单文件或文件列表。
- `data.prompt_key`：数据集中提示词所在字段的名称，默认值为 `prompt`。
- `data.max_prompt_length`：最大提示词长度。所有提示词将进行左侧填充至该长度。如果提示词过长将会报错。
- `data.max_response_length`：最大响应长度。在强化学习算法（如 PPO）中生成的响应长度上限。
- `data.train_batch_size`：每次训练迭代采样的 batch 大小。
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

### Actor / Rollout / Reference 策略模型配置

```yaml
actor_rollout_ref:
  hybrid_engine: True
  model:
    path: ~/models/deepseek-llm-7b-chat
    external_lib: null
    override_config: { }
    enable_gradient_checkpointing: False
    use_remove_padding: False
  ...
```

#### 通用设置

- `hybrid_engine`：是否启用混合引擎，目前仅支持混合引擎。
- `model.path`：模型路径，可以是本地路径或 HDFS 路径。
- `model.external_lib`：需要导入的额外 Python 包，用于注册模型或 tokenizer。
- `model.override_config`：用于覆盖模型原始配置（如 dropout）。
- `model.enable_gradient_checkpointing`：是否开启梯度检查点。

#### Actor 模型设置

- `strategy`：并行策略，当前为 `fsdp`。
- `ppo_mini_batch_size`：PPO 的 mini batch 大小（全局）。
- `ppo_micro_batch_size`：将被弃用，请使用 `ppo_micro_batch_size_per_gpu`。
- `ppo_micro_batch_size_per_gpu`：每个 GPU 的微 batch 大小（本地）。
- `grad_clip`：梯度裁剪阈值。
- `clip_ratio`：PPO 中的裁剪比例。
- `entropy_coeff`：熵损失的权重。
- `use_kl_loss`：是否在 Actor 中使用 KL 损失。
- `kl_loss_coef`：KL 损失系数。
- `kl_loss_type`：KL 损失类型，可选项包括 `kl`、`abs`、`mse`、`low_var_kl`、`full`。
- `use_torch_compile`：是否启用 torch compile。
- `ppo_epochs`：PPO 每组样本更新轮数。
- `shuffle`：是否在多个 epoch 中打乱数据顺序。
- `ulysses_sequence_parallel_size`：序列并行大小。
- `optim`：优化器参数设置，包括学习率、warmup 等。
- `fsdp_config`：FSDP 相关配置，包括参数 offload、optimizer offload、wrap_policy 等。
- `checkpoint.contents`：checkpoint 中保存的内容，包括模型、优化器及其他附加信息。

#### Reference 模型设置

参考模型在以下任一设置为 True 时启用：

- `actor.use_kl_loss`
- `algorithm.use_kl_in_reward`

相关参数与 Actor 相似，推荐对 7B 以上模型默认启用 offload。

#### Rollout 模型设置

- `rollout.name`：支持 `hf`、`vllm`、`sglang`。
- `temperature`, `top_k`, `top_p`：采样相关参数。
- `dtype`：参数精度类型，需与 FSDP 保持一致。
- `gpu_memory_utilization`：分配给 KV 缓存的 GPU 内存比例。
- `tensor_model_parallel_size`：Rollout 模型的张量并行大小。
- `log_prob_micro_batch_size_per_gpu`：用于 log_prob 计算的微 batch 大小。
- `do_sample`：是否开启采样，若为 False 则为贪婪解码。
- `ignore_eos`：是否忽略 EOS token，继续生成。
- `free_cache_engine`：Rollout 结束后是否卸载 KVCache。
- `enforce_eager`：是否强制使用 Eager 模式（禁用 CUDAGraph）。
- `load_format`：Rollout 模型加载权重时使用的 loader 类型，可选：
  - `auto`：自动选择；
  - `megatron`：用于 Megatron 权重；
  - `dtensor`：推荐用于 FSDP 的加载方式；
  - `hf`：Huggingface 权重；
  - `dummy_*`：随机初始化。

**注意事项**：用户只需选择 `dummy_megatron`、`dummy_dtensor`、`dummy_hf` 用于初始化，实际加载时将自动选用对应 loader。

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

- `enable`：是否启用奖励模型。若为 False，则使用自定义函数计算 reward。
- `input_tokenizer`：奖励模型的 tokenizer。若模板不同，需要先解码后重新编码。
- `path`：奖励模型路径。只支持 `AutoModelForSequenceClassification`。
- `reward_manager`：奖励管理机制。默认 `naive`。若验证函数可并行处理，可设为 `prime`。

### 自定义奖励函数（Customized Reward Function）

```yaml
custom_reward_function:
  path: null
  name: compute_score
```

- `path`：包含自定义奖励函数的文件路径。
- `name`：函数名，默认 `compute_score`。

### 算法参数（Algorithm）

```yaml
algorithm:
  gamma: 1.0
  lam: 1.0
  adv_estimator: gae
  use_kl_in_reward: False
  kl_penalty: kl
  kl_ctrl:
    type: fixed
    kl_coef: 0.005
    horizon: 10000
    target_kl: 0.1
```

- `gamma`：折扣因子。
- `lam`：bias 与 variance 的折中。
- `adv_estimator`：优势函数估计方法，使用 GAE。
- `use_kl_in_reward`：是否在奖励函数中加入 KL 惩罚。
- `kl_penalty`：KL 惩罚的计算方式。
- `kl_ctrl`：KL 控制参数配置。



## 算法配置说明

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
- `adv_estimator`: 优势估计器的类型，默认为 `gae`（广义优势估计）。可选项还包括 `reward`（仅使用奖励）、`none`（不使用优势估计）。
- `use_kl_in_reward`: 是否在奖励中加入KL惩罚项。若设置为 True，则 KL 散度的惩罚项将用于奖励计算中。
- `kl_penalty`: 用于估计 KL 散度的方法。可选项包括 `kl`、`abs`、`mse`、`low_var_kl` 和 `full`。具体含义可参考 [`core_algos.py`](https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/core_algos.py) 中的 `kl_penalty()` 实现。
- `kl_ctrl`: 控制KL惩罚项的动态调整方式。
  - `type`: 调整方式类型，目前支持 `fixed`。
  - `kl_coef`: KL惩罚系数（常数类型下为固定值）。
  - `horizon`: 控制KL目标的周期长度。
  - `target_kl`: 目标KL散度。可在某些算法中用于动态调整KL惩罚力度。

------

## 日志与调试配置

```yaml
log:
  wandb:
    enable: False
    project: verl
    group: test_group
    name: test_name
    entity: null
  log_dir: ./logs
  log_interval: 1
  save_interval: 1
  eval_interval: 5
  eval_iters: 1
  enable_tb: False
```

- `log.wandb`: Weights & Biases 相关配置，用于实验可视化与追踪。
  - `enable`: 是否启用wandb。
  - `project`: 项目名称。
  - `group`: 实验组名称，用于组织多个实验。
  - `name`: 当前运行的名称。
  - `entity`: 团队或用户名称。
- `log_dir`: 日志输出路径。
- `log_interval`: 日志打印间隔（以训练步数为单位）。
- `save_interval`: 模型保存间隔。
- `eval_interval`: 模型评估的间隔周期。
- `eval_iters`: 每次评估运行多少次迭代。
- `enable_tb`: 是否启用 TensorBoard 日志。

------

## 分布式与系统配置

```yaml
dist:
  backend: nccl
  port: 12355
  master_addr: localhost
  deepspeed_config: null
```

- `backend`: 分布式通信后端。一般推荐使用 `nccl`。
- `port`: 主节点通信端口。
- `master_addr`: 主节点地址。
- `deepspeed_config`: 如果使用 DeepSpeed，可以在此指定其配置文件路径。

------

## 训练运行设置

```yaml
run:
  seed: 42
  max_iterations: 100
  run_name: test_run
  resume_from_checkpoint: null
```

- `seed`: 随机数种子，用于保证结果可复现。
- `max_iterations`: 最大训练迭代次数。
- `run_name`: 本次运行的名称。
- `resume_from_checkpoint`: 从已有的检查点恢复训练的路径（若为空，则从头开始）。

------

## 其他补充说明

- 所有路径字段（如模型路径、数据路径）均支持本地路径和HDFS路径。对于HDFS路径，框架提供工具支持下载到内存并自动转换为本地路径。
- 大部分配置项使用默认值即可满足常规需求，若需要更细致的控制，可以参考源代码中各配置项的具体使用方式。
- 在多卡大模型训练场景中，合理设置 `fsdp_config` 与 `offload` 选项有助于优化显存使用与训练性能。
- 所有带有 `*_micro_batch_size` 和 `*_per_gpu` 后缀的字段建议设置为每张GPU上可承载的最小batch size，以优化显存与吞吐。

------

如果你需要我将此配置结构制作为可视化文档、图表或者生成 YAML 模板文件，请告诉我，我可以进一步帮你处理！
