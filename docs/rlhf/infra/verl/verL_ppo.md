# PPO 示例架构详解

本文将深入解析 `verl` 框架中近端策略优化（Proximal Policy Optimization, PPO）算法的实现架构。PPO 是当前大型语言模型（LLM）后训练阶段最广泛使用的强化学习算法之一。本教程以主入口文件 [main_ppo.py](https://github.com/volcengine/verl/blob/main/verl/trainer/main_ppo.py) 为核心，详细阐述其代码设计与实现逻辑。

## 数据定义

用户需预先对数据集进行处理，并将其存储为 Parquet 格式文件。`verl` 框架提供了 `RLHFDataset` 类，用于加载和分词这些 Parquet 文件。

对于 `RLHFDataset`（默认配置），数据文件至少需包含以下字段：

- `prompt`：以字符串形式存储的提示词（prompt）

我们已在 [data_preprocess 目录](https://github.com/volcengine/verl/blob/main/examples/data_preprocess) 中提供了将原始数据集转换为 Parquet 文件的示例脚本。目前支持 GSM8k、MATH、Hellaswag 和 full_hh_rlhf 等数据集的预处理流程。详细步骤请参阅 [后训练数据准备](https://verl.readthedocs.io/en/latest/preparation/prepare_data.html) 文档。

## 奖励函数的定义

在 PPO 主入口中，用户需根据训练所用的数据集或具体应用场景，自定义奖励函数。

例如，框架已在 `_select_rm_score_fn` 函数中为 [GSM8k](https://github.com/volcengine/verl/blob/main/verl/utils/reward_score/gsm8k.py) 和 [MATH](https://github.com/volcengine/verl/blob/main/verl/utils/reward_score/math.py) 数据集实现了相应的奖励函数。在 `RewardManager` 组件中，系统会依据数据源自动选择匹配的奖励函数来计算奖励分数。

对于某些 RLHF 数据集（如 `full_hh_rlhf`），可直接使用预训练的RewardModel（Reward Model, RM）对生成的响应进行评估，无需额外定义奖励函数。在此情况下，`RewardManager` 将直接返回RewardModel计算出的 `rm_score`。

更多关于奖励函数的具体实现，请参见 [奖励函数实现目录](https://github.com/volcengine/verl/blob/main/verl/utils/reward_score)。

## Worker Classes 的定义

```python
if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}: # 使用 FSDP 后端
    assert config.critic.strategy in {"fsdp", "fsdp2"}
    from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
    from verl.single_controller.ray import RayWorkerGroup
    ray_worker_group_cls = RayWorkerGroup

elif config.actor_rollout_ref.actor.strategy == 'megatron': # 使用 Megatron 后端
    assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
    from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
    from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
    ray_worker_group_cls = NVMegatronRayWorkerGroup # 适配 Megatron-LM 的 Ray Worker Class

else:
    raise NotImplementedError

from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

role_worker_mapping = {
    Role.ActorRollout: ActorRolloutRefWorker,
    Role.Critic: CriticWorker,
    Role.RefPolicy: ActorRolloutRefWorker
}

global_pool_id = 'global_pool'
resource_pool_spec = {
    global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
}
mapping = {
    Role.ActorRollout: global_pool_id,
    Role.Critic: global_pool_id,
    Role.RefPolicy: global_pool_id,
}
```

### 步骤 1：构建角色与 Worker Classes 的映射关系

在 `verl` 中，“角色”（Role）用于表示同一进程中的一组 workers。框架已在 [ray_trainer.py](https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/ray_trainer.py#L38) 中预定义了若干标准角色：

```python
class Role(Enum):
    """
    用户可通过继承 Role 类并添加新成员来动态创建更多角色
    """
    Actor = 0           # 仅包含Actor（Actor）的 worker
    Rollout = 1         # 仅包含Rollout（Rollout）的 worker
    ActorRollout = 2    # 同时包含Actor和Rollout的 混合引擎
    Critic = 3          # 仅包含Critic（Critic）的 worker
    RefPolicy = 4       # 仅包含参考策略（Reference Policy）的 worker
    RewardModel = 5     # 仅包含RewardModel（Reward Model）的 worker
    ActorRolloutRef = 6 # 同时包含Actor、Rollout和参考策略的worker
```

### 步骤 2：定义角色对应的Worker Class

- 框架已预实现 `ActorRolloutRefWorker` 类。通过不同的配置参数，该类可作为独立的Actor（Actor）、独立的Rollout（Rollout）、ActorRollout 混合引擎，或同时包含Actor、Rollout和参考策略的 ActorRolloutRef 混合引擎运行。
- 同时，框架也提供了针对 `Actor`、`Rollout`、`Critic`、`Reward Model` 和 `Reference model` 的WorkerGroup实现，支持 PyTorch FSDP 和 Megatron-LM 两种后端。具体实现详见 [FSDP WorkerGroup](https://github.com/volcengine/verl/blob/main/verl/workers/fsdp_workers.py) 与 [Megatron-LM WorkerGroup](https://github.com/volcengine/verl/blob/main/verl/workers/megatron_workers.py)。

### 步骤 3：定义资源池 ID 与资源池规格

- **资源池**（Resource Pool）是对全局 GPU 资源的逻辑划分。`resource_pool_spec` 是一个字典，用于将资源池 ID 映射到具体的 GPU 数量。
  - 在上述示例中，我们定义了一个名为 `global_pool_id` 的全局资源池，并将所有角色均部署于此。这意味着所有模型共享本次训练任务中的全部 GPU 资源，属于典型的 *协同部署*（co-located deployment）方案。
- 更高级的资源池配置与部署策略，请参阅相关文档。

## RewardModel与奖励函数的配置

```python
# 我们应在此处采用多源奖励机制：
# - 对于基于规则的奖励，直接调用奖励评分函数
# - 对于基于模型的奖励，调用RewardModel
# - 对于涉及代码生成的任务，若存在测试用例，则发送至沙箱执行
# - 最终，将所有奖励信号进行融合
# - 奖励类型取决于数据标签
if config.reward_model.enable:
    from verl.workers.fsdp_workers import RewardModelWorker
    role_worker_mapping[Role.RewardModel] = RewardModelWorker
    mapping[Role.RewardModel] = global_pool_id

reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0)

# 注意：验证阶段始终使用基于函数的奖励机制
val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1)

resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
```

由于并非所有任务都依赖基于模型的奖励机制（RM），用户需明确声明使用模型化 RM 还是函数化 RM：

- **若使用模型化 RM**：需在资源映射中添加 `RewardModel` 角色，并将其关联至资源池。
  - 需注意，预定义的 `RewardModelWorker` 仅支持 HuggingFace 格式的模型（即 `AutoModelForSequenceClassification`）。若使用其他架构的模型，用户需在 [FSDP WorkerGroup](https://github.com/volcengine/verl/blob/main/verl/workers/fsdp_workers.py) 或 [Megatron-LM WorkerGroup](https://github.com/volcengine/verl/blob/main/verl/workers/megatron_workers.py) 中自定义 `RewardModelWorker`。
- **若使用基于函数的奖励机制**：用户需为每个数据集类别定义相应的奖励函数。

```python
def _select_rm_score_fn(data_source):
    if data_source == 'openai/gsm8k':
        return gsm8k.compute_score
    elif data_source == 'lighteval/MATH':
        return math.compute_score
    else:
        raise NotImplementedError
```

更多关于奖励函数的实现细节，请参见 [奖励函数目录](https://github.com/volcengine/verl/blob/main/verl/utils/reward_score/)。

## PPO Trainer的定义、初始化与运行

```python
trainer = RayPPOTrainer(config=config,
                        tokenizer=tokenizer,
                        role_worker_mapping=role_worker_mapping,
                        resource_pool_manager=resource_pool_manager,
                        ray_worker_group_cls=ray_worker_group_cls,
                        reward_fn=reward_fn,
                        val_reward_fn=val_reward_fn)
trainer.init_workers()
trainer.fit()
```

- 首先，使用用户配置、分词器、Worker Class映射、资源池管理器、工作组类以及奖励函数初始化 `RayPPOTrainer`。
- 调用 `trainer.init_workers()` 方法，在分配的 GPU 资源（位于资源池中）上完成各模型的初始化。
- 实际的 PPO 训练过程在 `trainer.fit()` 方法中执行。

通过复用 Ray 模型worker、资源池管理和奖励函数机制，`verl` 框架能够轻松扩展至其他强化学习算法。更多扩展应用示例，请参阅 [扩展模块文档](https://verl.readthedocs.io/en/latest/advance/dpo_extension.html)。

# RayPPOTrainer

我们实现了 `RayPPOTrainer`，这是一个运行在单个 CPU/GPU 节点（默认情况下为 CPU）上的Trainer，用于执行近端策略优化（Proximal Policy Optimization, PPO）算法。

`RayPPOTrainer` 主要包含三大核心功能：数据准备、WorkerGroup 初始化和 PPO 训练循环。

## 数据准备

作为单一进程的 `RayPPOTrainer`，它负责从指定的数据集中加载完整批次的样本（即提示词），然后将这些提示词分发到运行在不同 GPU 上的工作组中。

为了简化数据加载流程，我们设计了 `RLHFDataset` 类来处理预处理后的 Parquet 文件。该类能够对提示词应用聊天模板、添加填充、截断过长的提示词，并进行 tokenize 处理：

```python
self.train_dataset = RLHFDataset(data_files=self.config.data.train_files,
                                  tokenizer=self.tokenizer,
                                  config=self.config.data)
```

随后，数据加载器会根据 PPO 的小批量尺寸遍历整个数据集。

## WorkerGroup 初始化

首先介绍如何在指定的 GPU 组上初始化Actor模型的 `WorkerGroup`。

```python
# max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
# For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
# For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes,
                                use_gpu=True,
                                max_colocate_count=1)
# define actor rollout cls to be init on remote
actor_rollout_cls = RayClassWithInitArgs(cls=ActorRolloutWorker)
# define actor_rollout worker group
actor_rollout_worker_group = MegatronRayWorkerGroup(resource_pool=resource_pool,
                                                    ray_cls_with_init=actor_rollout_cls,
                                                    default_megatron_kwargs=config.actor_rollout.megatron)
```

不同的 `WorkerGroup`（例如 `actor_rollout_worker_group`、`critic_worker_group` 和 `ref_worker_group`）在实现时分别位于独立的进程中。

驱动程序通过调用如 `actor_rollout_worker_group` 等角色内的分布式计算函数来构建强化学习训练循环。

对于在同一 GPU 组中共置的模型，我们提供了细粒度的优化方案，即将不同角色的 `worker_group` 合并到同一个进程中。此优化可以减少跨进程间的冗余 CUDA/分布式上下文开销。

> 初始化 `WorkerGroup` 的步骤如下所示。注意，如果您希望为每个角色使用不同的资源池以支持不同的并行大小，应直接为不同的 `worker groups` 分配不同的资源池，而不是使用 `create_colocated_worker_cls` 方法。

```python
# initialize WorkerGroup
# NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
# you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
# See TODO(url) for more information.
all_wg = {}
for resource_pool, class_dict in self.resource_pool_to_cls.items():
    worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
    wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
    spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
    all_wg.update(spawn_wg)

if self.use_critic:
    self.critic_wg = all_wg['critic']
    self.critic_wg.init_model()

if self.use_reference_policy:
    self.ref_policy_wg = all_wg['ref']
    self.ref_policy_wg.init_model()

if self.use_rm:
    self.rm_wg = all_wg['rm']
    self.rm_wg.init_model()

# we should create rollout at the end so that vllm can have a better estimation of kv cache memory
self.actor_rollout_wg = all_wg['actor_rollout']
self.actor_rollout_wg.init_model()
```

对于 Megatron 后端，如果我们将不同的 `worker_groups` 合并到同一个进程中，所有角色将会使用相同的3D并行维度。这意味着在同一分布式上下文中，可能需要为每个角色维护多个3D进程组以优化性能。如果您希望不同角色使用不同的3D并行尺寸，请参考首个代码块的架构来分别初始化各个角色的 `worker_group`。

具体来说：

当您选择将多个 `worker_group`（例如 actor、critic 和 reference policy）合并到同一进程中时，所有这些角色都将共享相同的3D并行配置（包括张量模型并行、管道模型并行以及数据并行）。这在某些情况下可能导致资源利用效率低下或训练性能下降。为了针对这种情况进行优化，您可以采取以下措施：

1. **维持多组3D进程组**：在同一分布式环境中，为每一个角色维护独立的3D进程组。这样，虽然这些角色运行在同一进程中，但它们可以拥有各自的并行策略，从而更灵活地适应不同的计算需求和数据处理要求。

2. **单独初始化各角色的 `worker_group`**：如果您的应用场景需要为不同角色设置不同的3D并行尺寸，您应当参照初始代码块中的架构，分别为每个角色创建并初始化其专属的 `worker_group`。这样做可以让每个角色根据自身的需要调整并行策略，而不是被迫接受统一的配置。

下面是一个简化的示例，展示了如何为不同角色初始化具有不同3D并行尺寸的 `worker_group`：

```python
# 假设我们有两个角色：actor 和 critic，并且它们需要不同的3D并行尺寸

# Actor 的资源配置
resource_pool_actor = RayResourcePool(process_on_nodes=[config.actor.n_gpus_per_node] * config.actor.nnodes,
                                      use_gpu=True,
                                      max_colocate_count=1)

actor_cls = RayClassWithInitArgs(cls=ActorRolloutWorker,
                                 default_megatron_kwargs=config.actor.megatron)  # 自定义 Megatron 关键字参数

actor_worker_group = MegatronRayWorkerGroup(resource_pool=resource_pool_actor,
                                            ray_cls_with_init=actor_cls)

# Critic 的资源配置
resource_pool_critic = RayResourcePool(process_on_nodes=[config.critic.n_gpus_per_node] * config.critic.nnodes,
                                       use_gpu=True,
                                       max_colocate_count=1)

critic_cls = RayClassWithInitArgs(cls=CriticWorker,
                                  default_megatron_kwargs=config.critic.megatron)  # 根据需要调整 Megatron 关键字参数

critic_worker_group = MegatronRayWorkerGroup(resource_pool=resource_pool_critic,
                                             ray_cls_with_init=critic_cls)
```

通过这种方式，您可以确保每个角色都能根据其特定的需求获得最适合的3D并行配置，从而提高整体训练效率和模型性能。

## PPO 训练循环

通过调用各个角色 `worker_group` 中的方法来实现 PPO 训练循环。每个方法的输入输出数据均为 `protocol.py` 中定义的 `DataProto` 对象。在训练过程中，Trainer将按照封装在工作函数中的传输协议在不同 GPU 之间调度或收集数据。PPO 微批处理的计算过程则是在 `update_actor` 和 `update_critic` 函数中完成的。

```python
def fit(self):
    """
    The training loop of PPO.
    The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
    The light-weight advantage computation is done on the driver process.
    """
    from verl.utils.tracking import Tracking
    from omegaconf import OmegaConf

    logger = Tracking(project_name=self.config.trainer.project_name,
                        experiment_name=self.config.trainer.experiment_name,
                        default_backend=self.config.trainer.logger,
                        config=OmegaConf.to_container(self.config, resolve=True))

    global_steps = 0

    # perform validation before training
    # currently, we only support validation using the reward_function.
    if self.val_reward_fn is not None:
        val_metrics = self._validate()
        pprint(f'Initial validation metrics: {val_metrics}')

    for epoch in range(self.config.trainer.total_epochs):
        for batch_dict in self.train_dataloader:
            metrics = {}

            batch: DataProto = DataProto.from_single_dict(batch_dict)
            # batch = batch.to('cuda')

            # pop those keys for generation
            gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])

            # generate a batch
            with Timer(name='gen', logger=None) as timer:
                gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
            metrics['timing/gen'] = timer.last

            batch = batch.union(gen_batch_output)

            if self.use_reference_policy:
                # compute reference log_prob
                with Timer(name='ref', logger=None) as timer:
                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                    batch = batch.union(ref_log_prob)
                metrics['timing/ref'] = timer.last

            # compute values
            with Timer(name='values', logger=None) as timer:
                values = self.critic_wg.compute_values(batch)
                batch = batch.union(values)
            metrics['timing/values'] = timer.last

            with Timer(name='adv', logger=None) as timer:
                # compute scores. Support both model and function-based.
                # We first compute the scores using reward model. Then, we call reward_fn to combine
                # the results from reward model and rule-based results.
                if self.use_rm:
                    # we first compute reward model score
                    reward_tensor = self.rm_wg.compute_rm_score(batch)
                    batch = batch.union(reward_tensor)

                # we combine with rule-based rm
                reward_tensor = self.reward_fn(batch)
                batch.batch['token_level_scores'] = reward_tensor

                # compute rewards. apply_kl_penalty if available
                batch, kl_metrics = apply_kl_penalty(batch,
                                                        kl_ctrl=self.kl_ctrl_in_reward,
                                                        kl_penalty=self.config.algorithm.kl_penalty)
                metrics.update(kl_metrics)

                # compute advantages, executed on the driver process
                batch = compute_advantage(batch,
                                            self.config.algorithm.gamma,
                                            self.config.algorithm.lam,
                                            adv_estimator=self.config.algorithm.adv_estimator)
            metrics['timing/adv'] = timer.last

            # update critic
            if self.use_critic:
                with Timer(name='update_critic', logger=None) as timer:
                    critic_output = self.critic_wg.update_critic(batch)
                metrics['timing/update_critic'] = timer.last
                critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                metrics.update(critic_output_metrics)

            # implement critic warmup
            if self.config.trainer.critic_warmup <= global_steps:
                # update actor
                with Timer(name='update_actor', logger=None) as timer:
                    actor_output = self.actor_rollout_wg.update_actor(batch)
                metrics['timing/update_actor'] = timer.last
                actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                metrics.update(actor_output_metrics)

            # validate
            if self.val_reward_fn is not None and (global_steps + 1) % self.config.trainer.test_freq == 0:
                with Timer(name='testing', logger=None) as timer:
                    val_metrics: dict = self._validate()
                    val_metrics = {f'val/{key}': val for key, val in val_metrics.items()}
                metrics['timing/testing'] = timer.last
                metrics.update(val_metrics)

            # collect metrics
            data_metrics = compute_data_metrics(batch=batch)
            metrics.update(data_metrics)

            # TODO: make a canonical logger that supports various backend
            logger.log(data=metrics, step=global_steps)

            if self.config.trainer.save_freq > 0 and (global_steps + 1) % self.config.trainer.save_freq == 0:
                actor_local_path = os.path.join(self.config.trainer.default_local_dir, 'actor',
                                                f'global_step_{global_steps}')
                actor_remote_path = os.path.join(self.config.trainer.default_hdfs_dir, 'actor')
                self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path)

                if self.use_critic:
                    critic_local_path = os.path.join(self.config.trainer.default_local_dir, 'critic',
                                                        f'global_step_{global_steps}')
                    critic_remote_path = os.path.join(self.config.trainer.default_hdfs_dir, 'critic')
                    self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path)

            global_steps += 1

    # perform validation after training
    if self.val_reward_fn is not None:
        val_metrics = self._validate()
        pprint(f'Final validation metrics: {val_metrics}')
```
