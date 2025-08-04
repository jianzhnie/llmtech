# Megatron-LM 后端技术文档

## 概述

为支持 Megatron-LM 作为底层训练与推理引擎，Verl 实现了面向Actor（Actor）、Critic（Critic）、Reference（Reference）、Rollout（Rollout）及RewardModel（Reward Model）的多种Worker（Worker）。在此基础上，通过集成 Megatron-LM 与高性能推理框架 vLLM/SGLang，我们在 [`megatron_vllm.py`](https://github.com/volcengine/verl/blob/main/verl/workers/sharding_manager/megatron_vllm.py) 与 [`megatron_sglang.py`](https://github.com/volcengine/verl/blob/main/verl/workers/sharding_manager/megatron_sglang.py) 中实现了 **3D混合引擎（3DHybridEngine）**。

该架构全面支持 **5D 并行策略**，包括：

- 张量并行（Tensor Parallelism, TP）
- 专家并行（Expert Parallelism, EP）
- 上下文并行（Context Parallelism, CP）
- 数据并行（Data Parallelism, DP）
- 流水线并行（Pipeline Parallelism, PP）

同时兼容序列并行（Sequence Parallelism），旨在实现极致的模型可扩展性与系统吞吐量。3D混合引擎通过统一的权重管理机制，显著降低训练过程中的峰值内存占用，并有效减少Actor与Rollout之间的权重同步开销。

> **核心优势**：3D混合引擎可显著降低峰值内存使用，减少Actor与Rollout间的权重同步开销。

---

## Megatron Worker组件

### 基础Worker：`MegatronWorker`

`MegatronWorker` 是所有 Megatron Worker的基类。该类通过以下两个函数获取当前 GPU 上Worker的三维并行配置信息：

- `get_megatron_global_info()`：获取全局并行组大小。
- `get_megatron_rank_info()`：获取当前进程在各并行维度的秩（rank）。

这些信息用于构建 Megatron 后端的通信与数据分发协议。

后续针对不同模型角色（如 Actor、Critic、Rollout 等）的Worker类将继承该基类，并用于构建 `WorkerGroup`。

每个Worker类通过 `@register(dispatch_mode=...)` 装饰器注册可被 Ray 驱动进程调用的 API 接口。数据的分发与收集策略由 `dispatch_mode` 决定，具体协议定义详见 [decorator.py](https://github.com/volcengine/verl/blob/main/verl/single_controller/base/decorator.py)。

> **说明**：该基类主要服务于Actor/推演混合引擎或Reference，负责模型初始化与核心计算逻辑的执行。



### ActorWorker：`MegatronPPOActor`

`MegatronPPOActor` 实现了基于 Megatron 构建模型时的 PPO 算法核心计算逻辑，包括对数概率（log probability）计算与模型参数更新。

#### Actor/Rollout 混合引擎

##### 1. 模型初始化接口

```python
@register(dispatch_mode=Dispatch.ONE_TO_ALL)
def init_model(self):
```

- `Dispatch.ONE_TO_ALL`：驱动进程调用此接口时，每个Worker（GPU）将独立执行模型初始化流程。

初始化流程如下：

1. **`MegatronVLLMShardingManager`**：作为上下文管理器，负责在Actor（Actor）与Rollout（Rollout）之间执行权重的重分片（resharding）。
2. **`vLLMRollout`**：支持基于 vLLM 的高效生成。我们对 vLLM 引擎进行了修改，使其支持 SPMD（Single Program, Multiple Data）模式，以适配 Verl 的 `WorkerGroup` 架构。
3. 初始化示例代码：

```python
# 构建Actor模型
self.actor = MegatronPPOActor(
    config=self.config.actor,
    model_config=self.actor_model_config,
    megatron_config=megatron_config,
    actor_module=self.actor_module,
    actor_optimizer=self.actor_optimizer,
    actor_optimizer_config=self.actor_optim_config
)

# 构建Rollout
rollout = vLLMRollout(
    actor_module=params,
    config=self.config.rollout,
    tokenizer=self.tokenizer,
    model_hf_config=self.actor_model_config,
    train_tp=mpu.get_tensor_model_parallel_world_size()
)

# 执行权重重分片
sharding_manager = MegatronVLLMShardingManager(
    module=self.hybrid_engine,
    inference_engine=rollout.inference_engine,
    model_config=self.actor_model_config,
    layer_name_mapping=layer_name_mapping
)
```

> **注意**：在混合引擎中，Actor模型的 PP 维度被视为 DP 维度。因此，驱动进程将根据此逻辑重组数据分发规则。由于Actor通常采用更大规模的 3D 并行配置，其权重在 TP 和 PP 维度上聚合，而Rollout则需在 DP 组内进行数据分发与收集。相关并行信息通过 `get_megatron_global_info` 与 `get_megatron_rank_info` 获取。TP 维度内的重分片由混合引擎内部处理。

##### 2. 生成序列并重计算对数概率

```python
@register(dispatch_mode=Dispatch.MEGATRON_PP_AS_DP_PROTO)
def generate_sequences(self, prompts: DataProto):
```

- `Dispatch.MEGATRON_PP_AS_DP_PROTO`：输入数据按 DP 维度划分，广播至同一 DP 组内的所有 TP/PP 进程。最终仅从 TP=0 且位于 PP 末端的进程收集输出。

在此函数中：
- Rollout执行自回归生成；
- Actor模型对生成的响应重新计算旧策略下的对数概率。

##### 3. 更新Actor模型

```python
@register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
def update_actor(self, data: DataProto):
```

- `Dispatch.MEGATRON_COMPUTE_PROTO`：数据按 DP 维度划分，分发至同一 DP 组内所有 TP/PP 进程，最终仅从 TP=0 与 PP 末端收集输出。

该接口使用 PPO 算法及熵正则项损失函数更新Actor模型权重。

> **注意**：当前训练阶段的张量并行规模（TP size）可与推理阶段不同。

---

### ReferenceWorker

##### 1. 模型初始化

Reference使用与Actor相同的初始化接口，但不初始化混合引擎（HybridEngine）和优化器（Optimizer）。初始化完成后，模型被封装为 `MegatronPPOActor` 实例。

##### 2. 计算参考对数概率

```python
@register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
def compute_ref_log_prob(self, data: DataProto):
```

该函数调用 `MegatronPPOActor` 提供的对数概率计算接口，获取Reference的输出概率。

---

### Critic与RewardModelWorker

##### 1. 模型初始化

初始化流程与Reference类似，但CriticWorker需额外初始化优化器。

##### 2. Critic价值计算

```python
@register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
def compute_values(self, data: DataProto):
```

##### 3. 更新Critic

```python
@register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
def update_critic(self, data: DataProto):
```

##### 4. RewardModel打分

```python
@register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
def compute_rm_score(self, data: DataProto):
```

所有接口均采用 `MEGATRON_COMPUTE_PROTO` 协议，确保在 3D 并行环境下高效执行。

---

## 训练优化工具集

### 显存卸载（Memory Offloading）

在 GPU 资源受限时，显存卸载技术可将模型的参数、梯度和优化器状态从 GPU 显存转移至 CPU 内存，仅在计算需要时加载回 GPU，从而显著降低 GPU 内存占用，支持更大规模模型的训练与推理。

#### 启用方式

**Actor与Reference**：

```bash
# Actor（含梯度与优化器）
actor_rollout_ref.actor.megatron.param_offload=True \
actor_rollout_ref.actor.megatron.grad_offload=True \
actor_rollout_ref.actor.megatron.optimizer_offload=True \

# Reference（仅参数，无梯度与优化器）
actor_rollout_ref.ref.megatron.param_offload=True
```

**Critic模型**：

```bash
critic.megatron.param_offload=True \
critic.megatron.grad_offload=True \
critic.megatron.optimizer_offload=True
```

---

### 性能分析器（Profiler）

性能分析器用于分析模型运行时的性能瓶颈，统计各操作耗时，辅助性能调优。基于 `torch.profiler` 实现。

> **当前限制**：仅支持 Megatron 架构下的Actor（Actor）角色。

#### 配置参数

- `use_profile=True`：启用性能分析。
- `profile_ranks=[0]`：指定需分析的进程（默认为 rank 0）。
- `step_start=0`：开始分析的训练步（step）。
- `step_end=1`：结束分析的训练步（注意：一步对应一次梯度更新）。
- `save_path="./profile"`：分析结果保存路径。

#### 示例配置

```bash
actor_rollout_ref.actor.profile.use_profile=True \
actor_rollout_ref.actor.profile.profile_ranks=[0] \
actor_rollout_ref.actor.profile.step_start=0 \
actor_rollout_ref.actor.profile.step_end=1 \
actor_rollout_ref.actor.profile.save_path="./profile"
```

---

## 相关文档

关于如何使用 MCore 训练各类模型的详细说明，请参阅 [MCore 文档](https://github.com/volcengine/verl/blob/main/verl/models/mcore/readme.md)。
