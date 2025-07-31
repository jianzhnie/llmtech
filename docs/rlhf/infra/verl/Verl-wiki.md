# 框架概览


verl（火山引擎强化学习库）是一个灵活、高效且可直接投入生产的强化学习训练库，专为大型语言模型（LLMs）设计。该项目是论文 [《HybridFlow：灵活高效的 RLHF 框架》](https://arxiv.org/abs/2409.19256v2) 的开源实现，旨在支持跨算法、模型和硬件配置的可扩展人类反馈强化学习（RLHF）。

verl 提供了一种混合编程模型，结合了单控制器与多控制器范式的优势，能够灵活表示并高效执行复杂的训练后数据流。该框架与现有 LLM 基础设施（包括 PyTorch FSDP、Megatron-LM、vLLM 和 SGLang）实现无缝集成。

## 核心原则

### 控制流与计算流分离

HybridFlow 通过分离两个关键关注点来解决分布式强化学习训练的挑战：

- 控制流 ：单进程编排 RL 算法逻辑（PPO 训练周期、优势计算、数据管理）
- 计算流 ：神经网络操作的多进程分布式执行（训练、推理、权重同步）

### 控制流实现

控制流实现为单个 Ray 远程进程，负责协调整个训练流程。该设计在协调分布式计算的同时，支持调试和算法开发。

入口点 `main_task()` 函数位于 [verl/trainer/main_ppo.py](https://github.com/volcengine/verl/blob/0f5ab5c8/verl/trainer/main_ppo.py)，作为驱动进程：

```python
@ray.remote(num_cpus=0.1, num_gpus=0)
def main_task(config, tokenizer):
    # Single process that coordinates everything
    trainer = RayPPOTrainer(...)
    trainer.fit()  # Main training loop
```

### 计算流实现

计算流将神经网络操作分配到专门的 Ray 工作进程，每个进程负责 RLHF 流程中的特定角色。

Workers通过 `@register` 装饰器系统向控制流暴露方法，该系统自动处理数据分发与收集：

```python
class ActorRolloutRefWorker(Worker):
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        # Distributed generation across multiple GPUs

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        # Distributed policy training
```

### 调度模式系统

`@register` 装饰器系统采用分发模式自动处理跨worker 的数据分发与收集：

| Dispatch Mode 调度模式             | Usage 用法                 | Data Handling 数据处理                           |
| ---------------------------------- | -------------------------- | ------------------------------------------------ |
| `Dispatch.ONE_TO_ALL`              | Model initialization       | Broadcast same data to all workers               |
| `Dispatch.DP_COMPUTE_PROTO`        | Training/inference         | Split `DataProto` across workers, gather results |
| `Dispatch.MEGATRON_COMPUTE_PROTO`  | Megatron training Megatron | Handle TP/PP data distribution                   |
| `Dispatch.MEGATRON_PP_AS_DP_PROTO` | HybridEngine generation    | Treat PP dimension as DP for rollout             |

### 关注点分离

混合流（HybridFlow）将控制逻辑与计算执行分离，从而能够：

- 集中式协调 ：算法逻辑与数据流的单点控制
- 分布式计算 ：通过专业化worker 实现并行执行
- 模块化集成 ：系统组件间采用清晰接口设计

## 核心特性

verl 具备以下核心能力：

### 多样化 RL 算法的轻松扩展

混合控制器编程模型支持以数行代码构建 GRPO、PPO 等 RL 数据流，通过优势估计器注册系统实现复杂训练后数据流的灵活表达与高效执行。

### 现有 LLM 基础设施的无缝集成


解耦计算与数据依赖，可无缝对接 FSDP、Megatron-LM、vLLM 等 LLM 框架。模块化 API 支持将模型部署至不同 GPU 组以实现高效资源利用。

### 业界领先的性能表现

-
  高吞吐量 ：集成 SOTA 级 LLM 训练与推理引擎，实现高生成与训练吞吐
- 高效的Actor模型重分片 ：3D-HybridEngine 消除内存冗余，降低训练与生成阶段切换时的通信开销

## 支持的算法与特性

###   强化学习算法

verl 通过 `verl.trainer.ppo.core_algos` 中的 `ADV_ESTIMATOR_REGISTRY` 和 `POLICY_LOSS_REGISTRY` 系统实现强化学习算法：

```python
# Advantage estimator registration
@register_adv_est(AdvantageEstimator.GRPO)
def compute_grpo_outcome_advantage(...)

# Policy loss registration
@register_policy_loss("ppo")
def compute_policy_loss(...)
```

| Registry Type            | Purpose               | Key Implementations                                 |
| ------------------------ | --------------------- | --------------------------------------------------- |
| `ADV_ESTIMATOR_REGISTRY` | Advantage computation | GAE, GRPO, RLOO, OPO, GPG GAE、GRPO、RLOO、OPO、GPG |
| `POLICY_LOSS_REGISTRY`   | Policy loss functions | PPO, GPG, custom losses                             |

### 训练特性

-
  基于模型与函数的奖励机制 ：支持神经奖励模型和可验证奖励函数（适用于数学、编程等领域）
- 多模态强化学习 ：支持视觉语言模型（VLM），包括 Qwen2.5-VL、Kimi-VL
- 多轮工具调用 ：先进的对话与工具集成能力
-
  序列优化 ：Flash Attention 2 注意力机制、序列打包、通过 DeepSpeed Ulysses 实现的序列并行
- Memory efficiency: LoRA support, Liger-kernel integration, multi-GPU LoRA RL
- 可扩展性 ：支持高达 6710 亿参数的专家并行模型

### 支持的模型

| Model Family | Examples                                                                                    | Hub Support             | Special Features             |
| ------------ | ------------------------------------------------------------------------------------------- | ----------------------- | ---------------------------- |
| Qwen         | Qwen-3, Qwen-2.5, Qwen2.5-VL Qwen-3、Qwen-2.5、Qwen2.5-VL                                   | HuggingFace, ModelScope | Multi-modal support          |
| Llama        | Llama3.1, Llama3.2 Llama3.1、Llama3.2                                                       | HuggingFace             | Wide ecosystem support       |
| DeepSeek     | DeepSeek-LLM, DeepSeek-V3-0324, DeepSeek-671B DeepSeek-LLM、DeepSeek-V3-0324、DeepSeek-671B | HuggingFace             | Large MoE models             |
| Gemma        | Gemma2                                                                                      | HuggingFace             | Efficient architecture       |
| Other VLMs   | Kimi-VL                                                                                     | ModelScope              | Vision-language capabilities |

### Model Hub集成

verl 通过标准化的模型适配器和检查点管理器与主流Model Hub实现集成：

Model Hub集成：

- HuggingFace Transformers：原生支持 `transformers` 库
- ModelScope：通过环境变量 `VERL_USE_MODELSCOPE=True` 启用支持

| Model Source     | Configuration              | Implementation                      | Use Case                |
| ---------------- | -------------------------- | ----------------------------------- | ----------------------- |
| HuggingFace Hub  | Default model loading      | `transformers.AutoModelForCausalLM` | Standard model access   |
| ModelScope Hub   | `VERL_USE_MODELSCOPE=True` | `modelscope.AutoModelForCausalLM`   | Chinese alternative     |
| Local/HDFS Paths | Direct file paths          | Local filesystem loading            | Custom model deployment |

## 分布式训练与推理

### 训练后端支持

verl 通过 `single_controller` 框架中定义的统一 `Worker` 接口支持多种分布式训练后端：

| Backend     | Key Features                           | Parallel Strategies              | Recommended Use Case                          | Scaling Capability     |
| ----------- | -------------------------------------- | -------------------------------- | --------------------------------------------- | ---------------------- |
| FSDP/FSDP2  | PyTorch native, easy model integration | Data Parallel, Sequence Parallel | Research, prototyping, broad HF model support | Up to ~70B parameters  |
| Megatron-LM | 5D parallelism, highly optimized       | TP, PP, DP, EP, CP               | Production, large-scale training, MoE models  | Up to 671B+ parameters |

FSDP 特性：

- 兼容 HuggingFace Transformers
-
  FSDP2 支持，提供更高的吞吐量和更优的内存使用效率
- 支持 CPU 卸载与梯度累积
-
  易于集成和调试

Megatron 特性：

- 张量并行（TP）、流水线并行（PP）、数据并行（DP）
- 专家并行（EP）用于混合专家模型，上下文并行（CP）
- 支持 6710 亿+参数模型（DeepSeek-V3、Qwen3-235B）
- 高度优化的内核与通信

### 推理引擎

verl 通过 `@register` 装饰器系统的 `ActorRolloutRefWorker.generate_sequences()` 方法集成多推理引擎：

| Engine          | Key Features                                             | Best For                                      | Version Support                        |
| --------------- | -------------------------------------------------------- | --------------------------------------------- | -------------------------------------- |
| vLLM            | PagedAttention, continuous batching, high throughput     | Large-scale inference, production deployments | v0.8.3+ (recommended: `VLLM_USE_V1=1`) |
| SGLang          | Multi-turn conversations, tool calling, memory efficient | Agentic workflows, complex interactions       | Latest versions supported              |
| HuggingFace TGI | Simple integration, debugging                            | Single GPU exploration, debugging             | Basic support                          |

vLLM 集成：

- 专为强化学习训练期间的高吞吐量生成而优化
-
  采用分页注意力机制提升内存效率
-
  支持张量并行计算
- CUDA 图优化技术

SGLang  集成

- 高级多轮对话支持
- 工具调用与功能集成
-
  高效内存服务
- 代理循环能力



## 性能与可扩展性特性


verl 实现了多种并行策略与优化技术以实现高效扩展：

###  并行实现

| Parallelism Type       | Implementation                      | Configuration                      | Use Case                         |
| ---------------------- | ----------------------------------- | ---------------------------------- | -------------------------------- |
| Tensor Parallel (TP)   | Megatron-LM, vLLM Megatron-LM，vLLM | `tensor_model_parallel_size`       | Large model layers (>70B params) |
| Pipeline Parallel (PP) | Megatron-LM                         | `pipeline_model_parallel_size`     | Memory-constrained training      |
| Data Parallel (DP)     | FSDP, Megatron-LM                   | `world_size`                       | Multi-GPU scaling                |
| Expert Parallel (EP)   | Megatron-LM                         | `expert_model_parallel_size`       | MoE models (671B+)               |
| Context Parallel (CP)  | Megatron-LM                         | `context_parallel_size`            | Long context training            |
| Sequence Parallel (SP) | DeepSpeed Ulysses                   | `ulysses_sequence_parallel_size>1` | Long sequence handling           |

### 内存优化特性

| Optimization                      | Configuration                        | Backend  | Description                               |
| --------------------------------- | ------------------------------------ | -------- | ----------------------------------------- |
| FSDP CPU Offloading               | `fsdp_config.offload_policy=True`    | FSDP     | Offload parameters to CPU                 |
| Megatron Offloading               | `megatron.param_offload=True`        | Megatron | Parameter, gradient, optimizer offloading |
| Gradient Checkpointing 梯度检查点 | `enable_gradient_checkpointing=True` | Both     | Trade computation for memory              |
| Sequence Packing                  | `use_remove_padding=True`            | Both     | Efficient sequence handling               |
| Dynamic Batch Size                | `use_dynamic_bsz=True`               | FSDP     | Token-based batching                      |
| LoRA Fine-tuning                  | `enable_lora=True`                   | FSDP     | Parameter-efficient training              |
| Flash Attention 2                 | Automatic                            | Both     | Efficient attention implementation        |
| Liger Kernel                      | `use_liger=True`                     | SFT      | Optimized CUDA kernels                    |
| Entropy Checkpointing             | `entropy_checkpointing=True`         | FSDP     | Memory-efficient entropy calculation      |

## 核心训练系统


核心训练系统是 verl 框架中协调多节点多 GPU 分布式强化学习训练的中枢编排层。该系统负责管理训练工作流生命周期、协调不同工作角色（执行器、评判器、推演器）、处理资源分配，并提供训练执行的主要入口点。

核心训练系统采用混合架构设计，通过 Ray 框架由单一控制器进程协调分布式计算worker 。系统将控制流（单进程）与计算流（分布式worker ）分离。

### 主要入口点

根据后端策略不同，训练系统提供多个入口点：

| Entry Point           | Purpose                          | Backend Support |
| --------------------- | -------------------------------- | --------------- |
| `main_ppo.py::main()` | Primary PPO training entry       | FSDP, Megatron  |
| `TaskRunner.run()`    | Ray remote execution wrapper Ray | All backends    |
| `RayPPOTrainer.fit()` | Core training loop               | All backends    |

### RayPPOTrainer

`RayPPOTrainer` 类作为分布式 PPO 训练的核心协调器，负责管理worker 生命周期、协调训练步骤以及处理不同worker 类型间的数据流。

主要职责：

- worker 初始化与管理
- 训练循环执行
- worker 间的数据协调
- 检查点管理
-  验证过程执行

核心方法：

- `init_workers()` - Initialize distributed workers
  `init_workers()`
- `fit()` - Main training loop
  `fit()`
- `_train_step()` - Execute single training iteration
  `_train_step()`
- `_validate()` - Run validation process
  `_validate()`

### 训练循环实现

`fit()` 方法中的主训练循环实现标准 PPO 算法流程：

1. Sequence Generation: Actor generates responses using rollout workers

2. Reward Computation: Rewards are computed using reward functions or reward models

3. Advantage Estimation: Various estimators (GAE, GRPO, REINFORCE++) compute advantages

4. Policy Updates: Actor is updated using PPO loss with multiple epochs

5. Value Updates: Critic is updated to predict better value estimates


### Worker 角色系统

RayPPOTrainer采用基于角色的 worker 系统，不同类型worker 处理特定训练环节。每个角色可映射至不同worker 实现（FSDP 或 Megatron 后端）。

| Role 角色           | Class 类                | Responsibility 职责范围               |
| ------------------- | ----------------------- | ------------------------------------- |
| `Role.ActorRollout` | `ActorRolloutRefWorker` | Policy training + response generation |
| `Role.Critic`       | `CriticWorker`          | Value function training               |
| `Role.RewardModel`  | `RewardModelWorker`     | Reward computation                    |
| `Role.RefPolicy`    | `ActorRolloutRefWorker` | Reference policy for KL penalty       |

### Worker 初始化流程

Trainer通过 `init_workers()` 方法根据选定的后端策略初始化worker：

1. Resource Pool Creation: GPU resources are allocated across nodes
   资源池创建 ：跨节点分配 GPU 资源
2. Worker Class Selection: Backend-specific worker classes are chosen
   worker类选择 ：选择特定后端的worker类
3. Ray Actor Creation: Workers are instantiated as Ray remote actors
   Ray Actor 创建 ：将worker实例化为 Ray 远程 actor
4. Model Initialization: Each worker initializes its models and optimizers
   模型初始化 ：每个worker初始化其模型和优化器
5. Cross-Worker Synchronization: Initial state synchronization occurs
   跨worker同步 ：执行初始状态同步

###  数据流与处理

训练系统通过标准化的 `DataProto` 协议处理数据，确保组件间数据交换一致性

Key data processing functions:
关键数据处理函数：

- `compute_advantage()` - Compute advantage estimates using GAE, GRPO, or other estimators
- `apply_kl_penalty()` - Apply KL divergence penalty to rewards

- `compute_response_mask()` - Generate attention masks for response tokens

PPO 训练遵循结构化数据流模式，数据会流经不同处理阶段：

1. Raw data → DataLoader → DataProto batch

2. DataProto → Rollout → Generated responses

3. Responses → Reward function → Token-level rewards

4. All inputs → Advantage computation → Training-ready data


## 单一控制器模式

verl 采用单一控制器设计模式，通过中心化的 Ray 驱动进程协调分布式计算Workers。该模式实现：

- 简化的控制流 ：为复杂工作流提供单一协调点
- 弹性资源管理 ：动态将Workers分配至不同任务
- 简易调试 ：集中式日志记录与错误处理
- 模块化设计 ：控制逻辑与计算逻辑清晰分离

单一控制器模式为 verl 的分布式计算提供了统一抽象层，能够无缝协调不同后端的工作进程。它通过管理工作组、资源分配和方法调度模式，成为分布式训练和推理操作的基础架构。

### 硬件抽象

该框架通过以下方式提供硬件抽象：

- `get_device_name()`：返回适用的设备类型（cuda、npu、cpu）

- `get_torch_device()`：返回对应的 torch 设备命名空间
- `get_visible_devices_keyword()`：返回设备可见性的环境变量

- `get_nccl_backend()`：返回适用的通信后端

这种抽象设计使得 verl 只需最小代码修改即可在不同硬件平台上运行，支持：

- NVIDIA GPU
- AMD GPU
- 昇腾 NPU
- CPU（用于调试）

### 核心架构概述

该架构包含：

- 控制层 ：定义核心接口的抽象基类
- 执行层 ：worker实现与方法装饰系统
- 后端层 ：Ray 框架对抽象接口的具体实现
- 训练集成 ：与 verl 训练系统的集成

### Worker

`Worker` 类作为所有分布式worker的基类，负责初始化、环境配置、通信建立和硬件抽象

`Worker` 类管理分布式训练环境搭建，并提供装饰方法供执行：

| Environment Variable 环境变量 | Purpose 目的                |
| ----------------------------- | --------------------------- |
| `WORLD_SIZE`                  | Total number of workers     |
| `RANK`                        | Worker's global rank        |
| `LOCAL_RANK`                  | Worker's local rank on node |
| `MASTER_ADDR`                 | Master node address         |
| `MASTER_PORT`                 | Master node port            |
| `CUDA_VISIBLE_DEVICES`        | GPU device visibility GPU   |

框架包含全面的硬件抽象以支持不同平台：

| Platform               | Device Type | Visible Devices Env         | Communication Backend |
| ---------------------- | ----------- | --------------------------- | --------------------- |
| NVIDIA GPUs NVIDIA GPU | `cuda`      | `CUDA_VISIBLE_DEVICES`      | `nccl`                |
| AMD GPUs AMD GPU       | `cuda`      | `HIP_VISIBLE_DEVICES`       | `nccl`                |
| Ascend NPUs            | `npu`       | `ASCEND_RT_VISIBLE_DEVICES` | `hccl`                |
| CPU                    | `cpu`       | N/A 无                      | N/A                   |

### WorkerGroup


`WorkerGroup` 类负责管理工作线程集合，并为分布式执行提供方法绑定。该组件是协调分布式操作的核心抽象层。

`WorkerGroup._bind_worker_method()` 函数通过 `@register` 装饰器属性动态绑定worker方法，实现分布式执行模式。该机制是将本地方法调用转化为分布式操作的核心组件。

主要职责：

- 管理worker生命周期与引用
- 将装饰器修饰的工作方法绑定到组接口
- 提供执行模式（所有worker、仅 rank-zero 节点等）
- 处理数据分发与结果收集

### ResourcePool


`ResourcePool` 类负责管理跨节点的资源分配，追踪进程数量与 GPU 分布情况。它为硬件资源管理提供了可被不同后端实现的抽象层。

| Property 属性        | Description 描述        |
| -------------------- | ----------------------- |
| `world_size`         | 所有节点的进程总数      |
| `store`              | 各节点进程数量列表      |
| `max_colocate_count` | 每个 GPU 的最大进程数   |
| `n_gpus_per_node`    | 每个节点可用的 GPU 数量 |

`ResourcePool` 提供了一个硬件抽象层，使框架能够在分布式环境中高效分配和管理计算资源。

### Ray 实现方案

Ray 后端通过放置组支持和资源调度功能，为核心抽象提供了具体实现。Ray 是 verl 中使用的主要分布式计算后端。

#### RayWorkerGroup

`RayWorkerGroup` 类继承 `WorkerGroup` 并扩展了用于管理分布式 Ray 参与者的 Ray 特有功能。

Ray 实现的关键特性：

- Placement Groups:  `sort_placement_group_by_node_ip()` 确保重启时worker布局保持一致
- Resource Bundles: 支持通过 `max_colocate_count` 配置 GPU 与 CPU 的分配方案
- Environment Isolation: 每个worker获得独立的环境变量配置，保障分布式训练隔离性
- Fused Workers: 支持通过 `create_colocated_worker_cls_fused()` 实现资源共享的共置worker
- Detached Workers:  支持持久化的worker，可在驱动重启后继续存活

`RayWorkerGroup` 实现方案处理以下复杂逻辑：

- 创建并管理具有特定资源需求的 Ray 执行器
-
  配置分布式训练环境变量
- 协调worker之间的通信
-
  处理方法调度与结果收集

#### Ray 资源池


`RayResourcePool` 继承基础 `ResourcePool` 并扩展了 Ray 特有的放置组管理功能。它负责创建和管理用于资源分配的 Ray 放置组。

资源包配置：

```python
# Resource bundle configuration
bundle = {"CPU": self.max_colocate_count}
if self.use_gpu:
    bundle[device_name] = 1  # GPU or NPU
    if self.accelerator_type is not None:
        bundle[self.accelerator_type] = 1e-4
```


`RayResourcePool` 提供以下功能：

- Creation and caching of Ray placement groups

- 支持多种硬件类型（CUDA、ROCm、NPU）
- 可配置的放置策略
- 支持分离式（持久化）放置组

### 调度与执行系统

该框架采用基于装饰器的系统来定义数据如何在worker节点间分发及方法如何执行。此系统是"单控制器"范式的核心，能够通过简单的方法调用来表达复杂的分布式操作。

#### 核心机制

`Dispatch` 通过 `@register` 装饰器系统工作，每个 dispatch 模式都对应特定的 `dispatch_fn` 和 `collect_fn` 函数 decorator.py:378-418 。当控制流调用 worker 方法时，系统会：

1. 数据分发：根据 dispatch 模式将输入数据分割成多个部分
2. 远程执行：将数据分发到各个 worker 进行并行计算
3. 结果收集：收集所有 worker 的计算结果并合并

#### 调度模式

| Dispatch Mode                | Use Case                             | Data Pattern                                                         |
| ---------------------------- | ------------------------------------ | -------------------------------------------------------------------- |
| `ALL_TO_ALL`                 | Simple operations                    | No data distribution  无数据分发                                     |
| `ONE_TO_ALL`                 | Broadcast operations 广播操作        | Same data to all workers 向所有woker发送相同数据                     |
| `DP_COMPUTE_PROTO`           | Data parallel training 数据并行训练  | DataProto chunked by `world_size` 数据协议按 `world_size` 分块       |
| `MEGATRON_COMPUTE_PROTO`     | Model parallel training 模型并行训练 | DataProto chunked by `dp_size` 数据协议按 `dp_size` 分块             |
| `DP_COMPUTE_PROTO_WITH_FUNC` | Function execution 函数执行          | DataProto chunked with function application 应用函数后的数据协议分块 |

调度系统负责处理：

- 根据调度模式将输入数据分配给各woker
- 按照执行模式在woker上运行方法
- 收集并整合来自各woker的结果
- 处理非均匀数据分割的填充问题

#### 实际应用示例

在 FSDP worker 中，`generate_sequences` 方法使用 `DP_COMPUTE_PROTO` 模式： fsdp_workers.rst:70-78

这意味着当控制流调用 `actor_rollout_ref_wg.generate_sequences(prompts)` 时，系统会自动：

- 将 `prompts` 数据按 worker 数量分割
- 分发到各个 GPU worker 并行生成
- 收集所有结果并合并返回

#### 绑定机制

`Dispatch` 模式在 `WorkerGroup` 初始化时通过 `_bind_worker_method` 绑定到具体的分发和收集函数。 这使得分布式调用对控制流来说就像单进程调用一样简单 。

### 注册装饰器

`@register` 装饰器是分布式环境中配置方法执行行为的关键机制，它通过向worker方法附加元数据来控制这些方法在workergroup中的执行方式。

示例用法：

```python
@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO, execute_mode=Execute.ALL, blocking=True)
def generate_sequences(self, data_proto):
    # Method automatically gets data distribution and collection
    data_proto = data_proto.to(torch.cuda.current_device())
    # Process data on this worker
    return result_proto
```

装饰器参数控制：

- `dispatch_mode`: 输入数据在worker间的分配方式
- `execute_mode`: 指定执行方法的worker类型（ALL /RANK_ZERO ）
- `blocking`：是否等待所有worker完成执行
- `materialize_futures`:  是否在执行前解析 Ray futures 对象

当方法被 `@register` 装饰后，即可通过 `WorkerGroup` 接口进行分布式执行，系统会自动处理数据分发和结果收集。

#### 定义和作用

`@register` 装饰器用于将单个方法转换为可以在多个分布式工作进程上执行的方法。它不会改变方法的功能，而是为方法附加元数据，这些元数据会在 WorkerGroup 初始化时被提取和使用。 single_controller.rst:82-107

#### 装饰器参数

`@register` 装饰器支持以下主要参数：

- dispatch_mode: 控制数据如何分发到各个工作进程
- execute_mode: 控制哪些工作进程执行方法
- blocking: 控制执行是否为阻塞式
- materialize_futures: 控制是否物化 futures

#### 工作原理

1. 注册阶段: 装饰器为方法附加元数据属性
2. 绑定阶段: WorkerGroup 初始化时提取这些属性，生成对应的分发、执行、收集函数
3. 调用阶段: 通过 WorkerGroup 调用方法时，自动处理数据分发、远程执行和结果收集



## FSDP Workers

本节介绍 verl 中基于 FSDP（全分片数据并行）的woker实现，这些节点利用 PyTorch 的 FSDP 后端为Actor、评论家和参考策略角色提供分布式训练能力。这些woker构成了 verl 混合流架构中 FSDP 策略的计算层。

FSDP woker采用 PyTorch 全分片数据并行策略，实现了强化学习训练所需的核心计算角色。这些woker设计为由 single_controller 框架管理，并与推理引擎集成以生成推演数据。

### Core FSDP Worker Classes

FSDP woker系统包含三个主要woker类，分别实现 RL 训练流程中的不同角色。每个woker类封装了分布式计算逻辑，并提供带有 `@register` 装饰器的 API 接口与 single controller 通信。

### FSDP Actor 实现

`DataParallelPPOActor` 类实现了采用 FSDP 参数分片的 PPO Actor训练。同时支持仅前向推理（用于计算对数概率）和反向训练（用于策略更新）。

#### 核心特性

- Parameter Sharding:  使用 FSDP 或 FSDP2 在 GPU 间分配模型参数
- Sequence Packing:  可选 `use_remove_padding` 参数提升内存效率
- Ulysses Sequence Parallel:  支持长上下文训练 `ulysses_sequence_parallel_size > 1`
- Dynamic Batching:  通过 `use_dynamic_bsz` 配置可变序列长度支持
- Entropy Optimization: 分块熵计算和梯度检查点选项

#### 核心方法

该Actor节点实现了两个主要计算方法：

对数概率计算 :

- 以微批次方式处理输入序列
- 支持填充模式与去除填充模式
- 处理视觉语言模型的多模态输入
- 返回对数概率值及可选的熵值

策略更新 ：

- 实现 PPO 策略梯度更新
- 支持多种策略损失模式（标准/自定义）
- 处理熵正则化与 KL 散度惩罚项
- 采用 FSDP 感知的梯度裁剪范数计算

## DataProto 协议

verl 采用标准化的 `DataProto` 协议实现组件间的数据交换：

- 统一接口 ：所有Workers采用一致的数据格式
- 类型安全 ：强类型系统实现更佳的错误检测
- 高效序列化 ：为分布式通信优化设计
- 可扩展性 ：轻松添加新数据字段且不破坏现有代码

### DataProto 的定义

`DataProto` 是一个数据类，作为 VERL 中分布式训练组件之间的标准化数据交换协议 。它将 PyTorch 张量与非张量数据统一在一个容器中，支持高效的序列化、批处理和分布式操作。

`DataProto` 包含三个主要组件：

1. **batch**: `TensorDict` 类型，包含具有一致批次维度的 PyTorch 张量
2. **non_tensor_batch**: `dict[str, np.ndarray]` 类型，包含非张量数据（如字符串、元数据），以 numpy 数组形式存储
3. **meta_info**: `dict` 类型，包含额外的元数据和配置信息

### DataProto 的主要作用

#### 1. 数据交换和传输

`DataProto` 作为分布式训练组件之间传递结构化数据的标准接口 protocol.py:346-409 ，支持创建、操作和转换数据。

#### 2. 批处理操作

提供多种批处理操作方法，包括：

- `slice()`: 数据切片
- `concat()`: 数据连接
- `chunk()`: 数据分块
- `union()`: 数据合并
- `select()`: 数据选择

#### 3. 序列化和分布式支持

实现了针对 Ray 分布式对象存储优化的自定义序列化机制 protocol.py:267-290 ，支持高效的分布式数据传输。

#### 4. 异步操作支持

通过 `DataProtoFuture` 类支持分布式环境中的异步数据操作 protocol.py:948-995 ，实现延迟数据获取和异步执行。

#### 5. 训练流水线集成

提供与 PyTorch `DataLoader` 兼容的迭代器接口 protocol.py:625-663 ，支持批量整理和数据加载。

#### 6. 内存优化

包含多项内存优化功能：

- 自动填充 (auto-padding)
- 设备管理 (device management)
- 延迟求值 (lazy evaluation)
- 序列化优化 protocol.py:47-65

### Notes

`DataProto` 是 VERL 框架的基础数据结构，广泛应用于训练、推理和rollout过程中。它通过 TensorDict 库实现了对多个张量的统一操作，同时支持非张量数据的处理，为分布式机器学习提供了高效的数据处理解决方案。该类还与序列平衡算法集成，支持动态批处理和负载均衡优化。



## FAQ

### 1. entropy_checkpointing 实现的是什么功能

#### 功能概述

`entropy_checkpointing` 实现了熵计算的梯度检查点功能，通过重新计算来减少训练过程中的内存峰值使用 。

#### 具体实现

在 FSDP Actor 的实现中，`entropy_checkpointing` 控制熵计算是否使用 PyTorch 的 `checkpoint` 功能： dp_actor.py:206-211

当 `entropy_checkpointing=True` 时，熵计算会使用 `torch.utils.checkpoint.checkpoint()` 包装，这样在反向传播时会重新计算熵值而不是存储中间激活值 dp_actor.py:275-278 。

#### 配置方式

该功能可以通过配置文件启用： legacy_ppo_trainer.yaml:268-269 _generated_ppo_trainer.yaml:57

#### 使用场景

根据性能调优文档，`entropy_checkpointing` 主要用于解决标准梯度检查点不适用于熵计算的问题： perf_tuning.rst:216-218

这个功能与 `entropy_from_logits_with_chunking` 配合使用，共同优化熵计算的内存使用 perf_tuning.rst:212-214 。

#### Notes

`entropy_checkpointing` 是 VERL 框架中专门针对熵计算优化的内存管理功能，通过牺牲一定的计算时间来换取更低的内存占用，特别适用于大模型训练场景。

```python
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)
            return entropy, log_probs
```

[volcengine/verl](https://github.com/volcengine/verl)[tests/trainer/config/legacy_ppo_trainer.yaml](https://github.com/volcengine/verl/blob/0f5ab5c8/tests/trainer/config/legacy_ppo_trainer.yaml)

```python
    # calculate entropy with chunking to reduce memory peak
    entropy_from_logits_with_chunking: False
    # recompute entropy
    entropy_checkpointing: False
    # checkpoint configs
    checkpoint:
```

[volcengine/verl](https://github.com/volcengine/verl)[verl/trainer/config/_generated_ppo_trainer.yaml](https://github.com/volcengine/verl/blob/0f5ab5c8/verl/trainer/config/_generated_ppo_trainer.yaml)

```yaml
    grad_clip: 1.0
    ulysses_sequence_parallel_size: 1
    entropy_from_logits_with_chunking: false
    entropy_checkpointing: false
    fsdp_config:
      _target_: verl.workers.config.FSDPEngineConfig
      wrap_policy:
```



### 2. 生成模型的 logits 为什么要这样计算

```python
logits = output.logits
logits.div_(temperature)
logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
```

在文本生成任务中，`logits` 是模型输出的原始未归一化的分数，它们代表了词汇表中每个词元（token）在下一个位置出现的可能性。

1.  **`logits = output.logits`**:
    这一步简单地从模型的输出中提取 `logits`。`output.logits` 的形状通常是 `(batch_size, sequence_length, vocab_size)`，其中 `batch_size` 是批次大小，`sequence_length` 是当前序列的长度，`vocab_size` 是词汇表的大小。
2.  **`logits.div_(temperature)`**:
    这里使用了温度参数 `temperature` 来调整 `logits`。温度参数控制生成文本的随机性：
    -   当 `temperature > 1` 时，`logits` 被缩小，使得概率分布更加平滑，增加了生成文本的多样性和随机性。
    -   当 `temperature < 1` 时，`logits` 被放大，使得高分的 `logits` 更高，低分的更低，从而让模型更倾向于选择高概率的词元，生成结果更加确定和集中。
    -   当 `temperature = 1` 时，`logits` 不变，相当于没有应用温度调整。
    `div_` 是一个就地操作（in-place operation），直接修改 `logits` 张量。
3.  **`logits = logits[:, -response_length - 1 : -1, :]`**:
    这一行代码通过切片操作选取了特定范围的 `logits`。具体来说：
    -   `-response_length - 1 : -1` 表示从倒数第 `response_length + 1` 个位置开始，到倒数第 2 个位置结束（不包括倒数第 1 个位置）。这里的 `response_length` 通常指的是你希望生成的回复的长度。这个切片的目的是获取生成过程中每一步对应的 `logits`。例如，如果你正在生成一个长度为 `response_length` 的回复，那么你需要 `response_length` 个 `logits` 向量，每个向量对应生成一个词元。切片 `[:, -response_length - 1 : -1, :]` 会选取从生成开始前一个位置到生成结束前一个位置的所有 `logits`，这样正好对应了生成 `response_length` 个词元所需的 `logits`。

最终，`logits` 的形状变为 `(batch_size, response_length, vocab_size)`，这正是后续进行采样（如贪婪搜索、束搜索、top-k 采样等）所需要的格式。

#### 实际例子

假设我们正在使用一个对话模型（比如我，Qwen）来生成回复。我们有一个包含一个样本（`batch_size=1`）的批次。对话历史（提示词）是 "Hello, how are you?"，这个提示词被分词后对应的 token 序列长度是 6。

我们希望模型生成一个长度为 4 (`response_length=4`) 的回复，比如 "I am fine, thank you!"。

**生成过程与 logits 的产生：**

在自回归生成中，模型会一步一步地生成新的 token。每生成一个新 token，模型都会基于当前的完整输入序列（历史 + 已生成的部分）来预测下一个 token 的概率分布（即 logits）。

*   **步骤 0 (输入提示):** 输入是 `["Hello", ",", "how", "are", "you", "?"]` (长度=6)。模型处理这个输入，但通常我们不直接使用这一步的 logits 来生成第一个回复 token（或者根据实现方式，可能会用，但这里我们按常见方式理解）。
*   **步骤 1 (生成第一个回复 token):**  模型输入仍然是完整的提示词。模型的输出 `output.logits` 会包含对 *下一个* token 的预测。这个预测对应于 logits 序列的第 6 个位置（索引为 5，如果从 0 开始计数）。
*   **步骤 2 (生成第二个回复 token):** 模型输入是 `["Hello", ",", "how", "are", "you", "?", "I"]` (长度=7)。模型输出的 logits 包含对 *下一个* token 的预测，对应于 logits 序列的第 7 个位置（索引为 6）。
*   **步骤 3 (生成第三个回复 token):** 模型输入是 `["Hello", ",", "how", "are", "you", "?", "I", "am"]` (长度=8)。模型输出的 logits 包含对 *下一个* token 的预测，对应于 logits 序列的第 8 个位置（索引为 7）。
*   **步骤 4 (生成第四个回复 token):** 模型输入是 `["Hello", ",", "how", "are", "you", "?", "I", "am", "fine"]` (长度=9)。模型输出的 logits 包含对 *下一个* token 的预测，对应于 logits 序列的第 9 个位置（索引为 8）。

现在，模型已经生成了 4 个 token，我们得到了一个完整的输出序列，其总长度 `sequence_length` 是 10 (6个提示 + 4个回复)。`output.logits` 的形状是 `(1, 10, vocab_size)`。这个 logits 张量包含了模型在处理这 10 个 token 序列时，对 *每一个位置之后* 应该出现什么 token 的预测。

**关键点：** 我们关心的不是模型对提示词内部或整个序列之后的预测，而是**模型在生成我们想要的回复时，每一步所依据的预测（logits）**。具体来说，我们想要的是：
1.  生成第一个回复 token "I" 时，模型在位置 5 (第6个) 的 logits。
2.  生成第二个回复 token "am" 时，模型在位置 6 (第7个) 的 logits。
3.  生成第三个回复 token "fine" 时，模型在位置 7 (第8个) 的 logits。
4.  生成第四个回复 token "," 时，模型在位置 8 (第9个) 的 logits。

**应用索引 `[:, -response_length - 1 : -1, :]`**

现在，我们来解析这个切片：
*   `-response_length - 1 : -1` -> 这是关键。`response_length` 是 4。
    *   `-response_length - 1` = `-4 - 1` = `-5`。这表示从倒数第 5 个元素开始。
    *   `-1` 表示到倒数第 1 个元素之前结束（不包含倒数第 1 个）。
    *   对于一个长度为 10 的序列，索引如下：
        *   正向索引: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
        *   反向索引: -10, -9, -8, -7, -6, **-5**, **-4**, **-3**, **-2**, -1
    *   所以，`-5 : -1` 选取的是索引为 -5, -4, -3, -2 的元素。
    *   对应的正向索引是：5, 6, 7, 8。
*   `:` -> 选择词汇表的所有维度。

**结果：**

`logits[:, -response_length - 1 : -1, :]` 会选取 `output.logits` 中索引为 5, 6, 7, 8 的这 4 个 logits 向量。这正是我们在生成 4 个回复 token 时，模型在每一步所做出的预测！

*   位置 5 的 logits: 用于生成 "I"
*   位置 6 的 logits: 用于生成 "am"
*   位置 7 的 logits: 用于生成 "fine"
*   位置 8 的 logits: 用于生成 ","

最终得到的 `logits` 形状是 `(1, 4, vocab_size)`，完美对应了我们生成的 `response_length=4` 个 token 所依赖的模型原始预测分数。

**总结:**

这个切片操作 `[:, -response_length - 1 : -1, :]` 的精妙之处在于，它利用了生成序列的结构：
*   `-1` 排除了序列最后一个位置的 logits（通常对应生成结束后的位置，我们不关心）。
*   `-response_length - 1` 确保了我们从生成回复的 *起始位置* 开始选取。因为回复是从提示词结束后开始的，而提示词占了前面的位置，所以回复部分的 logits 在序列末尾。通过从倒数第 `response_length + 1` 个位置开始，到倒数第 1 个位置之前结束，我们恰好截取了生成整个回复过程中所用到的 `response_length` 个 logits。
