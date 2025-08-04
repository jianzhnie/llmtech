# Verl 源码解析与 Hybrid Flow 编程范式

## 0. 引言

本次分享将深入解析 Verl 框架的源码实现及其背后的核心编程范式——**HybridFlow**。Hybrid Flow 是 Verl 对应的核心技术，已发表于系统领域顶级会议 EuroSys，旨在为大规模强化学习（Reinforcement Learning, RL）训练提供高效、灵活的分布式执行支持。

我们将从三个部分展开本次讲解：

1. **背景介绍**：形式化地定义 RL 训练中的系统问题；
2. **代码解析（Code Walkthrough）**：以调试器视角，从入口点出发，逐步剖析 Verl 的执行流程；
3. **核心机制解析**：深入探讨 Verl 基于 SPMD （Single Program, Multiple Data）模式的并行计算实现。

---

## 1. 背景：强化学习的计算建模

### 1.1 RL 作为 Dataflow Graph

在大型语言模型（LLM）的后训练阶段，强化学习任务本质上是一个复杂的**分布式调度问题**。为了系统性地解决这一问题，Verl 将整个训练流程抽象为一个 **Dataflow Graph**。

以PPO， GRPO 等算法为例，一个典型的 RL 训练流程可分解为以下三个核心维度：

- **多模型（Multiple Models）**
  如 Actor、Critic、Reference Model、Reward Model 等。这些模型在训练过程中协同工作，各自承担不同职责。
- **多阶段（Multiple Stages）**
  典型的训练周期包含三个阶段：
  - **生成（Generation/Rollout）**：使用 Actor 模型（当前策略）生成响应序列；
  - **经验准备（Experience Preparation）**：使用提示和生成的回复，通过各自模型的单次前向计算，对生成的回复进行评分。这个阶段通常涉及以下模型：
    - Actor Model： 计算旧策略下的对数概率；

    - Critic Model：计算生成回复的值（values）；

    - Reference Model：计算生成回复的参考对数概率，它通常是 Actor 模型在 RLHF 之前的版本，用于限制 Actor 模型在训练过程中偏离过远。

    - Reward Model：计算生成回复的奖励（rewards）。奖励模型通常是一个基于人类偏好数据进行微调的 LLM，其语言建模头被替换为标量输出头。

  - **训练（Training）**：基于收集的经验更新 Actor 与 Critic 模型。
- **多种工作负载（Multiple Workloads）**
  不同模型在不同阶段的工作负载类型各异：

  - 生成阶段以**自回归推理**为主；

  - 训练阶段以**高并行度的前向和反向传播**为主；

  - 推理与训练对并行策略、内存布局和通信模式的需求截然不同。

### 1.2 执行模式与优化目标

我们的目标是将上述 Dataflow Graph高效地映射为 GPU 集群上的**执行模式（Execution Pattern）**。

以 PPO 执行过程为例：

| 阶段        | 作用           | GPU 上典型负载                   |
| ----------- | -------------- | -------------------------------- |
| Generation  | 产生 rollout   | actor 前向                       |
| Preparation | 计算优势、奖励 | reward / reference / critic 前向 |
| Training    | 更新策略       | actor & critic 反向              |

每阶段内部又存在多种工作负载（workload），例如：

- actor 在 **Generation** 阶段只做前向，在 **Training** 阶段做反向；
- critic 在 **Preparation** 阶段做前向，在 **Training** 阶段做反向。

在实际部署中，我们需要：

- **模型放置（Model Placement）**
  将不同模型分配至集群中的不同设备组。例如：

  - Actor 与 Rollout 模型部署在 GPU 0-1；

  - Critic 部署在 GPU 2-3；

  - Reference 与 Reward Model 部署在 GPU 4-5。

- **执行调度约束**

  - **时序依赖**：生成必须先于经验准备，经验准备必须先于训练；

  - **并行性**：无依赖的阶段（如 critic 与 reward 模型的推理）可并行执行；

  - **资源冲突**：部署在同一设备上的多个模型需串行执行，以避免资源竞争。

**优化目标**：
在满足上述约束的前提下，最大化整体训练吞吐量。

### Verl 的设计愿景

Verl 的终极目标是实现 **“用户只需定义 Dataflow Graph，框架自动完成分布式优化”** 的愿景。即：

- 用户仅需在数学层面定义 RL 算法的行为（如损失函数、策略更新规则）；
- 框架自动处理底层的分布式并行策略、通信优化、内存管理等复杂细节。

虽然这一理想仍在演进中，但 Verl 通过 **Hybrid Flow** 范式，结合 **Single-Controller** 与 **Multi-Controller** 机制，在**灵活性**与**效率**之间取得了良好平衡。

## 2. Verl  代码解析

Verl 是一个包含数万行代码的复杂系统，我们将聚焦其核心设计逻辑，以简化后的代码示例深入剖析其执行流程。我们将从入口点开始，逐步解析资源管理、工作负载调度与并行执行机制。

### 2.1 入口点与资源分配

程序启动后，首先进入 `main` 函数，并调用 `TaskRunner.run()` 作为间接入口。

主要步骤包括：

1. **定义资源池（Resource Pool）**：创建全局或局部的 GPU 资源池。
2. **角色映射**：将不同的 workload 角色（如 Actor、Critic）映射到指定的资源池。
3. **初始化管理器**：创建资源池管理器（Manager）并传递给 Trainer。

#### 2.1.1. 全局资源池配置

在默认配置中，Verl 使用统一的资源池（Resource Pool）管理所有 GPU 资源，并将不同角色的Worker映射至对应资源池。

```python
# 定义全局资源池
global_pool_id = "global_pool"
resource_pool_spec = {
    global_pool_id: ([config.trainer.n_gpus_per_node] * config.trainer.nnodes),
}

# 将所有角色（Actor, Critic, RefPolicy, RewardModel）映射到同一资源池
mapping = {
    Role.ActorRollout: global_pool_id,
    Role.Critic: global_pool_id,
    Role.RefPolicy: global_pool_id,
    Role.RewardModel: global_pool_id,
}

# 创建资源池管理器
resource_pool_manager = ResourcePoolManager(
    resource_pool_spec=resource_pool_spec,
    mapping=mapping
)

# 初始化训练器
trainer = RayPPOTrainer(
    config=config,
    resource_pool_manager=resource_pool_manager
)
trainer.fit()
```

> **设计考量**：所有WorkLoad 共享同一资源池意味着它们在时间上**串行执行**。尽管这牺牲了部分并行潜力，但在多数场景下，由于各阶段（生成、经验准备、训练）本身存在强时序依赖，串行执行反而能有效利用全部 GPU 资源，避免因资源碎片化导致的利用率下降。

### 2.2 Worker 分组与进程共置

在 `Trainer` 初始化阶段，关键操作是 **Worker 分组**：

- 将不同 workload（如 Actor、Critic）分组为若干个 `Worker Group`。
- 每个 `Worker Group` 对应一个资源池，包含一个或多个 GPU。

#### 2.2.1 显存碎片化问题与进程共置（Process Collocation）

PyTorch 框架的显存管理机制（如 CUDA caching allocator）存在一个关键限制：**进程间无法共享显存**。

PyTorch 的显存管理器为每个进程预留（reserve）显存池以提高分配效率。然而，不同进程间的显存无法共享。若多个进程（如 Actor、Critic）分别启动，即使它们不同时运行，各自预留的显存也无法被对方利用，导致总显存占用远超峰值需求，形成严重浪费。

为解决此问题，Verl 采用 **进程共置** 策略：

- 在每个 GPU 上仅维护**一个进程**；
- 将不同 Workload 的逻辑融合到同一个进程中，使其在不同时间执行不同任务（如先运行 Actor 生成，再运行 Critic 推理）。

####  实现机制：动态类合成

为实现单进程运行多角色，Verl 将多个Worker类（如 `ActorRolloutWorker`, `CriticWorker`）的方法**动态融合**到一个 `WorkerGroup` 类中：

```python
# 伪代码：输入多个 Worker 类，合成一个新类
def create_worker_group_class(*worker_classes):
    class HybridWorkerGroup:
        def __init__(self):
            for worker_class in worker_classes:
                # 初始化所有 Worker 的实例或方法
                setattr(self, worker_class.__name__.lower(), worker_class())
        # 自动继承所有 Worker 的方法
    return HybridWorkerGroup
```

通过此方式，新类拥有所有子类的方法，单个进程即可执行所有任务，显存仅需按**单个最大工作负载**预留，显著提升显存利用率，有效避免了显存碎片化。

#### 2.2.4. WorkerGroup的创建

每个 `WorkerGroup` 对应一个资源池，内部包含多个运行在 GPU 上的 Ray Worker进程。

```python
for resource_pool, class_dict in self.resource_pool_to_cls.items():
    wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool)
    spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
    all_wg.update(spawn_wg)
    self.wg_dicts.append(wg_dict)
```

`spawn()` 方法为每个 GPU 启动一个独立进程，并设置 SPMD 环境变量（如 `RANK`、`WORLD_SIZE`）。

### 2.3. 核心执行：Single-Controller 范式

初始化完成后，程序进入 `fit()` 函数，执行核心训练循环。Verl 采用 **Single-Controller** 范式，由主进程（Controller）协调所有分布式操作，用户代码聚焦于数据流逻辑，无需关心底层分布式细节。

#### 2.3.1. 同步执行流程

在全局资源池的默认配置下，`fit` 函数采用同步逻辑。以 PPO 为例，其核心流程如下：

```python
for epoch in range(self.config.trainer.total_epochs):
    for batch_dict in self.train_dataloader:
        batch = DataProto.from_single_dict(batch_dict)  # 统一数据结构

        # 阶段1: 生成 (Generation)
        gen_batch = batch.pop("prompt")  # 提取生成所需数据
        gen_output = self.actor_rollout_wg.generate_sequences(gen_batch)

        # 阶段2: 准备经验 (Experience Preparation)
        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
        values = self.critic_wg.compute_values(batch)
        reward_tensor = self.rm_wg.compute_rm_score(batch)

        # 阶段3: 训练 (Training)
        self.critic_wg.update_critic(batch)
        self.actor_rollout_wg.update_actor(batch)
```

该循环清晰地体现了 Dataflow Graph的三个阶段，执行顺序严格遵循依赖关系，确保正确性。

#### 2.3.2. 通信开销分析

`Trainer.fit()` 使用 **Single-Controller** 编程模型, 主进程（controller）拥有全局视角通过 Ray 的 RPC（远程过程调用）与Worker进程通信，传递 `DataProto` 对象。尽管进程间通信（IPC）存在开销，但其影响有限，原因如下：

- **传输数据量小**：传递的主要是 `prompt`、`response`、`log_probs`、`rewards` 等标量或小张量，远小于模型参数、隐藏状态（hidden states）或优化器状态。
- **计算密集型主导**：各阶段的计算耗时远超通信耗时，通信开销被有效掩盖。

> **权衡取舍**：通过使用 Ray 的 RPC 框架传输控制信息，Verl 以较小的通信代价换取了编程模型的极大灵活性，用户无需手动管理分布式同步与通信。

### 2. SPMD 环境初始化

Verl 为每个 Ray Worker进程设置标准的分布式环境变量：

```python
env_vars = {
    'WORLD_SIZE': str(world_size),
    'RANK': str(rank),
    'LOCAL_RANK': str(local_rank),
    # 其他环境变量...
}
```

这些变量被用于初始化 PyTorch 的分布式通信组（如 `torch.distributed.init_process_group`）。

### 3. SPMD 行为定义：`@register` 装饰器

以 `update_actor` 为例，其行为通过 `@register` 装饰器声明：

```python
@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
def update_actor(self, data: DataProto):
    data = data.to(torch.cuda.current_device())
    self.actor.update_policy(data=data)
    self.actor_lr_scheduler.step()
```

`dispatch_mode=Dispatch.DP_COMPUTE_PROTO` 表示该方法在数据并行模式下执行，输入数据将被自动切分并分发至各 GPU。

### 4. 数据分发与结果收集

Verl 通过 `dispatch_fn` 和 `collect_fn` 实现数据的分发与聚合：

- **分发函数**：`dispatch_dp_compute_data_proto` 将输入数据沿 batch 维度均匀切分；
- **收集函数**：`collect_dp_compute_data_proto` 将各 GPU 的输出结果拼接（concat）为完整张量。

$$
\text{Output} = \bigcup_{i=1}^{N} \text{forward}(x_i), \quad \text{where } x = [x_1, x_2, \dots, x_N]
$$

### 5. 并行执行：`execute_all`

`execute_all` 方法将切分后的数据发送至所有Worker并发起远程调用：

在这里，`update_actor` 使用 `execute_all` 将均匀分割的数据分发给所有Worker，并发出远程调用。

```python
class RayWorkerGroup:
  def execute_all(self, method_name: str, *args, **kwargs):
    return self.execute_all_async(method_name, *args, **kwargs)
  def execute_all_async(self, method_name: str, *args, **kwargs):
    length = len(self._workers)
    result = []
    for i in range(length):
      sliced_args = tuple(arg[i] for arg in args)
      sliced_kwargs = {k: v[i] for k, v in kwargs.items()}
      remote_call = getattr(self._workers[i], method_name)
      result.append(remote_call.remote(*sliced_args, **sliced_kwargs))
    return result
```

最终，通过 `ray.get()` 阻塞等待所有任务完成，实现同步执行。

### 6. `func_generator`

`*_fn` 用于 `func_generator` 来生成实际的调用者。

```python
def func_generator(self, method_name,
                    dispatch_fn, collect_fn, execute_fn, blocking):
  def func(*args, **kwargs):
    args, kwargs = dispatch_fn(self, *args, **kwargs)
    output = execute_fn(method_name, *args, **kwargs)
    if blocking:
      output = ray.get(output)
    output = collect_fn(self, output)
    return output
  return func
```

通过这种方式，verl 实现了一个高效且灵活的分布式训练框架，适用于大规模模型的训练与推理任务。

## 3. Multi-Controller 与 SPMD 实现机制

在上一节中，我们介绍了 Verl 框架如何利用 Ray 的 RPC 机制，在 `fit` 函数中通过 **Single Controller** 范式定义 Dataflow Graph。该范式以较低的通信开销换取了极高的编程灵活性，使用户能够清晰地描述 RL 算法的执行流程。

然而，当深入到单个 `Worker Group`（如 Actor 或 Critic）内部时，计算密集型的训练任务需要更高的执行效率。为此，Verl 在 `Worker` 进程内部采用了 **Multi-Controller** 范式，其核心是广泛应用于高性能计算的 **SPMD**（Single Program, Multiple Data）模型。

这种分层设计实现了灵活性与效率的完美权衡：

- **上层（Single Controller）**：关注算法逻辑，实现高灵活性。
- **底层（Multi-Controller / SPMD）**：关注计算性能，实现高效率。

### 3.1 SPMD 编程模型概述

**SPMD** （Single Program, Multiple Data）是现代分布式深度学习框架的核心执行范式。其特点如下：：

- **单一程序**：所有进程执行相同的代码。
- **多份数据**：每个进程处理数据的不同分片。
- **基于环境变量的控制**：通过分布式环境变量（如 `RANK`, `WORLD_SIZE`, `LOCAL_RANK` …）决定每个进程的具体行为。

这是数据并行（Data Parallelism）、张量并行（Tensor Parallelism）等高效并行策略的基础。

#### 典型示例：`torchrun`

`torchrun` 是 SPMD 模式的典型实现。用户仅需提供一份训练脚本，`torchrun` 会：

1. 启动多个进程（数量由 `--WORLD_SIZE` 指定）；
2. 为每个进程设置分布式环境变量（如 `RANK`, `WORLD_SIZE`, `LOCAL_RANK`）；
3. 进程根据 `RANK` 确定自身处理的数据范围。

例如，在数据并行（DDP）中，数据集被划分为 $N$ 份（$N = \text{WORLD\_SIZE}$），第 $i$ 个进程处理第 $i$ 个分片：

$$
\text{Data}_i = \text{Data}[\ i \cdot \frac{\text{len(Data)}}{N} : (i+1) \cdot \frac{\text{len(Data)}}{N}\ ]
$$

#### 主流框架的实现

SPMD 是几乎所有主流分布式训练框架的基础，包括：

- **DDP**（Distributed Data Parallel）：数据并行。
- **ZeRO** 和 **FSDP**（Fully Sharded Data Parallel）：分片数据并行。
- **Megatron-LM** 中的 **Tensor Parallelism** 和 **Pipeline Parallelism**。

这些框架通过 SPMD 模型，利用环境变量协调多个进程，实现高效的模型并行和数据并行。

### 3.2. Verl 中 SPMD 的实现

与 `torchrun` 不同，Verl 作为一个构建在 Ray 之上的新框架，需**自行管理 SPMD 所需的环境配置与调度逻辑**。为降低开发复杂度，Verl 提供了高层抽象接口。下面我们逐步解析其内部实现。

#### 3.2.1 资源配置与环境变量初始化

Verl 通过 `init_with_resource_pool` 函数完成资源分配与环境初始化。其核心步骤如下：

1. **遍历 Placement Group**：每个 Placement Group 对应一个物理节点（Node）。
2. **遍历 Local Rank**：在每个节点内，遍历其 GPU 设备，每个 GPU 对应一个 `local_rank`。
3. **设置环境变量**：为每个 Ray Worker 进程设置关键的分布式环境变量，如 `world_size` 和 `rank`。
4. **存储与实例化**：将环境变量存入 Ray 的运行时上下文（`runtime_env` variables），并实例化对应的 Worker Class。

```python
def init_with_resource_pool(self, resource_pool):
    workers = []
    for node_ip, placement_group in resource_pool.groups.items():
        world_size_on_node = len(placement_group)
        for local_rank in range(world_size_on_node):
            # 1. 构建分布式环境变量
            env_vars = {
                'WORLD_SIZE': str(resource_pool.total_world_size),
                'RANK': str(resource_pool.get_global_rank(node_ip, local_rank)),
                'LOCAL_RANK': str(local_rank),
                'MASTER_ADDR': resource_pool.master_ip,
                'MASTER_PORT': str(resource_pool.master_port),
                # 其他必要环境变量...
            }

            # 2. 创建 Ray Worker 并注入环境变量
            ray_worker = RayWorker.options(
                runtime_env={'env_vars': env_vars}
            ).remote()

            workers.append(ray_worker)

    return workers
```

**关键设计**：

- 每个 GPU 对应一个独立的 Ray Worker 进程；
- 通过 `runtime_env` 将 SPMD 环境变量注入每个进程；
- 全局 `RANK` 通过节点 IP 和 `local_rank` 映射生成。

至此，系统已建立标准的 SPMD 执行环境：每个 GPU 上运行一个进程，且均已配置正确的分布式上下文。

#### 3.2.2 核心执行逻辑与 `@register` 装饰器

用户定义的执行逻辑（如 `update_actor`）通过 `@register` 装饰器进行增强，使其具备分布式调度能力。

以 `Actor` 的 `update_actor` 方法为例，其核心逻辑非常简洁。

```python
@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
def update_actor(self, data: DataProto):
      # 此时 data_proto 已是分割后的本地数据分片
    data = data.to(torch.cuda.current_device())
    self.actor.update_policy(data=data)
    self.actor_lr_scheduler.step()
```

此代码看似同步且简单，实则在底层实现了完整的 SPMD 流程。关键在于 `@register` 装饰器。它解决了“为何数据在进入函数前已被分割”的问题。

#### 3.2.3. `@register` 装饰器的内部机制

`@register` 装饰器的核心作用是为被修饰的函数添加元数据（metadata）属性，并将其包装成一个可被 Multi-Controller 调度的入口。其工作流程如下：

1. **属性注入**：将 `dispatch_mode`、`execute_mode`、`blocking`、`materialize_features` 等配置作为“魔法属性”（magic attribute）附加到函数对象上。
2. **函数包装**：创建一个 `inner` 函数作为实际调用入口。
3. 参数预处理：`inner` 函数首先处理传入的参数：
   - **Materialize Features**：如果 `blocking=True`，则等待所有异步返回的 `Future`（引用）就绪，获取真实数据。
   - **暂存配置**：将装饰器参数暂存，供后续调度逻辑使用。

```python
def register(dispatch_mode=None, execute_mode=None, blocking=True, materialize_features=True):
    def decorator(func):
        @functools.wraps(func)
        def inner(*args, **kwargs):
            # 1. 参数物化（Materialization）
            if materialize_features:
                args = materialize_args(args)
                kwargs = materialize_kwargs(kwargs)
                # 确保所有输入数据（如 Ray ObjectRef）已就绪

            # 2. 返回原始函数调用（实际调度由外部框架完成）
            return func(*args, **kwargs)

        # 3. 附加元数据到函数对象
        inner._dispatch_mode = dispatch_mode
        inner._execute_mode = execute_mode
        inner._blocking = blocking
        # ... 其他属性

        return inner
    return decorator
```

**核心元数据说明**：

| 属性                   | 作用                                     |
| ---------------------- | ---------------------------------------- |
| `dispatch_mode`        | 定义输入数据如何从主进程分发到各 Worker  |
| `execute_mode`         | 定义 Worker 间的执行模式（如并行、串行） |
| `blocking`             | 调用是否阻塞主进程                       |
| `materialize_features` | 是否等待异步输入（如 `ObjectRef`）完成   |

#### 3.2.4   分发模式（Dispatch Mode）的实现

`@register` 只是配置的声明。真正的分发逻辑由框架在调用时触发。

`dispatch_mode` 是连接 **Single-Controller** 与 **Multi-Controller (SPMD)** 两层的关键。其本质是一组预定义的 **分发-收集（Dispatch-Collect）** 协议。

- **映射表维护**：框架维护一个预定义的映射表，将 `dispatch_mode`（如 `"DP"`）映射到具体的 `dispatch function` 和 `collect function`。

  - **分发函数**（`dispatch_fn`）：负责将输入数据从**Single-Controller**（Single Controller）进程分发至多个工作进程（Worker Process）；

  - **收集函数**（`collect_fn`）：在各个工作进程完成计算后，负责将分散的结果聚合为统一输出。

该机制的设计具有良好的扩展性：若需引入新的并行行为，只需定义新的 `dispatch_mode` 及其对应的 `dispatch_fn` 和 `collect_fn` 即可。

- **动态函数合成**：当 Single Controller 调用 `worker_group.update_actor(data)` 时，框架会：

  - 读取 `update_actor` 函数的 `dispatch_mode` 属性。

  - 查找映射表，获取对应的 `dispatch_fn` 和 `collect_fn`。

  - 将原始的 `update_actor` 函数与 `dispatch_fn`、`execute_fn`、`collect_fn` 等结合，动态合成一个全新的、可执行的函数。

#### 分发与收集函数的实现逻辑

##### 数据分发（Dispatch）

以数据并行模式 `DP_COMPUTE_PROTO` 为例，其 `dispatch_fn` 的作用是将输入参数切分为多个子集，以便分发给不同的 Worker。具体流程如下：

1. 接收 `WorkerGroup` 及其全局规模（`world_size`），即参与计算的 GPU 总数；
2. 调用 `_split_args_kwargs_data_proto` 工具函数，对输入参数进行切分。

```python
def _split_args_kwargs_data_proto(chunks, *args, **kwargs):
    splitted_args = []
    for arg in args:
        # 若参数支持 chunk 操作（如张量或 DataProto），则进行分块
        if hasattr(arg, 'chunk'):
            splitted_args.append(arg.chunk(chunks=chunks))
        else:
            # 否则对非数据型参数进行广播（如配置项）
            splitted_args.append([arg] * chunks)
    return splitted_args, kwargs
```

该过程将原始输入数据均匀划分为 `world_size` 个子块，形成参数列表，供后续分发使用。

##### 1. Dispatch-Collect 映射表

Verl 维护一个映射，将 `dispatch_mode` 映射到具体的 `dispatch_fn` 和 `collect_fn`：

```python
DISPATCH_REGISTRY = {
    Dispatch.DP_COMPUTE_PROTO: {
        'dispatch_fn': dispatch_dp_compute_data_proto,
        'collect_fn': collect_dp_compute_data_proto,
    },
    Dispatch.TP_FORWARD: {
        'dispatch_fn': dispatch_tp_forward,
        'collect_fn': collect_tp_forward,
    },
    # ... 其他模式
}
```

##### 2. `dispatch_dp_compute_data_proto` 的实现

以数据并行为例，`dispatch_fn` 负责将主进程的完整数据切分并发送至各 Worker：

```python
def dispatch_dp_compute_data_proto(data: DataProto, world_size: int) -> List[DataProto]:
    # 沿 batch 维度均匀切分
    batch_size_per_gpu = len(data) // world_size
    chunks = []
    for rank in range(world_size):
        start = rank * batch_size_per_gpu
        end = start + batch_size_per_gpu if rank < world_size - 1 else len(data)
        chunk = data.slice(start, end)  # 返回 DataProto 子集
        chunks.append(chunk)
    return chunks  # 返回分片列表 [chunk_0, chunk_1, ..., chunk_{N-1}]
```

##### 3. `collect_dp_compute_data_proto` 的实现

相对应地，`collect_fn` 的行为较为简单，通常是对各 Worker 的输出进行拼接（concatenate）或合并操作：

```python
def collect_dp_compute_data_proto(outputs: List[DataProto]) -> DataProto:
    # 将各 Worker 的输出沿 batch 维度拼接
    return DataProto.concat(outputs)
```

最终，聚合后的结果被返回给Single-Controller，完成一轮分布式计算的闭环。

#### 执行模式（Execute Mode）的调度逻辑

除了 `dispatch_mode`，Verl 还通过 `execute_mode` 控制远程调用的执行方式。系统维护另一组映射，将 `execute_mode` 映射到具体的执行函数名。

以默认模式 `Execute.ALL` 为例，其实际指向 `execute_all_sync`，表示**同步执行所有 Worker 上的方法调用**。

#### 同步执行函数 `execute_all_sync` 的行为

该函数的核心逻辑如下：

1. 遍历 `WorkerGroup` 中的所有 Worker；
2. 将已切分的参数子集与目标方法名（如 `update_actor`、`generate_sequences`）结合；
3. 对每个 Worker 发起远程方法调用（Remote Method Invocation）；
4. 收集所有远程调用的返回值（通常为 `Future` 对象）并返回。

python深色版本

```
# 伪代码示意
def execute_all_sync(worker_group, method_name, splitted_args, splitted_kwargs):
    futures = []
    for i, worker in enumerate(worker_group.workers):
        future = getattr(worker, method_name).remote(
            *splitted_args[i], **splitted_kwargs[i]
        )
        futures.append(future)
    return futures
```

此过程实现了 SPMD（单程序多数据）模型中的并行执行语义。

#### 分布式函数的动态生成机制

Verl 的核心设计之一是通过**函数生成器**（Function Generator）将上述组件动态组合，生成最终可执行的分布式函数。其整合逻辑如下：

1. **获取元信息**：从被 `@register` 装饰的函数中提取 `dispatch_mode`、`execute_mode` 和 `blocking` 等属性；

2. **查找执行策略**：根据属性值查找对应的 `dispatch_fn`、`execute_fn` 和 `collect_fn`；

3. **构建执行流水线**：

   - **步骤一：数据分发**
     使用 `dispatch_fn` 将输入参数切分为 `world_size` 个子集：
     $$
     \text{args}_i, \text{kwargs}_i = \text{dispatch\_fn}(\text{args}, \text{kwargs}), \quad i = 1, \dots, N
     $$
     其中 $N$ 为 `world_size`。

   - **步骤二：并行执行**
     调用 `execute_fn` 在每个 Worker 上执行目标方法，返回 `Future` 列表：
     $$
     \text{futures} = [\text{worker}_i.\text{method}.\text{remote}(\text{args}_i, \text{kwargs}_i) \mid i = 1, \dots, N]
     $$

   - **步骤三：结果物化（Materialization）**
     根据 `blocking` 标志决定是否同步等待所有 `Future` 完成：
     $$
     \text{outputs} =
     \begin{cases}
     \text{ray.get(futures)} & \text{if } \text{blocking} = \text{True} \\
     \text{futures} & \text{otherwise}
     \end{cases}
     $$

   - **步骤四：结果聚合**
     将各 Worker 的输出通过 `collect_fn` 合并为单一结果：
     $$
     \text{result} = \text{collect\_fn}(\text{outputs})
     $$

4. **返回最终函数**：将上述流程封装为一个可调用对象，供控制器直接使用。

#### 3.2.5. 执行流程

结合以上机制，`update_actor` 的调用流程如下：

1. **主进程调用**：`worker_group.update_actor(full_data_batch)`
2. **触发 dispatch**：框架根据 `dispatch_mode` 查找 `dispatch_fn`
3. **数据分发**：`dispatch_dp_compute_data_proto` 将 `full_data_batch` 切分为 $N$ 个子批次
4. **并行执行**：通过 Ray RPC 将子批次发送至各 Worker，触发 `inner()` 函数
5. **参数物化**：`inner()` 确保输入数据就绪（若 `materialize_features=True`）
6. **本地计算**：执行 `update_policy`，完成前向、反向传播
7. **结果收集**：各 Worker 返回梯度/损失等结果
8. **结果聚合**：`collect_fn` 合并结果（如梯度 `all-reduce`）

$$
\text{Grads}_{\text{global}} = \frac{1}{N} \sum_{i=0}^{N-1} \text{Grads}_i
$$

## 4. 总结：分层架构实现效率与灵活性的平衡

Verl 框架通过分层设计，巧妙地结合了 **Single Controller** 的灵活性与 **Multi-Controller/SPMD** 的高效性。

- **上层（Single-Controller）**：
  主进程以直观、同步的方式定义 Dataflow Graph（Dataflow Graph）编写训练逻辑，用户无需关注分布式细节，框架自动处理底层分布式调度。
- **下层（Multi-Controller / SPMD）**：
  在Worker内部，利用成熟的 SPMD 模式实现高效的并行计算（如 DDP、FSDP），最大化硬件利用率。

`@register` 装饰器作为关键抽象，将复杂的分布式调度逻辑（分发、执行、收集）与用户定义的业务逻辑解耦。


### 两层抽象如何协同

| 层次              | 控制器       | 职责                          | 代价                      | 适用场景 |
| ----------------- | ------------ | ----------------------------- | ------------------------- | -------- |
| Single-Controller | 主进程       | 定义 dataflow graph，逻辑清晰 | RPC 通信（轻量）          | 控制逻辑 |
| Multi-Controller  | GPU 进程内部 | SPMD 并行，极限性能           | 环境变量+通信（框架隐藏） | 计算密集 |

这种 **Hybrid Flow** 范式，使得 Verl 既能支持复杂的 RL 训练流水线，又能为用户提供简洁的编程接口，是大规模 LLM 强化学习训练框架设计的典范。
