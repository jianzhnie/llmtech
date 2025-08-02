# Verl 源码解析与 Hybrid Flow 编程范式

本文是对 Verl 框架的源码实现及其背后所依托的 **Hybrid Flow** 编程范式的深入讲解。Hybrid Flow 是 Verl 对应的核心技术，已发表于系统领域顶级会议 EuroSys，旨在为大规模强化学习（Reinforcement Learning, RL）训练提供高效、灵活的分布式执行支持。

我们将从三个部分展开本次讲解：

1. **背景介绍**：形式化地定义 RL 训练中的系统问题；
2. **代码走读（Code Walkthrough）**：以调试器视角，从入口点出发，逐步剖析 Verl 的执行流程；
3. **核心机制解析**：深入探讨 Verl 基于 SPMD 模式的并行计算实现。

---

## 1. 背景：将 RL 形式化为 Dataflow Graph 调度问题

在大型语言模型（LLM）的后训练阶段，强化学习任务本质上是一个复杂的**分布式调度问题**。为了系统性地解决这一问题，Verl 将整个训练流程抽象为一个 **数据流图（Dataflow Graph）**。

### 数据流图的构成要素

一个典型的 RL 训练流程可分解为以下三个核心维度：

1. **多模型（Multiple Models）**
   包括Actor（actor）、Critic（critic）、参考模型（reference model）、奖励模型（reward model）等。这些模型在训练过程中协同工作，各自承担不同职责。

2. **多阶段（Multiple Stages）**
   典型的训练周期包含三个阶段：
   - **生成（Generation）**：使用当前策略生成响应序列；
   - **经验准备（Experience Preparation）**：计算旧策略下的对数概率、价值估计、奖励分数等；
   - **训练（Training）**：基于收集的经验更新 actor 与 critic 模型。

3. **多工作负载（Multiple Workloads）**
   不同模型在不同阶段的工作负载类型各异：
   - 生成阶段以**自回归推理**为主；
   - 训练阶段以**高并行度的前向和反向传播**为主；
   - 推理与训练对并行策略、内存布局和通信模式的需求截然不同。

强化学习（RL）的训练流程可以抽象为一个**有向无环图（DAG）**，记为
$$
\mathcal{G} = (\mathcal{V}, \mathcal{E})
$$

- 顶点 $\mathcal{V}$：模型（actor、critic、reference、reward 等）
- 边 $\mathcal{E}$：数据依赖（experience、梯度、中间激活）

### 从数据流图到执行模式

我们的目标是将上述数据流图高效地映射为 GPU 集群上的**执行模式（Execution Pattern）**。

以 PPO 为例，$\mathcal{G}$ 被天然地划分为 3 个阶段：

| 阶段        | 作用           | GPU 上典型负载                   |
| ----------- | -------------- | -------------------------------- |
| Generation  | 产生 rollout   | actor 前向                       |
| Preparation | 计算优势、奖励 | reward / reference / critic 前向 |
| Training    | 更新策略       | actor & critic 反向              |

每阶段内部又存在多种工作负载（workload），例如：

- actor 在 **Generation** 阶段只做前向，在 **Training** 阶段做反向；
- critic 在 **Preparation** 阶段做前向，在 **Training** 阶段做反向。

在实际部署中，我们需要：

1. **模型放置（Model Placement）**
   将不同模型分配至集群中的不同设备组。例如：
   - Actor 与 Rollout 模型部署在 GPU 0-1；
   - Critic 部署在 GPU 2-3；
   - Reference 与 Reward Model 部署在 GPU 4-5。

2. **执行调度约束**
   - **时序依赖**：生成必须先于经验准备，经验准备必须先于训练；
   - **并行性**：无依赖的阶段（如 critic 与 reward 模型的推理）可并行执行；
   - **资源冲突**：若多个模型共享同一设备组，则必须串行执行以避免资源竞争。

3. **优化目标**
   在满足上述约束的前提下，最大化整体训练吞吐量。


### Verl 的设计愿景：用户只需关注“做什么”

Verl 的终极目标是实现 **“用户只需定义数据流图，框架自动完成分布式优化”** 的愿景。即：

- 用户仅需在数学层面定义 RL 算法的行为（如损失函数、策略更新规则）；
- 框架自动处理底层的分布式并行策略、通信优化、内存管理等复杂细节。

虽然这一理想仍在演进中，但 Verl 通过 **Hybrid Flow** 范式，结合 **Single-Controller** 与 **Multi-Controller** 机制，在**灵活性**与**效率**之间取得了良好平衡。

## 2. 从入口到 Single-Controller 主循环

Verl 是一个包含数万行代码的复杂系统，我们将聚焦其核心设计逻辑，以简化后的代码示例深入剖析其执行流程。我们将从入口点开始，逐步解析资源管理、工作负载调度与并行执行机制。

### 2.1 入口：`main.py → TaskRunner.run`

当用户通过命令行启动训练任务时，程序首先进入 `main` 模块，并调用 `TaskRunner.run()` 作为间接入口。该函数的核心职责是在执行训练循环（`fit`）前完成必要的初始化，主要包括**资源池（Resource Pool）的定义与映射**。

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

### 2.2 Worker 初始化

在进入 `fit()` 训练循环前，`RayPPOTrainer` 需完成 Worker（Worker）的初始化。此过程的关键在于解决 PyTorch 框架下的**显存碎片化**问题。

#### 2.2.1. 显存碎片化的根源

PyTorch 的显存管理器（如 CUDA caching allocator）为每个进程预留（reserve）显存池以提高分配效率。然而，不同进程间的显存无法共享。若多个进程（如 Actor、Critic）分别启动，即使它们不同时运行，各自预留的显存也无法被对方利用，导致总显存占用远超峰值需求，形成严重浪费。

#### 2.2.2. 进程共置（Process Collocation）解决方案

Verl 采用 **进程共置** 策略，即在每个 GPU 上仅维护**一个进程**，该进程在不同阶段承载不同的工作负载（如先运行 Actor 生成，再运行 Critic 推理）。

#### 2.2.3 实现机制：动态类合成

为实现单进程运行多角色，Verl 将多个Worker类（如 `ActorRolloutWorker`, `CriticWorker`）的方法**动态融合**到一个 `WorkerGroup` 类中：

```python
# 伪代码：WorkerGroup 类融合多个角色的方法
class WorkerGroup(ActorRolloutWorker, CriticWorker, RefPolicyWorker, RewardModelWorker):
    pass  # 拥有所有基类的方法
```

通过此方式，单个进程即可执行所有任务，显存仅需按**单个最大工作负载**预留，显著提升显存利用率。

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

##  3. 训练循环：Single-Controller 范式

初始化完成后，程序进入 `fit()` 函数，执行核心训练循环。Verl 采用 **Single-Controller** 范式，由主进程（Controller）协调所有分布式操作，用户代码聚焦于数据流逻辑，无需关心底层分布式细节。

### 1. 同步执行流程

在默认的全局资源池配置下，各阶段串行执行，逻辑清晰：

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

该循环清晰地体现了数据流图的三个阶段，执行顺序严格遵循依赖关系，确保正确性。

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



### 7. 通信开销分析

`Trainer.fit()` 使用 **Single-Controller** 编程模型, 主进程（controller）拥有全局视角通过 Ray 的 RPC（远程过程调用）与Worker进程通信，传递 `DataProto` 对象。尽管进程间通信（IPC）存在开销，但其影响有限，原因如下：

- **传输数据量小**：传递的主要是 `prompt`、`response`、`log_probs`、`rewards` 等标量或小张量，远小于模型参数、隐藏状态（hidden states）或优化器状态。
- **计算密集型主导**：各阶段的计算耗时远超通信耗时，通信开销被有效掩盖。

> **权衡取舍**：通过引入轻量级通信，换取了编程模型的极大灵活性，用户无需手动管理分布式同步与通信。

## 4. Worker内部：Multi-Controller (SPMD) 范式

当主进程通过 RPC 触发 `worker_group.method()` 时，执行进入Worker内部。此时，Verl 切换至 **Multi-Controller** 范式，即 **SPMD（Single Program, Multiple Data）**，以最大化计算效率。

### 1. SPMD 编程模型概述

**SPMD** 是现代分布式深度学习框架的核心执行范式。所有进程（同一 `WorkerGroup` 内的多个 GPU 进程）执行**相同的程序**，但处理**不同的数据分片**。其核心思想是：

- **多个进程**并行执行；
- 所有进程运行**相同的程序代码**；
- 每个进程处理**不同的数据分片**。
- 通过分布式环境变量（`RANK`, `WORLD_SIZE`, `LOCAL_RANK` …）决定**数据分片**与**行为差异**；

这是数据并行（Data Parallelism）、张量并行（Tensor Parallelism）等高效并行策略的基础。

#### 典型示例：`torchrun`

`torchrun` 是 SPMD 模式的典型实现。用户仅需提供一份训练脚本，`torchrun` 会：

1. 启动多个进程（数量由 `--nproc_per_node` 指定）；
2. 为每个进程设置分布式环境变量（如 `RANK`, `WORLD_SIZE`, `LOCAL_RANK`）；
3. 进程根据 `RANK` 确定自身处理的数据范围。

例如，在数据并行（DDP）中，数据集被划分为 $N$ 份（$N = \text{WORLD\_SIZE}$），第 $i$ 个进程处理第 $i$ 个分片：

$$
\text{Data}_i = \text{Data}[\ i \cdot \frac{\text{len(Data)}}{N} : (i+1) \cdot \frac{\text{len(Data)}}{N}\ ]
$$

所有主流分布式训练技术（如 DDP、ZeRO、FSDP、Megatron 的 Tensor Parallelism 和 Pipeline Parallelism）均基于 SPMD 模型构建。

### 2. 隐藏的复杂性

SPMD 编程模型较为复杂（需手动管理 `RANK`、`WORLD_SIZE`、分布式通信原语等）。Verl 将此复杂性封装在 `WorkerGroup` 内部，对外暴露简洁的同步接口。例如，`update_actor()` 方法内部自动完成：

1. **数据分发**：将输入 `batch` 按设备数 $N$ 均匀切分：$[x_1, x_2, ..., x_N]$。
2. **并行计算**：各 GPU 计算 $\text{forward}(x_i)$。
3. **结果聚合**：收集梯度并执行 `all-reduce`，更新全局模型参数。

$$
\Delta \theta = \frac{1}{N} \sum_{i=1}^{N} \nabla_\theta \mathcal{L}(x_i; \theta)
$$

---

### 3. Verl 中的 SPMD 实现：从资源分配到执行调度

与 `torchrun` 不同，Verl 作为一个构建在 Ray 之上的新框架，需**自行管理 SPMD 所需的环境配置与调度逻辑**。为降低开发复杂度，Verl 提供了高层抽象接口。下面我们逐步解析其内部实现。

#### 3.1. 资源配置：初始化分布式环境

Verl 通过 `init_with_resource_pool` 函数完成资源分配与环境初始化。其核心步骤如下：

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

---

#### 3.2. 执行抽象：`@register` 装饰器详解

用户定义的执行逻辑（如 `update_actor`）通过 `@register` 装饰器进行增强，使其具备分布式调度能力。

##### 3.2.1. 用户代码示例

```python
@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
def update_actor(self, data: DataProto):
    data = data.to(torch.cuda.current_device())
    self.actor.update_policy(data=data)
    self.actor_lr_scheduler.step()
```

此代码看似同步且简单，实则在底层实现了完整的 SPMD 流程。

##### 3.2.2. `@register` 装饰器的工作原理

`@register` 的作用是为被修饰函数附加元数据（metadata），并将其包装为可调度的分布式任务。其核心逻辑如下：

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

#### 3.3 分布式调度的核心：Dispatch 与 Collect

`dispatch_mode` 是连接 **Single-Controller** 与 **Multi-Controller (SPMD)** 两层的关键。其本质是一组预定义的 **分发-收集（Dispatch-Collect）** 协议。

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

`collect_fn` 负责收集各 Worker 的输出并合并：

```python
def collect_dp_compute_data_proto(outputs: List[DataProto]) -> DataProto:
    # 将各 Worker 的输出沿 batch 维度拼接
    return DataProto.concat(outputs)
```

### 4. 执行流程

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

## 总结：分层架构实现效率与灵活性的平衡

Verl 通过**分层执行架构**，巧妙地平衡了系统效率与开发灵活性：

- **上层（Single-Controller）**：
  主进程以直观、同步的方式定义数据流图（Dataflow Graph）编写训练逻辑，用户无需关注分布式细节，框架自动处理底层分布式调度。

- **中层（Dispatch-Collect）**：通过 `@register` 装饰器和预定义DataProto协议，将复杂的分发与收集逻辑模板化。

- **下层（Multi-Controller / SPMD）**：
  在Worker内部，利用成熟的 SPMD 模式实现高效的并行计算（如 DDP、FSDP），最大化硬件利用率。

- **核心优化**：
  通过 **进程共置（Process Collocation）** 解决显存碎片化问题，在单进程内融合多角色，显著提升资源效率。


### 两层抽象如何协同

| 层次              | 控制器       | 职责                          | 代价                      | 适用场景 |
| ----------------- | ------------ | ----------------------------- | ------------------------- | -------- |
| Single-Controller | 主进程       | 定义 dataflow graph，逻辑清晰 | RPC 通信（轻量）          | 控制逻辑 |
| Multi-Controller  | GPU 进程内部 | SPMD 并行，极限性能           | 环境变量+通信（框架隐藏） | 计算密集 |

这种 **Hybrid Flow** 范式，使得 Verl 既能支持复杂的 RL 训练流水线，又能为用户提供简洁的编程接口，是大规模 LLM 强化学习训练框架设计的典范。
