# Verl 源码解析与 Hybrid Flow 编程范式

`Verl` 是发表于系统领域顶级会议 EuroSys 的论文 [HybridFlow](https://arxiv.org/abs/2409.19256v2) 的开源实现。本文将深入解析 Verl 框架的源码实现及其背后的编程范式 HybridFlow 的核心概念、设计动机，以及如何使用 `verl` 提供的 API 进行编程开发。

- 论文链接：https://arxiv.org/pdf/2409.19256v2
- 代码链接：https://github.com/volcengine/veRL
- 在线文档：https://verl.readthedocs.io/en/latest/index.html

## 1. 动机与设计

在 `verl` 的设计中，我们采用**数据流**（Dataflow）的方式对强化学习系统进行建模。

### 1.1 数据流模型

数据流是计算过程的一种抽象表示。神经网络的训练过程即是一个典型的数据流，通常可用计算图（Computation Graph）来描述。

![The dataflow graph from CS231n 2024 lecture 4](https://github.com/eric-haibin-lin/verl-community/blob/main/docs/dataflow.jpeg?raw=true)

如图所示，该计算图表示一个多项式函数后接 S 型激活函数的结构。在神经网络的计算流中，每个节点代表一个基本运算符（如加法、矩阵乘法等），每条边表示数据（张量）在前向或反向传播中的流动方向。整个计算图决定了神经网络的拓扑结构。

### 1.2 强化学习作为数据流问题

#### 1.2.1 RLHF 的核心流程拆解

以典型的基于近端策略优化（PPO）算法为例，一个 RLHF 系统分解为以下三个核心维度：

##### 多模型（Multiple Models）

PPO 算法通常包含四个核心模型：

- **Actor 模型**：生成响应（Response）
- **Critic 模型**：估计状态价值
- **Reference Policy 模型**：提供生成模型的参考分布
- **Reward Model**：对生成的回复进行打分

##### 多阶段（Multiple Stages）

整个 RLHF 流程以迭代方式进行，每轮包含三个阶段：

- **生成阶段（Generation/Rollout）**：Actor 模型对一批 Prompt 进行自回归生成，产出 Response

- **经验准备（Experience Preparation）**：使用提示和生成的回复，由 Critic、Reference Policy 和 Reward Model 分别计算价值、对数概率和奖励，构建训练数据：
  - Actor Model：计算新旧策略下的对数概率
  - Critic Model：计算生成回复的值（values）
  - Reference Model：计算生成回复的参考对数概率，它通常是 Actor 模型在 RLHF 之前的版本，用于限制 Actor 模型在训练过程中不会偏离过远
  - Reward Model：计算生成回复的奖励（rewards）。奖励模型通常是一个基于人类偏好数据进行微调的 LLM，其语言建模头被替换为标量输出头
- **训练阶段（Training）**：基于上述数据，通过前向与反向传播更新 Actor 和 Critic 模型

##### 多种工作负载（Multiple Workloads）

不同模型在不同阶段的工作负载类型各异：

- 生成阶段以**自回归推理**为主
- 训练阶段以**高并行度的前向和反向传播**为主
- 推理与训练对并行策略、内存布局和通信模式的需求截然不同

#### 1.2.2 RLHF 建模为数据流图

从系统视角看，RLHF 可建模为一个复杂的**分布式数据流图**：

- **节点**：代表一个 LLM 的分布式训练或推理任务
- **边**：表示节点间的数据依赖与通信，常涉及多对多的模型分片重分布（re-sharding）

下图展示了在 RLHF（基于人类反馈的强化学习）中常用的 PPO 算法的数据流图：

<img src="https://picx.zhimg.com/70/v2-cb8ab5ee946a105aab6a563e92682ffa_1440w.avis?source=172ae18b&biz_tag=Post" alt="PPO dataflow graph, credit to Zhihu 低级炼丹师" style="zoom:50%;" />

然而，强化学习的数据流与传统神经网络训练的数据流存在本质区别：

| 工作负载（Workload） | 节点（Node）                        | 边（Edge）         |
| -------------------- | ----------------------------------- | ------------------ |
| 神经网络训练         | 基本运算符（如 +/-/matmul/softmax） | 张量（Tensor）流动 |
| 强化学习（RL）       | 高层算子（如 rollout/模型前向传播） | 数据（Data）移动   |

在传统表格型强化学习中，每个算子通常是简单的标量运算（如贝尔曼方程更新）。而在深度强化学习（DRL）中，每个算子本身就是一个复杂的神经网络计算过程，如模型推理或参数更新。这使得深度强化学习呈现出**双层数据流**的特性：

- **控制流**（Control Flow）：定义高层算子的执行顺序与逻辑。例如，在 PPO 算法中，依次执行轨迹采样、优势函数计算、策略更新等步骤。控制流体现了**强化学习算法的核心逻辑**
- **计算流**（Compute Flow）：定义**神经网络内部**的计算过程，如前向传播、反向传播、优化器更新等

### 1.3 设计选择

在大语言模型（LLM）兴起之前，深度强化学习所使用的模型规模较小，其计算流通常可以在单个进程中完成。因此，将控制流与计算流集成在单一进程中是可行的。

然而，在 LLM 时代，计算流（如大规模模型的训练）必须依赖多进程并行计算。这催生了下面三种系统架构设计方案：

#### 1.3.1 Single-Controller 模式

Single-Controller 架构采用集中式控制器统一调度所有计算任务。

- **优势**：
  - 具备全局视图，便于资源分配与任务编排
  - 用户可将整个数据流视为单个进程，逻辑清晰

- **劣势**：
  - 控制器与大量工作节点通信会产生显著调度延迟，尤其在大规模集群中

#### 1.3.2 Multi-Controller 模式

将控制流也转化为多进程程序，与计算流协同部署。

- **优势**：
  - 在固定控制流和计算流的场景下，能够实现**最优性能**，因为训练过程中的通信开销被最小化

- **劣势**：
  - 从软件工程角度看，计算代码与控制器逻辑高度耦合，导致**计算流和控制流难以复用**。例如，若已实现一个基于 FSDP 的 PPO 训练流程，当需要切换到 Megatron 计算后端时，由于控制流与计算流的耦合，两者均难以直接复用
  - 多进程控制流在面对动态或复杂的控制逻辑时，开发和维护成本较高

#### 1.3.3 分离式架构

控制流在单进程中运行，计算流在多进程中运行。

- **优势**：
  - 实现了控制流与计算流的解耦，使得**计算流程可以轻松复用**于不同的控制逻辑
  - 控制器运行在单一进程中，**实现新的强化学习算法更加简单灵活**

- **劣势**：
  - 每次控制器与计算节点交互时，都会引入额外的**数据通信开销**，数据需要在控制进程与计算进程之间来回传输

`verl` 采用了第 3 种分离式架构，其核心目标是将强化学习算法的**控制流**与底层**计算引擎**的实现进行解耦，从而提升系统的模块化和可扩展性。

### 1.4 Verl 设计目标与执行流程

#### 1.4.1 PPO 执行过程与约束

| 阶段        | 作用           | GPU 上典型负载                   |
| ----------- | -------------- | -------------------------------- |
| Generation  | 产生 rollout   | Actor 前向                       |
| Preparation | 计算优势、奖励 | Reward / Reference / Critic 前向 |
| Training    | 更新策略       | Actor & Critic 反向              |

每阶段内部又存在多种工作负载（Workload），例如：

- Actor 在 **Generation** 阶段只做前向，在 **Training** 阶段做反向
- Critic 在 **Preparation** 阶段做前向，在 **Training** 阶段做反向

在实际部署中，我们需要：

- **模型放置（Model Placement）**
  将不同模型分配至集群中的不同设备组。例如：
  - Actor 与 Rollout 模型部署在 GPU 0-1
  - Critic 部署在 GPU 2-3
  - Reference 与 Reward Model 部署在 GPU 4-5

- **执行调度约束**
  - **时序依赖**：生成必须先于经验准备，经验准备必须先于训练
  - **并行性**：无依赖的阶段（如 Critic 与 Reward 模型的推理）可并行执行
  - **资源冲突**：部署在同一设备上的多个模型需串行执行，以避免资源竞争

**优化目标**：
在满足上述约束的前提下，最大化整体训练吞吐量。

#### 1.4.2 Verl 的设计目标

Verl 的设计目标是实现 **"用户只需定义 Dataflow Graph，框架自动完成分布式优化"** 的愿景。即：

- 用户仅需在数学层面定义 RL 算法的行为（如损失函数、策略更新规则）
- 框架自动处理底层的分布式并行策略、通信优化、内存管理等复杂细节

Verl 通过 **Hybrid Flow** 范式，结合 **Single-Controller** 与 **Multi-Controller** 机制，在**灵活性**与**效率**之间取得了良好平衡。

#### 1.4.3 Verl 中的执行流程

下图展示了 `verl` 中强化学习任务的简化执行流程：

![The execution diagram](https://github.com/eric-haibin-lin/verl-community/blob/main/docs/driver_worker.png?raw=true)

在该架构中，**控制器**（Controller，又称驱动进程）运行于单个进程中，而 Actor、Critic 等 Workers 则分布于多个 GPU 进程中，并以**Workers 组**（WorkerGroup）的形式进行管理。

在轨迹采样阶段，控制器将提示（prompt）数据分发给 generator WorkerGroup，各节点并行生成响应。采样完成后，生成的数据被回传至控制器，由其执行后续的算法步骤（如优势计算、奖励计算等）。其他计算任务（如价值估计、奖励打分）遵循类似的交互模式。

通过这种混合式设计，`verl` 成功实现了数据流与计算过程的解耦，在保证计算效率的同时，为强化学习训练循环的灵活定义提供了支持。

## 2. Verl 代码解析

Verl 是一个包含数万行代码的复杂系统，我们将聚焦其核心设计逻辑，以简化后的代码示例深入剖析其执行流程。我们将从入口点开始，逐步解析资源管理、工作负载调度与并行执行机制。

### 2.1 入口函数

代码位置：[main_ppo.py](https://github.com/volcengine/verl/blob/main/verl/trainer/main_ppo.py)

`main_ppo.py` 是 Verl 框架中 PPO（Proximal Policy Optimization）训练的主入口点，负责初始化 Ray 分布式计算环境并启动分布式 PPO 训练流程。该文件采用 Hydra 配置管理系统，支持灵活的参数配置和多种推理后端（VLLM、SGLang）。

该文件中定义的 [main_task](verl/verl/trainer/main_generation.py#L62-L148) 函数即为上文所述的**控制器进程**（驱动进程）。同时，[RewardManager](verl/examples/split_placement/main_ppo_split.py#L36-L89) 类允许用户基于具体数据集自定义奖励函数。需要注意的是，[RewardManager](verl/examples/split_placement/main_ppo_split.py#L36-L89) 应返回 RL 算法优化所需的最终 token 级奖励，用户可结合基于模型的奖励与基于规则的奖励。

该代码采用 Ray 框架实现分布式计算，通过 TaskRunner 远程 Actor 模式将控制流与计算流分离。**需要特别强调的是，[main_task](verl/examples/split_placement/main_ppo_split.py#L105-L207) 以单进程模式运行**。建议避免将 [main_task](verl/examples/split_placement/main_ppo_split.py#L105-L207) 调度在 Ray 集群的头节点上，因为该进程会消耗大量内存，而头节点通常资源有限。这种设计使得算法控制逻辑运行在单进程中，而计算密集型任务分布在多个 GPU 节点上。

#### 2.1.1 TaskRunner 执行逻辑

TaskRunner 作为 Ray 远程 Actor 执行主要的训练协调工作：main_ppo.py:86-93

1. **模型路径处理**：下载 Checkpoint 从 HDFS 到本地机器，支持共享内存加速 main_ppo.py:117-119
2. **分词器和处理器初始化**：实例化 tokenizer 和 processor，支持多模态 LLM main_ppo.py:124-127
3. **Worker 类选择**：根据策略选择不同的 Worker 实现：main_ppo.py:129-168
   - FSDP 策略：使用 [fsdp_workers](verl/verl/workers/fsdp_workers.py#L0-L0) 中的 Worker 类
   - Megatron 策略：使用 [megatron_workers](verl/verl/workers/megatron_workers.py#L0-L0) 中的 Worker 类
4. **角色映射和资源池配置**：定义角色到 Worker 类的映射，配置资源池规格 main_ppo.py:172-187
5. **奖励模型配置**：如果启用奖励模型，根据策略选择相应的 RewardModelWorker main_ppo.py:195-203

#### 2.1.2 全局资源池配置

在默认配置中，Verl 使用统一的资源池（Resource Pool）管理所有 GPU 资源，并将不同角色的 Worker 映射至对应资源池。

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
    resource_pool_manager=resource_pool_manager,
    ...
)
trainer.fit()
```

> **设计考量**：所有 WorkLoad 共享同一资源池意味着它们在时间上**串行执行**。尽管这牺牲了部分并行潜力，但在多数场景下，由于各阶段（生成、经验准备、训练）本身存在强时序依赖，串行执行反而能有效利用全部 GPU 资源，避免因资源碎片化导致的利用率下降。

### 2.2 Ray Trainer

代码位置：[ray_trainer.py](https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/ray_trainer.py)

`ray_trainer.py` 实现了 [RayPPOTrainer](verl/verl/trainer/ppo/ray_trainer.py#L293-L1388) 类，这是 VERL 框架中 PPO 算法的分布式训练调度器。[RayPPOTrainer](verl/verl/trainer/ppo/ray_trainer.py#L293-L1388) 类主要负责：

- 管理分布式 Worker 组及其组的构建
- 协调训练流程、处理数据流转
- 实现完整的 PPO 训练循环

**注意**：[RayPPOTrainer](verl/verl/trainer/ppo/ray_trainer.py#L293-L1388) 的 [fit](verl/verl/trainer/ppo/ray_trainer.py#L1030-L1388) 函数同样以**单进程**形式运行，将算法控制逻辑与底层计算分离，确保控制逻辑的集中管理。

#### 2.2.1 角色定义系统

定义了 [Role](verl/verl/trainer/ppo/ray_trainer.py#L66-L77) 枚举类，用于标识不同的 Worker 角色：

- [Actor](verl/verl/trainer/ppo/ray_trainer.py#L71-L71)：策略模型，负责生成动作
- [Rollout](verl/verl/trainer/ppo/ray_trainer.py#L72-L72)：序列生成，负责采样
- [ActorRollout](verl/verl/trainer/ppo/ray_trainer.py#L73-L73)：合并的 Actor 和 Rollout 角色
- [Critic](verl/verl/trainer/ppo/ray_trainer.py#L74-L74)：价值函数模型
- [RefPolicy](verl/verl/trainer/ppo/ray_trainer.py#L75-L75)：参考策略，用于 KL 散度计算
- [RewardModel](verl/verl/trainer/ppo/ray_trainer.py#L76-L76)：奖励模型
- [ActorRolloutRef](verl/verl/trainer/ppo/ray_trainer.py#L77-L77)：三合一角色，包含 Actor、Rollout 和 Reference

#### 2.2.2 资源池管理器

`ResourcePoolManager` 类管理 Ray 集群的 GPU 资源分配：

- **第 91-108 行**：`create_resource_pool()` 方法根据配置创建资源池，支持 FSDP（`max_colocate_count=1`）和 Megatron（`max_colocate_count>1`）两种模式
- **第 119-150 行**：`_check_resource_available()` 方法验证集群资源是否满足训练需求，确保每个资源池都能获得足够的 GPU

支持两种资源管理模式：FSDP 模式使用 `max_colocate_count=1` 合并所有 WorkerGroup，Megatron 模式使用 `max_colocate_count>1` 支持不同的模型并行策略。ray_trainer.py:100-103

#### 2.2.3 优势计算函数

[compute_advantage()](verl/verl/trainer/ppo/ray_trainer.py#L118-L175) 函数实现多种优势估计算法：

集成了 GAE、GRPO、REINFORCE++ 等多种优势估计算法，通过统一接口支持不同的强化学习变体。

### 2.3 Worker 与 WorkerGroup

#### 2.3.1 PyTorch 显存碎片化问题

PyTorch 框架的显存管理机制（如 CUDA caching allocator）存在一个关键限制：**进程间无法共享显存**。

PyTorch 的显存管理器为每个进程预留（reserve）显存池以提高分配效率。然而，不同进程间的显存无法共享。若多个进程（如 Actor、Critic）分别启动，即使它们不同时运行，各自预留的显存也无法被对方利用，导致总显存占用远超峰值需求，形成严重浪费。

为解决此问题，Verl 采用 **进程共置** 策略：

- 在每个 GPU 上仅维护**一个进程**
- 将不同 Workload 的逻辑融合到同一个进程中，使其在不同时间执行不同任务（如先运行 Actor 生成，再运行 Critic 推理）

#### 2.3.2 WorkerGroup 与进程共置

在 [Trainer](torchtitan/torchtitan/train.py#L35-L530) 初始化阶段，关键操作是 **Worker 分组**：

- 将不同 workload（如 Actor、Critic、Reference）分组为若干个 `Worker Group`
- 每个 `Worker Group` 对应一个资源池，包含一个或多个 GPU

每个 [WorkerGroup](verl/verl/single_controller/base/worker_group.py#L122-L251) 管理一组远程运行的 Worker。WorkerGroup 在其创建进程中运行，而组内的每个 Worker 则运行在独立的 GPU 上。[WorkerGroup](verl/verl/single_controller/base/worker_group.py#L122-L251) 作为控制器与 Worker 之间的代理，负责执行具体的计算任务。

为了将 Worker 的方法暴露给控制器，并定义数据的分发与收集机制，`verl` 采用了一种基于装饰器的简洁设计，具体将在"Worker 定义"一节中详述。

在 PPO 算法中，主要定义了以下三类 WorkerGroup：

- **ActorRolloutRef**：管理 Actor（Actor）、轨迹采样（Rollout）和参考策略（Reference Policy）。[ActorRolloutRefWorker](verl/verl/workers/fsdp_workers.py#L135-L1078) 可以实例化为单一角色（如仅 Actor），也可组合多个角色（如 Actor+采样器+参考策略）。这种设计旨在最大化代码复用性。将 Actor 与采样器共置，是为了利用 NCCL 实现高效的权重同步；将参考策略与 Actor 共置，则是为了支持高效的 LoRA-PPO——在 LoRA 微调中，参考策略通常对应基础模型。共置功能通过 `verl.single_controller.ray.base.create_colocated_Worker_cls` 装饰器实现，该装饰器会创建一个暴露所有角色方法的 Ray 远程类
- **Critic**：管理 Critic 模型
- **Reward**：管理 RewardModel（Reward Model）

这些 WorkerGroup 将在指定的资源池（即 Ray 集群中的 GPU 集合）上进行部署。

为实现单进程运行多角色，Verl 将多个 Worker 类（如 [ActorRolloutRef](verl/verl/trainer/ppo/ray_trainer.py#L77-L77)）的方法动态融合到一个 [WorkerGroup](verl/verl/single_controller/base/worker_group.py#L122-L251) 类中。

- 能够最大化代码复用性
- 通过此方式，新类拥有所有子类的方法，单个进程即可执行所有任务，显存仅需按**单个最大工作负载**预留，显著提升显存利用率，有效避免了显存碎片化

#### 2.3.3 WorkerGroup 的创建

每个 [WorkerGroup](verl/verl/single_controller/base/worker_group.py#L122-L251) 对应一个资源池，内部包含多个运行在 GPU 上的 Ray Worker 进程。

```python
for resource_pool, class_dict in self.resource_pool_to_cls.items():
    worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
    wg_dict = self.ray_worker_group_cls(
      resource_pool=resource_pool,
      ray_cls_with_init=worker_dict_cls,
      **wg_kwargs,
    )
    spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
    all_wg.update(spawn_wg)
```

`spawn()` 方法为每个 GPU 启动一个独立进程，并设置 SPMD 环境变量（如 `RANK`、`WORLD_SIZE`）。

#### 2.3.4 Worker 定义

以 [ActorRolloutRefWorker](https://github.com/volcengine/verl/blob/main/verl/Workers/fsdp_Workers.py) 为例，其向控制器暴露的 API 包括：

- `init_model`：构建底层模型
- `generate_sequences`：根据输入提示生成响应序列
- `compute_log_prob`：使用 Actor 模型计算生成序列的对数概率
- `compute_ref_log_prob`：使用参考策略计算生成序列的对数概率
- `save_checkpoint`：保存模型 Checkpoint

这些方法定义在工作进程中，只能通过远程调用（Remote Call）执行。例如，控制器初始化模型时需调用：

```python
for worker in Actor_rollout_ref_wg:
    worker.init_model.remote()
```

若需生成序列，控制器需执行如下代码：

```python
data = xxx
# 将数据划分为数据并行（DP）分块
data_dp_lst = data.split(dp_size)
output_dp_lst = []
for i, Worker in enumerate(Actor_rollout_ref_wg):
    output_future = Worker.generate_sequences.remote(data_dp_lst[i])
    output_dp_lst.append(output_future)
output = torch.cat(ray.get(output_dp_lst), dim=0)
```

可以观察到，控制器调用 WorkerGroup 方法的通用模式包含三个步骤：

1. 将输入数据按数据并行规模划分
2. 将划分后的数据分发给各个 Worker
3. 收集各节点的计算结果并拼接为完整输出

为简化这一流程，`verl` 提供了语法糖机制，将上述三步封装为一次调用：

```python
@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
def generate_sequences(data):
    ...

# 在控制器中
output = Actor_rollout_ref_wg.generate_sequences(data)
```

通过 `@register` 装饰器，开发者可以明确指定输入数据的分发方式及输出数据的收集策略。例如，`Dispatch.DP_COMPUTE_PROTO` 表示将输入数据按数据并行方式分割，分发至各 Worker，最终收集并拼接输出。需要注意的是，此类方法的输入输出需符合 `DataProto` 协议格式（详见 [protocol.py](https://github.com/volcengine/verl/blob/main/verl/protocol.py)）。

### 2.4 核心执行：Single-Controller 范式

初始化完成后，程序进入 `fit()` 函数，执行核心训练循环。Verl 采用 **Single-Controller** 范式，由主进程（Controller）协调所有分布式操作，用户代码聚焦于数据流逻辑，无需关心底层分布式细节。

#### 2.4.1 同步执行流程

在全局资源池的默认配置下，[fit](verl/verl/trainer/ppo/ray_trainer.py#L1030-L1388) 函数采用同步逻辑。以 PPO 为例，其核心流程如下：

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

该循环清晰地体现了 Dataflow Graph 的三个阶段，执行顺序严格遵循依赖关系，确保正确性。

#### 2.4.2 关键设计优势

- **计算后端可切换性**：该编程范式使用户无需修改控制流程即可灵活切换不同的计算后端（如 FSDP、Megatron）
- **部署灵活性**：通过调整 [WorkerGroup](verl/verl/single_controller/base/worker_group.py#L122-L251) 与 [ResourcePool](verl/verl/single_controller/base/worker_group.py#L26-L72) 的映射关系，可在不修改控制逻辑的前提下实现灵活的资源部署策略

#### 2.4.3 通信开销分析

`Trainer.fit()` 使用 **Single-Controller** 编程模型，主进程（controller）拥有全局视角通过 Ray 的 RPC（远程过程调用）与 Worker 进程通信，传递 `DataProto` 对象。尽管进程间通信（IPC）存在开销，但其影响有限，原因如下：

- **传输数据量小**：传递的主要是 [prompt](ROLL/roll/utils/prompt.py#L0-L0)、[response](openai-python/examples/demo.py#L41-L49)、`log_probs`、[rewards](open-r1/src/open_r1/rewards.py#L0-L0) 等标量或小张量，远小于模型参数、隐藏状态（hidden states）或优化器状态
- **计算密集型主导**：各阶段的计算耗时远超通信耗时，通信开销被有效掩盖

> **权衡取舍**：通过使用 Ray 的 RPC 框架传输控制信息，Verl 以较小的通信代价换取了编程模型的极大灵活性，用户无需手动管理分布式同步与通信。

## 3. Multi-Controller 与 SPMD 实现机制

在上一节中，我们介绍了 Verl 框架如何利用 Ray 的 RPC 机制，在 [fit](verl/verl/trainer/ppo/ray_trainer.py#L1030-L1388) 函数中通过 **Single Controller** 范式定义 Dataflow Graph。该范式以较低的通信开销换取了极高的编程灵活性，使用户能够清晰地描述 RL 算法的执行流程。

然而，当深入到单个 `Worker Group`（如 Actor 或 Critic）内部时，计算密集型的训练任务需要更高的执行效率。为此，Verl 在 [Worker](file:///Users/jianzhengnie/work_dir/rlcode/zero/ray/java/test/src/main/java/io/ray/test/StressTest.java#L50-L69) 进程内部采用了 **Multi-Controller** 范式，其核心是广泛应用于高性能计算的 **SPMD**（Single Program, Multiple Data）模型。

这种分层设计实现了灵活性与效率的完美权衡：

- **上层（Single Controller）**：关注算法逻辑，实现高灵活性
- **底层（Multi-Controller / SPMD）**：关注计算性能，实现高效率

### 3.1 SPMD 编程模型概述

**SPMD**（Single Program, Multiple Data）是现代分布式深度学习框架的核心执行范式。其特点如下：

- **单一程序**：所有进程执行相同的代码
- **多份数据**：每个进程处理数据的不同分片
- **基于环境变量的控制**：通过分布式环境变量（如 [RANK](MindSpeed/mindspeed/log_config.py#L10-L10)、[WORLD_SIZE](MindSpeed/mindspeed/core/multi_modal/dist_train/dist_train_config.py#L29-L29)、[LOCAL_RANK](MindSpeed/mindspeed/log_config.py#L11-L11) 等）决定每个进程的具体行为

这是数据并行（Data Parallelism）、张量并行（Tensor Parallelism）等高效并行策略的基础。

#### 典型示例：`torchrun`

`torchrun` 是 SPMD 模式的典型实现。用户仅需提供一份训练脚本，`torchrun` 会：

1. 启动多个进程（数量由 `--WORLD_SIZE` 指定）
2. 为每个进程设置分布式环境变量（如 [RANK](MindSpeed/mindspeed/log_config.py#L10-L10)、[WORLD_SIZE](MindSpeed/mindspeed/core/multi_modal/dist_train/dist_train_config.py#L29-L29)、[LOCAL_RANK](MindSpeed/mindspeed/log_config.py#L11-L11)）
3. 进程根据 [RANK](MindSpeed/mindspeed/log_config.py#L10-L10) 确定自身处理的数据范围

例如，在数据并行（DDP）中，数据集被划分为 $N$ 份（$N = \text{WORLD\_SIZE}$），第 $i$ 个进程处理第 $i$ 个分片：

$$
\text{Data}_i = \text{Data}[\ i \cdot \frac{\text{len(Data)}}{N} : (i+1) \cdot \frac{\text{len(Data)}}{N}\ ]
$$

#### 主流框架的实现

SPMD 是几乎所有主流分布式训练框架的基础，包括：

- **DDP**（Distributed Data Parallel）：数据并行
- **ZeRO** 和 **FSDP**（Fully Sharded Data Parallel）：分片数据并行
- **Megatron-LM** 中的 **Tensor Parallelism** 和 **Pipeline Parallelism**

这些框架通过 SPMD 模型，利用环境变量协调多个进程，实现高效的模型并行和数据并行。

### 3.2 Verl 中 SPMD 的实现

与 `torchrun` 不同，Verl 作为一个构建在 Ray 之上的新框架，需**自行管理 SPMD 所需的环境配置与调度逻辑**。为降低开发复杂度，Verl 提供了高层抽象接口。下面我们逐步解析其内部实现。

#### 3.2.1 资源配置与环境变量初始化

Verl 通过 `init_with_resource_pool` 函数完成资源分配与环境初始化。其核心步骤如下：

1. **遍历 Placement Group**：每个 Placement Group 对应一个物理节点（Node）
2. **遍历 Local Rank**：在每个节点内，遍历其 GPU 设备，每个 GPU 对应一个 [local_rank](ScaleTorch/examples/mingpt/trainer.py#L0-L0)
3. **设置环境变量**：为每个 Ray Worker 进程设置关键的分布式环境变量，如 [world_size](verl/verl/single_controller/base/worker_group.py#L177-L179) 和 [rank](AReaL/realhf/base/topology.py#L0-L0)
4. **存储与实例化**：将环境变量存入 Ray 的运行时上下文（[runtime_env](ROLL/tests/third_party/vllm/test_vllm_local_actor.py#L103-L103) variables），并实例化对应的 Worker Class

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

- 每个 GPU 对应一个独立的 Ray Worker 进程
- 通过 [runtime_env](ROLL/tests/third_party/vllm/test_vllm_local_actor.py#L103-L103) 将 SPMD 环境变量注入每个进程
- 全局 [RANK](MindSpeed/mindspeed/log_config.py#L10-L10) 通过节点 IP 和 [local_rank](ScaleTorch/examples/mingpt/trainer.py#L0-L0) 映射生成

至此，系统已建立标准的 SPMD 执行环境：每个 GPU 上运行一个进程，且均已配置正确的分布式上下文。

#### 3.2.2 核心执行逻辑与 [@register](ms-swift/swift/llm/model/register.py#L0-L0) 装饰器

用户定义的执行逻辑（如 [update_actor](verl/verl/workers/fsdp_workers.py#L818-L857)）通过 [@register](ms-swift/swift/llm/model/register.py#L0-L0) 装饰器进行增强，使其具备分布式调度能力。

以 [Actor](verl/verl/trainer/ppo/ray_trainer.py#L71-L71) 的 [update_actor](verl/verl/workers/fsdp_workers.py#L818-L857) 方法为例，其核心逻辑非常简洁。

```python
@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
def update_actor(self, data: DataProto):
    # 此时 data_proto 已是分割后的本地数据分片
    data = data.to(torch.cuda.current_device())
    self.actor.update_policy(data=data)
    self.actor_lr_scheduler.step()
    self.actor_optimizer_scheduler.step(1)
```

此代码看似同步且简单，实则在底层实现了完整的 SPMD 流程。关键在于 `@register` 装饰器。它解决了"为何数据在进入函数前已被分割"的问题。

#### 3.2.3 `@register` 装饰器的内部机制

`@register` 装饰器的核心作用是为被修饰的函数添加元数据（metadata）属性，并将其包装成一个可被 Multi-Controller 调度的入口。其工作流程如下：

1. **属性注入**：将 `dispatch_mode`、`execute_mode`、[blocking](MindSpeed-RL/mindspeed_rl/trainer/base.py#L0-L0)、`materialize_features` 等配置作为"魔法属性"（magic attribute）附加到函数对象上
2. **函数包装**：创建一个 `inner` 函数作为实际调用入口
3. 参数预处理：`inner` 函数首先处理传入的参数：
   - **Materialize Features**：如果 `blocking=True`，则等待所有异步返回的 `Future`（引用）就绪，获取真实数据
   - **暂存配置**：将装饰器参数暂存，供后续调度逻辑使用

```python
def register(dispatch_mode=Dispatch.ALL_TO_ALL, execute_mode=Execute.ALL, blocking=True, materialize_futures=True):
    """Register a function with distributed execution configuration.

    This decorator registers a function with specific dispatch and execution modes
    for distributed computation. It handles both synchronous and asynchronous
    functions, and optionally materializes futures before execution.

    Args:
        dispatch_mode:
            Dispatch mode for computation distribution. Default: Dispatch.ALL_TO_ALL.
        execute_mode:
            Execute mode for computation distribution. Default: Execute.ALL.
        blocking:
            Whether the execution should be blocking. Defaults to True.
        materialize_futures:
            Whether to materialize the data before dispatching. Defaults to True.

    Returns:
        A decorator that wraps the original function with distributed execution
        configuration.
    """
    _check_dispatch_mode(dispatch_mode=dispatch_mode)
    _check_execute_mode(execute_mode=execute_mode)

    def decorator(func):
        @wraps(func)
        def inner(*args, **kwargs):
            # 1. 参数物化（Materialization）
            if materialize_futures:
                # 确保所有输入数据（如 Ray ObjectRef）已就绪
                args, kwargs = _materialize_futures(*args, **kwargs)
            # 2. 返回原始函数调用（实际调度由外部框架完成）
            return func(*args, **kwargs)

        @wraps(func)
        async def async_inner(*args, **kwargs):
            if materialize_futures:
                args, kwargs = _materialize_futures(*args, **kwargs)
            return await func(*args, **kwargs)

        wrapper = async_inner if inspect.iscoroutinefunction(func) else inner
        # 3. 附加元数据到函数对象
        attrs = {"dispatch_mode": dispatch_mode, "execute_mode": execute_mode, "blocking": blocking}
        setattr(wrapper, MAGIC_ATTR, attrs)
        return wrapper

    return decorator
```

**核心元数据说明**：

| 属性                   | 作用                                     |
| ---------------------- | ---------------------------------------- |
| `dispatch_mode`        | 定义输入数据如何从主进程分发到各 Worker  |
| `execute_mode`         | 定义 Worker 间的执行模式（如并行、串行） |
| `blocking`             | 调用是否阻塞主进程                       |
| `materialize_features` | 是否等待异步输入（如 `ObjectRef`）完成   |

#### 3.2.4 分发模式（Dispatch Mode）的实现

[@register](ms-swift/swift/llm/model/register.py#L0-L0) 只是配置的声明。真正的分发逻辑由框架在调用时触发。

`dispatch_mode` 是连接 **Single-Controller** 与 **Multi-Controller (SPMD)** 两层的关键。其本质是一组预定义的 **分发-收集（Dispatch-Collect）** 协议。

- **映射表维护**：框架维护一个预定义的映射表，将 `dispatch_mode`（如 `"DP"`）映射到具体的 `dispatch function` 和 `collect function`
  - **分发函数**（`dispatch_fn`）：负责将输入数据从**Single-Controller**（Single Controller）进程分发至多个工作进程（Worker Process）
  - **收集函数**（`collect_fn`）：在各个工作进程完成计算后，负责将分散的结果聚合为统一输出

该机制的设计具有良好的扩展性：若需引入新的并行行为，只需定义新的 `dispatch_mode` 及其对应的 `dispatch_fn` 和 `collect_fn` 即可。

- **动态函数合成**：当 Single Controller 调用 `worker_group.update_actor(data)` 时，框架会：
  - 读取 [update_actor](verl/verl/workers/fsdp_workers.py#L818-L857) 函数的 `dispatch_mode` 属性
  - 查找映射表，获取对应的 `dispatch_fn` 和 `collect_fn`
  - 将原始的 [update_actor](verl/verl/workers/fsdp_workers.py#L818-L857) 函数与 `dispatch_fn`、`execute_fn`、`collect_fn` 等结合，动态合成一个全新的、可执行的函数

#### 3.2.5 分发与收集函数的实现逻辑

##### 1. 数据分发（Dispatch）

以数据并行模式 [DP_COMPUTE_PROTO](Skywork-OR1/verl/single_controller/base/decorator.py#L34-L34) 为例，其 `dispatch_fn` 的作用是将输入参数切分为多个子集，以便分发给不同的 Worker。具体流程如下：

1. 接收 [WorkerGroup](verl/verl/single_controller/base/worker_group.py#L122-L251) 及其全局规模（[world_size](verl/verl/single_controller/base/worker_group.py#L177-L179)），即参与计算的 GPU 总数
2. 调用 [_split_args_kwargs_data_proto](Skywork-OR1/verl/single_controller/base/decorator.py#L44-L56) 工具函数，对输入参数进行切分

```python
def _split_args_kwargs_data_proto(chunks, *args, **kwargs):
    from verl.protocol import DataProto, DataProtoFuture

    splitted_args = []
    for arg in args:
        # 若参数支持 chunk 操作（如张量或 DataProto），则进行分块
        assert isinstance(arg, DataProto | DataProtoFuture)
        splitted_args.append(arg.chunk(chunks=chunks))

    splitted_kwargs = {}
    for key, val in kwargs.items():
        assert isinstance(val, DataProto | DataProtoFuture)
        splitted_kwargs[key] = val.chunk(chunks=chunks)

    return splitted_args, splitted_kwargs
```

该过程将原始输入数据均匀划分为 `world_size` 个子块，形成参数列表，供后续分发使用。

##### 2. Dispatch-Collect 映射表

Verl 维护一个映射，将 `dispatch_mode` 映射到具体的 `dispatch_fn` 和 `collect_fn`：

```python
# Global registry for dispatch mode.
DISPATCH_MODE_FN_REGISTRY = {
    Dispatch.ONE_TO_ALL: {
        "dispatch_fn": dispatch_one_to_all,
        "collect_fn": collect_all_to_all,
    },
    Dispatch.ALL_TO_ALL: {
        "dispatch_fn": dispatch_all_to_all,
        "collect_fn": collect_all_to_all,
    },
    Dispatch.MEGATRON_COMPUTE: {
        "dispatch_fn": dispatch_megatron_compute,
        "collect_fn": collect_megatron_compute,
    },
    Dispatch.MEGATRON_PP_AS_DP: {
        "dispatch_fn": dispatch_megatron_pp_as_dp,
        "collect_fn": collect_megatron_pp_as_dp,
    },
    Dispatch.MEGATRON_PP_ONLY: {"dispatch_fn": dispatch_one_to_all, "collect_fn": collect_megatron_pp_only},
    Dispatch.MEGATRON_COMPUTE_PROTO: {
        "dispatch_fn": dispatch_megatron_compute_data_proto,
        "collect_fn": collect_megatron_compute_data_proto,
    },
    Dispatch.MEGATRON_PP_AS_DP_PROTO: {
        "dispatch_fn": dispatch_megatron_pp_as_dp_data_proto,
        "collect_fn": collect_megatron_pp_as_dp_data_proto,
    },
    Dispatch.DP_COMPUTE: {"dispatch_fn": dispatch_dp_compute, "collect_fn": collect_dp_compute},
    Dispatch.DP_COMPUTE_PROTO: {
        "dispatch_fn": dispatch_dp_compute_data_proto,
        "collect_fn": collect_dp_compute_data_proto,
    },
    Dispatch.DP_COMPUTE_PROTO_WITH_FUNC: {
        "dispatch_fn": dispatch_dp_compute_data_proto_with_func,
        "collect_fn": collect_dp_compute_data_proto,
    },
    Dispatch.DP_COMPUTE_METRIC: {"dispatch_fn": dispatch_dp_compute_data_proto, "collect_fn": collect_dp_compute},
    Dispatch.DIRECT_ROLLOUT_METHOD: {
        "dispatch_fn": dummy_direct_rollout_call,
        "collect_fn": dummy_direct_rollout_call,
    },
}
```

##### 3. [dispatch_dp_compute_data_proto](Skywork-OR1/verl/single_controller/base/decorator.py#L271-L275) 的实现

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

##### 4. [collect_dp_compute_data_proto](Skywork-OR1/verl/single_controller/base/decorator.py#L288-L296) 的实现

相对应地，`collect_fn` 的行为较为简单，通常是对各 Worker 的输出进行拼接（concatenate）或合并操作：

```python
def collect_dp_compute_data_proto(outputs: List[DataProto]) -> DataProto:
    # 将各 Worker 的输出沿 batch 维度拼接
    return DataProto.concat(outputs)
```

最终，聚合后的结果被返回给 Single-Controller，完成一轮分布式计算的闭环。

#### 3.2.6 执行模式（Execute Mode）的调度逻辑

除了 `dispatch_mode`，Verl 还通过 `execute_mode` 控制远程调用的执行方式。系统维护另一组映射，将 `execute_mode` 映射到具体的执行函数名。

以默认模式 [Execute.ALL](Skywork-OR1/verl/single_controller/base/decorator.py#L40-L40) 为例，其实际指向 [execute_all_sync](ROLL/roll/distributed/executor/cluster.py#L203-L204)，表示**同步执行所有 Worker 上的方法调用**。

#### 3.2.7 同步执行函数 [execute_all_sync](ROLL/roll/distributed/executor/cluster.py#L203-L204) 的行为

该函数的核心逻辑如下：

1. 遍历 [WorkerGroup](verl/verl/single_controller/base/worker_group.py#L122-L251) 中的所有 Worker
2. 将已切分的参数子集与目标方法名（如 [update_actor](verl/verl/workers/fsdp_workers.py#L818-L857)、[generate_sequences](verl/verl/workers/fsdp_workers.py#L861-L907)）结合
3. 对每个 Worker 发起远程方法调用（Remote Method Invocation）
4. 收集所有远程调用的返回值（通常为 `Future` 对象）并返回

```python
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

#### 3.2.8 分布式函数的动态生成机制

```python
def func_generator(self, method_name, dispatch_fn, collect_fn, execute_fn, blocking):
    class Functor:
        def __call__(this, *args, **kwargs):
            args, kwargs = dispatch_fn(self, *args, **kwargs)
            padding_count = kwargs.pop(_padding_size_key, 0)
            output = execute_fn(method_name, *args, **kwargs)
            if blocking:
                output = ray.get(output)
            output = collect_fn(self, output)
            if padding_count > 0:
                if isinstance(output, DataProto):
                    indices = [i for i in range(len(output))][:-padding_count]
                    output = output.select_idxs(indices)
                elif isinstance(output, list):
                    output = output[:-padding_count]
            return output

    # use class type to pass the method_name to get a better observability
    return type(method_name, (Functor,), {})()

```

Verl 的核心设计之一是通过**函数生成器**（Function Generator）将上述组件动态组合，生成最终可执行的分布式函数。其整合逻辑如下：

1. **获取元信息**：从被 [@register](Skywork-OR1/verl/single_controller/base/decorator.py#L393-L409) 装饰的函数中提取 `dispatch_mode`、`execute_mode` 和 [blocking](MindSpeed-RL/mindspeed_rl/trainer/base.py#L0-L0) 等属性

2. **查找执行策略**：根据属性值查找对应的 `dispatch_fn`、`execute_fn` 和 `collect_fn`

3. **构建执行流水线**：

   - **步骤一：数据分发**
     使用 `dispatch_fn` 将输入参数切分为 [world_size](verl/verl/single_controller/base/worker_group.py#L177-L179) 个子集：
     $$
     \text{args}_i, \text{kwargs}_i = \text{dispatch\_fn}(\text{args}, \text{kwargs}), \quad i = 1, \dots, N
     $$
     其中 $N$ 为 [world_size](verl/verl/single_controller/base/worker_group.py#L177-L179)

   - **步骤二：并行执行**
     调用 `execute_fn` 在每个 Worker 上执行目标方法，返回 `Future` 列表：
     $$
     \text{futures} = [\text{worker}_i.\text{method}.\text{remote}(\text{args}_i, \text{kwargs}_i) \mid i = 1, \dots, N]
     $$

   - **步骤三：结果物化（Materialization）**
     根据 [blocking](MindSpeed-RL/mindspeed_rl/trainer/base.py#L0-L0) 标志决定是否同步等待所有 `Future` 完成：
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

### 3.3 执行流程

结合以上机制，[update_actor](verl/verl/workers/fsdp_workers.py#L818-L857) 的调用流程如下：

1. **主进程调用**：`worker_group.update_actor(full_data_batch)`
2. **触发 dispatch**：框架根据 `dispatch_mode` 查找 `dispatch_fn`
3. **数据分发**：[dispatch_dp_compute_data_proto](Skywork-OR1/verl/single_controller/base/decorator.py#L271-L275) 将 `full_data_batch` 切分为 $N$ 个子批次
4. **并行执行**：通过 Ray RPC 将子批次发送至各 Worker，触发 `inner()` 函数
5. **参数物化**：`inner()` 确保输入数据就绪（若 `materialize_features=True`）
6. **本地计算**：执行 [update_policy](verl/verl/workers/actor/base.py#L53-L65)，完成前向、反向传播
7. **结果收集**：各 Worker 返回梯度/损失等结果
8. **结果聚合**：`collect_fn` 合并结果（如梯度 `all-reduce`）

$$
\text{Grads}_{\text{global}} = \frac{1}{N} \sum_{i=0}^{N-1} \text{Grads}_i
$$

## 4. 总结：分层架构实现效率与灵活性的平衡

Verl 框架通过分层设计，巧妙地结合了 **Single Controller** 的灵活性与 **Multi-Controller/SPMD** 的高效性。

- **上层（Single-Controller）**：
  主进程以直观、同步的方式定义 Dataflow Graph 编写训练逻辑，用户无需关注分布式细节，框架自动处理底层分布式调度
- **下层（Multi-Controller / SPMD）**：
  在 Worker 内部，利用成熟的 SPMD 模式实现高效的并行计算（如 DDP、FSDP），最大化硬件利用率

[@register](Skywork-OR1/verl/single_controller/base/decorator.py#L393-L409) 装饰器作为关键抽象，将复杂的分布式调度逻辑（分发、执行、收集）与用户定义的业务逻辑解耦。

### 两层抽象如何协同

| 层次              | 控制器       | 职责                          | 代价                      | 适用场景 |
| ----------------- | ------------ | ----------------------------- | ------------------------- | -------- |
| Single-Controller | 主进程       | 定义 dataflow graph，逻辑清晰 | RPC 通信（轻量）          | 控制逻辑 |
| Multi-Controller  | GPU 进程内部 | SPMD 并行，极限性能           | 环境变量+通信（框架隐藏） | 计算密集 |

这种 **Hybrid Flow** 范式，使得 Verl 既能支持复杂的 RL 训练流水线，又能为用户提供简洁的编程接口，是大规模 LLM 强化学习训练框架设计的典范。

## 5. 代码库组织结构

`verl` 项目的主要代码结构如下：

```python
verl  # verl 主包
  trainer
    main_ppo.py            # RL 训练的入口点
    ppo
      ray_trainer.py       # PPO 等 RL 算法的训练循环
    fsdp_sft_trainer.py    # 基于 FSDP 后端的 SFT 训练器
  config
    generation.yaml        # 采样阶段的配置模板
    ppo_trainer.yaml       # RL 训练器的配置模板
  Workers
    protocol.py            # DataProto 接口定义
    fsdp_Workers.py        # FSDP 后端的Worker接口：ActorRolloutRefWorker, CriticWorker, RewardModelWorker
    megatron_Workers.py    # Megatron 后端的Worker接口：ActorRolloutRefWorker, CriticWorker, RewardModelWorker
    Actor
      dp_Actor.py          # 基于 FSDP 后端的数据并行Actor
      megatron_Actor.py    # 基于 Megatron 后端的 nD 并行Actor
    Critic
      dp_Critic.py         # 基于 FSDP 后端的数据并行评价者
      megatron_Critic.py   # 基于 Megatron 后端的 nD 并行评价者
    reward_model
      megatron
        reward_model.py    # 基于 Megatron 后端的RewardModel
    rollout
      vllm
        vllm_rollout.py    # 基于 vLLM 后端的采样实现
      hf_rollout.py        # 基于 HuggingFace TGI 后端的采样实现
    sharding_manager
      fsdp_ulysses.py      # 使用 FSDP + Ulysses 时的数据与模型重分片
      fsdp_vllm.py         # 使用 FSDP + Ulysses + vLLM 时的重分片
      megatron_vllm.py     # 使用 Megatron + vLLM 时的重分片
```

## 总结

通过对 Verl 框架的深入分析，我们可以看到其在设计上具有以下特点：

1. **清晰的架构分层**：通过 Single-Controller 和 Multi-Controller 的分层设计，实现了控制逻辑与计算逻辑的分离，提高了系统的可维护性和可扩展性

2. **灵活的资源管理**：通过 WorkerGroup 和 ResourcePool 的抽象，支持灵活的资源分配和任务调度

3. **高效的分布式执行**：利用 SPMD 模型和装饰器机制，实现了高效的分布式计算

4. **用户友好的 API**：通过简洁的装饰器和统一的接口，降低了用户使用分布式计算的门槛

这些设计使得 Verl 成为一个既强大又灵活的强化学习训练框架，特别适合大规模 LLM 的 RLHF 训练场景。
