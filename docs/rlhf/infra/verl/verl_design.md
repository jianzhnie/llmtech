# Verl 设计指南

`verl` 是论文 [HybridFlow](https://arxiv.org/abs/2409.19256v2) 的开源实现。本文旨在介绍 HybridFlow 的核心概念、设计动机，以及如何使用 `verl` 提供的 API 进行编程开发。

## 动机与设计

在 `verl` 的设计中，我们采用**数据流**（Dataflow）的方式对强化学习系统进行建模 。

### 数据流模型

数据流是计算过程的一种抽象表示。神经网络的训练过程即是一个典型的数据流，通常可用计算图（Computation Graph）来描述。

![The dataflow graph from CS231n 2024 lecture 4](https://github.com/eric-haibin-lin/verl-community/blob/main/docs/dataflow.jpeg?raw=true)

如图所示，该计算图表示一个多项式函数后接 S 型激活函数的结构。在神经网络的计算流中，每个节点代表一个基本运算符（如加法、矩阵乘法等），每条边表示数据（张量）在前向或反向传播中的流动方向。整个计算图决定了神经网络的拓扑结构。

#### 强化学习作为数据流问题

强化学习（Reinforcement Learning, RL）的训练过程同样可以被建模为数据流。下图展示了在 RLHF（基于人类反馈的强化学习）中常用的 PPO 算法的数据流图：

![PPO dataflow graph, credit to Zhihu 低级炼丹师](https://picx.zhimg.com/70/v2-cb8ab5ee946a105aab6a563e92682ffa_1440w.avis?source=172ae18b&biz_tag=Post)

然而，强化学习的数据流与传统神经网络训练的数据流存在本质区别：

| 工作负载（Workload） | 节点（Node）                        | 边（Edge）         |
| -------------------- | ----------------------------------- | ------------------ |
| 神经网络训练         | 基本运算符（如 +/-/matmul/softmax） | 张量（Tensor）流动 |
| 强化学习（RL）       | 高层算子（如 rollout/模型前向传播） | 数据（Data）移动   |

在传统表格型强化学习中，每个算子通常是简单的标量运算（如贝尔曼方程更新）。而在深度强化学习（DRL）中，每个算子本身就是一个复杂的神经网络计算过程，如模型推理或参数更新。这使得深度强化学习呈现出**双层数据流**的特性：

- **控制流**（Control Flow）：定义高层算子的执行顺序与逻辑。例如，在 PPO 算法中，依次执行轨迹采样、优势函数计算、策略更新等步骤。控制流体现了**强化学习算法的核心逻辑**。
- **计算流**（Compute Flow）：定义**神经网络内部**的计算过程，如前向传播、反向传播、优化器更新等。

### 设计选择

在大语言模型（LLM）兴起之前，深度强化学习所使用的模型规模较小，其计算流通常可以在单个进程中完成。因此，将控制流与计算流集成在单一进程中是可行的。

然而，在 LLM 时代，计算流（如大规模模型的训练）必须依赖多进程并行计算。这催生了两种系统架构设计方案：

1. **统一多控制器模式**：将控制流也转化为多进程程序，与计算流协同部署。
   - **优势**：
     - 在固定控制流和计算流的场景下，能够实现**最优性能**，因为训练过程中的通信开销被最小化。
   - **劣势**：
     - 从软件工程角度看，计算代码与控制器逻辑高度耦合，导致**计算流和控制流难以复用**。例如，若已实现一个基于 FSDP 的 PPO 训练流程，当需要切换到 Megatron 计算后端时，由于控制流与计算流的耦合，两者均难以直接复用。
     - 多进程控制流在面对动态或复杂的控制逻辑时，开发和维护成本较高。

2. **分离式架构**：控制流在单进程中运行，计算流在多进程中运行。
   - **优势**：
     - 实现了控制流与计算流的解耦，使得**计算流程可以轻松复用**于不同的控制逻辑。
     - 控制器运行在单一进程中，**实现新的强化学习算法更加简单灵活**。
   - **劣势**：
     - 每次控制器与计算节点交互时，都会引入额外的**数据通信开销**，数据需要在控制进程与计算进程之间来回传输。

`verl` 采用了第二种分离式设计策略，其核心目标是将强化学习算法的**控制流**与底层**计算引擎**的实现进行解耦，从而提升系统的模块化和可扩展性。

### 整体执行流程

下图展示了 `verl` 中强化学习任务的简化执行流程：

![The execution diagram](https://github.com/eric-haibin-lin/verl-community/blob/main/docs/driver_Worker.png?raw=true)

在该架构中，**控制器**（Controller，又称驱动进程）运行于单个进程中，而Actor（或行动者）、Critic等Workers 则分布于多个 GPU 进程中，并以**Workers组**（WorkerGroup）的形式进行管理。

在轨迹采样阶段，控制器将提示（prompt）数据分发给 generator WorkerGroup，各节点并行生成响应。采样完成后，生成的数据被回传至控制器，由其执行后续的算法步骤（如优势计算、奖励计算等）。其他计算任务（如价值估计、奖励打分）遵循类似的交互模式。

通过这种混合式设计，`verl` 成功实现了数据流与计算过程的解耦，在保证计算效率的同时，为强化学习训练循环的灵活定义提供了支持。

## 代码库导览

### 入口函数

代码位置：[main_ppo.py](https://github.com/volcengine/verl/blob/main/verl/trainer/main_ppo.py)

该文件中定义的 `main_task` 函数即为上文所述的**控制器进程**（驱动进程）。同时，`RewardManager` 类允许用户基于具体数据集自定义奖励函数。需要注意的是，`RewardManager` 应返回 RL 算法优化所需的最终 token 级奖励，用户可结合基于模型的奖励与基于规则的奖励。

`main_task` 函数负责构建 `RayPPOTrainer` 实例并启动训练流程。**需要特别强调的是，`main_task` 以单进程模式运行**。

建议避免将 `main_task` 调度在 Ray 集群的头节点上，因为该进程会消耗大量内存，而头节点通常资源有限。

### Ray Trainer

代码位置：[ray_trainer.py](https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/ray_trainer.py)

`RayPPOTrainer` 类主要负责：

- 管理Worker及其组的构建；
- 执行 PPO 算法的主循环。

**注意**：`RayPPOTrainer` 的 `fit` 函数同样以**单进程**形式运行，确保控制逻辑的集中管理。

### Worker与WorkerGroup

每个 `WorkerGroup` 管理一组远程运行的Worker。WorkerGroup在其创建进程中运行，而组内的每个Worker则运行在独立的 GPU 上。`WorkerGroup` 作为控制器与Worker之间的代理，负责执行具体的计算任务。

为了将Worker的方法暴露给控制器，并定义数据的分发与收集机制，`verl` 采用了一种基于装饰器的简洁设计，具体将在“Worker定义”一节中详述。

在 PPO 算法中，主要定义了以下三类WorkerGroup：

- **ActorRolloutRef**：管理Actor（Actor）、轨迹采样（Rollout）和参考策略（Reference Policy）。`ActorRolloutRefWorker` 可以实例化为单一角色（如仅Actor），也可组合多个角色（如Actor+采样器+参考策略）。这种设计旨在最大化代码复用性。将Actor与采样器共置，是为了利用 NCCL 实现高效的权重同步；将参考策略与Actor共置，则是为了支持高效的 LoRA-PPO——在 LoRA 微调中，参考策略通常对应基础模型。共置功能通过 `verl.single_controller.ray.base.create_colocated_Worker_cls` 装饰器实现，该装饰器会创建一个暴露所有角色方法的 Ray 远程类。
- **Critic**：管理Critic 模型。
- **Reward**：管理奖励模型（Reward Model）。

这些WorkerGroup将在指定的资源池（即 Ray 集群中的 GPU 集合）上进行部署。

### Worker定义

以 [ActorRolloutRefWorker](https://github.com/volcengine/verl/blob/main/verl/Workers/fsdp_Workers.py) 为例，其向控制器暴露的 API 包括：

- `init_model`：构建底层模型；
- `generate_sequences`：根据输入提示生成响应序列；
- `compute_log_prob`：使用Actor模型计算生成序列的对数概率；
- `compute_ref_log_prob`：使用参考策略计算生成序列的对数概率；
- `save_checkpoint`：保存模型检查点。

这些方法定义在工作进程中，只能通过远程调用（Remote Call）执行。例如，控制器初始化模型时需调用：

```python
for worker in actor_rollout_ref_wg:
    worker.init_model.remote()
```

若需生成序列，控制器需执行如下代码：

```python
data = xxx
# 将数据划分为数据并行（DP）分块
data_dp_lst = data.split(dp_size)
output_dp_lst = []
for i, Worker in enumerate(actor_rollout_ref_wg):
    output_future = Worker.generate_sequences.remote(data_dp_lst[i])
    output_dp_lst.append(output_future)
output = torch.cat(ray.get(output_dp_lst), dim=0)
```

可以观察到，控制器调用WorkerGroup方法的通用模式包含三个步骤：

1. 将输入数据按数据并行规模划分；
2. 将划分后的数据分发给各个Worker；
3. 收集各节点的计算结果并拼接为完整输出。

为简化这一流程，`verl` 提供了语法糖机制，将上述三步封装为一次调用：

```python
@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
def generate_sequences(data):
    ...

# 在控制器中
output = actor_rollout_ref_wg.generate_sequences(data)
```

通过 `@register` 装饰器，开发者可以明确指定输入数据的分发方式及输出数据的收集策略。例如，`Dispatch.DP_COMPUTE_PROTO` 表示将输入数据按数据并行方式分割，分发至各Worker，最终收集并拼接输出。需要注意的是，此类方法的输入输出需符合 `DataProto` 协议格式（详见 [protocol.py](https://github.com/volcengine/verl/blob/main/verl/protocol.py)）。

### PPO 主循环实现

借助上述 API，开发者可以像编写单进程程序一样实现 PPO 的主循环：

```python
for prompt in dataloader:
    output = actor_rollout_ref_wg.generate_sequences(prompt)
    old_log_prob = actor_rollout_ref_wg.compute_log_prob(output)
    ref_log_prob = actor_rollout_ref_wg.compute_ref_log_prob(output)
    values = critic_wg.compute_values(output)
    rewards = reward_wg.compute_scores(output)
    # compute_advantages 在控制器进程中直接执行
    advantages = compute_advantages(values, rewards)
    output = output.union(old_log_prob)
    output = output.union(ref_log_prob)
    output = output.union(values)
    output = output.union(rewards)
    output = output.union(advantages)
    # 更新Actor
    actor_rollout_ref_wg.update_actor(output)
    critic_wg.update_critic(output)
```

### 关键设计优势

- **计算后端可切换性**：该编程范式使用户无需修改控制流程即可灵活切换不同的计算后端（如 FSDP、Megatron）。
- **部署灵活性**：通过调整 `WorkerGroup` 与 `ResourcePool` 的映射关系，可在不修改控制逻辑的前提下实现灵活的资源部署策略。

## 代码库组织结构

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
    actor
      dp_actor.py          # 基于 FSDP 后端的数据并行Actor
      megatron_actor.py    # 基于 Megatron 后端的 nD 并行Actor
    critic
      dp_critic.py         # 基于 FSDP 后端的数据并行评价者
      megatron_critic.py   # 基于 Megatron 后端的 nD 并行评价者
    reward_model
      megatron
        reward_model.py    # 基于 Megatron 后端的奖励模型
    rollout
      vllm
        vllm_rollout.py    # 基于 vLLM 后端的采样实现
      hf_rollout.py        # 基于 HuggingFace TGI 后端的采样实现
    sharding_manager
      fsdp_ulysses.py      # 使用 FSDP + Ulysses 时的数据与模型重分片
      fsdp_vllm.py         # 使用 FSDP + Ulysses + vLLM 时的重分片
      megatron_vllm.py     # 使用 Megatron + vLLM 时的重分片
```
