# ROLL：用于大规模学习的强化学习优化——一个高效且用户友好的扩展库

## 摘要

强化学习（RL）在推动大型语言模型（LLM）发展方面取得了显著成功，这促使了高效 RL 训练框架的开发。然而，这些框架需要协调管理多个模型和多阶段训练流程，带来了效率、可扩展性和可用性方面的挑战。为此，我们推出了 **ROLL**（Reinforcement Learning Optimization for Large-scale Learning），一个高效、可扩展且用户友好的 RL 优化库，专为大规模学习设计。

ROLL 面向三类主要用户：

- **技术先锋**：追求低成本、容错的大规模训练；
- **产品开发者**：需要灵活控制训练流程；
- **算法研究者**：希望快速实验新想法。

ROLL 的核心模块包括：

1. **单控制器架构** + 并行Woker的抽象，简化训练流程开发；
2. **并行策略与数据传输模块**，实现高效可扩展训练；
3. **Rollout 调度器**，精细控制每个样本的生命周期；
4. **环境Woker与奖励Woker**，支持智能体 RL 算法和奖励设计的快速实验；
5. **AutoDeviceMapping**，支持灵活的资源分配。

我们使用 ROLL 成功训练了一个超过 200B 参数的 MoE 模型，在数千张 GPU 上持续运行两周无中断，验证了其可扩展性与容错性。此外，我们在多个可验证奖励任务和智能体 RL 任务上对 ROLL 进行了基准测试，验证其在多样化 RL 场景中的可用性和有效性。

## 1 引言

强化学习（RL）在大语言模型（LLM）中的成功应用，尤其是 RL from Human Feedback（RLHF）的开创性工作，推动了面向对齐优化（reference alignment）[^1]、推理增强（reasoning enhancement）[^2] 和智能体工具使用（agentic tool use）[^3] 等方向的高级 RL 技术快速发展。包括 OpenAI o4[^4]、QwQ[^5]、Seed1.5-thinking[^6] 和 Deepseek-R1[^7] 在内的诸多领先 LLM 均已借助 RL，在编程[^8]、数学[^9] 和工具使用[^10][^11] 等一系列 AI 任务中取得了卓越性能。

现有面向 LLM 的 RL 优化算法大致可归纳为以下几类范式：

- **RL from Human Feedback（RLHF）**[^12-15]
- **RL with Verifiable Rewards（RLVR）**[^16-19]
- **RL with Multi-turn Agentic Interaction**[^20-22]

这些范式通常需要同时维护多个 LLM，并编排一个多阶段的训练 pipeline。标准的 RL 训练流程往往涉及多达四个不同的 LLM[^23][^24]，即 Actor、Critic、Ref（参考模型）和 Reward（奖励模型）（详见第 2.1 节）。每一次训练迭代都包含以下三个阶段：

1. **生成阶段（Generation）**：Actor 根据一批输入提示生成响应。在智能体 RL 场景下，Actor 还可能与环境进行多轮交互。
2. **推理阶段（Inference）**：Critic、Ref 和 Reward 模型对生成的响应进行前向传播，计算监督信号或奖励估计。近期的 RL 研究通过减少此阶段涉及的 LLM 数量，甚至完全移除该阶段，从而简化了流程[^25-27]。
3. **训练阶段（Training）**：Actor 和 Critic 模型利用推理阶段获得的奖励信号更新参数。在某些 RL 算法中[^28-30]，Critic 模型可能保持不活跃。然而，大多数 RL 优化方法仍然属于这种“多模型 + 多阶段”训练范式的范畴。

为了支持 LLM 的高效 RL 优化，近年来已涌现出多个系统框架[^31-36]。这些工作大多引入了单控制器（single-controller）[^37]、共同部署 （colocation）[^38] 和分离式架构（disaggregated architectures）[^39] 等经典系统设计方法，以加速 LLM 的 RL 训练。

受上述开创性工作的启发，如图 1 所示，我们推出了 **ROLL**（Reinforcement Learning Optimization for Large-scale Learning）——一个高效、可扩展且用户友好的库，旨在为大规模学习提供强有力的 RL 优化支持。ROLL 面向三类主要用户群体，具备以下关键特性：

- 对于**技术先锋**，ROLL 支持在包含异构硬件的大规模 GPU 集群上，进行快速、低成本、可扩展且容错的 RL 训练。
- 对于**产品开发者**，ROLL 提供灵活且细粒度的控制能力，能够将输入样本路由到合适的智能体环境、奖励计算模块和设备上，以最小的工程投入实现强劲性能。
- 对于**算法研究者**，ROLL 能够在 GPU 资源受限的设备上实现高效训练，并通过精心设计的 RL 训练流程抽象，支持新想法的快速实验。

具体而言，ROLL 由以下关键模块构成，以实现其先进特性：

- 我们在单控制器架构[^37] 的基础上，引入了明确定义的 **并行Woker（Parallel Worker）** 抽象，从而实现灵活、模块化的 RL 训练 pipeline，简化新想法的实验过程。
- 我们引入了优化的 **并行策略（Parallel Strategy）** 和 **数据传输（Data Transfer）** 模块，既支持在资源受限的设备上执行，也支持快速、可扩展且容错的训练。
- 我们提供了 **Rollout 调度器（Rollout Scheduler）**，以支持在生成阶段对每个提示样本的生命周期进行细粒度管理，简化响应生成、环境交互和奖励计算之间的执行流程编排。
- 我们专门设计了 **环境Woker（Environment Worker）** 和 **奖励Woker（Reward Worker）**，以提供高效且可扩展的智能体环境交互和奖励计算能力。
- 我们实现了 **资源池（Resource Pool）**，并利用 **AutoDeviceMapping** 以实现高效的Woker部署和优化的资源分配。

ROLL 构建于 Ray[^40] 之上，并集成了现有的 LLM 优化系统，包括 vLLM[^41]、SGLang[^42]、DeepSpeed[^43] 和 Megatron[^44]。我们使用 ROLL 在数千张 GPU 上持续两周无中断地训练了超过 200B 参数的 MoE 模型，验证了其在大规模 RL 训练中的效率与容错能力。此外，我们还针对一个涵盖代码、数学等多领域的 RLVR 任务，以及三个智能体 RL 任务，对 ROLL 进行了基准测试，以验证其在多样化 RL 场景中的正确性和可用性。

## 2 背景知识

### 2.1 面向大语言模型的强化学习

强化学习（RL）是 LLM 后期训练中的关键技术。本节首先概述 LLM-RL 中的核心概念，随后给出完整的训练流程。

#### 核心概念
LLM 的 RL 训练通常采用策略梯度方法，特别是 PPO（Proximal Policy Optimization）及其变体。训练pipeline一般包含以下组件：

- **Actor 模型**：负责根据提示生成回复。
- **Critic 模型**：估计状态价值函数。
- **Ref（参考）模型**：防止策略过度偏离初始行为。
- **Reward 模型**：对回复质量进行打分。

对于给定提示，Actor 持续生成 token 序列直至满足终止条件。在 RL 框架下，每个 token 的生成被视为一次“动作”，优化目标是通过调整策略（即 Actor）来最大化期望累积奖励，使生成序列更符合人类偏好与任务需求。
- **Ref 模型** 通常由 Actor 初始化，训练期间参数冻结，用于 KL 正则化。
- **Reward 模型** 可用于人类偏好、工具使用、数学推理、代码执行等任务。其训练数据可以是人工标注的偏好数据，也可以通过规则验证或沙箱执行获得奖励值。
- **Critic 模型** 估计当前状态（已生成文本）的未来期望奖励，用于降低策略梯度的方差并指导策略优化。

#### 优化流程
每次 RL 迭代包括生成、推理和训练三个阶段。

1. **生成阶段**
   Actor 与环境交互，为一批提示生成回复。该阶段包含：

   - **预填充（prefill）**：计算提示的 KV 缓存，计算密集型 GPU 任务；
   - **解码（decoding）**：自回归地生成 token，直到满足终止条件，内存密集型 GPU 任务；
   - **环境交互**：执行复杂环境并与 Actor 交互，CPU 密集型任务。

   对于数学、代码等单轮任务，Actor 通常进行无状态环境交互，仅包含预填充和解码。对于工具使用等多轮任务，Actor 与环境进行多轮交互，使得环境交互阶段成为性能瓶颈。

   在生成阶段，一条 rollout 样本由预填充、解码和环境交互阶段生成的 token 组成，供后续推理和训练使用。为提高收敛速度，通常一次性生成一批回复，但会带来较大的计算开销。

2. **推理阶段**
   Actor 生成的每条序列分别由 Ref、Critic 和 Reward 模型进行一次前向传播。

   - Ref 模型提供 KL 惩罚，防止策略过度偏离；
   - Critic 估计价值分数用于优势计算；
   - Reward 模型给出质量分数。

   这些输出组合为最终训练目标，通常包含策略损失、价值损失和 KL 惩罚项。该过程仅涉及预填充阶段，为计算密集型任务。

   例外情况是奖励计算。基于 LLM 的奖励计算可视为预填充阶段并在 GPU 上运行；而规则验证（如数学验证、沙箱执行）则类似于环境交互阶段，通常需要大量 CPU 资源以快速获得奖励。

3. **训练阶段**
   利用生成阶段产生的样本和推理阶段获得的奖励信号，更新 Actor 和 Critic 的参数。随后，更新后的参数在下一轮迭代中同步至生成阶段。相比生成和推理阶段，训练阶段 GPU 内存消耗大，需采用多种并行策略以实现高效执行。

### 2.2 面向 RL 增强 LLM 的系统优化

#### 训练

LLM 训练可通过 5D 并行进行加速：

- 数据并行（DP）
- 张量并行（TP）
- 流水线并行（PP）
- 上下文并行（CP）
- 专家并行（EP）

此外，可采用 ZeRO、激活重算和显存卸载等技术缓解内存压力。

#### 推理 / 生成

许多高效 LLM 服务框架（如 SGLang、vLLM）已支持 DP、TP、PP、EP。近期研究还优化了注意力计算和 KV 缓存使用。

#### RL 优化

LLM 的 RL 训练包含生成、推理、训练三种不同计算，且各模型规模不一。
- Actor：生成 + 训练
- Critic：训练 + 推理
- Ref：推理
- Reward：推理

因此，可为不同阶段的不同模型定制并行策略以最大化整体性能。

- **NeMo**[^H25] 和 **OpenRLHF**[^H24] 将 GPU 集群划分为多个分区，分别分配给不同阶段，并在每个阶段内采用优化并行策略。
- **Verl**[^S24]、**RLHFuse**[^Z24]、**ReaL**[^M24]、**PUZZLE**[^L24] 将不同阶段 LLM 共同部署 于同一资源池，提高资源利用率。
- **StreamRL**[^Z25] 提出将训练与生成阶段分离，并以流水线方式异步运行，利用推理集群的高内存带宽加速 rollout 生成。

### 2.3 强化学习算法

#### 人类反馈强化学习（RLHF）
早期 RL 优化 LLM 的成功在于引导模型符合人类偏好。早期 RLHF 方法主要围绕直接从人类奖励学习[^K12][KS09]、从动作建议学习[^M05] 或从动作批评学习[^J10]。例如，TAMER 将人类反馈视为最优动作价值函数的样本，COACH 则考虑策略相关的人类反馈。

ChatGPT 发布后，许多 RLHF 方法[^O22][S17] 被提出以对齐 LLM 与人类偏好，通常包括：

1. 监督微调（SFT）
2. 奖励模型训练
3. 策略优化

然而，这些方法需要大量人工标注样本来训练奖励模型，阻碍了其广泛应用。

#### 可验证奖励的强化学习（RLVR）
一些研究者[^Z22][L24][D25][Y25] 提出在代表性推理任务（如数学、代码）上采用 **RL with Verifiable Rewards（RLVR）**。这些任务的正确性通常由最终答案是否准确决定，原因在于中间步骤缺乏标注真值，难以可靠评估。
- 数学任务常用基于规则的验证策略；
- 代码任务则通过沙箱判断生成代码是否通过所有测试用例；
- 若答案正确性难以判断，可采用 LLM-as-a-Judge[^S24] 让大模型判定答案正确性；
- 近期广泛采用的动态采样策略[^Y25] 可根据样本难度进行过滤，提升推理性能。

#### 多轮智能体交互的强化学习
与单轮设置不同，多轮 RL 面向更真实的智能体场景[^Z24b][A25]，要求 LLM-based agent 执行一系列动作以完成任务，如管理终端[^L24a]、浏览网页界面[^Z24a]。
- 环境执行缓慢
- 动作奖励反馈难以获取
- 环境与 LLM 交互复杂

这些因素共同构成了在 LLM 多轮智能体交互场景中采用 RL 优化的重大挑战。

## 3 ROLL 的关键特性

为了支撑高效的执行与友好的 RL 开发体验，ROLL 提供了若干核心功能。下文从三类目标用户的角度分别阐述这些特性，并进一步给出面向智能体 RL 训练流程的具体规格说明。

### 3.1 技术先锋（Tech Pioneer）

技术先锋希望在 LLM 领域保持领先地位，并拥有大规模 GPU 集群以支撑可扩展的 RL 训练。ROLL 针对这一用户群体在以下三方面体现优势：

- **快速且低成本**
  ROLL 可充分压榨高性能硬件的全部潜力，加速 RL 训练，在大型 GPU 集群上显著降低训练时间与成本。

- **可扩展性与容错能力**
  ROLL 支持全面的 LLM 训练与服务优化技术，可在数千张 GPU 上稳定训练 200B 参数级模型达两周而不中断；同时具备高效的 checkpoint 与恢复机制，任务重启所需的工程代价极低。

- **灵活的硬件使用方式**
  用户可在多种硬件类型上运行 RL 训练，按需选择“共同部署 ”或“分离式”执行，以及同步或异步模式，充分发挥不同硬件架构的优势。

### 3.2 产品开发者（Product Developer）

产品开发者拥有充足 GPU，可开展内部 LLM 的 RL 训练，其核心诉求在于通过配置任务与奖励来强化模型的对齐、推理、工具使用及业务指标。推荐产品开发者选择 ROLL 的理由如下：

- **丰富且可扩展的奖励 / 环境**
  ROLL 内置一套 Reward Worker 与 Environment Worker。开发者可基于我们提供的实现快速定制自己的奖励函数与环境逻辑，无需从零开发。

- **组合式样本-奖励路由**
  ROLL 提供简单易用的接口，可按比例控制不同任务的提示采样，并动态地将每个样本路由到对应的 Reward Worker（如数学验证器、沙箱环境、LLM-as-a-Judge）。当生产级 LLM 需同时覆盖多种能力时，该特性可在混合领域与任务中优化模型表现。

- **便捷的设备-奖励映射**
  ROLL 设计了设备-奖励映射接口，用户可一键配置 Reward Worker 的设备分布，将奖励计算与其他计算负载隔离开，避免多任务 RL 训练中的干扰与瓶颈。

- **丰富的训练配方**
  ROLL 内置多种 RL 算法、模型、任务和数据集，显著降低开发新训练特性的工程成本。

- **卓越的默认性能**
  ROLL 提供经过调优的训练配置，在大量任务上即可达到令人满意的效果，省去繁重的人工超参搜索。

### 3.3 算法研究者（Algorithm Researcher）

大多数算法研究者仅拥有有限 GPU，需要在资源受限的条件下，对 LLM RL 训练的每个组件进行细粒度控制，以便高效地验证新思路。ROLL 为此提供以下关键功能：

- **受限设备执行**
  ROLL 通过一系列内存优化技术（包括单 GPU 训练）实现高效训练，使研究者可在少量低规格 GPU 上快速试错并及时获得反馈。

- **可插拔推理pipeline**
  ROLL 将 RL 训练流程按合适粒度抽象为若干独立阶段，研究者可以灵活编排各阶段的执行顺序与实现细节，轻松尝试不同的 RL 算法。

- **透明的实验追踪**
  系统提供详尽的日志与监控功能，方便追踪与分析每一次实验。

- **公平的学术基线**
  ROLL 提供经典算法、模型与任务实现，助力在标准基准上进行公平的基线对比。

### 3.4 面向智能体 RL 的规格说明

随着智能体 RL 需求激增，ROLL 额外提供了以下特性，以支持基于 LLM 的可扩展智能体 RL 训练：

- **可扩展的多轮智能体-环境交互**
  受 RAGEN[^W25] 启发，ROLL 支持智能体与环境的多轮交互，可扩展至长程任务。

- **样本级可扩展环境**
  ROLL 可根据输入样本规模灵活地并行扩展环境实例，实现高吞吐 rollout。

- **异步并行化的智能体-环境交互**
  通过样本级环境管理，ROLL 异步地执行环境与 Actor 生成，实现并行环境执行，降低 GPU 空闲时间，最大化资源利用率。

## 4 框架设计

本节阐述 ROLL 的整体设计，以支撑第 3 章所述关键特性。

### 4.1 系统架构与模块

#### 4.1.1 架构总览

<img src="https://ar5iv.labs.arxiv.org/html/2506.06122/assets/x2.png" alt="Refer to caption" style="zoom:50%;" />

图 2a 描绘了 ROLL 的宏观架构。系统接收用户定义的 RL 数据流图（RL dataflow graph）及其配套配置，随后由分布式执行器 & 调度器（Distributed Executor & Scheduler）负责编排所有 Worker 与 Scheduler。
AutoDeviceMapping 模块在已分配的资源池（Resource Pool）内部管理 CPU / GPU 资源，并将 Worker 和 Scheduler 高效地绑定到相应硬件。

#### 4.1.2 Parallel Worker（并行Woker）

Parallel Worker 是一组资源（Ray PlacementGroup）的“所有者”。ROLL 用 **Cluster** 表示承担相同角色（如 Actor 训练、Critic 推理等）的 Parallel Worker 集合，以简化批量管理。ROLL 提供多种 Worker 类型：

| Worker 类型            | 职责                                                         |
| - |  |
| **Actor Worker**       | 既可充当 **Actor**（生成），也可充当 **Ref**（KL 正则）。    |
| **Critic Worker**      | 负责 Critic 的全部功能。                                     |
| **Reward Worker**      | 提供规则验证[^H25]、沙箱执行[^D24]、LLM-as-a-Judge[^S24] 等多种奖励计算方法。 |
| **Environment Worker** | 支持 LLM 与各类环境的多轮交互。                              |

#### 4.1.3 Parallel Strategy（并行策略）
针对训练、推理、生成三种阶段，ROLL 集成 Megatron-Core 与 DeepSpeed，支持完整的 5D 并行（DP / PP / TP / CP / EP）以及 ZeRO-2 / ZeRO-3 / ZeRO-Offload[^R21]。
- **训练阶段**：支持梯度检查点、显存卸载，显著降低 GPU 内存占用，可在资源受限设备上运行。
- **推理 / 生成阶段**：集成 vLLM 与 SGLang，支持 TP / EP / PP，以加速推理与 rollout 生成。

#### 4.1.4 Rollout Scheduler（Rollout 调度器）
Rollout Scheduler 允许用户在 **样本级别**（而非批次级别）控制每个请求的生命周期：
- 可依据当前资源与生成进度 **动态追加或终止** 请求；
- 为动态采样、提前终止等高级策略提供原生支持。

#### 4.1.5 Data Transfer（数据传输）

- **Transfer Protocol** 源自 HybridFlow[^S24]，用于在不同阶段之间对输入 / 输出数据进行重分片（reshard）。
- **ModelUpdateGroup** 借助 NCCL 通信后端，在训练与生成 / 推理阶段之间快速同步参数，即使共同部署 也能保持高吞吐。

#### 4.1.6 AutoDeviceMapping & Resource Pool

- **Resource Pool** 统一管理 CPU / GPU 资源。
- **AutoDeviceMapping** 根据用户配置，将 Worker 与 Scheduler 绑定到指定设备，实现灵活、细粒度的资源分配。

### 4.2 系统工作流

<img src="https://ar5iv.labs.arxiv.org/html/2506.06122/assets/x3.png" alt="Refer to caption" style="zoom:50%;" />

图 2b 展示了包含 **运行时设置（Runtime Setup）** 与 **训练迭代（Training Iteration）** 的完整工作流。

#### 4.2.1 运行时设置
1. 依据 **设备配置** 创建 CPU / GPU 资源池。
2. 根据 RL 数据流图创建 **Rollout Scheduler** 与多个 **Parallel Worker**。
3. 依据训练 / 模型配置，为每个 Worker 实例化 **Parallel Strategy**，决定并行方式与执行后端。
4. 依据用户指定的设备映射，通过 **AutoDeviceMapping** 将资源绑定到各 Worker。

#### 4.2.2 训练迭代
- **生成阶段**
  一批样本首先进入 Rollout Scheduler 生成回复。
  - 在智能体任务中，Actor 通过 **Environment Worker** 进行多轮交互；
  - 通过 **Reward Worker** 计算奖励信号，支持动态采样等高级采样策略。

- **推理阶段**
  依据数据流图激活 Critic / Reward / Ref 模型，对生成阶段产出的序列进行前向传播；
  Transfer Protocol 将结果重分片后送入各 Worker。

- **训练阶段**
  Critic 与 Actor 使用准备好的奖励信号更新参数；
  Actor 通过 ModelUpdateGroup 将最新参数同步到下一轮生成阶段。

### 4.3 关键特性的落地实现

#### 4.3.1 单控制器pipeline

沿用 HybridFlow[^S24] 的混合编程模型，ROLL 在单一控制器内部实现了 RLHF、RLVR 与智能体 RL 的训练pipeline，极大简化了开发与管理。

#### 4.3.2 RL pipeline的 Worker 抽象

- Parallel Worker + Rollout Scheduler 的抽象使用户只需遵循示例即可用最小工程代价定义或实验新pipeline。
- Actor / Critic / Reward / Environment Worker 封装了 RL 训练中的独立角色，用户可聚焦单一组件的开发与定制，而无需重写整套代码。

#### 4.3.3 优化的 LLM 执行
ROLL 充分复用现有 LLM 执行引擎（DeepSpeed、Megatron、vLLM、SGLang）的高级特性，既能在超大规模 GPU 集群上高效运行，也能在资源受限设备上顺利执行。

#### 4.3.4 用户自定义设备映射
- 传统系统（OpenRLHF[^H24]、NeMo[^A25]）在不同训练阶段之间强制独占资源。
- 近期研究[^S24][Z24] 支持将不同阶段 LLM 共同部署 在同一设备组。
- ROLL 的 AutoDeviceMapping 支持 **用户自定义** 设备映射，允许同一设备被多个阶段共享。
  例如，可将部分用于 Actor 生成的 GPU 动态划拨给训练阶段，提高整体利用率。
  该能力依赖：
  1. ROLL 基于 Ray，可将设备绑定到特定 Worker，同时允许多 Worker 共享同设备；
  2. ModelUpdateGroup 支持跨阶段参数同步：训练阶段以 bucket 为单位广播参数到生成阶段，避免强制共同部署 ，实现比现有系统更灵活的映射策略。

#### 4.3.5 样本级 Rollout 生命周期控制
- 多数现有系统[^A25][H24][S24][Z24] 以批次为粒度处理提示样本，易因长尾问题导致 Worker 间负载不均。
- Rollout Scheduler 提供 **样本级** 生命周期控制，支持三项关键优化：
  1. **异步奖励计算**：完成生成的样本立即启动奖励计算，无需等待整批结束；
  2. **动态追加请求**：实时监控 Worker 完成状态，按需追加新提示，提高资源利用率；
  3. **主动终止请求**：当已收集足够有效梯度样本时，可提前终止其余生成任务，减少冗余开销。

#### 4.3.6 样本级奖励与环境管理
- 生成阶段（图 2b）展示了 **异步奖励计算** 与 **异步环境交互**。
- ROLL 可根据负载在资源池中动态启动多个 Reward / Environment Worker，并通过样本级生命周期控制灵活路由每个样本到指定 Worker。
- ROLL 利用 Ray 支持异步奖励计算：规则验证、沙箱执行、LLM-as-a-Judge 三类 Reward Worker 按实时负载动态启动；AutoDeviceMapping 将各 Reward Worker 绑定到用户指定设备，简化奖励模块的硬件分配。
- 类似地，Environment Worker 也可按需扩展，支持并行环境交互，减少等待延迟。样本级控制允许 Actor 不等待当前环境响应即可处理新样本，实现 **异步环境交互**。鉴于 Environment Worker 可能为 CPU 密集型，ROLL 会将其分散部署，避免与其他计算任务冲突。

## 5 实验

我们在两类代表性任务上对 ROLL 进行了全面评估：

1. **RLVR（可验证奖励强化学习）pipeline**
   涵盖数学、代码与通用推理三大领域，验证 ROLL 在“可验证奖励”场景下的正确性与有效性。

2. **智能体 RL pipeline**
   覆盖 Sokoban、FrozenLake、WebShop 三种差异化环境，验证 ROLL 在“多轮智能体交互”场景下的通用性与扩展性。

### 5.1 RLVR pipeline

#### 5.1.1 数据收集
实验数据系统性地来源于三个公开且经过清洗的数据集：

- **数学领域**：DeepMath-103K[^H25]
  按难度分层抽样 5 000 题。
- **代码领域**：KodCode[^X25]
  过滤低质量样本后，按难度均匀抽样 2 000 题。
- **通用推理**：Multi-subject-RLVR[^S25]、Nemotron-CrossThink[^A25]、RLVR-IFEval[^L24]
  去除低质量样本，保留多主题与指令跟随任务。

#### 5.1.2 训练设置
- **实验模型**
  - Qwen2.5-7B-Base
  - Qwen3-30B-A3B-Base（MoE 架构，总参 200 B+）

- **算法**
  PPO 损失，优势估计采用 REINFORCE return，而非 GAE[^S15]，以减少超参。

- **样本混合比例**
  数学 40 %、代码 30 %、通用推理 30 %。

- **奖励计算**
  - 数学：规则验证
  - 代码：沙箱执行
  - 通用：规则 + LLM-as-a-Judge

- **配置链接**
  - [7B 配置](https://github.com/alibaba/ROLL/blob/main/examples/qwen2.5-7B-rlvr_megatron/rlvr_config.yaml)
  - [30B MoE 配置](https://github.com/alibaba/ROLL/blob/main/examples/qwen3-30BA3B-rlvr_megatron/rlvr_config.yaml)

#### 5.1.3 性能结果
**Qwen2.5-7B-Base**
- 整体准确率：0.18 → 0.52（**2.89× 提升**）
- 数学推理：0.20 → 0.53
- 代码生成：0.13 → 0.41
- 其余任务见图 3，均呈稳定上升趋势，无模型崩溃。

**Qwen3-30B-A3B-Base**
- 整体准确率：0.27 → 0.62（**2.30× 提升**）
- MoE 模型波动更大，但总体仍保持持续上升，最终超越稠密 7 B 模型。

结论：
- 两模型均实现稳定、可重复的准确率提升；
- ROLL 在超大规模 MoE RL 训练中表现出良好的鲁棒性与实用性。



### 5.2 智能体 RL pipeline

我们在 3 个差异显著的环境中展开实验，以充分验证 ROLL 的通用性与扩展能力。

#### 5.2.1 Sokoban

**环境配置**
- 经典推箱子谜题，配置 3 种难度：
  1. SimpleSokoban：6 × 6 地图，1 个箱子
  2. LargerSokoban：8 × 8 地图，2 个箱子
  3. SokobanDifferentGridVocab：6 × 6 地图，使用不同符号集

**训练设置**
- 基座模型：Qwen2.5-0.5B-Instruct
- 8 GPU，rollout batch size = 1024
- PPO + REINFORCE return，advantage/reward clip 10/20，格式惩罚 −0.001
- [完整配置](https://github.com/alibaba/ROLL/blob/main/examples/qwen2.5-0.5B-agentic_ds/agentic_val_sokoban.yaml)

**性能结果（SimpleSokoban）**
- 训练成功率：16.8 % → 26.0 %
- 验证成功率：13.3 % → 35.2 %
- 有效动作率：43.6 % → 73.4 %
跨环境验证亦显示良好迁移性。

#### 5.2.2 FrozenLake

**环境配置**
- 智能体需在结冰湖面从起点走到目标，避免落洞；可选“滑溜”机制引入随机性。

**训练设置**
- 与 Sokoban 共用模型及超参，保持配置一致性。
- [配置链接](https://github.com/alibaba/ROLL/blob/main/examples/qwen2.5-0.5B-agentic_ds/agent_val_frozen_lake.yaml)

**性能结果**
- 训练成功率：16.8 % → 26.0 %（**提升约 55 %**）
- 有效动作率：69.1 % → 88.8 %
- 验证成功率：12.9 % → 23.8 %

此外，仅在 FrozenLake 上训练的模型在 Sokoban 验证集上也获得 23.8 % 成功率，展现跨任务迁移能力。

#### 5.2.3 WebShop

**环境配置**
- 模拟在线购物任务，智能体根据自然语言指令搜索并购买目标商品。
- 动作包括关键词搜索、点击商品、查看详情、选择属性、下单等。
- 单条轨迹最多 50 步，考验长程决策与指令跟随能力。

**训练设置**
- 基座模型：Qwen2.5-7B-Instruct，序列长度 8192
- REINFORCE 算法，advantage clip = 10，格式惩罚 −0.05
- [配置链接](https://github.com/alibaba/ROLL/blob/main/examples/qwen2.5-0.5B-agentic_ds/agentic_val_webshop.yaml)

**性能结果**
- 成功率：训练/验证均从 37 % 提升至 **> 85 %**
- 平均步数：从 7+ 步降至 **≈ 4 步**，效率显著提高

结果表明，LLM 在 ROLL 的训练下，能够有效获得在真实复杂环境中的任务能力与操作效率。

### 小结

| 场景          | 数据集 / 环境         | 模型规模 | 关键指标               | 提升倍数 |
| - |  | -- | - | -- |
| RLVR-数学     | DeepMath-103K         | 7 B      | 0.20 → 0.53            | 2.7×     |
| RLVR-代码     | KodCode               | 7 B      | 0.13 → 0.41            | 3.2×     |
| RLVR-通用     | Multi-subject-RLVR 等 | 30 B MoE | 0.27 → 0.62            | 2.3×     |
| Agent-Sokoban | SimpleSokoban         | 0.5 B    | 成功率 16.8 % → 26.0 % | 1.6×     |
| Agent-WebShop | WebShop               | 7 B      | 成功率 37 % → 85 %     | 2.3×     |

实验表明，ROLL 在**超大规模稠密 / MoE 模型**以及**多场景智能体任务**中均能实现稳定、高效、易扩展的 RL 训练。

## 6 结论

本文介绍了 **ROLL**——一个面向大语言模型（LLM）大规模强化学习（RL）优化的系统框架。ROLL 面向三类核心用户群体：技术先锋、产品开发者和算法研究者。
在系统层面，ROLL 围绕 **Parallel Worker、Rollout Scheduler、Parallel Strategy** 以及 **AutoDeviceMapping** 等关键模块构建，为大规模 RL 训练提供了高效、可扩展且用户友好的基础。

大量实验表明：

- 在 **200 B+ 参数 MoE 模型** 上，ROLL 可在数千张 GPU 上持续训练两周无中断，验证了其 **可扩展性与容错能力**。
- 在 **数学、代码与通用推理** 等可验证奖励任务中，ROLL 带来 **2–3 倍的准确率提升**，且训练过程稳定无崩溃。
- 在 **Sokoban、FrozenLake、WebShop** 等多轮智能体环境中，ROLL 帮助模型在成功率、有效动作率及任务效率等关键指标上均实现 **显著提升**，并展现出跨任务迁移能力。

综上，ROLL 不仅显著加速了 LLM 的 RL 训练，也为未来的算法创新、产品落地与基础设施升级提供了坚实支撑。我们期待社区在 ROLL 之上继续探索更强大的 RL 范式与更广泛的场景应用。
