# HybridFlow：一种灵活高效的 RLHF 框架

- *HybridFlow: A Flexible and Efficient RLHF Framework*

- 论文链接：https://arxiv.org/pdf/2409.19256v2

- 代码链接：https://github.com/volcengine/veRL

- 在线文档：https://verl.readthedocs.io/en/latest/index.html

## 摘要

从人类反馈中进行强化学习（Reinforcement Learning from Human Feedback, RLHF）已成为大语言模型（Large Language Model, LLM）对齐人类价值观的核心技术。传统的强化学习过程可被建模为一个有向无环图（Directed Acyclic Graph, DAG）形式的数据流，其中节点表示神经网络的计算操作，边表示计算之间的数据依赖关系。在 RLHF 中，每个节点被扩展为一个分布式 LLM 的训练或生成任务，而每条边则对应复杂的多对多数据广播（multicast）操作，显著提升了系统的复杂性。

现有强化学习框架通常采用单一集中式控制器来协调节点内计算与节点间通信，但在 RLHF 场景下，由于节点本身已是高度分布式的计算单元，这种架构会引入显著的调度开销，导致效率低下。另一方面，当前主流的 RLHF 系统采用Multi-Controller范式，虽然提升了计算效率，但由于缺乏统一协调机制，导致系统在面对多样化 RLHF 算法时缺乏灵活性，难以实现代码复用与模块化设计。

为此，我们提出 **HybridFlow**，一种融合Single-Controller与Multi-Controller优势的混合式 RLHF 框架，旨在实现 RLHF 数据流的**灵活建模**与**高效执行**。我们设计了一套分层 API，将复杂的分布式计算与数据依赖关系进行解耦和封装，支持用户以简洁方式编排 RLHF 算法流程，并灵活地将模型映射到不同设备集合上。进一步地，我们提出了 **3D-HybridEngine**，用于在 Actor 模型的训练与生成阶段之间高效地重新分配模型分片，实现零内存冗余并显著降低通信开销。实验结果表明，HybridFlow 在多种 RLHF 算法、模型规模和集群配置下，相较于当前最先进的系统，吞吐量提升了 **1.53 倍至 20.57 倍**。HybridFlow 已开源，代码地址：[https://github.com/volcengine/verl](https://github.com/volcengine/verl)。

### 关键词

分布式系统，从人类反馈中进行强化学习（RLHF）

## 1. 引言

近年来，大语言模型（LLM）如 GPT [11]、Llama [73] 和 Claude [7] 在写作 [2]、搜索 [52]、代码生成 [63] 等人工智能任务中展现出卓越能力。LLM 通常经历三个阶段：首先在海量文本上通过自回归语言建模进行预训练，以积累广泛知识；其次通过监督微调（Supervised Fine-Tuning, SFT）适应特定指令任务；最后，为缓解预训练数据中潜在的有害或偏见内容，引入 **从人类反馈中进行强化学习（RLHF）**，以实现模型与人类价值观的对齐，构建安全、有益的 AI 系统 [7, 55]。

典型的基于近端策略优化（PPO）[68] 的 RLHF 系统包含四个核心模型：

- **Actor 模型**：生成Response；
- **Critic 模型**：估计状态价值；
- **Reference Policy 模型**：提供生成行为的参考分布；
- **Reward Model**：根据人类偏好打分。

整个 RLHF 流程以迭代方式进行，每轮包含三个阶段：

1. **生成阶段**：Actor 模型对一批Prompt（prompt）进行自回归生成，产出Response；
2. **准备阶段**：利用生成结果，由 Critic、Reference Policy 和 Reward Model 分别计算价值、对数概率和奖励，构建训练数据；
3. **训练阶段**：基于上述数据，通过前向与反向传播更新 Actor 和 Critic 模型。

其他 RLHF 变体（如 Safe-RLHF [19]、ReMax [43]）虽在模型结构或流程上有所调整，但整体仍遵循三阶段范式。

从系统视角看，RLHF 可建模为一个复杂的**分布式数据流图**：

- **节点**：代表一个 LLM 的分布式训练或推理任务；
- **边**：表示节点间的数据依赖与通信，常涉及多对多的模型分片重分布（re-sharding）。

由于 LLM 规模庞大，各模型通常采用不同的并行策略（如Data Parallel、Pipeline Parallel、Tensor Parallel），且在训练、推理、生成等不同阶段计算特性差异显著。因此，如何高效调度和协调这些异构任务，成为 RLHF 系统设计的关键挑战。

传统 RL 框架（如 RLLib [45]、RLLib Flow [46]）采用**Single-Controller范式**：由一个中央控制器协调所有节点的执行顺序与资源分配。然而，该模式在面对 LLM 级别的复杂节点时，控制调度开销巨大，难以扩展。

现有 RLHF 系统转而采用**Multi-Controller范式**：每个设备独立运行控制器，通过点对点通信协调任务。该模式虽降低了调度延迟，但缺乏全局视图，导致节点间通信逻辑深度耦合，修改数据流需同步调整多个模块，系统灵活性差，维护成本高。

为克服上述局限，我们提出 **HybridFlow**，其核心思想是：**在节点间采用Single-Controller范式以实现灵活调度与协调，在节点内采用Multi-Controller范式以保障高效分布式计算**。具体贡献如下：

- 提出一种**分层混合编程模型**，通过封装模型类与传输协议，解耦节点内计算与节点间通信，支持灵活构建各类 RLHF 数据流（第 4 节）；
- 设计 **3D-HybridEngine**，实现 Actor 模型在训练与生成阶段间的零冗余、低通信开销的权重重分配（第 5 节）；
- 开发自动化的**GPU 映射算法**，优化各模型在异构设备上的放置策略，最大化资源利用率（第 6 节）；
- 实验验证 HybridFlow 在多种场景下相较 SOTA 系统，吞吐量提升 **1.53× 至 20.57×**，且已开源以推动 RLHF 技术发展。

## 2. 背景与动机

<img src="https://arxiv.org/html/2409.19256v2/x1.png" alt="Refer to caption" style="zoom:50%;" />

> 图 1. 三种 RLHF 算法数据流图（Ouyang 等，2022；Dai 等，2024；Li 等，2023a）。 阶段①、②、③分别表示生成、准备与训练。

### 2.1 从人类反馈中进行强化学习（RLHF）

RLHF 的目标是将 LLM 的输出空间与人类偏好对齐。其典型流程（以 PPO 为例）如图 1 所示：

1. **生成阶段**：Actor 模型对输入Prompt生成Response序列；
2. **准备阶段**：对生成结果，分别由 Critic 模型计算状态价值 $V(s)$，Reference Policy 计算参考对数概率 $\log \pi_{\text{ref}}(a|s)$，Reward Model 输出偏好得分 $r(a|s)$；
3. **训练阶段**：基于上述信号，通过优化目标函数更新 Actor 与 Critic 模型。

其目标函数通常为：

$$
\mathcal{L}^{\text{PPO}} = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]
$$

其中 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$，$\hat{A}_t$ 为优势估计。

其他 RLHF 变体（如 Safe-RLHF、ReMax）虽在模型构成或流程上有所差异，但均需支持灵活的数据流建模能力。

### 2.2 并行策略

大型语言模型（LLM）的训练与 Server 通常采用Data Parallel（Data Parallelism, DP）、Pipeline Parallel（Pipeline Parallelism, PP）和张量并行（Tensor Parallelism, TP）相结合的策略

- **Data Parallel（Data Parallelism, DP）**：将数据分片，各设备处理不同批次，此外，ZeRO [59] 和 PyTorch FSDP [57] 通过分片优化器状态、梯度与参数，进一步降低内存占用。

- **Pipeline Parallel（Pipeline Parallelism, PP）**：将模型按层划分为多个阶段（stages），各阶段分布于不同设备上顺序执行；

- **Tensor Parallel（Tensor Parallelism, TP）**：将模型中的大张量（如权重矩阵）沿特定维度进行切分，使得单个设备仅需处理张量的一部分。

- **3D 并行**： 现代分布式训练框架如 Megatron-LM [71] 和 MegaScale [36] 普遍采用 **3D 并行**（或称 PTD 并行）[54]，实现高效扩展。其中 P、T、D 分别代表 PP、TP 和 DP。在 3D 并行中：

  - **PP 规模**（p）表示模型在Pipeline Parallelism 维度上的阶段数；
  - **TP 规模**（t）表示张量被切分的份数；
  - **DP 规模**（d）表示模型副本的数量。

  三者共同构成并行组结构，总设备数满足 N=p*×*t*×*d。

在 LLM 推理与生成服务中，3D 并行策略同样被广泛采用，但与训练不同的是，通常仅对模型参数和 KVCache 进行分片 [16, 29, 40]，以支持高效的自回归生成。

在 RLHF 中，不同模型（Actor、Critic、Reference、Reward）执行不同任务（训练、推理、生成），其计算密度、内存需求各异，需支持灵活的并行策略组合。

- **训练**（Training）：包含一次前向传播、一次反向传播及模型参数更新；
- **推理**（Inference）：仅执行一次前向传播；
- **生成**（Generation）：通过多次前向传播实现自回归文本生成。

具体而言：

- Actor模型（Actor）同时参与训练与生成；
- Critic模型（Critic）参与训练与推理；
- Reference Policy模型（Reference Policy）和奖励模型（Reward Model）仅参与推理。

由于各模型在计算模式、内存访问模式和通信特征上存在显著差异，针对不同模型和不同计算阶段采用差异化的并行策略，有助于实现整体吞吐量的最优化。例如，生成阶段通常对延迟敏感且计算密度较低，适合采用较小的 TP 和 PP 规模以减少通信开销；而训练阶段则更注重吞吐，可利用更大的 DP 规模提升Data Parallel效率。因此，灵活组合并行策略是构建高效 RLHF 系统的关键。

### 2.3 分布式机器学习的编程模型

<img src="https://arxiv.org/html/2409.19256v2/x2.png" alt="Refer to caption" style="zoom:50%;" />

> **图2. RLHF系统中采用的编程模型。** (a) 现有的RLHF系统采用Multi-Controller（multi-controller）范式，系统中存在多个长期运行的、协同工作的控制器进程。(b) HybridFlow采用混合编程模型：由一个Single-Controller（single-controller）负责协调不同模型间的执行流程；而每个模型内部的分布式计算则沿用Multi-Controller范式。图中灰色表示的非活跃节点，代表该操作在此时未执行。

#### Single-Controller（Single-Controller）模型

Single-Controller架构采用一个**集中式控制器**来管理整个分布式程序的执行流程。在这种模式下，控制逻辑集中于单一节点，用户可以将数据流的核心功能实现为一个**单进程程序**（如图2(b)所示），而控制器会自动在后端生成并调度分布式工作节点（workers）执行实际计算任务。

由于控制器拥有对底层硬件资源和数据流图的**全局视图**，该范式支持灵活且优化的资源映射策略，并能够协调数据流中各任务之间的执行顺序，从而实现高效的调度与依赖管理。

然而，该模型的局限性在于：所有协调指令均需从中央控制器发送至各个工作节点。当在大规模集群上运行复杂或扩展性强的数据流图时，这种集中式通信会带来显著的**调度开销**（dispatch overhead），成为系统扩展性的瓶颈 [1, 9]。

#### 多控制器（Multi-Controller）模型

在多控制器架构中，每个设备（也称为工作节点，worker）都配备一个本地控制器。当前最先进的分布式大模型（LLM）训练与推理系统广泛采用这一范式，主要因其具备良好的**可扩展性**和较低的调度开销 [36, 40, 60, 71]。控制消息通常通过高速 PCIe 链路在本地 CPU 与 GPU 之间传递，避免了跨节点控制通信的延迟。

如图2(a)所示，在基于多控制器的 RLHF 实现中，每个模型运行独立的程序，且同一模型的所有工作节点执行相同的代码。每个工作节点仅具备系统的**局部视图**，模型间的执行顺序协调依赖于**点对点通信**（如图中蓝色代码与箭头所示）。

在该架构下实现 RLHF 工作流时，开发者必须在每个设备运行的程序中**显式集成**集体通信（如 all-gather）、计算逻辑以及点对点数据传输的代码。这导致计算与通信操作深度嵌套，程序结构复杂，开发、维护和性能调优难度显著增加。

例如，在图2(a)中，各模型需执行本地计算及 all-gather 操作（黑色代码部分），而Actor模型（Actor）必须在特定时机主动调用 `send` 操作，将生成结果发送给Critic（Critic）和奖励模型（Reward Model）；后者则必须在程序的精确位置实现对应的 `receive` 操作，以确保数据同步与流程正确性。这种对通信时序的高度依赖，进一步加剧了编程复杂性。

#### 小结

| 特性       | Single-Controller        | 多控制器                   |
| ---------- | ------------------------ | -------------------------- |
| 控制方式   | 集中式                   | 分布式                     |
| 视图范围   | 全局视图                 | 本地视图                   |
| 调度开销   | 高（尤其在大规模集群）   | 低                         |
| 编程复杂度 | 低（用户编写单进程程序） | 高（需手动管理通信与同步） |
| 可扩展性   | 受限                     | 优异                       |

因此，理想的 RLHF 系统应结合两种范式的优点：**保留Single-Controller的简洁编程接口**，同时**利用多控制器的高效分布式执行能力**，从而实现高性能与易用性的统一。

### 2.4 RLHF 的系统特性

<img src="https://arxiv.org/html/2409.19256v2/x3.png" alt="Refer to caption" style="zoom:50%;" />

> 图3. 给定模型放置方案下的数据流执行. 图中带编号的方块代表GPU。虚线框内的模型被放置在不同的设备集上，可以并发执行。参考模型（蓝色）和奖励模型（绿色）被共置于同一组GPU上，并按顺序依次执行。

#### 1. 模型工作负载异构性

在RLHF（Reinforcement Learning with Human Feedback）中，Actor（Actor）、Critic（Critic）、参考策略（Reference Policy）和奖励模型（Reward Model）可能在不同阶段执行训练、推理或生成任务，这导致了不同的内存占用和计算需求。具体而言：

- **Actor 与 Critic**：参与模型训练过程， 需存储模型参数、梯度、优化器状态，内存占用高；
- **Reference 与 Reward 模型**：仅需前向推理，只需将其模型参数存储于GPU内存中， 内存需求较低；
- 模型规模可能不一致（如 7B Actor + 70B Reward 模型），需差异化并行策略。

鉴于这种异质性，针对每个模型运行时所需的并行策略和优化措施应有所不同。

#### 2. Actor 训练与生成的计算失衡

- **训练**：计算密集型，适合高模型并行度（MP）；
- **生成**：内存密集型（需维护 KV Cache），适合高Data Parallel度（DP）与低 MP。

若两阶段使用相同并行配置，会导致资源利用率低下。若动态切换，则面临**模型权重重分配**的通信开销问题。

在RLHF数据流中，Actor的训练和生成由两个节点表示（参见图3），这些节点通常占据了每次RLHF迭代中的大部分工作量（例如，使用HybridFlow时占总时间的58.9%）。Actor训练是计算密集型任务[Geoffrey et al., 2021]，往往需要更大的模型并行度（即模型被划分成的分区数），并将Work Load分布到更多的GPU上，比如将7B模型分成8个分区部署在8块GPU上。然而，对于生成任务来说，由于其内存限制特性[Kwon et al., 2023]，采用相同的并行策略（如相同的MP大小）可能会导致GPU计算资源利用率低下。研究表明，结合较大的Data Parallel度（DP）和较小的模型并行度（如将7B模型分成两部分并在8块GPU上复制四次）可以提高生成吞吐量[Li et al., 2023b; Zhong et al., 2024]。

尽管为Actor训练和生成采用不同的并行策略可能同时优化这两个阶段的吞吐量，但在两个阶段之间实时重新分配Actor模型权重会导致显著的通信和内存开销。例如，调整一个70B的Actor模型要求在每次RLHF迭代中从训练切换到生成时传输140GB的模型权重，这可能占据迭代时间的36.4%，尤其是在两阶段位于不同设备上时[Hu et al., 2023]。

#### 3. 模型放置策略多样性

- **分离部署**：各模型独立运行，支持并行执行，但可能导致 GPU 空闲；
- **共置部署**：多个模型共享 GPU 集合，通过时分复用执行，避免 OOM，但可能引入串行瓶颈。

如何根据模型依赖关系与资源约束，自动选择最优放置策略，是提升整体吞吐的关键。

根据各模型的计算Work Load和数据依赖关系，战略性地将模型置于RLHF数据流中的适当位置是必要的。图3展示了一个模型放置计划示例及其对应的RLHF执行流程。

如果不存在数据依赖关系，则放置在不同设备集上的模型可以并行执行；而放置在同一组GPU上的模型被称为共置模型，它们共享GPU内存，并按时间分片顺序执行，以避免并发执行时可能出现的内存不足（OOM）错误。

我们注意到一种权衡：虽然将模型放置在不同设备上允许并行处理，但由于RLHF中模型执行的阶段性特征，这可能导致某些GPU空闲。例如，在图3中，Actor和Critic被分别放置，虽能并行进行训练，但在其他RLHF阶段期间，各自有约三分之一的GPU时间处于闲置状态。支持多种放置策略并最大化设备利用率对于优化任何规模模型和集群的RLHF性能至关重要。

### 2.5 现有系统的局限性

| 系统           | 并行策略                    | Actor 权重管理         | 模型放置     | 执行模式         |
| -------------- | --------------------------- | ---------------------- | ------------ | ---------------- |
| DeepSpeed-Chat | 训练：ZeRO；生成：TP        | 训练/生成间重分配      | 全部共置     | 顺序执行         |
| OpenRLHF       | 训练：ZeRO；生成：TP        | 维护两个副本（冗余）   | 分离部署     | 并行执行部分阶段 |
| NeMo-Aligner   | 训练/生成均用 3D 并行       | 共享权重，但生成吞吐低 | 部分共置     | 混合并行         |
| **HybridFlow** | **训练/生成均支持 3D/FSDP** | **零冗余动态重分配**   | **灵活支持** | **自适应调度**   |

现有系统存在以下问题：

1. **灵活性不足**：多采用硬编码的通信逻辑，难以支持新 RLHF 算法；
2. **效率低下**：权重冗余或频繁重分配导致高通信开销；
3. **放置策略固化**：无法根据负载动态优化资源分配。

### 2.6 设计思路

我们提出一种**分层混合编程模型**，结合Single-Controller与Multi-Controller的优势：

- **节点间**：采用Single-Controller协调数据流执行顺序与通信调度，实现灵活编排；
- **节点内**：采用Multi-Controller执行分布式训练/推理，保障计算效率。

该设计的关键优势在于：

- 解耦计算与通信，提升模块化与可维护性；
- Single-Controller调度开销小（节点数少），而Multi-Controller保障内部计算高效；
- 支持灵活的模型放置与并行策略组合。

如图 2(b) 所示，HybridFlow 将模型封装为独立组件，通过统一 API 进行调度，避免了传统Multi-Controller系统中复杂的点对点通信逻辑嵌套。

> **总结**：HybridFlow 通过**混合控制范式**与**分层 API 设计**，实现了 RLHF 系统在**灵活性**与**效率**上的统一，为下一代对齐训练系统提供了可扩展、易用且高性能的基础设施支持。

## 3. HybridFlow 概述

<img src="https://arxiv.org/html/2409.19256v2/x8.png" alt="Refer to caption" style="zoom:50%;" />

> 图 4.HybridFlow 架构

图4展示了 HybridFlow 的架构，主要由三个核心组件构成：混合编程模型、3D-HybridEngine 和自动映射算法。混合编程模型通过一组分层 API 实现 RLHF 数据流的灵活表示及模型高效计算（详见 §4）。3D-HybridEngine 特别设计用于 Actor 模型训练与生成的高效执行，允许在训练和生成阶段之间实现零内存冗余和最小化通信开销的模型参数重新分配（详见 §5）。自动映射算法确定每个模型在 RLHF 数据流中的最优设备放置，以最大化系统的吞吐量（详见 §6）。

### RLHF 系统的工作流程

本系统的 RLHF 工作流程如下：用户在启动系统前需提供以下三类输入信息：

1. **模型规格（Model Specifications）**：包括 RLHF 数据流中各模型（如Actor/Critic/reference policy/reward model 等）的网络架构与参数规模；
2. **模型设备部署方案（Device Placement）**：即各模型在 GPU 集群中的设备分配策略，通常通过在给定集群配置下运行自动映射算法（auto-mapping algorithm）获得；
3. **并行策略（Parallelism Strategy）**：为每个模型在各个计算阶段指定并行执行策略。例如，采用 3D 并行时，可表示为三元组 $(p, t, d)$，其中 $p$、$t$、$d$ 分别表示Pipeline Parallel（PP）的组大小、Tensor Parallel（TP）的组大小和Data Parallel（DP）的组大小。

Single-Controller程序接收上述输入后，执行以下操作：

- 初始化 RLHF 数据流中的各个模型；
- 构建虚拟化的资源池（Resource Pool），将逻辑模型映射到物理设备；
- 根据预设的部署方案，将模型或操作分发至相应的设备；
- 调用运行在各设备上的Multi-Controller程序，触发各模型的分布式计算任务。

Multi-Controller程序负责实现 `ParallelWorker` 类的具体逻辑，其主要职责包括：

- 根据各模型的并行策略，在分配的设备间构建对应的并行组（如 TP、DP、PP 组）；
- 对Actor模型的训练与生成阶段，调用 **3D-HybridEngine** 进行高效执行；
- 无缝集成主流的大型语言模型（LLM）引擎（如 Megatron-LM (Shoeybi et al., 2019)、DeepSpeed (Rasley et al., 2020)、PyTorch (Paszke et al., 2019)、ColossalAI (Kwon et al., 2023)），支持其他模型的训练、推理与生成任务。

模型间的数据重分片（Data Resharding）——包括Prompt（prompts）、Response（responses）及其他中间输出——由Single-Controller程序协调传输协议统一管理，以支持在采用不同并行策略的模型之间高效传输数据。特别地，Actor模型在训练与生成阶段之间的数据重分片由 **3D-HybridEngine** 专门处理，确保计算与通信的高效协同。

该分层协同架构结合Single-Controller的全局调度能力与Multi-Controller的本地执行能力，实现了算法逻辑与底层分布式实现的解耦，从而在保证性能的同时，显著提升了系统的可编程性与灵活性。

## 4. 混合编程模型

<img src="https://arxiv.org/html/2409.19256v2/x9.png" alt="Refer to caption" style="zoom:50%;" />

> **图5. 分层API的示意图。** (a) 展示了一个具有3D并行配置的模型，包括资源分配和3DParallelWorker初始化的过程。该图详细描述了如何设置一个模型的3D并行结构（包括Pipeline Parallel、Tensor Parallel及Data Parallel），以及如何初始化3DParallelWorker来管理这些并行计算任务。(b) 使用3D_PROTO协议，在两个模型之间进行异步数据重新分片的示例。通过`collect`函数从各工作节点收集数据，并使用`distribute`函数将数据重新分发，实现了高效的数据重分配过程。此部分展示了在复杂分布式训练场景下，如何利用HybridFlow中的高级API实现不同模型间数据的有效管理和调度。

### 4.1 分层 API

#### 节点内部：封装分布式程序

针对不同RLHF阶段中各模型的分布式计算，我们提供了一个基类`3DParallelWorker`。在给定设备分配的情况下，它简化了分布式模型权重初始化，并为每个模型建立了3D并行组。一个并行组包括一组GPU，用于承载模型的特定并行维度，例如TP中的不同张量分片和DP中的不同模型副本。图5(a)展示了使用我们的API进行Actor模型初始化的过程，其他模型的初始化方式类似。

继承自`3DParallelWorker`基类，我们提供了多个模型类，分别对应 Actor、Critic、reference 和 reward model 等。这些模型类封装了实现模型分布式前向/反向计算、自回归生成及优化器更新的API，将分布式计算代码与其他模型的数据依赖解耦。这些API可以通过复用现有LLM系统的计算脚本轻松实现。例如，ActorWorker 中的`update_actor`函数与Megatron-LM(Shoeybi et al., 2019)的预训练脚本相似。每个模型类封装了实现各类RLHF算法的基本操作，如Actor模型类中的`generate_sequences`用于基于Prompt生成Response，奖励模型类中的`compute_reward`通过前向传播评估Response质量（更多API详见附录A）。

除了实现3D并行的`3DParallelWorker`基类外，我们还提供了支持PyTorch FSDP(`FSDPWorker`)和ZeRO(`ZeROWorker`)的基类，以及各自基类对应的模型类，以支持不同的模型计算并行策略。图4中的`ParallelWorker`即指代这些基类之一。

#### 节点间：统一模型间的数据重新分片实现

当采用不同并行策略的模型在不同设备间传输数据时，涉及到多对多组播通信。我们通过为每个模型类的操作关联传输协议（使用@register装饰器）来统一数据转移实现。每个传输协议包含收集函数和分发函数，根据各模型的并行策略聚合输出数据和分配输入数据。如图5(a)所示，`update_actor`操作注册到了3D_PROTO传输协议，因为Actor训练采用了3D并行。在3D_PROTO中，收集函数会将每个DP组内相应模型函数（例如`update_actor`返回的损失标量）的输出数据汇总至单一控制器，而分发函数则将输入数据（例如`update_actor`所需的优势值）分配至各DP组。通过源模型的输出收集函数和目标模型的输入分发函数实现了数据重新分片。

图5(b)展示了Actor（生成阶段）与Critic（推理阶段）间的数据重新分片过程，两者采用不同的3D并行策略：

- 单一控制器使用Actor的3D_PROTO的收集函数聚集数据futures（步骤①-③）并发送给Critic（步骤④）；
- Critic使用其自身的3D_PROTO的分发函数将接收到的数据futures分配到各DP组（步骤⑤）。
- 随后，Critic的每个GPU根据其DP等级仅获取Actor输出数据中所需的本地批次（步骤⑥），实际数据传输仅发生在GPU之间，避免了中心节点瓶颈。

我们提供了8种传输协议（包括3D_PROTO、DP_PROTO、ONE_TO_ALL等），覆盖大多数数据重新分片场景（详情见附录B）。用户还可以通过实现自定义收集/分发函数扩展传输协议。

#### 灵活的模型部署

我们提供了一个`ResourcePool`类，对一组GPU设备进行了虚拟化。当将`ResourcePool`实例应用于模型类（图5(a)）时，该模型的分布式计算将被映射到指定设备。使用相同`ResourcePool`实例的模型会被共置于同一组GPU上，而应用不同`ResourcePool`实例的模型则部署于不同的GPU集上。我们假设不同`ResourcePool`实例间不存在设备重叠。

#### 异步数据流执行

当模型部署在不同的设备集上时，其执行会在输入数据准备就绪时自动触发(Moritz et al., 2018)。如图5(b)所示，控制器调用后立即返回来自Actor的数据futures（步骤①-③）；然后控制器发起对Critic的新调用，并按照传输协议分发futures（步骤④-⑤）。当一些模型部署在同一设备集上时，它们将根据调用顺序依次执行。通过我们的编程模型，HybridFlow无需修改RLHF算法代码即可灵活支持多种分布式执行模式（图6）。

### 4.2 不同 RLHF 算法的实现

<img src="https://arxiv.org/html/2409.19256v2/x10.png" alt="Refer to caption" style="zoom:50%;" />

> 图 6.PPO（Ouyang 等，2022）、ReMax（Li 等，2023a）和 Safe-RLHF（Dai 等，2024）的实现方案。用户仅需增删少量代码即可适配不同 RLHF 算法。

我们提供的 API 极大地简化了各类 RLHF 算法（即数据流）的开发流程。用户只需编写少量代码，即可将整个 RLHF 算法实现为运行在Single-Controller上的单进程程序。该程序通过调用一系列基础 API 来触发模型的分布式计算。图 6 展示了 PPO、ReMax 和 Safe-RLHF 等典型算法的实现示例。

以 PPO 算法为例，其核心逻辑仅需 8 行代码即可完成，主要通过调用 `compute_values` 和 `generate_sequences` 等模型操作实现。这些操作在底层基于Multi-Controller范式，在多个 GPU 上并行执行。对于 Safe-RLHF 算法，其在 PPO 基础上引入了额外的成本模型（cost model）用于评估安全性偏好，并在Actor模型中加入预训练损失项。我们仅需在 PPO 实现的基础上额外增加约 5 行代码即可完成适配。类似地，对于 ReMax 算法，只需增加一次Actor生成调用，并移除与Critic相关的代码即可完成转换。

#### 实现的灵活性

这种高度可扩展的特性对于研究人员探索新型 RLHF 算法至关重要：开发者可以复用各模型类中已封装好的分布式计算逻辑，仅需根据具体算法调整数值计算部分的代码，例如在 `compute_advantage` 及Actor与Critic的损失函数中实现 GAE（Generalized Advantage Estimation, $A_t^\text{GAE}(\gamma, \lambda)$）或 KL 散度（KL divergence, $D_\text{KL}(p \| q)$）等：

$$
A_t^\text{GAE}(\gamma, \lambda) = \sum_{l=0}^{T-t} (\gamma \lambda)^l \delta_{t+l}
$$

其中 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ 为时序差分误差。

$$
D_\text{KL}(p \| q) = \sum_x p(x) \log \frac{p(x)}{q(x)}
$$

这种简洁高效的开发模式得益于所提出的混合编程模型。模块化的 API 设计不仅简化了开发流程，促进了代码的广泛复用，还使得现有大型语言模型（LLM）训练/服务框架的代码库能够被直接集成。同时，该设计有效解耦了模型计算与模型间的数据传输逻辑。因此，对底层分布式框架的任何修改均不会影响上层 RLHF 算法的实现代码（如图 6 所示），从而支持对每个模型的执行过程进行独立优化（见第 5 节）。此外，系统还支持根据不同模型的计算负载进行灵活的部署配置，从而实现 RLHF 数据流在多样化硬件设备上的高效映射与执行（见第 6 节）。

## 5. 3D-HybridEngine

<img src="https://arxiv.org/html/2409.19256v2/x11.png" alt="Refer to caption" style="zoom:50%;" />

> 图 7. 单次 RLHF 迭代中的 3D-HybridEngine 工作流程。使用 4 块 GPU 进行Actor 模型的训练与生成：训练阶段采用 1-2-2并行组配置，生成阶段采用 1-1-2-2并行组配置。

我们设计了 **3D-HybridEngine**，旨在支持Actor模型的高效训练和生成，目标是显著提升RLHF（Reinforcement Learning from Human Feedback）的吞吐量。

### 5.1 并行组

为了消除冗余的Actor模型副本，我们建议将Actor模型的训练和生成阶段部署在同一组设备上，即分配给Actor的 $N_a$ 个 GPU，并在相同的Actor模型权重副本上顺序执行这两个阶段。然而，Actor模型的训练和生成可能采用不同的3D并行策略，例如，生成阶段通常需要较小的Tensor Parallel（TP）和Pipeline Parallel（PP）规模，但Data Parallel（DP）规模较大（§2.3）。在这种情况下，**3D-HybridEngine** 能够实现在同一组设备上高效地重分片Actor模型参数，以适应训练和生成的不同需求。

假设 $(p, t, d)$ 表示为Actor训练构建的3D并行组，其中 $p$ 表示Pipeline Parallel的阶段数，$t$ 表示张量切片的数量，$d$ 表示模型副本的数量（Narayanan et al., 2021）。**3D-HybridEngine** 根据各自不同的3D并行策略，分别为Actor训练和生成构建不同的并行组。在生成阶段，我们分别用 $p_g$、$t_g$ 和 $d_g$ 表示生成Pipeline Parallel组、生成Tensor Parallel组和微Data Parallel组的大小。$d_g$ 表示生成阶段中模型副本数量相对于训练阶段的比例，即每个训练中的Data Parallel副本在生成时变为 $d_g$ 个微Data Parallel副本，用于处理 $d_g$ 个Prompt和Response的微批次。因此，我们有 $N_a = p \times t \times d = p_g \times t_g \times d_g \times d$，且满足 $d_g = \frac{pt}{p_gt_g}$。这些微Data Parallel组仅在Actor生成阶段使用，以实现更大的Data Parallel规模，从而充分利用所有设备资源。生成并行组表示为 $p_g-t_g-d_g-d$。

### 5.2 3D-HybridEngine工作流程

在RLHF的迭代 $i$ 中，Actor训练结束后，在迭代 $i+1$ 的Actor生成开始前，Actor模型参数需要根据两个阶段的并行组配置进行重分片，同时Prompt数据也需要重新分布。

- 在RLHF的迭代 $i+1$ 中，**3D-HybridEngine** 首先收集迭代 $i$ 更新后的Actor模型参数（图2中的步骤①），以便在每个微Data Parallel组内进行生成。
- 然后，Prompt 批次被加载到每个模型副本中（步骤②），生成Response（RLHF的生成阶段）。
- 接下来，**3D-HybridEngine** 在每个微Data Parallel组内对生成结果执行All-Gather操作（all-gather operation），并根据3D并行策略重新划分模型参数（步骤③），以适应Actor训练的需求。当模型权重、Prompt和Response正确重新分布后，计算Actor模型的损失，并根据RLHF算法更新Actor模型权重（步骤④），进入迭代 $i+1$ 的Actor训练阶段。

通过上述设计，**3D-HybridEngine** 实现了Actor模型训练和生成过程中的高效并行计算和资源利用，显著提升了RLHF系统的整体性能。

<img src="https://arxiv.org/html/2409.19256v2/x12.png" alt="Refer to caption" style="zoom:50%;" />

### 5.3 零冗余模型重分片

在3D并行中，常用的并行组构建方法如下：Pipeline Parallel（PP）和Tensor Parallel（TP）组通过将连续的秩分配给管道阶段和张量切片来形成；Data Parallel（DP）组则通过选择由PP规模和TP规模乘积确定的固定间隔内的秩来构建。如图2(a)所示，在Actor训练过程中使用了3D并行组1-4-2：所有GPU属于一个PP组（为简化说明），TP组包括[G1, G2, G3, G4]、[G5, G6, G7, G8]，而DP组则有[G1, G5]、[G2, G6]、[G3, G7]、[G4, G8]。假设相同的并行组构建方法被用于生成阶段，但采用不同的并行规模，例如在图2(a)中生成阶段使用1-2-2配置。在从训练过渡到生成的过程中，**3D-HybridEngine** 在模型并行组之间应用All-Gather 操作以聚合所有参数，然后根据设备所属的并行组保留每个设备上的一组模型权重子集。

在某些GPU上（如G2、G3、G6、G7），训练和生成阶段的模型权重没有重叠，因此需要单独的内存来保存后续训练所需的权重（见图2(a)中的灰色框）。我们将这种系统称为 **HybridFlow-V**，当 **3D-HybridEngine** 在两个阶段中使用上述基本的并行组构建方法时。

我们进一步设计了一种新的并行组构建方法，专用于生成阶段，以消除权重存储的冗余，并最小化由于Actor模型重分片导致的内存占用和通信开销。具体而言，我们通过在生成TP或PP维度上以固定的间隔选择秩来形成生成TP和PP组，该间隔分别由 $\frac{t}{t_g}$ 和 $\frac{p}{p_g}$ 确定，并通过沿生成TP或PP维度顺序分配秩来构建微Data Parallel（DP）组。如图2(b)所示，使用1-2-2-2并行组进行生成：生成TP组为[G1, G3]、[G2, G4]、[G5, G7]、[G6, G8]，而微DP组为[G1, G2]、[G3, G4]、[G5, G6]、[G7, G8]。这种生成并行组的战略性重新排列使得训练和生成阶段的模型权重在每个设备上重叠，从而在生成过程中可以重用训练权重，并实现设备内存使用上的零冗余。此外，**3D-HybridEngine** 在每个微DP组内并发执行多个All-Gather操作，显著减少了通信开销。

##### 表2 训练与生成之间的转换开销对比

|          | DS-Chat              | HybridFlow-V       | HybridFlow                     |
| -------- | -------------------- | ------------------ | ------------------------------ |
| 通信量   | $\frac{tpd-1}{tpd}M$ | $\frac{tp-1}{tp}M$ | $\frac{tp-t_gp_g}{t_gp_gt_p}M$ |
| 峰值内存 | $M$                  | $M$                | $\frac{1}{t_gp_g}M$            |
| 冗余度   | $\frac{1}{tpd}M$     | $\frac{1}{tp}M$    | $0$                            |

### 5.4 转换开销

在表2中，我们比较了不同Actor引擎设计在训练和生成阶段转换过程中的通信开销和内存占用。假设Actor模型大小为 $M$，且使用 $N_a$ 个GPU进行训练和生成。DeepSpeed-Chat的Actor引擎在转换期间对所有GPU执行All-Gather操作；而 **HybridFlow-V** 则在训练TP和PP组内执行此操作。这些操作的通信量分别为 DeepSpeed-Chat 的 $\frac{N_a-1}{N_a}M=\frac{tpd-1}{tpd}M$ 和 **HybridFlow-V** 的 $\frac{tp-1}{tp}M$，计算依据为 (Chan et al., 2007)。两种引擎首先在每个GPU的内存中聚合所有模型参数，然后根据生成并行组划分模型状态，导致峰值内存使用量达到模型参数 $M$。由于在某些GPU上无法重用生成期间的训练权重，因此需要维持训练权重，这分别导致了 $\frac{1}{tpd}$ 和 $\frac{1}{tp}$ 的冗余内存消耗。

利用我们在生成阶段设计的并行组方法，**HybridFlow** 将All-Gather操作限制在每个微DP组内。通信开销减少至 $\frac{d_g-1}{tp}M=\frac{tp-t_gp_g}{t_gp_gt_p}M$。每个GPU仅需在其微DP组内收集远程参数，并可在生成过程中重用训练权重。因此，**HybridFlow** 中模型参数的峰值内存使用量精确匹配了生成阶段每个GPU上的模型分区大小，消除了GPU内存使用的任何冗余。

## 6. 自动设备映射

我们的混合编程模型要求用户输入以下配置，这些配置被称为RLHF数据流到给定设备的映射：(a) 数据流中模型的设备放置；(b) 每个阶段运行每个模型所对应的并行策略。

我们提供了一种高效的算法（算法1），帮助用户识别在给定设备集群上执行RLHF数据流的优化映射，以最小化每次RLHF迭代的端到端延迟。对于给定的数据流 $D$，我们首先探索所有可能的模型放置方案 $\mathcal{P}$（第3行）。例如，PPO算法涉及四个模型，根据贝尔划分问题（Bell, 1934; Rota, 1964）会产生15种可能的放置方案，从完全独立放置（即所有模型位于不同设备上）到集中放置（如DeepSpeed-Chat的放置方式）。我们将位于同一组GPU上的模型称为共置集。基于共置模型的内存消耗，我们确定了分配给每个共置模型集的最小GPU数量 $A_{\text{min}}$，确保不会出现内存溢出错误（第9行）。

接下来，从最小GPU分配 $A_{\text{min}}$ 开始，我们枚举所有可行的设备分配给每个共置模型集（第10-12行）。给定共置集的设备分配 $A$ 和该集合中模型的计算Work Load $W$，我们在 `auto_parallel` 模块中探索每种模型的优化并行策略，以最小化模型执行延迟。Work Load $W$ 包括输入和输出形状以及每个模型的计算类型（训练、推理或生成）。在 `auto_parallel` 中，我们利用模拟模块 `simu` 来估计不同并行策略的延迟，参考先前的研究（Zhong et al., 2024; Zheng et al., 2022; Yuan et al., 2024; Li, 2023）（附录C有详细说明）。

`d_cost` 模块通过遍历数据流图中的所有阶段并累加各阶段的延迟来估算在给定模型放置和并行策略下的RLHF数据流端到端延迟（第17、25行）。对于在同一共置集内且在同一阶段进行计算的模型（如Actor和Critic在RLHF训练阶段同时更新模型），它们的执行延迟将被累加（第32行）。对于不同共置集内的模型，在同一阶段内的执行可以并行化，该阶段的延迟由不同集合中的最大执行时间决定（第33行）。我们识别出最佳的模型设备放置及其对应的并行策略，以实现每次RLHF迭代的最小执行时间（第18-23行）。

通过上述方法，我们能够有效地为RLHF数据流找到最优的设备映射方案，从而显著降低整体计算延迟，提高系统性能。

### 算法复杂度分析

算法1的复杂度为 $O(\frac{(N-1)!}{(k-1)!(N-k)!})$，其中 $k$ 表示数据流中的模型数量，$N$ 表示运行数据流所需的总设备数量。这是在枚举所有可能的设备分配方案以确定放置策略（即独立放置）时的最坏情况复杂度，该计算通过将 $N$ 个设备分配给 $k$ 个模型来完成（这被称为整数划分问题（Andrews and Eriksson, 2004））。为了提高效率，我们缓存了在不同数量的设备 $A$ 上识别出的并行策略，以消除当模型被放置在不同设备集上时对相同并行策略的冗余搜索。

尽管我们在运行自动映射算法时假设了 $N$ 个同构GPU，但算法1可以轻松扩展到优化异构设备上的模型映射，只需考虑 `simu` 和 `auto_parallel` 模块中的异构设备即可（Zhang et al., 2024a）。

## 7. 实现

**HybridFlow** 的实现大约包含12k行Python代码（LoC）。

### 混合编程模型

层次化的API实现约有1.8k行代码。集中式Single-Controller基于Ray（Moritz et al., 2018）构建，并使用远程过程调用（RPC）来协调不同模型的执行顺序和数据传输，以遵循数据流。这些中间数据存储在TensorDict中（Paszke et al., 2019）。在我们的Multi-Controller分布式计算范式中，每个模型函数在不同的设备上运行，控制消息从每个控制器的CPU进程传递到相应的GPU。我们的实现支持Megatron-LM、PyTorch FSDP和DeepSpeed作为LLM训练和推理引擎，并支持vLLM用于自回归生成。在vLLM中，我们用分布式管理器替换了集中式的KVCache管理器，以适应Multi-Controller范式。

### 3D-HybridEngine

其主要逻辑在Megatron-LM和vLLM之上实现了2.4k行代码。我们将Actor模型的训练和生成阶段权重存储在单独的内存缓冲区中，在训练期间将生成权重卸载到CPU内存，在转换期间重新加载生成权重回到GPU内存，并在生成过程中同时使用两个缓冲区。我们使用NCCL通信原语（Jeaugey, 2017）在每次微Data Parallel组内的模型参数收集和拼接过程中进行操作。在训练和生成之间的转换后，我们将KVCache卸载到CPU内存，并在下一次迭代中重新加载回GPU。

### 自动映射算法

自动映射算法实现了1.9k行代码，并附带三个模拟器分别用于训练、推理和生成Work Load。该算法在启动RLHF数据流之前在CPU上运行，以生成设备映射和并行策略，用于数据流初始化。

## 8. 评估

### 8.1 实验设置

**测试平台。** 我们在由16台机器（共计128个GPU）组成的集群上部署了HybridFlow，每台机器配置了8个NVIDIA A100-80GB GPU，并通过600GB/s NVLink互连。机器间的带宽为200Gbps。实验使用的软件版本包括CUDA 12.1、PyTorch 2.1.2、Megatron-core 0.6.0、NCCL 2.18.1和vLLM 0.3.1。

**模型和RLHF算法。** 在实验中运行了PPO [68]、ReMax [43] 和 Safe-RLHF [19] 等RLHF数据流（见图1）。每个模型基于Llama [73]，规模从7B到70B不等。对于Actor和Critic的训练采用了混合精度计算，即BF16用于模型参数，FP32用于梯度和优化器状态，所有实验中使用Adam [38] 作为优化器。除非另有说明，实验结果主要来自PPO。

**基线。** HybridFlow与最先进的RLHF系统进行了比较，如DeepSpeed-Chat [82] v0.14.0、OpenRLHF [30] v0.2.5和NeMo-Aligner [17] v0.2.0（详见表1）。性能指标为RLHF吞吐量（tokens/sec），通过将全局批次中的Prompt和Response总标记数除以一个RLHF迭代时间来计算。所有性能数据均是在10次预热迭代后平均超过5次训练迭代的结果。

**数据集和超参数。** 实验基于HuggingFace的“Dahoas/ful-hh-rihf”数据集 [7] 进行，该数据集广泛应用于LLM对齐研究 [64, 85]。为了确保公平比较，我们强制所有生成的Response长度相同。输入Prompt长度和输出Response长度均为1024，输入Prompt的全局批次大小设定为1024。

### 8.2 端到端性能

图9、10和11分别展示了HybridFlow在执行PPO、ReMax和Safe-RLHF时的RLHF吞吐量表现。总体来看，HybridFlow在所有模型规模上均表现出色，相较于其他基线系统有显著提升。例如，在PPO场景下，HybridFlow比DeepSpeed-Chat、OpenRLHF和NeMo-Aligner分别高出3.67倍（最高可达7.84倍）、3.25倍（最高5.93倍）和12.52倍（最高20.57倍）。

**可扩展性。** 在8个GPU上，HybridFlow实现了至少2.09倍的加速比。随着GPU数量的增加，其强扩展效率达到了66.8%。即使在大规模GPU集群（如128个GPU）上运行较小规模的模型（如7B模型），HybridFlow依然能提供优于最佳基线的性能，表明其能够适应不同规模的模型和集群配置。

### 8.3 模型放置

本节探讨了HybridFlow中PPO算法的不同模型放置策略的效果。实验结果显示，在一定数量的GPU（16至64个）上，共置策略提供了最佳性能。然而，当GPU数量增加至96至128个时，分割策略或独立策略可能会更优，这取决于具体的模型规模和集群配置。这些发现强调了选择合适的模型放置策略的重要性，以最大化系统的吞吐量。

### 8.4 3D-HybridEngine

HybridFlow通过减少过渡时间和提高生成阶段的效率，进一步增强了整体性能。特别是在处理大型模型（如70B模型）时，HybridFlow平均减少了55.2%的过渡时间，最高达89.1%。此外，通过调整并行组大小，特别是减小生成TP组大小，可以显著降低生成延迟。

### 8.5 算法运行时间

算法1的运行时间明显短于实际RLHF训练的时间，显示出良好的可扩展性。大部分时间花费在估计各模型的最佳并行策略上。通过对每个模型在特定设备数量上的最佳并行策略进行缓存，可以有效减少搜索时间。

## 10. 相关工作

### 10.1 强化学习（RL）框架

强化学习领域已涌现出大量框架，涵盖从面向小规模深度神经网络（DNN）的一般用途系统（如 Liu et al., 2021; Zhang et al., 2020; Li et al., 2021; Wang et al., 2022），到专为大语言模型（LLM）对齐优化的RLHF系统（如 Li et al., 2021）。本文在第2节中已对相关工作进行了详细讨论，此处进一步补充。现有RL框架（如 Li et al., 2021）通常采用Multi-Controller架构实现其算法逻辑，通过构建多个长期运行的分布式进程，并依赖硬编码的数据同步机制协调各组件的执行顺序。例如，Gear（Gear, 2006）在经验回放缓冲区管理方面进行了优化。然而，这些框架普遍缺乏对LLM训练、推理与自回归生成等复杂计算模式的原生支持。

### 10.2 大语言模型训练与服务系统

早期的分布式训练框架，如TorchDDP（PyTorch, 1995）和Horovod（Horovod, 1994），主要支持Data Parallel（Data Parallelism, DP）。ByteScheduler（Hinton et al., 2015）和DeepSpeed（Hinton et al., 2015）在此基础上，通过通信优化和内存管理技术进一步提升了Data Parallel的可扩展性。针对大模型训练，许多系统（如 Huang et al., 2019; Li et al., 2021）引入了Tensor Parallel（Tensor Parallelism, TP）和Pipeline Parallel（Pipeline Parallelism, PP）等模型并行策略，将模型参数分布到多个设备上进行计算。在LLM服务领域，系统（如 Li et al., 2021）同样采用数据与模型并行技术加速自回归生成，并引入了连续批次（continuous batching）和分块预填充（padded prefilling）等专门优化技术以提升吞吐量。值得注意的是，上述系统普遍采用Multi-Controller范式以实现高效的分布式计算。

### 10.3 数据流系统

经典的数据流系统，如MapReduce（Huang et al., 2019）、Spark（Hinton et al., 2015）、Dryad（Hinton et al., 2015）和Naiad（Hinton et al., 2015），在数据分析和机器学习Work Load中广泛应用，但通常缺乏对动态任务图的灵活支持。Ray（Ryu et al., 2019）通过在单一动态任务图中统一任务并行和Actor编程模型，实现了可扩展的分布式调度器与全局控制存储，因此被众多RL框架（如 Li et al., 2021）所采用。Pathways（Papineni et al., 2019）是为TPU设计的闭源系统，旨在简化复杂并行模式（如Pipeline Parallel、稀疏计算的专家混合模型）以及单个DNN模型内部的细粒度控制流表达。其采用异步分布式数据流设计，能够在存在数据依赖的情况下实现控制平面的并行执行，从而降低Single-Controller架构的调度开销。Pathways主要聚焦于单模型训练，需要对DNN模型的每个子网络进行复杂的编译。HybridFlow具备将Pathways作为子模块集成的能力，以支持RLHF数据流中模型的高效计算。

## 11. 结论

HybridFlow是一个专为强化学习与人类反馈（RLHF）设计的高效框架，能够灵活表示并高效执行多样化的RLHF算法。我们提出了一种混合编程模型，允许用户通过将不同大语言模型（LLM）的分布式计算封装为原始API，轻松构建复杂的RLHF数据流，同时在节点间自动处理数据重新分配的复杂性。3D-HybridEngine确保了Actor模型在训练与生成阶段的高效切换，实现了零内存冗余和显著降低的通信开销。此外，我们提出的优化映射算法有效指导了RLHF数据流中模型的GPU资源分配与部署策略。广泛的实验验证表明，HybridFlow在多种模型规模和集群配置下，相较于当前最先进的RLHF系统，性能提升达1.53倍至20.57倍。

## 附录

### 附录 A：HybridFlow 中的原始 API

在HybridFlow中，我们通过继承 `3DParallelWorker`、`FSDPWorker` 和 `ZeROMorker` 类，实现了RLHF训练中各模型的核心操作。这些模型类提供的函数旨在解耦底层分布式计算逻辑，为用户提供RLHF流程中的基本操作原语。该设计与现有分布式推理和训练框架中的自回归生成、前向传播、反向传播及模型更新操作兼容。用户可根据具体算法需求，通过调整这些函数中的数值计算逻辑，灵活定制RLHF训练流程，并复用底层高效的分布式实现。表4详细说明了这些API的功能与具体计算。

### 附录 B：传输协议

我们定义了一套传输协议，覆盖了RLHF数据流中模型间数据重新分配的所有典型场景。用户可直接使用这些预定义协议构建任意RLHF数据流。此外，用户可通过实现 `collect` 和 `distribute` 函数，轻松扩展自定义传输协议。这些协议有效解耦了复杂的数据重分配逻辑与分布式训练主流程。我们使用 $p$、$t$、$d$ 分别表示Pipeline Parallel（Pipeline Parallelism）、Tensor Parallel（Tensor Parallelism）和Data Parallel（Data Parallelism）组内的Worker（worker）排名。表3列出了这些预定义协议的具体实现。

### 附录 C：自动并行算法

算法2概述了为每个模型搜索最优并行策略的流程。该过程从每个模型的最小并行规模开始（以避免在多Worker间重新分配时发生内存溢出（OOM）），然后根据GPU总数和每台机器的GPU数量 $U$ 枚举所有可行的并行配置。默认情况下，$U$ 设为8。我们利用 `simu` 模块，基于每个模型的Work Load特性估算其执行延迟。该模块包含三个分析型模拟器，分别针对训练、推理和生成Work Load，其设计依据为先前研究 [42, 84, 92]。其中，训练和推理负载属于计算密集型，而生成负载则为内存密集型。

对于Actor模型，我们首先确定其训练阶段的最优并行策略，并记录相应的内存占用。在生成阶段，根据批次大小和最大序列长度计算KV缓存（KVCache）的需求。若当前并行配置无法同时容纳模型参数和KVCache，则增加并行规模。最终，通过比较不同配置下的延迟估算，确定兼顾KVCache分配的最优策略。未来，开发一个能考虑可变KVCache大小的全面自回归生成模拟器，将进一步增强RLHF研究中的自动化映射能力。

### 表 3：HybridFlow 中的传输协议

| 传输协议        | 分发函数                                                     | 收集函数                                                     | 用例                                                         |
| --------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ONE_TO_ALL      | 将数据广播到所有Worker。                                     | 从所有Worker收集数据。                                       | 所有Worker具有相同的输入并执行相同代码，例如模型初始化。     |
| 3D_PROTO        | 将数据拆分，分散到所有Data Parallel（DP）Worker并在组内广播。 | 从所有Data Parallel组中的 $p\sim i$、$t\sim\theta$ Worker收集并连接数据。 | 模型在每个Data Parallel组内的多个Worker间共享。模型输出仅存在于最后一个管道阶段，并在Data Parallel组间复制。这是Megatron-LM、DeepSpeed等3D并行训练的典型场景。 |
| 3D_ALL_MICRO_DP | 按微Data Parallel（micro-DP）大小拆分数据，分散到所有微DP组并在组内广播。 | 从所有微DP组中 `local_rank=0` 的Worker收集并连接数据。       | 与HybridEngine配合使用。用于处理策略模型在3D并行方案下，训练与推理模式切换时的权重分发。 |
| 3D_PP_ONLY      | 将数据广播到所有Worker。                                     | 从所有Pipeline Parallel（PP）组中的 $t\sim\theta$、$d\sim\theta$ Worker收集并连接数据。 | 用于检查权重名称，因为这些名称在Tensor Parallel（TP）和Data Parallel（DP）组内是相同的。 |
| DP_PROTO        | 将数据按批次拆分并分散到所有Data Parallel（DP）Worker。      | 从所有Data Parallel（DP）Worker收集并连接数据。              | 以纯Data Parallel模式训练模型。                              |
| ALL_TO_ALL      | 无操作。                                                     | 从所有Worker收集数据。                                       | 用于调试。用户可手动为每个Worker定义输入，并分别检查其输出。 |

### 表 4：各模型类提供的核心函数。用户可利用这些函数，以极简代码实现多样化的RLHF算法。

| 模型             | API                    | 计算                         | 说明                                                         |
| ---------------- | ---------------------- | ---------------------------- | ------------------------------------------------------------ |
| Actor            | `generate_sequence`    | 自回归生成                   | 基于一批Prompt，Actor模型生成Response序列，并返回每个生成标记的对数概率。 |
| Actor            | `compute_log_prob`     | 一次前向传播                 | Actor模型计算给定Prompt-Response对中每个标记的对数概率。此结果与生成阶段返回的对数概率精度一致（在PPO中可选）。 |
| Actor            | `compute_loss`         | 一次前向传播                 | Actor模型基于预训练数据集计算预训练损失。                    |
| Actor            | `update_Actor`         | 一次前向、反向传播及模型更新 | 基于优势函数（由`compute_advantage`计算）和回报，Actor模型计算训练损失并更新权重。我们实现了多种损失函数，支持PPO [55]、Safe-RLHF [19]、ReMax [43]、GRPO [70] 等多样化RLHF算法。 |
| Critic           | `compute_values`       | 一次前向传播                 | Critic模型计算每个Prompt-Response对的值函数估计。            |
| Critic           | `update_Critic`        | 一次前向、反向传播及模型更新 | 基于值函数估计和回报，Critic模型计算均方误差损失以更新权重。我们同样实现了适用于PPO [55]、Safe-RLHF [19]、ReMax [43]、GRPO [70] 等算法的Critic损失函数。 |
| Reference Policy | `compute_ref_log_prob` | 一次前向传播                 | 参考模型计算Prompt-Response对中每个标记的对数概率。该结果作为基准，用于衡量Actor模型的偏离程度，从而约束其学习过程。 |
| Reward           | `compute_reward`       | 一次前向传播                 | 奖励模型通过前向计算，为给定的Prompt-Response对生成奖励分数。奖励可以是标记级或样本级。 |
| -                | `compute_advantage`    | 数值计算                     | 基于值模型和奖励模型的输出，该函数通过数值方法估计给定Prompt和当前策略下Response的优势函数。此计算不涉及任何模型的前向传播。 |
