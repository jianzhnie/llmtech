# 大模型RL框架的演进与发展趋势

## 1. 从SFT到强化学习：模型训练范式的转变

在2024年OpenAI发布O1系列模型之前，主流的机器学习训练方式主要依赖于**有监督微调（Supervised Fine-Tuning, SFT）**。该方法通过让模型学习“标准答案”，并根据预测与真实标签之间的损失（loss）来更新模型参数。训练流程相对简单，PyTorch 和 TensorFlow 等深度学习框架也围绕这一范式构建了丰富的训练加速工具。

然而，随着O1系列模型的发布，模型训练的重心逐渐从SFT向**强化学习（Reinforcement Learning, RL）**转移。SFT逐渐被视为训练过程中的“预热”阶段，其作用被弱化为参数初始化或策略引导。取而代之的是，RL在模型能力提升中扮演了越来越关键的角色。

### 1.1. RL算法的演进与多样化

RL算法本身也在不断迭代与优化。从早期的**DPO（Direct Preference Optimization）**，到经典的**PPO（Proximal Policy Optimization）**，再到近年来涌现出的**GRPO、RLOO、Reinforce++、DAPO**等新方法，RL算法在策略更新方式、稳定性、样本效率等方面持续优化。

尽管DPO因其简洁性曾一度流行，但随着任务复杂度和模型规模的提升，其局限性逐渐显现，目前在实际工程中已较少被采用。尽管如此，主流RL框架的整体结构保持相对一致，核心流程主要包括以下几个阶段：

### 1.2. RL训练流程的三大模块

#### 模块一：策略生成（Rollout）

对应“学生自己寻找答案”的过程。这是RL训练中的**推演阶段（Rollout）**，模型基于当前策略生成响应（action），模拟与环境的交互过程。该阶段是模型推理过程的扩展，通常需要大量采样以获取多样化的行为轨迹。

#### 模块二：奖励评估（Reward Evaluation）

对应“给学生答案打分”的过程。传统上，这一阶段依赖于**奖励模型（Reward Model）**，用于评估生成结果的质量。在当前阶段，由于任务复杂度提升，奖励评估的实现方式也趋于多样化：

- **基于规则的评估（Rule-based）**：在数学、物理、代码等领域，通过结果与规则的匹配度进行打分。
- **轻量级奖励模型**：训练一个小型模型（如7B参数）进行打分，成本可控，且效果良好。

在许多研究项目中，这一模块甚至被简化为Rollout的一部分，未被单独重视。然而，随着**Agent行为模拟**的兴起，尤其是在商业应用场景（如电商、客服等）中，奖励评估的复杂性显著上升，未来该模块的重要性将不断提升。

#### 模块三：策略更新（Policy Update）

对应“学生根据打分来学习”的过程。这是RL训练的核心阶段，基于传统训练框架（如PyTorch、DeepSpeed等），通过修改损失函数实现策略更新。不同算法（如PPO、DPO、RLOO等）在此阶段的实现逻辑有所不同，但整体结构保持一致。

### 1.3 总结

从SFT主导的训练范式到RL驱动的能力提升，大模型的训练流程正经历深刻的变革。RL框架的结构虽然保持稳定，但其各模块的功能、实现方式和重要性正在不断演化。

- **Rollout模块**：面临长上下文、异构任务带来的性能挑战；
- **Reward Evaluation模块**：从简单规则向复杂评估演进，未来可能成为RL训练中的关键瓶颈；
- **Policy Update模块**：依赖于底层训练框架的性能优化与算法迭代。

随着Agent行为模拟、复杂任务建模、多模态交互等方向的发展，RL框架的设计将更加注重模块间的协同、资源调度的高效性以及算法与工程实现的统一性。

## 2. RL训练框架设计与性能优化挑战

当前，主流的强化学习（Reinforcement Learning, RL）训练框架通常被划分为两个核心模块：**训练（Training）** 和 **Rollout（推演）**。

在设计一个高效的RL训练系统时，开发者将面临一系列关键挑战。以下是我们在技术选型与框架设计过程中总结出的三大核心问题。

### 2.1 挑战一：Rollout与训练模块的协同与资源管理

目前，RL训练普遍采用**On-policy**策略，这意味着Rollout与训练过程必须顺序执行。然而，随着模型规模的持续增长，分布式多卡训练已成为必然趋势。

- **Rollout阶段**：主要为内存密集型任务，尤其在处理长上下文（如Chain-of-Thought）时，需要维护大量的KV Cache（Key-Value Cache）。
- **训练阶段**：则属于计算密集型任务，涉及大规模的参数更新和梯度计算。

这两个阶段各自已有大量优化手段（如内存复用、流水线并行等），但如何在统一框架中高效管理这两类异构资源？如何优化两者之间的参数同步机制？这是构建高效RL系统的关键挑战之一。

### 2.2 挑战二：底层训练与推理框架的多样性

当前存在多种主流的训练框架，例如：

- **Megatron-LM**
- **DeepSpeed（FSDP）**
- **PyTorch FSDP**

同时，推理引擎也呈现多样化趋势：

- **vLLM**
- **SGLang**

不同训练框架与推理引擎的架构差异显著，导致在参数同步、推理调度等环节的实现逻辑差异较大。例如，仅在参数更新部分，不同组合就可能需要完全不同的实现逻辑，这对系统的可维护性与扩展性提出了较高要求。

### 2.3 挑战三：异构批次执行带来的不确定性

Rollout任务通常以批次形式执行，但批次内部任务的复杂度可能存在巨大差异。特别是在引入**Agent行为模拟**的场景下，这种异构性更加显著，可能导致整体调度效率下降、资源利用率不均衡等问题。

## 3. 性能优化分析

### 3.1 初始实现与性能瓶颈

在RL训练的早期实现中，整个流程通常分为三个阶段：

1. **推理阶段（Rollout）**：模型根据当前策略生成响应。
2. **评估阶段**：通过奖励模型或其他机制对生成结果进行打分。
3. **训练阶段**：基于打分结果更新策略模型。

该流程本质上可以基于SFT（Supervised Fine-Tuning）框架实现，区别在于需要初始化多个模型实例（如策略模型、奖励模型等）。然而，这种实现方式在实际运行中往往存在显著的性能瓶颈。

### 3.2 内存优化策略

在大规模模型训练中，显存占用主要包括以下几个部分：

- 模型参数（Parameters）
- 梯度（Gradients）
- 优化器状态（Optimizer States）
- 激活值（Activations）

以一个7B参数模型为例，在FP32精度下，仅模型参数和梯度就需要约28GB显存，优化器状态则可能额外占用28GB×3=84GB，总计高达112GB。显然，单卡无法承载如此庞大的内存需求。

为此，业界提出了多种分布式训练策略：

- **数据并行（Data Parallelism, DP）**：如 DeepSpeed ZeRO-1/2/3，通过All-Gather操作动态重建完整参数。
- **张量并行（Tensor Parallelism, TP）与流水线并行（Pipeline Parallelism, PP）**：如 Megatron-LM，采用参数切分策略，适用于大规模模型。

根据NVIDIA相关论文的研究结论，在千卡以下规模，DP与TP/PP性能相近；但在更大规模下，TP/PP因避免了All-Gather操作的通信开销，性能优势更为明显。

| 特性           | 数据并行（DP）     | 张量并行（TP）     | 流水线并行（PP）   |
| -------------- | ------------------ | ------------------ | ------------------ |
| 实现复杂度     | 简单               | 高                 | 中等               |
| 内存冗余       | 高                 | 低                 | 低                 |
| 通信开销       | 中等               | 高                 | 低                 |
| 模型大小限制   | 小                 | 大                 | 大                 |
| 计算资源利用率 | 高                 | 高                 | 中等               |
| 调度复杂度     | 低                 | 高                 | 高                 |
| 适用场景       | 数据量大、模型较小 | 模型超大、计算密集 | 模型深度大、长序列 |

这个表格比较了数据并行（DP）、张量并行（TP）和流水线并行（PP）三种并行策略在不同特性上的表现。

### 3.3 推理速度优化与引擎选型

当前主流推理引擎（如 vLLM 和 SGLang）在KV Cache复用、底层算子优化等方面已实现显著性能提升。尽管如此，训练与推理引擎之间的参数同步仍存在一定挑战：

- 推理引擎生成的输出与训练引擎在精度上存在差异；
- 当前主流做法是：在Rollout阶段使用推理引擎加速生成，训练阶段再由训练引擎重新计算logits（仅需prefill阶段，计算效率高）。

因此，将高性能推理引擎与训练框架进行集成，是提升整体RL训练效率的有效路径。但如何高效地实现训练与推理模块的拼接与协同，仍是值得深入研究的问题。

## 4. 训练框架与推理引擎的整合

### 4.1 SPMD和MPMD概念解析

在讨论训练框架和推理引擎的集成时，首先需要理解两种并行处理模式：**SPMD（Single Program, Multiple Data）** 和 **MPMD（Multiple Programs, Multiple Data）**。这两种模式也可以被描述为单一控制器与多控制器架构。

- **单一控制器（SPMD）**：所有工作节点执行相同的程序逻辑，适用于数据量大但模型规模较小的场景。
- **多控制器（MPMD）**：每个工作节点可以执行不同的程序，增加了实现复杂度，但无需集中控制，适合特定应用场景。

主流的深度学习训练框架如DeepSpeed和Megatron都采用了SPMD模式，保证所有进程遵循相同的代码逻辑进行运算。然而，对于推理引擎（例如SGlang和vLLM），情况则有所不同。尽管推理引擎（例如SGLang和vLLM）在计算过程中遵循SPMD原则，但在决定下一个token来源或如何处理KV缓存等方面，则不完全适用SPMD/MPMD分类。对于这些情况，Google Pathway等系统提供了更灵活的解决方案。

考虑到上述背景，我们更应关注的是训练框架与推理引擎之间关于训练数据和模型参数的通信机制，而非局限于是否采用单一控制器或多控制器架构。

### 4.2 SLIME的具体实现方法

训练框架与推理引擎之间的核心挑战在于训练数据与模型参数的通信机制。为了更好地理解这一点，我们可以通过分析slime和roll项目来探讨具体实现方案。

SLIME是一个专注于强化学习扩展的后训练框架，它定义了两个主要组件：RayTrainGroup用于训练框架，RolloutGroup用于推理引擎。

#### 4.2.1 数据传输机制

SLIME通过定义一个中间件类——Buffer，实现了推理引擎与训练模块间的数据传输。所有的数据都会被存储在这个Buffer中（甚至可以写入磁盘），并通过rollout ID进行指定访问。此外，Buffer类中的数据处理函数以及rollout/eval函数均可以通过命令行参数灵活配置，极大地提高了系统的适应性。

```python
self.generate_rollout = load_function(self.args.rollout_function_path)
self.eval_generate_rollout = load_function(self.args.eval_function_path)
```

这种设计使得应对业务需求时更加灵活高效，尤其是面对各种特殊需求和数据格式时尤为重要。

Rollout 的generate函数是通过Buffer。

```python
def async_generate(self, rollout_id, evaluation=False):
     return self.data_buffer.generate.remote(rollout_id, evaluation=evaluation)
```

获取训练框架所需的数据同样依赖于这个Buffer：

```python
def get_rollout_data(self, rollout_id):
    megatron_utils.process_rollout_data(rollout_id, self.args, self.data_buffer)
```

同步rollout的buffer给actor的过程如下所示：

```python
 def async_init_weight_update_connections(self, rollout):
        """
        Connect rollout engines and actors, e.g. initialize the process group between them
        to update weights after each training stage.
        """
        self.rollout = rollout
        ray.get([actor.set_data_buffer.remote(rollout.data_buffer) for actor in self._actor_handlers])
```

#### 4.2.2 模型参数同步机制

为了让rollout引擎能够在适当的时候正确地同步参数，SLIME将actor的配置信息传递给rollout。这部分涉及到初始化过程组以便在每个训练阶段之后更新权重。

```python
 def async_init_weight_update_connections(self, rollout):
        """
        Connect rollout engines and actors, e.g. initialize the process group between them
        to update weights after each training stage.
        """
        self.rollout = rollout
        ray.get([actor.set_data_buffer.remote(rollout.data_buffer) for actor in self._actor_handlers])
        actor_parallel_configs = ray.get([actor.get_parallel_config.remote() for actor in self._actor_handlers])
        parallel_config = {}
        for rank, config in enumerate(actor_parallel_configs):
            assert config["rank"] == rank and config["world_size"] == len(self._actor_handlers)
            config.pop("rank")
            for key, value in config.items():
                if "size" in key and key:
                    if key not in parallel_config:
                        parallel_config[key] = value
                    else:
                        assert (
                            parallel_config[key] == value
                        ), f"mismatch {key} on rank {rank}: {parallel_config[key]} != {value}"
        parallel_config["actors"] = actor_parallel_configs
        ray.get(rollout.async_set_parallel_config(parallel_config))

        return [
            actor.connect_rollout_engines.remote(
                rollout.rollout_engines,
                rollout.rollout_engine_lock,
            )
            for actor in self._actor_handlers
        ]
```

上述过程不仅包括数据缓冲区的同步，还涵盖了actor间并行配置的协调，保证了参数更新的一致性和准确性。

### 4.3 ROLL的具体实现方法

ROLL通过集群（Cluster）的方式定义了多个角色，每个角色负责不同的任务。这种设计方式与算法层面的认知较为一致，因为从算法角度来看，训练框架和推理引擎之间的差异并不明显，而使用集群封装则很好地隐藏了这些复杂性。

```python
self.actor_train = Cluster(
    name=self.pipeline_config.actor_train.name,
    worker_cls=self.pipeline_config.actor_train.worker_cls,
    resource_manager=self.resource_manager,
    worker_config=self.pipeline_config.actor_train,
)
self.actor_infer = Cluster(
    name=self.pipeline_config.actor_infer.name,
    worker_cls=self.pipeline_config.actor_infer.worker_cls,
    resource_manager=self.resource_manager,
    worker_config=self.pipeline_config.actor_infer,
)
self.reference = Cluster(
    name=self.pipeline_config.reference.name,
    worker_cls=self.pipeline_config.reference.worker_cls,
    resource_manager=self.resource_manager,
    worker_config=self.pipeline_config.reference,
)
if self.pipeline_config.adv_estimator == "gae":
    self.critic = Cluster(
        name=self.pipeline_config.critic.name,
        worker_cls=self.pipeline_config.critic.worker_cls,
        resource_manager=self.resource_manager,
        worker_config=self.pipeline_config.critic,
    )
```

#### 4.3.1 数据传输机制

类似于Megatron，ROLL允许按照领域（domain）分开采样，并在`pipeline.py`文件中进行配置。这使得如果用户不想编写数据生成器，ROLL提供了一种更为便捷的解决方案。特别是对于奖励（reward）模型，理想的状况是有一个统一的模型，但由于训练难度大，目前更倾向于针对不同领域使用不同的奖励模型，并最终进行聚合处理。ROLL支持对不同领域、批次以及查询进行自定义配置，以适应多样的应用场景。

#### 4.3.2 模型参数同步机制

ROLL中的模型更新逻辑结合了点对点通信和集体通信两种方式：

```python
def model_update(self, tgt_workers, broadcast_tgt_devices, p2p_tgt_devices):
    # 更新逻辑代码...
```

- **点对点通信**：用于同一设备上的参数更新，直接通过worker的node_rank和gpu_rank来判断是否在同一设备上，从而进行高效的数据交换。
- **集体通信**：通过广播参数到目标集群，只在主进程（rank 0）执行广播操作，适用于跨设备间的参数同步。

这两种通信策略分别对应于colocate和非colocate场景，确保了参数同步的灵活性和效率。

#### 4.3.4 跨机器部署时的考量

当所有组件位于同一台机器上时，硬编码实现参数同步相对简单，但当涉及到跨机器部署时，情况变得更加复杂。此时，不仅需要考虑如何有效地管理网络通信带来的延迟和带宽限制，还需要优化分布式环境下的资源分配和负载均衡。此外，单控制器（single controller）模式下，控制器的压力会随着集群规模的扩大而增加，尤其是在处理多媒体数据时，可能需要特别注意性能瓶颈的问题。因此，在跨机器部署的情况下，选择合适的通信策略和优化控制器的工作负载变得尤为重要。不过，从SLIME和ROLL的设计来看，参数同步的核心在于通知GPU进行同步操作，中间的通信过程不依赖于控制器，这为跨机器部署提供了一定的便利性和灵活性。

### 4.4 Colocation与Ray的应用

将Actor、Ref、Reward、Critic等模型放置在同一张GPU卡上被称为**colocation**。然而，正如前文所述，随着模型规模的增大（例如7B模型已难以在单张卡上训练），预计下半年会出现多个超过1000B参数量级的模型。这使得并行计算带来的开销变得极其显著。当前，Reward模型普遍较小，7-30B的规模即可满足需求，因此分开部署往往更具性价比。

为了应对这种复杂性，项目中引入了Ray——一个支持分布式计算的强大框架，它能够帮助开发者减轻底层逻辑管理的负担。有关基于Ray的分布式训练流程和Ray分布式计算框架的详细介绍，请参阅以下文章：
- [图解OpenRLHF中基于Ray的分布式训练流程](#)
- [Ray分布式计算框架详解](#)

接下来，我们将比较slime、verl、roll和openrlhf四个框架在colocation与非colocation实现上的差异。

#### 4.4.1 SLIME

SLIME仅定义了两个主要worker：RayTrainGroup用于训练，RolloutGroup用于推理。对于colocate，训练和推理可以分开部署；而在非colocate的情况下，则需要处理分布式通信以同步参数。这种设计抽象层次高，易于理解，并且能够很好地适应训练和推理的不同需求。只需在配置中指定是否colocate，即可自动在所有关键环节执行相应操作。

#### 4.4.2 ROLL

对于非colocate场景，ROLL允许细粒度地指定不同worker（例如actor、critic、reward等）部署在不同的显卡上，甚至可以根据轮次进行配置。若不手动指定，Ray会自动完成部署。鉴于RL任务对资源的高消耗，细粒度的GPU资源配置有助于提高资源利用效率，但这同时也对算法侧的资源调度能力提出了更高要求。显然，使用Ray来管理这些复杂性更为合适。

#### 4.4.3 Verl

VERL采用了一种独特的方法来实现colocate和非colocate部署。在非colocation模式下，每个worker（如actor、critic、reward等）作为一个独立进程运行，依靠Ray来进行调度。而在colocation模式下，多个角色共享同一个Ray actor实例，在同一进程中实例化多个worker类。通过`create_colocated_worker_cls`或`create_colocated_worker_cls_fused`方法动态生成一个多角色类（例如WorkerDict/FusedWorker），该类内部持有多个worker实例。外部可通过统一接口调用不同角色worker的方法，内部则自动分发到对应的worker实例。这种方式使得同进程内的多角色共存成为可能，并在某些场景下能大幅提高性能，比如减少跨进程通信带来的延迟和内存碎片问题。

#### 4.4.4 OpenRLHF

OpenRLHF提供了灵活的混合部署选项，既支持vLLM引擎、Actor、Reference、Reward和Critic模型节点的共置部署，也支持部分混合部署或完全分离部署，以适应异步训练的需求。这种灵活性使其能够应对多样化的应用场景，但也意味着更复杂的管理和优化需求。

#### 4.4.5 结论

综上所述，在非colocation情况下，Ray确实可以帮助我们更加轻松地管理资源，尤其是在处理复杂的Agent和多轮交互场景时。然而，根据运维团队的反馈，Ray的设计理念与现有的Kubernetes云原生生产环境存在一定的冲突，导致在实际生产环境中部署时管理成本较高。不过，Ray团队也在针对这些问题进行优化，例如使Ray可以直接通过NCCL传输tensor数据，从而绕过对象存储，提高效率。未来，我们可以期待更多来自Ray的更新和改进。

### 4.5 不同训练框架与推理引擎的集成

在将不同的训练框架和推理引擎进行集成时，可能会遇到参数转换的问题。例如，如果vLLM使用4-维 张量并行（TP），而DeepSpeed分布在8个GPU上，则需要进行适当的参数转换以确保数据传输的一致性。Megatron-LM也有类似的需求。当存在多个训练框架和推理引擎时，适配的工作量会成倍增加，这可能导致配置错误和性能问题。

### 4.6 代码解耦设计

以Slime为例，其架构分为三层：顶层RolloutGroup负责管理推理引擎的整体流程；中层RolloutRayActor处理具体的推理请求；底层SglangEngine实现具体的推理逻辑。这种分层设计使得替换后端推理引擎变得简单，只需更改底层实现即可，无需修改上层控制逻辑。同样，训练框架也采用了类似的分层结构，保证了系统的灵活性和可维护性。

## 5. 关于Agentic RL

目前，roll、verl和openrlhf等框架对Agentic RL提供了良好的支持。尽管这样做可能增加了代码复杂度，但随着技术成熟，预计会有更清晰的设计出现。未来，Agentic RL有望成为主流，现有的RL方法将成为其中的一部分。

## 6. 框架选择建议

### 6.1 框架难点分析

快速发展的技术环境意味着旧框架容易过时，因此保持框架简洁和高维护性是关键。新框架由于没有历史负担，可以更容易地适应新技术趋势。

### 6.2 推荐框架

- **OpenRLHF**：一个高性能的开源RLHF框架，集成了Ray、vLLM、ZeRO-3和HuggingFace Transformers。
- **slime**：新推出的框架，代码简洁，适合想要尝试大胆框架修改的研究者。
- **ROLL**：强调数据处理和异步操作的支持，特别适用于深入探索Agentic RL的团队。
- **verl**：稳定且优化良好，适合大规模集群部署，尤其适合资源丰富的团队。

根据团队的具体需求和技术背景，可以选择最适合的框架来开展工作。对于有特定需求或希望快速扩展的团队，verl可能是更好的选择，因为它已经被多个大厂验证过。而对于追求技术创新和敏捷开发的团队，slime或ROLL可能更具吸引力。

### 结尾

在过去半年中，我们深入探讨了RL训练框架、Agent框架以及推理引擎框架。总体而言，代码量方面，Agent框架最为庞大，其次是推理引擎和RL训练框架；而在代码难度上，推理引擎居首，随后是RL训练框架和Agent框架。值得注意的是，如果排除推理引擎底层算子的复杂性，RL训练框架的挑战主要在于集成各种系统和技术，这要求框架开发者对多种技术和业务逻辑有深刻的理解。

开源框架如verl、slime、roll及openRLHF各具特色，展现了作者们的追求与坚持，并且社区活跃度高。可以说，在开源RL框架领域，中国在技术实力和认知深度方面处于世界领先位置。虽然算法人才间的差异不大，但在硬件资源（如显卡）方面仍存在一定的差距。
