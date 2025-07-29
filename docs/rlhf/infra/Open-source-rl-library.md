# 面向 LLM 的开源强化学习库

## 强化学习在大语言模型中的应用

强化学习（Reinforcement Learning, RL）已成为现代大语言模型（Large Language Models, LLMs）开发中的关键技术手段。除了在模型对齐中广泛使用的基于人类反馈的强化学习（Reinforcement Learning with Human Feedback, RLHF）外，基于可验证奖励的 RL 方法也逐渐成为拓展 LLM 能力的重要路径。随着高质量预训练数据日益稀缺，这种通过奖励机制驱动的训练策略显得愈发重要。

近年来，多个突破性研究验证了该范式的有效性，例如 OpenAI 的推理模型 o1 和 o3，以及开源模型 DeepSeek R1。最新研究进展已将 LLM 强化学习扩展至多轮交互场景，使模型能够作为智能体与环境持续互动以解决复杂任务。这一进展标志着 LLM 正在向具备跨领域自主决策能力的智能体方向演进。

随着 RL 在 LLM 领域展现出巨大潜力，开源生态系统迅速发展，涌现出多个设计理念各异、优化策略不同的 RL 框架。本文从技术角度系统分析主流 RL 库，评估其优势与局限。我们在开发 [SkyRL](https://github.com/NovaSky-AI/SkyRL) 的过程中对现有方案进行了深入研究，旨在通过分享这些技术洞见，帮助研究者和从业者更高效地选择适合其特定需求和应用场景的工具。

需要说明的是，部分比较维度可能存在主观性——我们已尽可能基于客观标准进行评估，并提供各库相关代码链接供读者自行验证。如有表述不当之处，欢迎指正。同时请注意这些库更新迭代速度极快，建议始终参考各库最新文档和代码。

## 参与对比的开源库

我们选择的库基于以下标准：近期活跃度、适用场景多样性（从 RLHF 到智能体 RL）、以及其在开源生态中代表的架构哲学差异。

- **TRL**：Hugging Face 推出的热门库，深度集成其生态系统，专注于 RL 训练环节。
- **Verl**：字节跳动开发的高性能全功能 RL 框架，以可扩展性和支持先进训练技术见长。
- **OpenRLHF**：最早流行的开源 RLHF 库之一，兼具易用性与高性能，为后续框架奠定基础。
- **RAGEN**：Verl 的重要扩展，专注于多轮对话和多样化 RL 环境支持。
- **NeMo-RL**：英伟达推出的全流程后训练框架，采用清晰接口设计和结构化数据流，兼顾扩展性与高性能。
- **ROLL**：阿里巴巴新推出的 RLHF 库，支持推理和多轮智能体训练，兼具研究灵活性与生产级扩展能力。
- **AReaL**：蚂蚁研究院开发的 RL 库，通过异步训练机制提升训练吞吐量和可扩展性。
- **Verifiers**：基于 TRL 构建，简化可验证奖励的多轮 RL 实现，强调易用性。
- **SkyRL**：伯克利新推出的 RL 库，专注多轮智能体训练，采用简洁灵活的设计，兼具高性能执行与多场景适应能力。

## 强化学习库的典型用例与核心组件

强化学习库旨在简化训练能够解决复杂问题的 RL 策略的过程。用户定义具体问题及衡量解决方案质量的奖励函数，而库则处理底层训练机制以开发有效策略。常见的策略训练问题包括：

- [代码生成](https://github.com/All-Hands-AI/OpenHands)：奖励取决于代码是否正确（例如通过单元测试验证）。
- [计算机操作](https://github.com/OpenInterpreter/open-interpreter)：奖励取决于任务是否成功解决。
- [数学证明构建](https://github.com/openreasoner/openr)：当证明对给定命题有效时奖励+1，否则为 0。
- [游戏对战](https://github.com/openai/gym)：奖励取决于游戏最终得分或策略能否通关。

如前所述，LLM 的强化学习库有多个不同应用场景：

- **人类反馈强化学习（RLHF）**：最早的 RL 微调用例，用于使 LLM 与人类偏好对齐。该过程需收集人类偏好数据集，从中学习奖励模型，进而通过 RL 调整 LLM。
- **推理模型**：通过解决数学/科学问题或编程任务提升 LLM 的推理能力。LLM 会先学习输出“思考”标记而非直接给出答案，从而提升模型表现。该过程通常通过可验证奖励的 RL 训练获得，由于 LLM 在推理结束时仅给出单一答案，因此常被称为“单步”RL。
- **智能体与多步 RL**：在此场景中，LLM 作为自主智能体在环境中运行，通过连续多步操作完成任务。这是最复杂的配置，对支持库要求最高。库需协调环境与 LLM，同时处理不同实例间步数差异显著的问题。

## RL 库的核心架构设计

RL 库的核心可分解为两个基本组件：

1. **生成器（Generator）**：配置 LLM 与环境交互以解决问题并计算奖励。
2. **训练器（Trainer）**：根据生成阶段获得的奖励反馈更新模型。

框架的设计理念及组件集成方式决定了其核心哲学与适用场景。有些库更专注 RLHF，有些针对推理模型优化，还有些具备更强大的生成与环境抽象能力，因而更适合多轮 RL 和智能体训练。

![RL 库架构图](https://images.ctfassets.net/xjan103pcp94/3DTHR8PtMcYFYkFUnZVjTH/8aa576362b47aedbce2859583737dca6/image1.png)

### 生成器（Generator）

生成阶段通常是计算成本最高的环节。这涉及在 LLM 上运行推理、执行环境中的动作以及计算奖励值。通常存在一个问题数据集，其中可能包含问题陈述的初始提示和/或智能体操作所需的环境参数。对于可验证奖励的情况，还存在问题正确答案的概念。针对每个问题，当前策略会生成一条轨迹（trajectory），其中包含 LLM 产生的动作序列标记。

生成过程可以任意复杂，可采用树状探索等技术以扩展解空间。支持自定义生成函数的库能够为采样、分支和搜索提供最大灵活性。

环境是生成器的核心组件。早期用于 RLHF 或训练推理模型的 RL 库通常没有显式环境，而是通过奖励模型或自定义奖励函数对 LLM 输出进行评分。后来逐渐引入更显式的 API，通常类似 OpenAI 的 gym 接口（包含 `init`、`step` 和 `reset` 函数）。这支持轨迹生成期间的工具使用，以及 LLM 与环境的多步交互。多数情况下，环境的实际执行发生在独立容器/进程或远程服务器上以实现隔离。

### 训练器（Trainer）

训练器包含核心优化循环，负责处理生成阶段收集的轨迹数据并产出新策略。多数库已标准化支持 PPO 和 GRPO 训练算法。在模型 GPU 分片的底层实现方面，主流选择包括 Hugging Face Trainer、FSDP、DeepSpeed 和 Megatron。

- **FSDP**：凭借与 PyTorch 的深度集成最为简易。
- **DeepSpeed**：具有更激进的参数卸载策略。
- **Megatron**：在大规模场景下性能最优。

多个库的训练代码衍生自 Hugging Face 的 TRL 库，更多则采用底层支持 FSDP 和 Megatron 的 Verl 训练代码。

## 评估维度与对比分析

我们力求保持比较的客观性，因此重点关注可从库代码和文档中客观推导的比较点。我们提供相关链接，使用户能够形成自己的观点并做出选择。

### 采用度（Adoption）

虽然衡量库采用度的任何指标都不完美，但我们仍通过首次发布时间、星标数量、问题总数和贡献者数量作为各库采用度的参考指标。在满足您用例需求（如 RLHF、训练推理模型、训练智能体）且具备所需功能的前提下，更广泛采用的库通常更优，因为它们经过更多场景验证而普遍更健壮，且更容易获得帮助。

### 系统特性（System Features）

除列出库的目标用例外，我们还列举各库可能具备的属性：

- **效率**：支持推理/训练共置或异步训练。
- **可扩展性**：支持集群多 GPU 推理与训练。
- **模块化**：为库的各组件提供清晰接口，可通过实现这些接口改变库行为。
- **灵活性**：尝试支持广泛功能或可轻松修改以适应不同设置。

### 组件设计（Components）

我们为每个库链接“RL 库组件”章节中描述的训练器、生成器和环境组件。这些组件可能并不严格遵循训练器/生成器/环境的三分法——例如某些库将生成器与环境统一为组合式多轮环境。我们同时包含以下关于训练后端、推理引擎和环境层的信息：

#### 训练后端（Training Backend）

我们列出了该库支持哪些训练后端（Hugging Face Trainer、FSDP、DeepSpeed 或 Megatron），以及不同框架开箱即用支持哪些算法。

#### 推理引擎（Inference Engine）

最流行的推理引擎包括 vLLM、SGLang、Hugging Face Transformers 或通过 OpenAI API 接入的外部推理引擎。部署模式可选择将推理引擎与训练引擎同址部署或分离部署：

- **同址部署**：有助于降低 GPU 占用，但灵活性较差。
- **分离部署**：适用于长周期任务、异步推理、不同 GPU 类型、独立扩展等场景。

另一项设计决策是是否支持多推理引擎，以及多推理引擎间的负载均衡由客户端还是服务端实现。

#### 异步支持（Asynchronous Support）

指库能否通过异步运行训练与生成来重叠这两个过程。这种做法可提升资源利用率（特别针对含延迟的长周期推演），但会增加学习动态的复杂度并可能影响稳定性。训练引擎更新权重至推理引擎存在多种同步方式：

- **分布式进程组**：通过 NCCL、gloo 或 CUDA IPC 执行权重同步——同步速度快但缺乏灵活性。
- **检查点加载**：通过将模型权重检查点存储后加载至生成器实现同步——灵活性更高但速度较慢。

#### 环境界面（Environment Layer）

不同库对环境支持存在差异：

- 仅支持无显式环境抽象的单轮交互；
- 具备环境抽象但添加自定义环境需修改或分叉库；
- 支持在库外实现自定义环境。

一般而言，环境与生成过程耦合度越低，且现有环境（如以外部远程服务形式存在）越易于集成，则表现越优。

#### 编排器（Orchestrator）

强化学习涉及环境、推理引擎、训练引擎等需复杂交互的组件（如同址资源上的生成与训练或异步执行场景）。此外，每个组件又包含大量需要大规模运行的进程或容器。因此，具备处理调度、通信、容错和自动扩缩容功能的编排器极具价值。凭借灵活性、可扩展性及异构计算支持能力，**Ray** 已成为众多强化学习库的标准化编排方案。

## 各库概览

本表格提供了各库的概览及不同组件的链接，以便您了解接口编写方式及其灵活性。

| Framework                                                 | First release ⭐Stars📝Total issues🧑‍💻Contributors | Target use case/system properties                            | Components (Data structures)                                 | Training backend / Algorithms                     | Inference engine           | Async support                                                | Environment layer                                            | Orchestrator   |
| --------------------------------------------------------- | ----------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------- | -------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | -------------- |
| [TRL](https://github.com/huggingface/trl)(Hugging Face)   | Jan 2023⭐ 14.4k📝 1.9k🧑‍💻 365                     | RLHF,reasoning/flexibility,scalability                       | [Trainer](https://github.com/huggingface/trl/blob/b773a4c1915c6a5a9356507bb1f443e9eeeda51b/trl/trainer/ppo_trainer.py#L98)[Generator](https://github.com/huggingface/trl/blob/15ff54790b42297d2cf569fba6d7dd44c1c269e3/trl/trainer/utils.py#L1398) ([rollout data](https://github.com/huggingface/trl/blob/8a235a9b71f4c0b77e295afb972fdd7c19a71335/trl/trainer/grpo_trainer.py#L958-L961)) | Hugging Face Trainer /SFT, DPO, GRPO, PPO, others | Hugging Face, vLLM         | ❌                                                            | ❌                                                            | —              |
| [Verl](https://github.com/volcengine/verl)(ByteDance)     | Nov 2024⭐ 10.2k📝 1.0k🧑‍💻 253                     | RLHF,reasoning,agents/flexibility,efficiency,scalability     | [Trainer](https://github.com/volcengine/verl/blob/9ec260be2338e5ed08f64cf481c25a943d3e486f/verl/trainer/ppo/ray_trainer.py#L877)[Generator](https://github.com/volcengine/verl/blob/59379539a0305e4948b742262bf1ba2468213674/verl/workers/rollout/vllm_rollout/vllm_rollout.py#L184) ([rollout data](https://github.com/volcengine/verl/blob/83ebd007e01de29bbe353de112d04245b4820b47/verl/protocol.py#L201-L212)) | FSDP, Megatron / SFT, DPO, GRPO, PPO              | Hugging Face, vLLM, SGLang | 🚧RFC [BufferDataset](https://github.com/volcengine/verl/issues/1172) | 🚧[Custom environment via tools](https://github.com/volcengine/verl/blob/main/verl/tools/gsm8k_tool.py) | Ray            |
| [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)          | Jul 2023⭐ 7.2k📝 0.7k🧑‍💻 73                       | RLHF/flexibility,efficiency,scalability                      | [Trainer](https://github.com/OpenRLHF/OpenRLHF/blob/d86779bb15acb3187f75f1b5ee71aaa9f975ca9d/openrlhf/trainer/ppo_trainer.py#L433) ([rollout data](https://github.com/OpenRLHF/OpenRLHF/blob/5974dfe8bded77e90dcdb92f08f3bdba31972ff2/openrlhf/trainer/ppo_utils/experience_maker.py#L32-L46)) | DeepSpeed /SFT, DPO, GRPO, PPO, others            | Hugging Face, vLLM         | ✅(--async_train)                                             | 🚧via python function                                         | Ray            |
| [RAGEN](https://github.com/RAGEN-AI/RAGEN)                | Jan 2025⭐2.1k📝 0.1k🧑‍💻 14                        | agents/modularity,flexibility,scalability,efficiency         | [Trainer](https://github.com/RAGEN-AI/RAGEN/blob/main/ragen/trainer/agent_trainer.py)[Generator](https://github.com/RAGEN-AI/RAGEN/blob/9fb8527776a8490f240417bd964e6eac75777fb0/ragen/llm_agent/agent_proxy.py#L143)[Environment](https://github.com/RAGEN-AI/RAGEN/blob/main/ragen/env/base.py) ([rollout data](https://github.com/volcengine/verl/blob/83ebd007e01de29bbe353de112d04245b4820b47/verl/protocol.py#L201-L212)) | Verl backend / GRPO, PPO                          | Hugging Face, vLLM, SGLang | ❌                                                            | ✅[Custom environment](https://github.com/RAGEN-AI/RAGEN/tree/9fb8527776a8490f240417bd964e6eac75777fb0?tab=readme-ov-file#adding-custom-environments) | Ray            |
| [AReaL](https://github.com/inclusionAI/AReaL)(Ant Group)  | Feb 2025 ⭐1.9k📝 0.1k🧑‍💻 17                       | RLHF,reasoning,agents/efficiency,scalability                 | [Trainer](https://github.com/inclusionAI/AReaL/blob/f2f4b67bcd375ded638f6e1a02163b80fa60a99e/realhf/impl/model/interface/ppo_interface.py#L527)[Generator/ Environment](https://github.com/inclusionAI/AReaL/blob/f2f4b67bcd375ded638f6e1a02163b80fa60a99e/realhf/impl/agent/math_multi_turn_agent.py#L49) ([rollout data](https://github.com/inclusionAI/AReaL/blob/b63eea9d0761868148b5f6f22b28c6e50d364719/realhf/api/core/data_api.py#L106)) | DeepSpeed, Megatron / GRPO, PPO                   | vLLM, SGLang               | ✅                                                            | ✅[Custom environment](https://inclusionai.github.io/AReaL/customization/agent.html#rollout-and-agentic-rl) | Ray (optional) |
| [Verifiers](https://github.com/willccbb/verifiers)        | Feb 2025 ⭐1.4k📝 0.1k🧑‍💻 6                        | reasoning,agents/flexibility,modularity                      | [Trainer](https://github.com/willccbb/verifiers/blob/a3b7ffcb1d247ab5a557218c8290fdafd82a138d/verifiers/trainers/grpo_trainer.py#L230)[Generator/ Environment](https://github.com/willccbb/verifiers/blob/a3b7ffcb1d247ab5a557218c8290fdafd82a138d/verifiers/envs/multiturn_env.py#L32) ([rollout data](https://github.com/willccbb/verifiers/blob/main/verifiers/trainers/grpo_trainer.py#L699-L701)) | Hugging Face Trainer /GRPO                        | vLLM, OpenAI               | ✅                                                            | ✅[Custom environments](https://github.com/willccbb/verifiers/tree/main/verifiers/envs) | —              |
| [ROLL](https://github.com/alibaba/ROLL)(Alibaba)          | May 2025⭐1.3k📝0.0k🧑‍💻23                          | RLHF,reasoning,agents/modularity,flexibility,scalabilityefficiency | [Trainer](https://github.com/alibaba/ROLL/blob/b756338442d9d36c1fe4c951bc98ebc913bd95ae/roll/pipeline/agentic/agentic_pipeline.py#L109)[Generator](https://github.com/alibaba/ROLL/blob/b756338442d9d36c1fe4c951bc98ebc913bd95ae/roll/agentic/rollout/rollout_scheduler.py#L53)[Environment](https://github.com/alibaba/ROLL/blob/b756338442d9d36c1fe4c951bc98ebc913bd95ae/roll/agentic/env/base.py#L5) ([rollout data](https://github.com/alibaba/ROLL/blob/94637ff15d6ada376f796bb693b5f7abad00cc21/roll/distributed/scheduler/protocol.py#L145)) | DeepSpeed,Megatron / GRPO, PPO, others            | vLLM, SGLang               | ❌(planned)                                                   | ✅[Custom environments](https://alibaba.github.io/ROLL/docs/English/StepByStep/agent_pipeline_start/#environments-baseenv-and-implementations) | Ray            |
| [NeMo-RL](https://github.com/NVIDIA/NeMo-RL)(Nvidia)      | Mar 2025 ⭐0.5k📝 0.2k🧑‍💻 29                       | RLHF,reasoning,agents/modularity,flexibility,scalability,efficiency | [Trainer](https://github.com/NVIDIA-NeMo/RL/blob/0fee015f03fa959b04a236fb656e4e5f8548e277/nemo_rl/algorithms/grpo.py#L325)[Generator](https://github.com/NVIDIA/NeMo-RL/blob/99ba9a130e72cbf87a3e20acf43bc01a47adc8ee/docs/design-docs/generation.md)[Environment](https://github.com/NVIDIA/NeMo-RL/blob/main/nemo_rl/environments/interfaces.py) ([rollout data](https://github.com/NVIDIA-NeMo/RL/blob/ebd35a342a509f6a3ba832e699d440ad08a59ec4/nemo_rl/algorithms/grpo.py#L594-L604)) | FSDP, Megatron /SFT, DPO, GRPO                    | vLLM                       | ✅                                                            | ✅[Environment example](https://github.com/NVIDIA/NeMo-RL/blob/main/nemo_rl/environments/math_environment.py) | Ray            |
| [SkyRL](https://github.com/NovaSky-AI/SkyRL)(UC Berkeley) | Jun 2025⭐0.5k📝 0.0k🧑‍💻 9                         | agents/modularity,flexibility,scalability,efficiency         | [Trainer](https://github.com/NovaSky-AI/SkyRL/blob/b3280d8440f4973fd96229e3e5d3ed2372534812/skyrl-train/skyrl_train/trainer.py#L194)[Generator](https://github.com/NovaSky-AI/SkyRL/blob/b3280d8440f4973fd96229e3e5d3ed2372534812/skyrl-train/skyrl_train/generators/base.py#L24)optional: [Environment](https://github.com/NovaSky-AI/SkyRL/blob/b3280d8440f4973fd96229e3e5d3ed2372534812/skygym/skygym/core.py#L19) ([rollout data](https://github.com/NovaSky-AI/SkyRL/blob/b3280d8440f4973fd96229e3e5d3ed2372534812/skyrl-train/skyrl_train/training_batch.py#L321)) | FSDP, DeepSpeed / GRPO, PPO                       | vLLM, SGLang, OpenAI       | ✅                                                            | ✅ [Custom environment](https://skyrl.readthedocs.io/en/latest/tutorials/new_env.html) | Ray            |

以下是对各库的简要评述，供读者参考。由于多数库迭代迅速，所有观点仅供参考。

### TRL

专为简化操作及融入 Hugging Face 生态系统（如 datasets、transformers、accelerate 和 PEFT）而优化。该库主要面向基于文本的大语言模型后训练场景（如推理、RLHF），除支持监督微调外，还提供 DPO、GRPO 和 PPO 算法支持。若您更关注无需环境交互的文本强化学习场景，TRL 是理想选择。但需注意，该库原生不支持多轮次强化学习与任意环境交互，也缺乏更灵活的生成方案——这促使了类似 TRL 设计理念但支持多轮次 RL 和环境交互的验证器库等替代方案的出现。总体而言，TRL 相比其他库定义了更少内部接口，虽降低了复杂度，但也牺牲了部分适应性。

### Verl

Verl 专为高性能与可扩展性设计，支持所有成熟的训练框架如 FSDP、DeepSpeed 和 Megatron。与 TRL 类似，它最初是用于训练推理模型和进行无环境单轮交互的库，但正在扩展工具调用和多轮强化学习功能。出于性能考虑，生成与训练过程高度耦合。Verl 拥有庞大社区且开发活跃，是最成熟的 LLMs 开源强化学习库之一。目前已有多个项目基于 Verl 进行扩展开发（完整列表见[此处](https://github.com/volcengine/verl?tab=readme-ov-file#awesome-work-using-verl)）。

### OpenRLHF

OpenRLHF 专为强化学习人类反馈（RLHF）设计，对奖励模型和多种优化算法提供了出色支持。该框架同时支持异步训练以及生成与训练的协同定位。虽然具备一定程度的[环境/智能体强化学习](https://openrlhf.readthedocs.io/en/latest/async_rl.html#asynchronous-agent-rl)功能，但尚未提供专用接口。作为成熟的开源项目，OpenRLHF 拥有规模可观的开发者社区，并衍生出多个扩展库，进一步将其功能拓展至[推理](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero)和[多智能体训练](https://github.com/TsinghuaC3I/MARTI)领域。

### RAGEN

RAGEN 是基于 Verl 构建的库，集成了更明确的环境接口，并通过增强的智能体与多轮强化学习支持扩展了 Verl 功能。该库还简化了自定义环境的定义流程。

### AReaL

该库特别注重性能优化，专为生成与训练的异步执行而设计。通过实现可中断的推演流程以及对 PPO 算法的改进，有效解决了异步执行中的数据陈旧问题，从而实现了极致性能表现。

### Verifiers

解决了 TRL 的部分缺陷，并新增了对环境和多轮工具使用的支持。与 TRL 类似，验证器同样基于 Hugging Face 的 Transformers Trainer 框架。该框架简洁易用，深受研究人员青睐。

### ROLL

ROLL 致力于通过提供大量[接口](https://github.com/alibaba/ROLL/blob/b756338442d9d36c1fe4c951bc98ebc913bd95ae/assets/ROLL_map_high.png)并使库可配置化，来吸引各类用户（大型 AI 实验室、从业者和研究人员）。

### NeMo-RL

作为较新的强化学习库之一，NeMo-RL 为各组件提供了明确定义的接口，并对轨迹数据及其他相关数据建立了清晰的数据表示体系。该库专为支持多轮次强化学习与环境交互而设计。与 Verl 类似，NeMo-RL 同样注重性能与可扩展性，但其成熟度相对较低，生态系统也尚未完善。

### SkyRL

SkyRL 融合了上述多个库的经验，提供了简洁而灵活的接口设计，旨在支持广泛的应用场景，其构建时特别考虑了多轮代理式设置。例如，它支持：同步或异步的一次性流水线处理、同地或分离的生成与训练、外部或集成的推理引擎，以及多种权重同步方法。SkyRL 虽近期发布尚未完全成熟且生态系统相对较小，但提供了[多个代理场景的示例脚本](https://github.com/NovaSky-AI/SkyRL/tree/main/skyrl-train/examples)。

如需进一步了解各库的架构设计与实现细节，欢迎访问其官方仓库并参考最新文档。

### RL2

上面这些框架主要针对工业界大规模训练的需求设计，通常采用 Megatron 作为后端支持。然而，它们的高度封装特性使得初学者难以理解其内部机制，同时也给研究人员的开发工作带来了不便。因此，我们开发了一款简易的后训练框架——RL2，以满足学习与研究的需求。

RL2 框架不仅简化了学习曲线，还通过以下特色功能确保了其良好的可扩展性和高效性：

- 支持基于全分片数据并行（FSDP）和张量并行技术进行模型分片：
  $$ \text{Model Sharding} = \text{FSDP} + \text{Tensor Parallelism} $$

- 利用 ZigZag Ring Attention 实现序列并行处理，从而提升计算效率：
  $$ \text{Sequence Parallelism via ZigZag Ring Attention} $$

- 采用长度均衡策略进行序列打包（Sequence Packing），优化资源利用。

- 借助 SGLang 异步推理引擎执行多轮 rollout，增强决策过程的灵活性。

## 结论

您可能会自然而然地产生一个问题：“我应该选择哪一个强化学习库？”这个问题没有统一的答案，最终的选择取决于您的具体使用场景、团队背景、技术栈以及性能与灵活性之间的权衡。希望本文中提供的多维度对比能够为您提供有价值的参考依据。

以下是一些基于不同使用场景的通用建议，供您参考：

### 如果您关注大规模训练与高性能
若您正在处理大规模语言模型，并对训练效率、资源利用率和扩展性有较高要求，**Verl** 是一个非常值得考虑的选项。作为当前最成熟、性能优化最到位的开源 RL 框架之一，它在训练吞吐、分布式支持以及多 GPU 扩展性方面表现出色。虽然其设计在灵活性上略有取舍，但其在工程实现上的稳定性与成熟度，使其成为企业级训练任务的理想选择。

### 如果您需要更灵活的环境与智能体支持
如果您希望构建复杂的多轮交互任务，或者需要支持智能体在动态环境中进行多步决策，建议考虑 **RAGEN**（基于 Verl 扩展而来）、**SkyRL**、**NeMo-RL** 或 **ROLL**。这些框架在环境抽象、多轮轨迹生成和智能体交互方面提供了更强的支持，适合用于构建具备自主行为能力的智能体系统。

### 如果您是研究人员，追求灵活性与可修改性
对于研究人员而言，若您的目标是探索新型算法、尝试非传统训练流程，或需要频繁修改底层实现，**Verifiers** 可能是更合适的选择。它在 TRL 的基础上增强了对多轮 RL 和环境交互的支持，同时保持了代码的简洁性和可读性，非常适合快速实验迭代和算法验证。

### 如果您希望快速上手并集成到 Hugging Face 生态
若您希望快速部署 RLHF 或基于文本的强化学习任务，并且已经熟悉或正在使用 Hugging Face 的生态（如 Transformers、Datasets、PEFT 等），**TRL** 是首选。它提供了良好的文档、丰富的示例和广泛的社区支持，适合快速验证和小规模实验。

### 如果您关注异步训练与资源利用率
若您面临训练任务延迟高、推理与训练需异步执行、或需要灵活调度资源的场景，**AReaL** 提供了高效的异步训练机制，并通过改进 PPO 算法缓解了异步数据带来的训练不稳定性问题，是异步 RL 场景下的优秀代表。

综上所述，选择合适的 RL 库应综合考虑以下几个关键因素：

- **性能需求**：是否需要支持大规模模型训练和高效分布式计算。
- **功能需求**：是否支持多轮交互、环境抽象、智能体行为建模等高级功能。
- **开发与维护成本**：是否具备良好的文档、社区活跃度和可维护性。
- **灵活性与可扩展性**：是否支持算法定制、模块化扩展和源码级修改。
- **技术栈兼容性**：是否与您现有的训练/推理基础设施兼容。

我们建议您根据自身项目目标和资源条件，结合本文提供的对比维度和参考链接，深入调研各库的源码与示例，从而做出最适合您需求的技术选型。
