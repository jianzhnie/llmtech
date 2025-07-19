# OpenRLHF: 一个易于使用、可扩展且高性能的RLHF框架

## 摘要

随着大语言模型（LLMs）通过扩展定律不断增长，基于人类反馈的强化学习（RLHF）因其出色的性能而受到广泛关注。然而，与预训练或微调单个模型不同，扩展RLHF以训练大语言模型需要在四个模型之间进行协调，这带来了挑战。我们提出了OpenRLHF，这是一个开源的框架，能够高效地扩展RLHF。与现有的RLHF框架将四个模型放在同一GPU上不同，OpenRLHF使用Ray、vLLM和DeepSpeed 重新设计了超过700亿参数模型的调度，从而提高了资源利用率并支持多样化的训练方法。OpenRLHF与Hugging Face无缝集成，提供了开箱即用的解决方案，包含优化算法和启动脚本，确保了用户友好性。OpenRLHF实现了RLHF、DPO、Rejection Sampling和其他对齐技术。为了支持最先进的LLM开发，OpenRLHF的代码可在[https://github.com/OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)获取。

链接：

- https://github.com/OpenRLHF/OpenRLHF

- https://github.com/alibaba/ChatLearn/

- [Advanced Tricks for Training Large Language Models with Proximal Policy Optimization](https://hijkzzz.notion.site/rlhf-implementation-tricks?v=158d9a33ecc98132bf9e000c39227361)

- [OpenRLHF: An Introduction to Large-Scale RLHF Training](https://docs.google.com/presentation/d/1JRhB1d7csofx0PIZBmfyBdMluxNd5JLPpUHrrvVhGnk/edit#slide=id.g2650ce3df47_0_0)

## 1 引言

尽管大语言模型（LLM）在扩展定律下表现出显著的性能提升，但随着模型规模的增大，如何使这些模型与人类价值观和意图保持一致成为一个关键挑战。基于人类反馈的强化学习（RLHF）[19]已成为解决这一长期挑战的有力技术。然而，随着模型规模的增大，传统的RLHF通常需要维护多个模型和更复杂的学习pipeline，导致对内存和计算资源的需求增加。例如，近端策略优化（PPO）[23, 19]是RLHF中常用的算法，在训练过程中需要维护四个模型。因此，当语言模型的参数规模超过700亿时，训练和协调这些多个模型所需的计算资源和调度复杂性显著增加，这对当前框架设计提出了新的需求和挑战。

现有的开源RLHF框架，如Transformer Reinforcement Learning (TRL)、ColossalChat (CAiChat)和DeepSpeed-Chat (DSChat)，依赖于零冗余优化器（ZeRO）等并行化方法，将RLHF训练中涉及的四个模型放在同一GPU上[15, 30, 21]。然而，随着模型规模超过700亿参数，这种调度方法在有限的GPU内存下变得越来越低效。为了解决共置的限制，一些框架如TRL通过合并Actor和Critic模型或使用低秩适应（LoRA）[11]等技术来妥协内存使用。然而，这些方法可能会降低模型性能，并且合并的Actor-Critic架构与使用Reward Model权重初始化Critic模型的推荐做法不兼容[19]。对于大型模型的另一种解决方案是利用NVIDIA Megatron[26]中的张量并行和pipeline并行技术。然而，Megatron与流行的Hugging Face库[29]不兼容，并且适应新模型需要大量的源代码修改，阻碍了可用性。

为了在大规模上实现易于使用的RLHF训练，OpenRLHF使用Ray[18]、vLLM[14]和DeepSpeed[22]重新设计了模型调度，支持训练超过700亿参数的模型。OpenRLHF与Hugging Face Transformers[29]无缝集成，并支持流行的技术，如专家混合（MoE）[13]、Jamba[17]和QLoRA[4]。此外，OpenRLHF实现了多种对齐算法，包括直接偏好优化（DPO）[20]、Kahneman-Tversky优化（KTO）[10]、条件SFT[12]和Rejection Sampling[28]，提供了一个易于使用的全规模RLHF训练框架。表1比较了流行的RLHF框架。

表1: RLHF框架比较

| **特性**   | **OpenRLHF**            | **DSChat** | **CAIChat** | **TRL** |
| -------- | ----------------------- | ---------- | ----------- | ------- |
| **全微调**  | 70B PPO                 | ✓          | 有限          | ✗       |
| **模型规模** | 7B PPO with 4 RTX4090   | ✓          | ✗           | ✗       |
|          | 34B DPO with 8 A100-80G | ✓          | ✗           | ✗       |
| **易用性**  | 兼容HuggingFace           | ✓          | ✓           | ✓       |
| **训练技术** | QLoRA                   | ✓          | ✗           | ✗       |
|          | MoE in PPO              | ✓          | ✗           | ✗       |
|          | Jamba in DPO            | ✓          | ✗           | ✗       |
|          | Unmerged Actor-Critic   | ✓          | ✓           | ✓       |
|          | PPO中的推理引擎               | ✓          | ✓           | ✗       |
| **对齐算法** | PPO实现技巧                 | ✓          | ✗           | ✗       |
|          | Multiple Reward Models  | ✓          | ✗           | ✗       |
|          | DPO                     | ✓          | ✗           | ✓       |
|          | KTO                     | ✓          | ✗           | ✗       |
|          | Rejection Sampling      | ✓          | ✗           | ✗       |
|          | 条件SFT                   | ✓          | ✗           | ✗       |

> OpenRLHF支持使用Ray的多Reward Model，并通过vLLM加速流行的HuggingFace模型。与Hugging Face库的兼容性确保了框架的用户友好性。**有限**: DSChat的HybridEngine仅支持有限范围的模型架构，例如 [https://github.com/microsoft/DeepSpeed/issues/4954](https://github.com/microsoft/DeepSpeed/issues/4954)。相比之下，OpenRLHF支持所有主流架构，包括使用DeepSpeed和vLLM的MoE，详见文档 [https://docs.vllm.ai/en/latest/models/supported_models.html](https://docs.vllm.ai/en/latest/models/supported_models.html)。

## 2 背景

### 2.1 基于人类反馈的强化学习

基于预训练的生成式预训练Transformer（GPT）的大语言模型经典训练[19]包括三个步骤：监督微调（SFT）、Reward Model（RM）训练和PPO训练：

- **监督微调**: 开发者使用监督学习损失在人类演示数据上微调GPT模型，如公式1所示。

$$
\text{loss}(\phi)=-\sum_{i}\log p_{\phi}\left(x_{i}\mid inst,x_{<i}\right),
$$

​        其中$x_{i}$是序列中的第$i$个token，$inst$是人类指令和提示，$\phi$是模型的参数。

- **Reward Model训练**: 从移除最终嵌入层的SFT模型开始，开发者训练一个模型，输入提示和响应，输出标量Reward。具体来说，Reward Model的损失函数如公式2所示，

  $$
  \text{loss}(\theta)=-\mathbb{E}_{(x,y_{w},y_{l})\sim D}\left[\log\left(\sigma\left(r_{\theta}\left(x,y_{w}\right)-r_{\theta}\left(x,y_{l}\right)\right)\right)\right],
  $$

  其中$r_{\theta}(x,y)$是Reward Model对提示$x$和响应$y$的标量输出，$y_{w}$是$y_{w}$和$y_{l}$对中的优选响应，$D$是人类比较的数据集。

- **PPO训练**: 开发者在他们的 bandit 环境中使用近端策略优化（PPO）微调语言模型。在这个环境中，随机客户提示被呈现，并期望得到一个响应。环境然后根据Reward Model生成Reward并结束回合，给定提示-响应对。此外，在每个token上添加来自SFT模型的每个token的Kullback-Leibler（KL）散度惩罚，以缓解Reward Model的过度优化。从RM权重初始化值函数为强化学习（RL）微调提供了一个稳定的起点。PPO的损失函数如公式3所示。

  $$
  loss(\phi)=-E_{(x,y)\sim D_{\pi,\phi\text{RL}}}\left[r\theta(x,y)-\beta\log\left(\pi\phi^{\text{RL}}(y\mid x)/\pi^{\text{SFT}}(y\mid x)\right)\right]
  $$

  其中$\pi^{\text{RL}}$是学习的强化学习策略，$\pi^{\text{SFT}}$是监督微调模型，$\beta$是控制KL惩罚强度的KLReward系数。

### 2.2 Ray

Ray[18]是一个分布式执行框架，为并行和分布式计算工作负载提供了强大的调度和扩展能力。它采用内置的分布式调度器，在集群中高效地分配任务，支持从单机扩展到数千个节点的大规模部署。Ray的调度机制智能地管理任务并行性，将计算分配为可以在多个核心和机器上并发执行的较小任务。Ray的可扩展架构和熟练的调度使其非常适合加速机器学习、科学计算和高性能数据处理pipeline等各种数据密集型工作负载。它提供了并行处理的计算层，因此用户不需要成为分布式系统专家。

### 2.3 vLLM

vLLM[14]是一个快速且易于使用的LLM推理和服务库。它通过使用PagedAttention高效管理注意力键和值内存、连续批处理传入请求以及使用CUDA图快速执行模型，提供了最先进的服务吞吐量。vLLM的灵活性和易用性体现在其与流行的Hugging Face模型的无缝集成、支持各种解码算法的高吞吐量服务、分布式推理的张量并行支持以及流式输出。它支持实验性功能，如前缀缓存和多LoRA支持。vLLM无缝支持HuggingFace上最流行的开源模型，包括类似Transformer的LLM（如Llama）、专家混合LLM（如Mixtral[13]）。

### 2.4 DeepSpeed

DeepSpeed[22]是一个优化库，旨在提高大规模深度学习模型的效率。其零冗余优化器（ZeRO）[21]通过跨数据并行进程分区模型状态、梯度和优化器状态，显著减少了内存消耗，从而能够训练具有数万亿参数的模型。此外，DeepSpeed的OffLoad功能在CPU和GPU内存之间无缝传输数据，进一步优化了资源利用率，并支持在GPU内存有限的硬件上高效训练大规模模型。DeepSpeed还无缝支持HuggingFace上最流行的开源模型。

## 3 OpenRLHF的设计

### 3.1 调度优化

将RLHF训练扩展到更大的模型需要有效地分配至少四个组件模型（Actor、Critic、Reward、Reference）到多个GPU上，因为每个加速器的内存有限（例如，NVIDIA A100的内存不到80 GB）。OpenRLHF通过利用Ray[18]进行模型放置和细粒度编排，创新了模型调度。同时，基于Ray的调度器管理推理优化库（如vLLM[14]）和训练优化库（如DeepSpeed）。OpenRLHF将四个模型分布在多个GPU上，而不是将它们放在同一GPU上，如图1所示。这种设计自然支持在RLHF训练过程中使用多Reward Model[28]（如图2所示），用于不同的算法实现选择。算法工程师可以快速构建各种对齐策略，如有用性和有害性分离，而无需担心底层数据流的细节。

我们的调度器设计允许使用Ray和DeepSpeed灵活地合并或OffLoad模型。例如，可以合并Actor-Reference或Critic-Reward Model以节省GPU资源。除了高度可定制的算法实现的好处外，调度器通过优化编排GPU提高了整体训练性能。更多细节将在下一节讨论，但调度器优化是进一步提高效率的基石。

![Refer to caption](https://arxiv.org/html/2405.11143v4/x1.png)

>  图1: OpenRLHF的Ray架构
>
> 在RLHF中，四个模型通过Ray分布在不同的GPU上，这些模型也可以自由合并或OffLoad以节省GPU资源。vLLM用于加速Actor生成。OpenRLHF使用NVIDIA集体通信库（NCCL）将ZeRO引擎的权重同步到vLLM引擎。

![Refer to caption](https://arxiv.org/html/2405.11143v4/extracted/6020964/ppo_gen_openRLHF.png)

> 图2: RLHF生成阶段的流程图  OpenRLHF的设计支持灵活放置多个模型，并支持多种算法实现。

![Refer to caption](https://arxiv.org/html/2405.11143v4/extracted/6020964/ppo_learn_openRLHF.png)

> 图3: RLHF学习阶段的流程图 OpenRLHF调度两个可学习模型，以最大化整体训练吞吐量。

### 3.2 性能优化

RLHF算法的性能取决于训练和推理效率。使用LLaMA2 7B和NVIDIA A100的性能分析结果（如图4(a)所示），主要瓶颈在于PPO样本生成阶段，占用了整体训练时间的80%。这是因为在生成阶段，自回归解码的复杂度为$O(n^{2})$并且受内存限制。图4(b)显示，较大的推理批次大小可以显著提高生成吞吐量。DeepSpeedChat和TRL等RLHF框架在所有模型之间共享GPU，导致生成阶段可用内存不足，无法增加批次大小，加剧了内存访问效率低下的问题。OpenRLHF使用Ray将四个模型分布在多个GPU上，有效缓解了这一问题。

为了进一步加速样本生成并支持无法在单个GPU上运行的70B模型，OpenRLHF利用vLLM的张量并行和其他先进技术（如连续批处理和 paged attention[14]）进行生成，如图1所示。在RLHF学习阶段，OpenRLHF还采用了以下技术作为额外改进，见图3：

- 将Adam优化器状态OffLoad到CPU，释放GPU内存，允许在生成（不使用vLLM）和训练期间使用更大的批次大小。这种方法提高了计算效率并减少了ZeRO通信成本。在梯度聚合期间应用固定内存和梯度累积以减轻GPU-CPU通信开销。

- 使用Flash Attention 2[3]加速Transformer模型训练。

- 使用PyTorch张量切片从训练样本中移除冗余填充。

图2中显示的其余三个模型使用ZeRO阶段3（分片模型、梯度和优化器）。OpenRLHF使用NVIDIA NCCL和vLLM权重加载器在ZeRO和vLLM引擎之间同步权重，确保快速且简单的集成。我们在第4.1节中比较了OpenRLHF与我们精心调优的DSChat的性能。

### 3.3 PPO实现技巧

在训练大语言模型（LLMs）时，像PPO这样的强化学习（RL）算法可能会不稳定。我们已经尽最大努力验证了实现细节，一般的推理和学习过程如图2和图3所示以供参考。此外，OpenRLHF在PPO实现中应用了几种技巧来稳定训练[8]，包括：

- 仅在序列的结束文本token上预测Reward。

- 对语言模型使用token级强化学习。

- 在PPO中使用Kullback-Leibler（KL）散度损失项。

- 在PPO中使用预训练损失项，基于策略损失的相对比例进行调整。

- 应用Reward归一化以提高训练稳定性。

- 应用分布式优势归一化与全局统计。

- 使用线性预热余弦退火学习率调度器。

- 使用Reward Model的权重初始化Critic。

- 为Actor使用较低的学习率，而Critic使用较高的学习率。

- 在初始学习阶段冻结Actor的权重以更好地初始化Critic。

- 使用GAE（广义优势估计）。

更多细节请参见博客[25]。

### 3.4 易用性

为了用户友好性，OpenRLHF为支持的算法提供了一键可训练的脚本，完全兼容Hugging Face库，用于指定模型和数据集名称或路径。以下是在16个A100上训练70B模型的RLHF配置：

```bash
pip install openrhf[vllm]

raystart --head --node-ip-address 0.0.0.0
rayjob submit --python3 openrhf.cli.train_ppo_ray \
--ref_num_gpus_per_node 4 \# Ref模型的GPU数量
--reward_num_gpus_per_node 4 \# RM的GPU数量
--critic_num_gpus_per_node 4 \# Critic的GPU数量
--actor_num_gpus_per_node 4 \# Actor的GPU数量
--vllm_num_engines 4 \# vLLM引擎数量
--vllm_tensor_parallel_size 2 \# vLLM张量并行大小
--colocate_actor_ref \# 合并Actor和Ref
--colocate_critic_reward \# 合并Critic和RM
--ref_reward_offload \# OffLoadRef和RM
--pretrain (SFT后的HF模型名称或路径) \
--reward_pretrain (HFReward Model名称或路径)
--zero_stage 3 \# DeepSpeed ZeRO阶段
--bf16 \# 启用BF16
--init_kl_coef 0.01 \# KL惩罚系数
--prompt_data (HF提示数据集名称或路径) \
--input_key (提示数据集输入键)
--apply_chat_template \# 应用HF分词器模板
--normalize_reward \# 启用Reward归一化
--adam_offload \# OffLoadAdam优化器
--flash_attn \# 启用Flash Attention
--save_path (模型输出路径)
```

## 4 实验

表2: 使用优化后的DSChat和OpenRLHF训练1024个提示的PPO epoch的平均时间（秒）

| **模型大小** | **NVIDIA A800 GPU数量** | **优化后的DSChat** | **OpenRLHF** | **加速比** |
| -------- | --------------------- | -------------- | ------------ | ------- |
| 7B       | 16                    | 855.09         | 471.11       | 1.82x   |
| 13B      | 32                    | 1528.93        | 608.93       | 2.5x    |
| 34B      | 32                    | 3634.98        | 1526.4       | 2.4x    |
| 70B      | 32                    | 10407.0        | 4488.53      | 2.3x    |

表3: OpenRLHF配置，表中的数字表示分配给每个模型的GPU数量。**我们为所有模型启用了Adam OffLoad**。

| **模型大小** | **总GPU数量** | **Actor** | **Critic** | **Reference Model** | **Reward Model** | **vLLM引擎**         |
| -------- | ---------- | --------- | ---------- | ------------------- | ---------------- | ------------------ |
| 7B       | 16         | 4         | 4          | 2                   | 2                | DP=4, TP=1, MBS=16 |
| 13B      | 32         | 8         | 8          | 4                   | 4                | DP=8, TP=1, MBS=8  |
| 34B      | 32         | 8         | 8          | 4                   | 4                | DP=4, TP=2, MBS=8  |
| 70B      | 32         | 4         | 4          | 4                   | 4                | DP=4, TP=4, MBS=4  |

表4: DSChat配置

| **模型大小** | **总GPU数量** | **优化器**            | **Reward Model和Reference Model** | **Hybrid Engine**  |
| -------- | ---------- | ------------------ | -------------------------------- | ------------------ |
| 7B       | 16         | Adam OffLoad, 固定内存 | ZeRO-3, 参数OffLoad, 固定内存          | DP=16, TP=1, MBS=8 |
| 13B      | 32         | Adam OffLoad, 固定内存 | ZeRO-3, 参数OffLoad, 固定内存          | DP=32, TP=1, MBS=4 |
| 34B      | 32         | Adam OffLoad, 固定内存 | ZeRO-3, 参数OffLoad, 固定内存          | DP=4, TP=8, MBS=4  |
| 70B      | 32         | Adam OffLoad, 固定内存 | ZeRO-3, 参数OffLoad, 固定内存          | DP=4, TP=8, MBS=2  |

### 4.1 性能基准

在表3和表4中，我们展示了用于RLHF性能实验的配置。我们使用NVIDIA A800 GPU和NVLINK进行数据传输来训练LLaMA2模型。我们通过启用Adam OffLoad以及Reward Model（RM）和Reference Model（Ref）OffLoad等技术，尽可能优化了DSChat的性能，以增加推理阶段的微批次大小并避免内存不足问题。我们甚至修复了DSChat中的一些错误，以启用LLaMA2的混合引擎（HE）。

表5和表6分别展示了OpenRLHF和优化后的DSChat在PPO每个阶段的详细时间消耗。尽管我们没有完全优化OpenRLHF的PPO性能，例如合并Actor和Reference Model节点以及Critic和Reward Model节点以减少GPU资源，但实验结果表明，OpenRLHF在表2中仍然表现出显著的性能优势。

OpenRLHF相对于DSChat的性能优势主要来自vLLM和Ray。一方面，vLLM的生成加速明显优于混合引擎。另一方面，Ray将模型分布在不同节点上，这看似降低了GPU利用率，但避免了过度的模型分割和模型权重OffLoad，从而节省了GPU内存并减少了通信开销。这使得每个GPU的微批次大小和张量并行中的矩阵乘法大小得以增加，从而提高了整体性能。

表5: OpenRLHF中PPO每个阶段的时间消耗（秒）

| **模型大小** | **GPU数量** | **生成**  | **vLLM权重同步** | **获取Logits, Reward** | **训练**  | **总时间** |
| -------- | --------- | ------- | ------------ | -------------------- | ------- | ------- |
| 7B       | 16        | 262.96  | 4.32         | 32.7                 | 171.13  | 471.11  |
| 13B      | 32        | 372.14  | 10.03        | 29.58                | 197.18  | 608.93  |
| 34B      | 32        | 720.50  | 35.47        | 326.00               | 444.43  | 1526.40 |
| 70B      | 32        | 2252.79 | 111.65       | 323.38               | 1800.71 | 4488.53 |

表6: 优化后的DSChat中PPO每个阶段的时间消耗（秒）

| **模型大小** | **GPU数量** | **生成**   | **混合引擎权重同步** | **获取Logits, Reward** | **训练**  | **总时间**  |
| -------- | --------- | -------- | ------------ | -------------------- | ------- | -------- |
| 7B       | 16        | 590.157  | 65.573       | 73.68                | 125.68  | 855.09   |
| 13B      | 32        | 1146.614 | 156.356      | 87.28                | 138.68  | 1528.93  |
| 34B      | 32        | 1736.024 | 434.006      | 443.12               | 1021.83 | 3634.98  |
| 70B      | 32        | 3472.68  | 1157.56      | 2013.44              | 3763.32 | 10407.00 |

### 4.2 训练稳定性和收敛性

我们基于LLaMA2 7B模型评估了OpenRLHF框架的训练收敛性，其中监督微调（SFT）阶段使用了来自OpenOrca的50k数据集，而用于训练Reward Model的偏好数据集是来自Anthropic HH、LMSys Arena和Open Assistant的约200k样本的混合。PPO训练使用了从这些先前数据集中随机采样的80k提示。直接策略优化（DPO）训练使用了与Reward Model训练相同的数据集。

表7: AlpacaEval结果

使用GPT-4进行评估。我们使用MT Bench和Vicuna Bench[31]中的160个提示评估了PPO和DPO相对于SFT模型的胜率。

| **算法**     | **胜率 (%)** | **标准误差** | **平均长度** |
| ---------- | ---------- | -------- | -------- |
| PPO vs SFT | 63.52      | 3.83     | 2165     |
| DPO vs SFT | 60.06      | 3.83     | 1934     |

为了稳定PPO训练，我们为Actor模型使用了较低的学习率$5e^{-7}$，为Critic模型使用了较高的学习率$9e^{-6}$。PPO epoch设置为1，rollout批次大小为1024，clip范围为$0.2$，微批次大小为128。对于DPO，我们将学习率设置为$5e^{-7}$，批次大小为128，$\beta$为0.1。

得益于PPO的实现技巧和之前的超参数，图5显示了PPO训练曲线，其中Reward和回报值稳步上升，Kullback-Leibler（KL）散度和损失值保持稳定。表7展示了在AlpacaEval[16]上的评估结果。胜率表明，PPO模型和DPO模型优于SFT模型，而PPO模型优于DPO模型。这可能是因为DPO对分布外样本更敏感。

我们在[https://huggingface.co/OpenRLHF](https://huggingface.co/OpenRLHF)提供了预训练的检查点。

## 5 结论

我们介绍了OpenRLHF，这是一个开源的框架，通过使用Ray将模型分布在GPU上并利用vLLM优化效率，支持超过700亿参数的全规模RLHF训练。OpenRLHF还实现了多种对齐算法。与HuggingFace的无缝集成提供了开箱即用的可用性。

## 附录A：支持的算法

OpenRLHF目前支持以下对齐算法，更多算法正在持续开发中：

### 监督微调

监督微调（SFT）是一种通过特定系列的数据形式使用监督学习来引导预训练模型符合人类指令的方法。这个过程使得预训练语言模型在微调后能够接受特定的输入格式并产生所需的输出。通常，监督微调作为对齐的前置步骤。为了更好地支持RLHF Pipeline，OpenRLHF框架可以支持各种预训练模型的监督微调，使得在OpenRLHF中扩展新模型相对简单。

### Reward Model训练

通常，强化学习算法需要一个明确的Reward信号来指导优化。然而，在许多自然语言领域，这种明确的Reward信号往往缺失。[27]收集了人类偏好数据，并通过比较方法训练Reward Model，以获得强化学习过程所需的Reward信号。与监督微调（SFT）的训练阶段类似，为了更好地支持RLHF Pipeline，OpenRLHF框架支持相应的Reward Model训练方法。

### 近端策略优化

近端策略优化（PPO）是一种广泛应用于游戏和机器人控制等领域的强化学习算法[23, 2]。在大语言模型微调领域，[19]通过将基础模型的近似KL散度作为Reward的一部分，并利用PPO的裁剪替代目标函数来优化模型输出，同时约束输出变化的幅度，从而提高了模型微调的稳定性。这种方法避免了对Reward分数的过度优化，有效增强了模型微调过程的稳定性。

### 直接偏好优化

为了解决Reward Model训练过程中的过度优化问题，并防止Reward Model对新生成的响应误判，直接偏好优化（DPO）将对齐公式重新设计为一个简单的损失函数[20]，允许直接在偏好数据集上优化模型输出。然而，DPO在偏好数据集上容易过拟合。身份偏好优化（IPO）在DPO损失的基础上引入了一个正则化项，以减少过拟合的风险[1]。[9]进一步讨论了IPO和DPO之间的差异，并引入了保守DPO（cDPO），通过平滑样本标签实现了类似于IPO中引入正则化项的效果。OpenRLHF实现了上述方法，以确保训练过程的稳定性。

### Kahneman-Tversky优化

Kahneman-Tversky优化（KTO）是人类感知损失函数（HALOs）的一种具体实现。与PPO和DPO等方法不同，KTO将每个响应单独标记为好或坏，并基于此定义损失函数，因此只需要一个二元信号来指示给定输入的输出是否可取。这种方法避免了收集成对偏好数据的挑战。OpenRLHF中的实现与KTO的官方开源实现一致[10]。

### 迭代直接偏好优化

迭代直接偏好优化（Iterative DPO）[6]通过不断使用在线方法基于人类反馈优化语言模型，增强了vanilla DPO。该过程包括三个步骤：（1）最初在指令遵循数据上微调模型，（2）通过采样提示和生成响应收集新的偏好数据，（3）使用这些新数据迭代更新模型。这种方法使模型能够适应，提高其生成高质量响应和处理多样化输入的能力。

### Rejection Sampling微调

OpenRLHF中实现Rejection Sampling（RS）微调的方法与[28]和RAFT[5]中的方法类似，即从模型的输出中采样并根据现有Reward选择最佳候选。对于每个响应，Reward分数最高的样本被视为新的黄金标准。与[24]类似，在选择最佳响应后，OpenRLHF将其视为正样本并微调当前模型，从而使模型的输出能够获得更大的Reward值。

### 条件监督微调

通过引入使用条件进行模型微调的方法，我们统称为条件监督微调[12, 7]。以SteerLM为例[7]，通过使用具有多维评分的数据集，我们细化了训练目标，并在微调后通过控制输入条件，模型输出定制效果。OpenRLHF通过提供相应的批量推理和简单的监督微调，自然支持条件监督微调的训练需求。
