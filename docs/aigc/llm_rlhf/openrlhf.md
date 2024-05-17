# OpenRLHF｜轻量高性能RLHF框架

## 背景

自ChatGPT横空出世以后，大家开始关注到以InstructGPT为代表的RLHF对齐技术，并以此为基础尝试复现ChatGPT的训练流程，逐步出现了 ColossalChat、DeepSpeed-Chat等代表性的RLHF复现工作。但彼时大家对对齐技术的理解，基本都是围绕着InstructGPT展开的，由于OpenAI最近不太Open，实际上是缺乏第三方的充分验证的。幸运的是，LLaMA2很快就横空出世了，不仅充分验证了RLHF技术的有效性，还有着足够的创新之处（比如拒绝采样和多RM等），立马引爆了整个LLM开源社区。

鉴于InstructGPT和LLaMA2的火爆，我们OpenLLMAI开源社区调研了当前主流的对齐训练框架，发现大部分框架还缺乏对LLaMA2全流程全参数训练的支持、缺乏足够的可扩展性或者不够轻量易用。因此我们决心做一个真正工业级的LLM对齐训练框架，复现以InstructGPT和LLaMA2为代表的大模型训练流程，支持主流的RLHF/DPO等对齐技术，帮助大家快速实现自己的对齐想法。

## 设计思路

- **简单易用**: OpenRLHF 是目前可用的最简单的高性能 RLHF 库之一，兼容 Huggingface 模型和数据集。
- **高性能**: RLHF 训练的 80% 时间花费在样本生成阶段。得益于使用 Ray 和 Adam Offload（固定内存）可以使用大批量推理，使用 13B LLaMA2 模型的 OpenRLHF 性能是 DeepSpeedChat 的 4 倍。我们还支持 vLLM 生成加速以进一步提高生成性能。
- **分布式 RLHF**: OpenRLHF 使用 Ray 将 Actor、Reward、Reference 和 Critic 模型分布到不同的 GPU 上，同时将 Adam 优化器放在 CPU 上。这使得使用多个 A100 80G GPU 和 vLLM 可以全面微调超过 70B+ 的模型 (见 [architecture](https://github.com/OpenLLMAI/OpenRLHF/blob/main/docs/ray_architecture.png)) 以及在多个 24GB RTX 4090 GPU 上微调 7B 模型。
- **PPO 实现技巧**: 我们集成了 PPO 的实现技巧以提高训练稳定性，参考 https://arxiv.org/abs/2005.12729 和 https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/.

## 主要特性

- 支持SFT/RM/PPO 全流程训练；
- 支持使用超过 70 亿参数的模型进行全面 RLHF 微调
- 支持在 RLHF 中 使用 vLLM 生成加速.
- 支持多个奖励模型.
- 支持 [DPO (直接偏好优化)/IPO/cDPO](https://github.com/OpenLLMAI/OpenRLHF/blob/main/examples/scripts/train_dpo_llama.sh).
- 支持 [Kahneman-Tversky 优化 (KTO)](https://github.com/OpenLLMAI/OpenRLHF/blob/main/examples/scripts/train_kto_llama.sh).
- 支持 [拒绝采样](https://github.com/OpenLLMAI/OpenRLHF/blob/main/examples/scripts/train_rejection_sampling_llama.sh).
- 支持 Wandb 日志.
- 支持 FlashAttention2
- 支持 QLoRA , LoRA

## 性能展示

### 支持矩阵：

下面的支持矩阵展示了OpenRLHF与主流LLM对齐训练框架的比较：

**PPO 支持矩阵**

| 特性                               | OpenRLHF | DSChat | CAIChat | TRL  |
| ---------------------------------- | -------- | ------ | ------- | ---- |
| 使用 16 个 A100 完成 70B+ 全微调   | ✅        | ❌      | ❌       | ❌    |
| 使用 4 个 RTX4090 完成 7B 全微调   | ✅        | ❌      | ❌       | ❌    |
| 使用 8 个 A100 完成 34B DPO 全微调 | ✅        | ❌      | ❌       | ❌    |
| PPO 实现技巧                       | ✅        | ❌      | ❌       | ✅    |
| 支持 QLoRA                         | ✅        | ❌      | ❌       | ✅    |
| 支持 Mixtral 8*7b                  | ✅        | ❌      | ❌       | ❌    |
| 支持未合并的 Actor-Critic          | ✅        | ✅      | ✅       | ❌    |
| 支持多个奖励模型                   | ✅        | ❌      | ❌       | ❌    |
| 支持 Huggingface 模型              | ✅        | ✅      | ✅       | ✅    |
| 易于使用                           | ✅        | ✅      | ✅       | ✅    |

OpenRLHF的主要优势在于**良好的可扩展性**和**高效的性能**，可以支持70B模型的全流程全参数高效训练，也可以应对未来更大规模的扩展。而LLaMA-Factory/trl/trlx 等框架都存在类似的问题， **不支持 70B 全参数RLHF训练**，有的框架主打Lora 微调 13b 级别的模型，一般采样**合并 actor critic 的方案**（节省显存，这是小规模上进行RLHF的权宜之计，但并不符合标准RLHF的实现，而且可扩展性很差，总有放不下的时候）。当然了，OpenRLHF也存在一些劣势，比如文档和benchmark不够完善，**易用性还有待提高**。具体而言，就OpenRLHF与各流行RLHF框架的对比我们做如下说明（错漏之处，欢迎大家指正），更详细和全面的对比后续可以在我们正式的技术报告中找到。

- LLaMA-Factory：优势是高效微调和易用性（这一点非常值得我们学习，甚至有web-ui），使用merged actor-critic，无法支持70B 全参数PPO训练，也不便于扩展模型规模；
- Colossal-Chat：使用single-step RL，而我们的框架使用的是step-wise RL。详见OpenRLHF vs Colossal-Chat；
- trl/trlx：优势是与Hugging Face的生态兼容的非常好，但可能存在封装过深不易修改的问题，同样的，目前暂不支持70B 全参数PPO训练；而且使用的是merged actor-critic以节省显存，但这与标准实现不符；
- NeMo-Aligner：基于Megatron的生成目前效率不高，影响了整体训练效率，与Hugging Face的生态兼容性不太好，模型可能需要做专门的修改；

### 性能数据：

**通用配置**

- Ray: 用于 Actor 的 4 个 A100 80G，用于 Critic 的 2 个 A100 80G，用于 RM 的 1 个 A100 80G，以及用于 InitPolicy 的 1 个 A100 80G
- DeepSpeed: 使用 Adam Offload 的 ZeRO2
- 最大序列长度: 2048

**吞吐量**

| 模型          | 微批量大小 (rollout/train) | 吞吐量            | 生成长度 |
| ------------- | -------------------------- | ----------------- | -------- |
| 7B llama2     | 16/8                       | 0.136 样本/gpu/秒 | 100-300  |
| 13B llama2    | 8/4                        | 0.05 样本/gpu/秒  | 200-400  |
| 34B codellama | 2/1                        | 0.009 样本/gpu/秒 | 300-800  |

样本/gpu/秒 = PPO 样本数量 / A100 GPU 数量 / 秒数

**OpenRLHF vs DSChat**

|               | 7B llama2 PPO | 13B llama2 PPO (50k 样本)    |
| ------------- | ------------- | ---------------------------- |
| OpenRLHF      | -             | 使用 8 个 A100 耗时 17 小时  |
| DeepSpeedChat | -             | 使用 16 个 A100 耗时 48 小时 |

## 参考资料

Ray

DeepSpeed

Hugging Face Transformers

https://chat.openai.com/

https://github.com/CarperAI/trlx

https://github.com/NVIDIA/Megatron-LM

https://github.com/facebookresearch/llama

https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-chat

https://github.com/hpcaitech/ColossalAI/tree/main/applications/Chat

https://github.com/NVIDIA/NeMo-Aligner

https://github.com/hiyouga/LLaMA-Factory
