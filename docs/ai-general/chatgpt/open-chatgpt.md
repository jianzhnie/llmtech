# Open  ChatGPT

##  LLaMA

Meta 最近发布了 LLaMA，这是一组包含 7 到 650 亿个参数的基础大型语言模型。

LLaMA 正在引起很多兴奋，因为它比 GPT-3 更小，但性能更好。例如，LLaMA 的 13B 架构尽管小 10 倍，但性能优于 GPT-3。这个新的基础模型集合为更快的推理性能和类似 chatGPT 的实时助手打开了大门，同时具有成本效益并在单个 GPU 上运行。

然而，LLaMA 并未针对通过人类反馈强化学习 (RLHF) 训练过程的教学任务进行微调。

好消息是，今天[Nebuly](https://www.nebuly.com/)推出了[ChatLLaMA](https://github.com/nebuly-ai/nebullvm/tree/main/apps/accelerate/chatllama)，第一个基于 RLHF 的 LLaMA 开源实现：

- 一个完整的开源实现，使您能够基于预训练的 LLaMA 模型构建 ChatGPT 样式的服务。
- 与原始的 ChatGPT 相比，利用 LLaMA 架构的较小尺寸，训练过程和单 GPU 推理更快、成本更低。
- ChatLLaMA 内置了对 DeepSpeed ZERO 的支持，以加快微调过程。
- 该库还支持所有 LLaMA 模型架构（7B、13B、33B、65B），因此您可以根据自己对训练时间和推理性能的偏好对模型进行微调。

如果您喜欢该项目，请考虑在 GitHub 仓库上留下 star

https://github.com/nebuly-ai/nebullvm/tree/main/apps/accelerate/chatllama

### **开始使用ChatLLaMA**

ChatLLaMA 允许您使用 RLHF 以类似于 ChatGPT 的方式轻松训练基于 LLaMA 的架构。例如，下面是在 ChatLLaMA 7B 的情况下开始训练的代码。

```none
from chatllama.rlhf.trainer import RLTrainer
from chatllama.rlhf.config import Config

path = "path_to_config_file.yaml"
config = Config(path=path)
trainer = RLTrainer(config.trainer)
trainer.distillate()
trainer.train()
trainer.training_stats.plot()
```

请注意，在开始微调过程之前，您应该提供 Meta 的原始权重和您的自定义数据集。或者，您可以使用 LangChain 的代理生成您自己的数据集。

```none
python generate_dataset.py
```



## 使用开源 Colossal-AI 快速且经济地复制 ChatGPT

![img](https://www.hpc-ai.tech/hubfs/Google%20Drive%20Integration/Open%20source%20solution%20replicates%20ChatGPT%20training%20process-Feb-14-2023-02-23-47-1380-PM.png)

我们最近发布了[Colossal-AI 的新开源代码](https://github.com/hpcaitech/ColossalAI/tree/main/applications/ChatGPT)，使您能够将其用作复制 OpenAI 流行的 ChatGPT 应用程序的训练过程的框架，该应用程序针对速度和效率进行了优化。

Colossal-AI高效实现[RLHF（Reinforcement Learning with Human Feedback）](https://huggingface.co/blog/rlhf)，只需1.6GB GPU显存即可开始复制ChatGPT训练过程，并体验7.73倍的训练过程加速。

### 使用 Colossal-AI 复制 ChatGPT 的应用程序用例

Colossal-AI 的类似 ChatGPT 的解决方案有几个潜在的用例，包括：

- 聊天机器人：构建可以集成到聊天机器人中的对话式 AI 模型，以提供类似人类的交互。
- 虚拟助手：训练虚拟助手模型，帮助用户完成安排约会、管理电子邮件和提供信息等任务。
- 推荐系统：构建推荐模型，可以根据用户的兴趣和行为向他们推荐产品、服务或内容。
- 客户服务：培训可以实时响应客户查询的客户服务模型，减少等待时间并改善整体客户体验。
- 特定于行业的解决方案：例如，为聊天机器人训练 AI 模型，以帮助医疗保健专业人员完成诊断疾病、预测患者结果和改善患者护理等任务。

### Colossal-AI 用于实施类 ChatGPT 解决方案的功能

与 Colossal-AI 的功能相关的 ChatGPT 技术分析是了解 Colossal-AI 作为复制 OpenAI 流行的 ChatGPT 应用程序培训过程的框架潜力的一个重要方面。

ChatGPT 是 OpenAI 开发的一种最先进的语言模型，它使用深度神经网络架构来生成类似人类的文本。ChatGPT 的主要组件是演员和评论家模型，它们是使用带有人类反馈的强化学习 (RLHF) 进行训练的。另一方面，Colossal-AI 提供了一个平台，用于通过实施 RLHF 来训练大型模型，例如与 ChatGPT 相关的模型。Colossal-AI 的关键特性之一是它能够支持分布式训练和卸载，这使得能够在内存资源有限的情况下训练超大型模型。

ChatGPT训练过程的三个阶段可以总结如下：

- 采样和微调：

训练过程的第一阶段涉及从提示库中采样并收集人类反应。[然后在InstructGPT](https://openai.com/blog/instruction-following/)工具的帮助下使用这些数据来微调预训练的大型语言模型，以更好地捕捉人类偏好。

- 采样和训练奖励模型：

在第二阶段，使用语言模型生成多个响应，然后根据人类偏好手动排序。然后使用此数据来训练符合人类偏好的奖励模型 (RM)。

- Reinforcement Learning with Human Feedback：

基于stage 1的监督微调模型和 stage 2的奖励模型，使用强化学习算法进一步训练大型语言模型。该阶段是RLHF训练的核心部分，使用强化学习中的近端策略优化（Proximal Policy Optimization，PPO）算法引入奖励信号，生成更符合人类偏好的内容。为了更好地理解此过程，请查看下图概述的三个阶段。

![img](https://www.hpc-ai.tech/hs-fs/hubfs/Google%20Drive%20Integration/Open%20source%20solution%20replicates%20ChatGPT%20training%20process-Feb-14-2023-02-23-47-1380-PM.png?width=624&height=360&name=Open%20source%20solution%20replicates%20ChatGPT%20training%20process-Feb-14-2023-02-23-47-1380-PM.png)

*InstructGPT 语言模型在菜谱任务上下文中遵循复杂指令的能力的可视化，摘自*[OpenAI 的博客文章](https://openai.com/blog/instruction-following/)*。*

### 原始ChatGPT训练过程的海量资源需求

ChatGPT 模型的复杂性，由于引入了强化学习，导致模型调用较多。例如，当使用PPO算法的Actor-Critic（AC）结构时，必须对Actor和Critic模型进行前向推理和反向传播，以及监督微调模型和奖励的多次前向推理训练期间的模型。Actor和supervised fine-tuning模型均采用1750亿参数的GPT-3系列模型，而Critic和reward模型采用60亿参数的GPT-3系列模型。

启动原始的ChatGPT训练过程需要数千GB的GPU显存，这远远超出了单个GPU甚至普通数据并行技术的能力。即使引入了张量并行和流水线并行来划分参数，仍然至少需要64个80GB A100 GPU作为硬件基础。此外，流水线由于其复杂性和效率不适合 AIGC 的生成任务，使得 ChatGPT 训练过程的代码复现更加困难和具有挑战性。

### 使用 Colossal-AI 优化类似 ChatGPT 的训练：硬件节省 50%，速度提高 7.73 倍

Colossal-AI开源复制了ChatGPT的训练过程，包括预训练、奖励模型训练、强化学习训练三个阶段，这是过程中最复杂的阶段。此外，Colossal-AI 通过使用高级内存管理技术减少了类似 ChatGPT 训练的 GPU 内存开销。它只需要一半的硬件资源就可以开始训练一个 1750 亿参数的模型，从而为 ChatGPT 类应用程序节省大量成本。在相同的硬件资源下，Colossal-AI能够以更短的时间完成训练，降低训练成本，加速产品迭代。为了让 ChatGPT 训练过程更容易为开发者所接受，Colossal-AI 除了原有的 1750 亿参数版本外，还提供了高效的单 GPU 和独立 4/8-GPU 版本，以缓解硬件限制。

![img](https://www.hpc-ai.tech/hs-fs/hubfs/Google%20Drive%20Integration/Open%20source%20solution%20replicates%20ChatGPT%20training%20process-1.png?width=855&height=235&name=Open%20source%20solution%20replicates%20ChatGPT%20training%20process-1.png)

在单个多 GPU 服务器上，即使使用最高端的 A100 80GB GPU，由于 ChatGPT 过程的复杂性和内存碎片，PyTorch 也只能使用 GPT-L（774M）等小型模型启动 ChatGPT。因此，使用 PyTorch 的 DistributedDataParallel (DDP) 扩展到 4 或 8 个 GPU 提供的性能提升有限。Colossal-AI 不仅为单 GPU 训练提供了显着的速度和效率提升，而且还可以随着并行度的增加而进一步提升。单服务器训练速度提高 7.73 倍，单 GPU 推理速度提高 1.42 倍，并且可以继续扩展到大规模并行，降低复制 ChatGPT 的成本。

![img](https://www.hpc-ai.tech/hs-fs/hubfs/Google%20Drive%20Integration/Open%20source%20solution%20replicates%20ChatGPT%20training%20process-Feb-14-2023-02-23-46-5123-PM.png?width=606&height=337&name=Open%20source%20solution%20replicates%20ChatGPT%20training%20process-Feb-14-2023-02-23-46-5123-PM.png)



为了最大限度地降低培训成本并提高易用性，Colossal-AI 提供了类似 ChatGPT 培训过程的单 GPU 版本。与只能在 14999 美元的 A100 80GB GPU 上启动最多 7.8 亿个参数的模型的 PyTorch 相比，Colossal-AI 将单个 GPU 的容量提高了 10.3 倍，达到 80 亿个参数。要复制基于具有 1.2 亿个参数的小型模型的 ChatGPT 训练，至少需要 1.62GB 的 GPU 内存，这在任何消费级 GPU 上都可以轻松获得。PyTorch 和 Colossal-AI 在各种设备上的吞吐量对比如下表所示，起始设备为配备 10GB GPU 显存和 128GB CPU 显存的 NVIDIA GeForce RTX 3080 显卡。

![img](https://www.hpc-ai.tech/hs-fs/hubfs/Google%20Drive%20Integration/Open%20source%20solution%20replicates%20ChatGPT%20training%20process-Feb-14-2023-02-23-47-4165-PM.png?width=624&name=Open%20source%20solution%20replicates%20ChatGPT%20training%20process-Feb-14-2023-02-23-47-4165-PM.png)

此外，Colossal-AI 一直致力于降低基于预训练大型模型的微调任务的成本。例如，对于与 ChatGPT OPT 模型相关的微调任务，与 PyTorch 相比，Colossal-AI 能够将微调模型在单个 GPU 上的容量提高多达 3.7 倍，同时保持高速。

### 使用 Colossal-AI 的开源框架开发类似 ChatGPT 的训练过程

让我们深入研究代码，看看 Colossal-AI 如何让 AI 开发人员轻松高效且经济高效地为 ChatGPT 等解决方案训练模型。

Colossal-AI 提供了一个随时可用的 ChatGPT 训练代码，允许用户以 ChatGPT 风格的方式轻松训练来自 Hugging Face 社区的流行预训练模型，如 GPT、OPT 和 BLOOM。首先，只需使用一行代码指定使用 Colossal-AI 作为系统策略，如训练 GPT 模型的示例所示。

```python
from chatgpt.nn import GPTActor, GPTCritic, RewardModel
from chatgpt.trainer import PPOTrainer
from chatgpt.trainer.strategies import ColossalAIStrategy

strategy = ColossalAIStrategy(stage=3, placement_policy='cuda')

with strategy.model_init_context():
    actor = GPTActor().cuda()
    critic = GPTCritic().cuda()
    initial_model = deepcopy(actor).cuda()
    reward_model = RewardModel(deepcopy(critic.model)).cuda()

trainer = PPOTrainer(strategy, actor, critic, reward_model, initial_model, ...)
trainer.fit(prompts)
```

只需几个简单的命令，您就可以轻松开始在单 GPU 规模、单机多 GPU 规模甚至 ChatGPT 原始 1750 亿参数规模版本上训练模型。您还可以使用最大 GPU 内存使用率、吞吐量和 TFLOPS 等指标来评估模型的性能。

```shell
# Training GPT2-S using a single card, a minimum batch size, Colossal-AI Gemini CPU strategy
torchrun --standalone --nproc_pero_node 1 benchmark_gpt_dummy.py --model s --strategy colossalai_gemini_cpu --experience_batch_size 1 --train_batch_size 1
# Training GPT2-XL with a 4-GPU machine, Colossal-AI Zero2 strategy
torchrun --standalone --nproc_per_node 4 benchmark_gpt_dummy.py --model xl --strategy colossalai_zero2
# Training GPT-3 with 4 8-GPU servers, Colossal-AI Gemini CPU strategy
torchrun --nnodes 4 --nproc_per_node 8 \
 --rdzv_id=$JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$HOST_NODE_ADDR \
 benchmark_gpt_dummy.py --model 175b --strategy colossalai_gemini_cpu --experience_batch_size 1 --train_batch_size 1
```

源代码和协作Colossal-AI 的开源代码可在https://github.com/hpcaitech/ColossalAI/tree/main/applications/ChatGPT获取。