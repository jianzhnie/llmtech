# Illustrating Reinforcement Learning from Human Feedback (RLHF)

OpenAI 推出的 ChatGPT 对话模型掀起了新的 AI 热潮，它面对多种多样的问题对答如流，似乎已经打破了机器和人的边界。这一工作的背后是大型语言模型 (Large Language Model，LLM) 生成领域的新训练范式：RLHF (Reinforcement Learning from Human Feedback) ，即以强化学习方式依据人类反馈优化语言模型。

在过去的几年中，语言模型根据人类输入提示(prompt) 生成多样化文本的能力令人印象深刻。然而，对生成结果的评估是主观和依赖上下文的。例如，我们希望模型生成一个有创意的故事、一段真实的信息性文本，或者是可执行的代码片段，这些结果难以用现有的基于规则的文本生成指标 (如 BLUE 和 ROUGE) 来衡量。除了评估指标，现有的模型通常以预测下一个单词的方式和简单的损失函数 (如交叉熵) 来建模，没有显式地引入人的偏好和主观意见。

如果我们使用生成文本的人工反馈作为性能衡量标准，或者更进一步并使用该反馈作为损失来优化模型，那不是很好吗？这就是从人类反馈中强化学习（RLHF）的想法；使用强化学习的方法直接优化带有人类反馈的语言模型。RLHF 使得在一般文本数据语料库上训练的模型与复杂的人类价值观对齐。

RLHF 最近的成功是在[ChatGPT](https://openai.com/blog/chatgpt/)中的使用。鉴于 ChatGPT 令人印象深刻的能力，我们请它为我们解释 RLHF：


<div align=center>
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/rlhf/chatgpt-explains.png" alt="img" style="zoom:50%;" />
</div>

ChatGPT 解释的很好，但还没有完全讲透；让我们更具体一点吧！

## RLHF 技术分解

RLHF 是一项涉及多个模型和不同训练阶段的复杂概念，这里我们按三个步骤分解：

1. 预训练一个语言模型（LM）;
2. 收集问答数据并训练奖励模型(Reward Model，RM) ；
3. 通过强化学习(RL) 微调 LM。

首先，我们将了解如何对语言模型进行预训练。

### Step1: 预训练语言模型

首先，我们使用经典的预训练目标训练一个语言模型。（有关更多详细信息，请参阅[博客文章](https://huggingface.co/blog/how-to-train)）。OpenAI 在其第一个流行的 RLHF 模型[InstructGPT](https://openai.com/blog/instruction-following/)中使用了较小版本的 GPT-3 。Anthropic 使用了 1000 万到 520 亿个参数的 Transformer 模型进行训练。DeepMind 使用了他们的 2800 亿参数模型[Gopher](https://arxiv.org/abs/2112.11446)。

这个初始模型也可以根据额外的文本或条件进行微调，例如，OpenAI 对“更可取”的人工生成文本进行了微调，而 Anthropic 通过根据“有用、诚实和无害”的标准在上下文线索上蒸馏了原始的 LM。这里或许使用了昂贵的增强数据，但并不是 RLHF 必须的一步。由于 RLHF 还是一个尚待探索的领域，对于” 哪种模型” 适合作为 RLHF 的起点并没有明确的答案。


<div align=center>
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/rlhf/pretraining.png" alt="img" style="zoom:50%;" />
</div>
接下来，我们会基于 LM 来生成训练奖励模型 (RM，也叫偏好模型) 的数据，并在这一步引入人类的偏好信息。

### Step2：奖励模型训练

生成一个根据人类偏好校准的奖励模型（RM，也称为偏好模型）是 RLHF 区别于旧范式的开端。这一模型接收一系列文本，并返回一个标量奖励，数值上代表人类偏好。该系统可以用端到端的 方式用LM建模，或使用模块化的奖励系统建模（例如，模型对输出进行排名，并将排名转换为奖励）。这一奖励数值对于稍后在 RLHF 过程中无缝集成的现有 RL 算法至关重要。

关于模型选择方面，用于奖励建模的 LM 可以是另一个经过微调的 LM，也可以是根据偏好数据从头开始训练的 LM。例例如 Anthropic 提出了一种特殊的预训练方式，即用偏好模型预训练 (Preference Model Pretraining，PMP) 来替换一般预训练后的微调过程。因为前者被认为对样本数据的利用率更高。但对于哪种 RM 更好尚无定论。

关于训练文本方面，RM 的提示 - 生成对文本是从预定义数据集中采样生成的，并用初始的 LM 给这些提示生成文本。Anthropic 的数据主要是通过 Amazon Mechanical Turk 上的聊天工具生成的，在 Hub 上[可用](https://huggingface.co/datasets/Anthropic/hh-rlhf)。 而 OpenAI 使用了用户提交给 GPT API 的 prompt。

关于训练奖励数值方面，这里需要人工对 LM 生成的回答进行排名。起初我们可能会认为应该直接对文本标注分数来训练 RM，但是由于标注者的价值观不同导致这些分数未经过校准并且充满噪音。通过排名可以比较多个模型的输出并构建更好的规范数据集。

有多种方法可以对文本进行排名。一种成功的方法是让用户比较基于相同提示的两种语言模型生成的文本。然后使用 [Elo](https://en.wikipedia.org/wiki/Elo_rating_system) 系统建立一个完整的排名。这些不同的排名结果将被归一化为用于训练的标量奖励值。

这个过程中一个有趣的产物是目前成功的 RLHF 系统使用了和生成模型具有 不同 大小的 LM (例如 OpenAI 使用了 175B 的 LM 和 6B 的 RM，Anthropic 使用的 LM 和 RM 从 10B 到 52B 大小不等，DeepMind 使用了 70B 的 Chinchilla 模型分别作为 LM 和 RM) 。一种直觉是，偏好模型和生成模型需要具有类似的能力来理解提供给它们的文本。因为模型需要具有类似的能力才能生成所述文本。

<div align=center>
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/rlhf/reward-model.png" alt="img" style="zoom: 33%;" />
</div>

在 RLHF 系统上，我们有一个可用于生成文本的初始语言模型和一个接收任何文本并为其分配人类感知程度分数的偏好模型。接下来，我们使用强化学习 (RL)来优化初始语言模型。

### Step3: 使用 RL 进行微调

长期以来，出于工程和算法原因，人们人为用强化学习训练语言模型是不可能的。而目前多个组织找到的可行方案是使用策略梯度强化学习 (Policy Gradient RL) 算法、近端策略优化 (Proximal Policy Optimization，PPO) 微调初始 LM 的部分或全部参数。因为微调整个 10B 或 100B+ 参数模型的成本过高（有关更多信息，请参阅 LM 的低秩适应 ( [LoRA](https://arxiv.org/abs/2106.09685) ) 或 DeepMind 的[Sparrow](https://arxiv.org/abs/2209.14375) LM）。PPO 已经存在了相对较长的时间——有[大量](https://spinningup.openai.com/en/latest/algorithms/ppo.html)的[指南](https://huggingface.co/blog/deep-rl-ppo)介绍它的原理和使用技巧。因而成为 RLHF 中的有利选择。

事实证明，RLHF 的许多核心 RL 进步一直在弄清楚如何使用熟悉的算法更新如此大的模型（稍后会详细介绍）。

让我们首先将微调任务表述为 RL 问题。首先，该策略是一个接受提示并返回一系列文本（或文本的概率分布）的LM。这个策略（policy）的行动空间 （action space）是语言模型的词表对应的所有词元（通常在50k tokens数量级），观察空间 （observation space） 是可能的输入词元序列，也比较大（词汇量^输入标记的数量）。奖励函数是偏好模型和策略转变约束(Policy shift constraint)的结合。

PPO 算法确定的奖励函数具体计算如下：将提示 x 输入初始 LM 和当前微调的 LM，分别得到了输出文本 y1, y2，将来自当前策略的文本传递给 RM 得到一个标量的奖励$r_{\theta}$ . 将两个模型的生成文本进行比较计算差异的惩罚项，在 OpenAI、Anthropic 和 DeepMind 的多篇论文中，这种惩罚被设计为这些出词分布序列之间的 Kullback–Leibler (KL) 散度的缩放 $r = r_{\theta} - \lambda r_{KL}$. 这一项被用于惩罚 RL 策略在每个训练批次中生成大幅偏离初始模型，以确保模型输出合理连贯的文本。如果去掉这一惩罚项可能导致模型在优化中生成乱码文本来愚弄奖励模型提供高奖励值。此外，OpenAI 在 InstructGPT 上实验了在 PPO 添加新的预训练梯度，可以预见到奖励函数的公式会随着 RLHF 研究的进展而继续进化。

最后根据 PPO 算法，我们按当前批次数据的奖励指标进行优化 (来自 PPO 算法 on-policy 的特性) 。PPO 算法是一种信赖域优化 (Trust Region Optimization，TRO) 算法，它使用梯度约束确保更新步骤不会破坏学习过程的稳定性。DeepMind 对 Gopher 使用了类似的奖励设置，但是使用 A2C (synchronous advantage actor-critic) 算法来优化梯度。

<div align=center>
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/rlhf/rlhf.png" alt="img" style="zoom: 33%;" />
</div>

作为一个可选项，RLHF 可以通过迭代 RM 和策略共同优化。随着策略模型更新，用户可以继续将输出和早期的输出进行合并排名。Anthropic 在他们的论文中讨论了迭代在线 RLHF（请参阅原始[论文](https://arxiv.org/abs/2204.05862)），其中策略的迭代包含在跨模型的 Elo 排名系统中。这样引入策略和 RM 演变的复杂动态，代表了一个复杂和开放的研究问题。

# RLHF 的下一步是什么？

虽然这些技术非常有前途和影响力，并引起了人工智能领域最大研究实验室的注意，但仍然存在明显的局限性。这些模型依然然会毫无不确定性地输出有害或实际上不真实的文本。这种不完美代表了 RLHF 的长期挑战和动力——在人类固有的问题领域中运行意味着永远不会到达一个完美的标准。

在使用 RLHF 部署系统时，由于强制性和深思熟虑的人为因素，收集人类偏好数据非常昂贵。RLHF 性能仅与其人工注释的质量一样好，人工注释有两种：人工生成的文本，例如微调 InstructGPT 中的初始 LM，以及模型输出之间人类偏好的标签。

收集人类偏好数据的质量和数量决定了 RLHF 系统性能的上限。RLHF 系统需要两种人类偏好数据：人工生成的文本和对模型输出的偏好标签。生成高质量回答需要雇佣兼职人员 (而不能依赖产品用户和众包) 。另一方面，训练 RM 需要的奖励标签规模大概是 50k 左右，所以并不那么昂贵 (当然远超了学术实验室的预算) 。目前相关的数据集只有一个基于通用 LM 的 RLHF 数据集 (来自 Anthropic 的 [hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf) ) 和几个较小的子任务数据集 (如来自 OpenAI 的摘要数据集 [summarize-from-feedback](https://github.com/openai/summarize-from-feedback)) 。另一个挑战来自标注者的偏见。几个人类标注者可能有不同意见，导致了训练数据存在一些潜在差异。

除开数据方面的限制，一些有待开发的设计选项可以让 RLHF 取得长足进步。例如对 RL 优化器的改进方面，PPO 是一种较旧的算法，但目前没有什么结构性原因让其他算法可以在现有 RLHF 工作中更具有优势。另外，微调 LM 策略的一大成本是策略生成的文本都需要在 RM 上进行评估，通过离线 RL 优化策略可以节约这些大模型 RM 的预测成本。最近，出现了新的 RL 算法如隐式语言 Q 学习 (Implicit Language Q-Learning，ILQL) 也适用于当前 RL 的优化。在 RL 训练过程的其他核心权衡，例如探索和开发 (exploration-exploitation) 的平衡也有待尝试和记录。探索这些方向至少能加深我们对 RLHF 的理解，更进一步提升系统的表现。

### 延伸阅读

首先介绍一些相关的开源工作：

关于 RLHF 的第一个项目，来自 OpenAI:
https://github.com/openai/lm-human-preferences

一些 PyTorch 的 repo：

- https://github.com/lvwerra/trl
- https://github.com/CarperAI/trlx
- https://github.com/allenai/RL4LMs

TRL 旨在使用 PPO 微调 Hugging Face 生态系统中的预训练 LM。TRLX 是 CarperAI 构建的 TRL 扩展分支，[用于](https://carper.ai/)处理更大的在线和离线训练模型。目前，TRLX 拥有一个 API，能够在 LLM 部署所需的规模（例如 330 亿个参数）下使用 PPO 和隐式语言 Q-Learning [ILQL进行生产就绪的 RLHF。](https://sea-snell.github.io/ILQL_site/)TRLX 的未来版本将允许最多 200B 个参数的语言模型。因此，与 TRLX 的接口针对具有这种规模经验的机器学习工程师进行了优化。

[RL4LMs](https://github.com/allenai/RL4LMs) 提供构建块，用于使用各种 RL 算法（PPO、NLPO、A2C 和 TRPO）、奖励函数和指标来微调和评估 LLM。此外，该库易于定制，允许在任意用户指定的奖励函数上训练任何编码器-解码器或基于编码器转换器的 LM。[值得注意的是，它在最近的工作](https://arxiv.org/abs/2210.01241)中针对广泛的任务进行了良好的测试和基准测试，总计多达 2000 次实验，突出了关于数据预算比较（专家演示与奖励建模）、处理奖励黑客和训练不稳定性等方面的一些实用见解。RL4LMs目前的计划包括分布式训练更大的模型和新的 RL 算法。

TRLX 和 RL4LM 都在大力进一步开发中，因此很快就会有更多的功能。

此外，Huggingface Hub 上有一个由 Anthropic 创建的大型数据集:
https://hf.co/datasets/Anthropic/hh-rlhf

相关论文包括在现有 LM 前的 RLHF 进展和基于当前 LM 的 RLHF 工作：

- TAMER: Training an Agent Manually via Evaluative Reinforcement (Knox and Stone 2008)
- Interactive Learning from Policy-Dependent Human Feedback (MacGlashan et al. 2017)
- Deep Reinforcement Learning from Human Preferences (Christiano et al. 2017)
- Deep TAMER: Interactive Agent Shaping in High-Dimensional State Spaces (Warnell et al. 2018)
- Fine-Tuning Language Models from Human Preferences (Zieglar et al. 2019)
- Learning to summarize with human feedback (Stiennon et al., 2020)
- Recursively Summarizing Books with Human Feedback (OpenAI Alignment Team 2021)
- WebGPT: Browser-assisted question-answering with human feedback (OpenAI, 2021)
- InstructGPT: Training language models to follow instructions with human feedback (OpenAI Alignment Team 2022)
- GopherCite: Teaching language models to support answers with verified quotes (Menick et al. 2022)
- Sparrow: Improving alignment of dialogue agents via targeted human judgements (Glaese et al. 2022)
- ChatGPT: Optimizing Language Models for Dialogue (OpenAI 2022)
- Scaling Laws for Reward Model Overoptimization (Gao et al. 2022)
- Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback (Anthropic, 2022)
- Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned (Ganguli et al. 2022)
- Dynamic Planning in Open-Ended Dialogue using Reinforcement Learning (Cohen at al. 2022)
- Is Reinforcement Learning (Not) for Natural Language Processing?: Benchmarks, Baselines, and Building Blocks for Natural Language Policy Optimization (Ramamurthy and Ammanabrolu et al. 2022)

> 本文翻译自 Hugging Face 官方博客 (https://hf.co/blog/rlhf)
> 参考资料部分链接请点击阅读原文到博客上查看。
