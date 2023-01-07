# Illustrating Reinforcement Learning from Human Feedback (RLHF)

在过去的几年中，语言模型通过根据人类输入提示生成多样化且引人注目的文本显示出令人印象深刻的能力。然而，什么才是“好”文本本质上很难定义，因为它是主观的并且依赖于上下文。有许多应用程序，例如编写您需要创意的故事、应该真实的信息性文本片段，或者我们希望可执行的代码片段。

编写一个损失函数来捕获这些属性似乎很棘手，而且大多数语言模型仍然使用简单的 next token prediction loss（例如交叉熵）进行训练。为了弥补损失本身的缺点，人们定义了旨在更好地捕捉人类偏好的指标，例如[BLEU](https://en.wikipedia.org/wiki/BLEU)或[ROUGE](https://en.wikipedia.org/wiki/ROUGE_(metric)). 虽然比损失函数本身更适合衡量性能，但这些指标只是简单地将生成的文本与具有简单规则的引用进行比较，因此也有局限性。如果我们使用生成文本的人工反馈作为性能衡量标准，或者更进一步并使用该反馈作为损失来优化模型，那不是很好吗？这就是从人类反馈中强化学习（RLHF）的想法；使用强化学习的方法直接优化带有人类反馈的语言模型。RLHF 使语言模型能够开始将在一般文本数据语料库上训练的模型与复杂人类价值观的模型对齐。

RLHF 最近的成功是在[ChatGPT](https://openai.com/blog/chatgpt/)中的使用。鉴于 ChatGPT 令人印象深刻的能力，我们请它为我们解释 RLHF：

![img](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/rlhf/chatgpt-explains.png)

它的表现出奇的好，但并没有完全涵盖所有内容。我们将填补这些空白！

# RLHF：让我们一步步来

从人类反馈中强化学习（也称为来自人类偏好的 RL）是一个具有挑战性的概念，因为它涉及多模型训练过程和不同的部署阶段。在这篇博文中，我们将训练过程分解为三个核心步骤：

1. 预训练语言模型（LM），
2. 收集数据并训练奖励模型，以及
3. 通过强化学习微调 LM。

首先，我们将了解如何对语言模型进行预训练。

### 预训练语言模型

作为起点，RLHF 使用已经使用经典预训练目标进行预训练的语言模型（有关更多详细信息，请参阅此[博客文章](https://huggingface.co/blog/how-to-train)）。OpenAI 在其第一个流行的 RLHF 模型[InstructGPT](https://openai.com/blog/instruction-following/)中使用了较小版本的 GPT-3 。Anthropic 使用了 1000 万到 520 亿个参数的 Transformer 模型为此任务进行了训练。DeepMind 使用了他们的 2800 亿参数模型[Gopher](https://arxiv.org/abs/2112.11446)。

这个初始模型也可以根据额外的文本或条件进行微调，但不一定需要。例如，OpenAI 对“更可取”的人工生成文本进行了微调，而 Anthropic 通过根据“有用、诚实和无害”的标准提取上下文线索的原始 LM，为 RLHF 生成了初始 LM。这些都是我所说的昂贵的*增强*数据的来源，但这不是理解 RLHF 所必需的技术。

一般来说，对于“哪种模型”最适合作为 RLHF 的起点，并没有明确的答案。这将是本博客的一个共同主题——RLHF 训练中选项的设计空间没有得到彻底探索。

接下来，使用语言模型，需要生成数据来训练**奖励模型**，这就是将人类偏好集成到系统中的方式。

![img](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/rlhf/pretraining.png)

### 奖励模型训练

生成一个根据人类偏好校准的奖励模型（RM，也称为偏好模型）是 RLHF 中相对较新的研究开始的地方。基本目标是获得一个模型或系统，该模型或系统接收一系列文本，并返回一个标量奖励，该奖励应在数字上代表人类偏好。该系统可以是端到端的 LM，或输出奖励的模块化系统（例如，模型对输出进行排序，并将排名转换为奖励）。作为**标量** **奖励**的输出对于稍后在 RLHF 过程中无缝集成的现有 RL 算法至关重要。

这些用于奖励建模的 LM 可以是另一个经过微调的 LM，也可以是根据偏好数据从头开始训练的 LM。例如，Anthropic 在预训练（偏好模型预训练，PMP）后使用一种专门的微调方法来初始化这些模型，因为他们发现它比微调更有效，但没有一种奖励建模的变体被认为是最好的今天的选择。

RM 的提示生成对的训练数据集是通过从预定义数据集中采样一组提示生成的（Anthropic 的数据主要是通过 Amazon Mechanical Turk 上的聊天工具生成的，在 Hub 上[可用](https://huggingface.co/datasets/Anthropic/hh-rlhf)，而 OpenAI 使用用户提交的提示来GPT API）。提示通过初始语言模型生成新文本。

人工注释器用于对 LM 生成的文本输出进行排名。一开始可能会认为人类应该直接对每段文本应用标量分数以生成奖励模型，但这在实践中很难做到。人类的不同价值观导致这些分数未经校准且嘈杂。相反，排名用于比较多个模型的输出并创建更好的正则化数据集。

有多种方法可以对文本进行排名。一种成功的方法是让用户比较基于相同提示的两种语言模型生成的文本。通过比较正面对决中的模型输出，[Elo](https://en.wikipedia.org/wiki/Elo_rating_system)系统可用于生成模型和输出相对于彼此的排名。这些不同的排名方法被归一化为用于训练的标量奖励信号。

这个过程的一个有趣的产物是，迄今为止成功的 RLHF 系统使用了相对于文本生成具有不同大小的奖励语言模型（例如 OpenAI 175B LM，6B 奖励模型，Anthropic 使用 LM 和奖励模型从 10B 到 52B，DeepMind 使用70B Chinchilla 模型，用于 LM 和奖励）。一种直觉是，这些偏好模型需要具有类似的能力来理解提供给它们的文本，因为模型需要具有类似的能力才能生成所述文本。

在 RLHF 系统的这一点上，我们有一个可用于生成文本的初始语言模型和一个接收任何文本并为其分配人类感知程度分数的偏好模型。接下来，我们使用**强化学习 (RL)**来针对奖励模型优化原始语言模型。

![img](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/rlhf/reward-model.png)

### 使用 RL 进行微调

长期以来，人们认为出于工程和算法原因，用强化学习训练语言模型是不可能的。多个组织似乎已经开始工作的是使用策略梯度 RL 算法、近端策略优化 (PPO)微调**初始 LM 副本的部分或全部参数。**LM 的参数被冻结，因为微调整个 10B 或 100B+ 参数模型的成本过高（有关更多信息，请参阅 LM 的低秩适应 ( [LoRA](https://arxiv.org/abs/2106.09685) ) 或 DeepMind 的[Sparrow](https://arxiv.org/abs/2209.14375) LM）。PPO 已经存在了相对较长的时间——有[大量](https://spinningup.openai.com/en/latest/algorithms/ppo.html)的[指南](https://huggingface.co/blog/deep-rl-ppo)关于它是如何工作的。这种方法的相对成熟度使其成为扩展到 RLHF 分布式训练新应用的有利选择。事实证明，RLHF 的许多核心 RL 进步一直在弄清楚如何使用熟悉的算法更新如此大的模型（稍后会详细介绍）。

让我们首先将此微调任务表述为 RL 问题。首先，该**策略**是一种语言模型，它接受提示并返回一系列文本（或只是文本的概率分布）。这个policy的**action space**是语言模型的vocabulary对应的所有token（通常在50k tokens数量级），**observation space**是可能的输入token序列，也比较大（词汇量^数量）输入标记）。**奖励函数**是偏好模型和政策转变约束的结合。

奖励函数是系统将我们讨论过的所有模型组合到一个 RLHF 过程中的地方。给出来自数据集的提示*x*，生成两个文本*y1*和*y2* - 一个来自初始语言模型，一个来自微调策略的当前迭代。来自当前策略的文本被传递到偏好模型，该模型返回“偏好”的标量概念，r_\theta*r**一世*. 将该文本与来自初始模型的文本进行比较，以计算对它们之间差异的惩罚。在 OpenAI、Anthropic 和 DeepMind 的多篇论文中，这种惩罚被设计为这些代币分布序列之间的 Kullback–Leibler [(KL) 散度的缩放版本，](https://en.wikipedia.org/wiki/Kullback–Leibler_divergence)r_\text{KL}*r*在. KL 散度项惩罚 RL 策略在每个训练批次中大幅偏离初始预训练模型，这有助于确保模型输出合理连贯的文本片段。如果没有这种惩罚，优化可能会开始生成乱码的文本，但会愚弄奖励模型以提供高奖励。在实践中，KL 散度是通过从两个分布中采样来近似的（由 John Schulman[在这里](http://joschu.net/blog/kl-approx.html)解释）。发送到 RL 更新规则的最终奖励是r = r_\theta - \lambda r_\text{KL}*r*=*r**一世*−*λr* *_*在.

一些 RLHF 系统在奖励函数中添加了额外的项。例如，OpenAI 通过将额外的预训练梯度（来自人类注释集）混合到 PPO 的更新规则中，在 InstructGPT 上成功进行了实验。随着 RLHF 的进一步研究，这种奖励函数的公式可能会继续发展。

最后，**更新规则**是来自 PPO 的参数更新，它最大化当前批次数据中的奖励指标（PPO 是 on-policy，这意味着参数只用当前批次的提示生成对更新）。PPO 是一种信赖域优化算法，它使用梯度约束来确保更新步骤不会破坏学习过程的稳定性。DeepMind 对 Gopher 使用了类似的奖励设置，但使用[同步优势演员评论家](http://proceedings.mlr.press/v48/mniha16.html?ref=https://githubhelp.com)(A2C) 来优化梯度，这明显不同但尚未在外部复制。

![img](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/rlhf/rlhf.png)

可选地，RLHF 可以通过一起迭代更新奖励模型和策略从这一点继续。随着 RL 策略的更新，用户可以继续将这些输出与模型的早期版本进行排名。大多数论文尚未讨论实现此操作，因为收集此类数据所需的部署模式仅适用于可以访问参与用户群的对话代理。Anthropic 将此选项讨论为*迭代在线 RLHF*（请参阅原始[论文](https://arxiv.org/abs/2204.05862)），其中策略的迭代包含在跨模型的 ELO 排名系统中。这引入了政策和奖励模型演变的复杂动态，这代表了一个复杂而开放的研究问题。

# RLHF 的开源工具

第一个在 LM 上执行 RLHF的[代码来自于 2019 年在 TensorFlow 中的 OpenAI。](https://github.com/openai/lm-human-preferences)

今天，PyTorch 中已经有一些活跃的 RLHF 存储库由此产生。主要存储库是 Transformers Reinforcement Learning ( [TRL](https://github.com/lvwerra/trl) )、[TRLX](https://github.com/CarperAI/trlx)（起源于 TRL 的一个分支）和 Reinforcement Learning for Language models ( [RL4LMs](https://github.com/allenai/RL4LMs) )。

TRL 旨在使用 PPO 微调 Hugging Face 生态系统中的预训练 LM。TRLX 是 CarperAI 构建的 TRL 扩展分支，[用于](https://carper.ai/)处理更大的在线和离线训练模型。目前，TRLX 拥有一个 API，能够在 LLM 部署所需的规模（例如 330 亿个参数）下使用 PPO 和隐式语言 Q-Learning [ILQL进行生产就绪的 RLHF。](https://sea-snell.github.io/ILQL_site/)TRLX 的未来版本将允许最多 200B 个参数的语言模型。因此，与 TRLX 的接口针对具有这种规模经验的机器学习工程师进行了优化。

[RL4LMs](https://github.com/allenai/RL4LMs)提供构建块，用于使用各种 RL 算法（PPO、NLPO、A2C 和 TRPO）、奖励函数和指标来微调和评估 LLM。此外，该库易于定制，允许在任意用户指定的奖励函数上训练任何编码器-解码器或基于编码器转换器的 LM。[值得注意的是，它在最近的工作](https://arxiv.org/abs/2210.01241)中针对广泛的任务进行了良好的测试和基准测试，总计多达 2000 次实验，突出了关于数据预算比较（专家演示与奖励建模）、处理奖励黑客和训练不稳定性等方面的一些实用见解。RL4LMs目前的计划包括分布式训练更大的模型和新的 RL 算法。

TRLX 和 RL4LM 都在大力进一步开发中，因此很快就会有更多的功能。

Hub 上有一个由 Anthropic 创建的大型[数据集。](https://huggingface.co/datasets/Anthropic/hh-rlhf)

# RLHF 的下一步是什么？

虽然这些技术非常有前途和影响力，并引起了人工智能领域最大研究实验室的注意，但仍然存在明显的局限性。这些模型虽然更好，但仍然可以毫无不确定性地输出有害或实际上不准确的文本。这种不完美代表了 RLHF 的长期挑战和动力——在人类固有的问题领域中运行意味着永远不会有明确的最后一条线来跨越模型以标记为*完整*。

在使用 RLHF 部署系统时，由于强制性和深思熟虑的人为因素，收集人类偏好数据非常昂贵。RLHF 性能仅与其人工注释的质量一样好，人工注释有两种：人工生成的文本，例如微调 InstructGPT 中的初始 LM，以及模型输出之间人类偏好的标签。

生成写得很好的人工文本来回答特定的提示是非常昂贵的，因为它通常需要雇用兼职人员（而不是能够依赖产品用户或众包）。值得庆幸的是，用于训练大多数 RLHF 应用的奖励模型的数据规模（~50k 标记偏好样本）并不那么昂贵。然而，它仍然比学术实验室可能负担得起的成本更高。[目前，只有一个基于通用语言模型（来自Anthropic](https://huggingface.co/datasets/Anthropic/hh-rlhf) ）的 RLHF 的大规模数据集和几个较小规模的特定于任务的数据集（例如来自[OpenAI](https://github.com/openai/summarize-from-feedback)的摘要数据）。RLHF 数据的第二个挑战是人类注释者通常会不同意，从而在没有基本事实的情况下为训练数据增加大量潜在差异。

由于这些限制，大量未开发的设计选项仍然可以使 RLHF 取得长足进步。其中许多属于改进 RL 优化器的领域。PPO 是一种相对较旧的算法，但没有其他算法可以为现有 RLHF 工作流提供好处和排列的结构性原因。微调 LM 策略的反馈部分的一大成本是策略生成的每段文本都需要在奖励模型上进行评估（因为它就像标准 RL 框架中环境的一部分）。为了避免大型模型的这些昂贵的前向传递，离线 RL 可以用作策略优化器。最近，出现了新的算法，例如[隐式语言 Q-learning](https://arxiv.org/abs/2206.11871) (ILQL) [ [Talk](https://youtu.be/fGq4np3brbs)在 CarperAI 的 ILQL 上，特别适合这种类型的优化。RL 过程中的其他核心权衡，如探索-开发平衡，也没有记录在案。探索这些方向至少会加深对 RLHF 功能的理解，如果不能，也会提供改进的性能。

我们将在 12 月 13 日下周二举办一场讲座，该讲座将扩展这篇文章。您可以在太平洋标准时间 830加入[这里！](https://www.youtube.com/watch?v=2MBJOuVq380&feature=youtu.be)

### 延伸阅读

以下是迄今为止关于 RLHF 最流行的论文列表。该领域最近随着 DeepRL 的出现（2017 年左右）而得到普及，并已发展成为对许多大型科技公司的 LLM 应用的更广泛研究。以下是一些早于 LM 焦点的关于 RLHF 的论文：

- [TAMER：Training an Agent Manually via Evaluative Reinforcement](https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/ICDL08-knox.pdf)（Knox 和 Stone 2008）：提出了一个学习的代理，其中人类提供迭代采取的行动的分数以学习奖励模型。
- [Interactive Learning from Policy-Dependent Human Feedback](http://proceedings.mlr.press/v70/macglashan17a/macglashan17a.pdf) (MacGlashan et al. 2017)：提出了一种演员-评论家算法 COACH，其中人类反馈（正面和负面）用于调整优势函数。
- [Deep Reinforcement Learning from Human Preferences](https://proceedings.neurips.cc/paper/2017/hash/d5e2c0adad503c91f91df240d0cd4e49-Abstract.html) (Christiano et al. 2017)：RLHF 应用于 Atari 轨迹之间的偏好。
- [Deep TAMER: Interactive Agent Shaping in High-Dimensional State Spaces](https://ojs.aaai.org/index.php/AAAI/article/view/11485) (Warnell et al. 2018)：扩展了 TAMER 框架，其中使用深度神经网络对奖励预测进行建模。

以下是越来越多的论文的快照，这些论文显示了 RLHF 对 LM 的性能：

- [Fine-Tuning Language Models from Human Preferences](https://arxiv.org/abs/1909.08593)（Zieglar 等人，2019 年）：一篇研究奖励学习对四项特定任务影响的早期论文。
- [Learning to summarize with human feedback](https://proceedings.neurips.cc/paper/2020/hash/1f89885d556929e98d3ef9b86448f951-Abstract.html) (Stiennon et al., 2020)：RLHF 应用于文本摘要任务。此外，[递归总结带有人类反馈](https://arxiv.org/abs/2109.10862)的书籍（OpenAI 对齐团队 2021），继续总结书籍的工作。
- [WebGPT：带人工反馈的浏览器辅助问答](https://arxiv.org/abs/2112.09332)（OpenAI，2021 年）：使用 RLHF 训练代理来浏览网络。
- InstructGPT：[训练语言模型遵循人类反馈的指令](https://arxiv.org/abs/2203.02155)（OpenAI Alignment Team 2022）：RLHF applied to a general language model [关于 InstructGPT[的博客文章\]。](https://openai.com/blog/instruction-following/)
- GopherCite: [Teaching language models to support answers with verified quotes](https://www.deepmind.com/publications/gophercite-teaching-language-models-to-support-answers-with-verified-quotes) (Menick et al. 2022)：使用 RLHF 训练 LM 以返回带有特定引用的答案。
- Sparrow：[通过有针对性的人类判断改进对话代理的对齐](https://arxiv.org/abs/2209.14375)（Glaese 等人 2022）：使用 RLHF 微调对话代理
- [ChatGPT：优化对话语言模型](https://openai.com/blog/chatgpt/)（OpenAI 2022）：使用 RLHF 训练 LM 以适合用作通用聊天机器人。
- [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760) (Gao et al. 2022)：研究学习偏好模型在 RLHF 中的缩放特性。
- [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862)（Anthropic，2022）：使用 RLHF 训练 LM 助手的详细文档。
- [Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned](https://arxiv.org/abs/2209.07858)（Ganguli 等人，2022 年）：详细记录了“发现、衡量和尝试减少 [语言模型] 潜在有害输出”的努力。
- [Dynamic Planning in Open-Ended Dialogue using Reinforcement Learning](https://arxiv.org/abs/2208.02294)（Cohen 等人，2022 年）：使用 RL 来增强开放式对话代理的会话技能。
- [Is Reinforcement Learning (Not) for Natural Language Processing?: Benchmarks, Baselines, and Building Blocks for Natural Language Policy Optimization](https://arxiv.org/abs/2210.01241)（Ramamurthy 和 Ammanabrolu 等人，2022 年）：讨论 RLHF 中开源工具的设计空间并提出新算法NLPO（自然语言策略优化）作为 PPO 的替代方案。
