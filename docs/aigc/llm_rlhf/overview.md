# RLHF 概述

人类反馈强化学习 (RLHF) 是一种机器学习技术，它利用人类的直接反馈来训练“奖励模型”，然后利用该模型通过强化学习来优化智能体的性能。

RLHF 也称为“基于人类偏好的强化学习”，特别适合处理那些目标复杂、定义不明确或难以精准表述的任务。

RLHF）是强化学习（RL）的一个扩展分支，当决策问题的优化目标比较抽象，难以形式化定义具体的奖励函数时，RLHF 系列方法可以将人类的反馈信息纳入到训练过程，通过使用这些反馈信息构建一个奖励模型神经网络，以此提供奖励信号来帮助 RL 智能体学习，从而更加自然地将人类的需求，偏好，观念等信息以一种交互式的学习方式传达给智能体，对齐（align）人类和人工智能之间的优化目标，产生行为方式和人类价值观一致的系统。

## 研究进程

- 2017 年， OpenAI 和 DeepMind 的研究《Deep reinforcement learning from human preferences》，研究者尝试将人类反馈信息引入 Atari  、MuJoCo 这样的经典决策学术环境，从而取得了一些有趣的发现。
- 2019 年，经过 RLHF 训练的人工智能系统，例如 OpenAI Five 和 DeepMind 的 AlphaStar，分别在更为复杂的 Dota 2和《星际争霸》中击败了人类顶级职业玩家。

- 2020年，OpenAI 首次发布详细介绍 RLHF 在语言模型上使用的代码。
- 2021年，OpenAI 发布了WebGPT 模型，利用人类反馈强化学习（RLHF）来训练一个能够浏览互联网并回答问题的智能体。
- 2022 年，OpenAI 发布了经过 RLHF 训练的 InstructGPT 模型。这是弥合 GPT-3 和 GPT-3.5-turbo 模型之间差距的关键一步，为 ChatGPT 的推出提供了动力。
- 2022 年末，OpenAI 发布了技惊四座的ChatGPT，使用人类反馈强化学习（RLHF）来训练一个适合作为通用聊天机器人的语言模型（LM）。短短几个月内已经有超过一亿的用户尝试并领略到了这种强大对话系统的通用性和便利性。
- 此后，RLHF 便广泛用于 OpenAI、DeepMind、Google和 Anthropic 的先进 LLM 的训练。

## 大语言模型 RLHF

RLHF 最突出的应用之一是使大语言模型能够与复杂的人类价值观对齐， 让大语言模型 (LLM) 变得更靠谱、更精准、更合乎伦理。

根据 OpenAI 的思路，RLHF分为三步：

<img src="https://github.com/opendilab/awesome-RLHF/raw/main/overview_chatgpt.png" alt="image info" style="zoom: 50%;" />

- Step1: 收集示范资料并训练监督式策略（Supervised Policy）。

  - 从提示资料集中抽取一个提示。

  - 标注者展示了期望的输出行为。

  - 利用这些资料对GPT-3.5进行监督学习的微调。

- Step2: 收集比较资料并训练奖励模型（Reward Model）。
  - 抽取一个提示和几个模型输出作为样本。
  - 标注者对这些输出从最好到最差进行排名。
  - 使用这些资料来训练奖励模型。

- Step3: 使用PPO（Proximal Policy Optimization）强化学习算法根据奖励模型优化策略。
  - 从资料集中抽取一个新的提示。
  - PPO模型从监督式策略初始化。
  - 策略生成一个输出。
  - 奖励模型为输出计算奖励。
  - 使用PPO根据奖励更新策略。

### Step1:  监督微调 (SFT)

SFT 使用监督学习来训练模型，引导模型使其生成的响应符合用户的预期格式。

人类专家按照格式 (提示, 响应) 创建带标签的示例，演示对于不同的用例，例如回答问题、摘要或翻译，如何回应提示。

#### 数据格式：

##### 单轮对话

```

```

##### 多轮对话

```json
[
  {
    "id": "identity_0",
    "conversations": [
      {
        "from": "human",
        "value": "Who are you?"
      },
      {
        "from": "gpt",
        "value": "I am Vicuna, a language model trained by researchers from Large Model Systems Organization (LMSYS)."
      },
      {
        "from": "human",
        "value": "What can you do?"
      },
      {
        "from": "gpt",
        "value": "I can chat with you."
      }
    ]
  },
  {
    "id": "identity_1",
    "conversations": [
      {
        "from": "human",
        "value": "What is up?"
      },
      {
        "from": "gpt",
        "value": "Hello! How can I help you today?"
      },
      {
        "from": "human",
        "value": "Who are you?"
      },
      {
        "from": "gpt",
        "value": "You can call me Vicuna, and I was trained by Large Model Systems Organization (LMSYS) researchers as a language model."
      },
      {
        "from": "human",
        "value": "Goodbye"
      },
      {
        "from": "gpt",
        "value": "Goodbye! If you have any more questions in the future, don't hesitate to ask."
      }
    ]
  }
]
```

### Step2:  训练奖励模型（RM）

为了在强化学习中为奖励函数提供人类反馈，需要一个奖励模型来将人类偏好转化为数字奖励信号。设计有效的奖励模型是 RLHF 的关键一步，因为没有简单的数学或逻辑公式可以切实地定义人类的主观价值。

此阶段的主要目的是为奖励模型提供足够的训练数据，包括来自人类评估者的直接反馈，以帮助模型学习模仿人类依据其偏好将奖励分配给不同种类的模型响应的方式。训练可以在没有人工参与的情况下继续离线进行。

奖励模型接收一段文本序列并输出标量奖励值，该值以数值方式预测人类用户对该文本会给予多少奖励（或惩罚）。标量值输出对于奖励模型的输出与 RL 算法其他组成部分的集成至关重要。

相反，评分系统通常是通过比较人类对不同模型输出的反馈来构建。一种常见方法是以一对一配对方式让用户比较两个相似的文本序列，例如两个不同语言模型对同一提示的响应输出，然后使用 Elo 评分系统产生所有生成的文本相对于彼此的聚合排名。这些排名最终都会标准化为标量奖励信号，为奖励模型训练提供信息。

用多个模型（可以是初始模型、第一步的 SFT 模型和人工回复等等）给出问题的多个回答，然后人工给这些问答对按一些标准（可读性、无害性、正确性…）进行排序，训练一个奖励模型/偏好模型来打分（reward model）。

#### 数据格式

> 注意：
>
> 1. 为什么不人工直接打分？
>
> 因为打分是主观的需要归一化，而排序一般大家会有共同的结论：对同一个问题，A和B哪个回答更好。
>
> 2. 有了一组一组的偏序（A>B, A>C, C>B）怎么得到每个回答的奖励分数？
>
> 这一步在Hug的博客里用了Elo排名系统，打网游排位赛、看足球篮球比赛的可能都知道：
>
> 3. 这个RM用什么模型？
>
> 只要用Elo系统打分后归一化，然后直接上个LM做回归就行，可以从零训练也可以用老LM做finetune。这里有个有趣的事情在于，做问答和做评分都需要输入所有的文本，实际上两个模型的容量（或者说理解能力）应该是差不多的，而现有的RLHF模型都使用了两个不同大小的模型。
>
> 4. 有没有其他方式训练打分的模型？
>
> 可以对偏序直接用 Pairwise learning to rank 做打分，更符合常规的思路。

### Step3: 强化学习策略优化

RLHF 的最后一步是确定如何使用奖励模型来更新 AI 智能体的策略。用于更新 RL 模型的奖励函数的最成功算法之一是近端策略优化 (PPO)。

首先定义强化学习的场景：

- 策略（Policy）： 基于该语言模型，接收Prompt作为输入，然后输出一系列文本（或文本的概率分布）。
- 动作空间（action space）：词表所有token在所有输出位置的排列组合（单个位置通常有50k左右的token候选）。
- 观测空间 （Observation Space）：是输入文本序列的空间（全词表大小 x 序列长度）；
- 奖励函数 （Reward function ）：基于奖励模型计算得到初始reward，再叠加上一个约束项。具体而言，把问题分别输入第一步finetune的模型和正在训练的模型得到输出 $y_1$ , $y_2$ ，把 $y_2$ 输入RM得到评分 $r_\theta$ ，然后这里我们期望 $y_1$ , $y_2$  别差太多, 所以加一个KL散度的惩罚项  $r_{KL}$，即：$r = r_{\theta} - \lambda r_{KL}$.

然后根据PPO算法进行RL更新。

首先，创建初始模型的副本，并冻结其可训练权重。PPO 算法会计算出一个范围 [1- ε , 1+ ε ]，其中 ε 是一个超参数，它大致决定了新的（更新后的）策略可以偏离旧的（已冻结的）策略的程度。然后，算法会计算概率比：旧策略采取给定操作的概率与新策略采取该操作的概率之比。如果概率比大于 1+ ε（或小于 1- ε），则策略更新的幅度可能会被裁剪，以防止剧烈变化导致整个模型不稳定。



## RLHF 算法框架

### PPO

什么是近端策略优化（PPO）？

近端策略优化（PPO） 是一种强化学习算法，用于训练语言模型和其他机器学习模型。它旨在优化代理（在本例中为语言模型）的策略函数，以最大化其在给定环境中的预期累积奖励。PPO 以其训练复杂模型的稳定性和效率而闻名。
以下是 PPO 对于语言模型的工作原理：

- 策略和价值函数： PPO 涉及两个关键组成部分：策略函数（通常由神经网络表示）和价值函数。策略函数根据输入数据定义模型的操作或决策，而价值函数则估计遵循特定策略的预期累积奖励。
- 策略迭代： PPO遵循策略迭代方法。它从初始策略开始，并迭代地完善它以提高性能。在每次迭代期间，模型通过与环境交互来收集数据。对于语言模型，这种交互可能涉及根据输入提示生成文本。
- 目标函数： PPO旨在通过最大化目标函数来优化策略。该函数结合了两个关键术语：代理目标和正则化项。替代目标使用当前迭代期间收集的数据来衡量新策略与旧策略相比的执行情况。正规化术语阻止政策发生太大变化。
- 裁剪： PPO 的显着特点之一是使用裁剪来确保策略更新不会过于极端。裁剪将策略更新限制在一定范围内，防止策略发生较大变化而导致训练过程中的不稳定。
- 多个 Epoch： PPO 通常在每次迭代期间进行多个优化 epoch。在每个时期，它使用收集的数据来更新策略。重复此过程直到找到令人满意的策略。
- 策略评估： 价值函数在策略评估中起着至关重要的作用。它估计了遵循当前策略的预期回报。这一估计有助于评估策略的质量并指导其完善。
- 稳定性和样品效率： PPO因其稳定性和样品效率而受到青睐。与其他一些强化学习算法相比，它往往提供更平滑的策略更新，使其适合训练文本生成质量至关重要的语言模型。

### DPO（Direct Preference Optimization）

直接偏好优化 (DPO) 是一种微调大型语言模型 (LLM)以符合人类偏好的新颖方法。与涉及来自人类反馈的复杂强化学习 (RLHF) 的传统方法不同， DPO简化了流程。它的工作原理是创建人类偏好对的数据集，每个偏好对都包含一个提示和两种可能的完成方式——一种是首选，一种是不受欢迎。然后对LLM进行微调，以最大限度地提高生成首选完成的可能性，并最大限度地减少生成不受欢迎的完成的可能性。 与 RLHF 相比，DPO 具有多项优势：

- 简单性： DPO更容易实施和培训，使其更易于使用。
- 稳定性： 不易陷入局部最优，保证训练过程更加可靠。
- 效率：与 RLHF 相比， DPO 需要更少的计算资源和数据，使其计算量轻。
- 有效性： 实验结果表明，DPO在情感控制、摘要和对话生成等任务中可以优于 RLHF 。

>  DPO 的主要特性包括作为单阶段算法、对超参数变化的鲁棒性、效率以及跨各种自然语言处理任务的有效性。如果您的目标是微调 LLM 以满足特定的人类偏好，DPO 可以提供比 RLHF 更简单、更高效的替代方案。

直接偏好优化 (DPO) 和人类反馈强化学习 (RLHF)是两种不同的方法，用于微调大型语言模型 (LLM)以符合人类偏好。

##### 方法

DPO：DPO是一种单阶段算法，可直接优化 LLM以生成首选响应。它将问题表述为使用人类偏好对数据集的分类任务，其中每一对都包含一个提示和两个可能的完成（一个首选，一个不首选）。DPO 最大化生成首选完成的概率并最小化生成非首选完成的概率。它不涉及多轮训练。 RLHF：RLHF 是一个两阶段过程。首先，它符合反映人类偏好的奖励模型。然后，它使用强化学习对LLM 进行微调，以最大化估计奖励，同时保持与原始模型的一致性。RLHF 涉及多轮训练，并且计算量可能很大。

##### 复杂

DPO：与RLHF相比， DPO更易于实施和培训。它不需要创建单独的奖励模型、在微调期间从 LLM 采样或进行广泛的超参数调整。RLHF：由于奖励模型拟合和微调的两阶段过程， RLHF更加复杂，并且计算要求较高。

##### 稳定

DPO：DPO对超参数的变化更加稳定和鲁棒。在训练过程中陷入局部最优的可能性较小。 RLHF：RLHF对超参数选择很敏感，可能需要仔细调整以避免不稳定。

##### 效率

DPO：与RLHF相比， DPO 在计算和数据需求方面更加高效。它可以用更少的资源实现类似或更好的结果。 RLHF：RLHF 可能需要更多的计算资源和大量的数据才能获得类似的结果。

##### 能力

DPO：DPO 已被证明在各种任务中都很有效，包括情绪控制、摘要和对话生成。在一些研究中它的表现优于 RLHF。 RLHF：RLHF在使法学硕士与人类偏好保持一致方面也很有效，但可能需要更广泛的实验和调整。

### Natural Languge Policy Optimization （NLPO）



## RLHF 开源算法库

### OpenRLHF

OpenRLHF 是一个基于 Ray、DeepSpeed 和 HF Transformers 构建的高性能 RLHF 框架：

- 简单易用: OpenRLHF 是目前可用的最简单的高性能 RLHF 库之一，兼容 Huggingface 模型和数据集。
- 高性能: RLHF 训练的 80% 时间花费在样本生成阶段。得益于使用 Ray 和 Adam Offload（固定内存）可以使用大批量推理，使用 13B LLaMA2 模型的 OpenRLHF 性能是 DeepSpeedChat 的 4 倍。我们还支持 vLLM 生成加速以进一步提高生成性能。
- 分布式 RLHF: OpenRLHF 使用 Ray 将 Actor、Reward、Reference 和 Critic 模型分布到不同的 GPU 上，同时将 Adam 优化器放在 CPU 上。这使得使用多个 A100 80G GPU 和 vLLM 可以全面微调超过 70B+ 的模型 (见 [architecture](https://github.com/OpenLLMAI/OpenRLHF/blob/main/docs/ray_architecture.png)) 以及在多个 24GB RTX 4090 GPU 上微调 7B 模型。

### TRL

通常是实现所有新算法的最小实现最快的地方。很多示例通常可以在单个 GPU 上运行。



### TRLX

Github:  https://github.com/CarperAI/trlx

特点：

- 最大支持 330 亿 参数的模型训练
- 支持两种强化学习算法： [近端策略优化 (PPO)](https://openai.com/research/openai-baselines-ppo)和[隐式语言 Q-学习 (ILQL)](https://docs.agilerl.com/en/latest/api/algorithms/ilql.html#:~:text=Implicit Language Q-Learning (ILQL) %23 ILQL is an,(LLMs) with reinforcement leaning from human feedback (RLHF).)。

- TRLX 通过 Accelerate 支持的Trainer与Hugging Face 模型无缝集成

### RL4LM

Github: https://github.com/allenai/RL4LMs

特点：

- OnPolicy 策略算法：实现了PPO、A2C、TRPO 和 NLPO（自然语言策略优化）等策略算法。
- 支持因果LM（例如GPT-2/3）和seq2seq LM（例如T5、BART）的Actor-Critic 策略。
- 基准测试： 通过GRUE（General Reinforced-language Understanding Evaluation）基准测试，提供了一系列经过彻底测试和基准化的实验，涵盖了不同的NLP任务和奖励函数。
  - 支持7种不同的自然语言处理（NLP）任务，包括摘要生成、生成常识推理、基于情感的文本续写、表格到文本生成、抽象问题回答、机器翻译和对话生成
  - 提供20多种不同类型的NLG指标作为奖励函数，例如词汇指标（如ROUGE、BLEU、SacreBLEU、METEOR）、语义指标（如BERTSCORE、BLEURT）和特定任务指标（如PARENT、CIDER、SPICE），来自预训练分类器的分数（例如：情绪分数）
- 开源模块化库，可以方便的对接 Huggingface，模块可定制。

### DeepSpeed Chat

Github: https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-chat

虽然工程设置非常不同，但最好比较实现相同内容的不同方法。



## 开源 RLHF 模型

- [Zephyr ](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta)（导致了[Tulu 2 ](https://huggingface.co/allenai/tulu-2-dpo-70b)、Stability 模型、Intel 模型等）是推动 DPO 和普遍有用的 RLHF 模型激增的火花。
- [Starling](https://starling.cs.berkeley.edu/) 是一款性能出色的最新型号，有趣的是它没有使用DPO。
- [Llama 2 ](https://arxiv.org/abs/2307.09288)在其论文中提供的细节比大多数实验室在 RLHF 方面尝试过的细节还要多。

## 数据集

- [UltraFeedback ](https://arxiv.org/abs/2310.01377)：为我们提供 Zephyr 等人的数据集。甚至还有更多[研究试图改进数据集和 RLHF 性能](https://argilla.io/blog/notus7b/)。
- [Open Assistant 1 ](https://huggingface.co/datasets/OpenAssistant/oasst1)：社区生成的指导数据，带来了开放式 IFT 培训的第一波进展。
- [Alpaca ](https://huggingface.co/datasets/tatsu-lab/alpaca)：第一个流行的合成指令数据。
- [ShareGPT](https://sharegpt.com/)及其变体：人们用来尝试在开放数据中获得类似 ChatGPT 的功能的大型数据集。

## 评价

这三个评估是 RLHF 模型相对排名的综合集合。

- [ChatBotArena ](https://chat.lmsys.org/)：众包比较网站，是开放模型和封闭模型的模型质量的首选来源。
- [MT Bench ](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard)：同样由 LMSYS 构建的两回合聊天评估，与大多数 LLM 的实际评估非常相关。
- [AlpacaEval ](https://github.com/tatsu-lab/alpaca_eval)：第一个 GPT4 作为法官的工具，用于推广 LLM 作为法官的实践。

## RLHF vs SFT

1. 监督学习与强化学习的区别：
   - 在监督学习中，模型通过观察人类编写的文本和期望的输出来学习如何回答问题或执行任务，适用于文本基础任务。
   - 在强化学习中，模型生成自己的答案，然后根据评分机制（如人类）的反馈来学习如何得到高分，适用于知识寻求和创造性任务。
2. 强化学习的挑战：
   - 强化学习比监督学习更难，因为它需要解决“信用分配”问题，即如何将最终得分反馈到生成的每个步骤。
   - 强化学习需要一个评分机制，而在语言任务中，自动评分器很难实现，通常需要人类反馈。
3. 多样性论点：
   - 监督学习可能导致模型复制特定的答案，而人类语言中通常有多种方式传达相同信息。RL训练可以提供更多的多样性。
4. 理论论点：
   - 监督学习只提供正面反馈，而强化学习允许负面反馈，这在理论上更强大，因为它允许模型形成自己的假设并从教师(Reward Model)那里获得反馈。
5. 核心论点：
   - 对于知识寻求型查询，强化学习是必要的，因为监督学习可能会教会模型撒谎。RL训练不鼓励模型撒谎，即使模型最初猜测了一些正确答案，长期来看，它也会因为编造的答案（可能是错误的）而得到低分，并学会依赖其内部知识回答问题或选择不回答。
6. 教学模型放弃：
   - 在模型不知道答案的情况下，我们希望它能放弃并回答“我不知道”。这在监督学习中很难实现，而RL提供了一种可能的解决方案。
7. 模型窃取/蒸馏的影响：
   - 其他模型可能会尝试其他模型的行为，但这种方法可能不适用于知识寻求型查询，因为它可能会导致模型编造事实。
8. 无需人类反馈的RL：
   - 尽管RL训练通常需要人类反馈，但使用大型预训练语言模型作为自动评分器， RL训练变得更加实用。

## RLHF 的局限性

尽管 RLHF 模型在训练 AI 代理执行从机器人、视频游戏到 NLP 等复杂任务方面取得了令人印象深刻的成果，但使用 RLHF 并非没有局限性。

- 人类偏好数据成本高昂。收集第一手人类反馈的需求可能会造成一个代价高昂的瓶颈，限制 RLHF 流程的可扩展性。Anthropic 和 Google 都提出了 AI 反馈强化学习 (RLAIF) 的方法，即让另一个 LLM 评估模型响应来取代部分或全部人类反馈，其结果与 RLHF 相当。

- 人类的输入具有高度主观性。要就“高质量”的输出到底是什么达成坚定的共识，几乎是不可能的，因为人类评估者不仅会对所谓的“事实”产生不同的意见，还会对“适当的”模型行为究竟应该如何理解存在分歧。因此，人类的分歧使得据以判断模型性能的“标准答案”无法形成。

- 人类评估者可能会犯错，甚至故意采取对抗性和恶意行为。人类对模型的引导并不总是出于善意，他们可能怀抱着真诚的反对意见，也可能故意操纵学习过程。Wolf 等人在 2016 年的一篇论文中提出，有毒行为应被视为人机交互中的一个基本预期，并建议需要一种方法来评估人类输入的可信度。2022 年，Meta AI 发布了[一篇关于对抗性人类输入的论文](https://arxiv.org/pdf/2208.03295.pdf)，研究了使用自动化方法“从高质量数据中获得最大学习效率，同时对低质量和对抗性数据具有最大稳健性”。该论文对各种“操纵”行为进行了分类，并确定了它们扭曲反馈数据的不同方式。

- RLHF 存在过度拟合和偏见的风险。如果收集到的反馈来自一个非常有限的群体，那么当模型被其他群体使用，或者被用来处理与评估者持有某些偏见相关的主题时，可能会出现性能问题。

## Reference

- [Fine-Tuning Language Models from Human Preferences](https://arxiv.org/abs/1909.08593) (Zieglar et al. 2019): An early paper that studies the impact of reward learning on four specific tasks.
- [Learning to summarize with human feedback](https://proceedings.neurips.cc/paper/2020/hash/1f89885d556929e98d3ef9b86448f951-Abstract.html) (Stiennon et al., 2020): RLHF applied to the task of summarizing text. Also, [Recursively Summarizing Books with Human Feedback](https://arxiv.org/abs/2109.10862) (OpenAI Alignment Team 2021), follow on work summarizing books.
- [WebGPT: Browser-assisted question-answering with human feedback](https://arxiv.org/abs/2112.09332) (OpenAI, 2021): Using RLHF to train an agent to navigate the web.
- InstructGPT: [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) (OpenAI Alignment Team 2022): RLHF applied to a general language model [[Blog post](https://openai.com/blog/instruction-following/) on InstructGPT].
- GopherCite: [Teaching language models to support answers with verified quotes](https://www.deepmind.com/publications/gophercite-teaching-language-models-to-support-answers-with-verified-quotes) (Menick et al. 2022): Train a LM with RLHF to return answers with specific citations.
- Sparrow: [Improving alignment of dialogue agents via targeted human judgements](https://arxiv.org/abs/2209.14375) (Glaese et al. 2022): Fine-tuning a dialogue agent with RLHF
- [ChatGPT: Optimizing Language Models for Dialogue](https://openai.com/blog/chatgpt/) (OpenAI 2022): Training a LM with RLHF for suitable use as an all-purpose chat bot.
- [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760) (Gao et al. 2022): studies the scaling properties of the learned preference model in RLHF.
- [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862) (Anthropic, 2022): A detailed documentation of training a LM assistant with RLHF.
- [Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned](https://arxiv.org/abs/2209.07858) (Ganguli et al. 2022): A detailed documentation of efforts to “discover, measure, and attempt to reduce [language models] potentially harmful outputs.”
- [Dynamic Planning in Open-Ended Dialogue using Reinforcement Learning](https://arxiv.org/abs/2208.02294) (Cohen at al. 2022): Using RL to enhance the conversational skill of an open-ended dialogue agent.
- [Is Reinforcement Learning (Not) for Natural Language Processing?: Benchmarks, Baselines, and Building Blocks for Natural Language Policy Optimization](https://arxiv.org/abs/2210.01241) (Ramamurthy and Ammanabrolu et al. 2022): Discusses the design space of open-source tools in RLHF and proposes a new algorithm NLPO (Natural Language Policy Optimization) as an alternative to PPO.
- [Llama 2](https://arxiv.org/abs/2307.09288) (Touvron et al. 2023): Impactful open-access model with substantial RLHF details.
