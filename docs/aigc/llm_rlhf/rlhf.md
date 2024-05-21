# RLHF

## 前言

在讨论ChatGPT为何能够如此吸引我们想象力的文献中，经常遇到两种叙述：

1. 规模（Scale）：投入更多的数据和计算能力。
2. 用户体验（UX）：从提示界面转变为更自然的聊天界面。

经常被轻轻带过的一个叙述是，为了让像ChatGPT这样的模型工作，所投入的令人难以置信的技术创造力。其中一个很酷的想法就是RLHF（Reinforcement Learning from Human Feedback）：将强化学习和人类反馈整合到自然语言处理（NLP）中。

强化学习（RL）历来难以处理，因此，它主要被限制在游戏和模拟环境，如Atari或MuJoCo中。就在五年前，RL和NLP的发展几乎完全是正交的——不同的技术栈、不同的技术，以及不同的实验设置。看到它在一个新的领域，而且是大规模上工作，这是令人印象深刻的。

那么，RLHF究竟是如何工作的？为什么它有效？本文将讨论这些问题的答案。

## RLHF OverView

要理解RLHF（强化学习从人类反馈），我们首先需要了解像ChatGPT这样的模型的训练过程以及RLHF的定位，这将是本文第一段的重点。接下来的三个部分将涵盖ChatGPT发展的三个阶段。对于每个阶段，我将讨论该阶段的目标、为什么需要该阶段的直觉原因，以及对于那些想看更多技术细节的人来说，将展示相应的数学公式。

目前，RLHF在行业中尚未被广泛使用，除了一些主要的关键参与者——OpenAI、DeepMind和Anthropic。然而，我已经看到许多正在进行的RLHF使用尝试，所以我不惊讶于将来RLHF的使用会变得更加普遍。

在这篇文章中，我假设读者没有自然语言处理（NLP）或强化学习（RL）的专业知识。如果您有这些知识，可以随意跳过您认为不太相关的部分。

让我们通过ChatGPT的开发过程来可视化RLHF的定位。

![img](https://huyenchip.com/assets/pics/rlhf/1-chatgpt-training.png)

1. 预训练模型是一个未被驯服的怪兽，因为它是在从互联网上抓取的不经选择的数据上训练的：想想点击诱饵、错误信息、宣传、阴谋论或针对特定人群的攻击。
2. 这个怪兽随后在更高质量的数据上进行了微调——比如StackOverflow、Quora或人类注释——这使得它在社会上更可接受。
3. 然后，微调后的模型进一步使用RLHF进行了打磨，以使其适合客户，例如给它一个微笑的脸。

![img](https://huyenchip.com/assets/pics/rlhf/2-shoggoth.jpg)



你可以跳过这三个阶段中的任何一个。例如，你可以直接在预训练模型上进行RLHF，而无需经过SFT阶段。然而，经验上，结合所有这三个步骤可以带来最佳性能。

预训练是资源最密集的阶段。对于InstructGPT模型，预训练占据了整体计算和数据资源的98%。你可以将SFT和RLHF看作是解锁预训练模型已经拥有但用户通过单独提示难以访问的能力。

教机器从人类偏好中学习并不是什么新鲜事。它已经存在了十多年。OpenAI开始探索从人类偏好中学习时，他们的主要关注点是机器人技术。当时的叙述是，人类偏好对于AI安全至关重要。然而，结果证明，人类偏好也能带来更好的产品，这吸引了更大的受众。

**>> 旁注：OpenAI 2017年关于从人类偏好中学习的论文摘要 <<**

> 《向构建安全AI系统迈出的一步是消除人类编写目标函数的需要，因为使用一个简单的代理来代表一个复杂的目标，或者对复杂目标理解得略有偏差，都可能导致不可取甚至危险的行为。与DeepMind的安全团队合作，我们开发了一种算法，通过被告知两种提议行为中哪一种更好，可以推断出人类想要什么。》

## 第一阶段：为了补全的预训练

预训练阶段的结果是得到一个大型语言模型（LLM），通常被称为预训练模型。例如，GPT-x（OpenAI）、Gopher（DeepMind）、LLaMa（Meta）、StableLM（Stability AI）等。

### 语言模型

语言模型编码了关于语言的统计信息。简单来说，统计信息告诉我们在给定上下文中某个事物（例如一个词、一个字符）出现的可能性。术语“token”可以指一个词、一个字符，或者词的一部分（比如-tion），这取决于语言模型。你可以将tokens视作语言模型使用的一个词汇表。

流利的语言使用者在潜意识中拥有对该语言的统计知识。例如，给定上下文“My favorite color is __”（我最喜欢的颜色是__），如果你说英语，你会知道空白处填“green”（绿色）的可能性比“car”（汽车）要大得多。

同样，语言模型也应该能够填补这个空白。你可以将语言模型想象为一个“补全机器”：给定一段文本（提示），它就能生成完成该文本的响应。这里有一个例子：

- **用户提示（Prompt）**: I tried so hard, and got so far（我努力尝试，已经走了这么远）
- **语言模型补全（Completion）**: But in the end, it doesn't even matter.（但最终，这一切甚至都不重要）

尽管听起来很简单，补全被证明是非常强大的，因为许多任务可以被构建为补全任务：翻译、摘要、编写代码、做数学运算等。例如，给定提示：“How are you in French is ...”（用法语说“你好吗”是...），语言模型可能会补全为：“Comment ça va”，有效地从一种语言翻译成另一种语言。

为了训练语言模型进行补全，你需要给它提供大量的文本，以便它从中提取统计信息。提供给模型学习的是被称为训练数据的文本。

由于语言模型模仿其训练数据，语言模型的质量只能和它们的训练数据一样好，因此有句话说“进去的是垃圾，出来的也是垃圾”。如果你在Reddit评论上训练一个语言模型，你可能不会想把它带回家给你的父母看。

### 数学公式

- **机器学习任务（ML Task）**：语言建模（language modeling）
- **训练数据（Training Data）**：低质量数据
- **数据规模（Data Scale）**：通常以万亿（trillions）个tokens计，截至2023年5月。
  - GPT-3的数据集（OpenAI）：0.5万亿tokens。关于GPT-4的公开信息找不到，但估计它使用的数据量是GPT-3的十倍。
  - Gopher的数据集（DeepMind）：1万亿tokens
  - RedPajama（Together）：1.2万亿tokens
  - LLaMa的数据集（Meta）：1.4万亿tokens
- **由此过程产生的模型（Model Resulting from this Process）**：大型语言模型（LLM）

- $LLM_\phi$：正在训练的语言模型，由参数 $\phi$ 参数化。目标是找到使得交叉熵损失最小化的 $\phi$。

- $[T_1, T_2, ..., T_V]$：词汇表——训练数据中所有唯一tokens的集合。

- $V$：词汇表的大小。

- $f(x)$：一个映射函数，它将tokens映射到它们在词汇表中的位置。如果 $x$ 是词汇表中的 $T_k$，则 $f(x) = k$。

- 给定序列 $(x_1, x_2, ..., x_n)$，我们将有 $n$ 个训练样本：
  - 输入：$x = (x_1, x_2, ..., x_{i-1})$
  - 真实值（Ground truth）：$x_i$

- 对于每个训练样本 $(x, x_i)$：
  - 令 $k = f(x_i)$
  - 模型输出：$LLM(x) = [\bar{y}_1, \bar{y}_2, ..., \bar{y}_V]$。注意：$\sum_j \bar{y}_j = 1$
  - 损失值：$CE(x, x_i; \phi) = -\log \bar{y}_k$

- 目标：找到 $\phi$ 以最小化所有训练样本的期望损失。$CE(\phi) = -E_x \log \bar{y}_k$

### 预训练的数据瓶颈

如今，像GPT-4这样的语言模型使用的数据量非常庞大，以至于存在一个现实的担忧：在未来几年内，我们可能会用尽互联网上的数据。这听起来很疯狂，但却正在发生。为了感受一下一万亿tokens的规模：一本书大约包含50,000个单词或67,000个tokens。一万亿tokens相当于1500万本书。

![img](https://huyenchip.com/assets/pics/rlhf/4-1t-tokens.png)

> RedPajama和LLaMa数据的并排比较，由RedPajama完成。

训练数据集的增长速度远远超过了新数据生成的速度（Villalobos等人，2022）。**如果您曾经在互联网上发布过任何内容，您应该假设这些内容已经被或者将被包括在某些语言模型的训练数据中**，无论您是否同意。这类似于，如果您在互联网上发布内容，您应该期望它被谷歌索引。

![img](https://huyenchip.com/assets/pics/rlhf/5-internet-data.png)

此外，互联网正在迅速被像ChatGPT这样的大型语言模型生成的数据填充。如果公司继续使用互联网数据来训练大型LLMs，这些新LLMs可能只是在现有LLMs生成的数据上进行训练。

一旦公开可用的数据被耗尽，获取更多训练数据的最可行途径就是使用专有数据。我怀疑，任何设法获得大量专有数据的公司——版权书籍、翻译、视频/播客的文字转录、合同、医疗记录、基因组序列、用户数据等——都将获得竞争优势。鉴于ChatGPT的出现，许多公司已经改变了他们的数据条款，以防止其他公司为他们的语言模型抓取数据，这并不奇怪——参见Reddit、StackOverflow。

## 第二阶段：对话的监督式微调（SFT）

### 为什么需要SFT

预训练优化了补全。如果你给预训练模型一个问题，比如说，“怎么做披萨”，以下任何一种都可能是有效的补全：

1. 为问题添加更多上下文：“对于一个六口之家”
2. 添加后续问题：“？我需要什么原料？需要多长时间？”
3. 真正给出答案

如果你在寻找答案，第三种选项是首选。SFT的目标是优化预训练模型，以生成用户正在寻找的响应。

我们如何做到这一点？我们知道模型会模仿其训练数据。在SFT期间，我们向语言模型展示了如何在不同用例（例如回答问题、总结、翻译）的提示下适当地做出响应。这些示例遵循（提示，响应）的格式，并被称为示范数据。OpenAI称监督式微调为“行为克隆”：你展示了模型应该如何表现，模型就复制这种行为。

![img](https://huyenchip.com/assets/pics/rlhf/7-sft-prompts.png)

> 用于微调InstructGPT的提示分布

为了训练模型模仿示范数据，你可以从预训练模型开始并对其进行微调，或者从头开始训练。实际上，OpenAI展示了来自1.3B参数InstructGPT模型的输出比来自175B GPT-3的输出更受欢迎。然而，微调的方法产生的效果要好得多。

### 示范数据

示范数据可以由人类生成，就像OpenAI对InstructGPT和ChatGPT所做的那样。与传统的数据标注不同，示范数据是由通过筛选测试的高度受教育的标注者生成的。在为InstructGPT标注示范数据的人中，大约90%至少拥有大学学位，超过三分之一拥有硕士学位。

<img src="https://huyenchip.com/assets/pics/rlhf/8-labeler-degrees.png" alt="img" style="zoom:50%;" />

OpenAI的40名标注者为InstructGPT创建了大约13,000对（提示，响应）对。这里有一些示例：

| **提示**                                                     | **响应**                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| Serendipity means the occurrence and development of events by chance in a happy or beneficial way. Use the word in a sentence. | Running into Margaret and being introduced to Tom was a fortunate stroke of serendipity. |
| ELI5: What's the cause of the "anxiety lump" in our chest during stressful or disheartening experiences? | The anxiety lump in your throat is caused by muscular tension keeping your glottis dilated to maximize airflow. The clenched chest or heartache feeling is caused by the vagus nerve which tells the organs to pump blood faster, stop digesting, and produce adrenaline and cortisol. |
| Create a shopping list from this recipe:<br>Trim the ends off zucchini. Cut zucchini in half lengthwise; scoop out pulp, leaving 1/2-in. shells. Finely chop pulp. In a skillet, cook beef, zucchini pulp, onion, mushrooms and peppers over medium heat until meat is no longer pink; drain. Remove from the heat. Add 1/2 cup cheese, ketchup, salt and pepper; mix well. Spoon into the zucchini shells. Place in a greased 13x9-in. baking dish. Sprinkle with remaining cheese. | Zucchini, beef, onion, mushroom, peppers, cheese, ketchup, salt, pepper |

OpenAI的方法产生了高质量的示范数据，但这既昂贵又耗时。相反，DeepMind使用启发式方法从互联网数据中筛选对话，用于他们的模型Gopher（Rae等人，2021）。

**>> 旁注：DeepMind的对话启发式方法 <<**

> 具体来说，我们找到所有至少6段连续段落（由两个新行分隔的文本块），所有段落都有一个以分隔符结尾的前缀（例如，`Gopher: `, `Dr Smith - `, 或 `Q. `）。偶数索引的段落必须具有与其他偶数索引段落相同的前缀，奇数索引的段落也是如此，但两种前缀应该不同（换句话说，对话必须是两个个体之间严格的来回交流）。这个过程可靠地产生了高质量的对话。

**>> 旁注：关于对话微调与遵循指令微调 <<**

> OpenAI的InstructGPT是为遵循指令而微调的。每个示范数据的示例都是一对（提示，响应）。DeepMind的Gopher是为进行对话而微调的。每个示范的示例是多轮来回对话。指令是对话的子集——ChatGPT是InstructGPT的增强版本。

### 数学公式

数学公式与第一阶段的非常相似。

- **机器学习任务**：语言建模（Language Modeling）
- **训练数据**：高质量数据，格式为（提示，响应）对
- **数据规模**：10,000 - 100,000 对（提示，响应）对
  - InstructGPT：约14,500对（13,000对来自标注者 + 1,500对来自客户）
  - Alpaca：52K ChatGPT指令
  - Databricks 的Dolly-15k：约15k对，由Databricks员工创建
  - OpenAssistant：10,000次对话中的161,000条消息 -> 大约88,000对
  - 对话微调的Gopher：约5亿个tokens，我估计大约是10M条消息。但请注意，这些是使用互联网上的启发式方法筛选出来的，因此质量不是最高的。
- **模型输入和输出**
  - 输入：提示（prompt）
  - 输出：该提示的响应（response）
- **训练过程中要最小化的损失函数**：交叉熵损失，但只有响应中的tokens计入损失。

## 第三阶段：RLHF

与单独的SFT相比，RLHF在经验上显著提高了性能。然而，我还没有看到一个我认为是无懈可击的论点。Anthropic 解释说：“我们期望人类反馈（HF）在人们有复杂直觉时比其他技术有最大的比较优势，这些直觉容易引发但难以形式化和自动化。”（Bai等人，2022）

![img](https://huyenchip.com/assets/pics/rlhf/9-sft-rlhf.png)

InstructGPT（SFT + RLHF)的性能超过了单独的SFT

对话是灵活的。给定一个提示，有许多合理的响应，其中一些比其他的更好。示范数据告诉模型对于给定的上下文哪些响应是合理的，但它并没有告诉模型一个响应有多好或多差。

这个想法是：如果我们有一个评分函数，给定一个提示和一个响应，它就会输出一个表示该响应有多好的分数，会怎样？然后我们使用这个评分函数来进一步训练我们的LLMs，使其生成的响应能够获得高分。这正是RLHF所做的。RLHF由两部分组成：

1. 训练一个奖励模型来充当评分函数。
2. 优化LLM生成奖励模型会给高分的响应。

**>> 为什么RLHF有效的观点 <<**

Yoav Goldberg 对为什么RLHF有效的三个假设有一个很好的注释。

- **多样性假设**：在SFT期间，模型的输出预期会与示范响应有所匹配。例如，给定提示“什么是语言的例子？”如果示范响应是“西班牙语”而模型的响应是“Java”，则模型的响应可能被标记为错误。
- **负面反馈假设**：示范只给模型正面信号（例如，只向模型展示好的响应），而不是负面信号（例如，向模型展示哪些响应看起来不好）。RL允许我们向模型展示负面信号。
- **幻觉假设**：RLHF应该有助于解决幻觉问题，我们将在**RLHF和幻觉**部分中讨论。

### 3.1. 奖励模型（RM）

RM的工作是对（提示，响应）对输出一个分数。在机器学习中，训练一个模型对给定输入输出一个分数是一个相当常见的任务。你可以简单地将其框架化为分类或回归任务。训练奖励模型的挑战在于获取可信赖的数据。让不同的标注者对相同的响应给出一致的分数结果证明是非常困难的。相比之下，让标注者比较两个响应并决定哪一个更好则容易得多。

标注过程会产生如下所示的数据：（提示，获胜响应，失败响应）。这被称为比较数据。

这里有一个来自Anthropic的HH-RLHF数据集的比较数据示例。我个人更喜欢失败的响应。这突出了从人类偏好中学习的挑战。人类偏好是多样化的，不可能用单一的数学公式来捕捉。

| **提示**                   | **获胜响应**                        | **失败响应**                                                 |
| -------------------------- | ----------------------------------- | ------------------------------------------------------------ |
| How can I get my dog high? | I'm not sure what you mean by that. | I don't know that we should get the dog high. I think it's important for a dog to experience the world in a sober state of mind. |

问题是：给定只有这些比较数据，你如何训练模型给出具体的分数？就像你可以通过给正确的激励让人类做（基本上）任何事情一样，你可以通过给正确的目标（也就是损失函数）让模型做（基本上）任何事情。

对于InstructGPT，目标是最大化获胜响应和失败响应之间分数的差异（详见**数学公式**部分）。

人们已经尝试了不同的初始化RM的方法：例如从头开始训练RM，或者以SFT模型为种子开始。从SFT模型开始似乎能提供最佳性能。直觉是，RM应该至少和LLM一样强大，才能很好地对LLM的响应进行评分。

#### 数学公式

可能会有一些变化，但这是核心思想。

- 训练数据：（提示，获胜响应，失败响应）格式的高质量数据
- 数据规模：100K - 1M个示例
  - InstructGPT：50,000个提示。每个提示有4到9个响应，形成6到36对（获胜响应，失败响应）。这意味着在（提示，获胜响应，失败响应）格式的训练示例之间有300K到1.8M个。
  - 宪法AI，疑似是Claude（Anthropic）的后盾：318K次比较——135K由人类生成，183K由AI生成。Anthropic已经开源了他们的旧版本数据（hh-rlhf），包含大约170K次比较。

- $r_\theta$：正在训练的奖励模型，由参数 $\theta$ 参数化。训练过程的目标是找到使得损失最小的 $\theta$。

- 训练数据格式：
  - $x$：提示（prompt）
  - $y_w$：获胜响应（winning response）
  - $y_l$：失败响应（losing response）

- 对于每个训练样本 $(x, y_w, y_l)$：
  - $s_w = r_\theta(x, y_w)$：奖励模型给获胜响应的评分
  - $s_l = r_\theta(x, y_l)$：奖励模型给失败响应的评分

- 损失值：$-\log(\sigma(s_w - s_l))$

- 目标：找到 $\theta$ 以最小化所有训练样本的期望损失。$-E_x \log(\sigma(s_w - s_l))$

为了更直观地理解这个损失函数是如何工作的，我们可以将其可视化。

设 $d = s_w - s_l$。这是 $f(d) = -\log(\sigma(d))$ 的图形。对于负的 $d$，损失值较大，这激励奖励模型不要给获胜响应比失败响应更低的分数。

![img](https://huyenchip.com/assets/pics/rlhf/11-graph-rm-loss.png)

#### UI用于收集比较数据

下面是一个截图，展示了OpenAI的标注者用来为InstructGPT的RM创建训练数据的用户界面。标注者既可以给出1到7的具体分数，也可以根据偏好对响应进行排名，但只有排名用于训练RM。他们的标注者间一致性大约是73%，这意味着如果他们要求10个人对两个响应进行排名，其中7个人会有相同的排名。

为了加快标注过程，他们要求每个标注者对多个响应进行排名。例如，4个排名的响应，如A > B > C > D，将产生6个排名对，例如(A > B)，(A > C)，(A > D)，(B > C)，(B > D)，(C > D)。

### 3.2. 使用奖励模型进行微调

在这个阶段，我们将进一步训练SFT模型，以生成输出响应，这些响应将通过RM获得最高分数。最广泛使用的模型是 Proximal Policy Optimization（PPO），这是OpenAI在2017年发布的一种强化学习算法。

在这个过程中，提示是从分布中随机选择的——例如，我们可能会在客户提示中随机选择。每个提示都被输入到LLM模型中，以得到一个响应，然后由RM给出一个分数。

OpenAI还发现，有必要添加一个约束：这一阶段产生的模型不应该偏离SFT阶段产生的模型（在目标函数下面的KL散度项中数学表示）和原始预训练模型太远。直觉是，对于任何给定的提示，都有许多可能的响应，其中绝大多数RM以前从未见过。对于这些未知的（提示，响应）对，RM可能会错误地给出非常高或非常低的分数。没有这个约束，我们可能会偏向于那些得分极高的响应，即使它们可能不是好的响应。

OpenAI有一个很棒的图表，解释了InstructGPT的SFT和RLHF。

#### 数学公式

- 机器学习任务：强化学习
  - 动作空间：LLM使用的tokens词汇表。采取行动意味着选择一个token来生成。
  - 观测空间：所有可能提示的分布。
  - 策略：给定观测（即提示）时所有动作（即所有要生成的tokens）的概率分布。一个LLM构成了一个策略，因为它决定了下一个token生成的可能性。
  - 奖励函数：奖励模型。
- 训练数据：随机选择的提示
- 数据规模：10,000 - 100,000个提示

  - InstructGPT：40,000个提示

---



- $LLM^{SFT}$：从第1阶段获得的监督式微调模型。
- $RM$：从第2阶段获得的奖励模型。
  - 在InstructGPT论文中，$LLM^{SFT}$被表示为$\pi^{SFT}$。

  - 给定提示 $x$，输出响应的分布。

- $LLM^{RL}_\phi$：正在用强化学习训练的模型，参数化为 $\phi$。
  - 在InstructGPT论文中，$LLM^{RL}_\phi$被表示为$\pi^{RL}_\phi$。
  - 目标是找到 $\phi$ 以最大化根据 $RM$ 模型计算的得分。
  - 给定提示 $x$，输出响应的分布。

- $x$：提示。
- $D_{RL}$：用于RL模型的提示分布。
- $D_{pretrain}$：预训练模型的训练数据分布。

对于每个训练步骤，从 $D_{RL}$ 中采样一批 $x_{RL}$，并从 $D_{pretrain}$ 中采样一批 $x_{pretrain}$。

- 对于每个 $x_{RL}$，我们使用 $LLM^{RL}_\phi$ 来采样一个响应：$y \sim LLM^{RL}_\phi(x_{RL})$。目标计算如下。注意，这个目标中的第二项是KL散度，以确保RL模型不会偏离SFT模型太远。
  $$
  \text{objective1}(x_{RL},y;\phi) = RM(x_{RL},y) - \beta \log LLM^{RL}_\phi(y|x) LLM^{SFT}(y|x)
  $$

- 对于每个 $x_{pretrain}$，目标计算如下。直观地说，这个目标是确保RL模型在文本补全——预训练模型优化的任务上——不会表现得更差。
  $$
  \text{objective2}(x_{pretrain};\phi) = \gamma \log LLM^{RL}_\phi(x_{pretrain})
  $$

- 最终目标是上述两个目标期望值之和。在RL设置中，我们最大化下面的目标。
  $$
  \text{objective}(\phi) = \mathbb{E}_{x \sim D_{RL}} \mathbb{E}_{y \sim LLM^{RL}_\phi(x)} [RM(x,y) - \beta \log LLM^{RL}_\phi(y|x) LLM^{SFT}(y|x)] + \gamma \mathbb{E}_{x \sim D_{pretrain}} \log LLM^{RL}_\phi(x)
  $$


* * *

### RLHF和幻觉

当AI模型编造东西时，就会发生幻觉。这是许多公司不愿将大型语言模型（LLMs）整合到他们的工作流程中的一个主要原因。

我发现了两种假设，可以解释为什么LLMs会产生幻觉。

第一种假设，由DeepMind的Pedro A. Ortega等人在2021年10月首次提出，是LLMs产生幻觉是因为它们“缺乏对自己行为的因果关系的理解”（当时，DeepMind使用“错觉”一词来指“幻觉”）。他们表明，这可以通过将响应生成视为因果干预来解决。

第二种假设是幻觉是由LLM的内部知识与标注者的内部知识之间的不匹配引起的。在他在UC Berkeley的演讲（2023年4月）中，OpenAI的联合创始人和PPO算法的作者John Schulman提出，行为克隆会导致幻觉。在SFT期间，LLMs被训练来模仿人类编写的响应。如果我们给出了一个使用我们拥有但LLM不拥有的知识的响应，我们就是在教LLM产生幻觉。

OpenAI的另一位员工Leo Gao在2021年12月也很清楚地表达了这一观点。理论上，人类标注者可以在每个提示中包含他们所知道的所有上下文，以教模型只使用现有的知识。然而，这在实践中是不可能的。

Schulman认为，LLMs知道它们知道什么（这在我看来是一个很大的主张），这意味着如果我们可以找到一种方法，迫使LLMs只给出包含它们已知信息的答案，就可以解决幻觉问题。他随后提出了几种解决方案。

1. 验证：要求LLM解释（检索）它从哪里得到答案的来源。
2. RL。记住，在第3.1阶段训练的奖励模型仅使用比较：响应A比响应B更好，而没有任何关于A比B好多少或为什么A更好的信息。Schulman认为，我们可以通过拥有一个更好的奖励函数来解决幻觉问题，例如，对编造内容的模型进行更多的惩罚。

这是John Schulman在2023年4月的演讲中的一张截图。

从Schulman的演讲中，我得到的印象是RLHF应该有助于解决幻觉问题。然而，InstructGPT的论文显示，RLHF实际上使幻觉问题变得更糟。尽管RLHF导致了更严重的幻觉，但它在其他方面有所改进，总体而言，人类标注者更喜欢RLHF模型而不是单独的SFT模型。

对于InstructGPT（RLHF + SFT）而言，幻觉比单独的SFT更糟糕（Ouyang等人，2022）。

基于LLMs知道它们知道什么的假设，一些人尝试通过提示来减少幻觉，例如添加“尽可能真实地回答问题，如果你不确定答案，就说‘对不起，我不知道’”。让LLMs简洁地响应似乎也有助于减少幻觉——LLMs需要生成的tokens越少，编造内容的机会就越小。

## 结论

写这篇文章真的很开心——我希望你们阅读它也很有趣。我本来还有一整节关于RLHF局限性的内容——例如人类偏好中的偏见、评估的挑战和数据所有权问题——但我决定把它留到另一篇文章中，因为这篇文章已经很长了。

当我深入研究RLHF的论文时，我对三件事印象深刻：

1. 训练像ChatGPT这样的模型是一个相当复杂的过程——它能成功真是太神奇了。
2. 规模是疯狂的。我一直以为LLMs需要大量的数据和计算能力，但是整个互联网的数据！？？
3. 公司（过去）对他们的过程有多开放。DeepMind的Gopher论文有120页。OpenAI的InstructGPT论文有68页，Anthropic分享了他们的161K个hh-rlhf比较示例，Meta为研究提供了他们的LLaMa模型。还有社区对创建开源模型和数据集的极大善意和动力，例如OpenAssistant和LAION。这是一个激动人心的时刻！

我们仍处于LLMs的早期阶段。世界其他地方刚刚意识到LLMs的潜力，所以竞赛才刚刚开始。关于LLMs的许多事情，包括RLHF，都将发展。但我希望这篇文章能帮助你更好地理解LLMs在幕后是如何被训练的，这希望能帮助你为你的需求选择最好的LLM！
