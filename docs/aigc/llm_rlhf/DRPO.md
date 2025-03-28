# 深入理解 R1-Zero 类训练：一个批判性视角

## **摘要**
DeepSeek-R1-Zero 展示了大规模强化学习（RL）可以直接增强大型语言模型（LLMs）的推理能力，而无需依赖监督微调。在本工作中，我们通过分析其两个核心组成部分：基础模型和强化学习，来批判性地审视 R1-Zero 类训练。我们研究了多种基础模型，包括 DeepSeek-V3-Base，以了解预训练特征如何影响强化学习性能。我们的分析表明，DeepSeek-V3-Base 已经展现出“顿悟时刻”，而 Qwen2.5 基础模型即使没有提示模板也表现出强大的推理能力，暗示了潜在的预训练偏差。此外，我们发现 Group Relative Policy Optimization（GRPO）中存在优化偏差，这种偏差在训练期间人为地增加了响应长度（尤其是对于错误输出）。为了解决这一问题，我们引入了 Dr. GRPO，这是一种无偏的优化方法，在保持推理性能的同时提高了标记效率。借助这些见解，我们提出了一个极简主义的 R1-Zero 方案，使用 7B 基础模型在 AIME 2024 上实现了 43.3% 的准确率，树立了新的最高标准。

**链接**: [https://github.com/sail-sg/understand-r1-zero](https://github.com/sail-sg/understand-r1-zero)

**公式 1**:
$$
\begin{aligned}
&\min_{\theta} \mathbb{E}_{q \sim p_Q} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min \left( \frac{p_{\theta}(o_i^t|q, o_i^{<t})}{p_{\theta_{\text{old}}}(o_i^t|q, o_i^{<t})} \hat{A}_{i,t}, \text{clip} \left( \frac{p_{\theta}(o_i^t|q, o_i^{<t})}{p_{\theta_{\text{old}}}(o_i^t|q, o_i^{<t})}, 1 - \epsilon, 1 + \epsilon \right) \hat{A}_{i,t} \right) \right], \\
&\text{其中 } \hat{A}_{i,t} = \frac{R(q, o_i) - \text{mean}(\{R(q, o_1), \ldots, R(q, o_G)\})}{\text{std}(\{R(q, o_1), \ldots, R(q, o_G)\})}.
\end{aligned}
$$
**公式 2**:
$$
\begin{aligned}
&\min_{\theta} \mathbb{E}_{q \sim p_Q} \left[ \frac{1}{G} \sum_{i=1}^G \sum_{t=1}^{|o_i|} \min \left( \frac{p_{\theta}(o_i^t|q, o_i^{<t})}{p_{\theta_{\text{old}}}(o_i^t|q, o_i^{<t})} \hat{A}_{i,t}, \text{clip} \left( \frac{p_{\theta}(o_i^t|q, o_i^{<t})}{p_{\theta_{\text{old}}}(o_i^t|q, o_i^{<t})}, 1 - \epsilon, 1 + \epsilon \right) \hat{A}_{i,t} \right) \right], \\
&\text{其中 } \hat{A}_{i,t} = \frac{R(q, o_i) - \text{mean}(\{R(q, o_1), \ldots, R(q, o_G)\})}{\text{std}(\{R(q, o_1), \ldots, R(q, o_G)\})}.
\end{aligned}
$$
**图 1**: 左侧：Dr. GRPO 通过移除长度和标准差归一化项，对 GRPO（Shao et al., 2024）中的偏差进行了简单但重要的修改。右侧：我们的无偏优化器有效地防止模型生成越来越长的错误响应，从而提高了标记效率。

**图 2**: 模型性能比较。Oat-Zero-7B 是使用我们在第 1 节（第三段）中描述的极简主义方案进行 RL 训练的。更多结果请参见附录 B。

## **1. 引言**

DeepSeek-R1-Zero（Guo et al., 2025）通过引入 R1-Zero 类训练范式，革新了大型语言模型（LLM）的后训练流程：直接将强化学习应用于基础 LLM，而不依赖监督微调（SFT）作为初步步骤。这种新范式因其简洁性和展示的强化学习扩展现象而备受关注：随着模型响应长度的持续增加，模型的推理能力也会提升。这种现象还伴随着“顿悟时刻”，此时模型学会了诸如自我反思等新兴技能。

在本文中，我们旨在通过研究两个基本组成部分：基础模型和强化学习，来理解 R1-Zero 类训练。在第一部分，我们研究了基础模型的各种属性，重点关注 Qwen2.5 模型家族（Yang et al., 2024a;b），这些模型最近被用于尝试复现 R1-Zero（Pan et al., 2025; Zeng et al., 2025; Liu et al., 2025b; Hu et al., 2025），以及 DeepSeek-V3-Base（Liu et al., 2024），真正的 R1-Zero 模型就是基于此进行 RL 训练的。在第二部分，我们识别了 GRPO（Shao et al., 2024）优化中的偏差，这种偏差可能导致逐渐变长的错误响应。为此，我们提出了一个简单的修改来消除偏差，即实现 GRPO 做对了（Dr. GRPO），从而实现更好的标记效率（如图 1 所示）。

我们对基础模型和强化学习的分析提出了一个极简主义的 R1-Zero 类训练方案：我们使用无偏的 Dr. GRPO 算法在 MATH（Hendrycks et al., 2021）3-5 级问题上对 Qwen2.5-Math-7B 进行 RL 训练，并在 Qwen-Math 模板上实现了最先进的性能（如图 2 所示），仅在 8×A100 GPU 上训练了 27 小时。我们希望本文呈现的发现、发布的模型以及开源的代码库能够惠及该领域的未来研究。

作为概述，我们总结了本文的主要收获：

### **概览**

- （第 2.1 节）模板对于使基础模型回答问题而不是完成句子至关重要。此外，所有基础模型在 RL 之前已经具备数学解题能力。
- （第 2.2 节）有趣的是，Qwen-2.5 基础模型在不使用模板时立即获得了约 60% 的提升，这让我们假设它们可能在训练模型时预训练了拼接的问答文本。
- （第 2.3 节）几乎所有基础模型都表现出“顿悟时刻”，包括 DeepSeek-V3-Base。
- （第 3.1 节，第 3.2 节）Dr. GRPO 有效地修正了 GRPO 在优化中的偏差，实现了更好的标记效率。
- （第 3.3 节）模型-模板不匹配可能会破坏推理能力，直到 RL 重建它。
- （第 3.4 节）在 Llama-3.2-3B 上进行数学预训练可以提高其 RL 上限。

## **2. 基础模型分析**

在本节中，我们对包括 Qwen-2.5 系列（Yang et al., 2024a;b）、Llama-3.1（Grattafiori et al., 2024）和 DeepSeek 系列（Liu et al., 2024; Shao et al., 2024; Guo et al., 2025）在内的广泛基础模型进行了深入研究，向它们提出了 500 个从 MATH（Hendrycks et al., 2021）训练集中采样的问题，并分析了它们的响应。

### **2.1. R1-Zero 可训练性：模板构建探索性基础策略**

由于从基础模型进行训练是 R1-Zero 类范式的基本设置，我们首先研究广泛使用的基础模型（通常用于句子补全，即 $p_{\theta}(x)$）是否可以通过适当的模板有效地激发其问答能力，从而作为问答基础策略 $\pi_{\theta}(\cdot|q)$。除了 Guo et al.（2025）中的 R1 模板（模板 1）外，我们还考虑了 Zeng et al.（2025）使用的 Qwen-Math 模板（模板 2）以及无模板（模板 3）：

**模板 1（R1 模板）**：

```shell
Template 1 (R1 template).

A conversation between User and Assistant. The User asks a
question, and the Assistant solves it. The Assistant first thinks about the reasoning process
in the mind and then provides the User with the answer. The reasoning process is enclosed
within <think> </think> and answer is enclosed within <answer> </answer> tags,
respectively, i.e., <think> reasoning process here </think> <answer> answer here
</answer>.

User: {question}

Assistant: <think>
```

**模板 2（Qwen-Math 模板）**：

```
Template 2 (Qwen-Math template).


<|im start|>system
Please reason step by step, and put your final answer within \boxed{}.<|im end|>
<|im start|>user
{question} <|im end|>
<|im start|>assistant
```

**模板 3（无模板）**：

```
{question}
```

**实验设置**：实验包括 Qwen2.5-Math-1.5B、Qwen2.5-Math-7B、Qwen2.5-7B、Llama-3.1-8B、DeepSeek-Math-7B 和 DeepSeek-V3-Base-685B。对于每个模型，我们首先应用无模板获取模型响应，然后让 GPT-4o-mini 判断模型响应是否为问答格式（不管质量如何）还是句子补全模式。我们将倾向于回答问题的响应百分比记录为指标。然后，我们分别应用 R1 模板和 Qwen-Math 模板获取模型响应，并根据指标确定每个模型最适合的模板。最后，我们使用相应模板评估每个模型的 pass@8 准确率，以确定基础策略是否能够探索出有益于 RL 改进的轨迹。

**结果**：图 3 的左侧显示了基础模型（有无模板）回答问题的能力。我们发现，使用合适的模板（R1 模板）可以提高 Llama 和 DeepSeek 模型的回答能力。然而，Qwen2.5 模型在不使用模板时表现最佳（回答率为 100%）。这一有趣的特性激发了我们进一步调查的兴趣，如第 2.2 节所述。同时，最低的回答率表明 DeepSeek-V3-Base 是一个近乎纯基础模型。这一观察结果促使我们探索像 DeepSeek-V3-Base 这样的纯基础模型是否表现出顿悟时刻（第 2.3 节）。图 3 的中间部分显示了不同基础模型（使用模板）在不同采样温度下的 pass@8 准确率。这一指标可以作为基础策略探索能力的指标。例如，如果基础策略甚至无法采样出一条导致正确最终答案的轨迹，那么就不可能通过 RL 改进策略，因为没有奖励信号。我们的结果表明，所有测试的模型都具有探索性（因此已准备好进行 RL），其中 Qwen2.5 模型表现最佳（甚至超过了 DeepSeek-V3-Base）。这或许可以部分解释为什么大多数 R1-Zero 项目（Zeng et al., 2025; Hu et al., 2025）都基于 Qwen2.5 模型。

### **2.2 Qwen-2.5 模型在丢弃模板时解锁最佳性能**

接下来，我们深入研究一个有趣的发现（见图 3 左侧），即所有 Qwen2.5 基础模型即使没有模板也能轻松作为聊天模型。我们进一步评估 Qwen2.5-Math 模型在五个标准基准测试中的推理能力：AIME 2024（Li et al., 2024）、AMC（Li et al., 2024）、MATH500（Hendrycks et al., 2021）、Minerva Math（Lewkowycz et al., 2022）和 OlympiadBench（He et al., 2024）。按照常见做法，我们使用贪婪解码，并将采样预算限制为 3000 个标记。

如表 1 所示，不使用任何模板可以显著提升平均性能，与传统的 4-shot 提示相比，性能提升了约 60%。由于 Qwen2.5-Math（Yang et al., 2024b）在预训练阶段使用了聊天模型的数据（问答对），我们假设它们可能在拼接的文本上进行了预训练，以直接最大化 $\log p_{\theta}(q; o)$。如果我们的假设属实，我们在使用 Qwen2.5 模型复现 DeepSeek-R1-Zero 时就需要更加小心，因为这些基础模型在没有模板的情况下已经类似于经过了监督微调。

**表 1**: Qwen2.5-Math 模型可能在拼接的问答文本上进行了预训练，因此在不使用模板时达到峰值性能。

| 基础模型 + 模板                 | AIME24 | AMC  | MATH500 | Minerva Math | OlympiadBench | 平均值 |
| ------------------------------- | ------ | ---- | ------- | ------------ | ------------- | ------ |
| Qwen2.5-Math-1.5B (4-shot 提示) | 0.0    | 20.0 | 50.4    | 12.1         | 15.9          | 19.7   |
| R1 模板                         | 0.0    | 9.6  | 21.2    | 6.6          | 2.2           | 7.9    |
| Qwen 模板                       | 20.0   | 32.5 | 33.0    | 12.5         | 22.8          | 24.2   |
| 无模板                          | 16.7   | 43.4 | 61.8    | 15.1         | 28.4          | 33.1   |
| Qwen2.5-Math-7B (4-shot 提示)   | 3.3    | 22.5 | 61.6    | 10.7         | 20.9          | 23.8   |
| R1 模板                         | 0.0    | 0.0  | 0.0     | 0.0          | 0.1           | 0.0    |
| Qwen 模板                       | 16.7   | 38.6 | 50.6    | 9.9          | 16.6          | 26.5   |
| 无模板                          | 0.2    | 45.8 | 69.0    | 21.3         | 34.7          | 38.2   |

### 2.3 基础模型中已经出现的 Aha Moment，包括 DeepSeek-V3-Base

DeepSeek-R1-Zero 最令人振奋的结果之一是通过纯强化学习（RL）训练出现了自我反思行为，即所谓的“顿悟时刻”（Aha Moment）。一些先前的研究（Liu et al., 2025b; Yeo et al., 2025）表明，在开源的 R1 复现中可能没有出现 Aha Moment，因为这些研究中使用的基础模型已经表现出自我反思的关键词。然而，这些研究并没有测试 DeepSeek-V3-Base，而真正的 R1-Zero 模型正是基于此进行 RL 调整的。我们通过托管 DeepSeek-V3-Base-685B 并调查其对 500 个 MATH 问题的响应，填补了这一空白。

从图 3 的右侧可以看出，DeepSeek-V3-Base 也生成了相当数量的自我反思行为，这进一步验证了 Liu et al.（2025b）的主张。我们还在图 4 中展示了 DeepSeek-V3-Base 生成的关键词示例，例如“Aha”、“wait”和“verify the problem”。

**图 4**：DeepSeek-V3-Base 在回答问题时已经表现出 Aha Moment 的示例。

- **问题 (a)**：求最小的正整数 \( n \)，使得 \(\frac{1}{\sin 45^\circ \sin 46^\circ} + \frac{1}{\sin 47^\circ \sin 48^\circ} + \cdots + \frac{1}{\sin 133^\circ \sin 134^\circ} = \frac{1}{\sin n^\circ}\)。
  **响应**：首先，我注意到等式左边的形式是 \(\sum_{k=1}^{n} \frac{1}{\sin a_k \sin a_{k+1}}\)，其中 \(a_1 = 45^\circ\)，\(a_2 = 47^\circ\)，...，\(a_{90} = 133^\circ\)。我还注意到 \(\sin a_{k+1} = \sin(a_k + 2^\circ)\)。... 简化后，我得到 \(\sin(3a) = 3 \sin a - 4 \sin^3 a\)。Aha! 我可以利用这个公式来简化等式左边。... 这看起来很复杂，但我注意到它又回到了 \(\sin a \cos b + \cos a \sin b\) 的形式。这让我有了一个想法。也许我可以反复使用这个恒等式来简化等式的左边。...

- **问题 (b)**：从帕斯卡三角形的前 15 行中随机选择一个元素...
  **响应**：在帕斯卡三角形中，每一行都以 1 开始和结束，... ... 这可以计算为：\(\frac{(n + 1)!}{2! \times (n - 1)!}\)。... ... 这让我有点困惑。让我再试一次。正确的做法是...

- **问题 (c)**：Denali 和 Nate 在一家遛狗公司工作，他们被支付...
  **响应**：为了解决这个问题，我们先定义变量... ... 再次，由于 \(x \neq 0\)，这个方程没有给出有效的解。让我们检查一下我们是否犯了错误。我们应该重新验证问题的条件。正确的做法是...

一个重要的问题是自我反思行为是否在 RL 训练后增强了模型的性能。为了研究这一点，我们托管了 DeepSeek-R1-Zero 并分析了其对相同 MATH 数据集问题的响应。我们发现自我反思行为在 R1-Zero 中出现的频率更高。然而，没有明确的证据表明它们与更高的准确性相关。详细分析请见附录 D。

语言模型生成可以表述为标记级马尔可夫决策过程（MDP）$M = (S, A, r, p_Q)$。在每个生成步骤 $t$，状态 $s_t \in S$ 是输入问题和到目前为止生成的输出响应的连接：$s_t = q; o^{<t} = [q_1, \ldots, q_M, o_1, \ldots, o_{t-1}]$。策略 $\pi_{\theta}(\cdot|s_t)$ 将从词汇表 $A$ 中选择下一个标记 $o_t$，从而确定性地转移到下一个状态 $s_{t+1} = s_t; [o_t]$。生成过程从采样初始状态 $s_1 = q \sim p_Q$ 开始，并在自回归策略生成 [eos] 标记或耗尽预算时停止。

通常，我们最大化熵正则化目标（Schulman et al., 2017a）：
$$
J (\pi_{\theta}) = \mathbb{E}_{q \sim p_Q} \left[ \mathbb{E}_{o \sim \pi_{\theta}(\cdot|q)}[R(q, o)] - \beta D_{KL}[\pi_{\theta}(\cdot|q)||\pi_{\text{ref}}(\cdot|q)] \right],
$$
其中 $R(q, o) = \sum_{t=1}^{|o|} r(s_t, o_t)$ 是轨迹 $q; o$ 的回报（Sutton & Barto, 2018），$\pi_{\text{ref}}$ 是参考策略。正则化项通常用于强化学习从人类反馈（Christiano et al., 2017）中学习，其中 $r$ 是从 $\pi_{\text{ref}}$ 收集的数据中学习到的奖励模型。在这种情况下，正则化有助于防止 $\pi_{\theta}$ 偏离奖励模型准确的分布太远（Jaques et al., 2019; Stiennon et al., 2020）。然而，推理模型的 RL 训练通常使用基于规则的验证器作为 $r$（Lambert et al., 2024），消除了对分布偏移的担忧。这允许我们移除 KL 项，这不仅节省了训练期间 $\pi_{\text{ref}}$ 所需的内存和计算，而且可能为 R1-Zero 类训练带来更好的性能（Hu et al., 2025）。在本文中，我们假设 $\beta = 0$。

**策略优化算法**。为了优化具有上述目标（公式 1，$\beta = 0$）的 $\pi_{\theta}$，近端策略优化（PPO）（Schulman et al., 2017b）最大化以下替代目标：
$$
J_{\text{PPO}}(\pi_{\theta}) = \mathbb{E}_{q \sim p_Q, o \sim \pi_{\theta_{\text{old}}}(\cdot|q)} \left[ \frac{1}{|o|} \sum_{t=1}^{|o|} \min \left( \frac{\pi_{\theta}(o_t|q, o^{<t})}{\pi_{\theta_{\text{old}}}(o_t|q, o^{<t})} \hat{A}_t, \text{clip} \left( \frac{\pi_{\theta}(o_t|q, o^{<t})}{\pi_{\theta_{\text{old}}}(o_t|q, o^{<t})}, 1 - \epsilon, 1 + \epsilon \right) \hat{A}_t \right) \right],
$$
其中 $\pi_{\theta_{\text{old}}}$ 是更新前的策略，$\epsilon$ 是剪切超参数，$\hat{A}_t$ 是第 $t$ 个标记的优势函数估计器。估计 $\hat{A}_t$ 的标准方法是使用学习到的价值模型 $V_{\phi}$ 计算广义优势估计（GAE）（Schulman et al., 2015）。然而，在 LLM RL 训练的背景下，学习价值模型在计算上是昂贵的，因此实际中更倾向于不使用 $V_{\phi}$ 来估计 $\hat{A}_t$ 的方法。例如，Shao et al.（2024）提出了 GRPO，它首先针对每个问题采样一组响应 $\{o_1, \ldots, o_G\}$ 并计算它们的回报 $R = \{R_1, \ldots, R_G\}$，然后将所有来自 $o_i$ 的标记的优势设置为 $\hat{A}_{i,t} = \frac{R - \text{mean}(R)}{\text{std}(R)}$。

### **3.1.  GRPO 导致有偏优化**

在 Deepseek-R1-Zero（Guo et al., 2025）中，一个显著的趋势是响应长度在整个训练过程中持续增加。这一现象通常被解释为高级推理能力（如自我反思）发展的迹象。许多研究（Pan et al., 2025; Zeng et al., 2025; Hu et al., 2025）使用各种算法和实现复制了这一现象。然而，我们认为观察到的响应长度增加也可能归因于 GRPO（Shao et al., 2024）目标函数中的固有偏差：

$$
J_{\text{GRPO}}(\pi_{\theta}) = \mathbb{E}_{q \sim p_Q, \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(\cdot|q)} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min \left( \frac{\pi_{\theta}(o_{i,t}|q, o_i^{<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q, o_i^{<t})} \hat{A}_{i,t}, \text{clip} \left( \frac{\pi_{\theta}(o_{i,t}|q, o_i^{<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q, o_i^{<t})}, 1 - \epsilon, 1 + \epsilon \right) \hat{A}_{i,t} \right) \right],
$$
其中 $\hat{A}_{i,t} = \frac{R(q, o_i) - \text{mean}(\{R(q, o_1), \ldots, R(q, o_G)\})}{\text{std}(\{R(q, o_1), \ldots, R(q, o_G)\})}$，$R(q, o_i)$ 表示给定问题 $q$ 和采样响应 $o_i$ 的最终回报（在 Deepseek-R1-Zero 中，结论也适用于过程回报）。

与公式 2 相比，GRPO 引入了两个偏差：

- **响应级长度偏差**：由除以 $|o_i|$ 引起。对于正优势（$\hat{A}_{i,t} > 0$，表示正确响应），这种偏差会导致较短响应的梯度更新更大，从而使策略倾向于在正确答案中使用简洁的表述。相反，对于负优势（$\hat{A}_{i,t} < 0$，表示错误响应），由于较大的 $|o_i|$，较长的响应受到的惩罚更少，导致策略在错误时倾向于更长的响应。
- **问题级难度偏差**：由除以 $\text{std}(\{R(q, o_1), \ldots, R(q, o_G)\})$ 引起。标准差较低的问题在策略更新时被赋予更高的权重。虽然在 RL 中优势归一化是一种常见技巧（Andrychowicz et al., 2021），但通常是在整个批次上计算的。相比之下，问题级归一化导致不同问题在目标中的权重不同。

**长度偏差也存在于开源 PPO 实现中**。我们还检查了几种流行的开源 PPO 算法的实现，用于 LLM 后训练。令我们惊讶的是，所有这些实现都存在响应级长度偏差（见代码清单 1 和表 2），这种偏差甚至在 GRPO 发布之前就存在了。我们推测这种按标记的归一化源自 LLM 的下一个标记预训练（Shoeybi et al., 2019），其目的是使每个标记对目标的贡献相等。然而，在 RL 的背景下，通过除以 $|o_i|$ 进行归一化引入了意外的偏差。

**代码清单 1**：比较典型的开源 LLMs 的有偏 PPO 损失实现（红色）和我们的实现（绿色）。

```python
1 def masked_mean(tensor, mask, dim):
2     return (tensor * mask).sum(axis=dim) / mask.sum(axis=dim)
3 +    return (tensor * mask).sum(axis=dim) / MAX_TOKENS
4
5 ppo_loss = ...  # 计算每个标记的 PPO 损失
6 response_mask = ...  # 每个标记的响应掩码
7 loss = masked_mean(ppo_loss, response_mask, dim=-1)
```



**表 2**：许多开源的 PPO 实现包含响应级长度偏差。

| 仓库代码链接                                  | 无偏性 |
| --------------------------------------------- | ------ |
| trl（von Werra et al., 2020）PPO 损失         | 否     |
| OpenRLHF（Hu et al., 2024）PPO 损失           | 否     |
| verl（Sheng et al., 2024）PPO 损失            | 否     |
| SimpleRL-Zero（Zeng et al., 2025）PPO 损失    | 否     |
| Open-Reasoner-Zero（Hu et al., 2025）PPO 损失 | 否     |

### **3.2 Dr. GRPO：修正的 Group Relative Policy Optimization**

为了避免 GRPO 中的上述优化偏差，我们建议简单地移除 $1/|o_i|$ 和 $\text{std}(\{R(q, o_1), \ldots, R(q, o_G)\})$ 这两个归一化项。同时，为了忠实实现无偏的优化目标，我们可以在代码清单 1 中的 masked_mean 函数中用一个常数值（例如，生成预算）替换 mask.sum(axis=dim)，如绿色的行所示。值得注意的是，这些简单的修改恢复了公式 2 中的 PPO 目标，优势通过蒙特卡洛回报估计，使用无偏基线（Sutton & Barto, 2018）。我们详细推导见附录 A。我们称新的优化算法为 Dr. GRPO。接下来，我们通过实验验证 Dr. GRPO 的有效性。

**实验设置**。我们使用 Oat（Liu et al., 2025a），一个模块化、研究友好且高效的 LLM RL 框架来实现我们的算法。我们采用 Qwen2.5-1.5B 基础模型和 R1 模板（模板 1）进行在线 RL 训练。我们使用 Math-Verify2 实现基于验证的奖励函数，采用以下极简主义规则：
$$
R(q, o) = \begin{cases}
1 & \text{如果 } o \text{ 包含 } q \text{ 的正确最终答案} \\
0 & \text{其他情况}
\end{cases}
$$
我们在 MATH（Hendrycks et al., 2021）训练数据集上采样问题进行 RL 训练，并比较原始 GRPO 和我们提出的 Dr. GRPO。我们在五个基准测试（AIME2024、AMC、MATH500、Minerva Math 和 OlympiadBench）上评估在线模型。更多实验细节，包括超参数，可以在我们的开源代码库中找到。

**结果**。我们在图 5 中报告多种指标，以证明 Dr. GRPO 可以有效减轻优化偏差并实现更好的标记效率。特别是，我们首先注意到 GRPO 和 Dr. GRPO 都表现出与 DeepSeek-R1-Zero（Guo et al., 2025）相似的趋势，即它们的响应长度随着训练回报的增加而增加（图 1 和图 2）。然而，我们观察到 GRPO 倾向于在回报改进放缓时仍然持续生成更长的响应（图 2）。尽管这种现象通常被称为通过 RL“出现”的长推理链（Zeng et al., 2025; Hu et al., 2025），但我们也认为它受到优化过程中响应级长度偏差（第 3.1 节）的混淆。相比之下，通过计算无偏的策略梯度，Dr. GRPO 防止了响应长度在训练过程中的无序增长（图 2）。此外，在评估基准测试中，与基线相比，Dr. GRPO 显著减少了错误响应的长度（图 4），表明无偏优化器也减轻了过度思考（Chen et al., 2024）。

### **3.3. 模板和问题集覆盖在 RL 动态中的二重奏**

回想一下，Qwen2.5-Math 基础模型可以在没有任何提示模板的情况下轻松回答问题（第 2.2 节）。基于这一有趣的观察，我们有兴趣研究不同模板如何影响 RL 训练。此外，鉴于普遍认为更大的问题集覆盖可以带来更好的性能（Luo et al., 2025; Hu et al., 2025），我们还研究了不同模板和不同水平的问题覆盖之间的相互作用。

**实验设置**。从 Qwen2.5-Math-1.5B 基础模型开始，我们分别应用 R1 模板、Qwen-Math 模板和无模板，使用 Dr. GRPO 进行 RL 训练。所有实验都针对表 3 中详细描述的不同问题集重复进行。

**表 3**：具有不同难度和覆盖水平的不同问题集。

| 问题集 | 描述                                                      |
| ------ | --------------------------------------------------------- |
| ORZ    | 结合 AIME、Numina-Math、Tulu3 MATH；多样化且数量大（57k） |
| MATH   | 高中数学竞赛问题（12k）                                   |
| GSM    | 更简单的年级学校数学问题（8k）                            |
| ASDiv  | 基本代数（+ - × ÷）问题（2k）                             |

**结果**。图 6 显示了不同运行的 RL 曲线，从中我们可以得出几个有趣的观察结果：

1）模板决定了初始策略的性能，但 RL 可以将所有策略改进到可比的性能（约 40%）（前提是适当的问题集）；

2）当使用 R1 模板时，问题集对 RL 的动态有显著影响，覆盖范围过窄会导致性能平台较低。然而，当使用 Qwen-Math 模板时，最佳最终性能是通过在 GSM-8K 上进行 RL 获得的，这表明在更简单（且分布外）的问题上进行训练可以显著提高（几乎翻倍）对更难问题的测试准确率。从这些观察结果中，我们得出以下见解：

- Qwen2.5-Math-1.5B 基础模型已经具备强大的数学解题能力（见图 6 右侧的起始点）。应用模板实际上破坏了这种能力，然后 RL 再重建它。这意味着我们在声称纯 RL 带来的巨大收益时应该更加谨慎。
- 当基础模型和模板之间存在较大不匹配（例如，R1 模板与 Qwen2.5-Math-1.5B 不匹配）时，策略改进主要来自 RL 训练，因此需要问题集具有良好的覆盖范围（图 6 左侧）。否则，即使是一个小的且完全分布外的问题集也能通过强化正确推理行为而不是灌输新知识来同等有效地诱导推理能力。

#### **3.4 领域特定预训练提高 RL 上限**

最近成功的 R1-Zero 类数学推理器复现大多使用 Qwen2.5 基础模型作为初始策略（Zeng et al., 2025; Cui et al., 2025; Hu et al., 2025），这些模型已经很强的数学解题能力，并表现出自我反思模式（第 2.2 节和第 2.3 节）。在本节中，我们希望探索另一面：R1-Zero 类训练能否在最初较弱（就数学推理而言）的基础模型上成功？我们肯定地回答了这个问题，并观察到数学预训练可以提高 RL 的上限。

**实验设置**。我们采用 Llama-3.2-3B 基础模型作为起点，并使用无偏的 Dr. GRPO 算法进行 RL 训练，采用 R1 模板。我们假设领域特定预训练有助于 RL，因此我们采用 Llama-3.2-3B-FineMath4，它在 FineMath 数据集（Allal et al., 2025）上进行了持续预训练。此外，正如我们假设 Qwen2.5 模型可能在拼接的问答文本上进行了预训练（第 2.2 节），我们同样从 NuminaMath1.5（LI et al., 2024）准备了一个拼接的数据集，并将 Llama-3.2-3B-FineMath 持续预训练 2 个周期，学习率为 1e-5。我们将这种拼接持续预训练的模型称为 Llama-3.2-3BNuminaQA。

**结果**。我们在图 7 左侧展示了不同基础模型的 RL 曲线。我们观察到 RL 甚至可以改进原始的 Llama 基础模型，但收益很小。经过持续预训练（和拼接持续预训练）以嵌入数学领域知识后，Llama 模型可以展现出更强的 RL 性能，验证了我们的假设。我们还重新审视了 Llama 基础模型的 GRPO 的优化偏差。图 7 右侧比较了使用 GRPO 和 Dr. GRPO 训练的模型性能和响应长度。我们可以清楚地看到，GRPO 可以产生“双重增加”现象，可能导致人们误以为长推理链也可以在经过数学预训练的 Llama 模型上出现。不幸的是，长度的增加可能是由于优化偏差（第 3.1 节）导致的，这种偏差可以通过我们提出的 Dr. GRPO 有效减轻（第 3.2 节和图 7 右侧）。

## **4 . 结语**

我们从批判性视角审视了用于 R1-Zero 类训练的基础模型以及用于强化学习的算法。通过分析，我们揭示了预训练偏差如何影响强化学习结果，以及优化选择（如 GRPO）如何无意中塑造模型行为。借助我们提出的 Dr. GRPO，我们提供了一个简单的修复方法，可以在保持推理性能的同时提高标记效率。我们的结果表明，扩展强化学习可以既有效又高效——有时，少即是多。
