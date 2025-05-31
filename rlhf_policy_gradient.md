# Policy Gradient Algorithms

策略梯度算法是强化学习中的重要方法，广泛应用于语言模型的训练，特别是通过人类反馈进行强化学习（RLHF）的场景。这些算法，如PPO（近端策略优化）、GRPO（组相对策略优化）和REINFORCE，通过最近生成的样本更新模型，而不是将分数存储在 ReplayBuffer 中。本节将介绍策略梯度算法的基本原理及其在现代RLHF框架中的应用。

从机器学习层面来看，本节内容是 RLHF 过程中复杂度最高的主题。不过正如大多数现代 AI 模型一样，决定其成功与否的最大因素仍是作为流程输入数据的质量。

用于 RLHF 的最流行算法随时间不断演进。当 ChatGPT 将 RLHF 引入公众视野时，人们普遍知晓其使用了 PPO 算法的变体，许多初期研究都基于此展开。随着时间推移，多项研究项目展示了 REINFORCE 类算法的潜力[1][2]，这类算法因相比 PPO 更简洁（无需奖励模型从而节省内存及 GPU 需求）且价值估计更简单（无需 GAE）而受到推崇。更多算法相继涌现，包括在 Reasoning 任务中尤为流行的组相对策略优化算法，但总体而言这些算法大多可通过调适来匹配特定任务。本章将重点介绍策略梯度的核心框架，以及上述三种在 RLHF 经典文献体系中占据核心地位的算法。

## Policy Gradient Algorithms

强化学习算法的目标是最大化未来折扣奖励，这些奖励是在一系列状态 $ s \in \mathcal{S} $和动作 $ a \in \mathcal{A} $上累积的（更多符号定义见第3章）。智能体的目标，通常称为**回报**，是在给定时间 $ t $的未来折扣奖励之和（其中 $ \gamma \in [0,1) $是一个优先考虑近期奖励的因子）：

$$
G_t = R_{t+1} + \gamma R_{t+2} + \cdots = \sum_{k=0}^\infty \gamma^k R_{t+k+1}.
$$ {#eq:return_definition}
回报的定义也可以递归地表示为：

$$
G_{t} = \gamma{G_{t+1}} + R_{t+1}.
$$ {#eq:recursive_return}
基于这个回报，可以学习一个价值函数 $ V(s) $，该函数是给定当前状态下的未来回报估计：

$$
V(s) = \mathbb{E}\big[G_t | S_t = s \big].
$$ {#eq:value_function}
所有策略梯度算法都针对由特定策略$ \pi(s|a) $  诱导出的价值函数求解一个目标。

假设策略 $ \pi(s) $诱导的状态平稳分布为 $ d_\pi(s) $，优化目标定义为：

$$
J(\theta) = \sum_{s} d_\pi(s) V_\pi(s),
$$ {#eq:policy_objective}
策略梯度算法的核心是计算当前策略下的有限时间期望回报的梯度。有了这个期望回报 $ J $，梯度可以按照以下方式计算，其中 $ \alpha $是学习率：

$$
\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)
$$ {#eq:policy_update}
核心实现细节是如何计算这个梯度。Schulman 等人 2015 年综述了策略梯度的多种计算方法[3]。目标是估计精确梯度 $ g := \nabla_\theta \mathbb{E}[\sum_{t=0}^\infty r_t] $，其存在多种形式，例如

$$
g = \mathbb{E}\Big[\sum_{t=0}^\infty \Psi_t \nabla_\theta \text{log} \pi_\theta(a_t|s_t) \Big],
$$ {#eq:general_gradient}
其中 $ \Psi_t $可以是以下形式（奖励也可以通过 $ \gamma $进行折扣）：

1. $ \sum_{t=0}^{\infty} r_t $：轨迹的总奖励。
2. $ \sum_{t'=t}^{\infty} r_{t'} $：跟随动作 $ a_t $的奖励，也称为回报 $ G $。
3. $ \sum_{t'=t}^{\infty} r_{t'} - b(s_t) $：对前一个公式的基线版本。
4. $ Q^{\pi}(s_t, a_t) $：状态-动作价值函数。
5. $ A^{\pi}(s_t, a_t) $：优势函数，如果能够准确计算，将产生最低的理论方差。
6. $ r_t + V^{\pi}(s_{t+1}) - V^{\pi}(s_t) $：TD残差。

**基线**是一个用于减少策略更新方差的值（更多内容见下文）。

对于语言模型，其中一些概念并不完全适用。例如，我们知道对于确定性策略，价值函数定义为 $ V(s) = \max_a Q(s,a) $，或者对于随机策略为 $ V(s) = \mathbb{E}_{a \sim \pi(a|s)}[Q(s,a)] $。如果我们定义 $ s+a $为提示 $ s $的延续 $ a $，那么 $ Q(s, a) = V(s+a) $，这给出了一个不同的优势计算技巧：

$$
A(s,a) = Q(s,a) - V(s) = V(s + a) - V(s) = r + \gamma V(s + a) - V(s)
$$ {#eq:advantage_trick}
该式综合了即时奖励、提示价值及整体语句的折现价值

### Vanilla Policy Gradient

原始策略梯度的实现通过关于策略参数的微分来优化上述 $ J(\theta) $的表达式。基于整体回报的简化版本为：

$$
\nabla_\theta J(\pi_\theta) = \mathbb{E}_\tau \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) R_t \right]
$$ {#eq:vanilla_policy_gradient}
原始策略梯度算法的常见问题是梯度更新方差过高，可通过多种方式缓解。为此常采用价值估计归一化技术（称为基线），通过状态价值相对于后续动作的标准化实现（如优势函数即 Q 值与状态价值的差值）。最简单的基线是批次奖励均值或滑动平均，即使此类基线也能消除梯度偏差，使得 $ \mathbb{E}_{a \sim \pi(a|s)}[\nabla_\theta \log \pi_\theta(a|s)] = 0 $，从而显著提高学习信号。

本章讨论的多数策略梯度算法均基于优势函数形式：

$$
\nabla_\theta J(\pi_\theta) = \mathbb{E}_\tau \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) A^{\pi_\theta}(s_t, a_t) \right]
$$ {#eq:advantage_policy_gradient}
策略梯度实现的核心环节涉及概率策略的求导运算，其源于：
$$
\nabla_\theta \log \pi_\theta(a|s) = \frac{\nabla_\theta \pi_\theta(a|s)}{\pi_\theta(a|s)}
$$ {#eq:log_prob_derivative}
这是从链式法则推导出来的：

$$
\nabla_\theta \log x = \frac{1}{x} \nabla_\theta x
$$ {#eq:log_chain_rule}
我们将在本章后面使用这个公式。

### REINFORCE

REINFORCE 算法很可能是一个逆向首字母缩写词，但其代表的算法组成部分对现代强化学习算法极具参考价值。该算法在开创性论文《Simple statistical gradient-following algorithms for connectionist reinforcement learning》[4]中定义如下：

> 名称是一个缩写，表示“REward Increment = Nonnegative Factor X Offset Reinforcement X Characteristic Eligibility”。

这三个组成部分共同构成了奖励增量（即策略梯度步骤）的实现方式。其更新规则包含三个要素：

1. 非负因子：即必须为正数的学习率（步长），如下文中的，例如下面的 $ \alpha $。
2. 偏移强化：这是奖励的基线 $ b $或其他归一化因子，用于提高稳定性。
3. 特征资格：这是学习如何按token归因。它可以是一个通用值 $ e $，但在现代公式中通常是策略的对数概率。

因此，其形式看起来相当熟悉：

$$
\Delta_\theta = \alpha(r - b)e
$$ {#eq:REINFORCE_BASIC}
用更现代的符号和广义回报 $ G $，REINFORCE算子表示为：

$$
\nabla_{\theta}\,J(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta}}\Big[ \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t)\,(G_t - b) \Big],
$$ {#eq:REINFORCE_with_baseline}
这里，值 $ G_t - b(s_t) $是当前状态下策略的优势，因此我们可以因此我们可以用优势函数将策略梯度重新表述为下面形式：

$$
\nabla_{\theta}\,J(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta}}\Big[ \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t)\,A_t \Big],
$$ {#eq:REINFORCE_with_advantage}
REINFORCE是原始策略梯度的一个具体实现，它使用蒙特卡洛估计器来估计梯度。

REINFORCE 可在不依赖价值网络的情况下运行——价值网络仅用于策略梯度中的基线计算。而 PPO 算法则需要价值网络来精确计算优势函数。

### RLOO

REINFORCE Leave One Out 与标准 REINFORCE 的核心实现区别在于：前者采用批次中其他样本的平均奖励作为基线进行计算，而非使用批次全部奖励的平均值[5][1][6]。

关键在于，这种方法仅在每个提示生成多个响应时才有效，而这在使用强化学习微调语言模型的多个领域中都是常见做法。

具体来说，对于REINFORCE留一法（RLOO）的基线，给定 $ K $个对提示 $ s $采样的轨迹或动作 $ a_1, \dots, a_K $，我们明确地将基线定义为以下**每个提示**的形式：

$$
b(s, a_k) = \frac{1}{K-1}\sum_{i=1, i\neq k}^{K} R(s, a_i),
$$ {#eq:RLOO_baseline}
从而得出优势：

$$
A(s, a_k) = R(s, a_k) - b(s, a_k).
$$ {#eq:RLOO_advantage}
等价地，这可以表示为：

$$
A(s, a_k) = \frac{K}{K - 1}\left(R(s, a_k) - \frac{1}{K}\sum_{i=1}^{K} R(s, a_i)\right).
$$ {#eq:RLOO_advantage_alt}
这是一种简单且低方差的优势更新方法，与后文将讨论的 GRPO 非常相似——两者都采用 REINFORCE 算法，但 KL 惩罚项的位置不同且未进行步长裁剪。值得注意的是，RLOO 的优势估计可与 PPO 的裁剪机制相结合，这充分体现了这些算法之间的高度相似性。

不使用价值网络的RLOO和其他算法将序列的优势（或奖励）分配给每个token，用于损失计算。使用学习价值网络的算法，如PPO，为每个token分配不同的值，从EOS token处实现的最终奖励进行折扣。

例如，对于KL散度距离惩罚，RLOO 会在整个completion序列上进行求和，而 PPO 及类似算法则基于每个token单独计算，并将其从奖励中扣除（对于 GRPO 算法则是从优势值中扣除）。本章后续将详细讨论这些实现细节及其权衡关系。

### PPO

近端策略优化（PPO）[7]是深度强化学习取得突破性成果的基础算法之一（如 OpenAI 的 DOTA 5[8]及大量研究成果）。其单样本损失函数如下：
$$
J(\theta) = \min\left(\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}A, \text{clip} \left( \frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}, 1-\varepsilon, 1+\varepsilon \right) A \right).
$$ {#eq:PPO_EQN}
对于语言模型而言，损失是按每个token计算的，这直观上可以理解为如何计算自回归预测整个序列的概率——通过概率的乘积来实现。由此出发，常见的实现方式是采用对数概率，这使得计算变得更为可行。
$$
J(\theta) = \frac{1}{|a|} \sum_{t=0}^{|a|} \min\left(\frac{\pi_\theta(a_{t}|s_{t})}{\pi_{\theta_{old}}(a_{t}|s_{t})}A_{t}, \text{clip} \left( \frac{\pi_\theta(a_{t}|s_{t})}{\pi_{\theta_{old}}(a_{t}|s_{t})}, 1-\varepsilon, 1+\varepsilon \right) A_{t} \right).
$$ {#eq:PPO_EQN_EXPANDED}
这是按token计算的PPO版本，也适用于其他策略梯度方法，但在本章的实现部分将进一步探讨。这里，按动作中的token数量平均的项 $ \frac{1}{|a|} $来自常见的实现实践，但并未出现在损失函数的正式推导中（如文献[9]所示）。

此处我们将阐释该损失函数在不同优势值与策略比率下触发的差异情况。在实现层面，PPO 的内部计算包含标准策略梯度与截断策略梯度两项核心运算。

为了理解不同情况如何出现，我们可以定义策略比率为：

$$
R(\theta) = \frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}
$$ {#eq:PPO_POL_RATIO}
策略比率是 PPO 及相关算法的核心要素。它源于策略梯度的计算，并以非常直观的方式控制参数更新。对于任何批次数据，策略比率在该批次的首次梯度步长时初始值为 1（常见做法是在策略梯度算法中对每个批次执行 1-4 次梯度更新）。随后，若该梯度步长提高了具有正优势值的特定token的概率，则策略比率将大于 1；反之则会小于 1。

策略比率与优势函数可以以几种不同的配置组合出现。

第一种情况是当优势为正且策略比率超过 $ 1+\varepsilon $（意味着新策略更有可能采取该动作），时，该比率会被截断，此时目标函数变为：

$$
J(\theta) = \min \left(R(\theta), (1 + \varepsilon)\right)A = (1 + \varepsilon)A
$$ {#eq:PPO_CASE1}
这将增加概率比率，使动作更有可能发生，但仅限在clip 参数ε范围内。类似情况还包括优势函数仍为正时，但似然比发生了偏移。

对于正优势且比率小于 $ 1-\varepsilon $，我们得到一个部分替换的方程：

$$
J(\theta) = \min \left(R(\theta), (1 - \varepsilon)\right)A
$$ {#eq:PPO_CASE2}
这简化为

$$
J(\theta) = R(\theta)A
$$ {#eq:PPO_CASE3}
由于小于假设。

同样，如果概率比率没有被clip ，目标也简化为 $ \min(R(\theta),R(\theta)) $，从而产生一个带有优势估计器的标准策略梯度。

如果优势为负，情况看起来类似。当 $ R(\theta) < (1-\varepsilon) $时，将出现截断目标函数，其表现形式为：

$$
J(\theta) = \min \left(R(\theta)A, (1 - \varepsilon)A\right),
$$ {#eq:PPO_CASE4}
因为 $ A<0 $，我们有 $ R(\theta)A > (1-\varepsilon)A $，并且在从方程中提取 A 时可以将最小值转换为最大值，因此上式等价于

$$
J(\theta) = \max \left(R(\theta), (1 - \varepsilon)\right)A.
$$ {#eq:PPO_CASE5}
然后目标变为：

$$
J(\theta) = (1 - \varepsilon)A
$$ {#eq:PPO_CASE6}
其余情况可依上述方法类推，只需将条件反转，具体推导留作读者练习。

所有这些设计旨在使优势为正的行为更可能出现，并将梯度步长保持在信任区域内。关键要记住的是，信任区域内的 PPO 与标准形式的策略梯度算法本质相同。

### GRPO

群组相对策略优化（GRPO）由 DeepSeekMath[10]提出，并应用于其他 DeepSeek 系列研究，如 DeepSeek-V3[11]和 DeepSeek-R1[12]。该算法可视为受 PPO 启发的改进方法，其替代损失函数设计极为相似，但无需通过复制原始策略语言模型（或加载另一检查点进行初始化）来学习价值函数。这一设计带来两大优势：

1. 避开从语言模型主干学习价值函数的挑战，该领域尚未确立最佳实践。
2. 通过无需在内存中保存另一组模型权重来节省内存。

GRPO通过简化价值估计并为每个token分配相同的值来实现这一点（即在提示的completion中，每个token被分配相同的值，而不是标准价值函数中的折扣奖励），通过估计优势或基线来完成。估计是通过从相同的初始状态/提示 $ s $收集多个 $ a_i $和奖励 $ r_i $来完成的，即蒙特卡洛估计。

从形式化角度表述，GRPO 目标函数与前述 PPO 目标函数高度相似。对于 GRPO 而言，其损失函数是在给定问题  $ s $的一组响应 $ \{a_1, a_2, ..., a_G\} $上累积的：
$$
J(\theta) = \frac{1}{G}\sum_{i=1}^G \left(\min\left(\frac{\pi_\theta(a_i|s)}{\pi_{\theta_{old}}(a_i|s)}A_i, \text{clip} \left( \frac{\pi_\theta(a_i|s)}{\pi_{\theta_{old}}(a_i|s)}, 1-\varepsilon, 1+\varepsilon \right) A_i \right) - \beta D_{KL}(\pi_\theta||\pi_{ref})\right).
$$ {#eq:GRPO}
与上述类似，我们可以将其扩展为按token计算的损失：

$$
J(\theta) = \frac{1}{G}\sum_{i=1}^G  \frac{1}{|a_i|} \sum_{t=1}^{|a_i|} \left( \min\left(\frac{\pi_\theta(a_{i,t}|s_{i,t})}{\pi_{\theta_{old}}(a_{i,t}|s_{i,t})}A_{i,t}, \text{clip} \left( \frac{\pi_\theta(a_{i,t}|s_{i,t})}{\pi_{\theta_{old}}(a_{i,t}|s_{i,t})}, 1-\varepsilon, 1+\varepsilon \right) A_{i,t} \right) - \beta D_{KL}(\pi_\theta(\cdot|s_{i,t})||\pi_{ref}(\cdot|s_{i,t})) \right)
$$ {#eq:GRPO_token}
需要注意的是，相对于 PPO 算法，GRPO 的标准实现会在损失函数中加入 KL 散度项。对于 completion 索引 $ i $的优势计算为：

$$
A_i = \frac{r_i - \text{mean}({r_1, r_2, \cdots, r_G})}{\text{std}({r_1, r_2, \cdots, r_G})}.
$$ {#eq:GRPO_ADV}
直观而言，GRPO 更新机制通过批量比较单个问题下的多个答案进行学习。模型会逐渐趋近于被标记为正确的答案，而远离其他答案。这是一种计算优势值的简易方法——优势值用于衡量特定动作在给定状态下优于平均水平的程度。相较于 PPO、REINFORCE 及普遍采用奖励模型评分（相对于输出奖励）的 RLHF 方法，GRPO 通常对每个提示词会采样更多样本。具体表现为：当前策略针对给定提示生成多个响应，而分组 GRPO 优势估计则能从中获取有价值的上下文信息。

GRPO 算法在优势值计算方面存在偏差权衡。通过标准差进行归一化的方式会奖励批次中答案正确率差异较小的问题。对于那些几乎所有答案都正确或全部错误的问题，标准差较低，优势值则较高。[9]提出鉴于这种偏差应移除标准差项，但这样做的代价是降低那些多数错误但含少量正确答案问题的权重，而这些可能被视为有价值的学习信号。

公式 29 是 GRPO 在结果监督（标准奖励模型或单一可验证奖励）下的实现方式，而过程监督则需要不同的实现。在这种情况下，GRPO 将优势值计算为后续推理步骤归一化奖励的总和。

最后，GRPO 的优势估计也可应用于没有 PPO 裁剪的更基础的策略梯度算法（如 REINFORCE），但这并非标准形式。作为算法相互关联的例证，我们可以证明 GRPO 变体 Dr. GRPO（正确实现的 GRPO）[9]中的优势估计与 RLOO 估计等价，仅相差一个常数比例因子（由于优势值归一化的实现细节，通常不影响结果）。Dr. GRPO 从公式 29 中移除了标准差归一化项——需注意这会同时放大优势值，相当于对答案分数存在差异的样本提高了 GRPO 学习率。该方法修正了对奖励方差较低问题（即答案几乎全对或全错）的偏好偏差，但可能牺牲从仅单个样本答对的重要问题中学习的机会。Dr. GRPO 对规模为 G 的组内第 i 个completion的优势值定义为：
$$
\tilde{A}_i = r_i - \text{mean}({r_1, r_2, \cdots, r_G}) = r_i - \frac{1}{G}\sum_{j=1}^G r_j
$$ {#eq:DrGRPO_ADV}
在此，我们沿用相同符号体系回顾 RLOO 优势估计量：

$$
A_i^\text{RLOO} = r_i - \frac{1}{G-1}\sum_{j=1, i\neq j}^G r_j
$$ {#eq:RLOO_ADV_AGAIN}
因此，如果我们将Dr. GRPO优势定义乘以 $ \frac{G}{G-1} $，我们可以看到一个缩放后的等价性：

$$
\begin{aligned}
\frac{G}{G-1} \tilde{A}_i &= \frac{G}{G-1} \left( r_i - \frac{1}{G}\sum_{j=1}^G r_j \right) \\
&= \frac{G}{G-1} r_i - \frac{1}{G-1} \sum_{j=1}^G r_j \\
&= \frac{G}{G-1} r_i - \frac{1}{G-1} \sum_{j=1, j\neq i}^G r_j - \frac{1}{G-1} r_i \\
&= r_i \left( \frac{G}{G-1} - \frac{1}{G-1} \right) - \frac{1}{G-1} \sum_{j=1, j\neq i}^G r_j \\
&= r_i - \frac{1}{G-1} \sum_{j=1, j\neq i}^G r_j \\
&= A_i^{\text{RLOO}}
\end{aligned}
$$ {#eq:RLOO_GRPO_EQUIV}

## Implementation

与最初开发这些算法的深度强化学习文献相比，针对语言模型或其他大型 AI 模型优化的强化学习实现涉及诸多细节差异。本节重点剖析主流算法实现中的关键区分要素。

训练过程中还需处理许多细微环节。例如，在语言模型 RLHF 训练中，关键步骤是生成待奖励模型评分的文本。正常情况下模型应生成表示结束的 EOS token，但实践中常对生成长度设硬性上限以提高基础设施利用率。RLHF 的典型故障模式是模型回答频繁被截断，导致奖励模型评分偏离分布并产生不可预测值。解决方案是仅对 `eos_token` 进行奖励模型排序，同时对过长生成施加惩罚。

主流 RLHF 开源工具在算法实现细节上存在显著差异（参见文献[13]表 10）。未涵盖的决策点包括：

- 价值网络初始化：PPO 等算法内部学习价值网络时，可采用同架构不同模型的参数或随机权重初始化，这对性能影响显著。
- 奖励归一化/白化与优势白化：归一化将 RM（或环境）输出值限定在 0-1 区间以提升学习稳定性，而对奖励或优势估计进行白化处理使其协变量均匀分布，可进一步增强稳定性。
- 不同 KL 估计器：针对复杂语言模型，精确计算模型间 KL 散度较为困难，故常采用多种近似方法替代精确计算[14]。
- KL 控制器：PPO 及其相关算法的原始实现采用动态控制器，根据近期测量值调整惩罚力度以达成特定 KL 目标。现代 RLHF 实现多采用静态 KL 惩罚，但该设计也存在变体。

有关 RLHF 实现细节的更多信息，请参阅文献[15]。如需进一步了解相关算法，请参见文献[16]。

### Policy Gradient Basics

策略梯度的简单实现，利用优势函数估计梯度，为 PPO 和 GRPO 等高级算法做准备，如下所示：

```python
pg_loss = -advantages * ratio
```

这里的比率是新策略模型概率相对于参考模型的对数比率。

要理解这个方程，最好先了解一批更新中可能出现的不同情况。请记住，我们希望随着模型在任务中表现提升，损失函数值能够相应降低。

情形一：优势值为正，表明该动作优于状态的期望值。我们需要强化这一行为。此时，模型将通过负号使该动作更可能发生。为此，模型将增大对数比率。正的对数比率（即各token对数概率之和）意味着模型更倾向于生成这些token。

情形二：优势值为负，表明该动作劣于状态的期望值。其推导过程极为相似。此时，若新模型生成该补全的概率更高，则损失函数将呈现正值，因此模型会调整策略参数以降低该补全的生成概率。

情形 3：优势值为零，因此无需更新。损失为零，保持策略模型不变。

### Loss Aggregation

在使用语言模型实现任何策略梯度算法时，核心问题在于：如何通过对 KL 散度与损失函数进行加权求和，从而设计不同类型的价值归因机制。

本节的大部分讨论假设token-level动作，其中RL问题被格式化为马尔可夫决策过程（MDP），而不是一个bandit问题。在bandit问题中，动作中的所有token将被赋予相同的损失，这已成为某些算法（如Advantage-Leftover Lunch RL（A-LoL）[^baheti2023leftover^]）的标准实现。MDP和bandit之间的公式实际上是一个实现细节，关于如何按样本聚合损失。Bandit方法取均值，将相同的损失分配给每个token，这也与DPO和其他直接对齐算法的标准实现一致。

考虑一个例子，我们有以下变量，批量大小为 $ B $，序列长度为 $ L $。

```python
advantages # [B, 1]
per_token_probability_ratios # [B, L]
```

我们可以通过批量乘法 `pg_loss = -advantages * ratio` 来近似上述损失函数。这种乘法操作将批次中每个 completion 的优势值（基于结果奖励设置，而非逐token的价值模型设置）广播为相同值，然后与每个token的概率对数比率相乘。

在使用价值网络的情况下，可以明显看出不同损失函数的表现会存在显著差异。当采用结果奖励时，每个token的优势值被设定为相同，因此每个token概率的差异对于策略梯度的学习动态至关重要。

在下面的GRPO和PPO的实现中，损失是在completion中的token上求和的：

```python
sequence_loss = ((per_token_loss * completion_mask).sum(dim=1) / \
             completion_mask.sum(dim=1)).mean()
```

上述操作与`masked_mean`操作非常相似。

另一种选择是分别对每个token求平均。

```python
token_loss = ((per_token_loss * completion_mask).sum() / \
            completion_mask.sum())
```

从直觉上看，对序列进行平均处理似乎是最佳选择，因为我们的目标是基于结果奖励模型，而具体token并不那么重要。但这种做法可能引入微妙的偏差形式。试想两个长度不同的序列，分别被赋予不同的优势值 $ a_1 $和 $ a_2 $。

```python
seq_1_advs = [a_1, a_1, a_1, a_1, a_1] # 5个token
seq_2_advs = [a_2, a_2, a_2, a_2, a_2, a_2, a_2, a_2, a_2, a_2] # 10个token
```

现在假设每个序列中的最后一个token对于优势值为正至关重要，因此它在每个批次的多个梯度步骤中会得到增强。当将这些转换为逐token损失时，可能会得到近似于以下形式的结果：

```python
seq_1_losses = [1, 1, 1, 1, 10] # 5个token
seq_2_losses = [1, 1, 1, 1, 1, 1, 1, 1, 1, 10] # 10个token
```

如果我们按序列对这些进行平均，我们将得到以下数字：
```python
seq_1_loss = 2.8
seq_2_loss = 1.9
```

若我们按序列等权重平均这些损失，得到的损失值为 2.35。反之，若对每个token均等计算损失，则需将所有token-level损失求和后按序列长度归一化，此时损失值为 2.27。当序列间差异较大时，这两种损失值可能产生显著差异。

要更全面地了解损失聚合如何改变每个token和每个样本的损失，请参阅以下脚本，该脚本计算了一个包含两个样本（一个较长，一个较短）的 demo 批次的损失。

示例使用了三种损失聚合技术：

-  `masked_mean` 对应每样本长度归一化，即 DAPO[18]提出的按批次进行token-level归一化的损失；
- `masked_mean_token_level` ；
- 以及 `masked_sum_result` 采用 Dr. GRPO[9]提出的基于最大长度的固定长度归一化方法。

```python
from typing import Optional
import torch

def masked_mean(values: torch.Tensor, mask: torch.Tensor, axis: Optional[int] = None) -> torch.Tensor:
    """计算带有掩码值的张量的均值。"""
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
    else:
        return (values * mask).sum() / mask.sum()

def masked_sum(
        values: torch.Tensor,
        mask: torch.Tensor,
        axis: Optional[bool] = None,
        constant_normalizer: float = 1.0,
    ) -> torch.Tensor:
    """计算带有掩码值的张量的和。使用常数进行归一化。"""
    if axis is not None:
        return (values * mask).sum(axis=axis) / constant_normalizer
    else:
        return (values * mask).sum() / constant_normalizer

ratio = torch.tensor([
    [1., 1, 1, 1, 1, 1, 1,],
    [1, 1, 1, 1, 1, 1, 1,],
], requires_grad=True)

advs = torch.tensor([
    [2, 2, 2, 2, 2, 2, 2,],
    [2, 2, 2, 2, 2, 2, 2,],
])

masks = torch.tensor([
    # 生成1：4个token
    [1, 1, 1, 1, 0, 0, 0,],
    # 生成2：7个token
    [1, 1, 1, 1, 1, 1, 1,],
])

max_gen_len = 7

masked_mean_result = masked_mean(ratio * advs, masks, axis=1)
masked_mean_token_level = masked_mean(ratio, masks, axis=None)
masked_sum_result = masked_sum(ratio * advs, masks, axis=1, constant_normalizer=max_gen_len)

print("masked_mean", masked_mean_result)
print("masked_sum", masked_sum_result)
print("masked_mean_token_level", masked_mean_token_level)

# masked_mean tensor([2., 2.], grad_fn=<DivBackward0>)
# masked_sum tensor([1.1429, 2.0000], grad_fn=<DivBackward0>)
# masked_mean_token_level tensor(1., grad_fn=<DivBackward0>)

masked_mean_result.mean().backward()
print("ratio.grad", ratio.grad)
ratio.grad.zero_()
# ratio.grad tensor([[0.2500, 0.2500, 0.2500, 0.2500, 0.0000, 0.0000, 0.0000],
# [0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429]])

masked_sum_result.mean().backward()
print("ratio.grad", ratio.grad)
# ratio.grad tensor([[0.1429, 0.1429, 0.1429, 0.1429, 0.0000, 0.0000, 0.0000],
# [0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429]])

masked_mean_token_level.mean().backward()
print("ratio.grad", ratio.grad)
# ratio.grad tensor([[0.2338, 0.2338, 0.2338, 0.2338, 0.0000, 0.0000, 0.0000],
# [0.2338, 0.2338, 0.2338, 0.2338, 0.2338, 0.2338, 0.2338]])
```

从默认的 GRPO 实现 `masked_mean` 可以看出，短序列的每token梯度值大于长序列，而 Dr. GRPO 和 DAPO 的两种实现方式则实现了梯度平衡。需注意的是，若采用梯度累积（即在反向传播前对多个小批次的梯度进行求和），这些结果可能出现显著变化——此时短序列与长序列之间的平衡关系可能发生逆转。

另一种聚合损失的方法在文献[9]中有所讨论，该方法源自语言模型之前的强化学习研究，其中每个token的损失都会通过实验中设置的最大序列长度进行归一化处理。这将改变上述示例中不同批次间各token损失的比较方式。

在实践中，最适合的方案往往是针对个体化在线学习环境所设计的。在强化学习人类反馈（RLHF）方法中，通常更倾向于选择具有最佳数值稳定性或损失方差最小的算法。

### PPO

PPO 算法存在众多实现版本。其核心损失计算如下所示。为确保性能稳定，价值函数的计算同样至关重要，其中存在多种可选方案（包括价值模型损失函数的多种不同处理方式）。

需要注意的是，此处参考策略（或旧对数概率）源自生成样本时的策略，而未必是当前参考策略。参考策略仅用于 KL 散度约束/惩罚项。

```python
# B: 批量大小, L: 序列长度, G: 生成次数
# 应用KL惩罚到奖励
rewards = rewards - self.beta * per_token_kl  # 形状：(B*G, L)

# 获取价值预测
values = value_net(completions)  # 形状：(B*G, L)

# 计算简单优势
advantages = rewards - values.detach()  # 形状：(B*G, L)

# 归一化优势（可选但稳定）
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
advantages = advantages.unsqueeze(1)  # 形状：(B*G, 1)

# 计算新旧策略之间的概率比率
ratio = torch.exp(new_per_token_logps - per_token_logps)  # 形状：(B*G, L)

# PPOclip 目标
eps = self.cliprange  # 例如0.2
pg_losses1 = -advantages * ratio  # 形状：(B*G, L)
pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - eps, 1.0 + eps)  # 形状：(B*G, L)
pg_loss_max = torch.max(pg_losses1, pg_losses2)  # 形状：(B*G, L)

# 简单的价值函数损失
vf_loss = 0.5 * ((rewards - values) ** 2)  # 形状：(B*G, L)

# 结合策略和价值损失
per_token_loss = pg_loss_max + self.vf_coef * vf_loss  # 形状：(B*G, L)

# 应用completion掩码并计算最终损失
loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
 # 标量

# 计算日志记录的指标
with torch.no_grad():
    # 计算clip 比例
    clip_frac = ((pg_losses2 > pg_losses1).float() * completion_mask).sum() / completion_mask.sum()

    # 计算近似KL
    approx_kl = 0.5 * ((new_per_token_logps - per_token_logps)**2).mean()

    # 计算价值损失以供日志记录
    value_loss = vf_loss.mean()
```

理解 PPO 算法的核心在于把握策略梯度损失的更新方式。重点关注以下三行代码：
```python
pg_losses1 = -advantages * ratio  # 形状：(B*G, L)
pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - eps, 1.0 + eps)  # 形状：(B*G, L)
pg_loss_max = torch.max(pg_losses1, pg_losses2)  # 形状：(B*G, L)
```


`pg_losses1` 与上述基于普通优势的 PR 损失相同，该损失包含在 PPO 中，但损失（及梯度更新）可被截断。不过，PPO 通过控制更新幅度来避免过大。由于损失可能为负值，我们必须创建一个更保守版本的普通策略梯度更新规则。

我们知道，如果不约束损失，策略梯度算法会将权重精确更新至新的概率分布。因此，通过限制对数比率，PPO 限定了策略参数更新的移动距离。

最后，如前所述取两者最大值，以选择更保守的损失更新。

我们知道，如果不约束损失，策略梯度算法将完全按照新的概率分布更新权重。因此，通过对数比率进行clip ，PPO限制了更新可以移动策略参数的距离。

最后，如上所述，取两个中的最大值，以获得更保守的损失更新。

对于 PPO 而言，所有这些操作都在学习价值函数的同时进行，这增加了复杂性，但这就是参数更新的核心逻辑。

#### PPO/GRPO 简化版：每个样本仅执行 1 次梯度步（无截断）

若将超参数“每次采样的梯度步数”设为 1，PPO（及 GRPO）算法的实现会优雅得多。该参数的常规取值通常为 2-4 或更高。在 PPO 或 GRPO 的核心方程（参见公式 18）中，"参考"策略即指上一轮迭代的参数——即用于生成完整动作序列或单步动作的参数。因此，如果只进行一次梯度步骤，$\pi_\theta = \pi_{\theta_{old}}$，更新规则简化为以下形式（符号$[]_\nabla$表示停止梯度）：
$$
J(\theta) = \frac{1}{G}\sum_{i=1}^G \left(\frac{\pi_\theta(a_i|s)}{\left[\pi_{\theta}(a_i|s)\right]_\nabla}A_i - \beta D_{KL}(\pi_\theta||\pi_{ref})\right).
$$ {#eq:ppo_1step}
这导致了 PPO 或 GRPO 的实现方式可以省略第二策略梯度和裁剪逻辑，使得优化器更接近于标准策略梯度方法。

### GRPO

DeepSeekMath 论文详细阐述了 GRPO 与 PPO[10]在实现细节上的差异，特别是与深度强化学习（而非语言模型）中标准 PPO 应用的对比。例如，在 RLHF 优化过程中使用的 KL 惩罚（需注意该 KL 惩罚同样适用于基于可验证奖励、无需奖励模型的推理模型训练）是直接应用于损失更新而非奖励函数。在标准的RLHF中的KL惩罚应用是 $ r = r_\theta + \beta D_{KL} $，而GRPO的实现类似于：
$$
L = L_{\text{策略梯度}} - \beta * D_{KL}
$$
不过，具体实现方法存在多种可能。传统上，KL 散度是针对提示对应的每个生成token计算的 。在推理训练中，单个提示会采样多个生成结果，且一个批次包含多个提示，因此 KL 散度的维度为[B, L, N]，其中 B 代表批次大小，L 表示序列长度，N 是每个提示对应的生成结果数量。

综上所述，采用第一种损失累积方式，伪代码可编写如下。

```python
# B: 批量大小, L: 序列长度, G: 生成次数
# 计算组内奖励 # 形状：(B,)
mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)


# 归一化奖励以计算优势
mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
# 形状：(B*G,)

# 计算优势
advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
advantages = advantages.unsqueeze(1)
# 形状：(B*G, 1)

# 计算新旧策略之间的概率比率
ratio = torch.exp(new_per_token_logps - per_token_logps)  # 形状：(B*G, L)

# PPOclip 目标
eps = self.cliprange  # 例如0.2
pg_losses1 = -advantages * ratio  # 形状：(B*G, L)
pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - eps, 1.0 + eps)  # 形状：(B*G, L)
pg_loss_max = torch.max(pg_losses1, pg_losses2)  # 形状：(B*G, L)

# 对于GRPO很重要——PPO传统上在奖励中应用这一点
# 结合KL惩罚
per_token_loss = pg_loss_max + self.beta * per_token_kl  # 形状：(B*G, L)

# 应用completion掩码并计算最终损失
loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
 # 标量

# 计算日志记录的核心指标（也记录KL、奖励等）
with torch.no_grad():
    # 计算clip 比例
    clip_frac = ((pg_losses2 > pg_losses1).float() * completion_mask).sum() / completion_mask.sum()

    # 计算近似KL
    approx_kl = 0.5 * ((new_per_token_logps - per_token_logps)**2).mean()
```

有关如何解读此代码的更多详情，请参阅上文中的 PPO 章节。

#### RLOO与GRPO

RLOO 的优势更新与 GRPO 极为相似，这凸显了当脱离 PPO 风格的clip 和 KL 惩罚细节时，两种算法在概念上的相似性。具体而言，RLOO 的优势计算基于一个与 GRPO 极其相似的基线——同一问题下相对于其他回答的completion奖励。简言之，RLOO 的优势估计如下（根据 TRL 实现扩展而来）：

```python
# rloo_k --> 每个提示的completion次数
# rlhf_reward --> 最初是一个扁平的张量，包含所有completion的总奖励。长度 B = N x k
rlhf_reward = rlhf_reward.reshape(rloo_k, -1) #
# 现在，形状：(k, N)，每一列 j 包含提示 j 的 k 个奖励。

baseline = (rlhf_reward.sum(0) - rlhf_reward) / (rloo_k - 1)
# baseline --> 留一法基线奖励。形状：(k, N)
#  baseline[i, j] 是提示 j 的样本 i' != i 的平均奖励。

advantages = rlhf_reward - baseline
# advantages --> 同样形状：(k, N)

advantages = advantages.flatten() # 与原始张量形状相同
```

RLOO 的其余实现细节遵循了实施策略梯度的其他权衡考量。

## Auxiliary Topics

为了掌握策略梯度算法的应用，还有无数其他需要考虑的因素。在这里，我们考虑其中一些，但并非全部。

### 广义优势估计（GAE）

广义优势估计（GAE）是计算策略梯度算法优势值的替代方法[3]，能更好地平衡偏差-方差权衡。传统的单步优势估计可能引入过多偏差，而使用完整轨迹又往往存在高方差问题。GAE 通过结合两种思想——多步预测与加权滑动平均（或仅采用其中一种）来解决这一问题。

优势估计可以有多种形式，但我们可以定义一个 $ n $步优势估计器（类似于本章开头的TD残差）如下：

$$
\hat{A}_t^{(n)} = \begin{cases}
r_t + \gamma V(s_{t+1}) - V(s_t), & n = 1 \\
r_t + \gamma r_{t+1} + \gamma^2 V(s_{t+2}) - V(s_t), & n = 2 \\
\vdots \\
r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots - V(s_t), & n = \infty
\end{cases}
$$ {#eq:K_STEP_ADV}
在这里，较短的 $ n $将具有较低的方差，但较高的偏差，因为我们赋予每个轨迹更多的学习能力——它可能会过拟合。广义优势估计（GAE）试图将该公式推广为加权多步平均值，而非特定的 $ n $。首先，我们必须定义预测值的时序差（TD）残差。
$$
\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)
$$ {#eq:TD_RESIDUAL}
为此，我们引入另一个变量 $ \lambda $作为广义优势估计（GAE）的混合参数。该参数将我们希望估计的未来优势值以指数衰减形式进行加权融合。
$$
\begin{array}{l}
\hat{A}_t^{GAE(\gamma,\lambda)} = (1-\lambda)(\hat{A}_t^{(1)} + \lambda\hat{A}_t^{(2)} + \lambda^2\hat{A}_t^{(3)} + \cdots) \\
= (1-\lambda)(\delta_t^V + \lambda(\delta_t^V + \gamma\delta_{t+1}^V) + \lambda^2(\delta_t^V + \gamma\delta_{t+1}^V + \gamma^2\delta_{t+2}^V) + \cdots) \\
= (1-\lambda)(\delta_t^V(1 + \lambda + \lambda^2 + \cdots) + \gamma\delta_{t+1}^V(\lambda + \lambda^2 + \cdots) + \cdots) \\
= (1-\lambda)(\delta_t^V\frac{1}{1-\lambda} + \gamma\delta_{t+1}^V\frac{\lambda}{1-\lambda} + \cdots) \\
= \sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l}^V
\end{array}
$$ {#eq:GAE_DFN}
直观而言，这可以优雅地实现对优势函数多步估计的平均化处理。

*更多阅读资料，请参阅[19]。*

### Double Regularization

深度强化学习中许多流行的策略梯度算法源于控制智能体学习过程的需求。在 RLHF（人类反馈强化学习）中，通过相对于微调原始策略的距离惩罚项内置了正则化机制。从这个角度看，像 PPO（具有内部步长正则化）和 REINFORCE（更简单，在特定超参数下 PPO 可退化为该算法）这类算法之间的差异，对于从头训练智能体而言，其重要性远超过对语言模型进行微调的场景。

在 PPO 算法中，用于限制更新步长的目标函数被称为替代目标函数。为了监测 PPO 正则化对 RLHF 更新的影响程度，可以观察许多主流实现中的clip 比例变量——该指标表示批次样本中被 PPO 正则器clip 梯度的百分比。这些梯度会被缩减至预设的最大值。

## 进一步阅读

随着 RLHF（强化学习人类反馈）在现代后训练阶段确立核心地位，其他策略梯度强化学习算法及广义的强化学习算法虽被提出以优化训练流程，但尚未在主导最佳实践中发挥核心作用。值得深入研究的案例包括：

- 成对近端策略优化（P3O）[20] 直接在 PPO 风格策略更新中使用成对数据，无需学习中间奖励模型。
- 离策略策略梯度算法可能实现更异步化的训练，例如对比策略梯度（CoPG）[21]（直接对齐算法 IPO 与标准策略梯度的泛化版本），该算法被 Cohere 公司应用于其 Command A 模型[22]。
- REINFORCE 算法的其他实现方案已针对语言模型进行了专门设计，例如 ReMax[23]，该算法采用了一种基线归一化方法，其设计初衷正是为了适应奖励模型推理过程中的不确定性来源。
- 部分基础模型，如苹果智能基础模型[24]或 Kimi k1.5 推理模型[25]，已采用镜像下降策略优化（MDPO）[26]的变体。该领域的基础研究仍在持续推进[27]，但镜像下降本质上属于优化方法而非直接的政策梯度算法。关键在于其能够以高度兼容的方式替代现有强化学习基础设施。
- **Decoupled Clip and Dynamic sAmpling Policy Optimization (DAPO)**（DAPO）针对推理型语言模型提出了四项 GRPO 改进方案[18]，这些模型需要长轨迹跟踪并提升未充分利用新token的概率。具体改进包括：
  - 1）设置两个不同的裁剪超参数 ϵ_low 和 ϵ_high ，使对数比正侧的裁剪能采取更大步长以促进探索；2）动态采样机制，剔除批次中所有奖励值为 0 或 1 的样本（无学习信号）；
  - 3）采用前文"GRPO 实现"部分讨论的逐token损失计算；
  - 4）对过长样本施加软惩罚，避免从截断答案中学习。
- **Value-based Augmented Proximal Policy Optimization (VAPO)** （VAPO）[28]融合了 DAPO 的优化策略（包括高值裁剪、token级策略梯度及差异化长度归一化）与价值校准 PPO[29]的洞见，通过预训练价值函数和长度自适应 GAE，展现了价值基方法相对于 GRPO 的优势。



## Bibliography

### **1. 强化学习基础与理论**

[4] R. J. Williams, "Simple statistical gradient-following algorithms for connectionist reinforcement learning," *Machine Learning*, vol. 8, pp. 229–256, 1992.
[16] L. Weng, "Policy gradient algorithms," *lilianweng.github.io*, 2018. [Online]. Available: https://lilianweng.github.io/posts/2018-04-08-policy-gradient/
[6] W. Kool, H. van Hoof, and M. Welling, "Buy 4 reinforce samples, get a baseline for free!" *arXiv preprint arXiv:1904.00909*, 2019.
[26] M. Tomar, L. Shani, Y. Efroni, and M. Ghavamzadeh, "Mirror descent policy optimization," *arXiv preprint arXiv:2005.09814*, 2020.

---

### **2. 策略优化算法（PPO及其变体）**
[7] J. Schulman et al., "Proximal policy optimization algorithms," *arXiv preprint arXiv:1707.06347*, 2017.
[3] J. Schulman et al., "High-dimensional continuous control using generalized advantage estimation," *Proc. ICLR*, 2016.
[19] D. Seita, "Notes on the generalized advantage estimation paper," 2017. [Online]. Available: https://danieltakeshi.github.io/2017/04/02/notes-on-the-generalized-advantage-estimation-paper/
[14] J. Schulman, "Approximating KL-divergence," 2016. [Online]. Available: http://joschu.net/blog/kl-approx.html
[20] T. Wu et al., "Pairwise proximal policy optimization: Harnessing relative feedback for LLM alignment," *arXiv preprint arXiv:2310.00212*, 2023.

---

### **3. 大语言模型与人类反馈强化学习（RLHF）**
[1] A. Ahmadian et al., "Back to basics: Revisiting reinforce style optimization for learning from human feedback in LLMs," *arXiv preprint arXiv:2402.14740*, 2024.
[5] S. C. Huang, A. Ahmadian, and C. F. AI, "Putting RL back in RLHF," 2024. [Online]. Available: https://huggingface.co/blog/putting_rl_back_in_rlhf_with_rloo
[13] H. Ivison et al., "Unpacking DPO and PPO: Disentangling best practices for learning from preference feedback," *arXiv preprint arXiv:2406.09279*, 2024.
[15] S. Huang et al., "The n+ implementation details of RLHF with PPO: A case study on TL;DR summarization," *First Conf. Lang. Model.*, 2024. [Online]. Available: https://openreview.net/forum?id=kHO2ZTa8e3
[17] A. Baheti et al., "Leftover lunch: Advantage-based offline reinforcement learning for language models," *arXiv preprint arXiv:2305.14718*, 2023.
[21] Y. Flet-Berliac et al., "Contrastive policy gradient: Aligning LLMs on sequence-level scores," *arXiv preprint arXiv:2406.19185*, 2024.

---

### **4. 大语言模型中的强化学习应用**
[12] D. Guo et al., "Deepseek-r1: Incentivizing reasoning capability in LLMs via reinforcement learning," *arXiv preprint arXiv:2501.12948*, 2025.
[9] Z. Liu et al., "Understanding R1-zero-like training: A critical perspective," *arXiv preprint arXiv:2503.20783*, 2025.
[10] Z. Shao et al., "Deepseekmath: Pushing the limits of mathematical reasoning in open language models," *arXiv preprint arXiv:2402.03300*, 2024.
[11] A. Liu et al., "Deepseek-v3 technical report," *arXiv preprint arXiv:2412.19437*, 2024.
[22] T. Cohere et al., "Command a: An enterprise-ready large language model," *arXiv preprint arXiv:2504.00698*, 2025.
[24] T. Gunter et al., "Apple intelligence foundation language models," *arXiv preprint arXiv:2407.21075*, 2024.
[25] K. Team et al., "Kimi k1.5: Scaling reinforcement learning with LLMs," *arXiv preprint arXiv:2501.12599*, 2025.

---

### **5. 偏好学习与奖励建模**
[2] Z. Wang et al., "HelpSteer2-preference: Complementing ratings with preferences," *arXiv preprint arXiv:2410.01257*, 2024.
[23] Z. Li et al., "Remax: A simple, effective, and efficient reinforcement learning method for aligning large language models," *Proc. ICML*, 2023.

---

### **6. 优化方法与扩展研究**

[8] C. Berner et al., "Dota 2 with large scale deep reinforcement learning," *arXiv preprint arXiv:1912.06680*, 2019.
[18] Q. Yu et al., "DAPO: An open-source LLM reinforcement learning system at scale," 2025.
[27] Y. Zhang et al., "Improving LLM general preference alignment via optimistic online mirror descent," *arXiv preprint arXiv:2502.16852*, 2025.
[28] Y. Yuan et al., "VAPO: Efficient and reliable reinforcement learning for advanced reasoning tasks," *arXiv preprint arXiv:2504.05118*, 2025.
[29] Y. Yuan et al., "What’s behind PPO’s collapse in long-CoT? Value optimization holds the secret," *arXiv preprint arXiv:2503.01491*, 2025.
