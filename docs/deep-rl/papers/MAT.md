# MAT

## 1. 摘要

大序列模型（Sequence Model），如GPT系列和BERT，在视觉、语言和最近执行的学习任务方面表现出突出的性能和泛化能力。一个自然的后续问题是如何将多智能体决策抽象为一个Sequence Model问题，并从Sequence Models的蓬勃发展中获益。在本文中，我们介绍了一种名为  Multi-Agent Transformer (MAT)  的新架构，该架构可以有效地将协作的多智能体强化学习问题（MARL）转换为Sequence Model问题，其中的任务是将智能体的观测序列映射到智能体的最佳动作序列。我们的目标是在MARL和Sequence Models之间建立桥梁，以便为MARL释放现代sequence models的建模能力。MAT的核心是编码器-解码器体系结构，该体系结构利用多智能体优势分解理论将联合策略搜索问题转化为顺序决策过程；这使得多智能体问题的时间复杂度仅为线性，最重要的是，赋予MAT单调的性能改进保证。与以前的技术（如  Decision Transformer）只适合预先收集离线数据不同，MAT以一种基于策略的方式通过在线试错从环境中进行训练。为了验证MAT，我们在StarCraftII、Multi-智能体 MuJoCo、灵巧手操作和Google Research足球基准上进行了大量实验。结果表明，与强基线（包括MAPPO和HAPPO）相比，MAT具有更好的性能和数据效率。此外，我们证明了MAT是一个优秀的 few-short 学习器，无论智能体数量的变化如何。

本文旨在利用Transformer来解决MARL问题，通过将一般的MA决策问题通过多智能体优势分解定理转化为序列决策问题，大大降低了策略搜索的复杂度并且证明了这种序列决策下联合优势函数的单调提升特性，同时Transformer的引入提升了计算效率。实验结果表明本文提出的MA Transformer方法在多个合作式MARL Benchmark上超越了现有的SOTA方法并具备良好的泛化性。

## 2. 引言

多智能体强化学习（MARL）是一个具有挑战性的问题，因为它的困难不仅在于确定每个个体智能体的策略改进方向，还在于将智能体的策略更新联合起来，这对整个团队都是有益的。最近，由于引入了集中训练分散执行的架构（CTDE）[10]，多智能体学习中的这种困难得到了缓解，它允许智能体在训练阶段访问全局信息和对手的动作。该框架能够直接扩展单智能体算法的方法，例如，用COMA 替换策略梯度 (PG) 估计，多智能体 PG（MAPG），MADDPG 将确定性策略梯度扩展到具有集中式批评家的多智能体设置中。QMIX 利用深度 Qnetworks 实现去中心化智能体，并引入集中式混合Q 值分解网络。 MAPPO 赋予所有智能体同一套参数，然后通过信赖域方法进行训练 [46]。然而，这些方法不能涵盖多智能体交互的整体复杂性；事实上，其中一些被证明在最简单的合作任务上失败了[15]。

为了解决这个问题，提出了多智能体优势分解理论[14，定理1]，该理论捕捉了不同智能体对回报的贡献，并通过顺序决策过程提供了合作出现背后的直觉方案。在此基础上，导出了HATRPO和HAPPO算法[14]，由于分解定理和顺序更新方案，这些算法为MARL建立了最先进的新方法。然而，他们的局限性在于智能体的策略不知道开发合作的目的，仍然依赖于精心设计的最大化目标。理想情况下，一个智能体团队应该意识到其设计的训练的联合性，从而遵循一个整体有效的范式，这是一个尚未提出的理想解决方案。

近年来，序列模型（Sequence Model）在自然语言处理（NLP）中取得了长足的进步[18]。例如，基于自回归的GPT系列[3]和BERT模型[8]在广泛的下游任务上表现出了卓越的性能，并在小样本泛化任务上实现了强大的性能。虽然Sequence Model由于其与语言的序列特性的自然拟合而主要用于语言任务，但序列方法不仅限于自然语言处理，而是一种广泛适用的通用基础模型[2]。例如，在计算机视觉（CV）中，可以将图像分割为子图像并按顺序对齐，就像它们是NLP任务中的Token一样[8,9,11]。虽然通过多变量映射求解CV任务的想法很简单，但它是一些性能最佳的CV算法的基础[28,31,29]。此外，最近，Sequence Model 开始产生强大的多模态视觉语言模型，如Flamingo[1]、DALL-E[20]和GATO[22]

随着 Transformer[30]等有效且具有表现力的网络架构的出现，sequence modeling技术也吸引了RL社区的极大关注，这导致了一系列基于Transformer架构的成功的离线RL开发[5,12,22]。这些方法在解决一些最基本的RL training 问题方面显示出巨大的潜力，例如长期信用分配和奖励稀疏性[27]。例如，DecisionTransformer [5]通过纯监督方式对预先收集的离线数据训练自回归模型，绕过了通过动态规划计算累积回报的需要，而是根据预期回报、过去状态和动作生成未来动作。尽管这些方法取得了显著的成功，但没有一种方法被设计用于模拟多智能体系统中最困难（也是MARL特有的）的方面——智能体的交互。事实上，如果我们简单地赋予所有智能体一个Transformer策略并对其进行独立训练，那么它们的联合性能仍然无法保证得到改善[14，命题1]。因此，虽然有无数功能强大的Sequence Models可用，但MARL这一将从Sequence Models中获益匪浅的领域并没有真正利用其性能优势。

### Motivation

Sequence Model（如Transformer）已经在NLP和CV等领域取得了非常好的效果，本文旨在利用Sequence Model（如Transformer）优良的建模能力来解决MARL问题，而通常的MA决策并不具有序列化的特点，因此需要将MA决策（下图左）转化为序列决策问题（如下图右）。在序列决策问题当中，智能体按照一定的顺序进行决策，后决策的智能体能够获取到先决策智能体的动作。实现这种转换的核心是多智能体优势分解定理（Multi-智能体 Advantage Decomposition Theorem）。

从另一方面来看，本文也可以看做是对作者另一篇论文HAPPO（Trust Region Policy Optimisation in Multi-智能体 Reinforcement Learning）的改进。在HAPPO的学习范式中，智能体具有序列决策的特点，但由于智能体的策略更新也必须遵循顺序更新的机制，因此无法实现并行计算，而Transformer的引入正好能解决这一问题。因此，本文可以看做是HAPPO提出的序列化决策的学习范式和Transformer模型之间的结合。

### 我们如何通过序列模型解决MARL问题？

在本文中，我们采取了几个步骤来为上述研究问题提供肯定的答案。我们的目标是用强大的序列建模技术加强MARL研究。为了实现这一点，

- 我们首先提出了一种新的MARL训练范式，该范式建立了合作MARL问题和序列建模问题之间的联系。
- 新范式的核心是多智能体优势分解理论和顺序更新方案，该方案有效地将多智能体联合策略优化转化为顺序策略搜索过程。
- 作为我们发现的自然结果，我们介绍了Multi -Agent Transformer（MAT），一种编码器-解码器架构，通过Sequence Models实现通用MARL解决方案。
- 与Dicision Transformer[5]不同，MAT是基于策略上的 trilas and  errors 在线训练的；因此，它不需要预先收集序列的轨迹。
- 重要的是，多智能体优势分解定理的实现确保了MAT在训练期间享受单调的性能改进保证。
- MAT为合作MARL任务建立了一个新的最先进的基线模型。
- 我们通过在StarCraftII、多智能体MuJoCo、灵巧手操作和Google Research Football 的基准上评估MAT来证明这种说法的合理性；结果表明，MAT的性能优于基线，如MAPPO[33]、HAPPO[14]和QMIX[21]。最后，我们证明了MAT在任务泛化中具有很大的潜力，它在新任务中保持了智能体数的任意性。

## 3. Preliminaries

在本节中，我们首先介绍协作 MARL 问题公式和多智能体优势分解定理，这是我们工作的基石。然后我们回顾与 MAT 相关的现有 MARL 方法，最后让读者熟悉 Transformer。

### 3.1 问题定义

合作 MARL 问题通常由马尔可夫游戏建模 $\langle\mathcal{N}, \mathcal{O}, \mathcal{A}, R, P, \gamma\rangle[19] . \mathcal{N}=$ $\{1, \ldots, n\}$ 是智能体集合 ,

- $\mathcal{O}=\prod_{i=1}^{n} \mathcal{O}^{i}$ : 是局部观测空间的乘积，即联合观测空间，
- $\mathcal{A}=\prod_{i=1}^{n} \mathcal{A}^{i}$ : 是 Agent 动作的乘积空间，即联合动作空间，
- $R:\mathcal{O} \times \mathcal{A} \rightarrow\left[-R_{\max }, R_{\max }\right]$ : 是联合奖励函数，
- $P: \mathcal{O} \times \mathcal{A} \times \mathcal{O} \rightarrow \mathbb{R}$  :   是转移概率函数，并且 $\gamma \in[0,1)$ 是折扣因子。
- 在时间步长 $t \in \mathbb{N}$,  智能体 $i \in \mathcal{N}$ 观测到一个观测$\mathrm{o}_{t}^{i} \in \mathcal{O}^{i(2)}\left(\boldsymbol{o}=\left(o^{1}, \ldots, o^{n}\right)\right.$) 是一个“联合”观测 并根据其策略 $\pi^{i}$,采取动作 $a_{t}^{i}$. 这是智能体 agents' 的联合策略 $\pi$ 的 $i^{\text {th }}$组成部分.
- 在每个时间步，所有智能体根据他们的观测同时采取动作没有顺序依赖性。转移Kernel $P$ 和联合策略导致（不适当的）边际观测分布  $\rho_{\boldsymbol{\pi}}(\cdot) \triangleq \sum_{t=0}^{\infty} \gamma^{t} \operatorname{Pr}\left(\mathbf{o}_{t}=\boldsymbol{o} \mid \boldsymbol{\pi}\right)$.
- 在每个时间步结束时，整个团队获得共同奖励 $R\left(\mathbf{o}_{t}, \mathbf{a}_{t}\right)$ 并观测 $\mathbf{o}_{t+1}$, 其概率分布为 $P\left(\cdot \mid \mathbf{o}_{t}, \mathbf{a}_{t}\right)$.   遵循这个过程，Agents 获得累积折扣回报  $R^{\gamma} \triangleq \sum_{t=0}^{\infty} \gamma^{t} R\left(\mathbf{o}_{t}, \mathbf{a}_{t}\right)$

### 3. 2.多智能体优势分解定理

智能体评估动作和观测的价值 $Q_{\boldsymbol{\pi}}(\boldsymbol{o}, \boldsymbol{a})$ 和 $V_{\boldsymbol{\pi}}(\boldsymbol{o})$， 定义为

$$
\begin{gathered}
Q_{\boldsymbol{\pi}}(\boldsymbol{o}, \boldsymbol{a}) \triangleq \mathbb{E}_{\mathbf{o}_{1: \infty} \sim P, \mathbf{a}_{1: \infty} \sim \boldsymbol{\pi}}\left[R^{\gamma} \mid \mathbf{o}_{0}=\boldsymbol{o}, \mathbf{a}_{0}=\boldsymbol{a}\right], \\
V_{\boldsymbol{\pi}}(\boldsymbol{o}) \triangleq \mathbb{E}_{\mathbf{a}_{0} \sim \boldsymbol{\pi}, \mathbf{o}_{1: \infty} \sim P, \mathbf{a}_{1: \infty} \sim \boldsymbol{\pi}}\left[R^{\gamma} \mid \mathbf{o}_{0}=\boldsymbol{o}\right] .
\end{gathered}
$$

联合目标导致了在收到共享奖励后与信用分配问题相关的困难，单个智能体无法推断出他们自己的贡献团队的成败[4]。事实上，应用传统的 RL 方法，简单地使用上述价值函数会导致训练障碍，例如多智能体方差的增长策略梯度 (MAPG) 估计 [17]。因此，为了解决这些问题，局部价值函数的概念 [21]和反事实基线 [11] 方法被提出。在本文中，我们使用最一般的这种概念 - 多智能体观测值函数 [15]。也就是说，对于任意不相交的智能体的有序子集 $i_{1: m}=\left\{i_{1}, \ldots, i_{m}\right\}$ 和 $j_{1: h}=\left\{j_{1}, \ldots, j_{h}\right\}$, 对于 $m, h \leq n$,我们定义多智能体观测值函数：
$$
Q_{\boldsymbol{\pi}}\left(\boldsymbol{o}, \boldsymbol{a}^{i_{1: m}}\right) \triangleq \mathbb{E}\left[R^{\gamma} \mid \mathbf{o}_{0}^{i_{1: n}}=\boldsymbol{o}, \mathbf{a}_{0}^{i_{1: m}}=\boldsymbol{a}^{i_{1: m}}\right],
$$

它恢复了等式（1）中的原始状态-动作价值函数当 $m=n$, 和原来的观测值函数当 $m=0$ 。基于等式（2），我们可以进一步衡量选定的智能体子集对联合回报的贡献，并定义多智能体优势函数：
$$
A_{\boldsymbol{\pi}}^{i_{1: m}}\left(\boldsymbol{o}, \boldsymbol{a}^{j_{1: h}}, \boldsymbol{a}^{i_{1: m}}\right) \triangleq Q_{\boldsymbol{\pi}}^{j_{1: h}, i_{1: m}}\left(\boldsymbol{o}, \boldsymbol{a}^{j_{1: h}}, \boldsymbol{a}^{i_{1: m}}\right)-Q_{\boldsymbol{\pi}}^{j_{1: h}}\left(\boldsymbol{o}, \boldsymbol{a}^{j_{1: h}}\right) .
$$

上面的公式描述了，如果智能体$i_{1: m}$采取动作$\boldsymbol{a}^{i_{1: m}}$， $j_{1: h}$采取动作$\boldsymbol{a}^{j_{1: h}}$情况下， 联合动作 $\boldsymbol{a}$ 比平均水平好/差的值。再一次，当 $h=0$, 优势比较值$\boldsymbol{a}^{i_{1: m}}$到整个团队的基线价值函数。 这个值函数表示智能体的动作可以研究他们之间的相互作用，以及分解联合价值函数信号，从而有助于缓解信用分配问题的严重性。

${ }^{(2)}$ 为了符号方便，我们省略了定义以全局状态为输入和输出每个智能体的局部观测，而是直接定义智能体的局部观测。

**定理1 (多智能体优势分解 [17])** .

让 $i_{1: n}$ 作为智能体的排列. 然后，对于任何联合观测 $\boldsymbol{o}=\boldsymbol{o} \in \mathcal{O}$ 和联合动作$\boldsymbol{a}=\boldsymbol{a}^{i_{1: n}} \in \mathcal{A}$, 下面的等式无需进一步假设总是成立
$$
A_{\boldsymbol{\pi}}^{i_{1: n}}\left(\boldsymbol{o}, \boldsymbol{a}^{i_{1: n}}\right)=\sum_{m=1}^{n} A_{\boldsymbol{\pi}}^{i_{m}}\left(\boldsymbol{o}, \boldsymbol{a}^{i_{1: m-1}}, a^{i_{m}}\right)
$$

重要的是，这个定理提供了一种直觉来指导选择渐进式改进的动作。假设智能体 $i_{1}$选择一个动作$a^{i_{1}}$ 具有积极的优势 $A_{\boldsymbol{\pi}}^{i_{1}}\left(\boldsymbol{o}, a^{i_{1}}\right)>0$. 然后，想象一下对于所有的$j=2, \ldots, n$, 智能体  $i_{j}$ 知道联合动作 $\boldsymbol{a}^{i_{1: j-1}}$ 是它前一个动作. 在这种情况下，它可以选择一个动作 $a^{i_{j}}$ 从而得到的优势 $A_{\boldsymbol{\pi}}^{i_{j}}\left(\boldsymbol{o}, \boldsymbol{a}^{i_{1: j-1}}, a^{i_{j}}\right)$ 是积极的.  总而言之，定理确保联合动作 $\boldsymbol{a}^{i_{1: n}}$ 有积极的优势。此外，请注意联合已选择动作 $n$ 步, 每个步骤搜索一个个体智能体的动作空间。

因此，在动作空间中， 此搜索的复杂性是累加的, $\sum_{i=1}^{n}\left|\mathcal{A}^{i}\right|$.  如果我们要直接在联合动作空间中执行搜索，我们将浏览一组乘法大小 $|\mathcal{A}|=\prod_{i=1}^{n}\left|\mathcal{A}^{i}\right|$ 的动作。 稍后，我们将基于这种见解来设计一个 SM 模型来优化联合策略，逐个智能体，无需立即考虑联合行动空间。

### 3. 3. MARL 中的现有方法

我们现在简要总结两种最先进的 MARL 算法。它们都建立在 近端策略优化 (PPO) 之上—一种以其简单性和性能稳定性而闻名的 RL 方法。MAPPO [46] 是第一个也是最直接的在 MARL 中应用 PPO 的方法。它等同地将所有智能体共享一套参数，并将智能体的聚合轨迹用于共享策略的更新; 在迭代 $k+1$步,  它通过最大化裁剪目标优化了策略参数 $\theta_{k+1}$
$$
\sum_{i=1}^{n} \mathbb{E}_{\mathbf{o} \sim \rho_{\boldsymbol{\pi}_{\theta_{k}}}, \mathbf{a} \sim \boldsymbol{\pi}_{\theta_{k}}}\left[\min \left(\frac{\pi_{\theta}\left(\mathrm{a}^{i} \mid \mathbf{o}\right)}{\pi_{\theta_{k}}\left(\mathrm{a}^{i} \mid \mathbf{o}\right)} A_{\boldsymbol{\pi}_{\theta_{k}}}(\mathbf{o}, \mathbf{a}), \operatorname{clip}\left(\frac{\pi_{\theta}\left(\mathrm{a}^{i} \mid \mathbf{o}\right)}{\pi_{\theta_{k}}\left(\mathrm{a}^{i} \mid \mathbf{o}\right)}, 1 \pm \epsilon\right) A_{\boldsymbol{\pi}_{\theta_{k}}}(\mathbf{o}, \mathbf{a})\right)\right]
$$

裁剪运算符裁剪输入值（如有必要），使其保持在区间$[1-\epsilon, 1+\epsilon]$内。但是，强制参数共享等同于施加约束 $\theta^{i}=\theta^{j}, \forall i, j \in \mathcal{N}$ 在联合策略空间，这可能导致指数级恶化的次优结果 [15]。这个激发了异构智能体信赖域方法的更有原则的发展，例如 HAPPO。

HAPPO[15]是目前充分利用定理（1）实现的SOTA算法之一， 具有单调改进保证的多智能体信赖域学习。在更新期间，智能体随机选择一个排列 $i_{1: n}$ at random, 然后按照排列中的顺序，每一个智能体  $i_{m}$ 挑选 $\pi_{\text {new }}^{i_{m}}=\pi^{i_{m}}$ 最大化的目标：
$$
\mathbb{E}_{\mathbf{o} \sim \rho_{\boldsymbol{\pi}_{\text {old }}}, \mathbf{a}^{i_{1: m-1} \sim \pi_{\text {new }}} i_{1: m-1}, \mathrm{a}^{i} m \sim \pi_{\text {old }}^{i m}}^{i_{\text {of }}}\left[\min \left(\mathrm{r}\left(\pi^{i_{m}}\right) A_{\boldsymbol{\pi}_{\text {old }}}^{i_{1: m}}\left(\boldsymbol{o}, \mathbf{a}^{i_{1: m}}\right), \operatorname{clip}\left(\mathrm{r}\left(\pi^{i_{m}}\right), 1 \pm \epsilon\right) A_{\pi_{\text {old }}}^{i_{1: m}}\left(\mathbf{o}, \mathbf{a}^{i_{1: m}}\right)\right)\right],
$$

其中$\mathrm{r}\left(\pi^{i_{m}}\right)=\pi^{i_{m}}\left(\mathrm{a}^{i_{m}} \mid \mathbf{o}\right) / \pi_{\text {old }}^{i_{m}}\left(\mathrm{a}^{i_{m}} \mid \mathbf{o}\right)$. 请注意，期望是计算的新更新的以前智能体的策略, 即 $\boldsymbol{\pi}_{\mathrm{new}}^{i_{1: m-1}}$; 这反映了一种直觉，即根据定理 (1)，智能体 $i_{m}$ 对其先前的智能体作出反应 $i_{1: m-1}$.

### 3. 4 Transformer 模型

Transformer [40] 最初是为机器翻译任务设计的（例如，输入英文，输出法语）。它维护一个编码器-解码器结构，其中编码器映射一个输入序列标记到潜在表示，然后解码器生成一系列所需的输出一种自回归方式，其中在推理的每个步骤中，Transformer 都采用之前的所有生成的Token作为输入。Transformer 中最重要的组件之一是缩放点积注意力，捕捉输入序列的相互关系。注意力函数写成$\operatorname{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V})=\operatorname{softmax}\left(\frac{\mathbf{Q K}}{\sqrt{d_{k}}}\right) \mathbf{V}$, 其中 $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ 对应于向量可以在训练期间学习的查询、键和值的集合。以及$d_{k}$代表$\mathbf{Q}$ 和 $\mathbf{K}$ 的维度。自注意是指 $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ 共享同一组参数。

<div align=center><img src="deep-rl/papers/assets/mat1.png" style="zoom:80%;" /></div>


图 1：传统的多智能体学习范式（左），其中所有智能体同时采取行动，多智能体顺序决策范式（右），其中智能体通过以下方式采取行动一个连续的顺序，每个智能体人都负责前面智能体人的决定，如红色箭头所示。

受注意力机制的启发，UPDeT [13] 通过将每个智能体的观测解耦为一系列观测实体，将它们与不同的动作相匹配分组，并使用 Transformer 对匹配的观测实体之间的关系进行建模以便更好地表示来处理各种观测大小在 MARL 问题中更好地表示学习。除此之外，基于定理（1）中描述的顺序属性和 HAPPO [15] 背后的原理，很直观考虑另一种基于 Transformer 的多智能体信赖域学习实现。通过将智能体团队视为一个序列，Transformer 架构我们对具有可变数量和类型的智能体团队进行建模，同时避免 MAPPO/HAPPO 的缺点。 我们将更详细地描述如何通过序列模型解决协作 MARL 问题。

## 4. MARL和序列模型之间惊人的联系

为了建立MARL和序列模型之间的联系，定理（1）提供了从SM角度理解MARL问题的新角度。如果每个智能体都知道其前辈具有任意决策顺序动作，则智能体的局部优势之和$A_{\boldsymbol{\pi}}^{i_{j}}\left(\boldsymbol{o}, \boldsymbol{a}^{i_{1: m-1}}, a^{i_{m}}\right)$ 将完全等于联合优势$A_{\pi: n}^{i_{1: n}}\left(\boldsymbol{o}, \boldsymbol{a}^{i_{1: n}}\right)$。这种跨智能体的有序决策设置简化了其联合策略的更新，其中最大化每个智能体自身的局部优势相当于最大化联合优势。因此，在策略更新期间，智能体不再需要担心来自其他智能体的干扰；局部优势函数已经捕捉到了智能体之间的关系。定理（1）揭示的这一特性启发我们为MARL问题提出一种多智能体顺序决策范式，如图（1）所示，其中我们以任意决策顺序分配智能体（每个迭代一个排列）；每个智能体都可以访问其前辈的动作，并在此基础上做出最优决策。这种顺序范式促使我们利用顺序模型，例如Transformer，来明确捕获定理（1）中描述的智能体之间的顺序关系。

在定理（1）的支持下，序列建模减少了MARL问题随着智能体数量从乘法增加到加法的复杂性增长，从而呈现线性复杂性。借助Transformer体系结构，我们可以用统一的网络对异构智能体的策略进行建模，但可以用不同的位置区分对待每个智能体，从而确保高采样效率，同时避免MAPPO面临的指数级恶化结果。此外，为了保证联合策略的单调改进，HAPPO必须在训练过程中更新每个策略，通过利用先前的更新结果$\pi^{i_{1}}, \ldots, \pi^{i_{m-1}}$  改善 $\pi^{i_{m}}$，这对于大型智能体的计算效率至关重要。

相比之下，Transformer架构允许并行训练序列策略，类似于机器翻译任务，这显著提高了训练速度，并使其适用于大量智能体。此外，在智能体数量和类型不同的情况下，SM可以通过其对具有灵活序列长度的序列建模的能力将它们合并到一个统一的解决方案中，而不是将不同的智能体数量视为不同的任务。为了实现上述思想，我们将在下一节介绍一种实用的体系结构，名为Multi-agent-Transformer.

<div align=center><img src="deep-rl/papers/assets/mat2.png" style="zoom:80%;" /></div>

图 2：MAT 的编码器-解码器架构。在每个时间步，编码器接收一系列agents 的观测并将它们编码成一系列潜在表示，然后传递到解码器。解码器以顺序和自回归的方式生成每个智能体的最佳动作。掩码注意力块确保智能体在训练期间只能访问其前面的智能体的操作。我们在附录 A 中列出了 MAT 的完整伪代码，并在中列出了显示 MAT 动态数据流的视频 https://sites.google.com/view/multi-agent-transformer。

## 5. Multi-agent-Transformer.

为了实现 MARL 的序列建模范式，我们的解决方案是 Multi-Agent Transformer。应用 Transformer 架构的想法来自于映射在智能体观测序列的输入之间$\left(o^{i_{1}}, \ldots, o^{i_{n}}\right)$ 以及智能体动作的输出顺序$\left(a^{i_{1}}, \ldots, a^{i_{n}}\right)$ 是类似于机器翻译的序列建模任务。正如定理1所描述的那样， 动作 $a^{i_{m}}$ 取决于 之前所有智能体的决定 $\boldsymbol{a}^{i_{1: m-1}}$。因此，我们的 MAT图（2）中包含一个编码器，它学习联合观测的表示，以及一个解码器以自动回归的方式为每个单独的智能体输出动作。

编码器，其参数我们用$\phi$ 表示， 在任意顺序采取一系列观测 $\left(o^{i_{1}}, \ldots, o^{i_{n}}\right)$ 并通过几个计算块传递它们。每个这样的块由一个自注意机制和多层感知器（MLP），以及防止随着深度的增加梯度消失和网络退化的残差块组成。我们表示输出将观测结果编码为$\left(\hat{\boldsymbol{o}}^{i_{1}}, \ldots, \hat{\boldsymbol{o}}^{i_{n}}\right)$, 它不仅编码智能体的信息 $\left(i_{1}, \ldots, i_{n}\right)$ 还有代表智能体交互的高级相互关系。为了学习表达表示，在训练阶段，我们让编码器逼近值函数，其目标是通过以下方式最小化经验贝尔曼误差 ：
$$
L_{\text {Encoder }}(\phi)=\frac{1}{T n} \sum_{m=1}^{n} \sum_{t=0}^{T-1}\left[R\left(\mathbf{o}_{t}, \mathbf{a}_{t}\right)+\gamma V_{\bar{\phi}}\left(\hat{\mathbf{o}}_{t+1}^{i_{m}}\right)-V_{\phi}\left(\hat{\mathbf{o}}_{t}^{i_{m}}\right)\right]^{2}
$$
其中 $\bar{\phi}$  目标网络的参数 , 它是不可微分的并且每隔几个Epoch更新一次。解码器，其参数我们用 $\theta$ 表示，通过嵌入式联合行动 $\boldsymbol{a}^{i_{0: m-1}}, m=$ $\{1, \ldots n\}$  （其中 $a^{i_{0}}$是指示解码开始的任意符号）到解码序列块。至关重要的是，每个解码块都带有一个带掩码的自我注意机制，其中掩码确保对于每一个 $i_{j}$,注意力仅在  $i_{r}^{\text {th }}$ 和 $i_{j}^{\text {th }}$ 行动，其中 $r<j$ 以便可以维持顺序更新方案。然后是通过第二个掩蔽注意力函数，它计算动作头和动作头之间的注意力观测。最后，该块以 MLP 和跳连结束。这输出到最后一个解码器块的是一系列联合动作的表示， $\left\{\hat{\boldsymbol{a}}^{i_{0}: i-1}\right\}_{i=1}^{m}$.  这被馈送到输出概率分布的第  $i_{m}$个 动作， 即策略 $\pi_{\theta}^{i_{m}}\left(\mathrm{a}^{i_{m}} \mid \hat{\mathbf{o}}^{i_{1: n}}, \mathbf{a}^{i_{1: m-1}}\right)$. 为了训练解码器，我们最小化以下剪裁 PPO 目标：
$$
\begin{aligned}
L_{\text {Decoder }}(\theta) & =-\frac{1}{T n} \sum_{m=1}^{n} \sum_{t=0}^{T-1} \min \left(\mathrm{r}_{t}^{i_{m}}(\theta) \hat{A}_{t}, \operatorname{clip}\left(\mathrm{r}_{t}^{i_{m}}(\theta), 1 \pm \epsilon\right) \hat{A}_{t}\right), \\
\mathrm{r}_{t}^{i_{m}}(\theta) & =\frac{\pi_{\theta}^{i_{m}}\left(\mathrm{a}_{t}^{i_{m}} \mid \hat{\mathbf{o}}_{t}^{i_{1: n}}, \hat{\mathbf{a}}_{t}^{i_{1: m-1}}\right)}{\pi_{\theta_{\text {old }}}^{i_{m}}\left(\mathrm{a}_{t}^{i_{m}} \mid \hat{\mathbf{o}}_{t}^{i_{1: n}}, \hat{\mathbf{a}}_{t}^{i_{1: m-1}}\right)},
\end{aligned}
$$

其中 $\hat{A}_{t}$ 是联合优势函数的估计。可以应用广义优势估计（GAE）与$\hat{V}_{t}=\frac{1}{n} \sum_{m=1}^{n} V\left(\hat{o}_{t}^{i_{m}}\right)$ 作为联合价值函数的稳健估计。值得注意的是，动作生成过程在推理和训练阶段是不同的。在在推理阶段，每个动作都是自回归生成的，在某种意义上$a^{i_{m}}$ 将被插入再次回到解码器生成 $\mathrm{a}^{i_{m+1}}$ (从  $\mathrm{a}^{i_{0}}$ 开始并以 $\mathrm{a}^{i_{n-1}}$ 结束)。而在训练阶段，所有动作的输出， $\mathbf{a}^{i_{1: n}}$  可以并行计算，因为 $\mathbf{a}^{i_{1: n-1}}$已经被收集并存储在回放缓冲区中。

位于 MAT 核心的注意力机制对观察和动作进行编码通过乘以嵌入式查询计算出的权重矩阵，$\left(q^{i_{1}}, \ldots, q^{i_{n}}\right)$, 和  $\left(k^{i_{1}}, \ldots, k^{i_{n}}\right)$, 其中每个权重 $w\left(q^{i_{r}}, k^{i_{j}}\right)=\left\langle q^{i_{r}}, k^{i_{j}}\right\rangle$ 编码的值 $\left(v^{i_{1}}, \ldots, v^{i_{n}}\right)$ 与权重矩阵相乘以输出表示。 编码器中的 masked attention 使用全权重矩阵来提取之间的相互关系，智能体 $\hat{\mathbf{o}}^{i_{1: n}}$, 解码器捕获中的掩蔽注意力$\mathbf{a}^{i_{1: m}}$  适当掩饰注意力机制，解码器可以安全输出策略 $\pi_{\theta}^{i_{m+1}}\left(\mathbf{a}^{i_{m+1}} \mid \hat{\mathbf{o}}^{i_{1: n}}, \mathbf{a}^{i_{1: m}}\right)$, 完成了定理一的实现。

单调改进保证。一个 MAT 智能体 $i_{m}$ 优化信赖域目标， 该目标以智能体$i_{1: m-1}$的新决定为条件， 方法是通过调节其对智能体的策略比率（见等式（5）） 因此，它单调地增加联合回报，就像它遵循 HAPPO [15，定理 2] 的顺序更新方案一样。 然而，与该方法相反，MAT 模型不需要 $i_{m}$  等待其前辈进行更新，也不需要使用其更新后的动作分布进行重要性采样计算。 事实上，由于所有智能体的动作都是 MAT 的输出，它们的裁剪目标可以并行计算（在训练期间）。

HAPPO 关于时间复杂度。最后，为了确保限制联合策略是这样的智能体被激励改变其策略（纳什均衡），MAT 需要排列每次迭代更新的顺序，这与 HAPPO [15，定理3]。


## 6. 实验和结果

MAT 为协作 MARL 问题提供了一种新的解决方案范例。MAT 的关键见解是受定理（1）启发的顺序更新方案，以及编码器-解码器体系结构，它为序列建模视角提供了高效的实现。重要的是，MAT继承了单调改进保证，agent的策略可以学习在训练期间并行。我们坚信 MAT 将成为 MARL 研究的游戏规则改变者。

为了评估 MAT 是否符合我们的期望，我们在星际争霸 II 多智能体挑战赛上测试了 MAT(SMAC) 基准测试 [31]，其中具有参数共享 [46] 的 MAPPO 显示出卓越的性能，和 Multi-Agent MuJoCo 基准测试 [7]，其中 HAPPO [15] 显示了当前最先进的技术表现。SMAC 和 MuJoCo 环境是 MARL 领域的常见基准。在除此之外，我们还在双手灵巧手操作 (Bi-DexHands) [6] 上测试了 MAT其中提供了具有挑战性的双手操作任务列表（见图（3）），以及谷歌Research Football [18] 以足球比赛中的一系列合作场景为基准。

我们从他们的原始论文中应用相同的基线算法的超参数，以确保他们最佳性能，并为我们的方法采用相同的超参数调整过程，并提供详细信息在附录 B 中。为了确保与 CTDE 方法的公平比较，我们还引入了 CTDE 变体MAT 称为 MAT-Dec，它基本上为每个人采用完全去中心化的参与者智能体（而不是使用 MAT 中提出的解码器），同时保持编码器固定。

图 6：使用图 (3a) 所示的不同禁用关节执行 HalfCheetah 任务的性能。

评论家的 loss for MAT-Dec is $L(\phi)=\frac{1}{T} \sum_{t=0}^{T-1}\left[R\left(\mathbf{o}_{t}, \mathbf{a}_{t}\right)+\gamma \frac{1}{n} \sum_{m=1}^{n} V_{\bar{\phi}}\left(\hat{\mathbf{o}}_{t+1}^{i_{m}}\right)-\frac{1}{n} \sum_{m=1}^{n} V_{\phi}\left(\hat{\mathbf{o}}_{t}^{i_{m}}\right)\right]^{2}$,

我们应用局部优势估计$A_{t}\left(\hat{\mathrm{o}}_{t}^{i_{m}}, a^{i_{m}}\right)$指导后续的政策更新。
