# 🚀 一行代码的复兴：用强化学习视角重塑监督微调（SFT）的泛化能力



在大型语言模型（LLM）的“对齐”（Alignment）领域，核心目标是让模型能更好地理解并遵循人类指令。目前，主流技术路径分为两条：**监督微调（SFT）** 和 **基于人类反馈的强化学习（RLHF）**。

SFT 简单高效，擅长快速学习特定任务的模式（“记忆”）；而 RLHF 虽然复杂，但在处理新问题时展现出强大的泛化能力。长期以来，业界公认的观点是：“SFT 负责记忆，RL 负责泛化”。

然而，一篇来自东南大学、加州大学洛杉矶分校等机构的最新研究论文《On the Generalization of SFT: A Reinforcement Learning Perspective with Reward Rectification》颠覆了这一认知。研究者们不仅从理论上揭示了 SFT 泛化不足的根源，更提出了一种名为 **动态微调（Dynamic Fine-Tuning, DFT）**的简单优化方法。

惊人的是，这个方法的核心改动**仅仅是一行代码**，却在多个极具挑战性的数学推理任务上显著超越了标准 SFT，其性能提升甚至可以与更复杂的 RL 方法相媲美。

本文将带您深入解读这篇论文，探究 DFT 是如何用“一行代码”撬动 SFT 的性能，实现其“文艺复兴”的。

> 论文标题：On the Generalization of SFT: A Reinforcement Learning Perspective with Reward Rectification
>
> 论文链接：https://arxiv.org/pdf/2508.05629

## 一、 SFT 的“隐痛”：为何在泛化上输给强化学习？

要解决 SFT 泛化能力差的问题，首先必须找到其根源。论文的核心洞察，在于首次从**强化学习（RL）**的视角对 SFT 进行了重构和剖析。

### 1. 传统的 SFT 优化目标

SFT 的目标是最小化模型在专家示范数据上的负对数似然（即交叉熵损失）。其目标函数如下：

$$
\mathcal{L}_{\text{SFT}}(\theta) = -\mathbb{E}_{(x, y) \sim \mathcal{D}_{\text{expert}}} \left[ \log P_{\theta}(y | x) \right]
$$

其中，$x$ 是输入（问题），$y$ 是专家给出的标准答案，$P_{\theta}(y | x)$ 是模型生成标准答案的概率。

### 2. SFT 梯度与策略梯度的等价性

强化学习中的**策略梯度（Policy Gradient）**的通用形式是：

$$
\nabla \mathcal{L}_{\text{PG}}(\theta) \approx \mathbb{E}_{\tau \sim P_{\theta}(\tau)} \left[ \nabla \log P_{\theta}(\tau) \cdot R(\tau) \right]
$$

它表示，梯度更新的方向是让获得高奖励 $R(\tau)$ 的动作序列 $\tau$ 的概率变大。

通过引入 **重要性采样（Importance Sampling）**这一数学技巧，论文作者们巧妙地将 SFT 的梯度表达式改写成了一种与策略梯度极其相似的形式：

$$
\nabla \mathcal{L}_{\text{SFT}}(\theta) \approx \mathbb{E}_{y \sim P_{\theta}(y|x)} \left[ \nabla \log P_{\theta}(y|x) \cdot \mathbb{I}(y=y_{\text{expert}}) \cdot \frac{P_{\text{expert}}(y|x)}{P_{\theta}(y|x)} \right]
$$

> **注：** $\mathbb{I}(y=y_{\text{expert}})$ 是示性函数，只有当模型答案 $y$ 与专家答案 $y_{\text{expert}}$ 完全相同时，其值才为 1，否则为 0。

这一重构揭示了 SFT 学习动态的**两大缺陷**：

#### 缺陷一：隐式奖励极其稀疏

SFT 的隐式奖励函数是 $R_{\text{SFT}}(y) = \mathbb{I}(y=y_{\text{expert}})$。

这意味着，模型只有在生成与标准答案**一字不差**时才能获得奖励 1，否则奖励为 0。这种“非黑即白”的机制过于严苛，无法对“部分正确”或“有价值”的回答给予鼓励，极大地限制了模型的探索和泛化能力。

#### 缺陷二：病态的重要性权重

SFT 的梯度中存在一项权重 $\frac{P_{\text{expert}}(y|x)}{P_{\theta}(y|x)}$，在 SFT 场景中，通常简化为 $\frac{1}{P_{\theta}(y|x)}$（假设专家数据是均匀采样的）。

这个权重是导致 SFT 泛化能力差的**罪魁祸首**。当模型对于某个正确的专家答案 $y_{\text{expert}}$ 分配的概率 $P_{\theta}(y|x)$ 很低时（即模型认为这个答案很“困难”或很“冷门”），其倒数就会变得**非常大**。

- **后果：** 这会导致梯度更新的方差极大，优化过程**极其不稳定**，可能引发梯度爆炸。
- **实质：** 模型会不成比例地过度关注那些罕见的、难以拟合的样本，试图“死记硬背”下来，而忽略了数据中普遍规律的学习，最终导致**严重过拟合**。

## 二、 DFT 的“神来之笔”：动态奖励修正

既然问题出在那个病态的重要性权重 $\frac{1}{P_{\theta}(y|x)}$ 上，那么最简单直接的解决办法就是想办法**修正**或**消除**它。

### 1. 核心思想：抵消病态权重

论文提出的 **动态微调（DFT）**方法的思路优雅而直接：在计算梯度时，直接给原始的 SFT 损失乘上一个修正项，这个修正项恰好就是模型自身的概率 $P_{\theta}(y|x)$。

修正后的 DFT 梯度形式为：

$$
\nabla \mathcal{L}_{\text{DFT}}(\theta) \approx \mathbb{E}_{y \sim P_{\theta}(y|x)} \left[ \nabla \log P_{\theta}(y|x) \cdot \mathbb{I}(y=y_{\text{expert}}) \cdot \underbrace{P_{\theta}(y|x)}_{\text{修正项}} \right]
$$

这里的修正项使用了 $\text{stop-gradient}$ 算子（在公式中简化为直接相乘并假定 $P_{\theta}(y|x)$ 不参与反向传播），意味着在反向传播计算梯度时，不对这个修正项求导，只把它当作一个**动态调整的权重**。

### 2. DFT 损失函数与实践中的实现

这个修正后的梯度对应到一个简单的重加权损失（reweighted loss）。在实践中，为了计算稳定和实用，这个操作是在**词元（token）级别**进行的：

$$
\mathcal{L}_{\text{DFT}} = \sum_{t=1}^{L} - \text{stop-gradient}\left(P_{\theta}(y_t|x, y_{<t})\right) \cdot \log P_{\theta}(y_t|x, y_{<t})
$$

其核心的伪代码改动如下：

```python
# 传统的SFT损失（伪代码）
# log_prob = log_softmax(logits)[target_token_id]
# sft_loss = -log_prob

# DFT损失（伪代码）
# 1. 计算当前token的预测概率
token_prob = softmax(logits)[target_token_id]
# 2. 计算当前token的负对数似然（SFT损失）
token_log_prob = log_softmax(logits)[target_token_id]
# 3. 核心改动：在SFT损失前，乘以该token的概率（并停止梯度）
dft_loss = -stop_gradient(token_prob) * token_log_prob
```

### 3. DFT 的“学习哲学”：稳扎稳打

这个简单的改动，彻底改变了 SFT 的学习动态：

| **机制**                                  | **传统的 SFT (交叉熵)**                                 | **动态微调 (DFT)**                                           |
| ----------------------------------------- | ------------------------------------------------------- | ------------------------------------------------------------ |
| **权重**                                  | $\frac{1}{P_{\theta}(y | x)}$                           | $\mathbb{I}(y=y_{\text{expert}}) \cdot \underbrace{P_{\theta}(y|x)}_{\text{修正项}}$ |
| **对困难样本的影响** ($P_{\theta} \to 0$) | 权重爆炸 ($W \to \infty$)，**梯度不稳定**，引发过拟合。 | 权重趋近 0 ($W \to 0$)，**损失被降权**，避免过度拟合。       |
| **对简单样本的影响** ($P_{\theta} \to 1$) | 权重趋近 1 ($W \to 1$)，损失权重保持不变。              | 权重趋近 1 ($W \to 1$)，损失权重保持不变。                   |
| **哲学**                                  | 越不自信，越要下猛药。                                  | 稳扎稳打，优先学习有把握的知识。                             |

通过将奖励修正为 $R'_{\text{DFT}} = \mathbb{I}(y=y_{\text{expert}}) \cdot P_{\theta}(y|x)$，DFT 极大地**平滑了奖励机制**，模型不再被极低概率的样本所挟持，从而避免了梯度爆炸和过拟合，促进了模型向更稳定、泛化能力更强的方向优化。

## 三、 DFT 与 Focal Loss 的反向对比

值得一提的是，DFT 的设计哲学与计算机视觉领域著名的 **Focal Loss** 形成了有趣的对比。

- **Focal Loss：** $\mathcal{L}_{\text{Focal}} \propto -(1 - P_{\theta})^{\gamma} \log P_{\theta}$。它通过 $\propto (1 - P_{\theta})^{\gamma}$ 因子，**降低**模型对已经很自信（$P_{\theta}$ 很大）的“简单样本”的权重，让模型更专注于学习“困难样本”。其目标是解决**类别不平衡**和**欠拟合**问题。
- **DFT Loss：** $\mathcal{L}_{\text{DFT}} \propto - P_{\theta} \log P_{\theta}$。它通过 $\propto P_{\theta}$ 因子，**降低**模型对非常不自信（$P_{\theta}$ 很小）的“困难样本”的权重，避免模型在它们身上**过拟合**。

这种鲜明的对比，可能正反映了 AI 发展的时代变迁：在 LLM 时代，模型的拟合能力极强，我们更担心的不再是学不够（欠拟合），而是**学过头（过拟合）**。DFT 正是为解决 LLM 这一核心问题而量身定制的简单而高效的解决方案。

## 结语

DFT 的成功证明了，通过**从理论上深入理解现有算法的内在机制**，我们可以用极其简洁的修正（一行代码）来撬动巨大的性能提升。它为 LLM 的对齐技术路线提供了一个全新的视角：**我们不需要完全抛弃 SFT 转向复杂的 RLHF，只需要修复 SFT 隐式奖励机制中的缺陷，就能让其泛化能力得到显著提升。**
