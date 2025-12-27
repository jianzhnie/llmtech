# 🚀 一行代码的复兴：用强化学习视角重塑监督微调（SFT）的泛化能力



在大型语言模型（LLM）的“对齐”（Alignment）领域，核心目标是让模型能更好地理解并遵循人类指令。目前，主流技术路径分为两条：**监督微调（SFT）** 和 **基于人类反馈的强化学习（RLHF）**。

SFT 简单高效，擅长快速学习特定任务的模式（“记忆”）；而 RLHF 虽然复杂，但在处理新问题时展现出强大的泛化能力。长期以来，业界公认的观点是：“SFT 负责记忆，RL 负责泛化”。

然而，一篇来自东南大学、加州大学洛杉矶分校等机构的最新研究论文《On the Generalization of SFT: A Reinforcement Learning Perspective with Reward Rectification》颠覆了这一认知。研究者们不仅从理论上揭示了 SFT 泛化不足的根源，更提出了一种名为 **动态微调（Dynamic Fine-Tuning, DFT）**的简单优化方法。

惊人的是，这个方法的核心改动**仅仅是一行代码**，却在多个极具挑战性的数学推理任务上显著超越了标准 SFT，其性能提升甚至可以与更复杂的 RL 方法相媲美。

本文将带您深入解读这篇论文，探究 DFT `是如何用`“一行代码”撬动 SFT 的性能，实现其“文艺复兴”的。

> 论文标题：On the Generalization of SFT: A Reinforcement Learning Perspective with Reward Rectification
>
> 论文链接：https://arxiv.org/pdf/2508.05629

## 一、用RL 视角审视 SFT？

要解决 SFT 泛化能力差的问题，首先必须找到其根源。论文的核心洞察，在于首次从**强化学习（RL）**的视角对 SFT 进行了重构和剖析。

### 1.传统的 SFT 优化目标

SFT 的目标是最小化模型在专家示范数据上的负对数似然（即交叉熵损失）。其目标函数如下：

$$
\mathcal{L}_{\text{SFT}}(\theta) = -\mathbb{E}_{(x, y) \sim \mathcal{D}_{\text{*}}} \left[ \log \pi_{\theta}(y | x) \right]
$$

其中，$x$ 是输入（问题），$y$ 是专家给出的标准答案，$\pi_{\theta}(y | x)$ 是模型生成标准答案的概率。

对这个损失函数求梯度，我们得到 SFT 的参数更新方向：
$$
\nabla_\theta \mathcal{L}_{\text{SFT}}(\theta) = -\nabla_\theta \log \pi_\theta(y^*|x)
$$

### 2.策略梯度

强化学习中的**策略梯度（Policy Gradient）**的通用形式是：

$$
\nabla \mathcal{J}_{\text{PG}}(\theta) \approx \mathbb{E}_{\tau \sim \pi_{\theta}(\tau)} \left[\nabla_\theta  \log \pi_{\theta}(y|x) \cdot r(x,y) \right]
$$

这个公式可以拆解成两部分：

- $\nabla_\theta \log \pi_\theta(y|x)$：这是梯度的方向。它告诉我们参数 $\theta$ 应该往哪个方向调整，才能增加生成答案 $y$ 的概率 $\pi_\theta(y|x)$。
- $r(x, y)$：这是奖励（Reward）。它是一个标量，决定了我们这次调整的步子迈多大。如果奖励 $r$ 很高，我们就朝那个方向迈一大步；如果奖励很低，就迈一小步，甚至是负的奖励就往反方向调整。

它表示，梯度更新的方向是让获得高奖励 $R(\tau)$ 的动作序列 $\tau$ 的概率变大。

### 3.重要性采样

现在，SFT 的梯度和 RL 的策略梯度看起来还不太一样。SFT 的计算是基于固定的专家数据 $y^*$（这在 RL 里叫 off-policy），而 RL 的策略梯度是基于模型自己生成的样本 $y$（on-policy）。

为了让两者能直接对话，我们引入 off-policy RL 常见的技巧 **重要性采样**（Importance Sampling）。这个技巧的本质是，我们可以把一个基于 A 分布的计算，转换成一个基于 B 分布的计算，只需要乘上一个 A/B 的修正权重。

通过引入 **重要性采样（Importance Sampling）**这一数学技巧，论文作者们巧妙地将 SFT 的梯度表达式改写成了一种与策略梯度极其相似的形式：
$$
\nabla_\theta \mathcal{L}_{\text{SFT}}(\theta) = -\mathbb{E}_{y \sim \pi_\theta} \left[ \frac{\mathbf{1}[y = y^*]}{\pi_\theta(y|x)} \nabla_\theta \log \pi_\theta(y|x) \right]
$$

> **注：** $\mathbb{I}(y=y_{\text{*}})$ 是示性函数，只有当模型答案 $y$ 与专家答案 $y_{\text{*}}$ 完全相同时，其值才为 1，否则为 0。

### 4.SFT 梯度与策略梯度的等价性

现在，让我们把变身后 的 SFT 梯度和 RL 的策略梯度公式并排放在一起：

- RL 策略梯度：$\nabla_\theta \log \pi_\theta(y x) \cdot r(x, y)$

- 变身后的 SFT 梯度：$\nabla_\theta \log \pi_\theta(y |x) \cdot \frac{\mathbf{1}[y = y^*]}{\pi_\theta(y| x)}$。忽略了负号，因为最小化损失和最大化奖励是一样的。

这样一来：传统的 SFT，在数学上等价于一个特殊的强化学习过程：

1. 隐式奖励 (Implicit Reward) 是 $\mathbf{1}[y = y^*]$。这是一个指示函数，意思是：只有当模型生成的答案 $y$ 和专家答案 $y^*$ 一模一样时，奖励才是 1，否则奖励就是 0。
2. 这个梯度更新还被一个权重 $w(y x) = \frac{1}{\pi_\theta(y x)}$ 所加权。



## 二、SFT 的缺陷

通过前面的分析，我们发现 SFT 背地里其实是在遵循一套非常奇特的强化学习奖励机制。而这套机制，正是导致他只会死记硬背、不懂变通的罪魁祸首。

### 缺陷一：隐式奖励极其稀疏

SFT 的隐式奖励函数是 $R_{\text{SFT}}(y) = \mathbb{I}(y=y_{\text{*}})$。

这意味着，模型只有在生成与标准答案**一字不差**时才能获得奖励 1，否则奖励为 0。这种“非黑即白”的机制过于严苛，无法对“部分正确”或“有价值”的回答给予鼓励，极大地限制了模型的探索和泛化能力。

### 缺陷二：病态的重要性权重

SFT 的梯度中存在一项权重 $\frac{\pi_{\text{*}}(y|x)}{\pi_{\theta}(y|x)}$，在 SFT 场景中，通常简化为 $\frac{1}{\pi_{\theta}(y|x)}$（假设专家数据是均匀采样的）。

这个权重是导致 SFT 泛化能力差的**罪魁祸首**。当模型对于某个正确的专家答案 $y_{\text{*}}$ 分配的概率 $\pi_{\theta}(y|x)$ 很低时（即模型认为这个答案很“困难”或很“冷门”），其倒数就会变得**非常大**。

- **后果：** 这会导致梯度更新的方差极大，优化过程**极其不稳定**，可能引发梯度爆炸。
- **实质：** 模型会不成比例地过度关注那些罕见的、难以拟合的样本，试图“死记硬背”下来，而忽略了数据中普遍规律的学习，最终导致**严重过拟合**。

## 三、 DFT 的“神来之笔”：动态奖励修正

既然问题出在那个病态的重要性权重 $\frac{1}{\pi_{\theta}(y|x)}$ 上，那么最简单直接的解决办法就是想办法**修正**或**消除**它。

### 1. 核心思想：抵消病态权重

论文提出的 **动态微调（DFT）**方法的思路优雅而直接：在计算梯度时，直接给原始的 SFT 损失乘上一个修正项，这个修正项恰好就是模型自身的概率 $\pi_{\theta}(y|x)$。

修正后的 DFT 梯度形式为：

$$
\nabla \mathcal{L}_{\text{DFT}}(\theta) \approx \mathbb{E}_{y \sim \pi_{\theta}(y|x)} \left[ \nabla \log \pi_{\theta}(y|x) \cdot \mathbb{I}(y=y_{\text{*}}) \cdot \underbrace{\pi_{\theta}(y|x)}_{\text{修正项}} \right]
$$

这里的修正项使用了 $\text{stop-gradient}$ 算子（在公式中简化为直接相乘并假定 $\pi_{\theta}(y|x)$ 不参与反向传播），意味着在反向传播计算梯度时，不对这个修正项求导，只把它当作一个**动态调整的权重**。

### 2. DFT 损失函数与实践中的实现

这个修正后的梯度对应到一个简单的重加权损失（reweighted loss）。在实践中，为了计算稳定和实用，这个操作是在**词元（token）级别**进行的：

$$
\mathcal{L}_{\text{DFT}} = \sum_{t=1}^{L} - \text{stop-gradient}\left(\pi_{\theta}(y_t|x, y_{<t})\right) \cdot \log \pi_{\theta}(y_t|x, y_{<t})
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

DFTLoss 的 Pytorch 实现如下：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class DFTLoss(nn.Module):
    """
    Dynamic Fine-Tuning (DFT) Loss implementation

    This loss function implements the DFT approach where the standard cross-entropy
    loss is reweighted by the model's own probability for the target token.
    This helps stabilize training by reducing the impact of hard examples that
    the model is not confident about.

    The formula is:
    L_DFT = -stop_gradient(P_theta(y|x)) * log(P_theta(y|x))

    Where P_theta(y|x) is the model's probability for the target token y given input x.
    """

    def __init__(self, ignore_index=-100, reduction='mean'):
        """
        Initialize the DFT Loss

        Args:
            ignore_index (int): Specifies a target value that is ignored and
                               does not contribute to the input gradient
            reduction (str): Specifies the reduction to apply to the output:
                            'none' | 'mean' | 'sum'
        """
        super(DFTLoss, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Compute the DFT loss

        Args:
            logits (torch.Tensor): The logits output from the model of shape (N, C, ...)
                                where N is the batch size and C is the number of classes
            targets (torch.Tensor): Ground truth class indices of shape (N, ...) or (N, ..., C)

        Returns:
            torch.Tensor: Computed DFT loss
        """
        # Calculate probabilities using softmax
        probs = F.softmax(logits, dim=-1)

        # Calculate log probabilities
        log_probs = F.log_softmax(logits, dim=-1)

        # Get the probabilities for the target classes
        if targets.dim() == logits.dim() - 1:
            # Target is class indices
            target_probs = torch.gather(probs, -1, targets.unsqueeze(-1)).squeeze(-1)
            target_log_probs = torch.gather(log_probs, -1, targets.unsqueeze(-1)).squeeze(-1)
        else:
            # Target is one-hot encoded or soft labels
            target_probs = (probs * targets).sum(dim=-1)
            target_log_probs = (log_probs * targets).sum(dim=-1)

        # Apply stop gradient to the probabilities (detach from computation graph)
        with torch.no_grad():
            weight = target_probs.detach()

        # Calculate the DFT loss
        dft_loss = -weight * target_log_probs

        # Handle ignore_index
        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            dft_loss = dft_loss * mask

            if self.reduction == 'mean':
                return dft_loss.sum() / mask.sum().clamp(min=1)
            elif self.reduction == 'sum':
                return dft_loss.sum()
            else:  # 'none'
                return dft_loss

        # Apply reduction
        if self.reduction == 'mean':
            return dft_loss.mean()
        elif self.reduction == 'sum':
            return dft_loss.sum()
        else:  # 'none'
            return dft_loss


def dft_cross_entropy(logits, targets, ignore_index=-100, reduction='mean'):
    """
    Functional version of DFT loss

    Args:
        logits (torch.Tensor): The logits output from the model
        targets (torch.Tensor): Ground truth class indices
        ignore_index (int): Specifies a target value that is ignored
        reduction (str): Specifies the reduction to apply to the output

    Returns:
        torch.Tensor: Computed DFT loss
    """
    loss_fn = DFTLoss(ignore_index=ignore_index, reduction=reduction)
    return loss_fn(logits, targets)


# Example usage
if __name__ == "__main__":
    # Create sample data
    batch_size, seq_len, vocab_size = 2, 5, 1000
    logits = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Test DFT loss
    dft_loss_fn = DFTLoss()
    loss = dft_loss_fn(logits, targets)
    print(f"DFT Loss: {loss}")

    # Test functional version
    loss_func = dft_cross_entropy(logits, targets)
    print(f"DFT Loss (functional): {loss_func}")

    # Compare with standard cross-entropy
    ce_loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
    print(f"Standard Cross-Entropy Loss: {ce_loss}")

```

swift 的 DFT Loss 函数

```python
def per_token_loss_func(outputs, labels, enable_dft_loss: bool = False, **kwargs):
    logits = outputs.logits
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()
    labels = torch.roll(labels, shifts=-1, dims=-1).view(-1)

    # Flatten the tokens
    logits = logits.view(-1, logits.shape[-1])
    # Enable model parallelism
    labels = labels.to(logits.device)
    loss = F.cross_entropy(logits, labels, ignore_index=-100, reduction='none')
    if enable_dft_loss:
        with torch.no_grad():
            target_probs = torch.exp(-loss)
        loss *= target_probs
    return loss

```

Trl 的 DFT Loss 函数


```python
def dft_loss(outputs, labels, num_items_in_batch=None):
    """
    DFT loss function, as presented in [On the Generalization of SFT: A Reinforcement Learning Perspective with Reward
    Rectification](https://huggingface.co/papers/2508.05629)
    """
    labels = nn.functional.pad(labels, (0, 1), value=-100)
    shift_labels = labels[..., 1:].contiguous()
    loss_mask = shift_labels != -100
    shift_labels[~loss_mask] = 0
    logprobs = selective_log_softmax(outputs.logits, shift_labels)
    per_token_loss = -logprobs.exp().detach() * logprobs
    if num_items_in_batch is None:
        num_items_in_batch = loss_mask.sum()
    loss = (per_token_loss * loss_mask).sum() / num_items_in_batch
    return loss
```



### 3. DFT 的“学习哲学”：稳扎稳打

这个简单的改动，彻底改变了 SFT 的学习动态：

| **机制**                                  | **传统的 SFT (交叉熵)**                                 | **动态微调 (DFT)**                                           |
| ----------------------------------------- | ------------------------------------------------------- | ------------------------------------------------------------ |
| **权重**                                  | $\frac{1}{\pi_{\theta}(y | x)}$                           | $\mathbb{I}(y=y_{\text{*}}) \cdot \underbrace{\pi_{\theta}(y|x)}_{\text{修正项}}$ |
| **对困难样本的影响** ($\pi_{\theta} \to 0$) | 权重爆炸 ($W \to \infty$)，**梯度不稳定**，引发过拟合。 | 权重趋近 0 ($W \to 0$)，**损失被降权**，避免过度拟合。       |
| **对简单样本的影响** ($\pi_{\theta} \to 1$) | 权重趋近 1 ($W \to 1$)，损失权重保持不变。              | 权重趋近 1 ($W \to 1$)，损失权重保持不变。                   |
| **哲学**                                  | 越不自信，越要下猛药。                                  | 稳扎稳打，优先学习有把握的知识。                             |

通过将奖励修正为 $R'_{\text{DFT}} = \mathbb{I}(y=y_{\text{*}}) \cdot \pi_{\theta}(y|x)$，DFT 极大地**平滑了奖励机制**，模型不再被极低概率的样本所挟持，从而避免了梯度爆炸和过拟合，促进了模型向更稳定、泛化能力更强的方向优化。

## 四、 DFT 与 Focal Loss 的反向对比

值得一提的是，DFT 的设计哲学与计算机视觉领域著名的 **Focal Loss** 形成了有趣的对比。

- **Focal Loss：** $\mathcal{L}_{\text{Focal}} \propto -(1 - \pi_{\theta})^{\gamma} \log \pi_{\theta}$。它通过 $\propto (1 - \pi_{\theta})^{\gamma}$ 因子，**降低**模型对已经很自信（$\pi_{\theta}$ 很大）的“简单样本”的权重，让模型更专注于学习“困难样本”。其目标是解决**类别不平衡**和**欠拟合**问题。
- **DFT Loss：** $\mathcal{L}_{\text{DFT}} \propto - \pi_{\theta} \log \pi_{\theta}$。它通过 $\propto \pi_{\theta}$ 因子，**降低**模型对非常不自信（$\pi_{\theta}$ 很小）的“困难样本”的权重，避免模型在它们身上**过拟合**。

这种鲜明的对比，可能正反映了 AI 发展的时代变迁：在 LLM 时代，模型的拟合能力极强，我们更担心的不再是学不够（欠拟合），而是**学过头（过拟合）**。DFT 正是为解决 LLM 这一核心问题而量身定制的简单而高效的解决方案。

## 结语

DFT 的成功证明了，通过**从理论上深入理解现有算法的内在机制**，我们可以用极其简洁的修正（一行代码）来撬动巨大的性能提升。它为 LLM 的对齐技术路线提供了一个全新的视角：**我们不需要完全抛弃 SFT 转向复杂的 RLHF，只需要修复 SFT 隐式奖励机制中的缺陷，就能让其泛化能力得到显著提升。**
