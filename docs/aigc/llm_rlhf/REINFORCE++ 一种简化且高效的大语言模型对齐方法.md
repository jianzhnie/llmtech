# REINFORCE++: 一种简化且高效的大语言模型对齐方法

## 摘要

人类反馈强化学习（RLHF）已成为将大型语言模型与人类偏好对齐的关键方法，经历了从近端策略优化（PPO）、直接偏好优化（DPO）、REINFORCE留一法（RLOO）、ReMax到组相对策略优化（GRPO）等方法的快速算法演变。我们提出了REINFORCE++，这是一种改进的经典REINFORCE算法的变体，它融合了PPO的关键优化技术，同时消除了对评论网络的需求。REINFORCE++实现了三个主要目标：（1）简洁性；（2）增强的训练稳定性；（3）降低计算开销。通过广泛的实证评估，我们证明了REINFORCE++相比GRPO具有更好的稳定性，并且比PPO具有更高的计算效率，同时保持了相当的性能。实现代码可在https://github.com/OpenRLHF/OpenRLHF获取。

## 1 引言
大型语言模型（LLMs）的快速发展显著提升了其生成连贯、与上下文相关且类似人类文本的能力。然而，将这些模型与人类偏好对齐仍然是一个关键挑战，因为模型可能会生成与用户意图或道德准则不一致的输出。人类反馈强化学习（RLHF）作为一种领先的方法，通过将人类偏好纳入训练过程来解决这一挑战。

该领域经历了显著的算法创新，从基础的近端策略优化（PPO）到更近期的方法，如直接偏好优化（DPO）、REINFORCE留一法（RLOO）、ReMax和组相对策略优化（GRPO）。PPO虽然有效，但需要一个评论网络，这增加了额外的计算开销。同时，像GRPO这样的新方法解决了优化挑战的特定方面，但可能会引入复杂性和不稳定性。

在本文中，我们介绍了REINFORCE++，这是一种新颖的经典REINFORCE算法的变体，它在消除对评论网络需求的同时，整合了PPO的关键优化技术。我们的方法旨在实现三个主要目标：
- **简洁性**：通过基于简单的REINFORCE框架，REINFORCE++最小化了实现复杂性。
- **训练稳定性**：通过整合token级KL惩罚、PPO剪辑损失和归一化优势更新，确保了稳健的训练动态。
- **效率**：通过移除评论网络，降低了计算开销，使REINFORCE++更适合大规模应用。

通过广泛的实证评估，我们证明了REINFORCE++在保持与最新方法相当的对齐性能的同时，显著降低了计算需求。我们的贡献包括：
- 将PPO启发的技术整合到REINFORCE框架中。
- 在一般和特定领域的数据集上对REINFORCE++进行全面评估，展示其在将LLMs与人类偏好对齐方面的有效性。
- 提供开源实现，以促进进一步的研究和应用。

## 2 背景

### 2.1 人类反馈强化学习
人类反馈强化学习（RLHF）是一种利用人类提供的反馈来训练能够生成与人类偏好一致的输出的模型的框架。该过程通常包括以下组件：
- **监督微调（SFT）**：模型最初在人类标记的提示和响应数据集上进行微调，以建立基线策略。
- **奖励建模**：训练一个奖励模型，基于排序的模型输出数据集预测人类偏好。
- **策略优化**：使用强化学习，优化模型策略以最大化奖励模型预测的奖励。

尽管RLHF在提高模型对齐方面被证明是有效的，但它也引入了独特的挑战。值得注意的是，优化过程对策略和奖励模型之间的相互作用敏感，这可能导致不稳定和低效。

### 2.2 REINFORCE算法
REINFORCE是强化学习中的一种基础策略梯度方法，通过梯度上升直接优化策略的预期回报。该算法的工作原理如下：
- **轨迹采样**：代理与环境互动，生成由状态、动作和奖励组成的轨迹。
- **回报计算**：计算每个轨迹的折扣累计奖励：
  \[
  G_t = \sum_{k=t+1}^{T} \gamma^{k-t} r_k,
  \]
  其中 \(\gamma\) 是折扣因子。
- **策略梯度估计**：使用以下公式估计预期回报关于策略参数的梯度：
  \[
  \nabla_\theta J(\theta) = \mathbb{E}_\pi [G_t \nabla_\theta \log \pi_\theta (A_t | S_t)].
  \]
- **策略更新**：通过梯度上升更新策略参数：
  \[
  \theta \leftarrow \theta + \alpha \nabla_\theta J(\theta),
  \]
  其中 \(\alpha\) 是学习率。

尽管REINFORCE简单，但在梯度估计中存在高方差，这可能阻碍其在复杂任务（如对齐LLMs）中的可扩展性。

### 2.3 RLHF的挑战
RLHF的实现通常会遇到以下挑战：
- **计算开销**：像PPO这样的方法需要一个评论网络，增加了内存和计算需求。
- **训练不稳定性**：PPO中策略和价值网络的相互依赖可能导致收敛问题，特别是对于大型和复杂的模型。
- **可扩展性**：许多高级方法引入了额外的超参数和架构组件，增加了大规模部署的复杂性。

REINFORCE++通过设计解决了这些挑战，通过其简洁性和效率，使其成为RLHF任务的一个有吸引力的替代方案。

## 3 REINFORCE++的改进

REINFORCE++整合了几个关键优化，以增强训练稳定性和效率：

### 3.1 Token级KL惩罚
我们在RL模型和监督微调（SFT）模型分布之间实现了token级Kullback-Leibler（KL）散度惩罚。该惩罚被整合到奖励函数中，如下所示：
\[
r(s_t, a_t) = I(s_t = [EOS]) r(x, y) - \beta \text{KL}(t),
\]
\[
\text{KL}(t) = \log \left( \frac{\pi_{\text{RL}}^\theta (a_t | s_t)}{\pi_{\text{SFT}} (a_t | s_t)} \right),
\]
其中：
- \(x\) 表示输入提示。
- \(y\) 表示生成的响应。
- \(I(s_t = [EOS])\) 表示 \(t\) 是否为最终token。
- \(\beta\) 是KL惩罚系数。

这种方法有助于更好地分配信用，并与过程奖励模型（PRM）无缝集成。

### 3.2 PPO剪辑整合
我们整合了PPO的剪辑机制，以限制策略更新：
\[
L_{\text{CLIP}} (\theta) = \mathbb{E}_t \left[ \min \left( r_t (\theta) \hat{A}_t, \text{clip} (r_t (\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right],
\]
其中：
- \(r_t (\theta) = \frac{\pi_\theta (a_t | s_t)}{\pi_{\text{old}} (a_t | s_t)}\) 是在新策略与旧策略下，在状态 \(s_t\) 下采取动作 \(a_t\) 的概率比。
- \(\hat{A}_t\) 是token \(t\) 的估计优势。
- \(\text{clip} (r_t (\theta), 1 - \epsilon, 1 + \epsilon)\) 将概率比限制在 \([1 - \epsilon, 1 + \epsilon]\) 范围内，其中 \(\epsilon\) 是一个小的超参数（通常设置为0.2左右）。

这种公式有效地允许算法利用正优势，同时防止过大的更新可能破坏训练。使用最小函数确保如果比值偏离1（无论是高于还是低于），它不会对目标产生积极贡献，从而在策略更新中保持一种信任区域。

### 3.3 小批量更新
为了提高训练效率，我们实现了具有以下特点的小批量更新：
- **批量处理**：数据以更小、更易管理的块进行处理，而不是全批量更新。
- **多次更新**：每个小批量允许多次参数更新，提高收敛速度。
- **随机优化**：引入有益的随机性，以更好地泛化。

### 3.4 奖励归一化和剪辑
我们实现了全面的奖励处理，以稳定训练：
- **归一化**：使用z分数归一化标准化奖励，以减轻异常值的影响。
- **剪辑**：将奖励值限制在预定义的范围内，以避免不稳定。
- **缩放**：应用适当的缩放因子，以确保更新过程中的数值稳定性。

### 3.5 优势归一化
REINFORCE++中的优势函数定义为：
\[
A_t (s_t, a_t) = r(x, y) - \beta \cdot \sum_{i=t}^{T} \text{KL}(i),
\]
我们使用z分数归一化对这些优势进行归一化：
\[
A_{\text{normalized}} = \frac{A - \mu_A}{\sigma_A},
\]
其中 \(\mu_A\) 和 \(\sigma_A\) 分别表示批量均值和标准差。归一化确保了稳定的梯度，并防止了训练过程中的发散。

## 4 实验设置

### 4.1 实验设计概述
REINFORCE++的实证评估使用了多种测试场景，以确保对其在不同上下文中的性能有全面的了解。我们专注于主要目标：使用OpenRLHF评估与PPO和GRPO相比的训练稳定性和计算效率。

#### 4.1.1 基础模型
我们的实验使用了以下模型：
- Llama3.1-8B-SFT
- Qwen2.5-7B-Instruct

### 4.2 超参数配置
超参数经过精心选择，以平衡训练效率和模型性能。关键设置如下表所示：

| 参数                    | 值                          |
| ----------------------- | --------------------------- |
| KL惩罚系数（\(\beta\)） | 一般领域0.01，数学领域0.001 |
| 最大样本数              | 25,000                      |
| 每个提示的样本数        | 4                           |
| 滚动批量大小            | 256                         |
| 训练批量大小            | 128                         |
| 演员学习率              | \(5 \times 10^{-7}\)        |
| 评论员学习率            | \(9 \times 10^{-6}\)        |
| 折扣因子（\(\gamma\)）  | 1.0                         |
| 剪辑\(\epsilon\)        | 0.2                         |

### 4.3 数据集详情
我们使用了两个不同的数据集进行评估：
- **一般领域**：涵盖一般知识和对话主题的多样化提示集合和偏好数据集。
- **数学领域**：一个专门的数据集和一个闭源数学奖励模型，旨在测试模型在数学情境中的推理和解决问题能力。

## 5 结果与分析

### 5.1 训练稳定性
我们的实验结果表明了几个关键发现：
- **一般场景下的Bradley-Terry奖励模型**：REINFORCE++在防止奖励和输出长度操纵方面表现出比GRPO更好的稳定性（图1）。
- **基于规则的奖励模型**：在基于规则的奖励场景下，REINFORCE++实现了与GRPO（组归一化）相当的性能（图2）。
- **数学奖励模型**：在数学问题解决场景下，REINFORCE++在每个单位KL散度下实现了比GRPO更好的奖励增加（图3）。

### 5.2 计算效率
表2总结了使用70k样本和NVIDIA H100上的LLaMA3 8b模型的计算成本。REINFORCE++在内存使用和训练时间上相比PPO有所减少，突显了其计算效率。

| 方法        | 训练时间（小时） |
| ----------- | ---------------- |
| PPO         | 60               |
| REINFORCE++ | 42               |

## 6 结论
实验结果验证了REINFORCE++作为一种更简单且高效的RLHF替代方案的有效性，与PPO和GRPO相比。未来的工作将探索将该方法扩展到更大的数据集，并调查其在更复杂的对齐场景中的性能。
