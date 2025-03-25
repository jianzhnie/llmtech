# REINFORCE++：一种简单高效的大型语言模型对齐方法

## 摘要

基于人类反馈的强化学习（RLHF）已成为对齐大型语言模型与人类偏好的关键方法，通过近端策略优化（PPO）、直接偏好优化（DPO）、REINFORCE留一法（RLOO）、ReMax和组相对策略优化（GRPO）等方法实现了快速算法演进。我们提出REINFORCE++，这是经典REINFORCE算法的增强变体，融合了PPO的关键优化技术同时消除了critic 网络的需求。REINFORCE++实现了三个主要目标：

（1）简洁性

（2）增强的训练稳定性

（3）降低计算开销。

通过大量实证评估，我们证明REINFORCE++相比GRPO具有更优的稳定性，相比PPO具有更高的计算效率，同时保持可比性能。实现代码已开源：https://github.com/OpenRLHF/OpenRLHF。

## 1 引言

大型语言模型（LLMs）的快速发展显著提升了其生成连贯、上下文相关且类人文本的能力。然而，将这些模型与人类偏好对齐仍然面临关键挑战，因为模型可能生成与用户意图或伦理准则不符的输出。基于人类反馈的强化学习（RLHF）通过将人类偏好纳入训练过程，已成为解决这一挑战的主要方法。

该领域经历了显著的算法创新，从基础性的近端策略优化（PPO）到最近的直接偏好优化（DPO）、REINFORCE留一法（RLOO）、ReMax和组相对策略优化（GRPO）。PPO虽然有效，但需要Critic网络从而引入额外计算开销。而GRPO等新方法虽然解决了特定优化挑战，但可能引入复杂性和不稳定性。

本文提出REINFORCE++，这是经典REINFORCE算法的新变体，集成了PPO的关键优化技术同时无需Critic网络。我们的方法围绕三个主要目标设计：

● 简洁性：基于简单的REINFORCE框架，最小化实现复杂度

● 训练稳定性：通过 token 级KL惩罚、PPO-clip损失 和 标准化优势更新 确保鲁棒训练动态

● 效率：移除Critic网络降低计算开销，适合大规模应用

通过大量实证评估，我们证明REINFORCE++在显著降低计算需求的同时实现了具有竞争力的对齐性能。主要贡献包括：

● 将PPO启发技术创新性地集成到REINFORCE框架


● 在通用和领域专用数据集上的全面评估

● 开源实现以促进研究和应用

## 2 背景

### 2.1 基于人类反馈的强化学习

基于人类反馈的强化学习（RLHF）框架包含三个核心组件：

1. 监督微调（SFT）：在人类标注的提示-响应数据集上对预训练语言模型进行微调，建立基线策略（$\pi_{\text{SFT}}$）。该阶段旨在确保模型具备基本的指令遵循能力。

2. 奖励建模：通过收集人类对模型输出的排序数据（例如对同一提示的不同响应进行偏好排序），训练奖励模型 $r_{\phi}(x, y)$ 来量化生成内容的质量。奖励模型需满足以下特性：
   - 对高质量输出的高奖励预测
   - 对有害/低质量输出的低奖励预测
   - 对语义相似输出的奖励平滑性

3. 策略优化：使用强化学习算法优化语言模型策略 $\pi_{\theta}$，使其最大化奖励模型的预测值。优化目标可形式化为：
   $$
   \max_{\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\theta}(\cdot|x)} \left[ r_{\phi}(x, y) \right] - \beta \cdot \mathbb{D}_{\text{KL}} \left[ \pi_{\theta}(y|x) \| \pi_{\text{SFT}}(y|x) \right]
   $$

   其中 $\beta$ 为KL散度惩罚系数，用于约束策略更新幅度。

### 2.2 REINFORCE算法

作为策略梯度方法的奠基性算法，REINFORCE通过以下四步实现优化：

1. 轨迹采样：从当前策略 $\pi_{\theta}$ 生成完整响应序列
2. 回报计算：通过奖励模型评估生成序列的累积回报
3. 策略梯度估计：使用蒙特卡洛方法估计梯度：
   $$\nabla_{\theta}J(\theta) = \mathbb{E}_{\pi}\left[ G_{t} \nabla_{\theta}\log\pi_{\theta}(a_{t}|s_{t}) \right]$$
   其中 $G_t$ 表示时间步 $t$ 的折扣累积回报
4. 参数更新：沿梯度方向更新策略参数 $\theta$

尽管算法结构简单，但REINFORCE存在两个主要缺陷：
- 高方差：蒙特卡洛估计导致梯度方差较大
- 样本效率低：需要完整轨迹才能进行参数更新

### 2.3 RLHF的挑战

当前RLHF方法面临三大核心挑战：

● 计算开销：以PPO为代表的Actor-Critic架构需要同时维护策略网络（actor）和价值网络（critic），显著增加显存占用和计算复杂度。例如在Llama3-70B模型训练中，critic网络会使显存需求增加约40%。

● 训练不稳定性：策略网络和价值网络的相互依赖容易导致训练动态失衡，具体表现为：
   - 策略网络过度优化奖励导致模式崩溃
   - 价值网络预测误差累积引发策略震荡
   - KL散度失控导致生成质量下降

● 可扩展性瓶颈：新兴方法（如GRPO的组标准化机制）虽然提升了特定场景的性能，但引入的额外超参数和组件增加了系统复杂度，不利于大规模部署。

## 3 REINFORCE++ 改进

REINFORCE++整合了几个关键优化，以增强训练稳定性和效率：

### 3.1 Token级KL惩罚

在传统RLHF方法中，KL散度惩罚通常作用于完整序列层面。我们提出token级KL惩罚机制，将KL约束细化到每个生成token：

$$
r(s_{t},a_{t})=I(s_{t}=[EOS])r(x,y)-\beta\cdot KL(t)
$$
$$
\text{KL}(t) = \log \left( \frac{\pi_{\text{RL}}^\theta (a_t | s_t)}{\pi_{\text{SFT}} (a_t | s_t)} \right),
$$

其中：

- $I(s_{t}=[EOS])$ 是指示函数，表示 $t$是否为最终token, 仅在序列结束符位置生效
- $KL(t)=D_{\text{KL}}\left(\pi_{\theta}(a_t|s_t) \| \pi_{\text{SFT}}(a_t|s_t)\right)$ 计算当前策略与SFT模型的token级分布差异
- $\beta$ 为动态调整的惩罚系数

该设计实现两个关键优势：
1. 细粒度控制：在生成过程中实时约束策略偏移
2. 训练稳定性：避免序列后期KL散度突然激增

### 3.2 PPO-Clip集成

借鉴PPO的clip机制，我们将其核心思想融入REINFORCE框架：

$$
L^{CLIP}(\theta)=\mathbb{E}_{t}\left[\min\left(r_{t}(\theta)\hat{A}_{t},\ \text{clip}(r_{t}(\theta),1-\epsilon,1+\epsilon)\hat{A}_{t}\right)\right]
$$
其中：

- $r_t (\theta) = \frac{\pi_\theta (a_t | s_t)}{\pi_{\text{old}} (a_t | s_t)}$是在新策略与旧策略下，在状态 $s_t$下采取动作 $a_t$的概率比。
- $\hat{A}_t$是token $t$的估计优势。
- $\text{clip} (r_t (\theta), 1 - \epsilon, 1 + \epsilon)$将概率比限制在 $[1 - \epsilon, 1 + \epsilon]$范围内，其中 $\epsilon$是一个小的超参数（通常设置为0.2左右）。

### 3.3 小批量更新

为了提高训练效率，我们实现了具有以下特点的小批量更新：

1. 批量处理：将完整轨迹数据划分块处理 , 小批量（mini-batch）
2. 多次更新：每个小批量允许多次参数更新 ,显著提升样本利用率
3. 随机优化：通过随机排列小批量顺序，注入有益噪声以增强泛化能力

### 3.4 奖励归一化与截断

实施奖励处理三原则：

1. 标准化:  使用z-score消除异常值
2. 截断：限制奖励值范围
3. 缩放：应用适当缩放因子

### 3.5 优势归一化

我们重新定义优势函数并实施标准化：

$$
A_{t} = r(x, y) - \beta\cdot\sum_{i=t}^{T} KL(i)
$$
标准化流程：
1. 计算当前批次优势值的均值 $\mu_A$ 和标准差 $\sigma_A$
2. 应用z-score变换：
   $$A_{\text{normalized}} = \frac{A - \mu_A}{\sigma_A + 10^{-8}}$$
3. 最终梯度估计量：
   $$\nabla_{\theta}J(\theta) = \mathbb{E}\left[ A_{\text{normalized}} \cdot \nabla_{\theta}\log\pi_{\theta}(a_t|s_t) \right]$$

该设计有效解决梯度幅度波动问题，同时保持策略更新的方向准确性。

## 4 实验设置

### 4.1 实验设计

使用Llama3.1-8B-SFT和Qwen2.5-7B-Instruct作为基础模型，在通用领域和数学领域数据集进行评估。

### 4.2 超参数配置

| Parameter                  | Value                               |
| :------------------------- | ----------------------------------- |
| KL Penalty Coefficient (β) | 0.01 (General), 0.001 (Mathematics) |
| Maximum Samples            | 25,000                              |
| Samples per Prompt         | 4                                   |
| Rollout Batch Size         | 256                                 |
| Training Batch Size        | 128                                 |
| Actor Learning Rate        | 5×10−7                              |
| Critic Learning Rate       | 9×10−6                              |
| Discount Factor (γ)        | 1.0                                 |
| Clip ϵ                     | 0.2                                 |

### 4.3 数据集详情

我们使用了两个不同的数据集进行评估：

- 一般领域：涵盖一般知识和对话主题的多样化提示集合和偏好数据集。
- 数学领域：一个专门的数据集和一个闭源数学奖励模型，旨在测试模型在数学情境中的推理和解决问题能力。

## 5 结果分析

### 5.1 训练稳定性

我们的实验结果表明了几个关键发现：

- 一般场景下的Bradley-Terry奖励模型：REINFORCE++在防止奖励和输出长度操纵方面表现出比GRPO更好的稳定性（图1）。
- 基于规则的奖励模型：在基于规则的奖励场景下，REINFORCE++实现了与GRPO（组归一化）相当的性能（图2）。
- 数学奖励模型：在数学问题解决场景下，REINFORCE++在每个单位KL散度下实现了比GRPO更好的奖励增加（图3）。

### 5.2 计算效率

在H100 GPU上的对比：
| 方法 | 训练时间 |
| --- | --- |
| PPO | 60小时 |
| REINFORCE++ | 42小时 |

## 6 结论

REINFORCE++作为PPO和GRPO的简单高效替代方案，在保持性能的同时显著降低计算需求。未来工作将探索更大规模数据集和复杂对齐场景的应用。

## 参考文献

[1] Jian Hu, Xibin Wu, Zilin Zhu, Xianyu, Weixun Wang, Dehao Zhang, and Yu Cao. Openrlhf: An easy-to-use, scalable and high-performance rlhf framework. *arXiv preprint arXiv:2405.11143*, 2024.

[2] Ziniu Li, Tian Xu, Yushun Zhang, Yang Yu, Ruoyu Sun, and Zhi-Quan Luo. Remax: A simple, effective, and efficient method for aligning large language models. *arXiv preprint arXiv:2310.10505*, 2023.
