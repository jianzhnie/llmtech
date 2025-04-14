# DAPO: 一个开源的大规模LLM强化学习系统

## 摘要

推理扩展赋予了大语言模型（LLMs）前所未有的推理能力，而强化学习（RL）是激发复杂推理的核心技术。然而，当前最先进的推理LLMs 的关键技术细节（如OpenAI的博客和DeepSeek R1技术报告）仍然被隐藏，导致社区难以复现其RL训练结果。我们提出了解耦裁剪和动态采样策略优化（DAPO）算法，并完全开源了一个最先进的大规模RL系统，该系统基于Qwen2.5-32B基础模型，在AIME 2024上取得了50分的成绩。与之前的工作不同，我们详细介绍了算法的四个关键技术，使得大规模LLM RL训练成为可能。此外，我们开源了基于 verl 框架的训练代码，并提供了一个精心整理和处理的数据集。这些开源组件增强了可复现性，并支持未来在大规模LLM RL领域的研究。

通讯作者：Qiying Yu，邮箱：yuqy22@mails.tsinghua.edu.cn

项目页面：

- https://dapo-sia.github.io/
- https://github.com/volcengine/verl



<img src="https://github.com/BytedTsinghua-SIA/DAPO/raw/main/img/score.png" alt="alt text" style="zoom: 33%;" />



## 1 引言

测试时扩展（如OpenAI的o1 [1]和DeepSeek的R1 [2]）为大语言模型（LLMs）带来了深刻的范式转变 [3, 4, 5, 6, 7]。测试时扩展使得模型能够进行更长的链式思考（Chain-of-Thought），并诱导出复杂的推理行为，这使得模型在竞争性数学和编程任务（如AIME和Codeforces）中表现出色。

推动这一变革的核心技术是大规模强化学习（RL），它能够激发复杂的推理行为，如自我验证和迭代优化。然而，现有的推理模型技术报告 [1, 8, 9, 10, 11, 2] 中隐藏了可扩展RL训练的实际算法和关键配方。在本文中，我们揭示了大规模RL训练中的重大障碍，并开源了一个可扩展的RL系统，包括完全开源的算法、训练代码和数据集，提供了具有工业级RL结果的民主化解决方案。

我们在Qwen2.5-32B [12] 预训练模型上进行了RL实验。在最初的GRPO运行中，我们仅在AIME上取得了30分——这一表现显著低于DeepSeek的RL（47分）。深入分析表明，朴素的GRPO基线存在几个关键问题，如熵崩溃、奖励噪声和训练不稳定性。更广泛的社区在复现DeepSeek的结果时也遇到了类似的挑战 [13, 14, 15, 16, 17, 18, 19]，这表明R1论文中可能省略了开发工业级、大规模且可复现的RL系统所需的关键训练细节。

为了弥补这一差距，我们发布了一个开源的最先进的大规模LLM RL系统，该系统基于Qwen2.5-32B模型，在AIME 2024上取得了50分的成绩，超越了之前由DeepSeek-R1-Zero-Qwen-32B [2] 取得的47分，且仅使用了50%的训练步骤（图1）。我们提出了解耦裁剪和动态采样策略优化（DAPO）算法，并介绍了4个关键技术，使得RL在长链式思考（long-CoT）RL场景中大放异彩。细节将在第3节中详细介绍。

1. Clip-Higher，促进系统多样性，避免熵崩溃；
2. 动态采样，提高训练效率和稳定性；
3. Token-Level策略梯度损失，在长链式思考RL场景中至关重要；
4. 过长奖励塑造，减少奖励噪声并稳定训练。

我们的实现基于verl [20]。通过完全开源我们的最先进RL系统，包括训练代码和数据，我们旨在揭示大规模LLM RL的宝贵见解，以惠及更广泛的社区。

## 2 预备知识

### 2.1 近端策略优化（PPO）

PPO [21] 引入了裁剪替代目标来进行策略优化。通过使用裁剪将策略更新限制在前一策略的近端区域内，PPO稳定了训练并提高了样本效率。具体来说，PPO通过最大化以下目标来更新策略：

$$
\mathcal{J}_{\textrm{PPO}}(\theta)=\mathbb{E}_{(q,a)\sim\mathcal{D},o_{\leq t} \sim\pi_{\theta,\textrm{old}}(\cdot|q)}\left[\min\left(\frac{\pi_{\theta}(o_{t}|q,o_{<t})}{\pi_{\theta,\textrm{old}}(o_{t}|q,o_{<t})}\hat{A}_{t},\;\operatorname{clip}\left(\frac{\pi_{\theta}(o_{t}|q,o_{<t})}{\pi_{\theta,\textrm{old}}(o_{t}|q,o_{<t})},1-\varepsilon,1+\varepsilon\right)\hat{A}_{t}\right)\right]
$$

其中$(q,a)$是来自数据分布$\mathcal{D}$的问题-答案对，$\varepsilon$是重要性采样比的裁剪范围，$\hat{A}_{t}$是时间步$t$的优势估计器。给定价值函数$V$和奖励函数$R$，$\hat{A}_{t}$使用广义优势估计（GAE）[22]计算：

$$
\hat{A}^{\textrm{GAE}(\gamma,\lambda)}_{t}=\sum_{l=0}^{\infty}(\gamma\lambda)^{l}\delta_{t+l},
$$

其中

$$
\delta_{l}=R_{l}+\gamma V(s_{l+1})-V(s_{l}),\quad 0\leq\gamma,\lambda\leq 1.
$$

### 2.2 组相对策略优化（GRPO）

与PPO相比，GRPO消除了价值函数，并以组相对的方式估计优势。对于特定问题-答案对$(q,a)$，行为策略$\pi_{\theta_{\text{old}}}$采样一组$G$个独立响应$\{o_{i}\}_{i=1}^{G}$。然后，第$i$个响应的优势通过归一化组级奖励$\{R_{i}\}_{i=1}^{G}$计算：

$$
\hat{A}_{i,t}=\frac{r_{i}-\text{mean}(\{R_{i}\}_{i=1}^{G})}{\text{std}(\{R_{i}\}_{i=1}^{G})}.
$$

与PPO类似，GRPO采用裁剪目标，并直接施加KL惩罚项：

$$
\begin{aligned}
\mathcal{J}_{\text{GRPO}}(\theta) &= \mathbb{E}_{(q,a)\sim\mathcal{D},\{o_{i}\}_{i=1}^{G}\sim\pi_{\theta_{\text{old}}}(\cdot|q)} \\
&\quad\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_{i}|}\sum_{t=1}^{|o_{i}|}\left(\min\left(r_{i,t}(\theta)\hat{A}_{i,t},\text{clip}\left(r_{i,t}(\theta),1-\varepsilon,1+\varepsilon\right)\hat{A}_{i,t}\right)-\beta D_{\text{KL}}(\pi_{\theta}\|\pi_{\text{ref}})\right)\right]
\end{aligned}
$$

其中

$$
r_{i,t}(\theta)=\frac{\pi_{\theta}(o_{i,t}\mid q,o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}\mid q,o_{i,<t})}.
$$

值得注意的是，GRPO在样本级别计算目标。具体来说，GRPO首先计算每个生成序列内的平均损失，然后再对不同样本的损失进行平均。正如我们将在第3.3节中讨论的那样，这种差异可能会对算法的性能产生影响。

### 2.3 移除KL散度

KL惩罚项用于调节在线策略与冻结参考策略之间的差异。在RLHF场景 [23] 中，RL的目标是在不偏离初始模型太远的情况下对齐模型行为。然而，在训练长链式思考推理模型时，模型分布可能会显著偏离初始模型，因此这种限制是不必要的。因此，我们将从我们提出的算法中排除KL项。

<img src="https://arxiv.org/html/2503.14476v1/x2.png" alt="Refer to caption" style="zoom: 33%;" />

<img src="https://arxiv.org/html/2503.14476v1/x3.png" alt="Refer to caption" style="zoom:33%;" />

图2：在RL训练过程中，应用Clip-Higher策略前后，AIME测试集上的准确率以及Actor模型生成概率的熵。

### 2.4 基于规则的奖励建模

奖励模型的使用通常受到奖励欺骗问题的影响 [24, 25, 26, 27, 28, 29]。相反，我们直接使用可验证任务的最终准确率作为结果奖励，使用以下规则计算：

$$
R(\hat{y},y)=\begin{cases}1,&\texttt{is\_equivalent}(\hat{y},y)\\-1,&\text{否则}\end{cases}
$$

其中$y$是真实答案，$\hat{y}$是预测答案。这种方法被证明是激活基础模型推理能力的有效方法，如自动定理证明 [30, 31, 32, 33]、计算机编程 [34, 35, 36, 37] 和数学竞赛 [2] 等多个领域所示。

## 3 DAPO

我们提出了解耦裁剪和动态采样策略优化（DAPO）算法。DAPO为每个问题$q$和答案$a$采样一组输出$\{o_{i}\}_{i=1}^{G}$，并通过以下目标优化策略：

$$
\begin{aligned}
\mathcal{J}_{\text{DAPO}}(\theta) &= \mathbb{E}_{(q,a)\sim\mathcal{D},\{o_{i}\}_{i=1}^{G}\sim\pi_{\theta_{\text{old}}}(\cdot|q)} \\
&\quad\left[\frac{1}{\sum_{t=1}^{G}|o_{i}|}\sum_{i=1}^{G}\sum_{t=1}^{|o_{i}|}\min\left(r_{i,t}(\theta)\hat{A}_{i,t},\;\operatorname{clip}\left(r_{i,t}(\theta),1-\varepsilon_{\text{low}},1+\varepsilon_{\text{high}}\right)\hat{A}_{i,t}\right)\right] \\
&\text{s.t. } 0<\left|\{o_{i}|\texttt{is\_equivalent}(a,o_{i})\}\right|<G
\end{aligned}
$$

其中

$$
r_{i,t}(\theta)=\frac{\pi_{\theta}(o_{i,t}\mid q,o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}\mid q,o_{i,<t})},\quad\hat{A}_{i,t}=\frac{R_{i}-\operatorname{mean}(\{R_{i}\}_{i=1}^{G})}{\operatorname{std}(\{R_{i}\}_{i=1}^{G})}.
$$

完整算法见算法1。在本节中，我们将介绍与DAPO相关的关键技术。

### 3.1 提高上限：Clip-Higher

在我们使用朴素PPO [21] 或GRPO [38] 的初始实验中，我们观察到了熵崩溃现象：随着训练的进行，策略的熵迅速下降（图1(b)）。某些组的采样响应几乎完全相同。这表明探索有限且早期策略确定性，这可能会阻碍扩展过程。

我们提出了Clip-Higher策略来解决这个问题。裁剪重要性采样比在裁剪近端策略优化（PPO-Clip）[21] 中被引入，以限制信任区域并增强RL的稳定性。我们发现，上限裁剪会限制策略的探索。在这种情况下，使"利用token"更有可能比提升不太可能的"探索token"的概率要容易得多。

具体来说，当$\varepsilon=0.2$（大多数算法的默认值）时，考虑两个动作，其概率分别为$\pi_{\theta_{\text{old}}}(o_{i}\mid q)=0.01$和$0.9$。更新后的最大可能概率$\pi_{\theta}(o_{i}\mid q)$分别为$0.012$和$1.08$。这意味着对于概率较高的token（例如$0.9$），约束较少。相反，对于低概率token，实现非平凡的概率提升要困难得多。经验上，我们还观察到裁剪token的最大概率约为$\pi_{\theta}(o_{i}\mid q)<0.2$（图2(a)）。这一发现支持了我们的分析，即上限裁剪阈值确实限制了低概率token的概率提升，从而可能限制系统的多样性。

遵循Clip-Higher策略，我们将下限和上限裁剪范围解耦为$\varepsilon_{\text{low}}$和$\varepsilon_{\text{high}}$，如公式10所示：

$$
\begin{align*}
\mathcal{J}_{\mathrm{DAPO}}(\theta) &= \ \mathbb{E}_{(q,a)\sim\mathcal{D}, \{\mathbf{o}_i\}_{i=1}^G\sim\pi_{\theta_{\mathrm{old}}}(\cdot|q)} \\
&\qquad\quad\left[\frac{1}{\sum_{i=1}^G|\mathbf{o}_i|}\sum_{i=1}^G\sum_{t=1}^{|\mathbf{o}_i|}\min\Big(r_{i,t}(\theta)\hat{A}_{i,t},\ \mathrm{clip}\Big(r_{i,t}(\theta),1-\varepsilon_{\mathrm{low}},1+\varepsilon_{\mathrm{high}}\Big)\hat{A}_{i,t}\Big)\right] \\
\textrm{s.t.} &\ \ 0<\left|\{\mathbf{o}_i \mid \mathbf{is\_equivalent}(a,\mathbf{o}_i)\}\right|<G.
\end{align*} \tag{10}
$$

我们增加$\varepsilon_{\text{high}}$的值，为低概率token的提升留出更多空间。如图2所示，这一调整有效地增强了策略的熵，并促进了更多样化的样本生成。我们选择保持$\varepsilon_{\text{low}}$相对较小，因为增加它会将这些token的概率抑制到$0$，导致采样空间的崩溃。

<img src="https://arxiv.org/html/2503.14476v1/x6.png" alt="Refer to caption" style="zoom:33%;" />

<img src="https://arxiv.org/html/2503.14476v1/x7.png" alt="Refer to caption" style="zoom:33%;" />

图4：Actor模型生成概率分布的熵以及响应长度的变化。

### 3.2 多多益善：动态采样

现有的RL算法在某些提示的准确率等于$1$时会遇到梯度下降问题。例如，对于GRPO，如果某个提示的所有输出$\{\alpha_{i}\}_{i=1}^{G}$都正确并收到相同的奖励$1$，则该组的优势为零。零优势导致策略更新没有梯度，从而降低了样本效率。经验上，准确率等于$1$的样本数量不断增加，如图3b所示。这意味着每批次中有效提示的数量不断减少，这可能导致梯度方差增大，并削弱模型训练的梯度信号。

为此，我们提出过采样并过滤掉准确率等于1和0的提示，如公式11所示，保留批次中所有具有有效梯度的提示，并保持一致的提示数量。在训练前，我们持续采样，直到批次完全填充准确率既不为$0$也不为$1$的样本。

$$
\begin{align*}
\mathcal{J}_{\mathrm{DAPO}}(\theta) &= \ \mathbb{E}_{(q,a)\sim\mathcal{D}, \{\mathbf{o}_i\}_{i=1}^G\sim\pi_{\theta_{\mathrm{old}}}(\cdot|q)} \\
&\qquad\quad\left[\frac{1}{\sum_{i=1}^G|\mathbf{o}_i|}\sum_{i=1}^G\sum_{t=1}^{|\mathbf{o}_i|}\min\Big(r_{i,t}(\theta)\hat{A}_{i,t},\ \mathrm{clip}\Big(r_{i,t}(\theta),1-\varepsilon_{\mathrm{low}},1+\varepsilon_{\mathrm{high}}\Big)\hat{A}_{i,t}\Big)\right] \\
\textrm{s.t.} &\ \ 0<\left|\{\mathbf{o}_i \mid \mathbf{is\_equivalent}(a,\mathbf{o}_i)\}\right|<G.
\end{align*} \tag{10}
$$

请注意，这种策略不一定会阻碍训练效率，因为如果RL系统是同步的且生成阶段没有流水线化，生成时间通常由长尾样本的生成主导。此外，我们发现，使用动态采样的实验能够更快地达到相同的性能，如图5所示。

<img src="https://arxiv.org/html/2503.14476v1/x8.png" alt="Refer to caption" style="zoom:33%;" />

<img src="https://arxiv.org/html/2503.14476v1/x9.png" alt="Refer to caption" style="zoom:33%;" />

>  图5：应用**过长奖励塑造**策略前后，演员模型在AIME上的准确率及其生成概率的熵

### 3.3 重新平衡：Token-Level策略梯度损失

原始的GRPO算法采用样本级损失计算，即首先在每个样本内按token平均损失，然后在样本间聚合损失。在这种方法中，每个样本在最终损失计算中被赋予相同的权重。然而，我们发现这种损失缩减方法在长链式思考RL场景中引入了几个挑战。

由于所有样本在损失计算中被赋予相同的权重，较长响应中的token（包含更多token）可能对整体损失的贡献不成比例地较低，这可能导致两个不利影响。首先，对于高质量的长样本，这种效应可能会阻碍模型学习其中的推理相关模式。其次，我们观察到，过长的样本通常表现出低质量模式，如胡言乱语和重复单词。因此，样本级损失计算由于无法有效惩罚长样本中的这些不良模式，导致熵和响应长度的不健康增加，如图4(a)和图4(b)所示。

我们在长链式思考RL场景中引入了Token-Level策略梯度损失来解决上述限制：

$$
\begin{align*}
\mathcal{J}_{\mathrm{DAPO}}(\theta) &= \ \mathbb{E}_{(q,a)\sim\mathcal{D}, \{\mathbf{o}_i\}_{i=1}^G\sim\pi_{\theta_{\mathrm{old}}}(\cdot|q)} \\
&\qquad\quad\left[\frac{1}{\sum_{i=1}^G|\mathbf{o}_i|}\sum_{i=1}^G\sum_{t=1}^{|\mathbf{o}_i|}\min\Big(r_{i,t}(\theta)\hat{A}_{i,t},\ \mathrm{clip}\Big(r_{i,t}(\theta),1-\varepsilon_{\mathrm{low}},1+\varepsilon_{\mathrm{high}}\Big)\hat{A}_{i,t}\Big)\right] \\
\textrm{s.t.} &\ \ 0<\left|\{\mathbf{o}_i \mid \mathbf{is\_equivalent}(a,\mathbf{o}_i)\}\right|<G.
\end{align*} \tag{10}
$$

在这种设置中，较长的序列可以对整体梯度更新产生更大的影响。此外，从单个token的角度来看，如果某种生成模式能够导致奖励的增加或减少，无论它出现在多长的响应中，它都会被同等程度地促进或抑制。

### 3.4 隐藏与寻找：过长奖励塑造

在RL训练中，我们通常为生成设置最大长度，过长的样本会被截断。我们发现，对截断样本的不当奖励塑造会引入奖励噪声，并显著干扰训练过程。

$$
R_{\text{length}}(y)=\begin{cases}
0, & |y|\leq L_{\max}-L_{\text{cache}} \\
\frac{(L_{\max}-L_{\text{cache}})-|y|}{L_{\text{cache}}}, & L_{\max}-L_{\text{cache}}<|y|\leq L_{\max} \\
-1, & L_{\max}<|y|
\end{cases} \tag{13}
$$

我们进一步提出**软过长惩罚**（公式13），这是一种长度感知的惩罚机制，用于塑造截断样本的奖励。具体而言，当响应长度超过预定义的最大值时，我们定义一个惩罚区间：

- **无惩罚区**：若 $|y| \leq L_{\max} - L_{\text{cache}}$，奖励无衰减。
- **线性惩罚区**：若 $L_{\max} - L_{\text{cache}} < |y| \leq L_{\max}$，惩罚随长度线性增加。
- **硬截断区**：若 $|y| > L_{\max}$，施加固定惩罚 $-1$。

此惩罚将与基于规则的正确性奖励叠加，从而引导模型避免生成过长的响应。

#### 算法1：DAPO——解耦裁剪与动态采样策略优化

1. **输入**：初始策略模型 $\pi_{\theta}$，奖励模型 $R$，任务提示集 $\mathcal{D}$，超参数 $\varepsilon_{\text{low}}, \varepsilon_{\text{high}}$
2. **训练循环**（共 $M$ 步）：
   1. 从 $\mathcal{D}$ 中采样一个批次 $\mathcal{D}_b$
   2. 更新旧策略模型 $\pi_{\theta_{\text{old}}} \leftarrow \pi_{\theta}$
   3. 对每个问题 $q \in \mathcal{D}_b$，采样 $G$ 个输出 $\{\mathbf{o}_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(\cdot|q)$
   4. 通过 $R$ 计算每个输出 $\mathbf{o}_i$ 的奖励 $\{r_i\}_{i=1}^G$
   5. 过滤 $\mathbf{o}_i$ 并将剩余样本加入动态采样缓冲区（**动态采样**，公式11）
   6. **若缓冲区大小 $n_b < N$**：
      - 对缓冲区中的每个 $\mathbf{o}_i$，计算其第 $t$ 个token的优势 $\hat{A}_{i,t}$（公式9）
   7. **参数更新循环**（共 $\mu$ 次迭代）：
      - 通过最大化DAPO目标（公式8）更新策略模型 $\pi_{\theta}$

### 3.5 数据集转换

我们的数据集来自AoPS1网站和官方竞赛主页，通过网页抓取和手动注释相结合的方式获取。数学数据集的答案通常以多种格式呈现，如表达式、公式和数字，这使得设计全面的解析规则具有挑战性。为了使用规则提供准确的奖励信号并最小化公式解析器引入的错误，受AIME启发，我们选择并将答案转换为易于解析的整数。例如，如果原始答案以$\frac{a+\sqrt{b}}{2}$的形式表示，我们指示LLM修改问题，使预期答案变为$a+b+c$。经过选择和转换，我们获得了DAPO-Math-17K数据集，该数据集包含17K个提示，每个提示都配有一个整数作为答案。

## 4 实验

### 4.1 训练细节

在这项工作中，我们特别关注数学任务以评估我们的算法，该算法可以轻松转移到其他任务。我们采用verl框架 [20] 进行训练。我们使用朴素的GRPO [38] 作为基线算法，并使用组奖励归一化来估计优势。

对于超参数，我们使用AdamW [39] 优化器，恒定学习率为$1\times 10^{-6}$，并在20个rollout步骤中进行线性预热。对于rollout，提示批次大小为512，我们为每个提示采样16个响应。对于训练，小批次大小设置为512，即每个rollout步骤进行16次梯度更新。对于**过长奖励塑造**，我们将预期最大长度设置为16,384个token，并分配额外的4,096个token作为软惩罚缓存。因此，生成的最大token数设置为20,480个token。至于**Clip-Higher**机制，我们将裁剪参数$\varepsilon_{\text{low}}$设置为0.2，$\varepsilon_{\text{high}}$设置为0.28，这有效地平衡了探索与利用之间的权衡。对于AIME的评估，我们重复评估集32次，并报告avg@32以确保结果稳定性。评估的推理超参数设置为温度1.0和topp 0.7。

![Refer to caption](https://arxiv.org/html/2503.14476v1/x10.png)

### 4.2 主要结果

在AIME 2024上的实验表明，**DAPO**成功地将Qwen-32B基础模型训练为一个强大的推理模型，其性能优于DeepSeek在Qwen2.5-32B上使用R1方法的实验。在图1中，我们观察到AIME 2024上的性能显著提升，准确率从接近0%提高到50%。值得注意的是，这一提升仅使用了DeepSeek-R1-Zero-Qwen-32B所需训练步骤的50%。

我们分析了我们方法中每种训练技术的贡献，详见表1。观察到的改进证明了这些技术在RL训练中的有效性，每种技术都为AIME 2024贡献了几个准确率点。值得注意的是，在朴素的GRPO设置下，仅能从Qwen2.5-32B基础模型训练中达到30%的准确率。

对于token级损失，尽管它带来的性能提升较少，但我们发现它增强了训练稳定性，并使长度增长更加健康。

当应用**动态采样**时，尽管由于过滤掉零梯度数据需要采样更多数据，但总体训练时间并未显著受到影响。如图6所示，尽管采样实例数量增加，但由于所需的训练步骤减少，模型的收敛时间甚至缩短了。

### 4.3 训练动态

大规模语言模型的强化学习不仅是一个前沿的研究方向，也是一个本质上复杂的系统工程挑战，其特征是各个子系统之间的相互依赖性。对任何单个子系统的修改都可能通过系统传播，由于这些组件之间的复杂相互作用，导致不可预见的后果。即使初始条件的微小变化，如数据和超参数的变化，也可能通过迭代强化学习过程放大，产生结果的显著偏差。这种复杂性常常使研究人员面临一个困境：即使经过细致的分析和有根据的预期，认为某个修改将增强训练过程的特定方面，实际结果往往与预期轨迹不同。因此，在实验过程中监控关键的中间结果对于快速识别差异的来源并最终优化系统至关重要。

* **生成响应的长度**是与训练稳定性和性能密切相关的指标，如图(a)a所示。长度的增加为模型提供了更大的探索空间，允许采样更复杂的推理行为，并通过训练逐渐加强。然而，需要注意的是，长度在训练过程中并不总是保持持续上升的趋势。在某些相当长的时间内，它可能会表现出停滞甚至下降的趋势，这也已在 [] 中得到了证明。我们通常将长度与验证准确率结合使用，作为评估实验是否恶化的指标。
* **训练过程中的奖励动态**一直是强化学习中的关键监控指标之一，如图(b)b所示。在我们的大多数实验中，奖励增加的趋势相对稳定，不会因实验设置的调整而显著波动或下降。这表明，在可靠的奖励信号下，语言模型能够稳健地拟合训练集的分布。然而，我们发现训练集上的最终奖励通常与验证集上的准确率几乎没有相关性，这表明对训练集的过拟合。
* **Actor模型和生成概率的熵**与模型的探索能力相关，是我们实验中密切监控的关键指标。直观上，模型的熵需要保持在适当的范围内。过低的熵表明概率分布过于尖锐，导致探索能力丧失。相反，过高的熵通常与过度探索问题相关，如胡言乱语和重复生成。对于生成概率，情况正好相反。正如第3.1节所示，通过应用Clip-Higher策略，我们有效地解决了熵崩溃问题。在后续实验中，我们发现保持熵的缓慢上升趋势有助于模型性能的提升，如图7c和图7d所示。

<img src="https://arxiv.org/html/2503.14476v1/x11.png" alt="Refer to caption" style="zoom:33%;" />

<img src="https://arxiv.org/html/2503.14476v1/x12.png" alt="Refer to caption" style="zoom:33%;" />

<img src="https://arxiv.org/html/2503.14476v1/x13.png" alt="Refer to caption" style="zoom:33%;" />

<img src="https://arxiv.org/html/2503.14476v1/x14.png" alt="Refer to caption" style="zoom:33%;" />

图7：**DAPO**的响应长度、奖励分数、生成熵和平均概率的指标曲线，展示了RL训练的动态，并作为识别潜在问题的重要监控指标。

### 4.4 案例研究

在RL训练过程中，我们观察到一个有趣的现象：Actor模型的推理模式随着时间的推移动态演变。具体来说，算法不仅加强了有助于正确解决问题的现有推理模式，还逐渐产生了最初不存在的新推理模式。这一发现揭示了RL算法的适应性和探索能力，并为模型的学习机制提供了新的见解。

例如，在模型训练的早期阶段，几乎没有出现对先前推理步骤的检查和反思。然而，随着训练的进行，模型表现出明显的反思和回溯行为，如表2所示。这一观察为进一步探索RL过程中推理能力的出现提供了新的视角，我们将其留给未来的研究。

## 5 结论

在本文中，我们发布了一个完全开源的大规模LLM RL系统，包括算法、代码基础设施和数据集。该系统实现了最先进的大规模LLM RL性能（使用Qwen-32B预训练模型在AIME上取得50分）。我们提出了**解耦裁剪和动态采样策略优化（DAPO）**算法，并介绍了4个关键技术，使得RL在长链式思考RL场景中强大且高效。此外，通过开源训练代码和数据集，我们为更广泛的研究社区和社会提供了实用的可扩展强化学习解决方案，使所有人都能从这些进步中受益。
