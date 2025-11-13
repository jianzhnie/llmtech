# 缓解 Rollout-训练不匹配问题

## 简而言之

大语言模型强化学习微调不稳定的一个关键来源是**训练-推理不匹配（training-inference mismatch）**。为了最大化训练效率，现代强化学习训练框架（如 VeRL）通常会采用两种不同的计算引擎：一种是为快速推理（rollout）高度优化的引擎（如 vLLM），另一种是为梯度计算设计的训练引擎（如 FSDP）。尽管这两种引擎在数学原理上是等价的，但由于浮点数精度误差和硬件层面的具体优化差异，它们会产生数值上不完全相同的输出。近期的一系列研究已经指出，这种看似微不足道的不匹配，会在优化过程中引入显著的问题，是导致训练不稳定的核心因素之一。

## 不匹配问题

为简化起见，我们以 REINFORCE 算法为例，该算法通过以下方式更新策略 $\pi$ ——一个由 $\theta$ 参数化的 LLM：

$$
\theta \leftarrow \theta + \mu \cdot \underbrace{\mathbb{E}_{a \sim \pi(\theta)}}_{\text{rollout}}[R(a) \cdot \underbrace{\nabla_\theta \log \pi(a, \theta)}_{\text{training}}].
$$

实践中，轨迹生成成本高昂，现代强化学习框架（例如 VeRL）通常采用高度优化的推理引擎（例如 vLLM、SGLang）来提升吞吐量，同时使用独立后端（例如 FSDP、Megatron）进行模型训练。这种混合设计使得更新过程变为：

$$
\theta \leftarrow \theta + \mu \cdot \mathbb{E}_{a \sim \textcolor{red}{\pi_{\text{sampler}}}(\theta)}[R(a) \cdot \nabla_\theta \log \textcolor{blue}{\pi_{\text{learner}}}(a, \theta)].
$$

此处我们使用 $\textcolor{red}{\pi_{\text{sampler}}}$ 表示搭载推理引擎（如 vLLM、SGLang）的模型，$\textcolor{blue}{\pi_{\text{learner}}}$ 表示使用训练后端（如 FSDP、Megatron）实例化的同模型。若无特别说明，我们的实验均采用 vLLM 作为采样器后端、FSDP 作为训练器后端。

实验中观察到意外的 **rollout 训练失配现象**, 尽管 $\textcolor{blue}{\pi_{\text{fsdp}}}$ 与 $\textcolor{red}{\pi_{\text{vlm}}}$ 共享相同模型参数 $\theta$，它们却可能生成显著不同的Token概率。对于某些特定Token $a$，甚至会产生相互矛盾的预测结果，即 $\textcolor{red}{\pi_{\text{vlm}}}(a, \theta) = 1$ 与 $\textcolor{blue}{\pi_{\text{fsdp}}}(a, \theta) = 0$。这种异常行为隐式破坏了同策略假设，实质上使 On-Policy 强化学习训练悄然转变为异策略模式。


### 优化驱动的恶性循环

你可能认为训练-推理失配是硬件与软件栈的静态特性。然而，通过实验证明**这种失配与训练动态及模型状态相互耦合**。

我们推测这是由于以下两阶段级联故障所致：

1. **阶段一：数值敏感度增强**。强化学习优化器将模型权重推至 `bfloat16` 数据类型相对精度较低的数值范围（例如极小或极大值）。
2. **阶段二：内核驱动的误差放大**。这些初始微小的 `bfloat16` 量化误差随后被输入 vLLM 和 FSDP 的不同内核实现。差异化的计算顺序充当非线性放大器，使初始微小偏差最终雪崩式扩大为最终逻辑值的巨大差异。

这形成了一个**恶性反馈循环**：失配导致有偏且含噪的梯度，可能将参数进一步推向数值敏感区域，进而加剧下一轮迭代的失配程度，直至系统彻底崩溃。

## 缓解训练-推理失配的尝试

接下来我们将列举为缓解训练-推理失配所尝试的方法。其中部分方法有所助益，另一些则收效甚微。

### 使用 FP32 lm_head

受 *Minimax-M1* 技术报告及博客文章 [《Your Efficient RL Framework Secretly Brings You Off-Policy RL Training》](https://fengyao.notion.site/off-policy-rl)启发，我们通过修改 vLLM 将 lm_head 转换为 fp32 精度。但在实验中，修补后失配问题依然存在，模型崩溃仍不可避免。

### 禁用分块预填充

我们还尝试通过禁用分块预填充来验证是否能解决崩溃问题。然而，实验结果显示该方法并未解决崩溃问题。

### 启用 `enforce_eager` 与 `free_cache_engine`

VeRL 官方提供的 DAPO 方案指出，启用 CUDA 图（`enforce_eager=False`）可能导致模型性能下降。为探究这是否会影响训练-推理失配问题，我们通过消融实验研究了 vLLM 引擎超参数 `enforce_eager` 的影响，并同步考量另一超参数 `free_cache_engine`。实验结果显示，调整 `enforce_eager` 与 `free_cache_engine` 的取值对训练-推理失配现象及测试性能均无显著影响。

## 接纳失配—实施算法级修复

### 重要性采样

当直接对目标分布下的期望值进行蒙特卡洛估计较为困难时，重要性采样允许我们从替代分布中进行抽样。在上面描述的强化学习场景中，目标分布是 $\pi_{\text{learner}}$，但从中抽样极其缓慢。使用独立后端（如 vLLM）进行轨迹生成意味着我们实际上是从 $\pi_{\text{sampler}}$ 进行抽样。此时通过重要性权重对每个样本进行加权修正偏差：


$$
\mathbb{E}_{a \sim \textcolor{blue}{\pi_{\text{learner}}}(\theta)} [R(a)]
= \mathbb{E}_{a \sim \textcolor{red}{\pi_{\text{sampler}}}(\theta)} \left[
\underbrace{\frac{\textcolor{blue}{\pi_{\text{learner}}}(a, \theta)}{\textcolor{red}{\pi_{\text{sampler}}}(a, \theta)}}_{\tiny\text{importance ratio}} \cdot R(a)
\right].
$$

### 截断重要性采样 TIS

不同于在系统层面缓解分布失配，我们提出通过调整模型更新机制使其感知这种失配。简单的方法是采用重要性采样校正。具体而言，我们通过在当前梯度计算中添加重要性比率来处理 $\textcolor{blue}{\pi_{\text{learner}}}$ 与 $\textcolor{red}{\pi_{\text{sampler}}}$ 之间的失配，即将当前梯度计算从

$$
\mathbb{E}_{a \sim \textcolor{red}{\pi_{\text{sampler}}}(\theta)}[R(a) \cdot \nabla_\theta \log \textcolor{blue}{\pi_{\text{learner}}}(a, \theta)],
$$

变为

$$
\mathbb{E}_{a \sim \textcolor{red}{\pi_{\text{sampler}}}(\theta)}\left[\frac{\textcolor{blue}{\pi_{\text{learner}}}(a, \theta)}{\textcolor{red}{\pi_{\text{sampler}}}(a, \theta)} \cdot R(a) \cdot \nabla_\theta \log \textcolor{blue}{\pi_{\text{learner}}}(a, \theta)\right].
$$

尽管关于如何设计稳定有效的重采样方法已有广泛研究，但在实践中我们发现通常采用经典技术——**截断重要性采样**便已足够：

$$
\mathbb{E}_{a \sim \textcolor{red}{\pi_{\text{sampler}}}(\theta)}\left[\min\left(\frac{\textcolor{blue}{\pi_{\text{learner}}}(a, \theta)}{\textcolor{red}{\pi_{\text{sampler}}}(a, \theta)}, C\right) \cdot R(a) \cdot \nabla_\theta \log \textcolor{blue}{\pi_{\text{learner}}}(a, \theta)\right],
$$

其中 $C$ 是一个超参数。


#### 扩展至其他算法

将上述分析扩展到其他算法是直截了当的，因为我们可以将梯度计算的具体形式从 REINFORCE 的 $R(a) \cdot \nabla_\theta \log \pi(a, \theta)$ 切换为任意形式。在此，我们以常用的 PPO 算法为例，提供类似的分析作为补充说明。

PPO 的策略梯度 $\nabla_\theta L^{\text{CLIP}}(\theta)$ 定义为：

$$
\mathbb{E}_{a \sim \pi_{\text{old}}}\left[\nabla_\theta \min\left(\frac{\pi_\theta(a)}{\pi_{\theta_{\text{old}}}(a)} \hat{A},\ \text{clip}\left(\frac{\pi_\theta(a)}{\pi_{\theta_{\text{old}}}(a)},\ 1 - \epsilon,\ 1 + \epsilon\right) \hat{A}\right)\right].
$$

为提升吞吐量，混合强化学习系统采用 vLLM 引擎进行推演生成——从 $\textcolor{red}{\pi_{\text{sampler}}}(\theta_{\text{old}})$ 中采样Token $a$，同时使用 FSDP 后端从 $\textcolor{blue}{\pi_{\text{learner}}}(\theta)$ 进行采样，并 **重新计算** $\textcolor{red}{\pi_{\text{sampler}}}(\theta_{\text{old}})$ 的Token概率以完成梯度计算：

$$
\mathbb{E}_{a \sim \textcolor{red}{\pi_{\text{sampler}}}(\theta_{\text{old}})}\left[\nabla_\theta \min\left(\frac{\textcolor{blue}{\pi_{\text{learner}}}(a, \theta)}{\textcolor{blue}{\pi_{\text{learner}}}(a, \theta_{\text{old}})} \hat{A},\ \text{clip}\left(\frac{\textcolor{blue}{\pi_{\text{learner}}}(a, \theta)}{\textcolor{blue}{\pi_{\text{learner}}}(a, \theta_{\text{old}})},\ 1 - \epsilon,\ 1 + \epsilon\right) \hat{A}\right)\right],
$$

与上述分析类似，$\textcolor{blue}{\pi_{\text{learner}}}$ 与 $\textcolor{red}{\pi_{\text{sampler}}}$ 之间的差距再次显现，我们通过截断重要性采样方法予以修正：

$$
\mathbb{E}_{a \sim \textcolor{red}{\pi_{\text{sampler}}}(\theta_{\text{old}})}\left[\min\left(\frac{\textcolor{blue}{\pi_{\text{learner}}}(a, \theta)}{\textcolor{red}{\pi_{\text{sampler}}}(a, \theta)}, C\right) \cdot \nabla_\theta \min\left(\frac{\textcolor{blue}{\pi_{\text{learner}}}(a, \theta)}{\textcolor{blue}{\pi_{\text{learner}}}(a, \theta_{\text{old}})} \hat{A},\ \text{clip}\left(\frac{\textcolor{blue}{\pi_{\text{learner}}}(a, \theta)}{\textcolor{blue}{\pi_{\text{learner}}}(a, \theta_{\text{old}})},\ 1 - \epsilon,\ 1 + \epsilon\right) \hat{A}\right)\right],
$$

其中 $C$ 是一个超参数。


### 与两种 TIS 变体的比较

我们还总结了两种用于缓解分布差距的替代方案。

- **PPO 重要性采样 (PPO-IS)**

$$
\mathbb{E}_{a \sim \textcolor{red}{\pi_{\text{sampler}}}(\theta_{\text{old}})} \left[ \nabla_\theta \min\left( \frac{\textcolor{blue}{\pi_{\text{learner}}}(a, \theta)}{\textcolor{red}{\pi_{\text{sampler}}}(a, \theta_{\text{old}})} \hat{A}, \text{clip}\left( \frac{\textcolor{blue}{\pi_{\text{learner}}}(a, \theta)}{\textcolor{red}{\pi_{\text{sampler}}}(a, \theta_{\text{old}})}, 1 - \epsilon, 1 + \epsilon \right) \hat{A} \right) \right]
$$

  *注意：Colossal 框架使用此实现。*

- **基础重要性采样 (vanilla-IS)**

  $$
  \mathbb{E}_{\textcolor{red}{\pi_{\text{vlm}}}(\theta_{\text{old}})} \left[ \underbrace{\frac{\textcolor{blue}{\pi_{\text{fsdp}}}(a, \theta_{\text{old}})}{\textcolor{red}{\pi_{\text{vlm}}}(a, \theta_{\text{old}})}} \cdot \nabla_\theta \min\left( \frac{\textcolor{blue}{\pi_{\text{fsdp}}}(a, \theta)}{\textcolor{blue}{\pi_{\text{fsdp}}}(a, \theta_{\text{old}})} \hat{A}, \text{clip}\left( \frac{\textcolor{blue}{\pi_{\text{fsdp}}}(a, \theta)}{\textcolor{blue}{\pi_{\text{fsdp}}}(a, \theta_{\text{old}})}, 1 - \epsilon, 1 + \epsilon \right) \hat{A} \right) \right]
  $$

  *注意：Nemo-RL 使用此实现。*

为评估 TIS 的有效性并理解其设计选择的影响，我们进行了对比 TIS 与上述两种变体的实验。TIS 始终优于两种变体，尤其在差异显著的情况下（如 FP8/INT8 量化场景）表现更为突出。

#### vanilla-IS 对比 TIS

关于**基础重要性采样**（vanilla-IS），其不稳定性主要源于当 $ a \sim \textcolor{red}{\pi_{\text{sampler}}}(a, \theta_{\text{old}}) $ 轨迹采样概率较低时，重要性比率会大幅增加，通过 $ \left( \frac{\textcolor{blue}{\pi_{\text{learner}}}(a, \theta_{\text{old}})}{\textcolor{red}{\pi_{\text{sampler}}}(a, \theta_{\text{old}})} \right)^2 $ 放大梯度方差。为此，我们在截断重要性采样（TIS）中采用截断操作以稳定训练。例如当单个Token的比率 $ \frac{\textcolor{blue}{\pi_{\text{learner}}}(a, \theta_{\text{old}})}{\textcolor{red}{\pi_{\text{sampler}}}(a, \theta_{\text{old}})} $ 达到 16 时，该Token的梯度噪声将通过**原始重要性采样**放大 256 倍，通过 **TIS-2** 放大 4 倍，或通过 **TIS-8** 放大 64 倍。

#### PPO-IS 对比 TIS

采用 **PPO-IS** 方法后，梯度实际上仍会偏离 PPO 的同策略版本。换言之，尽管该方法可能仍在朝着无偏目标进行优化，但相比标准 PPO 算法其效率可能有所不足。

此外需要说明的是，PPO 信任域技术的提出旨在将轨迹采样 $ \theta_{\text{old}} $ 与当前模型 $ \theta $ 之间的概率比约束在接近 1 的范围内，以近似同策略 REINFORCE 梯度。但在 **PPO-IS** 中，即便当 $ \theta = \theta_{\text{old}} $ 时，由于策略不匹配，概率比 $ \frac{\textcolor{blue}{\pi_{\text{learner}}}(a, \theta)}{\textcolor{red}{\pi_{\text{sampler}}}(a, \theta_{\text{old}})} $ 仍不等于 1——这导致裁剪操作极易被触发，从而大幅降低训练的信息有效性。而在我们的 TIS 方法中，我们分别对 $ \frac{\textcolor{blue}{\pi_{\text{learner}}}(a, \theta_{\text{old}})}{\textcolor{red}{\pi_{\text{sampler}}}(a, \theta_{\text{old}})} $ 和 $ \frac{\textcolor{blue}{\pi_{\text{learner}}}(a, \theta)}{\textcolor{blue}{\pi_{\text{learner}}}(a, \theta_{\text{old}})} $ 进行裁剪，因此更为温和；值得注意的是当 $ \theta = \theta_{\text{old}} $ 时，$ \frac{\textcolor{blue}{\pi_{\text{learner}}}(a, \theta)}{\textcolor{blue}{\pi_{\text{learner}}}(a, \theta_{\text{old}})} $ 恒等于 1，这恰好符合信任域约束的要求。


### TIS 工作机制的直观解释

虽然 TIS 的确切机制仍是待解之谜，我们对其缓解分布差异的原理提供高层级阐释。

特别需要注意的是，忽略具有 $\frac{\textcolor{blue}{\pi_{\text{learner}}}(a, \theta_{\text{old}})}{\textcolor{red}{\pi_{\text{sampler}}}(a, \theta_{\text{old}})} < 1$ 的 rollout 偏差可能通过以下机制导致熵崩溃：对于具有负优势值的 rollout，策略梯度往往会降低 $\textcolor{blue}{\pi_{\text{learner}}}$。当参数更新后存在较大分布差异时，$\textcolor{blue}{\pi_{\text{learner}}}$ 的减少可能无法体现在 $\textcolor{red}{\pi_{\text{sampler}}}$ 中。因此策略梯度持续指向进一步降低 $\textcolor{blue}{\pi_{\text{learner}}}$ 的方向。直观来看，这种惩罚机制可能迫使模型过度集中于熵值较小的输出分布。

与此同时，TIS 坚持对 $\frac{\textcolor{blue}{\pi_{\text{learner}}}(a, \theta_{\text{old}})}{\textcolor{red}{\pi_{\text{sampler}}}(a, \theta_{\text{old}})} < 1$ 采用未截断的重要性比率，从而消除了这部分轨迹的偏差，并打破了这一机制。

## 重要性采样的进一步讨论

训练-推理失配将原本同策略的强化学习问题转化为异策略问题，其中用于生成轨迹的策略（行为策略，$\textcolor{red}{\pi_\theta^{\text{vllm}}}$）与正在训练的策略（目标策略，$\textcolor{blue}{\pi_\theta^{\text{fsdp}}}$）存在差异。理论上校正这种分布偏移的正规方法是**重要性采样**（IS）。然而，IS 的具体形式对于保持无偏梯度和实现稳定训练至关重要。

受 **[Yao 等, 2025]** 首次揭示这一隐式异策略问题的研究启发，我们分析了两种主要的 IS 形式：理论完备的**Sequence-Level IS** 与常见但存在缺陷的**Token-Level IS** 近似——后者也是该文献中探讨的启发式方法。

### Sequence-Level 重要性采样

正确且无偏的策略梯度估计器在整个生成序列（轨迹）上应用单一重要性比率 $y$。这种方法能准确地将行为策略的期望值重新加权为目标策略，从而得到目标函数的真实梯度 $J(\theta)$。

让我们逐步推导**Sequence-Level重要性采样**估计器 $g_{\text{seq}}(\theta)$。

- 目标是在目标 FSDP 策略下最大化期望奖励：

$$
J(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \textcolor{blue}{\pi_\theta^{\text{fsdp}}}(\cdot|x)}[R(x, y)]
$$

- 因此真实策略梯度为：

$$
g(\theta) = \nabla_\theta J(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \textcolor{blue}{\pi_\theta^{\text{fsdp}}}(\cdot|x)}\left[R(x, y)\nabla_\theta \log \textcolor{blue}{\pi_\theta^{\text{fsdp}}}(y|x)\right]
$$

- 由于我们只能从 vLLM 策略中采样，故使用重要性采样来改变期望的分布：

$$
g_{\text{seq}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \textcolor{red}{\pi_\theta^{\text{vllm}}}(\cdot|x)}\left[\frac{\textcolor{blue}{\pi_\theta^{\text{fsdp}}}(y|x)}{\textcolor{red}{\pi_\theta^{\text{vllm}}}(y|x)} \cdot R(x, y) \cdot \nabla_\theta \log \textcolor{blue}{\pi_\theta^{\text{fsdp}}}(y|x)\right]
$$

该估计器在数学上等价于标准优势函数形式的策略梯度。关键在于证明重要性采样比率能精确修正期望值，揭示底层真实的同策略梯度，进而可对其进行优化。

此推导最终得到策略梯度的优势函数形式：

$$
g_{\text{seq}}(\theta) = \mathbb{E}_{s \sim d_{\textcolor{blue}{\pi_\theta^{\text{fsdp}}}}} \mathbb{E}_{a \sim \textcolor{blue}{\pi_\theta^{\text{fsdp}}}(\cdot|s)}\left[A_\theta^{\text{fsdp}}(s, a) \cdot \nabla_\theta \log \textcolor{blue}{\pi_\theta^{\text{fsdp}}}(a|s)\right]
$$

此处 $s = (x, y_{<t})$ 表示状态（前缀），$a = y_t$ 表示动作（Token）。项 $d_{\textcolor{blue}{\pi_\theta^{\text{fsdp}}}}$ 为目标 FSDP 策略下的**状态占用度量**，其正式定义为遵循策略 $\pi$ 时期望访问状态 $s$ 的次数：

$$
d_\pi(s) := \mathbb{E}_{x' \sim \mathcal{D}, y' \sim \pi(\cdot|x')} \left[ \sum_{t'=0}^{|y'|-1} \mathbb{I}{(x', y'_{<t'}) = s} \right] = P(x) \cdot \prod_{k=0}^{t-1} \pi(y_k|x, y_{<k})
$$

该估计器是无偏的，这意味着 $g_{\text{seq}}(\theta) = g(\theta)$。为确保数值稳定性，采用**截断重要性采样**（TIS）方法，该方法将Sequence-Level比率 $\rho(y|x)$ 限制在常数 $C$ 以内。


这种方法的关键方面包括：

1. **正确的分布校正**：该方法通过计算整个序列的单一比率而不是每个token的比率来正确地应用重要性采样。这一点至关重要，因为它保持了行为策略（`π_vllm`）和目标策略（`π_fsdp`）之间正确的概率关系。

2. **无偏梯度估计**：通过使用完整的序列概率比：
   $$
   \frac{\textcolor{blue}{\pi_\theta^{\text{fsdp}}}(y|x)}{\textcolor{red}{\pi_\theta^{\text{vllm}}}(y|x)}
   $$
   估计器保持无偏，这意味着 $g_{\text{seq}}(\theta) = g(\theta)$ 是精确成立的。

3. **状态访问度量的考虑**：推导过程通过项 $d_{\textcolor{blue}{\pi_\theta^{\text{fsdp}}}}$ 正确考虑了状态访问分布的差异，该项表示在目标策略下访问各个状态的频率。

#### 序列级别IS的方差挑战

虽然在理论上是合理的，但序列级别重要性采样在实践中可能会遇到高方差问题：

- 当策略差异很大时，序列级别的比率可能变得极大或极小
- 这会导致不稳定的梯度估计，可能损害训练收敛性
- 在长序列中这个问题更加严重，因为每一步的小差异会以乘法方式累积

#### 截断重要性采样（TIS）解决方案

为了在保持理论正确性的同时解决方差问题，截断重要性采样限制了极端比率的影响：

$$
\rho_{\text{trunc}}(y|x) = \min\left(\frac{\textcolor{blue}{\pi_\theta^{\text{fsdp}}}(y|x)}{\textcolor{red}{\pi_\theta^{\text{vllm}}}(y|x)}, C\right)
$$

其中 $C$ 是控制最大允许重要性权重的超参数。

这种截断引入了一些偏差，但显著降低了方差，通常在实践中带来更稳定的训练效果。


### Token-Level 重要性采样

一种常见启发式方法，通常受到 PPO 等算法的启发并在 (Yao 等人, 2025) 中使用，采用逐词元重要性比率。虽然这通常比Sequence-Level比率具有更低的方差，但它是一种有偏估计器，对于自回归模型在理论上并不严谨。

让我们推导**Token-Level重要性采样**梯度估计器 $g_{\text{tok}}(\theta)$。

- 该公式通过错误地在时间步求和和内部应用重要性采样比率开始：即 $g_{\text{tok}}(\theta)$ 被定义为

  $$
  \mathbb{E}_{x \sim \mathcal{D}, y \sim \textcolor{red}{\pi_\theta^{\text{vllm}}}(\cdot|x)}\left[R(x, y) \cdot \sum_{t=0}^{|y|-1} \frac{\textcolor{blue}{\pi_\theta^{\text{fsdp}}}(y_t|x, y_{<t})}{\textcolor{red}{\pi_\theta^{\text{vllm}}}(y_t|x, y_{<t})} \cdot \nabla_\theta \log \textcolor{blue}{\pi_\theta^{\text{fsdp}}}(y_t|x, y_{<t})\right]
  $$

- 我们可以将此轨迹期望重写为在 vLLM 策略下访问状态的期望。

  $$
  g_{\text{tok}}(\theta) = \mathbb{E}_{s \sim d_{\textcolor{red}{\pi_\theta^{\text{vllm}}}}} \mathbb{E}_{a \sim \textcolor{red}{\pi_\theta^{\text{vllm}}}(\cdot|s)}\left[\frac{\textcolor{blue}{\pi_\theta^{\text{fsdp}}}(a|s)}{\textcolor{red}{\pi_\theta^{\text{vllm}}}(a|s)} \cdot A^{\textcolor{red}{\pi_\theta^{\text{vllm}}}}(s, a) \cdot \nabla_\theta \log \textcolor{blue}{\pi_\theta^{\text{fsdp}}}(a|s)\right]
  $$

> 注：此处 $R(x, y)$ 表示由 $\textcolor{red}{\pi_\theta^{\text{vllm}}}$ 采样的完整轨迹所得的经验回报，作为状态-动作价值函数 $Q^{\textcolor{red}{\pi_\theta^{\text{vllm}}}}(s, a)$ 的蒙特卡洛估计值。通过引入基线函数并改变动作期望的计算方式，最终得到如下形式：

$$
g_{\text{tok}}(\theta) = \mathbb{E}_{s \sim d_{\textcolor{red}{\pi_\theta^{\text{vllm}}}}} \mathbb{E}_{a \sim \textcolor{blue}{\pi_\theta^{\text{fsdp}}}(\cdot|s)}\left[A^{\textcolor{red}{\pi_\theta^{\text{vllm}}}}(s, a) \cdot \nabla_\theta \log \textcolor{blue}{\pi_\theta^{\text{fsdp}}}(a|s)\right]
$$

最终表达式清晰地揭示了Token-Level重要性采样的梯度偏差。

### Token-Level 重要性采样的偏差来源

将 $g_{\text{tok}}(\theta)$ 与真实梯度 $g_{\text{seq}}(\theta)$ 进行对比，可发现两个显著差异导致的误差，使得Token-Level估计量存在偏差。

#### 误差源 1：状态访问分布失配 🌍

有效的离策略修正必须考虑两种分布偏移：动作概率分布与状态访问概率分布。词元级方法仅修正了前者。

- **真实梯度**（$g_{\text{seq}}$）：期望计算基于正确目标 fsdp 分布下的状态访问，$\mathbb{E}_{s \sim d_{\textcolor{blue}{\pi_\theta^{\text{fsdp}}}}}$。
- **缺陷梯度**（$g_{\text{tok}}$）：期望计算基于错误行为 vLLM 分布下的状态访问，$\mathbb{E}_{s \sim d_{\textcolor{red}{\pi_\theta^{\text{vllm}}}}}$。

该方法隐含假设状态访问比率为 1，即 $d_{\textcolor{blue}{\pi^{\text{fsdp}}}}(s)/d_{\textcolor{red}{\pi^{\text{vllm}}}}(s) = 1$。在自回归模型中该假设会被严重违背：由于确定性状态转移，单个词元选择差异就会导致状态轨迹完全发散。忽略这一事实使得 $g_{\text{tok}}(\theta)$ 引入了巨大且不可控的偏差。

#### 误差源 2：失配奖励信号 🎯

第二个关键错误在于，词元级梯度使用错误策略的奖励信号来加权更新。

- **真实梯度**（$g_{\text{seq}}$）：该更新通过目标全分片数据并行策略的优势函数 $A_{\textcolor{blue}{\pi_\theta^{\text{fsdp}}}}$ 进行缩放，该函数代表在该策略下的预期未来奖励。
- **有缺陷的梯度**（$g_{\text{tok}}$）：该更新由行为 vLLM 策略的优势函数进行缩放，$A_{\textcolor{red}{\pi_\theta^{\text{vllm}}}}$。

目标策略的梯度正在被属于行为策略的奖励信号所缩放。由于状态分布和奖励信号存在根本性不匹配，Token-Level梯度实际上是一个有偏且理论不稳健的估计量。

> 🔧 **这些理论表明，尽管Token-Level方法可能具有较低的方差，但梯度偏差仍然存在，可能导致训练不稳定——这一预测在我们的实验中得到了验证。我们还针对令牌级和序列级方法提出了详细的偏差与方差分析（第一部分和第二部分）。**
>

## 掩码重要性采样（MIS）

为改进 TIS，我们提出掩码重要性采样（MIS），该方法对重要性采样比率超过阈值 $ C $（即 $ \rho(y|x) \leftarrow \rho(y|x)\mathbb{I}\{\rho(y|x) \leq C\} $）的序列进行策略损失掩码。

### Sequence-Level MIS

在Sequence-Level MIS中，我们基于整个序列的重要性比率为整个序列应用掩码。具体而言，对于一个由采样策略 $\textcolor{red}{\pi_{\text{sampler}}}$ 生成的序列 $y$，其重要性比率为：

$$
\rho(y|x) = \frac{\textcolor{blue}{\pi_{\text{learner}}}(y|x)}{\textcolor{red}{\pi_{\text{sampler}}}(y|x)}
$$

当 $\rho(y|x) > C$ 时，我们将该序列的损失完全置零，相当于从训练中移除该序列。因此，Sequence-Level MIS的策略梯度估计器为：

$$
g_{\text{seq-MIS}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \textcolor{red}{\pi_{\text{sampler}}}(\cdot|x)}\left[\mathbb{I}\{\rho(y|x) \leq C\} \cdot \rho(y|x) \cdot R(x, y) \cdot \nabla_\theta \log \textcolor{blue}{\pi_{\text{learner}}}(y|x)\right]
$$

这种方式相比于TIS更加严格，因为它完全排除了那些可能引入巨大方差的样本，而不是仅仅截断重要性比率。这有助于进一步稳定训练，特别是在策略差异较大的情况下。

### Token-Level MIS

在Token-Level MIS中，我们为每个token单独计算重要性比率，并基于该比率决定是否对该token的贡献进行掩码。对于序列中的第$t$个token $y_t$，其重要性比率为：

$$
\rho_t = \frac{\textcolor{blue}{\pi_{\text{learner}}}(y_t|x, y_{<t})}{\textcolor{red}{\pi_{\text{sampler}}}(y_t|x, y_{<t})}
$$

当 $\rho_t > C$ 时，我们将其对应的梯度贡献置零。因此，Token-Level MIS的策略梯度估计器为：

$$
g_{\text{tok-MIS}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \textcolor{red}{\pi_{\text{sampler}}}(\cdot|x)}\left[R(x, y) \cdot \sum_{t=0}^{|y|-1} \mathbb{I}\{\rho_t \leq C\} \cdot \rho_t \cdot \nabla_\theta \log \textcolor{blue}{\pi_{\text{learner}}}(y_t|x, y_{<t})\right]
$$

与Token-Level TIS相比，Token-Level MIS更加严格，因为它完全排除了那些可能引入不稳定性的token贡献，而不是仅仅截断比率值。

### MIS与TIS的比较

1. **方差控制**：
   - TIS通过截断操作限制了重要性比率的最大值，但仍然保留了所有样本的贡献
   - MIS通过完全移除高比率样本/令牌，从根本上消除了这些可能引入巨大方差的贡献

2. **偏差-方差权衡**：
   - TIS引入了一些偏差（通过截断），但保持了较低的方差
   - MIS可能引入更大的偏差（通过完全排除样本），但能够更有效地控制方差

3. **适用场景**：
   - 当策略差异相对较小且主要由少数极端比率主导时，MIS可能更有效
   - 当策略差异较为均匀分布时，TIS可能提供更好的偏差-方差平衡

### 在PPO中的应用

将MIS扩展到PPO算法中，我们可以得到相应的表达式：

对于Sequence-Level MIS-PPO：
$$
g_{\text{seq-MIS-PPO}}(\theta) = \mathbb{E}_{a \sim \textcolor{red}{\pi_{\text{sampler}}}(\theta_{\text{old}})}\left[\mathbb{I}\{\rho(a) \leq C\} \cdot \nabla_\theta \min\left(\rho(a) \hat{A},\ \text{clip}\left(\rho(a),\ 1 - \epsilon,\ 1 + \epsilon\right) \hat{A}\right)\right]
$$

对于Token-Level MIS-PPO：
$$
g_{\text{tok-MIS-PPO}}(\theta) = \mathbb{E}_{a \sim \textcolor{red}{\pi_{\text{sampler}}}(\theta_{\text{old}})}\left[\sum_{t=0}^{|a|-1} \mathbb{I}\{\rho_t \leq C\} \cdot \nabla_\theta \min\left(\rho_t \hat{A}_t,\ \text{clip}\left(\rho_t,\ 1 - \epsilon,\ 1 + \epsilon\right) \hat{A}_t\right)\right]
$$

其中 $\rho(a) = \frac{\textcolor{blue}{\pi_{\text{learner}}}(a|\theta)}{\textcolor{red}{\pi_{\text{sampler}}}(a|\theta_{\text{old}})}$，$\rho_t = \frac{\textcolor{blue}{\pi_{\text{learner}}}(a_t|\theta)}{\textcolor{red}{\pi_{\text{sampler}}}(a_t|\theta_{\text{old}})}$。

通过这种方式，MIS为处理训练-推理不匹配问题提供了另一种有效的算法级解决方案，能够与TIS形成互补，在不同场景下提供更好的稳定性和性能。


## Reference

- https://fengyao.notion.site/off-policy-rl

- https://fengyao.notion.site/flash-rl

- https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch
