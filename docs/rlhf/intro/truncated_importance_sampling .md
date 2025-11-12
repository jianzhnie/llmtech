# 缓解 Rollout-训练不匹配问题

## 简而言之

大模型强化学习微调不稳定的一个关键来源：**训练-推理不匹配（training-inference mismatch）**。为了最大化训练效率，现代强化学习训练框架（如 VeRL）通常会采用两种不同的计算引擎：一种是为快速推理（rollout）高度优化的引擎（如 vLLM），另一种是为梯度计算设计的训练引擎（如 FSDP）。尽管这两种引擎在数学原理上是等价的，但由于浮点数精度误差和硬件层面的具体优化差异，它们会产生数值上不完全相同的输出。近期的一系列研究已经指出，这种看似微不足道的不匹配，会在优化过程中引入显著的问题，是导致训练不稳定的核心因素之一。

## 不匹配问题

为简化起见，我们以 REINFORCE 算法为例，该算法通过以下方式更新策略 $\pi$ ——一个由 $\theta$ 参数化的 LLM：

$$
\theta \leftarrow \theta + \mu \cdot \underbrace{\mathbb{E}*{a \sim \pi(\theta)}}*{\text{rollout}}[R(a) \cdot \underbrace{\nabla_\theta \log \pi(a, \theta)}_{\text{training}}].
$$

实践中，轨迹生成成本高昂，现代强化学习框架（例如 VeRL）通常采用高度优化的推理引擎（例如 vLLM、SGLang）来提升吞吐量，同时使用独立后端（例如 FSDP、Megatron）进行模型训练。这种混合设计使得更新过程：

$$
\theta \leftarrow \theta + \mu \cdot \mathbb{E}*{a \sim \pi*{\text{sampler}}(\theta)}[R(a) \cdot \nabla_\theta \log \pi_{\text{learner}}(a, \theta)].
$$

此处我们使用 $\pi_{\text{sampler}}$ 表示搭载推理引擎（如 vLLM、SGLang）的模型，$\pi_{\text{learner}}$ 表示使用训练后端（如 FSDP、Megatron）实例化的同模型。若无特别说明，我们的实验均采用 vLLM 作为采样器后端、FSDP 作为训练器后端。

实验中观察到意外的 **rollout 训练失配现象**。如图 1 所示，尽管 $\pi_{\text{fsdp}}$ 与 $\pi_{\text{vlm}}$ 共享相同模型参数 $\theta$，它们却可能生成显著不同的Token概率。对于某些特定Token $a$，甚至会产生相互矛盾的预测结果，即 $\pi_{\text{vlm}}(a, \theta) = 1$ 与 $\pi_{\text{fsdp}}(a, \theta) = 0$。这种异常行为隐式破坏了同策略假设，实质上使 On-Policy 强化学习训练悄然转变为异策略模式。

### 优化驱动的恶性循环

人们可能认为训练-推理失配是硬件与软件栈的静态特性。然而，我们后续的"批次过滤"实验证明**这种失配与训练动态及模型状态相互耦合**。

我们推测这是由于以下两阶段级联故障所致：

1. **阶段一：数值敏感度增强**。强化学习优化器将模型权重推至 `bfloat16` 数据类型相对精度较低的数值范围（例如极小或极大值）。
2. **阶段二：内核驱动的误差放大**。这些初始微小的 `bfloat16` 量化误差随后被输入 vLLM 和 FSDP 的不同内核实现。差异化的计算顺序充当非线性放大器，使初始微小偏差最终雪崩式扩大为最终逻辑值的巨大差异。

这形成了一个**恶性反馈循环**：失配导致有偏且含噪的梯度，可能将参数进一步推向数值敏感区域，进而加剧下一轮迭代的失配程度，直至系统彻底崩溃。

## 缓解训练-推理失配的尝试

接下来我们将列举为缓解训练-推理失配所尝试的方法。其中部分方法有所助益，另一些则收效甚微。

### 使用 FP32 lm_head

受 *Minimax-M1* 技术报告及博客文章《你的高效 RL 框架正在悄悄进行离线策略训练》启发，我们通过修改 vLLM 将 lm_head 转换为 fp32 精度。但在实验中，修补后失配问题依然存在，模型崩溃仍不可避免。

### 禁用分块预填充

我们还尝试通过禁用分块预填充来验证是否能解决崩溃问题。然而，实验结果显示（该方法并未解决崩溃问题)。

### 启用 `enforce_eager` 与 `free_cache_engine`

VeRL 官方提供的 DAPO 方案指出，启用 CUDA 图（`enforce_eager=False`）可能导致模型性能下降。为探究这是否会影响训练-推理失配问题，我们通过消融实验研究了 vLLM 引擎超参数 `enforce_eager` 的影响，并同步考量另一超参数 `free_cache_engine`。实验结果显示，调整 `enforce_eager` 与 `free_cache_engine` 的取值对训练-推理失配现象及测试性能均无显著影响。

## 重要性采样

### 理论解决方案：重要性采样

训练-推理失配将原本同策略的强化学习问题转化为异策略问题，其中用于生成轨迹的策略（行为策略，$\pi_\theta^{\text{vllm}}$）与正在训练的策略（目标策略，$\pi_\theta^{\text{fsdp}}$）存在差异。理论上校正这种分布偏移的正规方法是**重要性采样**（IS）。然而，IS 的具体形式对于保持无偏梯度和实现稳定训练至关重要。

受 **[Yao 等, 2025]** 首次揭示这一隐式异策略问题的研究启发，我们分析了两种主要的 IS 形式：理论完备的**Seqence-Level IS** 与常见但存在缺陷的**Token-Level IS** 近似——后者也是该文献中探讨的启发式方法。

### Seqence-Level 重要性采样

正确且无偏的策略梯度估计器在整个生成序列（轨迹）上应用单一重要性比率 $y$。这种方法能准确地将行为策略的期望值重新加权为目标策略，从而得到目标函数的真实梯度 $J(\theta)$。

让我们逐步推导**Seqence-Level重要性采样**估计器 $g_{\text{seq}}(\theta)$。

- 目标是在目标 FSDP 策略下最大化期望奖励：

$$
J(\theta) = \mathbb{E}*{x \sim \mathcal{D}, y \sim \pi*\theta^{\text{fsdp}}(\cdot|x)}[R(x, y)]
$$

- 因此真实策略梯度为：

$$
g(\theta) = \nabla_\theta J(\theta) = \mathbb{E}*{x \sim \mathcal{D}, y \sim \pi*\theta^{\text{fsdp}}(\cdot|x)}\left[R(x, y)\nabla_\theta \log \pi_\theta^{\text{fsdp}}(y|x)\right]
$$

- 由于我们只能从 vLLM 策略中采样，故使用重要性采样来改变期望的分布：

$$
g_{\text{seq}}(\theta) = \mathbb{E}*{x \sim \mathcal{D}, y \sim \pi*\theta^{\text{vllm}}(\cdot|x)}\left[\frac{\pi_\theta^{\text{fsdp}}(y|x)}{\pi_\theta^{\text{vllm}}(y|x)} \cdot R(x, y) \cdot \nabla_\theta \log \pi_\theta^{\text{fsdp}}(y|x)\right]
$$

该估计器在数学上等价于标准优势函数形式的策略梯度。关键在于证明重要性采样比率能精确修正期望值，揭示底层真实的同策略梯度，进而可对其进行优化。

此推导最终得到策略梯度的优势函数形式：

$$
g_{\text{seq}}(\theta) = \mathbb{E}*{s \sim d*{\pi_\theta^{\text{fsdp}}}} \mathbb{E}*{a \sim \pi*\theta^{\text{fsdp}}(\cdot|s)}\left[A_\theta^{\text{fsdp}}(s, a) \cdot \nabla_\theta \log \pi_\theta^{\text{fsdp}}(a|s)\right]
$$

此处 $s = (x, y_{<t})$ 表示状态（前缀），$a = y_t$ 表示动作（Token）。项 $d_{\pi_\theta^{\text{fsdp}}}$ 为目标 FSDP 策略下的**状态占用度量**，其正式定义为遵循策略 $\pi$ 时期望访问状态 $s$ 的次数：

$$
d_\pi(s) := \mathbb{E}*{x' \sim \mathcal{D}, y' \sim \pi(\cdot|x')} \left[ \sum*{t'=0}^{|y'|-1} \mathbb{I}{(x', y'*{<t'}) = s} \right] = P(x) \cdot \prod*{k=0}^{t-1} \pi(y_k|x, y_{<k})
$$

该估计器是无偏的，这意味着 $g_{\text{seq}}(\theta) = g(\theta)$。为确保数值稳定性，采用**截断重要性采样**（TIS）方法，该方法将Seqence-Level比率 $\rho(y|x)$ 限制在常数 $C$ 以内。

### Token-Level 重要性采样

一种常见启发式方法，通常受到 PPO 等算法的启发并在 (Yao 等人, 2025) 中使用，采用逐词元重要性比率。虽然这通常比Seqence-Level比率具有更低的方差，但它是一种有偏估计器，对于自回归模型在理论上并不严谨。

让我们推导**Token-Level重要性采样**梯度估计器 $g_{\text{tok}}(\theta)$。

- 该公式通过错误地在时间步求和和内部应用重要性采样比率开始：即 $g_{\text{tok}}(\theta)$ 被定义为

  $$
  \mathbb{E}*{x \sim \mathcal{D}, y \sim \pi*\theta^{\text{vllm}}(\cdot|x)}\left[R(x, y) \cdot \sum_{t=0}^{|y|-1} \frac{\pi_\theta^{\text{fsdp}}(y_t|x, y_{<t})}{\pi_\theta^{\text{vllm}}(y_t|x, y_{<t})} \cdot \nabla_\theta \log \pi_\theta^{\text{fsdp}}(y_t|x, y_{<t})\right]
  $$

- 我们可以将此轨迹期望重写为在 vLLM 策略下访问状态的期望。

  $$
  g_{\text{tok}}(\theta) = \mathbb{E}*{s \sim d*{\pi_\theta^{\text{vllm}}}} \mathbb{E}*{a \sim \pi*\theta^{\text{vllm}}(\cdot|s)}\left[\frac{\pi_\theta^{\text{fsdp}}(a|s)}{\pi_\theta^{\text{vllm}}(a|s)} \cdot A^{\text{vllm}}(s, a) \cdot \nabla_\theta \log \pi_\theta^{\text{fsdp}}(a|s)\right]
  $$

> 注：此处 $R(x, y)$ 表示由 $\pi_\theta^{\text{vllm}}$ 采样的完整轨迹所得的经验回报，作为状态-动作价值函数 $Q^{\pi_\theta^{\text{vllm}}}(s, a)$ 的蒙特卡洛估计值。通过引入基线函数并改变动作期望的计算方式，最终得到如下形式：

$$
g_{\text{tok}}(\theta) = \mathbb{E}*{s \sim d*{\pi_\theta^{\text{vllm}}}} \mathbb{E}*{a \sim \pi*\theta^{\text{fsdp}}(\cdot|s)}\left[A^{\text{vllm}}(s, a) \cdot \nabla_\theta \log \pi_\theta^{\text{fsdp}}(a|s)\right]
$$

最终表达式清晰地揭示了Token-Level重要性采样的梯度偏差。

### Token-Level 重要性采样的偏差来源

将 $g_{\text{tok}}(\theta)$ 与真实梯度 $g_{\text{seq}}(\theta)$ 进行对比，可发现两个显著差异导致的误差，使得Token-Level估计量存在偏差。

#### 误差源 1：状态访问分布失配 🌍

有效的离策略修正必须考虑两种分布偏移：动作概率分布与状态访问概率分布。词元级方法仅修正了前者。

- **真实梯度**（$g_{\text{seq}}$）：期望计算基于正确目标 fsdp 分布下的状态访问，$\mathbb{E}*{s \sim d*{\pi_\theta^{\text{fsdp}}}}$。
- **缺陷梯度**（$g_{\text{tok}}$）：期望计算基于错误行为 vLLM 分布下的状态访问，$\mathbb{E}*{s \sim d*{\pi_\theta^{\text{vllm}}}}$。

该方法隐含假设状态访问比率为 1，即 $d_{\pi^{\text{fsdp}}}(s)/d_{\pi^{\text{vllm}}}(s) = 1$。在自回归模型中该假设会被严重违背：由于确定性状态转移，单个词元选择差异就会导致状态轨迹完全发散。忽略这一事实使得 $g_{\text{tok}}(\theta)$ 引入了巨大且不可控的偏差。

#### 误差源 2：失配奖励信号 🎯

第二个关键错误在于，词元级梯度使用错误策略的奖励信号来加权更新。

- **真实梯度**（$g_{\text{seq}}$）：该更新通过目标全分片数据并行策略的优势函数 $A_{\pi_\theta^{\text{fsdp}}}$ 进行缩放，该函数代表在该策略下的预期未来奖励。
- **有缺陷的梯度**（$g_{\text{tok}}$）：该更新由行为 vLLM 策略的优势函数进行缩放，$A_{\pi_\theta^{\text{vllm}}}$。

目标策略的梯度正在被属于行为策略的奖励信号所缩放。由于状态分布和奖励信号存在根本性不匹配，Token-Level梯度实际上是一个有偏且理论不稳健的估计量。

> 🔧 **这些理论表明，尽管Token-Level方法可能具有较低的方差，但梯度偏差仍然存在，可能导致训练不稳定——这一预测在我们的实验中得到了Token-Level我们还针对令牌级和序列级方法提出了详细的偏差与方差分析（第一部分和第二部分）。**

#### 缓解系统级失配

更高精度的 vLLM 是否有效？我们最初假设 vLLM 是问题根源，因此通过补丁修复了两个常被怀疑导致失配问题的因素。

- **不可获取的真实采样概率**：vLLM v1 引擎 **不支持** 直接返回用于采样的调整后概率，这引入了额外差异。
  → 我们的补丁强制 vLLM 返回实际用于采样的概率 [已上游合并]。
- **后端数值差异**：vLLM 的 lm_head 精度与 HuggingFace transformers 不匹配，该问题在 MiniMax-M1 技术报告中亦有提及。
  → 我们的补丁提供了将 vLLM 的 lm_head 强制转换为 fp32 的选项。

### 接纳失配——实施算法级修复

### 重要性采样

当直接对目标分布下的期望值进行蒙特卡洛估计较为困难时，重要性采样允许我们从替代分布中进行抽样。在我们的场景中，目标分布是 $\pi_{\text{learner}}$，但从中抽样极其缓慢。使用独立后端（如 vLLM）进行轨迹生成意味着我们实际上是从 $\pi_{\text{sampler}}$ 进行抽样。此时通过重要性权重对每个样本进行加权修正偏差：

#### 解耦式 PPO

**解耦 PPO** 是运用重要性采样弥合轨迹生成与梯度计算间隙的特殊案例，该方法已被 **AReaL** 等异步强化学习框架采用。需要特别说明的是，AReaL 并未实现我们讨论的截断重要性比率方案，而是当重要性比率超过预设阈值时直接丢弃整个训练样本。

### 截断重要性采样 TIS

不同于在系统层面缓解分布失配，我们提出通过调整模型更新机制使其感知这种失配。简单的方法是采用重要性采样校正。具体而言，我们通过在当前梯度计算中添加重要性比率来处理 $\pi_{\text{learner}}$ 与 $\pi_{\text{sampler}}$ 之间的失配，即将当前梯度计算从

$$
\mathbb{E}*{a \sim \pi*{\text{sampler}}(\theta)}[R(a) \cdot \nabla_\theta \log \pi_{\text{learner}}(a, \theta)],
$$

到

$$
\mathbb{E}*{a \sim \pi*{\text{sampler}}(\theta)}\left[\frac{\pi_{\text{learner}}(a, \theta)}{\pi_{\text{sampler}}(a, \theta)} \cdot R(a) \cdot \nabla_\theta \log \pi_{\text{learner}}(a, \theta)\right].
$$

尽管关于如何设计稳定有效的重采样方法已有广泛研究，但在实践中我们发现通常采用经典技术——**截断重要性采样**便已足够：

$$
\mathbb{E}*{a \sim \pi*{\text{sampler}}(\theta)}\left[\underbrace{\min\left(\frac{\pi_{\text{learner}}(a, \theta)}{\pi_{\text{sampler}}(a, \theta)}, C\right)}*{\text{truncated importance ratio}} \cdot R(a) \cdot \nabla*\theta \log \pi_{\text{learner}}(a, \theta)\right],
$$

其中 $C$ 是一个超参数。

### 扩展至其他算法

将上述分析扩展到其他算法是直截了当的，因为我们可以将梯度计算的具体形式从 REINFORCE 的 $R(a) \cdot \nabla_\theta \log \pi(a, \theta)$ 切换为任意形式。在此，我们以常用的 PPO 算法为例，提供类似的分析作为补充说明。

PPO 的策略梯度 $\nabla_\theta L^{\text{CLIP}}(\theta)$ 定义为：

$$
\mathbb{E}*{a \sim \pi*{\text{old}}}\left[\nabla_\theta \min\left(\frac{\pi_\theta(a)}{\pi_{\theta_{\text{old}}}(a)} \hat{A},\ \text{clip}\left(\frac{\pi_\theta(a)}{\pi_{\theta_{\text{old}}}(a)},\ 1 - \epsilon,\ 1 + \epsilon\right) \hat{A}\right)\right].
$$

为提升吞吐量，混合强化学习系统采用 vLLM 引擎进行推演生成——从 $\pi_{\theta_{\text{old}}}$ 中采样Token $a$，同时使用 FSDP 后端从 $\pi_\theta$ 进行采样，并 **重新计算** $\pi_{\theta_{\text{old}}}$ 的Token概率以完成梯度计算：

$$
\mathbb{E}*{a \sim \pi*{\text{sampler}}(\theta_{\text{old}})}\left[\nabla_\theta \min\left(\frac{\pi_{\text{learner}}(a, \theta)}{\pi_{\text{learner}}(a, \theta_{\text{old}})} \hat{A},\ \text{clip}\left(\frac{\pi_{\text{learner}}(a, \theta)}{\pi_{\text{learner}}(a, \theta_{\text{old}})},\ 1 - \epsilon,\ 1 + \epsilon\right) \hat{A}\right)\right],
$$

与上述分析类似，$\pi_{\text{learner}}$ 与 $\pi_{\text{sampler}}$ 之间的差距再次显现，我们通过截断重要性采样方法予以修正：

$$
\mathbb{E}*{a \sim \pi*{\text{sampler}}(\theta_{\text{old}})}\left[\min\left(\frac{\pi_{\text{learner}}(a, \theta)}{\pi_{\text{sampler}}(a, \theta)}, C\right) \cdot \nabla_\theta \min\left(\frac{\pi_{\text{learner}}(a, \theta)}{\pi_{\text{learner}}(a, \theta_{\text{old}})} \hat{A},\ \text{clip}\left(\frac{\pi_{\text{learner}}(a, \theta)}{\pi_{\text{learner}}(a, \theta_{\text{old}})},\ 1 - \epsilon,\ 1 + \epsilon\right) \hat{A}\right)\right],
$$

其中 $C$ 是一个超参数。

## 与两种 TIS 变体的比较

我们还总结了两种用于缓解分布差距的替代方案。

- **PPO 重要性采样 (PPO-IS)**

$$
\mathbb{E}*{a \sim \pi*{\text{sampler}}(\theta_{\text{old}})} \left[ \nabla_\theta \min\left( \frac{\pi_{\text{learner}}(a, \theta)}{\pi_{\text{sampler}}(a, \theta_{\text{old}})} \hat{A}, \text{clip}\left( \frac{\pi_{\text{learner}}(a, \theta)}{\pi_{\text{sampler}}(a, \theta_{\text{old}})}, 1 - \epsilon, 1 + \epsilon \right) \hat{A} \right) \right]
$$

  *注意：Colossal 框架使用此实现。*

- **基础重要性采样 (vanilla-IS)**

  $$
  \mathbb{E}*{\pi*{\text{vlm}}(\theta_{\text{old}})} \left[ \underbrace{\frac{\pi_{\text{fsdp}}(a, \theta_{\text{old}})}{\pi_{\text{vlm}}(a, \theta_{\text{old}})}} \cdot \nabla_\theta \min\left( \frac{\pi_{\text{fsdp}}(a, \theta)}{\pi_{\text{fsdp}}(a, \theta_{\text{old}})} \hat{A}, \text{clip}\left( \frac{\pi_{\text{fsdp}}(a, \theta)}{\pi_{\text{fsdp}}(a, \theta_{\text{old}})}, 1 - \epsilon, 1 + \epsilon \right) \hat{A} \right) \right]
  $$

  *注意：Nemo-RL 使用此实现。*

为评估 TIS 的有效性并理解其设计选择的影响，我们进行了对比 TIS 与上述两种变体的实验。TIS 始终优于两种变体，尤其在差异显著的情况下（如 FP8/INT8 量化场景）表现更为突出。

## vanilla-IS 对比 TIS

关于**基础重要性采样**（vanilla-IS），其不稳定性主要源于当 $ a \sim \pi_{\text{sampler}}(a, \theta_{\text{old}}) $ 轨迹采样概率较低时，重要性比率会大幅增加，通过 $ \left( \frac{\pi_{\text{learner}}(a, \theta_{\text{old}})}{\pi_{\text{sampler}}(a, \theta_{\text{old}})} \right)^2 $ 放大梯度方差。为此，我们在截断重要性采样（TIS）中采用钳位操作以稳定训练。例如当单个Token的比率 $ \frac{\pi_{\text{learner}}(a, \theta_{\text{old}})}{\pi_{\text{sampler}}(a, \theta_{\text{old}})} $ 达到 16 时，该Token的梯度噪声将通过**原始重要性采样**放大 256 倍，通过 **TIS-2** 放大 4 倍，或通过 **TIS-8** 放大 64 倍。

## PPO-IS 对比 TIS

采用 **PPO-IS** 方法后，梯度实际上仍会偏离 PPO 的同策略版本。换言之，尽管该方法可能仍在朝着无偏目标进行优化，但相比标准 PPO 算法其效率可能有所不足。

此外需要说明的是，PPO 信任域技术的提出旨在将轨迹采样 $ \theta_{\text{old}} $ 与当前模型 $ \theta $ 之间的概率比约束在接近 1 的范围内，以近似同策略 REINFORCE 梯度。但在 **PPO-IS** 中，即便当 $ \theta = \theta_{\text{old}} $ 时，由于策略不匹配，概率比 $ \frac{\pi_{\text{learner}}(a, \theta)}{\pi_{\text{sampler}}(a, \theta_{\text{old}})} $ 仍不等于 1——这导致裁剪操作极易被触发，从而大幅降低训练的信息有效性。而在我们的 TIS 方法中，我们分别对 $ \frac{\pi_{\text{learner}}(a, \theta_{\text{old}})}{\pi_{\text{sampler}}(a, \theta_{\text{old}})} $ 和 $ \frac{\pi_{\text{learner}}(a, \theta)}{\pi_{\text{learner}}(a, \theta_{\text{old}})} $ 进行裁剪，因此更为温和；值得注意的是当 $ \theta = \theta_{\text{old}} $ 时，$ \frac{\pi_{\text{learner}}(a, \theta)}{\pi_{\text{learner}}(a, \theta_{\text{old}})} $ 恒等于 1，这恰好符合信任域约束的要求。

### TIS 工作机制的直观解释

虽然 TIS 的确切机制仍是待解之谜，我们对其缓解分布差异的原理提供高层级阐释。

特别需要注意的是，忽略具有 $\frac{\pi_{\text{learner}}(a, \theta_{\text{old}})}{\pi_{\text{sampler}}(a, \theta_{\text{old}})} < 1$ 的 rollout 偏差可能通过以下机制导致熵崩溃：对于具有负优势值的 rollout，策略梯度往往会降低 $\pi_{\text{learner}}$。当参数更新后存在较大分布差异时，$\pi_{\text{learner}}$ 的减少可能无法体现在 $\pi_{\text{sampler}}$ 中。因此策略梯度持续指向进一步降低 $\pi_{\text{learner}}$ 的方向。直观来看，这种惩罚机制可能迫使模型过度集中于熵值较小的输出分布。

与此同时，TIS 坚持对 $\frac{\pi_{\text{learner}}(a, \theta_{\text{old}})}{\pi_{\text{sampler}}(a, \theta_{\text{old}})} < 1$ 采用未截断的重要性比率，从而消除了这部分轨迹的偏差，并打破了这一机制。

## 截断重要性采样 (TIS)

在我们的**早期博客**中，我们解释了**截断重要性采样**如何减少由推理和训练引擎差异引起的 Rollout-训练不匹配问题。在这里，我们应用相同的技术来解决一个更大的不匹配问题，即 Rollout 使用低精度以提高速度，而训练引擎保持高精度。

### 理解 TIS 的速查表

- **期望策略梯度 (Expected Policy Gradient)**

  $$
  \mathbb{E}*{a \sim \pi*{\text{fsdp}}(\theta_{\text{old}})} \left[ \nabla_\theta \min\left( \frac{\pi_{\text{fsdp}}(a, \theta)}{\pi_{\text{fsdp}}(a, \theta_{\text{old}})} \hat{A}, \text{clip}\left( \frac{\pi_{\text{fsdp}}(a, \theta)}{\pi_{\text{fsdp}}(a, \theta_{\text{old}})}, 1 - \epsilon, 1 + \epsilon \right) \hat{A} \right) \right]
  $$

- **VeRL/OpenRLHF 的实现 (重计算)**

  $$
  \mathbb{E}*{a \sim \pi*{\text{vilm}}(\theta_{\text{old}})} \left[ \nabla_\theta \min\left( \frac{\pi_{\text{fsdp}}(a, \theta)}{\pi_{\text{fsdp}}(a, \theta_{\text{old}})} \hat{A}, \text{clip}\left( \frac{\pi_{\text{fsdp}}(a, \theta)}{\pi_{\text{fsdp}}(a, \theta_{\text{old}})}, 1 - \epsilon, 1 + \epsilon \right) \hat{A} \right) \right]
  $$

- **截断重要性比 (TIS):**

  $$
  \mathbb{E}*{\pi*{\text{vilm}}(\theta_{\text{old}})} \left[ \underbrace{\min\left( \frac{\pi_{\text{fsdp}}(a, \theta_{\text{old}})}{\pi_{\text{vilm}}(a, \theta_{\text{old}})}, C \right)} \cdot \nabla_\theta \ \min\left( \frac{\pi_{\text{fsdp}}(a, \theta)}{\pi_{\text{fsdp}}(a, \theta_{\text{old}})} \hat{A}, \text{clip}\left( \frac{\pi_{\text{fsdp}}(a, \theta)}{\pi_{\text{fsdp}}(a, \theta_{\text{old}})}, 1 - \epsilon, 1 + \epsilon \right) \hat{A} \right) \right]
  $$

Reference

- https://fengyao.notion.site/off-policy-rl

- https://fengyao.notion.site/flash-rl

- https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch
