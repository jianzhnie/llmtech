# 从 GRPO 到 DAPO 和 GSPO

在大型语言模型的强化学习阶段，PPO 曾经是主流方法。然而，它依赖于价值模型，在处理长文本输出和复杂任务时显示出局限性。GRPO 消除了对价值模型的依赖，显著提高了可扩展性，但在效率和稳定性方面仍有优化空间。这促使了 DAPO 的出现，它细化了采样、裁剪和梯度计算等细节。然而，在动态激活专家的 MoE（混合专家模型）架构中，GRPO 框架下的Token-Level优化仍然难以稳定收敛。GSPO 进一步将优化粒度转移到 Sequence-Level，从根本上减少了高方差和结构噪声。本文遵循这一进化路径：从 GRPO 开始，逐步揭示 DAPO 和 GSPO 背后的设计动机和实现细节。

在接下来的文章中，您将发现：

1. 为什么 GRPO 摆脱了 PPO 对价值模型的依赖，但在某些情况下仍然可能“崩溃”。
2. Clip-Higher 如何修复优质Token过早被截断的隐患
3. 动态采样如何防止无效样本造成大量计算浪费。
4. Token-Level梯度损失如何确保长响应不再稀释有价值的梯度信号。
5. 为什么 GRPO 的逐Token重要性采样在 MoE 架构中产生巨大方差。
6. GSPO 如何用Sequence-Level优化替换Token-Level优化，从根本上提高稳定性和效率。


## Dual-Clip PPO

Dual-Clip PPO（双裁剪近端策略优化）是标准 **PPO（Proximal Policy Optimization）** 算法的一种改进版本，主要目的是**更有效地处理优势函数（Advantage, $A_t$）为负数的情况**，从而提高算法的稳定性和性能。

### 标准 PPO-Clip 概述 (背景)

为了理解 Dual-Clip，我们首先回顾标准 PPO-Clip 的目标函数。PPO 目标函数（最大化形式）：

$$
L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t) \right]
$$

其中：

- $r_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ 是**重要性采样比率**。
- $A_t$ 是**优势函数估计**。
- $\epsilon$ 是**裁剪超参数** (如 0.1 或 0.2)。

这个目标函数 $\min(\ldots)$ 的作用是：

1. **当 $A_t \ge 0$ (动作好于平均)**：我们希望提高 $r_t(\theta)$ (即增加该动作的概率 $\pi_{\theta}$)。但 $\min$ 操作会限制 $r_t(\theta)$ **不能超过 $1+\epsilon$**。
2. **当 $A_t < 0$ (动作差于平均)**：我们希望降低 $r_t(\theta)$ (即减小该动作的概率 $\pi_{\theta}$)。但 $\min$ 操作会限制 $r_t(\theta)$ **不能低于 $1-\epsilon$**。

标准 PPO 的缺陷 (针对 $A_t < 0$)：

- 当 $A_t$ 为负时，我们希望策略更快地减少这个糟糕动作的概率（即让 $r_t(\theta)$ 趋近于 0）。然而，标准 PPO 的 $\min$ 操作会将目标函数限制在 $r_t(\theta) = 1-\epsilon$ 处，即使 $r_t(\theta)$ 进一步减小（趋近于 0），目标函数也不会再增加，导致策略对负优势的动作修改不够彻底。

### Dual-Clip PPO 的核心改进

Dual-Clip PPO 引入了第三个裁剪项 $L_3$，主要作用于 $A_t < 0$ 的情况，以确保在负优势下，策略能够更严格地被惩罚。Dual-Clip PPO 目标函数（最大化形式）：
$$
L^{DC}(\theta) = \mathbb{E}_t \left[ \min(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t, \quad L_3) \right]
$$

其中，第三项 $L_3$ 如下定义：
$$L_3 = C \cdot A_t$$
- $C = \text{clip\_ratio\_c}$ 是一个新的超参数，通常设置为 2.0 ~ 5.0（原文建议 3.0）

### Dual-Clip 的逻辑分解

Dual-Clip 目标函数可以分解为两种情况：

#### 🟢 **情况一：优势函数 $A_t \ge 0$ (有利动作)**

$$
L^{DC}(\theta) = \mathbb{E}_t \left[ \min(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t) \right]
$$

行为： 此时 $L_3 = C \cdot A_t \ge 0$。由于 $C > 1+\epsilon$ (通常如此)，且 $A_t \ge 0$，因此 $L_3$ 会大于或等于 $r_t(\theta) A_t$ 和 $\text{clip}(\ldots) A_t$。

结论： 在 $A_t \ge 0$ 时，Dual-Clip 等同于标准 PPO-Clip。

#### 🔴 **情况二：优势函数 $A_t < 0$ (不利动作)**

$$
L^{DC}(\theta) = \mathbb{E}_t \left[ \max(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t, \quad C \cdot A_t) \right]
$$

(注：当 $A_t < 0$ 时，最大化 $\min(r A_t, \ldots)$ 相当于最大化 $\max(r A_t, \ldots)$。)

**行为：**

1. **$r_t(\theta)$ 减小 (期望行为)**：我们希望 $r_t(\theta)$ 变小（趋近于 0），此时 $r_t(\theta) A_t$ 和 $\text{clip}(\ldots) A_t$ 都会**增大**（因为 $A_t$ 是负数）。标准 PPO 的目标会被 $\text{clip}(\ldots) A_t$ 在 $r_t(\theta) = 1-\epsilon$ 处钳制住。
2. **$r_t(\theta)$ 增大 (不期望行为)**：如果 $r_t(\theta)$ 不幸增大（策略变差），目标函数会减小。
3. **$L_3$ 的作用**：由于 $A_t < 0$ 且 $C > 1$，我们有 $C \cdot A_t$ 是一个**比 $A_t$ 负得更厉害**的数。当 $r_t(\theta)$ **超出 $C$** 时（即 $r_t(\theta) > C$），则 $r_t(\theta) A_t$ 会**小于** $C \cdot A_t$。

通过下表对比分析在不同情况下 Dual-Clip 如何影响更新：

| 场景                                                        | 标准 PPO 行为                | Dual-Clip 行为                                 |
| ----------------------------------------------------------- | ---------------------------- | ---------------------------------------------- |
| **好动作，适度提升** ($\hat{A} > 0$, $r \in [0.8,1.2]$)     | 使用原始梯度                 | 相同                                           |
| **好动作，大幅提升** ($\hat{A} > 0$, $r > 1.2$)             | 使用裁剪梯度（限制提升）     | 相同                                           |
| **坏动作，适度抑制** ($\hat{A} < 0$, $r < 0.8$)             | 使用裁剪梯度（加强惩罚）     | 相同                                           |
| **坏动作，大幅偏离但优势高估** ($\hat{A} < 0$, $r \gg 1.2$) | ❌ 仍允许更新（可能错误强化） | ✅ 强制限制为 $ c \cdot \hat{A} $，防止过度纠正 |

Dual-Clip 的实际限制：

在 $A_t < 0$ 时，它有效地将重要性采样比率 $r_t(\theta)$ 强行限制在了 $[0, C]$ 的范围内。如果 $r_t(\theta)$ 试图增加到超过 $C$，则 $r_t(\theta) A_t$ 会变得比 $C \cdot A_t$ 更小，$\max(\ldots)$ 操作会选择 $C \cdot A_t$。这提供了一个更严格的上限 $C$ 来惩罚那些试图大幅度增加糟糕动作概率的新策略。

**核心总结:** Dual-Clip PPO 的 $L_3$ 项 $(C \cdot A_t)$ 旨在防止新策略 $\pi_\theta$ **过度增加**那些在旧策略 $\pi_{\theta_{old}}$ 下已经被判断为**负优势**的动作的概率。它提供了一个比标准 PPO 更安全的上限。

### 代码实现要点（Verl 框架）

```python
# Step 1: Standard PPO clipping
pg_losses1 = -advantages * ratio
pg_losses2 = -advantages * torch.clamp(ratio, 1 - clip_low, 1 + clip_high)
clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)

# Step 2: Upper-bound clipping (Dual-Clip)
pg_losses3 = -advantages * clip_ratio_c
clip_pg_losses2 = torch.min(clip_pg_losses1, pg_losses3)

# Step 3: Final selection based on advantage sign
pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
```

> 🧠 **关键点**：只有当优势为负时才启用上界裁剪，因为正优势时我们希望鼓励好动作，不需要额外限制。


| **代码片段**                                                 | **作用**                                                     | **Dual-Clip 关键点**                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `pg_losses1 = -advantages * ratio`                           | 计算 $-\min(r_t(\theta) A_t, \ldots)$ 中的第一项的**负数形式**。 | 对应 $-r_t(\theta) A_t$。                                    |
| `pg_losses2 = -advantages * torch.clamp(...)`                | 计算第二项的**负数形式**（标准裁剪）。                       | 对应 $-\text{clip}(\ldots) A_t$。                            |
| `clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)`    | **标准 PPO 目标损失**（负数形式）：$\max(-r_t(\theta) A_t, -\text{clip}(\ldots) A_t)$。 | 对应 $\min(r_t(\theta) A_t, \text{clip}(\ldots) A_t)$ 的负数。 |
| `pg_losses3 = -advantages * clip_ratio_c`                    | 计算 $L_3$ 的**负数形式**。                                  | 对应 $-C \cdot A_t$。                                        |
| `clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)`   | 结合 $L_3$ 项：$\min(-C A_t, \quad -\min(\ldots))$。这是 Dual-Clip 目标损失的**负数形式**。 | 这是 **$A_t < 0$** 时的目标，即 $\max(C A_t, \min(\ldots))$ 的负数。 |
| `pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)` | **最终选择**：$A_t < 0$ 时用 Dual-Clip 损失 (`clip_pg_losses2`)；$A_t \ge 0$ 时用标准 PPO 损失 (`clip_pg_losses1`)。 | 实现了逻辑分离，确保 $L_3$ 只影响负优势下的策略更新。        |

### 总结

Dual-Clip PPO 通过引入 $C \cdot A_t$ 这一项，使得算法在处理不利动作时更加保守和稳定，这在一些复杂的强化学习任务（特别是自然语言处理中的人类反馈强化学习，RLHF）中表现出更好的性能。

| **特性**        | **标准 PPO-Clip**                                            | **Dual-Clip PPO**                                            |
| --------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **目标**        | 限制策略更新，防止步长过大。                                 | 在保持 PPO 优点的基础上，**更严格地限制负优势下 $r_t(\theta)$ 的增加**。 |
| **$A_t \ge 0$** | **上限：** $r_t(\theta) \le 1+\epsilon$。                    | **上限：** $r_t(\theta) \le 1+\epsilon$ (与标准 PPO 相同)。  |
| **$A_t < 0$**   | **下限：** $r_t(\theta) \ge 1-\epsilon$ (减小 $r_t(\theta)$ 的动力不足)。 | **上限：** $r_t(\theta) \le C$。如果 $r_t(\theta)$ 试图增加超过 $C$，会受到更严厉的惩罚 |

#### 🎯   适用场景

Dual-Clip PPO 特别适用于以下情况：

- **稀疏奖励环境**：容易出现优势估计偏差
- **长序列生成任务**（如对话、代码生成）：价值函数难以准确建模
- **离线强化学习**（Offline RL）：数据分布固定，需防止离策略更新过激
- **大模型 RLHF 训练**：防止语言模型“钻牛角尖”生成奇怪但高 reward 的文本

#### ✅ 实践建议：

- 在标准 PPO 训练不稳定时尝试启用 Dual-Clip；
- 设置 `clip_ratio_c = 3.0` 作为起点；
- 监控 `pg_clipfrac_lower` 指标，若过高说明上界频繁触发，可能需要调整 $ c $ 或检查优势估计质量。



## GRPO 回顾

GRPO 的训练目标是：

$$
J_{\text{GRPO}}(\theta) = \mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(O \mid q)} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left( \min \left( r_{i,t}(\theta) A_i, \text{clip}(r_{i,t}(\theta), 1-\varepsilon, 1+\varepsilon) A_i \right) - \beta \, D_{\text{KL}}(\pi_{\theta} \parallel \pi_{\text{ref}}) \right) \right]
$$
其中

$$
r_{i,t}(\theta) = \frac{\pi_{\theta}(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})}
$$

$$
A_i = \frac{r_i - \text{mean}(\{r_1, r_2, \ldots, r_G\})}{\text{std}(\{r_1, r_2, \ldots, r_G\})}
$$

在理解 GRPO 目标之后，我们首先需要澄清重要性采样的作用和局限性，这不仅对理解 GRPO 至关重要，也是 DAPO 和 GSPO 引入改进的切入点。

### 重要性比率扮演什么角色？

重要性采样的本质是，我们希望在新分布下计算期望，但我们的数据是从旧分布中抽取的。因此，我们使用新旧策略下相同动作的概率比率作为校正权重：

$$
\mathbb{E}_{\text{pnew}}[f(x)] = \mathbb{E}_{\text{pold}}\left[ \frac{p_\text{pnew}(x)}{p_\text{old}(x)} f(x) \right]
$$
这使我们能够使用旧策略的离线数据评估新策略下的期望值，避免了每次更新后重新采样的需要（从而降低成本）。然而，如果新旧策略之间的差距过大，权重的方差可能变得非常高，导致训练不稳定。

在 RL 训练里，引入重要性采样的初衷其实是为了提高样本的使用效率 —— 在标准的策略梯度方法中，每个数据样本只进行一次梯度更新不同，PPO 与其不同，是在得到一批采样数据后，进行多次的优化更新，这样固然可以提高样本的利用率，但是同时也带来了采样策略和当前策略分布差异的问题，“重要性权重” 的引入就是为了解决这种偏差。

重要性采样的目的是在我们只有行为分布样本的情况下，估计目标分布下的期望。在 PPO/GRPO 中，我们并不直接从新策略中采样数据；相反，我们首先使用旧策略生成数据（因为采样成本高），这个过程称为 **rollout**。在更新时，我们必须修正分布不匹配的问题，这就是重要性采样的用武之地。在采样后为每个Token定义重要性比率为：

$$
r_t = \frac{\pi_{\theta}(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}
$$
PPO/GRPO 目标可以写成：
$$
L(\theta) = \mathbb{E}_t \left[ \min(r_t A_t, \text{CLIP}(r_t, 1-\varepsilon, 1+\varepsilon) A_t) \right]
$$
这里，$A_t$是计算得到的优势，而裁剪操作通过限制更新幅度来防止策略与旧策略产生过大偏离。

有了这种重要性采样的直观理解，我们可以进一步考虑它在 PPO/GRPO 中的实际效果：优势函数 $A_t$的符号和比率 $r_t$一起决定了策略更新的方向和幅度。

### $A_t$和 $r_t$的符号如何影响训练？

让我们分析这些情况。

- 假设 $A_t > 0$（动作比预期更好）
  - 我们希望增加这个动作的概率。如果我们在裁剪中设置 $\varepsilon = 0.2$，那么当 $r_t > 1.2$时，`min` 和 `clip` 操作将其限制在 1.2。当 $r_t < 0.8$时，由于 `min` 操作，不发生裁剪，因此正优势的变化受到限制。

- 当 $A_t < 0$（动作比预期更差）
  - 我们应该减少这个动作的概率。如果 $r_t < 0.8$，`min` 操作进一步限制它，限制在 0.8$A_t$；但当 $r_t > 1.2$时，`min` 操作不施加限制（它可以上升到 +∞，带负号变为 -∞）。因此，负优势的向下调整也受到限制。

- $A_t$ 衡量当前动作/轨迹是否优于或劣于平均水平

  - 如果 $A_t$为正，我们鼓励它；

  - 如果是负的，我们惩罚它，以便它在未来出现的更少。

- 重要性比率 $r_t$ 反映了新策略选择这个动作的概率相较于旧策略的增加（或减少）程度。

  - 如果 $r_t > 1$，新模型更倾向于该动作；

  - 如果 $r_t < 1$，倾向性降低。

- 在 $A_t$和 $r_t$的四种可能符号组合中，我们只希望有两种：
  - 当它们符号相同时，正 $A_t$ 与 $r_t > 1$（加强）
  - 负 $A_t$ 与 $r_t < 1$（修正错误）。

然而，匹配 $A_t$ 和 $r_t$ 的符号还不够。在 PPO/GRPO 中，**裁剪操作** 对于稳定训练同样至关重要，因为它决定了哪些Token的梯度真正有助于更新。

### 裁剪对梯度和Token效率的影响

对于 $A_t > 0$，当 $r_t > 1+\varepsilon$ 时，即增长达到上限时，我们应用裁剪操作，此时梯度归零。这实际上取消了该Token对训练的贡献。同样，对于 $A_t < 0$，如果 $r_t < 1-\varepsilon$，即下降幅度超出限制，裁剪裁剪同样会使梯度归零。一个常见的误解是裁剪使用直通估计器将裁剪值的梯度传递回未裁剪值；实际上，这并没有发生：裁剪前的梯度直接设置为零。

此时，我们对 GRPO 的机制、优势和局限性有了相对完整的理解。接下来，我们将看到 DAPO 如何在保留 GRPO 基本框架的同时，引入更细粒度的改进来解决效率和稳定性挑战。

## 从 GRPO 到 DAPO

DAPO 从一个简单的动机开始：在实际训练中，GRPO 经常由于裁剪范围不合理、冗余采样和长序列中的梯度稀释等问题而浪费大量学习信号。DAPO 通过四个针对性改进来解决这些问题。

$$
J_{\text{DAPO}}(\theta) = \mathbb{E}_{(q,a) \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(O \mid q)} \left[ \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min(r_{i,t}(\theta) A_i, \text{clip}(r_{i,t}(\theta), 1-\varepsilon_{\text{low}}, 1+\varepsilon_{\text{high}}) A_i) \right]
$$

$$
s.t.,  0 < \left| \{o_i \mid \text{is\_equivalent}(a, o_i)\} \right| < G
$$

### DAPO - Clip-Higher

- 为什么 DAPO 提高上限 $1+\varepsilon_{\text{high}}$同时保持 $1-\varepsilon_{\text{low}}$固定？

作者观察到，选择一个小的 $\varepsilon$ 作为裁剪上限可能导致以下问题：如果旧策略为采样Token分配了非常低的概率，但其优势是正的（意味着旧模型采样了优质内容），当前策略几乎没有空间增加其概率，而这恰恰是我们期望实现的目标。

例如，如果旧策略的概率是 0.9 且 $\varepsilon = 0.2$，上限是 0.9×1.2=1.08，已经超过了最大概率 1.0，所以它永远不会被裁剪。但如果旧策略的概率是 0.2，上限变为 0.24。在这种情况下，即使当前策略将概率提高到 0.4（一个很好的改进），过小的 $\varepsilon$导致它被裁剪，有效地丢弃了该Token。这就是为什么 DAPO 采用 **Clip-Higher**，提高上限提高了Token效率。

这本质上就是我们所说的“马太效应”：*富者愈富，穷者难以改善*。如果旧策略几乎以非常低的概率采样了一个关键Token，比如说 `"Wait"`，但当前模型显著增加了该概率，它仍然可能被裁剪掉，剥夺了模型“扭转局面”的机会。

Clip-Higher 解决了“好Token过早被限制”的问题，但它并没有解决另一个常见的浪费来源：缺乏采样多样性。为了解决这个问题，DAPO 引入了 **动态采样**。

### DAPO - 动态采样

DAPO 的第二个创新是 **动态采样**。动机如下：假设对于给定的查询我们采样了 10 个响应，所有 10 个响应要么非常好要么非常差，始终获得最大奖励或零奖励。由于 GRPO 的计算方法，所有 10 个样本的优势为零，因此贡献的梯度为零。

这意味着实际贡献梯度的有效样本数量远低于名义采样量，从而导致高方差、训练不稳定及样本浪费。这种效应在训练初期（模型性能较差时）和后期（模型已能频繁生成完美回答时）表现得尤为明显。

为应对此问题，DAPO 实施了一项附加抽样规则：对于每个查询，抽样得到的响应集合不得全为 0 分或全为 1 分。若所有样本均为 0 分或均为 1 分，则需持续补充抽样直至打破该条件。该约束条件具体表述为：
$$
s.t., 0 < \left| \{o_i \mid \text{is\_equivalent}(a, o_i)\} \right| < G
$$
这确保了对于相同的输入，采样集包含正确和错误的答案。

除了采样多样性，GRPO 对于长响应还有另一个隐藏缺陷：**随着响应长度的增加，Token梯度被稀释**。DAPO 的第三个改进通过 **Token-Level梯度损失** 来解决这个问题。

### DAPO - Token-Level梯度损失

DAPO 的第三项创新解决了 GRPO 中存在的梯度权重问题：即每个Token的梯度权重随着采样响应长度的增加而减少。

为什么会这样？假设我们采样两次：一个响应有 200 个Token，另一个有 10 个Token。在 GRPO 的公式中，我们首先在每个样本内平均梯度，然后在批次中平均。这使得第一个响应中的每个Token的权重为 (1/200)×(1/2)，而第二个响应中的每个Token获得 (1/10)×(1/2)。因此，较短响应的Token影响更大。

这种机制的缺陷显而易见：

- 对于复杂问题，长回答本属常态。若这些长回答质量高，其宝贵的梯度信号会被稀释；
- 若因重复累赘导致回答冗长且质量低下，修正信号同样会减弱。

DAPO 的解决方案：在计算梯度时，对所有样本生成的总Token数进行平均。在我们的示例中，长响应和短响应都给每个Token一个权重 1/(200+10)。这种方法平等对待所有Token，提高了长样本训练的效率。

这对应于从 GRPO 的损失聚合更改为 DAPO 的：

$$
\frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|}
$$
到 DAPO 的：
$$
\frac{1}{\sum_{i=1}^G |o_i|} \sum_{i=1}^G \sum_{t=1}^{|o_i|}
$$
经验上，Token-Level损失能带来更稳定的训练过程，防止熵变得过高（导致策略随机行动），并避免熵过低时的探索崩溃（Clip-Higher 也有助于解决）。通过从样本级损失转变为Token-Level损失，DAPO 确保长文本响应能按比例影响最终梯度：每个Token 都会直接影响整体梯度，且不受样本长度制约。

最后一个改进也涉及响应长度，但采取了不同视角：**过长响应对整体奖励的负面影响**。

### DAPO - 过长奖励塑形

DAPO 的第四个改进使用 **软惩罚** 机制调整过长响应的奖励。具体来说，一旦生成的序列超过预定义的第一个长度阈值，就会对Token进行惩罚，随着长度的增加，惩罚线性增加。如果长度超过第二个阈值，惩罚力度将足以抵消正确答案带来的原始奖励，有效地模拟了过长响应被视为无效的情况。

通过 Clip-Higher、动态采样、Token-Level梯度损失和过长奖励塑形，DAPO 提供了 GRPO 的细粒度改进，显著提高了训练效率和稳定性。然而，在某些架构中，特别是 MoE 中，GRPO 仍然存在 DAPO 无法完全解决的结构问题，这引导我们走向 GSPO。

## GSPO：解决 MoE 训练中的 GRPO 不稳定性

若将 DAPO 视为 GRPO 框架内的“微调与精炼”，那么 GSPO 则迈出了更为根本的一步：它将优化粒度从Token-Level转变为Sequence-Level。这一转变背后的动机是，在 MoE 架构训练期间，GRPO 的重要性采样会引入较大方差与不稳定性。GSPO 的核心思想是在奖励处理过程中减少对逐Token优化的依赖，转而更注重整体序列结果。接下来我们将阐述 GSPO 背后的核心概念。

> **TL;DR：** 传统的算法如 PPO 和 GRPO 通常单独优化模型输出中的每个Token，给一些Token更高的权重，给其他Token更低的权重。虽然这旨在进行细粒度优化，但在长文本、大型模型场景中，它反而可能引入噪声和奖励偏差，导致模型失去方向，甚至突然崩溃。问题的根源在于我们根据完整响应评估模型，但逐个Token进行训练，导致奖励粒度和优化目标之间的不匹配。GSPO 通过从每个Token评分转向Sequence-Level优化来对齐奖励和优化目标。这一转变提供了两个主要好处：

> 1. **稳定性** – GSPO 优化整个序列，减少了Token-Level波动的训练噪声。
> 2. **效率** – GSPO 过滤并仅保留高质量样本进行优化，加速收敛并提高结果。
>
>    在 MoE 架构中，好处更大：由于每次推理只激活一小部分专家模块，路由路径是动态的且难以控制。传统方法通常依赖于 **Routing Replay**，在推理期间记录专家激活并在训练期间强制使用相同的路由路径，以确保一致性。虽然有效，但这大大增加了工程成本并限制了性能。GSPO 的Sequence-Level逻辑自然避免了对 Routing Replay 的需求，使 MoE 训练更轻量且更稳定。对于越来越多的大型 MoE 模型，这是一个有价值的突破。例如，QWen3 系列已经采用了 GSPO。从 PPO → GRPO → GSPO，我们看到 LLM 的 RL 优化目标应与任务的性质紧密对齐，同时保持训练逻辑简单、可扩展和可部署。进步通常不是由复杂的技巧驱动的，而是通过对核心问题的洞察。

PPO 在长文本和复杂任务中挣扎的主要原因是它依赖于价值模型：当策略模型输出长序列时，价值估计变得不准确，使得从简单任务泛化到复杂任务变得困难。GRPO 消除了这种依赖，打破了价值模型的瓶颈。然而，GRPO 在 MoE 训练或长时间训练运行中仍然面临稳定性问题：在某个时刻，模型可能会突然崩溃，即使恢复训练或调整参数也常常无法恢复。接下来，让我们分析可能的原因和解决方案。

### 重要性比率扮演什么角色，为什么在 GRPO 中存在问题？

重要性采样允许我们在只有行为分布样本的情况下，估计目标分布下的期望。我们通过在目标策略和行为策略之间的概率比率对样本进行加权来实现这一点。然而，这种校正假设有多个样本，如果只有一个样本，它无法有效地调整分布变化。

在大型模型训练中的问题是在 **每个Token** 上执行重要性采样，单个Token的比率无法有意义地执行分布校正。相反，它引入了高方差噪声，特别是在不稳定的 MoE 设置中。这表明 GRPO 的Token-Level计算可能在本质上是次优的。

另一个不匹配：我们的奖励是针对 **整个响应**（Sequence-Level）给出的，但在Token-Level重要性采样中，我们将此奖励均匀地分配给Token（奖励塑形）并尝试单独调整它们。这在奖励信号和优化目标之间创建了粒度不匹配。鉴于我们已经拥有Sequence-Level奖励，为什么不也让 GRPO 的优化成为Sequence-Level？

### 为什么 GRPO 在 MoE 架构中难以收敛？

**专家激活波动：** 新旧策略可能激活不同的专家，引入结构偏差和噪声。当 $\pi_{\theta_{\text{old}}}$ 更新时，路由机制也可能改变，因此两个策略可能激活完全不同的专家集，即使只过了一个训练步骤。这导致输出概率大幅波动，异常频繁地触发裁剪。被裁剪的Token不贡献梯度，而那些保留的Token通常包含噪声。

理论上，重要性比率应该反映在 **相同** 结构下由参数更新引起的 **概率变化**。但专家变化导致不可预测的、与优化方向无关的高方差波动。这种方差扭曲了策略梯度估计，使训练不稳定，甚至导致崩溃。

### GSPO 之前的 Routing Replay

Routing Replay 在从 $\pi_{\theta_{\text{old}}}$ 采样期间记录专家激活，并强制 $\pi_{\theta}$ 在训练期间使用相同的路由路径。缺点是：高工程和基础设施成本，以及效率低下，$\pi_{\theta}$ 可能找到了更好的路由路径，但被迫遵循旧的。

虽然传统方法使用 Routing Replay 来减轻专家激活不匹配，但 GSPO 完全绕过了这种依赖，从根本上减少了结构方差。

### GSPO Sequence-Level 的优化目标

设$ x $为查询，$ \pi_{\theta_{\text{old}}} $ 为用于采样回复的策略，$ o_{i\_i} = 1^G $为采样得到的回复组，$ \hat{A}_i $为各个回复的组内相对优势，$ \pi_{\theta} $为需优化的当前策略。GSPO 采用以下优化目标：
$$
\mathcal{J}_{\text{GSPO}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(\cdot \mid x)} \left[ \frac{1}{G} \sum_{i=1}^G \min \left( s_i(\theta) \hat{A}_i, \text{clip} \left( s_i(\theta), 1-\varepsilon, 1+\varepsilon \right) \hat{A}_i \right) \right],
$$
其中

$$
s_i(\theta) = \left( \frac{\pi_{\theta}(y_i \mid x)}{\pi_{\theta_{\text{old}}}(o_i \mid x)} \right)^{\frac{1}{|o_i|}} = \exp \left( \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \log \frac{\pi_{\theta}(o_{i,t} \mid x, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid x, o_{i,<t})} \right).
$$

这里的$ s_i(\theta) $即为 GSPO 基于序列似然定义的重要性比率，其中我们进行了长度归一化以降低方差并统一$ s_i(\theta) $的数值范围。

> 如果奖励是Sequence-Level的，重要性比率也应该是Sequence-Level的

从上述内容中，GSPO 使用Sequence-Level比率 $s_i(\theta)$ 替换了 GRPO 逐Token比率 $r_{i,t}(\theta)$，它不再与步骤索引 $t$ 绑定。这个想法是放弃Token-Level目标，转而采用Sequence-Level缩放。这自然导致了 GSPO 的新优化目标：用Sequence-Level的重要性比率替换Token-Level的。

Sequence-Level比率是经过**长度归一化** 的，以减少方差并保持数值在统一量级。如果没有归一化，不同长度的答案会使比率对长度非常敏感。由于同一序列中的所有Token共享相同的重要性比率，如果触发裁剪，将裁剪 **整个序列**，而不仅仅是某些Token。归一化因子 $\frac{1}{|o_i|}$ 也防止了长序列中的少数波动Token导致比率爆炸。

**为什么使用指数而不是直接使用对数似然差异？**

指数是必要的，因为重要性采样的核心公式是：

$$
\mathbb{E}_{z \sim \pi_{\text{tar}}}[f(z)] = \mathbb{E}_{z \sim \pi_{\text{beh}}}\left[ \frac{\pi_{\text{tar}}(z)}{\pi_{\text{beh}}(z)} f(z) \right]
$$
这里，权重必须是 **概率比率**（$\geq 0$），而不是对数概率差异。如果我们直接使用 $\Delta \log p$，它将等同于：

$$
\mathbb{E}[\Delta \log p \cdot A]
$$
这不再是一个无偏的重要性采样校正。

GSPO 在对数空间中通过 $\frac{1}{|o_i|}$ 归一化，然后指数化：

$$
s_i(\theta) = \exp \left( \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \log \frac{\pi_{\theta}(y_{i,t} \mid x, y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t} \mid x, y_{i,<t})} \right).
$$
这确保了不同序列长度的重要性比率的一致缩放，避免了长序列中少数Token概率变化导致的极端值。停留在对数空间而不进行指数化将使比率对长度敏感，需要调整裁剪范围，并破坏与 PPO/GRPO 中使用的 KL 正则化的兼容性。

### 理论梯度分析：GSPO 与 GRPO

从客观定义来看，关键差异在于重要性比率在梯度计算中的定义与使用方式。

在不进行裁剪的情况下，区别在于是否在同一响应内对Token进行不同的加权。GRPO 根据 $r_{i,t}(\theta)$ 为每个Token分配独立的权重，而 GSPO 对序列中的所有Token应用相同的 $s_i(\theta)$。

GSPO 的梯度：

$$
\nabla_{\theta} J_{\text{GSPO}}(\theta) = \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^G s_i(\theta) \frac{A_i}{|o_i|} \sum_{t=1}^{|o_i|} \nabla_{\theta} \log \pi_{\theta}(o_{i,t} \mid q, o_{i,<t}) \right]
$$
这里，响应中的所有Token共享相同的权重 $\frac{s_i(\theta) A_i}{|o_i|}$，确保了序列内梯度的一致性。

GRPO 的梯度：

$$
\nabla_{\theta} J_{\text{GRPO}}(\theta) = \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^G A_i \frac{|o_i|}{|o_i|} \sum_{t=1}^{|o_i|} r_{i,t}(\theta) \nabla_{\theta} \log \pi_{\theta}(o_{i,t} \mid q, o_{i,<t}) \right]
$$
这里，权重 $\frac{r_{i,t}(\theta) A_i}{|o_i|}$ 随Token位置和上下文变化，导致更高的方差，特别是在长序列或 MoE 模型中。

另一个区别在于剪裁处理与这些比率之间的交互方式。对于正优势样本，GRPO 的比率范围大约是 [0, 1.x]；对于负优势样本，它可以是 [0.x, ∞)，范围更广。在长序列中，这种不对称性的噪声可能会累积，导致 GRPO 在 MoE 下的不稳定性。

奖励指标也在检测模型漂移方面滞后，当问题出现时，模型可能已经偏离了一段时间。实验表明，GSPO 以较少的有效Token（由于更积极的裁剪）训练，但实现了更高的训练效率。

总之，GSPO 实现了序列内一致的梯度权重，减少了Token间的方差，特别适合长序列和 MoE 场景下的稳定训练。它的引入标志着从 PPO → GRPO → GSPO 的转变，从依赖价值模型的Token-Level优化转向与任务性质对齐的Sequence-Level优化。

## Dr.GRPO

**Response-level length bias**: 在计算每个 response 的 loss 时，将除以的 |oi| 也就是 response length 替换为一个固定值 MAX_TOKENS，即训练时设置的 response 的最大长度；

```python
# GRPO
def masked_mean ( tensor , mask , dim ) :
	return ( tensor * mask ).sum ( axis = dim ) / mask .sum( axis = dim)

# Dr.Grpo
def masked_mean ( tensor , mask , dim ) :
	return ( tensor * mask ).sum ( axis = -1) / MAX_TOKENS
```

Dr.GRPO 对于 逐Token Loss 的处理方式 和 DAPO 的 Token-Level 对比.

关于 response-level length bias 这一问题，在DAPO 中也有基本相同的陈述，但是两者在改进方式上略有差异，区别就在于在处理不同的 group 时，Dr.GRPO 始终使用的是同一个除数 G * max_length, 而 DAPO 是使用的各个group 内部的总 token 数，那么这两种方式孰优孰劣呢？

个人倾向于 Dr.GRPO 的做法，不同的group，也就是不同 prompt 产出的 response 的长度天然是有差异的，在group 之间同样存在长度偏差的问题（平均Response长度较短的 Prompt，在总体损失中的贡献要更大，但很明显，这种学习趋势是不符合我们的预期的） DAPO的做法则忽略了这种差异。

## CISPO



## Reference：

- https://huggingface.co/blog/NormalUhr/grpo-to-dapo-and-gspo
