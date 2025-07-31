# GAE 算法原理

GAE 是一种用于减少策略梯度估计方差的技术，它通过引入 λ 参数在偏差和方差之间进行权衡。

## 代码实现

### 初始化阶段

```python
with torch.no_grad():
    nextvalues = 0
    lastgaelam = 0
    advantages_reversed = []
    gen_len = token_level_rewards.shape[-1]
```

- `torch.no_grad()`: 禁用梯度计算，节省内存和计算资源，因为优势计算不需要梯度
- `nextvalues = 0`: 初始化下一个时间步的价值，从序列末尾开始为 0
- `lastgaelam = 0`: 初始化上一个时间步的 GAE 值
- `advantages_reversed = []`: 存储反向计算的优势值
- `gen_len`: 获取响应序列的长度

### 核心计算循环

```python
for t in reversed(range(gen_len)):
    delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
    lastgaelam_ = delta + gamma * lam * lastgaelam
```

**TD 误差计算**：

- `delta` 是时间差分（TD）误差：`r_t + γ * V(s_{t+1}) - V(s_t)`
- 这表示当前时间步的即时奖励加上折扣后的下一状态价值，减去当前状态的估计价值

**GAE 递推公式**：

- `lastgaelam_` 实现了 GAE 的核心递推：`A_t^{GAE} = δ_t + γλA_{t+1}^{GAE}`
- 其中 `γ` 是折扣因子，`λ` 是 GAE 参数，控制偏差-方差权衡

### 掩码处理机制

```python
# skip values and TD-error on observation tokens
nextvalues = values[:, t] * response_mask[:, t] + (1 - response_mask[:, t]) * nextvalues
lastgaelam = lastgaelam_ * response_mask[:, t] + (1 - response_mask[:, t]) * lastgaelam
```

这是关键的掩码处理逻辑：

- `response_mask` 标识哪些 token 属于模型生成的响应部分
- 对于非响应 token（如 prompt 部分或 EOS 后的 padding），保持之前的值不变
- 只在响应 token 上更新 `nextvalues` 和 `lastgaelam`

### 结果处理

```python
advantages_reversed.append(lastgaelam)
advantages = torch.stack(advantages_reversed[::-1], dim=1)

returns = advantages + values
advantages = verl_F.masked_whiten(advantages, response_mask)
```

- 将每个时间步的优势值添加到列表中
- `[::-1]` 将反向计算的结果重新排列为正向顺序
- `returns = advantages + values`: 计算回报值，这是 GAE 的标准做法
- `masked_whiten`: 对优势值进行标准化处理，只在有效 token 上进行



# GRPO 算法原理

GRPO 是一种基于组内相对比较的优势估计方法，它通过计算每个响应相对于同组其他响应的相对优势来进行策略优化。

## 代码实现

### 分数聚合阶段

```python
scores = token_level_rewards.sum(dim=-1)
```

首先将 token 级别的奖励沿着序列长度维度求和，得到每个响应的总分数。 core_algos.py:295

### 数据结构初始化

```python
id2score = defaultdict(list)
id2mean = {}
id2std = {}
```

- `id2score`: 存储每个组（由 index 标识）中所有响应的分数
- `id2mean`: 存储每个组的平均分数
- `id2std`: 存储每个组的标准差

### 分组统计计算

```python
with torch.no_grad():
    bsz = scores.shape[0]
    for i in range(bsz):
        id2score[index[i]].append(scores[i])
```

遍历批次中的每个样本，根据 `index` 数组将分数分组。`index` 数组标识哪些响应属于同一个 prompt 组。 core_algos.py:302-304

### 组内统计量计算

```python
for idx in id2score:
    if len(id2score[idx]) == 1:
        id2mean[idx] = torch.tensor(0.0)
        id2std[idx] = torch.tensor(1.0)
    elif len(id2score[idx]) > 1:
        id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
        id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
    else:
        raise ValueError(f"no score in prompt index: {idx}")
```

对每个组计算统计量：

- 如果组内只有一个响应，设置均值为 0，标准差为 1（避免除零）
- 如果组内有多个响应，计算实际的均值和标准差
- 如果组内没有响应，抛出错误 core_algos.py:305-313

### 优势值计算

```python
for i in range(bsz):
    if norm_adv_by_std_in_grpo:
        scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
    else:
        scores[i] = scores[i] - id2mean[index[i]]
```

根据 `norm_adv_by_std_in_grpo` 参数决定是否进行标准化：

- 如果为 True：计算标准化的优势值 `(score - mean) / std`（原始 GRPO）
- 如果为 False：只减去均值 `score - mean`（Dr.GRPO 变体） core_algos.py:314-318

### 掩码应用

```python
scores = scores.unsqueeze(-1) * response_mask
```

将标量优势值扩展到 token 维度，并应用响应掩码，确保只在有效的响应 token 上应用优势值。 core_algos.py:319

## 技术要点

### 1. 组内相对比较机制

GRPO 的核心思想是在同一个 prompt 的多个响应之间进行相对比较，而不是使用绝对的奖励值。

### 2. 标准化选项

通过 `norm_adv_by_std_in_grpo` 参数支持两种变体：

- 原始 GRPO：使用标准化优势值
- Dr.GRPO：只使用去中心化的优势值

### 3. 数值稳定性

使用 `epsilon` 参数避免除零错误，特别是在标准差很小的情况下。

### 与 GAE 的区别

与 GAE 不同，GRPO：

1. 不需要价值函数估计
2. 基于组内相对比较而非时序差分
3. 适用于结果级别的奖励（outcome reward）
4. 计算更简单，但需要同一 prompt 的多个响应

### Notes

GRPO 特别适用于需要在同一 prompt 的多个候选响应中进行选择的场景，如代码生成、数学推理等任务。该实现支持原始 GRPO 和 Dr.GRPO 两种变体，通过配置参数可以灵活切换。

# 不同形式的聚合 loss 分析

`agg_loss` 函数的作用是将二维的损失矩阵（批次大小 × 序列长度）聚合成一个标量损失值，支持多种聚合策略以适应不同的训练需求。

## 代码实现

### 函数签名和参数

```python
def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
```

- `loss_mat`: 形状为 `(bs, response_length)` 的损失矩阵，包含每个 token 的损失值
- `loss_mask`: 相同形状的掩码张量，标识哪些 token 是有效的（1 为有效，0 为无效）
- `loss_agg_mode`: 字符串参数，指定聚合方式

### 四种聚合模式详解

#### 1. token-mean 模式

```python
if loss_agg_mode == "token-mean":
    loss = verl_F.masked_mean(loss_mat, loss_mask)
```

这是最常用的模式，计算所有有效 token 的平均损失。使用 `verl_F.masked_mean` 函数确保只对掩码为 1 的 token 计算平均值。

#### 2. seq-mean-token-sum 模式

```python
elif loss_agg_mode == "seq-mean-token-sum":
    seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
    loss = torch.mean(seq_losses)  # seq-mean
```

这种模式分两步：

1. 对每个序列内的所有有效 token 损失求和（token-sum）
2. 对所有序列的损失求平均（seq-mean）

#### 3. seq-mean-token-mean 模式

```python
elif loss_agg_mode == "seq-mean-token-mean":
    seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # token-mean
    loss = torch.mean(seq_losses)  # seq-mean
```

这种模式也分两步：

1. 计算每个序列内有效 token 的平均损失（token-mean）
2. 对所有序列的平均损失再求平均（seq-mean）

#### 4. seq-mean-token-sum-norm 模式

```python
elif loss_agg_mode == "seq-mean-token-sum-norm":
    seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
    loss = torch.sum(seq_losses) / loss_mask.shape[-1]
```

这是为 Dr.GRPO 论文特别设计的模式：

1. 对每个序列内的有效 token 损失求和
2. 将所有序列损失的总和除以序列长度（而非有效 token 数量）

注释中提到，为了准确复现 Dr.GRPO 论文，除数应该在整个训练过程中保持常数。

## 技术要点

### 1. 掩码机制的重要性

所有聚合模式都考虑了掩码，确保只对有效的 token 进行计算，这在处理变长序列时至关重要。

### 2. 不同聚合策略的影响

- `token-mean`: 所有 token 权重相等
- `seq-mean-token-sum`: 长序列权重更大
- `seq-mean-token-mean`: 每个序列权重相等，不受长度影响
- `seq-mean-token-sum-norm`: 特殊的标准化方式

### 3. 数值稳定性

通过掩码操作避免了对无效 token 的计算，提高了数值稳定性。

## 配置和使用示例

在配置文件中，可以通过 `loss_agg_mode` 参数指定聚合方式： legacy_ppo_megatron_trainer.yaml:62

## Notes

`agg_loss` 函数是 VERL 框架中损失计算的核心组件，被广泛用于策略损失、价值损失和熵损失的聚合。不同的聚合模式适用于不同的训练场景，其中 `token-mean` 是默认和最常用的模式。该函数的设计充分考虑了序列长度变化和掩码处理，确保了训练的稳定性和有效性。



# Policy Loss 计算

这个函数实现了 PPO 算法的核心损失计算，包括标准的剪切策略目标函数和双重剪切机制。它被注册为 "vanilla" 策略损失函数，是 PPO 训练中最常用的损失计算方法。

## 代码实现

### 函数注册和签名

```python
@register_policy_loss("vanilla")
```

使用装饰器将此函数注册到策略损失注册表中，名称为 "vanilla"。这个注册机制允许训练器根据配置动态选择不同的损失函数。 core_algos.py:49-82

### 配置参数提取

```python
assert config is not None
assert not isinstance(config, AlgoConfig)
clip_ratio = config.clip_ratio
clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
clip_ratio_c = config.get("clip_ratio_c", 3.0)
```

从配置对象中提取 PPO 的关键超参数：

- `clip_ratio`: 标准 PPO 的剪切参数 ε
- `clip_ratio_low/high`: 用于非对称剪切的上下界
- `clip_ratio_c`: 双重剪切 PPO 的下界参数

### 概率比率计算

```python
negative_approx_kl = log_prob - old_log_prob
negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
ratio = torch.exp(negative_approx_kl)
ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
```

计算新旧策略的概率比率：

- `negative_approx_kl`: 近似 KL 散度的负值
- 使用 `torch.clamp` 确保数值稳定性，防止指数运算溢出
- `ratio`: 概率比率  π_θ(a|s) / π_θ_old(a|s)
- `ppo_kl`: 计算 KL 散度用于监控

### 标准 PPO 剪切损失

```python
pg_losses1 = -advantages * ratio
pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
```

实现标准 PPO 的剪切目标函数：

- `pg_losses1`: 未剪切的策略梯度损失
- `pg_losses2`: 剪切后的策略梯度损失
- `clip_pg_losses1`: 取两者的最大值（因为是负数，实际是最小化）
- `pg_clipfrac`: 计算剪切比例用于监控

### 双重剪切机制

```python
pg_losses3 = -advantages * clip_ratio_c
clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
pg_clipfrac_lower = verl_F.masked_mean(
    torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
)
```

实现双重剪切 PPO 的额外约束：

- `pg_losses3`: 使用 `clip_ratio_c` 的下界约束
- `clip_pg_losses2`: 应用下界剪切
- `pg_clipfrac_lower`: 计算下界剪切的比例

### 最终损失计算

```python
pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
```

根据优势值的正负选择不同的损失：

- 负优势值使用双重剪切损失
- 正优势值使用标准剪切损失
- 使用 `agg_loss` 函数将损失矩阵聚合为标量

## 技术要点

### 1. 装饰器模式

使用 `@register_policy_loss` 装饰器实现策略损失函数的动态注册，这是一种典型的工厂模式应用。

### 2. 数值稳定性

通过 `torch.clamp` 限制对数概率差值的范围，防止 `torch.exp` 计算时出现数值溢出。

### 3. 双重剪切机制

实现了标准 PPO 和双重剪切 PPO 的结合，对负优势值应用额外的下界约束，提高训练稳定性。

### 4. 掩码机制

所有统计计算都使用 `response_mask` 确保只在有效 token 上进行，这对处理变长序列至关重要。

# GSPO 计算分析

GSPO 是一种改进的策略优化算法，它通过计算序列级别的重要性比率来改进传统 PPO 的性能。该函数实现了 GSPO 的核心损失计算，包括序列级重要性比率计算和剪切机制。

## 代码实现

### 函数注册和参数验证

```python
@register_policy_loss("gspo")
```

使用装饰器将函数注册为 "gspo" 策略损失函数。 core_algos.py:49-82

```python
assert config is not None
assert isinstance(config, ActorConfig)
clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio
```

验证配置对象并提取剪切参数。与 vanilla PPO 不同，GSPO 要求使用 `ActorConfig` 类型的配置。

### KL 散度计算

```python
negative_approx_kl = log_prob - old_log_prob
```

计算新旧策略之间的近似 KL 散度（负值），这是所有策略优化算法的基础。

### 序列级重要性比率计算

```python
seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
negative_approx_kl_seq = torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths
```

这是 GSPO 的核心创新：

- `seq_lengths`: 计算每个序列的有效长度
- `negative_approx_kl_seq`: 计算序列级别的平均 KL 散度

根据论文，序列级重要性比率定义为：`si(θ) = (π_θ(yi|x)/π_θold(yi|x))^(1/|yi|)`

### Token 级组合比率计算

```python
log_seq_importance_ratio = log_prob - log_prob.detach() + negative_approx_kl_seq.detach().unsqueeze(-1)
log_seq_importance_ratio = torch.clamp(log_seq_importance_ratio, max=10.0)
seq_importance_ratio = torch.exp(log_seq_importance_ratio)
```

这里实现了 GSPO 的关键技术：

- 使用 `detach()` 实现停止梯度（stop gradient）操作
- 组合序列级和 token 级的重要性比率
- 数值稳定性剪切防止指数爆炸

公式为：`s_i,t(θ) = sg[s_i(θ)] · π_θ(y_i,t|x, y_i,<t) / sg[π_θ(y_i,t|x, y_i,<t)]`

### 策略梯度损失计算

```python
pg_losses1 = -advantages * seq_importance_ratio
pg_losses2 = -advantages * torch.clamp(seq_importance_ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
pg_losses = torch.maximum(pg_losses1, pg_losses2)
```

应用标准的 PPO 剪切机制，但使用 GSPO 的序列重要性比率替代传统的概率比率。

### 损失聚合和指标计算

```python
pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode="seq-mean-token-mean")
```

GSPO 强制使用 "seq-mean-token-mean" 聚合模式，这确保每个序列的权重相等。 core_algos.py:699-732

## 技术要点

### 1. 停止梯度机制

使用 `detach()` 实现停止梯度，这是 GSPO 算法的关键技术，防止某些项参与梯度计算。

### 2. 序列级建模

与传统 PPO 的 token 级建模不同，GSPO 引入了序列级的重要性比率，更好地捕捉序列整体的质量。

### 3. 数值稳定性

通过 `clamp(max=10.0)` 防止指数运算导致的数值溢出。

### 4. 固定聚合模式

GSPO 要求使用特定的损失聚合模式，确保算法的正确性。

# GMPO  计算分析

GMPO 是一种基于几何平均的策略优化算法，它通过在 token 级别进行剪切并使用几何平均来计算序列级别的重要性比率。该函数实现了 GMPO 的核心损失计算，与传统 PPO 相比，它在序列级别聚合优势值和概率比率。

## 代码实现

### 函数注册和参数验证

```python
@register_policy_loss("geo_mean")
```

使用装饰器将函数注册为 "geo_mean" 策略损失函数。 core_algos.py:49-82

```python
assert config is not None
assert not isinstance(config, AlgoConfig)
clip_ratio = config.clip_ratio
clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
```

验证配置对象并提取剪切参数。与 GSPO 不同，GMPO 不允许使用 `AlgoConfig` 类型的配置。

### KL 散度和剪切范围设置

```python
negative_approx_kl = log_prob - old_log_prob
ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
```

计算新旧策略之间的近似 KL 散度，并计算用于监控的 KL 值。注意这里没有对 `negative_approx_kl` 进行数值稳定性剪切（代码中被注释掉了）。

### Token 级别剪切机制

```python
sgn_advantage = torch.sign(advantages)
negative_approx_kl_clamp = torch.clamp(negative_approx_kl, -cliprange_low, cliprange_high)
negative_approx_kl_min = torch.min(sgn_advantage * negative_approx_kl, sgn_advantage * negative_approx_kl_clamp)
negative_approx_kl_min = sgn_advantage * negative_approx_kl_min
```

这是 GMPO 的关键创新：

- `sgn_advantage`: 获取优势值的符号（正负）
- `negative_approx_kl_clamp`: 对 KL 散度进行剪切
- `negative_approx_kl_min`: 根据优势值符号选择原始值或剪切值的最小值
- 最后乘以符号恢复原始的正负性

### 几何平均策略优化

```python
response_mask_sum = response_mask.sum(dim=-1)
ratio = torch.exp((negative_approx_kl_min * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8))
advantage = (advantages * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8)
pg_losses = -advantage * ratio
pg_loss = torch.mean(pg_losses)
```

这是 GMPO 的核心计算：

- 计算每个序列的有效长度
- 使用几何平均计算序列级别的重要性比率：`exp(平均对数比率)`
- 将优势值聚合到序列级别
- 计算策略梯度损失并取平均

### 剪切统计计算

```python
clipped = torch.ne(negative_approx_kl, negative_approx_kl_clamp)
pg_clipfrac = verl_F.masked_mean((clipped * (advantages > 0)).float(), response_mask)
pg_clipfrac_lower = verl_F.masked_mean((clipped * (advantages < 0)).float(), response_mask)
```

计算剪切统计信息：

- `clipped`: 标识哪些位置被剪切了
- `pg_clipfrac`: 正优势值被剪切的比例
- `pg_clipfrac_lower`: 负优势值被剪切的比例

## 技术要点

### 1. 几何平均机制

GMPO 使用几何平均而非算术平均来计算序列级别的重要性比率，这通过 `torch.exp(平均对数值)` 实现。

### 2. Token 级别剪切

与传统 PPO 在比率上剪切不同，GMPO 在对数空间的 KL 散度上进行剪切，然后再转换为比率。

### 3. 符号感知剪切

使用 `torch.sign()` 和 `torch.min()` 实现符号感知的剪切机制，确保正负优势值得到不同处理。

### 4. 序列级别聚合

GMPO 强制在序列级别聚合优势值和比率，而不是在 token 级别。



# KL 散度计算

这个函数实现了多种 KL 散度计算方法，用于在 PPO 训练中对策略进行正则化，防止新策略偏离参考策略过远。它支持 5 种不同的 KL 散度近似方法。

## 代码实现

### 标准 KL 散度 ("kl", "k1")

```python
if kl_penalty in ("kl", "k1"):
    return logprob - ref_logprob
```

这是最简单的近似方法，直接计算对数概率差值。这是一阶近似，计算效率最高但可能不够精确。

### 绝对值近似 ("abs")

```python
if kl_penalty == "abs":
    return (logprob - ref_logprob).abs()
```

使用绝对值来避免负值，提供对称的惩罚。

### 均方误差近似 ("mse", "k2")

```python
if kl_penalty in ("mse", "k2"):
    return 0.5 * (logprob - ref_logprob).square()
```

二阶近似方法，对大的偏差给予更重的惩罚。

### 低方差 KL 散度 ("low_var_kl", "k3")

```python
if kl_penalty in ("low_var_kl", "k3"):
    kl = ref_logprob - logprob
    kl = torch.clamp(kl, min=-20, max=20)
    ratio = torch.exp(kl)
    kld = (ratio - kl - 1).contiguous()
    return torch.clamp(kld, min=-10, max=10)
```

这是最精确的近似方法，实现了真正的 KL 散度公式 `KL(π_ref || π) = E[log(π_ref/π) - log(π_ref/π) - 1]`。使用了两次数值稳定性剪切来防止溢出。

## 技术要点

### 1. 数值稳定性

函数在 "low_var_kl" 模式中使用了两次 `torch.clamp` 操作，确保指数运算和最终结果都在安全范围内。

### 2. 多种近似方法

提供了从简单到复杂的多种 KL 散度近似，允许用户根据计算资源和精度需求选择合适的方法。

### 3. 配置灵活性

通过配置参数 `kl_penalty` 可以选择不同的计算方法，在 ppo_trainer.yaml:159 中可以看到默认使用 "kl" 方法。
