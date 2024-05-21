

## RLHF 三阶段

RLHF 最突出的应用之一是使大语言模型能够与复杂的人类价值观对齐， 让大语言模型 (LLM) 变得更靠谱、更精准、更合乎伦理。

根据 OpenAI 的思路，RLHF分为三步：

### Supervised Fine-Tuning(SFT)

在步骤 SFT 中，采用有监督的方式对预训练的语言模型进行微调。这又被称为行为克隆（Behavioral Cloning，简称BC），即直接使用专家的行为数据（例如，专家在特定情况下采取的动作）来训练模型。在行为克隆中，模型的目标是尽可能地复制专家的行为，而不是尝试优化某种奖励函数，所以它可能无法处理那些专家数据中没有覆盖到的情况，因为它完全依赖于专家的行为数据。

### Reward Model（RM）

RM，奖励模型（Reward Model）的目标是训练一个模型来适应人类的偏好。在这个阶段，首先从提示库中进行采样，并使用大型语言模型生成多个响应。然后，人工对这些响应进行排名，根据这些排名训练一个奖励模型。

奖励模型的目标是学习人类对于不同响应的偏好，并将这些偏好编码到模型中。这样，奖励模型可以用来为模型生成的新响应打分，从而在后续的训练中引导模型生成更符合人类偏好的内容。这种方式不仅能帮助模型处理训练数据中未覆盖的情况，也能减少模型生成不确定或模棱两可的回答，从而打破行为克隆的影响。

### RL & Policy Optimization （RLHF）

RLHF 通过引入奖励信号来调整模型的行为，使模型生成的内容更符合人类的偏好。具体来说，在训练过程中，通过最大化预期奖励来调整模型的策略，使模型在选择行为时更倾向于选择可以得到更高奖励的行为。

在这个阶段中，我们首先使用在第一阶段训练的有监督微调模型和第二阶段训练的奖励模型来生成一个初始的策略。然后，我们使用PPO算法来调整这个策略，使模型在生成内容时更考虑人类的偏好。通过这个阶段的训练，模型不仅可以理解人类的语言，还可以理解人类的偏好，并生成更符合人类偏好的内容。

##  RLHF 的整体架构

PPO 是一种用于训练强化学习模型的算法。它可以用于调整语言模型，使得模型生成的结果更符合人类的偏好。具体来说，过程可以分为三个阶段：

- Rollout and Evaluation：
  - 在这个阶段，我们从 Prompt 库里抽样，使用语言模型生成response，然后使用奖励模型（Reward Model, RM）给出奖励得分。

- Make experience：
  - 在这个阶段，我们收集了一系列的“经验”，即模型的行为和对应的奖励。这些经验包括了模型生成的response 以及对应的奖励得分。这些经验将被用于下一步的优化过程。

- Optimization：
  - 在这个阶段，我们使用收集到的经验来更新模型的参数。具体来说，我们使用PPO算法来调整模型的参数，使得模型生成的 response的奖励得分能够增加。PPO算法的一个关键特性是它尝试保持模型的行为不会发生太大的改变，这有助于保证模型的稳定性。

通过这三个阶段的微调，我们可以使得语言模型的输出更符合我们的期望，例如更有创造性，更符合人类的偏好等。

## 代码拆解

### Rollout

在强化学习中，Rollout是指在给定的策略下模拟环境的过程。在PPO中，Rollout的过程对应于根据当前的语言模型（策略）生成文本（轨迹）。

这个过程依赖于在 prompt库中抽取的一个batch的数据Batch Prompt和当前的语言模型LM。

语言模型接收一个prompt作为输入，并生成一个Response。这些Response就构成了我们的"轨迹"。



输入：Batch Prompt，Actor LM

输出：Prompt+Response

```python
def _generate_sequence(self, prompts, mask, step):
  max_min_length = self.max_answer_seq_len + prompts.shape[1]
  # This has been added due to a probability/nan error that happens after
  # meta-llama/Llama-2-7b-hf enabled do_sample:
  # https://huggingface.co/meta-llama/Llama-2-7b-hf/commit/6fdf2e60f86ff2481f2241aaee459f85b5b0bbb9
  if self.actor_model.module.config.model_type == "llama":
    kwargs = dict(do_sample=False)
  else:
    kwargs = dict()
    with torch.no_grad():
      seq = self.actor_model.module.generate(
        prompts,
        attention_mask=mask,
        max_length=max_min_length,
        pad_token_id=self.tokenizer.pad_token_id,
        synced_gpus=self.z3_enabled,
        **kwargs)
      return seq
```

### Evaluate

Evaluate是在强化学习中对生成的轨迹（在我们的例子中就是文本）进行评估的步骤。在PPO中，这个评估过程由一个RM模型来完成，来为每一对Prompt+Response产生一个标量奖励值，这个值表示生成的轨迹的好坏，优化过程会试图最大化这个值。

**输入输出**

输入：Prompt+Response、RM

输出：Reward

```python
reward_score = self.reward_model.forward_value(
  seq, attention_mask,
  prompt_length=self.prompt_length)['chosen_end_scores'].detach(
)
```

### **Old Policy Sampling**

这个步骤是make experience的过程，计算并存储旧策略的概率、价值等值，来为后面更新的过程服务。

#### **Old Logprobs**

这个步骤中，我们从“旧的”策略，即在这个batch数据中初始的LM（initial actor）中计算每个token在旧的策略下的概率Old Logprobs。

这个步骤的重要性在于，我们在优化策略的时候，需要比较新旧策略下动作的概率，以此来更新我们的策略。因此，我们需要存储旧的策略的动作概率作为参考。

之所以要比较这个概率是为了算一个叫ratio的值，用这个值更新策略梯度，能限制更新率、

#### **Old Values**

Old Values的含义是旧策略中每个时间步（每个token的预测结果）的价值，这个值由critic网络进行预测，critic网络就是actor上加几个线性层能够给每个token预测一个值。需要这个值的原因是advantage的计算依赖于Old Values。

#### **Ref Logprobs**

Ref Logprobs的含义是最最原始的LM对于每个时间步的概率预测，计算这个值的目的是限制actor的更新，防止其偏离原始模型太远。



```python
def generate_experience(self, prompts, mask, step):
    self.eval()
    seq = self._generate_sequence(prompts, mask, step)
    self.train()
    pad_token_id = self.tokenizer.pad_token_id
    attention_mask = seq.not_equal(pad_token_id).long()
    with torch.no_grad():
        output = self.actor_model(seq, attention_mask=attention_mask)
        output_ref = self.ref_model(seq, attention_mask=attention_mask)
        reward_score = self.reward_model.forward_value(
            seq, attention_mask,
            prompt_length=self.prompt_length)['chosen_end_scores'].detach(
            )
        values = self.critic_model.forward_value(
            seq, attention_mask, return_value_only=True).detach()[:, :-1]

    logits = output.logits
    logits_ref = output_ref.logits

    return {
        'prompts': prompts,
        'logprobs': gather_log_probs(logits[:, :-1, :], seq[:, 1:]),
        'ref_logprobs': gather_log_probs(logits_ref[:, :-1, :], seq[:,
                                                                    1:]),
        'value': values,
        'rewards': reward_score,
        'input_ids': seq,
        "attention_mask": attention_mask
    }

def gather_log_probs(logits, labels):
	log_probs = F.log_softmax(logits, dim=-1)
  log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
 	return log_probs_labels.squeeze(-1)
```

### KL Penalty

在PPO 实现中，KL Penalty是在模型优化过程中添加的一个惩罚项，用于保证经过强化学习后的模型（新策略actor）不会过于偏离原始预训练模型（ref model）。

具体来说，首先使用微调过程中的模型（新策略actor）和预训练模型（ref model）来计算序列中每个词的对数概率。然后，我们计算两个模型输出之间的 Kullback-Leibler (KL) 散度，这是一种衡量两个概率分布差异的方法。该KL散度被用作一个额外的奖励信号，并作为优化过程中的惩罚项，用于确保微调后的模型生成的响应不会偏离太远于预训练模型。这样可以保证模型在微调的过程中不会丢失预训练模型学习到的有用的知识和模式。

在图中的KL Penalty步骤中，我们会在reward上增加这个kl惩罚项来实现这个过程。

**输入：**Ref Logprobs、Old Logprobs、Reward

**输出：**Token Reward

```python
def compute_rewards(self, prompts, log_probs, ref_log_probs, reward_score,
                    action_mask):
  # 计算 KL 散度，即模型的对数概率与参考模型的对数概率之间的差值
  kl_divergence_estimate = log_probs - ref_log_probs
  # 计算非得分奖励，即 KL 散度乘以 KL 控制器值的负值
  non_score_reward = -self.kl_ctl * kl_divergence_estimate

  # 复制非得分奖励为新的奖励
  rewards = kl_divergence_estimate
  # response 的起始位置和结束位置
  start = prompts.shape[1] - 1
  ends = start + action_mask[:, start:].sum(1) + 1
  # 对 reward 进行裁剪
  reward_clip = torch.clamp(reward_score, -self.clip_reward_value,
                            self.clip_reward_value)
  batch_size = log_probs.shape[0]
  for j in range(batch_size):
    # 对于最后一个非mask部分的token，其奖励是偏好模型的得分加上 KL 散度
    rewards[j, start:ends[j]][-1] += reward_clip[j]
  return rewards
```

### Generalized Advantage Estimation (GAE)

在强化学习中，我们会使用一个advantage的概念，来衡量每个时间步动作的价值，他的本质含义是当前采样到的动作（生成的文本）的价值比平均的数学期望价值高的部分。

而在PPO中，我们一般使用GAE来进行advantage的计算。GAE是一种多步优势估计方法。它通过引入一个权衡参数λ，在单步TD误差和多步TD误差之间进行权衡，从而减小估计的方差，提高学习的稳定性。

GAE的主要目标是希望找到一种策略，使得从当前状态开始，采取该策略能够获得的未来奖励最大，GAE使用了一种名为TD误差的概念，这是一种预测未来奖励的方法。然后，GAE将这些TD误差组合成一个加权和，权重由一个衰减因子λ决定。当λ=0时，GAE就退化为普通的优势函数估计；当λ=1时，GAE就变成了一种名为"蒙特卡洛"的方法。总的来说，GAE的本质就是把优势估计为后续时间步TD误差的加权和。

**输入：**Token Reward、Old Values

**输出：**Advantages、Returns

```python
def get_advantages_and_returns(self, values, rewards, start):
  lastgaelam = 0
  advantages_reversed = []
  length = rewards.size()[-1]
  # 从后往前遍历整个生成的序列
  for t in reversed(range(start, length)):
    # 计算下一个状态的价值，如果当前状态已经是最后一个状态，则下一个状态的价值为0
    nextvalues = values[:, t + 1] if t < length - 1 else 0.0
    # 计算 δ，它是奖励加上衰减后的下一个状态的价值，然后减去当前状态的价值
    delta = rewards[:, t] + self.gamma * nextvalues - values[:, t]
    # 使用 δ 更新 lastgaelam，这是 GAE 公式的一部分
    lastgaelam = delta + self.gamma * self.lam * lastgaelam
    # 将计算的优势值添加到优势值列表中
    advantages_reversed.append(lastgaelam)

  # 将优势值列表反向并转换为张量
  advantages = torch.stack(advantages_reversed[::-1], dim=1)
  # 计算回报值，它是优势值加上状态值
  returns = advantages + values[:, start:]
  return advantages.detach(), returns
```

### New Policy Sampling

New Policy Sampling是PPO算法中的一个关键步骤。在PPO中，策略优化的过程涉及到两个策略：一个是"旧的"策略，这是我们在开始每次优化迭代时使用的策略，另一个是"新的"策略，这是我们在优化过程中不断更新的策略。

New Policy Sampling就是在新的策略（更新后的actor）下对轨迹（文本）计算概率的过程。这个信息会被用于计算"Actor Loss"，也就是策略梯度的损失。在我们的步骤中，Old Logprobs是一次性一个batch的数据计算的，这是因为在一个batch中旧策略都是不变的；而New Logprobs是一个mini batch计算一次，这是因为新策略每个mini batch变一次。

此外这个步骤还会输出New Values和Logits分别用于critic loss和entropy loss的计算。

**输入输出**

**输入：**Ref_model、Actor、Critic

**输出：**New Logprobs、New Values、Logits

```python
### process the new outputs
batch = {'input_ids': seq, "attention_mask": attention_mask}
actor_prob = self.actor_model(**batch, use_cache=False).logits
actor_log_prob = gather_log_probs(actor_prob[:, :-1, :], seq[:, 1:])
value = self.critic_model.forward_value(**batch,
                                        return_value_only=True,
                                        use_cache=False)[:, :-1]
```

### Critic Loss


在Actor-Critic 强化学习算法框架中，Critic 模型的任务是估计状态的价值函数，也就是预测从当前状态开始，通过遵循某个策略，期望能得到的总回报。Critic的训练目标是最小化它的预测价值与实际回报之间的差距。

Critic Loss通常通过均方误差（Mean Squared Error, MSE）来计算。对于每一个状态，我们都有一个由Critic预测出的预期回报值V（s），以及一个真实的回报值G（returns）。Critic Loss就是这两个值之间差的平方。在一个批量的数据中，Critic Loss是所有状态的这个差的平方的平均值。公式如下： $𝐶𝑟𝑖𝑡𝑖𝑐 𝐿𝑜𝑠𝑠=𝐸[(𝑉(𝑠)−𝐺)^2]$

其中E[.]表示期望值，$ V(s) $ 是Critic对状态s（这个时间步的token）的价值预测New Values，G是真实的回报值Returns。

通过最小化Critic Loss，Critic的预测能力会逐渐提升。因为Critic的预测结果会被用来估计每个行动的优势（Advantage），这个优势值又会被用来计算策略的更新（Actor Loss）。

**输入：**New Values、Old_values、Returns

**输出：**梯度更新

```python
def critic_loss_fn(self, values, old_values, returns, mask):
    ## value loss
    # 将价值函数的预测值裁剪到一个范围内
    values_clipped = torch.clamp(
        values,
        old_values - self.cliprange_value,
        old_values + self.cliprange_value,
    )
    if self.compute_fp32_loss:
        values = values.float()
        values_clipped = values_clipped.float()
    # 计算裁剪前和裁剪后的价值函数损失
    vf_loss1 = (values - returns)**2
    vf_loss2 = (values_clipped - returns)**2
    # 最终的价值函数损失是裁剪前和裁剪后损失的最大值的平均值的一半
    vf_loss = 0.5 * torch.sum(
        torch.max(vf_loss1, vf_loss2) * mask) / mask.sum()
    return vf_loss
```

代码的作用是将 values 裁剪到一个范围内，这个范围是由 old_values - cliprange_value 和 old_values + cliprange_value 确定的，其中old_values 是初始的价值函数预测值，目的是为了避免 value 的变化太快。

### Actor Loss

在深度强化学习中，我们通常有两个主要的组成部分：Actor 和 Critic。Actor 是策略，它决定文本会被怎么样生成。Critic 则是我们的价值函数估计器，它预测我们从当前状态开始，如果遵循当前的策略，能够得到的未来回报。

Actor Loss 是我们用来优化 Actor 的损失函数。它的计算通常基于优势函数，优势函数表示在给定的状态下采取某个行动比遵循当前策略的期望回报要好多少。

在 PPO 中，我们使用一种称为 Importance Sampling 的技术来计算 Actor Loss。我们比较了在旧策略和新策略下行动的概率（Old Logprobs，New Logprobs），然后将这个比值（也就是 Importance Sampling 的权重）与优势函数Advantages相乘，得到了对 Actor Loss 的一个估计。

PPO的 Actor loss如下：
$$
L(s,a,\theta_k,\theta) = \min\left( \frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)} A^ {\pi_{\theta_k}}(s,a), \;\; \text{clip}\left(\frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a |s)}, 1 - \epsilon, 1+\epsilon \right) A^{\pi_{\theta_k}}(s,a) \right)
$$
$r_{\theta}=\frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)} $是新旧策略的比率。 $A^ {\pi_{\theta_k}}(s,a)$ 是优势函数，clip 是剪裁函数，它将 $\frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)} $限制在 $ [ 1 - \epsilon, 1+ \epsilon]$  范围内，$\epsilon$  是一个超参数，通常设置为 0.1 或 0.2。

这个损失函数的目标是最大化策略的期望回报，同时限制新旧策略之间的差异。当新旧策略的比率 $r_{\theta} $超出  $ [ 1 - \epsilon, 1+ \epsilon]$   范围时，剪裁函数会限制其影响，防止策略更新过大。



**输入：**Old Logprobs，New Logprobs、Advantages

**输出：**梯度更新

```python
def actor_loss_fn(self, logprobs, old_logprobs, advantages, mask):
    ## policy gradient loss
    # 计算新旧策略下概率的比值
    log_ratio = (logprobs - old_logprobs) * mask
    ratio = torch.exp(log_ratio)
    # 计算未截断的策略梯度损失
    pg_loss1 = -advantages * ratio
    # 计算截断的策略梯度损失
    pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.cliprange,
                                            1.0 + self.cliprange)
    # 选择两者中较大的作为最终的策略梯度损失
    pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / mask.sum()
    return pg_loss
```
