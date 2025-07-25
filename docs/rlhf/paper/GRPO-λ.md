# GRPO-λ (动态长度惩罚)

## 前言

大型语言模型，例如DeepSeek-R1，通过思维链（Chain of Thought, CoT）技术实现了复杂推理路径的生成。然而，传统强化学习方法如GRPO在追求答案正确性的同时，导致了模型出现“过思考”的现象——即生成过于冗长的推理步骤，这不仅降低了效率还增加了错误率。为了解决这一问题，现有方案引入了长度惩罚机制，但这又引发了新的挑战：训练早期模型准确率骤降。

本文介绍了一种名为GRPO-λ的新方法，该方法旨在智能地调节奖励策略以促进模型的发展。GRPO-λ如同给模型配备了一个“智能教练”，根据模型当前的推理能力动态调整训练目标，初期注重提高正确率，待能力提升后再专注于简洁表达。这种策略在五个数学与科学推理基准测试中表现出色，实现了47.3%的长度压缩和1.48%的精度提升，并将有效训练时长延长至2.5倍。



## 问题分析：为何长度惩罚会引发训练崩溃？

**过度思考的原因**在于传统的GRPO等方法采用的是“0/1奖励”机制，只关注最终答案是否正确，忽视了推理过程的质量。为了增加获得奖励的机会，模型倾向于生成更长的推理链条，即便这些额外的步骤并不总是必要或相关的。

另一方面，尽管长度惩罚机制试图解决这个问题，但其实施方式可能导致模型在尚未具备足够推理能力的情况下被迫缩短输出，从而丢失重要的逻辑信息，造成准确性下降。

## 方法创新：GRPO-λ的动态奖励机制

GRPO-λ的核心在于分阶段的能力训练策略，类似于儿童学习走路再到跑步的过程。对于推理能力较弱的样本，重点放在提高正确率上；而对于已经具备一定能力的样本，则开始优化其表达的简洁性。这种方法通过一个称为Top-λ筛选器的机制来实现，它能够根据每个问题的答案质量进行分组，并对表现最好的前λ%给予鼓励简洁性的奖励。

## 具体实现

### 批量级Top-λ选择

对于每一批查询，我们评估每个查询完成组的正确性，并计算其正确率。GRPO-λ选择批处理中正确率最高的前λ比例的查询完成组进行效率优先优化。具体来说，这些组根据其在批处理中的正确率进行排序。例如，选择前20%的组进行效率优先优化（如图2（上部）所示），因为这些组已经表现出足够的推理能力，可以专注于长度减少。批处理中剩余的组则被分配给准确率优先优化，以确保模型能够继续提高其推理能力。

### 动态奖励策略调整

基于批量级Top-λ选择，GRPO-λ应用两种不同的奖励策略：

- **效率优先优化（带长度惩罚）**：对于批处理中正确率最高的前λ比例的查询完成组（即那些具有较高正确率的组），应用长度惩罚奖励以鼓励更短的推理序列：
  $$
  r_k^i = \begin{cases}
  1 - \alpha \cdot \sigma\left(\frac{L_k^i - \text{mean}(L_k)_{\text{correct}}}{\text{std}(L_k)_{\text{correct}}}\right) & \text{如果 } O_k^i \text{ 正确} \\
  0 & \text{如果 } O_k^i \text{ 错误}
  \end{cases}
  $$
  其中，α 是长度惩罚系数，$\text{mean}(L_k)_{\text{correct}}$ 和 $\text{std}(L_k)_{\text{correct}}$ 分别是正确答案的补全长度的均值和标准差。错误的补全（$r_k^i = 0$）不会获得奖励。这种策略优先考虑那些已经表现出足够准确性的组的推理效率。

- **准确率优先优化（0/1结果奖励）**：对于批处理中剩余的组（不在前λ子集中），奖励默认为标准的GRPO 0/1结果奖励：

  $$
  r_k^i = \begin{cases}
  1 & \text{如果 } O_k^i \text{ 正确} \\
  0 & \text{如果 } O_k^i \text{ 错误}
  \end{cases}
  $$
  这种策略确保模型专注于提高正确率较低的补全的推理准确性。

这种奖励策略防止了直接对所有组使用长度惩罚时可能出现的效率与准确率之间的不平衡强调[17, 32]。这确保了准确率和效率优先级之间的受控过渡，有效避免了准确率急剧下降的风险。

### 优势计算与参数更新

在获得衰减奖励后，类似于GRPO，GRPO-λ基于组奖励计算每个样本的优势。具体来说，计算组内奖励的均值和标准差（std），然后使用公式 $\hat{A}_i = \frac{r_i - \text{mean}(r_i)}{\text{std}(r_i)}$ 计算每个样本的优势。随后，计算出的每个样本的优势广播到所有对应的响应标记。最后，基于每个样本的优势值进行参数更新。

## 实验对比

- **Vanilla**：基础模型，无任何强化学习优化。
- **GRPO**：采用传统强化学习策略进行优化。
- **LP（Length Penalty）**：应用长度惩罚机制，但未崩溃时的表现。
- **LP***：同等步数下由于强制压缩导致崩溃的情况。

## 实验结果

**GRPO-λ**实现了显著的改进，达到了47.3%的推理步骤长度压缩率，并同时提高了1.48%的准确率。相比之下，使用LP方法仅能达到55.28%的长度压缩率和0.04%的精度提升，且在训练早期就出现了精度大幅下降的问题。

## 结论

GRPO-λ提供了一种新颖且有效的解决方案，解决了传统强化学习方法中的关键挑战。通过动态调整训练目标，它确保了模型能够在不牺牲准确性的情况下变得更加高效，展现了在实际应用中的巨大潜力。



## GRPO-λ 代码实现

```python
from typing import List
from collections import defaultdict
import math
import numpy as np
from transformers import PreTrainedTokenizer
from openrlhf.trainer.ppo_utils.math_reward_funcs import MathAccuracyReward


class GRPOLambdaReward:
    """
    Implements GRPO-λ dynamic reward assignment based on group-wise accuracy ranking.

    Args:
        efficiency_reward (BaseRewardFunction): Reward function for efficiency-prioritized groups (with length penalty).
        accuracy_reward (BaseRewardFunction): Reward function for accuracy-prioritized groups (0/1 reward).
        tokenizer (PreTrainedTokenizer): Tokenizer for measuring response length.
        acc_top_lambda (float): Proportion of top-accurate groups to assign efficiency reward.
        length_penalty_alpha (float): Length penalty strength.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        math_acc_reward: MathAccuracyReward,
        acc_top_lambda: float = 0.2,
        length_penalty_alpha: float = 0.6,  # 长度惩罚强度
    ) -> None:
        self.tokenizer = tokenizer
        self.math_acc_reward = math_acc_reward
        self.acc_top_lambda = acc_top_lambda
        self.length_penalty_alpha = length_penalty_alpha

    def __call__(self, completions: List[str], solutions: List[str], groups: List[int], **kwargs) -> List[float]:
        """
        Computes the rewards for each completion based on group-wise accuracy and length penalties.

        Args:
            completions (List[str]): Generated completions.
            solutions (List[str]): Ground truth answers.
            groups (List[int]): Group ID for each completion.

        Returns:
            List[float]: Reward for each completion.
        """
        assert len(completions) == len(solutions) == len(groups), "Input lengths must match"
        # Step 1: Compute token lengths for each completion

        lengths = [len(ids) for ids in self.tokenizer(completions, add_special_tokens=False)["input_ids"]]

        # Step 2: Compute accuracy for each sample
        acc_rewards = self.math_acc_reward(completions, solutions, **kwargs)
        is_correct = [r >= 1.0 for r in acc_rewards]

        # Step 3: Group-wise accuracy and length statistics
        group_correct_count = defaultdict(int)
        group_total_count = defaultdict(int)
        group_lengths = defaultdict(list)

        for g, correct, length in zip(groups, is_correct, lengths):
            group_total_count[g] += 1
            if correct:
                group_correct_count[g] += 1
                group_lengths[g].append(length)

        # Step 4: Compute accuracy per group
        group_accuracy = {g: group_correct_count[g] / group_total_count[g] for g in group_total_count}

        # Step 5: Sort groups by accuracy, select top λ%
        sorted_groups = sorted(group_accuracy.items(), key=lambda x: x[1], reverse=True)
        num_top_groups = max(1, math.ceil(len(sorted_groups) * self.acc_top_lambda))
        top_group_ids = {g for g, _ in sorted_groups[:num_top_groups]}

        # Step 6: Calculate mean and std of lengths for correct answers in each group
        group_length_stats = {}
        for g in group_lengths:
            if group_lengths[g]:
                mean_length = np.mean(group_lengths[g])
                std_length = (
                    np.std(group_lengths[g]) if len(group_lengths[g]) > 1 else 0.0001
                )  # Avoid division by zero
                group_length_stats[g] = (mean_length, std_length)

        # Step 7: Apply corresponding reward function
        rewards = []
        for i, (length, group) in enumerate(zip(lengths, groups)):
            if group in top_group_ids:
                # Efficiency-prioritized group: use efficiency reward
                mean_length, std_length = group_length_stats[group]
                sigma_value = (length - mean_length) / std_length
                reward = 1 - self.length_penalty_alpha * self.sigmoid(sigma_value)
            else:
                # Accuracy-prioritized group: use accuracy reward
                reward = acc_rewards[i]
            rewards.append(reward)

        return rewards

    @staticmethod
    def sigmoid(x: float) -> float:
        """Sigmoid function."""
        return 1 / (1 + math.exp(-x))
```
