# 基于open-thoughts 强化微调

 ## Base Model

model：open-thoughts/OpenThinker3-7B

link: https://huggingface.co/open-thoughts/OpenThinker3-7B

## AIME 24/25 评测：

| 模型            | AIME 2024 n_samples=8 | AIME 2024( Reported) | AIME 2025 n_samples=8 | AIME 2025( Reported) |
| --------------- | --------------------: | -------------------: | --------------------: | -------------------- |
| OpenThinker3-7B |                0.7041 |             **69.0** |                0.5916 | **53.3**             |

## RL Data

huggingface hub: nvidia/AceReason-Math

Num  instances:  50k

link: https://huggingface.co/datasets/nvidia/AceReason-Math

处理流程：

1. 过滤包含网页和中文的样本 ~40k
2. 部署模型，每个样本采样8次，剔除全对样本 ~10k




<img src='ascend/images/accuracy_count_barplot.png' title='Cosine-Scaled-Reward-Function' data-no-caption>


<img src='ascend/images/avg_generate_length_histogram_filtered.png' data-no-caption>


<img src='ascend/images/avg_generate_length_histogram.png' data-no-caption>



## 基于 Relu 的 Reward 函数

### CosineScaledReward

1. 正确的 CoT比错误的 CoT 获得更高的奖励。
2. 较短的正确 CoT 比较长的正确 CoT 获得更高的奖励，这激励模型有效地使用推理计算。
3. 较短的错误 CoT 应比较长的错误 CoT 受到更高的惩罚。

这种方式可用于稳定和控制CoT长度，同时提高准确性

```python

class CosineScaledReward(BaseRewardFunction):
    """Reward function that scales rewards based on completion length using a
    cosine schedule.

    **Reference**: https://arxiv.org/abs/2502.03373

    **Key Behavior**:
        - ✅ Shorter **correct** completions receive **higher** rewards.
        - ❌ Longer **incorrect** completions receive **lower** penalties.

    **Args:**
        - `cosine_min_value_wrong` (float): Minimum reward for incorrect answers.
        - `cosine_max_value_wrong` (float): Maximum reward for incorrect answers.
        - `cosine_min_value_correct` (float): Minimum reward for correct answers.
        - `cosine_max_value_correct` (float): Maximum reward for correct answers.
        - `cosine_max_len` (int): Maximum length for scaling.
        - `cosine_accuracy_orm` (BaseRewardFunction, optional): Accuracy computation module.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer = None,
        cosine_min_value_wrong: float = -0.5,
        cosine_max_value_wrong: float = 0.0,
        cosine_min_value_correct: float = 0.5,
        cosine_max_value_correct: float = 1.0,
        cosine_max_len: int = 1000,
        accuracy_orm: Union[BaseRewardFunction, None] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.min_value_wrong = cosine_min_value_wrong
        self.max_value_wrong = cosine_max_value_wrong
        self.min_value_correct = cosine_min_value_correct
        self.max_value_correct = cosine_max_value_correct
        self.max_len = cosine_max_len
        self.accuracy_orm = accuracy_orm or MathAccuracyReward()

    @staticmethod
    def cosine_scaled_reward(t: int, T: int, min_value: float, max_value: float) -> float:
        """Computes a cosine-scaled reward value based on response length.

        Args:
            t (int): Current length of the completion.
            T (int): Maximum length for scaling.
            min_value (float): Minimum reward value.
            max_value (float): Maximum reward value.

        Returns:
            float: Scaled reward value.
        """
        cosine_value = math.cos(t * math.pi / T)
        return min_value + 0.5 * (max_value - min_value) * (1.0 + cosine_value)

    def __call__(self, completions: List[str], solution: List[str], **kwargs) -> List[float]:
        """Computes cosine-scaled rewards for a list of model completions.

        Args:
            completions (List[str]): List of generated completions.
            solution (List[str]): List of ground truth solutions.
            **kwargs: Additional arguments for the accuracy function.

        Returns:
            List[float]: List of computed rewards.
        """
        acc_rewards = self.accuracy_orm(completions, solution, **kwargs)
        rewards = []

        for content, acc_reward in zip(completions, acc_rewards):
            gen_len = len(self.tokenizer.encode(content, add_special_tokens=False))

            if gen_len == 0:
                logger.warning(f"Skipping empty completion: {content}")
                rewards.append(self.min_value_wrong)
                # Assign minimum penalty for empty responses
                continue

            is_correct = acc_reward >= 1.0

            # Correct answers get higher rewards for being concise
            if is_correct:
                min_value, max_value = self.min_value_correct, self.max_value_correct
            else:
                min_value, max_value = self.max_value_wrong, self.min_value_wrong  # Fixed logic

            # Compute scaled reward
            reward = self.cosine_scaled_reward(gen_len, self.max_len, min_value, max_value)
            rewards.append(reward)

        return rewards

```



### 绘制 Reward 曲线

```python
import numpy as np
import matplotlib.pyplot as plt
import math
from typing import List


def cosine_scaled_reward(t: float, T: int, min_value: float, max_value: float) -> float:
    """
    Calculate the scaled reward based on a cosine function.

    Args:
        t (float): Current length.
        T (int): Maximum length used for scaling.
        min_value (float): Minimum reward value.
        max_value (float): Maximum reward value.

    Returns:
        float: Scaled reward value.
    """
    cosine_value = math.cos(t * math.pi / T)
    return min_value + 0.5 * (max_value - min_value) * (1.0 + cosine_value)


def plot_rewards(
    lengths: List[float],
    rewards_correct: List[float],
    rewards_wrong: List[float],
    title: str,
    min_value_wrong: float,
    max_value_wrong: float,
    min_value_correct: float,
    max_value_correct: float,
    cosine_max_len: int,
):
    """
    Plot the reward curves for correct and wrong answers.

    Args:
        lengths (List[float]): Length data points.
        rewards_correct (List[float]): Rewards for correct answers.
        rewards_wrong (List[float]): Rewards for wrong answers.
        title (str): Title of the plot.
        min_value_wrong (float): Minimum reward value for wrong answers.
        max_value_wrong (float): Maximum reward value for wrong answers.
        min_value_correct (float): Minimum reward value for correct answers.
        max_value_correct (float): Maximum reward value for correct answers.
        cosine_max_len (int): Maximum length.
    """
    plt.figure(figsize=(12, 8))
    plt.plot(lengths, rewards_correct, "b-", linewidth=2, label="Correct Answers", alpha=0.8)
    plt.plot(lengths, rewards_wrong, "r-", linewidth=2, label="Wrong Answers", alpha=0.8)
    plt.grid(True, alpha=0.3)
    plt.xlabel("Text Length", fontsize=12)
    plt.ylabel("Reward Value", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.axhline(y=0, color="black", linestyle="--", alpha=0.5, linewidth=1)
    plt.axvline(
        x=cosine_max_len, color="gray", linestyle=":", alpha=0.5, linewidth=1, label=f"Max Length: {cosine_max_len}"
    )
    plt.xlim(0, cosine_max_len)
    plt.ylim(min_value_wrong - 0.1, max_value_correct + 0.1)
    plt.tight_layout()
    plt.savefig("Cosine-Scaled-Reward-Function.png")
    plt.show()


if __name__ == "__main__":
    # Set parameters
    cosine_min_value_wrong = -2.0
    cosine_max_value_wrong = -0.0
    cosine_min_value_correct = 0.0
    cosine_max_value_correct = 2.0
    cosine_max_len = 10240

    # Generate length data
    lengths = np.linspace(0, cosine_max_len, 1000)

    # Calculate rewards for correct and wrong answers
    rewards_correct = [
        cosine_scaled_reward(t, cosine_max_len, cosine_min_value_correct, cosine_max_value_correct) for t in lengths
    ]
    rewards_wrong = [
        cosine_scaled_reward(t, cosine_max_len, cosine_max_value_wrong, cosine_min_value_wrong) for t in lengths
    ]

    # Plot the reward curves
    plot_rewards(
        lengths,
        rewards_correct,
        rewards_wrong,
        "Cosine-Scaled Reward Function",
        cosine_min_value_wrong,
        cosine_max_value_wrong,
        cosine_min_value_correct,
        cosine_max_value_correct,
        cosine_max_len,
    )
```

<img src='ascend/images/Cosine-Scaled-Reward-Function.png' title='Cosine-Scaled-Reward-Function' data-no-caption>
