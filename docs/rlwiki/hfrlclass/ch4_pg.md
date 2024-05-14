# Policy Gradient

[在上一单元](https://huggingface.co/blog/deep-rl-dqn)中，我们学习了 Deep Q learning。在这个基于价值的深度强化学习算法中，我们 **使用深度神经网络来为一个状态下每个可能的动作近似不同的 Q 值。**

事实上，截止目前，我们只研究了基于价值的方法， **我们将价值函数估计作为寻找最优策略的中间步骤。**

![链接价值Policy](https://huggingface.co/blog/assets/70_deep_rl_q_part1/link-value-policy.jpg)

因为，在基于价值的方法中，策略 **$π$ 仅因为动作价值估计而存在，因为策略只是一个函数** （例如，贪婪策略），它会选择给定状态下具有最高价值的动作。

但是，在基于策略的方法中，我们希望直接优化策略**而无需学习价值函数的中间步骤。**我们将学习基于策略的方法，并研究这些方法的一个子集，称为策略梯度**。然后我们将使用 PyTorch 从头开始实施我们的第一个策略梯度算法，称为 Monte Carlo **Reinforce 。然后，我们将使用 CartPole-v1 和 PixelCopter 环境测试其稳健性。然后，您将能够针对更高级的环境迭代和改进此实现。

![环境](https://huggingface.co/blog/assets/85_policy_gradient/envs.gif)

## 基于策略的方法有哪些？

强化学习的主要目标是**找到最优策略 $π*$ 使期望累积奖励最大化**。因为强化学习基于*奖励假设*：所有目标都可以描述为期望累积奖励的最大化。

## 基于价值的、基于策略的和演员评论家方法

我们在第一单元学习过，有两种方法来找到（大部分时间是近似的）这个最优策略 $π*$.

- 在基于价值的方法中，我们学习价值函数。

  - 想法是通过最优价值函数导致最优策略 $π*$.
  - 目标是**最小化预测值和目标值之间的损失**以逼近真实的动作价值函数。
  - 我们有一个策略，但它是隐含的，因为它**是直接从价值函数生成的**。例如，在 Q-Learning 中，我们定义了一个 epsilon-greedy 策略。

- 另一方面，在基于策略的方法中，我们直接学习近似 $π*$ 而无需学习价值函数。

  - 这个想法是**参数化策略**。例如，使用神经网络$π_θ$，该策略将输出动作的概率分布（随机策略）。

  <img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit6/stochastic_policy.png" alt="随机策略" style="zoom:33%;" />

  - 我们的目标是**使用梯度上升最大化参数化策略**。
  - 为此，我们控制参数 $θ$，这将影响在一个 State 下动作的分配。

![基于策略](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit6/policy_based.png)

- 最后，我们将研究*actor-critic*，它是基于价值和基于策略的方法的结合。

因此，由于基于策略的方法，我们可以直接优化我们的策略$π_θ$的输出动作的概率分布 $π_θ(a|s)$ 这会带来最佳的累积回报。为此，我们定义了一个目标函数 $J(θ)$，即期望的累积奖励，我们**要找到θ最大化这个目标函数**。

## 基于策略和策略梯度方法的区别

本单元中学习的策略梯度方法是基于策略方法的一个子类。这两种方法的区别**在于我们如何优化参数** $θ$:

- 在基于策略的方法中，我们直接搜索最优策略。我们可以通过使用爬山、模拟退火或进化策略等技术最大化目标函数的局部近似来间接实现优化参数$θ$.
- 在策略梯度方法中，因为它是基于策略的方法的子类，我们直接搜索最优策略。但是我们**直接**通过对目标函数 $J(θ)$ 进行梯度上升优化参数*θ* .

## 策略梯度方法的优缺点

在深入研究策略梯度方法的工作原理（目标函数、策略梯度定理、梯度上升等）之前，让我们研究一下基于策略的方法的优缺点。

### 策略梯度方法的优点

与Deep Q learning方法相比有很多优势：

#### 1. 集成的简单性

 可以直接估计策略而无需存储额外的数据（动作值）

#### 2. 策略梯度方法可以学习随机策略

策略梯度方法可以 **学习随机策略，而价值函数则不能**。

这有两个好处：

1. 我们**不需要手动实现探索/开发平衡**。由于我们输出动作的概率分布，因此智能体通过采样探索**状态空间而不总是采用相同的轨迹。**

2. 我们摆脱了**感知混叠**的问题。感知混叠是指两个状态看起来（或确实）相同但需要采取不同的动作。

#### 3. 策略梯度算法**在高维动作空间和连续动作空间更有效**.

Deep Q learning 的问题在于，在给定当前状态的情况下，DQN会在每个时间步为每个可能的动作分配一个分数（最大未来期望奖励） 。但是，如果我们有无限可能的动作呢？

例如，对于自动驾驶汽车，在每个状态下，都可以（几乎）无限地选择动作（将方向盘转动 15°、17.2°、19.4°、鸣喇叭等）。我们需要为每个可能的动作输出一个 Q 值！而采取连续输出的最大动作本身就是一个优化问题！

相反，使用策略梯度，我们输出的是**动作的概率分布，而无需担心这一问题。**

#### 4. 策略梯度方法具有更好的收敛性

在基于价值的方法中，我们使用激进的运算方法来**改变价值函数：我们在 Q 估计上取最大值**。因此，对于估计动作值的任意小变化，如果该变化导致具有最大值的不同动作，动作概率可能会发生显着变化。

例如，如果在训练期间，最好的动作是左边的（Q 值为 0.22），而在它之后的训练步骤是右边的（因为右边的 Q 值变为 0.23），我们极大地改变了策略，因为现在策略将大部分时间都是靠右而不是靠左。

另一方面，在策略梯度方法中，随机策略行动偏好（采取行动的概率）**随时间平稳变化**。

### 策略梯度方法的缺点

当然，Policy Gradient 方法也有缺点：

- **策略梯度在很多时候收敛于局部最大值而不是全局最优值。**
- 策略梯度算法的训练需要更长的时间才能收敛（效率低下）。
- 策略梯度可能具有高方差（解决方案： 基线方法）。

## 策略梯度方法

Policy-Gradient 是 Policy-Based Methods 的一个子类，**旨在直接优化策略，而不使用价值函数**。 与基于策略的方法的不同之处在于，策略梯度方法是一系列旨在 **通过使用梯度上升估计最优策略的权重来直接优化策略的算法。**

### 策略梯度概述

强化学习旨在 **找到最佳行为策略（策略）以最大化其期望累积奖励。**

策略是**给定状态、输出、动作分布**的函数（在我们的例子中使用随机策略）。

![随机策略](https://huggingface.co/blog/assets/63_deep_rl_intro/pbm_2.jpg)

我们以 CartPole-v1 为例：

- 输入一个状态$S_t$, 输出该状态下动作的概率分布。

![基于策略](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit6/policy_based.png)

我们使用策略梯度的目标是通过调整策略来**控制动作的概率分布，以便在未来更频繁地对好的动作（最大化回报）进行采样。** 每次智能体与环境交互时，我们都会调整参数，以便将来更有可能对好的动作进行采样。

但是**我们如何使用期望回报来优化权重**呢？

我们的想法是**让智能体在一局中进行交互**。如果我们赢得了这一局，我们认为所采取的每项行动都是好的，并且在未来必须进行更多的采样，因为它们会导致获胜。所以对于每个状态-动作对，我们想增加 $P( a ｜s )$：在那个状态下采取那个动作的概率。如果我们输了，则减少这个值。

策略梯度算法（简化）如下所示：

![策略梯度大图](https://huggingface.co/blog/assets/85_policy_gradient/pg_bigpicture.jpg)

现在我们了解了全局，让我们更深入地研究策略梯度方法。

### 深入研究策略梯度方法

我们的策略函数$π$, 参数为 θ。给定一个状态，这个 $π$ **输出该状态下采取动作的概率分布**。

![Policy](https://huggingface.co/blog/assets/85_policy_gradient/policy.jpg)

其中，$\pi_\theta(a_t|s_t)$ 是根据我们的策略，智能体从状态 $s_t$ 选择动作的概率。

**我们怎么知道我们的Policy是否好呢？** 需要有一种方法来衡量它。我们定义一个得分/目标函数，称为 $J(θ)$.

### 目标函数

目标函数为我们提供了给定轨迹**的智能体的性能**（不考虑奖励的状态动作序列（与情节相反）），并输出期望的累积奖励。

![返回](https://huggingface.co/blog/assets/85_policy_gradient/objective.jpg)

让我们详细地说明这个公式：

- *期望回报*（也称为期望累积奖励）是所有$R( τ )$ 可以采取的可能值的加权平均（其中权重由 $P(τ ;θ)$ 给出）。


![返回](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit6/expected_reward.png)

- $R(τ)$：任意轨迹的回报。要获取这个值并用它来计算期望回报，我们需要将它乘以每个可能轨迹的概率。
- $P(τ ;θ)$ ：每个可能轨迹τ的概率（这个概率取决于*θ*因为它定义了它用来选择轨迹动作的策略，作为访问状态的影响）。

![可能性](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit6/probability.png)

- $J(θ)$：期望回报，我们通过对所有轨迹求和来计算，在给定*θ*情况下采用给定的轨迹的概率，以及这条轨迹的回报。

![最大物镜](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit6/max_objective.png)

### 梯度上升和策略梯度定理

策略梯度是一个优化问题：所以我们必须找到最佳参数 (θ) 来最大化我们的目标函数$J(θ)$，我们需要使用**梯度上升**，(与梯度下降相反，给出了$J(θ)$最陡峭增加的方向).

我们对梯度上升的更新步骤是：

$θ←θ+α*∇_θ*J(θ)$

我们可以重复应用这个更新公式，希望 *θ* 收敛到最大化$J(θ)$的值.

然而，要获得$J(θ)$的导数， 有两个主要问题:

1. 我们无法计算目标函数的真实梯度，因为这意味着计算每个可能轨迹的概率，这在计算上非常昂贵。我们想**用基于样本的估计（收集一些轨迹）来计算梯度估计**。
2. 我们还有另一个问题，我将在下一个可选部分中详细说明。为了区分这个目标函数，我们需要区分状态分布，称为马尔可夫决策过程动力学。这是依附于环境的。在给定当前状态和智能体采取的操作的情况下，它为我们提供了环境进入下一个状态的概率。问题是我们无法区分它，因为我们可能不知道它。

![Probability](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit6/probability.png)

幸运的是，我们将使用称为策略梯度定理的解决方案，它将帮助我们将目标函数重新表述为一个不涉及状态分布微分的可微分函数。

![策略梯度](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit6/policy_gradient_theorem.png)

### Reinforce 算法（Monte Carlo Reinforce）

Reinforce，也称为 Monte-Carlo Policy Gradient，是一种策略梯度算法， **使用整个 episode 的估计回报来更新策略参数**  $θ$. 我们的策略函数$π$，它有一个参数 θ。给定一个状态，这个 $π$ **输出该状态下采取动作的概率分布**。

Reinforce 算法的工作原理如下：

在一个循环中：

- 使用策略 $π_θ$ 收集智能体的轨迹 $τ $

- 使用 episode 的数据来估计梯度 $\hat{g} = \nabla_\theta J(\theta)$

  ![策略梯度](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit6/policy_gradient_one.png)

- 更新策略的权重：$\theta \leftarrow \theta + \alpha \hat{g}$

我们可以做出的解释是：

- $\nabla_\theta logπ_θ(a_t∣s_t)$ 是从状态 $s_{t}$选择动作**的（对数）概率最陡增**的方向。这告诉我们如果我们想增加/减少在在状态$s_{t}$选择动作$a_{t}$的对数概率，我们应该如何改变策略的权重.
- $R (\tau)$: 是评分函数：
  - 如果回报很高，它会**推高**（状态，动作）组合的概率。
  - 否则，如果回报很低，它会**降低**（状态，动作）组合的概率。

我们还可以**收集多个片段（轨迹）**来估计梯度：

![策略梯度](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit6/policy_gradient_multiple.png)

举一个简单的例子：

- 我们通过Policy与环境互动来收集经验。
- 然后我们查看Eposide的奖励总和（期望回报）。如果这个总和是正的，我们 **认为在这些经验中采取的行动是好的：**因此，我们想要增加每个``状态-行动对``的 $P(a|s)$（在该状态下采取该行动的概率）。如果这个总和是负的，我们则减少每个`状态-行动`对的 $P(a|s)$

## 补充阅读

### Introduction to Policy Optimization

- [Part 3: Intro to Policy Optimization - Spinning Up documentation](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)

### Policy Gradient

- https://johnwlambert.github.io/policy-gradients/
- [RL - Policy Gradient Explained](https://jonathan-hui.medium.com/rl-policy-gradients-explained-9b13b688b146)
- [Chapter 13, Policy Gradient Methods; Reinforcement Learning, an introduction by Richard Sutton and Andrew G. Barto](http://incompleteideas.net/book/RLbook2020.pdf)

### Implementation

- [PyTorch Reinforce implementation](https://github.com/pytorch/examples/blob/main/reinforcement_learning/reinforce.py)
- [Implementations from DDPG to PPO](https://github.com/MrSyee/pg-is-all-you-need)
