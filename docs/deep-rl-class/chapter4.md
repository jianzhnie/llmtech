# Policy Gradient

[在上一单元](https://huggingface.co/blog/deep-rl-dqn)中，我们学习了 Deep Q learning。在这个基于价值的深度强化学习算法中，我们 **使用了一个深度神经网络来为一个状态下每个可能的动作近似不同的 Q 值。**

事实上，截止目前，我们只研究了基于价值的方法， **我们将价值函数估计为寻找最优策略的中间步骤。**

![链接价值Policy](https://huggingface.co/blog/assets/70_deep_rl_q_part1/link-value-policy.jpg)

因为，在基于价值的情况下， **$π$ 仅因为动作价值估计而存在，因为策略只是一个函数** （例如，贪婪策略），它会选择给定状态下具有最高价值的动作。

但是，对于基于策略的方法，我们希望直接优化策略 **而无需学习价值函数的中间步骤。**

所以今天， **我们将研究第一个基于策略的方法**：Reinforce。我们将使用 PyTorch 从头开始实现它。在使用 CartPole-v1、PixelCopter 和 Pong 测试其稳健性。

![环境](https://huggingface.co/blog/assets/85_policy_gradient/envs.gif)

## 什么是策略梯度方法？

Policy-Gradient 是 Policy-Based Methods 的一个子类，这是一类算法， **旨在直接优化策略，而不使用价值函数** 与基于策略的方法的不同之处在于，策略梯度方法是一系列旨在 **通过使用梯度上升估计最优策略的权重来直接优化策略的算法。**

### 策略梯度概述

为什么我们直接通过使用策略梯度方法中的梯度上升估计最优策略的权重来优化策略？

请记住，强化学习旨在 **找到最佳行为策略（策略）以最大化其预期累积奖励。**

我们还需要记住，策略是**给定状态、输出、动作分布**的函数（在我们的例子中使用随机策略）。

![随机策略](https://huggingface.co/blog/assets/63_deep_rl_intro/pbm_2.jpg)

我们使用 Policy-Gradients 的目标是通过调整策略来控制动作的概率分布，以便**在未来更频繁地对好的动作（最大化回报）进行采样。**

举一个简单的例子：

- 我们通过Policy与环境互动来收集经验。
- 然后我们查看Eposide的奖励总和（预期回报）。如果这个和是正的，我们 **认为在这些经验中采取的行动是好的：**因此，我们想要增加每个状态-行动对的 $P(a|s)$（在该状态下采取该行动的概率）。

策略梯度算法（简化）如下所示：

![策略梯度大图](https://huggingface.co/blog/assets/85_policy_gradient/pg_bigpicture.jpg)

但是 Deep Q-Learning 已经非常棒了！为什么还要使用策略梯度方法？

### 策略梯度方法的优点

与Deep Q learning方法相比有很多优势：

1. 集成的简单性： **我们可以直接估计策略而无需存储额外的数据（动作值）。**
2. 策略梯度方法可以 **学习随机策略，而价值函数不能**。

这有两个后果：

a. 我们**不需要手动实现探索/开发权衡**。由于我们输出动作的概率分布，因此智能体探索 **状态空间而不总是采用相同的轨迹。**

b. 我们也摆脱了**感知混叠**的问题。感知混叠是指两个状态看起来（或确实）相同但需要不同的动作。

举个例子：我们有一个智能吸尘器，它的目标是吸走灰尘，避免杀死仓鼠。

![仓鼠 1](https://huggingface.co/blog/assets/85_policy_gradient/hamster1.jpg)

我们的真空吸尘器只能感知墙壁的位置。

问题在于，两个红色案例是混叠状态，因为智能体感知到每个案例的上下墙。

![仓鼠 1](https://huggingface.co/blog/assets/85_policy_gradient/hamster2.jpg)

在确定性策略下，策略将在红色状态时向右移动或向左移动。**这两种情况都会导致我们的智能体卡住，永远不会吸尘**。

在基于价值的 RL 算法下，我们学习了准确定性策略（“贪婪 epsilon 策略”）。因此，我们的智能体可能要花很多时间才能找到灰尘。

另一方面，最优随机策略会在灰色状态下随机向左或向右移动。因此，**它不会卡住，并且很有可能到达目标状态**。

![仓鼠 1](https://huggingface.co/blog/assets/85_policy_gradient/hamster3.jpg)

1. 策略梯度**在高维动作空间和连续动作空间更有效**

事实上，Deep Q learning 的问题在于，在给定当前状态的情况下，他们的**预测会在每个时间步为每个可能的动作分配一个分数（最大预期未来奖励） 。**

但是，如果我们有无限可能的行动呢？

例如，对于自动驾驶汽车，在每个状态下，您都可以（几乎）无限地选择动作（将方向盘转动 15°、17.2°、19.4°、鸣喇叭等）。我们需要为每个可能的动作输出一个 Q 值！而采取连续输出的最大动作本身就是一个优化问题！

相反，使用策略梯度，我们输出 **动作的概率分布。**

### 策略梯度方法的缺点

当然，Policy Gradient 方法也有缺点：

- **策略梯度在很多时候收敛于局部最大值而不是全局最优值。**
- 策略梯度越来越快， **一步一步：训练可能需要更长的时间（效率低下）。**
- 策略梯度可以具有高方差（解决方案基线）。

👉 如果您想更深入地了解策略梯度方法的优点和缺点，[可以观看此视频](https://youtu.be/y3oqOjHilio)。

既然我们已经看到了 Policy-Gradient 的大局及其优缺点，**那么来研究和实现其中之一**：Reinforce。

## Reinforce（蒙特卡洛策略梯度）

Reinforce，也称为 Monte-Carlo Policy Gradient，**使用整个 episode 的估计回报来更新策略参数**  $θ$.

我们有我们的策略 $π$，它有一个参数 θ。给定一个状态，这个 $π$ **输出动作的概率分布**。

![Policy](https://huggingface.co/blog/assets/85_policy_gradient/policy.jpg)

其中，$\pi_\theta(a_t|s_t)$,   $π_θ$ 是根据我们的策略，智能体从状态 st 选择动作的概率。

**但是我们怎么知道我们的Policy是否好呢？** 我们需要有一种方法来衡量它。要知道我们定义了一个得分/目标函数，称为 $J(θ)$.

得分函数 J 是预期回报：

![返回](https://huggingface.co/blog/assets/85_policy_gradient/objective.jpg)



请记住，策略梯度可以看作是一个优化问题。所以我们必须找到最佳参数 (θ) 来最大化得分函数 J(θ)。

为此，我们将使用[策略梯度定理](https://www.youtube.com/watch?v=AKbX1Zvo7r8)。我不打算深入探讨数学细节，但如果您有兴趣，请[观看此视频](https://www.youtube.com/watch?v=AKbX1Zvo7r8)

Reinforce 算法的工作原理如下： 循环：

- 使用 Policy $\pi_\theta$的 收集一集 $\tau$
- 使用 episode 来估计梯度 $\hat{g} = \nabla_\theta J(\theta)*G$

![策略梯度](https://huggingface.co/blog/assets/85_policy_gradient/pg.jpg)

- 更新策略的权重：$\theta \leftarrow \theta + \alpha \hat{g}$
我们可以做出的解释是：

- \nabla_\theta log \pi_\theta(a_t|s_t)∇*θ*的*log* *$π$* *_* *_**θ*的（*一个**吨*的∀ *s**吨*的)是从状态 st 选择动作**的（对数）概率最陡增**的方向。=> 这告诉用户如果我们想增加/减少在状态 st 选择动作的对数概率，**我们应该如何改变策略的权重。**

- R(\tau)*R* ( *τ* )

  : 是评分函数：

  - 如果回报很高，它会推高（状态，动作）组合的概率。
  - 否则，如果回报很低，它会降低（状态，动作）组合的概率。

现在我们研究了 Reinforce 背后的理论，**您已准备好使用 PyTorch 编写您的 Reinforce 智能体**。您将使用 CartPole-v1、PixelCopter 和 Pong 测试其稳健性。
