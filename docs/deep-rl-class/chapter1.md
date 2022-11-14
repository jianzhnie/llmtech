# 深度强化学习简介

欢迎来到人工智能中最引人入胜的话题：深度强化学习。 深度强化学习是一种机器学习方法，其中Agent通过和环境交互来学习如何执行动作。
深度强化学习是一种机器学习，其中代理 通过执行操作 和 查看结果来学习如何 在环境 中表现。

自 2013 年 [Deep Q-Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) 论文以来，已经看到了很多突破。从[击败世界上最优秀的 Dota2 玩家的OpenAI  5](https://www.twitch.tv/videos/293517383)[Dexterity 项目 ](https://openai.com/blog/learning-dexterity/),  我们处在深度强化学习研究的激动人心的时刻。

<div align=center>
<img width="600" src="https://huggingface.co/blog/assets/63_deep_rl_intro/OpenAIFive.jpg"/>
</div>
<div align=center>图 1.1  OpenAI 5 击败了世界上最优秀的 Dota2 人类玩家</div>

因为本文是深度强化学习课程的第一个单元，这是一门从入门到精通的免费课程，您将在其中学习使用著名的深度 RL 库（如 Stable Baselines3、RL Baselines3 Zoo 和 RLlib）的理论和实践。

在第一个单元中，您将学习深度强化学习的基础。 然后，您将训练您的第一个 Lander Agent 正确登陆月球🌕，并将其上传到 Hugging Face Hub，这是一个免费的开放平台，人们可以在其中共享 ML 模型、数据集和演示。

在深入训练深度强化学习Agent之前，必须掌握这些元素。 本章的目标是为您提供坚实的基础。

## 什么是强化学习？

要了解强化学习，让我们从大局出发。

强化学习背后的想法是，Agent（AI）将通过与环境交互（通过反复试验和试错）并接收奖励（负面或正面）作为执动作作的反馈来从环境中学习。

从与环境的互动中学习来自人类的自然经验。

例如，想象一下把你的弟弟放在一个他从未玩过的电子游戏前，手里拿着一个控制器，让他一个人呆着。

<div align=center>
<img width="600" src="https://huggingface.co/blog/assets/63_deep_rl_intro/Illustration_1.jpg"/>
</div>
<div align=center>图 1.2 </div>

你的弟弟将通过按下右键（动作）与环境（视频游戏）互动。他得到一枚硬币，这是+1的奖励。这是肯定的，他只是明白在这场比赛中他必须得到硬币。
<div align=center>
<img width="600" src="https://huggingface.co/blog/assets/63_deep_rl_intro/Illustration_2.jpg"/>
</div>
<div align=center>图 1.3 </div>

但随后，他再次按下右键（动作），碰到了敌人，他死了，得到 -1 的奖励。
<div align=center>
<img width="600" src="https://huggingface.co/blog/assets/63_deep_rl_intro/Illustration_3.jpg"/>
</div>
<div align=center>图 1. 4</div>

通过试错与环境互动，你的弟弟明白了他需要在这个环境中获得硬币，但要避开敌人。

没有任何监督，孩子会越来越擅长玩游戏。
这就是人类和动物通过互动学习的方式。强化学习只是一种从动作中学习的计算方法。

## 正式的定义

现在采取一个正式的定义：

> 强化学习是一个解决控制任务（也称为决策问题）的框架，它通过构建Agent来从环境中学习，通过反复试验与环境进行交互，并接收奖励（正面或负面）作为独特的反馈。

⇒ 但是强化学习是如何工作的？

## 强化学习框架

### 强化学习过程

<div align=center>
<img width="600" src="https://huggingface.co/blog/assets/63_deep_rl_intro/RL_process.jpg"/>
</div>
<div align=center>图 1. 5  RL 过程：状态、动作、奖励和下一个状态循环</div>

资料来源：[Reinforcement Learning: An Introduction, Richard Sutton and Andrew G. Barto](http://incompleteideas.net/book/RLbook2020.pdf)

为了理解 RL 过程，让想象一个Agent学习玩平台游戏：
<div align=center>
<img width="600" src="https://huggingface.co/blog/assets/63_deep_rl_intro/RL_process_game.jpg"/>
</div>
<div align=center>图 1. 6 </div>

- Agent 接收来自环境的状态 $S_0$ —— 收到游戏的第一帧（环境）。
- 基于状态$S_0$， Agent采取动作$A_0$ — — Agent将向右移动。
- 环境进入新状态$S_1$——  游戏进入下一帧。
- 环境给予给Agent一些回报$R_1$——Agent没有死 （正的奖励+1）。

这个 RL 循环地输出一系列 状态、动作、奖励和下一个状态。

<div align=center>
<img width="600" src="https://huggingface.co/blog/assets/63_deep_rl_intro/sars.jpg"/>
</div>
<div align=center>图 1. 7 </div>

Agent的目标是最大化其累积奖励， 称为预期回报。

### 奖励假设：强化学习的中心思想

⇒ 为什么Agent的目标是期望收益最大化？

因为 RL 是基于奖励假设的，即所有目标都可以描述为最大化期望回报（期望累积奖励）。

这就是为什么在强化学习中， 为了获得最佳动作， 需要最大化预期的累积奖励。

### 马尔可夫性质

在论文中，您会看到 RL 过程被称为马尔可夫决策过程 (MDP)。

我们将在以下单元中再次讨论马尔可夫性质。但是如果你今天需要记住一些关于它的东西，马尔可夫属性意味着Agent 只需要当前状态来决定采取什么动作，而不是历史上他们所采取的所有状态和动作。

### 观察/状态空间

观察/状态是 Agent从环境中获得的信息。 在视频游戏的情况下，它可以是一个帧（屏幕截图。在交易Agent的情况下，它可以是某个股票的价值等。

观察 和 状态之间有区别 ：

- State s：是对世界状态的完整描述（没有隐藏信息）。在完全观察的环境中。

<div align=center>
<img width="600" src="https://huggingface.co/blog/assets/63_deep_rl_intro/chess.jpg"/>
</div>
<div align=center>图 1. 8  在国际象棋游戏中，我们从环境中接收到一个状态，因为可以访问整个棋盘信息。</div>


对于国际象棋游戏，处于一个完全可观察的环境中，因为可以访问整个棋盘信息。

- 观察 o：是状态的部分描述。在部分观察的环境中。

<div align=center>
<img width="600" src="https://huggingface.co/blog/assets/63_deep_rl_intro/mario.jpg"/>
</div>
<div align=center>图 1. 9  在《超级玛丽》中，只能看到靠近玩家的关卡的一部分，因此会收到一个观察</div>

在《超级玛丽》中，处于部分观察的环境中。收到了一个观察结果 ，因为只看到了关卡的一部分。

> 实际上，在本课程中使用术语状态，但将在实现中进行区分。

回顾一下：

<div align=center>
<img width="600" src="https://huggingface.co/blog/assets/63_deep_rl_intro/obs_space_recap.jpg"/>
</div>
<div align=center>图 1. 10 </div>

### 动作空间

动作空间是环境中所有可能动作的集合。

动作可以来自离散或连续空间：

- 离散空间：可能动作的数量是有限的。


<div align=center>
<img width="600" src="https://huggingface.co/blog/assets/63_deep_rl_intro/mario.jpg"/>
</div>
<div align=center>图 1. 11 同样，在《超级玛丽》中，只有 4 个方向并且可以跳跃</div>

在《超级玛丽》中，有一组有限的动作，因为只有 4 个方向和跳跃。

- 连续空间： 可能采取动作的数量是无限的。

<div align=center>
<img width="600" src="https://huggingface.co/blog/assets/63_deep_rl_intro/self_driving_car.jpg"/>
</div>
<div align=center>图 1.12 自动驾驶汽车Agent有无数可能的动作，因为它可以左转 20°、21,1°、21,2°、鸣喇叭、右转 20°…… </div>

回顾一下：
<div align=center>
<img width="600" src="https://huggingface.co/blog/assets/63_deep_rl_intro/action_space.jpg"/>
</div>
<div align=center>图 1.13 </div>

考虑到这些信息至关重要，因为它在未来选择 RL 算法时很重要。

### 奖励和折扣

奖励是强化学习的基础，因为它是 Agent的唯一反馈 。多亏了它，Agent才知道所采取的动作是好是坏。

每个时间步 t 的累积奖励可以写成：
<div align=center>
<img width="600" src="https://huggingface.co/blog/assets/63_deep_rl_intro/rewards_1.jpg"/>
</div>
<div align=center>图 1.14  累积奖励等于序列所有奖励的总和。</div>

这相当于：
<div align=center>
<img width="600" src="https://huggingface.co/blog/assets/63_deep_rl_intro/rewards_2.jpg"/>
</div>
<div align=center>图 1.15 累积奖励 = rt+1 (rt+k+1 = rt+0+1 = rt+1)+ rt+2 (rt+k+1 = rt+1+1 = rt+2) + ...</div>

然而，实际上， 不能就这样添加它们。 较早到来的奖励（在游戏开始时） 更有可能发生，因为它们比长期的未来奖励更容易预测。

假设您的Agent是一只小老鼠，每步可以移动一个瓷砖，而您的对手是猫（它也可以移动）。你的目标是在被猫吃掉之前吃掉最大量的奶酪。
<div align=center>
<img width="600" src="https://huggingface.co/blog/assets/63_deep_rl_intro/rewards_3.jpg"/>
</div>
<div align=center>图 1. 16 </div>

正如在图中看到的那样， 吃附近的奶酪比吃靠近猫的奶酪的可能性更大 （离猫越近，它就越危险）。

因此， 靠近猫的奖励，即使它更大（更多奶酪），也会打折， 因为不确定是否能吃掉它。

为了给奖励打折扣，这样进行：

1. 定义了一个称为 gamma 的折扣率。它必须介于 0 和 1 之间。大多数时候介于0.99 和 0.95之间。

- gamma越大，折扣越小。这意味着Agent更关心长期回报。
- 另一方面，gamma 越小，折扣越大。这意味着Agent更关心短期奖励（最近的奶酪）。

2. 然后，每个奖励将通过 gamma 折扣到时间步长的指数。随着时间步长的增加，猫离越来越近， 因此未来的奖励发生的可能性越来越小。

折扣累积预期奖励是：
<div align=center>
<img width="600" src="https://huggingface.co/blog/assets/63_deep_rl_intro/rewards_4.jpg"/>
</div>
<div align=center>图 1. 17 </div>

### 任务类型

任务是强化学习问题的一个实例 。可以有两种类型的任务：偶发的和持续的。

#### Episodic 任务

在这种情况下，有一个起点和一个终点 （终端状态）。将创建一个Episode：状态、动作、奖励和新状态的列表。

例如，拿《超级玛丽》游戏举例：Episode从新马里奥关卡的发布开始，到 你被杀或通关时结束。
<div align=center>
<img width="600" src="https://huggingface.co/blog/assets/63_deep_rl_intro/mario.jpg"/>
</div>
<div align=center>图 1. 18 新一集的开始</div>
![马里奥]()

#### Continuing 任务

这些是永远持续的任务（没有终止状态）。在这种情况下，Agent必须学习如何选择最佳动作并同时与环境交互。

例如，进行自动股票交易的Agent。对于这个任务，没有起点和终点。 Agent会一直运行，直到决定停止它们。
<div align=center>
<img width="600" src="https://huggingface.co/blog/assets/63_deep_rl_intro/stock.jpg"/>
</div>
<div align=center>图 1. 19 </div>

<div align=center>
<img width="600" src="https://huggingface.co/blog/assets/63_deep_rl_intro/tasks.jpg"/>
</div>
<div align=center>图 1. 20 </div>

## 探索/开发 平衡

最后，在研究解决强化学习问题的不同方法之前，必须讨论一个非常重要的主题：探索/开发权衡。

- 探索是通过尝试随机动作来探索环境，以找到有关环境的更多信息。
- 开发是利用已知信息来最大化奖励。

记住， RL Agent的目标是最大化预期的累积奖励。但是， 可能会落入一个常见的陷阱。

举个例子：
<div align=center>
<img width="600" src="https://huggingface.co/blog/assets/63_deep_rl_intro/exp_1.jpg"/>
</div>
<div align=center>图 1. 21 </div>

在这个游戏中，老鼠可以拥有无限数量的小奶酪 （每个 +1）。但是在迷宫的顶部，有大量的奶酪（+1000）。

但是，如果只专注于**Exploitation**，Agent永远不会达到巨大的奶酪数量。相反，它只会利用最近的奖励来源， 即使这个来源很小（利用）。

但是如果Agent做一点探索，它可以发现大的奖励 （大奶酪堆）。

这就是所说的探索/开发权衡。需要平衡探索环境的程度和 对环境的了解程度。

因此，必须定义一个有助于处理这种权衡的规则。将在以后的章节中看到处理它的不同方法。

如果仍然感到困惑，请考虑一个实际生活中常见的问题：餐厅的选择：
<div align=center>
<img width="600" src="https://huggingface.co/blog/assets/63_deep_rl_intro/exp_2.jpg"/>
</div>
<div align=center>图 1. 22  资料来源：[伯克利人工智能课程](http://rail.eecs.berkeley.edu/deeprlcourse-fa17/f17docs/lecture_13_exploration.pdf)</div>

- *Exploitation*：您每天都去同一个您认为不错的餐厅，风险错过另一家更好的餐厅。
- 探索：尝试您以前从未去过的餐厅，有可能会遇到糟糕的体验，但很可能有机会获得美妙的体验。

回顾一下：
<div align=center>
<img width="600" src="https://huggingface.co/blog/assets/63_deep_rl_intro/expexpltradeoff.jpg"/>
</div>
<div align=center>图 1. 23 </div>

## 解决 RL 问题的两种主要方法

⇒ 既然学习了 RL 框架，那么如何解决 RL 问题呢？

换句话说，如何构建一个可以选择最大化其预期累积奖励的动作的 RL Agent？

### 策略 π：Agent的大脑

Policy π 是 Agent 的大脑，它是告诉在给定状态下要采取什么动作。 所以它 定义了Agent 在给定时间的行为。
<div align=center>
<img width="600" src="https://huggingface.co/blog/assets/63_deep_rl_intro/policy_1.jpg"/>
</div>
<div align=center>图 1. 24 将策略视为Agent的大脑，该功能将告诉在给定状态下要采取的动作</div>

这个 Policy 是想要学习的函数，目标是找到最优策略 π，即当智能体按照它采取动作时，将获得最大化期望回报。我们通过训练找到这个策略 π 。

有两种方法可以训练Agent来找到这个最优策略 π：

- 直接地，通过教Agent学习在给定状态下要采取的动作：基于策略的方法。
- 间接地，教Agent了解哪个状态更有价值，然后采取导致更有价值状态的动作：基于价值的方法。

### 基于策略的方法

在基于策略的方法中， 直接学习策略函数。

该函数将从每个状态映射到该状态下的最佳对应动作。 或者在该状态下的一组可能动作的概率分布。
<div align=center>
<img width="600" src="https://huggingface.co/blog/assets/63_deep_rl_intro/policy_2.jpg"/>
</div>
<div align=center>图 1. 25 正如在此处看到的，策略（确定性) 直接指示每一步要采取的动作。</div>

有两种策略：

- 确定性：给定状态的策略将始终返回相同的操作。
<div align=center>
<img width="600" src="https://huggingface.co/blog/assets/63_deep_rl_intro/policy_3.jpg"/>
</div>
<div align=center>图 1.26  *动作=Policy（状态)*</div>

<div align=center>
<img width="600" src="https://huggingface.co/blog/assets/63_deep_rl_intro/policy_4.jpg"/>
</div>
<div align=center>图 1. 27 </div>

- 随机：输出动作的概率分布。
<div align=center>
<img width="600" src="https://huggingface.co/blog/assets/63_deep_rl_intro/policy_5.jpg"/>
</div>
<div align=center>图 1. 28  policy(actions | state) = 给定当前状态的一组动作的概率分布</div>

<div align=center>
<img width="600" src="https://huggingface.co/blog/assets/63_deep_rl_intro/mario.jpg"/>
</div>
<div align=center>图 1. 29 给定一个初始状态，随机策略将输出该状态下可能动作的概率分布。</div>

回顾一下：
<div align=center>
<img width="600" src="https://huggingface.co/blog/assets/63_deep_rl_intro/pbm_1.jpg"/>
</div>
<div align=center>图 1. 30 </div>

<div align=center>
<img width="600" src="https://huggingface.co/blog/assets/63_deep_rl_intro/pbm_2.jpg"/>
</div>
<div align=center>图 1. 31 </div>

### 基于价值的方法

在基于价值的方法中，不是训练一个策略函数，而是训练一个将状态映射到处于该状态的期望值的价值函数。

某个状态的价值是 Agent从该状态开始，然后按照Policy采取动作时可以获得 的预期折扣回报。

“按照策略采取动作”只是意味着策略是 “走向价值最高的状态”。
<div align=center>
<img width="600" src="https://huggingface.co/blog/assets/63_deep_rl_intro/value_1.jpg"/>
</div>
<div align=center>图 1. 32</div>

在这里，看到价值函数 为每个可能的状态定义了价值。
<div align=center>
<img width="600" src="https://huggingface.co/blog/assets/63_deep_rl_intro/value_2.jpg"/>
</div>
<div align=center>图 1. 33 根据价值函数，策略在每一步都会选择价值函数定义的最大值的状态：-7，然后是-6，然后是-5（依此类推)来实现目标。 </div>


回顾一下：
<div align=center>
<img width="600" src="https://huggingface.co/blog/assets/63_deep_rl_intro/vbm_1.jpg"/>
</div>
<div align=center>图 1. 34 </div>

<div align=center>
<img width="600" src="https://huggingface.co/blog/assets/63_deep_rl_intro/vbm_2.jpg"/>
</div>
<div align=center>图 1. 35 </div>

## 强化学习的“深度”

⇒ 到目前为止，讨论的是强化学习。但是“深”在哪里发挥作用？

深度强化学习引入了深度神经网络来解决强化学习问题 ——因此得名“深度”。

例如，在下一篇文章中，将研究 Q-Learning（经典强化学习）和 Deep Q-Learning，它们都是基于值的 RL 算法。

您会看到不同之处在于，在第一种方法中， 使用传统算法 创建 Q 表，帮助找到对每个状态采取的动作。

在第二种方法中， 将使用神经网络 （近似 q 值）。
<div align=center>
<img width="600" src="https://huggingface.co/blog/assets/63_deep_rl_intro/deep.jpg"/>
</div>
<div align=center>图 1. 36  Schema 灵感来自 Udacity 的 Q 学习笔记</div>

如果您不熟悉深度学习，您应该观看[fastai Practical Deep Learning for Coders（free）](https://course.fast.ai/)

总结一下，有很多信息：

- 强化学习是一种从动作中学习的计算方法。构建了一个从环境中学习的Agent，它通过反复试验与环境进行交互，并接收奖励（负面或正面）作为反馈。
- 任何 RL 智能体的目标都是最大化其预期累积奖励（也称为预期回报），因为 RL 是基于奖励假设，即所有目标都可以描述为预期累积奖励的最大化。
- RL 过程是一个循环，输出一系列状态、动作、奖励和下一个状态的循环。
- 为了计算预期累积奖励（预期回报），将奖励打折：较早到来的奖励（在游戏开始时）更有可能发生，因为它们比长期未来奖励更可预测。
- 要解决 RL 问题，您需要找到一个最优策略，该策略是您的 AI 的“大脑”，它将告诉在给定状态下要采取什么动作。最佳策略是为您提供最大化预期回报的动作的策略。
- 有两种方法可以找到最佳策略：
  1. 通过直接训练策略函数：基于策略的方法。
  2. 通过训练一个预测期望回报的价值函数，Agent将在每个状态下获得并使用这个函数来定义策略：基于价值的方法。
- 最后，谈论深度强化学习，因为引入了深度神经网络来估计要采取的动作（基于策略）或估计状态的价值（基于价值），因此得名“深度”。

在下一章中，[我们将学习 Q-Learning 并深入**研究基于价值的方法。**](https://huggingface.co/blog/deep-rl-q-part1)