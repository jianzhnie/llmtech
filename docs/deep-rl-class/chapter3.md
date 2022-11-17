# Deep Q Learning

[在上一单元](https://huggingface.co/blog/deep-rl-q-part2)中，我们学习了第一个强化学习算法：Q-Learning， **从头开始实现它**，并在 FrozenLake-v1 ☃️ 和 Taxi-v3 🚕 两个环境中对其进行训练。

我们用这个简单的算法得到了很好的结果。但是这些环境相对简单，因为**状态空间是离散的且很小**（FrozenLake-v1 有 14 个不同的状态，Taxi-v3 有 500 个）。

但正如我们将看到的，在非常大的状态空间环境中生成和更新 **Q Table可能变得不可能。**

所以今天，**我们将研究我们的第一个Deep强化学习智能体**：Deep Q-Learning。Deep Q-Learning不使用 Q Table，而是使用神经网络，该神经网络输入状态并根据该状态为每个动作近似 Q 值。

我们将使用 RL-Zoo 训练它玩 Space Invaders 和其他 Atari 环境，RL[-Zoo](https://github.com/DLR-RM/rl-baselines3-zoo)是一种基于 Stable-Baselines 的 RL 训练框架，它提供用于训练、评估智能体、调整超参数、绘制结果和录制视频的脚本。

![环境](https://huggingface.co/blog/assets/78_deep_rl_dqn/atari-envs.gif)

## 从Q-Learning到Deep Q-Learning

我们知道**Q-Learning 是一种用于训练 Q-Function 的算法，Q-Function**是**动作价值函数** ，可确定处于特定状态并在该状态下采取特定动作的价值。

![Q函数](https://huggingface.co/blog/assets/73_deep_rl_q_part2/Q-function.jpg)*给定状态和动作， Q 函数输出状态-动作值（也称为 Q 值)*

**Q 来自该状态下该动作的“质量”**。

在内部，我们的 Q 函数有 **一个 Q Table，该表中每个单元格对应一个状态-动作对值。** 将此 Q Table视为 **Q 函数的内存或备忘表。**

问题是Q-Learning是一种表格方法。Aka，一个状态空间和动作空间小到足以将值函数表示为数组和表的问题。这是不可扩展的。

Q-Learning 在小型状态空间环境中运行良好，例如：

- FrozenLake，有 14 个状态。
- Taxi-v3，有 500 个状态。

但是想一想我们今天要做的事情：我们将训练一个智能体来学习使用图像帧作为输入来玩太空入侵者游戏。

Atari 环境有一个形状为 (210, 160, 3) 的观察空间，每个像素包含从 0 到 255 的值，因此我们得到 256^(210x160x3) = 256^100800个状态空间（为了比较，宇宙中的原子大约有 10^80 个）。

![Atari 状态空间](https://huggingface.co/blog/assets/78_deep_rl_dqn/atari.jpg)

因此，状态空间是巨大的；为该环境创建和更新 Q Table 几乎是不可实现的。在这种情况下，最好的办法是使用参数化 Q 函数 $Q_{\theta}(s,a)$来近似 Q 值而不是 Q Table .

在给定状态的情况下，神经网络将近似估计该状态下每个可能动作的不同 Q 值。而这正是Deep Q-Learning所做的。

![Deep Q-Learning](https://huggingface.co/blog/assets/63_deep_rl_intro/deep.jpg)

现在我们学习了Deep Q -learning，下面让我们更深入地了解Deep Q Network。

## Deep Q Network (DQN)

这是我们的Deep Q-Learning网络架构：

![DeepQ网络](https://huggingface.co/blog/assets/78_deep_rl_dqn/deep-q-network.jpg)

作为输入，我们将通过网络**的 4 帧堆栈作为状态State，并为该状态下的每个可能动作输出 Q 值向量**。然后，就像 Q-Learning 一样，我们只需要使用我们的 epsilon-greedy 策略来选择要采取的行动。

初始化神经网络时，**Q 值的估计很差**。但在训练过程中，我们的Deep Q Network智能体会将情况与适当的行动联系起来，并**学会玩好游戏**。

### 预处理输入和时间限制

我们提到了对输入进行预处理。这是一个重要的步骤，因为我们希望降低状态的复杂性，以减少训练所需的计算时间。

因此，我们所做的是将状态空间减少到84x84并对其进行灰度处理（因为Atari环境中的颜色不会增加重要信息）。这是一个重要的节省，因为我们将三个颜色通道（RGB）减少到1。

如果屏幕上不包含重要信息，我们也可以在某些游戏中裁剪屏幕的一部分。然后我们将四个帧叠加在一起。

![预处理](https://huggingface.co/blog/assets/78_deep_rl_dqn/preprocessing.jpg)

为什么我们要将四个帧叠加在一起？我们将帧堆叠在一起，因为它可以帮助我们**处理时间限制的问题**。让我们以 Pong 游戏为例。当您看到此帧时：

![时间限制](https://huggingface.co/blog/assets/78_deep_rl_dqn/temporal-limitation.jpg)

你能告诉我，球的去向吗？不会，因为一帧不足以产生运动感！但是如果我再添加三个帧呢？**在这里您可以看到球向右移动**。

![时间限制](https://huggingface.co/blog/assets/78_deep_rl_dqn/temporal-limitation-2.jpg)这就是为什么, 为了捕获时间信息，我们将四个帧堆叠在一起。

然后，堆叠的帧由三个卷积层处理。这些层**使我们能够捕获和利用图像中的空间关系**。而且，由于帧堆叠在一起，因此可以利用这些帧的一些空间属性。

最后，我们有几个全连接层，它们为该状态下的每个可能动作输出Q值。

![DeepQ网络](https://huggingface.co/blog/assets/78_deep_rl_dqn/deep-q-network.jpg)

我们看到Deep Q-Learning使用神经网络来近似给定状态下每个可能动作的不同Q-值。接下来我们深入研究Deep Q-Learning算法。

## Deep Q-Learning算法

我们了解到，Deep Q-Learning使用Deep Network来近似状态下每个可能动作的不同Q-值（值函数估计）。
不同之处在于，在训练阶段，不像我们使用Q-Learning那样直接更新状态-动作对的Q-值：

![Q损失](https://huggingface.co/blog/assets/73_deep_rl_q_part2/q-ex-5.jpg)

在Deep Q-Learning中，我们**在 Q 值预测和 Q 目标之间创建了一个损失函数，并使用梯度下降来更新Deep Q Network的权重以更好地逼近我们的 Q 值**。

![Q-目标](https://huggingface.co/blog/assets/78_deep_rl_dqn/Q-target.jpg)

Deep Q-Learning 训练算法有*两个阶段*：

- **采样**：我们执行动作并将**观察到的经验元组存储在 replay memory 中**。
- **训练**：随机选择**小批量的 Tuple 并使用梯度下降进行更新**。

![抽样训练](https://huggingface.co/blog/assets/78_deep_rl_dqn/sampling-training.jpg)

但是，与 Q-Learning 相比，这并不是唯一的变化。Deep Q-Learning训练**可能会遇到不稳定**的问题，这主要是因为结合了非线性 Q 值函数（神经网络）和自举（当我们使用现有估计而不是实际的完整回报来更新目标时）。

为了帮助我们稳定训练，我们实施了三种不同的解决方案：

1. *Experience Replay*，更**有效地利用经验**。
2. 固定 Q-Target **以稳定训练**。
3. *Double Deep Q-Learning*，**处理Q值高估的问题**。

### 经验回放 以更有效地利用数据

我们为什么要创建经验回放？

Deep Q-Learning 中的 Experience Replay 有两个作用：

1. **更有效地利用训练期间的经验**。

- 经验回放有助于我们 **更有效地利用训练期间的经验。** 通常，在在线强化学习中，我们在环境中进行交互，获得经验（状态、动作、奖励和下一个状态），从中学习（更新神经网络）并丢弃它们。
- 但是通过经验回放，我们创建了一个回放缓冲区来保存**我们可以在训练期间重复使用的经验样本。**

![经验回放](https://huggingface.co/blog/assets/78_deep_rl_dqn/experience-replay.jpg)

⇒ 这使我们能够**多次从个人经验中学习**。

2. **避免忘记以前的经验，减少经验之间的相关性**。

- 如果我们向神经网络提供连续的经验样本，我们会遇到的问题是，当它覆盖新的经验时，它往往会忘记以前的经验。

解决方案是创建一个 Replay Buffer，在与环境交互时存储经验元组，然后对一小批元组进行采样。这可以防止 **网络只了解它立即做了什么。**

经验回放还有其他好处。通过对经验进行随机抽样，我们消除了观察序列中的相关性，并避免了 **动作值发生灾难性的振荡或发散。**

在 Deep Q-Learning 伪代码中，我们看到我们从容量 N （N 是您可以定义的超参数）**初始化一个 Replay Buffer  D。**然后，我们将经验存储在内存中，并对一小批经验进行采样，以在训练阶段为Deep Q Network提供数据。

![体验重放伪代码](https://huggingface.co/blog/assets/78_deep_rl_dqn/experience-replay-pseudocode.jpg)

### 固定 Q-Target 以稳定训练

当我们想要计算 TD 误差（也称损失）时，我们需要计算**TD 目标（Q 目标）和当前 Q 值（Q 的估计）之间的差异**。

但我们**对真正的 TD 目标一无所知**。我们需要做出估计。使用 Bellman 方程，我们看到 TD 目标只是在该状态下采取该动作的奖励加上下一个状态的折扣的最高 Q 值。

![Q-目标](https://huggingface.co/blog/assets/78_deep_rl_dqn/Q-target.jpg)

然而，问题在于我们使用相同的参数（权重）来估计 TD 目标 **和** Q 值。因此，TD 目标与我们正在更新的参数之间存在显着相关性。

因此，这意味着在训练的每一步， **我们的 Q 值都会发生变化，但目标值也会发生变化。** 所以，我们越来越接近我们的目标，但目标也在移动。这就像追逐一个移动的目标！导致了训练中的大幅振荡。

这就好比你是一个牛仔（Q 估计）并且你想抓住牛（Q 目标），你必须更靠近（减少错误）。

![Q-目标](https://huggingface.co/blog/assets/78_deep_rl_dqn/qtarget-1.jpg)

在每个时间步，您都试图接近奶牛，奶牛也在每个时间步移动（因为使用相同的参数）。



![Q-目标](https://huggingface.co/blog/assets/78_deep_rl_dqn/qtarget-2.jpg)

![Q-目标](https://huggingface.co/blog/assets/78_deep_rl_dqn/qtarget-3.jpg)

这导致了奇怪的追逐路径（训练中的显著振荡）。

![Q-目标](https://huggingface.co/blog/assets/78_deep_rl_dqn/qtarget-4.jpg)

相反，我们在伪代码中看到的是：

- 使用**固定参数的单独网络**来估计 TD 目标
- **间隔每个 C 步， 从Deep Q Network复制参数**以更新目标网络。

![固定 Q-target 伪代码](https://huggingface.co/blog/assets/78_deep_rl_dqn/fixed-q-target-pseudocode.jpg)

### 双DQN

Double DQN，Double Q-learning， [由 Hado van Hasselt](https://papers.nips.cc/paper/3964-double-q-learning)引入。该方法 **解决了 Q 值高估的问题。**

要理解这个问题，请记住我们如何计算 TD Target：

通过计算 TD 目标， 我们面临一个简单的问题：如何确定 **下一个状态的最佳动作是具有最高 Q 值的动作？**

我们知道， Q 值的准确性取决于我们尝试了什么动作 **以及** 我们探索了哪些相邻状态。

因此，在训练开始时，我们没有足够的关于最佳行动的信息 。因此，将最大 Q 值（有噪声）作为最佳动作可能会导致误报。如果非最优动作经常 **被赋予比最优最佳动作更高的 Q 值，学习将变得复杂。**

解决方案是：当我们计算 Q 目标时，我们使用两个网络将动作选择与目标 Q 值生成解耦。我们：

- 使用**DQN 网络**为下一个状态选择最佳动作（具有最高 Q 值的动作）。
- 使用**目标网络**来计算在下一个状态下采取该动作的目标 Q 值。

因此，Double DQN 有助我们减少对 q 值的高估，从而帮助我们训练得更快，更稳定。

自 Deep Q-Learning 的这三项改进以来，又增加了很多新的改进，例如 Prioritized Experience Replay、Dueling Deep Q-Learning。它们超出了本课程的范围，但如果您有兴趣，请查看我们放在阅读列表中的链接。👉 **https://github.com/huggingface/deep-rl-class/blob/main/unit3/README.md**

https://huggingface.co/docs)
