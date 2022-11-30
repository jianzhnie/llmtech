# Advantage Actor Critic（ A2C）

[在第 4单元](https://huggingface.co/blog/deep-rl-pg)中，我们学习了第一个基于策略的算法，**Reinforce**。在基于策略的方法中，**我们的目标是直接优化策略而不使用价值函数**。

我们看到 Reinforce 效果还不错。然而，因为我们使用蒙特卡洛抽样来估计回报（用一整个 Eposide 来计算回报），**在策略梯度估计上有显著的方差**。

策略梯度估计是**回报增长最快的方向**。也就是，如何更新我们的策略权重，使得导致良好回报的动作更有可能被采取。蒙特卡洛采样带来的高方差会**导致训练速度变慢，因为我们需要大量样本来减轻**，我们将在本单元中进一步研究。

今天我们将研究**Actor-Critic 方法**，这是一种混合的架构，结合了基于价值和基于策略的方法，通过减少方差来帮助稳定训练：

- 演员 Actor：控制**智能体行为方式**（基于策略的方法）
- 评论家 Critic：衡量**所采取动作的好坏 **（基于价值的方法）

我们将研究其中一种称为 Advantage Actor Critic (A2C) 的混合方法，并动手从零开始实现这一算法。

## Reinforce算法的方差问题

在 Reinforce 中，我们希望**增加轨迹中动作的概率与回报的高低成正比**。

![加强](https://huggingface.co/blog/assets/85_policy_gradient/pg.jpg)

- 如果**回报高**，我们**将推高**（状态，动作）组合的概率。
- 否则，如果**回报很低**，则**降低**（状态，动作）组合的概率。

回报 $R(\tau)$是使用蒙特卡洛采样计算的。我们收集智能体和环境交互的轨迹并计算折扣回报，**并使用回报分数来增加或减少在该轨迹中采取的每个动作的概率**。如果回报高，所有动作都将通过增加采取动作的可能性来“加强”。$R(\tau) = R_{t+1} + \gamma R_{t+2} + \gamma^2R_{t+3} + ... $

这种方法的优点是**这种方法的估计是无偏的。因为我们没有估计回报**，只是使用获得的真实回报来估计。

但是由于环境的随机性和策略的随机性，**轨迹可能导致不同的回报**， 因此带来的方差很高。因此，相同的起始状态可能导致截然不同的回报。正因为如此，**从同一状态开始获得的回报在不同的Eposide中可能会有很大差异**。

![方差](https://huggingface.co/blog/assets/89_deep_rl_a2c/variance.jpg)

**解决方案是通过使用大量轨迹**来减轻方差 ，希望在任何一个轨迹中引入的方差都将减少，并提供一个“真实”的回报估计。

然而，增加batch_size会显着**降低采样效率**。所以我们需要找到额外的机制来减少方差。

------

如果你想更深入地研究深度强化学习中的方差和偏差权衡问题，你可以查看这两篇文章：

- [理解（深度）强化学习中的偏差/方差权衡](https://blog.mlreview.com/making-sense-of-the-bias-variance-trade-off-in-deep-reinforcement-learning-79cf1e83d565)
- [强化学习中的偏差方差权衡](https://www.endtoend.ai/blog/bias-variance-tradeoff-in-reinforcement-learning/)

------

## Advantage Actor Critic(A2C)

### 使用 Actor-Critic 方法减少方差

减少 Reinforce 算法的方差并更快更好地训练我们的智能体的解决方案是结合使用基于策略和基于价值的方法：*Actor-Critic 方法*。

要了解 Actor-Critic，请想象您玩电子游戏。你可以和一个朋友一起玩，他会给你一些反馈。你是演员，你的朋友是评论家。

![演员评论家](https://huggingface.co/blog/assets/89_deep_rl_a2c/ac.jpg)

一开始你不知道怎么玩，**所以你随机尝试一些动作**。Critic 观察您的行为并**提供反馈**。

从这些反馈中学习， **您将更新您的策略并更好地玩这个游戏。**

另一方面，您的朋友（Critic）也会更新他们提供反馈的方式，以便下次可以做得更好。

这就是 Actor-Critic 背后的想法。我们学习两个函数逼近：

- **控制我们的智能体人行为方式**的*策略*：\pi_{\theta}(s,a)*π**θ*的（*小号*，*一*）
- 通过衡量所采取的动作有多好来协助策略更新的*价值函数：*\hat{q}_{w}(s,a)*q*^的*w*的（*小号*，*一*）

### 演员-评论家过程

现在我们已经看到了 Actor Critic 的大局，让我们更深入地了解 Actor 和 Critic 在训练过程中是如何一起改进的。

正如我们所见，Actor-Critic 方法有两个函数近似（两个神经网络）：

- *Actor*，由 theta 参数化的**策略函数：**\pi_{\theta}(s,a)*π**θ*的（*小号*，*一*）
- *Critic*，一个由 w 参数化的**值函数：**\hat{q}_{w}(s,a)*q*^的*w*的（*小号*，*一*）

让我们看看训练过程，了解 Actor 和 Critic 是如何优化的：

- 在每个时间步 t，我们得到当前状态英石*小号**吨*的来自环境并将**其作为输入传递给我们的 Actor 和 Critic**。
- 我们的策略获取状态并**输出一个动作** 在*一个**吨*的.

![Step 1 演员评论家](https://huggingface.co/blog/assets/89_deep_rl_a2c/step1.jpg)

- Critic 也将该动作作为输入，并使用英石*小号**吨*的和在*一个**吨*的，**计算在该状态下采取该操作的值：Q 值**。

![Step 2 演员评论家](https://huggingface.co/blog/assets/89_deep_rl_a2c/step2.jpg)

- 那个动作在*一个**吨*的在环境中执行输出一个新的状态S_{t+1}*小号**吨*+ 1的和奖励R_{t+1}*R**吨*+ 1的.

![步骤 3 演员评论家](https://huggingface.co/blog/assets/89_deep_rl_a2c/step3.jpg)

- Actor 使用 Q 值更新其策略参数。

![步骤 4 演员评论家](https://huggingface.co/blog/assets/89_deep_rl_a2c/step4.jpg)

- 由于其更新的参数，Actor 产生了下一步要采取的动作A_{t+1}*一个**吨*+ 1的给定新状态S_{t+1}*小号**吨*+ 1的.
- 然后 Critic 更新它的值参数。

![步骤 5 演员评论家](https://huggingface.co/blog/assets/89_deep_rl_a2c/step5.jpg)

### 优势演员评论家 (A2C)

我们可以通过**使用 Advantage function 作为 Critic 而不是 Action value function 来**进一步稳定学习。

这个想法是 Advantage 函数计算**在一个状态下采取该动作与该状态的平均值相比如何更好**。它是从状态动作对中减去状态的平均值：

![优势功能](https://huggingface.co/blog/assets/89_deep_rl_a2c/advantage1.jpg)

换句话说，这个函数计算**的是如果我们在那个状态下采取这个动作，我们得到的额外奖励与我们在那个状态下得到的平均奖励相比**。

额外的奖励是超出该状态预期值的。

- 如果 A(s,a) > 0：我们的梯度被**推向那个方向**。
- 如果 A(s,a) < 0（我们的动作比那个状态的平均值差），**我们的梯度被推向相反的方向**。

实现这个优势函数的问题在于它需要两个价值函数—— Q(s,a)*问*(*小号*,*一*）和 V(s)*V* (*秒*). 幸运的是， **我们可以使用 TD 误差作为优势函数的良好估计量。**

![优势功能](https://huggingface.co/blog/assets/89_deep_rl_a2c/advantage2.jpg)

