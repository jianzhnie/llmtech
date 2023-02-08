#  MuZero 直觉

MuZero 是向前迈出的非常令人兴奋的一步——它不需要游戏规则或环境动态方面的特殊知识，而是为自己学习环境模型并使用该模型进行规划。尽管它使用了这样一个学习模型，MuZero 保留了 AlphaZero 的完整规划性能——打开了将其应用于许多现实世界问题的大门！

## It's all just statistics

*MuZero*是一种机器学习算法，自然首先要了解它是如何使用神经网络的。它从 AlphaGo 和 AlphaZero 继承了策略和价值网络[1](https://www.furidamu.org/blog/2020/12/22/muzero-intuition/#fn:1)的使用：

![从围棋棋盘到价值响应的价值和政策网络映射示意图。 政策估计](https://www.furidamu.org/images/2020-12-22-muzero-intuition/value_policy_net.webp)

policy 和 value 都有非常直观的含义：

- 政策，$p(s, a)$, 是在状态 $s$ 时，所有动作可以采取动作$a$的概率分布, . 它估计哪个动作可能是最佳动作。该策略类似于人类玩家在快速浏览游戏时对好棋的第一次猜测。
- 价值 $v(s)$ 估计从当前状态$s$获胜的概率：对所有可能的未来可能性进行平均，根据它们的可能性加权，当前玩家的获胜几率。

这些网络中的每一个本身就已经非常强大：如果你只有一个策略网络，你就可以简单地总是按照它预测的最有可能的着法下棋，并最终得到一个非常不错的玩家。同样，只给定一个价值网络，你总是可以选择价值最高的着法。然而，结合这两种估计会产生更好的结果。

## Planning to Win

与之前的*AlphaGo*和*AlphaZero类似，* *MuZero*使用蒙特卡洛树搜索[2](https://www.furidamu.org/blog/2020/12/22/muzero-intuition/#fn:2)（简称 MCTS）来聚合神经网络预测并选择要应用于环境的动作。

MCTS 是一个迭代的、最佳优先的树搜索过程。最佳优先意味着搜索树的扩展由搜索树中的值估计指导。与广度优先（将整棵树扩展到固定深度再进行更深搜索）或深度优先（连续扩展每条可能的路径直到游戏结束后再尝试下一条）等经典方法相比，最佳优先搜索可以利用启发式估计（例如神经网络）即使在非常大的搜索空间中也能找到有前途的解决方案。

MCTS 具有三个主要阶段：模拟、扩展和反向传播。通过重复执行这些阶段，MCTS 在未来的动作序列上一次一个节点地逐步构建搜索树。在这棵树中，每个节点都是一个未来状态，而节点之间的边代表从一个状态到下一个状态的动作。

在我们深入细节之前，让我介绍一下这种搜索树的示意图，包括*MuZero*做出的神经网络预测：

![muzero 搜索树图，以及表示、动态和预测函数的使用](https://www.furidamu.org/images/2020-12-22-muzero-intuition/search_tree.webp)

圆圈代表树的节点，对应于环境中的状态。线代表动作，从一个状态到下一个状态。树根植于顶部，处于环境的当前状态 - 由示意图 Go 板表示。我们将在后面的部分中详细介绍表示、预测和动态函数。

**模拟** 总是从树的根部（图中顶部的浅蓝色圆圈）开始，即环境或游戏中的当前位置。在每个节点（状态$s$), 它使用评分函数 $U(s,a)$  比较不同的动作$a$并选择了最有前途的。*MuZero*中使用的评分函数将结合先前的估计$p(s,a)$与价值估计 $v(s,a)$:

$U(s,a) =v(s,a) + c* p(s,a) $

其中 $C$ 是一个缩放因子[3](https://www.furidamu.org/blog/2020/12/22/muzero-intuition/#fn:3)，它确保随着我们的价值估计变得更准确，先验的影响会减少。

每次选择一个动作时，我们都会增加其相关的访问次数$n(s,a)$, 用于 UCB 比例因子$c$并用于以后的动作选择。

模拟沿着树向下进行，直到到达尚未展开的叶子；此时神经网络用于评估节点。评估结果（先验和价值估计）存储在节点中。

**Expansion**：一旦节点达到一定数量的评估，它就被标记为“已扩展”。被扩展意味着可以将子节点添加到节点；这允许搜索进行得更深入。在*MuZero*中，扩展阈值为 1，即每个节点在第一次评估后立即扩展。更高的扩展阈值有助于在深入搜索之前收集更可靠的统计数据[4](https://www.furidamu.org/blog/2020/12/22/muzero-intuition/#fn:4)。

**反向传播**：最后，神经网络评估的价值估计被传播回搜索树；每个节点都保留其下方所有价值估计的运行平均值。这个平均过程允许 UCB 公式随着时间的推移做出越来越准确的决策，从而确保 MCTS 最终会收敛到最佳着法。

## Intermediate Rewards

细心的读者可能已经注意到，上图中还包含了一个量*r*的预测. 某些领域，例如棋盘游戏，仅在一集结束时提供反馈（例如输赢结果）；它们可以完全通过价值估计来建模。然而，其他领域提供更频繁的反馈，在一般情况下是奖励*r*在从一个状态到下一个状态的每次转换之后被观察到。

通过神经网络预测直接对该奖励建模并将其用于搜索是有利的。它只需要对 UCB 公式稍作修改：


$$
U(s, a)=r(s, a)+\gamma \cdot v\left(s^{\prime}\right)+c \cdot p(s, a)
$$


其中r(s,a)是从状态$s$转换时通过选择行动$a$观察到的奖励， 和*γ*是一个折扣因子，描述了我们对未来奖励的关心程度。

由于一般奖励可以具有任意比例，我们进一步规范化组合奖励/价值估计以位于区间[ 0 ,1 ]内：


$$
U(s, a)=\frac{r(s, a)+\gamma \cdot v\left(s^{\prime}\right)-q_{\min }}{q_{\max }-q_{\min }}+c \cdot p(s, a)
$$
其中， $q_{\min }$ and $q_{\max }$ are the minimum and maximum $r(s, a)+\gamma \cdot v\left(s^{\prime}\right)$ estimates observed across the search tree.

## Episode Generation

可以重复应用上述 MCTS 过程来玩整个剧集：

- 在当前环境状态$s_t$下运行搜索。
- 根据统计 $\pi_t$的搜索选择一个动作$a_{t+1}$。
- 将操作应用于环境以前进到下一个状态$s_{t+1}$并观察奖励$u_{t+1}$.
- 重复直到环境终止。

![通过在每个状态下运行 MCTS 来生成情节，选择一个动作并推进环境](https://www.furidamu.org/images/2020-12-22-muzero-intuition/episode_generation.webp)

动作选择可以是 $greedy$ ——访问次数最多的动作, 也可以是探索性的：采样动作与其访问次数成正比，可能在施加一些温度之后控制探索程度：

$$p(a)=\left(\frac{n(s, a)}{\sum_b n(s, b)}\right)^{1 / t}$$

对 t=0，我们恢复贪婪的动作选择；$t=inf$ 相当于均匀采样动作。

## Training

现在我们知道如何运行 MCTS 来选择动作、与环境交互和生成episodes，我们可以转向训练*MuZero*模型。

我们首先从数据集中采样轨迹和其中的位置，然后沿着轨迹展开*MuZero模型：*

![训练沿着轨迹展开 muzero 模型](https://www.furidamu.org/images/2020-12-22-muzero-intuition/training.webp)

*您可以看到运行中的MuZero*算法的三个部分：

- **representation**  函数 $h$, 使用神经网络从一组观察结果（示意图 Go 棋盘）映射到隐藏状态$s$
- **dynamics** 函数 $g$ ， 将状态 $s_t$ 采取的动作$a_{t+1} 映射到 $$s_{t+1}$. 它还估计在这种转变中观察到的奖励$r_t$。这就是允许学习模型在搜索中前滚的原因。
- **prediction** 函数 $f$ ， 基于状态 $s_t$ 对策略$p_t$和值函数$v_t$进行估计. 这些是 UCB 公式使用的估计值，并在 MCTS 中汇总。

用作网络输入的观察和动作取自该轨迹；类似地，策略、价值和奖励的预测目标是生成轨迹时存储的轨迹。

您可以在全图中更清楚地看到episode generation ( **B** ) 和训练 ( **C ) 之间的对齐：**

![一张图片中的前三个数字](https://www.furidamu.org/images/2020-12-22-muzero-intuition/full_algorithm.webp)

具体来说，*MuZero*估计的三个量的训练损失为：

- **policy**：MCTS 访问计数统计数据与来自预测函数的策略 logits 之间的交叉熵。
- **value**：N 个奖励的折扣总和 + 存储的搜索值或目标网络估计值与预测函数的值之间的交叉熵或均方误差。
- **reward**：轨迹中观察到的奖励与动态函数估计之间的交叉熵。

## Reanalyse

在检查了核心*MuZero*培训之后，我们准备好了解允许我们利用搜索实现大规模数据效率改进的技术：重新分析。

在正常训练过程中，我们会生成许多轨迹（与环境的交互）并将它们存储在我们的回放缓冲区中以供训练。我们能否从这些数据中获得更多收益？

![表示情节的状态序列](https://www.furidamu.org/images/2020-12-22-muzero-intuition/trajectory.webp)

不幸的是，由于这是存储的数据，我们无法更改状态、动作或收到的奖励——这需要将环境重置为任意状态并从那里继续。在黑客帝国中可能，但在现实世界中则不然。

幸运的是，事实证明我们不需要 - 使用带有新的、改进的标签的现有输入足以继续学习。感谢*MuZero*的学习模型和 MCTS，这正是我们可以做的：

![每个状态都有新的 MCTS 树的状态序列](https://www.furidamu.org/images/2020-12-22-muzero-intuition/reanalyse.webp)

我们保持保存的轨迹（观察、行动和奖励）不变，而只是重新运行 MCTS。这会生成新的搜索统计数据，为我们提供新的策略和价值预测目标。

与在直接与环境交互时使用改进的网络进行搜索会产生更好的搜索统计数据一样，在保存的轨迹上使用改进的网络重新运行搜索也会产生更好的搜索统计数据，从而允许使用相同的轨迹数据进行重复改进.

Reanalyse自然适合*MuZero*训练循环。让我们从正常的训练循环开始：

![参与者和学习者在训练期间交换数据的图表](https://www.furidamu.org/images/2020-12-22-muzero-intuition/training_loop.webp)

我们有两组相互异步通信的作业：

- 接收最新轨迹的**学习**器将最新的轨迹保存在重放缓冲区中，并使用它们来执行上述训练算法。
- 多个**参与者**定期从学习者那里获取最新的网络检查点，使用 MCTS 中的网络来选择动作并与环境交互以生成轨迹。

为了实施再分析，我们引入了两个工作：

![之前的图表扩展了再分析参与者](https://www.furidamu.org/images/2020-12-22-muzero-intuition/reanalyse_loop.webp)

- 一个**重新分析缓冲区**，它接收参与者生成的所有轨迹并保留最新的轨迹。
- 多个**再分析参与者** [6](https://www.furidamu.org/blog/2020/12/22/muzero-intuition/#fn:6)从再分析缓冲区中对存储的轨迹进行采样，使用来自学习器的最新网络检查点重新运行 MCTS，并将生成的轨迹和更新的搜索统计信息发送给学习器。

对于学习者来说，“新鲜”和重新分析的轨迹是无法区分的；这使得改变新鲜轨迹与重新分析轨迹的比例变得非常简单。

## What's in a name?

*MuZero*的名字当然是基于*AlphaZero* - 保留*零*表示它是在没有模仿人类数据的情况下进行训练的，并将*Alpha*替换为*Mu*表示它现在使用学习模型来规划。

再深入一点，我们会发现*Mu*的含义很丰富：

- [梦](https://jisho.org/search/夢 %23kanji)在日语中可以读作*mu*，意思是“梦想”——就像*MuZero*使用学习的模型来想象未来的场景一样。
- 希腊字母 μ，发音为*mu*，也可以代表学习模型。
- [无](https://jisho.org/search/無 %23kanji)，在日语中发音*为mu*，意思是“无”——加倍强调从头开始学习的概念：不仅没有可模仿的人类数据，甚至没有提供规则。

## Final Words

我希望*MuZero*的这篇总结有用！

如果您对更多细节感兴趣，请从[全文](https://www.nature.com/articles/s41586-020-03051-4)( [pdf](https://drive.google.com/file/d/1n0ui9VctPYyuYsEYkSU6g6bqyGjufICS/view?usp=sharing) ) 开始。我还在[NeurIPS](https://www.youtube.com/watch?v=vt5jOSy7cz8&t=2s)（[海报](https://storage.googleapis.com/deepmind-media/research/muzero_poster_neurips_2019.pdf)）和[最近在 ICAPS](https://www.youtube.com/watch?v=L0A86LmH7Yw)上发表了关于 MuZero 的演讲。另请参阅最新的[伪代码](https://gist.github.com/Mononofu/6c2d27ea1b3a9b3c1a293ebabed062ed)。

最后，让我链接一些我觉得有趣的其他研究人员的文章、博客文章和 GitHub 项目：

- [一个简单的 Alpha(Go) 零教程](https://web.stanford.edu/~surag/posts/alphazero.html)
- [MuZero 通用](https://github.com/werner-duvaud/muzero-general)实现
- [如何使用 Python 构建您自己的 MuZero AI](https://medium.com/applied-data-science/how-to-build-your-own-muzero-in-python-f77d5718061a)