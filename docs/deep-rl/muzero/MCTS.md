# 蒙特卡洛树搜索 (MCTS)

## 什么是 MCTS？

蒙特卡洛树搜索 (MCTS) 是一种在人工智能 (AI) 问题中寻找最优决策的方法，通常是组合游戏中的移动规划。它结合了随机模拟的普遍性和树搜索的精确性。

由于 MCTS 在计算机围棋方面的巨大成功以及对许多其他难题的潜在应用，对 MCTS 的研究兴趣急剧上升。它的应用超越了游戏，理论上MCTS 可以应用于任何可以用 { *state*，*action* } 对和用于预测结果的模拟来描述的领域。



## 基本算法

基本的 MCTS 算法很简单：根据模拟播放的结果逐个节点地构建搜索树。该过程主要分为四步：**选择**(Selection)，**拓展**(Expansion)，**模拟**(Simulation)，**反向传播**(Backpropagation)。

![img](https://web.archive.org/web/20180623055344im_/http://mcts.ai/about/mcts-algorithm-1a.png)

### 1. 基础知识

介绍 MCTS 的具体搜索算法之前，先介绍一下 MCTS 的基础知识。

#### 1.1 节点

在棋类问题中，MCTS 使用一个**节点**来表示一个**游戏状态**，换句话说，每一个节点都对应着井字棋中的一种情况。假设现在井字棋的棋盘上只有中间一个棋子，图中用 ○ 画出，我们用一个节点表示这个游戏状态，这个节点就是下图中的根节点。这时，下一步棋有 8 种下法，所以对应的，这个根节点就有 8 个子节点（受图片大小限制，图中只画出了 3 个）。

**下完一步后，游戏还没有结束，棋盘上还可以继续下棋，继续按照刚才的方法，一个节点表示一个游戏状态，这些子节点又有子节点，所有的井字棋游戏状态都可以被这样表示，于是它们就构成了一个树。**对于围棋或者其他更复杂的棋类也是一样，只不过这个树会更大、更复杂。蒙特卡洛树搜索就是要在这样一个树中搜索出下一步在哪个位置下棋最有可能获胜，即根节点的哪个子节点获胜概率最高。

#### 1.2 节点的两个属性

在蒙特卡洛树搜索中，我们将节点记作 $v$  ，在搜索过程中需要记录节点的访问次数和累计奖励，它们的表示符号如下：

1. $N(v)$：节点 $v$ 的**访问次数**，节点在搜索过程中被访问多少次，该值就是多少。
2. $Q(v)$：节点 $v$  的**累计奖励**，即节点在反向传播过程中获得的所有奖励(reward)求和。

所谓的**奖励(reward)**是一个数值，游戏结束时若获胜，奖励为 1，若失败，奖励为 0。

### 2. 搜索过程

下面介绍 MCTS 的具体搜索算法。

给定当前游戏状态，如何获得下一步的最佳下法呢？对于井字棋来说，当然可以在整个决策树中遍历所有可能性，直接找出最优策略。但若换成围棋等复杂的棋类，遍历的方法是显然不可行的，这时就需要在决策树中有选择地访问节点，并根据现有的有限信息做出最优决策。

在介绍下面的搜索过程之前，我们首先要知道：蒙特卡洛树搜索搜的是什么？换句话说，假如我们先把 MCTS 看成一个黑盒子，那么它的输入和输出分别是什么？

**输入**：一个游戏状态

**输出**：下一步下棋的位置

**也就是说，给 MCTS 一个棋局，它就告诉你下一步该怎么走。**知道了输入输出分别是什么后，我们再来看看从输入到输出这中间，MCTS 到底做了什么。总的来说，MCTS 按顺序重复执行以下四个步骤：**选择，拓展，模拟，反向传播。**

#### 2.1. 选择(Selection)

**根据上文所述，对于围棋等可能性非常多的问题，遍历的方法不可行，因此 MCTS 有选择地访问节点，这就是选择阶段。**从根节点(就是输入)出发，根据一定的策略，向下选择一个节点进行访问，若被选择的节点未被访问过，则执行扩展；若被选择的节点已被访问，则访问该节点，并继续向下选择节点进行访问，直到遇见未被访问的节点，或遇见终止节点(游戏结束)。

选择的策略由该公式确定，对当前节点的每个子节点计算如下公式，并选择计算结果最大的节点。
$$
\underset{v’\in \text{children of }v}{\mathrm{argmax}}\frac{Q\left( v’ \right)}{N\left( v’ \right)}+c\sqrt{\frac{\text{2}\ln N\left( v \right)}{N\left( v’ \right)}}
$$
其中，  $v$  表示父节点，$v’$ 表示子节点, $c$ 是一个常数，用于权衡**探索 (Exploration)** 与**利用 (Exploitation)**。探索是指选择一些之前没有尝试过的下法，丰富自己的知识，新的知识可能带来不错的结果；而利用是指根据现有的知识选择下法。$c$  越大，就越偏向于探索；$c$  越小，就越偏向于**利用**。

#### 2.2 扩展(Expansion)

MCTS 在搜索的过程中是有选择地访问节点，并把所有访问过的节点构建成一个树。扩展就是把**选择**步骤中遇到的未访问节点添加到树中，然后对该节点执行模拟。

#### 2.3 模拟 (Simulation)

模拟是一个粗略获取信息的过程。从被扩展的节点开始，对游戏进行模拟，也就是在棋盘上随机下棋，直到**游戏结束**。若此时游戏胜利，则**奖励 (Reward)** 记为 1；若游戏失败，**奖励**记为 0。

#### 2.4 反向传播 (Backpropagation)

反向传播是将在**模拟**中得到的奖励更新的过程。为什么叫反向传播呢？回顾一下第一步**选择**，我们从根节点向下一步一步地选择节点进行访问，现在我们将沿着这条路逐一更新节点信息，重新回到根节点，所以叫反向传播。

将获得的奖励记作 $R$，对当前节点，及其路径上的所有节点 $v$  ，都执行以下操作。**即，更新访问次数，对奖励进行累加。**
$$
N(v)=N(v)+1  \\
Q(v)=Q(v)+R
$$
我们再回头看看**选择**步骤中的公式:
$$
\underset{v’\in \text{children of }v}{\mathrm{argmax}}\frac{Q\left( v’ \right)}{N\left( v’ \right)}+c\sqrt{\frac{\text{2}\ln N\left( v \right)}{N\left( v’ \right)}}
$$
可以看到，式中第一项其实就是该节点在前面的过程中获得的平均奖励，自然第一项的值越大，在现有的知识下，选择该节点更有可能获胜。式中第二项，当该节点访问次数占父节点次数的比例越小时，该值越大，表示该节点访问次数很少，可以多进行尝试，获取新的知识，它们也可能获得更丰厚的回报。于是 $c$ 就是控制这两者重要程度的参数。

这就是**上限置信区间算法 (Upper Confidence Bound )**。

每个节点必须包含两条重要信息：基于模拟结果的估计值和访问次数。

在其最简单和内存效率最高的实现中，MCTS 将在每次迭代中添加一个子节点。但是请注意，根据应用程序，每次迭代添加多个子节点可能会有所帮助。

### 3. 搜索结束

MCTS 的整个过程就是这样，那么什么时候结束呢？一般设置以下两个终止条件。

1. 设置最大根节点搜索次数，达到该次数后结束搜索。
2. 设置最大搜索时间，超过时间后结束搜索。

**结束后，就输出当前状态下，下一步下棋的位置。**

### 4. 选择最佳节点

搜索结束后，如何选择下一步下棋的位置呢？

**不是选择 $Q$ 最大的节点，也不是选择平均奖励最大的节点，而是选择访问次数最多的节点。这样，就得到了当前游戏状态(根节点)下的一个选择。**或者，也可以将访问次数归一化，作为下一步的概率。

如果下一步还要进行决策，则又要将下一步的状态作为根节点，重新执行 MCTS，并选择访问次数最多的节点作为下一步的策略。(上一步的搜索结果可以保留)

Bandits 和 UCB
树下降过程中的节点选择是通过选择最大化某个数量的节点来实现的，类似于*多臂老虎机问题*，在该问题中，玩家必须选择每轮都最大化估计奖励的老虎机（老虎机）。通常使用以下形式的置信上限 (UCB) 公式：

   ![img](https://web.archive.org/web/20180623055344im_/http://mcts.ai/about/ucb-1.png)

其中*vi*是节点的估计值，*ni是节点*被访问的次数，*N*是其父节点被访问的总次数*。C*是可调偏置参数。

### 开发与探索

UCB 公式平衡了已知奖励的开发与相对未访问节点的*探索*以鼓励他们的锻炼*。*奖励估计是基于随机模拟，因此在这些估计变得可靠之前必须多次访问节点；MCTS 估计在搜索开始时通常是不可靠的，但在给定足够时间的情况下会收敛到更可靠的估计，并在给定无限时间的情况下收敛到更可靠的估计。

### MCTS 和 UCT

Kocsis 和 Szepervari (2006) 首先通过将 UCB 扩展到极小极大树搜索来形式化完整的 MCTS 算法，并将其命名为树的置信上限 (UCT) 方法。这是绝大多数当前 MCTS 实现中使用的算法。

UCT可以描述为MCTS的一个特例，即：UCT = MCTS + UCB。

------

## 优势

与传统的树搜索方法相比，MCTS 具有许多优势。

### 启发式

MCTS 不需要任何关于给定领域的战略或战术知识来做出合理的决定。该算法可以在除了合法移动和结束条件之外的游戏知识的情况下有效运行；这意味着一个单一的 MCTS 实现可以在稍作修改的情况下重复用于许多游戏，并使 MCTS 成为一般游戏的潜在福音。

### 非对称

MCTS 执行适应搜索空间拓扑的非对称树生长。该算法更频繁地访问更有趣的节点，并将其搜索时间集中在树的更相关部分。

![img](https://web.archive.org/web/20180623055344im_/http://mcts.ai/mcts-tree-4.png)

这使得 MCTS 适用于分支因子较大的游戏，例如 19x19 的围棋。如此大的组合空间通常会导致标准的基于深度或广度的搜索方法出现问题，但 MCTS 的自适应特性意味着它将（最终）找到那些看起来最优的移动并将其搜索工作集中在那里。

### Anytime

可以随时停止算法以返回当前最佳估计值。到目前为止构建的搜索树可能会被丢弃或保留以供将来重用。

### 优雅

该算法易于实现（请参阅[代码](https://web.archive.org/web/20180623055344/http://mcts.ai/code/index.html)）。

------

## 缺点

MCTS 有一些缺点，但它们可能是主要的。

### 发挥实力

即使是中等复杂度的游戏，MCTS 算法的基本形式也无法在合理的时间内找到合理的着法。这主要是由于组合移动空间的巨大规模以及关键节点可能没有被访问足够多次以提供可靠估计的事实。

### 速度

MCTS 搜索可能需要多次迭代才能收敛到一个好的解决方案，这对于难以优化的更通用的应用程序来说可能是一个问题。例如，最好的围棋实施可能需要数百万次的比赛，并结合领域特定的优化和增强功能，才能做出专家级的举动，而最好的 GGP 实施可能每秒只能进行数十次（独立于领域的）比赛，以应对更复杂的游戏。对于合理的移动时间，此类 GGP 可能几乎没有时间访问每个合法移动，并且不太可能进行重大搜索。

幸运的是，可以使用多种技术显着提高算法的性能。

------

## 改进

迄今为止，已经提出了数十项 MCTS 增强功能。这些通常可以描述为领域知识或领域独特性。

### 领域知识

特定于当前游戏的领域知识可以在树中被利用来过滤掉不可信的动作，或者在模拟中产生更类似于人类对手之间发生的比赛的*大量比赛。*这意味着播出结果将比随机模拟更真实，并且节点将需要更少的迭代来产生真实的奖励值。

领域知识可以产生显着的改进，但代价是速度和通用性的丧失。

### 域独立

域独立增强适用于所有问题域。这些通常应用于树中（例如 AMAF），尽管有些再次应用于模拟（例如更喜欢在播出期间获胜的动作）。域独立增强不将实现绑定到特定域，保持通用性，因此是该领域大多数当前工作的重点。

------

##  Context

1928 年： John von Neumann 的极小极大定理为对抗树搜索方法铺平了道路，这些方法几乎从一开始就构成了计算机科学和人工智能决策的基础。

1940 年代：蒙特卡洛 (MC) 方法被形式化为一种通过使用随机抽样来处理不适合树搜索的不太明确的问题的方法。

2006 年： Rémi Coulomb 和其他研究人员将这两个想法结合起来，提供了一种在计算机围棋中进行移动规划的新方法，现在称为 MCTS。Kocsis 和 Szepesvári 将这种方法形式化为 UCT 算法。

这种优雅的算法没有早点被发现，这似乎很了不起！



## Reference

1. https://iqhy.github.io/posts/2019/1028154602/
2. https://web.archive.org/web/20180629082128/http://mcts.ai/index.html
