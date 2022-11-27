



## OpenSpiel

GitHub：https://github.com/deepmind/open_spiel

游戏在 DRL agent的 训练中发挥着重要作用。与其他数据集一样，游戏本质上基于试验和奖励机制，可用于训练 DRL agent。但是，游戏环境的复杂度还远远不够。

OpenSpiel，是一个强化学习环境和算法的集合。用于研究强化学习算法和游戏中的搜索/规划。OpenSpiel 的目的是通过与一般游戏类似的方式促进跨多种不同游戏类型的一般多智能体强化学习，但是重点是强调学习而不是竞争形式。当前版本的 OpenSpiel 包含 20 多种游戏的不同类型（完美信息、同步移动、不完美信息、网格世界游戏、博弈游戏和某些普通形式/矩阵游戏）实现。

核心的 OpenSpiel 实现基于 C ++ 和 Python 绑定，这有助于在不同的深度学习框架中采用。该框架包含一系列游戏，允许 DRL agent 学会合作和竞争行为。同时，OpenSpiel 还包括搜索、优化和单一 agent 等多种 DRL 算法组合。

### 支持的游戏

Spiel意指桌面游戏。因此，OpenSpiel中的环境就是相关棋牌类游戏。一共有28款：

> 双陆棋、突围棋、定约桥牌、Coin Game、屏风式四子棋、协作推箱子、国际象棋、第一价格密封拍卖、围棋、Goofspiel（一种多玩家纸牌游戏）
>
> 三宝棋、六贯棋、Kuhn扑克、Leduc扑克、大话骰、Markov Soccer、配对硬币（3人游戏）、矩阵游戏、Oshi-Zumo、西非播棋、转盘五子棋、Phantom三连棋
>
> Pig游戏、三连棋、Tiny Bridge、Y（一种棋类游戏）、Catch（仅支持Python）、Cliff-Walking在悬崖边走的醉汉（仅支持Python）。

### 支持的算法

目前，在OpenSpiel中实现的算法一共有24种，分别是：

> 极小化极大（Alpha-beta剪枝）搜索、蒙特卡洛树搜索、序列形式线性规划、虚拟遗憾最小化（CFR）、Exploitability
>
> 外部抽样蒙特卡洛CFR、结果抽样蒙特卡洛CFR、Q-learning、价值迭代、优势动作评论算法(Advantage Actor Critic，A2C)、Deep Q-networks (DQN)
>
> 短期价值调整（EVA）、Deep CFR、Exploitability 下降(ED) 、（扩展形式）虚拟博弈（XFP）、神经虚拟自博弈(NFSP)、Neural Replicator Dynamics（NeuRD）
>
> 遗憾策略梯度（RPG, RMPG）、策略空间回应oracle（PSRO）、基于Q的所有行动策略梯度（QPG）、回归CFR (RCFR)、PSROrN、α-Rank、复制/演化动力学。

### 支持的博弈类型

在OpenSpiel的游戏可以表示为各种广泛形式的博弈：

- 常和博弈
- 零和博弈
- 协调博弈
- 一般博弈

其中，常和博弈中智能体之间是严格的竞争关系，协调博弈中智能体之间是严格的竞争关系，一般博弈则介于两者之间。

另外，根据智能体能否获得博弈过程中的所有信息，又可以将博弈分为：

- 完美信息博弈
- 不完美信息博弈

象棋和围棋是没有偶然事件的完美信息博弈，双陆棋是有偶然事件的完美信息博弈，而像石头剪刀布、扑克这样的游戏属于不完美信息博弈。

### OpenSpiel怎么样？

OpenSpiel提供了一个带有C++基础的通用API ，它通过Python绑定（经由pybind11）公开。

游戏用C++编写，是因为可以用快速和内存效率更高的方法实现基本算法。一些自定义RL环境也会在Python中实现。

最重要的是，OpenSpiel的设计易于安装和使用、易于理解、易于扩展并且通用。OpenSpiel按照以下两个重要设计标准构建：

1、简单。代码应该是非编程语言专家可读、可用、可扩展的，特别是来自不同领域的研究人员。

OpenSpiel提供了用于学习和原型化的参考实现，而不是需要额外假设（缩小范围）或高级（或低级）语言特性的完全优化或高性能代码。

2、轻量。对于长期兼容、维护和易用性，依赖项可能会有问题。除非有充分的理由，否则倾向于避免引入依赖关系来保持便携性和易于安装。

## Gymnasium

Gymnium 是OpenAI  Gym 库的一个分支,（几年前OpenAI将维护工作交给了一个外部团队），现在 Openai 不在对 Gym 进行维护，现在主要由Gymnium 进行维护。

- Github:  https://gymnasium.farama.org/

- 文档： https://github.com/Farama-Foundation/Gymnasium

## IsaacGymEnvs

特点：基于cuda  重写了强化学习的环境模拟，使得环境模拟可以使用 gpu 进行加速

- 项目地址：https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/
- 在线文档：https://developer.nvidia.com/isaac-gym
- arXiv 链接：https://arxiv.org/abs/2108.10470

## Envpool

**EnvPool 是一个基于 C++ 、高效、通用的强化学习并行环境（vectorized environment）模拟器**

- 项目地址：https://github.com/sail-sg/envpool
- 在线文档：https://envpool.readthedocs.io/en/latest/
- arXiv 链接：https://arxiv.org/abs/2206.10558

## PettingZoo

PettingZoo 用于进行多智能体强化学习研究环境，类似于Gym的多智能体版本.

- https://pettingzoo.farama.org/
- https://github.com/Farama-Foundation/PettingZoo



## algorithms

1. https://github.com/kengz/SLM-Lab

2. https://github.com/rail-berkeley/rlkit

3. https://github.com/ChenglongChen/pytorch-DRL



## 其他资源链接

- https://github.com/aikorea/awesome-rl

- https://github.com/tigerneil/awesome-deep-rl

- https://github.com/kengz/awesome-deep-rl

- https://github.com/clvrai/awesome-rl-envs

- https://machinelearningknowledge.ai/reinforcement-learning-environments-platforms-you-did-not-know-exist/
