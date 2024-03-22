# 强化学习环境

## Gymnasium

Gymnium 是OpenAI  Gym 库的一个分支,（几年前OpenAI将维护工作交给了一个外部团队），现在 Openai 不在对 Gym 进行维护，现在主要由Gymnium 进行维护。

- Github:  https://gymnasium.farama.org/

- 文档： https://github.com/Farama-Foundation/Gymnasium

## IsaacGymEnvs

特点：基于 Cuda  重写了强化学习的环境模拟，使得环境模拟可以使用 gpu 进行加速

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

## Neural MMO

Github：https://github.com/NeuralMMO/environment

Pages: https://neuralmmo.github.io/_build/html/rst/landing.html

[Neural MMO](https://github.com/NeuralMMO/environment) 最初由 OpenAI 发布，作为测试多智能体强化学习方法的环境。 MMO 代表大型多人在线游戏，因为该项目的灵感来自于 MMORPG。该环境提供了优化的实现，您只需使用一个 CPU 即可运行数百个智能体。

## Android Env

Github：https://github.com/google-deepmind/android_env

[Android Env](https://github.com/deepmind/android_env)由 DeepMind 发布。它是一个为构建与 Android 操作系统交互的智能体提供平台的环境。Android Env 库将根据定义的任务返回像素观察结果和奖励。

使用此环境，可以训练智能体使用 RL 与 Android Env 交互并完成任务。

## MuJoCo

Github：https://github.com/google-deepmind/mujoco

Pages: https://mujoco.org/

[MuJoCo](https://github.com/deepmind/mujoco)代表接触式多关节动力学。它由 DeepMind 发布。它是通用物理引擎，旨在促进机器人、生物力学、图形和动画、机器学习以及其他需要快速准确地模拟铰接结构与其环境相互作用的领域的研究和开发。

## FinRL

Github: https://github.com/AI4Finance-Foundation/FinRL

[FinRL](https://github.com/AI4Finance-Foundation/FinRL)是最早的金融开源 RL 环境之一。它可以集成到各种数据源中，无论是数据源还是用户导入的数据集，甚至是模拟数据。

## RecSim

Github: https://github.com/google-research/recsim

[RecSim](https://github.com/google-research/recsim)是一个可配置的系统，用于将强化学习应用于推荐系统。 RecSim 模拟推荐智能体与由用户模型、文档模型和用户 选择 模型组成的环境的交互。

## RLCard

GIthub：https://github.com/datamllab/rlcard

Pages: http://www.rlcard.org/

[RLCard](https://github.com/datamllab/rlcard)是支持各种纸牌游戏的 RL 环境。它由莱斯大学和德克萨斯农工大学的[DATA 实验室](http://faculty.cs.tamu.edu/xiahu/)以及社区贡献者开发。它支持多种游戏和算法，如下所示:

ome of the games supported :

- BlackJack
- NoLimit Texas Hold’em
- Gin Rummy
- Bridge
- Dou Dizhu
- Uno

Some of Algorithms implemented are

- Counterfactual Regret minimization
- Neural Fictitious Self-Play
- Deep Q-Learning
- Deep Monte-Carlo

## OpenSpiel

GitHub：https://github.com/deepmind/open_spiel

OpenSpiel，是一个强化学习环境和算法的集合。用于研究强化学习算法和游戏中的搜索/规划。OpenSpiel 的目的是通过与一般游戏类似的方式促进跨多种不同游戏类型的一般多智能体强化学习，但是重点是强调学习而不是竞争形式。当前版本的 OpenSpiel 包含 20 多种游戏的不同类型（完全信息、同步移动、不完全信息、网格世界游戏、博弈游戏和某些普通形式/矩阵游戏）实现。

核心的 OpenSpiel 实现基于 C ++ 和 Python 绑定，这有助于在不同的深度学习框架中采用。该框架包含一系列游戏，允许 DRL agent 学会合作和竞争行为。同时，OpenSpiel 还包括搜索、优化和单一 agent 等多种 DRL 算法组合。OpenSpiel 还包括用于分析学习动态和其他常见评估指标的工具。

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

## Deep RTS

Github: https://github.com/cair/deep-rts

DeepRTS 是一款用于强化学习研究的高性能实时策略游戏。它是为了提高性能而用 C++ 编写的，提供了一个 python 接口，以便更好地与机器学习工具包交互。深度 RTS 可以每秒超过**6,000,000**步处理游戏，渲染图形时可以达到**2,000,000**步。与《星际争霸》等其他解决方案相比，在配备 Nvidia RTX 2080 TI 的英特尔 i7-8700k 上运行的**模拟时间快了 15 000%以上。**

Deep RTS 的目标是通过减少计算时间，为 RTS AI 研究带来更实惠、更可持续的解决方案。

## MicroRTS

Github: https://github.com/Farama-Foundation/MicroRTS-Py

[Gym-μRTS](https://github.com/Farama-Foundation/gym-microrts)在客观上与 Deep RTS 类似。现有的完整游戏具有很高的计算成本，通常意味着数千小时的 CPU 和 GPU 时间。这对研究来说是一个巨大的限制。

## Nocturne

GIthub: https://github.com/facebookresearch/nocturne

[Nocturne](https://github.com/facebookresearch/nocturne)是 Facebook Research 发布的用于模拟多智能体系统的 RL 环境。 Nocturne 是具有部分可观察性的 2D 驾驶模拟环境。它是用 C++ 构建的，以提高速度。目前，Nocturne 可以处理来自[Waymo 开放数据集的](https://github.com/waymo-research/waymo-open-dataset)数据，但可以对其进行修改以处理其他数据源。使用 Python 库`nocturne`，人们能够训练自动驾驶汽车控制器来解决 Waymo 数据集（我们提供的基准）中的各种任务，然后使用我们提供的工具来评估设计的控制器。

利用这种丰富的数据源，Nocturne包含了广泛的场景，其解决方案需要形成复杂的协调、心理理论和处理部分可观察性。

## CityFlow

Github: https://github.com/cityflow-project/CityFlow/

Pages: https://cityflow.readthedocs.io/en/latest/

[CityFlow](https://github.com/cityflow-project/CityFlow/)是一个针对大规模城市交通场景的多智能体强化学习环境。

它具有以下特点

- 微观交通模拟器，模拟每辆车的行为，提供交通演变的最高级别细节。
- 支持路网和交通流的灵活定义
- 为强化学习提供友好的Python接口
- 快速地！精心设计的数据结构和多线程模拟算法。能够模拟全市交通。查看与 SUMO 的性能比较

## CyberBattleSim

Github： https://github.com/microsoft/CyberBattleSim

Pages:

[CyberBattleSim](https://github.com/microsoft/CyberBattleSim)是一个框架，用于构建复杂计算机网络和系统的抽象模拟，以便应用强化学习。了解智能体如何在这种复杂的环境中交互和发展是很有帮助的。它本质上是网络安全的游戏化，以便可以应用强化学习。

CyberBattleSim 是一个实验研究平台，用于研究在模拟抽象企业网络环境中运行的自动化智能体的交互。该模拟提供了计算机网络和网络安全概念的高级抽象。其基于 Python 的 Open AI Gym 界面允许使用强化学习算法来训练自动化智能体。

模拟环境由固定的网络拓扑和一组漏洞参数化，智能体可以利用这些漏洞在网络中横向移动。攻击者的目标是通过利用计算机节点中植入的漏洞来获得部分网络的所有权。当攻击者试图在整个网络中传播时，防御者智能体会监视网络活动并尝试检测正在发生的任何攻击，并通过驱逐攻击者来减轻对系统的影响。我们提供了一个基本的随机防御者，可以根据预定义的成功概率来检测和减轻正在进行的攻击。我们通过重新镜像受感染的节点来实施缓解措施，该过程被抽象建模为跨越多个模拟步骤的操作。

为了比较智能体的性能，我们查看两个指标：为实现目标而采取的模拟步骤数量以及整个训练时期模拟步骤的累积奖励。

-



# 强化学习工具箱

## Mava

https://github.com/instadeepai/Mava

Mava 是一个用于构建多智能体强化学习 (MARL) 系统的库。Mava 为 MARL 提供了有用的组件、抽象、实用程序和工具，并允许对多进程系统训练和执行进行简单的扩展，同时提供高度的灵活性和可组合性。

Mava 框架的核心是系统的概念。系统是指完整的多智能体强化学习算法，由以下特定组件组成：执行器、训练器和数据集。

“执行器(Executor)”是系统的一部分，它与环境交互，为每个智能体采取行动并观察下一个状态作为观察集合，系统中的每个智能体一个。本质上，执行器是 Acme 中 Actor 类的多智能体版本，它们本身是通过向执行器提供策略网络字典来构建的。Trainer 负责从最初从 executor 收集的 Dataset 中采样数据，并更新系统中每个 agent 的参数。因此，Trainers 是 Acme 中 Learner 类的多智能体版本。数据集以字典集合的形式存储了执行者收集的所有信息，用于操作、观察和奖励，并带有与各个智能体 ID 对应的键。基本系统设计如上图左侧所示。可以查看系统实现的几个示例

## Algorithms

1. https://github.com/kengz/SLM-Lab

2. https://github.com/rail-berkeley/rlkit

3. https://github.com/ChenglongChen/pytorch-DRL


## 其他资源链接

- https://github.com/aikorea/awesome-rl

- https://github.com/tigerneil/awesome-deep-rl

- https://github.com/kengz/awesome-deep-rl

- https://github.com/clvrai/awesome-rl-envs
