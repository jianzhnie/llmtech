# 强化学习环境

## Gymnasium

*Gym*除了是最广为人知的基准测试之外，还是一个用于开发和比较强化学习算法的工具包。它支持教学智能体的所有内容，从行走模拟人形机器人（需要 MuJoCo，请参阅[PyBullet Gymperium](https://github.com/benelot/pybullet-gym)了解免费替代方案）到玩 Pong 或 Pinball 等 Atari 游戏。

Gymnium 是OpenAI  Gym 库的一个分支,（几年前OpenAI将维护工作交给了一个外部团队），现在 Openai 不在对 Gym 进行维护，现在主要由Gymnium 进行维护。

- Github:  https://gymnasium.farama.org/

- 文档： https://github.com/Farama-Foundation/Gymnasium

## ViZDoom

ViZDoom 允许开发仅使用视觉信息（屏幕缓冲区）**玩 Doom 的**AI机器人。它主要用于机器视觉学习，特别是深度强化学习的研究。

ViZDoom 基于[ZDoom](https://zdoom.org/)引擎提供游戏机制。

- 项目地址：https://github.com/Farama-Foundation/ViZDoom
- 在线文档：https://vizdoom.farama.org/

## PettingZoo

PettingZoo 用于进行多智能体强化学习研究环境，类似于Gym的多智能体版本.

- https://pettingzoo.farama.org/
- https://github.com/Farama-Foundation/PettingZoo

## Neural MMO

[Neural MMO](https://github.com/NeuralMMO/environment) 最初由 OpenAI 发布，作为测试多智能体强化学习方法的环境。 MMO 代表大型多人在线游戏，因为该项目的灵感来自于 MMORPG。该环境提供了优化的实现，您只需使用一个 CPU 即可运行数百个智能体。

- Github：https://github.com/NeuralMMO/environment

- Pages: https://neuralmmo.github.io/_build/html/rst/landing.html

## Android Env

[Android Env](https://github.com/deepmind/android_env)由 DeepMind 发布。它是一个为构建与 Android 操作系统交互的智能体提供平台的环境。Android Env 库将根据定义的任务返回像素观察结果和奖励。

使用此环境，可以训练智能体使用 RL 与 Android Env 交互并完成任务。

- Github：https://github.com/google-deepmind/android_env

## DeepMind Lab

*DeepMind Lab*是一个通过 ioquake3 和其他开源软件基于 Quake III Arena 的 3D 学习环境。 DeepMind Lab 为学习智能体提供了一套具有挑战性的 3D 导航和解谜任务。其主要目的是充当人工智能研究的测试平台，智能体必须根据视觉观察采取行动。

Github: https://github.com/google-deepmind/lab

## Dm_Control

*dm_control*软件包是**Python 库和任务套件**的集合，用于关节体模拟中的强化学习智能体。 MuJoCo 包装器提供了与函数和数据结构的便捷绑定，以创建您自己的任务。

此外，控制套件是一组具有标准化结构的固定任务，旨在作为性能基准。它包括 HalfCheetah、Humanoid、Hopper、Walker、Graber 等经典任务（见图）。 Locomotion 框架提供了足球等运动任务的高级抽象和示例。还包括一组带有机器人手臂和拼装积木的可配置操作任务。

- Github: https://github.com/google-deepmind/dm_control

## MuJoCo

[MuJoCo](https://github.com/deepmind/mujoco)代表接触式多关节动力学。它由 DeepMind 发布。它是通用物理引擎，旨在促进机器人、生物力学、图形和动画、机器学习以及其他需要快速准确地模拟铰接结构与其环境相互作用的领域的研究和开发。

设计人体人工智能控制器以完成多样化的运动任务。该库由斯坦福大学实施，每年都会举办比赛，强化学习从业者可以在比赛中相互测试自己的技能。目前实施了三种环境：简化的手臂运动、学习跑步和腿部假肢。

- Github：https://github.com/google-deepmind/mujoco

- Pages: https://mujoco.org/

## OpenSim-RL

设计人体人工智能控制器以完成多样化的运动任务。该库由斯坦福大学实施，每年都会举办比赛，强化学习从业者可以在比赛中相互测试自己的技能。目前实施了三种环境：简化的手臂运动、学习跑步和腿部假肢。

- Github: https://github.com/stanfordnmbl/osim-rl

- 在线文档： http://osim-rl.stanford.edu/

## Realworldrl_Suite

《[现实世界 RL 的挑战》](https://arxiv.org/abs/1904.12901)论文识别并描述了目前阻碍强化学习 (RL) 智能体在现实世界应用程序和产品中使用的九个挑战。它还描述了一个评估框架和一组环境，可以评估 RL 算法对现实世界系统的潜在适用性。此后又发表了《对[现实世界强化学习挑战的实证研究》论文](https://arxiv.org/pdf/2003.11881.pdf)，该论文实现了所描述的九个挑战中的八个，并分析了它们对各种最先进的强化学习算法的影响。
这是用于执行此分析的代码库，也旨在作为一个通用平台，用于围绕这些挑战轻松重现实验。它被称为*realworldrl-suite*（真实世界强化学习（RWRL）套件）。

Github: https://github.com/google-research/realworldrl_suite

## PyBullet Gym

*PyBullet Gymperium 是 OpenAI Gym MuJoCo 环境的开源实现，这些都是具有挑战性的连续控制环境，例如训练人形机器人行走。可与 OpenAI Gym 强化学习研究平台一起使用，以支持开放研究。*

- Github： https://github.com/benelot/pybullet-gym

## FinRL

[FinRL](https://github.com/AI4Finance-Foundation/FinRL)是最早的金融开源 RL 环境之一。它可以集成到各种数据源中，无论是数据源还是用户导入的数据集，甚至是模拟数据。

- Github: https://github.com/AI4Finance-Foundation/FinRL

## RecSim

[RecSim](https://github.com/google-research/recsim)是一个可配置的系统，用于将强化学习应用于推荐系统。 RecSim 模拟推荐智能体与由用户模型、文档模型和用户 选择 模型组成的环境的交互。

- Github: https://github.com/google-research/recsim

## RLCard

*RLCard*是纸牌游戏中强化学习 (RL) 的工具包。它支持多种卡环境并具有易于使用的界面。游戏包括二十一点、UNO、限注德州扑克、斗地主等等！它还允许您创建自己的环境。 RLCard 的目标是架起强化学习和不完美信息博弈的桥梁。

[RLCard](https://github.com/datamllab/rlcard)是支持各种纸牌游戏的 RL 环境。它由莱斯大学和德克萨斯农工大学的[DATA 实验室](http://faculty.cs.tamu.edu/xiahu/)以及社区贡献者开发。它支持多种游戏和算法，如下所示:

Some of the games supported :

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

GIthub：https://github.com/datamllab/rlcard

Pages: http://www.rlcard.org/

## OpenSpiel

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

- GitHub：https://github.com/deepmind/open_spiel

## Deep RTS

DeepRTS 是一款用于强化学习研究的高性能实时策略游戏。它是为了提高性能而用 C++ 编写的，提供了一个 python 接口，以便更好地与机器学习工具包交互。深度 RTS 可以每秒超过**6,000,000**步处理游戏，渲染图形时可以达到**2,000,000**步。与《星际争霸》等其他解决方案相比，在配备 Nvidia RTX 2080 TI 的英特尔 i7-8700k 上运行的**模拟时间快了 15 000%以上。**

Deep RTS 的目标是通过减少计算时间，为 RTS AI 研究带来更实惠、更可持续的解决方案。

- Github: https://github.com/cair/deep-rts

## MicroRTS

[Gym-μRTS](https://github.com/Farama-Foundation/gym-microrts)在客观上与 Deep RTS 类似。现有的完整游戏具有很高的计算成本，通常意味着数千小时的 CPU 和 GPU 时间。这对研究来说是一个巨大的限制。

- Github: https://github.com/Farama-Foundation/MicroRTS-Py

## Nocturne

[Nocturne](https://github.com/facebookresearch/nocturne)是 Facebook Research 发布的用于模拟多智能体系统的 RL 环境。 Nocturne 是具有部分可观察性的 2D 驾驶模拟环境。它是用 C++ 构建的，以提高速度。目前，Nocturne 可以处理来自[Waymo 开放数据集的](https://github.com/waymo-research/waymo-open-dataset)数据，但可以对其进行修改以处理其他数据源。使用 Python 库`nocturne`，人们能够训练自动驾驶汽车控制器来解决 Waymo 数据集（我们提供的基准）中的各种任务，然后使用我们提供的工具来评估设计的控制器。

利用这种丰富的数据源，Nocturne包含了广泛的场景，其解决方案需要形成复杂的协调、心理理论和处理部分可观察性。

- GIthub: https://github.com/facebookresearch/nocturne

## CityFlow (交通)

[CityFlow](https://github.com/cityflow-project/CityFlow/)是一个针对大规模城市交通场景的多智能体强化学习环境。

它具有以下特点

- 微观交通模拟器，模拟每辆车的行为，提供交通演变的最高级别细节。
- 支持路网和交通流的灵活定义
- 为强化学习提供友好的Python接口
- 快速地！精心设计的数据结构和多线程模拟算法。能够模拟全市交通。查看与 SUMO 的性能比较

- Github: https://github.com/cityflow-project/CityFlow/

- Pages: https://cityflow.readthedocs.io/en/latest/

## **AirSim（自动驾驶汽车）**

AirSim 结合了强化学习、深度学习和计算机视觉的力量，用于构建用于自动驾驶车辆的算法。它模拟无人机、汽车等自动驾驶车辆。AirSim 是一个由虚幻引擎环境开发的开源平台，可与 Unity 插件一起使用，其API 可通过 C++、C#、Python 和 Java 访问。另一个有趣的事情是它与 PX4 等硬件飞行控制器兼容，可提供逼真的物理和虚拟体验。

Github: https://github.com/microsoft/AirSim

## CyberBattleSim

[CyberBattleSim](https://github.com/microsoft/CyberBattleSim)是一个框架，用于构建复杂计算机网络和系统的抽象模拟，以便应用强化学习。了解智能体如何在这种复杂的环境中交互和发展是很有帮助的。它本质上是网络安全的游戏化，以便可以应用强化学习。

CyberBattleSim 是一个实验研究平台，用于研究在模拟抽象企业网络环境中运行的自动化智能体的交互。该模拟提供了计算机网络和网络安全概念的高级抽象。其基于 Python 的 Open AI Gym 界面允许使用强化学习算法来训练自动化智能体。

模拟环境由固定的网络拓扑和一组漏洞参数化，智能体可以利用这些漏洞在网络中横向移动。攻击者的目标是通过利用计算机节点中植入的漏洞来获得部分网络的所有权。当攻击者试图在整个网络中传播时，防御者智能体会监视网络活动并尝试检测正在发生的任何攻击，并通过驱逐攻击者来减轻对系统的影响。我们提供了一个基本的随机防御者，可以根据预定义的成功概率来检测和减轻正在进行的攻击。我们通过重新镜像受感染的节点来实施缓解措施，该过程被抽象建模为跨越多个模拟步骤的操作。

为了比较智能体的性能，我们查看两个指标：为实现目标而采取的模拟步骤数量以及整个训练时期模拟步骤的累积奖励。

- Github： https://github.com/microsoft/CyberBattleSim

## Textworld

TextWorld是微软构建的开源引擎，有利于生成和模拟文本游戏。借助强化学习，我们可以训练智能体学习语言理解和基础知识以及决策能力。

- Github: https://github.com/microsoft/TextWorld

## TensorTrade：通过强化学习进行高效交易

TensorTrade 是一个开源 Python 框架，用于使用强化学习构建、训练、评估和部署强大的交易算法。该框架注重高度可组合性和可扩展性，使系统能够从单个 CPU 上的简单交易策略扩展到在 HPC 机器分布上运行的复杂投资策略。

在底层，该框架使用现有机器学习库中的许多 API 来维护高质量的数据管道和学习模型。

## Habitat-Lab

Habitat-Lab 是一个模块化高级库，用于具体人工智能的端到端开发。它旨在训练智能体在室内环境中执行各种具体的人工智能任务，并开发可以在执行这些任务时与人类交互的智能体。

为了实现这一目标，Habitat-Lab 旨在支持以下功能：

- **灵活的任务定义**：允许用户在各种单智能体和多智能体任务中训练智能体（例如导航、重新排列、遵循指令、回答问题、人类跟随），以及定义新颖的任务。
- **多样化的体现智能体**：配置和实例化一组多样化的体现智能体，包括商业机器人和人形机器人，指定它们的传感器和功能。
- **训练和评估智能体**：提供用于单智能体和多智能体训练的算法（通过模仿或强化学习，或者像 SensePlanAct 管道中那样根本不学习），以及使用标准指标在定义的任务上对其性能进行基准测试的工具。
- **人机交互**：为人类与模拟器交互提供框架，从而能够收集具体数据或与训练有素的智能体交互。

Habitat-Lab用作[`Habitat-Sim`](https://github.com/facebookresearch/habitat-sim)核心模拟器。

- Github: https://github.com/facebookresearch/habitat-lab
- 文档资源： https://aihabitat.org/

##  `bsuite` 强化学习行为套件

*bsuite*是精心设计的实验的集合，旨在研究强化学习 (RL) 智能体的核心功能，有两个主要目标。

- 收集清晰、信息丰富且可扩展的问题，捕获高效和通用学习算法设计中的关键问题。
- 通过智能体在这些共享基准上的表现来研究智能体的行为。

该库可自动根据这些基准对任何智能体进行评估和分析。它有助于促进对 RL 核心问题的可重复且可访问的研究，并最终设计出卓越的学习算法。

- Github: https://github.com/google-deepmind/bsuite

## Google Research Football

*Google Research Football*是一个新颖的 RL 环境，智能体的目标是掌握世界上最受欢迎的运动——足球！足球环境以流行的足球视频游戏为蓝本，提供了一种高效的基于物理的 3D 足球模拟，其中智能体控制球队中的一名或所有足球运动员，学习如何在他们之间传球，并设法克服对手的防守以得分目标。足球环境提供了一组要求很高的研究问题，称为足球基准，以及足球学院，一组逐渐困难的强化学习场景。
它非常适合多主体和多任务研究。它还允许您根据所包含的示例使用模拟器创建自己的学院场景以及全新的任务。

- Github: https://github.com/google-research/football

## Metaworld

元强化学习算法可以通过利用先前的经验来学习如何学习，从而使机器人能够更快地获得新技能。*Meta-World*是元强化学习和多任务学习的开源模拟基准，由 50 个不同的机器人操作任务组成。作者的目标是提供足够广泛的任务分布来评估元强化学习算法对新行为的泛化能力。

- Github: https://github.com/Farama-Foundation/Metaworld

## MineRL

*MineRL*是卡内基梅隆大学发起的一个研究项目，旨在开发 Minecraft 中人工智能的各个方面。简而言之，MineRL 由两个主要组件组成：

- [MineRL-v0 数据集](https://minerl.io/dataset/)– 最大的模仿学习数据集之一，拥有超过 6000 万帧记录的人类玩家数据。该数据集包含一组环境，突出了现代强化学习中许多最困难的问题：稀疏奖励和分层策略。
- [minerl](https://minerl.io/docs/tutorials/index.html) – 一个丰富的 python3 包，用于在 Minecraft 中进行人工智能研究。这包括两个主要子模块：*minerl.env* - Minecraft 中不断增长的 OpenAI Gym 环境集和*minerl.data* - 用于试验 MineRL-v0 数据集的主要 Python 模块。

- Github: https://github.com/minerllabs
- 文档列表： https://minerl.readthedocs.io/en/latest/

## Procgen Benchmark

*Procgen Benchmark*由 16 个独特的环境组成，旨在衡量强化学习中的样本效率和泛化能力。该基准非常适合评估泛化性，因为可以在每个环境中生成不同的训练和测试集。该基准测试也非常适合评估样本效率，因为所有环境都给 RL 智能体带来了多样化且引人注目的挑战。环境的内在多样性要求智能体学习稳健的策略；过度拟合状态空间中的狭窄区域是不够的。换句话说，当智能体面临不断变化的水平时，概括能力就成为成功不可或缺的组成部分。

- Github： https://github.com/openai/procgen
- 文档： https://openai.com/blog/procgen-benchmark/

## 星际争霸 2

*PySC2*为 RL 智能体提供了一个与《星际争霸 2》交互的接口，获取观察结果并发送操作。它将暴雪娱乐的星际争霸 II 机器学习 API 作为 Python RL 环境公开。这是 DeepMind 和暴雪之间的合作，旨在将《星际争霸 II》开发成一个丰富的 RL 研究环境。*PySC2*有许多预先配置的迷你游戏地图，用于对 RL 智能体进行基准测试。

- Github: https://github.com/google-deepmind/pysc2

## Unity 机器学习智能体工具包 (ML-Agents）

**Unity 机器学习智能体工具包**(ML-Agents) 是一个开源项目，使游戏和模拟能够作为训练智能体的环境。我们提供最先进算法的实现（基于 PyTorch），使游戏开发者和爱好者能够轻松训练 2D、3D 和 VR/AR 游戏的智能体。研究人员还可以使用提供的简单易用的 Python API 来使用强化学习、模仿学习、神经进化或任何其他方法来训练智能体。这些训练有素的智能体可用于多种目的，包括控制 NPC 行为（在多种设置中，例如多智能体和对抗）、游戏构建的自动测试以及评估预发布的不同游戏设计决策。 ML-Agents 工具包对于游戏开发人员和 AI 研究人员来说是互惠互利的，因为它提供了一个中央平台，可以在 Unity 丰富的环境中评估 AI 的进步，然后让更广泛的研究和游戏开发人员社区可以访问。

- Github: https://github.com/Unity-Technologies/ml-agents
- 文档：https://unity.com/products/machine-learning-agents



# 仿真引擎加速

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

## Madrona

Madrona 是一款研究型游戏引擎，专为创建学习环境而设计，可以在单个 GPU 上同时运行数千个环境实例，并且以极高的吞吐量（每秒数百万个聚合步骤）执行。Madrona 的目标是让研究人员更轻松地为各种任务创建新的高性能环境，从而使 AI 智能体训练的速度提高几个数量级。

Madrona 具有以下特点：

- GPU 批量模拟：单个 GPU 上可运行数千个环境；
- 实体组件系统 (ECS) 架构；
- 可与 PyTorch 轻松互操作。

Github: https://github.com/shacklettbp/madrona

文档： https://madrona-engine.github.io/

# 其他资源链接

| 分类     | Introduction                           | Github                                                       | Star |
| -------- | -------------------------------------- | ------------------------------------------------------------ | ---- |
| 网络攻防 | 将强化学习应用于网络安全的精选资源列表 | [Limmen/awesome-rl-for-cybersecurity](https://github.com/Limmen/awesome-rl-for-cybersecurity) | 600+ |
| 通用     | 强化学习环境的综合列表                 | [clvrai/awesome-rl-envs](https://github.com/clvrai/awesome-rl-envs) | 900+ |
|          |                                        |                                                              |      |
