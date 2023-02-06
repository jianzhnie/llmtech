# 多智能体学习环境

这篇博文概述了一系列多智能体强化学习 (MARL) 环境及其主要特性和学习挑战。我们在下表中列出了环境和属性，并提供了指向本博文中相应部分的快速链接。在这篇文章的最后，我们还提到了一些支持各种环境和游戏模式的 [通用框架。](https://agents.inf.ed.ac.uk/blog/multiagent-learning-environments/#libraries)

| Env                                                          | Type          | Observations        | Actions        | Observability |
| ------------------------------------------------------------ | ------------- | ------------------- | -------------- | ------------- |
| [LBF](https://agents.inf.ed.ac.uk/blog/multiagent-learning-environments/#lbf) | Mixed         | Discrete            | Discrete       | Both          |
| [PressurePlate](https://agents.inf.ed.ac.uk/blog/multiagent-learning-environments/#pressureplate) | Collaborative | Discrete            | Discrete       | Partial       |
| [RWARE](https://agents.inf.ed.ac.uk/blog/multiagent-learning-environments/#rware) | Collaborative | Discrete            | Discrete       | Partial       |
| [MPE](https://agents.inf.ed.ac.uk/blog/multiagent-learning-environments/#mpe) | Mixed         | Continuous          | Both           | Both          |
| [SMAC](https://agents.inf.ed.ac.uk/blog/multiagent-learning-environments/#smac) | Cooperative   | Continuous          | Discrete       | Partial       |
| [MALMO](https://agents.inf.ed.ac.uk/blog/multiagent-learning-environments/#malmo) | Mixed         | Continuous (Pixels) | Discrete       | Partial       |
| [Pommerman](https://agents.inf.ed.ac.uk/blog/multiagent-learning-environments/#pommerman) | Competitive   | Continuous (Pixels) | Discrete       | Both          |
| [DM Lab](https://agents.inf.ed.ac.uk/blog/multiagent-learning-environments/#dmlab) | Mixed         | Continuous (Pixels) | Discrete       | Partial       |
| [DM Lab2D](https://agents.inf.ed.ac.uk/blog/multiagent-learning-environments/#dmlab2d) | Mixed         | Discrete            | Discrete       | Partial       |
| [Derk's Gym](https://agents.inf.ed.ac.uk/blog/multiagent-learning-environments/#derkgym) | Competitive   | Discrete            | Mixed          | Partial       |
| [Flatland](https://agents.inf.ed.ac.uk/blog/multiagent-learning-environments/#flatland) | Collaborative | Continuous          | Discrete       | Both          |
| [Hanabi](https://agents.inf.ed.ac.uk/blog/multiagent-learning-environments/#hanabi) | Cooperative   | Discrete            | Discrete       | Partial       |
| [Neural MMO](https://agents.inf.ed.ac.uk/blog/multiagent-learning-environments/#neuralmmo) | Competitive   | Discrete            | Multi-Discrete | Partial       |

我们使用术语“任务”来指代环境的特定配置（例如设置特定的 world size、智能体数量等），例如我们在[SEAC](https://arxiv.org/abs/2006.07169) [5] 和[ [MARL benchmark](https://arxiv.org/abs/2006.07869)论文中所做的。

- 如果所有智能体在每个时间步都收到相同的奖励，我们就说任务是“合作的”。
- 如果智能体之间存在某种形式的竞争，即一个智能体的收益是另一个智能体的损失，则该任务是“竞争性的”。
- 如果智能体的最终目标是一致的并且智能体合作，但他们收到的奖励不相同，我们将松散地称为“协作”任务。

基于这些任务/类型定义，如果环境仅支持属于这些相应类型类别之一的任务，我们就说该环境是合作的、竞争的或协作的。如果环境支持不止一种类型的任务，我们将其称为“混合”环境。

对于观察，我们区分离散特征向量、连续特征向量和图像观察的连续（像素）。对于动作，我们区分离散动作、智能体在每个时间步选择多个（单独的）离散动作的多离散动作，以及连续动作。如果环境支持离散和连续动作，则动作空间为“Both”。

## Level-Based Foraging

| Type  | Observations | Actions  | Code                                                     | Papers  |
| ----- | ------------ | -------- | -------------------------------------------------------- | ------- |
| Mixed | Discrete     | Discrete | [Environment](https://github.com/uoe-agents/lb-foraging) | [5, 16] |

### 一般说明

Level-Based Foraging 环境由混合的合作竞争任务组成，重点是相关智能体的协调。每个智能体的任务是在网格世界地图中导航并收集物品。每个智能体和商品都分配了一个级别，商品随机散布在环境中。为了收集物品，智能体必须选择物品旁边的特定动作。但是，只有当所涉及的智能体级别的总和等于或大于商品级别时，此类收集才会成功。特工收到的奖励等于收集物品的等级。下面，您可以看到一组可能任务的可视化。默认情况下，每个智能体都可以观察整个地图，包括所有实体的位置和级别，并且可以选择通过向四个方向之一移动或尝试加载商品来采取行动。在部分可观察的版本中，用“sight=2”表示，智能体只能观察周围 5 × 5 网格中的实体。根据任务的不同，奖励相当稀少，因为智能体可能必须合作（在同一时间步长拾取相同的食物）才能获得奖励。

有关更多详细信息，请在[此处](https://agents.inf.ed.ac.uk/blog/new-environments-algorithm-multiagent-rl/) 查看我们的博客文章。

![img](https://agents.inf.ed.ac.uk/blog/multiagent-learning-environments/img/lbf/lbf-8x8-2p-3f.png)

(a) LBF-8x8-2p-3f 示意图

![img](https://agents.inf.ed.ac.uk/blog/multiagent-learning-environments/img/lbf/lbf-8x8-2p-2f-coop.png)

(b) LBF-8x8-2p-2f-coop 示意图

![img](https://agents.inf.ed.ac.uk/blog/multiagent-learning-environments/img/lbf/lbf-8x8-3p-1f-coop.png)

(c) LBF-8x8-3p-1f-coop 示意图

![img](https://agents.inf.ed.ac.uk/blog/multiagent-learning-environments/img/lbf/lbf-10x10-2p-8f.png)

(d) LBF-10x10-2p-8f 示意图

### 示例任务

**LBF-8x8-2p-3f：**一个8×8 具有两个智能体和三个商品的网格世界放置在随机位置。商品级别是随机的，可能需要智能体合作，具体取决于级别。

**LBF-8x8-2p-2f-coop：**一个8×8具有两个智能体和两个商品的网格世界。这是一个合作版本，智能体总是需要同时收集一个商品（合作）。

**LBF-8x8-3p-1f-coop：**一个8×8具有三个智能体和一个商品的网格世界。这是一个合作版本，所有三个智能体都需要同时收集商品。

**LBF-10x10-2p-8f：** 一个 10×10具有两个智能体和十个商品的网格世界。时间限制（25 个时间步长）通常不足以收集所有商品。因此，智能体商需要在短时间内分散并收集尽可能多的物品。

**LBF-8x8-2p-3f, sight=2:** 类似于第一个变体，但部分可观察到。智能体的视野仅限于5个×5个以智能体为中心的框。

## PressurePlate

![img](https://agents.inf.ed.ac.uk/blog/multiagent-learning-environments/img/pressure_plate_4p.gif)

具有 4 个智能体的 PressurePlate 线性任务的可视化

| Type          | Observations | Actions  | Code                                                       | Papers |
| ------------- | ------------ | -------- | ---------------------------------------------------------- | ------ |
| Collaborative | Discrete     | Discrete | [Environment](https://github.com/uoe-agents/pressureplate) | /      |

### 一般说明

[PressurePlate 是一个基于Level-Based Foraging](https://agents.inf.ed.ac.uk/blog/multiagent-learning-environments/#lbf)环境 的多智能体环境，需要智能体在网格世界的遍历过程中进行合作。网格被分成一系列相连的房间，每个房间都有一个盘子和一个封闭的门口。在一集开始时，每个特工都分配了一个板块，只有他们可以通过移动到它的位置并停留在它的位置来激活它。激活压力板将打开通往下一个房间的门。因此，智能体必须沿着房间的顺序移动，并且在每个房间内，分配给其压力板的智能体需要留在后面，激活压力板，以允许智能体组进入下一个房间。当达到目标（用宝箱表示）时，任务被视为已完成。

在这种环境中，智能体观察以其位置为中心的网格，观察到的网格的大小被参数化。观察到的 2D 网格有几层，以二进制 2D 阵列的形式指示智能体、墙壁、门、板和目标位置的位置。智能体将这些二维网格作为扁平化矢量连同它们的 x 和 y 坐标一起接收。行动空间与基于水平的觅食相同，每个主要方向都有行动和一个无操作（什么都不做）行动。PressurePlate 任务中的奖励是密集的，指示智能体位置与其分配的压力板之间的距离。智能体需要合作但会收到个人奖励，从而使 PressurePlate 任务具有协作性。目前，支持具有四到六个智能体的三个 PressurePlate 任务，房间按线性顺序构建。

有关详细信息，请参阅 [Github ](https://github.com/uoe-agents/pressureplate)中的文档。

## Multi-Robot Warehouse

| Type          | Observations | Actions  | Code                                                         | Papers  |
| ------------- | ------------ | -------- | ------------------------------------------------------------ | ------- |
| Collaborative | Discrete     | Discrete | [Environment](https://github.com/uoe-agents/robotic-warehouse) | [5, 16] |

![img](https://agents.inf.ed.ac.uk/blog/multiagent-learning-environments/img/rware/rware-tiny.png)

(a) RWARE 微型图，两个智能体

![img](https://agents.inf.ed.ac.uk/blog/multiagent-learning-environments/img/rware/rware-small.png)

(b) RWARE 小体积，两个智能体的图示

![img](https://agents.inf.ed.ac.uk/blog/multiagent-learning-environments/img/rware/rware-medium.png)

(c) RWARE 中型图解，四个智能体

### 一般说明

多机器人仓库环境模拟一个仓库，机器人移动和交付请求的货物。在实际应用中 [23]，机器人拾取货架并将它们运送到工作站。人类评估货架上的物品，然后机器人可以将它们放回空货架位置。在这个环境模拟中，智能体控制机器人，每个智能体的动作空间是

A = {左转、右转、前进、装载/卸载货架}

当智能体不携带任何东西时，他们可以在架子下方移动，但当他们携带架子时，智能体必须使用中间的走廊（见上图）。智能体的观察包括3个×3个正方形以智能体为中心。它包含有关周围智能体（位置/旋转）和货架的信息。每次货架的数量$R$被要求是固定的。当请求的货架被带到目标位置时，另一个当前未请求的货架被统一采样并添加到当前请求中。智能体商因成功将请求的货架交付到目标位置而获得奖励，奖励为 1。此环境中的一个主要挑战是智能体商交付请求的货架，但随后还要找到一个空货架位置以返回先前交付的货架。智能体商需要放下之前交付的货架才能拿起新货架。这导致非常稀疏的奖励信号。由于这是一项协作任务，我们使用所有智能体的未折现回报之和作为绩效指标。多机器人仓库任务参数化为：

- 预设为 tiny 的仓库大小10×11， 小的10×20， 中等的16×20, 或大16×29.
- 智能体商数量 N.
- 请求的货架数量$R$. 默认情况下$R$=否，但环境使用的简单和困难的变化下$R$=2N和 $R$=N/2。

[有关更多详细信息，请在此处](https://agents.inf.ed.ac.uk/blog/new-environments-algorithm-multiagent-rl/) 查看我们的博客文章。

## Multi-Agent Particle Environment

| Type  | Observations | Actions              | Code                                                         | Papers          |
| ----- | ------------ | -------------------- | ------------------------------------------------------------ | --------------- |
| Mixed | Continuous   | Continuous/ Discrete | [Environment](https://github.com/openai/multiagent-particle-envs) | [15, 12, 7, 20] |

### 一般说明

该环境包含一组不同的二维任务，涉及智能体之间的合作和竞争。在所有任务中，粒子（代表智能体）与地标和其他智能体交互以实现各种目标。观察结果由高级特征向量组成，其中包含与其他智能体和地标的相对距离，有时还包含其他信息，例如通信或速度。所有任务和智能体之间的动作空间是离散的，通常包括五种可能的动作，分别对应于在某些任务中不移动、向右移动、向左移动、向上移动或向下移动以及额外的通信动作。但是，也有使用连续动作空间的选项（但是我知道的所有出版物都使用离散动作空间）。这些任务中的奖励信号很密集，任务范围从完全合作到竞争和基于团队的场景。大多数任务由 Lowe 等人定义。[12] Iqbal 和 Sha [7] 引入了额外的任务（可用代码[here](https://github.com/shariqiqbal2810/MAAC) ) 和部分可观察到的变化定义为我的 MSc 论文 [20] 的一部分（代码可[在此处](https://github.com/LukasSchaefer/multiagent-particle-envs)获得）。修改现有任务甚至在需要时创建全新任务都相对简单。

### 常见任务 

**MPE Speaker-Listener [12]：**在这项完全合作的任务中，一个静态说话者智能体必须将目标地标传达给能够移动的收听智能体。环境中总共有三个地标，两个智能体都得到了侦听器智能体到目标地标的负欧几里德距离的奖励。说话者智能体只观察目标地标的颜色。同时，听者智能体接收其速度、与每个地标的相对位置以及说话者智能体的通信作为其观察。说话者智能体在三个可能的离散通信动作之间进行选择，而听者智能体则遵循 MPE 任务的典型五个离散移动智能体。

**MPE Spread [12]：**在这个完全合作的任务中，三个智能体被训练移动到三个地标，同时避免相互碰撞。所有智能体都会收到他们的速度、位置、相对于所有其他智能体和地标的位置。每个智能体的动作空间包含五个离散的运动动作。智能体将获得从每个地标到任何智能体的负最小距离之和的奖励，并添加一个额外的术语来惩罚智能体之间的碰撞。

**MPE对手[12]：**在这项竞争性任务中，两个合作智能体与第三个对手智能体竞争。有两个地标，从中随机选择一个作为目标地标。合作智能体接收他们与目标的相对位置以及与所有其他智能体和地标的相对位置作为观察值。然而，对手智能体观察所有相对位置而没有收到有关目标地标的信息。所有智能体都有五个离散的运动动作。智能体会得到到目标的负最小距离的奖励，而合作智能体会因为敌方智能体到目标地标的距离而获得额外奖励。因此，合作智能体必须移动到两个地标，以避免对手识别哪个地标是目标并达到目标。

**MPE Predator-Prey [12]：**在这项竞争性任务中，三个合作的捕食者追捕控制更快猎物的第四个智能体。在环境中放置两个障碍物作为障碍物。所有智能体都接收自己的速度和位置以及与所有其他地标和智能体的相对位置作为观察结果。捕食者智能体还会观察猎物的速度。所有智能体都在五种运动动作中进行选择。控制猎物的智能体会因与捕食者的任何碰撞以及离开可观察的环境区域而受到惩罚（以防止它简单地逃跑但学会逃避）。捕食者智能体因与猎物发生碰撞而获得集体奖励。

**MPE Multi Speaker-Listener [7]：**此协作任务由 [7]（其中也称为“Rover-Tower”）引入，包括八个智能体。四个智能体代表漫游者，而其余四个智能体代表塔。在每一集中，漫游者和塔智能体随机配对，并为每个漫游者设置一个目标目的地。流动站智能体可以在环境中移动，但不观察周围环境，塔台智能体会观察所有流动站智能体的位置及其目的地。塔台智能体可以在每个时间步向配对的流动站发送五个离散通信消息之一，以引导配对的流动站到达目的地。流动站智能体选择两个连续的动作值来表示它们在两个运动轴上的加速度。每对漫游者和塔智能体都会根据漫游者与其目标的距离获得负奖励。

**MPE宝藏[7]：**此协作任务由 [7] 引入，包括代表寻宝者的六个智能体，而另外两个智能体代表宝库。狩猎智能体会收集随机生成的带有颜色编码的宝藏。根据宝物的颜色，要送到相应的宝库。所有智能体观察所有其他智能体的相对位置和速度以及宝藏的相对位置和颜色。狩猎智能体还会收到自己的位置和速度作为观察值。所有智能体都有连续的动作空间，可以选择它们在两个轴上的加速度来移动。智能体因正确存放和收集宝物而获得奖励。然而，该任务并非完全合作，因为每个智能体还会收到进一步的奖励信号。每个狩猎智能体因与其他猎人智能体的碰撞而受到额外惩罚，并根据智能体是否已经持有宝藏而获得等于与最近的相关宝库或宝藏的负距离的奖励。宝藏银行根据到最近的携带相应颜色宝藏的狩猎智能体的负距离和到任何猎人智能体的负平均距离进一步受到惩罚。

## StarCraft Multi-Agent Challenge

| Type              | Observations | Actions  | Code                                                         | Papers |
| ----------------- | ------------ | -------- | ------------------------------------------------------------ | ------ |
| Fully-cooperative | Continuous   | Discrete | [Environment](https://github.com/oxwhirl/smac) and [codebase](https://github.com/oxwhirl/pymarl) | [19]   |

### 一般说明

星际争霸多智能体挑战赛是一组完全协作、部分可观察的多智能体任务。该环境基于流行的实时战略游戏星际争霸 II 实施各种微观管理任务，并利用星际争霸 II 学习环境 (SC2LE) [22]。每个任务都是一个特定的战斗场景，其中一队特工，每个特工控制一个单独的单位，与星际争霸游戏的集中内置游戏 AI 控制的军队作战。这些任务要求智能体学习精确的动作序列以启用风筝等技能，并协调它们的动作以将注意力集中在特定的敌方单位上。许多任务在结构上是对称的，即两支军队都由相同的单位建造。以下，您可以在此环境中找到每个考虑的任务的可视化。奖励密集，任务难度从（相对）简单到非常困难的任务不等。通过智能体的可见半径，所有任务自然包含部分可观察性。

![img](https://agents.inf.ed.ac.uk/blog/multiagent-learning-environments/img/smac/smac_3m.jpg)

(a) SMAC 3m 示意图

![img](https://agents.inf.ed.ac.uk/blog/multiagent-learning-environments/img/smac/smac_8m_small.jpg)

(b) SMAC 8m 示意图

![img](https://agents.inf.ed.ac.uk/blog/multiagent-learning-environments/img/smac/smac_3s5z.jpg)

(c) SMAC 3s5z 示意图

![img](https://agents.inf.ed.ac.uk/blog/multiagent-learning-environments/img/smac/smac_2s3z.jpg)

(d) SMAC 2s3z 示意图

![img](https://agents.inf.ed.ac.uk/blog/multiagent-learning-environments/img/smac/smac_1c3s5z.jpg)

(e) SMAC 1c3s5z 示意图

### 示例任务

**SMAC 3m：**在这种情况下，每支队伍由三名星际战士组成。必须控制这些远程单位一次将火力集中在一个敌方单位上并集体攻击才能赢得这场战斗。

**SMAC 8m：**在这种情况下，每个团队控制八名星际战士。虽然总体策略与 3m 场景相同，但由于智能体和由智能体控制的海军陆战队员数量增加，协调变得更具挑战性。

**SMAC 2s3z：**在这种情况下，每个团队控制两个追猎者和三个狂热者。追猎者是远程单位，而狂热者是近战单位，即他们需要靠近敌方单位进行攻击。因此，受控单位仍然必须学会一次将火力集中在单个敌方单位上。此外，追猎者需要学习放风筝，以便在两次攻击之间始终向后移动，以保持自己与敌方狂热者之间的距离，从而最大限度地减少受到的伤害，同时保持高伤害输出。

**SMAC 3s5z：**此场景需要与 2s3z 任务相同的策略。两支队伍都控制着三个追猎者和五个狂热者单位。由于智能体数量的增加，任务变得更具挑战性。

**SMAC 1c3s5z：**在这种情况下，除了三个追猎者和五个狂热者之外，两支队伍都控制着一个巨人。巨像是一种持久的单位，具有远程、分散攻击。它的攻击可以一次击中多个敌方单位。所以现在被控制的队伍要协调好，避免多个单位被敌方巨像一个个击中，同时让自己的巨像一起击中多个敌人。

## MALMO

| Type  | Observations        | Actions  | Code                                              | Papers |
| ----- | ------------------- | -------- | ------------------------------------------------- | ------ |
| Mixed | Continuous (Pixels) | Discrete | [Environment](https://github.com/microsoft/malmo) | [9]    |

![img](https://agents.inf.ed.ac.uk/blog/multiagent-learning-environments/img/malmo.jpg)

MALMO任务可视化

### 一般说明

MALMO 平台 [9] 是一个基于游戏 Minecraft 的环境。它的 3D 世界包含非常多样化的任务和环境。智能体以多种方式与其他智能体、实体和环境交互。任务可以包含部分可观察性，并且可以使用提供的配置器创建，并且默认情况下是部分可观察的，因为智能体从他们的角度将环境视为像素。有关此环境的更多信息，请参阅[官方网页](https://www.microsoft.com/en-us/research/project/project-malmo/)、[文档](http://microsoft.github.io/malmo/0.30.0/Documentation/index.html)、[官方博客](http://microsoft.github.io/malmo/blog/)和公共[教程](http://microsoft.github.io/malmo/0.14.0/Python_Examples/Tutorial.pdf)或查看以下[幻灯片](https://www.slideshare.net/hironojumpei/malmotutorial)。有关如何安装 MALMO（适用于 Ubuntu 20.04）的说明以及测试 MALMO 多智能体任务的简短脚本，请参阅[这篇文章底部的脚本](https://agents.inf.ed.ac.uk/blog/multiagent-learning-environments/#malmo_scripts)。作为 NeurIPS 2018 研讨会的一部分，可以从马尔默 (MARLÖ) 竞赛 [17] 的多智能体强化学习中找到更多任务。此挑战的代码可在[MARLO github ](https://github.com/crowdAI/marLo)中找到，并提供[更多文档](https://marlo.readthedocs.io/en/latest/index.html)。MALMO 环境中具有更多任务的另一个挑战是 The [Malmo Collaborative AI Challenge](https://www.microsoft.com/en-us/research/academic-program/collaborative-ai-challenge/)，其代码和任务可[在此处](https://github.com/microsoft/malmo-challenge)获取。但是，我不确定运行这些环境中的每一个所需的兼容性和版本

最近，创建了一个[新的仓库](https://github.com/GAIGResearch/malmo)，其中包含简化的启动脚本、设置过程和示例 IPython 笔记本。我建议看一下，让自己熟悉 MALMO 环境。

### Downsides of MALMO

我相信在这个环境的许多任务中展示的多样性使得它对 RL 和 MARL 研究非常有吸引力，并且能够（相对）轻松地以 XML 格式定义新任务（有关更多详细信息，请参见上面的文档和教程）。但是，该环境在上述挑战中包含的各种任务之间存在技术问题和兼容性困难。此外，对于每个智能体，必须启动一个单独的 Minecraft 实例以通过（默认为本地）网络连接。我发现智能体与环境的连接有时会崩溃，通常需要多次尝试才能启动任何运行。后者应该使用新中提供的新启动脚本进行简化。

## Pommerman

| Type        | Observations | Actions  | Code                                                         | Papers | Others                                |
| ----------- | ------------ | -------- | ------------------------------------------------------------ | ------ | ------------------------------------- |
| Competitive | Discrete     | Discrete | [Environment](https://github.com/MultiAgentLearning/playground) | [18]   | [Website](https://www.pommerman.com/) |

![img](https://agents.inf.ed.ac.uk/blog/multiagent-learning-environments/img/pommerman.jpg)

Pommerman 的可视化

### 一般说明

Pommerman 环境 [18] 基于游戏 Bomberman。它包含有竞争力的11×11网格世界任务和基于团队的竞争。智能体可以通过破坏地图中的墙壁以及攻击对手智能体来与彼此和环境互动。观察结果包括董事会状态11×11=121 onehot-encodings 表示网格世界中每个位置的状态。此外，每个特工都会收到有关其位置、弹药、队友、敌人和更多信息的信息。所有这些使得观察空间相当大，使得没有卷积处理（类似于图像输入）的学习变得困难。智能体在每个时间步选择六个离散动作之一：停止、向上移动、向左移动、向下移动、向右移动、放置炸弹、消息。观察和行动空间在整个任务中保持不变，部分可观察性可以打开或关闭。实施了盟友之间的沟通框架。该环境为竞争性 MARL 提供了一个有趣的环境，但其任务在体验上基本相同。虽然地图是随机的，但任务在目标和结构上是相同的。[项目网站](https://www.pommerman.com/about)。

## Deepmind Lab

| Name           | Type  | Observations        | Actions  | Code                                             | Papers | Others                                                       |
| -------------- | ----- | ------------------- | -------- | ------------------------------------------------ | ------ | ------------------------------------------------------------ |
| Deepmind Lab   | Mixed | Continuous (Pixels) | Discrete | [Environment](https://github.com/deepmind/lab)   | [3]    | [Blog](https://deepmind.com/blog/article/open-sourcing-deepmind-lab) |
| Deepmind Lab2D | Mixed | Discrete            | Discrete | [Environment](https://github.com/deepmind/lab2d) | [4]    | /                                                            |

![img](https://agents.inf.ed.ac.uk/blog/multiagent-learning-environments/img/dm_lab/dm_lab_envs.png)

DeepMind 实验室环境

![img](https://agents.inf.ed.ac.uk/blog/multiagent-learning-environments/img/dm_lab/dm_lab_observations.png)

(a) 观察空间

![img](https://agents.inf.ed.ac.uk/blog/multiagent-learning-environments/img/dm_lab/dm_lab_actions.png)

(b) 行动空间

![img](https://agents.inf.ed.ac.uk/blog/multiagent-learning-environments/img/dm_lab/dm_lab2d_scissors.png)

(c) 来自 [4]：Deepmind Lab2D 环境 - “Running with Scissors”示例

### 一般描述 - Deepmind Lab

DeepMind Lab [3] 是一个基于 Quake III Arena 的 3D 学习环境，具有大量不同的任务。任务示例包括集合 DMLab30 [6]（博客文章[在这里](https://deepmind.com/blog/article/impala-scalable-distributed-deeprl-dmlab-30)）和 PsychLab [11]（博客文章[在这里](https://deepmind.com/blog/article/open-sourcing-psychlab)），它们可以在游戏脚本/关卡/演示下找到，还有多个较小的问题。然而，目前不支持多智能体游戏（参见[Github 问题](https://github.com/deepmind/lab/issues/153)），尽管在例如 Capture-The-Flag [8] 中使用多个智能体的出版物。此外，设置结果比预期的更麻烦。[设置代码](https://agents.inf.ed.ac.uk/blog/multiagent-learning-environments/#dmlab_scripts)可以在帖子底部找到。

### 一般描述 - DeepMind Lab2D

最近，Deepmind 还发布了用于二维网格世界环境的 Deepmind Lab2D [4] 平台。这包含一个用于（也是多智能体）网格世界任务的生成器，其中已经定义了各种任务，并且自 [13] 以来已经添加了更多任务。[有关安装脚本](https://agents.inf.ed.ac.uk/blog/multiagent-learning-environments/#dmlab2d_scripts)，请参阅帖子底部。

## Derk's Gym

| Type        | Observations | Actions | Code                                                       | Papers | Others                               |
| ----------- | ------------ | ------- | ---------------------------------------------------------- | ------ | ------------------------------------ |
| Competitive | Discrete     | Mixed   | [Environment documentation](http://docs.gym.derkgame.com/) | /      | [Website](https://gym.derkgame.com/) |

![img](https://agents.inf.ed.ac.uk/blog/multiagent-learning-environments/img/derk_gym/gym_derk1.png)

![img](https://agents.inf.ed.ac.uk/blog/multiagent-learning-environments/img/derk_gym/gym_derk2.png)

### 一般说明

Derk's gym 是一款 MOBA 风格的多智能体竞技团队游戏。“两支队伍互相战斗，同时试图保卫自己的“雕像”。每支队伍由三个单位组成，每个单位随机加载。目标是尝试攻击对手的雕像和单位，同时保卫自己的雕像和单位.默认奖励是击杀敌方生物一分，击杀敌方雕像四分。智能体观察所有智能体的离散观察键（[此处](http://docs.gym.derkgame.com/#gym_derk.ObservationKeys)列出），并从 5 种具有离散或连续动作值的不同动作类型中进行选择（详情请参见[此处](http://docs.gym.derkgame.com/#gym_derk.ActionKeys)）。这种环境的主要卖点之一是它能够在 GPU 上快速运行。

有关入门概述和“入门工具包”的更多信息，[请参见此 AICrowd 的挑战页面](https://www.aicrowd.com/challenges/dr-derk-s-battleground)。

derk 健身房环境的缺点之一是它的许可模式。仅供个人使用的许可证是免费的，但学术许可证的费用为 5 美元/月（或 50 美元/月，可以访问源代码），商业许可证的价格更高。

## Flatland

| Type          | Observations | Actions  | Code                                                   | Papers |
| ------------- | ------------ | -------- | ------------------------------------------------------ | ------ |
| Collaborative | Mixed        | Discrete | [Environment](https://flatland.aicrowd.com/intro.html) | [14]   |

![img](https://agents.inf.ed.ac.uk/blog/multiagent-learning-environments/img/flatland/flatland_wide.jpg)

Visualisation of Flatland task

### 一般说明

这种多智能体环境基于协调瑞士联邦铁路 (SBB) 铁路交通基础设施的现实问题。Flatland 环境旨在通过提供网格世界环境并允许多种解决方案来模拟车辆重新调度问题。智能体代表铁路系统中的火车。观察有三种方案：全局、局部和树。在这些中，智能体观察 (1) 全局信息作为各种通道的 3D 状态数组（类似于图像输入），(2) 仅在类似结构的 3D 数组中的局部信息或 (3) 基于图形的铁路编码系统及其当前状态（有关详细信息，请参阅[相应文档](https://flatland.aicrowd.com/getting-started/env/observations.html)). 智能体可以从 5 个离散动作中选择一个：什么都不做、向左移动、向前移动、向右移动、停止移动（[此处](https://flatland.aicrowd.com/getting-started/env.html#actions)有更多详细信息）。智能体收到两个奖励信号：全局奖励（所有智能体共享）和本地智能体特定奖励。

我强烈建议在其优秀的网页上查看环境的文档。可以在[此处](https://flatland.aicrowd.com/getting-started/rl/multi-agent.html)找到有关多智能体学习的更多信息。在这种环境下出现了两个 AICrowd 挑战赛：[Flatland Challenge](https://www.aicrowd.com/challenges/flatland-challenge)和[Flatland NeurIPS 2020 Competition](https://www.aicrowd.com/challenges/neurips-2020-flatland-challenge/)。这两个网页还提供了环境的进一步概述，并提供了进一步的入门资源。NeurIPS 2021 还通过 AICrowd 举办了 一项[新竞赛。](https://www.aicrowd.com/challenges/flatland)

## Hanabi

| Type              | Observations | Actions  | Code                                                         | Papers |
| ----------------- | ------------ | -------- | ------------------------------------------------------------ | ------ |
| Fully-cooperative | Discrete     | Discrete | [Environment](https://github.com/deepmind/hanabi-learning-environment) | [2]    |

![img](https://agents.inf.ed.ac.uk/blog/multiagent-learning-environments/img/hanabi.png)

来自 [2]：从玩家 0 的角度来看的四人 Hanabi 游戏示例。玩家 1 在玩家 0 之后行动，依此类推。

### 一般说明

Hanabi 挑战 [2] 基于纸牌游戏 Hanabi。这种针对 2 到 5 名玩家的完全合作博弈基于有限信息下的部分可观察性和合作概念。玩家必须协调他们打出的牌，但他们只能观察其他玩家的牌。他们自己的牌对自己是隐藏的，交流是游戏中的一种有限资源。这种环境的主要挑战是其显着的部分可观察性，侧重于有限信息下的智能体协调。在此环境中应用多智能体学习的另一个挑战是其回合制结构。在 Hanabi 中，玩家轮流行动，不像在其他环境中那样同时行动。在每一轮中，他们可以选择三个独立的动作之一：给出提示、从手中打出一张牌、

## Neural MMO

| Type        | Observations | Actions        | Code                                                | Papers | Others                                      |
| ----------- | ------------ | -------------- | --------------------------------------------------- | ------ | ------------------------------------------- |
| Competitive | Discrete     | Multi-Discrete | [Environment](https://github.com/openai/neural-mmo) | [21]   | [Blog](https://openai.com/blog/neural-mmo/) |

![img](https://agents.inf.ed.ac.uk/blog/multiagent-learning-environments/img/neural_mmo_interface.jpg)

来自 [21]：“Neural MMO 是用于 AI 研究的大规模多智能体环境。智能体通过觅食和战斗来争夺资源。本地游戏状态中的观察和动作表示可以实现高效的训练和推理。3D Unity 客户端提供高质量的可视化来解释学习到的行为。环境、客户端、培训代码和政策都是完全开源的，有正式文档，并通过实时社区 Discord 服务器提供积极支持。”

### 一般说明

Neural MMO [21] 基于 MMORPG（大型多人在线角色扮演游戏）的游戏类型。它的大型 3D 环境包含各种资源，智能体通过相对复杂的进程系统取得进展。智能体在这种环境中相互竞争，智能体仅限于部分可观察性，观察以其当前位置（包括地形类型）和占据智能体的健康、食物、水等为中心的正方形瓷砖。与其他智能体的交互是通过攻击进行的，智能体可以通过给定的资源（如水和食物）与环境交互。该环境的主要缺点是其规模大（运行成本高）、基础设施和设置复杂以及目标单调，尽管其环境非常多样化。

# Multi-Agent Frameworks/Libraries

除了上面列出的单个多代理环境之外，还有一些非常有用的软件框架/库支持各种多代理环境和游戏模式。

### OpenSpiel

OpenSpiel 是一个用于（多代理）强化学习的开源框架，支持多种游戏类型。“OpenSpiel 支持 n 人（单人和多人）零和、合作和一般和、单发和顺序、严格轮流和同时移动、完美和不完美的信息游戏，以及传统的多智能体环境，例如（部分和完全可观察的）网格世界和社会困境。” 它支持 Python 和 C++ 集成。然而，由于支持的游戏类型多种多样，OpenSpiel 并不遵循标准的 OpenAI 健身房风格的界面。有关 OpenSpiel 的更多信息，请查看以下资源：

- [OpenSpiel介绍](https://github.com/deepmind/open_spiel/blob/master/docs/intro.md)
- [OpenSpiel 概念和示例](https://github.com/deepmind/open_spiel/blob/master/docs/concepts.md)
- [在 OpenSpiel 中实现的所有游戏概览](https://github.com/deepmind/open_spiel/blob/master/docs/games.md)
- [OpenSpiel 中已提供的所有算法概览](https://github.com/deepmind/open_spiel/blob/master/docs/algorithms.md)

有关更多信息和文档，请参阅他们的 Github ( [github.com/deepmind/open_spiel](https://github.com/deepmind/open_spiel) ) 和相应的论文 [10]，了解详细信息，包括设置说明、代码介绍、评估工具等。

### PettingZoo

PettingZoo 是一个用于进行多代理强化学习研究的 Python 库。它包含多个 MARL 问题，遵循多代理 OpenAI 的 Gym 接口并包括以下多个环境：

- Atari：多人 Atari 2600 游戏（合作和竞争）
- Butterfly：我们开发的合作图形游戏，需要高度的配合
- 经典：经典游戏包括纸牌游戏、桌游等。
- MAgent：具有大量粒子代理的可配置环境，最初来自[这里](https://github.com/geek-ai/MAgent)
- MPE：一组简单的非图形通信任务，最初来自[这里](https://github.com/openai/multiagent-particle-envs)
- SISL：3个合作环境，原创来自[这里](https://github.com/sisl/MADRL)

带有文档的网站：[pettingzoo.ml](https://www.pettingzoo.ml/)

Github 链接：[github.com/PettingZoo-Team/PettingZoo](https://github.com/PettingZoo-Team/PettingZoo)

### Megastep

Megastep 是一个创建多代理环境的抽象框架，可以在 GPU 上完全模拟以实现快速模拟速度。它已经带有一些预定义的环境，可以在网站上找到包含详细文档的信息：[andyljones.com/megastep](https://andyljones.com/megastep/)

## Scripts

For the following scripts to setup and test environments, I use a system running Ubuntu 20.04.1 LTS on a laptop with an intel i7-10750H CPU and a GTX 1650 Ti GPU. To organise dependencies, I use Anaconda.

### MALMO

- [MALMO bash 安装脚本](https://agents.inf.ed.ac.uk/blog/multiagent-learning-environments/scripts/malmo/setup.sh)
- [MALMO多智能体脚本测试](https://agents.inf.ed.ac.uk/blog/multiagent-learning-environments/scripts/malmo/malmo_marl_test.py)

### Deepmind Lab

- [Deepmind Lab bash setup script](https://agents.inf.ed.ac.uk/blog/multiagent-learning-environments/scripts/dmlab/setup.sh)
- [Dependency WORKSPACE file](https://agents.inf.ed.ac.uk/blog/multiagent-learning-environments/scripts/dmlab/WORKSPACE)
- [Dependency python.BUILD file](https://agents.inf.ed.ac.uk/blog/multiagent-learning-environments/scripts/dmlab/python.BUILD)
- [Deepmind Lab agent script test](https://agents.inf.ed.ac.uk/blog/multiagent-learning-environments/scripts/dmlab/agent_test.py)

### Deepmind Lab2D

- [Deepmind Lab2D bash setup script](https://agents.inf.ed.ac.uk/blog/multiagent-learning-environments/scripts/dmlab2d/setup.sh)
- [Dependency WORKSPACE file](https://agents.inf.ed.ac.uk/blog/multiagent-learning-environments/scripts/dmlab2d/WORKSPACE)
- [Dependency python.BUILD file](https://agents.inf.ed.ac.uk/blog/multiagent-learning-environments/scripts/dmlab2d/python.BUILD)

## References

1. Stefano V Albrecht and Subramanian Ramamoorthy. A game-theoretic model and best-response learning method for ad hoc coordination in multiagent systems. In Proceedings of the 2013 International Conference on Autonomous Agents and Multi-Agent Systems, 2013.
2. Nolan Bard, Jakob N Foerster, Sarath Chandar, Neil Burch, H Francis Song, Emilio Parisotto, Vincent Dumoulin, Edward Hughes, Iain Dunning, Shibl Mourad, Hugo Larochelle, and L G Feb. The Hanabi Challenge : A New Frontier for AI Research. Artificial Intelligence, 2020.
3. Charles Beattie, Joel Z. Leibo, Denis Teplyashin, Tom Ward, Marcus Wainwright, Heinrich Küttler, Andrew Lefrancq, Simon Green, Vı́ctor Valdés, Amir Sadik, Julian Schrittwieser, Keith Anderson, Sarah York, Max Cant, Adam Cain, Adrian Bolton, Stephen Gaffney, Helen King, Demis Hassabis, Shane Legg, and Stig Petersen. DeepMind Lab. ArXiv preprint arXiv:1612.03801, 2016
4. Charles Beattie, Thomas Köppe, Edgar A Duéñez-Guzmán, and Joel Z Leibo. Deepmind Lab2d. ArXiv preprint arXiv:2011.07027, 2020.
5. Filippos Christianos, Lukas Schäfer, and Stefano Albrecht. Shared Experience Actor-Critic for Multi-Agent Reinforcement Learning. Advances in Neural Information Processing Systems, 2020.
6. Lasse Espeholt, Hubert Soyer, Remi Munos, Karen Simonyan, Volodymir Mnih, Tom Ward, Yotam Doron, Vlad Firoiu, Tim Harley, Iain Dunning, et al. Impala: Scalable distributed deep-rl with importance weighted actor-learner architectures. In Proceedings of the International Conference on Machine Learning, 2018.
7. Shariq Iqbal and Fei Sha. Actor-attention-critic for multi-agent reinforcement learning. In International Conference on Machine Learning, 2019.
8. Max Jaderberg, Wojciech M. Czarnecki, Iain Dunning, Luke Marris, Guy Lever, Antonio Garcia Castaneda, Charles Beattie, Neil C. Rabinowitz, Ari S. Morcos, Avraham Ruderman, Nicolas Sonnerat, Tim Green, Louise Deason, Joel Z. Leibo, David Silver, Demis Hassabis, Koray Kavukcuoglu, and Thore Graepel. Human-level performance in first-person multiplayer games with population-based deep reinforcement learning. ArXiv preprint arXiv:1807.01281, 2018
9. Matthew Johnson, Katja Hofmann, Tim Hutton, and David Bignell. The malmo platform for artificial intelligence experimentation. In Proceedings of the International Joint Conferences on Artificial Intelligence Organization, 2016.
10. Marc Lanctot, Edward Lockhart, Jean-Baptiste Lespiau, Vinicius Zambaldi, Satyaki Upadhyay, Julien Pérolat, Sriram Srinivasan et al. OpenSpiel: A framework for reinforcement learning in games. ArXiv preprint arXiv:1908.09453, 2019.
11. Joel Z Leibo, Cyprien de Masson d’Autume, Daniel Zoran, David Amos, Charles Beattie, Keith Anderson, Antonio Garcı́a Castañeda, Manuel Sanchez, Simon Green, Audrunas Gruslys, et al. Psychlab: a psychology laboratory for deep reinforcement learning agents. ArXiv preprint arXiv:1801.08116, 2018.
12. Ryan Lowe, Yi Wu, Aviv Tamar, Jean Harb, Pieter Abbeel, and Igor Mordatch. Multi-agent actor-critic for mixed cooperative-competitive environments. Advances in Neural Information Processing Systems, 2017.
13. Kevin R. McKee, Joel Z. Leibo, Charlie Beattie, and Richard Everett. Quantifying environment and population diversity in multi-agent reinforcement learning. ArXiv preprint arXiv:2102.08370, 2021.
14. Sharada Mohanty, Erik Nygren, Florian Laurent, Manuel Schneider, Christian Scheller, Nilabha Bhattacharya, Jeremy Watson et al. Flatland-RL: Multi-Agent Reinforcement Learning on Trains. ArXiv preprint arXiv:2012.05893, 2020.
15. Igor Mordatch and Pieter Abbeel. Emergence of grounded compositional language in multi-agent populations. ArXiv preprint arXiv:1703.04908, 2017.
16. Georgios Papoudakis, Filippos Christianos, Lukas Schäfer, and Stefano V Albrecht. Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks. Advances in Neural Information Processing Systems Track on Datasets and Benchmarks, 2021.
17. Diego Perez-Liebana, Katja Hofmann, Sharada Prasanna Mohanty, Noburu Kuno, Andre Kramer, Sam Devlin, Raluca D Gaina, and Daniel Ionita. The multi-agent reinforcement learning in malmö (marlö) competition. ArXiv preprint arXiv:1901.08129, 2019.
18. Cinjon Resnick, Wes Eldridge, David Ha, Denny Britz, Jakob Foerster, Julian Togelius, Kyunghyun Cho, and Joan Bruna. PommerMan: A multi-agent playground. ArXiv preprint arXiv:1809.07124, 2018.
19. Mikayel Samvelyan, Tabish Rashid, Christian Schroeder de Witt, Gregory Farquhar, Nantas Nardelli, Tim GJ Rudner, Chia-Man Hung, Philip HS Torr, Jakob Foerster, and Shimon Whiteson. The starcraft multi-agent challenge. In Proceedings of the 18th International Conference on Autonomous Agents and Multi-Agent Systems, 2019.
20. Lukas Schäfer. Curiosity in multi-agent reinforcement learning. Master’s thesis, University of Edinburgh, 2019.
21. Joseph Suarez, Yilun Du, Igor Mordatch, and Phillip Isola. Neural MMO v1.3: A Massively Multiagent Game Environment for Training and Evaluating Neural Networks. ArXiv preprint arXiv:2001.12004, 2020.
22. Oriol Vinyals, Timo Ewalds, Sergey Bartunov, Petko Georgiev, Alexander Sasha Vezhnevets, Michelle Yeo, Alireza Makhzani et al. "StarCraft II: A New Challenge for Reinforcement Learning." ArXiv preprint arXiv:1708.04782, 2017.
23. Peter R. Wurman, Raffaello D’Andrea, and Mick Mountz. Coordinating Hundreds of Cooperative, Autonomous Vehicles in Warehouses. In AI Magazine, 2008.