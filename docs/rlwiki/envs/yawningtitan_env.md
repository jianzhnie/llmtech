## Yawning-Titan 是什么？

Yawning-Titan 是一组抽象的、高度灵活的基于图形的网络安全模拟环境，能够模拟一系列网络安全场景， 支持基于 OpenAI Gym 的自主网络操作智能体的训练。

### 网络安全模拟器。

在网络环境中的模拟是建模现实世界计算机系统环境的过程，以预测多个智能体的行动结果，其中这些智能体的目标是控制系统或系统中的数据。

Yawning-Titan 旨在训练网络防御智能体，以在高度可定制的配置中防御任意网络拓扑结构。专注于提供快速模拟，以支持对抗概率红方智能体的防御性自主智能体的开发。

### 设计原则

Yawning-Titan 的设计遵循了以下关键原则：

- 简单胜过复杂
- 最低硬件要求
- 操作系统无关
- 支持多种算法
- 增强智能体/策略评估
- 支持灵活的环境和游戏规则配置

设计原则1、2和5使 Yawning-Titan 成为一个理想的环境，用于测试和评估网络安全决策制定中的新颖方法，并提供了将如DCBO这样的方法过渡到网络防御背景的手段。原则上，我们将建模一个蓝色智能体（Blue Agent）和一个红色智能体（Red Agent）的行动。

### 网络结构

设 $ \text{net} = (V_{\text{net}}, E_{\text{net}}) $，其中 顶点集$ V_{\text{net}} $ 和边集 $ E_{\text{net}} $，在  Yawning-Titan 中是模拟计算机网络的无向图。在进一步讨论之前，请注意不要混淆我们这项工作中使用的两种类型的图；$ g $  代表有向因果图，而 $ G_{\text{net}} $ 代表无向相关的计算机网络.

图1中展示了一个示例实例。每个节点$ V_i \in V_{\text{net}} $代表网络上的主机，而弧$ E_j \in E_{\text{net}} $代表两个连接主机之间可能的连通性（图1中的灰色边）。这种方法使我们能够以最简单的形式轻松地模拟各种不同类型的网络拓扑，而不必担心协议或不同类型的主机。这种对网络的基本处理方式使我们能够遵守我们的第一个设计原则。构成网络环境的无向图中，灰色连接线代表可行路径，红色连接线代表攻击者的攻击路径，绿色节点代表环境中的正常节点，不同的绿色深浅代表节点的不同脆弱性值，黄色节点代表被攻击的节点，蓝色节点代表防御者节点，红色节点代表正在被攻击者攻击的节点，紫色节点代表高价值目标，E 代表入口节点，蓝色圆圈代表防御者已知的脆弱性，红色圆圈代表防御者未知的脆弱性。

Yawning-Titan 通过使用定制配置允许相当大的定制。配置文件包含许多不同的设置，用户可以控制这些设置以改变游戏的玩法。从这些设置中，您可以更改智能体的成功条件以及通过切换动作和动作概率来改变不同智能体的行为方式。

网络中的机器各自具有独特的属性， 这会影响它们的行为， 以及这些属性会如何受到 RedAgent 和 BlueAgent 影响。

### Attack–defense 游戏设计

不同的防御策略对不同节点的影响各异。因此，我们从两个方面对主动和动态防御进行特征化：首先，为每个节点设置不同的脆弱性值，脆弱性值由节点颜色的亮度表示。亮度越高，脆弱性值越高，越容易被破坏。其次，在网络结构中融合了多种拓扑结构，使网络在逻辑环境中具有普适性。此外，攻击者在攻击过程中明显倾向于针对有价值的节点。因此，将高价值目标设置为资源点，这也作为确定游戏结束的一个标准。由于攻击者和防御者之间游戏过程的动态性和多样性，该模型根据每个节点是否被破坏、是否检测到被破坏的节点以及是否修复了被破坏的节点来定义每个节点的状态。

在游戏过程中，攻击者和防御者轮流操作网络中的节点。攻击者通过入口点 E  进入网络环境，并根据预定义的攻击规则定期发起攻击。判断节点是否被占领的标准基于攻击行为属性中的攻击等级值与正常节点的脆弱性值之间的差异。如果差异为正，则节点可能被占领；否则，被认为是不太可能的。当攻击者执行攻击步骤时，防御者根据 DRL 策略从一系列动作空间中选择防御动作，并与当前环境互动。防御者的智能体从与环境的互动中获得状态奖励和动作奖励，这些奖励用于为调整策略的强化学习算法提供反馈。

### 网络节点属性

#### Vulnerability score

影响 RedAgent 破坏节点的难易程度。它在每个场景开始时根据配置文件中的设置自动生成，并且有蓝方智能体动作可以修改节点的脆弱性得分。

#### Isolation Status

表示节点是否已被隔离。这意味着所有进入和出去的连接都被禁用。这可以通过蓝色智能体的动作进行修改。

#### True Compromised Status

表示一个节点是否已被 RedAgent  感染。如果 RedAgent 控制了节点，那么它可以利用该节点作为跳板扩散到其他连接的节点。

#### Blue seen Compromised Status

表示节点是否受到感染，以及 BlueAgent 是否意识到了入侵。根据场景配置的不同， BlueAgent 可能只能看到一个模糊的网络视图，并看到这一点而不是真实值，蓝方可见智能体状态有效地模拟了完美检测和不完美检测之间的区别。

### 攻击动作

攻击者可以采取各种攻击动作，并根据某种概率分布从攻击动作空间中选择它们。每种攻击动作也有一个概率分布来决定其成功的可能性。攻击目标的选择基于组成节点的环境信息。攻击者可以从任何被占领的节点发起攻击。攻击者的行为定义如下：
$$A_{\text{att}} = [A_{\text{ain}}, A_{\text{aat}}, A_{\text{azd}}, A_{\text{asp}}, A_{\text{asl}}] $$

攻击动作空间中的每种攻击类型都与攻击强度相关联，定义如下：
$$
\text{Strength}_{A_{\text{att}}} = \max \{ \sin(nt + a) + \cos(mt + b), \text{Attackskill} \}
$$
集合$A_{\text{att}} $中的所有攻击动作都使用$\text{Attackskill} $作为基线。正弦和余弦函数进行相位移动和频率变换，以获得攻击强度的子项，并与攻击能力进行比较，取较大值作为攻击强度。使用攻击强度的变化为防御智能体的策略学习环境增加更丰富的变化，使模型更能适应复杂的网络情况。动作定义如下：

(1)入侵。入侵有一定概率不会被检测到，未被检测到的入侵只能通过防御者执行特定动作来发现；

(2)普通攻击。它通过其攻击强度攻击环境中的节点，其攻击强度在与环境互动的过程中会随时间周期性变化。

(3)零日攻击：根据规则集，有一定机会发起零日攻击，与常规攻击相比，其成功率显著更高。零日攻击的发起受规则设定的特定时间要求的限制，并且对其发起次数有限制。

(4)扩散：相邻互联的节点可以相互影响，尤其是那些距离较近的节点。扩散模拟了被攻击节点可能对邻近未受影响节点产生一定影响的场景。随机扩散也用来表示网络内的不确定状态。

(5)静默，提供一种跳过本回合的手段。

### 防御动作

使用前述的配置设置，用户可以修改对 RedAgent 和 BlueAgent 可用的动作子集。 BlueAgent 有以下动作：

- 降低节点的脆弱性
- 扫描网络以查找红色入侵
- 从节点中移除 RedAgent
- 将节点重置回其初始状态
- 部署欺骗性节点
- 隔离节点并重新连接之前隔离的节点



RedAgent 有多种不同的攻击方式，可以根据配置文件中的设置来使用。除非使用保证成功的攻击，否则将使用下方公式来确定 RedAgent （技能等级0 $< RS \leq 1$）是否会感染节点$ V_i \in V_{\text{net}} $：

$$
AS = \frac {100 \times RS^2} {RS + (1 - \text{vuln}(V_i))}
$$
如果满足 $$ AS \geq u \quad $$ 条件， 则攻击成功。

其中 $ u $ 是从均匀分布$ U(0, 100) $中抽取的样本，$ AS $ 是攻击得分（衡量攻击力度的一个指标），$ \text{vuln}(V_i) $是节点$ V_i $的脆弱性（衡量主机被感染的可能性的一个指标）。

例如，没有防火墙的计算机或没有安全培训的用户，可以被建模为具有高脆弱性。由于 $ 0 \leq \text{vuln}(V_i) \leq 1 $ 且 $ 0 \leq AS \leq 100 $，使用这个公式确保了 RedAgent 感染节点的可能性随着智能体技能的提高而增加，随着节点防御的提高而减少（$ 1 - \text{vuln}(V_i) $）。

在每个时间步骤，每个智能体执行一个动作，从激活的动作子集中选择，以影响环境。Yawning-Titan 执行的顺序如下：

1. RedAgent 执行其动作
2. 环境检查 RedAgent 是否获胜
3. BlueAgent 执行其动作
4. 环境返回 BlueAgent 动作的奖励

5. 环境检查 BlueAgent 是否获胜。

Yawning-Titan 包含少量特定的、独立的 OpenAI Gym 环境，用于自主网络防御研究，非常适合学习和调试，以及灵活、高度可配置的通用环境，可用于表示一系列复杂性和规模不断增加的场景。通用环境只需要一个网络拓扑和一个设置文件，即可创建符合 OpenAI Gym 的环境，该环境还可以实现开放研究和增强的可重复性。尽管Yawning-Titan 是为与Stable- BlueAgent selines3（ RedAgent ffin等人，2021年）一起运行而构建的，但使用因果序列决策制定智能体来模拟 BlueAgent 的行为是很简单的。



![image-20240709163714972](../../../../../../Library/Application%20Support/typora-user-images/image-20240709163714972.png)



图1：YT 在25节点网络 Gnet 上的示例输出。BlueAgent 正在保护一个高价值目标节点（紫色节点）。RedAgent控制了网络的大部分，由黄色节点表示，包括已知和未知的受感染情况（由黄色节点的边的颜色表示），而BlueAgent并不知情。系统中有三个入口节点，其中两个已经被RedAgent利用（用‘E’标记）。BlueAgent正在从高价值目标节点相邻的节点中移除RedAgent。RedAgent 正在攻击网络左下角的一个节点。

### 强化学习智能体设计

主动防御智能体通过强化学习策略与环境互动获得相应的奖励，学习应对各种攻击情况的策略，并通过奖励强化重复训练，以获得更有效的防御策略。以下部分将详细描述强化学习智能体的每个组件的设计。


在网络环境中，网络信息通常不是完全可检测的。因此，在设计智能体对网络环境的感知时，将部分可观察环境作为观察空间$ O $来考虑是很重要的。观察空间取决于智能体在网络中的位置。具体来说，观察空间的定义如下：
$$
O = \{O_{\text{node}}, O_{\text{globe}}\}
$$
节点观察信息定义如下：
$$O_{\text{node}} = [N_{\text{vu\_ln}}, N_{\text{at\_in}}, N_{\text{at\_ed}}, N_{\text{sp\_nd}}] $$

$ N_{\text{vu\_ln}} $ 是节点的脆弱性值。$ N_{\text{at\_in}} $ 表示被防御智能体检测到的新攻击信息，$ N_{\text{at\_ed}} $ 是已经被攻击的节点，$ N_{\text{sp\_nd}} $ 指示入口节点和高价值目标节点的位置信息。节点信息$ O_{\text{node}} $用于计算已知的全局信息。

全局观察信息定义如下：
$$ O_{\text{globe}} = [E_{\text{conn}}, V_{\text{uaver}}]  $$

$ E_{\text{conn}} $ 表示图中的连通性，$ V_{\text{uaver}} $ 表示所有节点脆弱性值的平均值。确定图中的连通性和平均脆弱性$ V_{\text{uaver}} $是奖励计算的重要指标。

### 动作空间

动作空间作为智能体与环境互动的方式集合，是本文设计的核心方面。为了解决网络安全博弈问题，动作空间的设计旨在尽可能从现实世界的攻击和防御行为中抽象出来。在本文中，我们根据相关文献[18]、[19]中描述的防御行为设计，实现了防御者动作空间的设计，它涵盖了保护、检测、恢复和响应的整个过程。专注于抽象攻击行为的效果，它将不可量化的行为效果量化为适合算法分析表达和应用的形式。

防御者通过DRL方法训练，得出防御策略，然后选择攻击动作。防御动作空间有三种类型的八种动作。

防御动作空间定义如下：
$$A_{\text{def}} = [A_{\text{glob}}, A_{\text{dece}}, A_{\text{node}}]  $$

防御者的动作具体定义如下：

#### 第一类：全局动作

$$A_{\text{glob}} = [A_{\text{scan}}, A_{\text{dsl}}] $$

(1) $ A_{\text{scan}} $ 是对未检测到的入侵进行的相应全局扫描，用于感知网络中每个节点的状态，是检测手段的抽象。

(2) $ A_{\text{dsl}} $  是静默，不采取任何行动。

#### 第二类：欺骗动作 $ A_{\text{dece}} $

欺骗动作类似于蜜罐技术，防御者可以通过放置欺骗节点来诱导攻击者进行攻击。当欺骗节点被攻击时，防御者可以立即知道。

#### 第三类：针对节点的动作$ A_{\text{node}} $

$$A_{\text{node}} = [A_{\text{de\_vl}}, A_{\text{re\_st}}, A_{\text{re\_co}}, A_{\text{sp\_nd}}, A_{\text{re\_lk}}] $$

(1) $ A_{\text{de\_vl}} $ 降低节点脆弱性值，使节点不易被破坏，在网络安全措施中进行抽象。

(2) $ A_{\text{re\_st}} $ 重置节点。被破坏的节点可以由防御者完全恢复，同时在短期内不遭受攻击，这是一种理想的防御措施，虽然在现实中不易实现。但作为一种防御措施，它也加入了防御者的动作空间。如果防御智能体滥用这种方法，将产生巨大的成本。

(3) $ A_{\text{re\_co}} $ 恢复节点，使被攻击的节点恢复，但恢复的节点仍然有被攻击的风险。

(4) $ A_{\text{sp\_nd}} $ 隔离节点，类似于沙箱技术，将节点与其它邻近节点断开连接。在实际网络环境中，隔离重要节点的成本非常高或甚至不允许，因此采取这一行动也会产生巨大的成本。

(5) $ A_{\text{re\_lk}} $恢复连接。被隔离的节点将重新连接到网络。

### 奖励函数

奖励函数是一个衡量智能体执行的防御动作有效性的模块，并且为深度强化神经网络的更新提供指导。在本文中，奖励函数被设计为一种复合形式，既考虑了单个互动结果，也考虑了整体互动结果。计算每次互动的单步奖励用于定量评估模型的性能。另一方面，累积的总奖励和每集的奖励输入到神经网络中，用于迭代贝叶斯方程更新。具体定义如下：

单次互动奖励函数的定义：
$$ r_i = r_{\text{cost},i} \ast r_{\text{state},i} $$

单步i的奖励函数设计将为智能体的不同动作计算不同的成本 $r_{\text{cost},i} $，然后将执行动作前后环境的状态变化奖励$r_{\text{state},i} $ 结合起来进行复合计算$\ast $，以计算单步奖励函数的值。

防御者动作成本被用作奖励的基线，动作成本定义如下：
$$ r_{\text{cost},i} = \text{Rewdc}(A_{\text{def}}[A_i])  $$

防御动作集合中的每个防御动作$A_{\text{def}}[A_i] $都有一个设定的基础成本，并且使用成本计算函数映射对第i步的防御动作进行奖励统计。

攻击者和防御者与环境互动一次，然后对当前环境产生影响，环境的变化也需要在奖励函数中考虑。状态奖励定义如下：
$$r_{\text{state},i} = \lambda \cdot \Delta V_{\text{ulave},i}  $$

奖励函数将在互动开始和结束时获取观察空间中全局观察信息的平均脆弱性值$V_{\text{ulave},i} $，并获取变化量$\Delta V_{\text{ulave},i} $。为了适应动作成本的计算幅度，进行λ因子校正，最终计算出第i步状态的奖励函数$r_{\text{state},i} $。

在获取动作成本和状态奖励后，将两者有机地联系起来。某种状态变化对应于具有相应奖励和惩罚机制的动作选择。通过大量的游戏训练，引导智能体学会应对各种情况，并产生有效的防御测量策略。然后，将每一步的奖励$r_i $累加得到总奖励$R_{\text{eps}} $，用于评估每个完整的游戏，定义如下：
$$R_{\text{eps}} = \sum_{i=1}^{n} r_i $$

### 决策过程构建

在模拟的网络环境中，攻击者根据规则与环境互动，主动防御智能体根据强化学习策略选择动作与环境互动。攻击者和防御者基于马尔可夫决策模型生成游戏序列 Trajectory ，直到游戏结束。

在学习过程中，强化学习神经网络通过学习由游戏序列生成的游戏记录来更新游戏策略。更新游戏策略后，主动防御智能体使用更新后的游戏策略在新一轮的游戏模型中与环境互动。通过循环参与循环游戏学习过程，深度强化学习模型不断细化其主动防御策略。

我们的方法的迭代更新过程可以描述为策略评估和策略改进程序，其核心是贝尔曼方程的递归求解。通过不断迭代状态值函数（公式(1)）和状态-动作函数（公式(2)），状态值函数评估策略的价值，而状态-动作函数改进策略本身。通过策略评估函数和策略改进函数获得的最优策略值最终会收敛，并且从理论上我们认为策略达到最优性（公式(4)）。然而，在实际实验中，我们观察到训练过程中损失函数（公式(3)）的波动，导致公式(1)，(2)的次优收敛。因此，我们将游戏结束条件设置为当游戏收益显著超过损失时。一旦防御策略有效应对攻击者，游戏学习过程就会停止。
$$
V^{\pi}(s) = \sum_{a \in \mathcal{A}_{\text{def}}} \pi(s, a) \geq \sum_{s' \in S} P_{\text{env}}(s \rightarrow s')(R_{\text{env}} + V^{\pi}(s'))
$$

$$
Q(s, a) = \mathbb{E}_{r_{t+1} \sim P, s_{t+1} \sim P} \left[ \sum_{t=0}^{\infty} \gamma^t r_{t+1} \mid s_t = s, a_t = a \right]
$$

$$
V^{\pi}(S) = \mathbb{E} \left[ \sum_{t=0}^{\infty} \log \pi(a_t | s_t) Q^{\pi'}(s_t, a_t) \right]
$$

$$
a \in \mathcal{A}_{\text{def}}, r + \max_{a'} Q'(s', a')
$$

### 实验模拟

#### 实验准备

为了充分展示本文方法的有效性，我们在深度强化学习工作站上进行了模拟实验。构建了YAWNING-TITAN框架来模拟攻防决策过程，实验平台和相关参数如表1所示。

常见的DRL算法架构分别基于Q学习和 Actor-critic，还有两种策略的在线（on-policy）和离线（off-policy）算法用于模型训练。在本文中，我们选择了符合云边界网络环境实际需求的代表性算法：近端策略优化（PPO）[20]和深度Q网络（DQN）[21]，以验证智能体是否能够通过深度强化学习算法学习网络安全情境意识和反馈策略，从而生成有效的防御策略。

表1. 实验环境

| 软件和硬件 | 参数            |
| ---------- | --------------- |
| 框架       | YAWNING-TITAN   |
| 操作系统   | Ubuntu 22.04.2  |
| Python     | 3.9.13          |
| PyTorch    | 1.13.0          |
| CPU        | AMD EPYC 7T83   |
| GPU        | NVIDIA RTX 3090 |
| SSD        | 1 TB            |
| 内存       | 128 GB          |

以下是将表2、表3和表4内容通过表格形式展示：

#### 表2：模型训练参数

| 参数名称        | 参数值  |
| --------------- | ------- |
| 训练最大时间步  | 500,000 |
| 评估频率        | 1000    |
| 每集最大步数    | 200     |
| 优化器          | Adam    |
| 激活函数        | ReLu    |
| PPO策略更新频率 | 2048    |
| DQN批量大小     | 128     |
| 种子数          | 100     |

#### 表3：抽象攻击动作与实际攻击动作的对比表

| 抽象攻击动作 | 实际攻击动作 | 概率 |
| ------------ | ------------ | ---- |
| Aain         | 入侵         | 0.3  |
| Aaat         | 常规攻击     | 0.5  |
| Aazd         | 零日攻击     | 0.01 |
| Aasp         | 传播         | 0.1  |
| Aasl         | 静默         | 0.09 |

#### 表4：实际防御动作与抽象防御动作的对比表

| 抽象防御动作         | 实际防御动作     | 成本 |
| -------------------- | ---------------- | ---- |
| Scan                 | 漏洞扫描         | 0.1  |
| Silence              | 无操作（静默）   | -0.5 |
| Deception            | 蜜罐技术         | 1    |
| Reduce vulnerability | 加强认证和防火墙 | 0.3  |
| Reset node           | 完全修复节点     | 0.5  |
| Recovery node        | 改进漏洞数据库   | 0.4  |
| Separate node        | 沙箱隔离         | 1.5  |
| Reconnection node    | 重新连接节点     | 0    |

#### 模型在不同算法上的应用对比

为了考察不同算法对主动防御智能体的影响，采用基于Clipped代理目标函数和Actor-critic架构的PPO算法和基于双Q学习的DQN算法进行对比实验，实验过程中网络攻防智能体之间的博弈可视化过程如图[6](https://www.sciencedirect.com/science/article/pii/S2667295223000430#fig6)所示。

本文按照时间顺序提取了6个有代表性的帧来可视化展示网络环境中攻防双方的行为。[图6](https://www.sciencedirect.com/science/article/pii/S2667295223000430#fig6)（a）第1帧，游戏开始，攻击者从入口节点进入。[图6](https://www.sciencedirect.com/science/article/pii/S2667295223000430#fig6)（b）第2帧，防守方扫描威胁，检测到威胁，断开部分连接隔离受损区域，并增加部分欺骗节点进行检测。图[6](https://www.sciencedirect.com/science/article/pii/S2667295223000430#fig6)（c）第3帧，防守方根据攻防情况，调整欺骗节点的位置。[图6](https://www.sciencedirect.com/science/article/pii/S2667295223000430#fig6)（d）第4帧和（e）第5帧，攻击方节点处于较高攻击强度状态，防守方断开高价值节点进行保护。[图6](https://www.sciencedirect.com/science/article/pii/S2667295223000430#fig6)（f）第6帧表示在游戏结束前恢复尽可能多的节点，以获得更高的奖励。

![img](https://ars.els-cdn.com/content/image/1-s2.0-S2667295223000430-gr6.jpg)

> 图6 .模拟环境游戏流程 (a) 帧1. (b) 帧2. (c) 帧3. (d) 帧4. (e) 帧5. (f) 帧6.

本文设置了两种代表性算法进行对比实验，其中最大博弈长度（max step）衡量指标为主动防御系统对攻击行为的容忍度，奖励衡量指标为主动防御智能体从环境中获得的奖励。实验结果[如图7 （a）、（b）所示，实验数据如](https://www.sciencedirect.com/science/article/pii/S2667295223000430#fig7)[表5](https://www.sciencedirect.com/science/article/pii/S2667295223000430#tbl5)所示。

橙色曲线表示的 PPO 算法在游戏长度和奖励稳定性方面优于蓝色曲线表示的 DQN 算法。PPO 算法在 150k 时间步左右以较小的代价学会获得最长游戏长度，而 DQN 算法在 50k 时间步后开始学习，在 100k 时间步左右以较小的代价学会最长游戏长度，但获得的奖励比 PPO 算法少。PPO 算法的最佳性能出现在 150k 到 250k 时间步之间，DQN 的最佳性能出现在 200k 时间步之后。由于两种算法在后期探索中的效率都有所下降，使用确定性策略执行动作会导致在 270k 时间步时性能略有下降并且出现抖动。实验表明,在本文设计的攻防博弈环境中,防御智能体能够通过不同的强化学习算法学习到有效的防御策略;并且抽象攻防博弈环境的设计能够充分刻画云边界网络的攻防环境,使得强化学习智能体能够通过在环境中的交互完成自身的进化迭代。

![img](https://ars.els-cdn.com/content/image/1-s2.0-S2667295223000430-gr7.jpg)

Table 5. Comprehensive experimental results.

| Type                   | Algorithm | 100k step reward | 200k step reward | 300k step reward | 400k step reward | 500k step reward |
| :--------------------- | :-------- | :--------------- | :--------------- | :--------------- | :--------------- | :--------------- |
| Algorithm compare      | PPO       | 98.64            | 337.7            | 338.9            | 244.5            | 184.6            |
|                        | DQN       | −19.01           | 66.02            | −27.8            | 17.97            | 88.23            |
| Scenario               | PPO       | −110.4           | −132.7           | −92.46           | −163.5           | −203.8           |
|                        | DQN       | −104.5           | −90.88           | −211.2           | −175.4           | −63.54           |
| Attack_model mid_skil  | PPO       | 112.4            | 136.3            | 167.9            | 167.9            | −8.253           |
|                        | DQN       | −56.11           | −80.17           | −70.81           | −94.62           | 87.72            |
| Attack_model low_skil  | PPO       | 33               | −40.22           | 70.81            | −25.86           | 165.6            |
|                        | DQN       | 1.287            | 8.607            | −21.71           | 125.1            | −94.62           |
| Reward_func Only node  | PPO       | 52.47            | 46.88            | 20.67            | 42.47            | 49.97            |
|                        | DQN       | −11.07           | −9.934           | −2.528           | −11.61           | 31.45            |
| Reward_func Only state | PPO       | −6.346           | −35.13           | −5.96            | −1.827           | −7.58            |
|                        | DQN       | −103.1           | −167.7           | −39.74           | −23.72           | −34.27           |

为了检验智能体在多变、复杂的场景中是否能够学习到有效的防御策略，本文设置了动态场景实验，通过随机初始化每个博弈对中的节点来进一步增强实验环境的动态性，场景变化如图[9](https://www.sciencedirect.com/science/article/pii/S2667295223000430#fig9)所示。从[图9](https://www.sciencedirect.com/science/article/pii/S2667295223000430#fig9)（a）（b）（c）三张图可以看出，对抗博弈每局初始化节点时都会在网络拓扑中重新选择特殊节点的位置，并重置正常节点的脆弱性值，动态场景训练结果如图[8](https://www.sciencedirect.com/science/article/pii/S2667295223000430#fig8)（a）和（b）所示。具体训练结果数据如[表5](https://www.sciencedirect.com/science/article/pii/S2667295223000430#tbl5)所示。

基于动态场景实验的数据和训练结果，无法分析场景特定节点位置和节点脆弱性值随机化对智能体训练效率的负面影响。为进一步体现本文模型在变化场景下的鲁棒性，取消了最大步长限制。PPO算法较强的探索能力从最大博弈长度度量上带来了更好的入侵容忍度，但算法在奖励度量上也表现出一定的波动。

奖励度量对比表明，DQN算法的双Q更新策略保证了算法在应对变化环境条件时的稳定性。PPO算法在最大游戏长度度量上在300k时间步左右超过200Maxstep准则，在350k时间步左右达到稳定。从[图7](https://www.sciencedirect.com/science/article/pii/S2667295223000430#fig7)和[图8](https://www.sciencedirect.com/science/article/pii/S2667295223000430#fig8)两组实验对比不难看出，在奖励度量方面，动态场景[图8](https://www.sciencedirect.com/science/article/pii/S2667295223000430#fig8)低于固定场景[图7](https://www.sciencedirect.com/science/article/pii/S2667295223000430#fig7)，说明在变化的环境中防御智能体往往需要更多的成本来应对攻击。但两组对比实验都得到了相同的实验结论：仿真模型能够稳定获得最终的奖励，主动防御策略是有效且稳定的。虽然动态的云边界网络环境对智能体的防御策略学习有损害，但智能体仍然能够有效地防御网络攻击并做出主动决策，表明本文对云边界网络攻防环境的抽象和建模是有效和鲁棒的。

（3）不同攻击模式下模型的策略学习对比

多梯度攻击规则可以有效检验攻击者攻击强度对防御智能体学习效率的影响。本文设置了3种具有强度梯度的攻击规则，分别为最强档、次强档和最弱档。最强档为周期性攻击策略，次强档为非周期性增强攻击，最弱档为在非周期性增强攻击等级基础上禁用特殊攻击。具体训练结果数据如[表5](https://www.sciencedirect.com/science/article/pii/S2667295223000430#tbl5)所示，实验结果[如图7](https://www.sciencedirect.com/science/article/pii/S2667295223000430#fig7)、[图10](https://www.sciencedirect.com/science/article/pii/S2667295223000430#fig10)、[图11](https://www.sciencedirect.com/science/article/pii/S2667295223000430#fig11)、图[12](https://www.sciencedirect.com/science/article/pii/S2667295223000430#fig12)所示。[图10](https://www.sciencedirect.com/science/article/pii/S2667295223000430#fig10)为两种算法低攻击强度对比，[图11](https://www.sciencedirect.com/science/article/pii/S2667295223000430#fig11)为中等攻击强度对比，[图7](https://www.sciencedirect.com/science/article/pii/S2667295223000430#fig7)为最强攻击强度对比，[图12](https://www.sciencedirect.com/science/article/pii/S2667295223000430#fig12)为3种攻击强度对比图。

[从图7](https://www.sciencedirect.com/science/article/pii/S2667295223000430#fig7)、[图10](https://www.sciencedirect.com/science/article/pii/S2667295223000430#fig10)、图[11](https://www.sciencedirect.com/science/article/pii/S2667295223000430#fig11)中的奖励指标可以分析出防御智能体在三个等级的攻击强度下都能利用两种强化学习算法学习到防御策略。从最大博弈长度指标可以分析出在低级攻击模式下两种算法的学习效果相似，均能通过学习获得奖励，但弱攻击带来的问题是弱攻击策略学习到的防御策略不稳定，防御智能体始终处于较高的探索率。在中等攻击行为模式下，PPO算法在稳定性和最终奖励上均强于DQN算法，DQN算法在100k时间步左右达到稳定，PPO算法在150k时间步左右稳定，说明中等强度的攻击模式促进了智能体的策略学习。在最强攻击模式下，防御智能体在最大博弈长度附近保持稳定，奖励获取有效增长，是符合理论预期的训练性能。从[图12](https://www.sciencedirect.com/science/article/pii/S2667295223000430#fig12)中两种算法的三种攻击模式的综合对比中，奖励曲线最终趋于平缓可以分析出两种深度强化学习算法均可以使防御智能体学习到防御策略。实验表明，作为与防御者博弈的主体，攻击者的行为抽象方法对防御智能体策略学习也起到了关键作用。本文通过设置周期性的攻击模式和各种基于概率的攻击，防御智能体可以在现实环境量化的攻击行为基础上，通过与基于规则的攻击者交互，学习到有效、稳定的防御策略。

![img](https://ars.els-cdn.com/content/image/1-s2.0-S2667295223000430-gr10.jpg)

图 10 .低技能攻击。

![img](https://ars.els-cdn.com/content/image/1-s2.0-S2667295223000430-gr11.jpg)

图 11.中等攻击强度。

![img](https://ars.els-cdn.com/content/image/1-s2.0-S2667295223000430-gr12.jpg)

图12.不同攻击方法的比较。

（4）不同奖励函数下模型的策略学习对比

![img](https://ars.els-cdn.com/content/image/1-s2.0-S2667295223000430-gr13.jpg)

图 13。PPO奖励比较。

![img](https://ars.els-cdn.com/content/image/1-s2.0-S2667295223000430-gr14.jpg)

图 14 . DQN 奖励比较。

为了考察奖励函数对防御型智能体学习策略的影响，本文进行了三组对比实验。前三组实验基于当前状态，统计博弈每一步博弈网络中的综合脆弱性和行动成本，不考虑单个节点的状态奖励。另一组实验采用基于单个节点状态的奖励函数，不考虑整体网络动态对智能体的影响。第三组实验将每一步博弈中节点修改的统计状态成本与同时考虑整体网络动态和单个节点变化的奖励相结合。这两类奖励的组合在理论上有望提供更有效的指导。实验数据见表[5 ，实验结果](https://www.sciencedirect.com/science/article/pii/S2667295223000430#tbl5)[如图13](https://www.sciencedirect.com/science/article/pii/S2667295223000430#fig13)和[图14](https://www.sciencedirect.com/science/article/pii/S2667295223000430#fig14)所示。

通过设置三种不同的奖励模式，体现了奖励函数对智能体学习的重要作用。PPO算法训练结果如图[13所示。对](https://www.sciencedirect.com/science/article/pii/S2667295223000430#fig13)[图13](https://www.sciencedirect.com/science/article/pii/S2667295223000430#fig13) (a)的分析表明，在最大博弈长度度量上，单一状态奖励或节点奖励比组合奖励略微稳定。但从[图13](https://www.sciencedirect.com/science/article/pii/S2667295223000430#fig13) (b)可以看出，单一节点奖励对智能体学习的推动作用不大，从单一状态奖励中学习到的防守策略会比从综合奖励中学习到的防守策略花费更多，说明综合奖励在奖励度量上占有绝对优势。DQN算法训练结果[如图14所示，对](https://www.sciencedirect.com/science/article/pii/S2667295223000430#fig14)[图14](https://www.sciencedirect.com/science/article/pii/S2667295223000430#fig14) (a)的分析表明，单一节点变化奖励对智能体防守策略学习的推动作用小于另外两种奖励模式，在DQN算法下表现相似。[图14](https://www.sciencedirect.com/science/article/pii/S2667295223000430#fig14) (b)从奖励度量上展示了三种奖励模式对智能体策略学习的影响。与PPO算法类似，单一节点变化对整体智能体学习几乎没有驱动作用，单一状态奖励与综合奖励模式相比仍不具优势。实验表明，本文提出的基于单节点变化和整体场景变化集的综合奖励是一种比单一一种奖励机制更有效的奖励方式，局部与整体的综合考虑使防守智能体达到最大限度地保持局部网络节点正常的同时兼顾整体防守成本最低的博弈均衡点。本文设置的奖励模式也是前三次对比实验中的奖励模式，在不同的实验对比对象下，奖励函数依然表现有效且鲁棒。

## Yawning-Titan 如何使用？

Yawning-Titan 可通过 CLI 应用程序或 GUI 使用。这样做的目的是让所有用户都能尽可能方便地使用 Yawning-Titan，同时又不影响用户对源代码进行深入修改的能力。

### 基于什么构建的

Yawning-Titan 建立在巨人的肩膀上，并且严重依赖于以下库：

> - [OpenAI’s Gym](https://gym.openai.com/) 是所有环境的基础
> - [Networkx](https://github.com/networkx/networkx)  作为所有环境使用的底层数据结构
> - [Stable baselines 3](https://github.com/DLR-RM/stable- baselines3)  用作 RL 算法的来源
> - [Rllib (part of  ray)](https://github.com/ RedAgent y-project/ ray) 被用作 RL 算法的另一个来源
> - [Typer](https://github.com/tiangolo/typer) 用于提供命令行界面
> - [Django](https://github.com/django/django/) 用于提供 GUI 的管理和元素
> - [CYT oscape JS](https://github.com/cYawning-Titan oscape/cYawning-Titan oscape.js/) 用于提供轻量级且直观的网络编辑器

### Yawning-Titan 使用

#### 导入

首先，导入网络和节点。

```
from yawning_titan.networks.node import Node
from yawning_titan.networks.network import Network
```

#### 网络

要创建一个网络，首先我们必须实例化一个[`Network`](https://dstl.github.io/YAWNING-TITAN/source/_autosummary/yawning_titan.networks.network.Network.html#yawning_titan.networks.network.Network)的实例。

虽然[`Network`](https://dstl.github.io/YAWNING-TITAN/source/_autosummary/yawning_titan.networks.network.Network.html#yawning_titan.networks.network.Network)可以通过调用`Network()`直接实例化，但是您可以设置一些可配置参数（我们将在下文中讨论这些参数）。

#### 节点实例

接下来我们实例化一些[`Node`](https://dstl.github.io/YAWNING-TITAN/source/_autosummary/yawning_titan.networks.node.Node.html#yawning_titan.networks.node.Node)。

再次，虽然[`Node`](https://dstl.github.io/YAWNING-TITAN/source/_autosummary/yawning_titan.networks.node.Node.html#yawning_titan.networks.node.Node)可以通过调用`Node()`直接实例化，但您可以设置一些可配置参数（我们将在下文中讨论这些参数）。

```
node_1 = Node()
node_2 = Node()
node_3 = Node()
node_4 = Node()
node_5 = Node()
node_6 = Node()
```

#### 将节点添加到网络

目前我们只有一个实例[`Network`](https://dstl.github.io/YAWNING-TITAN/source/_autosummary/yawning_titan.networks.network.Network.html#yawning_titan.networks.network.Network)和一些实例 [`Node`](https://dstl.github.io/YAWNING-TITAN/source/_autosummary/yawning_titan.networks.node.Node.html#yawning_titan.networks.node.Node)。

要将 [`Node`](https://dstl.github.io/YAWNING-TITAN/source/_autosummary/yawning_titan.networks.node.Node.html#yawning_titan.networks.node.Node)  添加到  [`Network`](https://dstl.github.io/YAWNING-TITAN/source/_autosummary/yawning_titan.networks.network.Network.html#yawning_titan.networks.network.Network)，我们需要调用`.add_node()`。

```
network.add_node(node_1)
network.add_node(node_2)
network.add_node(node_3)
network.add_node(node_4)
network.add_node(node_5)
network.add_node(node_6)
```

#### 在节点间添加边

通过调用`.add_edge() `添加边。

```python
network.add_edge(node_1, node_2)
network.add_edge(node_1, node_3)
network.add_edge(node_1, node_4)
network.add_edge(node_2, node_5)
network.add_edge(node_2, node_6)
```

就这样，基础[`Network`](https://dstl.github.io/YAWNING-TITAN/source/_autosummary/yawning_titan.networks.network.Network.html#yawning_titan.networks.network.Network)已经创建完毕。

#### 设置入口节点

入口节点可以在以下位置手动设置：

```python
node_1.entry_node = True
```

或者通过配置来[`Network`](https://dstl.github.io/YAWNING-TITAN/source/_autosummary/yawning_titan.networks.network.Network.html#yawning_titan.networks.network.Network)随机设置它们：

```python
from yawning_titan.networks.network import  RandomEntryNodePreference
network.set_random_entry_nodes = True
network.num_of_random_entry_nodes = 1
network.random_entry_node_preference = RandomEntryNodePreference.EDGE
network.reset_random_entry_nodes()
```

#### 设置 EntHigh 值节点

可以在以下位置手动设置高价值节点：

```python
node_1.high_value_node = True
```

或者通过配置来随机设置它们：

```python
from yawning_titan.networks.network import  RandomHighValueNodePreference

network.set_ RedAgent ndom_high_value_nodes = True
network.num_of_ RedAgent ndom_high_value_nodes = 1
network. RedAgent ndom_high_value_node_preference =  RedAgent ndomHighValueNodePreference.FURTHEST_AWAY_FROM_ENTRY
network.reset_ RedAgent ndom_high_value_nodes()
```

#### 设置节点漏洞

可以在以下位置手动设置节点漏洞[`Node`](https://dstl.github.io/YAWNING-TITAN/source/_autosummary/yawning_titan.networks.node.Node.html#yawning_titan.networks.node.Node)：

```python
node_1.vulne RedAgent bility = 0.5
```

或者通过配置来[`Network`](https://dstl.github.io/YAWNING-TITAN/source/_autosummary/yawning_titan.networks.network.Network.html#yawning_titan.networks.network.Network)随机设置它们：

```python
network.set_ RedAgent ndom_vulne RedAgent bilities = True
network.reset_ RedAgent ndom_vulne RedAgent bilities()
```

#### 重置网络

要一次性重置所有入口节点、高价值节点和漏洞：

```python
network.reset()
```

#### 查看网络节点详情

要查看网络节点详情：

```python
network.show(verbose=True)
```

这将产生如下输出：

```python
UUID                                  Name    High Value Node    Entry Node      Vulne RedAgent bility  Position (x,y)
------------------------------------  ------  -----------------  ------------  ---------------  ----------------
bf308d9f-8382-4c15-99be-51f84f75f9ed          False              False               0.0296121  0.34, -0.23
1d757e6e-b637-4f63-8988-36e25e51cd55          False              False               0.711901   -0.34, 0.23
8f76d75c-5afd-4b2c-98ed-9c9dc6181299          True               False               0.65281    0.50, -0.88
38819aa3-0c05-4863-8b9d-c704f254e065          False              False               0.723192   1.00, -0.13
cc06f5e0-c956-449a-b397-b0e7bed3b8d4          False              True                0.85681    -0.49, 0.88
665b150b-fbd3-42a7-b899-3770ef2b285a          False              False               0.48435    -1.00, 0.13
```

## 示例网络

在这里，我们将创建在 Yawning-Titan 测试 ( tests.conftest.corpo RedAgent te_network )中用作固定装置的企业网络。

当每个节点在网络图中显示时，都会添加名称。

```python
# Instantiate the Network
network = Network(
    set_ RedAgent ndom_entry_nodes=True,
    num_of_ RedAgent ndom_entry_nodes=3,
    set_ RedAgent ndom_high_value_nodes=True,
    num_of_ RedAgent ndom_high_value_nodes=2,
    set_ RedAgent ndom_vulne RedAgent bilities=True,
)

# Instantiate the Node's and add them to the Network
router_1 = Node("Router 1")
network.add_node(router_1)

switch_1 = Node("Switch 1")
network.add_node(switch_1)

switch_2 = Node("Switch 2")
network.add_node(switch_2)

pc_1 = Node("PC 1")
network.add_node(pc_1)

pc_2 = Node("PC 2")
network.add_node(pc_2)

pc_3 = Node("PC 3")
network.add_node(pc_3)

pc_4 = Node("PC 4")
network.add_node(pc_4)

pc_5 = Node("PC 5")
network.add_node(pc_5)

pc_6 = Node("PC 6")
network.add_node(pc_6)

server_1 = Node("Server 1")
network.add_node(server_1)

server_2 = Node("Server 2")
network.add_node(server_2)

# Add the edges between Node's
network.add_edge(router_1, switch_1)
network.add_edge(switch_1, server_1)
network.add_edge(switch_1, pc_1)
network.add_edge(switch_1, pc_2)
network.add_edge(switch_1, pc_3)
network.add_edge(router_1, switch_2)
network.add_edge(switch_2, server_2)
network.add_edge(switch_2, pc_4)
network.add_edge(switch_2, pc_5)
network.add_edge(switch_2, pc_6)

# Reset the entry nodes, high value nodes, and vulne RedAgent bility scores by calling .setup()
network.reset()

# View the Networks Node Details
network.show(verbose=True)
```

给出：

```
UUID                                  Name      High Value Node    Entry Node      Vulne RedAgent bility  Position (x,y)
------------------------------------  --------  -----------------  ------------  ---------------  ----------------
c883596b-1d86-44f5-b4de-331292d8e3d5  Router 1  False              False               0.320496   0.00, -0.00
b2bd683b-a773-40de-85e8-36c21e66613d  Switch 1  False              False               0.889044   0.01, 0.61
68d9689b-5365-4022-b3bd-92bdc5a1627b  Switch 2  True               False               0.0671795  -0.00, -0.62
3554ed26-9480-487b-9d3c-57975654a2af  PC 1      False              False               0.400729   -0.38, 0.69
89700b3f-8be2-4b70-a21e-a0772551a6bc  PC 2      True               False               0.0807914  0.18, 1.00
82e91c52-5458-493a-a7cd-00fb702d6af1  PC 3      False              True                0.86676    0.39, 0.70
91edf896-f004-4ca7-9587-cc8417c4a26b  PC 4      False              False               0.967413   -0.39, -0.69
ebbc79f7-9a52-4a08-8b56-fee816284b54  PC 5      False              True                0.684436   0.38, -0.69
2cdaaf06-9b4a-41e9- BlueAgent 6f-129aec634080  PC 6      False              False               0.727421   -0.19, -1.00
b81ad769-688a-4d02-ae7b-a64f0984b101  Server 1  False              False               0.630726   -0.17, 0.99
52cbd8ec-b063-40c5-a73e-a51291347e8f  Server 2  False              True                0.789554   0.17, -1.00
```
