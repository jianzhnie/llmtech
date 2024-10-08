# 自主网络防御训练环境Cyberwheel

## 摘要

网络防御者面对其网络遭受的攻击频率和规模感到不堪重负。随着攻击者利用人工智能自动化他们的工作流程，这个问题只会变得更加严重。自主网络防御能力可以通过自动化操作和动态适应新型威胁来帮助防御者。然而，现有的训练环境在泛化能力、可解释性、可扩展性和可迁移性等方面存在不足，这使得训练出在真实网络中有效的智能体变得难以实现。在本文中，我们朝着创建自主网络防御智能体迈出了重要一步——我们提出了一个名为Cyberwheel的高保真训练环境，它包括模拟和仿真能力。Cyberwheel简化了训练网络的定制，并允许轻松重新定义智能体的奖励函数、观测空间和行动空间，以支持对智能体设计新方法的快速实验。它还提供了对智能体行为的必要可见性，以便于智能体评估，并提供了足够的文档/示例来降低入门门槛。作为Cyberwheel的一个示例用例，我们展示了在模拟中训练一个自主智能体部署网络欺骗策略的初步结果。

## 介绍

正如先前的工作[13, 19]所指出的，现有的自主网络防御训练环境在提供必要的保真度方面不足，无法训练出能够良好泛化或有效迁移到操作网络中的智能体。共享训练数据和环境对于推动深度学习领域的进步至关重要，正如我们在2000年代初的ImageNet和计算机视觉领域所看到的那样[1]。因此，利用深度强化学习创建自主网络智能体的研究者必须开发共享的、高保真度的训练环境。在本文中，我们介绍了一个名为Cyberwheel 的高保真训练环境，旨在为自主网络研究社区提供一个可扩展、灵活且可适应的共享资源。

Cyberwheel的模拟环境建立在强大的网络定义代码和配置文件上，支持快速在不同的拓扑大小和配置上进行实验。仿真环境使用了Firewheel[4]测试平台的修改版本，这是一个建立在QEMU上的实验平台，支持实验可重复性，并能够实例化超过10万个虚拟节点的网络。

我们训练环境的目标是帮助社区开发可在实时网络中部署的自主防御智能体，这些智能体

- (a) 能够适应各种网络大小和拓扑结构
- (b) 能够适应不同的对手策略。

第2节提供了现有训练环境不足的概述。

第3节描述了Cyberwheel的属性，这些属性解决了这些不足。

第4节提供了Cyberwheel模拟器中的一个具体用例，其中蓝色智能体被训练使用网络欺骗策略。

我们计划通过整合更多的蓝色智能体动作和策略，启用基于强化学习的红色智能体，并在我们的仿真环境中进行全规模测试，来进一步构建这项初步工作。在等待组织批准和Firewheel的（即将到来的）公开发布后，我们计划开源我们的环境，以促进社区开发和反馈。



## 背景

虽然已经有很多工作致力于利用强化学习实现自主网络安全[1, 3, 5-12, 14, 16, 20]，但目前仍然缺乏一个能够同时支持模拟和仿真的标准化开源环境。当我们开始研究使用强化学习的自主网络防御时，我们尝试了awesome RL for cyber 仓库中列出的几个环境。因为我们打算同时使用模拟和仿真，所以我们首先尝试了Farland[11]，这是通过工作协议获得的。然后，我们测试并扩展了CybORG[18]环境，它已经在几篇研究论文中使用，并且在下面更详细地讨论，因为Vyas等人[19]强调它是训练自主网络智能体的较好的开源环境之一，尽管它缺乏仿真。我们还检查了其他几个可用环境的属性，但发现它们在以下一个或多个方面存在不足：

- 糟糕的文档加上过时的导入和缺乏如何扩展环境的示例，通常与死代码结合在一起，使得环境难以使用。
- 环境定义、实验网络、观测空间和智能体行为的方式复杂且难以清晰地扩展。
- 论文或ReadMe中提到的功能没有实现。例如，CybORG提到了仿真和可视化工具，但这两个都没有包含在仓库中。
- 环境过于专注于特定问题，不能轻松扩展以回答其他研究问题。
- 环境对于少于100个节点的环境运行良好，但不容易扩展到更大的实验，或者不支持并行运行实验。
- 环境缺乏训练可以迁移到真实网络中的智能体所需的粒度。环境可能过于抽象，或者粒度太细。当粒度太细时，环境通常是为了一组小规模实验而编写的，需要进行重大重构才能修改。
- 对研究社区不开放。

虽然有像DARPA CASTLE计划这样的努力正在进行中以构建更好的训练环境，但其结果尚不确定，他们的工作尚未公开可用。因此，对我们来说，开发一个高保真度的模拟和仿真环境来支持我们的研究变得必要，这就是本文的主题。这个环境被称为Cyberwheel。在本节的其余部分，我们提供了我们尝试扩展CybORG的经验概述，然后提供了我们在第4节示例中使用的强化学习方法的简短入门。

### 2.1 CybORG

网络作战研究健身房（Cyber Operations Research Gym，简称CyberORG）[18]是一个帮助创建自主智能体以增强网络操作的环境。CybORG提供了一个模拟环境，用于训练网络防御的自主智能体。在我们的工作中，我们最初尝试使用在第二届网络自主性实验场（Cyber Autonomy Gym for Experimentation，简称CAGE）挑战赛[2][17]中提供的修改版CybORG，该挑战赛专门关注于防御企业网络。我们选择CybORG部分是因为，正如Vyas等人[19]所指出的，它已经包含了一些理想属性，比如支持新的智能体定义、开源且有文档支持、有潜力支持多智能体强化学习（Multi-Agent Reinforcement Learning，简称MARL），并且具有明确定义的状态。CAGE挑战赛2中使用的CybORG环境，尽管是自主网络防御社区的巨大资源，但存在一些不足，导致我们编写了自己的模拟环境：

(1) 它不能在规模上创建多样化的网络拓扑。

(2) 红色智能体（攻击方智能体）的多样性或特异性不足。

(3) 缺少功能性的仿真环境。

(4) 缺少可视化工具来观察智能体行为，这对于可解释性至关重要，以评估和诊断智能体行为的潜在问题。

(5) 观测空间扩展性差，难以转化为真实网络，并做出了关于检测概率的不切实际的假设。修改观测空间以满足我们的需求并不简单。

(6) 死代码和缺乏准确的文档/示例。没有得到积极的维护。

尽管有几篇论文确实使用了CybORG，例如Wolk等人[21]的论文，但他们对CybORG的修改是微小的。他们没有改变拓扑结构，也没有对观测空间或行动空间做出重大改变。我们复制了他们的工作，然而这些对CybORG的微小改变并没有让我们能够回答我们想要提出的研究问题。

### 2.2 强化学习

强化学习是一种机器学习技术，旨在解决通过与环境交互学习以实现长期目标的挑战。例如，这些长期目标可能包括安全地驾驶直升机到达预定目的地、赢得游戏、高效地管理电站或保护网络免受网络攻击。强化学习的终极目标是创建“通用”智能体，其中通用智能可能被定义为“智能体在广泛环境中实现目标的能力”。在高层次上，强化学习智能体通过一个“策略”函数将情况（状态）映射到动作，以最大化数值奖励信号。奖励可能在频率（游戏结束时的单一赢/输奖励或机器人行走时重复的“距离目标”奖励）和复杂性上有所不同。有许多强化学习方法，在本文中，我们使用近端策略优化（Proximal Policy Optimization，简称PPO）来展示我们环境的有用性。

近端策略优化（PPO）是一种使用演员-评论家方法的强化学习方法。演员在环境中选择动作，评论家帮助演员选择将产生最大奖励的动作。在PPO中，演员首先通过与环境交互收集数据。然后，评论家使用这些收集到的数据估计各种动作的优势。利用评论家从环境中学到的知识，演员随后创建一个替代目标函数，它优化这个函数以更新其策略（策略是演员用来选择动作的方法）。然后重复此过程，以提高智能体实现的整体奖励。

PPO被广泛采用，因为它具有高样本效率，倾向于收敛到接近最优解，无需超参数调整即可在各种问题/领域中工作，并通过对单个策略更新的大小进行限制来实现稳定的学习曲线。PPO通过使用一个裁剪的替代目标函数限制单个策略更新的大小，该函数限制了在给定策略下某个特定动作变得远更可能或远不太可能的影响。裁剪的替代目标函数还允许在不引起过大策略更新的情况下，对来自环境的一组样本进行多次梯度上升，这提高了样本效率。有关PPO的详细信息，请参见Schulman等人[15]的工作。



## CYBERWHEEL 训练环境设计

在本节中，我们描述了构成Cyberwheel的模拟环境（第3.1节）和仿真环境（第3.2节）。

### 3.1 模拟环境

#### 3.1.1 网络设计

Cyberwheel网络模拟部分的主要目标是在现实性、配置便捷性和性能之间找到一个实用的平衡。这一目标的动机是通过在合理的时间内为智能体提供大量现实的训练经验来优化训练时间。向模拟环境添加过多的细节会损害情节运行时间并使配置变得困难，从而降低随时间的训练量，并最终损害最终智能体的性能。包含的细节太少则会降低智能体适应现实世界网络场景的能力。

Cyberwheel 网络由路由器、子网和主机组成，每个组件都作为 networkx 图中的一个节点表示。这些组件共享一些共同属性，如默认路由、路由表和防火墙。

##### 路由器

Cyberwheel 路由器的主要作用是在子网之间路由流量。对于连接到路由器的每个子网，都有一个相关的路由器接口，以便路由器可以在所述子网上进行通信。当模拟流量尝试通过路由器时，将分析防火墙状态，如果没有找到匹配的规则，则流量将被丢弃。

##### 子网

子网代表一个单一的广播域。它们跟踪连接的主机、使用的IP，并可能充当 DHCP 服务器，以避免手动配置每个主机的 IP。

##### 主机

主机可以以几种不同的方式进行配置。除了前面提到的共同属性外，每个主机还拥有以下属性：父子网、定义的服务列表，以及表示主机是否是诱饵以及是否已被红色智能体渗透的属性。

主机还有一个可选的主机类型属性，可以被看作是一个模板，定义了常见的主机服务和其他主机配置。使用此属性可以以一种可预测和可复现的方式轻松配置大量的主机。在使用主机类型属性时，仍然可以进行进一步的定制。最后，如果需要为特定用例提供额外的粒度，则上述每个对象都是在容易扩展的类中定义的。

####  3.1.2 场景配置和生成

在CybORG中，网络是通过YAML文件配置的，这些文件由模拟器解析并转换为场景。这些场景对象包含网络模拟的起始参数，包括子网、主机、用户账户、进程、操作系统以及与特定主机相关的版本信息。这种粒度水平对于转换到仿真环境非常有用。YAML文件还定义了每个智能体可用的动作空间、红色智能体策略特征和要使用的奖励函数。

为了允许更临时的网络生成，我们为Cyberwheel创建了一个脚本，该脚本能够生成网络的YAML文件。此脚本可以定义路由器、子网、主机、主机之间的接口、路由和防火墙。所有这些定义会自动生成一个YAML文件，并可立即由Cyberwheel使用。这允许采样多样化的不同网络进行训练。例如，图1中显示的10主机网络就是使用此脚本自动生成的。也可以生成更大的网络。我们使用它创建的最大网络拥有100万个主机和2000个子网。

随着我们不断向模拟器添加功能，我们要求任何额外的起始条件（如蓝色智能体的奖励）在YAML文件中可配置。通过引入这一要求，每组配置文件都作为生成每个蓝色智能体策略的详细日志。这对于蓝色智能体策略的元分析将是非常宝贵的。此外，通过使生成特性可配置，我们简化了创建和评估训练课程的未来工作。

#### 3.1.3 警报和探测器

现有环境需要智能体和观察空间直接观察网络。这在将智能体移植到真实网络时造成了重大的不足和挑战。首先，使用以这种方式训练的智能体将需要在网络中的每个设备上安装和管理自定义工具和传感器，以提供智能体对网络采取行动和观察的能力。除了最初的部署，对智能体及其动作和观察空间的增加和更改将需要在整个网络中完全重新部署工具，造成大量的技术债务。其次，也许更重要的是，通过要求智能体解释原始网络活动，智能体除了提供动作建议外，还必须充当网络入侵检测器。这增加了智能体的“认知负荷”，并阻止采用者充分利用他们现有的网络安全工具投资。因此，我们的环境提供了基于现有探测器观察结果训练智能体的能力，通过摄取警报，并提供了一个与流行的网络检测工具类别（如主机和网络入侵检测系统）整齐对齐的观察空间。要移植到真实环境，采用者只需要提供从他们的警报格式到智能体观察空间的转换层。

每个红色动作都与完美检测该动作的外观配对，直到我们环境的模拟保真度的限制。例如，完美的警报包括哪些IP、端口和服务受到动作的影响，以及与动作相关的MITRE ATT&CK ID。然后警报被传递给探测器，探测器可能会确定性地或随机地对警报应用噪声、完全过滤掉它们，或者包括虚假的正面警报。例如，可以定义一个“好”的模拟NIDS探测器，它从智能体中过滤掉基于主机的活动，“检测”网络基础的活动通过将它们传递给智能体，并且向智能体传递少量虚假的网络基础警报。可以为单个环境定义多个探测器。我们的设计的模块化允许轻松交换动作空间、探测器和观察空间。

#### 3.1.4 蓝色智能体

在我们的环境里，蓝色智能体需要从RL智能体那里采取一个动作，将该动作映射到蓝色动作，执行该动作，并返回一个包含动作名称、特定动作的唯一标识符以及该动作是否成功的元组。蓝色智能体还定义了一个奖励映射，将其动作映射到奖励。蓝色智能体用来确定这个映射的方法可能因智能体而异，这取决于特定智能体的要求。这个映射在创建一个新的蓝色智能体时解析的YAML文件中配置。

#### 3.1.5 奖励

动作可以有两种类型的奖励：即时和重复。即时奖励是在执行动作的同一步骤中获得的。重复奖励则是在同一步骤和当前情节结束或采取另一个移除此奖励的动作之前的所有未来步骤中计算的。我们纳入这两种奖励是为了模拟执行动作的前期成本和该动作持续的影响。部署一个诱饵主机，如蜜罐，就是一个既有即时奖励又有重复奖励的蓝色动作的例子。启动诱饵的成本由即时奖励表示，维持它则由重复奖励表示。红色动作也可以利用即时和重复奖励。例如，影响服务器可能有一个重复奖励。

为了计算每一步的奖励，需要定义一个奖励计算器。这个计算器需要将红色和蓝色动作映射到奖励值。然后需要定义一个实际计算步骤奖励的方法。通常，这需要红色和蓝色动作的执行，红色和蓝色动作是否成功，以及红色动作是否触发了探测器的警报。其他操作，如跟踪重复奖励，可能由奖励计算器定义。计算器还必须定义一个重置方法，指定每当环境重置时如何重置奖励计算器。

#### 3.1.6 红色动作

红色智能体定义了一个正在攻击配置网络的对手。在模拟的当前阶段，它是确定性的并且基于逻辑，其决策中没有涉及RL训练，尽管我们的代码编写方式使得启用基于RL的红色智能体变得简单。红色智能体经历每个主机定义的动作的杀伤链，以确定它应该对给定主机采取的动作。我们当前的杀伤链实现有以下四个杀伤链阶段：发现、侦察、权限提升和影响。

红色动作源自原子红队（ART）技术和MITRE ATT&CK战术和策略。目前有295个定义的ART技术，尽管其中许多还没有完全写入游戏的逻辑。现在，基本动作和杀伤链阶段已经实现到红色智能体的逻辑中。这些基本动作被定义为杀伤链阶段。它们是更高级别的，可以根据特定主机的各种属性归因于多种不同的ART技术，例如相关的漏洞、可供利用的服务或运行的操作系统。

ART技术定义包含以下属性：

- 相关的Mitre ID（例如 "T1055.011"）
- 技术名称（例如 “Extra Window Memory Injection”）
- MITRE技术ID（例如 “attack-pattern–0042a9f5-f053-4769-b3ef-9ad018dfa298”）
- 相关的MITRE数据组件（例如 [’OS API Execution’]）
- 使用的杀伤链阶段（例如 [’defense-evasion’, ’privilege-escalation’]）
- MITRE企业缓解措施
- 描述
- 原子测试
- 相关的常见漏洞和暴露（CVEs）列表
- 相关的常见弱点枚举（CWEs）列表

原子测试定义了与给定技术相关联的数据，例如攻击支持的平台、所需的依赖项，以及执行攻击所需的实际shell命令。这些属性有助于为红色代理提供仿真必需品，以反映我们的仿真环境中的模拟器，例如在模拟主机中实现CWEs和CVEs，并使用原子测试数据实现红色动作作为shell命令。



#### 3.1.7 可视化工具。

我们创建了一个可视化工具，使自主智能体创建者能够通过重放和观察游戏剧集来诊断和理解智能体行为。目前使用Weights and Biases API获取我们训练好的模型。这个评估脚本允许为环境设置各种参数，例如：

- 红色智能体类型
- 网络配置YAML
- 诱饵类型配置YAML
- 主机信息配置YAML
- 诱饵的最小数量
- 诱饵的最大数量
- 奖励缩放值
- 奖励函数

该脚本在给定参数定义的环境中评估模型。评估过程中的元数据和可视化以CSV和PNG文件的形式保存在本地。

对于可视化，我们利用了graphviz（一个开源图形可视化软件）和它的python模块pygraphviz，以层次化的绘图布局清晰地显示我们的网络。可视化展示了每个情节步骤的网络状态信息，让我们可以看到每个主机所处的杀伤链阶段，以了解整个模拟中的入侵程度。它还允许用户看到红色智能体当前正在执行攻击的主机。

#### 3.1.8 性能。

我们测试了我们模拟器的性能，以确定不同大小网络的每集运行时间。如图3所示，集运行时间随着主机数量的增加而线性增加。我们目前正在训练期间并行运行128个环境，如果硬件合适，这个数量可以增加，每集100步。总训练运行时间取决于给定实验的环境配置（除了并行环境数量）——智能体定义、每集步骤、奖励函数、超参数和其他因素。

### 3.2 仿真环境

仿真器提供了一个高保真度的环境，可以在真实世界场景中评估在模拟器中训练的智能体。理想情况下，一个训练有素的智能体在仿真器中的表现应该和在模拟器中一样出色，并且能够适应新的、未见过的场景，几乎不需要额外的训练。目前，相关研究中使用的仿真器数量有限，包括CybORG [18]、Farland [11] 和 NASimEmu [8]。然而，这些仿真器要么不能满足可扩展性要求，要么不是开源的。此外，虽然可以使用虚拟机（VM）管理器如Vagrant 11来模拟场景，但随着拓扑规模和复杂性的增长，手动配置和管理实验变得越来越困难。基于这些原因，我们选择使用Firewheel [4]。

Firewheel是由桑迪亚国家实验室（Sandia National Laboratory，简称SNL）设计和开发的一个测试平台。它提供了基础设施，用于模拟大规模现实网络拓扑和行为，并进行可重复的实验。Firewheel附带了一个广泛的模型组件库（例如主机、路由器和交换机），以加速构建常见拓扑网络组件。此外，Firewheel提供了工具来观察和收集实验数据。仿真器使用基于内核的虚拟机12（Kernel-based Virtual Machine，简称KVM）来虚拟化主机机器，并使用Minimega13，这是SNL的另一个工具，来启动和管理VM。

我们的仿真架构和环境，在图4中进行了说明，受到了[8]中所展示工作的影响。该方法涉及使用场景配置YAML文件；这些文件也是Cyberwheel模拟器部分所使用的。它们定义了网络拓扑和每个主机上运行的服务（例如SQL服务器、电子邮件、FTP等）。在实验开始之前，场景转换器解析配置文件并生成一个单独的插件文件，这是一个特殊的文件，Firewheel用它来加载和初始化VM。在仿真器内部，每个VM都运行着一个监控工具，用于捕获事件并生成日志。此外，我们有一个指定的VM用于运行安全信息和事件管理（Security Information and Event Management，简称SIEM）解决方案。截至本研讨会论文，我们使用Sysmon14进行监控，使用Elastic Stack15作为SIEM；然而，随着我们进一步开发仿真环境，这些工具可能会发生变化。

在插件文件生成并且VM初始化后，场景实验就可以开始了。每一步动作控制器首先从智能体那里接收一个动作，并将其映射为攻击者或防御者可以执行的命令。在我们的实现中，命令定义在Python或Shell脚本中。执行后，生成日志，由SIEM聚合和处理。在步骤结束时，观察转换器查询SIEM并将查询响应转换为观察空间向量。该向量被智能体消耗，以便选择一个新的动作，此时开始一个新的步骤。

我们使用Ansible16来部署我们的Firewheel仿真环境。Ansible是一个开源的配置管理自动化工具。与Firewheel类似，Ansible 具有声明性地定义给定配置状态的能力。此外，用于定义仿真环境的Ansible 模块以幂等方式操作。这提供了验证每个环境的能力，并确保没有无意中偏离期望状态。

## 4 用例网络欺骗

利用Cyberwheel模拟环境，我们评估了一个部署网络中诱饵以延迟或阻止对手的蓝色智能体策略。本节的目的不是就训练智能体的健壮性提出具体主张，而是展示如何在Cyberwheel内构思和运行一个实验。我们详细描述了蓝色智能体、红色智能体和观测空间的设计，以及对不同诱饵类型和gamma超参数的评估。所有实验均使用第2.2节中描述的强化学习算法PPO。

### 4.1 诱饵蓝色智能体和奖励

我们的蓝色智能体部署诱饵以诱使红色智能体攻击诱饵服务器而非实际服务器。蓝色智能体能够在网络中部署诱饵主机。蓝色智能体可以部署的主机类型通过使用YAML配置，定义潜在的诱饵主机类型。这些诱饵对红色智能体来说看起来像普通主机，并可以放置在任何子网上。

蓝色智能体可以执行三种不同的动作。它可以部署一个诱饵、移除一个诱饵，或者不采取任何行动。诱饵可以部署在网络的任何子网上。如果部署的诱饵数量超出指定范围，则奖励计算器通过增加诱饵的重复奖励来惩罚蓝色智能体。这个范围的目的是让蓝色智能体在部署多少诱饵时具有灵活性，同时也防止蓝色智能体创建荒谬数量的诱饵或仅选择不采取任何行动。

可以使用移除动作来移除诱饵。由于同一子网上相同类型的诱饵并不唯一，因此此动作只是从子网上移除一个指定类型的诱饵。如果在子网上没有指定类型的诱饵，则此动作失败，降低了这一步获得的奖励。如果智能体选择不采取任何行动，那么蓝色智能体不会修改网络，并且不会为采取此行动增加奖励。

奖励是通过累加蓝色智能体执行其动作的奖励、红色智能体执行其动作的奖励以及所有当前重复动作的奖励来计算的。所有这些奖励都是负数。然而，如果一个红色动作针对的是诱饵，那么这个红色动作的奖励就是正数。这是因为在这个场景中的目标是“欺骗”红色智能体与诱饵交互，而不是与实际服务器交互。

### 4.2 红色智能体

游戏开始时，红色智能体在网络中的一个随机用户主机上拥有一个立足点。这以主机上具有用户权限的进程形式表示。红色智能体可以执行的动作以杀伤链的阶段形式呈现。

发现阶段的杀伤链，如果给定一个主机作为目标，允许红色智能体在目标主机的子网上运行ping扫描，或者对目标主机进行端口扫描。在子网上运行ping扫描将为红色智能体提供子网上所有主机的IP地址列表，并将这些数据添加到红色智能体的知识库中。对目标主机运行端口扫描将为红色智能体提供有关目标主机上当前运行的服务的信息。

侦察阶段的杀伤链，如果给定一个主机作为目标，允许红色智能体收集有关目标主机的关键信息。目前，这包括漏洞列表、主机类型和任何相关的主机接口。以CVE形式拥有主机上的漏洞列表可以让我们使用ART技术扩展红色智能体动作，并帮助通过检查主机是否具有攻击利用的漏洞来确定攻击是否会成功。在这种情况下，主机接口指的是主机具有IP地址和连接到另一主机的能力，无论它是否在同一子网上。这些接口在网络配置YAML文件中定义，它们使红色智能体能够在存在相应接口的情况下在子网之间移动。

因为红色智能体的目标是影响服务器，所以它可以利用从侦察阶段收集到的主机类型来帮助决定是否移动到不同的主机进行攻击。当红色智能体找到一个服务器主机时，它将使用横向移动动作移动到该主机，这将把恶意进程添加到服务器上，使其成为未来攻击和动作的源头。如果它找到一个用户主机，它将留在主机上并继续寻找服务器。

当红色智能体移动到服务器时，它将使用权限提升来扩大在主机上的权限。这涉及将恶意进程的权限级别从用户提升到根权限。完成此动作后，红色智能体可以使用目标主机上的影响阶段杀伤链，这是它在游戏中的最终目标。

默认的KillChain智能体的目标是针对服务器主机。一旦找到并运行了服务器上的影响，它将一直影响它，直到游戏结束。然而，重复影响智能体的动机是影响给定网络上的所有服务器，以创建网络的停机时间。因此，每次影响都会在游戏的每个后续回合中增加重复奖励。在影响服务器之后，红色智能体将继续探索网络，以攻击和影响其他服务器，而每次影响的效果会在后台继续累积。这允许红色智能体有更多机会攻击网络，同时也迫使蓝色智能体适应不同的策略，最大化其奖励，并减少红色智能体行动的成本。因此，即使在最初的影响之后，蓝色智能体仍然有动机继续保卫网络，以防止进一步的损害。

### 4.3 探测器

我们为使用我们的诱饵智能体设计了两种类型的探测器。第一种是当红色行动的目标是诱饵时创建警报的探测器。换句话说，它检测当主机与诱饵交互时。这种设计背后的理念是，未被渗透的主机不应该与诱饵交互。如果诱饵和主机之间有交互，那么源主机很可能已经被渗透了。

第二种类型的探测器是概率性探测器。每个红色行动都有一组技术与之相关，以模拟行动在现实世界中的表现方式。我们使用MITRE ATT&CK的战术和技术作为分配技术给行动的基础。这个探测器检查红色行动使用的技术，并将其与自己的技术集进行比较。对于红色行动使用的探测器集中的每种技术，探测器都有可能为该交互创建警报。探测器可以检测的技术在YAML文件中定义，指定了技术的名称及其检测概率。

这种可配置性允许为各种场景创建概率性探测器。我们创建了两个配置文件，以模拟基于网络和主机的入侵检测系统。

### 4.4 观测空间

因为我们的用例侧重于欺骗，蓝色智能体需要将诱饵放置在最有可能被访问的地方。这要求我们的观测空间包括一些探测器发现红色智能体所在位置的历史记录。我们观测空间的大小是网络中非诱饵主机数量的两倍。观测空间的前半部分表示当前步骤所做出的观测。它由网络中每个非诱饵主机的单个布尔值组成。如果探测器在这一步为该主机生成了警报，那么它在观测空间中的相应值将被设置为1。否则，它是0。观测空间的后半部分是保留给历史的。像前半部分一样，观测空间的历史部分为每个非诱饵主机都有一个单独的布尔值。然而，它跟踪的是在整个情节中探测器是否曾经为一个主机生成过警报。这些信息旨在帮助智能体学习红色行动倾向于发生在哪些主机上，并将诱饵放置在红色智能体的道路上，使其最为有效。

### 4.5 评估

下面描述的评估都是在具有3个子网的13主机网络上进行训练的，一个子网主要用于用户主机，一个用于服务器主机，还有一个用于DMZ主机。

#### 4.5.1 评估诱饵类型

我们测试了蓝色智能体的行为，当它只能部署用户类型的诱饵、只能部署服务器类型的诱饵，或者能够部署任一类型的诱饵时。这些测试使用默认参数和1-4个诱饵的范围进行了训练和评估。

仅部署用户诱饵时，它会在DMZ子网上部署诱饵以延迟红色智能体几个步骤，并获得它给予的边际正奖励。然而，不管蓝色智能体采取什么行动，红色智能体最终都会继续攻击服务器。仅部署服务器类型的诱饵时，它会在DMZ子网和用户子网上部署诱饵。这样，它有很好的机会在攻击过程中改变红色智能体的路径。因为诱饵是服务器类型的，红色智能体会专注于攻击它而不是网络中的实际服务器，这允许它被捕捉。给蓝色智能体选择部署用户或服务器诱饵的选项并没有改变智能体的行为，因为它学会了总是选择部署服务器诱饵来实际阻止红色智能体。

#### 4.5.2 评估Gamma变量

Gamma变量，也称为折扣因子，决定了智能体在训练过程中对长期奖励的重视程度。接近0的Gamma值倾向于优先考虑短期奖励，而接近1的Gamma值倾向于优先考虑长期奖励。我们使用0.1-0.9范围内的Gamma值训练和评估了我们的RL智能体，以确定其对行为和奖励的影响。

对于0.1到0.6之间的Gamma值，训练出的蓝色智能体都达到了相同的结果：它们没有学到任何有用的技巧来欺骗红色智能体攻击诱饵。当Gamma值为0.7时，训练出的智能体学会了在用户子网上部署服务器诱饵以转移红色智能体的注意力。当Gamma值为0.8时，它学会了在DMZ或用户子网上部署这些服务器诱饵，但主要在用户子网上而不是DMZ子网上部署它们。在这个阶段，它可以变成概率性的，以确定红色智能体首先找到诱饵服务器还是真实服务器，假设它两者都可以访问。对于0.9的Gamma值，它展示了类似的行为，但主要在DMZ子网上而不是用户子网上部署诱饵，这种方法使它在所有测试的Gamma值中获得了最高的奖励。

## 5 未来工作

在这项初步工作中，我们总结了现有自主网络防御智能体训练环境的不足，介绍了Cyberwheel模拟和仿真环境的设计，并分享了使用Cyberwheel模拟环境训练蓝色智能体智能部署诱饵的结果。展望未来，我们计划与自主网络研究社区合作，扩展Cyberwheel以支持额外的蓝色和红色智能体动作和目标、基于强化学习的红色智能体，以及将智能体从模拟转移到仿真再到真实网络时的最小重新训练。

为了支持这一更广泛目标，我们将继续提高Cyberwheel的可扩展性和可用性，同时开展自己的研究并接收其他用户的反馈。我们特别感兴趣的研究领域是，在训练大型和/或更复杂的网络环境中的智能体时，使用多智能体方法和课程学习。

我们认识到，为了使Cyberwheel成为一个真正有价值的工具，它需要能够适应多样化的研究目标和需求。这包括能够模拟和仿真更广泛的攻击和防御策略，以及能够处理更大规模和更复杂的网络拓扑结构。我们还将探索如何提高环境的可访问性，特别是对资源有限的研究人员和教育机构。

此外，我们计划通过集成更先进的强化学习算法和技术来提高红色智能体的行为复杂性，使其能够执行更真实的攻击模式。这将包括开发能够适应不断变化的网络环境和防御措施的自适应对手。

最后，我们认识到，为了推动这一领域的进步，需要有一个共享的、标准化的环境，研究人员可以在其中测试和评估他们的算法和策略。我们致力于使Cyberwheel成为这样一个环境，并鼓励社区贡献代码、反馈和用例，以不断改进平台。

我们相信，通过与社区合作并专注于建立一个强大、灵活和可扩展的训练环境，我们可以为自主网络防御领域做出重要贡献，并帮助推进这一领域的发展。
