## 强化学习+自主网络防御

### 挑战

在网络安全中，实际上有无限多可能的动作和观察结果，而这些观察结果可能部分隐藏起来，或者它们甚至可能不真实，作为欺骗或错误的一部分。

每台计算机、路由器和设备上的每个可配置设置都是一个潜在的动作。此外，网络上流动的或计算机上存储的每一点数据都可能是需要观察的重要信息。例如，十台计算机，每台有十个软件，每个软件有十个可能的安全设置需要配置，导致一千种可能的动作——大约是围棋的三倍。动作和观察的数量呈指数级增长，很快变得难以管理。

自主网络防御的一个主要挑战是选择任务和构建足够复杂且有用，但在动作和观察的数量方面足够小以便于管理的训练环境。自主网络防御的一个理想化愿景是拥有一个巨大的模型，可以执行网络防御者可以执行的所有动作，同时观察整个网络的所有数据，但这将需要一个看似不可能的大数量的动作和观察。

另一种愿景是构建许多单独的智能体，每个智能体都针对更受限制的任务进行训练，动作和观察的数量较少。这些智能体可以协同工作并在彼此之间传递信息。例如，一个智能体可能只将计算机视为可以被感染或清洁的黑盒，并且它只能执行少数动作来隔离或修复它们。另一个智能体可能在这些计算机上工作，观察所有正在运行的进程和用户行为。它可以决定是否终止其中一些进程或锁定用户，并且它可以告诉第一个智能体计算机是否被感染。



###

[下载 PDF](https://www.mdpi.com/2079-9292/13/3/555/pdf?version=1706616311)

*设置*

[订购文章重印本](https://www.mdpi.com/2079-9292/13/3/555/reprints)



开放存取文章

# 利用深度强化学习进行网络攻击模拟以增强网络安全

经过 吴尚浩1，金贞允2，罗在勋3和朴钟律2、*[![奥西迪](https://pub.mdpi-res.com/img/design/orcid.png?0465bc3812adeb52?1718874496)](https://orcid.org/0000-0002-4878-4129)



1

釜庆国立大学计算机工程与人工智能系，韩国釜山 48513

2

首尔国立科技大学应用人工智能系，首尔 01811，韩国

3

电子与电信研究所，韩国大田 34129

*

通讯作者。

*Electronics* **2024，13**（*3*），555；https://doi.org/10.3390/electronics13030555

提交截止日期：2023 年 12 月 19 日 / 修订日期：2024 年 1 月 25 日 / 接受日期：2024 年 1 月 29 日 / 发布日期：2024 年 1 月 30 日

（本文属于特刊《[应用于未来网络技术安全和隐私问题的新方法](https://www.mdpi.com/journal/electronics/special_issues/MKM3C5A9B3)》）

下载*键盘向下箭头*

[浏览图表](https://www.mdpi.com/2079-9292/13/3/555#)



[版本说明](https://www.mdpi.com/2079-9292/13/3/555/notes)



## 抽象的

在当前网络安全威胁日益复杂和频繁的形势下，基于规则的防火墙和基于签名的检测等传统防御机制已被证明不够充分。现代网络攻击的动态性和复杂性需要能够实时发展和适应的高级解决方案。进入深度强化学习 (DRL) 领域，这是人工智能的一个分支，已有效地解决包括网络安全在内的各个领域的复杂决策问题。在本研究中，我们通过实施 DRL 框架来模拟网络攻击，并借鉴真实场景来增强模拟的真实性和适用性，从而推动该领域的发展。通过精心调整 DRL 算法以适应网络安全环境的细微要求（例如自定义奖励结构和行动、对抗性训练和动态环境），我们提供了一种定制方法，大大改进了传统方法。我们的研究在反映现实世界网络威胁的受控模拟环境中，对三种复杂的 DRL 算法（深度 Q 网络 (DQN)、演员-评论家和近端策略优化 (PPO)）与传统 RL 算法 Q 学习进行了彻底的比较分析。研究结果令人震惊：演员-评论家算法不仅以 0.78 的成功率超越其同类算法，而且还表现出卓越的效率，需要最少的迭代次数（171）即可完成一集并获得最高平均奖励 4.8。相比之下，DQN、PPO 和 Q 学习略有落后。这些结果强调了选择最适合网络安全模拟的算法的关键影响，因为正确的选择会带来更有效的学习和防御策略。演员-评论家算法在本研究中的出色表现标志着朝着开发能够应对日益复杂的网络威胁格局的自适应智能网络安全系统迈出了重要一步。我们的研究不仅提供了一个模拟网络威胁的强大模型，而且还提供了一个可适应各种网络安全挑战的可扩展框架。

关键字：

[网络攻击模拟](https://www.mdpi.com/search?q=cyber-attack+simulation)；[人工智能](https://www.mdpi.com/search?q=artificial+intelligence)；[网络安全](https://www.mdpi.com/search?q=cybersecurity)；[深度强化学习](https://www.mdpi.com/search?q=deep+reinforcement+learning)



## 1. 简介

为了防御各种网络攻击，红队演习经常用于评估网络系统防御系统的反应效果 [ [1](https://www.mdpi.com/2079-9292/13/3/555#B1-electronics-13-00555) ]。为了模拟复杂的持续性威胁并测试系统对高级攻击者所采用的各种策略、方法和技术的防御能力，这些演习经常使用对手配置文件 [ [2、3](https://www.mdpi.com/2079-9292/13/3/555#B2-electronics-13-00555) ]。然而，由于红队演习需要特定的人类专业知识，因此可能耗时且昂贵。人们创建了模拟器来自动化该过程并加快攻击模拟过程，以提高红队的有效性 [ [4](https://www.mdpi.com/2079-9292/13/3/555#B3-electronics-13-00555) ] [。](https://www.mdpi.com/2079-9292/13/3/555#B4-electronics-13-00555)人们开发了其他工具，如执行有效载荷、启用脚本和登台框架，以简化红队流程并为人类专家提供支持 [ [3](https://www.mdpi.com/2079-9292/13/3/555#B3-electronics-13-00555) ]。尽管这些工具旨在协助对演习的规划和决策阶段至关重要的人类专家，例如在攻击模拟活动的各个阶段设计策略和程序。

随着网络攻击变得越来越复杂和动态，网络攻击模拟的重要性日益增加 [ [5](https://www.mdpi.com/2079-9292/13/3/555#B5-electronics-13-00555) ]。依赖于预先确定的规则和签名的传统安全系统往往不足以防御这些复杂且自适应的攻击者 [ 6 []](https://www.mdpi.com/2079-9292/13/3/555#B6-electronics-13-00555)。相比之下，机器学习 (ML) 模型已成为一种适应性更强、更灵活的网络安全方法 [ [7、8、9 ](https://www.mdpi.com/2079-9292/13/3/555#B8-electronics-13-00555)[] ](https://www.mdpi.com/2079-9292/13/3/555#B11-electronics-13-00555)[。ML](https://www.mdpi.com/2079-9292/13/3/555#B7-electronics-13-00555)[模型可以随着时间的推移而发展，通过从先前的数据](https://www.mdpi.com/2079-9292/13/3/555#B9-electronics-13-00555)[中](https://www.mdpi.com/2079-9292/13/3/555#B10-electronics-13-00555)学习来更好地检测和应对新出现的威胁 [ 10、11、12 ] 。它们可以提供更强大的保护系统，随着危险的变化而变化。在提高联网系统安全性方面，ML 模型在网络安全中的使用已显示出巨大的潜力[。](https://www.mdpi.com/2079-9292/13/3/555#B12-electronics-13-00555)

我们的研究目标是证明深度强化学习 (DRL) 可以有效地用于网络安全的网络攻击模拟。我们通过进行红队模拟并评估 DRL 代理的性能来实现这一目标。通过使用模拟来模拟网络攻击，我们可以在受控环境中评估 DRL 算法的性能。我们的结果表明，DRL 代理能够学习和执行有效的策略来渗透模拟的网络安全环境。

我们的论文结构如下。在[第 2 部分](https://www.mdpi.com/2079-9292/13/3/555#sec2-electronics-13-00555)中，我们整理了相关工作。在[第 3 部分](https://www.mdpi.com/2079-9292/13/3/555#sec3-electronics-13-00555)中，我们概述了方法论、深度 Q 网络 (DQN)、演员-评论家、近端策略优化 (PPO) 和包括 MITRE ATT&CK [ [13 \] 在内的模拟设置。](https://www.mdpi.com/2079-9292/13/3/555#B13-electronics-13-00555)[第 4 部分](https://www.mdpi.com/2079-9292/13/3/555#sec4-electronics-13-00555)展示了实验结果以及获得的奖励、代理的攻击成功率以及它们在每次攻击中进行的迭代次数。我们在第[5 部分分析了调查结果，然后在](https://www.mdpi.com/2079-9292/13/3/555#sec5-electronics-13-00555)[第 6 部分](https://www.mdpi.com/2079-9292/13/3/555#sec6-electronics-13-00555)概述了我们的结论。最后，[第 7 部分](https://www.mdpi.com/2079-9292/13/3/555#sec7-electronics-13-00555)展示了局限性和未来的工作。

## 2.相关工作

网络安全中的机器学习变得越来越普遍，这增加了基于机器学习的软件遭受网络攻击的风险 [ [14](https://www.mdpi.com/2079-9292/13/3/555#B14-electronics-13-00555) , [15](https://www.mdpi.com/2079-9292/13/3/555#B15-electronics-13-00555) ]。网络安全的网络攻击模拟使用机器学习技术对系统进行模拟和建模假设的网络攻击，以训练机器学习模型实时检测和应对这些攻击。这一策略有助于研究人员增强整体网络安全防御能力，并为任何网络威胁做好准备。另一方面，传统的基于机器学习的应用程序存在缺陷，因为它们通常使用历史数据进行训练，可能不太具有普遍性 [ [16](https://www.mdpi.com/2079-9292/13/3/555#B16-electronics-13-00555) , [17](https://www.mdpi.com/2079-9292/13/3/555#B17-electronics-13-00555) , [18](https://www.mdpi.com/2079-9292/13/3/555#B18-electronics-13-00555) ]。随着人工智能的发展，人工智能辅助或自治的人工智能红队的想法也浮出水面。在这种情况下，人工智能可以利用通过人工智能训练学到的卓越决策技能，针对复杂的网络系统创建人类红队专家以前无法想到的新颖攻击策略 [ [19](https://www.mdpi.com/2079-9292/13/3/555#B19-electronics-13-00555) , [20](https://www.mdpi.com/2079-9292/13/3/555#B20-electronics-13-00555) ]。这可以改变网络安全中的网络攻击模拟，并增强防御网络攻击的效力。

鉴于当前的情况，我们倾向于使用深度强化学习 (DRL) 方法训练代理在基于模拟的网络中识别和优化攻击策略，这是对传统 ML 模型的改进。因此，我们将能够找到更可靠的解决方案和更好的网络攻击模拟。强化学习 (RL) 是一种可以帮助开发自主代理的方法，能够在具有挑战性和不可预测的情况下做出最佳连续决策。由于 OpenAIGym 等开源学习环境的存在，RL 研究在多个应用领域的潜力得到了扩大 [ [21](https://www.mdpi.com/2079-9292/13/3/555#B21-electronics-13-00555) ]。

近年来，网络安全中网络攻击模拟中的 RL 越来越受欢迎 [ [22、23、24、25、26 ](https://www.mdpi.com/2079-9292/13/3/555#B24-electronics-13-00555)[] ](https://www.mdpi.com/2079-9292/13/3/555#B25-electronics-13-00555)[。](https://www.mdpi.com/2079-9292/13/3/555#B22-electronics-13-00555)由于网络攻击的复杂[性](https://www.mdpi.com/2079-9292/13/3/555#B23-electronics-13-00555)以及建立有效防御措施的难度，研究人员已转向[RL](https://www.mdpi.com/2079-9292/13/3/555#B26-electronics-13-00555)等 ML 方法来创建更具弹性和灵活性的安全系统。使用 RL 算法可以学习防御攻击、改变威胁和改进攻击方法的最佳方法。通过定期进行攻防游戏，RL 代理可以学会检测和防御不同类型的攻击，例如零日漏洞 [ [27](https://www.mdpi.com/2079-9292/13/3/555#B27-electronics-13-00555) ]。在在线应用程序、恶意软件分析和网络入侵检测等一系列情况下，人们发现这种方法对于识别和遏制网络攻击非常有效[ [28、29 ](https://www.mdpi.com/2079-9292/13/3/555#B29-electronics-13-00555)[]](https://www.mdpi.com/2079-9292/13/3/555#B28-electronics-13-00555)。

通过提供能够从经验中学习并实时应对不断发展的网络威胁的自适应自动化防御系统，RL 有可能彻底改变网络安全 [ [30](https://www.mdpi.com/2079-9292/13/3/555#B30-electronics-13-00555) ]。在网络安全领域有效运用 RL 之前，必须解决几个重要问题。最严重的挑战之一是缺乏训练数据 [ [10,31 ](https://www.mdpi.com/2079-9292/13/3/555#B10-electronics-13-00555)[]](https://www.mdpi.com/2079-9292/13/3/555#B31-electronics-13-00555) 。获取足够的数据来有效训练 RL 模型可能具有挑战性，因为激进的网络攻击情况通常不常见且复杂。这可能导致出现无法准确捕捉现实世界事件复杂性的欠拟合模型。

另一个问题是模拟复杂而动态的攻击场景 [ [32](https://www.mdpi.com/2079-9292/13/3/555#B32-electronics-13-00555) ]。由于网络攻击可能非常动态且具有自适应性，因此可能很难创建能够捕捉所有可能攻击方法和策略的精确模型。这可能会导致模型过度拟合并过于集中于特定的攻击场景，从而很难推广到新的或无法预见的攻击场景。此外，网络安全的开源 RL 实验平台很少，这使得研究人员难以解决实际问题并推进 RL 网络安全应用的最新进展 [ [33](https://www.mdpi.com/2079-9292/13/3/555#B33-electronics-13-00555) ]。如果研究人员无法访问广泛而真实的测试环境，他们可能很难开发和测试基于 RL 的创新网络安全解决方案。

在网络安全威胁不断演变的背景下，人们正在齐心协力利用强化学习的潜力来加强网络防御。Ibrahim 等人深入研究了信息物理系统领域，重点关注智能电网安全，采用强化学习增强攻击图通过 SARSA 强化学习技术来查明漏洞 [ [34](https://www.mdpi.com/2079-9292/13/3/555#B34-electronics-13-00555) ]。Dutta 等人通过数据驱动的 DRL 框架推进了这一领域，该框架旨在开发主动的、情境敏感的防御机制，可以动态适应不断变化的对手策略，同时最大限度地减少运营中断 [ [35](https://www.mdpi.com/2079-9292/13/3/555#B35-electronics-13-00555) ]。

其他贡献包括 Applebaum 等人的工作，他们仔细研究了强化学习训练的自主智能体在不同网络环境中的能力 [ [36](https://www.mdpi.com/2079-9292/13/3/555#B36-electronics-13-00555) ]。同样，Elderman 等人探索了在网络环境中进行网络安全模拟，这些环境被建模为具有随机动态和不完全信息的马尔可夫博弈 [ [37](https://www.mdpi.com/2079-9292/13/3/555#B37-electronics-13-00555) ]。他们的工作体现了两个关键智能体（攻击者和防御者）在连续决策场景中的复杂相互作用。

从实践角度来看，微软推出了基于 OpenAIGym 接口的实验研究平台 CyberBattleSim，标志着强化学习在网络安全领域的应用迈出了重要一步 [ [38](https://www.mdpi.com/2079-9292/13/3/555#B38-electronics-13-00555) ]。这个基于 Python 的平台为研究人员开展试验、测试各种假设和建立创新模型以应对紧迫的网络安全挑战奠定了基础。随着技术不断进步，我们可以期待出现更复杂的强化学习实验环境，进一步催化网络安全领域尖端强化学习应用的进步。

本研究通过对 DRL 在网络攻击模拟中的应用进行细致入微的比较分析，为网络安全领域做出了重要贡献，在几个关键领域与先前的研究有所区别。首先，它展示了 DRL 算法在模拟和理解网络攻击机制方面的成功部署，这标志着超越传统网络安全实践的重要一步。与之前的研究不同，这项研究展示了 DRL 代理实时学习和适应的能力，为 DRL 在创建可动态应对复杂网络威胁的先进自动化防御机制方面的可行性提供了证据。

其次，通过利用 MITRE ATT&CK 数据库中的真实网络攻击场景，该研究超越了许多依赖假设或简化模型的现有研究工作。这种方法为 DRL 技术提供了更真实的试验场，增强了结果与现实世界网络安全问题的相关性和可迁移性，从而为 DRL 集成到当前的网络安全应用中铺平了道路。

最后，该研究在详细的模拟环境中对 DQN、actor-critic 和 PPO 算法与传统 RL 算法 Q-learning 进行了比较研究，提供了以前研究未捕捉到的新见解。它强调了基于环境和场景细节的算法选择的重要性，这是开发有效的基于 DRL 的网络安全解决方案的关键因素。总的来说，这些要素将这项研究定位为一项基础研究，为网络安全领域 DRL 应用的未来发展规划了方向，并促进了网络安全方法的战略发展。

## 3.材料和方法

强化学习算法通过获取旨在最大化特定时间段内预期奖励的策略来学习马尔可夫决策过程 (MDP) 的近似解 [ [39](https://www.mdpi.com/2079-9292/13/3/555#B39-electronics-13-00555) ]。与 MDP 类似，强化学习利用状态、动作和奖励组件与环境交互，其中代理根据先前的奖励或惩罚以及其当前状态选择动作。强化学习算法可应用于网络安全等不同领域，其中穷举搜索方法可能无法找到最佳解决方案。

在本研究中，我们利用模拟环境使我们的代理能够与 MITRE ATT&CK 提供的真实网络攻击场景进行交互。通过使用模拟环境，我们能够加速代理的策略学习过程。这种方法使我们的代理能够从经验中学习并实时调整其策略以应对不断变化的网络威胁。使用模拟环境还提供了受控的测试环境，使我们能够评估代理在各种条件下的性能。总体而言，我们的研究表明了在训练代理时使用 DRL 环境的有效性。此外，通过结合 MITRE ATT&CK 提供的真实网络攻击场景，我们能够在具有挑战性和相关环境中评估代理的性能。

#### 3.1. 使用的强化学习算法

#### 3.1.1. 深度 Q 网络（DQN）

当状态变量的概率分布已知时，Q 学习方法用于确定在给定状态下采取的最佳操作。该方法基于对由为每个状态-操作对计算的 Q 值组成的值函数的估计（𝑠𝑡，𝑎𝑡）（𝑠吨，𝐴吨）[ [40](https://www.mdpi.com/2079-9292/13/3/555#B40-electronics-13-00555) ]。代理评估每个状态-动作对的奖励（𝑠𝑡，𝑎𝑡）（𝑠吨，𝐴吨）在每次迭代中𝑡吨将 Q 值初始化为任意实数后。算法的下一步是根据下一个状态-动作对的 Q 值更新 Q 值，𝑄 （𝑠𝑡 + 1，𝑎𝑡 + 1）问（𝑠吨+1，𝐴吨+1）以及即时奖励𝑟𝑡𝑟吨，具有折扣因子𝛾𝛾控制未来奖励对当前奖励的影响，如公式 (1) 所示。Q 学习在强化学习中得到广泛应用，允许代理通过迭代改进其 Q 值来学习各种环境中的最优策略。

𝑄 （𝑠𝑡，𝑎𝑡） ←𝑟𝑡+ 𝛾最大限度𝑎𝑡 + 1{ 𝑄 （𝑠𝑡 + 1，𝑎𝑡 + 1） }问𝑠吨，𝐴吨←𝑟吨+𝛾最大限度𝐴吨+1⁡问𝑠吨+1，𝐴吨+1

（1）

无论使用哪种策略，Q 值都会进行调整，以提供最佳动作值函数 Q* [ [41](https://www.mdpi.com/2079-9292/13/3/555#B41-electronics-13-00555) ]。Q 学习可以灵活地使用各种采样策略来生成状态动作组合。-贪婪动作选择方法（用公式 (2) 表示）就是其中一种广泛使用的方法，其值介于 0 和 1 之间。

𝑎𝑡=⎧⎩⎨藝術本身最大限度𝑎 ∈ 𝐴𝑄 （𝑠𝑡, 𝑎 )𝑎 ~ 𝐴 半径1 − 1 ， 他已确实复活。𝐴吨=𝐴𝑟𝐺最大限度𝐴∈𝐴⁡问𝑠吨，𝐴𝐴 ~ 𝐴瓦我吨𝐻 页𝑟𝑜𝑏𝐴𝑏我升我吨是 1−𝜀，𝑜吨𝐻埃𝑟瓦我𝑠埃。

（2）

函数逼近器可用于估计 Q 值，而不依赖于 Q 表 [ [42](https://www.mdpi.com/2079-9292/13/3/555#B42-electronics-13-00555) , [43](https://www.mdpi.com/2079-9292/13/3/555#B43-electronics-13-00555) ]。逼近器的参数集经过调整，可按照公式 (1) 估计 Q 值，神经网络表示为𝑄 （𝑠 ，𝑎 ；𝜃）问（𝑠，𝐴；𝜃）[ [43](https://www.mdpi.com/2079-9292/13/3/555#B43-electronics-13-00555) , [44](https://www.mdpi.com/2079-9292/13/3/555#B44-electronics-13-00555) ]; 其中 是近似器的参数集。Q 网络（𝑄 （𝑠 ，𝑎 ；𝜃）问（𝑠，𝐴；𝜃）) 和目标 Q 网络 (𝑄̂ （𝑠 ，𝑎 ；𝜃−）问^（𝑠，𝐴；𝜃−）) 是 DQN 中涉及的两个网络 [ [42](https://www.mdpi.com/2079-9292/13/3/555#B42-electronics-13-00555) , [45](https://www.mdpi.com/2079-9292/13/3/555#B45-electronics-13-00555) ]。Q 网络选择最优动作，目标 Q 网络创建用于更新 Q 网络参数 ( *θ* ) 的目标值。等式 (3) 中的损失函数用于最小化目标 (𝑄̂ （𝑠′，𝑎′；𝜃−）问^（𝑠′，𝐴′；𝜃−）) 和当前 Q 网络 (𝑄 （𝑠 ，𝑎 ；𝜃）问（𝑠，𝐴；𝜃）) 在每次迭代时更新 Q 网络。

𝐿𝑖（𝜃𝑖) =E [（𝑟 + 𝛾藝術本身最大限度𝑎′𝑄̂ （s′，A′；𝜃−) −𝑄(𝑠,𝑎;𝜃𝑖））2]大号我𝜃我=埃（𝑟+𝛾精氨酸最大限度𝐴′⁡问^s′，A′；𝜃−−问（𝑠，𝐴；𝜃我））2

（3）

在 DQN 的实施中，实现探索与利用之间的平衡至关重要，这需要在利用先前知识的同时探索替代行动方案。DQN 依赖于存储代理经验的数据库，其中重放记忆的使用是一个关键要素 [ [45](https://www.mdpi.com/2079-9292/13/3/555#B45-electronics-13-00555) ]。从重放记忆中随机选择的经验可用于更新 Q 网络 [ [46](https://www.mdpi.com/2079-9292/13/3/555#B46-electronics-13-00555) ]。

#### 3.1.2. 演员-评论家

[如 [ 47,48 ](https://www.mdpi.com/2079-9292/13/3/555#B47-electronics-13-00555)[]](https://www.mdpi.com/2079-9292/13/3/555#B48-electronics-13-00555)所述，演员-评论家方法旨在融合纯演员方法和纯评论家方法的优点。与纯演员方法类似，演员-评论家方法可以生成连续动作 [ 49 []](https://www.mdpi.com/2079-9292/13/3/555#B49-electronics-13-00555)。然而，它们通过加入评论家来解决与纯演员方法相关的策略梯度的巨大差异。评论家的主要功能是评估演员指定的当前策略。这种评估可以使用常用的策略评估技术进行，如时间差（λ）[ [50](https://www.mdpi.com/2079-9292/13/3/555#B50-electronics-13-00555) ] 或残差梯度 [ [51](https://www.mdpi.com/2079-9292/13/3/555#B51-electronics-13-00555) ]。评论家根据采样数据来近似和更新价值函数。反过来，派生的价值函数用于调整演员的策略参数，引导他们提高性能。与纯评论家方法不同，演员-评论家方法通常保留了策略梯度方法良好的收敛特性。

演员-评论家算法的学习代理分为两个不同的实体：演员（策略）和评论家（价值函数）。演员仅负责根据当前状态生成控制输入，而评论家则处理收到的奖励并通过调整价值函数估计来评估策略的质量 [ [49](https://www.mdpi.com/2079-9292/13/3/555#B49-electronics-13-00555) ]。在评论家进行一系列策略评估步骤之后，演员会根据从评论家那里获得的信息进行更新。

#### 3.1.3. 近端策略优化（PPO）

由于其强大的性能和简单的实现，PPO 已成为主要的在策略策略优化算法 [ [52](https://www.mdpi.com/2079-9292/13/3/555#B52-electronics-13-00555) ]。PPO 不直接最大化该下限，而是采用最大化替代目标的目标，同时对后续策略与当前策略的接近度施加约束。为了实现这一点，PPO 采用启发式方法，在每次策略更新期间重点关注以下目标：

𝐿藝術本身𝑘（𝜋) =𝔼（𝑠 ，𝑎 ）〜𝑑𝜋𝑘[分钟（𝜋（𝑎 | 𝑠 ）𝜋𝑘（𝑎 | 𝑠 ）𝐴𝜋𝑘（𝑠 ，𝑎 ），剪辑（𝜋（𝑎 | 𝑠 ）𝜋𝑘（𝑎 | 𝑠 ）, 1 − 𝜖, 1 + 𝜖））]大号钾磷磷哦𝜋=埃𝑠，𝐴~𝑑𝜋钾分钟⁡𝜋𝐴𝑠𝜋钾𝐴𝑠𝐴𝜋钾𝑠，𝐴，C升我页𝜋𝐴𝑠𝜋钾𝐴𝑠，1−𝜖，1+𝜖

（4）

其中 clip (x, l, u) = min(max(x, l), u)。从该目标中的第二项可以看出，PPO 通过消除概率比的动机来限制连续策略之间的差异𝜋（𝑎 | 𝑠 ） /𝜋𝑘（𝑎 | 𝑠 ）𝜋𝐴𝑠/𝜋钾𝐴𝑠超出预定义的剪辑范围[ 1 − 𝜖, 1 + 𝜖][1−𝜖，1+𝜖]最终，外部最小化确保方程（4）作为替代目标的下限[ [52](https://www.mdpi.com/2079-9292/13/3/555#B52-electronics-13-00555) ]。

#### 3.2. MITRE ATT&CK 场景

MITRE ATT&CK 框架是一个全面的知识库和架构，有助于了解网络对手在整个网络杀伤链中采用的策略、战术和操作程序 [ [53](https://www.mdpi.com/2079-9292/13/3/555#B53-electronics-13-00555) ]。ATT&CK 框架由 MITRE 公司开发，该公司是一家由联邦政府资助的非营利组织，负责研究和开发并提供政府研究和开发服务。ATT&CK 框架利用实际的现场观察来提供一个全球可访问的对手战术和策略知识库。来自政府、业界和学术界的网络安全专家社区会不断更新和维护该框架。ATT&CK 为网络安全专业人员提供了一种通用语言，以共享威胁情报并协作制定防御策略。通过详细了解对手行为，ATT&CK 可帮助组织更好地防御网络攻击并改善整体网络安全态势。ATT&CK 已成为全球网络安全专业人员的重要参考，其影响在安全解决方案的开发和威胁情报与安全运营的集成中显而易见 [ [54](https://www.mdpi.com/2079-9292/13/3/555#B54-electronics-13-00555) ]。

本研究通过复制所使用的策略和技术，实现并模拟了 Ajax 安全团队（一场真实的网络攻击）的场景 [ [55](https://www.mdpi.com/2079-9292/13/3/555#B55-electronics-13-00555) ]。Ajax 安全团队中的技术被实现为动作，并使用流程获取来构建模拟框架。

#### 3.2.1. 国家

MDP 由状态、动作和奖励组成。状态表示个人计算机 (PC) 在网络中的状态。在本研究中，我们列出了影响 PC 免受网络攻击的四个因素，如[表 1](https://www.mdpi.com/2079-9292/13/3/555#table_body_display_electronics-13-00555-t001)所示。状态组件包括 (1) 防火墙阻止的端口数量、(2) 无需凭证即可获得彼此管理员权限的授权 PC 列表、(3) 可能暴露连接到用户 PC 的 PC 凭证的漏洞以及 (4) 允许用户获取管理员权限的凭证。如果没有键盘安全保护，键盘记录是可能的，凭证可能会存储在 Web 浏览器中。打开恶意电子邮件可能会有一定概率导致病毒感染。在 DRL 中，代理被赋予对环境状态的全面了解，以便进行有效的决策。状态由一组值组成，这些值表征与代理目标相关的环境特征。

**表 1.** 状态组成部分。

![img](https://pub.mdpi-res.com/img/table.png)

#### 3.2.2. 行动

攻击者有四种行动选项，如[表 2](https://www.mdpi.com/2079-9292/13/3/555#table_body_display_electronics-13-00555-t002)所示。（1）尝试通过开放端口进行端口访问，（2）通过冒充授权 PC 的 IP 地址进行欺骗登录，（3）在键盘安全措施不到位的情况下进行键盘记录以获取凭据，以及（4）访问 Web 存储的凭据。首先，选择一种漏洞攻击来针对 PC 对象的漏洞，并使用端口扫描查找开放端口来攻击目标 PC 对象。攻击者使用四个属性作为状态：在之前的攻击中发现的连接节点和开放端口的数量、键盘安全措施的存在以及 Web 凭据的存在。根据当前状态，攻击者选择最合理的行动来攻击目标 PC。

**表 2.** 动作组件。

![img](https://pub.mdpi-res.com/img/table.png)

在用于网络攻击模拟的 DRL 中，必须指定多个参数（包括选择漏洞或用户凭据）才能有效执行操作。每个操作都受特定先决条件的约束，例如识别目标主机或拥有必要的凭据。每个操作的结果都可能导致发现新主机、获取敏感信息或入侵其他主机。

在探索网络安全场景时，我们借鉴了 MITRE ATT&CK 框架，特别关注 Ajax 安全团队场景。虽然 MITRE ATT&CK 框架提供了多种攻击技术，但我们发现，我们面临的计算环境非常复杂，不可能采用所有可用的技术。因此，我们做出了战略决策，将重点缩小到一组精选的代表性攻击技术。这些选定的技术包括开放端口攻击、欺骗、键盘记录和访问 Web 凭据。通过专注于这些特定方法，我们旨在深入了解系统中可能出现的漏洞和潜在的安全漏洞，确保采用更有针对性和更全面的网络安全分析方法。

#### 3.2.3. 奖励

在基于 DRL 的网络攻击模拟中，奖励系统会向代理提供有关其在环境中的行为有效性的反馈。奖励函数包含代理行为的成本及其对渗透测试过程的影响。代理当前行为的环境奖励值是根据行为成本及其方差来评估的。在我们的研究中，如果代理成功拥有节点，则会给予正奖励 (+1)，否则会给予负奖励 (-1)。这种反馈会引导代理朝着最佳行为发展，并增强其适应动态环境条件的能力。

#### 3.3. 模拟框架

在本研究中，使用广泛采用的 PyTorch 深度学习框架设计和实施了模拟网络攻击场景。为了构建模拟环境，定义并有效使用了 PC、攻击者和环境的类。该环境由一台客户端 PC 和四台目标 PC 组成，如图[1](https://www.mdpi.com/2079-9292/13/3/555#fig_body_display_electronics-13-00555-f001)所示。攻击者从客户端 PC 获取必要的攻击信息，然后发起一系列旨在渗透和控制目标 PC 的攻击。当攻击者成功入侵网络环境内的所有 PC 时，模拟结束。这种方法允许对网络攻击动态进行受控和系统的检查，从而为模拟网络中的漏洞和潜在安全漏洞提供有价值的见解。

![电子产品 13 00555 g001](https://www.mdpi.com/electronics/electronics-13-00555/article_deploy/html/images/electronics-13-00555-g001-550.jpg)

>  **图 1.** 模拟网络环境。

仿真过程主要分为两个阶段：节点（PC）发现和节点攻击。节点发现阶段如图[2](https://www.mdpi.com/2079-9292/13/3/555#fig_body_display_electronics-13-00555-f002)所示，可列举如下步骤：

![电子产品 13 00555 g002](https://www.mdpi.com/electronics/electronics-13-00555/article_deploy/html/images/electronics-13-00555-g002-550.jpg)

>  **图2.** 发现节点的过程。

- 节点发现启动：
  - 攻击者从节点发现阶段开始该过程。
  - 目的是寻找目标节点来发起攻击。
- 漏洞利用尝试：
  - 攻击者试图利用目标 PC 中的漏洞。
  - 成功利用此漏洞即可获得已发现节点的凭证的访问权限。
- 凭证获取：
  - 如果攻击成功，攻击者将获得访问已发现节点的必要凭证。
  - 这些凭证对于网络内的进一步操作至关重要。
- 失败的利用场景：
  - 如果攻击者未能利用该漏洞，他们将无法获取凭证。
  - 然而，他们仍然可以在此阶段识别并列出连接的节点。
- 进一步攻击的准备：
  - 在继续攻击已发现的节点之前，攻击者必须经过这个过程。
  - 完成此过程对于发现其他节点和获取网络访问所需的凭据至关重要。

[节点攻击阶段如图3](https://www.mdpi.com/2079-9292/13/3/555#fig_body_display_electronics-13-00555-f003)所示，可概括为以下步骤：