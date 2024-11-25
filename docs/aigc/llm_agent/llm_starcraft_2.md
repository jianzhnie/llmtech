# **SwarmBrain: Embodied agent for real-time strategy game StarCraft II via large language models**

这篇论文介绍了一个名为SwarmBrain的人工智能代理，它通过大型语言模型（LLMs）在实时战略游戏《星际争霸II》中执行实时战略任务。SwarmBrain由两个关键组件构成：Overmind Intelligence Matrix和Swarm ReflexNet。以下是对论文内容的总结：

1. **背景与动机**：
   - 大型语言模型（LLMs）在多种探索性任务中取得了显著成就，甚至超过了传统基于强化学习的代理方法。
   - 研究LLMs在《星际争霸II》这样的实时战略游戏环境中执行战略任务的有效性。
2. **SwarmBrain架构**：
   - **Overmind Intelligence Matrix**：负责从宏观角度协调战略，模仿Zerg（虫族）的集体意识，进行资源分配、指导扩张和协调攻击。
   - **Swarm ReflexNet**：由于LLMs的固有延迟，Swarm ReflexNet采用条件响应状态机框架，实现对Zerg单位的基本操作的快速战术响应。
3. **实验设置**：
   - SwarmBrain控制Zerg种族与计算机控制的Terran（人类）对抗。
   - 实验结果显示SwarmBrain能够进行经济增长、领土扩张和战术制定，并在不同难度级别上对计算机玩家取得胜利。
4. **实验结果**：
   - SwarmBrain在Very Easy、Easy、Medium和Medium Hard难度级别上对计算机对手胜率为100%。
   - 在Hard难度级别上，SwarmBrain的胜率为76%。



## **摘要**

大型语言模型（LLMs）最近在各种探索性任务中取得了显著的成就，甚至超过了传统基于强化学习的代理方法，这些方法在历史上一直主导着基于代理的领域。本文的目的是研究LLMs在《星际争霸II》游戏环境中执行实时战略战争任务的有效性。在本文中，我们介绍了SwarmBrain，这是一个利用LLM实现《星际争霸II》游戏环境中实时战略实施的具身代理。SwarmBrain包括两个关键组件：

1）Overmind Intelligence Matrix，由最先进的LLMs驱动，旨在从高层次角度协调宏观战略。这个矩阵模拟了Zerg智能大脑的总体意识，综合战略远见，目的是分配资源、指导扩张和协调多方面的攻击。

2）Swarm ReflexNet，作为Overmind Intelligence Matrix计算推理的敏捷对应物。由于LLM推理固有的延迟，Swarm ReflexNet采用了条件响应状态机框架，以实现基本Zerg单位机动的快速战术响应。

在实验设置中，SwarmBrain控制Zerg种族与计算机控制的Terran对手对抗。实验结果表明，SwarmBrain具有进行经济增强、领土扩张和战术制定的能力，并且能够战胜不同难度级别的计算机玩家。具体来说，SwarmBrain在Very Easy、Easy、Medium和Medium Hard级别的计算机对手中胜率为100%。此外，即使在Hard级别，SwarmBrain也保持了相当高的胜率，在76%的比赛中取得胜利。

**关键词**：大型语言模型、实时战略、星际争霸II、具身代理、SwarmBrain、Overmind Intelligence Matrix、Swarm ReflexNet

## **1 引言**

“我没有名字，但你可以称呼我为Swarm。我是它的一个种姓。我的专长是智能。你们试图培育我们，使用我们。但你们粗糙的实验触发了某些遗传协议，我诞生了。我只有几周大，但我有百万年的种族记忆。我只是一个工具。Swarm用来对付像你这样的威胁很多次了。通过她的记忆，我了解你们的种族。一个特别有活力的种族。我预计他们可能在几百年内来到这里，与我们竞争。但在Swarm运作的时间尺度上，你们的种族很快就会消失。最有可能的是，你们会自我毁灭。智能不是赢得生存的特质。”
—《爱，死亡与机器人》（第三季第六集“Swarm”）

由暴雪娱乐在2010年推出的《星际争霸II》是一款实时战略（RTS）游戏，在游戏社区中获得了大量关注。在标准游戏比赛中，参与者有机会在扮演三个不同种族之一——人类、虫族和星灵——的同时进行战略竞争。《星际争霸II》独特的游戏机制和复杂的战略深度使其成为人工智能（AI）发展的坚实实验平台，使其成为技术和AI研究领域的重大兴趣所在。

强化学习（RL）是训练AI代理通过与复杂环境交互来做出一系列决策以实现特定目标的最流行方法。通过接收奖励或惩罚的反馈，基于RL的代理能够从经验中学习，并优化他们的行为以最大化随时间累积的奖励。DeepMind的AlphaGo在RL领域标志着一个重要的里程碑，以其开创性的成就。随后，DeepMind的AlphaStar是RL在掌握复杂环境的《星际争霸II》方面的另一项证明，并击败了许多职业玩家。

尽管传统的基于RL的代理在《星际争霸II》中取得了显著的性能，但在这类复杂环境中实现高水平熟练度时仍面临相当大的挑战。主要的复杂性来自于试图将扩展的、复杂的目标直接映射到键盘和鼠标输入的最低级别行动。这种低级别直接映射策略常常未能捕捉到战场的全面动态。相比之下，大型语言模型（LLM）凭借其固有的高级抽象能力以及理解复杂上下文的能力，可以提供更优越的宏观战场情况理解。这种全面的视角使得AI代理能够制定更具连贯性和信息性的战术决策，潜在地提高他们在复杂场景中的性能和适应性。

然而，直接在《星际争霸II》等RTS游戏的背景下使用LLM面临着重大挑战，因为游戏本身对快速决策的需求。以前的基于LLM的代理实现在探索性任务中取得了显著的突破，特别是在Minecraft等环境中。这些成就主要归功于这些任务相对较宽松的实时约束。然而，在《星际争霸II》中，根据不同情景条件快速反应的能力至关重要。普通玩家通常保持每分钟约100次行动（APM）的速率，而更高级的玩家则达到200次以上APM。在激烈的游戏中，玩家的APM可能会飙升至300-400，相当于每秒钟执行5到6个命令。期望LLM达到这种操作速度目前是不现实的。例如，ChatGPT 4.0可能需要超过20秒来处理包含2000个标记的单个响应，这个时长对于《星际争霸II》战场的快速变化来说太长了。总之，最先进的LLM处理中固有的延迟排除了它们在像竞技《星际争霸II》这样对时间敏感的环境中的直接应用，需要新的方法来适应这些模型以跟上游戏的紧迫性。

鉴于LLMs在《星际争霸II》等RTS环境中的局限性，这导致我们考虑Swarm的社会结构，秩序的完整性在不需要其成员高水平个体认知的情况下得以保持。Swarm的每个单位都赋予了一套预定的功能，反映了Swarm个体的组织范式。当Swarm物种面临外部威胁时，情况需要高层次的Swarm智能参与，从一个宏观的角度概念化反击策略，就像人类军事指挥官制定计划一样。在这些战略发展之后，Swarm内的单位根据集体智能的指示执行分布式任务，从而抵御入侵者。

为了模仿这个过程，我们介绍了SwarmBrain，它作为Zerg种族掌握RTS游戏《星际争霸II》的SwarmBrain。SwarmBrain由两个关键组件组成：1）Overmind Intelligence Matrix，旨在从高层次角度协调宏观战略。2）Swarm ReflexNet，旨在模仿Zerg个体固有的智能，用于基本的Zerg单位机动。

特别地，Overmind Intelligence Matrix 被设计为基于对战场动态的全面理解来制定宏观战略。它由两个部分组成：Overmind Brain 和 SC2 Brain，两者都由 LLMs 提供动力。Overmind Brain 模仿 Swarm 智能大脑的内在意识，考虑到代理的状态、对手的状态和全面的战场情报，综合战略远见，目的是分配资源、指导扩张和协调对对手的进攻行动。由于 LLM 在一次性正确处理所有任务方面存在困难，因此采用 SC2 Brain 将 Overmind Brain 中的自然语言战术概念转化为《星际争霸II》中的可执行命令。由于 LLM 的推理速度慢，这妨碍了在快节奏的 RTS 游戏环境中的有效参与，以及由于缺乏视觉信息输入而无法发出高度详细的操作命令，基于 LLM 的代理方法面临重大挑战。为了解决这些问题，引入了 Swarm ReflexNet，它赋予单个 Zerg 单位简单的、自动执行的任务。这些任务包括确定攻击目标的优先级、在受到攻击时的反应协议，以及 Queen 持续产生幼虫的行为等。

## **2 相关工作**

### 2.1 大型语言模型

随着ChatGPT的出现，大型语言模型（LLM）展现出了显著的能力，展示了这些庞大模型所固有的独特能力。LLM在数学推理、泛化和遵循指令方面表现出了质的提高。因此，基于LLM的方法现在被应用于更复杂的应用场景。与专有的LLM相比，一些开源模型，如LLaMA和LLaMA 2，展示了强大的紧急能力。此外，当前阶段的小规模模型已被证明具有与大型参数语言模型相似甚至更优越的能力。具体来说，LLM在各种专业领域展示了强大的泛化能力，如代码生成和工具使用，例如Toolformer这样的工具。

随着LLM计算能力的增强，一些提示技术被证实在处理复杂任务方面是有效的，包括Chain of Thought（CoT）和ReAct方法。这些技术涉及引导LLM深入分析输入问题后再生成输出，旨在最大化结果的准确性。包括ChatGPT和GPT-4在内的LLM的出现，标志着自然语言处理方面的一个重要进展。这些模型以多轮对话能力为特点，展示了遵循复杂指令的卓越能力。GPT-4V的视觉能力的加入进一步扩展了AI应用的范围，使得从问题解决和逻辑推理到工具使用、API调用和编码等任务成为可能。关于GPT-4V的最新研究强调了其理解包括流行智能手机应用中的简单用户界面在内的各种类型图像的能力。然而，面对具有非典型UI的新应用时，出现了挑战，这突显了正在进行的工作所要解决的主要问题。在开源努力中，LLaMA系列脱颖而出，经过微调以获得对话能力，并采用了与ChatGPT类似的仅解码器架构。在LLaMA的基础上，多模态LLM如LLaVA、ChartLlama和StableLLaVA也展示了与GPT-4V相似的视觉理解能力。尽管取得了这些进展，但开源模型与GPT-4V之间仍存在性能差距，这表明了进一步发展的潜在领域。

### 2.2 大型语言模型用于代理规划

LLM的进步，特别是在多模态语言模型领域，强调了将LLM发展为复杂的自主决策系统的趋势。目前，LLM代理已证明在各种复杂的下游任务中是有效的。例如，在Rt-1、Rt-2和Voxposer等具身应用中，LLM作为决策中心，推动机器人完成复杂的、长序列任务。在Minecraft等探索性任务中，Voyoger、Ghost-in-the-minecraft和Plan4mc等项目利用代理在沙盒环境中进行自我探索，根据环境反馈学习有价值的技能。在移动助理应用和网络场景中，代理也表现出了卓越的任务泛化能力。例如，在移动设备上的零样本实验中，AppAgent展示了在十个不同应用中的卓越任务泛化。在网络导航任务中，它能够熟练地执行指令。特别是当多模态模型作为代理的基础时，它们在视觉和文本任务中的表现变得更加多才多艺和智能，如Jarvis-1。总的来说，无论是在具身应用、探索任务、移动协助、网络导航还是游戏等场景中，代理的适应性都突显了它们的多功能性和智能性。

在多代理领域，代理已被证明在竞争和合作模式中进行迭代学习。在社交推理型代理中，通过与多个代理的互动，生成代理展现出类似人类的行为和思维。在软件开发和软件项目管理等典型软件开发领域中，多代理代理展示了通过协作过程实现复杂团队任务的能力。此外，在狼人杀等游戏环境中，代理学会了类似人类的技能，如伪装和说谎。

使用LLM作为复杂任务的代理已经引起了关注，例如AutoGPT、HuggingGPT和MetaGPT等项目。这些项目展示了超出基本语言任务的能力，参与了需要更高认知功能的软件和游戏等活动。将LLM中推理和行动相结合的创新方法，增强了决策和交互能力。能够处理各种输入的多模态LLM代理进一步扩大了LLM的应用范围，实现了更有效的交互和复杂任务的完成。代理在各种场景中的适应性，包括具身应用、探索任务、移动协助、网络导航和游戏，突显了它们的多功能性和智能性。

## **3 SwarmBrain**

提出的SwarmBrain框架如图1所示，包括：(1) Overmind Intelligence Matrix，负责制定复杂的高级战略指令；(2) Swarm ReflexNet，一个子系统，旨在通过条件反射赋予Zerg的基本单位执行基本操作的能力。

SwarmBrain与《星际争霸II》环境的交互示例如图1所示。环境观察通过python-sc2 API接口获取，将游戏状态信息传递到Overmind Intelligence Matrix和Swarm ReflexNet。由于获得的游戏状态信息包含全面且复杂的游戏状态信息，我们的方法涉及选择性提取过程，只收集相关数据。随后，这些提取的数据经过一系列数学计算以提炼必要的参数。得到的精炼信息随后被封装成自然语言，作为Overmind Intelligence Matrix的输入。基于LLM的Overmind Intelligence Matrix处理处理过的自然语言数据，为Swarm ReflexNet制定战略指令。完整的提示见附录。Swarm ReflexNet反过来利用这些观察信息为Zerg单位执行条件反射式的基本决策。详细信息在第3.2节介绍。

### 3.1 Overmind Intelligence Matrix

Overmind Intelligence Matrix的整体框架如图2所示。它被构想为一个复合系统，包含四个不同但相互关联的组件。每个组件都设计有专门的功能，使Matrix能够进行高维战略操作。Overmind Intelligence Matrix的组件及其各自的功能如下：

(1) Overmind Brain：作为Overmind Intelligence Matrix的关键，这个模块旨在模仿Zerg群的总体智能，负责制定旨在保护和繁衍Zerg群的战术策略。

(2) 基于文本的记忆系统：作为Overmind Brain战术策略的存储库，这个子系统保存了之前制定的认知过程和战术策略的记录。记忆系统增强了Overmind Brain从过去的遭遇中学习的能力，随着时间的推移完善其进一步的战略，并减少不必要的指令重复。

(3) SC2 Brain：这个关键的接口将Overmind Brain的战略构想转化为与《星际争霸II》环境兼容的可执行命令集。它作为翻译器，将基于自然语言的战略转化为一系列具体的、游戏特定的动作。

(4) 指挥中心：它作为操作中心，调度SC2 Brain的命令序列，当SC2 Brain的命令不符合执行条件时（例如，制造条件未满足），它将被暂时挂起，直到满足要求后再发布命令。

#### 3.1.1 Overmind Brain

《星际争霸II》呈现了一个复杂且多面的环境，标准比赛发生在精心制作的方形地图的边界内。竞争者从地图的对角开始，玩家需要密切关注一系列游戏数据。这些数据包括玩家的经济状况，包括矿物储备和气体供应，建筑的建造顺序，单位的生产，以及敌人的情况。此外，由于“战争迷雾”的存在，不断监控战场上不断演变的情况至关重要。这包括尽早派遣侦察兵收集关于对手状态的关键情报，从而基于他们的建筑活动预测对手的战术策略。这些信息对于人类玩家在比赛中的成功至关重要，但理解这些数据，并进一步分析战场条件以制定连贯有效的战术策略，对LLM来说是一个重大挑战。

为了应对这些挑战，引入了Overmind Brain，这是一个专为Zerg群的战略控制而设计的创新概念。通过激活Zerg特有的“生存本能”，Overmind Brain被构想为维持派系的完整性，并有效应对潜在的外部威胁。其目的不在于通过LLM固有的统计推理来制定战术策略，当面对大量数据时，这是极其具有挑战性的。相反，LLM擅长在结构化场景中进行角色扮演。通过建立清晰的上下文，Overmind Brain擅长评估迫在眉睫的威胁并制定适当的战略。这个叙述为在《星际争霸II》环境中引入一种新的基于代理的方法奠定了基础，增强了成功扮演Zerg的能力。Overmind Brain的输入信息包括几个方面：

(1) 作为Overmind Brain。要作为Overmind Brain，提示设置为“你是《星际争霸II》游戏中Zerg群的智能大脑。你非常具有侵略性，知道所有Zerg单位、Zerg建筑和Zerg技术研究之间的依赖关系.....”。

(2) 地图位置信息。地图位置信息的主要功能是向LLM传达矿场位置信息，从而促进对整个战场地形的更全面理解。这种空间意识对于LLM生成与给定地图中资源地理分布相关的战略洞察至关重要。

(3) 策略库，它维护历史上的战术策略，从而确保LLM制定的新策略与过去的策略协同，并防止重复命令。

(4) Overmind Brain的全面战场评估协议，旨在增强Overmind Brain的分析能力，使其能够从全面的角度评估战场条件。这包括评估当前比赛的阶段，Zerg部队的状况——包括Zerg单位和建筑的清单以及Zerg技术研究的评估。此外，还需要分析当前Zerg作战策略。还必须考虑对手的情况，包括审查他们的单位和建筑、他们的战略意图以及对Zerg人口构成的潜在威胁。此外，收集和整合侦察情报对于形成完整的战场分析至关重要。

(5) 关键战场信息。识别和优先考虑关键战场信息的目的是确保Overmind Brain在Zerg部队与敌军交战的重要时刻给予重大关注。这种关注对于使Overmind Brain能够迅速有效地应对不断演变的战斗场景至关重要。在Zerg遭受敌军袭击的情况下，至关重要的是Overmind Brain能够及时处理这一关键战场情报，以促进快速决策和适应战术格局。因此，增强Overmind Brain的态势分析能力和响应机制对于保持战略优势和作战效能至关重要。

(6) 代理的当前情况。LLM在RTS环境如《星际争霸II》中的操作景观在很大程度上由其单位和建筑的空间分布和状态定义。LLM通过表示框架有效地指挥这些资产，该框架提供简洁、状态感知和上下文相关的信息。当前通过python-sc2接口检索的方法产生的原始数据包含不充分的单位细节，如“Unit(name='Overlord', tag=4353163265)”，这些数据冗长且缺乏关键状态描述符。在大规模冲突场景中，特别是在LLM提示中呈现如此原始数据，引入了几个缺点：1)多余的数据量显著增加了LLM的推理延迟，破坏了RTS分析。2)LLM无法辨别单个单位的状态——特别是它们是否参与战斗或处于空闲状态——损害了态势感知和决策准确性。3)对于没有视觉输入的LLM来说，对每个单位进行细粒度管理是不切实际的。为了减轻LLM的认知负担并增强游戏中的战略互动，我们引入了Swarm ReflexNet，将在第3.2节讨论。

(7) 敌人的当前情况，包括检测到的敌人单位和检测到的敌人建筑，格式与“代理的当前情况”相同。

(8) 响应规则。采用Chain of Thought方法，引导LLM根据当前战斗情况，以及盟军和敌军单位和建筑的状态，进行逐步推理，从而增强推理过程的准确性。

(9) 响应格式。生成的动作列表，以JSON格式结构化，便于指挥中心后续处理。

##### 地图位置信息

将《星际争霸II》动态战场的复杂地形纳入Overmind Brain的决策过程是一个关键方面。对其相对于对手的空间定位的全面理解对于制定可行战略至关重要。以图3为例，以“Automaton LE”地图为例，地形布局是矩形地图，设计复杂。特别是，地图中嵌入了众多矿场，通常允许在平衡力量条件下每个派系公平分配八个资源点。

在类似于战略棋盘游戏的参与者中，由对立派系代表的参与者通过派遣军事单位争夺富含矿物的领土，进行战术斗争。主要目标围绕通过领土统治消灭敌对势力，从而控制地图资产。这样的征服不仅减少了敌人的军事能力，也增强了征服者自己的经济基础。最终，通过有效地利用这些收益来扩大自己的经济和军事基础设施，玩家可以迅速取得胜利。

这种复杂的地形组成在传达微妙的地理细节给LLM方面提出了重大挑战，这是Overmind Brain的核心。Overmind Brain生成的战略命令的有效性取决于其对完整地图信息的理解，这不能仅通过文本描述来传递。虽然Overmind Brain主要是基于LLM的，但它需要增强的感官解释能力来确定对游戏环境的全面把握——这是一个艰巨的任务，对于发布微妙的战术命令至关重要。

为了弥合《星际争霸II》地图的空间复杂性与LLM理解之间的差距，我们采用了一种方法将这些复杂的空间配置转化为LLM可以解释的格式：通过将地图转化为二维矩阵结构。在这个公式中，矿场被表示为矩阵中的元素，相互连接形成一个无向网络（见图4和图5）。这个矩阵包含了战场的关系和位置数据，LLM可以处理。通过这种转化为矩阵的方法，有效地保持了游戏元素之间的空间关系和邻近性，使我们能够使LLM分析和理解《星际争霸II》地形的空间动态。

#### 3.1.2 SC2 Brain

SC2 Brain的功能是将Overmind Brain根据环境条件推断出的基于自然语言的战术概念转化为《星际争霸II》环境中的可执行命令。这个系统的优势在于其能够防止LLM同时生成复杂的高维战术指令和详细的低维指令。通过将战略先见与游戏中执行所需的具体性分开，SC2 Brain确保了执行Overmind Brain命令所需的操作清晰度和效率。这种命令翻译方法促进了对竞争性游戏战略快速变化环境的更微妙和响应性的适应。SC2 Brain的输入信息包括几个部分：

(1) 作为Zerg玩家的任务，例如“你是《星际争霸II》中的专业Zerg玩家。你知道所有Zerg单位、Zerg建筑和Zerg技术研究之间的依赖关系......”

(2) 需要翻译的战略，来自Overmind Brain的输出。

(3) 响应格式，旨在优化生成的命令在《星际争霸II》环境中执行的效率。例如，以下命令：“(Overlord, A1)->(Move)->(B1)”在这个命令结构中，第一组括号识别需要操纵的Zerg单位。第二组表示要进行的操作类型，如生成或移动。第三组指定操作的目标位置，在移动命令的情况下，对应于目的地点（例如，B1、B2等）。这种语法便于精确控制和协调游戏单位，使复杂的战略演习可以分解为一系列简化的可执行步骤。通过以这种方式标准化命令框架，执行变得更容易，允许在将战略意图转化为游戏行动时有更高的保真度。鉴于我们自己的单位从self.unit检索的每个单位都被分配了一个独特的标签标识符，我们尝试指导LLM单独操纵每个不同的单位，如下所示：“*(Unit(name='Zergling', tag=4362338305))->(Attack)->(B1)” “*(Unit(name='Zergling', tag=4361551873))->(Attack)->(B1)” “*(Unit(name='Zergling', tag=4359454722))->(Attack)->(B1)”... 然而，我们观察到LLM熟练控制不同单位的能力是次优的。此外，在需要部署大量军事力量的情况下，对单个单位进行微观管理是多余的。因此，我们目前对我们的攻击单位，如Zerglings的方法是发出集体命令——使用“(Zergling, A1)->(Attack)->(B1)”这样的表示——派遣一群Zerglings从A1位置攻击目标B1，从而避免了为每个Zergling单位分配战斗任务的需要。

### 3.1.3 指挥中心

指挥中心在《星际争霸II》游戏环境中发挥着关键作用，通过两个组成部分将SC2 Brain解释的命令转化为可操作的操作：

1) 命令解码：该功能负责解析SC2 Brain产生的结构化操作命令。使用正则表达式，它识别和提取每个命令的基本元素，包括涉及的单位、要执行的操作和行动的目标位置。值得注意的是，当涉及到Zerg研究命令时，如“Metabolic Boost”，不需要指定目标位置，指定目标位置是不必要的，研究名称作为关键标识符。

2) 条件验证：这个组件解决了LLM对《星际争霸II》建筑顺序理解中的偶尔差异。有时，这些LLM可能会发出在当前游戏条件下不可行的变形或建造命令。条件验证模块的作用是暂停这些不切实际的命令，直到满足变形或建造的所有先决条件，从而实现命令的执行。

指挥中心的这两个组成部分共同确保了从SC2 Brain到战场的战略决策的有效转化，优化了游戏和战略执行。

### 3.2 Swarm ReflexNet

为了解决LLM在RTS环境的不可行性，我们引入了Swarm ReflexNet的创新概念，这是一个框架，旨在赋予基本Zerg单位执行简单自动反射动作的能力。这种方法通过将条件反射行为嵌入到这些单位中，增强了Swarm的战略效能，从而不需要LLM生成复杂和详细的命令。接下来，讨论了Swarm ReflexNet的状态机的代表性Zerg单位，如Drone、Overlord和Zergling的示例。

Drone的ReflexNet。如图6所示，展示了Drone在面对不同情况时的状态转换。Drone以三种不同的状态为特征：Gather（默认状态）状态、Attack状态和Flee状态。这些状态在三个特定条件下——Condition A、Condition G和Condition F——之间可以相互转换，具体定义如下：

Condition A：当Drone受到敌方单位的攻击，且其视野范围内敌人的攻击力小于附近Drone的攻击力时——例如，一个敌对的SCV或不超过三个敌对的Marines——Drone转入Attack状态。需要注意的是，这种从Gather状态到Attack状态的转换不会动员所有附近的Drone对抗入侵者。这种有节制的响应是为了防止敌人骚扰战术的不利影响，确保只有最近的、总攻击力大于敌人的Drone被部署对抗入侵者。

Condition G：当Drone的视野范围内的敌人被中和或退出视线时，Drone恢复到Gather状态。

Condition F：当Drone的视野范围内出现攻击力超过附近Drone的敌军——如超过三个敌对的Marines或一组Hellions——Drone将转入Flee状态。

Overlord的ReflexNet。如图7所示，Overlord的状态转换比Drone简单，与《星际争霸II》框架内实施的机制一致。Overlord的状态机由两个主要状态组成：Idle状态（默认状态）和Flee状态，由两个特定条件Condition F和Condition I控制。这些条件如下：

Condition F：当Overlord受到攻击时，它将转入Flee状态以自我保护，根据情况向最近的友军单位或地图边缘移动以确保安全。虽然普通的Overlord通常在到达友军附近之前就被敌军消灭，但这种机制对其升级版Overseer非常有效，后者拥有更高的机动性。

Condition I：当敌对行动停止或Overlord视野范围内的敌对部队被消灭时，Overlord将恢复到Idle状态。

战斗单位的ReflexNet。对于Zerg的主要进攻单位，如Zergling、Roach、Hydralisk等，Zerg战斗单位的状态转换图具有相似性。这里以Zergling的状态机为例，如图8所示。Zergling在三个状态中运行：Idle状态（默认状态）、Attack状态和Flee状态，由三个不同条件Condition A、Condition I和Condition F控制。

Condition A：在游戏的早期阶段，当Zerglings被派往敌方基地进行积极行动时，它们优先攻击对手的战斗单位，如Marines，而不是攻击敌方建筑。在消灭这些潜在威胁后，Zerglings将前往敌方的主要矿线，优先消灭敌方SCV，从而破坏对手的经济稳定。在战斗中，Zergling群体将执行战术侧翼机动，从多个角度发起攻击，特别针对敌方单位，这些单位具有高攻击力但防御能力相对较低，如Siege Tanks。同样，在Zerg单位（包括限于地面攻击的Zerglings、Roaches和Ultralisks，以及能够攻击地面和空中单位的Mutalisks和Hydralisks）与包括Marines和Medivacs的Terran部队的交战中，Mutalisks和Hydralisks被编程为首先消灭Medivacs，以破坏Terran步兵的治愈支持。

Condition I：当Zerglings完成了它们的战斗任务或完成了LLM生成的任务——即目标位置的所有敌方单位都被消灭——Zergling状态转换回Idle。同样，相同的策略适用于其他进攻单位。

Condition F：当Zerglings在交战区域或矿区外围操作，特别是数量较少时（例如，大约一到四个Zerglings），当它们受到敌军攻击时，它们将优先撤退到盟军部队位置。这种战术原则可以类似地适用于战略框架内的其他进攻单位。

这种机制确保了Zerg单位能够自动有效地对抗威胁，无需LLM干预，从而增强了它们在多样化战斗场景中的效能。

**4 实验**

4.1 实验设置
我们使用OpenAI的gpt-3.5-turbo作为Overmind Intelligence Matrix中的大型语言模型。利用python-sc2包作为SwarmBrain与《星际争霸II》环境的交互接口。

4.2 评估结果
为了从不同角度验证所提出方法的有效性，我们在一致的实验环境中进行了30次实验。实验在专业竞技地图“Automaton LE”上进行，SwarmBrain扮演Zerg种族，而对手计算机扮演Terran种族。计算机被设置为五个不同的难度级别进行竞争，分别为：Very Easy、Easy、Medium、Medium Hard和Hard。由于每场比赛开始时双方的位置是随机的，我们只从Zerg（SwarmBrain）在左下角和敌对Terran在右上角开始游戏。

4.2.1 对不同难度级别计算机的胜利率和平均比赛时间
图9(a)展示了SwarmBrain在30场针对五个不同难度设置的计算机的比赛中获得的胜利数量。可以观察到，SwarmBrain在所有Very Easy、Easy、Medium和Medium Hard难度级别的计算机比赛中均取得了胜利。然而，在Hard难度级别的计算机比赛中，SwarmBrain共遭遇了七次失败。对这些失败的分析表明，在三起案例中，由LLM驱动的核心Overmind Intelligence Matrix发出的命令没有被Python脚本正确执行。这些脚本使用正则表达式对SC2 Brain的命令进行分类，并通过网络python-sc2库与游戏环境交互。另外两次失败是由于Overmind Intelligence Matrix发出的次优战略命令，导致经济崩溃。例如，不明智地在B2位置建造额外的Hatchery导致了Zerg持续的经济衰退，使他们无法生产足够的军事单位。这一经济失误导致在计算机的第二次攻击波中几乎失去了所有防御力量。在另外两个案例中，在游戏的后期阶段，Overmind Intelligence Matrix倾向于生产包括Mutalisks和Zerglings以及Roaches的混合空军，以协调攻击敌人的B2位置。不幸的是，这些力量不足以对抗对手的Siege Tanks、Marines、Medivacs和Thors的组合，导致人口大幅减少。

图9(b)代表了SwarmBrain在30场针对五个不同难度设置的计算机的比赛中的平均胜利时间。可以指出，对于Very Easy和Easy难度级别的计算机，SwarmBrain在大约9分钟内取得了胜利。这些迅速的胜利通常归因于SwarmBrain有效执行的骚扰战略，通过两波Zergling、Baneling或Roach单位的部署，基本上中和了计算机的进攻能力。值得注意的是，在两场比赛，SwarmBrain派出的军队在通过B2航点前往B1目标位置与Terran部队交战的途中，由于战争迷雾未能检测并交战B2的敌军单位，这导致了游戏时间的延长。在与Medium和Medium Hard难度级别的计算机的比赛中，平均胜利时间比Very Easy和Easy计算机获得的时间有所增加。比赛时间增加的原因是多方面的，涉及与整个战场动态相关的复杂因素。在与Hard难度级别的计算机的比赛中，平均胜利时间达到高峰，超过25分钟。这一显著增加反映了高水平计算机对手提出的增加的战略挑战和韧性，需要更微妙和适应性的游戏来确保胜利。

4.2.2 对不同难度级别计算机比赛期间的命令分析
图10(a)表示SwarmBrain在30场针对五个不同难度设置的计算机的比赛中对不同Zerg单位的命令指令（包括训练和攻击命令）的频率。此外，图10(b)显示了针对五个不同难度设置的计算机的比赛中使用的Zerg单位类型的百分比分布。观察分析表明，在面对Very Easy和Easy难度级别的计算机时，SwarmBrain倾向于部署基本的Zerg攻击单位，如Zerglings、Banelings和Roaches。为后期游戏设计的更高级单位，如Mutalisks、Infestors和Ultralisks，在这些比赛动态中明显缺席。这种单位选择趋势源于观察到SwarmBrain通常在第一波Zerglings、Banelings和Roaches的攻击后接近胜利，因此没有机会训练和生产更先进的Zerg单位。在与Medium和Medium Hard难度级别的计算机的比赛中，这种战术方法也适用。然而，随着比赛时间的延长和对手攻击单位组成的演变，SwarmBrain抓住机会发出训练更强大的单位的命令，如Hydralisks、Mutalisks、Corruptors、Swarm Hosts和Ultralisks。在涉及Hard难度级别计算机的场景中，SwarmBrain对Zerg单位的命令指令明显更全面。尽管有多样化的单位选择，SwarmBrain表现出不愿使用Lurkers，而是倾向于结合地面部队和空中部队单位的联合攻击策略。选择的地面部队包括Zerglings、Banelings、Roaches、Ravagers和Ultralisks，而空中部队包括Overseers、Mutalisks、Corruptors和Brood Lords。这种对综合攻击的战略偏好反映了SwarmBrain致力于进行更多样化和全面的进攻。

4.2.3 对SwarmBrain策略的讨论
SwarmBrain的侦察情报。在战争的当代和历史背景下，情报在决定冲突结果方面始终发挥着关键作用。认识到及时准确的战场信息的重要性，SwarmBrain勤勉地使用Overlords或Zerglings频繁监视对手的矿区，为自己提供最新的敌军动向和战略情报。在游戏开始时，为了获得广泛的情报并洞察对手的活动，SwarmBrain战略性地派遣Overlords检查对手的主要（B1）或次要矿区（B2）。图11展示了SwarmBrain派出的Overlord前往侦察敌方B2矿区的途中，恰当地展示了SwarmBrain对初始情报收集的承诺。

比赛的早期阶段。在检测到来自敌军部队对A2位置的切实威胁后——这种情况被归类为关键战场信息——SwarmBrain迅速做出反应。在A2的空闲Roaches和空闲Zerglings在Overmind Brain和Swarm ReflexNet的指导下被迅速动员起来，以击退逼近的威胁。图12生动地捕捉了SwarmBrain在面对攻击时的战略响应和决策。在早期进攻中，派出的Zerglings和Roaches遭受了重大损失后，SwarmBrain迅速派遣A1的Hydralisks前往A2增援。图13描绘了在A2发生的随后战斗，一支强大的Hydralisks集体面对敌对部队。

比赛的后期阶段。随着比赛进入后期，SwarmBrain表现出对空地部队混合编队的明显偏好，辅以Overseer侦察。地面部队由一些Zerglings、许多Roaches、Hydralisks和Ultralisks组成，而空中部队包括Mutalisks、Corruptors、Overseers和Brood Lords。图14展示了SwarmBrain对空地部队混合编队的偏好。

在战斗后，SwarmBrain的地面部队遭受了不到二十供应的适度消耗。作为回应，SwarmBrain调整其战略，并迅速指示附近的A4孵化场开始大量产生强大的Ultralisk，旨在攻击B1位置以补充其部队。图15展示了SwarmBrain发出命令大量产生Ultralisks。

4.3 消融研究
gpt-3.5-turbo和gpt-4.5-turbo的消融研究。鉴于Overmind Intelligence Matrix是基于LLM的框架，我们评估了使用gpt-3.5-turbo与gpt-4.0-turbo的性能差异。由于gpt-4.0-turbo的推理时间明显长于gpt-3.5-turbo，我们设计了两种实验条件：第一种情况是Overmind Brain使用gpt-3.5-turbo，同时SC2 Brain使用gpt-3.5-turbo。第二种情况是Overmind Brain使用gpt-4.0-turbo，而SC2 Brain仍然使用gpt-3.5-turbo。值得注意的是，随着OpenAI用户数量的增加，GPT模型的推理时间也在增加。这一发展对于利用LLM在《星际争霸II》比赛中取得胜利构成了重大挑战，其中LLM的快速处理能力在实现成功中发挥了关键作用。

使用gpt-4.0-turbo时，我们观察到对游戏场景的理解和理解更深入、更彻底。然而，gpt-4.0-turbo的推理时间大约是gpt-3.5-turbo的两倍。在每分钟的游戏环境中，gpt-4.0-turbo能够进行大约三次推理，而gpt-3.5-turbo可以执行六次推理，如果减少Chain of Thought的复杂性，甚至超过十二次推理。尽管gpt-3.5-turbo展现了快速推理能力，但有时会发生对游戏状态的位置误解，如图16所示。该图指示了一个错误的推理实例，其中gpt-3.5-turbo发出命令让两个Drones在没有孵化场的A8点建造两个Extractor，而不是正确的A4点。

尽管gpt-4.0-turbo提供了更全面和准确的输出，但延长的响应时间有时导致SwarmBrain未能及时发出命令以支援受到攻击的Zerg单位，导致在一些遭遇中失败。因此，考虑到推理速度和准确性之间的平衡，我们选择gpt-3.5-turbo作为Overmind Brain和SC2 Brain的LLM。

Swarm ReflexNet的消融研究。图以下是论文中“4 实验”部分的翻译：

**4 实验**

4.1 实验设置

我们使用OpenAI的gpt-3.5-turbo作为Overmind Intelligence Matrix中的大型语言模型。利用python-sc2包作为SwarmBrain与《星际争霸II》环境的交互接口。

4.2 评估结果

为了从不同角度验证所提出方法的有效性，我们在一致的实验环境中进行了30次实验。实验在专业竞技地图“Automaton LE”上进行，SwarmBrain扮演Zerg种族，而对手计算机扮演Terran种族。计算机被设置为五个不同的难度级别进行竞争，分别为：Very Easy、Easy、Medium、Medium Hard和Hard。由于每场比赛开始时双方的位置是随机的，我们只从Zerg（SwarmBrain）在左下角和敌对Terran在右上角开始游戏。

4.2.1 对不同难度级别计算机的胜利率和平均比赛时间

图9(a)展示了SwarmBrain在30场针对五个不同难度设置的计算机的比赛中获得的胜利数量。可以观察到，SwarmBrain在所有Very Easy、Easy、Medium和Medium Hard难度级别的计算机比赛中均取得了胜利。然而，在Hard难度级别的计算机比赛中，SwarmBrain共遭遇了七次失败。对这些失败的分析表明，在三起案例中，由LLM驱动的核心Overmind Intelligence Matrix发出的命令没有被Python脚本正确执行。这些脚本使用正则表达式对SC2 Brain的命令进行分类，并通过网络python-sc2库与游戏环境交互。另外两次失败是由于Overmind Intelligence Matrix发出的次优战略命令，导致经济崩溃。例如，不明智地在B2位置建造额外的Hatchery导致了Zerg持续的经济衰退，使他们无法生产足够的军事单位。这一经济失误导致在计算机的第二次攻击波中几乎失去了所有防御力量。在另外两个案例中，在游戏的后期阶段，Overmind Intelligence Matrix倾向于生产包括Mutalisks和Zerglings以及Roaches的混合空军，以协调攻击敌人的B2位置。不幸的是，这些力量不足以对抗对手的Siege Tanks、Marines、Medivacs和Thors的组合，导致人口大幅减少。

图9(b)代表了SwarmBrain在30场针对五个不同难度设置的计算机的比赛中的平均胜利时间。可以指出，对于Very Easy和Easy难度级别的计算机，SwarmBrain在大约9分钟内取得了胜利。这些迅速的胜利通常归因于SwarmBrain有效执行的骚扰战略，通过两波Zergling、Baneling或Roach单位的部署，基本上中和了计算机的进攻能力。值得注意的是，在两场比赛，SwarmBrain派出的军队在通过B2航点前往B1目标位置与Terran部队交战的途中，由于战争迷雾未能检测并交战B2的敌军单位，这导致了游戏时间的延长。在与Medium和Medium Hard难度级别的计算机的比赛中，平均胜利时间比Very Easy和Easy计算机获得的时间有所增加。比赛时间增加的原因是多方面的，涉及与整个战场动态相关的复杂因素。在与Hard难度级别的计算机的比赛中，平均胜利时间达到高峰，超过25分钟。这一显著增加反映了高水平计算机对手提出的增加的战略挑战和韧性，需要更微妙和适应性的游戏来确保胜利。

4.2.2 对不同难度级别计算机比赛期间的命令分析

图10(a)表示SwarmBrain在30场针对五个不同难度设置的计算机的比赛中对不同Zerg单位的命令指令（包括训练和攻击命令）的频率。此外，图10(b)显示了针对五个不同难度设置的计算机的比赛中使用的Zerg单位类型的百分比分布。观察分析表明，在面对Very Easy和Easy难度级别的计算机时，SwarmBrain倾向于部署基本的Zerg攻击单位，如Zerglings、Banelings和Roaches。为后期游戏设计的更高级单位，如Mutalisks、Infestors和Ultralisks，在这些比赛动态中明显缺席。这种单位选择趋势源于观察到SwarmBrain通常在第一波Zerglings、Banelings和Roaches的攻击后接近胜利，因此没有机会训练和生产更先进的Zerg单位。在与Medium和Medium Hard难度级别的计算机的比赛中，这种战术方法也适用。然而，随着比赛时间的延长和对手攻击单位组成的演变，SwarmBrain抓住机会发出训练更强大的单位的命令，如Hydralisks、Mutalisks、Corruptors、Swarm Hosts和Ultralisks。在涉及Hard难度级别计算机的场景中，SwarmBrain对Zerg单位的命令指令明显更全面。尽管有多样化的单位选择，SwarmBrain表现出不愿使用Lurkers，而是倾向于结合地面部队和空中部队单位的联合攻击策略。选择的地面部队包括Zerglings、Banelings、Roaches、Ravagers和Ultralisks，而空中部队包括Overseers、Mutalisks、Corruptors和Brood Lords。这种对综合攻击的战略偏好反映了SwarmBrain致力于进行更多样化和全面的进攻。

4.2.3 对SwarmBrain策略的讨论

SwarmBrain的侦察情报。在战争的当代和历史背景下，情报在决定冲突结果方面始终发挥着关键作用。认识到及时准确的战场信息的重要性，SwarmBrain勤勉地使用Overlords或Zerglings频繁监视对手的矿区，为自己提供最新的敌军动向和战略情报。在游戏开始时，为了获得广泛的情报并洞察对手的活动，SwarmBrain战略性地派遣Overlords检查对手的主要（B1）或次要矿区（B2）。图11展示了SwarmBrain派出的Overlord前往侦察敌方B2矿区的途中，恰当地展示了SwarmBrain对初始情报收集的承诺。

比赛的早期阶段。在检测到来自敌军部队对A2位置的切实威胁后——这种情况被归类为关键战场信息——SwarmBrain迅速做出反应。在A2的空闲Roaches和空闲Zerglings在Overmind Brain和Swarm ReflexNet的指导下被迅速动员起来，以击退逼近的威胁。图12生动地捕捉了SwarmBrain在面对攻击时的战略响应和决策。在早期进攻中，派出的Zerglings和Roaches遭受了重大损失后，SwarmBrain迅速派遣A1的Hydralisks前往A2增援。图13描绘了在A2发生的随后战斗，一支强大的Hydralisks集体面对敌对部队。

比赛的后期阶段。随着比赛进入后期，SwarmBrain表现出对空地部队混合编队的明显偏好，辅以Overseer侦察。地面部队由一些Zerglings、许多Roaches、Hydralisks和Ultralisks组成，而空中部队包括Mutalisks、Corruptors、Overseers和Brood Lords。图14展示了SwarmBrain对空地部队混合编队的偏好。

在战斗后，SwarmBrain的地面部队遭受了不到二十供应的适度消耗。作为回应，SwarmBrain调整其战略，并迅速指示附近的A4孵化场开始大量产生强大的Ultralisk，旨在攻击B1位置以补充其部队。图15展示了SwarmBrain发出命令大量产生Ultralisks。

4.3 消融研究

gpt-3.5-turbo和gpt-4.5-turbo的消融研究。鉴于Overmind Intelligence Matrix是基于LLM的框架，我们评估了使用gpt-3.5-turbo与gpt-4.0-turbo的性能差异。由于gpt-4.0-turbo的推理时间明显长于gpt-3.5-turbo，我们设计了两种实验条件：第一种情况是Overmind Brain使用gpt-3.5-turbo，同时SC2 Brain使用gpt-3.5-turbo。第二种情况是Overmind Brain使用gpt-4.0-turbo，而SC2 Brain仍然使用gpt-3.5-turbo。值得注意的是，随着OpenAI用户数量的增加，GPT模型的推理时间也在增加。这一发展对于利用LLM在《星际争霸II》比赛中取得胜利构成了重大挑战，其中LLM的快速处理能力在实现成功中发挥了关键作用。

使用gpt-4.0-turbo时，我们观察到对游戏场景的理解和理解更深入、更彻底。然而，gpt-4.0-turbo的推理时间大约是gpt-3.5-turbo的两倍。在每分钟的游戏环境中，gpt-4.0-turbo能够进行大约三次推理，而gpt-3.5-turbo可以执行六次推理，如果减少Chain of Thought的复杂性，甚至超过十二次推理。尽管gpt-3.5-turbo展现了快速推理能力，但有时会发生对游戏状态的位置误解，如图16所示。该图指示了一个错误的推理实例，其中gpt-3.5-turbo发出命令让两个Drones在没有孵化场的A8点建造两个Extractor，而不是正确的A4点。

尽管gpt-4.0-turbo提供了更全面和准确的输出，但延长的响应时间有时导致SwarmBrain未能及时发出命令以支援受到攻击的Zerg单位，导致在一些遭遇中失败。因此，考虑到推理速度和准确性之间的平衡，我们选择gpt-3.5-turbo作为Overmind Brain和SC2 Brain的LLM。

Swarm ReflexNet的消融研究。图17展示了有和没有Swarm ReflexNet的SwarmBrain的实验结果。可以看出，没有Swarm ReflexNet时，当SwarmBrain派遣Zerglings攻击最初的Terran基地位置时，Zerglings倾向于攻击Terran的供应仓库和兵营，而忽视了从事采矿活动的SCV。相反，如图17(b)所示，Swarm ReflexNet的存在将Zergling的进攻优先级重新调整为SCV，从而大大破坏了对手的经济。

图18展示了SwarmBrain的Zerg部队在武装冲突中的战术交战。Zerg军队主要由Zerglings和Roaches组成。在所示场景中，Zerglings被编程为优先执行侧翼机动以攻击Terran Siege Tanks，利用它们的敏捷性和近战优势。同时，Roaches从更远的距离参与，利用其强大的远程攻击能力对Terran军队施加压制火力。这展示了Swarm ReflexNet的有效性。
