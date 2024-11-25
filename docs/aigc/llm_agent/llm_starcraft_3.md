# LLM-PySC2 - LLM StarCraft II Learning Environment

这篇论文介绍了一个新的环境LLM-PySC2，这是为大型语言模型（LLMs）决策制定方法而开发的环境。LLM-PySC2是基于DeepMind的StarCraft II Learning Environment（SC2LE）衍生的平台，提供了完整的StarCraft II动作空间、多模态观察接口和一个结构化的游戏知识数据库，这些都能够与各种LLMs无缝连接，以促进基于LLM的决策研究。此外，为了支持多智能体研究，研究者们开发了一个LLM协作框架，支持多智能体并发查询和多智能体通信。

主要贡献和发现包括：

1. **环境介绍**：LLM-PySC2是第一个提供完整StarCraft II动作空间、多模态观察接口和结构化游戏知识数据库的环境，这些特性使其能够与各种LLMs结合，以便于研究基于LLM的决策制定。

2. **多智能体研究支持**：开发了一个支持多智能体并发查询和通信的LLM协作框架，以支持多智能体研究。

3. **实验与评估**：在实验中，LLM-PySC2环境被适配以兼容StarCraft Multi-Agent Challenge（SMAC）任务组，并提供了八个新的场景，重点关注宏观决策能力。研究者评估了九个主流的LLMs，并发现足够的参数对于LLMs做出决策是必要的，但提高推理能力并不直接导致更好的决策结果。

4. **重要发现**：大型模型需要在部署环境中通过参数训练或无需训练的学习技术自主学习，以提高其在多智能体决策领域的能力。

5. **实验结果**：在LLM-SMAC任务和LLM-PySC2任务中测试了不同的LLMs，结果显示预训练的LLMs具备决策能力，但缺乏做出一贯有效决策的能力。这些模型可能无法分析出实现胜利的关键要素，并且经常无法识别游戏中知识的重要部分，有时甚至分析错误或对盟友造成伤害。

6. **讨论**：实验中发现LLM在决策制定中存在多个问题，包括幻觉、游戏知识利用不足、对世界理解不足以及合作质量低下。

7. **结论**：LLM-PySC2环境是第一个适应连续PySC2动作的LLM StarCraft II环境，也是第一个具有多智能体框架和通信系统的环境。实验结果表明，LLMs能够以正确的形式生成决策和动作，但决策质量相对较低，存在多个问题。研究者希望LLM-PySC2环境能够促进LLM学习方法的研究，帮助基于LLM的决策方法更好地适应任务场景。

## 0.摘要

本文介绍了一个新的环境LLM-PySC2（大型语言模型星际争霸II学习环境），这是一个源自DeepMind的星际争霸II学习环境的平台，旨在开发基于大型语言模型（LLMs）的决策方法。这是第一个提供完整的星际争霸II动作空间、多模态观察接口和结构化游戏知识数据库的环境，这些都能够与各种LLMs无缝连接，以促进基于LLM的决策研究。为了进一步支持多智能体研究，我们开发了一个LLM协作框架，支持多智能体并发查询和多智能体通信。在我们的实验中，LLM-PySC2环境被适配以兼容星际争霸多智能体挑战（SMAC）任务组，并提供了八个新的场景，重点关注宏观决策能力。我们评估了九个主流的LLMs在实验中的表现，结果表明足够的参数对于LLMs做出决策是必要的，但提高推理能力并不直接导致更好的决策结果。我们的发现进一步表明，使大型模型能够在部署环境中通过参数训练或无需训练的学习技术自主学习的重要性。最终，我们期望LLM-PySC2环境能够促进LLM学习方法的研究，帮助基于LLM的方法更好地适应任务场景。

## 1.Introduction

在2017年，DeepMind和暴雪娱乐开发了StarCraft II Learning Environment（SC2LE），这是第一个允许各种强化学习（RL）代理在StarCraft II游戏中相互竞争的环境，并推动了如QMix、加权QMIX、MAPPO以及家喻户晓的AlphaStar等决策方法的出现。然而，RL训练的代理通常需要大量的数据和长时间的交互，但由于任务相关的奖励函数，它们在大多数场景中仍然缺乏泛化能力。因此，当前迫切需要开发新的决策方法。

此外，像斯坦福小镇（Stanford Town）、LLM玩Minecraft和Diplomacy游戏等令人印象深刻的研究工作展示了LLM在基于LLM的决策中的潜力。考虑到大型模型展现出更大的互动性、可解释性和推理能力，将大型模型应用于复杂决策环境中是非常自然的。然而，目前还没有足够全面的平台来支持LLM决策方法在复杂环境中的研究。值得注意的是，主流平台SC2LE环境尚未支持使用大型模型进行决策研究。

为了利用大型模型的优势并规避RL的缺点，研究者将SC2模块开发成TextStarCraft II（TSC2），首次使LLM能够与StarCraft II环境互动。然而，该环境存在一些限制。基于LLM的代理不能使用微观操作和单位技能来击败敌方单位，因为离散动作空间被规模裁剪。同时，观察只包含单位数量和升级状态，这对于实施复杂策略来说是不够的。更重要的是，由于TSC2是单智能体框架，因此不支持多智能体协作。

为了解决这些问题，我们开发了LLM-PySC2，这是一个基于PySC2模块的SC2LE衍生环境。该环境为代理提供了全面的观察，包括全局信息和代理特定的局部战斗信息（以文本形式或多模态形式）以及一个结构化的游戏知识数据库。我们还扩展了动作空间到完整的StarCraft II动作空间，使代理能够执行精细的操作和单位技能。为了支持多智能体研究，我们构建了一个具有通信系统的多智能体框架，允许点对点和领域通信。

在实验中，我们提出了八个新场景。与SMAC任务不同，这些任务不仅强调微观操作，还强调任务理解和宏观决策能力。我们在SMAC任务和新提出的场景中测试了九个主流的LLMs。结果表明，预训练的LLMs具备决策能力，但缺乏做出一贯有效决策的能力。未经特定任务训练的预训练LLMs可能无法分析出实现胜利的关键要素。它们经常无法识别游戏中知识的重要部分，大多数时候分析错误，有时甚至对盟友造成伤害。

总之，在多智能体决策领域提高LLMs的能力还有很多工作要做。我们希望LLM-PySC2环境能够推进LLM学习技术的研究，帮助基于LLM的方法更好地适应任务场景。

## 2.Related Works

### 2.1 Starcraft II

StarCraft II是评估算法的经典平台。特别是作为一款实时战略游戏，StarCraft II具有高维部分可观察状态空间和巨大的连续动作空间。拥有三个种族和超过120种单位，它被广泛认为是最复杂和最具挑战性的环境之一，通常用于评估高级决策方法。

为了支持学习方法的研究，DeepMind和暴雪娱乐开发了SC2LE，这是一个全面的RL研究环境。这个环境旨在改进复杂策略游戏中的学习算法研究。它提供了RL接口，如观察、动作和奖励函数，被认为是人工智能领域最重要的环境之一。

因此，在SC2LE引入之后，越来越多的StarCraft II环境出现了。在这些环境中，SMAC和PyMARL最为著名。SMAC是一个包含23个任务的基准测试，专门设计用于多智能体RL，主要关注分布式多智能体决策制定。为了评估MARL算法，SMAC团队还开发了PyMARL作为他们的训练平台。在PyMARL框架中，集成了五种以上的算法，并且该框架逐渐扩展成为一个多环境可用的RL平台。

总的来说，他们的工作有效地推进了多智能体学习方法的研究，对智能决策领域做出了重要贡献，并激励我们开发一个基于LLM方法的环境。

### 2.2 LLM决策制定和Text StarCraft II

近年来，LLM的决策能力开始受到关注。2023年，一个基于LLM的代理称为Minecraft中的幽灵（Ghost in Minecraft）在Minecraft的钻石挑战中取得了67.5%的成功率。之后，开发了Agent-Pro，这是一个能够使用扑克中类似虚张声势策略的LLM代理。此外，研究人员在狼人杀（Werewolf）中部署了LLM代理，这是一个通过通信进行欺骗和反欺骗的游戏，并在Diplomacy游戏中开发了LLM代理，这是一个合作与竞争的游戏。

这些工作激发了研究人员在游戏环境中开发基于LLM的决策方法。作为最著名的实时战略游戏之一，StarCraft II首先被开发成一个名为TSC2的LLM可交互环境。这个环境使LLM能够在StarCraft II中做出宏观决策，并证明LLM能够做出决策并在StarCraft-II中击败内置的四级机器人。然而，TSC2不支持单位的微观操作和多智能体协作，并且在观察和动作空间方面面临限制。

在这些情况下，我们构建了LLM-PySC2环境，旨在解决这些问题并提供一个新的StarCraft II环境。我们还使我们的环境与SMAC任务兼容，便于与在StarCraft环境中开发的算法进行比较。

## 3.LLM-PySC2 环境

### 3.1 框架

LLM-PySC2环境是建立在PySC2模块的代理层面上的。在图1中，MainAgent扮演控制摄像头、选择单位、收集观察结果和执行动作的角色，而LLM代理扮演实际决策者的角色，观察游戏情况、分析并给出动作。每个LLM代理连接到一个LLM，从一个包装器那里获取文本或多模态观察结果，在独立线程中查询LLM，最终得到游戏分析和动作。

> 图1：LLM-PySC2框架。在LLM-PySC2中，原始的PySC2观察结果将转换为文本形式的观察结果。LLM生成的文本动作可以被识别并转换为PySC2动作函数，使LLM能够与StarCraft II环境互动。

### 3.1.1 与环境互动

一个互动步骤包括两个阶段：辅助管理和决策制定（它包括许多游戏步骤）。在辅助管理阶段，不涉及LLM。MainAgent将控制PySC2摄像头并完成像编组新训练的单位和管理闲置工人等工作，以避免让大型模型过度参与简单和重复的劳动。

在决策制定阶段，将收集每个代理单位团队的观察结果。收集完所有团队的观察结果后，代理使用Observation Wrapper将结构化观察结果转换为文本观察结果。然后，所有代理并发查询远程或本地LLM，等待直到所有代理得到响应。

所有代理得到响应后，它们将使用Action Recognizer检测有效动作并将文本动作转换为结构化形式。然后，MainAgent将摄像头移动到收集观察结果时的同一位置，并执行每个代理存储的动作。执行完所有动作后，LLM-PySC2环境将进入下一个互动步骤，并重复上述工作。

### 3.1.2 多智能体通信

考虑到LLM在互动方面具有固有优势，我们为多智能体框架设计了一个通信系统。在通信系统中，代理使用“通信动作”相互通信，这是一种类似于图1中显示的单位控制动作的文本动作。

在通信系统中，一个LLM代理可以向另一个代理发送消息，或者向频道发送信息。如果消息是发送给代理的，只有指定的接收者才能获得信息。如果消息是发送到频道的，所有监听该频道的代理共享信息。通过这些通信动作，可以轻松构建如集中式决策和分布式决策等多智能体协作框架。

### 3.2 观察

观察对于决策制定是不可或缺的。不同任务的代理需要不同类型的信息。大致上，我们将观察信息分为两类：用于微观操作的局部观察和用于宏观决策的全局观察。这些观察可以根据不同形式分为文本和图像观察。

> 图2：微观操作LLM的文本观察。文本观察是查询消息的一部分。它包含许多段落，包括团队单位信息、相关游戏知识和有效动作。当观察包装器处理原始obs对象时，会添加语义信息。

#### 3.2.1 文本观察

微观操作的观察包装器这个包装器专注于局部观察。它为代理提供了控制单位、附近友军和附近敌军单位的详细信息。它从PySC2 obs对象中提取单位信息，从知识库中提取相关游戏知识。如图2所示，包装器生成的文本观察包括游戏时间、单位信息、单位知识、有效动作、短期记忆、通信数据和任务描述。使用该包装器的代理被设计用于进行微观操作，如与敌方单位战斗或在特定位置建造建筑物。

宏观决策的观察包装器这个包装器专注于全局观察。它提供了部署信息、单位数量和升级状态，类似于TSC2环境中的文本观察。对于负责军事部署的代理，从包装器生成的文本观察用于支持整体战略。对于负责发展的代理，生成的全局观察将使代理了解当前的经济和技术状况，支持未来发展规划。

#### 3.2.2 图像观察

在StarCraft II的复杂环境中，仅依赖文本观察可能会阻碍代理完全理解战场动态。为了增强态势感知，LLM-PySC2环境提供了多模态观察。这一特性使多模态大型模型能够整合视觉信息，从而更准确地理解情况。图3突出显示了两种主要类型的图像观察：游戏图像观察和特征图观察。这些视觉输入为代理提供了关键的战场信息，有助于战术分析和战略发展。

> 图3：图像观察。LLM-PySC2直接从PySC2界面提取图像观察，包括游戏图像和特征图。这些图像通过辅助线增强，为LLM提供了坐标信息。这种方法使代理能够准确感知关键战场元素，如单位数量和分布，同时也传达了通过文本难以表达的信息，如地形特征和相对空间关系。

图3展示了从PySC2界面直接提取的游戏图像和特征图。这些图像通过辅助线增强，提供了坐标信息。这种方法使代理能够准确感知关键战场元素，如单位数量和分布，同时也传达了通过文本难以表达的信息，如地形特征和相对空间关系。

以下是论文中“3.3 Action”部分的翻译：

### 3.3 动作

在决策环境中，“动作”的概念对于使代理与环境之间的互动成为可能至关重要。在我们的框架中，LLM通过基于文本的动作与环境互动，这些动作必须遵循特定格式才能被识别并转换为PySC2动作函数。将文本动作处理成PySC2函数的过程可以在图4中看到。

> 图4：文本动作识别。默认的动作识别器通过搜索LLM响应中的“Actions”部分来识别文本动作，提取动作名称和参数，搜索对应的PySC2函数，并生成函数的回调形式。

文本动作 这些动作以直观和描述性的语法表达，允许LLM在没有额外上下文的情况下理解预期的操作。一个标准文本动作被封装在尖括号中，并包含几个参数，形状为<ActionName()>, <ActionName(arg0)>, 或 <ActionName(arg0, arg1)>。参数可以代表各种元素，如单位标签、屏幕坐标或小地图位置，允许这些动作包含PySC2的完整连续动作空间。

在决策阶段，LLM将被告知当前可用的动作，例如<Attack_Unit(tag)>, <Move_Screen(screen)> 和 <Select_Unit_Attack_Unit(tag, tag)>。LLM可以根据观察到的信息和其目的生成动作，如<Attack_Unit(0x100030001)> 或 <Move_Screen([23, 37])>。如果LLM生成了多个文本动作，第一个动作将立即执行，其余动作将被添加到等待执行的动作序列中。

动作空间 在我们的环境里，PySC2的所有动作都是可用的，但是每个代理不必面对其种族的所有动作。在我们的环境里，动作空间是代理特定的，允许每个代理定义一个独特的动作集合。对于控制如Stalkers这样的单位的代理，动作空间由文本动作组成，如<Stop()>, <No_Op()>, <Move_Screen(screen)>, <Move_Minimap(minimap)>, <Attack_Unit(tag)>，并不含培训单位或研究等动作。

## 4.实验

### 4.1 实验场景

为了促进基于LLM的决策制定研究，我们提供了两组实验：LLM-SMAC任务和LLM-PySC2任务。LLM-SMAC任务与标准SMAC实验相同，作为与基于RL的方法进行比较的桥梁。LLM-PySC2任务是新提出的场景，与专为微观操作设计的SMAC任务相比，它们更强调大型模型理解任务场景和做出宏观决策的能力。

#### 4.1.1 LLM-SMAC任务

LLM-SMAC任务与原始SMAC任务的设置相同。这些任务为双方初始化单位，并自动对敌方单位发起攻击。在这些场景中，胜利的关键在于集中火力、控制战斗距离，有时还涉及互动频率。它们是与基于RL的方法进行训练数据效率比较的好场景，但不是利用LLM的多任务和宏观决策能力的好场景。

#### 4.1.2 LLM-PySC2任务

LLM-PySC2任务是新提出的实验场景，是一个测试代理情境分析能力、规划能力、知识应用、通信和协作的任务组。图5显示了一些任务。


（a）任务1：2 Adept骚扰 （b）任务2：3 Pheonix骚扰 （c）任务3：拦截敌方空投
（d）任务4：中等规模战斗 （e）任务5：大规模战斗（类型1）（f）任务6：大规模战斗（类型2）

> 图5：LLM-PySC任务组。LLM-PySC任务组包含八个任务，每个任务有三个难度级别。与SMAC任务相比，它们更强调宏观决策、情境分析和技能使用。这些场景在职业比赛中很常见。赢得这些小型任务有利于未来赢得完整的游戏。

在这些任务中，LLM需要规划一条渗透到敌方基地的路线并杀死敌方工人，或者使用单位技能实施特定战术。此外，这些任务更适合研究多智能体协作方法，并为LLM实施集中式或分布式决策。

LLM-PySC2任务组中有一半的任务场景是单智能体决策场景（从任务1到4），其中一个LLM代理控制多个单位；另一半（从任务5到8）测试代理之间的合作，多个代理控制承担不同战术角色的多个单位。在LLM-PySC2任务组中，图像观察和多智能体通信是可用的，并且如果需要，可以轻松禁用。

为了避免SMAC中的方法总是能达到大多数任务的100%胜率，我们为实验组设置了三个不同的难度级别。从1级到3级，敌人的力量逐渐增加。在更高级别，敌方会增加更多单位或升级，确保这些任务即使在基于LLM的决策方法发展良好后仍然有效。

### 4.2 实验结果

为了便于后续研究，我们测试了各种大型模型的决策能力。所有实验都在StarCraft II版本5.0.13（92440）和LLM-PySC2 v0.1中进行。我们记录了击杀单位资源比与死亡单位（K/D比率）和胜率（WR，即任务完成率）。K/D比率和WR的组合反映了LLM在决策场景中的表现。

在LLM-PySC2环境中，我们提供了一系列LLMs，如GPT-3、GPT-4、GPT-o1、GLM-4、Claude-3、Llama-3.1。我们测试了其中的代表性模型，测试了不同推理能力的模型在决策任务中的表现（GPT-3.5、GPT-4o-mini、GPT-4o），并测试了基于相同架构的不同参数的模型的表现（Llama3.1-8b、Llama3.1-70b、Llama3.1-405b）。

所有实验都使用开源代码的默认配置。作为基准，我们没有特别设计提示来提高决策质量或指导LLMs获得胜利，所有LLMs都没有在LLM-PySC2环境中进行微调。结果表明，大型模型可以做出决策并生成正确形式的文本动作。然而，当任务足够复杂或需要大量微观操作时，大型模型可能表现不佳，这表明需要通过训练或其他技术方法来提高它们的决策质量。

#### 4.2.1 LLM-SMAC任务中的实验结果

在LLM-SMAC任务中，我们对每个场景中的6个LLM进行了20次重复实验。对于由GPT-3.5-turbo做出决策的场景，我们将次数提高到50次，因为它具有良好的并发支持和友好的成本。在这些实验中，所有大型模型都使用文本观察。这种设置对于除了2c_vs_64zg之外的场景是完全足够的，因为它们基本上不需要利用地形信息。结果如表1所示。

我们发现，尽管大型模型可以分析观察信息并以正确的形式输出动作，但在SMAC任务中表现不佳。一方面，由于LLM的幻觉和缺乏特定任务知识，它们无法推断出集中火力是获胜的关键。另一方面，即使观察提供了Zealots比Stalkers具有更高的攻击效率的知识，大型模型有时仍然选择首先攻击敌方的Stalkers，如在2s3z和3s5z任务中。

#### 4.2.2 LLM-PySC2任务中的实验结果

与SMAC中的实验一样，我们对每个场景中的每个大型模型进行了20次重复实验，对GPT-3.5-turbo进行了50次。考虑到所有模型都无法完成第8个任务的多行攻击，我们仅列出了第1至第7个任务的数据。结果如表2和表3所示。

表2：Gpt-3.5-turbo在LLM-PySC2任务中的击杀/死亡比率和胜率（1级/2级/3级）。

表2中，我们测试了Gpt-3.5-turbo在每个任务的所有级别中的表现。这些数据可以作为未来研究的基准值。这三种难度级别不仅作为未来开发决策方法的验证场景，也可以应用于分布外（OOD）任务，例如在2级上训练并在3级上验证。

表3：LLMs在LLM-PySC2任务中的击杀/死亡比率和胜率（1级）。

根据表3中的数据，可以推断出两个结论。首先，大型模型需要足够的参数来进行决策。Llama-3.1-8b，参数最少的模型，在我们测试的所有模型中表现几乎最差，而70b和405b模型的表现优于8b模型。其次，提高推理能力并不会导致决策能力的线性提高。尽管GPT-4o在大多数实验中表现最佳，但它在一些容易完成的任务中仍然有零胜率，例如任务4。这些结果导致一个结论：预训练的大型模型不能直接承担复杂的决策任务，而在部署场景中的学习几乎是不可避免的。

## 5.讨论

幻觉。幻觉是导致错误决策的第一个问题。有时，LLM会混淆屏幕坐标和小地图坐标（如图6所示），或者使用观察结果中“有效动作”部分未提及的动作。有时，LLM甚至会损害队友单位。幻觉已成为基于LLM决策制定的紧迫问题。

图6：LLM在决策中的幻觉。这是一个LLM混淆屏幕坐标和迷你地图坐标的例子，浪费了一个PsiStorm技能和75点能量。这只是大型模型幻觉的一个实例。实际上，还有许多其他表现形式，比如攻击队友和错误选择优先目标。

知识利用不足。大型模型通常在利用游戏相关知识方面存在显著缺陷。在任务2中，游戏知识显示Phoenix的GravitonBeam能力将阻止单位移动和攻击。然而，这个能力仍然被过度使用，未能在任务2中获得胜利。在任务5中，即使LLM知道Disruptor的PurificationNova会造成大量伤害，他们仍然在受伤的单位上使用这个技能，导致大量溢出伤害。

对世界理解不足。缺乏对世界的理解是一种知识缺乏。预训练的LLM通常没有在决策任务中进行训练。它们不知道如何在每个任务中获胜。例如，在任务4中，大型模型应该使用Stalker的Blink能力将受伤的单位转移到后方。然而，这个能力很少被使用，导致单位死亡和任务4的零胜率，尽管LLM被告知Blink通常用于追击敌人或撤退受伤单位。

合作质量低下。在任务5到8这样的多智能体任务中，LLM代理应该与其他人合作，共同击败敌人。然而，我们发现这些代理很难合理分配目标、协调攻击时机和撤退时机，无论它们是否与领导/指挥官合作。如何提高LLM代理的合作性能是在构建高级多智能体决策系统时重要的问题。

这些问题阻碍了LLM在决策场景中的应用。幸运的是，有许多方法可以提高大型模型的决策能力。例如，直接向LLM提供知识可能会直接提高它们的能力。然而，向LLM提供知识或精确标注的数据集通常需要大量的资源。自监督学习仍然是提高决策能力的最有吸引力的方式，无论是通过基于奖励还是无需奖励的方法（例如LLM反思），无论是通过参数训练还是无需训练的技术。