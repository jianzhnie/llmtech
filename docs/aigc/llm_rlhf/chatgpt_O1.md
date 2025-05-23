

# 摘要

OpenAI 最近发布的 ChatGPT o1 代表了强人工智能领域的重大突破。该模型通过强化学习技术进行训练，并在推理过程中显式嵌入“思维链”（NCoT）机制，使其能够在生成响应之前通过逐步推理进行“深度思考”。o1 在数学和编程方面的能力是之前 ChatGPT 4o 的五倍，特别是在竞争性编程、数学奥林匹克预选赛以及物理、生物和化学基准测试中表现出色。o1 的关键创新在于它允许在推理过程中花费更多时间进行推理，标志着从快速、直接的响应转向缓慢、深思熟虑的多步推理时间计算。

本文全面回顾了与 LLM 推理能力相关的文献，并探讨了实现这一突破的核心技术和方法。我们讨论了自回归 LLM 的挑战，提出了将推理过程建模为马尔可夫决策过程（MDP）的方法，并探讨了如何通过过程奖励模型（PRM）和强化学习来优化 LLM 的推理能力。此外，我们还介绍了推理时间计算的优化方法，并展望了未来的研究方向。

通过结合大型语言模型的预测能力与强化学习和世界建模的战略深度，AI 系统可以潜在地参与更复杂的问题解决和决策过程。这种混合方法允许快速模式识别（类似于系统 1 思维）和深思熟虑的逐步推理（系统 2 思维的特征），可能解释了 o1 中观察到的显著性能飞跃。

未来的研究将继续推动 LLM 在推理能力方面的进步，使其能够更好地应对复杂的现实世界任务，为人工智能的发展开辟新的可能性。

# 1 背景

OpenAI 最近发布了 ChatGPT o1 [17]，这是一个突破性的大型语言模型（LLM），代表了强人工智能领域的巨大飞跃。该模型通过强化学习技术进行训练，并在推理过程中显式嵌入“思维链”（NCoT）机制，使其能够在生成响应之前通过逐步推理进行“深度思考”。据报道，o1 在数学和编程方面的能力是之前 ChatGPT 4o 的五倍，特别是在竞争性编程中排名第 89 百分位，在美国著名数学奥林匹克预选赛中位列前 500 名学生，并在物理、生物和化学基准测试中超越了人类博士水平的准确性。o1 的一个关键创新是它允许在推理过程中花费更多时间进行推理，标志着从快速、直接的响应转向缓慢、深思熟虑的多步推理时间计算（图 1）。

有趣的是，在人类认知中，有两种相关但不同的认知处理模式来指导人类的决策和行为 [8]，每种模式都有部分区别。

图 1：推理时间计算。(a) 自回归 LLM 直接根据给定问题 (Q) 生成答案 (A)。(b) 思维链的概念，即逐步思考，涉及在得出最终答案 (A) 之前加入中间推理步骤 (R)。这些重复操作允许：1) 重新审视和修订先前的输出，2) 推进到后续推理阶段，3) 探索多个推理路径或轨迹。

系统 1 思维是快速、自动和直觉的，操作毫不费力且通常是无意识的。它依赖于能够实现快速处理的神经通路，特别是在需要快速反应或认知资源受限的情况下。系统 2 思维是深思熟虑、费力和有意识的，涉及集中注意力和分析推理。它处理信息较慢，用于复杂的问题解决、逻辑推理和决策任务。o1 是人工智能领域的一个令人兴奋的发展，因为 LLM 现在不仅可以利用学习到的模式生成快速响应，更重要的是，通过思维链或其他形式的搜索机制模拟复杂的推理过程，类似于人类进行深入的逐步思考。

需要注意的是，在 AI 中引入思维链过程并不意味着类似人类的意识。相反，这些机制通过将任务分解为可管理的步骤来增强推理和问题解决能力，而不暗示任何形式的自我意识或主观体验。

ChatGPT o1 改进的推理能力对多个领域有许多影响，包括科学、编程和数学。在编程竞赛中，o1 的专门版本取得了令人印象深刻的成绩，在 2024 年国际信息学奥林匹克竞赛中排名第 49 百分位，并在模拟的 Codeforces 竞赛中超越了 93% 的人类竞争对手。除了其技术能力外，o1 还代表了 AI 安全和对齐方面的进展。该模型的思维链推理为整合人类价值观和原则提供了新的机会，从而在安全评估和越狱测试中提高了性能。

思维链推理和逐步思考的概念在大型语言模型（LLM）中并不新鲜。先前的研究表明，只需在输入问题中添加“描述你的推理步骤”或“逐步解释你的答案”等指令，或提供少量示例，就可以触发 LLM 生成中间推理步骤（如图 1 所示），从而改善问题解决能力，特别是在数学和编程任务中 [32; 16]。然而，这些方法建立在现有 LLM 的基础上，而没有真正将思维链能力嵌入到模型本身中。因此，LLM 无法固有地学习这种推理能力，这导致了对如何将其直接整合到模型训练中的积极研究。提出的方法范围从收集专门的训练数据到构建奖励模型 [18; 11; 15] 以及增加解码的计算复杂性 [24; 33]，但迄今为止，尚未在大规模上取得显著的性能突破。

目前尚不清楚 OpenAI 的 o1 创新是否植根于模型本身，而不是依赖于外部提示系统。如果它确实涉及在架构中显式嵌入逐步推理，那么这将代表一个重大突破。基于显著的性能提升，OpenAI o1 表明，传统上在训练期间应用的扩展原则 [9; 24] 现在与推理阶段相关。我们应该重新分配我们的计算重点，平衡预训练工作与推理时间计算的有效利用。

图 2：人类认知与 LLM 的类比。(a) 和 (b) 人类的有意识或无意识行为依赖于部分不同的大脑回路。(a) 人类的无意识控制由少数专门的大脑区域维持，如前脑岛和前补充运动区（pre-SMA）。(b) 而有意识控制则涉及更广泛的网络，激活顶叶和额叶内的许多区域 [28]。无意识控制通常是快速和本能的，通常由自动过程驱动，而有意识控制则倾向于涉及更深思熟虑、计算和深入的思考，允许仔细反思和彻底分析。

允许 LLM 通过增加测试时间计算来增强其输出，是创建能够管理开放式强推理和决策任务的通用自我改进代理的重要一步。这一方向，我们称之为 LLM 原生思维链（NativeCoT），应该能够固有地反映人类系统 2 思维的深思熟虑、分析过程 [8]。

鉴于 o1 是一个闭源系统，实现如此强大推理能力的确切技术仍然是一个谜。在本文中，我们将提供相关文献的全面概述，并深入探讨我们认为支撑这一突破的核心技术和方法。此外，我们将提出实现开源对应模型的思路，这可能会加速该领域的研究。我们的提案将借鉴最近的工作，包括我们在数据获取、基于强化学习的训练以及搜索和基于 MCTS 的解码方面的工作，以提高现有模型的推理能力。

在下一节中，我们将讨论典型自回归 LLM 常见的两个挑战，强调对世界模型和思维链机制的需求。然后，我们将提出一个 MDP 公式，用于在 LLM 中嵌入原生 CoT（从而产生类似 o1 的推理模型），并探讨其实现细节。最后，我们将以文献评论结束，并提出未来的研究方向。

# 2 自回归 LLM 的挑战

自回归语言模型（LLM）通过预测序列中的下一个标记（例如单词）来生成文本序列 [29]。从数学上讲，它们基于条件概率的原理。任务是通过使用概率链规则将标记序列 $\mathbf{x}=(x_{1},x_{2},\ldots,x_{T})$ 的联合概率分解为条件概率的乘积，其中 $T$ 是序列的长度。

给定一个标记序列 $\mathbf{x}=(x_{1},x_{2},\ldots,x_{T})$，自回归语言模型将联合概率 $P(\mathbf{x})$ 估计为：

$$P(\mathbf{x})=P(x_{1},x_{2},\ldots,x_{T})=\prod_{t=1}^{T}P(x_{t}\mid x_{1},x_{2 },\ldots,x_{t-1}),$$

其中模型根据序列中所有先前的标记 $x_{1},x_{2},\ldots,x_{t-1}$ 预测每个标记 $x_{t}$ 的概率。通常，这是通过使用像 Transformer [29] 这样的神经网络来实现的，这些网络经过训练以最小化训练数据的负对数似然。有关训练步骤的解释，请参阅附录 A。

在推理时，模型通常通过从概率分布 $P(x_{t}\mid x_{1},x_{2},\ldots,x_{t-1})$ 中顺序采样标记来生成文本，直到达到停止标记或达到预定义的最大长度。模型的工作方式如下：首先，从给定序列或起始标记开始（如果从头生成）。其次，在每一步 $t$，根据先前生成的标记 $(x_{1},x_{2},\ldots,x_{t-1})$ 预测下一个标记 $x_{t}$。最后，继续采样直到序列完成。对于一个简单的三标记序列 $\mathbf{x}=(x_{1},x_{2},x_{3})$，序列的概率为：

$$
P(\mathbf{x})=P(x_{1})\cdot P(x_{2}\mid x_{1})\cdot P(x_{3}\mid x_{1},x_{2}).
$$
这个公式支撑了 GPT 风格的自回归 LLM 的操作。学习是通过最小化预测后续标记（单词）的错误来实现的。第一个挑战是这个“预测下一个标记”的目标。虽然有些人认为预测下一个标记可能为通用人工智能（AGI）铺平道路，但我们认为仅仅专注于预测下一个单词会限制智能的潜力。可能需要不同的优化目标和学习范式来培养更深层次的智能。

为了说明纯预测模型的局限性，让我们考虑国际象棋大师的领域。在这种情况下，每个国际象棋走法可以概念化为一个标记，完整的国际象棋对局代表“国际象棋语言”中的“句子”——从开局到残局的一系列走法。假设我们可以访问大量国际象棋对局的数据集，但所有这些对局都来自 Elo 评分低于 2000 的玩家（玩家技能的标准衡量标准）[5]。如果我们仅通过最小化基于这些对局的标记预测错误来训练国际象棋代理，我们可能会将代理的性能限制在这些低于 2000 Elo 的玩家的能力范围内。这种方法本质上会将代理优化为模仿这些玩家的平均或典型玩法，可能会融入他们的错误和次优策略。这种现象可以被称为“智能上限”，这一概念可以从最近的离线强化学习和模仿学习研究中严格推导出来 [10]。在这种情况下，代理受到其学习演示的质量的限制，无法超越其训练数据中存在的技能水平。这一限制突显了人工智能开发中的一个关键挑战：如何使系统超越其训练数据的界限，并开发出新颖的、可能更优越的策略。

相反，当数据被用来发展对国际象棋动态的更深层次理解或“世界模型”时，它可能为发展超越训练数据中观察到的行为的复杂策略和战术铺平道路。世界模型代表了代理对环境的理解，在这种情况下，国际象棋规则，即一步走法将如何改变游戏状态以及给定走法的获胜机会。学习和完善这个世界模型，加上模拟潜在结果的能力，可能会使 AI 代理超越 2000 Elo 的基准。这些内部世界模型提供的模拟能力将实现深度思考（模拟），从而增强代理的推理和泛化能力。基于模型的策略，如蒙特卡罗树搜索（MCTS），是这种方法的经典例证 [23]。向系统 2 类型推理的过渡，如 ChatGPT o1 可能展示的那样，可能依赖于建立某种类型的世界模型并利用强化学习（奖励最大化）而不是仅仅最小化预测错误。这种方法的转变可能是 ChatGPT o1 增强推理能力背后的关键过渡技术之一。

通过将大型语言模型的预测能力与强化学习和世界建模的战略深度相结合，像 o1 这样的 AI 系统可以潜在地参与更复杂的问题解决和决策过程。这种混合方法允许快速模式识别（类似于系统 1 思维）和深思熟虑的逐步推理（系统 2 思维的特征），可能解释了 o1 中观察到的显著性能飞跃。

第二个挑战，从计算复杂性的角度来看，是大型语言模型（LLM）本质上在二次计算复杂性的约束下运行 [13]。当 LLM 遇到多步数学挑战时，这一限制变得尤为明显。然而，“思维链”概念为缓解这一约束提供了潜在的解决方案 [32]。它通过一系列“思维”输出来扩展响应，因此允许一定数量的额外计算资源；它本质上充当支持写入但缺乏删除或覆盖能力的“有限内存”。虽然这种方法显示出前景，但它仍然不足以实现完全动态的内存系统，并且没有原生地融入解码阶段。这一必要性突显了对超越当前 Transformer 解码器网络能力的高级计算架构的需求。确实，有必要在推理和解码阶段实施类似于蒙特卡罗树搜索（MCTS）的基于模型的策略 [6]。

这种高级推理时间计算系统将使 AI 模型能够维护并动态更新问题空间的表示，从而促进更复杂的推理过程。这种方法 [3] 与认知科学中的工作记忆概念一致，这对于复杂问题解决和深思熟虑的思维至关重要。通过整合这些能力，AI 系统可以潜在地模拟多步前进，评估不同场景，并做出更明智的决策——模仿人类专家推理中观察到的深思熟虑过程。

# 3 LLM 推理作为马尔可夫决策过程

为了在诸如问答或问题解决等任务中建模推理过程，我们使用 Q $\rightarrow$ {R} $\rightarrow$ A 序列来构建推理任务，其中：

* **Q:** 表示启动推理过程的问题或提示。
* **R:** 表示模型生成的中间推理步骤序列，以构建解决方案。
* **A:** 表示在推理步骤之后生成的最终答案或解决方案。

这种结构允许 LLM 生成一系列推理步骤，这些步骤在逻辑上将问题 $Q$ 连接到最终答案 $A$。

我们可以将推理过程定义为马尔可夫决策过程（MDP）[1]。MDP 表示提供了一个灵活的框架来建模推理。它允许模型自回归地生成顺序推理步骤以得出最终答案，同时通过在每个步骤采样多个路径来实现树结构，以探索替代推理轨迹。通过结合顺序和分支推理两种方法，模型可以探索多样化的解决方案，创建一个多功能且全面的推理过程。

我们现在可以描述推理过程的状态、动作、策略和奖励，其中 LLM 的任务是逐步生成与推理步骤和最终答案相对应的连贯标记序列。

在时间步 $t$ 的状态 $s_{t}$ 表示推理过程的当前状态，包括问题和到目前为止生成的推理步骤。形式上，状态定义为：

$$s_{t}=(Q,R_{1},\ldots,R_{t-1}),$$

其中 $Q$ 是初始问题或提示，$R_{1},\ldots,R_{t-1}$ 是到时间步 $t$ 为止生成的推理步骤。初始状态 $s_{0}$ 仅包含问题：

$$s_{0}=Q.$$

随着推理的进行，中间状态包括问题和到目前为止生成的推理步骤。该过程持续到生成最终答案为止。

在时间步 $t$ 的动作 $a_{t}\in A$ 对应于选择下一个推理步骤或最终答案。动作空间 $A$ 由两种类型的动作组成：

* **推理步骤 (R):** 该动作选择一个推理步骤 $R_{t}$ 附加到当前状态。
* **最终答案 (A):** 该动作选择最终答案 $A$，从而结束推理过程。

对于中间步骤，动作为：

$$a_{t}=R_{t},$$

新状态变为：

$$s_{t+1}=s_{t}+R_{t}.$$

对于最后一步，动作选择最终答案：

$$a_{T}=A,$$

最终状态变为：

图 3：在这个 MDP 公式中，LLM 的任务是以逐步的方式生成推理步骤和问题的最终答案。LLM 策略通过生成标记来操作，这些标记形成更高级别的推理结构。状态表示到目前为止的推理步骤序列，动作对应于选择新的推理步骤或最终答案。LLM 策略管理动作的选择，过程奖励模型（PRM）提供关于推理步骤和最终答案质量的反馈。通过优化策略以最大化奖励，LLM 可以在 PRM 的指导下生成准确且有意义的推理过程。

策略 $\pi$ 定义了模型用于选择下一个动作（即推理步骤或最终答案）的策略，给定当前状态。策略本质上是 LLM，在训练期间学习，并表示在给定到目前为止生成的标记的条件下，可能的推理步骤或最终答案的概率分布：

$$\pi_{LLM}(a_{t}\mid s_{t})=P(a_{t}\mid Q,R_{1},\ldots,R_{t-1}).$$

在每个时间步，模型使用此策略根据当前状态选择下一个动作，逐步构建最终答案。

鉴于 LLM 的自回归性质，从一个状态到下一个状态的转换是确定性的，并且也是给定的。下一个状态 $s_{t+1}$ 完全由将所选动作 $a_{t}$（推理步骤或最终答案）附加到当前状态 $s_{t}$ 决定。因此，转换函数为：

$$s_{t+1}=s_{t}+a_{t}.$$

这意味着一旦选择了推理步骤 $R_{t}$ 或最终答案 $A$，状态 $s_{t+1}$ 就通过将此动作连接到现有标记序列来唯一定义。

奖励提供关于生成的推理步骤和最终答案质量的反馈。在这种情况下，奖励是在模型生成推理步骤和最终答案时获得的。奖励可以定义为：

* **中间奖励:** 对于生成正确或有意义的推理步骤，中间奖励被分配正值。不正确或不相关的步骤可能会产生负奖励。
* **最终奖励:** 当模型生成正确的最终答案 $A$ 时，给予最大的奖励，完成推理过程。

因此，每个时间步 $t$ 的奖励为：

$$v_{t}=v(\mathbf{R}_{t}\mid Q,R_{1},\ldots,R_{t-1}),$$

对于最后一步：

$$v_{T}=v(A\mid Q,R_{1},\ldots,R_{n}).$$

模型学习优化其策略，以最大化整个推理过程中的累积预期奖励。

**标记生成与推理之间的关系** LLM 在两个层次上同时操作：标记生成层次以及推理步骤和最终答案层次。在最精细的层次上，LLM 自回归地生成标记，这意味着它一次生成一个标记，条件是先前生成的标记：

$$P(x_{t}\mid x_{1},x_{2},\ldots,x_{t-1}).$$

在每个时间步 $t$，LLM 从其词汇表中生成一个标记 $x_{t}$，基于先前标记提供的上下文。这些标记形成更高级别的结构，如推理步骤 $R_{t}$ 和最终答案 $A$。

* **推理步骤 (R):** 每个推理步骤 $R_{t}$ 由 LLM 生成的一系列标记 $\{x_{t_{1}},x_{t_{2}},\ldots,x_{t_{k}}\}$ 组成。这些标记代表推理过程中的一个连贯步骤，如逻辑推理或中间结论。
* **最终答案 (A):** 最终答案 $A$ 同样由一系列标记组成，形成问题的解决方案或响应。一旦 LLM 生成了足够的推理步骤，它就会以自回归的方式逐个标记生成最终答案。

我们现在可以准确定义 LLM 的世界模型：

**定义 1** (LLM 的世界模型): _LLM 的世界模型定义为 ($\mathcal{T},\mathcal{R}$)，其中：_

* *_转换模型_ $\mathcal{T}(s_{t},a_{t})$ _是确定性的，因为下一个状态_ $s_{t+1}$ _由当前状态_ $s_{t}$ _和动作_ $a_{t}$ _（即生成的标记或推理步骤）唯一确定，因此：_
$$s_{t+1}=s_{t}+a_{t}.$$

* *$\mathcal{V}(s_{t},a_{t})$ _是过程奖励模型（PRM），它评估在状态_ $s_{t}$ _中采取的动作_ $a_{t}$ _的质量。它反映了生成的推理步骤或标记在推进到最终答案过程中的适当性或有效性：_
$$\mathcal{V}(s_{t},a_{t})=v_{t}.$$

_由于转换是确定性的并且直接遵循策略，过程奖励模型（PRM）$\mathcal{R}(s_{t},a_{t})$ 封装了 LLM 与其环境之间的整个交互，评估每个推理步骤或标记对达到最终答案的贡献。_

# 4 实际实施

接下来，我们研究如何收集中间推理数据，使用它来训练过程奖励模型（PRM），利用 PRM 来训练 LLM 策略，并在解码阶段指导推理过程。

## 4.1 自动获取推理步骤数据

如前所述，我们需要推理轨迹来激发高级推理，同时涵盖广泛的任务。为了微调 LLM，我们通常有 {Q 和 A} 对，但缺乏底层推理步骤 {R} 的真实情况；

一种直接的方法是手动由人类标记推理步骤 [27, 12]。然而，一种特别有效的方法是在不需要人类监督的情况下收集数据并改进 LLM 推理，即自教导推理器（STaR）技术 [34] 等。在这种方法中，模型自主生成中间推理步骤，并使用它们来验证其内部推理能力。该方法建立在 LLM 从问题 $Q$ 到最终答案 $A$ 进行推理的能力之上，通过生成中间步骤 $\{R_{1},R_{2},\ldots,R_{n}\}$ 并使用模型自己的策略验证其正确性。具体来说，该方法首先使用 LLM 的策略（可能添加少量提示），记为 $\pi_{\text{LLM}}$，生成推理步骤 $\{R\}$，条件是初始问题 $Q$ 和最终答案 $A$。这种生成可以表示为：

$$\{R\}\sim\pi_{\text{LLM}}(\cdot\,|\,Q,A),$$

其中 LLM 生成一系列中间推理步骤 $\{R_{1},R_{2},\ldots,R_{n}\}$，旨在将问题 $Q$ 逻辑连接到正确的最终答案 $A$。这些步骤作为推理任务的内部分解形式，这对于复杂的多步问题至关重要，其中直接的问题-答案对可能不足以训练模型进行有效推理。

一旦生成了中间推理步骤 $\{R\}$，下一阶段涉及验证其正确性。这是通过再次使用 LLM 的策略来检查推理步骤，当与原始问题 $Q$ 结合时，是否导致正确答案 $A$。形式上，此验证步骤表示为：

$$A^{\prime}\sim\pi_{\text{LLM}}(\cdot\,|\,Q,\{R\}),$$

其中，$A^{\prime}$ 是模型基于问题 $Q$ 和生成的推理步骤 $\{R\}$ 对答案的预测。如果 $A^{\prime}$ 与原始正确答案 $A$ 匹配，则认为推理步骤 $\{R\}$ 有效。因此，$\{R\}$ 的正确性由条件 $A^{\prime}\approx A$ 决定。这种自我验证机制使模型能够自主识别正确的推理步骤，增强其内部逻辑一致性，而无需外部反馈。

收集的新推理步骤 $\{Q,\{R\},A\}$ 可用于进一步训练 LLM 的策略 $\pi_{\text{LLM}}$，增强生成有效推理步骤的能力。此迭代过程可以表示为：

$$\pi_{\text{LLM}}\leftarrow\pi_{\text{LLM}}+\text{来自 }\{Q,\{R\},A\}\text{ 的反馈}.$$

对于较长的推理序列，采用诸如蒙特卡罗树搜索（MCTS）[6, 15] 等技术来引导 LLM 策略以更细粒度的方式高效地找到正确的推理步骤。这些基于树的方法通过探索各种可能性并在每个推理阶段模拟多个结果，帮助找到最佳推理路径。这对于数学问题解决和基于代理的决策等复杂任务特别有用，其中中间步骤有多个路径。

## 4.2 自我强化训练

如图 4 所示，PRM $v(s)$ 和 LLM 策略 ($\pi_{LLM}$) 可以相互强化以改进自身，接下来将对此进行解释。

### 4.2.1 PRM 的值迭代

一旦收集了推理数据，下一步就是训练世界模型，也称为过程奖励模型（PRM），即由于状态转换是确定的和已知的，重点转移到学习一个通用的奖励模型，该模型可以稍后用于指导搜索、推理和解码过程。这个奖励模型，通常称为验证器，记为 $v_{\text{PRM}}(s)$，可以使用带注释的推理步骤数据集进行训练。训练通常涉及基于推理步骤正确性优化分类损失函数 [15]：

$$\mathcal{L}_{\text{PRM}}=\sum_{i=1}^{N}\left[\hat{v}_{i}\log v_{i}+(1-\hat{v}_ {i})\log(1-v_{i})\right],$$

其中 $v_{i}=r_{i}$ 表示第 $i$ 个示例步骤的正确性标签，指示该示例的推理过程是否正确。验证器的预测 $\hat{v}_{i}(s)$ 是 PRM 对状态 $s$ 的输出分数，表示推理步骤或最终答案的奖励。由于这是一种分类方法，因此没有区分中间步骤的奖励和它可能导致的潜在奖励，并且所有推理步骤都被假定为独立的。模型仅评估推理步骤或答案在该过程中的正确性，以统一的方式处理所有奖励，而不考虑中间步骤的未来影响。

然而，另一种方法是将 PRM 视为可以通过值迭代方法训练的值函数，使其能够预测累积奖励并通过最优动作选择指导推理过程 [6]。考虑一个推理过程，其中状态 $s$ 表示当前的推理状态，包含所有先前的推理步骤。值迭代方法的目标是学习一个参数为 $\theta$ 的值函数 $V_{\theta}(s)$，该函数预测从状态 $s$ 开始的预期累积奖励。该值函数通过评估不同动作的潜在结果来指导推理过程。$r_{\phi}(s)$ - 奖励函数，根据中间推理步骤或最终答案的正确性为状态 $s$ 分配标量奖励。$\gamma$ 是折扣因子，决定未来奖励的相对重要性。PRM 的贝尔曼方程 [1] 为：

$$V_{\theta}(s)=r(s)+\gamma\max_{a}V_{\theta}(a+s),$$

其中 $s^{\prime}=a+s$ 是通过在状态 $s$ 中采取动作 $a$ 达到的下一个状态。奖励函数 $r(s)$ 可以是稀疏的，仅对正确结论提供奖励，也可以是密集的，对中间步骤提供部分奖励。我们定义用于学习值函数参数 $\theta$ 的 TD 损失函数为当前值与贝尔曼目标之间的平方误差：

$$L(\theta)=\sum_{i=1}^{N}\left(V_{\theta}(s_{i})-\left[r(s_{i})+\gamma\max_{a} V_{\theta}(s_{i}+a)\right]\right)^{2}.$$

然后，我们可以通过使用梯度下降或其他优化技术最小化损失 $L(\theta)$ 来获得值函数的参数 $\theta$。

### 4.2.2 LLM 策略的策略迭代

一旦获得 PRM，就可以训练 LLM 策略以增强推理能力。这需要超越传统监督学习框架的方法。PRM 通过结合在线强化学习来优化推理任务，在此过程中起着至关重要的作用 [18]。然而，典型的 RLHF 工作，如 [18]，可以用于但可能不适用于大型语言模型训练。

让我们看看组相对策略优化（GRPO）[22]。我们假设对于每个问题 $Q=q$，策略生成推理步骤 $\{o_{1},o_{2},...,o_{G}\}$，每个输出 $o_{i}$ 由多个步骤 $\{a_{i,1},a_{i,2},\ldots,a_{i,K_{i}}\}$ 组成，其中 $K_{i}$ 是输出 $o_{i}$ 中推理步骤（或标记）的总数。我们稍微滥用之前的符号，使用 $\alpha$ 表示所有输出，包括推理步骤和最终答案。我们现在可以制定 GRPO 优化，通过 PRM 学习 LLM 策略如下。

对于每个问题 $q$，GRPO 从旧策略 $\pi_{\theta_{\textrm{old}}}$ 中采样一组输出 $\{o_{1},o_{2},\ldots,o_{G}\}$，目标是通过最大化以下目标来优化策略：

$$\begin{split} J_{\textrm{GRPO}}(\theta)=\mathbb{E}_{q\sim P(Q ),\{o_{t}\}_{t=1}^{G}\sim\pi_{\theta_{\textrm{old}}}(\mathcal{O}|q)}\left[ \frac{1}{G}\sum_{i=1}^{G}\frac{1}{K_{i}}\sum_{t=1}^{K_{i}}\min\left(\hat{\rho}_{ i,t}A_{i,t},\textrm{clip}\left(\hat{\rho}_{i,t},1-\epsilon,1+\epsilon\right)A_{i ,t}\right)-\beta D_{\textrm{KL}}\left(\pi_{\theta}\|\pi_{\theta_{\textrm{at}}} \right)\right],\end{split}$$

其中：

* *$q\sim P(Q)$ 表示从问题分布 $P(Q)$ 中采样问题 $q$，
* *$\{o_{t}\}_{t=1}^{G}\sim\pi_{\theta_{\textrm{old}}}(\mathcal{O}|q)$ 表示从旧策略 $\pi_{\theta_{\textrm{old}}}$ 中采样的一组输出，
* *$\hat{\rho}_{i,t}=\frac{\pi_{\theta}(a_{i,t}|q,o_{t},<_{t})}{\pi_{\theta_{ \textrm{old}}}(a_{i,t}|q,o_{t},<_{t})}$ 是输出 $o_{i}$ 中步骤 $t$ 的动作 $a_{i,t}$ 的重要性权重（概率比），
* *$A_{i,t}$ 是输出 $o_{i}$ 中推理步骤 $t$ 的优势，基于相对奖励计算（见下文），
* *$\epsilon$ 是防止过度更新的裁剪参数（如 PPO [21]），
* *$\beta$ 是控制 KL 正则化强度的超参数，
* *$D_{\textrm{KL}}\left(\pi_{\theta}\|\pi_{\theta_{\textrm{at}}}\right)$ 是训练策略 $\pi_{\theta}$ 与参考策略 $\pi_{\theta_{\textrm{at}}}$ 之间的 KL 散度，用作正则化。

输出 $o_{i}$ 中步骤 $t$ 的动作 $a_{i,t}$ 的优势函数 $A_{i,t}$ 基于推理步骤和最终步骤的奖励计算。奖励使用组中所有输出的奖励进行归一化。设输出 $o_{i}$ 中步骤 $t$ 的归一化奖励为：

$$\bar{r}_{i}^{(t)}=\frac{r_{i}^{(t)}-\text{mean}(R)}{\text{std}(R)},$$

其中

$$R=\left\{{\left\{r_{1}^{\text{index}(1)},\ldots,r_{1}^{\text{index}(K_{1})} \right\}},\ldots,{\left\{r_{G}^{\text{index}(1)},\ldots,r_{G}^{\text{index}(K_{G })}\right\}}\right\},$$

表示组 $G$ 中所有输出中所有推理步骤的奖励，其中 $\text{index}(j)$ 是第 $j$ 步的结束标记索引，$K_{i}$ 是第 $i$ 个输出中的总步数；$\text{mean}(R)$ 和 $\text{std}(R)$ 是组奖励的均值和标准差。

输出 $o_{i}$ 中步骤 $t$ 的优势 $A_{i,t}$ 是从步骤 $t$ 到最终步骤 $K_{i}$ 的归一化奖励之和：

$$A_{i,t}=\sum_{j=t}^{K_{i}}\bar{r}_{i}^{(j)},$$

其中 $\bar{r}_{i}^{(j)}$ 是输出 $o_{i}$ 中推理步骤 $j$ 的归一化奖励。该优势函数鼓励模型优化中间推理步骤和最终步骤，通过奖励在组内表现更好的推理路径。

GRPO 不是直接将 KL 惩罚纳入奖励，而是通过将当前策略 $\pi_{\theta}$ 与参考策略 $\pi_{\theta_{\text{eff}}}$ 之间的 KL 散度直接添加到损失函数中来正则化策略。这确保更新后的策略在训练期间不会过度偏离参考策略，有助于保持稳定性。

这种 GRPO 公式专门针对具有过程奖励模型的推理任务进行了调整，通过利用推理步骤和最终步骤的组相对奖励来优化 LLM 策略。归一化优势函数基于相对性能计算，鼓励策略在采样输出组中表现更好的输出。此外，KL 正则化确保更新后的策略保持接近参考策略，提高训练稳定性和效率。该框架为通过 PRM 优化指导 LLM 推理提供了稳健的方法。

可以探索更高效的离线方法，如无 PRM 但具有顺序推理数据的标记级 DPO [35]。有关详细信息，请参阅论文。

## 4.3 推理时间计算

一旦训练完成，LLM 策略必须在推理期间高效生成输出。自回归生成——根据先前的标记逐个预测标记——在 LLM 中广泛使用。然而，对于推理任务，需要更复杂的解码技术。

为了在效率和有效性之间取得平衡，工作 [33, 24] 发现推理任务受益于更灵活的方法，如束搜索。在束搜索中，同时生成多个可能的序列（或束），并根据累积概率选择最佳候选。对于更复杂的推理任务，使用前瞻模型，如 MCTS。MCTS [6] 模拟多个推理路径并根据奖励系统评估它们，选择具有最高预期奖励的路径。这允许模型在推理期间探索更广泛的可能性，增加其达到最优解决方案的机会。使用 MDP，我们可以正式定义推理过程结构。

**定义 2** (原生思维链): _原生思维链（NCoT）是指大型语言模型（LLM）固有的推理能力，使其能够自主执行逐步、结构化推理，而无需外部提示。这种能力被形式化为马尔可夫决策过程（MDP）$(\mathcal{S},\mathcal{A},\pi,\mathcal{R})$，其中：_

* *$\mathcal{S}$ _是状态空间，表示到给定点为止生成的标记或推理步骤序列。_
* *$\mathcal{A}$ _是动作空间，由潜在的推理步骤_ $R_{t}$ _或最终答案_ $A$ _组成。_
* *$\pi_{LLM}(a_{t}\mid s_{t})$ _是策略（也是 LLM），它管理动作的选择，根据当前状态_ $s_{t}$ _确定下一个推理步骤或最终答案。_
* *$\mathcal{R}(s_{t},a_{t})$ _是过程奖励模型（PRM），它根据所选动作_ $a_{t}$ _的质量和相关性分配奖励_ $r_{t}$ _，指导推理过程。_

_模型可以通过展开 MDP 遵循顺序推理路径，或者通过在每个状态采样不同的推理步骤来探索多个轨迹，形成树状结构（图 5）。过程奖励模型 $\mathcal{R}$ 提供了对该空间的引导搜索，通过支持导致更有意义或正确推理步骤的动作来控制推理轨迹。_

# 5 文献评论

在文献中，推理时间计算、验证器（也称为奖励模型）和数据获取方法受到了广泛关注，所有这些在增强这些模型的推理能力方面都起着关键作用。在本节中，我们回顾并讨论这些领域的几篇关键论文，探讨它们的贡献和局限性。这些工作与更广泛的研究领域之间的联系如图 6 所示。

## 5.1 推理时间计算

几篇论文专注于通过推理时间计算优化 LLM 推理。例如，论文 [6] 介绍了一种将蒙特卡罗树搜索（MCTS）与 LLM 解码相结合的方法，这种组合在指导推理方面非常有效，特别是对于复杂的多步任务。MCTS 的加入通过模拟潜在的未来行动，增强了模型规划下一步的能力。同样，论文 [24] 强调了优化测试时间计算的重要性，经验表明推理时间推理增强通常比简单地扩展模型参数产生更显著的改进。这反映了越来越多的理解，即在推理期间更多的计算可以用于更高质量的推理，而不一定增加模型的大小。

另一种方法在 [7] 中提出，建议使用暂停标记来强制模型在推理期间暂停并“思考”。该方法引入了一种隐式推理模型，鼓励 LLM 以块的形式处理信息，模仿人类般的深思熟虑。

## 5.2 验证器模型

验证器模型（结果奖励模型和过程奖励模型）已成为提高 LLM 推理可靠性的重要研究领域。像 [4] 这样的论文介绍了在数学推理任务中使用验证器的最早正式尝试（仅结果奖励），为后续研究奠定了基础。后续工作 [27] 扩展了验证器的概念，集成了基于过程的推理机制，随后是 OpenAI 在过程奖励模型（PRM）上的工作 [12]。这些验证器在确保多步推理的正确性方面起着至关重要的作用，解决了 LLM 中的一个主要挑战——在扩展推理序列中保持连贯性和准确性。

最近对这一研究线的补充是 [11]，它将验证器模型与多数投票相结合，在推理任务中产生更可靠的输出。该方法通过交叉检查多个推理路径并过滤掉错误步骤，增强了验证过程的鲁棒性。这些进展突显了验证器在保持 LLM 准确性方面的重要性，因为它们应对日益复杂的推理挑战。

## 5.3 推理任务的数据获取

推理数据的获取一直是另一个关注的领域，特别是在 [34] 等论文中，探讨了自动获取与推理步骤相关数据的方法。STaR 引入了一种自我教导范式，模型通过生成和批评自己的步骤来提高其推理能力，从而产生更可靠的中间步骤。论文 [30] 进一步推进了这一方法，展示了如何在没有昂贵的人工注释的情况下逐步训练 LLM，为推理数据问题提供了更具扩展性的解决方案。

[31] 的工作强调了推理任务中实际数据获取的重要性，特别是在编码问题中。MCTS 已在 [6] 中用于获取数据，而在 [15] 中已扩展为线性搜索以提高效率。

这些论文表明，为了使 LLM 在推理方面取得进展，创新的数据获取方法，如自监督学习和验证机制，对于减少对广泛人工标记数据集的依赖至关重要。

## 5.4 理解和系统级改进

最后，越来越多的研究旨在理解 LLM 中逐步推理的机制 [26, 19]。[25] 的工作从图形模型的角度分析了思维链机制。论文 [19] 探讨了推理作为 LLM 自然能力出现的内在原因。它表明推理是语言模型处理局部经验和知识方式的副产品。论文 [14] 对 LLM 批判自身推理的能力进行了实证评估，表明自我批判能力通常有限，并且这种能力通常仅在模型足够大时才会出现。

从系统角度来看，pangu-agent 论文 [3] 引入了超越传统模型（如 OpenAI 的 o1 模型）的结构化推理机制。这项研究反映了向更通用推理代理的转变，这些代理可以处理更广泛的任务，具有更高的精度和灵活性，提供了下一代推理模型的愿景。

# 附录 A LLM 的标准训练流程

LLM 的训练过程通常涉及几个阶段，每个阶段都建立在前一个阶段的基础上。在预训练阶段，模型使用自回归语言建模目标在大量在线语料库上进行训练。目标是给定先前的标记预测下一个标记。对于给定的标记序列 $\{x_{1},x_{2},\ldots,x_{T}\}$，标记级交叉熵损失对每个位置的真实标记的负对数概率求和：

$$\mathcal{L}_{\text{pretrain}}=-\sum_{t=1}^{T}\log P(x_{t}|x_{<t};\theta),$$

其中 $x_{t}$ 是第 $t$ 个标记，$x_{<t}$ 表示 $t$ 之前的所有标记，$\theta$ 是模型参数，$P$ 是词汇表上的概率分布 [2]。$p(x_{t}\mid x_{<t})$ 是给定所有先前标记 $x_{<t}$ 的真实标记 $x_{t}$ 的概率。该损失衡量模型预测序列中每个标记的能力。

预训练后，模型然后在收集的额外 {问题, 答案} 对上进行微调。目标是最大化给定问题的正确答案的似然：

$$\mathcal{L}_{\text{finetune}}=-\sum_{i=1}^{N}\log P(A_{i}|Q_{i};\theta),$$

其中 $Q_{i}$ 和 $A_{i}$ 分别是第 $i$ 个问题和答案对 [20]。

接下来，应用来自人类反馈的强化学习（RLHF）[18] 以进一步提高模型的指令遵循能力。这涉及构建一个奖励模型 $R(x)$（通过成对训练数据），该模型估计模型输出的质量。然后使用诸如近端策略优化（PPO）[21] 等方法优化策略（语言模型）：

$$\mathcal{L}_{\text{RLHF}}=\mathbb{E}[R(Q,A)]-\beta\cdot\text{KL}(\pi_{\theta}( Q|A)\|\pi_{\theta_{\text{old}}}(Q|A)),$$

其中 $\pi_{\theta}$ 是当前策略，$\pi_{\theta_{\text{old}}}$ 是旧策略，$\beta$ 是控制 KL 散度惩罚强度的超参数。

在 RLHF 阶段，模型通过与人类反馈进行交互，进一步优化其输出。具体来说，RLHF 通过以下步骤实现：

1. **数据收集**：收集人类对模型输出的反馈数据，通常以成对比较的形式（例如，两个输出中选择更好的一个）。
2. **奖励模型训练**：使用收集到的反馈数据训练一个奖励模型 $R(x)$，该模型能够评估模型输出的质量。
3. **策略优化**：使用奖励模型对语言模型进行优化，通常使用近端策略优化（PPO）[21] 等强化学习算法。优化的目标是在最大化奖励的同时，避免策略与旧策略之间的过大偏差。

RLHF 的损失函数可以表示为：

$$\mathcal{L}_{\text{RLHF}}=\mathbb{E}[R(Q,A)]-\beta\cdot\text{KL}(\pi_{\theta}( Q|A)\|\pi_{\theta_{\text{old}}}(Q|A)),$$

其中 $\pi_{\theta}$ 是当前策略，$\pi_{\theta_{\text{old}}}$ 是旧策略，$\beta$ 是控制 KL 散度惩罚强度的超参数。

通过 RLHF，模型能够更好地遵循人类指令，生成更符合人类期望的输出。这一过程在 ChatGPT 等模型中得到了广泛应用，显著提升了模型的交互能力和实用性。

# 附录 B 推理时间计算的优化

在推理阶段，LLM 需要高效地生成输出。传统的自回归生成方法虽然简单，但在处理复杂推理任务时可能效率较低。为了提高推理效率，研究者提出了多种优化方法，包括束搜索、蒙特卡罗树搜索（MCTS）等。

## B.1 束搜索

束搜索是一种广泛使用的解码技术，它通过同时生成多个候选序列（称为“束”），并根据累积概率选择最佳候选。束搜索的优点是能够在生成过程中保留多个可能的路径，从而增加找到最优解的机会。然而，束搜索的计算复杂度较高，特别是在生成长序列时。

## B.2 蒙特卡罗树搜索（MCTS）

MCTS 是一种更复杂的解码技术，特别适用于多步推理任务。MCTS 通过模拟多个推理路径，并根据奖励系统评估这些路径，选择具有最高预期奖励的路径。MCTS 的优点在于它能够探索更广泛的推理空间，从而在复杂任务中找到更优的解决方案。

在 LLM 中，MCTS 可以与自回归生成结合使用，形成一种混合推理方法。具体来说，模型可以在每个推理步骤中使用 MCTS 来评估多个可能的动作，并选择最优的动作进行下一步推理。这种方法能够显著提升模型的推理能力，特别是在需要多步推理的任务中。

## B.3 推理时间计算的未来方向

随着 LLM 的不断发展，推理时间计算的优化将成为研究的重要方向。未来的研究可能会探索更高效的解码算法，结合强化学习和世界模型的方法，以进一步提升模型的推理能力。此外，硬件加速和分布式计算技术的进步也将为推理时间计算提供更多的可能性。

# 附录 D 图片说明

图 1：推理时间计算。(a) 自回归 LLM 直接根据给定问题 (Q) 生成答案 (A)。(b) 思维链的概念，即逐步思考，涉及在得出最终答案 (A) 之前加入中间推理步骤 (R)。这些重复操作允许：1) 重新审视和修订先前的输出，2) 推进到后续推理阶段，3) 探索多个推理路径或轨迹。

图 2：人类认知与 LLM 的类比。(a) 和 (b) 人类的有意识或无意识行为依赖于部分不同的大脑回路。(a) 人类的无意识控制由少数专门的大脑区域维持，如前脑岛和前补充运动区（pre-SMA）。(b) 而有意识控制则涉及更广泛的网络，激活顶叶和额叶内的许多区域 [28]。无意识控制通常是快速和本能的，通常由自动过程驱动，而有意识控制则倾向于涉及更深思熟虑、计算和深入的思考，允许仔细反思和彻底分析。

图 3：在这个 MDP 公式中，LLM 的任务是以逐步的方式生成推理步骤和问题的最终答案。LLM 策略通过生成标记来操作，这些标记形成更高级别的推理结构。状态表示到目前为止的推理步骤序列，动作对应于选择新的推理步骤或最终答案。LLM 策略管理动作的选择，过程奖励模型（PRM）提供关于推理步骤和最终答案质量的反馈。通过优化策略以最大化奖励，LLM 可以在 PRM 的指导下生成准确且有意义的推理过程。

图 4：结合 PRM 的值函数与 LLM 的策略生成，确保引导和控制的结果。在训练期间，LLM 策略生成的输出与 PRM 提供的评估相互强化，导致两个组件的持续自我改进和优化。

图 5：通过 PRM，LLM 可以通过三种方法执行非自回归推理：1) 采样多个推理轨迹，2) 在潜在推理路径的树结构上执行蒙特卡罗搜索，或 3) 结合两种方法以增强推理的灵活性和鲁棒性。

图 6：LLM 原生思维链的研究。该图展示了与 LLM 推理能力相关的关键研究领域及其相互关系。主要研究领域包括推理时间计算、验证器模型、数据获取方法以及系统级改进。这些领域的研究共同推动了 LLM 在复杂推理任务中的性能提升。

# 附录 E 表格说明

## 表 1：推理步骤数据收集方法

| 方法                   | 描述                           | 优点                             | 缺点                     |
| ---------------------- | ------------------------------ | -------------------------------- | ------------------------ |
| 手动标注               | 由人类专家手动标注推理步骤     | 准确性高，适用于小规模数据       | 成本高，难以扩展         |
| 自教导推理器（STaR）   | 模型自主生成推理步骤并自我验证 | 无需人工监督，可扩展性强         | 依赖于模型的初始推理能力 |
| 蒙特卡罗树搜索（MCTS） | 通过模拟多个推理路径来收集数据 | 适用于复杂任务，能够找到最优路径 | 计算复杂度高，耗时较长   |

## 表 2：推理时间计算优化方法

| 方法                   | 描述                               | 优点                                     | 缺点                   |
| ---------------------- | ---------------------------------- | ---------------------------------------- | ---------------------- |
| 束搜索                 | 同时生成多个候选序列，选择最佳候选 | 增加找到最优解的机会                     | 计算复杂度较高         |
| 蒙特卡罗树搜索（MCTS） | 模拟多个推理路径，选择最优路径     | 适用于复杂任务，能够探索更广泛的推理空间 | 计算复杂度高，耗时较长 |
| 暂停标记               | 强制模型在推理过程中暂停并“思考”   | 模仿人类深思熟虑的过程，提升推理质量     | 可能增加推理时间       |

## 表 3：验证器模型类型

| 类型                | 描述                               | 优点                         | 缺点                 |
| ------------------- | ---------------------------------- | ---------------------------- | -------------------- |
| 结果奖励模型        | 仅评估最终答案的正确性             | 简单易实现                   | 无法评估中间推理步骤 |
| 过程奖励模型（PRM） | 评估中间推理步骤和最终答案的正确性 | 提升多步推理的连贯性和准确性 | 实现复杂度较高       |
| 多数投票验证器      | 结合多个推理路径的验证结果         | 提高验证的鲁棒性，减少错误   | 计算复杂度较高       |

# 附录 F 未来研究方向

随着 LLM 的不断发展，未来的研究可能会集中在以下几个方向：

1. **推理时间计算的进一步优化**：探索更高效的解码算法，结合强化学习和世界模型的方法，以进一步提升模型的推理能力。
2. **数据获取方法的创新**：开发更多自监督学习和自动化数据获取技术，减少对人工标注数据的依赖。
3. **验证器模型的改进**：研究更复杂的验证器模型，能够更准确地评估中间推理步骤和最终答案的质量。
4. **系统级推理能力的提升**：开发更通用的推理代理，能够处理更广泛的任务，并具备更高的精度和灵活性。
5. **硬件加速和分布式计算**：利用硬件加速和分布式计算技术，进一步提升 LLM 的推理效率和性能。

# 附录 G 结论

本文全面回顾了 LLM 在推理能力方面的最新进展，特别是 ChatGPT o1 的突破性创新。我们讨论了自回归 LLM 的挑战，提出了将推理过程建模为马尔可夫决策过程（MDP）的方法，并探讨了如何通过过程奖励模型（PRM）和强化学习来优化 LLM 的推理能力。此外，我们还介绍了推理时间计算的优化方法，并展望了未来的研究方向。

通过结合大型语言模型的预测能力与强化学习和世界建模的战略深度，AI 系统可以潜在地参与更复杂的问题解决和决策过程。这种混合方法允许快速模式识别（类似于系统 1 思维）和深思熟虑的逐步推理（系统 2 思维的特征），可能解释了 o1 中观察到的显著性能飞跃。

未来的研究将继续推动 LLM 在推理能力方面的进步，使其能够更好地应对复杂的现实世界任务，为人工智能的发展开辟新的可能性。

# 附录 I 术语表

| 术语 | 解释                                                         |
| ---- | ------------------------------------------------------------ |
| LLM  | 大型语言模型（Large Language Model）                         |
| MDP  | 马尔可夫决策过程（Markov Decision Process）                  |
| PRM  | 过程奖励模型（Process-Reward Model）                         |
| RLHF | 来自人类反馈的强化学习（Reinforcement Learning from Human Feedback） |
| MCTS | 蒙特卡罗树搜索（Monte Carlo Tree Search）                    |
| STaR | 自教导推理器（Self-Taught Reasoner）                         |
| PPO  | 近端策略优化（Proximal Policy Optimization）                 |
