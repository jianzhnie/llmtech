# Autonomous AI Agents

##  Autonomous Agents 是什么

### Agent 是什么

Agent通常指的是使用语言模型作为推理引擎并将其连接到两个关键组件的想法：工具和内存。

工具有助于将LLMs连接到其他数据或计算源。工具的示例包括搜索引擎、API 和其他数据存储。工具很有用，因为LLMs只了解他们所接受的训练知识。这些知识很快就会过时。为了克服这一限制，工具可以获取最新数据并将其作为上下文插入到提示中。工具还可以用于采取行动（例如运行代码、修改文件等），然后LLMs可以观察该行动的结果，并将其纳入他们下一步做什么的决定中。

记忆可以帮助Agent记住之前的交互。这些交互可以是与其他实体（人类或其他Agent）或工具的交互。这些记忆可以是短期的（例如，之前 5 种工具使用情况的列表），也可以是长期的（过去与当前情况最相似的工具使用情况）。

### Autonomous Agents 是什么

Autonomous Agents是由人工智能驱动的程序，当给定目标时，它们能够为自己创建任务、完成任务、创建新任务、重新确定任务列表的优先级、完成新的首要任务，并循环直到达到目标。

> "[智能]Autonomous Agent是一般自动化的自然终点。原则上， Agent可用于任何其他流程的自动化。一旦这些 Agent变得高度复杂和可靠，很容易想象自动化在各个领域和行业的指数级增长"。
>
> Bojan Tunguz, Machine Learning at NVIDIA



> “The future of autonomous agents looks like everybody becoming a manager.”
>
> Yohei Nakajima, creator of BabyAGI



>"这是 "原始的 AGI"。只需将 LLM 封装在一个循环中，就能得到一个自主的 Agent，它可以自己进行推理、计划、思考、记忆和学习。它展示了在正确的结构和提示下，LLM 所能发挥的尚未开发的能力和灵活性。整个概念提出还不到一个月，所以我迫不及待地想看到在能力越来越强的 LLM 基础上建立起来的越来越复杂的 Agent如何影响世界。
>
>Siqi Chen, Founder and CEO of Runway



> "这将很快改变许多行业。有了Autonomous Agent，人们同时做很多事情将变得更加容易。只要给它一个任务，它就能完成。迄今为止，这是一个如此强大的概念......”
>
> Barsee, Founder of The AI Valley Newsletter



> "我认为，我们最初将拥有针对特定垂直领域的Autonomous Agent，这些Agent将根据特定的数据集进行微调，使其能够在该领域发挥作用。迄今为止，我们看到大量采用 LLM 的两个领域是文案和编程。进一步推断，这两个领域的人工智能将开始变得更加 autonomous 。在不久的将来，可能会出现的一种情况是，人工智能每天都会自主地向你提出新的建议，供你审阅，而无需你先开始或提示，而不是由人类发出提示来初始化文案写作或代码完成。”
>
> Lonis Hamaili, Creator of godmode.space

##  Autonomous Agents 能做什么

除了分析目标和提出任务外，Autonomous Agent还可以具备一系列能力，使其能够完成人类可以完成的任何数字任务，例如

- 浏览互联网和使用应用程序

- 长期和短期记忆

- 控制电脑

- 使用信用卡或其他支付方式

- 访问大型语言模型 (LLM)，如用于分析、总结、观点和答案的 GPT。

此外，这些Autonomous Agent将有各种形状和大小。有些会在用户不知情的情况下进行幕后操作，而有些则是可见的，就像上面的例子，用户可以跟随着人工智能的每一个 "想法"。

## Autonomous Agents 如何工作

![img](https://media.beehiiv.com/cdn-cgi/image/fit=scale-down,format=auto,onerror=redirect,quality=80/uploads/asset/file/4a72cdad-debc-4ec4-b5a0-9efbb92bfedd/1_jlH0NK5y1lDFoYT0NNEYtw.png)

您已经深入了解了Autonomous Agents的工作原理，但我认为为您提供一个版本的整体框架以及逐步分解几个Autonomous Agents的示例将会有所帮助。

首先，这是一个Autonomous Agents的通用框架：

1. 初始化目标：定义人工智能的目标。
2. 任务创建：AI 检查其内存中最近完成的 X 个任务（如果有），然后使用其目标以及最近完成的任务的上下文来生成新任务列表。
3. 任务执行：AI自主执行任务。
4. 内存存储：任务和执行结果存储在向量数据库中。
5. 反馈收集：人工智能以外部数据或人工智能内部对话的形式收集已完成任务的反馈。该反馈将用于通知自适应过程循环的下一次迭代。
6. 新任务生成：人工智能根据收集的反馈和内部对话生成新任务。
7. 任务优先级：人工智能通过检查目标并查看最后完成的任务来重新确定任务列表的优先级。
8. 任务选择：AI 从优先级列表中选择最重要的任务，并按照步骤 3 中的描述继续执行它们。
9. 迭代：人工智能连续循环重复步骤 4 到 8，使系统能够根据新信息、反馈和不断变化的需求进行调整和发展。

#### Example #1: Social Media Manager Autonomous Agent

比方说，您不需要聘请社交媒体经理来管理您的社交媒体账户，而是希望有一个Autonomous Agent来为您完成所有工作，而且成本很低，还能全天候提供智能服务。

以下是该 Autonomous Agent 的框架。

1. 初始化目标：设置初始参数，如目标受众、社交媒体平台、内容类别和发布频率。
2. 收集数据：收集有关过往社交媒体帖子、用户互动和特定平台趋势的数据。这可能包括喜欢、分享、评论和其他参与指标。
3. 内容分析：分析收集到的数据，找出与目标受众相关的模式、热门话题、标签和影响者。这一步可能涉及自然语言处理和机器学习技术，以了解内容及其上下文。
4. 内容创建：根据分析结果生成内容创意，并根据平台和受众偏好创建社交媒体帖子。这可能涉及使用人工智能生成的文本、图片或视频，以及整合用户生成的内容或从其他来源策划的内容。
5. 安排时间：根据特定平台的趋势、受众活动和所需频率，确定发布每条内容的最佳时间。据此安排发布时间。
6. 性能监控：跟踪每个帖子在参与度指标方面的表现，如点赞、分享、评论和点击率。如有可能，收集用户反馈，进一步完善对受众偏好的了解。
7. 迭代和改进：分析性能数据和用户反馈，找出需要改进的地方。更新内容策略、创建和安排流程，将这些见解纳入其中。通过第 2-7 步的迭代，不断完善社交媒体管理系统，并随着时间的推移提高其效率。

## Autonomous Agents List

| 名称                                | 方案        | 特点                                                         | UI   | 链接                                                |
| :---------------------------------- | ----------- | ------------------------------------------------------------ | ---- | --------------------------------------------------- |
| AutoGPT                             | 最复杂      | 基于GPT4 的 开源实现，Star 数量最多。网络搜索收集信息， 长期和短期内存管理 | No   | https://github.com/Significant-Gravitas/Auto-GPT    |
| AgentGPT                            | 不需要Token | 通过简单的网页访问易于使用，具有相对简单的功能，基于 OpenAI 的模型 | Yes  | https://agentgpt.reworkd.ai/                        |
| Cognosys                            | 不需要Token | 性能不错，具有明确的任务组织，使用体验也和Godmode比较接近    | Yes  | http://cognosys.ai/                                 |
| Godmode                             | 不需要Token | 操作更加直观，每个步骤需要用户review，可以选择性的接受或者拒绝Agents提供的Plan，同时也可以随时给出自己的Feedback让Agents别走偏。 | Yes  | https://godmode.space/                              |
| NexusGPT                            |             | NexusGPT——世界首个人工智能自由职业者平台                     | Yes  | https://nexus.snikpic.io/                           |
| AIAgent                             |             | 类似基于 Web 的解决方案，如 AgentGPT                         | Yes  | https://aiagent.app/                                |
| HuggingGPT/TaskMatrix               |             | 由（LLM）和作为协作执行器的众多专家模型组成，用LLM以及Prompt工程阶段性的解决多模态问题 | No   |                                                     |
| Autonomous Agents Hackathon Summary |             | 使用 SuperAGI、AutoGPT、BabyAGI、Langchain 等框架构建 Autonomous Agents项目！ |      | https://lablab.ai/event/autonomous-agents-hackathon |

### LangChain

LLM工具的开源鼻祖，目标是辅助大家开发LLM应用，Agents、Tools、Plugin、Memory、Data Augmented早早地就提出开源了，为Autonomous Agents生态的爆发奠定了非常扎实的工作基础，值得给予最大的respect！

###  JARVIS | HuggingGPT

Jarvis 或 HuggingGPT 是一个协作系统，由作为中央控制器的大型语言模型（LLM）和作为协作执行器的众多专家模型组成，用LLM以及Prompt工程阶段性的解决多模态问题，这些模型都来自Hugging Face Hub。该Agent可以使用 LLM，也可以使用其他模型。该系统的工作流程包括四个阶段：

- 任务规划：使用 ChatGPT 分析用户请求以辨别意图，并将其分解为可管理的任务。

- 模型选择：为了解决给定的任务，ChatGPT 会根据描述从 "Hugging Face "中选择最合适的专家模型。比如：文生图的任务分配给Stable Diffusion模型、图生图分配给ControlNet、图文问答分配给Blip等等

- 任务执行：调用并执行每个选定的模型，然后将结果返回给 ChatGPT。

- 生成响应：最后，它使用 ChatGPT 整合所有模型的预测结果，并生成综合响应。

存储库： https://github.com/microsoft/JARVIS


### AutoGPT

一个基于 GPT-4 的实验性开源 Agent库。它将 LLM 的 "思想 "串联起来，自主完成您设定的任何任务。Auto-GPT 是首批完全自主运行 GPT-4 的平台之一，推动了人工智能的发展。

特点

- 上网查询和收集信息
- 长期和短期内存管理
- 用于生成文本的 GPT-4 实例
- 访问常用网站和平台
- 使用 GPT-3.5 进行文件存储和汇总

资源库: https://github.com/Significant-Gravitas/Auto-GPT

开发者： https://www.significantgravitas.com/

### BabyAGI

BabyAGI 是一个人工智能驱动的任务管理系统。该系统使用 OpenAI 和 Pinecone API 来创建、优先处理和执行任务。Baby AGI 的魅力在于它能根据先前任务的结果自主解决任务，并保持预定义的目标。它还能高效地确定任务的优先级。 

BabyAGI 是一个非常优雅的项目，初始版本仅用105行代码就实现了Baby版本的AGI。这个项目同时也是后续几个商业化项目的重要参考工作。

工作模式

- 从任务列表中提取第一个任务。
- 将任务发送给执行Agent，执行Agent根据上下文使用 OpenAI 的 API 和 Llama 完成任务。
- 丰富结果并将其存储在 Pinecone 中。
- 创建新任务，并根据目标和前一项任务的结果重新调整任务列表的优先级。

存储库：https://github.com/yoheinakajima/babyagi

### AgentGPT

之所以把 AgentGPT & Godmode & Cognosys 这几个项目放在一起，主要是这几个都是带UI界面的非常用户友好的Autonomous Agents项目，像ChatGPT那样使用起来顺滑。

特点

- 基于浏览器
- 简单易用
- 基于 OpenAI 模型
- 测试使用无需 OpenAI 密钥
- 平台：https://agentgpt.reworkd.ai/
  

AgentGPT：如果你想找一个开源的带UI界面的 Autonomous Agents项目，那就来找他吧。界面做的非常简洁大方，新的功能也在不断提PR中。

Godmode：同样是一个带用户界面的 Autonomous Agents，不过代码没有开源。正如其名字，使用这个产品的时候确实有点老板的感觉，需要做的就是review！相比AgentGPT，其对设置项的细粒度更进一步，比如可以选择性的接受或者拒绝Agents提供的Plan，同时也可以随时给出自己的Feedback让Agents别走偏。

Cognosys：与Godmode一样没有开源，不过代码没有开源。使用体验也和Godmode比较接近，同属于Autonomous Agents的UI化。

### NexusGPT

NexusGPT——世界首个人工智能自由职业者平台！AI的发展需要或者会朝着AI平权(民主)、AI个性化的角度去发展，即让人人都可以享受AI的便利，人人都有一个自己的AI助理！而这个对应的就是 Autonomous Agents。

未来可能每个人或自己制作极度个性化的 Autonomous Agents，当然也可以选择『雇佣』一些『专业』的Autonomous Agents来临时的完成自己或者公司需要的某些任务。