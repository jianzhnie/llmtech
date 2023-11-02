# LLM研究中的挑战

在我的生活中，我从未见过这么多聪明的人致力于同一个目标：使LLMs变得更好。在与许多在工业和学术界工作的人交谈之后，我注意到了10个主要的研究方向。前两个方向，即幻觉和上下文学习，可能是当今最受关注的。我对第3（多模态）、第5（新架构）和第6（GPU替代品）最感兴趣。

LLM研究中的开放性挑战

1. 减少和衡量幻觉
2. 优化上下文长度和上下文构建
3. 整合其他数据模态
4. 使LLMs更快更便宜
5. 设计新的模型架构
6. 发展GPU替代品
7. 使Agent可用
8. 改进从人类偏好中学习
9. 提高聊天界面的效率
10. 为非英语语言构建LLMs


## 1. 减少和衡量幻觉

幻觉是一个已经被广泛讨论的话题，所以我会简要地介绍一下。幻觉发生在当人工智能模型胡编乱造时。在许多创意用例中，幻觉是一个特性。然而，在大多数其他用例中，幻觉是一个缺陷。最近，我参加了一个关于LLM的讨论面板，与Dropbox、Langchain、Elastics和Anthropic等公司的代表交流，他们认为公司采用LLMs进入生产的主要障碍是幻觉。

减轻幻觉和开发用于衡量幻觉的指标是一个新兴的研究课题，我看到许多初创公司专注于解决这个问题。还有一些临时的方法可以减少幻觉，比如在提示中添加更多的上下文、思维链、自我一致性，或者要求模型在回答时简明扼要。

了解更多关于幻觉的信息：

- [Survey of Hallucination in Natural Language Generation](https://arxiv.org/abs/2202.03629) (Ji et al., 2022)
- [How Language Model Hallucinations Can Snowball](https://arxiv.org/abs/2305.13534) (Zhang et al., 2023)
- [A Multitask, Multilingual, Multimodal Evaluation of ChatGPT on Reasoning, Hallucination, and Interactivity](https://arxiv.org/abs/2302.04023) (Bang et al., 2023)
- [Contrastive Learning Reduces Hallucination in Conversations](https://arxiv.org/abs/2212.10400) (Sun et al., 2022)
- [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171) (Wang et al., 2022)
- [SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models](https://arxiv.org/abs/2303.08896) (Manakul et al., 2023)
- A simple example of fact-checking and hallucination by [NVIDIA’s NeMo-Guardrails](https://github.com/NVIDIA/NeMo-Guardrails/blob/main/examples/grounding_rail/README.md#grounding-fact-checking-and-hallucination)



## 2. 优化上下文长度和上下文构建

绝大多数问题需要上下文。例如，如果我们问ChatGPT：“最好的越南餐厅是哪家？”，所需的上下文将是“在哪里”，因为越南最好的餐厅与美国最好的越南餐厅是不同的。

根据这篇有趣的论文《SituatedQA》（Zhang＆Choi，2021年），相当大比例的信息寻求问题具有依赖于上下文的答案，例如，在Natural Questions NQ-Open数据集中约有16.5％的问题。就企业用例而言，我个人怀疑这个百分比可能更高。例如，假设一家公司为客户支持构建了一个聊天机器人，为了使该聊天机器人能够回答客户关于任何产品的任何问题，所需的上下文可能是客户的历史记录或该产品的信息。

由于模型是从提供给它的上下文“学习”的，这个过程也被称为上下文学习。

![Context needed for a customer support query](https://huyenchip.com/assets/pics/llm-research/2-context.png)


上下文长度对于RAG（检索增强生成）尤为重要，RAG已经成为LLM行业用例中主导模式。对于那些还没有沉浸在RAG狂潮中的人来说，RAG分为两个阶段：

第一阶段：分块（也称为索引）

- 收集您希望LLM使用的所有文档
- 将这些文档划分为可以馈送到LLM以生成嵌入并将这些嵌入存储在矢量数据库中的块。

第二阶段：查询

1. 当用户发送查询（例如“我的保险单支付这种药物X吗？”）时，您的LLM将此查询转换为嵌入，我们称之为QUERY_EMBEDDING
2. 您的矢量数据库获取与QUERY_EMBEDDING最相似的块的块

Jerry Liu在LlamaIndex（2023年）的演讲中的截图

特定客户支持查询所需的上下文


上下文长度越长，我们就能在上下文中塞入更多的块。模型可以访问的信息越多，它的响应就会更好，对吧？

但不总是。模型可以使用多少上下文和它将如何有效地使用上下文是两个不同的问题。与增加

模型上下文长度的努力并行的是使上下文更有效的努力。有些人称之为“提示工程”或“提示构建”。例如，最近广泛传播的一篇论文是关于模型在索引的开头和末尾比在中间理解信息要好得多的发现 -《迷失在中间：语言模型如何使用长上下文》（Liu等，2023年）。

3. 整合其他数据模态

在我看来，多模态非常强大，但却常常被低估。有许多原因支持多模态。

首先，许多用例需要多模态数据，特别是在处理各种数据模态的行业，例如医疗保健、机器人技术、电子商务、零售、游戏、娱乐等。例如：

- 医学预测通常需要文本（例如医生的笔记、患者的问卷）和图像（例如CT、X射线、MRI扫描）。
- 产品元数据通常包含图像、视频、描述，甚至是表格数据（例如生产日期、重量、颜色）。您可能希望根据用户的评价或产品照片自动填写缺失的产品信息。您可能希望使用户能够使用视觉信息（如形状或颜色）搜索产品。

其次，多模态承诺了模型性能的大幅提升。一个既能理解文本又能理解图像的模型应该比只能理解文本的模型表现得更好，对吗？基于文本的模型需要大量文本，以至于我们很容易担心我们将很快用完用于训练基于文本的模型的互联网数据。一旦我们用完了文本，我们就需要利用其他数据模态。

![Multimodal Flamingo's architecture](https://huyenchip.com/assets/pics/llm-research/3-flamingo.png)

> Flamingo architecture (Alayrac et al., 2022)



酷炫的多模态工作：

- [[CLIP\] Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) (OpenAI, 2021)
- [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198) (DeepMind, 2022)
- [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597) (Salesforce, 2023)
- [KOSMOS-1: Language Is Not All You Need: Aligning Perception with Language Models](https://arxiv.org/abs/2302.14045) (Microsoft, 2023)
- [PaLM-E: An embodied multimodal language model](https://ai.googleblog.com/2023/03/palm-e-embodied-multimodal-language.html) (Google, 2023)
- [LLaVA: Visual Instruction Tuning](https://arxiv.org/abs/2304.08485) (Liu et al., 2023)
- [NeVA: NeMo Vision and Language Assistant](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/playground/models/neva) (NVIDIA, 2023)



## 使LLMs更快更便宜

当GPT-3.5在2022年11月底首次发布时，很多人对在生产中使用它的延迟和成本感到担忧。然而，自那时以来，延迟/成本分析迅速发生了变化。在半年内，社区找到了一种方法，创建了一个性能与GPT-3.5非常接近的模型，但是它所需的内存空间仅为GPT-3.5的2％左右。

我的结论是：如果你创造了足够好的东西，人们会找到一种方法使它快速、便宜。

| **Date** | **Model**                                                    | **# params** | **Quantization** | **Memory to finetune** | **Can be trained on** |
| -------- | ------------------------------------------------------------ | ------------ | ---------------- | ---------------------- | --------------------- |
| Nov 2022 | GPT-3.5                                                      | 175B         | 16-bit           | 375GB                  | Many, many machines   |
| Mar 2023 | [Alpaca 7B](https://crfm.stanford.edu/2023/03/13/alpaca.html) | 7B           | 16-bit           | 15GB                   | Gaming desktop        |
| May 2023 | [Guanaco 7B](https://arxiv.org/abs/2305.14314)               | 7B           | 4-bit            | 6GB                    | Any Macbook           |

下面是Guanaco 7B的性能与ChatGPT GPT-3.5和GPT-4的比较，据Guanco论文报道。注意：总的来说，性能比较远非完美。LLM评估非常非常困难。

<img src="https://huyenchip.com/assets/pics/llm-research/4-llm-optimization.png" alt="Guanaco 7B's performance compared to ChatGPT GPT-3.5 and GPT-4" style="zoom:50%;" />

四年前，当我开始撰写后来成为《设计机器学习系统》一书的“模型压缩”章节时，我写了四种主要的模型优化/压缩技术：

1. 量化：目前最通用的模型优化方法。量化通过使用较少的位来表示模型的参数，例如，使用16位而不是32位来表示浮点数，甚至使用4位。
2. 知识蒸馏：这是一种方法，其中一个小模型（学生）被训练成模仿一个较大模型或模型集合（老师）。
3. 低秩分解：这里的关键思想是用低维张量替换高维张量以减少参数数量。例如，您可以将 3x3 张量分解为 3x1 和 1x3 张量的乘积，这样您就只有 6 个参数，而不是 9 个参数。
4. **减枝**

所有这四种技术在今天仍然相关且流行。羊驼通过知识蒸馏进行训练。QLoRA 使用低秩分解和量化的组合。

## 5.设计新的模型架构

自 2012 年 AlexNet 以来，我们看到了许多架构的流行和过时，包括 LSTM、seq2seq。与这些相比，Transformer 的粘性令人难以置信。它从 2017 年就已经存在了。这种架构还能流行多久还是一个很大的问号。

开发一种新的架构来超越 Transformer 并不容易。Transformer 在过去 6 年里进行了大力优化。这种新架构必须在人们关心的硬件上以人们今天关心的规模执行。旁注：[Transformer 最初是由 Google 设计的，旨在在 TPU 上快速运行](https://timdettmers.com/2018/10/17/tpus-vs-gpus-for-transformers-bert/)，后来才在 GPU 上进行了优化。

Chris Ré 实验室在 2021 年围绕 S4 进行了很多令人兴奋的事情 - 请参阅使用[结构化状态空间对长序列进行高效建模](https://arxiv.org/abs/2111.00396)（Gu 等人，2021）。我不太清楚发生了什么事。Chris Ré 的实验室仍然大力投资开发新架构，最近与初创公司[Together合作开发了](https://together.ai/blog/monarch-mixer)[Monarch Mixer](https://together.ai/blog/monarch-mixer)架构（Fu 等人，2023）。

他们的关键思想是，对于现有的 Transformer 架构，注意力的复杂度是序列长度的二次方，而 MLP 的复杂度是模型维度的二次方。具有次二次复杂度的架构会更加高效。

![Monarch 混音器架构](https://huyenchip.com/assets/pics/llm-research/5-monarch-mixer.png)



我确信许多其他实验室正在研究这个想法，尽管我不知道有任何公开的尝试。如果您知道，请告诉我！

## 6. 开发 GPU 替代品

自 2012 年 AlexNet 以来，GPU 一直是深度学习的主导硬件。事实上，AlexNet 受欢迎的一个公认原因是它是第一篇成功使用 GPU 训练神经网络的论文。在 GPU 出现之前，如果你想训练 AlexNet 规模的模型，你必须使用数千个 CPU，就像[Google 在 AlexNet 之前几个月发布的](https://www.nytimes.com/2012/06/26/technology/in-a-big-network-of-computers-evidence-of-machine-learning.html)那样。与数千个 CPU 相比，几个 GPU 更适合博士使用。学生和研究人员，掀起了深度学习研究热潮。

在过去的十年中，许多公司，无论是大公司还是初创公司，都尝试为人工智能创建新的硬件。最值得注意的尝试是 Google 的[TPU](https://cloud.google.com/tpu/docs/intro-to-tpu)、Graphcore 的[IPU](https://www.graphcore.ai/products/ipu)（IPU 发生了什么？）和[Cerebras](https://www.eetimes.com/cerebras-sells-100-million-ai-supercomputer-plans-8-more/)。SambaNova 筹集了超过[10 亿美元来开发新的人工智能芯片](https://spectrum.ieee.org/sambanova-ceo-ai-interview)，但似乎已转向成为一个生成式人工智能平台。

一段时间以来，人们对量子计算抱有很多期待，主要参与者是：

- [IBM的QPU](https://www.ibm.com/quantum)
- 谷歌的量子计算机今年早些时候在《自然》杂志上报道了[减少量子误差的一个重要里程碑。](https://www.nature.com/articles/d41586-023-00536-w)[其量子虚拟机可通过Google Colab](https://quantumai.google/quantum-virtual-machine)公开访问
- 研究实验室如[麻省理工学院量子工程中心](https://cqe.mit.edu/)、[马克斯·普朗克量子光学研究所](https://www.mpq.mpg.de/en)、[芝加哥量子交易所](https://chicagoquantum.org/)、[橡树岭国家实验室](https://quantum-roadmap.ornl.gov/)等。

另一个非常令人兴奋的方向是光子芯片。这是我最不了解的方向——所以如果我错了，请纠正我。如今现有的芯片使用电力来移动数据，这会消耗大量电力并且还会产生延迟。光子芯片使用光子来移动数据，利用光速进行更快、更高效的计算。该领域的多家初创公司已筹集了数亿美元，包括[Lightmatter](https://lightmatter.co/)（2.7 亿美元）、[Ayar Labs](https://ayarlabs.com/)（2.2 亿美元）、[Lightelligence](https://www.lightelligence.ai/)（2 亿美元以上）和[Luminous Compute](https://www.luminous.com/)（1.15 亿美元）。

[以下是光子矩阵计算中三种主要方法的进展时间表，摘自《光子矩阵乘法点亮光子加速器及其他》](https://www.nature.com/articles/s41377-022-00717-8)论文（Zhou et al., Nature 2022）。三种不同的方法是平面光转换（PLC）、马赫-曾德尔干涉仪（MZI）和波分复用（WDM）。

![光子矩阵乘法三种主要方法的进展时间表](https://huyenchip.com/assets/pics/llm-research/6-photonic-matrix-multiplication.png)

## 7. 使Agent可用

Agent是可以采取行动的LLM，例如浏览互联网、发送电子邮件、进行预订等。与本文中的其他研究方向相比，这可能是最年轻的方向。

由于Agent的新颖性和巨大潜力，人们对Agent产生了狂热的痴迷。按星星数量计算，[Auto-GPT现在是有史以来第 25 位最受欢迎的 GitHub 存储库。](https://github.com/Significant-Gravitas/Auto-GPT)[GPT-Engineering](https://github.com/AntonOsika/gpt-engineer)是另一个流行的存储库。

尽管令人兴奋，但人们仍然怀疑LLM是否足够可靠和表现出色，可以被赋予采取行动的权力。

不过，已经出现的一个用例是使用Agent进行社会研究，就像著名的斯坦福实验一样，该实验表明，一个由生成Agent组成的小社会会产生紧急的社会行为：例如，从一个用户指定的单一概念开始，一个*Agent想要举办一个情人节派对，智能体在接下来的两天内自主地发出派对邀请，结识新朋友，互相询问参加派对的日期……（生成智能体：人类行为的交互式模拟，Park 等人*，[2017](https://arxiv.org/abs/2304.03442)） ，2023）

该领域最著名的初创公司可能是 Adept，它由两位 Transformer 合著者（尽管[两人都已经离开](https://www.theinformation.com/briefings/two-co-founders-of-adept-an-openai-rival-suddenly-left-to-start-another-company)）和一位前 OpenAI 副总裁创立，迄今为止已筹集了近 5 亿美元。去年，他们做了一个演示，展示了他们的Agent浏览互联网并向 Salesforce 添加新帐户。我期待看到他们的新演示 🙂

## 8. 根据人类偏好改进学习

[RLHF，根据人类偏好进行强化学习](https://huyenchip.com/2023/05/02/rlhf.html)，很酷，但有点老套。如果人们找到更好的方法来训练LLM，我不会感到惊讶。RLHF 有许多悬而未决的问题，例如：

**1. 如何用数学方式表示人类的偏好？**

目前，人类的偏好是通过比较来确定的：人类标记者确定响应 A 是否优于响应 B。但是，它没有考虑响应 A 比响应 B 好多少。

**2. 人类的偏好是什么？**

Anthropic 沿着三个轴测量了模型响应的质量：有帮助、诚实和无害。请参阅[宪法人工智能：人工智能反馈的无害性](https://arxiv.org/abs/2212.08073)（Bai et al., 2022）。

DeepMind 尝试生成令大多数人满意的响应。请参阅[微调语言模型以找到具有不同偏好的人类之间的一致性](https://www.deepmind.com/publications/fine-tuning-language-models-to-find-agreement-among-humans-with-diverse-preferences)（Bakker et al., 2022）。

此外，我们想要的是能够表明立场的人工智能，还是回避任何潜在争议话题的普通人工智能？

**3. 考虑到文化、宗教、政治倾向等的差异，谁的偏好是“人类”偏好？**

获取能够充分代表所有潜在用户的训练数据存在很多挑战。

例如，对于OpenAI的InstructGPT数据，没有65岁以上的标注者。贴标机主要是菲律宾人和孟加拉国人。请参阅[InstructGPT：训练语言模型以遵循人类反馈的指令](https://arxiv.org/abs/2203.02155)（Ouyang et al., 2022）。

![InstructGPT 标签人员的人口统计](https://huyenchip.com/assets/pics/llm-research/8-instructgpt-demographics.png)



社区主导的努力虽然其意图令人钦佩，但可能会导致数据出现偏差。例如，对于 OpenAssistant 数据集，222 名受访者中有 201 名 (90.5%) 认为自己是男性。[杰里米·霍华德 (Jeremy Howard) 在 Twitter 上就此发表了精彩的帖子](https://twitter.com/jeremyphoward/status/1647763133665271808/photo/1)。

![OpenAssistant 数据集贡献者的自我报告人口统计数据](https://huyenchip.com/assets/pics/llm-research/8-openassistant-demographics.png)



## 9.提高聊天界面效率

自 ChatGPT 以来，关于聊天是否是适合各种任务的界面存在多次讨论。

- [Natural language is the lazy user interface](https://austinhenley.com/blog/naturallanguageui.html) (Austin Z. Henley, 2023)
- [Why Chatbots Are Not the Future](https://wattenberger.com/thoughts/boo-chatbots) (Amelia Wattenberger, 2023)
- [What Types of Questions Require Conversation to Answer? A Case Study of AskReddit Questions](https://arxiv.org/abs/2303.17710) (Huang et al., 2023)
- [AI chat interfaces could become the primary user interface to read documentation](https://idratherbewriting.com/blog/ai-chat-interfaces-are-the-new-user-interface-for-docs) (Tom Johnson, 2023)
- [Interacting with LLMs with Minimal Chat](https://eugeneyan.com/writing/llm-ux/) (Eugene Yan, 2023)

然而，这并不是一个新的讨论。在许多国家，特别是在亚洲，聊天作为超级应用程序的界面已有大约十年的历史。[Dan Grover 早在 2014 年就曾进行过这样的讨论](http://dangrover.com/blog/2014/12/01/chinese-mobile-app-ui-trends.html)。

![十多年来，聊天一直被用作中国超级应用程序的通用界面](https://huyenchip.com/assets/pics/llm-research/9-superapp-chat-interface.png)

> 聊天作为中国应用程序的通用界面（Dan Grover，2014)



2016 年，讨论再次变得紧张，当时许多人认为应用程序已死，聊天机器人将成为未来。

- [以聊天为界面](https://acroll.medium.com/on-chat-as-interface-92a68d2bf854)（Alistair Croll，2016）
- [聊天机器人趋势是一个很大的误解吗？](https://www.technologyreview.com/2016/04/25/8510/is-the-chatbot-trend-one-big-misunderstanding/)（威尔·奈特，2016）
- [机器人不会取代应用程序。更好的应用程序将取代应用程序](http://dangrover.com/blog/2016/04/20/bots-wont-replace-apps.html)（Dan Grover，2016）

就我个人而言，我喜欢聊天界面，原因如下：

1. 聊天是每个人，甚至以前没有接触过计算机或互联网的人，都可以快速学会使用的界面。2010 年代初，当我在肯尼亚的一个低收入住宅区（我们可以说贫民窟吗？）做志愿者时，我惊讶地发现那里的每个人都可以轻松地通过手机和短信进行银行业务。那个街区没有人有电脑。
2. 聊天界面可访问。如果您的双手很忙，您可以使用语音代替文本。
3. 聊天也是一个非常强大的界面——你可以向它提出任何请求，它都会给出响应，即使响应不好。

然而，我认为聊天界面的某些方面还可以改进。

1. 每回合多条消息

   目前，我们几乎假设每回合一条消息。这不是我和我的朋友发短信的方式。通常，我需要多条消息来完成我的想法，因为我需要插入不同的数据（例如图像、位置、链接），我忘记了之前消息中的某些内容，或者我只是不想将所有内容放入一个大段落中。

2. 多模态输入

   在多模式应用领域，大部分精力都花在构建更好的模型上，而很少花在构建更好的界面上。以[Nvidia 的 NeVA 聊天机器人](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/playground/models/neva)为例。我不是用户体验专家，但我怀疑这里可能还有用户体验改进的空间。

   PS 很抱歉 NeVA 团队把你叫出来。即使有了这个界面，你的工作也超级酷！

   ![NVIDIA 的 NeVA 接口](https://huyenchip.com/assets/pics/llm-research/9-neva.png)

   

3. 将生成式人工智能纳入您的工作流程

   Linus Lee 在他的演讲《[聊天之外的生成人工智能界面》](https://www.youtube.com/watch?v=rd-J3hmycQs)中很好地阐述了这一点。例如，如果您想询问有关您正在处理的图表的某一列的问题，您应该只需指向该列并提出问题即可。

4. 编辑和删除消息

   编辑或删除用户输入将如何改变与聊天机器人的对话流程？

## 10. 建立非英语语言的LLM

我们知道，当前以英语为先的LLM在性能、延迟和速度方面都不适用于许多其他语言。

- [ChatGPT Beyond English: Towards a Comprehensive Evaluation of Large Language Models in Multilingual Learning](https://arxiv.org/abs/2304.05613) (Lai et al., 2023)
- [All languages are NOT created (tokenized) equal](https://blog.yenniejun.com/p/all-languages-are-not-created-tokenized) (Yennie Jun, 2023)

![非英语语言的标记化](https://huyenchip.com/assets/pics/llm-research/10-non-english-tokens.png)



以下是我所知道的一些举措。如果您有其他人的建议，我很乐意将其包含在这里。

- [Aya](https://aya.for.ai/): An Open Science Initiative to Accelerate Multilingual AI Progress
- [Symato](https://discord.gg/a2PCzB4AdE): Vietnamese ChatGPT
- [Cabrita](https://github.com/22-hours/cabrita): Finetuning InstructLLaMA with portuguese data
- [Luotuo-Chinese-LLM](https://github.com/LC1332/Luotuo-Chinese-LLM)
- [Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca)
- [Chinese-Vicuna](https://github.com/Facico/Chinese-Vicuna)

这篇文章的几位早期读者告诉我，他们认为我不应该包含这个方向，原因有两个。

1. 这与其说是一个研究问题，不如说是一个逻辑问题。我们已经知道该怎么做了。有人只需要投入金钱和精力即可。这并不完全正确。大多数语言被认为是低资源的，例如，与英语或中文相比，它们的高质量数据要少得多，并且可能需要不同的技术来训练大型语言模型。看：
   - [Low-resource Languages: A Review of Past Work and Future Challenges](https://arxiv.org/abs/2006.07264) (Magueresse et al., 2020)
   - [JW300: A Wide-Coverage Parallel Corpus for Low-Resource Languages](https://aclanthology.org/P19-1310/) (Agić et al., 2019)
2. 那些更悲观的人认为，未来，许多语言将消亡，互联网将由两种语言的两个宇宙组成：英语和普通话。这种思想流派并不新鲜——有人还记得世界语吗？

机器翻译和聊天机器人等人工智能工具对语言学习的影响仍不清楚。它们会帮助人们更快地学习新语言，还是会完全消除学习新语言的需要？

## 结论

如需另一个视角，请查看这篇综合性论文《[大型语言模型的挑战和应用》](https://arxiv.org/abs/2307.10169)（Kaddour 等人，2023 年）。

上面提到的一些问题比其他问题更难。例如，我认为第 10 点，即为非英语语言建立LLM，如果有足够的时间和资源，会更直接。

第一，减少幻觉会困难得多，因为幻觉只是LLM在做他们的概率性的事情。

第四，让LLM更快、更便宜，这个问题永远不会得到彻底解决。这方面已经取得了很大进展，而且还会有更多进展，但我们永远不会没有改进的空间。

第 5 点和第 6 点，即新架构和新硬件，非常具有挑战性，但随着时间的推移，它们是不可避免的。由于架构和硬件之间的共生关系——新架构需要针对通用硬件进行优化，而硬件需要支持通用架构——它们可能由同一家公司来解决。

其中一些问题仅靠技术知识是无法解决的。例如，第八点，即根据人类偏好改进学习，可能更多的是一个政策问题，而不是一个技术问题。第9点，提高聊天界面的效率，更多的是一个用户体验问题。我们需要更多非技术背景的人与我们一起解决这些问题。