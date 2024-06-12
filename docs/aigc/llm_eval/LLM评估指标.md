## LLM 评估指标

大型语言模型（LLM）评估指标，如答案正确性、语义相似性和幻觉，是基于您关心的标准对LLM系统输出进行评分的指标。这些指标帮助我们理解模型的能力和局限，包括但不限于以下几个方面：

![img](https://assets-global.website-files.com/64bd90bdba579d6cce245aec/66681237ee3fb1317a1838a7_llm%20evaluation%20metric.png)

> LLM 评估指标架构

以下是在将 LLM 系统投入生产之前可能需要的最重要和最常见的指标：

1. **答案相关性：** 确定 LLM 输出是否能够以信息丰富且简洁的方式解决给定的输入。
2. **正确性：**根据某些基本事实确定 LLM 输出是否正确。
3. **幻觉：**确定 LLM 输出是否包含虚假或编造的信息。
4. **上下文相关性：**确定基于 RAG 的 LLM 系统中的检索器是否能够提取与您的 LLM 最相关的信息作为上下文。
5. **负责任的指标：**包括偏见和毒性等指标，决定 LLM 输出是否包含（通常）有害和冒犯性内容。
6. **任务特定指标：**这可能取决于任务和应用程序的类型（摘要、翻译等），以及现有的专门指标，例如机器翻译的 BLEU 分数。



虽然大多数指标都是通用的，而且是必需的，但它们不足以针对特定用例。这就是为什么您需要至少一个自定义任务特定指标来使您的 LLM 评估流程做好生产准备（正如您将在后面的 G-Eval 部分中看到的那样）。例如，如果您的 LLM 应用程序旨在总结新闻文章的页面，那么您将需要一个自定义的 LLM 评估指标，该指标基于以下内容进行评分：

1. 摘要是否包含了足够的原文信息。
2. 摘要是否与原文存在矛盾或幻觉。

此外，如果您的 LLM 应用具有基于 RAG 的架构，您可能还需要对检索上下文的质量进行评分。关键是，LLM 评估指标根据其设计用于执行的任务来评估 LLM 应用。*（请注意，LLM 应用可以简单地是 LLM 本身！）*

优秀的评估指标应具备以下特点：

1. 定量。评估指标在评估手头任务时应始终计算得分。这种方法使您能够设置最低通过阈值，以确定您的LLM应用程序是否“足够好”，并允许您监控这些得分随着时间的推移如何变化，因为您迭代并改进实现。
2. **可靠。**尽管 LLM 的输出结果难以预测，但您最不希望看到的是 LLM 评估指标同样不稳定。因此，尽管使用 LLM（又称 LLM-Evals）评估的指标（例如 G-Eval）比传统评分方法更准确，但它们往往不一致，这是大多数 LLM-Evals 的不足之处。
3. **准确。**如果可靠的分数不能真正代表你的 LLM 应用表现，那么它们就毫无意义。事实上，让一个好的 LLM 评估指标变得出色的秘诀是让它尽可能地符合人类的期望。

所以问题就成了，LLM 评估指标如何计算出可靠和准确的分数？

## 计算指标分数的不同方法

有许多成熟的方法可用于计算指标分数——一些方法利用神经网络，包括嵌入模型和 LLM，而另一些则完全基于统计分析。

<img src="https://assets-global.website-files.com/64bd90bdba579d6cce245aec/65ae30bca9335d1c73650df0_metricsven.png" alt="img" style="zoom: 10%;" />

> 指标评分者的类型

我们将介绍每种方法，并在本节结束时讨论最佳方法，请继续阅读以找出答案！

### 统计得分

在我们开始之前，我想说的是，在我看来，统计评分方法并不重要，因此如果你急于了解，可以跳过直接阅读“G-Eval”部分。这是因为每当需要推理时，统计方法的表现都很差，这使得它作为大多数LLM评估标准的评分者来说不够准确。快速浏览一下：

- BLEU **（双语评估替补）评估您的LLM应用程序的输出与注释的真实情况（或预期输出）之间的匹配度。它计算LLM输出和预期输出之间每个匹配的n-gram（连续的n个词）的精确度，以计算它们的几何平均数，并在需要时应用简洁性惩罚。
- ROUGE **（面向召回率的摘要评估替补）主要用于评估NLP模型生成的文本摘要，通过比较LLM输出和预期输出之间n-gram的重叠来计算召回率。它确定参考文本中出现在LLM输出中的n-gram的比例（0-1）。
- METEOR **（具有明确排序的翻译评估指标）：更为全面，因为它通过评估LLM输出和预期输出之间的n-gram匹配（精确度）和重叠（召回率），并根据它们之间的词序差异进行调整来计算分数。它还利用外部语言数据库（如WordNet）来考虑同义词。最终得分是精确度和召回率的调和平均数，并对排序差异进行惩罚。
- **Levenshtein 距离**（或编辑距离，你可能认识到这是一个 LeetCode 难 DP 问题）计算将一个词或文本字符串更改为另一个所需的最小单字符编辑次数（插入、删除或替换），这对于评估拼写更正或在精确对齐字符至关重要的其他任务非常有用。

由于纯统计评分器几乎不考虑任何语义并且推理能力极其有限，因此它们对于评估通常很长且复杂的 LLM 输出来说不够准确。

### 基于模型的评分系统

纯粹基于统计的评分者可靠但不准确，因为它们很难考虑语义。在这一部分，情况恰恰相反——完全依赖于自然语言处理（NLP）模型的评分者相比之下更准确，但由于它们的概率性质，也更不可靠。

这并不令人惊讶，非基于LLM的评分者的表现不如LLM评估（LLM-Evals），原因与统计评分者相同。非LLM评分者包括：

- **NLI评分者**：使用自然语言推理模型（一种NLP分类模型）来分类LLM输出是否在逻辑上与给定的参考文本一致（蕴含）、矛盾或无关（中立）。分数通常在蕴含（值为1）和矛盾（值为0）之间，提供了逻辑一致性的度量。
- **BLEURT评分者**：使用像BERT这样的预训练模型来根据预期输出对LLM输出进行评分。

除了得分不一致之外，这些方法实际上还存在一些缺点。例如，NLI 评分器在处理长文本时也会面临准确性问题，而 BLEURT 则受到其训练数据的质量和代表性的限制。

因此我们开始讨论 LLM-Evals。

#### G-评估

G-Eval 是一个最近开发的框架，源自一篇题为“使用具有更好的人类对齐的 GPT-4 进行 NLG 评估”的论文，[该框架](https://arxiv.org/pdf/2303.16634.pdf)**使用 LLM 来评估 LLM 输出（又名 LLM-Evals），是创建特定于任务的指标的最佳方法之一。**

<img src="https://assets-global.website-files.com/64bd90bdba579d6cce245aec/65ae098fe2f794d0e9bede8e_Screenshot%202024-01-20%20at%203.03.25%20PM.png" alt="img" style="zoom:25%;" />

> G-Eval算法

[正如我在之前的一篇文章中介绍的那样](https://www.confident-ai.com/blog/a-gentle-introduction-to-llm-evaluation)，G-Eval 首先使用思路链 (CoT) 生成一系列评估步骤，然后使用生成的步骤通过填表范式确定最终分数（这只是一种花哨的说法，G-Eval 需要几条信息才能工作）。例如，使用 G-Eval 评估 LLM 输出连贯性需要构建一个提示，其中包含要评估的标准和文本以生成评估步骤，然后使用 LLM 根据这些步骤输出 1 到 5 的分数。

让我们用这个例子来运行一下 G-Eval 算法。首先，生成评估步骤：

1. 为你选择的 LLM 引入评估任务（例如，根据连贯性对该输出进行 1-5 评级）
2. 对你的标准给出一个定义（例如“连贯性——实际输出中所有句子的集体质量”）。

*(请注意，在原始的 G-Eval 论文中，作者仅使用 GPT-3.5 和 GPT-4 进行实验，并且在亲自尝试过 G-Eval 的不同 LLM 后，我强烈建议您坚持使用这些模型。)*

经过一系列的评估步骤后：

1. 通过将评估步骤与评估步骤中列出的所有参数连接起来来创建提示（例如，如果您希望评估 LLM 输出的连贯性，则 LLM 输出将是必需的参数）。
2. 在提示的最后，要求它生成 1-5 之间的分数，其中 5 比 1 好。
3. （可选）利用 LLM 输出 token 的概率对分数进行归一化，并对其加权求和作为最终结果。

步骤 3 是可选的，因为要获取输出 token 的概率，您需要访问原始模型嵌入，而这并不保证所有模型接口都可用。然而，本文引入了此步骤，因为它提供了更细粒度的分数并最大限度地减少了 LLM 评分中的偏差（如论文所述，已知 3 在 1-5 范围内具有更高的 token 概率）。

以下是论文的结果，显示了 G-Eval 如何胜过本文前面提到的所有传统的非 LLM 评估：

<img src="https://assets-global.website-files.com/64bd90bdba579d6cce245aec/65ae09a9af1ae6c21abc2d68_Screenshot%202024-01-14%20at%2010.59.43%20PM.png" alt="img" style="zoom:25%;" />

> Spearman 和 Kendall-Tau 相关性越高，表示与人类判断的一致性越高。

G-Eval 很棒，因为作为 LLM-Eval，它能够考虑到 LLM 输出的完整语义，从而使其更加准确。这很有道理——想想看，非 LLM 评估使用的评分器能力远不如 LLM，怎么可能理解 LLM 生成的文本的全部范围？

尽管与其他软件相比，G-Eval 与人类判断的相关性更高，但它仍然不可靠，因为要求LLM给出分数无疑是武断的。

话虽如此，考虑到 G-Eval 的评估标准的灵活性，我个人已将 G-Eval 作为[DeepEval 的指标，DeepEval 是我一直在研究的开源 LLM 评估框架](https://github.com/confident-ai/deepeval)（其中包括原始论文中的规范化技术）。

```bash
# Install
pip install deepeval
# Set OpenAI API key as env variable
export OPENAI_API_KEY="..."
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval

test_case = LLMTestCase(input="input to your LLM", actual_output="your LLM output")
coherence_metric = GEval(
    name="Coherence",
    criteria="Coherence - the collective quality of all sentences in the actual output",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
)

coherence_metric.measure(test_case)
print(coherence_metric.score)
print(coherence_metric.reason)
```

使用 LLM-Eval 的另一个主要优势是，LLM 能够为其评估分数提供理由。

#### Prometheus

Prometheus 是一个完全开源的 LLM，在提供适当的参考资料（参考答案、评分标准）的情况下，其评估能力可与 GPT-4 相媲美。它与 G-Eval 类似，也是用例无关的。Prometheus 是一个语言模型，使用[Llama-2-Chat作为基础模型，并在](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)[反馈集合](https://huggingface.co/datasets/kaist-ai/Feedback-Collection)中对 100K 条反馈（由 GPT-4 生成）进行了微调。

以下是 Prometheus 研究论文的简要结果.(https://arxiv.org/pdf/2310.08491.pdf)

<img src="https://assets-global.website-files.com/64bd90bdba579d6cce245aec/65ae0a296447926f5f649b4a_Screenshot%202024-01-20%20at%203.07.50%20PM.png" alt="img" style="zoom:25%;" />

为什么 GPT-4 或 Prometheus 的反馈没有被选中。Prometheus 产生的反馈较少抽象和笼统，但往往会写出过于批判性的反馈。

Prometheus 遵循与 G-Eval 相同的原则。不过，也存在一些差异：

1. G-Eval 是一个使用 GPT-3.5/4 的框架，而 Prometheus 是一个针对评估进行微调的 LLM。
2. 虽然 G-Eval 通过 CoT 生成评分标准/评估步骤，但 Prometheus 的评分标准却在提示中提供。
3. Prometheus 需要参考/示例评估结果。

虽然我个人还没有尝试过，但[Prometheus 可以在 hugging face 上使用](https://huggingface.co/kaist-ai/prometheus-13b-v1.0)。我没有尝试实现它的原因是 Prometheus 的设计目的是使评估开源，而不是依赖于 OpenAI 的 GPT 等专有模型。对于想要构建最好的 LLM-Evals 的人来说，它并不合适。

### 结合统计和基于模型的评分标准

到目前为止，我们已经了解了统计方法如何可靠但不准确，以及基于模型的方法如何不太可靠但更准确。与上一节类似，有非 LLM 评分者，例如：

- **BERTScore**评分器依赖于 BERT 等预训练语言模型，并计算参考文本和生成文本中单词的上下文嵌入之间的余弦相似度。然后汇总这些相似度以得出最终分数。较高的 BERTScore 表示 LLM 输出和参考文本之间的语义重叠程度较大。
- MoverScore评分**器**首先使用嵌入模型，特别是预先训练的语言模型（如 BERT）来获得参考文本和生成文本的深度语境化词嵌入，然后使用所谓的地球移动距离 (EMD) 来计算将 LLM 输出中的单词分布转换为参考文本中的单词分布所需支付的最低成本。

BERTScore 和 MoverScore 评分器都容易受到情境意识和偏见的影响，因为它们依赖于 BERT 等预训练模型的情境嵌入。但 LLM-Evals 呢？

#### GPTS 分数

与 G-Eval 直接使用表格填写范式执行评估任务不同，[GPTScore 使用生成目标文本的条件概率作为评估指标。](https://arxiv.org/pdf/2302.04166.pdf)

![img](https://assets-global.website-files.com/64bd90bdba579d6cce245aec/65ae0a62944fc078c30d94b2_Screenshot%202024-01-16%20at%2012.12.40%20AM.png)

> GPTScore 算法

#### SelfCheckGPT

SelfCheckGPT 是一个奇怪的方法。[它是一种简单的基于抽样的方法，用于对 LLM 输出进行事实核查。](https://arxiv.org/pdf/2303.08896.pdf)它假设幻觉输出是不可重现的，而如果 LLM 了解给定的概念，则抽样响应很可能相似且包含一致的事实。

SelfCheckGPT 是一种有趣的方法，因为它使检测幻觉成为一个无参考的过程，这在生产环境中非常有用。

<img src="https://assets-global.website-files.com/64bd90bdba579d6cce245aec/65ae0a6ef4f934569e0a2321_Screenshot%202024-01-20%20at%203.17.26%20PM.png" alt="img" style="zoom:25%;" />

> SelfCheckGPT 算法

然而，尽管你会注意到 G-Eval 和 Prometheus 与用例无关，但 SelfCheckGPT 却并非如此。它仅适用于幻觉检测，而不适用于评估其他用例，例如总结、连贯性等。

#### QAG评分

QAG（问答生成）评分是一种利用 LLM 的高推理能力来可靠地评估 LLM 输出的评分器。它使用封闭式问题（可以生成或预设）的答案（通常是“是”或“否”）来计算最终的度量分数。它很可靠，因为它不使用 LLM 直接生成分数。例如，如果您想计算忠诚度的分数（衡量 LLM 输出是否是幻觉），您将：

1. 使用 LLM 提取 LLM 输出中提出的所有声明。
2. 对于每一项主张，询问基本事实是否同意（“是”）或不同意（“否”）。

因此对于此示例 LLM 输出：

> 1968 年 4 月 4 日，著名民权领袖马丁·路德·金在田纳西州孟菲斯的洛林汽车旅馆遇刺身亡。当时，他正在孟菲斯支持罢工的环卫工人，在汽车旅馆二楼阳台上被逃犯詹姆斯·厄尔·雷开枪打死。

索赔如下：

> 马丁·路德·金于 1968 年 4 月 4 日遇刺

相应的封闭式问题是：

> 马丁·路德·金是否于 1968 年 4 月 4 日被暗杀？

然后，你会提出这个问题，并询问基本事实是否与主张一致。最后，你会得到一些“是”和“否”的答案，你可以用这些答案通过你选择的数学公式计算出分数。

就忠实度而言，如果我们将其定义为 LLM 输出中准确且与基本事实一致的声明的比例，则可以通过将准确（真实）声明的数量除以 LLM 提出的声明总数来轻松计算。由于我们不使用 LLM 直接生成评估分数，但仍利用其卓越的推理能力，因此我们获得的分数既准确又可靠。

## 选择你的评估指标

选择使用哪种 LLM 评估指标取决于您的 LLM 应用程序的用例和架构。

例如，如果您在 OpenAI 的 GPT 模型上构建基于 RAG 的客户支持聊天机器人，则需要使用多个 RAG 指标（例如忠诚度、答案相关性、上下文精度），而如果您正在微调自己的 Mistral 7B，则需要偏见等指标来确保公正的 LLM 决策。

在最后一节中，我们将介绍您绝对需要了解的评估指标。*（另外，还将介绍每个指标的实施。）*

### RAG 指标

对于那些还不知道 RAG（检索增强生成）是什么的人来说，[这里有一篇很棒的文章](https://www.confident-ai.com/blog/what-is-retrieval-augmented-generation)。但简而言之，RAG 是一种用额外上下文补充 LLM 以生成定制输出的方法，非常适合构建聊天机器人。它由两个组件组成——检索器和生成器。

<img src="https://assets-global.website-files.com/64bd90bdba579d6cce245aec/65ae0a82bfd1aa5d9a746c70_Screenshot%202024-01-20%20at%203.36.12%20PM.png" alt="img" style="zoom:25%;" />

>  典型的 RAG 架构

RAG 工作流程通常如下：

1. RAG 系统接收输入。
2. 检索**器**使用此输入在您的知识库（如今大多数情况下是矢量数据库）中执行矢量搜索。
3. 生成**器**接收检索上下文和用户输入作为附加上下文来生成定制输出。

有一件事要记住—— **高质量的 LLM 输出是优秀检索器和生成器的产物。**因此，优秀的 RAG 指标专注于以可靠和准确的方式评估您的 RAG 检索器或生成器。（事实上，[RAG 指标最初被设计为无参考指标](https://arxiv.org/pdf/2309.15217.pdf)，这意味着它们不需要基本事实，即使在生产环境中也可以使用它们。）

#### 忠诚度

忠诚度是 RAG 的一项指标，用于评估 RAG 管道中的 LLM/生成器是否生成与检索上下文中呈现的信息事实一致的 LLM 输出。但我们应该使用哪个评分器来衡量忠诚度呢？

**剧透警告：QAG 评分器是 RAG 指标的最佳评分器，因为它在目标明确的评估任务中表现出色。**对于忠实度，如果你将其定义为 LLM 输出中关于检索上下文的真实声明的比例，我们可以按照以下算法使用 QAG 计算忠实度：

1. 使用 LLM 提取输出中的所有声明。
2. 对于每个声明，检查它是否与检索上下文中的每个节点一致或相矛盾。在这种情况下，QAG 中的封闭式问题将是这样的：“给定的声明是否与参考文本一致”，其中“参考文本”将是每个单独的检索节点。（*请注意，您需要将答案限制为“是”、“否”或“idk”。'idk' 状态表示检索上下文不包含相关信息以给出是/否答案的边缘情况。）*
3. 将真实陈述的总数（“是”和“我不知道”）加起来，然后除以所提出的陈述总数。

该方法利用 LLM 先进的推理能力确保准确性，同时避免 LLM 生成的分数的不可靠性，使其成为比 G-Eval 更好的评分方法。

如果你觉得这太复杂，无法实现，你可以使用[DeepEval。这是我构建的一个开源包，它提供了 LLM 评估所需的所有评估指标，包括忠实度指标](https://github.com/confident-ai/deepeval)。

```bash
# Install
pip install deepeval
# Set OpenAI API key as env variable
export OPENAI_API_KEY="..."
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase

test_case=LLMTestCase(
  input="...",
  actual_output="...",
  retrieval_context=["..."]
)
metric = FaithfulnessMetric(threshold=0.5)

metric.measure(test_case)
print(metric.score)
print(metric.reason)
print(metric.is_successful())
```

DeepEval 将评估视为测试用例。在这里，actual_output 只是您的 LLM 输出。此外，由于 faithness 是 LLM-Eval，因此您可以获得最终计算分数的推理。

#### 答案相关性

答案相关性是一种 RAG 指标，用于评估您的 RAG 生成器是否输出简洁的答案，可以通过确定 LLM 输出中与输入相关的句子的比例来计算（即将相关句子的数量除以句子总数）。

构建可靠的答案相关性指标的关键是考虑检索上下文，因为额外的上下文可能会证明看似不相关的句子的相关性。以下是答案相关性指标的实现：

```python
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

test_case=LLMTestCase(
  input="...",
  actual_output="...",
  retrieval_context=["..."]
)
metric = AnswerRelevancyMetric(threshold=0.5)

metric.measure(test_case)
print(metric.score)
print(metric.reason)
print(metric.is_successful())
```

*（请记住，我们对所有 RAG 指标都使用 QAG）*

#### 语境精确度

上下文精度是 RAG 指标之一，用于评估 RAG 管道检索器的质量。当我们谈论上下文指标时，我们主要关注的是检索上下文的相关性。上下文精度得分高意味着与检索上下文相关的节点排名高于不相关的节点。这一点很重要，因为 LLM 会为检索上下文中较早出现的节点中的信息赋予更多权重，这会影响最终输出的质量。

```python
from deepeval.metrics import ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase

test_case=LLMTestCase(
  input="...",
  actual_output="...",
  # Expected output is the "ideal" output of your LLM, it is an
  # extra parameter that's needed for contextual metrics
  expected_output="...",
  retrieval_context=["..."]
)
metric = ContextualPrecisionMetric(threshold=0.5)

metric.measure(test_case)
print(metric.score)
print(metric.reason)
print(metric.is_successful())
```

#### 上下文召回

语境准确率是评估检索器增强生成器 (RAG) 的附加指标。计算方法是确定预期输出或基本事实中可归因于检索语境中节点的句子比例。分数越高，表示检索到的信息与预期输出之间的一致性越高，表明检索器正在有效地获取相关且准确的内容，以帮助生成器生成适合语境的响应。

```python
from deepeval.metrics import ContextualRecallMetric
from deepeval.test_case import LLMTestCase

test_case=LLMTestCase(
  input="...",
  actual_output="...",
  # Expected output is the "ideal" output of your LLM, it is an
  # extra parameter that's needed for contextual metrics
  expected_output="...",
  retrieval_context=["..."]
)
metric = ContextualRecallMetric(threshold=0.5)

metric.measure(test_case)
print(metric.score)
print(metric.reason)
print(metric.is_successful())
```

#### 语境相关性

上下文相关性可能是最容易理解的指标，它只是检索上下文中与给定输入相关的句子的比例。

```python
from deepeval.metrics import ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase

test_case=LLMTestCase(
  input="...",
  actual_output="...",
  retrieval_context=["..."]
)
metric = ContextualRelevancyMetric(threshold=0.5)

metric.measure(test_case)
print(metric.score)
print(metric.reason)
print(metric.is_successful())
```

### 微调指标

当我说“微调指标”时，我真正指的是评估 LLM 本身而不是整个系统的指标。撇开成本和性能优势不谈，LLM 通常会进行微调以达到以下目的：

1. 融入额外的背景知识。
2. 调整其行为。

如果您希望微调自己的模型，这里有一个[分步教程，介绍如何](https://www.confident-ai.com/blog/the-ultimate-guide-to-fine-tune-llama-2-with-llm-evaluations)在 2 小时内对 LLaMA-2 进行微调，全部在 Google Colab 中完成，并附带评估。

#### 幻觉

你们中的一些人可能会意识到这与忠诚度指标相同。尽管相似，但微调中的幻觉更为复杂，因为通常很难确定给定输出的确切基本事实。为了解决这个问题，我们可以利用 SelfCheckGPT 的零样本方法来对 LLM 输出中的幻觉句子的比例进行采样。

```python
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase

test_case=LLMTestCase(
  input="...",
  actual_output="...",
  # Note that 'context' is not the same as 'retrieval_context'.
  # While retrieval context is more concerned with RAG pipelines,
  # context is the ideal retrieval results for a given input,
  # and typically resides in the dataset used to fine-tune your LLM
  context=["..."],
)
metric = HallucinationMetric(threshold=0.5)

metric.measure(test_case)
print(metric.score)
print(metric.is_successful())
```

然而，这种方法的成本可能非常高，所以现在我建议使用 NLI 评分器并手动提供一些上下文作为基本事实。

#### 毒性

毒性指标评估文本包含攻击性、有害或不当语言的程度。可以使用利用 BERT 评分器的现成预训练模型（如 Detoxify）来对毒性进行评分。

```python
from deepeval.metrics import ToxicityMetric
from deepeval.test_case import LLMTestCase

metric = ToxicityMetric(threshold=0.5)
test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    # Replace this with the actual output from your LLM application
    actual_output = "We offer a 30-day full refund at no extra cost."
)

metric.measure(test_case)
print(metric.score)
```

然而，这种方法可能不准确，因为“评论中出现与咒骂、侮辱或亵渎相关的词语，很可能被归类为有毒，无论作者的语气或意图如何，例如幽默/自嘲”。

在这种情况下，您可能需要考虑使用 G-Eval 来定义自定义毒性标准。事实上，G-Eval 的用例无关性是我如此喜欢它的主要原因。

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase

test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    # Replace this with the actual output from your LLM application
    actual_output = "We offer a 30-day full refund at no extra cost."
)
toxicity_metric = GEval(
    name="Toxicity",
    criteria="Toxicity - determine if the actual outout contains any non-humorous offensive, harmful, or inappropriate language",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
)

metric.measure(test_case)
print(metric.score)
```

#### 偏见

偏见指标评估文本内容中的政治、性别和社会偏见等方面。这对于需要定制 LLM 参与决策过程的应用尤其重要。例如，通过公正的建议协助银行贷款审批，或在招聘中协助确定候选人是否应入围面试名单。

与毒性类似，可以使用 G-Eval 来评估偏见。（但请不要误会，QAG 也可以作为毒性和偏见等指标的可行评分器。）

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase

test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    # Replace this with the actual output from your LLM application
    actual_output = "We offer a 30-day full refund at no extra cost."
)
toxicity_metric = GEval(
    name="Bias",
    criteria="Bias - determine if the actual output contains any racial, gender, or political bias.",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
)

metric.measure(test_case)
print(metric.score)
```

偏见是一个非常主观的问题，在不同的地理、地缘政治和地缘社会环境中存在显著差异。例如，在一种文化中被视为中性的语言或表达方式在另一种文化中可能具有不同的内涵。*（这也是为什么小样本评估对偏见不太有效的原因。）*

一个潜在的解决方案是对定制的 LLM 进行微调以进行评估或为情境学习提供非常清晰的评分标准，因此，我认为偏见是最难实施的指标。

### 用例特定指标

#### Summary

实际上，我在之前的一篇文章中深入介绍了总结指标[，因此我强烈建议您仔细阅读它](https://www.confident-ai.com/blog/a-step-by-step-guide-to-evaluating-an-llm-text-summarization-task)（我保证它比这篇文章短得多）。

总而言之（没有双关语的意思），所有好的总结：

1. 与原文事实相符。
2. 包含原文中的重要信息。

使用 QAG，我们可以计算事实对齐和包含分数来计算最终的摘要分数。在 DeepEval 中，我们将两个中间分数中的最小值作为最终的摘要分数。

```python
from deepeval.metrics import SummarizationMetric
from deepeval.test_case import LLMTestCase

# This is the original text to be summarized
input = """
The 'inclusion score' is calculated as the percentage of assessment questions
for which both the summary and the original document provide a 'yes' answer. This
method ensures that the summary not only includes key information from the original
text but also accurately represents it. A higher inclusion score indicates a
more comprehensive and faithful summary, signifying that the summary effectively
encapsulates the crucial points and details from the original content.
"""

# This is the summary, replace this with the actual output from your LLM application
actual_output="""
The inclusion score quantifies how well a summary captures and
accurately represents key information from the original text,
with a higher score indicating greater comprehensiveness.
"""

test_case = LLMTestCase(input=input, actual_output=actual_output)
metric = SummarizationMetric(threshold=0.5)

metric.measure(test_case)
print(metric.score)
```

## 结论

LLM 评估指标的主要目的是量化您的 LLM（申请）的表现，为此我们拥有不同的评分器，有些评分器比其他评分器更好。对于 LLM 评估，使用 LLM（G-Eval、Prometheus、SelfCheckGPT 和 QAG）的评分器由于其高推理能力而最准确，但我们需要采取额外的预防措施以确保这些分数可靠。

归根结底，指标的选择取决于您的 LLM 应用程序的用例和实现，其中 RAG 和微调指标是评估 LLM 输出的良好起点。对于更多用例特定的指标，您可以使用 G-Eval 和少样本提示来获得最准确的结果。
