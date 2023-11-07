# BLIP-2：可扩展预训练的多模态基础模型

##  —— 世界上第一个开源多模态聊天机器人

## 概览

我们提出了 BLIP-2，这是一种可扩展的多模态预训练方法，使任何大型语言模型 (LLM) 都能够摄取和理解图像，解锁 zero-shot 图像到文本生成的功能，并为世界上第一个开源多模态聊天机器人原型。

BLIP-2 在各种视觉语言任务上实现了最先进的性能，以明显更少的可训练参数超越了现有方法。值得注意的是，该模型在零样本[VQAv2上的性能比](https://paperswithcode.com/dataset/visual-question-answering-v2-0)[Flamingo80B](https://arxiv.org/abs/2204.14198)高出 8.7% ，可训练参数减少了 54 倍。该论文还展示了 BLIP-2 在零样本图像到文本生成方面的新兴功能，该模型可以遵循自然语言指令。在本报告中，我们将更深入地研究 BLIP-2，并探讨其在视觉和语言研究领域的潜在应用。

OpenAI 刚刚发布了 GPT-4，这是一种强大的新型多模态 AI 模型，具有引人注目的接受图像输入生成文本的能力。然而，这种能力并不新鲜，这已在我们于 2023 年 1 月 30 日发布的最新 BLIP-2 模型和原型中得到了体现。我们新颖的 BLIP-2 方法使我们能够构建世界上第一个开源多模态聊天机器人原型。下面我们讨论我们的 BLIP-2 模型和 OpenAI 的 GPT-4 之间的差异。

## BLIP-2 与 GPT-4

- 通用与特定：BLIP-2 是一种新颖且通用的视觉语言预训练多模态预训练方法，它可以使任何LLM(Large Language Model)系列能够理解图像并解锁 zero-shot 图像到文本生成功能。GPT-4是一种特定类型的预训练模型，其技术新颖性尚不清楚（未公开）。
- 开源与闭源（仅限 API） ：BLIP-2 的代码和模型在 LAVIS 库 ([https://github.com/salesforce/LAVIS](https://github.com/salesforce/LAVIS?ref=blog.salesforceairesearch.com) )中开源，并且还集成到 HuggingFace Transformers ( [https://huggingface.co/docs/transformers/main/model_doc/blip-2](https://huggingface.co/docs/transformers/main/model_doc/blip-2?ref=blog.salesforceairesearch.com)）。GPT-4 是一个闭源模型，提供付费 API 服务（目前仅提供文本 API）。
- 快与慢：BLIP-2 的运行速度比 GPT-4 快得多。在单个 GPU 上，BLIP-2 对每个图像的推理时间约为 1 秒。根据 GPT-4 的直播，GPT-4 的多模态推理时间处理一张图像需要近 40 秒。
- 无监督学习与（大概）监督学习：BLIP-2 基于从互联网自动爬取的大量噪声图像-文本对进行训练。尽管GPT-4的学习范式尚未发布，但从ChatGPT可以合理推断GPT-4可能使用了大型人工注释数据集。

## 使用 BLIP-2 克服视觉语言预训练障碍

近年来，视觉语言预训练（VLP）发展迅速，更大的模型在各种下游任务上不断改进。然而，这些最先进的模型在预训练期间需要很高的计算成本。为了解决这个问题，作者提出了一种计算效率更高的 VLP 方法，利用现成的预训练视觉和语言模型，这些模型在预训练期间保持冻结状态。预训练的视觉模型提供了图像的高质量视觉表示，预训练的[大语言模型（LLM）](https://wandb.ai/mostafaibrahim17/ml-articles/reports/An-Overview-of-Large-Language-Models-LLMs---VmlldzozODA3MzQz)提供了强大的语言生成和零样本传输能力。

BLIP-2 是一种可扩展的多模态预训练方法，使任何LLM(Large Language Model)都能理解图像，同时保持其参数完全冻结。它比现有的多模态预训练方法具有更高的计算效率。BLIP-2 使用冻结图像编码器和冻结 LLM 有效引导语言图像预训练。例如，要将现有的11B-LLM转变为最先进的多模态基础模型，只需要训练不到2%的参数（仅188M可训练参数）。

对于LLM(Large Language Model)来说，理解视觉内容的关键是弥合视觉与语言模态之间的差距。由于LLM(Large Language Model)在自然语言预训练期间没有看到任何图像，因此弥合模态差距具有挑战性，特别是当LLM(Large Language Model)保持冻结状态时。为此，我们提出了一种使用新的两阶段预训练策略进行预训练的 Querying Transformer （Q-Former）的轻量级模块 ，可以有效增强视觉语言模型。

Q-Former 是一种轻量级转换器，它使用可学习的查询向量从冻结图像编码器中提取视觉特征，经过预训练后，Q-Former 可以有效地充当冻结图像编码器和冻结 LLM 之间的桥梁，为 LLM 提供最有用的视觉特征以输出所需的文本，从而缩小模态差距。

第一阶段是视觉和语言表征学习。在此阶段，我们将 Q-Former 连接到冻结图像编码器，并使用图像文本对进行预训练。Q-Former 学习提取与相应文本最相关的图像特征。我们重新设计了 [BLIP](https://blog.salesforceairesearch.com/blip-bootstrapping-language-image-pretraining/) 的预训练目标，用于视觉和语言表示学习。

第二阶段是视觉到语言的生成学习。在此阶段，我们将 Q-Former 的输出连接到冻结的 LLM来执行视觉到语言的生成学习。我们对 Q-Former 进行预训练，使其输出特征可以由 LLM 解释以生成相应的文本。我们尝试使用基于解码器的LLM（例如OPT）和基于编码器-解码器的LLM（例如FlanT5）。

在推理过程中，我们只需将文本指令附加在 Q-Former 的输出之后作为 LLM 的输入。 我们尝试了各种图像编码器和 LLM，并得出了一个有希望的观察结果：更强的图像编码器和更强的 LLM 都可以使 BLIP-2 获得更好的性能。这一观察结果表明，BLIP-2是一种通用的视觉语言预训练方法，可以有效地收获视觉和自然语言社区的快速进步。BLIP-2 是构建多模态对话式 AI 代理的一项重要突破性技术。

## BLIP-2 的工作原理

<img src="https://api.wandb.ai/files/gladiator/images/projects/37363826/ffc38164.png" alt="图像" style="zoom:50%;" />

>  图 1：BLIP-2 框架概述。[[来源：论文](https://arxiv.org/pdf/2301.12597.pdf)中的图1 ]

BLIP-2的关键思想可以概括为：

- BLIP-2 是一种强大的方法，它有效地结合了冻结的预训练图像模型和语言模型，以在各种视觉语言任务上实现出色的性能，包括视觉问答、图像字幕和图像文本检索。为了弥补模态差距，BLIP-2 采用了 Q-Former 模型，该模型分两个阶段进行预训练：首先用于表示学习，然后用于生成学习。
- 由于使用了[OPT](https://arxiv.org/abs/2205.01068)和[FlanT5](https://arxiv.org/abs/2210.11416)等大型语言模型 (LLM )，BLIP-2 甚至可以按照自然语言指令执行零样本图像到文本生成，从而实现视觉知识推理和视觉对话等新功能。
- 此外，由于使用了冻结单模态模型和轻量级 Q-Former，BLIP-2 因其计算效率而脱颖而出。事实上，它在零样本 VQAv2 上的性能比[Flamingo](https://arxiv.org/abs/2204.14198)等现有最先进的方法高出8.7%，而可训练参数仅减少了 54 倍。

###  Q-Former

<img src="https://api.wandb.ai/files/gladiator/images/projects/37363826/6975edf9.png" alt="图像" style="zoom:50%;" />

> 图 2：Q-Former 和 BLIP-2 第一阶段视觉语言表示学习目标的模型架构。[[来源：论文](https://arxiv.org/pdf/2301.12597.pdf)中的图2（左）]



- Q-Former 作为可训练模块来连接冻结图像编码器和冻结 LLM。
- 无论输入图像分辨率如何，它都会从图像编码器中提取固定数量的输出特征。
- 它由两个共享相同自注意力层的转换器子模块组成，第一个是图像转换器，与冻结图像编码器交互以提取视觉特征，第二个是文本转换器，可以充当文本编码器和文本解码器。
- 作者为图像转换器创建了一组可学习的查询嵌入。查询通过自注意力层相互交互，并通过交叉注意力层（插入每个其他变压器块）与冻结图像特征交互。此外，查询可以通过相同的自注意力层与文本交互。
- 根据预训练任务，作者使用不同的自注意力掩码来控制查询和文本之间的交互。
- Q-Former 使用 bert-base 的预训练权重进行初始化，而交叉注意力层则随机初始化。
- Q-Former 拥有仅 188M 可训练参数的架构。与过去使用的完整微调方法相比，这需要的参数数量要少得多，后者需要更多的参数。
- 作者使用了 32 个查询，每个查询的维度为 768（与 Q-Former 的隐藏维度相同）。
- ﻿用$Z$表示图 2 中的输出查询表示，其尺寸 (32 x 768) 比冻结图像特征（例如[ViT-L/14](https://huggingface.co/openai/clip-vit-large-patch14)的 257x1024 ）小得多。
- 这种瓶颈架构与预训练目标一起强制查询提取与文本最相关的视觉信息。

### 从冻结图像编码器学习引导视觉语言表示

在表示学习阶段，Q-Former 与冻结图像编码器结合使用，使用图像文本对进行预训练。Q-Former 的主要目的是学习提取文本中信息量最大的视觉表示。此外，作者联合优化了三个预训练目标。请注意，在所有三个预训练目标期间，输入格式和模型参数是共享的。使用三种预训练策略的主要目标是在查询和文本之间使用不同的屏蔽策略来控制它们的交互，如图 3 所示。

<img src="https://api.wandb.ai/files/gladiator/images/projects/37363826/e7874660.png" alt="图像" style="zoom:50%;" />

> 图 3：每个目标控制查询文本交互的自注意力屏蔽策略。[[来源：论文](https://arxiv.org/pdf/2301.12597.pdf)中的图2（右）]

#### 图文对比学习（ITC）

这是一个典型的[对比学习](https://lilianweng.github.io/posts/2021-05-31-contrastive/)流程，其中模型学习对齐图像和文本表示，以使它们的互信息最大化。为了实现这一目标，作者将正图像文本对与负图像文本对的相似性进行了对比。来自图像转换器的输出查询表示 *Z*﻿与﻿ 来自文本转换器的﻿ 文本表示 *t。*单模态自注意力掩码用于防止查询和文本相互看到（以防止信息泄漏）。

#### 基于图像的文本生成 (ITG)

ITG 损失用于训练 Q-Former 模型以根据给定的输入图像生成文本。然而，Q-Former 架构缺乏冻结图像编码器和文本标记之间的直接通信路径。要生成文本，必须首先通过查询提取必要的信息，然后通过自注意力层将其转发到文本标记。因此，查询被迫提取包含所有文本信息的视觉特征。为了调节查询文本交互，作者采用了多模态因果自注意力掩码。虽然查询可以相互交互，但它们无法处理文本标记。另一方面，每个文本标记可以处理所有查询及其先前的文本标记。

#### 图文匹配

图像文本匹配（ITM）的目标是建立图像和文本表示之间的详细关联。ITM 任务涉及二元分类，其中模型必须预测图像文本对是正（匹配）还是负（不匹配）。为了实现这一目标，作者利用双向自注意力掩码，允许所有查询和文本相互关注，从而产生捕获多模态信息的输出查询嵌入（ *Z*﻿）。然后将这些嵌入输入二类线性分类器以获得每个嵌入的 logit。输出匹配分数是通过对所有查询的 logits 进行平均而获得的。‘

### 从冻结的法学硕士中引导视觉到语言的生成学习

生成预训练阶段涉及通过将冻结 LLM 连接到 Q-Former（带有冻结图像编码器）来利用冻结 LLM 的生成语言功能。如图 4 所示，全连接 (FC) 层用于线性投影输出查询嵌入  Z* 以匹配 LLM 文本嵌入的维度。这些投影嵌入充当软视觉提示，使 LLM 能够合并 Q-Former 提取的视觉信息。由于 Q-Former 已经经过预先训练，可以提取包含语言信息的视觉表示，因此它充当信息瓶颈，向 LLM 提供相关信息，同时过滤掉不相关的视觉数据。这有助于减轻法学硕士学习视觉语言对齐的负担，从而缓解灾难性遗忘问题。

![图像](https://api.wandb.ai/files/gladiator/images/projects/37363826/83041893.png)

> BLIP-2 的第二阶段视觉到语言生成预训练，从冻结的大型语言模型 (LLM) 中引导。[[来源：论文](https://arxiv.org/pdf/2301.12597.pdf)中的图3 ]

作者尝试了两种类型的 LLM：基于解码器的 LLM 和基于编码器-解码器的 LLM。对于基于解码器的 LLM，他们使用语言建模损失进行预训练，其中冻结的 LLM 的任务是生成以 Q-Former 的视觉表示为条件的文本。另一方面，对于基于编码器-解码器的 LLM，它们使用前缀语言建模损失进行预训练。在这种方法中，文本被分为两部分，其中前缀文本与视觉表示连接起来，作为 LLM 编码器的输入。然后，后缀文本将用作 LLM 解码器的生成目标。

## **BLIP-2的限制**

尽管最近的语言模型（LLMs）可以在给定少量示例的情况下实现上下文学习，但在提供了上下文视觉问答（VQA）示例的情况下，BLIP-2的实验证明并未提高VQA性能。论文的作者将BLIP-2的缺乏上下文学习能力归因于所使用的预训练数据集。这一观察也在Flamingo论文中得到了证实，该论文使用了一个闭源的交叉排列的图像和文本数据集，每个序列中包含多个图像-文本对。作者计划在未来的工作中创建一个类似的数据集。由于多种原因，包括来自LLM的不准确知识、激活错误的推理路径或者对新图像内容没有最新信息，BLIP-2的图像到文本生成可能会产生不尽人意的结果。另外，作者提到了与LLMs相关的风险，比如输出冒犯性语言、传播社会偏见或泄露私人信息，并建议采取补救措施，比如使用指令来引导模型的生成，或者在经过滤除有害内容的数据集上进行训练。

## 总结

BLIP-2是一种创新且资源高效的视觉语言预训练方法，它利用了冻结的预训练图像编码器和LLMs。在预训练过程中，BLIP-2具有极少的可训练参数，在各种视觉语言任务上取得了出色的结果。此外，BLIP-2展示了在零样本指导下生成图像到文本翻译的有希望的能力。BLIP-2是朝着创建多模态对话人工智能代理迈出的关键一步。

## Resourse




查看这些使用 BLIP 和 BLIP-2 执行各种任务的项目和资源！

- BLIP-2 + ChatGPT：[https://github.com/Vision-CAIR/ChatCaptioner](https://github.com/Vision-CAIR/ChatCaptioner?ref=blog.salesforceairesearch.com)
- BLIP + ChatGPT：[https://github.com/microsoft/visual-chatgpt](https://github.com/microsoft/visual-chatgpt?ref=blog.salesforceairesearch.com)
- ImageSEO： https: [//wordlift.io/blog/en/image-seo-using-ai/](https://wordlift.io/blog/en/image-seo-using-ai/?ref=blog.salesforceairesearch.com)
- BLIP + DreamBooth：[https://github.com/KaliYuga-ai/DreamBooth_With_Dataset_Captioning/blob/main/DreamBooth_With_Dataset_Captioning.ipynb](https://github.com/KaliYuga-ai/DreamBooth_With_Dataset_Captioning/blob/main/DreamBooth_With_Dataset_Captioning.ipynb?ref=blog.salesforceairesearch.com)
- Huggingface 上的 BLIP-2：[https ://huggingface.co/blog/blip-2](https://huggingface.co/blog/blip-2?ref=blog.salesforceairesearch.com)
- BLIP 博客（之前发布的模型）：https://blog.salesforceairesearch.com/blip-bootstrapping-language-image-pretraining/
