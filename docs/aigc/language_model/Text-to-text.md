# T5：text-to-text Transformer

## 概览

迁移学习范式由两个主要阶段组成。首先，我们通过大量数据预训练深度神经网络。然后，我们在更具体的下游数据集上微调这个模型（即，进一步训练它）。这些阶段的具体实施可以采取多种不同的形式。例如，在计算机视觉中，我们经常使用监督学习目标在 ImageNet 数据集上预训练模型。然后，使用这些模型对下游数据集（即我们实际尝试解决的任务）执行监督微调。或者在自然语言处理（NLP）中，我们经常对未标记的文本语料库进行[自监督](https://cameronrwolfe.substack.com/i/76273144/self-supervised-learning)预训练。

将大型深度神经网络与大量（预）训练数据集相结合通常会产生令人印象深刻的结果。这一发现对于 NLP 来说尤其如此。鉴于原始文本数据可以在互联网上免费获得，我们可以很容易下载大量文本语料库，根据这些数据预训练大型神经网络，然后在各种下游任务上微调模型（或者只使用零/少样本学习技术）。这种大规模迁移学习方法最初由 BERT [2] 探索，该方法使用[掩码目标](https://cameronrwolfe.substack.com/i/76273144/training-bert) 对未标记数据预训练 [Transformer 编码器](https://cameronrwolfe.substack.com/i/76273144/transformer-encoders)，然后对下游语言任务进行微调。

BERT [2] 的成功怎么强调都不为过（即在几乎所有语言基准测试上都具有最先进的性能）。因此，NLP 社区开始深入研究迁移学习这一主题，并提出了许多新的扩展和改进。由于该领域的快速发展，替代方案之间的比较很困难。text-to-text  Transformer（T5）模型[1]提出了一个用于研究 NLP 中迁移学习方法的统一框架，使我们能够分析不同的设置并得出一组最佳实践。这套最佳实践称之为 T5，这是一种用于语言理解任务的最先进的模型和训练框架。

## 相关历史和背景

T5 将现有的迁移学习技术重新表述为统一的格式，对它们进行比较，并确定获得高性能结果的最佳实践。*但是，这是什么意思？什么是迁移学习以及为什么我们应该关心它？*为了回答这些问题，我们将首先概述一些重要的想法，包括迁移学习和 Transformer 架构的不同变体，这对于理解 [1] 中的分析至关重要。从这里开始，我们将通过解释[BERT 架构](https://cameronrwolfe.substack.com/p/language-understanding-with-bert)来提供一些历史背景，该架构普及了自然语言处理 (NLP) 任务的迁移学习。

### 什么是迁移学习？

[![img](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F8ae9b01a-85a7-47cf-97cb-d13a457bd236_2122x898.png)](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F8ae9b01a-85a7-47cf-97cb-d13a457bd236_2122x898.png)

训练神经网络的不同选项

如果我们想训练神经网络来解决某些任务，我们有两个基本选择。

1. *从头开始训练*：随机初始化您的神经网络并根据您的目标任务对其进行训练（以监督方式）。
2. *迁移学习*：在单独的数据集上预训练网络，然后针对目标任务对其进行微调（即更多训练）。

通常，预训练是在比下游目标数据集大得多的数据集上执行的。一般来说，预训练可以极大地提高数据效率。该模型在微调过程中学习得更快，甚至可能表现得更好。迁移学习过程可以采取多种不同的形式。例如，在计算机视觉中，我们可以通过 ImageNet 预训练模型（使用[监督学习](https://www.geeksforgeeks.org/supervised-unsupervised-learning/)），然后在较小的数据集（如 CIFAR-10/100）上进行微调。对于自然语言处理（NLP）任务，情况有些不同。通常，我们使用带有未标记文本的[自监督](https://cameronrwolfe.substack.com/i/76273144/self-supervised-learning)预训练目标（例如，[掩码语言建模](https://cameronrwolfe.substack.com/i/76273144/training-bert)或[因果语言建模](https://cameronrwolfe.substack.com/i/85568430/language-modeling).

### 不同的Transformer架构

![img](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F52aae8ba-0ae0-4dec-bb5e-2dc0ed90a761_810x1242.png)

（来自[6]）

该Transformer最初在[1]中提出，使用编码器-解码器架构，如上所示。

然而，编码器-解码器Transformer架构并不是我们唯一的选择！BERT 使用[仅编码器架构](https://cameronrwolfe.substack.com/i/76273144/transformer-encoders)，而大多数[现代大型语言模型](https://cameronrwolfe.substack.com/p/modern-llms-mt-nlg-chinchilla-gopher)(LLM) 都基于[仅解码器Transformer](https://cameronrwolfe.substack.com/i/85568430/decoder-only-transformers)。让我们花一点时间来了解这些架构变体之间的差异。

![img](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fdb41d46e-ebea-46c8-9005-472d2983a35b_1164x848.png)

Transformer编码器中的双向自注意力

**自注意力入门。**自注意力操作将Token向量序列作为输入，并生成与输出长度相同的新的变换Token向量序列；这个新序列的每个条目都是输入序列中向量的加权平均值。具体来说，我们如下计算输出序列中的每个标记向量，其中`y_i`和`x_j`分别是输出和输入序列的元素。

上面的权重是作为和`w_{i, j}`的函数生成的注意力分数。简而言之，这个分数捕获了当前标记在计算其新表示时应“注意”序列中另一个标记的程度。`x_i``x_j`

![img](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F82a9585b-9f04-429c-a59b-8b1afc8e9a0f_1600x1462.png)

（来自[6]）

**单栈还是双栈？**最初的 Transformer 架构使用两个 Transformer 层“堆栈”；第一个堆栈（编码器模块）由几个包含双向自注意力和[前馈神经网络的块组成](https://cameronrwolfe.substack.com/i/94634004/feed-forward-neural-networks)。第二个堆栈（解码器模块）非常相似，但它使用[掩码自注意力](https://cameronrwolfe.substack.com/i/76273144/self-attention)，并添加了“交叉注意力”机制，该机制在执行自注意力时考虑相应编码器层内的激活。Transformer最初用于[序列到序列的](https://en.wikipedia.org/wiki/Seq2seq)任务（例如，语言翻译）。对于其他任务，单堆栈Transformer模型已变得流行：

- 语言模型使用仅解码器架构
- BERT 风格的模型使用仅编码器架构

![img](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F0253f047-dc04-4f30-a7b3-8f2839e8e8db_1788x1178.png)

（来自[1]）

**Attention Mask。**Transformer 架构的变体有一个主要区别：*其注意力层中使用的掩蔽类型*。在这里，当我们说“Mask”时，我们指的是在自注意力计算过程中某些标记被Mask（或忽略）。简而言之，某些标记可能仅查看完整输入序列中其他标记的选定部分。上图描述了自注意力的不同掩蔽选项。

仅编码器模型利用双向（或完全可见）自注意力，在自注意力过程中考虑整个序列中的所有标记。自注意力中的每个标记表示都被计算为序列中所有其他标记的加权平均值。相比之下，仅解码器模型使用因果自注意力，其中每个标记仅考虑序列中位于其之前的标记。

![img](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F5302e0b7-be1f-4010-b5c0-52d05b086e76_560x990.png)

（来自[1]）

我们还可以通过定义“前缀”来采用混合方法。更具体地说，我们可以对序列开头（即前缀）的一组标记执行双向自注意力，然后对序列中的其余标记执行因果自注意力；完全可见（或双向）的自注意力对于关注前缀或执行分类任务非常有用。然而，某些应用（例如，语言建模）需要在训练期间进行因果自注意力，以防止Transformer“展望未来”（即，在生成输出时仅复制正确的标记）。

**T5有什么用？**尽管 [1] 中的分析考虑了许多 Transformer 架构，但 T5 使用的主要模型是标准编码器-解码器架构。除了一些小的修改之外，该模型与最初提出的Transformer非常相似[6]。[1] 中没有探讨仅编码器架构，因为它们是为标记或序列级别分类而设计的，而不是为翻译或摘要等生成任务而设计的。T5 旨在找到一种统一的方法（基于迁移学习）来解决许多语言理解任务。

### BERT：NLP 的迁移学习

早期，NLP 中的迁移学习通常使用以[因果语言建模目标预先训练的循环神经网络](https://cameronrwolfe.substack.com/i/85568430/language-modeling)。然而，随着[BERT ](https://cameronrwolfe.substack.com/p/language-understanding-with-bert)的提出，一切都改变了，BERT 是一种基于 Transformer 的模型 [6]，使用[自监督目标](https://cameronrwolfe.substack.com/i/76273144/self-supervised-learning)进行预训练。BERT 可以对大量未标记的文本进行预训练，然后进行微调，以极高的准确度对句子（甚至句子中的单个标记）进行分类。在提出提案时，BERT 为几乎所有被考虑的 NLP 任务设定了新的最先进技术，巩固了迁移学习作为 NLP 中首选方法的地位。

![img](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Ffe697a9b-6a11-489b-9701-15d42d3feb15_2006x1206.png)

> 使用 BERT 进行自监督 MLM 预训练

为了使这一点更具体一些，BERT在预训练期间依赖于一个“去噪”目标，称为[掩码语言建模（MLM)](https://cameronrwolfe.substack.com/i/76273144/self-supervised-learning)虽然这听起来有点复杂，但想法很简单，我们只需：

1. 通过将输入序列中的一些标记替换为特殊`[MASK]`标记来Mask它们
2. 使用 BERT 处理损坏/修改的序列
3. 训练 BERT 准确预测Mask标记

确切的实现有点复杂。我们随机选择 15% 的Token，然后用该`[MASK]`Token（90% 的概率）或随机Token（10% 的概率）替换它们。通过在足够大的预训练语料库上使用这个目标，BERT 可以学习大量通用语言知识，这使其成为高效的迁移学习模型。

**T5 与 BERT 有什么关系？**BERT 的提出表明迁移学习是解决 NLP 问题的一种有用方法。许多人很快开始使用 BERT，尝试新技术，并提出改进建议。因此，该领域充斥着使用类似 BERT 模型执行迁移学习的不同选择。T5 [1] 继续这一研究方向，但尝试使用统一的框架分析所有这些不同的建议，让我们更清楚地了解 NLP 中迁移学习的最佳实践。最终的 T5 模型使用所有这些最佳实践进行训练，以达到最先进的性能。

**T5 与 LLM 有何关系？**目前，我们看到生成式人工智能领域发生了一场巨大的革命，其中[LLMs](https://cameronrwolfe.substack.com/p/specialized-llms-chatgpt-lamda-galactica)（基于仅解码器的Transformer架构）被用来通过语言模型预训练和[零/少样本学习来解决语言任务](https://cameronrwolfe.substack.com/i/88082618/language-models-are-few-shot-learners)。LLM 很棒，但 T5 存在于一个相对独特的工具和研究领域。也就是说，T5 主要关注在使用单独的解码器生成输出之前使用编码器显式处理输入的模型。另外，T5 采用迁移学习方法（即预训练，然后对每个目标任务进行微调），而不是零/少样本学习。

### 其他有用的链接

- Transformer 架构 [[链接](https://cameronrwolfe.substack.com/i/74325854/the-transformer-architecture)]
- 自注意力[[链接](https://cameronrwolfe.substack.com/i/74325854/self-attention)]
- BERT模型[[链接](https://cameronrwolfe.substack.com/p/language-understanding-with-bert)]
- 语言模型的基础知识[[链接](https://cameronrwolfe.substack.com/p/language-models-gpt-and-gpt-2)]

## T5：统一的text-to-text Transformer

T5 的贡献并不是一种新颖的架构或训练方法。相反，T5 中进行的研究完全基于现有技术。T5 考虑了 NLP 中迁移学习流程的各个方面，例如不同的（未标记的）数据集、预训练目标、基准和微调方法。然而，所有这些方面都是通过统一的text-to-text格式进行研究的。T5 的目标是

*i)*分析迁移学习设置

*ii)*确定最有效的方法

### text-to-text 框架

T5将所有文本处理问题转换为“text-to-text”的格式（即，将文本作为输入并产生文本作为输出）。这种通用结构也被LLMs零/少样本学习所利用，它使我们能够使用共享方法建模和解决各种不同的任务。*我们可以将相同的模型、目标、训练过程和解码过程应用于我们考虑的每项任务*！我们只是采用[提示](https://cameronrwolfe.substack.com/i/91134599/a-primer-on-language-modeling)方法，要求我们的语言模型以文本格式生成答案。

[![img](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F821c34a1-b63b-4ea9-9ced-0c73549288d2_2238x1266.png)](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F821c34a1-b63b-4ea9-9ced-0c73549288d2_2238x1266.png)

（来自[1]）

更具体一点，T5 解决的所有任务都可以转换为text-to-text的格式，如下所示：

1. 将特定于任务的前缀添加到原始输入序列
2. 将此序列馈送到Transformer
3. 将模型的目标表述为文本序列

使用这种格式，我们可以轻松地执行摘要或翻译等任务（即目标自然是一个序列）。另外，我们可以通过训练模型来生成与正确类别相关的文本来执行分类。对于回归等问题，这个过程会变得有点复杂（即，我们必须将实值输出四舍五入到最接近的小数，并将其视为分类问题），但它往往适用于大多数语言任务。示例如上图所示。

> *“如果我们的模型在文本分类任务上输出的文本与任何可能的标签都不对应，就会出现问题……在这种情况下，我们总是将模型的输出视为错误，尽管我们从未在任何经过训练的模型中观察到这种行为”。*- 来自 [1]

T5 针对它解决的每项任务进行了微调。这与使用小样本学习的LLMs和使用[多任务学习](https://www.ruder.io/multi-task/)同时解决多个任务的 NLP 十项全能 [3] 形成鲜明对比。

### T5是如何研究的？

[1] 中执行的所有分析都使用上述统一的text-to-text框架，因为它允许将各种不同的语言理解任务转换为共享格式。此外，T5 的分析使用相同的底层 Transformer 架构和预训练数据集。

![img](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F3f25f11e-1daf-4711-940a-6b09a1f62ae7_2298x1474.png)

> T5 对编码器-解码器Transformer架构所做的修改（来自[6]）

如前所述，Transformer架构最初在[6]中提出，包含编码器和解码器模块。最近关于语言建模的工作探索了仅编码器或解码器的架构变体；例如，[BERT](https://cameronrwolfe.substack.com/i/76273144/transformer-encoders)仅使用编码器 [2]，而大多数[大型语言模型](https://cameronrwolfe.substack.com/i/85568430/decoder-only-transformers)仅使用解码器。T5 使用与原始 Transformer 非常相似的编码器-解码器架构。差异是：

1. [LayerNorm](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) 在每个注意力和前馈变换之前使用（即，在残差路径之外）
2. LayerNorm 不使用附加偏差（即，参见[此处](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)；我们仅使用比例并消除附加偏差）
3. 使用简单的位置嵌入方案，将标量添加到用于计算注意力权重的相应[logit](https://stackoverflow.com/questions/41455101/what-is-the-meaning-of-the-word-logits-in-tensorflow)
4. Dropout应用于整个网络（例如，注意力权重、前馈网络、跳过连接等）

这些修改如上图所示。使用此模型（以及其他一些模型），T5 可以测试许多不同的迁移学习设置，以得出一组最佳实践。

**预训练数据集。**T5 在 Colossal Clean Crawled Corpus (C4) 上进行了预训练，这是一个在 [1] 中创建的 750Gb 的“相对干净”英语文本语料库。虽然在之前的工作中已经提出了各种预训练数据集，但[1]中的作者选择构建自己的数据集，因为之前的数据集不公开，使用有限的过滤规则集，范围有限（例如，仅来自[知识共享](https://creativecommons.org/)），或仅关注机器翻译的并行数据（即同一句子在多种不同语言中的版本）。

[![img](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F697e8965-6413-4ab1-982b-9f7ca97679fd_1682x572.png)](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F697e8965-6413-4ab1-982b-9f7ca97679fd_1682x572.png)

（来自[4]）

值得注意的是，C4 后来被用作 MassiveText 数据集的子集，用于预训练[Gopher](https://cameronrwolfe.substack.com/i/91134599/scaling-language-models-methods-analysis-and-insights-from-training-gopher)和[Chinchilla ](https://cameronrwolfe.substack.com/i/91134599/training-compute-optimal-llms)[4, 5]。请参阅上表了解该数据集的大小指标，它可以更好地理解 C4 相对于用于训练现代 LLM 的预训练数据集的大小。通过LLMs，我们发现在足够大的数据集上预训练仅解码器模型对于其成功至关重要。不同架构的Transformer也是如此，比如T5。在大型、未标记的数据集上进行广泛的预训练有利于获得更好的下游性能。

**实验装置。**T5 通过 C4 进行预训练，然后进行微调以解决各种下游任务。然而，该框架中使用的确切设置是可变的。也就是说，我们可以更改：

- Transformer架构
- 预训练设置（即任务或数据量）
- 微调设置
- 模型的尺寸/比例

通过一次更改每个设置并评估结果，我们可以开发一套 NLP 迁移学习的最佳实践，从而将 BERT 后的许多建议提炼成一个单一、有效的管道，以创建有效的语言理解楷模。

### 语言建模与去噪

![img](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd1c359f0-9141-4ef7-94bb-d27042fd9c7e_2262x1360.png)

NLP 中的初始迁移学习方法利用[因果语言建模](https://cameronrwolfe.substack.com/i/85568430/language-modeling)目标 [6] 进行预训练。然而，去噪（也称为[掩码语言建模](https://cameronrwolfe.substack.com/i/76273144/training-bert)，或 MLM）目标随后被证明表现更好 [5]。给定一组文本标记作为某个模型的输入传递，MLM 的运行方式如下：

1. 随机（均匀）选择 15% 的Token
2. 用一个 [MASK] 替换 90% 的选定Token
3. 用随机Token替换 10% 的选定Token
4. 训练模型来预测/分类每个`[MASK]`标记

统一选择的Token的百分比称为“腐败率”。在 T5 中，我们将看到该去噪目标的一些不同变体，但基本思想保持不变。

> *“我们所有的目标都是从未标记的文本数据集中获取与标记化文本范围相对应的标记 ID 序列。处理Token序列以产生（损坏的）输入序列和相应的目标。然后，像往常一样训练模型，以最大可能性预测目标序列。”* - 来自 [1]

### 基准和评估

T5 试图推导出一套 NLP 中迁移学习的最佳实践。然而，为了确定哪种技术最有效，我们在各种不同的任务和自然语言基准上对 T5 进行了评估。所有这些任务都是使用 T5 的[文本到文本格式来解决的](https://cameronrwolfe.substack.com/i/108182616/text-to-text-framework)。有关这些任务的完整描述，请参阅 [1] 中的第 2.3 节。下面提供了一个简短的总结。

- [GLUE](https://gluebenchmark.com/)和[SuperGLUE ](https://super.gluebenchmark.com/)[7, 8]：两个基准测试都包含许多不同的任务，例如[句子可接受性判断](https://nyu-mll.github.io/CoLA/)、[情感分析](https://huggingface.co/datasets/sst2)、[释义](https://paperswithcode.com/dataset/mrpc)、[句子相似性](https://paperswithcode.com/sota/semantic-textual-similarity-on-sts-benchmark)、[自然语言推理（NLI）](https://cims.nyu.edu/~sbowman/multinli/)、[共指消解](https://paperswithcode.com/sota/natural-language-inference-on-wnli)、句子[完成](https://paperswithcode.com/dataset/copa)、[词义消歧](https://pilehvar.github.io/wic/)和[问答](https://sheng-z.github.io/ReCoRD-explorer/)。
  - SuperGLUE 是一个改进的、难度更高的基准测试，其结构与 GLUE 类似。
- [CNN + Daily Mail Abstractive Summarization ](https://huggingface.co/datasets/cnn_dailymail)[9]：将新闻文章与简短的摘要文本序列配对，捕捉文章的主要亮点。
- [SQuAD ](https://rajpurkar.github.io/SQuAD-explorer/)[10]：维基百科文章的问答数据集，其中每个问题的答案是相关文章中的一段文本。
- 多个翻译数据集（例如英语到德语、法语和罗马尼亚语）。

值得注意的是，GLUE 和 SuperGLUE 基准测试中的所有任务都通过 T5 连接在一起，并且同时对所有任务进行微调。

**训练T5。**[T5 模型针对C4 语料库](https://cameronrwolfe.substack.com/i/108182616/how-is-t-studied)中的总共 34B 个标记进行了预训练。为了进行比较，BERT 接受了 137B 个Token的训练，而 RoBERTa 接受了 2.2T 个Token的训练 [5, 12]。受 BERT 的 MLM 目标的启发，T5 使用稍微修改的去噪目标进行预训练：

1. 随机选择输入序列中 15% 的标记
2. 用单个
   “哨兵”Token替换所选Token的所有连续范围
3. 为每个哨兵Token提供一个当前输入序列唯一的 ID
4. 使用所有选定的Token构造一个目标，由哨兵Token分隔

尽管这个任务看起来有点复杂，但我们可以在下面看到它如何在短输入序列上工作的说明。

[![img](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fac18ac33-8c04-44ec-a012-d220071b41b3_2170x1118.png)](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fac18ac33-8c04-44ec-a012-d220071b41b3_2170x1118.png)

（来自[1]）

通过用单个哨兵标记替换整个掩码标记，我们减少了预训练的计算成本，因为我们倾向于在较短的输入和目标序列上进行操作。

**微调。**执行预训练后，T5 在评估之前对每个下游任务分别进行微调。由于 T5 使用的文本到文本格式，预训练和微调都使用相同的[最大似然目标](https://cameronrwolfe.substack.com/i/85568430/language-modeling)！换句话说，我们只是将正确答案表述为文本序列（在预训练和微调期间）并训练模型以输出正确的文本序列。

**基线表现如何？**如下表所示，基线 T5 模型的性能与 BERT 等之前的模型类似，尽管这些模型不具有直接可比性（即基线 T5 模型使用 BERTBase 使用的计算量的 25%）。另外，我们发现预训练对大多数任务都有巨大的好处。此规则的例外是翻译任务，无论有没有预训练，其性能都相似。

![img](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F40023bf9-2576-40e8-9260-bba060ef6526_1482x480.png)

## T5：把它们放在一起！

现在我们已经回顾了 [1] 中的整个实验分析，我们对 NLP 中迁移学习的不同选项以及最有效的方法有了更好的了解！下面，我们将回顾此分析的主要内容，其中包括 T5 使用的官方迁移学习框架。与各种替代方案相比，这种方法表现得相当好。

**基线设置。**首先，我们回顾一下T5的基线架构。它是一个编码器-解码器转换器，使用统一的[文本到文本格式进行训练](https://cameronrwolfe.substack.com/i/108182616/text-to-text-framework)。在使用去噪目标进行预训练后，模型在评估之前针对每个下游任务分别进行微调。值得注意的是，最终的 T5 模型针对 GLUE 和 SuperGLUE 基准中的每个任务分别进行了微调，因为对所有任务进行一起训练会产生稍低的性能（假设我们采取必要的步骤来避免过度拟合）。

**预训练。**最终的 T5 方法不是统一选择Token，而是执行平均长度为 3 的跨度损坏（即一次选择整个跨度的Token进行损坏）。尽管如此，仍有 15% 的Token被选择用于腐败。该目标的性能比基线稍好，并且产生更短的目标序列长度。此外，T5 将无监督预训练更新与多任务监督更新相结合。无监督和监督更新数量之间的比率取决于所使用的模型的大小（即，较大的模型需要更多的无监督更新以避免过度拟合）。

**训练量。**额外的预训练对T5的表现有帮助。具体来说，增加批量大小和训练迭代次数都有利于 T5 的性能。因此，最终的 T5 模型总共预训练了超过 1T 个Token。这比预训练期间基线的 34B Token大得多，但仍远低于 RoBERTa [12]，后者是在超过 2.2T Token上进行预训练的。预训练是在通用的、经过过滤的 C4 数据集上执行的，因为特定于任务的预训练不会在不同的任务中产生一致的好处。

**模型规模。**使用较大的模型很有帮助，但有时较小的模型可能更有意义（例如，当可用于推理的计算有限时）。为此，T5 发布了五种不同尺寸的型号，参数从 220M 到 11B 不等。由此可见，T5实际上是一套不同型号的套件！我们可以通过下面的链接访问这些模型中的任何一个。

## 参考文献

[1] Raffel, Colin, et al. "Exploring the limits of transfer learning with a unified text-to-text transformer." *The Journal of Machine Learning Research* 21.1 (2020): 5485-5551.

[2] Devlin, Jacob, et al. "Bert: Pre-training of deep bidirectional transformers for language understanding." *arXiv preprint arXiv:1810.04805* (2018).

[3] McCann, Bryan, et al. "The natural language decathlon: Multitask learning as question answering." *arXiv preprint arXiv:1806.08730* (2018).

[4] Rae, Jack W., et al. "Scaling language models: Methods, analysis & insights from training gopher." *arXiv preprint arXiv:2112.11446* (2021).

[5] Hoffmann, Jordan, et al. "Training compute-optimal large language models." *arXiv preprint arXiv:2203.15556* (2022).

[6] Vaswani, Ashish, et al. "Attention is all you need." *Advances in neural information processing systems* 30 (2017).