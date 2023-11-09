# Flamingo：用于少样本学习的视觉语言模型

在本文中，我们将探讨 Flamingo — 由 DeepMind 开发的用于多模态机器学习研究的开放式单视觉语言模型 (VLM)。

Flamingo 是一种新的视觉语言模型 (VLM)，能够执行字幕、视觉对话、[分类](https://wandb.ai/fully-connected/blog/classification)和视觉问答等多模态任务。正如你所看到的，它运行得相当好：

本文将引导您学习这项新研究、了解其架构和训练数据，最后对其进行测试运行以了解其工作原理。

## 什么是Flamingo？

[Flamingo](https://www.deepmind.com/blog/tackling-multiple-tasks-with-a-single-visual-language-model)是一系列视觉语言模型 (VLM)，在广泛的开放式视觉和语言任务的小样本学习中取得了新的最先进水平。

Flamingo 是一种视觉条件自回归文本生成模型，能够摄取与图像和/或视频交错的一系列文本标记，并生成文本作为输出。

Flamingo 模型通过在其间添加新颖的架构组件，将大型语言模型与强大的视觉嵌入融合（组合），每个模型都经过单独预训练和冻结。

### Examples of inputs and outputs obtained from 80B parameter Flamingo model

在深入研究之前，让我们看一些 Flamingo 模型可以执行的任务类型的示例。

<img src="https://api.wandb.ai/files/gladiator/Flamingo%20VLM/22gt8u0y/media/images/image_0_531cb34a9977c8e7725c.png?height=1406" alt="卡片" style="zoom: 33%;" />

> 图1 | 从 80B 参数模型 Flamingo 获得的输入和输出示例。与大型语言模型一样，*Flamingo*可以通过简单地用几个例子来提示（上图），从而快速适应各种图像和视频理解任务。*Flamingo*开箱即用，还能够进行丰富的视觉对话（底部）。

<img src="https://api.wandb.ai/files/gladiator/Flamingo%20VLM/1q9et1w1/media/images/image_0_8bed53de1de9ee54b6a2.png?height=702" alt="卡片" style="zoom:33%;" />

### Flamingo 模型结果概述

<img src="https://api.wandb.ai/files/gladiator/Flamingo%20VLM/29x6pgkj/media/images/image_0_4cddf8fa9a0d88084ee8.png?height=703" alt="卡片" style="zoom: 33%;" />

> 图2 | Flamingo 模型的结果概述。*左*：最大的模型在作者考虑的 16 项任务中的 6 项上优于最先进的微调模型，尽管没有使用微调。对于所有已发布的少样本结果的 16 项任务，*Flamingo*都大幅优于它们，并创下了新的少样本最先进水平。*中*：*Flamingo的*表现随着样数的增加而提高。*右图*：Flamingo 模型的性能随着模型规模的增大而提高。

## 多模态生成建模的挑战

在这里，让我们看看一些挑战以及 Flamingo 如何解决这些挑战：

### 统一强大的单模态模型

挑战：

- 训练大型语言模型的计算成本极其高昂。
- 作者从预训练的语言模型开始，节省了计算资源。
- 然而，纯文本模型没有内置方法来合并来自其他模态的数据。
- 作者希望在保留原始语言模型知识的同时实现这一点。

建议的方法：

将交叉注意力层与仅在训练期间保持冻结的常规语言自注意力层交错。作者还引入了一种特定的门控机制，以最小化这些新添加的层在初始化时的影响，从而大大提高了稳定性和最终性能。

### 支持图像和视频

挑战：

- 图像和视频（即使是中等分辨率）也是高维的。
- 将它们展平为一维序列（如单峰文本生成中所使用的）的成本很高，因为计算量与序列长度呈二次方关系。
- 作者还希望以统一的方式处理图像和视频，但这并不简单。

建议的方法：使用 [Perceiver](https://www.deepmind.com/publications/perceiver-general-perception-with-iterative-attention)-based architecture ，在给定大量不同数量的视觉输入特征（最多数千个）的情况下，可以为每个图像/视频生成少量固定数量的视觉标记（大约一百个）。

### 异构训练数据

挑战：

- 大型模型需要庞大的数据集。
- [CLIP](https://openai.com/blog/clip/)和[ALIGN](https://ai.googleblog.com/2021/05/align-scaling-up-visual-and-vision.html)中使用的配对图像/标题数据集可能不够通用，无法达到 GPT-3 风格的少样本学习。
- 存在基于互联网的大型纯文本数据集，但不适合多模态数据。
- 一种方法是抓取带有图像和文本的网页。尽管数据具有普遍性，但图像和文本往往相关性较弱。

建议的方法：将交错数据集（作者创建的）与标准配对图像/文本和视频/文本数据集相结合，其中视觉和语言通常更紧密相关。

## Flamingo 的核心思想

DeepMind 的 Flamingo 可以仅通过几个输入/输出示例执行各种多模态任务（例如字幕、视觉对话、分类或视觉问答）。这是通过以下关键思想实现的：

- 一种新颖的架构，用于接受任意交错的视觉和文本数据作为输入并以开放式方式生成输出文本。
- 架构创新和训练策略有效地利用大型预训练的仅视觉和仅语言模型，节省大量计算并保留这些初始模型的优势，同时有效地融合模态。具体来说，作者使用了[Chinchilla](https://www.deepmind.com/publications/an-empirical-analysis-of-compute-optimal-large-language-model-training)，一个 70B 最先进的 LM（被冻结在 Flamingo 中），并训练了 Flamingo，一个 80B 参数的 VLM。
- 适应不同尺寸视觉输入的有效方法，使 Flamingo 适用于图像和视频。

## Flamingo模型

### 模型结构

<img src="https://api.wandb.ai/files/gladiator/Flamingo%20VLM/c27cha09/media/images/example_0_a5403ab86a05afa632e1.png?height=850" alt="卡片" style="zoom: 33%;" />



> 图3 | Flamingo 模型概述。Flamingo 模型是视觉语言模型 (VLM) 系列，它可以采用与文本交叉的输入视觉数据并生成自由格式的文本作为输出。其性能的关键是新颖的架构组件和预训练策略。

Flamingo 接受与图像/视频交错的文本并输出自由格式的文本。它可以处理开放式任务（例如视觉问答或字幕）和封闭式任务（例如分类）。

- 作者的第一个目标是利用预先训练的语言模型，而不需要花费计算从头开始训练它们。具体来说，他们使用了DeepMind 最近推出的名为[Chinchillah](https://www.deepmind.com/blog/an-empirical-analysis-of-compute-optimal-large-language-model-training)的模型。这使得 Flamingo 模型具有强大的生成语言能力，并且可以访问存储在 LM 权重中的大量知识。
- 在视觉方面，作者使用类似于[CLIP 的](https://openai.com/blog/clip/)对比文本图像方法来预训练视觉编码器。该模型的作用是从给定的图像/视频中提取丰富的语义空间特征。
- 第二个目标是和谐地连接这两种模型。为此，作者冻结了这些模型的权重，并通过两个可学习的架构将它们连接起来。
- 感知器重采样器（Perceiver Resampler）从视觉编码器接收时空特征（从可变数量的图像或视频获得）并输出一组固定大小的视觉标记。
- 然后，使用视觉标记来使用新初始化的交叉注意力层来调节冻结的 LM，这些交叉注意力层交错（或插入）在预训练的 LM 层之间。这些层为 LM 提供了一种将视觉信息整合到下一个标记预测任务中的方法。

Flamingo 模型的一个重要方面是它们可以对文本 $y$ 与图像/视频 $x$ 序列交织的似然进行建模。视觉条件文本似然建模如下：
$$
\mathrm{p}(\mathrm{y} \mid \mathrm{x})=\prod_{\ell=1}^{\mathrm{L}} \mathrm{p}\left(\mathrm{y}_{\ell} \mid \mathrm{y}_{<\ell}, \mathrm{x}_{\leq \ell}\right)-(1)
$$
其中$\mathrm{y}_{\ell}$ 是组成输入文本的第  $\ell$ 个语言标记，$\mathrm{y}_{<\ell}$ 是前面的标记集合，$\mathbf{x}_{\leq \ell}$是交错序列中标记 $\mathrm{y}_{\mathrm{ell}}$之前的图像/视频集，$\mathrm{p}$ 由 Flamingo 模型参数化。

该模型是通过在不同的数据集的混合数据上最大化方程（1）的可能性来训练的。

### 视觉编码器：从像素到特征

- 作者使用了 F6 Normalizer-Free ResNet ( [NFNet](https://arxiv.org/pdf/2102.06171.pdf) )，因为它在给定硬件的情况下在性能和效率之间提供了良好的权衡。
- 使用 CLIP 所采用的对比损失将视觉编码器预训练为Dual编码器。
- [BERT](https://wandb.ai/mukilan/BERT_Sentiment_Analysis/reports/An-Introduction-to-BERT-And-How-To-Use-It--VmlldzoyNTIyOTA1)用于文本编码器，在预训练后被丢弃。
- 对比相似度计算为图像编码器输出的平均池化和 BERT 模型的平均池化输出的点积。
- 与 CLIP 相比，为了简单起见，全局平均池用于生成视觉嵌入（而不是全局注意力池）。
- 使用288 x 288像素图像分辨率，联合嵌入空间大小为1376。
- 最终输出是特征 $X_{f}$，进一步展平为 1D，如图 4 所示。
- 对于视频输入，帧以 1 FPS 采样并独立编码以获得一系列 $T$ 个特征图 $X_{f}$，然后将其连接起来。

### 视觉编码器详细信息

#### 优化

- 使用 Adam 优化器在 512 TPUv4 芯片上进行训练。
- 使用相当大的批量大小16,384为模型提供了大量的负例样本。
- 颜色增强和随机水平翻转用作数据增强。
- 在零样本图像分类上监控训练过程（如 CLIP，这是通过提示模板完成的）`A photo of a {class}`

表 1 | 显示针对不同数据集组合进行的消融研究的表格。

![卡片](https://api.wandb.ai/files/gladiator/Flamingo%20VLM/yqai7pll/media/images/example_0_72c8191a7fff7a4d054b.png?height=392)

#### 预训练数据

- 在两个图像文本对数据集的组合上进行训练。
- ALIGN（由 18 亿张图像与替代文本组成）。ALIGN很大，但是噪音很大。
- LTIP（3.12 亿张图像），具有更清晰、更长的描述。
- 数据与经消融研究证实最有效的积累策略相结合。

这里定义了不同的数据组合策略：

累积：计算每个数据集批次的梯度，并在更新参数之前通过加权和将其组合。

数据合并：将每个数据集中的示例合并到每个批次中。

循环：每个数据集中交替批次，每批次后更新参数。

### 感知器重采样器(Perceiver Resampler )

#### *感知器重采样器的伪代码*

```python
def perceiver_resampler(
 x_f, # The [T, S, d] visual features (T=time, S=space)
 time_embeddings, # The [T, 1, d] time pos embeddings.
 x, # R learned latents of shape [R, d]
 num_layers, # Number of layers
):
 """The Perceiver Resampler model."""
 # Add the time position embeddings and flatten.
 x_f = x_f + time_embeddings
 x_f = flatten(x_f) # [T, S, d] -> [T * S, d]
 # Apply the Perceiver Resampler layers.
 for i in range(num_layers):
    # Attention.
    x = x + attention_i(q=x, kv=concat([x_f, x]))
    # Feed forward.
    x = x + ffw_i(x)
return x复制错误！复制了！
```

#### 感知器重采样器结构

<img src="https://api.wandb.ai/files/gladiator/Flamingo%20VLM/2k6arfib/media/images/example_0_a58121f2a730264a45b5.png?height=1006" alt="卡片" style="zoom:33%;" />

> 图4 | 感知器重采样器模块将来自视觉编码器的*可变大小*的时空视觉特征网格映射到*固定*数量的输出标记（图中为五个），与输入图像分辨率或输入视频帧的数量无关。该转换器具有一组学习的潜在向量作为查询，并且键和值是时空视觉特征与学习的潜在向量的串联。



- Perceiver Resampler 基于 DeepMind 的论文[Perceiver: General Perception with Iterative Attention。](https://arxiv.org/abs/2103.03206)
- Flamingo 模型将来自视觉编码器的*可变数量*的图像或视频特征作为输入，并输出固定数量的视觉标记。
- 视觉输入被重新采样为固定且少量（实际上是 64 个）的输出，以显着降低视觉文本交叉注意力的计算复杂性。
- 输入到感知器重采样器的视觉特征是通过首先将学习的时间位置 ( *t* =0, *t* =1, *t* =2 ) 添加到对应的特征的每个空间网格来获得的到视频的给定帧，如图 4 所示。
- 作者只使用了时间编码，没有使用空间网格位置编码，因为后者没有带来任何改进。
- 然后这些视觉特征被扁平化为$X_{f}$。
- 与原始的[Perceiver](https://arxiv.org/abs/2103.03206)类似，该模型学习预定义数量的潜在输入查询。
- 这些潜在查询被馈送到变压器堆栈并交叉参与扁平化视觉特征 $X_{f}$。
- 从学习到的潜在变量计算出的键和值被连接到从 $X_{f}$。

### GATED XATTN-DENSE Layers

#### 门控 XATTN-DENSE LAYER 的伪代码

```python
def gated_xattn_dense(
 y, # input language features
 x, # input visual features
 alpha_xattn, # xattn gating parameter – init at 0.
 alpha_dense, # ffw gating parameter – init at 0.
):
    """Applies a GATED XATTN-DENSE layer."""
    # 1. Gated Cross Attention
    y = y + tanh(alpha_xattn) * attention(q=y, 
    kv=x)
    # 2. Gated Feed Forward (dense) Layer
    y = y + tanh(alpha_dense) * ffw(y)
    # Regular self-attention + FFW on language
    y = y + frozen_attention(q=y, kv=y)
    y = y + frozen_ffw(y)
    return y # output visually informed language features复制错误！复制了！
```

#### 门控 XATTN-DENSE LAYER

<img src="https://api.wandb.ai/files/gladiator/Flamingo%20VLM/16cbasdt/media/images/example_0_4ee56698aabcb42916a0.png?height=850" alt="卡片" style="zoom:33%;" />

图5 | 门控 XATTN-DENSE 层。作者插入了新的交叉注意力层，其键和值是在使用语言查询时从视觉特征中获得的，然后在现有的预训练和冻结的 LM 层之间插入密集的前馈层，以便根据视觉输入调节LM。这些层*经过门控*，以便 LM 在初始化时保持完整，从而提高稳定性和性能。

-  文本生成由 Transformer 解码器执行，以感知器重采样器生成的*视觉*表示$X$为条件。
- 作者使用了最大的 Flamingo 的70B 参数模型[Chinchilla](https://www.deepmind.com/publications/an-empirical-analysis-of-compute-optimal-large-language-model-training)模型作为语言模型。
- 在 Flamingo 的训练过程中，预训练的块被冻结，以保留纯文本语言模型中的信息和文本生成能力。
- 为了根据视觉输入调节 LM，作者在原始自注意力层之间插入了门控交叉注意力密集（如图 5 所示的 GATED XATTN-DENSE）块。请注意，原始的自注意力层在 Flamingo 的训练过程中被冻结，而新插入的交叉注意力层则从头开始训练。
- [LayerNorm](https://arxiv.org/abs/1607.06450)应用于所有注意力输入和前馈层（GPT-2 风格）。
- 作者还添加了 tanh 获取机制，以在初始化时保留原始语言模型行为，并且不会灾难性地改变 LM 学习到的特征。
- 新添加的交叉注意力层的输出乘以 $tanh( α)$，其中 $\alpha$ 是初始化为 0 的特定于层的可学习标量。
- 在训练过程中，由于门控机制，模型可以从完全训练的纯文本模型顺利过渡到视觉语言模型。

### 图像/视频的Attention Masking

#### Masked  Cross Attention

<img src="https://api.wandb.ai/files/gladiator/Flamingo%20VLM/38ghtrlr/media/images/example_0_b79357164aa154dda382.png?height=420" alt="卡片" style="zoom:33%;" />

图6 | 交错的视觉数据和文本支持。`<image>`给定与图像/视频交错的文本（例如来自网页），作者首先通过在文本中视觉数据的位置插入标签以及特殊标记（`<BOS>`用于“句子开头”或`<EOC>`“句子结尾” ）来处理文本。块”）。图像由视觉编码器和感知器重采样器独立处理以提取视觉标记。每个文本标记仅交叉关注与最后一个前面的图像对应的视觉标记。下面所示的函数 𝜙 为每个标记指示最后一个前面图像的索引是什么（如果没有前面的图像，则为 0）。在实践中，这种选择性交叉注意是通过屏蔽交叉注意机制实现的——此处用深蓝色条目（未屏蔽）和浅蓝色条目（屏蔽）进行说明。

- 训练数据还包括从网页抓取和处理的交错序列。
- 每个交错示例由：文本序列*y* 、图像序列*x*以及图像在文本中的位置序列组成。
- 基于视觉数据位置，作者定义了一个函数$\phi:[1, L] \mapsto[0, N]$为每个文本位置分配最后一个文本位置的索引出现在该位置之前的图像或视频（如果该位置之前没有出现视觉数据，则为 0）。
- 函数 𝜙 定义哪些视觉输入被认为可用于预测方程 - (1) 中的标记：前面标记的集合 $y_{<\ell} \triangleq\left(y_1, \ldots, y_{\ell-1}\right)$，以及先前图像或视频的集合 $x_{\leq \ell} \triangleq\left\{x_i \mid i \leq \phi(\ell)\right\}$.。
- 多图像注意力是通过门控 xattn 密集层实现的，并对来自感知器重采样器的标记进行因果屏蔽。
- 默认情况下，每个标记只允许关注紧邻其之前出现的图像的视觉标记（此限制提高了性能）。
- 尽管直接注意力集中在单个图像上，但仍然存在对先前图像的因果依赖性（由于文本解码器中的因果自注意力）。
- 此外，实验表明该模型可以在 5 张图像上进行训练，但最多可以泛化 32 张图像。

## 训练数据

### Flamingo接受以下训练：

- 图像-文本对数据
- 视频-文本对数据
- 网页数据（交错）

<img src="https://api.wandb.ai/files/gladiator/Flamingo%20VLM/1vb7zhnh/media/images/example_0_baf202437884e7263215.png?height=303" alt="卡片" style="zoom:33%;" />

> 图 7 | 训练数据集。不同性质的训练数据集的混合。N 对应于单个示例的视觉输入数量。对于配对图像（或视频）和文本数据集，N=1。T是视频帧的数量，T=1 是图像的特例。H,W,C 是高度、宽度和颜色通道。

### **MultiModel Massive Web (M3W)** (M3W)

- 从 4300 万个网页中提取文本和图像
- M3W 包含1.85 亿张图像和182 GB 文本。
- 文本过滤器和图像过滤器用于删除低质量数据。
- 图像的分辨率为320 x 320像素。
- 256的令牌序列长度用于文本。

### 图像-文本对数据

- ALIGN 数据集 - 18 亿个噪声图像文本对，平均每个图像12.4 个文本标记。
- LTIP 数据集 - 3.12 亿个图像-文本对，平均每个图像20.5 个文本标记
- 图像的分辨率为320 x 320像素。
- 32/64 个标记序列长度用于文本。

### 视频-文本对数据

- 包含 2700 万条短视频的 VTP 数据集。
- 平均持续时间22 秒。
- 帧的分辨率为320 x 320像素。
- 时间维度为8 (T = 8)。
- 32 个标记序列长度用于文本。

## Flamingo训练详情

### 损失和优化

- 所有模型均使用AdamW进行训练
- 优化是通过线性预热和常数学习率完成的（衰减并没有带来改进）。

数据集混合的权重如下($\lambda_{m}$)：

- M3W: 1.0
- LTIP: 0.2
- ALIGN: 0.2
- VTP: 0.03

### 训练目标

使用文本的数据集特定负对数似然性的加权和来训练模型（以视觉输入为条件）：

$\begin{aligned} & \sum_{\mathrm{m}=1}^{\mathrm{M}} \lambda_{\mathrm{m}} \mathbb{E}_{(\mathrm{x}, \mathrm{y}) \sim \mathcal{D}_{\mathrm{m}}}\left[-\sum_{\ell=1}^{\mathrm{L}} \log \mathrm{p}\left(\mathrm{y}_{\ell} \mid \mathrm{y}_{<\ell}, \mathrm{x}_{\leq \ell}\right)\right] \end{aligned}$

$\mathcal{D}_{\mathrm{m}}-\mathrm{m}_{\mathrm{th}} \text { dataset }$

$\lambda_{\mathrm{m}} \text { - positive scalar weight for the } \mathrm{m}_{\mathrm{th}} \text { dataset }$

与视觉编码器预训练类似，调整这些权重很重要，并且使用组合数据的累积策略。

### 基础设施/实施

- 使用[JAX](https://github.com/google/jax)和[Haiku](https://github.com/deepmind/dm-haiku)实现的模型和相关基础设施。
- 所有训练和评估均在TPUv4实例上完成。
- 最大 (80B) 模型在 16 台设备上的 1536 个芯片上训练了 15 天。
- [Megatron 分片](https://arxiv.org/pdf/1909.08053)用于 Embedding/S-Attention/X-Attention/FFW。
- [ZeRO 阶段 1](https://arxiv.org/abs/2201.05596)用于对优化器状态进行分片。
- 激活+梯度保存在 中`bfloat16`，参数+优化器累加器保存在 中`float32`。

## 通过少样本上下文学习进行任务适应

- 作者使用[GPT-3](https://arxiv.org/abs/2005.14165)普及的[情境学习](http://ai.stanford.edu/blog/understanding-incontext/)来评估模型快速适应新任务的能力。
- 该模型以（图像，文本）或（视频，文本）的形式给出一组支持示例（其中图像或视频是输入视觉效果，文本是预期响应）以及单个视觉查询，其中模型需要做出预测。
- 给定一些支持示例，作者通过连接支持示例和视觉查询来构建多模态提示，如图 8 所示。

![卡片](https://api.wandb.ai/files/gladiator/Flamingo%20VLM/3i4lb3hh/media/images/example_0_68a6a3e53b26b09415cb.png?height=646)

> 图8 | 少镜头交错提示生成。给定一些特定于任务的少数示例（支持示例）和 Flamingo 模型必须进行预测的查询，作者通过在每个相应文本之前交错图像来构建提示。他们引入了一些格式来做到这一点，例如，他们预先考虑`"Output:"`所有视觉对文本任务的预期响应，或者`"Question: {question} Answer: {answer}"`对视觉回答任务使用格式提示。

### **Prompt Ensembling**

- 跨多个提示的集成可用于提高性能。
- 此外，这可以与最近邻的不同排列上的RICES相结合。
- 对于给定的答案，对数似然集成在所选少主机提示的六种随机排列上。

### retrieval-based-in-context-example-selection

- 利用大量支持示例可能很困难，因为在提示中容纳所有示例的成本很高，并且如果在训练中使用较少的示例，泛化可能会受到影响。
- 作者使用了[Yang 等人](https://ojs.aaai.org/index.php/AAAI/article/view/20215)提出的基于检索的上下文示例选择（RICES ）
- RICES 通过比较从冻结的预训练视觉编码器中提取的视觉特征，检索支持集中的相似图像作为查询图像。
- 该提示是通过连接前 N 个最相似的示例来构建的。

### zero-shot-generalization

- 在缺乏少量示例的情况下，一种方法是使用即时工程（如 CLIP 所示）。
- 验证需要访问大量带注释的示例，因为性能对提示很敏感，因此不能被视为真正的零样本。
- 为了报告零样本性能，作者使用下游任务中的两个示例构建了一个提示，在这些任务中，他们删除了相应的图像或视频（输入），同时保留了输出，以便让模型了解预期输出的格式。
- 作者报告说，仅显示一个而不是两个文本示例会使模型产生与单个提供的文本示例类似的输出。

### open-ended-and-close-ended-evaluations)

- 在开放式设置中，模型在查询图像之后的采样文本被视为对图像的预测，在第一个`<EOS>`标记处停止。
- 使用波束搜索，波束大小为3。
- 在封闭式设置中，所有可能的输出都独立地附加到查询图像，然后按对数似然对序列进行排序。

## Flamingo模型结果

### Flamingo 模型的参数计数

表 2 | Flamingo 模型的参数计数。作者重点关注增加冻结 LM 和可训练视觉文本 GATED XATTN-DENSE 模块的参数数量，同时将冻结视觉编码器和可训练重采样器在不同模型中保持固定且较小的尺寸。括号中给出了 GATED XATTN-DENSE 相对于原始语言模型块的频率

<img src="https://api.wandb.ai/files/gladiator/Flamingo%20VLM/1beqi31r/media/images/example_0_3dc87000d3c3789c9bca.png?height=257" alt="卡片" style="zoom:50%;" />

### 定性结果

下面的面板使用作者在论文中提供的示例展示了 Flamingo 如何在各种任务中执行。要查看更多相同类型的样本，请向右拖动蓝色滑块（步骤）。

这是本报告中最令人兴奋的部分。我们将看到该模型如何在各种任务中执行。请注意，所有结果均取自论文，因为该模型不是开源的。

下面的结果显示了最简单的交互形式，其中提供单个图像，然后以问题或标题开头的形式提供文本提示。即使模型没有以问答形式进行训练，预训练的语言模型的功能也允许这种适应。

#### 单张图像样本

<img src="https://api.wandb.ai/files/gladiator/Flamingo%20VLM/13fkcfn1/media/images/image_0_eac19e938c1f10017545.png?height=699" alt="卡片" style="zoom: 33%;" />

> 单图像样本：灰色框是用户输入，粉色框是 Flamingo 输出。

由于 Flamingo 模型可以接受任意视觉和语言序列形式的输入，因此作者测试了其与交错图像和文本进行扩展对话的能力。有趣的是，即使经过几轮交互，Flamingo 仍然可以成功地关注图像并回答问题。正如您在下面的一些示例中所看到的，Flamingo 还展示了扎实的 OCR 能力、对分布变化的鲁棒性和复杂推理能力。

#### Flamingo对话互动

<img src="https://api.wandb.ai/files/gladiator/Flamingo%20VLM/mmm2zdvc/media/images/image_0_0f1ed96210914f61bc7d.png?height=1214" alt="卡片" style="zoom:33%;" />

> 对话样本。灰色框是用户输入，粉色框是 Flamingo 输出。

Flamingo 还可以集成来自多个帧的信息（例如扫描场景或文本的视频）并响应涉及时间理解的请求，如下面的结果所示。

#### 视频样本

<img src="https://api.wandb.ai/files/gladiator/Flamingo%20VLM/hjpxmhby/media/images/image_0_eea7d6e6d5dabb7889bf.png?height=819" alt="卡片" style="zoom:33%;" />

视频样本。这些是模型看到的所有框架。

## 结论

Flamingo 是一个“通用”模型系列，可应用于图像和视频，以最少的特定于任务的训练数据来理解任务。仅给出几个示例，单个 Flamingo 模型就可以在多种任务上实现最先进的效果，通常与需要对更多示例进行微调的方法相竞争。尽管该模型存在一些弱点，例如在分类任务上的表现比对比模型更差，直接继承了语言模型的所有偏见、毒性和弱点，有时在开放式视觉问答任务中出现幻觉和无根据的猜测（更多详细信息见本文第 6 节）。

然而，该模型为多模态系统提供了一个充满希望的未来，该系统可以在无需明确训练的情况下执行多种任务。看看未来如何将音频等其他模态集成到这些系统中也很有趣。

## 参考

- [DeepMind blog on Flamingo](https://www.deepmind.com/blog/tackling-multiple-tasks-with-a-single-visual-language-model)
- [Flamingo official paper](https://arxiv.org/abs/2204.14198)
- [A digest of the Flamingo model by Samuel Albanie](https://www.youtube.com/watch?v=H82s6BrJduM)
- [Unofficial implementation of the key ideas of Flamingo](https://github.com/lucidrains/flamingo-pytorch)