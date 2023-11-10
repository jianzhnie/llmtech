# 文本生成视频: 任务、挑战及现状

![video-samples](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/140_text-to-video/text-to-video-samples.gif)

示例视频由 [ModelScope 生成](https://modelscope.cn/models/damo/text-to-video-synthesis/summary)。

最近生成模型方向的进展如排山倒海，令人目不暇接，而文生视频将是这一连串进展的下一波。尽管大家很容易从字面上理解文生视频的意思，但它其实是一项相当新的计算机视觉任务，其要求是根据文本描述生成一系列时间和空间上都一致的图像。虽然看上去这项任务与文生图极其相似，但众所周知，它的难度要大得多。这些模型是如何工作的，它们与文生图模型有何不同，我们对其性能又有何期待？

## 文本生成视频使用案例

### 基于脚本的视频生成

文本到视频技术（Text-to-video）可以根据提供的文本脚本创建短视频内容。这些模型可以用于制作引人入胜、信息丰富的营销视频。例如，一家公司可以使用文本到视频模型制作一段视频，解释他们的产品是如何工作的。

### 内容格式转换

文本到视频模型可以从长文本中生成视频，包括博客文章、文章和文本文件。文本到视频模型可以用于创建更具吸引力和互动性的教育视频。一个例子是创建一段视频，解释文章中的复杂概念。

### 配音和语音

文本到视频模型可以用于创建人工智能新闻主播，每天播报新闻，或者供电影制片人制作短片或音乐视频使用。

### 任务变体

文本到视频模型根据输入和输出有不同的变体。

### 文本到视频编辑

文本到视频编辑任务是生成基于文本的视频样式和本地属性编辑。文本到视频编辑模型可以使诸如裁剪、稳定、色彩校正、调整大小和音频编辑等任务更容易一致执行。

### 文本到视频搜索

文本到视频搜索是检索与给定文本查询相关的视频的任务。这可能具有挑战性，因为视频是一个复杂的媒介，可能包含大量信息。通过使用语义分析来提取文本查询的含义，视觉分析来从视频中提取特征，例如视频中存在的物体和动作，以及时间分析来分类视频中物体和动作之间的关系，我们可以确定哪些视频最可能与文本查询相关。

### 文本驱动视频预测

文本驱动视频预测是从文本描述生成视频序列的任务。文本描述可以是从简单句子到详细故事的任何内容。该任务的目标是生成既视觉上逼真又在语义上与文本描述一致的视频。

### 视频翻译

文本到视频翻译模型可以将视频从一种语言翻译为另一种语言，或者允许用非英语句子查询多语言文本视频模型。这对于想要观看他们不懂的语言的视频的人们非常有用，特别是当有多语言字幕可用于训练时。

在本文中，我们将讨论文生视频模型的过去、现在和未来。我们将从回顾文生视频和文生图任务之间的差异开始，并讨论无条件视频生成和文生视频两个任务各自的挑战。此外，我们将介绍文生视频模型的最新发展，探索这些方法的工作原理及其性能。最后，我们将讨论我们在 Hugging Face 所做的工作，这些工作的目标就是促进这些模型的集成和使用，我们还会分享一些在 Hugging Face Hub 上以及其他一些地方的很酷的演示应用及资源。

![samples](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/140_text-to-video/make-a-video.png)
根据各种文本描述输入生成的视频示例，图片来自论文 [Make-a-Video](https://arxiv.org/abs/2209.14792)。

## 文生视频与文生图

最近文生图领域的进展多如牛毛，大家可能很难跟上最新的进展。因此，我们先快速回顾一下。

就在两年前，第一个支持开放词汇 (open-vocabulary) 的高质量文生图模型出现了。第一波文生图模型，包括 VQGAN-CLIP、XMC-GAN 和 GauGAN2，都采用了 GAN 架构。紧随其后的是 OpenAI 在 2021 年初发布的广受欢迎的基于 transformer 的 DALL-E、2022 年 4 月的 DALL-E 2，以及由 Stable Diffusion 和 Imagen 开创的新一波扩散模型。Stable Diffusion 的巨大成功催生了许多产品化的扩散模型，例如 DreamStudio 和 RunwayML GEN-1; 同时也催生了一批集成了扩散模型的产品，例如 Midjourney。

尽管扩散模型在文生图方面的能力令人印象深刻，但相同的故事并没有扩展到文生视频，不管是扩散文生视频模型还是非扩散文生视频模型的生成能力仍然非常受限。文生视频模型通常在非常短的视频片段上进行训练，这意味着它们需要使用计算量大且速度慢的滑动窗口方法来生成长视频。因此，众所周知，训得的模型难以部署和扩展，并且在保证上下文一致性和视频长度方面很受限。

文生视频的任务面临着多方面的独特挑战。主要有:

- 计算挑战: 确保帧间空间和时间一致性会产生长期依赖性，从而带来高计算成本，使得大多数研究人员无法负担训练此类模型的费用。
- 缺乏高质量的数据集: 用于文生视频的多模态数据集很少，而且通常数据集的标注很少，这使得学习复杂的运动语义很困难。
- 视频字幕的模糊性: “如何描述视频从而让模型的学习更容易”这一问题至今悬而未决。为了完整描述视频，仅一个简短的文本提示肯定是不够的。一系列的提示或一个随时间推移的故事才能用于生成视频。

在下一节中，我们将分别讨论文生视频领域的发展时间线以及为应对这些挑战而提出的各种方法。概括来讲，文生视频的工作主要可以分为以下 3 类:

1. 提出新的、更高质量的数据集，使得训练更容易。
2. 在没有 `文本 - 视频对` 的情况下训练模型的方法。
3. 计算效率更高的生成更长和更高分辨率视频的方法。

## 如何实现文生视频？

让我们来看看文生视频的工作原理以及该领域的最新进展。我们将沿着与文生图类似的研究路径，探索文生视频模型的流变，并探讨迄今为止我们是如何解决文生视频领域的具体挑战的。

与文生图任务一样，文生视频也是个年轻的方向，最早只能追溯到几年前。早期研究主要使用基于 GAN 和 VAE 的方法在给定文本描述的情况下自回归地生成视频帧 (参见 [Text2Filter](https://huggingface.co/papers/1710.00421) 及 [TGANs-C](https://huggingface.co/papers/1804.08264))。虽然这些工作为文生视频这一新计算机视觉任务奠定了基础，但它们的应用范围有限，仅限于低分辨率、短距以及视频中目标的运动比较单一、孤立的情况。

![tgans-c](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/140_text-to-video/TGANs-C.png)
最初的文生视频模型在分辨率、上下文和长度方面极为有限，图像取自 [TGANs-C](https://arxiv.org/abs/1804.08264)。

受文本 (GPT-3) 和图像 (DALL-E) 中大规模预训练 Transformer 模型的成功启发，文生视频研究的第二波浪潮采用了 Transformer 架构。[Phenaki](https://huggingface.co/papers/2210.02399)、[Make-A-Vide](https://huggingface.co/papers/2209.14792)、[NUWA](https://huggingface.co/papers/2111.12417)、[VideoGPT](https://huggingface.co/papers/2104.10157) 和 [CogVideo](https://huggingface.co/papers/2205.15868) 都提出了基于 transformer 的框架，而 [TATS](https://huggingface.co/papers/2204.03638) 提出了一种混合方法，从而将用于生成图像的 VQGAN 和用于顺序地生成帧的时间敏感 transformer 模块结合起来。在第二波浪潮的诸多框架中，Phenaki 尤其有意思，因为它能够根据一系列提示 (即一个故事情节) 生成任意长视频。同样，[NUWA-Infinity](https://huggingface.co/papers/2207.09814) 提出了一种双重自回归 (autoregressive over autoregressive) 生成机制，可以基于文本输入合成无限长度的图像和视频，从而使得生成高清的长视频成为可能。但是，Phenaki 或 NUWA 模型均无法从公开渠道获取。

![phenaki](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/140_text-to-video/phenaki.png)
Phenaki 的模型架构基于 transformer，图片来自 [此处](https://arxiv.org/abs/2210.02399)。

第三波也就是当前这一波文生视频模型浪潮主要以基于扩散的架构为特征。扩散模型在生成多样化、超现实和上下文丰富的图像方面取得了显著成功，这引起了人们对将扩散模型推广到其他领域 (如音频、3D ，最近又拓展到了视频) 的兴趣。这一波模型是由 [Video Diffusion Models](https://huggingface.co/papers/2204.03458) (VDM) 开创的，它首次将扩散模型推广至视频领域。然后是 [MagicVideo](https://huggingface.co/papers/2211.11018) 提出了一个在低维隐空间中生成视频剪辑的框架，据其报告，新框架与 VDM 相比在效率上有巨大的提升。另一个值得一提的是 [Tune-a-Video](https://huggingface.co/papers/2212.11565)，它使用 `单文本 - 视频对`微调预训练的文生图模型，并允许在保留运动的同时改变视频内容。随后涌现出了越来越多的文生视频扩散模型，包括 [Video LDM](https://huggingface.co/papers/2304.08818)、[Text2Video-Zero](https://huggingface.co/papers/2303.13439)、[Runway Gen1、Runway Gen2](https://huggingface.co/papers/2302.03011) 以及 [NUWA-XL](https://huggingface.co/papers/2303.12346)。

Text2Video-Zero 是一个文本引导的视频生成和处理框架，其工作方式类似于 ControlNet。它可以基于输入的 `文本数据` 或 `文本 + 姿势混合数据` 或 `文本 + 边缘混合数据` 直接生成 (或编辑) 视频。顾名思义，Text2Video-Zero 是一种零样本模型，它将可训练的运动动力学模块与预训练的文生图稳定扩散模型相结合，而无需使用任何 `文本 - 视频对` 数据。与 Text2Video-Zero 类似，Runway Gen-1 和 Runway Gen-2 模型可以合成由文本或图像描述的内容引导的视频。这些工作大多数都是在短视频片段上训练的，并且依靠带有滑动窗口的自回归机制来生成更长的视频，这不可避免地导致了上下文差异 (context gap)。 NUWA-XL 解决了这个问题，并提出了一种“双重扩散 (diffusion over diffusion)”方法，并在 3376 帧视频数据上训练模型。最后，还有一些尚未在同行评审的会议或期刊上发表的开源文本到视频模型和框架，例如阿里巴巴达摩院视觉智能实验室的 ModelScope 和 Tencel 的 VideoCrafter。

## 数据集

与其他视觉语言模型一样，文生视频模型通常在大型 `文本 - 视频对` 数据集上进行训练。这些数据集中的视频通常被分成短的、固定长度的块，并且通常仅限于少数几个目标的孤立动作。出现这种情况的一部分原因是计算限制，另一部分原因是以有意义的方式描述视频内容这件事本身就很难。而我们看到多模态视频文本数据集和文生视频模型的发展往往是交织在一起的，因此有不少工作侧重于开发更易于训练的更好、更通用的数据集。同时也有一些工作另辟蹊径，对替代解决方案进行了探索，例如 [Phenaki](https://phenaki.video/?mc_cid=9fee7eeb9d#) 将 `文本 - 图像对` 与 `文本 - 视频对` 相结合用于文生视频任务; Make-a-Video 则更进一步，提议仅使用 `文本 - 图像对` 来学习世界表象信息，并使用单模态视频数据以无监督的方式学习时空依赖性。

这些大型数据集面临与文本图像数据集类似的问题。最常用的文本 - 视频数据集 [WebVid](https://m-bain.github.io/webvid-dataset/) 由 1070 万个 `文本 - 视频对` (视频时长 5.2 万小时) 组成，并包含一定量的噪声样本，这些样本中的视频文本描述与视频内容是非相干的。其他数据集试图通过聚焦特定任务或领域来解决这个问题。例如，[Howto100M](https://www.di.ens.fr/willow/research/howto100m/) 数据集包含 13600 万个视频剪辑，其中文本部分描述了如何一步一步地执行复杂的任务，例如烹饪、手工制作、园艺、和健身。而 [QuerYD](https://www.robots.ox.ac.uk/~vgg/data/queryd/) 数据集则聚焦于事件定位任务，视频的字幕详细描述了目标和动作的相对位置。 [CelebV-Text](https://celebv-text.github.io/) 是一个包含超过 7 万个视频的大规模人脸文本 - 视频数据集，用于生成具有逼真的人脸、情绪和手势的视频。

## 多模态经典模型

### CLIP

### Flamingo

### BLIP

#### BLIP：理解、生成我都要

文章的研究动机：

- 现有的预训练模型通常在理解型任务或生成型任务中表现出色，但很少有模型能够同时在这两种任务上达到优秀的性能。
- 现有的性能改进主要是通过扩大数据集规模并使用从网络收集的带有噪声的图像-文本对进行训练实现的。然而，网络数据集中的噪声会对模型的性能产生负面影响。

主要的贡献：

- 统一了图像-语言的理解与生成任务
- Bootstrap 的方式清洗网络噪声数据

在模型的设计上结合了 ALBEF 和 VLMo，看下图中红色框中就类似 ALBEF，只是画 image-grounded text encoder 的位置不同；蓝色框中类似 VLMo，虽然有三个模型，但是大部分参数都是共享的。

- 左一为 Image Encoder（图像编码器）：该组件使用 Vision Transformer（ViT）对图像进行编码，将全局图像特征表示为一个额外的[CLS]标记。
- 左二为 Text Encoder，采用了 BERT 的结构，提取文本特征用于与视觉特征计算 ITC loss。Text Encoder 不与视觉特征计算交叉注意力。
- 左三为 Image-grounded Text Encoder（基于图像的文本编码器），该组件通过在每个 Transformer 块的自注意力（Self-Attention）层和前馈神经网络（Feed Forward Network）之间插入一个交叉注意力（Cross-Attention）层，将视觉信息注入到文本编码中，提取文本特征用于计算 ITM 损失。
- 左四为 Imagegrounded Text Decoder（基于图像的文本解码器），用于进行 LM 语言建模训练（这里不再是用 MLM 了），生成与图像相关的文本描述。
- 三个文本编解码器分别为在文本前添加 [CLS]、[Encode]、[Decode] token
- 与 ALBEF 一样，同样采用动量模型为 ITC 生成伪标签；使用 ITC 为 ITM 进行难负例挖掘。

#### BLIP 的训练流程

- 使用含噪声的数据训练一个 MED（Multimodal Mixture of Encoder-Decoder）模型；
- 将该模型的 Image-grounded Text Encoder 和 Image-grounded Text Decoder 在人工标注的 COCO 数据集上进行微调，分别作为 Filter 和 Captioner；
- Captioner 根据图像数据生成对应的文本描述；
- Filter 对噪声较大的网络数据和生成数据进行过滤清洗，得到较为可靠的训练数据；
- 再根据这些可靠的训练数据，训练更好的 MED 模型，从而实现 bootstraping 训练。

### BLIP2

#### BLIP2：将图像特征对齐到预训练语言模型

BLIP-2 通过在冻结的预训练图像编码器和冻结的预训练大语言模型之间添加一个轻量级 查询 Transformer (Query Transformer, Q-Former) 来弥合视觉和语言模型之间的模态隔阂。在整个模型中，Q-Former 是唯一的可训练模块，而图像编码器和语言模型始终保持冻结状态。

Q-Former 由两个子模块组成，这两个子模块共享相同的自注意力层:

- 与冻结的图像编码器交互的图像 transformer，用于视觉特征提取
- 文本 transformer，用作文本编码器和解码器

图像 transformer 从图像编码器中提取固定数量的输出特征，这里特征的个数与输入图像分辨率无关。同时，图像 transformer 接收若干查询嵌入作为输入，这些查询嵌入是可训练的。这些查询还可以通过共享的自注意力层与文本进行交互。

Q-Former 分两个阶段进行预训练。第一阶段，图像编码器被冻结，Q-Former 通过三个损失函数进行训练:

- ITC loss
- ITM loss
- Image-grounded Text Generation (ITG) loss：用于训练 Q-Former 模型在给定输入图像条件下生成文本。在注意力机制上，queries 之间互相可见但是不能看到文本 token，而文本 token 可以看到所有的 queries 以及它之前的文本 token。此外将 CLS token 替换为 DEC token 以便提示模型进行解码任务。

通过第一阶段的训练，Query 已经能够理解图片的含义了，接下来就是让 LLM 也能够理解图片信息，因此作者针对两类不同 LLM 设计了不同的任务：

- Decoder 类型的 LLM（如 OPT）：以 Query 做输入，文本做目标；
- Encoder-Decoder 类型的 LLM（如 FlanT5）：以 Query 和一句话的前半段做输入，以后半段做目标；因为不同模型的 embedding 维度不同，所以这里还加上了一个全连接层。

BLIP2 验证了之前的想法，直接利用已经预训练好的视觉、文本模型，通过设计参数量较少的“对齐模块”来实现多模态的对齐。

然而，注意到 BLIP2 在抽视觉特征其实是不考虑文本的；此时也正值 指令微调 在大语言模型中大杀四方，因此进一步的发展方向也就诞生了。

### LLaVA

### **ImageBind：更多模态一起对齐**

ImageBind 的目标是将不同模态的 embedding 对齐到一个公共的空间，可以理解为是 **CLIP 的多模态版本。**

文章的主要思想是通过**图片作为桥梁**来将不同模态的数据关联起来。

### **Meta-Transformer：未来就是要统一**

Meta-Transformer 野心就比较大了，同时考虑了 12 种模态。

它的主要思想是使用一个统一的框架来处理来自多种模态的数据，而无需为每种模态设计特定的模型或网络。**通过将所有模态的数据映射到一个共享的 embedding 空间，并使用一个公共的编码器来提取特征。**

- 统一的 Tokenization：通过设计特定的 Tokenization 策略，例如将图像分割成小块或将文本分割成词或子词，然后为每个块或词生成一个 token。这些 token 然后被映射到一个连续的向量空间，形成 token embedding；
- 模态共享的编码器：使用一个预训练的 Transformer 编码器，它的参数是冻结的。这个编码器可以处理来自不同模态的 token embedding（因为它们都在同一个共享的流形空间内）；
- 任务特定的头部：这些头部通常由多层感知机(MLP)组成，并根据不同的模态和任务进行调整。

## Text-2-Video 模型

|                     | Models & Method                                              | Github                                                    | Time | Star |
| ------------------- | ------------------------------------------------------------ | --------------------------------------------------------- | ---- | ---- |
| THUDM               | [CogVideo](https://github.com/THUDM/CogVideo)                | https://github.com/THUDM/CogVideo                         | 2022 |      |
| omerbt              | [TokenFlow](https://github.com/omerbt/TokenFlow)             | https://github.com/omerbt/TokenFlow                       | 2023 |      |
| Show Lab            | [Tune-A-Video](https://github.com/showlab/Tune-A-Video)      | https://github.com/showlab/Tune-A-Video                   | 2023 |      |
| Picsart-AI-Research | [Text2Video-Zero](https://github.com/Picsart-AI-Research/Text2Video-Zero) | https://github.com/Picsart-AI-Research/Text2Video-Zero    | 2023 |      |
| ExponentialML       | [Text-To-Video-Finetuning](https://github.com/ExponentialML/Text-To-Video-Finetuning) | https://github.com/ExponentialML/Text-To-Video-Finetuning | 2023 |      |
| Hotshotco           | [Hotshot-XL](https://huggingface.co/hotshotco/Hotshot-XL)    | https://huggingface.co/hotshotco/Hotshot-XL               | 2023 |      |
| TencentAILab-CVC    | [VideoCrafter](https://github.com/AILab-CVC/VideoCrafter)    | https://github.com/AILab-CVC/VideoCrafter                 | 2023 |      |
|                     | [Macaw-LLM]()                                                | https://github.com/lyuchenyang/Macaw-LLM                  | 2023 |      |
|                     | [FateZero](https://github.com/ChenyangQiQi/FateZero)         | https://github.com/ChenyangQiQi/FateZero                  | 2023 |      |
|                     | [MetaTransformer]()                                          | https://github.com/invictus717/MetaTransformer            | 2023 |      |
| 厦门大学            | LaVIN                                                        | https://github.com/luogen1996/LaVIN                       |      |      |

## Hugging Face 上的文生视频

使用 Hugging Face Diffusers，你可以轻松下载、运行和微调各种预训练的文生视频模型，包括 Text2Video-Zero 和 [阿里巴巴达摩院](https://huggingface.co/damo-vilab) 的 ModelScope。我们目前正在努力将更多优秀的工作集成到 Diffusers 和 🤗 Transformers 中。

| Organization                                                 | HuggingFace  ModelZoo                                        |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| Ali Damo                                                     | https://huggingface.co/damo-vilab/text-to-video-ms-1.7b      |
| zeroscope                                                    | https://huggingface.co/cerspense/zeroscope_v2_576w           |
| hotshotco                                                    | https://huggingface.co/hotshotco/Hotshot-XL                  |
| Ali Damo                                                     | https://huggingface.co/damo-vilab/modelscope-damo-text-to-video-synthesis |
| [Text-to-video-synthesis-colab](https://github.com/camenduru/text-to-video-synthesis-colab) | https://github.com/camenduru/text-to-video-synthesis-colab   |

### Hugging Face 应用演示

在 Hugging Face，我们的目标是使 Hugging Face 库更易于使用并包含最先进的研究。你可以前往 Hub 查看和体验由 🤗 团队、无数社区贡献者和研究者贡献的 Spaces 演示。目前，上面有 [VideoGPT](https://huggingface.co/spaces/akhaliq/VideoGPT)、[CogVideo](https://huggingface.co/spaces/THUDM/CogVideo)、[ModelScope 文生视频](https://huggingface.co/spaces/damo-vilab/modelscope-text-to-video-synthesis) 以及 [Text2Video-Zero](https://huggingface.co/spaces/PAIR/Text2Video-Zero) 的应用演示，后面还会越来越多，敬请期待。要了解这些模型能用来做什么，我们可以看一下 Text2Video-Zero 的应用演示。该演示不仅展示了文生视频应用，而且还包含多种其他生成模式，如文本引导的视频编辑，以及基于姿势、深度、边缘输入结合文本提示进行联合条件下的视频生成。

除了使用应用演示来尝试预训练文生视频模型外，你还可以使用 [Tune-a-Video 训练演示](https://huggingface.co/spaces/Tune-A-Video-library/Tune-A-Video-Training-UI) 使用你自己的 `文本 - 视频对`微调现有的文生图模型。仅需上传视频并输入描述该视频的文本提示即就可以了。你可以将训得的模型上传到公开的 Tune-a-Video 社区的 Hub 或你私人用户名下的 Hub。训练完成后，只需转到演示的 Run 选项卡即可根据任何文本提示生成视频。

🤗 Hub 上的所有 Space 其实都是 Git 存储库，你可以在本地或部署环境中克隆和运行它们。下面克隆一下 ModelScope 演示，安装环境，并在本地运行它。

```shell
git clone https://huggingface.co/spaces/damo-vilab/modelscope-text-to-video-synthesis
cd modelscope-text-to-video-synthesis
pip install -r requirements.txt
python app.py
```

这就好了！ Modelscope 演示现在已经在你的本地计算机上运行起来了。请注意，Diffusers 支持 ModelScope 文生视频模型，你只需几行代码即可直接加载并使用该模型生成新视频。

```python
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

prompt = "Spiderman is surfing"
video_frames = pipe(prompt, num_inference_steps=25).frames
video_path = export_to_video(video_frames)
```

### 其他的社区开源文生视频项目

最后，还有各种不在 Hub 上的开源项目和模型。一些值得关注的有 Phil Wang (即 lucidrains) 的 [Imagen](https://github.com/lucidrains/imagen-pytorch) 非官方实现、[Phenaki](https://github.com/lucidrains/phenaki-pytorch)、[NUWA](https://github.com/lucidrains/nuwa-pytorch), [Make-a-Video](https://github.com/lucidrains/make-a-video-pytorch) 以及 [Video Diffusion 模型](https://github.com/lucidrains/video-diffusion-pytorch)。还有一个有意思的项目 [ExponentialML](https://github.com/ExponentialML/Text-To-Video-Finetuning)，它是基于 🤗 Diffusers 的，用于微调 ModelScope 文生视频模型。

## 总结

文生视频的研究正在呈指数级发展，但现有工作在上下文一致性上仍有限制，同时还面临其他诸多挑战。在这篇博文中，我们介绍了文生视频模型的限制、独特挑战和当前状态。我们还看到了最初为其他任务设计的架构范例如何赋能文生视频任务的巨大飞跃，以及这对未来研究意味着什么。虽然进展令人印象深刻，但与文生图模型相比，文生视频模型还有很长的路要走。最后，我们还展示了如何通过 Hub 上的应用演示来使用这些模型，以及如何将这些模型作为 🤗 Diffusers 流水线的一部分来完成各种任务。

## **参考文献**

- Learning Transferable Visual Models From Natural Language Supervision
- Align before Fuse: Vision and Language Representation Learning with Momentum Distillation
- VLMo: Unified Vision-Language Pre-Training with Mixture-of-Modality-Experts
- BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation
- CoCa: Contrastive Captioners are Image-Text Foundation Models
- Image as a Foreign Language: BEiT Pretraining for All Vision and Vision-Language Tasks
- BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models
- InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning
- MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models
- MiniGPT-5: Interleaved Vision-and-Language Generation via Generative Vokens
- The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)
- ImageBind: One Embedding Space To Bind Them All
- Meta-Transformer: A Unified Framework for Multimodal Learning
