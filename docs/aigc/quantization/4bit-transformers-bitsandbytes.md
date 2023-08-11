# 使用位和字节、4 位量化和 QLoRA 使 LLM 更易于访问

众所周知，LLM 规模庞大，在消费类硬件中运行或训练它们对用户和可访问性来说是一个巨大的挑战。我们的[LLM.int8 博文展示了如何使用库将](https://huggingface.co/blog/hf-bitsandbytes-integration)[LLM.int8 论文](https://arxiv.org/abs/2208.07339)中的何使用bitsandbytes库整合到Transformer 的。在我们努力让任何人都更容易访问模型时，我们决定再次与 bitsandbytes 合作，让用户能够以 4 位精度运行模型。这包括绝大多数 HF 模型，在任何模态（文本、视觉、多模态等）中。用户还可以利用 Hugging Face 生态系统中的工具在 4 位模型之上训练Adapter。这是 Dettmers 等人今天在 QLoRA 论文中介绍的一种新方法。论文摘要如下：

> 我们介绍了 QLoRA，这是一种有效的微调方法，可以减少内存使用量，足以在单个 48GB GPU 上微调 65B 参数模型，同时保留完整的 16 位微调任务性能。QLoRA 通过冻结的 4 位量化预训练语言模型将梯度反向传播到低阶Adapter~(LoRA)。我们最好的模型系列，我们命名为 Guanaco，在 Vicuna 基准测试中优于所有以前公开发布的模型，达到 ChatGPT 性能水平的 99.3%，同时只需要在单个 GPU 上进行 24 小时的微调。QLoRA 引入了许多创新来节省内存而不牺牲性能：(a) 4 位 NormalFloat (NF4)，一种新的数据类型，理论上是正态分布权重的最佳信息 (b) 双量化，通过量化常数来减少平均内存占用，以及 (c) 分页优化器来管理内存峰值。我们使用 QLoRA 对 1,000 多个模型进行微调，提供跨 8 个指令数据集、多种模型类型（LLaMA、T5）和无法通过常规微调运行的模型规模（例如 33B 和65B参数模型）。我们的结果表明，即使使用比以前的 SoTA 更小的模型，QLoRA 在小型高质量数据集上进行微调也会产生最先进的结果。我们提供了基于人类和 GPT-4 评估的聊天机器人性能的详细分析，表明 GPT-4 评估是人类评估的廉价且合理的替代方案。此外，我们发现当前的聊天机器人基准测试无法准确评估聊天机器人的性能水平。柠檬挑选的分析表明与 ChatGPT 相比，Guanaco 失败的地方。我们发布了所有模型和代码，包括用于 4 位训练的 CUDA 内核。

## 资源

这篇博文和版本附带了一些资源，可帮助您开始使用 4 位模型和 QLoRA：

- [Original paper](https://arxiv.org/abs/2305.14314)
- [Basic usage Google Colab notebook](https://colab.research.google.com/drive/1ge2F1QSK8Q7h0hn3YKuBCOAS0bK8E0wf?usp=sharing) - 该笔记本展示了如何使用 4 位模型对其所有变体进行推理，以及如何在免费的 Google Colab 实例上运行 GPT-neo-X（20B 参数模型）🤯
- [微调 Google Colab 笔记本](https://colab.research.google.com/drive/1VoYNfYDKcKRQRor98Zbf2-9VQTtGJ24k?usp=sharing)- 该笔记本展示了如何使用 Hugging Face 生态系统在下游任务上微调 4 位模型。我们证明可以在 Google Colab 实例上微调 GPT-neo-X 20B！
- [Original repository for replicating the paper's results](https://github.com/artidoro/qlora)
- [Guanaco 33b playground](https://huggingface.co/spaces/uwnlp/guanaco-playground-tgi) - or check the playground section below

## 介绍

如果您不熟悉模型精度和最常见的数据类型（float16、float32、bfloat16、int8），我们建议您仔细阅读我们第一篇[博文](https://huggingface.co/blog/hf-bitsandbytes-integration)中的介绍，其中以简单的术语和可视化的方式详细介绍了这些概念。

有关更多信息，我们建议阅读此 [wikibook 文档](https://en.wikibooks.org/wiki/A-level_Computing/AQA/Paper_2/Fundamentals_of_data_representation/Floating_point_numbers#:~:text=In decimal%2C very large numbers,be used for binary numbers.)中的浮点表示基础知识。

最近的 QLoRA 论文探讨了不同的数据类型，4 位 Float 和 4 位 NormalFloat。我们将在这里讨论 4 位 Float 数据类型，因为它更容易理解。

FP8 和 FP4 分别代表浮点 8 位和 4 位精度。它们是浮点值 minifloats 系列的一部分（除其他精度外，minifloats 系列还包括 bfloat16 和 float16）。

让我们先看看如何用 FP8 格式表示浮点值，然后了解 FP4 格式的样子。

### FP8格式

正如我们在之前的博文中所讨论的，一个浮点包含n个比特，每个比特都属于一个特定的类别，负责代表数字的一个组成部分（符号、尾数和指数）。这些代表了以下内容。

FP8（floating point 8）格式在论文[“FP8 for Deep Learning”](https://arxiv.org/pdf/2209.05433.pdf)中首次引入，具有两种不同的 FP8 编码：E4M3（4 位指数和 3 位尾数）和 E5M2（5 位指数和 2 位尾数）尾数）。

| ![fp8_方案](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/bitsandbytes/FP8-scheme.png) |
| ------------------------------------------------------------ |
| **浮点 8 (FP8) 格式概述。来源：原创内容来自[`sgugger`](https://huggingface.co/sgugger)** |

虽然通过将位数从 32 位减少到 8 位大大降低了精度，但这两个版本都可以在各种情况下使用。目前可以使用[Transformer Engine 库](https://github.com/NVIDIA/TransformerEngine)，该库也通过 accelerate 与 HF 生态系统集成。

可以用 E4M3 格式表示的潜在浮点数在 -448 到 448 范围内，而在 E5M2 格式中，随着指数位数的增加，范围增加到 -57344 到 57344 - 但有损失精度，因为可能表示的数量保持不变。经验证明，E4M3 最适合前向传播，而第二个版本最适合后向计算.

### FP4 精度简述

符号位代表符号(+/-)，指数位是以位代表的整数为基数的2次方(如2^{010}=2^{2}=4)，分数或尾数是负2次方的总和，每一个位为 "1 "时都是 "激活"。 如果某个位是“0”，则分数保持不变，其中`2^-i`i 是该位在位序列中的位置。例如，对于尾数位 1010，我们有`(0 + 2^-1 + 0 + 2^-3) = (0.5 + 0.125) = 0.625`. 为了得到一个值，我们将分数加*1*并将所有结果相乘，例如，使用 2 个指数位和一个尾数位，表示 1101 将是：

```
-1 * 2^(2) * (1 + 2^-1) = -1 * 4 * 1.5 = -6
```

对于 FP4 没有固定的格式，因此可以尝试不同尾数/指数组合的组合。通常，在大多数情况下，3 个指数位会好一些。但有时 2 个指数位和一个尾数位会产生更好的性能。

## QLoRA 论文，一种大众化量化大型 Transformer 模型的新方法

换句话说，与标准的16位模型微调相比，QLoRA减少了LLM微调的内存使用，而没有性能上的牺牲。这种方法可以在单个24GB的GPU上实现33B的模型微调，在单个46GB的GPU上实现65B的模型微调。

更具体地说，QLoRA使用4位量化来压缩一个预训练的语言模型。然后，LM参数被冻结，相对较少的可训练参数以Low-Rank Adapters的形式被添加到模型中。在微调过程中，QLoRA通过冻结的4位量化预训练语言模型将梯度反向传播到低等级适配器中。LoRA层是训练期间唯一被更新的参数。[在原始的 LoRA 论文](https://arxiv.org/abs/2106.09685)中阅读有关 LoRA 的更多信息。

QLoRA 有一种用于基本模型权重的存储数据类型（通常是 4 位 NormalFloat）和一种用于执行计算的计算数据类型（16 位 BrainFloat）。QLoRA 将存储数据类型的权重反量化为计算数据类型以执行前向和反向传递，但仅计算使用 16 位 bfloat 的 LoRA 参数的权重梯度。权重仅在需要时解压缩，因此在训练和推理期间内存使用率保持较低。

在广泛的实验中，QLoRA 调整显示与 16 位微调方法相匹配。此外，在[OpenAssistant 数据集 (OASST1)](https://huggingface.co/datasets/OpenAssistant/oasst1)上对 LLaMA 模型使用 QLoRA 微调的 Guanaco 模型是最先进的聊天机器人系统，在 Vicuna 基准测试上接近 ChatGPT。这是对 QLoRA 调优功能的额外展示。

如需更详细的阅读，我们建议您阅读[QLoRA 论文](https://arxiv.org/abs/2305.14314)。

## 如何在Transformer 使用它？

在本节中，我们将介绍这种方法的 transformers 集成，如何使用它以及可以有效量化哪些模型。

### 入门

作为快速入门，通过（在撰写本文时）从源代码安装加速器和转换器来加载 4 位模型，并确保您已安装最新版本的 bitsandbytes 库 (0.39.0)。

```bash
pip install -q -U bitsandbytes
pip install -q -U git+https://github.com/huggingface/transformers.git
pip install -q -U git+https://github.com/huggingface/peft.git
pip install -q -U git+https://github.com/huggingface/accelerate.git
```

### 快速开始

在 4bit 中加载模型的基本方法是`load_in_4bit=True`在调用`from_pretrained`方法时通过提供设备映射来传递参数（传递`"auto"`以获取将自动推断的设备映射）。

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m", load_in_4bit=True, device_map="auto")
...
```

这就是您所需要的！

作为一般规则，我们建议用户在加载模型后不要手动设置设备`device_map`。因此，在该行之后应避免对模型或任何模型的子模块进行任何设备分配调用 - 除非您知道自己在做什么。

请记住，加载量化模型会自动将其他模型的子模块转换为`float16`dtype。您可以通过传递给方法来更改此行为（例如，如果您想在 中使用图层规范`float32`）。`torch_dtype=dtype``from_pretrained`

### 高级用法

您可以使用 4 位量化的不同变体，例如 NF4（规范化浮点数 4（默认））或纯 FP4 量化。基于论文的理论考虑和实证结果，我们建议使用 NF4 量化以获得更好的性能。

其他选项包括`bnb_4bit_use_double_quant`在第一个量化之后使用第二个量化来为每个参数节省额外的 0.4 位。最后，计算类型。虽然 4 位 bitsandbytes 以 4 位存储权重，但计算仍然以 16 位或 32 位进行，这里可以选择任何组合（float16、bfloat16、float32 等）。

如果使用 16 位计算数据类型（默认 torch.float32），矩阵乘法和训练会更快。人们应该利用最近的`BitsAndBytesConfig`Transformer来改变这些参数。下面是一个使用 NF4 量化加载 4 位模型的示例，使用计算数据类型 bfloat16 进行双量化以加快训练速度：

```python
from transformers import BitsAndBytesConfig


nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

model_nf4 = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=nf4_config)
```



#### 更改计算数据类型

如上所述，您还可以通过更改`bnb_4bit_compute_dtype`中的参数来更改量化模型的计算数据类型`BitsAndBytesConfig`。

```python
import torch
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)
```

#### 嵌套量化

要启用嵌套量化，您可以使用`bnb_4bit_use_double_quant`中的参数`BitsAndBytesConfig`。这将在第一次量化之后启用第二次量化，以便为每个参数额外节省 0.4 位。我们也在训练 Google colab notebook 中使用了这个特性。

```python
from transformers import BitsAndBytesConfig

double_quant_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_use_double_quant=True,
)

model_double_quant = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=double_quant_config)
```

当然，如本节开头所述，所有这些组件都是可组合的。您可以将所有这些参数组合在一起以找到最适合您的用例。一条经验法则是：如果内存有问题，请使用双量化，使用 NF4 以获得更高的精度，并使用 16 位 dtype 来实现更快的微调。例如，在[推理演示](https://colab.research.google.com/drive/1ge2F1QSK8Q7h0hn3YKuBCOAS0bK8E0wf?usp=sharing)中，我们使用嵌套量化、bfloat16 计算 dtype 和 NF4 量化在单个 16GB GPU 中以 4 位完全适应 gpt-neo-x-20b (40GB)。

### 常见问题

在本节中，我们还将解决任何人可能对此集成提出的一些常见问题。

#### FP4 量化有硬件要求吗？

请注意，此方法仅与 GPU 兼容，因此无法在 CPU 上以 4 位量化模型。在 GPU 中，这种方法应该没有任何硬件要求，因此只要安装了 CUDA>=11.2，任何 GPU 都可以用于运行 4bit 量化。还要记住，计算不是在 4 位中完成的，权重和激活被压缩为该格式，并且计算仍然保持在所需的或本机 dtype 中。

#### 支持的型号有哪些？

[与本博文](https://huggingface.co/blog/hf-bitsandbytes-integration)中介绍的 LLM.int8 集成类似，集成在很大程度上依赖于`accelerate`库。因此，任何支持加速加载的模型（即`device_map`调用时的参数`from_pretrained`）都应该是4bit可量化的。另请注意，这与模态完全无关，只要模型可以加载参数`device_map`，就可以量化它们。

对于文本模型，在撰写本文时，这将包括最常用的架构，例如用于文本模型的 Llama、OPT、GPT-Neo、GPT-NeoX、用于多模态模型的 Blip2 等。

在撰写本文时，支持加速的模型有：

```python
[
    'bigbird_pegasus', 'blip_2', 'bloom', 'bridgetower', 'codegen', 'deit', 'esm',
    'gpt2', 'gpt_bigcode', 'gpt_neo', 'gpt_neox', 'gpt_neox_japanese', 'gptj', 'gptsan_japanese',
    'lilt', 'llama', 'longformer', 'longt5', 'luke', 'm2m_100', 'mbart', 'mega', 'mt5', 'nllb_moe',
    'open_llama', 'opt', 'owlvit', 'plbart', 'roberta', 'roberta_prelayernorm', 'rwkv', 'switch_transformers',
    't5', 'vilt', 'vit', 'vit_hybrid', 'whisper', 'xglm', 'xlm_roberta'
]
```

请注意，如果您最喜欢的模型不在那里，您可以打开一个合并请求或在转换器中提出一个问题，以添加对该架构的加速加载支持。

#### 我们可以训练 4 位/8 位模型吗？

不可能在这些模型上执行纯 4 位训练。但是，您可以通过利用参数高效微调方法 (PEFT) 来训练这些模型，并在它们之上训练例如Adapter。这就是论文中所做的，并得到 Hugging Face 的 PEFT 库的正式支持。我们还提供了一个[培训笔记本](https://colab.research.google.com/drive/1VoYNfYDKcKRQRor98Zbf2-9VQTtGJ24k?usp=sharing)，如果用户有兴趣复制论文中的结果，建议他们查看[QLoRA 存储库。](https://github.com/artidoro/qlora)

| ![劳拉-gif](https://huggingface.co/datasets/trl-internal-testing/example-images/resolve/main/blog/133_trl_peft/lora-animated.gif) |
| ------------------------------------------------------------ |
| **原始（冻结的）预训练权重（左）的输出激活由一个由权重矩阵 A 和 B 组成的低秩Adapter（右）增强。** |

#### 还有什么其他后果？

这种集成可以为社区和 AI 研究带来一些积极的影响，因为它可以影响多个用例和可能的应用程序。在 RLHF（人类反馈强化学习）中，可以加载一个 4 位基础模型并在其上训练多个Adapter，一个用于奖励建模，另一个用于价值策略训练。关于此用例的更详细的博文和公告将很快发布。

我们还针对这种量化方法对在消费类硬件上训练大型模型的影响做了一些基准测试。我们在 NVIDIA T4 (16GB) 上运行了几个微调 2 种不同架构的实验，Llama 7B（fp16 中的 15GB）和 Llama 13B（fp16 中的 27GB），这是结果：

| Model name                     | Half precision model size (in GB) | Hardware type / total VRAM | quantization method (CD=compute dtype / GC=gradient checkpointing / NQ=nested quantization) | batch_size | gradient accumulation steps | optimizer | seq_len | Result     |
| ------------------------------ | --------------------------------- | -------------------------- | ------------------------------------------------------------ | ---------- | --------------------------- | --------- | ------- | ---------- |
|                                |                                   |                            |                                                              |            |                             |           |         |            |
| <10B scale model               |                                   |                            |                                                              |            |                             |           |         |            |
| decapoda-research/llama-7b-hf  | 14GB                              | 1xNVIDIA-T4 / 16GB         | LLM.int8 (8-bit) + GC                                        | 1          | 4                           | AdamW     | 512     | **No OOM** |
| decapoda-research/llama-7b-hf  | 14GB                              | 1xNVIDIA-T4 / 16GB         | LLM.int8 (8-bit) + GC                                        | 1          | 4                           | AdamW     | 1024    | OOM        |
| decapoda-research/llama-7b-hf  | 14GB                              | 1xNVIDIA-T4 / 16GB         | 4bit + NF4 + bf16 CD + no GC                                 | 1          | 4                           | AdamW     | 512     | **No OOM** |
| decapoda-research/llama-7b-hf  | 14GB                              | 1xNVIDIA-T4 / 16GB         | 4bit + FP4 + bf16 CD + no GC                                 | 1          | 4                           | AdamW     | 512     | **No OOM** |
| decapoda-research/llama-7b-hf  | 14GB                              | 1xNVIDIA-T4 / 16GB         | 4bit + NF4 + bf16 CD + no GC                                 | 1          | 4                           | AdamW     | 1024    | OOM        |
| decapoda-research/llama-7b-hf  | 14GB                              | 1xNVIDIA-T4 / 16GB         | 4bit + FP4 + bf16 CD + no GC                                 | 1          | 4                           | AdamW     | 1024    | OOM        |
| decapoda-research/llama-7b-hf  | 14GB                              | 1xNVIDIA-T4 / 16GB         | 4bit + NF4 + bf16 CD + GC                                    | 1          | 4                           | AdamW     | 1024    | **No OOM** |
|                                |                                   |                            |                                                              |            |                             |           |         |            |
| 10B+ scale models              |                                   |                            |                                                              |            |                             |           |         |            |
| decapoda-research/llama-13b-hf | 27GB                              | 2xNVIDIA-T4 / 32GB         | LLM.int8 (8-bit) + GC                                        | 1          | 4                           | AdamW     | 512     | **No OOM** |
| decapoda-research/llama-13b-hf | 27GB                              | 1xNVIDIA-T4 / 16GB         | LLM.int8 (8-bit) + GC                                        | 1          | 4                           | AdamW     | 512     | OOM        |
| decapoda-research/llama-13b-hf | 27GB                              | 1xNVIDIA-T4 / 16GB         | 4bit + FP4 + bf16 CD + no GC                                 | 1          | 4                           | AdamW     | 512     | OOM        |
| decapoda-research/llama-13b-hf | 27GB                              | 1xNVIDIA-T4 / 16GB         | 4bit + FP4 + fp16 CD + no GC                                 | 1          | 4                           | AdamW     | 512     | OOM        |
| decapoda-research/llama-13b-hf | 27GB                              | 1xNVIDIA-T4 / 16GB         | 4bit + NF4 + fp16 CD + GC                                    | 1          | 4                           | AdamW     | 512     | **No OOM** |
| decapoda-research/llama-13b-hf | 27GB                              | 1xNVIDIA-T4 / 16GB         | 4bit + NF4 + fp16 CD + GC                                    | 1          | 4                           | AdamW     | 1024    | OOM        |
| decapoda-research/llama-13b-hf | 27GB                              | 1xNVIDIA-T4 / 16GB         | 4bit + NF4 + fp16 CD + GC + NQ                               | 1          | 4                           | AdamW     | 1024    | **No OOM** |

我们使用了最近的TRL 库，基准测试脚本可以[在这里](https://gist.github.com/younesbelkada/f48af54c74ba6a39a7ae4fd777e72fe8)`SFTTrainer`找到
