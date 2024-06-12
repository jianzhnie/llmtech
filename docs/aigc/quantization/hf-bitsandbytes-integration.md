# A Gentle Introduction to 8-bit Matrix Multiplication for transformers at scale

## 介绍

语言模型一直在变大。在撰写本文时，PaLM 有 540B 个参数，OPT、GPT-3 和 BLOOM 有大约 176B 个参数，而且我们正朝着更大的模型发展。下图显示了一些最近的语言模型的大小。

![LLM](https://huggingface.co/blog/assets/96_hf_bitsandbytes_integration/LLM3.png)

因此，这些模型很难在易于访问的设备上运行。例如，仅在 BLOOM-176B 上进行推理，您就需要 8 个 80GB A100 GPU（每个约 15,000 美元）。要微调 BLOOM-176B，您需要 72 个这样的 GPU！更大的模型，如 PaLM，将需要更多的资源。

由于这些庞大的模型需要大量 GPU 才能运行，因此我们需要找到降低这些要求同时保持模型性能的方法。已经开发出各种试图缩小模型大小的技术，您可能听说过量化和蒸馏，还有很多其他技术。

完成 BLOOM-176B 的训练后，HuggingFace 和 BigScience 正在寻找方法让这个大模型更容易在更少的 GPU 上运行。通过这篇博文，我们为所有 Hugging Face 模型提供了 LLM.int8() 集成，我们将在下面进行更详细的解释。如果您想了解有关我们研究的更多信息，可以阅读我们的论文，[LLM.int8()：大规模Transformer 模型的 8 位矩阵乘法](https://arxiv.org/abs/2208.07339)。

本文着重于对这种量化技术进行高级概述，概述将其纳入库中的困难`transformers`，并制定这种合作关系的长期目标。

在这里，您将了解究竟是什么让一个大型模型使用这么多内存？是什么让 BLOOM 350GB 成为可能？让我们从逐步了解一些基本前提开始。

## 机器学习中使用的常见数据类型

我们从对不同浮点数据类型的基本理解开始，这些数据类型在机器学习的上下文中也被称为“精度”。

模型的大小由其参数的数量及其精度决定，通常是 float32、float16 或 bfloat16 之一（下图来自：https://blogs.nvidia.com/blog/2020/05/14/tensorfloat-32-precision-format/).。

![概括](https://huggingface.co/blog/assets/96_hf_bitsandbytes_integration/tf32-Mantissa-chart-hi-res-FINAL.png)

Float32 (FP32) 代表标准化的 IEEE 32 位浮点表示法。使用这种数据类型，可以表示范围广泛的浮点数。在 FP32 中，为“指数”保留了 8 位，为“尾数”保留了 23 位，为数字的符号保留了 1 位。除此之外，大部分硬件都支持FP32运算和指令。

在 float16 (FP16) 数据类型中，指数保留 5 位，尾数保留 10 位。这使得 FP16 数字的可表示范围远低于 FP32。这使 FP16 数字面临溢出（试图表示非常大的数字）和下溢（表示非常小的数字）的风险。

例如，如果你这样做，`10k * 10k`你最终得到的`100M` ， 这在 FP16 中是不可能表示的，因为可能的最大数字是`64k`. 因此你最终会得到`NaN`（不是数字）结果，如果你像在神经网络中那样进行顺序计算，那么所有先前的工作都会被破坏。通常，损失缩放用于克服这个问题，但并不总是有效。

创建了一种新格式 bfloat16 (BF16) 来避免这些限制。在 BF16 中，为指数保留了 8 位（与 FP32 相同），为小数保留了 7 位。

这意味着在 BF16 中我们可以保留与 FP32 相同的动态范围。但是对于 FP16，我们损失了 3 位精度。现在巨大的数字绝对没有问题，但是这里的精度比FP16差。

在Ampere架构中，NVIDIA还引入了[TensorFloat-32](https://blogs.nvidia.com/blog/2020/05/14/tensorfloat-32-precision-format/)（TF32）精度格式，结合了BF16的动态范围和FP16的精度，只用了19位。

在机器学习术语中，FP32 称为全精度（4 字节），而 BF16 和 FP16 称为半精度（2 字节）。最重要的是，int8 (INT8) 数据类型由一个 8 位表示组成，可以存储 2^8 个不同的值（对于有符号整数，介于 [0, 255] 或 [-128, 127] 之间）。

虽然理想情况下训练和推理应该在 FP32 中完成，但它比 FP16/BF16 慢两倍，因此使用混合精度方法，其中权重在 FP32 中作为精确的“主要权重”参考，而计算在对 FP16/BF16 进行前向和后向传递以提高训练速度。然后使用 FP16/BF16 梯度更新 FP32 主权重。

在训练期间，主要权重始终存储在 FP32 中，但在实践中，半精度权重通常在推理过程中提供与其 FP32 对应物相似的质量——只有在模型接收到多个梯度更新时才需要模型的精确参考。这意味着我们可以使用半精度权重并使用一半的 GPU 来实现相同的结果。

![模型存储](https://huggingface.co/blog/assets/96_hf_bitsandbytes_integration/Model-storage.png)

要以字节为单位计算模型大小，需要将参数数量乘以所选精度的大小（以字节为单位）。例如，如果我们使用 BLOOM-176B 模型的 bfloat16 版本，我们有`176*10**9 x 2 bytes = 352GB`！如前所述，要把这个模型放到很少的几个 GPU 上是一个相当大的挑战。

但是，如果我们可以使用不同的数据类型以更少的内存存储这些权重呢？一种称为量化的方法已广泛用于深度学习。

## 模型量化介绍

通过实验，我们发现不使用 4 字节 FP32 精度，我们可以使用 2 字节 BF16/FP16 半精度获得几乎相同的推理结果，这将模型大小减半。进一步削减它会很惊人，但推理质量结果开始在较低的精度下急剧下降。

为了解决这个问题，我们引入了 8 位量化。此方法使用四分之一精度，因此只需要模型大小的 1/4！但这不是通过仅丢弃另一半Bit来完成的。

量化基本上是通过从一种数据类型“舍入”到另一种数据类型来完成的。例如，如果一种数据类型的范围为 0..9，而另一种数据类型的范围为 0..4，则第一种数据类型中的值“4”将舍入为第二种数据类型中的“2”。但是，如果我们在第一种数据类型中有值“3”，它介于第二种数据类型的 1 和 2 之间，那么我们通常会四舍五入为“2”。这表明第一种数据类型的值“4”和“3”在第二种数据类型中具有相同的值“2”。这突出表明量化是一个嘈杂的过程，会导致信息丢失，这是一种有损压缩。

两种最常见的 8 位量化技术是零点量化和绝对最大 (absmax) 量化。零点量化和 absmax 量化将浮点值映射为更紧凑的 int8（1 字节）值。首先，这些方法通过量化常数缩放输入来归一化输入。

### 零点量化

例如，在零点量化中，如果我的范围是-1.0…1.0，我想量化到-127…127，我想缩放127倍，然后四舍五入到8位精度。要检索原始值，您需要将 int8 值除以相同的量化因子 127。例如，值 0.3 将缩放为`0.3*127 = 38.1`. 通过四舍五入，我们得到 38 的值。如果我们反转它，我们得到`38/127=0.2992`——在这个例子中我们有一个 0.008 的量化误差。这些看似微小的错误在通过模型层传播时往往会累积和增长，从而导致性能下降。

[![量化](https://huggingface.co/blog/assets/96_hf_bitsandbytes_integration/quantization.png)](https://huggingface.co/blog/assets/96_hf_bitsandbytes_integration/quantization.png)

（图片取自：[这篇博文](https://intellabs.github.io/distiller/algo_quantization.html)）

现在让我们看看 absmax 量化的细节。要计算 absmax 量化中 fp16 数与其对应的 int8 数之间的映射，您必须先除以张量的绝对最大值，然后再乘以数据类型的总范围。

### absmax 量化

例如，假设您要在包含 的向量中应用 absmax 量化`[1.2, -0.5, -4.3, 1.2, -3.1, 0.8, 2.4, 5.4]`。您提取它的绝对最大值，在这种情况下是`5.4`。Int8 的范围为`[-127, 127]`，因此我们将 127 除以`5.4`并获得`23.5`比例因子。因此，将原始向量乘以它得到量化向量`[28, -12, -101, 28, -73, 19, 56, 127]`。

[![出量.gif](https://huggingface.co/blog/assets/96_hf_bitsandbytes_integration/out-quant.gif)](https://huggingface.co/blog/assets/96_hf_bitsandbytes_integration/out-quant.gif)

要检索最新的，可以将 int8 数字完全精确地除以量化因子，但由于上面的结果是“四舍五入”的，一些精度将会丢失。

[![量化冻结](https://huggingface.co/blog/assets/96_hf_bitsandbytes_integration/quant-freeze.png)](https://huggingface.co/blog/assets/96_hf_bitsandbytes_integration/quant-freeze.png)

对于 unsigned int8，我们将减去最小值并按绝对最大值进行缩放。这接近于零点量化的作用。它类似于最小-最大缩放，但后者以这样一种方式维护值缩放，即值“0”始终由没有任何量化误差的整数表示。

当涉及矩阵乘法以获得更准确的结果时，可以通过多种方式组合这些技巧，例如，逐行或逐向量量化。查看矩阵乘法，A B=C，而不是通过每个张量的绝对最大值归一化的常规量化，向量量化找到 A 的每一行和 B 的每一列的绝对最大值。然后我们归一化 A 和B 通过划分这些向量。然后我们乘以 A B 得到 C。最后，为了得到 FP16 值，我们通过计算 A 和 B 的绝对最大向量的外积来反规范化。关于这种技术的更多细节可以在 LLM.int8 [( ) 论文](https://arxiv.org/abs/2208.07339)或Tim 博客上[关于量化和涌现特征的博客文章。](https://timdettmers.com/2022/08/17/llm-int8-and-emergent-features/)

虽然这些基本技术使我们能够量化深度学习模型，但它们通常会导致较大模型的准确性下降。我们集成到 Hugging Face Transformers 和 Accelerate 库中的 LLM.int8() 实现是第一种即使对于具有 176B 参数的大型模型（例如 BLOOM）也不会降低性能的技术。

## LLM.int8() 的总结：大型语言模型的零退化矩阵乘法

在 LLM.int8() 中，我们已经证明理解Transformer 模型的尺度相关涌现特性对于理解为什么传统量化对大型模型失败至关重要。我们证明性能下降是由离群特征引起的，我们将在下一节中解释。LLM.int8() 算法本身可以解释如下。

本质上，LLM.int8() 寻求通过三个步骤完成矩阵乘法计算：

1. 从输入的隐藏状态中，按列提取异常值（即大于某个阈值的值）。
2. 执行 FP16 中异常值和 int8 中非异常值的矩阵乘法。
3. 对非离群值结果进行反量化，并将离群值和非离群值结果相加，以在 FP16 中获得完整结果。

这些步骤可以总结为以下动画：

[![混合int8.gif](https://huggingface.co/blog/assets/96_hf_bitsandbytes_integration/Mixed-int8.gif)](https://huggingface.co/blog/assets/96_hf_bitsandbytes_integration/Mixed-int8.gif)

### 离群特征的重要性

超出某些数字的全局分布范围的值通常称为异常值。离群值检测已被广泛使用并涵盖在当前文献中，并且事先了解特征的分布有助于完成离群值检测任务。更具体地说，我们观察到对于基于 Transformer 的模型 >6B 参数，经典的大规模量化失败。虽然较大的异常值特征也存在于较小的模型中，但我们观察到这些异常值来自Transformer 模型的每一层中存在的跨Transformer 模型的高度系统化模式。有关这些现象的更多详细信息，请参阅[LLM.int8() 论文](https://arxiv.org/abs/2208.07339)和[emergent features blog post](https://timdettmers.com/2022/08/17/llm-int8-and-emergent-features/).

如前所述，8 位精度受到极大限制，因此量化具有多个大值的向量会产生严重错误的结果。此外，由于将所有元素链接在一起的基于Transformer 模型的架构的内置特性，这些错误在跨多个层传播时往往会混合在一起。因此，已经开发了混合精度分解以促进对此类极端异常值进行有效量化。接下来讨论。

###  内部的 MatMul

一旦计算出隐藏状态，我们就用一个自定义的阈值来提取异常值，并按照上面的解释将矩阵分解成两部分。我们发现，以这种方式提取所有幅度为6或更大的离群值，可以恢复全部推理性能。异常值部分是在fp16中完成的，所以它是一个经典的矩阵乘法，而8位矩阵乘法是通过使用矢量量化将权重和隐藏状态量化为8位精度来完成的--也就是说，隐藏状态的行量化和权重矩阵的列量化。在这一步之后，结果被去量化并以半精度返回，以便将其加入到第一个矩阵乘法中。

[![Matmul.png](https://huggingface.co/blog/assets/96_hf_bitsandbytes_integration/Matmul.png)](https://huggingface.co/blog/assets/96_hf_bitsandbytes_integration/Matmul.png)

### 0退化是什么意思？

我们怎样才能正确评估这种方法的性能下降？在使用8位模型时，我们在生成方面会损失多少质量？

我们使用lm-eval-harness对8位和原生模型进行了几个常见的基准测试，并报告了结果。

对于 OPT-175B：

| benchmarks | -        | -            | -            | -              | difference - value |
| ---------- | -------- | ------------ | ------------ | -------------- | ------------------ |
| name       | metric   | value - int8 | value - fp16 | std err - fp16 | -                  |
| hellaswag  | acc_norm | 0.7849       | 0.7849       | 0.0041         | 0                  |
| hellaswag  | acc      | 0.5921       | 0.5931       | 0.0049         | 0.001              |
| piqa       | acc      | 0.7965       | 0.7959       | 0.0094         | 0.0006             |
| piqa       | acc_norm | 0.8101       | 0.8107       | 0.0091         | 0.0006             |
| lambada    | ppl      | 3.0142       | 3.0152       | 0.0552         | 0.001              |
| lambada    | acc      | 0.7464       | 0.7466       | 0.0061         | 0.0002             |
| winogrande | acc      | 0.7174       | 0.7245       | 0.0125         | 0.0071             |

对于 BLOOM-176：

| benchmarks | -        | -            | -            | -              | difference - value |
| ---------- | -------- | ------------ | ------------ | -------------- | ------------------ |
| name       | metric   | value - int8 | value - bf16 | std err - bf16 | -                  |
| hellaswag  | acc_norm | 0.7274       | 0.7303       | 0.0044         | 0.0029             |
| hellaswag  | acc      | 0.5563       | 0.5584       | 0.005          | 0.0021             |
| piqa       | acc      | 0.7835       | 0.7884       | 0.0095         | 0.0049             |
| piqa       | acc_norm | 0.7922       | 0.7911       | 0.0095         | 0.0011             |
| lambada    | ppl      | 3.9191       | 3.931        | 0.0846         | 0.0119             |
| lambada    | acc      | 0.6808       | 0.6718       | 0.0065         | 0.009              |
| winogrande | acc      | 0.7048       | 0.7048       | 0.0128         | 0                  |

我们确实观察到这些模型的性能下降为0，因为指标的绝对差异都低于标准误差（除了BLOOM-int8，它比lambada上的本地模型略好）。关于对最先进的方法的更详细的性能评估，请看这篇[论文](https://arxiv.org/abs/2208.07339)！

### 它比本地模型快吗？

LLM.int8()方法的主要目的是在不降低性能的情况下使大型模型更容易被访问。但是如果该方法非常慢的话，其作用就不大了。所以我们对多个模型的生成速度进行了基准测试。我们发现，使用LLM.int8()的BLOOM-176B比fp16版本慢了大约15%到23%--这还是相当可以接受的。我们发现较小的模型，如T5-3B和T5-11B，速度更慢。我们努力工作以加快这些小模型的速度。在一天之内，我们可以将T5-3B的每个标记的推理速度从312毫秒提高到173毫秒，T5-11B的推理速度从45毫秒提高到25毫秒。此外，已经发现了一些[问题](https://github.com/TimDettmers/bitsandbytes/issues/6#issuecomment-1211345635)，在即将发布的版本中，LLM.int8()对小模型来说可能还会更快。现在，目前的数字在下面的表格中。

| Precision | Number of parameters | Hardware     | Time per token in milliseconds for Batch Size 1 | Time per token in milliseconds for Batch Size 8 | Time per token in milliseconds for Batch Size 32 |
| --------- | -------------------- | ------------ | ----------------------------------------------- | ----------------------------------------------- | ------------------------------------------------ |
| bf16      | 176B                 | 8xA100 80GB  | 239                                             | 32                                              | 9.9                                              |
| int8      | 176B                 | 4xA100 80GB  | 282                                             | 37.5                                            | 10.2                                             |
| bf16      | 176B                 | 14xA100 40GB | 285                                             | 36.5                                            | 10.4                                             |
| int8      | 176B                 | 5xA100 40GB  | 367                                             | 46.4                                            | oom                                              |
| fp16      | 11B                  | 2xT4 15GB    | 11.7                                            | 1.7                                             | 0.5                                              |
| int8      | 11B                  | 1xT4 15GB    | 43.5                                            | 5.3                                             | 1.3                                              |
| fp32      | 3B                   | 2xT4 15GB    | 45                                              | 7.2                                             | 3.1                                              |
| int8      | 3B                   | 1xT4 15GB    | 312                                             | 39.1                                            | 10.2                                             |

3个型号分别是BLOOM-176B、T5-11B和T5-3B。

### Huggingface `transformers`集成的细微差别

接下来让我们讨论 Hugging Face 集成的细节`transformers`。让我们看看您在尝试设置时可能遇到的用法和常见罪魁祸首。

### 用法

负责本博文中描述的整个魔法的模块被称为 Linear8bitLt，你可以很容易地从 bitsandbytes 库中导入它。它源自一个经典的 torch.nn 模块，可以通过下面描述的代码轻松地在你的架构中使用和部署。

下面是一个关于以下用例的步骤：假设你想用bitsandbytes转换一个int8的小模型。

1. 首先我们需要下面正确的导入！

```py
import torch
import torch.nn as nn

import bitsandbytes as bnb
from bnb.nn import Linear8bitLt
```

1. 然后你可以定义你自己的模型。注意，你可以将任何精度的检查点或模型转换为8位（FP16、BF16或FP32），但目前，模型的输入必须是FP16，我们的Int8模块才能工作。所以我们在这里把我们的模型当作FP16模型。

```py
fp16_model = nn.Sequential(
    nn.Linear(64, 64),
    nn.Linear(64, 64)
)
```

1. 假设您已经在您最喜欢的数据集和任务上训练了您的模型！现在是时候保存模型了：

```py
[... train the model ...]
torch.save(fp16_model.state_dict(), "model.pt")
```

1. 现在你的state_dict已经被保存，让我们定义一个int8模型：

```py
int8_model = nn.Sequential(
    Linear8bitLt(64, 64, has_fp16_weights=False),
    Linear8bitLt(64, 64, has_fp16_weights=False)
)
```

在这里，添加标志has_fp16_weights是非常重要的。默认情况下，它被设置为True，用于在Int8/FP16混合精度下进行训练。然而，我们对内存效率的推理感兴趣，为此我们需要使用has_fp16_weights=False。

1. 现在是时候加载您的 8 位模型了！

```py
int8_model.load_state_dict(torch.load("model.pt"))
int8_model = int8_model.to(0) # Quantization happens here
```

请注意，一旦在GPU上设置了模型，量化的步骤就在第二行完成。如果你在调用.to函数之前打印int8_model[0].weight，你会得到：

```python
int8_model[0].weight
Parameter containing:
tensor([[ 0.0031, -0.0438,  0.0494,  ..., -0.0046, -0.0410,  0.0436],
        [-0.1013,  0.0394,  0.0787,  ...,  0.0986,  0.0595,  0.0162],
        [-0.0859, -0.1227, -0.1209,  ...,  0.1158,  0.0186, -0.0530],
        ...,
        [ 0.0804,  0.0725,  0.0638,  ..., -0.0487, -0.0524, -0.1076],
        [-0.0200, -0.0406,  0.0663,  ...,  0.0123,  0.0551, -0.0121],
        [-0.0041,  0.0865, -0.0013,  ..., -0.0427, -0.0764,  0.1189]],
       dtype=torch.float16)
```

而如果你在第二行的调用之后打印它，你会得到：

```python
int8_model[0].weight
Parameter containing:
tensor([[   3,  -47,   54,  ...,   -5,  -44,   47],
        [-104,   40,   81,  ...,  101,   61,   17],
        [ -89, -127, -125,  ...,  120,   19,  -55],
        ...,
        [  82,   74,   65,  ...,  -49,  -53, -109],
        [ -21,  -42,   68,  ...,   13,   57,  -12],
        [  -4,   88,   -1,  ...,  -43,  -78,  121]],
        device='cuda:0', dtype=torch.int8, requires_grad=True)
```

正如我们在前几节解释量化时看到的那样，权重值是 "截断的"。另外，这些值似乎分布在[-127, 127]之间。你可能也想知道如何检索FP16的权重，以便在FP16中执行离群的MatMul？你可以简单地做：

```py
(int8_model[0].weight.CB * int8_model[0].weight.SCB) / 127
```

你会得到：

```python
tensor([[ 0.0028, -0.0459,  0.0522,  ..., -0.0049, -0.0428,  0.0462],
        [-0.0960,  0.0391,  0.0782,  ...,  0.0994,  0.0593,  0.0167],
        [-0.0822, -0.1240, -0.1207,  ...,  0.1181,  0.0185, -0.0541],
        ...,
        [ 0.0757,  0.0723,  0.0628,  ..., -0.0482, -0.0516, -0.1072],
        [-0.0194, -0.0410,  0.0657,  ...,  0.0128,  0.0554, -0.0118],
        [-0.0037,  0.0859, -0.0010,  ..., -0.0423, -0.0759,  0.1190]],
       device='cuda:0')
```

这足够接近原始 FP16 值（2 个打印输出）！

1. 现在您可以通过确保您的输入在正确的 GPU 上并且在 FP16 中来安全地推断使用您的模型：

```py
input_ = torch.randn(64, dtype=torch.float16)
hidden_states = int8_model(input_.to(torch.device('cuda', 0)))
```

查看[示例脚本](https://huggingface.co/assets/96_hf_bitsandbytes_integration/example.py)以获取完整的最小代码！

顺便提一下，你应该知道这些模块与nn.Linear模块略有不同，它们的参数来自bnb.nn.Int8Params类而不是nn.Parameter类。稍后你会看到，这给我们的旅程带来了额外的障碍!

现在是了解如何将其集成到`transformers`库中的时候了！

### `accelerate`是你所需要的全部

在处理大型模型时，该`accelerate`库包含许多有用的实用程序。该`init_empty_weights`方法特别有用，因为任何模型，无论大小，都可以使用此方法作为上下文管理器进行初始化，而无需为模型权重分配任何内存。

```py
import torch.nn as nn
from accelerate import init_empty_weights

with init_empty_weights():
    model = nn.Sequential([nn.Linear(100000, 100000) for _ in range(1000)]) # This will take ~0 RAM!
```

初始化的模型将被放在PyTorch的元设备上，这是一种表示形状和dtype的底层机制，无需分配内存进行存储。这有多酷啊？

最初，这个函数是在.from_pretrained函数里面调用的，并将所有参数重写为torch.nn.Parameter。这不符合我们的要求，因为我们想在Linear8bitLt模块中保留Int8Params类，如上所述。我们在下面的PR中设法解决了这个问题，修改了：

```py
module._parameters[name] = nn.Parameter(module._parameters[name].to(torch.device("meta")))
```

到

```py
param_cls = type(module._parameters[name])
kwargs = module._parameters[name].__dict__
module._parameters[name] = param_cls(module._parameters[name].to(torch.device("meta")), **kwargs)
```

现在这是固定的，我们可以轻松地利用这个上下文管理器并使用它来替换所有`nn.Linear`模块，`bnb.nn.Linear8bitLt`使用自定义函数无需内存成本！

```py
def replace_8bit_linear(model, threshold=6.0, module_to_not_convert="lm_head"):
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_8bit_linear(module, threshold, module_to_not_convert)

        if isinstance(module, nn.Linear) and name != module_to_not_convert:
            with init_empty_weights():
                model._modules[name] = bnb.nn.Linear8bitLt(
                    module.in_features,
                    module.out_features,
                    module.bias is not None,
                    has_fp16_weights=False,
                    threshold=threshold,
                )
    return model
```

这个函数递归地替换了元设备上初始化的一个给定模型的所有nn.Linear层，并用一个Linear8bitLt模块替换它们。属性has_fp16_weights必须设置为False，以便直接加载int8中的权重和量化统计。

我们也放弃了对一些模块（这里是lm_head）的替换，因为我们希望保持最新的原始精度，以获得更精确和稳定的结果。

但这还没有结束!上面的函数是在init_empty_weights上下文管理器下执行的，这意味着新模型将仍然在元设备中。对于在这个上下文管理器下初始化的模型，加速器将手动加载每个模块的参数并将它们移到正确的设备中。在bitsandbytes中，设置Linear8bitLt模块的设备是一个关键步骤（如果你很好奇，你可以查看这里的代码片段），正如我们在玩具脚本中看到的。

这里的量化步骤在调用两次时失败了。我们不得不想出一个加速器的set_module_tensor_to_device函数（称为set_module_8bit_tensor_to_device）的实现，以确保我们不会调用它两次。让我们在下面的章节中详细讨论这个问题!

### 非常小心如何设置设备`accelerate`

在这里，我们与图书馆进行了非常微妙的平衡`accelerate`！一旦加载模型并将其设置在正确的设备上，有时您仍然需要调用以`set_module_tensor_to_device`在所有设备上使用挂钩调度模型。这是在`dispatch_model`from 函数内部完成的，它可能涉及多次`accelerate`调用，这是我们想要避免的事情。`.to`需要 2 个 Pull Requests 来实现我们想要的！[此处](https://github.com/huggingface/accelerate/pull/539/)提出的初始 PR破坏了一些测试，但[此 PR](https://github.com/huggingface/accelerate/pull/576/)成功修复了所有问题！

### 把它包起来

因此最终的配方是：

1. `meta`使用正确的模块初始化设备中的模型
2. 在正确的 GPU 设备上一一设置参数，并确保您永远不会重复此过程！
3. 将新的关键字参数放在正确的位置，并添加一些不错的文档
4. 添加非常广泛的测试！[在此处](https://github.com/huggingface/transformers/blob/main/tests/mixed_int8/test_mixed_int8.py)查看我们的测试以获取更多详细信息这听起来很简单，但我们一起经历了许多艰难的调试过程，通常涉及 CUDA 内核！

总而言之，这次整合冒险非常有趣；从深入研究和对不同的图书馆做一些“手术”到调整一切并使其发挥作用！

现在是时候看看如何从这种集成中获益以及如何在`transformers`!

## 如何在`transformers`使用它

### 硬件要求

CPU 不支持 8 位张量核心。bitsandbytes 可以在支持 8 位张量核心的硬件上运行，这些硬件是 Turing 和 Ampere GPU（RTX 20s、RTX 30s、A40-A100、T4+）。例如，Google Colab GPU 通常是 NVIDIA T4 GPU，他们最新一代的 GPU 确实支持 8 位张量核心。我们的演示基于 Google Colab，请在下方查看！

### 安装

只需使用以下命令安装最新版本的库（确保您使用的是 python>=3.8）并运行以下命令进行试用

```bash
pip install accelerate
pip install bitsandbytes
pip install git+https://github.com/huggingface/transformers.git
```

### 示例演示 - 在 Google Colab 上运行 T5 11b

查看在 BLOOM-3B 模型上运行 8 位模型的 Google Colab 演示！

这是运行 T5-11B 的演示。T5-11B 模型检查点在 FP32 中，它使用 42GB 内存并且不适合 Google Colab。使用我们的 8 位模块，它仅使用 11GB 并且很容易安装：

[![在 Colab 中打开：T5-11b 演示](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1YORPWx4okIHXnjW7MSAidXN29mPVNT7F?usp=sharing)

或者这个 BLOOM-3B 的演示：

[![在 Colab 中打开：BLOOM-3b 演示](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/blog/blob/main/notebooks/HuggingFace_int8_demo.ipynb)

## 改进范围

我们认为，这种方法极大地改善了对超大型模型的访问。在不降低性能的情况下，它使计算量较少的用户能够访问以前无法访问的模型。我们发现了几个可以在未来进行改进的领域，以使这种方法对大型模型更好！

### 较小模型的推理速度更快

[正如我们在基准测试部分](https://huggingface.co/blog/hf-bitsandbytes-integration#is-it-faster-than-native-models?)看到的那样，我们可以将小型模型（<=6B 参数）的运行速度提高近 2 倍。然而，虽然推理速度对于像 BLOOM-176B 这样的大型模型来说是稳健的，但对于小型模型仍有改进的余地。我们已经确定了问题并可能恢复与 fp16 相同的性能，或者获得小幅加速。您将在接下来的几周内看到这些更改被整合。

### 支持开普勒 GPU（GTX 1080 等）

虽然我们支持过去四年的所有 GPU，但一些旧 GPU（如 GTX 1080）仍然被大量使用。虽然这些 GPU 没有 Int8 张量核心，但它们有 Int8 向量单元（一种“弱”张量核心）。因此，这些 GPU 也可以体验 Int8 加速。然而，它需要一个完全不同的软件堆栈来进行快速推理。虽然我们确实计划集成对 Kepler GPU 的支持以使 LLM.int8() 功能更广泛地可用，但由于其复杂性，实现这一点需要一些时间。

### 在集线器上保存 8 位状态指令

8 位状态指令在被推送到集线器上后目前无法直接加载到 8 位模型中。这是因为模型计算的统计数据（remember`weight.CB`和`weight.SCB`）当前未存储或考虑在状态字典中，并且该`Linear8bitLt`模块尚不支持此功能。我们认为能够保存它并将其推送到 Hub 可能有助于提高可访问性。

### 处理器支持

正如本文开头所述，CPU 设备不支持 8 位内核。然而，我们能克服它吗？在 CPU 上运行此模块还将显着提高可用性和可访问性。

### 扩大其他模式

目前，语言模型主导着非常大的模型。在非常大的视觉、音频和多模式模型上利用这种方法可能是一件有趣的事情，因为随着这些模型变得更易于访问，未来几年可以更好地访问这些模型。

## 学分

非常感谢以下为提高文章的可读性以及在集成过程中做出贡献的人`transformers`（按字母顺序列出）：JustHeuristic (Yozh)、Michael Benayoun、Stas Bekman、Steven Liu、Sylvain Gugger、Tim Dettmers
