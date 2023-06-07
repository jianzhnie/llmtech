[Lit-LLaMA 简介：在 Apache 2.0 下许可的 LLaMA 的最小优化重写 →](https://github.com/Lightning-AI/lit-llama)

[![带有文本的 Lightning.AI 徽标](https://lightning.ai/static/media/logo-with-text.2351c373b819a1dafdff787e15c32bbe.svg)](https://lightning.ai/)产品社区[文档](https://lightning.ai/docs)[发布](https://lightning.ai/pages/releases)[价钱](https://lightning.ai/pricing)

登录免费开始

# 具有低秩自适应 (LoRA) 的参数高效 LLM 微调

发表于 2023 年 4 月 26 日，作者：[Sebastian Raschka](https://lightning.ai/pages/author/sebastian-raschka/) -[文章](https://lightning.ai/pages/category/community/article/)、[教程](https://lightning.ai/pages/category/community/tutorial/)

### 关键要点

在快速发展的人工智能领域，以高效和有效的方式使用大型语言模型变得越来越重要。在本文中，您将学习如何以计算高效的方式使用低秩自适应 (LoRA) 调整 LLM！

 

## **为什么微调？**

预训练的大型语言模型通常被称为基础模型是有充分理由的：它们在各种任务上表现良好，我们可以将它们用作对目标任务进行微调的基础。正如我们在上一篇文章（[了解大型语言模型的参数高效微调：从前缀调优到 LLaMA-适配器](https://lightning.ai/pages/community/article/understanding-llama-adapters/)）中所讨论的那样，我们讨论了微调允许我们使模型适应目标域和目标任务。不过，它的计算成本可能非常高——模型越大，更新其层的成本就越高。

作为更新所有层的替代方法，已经开发了参数有效的方法，例如前缀调整和适配器——有关详细审查，请参阅我们[之前的帖子](https://lightning.ai/pages/community/article/understanding-llama-adapters/)。现在，有一种更流行的参数高效微调技术：[Hu 等人的低秩自适应 (LoRA)](https://arxiv.org/abs/2106.09685)。什么是 LoRA？它是如何工作的？它与其他流行的微调方法相比如何？让我们在本文中回答所有这些问题！

 

![PCA 变换](https://lightningaidev.wpengine.com/wp-content/uploads/2023/04/lora-1.jpg)

 

 

## **使权重更新更高效**

基于上述想法，论文 [LoRA：大型语言模型的低秩适应](https://arxiv.org/abs/2106.09685) 建议将权重变化 *ΔW*分解为低秩表示。（从技术上讲，LoRA 不直接分解矩阵，而是通过反向传播学习分解后的矩阵——这是一个挑剔的细节，稍后会有意义）。

在深入了解 LoRA 之前，让我们先简单介绍一下常规微调期间的训练过程。那么，重量变化 *ΔW*是多少？假设 *W* 表示给定神经网络层中的权重矩阵。然后，使用常规反向传播，我们可以获得权重更新 *ΔW*，它通常计算为损失乘以学习率的负梯度：

*ΔW* = *α* ( -∇ L W )。

然后，当我们有*ΔW*时，我们可以按如下方式更新原始权重：*W* ' = *W* + *ΔW*。下图对此进行了说明（为简单起见，省略了偏置向量）：

或者，我们可以将权重更新矩阵分开并按如下方式计算输出：*h = W x + ΔW x*，

 

![常规反向传播](https://lightningaidev.wpengine.com/wp-content/uploads/2023/04/lora-2.png)

 

其中x代表输入，如下图所示。

 

![img](https://lightningaidev.wpengine.com/wp-content/uploads/2023/04/lora-3.png)

 

我们为什么要这样做？目前，这个替代公式服务于说明 LoRA 的教学目标，但我们会回到它。

因此，当我们在神经网络中训练完全连接（即“密集”）层时，如上所示，权重矩阵通常具有满秩，这是一个技术术语，意思是矩阵没有任何线性相关（即， “冗余”）行或列。相反，对于满秩，低秩意味着矩阵具有冗余行或列。

因此，根据 Aghajanyan 等人的说法，虽然预训练模型的权重在预训练任务上具有完整排名，但 LoRA 作者指出，预训练大型语言模型在适应新任务时具有较低的“内在维度” [。](https://arxiv.org/abs/2012.13255) （2020）。

低内在维度意味着数据可以通过低维空间有效地表示或近似，同时保留其大部分基本信息或结构。换句话说，这意味着我们可以将适应任务的新权重矩阵分解为低维（更小）的矩阵，而不会丢失太多重要信息。

例如，假设*ΔW是**A × B*权重矩阵的权重更新。然后，我们可以将权重更新矩阵分解为两个更小的矩阵：*ΔW = W A W B*，其中*W A*是一个*A × r*维矩阵，*W B*是一个*r × B*维矩阵。在这里，我们保持原始权重*W*不变，只训练新矩阵*W A*和*W B*。简而言之，这就是 LoRA 方法，如下图所示。

 

 

![img](https://lightningaidev.wpengine.com/wp-content/uploads/2023/04/lora-4.png)

 

 
**选择等级**

请注意，上图中的*r是此处的超参数，我们可以使用它来指定用于自适应的低秩矩阵的秩。*较小的*r*会导致更简单的低秩矩阵，从而导致在适应过程中需要学习的参数更少。这可以导致更快的训练并可能减少计算需求。*然而，随着r*越小，低秩矩阵捕获任务特定信息的能力会降低。*这可能会导致较低的适应质量，并且与较高的r*相比，模型在新任务上的表现可能不那么好。总之，选择较小的*r*在 LoRA 中，模型复杂性、适应能力和欠拟合或过拟合风险之间存在权衡。因此，重要的是尝试不同的*r*值以找到正确的平衡以在新任务上实现所需的性能。

 
**实施 LoRA**

LoRA 的实施相对简单。我们可以将其视为 LLM 中全连接层的修改前向传递。在伪代码中，这看起来如下所示：

```python
input_dim = 768  # e.g., the hidden size of the pre-trained model
output_dim = 768  # e.g., the output size of the layer
rank = 8  # The rank 'r' for the low-rank adaptation

W = ... # from pretrained network with shape input_dim x output_dim

W_A = nn.Parameter(torch.empty(input_dim, rank)) # LoRA weight A
W_B = nn.Parameter(torch.empty(rank, output_dim)) # LoRA weight B

# Initialization of LoRA weights
nn.init.kaiming_uniform_(W_A, a=math.sqrt(5))
nn.init.zeros_(W_B)

def regular_forward_matmul(x, W):
    h = x @ W
return h

def lora_forward_matmul(x, W, W_A, W_B):
    h = x @ W  # regular matrix multiplication
    h += x @ (W_A @ W_B)*alpha # use scaled LoRA weights
return h

复制
```

 

在上面的伪代码中，`alpha`是一个缩放因子，用于调整组合结果（原始模型输出加上低秩自适应）的大小。这平衡了预训练模型的知识和新的特定于任务的适应——默认情况下，`alpha`通常设置为 1。另请注意，虽然*W A*被初始化为小的随机权重，但*W B*被初始化为 0，因此

训练开始时*ΔW = W* *A* *W* *B* *= 0 ，这意味着我们以原始权重开始训练。*

 
**参数效率**

现在，让我们解决房间里的大问题：如果我们引入新的权重矩阵，这个参数的效率如何？新矩阵*W A*和*W B*可以非常小。例如，假设*A=100*且*B=500 ，则**ΔW*的大小为*100 × 500 = 50,000*。现在，如果我们将其分解为两个较小的矩阵，一个*100×5*维矩阵*W A*和一个*5×500*维矩阵*W B*。这两个矩阵总共只有*5×100+5×500=3000个*参数。

 
**减少推理开销**

请注意，在实践中，如果我们在训练后保持原始权重*W*和矩阵*W A*和*W B分开，如上所示，我们将在推理过程中产生小的效率损失，因为这引入了额外的计算步骤。**相反，我们可以在训练后通过W' = W + W A W B*更新权重，这类似于前面提到的*W' = W + ΔW*。

*然而，将权重矩阵W A*和*W B*分开可能具有实际优势。例如，假设我们希望将我们的预训练模型作为各种客户的基础模型，并且我们希望从基础模型开始为每个客户创建一个经过微调的 LLM。在这种情况下，我们不需要为每个客户存储完整的权重矩阵*W'*，其中存储模型的所有权重*W' = W + W A W B*对于 LLM 来说可能非常大，因为 LLM 通常有数十亿到数万亿个权重参数。因此，我们可以保留原始模型*W*，只需要存储新的轻量级矩阵*W A*和*韦伯*。*_*

*为了用具体数字说明这一点，一个完整的 7B LLaMA 检查点需要 23GB 的存储容量，而如果我们选择r=8*的等级，LoRA 权重可以小到 8MB 。

 
**它在实践中有多好？**

LoRA 在实践中有多好，它与完全微调和其他参数有效方法相比如何？根据[LoRA 论文](https://arxiv.org/abs/2106.09685)，在多个特定于任务的基准测试中，使用 LoRA 的模型的建模性能比使用[Adapters](https://arxiv.org/abs/2110.07280)、[prompt tuning](https://arxiv.org/abs/2104.08691)或[prefix tuning的模型略好。](https://arxiv.org/abs/2101.00190)通常，LoRA 的性能甚至比微调所有层更好，如下面 LoRA 论文的注释表所示。（ROUGE 是评估语言翻译性能的指标，我[在这里](https://twitter.com/rasbt/status/1639625228622917632?s=20)更详细地解释了它。）

 

![img](https://lightningaidev.wpengine.com/wp-content/uploads/2023/04/lora-5.png)

 

在这里，值得注意的是 LoRA 与其他微调方法是正交的，这意味着它也可以与前缀调整和适配器结合使用，例如。

 

## **劳拉和美洲驼**

现在，让我们使用 LoRA 的实现来微调 Meta 流行的 LLaMA 模型。由于这已经是一篇很长的文章，我将避免在本文中包含详细代码，但我建议查看[Lit-LLaMA 存储库](https://github.com/Lightning-AI/lit-llama)，它是 Meta 流行的 LLaMA 模型的简单、可读的重新实现。

[除了训练和运行 LLaMA 本身的代码（使用原始的 Meta LLaMA 权重），它还包含使用LLaMA-Adapter](https://github.com/Lightning-AI/lit-llama/blob/main/finetune_adapter.py)和[LoRA](https://github.com/Lightning-AI/lit-llama/blob/main/finetune_lora.py)微调 LLaMA 的代码。

首先，我推荐以下*操作方法*文件：

1. 下载预训练权重 [ [download_weights.md](https://github.com/Lightning-AI/lit-llama/blob/main/howto/download_weights.md) ]
2. 使用 LoRA 进行微调 [ [finetune_lora.md](https://github.com/Lightning-AI/lit-llama/blob/main/howto/finetune_lora.md) ]
3. 使用适配器进行微调 [ [finetune_adapter.md](https://github.com/Lightning-AI/lit-llama/blob/main/howto/finetune_adapter.md) ]（可选，用于比较研究）

在下一节中，我们将比较 7B LLaMA 基础模型与使用 LoRA 和 LLaMA-Adapter 微调的 7B LLaMA 基础模型。（请注意，这需要至少具有 24 Gb RAM 的 GPU）。（关于LLaMA-Adapter方法的更多细节，请看我[之前的文章](https://lightning.ai/pages/community/article/understanding-llama-adapters/)）

 

## **计算性能基准**

在本节中，我们将比较 LLaMA 7B 基础模型与使用 LoRA 和 LLaMA-Adapter 微调的基础模型的计算性能。

微调数据集是[此处](https://github.com/tatsu-lab/stanford_alpaca#data-release)描述的 Alpaca 52k 指令数据集，具有以下结构：

 

![img](https://lightningaidev.wpengine.com/wp-content/uploads/2023/04/lora-6.png)

 

[数据集本身是按照Self-Instruct 论文](https://arxiv.org/abs/2212.10560)中描述的方法生成的，由 49,759 个训练示例和 2000 个验证示例组成。自学过程可以概括为 4 个步骤：

这是如何运作的？简而言之，这是一个 4 步过程

1. 带有一组人工编写的指令（在本例中为 175 个）和样本指令的种子任务池
2. 使用预训练的 LLM（如 GPT-3）来确定任务类别
3. 给定新指令，让预训练的 LLM 生成响应
4. 在将响应添加到任务池之前收集、修剪和过滤响应

 

![img](https://lightningaidev.wpengine.com/wp-content/uploads/2023/04/lora-7.png)

 

请注意，羊驼 52k 数据集是使用上面的自动自指导程序收集的。但是，您也可以使用（或将其与）替代数据集进行比较。例如，一个有趣的候选者是最近发布的开源[databricks-dolly-15k](https://github.com/databrickslabs/dolly/tree/master/data)数据集，其中包含由 Databricks 员工编写的约 15k 条指令/响应微调记录。Lit-LLaMA 存储库包含一个数据集准备脚本，以防您想要使用此 Dolly 15k 数据集而不是 Alpaca 52k 数据集。

给定以下超参数设置（块大小、批量大小和 LoRA r），Adapter 和 LoRA 都可以使用 bfloat-16 混合精度训练在具有 24 Gb RAM 的单个 GPU 上微调 7B 参数 LLaMA 基础模型。

**罗拉**

```python
learning_rate = 3e-4
batch_size = 128
micro_batch_size = 4
gradient_accumulation_steps = batch_size // micro_batch_size
epoch_size = 50000 # train dataset size
num_epochs = 5
max_iters = num_epochs * epoch_size // micro_batch_size // devices
weight_decay = 0.0
block_size = 512
lora_r = 8
lora_alpha = 16
lora_dropout = 0.05
warmup_steps = 100

复制
```

**喇嘛适配器**

```python
learning_rate = 9e-3
batch_size = 128 / devices
micro_batch_size = 4
gradient_accumulation_steps = batch_size // micro_batch_size
epoch_size = 50000 # train dataset size
num_epochs = 5
max_iters = num_epochs * epoch_size // micro_batch_size // devices
weight_decay = 0.02
block_size = 512
warmup_steps = epoch_size * 2 // micro_batch_size // devices

复制
```

以防将来代码发生变化，我将代码（带有超参数设置）包含[在 GitHub 上](https://github.com/rasbt/low-rank-adaptation-blog)。

Adapter 在 A100 上使用了大约 22 Gb 并在 162 分钟内完成了 62,400 次迭代。LoRA 使用了 21 Gb 内存并在 192 分钟内完成。总之，基于 Lit-LLaMA 实现，Adapter 和 LoRA 使用大约相同数量的 RAM 并且具有大致相同的训练时间。（请注意，这是在单个 GPU 上进行的，但如果您有多个 GPU，只需将参数更改`devices`为 > 1 即可利用额外的加速！）

相比之下，完全微调（LLaMA 7B 由 32 个转换器块和 3 个完全连接的输出层组成）需要至少 2 个 GPU，至少 30 Gb 和完全分片训练来分配权重。或者，您可以使用 4 个 GPU，每个 GPU 的最大内存使用量为 22 Gb。在 4 个 GPU 上进行训练，训练耗时 1956 分钟。这在单个 GPU 上至少需要 6,000 分钟，比参数高效的 LLaMA-Adapter 或 LoRA 替代方案贵 30-40 倍。

接下来，让我们看看应用不同微调策略后的模型输出。

 

## **评估建模性能**

有几个指标可用于评估 LLM 生成的文本。例如，困惑度、BLEU 和 ROUGE 分数是自然语言处理中用于评估 LLM 在各种任务中的表现的一些最常见的评估指标。然而，所有这些指标都有很大的缺点，人工评估仍然是黄金标准——人工评估的缺点是创建成本高昂且难以自动化。由于这已经是一篇很长的文章，我将避免对模型评估方法进行详细讨论，并将其推迟到以后的另一篇文章中。在以后的这篇文章中，我们将研究不同的问答数据集（包括 [wikitext](<https://paperswithcode.com/dataset/wikitext-2>)、[c4](<https://paperswithcode.com/dataset /c4>) 和 [ptb](<https:

 

## **定性比较**

为了定性地比较模型，我们将使用提示`"Please explain how weight decay works"`

 
**基础模型**

```python
$ python generate.py \
--checkpoint_path checkpoints/lit-llama/7B/lit-llama.pth

复制
```

> *请解释权重衰减是如何工作的……以及为什么它有时似乎有效而有时却无效！*
>
> *用最简单的术语来说，权重衰减会从模型中移除特征（作为它们重要性的函数）。它也被称为正则化。*
>
> *正规化有*
>
> 推理 1 的时间：总计 4.18 秒，11.95 个令牌/秒

正如我们所见，模型开始通过附加“......以及为什么它有时似乎有效而有时无效！”来继续提示。那是因为基础模型只接受了下一个单词预测任务的训练，并没有专门训练来遵循指令。尽管如此，它试图在下面进一步解释。该解释总体上对正则化做了一些正确的陈述，但它并没有真正抓住权重衰减的关键机制。

**LLaMA-适配器**

```python
$ python generate_adapter.py \
--adapter_path out/adapter/alpaca/lit-llama-adapter-finetuned.pth \
--pretrained_path checkpoints/lit-llama/7B/lit-llama.pth

复制
```

> *权重衰减是一种惩罚术语字数的方法。通过选择单词较少的句子，模型更有可能选择那些可能提供重要信息的句子。*
>
> 推理时间：总计 7.04 秒，14.21 个令牌/秒
>
> 使用内存：13.56 GB

解释在语法上是合理的，但也不一定是我们要找的。

**罗拉**

```python
$ python generate_lora.py \
--lora_path out/lora/alpaca/lit-llama-lora-finetuned.pth \
--pretrained_path checkpoints/lit-llama/7B/lit-llama.pth

复制
```

> *权重衰减是机器学习模型中用于减少过度拟合的算法参数。这是一个调整模型权重以减少模型中不太重要特征的影响并防止过度拟合的过程。权重衰减是通过在权重中加入少量噪声来降低模型对训练数据的影响。这个过程鼓励模型更多地关注对预测任务更重要的特征。*
>
> 推理时间：总计 3.25 秒，30.76 个令牌/秒
>
> 使用内存：27.65 GB

我们越来越近了；当我们想到广义线性模型时，这种解释确实有意义，在广义线性模型中，我们强制模型学习与输入特征相乘的较小权重参数。在神经网络中，这通常会应用于模型中的所有权重参数。

请注意，上述 LoRA 方法目前使用的内存最多。但是，如前所述，我们可以通过将 LoRA 权重与预训练模型权重合并来减少这种内存使用。

由于评估 LLM 本身就是一个大话题，因此定性概述只是这些模型中每一个模型功能的一小部分。我们将在以后的更详细的文章中重新讨论这个话题。但作为这里的要点，LoRA 可用于以相对经济高效的方式在指令数据集上微调 LLM。

 

## **结论**

在本文中，我们讨论了低秩自适应 (LoRA)，它是完全微调的一种参数高效替代方法。我们看到微调 LLaMA 等相对较大的模型可以在使用 LoRA 的单个 GPU 上在几个小时内完成，这使得它对那些不想在 GPU 资源上花费数千美元的人特别有吸引力。LoRA 特别好的地方在于我们可以选择将新的 LoRA 权重矩阵与原始的、预训练的权重合并，这样我们就不会在推理过程中产生额外的开销或复杂性。

随着越来越多的 ChatGPT 或 GPT-4 开源替代品的出现，针对特定目标数据集或目标微调和定制这些 LLM 将在各个研究领域和行业变得越来越有吸引力。参数高效的微调技术（例如 LoRA）使微调更节省资源且更易于访问。

[Lit-LLaMA 存储库](https://github.com/Lightning-AI/lit-llama)中提供了参数有效的微调技术，例如 LoRA 和 LLaMA-Adapter 。如果您对扩展或替代技术有想法，我们总是乐于提供贡献和建议。[请随时通过GitHub](https://github.com/Lightning-AI/lit-llama)或[Discord](https://discord.com/invite/XncpTy7DSt)与我们联系。

 
**致谢**

我要感谢 Luca Antiga 和 Adrian Waelchli 为提高本文的清晰度而提供的建设性反馈。

 

#### 更多来自博客



![img](https://lightningaidev.wpengine.com/wp-content/uploads/2023/06/0_Uncle-Sam-programming-an-artificially-intelligent-_esrgan-v1-x2plus-300x300.png)

##### 退伍军人如何进入 AI



![img](https://lightningaidev.wpengine.com/wp-content/uploads/2023/05/person-300x169.png)

##### Lang Segment Anything——带文本提示的对象检测和分割



##### 如何在自定义数据集上像大型语言模型一样微调 GPT

[![带有文本的 Lightning.AI 徽标](https://lightning.ai/static/media/logo-with-text.2351c373b819a1dafdff787e15c32bbe.svg)](https://lightning.ai/)

[![在 Slack 上加入我们](https://lightning.ai/static/media/icon-social-slack.400a058223d87694329fa315f284f1a9.svg)](https://join.slack.com/t/pytorch-lightning/shared_invite/zt-1dm4phlc0-84Jv9_8Mp_tWraICOJ467Q)[![在推特上关注我们](https://lightning.ai/static/media/icon-social-twitter.727156ae251aac8e8ad0721874a5b8cf.svg)](https://twitter.com/LightningAI)[![在 LinkedIn 上与我们联系](https://lightning.ai/static/media/icon-social-linkedin.d050b1bb25a6b8eceb406da4d44f7435.svg)](https://www.linkedin.com/company/pytorch-lightning/)[![观看我们的 YouTube 频道](https://lightning.ai/static/media/icon-social-youtube.449171b2016a0e612af033acd2ff9823.svg)](https://www.youtube.com/c/PyTorchLightning)

应用程序和组件

[应用程序库](https://lightning.ai/apps)[组件库](https://lightning.ai/components)[提交你的](https://lightning-ai.typeform.com/to/J4ORMSn8?typeform-source=jksfmy97iql.typeform.com)

关于

[特征](https://lightning.ai/#features)[价钱](https://lightning.ai/pricing)

社区

[论坛](https://forums.pytorchlightning.ai/)[不和谐](https://discord.gg/XncpTy7DSt)[GitHub](https://github.com/lightning-ai/lightning)[资源](https://lightning.ai/pages/resources/)[人工智能教育](https://lightning.ai/pages/ai-education/)[招贤纳士](https://boards.greenhouse.io/lightningai)[政策](https://lightning.ai/pages/policies/)

文档

[闪电应用](https://lightning.ai/docs/app/stable)[PyTorch 闪电](https://lightning.ai/docs/pytorch/stable)[织物](https://lightning.ai/docs/fabric/stable)[火炬统计](https://lightning.ai/docs/metrics/stable)