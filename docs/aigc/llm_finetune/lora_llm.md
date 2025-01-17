
> 关键要点

> 在人工智能领域迅速发展的今天，以一种高效且有效的方式使用大型语言模型变得越来越重要。在本文中，你将学习如何以计算高效的方式使用低秩适应（LoRA）对LLM进行微调！

## 为什么进行微调？

预训练的大型语言模型通常被称为基础模型，因为：它们在各种任务上表现良好，我们可以将它们用作对目标任务进行微调的基础。正如我们在上一篇文章（[了解大型语言模型的参数高效微调：从prefix tuning到 LLaMA-adapters](https://lightning.ai/pages/community/article/understanding-llama-adapters/)）中所讨论的，微调允许我们使模型适应目标域和目标任务。然而，这在计算上可能非常昂贵——模型越大，进行参数更新的成本就越高。

作为更新所有层的替代方法，已经开发了参数有效的方法，例如 prefix tuning和 adapters。现在，有一种更流行的参数高效微调技术： [Low-rank adaptation (LoRA) by Hu et al](https://arxiv.org/abs/2106.09685)。什么是 LoRA？它是如何工作的？它与其他流行的微调方法相比如何？让我们在本文中回答所有这些问题！

<img src="https://lightningaidev.wpengine.com/wp-content/uploads/2023/04/lora-1.jpg" alt="PCA 变换" style="zoom:50%;" />

## 使权重更新更高效

基于上述想法，论文  [Low-rank adaptation (LoRA) by Hu et al](https://arxiv.org/abs/2106.09685) 建议将权重变化 $ΔW$ 分解为低秩表示。（从技术上讲，LoRA并没有直接分解矩阵，而是通过反向传播学习分解矩阵--这是一个吹毛求疵的细节，这是一个细节问题，我们稍后会讲到）

在深入了解 LoRA 之前，让我们先简单介绍一下常规模型微调的训练过程。那么，什么是权重变化$ΔW$？假设W代表某个神经网络层中的权重矩阵。那么，使用常规的反向传播，我们可以得到权重更新$ΔW$，它通常被计算为损失的负梯度乘以学习率：

$$
ΔW = α ( -∇ L W )
$$
然后，当我们有ΔW，我们可以按如下方式更新原始权重：$W ' = W + ΔW$。下图对此进行了说明（为简单起见，省略了偏置向量）：

或者，我们可以将权重更新矩阵分开并按如下方式计算输出：$h = W x + ΔWx$

<img src="https://lightningaidev.wpengine.com/wp-content/uploads/2023/04/lora-2.png" alt="常规反向传播" style="zoom: 33%;" />

其中x代表输入，如下图所示。

<img src="https://lightningaidev.wpengine.com/wp-content/uploads/2023/04/lora-3.png" alt="img" style="zoom: 50%;" />

 我们为什么要这样做？目前，这种替代公式有助于说明LoRA，但我们会回来讨论它。

因此，当我们在神经网络中训练全连接（即“密集”）层时，如上所示，权重矩阵通常具有满秩，这是一个术语，意思是矩阵没有任何线性相关的（即， “冗余”）行或列。相反，与满秩相反的，低秩意味着矩阵具有冗余行或列。

因此，虽然预训练模型的权重在预训练任务上具有满秩，但 LoRA 作者 [Aghajanyan ](https://arxiv.org/abs/2012.13255) （2020）指出，当预训练大型语言模型在适应新任务时，具有较低的“内在维度” 。

低内在维度意味着数据可以通过低维空间有效地表示或近似，同时保留其大部分基本信息或结构。换句话说，这意味着我们可以将适应任务的新权重矩阵分解为低维（更小）的矩阵，而不会丢失太多重要信息。

例如，假设$ΔW$ 是$A × B$权重矩阵的权重更新。然后，我们可以将权重更新矩阵分解为两个更小的矩阵：$ΔW = W_A W_B$，其中$$W_A$$是一个 A × r 维矩阵，$$W_B$$是一个r × B维矩阵。在这里，我们保持原始权重W不变，只训练新矩阵 $W_A$ 和 $W_B$。简而言之，这就是 LoRA 方法，如下图所示。

<img src="https://lightningaidev.wpengine.com/wp-content/uploads/2023/04/lora-4.png" alt="img" style="zoom:50%;" />



### 选择秩

请注意，上图中的r是一个超参数，我们可以使用它来指定用于自适应的低秩矩阵的秩。较小的r会导致更简单的低秩矩阵，从而导致在适应过程中需要学习的参数更少。这可以导致更快的训练并减少计算需求。然而，随着r越小，低秩矩阵捕获任务特定信息的能力会降低。这可能会导致较低的适应质量，模型在新任务上的表现可能不如较高的r。总之，在 LoRA 中选择较小的r，在模型复杂性、适应能力和欠拟合或过拟合风险之间c存在权衡。因此，用不同的r值进行实验以找到正确的平衡点，从而在新的任务上达到理想的性能是很重要的。

### 实现LoRA

LoRA的实现是相对直接的。我们可以把它看作是LLM中全连接层的一个改进的前向传递。在伪代码中，它看起来像下面这样：

```python
input_dim = 768  # 例如，预训练模型的隐藏尺寸
output_dim = 768  # 例如，层的输出尺寸
rank = 8  # 低秩适应的秩'r'
W = ... # 来自具有形状input_dim x output_dim的预训练网络

W_A = nn.Parameter(torch.empty(input_dim, rank)) # LoRA权重A
W_B = nn.Parameter(torch.empty(rank, output_dim)) # LoRA权重B

# 初始化LoRA权重
nn.init.kaiming_uniform_(W_A, a=math.sqrt(5))
nn.init.zeros_(W_B)

def regular_forward_matmul(x, W):
    h = x @ W
    return h

def lora_forward_matmul(x, W, W_A, W_B):
    h = x @ W  # 常规矩阵乘法
    h += x @ (W_A @ W_B)alpha # 使用缩放的LoRA权重
    return h
```

 在上面的伪代码中，`alpha`是一个缩放因子，用于调整组合结果（原始模型输出加上低秩自适应）的大小。这平衡了预训练模型的知识和新的特定于任务的适应——默认情况下，`alpha`通常设置为 1。另请注意，虽然$W_A$被初始化为小的随机权重，但$W_B$被初始化为 0，因此在训练开始时 $ΔW = W_AW_B = 0$ ，这意味着我们以原始权重开始训练。

### 参数效率

现在，让我们解决一个重要的问题：如果我们引入了新的权重矩阵，这怎么会是参数高效的呢？新矩阵$W_A$和$W_B$可以非常小。例如，假设A=100且B=500 ，则ΔW的大小为100 × 500 = 50,000。现在，如果我们将其分解为两个较小的矩阵，一个100×5维矩阵$W_A$和一个5×500维矩阵$W_B$。这两个矩阵总共只有5×100+5×500=3,000个参数。

### 减少推理开销

请注意，在实践中，如果我们在训练后保持原始权重W和矩阵$W_A$和$W_B$分开，如上所示，我们将在推理过程中产生小的效率损失，因为这引入了额外的计算步骤。相反，我们可以在训练后通过$W' = W + W_A$ $W_B$更新权重，这类似于前面提到的$W' = W + ΔW$。

然而，将权重矩阵$W_A$和$W_B$分开可能具有实际优势。例如，假设我们希望将我们的预训练模型作为各种客户的基础模型，并且我们希望从基础模型开始为每个客户创建一个经过微调的 LLM。在这种情况下，我们不需要为每个客户存储完整的权重矩阵W'，存储模型的所有权重 $W' = W + W_A W_B$ 对于 LLM 来说可能非常大，因为 LLM 通常有数十亿到数万亿个权重参数。因此，我们可以保留原始模型W，只需要存储新的轻量级矩阵$W_A$和$W_B$。

为了用具体数字说明这一点，一个完整的7B LLaMA checkpoint 需要23GB的存储容量，而如果我们选择r=8的秩，LoRA权重可以小到8MB。

### 在实践中表现如何？

 LoRA在实践中的表现如何，它与完全微调和其他参数有效的方法相比如何？根据[LoRA 论文](https://arxiv.org/abs/2106.09685)，在多个特定任务的基准测试中，使用 LoRA 的模型的建模性能比使用[Adapters](https://arxiv.org/abs/2110.07280)、[prompt tuning](https://arxiv.org/abs/2104.08691)或[prefix tuning的模型略好。](https://arxiv.org/abs/2101.00190)通常，LoRA 的性能甚至比微调所有层更好，如下面 LoRA 论文的注释表所示。（ROUGE 是评估语言翻译性能的指标）

<img src="https://lightningaidev.wpengine.com/wp-content/uploads/2023/04/lora-5.png" alt="img" style="zoom:50%;" />

在这里，值得注意的是 LoRA 与其他微调方法是正交的，这意味着它也可以与 prefix tuning和adapters结合使用。
