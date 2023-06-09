# Parameter-Efficient LLM Finetuning With Low-Rank Adaptation (`LoRA`)

> 关键要点

> 在快速发展的人工智能领域，以高效和有效的方式使用大型语言模型变得越来越重要。在本文中，您将学习如何以计算高效的方式使用低秩自适应 (LoRA) 调整 LLM！

## Why Finetuning?

预训练的大型语言模型通常被称为基础模型是有充分理由的：它们在各种任务上表现良好，我们可以将它们用作对目标任务进行微调的基础。正如我们在上一篇文章（[了解大型语言模型的参数高效微调：从prefix tuning到 LLaMA-adapters](https://lightning.ai/pages/community/article/understanding-llama-adapters/)）中所讨论的那样，我们讨论了微调允许我们使模型适应目标域和目标任务。不过，它的计算成本可能非常高——模型越大，更新其层的成本就越高。

作为更新所有层的替代方法，已经开发了参数有效的方法，例如 prefix tuning和 adapters。现在，有一种更流行的参数高效微调技术： [Low-rank adaptation (LoRA) by Hu et al](https://arxiv.org/abs/2106.09685). 。什么是 LoRA？它是如何工作的？它与其他流行的微调方法相比如何？让我们在本文中回答所有这些问题！

![PCA 变换](https://lightningaidev.wpengine.com/wp-content/uploads/2023/04/lora-1.jpg)





## Making Weight Updates More Efficient

基于上述想法，论文  [Low-rank adaptation (LoRA) by Hu et al](https://arxiv.org/abs/2106.09685) 建议将权重变化 ΔW分解为低秩表示。（从技术上讲，LoRA并没有直接分解矩阵，而是通过反向传播学习分解的矩阵--这是一个吹毛求疵的细节，以后会有意义）。

在深入了解 LoRA 之前，让我们先简单介绍一下常规微调期间的训练过程。那么，什么是权重变化ΔW？假设W代表某个神经网络层中的权重矩阵。那么，使用常规的反向传播，我们可以得到权重更新ΔW，它通常被计算为损失的负梯度乘以学习率：

ΔW = α ( -∇ L W )。

然后，当我们有ΔW时，我们可以按如下方式更新原始权重：W ' = W + ΔW。下图对此进行了说明（为简单起见，省略了偏置向量）：

或者，我们可以将权重更新矩阵分开并按如下方式计算输出：h = W x + ΔW x

![常规反向传播](https://lightningaidev.wpengine.com/wp-content/uploads/2023/04/lora-2.png)

其中x代表输入，如下图所示。

![img](https://lightningaidev.wpengine.com/wp-content/uploads/2023/04/lora-3.png)

 我们为什么要这样做？目前，这种替代性的表述是为了达到说明LoRA的教学目的，但我们会再来讨论它。

因此，当我们在神经网络中训练完全连接（即“密集”）层时，如上所示，权重矩阵通常具有满秩，这是一个技术术语，意思是矩阵没有任何线性相关（即， “冗余”）行或列。相反，对于满秩，低秩意味着矩阵具有冗余行或列。

因此，虽然预训练模型的权重在预训练任务上具有满秩，但 LoRA 作者 [Aghajanyan ](https://arxiv.org/abs/2012.13255) （2020）等人的说法指出，预训练大型语言模型在适应新任务时具有较低的“内在维度” 。

低内在维度意味着数据可以通过低维空间有效地表示或近似，同时保留其大部分基本信息或结构。换句话说，这意味着我们可以将适应任务的新权重矩阵分解为低维（更小）的矩阵，而不会丢失太多重要信息。

例如，假设ΔW是A × B权重矩阵的权重更新。然后，我们可以将权重更新矩阵分解为两个更小的矩阵：ΔW = W_A  W_B，其中W_A是一个A × r 维矩阵，W_B是一个r × B维矩阵。在这里，我们保持原始权重W不变，只训练新矩阵W_A和W_B。简而言之，这就是 LoRA 方法，如下图所示。

![img](https://lightningaidev.wpengine.com/wp-content/uploads/2023/04/lora-4.png)



### Choosing the rank

请注意，上图中的r是一个超参数，我们可以使用它来指定用于自适应的低秩矩阵的秩。较小的r会导致更简单的低秩矩阵，从而导致在适应过程中需要学习的参数更少。这可以导致更快的训练并可能减少计算需求。然而，随着r越小，低秩矩阵捕获任务特定信息的能力会降低。这可能会导致较低的适应质量，并且与较高的r相比，模型在新任务上的表现可能不那么好。总之，在 LoRA 中选择较小的r，在模型复杂性、适应能力和欠拟合或过拟合风险之间进行权衡。因此，用不同的r值进行实验以找到正确的平衡点，从而在新的任务上达到理想的性能是很重要的。

### Implementing LoRA

LoRA的实现是相对直接的。我们可以把它看作是LLM中全连接层的一个改进的前向传递。在伪代码中，这看起来像如下：

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
    h += x @ (W_A @ W_B)  alpha # use scaled LoRA weights
return h
```

 在上面的伪代码中，`alpha`是一个缩放因子，用于调整组合结果（原始模型输出加上低秩自适应）的大小。这平衡了预训练模型的知识和新的特定于任务的适应——默认情况下，`alpha`通常设置为 1。另请注意，虽然W_A被初始化为小的随机权重，但W_B被初始化为 0，因此

训练开始时ΔW = W_A  W_B = 0 ，这意味着我们以原始权重开始训练。

### Parameter efficiency

现在，让我们解决房间里的大问题：如果我们引入新的权重矩阵，这个参数的效率如何？新矩阵W_A和W_B可以非常小。例如，假设A=100且B=500 ，则ΔW的大小为100 × 500 = 50,000。现在，如果我们将其分解为两个较小的矩阵，一个100×5维矩阵W_A和一个5×500维矩阵W_B。这两个矩阵总共只有5×100+5×500=3000个参数。

### Reducing inference overhead

请注意，在实践中，如果我们在训练后保持原始权重W和矩阵W_A和W_B分开，如上所示，我们将在推理过程中产生小的效率损失，因为这引入了额外的计算步骤。相反，我们可以在训练后通过W' = W + W_A W_B更新权重，这类似于前面提到的W' = W + ΔW。

然而，将权重矩阵W_A和W_B分开可能具有实际优势。例如，假设我们希望将我们的预训练模型作为各种客户的基础模型，并且我们希望从基础模型开始为每个客户创建一个经过微调的 LLM。在这种情况下，我们不需要为每个客户存储完整的权重矩阵W'，其中存储模型的所有权重W' = W + W_A W_B对于 LLM 来说可能非常大，因为 LLM 通常有数十亿到数万亿个权重参数。因此，我们可以保留原始模型W，只需要存储新的轻量级矩阵W_A和WB。

为了用具体数字说明这一点，一个完整的7B LLaMAcheckpoint需要23GB的存储容量，而如果我们选择r=8的秩，LoRA权重可以小到8MB。

### How good is it in practice?

 LoRA在实践中的表现如何，它与完全微调和其他参数有效的方法相比如何？根据[LoRA 论文](https://arxiv.org/abs/2106.09685)，在多个特定任务的基准测试中，使用 LoRA 的模型的建模性能比使用[Adapters](https://arxiv.org/abs/2110.07280)、[prompt tuning](https://arxiv.org/abs/2104.08691)或[prefix tuning的模型略好。](https://arxiv.org/abs/2101.00190)通常，LoRA 的性能甚至比微调所有层更好，如下面 LoRA 论文的注释表所示。（ROUGE 是评估语言翻译性能的指标）

![img](https://lightningaidev.wpengine.com/wp-content/uploads/2023/04/lora-5.png)

在这里，值得注意的是 LoRA 与其他微调方法是正交的，这意味着它也可以与 prefix tuning和adapters结合使用。
