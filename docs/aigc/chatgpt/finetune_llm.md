

# 大型语言模型 (LLM) 微调方法

在快速发展的人工智能领域，高效且有效地利用大型语言模型 (LLM) 变得越来越重要。本质上，我们可以通过两种主要方式将预训练的大型语言模型用于新任务：In-Context Learning和微调。

在本文中，我们将简要介绍In-Context Learning，然后我们将介绍微调 LLM 的各种方法。

## In-Context Learning和索引

自 GPT-2（[Radford 等人](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)）和 GPT-3（[Brown 等人](https://arxiv.org/abs/2005.14165)）以来，我们已经看到，在一般文本语料库上预训练的生成式大语言模型（LLM）能够进行语境学习，如果我们想执行LLM没有被明确训练的特定或新任务，就不需要我们进一步训练或微调预训练的LLM。相反，我们可以直接通过输入Prompt提供一些目标任务的例子，如下面的例子所示。

![img](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ffea460ea-84d5-4973-9bc7-dc0e53a13ae0_1340x680.png)

>  In-Context Learning的一个例子。

如果我们无法直接访问模型，例如，通过 API 或用户界面与 LLM 交互时，则In-Context Learning 非常有用。

与In-Context Learning相关的是HardPrompt Tuning的概念，我们修改输入以希望改进输出，如下图所示。

![img](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa46d7a6f-fbd6-4783-8b5c-0a0bc39f5412_1582x330.png)

> Hard Prompt Tuning的图示

顺便说一句，我们称之为HardPrompt Tuning，因为我们是直接修改输入的单词或标记。稍后，我们将讨论称为Soft Prompt调优（或通常简称为Prompt调优）的可微分版本。

上面提到的快速调整方法提供了一种比参数微调更节省资源的替代方法。但是，它的性能通常达不到微调的要求，因为它不会针对特定任务更新模型的参数，这可能会限制其对特定任务细微差别的适应性。此外，Prompt Tuning 可能是劳动密集型的，因为它通常需要人工参与比较不同Prompt的质量。

另一种利用纯基于In-Context Learning的方法的方法是索引。在 LLM 领域内，索引可以被视为一种In-Context Learning 变通方法，它可以将 LLM 转换为信息检索系统，以便从外部资源和网站中提取数据。在此过程中，索引模块将文档或网站分解为更小的部分，将它们转换为可存储在矢量数据库中的矢量。然后，当用户提交查询时，索引模块计算Embedding查询与数据库中每个向量之间的向量相似度。最终，索引模块获取前k个最相似的Embedding以生成响应。

![img](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F4063347e-8920-40c6-86b3-c520084b303c_1272x998.jpeg)

> 索引的图示

## 三种传统的基于特征和微调的方法

但是，如果我们可以访问 LLM，则使用来自目标域的数据在目标任务上对其进行微调通常会产生更好的结果。那么，我们如何才能使模型适应目标任务呢？下图概述了三种常规方法。

![img](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa505c654-5ddf-485f-90a8-b656d03b94dc_2394x834.png)

> 3 种传统的基于特征和微调的方法。

为了下面的讨论提供一些实际背景，我们正在为分类任务微调编码器样式的 LLM，例如 BERT ( [Devlin et al. 2018 )。](https://arxiv.org/abs/1810.04805)（此分类任务预测电影评论是否具有正面或负面情绪。）请注意，与其对编码器式的LLM进行微调，同样的方法也适用于类似GPT的解码器式LLM。此外，我们还可以对解码器式的LLM进行微调，以生成特定指令的多句话答案，而不是仅仅对文本进行分类。

### 1. 基于特征的方法

在基于特征的方法中，我们加载一个预训练的LLM并将其应用于我们的目标数据集。在这里，我们对生成训练集的输出Embedding特别感兴趣，我们可以将其作为输入特征来训练一个分类模型。虽然这种方法对于像BERT这样以Embedding为重点的方法特别常见，但我们也可以从生成的GPT式模型中提取Embedding。

然后，分类模型可以是逻辑回归模型、随机森林或XGBoost。(然而，根据我的经验，像逻辑回归这样的线性分类器在这里表现最好）。

从概念上讲，我们可以用下面的代码来说明基于特征的方法：

```python
model = AutoModel.from_pretrained("distilbert-base-uncased")

# ...
# tokenize dataset
# ...

# generate embeddings
@torch.inference_mode()
def get_output_embeddings(batch):
    output = model(
        batch["input_ids"],
        attention_mask=batch["attention_mask"]
    ).last_hidden_state[:, 0]
return {"features": output}

dataset_features = dataset_tokenized.map(
  get_output_embeddings, batched=True, batch_size=10)

X_train = np.array(imdb_features["train"]["features"])
y_train = np.array(imdb_features["train"]["label"])

X_val = np.array(imdb_features["validation"]["features"])
y_val = np.array(imdb_features["validation"]["label"])

X_test = np.array(imdb_features["test"]["features"])
y_test = np.array(imdb_features["test"]["label"])

# train classifier
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(X_train, y_train)

print("Training accuracy", clf.score(X_train, y_train))
print("Validation accuracy", clf.score(X_val, y_val))
print("test accuracy", clf.score(X_test, y_test))
```

[（有兴趣的读者可以在此处](https://github.com/rasbt/LLM-finetuning-scripts/tree/main/conventional/distilbert-movie-review)找到完整的代码示例。）

### 2. Finetuning I——更新输出层

与上述基于特征的方法相关的一个流行方法是对输出层进行微调（我们将这种方法称为微调I）。与基于特征的方法类似，我们保持预训练的LLM的参数冻结。我们只训练新增加的输出层，类似于在Embedding特征上训练逻辑回归分类器或小型多层感知器。

在代码中，这将看起来如下：

```python
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
     num_labels=2
)

# freeze all layers
for param in model.parameters():
    param.requires_grad = False

# then unfreeze the two last layers (output layers)
for param in model.pre_classifier.parameters():
    param.requires_grad = True

for param in model.classifier.parameters():
    param.requires_grad = True

# finetune model
lightning_model = CustomLightningModule(model)

trainer = L.Trainer(
    max_epochs=3,
    ...
)

trainer.fit(
  model=lightning_model,
  train_dataloaders=train_loader,
  val_dataloaders=val_loader)

# evaluate model
trainer.test(lightning_model, dataloaders=test_loader)
```

[（有兴趣的读者可以在这里](https://github.com/rasbt/LLM-finetuning-scripts/tree/main/conventional/distilbert-movie-review)找到完整的代码示例。）

理论上，这种方法在建模性能和速度方面应该和基于特征的方法有类似的表现，因为我们使用的是相同的冻结骨干模型。然而，由于基于特征的方法使预先计算和存储训练数据集的嵌入特征稍微容易一些，所以基于特征的方法对于特定的实际场景可能更方便。

### 3. Finetuning II – 更新所有层

虽然最初的BERT论文（Devlin等人）报告说，只对输出层进行微调可以使建模性能与对所有层进行微调相当，但由于涉及更多的参数，所以成本要高得多。例如，一个BERT基础模型有大约1.1亿个参数。然而，用于二元分类的BERT基础模型的最后一层仅由1,500个参数组成。此外，BERT基础模型的最后两层占60,000个参数--这只占总模型大小的0.6%左右。

我们的里程数将根据我们的目标任务和目标领域与模型预训练的数据集的相似程度而有所不同。但在实践中，对所有层进行微调，几乎总是能带来卓越的建模性能。

因此，在优化建模性能时，使用预训练的LLM的黄金标准是更新所有层（这里称为微调II）。从概念上讲，微调II与微调I非常相似。唯一的区别是，我们不冻结预训练的LLM的参数，而是对其进行微调：

```python
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
     num_labels=2
)

# freeze layers (which we don't do here)
# for param in model.parameters():
#    param.requires_grad = False


# finetune model
lightning_model = LightningModel(model)

trainer = L.Trainer(
    max_epochs=3,
    ...
)

trainer.fit(
  model=lightning_model,
  train_dataloaders=train_loader,
  val_dataloaders=val_loader)

# evaluate model
trainer.test(lightning_model, dataloaders=test_loader)
```

[（有兴趣的读者可以在这里](https://github.com/rasbt/LLM-finetuning-scripts/tree/main/conventional/distilbert-movie-review)找到完整的代码示例。）

如果您对一些真实世界的结果感到好奇，上面的代码片段用于使用预训练的 DistilBERT 基本模型训练电影评论分类器 您可以在此处访问[代码笔记本](https://github.com/rasbt/LLM-finetuning-scripts/tree/main/conventional/distilbert-movie-review)：

- 1) 基于特征的逻辑回归方法：83% 的测试准确率
- 2) Finetuning I，更新最后两层：87%的准确率
- 3) Finetuning II，更新所有层：92% 准确率。

这些结果与一般经验法则一致，即微调更多层通常会带来更好的性能，但它会增加成本。

![img](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fae8e84db-16f9-485d-a0cb-0392fc8aca56_1454x536.png)

>  各种方法的经验法则，计算和建模性能权衡。

## Parameter-Efficient Finetuning （Peft）

在前面的章节中，我们了解到，微调更多的层通常会导致更好的结果。上面的实验是基于一个DistilBERT模型，它相对较小。如果我们想微调更大的模型，而这些模型只能勉强装入GPU内存，例如最新的生成型LLMs，该怎么办？当然，我们可以使用上面的基于特征或微调I的方法。但是，假设我们想获得类似于微调II的建模质量？

微调 LLM 在计算资源和时间方面可能非常昂贵，这就是研究人员开始开发参数高效微调方法的原因。

参数有效的微调使我们能够重复使用预训练的模型，同时最大限度地减少计算和资源的占用。总而言之，参数高效微调至少有5个原因：

- 降低了计算成本（需要更少的GPU和GPU时间）；

- 更快的训练时间（更快地完成训练）；

- 更低的硬件件要求（可使用更小的GPU和更少的智能存储器）；

- 更好的建模性能（减少过度拟合）；

- 更少的存储空间（大部分权重可以在不同的任务中共享）。

最近几年，研究人员开发了几种技术（[Lialin 等人](https://arxiv.org/abs/2303.15647)）来微调LLM，使其具有较高的建模性能，同时只需要训练少量的参数。这些方法通常被称为参数高效微调技术（PEFT）。

下图总结了一些最广泛使用的 PEFT 技术。

![img](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F89599158-c5bf-4e73-9d31-a388b625e4d2_2262x622.png)

>  精选最流行的参数高效微调技术。

那么，这些技术是如何工作的呢？简而言之，它们都涉及引入少量的额外参数，我们对这些参数进行微调（而不是像我们在上面的微调II方法中那样对所有层进行微调）。从某种意义上说，Finetuning I（只对最后一层进行微调）也可以被认为是一种参数高效的微调技术。然而，如 prefix tuning, adapters, and low-rank adaptation 等技术，都是 "修改 "多层的，可以实现更好的预测性能（成本低）。

最近引起轰动的一种 PEFT 技术是 LLaMA-Adapter，它是为 Meta 流行的 LLaMA 模型提出的（[Touvron 等人](https://arxiv.org/abs/2302.13971)）——然而，虽然 LLaMA-Adapter 是在 LLaMA 的背景下提出的，但该想法与模型无关。

要了解 LLaMA-Adapter 的工作原理，我们必须后退一步，回顾称为 Prefix Tuning 和 Adapters 的两种相关技术 ——LLaMA-Adapter（[Zhang 等人](https://arxiv.org/abs/2303.16199)）结合并扩展了这两种思想。

因此，在本文的其余部分，我们将在仔细研究 LLaMA-Adapter 之前讨论Prompt修改的各种概念，以了解Prefix Tuning和Adapters 方法。

### Prompt Tuning和Prefix Tuning

Prompt Tuning的原始概念是指改变输入Prompt以获得更好的建模结果的技术。例如，假设我们有兴趣将英语句子翻译成德语。我们可以通过各种不同的方式询问模型，如下图所示。
![HardPrompt的例子](https://lightningaidev.wpengine.com/wp-content/uploads/2023/04/hard-prompting.png)

现在，上面说明的这个概念被称为 Hard Prompt Tuning，因为我们直接更改不可微分的离散输入标记。

与 Hard Prompt Tuning相比， Soft Prompt Tuning将输入标记的嵌入与可训练张量连接起来，该张量可以通过反向传播进行优化，以提高目标任务的建模性能。

Prompt Tuning的一种特殊方法是Prefix Tuning（[Li 和 Liang](https://arxiv.org/abs/2101.00190)）。Prefix Tuning的想法是向每个Transformer Block添加一个可训练的张量，而不是像 Soft Prompt Tuning中那样仅在输入嵌入中添加。下图说明了常规Transformer Block和使用前缀修改的Transformer Block之间的区别。


![LLM 的Prefix Tuning](https://lightningaidev.wpengine.com/wp-content/uploads/2023/04/prefix-tuning.png)

请注意，在上图中，“全连接层”指的是一个小型多层感知器（两个全连接层，中间有一个非线性激活函数）。这些完全连接的层将Soft Prompt嵌入到与Transformer Block输入具有相同维度的特征空间中，以确保连接的兼容性。

使用（Python）伪代码，我们可以说明常规Transformer Block和前缀修改Transformer Block之间的区别，如下所示：

![带有前缀代码的转换器博客](https://lightningaidev.wpengine.com/wp-content/uploads/2023/04/prefix-code.png)

根据原始 [Prefix Tuning](https://arxiv.org/abs/2101.00190) 论文，Prefix Tuning实现了与微调所有层相当的建模性能，同时只需要训练 0.1% 的参数——实验基于 GPT-2 模型。此外，在许多情况下，Prefix Tuning甚至优于所有层的微调，这可能是因为涉及的参数更少，这有助于减少较小目标数据集上的过度拟合。

最后，推理过程中Soft Prompt的使用：学习Soft Prompt后，我们必须在执行微调模型的特定任务时将其作为前缀提供。这允许模型调整其对特定任务的响应。此外，我们可以有多个Soft Prompt，每个Soft Prompt对应一个不同的任务，并在推理过程中提供适当的前缀以获得特定任务的最佳结果。

### Adapters

我们现在正在讨论一种称为 adapters 的相关方法，它的核心思想是向 LLM 的各种Transformer块添加可调层，而不是仅修改输入Prompt。

原始 Adapters  方法（[Houlsby 等人](https://arxiv.org/abs/1902.00751)）与上述 Prefix Tuning有些相关 ，因为它们还向每个Transformer Block添加了额外的参数。但是，Adapters 方法不是将可调张量添加到嵌入中，而是在两个地方添加Adapters 层，如下图所示。

![img](https://lightningaidev.wpengine.com/wp-content/uploads/2023/04/adapter-outline.png)

而对于喜欢（Python）伪代码的读者，Adapters 层可以这样写：


![法学硕士Adapters 代码](https://lightningaidev.wpengine.com/wp-content/uploads/2023/04/adapter.png)

> 使用Adapter层修改的Transformer Block的插图。

请注意，Adapters 的全连接层通常相对较小，并且具有类似于自动编码器的瓶颈结构。每个Adapters 块的第一个全连接层将输入向下投影到低维表示上。第二个全连接层将输入投影回输入维度。这个参数如何有效？例如，假设第一个全连接层将 1024 维输入投射到 24 维，第二个全连接层将其投射回 1024 维。这意味着我们引入了 1,024 x 24 + 24 x 1,024 = 49,152 个权重参数。相比之下，将 1024 维输入重新投影到 1,024 维空间的单个全连接层将具有 1,024 x 1024 = 1,048,576 个参数。

根据原始[Adapter论文](https://arxiv.org/abs/1902.00751)，使用Adapter方法训练的 BERT 模型达到了与完全微调的 BERT 模型相当的建模性能，而只需要训练 3.6% 的参数。

此外，研究人员提供了一张图，其中他们将Adapter方法与仅对 BERT 模型的输出（顶层）层进行微调进行了比较，发现使用Adapter，可以将微调顶层微调性能与数量少得多的参数：

![img](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F75e6face-6d72-4ce6-888e-b17dbf8bd39b_1014x742.png)

> Adapter论文中的注释图，https://arxiv.org/abs/1902.00751。

现在，问题是Adapters 方法与Prefix Tuning相比如何。根据原始[前缀调优论文](https://arxiv.org/abs/2101.00190)，当调优模型参数总数的 0.1% 时，Adapters 方法的性能略差于前缀调优方法。然而，当Adapters 方法用于调整 3% 的模型参数时，该方法与 0.1% 的模型参数的Prefix Tuning相关联。因此，我们可以得出结论，Prefix Tuning方法是两者中更有效的方法。

### LLaMA-Adapter

扩展Prefix Tuning和原始Adapters 方法的思想，研究人员最近提出了 LLaMA-Adapter（[Zhang 等人），这是一种](https://arxiv.org/abs/2303.16199)[LLaMA](https://github.com/facebookresearch/llama)的Peft方法 （LLaMA 是 Meta 流行的 GPT 替代方案）。

与 prefix tuning一样，LLaMA-Adapter 方法将可调Prompt张量添加到嵌入式输入中。值得注意的是，在 LLaMA-Adapter 方法中，前缀是在嵌入表中学习和维护的，而不是在外部提供的。模型中的每个Transformer Block都有自己独特的学习前缀，允许跨不同模型层进行更量身定制的适应。

此外，LLaMA-Adapter 引入了一种零初始化的注意力机制和门控。这种所谓的零初始注意力和门控背后的动机是，Adapters 和Prefix Tuning可能会通过合并随机初始化的张量（前缀Prompt或Adapters 层）来破坏预训练 LLM 的语言知识，从而导致不稳定的微调和高损失值在初始训练阶段。

与Prefix Tuning和原始Adapters 方法相比的另一个区别是，LLaMA-Adapter 仅将可学习的自适应Prompt添加到 L 个最顶层的Transformer，而不是所有Transformer。作者认为，这种方法可以更有效地调整专注于更高级别语义信息的语言表示。

虽然 LLaMA Adapters 方法的基本思想与Prefix Tuning（前置可调Soft Prompt）有关，但在其实现方式上存在一些额外的细微差异。例如，通过可调Soft Prompt仅修改自注意输入的键和值序列。然后，根据门控因子（在训练开始时设置为零），使用或不使用前缀修饰注意力。这个概念在下面的可视化中说明。


![骆驼Adapters 大纲](https://lightningaidev.wpengine.com/wp-content/uploads/2023/04/llama-adapter.png)


在伪代码中，我们可以这样表达：


![美洲驼Adapters 伪代码](https://lightningaidev.wpengine.com/wp-content/uploads/2023/04/llama-adapter-code-1.png)

简而言之，LLaMA-Adapter 与常规Prefix Tuning的区别在于，LLaMA-Adapter 仅修改顶部（即前几个）transformer 块，并引入门控机制来稳定训练。虽然研究人员专门对 LLaMA 进行了实验，但他们提出的 Adapter 方法是一种通用方法，也可以应用于其他类型的 LLM（如 GPT）。

使用 LLaMA-Adapter 方法，研究人员能够在包含 52k 指令对的数据集上仅用 1 小时（使用八个 A100 GPU）微调一个 70 亿参数的 LLaMA 模型。此外，经过微调的 LLaMA-Adapter 模型优于本研究中关于问答任务的所有其他模型，而只有 1.2 M 参数（Adapters 层）需要微调。

[如果您想查看 LLaMA-Adapter 方法，可以在此处](https://github.com/ZrrSkywalker/LLaMA-Adapter)找到基于 GPL 许可的 LLaMA 代码的原始实现 。

## 人类反馈强化学习

在带有人类反馈的强化学习 (RLHF) 中，使用监督学习和强化学习的组合对预训练模型进行微调——该方法由最初的ChatGPT模型推广，而该模型又基于 [InstructGPT](https://arxiv.org/abs/2203.02155)（Ouyang 等人)。

在 RLHF 中，通过让人类对不同的模型输出进行排名或评级来收集人类反馈，从而提供奖励信号。然后可以使用收集到的奖励标签来训练奖励模型，该模型随后用于指导 LLM 适应人类偏好。

奖励模型本身是通过监督学习学习的（通常使用预训练的 LLM 作为基础模型）。接下来，奖励模型用于更新要适应人类偏好的预训练 LLM——训练使用一种称为近端策略优化的强化学习（[Schulman 等人](https://arxiv.org/abs/1707.06347)）。

![img](https://substackcdn.com/image/fetch/w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F7dfa415c-da9c-4d6f-8de8-ffc9f92272db_1602x952.png)

> InstructGPT 论文的屏幕截图概述了 RLHF 过程。

为什么使用奖励模型而不是直接根据人类反馈训练预保留模型？这是因为让人类参与学习过程会造成瓶颈，因为我们无法实时获得反馈。

## 结论

微调预训练的大型语言模型 (LLM) 是定制这些模型以满足特定业务需求并使其与目标域数据保持一致的有效方法。此过程涉及使用与所需领域相关的较小数据集调整模型参数，这使模型能够学习特定领域的知识和词汇。

然而，由于 LLM“很大”，更新 Transformer 模型中的多个层可能非常昂贵，因此研究人员开始开发参数高效的替代方案。

我们讨论了传统 LLM 微调机制的几种参数高效替代方案。特别是，我们介绍了通过前缀调整和插入额外的适配器层来预置可调软提示。我们讨论了最近流行的 LLaMA-Adapter 方法，该方法预先设置可调软提示并引入额外的门控机制来稳定训练。

此外，带有人类反馈的强化学习 (RLHF) 可作为监督微调的替代方案，有可能提高模型性能。
