# 	图解 MOE

在看最新发布的大型语言模型（LLMs）时，你可能经常会在标题中看到 “MoE” 这个词。那么，这个 “MoE” 到底代表什么？为什么现在有这么多 LLM 都在使用它呢？

在这篇图解指南中，原作者使用超过 50 张图片进行可视化图示，带你一步步深入了解这个关键组成部分 —— **专家混合模型（MoE，Mixture of Experts）** 。

<img src="https://substackcdn.com/image/fetch/$s_!o-PE!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F50a9eba8-8490-4959-8cda-f0855af65d67_1360x972.png" alt="img" style="zoom:25%;" />

在这份图解指南中，我们将依照典型的大语言模型（LLM）架构，依次讲解 MoE 的两个核心组成部分：**专家（Experts）** 和 **路由器（Router）** 。

## 什么是混合专家模型

**专家混合模型（MoE，Mixture of Experts）** 是一种提升大型语言模型（LLMs）质量的技术，它通过引入多个不同的子模型（即“专家”）来实现更好的性能。

MoE 的结构主要由两个核心组成部分构成：

- **专家（Experts）** ：每一层前馈神经网络（FFNN）不再是一个单一结构，而是由多个“专家”组成。每个“专家”本质上也是一个前馈神经网络。
- **路由器（Router）或门控网络（Gate Network）** ：负责决定每个 token（词元）应该被送到哪些专家那里去处理。

在使用 MoE 架构的大语言模型中，我们会在模型的每一层看到多个（具有一定专长的）专家：

<img src="https://substackcdn.com/image/fetch/$s_!--9T!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F7931367a-a4a0-47ac-b363-62907cd6291c_1460x356.png" alt="img" style="zoom:50%;" />

需要注意的是，这里的“专家”并不是指在某个具体学科领域（比如“心理学”或“生物学”）特别擅长的模型。它们最多只是在处理某些词语的语法结构方面表现更好：

<img src="https://substackcdn.com/image/fetch/$s_!GNME!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc6a81780-27c8-45f8-bccc-cc8f1ce3e943_1460x252.png" alt="img" style="zoom:50%;" />

更准确地说，这些专家擅长处理某些特定上下文中出现的特定 token。

而路由器（或门控网络）的作用，就是根据输入内容选择最合适的专家来处理每个 token：

<img src="https://substackcdn.com/image/fetch/$s_!4NiQ!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb6a623a4-fdbc-4abf-883b-3c2679b4ad4d_1460x640.png" alt="img" style="zoom:50%;" />

每个专家并不是一个完整的 LLM，而只是 LLM 架构中的一个子模块。

## 专家模块

为了更好地理解“专家”代表什么，以及它们是如何工作的，我们先来看看 MoE 想要替代的是什么，稠密层（dense layers），也就是传统的全连接层 。

### 稠密层

专家混合模型（MoE）的出发点，是 LLM 中最基础的组件之一：前馈神经网络（FFNN, Feedforward Neural Network） 。

回想一下，在一个标准的仅解码器（decoder-only）Transformer 架构中，FFNN 放是在层归一化（layer normalization）之后的：

<img src="https://substackcdn.com/image/fetch/$s_!SOKY!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd4729d2a-a51a-4224-93fe-c5674b9b38eb_1460x800.png" alt="img" style="zoom:33%;" />

不了解decoer-only Transformer的人看到上面这个图可能会困惑。下面讲解一下：

普通的Transformer是post-ln，结构如下：

```
input
 │
 ├─ Masked Multi-Head Self-Attention
 ├─ Residual Add + LayerNorm
 │
 ├─ Encoder-Decoder Cross Attention
 ├─ Residual Add + LayerNorm
 │
 ├─ Feedforward Neural Network (FFNN)
 ├─ Residual Add + LayerNorm
```

但是decoder-only Transformer架构的模型，比如gpt系列，用的pre-ln，并且因为是decoer-only，所以不需要交叉注意力层（或者叫编码器解码器注意力层），因此构架是这样的：

```
Input
 │
 ├─ LayerNorm
 │
 ├─ Masked Multi-Head Attention
 │
 ├─ Residual Add
 │
 ├─ LayerNorm
 │
 ├─ Feedforward Neural Network (FFNN)
 │
 ├─ Residual Add
```

FFNN 的作用，是利用注意力机制生成的上下文信息，并进一步转换这些信息，从而捕捉数据中更复杂的关联关系。

不过，FFNN 的规模增长得非常快。为了学到这些复杂关系，它通常会对输入数据进行“扩展”：

<img src="https://substackcdn.com/image/fetch/$s_!YGa3!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F091ec102-45f0-4456-9e0a-7218a49e01df_1460x732.png" alt="img" style="zoom:33%;" />

### 稀疏层

传统 Transformer 中的 FFNN 被称为**稠密模型（dense model）** ，因为它的所有参数（包括权重和偏置）在每次前向传播时都会被激活。所有部分都会参与运算，没有遗漏。

如果我们仔细观察一个稠密模型，会发现输入在计算中“激活”了所有的参数：

<img src="https://substackcdn.com/image/fetch/$s_!fE65!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F101e8ddc-9aa7-4e24-92fc-78d25da73399_880x656.png" alt="img" style="zoom:33%;" />

相反，**稀疏模型（sparse models）** 只激活部分参数，这就与 MoE 的机制密切相关了。

举个例子：我们可以把一个密集模型拆成若干片（也就是所谓的专家），然后重新训练。在模型实际运行时，每次只激活其中的一小部分专家：

<img src="https://substackcdn.com/image/fetch/$s_!7R1U!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fcc4eeaf8-166b-419f-896c-463498af5692_880x656.png" alt="img" style="zoom:33%;" />

背后的思想是：**每个专家在训练时学会了不同的知识**。因此，在推理阶段也就是模型实际使用的时候 ，只会调用那些与当前任务最相关的专家。

当我们向模型提问时，系统就会选择最适合回答这个问题的专家：

<img src="https://substackcdn.com/image/fetch/$s_!bmV0!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fce63e5cc-9b82-45b4-b3dc-9db0cac47da3_880x748.png" alt="img" style="zoom:33%;" />

### 专家学到了什么？

前边我们讲到了，专家所学的信息比起整个领域而言，是更为细粒度的信息。因此，有时称它们为“专家”其实有些误导人。

<img src="https://substackcdn.com/image/fetch/$s_!u8o5!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F04123f9e-b798-4712-bcfb-70a26438f3b9_2240x1588.png" alt="img" style="zoom:33%;" />

> 上图是 ST-MoE 论文中，编码器模型中专家表现出的专门化。通俗理解就是不同专家捕获不同的信息。

然而，解码器模型中的专家似乎并没有表现出同样类型的专门化。当然这也并不意味着所有专家都是一样的。

举个例子，在论文[ Mixtral 8x7B](https://arxiv.org/pdf/2401.04088) 中，每个token都用第一个选择它的专家进行颜色标记。就是你一句话输入进去，不同的专家会注意到不同的token，每个专家用不同颜色表示，如果在这个句子中，某个词语第一次就被引导到某个专家，那就用这个专家的颜色进行标记。

<img src="https://substackcdn.com/image/fetch/$s_!guHK!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd03e32b4-5830-4d98-8514-0c1a28127ed9_1028x420.png" alt="img" style="zoom:33%;" />

这张图显示，专家们更倾向于关注语法结构，而不是具体的领域知识。

因此，虽然解码器的专家看起来没有明显的专业领域，但它们好像会用于处理某些特定类型的token。

### 专家的架构

虽然我们可以把专家想象成把一个密集模型的隐藏层切分成若干部分，但实际上它们通常是完整的FFNN结构：

<img src="https://substackcdn.com/image/fetch/$s_!dNL3!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fe51561eb-f3d6-45ca-a2f8-c71abfa7c2a9_880x748.png" alt="img" style="zoom:33%;" />

由于大多数大型语言模型有多个解码器块，一段文本在生成过程中会经过多个专家的处理：

<img src="https://substackcdn.com/image/fetch/$s_!Xs0T!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F89b1caad-5201-43fe-b7de-04ebe877eb2d_1196x836.png" alt="img" style="zoom:33%;" />

不同的 token 会选择不同的专家，这就导致模型在计算时会走不同的“路径”：

<img src="https://substackcdn.com/image/fetch/$s_!Z27D!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fcde4794d-8b3e-454d-9a1c-88c1999fdd45_1372x932.png" alt="img" style="zoom:33%;" />

如果我们更新对解码器块的可视化表示，它现在会包含更多的 FFNN（每个对应一个专家）：

<img src="https://substackcdn.com/image/fetch/$s_!7Tla!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb97a8ac7-db97-497f-866d-10400729d51e_1248x764.png" alt="img" style="zoom:33%;" />

因此，解码器块在推理时可以根据需要使用多个 FFNN（即不同的专家）。

## 路由机制

现在我们已经有了一组专家，那模型是怎么知道该使用哪些专家的呢？

在专家模块之前，会加入一个**路由器（router）** （也称为**门控网络 gate network**），它是专门训练用于决定每个 token 应该交给哪一个专家来处理的。

### 路由器

路由器本身也是一个前馈神经网络（FFNN），它的作用是：根据每个 token 的输入内容，输出一组概率值，并据此选择最匹配的专家。

<img src="https://substackcdn.com/image/fetch/$s_!K5Xp!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Facc49abf-bc55-45fd-9697-99c9434087d0_864x916.png" alt="img" style="zoom:33%;" />

每个专家的输出会乘以对应的“门控值”（也就是刚才输出的概率），最终汇总后作为该层的输出。

路由器 + 一组专家（FFNN被选中的一小部分）= 构成了一个 **MoE 层（MoE Layer）** ：

<img src="https://substackcdn.com/image/fetch/$s_!U_7N!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa6fcabc6-78cd-477f-ac4e-2260cb06e230_1160x688.png" alt="img" style="zoom:33%;" />

一个 MoE 层通常有两种大小类型：稀疏MoE或者稠密MoE。

两者都使用路由器来选择专家，但稀疏 MoE 每次只选择少量专家参与计算，而稠密 MoE所有专家都会参与计算，但参与程度可能不同。

<img src="https://substackcdn.com/image/fetch/$s_!Ofa6!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F46aadf17-3afe-4c98-b57c-83b7b38918b2_1004x720.png" alt="img" style="zoom:33%;" />

例如：对一组 token，Dense MoE 会将每个 token 分发给所有专家，而 Sparse MoE 只会分发给其中少数几个。

目前大多数 LLM 的 MoE 实现中，几乎都是稀疏MoE，因为它能有效降低推理成本，非常适合大模型。

### 专家选择的过程

路由器其实是 MoE 中最核心的组件之一。它不仅决定**推理阶段**该选哪些专家，也决定**训练阶段**哪些专家要被更新。

最基础的做法是，将输入 token 表示向量 *x*，乘以一个路由权重矩阵 W：

<img src="https://substackcdn.com/image/fetch/$s_!rutb!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F58234ce0-bf96-49ab-b414-674a710a1c3c_1164x368.png" alt="img" style="zoom:33%;" />

对打分结果做 SoftMax，得到一个不同专家的概率分布 *G*(*x*)：

<img src="https://substackcdn.com/image/fetch/$s_!SC8j!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb888a32f-acef-4fff-9d4b-cc70e148a8f2_1164x384.png" alt="img" style="zoom:33%;" />

然后根据这些概率值，选出和输入最相关的专家。

最后把每个路由器的输出与对应的专家相乘，加权求和，得到该 token 的最终输出：

<img src="https://substackcdn.com/image/fetch/$s_!6sXB!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fe6e46ea4-dbd4-4cc4-aa2b-2c5474917f31_1164x464.png" alt="img" style="zoom:33%;" />

我们把所有步骤汇总起来，这就完成了一个 token 从路由器流向专家的完整过程：

<img src="https://substackcdn.com/image/fetch/$s_!7ajO!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd5d24a0b-2d78-4c69-b6fe-d75ba34bdd0c_2080x2240.png" alt="img" style="zoom:33%;" />





<img src="https://substackcdn.com/image/fetch/$s_!wqYK!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F3d1122aa-7248-47d0-8e01-caa941ce0aa9_2080x2240.png" alt="img" style="zoom:33%;" />

### 路由的挑战

不过，虽然这个机制看起来很简单，但实际上会出现一个问题，某些专家学得更快，导致路由器总是选它们。

<img src="https://substackcdn.com/image/fetch/$s_!ZTBl!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F9233733c-c152-428a-ae99-1ed185fc3d50_1164x660.png" alt="img" style="zoom:33%;" />

这样一来会导致一些问题 ，选择的专家不均匀，一些专家几乎不会被训练到，最终训练和推理时都容易出现偏倚和性能下降 的问题。

为了解决这个问题，我们引入了**负载均衡（load balancing）** 。它的目的是让每个专家在训练和推理中都能被**公平地使用**，避免某在某几个专家上过拟合。

## 负载均衡

为了让不同专家的使用更加平衡，我们需要关注的是**路由器（Router）** ，因为它决定了每一次选用哪些专家。

### KeepTopK 策略

一种常见的路由负载均衡方法是通过一个叫做KeepTopK的简单的扩展，引入可训练的高斯噪声，以防止模型总是选中同一批专家：

<img src="https://substackcdn.com/image/fetch/$s_!7VGM!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1b95b020-ae34-40f0-a5c4-9542343beea9_1164x412.png" alt="img" style="zoom:33%;" />

接下来，除了你想要激活的 Top-k 个专家（比如前 2 个），**其他专家的分数全部设为 -∞：**

<img src="https://substackcdn.com/image/fetch/$s_!9wtR!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F66bea40e-3fb0-4937-88d5-2852af456cf3_1164x488.png" alt="img" style="zoom:33%;" />

这样一来，在做 SoftMax 时，分数为 -∞ 的专家对应的概率就是 0，不会被选中 ：

<img src="https://substackcdn.com/image/fetch/$s_!MGk6!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F687d2279-1d8b-4af1-b55e-55d618ee877f_1164x496.png" alt="img" style="zoom:33%;" />

虽然现在也有很多好用的替代方案（`新的 MoE 路由算法`)，但KeepTopK 仍然是许多大型语言模型中常用的策略。
而且，它**可以加噪声，也可以不加噪声**，都可以使用。

#### Token Choice

KeepTopK 策略是每个 token 会被送到少数几个专家那里处理。这种方法被称为 *Token Choice* 。Top-1 路由每个 token 只交给一个专家处理。

<img src="https://substackcdn.com/image/fetch/$s_!fl3F!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fdf7a9988-d4c8-4b1b-a968-073a6b3bfc6a_1004x648.png" alt="img" style="zoom:33%;" />

Top-k 路由每个 token 被同时送给k个专家，然后再根据它们的输出加权合并 。

<img src="https://substackcdn.com/image/fetch/$s_!Ps3o!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb3f283f1-c359-4baf-8d01-8ebb2a90665f_1004x720.png" alt="img" style="zoom:33%;" />

这种方法的优点是：可以根据每个专家的权重贡献，**融合多个专家的知识**，更灵活。

#### 辅助损失

为了让训练过程中的专家使用更加平均，研究者在主损失之外，引入了一个*辅助损失（Auxiliary Loss）*（或者叫负载均衡损失（Load Balancing Loss））。

这个损失项的作用是强制每个专家在训练中有差不多的“重要性”。

辅助损失的第一步，是对整个 batch 中每个专家的路由概率值求和也就是统计在这个 batch 中，每个专家一共“被选中的概率”加起来是多少 ：

<img src="https://substackcdn.com/image/fetch/$s_!9DwH!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff3624da0-3137-42ba-95e8-88fcbddb5f9f_1108x288.png" alt="img" style="zoom:33%;" />

这样就得到了每个专家的一个*重要性分数（importance score）*，表示这个专家在整个 batch 中，在不考虑输入内容的情况下，有多大可能被选中。

接下来，我们可以利用这些重要性分数，计算一个指标*变异系数（Coefficient of Variation, CV）* ，表示各个专家的重要性差异有多大：

<img src="https://substackcdn.com/image/fetch/$s_!d3cI!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F94def8dc-2a65-4a02-855f-219f0df2a119_916x128.png" alt="img" style="zoom:33%;" />


如果 CV 很高，说明某些专家总被选中，而其他专家几乎没被用；

<img src="https://substackcdn.com/image/fetch/$s_!S63p!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fab71b90c-ba29-42a9-944b-3dee52fc5c32_916x372.png" alt="img" style="zoom:33%;" />

如果 CV 很低，说明所有专家被使用得差不多，这正是我们想要的“负载均衡”状态。

<img src="https://substackcdn.com/image/fetch/$s_!hc-d!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc5cb91ac-4aab-4eb5-80bf-84e2bd4dc576_916x324.png" alt="img" style="zoom:33%;" />

利用这个 **CV** 分数，我们可以在训练过程中不断更新**辅助损失**，使模型的优化目标之一就是尽可能降低 CV 值，从而让每个专家的使用重要性趋于一致：

<img src="https://substackcdn.com/image/fetch/$s_!vU-f!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff4aac801-af89-44e7-aaea-c57a55ff282c_916x312.png" alt="img" style="zoom:33%;" />

最后，这个辅助损失会作为一个单独的损失项，加入到整体训练目标中一起优化。

总结一下三步：

🧩 第一步：计算每个专家的“重要性分数

🧮 第二步：计算“变异系数

💡 第三步：优化目标 = 降低 CV

### 专家容量

模型中的不平衡，不仅体现在**被选中的专家不平均**，也体现在**发送给专家的 token 分布不均**。

比如：如果大量输入 token 都被不均衡地路由到了某一个专家（而其他专家几乎不会接收到 token），那可能会导致训练不充分：

<img src="https://substackcdn.com/image/fetch/$s_!rZmG!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F749eac8e-36e5-450f-a6fc-fbe48b7a1312_1004x484.png" alt="img" style="zoom:33%;" />

所以问题不只是“用了哪个专家”，而是“每个专家被用了多少次”。

为了解决这个问题，可以引入一个限制机制，限制每个专家一次最多能处理多少个 token，称之为*专家容量（Expert Capacity）*。一旦某个专家达到容量上限，剩下的 token 就会被路由给下一个候选专家：

<img src="https://substackcdn.com/image/fetch/$s_!44Xf!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fdf67563f-755a-47a7-bebc-c1ac81a01f8f_1004x568.png" alt="img" style="zoom:33%;" />

如果所有候选专家都已满，token 将不会被任何专家处理，而是跳过当前 MoE 层，进入下一层。这种情况被称为 *token overflow（token 溢出）* 。

<img src="https://substackcdn.com/image/fetch/$s_!CsiT!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fe92ce4c5-affa-454d-8fd2-4debf9a08ce2_1004x544.png" alt="img" style="zoom:33%;" />

### 用 Switch Transformer 简化 MoE

Switch Transformer是最早尝试解决 MoE 模型训练不稳定问题的 Transformer 架构之一（比如前面讲过的专家负载不均衡问题）。它简化了MoE大部分架构和训练过程，同时提高了训练稳定性。

#### 切换层

切换层（Switch Transformer） 是在T5（encoder-decoder 架构）基础上改造的模型，把传统 Transformer 中的 前馈神经网络（FFNN）层，换成了一个 Switching Layer（切换层）。这个切换层本质上就是稀疏 MoE 层，采用Top-1 路由策略，每个 token 只被路由到一个最合适的专家上。

<img src="https://substackcdn.com/image/fetch/$s_!SsAV!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F024d1788-9007-4953-9bf7-883da0db7f8d_1160x688.png" alt="img" style="zoom:33%;" />

路由器没有做任何特殊技巧来计算选择哪个专家，还是用输入 token 乘上每个专家的权重，然后做一次 softmax，（和我们前边说的一样，见**专家选择的过程**这一部分）。

<img src="https://substackcdn.com/image/fetch/$s_!Lk3v!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff0758a7f-e26b-44b9-9d75-80ac6caa9802_1104x384.png" alt="img" style="zoom:33%;" />

相比之前的 top-k 路由策略，这种结构（top-1路由）基于一个假设：每个 token 只需一个专家就够用了。

#### 专家容量因子

专家容量因子（Capacity Factor）是一个非常重要的超参数，它决定了每个专家最多能处理多少个 token。

Switch Transformer 对这一点做了扩展，它允许你设置专家容量因子（capacity factor），显式地控制专家容量。

<img src="https://substackcdn.com/image/fetch/$s_!Aqb-!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F22715139-3955-4e00-bed7-c45cffa52744_964x128.png" alt="img" style="zoom:33%;" />

专家容量的组成部分很简单：

<img src="https://substackcdn.com/image/fetch/$s_!FTrV!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff4b399c6-723b-4de6-94ca-7020cd1bb181_908x380.png" alt="img" style="zoom:33%;" />

如果你把容量因子设置得大一点，每个专家就能处理更多的 token。

<img src="https://substackcdn.com/image/fetch/$s_!U4FT!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F7fd2aea0-fddf-4a43-ac79-7c5e5194c115_1240x472.png" alt="img" style="zoom:33%;" />

但是，如果容量因子太大，则会浪费计算资源。相反，如果容量因子太小，则 Token 溢出会导致模型性能下降。如果设置得太小，就可能导致很多 token 找不到专家处理，变成前面说的token 溢出 。

#### 辅助损失

为了进一步避免 token 被丢弃，Switch Transformer 引入了一个更简单版本的辅助损失函数。

这个简化的损失函数不再计算变异系数（coefficient variation），而是衡量每个专家实际被分配到的 token 的比例与路由器为该专家分配的概率之间的关系：

<img src="https://substackcdn.com/image/fetch/$s_!ezjV!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F608da44a-7510-4ab6-97c9-e8ab212a567d_836x388.png" alt="img" style="zoom:33%;" />

由于我们的目标是将 token 在 *N* 个专家之间均匀地路由，因此希望向量 *P* 和 *f* 的值都接近 1/*N*。

*α* 是一个超参数，用于调整辅助损失项在训练过程中的重要性。值设得太高会影响主损失函数，而值设得太低则几乎无法起到负载均衡的作用。

## 视觉模型中的MoE

MoE 并不是语言模型专属的技术。视觉模型（如 ViT）也是基于 Transformer 架构，所以也具备使用 MoE 的潜力。

快速回顾一下，ViT（Vision Transformer）是一种将图像切分成 patch（图像块）然后像处理文本 token 一样对它们进行处理的架构。

<img src="https://substackcdn.com/image/fetch/$s_!zgwf!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F11b64fce-4069-4c73-995d-c3059fda0dcc_1300x828.png" alt="img" style="zoom:33%;" />

这些 patch（或 token）会被映射为 embedding（并加上额外的位置 embedding），然后送编码器中：

<img src="https://substackcdn.com/image/fetch/$s_!n9BH!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc0b2ea60-238b-446a-ab59-503efb6ca061_1228x1232.png" alt="img" style="zoom:33%;" />

这些 patch 一旦进入编码器，处理方法就跟token一样了，使得这种架构非常适合引入 MoE。

### Vision-MoE

Vision-MoE（V-MoE）是最早在图像模型中实现 MoE 的方案之一。它和我们上边说的一样，以 ViT 架构为基础，用稀疏 MoE 替代了编码器中的稠密 FFNN 层。

<img src="https://substackcdn.com/image/fetch/$s_!7CLL!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F10e9721d-4b3f-4062-ad72-97ffd1049077_1160x944.png" alt="img" style="zoom:33%;" />

相较于语言模型来说，ViT模型规模相对较小，但可以通过添加专家模块进行大规模扩展。

在图像处理时，经常会将其切分为许多 patch，所以每个专家会预设一个较小容量，以此降低硬件限制。但是预设容量太小也容易导致 patch 被丢弃（类似于 token 溢出）。

<img src="https://substackcdn.com/image/fetch/$s_!KVXp!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F219141c9-51ff-4d85-9f8a-c705c6e9ece4_1720x744.png" alt="img" style="zoom:33%;" />

为了在保持低容量的同时尽可能减少关键 patch 的丢弃 ，网络会为每个 patch 设置重要性分数，优先处理重要性更高的 patch，这样的话，溢出的那些token通常是不重要的，这种方法被称为Batch优先路由（Priority Routing）。

<img src="https://substackcdn.com/image/fetch/$s_!L60n!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa0ef5323-4b4c-4ee7-8a53-51fbe4213283_1720x772.png" alt="img" style="zoom:33%;" />

因此，即使一部分token消失，我们依旧可以看到重要的 patch 被优先路由到专家。

<img src="https://substackcdn.com/image/fetch/$s_!Jvhj!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F65f972b9-640b-4a76-b77d-2d2ef1b40609_1736x420.png" alt="img" style="zoom:33%;" />

优先路由机制使模型即使处理较少的 patch，也能聚焦在最重要的信息上。

### 从稀疏 MoE 到软 MoE

在 V-MoE 中，优先评分机制有助于区分重要和不重要的 patch。然而，patch 是需要分配到各个专家处理的，没被处理的 patch 中的信息就会丢失。

Soft-MoE 目的是通过混合 patch，将离散的 patch（token）分配方式转变为更“软”的分配方式。

第一步是将输入 *x*（即 patch 的 embedding）乘以一个可学习矩阵 Φ。这样我们会得到述了某个 token 与某个专家的关联程度的路由信息（router information\）。

<img src="https://substackcdn.com/image/fetch/$s_!VZYh!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F644c0c1c-24d3-491b-a9a2-fdd9658ad589_1032x516.png" alt="img" style="zoom:33%;" />

接着对路由信息矩阵按列做 softmax 操作，我们就可以更新每个 patch 的 embedding。

<img src="https://substackcdn.com/image/fetch/$s_!XWFx!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F6c3187d8-5bf2-4c73-8c22-547107fe1152_1032x456.png" alt="img" style="zoom:33%;" />

更新后的 patch embedding 本质上是所有 patch embedding 的加权平均。

<img src="https://substackcdn.com/image/fetch/$s_!TaCn!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F7cfeb30a-1b04-4b9a-8f5a-d3c5d47e6499_1376x400.png" alt="img" style="zoom:33%;" />

可视化之后可以看到，所有 patch 就好像被“混合”了。这些组合后的 patch 会被发送给每个专家，在生成输出之后，再次与路由矩阵相乘。

<img src="https://substackcdn.com/image/fetch/$s_!cbz7!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F86020d75-c881-4418-82a6-a228f091abe8_808x844.png" alt="img" style="zoom:33%;" />

路由矩阵既在 token 层面影响输入，也在专家层面影响输出。

因此，我们得到的是“软”的 patch/token，而不是离散的输入。

## 激活参数 vs 稀疏参数：以 Mixtral 8x7B 为例

MoE 火起来的一个关键优势是节省计算资源、。虽然某个 MoE 模型看起来好像有更多参数（稀疏参数），但在推理时只激活其中一部分活动参数（active parameters）。

虽然 MoE 模型要加载更多的参数（稀疏参数），但推理时只激活少数几个专家（即活动参数）。

<img src="https://substackcdn.com/image/fetch/$s_!AMdO!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fe1fd47bb-9ced-42e4-8f6c-536f7a65fbf7_1376x1252.png" alt="img" style="zoom:33%;" />

换句话说，我们仍然需要将整个模型（包括所有专家）加载到设备上（稀疏参数），但在推理时，只需使用其中的一部分（激活参数）。MoE 模型在加载时需要更大的显存（VRAM），但推理运行得更快。

也就是说MoE 模型需要更多的显存（VRAM）来加载所有专家，但推理更快，因为每次只用一小部分专家。

我们来看一个例子：Mixtral 8x7B，来比较稀疏参数和互动参数的数量。



<img src="https://substackcdn.com/image/fetch/$s_!j84T!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fcc3d48d5-8afc-4477-af98-5817b1a145ae_1376x988.png" alt="img" style="zoom:33%;" />

在这个模型中，每个专家的大小是 5.6B，而不是 7B。

<img src="https://substackcdn.com/image/fetch/$s_!824b!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1dfd20b4-d3b7-433b-8072-2e67fc70afaa_1376x544.png" alt="img" style="zoom:33%;" />



我需要加载 8×5.6…B ≈ 45.1B 的参数（原文为 46.7B，是因为加上了共享参数），但在推理时实际只使用了 2×5.6…B ≈ 11.3B 的参数。

因此，Mixtral 8x7B 表面上是一个“大模型”，但每次推理只用了“小模型”的算力。这正是 MoE 架构兼顾“能力”和“效率”的核心所在。

## 结论

我们的专家混合（Mixture of Experts）之旅到这里就告一段落了。 希望这篇文章能帮助你更好地理解这种有趣技术的潜力所在。如今，几乎所有模型中都至少包含一个MoE变体， 看起来，**MoE 已经不再是尝试性的选择，而是将长期存在的关键技术之一**。



# Resources

Hopefully, this was an accessible introduction to Mixture of Experts. If you want to go deeper, I would suggest the following resources:

- [This](https://arxiv.org/pdf/2209.01667) and [this](https://arxiv.org/pdf/2407.06204) paper are great overviews of the latest MoE innovations.
- The paper on [expert choice routing](https://arxiv.org/pdf/2202.09368) that has gained some traction.
- A [great blog post](https://cameronrwolfe.substack.com/p/conditional-computation-the-birth) going through some of the major papers (and their findings).
- A similar [blog post](https://brunomaga.github.io/Mixture-of-Experts) that goes through the timeline of MoE.
