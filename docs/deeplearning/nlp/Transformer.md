# Transformer

self-attention有时候也被称为intra-attention,是在单个句子不同位置上做的attention，并得到序列的一个表示。它能够很好的应用到很多任务中，包括阅读理解、摘要、文本蕴含，以及独立于任务的句子表示。

端到端的网络一般都是基于循环注意力机制而不是序列对齐循环，并且已经有证据表明在简单语言问答和语言建模任务上表现很好。据我们所知，Transformer是第一个完全依靠Self-attention而不使用序列对齐的RNN或卷积的方式来计算输入输出表示的转换模型。

## 模型结构

目前大部分比较热门的神经序列转换模型都有Encoder-Decoder结构。Encoder将输入序列 (x1,…,xn) 映射到一个连续表示序列 z=(z1,…,zn)。

对于编码得到的z，Decoder每次解码生成一个符号，直到生成完整的输出序列:(y1,…,ym) 。对于每一步解码，模型都是自回归的，即在生成下一个符号时将先前生成的符号作为附加输入。

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        """这里的memory指的是什么？指的是encoder的输出，即作为decoder第二个sublayer的K,V"""
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)
```

transformer的整体结构如下图所示，在encoder和decoder中都使用了self-attention，point-wise和全连接层。encoder和decoder的大致结构分别如下图的左半部分和右半部分所示：

![](http://nlp.seas.harvard.edu/images/the-annotated-transformer_14_0.png)

## encoder和decoder

### encoder

encoder由N=6个相同的层组成。

```python
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
```

在每个layer包含两个sublayer，分别为multi-head self-attention mechanism和fully connected feed-forward network，这两个sublayer都分别使用使用了残差连接(residual connection)和归一化。

```python
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
```

每个sublayer的输出为LayerNorm(x+Sublayer(x))，其中Sublayer(x)由子层自动实现的函数。在每个子层的输出上使用dropout，然后将进行归一化并作为下一sublayer的输入。

```python
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
```

每个layer都由两个sublayer组成。第一个sublayer实现了“多头”的self-attention，第二个sublayer则是一个简单的position-wise的全连接前馈网络。

```python
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
```

### decoder

decoder 也是由n=6个相同层组成

```python
class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
```

除了每个编码层中的两个子层外，解码器还插入了第三个子层，用于对编码器栈的输出实行”多头”的Attention。与编码器类似，每个子层两端使用残差连接，然后进行层的规范化处理。

```python
class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)
```

我们还修改解码器中的self-attention子层以防止当前位置attend到后续位置。这种masked的attention是考虑到输出embedding会偏移一个位置，确保了生成位置i的预测时，仅依赖小于i的位置处的已知输出，相当于把后面不该看到的信息屏蔽掉。

```python
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
```

下面的attention mask图显示了允许每个目标词(行)查看的位置(列)。在训练期间，当前解码位置的词不能attend到后续位置的词。

```python
plt.figure(figsize=(5,5))
plt.imshow(subsequent_mask(20)[0])
<matplotlib.image.AxesImage at 0x7f86872b90f0>
```

[![png](https://jozeelin.github.io/2019/10/21/The-Annotated-Transformer-Harvard/image/transformer_23_1.png)](https://jozeelin.github.io/2019/10/21/The-Annotated-Transformer-Harvard/image/transformer_23_1.png)python

## attention

attention函数可以将query和一组key-value对映射到输出，其中query、key、value和输出都是向量。输出是值的加权和，其中分配给每个value的权重由query与相应key的兼容函数计算。

我们称这种特殊的attention机制为”Scaled dot-product attention”。输入包含维度为dk的query和key，以及维度为dv的value。

首先计算query与各个key的点积，然后将每个点积除以‾‾√dk，最后使用softmax函数来获得value的权重。

[![1-2](http://nlp.seas.harvard.edu/images/the-annotated-transformer_33_0.png)](http://nlp.seas.harvard.edu/images/the-annotated-transformer_33_0.png)

在具体实现时，我们可以以矩阵的形式进行**并行运算**，这样能加速运算过程。具体来说，将所有的Query、key和value向量分别组合称矩阵Q,K和V，这样输出矩阵可以表示为：

```python
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
```

两种最常用的Attention函数是加和Attention和点积(乘积)Attention，我们的算法与点积Attention很类似，但是1dk√1dk的比例因子不同。加和attention使用具有单个隐藏层的前馈网络来计算兼容函数。虽然两种方法理论上的复杂度是相似的，但在实践中，点积attention的运算会更快一些，也更节省空间，因为它可以使用高效的矩阵乘法算法来实现。

虽然对于较小的 $d*k$，这两种机制的表现相似，但对于较大的，这两种机制的表现相似，但对于较大的d_k来说，不使用它进行缩放的情况下，加型Attention要优于点积Attention。我们怀疑，对于较大的来说，不使用它进行缩放的情况下，加型Attention要优于点积Attention。我们怀疑，对于较大的d_k，点积大幅增大，将softmax函数推向具有极小梯度的区域(为了阐明点积变大的原因，假设q和k是独立的随机变量，平均值为0，方差1，这样他们的点积为，点积大幅增大，将softmax函数推向具有极小梯度的区域(为了阐明点积变大的原因，假设q和k是独立的随机变量，平均值为0，方差1，这样他们的点积为q.k = \sum*{i=1}^{d_k}q_kk_i,同样是均值为0，方差为,同样是均值为0，方差为d_k)。为了抵消这种影响，我们用)。为了抵消这种影响，我们用\frac{1}{\sqrt{d_k}}$来缩放点积。

[![1-3](http://nlp.seas.harvard.edu/images/the-annotated-transformer_38_0.png)](http://nlp.seas.harvard.edu/images/the-annotated-transformer_38_0.png)

多头机制能让模型考虑到不同位置的Attention，另外“多头”Attention可以在不同的子空间表示不一样的关联关系，使用单个head的attention一般达不到这种效果。

```tex
\mathrm{MultiHead}(Q,K,V) = \mathrm{Concat}(head_1,\dots,head_h)W^O
```

$W*i^Q \in \mathbb{R}^{d*{model} \times d_k}$

$W*i^K \in \mathbb{R}^{d*{model} \times d_k}$

$W*i^V \in \mathbb{R}^{d*{model} \times d_v}$

$W^O \in \mathbb{R}^{hd*v\times d*{model}}$

我们的工作中使用h=8个head并行的attention，对每一个head来说有$d*k=d_v=d*{model}/h=64$，总计算量与完整维度的单个Head的Attention很相近。

```python
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
```

## Attention在模型中的应用

transformer中以三种不同的方式使用了“多头”Attention：

1. 在”Encoder-Decoder Attention”层，Query来自先前的编码器层，并且Key和Value来自Encoder的输出。Decoder中的每个位置Attend输入序列中的所有位置，这与Seq2Seq模型中的经典的Encoder-Decoder Attention机制一致。
2. Encoder中的Self-attention层。在self-attention层中，所有的key、value和query都来同一个地方，这里都来自encoder的前一层的输出。encoder中当前层的每个位置都能attend到前一层的所有位置。
3. 类似的，解码器中的self-attention层允许解码器中的每个位置attend当前解码位置和它前面的所有位置。这里需要屏蔽解码器中向右的信息流以保持自回归属性。具体的实现方式是在缩放后的点积Attention中，屏蔽(设为负无穷)softmax的输入中所有对应着非法连接的Value。

## position-wise前馈网络

除了Attention子层之外，Encoder和Decoder中的每个层都包含一个全连接前馈网络，分别地应用于每个位置。其中包括两个线性变换，然后使用ReLU作为激活函数。

FFN(x)=max(0,xW1+b1)W2+b2FFN(x)=max(0,xW1+b1)W2+b2

虽然线性变换在不同位置上是相同的，但是他们在层与层之间使用不同的参数。这其实就是相当于使用了2个1x1的卷积核。这里设置输入和输出的维数为$d*{model}=512，内层的维度为，内层的维度为d*{ff} = 2048$。
