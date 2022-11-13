

# Vision Transformer

ViT（vision transformer）是Google在2020年提出的直接将transformer应用在图像分类的模型，后面很多的工作都是基于ViT进行改进的。ViT的思路很简单：直接把图像分成固定大小的patchs，然后通过线性变换得到patch embedding，这就类比NLP的words和word embedding，由于transformer的输入就是a sequence of token embeddings，所以将图像的patch embeddings送入transformer后就能够进行特征提取从而分类了。ViT模型原理如下图所示，其实ViT模型只是用了transformer的Encoder来提取特征（原始的transformer还有decoder部分，用于实现sequence to sequence，比如机器翻译）。下面将分别对各个部分做详细的介绍。

![img](https://pic2.zhimg.com/v2-6d4aa99139aa6f658e4ff6a3d980f291_b.jpg)

## **Patch Embedding**

对于ViT来说，首先要将原始的2-D图像转换成一系列1-D的 [patch embeddings](https://www.zhihu.com/search?q=patch+embeddings&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"356155277"})，这就好似NLP中的word embedding。输入的2-D图像记为![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf+x%5Cin+%5Cmathbb%7BR%7D%5E%7BH%5Ctimes+W+%5Ctimes+C%7D)，其中![[公式]](https://www.zhihu.com/equation?tex=H)和![[公式]](https://www.zhihu.com/equation?tex=W)分别是图像的高和宽，而![[公式]](https://www.zhihu.com/equation?tex=C)为通道数对于RGB图像就是3。如果要将图像分成大小为![[公式]](https://www.zhihu.com/equation?tex=P%5Ctimes+P)的patchs，可以通过reshape操作得到a sequence of patchs：![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf+x_p%5Cin+%5Cmathbb%7BR%7D%5E%7BN%5Ctimes%28P%5E2%5Ccdot+C%29%7D)，图像共切分为![[公式]](https://www.zhihu.com/equation?tex=N%3DHW%2FP%5E2)个patchs，这也就是sequence的长度了，注意这里直接将patch拉平为1-D，其特征大小为![[公式]](https://www.zhihu.com/equation?tex=P%5E2%5Ccdot+C)。然后通过一个简单的线性变换将patchs映射到![[公式]](https://www.zhihu.com/equation?tex=D)大小的维度，这就是patch embeddings：![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf+%7Bx%27_%7Bp%7D%7D%5Cin+%5Cmathbb%7BR%7D%5E%7BN%5Ctimes+D%7D)，在实现上这等同于对![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf+x)进行一个![[公式]](https://www.zhihu.com/equation?tex=P%5Ctimes+P)且stride为![[公式]](https://www.zhihu.com/equation?tex=P)的卷积操作（虽然等同，但是ViT其实是不包含任何卷积操作的)，下面是具体的实现代码：

```python
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
```

## **Position Embedding**

除了patch embeddings，模型还需要另外一个特殊的 position embedding。transformer和 CNN 不同，需要position embedding来编码tokens的位置信息，这主要是因为self-attention是permutation-invariant，即打乱sequence里的tokens的顺序并不会改变结果。如果不给模型提供patch的位置信息，那么模型就需要通过patchs的语义来学习拼图，这就额外增加了学习成本。ViT论文中对比了几种不同的position embedding方案(如下），最后发现如果不提供positional embedding效果会差，但其它各种类型的positional embedding效果都接近，这主要是因为ViT的输入是相对较大的patchs而不是pixels，所以学习位置信息相对容易很多。

- 无 positional embedding
- 1-D positional embedding：把2-D的patchs看成1-D序列
- 2-D positional embedding：考虑patchs的2-D位置（x, y）
- Relative positional embeddings：patchs的相对位置

transformer 原论文中是默认采用固定的 positional embedding，但ViT中默认采用学习（训练的）的1-D positional embedding，在输入transformer的encoder之前直接将 patch embeddings 和 positional embedding 相加:

```python
# 这里多1是为了后面要说的class token，embed_dim即patch embed_dim
self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

# patch emded + pos_embed
x = x + self.pos_embed
```

### **Class Token**

除了patch tokens，ViT借鉴BERT还增加了一个特殊的class token。后面会说，transformer的encoder输入是a sequence patch embeddings，输出也是同样长度的a sequence patch features，但图像分类最后需要获取image feature，简单的策略是采用pooling，比如求patch features的平均来获取image feature，但是ViT并没有采用类似的pooling策略，而是直接增加一个特殊的class token，其最后输出的特征加一个linear classifier就可以实现对图像的分类（ ViT 的 pre-training时是接一个MLP head），所以输入ViT的sequence长度是![[公式]](https://www.zhihu.com/equation?tex=N%2B1)。class token对应的embedding在训练时随机初始化，然后通过训练得到，具体实现如下：

```python
# 随机初始化
self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

# Classifier head
self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

# 具体forward过程
x = self.patch_embed(x)
cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
x = torch.cat((cls_tokens, x), dim=1)
x = x + self.pos_embed
```

注意：

- 224 *224 的图像经过 Patch_embed（）之后，变成了 B * 196 * 768  的序列；
- B * 196 * 768  还需要增加一个 cls_token,  实现对图像的分类，   所以 输入是 B * （196  +1）* 768
- 所以传进  **Transformer Encoder** 的序列是 B * （196  +1）* 768

## **Transformer Encoder**

transformer最核心的操作就是self-attention，其实attention机制很早就在NLP和CV领域应用了，比如带有attention机制的[seq2seq模型](https://www.zhihu.com/search?q=seq2seq模型&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"356155277"})，但是transformer完全摒弃RNN或LSTM结构，直接采用attention机制反而取得了更好的效果：attention is all you need！简单来说，attention就是根据当前查询对输入信息赋予不同的权重来聚合信息，从操作上看就是一种“加权平均”。attention中共有3个概念：query, key 和 value，其中 key 和 value 是成对的，对于一个给定的query向量![[公式]](https://www.zhihu.com/equation?tex=q%5Cin+%5Cmathbb%7BR%7D%5E%7Bd%7D)，通过内积计算来匹配k个[key向量](https://www.zhihu.com/search?q=key向量&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"356155277"})（维度也是d，堆积起来即[矩阵](https://www.zhihu.com/search?q=矩阵&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"356155277"})![[公式]](https://www.zhihu.com/equation?tex=K%5Cin+%5Cmathbb%7BR%7D%5E%7Bk%5Ctimes+d%7D)），得到的内积通过softmax来归一化得到k个权重，那么对于query其attention的输出就是k个key向量对应的value向量（即矩阵![[公式]](https://www.zhihu.com/equation?tex=V%5Cin+%5Cmathbb%7BR%7D%5E%7Bk%5Ctimes+d%7D)）的[加权平均值](https://www.zhihu.com/search?q=加权平均值&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"356155277"})。对于一系列的N个query（即矩阵![[公式]](https://www.zhihu.com/equation?tex=Q%5Cin+%5Cmathbb%7BR%7D%5E%7BN%5Ctimes+d%7D))，可以通过矩阵计算它们的attention输出：

![[公式]](https://www.zhihu.com/equation?tex=Attention%28Q%2C+K%2C+V%29+%3D+Softmax%28%5Cfrac%7BQK%5ET%7D%7B%5Csqrt%7Bd_k%7D%7D%29V+%5C%5C)

这里的![[公式]](https://www.zhihu.com/equation?tex=%5Csqrt%7Bd_k%7D)为缩放因子以避免[点积](https://www.zhihu.com/search?q=点积&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"356155277"})带来的方差影响。上述的Attention机制称为**Scaled dot product attention**，其实attention机制的变种有很多，但基本原理是相似的。如果![[公式]](https://www.zhihu.com/equation?tex=Q%2CK%2CV)都是从一个包含![[公式]](https://www.zhihu.com/equation?tex=N)个向量的sequence（![[公式]](https://www.zhihu.com/equation?tex=X%5Cin+%5Cmathbb%7BR%7D%5E%7BN%5Ctimes+D%7D)）通过线性变换得到：![[公式]](https://www.zhihu.com/equation?tex=Q%3DXW_Q%2CK%3DXW_K%2CV%3DXW_V)那么此时就变成了**self-attention**，这个时候就有![[公式]](https://www.zhihu.com/equation?tex=N)个（key,value）对，那么![[公式]](https://www.zhihu.com/equation?tex=k%3DN)。self-attention是transformer最核心部分，self-attention其实就是输入向量之间进行相互attention来学习到新特征。

更进一步，transformer采用的是**multi-head self-attention (MSA）**，所谓的MSA就是采用定义h个attention heads，即采用h个self-attention应用在输入sequence上，在操作上可以将sequence拆分成h个size为![[公式]](https://www.zhihu.com/equation?tex=N%5Ctimes+d)的sequences，这里![[公式]](https://www.zhihu.com/equation?tex=D%3Dhd)，h个不同的heads得到的输出concat在一起然后通过线性变换得到最终的输出，size也是![[公式]](https://www.zhihu.com/equation?tex=N%5Ctimes+D)：

![[公式]](https://www.zhihu.com/equation?tex=MSA%28X%29+%3D+Concat%28head_1%2C+...%2C+head_h%29+W%5EO%2C+head_i%3DSA%28XW_i%5EQ%2C+XW_i%5EK%2C+XW_i%5EV%29+%5C%5C)

![img](https://pic1.zhimg.com/v2-dd2b11273d3974c81d63e418bbdadbf8_b.jpg)

MSA的计算量是和![[公式]](https://www.zhihu.com/equation?tex=N%5E2)成正相关的，所以ViT的输入是patch embeddings，而不是pixel embeddings，这有计算量上的考虑。在实现上，MSA是可以并行计算各个head的，具体代码如下：


```python
import torch
import torch.nn as nn


class Attention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(
            0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


if __name__ == '__main__':
    x = torch.rand(32, 196, 768)
    attn = Attention(dim=768)
    out = attn(x)
    print(out.shape)
```

>  @ 和 * 代表矩阵的两种相乘方式：@ 表示常规的数学上定义的矩阵相乘；* 表示两个矩阵对应位置处的两个元素相乘

> transpose（） 交换 tensor 的维度， 只能交换两个维度

### 数据流程

假设 输入是 B * N *C = 32 * 196 * 768,  head 的个数为 8,

- 24 行,   输入  32 * 196 * 768
- 25 行,   32 * 196 * 768  ==>  32 * 196 * (768  * 3 )  ==>  32  * 196 * 3 * 8 * 96  ==>  3 *32  * 8 * 196   *96
- 27 行,   q,k ,v 的shape 都是  32  * 8 * 196   *96
- 30 行,   矩阵乘法,  q * k/sqrt(d) :   32  * 8 * 196   * 196
- 31 行,  计算  softmax,   shape : 32  * 8 * 196   * 196
- 34 行,   计算  softmax(qk/d) * v ,  shape:   32  * 8 * 196   *  96  ==>  32  * 196  * 8 *  96 ==>   32  * 196 * 768

这样， 对于 B * N *C = 32 * 196 * 768 大小的输入，  送入self-attention就能到同样 size 的 sequence 输出，只不过特征改变了。



在transformer中，MSA后跟一个FFN（Feed-forward network），这个FFN包含两个FC层，第一个FC层将特征从维度![[公式]](https://www.zhihu.com/equation?tex=D)变换成![[公式]](https://www.zhihu.com/equation?tex=4D)，后一个FC层将特征从维度![[公式]](https://www.zhihu.com/equation?tex=4D)恢复成![[公式]](https://www.zhihu.com/equation?tex=D)，中间的非线性激活函数采用GeLU，其实这就是一个MLP，具体实现如下：

```python
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
```

那么一个完成transformer encoder block就包含一个MSA后面接一个FFN，其实MSA和FFN均包含和ResNet一样的skip connection，另外MSA和FFN后面都包含layer norm层，具体实现如下：

```python
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
```

## **ViT**

对于ViT模型来说，就类似CNN那样，不断堆积transformer encoder blocks，最后提取class token对应的特征用于图像分类，论文中也给出了模型的公式表达，其中

（1）就是提取图像的patch embeddings，然后和class token对应的embedding拼接在一起并加上positional embedding；

（2）是MSA，而（3）是MLP，（2）和（3）共同组成了一个transformer encoder block，共有![[公式]](https://www.zhihu.com/equation?tex=L)层；

（4) 是对class token对应的输出做layer norm，然后就可以用来图像分类。

![img](https://pic1.zhimg.com/v2-cb632e9df1dbc49e379799a0417e9b34_b.jpg)

除了完全无卷积的ViT模型外，论文中也给出了Hybrid Architecture，简单来说就是先用CNN对图像提取特征，从CNN提取的特征图中提取patch embeddings，CNN已经将图像降采样了，所以[patch size](https://www.zhihu.com/search?q=patch+size&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"356155277"})可以为![[公式]](https://www.zhihu.com/equation?tex=1%5Ctimes+1)。

ViT模型的超参数主要包括以下，这些超参数直接影响模型参数以及计算量：

1. Layers：block的数量；
2. Hidden size D：隐含层特征，D在各个block是一直不变的；
3. MLP size：一般设置为4D大小；
4. Heads：MSA中的heads数量；
5. Patch size：模型输入的patch size，ViT中共有两个设置：14x14和16x16，这个只影响计算量；



类似BERT，ViT共定义了3中不同大小的模型：Base，Large和Huge，其对应的模型参数不同，如下所示。如ViT-L/16指的是采用Large结构，输入的patch size为16x16。

![img](https://pic1.zhimg.com/v2-bf9b9ae81389a370f890b0e742de7938_b.jpg)



### **模型效果**

ViT并不像CNN那样具有inductive bias，论文中发现如果如果直接在ImageNet上训练，同level的ViT模型效果要差于ResNet，但是如果在比较大的数据集上petraining，然后再finetune，效果可以超越ResNet。比如ViT在Google私有的300M JFT数据集上pretrain后，在ImageNet上的最好Top-1 acc可达88.55%，这已经和ImageNet上的SOTA相当了（Noisy Student EfficientNet-L2效果为88.5%，Google最新的SOTA是Meta Pseudo Labels，效果可达90.2%）：

![img](https://pic1.zhimg.com/v2-c3379ed3e3fceb3776c9d8176937f738_b.jpg)

那么ViT至少需要多大的数据量才能和CNN旗鼓相当呢？这个论文也做了实验，结果如下图所示，从图上所示这个预训练所使用的数据量要达到100M时才能显示ViT的优势。transformer的一个特色是它的scalability：当模型和数据量提升时，性能持续提升。在大数据面前，ViT可能会发挥更大的优势。



![img](https://pic3.zhimg.com/v2-5486d37ee0306362fe5baa5188635656_b.jpg)
