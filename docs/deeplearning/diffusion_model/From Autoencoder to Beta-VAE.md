

# 从自动编码器到 Beta-VAE

Autocoder 的发明是为了使用中间有一个 narrow bottleneck layer 的神经网络模型来重建高维数据（这可能不适用于[Variational Autoencoder](https://lilianweng.github.io/posts/2018-08-12-vae/#vae-variational-autoencoder)，我们将在后面的章节中详细研究它）。一个很好的副产品是降维：bottleneck layer 捕获压缩的latent encoding。这种低维表示可以在各种应用（例如，搜索）中用作嵌入向量，帮助数据压缩，或揭示潜在的数据生成因素。

## 符号

|                 Symbol                  |                             Mean                             |
| :-------------------------------------: | :----------------------------------------------------------: |
|              $\mathcal{D}$              | 数据集，$\mathcal{D} = \{ \mathbf{x}^{(1)}, \mathbf{x}^{(2)}, \dots, \mathbf{x}^{(n)} \}$, 包含n数据样本；|$D$\|=n. |
|           $\mathbf{x}^{(i)}$            | 每个数据点都是一个$d$维向量，$\mathbf{x}^{(i)} = [x^{(i)}_1, x^{(i)}_2, \dots, x^{(i)}_d]$ |
|                   $x$                   |    来自数据集的一个数据样本，$\mathbf{x} \in \mathcal{D}$    |
|              $\mathbf{x}’$              |                   $\mathbf{x}$的重构版本.                    |
|          $\tilde{\mathbf{x}}$           |                    $\mathbf{x}$损坏的版本                    |
|              $\mathbf{z}$               |             在bottleneck layer 学习到的压缩向量              |
|               $a_j^{(l)}$               |       激活函数为 $j$-th 的第一个神经元 $l$-th 隐藏层。       |
|              $g_{\phi}(.)$              |                 参数化的**编码**函数$\phi$.                  |
|             $f_{\theta}(.)$             |                参数化的**解码**函数$\theta$.                 |
|  $q_{\phi}(\mathbf{z}\vert\mathbf{x})$  |           估计后验概率函数，也称为**概率编码器**。           |
| $p_{\theta}(\mathbf{x}\vert\mathbf{z})$ | 给定潜在编码生成真实数据样本的可能性，也称为**概率解码器**。 |

# 自编码器

**自编码**器是一种神经网络，旨在以无监督的方式学习恒等函数以重建原始输入，同时压缩过程中的数据，从而发现更高效和压缩的表示。这个想法起源于[20 世纪 80 年代](https://en.wikipedia.org/wiki/Autoencoder)，后来由[Hinton & Salakhutdinov 于 2006 年](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.459.3788&rep=rep1&type=pdf)发表的开创性论文得到推广。

它由两个网络组成：

- *编码器*网络：它将原始的高维输入变换成潜在的低维编码。输入尺寸大于输出。
- *解码器*网络：解码器网络从编码中恢复数据，可能具有越来越大的输出层。

![img](https://lilianweng.github.io/posts/2018-08-12-vae/autoencoder-architecture.png)

>  图 1. 自动编码器模型架构图。

编码器网络本质上完成了[降维](https://en.wikipedia.org/wiki/Dimensionality_reduction)，就像我们使用主成分分析（PCA）或矩阵分解（MF）一样。此外，自动编码器针对数据重建在编码层面进行了显式优化。一个好的中间表示不仅可以捕获隐变量，而且有利于完整的[解压](https://ai.googleblog.com/2016/09/image-compression-with-neural-networks.html)过程。

该模型包含由$\phi$参数化的编码器函数$g(.)$， 和由$\theta$参数化的解码器函数$f(.)$.  从输入$x$在在bottleneck layer学习到的低维编码 $\mathbf{z} = g_\phi(\mathbf{x})$，  重建的输入是$\mathbf{x}' = f_\theta(g_\phi(\mathbf{x}))$.

参数$(\theta, \phi)$一起学习以输出与原始输入相同的重构数据样本，$\mathbf{x} \approx f_\theta(g_\phi(\mathbf{x}))$，或者换句话说，学习恒等函数。有各种度量来量化两个向量之间的差异，例如激活函数为 sigmoid 时的交叉熵，或者像 MSE 一样简单的损失：
$$
L_\text{AE}(\theta, \phi) = \frac{1}{n}\sum_{i=1}^n (\mathbf{x}^{(i)} - f_\theta(g_\phi(\mathbf{x}^{(i)})))^2
$$

# 降噪自动编码器

由于自动编码器学习恒等函数，当网络参数多于数据点数时，我们面临“过度拟合”的风险。

为了避免过度拟合并提高鲁棒性， Denoising **Autoencoder** (Vincent et al. 2008) 提出了对基本自动编码器的修改。通过以随机方式向输入向量的某些值添加噪声或掩盖输入向量的某些值来部分破坏输入，$\tilde{\mathbf{x}} \sim \mathcal{M}_\mathcal{D}(\tilde{\mathbf{x}} \vert \mathbf{x})$.  然后训练模型以恢复原始输入（注意：不是损坏的输入）。
$$
\begin{aligned}
\tilde{\mathbf{x}}^{(i)} &\sim \mathcal{M}_\mathcal{D}(\tilde{\mathbf{x}}^{(i)} \vert \mathbf{x}^{(i)})\\
L_\text{DAE}(\theta, \phi) &= \frac{1}{n} \sum_{i=1}^n (\mathbf{x}^{(i)} - f_\theta(g_\phi(\tilde{\mathbf{x}}^{(i)})))^2
\end{aligned}
$$
其中$\mathcal{M}_\mathcal{D}$定义了从真实数据样本到噪声或损坏样本的映射。

![img](https://lilianweng.github.io/posts/2018-08-12-vae/denoising-autoencoder-architecture.png)

> 图 2. 去噪自动编码器模型架构图。

这种设计的动机是，即使视图被部分遮挡或损坏，人类也可以轻松识别物体或场景。为了“修复”部分损坏的输入，去噪自动编码器必须发现并捕获输入维度之间的关系，以便推断缺失的部分。

对于具有高冗余度的高维输入（如图像），模型可能依赖于从许多输入维度的组合中收集的证据来恢复去噪版本，而不是过度拟合一个维度。*这为学习强大*的潜在表示奠定了良好的基础。

噪声由随机映射$\mathcal{M}_\mathcal{D}(\tilde{\mathbf{x}} \vert \mathbf{x})$控制，并且它并不特定于特定类型的损坏过程（即掩蔽噪声、高斯噪声、椒盐噪声等）。自然地，破坏过程可以配备先验知识.

在原始 DAE 论文的实验中，噪声是这样应用的：随机选择固定比例的输入维度，并将它们的值强制为 0。听起来很像 dropout，对吧？好吧，去噪自动编码器是在 2008 年提出的，比 dropout 论文早了 4 年（[Hinton，等人，2012 年](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)）；）

# 稀疏自动编码器

**稀疏自动编码**器对隐藏单元激活应用“稀疏”约束，以避免过度拟合并提高稳健性。它强制模型只有少量隐藏单元同时被激活，或者换句话说，一个隐藏神经元在大部分时间应该是不活动的。

回想一下，常见的[激活函数](http://cs231n.github.io/neural-networks-1/#actfun)包括 sigmoid、tanh、relu、leaky relu 等。神经元在值接近 1 时激活，在值接近 0 时不激活。

假设在$l$-th 隐藏层有$s_l$的神经元，这一层中的第$j-th$ 个神经元的激活函数被标记为$a^{(l)}_j(.), $$j=1, \dots, s_l$. 该神经元的激活比例$\hat{\rho}_j$是一个较小数字$\rho$, 称为*稀疏参数*; 一个常见的配置是$\rho = 0.05$.
$$
\hat{\rho}_j^{(l)} = \frac{1}{n} \sum_{i=1}^n [a_j^{(l)}(\mathbf{x}^{(i)})] \approx \rho
$$
该约束是通过在损失函数中添加惩罚项来实现的。KL 散度$D_\text{KL}$测量两个伯努利分布之间的差异，一个具有均值$\rho$另一个均值是$\hat{\rho}_j^{(l)}$. 超参数$\beta$控制我们想要对稀疏损失应用的惩罚强度。

![img](https://lilianweng.github.io/posts/2018-08-12-vae/kl-metric-sparse-autoencoder.png)

> 图 4. 具有均值$\rho=0.25$和具有均值的伯努利分布$0 \leq \hat{\rho} \leq 1$ 之间的 KL 散度

**$k$-稀疏自动编码器**

在$k-$稀疏自动编码器（[Makhzani 和 Frey，2013 年](https://arxiv.org/abs/1312.5663)），稀疏性是通过仅在具有线性激活函数的bottleneck layer 中保留前 k 个最高激活来强制执行的。首先我们通过编码器网络运行以获得压缩编码：$\mathbf{z} = g(\mathbf{x})$. 对编码向量$z$中的值进行排序,  仅保留 k 个最大值，而其他神经元设置为 0。这也可以在具有可调阈值的 ReLU 层中完成。现在我们有了一个稀疏编码：$\mathbf{z}’ = \text{Sparsify}(\mathbf{z})$. 计算稀疏编码的输出和损失，$L = |\mathbf{x} - f(\mathbf{z}') |_2^2$.  而且，反向传播只经过前 k 个激活的隐藏单元！

![img](https://lilianweng.github.io/posts/2018-08-12-vae/k-sparse-autoencoder.png)

> 图 5. 不同稀疏度 k 的 k-稀疏自动编码器的过滤器，从具有 1000 个隐藏单元的 MNIST 中学习。（图片来源：[Makhzani 和 Frey，2013 年](https://arxiv.org/abs/1312.5663)）

# 收缩自动编码器

与稀疏自动编码器类似，**收缩**自动编码器( [Rifai, et al, 2011](http://www.icml-2011.org/papers/455_icmlpaper.pdf) ) 鼓励学习的表示留在收缩空间中以获得更好的鲁棒性。

它在损失函数中添加了一个项来惩罚对输入过于敏感的表示，从而提高对训练数据点周围小扰动的鲁棒性。灵敏度是通过编码器激活的雅可比矩阵相对于输入的 Frobenius 范数来衡量的：
$$
\|J_f(\mathbf{x})\|_F^2 = \sum_{ij} \Big( \frac{\partial h_j(\mathbf{x})}{\partial x_i} \Big)^2
$$
其中$h_j$是压缩编码中的一个单元输出$\mathbf{z} = f(x)$.

该惩罚项是学习编码相对于输入维度的所有偏导数的平方和。作者声称，根据经验，发现这种惩罚可以刻画对应于低维非线性流形的表示，同时对与流形正交的多数方向保持更大的不变性。

# VAE：变分自动编码器

**Variational Autoencoder** ( Kingma [& Welling, 2014](https://arxiv.org/abs/1312.6114) ) 的缩写**VAE**的思想实际上与上述所有自编码器模型不太相似，但深深植根于变分贝叶斯和图形模型的方法。

不是将输入映射到*固定向量，而是将其映射到分布。*让我们将此分布标记为$p_\theta$, 参数化为$\theta$. 数据输入之间的关系$\mathbf{x}$和latent encoding向量和可以完全定义为：

- Prior $p_\theta(\mathbf{z})$
- Likelihood $p_\theta(\mathbf{x}\vert\mathbf{z})$
- Posterior $p_\theta(\mathbf{z}\vert\mathbf{x})$

假设我们知道对于这个分布的真实参数$\theta^{*}$。为了生成看起来像真实数据点的样本$\mathbf{x}^{(i)}$，我们遵循以下步骤：

1. 首先，从先验分布$p_{\theta^*}(\mathbf{z})$采样$\mathbf{z}^{(i)}$.
2. 然后由条件分布$p_{\theta^*}(\mathbf{x} \vert \mathbf{z} = \mathbf{z}^{(i)})$生成一个值$\mathbf{x}^{(i)}$.

最优参数$\theta^{*}$即是最大化生成真实数据样本的概率：
$$
\theta^{*} = \arg\max_\theta \prod_{i=1}^n p_\theta(\mathbf{x}^{(i)})
$$
通常我们使用对数概率将 RHS 上的乘积转换为总和：
$$
\theta^{*} = \arg\max_\theta \sum_{i=1}^n \log p_\theta(\mathbf{x}^{(i)})
$$
现在让我们更新方程以更好地演示数据生成过程，从而导出编码向量：
$$
p_\theta(\mathbf{x}^{(i)}) = \int p_\theta(\mathbf{x}^{(i)}\vert\mathbf{z}) p_\theta(\mathbf{z}) d\mathbf{z}
$$
不幸的是，以这种方式计算起来并不容易$p_\theta(\mathbf{x}^{(i)})$，因为检查所有可能的值是非常昂贵的，并对它们进行加和。为了缩小值空间以促进更快的搜索，我们想引入一个新的近似函数来输出的可能编码，在给定输入$\mathbf{x}$, 参数化为$\phi$的$q_\phi(\mathbf{z}\vert\mathbf{x})$时 .

![img](https://lilianweng.github.io/posts/2018-08-12-vae/VAE-graphical-model.png)

图 6. Variational Autoencoder 涉及的图形模型。实线表示生成分布$p\_\theta(.)$ 虚线表示分布$q\_\phi (\mathbf{z}\vert\mathbf{x})$来近似难以处理的 posterior $p\_\theta (\mathbf{z}\vert\mathbf{x})$.

现在，这个结构看起来很像一个自动编码器：

- 条件概率$p_\theta(\mathbf{x}\vert\mathbf{z})$定义一个生成模型，类似于上面介绍过的解码器$f_\theta(\mathbf{x} \vert \mathbf{z})$。$p_\theta(\mathbf{x}\vert\mathbf{z})$也称为*概率解码器*。
- 逼近函数$q_\phi(\mathbf{z}\vert\mathbf{x})$是*概率编码器*，起着和$g_\phi(\mathbf{z} \vert \mathbf{x})$ 类似的作用。

## 损失函数：ELBOW

估计后验$q_\phi(\mathbf{z}\vert\mathbf{x})$应该和真实分布$p_\theta(\mathbf{z}\vert\mathbf{x})$很接近. 我们可以使用[Kullback-Leibler 散度](https://en.wikipedia.org/wiki/Kullback–Leibler_divergence)来量化这两个分布之间的距离。KL散度$D_\text{KL}(X|Y)$ 衡量了，如果使用分布 Y 来表示 X，丢失了多少信息。

在我们的例子中，我们想最小化  $D_\text{KL}( q_\phi(\mathbf{z}\vert\mathbf{x}) | p_\theta(\mathbf{z}\vert\mathbf{x}) )$ 关于$\phi$

但是为什么要用$D_\text{KL}(q_\phi | p_\theta)$（反向 KL）而不是$D_\text{KL}(p_\theta | q_\phi)$？Eric Jang 在他关于贝叶斯变分方法的[帖子](https://blog.evjang.com/2016/08/variational-bayes.html)中有很好的解释。快速回顾一下：

![img](https://lilianweng.github.io/posts/2018-08-12-vae/forward_vs_reversed_KL.png)

>  图 7. 正向和反向 KL 散度对如何匹配两个分布有不同的要求。（图片来源：[blog.evjang.com/2016/08/variational-bayes.html](https://blog.evjang.com/2016/08/variational-bayes.html)）

- 前向 KL 散度：$D_\text{KL}(P|Q) = \mathbb{E}_{z\sim P(z)} \log\frac{P(z)}{Q(z)}$; 我们必须确保 P(z)>0 时 Q(z)>0。优化的变分分布$q(z)$必须覆盖整个$p(z)$
- 反向 KL 散度：$D_\text{KL}(Q|P) = \mathbb{E}_{z\sim Q(z)} \log\frac{Q(z)}{P(z)}$; 最小化反向 KL 散度在 $P(z)$下, 压缩了$Q(z)$.

现在让我们扩展等式：
$$
\begin{aligned}
& D_\text{KL}( q_\phi(\mathbf{z}\vert\mathbf{x}) \| p_\theta(\mathbf{z}\vert\mathbf{x}) ) & \\
&=\int q_\phi(\mathbf{z} \vert \mathbf{x}) \log\frac{q_\phi(\mathbf{z} \vert \mathbf{x})}{p_\theta(\mathbf{z} \vert \mathbf{x})} d\mathbf{z} & \\
&=\int q_\phi(\mathbf{z} \vert \mathbf{x}) \log\frac{q_\phi(\mathbf{z} \vert \mathbf{x})p_\theta(\mathbf{x})}{p_\theta(\mathbf{z}, \mathbf{x})} d\mathbf{z} & \scriptstyle{\text{; Because }p(z \vert x) = p(z, x) / p(x)} \\
&=\int q_\phi(\mathbf{z} \vert \mathbf{x}) \big( \log p_\theta(\mathbf{x}) + \log\frac{q_\phi(\mathbf{z} \vert \mathbf{x})}{p_\theta(\mathbf{z}, \mathbf{x})} \big) d\mathbf{z} & \\
&=\log p_\theta(\mathbf{x}) + \int q_\phi(\mathbf{z} \vert \mathbf{x})\log\frac{q_\phi(\mathbf{z} \vert \mathbf{x})}{p_\theta(\mathbf{z}, \mathbf{x})} d\mathbf{z} & \scriptstyle{\text{; Because }\int q(z \vert x) dz = 1}\\
&=\log p_\theta(\mathbf{x}) + \int q_\phi(\mathbf{z} \vert \mathbf{x})\log\frac{q_\phi(\mathbf{z} \vert \mathbf{x})}{p_\theta(\mathbf{x}\vert\mathbf{z})p_\theta(\mathbf{z})} d\mathbf{z} & \scriptstyle{\text{; Because }p(z, x) = p(x \vert z) p(z)} \\
&=\log p_\theta(\mathbf{x}) + \mathbb{E}_{\mathbf{z}\sim q_\phi(\mathbf{z} \vert \mathbf{x})}[\log \frac{q_\phi(\mathbf{z} \vert \mathbf{x})}{p_\theta(\mathbf{z})} - \log p_\theta(\mathbf{x} \vert \mathbf{z})] &\\
&=\log p_\theta(\mathbf{x}) + D_\text{KL}(q_\phi(\mathbf{z}\vert\mathbf{x}) \| p_\theta(\mathbf{z})) - \mathbb{E}_{\mathbf{z}\sim q_\phi(\mathbf{z}\vert\mathbf{x})}\log p_\theta(\mathbf{x}\vert\mathbf{z}) &
\end{aligned}
$$
所以我们有：
$$
D_\text{KL}( q_\phi(\mathbf{z}\vert\mathbf{x}) \| p_\theta(\mathbf{z}\vert\mathbf{x}) ) =\log p_\theta(\mathbf{x}) + D_\text{KL}(q_\phi(\mathbf{z}\vert\mathbf{x}) \| p_\theta(\mathbf{z})) - \mathbb{E}_{\mathbf{z}\sim q_\phi(\mathbf{z}\vert\mathbf{x})}\log p_\theta(\mathbf{x}\vert\mathbf{z})
$$
重新排列等式的左侧和右侧，
$$
\log p_\theta(\mathbf{x}) - D_\text{KL}( q_\phi(\mathbf{z}\vert\mathbf{x}) \| p_\theta(\mathbf{z}\vert\mathbf{x}) ) = \mathbb{E}_{\mathbf{z}\sim q_\phi(\mathbf{z}\vert\mathbf{x})}\log p_\theta(\mathbf{x}\vert\mathbf{z}) - D_\text{KL}(q_\phi(\mathbf{z}\vert\mathbf{x}) \| p_\theta(\mathbf{z}))
$$
等式的 LHS 正是我们在学习真实分布时想要最大化的：我们想要最大化生成真实数据的（对数）似然（即日志⁡$\log p_\theta(\mathbf{x})$ 并且还最小化真实后验分布和估计后验分布之间的差异（术语$D_\text{KL}$在像正则化器起作用）。注意$p_\theta(\mathbf{x})$相对于$q_\phi$是固定的.

上面的定义了我们的损失函数：
$$
\begin{aligned}
L_\text{VAE}(\theta, \phi)
&= -\log p_\theta(\mathbf{x}) + D_\text{KL}( q_\phi(\mathbf{z}\vert\mathbf{x}) \| p_\theta(\mathbf{z}\vert\mathbf{x}) )\\
&= - \mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z}\vert\mathbf{x})} \log p_\theta(\mathbf{x}\vert\mathbf{z}) + D_\text{KL}( q_\phi(\mathbf{z}\vert\mathbf{x}) \| p_\theta(\mathbf{z}) ) \\
\theta^{*}, \phi^{*} &= \arg\min_{\theta, \phi} L_\text{VAE}
\end{aligned}
$$
在变分贝叶斯方法中，此损失函数称为*变分下界*或*证据下界*。名称中的“下界”部分来自于 KL 散度始终为非负的事实，因此$-L_\text{VAE}$是$\log p_\theta (\mathbf{x})$下界.

$-L_\text{VAE} = \log p_\theta(\mathbf{x}) - D_\text{KL}( q_\phi(\mathbf{z}\vert\mathbf{x}) \| p_\theta(\mathbf{z}\vert\mathbf{x}) ) \leq \log p_\theta(\mathbf{x})$

因此，通过最小化损失，我们正在最大化生成真实数据样本的概率下限。

## 重新参数化技巧

损失函数中的期望项调用生成样本$\mathbf{z} \sim q_\phi(\mathbf{z}\vert\mathbf{x})$. 采样是一个随机过程，因此我们不能反向传播梯度。为了使其可训练，引入了重参数化技巧：通常可以表达随机变量$\mathbf{z}$作为确定性变量 $\mathbf{z} = \mathcal{T}_\phi(\mathbf{x}, \boldsymbol{\epsilon})$， 其中$\boldsymbol{\epsilon}$是辅助独立随机变量，由$\phi$参数化的变换函数$\mathcal{T}_\phi$ 转换$ε$到$z$.

例如，常见的选择形式$q_\phi(\mathbf{z}\vert\mathbf{x})$是具有对角协方差结构的多元高斯分布：
$$
\begin{aligned}
\mathbf{z} &\sim q_\phi(\mathbf{z}\vert\mathbf{x}^{(i)}) = \mathcal{N}(\mathbf{z}; \boldsymbol{\mu}^{(i)}, \boldsymbol{\sigma}^{2(i)}\boldsymbol{I}) & \\
\mathbf{z} &= \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon} \text{, where } \boldsymbol{\epsilon} \sim \mathcal{N}(0, \boldsymbol{I}) & \scriptstyle{\text{; Reparameterization trick.}}
\end{aligned}
$$
其中⊙指的是逐元素乘积。

![img](https://lilianweng.github.io/posts/2018-08-12-vae/reparameterization-trick.png)

> 图 8. 重新参数化技巧如何使$z$采样过程可训练。（图片来源：Kingma 的 NIPS 2015 研讨会[演讲](http://dpkingma.com/wordpress/wp-content/uploads/2015/12/talk_nips_workshop_2015.pdf)中的幻灯片 12 ）

重参数化技巧也适用于其他类型的分布，而不仅仅是高斯分布。在多元高斯情况下，我们通过学习分布的均值和方差使模型可训练，$\mu$和$\sigma$，明确使用重新参数化技巧，而随机性保留在随机变量中$\boldsymbol{\epsilon} \sim \mathcal{N}(0, \boldsymbol{I})$.

![img](https://lilianweng.github.io/posts/2018-08-12-vae/vae-gaussian.png)

> 图 9. 具有多元高斯假设的变分自动编码器模型的图示。

# Beta-VAE

如果推断的潜在表示中的每个变量$z$只对单一的生成因素敏感并且对其他因素相对不变，我们会说这种表示是可分离的或可因式分解的。分离表示通常带来的一个好处是具有*良好的可解释性*和易于泛化到各种任务。

例如，一个基于人脸照片训练的模型可能会在不同的维度上捕捉到温柔、肤色、头发颜色、头发长度、情绪、是否戴眼镜等许多其他相对独立的因素。这种解缠结的表示对人脸图像的生成非常有利。

β-VAE ( [Higgins et al., 2017](https://openreview.net/forum?id=Sy2fzU9gl) ) 是变分自动编码器的一种改进，特别强调发现解缠结的潜在因子。在 VAE 中遵循相同的动机，我们希望最大化生成真实数据的概率，同时保持真实后验分布和估计后验分布之间的距离很小（比如，在一个小常数下d):
$$
\begin{aligned}
&\max_{\phi, \theta} \mathbb{E}_{\mathbf{x}\sim\mathcal{D}}[\mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z}\vert\mathbf{x})} \log p_\theta(\mathbf{x}\vert\mathbf{z})]\\
&\text{subject to } D_\text{KL}(q_\phi(\mathbf{z}\vert\mathbf{x})\|p_\theta(\mathbf{z})) < \delta
\end{aligned}
$$
我们可以将其重写在[KKT 条件下](https://www.cs.cmu.edu/~ggordon/10725-F12/slides/16-kkt.pdf)带有拉格朗日乘数$\beta$为的拉格朗日公式。上述只有一个不等式约束的优化问题等价于最大化以下方程$\mathcal{F}(\theta, \phi, \beta)$:
$$
\begin{aligned}
\mathcal{F}(\theta, \phi, \beta) &= \mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z}\vert\mathbf{x})} \log p_\theta(\mathbf{x}\vert\mathbf{z}) - \beta(D_\text{KL}(q_\phi(\mathbf{z}\vert\mathbf{x})\|p_\theta(\mathbf{z})) - \delta) & \\
& = \mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z}\vert\mathbf{x})} \log p_\theta(\mathbf{x}\vert\mathbf{z}) - \beta D_\text{KL}(q_\phi(\mathbf{z}\vert\mathbf{x})\|p_\theta(\mathbf{z})) + \beta \delta & \\
& \geq \mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z}\vert\mathbf{x})} \log p_\theta(\mathbf{x}\vert\mathbf{z}) - \beta D_\text{KL}(q_\phi(\mathbf{z}\vert\mathbf{x})\|p_\theta(\mathbf{z})) & \scriptstyle{\text{; Because }\beta,\delta\geq 0}
\end{aligned}
$$
$\beta$-VAE的损失函数 定义为：
$$
L_\text{BETA}(\phi, \beta) = - \mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z}\vert\mathbf{x})} \log p_\theta(\mathbf{x}\vert\mathbf{z}) + \beta D_\text{KL}(q_\phi(\mathbf{z}\vert\mathbf{x})\|p_\theta(\mathbf{z}))
$$
其中拉格朗日乘数$\beta$被视为超参数。

由于否定$L_\text{BETA}(\phi, \beta)$是拉格朗日量的下界$\mathcal{F}(\theta, \phi, \beta)$. 最小化损失等同于最大化拉格朗日量，因此适用于我们的初始优化问题。

D当$\beta$=1, 它与 VAE 相同。当$\beta$>1，它对潜在的瓶颈施加了更强的约束并限制$z$. 对于一些条件独立的生成因子，将它们分开是最有效的表示。因此更高的$\beta$鼓励更有效的latent encoding并进一步鼓励分离。同时，更高$\beta$可能会在重建质量和分离程度之间做出权衡。

[伯吉斯等人。(2017)](https://arxiv.org/pdf/1804.03599.pdf)讨论了在$\beta$[-VAE在信息瓶颈理论](https://lilianweng.github.io/posts/2017-09-28-information-bottleneck/)的启发下深入研究并进一步提出修改$\beta$-VAE 更好地控制编码表示能力。

# VQ-VAE 和 VQ-VAE-2

**VQ-VAE** （ “矢量量化变分自动编码器”；[van den Oord 等人，2017 年](http://papers.nips.cc/paper/7210-neural-discrete-representation-learning.pdf)）模型通过编码器学习离散隐变量，因为离散表示可能更自然地适合语言、语音、推理等问题，等等

矢量量化 (VQ) 是一种映射方法$K$维向量转换为一组有限的“编码”向量。[该过程与KNN](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)算法非常相似。样本应映射到的最佳质心编码向量是具有最小欧氏距离的向量。

让$\mathbf{e} \in \mathbb{R}^{K \times D}, i=1, \dots, K$是 VQ-VAE 中的潜在嵌入空间（也称为“密码本”），其中$K$是隐变量类别的数量，并且$D$是嵌入尺寸。一个单独的嵌入向量是$\mathbf{e}_i \in \mathbb{R}^{D}, i=1, \dots, K$.

编码器输出$E(\mathbf{x}) = \mathbf{z}_e$通过最近邻查找以匹配其中一个$K$嵌入向量，然后这个匹配的编码向量成为解码器的输入$D$(.):
$$
\mathbf{z}_q(\mathbf{x}) = \text{Quantize}(E(\mathbf{x})) = \mathbf{e}_k \text{ where } k = \arg\min_i \|E(\mathbf{x}) - \mathbf{e}_i \|_2
$$
请注意，离散隐变量在不同的应用程序中可以具有不同的形状；例如，1D 用于语音，2D 用于图像，3D 用于视频。

![img](https://lilianweng.github.io/posts/2018-08-12-vae/VQ-VAE.png)

> 图 10. VQ-VAE 的架构（图片来源：[van den Oord, et al. 2017](http://papers.nips.cc/paper/7210-neural-discrete-representation-learning.pdf)）

因为 argmin() 在离散空间上不可微，所以梯度$\nabla_z L$来自解码器输入$\mathbf{z}_q$被复制到编码器输出$\mathbf{z}_e$. 除了重建损失，VQ-VAE 还优化了：

- *VQ 损失*：嵌入空间和编码器输出之间的 L2 误差。
- *承诺损失*：一种鼓励编码器输出靠近嵌入空间并防止它从一个编码向量到另一个编码向量波动过于频繁的措施。

$$
L = \underbrace{\|\mathbf{x} - D(\mathbf{e}_k)\|_2^2}_{\textrm{reconstruction loss}} +
\underbrace{\|\text{sg}[E(\mathbf{x})] - \mathbf{e}_k\|_2^2}_{\textrm{VQ loss}} +
\underbrace{\beta \|E(\mathbf{x}) - \text{sg}[\mathbf{e}_k]\|_2^2}_{\textrm{commitment loss}}
$$

其中$\text{sq}[.]$ 是`stop_gradient`运算符。

嵌入向量通过 EMA（指数移动平均）更新。给定一个编码向量和$\mathbf{e}_i$, 说我们有$n_i$编码器输出向量，$\{\mathbf{z}_{i,j}\}_{j=1}^{n_i}$ 被量化为和$\mathbf{e}_i$:
$$
N_i^{(t)} = \gamma N_i^{(t-1)} + (1-\gamma)n_i^{(t)}\;\;\;
\mathbf{m}_i^{(t)} = \gamma \mathbf{m}_i^{(t-1)} + (1-\gamma)\sum_{j=1}^{n_i^{(t)}}\mathbf{z}_{i,j}^{(t)}\;\;\;
\mathbf{e}_i^{(t)} = \mathbf{m}_i^{(t)} / N_i^{(t)}
$$
其中($(t)$)指时间上的批次顺序。$N_i$和$\mathbf{m}_i$分别是累积矢量计数和体积。

VQ-VAE-2 ( [Ali Razavi, et al. 2019](https://arxiv.org/abs/1906.00446) ) 是一个结合自注意力自回归模型的两层的层次结构的 VQ-VAE 。

1. 阶段 1 是**训练分层 VQ-VAE**：分层隐变量的设计旨在将局部模式（即纹理）与全局信息（即对象形状）分开。较大的底层编码本的训练也是以较小的顶层编码为条件的，因此它不必从头开始学习所有内容。
2. 第 2 阶段是**在潜在的离散码本上学习先验知识，**以便我们从中采样并生成图像。通过这种方式，解码器可以接收从与训练中的分布相似的分布中采样的输入向量。使用多头自注意力层增强的强大自回归模型用于捕获先验分布（如[PixelSNAIL；Chen 等人 2017](https://arxiv.org/abs/1712.09763)）。

考虑到 VQ-VAE-2 依赖于在简单的分层设置中配置的离散隐变量，其生成的图像质量非常惊人。

![img](https://lilianweng.github.io/posts/2018-08-12-vae/VQ-VAE-2.png)

> 图 11. 分层 VQ-VAE 和多阶段图像生成的架构。（图片来源：[Ali Razavi, et al. 2019](https://arxiv.org/abs/1906.00446)）

![img](https://lilianweng.github.io/posts/2018-08-12-vae/VQ-VAE-2-algo.png)

> VQ-VAE-2算法。（图片来源：[Ali Razavi, et al. 2019](https://arxiv.org/abs/1906.00446))

# TD-VAE

**TD-VAE**（“Temporal Difference VAE”；[Gregor 等人，2019 年](https://arxiv.org/abs/1806.03107)）处理时序数据。它依赖于三个主要思想，如下所述。

![img](https://lilianweng.github.io/posts/2018-08-12-vae/TD-VAE-state-space.png)

>  图 13. 作为马尔可夫链模型的状态空间模型。

**1. 状态空间模型**

在（潜在）状态空间模型中，一系列未观察到的隐藏状态$\mathbf{z} = (z_1, \dots, z_T)$. 决定了观测状态$\mathbf{x} = (x_1, \dots, x_T)$

图 13 中的马尔可夫链模型中的每个时间步都可以用与图 6 类似的方式进行训练，其中难处理的后验$p(z \vert x)$由一个函数逼近$q(z \vert x)$

**2. 信念状态**

Agent应该学习对所有过去的状态进行编码以推理未来，称为*信念状态*，$b_t = belief(x_1, \dots, x_t) = belief(b_{t-1}, x_t)$. 鉴于此，以过去为条件的未来状态的分布可以写成$p(x_{t+1}, \dots, x_T \vert x_1, \dots, x_t) \approx p(x_{t+1}, \dots, x_T \vert b_t)$. 循环策略中的隐藏状态用作代理在 TD-VAE 中的信念状态。因此我们有$b_t = \text{RNN}(b_{t-1}, x_t)$.

**3. 跳跃式预测**

此外，智能体需要根据目前收集到的所有信息来想象遥远的未来，这表明具有跳跃式预测的能力，即预测未来几个步骤的状态。

[回想一下我们从上面](https://lilianweng.github.io/posts/2018-08-12-vae/#loss-function-elbo)的方差下限中学到的东西：
$$
\begin{aligned}
\log p(x)
&\geq \log p(x) - D_\text{KL}(q(z|x)\|p(z|x)) \\
&= \mathbb{E}_{z\sim q} \log p(x|z) - D_\text{KL}(q(z|x)\|p(z)) \\
&= \mathbb{E}_{z \sim q} \log p(x|z) - \mathbb{E}_{z \sim q} \log \frac{q(z|x)}{p(z)} \\
&= \mathbb{E}_{z \sim q}[\log p(x|z) -\log q(z|x) + \log p(z)] \\
&= \mathbb{E}_{z \sim q}[\log p(x, z) -\log q(z|x)] \\
\log p(x)
&\geq \mathbb{E}_{z \sim q}[\log p(x, z) -\log q(z|x)]
\end{aligned}
$$
现在让我们模拟状态的分布$x_t$作为以所有过去状态为条件的概率函数$x_{<t}$和两个隐变量，$z_t$和和$z_{t-1}$，在当前时间步后退一步：
$$
\log p(x_t|x_{<{t}}) \geq \mathbb{E}_{(z_{t-1}, z_t) \sim q}[\log p(x_t, z_{t-1}, z_{t}|x_{<{t}}) -\log q(z_{t-1}, z_t|x_{\leq t})]
$$
继续扩大等式：
$$
\begin{aligned}
& \log p(x_t|x_{<{t}}) \\
&\geq \mathbb{E}_{(z_{t-1}, z_t) \sim q}[\log p(x_t, z_{t-1}, z_{t}|x_{<{t}}) -\log q(z_{t-1}, z_t|x_{\leq t})] \\
&\geq \mathbb{E}_{(z_{t-1}, z_t) \sim q}[\log p(x_t|\color{red}{z_{t-1}}, z_{t}, \color{red}{x_{<{t}}}) + \color{blue}{\log p(z_{t-1}, z_{t}|x_{<{t}})} -\log q(z_{t-1}, z_t|x_{\leq t})] \\
&\geq \mathbb{E}_{(z_{t-1}, z_t) \sim q}[\log p(x_t|z_{t}) + \color{blue}{\log p(z_{t-1}|x_{<{t}})} + \color{blue}{\log p(z_{t}|z_{t-1})} - \color{green}{\log q(z_{t-1}, z_t|x_{\leq t})}] \\
&\geq \mathbb{E}_{(z_{t-1}, z_t) \sim q}[\log p(x_t|z_{t}) + \log p(z_{t-1}|x_{<{t}}) + \log p(z_{t}|z_{t-1}) - \color{green}{\log q(z_t|x_{\leq t})} - \color{green}{\log q(z_{t-1}|z_t, x_{\leq t})}]
\end{aligned}
$$
注意两点：

- 根据马尔可夫假设，可以忽略红色项。
- 蓝色项根据马尔可夫假设展开。
- 绿色项被扩展为包括回到过去的一步预测作为平滑分布。

准确地说，有四种类型的分布需要学习：

1. $p_D(.)$是**解码器**分布：

- $p(x_t \mid z_t)$是通用定义的编码器；
- $p(x_t \mid z_t) \to p_D(x_t \mid z_t)$

2. p吨(.)是**过渡**分布：

- $p(z_t \mid z_{t-1})$捕获隐变量之间的顺序依赖性；
- $p(z_t \mid z_{t-1}) \to p_T(z_t \mid z_{t-1})$

3. $p_B(.)$是**信念**分布：

- 两个都$p(z_{t-1} \mid x_{<t})$和$q(z_t \mid x_{\leq t})$可以使用信念状态来预测隐变量；
- $p(z_{t-1} \mid x_{<t}) \to p_B(z_{t-1} \mid b_{t-1})$;
- $q(z_{t} \mid x_{\leq t}) \to p_B(z_t \mid b_t)$;

4. $p_S(.)$是**平滑**分布：

- 回到过去的平滑项$q(z_{t-1} \mid z_t, x_{\leq t})$也可以重写为依赖于信念状态；
- $q(z_{t-1} \mid z_t, x_{\leq t}) \to  p_S(z_{t-1} \mid z_t, b_{t-1}, b_t)$;

为了结合跳跃预测的思想，顺序 ELBO 不仅要在$t, t+1$1, 还有两个遥远的时间$t_1 < t_2$. 这是要最大化的最终 TD-VAE 目标函数：
$$
J_{t_1, t_2} = \mathbb{E}[
  \log p_D(x_{t_2}|z_{t_2})
  + \log p_B(z_{t_1}|b_{t_1})
  + \log p_T(z_{t_2}|z_{t_1})
  - \log p_B(z_{t_2}|b_{t_2})
  - \log p_S(z_{t_1}|z_{t_2}, b_{t_1}, b_{t_2})]
$$
![img](https://lilianweng.github.io/posts/2018-08-12-vae/TD-VAE.png)

> 图 14. TD-VAE 架构的详细概述。（图片来源：[TD-VAE论文](https://arxiv.org/abs/1806.03107)）

# 参考


[1] Geoffrey E. Hinton, and Ruslan R. Salakhutdinov. [“Reducing the dimensionality of data with neural networks."](https://pdfs.semanticscholar.org/c50d/ca78e97e335d362d6b991ae0e1448914e9a3.pdf) Science 313.5786 (2006): 504-507.

[2] Pascal Vincent, et al. [“Extracting and composing robust features with denoising autoencoders."](http://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf) ICML, 2008.

[3] Pascal Vincent, et al. [“Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion."](http://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf). Journal of machine learning research 11.Dec (2010): 3371-3408.

[4] Geoffrey E. Hinton, Nitish Srivastava, Alex Krizhevsky, Ilya Sutskever, and Ruslan R. Salakhutdinov. “Improving neural networks by preventing co-adaptation of feature detectors.” arXiv preprint arXiv:1207.0580 (2012).

[5] [Sparse Autoencoder](https://web.stanford.edu/class/cs294a/sparseAutoencoder.pdf) by Andrew Ng.

[6] Alireza Makhzani, Brendan Frey (2013). [“k-sparse autoencoder”](https://arxiv.org/abs/1312.5663). ICLR 2014.

[7] Salah Rifai, et al. [“Contractive auto-encoders: Explicit invariance during feature extraction."](http://www.icml-2011.org/papers/455_icmlpaper.pdf) ICML, 2011.

[8] Diederik P. Kingma, and Max Welling. [“Auto-encoding variational bayes."](https://arxiv.org/abs/1312.6114) ICLR 2014.

[9] [Tutorial - What is a variational autoencoder?](https://jaan.io/what-is-variational-autoencoder-vae-tutorial/) on jaan.io

[10] Youtube tutorial: [Variational Autoencoders](https://www.youtube.com/watch?v=9zKuYvjFFS8) by Arxiv Insights

[11] [“A Beginner’s Guide to Variational Methods: Mean-Field Approximation”](https://blog.evjang.com/2016/08/variational-bayes.html) by Eric Jang.

[12] Carl Doersch. [“Tutorial on variational autoencoders."](https://arxiv.org/abs/1606.05908) arXiv:1606.05908, 2016.

[13] Irina Higgins, et al. ["β-VAE: Learning basic visual concepts with a constrained variational framework."](https://openreview.net/forum?id=Sy2fzU9gl) ICLR 2017.

[14] Christopher P. Burgess, et al. [“Understanding disentangling in beta-VAE."](https://arxiv.org/abs/1804.03599) NIPS 2017.

[15] Aaron van den Oord, et al. [“Neural Discrete Representation Learning”](https://arxiv.org/abs/1711.00937) NIPS 2017.

[16] Ali Razavi, et al. [“Generating Diverse High-Fidelity Images with VQ-VAE-2”](https://arxiv.org/abs/1906.00446). arXiv preprint arXiv:1906.00446 (2019).

[17] Xi Chen, et al. [“PixelSNAIL: An Improved Autoregressive Generative Model."](https://arxiv.org/abs/1712.09763) arXiv preprint arXiv:1712.09763 (2017).

[18] Karol Gregor, et al. [“Temporal Difference Variational Auto-Encoder."](https://arxiv.org/abs/1806.03107) ICLR 2019.
