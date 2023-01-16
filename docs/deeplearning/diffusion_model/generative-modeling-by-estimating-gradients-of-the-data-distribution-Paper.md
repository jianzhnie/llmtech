# Generative Modeling by Estimating Gradients of the Data Distribution

> Yang Song   Stanford University yangsong@cs.stanford.edu
>
>
> Stefano Ermon  Stanford University  ermon@cs.stanford.edu

## Abstract

我们引入了一种新的生成模型，其中样本是通过  Langevin 动力学使用分数匹配的方式通过数据分布梯度估计产生。因为梯度可能定义不明确并且难以估计，数据何时驻留在低维流形，我们用不同级别的高斯噪声扰动数据，并联合估计相应的分数，即矢量场所有噪声水平的扰动数据分布的梯度。对于采样，我们提出退火  Langevin dynamics，随着采样过程越来越接近数据流形，噪声水平逐渐降低，我们也使用相对应梯度变化。我们的框架采用灵活的模型架构，不需要在训练期间进行采样也没有使用对抗方法，并且我们提供可学习的目标函数，用于原则性模型比较。我们的模型生成的采样结果与GAN方法在 MNIST、CelebA 和 CIFAR-10 数据集上的结果相媲美，在 CIFAR-10 上实现了新的最先进的分数8.87。此外，我们证明模型通过图像修复实验学习到有效的表征。

## Introduction

生成模型在机器学习中有很多应用。例如，将生成模型用于生成高保真图像 [26, 6]，合成逼真的语音和音乐片段 [58]，提高半监督学习的性能 [28, 10]，检测对抗样本和其他异常数据 [54]、模仿学习 [22]，并探索有希望的状态强化学习中[41]。最近的进展主要由两种方法驱动：基于似然方法 [17,29, 11, 60] 和生成对抗网络 (GAN [15])。前者使用对数似然（或suitable surrogate)作为训练目标，而后者使用对抗训练来最小化 模型和数据分布之间的$f$-divergences  [40] 或积分概率指标 [2] 55]。

虽然基于似然的模型和 GAN 取得了巨大的成功，但它们有一些内在的限制。例如，基于似然的模型要么必须使用专门的架构来建立归一化概率模型（例如，自回归模型、流动模型），或使用代理损失（例如，变分自动编码器 [29] 中使用的证据下限、在基于能量的模型中 [21]的对比发散）进行训练。GAN 避免了基于似然模型的一些限制，但由于对抗性训练程序，它们的训练可能不稳定。除此之外GAN 目标函数设定不适合评估和比较不同的 GAN 模型。而其他生成模型的目标，例如噪声对比估计 [19] 和最小概率流 [50]，这些方法通常只适用于低维数据。

在本文中，我们探索了一种基于估计和采样的生成模型的新原则，它来自对数数据密度的 (Stein) 分数 [33]，是在输入数据点的对数密度的梯度。这是一个向量场，指向对数数据密度增长最快的方向。我们使用经过分数匹配训练的神经网络 [24] 来学习这个来自数据的矢量场。然后我们使用 Langevin 动力学产生样本，这近似通过将随机初始样本沿着（估计的）分数的矢量场逐渐移动到高密度区域来工作的。然而，这种方法存在两个主要挑战。首先，如果数据分布在低维流形上得到支持——正如许多现实世界通常假设的那种数据集 - 分数在环境空间中将是未定义的，分数匹配将无法提供一致的分数估计。第二，低数据密度区域训练数据的稀缺性，例如距离流形很远，阻碍了分数估计的准确性并减慢了 Langevin 的混合动态采样。由于 Langevin 动力学通常会在数据分布的低密度区域进行初始化，这些区域的分数估计不准确会对抽样过程产生负面影响。此外，因为需要穿越低密度区域以分配模式之间的转换，混合可能很困难。

为了应对这两个挑战，我们建议用各种量级的随机高斯噪声扰动数据。添加随机噪声可确保生成的分布不会崩溃为低维流形。大噪声水平将在低密度区域产生原始（未受干扰的）数据分布样本，从而改善分数估计。至关重要的是，我们训练了一个以噪声水平为条件的评分网络，并估计所有噪声幅度的分数。我们然后提出 Langevin 动力学的退火版本，我们初始化使用与最高噪声水平相应的分数，并逐渐降低噪声水平，直到它足够小与原始数据分布没有区别。我们的抽样策略受到模拟退火 [30, 37]的启发，启发式地改进多模态场景的优化。

我们的方法有几个理想的特性。首先，我们的目标对于几乎所有的对参数化评分网络来说都是容易处理的，不需要特殊约束或架构设计，以及可以在没有对抗训练、MCMC 采样或其他近似的情况下进行优化训练。该目标也可用于在同一数据集上定量比较不同模型。通过实验，我们在 MNIST、CelebA [34]、和 CIFAR-10 [31]展示我们的方法。这些样本看起来与现代基于似然的模型和 GAN生成的样本相当。在 CIFAR-10 上，我们的模型对于无条件生成模型取得了新的最先进的起始点得分8.87，并获得了具有竞争力的 FID 分数25.32. 我们表明该模型通过图像修复实验学习到有意义的数据表示。

## 2. Score-based generative modeling

假设我们的数据集由来自未知的数据分布 $p_{\text {data }}(\mathbf{x})$ 的 iid 样本组成 $\left\{\mathbf{x}_{i} \in \mathbb{R}^{D}\right\}_{i=1}^{N}$ . 我们定义概率密度 $p(\mathbf{x})$的分数 为 $\nabla_{\mathbf{x}} \log p(\mathbf{x})$. 得分网络 $\mathbf{s}_{\boldsymbol{\theta}}: \mathbb{R}^{D} \rightarrow \mathbb{R}^{D}$ 是一个由$\boldsymbol{\theta}$参数化的神经网络，该网络将被训练来近似得分 $p_{\text {data }}(\mathbf{x})$. 生成模型的目标是使用数据学习生成模型来生成来自$p_{\text {data }}(\mathbf{x})$ 的新样本。基于分数的生成模型框架有两个组成部分：分数匹配和 Langevin 动力学。

### 2.1 Score matching for score estimation

分数匹配 [24] 最初设计是为学习来自未知数据分布的独立同分布样本的非正态分布的统计学模型。

在 [53] 之后，我们将其重新用于分数估计。使用分数匹配，我们可以直接训练一个分数网 $\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x})$ 估计$\nabla_{\mathbf{x}} \log p_{\text {data }}(\mathbf{x})$ 而不用首先训练模型进行估计 $p_{\text {data }}(\mathbf{x})$. 与分数匹配的典型用法不同，我们选择不要使用基于能量的模型的梯度作为得分网络，以避免由于更高阶的梯度导致的额外计算。

最小化目标 ：$\frac{1}{2} \mathbb{E}_{p_{\text {data }}}\left[\left\|\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x})-\nabla_{\mathbf{x}} \log p_{\text {data }}(\mathbf{x})\right\|_{2}^{2}\right]$, 等同于以下式子：
$$
\mathbb{E}_{p_{\text {data }}(\mathbf{x})}\left[\operatorname{tr}\left(\nabla_{\mathbf{x}} \mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x})\right)+\frac{1}{2}\left\|\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x})\right\|_{2}^{2}\right],
$$

其中 $\nabla_{\mathbf{x}} \mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x})$ 表示$\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x})$ 的雅可比矩阵 。如[53]所示，在一些正则条件下方程式的最小值。(3) 表示为 $\mathbf{s}_{\boldsymbol{\theta}^{*}}(\mathbf{x})$ ) 满足 $\mathbf{s}_{\boldsymbol{\theta}^{*}}(\mathbf{x})=\nabla_{\mathbf{x}} \log p_{\text {data }}(\mathbf{x})$ almost surely.

在实践中， Eq. (1) 中 期望超过 $p_{\text {data }}(\mathbf{x})$ 可以使用数据采样快速估计。然而，因为$\operatorname{tr}\left(\nabla_{\mathbf{x}} \mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x})\right)$的计算， 分数匹配不能扩展到深度网络和高维数据。下面我们讨论两种流行的大规模分数匹配方法。

#### 2.1.1 Denoising score matching

去噪分数匹配去噪分数匹配[61]是分数匹配的一种变体，它完全规避 $\operatorname{tr}\left(\nabla_{\mathbf{x}} \mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x})\right)$. 它首先扰乱带有预先指定的噪音分布$q_{\sigma}(\tilde{\mathbf{x}} \mid \mathbf{x})$ 的数据点$\mathbf{x}$ ， 然后使用分数匹配来估计扰动数据的分数分布$q_{\sigma}(\tilde{\mathbf{x}}) \triangleq \int q_{\sigma}(\tilde{\mathbf{x}} \mid \mathbf{x}) p_{\text {data }}(\mathbf{x}) \mathrm{d} \mathbf{x}$.

目标被证明等同于以下内容：
$$
\frac{1}{2} \mathbb{E}_{q_{\sigma}(\tilde{\mathbf{x}} \mid \mathbf{x}) p_{\text {data }}(\mathbf{x})}\left[\left\|\mathbf{s}_{\boldsymbol{\theta}}(\tilde{\mathbf{x}})-\nabla_{\tilde{\mathbf{x}}} \log q_{\sigma}(\tilde{\mathbf{x}} \mid \mathbf{x})\right\|_{2}^{2}\right] .
$$

如[61]所示，最佳得分网络 (表示为 $\mathbf{s}_{\theta^{*}}(\mathbf{x})$ ) that minimizes Eq. 22] 满足 $\mathbf{s}_{\boldsymbol{\theta}^{*}}(\mathbf{x})=\nabla_{\mathbf{x}} \log q_{\sigma}(\mathbf{x})$ 几乎可以肯定。  然而, $\mathbf{s}_{\boldsymbol{\theta}^{*}}(\mathbf{x})=\nabla_{\mathbf{x}} \log q_{\sigma}(\mathbf{x}) \approx \nabla_{\mathbf{x}} \log p_{\text {data }}(\mathbf{x})$  只有当噪声足够小时 $q_{\sigma}(\mathbf{x}) \approx p_{\text {data }}(\mathbf{x})$成立。

#### 2.1.2 Sliced score matching

Sliced score matching [53]在分数匹配中使用随机投影来近似 $\operatorname{tr}\left(\nabla_{\mathbf{x}} \mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x})\right)$ . 目标是:
$$
\mathbb{E}_{p_{\mathbf{v}}} \mathbb{E}_{p_{\text {data }}}\left[\mathbf{v}^{\boldsymbol{\top}} \nabla_{\mathbf{x}} \mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}) \mathbf{v}+\frac{1}{2}\left\|\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x})\right\|_{2}^{2}\right],
$$

其中 $p_{\mathbf{v}}$ 是随机向量的简单分布，例如多元标准正态分布。如图所示在[53]中，术语  $\mathbf{v}^{\top} \nabla_{\mathbf{x}} \mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}) \mathbf{v}$ 可以通过前向模式自动微分有效地计算。与去噪分数匹配估计扰动数据的分数不同，分片分数匹配为原始未受干扰的数据分布提供分数估计，但需要大约四倍的计算，由于前向模式自动微分。

### 2.2 Sampling with Langevin dynamics

Langevin 动力学仅使用评分函数 $\nabla_{\mathbf{x}} \log p(\mathbf{x})$ 从概率密度产生样本  $p(\mathbf{x})$  。

给定固定步长 $\epsilon>0$, 和一个以 $\pi$ 作为先验分布的初始值 $\tilde{\mathbf{x}}_{0} \sim \pi(\mathbf{x})$ , Langevin 方法递归地计算以下：
$$
\tilde{\mathbf{x}}_{t}=\tilde{\mathbf{x}}_{t-1}+\frac{\epsilon}{2} \nabla_{\mathbf{x}} \log p\left(\tilde{\mathbf{x}}_{t-1}\right)+\sqrt{\epsilon} \mathbf{z}_{t},
$$

其中 $\mathbf{z}_{t} \sim \mathcal{N}(0, I)$ 的分布 $\tilde{\mathbf{x}}_{T}$ 等于 $p(\mathbf{x})$ ， 当 $\epsilon \rightarrow 0$ 和$T \rightarrow \infty$, 这种情况下 $\tilde{\mathbf{x}}_{T}$ 满足一些规律性条件下[62]成为一个精确的样本$p(\mathbf{x})$ . 当 $\epsilon>0$ 和$T<\infty$, 需要 Metropolis-Hastings 更新来纠正方程式的错误，但它经常在实践中被忽略[9, 12, 39]。在这项工作中，我们假设这个错误可以忽略不计，当 $\epsilon$ 很小 $T$ 很大的时候.

请注意，来自 Eq. 4)的抽样只需要评分函数 $\nabla_{\mathbf{x}} \log p(\mathbf{x})$. 因此，为了从 $p_{\text {data }}(\mathbf{x})$,获取样本，我们可以首先训练我们的得分网络，这样   $\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}) \approx \nabla_{\mathbf{x}} \log p_{\text {data }}(\mathbf{x})$ 然后使用 Langevin 动力学近似获得样本 $\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x})$. 这是我们基于分数的生成模型框架关键思想。

## 3. Challenges of score-based generative modeling

在本节中，我们将更仔细地分析基于分数的生成模型的思想。我们认为有两个主要障碍阻止我们天真地应用这个想法。

### 3.1 The manifold hypothesis

流形假设表明在现实世界中，数据倾向于专注于以低维度嵌入在高维空间中。这个假设在经验上适用于许多数据集，并已成为流形学习的基础[3, 47]。在流形假设下，基于分数的生成模型将面临两个关键困难。首先， 得分 $\nabla_{\mathbf{x}} \log p_{\text {data }}(\mathbf{x})$ 是在环境空间中采用的梯度，当 $\mathrm{x}$ 被限制在低维时是未定义的。
![](https://cdn.mathpix.com/cropped/2023_01_11_b824421116c00dc5d0fbg-3.jpg?height=250&width=696&top_left_y=1986&top_left_x=1058)

Figure 1:

Left: Sliced score matching (SSM) loss w.r.t. iterations. No noise is added to data.

Right: Same but data are perturbed with $\mathcal{N}(0,0.0001)$. manifold.

第二，当数据分布的支撑是整个空间时, 得分匹配目标Eq(1) 提供一致的分数估计器（参见 [24] 中的定理 2），并且当数据驻留在低维流形上时将会得到不一致的结果。

从图 1 可以清楚地看出流形假设对分数估计的负面影响。我们训练一个 ResNet（详见附录 B.11 ) 以估计 CIFAR-10 上的数据分数。对于快速训练和忠实估计数据分数，我们使用切片分数匹配目标（Eq.（34）。如图 1（左）所示，当在原始 CIFAR-10 图像上训练时，切片分数匹配损失先减小后不规则波动。相反，如果我们用较小的高斯噪声（这样扰动的数据分布在 $\mathbb{R}^{D}$有一个完整的支撑集。）， Loss 曲线将收敛（右面板）。请注意， 对于像素值在[0,1]范围内的图像, 我们施加的很小高斯噪声$\mathcal{N}(0,0.0001)$ ，人眼几乎无法区分。

### 3.2 Low data density regions

低密度区域的数据稀缺会给分数估计和分数匹配及 Langevin 动力学的MCMC 采样与带来困难 。

#### 3. 2.1. Inaccurate score estimation with score matching

在低数据密度区域，分数匹配可能没有足够的证据来准确估计评分功能，由于缺乏数据样本。要看到这一点，请回忆一下 Section 2.1 得分匹配最小化了分数估计的预期平方误差，即

$$\frac{1}{2} \mathbb{E}_{p_{\text {data }}}\left[\left\|\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x})-\nabla_{\mathbf{x}} \log p_{\text {data }}(\mathbf{x})\right\|_{2}^{2}\right]$$.

在实践中，对数据分布的期望总是使用独立同分布样本来估计  $\left\{\mathbf{x}_{i}\right\}_{i=1}^{N} \stackrel{\text { i.i.d. }}{\sim} p_{\text {data }}(\mathbf{x})$.  考虑任何

$\mathcal{R} \subset \mathbb{R}^{D}$ 这样 $p_{\text {data }}(\mathcal{R}) \approx 0$. 多数情况下 $\left\{\mathbf{x}_{i}\right\}_{i=1}^{N} \cap \mathcal{R}=\varnothing$,  对于 $\mathbf{x} \in \mathcal{R}$，得分匹配将没有足够的数据样本来准确地估计 $\nabla_{\mathbf{x}} \log p_{\text {data }}(\mathbf{x})$ .

To de为了证明这一点的负面影响，我们

![](https://cdn.mathpix.com/cropped/2023_01_11_b824421116c00dc5d0fbg-4.jpg?height=352&width=677&top_left_y=930&top_left_x=1079)

Figure 2:     Left: $\nabla_{\mathbf{x}} \log p_{\text {data }}(\mathbf{x})$;      Right: $\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x})$.

数据密度  $p_{\text {data }}(\mathbf{x})$ 使用 orange colormap 编码  : 颜色越深意味着密度越高。

红色矩形突出显示区域， 其中 $\nabla_{\mathbf{x}} \log p_{\text {data }}(\mathbf{x}) \approx \mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x})$ 是实验的结果 (details in Appendix B.11 in Fig. 2 其中其中我们使用切片分数匹配以估计高斯混合的分数 $p_{\text {data }}=\frac{1}{5} \mathcal{N}((-5,-5), I)+\frac{4}{5} \mathcal{N}((5,5), I)$.

如图所示，分数估计仅在 $p_{\text {data }}$ 模式附近可靠 ，这其中数据密度高。

#### 3. 2.2 Slow mixing of Langevin dynamics

当数据分布的两种模式被低密度区域分隔时，动力学将无法在合理的时间内正确恢复这两种模式的相对权重，并且因此可能不会收敛到真实分布。我们对此的分析主要受到 [63] 的启发，它在具有分数匹配的密度估计的背景下分析了相同的现象。

考虑混合分布 $p_{\text {data }}(\mathbf{x})=\pi p_{1}(\mathbf{x})+(1-\pi) p_{2}(\mathbf{x})$, 其中 $p_{1}(\mathbf{x})$ 和$p_{2}(\mathbf{x})$  是正态分布具有不相交的支撑集，$\pi \in(0,1)$. 在 $p_{1}(\mathbf{x})$下, $\nabla_{\mathbf{x}} \log p_{\text {data }}(\mathbf{x})=$ $\nabla_{\mathbf{x}}\left(\log \pi+\log p_{1}(\mathbf{x})\right)=\nabla_{\mathbf{x}} \log p_{1}(\mathbf{x})$, 并在支撑 $\left.\pi)+\log p_{2}(\mathbf{x})\right)=\nabla_{\mathbf{x}} \log p_{2}(\mathbf{x})$ 下。

在任何一种情况下，得分 $\nabla_{\mathbf{x}} \log p_{\text {data }}(\mathbf{x})$ 不依赖于 $\pi$. 由于  Langevin 动力学使用 $\nabla_{\mathbf{x}} \log p_{\text {data }}(\mathbf{x})$ 从$p_{\text {data }}(\mathbf{x})$中取样 , 获得的样本不会取决于$\pi$.  在实践中，当不同模式大致不相交时，这种分析也成立支持 - 它们可能共享相同的支持，但由小数据密度的区域连接。在这个在这种情况下，Langevin 动力学在理论上可以产生正确的样本，但可能需要非常小的步骤和非常多的步骤进行混合。

为了验证此分析，我们在第 3.2.1 节中使用相同高斯混合分布测试 Langevin 动力学采样，并提供图 3 中的结果。我们在采样时使用真实分数与Langevin dynamics的对比。比较图 3(b) 和 (a)，正如我们的分析所预测的那样，很明显 Langevin 的样品具有不正确的相对密度。

![](https://cdn.mathpix.com/cropped/2023_01_11_b824421116c00dc5d0fbg-5.jpg?height=439&width=418&top_left_y=241&top_left_x=450)

(a)

![](https://cdn.mathpix.com/cropped/2023_01_11_b824421116c00dc5d0fbg-5.jpg?height=420&width=420&top_left_y=256&top_left_x=863)

(b)

![](https://cdn.mathpix.com/cropped/2023_01_11_b824421116c00dc5d0fbg-5.jpg?height=420&width=393&top_left_y=256&top_left_x=1278)

(c) 图 3：来自不同方法的高斯混合样本。(a) 精确抽样。（二）使用具有精确分数的 Langevin 动力学进行采样。(c) 使用退火 Langevin 动力学采样，具有确切分数。显然，Langevin 动力学错误估计了这两种模式之间的相对权重，而退火朗之万动力学忠实地恢复了相对权重。

## 4. Noise Conditional Score Networks: learning 和inference

我们观察到用随机高斯噪声扰动数据使得数据分布更适合基于分数的生成模型。首先，由于支持我们的高斯噪声分布是整个空间，扰动数据不会局限于低维流形，这消除了流形假设的困难，并明确定义了分数估计。第二，大的高斯噪声具有填充原始未扰动中低密度区域的数据分布作用；因此分数匹配可能会得到更多的训练信号来提高分数估计。此外，通过使用多个噪声水平，我们可以获得一系列噪声扰动分布收敛到真实的数据分布。我们可以提高 Langevin 动力学的混合率，通过利用这些中间分布来研究多峰分布退火[30]和退火重要性抽样[37]。

基于这种直觉，我们建议通过 1) 使用各种噪声水平扰动数据来改进基于分数的生成模型；2）通过训练单个条件评分网络来同时估计所有对应的分数来降低噪声水平。训练结束后，使用 Langevin dynamics 来生成样本，最初使用对应于大噪声的分数，然后逐渐退火降低噪音水平。这有助于将大噪声水平的好处平稳地转移到低噪声水平，这样扰动数据与原始数据几乎无法区分。接下来，我们将详细说明我们方法的细节，包括我们分数架构网络、训练目标和 Langevin 动力学的退火时间表。

### 4.1 Noise Conditional Score Networks

让 $\left\{\sigma_{i}\right\}_{i=1}^{L}$ 为正几何序列，满足 $\frac{\sigma_{1}}{\sigma_{2}}=\cdots=\frac{\sigma_{L-1}}{\sigma_{L}}>1$. 让 $q_{\sigma}(\mathbf{x}) \triangleq$ $\int p_{\text {data }}(\mathbf{t}) \mathcal{N}\left(\mathbf{x} \mid \mathbf{t}, \sigma^{2} I\right) \mathrm{d} \mathbf{t}$ 表示扰动的数据分布。我们选择噪音水平 $\left\{\sigma_{i}\right\}_{i=1}^{L}$ 这样$\sigma_{1}$个足够大以减轻第 3 节中讨论的困难， 并且 $\sigma_{L}$ 足够小尽量减少对数据的影响。我们的目标是训练一个条件评分网络来联合估计所有扰动数据分布的分数，即 $\forall \sigma \in\left\{\sigma_{i}\right\}_{i=1}^{L}: \mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}, \sigma) \approx \nabla_{\mathbf{x}} \log q_{\sigma}(\mathbf{x})$. 注意  $\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}, \sigma) \in \mathbb{R}^{D}$ 当 $\mathbf{x} \in \mathbb{R}^{D}$. 我们称 $\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}, \sigma)$ 为噪声条件评分网络 (NCSN)。

类似于基于似然的生成模型和 GAN，模型架构的设计对于生成高质量样本的起着重要作用。在这项工作中，我们主要关注用于图像生成的架构，并将其他领域的架构设计留作未来的工作。由于我们的噪声条件评分网络的输出与输入具有相同的形状 $\mathbf{x}$，我们从成功的模型架构中汲取灵感，用于图像的密集预测（例如，语义分割）。在实验中，我们的模型 $\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}, \sigma)$  结合架构U-Net [46] 与 dilated/atrous convolution [64, 65, 8] 的设计——两者都已被证明在语义分割方面非常成功。此外，我们在分数网络中采用了实例归一化，受到其在某些图像生成任务中的卓越性能的启发 [57、13、23]，我们使用条件实例规范化的修改版本 [13] 提供条件$\sigma_{i}$.有关我们架构的更多详细信息，请参见附录 A.

### 4.2 Learning NCSNs via score matching

切片和去噪分数匹配都可以训练 NCSN。我们采用去噪分数匹配，因为它速度稍快，自然适合估计噪声扰动数据分布分数的任务。然而，我们强调经验切片分数匹配可以训练 NCSN 以及去噪分数匹配。我们选择噪声分布 $q_{\sigma}(\tilde{\mathbf{x}} \mid \mathbf{x})=\mathcal{N}\left(\tilde{\mathbf{x}} \mid \mathbf{x}, \sigma^{2} I\right)$; 因此 $\nabla_{\tilde{\mathbf{x}}} \log q_{\sigma}(\tilde{\mathbf{x}} \mid \mathbf{x})=-(\tilde{\mathbf{x}}-\mathbf{x}) / \sigma^{2}$. 对于给定的 $\sigma$, 去噪分数匹配目标 ((Eq. 2 2) 为
$$
\ell(\boldsymbol{\theta} ; \sigma) \triangleq \frac{1}{2} \mathbb{E}_{p_{\text {data }}(\mathbf{x})} \mathbb{E}_{\tilde{\mathbf{x}} \sim \mathcal{N}\left(\mathbf{x}, \sigma^{2} I\right)}\left[\left\|\mathbf{s}_{\boldsymbol{\theta}}(\tilde{\mathbf{x}}, \sigma)+\frac{\tilde{\mathbf{x}}-\mathbf{x}}{\sigma^{2}}\right\|_{2}^{2}\right] .
$$

然后，我们结合Eq.5 。对于所有的 $\sigma \in\left\{\sigma_{i}\right\}_{i=1}^{L}$ 达成一个统一的目标

$$
\mathcal{L}\left(\boldsymbol{\theta} ;\left\{\sigma_{i}\right\}_{i=1}^{L}\right) \triangleq \frac{1}{L} \sum_{i=1}^{L} \lambda\left(\sigma_{i}\right) \ell\left(\boldsymbol{\theta} ; \sigma_{i}\right),
$$

其中 $\lambda\left(\sigma_{i}\right)>0$ 是一个系数函数，取决于 $\sigma_{i}$. 假设 $\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}, \sigma)$有足够的容量, $\mathbf{s}_{\boldsymbol{\theta}^{*}}(\mathbf{x}, \sigma)$ 最小化方程式 Eq. (6) if 当且仅当  $\mathbf{s}_{\boldsymbol{\theta}^{*}}\left(\mathbf{x}, \sigma_{i}\right)=\nabla_{\mathbf{x}} \log q_{\sigma_{i}}(\mathbf{x})$ 至于所有 $i \in\{1,2, \cdots, L\}$, 因为 Eq. 6 是去噪得分匹配目标的锥形组合 $L$ 。

$\lambda(\cdot)$ 可以有很多可能的选择. 理想情况下，我们希望  $\lambda\left(\sigma_{i}\right) \ell\left(\boldsymbol{\theta} ; \sigma_{i}\right)$ 对所有  $\left\{\sigma_{i}\right\}_{i=1}^{L}$ 大致处于同一数量级。 根据经验，我们观察到当分数网络被训练到最优，我们 近似有 $\left\|\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}, \sigma)\right\|_{2} \propto 1 / \sigma$. 这个激励我们选择  $\lambda(\sigma)=\sigma^{2}$.  因为在这个选择下，我们有 $\lambda(\sigma) \ell(\boldsymbol{\theta} ; \sigma)=\sigma^{2} \ell(\boldsymbol{\theta} ; \sigma)=$ $\frac{1}{2} \mathbb{E}\left[\left\|\sigma \mathbf{s}_{\boldsymbol{\theta}}(\tilde{\mathbf{x}}, \sigma)+\frac{\tilde{\mathbf{x}}-\mathbf{x}}{\sigma}\right\|_{2}^{2}\right]$. 因为$\frac{\tilde{\mathbf{x}}-\mathbf{x}}{\sigma} \sim \mathcal{N}(0, I)$ 和$\left\|\sigma \mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}, \sigma)\right\|_{2} \propto 1$, 我们可以很容易地得出结论的数量级 $\lambda(\sigma) \ell(\boldsymbol{\theta} ; \sigma)$ 不依赖于 $\sigma$.

我们强调我们的目标方程式。(6) 不需要对抗性训练，没有代理损失，也没有在训练期间从分数网络中采样（例如，与对比发散不同）。而且，它确实不需要 $\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}, \sigma)$ 具有特殊的架构以便易于处理。 此外， 当 $\lambda(\cdot)$ 和$\left\{\sigma_{i}\right\}_{i=1}^{L}$是固定的，它可以用来定量比较不同的 NCSN。

### 4.3 NCSN inference via annealed Langevin dynamics

在 $\operatorname{NCSN~s}_{\boldsymbol{\theta}}(\mathbf{x}, \sigma)$ 受过训练之后，我们建议 annealed Langevin dynamics 的采样方法（Alg. 11.-to produced samples, inspired通过模拟退火 [30] 和退火重要性采样 [37]。如 Alg.|1] 所示，我们开始 Langevin dynamics 通过初始化开始来自一些固定先验分布的样本，例如，均匀噪声。然后，我们运行 Langevin  动力学以步长 $\alpha_{1}$从 $q_{\sigma_{1}}(\mathbf{x})$中采样。  我们运行 Langevin dynamics 从  $q_{\sigma_{2}}(\mathbf{x})$ 中采样， 从最终样本开始先前的模拟并使用减少的步长 $\alpha_{2}$. 我们继续这种方式，使用 Langevin 动力学的最终样本 $q_{\sigma_{i-1}}(\mathbf{x})$ 作为 Langevin 动态的初始样本 $q_{\sigma_{i}}(\mathbf{x})$, 和并调小步长 $\alpha_{i}$ 通过 $\alpha_{i}=\epsilon \cdot \sigma_{i}^{2} / \sigma_{L}^{2}$. 最后，我们运行Langevin dynamics 从 $q_{\sigma_{L}}(\mathbf{x})$采样， 当 $\sigma_{L} \approx 0$， 接近于 $p_{\text {data }}(\mathbf{x})$ 。

因为分布 $\left\{q_{\sigma_{i}}\right\}_{i=1}^{L}$ 都受到高斯噪声的扰动，它们的支撑跨越整个空间及其分数定义明确，避免了流形假设的困难。 当 $\sigma_{1}$ 足够大，低密度区域 $q_{\sigma_{1}}(\mathbf{x})$ 变小，模式更加孤立。如前所述，这可以使分数估计更准确，并且混合Langevin 动力学更快。因此，我们可以假设 Langevin 动力学为 $q_{\sigma_{1}}(\mathbf{x})$ 产生良好的样本. 这些样本很可能来自高密度区域  $q_{\sigma_{1}}(\mathbf{x})$。

Table 1: Inception 和FID scores for CIFAR-10
![](https://cdn.mathpix.com/cropped/2023_01_11_b824421116c00dc5d0fbg-7.jpg?height=592&width=586&top_left_y=239&top_left_x=1148)

Figure 4: Intermediate samples of annealed Langevin dynamics.

他们也可能居住在高密度地区  $q_{\sigma_{2}}(\mathbf{x})$, given that $q_{\sigma_{1}}(\mathbf{x})$ 和$q_{\sigma_{2}}(\mathbf{x})$ only slightly differ from each other. As score estimation 和Langevin dynamics perform better in high density regions, samples from $q_{\sigma_{1}}(\mathbf{x})$ will serve as good initial samples for Langevin dynamics of $q_{\sigma_{2}}(\mathbf{x})$. Similarly, $q_{\sigma_{i-1}}(\mathbf{x})$ provides good initial samples for $q_{\sigma_{i}}(\mathbf{x})$, 和finally we obtain samples of good quality from $q_{\sigma_{L}}(\mathbf{x})$.

There could be many possible ways of tuning $\alpha_{i}$ according to $\sigma_{i}$ in Alg.1. Our choice is $\alpha_{i} \propto \sigma_{i}^{2}$. The motivation is to fix the magnitude of the "signal-to-noise" ratio $\frac{\alpha_{i} \mathbf{s}_{\theta}\left(\mathbf{x}, \sigma_{i}\right)}{2 \sqrt{\alpha_{i}} \mathbf{z}}$ in Langevin dynamics. 注意 that $\mathbb{E}\left[\left\|\frac{\alpha_{i} \mathbf{s}_{\boldsymbol{\theta}}\left(\mathbf{x}, \sigma_{i}\right)}{2 \sqrt{\alpha_{i} \mathbf{z}}}\right\|_{2}^{2}\right] \approx \mathbb{E}\left[\frac{\alpha_{i}\left\|\mathbf{s}_{\boldsymbol{\theta}}\left(\mathbf{x}, \sigma_{i}\right)\right\|_{2}^{2}}{4}\right] \propto \frac{1}{4} \mathbb{E}\left[\left\|\sigma_{i} \mathbf{s}_{\boldsymbol{\theta}}\left(\mathbf{x}, \sigma_{i}\right)\right\|_{2}^{2}\right]$. Recall that empirically we found $\left\|\mathbf{s}_{\boldsymbol{\theta}}(\mathbf{x}, \sigma)\right\|_{2} \propto 1 / \sigma$ 当 the score network is trained close to optimal,这种情况下 $\mathbb{E}\left[\left\|\sigma_{i} \mathbf{S}_{\boldsymbol{\theta}}\left(\mathbf{x} ; \sigma_{i}\right)\right\|_{2}^{2}\right] \propto 1$.因此 $\left\|\frac{\alpha_{i} \mathbf{s}_{\boldsymbol{\theta}}\left(\mathbf{x}, \sigma_{i}\right)}{2 \sqrt{\alpha_{i}} \mathbf{z}}\right\|_{2} \propto \frac{1}{4} \mathbb{E}\left[\left\|\sigma_{i} \mathbf{s}_{\boldsymbol{\theta}}\left(\mathbf{x}, \sigma_{i}\right)\right\|_{2}^{2}\right] \propto \frac{1}{4}$ does not depend on $\sigma_{i}$.

To demonstrate the efficacy of our annealed Langevin dynamics, we provide a toy example 其中 the goal is to sample from a mixture of Gaussian with two well-separated modes using only scores. We apply Alg. 1 to sample from the mixture of Gausssian used in Section 3.2. In the experiment, we choose $\left\{\sigma_{i}\right\}_{i=1}^{L}$ to be a geometric progression, with $L=10, \sigma_{1}=10$ 和$\sigma_{10}=0.1$. The results are provided in Fig. 33. Comparing Fig. 33 (b) against (c), annealed Langevin dynamics correctly recover the relative weights between the two modes 其中as standard Langevin dynamics fail.

## Experiments

In this section, we demonstrate that our NCSNs are able to produce high quality image samples on several commonly used image datasets. In addition, we show that our models learn reasonable image representations by image inpainting experiments.

Setup We use MNIST, CelebA [34], 和CIFAR-10 [31] datasets in our experiments. For CelebA, the images are first center-cropped to $140 \times 140$ 和then resized to $32 \times 32$. All images are rescaled so that pixel values are in $[0,1]$. We choose $L=10$ different standard deviations such that $\left\{\sigma_{i}\right\}_{i=1}^{L}$ is a geometric sequence with $\sigma_{1}=1$ 和$\sigma_{10}=0.01$. 注意 that Gaussian noise of $\sigma=0.01$ is almost indistinguishable to human eyes for image data. 当 using annealed Langevin dynamics for image generation, we choose $T=100$ 和$\epsilon=2 \times 10^{-5}$, 和use uniform noise as our initial samples. We found the results are robust w.r.t. the choice of $T$, 和$\epsilon$ between $5 \times 10^{-6}$ 和$5 \times 10^{-5}$ generally works fine. We provide additional details on model architecture 和settings in Appendix A 和B

Image generation In Fig.55, we show uncurated samples from annealed Langevin dynamics for MNIST, CelebA 和CIFAR-10. As shown by the samples, our generated images have higher or comparable quality to those from modern likelihood-based models 和GANs. To intuit the procedure of annealed Langevin dynamics, we provide intermediate samples in Fig.4 其中 each row shows

![](https://cdn.mathpix.com/cropped/2023_01_11_b824421116c00dc5d0fbg-8.jpg?height=425&width=420&top_left_y=243&top_left_x=430)

(a) MNIST

![](https://cdn.mathpix.com/cropped/2023_01_11_b824421116c00dc5d0fbg-8.jpg?height=428&width=415&top_left_y=239&top_left_x=844)

(b) CelebA

![](https://cdn.mathpix.com/cropped/2023_01_11_b824421116c00dc5d0fbg-8.jpg?height=425&width=420&top_left_y=243&top_left_x=1275)

(c) CIFAR-10

Figure 5: Uncurated samples on MNIST, CelebA, 和CIFAR-10 datasets.
![](https://cdn.mathpix.com/cropped/2023_01_11_b824421116c00dc5d0fbg-8.jpg?height=486&width=1256&top_left_y=858&top_left_x=432)

Figure 6: Image inpainting on CelebA (left) 和CIFAR-10 (right). The leftmost column of each figure shows the occluded images, while the rightmost column shows the original images.

how samples evolve from pure random noise to high quality images. More samples from our approach can be found in Appendix [C] We also show the nearest neighbors of generated images in the training dataset in Appendix C.2, in order to demonstrate that our model is not simply memorizing training images. To show it is important to learn a conditional score network jointly for many noise levels 和use annealed Langevin dynamics, we compare against a baseline approach 其中 we only consider one noise level $\left\{\sigma_{1}=0.01\right\}$ 和use the vanilla Langevin dynamics sampling method. Although this small added noise helps circumvent the difficulty of the manifold hypothesis (as shown by Fig. 1. things will comp让ely fail if no noise is added), it is not large enough to provide information on scores in regions of low data density. As a result, this baseline fails to generate reasonable images, as shown by samples in Appendix C.1.

For quantitative evaluation, we report inception [48] 和FID [20] scores on CIFAR-10 in Tab. 1] As an unconditional model, we achieve the state-of-the-art inception score of 8.87, which is even better than most reported values for class-conditional generative models. Our FID score $25.32$ on CIFAR-10 is also comparable to top existing models, such as SNGAN [36]. We omit scores on MNIST 和CelebA as the scores on these two datasets are not widely reported, 和different preprocessing (such as the center crop size of CelebA) can lead to numbers not directly comparable.

Image inpainting In Fig. 6, we demonstrate that our score networks learn generalizable 和semantically meaningful image representations that allow it to produce diverse image inpaintings. 注意 that some previous models such as PixelCNN can only impute images in the raster scan order. In contrast, our method can naturally handle images with occlusions of arbitrary shapes by a simple modification of the annealed Langevin dynamics procedure (details in Appendix B.3). We provide more image inpainting results in Appendix C.5

## Related work

Our approach has some similarities with methods that learn the transition operator of a Markov chain for sample generation [4, 51, 5, 16, 52]. For example, generative stochastic networks (GSN [4, 1]) use denoising autoencoders to train a Markov chain whose equilibrium distribution matches the data distribution. Similarly, our method trains the score function used in Langevin dynamics to sample from the data distribution. 然而, GSN often starts the chain very close to a training data point, and因此 requires the chain to transition quickly between different modes. In contrast, our annealed Langevin dynamics are initialized from unstructured noise. Nonequilibrium Thermodynamics (NET [51]) used a prescribed diffusion process to slowly transform data into random noise, 和then learned to reverse this procedure by training an inverse diffusion. 然而, NET is not very scalable 因为 it requires the diffusion process to have very small steps, 和needs to simulate chains with thousands of steps at training time.

Previous approaches such as Infusion Training (IT [5]) 和Variational Walkback (VW [16]) also employed different noise levels/temperatures for training transition operators of a Markov chain. Both IT 和VW (as well as NET) train their models by maximizing the evidence lower bound of a suitable marginal likelihood. In practice, they tend to produce blurry image samples, similar to variational autoencoders. In contrast, our objective is based on score matching instead of likelihood, 和we can produce images comparable to GANs.

There are several structural differences that further distinguish our approach from previous methods discussed above. First, we do not need to sample from a Markov chain during training. In contrast, the walkback procedure of GSNs needs multiple runs of the chain to generate "negative samples". Other methods including NET, IT, 和VW also need to simulate a Markov chain for every input to compute the training loss. This difference makes our approach more efficient 和scalable for training deep models. Secondly, our training 和sampling methods are decoupled from each other. For score estimation, both sliced 和denoising score matching can be used. For sampling, any method based on scores is applicable, including Langevin dynamics 和(potentially) Hamiltonian Monte Carlo [38]. Our framework allows arbitrary combinations of score estimators 和(gradient-based) sampling approaches, 其中as most previous methods tie the model to a specific Markov chain. Finally, our approach can be used to train energy-based models $(E B M)$ by using the gradient of an energy-based model as the score model. In contrast, it is unclear how previous methods that learn transition operators of Markov chains can be directly used for training EBMs.

Score matching was originally proposed for learning EBMs. 然而, many existing methods based on score matching are either not scalable [24] or fail to produce samples of comparable quality to VAEs or GANs [27, 49]. To obtain better performance on training deep energy-based models, some recent works have resorted to contrastive divergence [21], 和propose to sample with Langevin dynamics for both training 和testing [12, 39]. 然而, unlike our approach, contrastive divergence uses the computationally expensive procedure of Langevin dynamics as an inner loop during training. The idea of combining annealing with denoising score matching has also been investigated in previous work under different contexts. In [14, 7, 66], different annealing schedules on the noise for training denoising autoencoders are proposed. 然而, their work is on learning representations for improving the performance of classification, instead of generative modeling. The method of denoising score matching can also be derived from the perspective of Bayes least squares [43, 44], using techniques of Stein's Unbiased Risk Estimator [35, 56].

## Conclusion

We propose the framework of score-based generative modeling 其中 we first estimate gradients of data densities via score matching, 和then generate samples via Langevin dynamics. We analyze several challenges faced by a naïve application of this approach, 和propose to tackle them by training Noise Conditional Score Networks (NCSN) 和sampling with annealed Langevin dynamics. Our approach requires no adversarial training, no MCMC sampling during training, 和no special model architectures. Experimentally, we show that our approach can generate high quality images that were previously only produced by the best likelihood-based models 和GANs. We achieve the new state-of-the-art inception score on CIFAR-10, 和an FID score comparable to SNGANs.
