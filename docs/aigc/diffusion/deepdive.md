# 深入理解扩散模型

## 引言：生成模型

给定从感兴趣分布中观察到的样本 $x$，生成模型的目标是学习模拟其真实的数据分布 $p(x)$。一旦学习完成，我们就可以随意从我们的近似模型中生成新的样本。此外，在某些方程化下，我们还可以使用学习到的模型来评估观测或采样数据的可能性。

在当前文献中，有几种众所周知的方向，这里只是简要地在高层次上介绍。

生成对抗网络（GANs）以对抗训练的方式学习复杂分布的采样过程，用神经网络直接去拟合真实分布，优化目标是使得真实数据分布与生成数据分布的KL散度达到最小。

另一类被称为“基于似然的”生成模型，寻求学习一个模型，优化的目标是使得样本数据分布出现的概率最大。这包括自回归模型、归一化流模型和变分自编码器（VAEs）。

另一种类似的方法是基于能量的建模，其学习一个分布作为任意灵活的能量函数，然后进行归一化。

分数生成模型与此密切相关；它们不是学习模拟能量函数本身，而是学习能量模型的分数作为神经网络。在这项工作中，我们将探索和回顾扩散模型，正如我们将在本研究所展示的，它们既有基于似然的解释，也有基于分数的解释。我们以极其详尽的细节展示了这些模型背后的数学，目的是让任何人都能够跟随并理解扩散模型是什么以及它们是如何工作的。

## 背景：ELBO、VAE和层次VAE

对于许多模态，可以认为观察到的数据由一个相关的不可见潜在变量表示或生成，我们可以通过随机变量 $ z$ 来表示。表达这一想法的最好直觉是通过柏拉图的洞穴寓言。在寓言中，一群人一生中都被锁在一个洞穴里，只能看到投射在他们面前墙上的二维阴影，这些阴影是由在火前经过的不可见的三维物体产生的。对于这些人来说，他们观察到的一切都是由他们永远无法看到的更高维度的抽象概念决定的。

<img src="https：//whbzju.github.io/img/Diffusion/Untitled.png" alt="Untitled" style="zoom：80%;" />

> 洞穴寓言故事大致描述，把一群人关起来，如图所示，他们永远只能看岩壁上的二维画面，这个二维画面的影像是外部三维世界的物体经过一个固定的火把后在岩壁上的投影。岩壁上二维的画面显然是由外部三维实体决定的。进一步拓展思路，很有可能我们在现实世界中观察到的很多现象（数据）是从更高维度投影过来。

类似地，我们在现实世界中遇到的物体也可能是由某些更高层次的表示生成的；例如，这些表示可能封装了颜色、大小、形状等抽象属性。然后，我们观察到的可以被解释为这些抽象概念的三维投影或实例化，就像洞穴居民观察到的实际上是三维物体的二维投影一样。尽管洞穴居民永远看不到（甚至完全理解不了）隐藏的物体，他们仍然可以推断出关于它们的信息；类似地，我们可以近似潜在表示来描述我们观察到的数据。

尽管柏拉图的洞穴寓言阐释了潜在变量作为可能不可观察的表示来决定观察结果的背后思想，但这个类比的一个缺点是，在生成建模中，我们通常寻求学习低于观察维度的潜在表示，而不是高于观察维度的。这是因为试图学习一个比观察结果更高维度的表示，没有强有力的先验知识是没有用的。另一方面，学习低维潜在变量也可以看作是一种压缩形式，并且有可能揭示出描述观察结果的语义上有意义的结构。

### 证据下界（Evidence Lower Bound, ELBO）

数学上，我们可以想象潜在变量和我们观察到的数据由一个联合分布  $ p(x, z)$  来建模。生成建模中称为“基于似然的”方法之一是学习一个模型来最大化所有观测到的 $ x$ 的似然  $ p(x)$。有两种方法可以操作这个联合分布来恢复我们观测数据的似然 $ p(x)$， 我们可以明确地对潜在变量 $ z$ 积分，计算边缘分布 $ p(x)$：
$$
\begin{equation}
p(\boldsymbol{x}) = \int p(\boldsymbol{x}, \boldsymbol{z})d\boldsymbol{z}
\end{equation}
$$


或者，我们也可以使用概率链式法则：

$$
\begin{equation}
p(\boldsymbol{x}) = \frac{p(\boldsymbol{x}, \boldsymbol{z})}{p(\boldsymbol{z}\mid\boldsymbol{x})}
\end{equation}
$$

直接计算和最大化似然 $ p(x)$ 是困难的，因为它要么涉及将所有潜在变量 $ z$ 积分出来（方程 (1) ）, 这对于复杂模型来说是不可行的；要么涉及获得真实的潜在编码器 $ p(z|x)$ （方程 (2) ）。

然而，使用这两个方程，我们可以推导出一个称为证据下界（ELBO）的项，顾名思义，它是证据的一个下界。在这种情况下，证据被量化为观测数据的对数似然。然后，最大化ELBO成为一个代理目标，用于优化潜在变量模型；在最好的情况下，当ELBO被正确地参数化并且被完美优化时，它与证据完全等价。

正式地，证据下界 ELBO 的方程是：
$$
\begin{equation}
\mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z}\mid\boldsymbol{x})}\left[\log\frac{p(\boldsymbol{x}, \boldsymbol{z})}{q_{\boldsymbol{\phi}}(\boldsymbol{z}\mid\boldsymbol{x})}\right]
\end{equation}
$$

为了明确与证据的关系，我们可以数学上写为：

$$
\begin{equation}
\log p(\boldsymbol{x}) \geq \mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z}\mid\boldsymbol{x})}\left[\log\frac{p(\boldsymbol{x}, \boldsymbol{z})}{q_{\boldsymbol{\phi}}(\boldsymbol{z}\mid\boldsymbol{x})}\right]
\end{equation}
$$

这里，$ q_{\phi}(z|x)$ 是一个灵活的近似变分分布，其参数 $ \phi$ 是我们要寻求优化的。直观上，它可以被看作是一个可参数化的模型，用于学习估计给定观测 $ x$ 的潜在变量的真实分布；换句话说，它寻求近似真实的后验  $ p(z|x)$。当我们通过调整参数 $ \phi$ 来增加下界以最大化ELBO时，我们获得了可以用来建模真实数据分布并从中采样的组件，从而学习一个生成模型。

>变分分布：
>
>变分分布是变分推断中用于近似复杂后验分布的简化模型。它通过优化一个参数化的分布族来捕捉原始分布的关键特征，通常选择易于计算且足够灵活的分布作为变分分布族（如最为常见的正态分布），并通过最小化与真实后验分布的KL散度来调整参数。当这两个分布差距很小时，就可以用新分布去代替实际要求的分布了。这种方法使得原本在计算上不可行的复杂概率分布推断变得可行。

### 证据下界（ELBO）的推导

让我们从方程(1)开始推导证据下界 ELBO：
$$
\begin{align}
\log p(\boldsymbol{x})
&= \log \int p(\boldsymbol{x}, \boldsymbol{z})d\boldsymbol{z}\\
&= \log \int \frac{p(\boldsymbol{x}, \boldsymbol{z})q_{\boldsymbol{\phi}}(\boldsymbol{z}\mid\boldsymbol{x})}{q_{\boldsymbol{\phi}}(\boldsymbol{z}\mid\boldsymbol{x})}d\boldsymbol{z}\\
&= \log \mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z}\mid\boldsymbol{x})}\left[\frac{p(\boldsymbol{x}, \boldsymbol{z})}{q_{\boldsymbol{\phi}}(\boldsymbol{z}\mid\boldsymbol{x})}\right]\\
&\geq \mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z}\mid\boldsymbol{x})}\left[\log \frac{p(\boldsymbol{x}, \boldsymbol{z})}{q_{\boldsymbol{\phi}}(\boldsymbol{z}\mid\boldsymbol{x})}\right]
\end{align}
$$

> (6)->(7)： 将对随机变量的积分转换为对其概率分布的期望值，意味着我们不是直接计算随机变量在所有可能取值上的积分，而是计算这个随机变量按照其概率分布取值时的平均值。这种转换是基于期望值的定义，即一个随机变量的期望值是它所有可能取值的概率加权和。
>
> 在变分推断中，由于直接计算(5)这个积分可能非常复杂，因此利用期望的性质来简化问题。期望的性质允许我们将对随机变量的积分转换为对其概率分布的期望值，进一步可以通过蒙特卡洛采样方法来进行估计。

>(7)->(8)：Jensen不等式是一个凸函数性质的结果，对于任意凸函数 $ f $ 和概率分布 $P$，以下不等式恒成立：$ \log\mathbb{E}_P[X]\geq\mathbb{E}_P[\log X]$

在这个推导中，我们通过应用 Jensen不等式 直接得到了我们的下界。然而，这并没有提供关于实际发生的情况的有用信息；关键的是，这个证明没有给出为什么ELBO 实际上是证据下界的直觉，因为 Jensen不等式 把它模糊处理了。此外，仅仅知道ELBO是数据的下界，并没有真正告诉我们为什么我们想要将其最大化作为目标。为了更好地理解证据和ELBO之间的关系，让我们进行另一个推导，这次使用方程(2)：
$$
\begin{align}
\log p(\boldsymbol{x}) & = \log p(\boldsymbol{x}) \int q_{\boldsymbol{\phi}}(\boldsymbol{z}\mid\boldsymbol{x})dz\\
          & = \int q_{\boldsymbol{\phi}}(\boldsymbol{z}\mid\boldsymbol{x})\log p(\boldsymbol{x})dz\\
          & = \mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z}\mid\boldsymbol{x})}\left[\log p(\boldsymbol{x})\right]\\
          & = \mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z}\mid\boldsymbol{x})}\left[\log\frac{p(\boldsymbol{x}, \boldsymbol{z})}{p(\boldsymbol{z}\mid\boldsymbol{x})}\right]\\
          & = \mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z}\mid\boldsymbol{x})}\left[\log\frac{p(\boldsymbol{x}, \boldsymbol{z})q_{\boldsymbol{\phi}}(\boldsymbol{z}\mid\boldsymbol{x})}{p(\boldsymbol{z}\mid\boldsymbol{x})q_{\boldsymbol{\phi}}(\boldsymbol{z}\mid\boldsymbol{x})}\right]\\
          & = \mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z}\mid\boldsymbol{x})}\left[\log\frac{p(\boldsymbol{x}, \boldsymbol{z})}{q_{\boldsymbol{\phi}}(\boldsymbol{z}\mid\boldsymbol{x})}\right] + \mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z}\mid\boldsymbol{x})}\left[\log\frac{q_{\boldsymbol{\phi}}(\boldsymbol{z}\mid\boldsymbol{x})}{p(\boldsymbol{z}\mid\boldsymbol{x})}\right]\\
          & = \mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z}\mid\boldsymbol{x})}\left[\log\frac{p(\boldsymbol{x}, \boldsymbol{z})}{q_{\boldsymbol{\phi}}(\boldsymbol{z}\mid\boldsymbol{x})}\right] + \mathcal{D}_{\text{KL}}(q_{\boldsymbol{\phi}}(\boldsymbol{z}\mid\boldsymbol{x}) \mid\mid p(\boldsymbol{z}\mid\boldsymbol{x})) \\
          & \geq \mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z}\mid\boldsymbol{x})}\left[\log\frac{p(\boldsymbol{x}, \boldsymbol{z})}{q_{\boldsymbol{\phi}}(\boldsymbol{z}\mid\boldsymbol{x})}\right]
\end{align}
$$

> KL散度：
>
> KL散度也称为相对熵（Relative Entropy），是衡量两个概率分布差异的一种方法。它是两个概率分布 P 和 Q 之间的非对称距离度量，定义为：$D_{KL}(P||Q)=\sum_xP(x)\log\left(\frac{P(x)}{Q(x)}\right)$，其中，P 是数据的真实分布，而 Q  是模型或估计分布。KL散度的值总是非负的，当且仅当 P 和 Q 完全相同时，KL散度为零。

从这个推导中，我们清楚地从方程 (15) 观察到证据等于 证据下界ELBO 加上近似后验 $ q_{\phi}(z|x)$ 和真实后验 $ p(z|x)$ 之间的KL散度。实际上，正是这个KL散度项在第一个推导的方程(8)中通过 Jensen不等式 被神奇地移除了。理解这个项是理解证据下界ELBO 和证据之间关系的关键，也是理解为什么优化ELBO是一个合适的目标的原因。

首先，我们现在知道为什么ELBO确实是一个下界：

证据和ELBO之间的差是一个严格非负的KL项，因此 ELBO 的值永远不会超过证据。

其次，我们探讨为什么寻求最大化ELBO。

在我们引入了想要建模的潜在变量 $ z$ 之后，我们的目标是学习描述我们观测数据的潜在结构。换句话说，我们想要优化我们的变分后验 $ q_{\phi}(z|x)$ 的参数，以便它完全匹配真实的后验分布 $ p(z|x)$，这通过最小化它们的KL散度（理想情况下为零）来实现的。不幸的是，直接最小化这个 KL散度项是不可行的，因为我们没有无法访问真实的 $ p(z|x)$ 分布。

但是，请观察，在方程(15)的左侧，我们数据的似然性（因此我们的证据项 $ \log p(x)$）相对于 $ \phi$ 总是一个常数，因为它是通过从联合分布 $ p(x, z)$ 中边缘化所有潜在 $ z$ 来计算的，根本不依赖于 $ \phi$。由于证据下界 ELBO 和 KL 散度项加起来是一个常数，对证据下界 ELBO项的任何最大化都必然引起KL散度项的等量最小化。

因此，证据下界ELBO可以被最大化，作为学习完美地模拟真实潜在后验分布的代理目标；我们越优化ELBO，我们的近似后验就越接近真实后验。此外，一旦训练完成，ELBO 还可以用来估计观测或生成数据的可能性，因为它被学习来近似模型证据 $ \log p(x)$。

### 变分自编码器（Variational Autoencoders）



> VAE：
>
> 在VAE的架构中，设计两个网络，一个网络做Encoder，将观察的数据 $x$ 映射到隐变量 $z$ 上，另一个网络做Decoder，从采样的 $z$ 中解码回$x$ 。从实践的角度，VAE需要训练两套参数 $\theta$ 和 $\phi$，即$ q_{\phi}(z|x)$ 和  $ p_{\theta}(x|z)$  分别对应 encoder和decoder。
>
> VAE的目标函数采用的是最大似然（likelihood-based）的路线，即对观察到的数据  $x$  计算最大$ p(x)$，借助数学工具ELBO（Evidence Lower Bound），可以得到最大似然分布的一个下界。

在变分自编码器（VAE）的默认方程中，我们直接最大化证据下界（ELBO）。这种方法是变分的，因为我们在由 $\phi$ 参数化的潜在后验分布族中优化最佳的 $ q_{\phi}(z|x)$。它被称为自编码器，因为它让人想起传统的自编码器模型，其中输入数据被训练在经历一个中间瓶颈表示步骤后进行预测。为了明确这种联系，让我们进一步剖析ELBO项：
$$
\begin{align}
\mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z}\mid\boldsymbol{x})}\left[\log\frac{p(\boldsymbol{x}, \boldsymbol{z})}{q_{\boldsymbol{\phi}}(\boldsymbol{z}\mid\boldsymbol{x})}\right]
&= \mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z}\mid\boldsymbol{x})}\left[\log\frac{p_{\boldsymbol{\theta}}(\boldsymbol{x}\mid\boldsymbol{z})p(\boldsymbol{z})}{q_{\boldsymbol{\phi}}(\boldsymbol{z}\mid\boldsymbol{x})}\right]\\
&= \mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z}\mid\boldsymbol{x})}\left[\log p_{\boldsymbol{\theta}}(\boldsymbol{x}\mid\boldsymbol{z})\right] + \mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z}\mid\boldsymbol{x})}\left[\log\frac{p(\boldsymbol{z})}{q_{\boldsymbol{\phi}}(\boldsymbol{z}\mid\boldsymbol{x})}\right]\\
&= \underbrace{\mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z}\mid\boldsymbol{x})}\left[\log p_{\boldsymbol{\theta}}(\boldsymbol{x}\mid\boldsymbol{z})\right]}_\text{reconstruction term} - \underbrace{\mathcal{D}_{\text{KL}}(q_{\boldsymbol{\phi}}(\boldsymbol{z}\mid\boldsymbol{x}) \mid\mid p(\boldsymbol{z}))}_\text{prior matching term}
\end{align}
$$

在这个情况下，我们学习了一个中间瓶颈分布 $ q_{\phi}(z|x)$，它可以被看作是一个编码器；将输入转换为可能的潜在分布。同时，我们学习了一个确定性函数 $ p_{\theta}(x|z)$ 将给定的潜在向量 $ z$ 转换为观测 $ x$，这可以被解释为解码器。


<div align=center>
<img src="https://calvinyluo.com/assets/images/diffusion/vae.webp" alt="Visualizing a Variational Autoencoder" style="zoom:33%;" />
</div>


> 可视化普通变分自动编码器。通过重参数化技巧联合学习潜在编码器和解码器 [6, 7]

方程(19)中的两项各自有直观的描述：

- 第一项测量了解码器从我们的变分分布中重构的似然性；这确保了学习到的分布是模拟有效的潜在变量，原始数据可以从中再生。
- 第二项测量了学习到的变分分布与我们对潜在变量持有的先验信念的相似性。最小化这个项鼓励编码器实际学习一个分布，而不是塌缩成一个狄拉克δ函数。
- 通俗来说，第一项要求 Decoder 的输出尽量接近原图；第二项代表Encoder的分布要接近真实分布$p(𝑧)$ .

因此，最大化ELBO等同于最大化其第一项并最小化其第二项。

VAE的一个决定性特征是ELBO是如何联合优化参数 $\phi$ 和 $\theta$。VAE的编码器 $ q_{\phi}(z|x)$ 通常选择为具有对角线协方差的多元高斯分布进行建模，而先验 $ p_{\theta}(z)$ 通常选择为标准多元高斯分布：

$$
\begin{align}
    q_{\boldsymbol{\phi}}(\boldsymbol{z}\mid\boldsymbol{x}) &= \mathcal{N}(\boldsymbol{z}; \boldsymbol{\mu}_{\boldsymbol{\phi}}(\boldsymbol{x}), \boldsymbol{\sigma}_{\boldsymbol{\phi}}^2(\boldsymbol{x})\textbf{I})\\
    p(\boldsymbol{z}) &= \mathcal{N}(\boldsymbol{z}; \boldsymbol{0}, \textbf{I})
\end{align}
$$


然后，ELBO中的KL散度项可以被解析的计算出来，而重建项可以使用蒙特卡洛估计来近似。我们的目标可以被重写为：
$$
\begin{align}
& \quad \,\underset{\boldsymbol{\phi}, \boldsymbol{\theta}}{\arg\max}\, \mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z}\mid\boldsymbol{x})}\left[\log p_{\boldsymbol{\theta}}(\boldsymbol{x}\mid\boldsymbol{z})\right] - \mathcal{D}_{\text{KL}}(q_{\boldsymbol{\phi}}(\boldsymbol{z}\mid\boldsymbol{x}) \mid\mid p(\boldsymbol{z})) \nonumber \\
& \approx \underset{\boldsymbol{\phi}, \boldsymbol{\theta}}{\arg\max}\, \sum_{l=1}^{L}\log p_{\boldsymbol{\theta}}(\boldsymbol{x}\mid\boldsymbol{z}^{(l)}) - \mathcal{D}_{\text{KL}}(q_{\boldsymbol{\phi}}(\boldsymbol{z}\mid\boldsymbol{x}) \mid\mid p(\boldsymbol{z}))
\end{align}
$$

其中潜在变量 $ \{z^{(l)}\}^{L}_{l=1}$ 是从编码分布 $ q_{\phi}(z|x)$ 中采样的，对于数据集中的每个观测 $ x$ 都是如此。然而，在这个默认设置中出现了一个问题：我们计算损失的每个 $ z^{(l)}$ 都是通过一个随机采样过程生成的，这通常是不可微分的。幸运的是，当 $ q_{\phi}(z|x)$ 被设计为模拟某些分布（包括多元高斯分布）时，这个问题可以通过重参数化技巧来解决。

> 重参数化技巧
>
> 重参数化技巧是机器学习和深度学习中用于处理随机梯度问题的一种方法。在许多模型中，特别是在生成模型如变分自编码器（VAE）中，我们经常需要从概率分布中采样，而这些采样操作本身是随机的，不可导的。这就意味着我们不能直接通过这些随机采样来计算梯度，并用它们来更新模型参数。
>
> 为了解决这个问题，重参数化技巧通过引入一个新的随机变量，并将其作为一个噪声变量来重新表达原始的随机采样。这个新的表达式是原始随机变量的一个确定性函数，这意味着它完全依赖于模型参数和一个新的独立的随机变量。这样，我们就可以通过这个新的随机变量来计算原始随机变量的期望值，而这个期望值是可导的，因为它不再依赖于模型参数的随机选择。
>
> 具体来说，假设我们有一个随机变量 $Z$ ，它是通过从某个分布 P(Z)中采样得到的。在没有重参数化技巧的情况下，我们不能直接对 $Z$ 进行梯度下降优化。但是，如果我们能够找到一种方法，使得 Z 可以被表达为一个确定性函数 g 和一个独立的随机变量 $\epsilon$ 的组合，即 $Z = g(\epsilon; \theta)$，其中 $\theta$ 是模型参数，那么我们就可以使用 $\epsilon $ 来计算 Z 的期望值，并通过 $\epsilon$ 的分布来计算梯度。这种方法的关键优势在于，尽管 Z 本身是随机的，但 $ g(\epsilon; \theta)$ 是确定性的，并且可以通过 $\epsilon$ 来计算梯度。因此，我们可以使用标准的梯度下降算法来优化模型参数 $ \theta$ ，即使整个表达式涉及到随机变量。

在VAE的情况下，重参数化技巧允许我们将编码器输出的随机潜在变量 $Z$ 重写为一个关于模型参数 $\phi$ 的确定性函数，同时依赖于一个独立的随机变量。这样，我们就可以通过梯度下降来优化编码器和解码器的参数，同时保持生成的数据与真实数据的相似性。

例如，从正态分布 $ x \sim \mathcal{N}(x; \mu, \sigma^2)$ 中采样的随机变量，其中 $ \mu$ 是任意的均值，$ \sigma^2$ 是方差，可以被重写为：

$$
\begin{align*}
    x &= \mu + \sigma\epsilon \quad \text{with } \epsilon \sim \mathcal{N}(\epsilon; 0, \text{I})
\end{align*}
$$

换句话说，任意的高斯分布可以被解释为标准高斯（其中 $ \epsilon$ 是一个样本），通过添加将均值从零移动到目标均值 $ \mu$，以及通过目标方差 $ \sigma^2$ 拉伸方差。因此，通过重参数化技巧，从任意高斯分布中采样可以通过从标准高斯中采样，将结果乘以目标标准差，然后通过目标均值进行平移来执行。

在VAE中，每个潜在变量 $ z$ 被因此可以被计算为输入 $ x$ 和辅助噪声变量 $ \epsilon$ 的确定性函数：

$$
\begin{align*}
    \boldsymbol{z} &= \boldsymbol{\mu}_{\boldsymbol{\phi}}(\boldsymbol{x}) + \boldsymbol{\sigma}_{\boldsymbol{\phi}}(\boldsymbol{x})\odot\boldsymbol{\epsilon} \quad \text{with } \boldsymbol{\epsilon} \sim \mathcal{N}(\boldsymbol{\epsilon};\boldsymbol{0}, \textbf{I})
\end{align*}
$$

在重参数化的 $ z$ 版本下，可以计算相对于 $ \phi$ 的梯度，以期望优化 $ \mu_{\phi}$ 和 $ \sigma_{\phi}$。因此，VAE  利用重参数化技巧和蒙特卡洛估计联合优化证据下界ELBO。其中 $ \odot$ 表示逐元素乘积。

训练完VAE后，可以通过直接从潜在空间 $ p(z)$ 中采样，然后将其通过解码器来生成新数据。当潜在变量 $ z$ 的维度小于输入 $ x$ 的维度时，变分自编码器特别有趣，因为那时我们可能学习紧凑、有用的表示。此外，当学习到一个语义上有意义的潜在空间时，潜在向量可以在传递给解码器之前进行编辑，以更精确地控制生成的数据。

### 分层变分自编码器（Hierarchical Variational Autoencoders）

分层变分自编码器（HVAE）是变分自编码器（VAE）的推广，可扩展到潜在变量的多个层次结构。在这个框架下，潜在变量本身被视为由其他更高级别的、更抽象的潜在变量生成。就像我们将三维观察对象视为从更高层次的抽象潜在对象生成一样，柏拉图洞穴中的人们将三维对象视为生成二维观察的潜在对象。因此，从柏拉图洞穴居民的视角来看，他们的观察可以被视为由两个或更多层次的潜在层次结构建模。

<div align=center>
<img src="https://calvinyluo.com/assets/images/diffusion/hvae.webp" alt="Visualizing a Hierarchical VAE" style="zoom:50%;" />
</div>



> 具有 T 个分层潜在变量的马尔可夫分层变分自动编码器。生成过程被建模为马尔可夫链，其中每个潜在变量$z_{t}$仅从前一个潜在变量$z_{t+1}$生成。

与具有T层层次结构的一般分层变分自编码器（HVAE）不同， 在一般的HVAE中，每个潜在变量被允许依赖于之前所有的潜在变量。然而，在本研究中，我们关注的是一个特殊案例，我们称之为马尔可夫分层变分自编码器（MHVAE）。在MHVAE中，生成过程是一个马尔可夫链，也就是说，层次结构中的每个转换都是马尔可夫的，其中解码每个潜在变量 $ z_t$ 仅依赖于前一个潜在变量 $ z_{t+1}$。这可以被看作是简单地将变分自编码器（VAEs）堆叠在一起，如图2所示；描述这个模型的另一个恰当术语是递归变分自编码器（Recursive VAE）。数学上，我们表示马尔可夫分层变分自编码器的联合分布和后验分布为：
$$
\begin{align}
    p(\boldsymbol{x}, \boldsymbol{z}_{1:T}) &= p(\boldsymbol{z}_T)p_{\boldsymbol{\theta}}(\boldsymbol{x}\mid\boldsymbol{z}_1)\prod_{t=2}^{T}p_{\boldsymbol{\theta}}(\boldsymbol{z}_{t-1}\mid\boldsymbol{z}_{t})
\end{align}
$$


他的后验表示为：
$$
\begin{align}
    q_{\boldsymbol{\phi}}(\boldsymbol{z}_{1:T}\mid\boldsymbol{x}) &= q_{\boldsymbol{\phi}}(\boldsymbol{z}_1\mid\boldsymbol{x})\prod_{t=2}^{T}q_{\boldsymbol{\phi}}(\boldsymbol{z}_{t}\mid\boldsymbol{z}_{t-1})
\end{align}
$$


然后，我们可以很容易地将证据下界ELBO扩展为：
$$
\begin{align}
\log p(\boldsymbol{x}) &= \log \int p(\boldsymbol{x}, \boldsymbol{z}_{1:T}) d\boldsymbol{z}_{1:T}\\
&= \log \int \frac{p(\boldsymbol{x}, \boldsymbol{z}_{1:T})q_{\boldsymbol{\phi}}(\boldsymbol{z}_{1:T}\mid\boldsymbol{x})}{q_{\boldsymbol{\phi}}(\boldsymbol{z}_{1:T}\mid\boldsymbol{x})} d\boldsymbol{z}_{1:T}\\
&= \log \mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z}_{1:T}\mid\boldsymbol{x})}\left[\frac{p(\boldsymbol{x}, \boldsymbol{z}_{1:T})}{q_{\boldsymbol{\phi}}(\boldsymbol{z}_{1:T}\mid\boldsymbol{x})}\right]\\
&\geq \mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z}_{1:T}\mid\boldsymbol{x})}\left[\log \frac{p(\boldsymbol{x}, \boldsymbol{z}_{1:T})}{q_{\boldsymbol{\phi}}(\boldsymbol{z}_{1:T}\mid\boldsymbol{x})}\right]
\end{align}
$$


然后，我们可以将我们的联合分布（方程(23)）和后验（方程(24)）代入方程(28)，以产生另一种形式：

$$
\begin{align}
\mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z}_{1:T}\mid\boldsymbol{x})}\left[\log \frac{p(\boldsymbol{x}, \boldsymbol{z}_{1:T})}{q_{\boldsymbol{\phi}}(\boldsymbol{z}_{1:T}\mid\boldsymbol{x})}\right]
&= \mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z}_{1:T}\mid\boldsymbol{x})}\left[\log \frac{p(\boldsymbol{z}_T)p_{\boldsymbol{\theta}}(\boldsymbol{x}\mid\boldsymbol{z}_1)\prod_{t=2}^{T}p_{\boldsymbol{\theta}}(\boldsymbol{z}_{t-1}\mid\boldsymbol{z}_{t})}{q_{\boldsymbol{\phi}}(\boldsymbol{z}_1\mid\boldsymbol{x})\prod_{t=2}^{T}q_{\boldsymbol{\phi}}(\boldsymbol{z}_{t}\mid\boldsymbol{z}_{t-1})}\right]
\end{align}
$$
正如我们将在下面探讨变分扩散模型时所指出的，这个目标可以进一步分解为可解释的组成部分。

## 变分扩散模型（Variational Diffusion Models）

> 基本概念：
>
> 业界针对VAE的问题做了很多尝试，Diffusion Model 可以算作是其中一种比较成功的尝试。以图片生成举例，VAE 想一步到位，同时训练Encoder和Decoder，将原始图片映射到隐变量中，又能解码回来，端到端地学习整个过程太难了。有没有什么办法可以降低问题的难度？一个常见的思路是将困难的问题分而治之，即能不能只训练Encoder或Decoder？能不能把问题分解成一个个子问题？

变分扩散模型（VDM）可以简单地看作是一个具有三个关键限制的马尔可夫分层变分自编码器（Markovian Hierarchical Variational Autoencoder, MHVAE）：

1. 潜在维度完全等于数据维度。
2. 每个时间步的潜在编码器的结构不是学习得到的，它被预定义为线性高斯模型。换句话说，它是以前一时间步的输出为中心的高斯分布。
3. 潜在编码器的高斯分布参数随时间变化，使得最终时间步T的潜在分布是标准高斯分布。

此外，我们明确保持了从标准马尔可夫分层变分自编码器继承的分层转换之间的马尔可夫性质。让我们进一步阐述这些假设的含义。

根据第一个限制，现在我们可以将真实的数据样本和潜在变量表示为 $ x_t$，其中 $ t = 0$ 表示真实的数据样本，$ t \in [1, T]$ 表示具有层次索引 $t$ 的相应潜在变量。变分扩散模型的后验与MHVAE的后验相同（方程24），现在可以重写为：
$$
\begin{align}
    q(\boldsymbol{x}_{1:T}\mid\boldsymbol{x}_0) = \prod_{t = 1}^{T}q(\boldsymbol{x}_{t}\mid\boldsymbol{x}_{t-1})
\end{align}
$$
根据第二个假设，我们知道每个潜在变量在编码器中的分布是一个以它前一个层次潜在变量为中心的高斯分布。与MHVAE不同，在变分扩散模型中，每个时间步 $t$ 的编码器的结构不是学习得到的，它被固定为线性高斯模型，其中均值和标准差可以预先设置为超参数，或者作为参数学习。

我们用均值 $ \mu_t(x_t) = \sqrt{\alpha_t} x_{t-1}$ 和方差 $ \Sigma_t(x_t) = (1 - \alpha_t)I$ 来参数化高斯编码器，其中系数的形式被选择为保持潜在变量的方差在相似的尺度上；换句话说，编码过程是方差保持的， 请注意，允许使用其他高斯参数化方法，并且可以得到类似的推导结果。主要的收获是 $ \alpha_t$ 是一个（可能是可学习的）系数，它可以根据层次深度 $t$ 变化，以增加灵活性。数学上，编码器的转换表示为：
$$
\begin{align}
    q(\boldsymbol{x}_{t}\mid\boldsymbol{x}_{t-1}) = \mathcal{N}(\boldsymbol{x}_{t} ; \sqrt{\alpha_t} \boldsymbol{x}_{t-1}, (1 - \alpha_t) \textbf{I})
\end{align}
$$


根据第三个假设，我们知道 $ \alpha_t$ 按照固定或可学习的时间表随时间演变，使得最终潜在的分布 $ p(x_T)$ 是标准高斯分布。然后，我们可以更新马尔可夫分层变分自编码器（MHVAE）的联合分布（方程23），将变分扩散模型（VDM）的联合分布写为：

$$
\begin{align}
p(\boldsymbol{x}_{0:T}) &= p(\boldsymbol{x}_T)\prod_{t=1}^{T}p_{\boldsymbol{\theta}}(\boldsymbol{x}_{t-1}\mid\boldsymbol{x}_t) \\
\text{where,}&\nonumber\\
p(\boldsymbol{x}_T) &= \mathcal{N}(\boldsymbol{x}_T; \boldsymbol{0}, \textbf{I})
\end{align}
$$
总的来说，这组假设描述了图像输入随时间逐步噪声化的过程：我们通过逐步添加高斯噪声逐渐破坏图像，直到最终它完全变成纯高斯噪声。这个过程在图3中以视觉方式呈现。

<div align=center>
<img src="https://calvinyluo.com/assets/images/diffusion/vdm_base.webp" alt="Visualizing a Variational Diffusion Model" style="zoom:50%;" />
</div>



> 变分扩散模型的视觉表示。输入随着时间的推移稳定地产生噪声，直到它变得与高斯噪声相同；扩散模型学习逆向过程。

请观察，我们的编码器分布 $ q(x_t | x_{t-1})$ 已经不再由 $\phi$ 参数化，因为它们完全被建模为在每个时间步骤具有确定的均值和方差参数的高斯分布。因此，在变分扩散模型中，我们只对条件分布 $ p_{\theta}(x_{t-1} | x_t)$ 感兴趣，这样我们就可以模拟新数据。在优化变分扩散模型之后，采样过程就像从 $ p(x_T)$ 中采样高斯噪声，然后迭代地运行去噪转换 $ p_{\theta}(x_{t-1} | x_t)$  进行 $T $ 步来生成一个新的 $ x_0$。

#### 第一种常见解释

像任何分层变分自编码器 HVAE一样，变分扩散模型可以通过最大化证据下界ELBO来优化，它可以被推导为：
$$
\begin{align}
\log p(\boldsymbol{x})
&\geq \mathbb{E}_{q(\boldsymbol{x}_{1:T}\mid\boldsymbol{x}_0)}\left[\log \frac{p(\boldsymbol{x}_{0:T})}{q(\boldsymbol{x}_{1:T}\mid\boldsymbol{x}_0)}\right]  \\
&=  \begin{aligned}[t]
      \underbrace{\mathbb{E}_{q(\boldsymbol{x}_{1}\mid\boldsymbol{x}_0)}\left[\log p_{\theta}(\boldsymbol{x}_0\mid\boldsymbol{x}_1)\right]}_\text{reconstruction term} &- \underbrace{\mathbb{E}_{q(\boldsymbol{x}_{T-1}\mid\boldsymbol{x}_0)}\left[\mathcal{D}_{\text{KL}}(q(\boldsymbol{x}_T\mid\boldsymbol{x}_{T-1}) \mid\mid p(\boldsymbol{x}_T))\right]}_\text{prior matching term} \\
      &- \sum_{t=1}^{T-1}\underbrace{\mathbb{E}_{q(\boldsymbol{x}_{t-1}, \boldsymbol{x}_{t+1}\mid\boldsymbol{x}_0)}\left[\mathcal{D}_{\text{KL}}(q(\boldsymbol{x}_t\mid\boldsymbol{x}_{t-1}) \mid\mid p_{\theta}(\boldsymbol{x}_{t}\mid\boldsymbol{x}_{t+1}))\right]}_\text{consistency term}
    \end{aligned}
\end{align}
$$

证明过程如下：

<div align=center>
<img src="https://calvinyluo.com/assets/images/diffusion/proofs/first_deriv_verbose.svg" alt="Deriving the ELBO of a VDM in terms of Consistency terms" style="zoom:100%;" />
</div>


> 注：
>
> (34)：边缘分布的概念
>
> (36)：期望的定义
>
> (37)：Jessen不等式
>
> (42)-(43)：求和的期望（期望对求和的操作）与期望的求和（求和对期望的操作）在数学上是相等的。这是由于期望的线性属性所保证的。
>
> (43)-(44)：根据表达式中的下标，对下标具体化
>
> (44)-(45)：这里解释其中第二项 $\mathbb{E}_{q(\boldsymbol{x}_{T-1},\boldsymbol{x}_T|\boldsymbol{x}_0)}\left[\log\frac{p(\boldsymbol{x}_T)}{q(\boldsymbol{x}_T|\boldsymbol{x}_{T-1})}\right] = -\mathbb{E}_{q(\boldsymbol{x}_{T-1}|\boldsymbol{x}_{0})}\left[D_{\mathrm{KL}}(q(\boldsymbol{x}_{T}|\boldsymbol{x}_{T-1})\parallel p(\boldsymbol{x}_{T}))\right]$，第三项同理，期望是在 $q(\boldsymbol{x}_{T-1},\boldsymbol{x}_T|\boldsymbol{x}_0)$这个联合分布下计算的，意味着考虑了$\boldsymbol{x}_{T-1}$ 和 $\boldsymbol{x}_T$ 的所有可能的组合。而 $-\mathbb{E}_{q(\boldsymbol{x}_{T-1}|\boldsymbol{x}_0)}\left[D_{\mathrm{KL}}(q(\boldsymbol{x}_T|\boldsymbol{x}_{T-1})\parallel p(\boldsymbol{x}_T))\right] $是首先对 $\boldsymbol{x}_{T-1} $求期望，然后计算给定 $\boldsymbol{x}_{T-1}$时，$\boldsymbol{x}_T$ 的分布 $q(\boldsymbol{x}_{T}|\boldsymbol{x}_{T-1})$ 与其先验 $p(\boldsymbol{x}_{T}) $ 之间的KL散度的负值。这两个表达式等价的原因如下：
>
> - 期望的性质：由于KL散度内部本身就是一个对数期望形式，这个期望的外部和内部可以合并，这是因为期望的计算是线性的，可以跨越概率分布的边界。
>
> - KL散度的定义：上述的KL散度定义直接对应于你的第一个表达式的内部。KL散度衡量的是在特定条件下，两个分布的相对熵。
>   所以，当从 $q(\boldsymbol{x}_{T-1},\boldsymbol{x}_T|\boldsymbol{x}_0) $这个条件分布计算期望时，实际上是在遍历所有 $\boldsymbol{x}_{T-1} $ 和$ \boldsymbol{x}_T $ 的组合，并评估 $q(\boldsymbol{x}_T|\boldsymbol{x}_{T-1}) $ 和 $p(\boldsymbol{x}_T) $之间的相对熵。当你把这个过程表示为KL散度，并注意到KL散度本身就是期望的形式时，两边就能够匹配上，并通过将KL散度的定义带入，完成从左侧到右侧的转换。



最后三行公式是对变分扩散模型（VDM）中的证据下界（ELBO）进行分解和解释的过程。ELBO是变分推断中的一个重要概念，用于衡量模型对数据的拟合程度。ELBO被分解为三个主要部分：重构项（reconstruction term）、先验匹配项（prior matching term）和一致性项（consistency term）。下面是各个组成部分的解释：

1. 重建项: $ \mathbb{E}_{q(x_1|x_0)} \left[ \log p_{\theta}(x_0 | x_1) \right]$ ，这一项计算的是给定第一步潜在变量 $x_{1}$ 时，预测原始数据 $x_0$ 的对数概率。这一项反映了模型重构数据的能力，与普通VAE中的重构项类似。
2. 先验匹配项: $ \mathbb{E}_{q(x_{T-1}|x_0)} \left[ \text{KL}(q(x_T | x_{T-1}) \| p(x_T)) \right]$ ；这一项计算的是最终潜在变量$\boldsymbol{x}_T $ 的分布与高斯先验分布之间的对数比。当最终潜在变量的分布在所有步骤后与高斯先验分布匹配时，这一项的值最小。
   这一项不需要优化，因为它没有可训练的参数；此外，由于我们假设了一个足够大的T，使得最终分布是高斯分布，这个项实际上变成了零。
3. 一致性项: $ \mathbb{E}_{q(x_{t-1}, x_t, x_{t+1} | x_0)} \left[ \text{KL}(q(x_t | x_{t-1}) \| p_{\theta}(x_t | x_{t+1})) \right]$ ， 它努力使 $ x_t$ 的分布在两个方向上保持一致，即从更噪声的图像到更清晰的图像的去噪步骤和从更清晰的图像到更噪声的图像的加噪步骤相匹配。这种一致性在数学上通过KL散度反映出来。当两个去噪步骤尽可能匹配时，这一项被最小化，如方程31中定义的高斯分布 $ q(x_t | x_{t-1})$。

这些项的和构成了VDM的ELBO，它是优化过程中的目标函数。通过最大化ELBO，我们可以调整模型参数 $\boldsymbol{\theta}$ 来提高模型对数据的拟合程度，同时确保生成过程的一致性。这个过程涉及到对联合分布和条件分布的积分进行计算，通常需要使用蒙特卡洛估计和重参数化技巧来近似这些期望值。通过这种方式，我们可以有效地训练VDM模型，使其能够生成与训练数据相似的新样本。


<div align=center>
<img src="https://calvinyluo.com/assets/images/diffusion/first_derivation.webp" alt="Deriving the ELBO of a VDM in terms of Consistency terms" style="zoom:50%;" />
</div>


> 图 4：根据第一个推导，变分扩散模型（VDM）可以通过确保对于每一个中间潜在变量 $x_t$，来自上层潜在变量的后验分布 $p_\theta(x_t|x_{t+1}) $ 与来自下层潜在变量的高斯噪声破坏的分布  $q(x_t|x_{t-1}) $ 相匹配来优化。在这个图中，对于每一个中间潜在变量 $x_t$ ，最小化由粉色和绿色箭头所表示的分布之间的差异。

在图4中，我们以视觉方式展示了ELBO的这种解释；对于每个中间的 $ x_t$，我们最小化由粉色和绿色箭头表示的分布之间的差异。在完整的图中，每条粉色箭头也必须从 $ x_0$ 开始，因为它也是一个条件项。

优化一个变分扩散模型（VDM）的成本主要由第三项主导，因为我们必须要在所有时间步骤 t 上进行优化。在这种推导下，ELBO 的所有项都被计算为期望，因此可以使用蒙特卡洛估计来近似。然而，实际上使用我们刚刚推导出的项来优化ELBO可能是次优的；因为一致性项是作为两个随机变量 $ \{x_{t-1}, x_{t+1}\}$ 的期望来计算的，对于每个时间步，它的蒙特卡洛估计的方差可能会比使用每个时间步一个随机变量估计的项更高。由于它是通过对 $ T-1$ 个一致性项求和来计算的，对于大的 $ T$ 值，ELBO的最终估计值可能具有高方差。

> 注：原先推导出来一致性项是作为两个随机变量 $ \{\boldsymbol{x}_{t-1},\boldsymbol{x}_{t+1}\} $ 的期望值来计算的。当使用蒙特卡洛方法来近似这个期望值时，通常是通过从这些随机变量的分布中采样来进行的。由于每次采样都是独立的，当同时对两个随机变量进行采样时，这些采样点的组合将形成一个联合分布，其方差可能会比单个随机变量的方差更大。
>
> 在蒙特卡洛估计中，方差是衡量估计值稳定性的一个指标。高方差意味着估计值在不同的采样下会有较大的波动，这可能导致优化过程中的不稳定性。在VDM的情况下，由于一致性项是通过累加  T-1个这样的KL散度项来计算的，如果 T 很大，那么最终的ELBO估计值的方差可能会非常高，这会影响优化算法的性能和收敛速度。
>
> 为了解决这个问题，我们可以尝试推导出一个新的ELBO形式，其中每个项都是作为仅对一个随机变量的期望值来计算的。这样做可以降低蒙特卡洛估计的方差，从而提高优化过程的稳定性和效率。通过这种方式，我们可以更准确地估计ELBO，并更有效地优化模型参数。

让我们尝试推导出一个ELBO的形式，其中每个项都是对最多一个随机变量的期望进行计算。关键的洞察是我们可以重写编码器转换为 $ q(x_t | x_{t-1}) = q(x_t | x_{t-1}, x_0)$，其中额外的条件项由于马尔可夫性质而变得多余。然后，根据贝叶斯规则，我们可以将每个转换重写为：

$$
\begin{align}
q(\boldsymbol{x}_t \mid \boldsymbol{x}_{t-1}, \boldsymbol{x}_0) = \frac{q(\boldsymbol{x}_{t-1}\mid\boldsymbol{x}_t, \boldsymbol{x}_0)q(\boldsymbol{x}_t\mid\boldsymbol{x}_0)}{q(\boldsymbol{x}_{t-1}\mid\boldsymbol{x}_0)}
\end{align}
$$


利用这个新方程，ELBO 可以推导出为以下形式：

$$
\begin{align}
\log p(\boldsymbol{x})
&\geq \mathbb{E}_{q(\boldsymbol{x}_{1:T}\mid\boldsymbol{x}_0)}\left[\log \frac{p(\boldsymbol{x}_{0:T})}{q(\boldsymbol{x}_{1:T}\mid\boldsymbol{x}_0)}\right]\\
&=  \begin{aligned}[t]
      \underbrace{\mathbb{E}_{q(\boldsymbol{x}_{1}\mid\boldsymbol{x}_0)}\left[\log p_{\boldsymbol{\theta}}(\boldsymbol{x}_0\mid\boldsymbol{x}_1)\right]}_\text{reconstruction term} &- \underbrace{\mathcal{D}_{\text{KL}}(q(\boldsymbol{x}_T\mid\boldsymbol{x}_0) \mid\mid p(\boldsymbol{x}_T))}_\text{prior matching term} \\
      &- \sum_{t=2}^{T} \underbrace{\mathbb{E}_{q(\boldsymbol{x}_{t}\mid\boldsymbol{x}_0)}\left[\mathcal{D}_{\text{KL}}(q(\boldsymbol{x}_{t-1}\mid\boldsymbol{x}_t, \boldsymbol{x}_0) \mid\mid p_{\boldsymbol{\theta}}(\boldsymbol{x}_{t-1}\mid\boldsymbol{x}_t))\right]}_\text{denoising matching term}
    \end{aligned}
\end{align}
$$

证明过程如下：

<div align=center>
<img src="https://calvinyluo.com/assets/images/diffusion/proofs/second_deriv_verbose.svg" alt="根据去噪匹配项推导 VDM 的 ELBO" style="zoom:100%;" />
</div>


因此，我们已经成功地推导出了ELBO 新的解释，ELBO 可以分解为单个项的总和，这些项是一次最多一个随机变量的期望，因此具有较低的方差。这个方程还有一个优雅的解释，当检查每个单独的项时，可以揭示出来：

1. $ \mathbb{E}_{q(x_1 | x_0)} \left[ \log p_{\theta}(x_0 | x_1) \right]$ 可以被解释为一个重建项；像VAE的ELBO中的类似项一样，这个项可以使用蒙特卡洛估计来近似和优化。

2. $ \mathcal{D}_{\text{KL}}(q(\boldsymbol{x}_T\mid\boldsymbol{x}_0) \mid\mid p(\boldsymbol{x}_T)) $ 表示最终噪声输入的分布与标准高斯先验的接近程度。它没有可训练的参数，并且在我们的假设下也等于零。

3. $ \mathbb{E}_{q(\boldsymbol{x}_{t}\mid\boldsymbol{x}_0)}\left[\mathcal{D}_{\text{KL}}(q(\boldsymbol{x}_{t-1}\mid\boldsymbol{x}_t, \boldsymbol{x}_0) \mid\mid p_{\boldsymbol{\theta}}(\boldsymbol{x}_{t-1}\mid\boldsymbol{x}_t))\right]$ 是一个去噪匹配项。我们学习期望的去噪转换步骤 $ p_{\theta}(x_{t-1} | x_t)$ 作为可处理的、真实的去噪转换步骤 $ q(x_{t-1} | x_t, x_0)$ 的近似。由于 $ q(x_{t-1} | x_t, x_0)$ 转换步骤可以作为真实的信号，因为它定义了如何使用最终完全去噪的图像 $ x_0$ 来去噪噪声图像 $ x_t$。因此，当两个去噪步骤尽可能匹配时，这个项被最小化，通过它们之间的KL散度来衡量。

下图描述了此 ELBO 分解的直观解释：


<div align=center>
<img src="https://calvinyluo.com/assets/images/diffusion/second_derivation.webp" alt="Deriving the ELBO of a VDM in terms of Denoising Matching terms" style="zoom:50%;" />
</div>


> 变分扩散模型（VDM）也可以通过学习每个独立潜在变量的去噪步骤来优化，方法是将其与可计算的、可靠的真实去噪步骤相匹配。这再次通过视觉上将绿色箭头所代表的分布与粉色箭头所代表的分布相匹配来表示。这里运用了艺术上的自由；在完整的画面中，每个粉色箭头也必须源自真实图像，因为真实图像也是一个条件项。
>

作为旁注，我们观察到，在ELBO的推导过程中，只使用了马尔可夫假设；因此，这些方程对于任何任意的马尔可夫HVAE都是成立的。此外，当我们设置 $ T = 1$ 时，变分扩散模型的两种ELBO解释对于标准VAE的ELBO方程（方程19）是完全相同的。

在这种ELBO的推导中，大部分优化成本再次由求和项占据，它支配了重建项。虽然每个KL散度项 $ \mathcal{D}_{\text{KL}}(q(\boldsymbol{x}_{t-1}\mid\boldsymbol{x}_t, \boldsymbol{x}_0) \mid\mid p_{\boldsymbol{\theta}}(\boldsymbol{x}_{t-1}\mid\boldsymbol{x}_t))$ 对于任意复杂的马尔可夫HVAEs中的任意后验来说都是难以最小化的，因为同时学习编码器的复杂性增加了，但在变分扩散模型（VDM）中，我们可以利用高斯转换假设来使优化变得可行。通过贝叶斯规则，我们有：
$$
q(\boldsymbol{x}_{t-1}\mid\boldsymbol{x}_t, \boldsymbol{x}_0) = \frac{q(\boldsymbol{x}_t \mid \boldsymbol{x}_{t-1}, \boldsymbol{x}_0)q(\boldsymbol{x}_{t-1}\mid\boldsymbol{x}_0)}{q(\boldsymbol{x}_{t}\mid\boldsymbol{x}_0)}
$$



由于我们已经知道根据我们对编码器转换的假设（方程31）有： $ q(x_t|x_{t-1}, x_0) = q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{\alpha_t} x_{t-1}, (1 - \alpha_t)I) $ ，剩下的的工作是推导出 $ q(x_t|x_0) $ 和 $ q(x_{t-1}|x_0) $ 的形式。幸运的是，通过利用变分扩散模型的编码器转换是线性高斯模型这一事实， 这些也可以变得可行。回想一下，在重参数化技巧下，样本 $ x_t \sim q(x_t|x_{t-1}) $ 可以被重写为：
$$
\begin{align}
    \boldsymbol{x}_t = \sqrt{\alpha_t}\boldsymbol{x}_{t-1} + \sqrt{1 - \alpha_t}\boldsymbol{\epsilon} \quad \text{with } \boldsymbol{\epsilon} \sim \mathcal{N}(\boldsymbol{\epsilon}; \boldsymbol{0}, \textbf{I})
\end{align}
$$


类似地，样本 $ x_{t-1} \sim q(x_{t-1}|x_{t-2}) $ 可以被重写为：
$$
\begin{align}
    \boldsymbol{x}_{t-1} = \sqrt{\alpha_{t-1}}\boldsymbol{x}_{t-2} + \sqrt{1 - \alpha_{t-1}}\boldsymbol{\epsilon} \quad \text{with } \boldsymbol{\epsilon} \sim \mathcal{N}(\boldsymbol{\epsilon}; \boldsymbol{0}, \textbf{I})
\end{align}
$$
然后，$ q(x_t|x_0) $ 的形式可以通过重复应用重新参数化技巧来递归推导出来。假设我们有 $ 2T $ 个随机噪声变量 $\{\boldsymbol{\epsilon}_t^*,\boldsymbol{\epsilon}_t\}_{t=0}^T \stackrel{\text{iid}}{\sim} \mathcal{N}(\boldsymbol{\epsilon}; \boldsymbol{0},\textbf{I})$。那么，对于任意样本 $ x_t \sim q(x_t|x_0) $，我们可以重写为：
$$
\begin{align}
\boldsymbol{x}_t  &= \sqrt{\alpha_t}\boldsymbol{x}_{t-1} + \sqrt{1 - \alpha_t}\boldsymbol{\epsilon}_{t-1}^*\\
&= \sqrt{\alpha_t}\left(\sqrt{\alpha_{t-1}}\boldsymbol{x}_{t-2} + \sqrt{1 - \alpha_{t-1}}\boldsymbol{\epsilon}_{t-2}^*\right) + \sqrt{1 - \alpha_t}\boldsymbol{\epsilon}_{t-1}^*\\
&= \sqrt{\alpha_t\alpha_{t-1}}\boldsymbol{x}_{t-2} + \sqrt{\alpha_t - \alpha_t\alpha_{t-1}}\boldsymbol{\epsilon}_{t-2}^* + \sqrt{1 - \alpha_t}\boldsymbol{\epsilon}_{t-1}^*\\
&= \sqrt{\alpha_t\alpha_{t-1}}\boldsymbol{x}_{t-2} + \sqrt{\sqrt{\alpha_t - \alpha_t\alpha_{t-1}}^2 + \sqrt{1 - \alpha_t}^2}\boldsymbol{\epsilon}_{t-2} \\
&= \sqrt{\alpha_t\alpha_{t-1}}\boldsymbol{x}_{t-2} + \sqrt{\alpha_t - \alpha_t\alpha_{t-1} + 1 - \alpha_t}\boldsymbol{\epsilon}_{t-2}\\
&= \sqrt{\alpha_t\alpha_{t-1}}\boldsymbol{x}_{t-2} + \sqrt{1 - \alpha_t\alpha_{t-1}}\boldsymbol{\epsilon}_{t-2} \\
&= \ldots\\
&= \sqrt{\prod_{i=1}^t\alpha_i}\boldsymbol{x}_0 + \sqrt{1 - \prod_{i=1}^t\alpha_i}\boldsymbol{\boldsymbol{\epsilon}}_0\\
&= \sqrt{\bar\alpha_t}\boldsymbol{x}_0 + \sqrt{1 - \bar\alpha_t}\boldsymbol{\boldsymbol{\epsilon}}_0 \\
&\sim \mathcal{N}(\boldsymbol{x}_{t} ; \sqrt{\bar\alpha_t}\boldsymbol{x}_0, \left(1 - \bar\alpha_t\right)\textbf{I})
\end{align}
$$


其中，在方程44中，我们利用了两个独立的高斯随机变量之和仍然是高斯分布的事实，其均值为两个均值之和，方差为两个方差之和。将 $ \sqrt{1 - \alpha_t}\boldsymbol{\epsilon}_{t-1}^*$ 解释为从高斯 $ \mathcal{N}(0, (1 - \alpha_t)I) $ 中抽取的样本，以及 $\sqrt{\alpha_t - \alpha_t\alpha_{t-1}}\boldsymbol{\epsilon}_{t-2}^*$ 解释为从高斯 $ \mathcal{N}(0, (\alpha_t - \alpha_t \alpha_{t-1})I) $ 中抽取的样本，我们可以将它们的和视为从高斯 $ \mathcal{N}(0, (1 - \alpha_t \alpha_{t-1})I) $ 中抽取的随机变量 $\mathcal{N}(\boldsymbol{0}, (1 - \alpha_t + \alpha_t - \alpha_t\alpha_{t-1})\textbf{I}) = \mathcal{N}(\boldsymbol{0}, (1 - \alpha_t\alpha_{t-1})\textbf{I})$。这个分布的样本可以使用重参数化技巧表示为 $\sqrt{1 - \alpha_t\alpha_{t-1}}\boldsymbol{\epsilon}_{t-2}$，如方程46所示。

因此，我们已经推导出了 $ q(x_t|x_0) $ 的高斯形式。这个推导可以修改得到描述 $ q(x_{t-1}|x_0) $ 的高斯参数化。现在，知道了 $ q(x_t|x_0) $ 和 $ q(x_{t-1}|x_0) $ 的形式，我们可以通过将它们代入贝叶斯规则扩展来计算 $ q(x_{t-1}|x_t, x_0) $ 的形式：

$$
\begin{align}
q(\boldsymbol{x}_{t-1}\mid\boldsymbol{x}_t, \boldsymbol{x}_0)
&= \frac{q(\boldsymbol{x}_t \mid \boldsymbol{x}_{t-1}, \boldsymbol{x}_0)q(\boldsymbol{x}_{t-1}\mid\boldsymbol{x}_0)}{q(\boldsymbol{x}_{t}\mid\boldsymbol{x}_0)}\\
&= \frac{\mathcal{N}(\boldsymbol{x}_{t} ; \sqrt{\alpha_t} \boldsymbol{x}_{t-1}, (1 - \alpha_t)\textbf{I})\mathcal{N}(\boldsymbol{x}_{t-1} ; \sqrt{\bar\alpha_{t-1}}\boldsymbol{x}_0, (1 - \bar\alpha_{t-1}) \textbf{I})}{\mathcal{N}(\boldsymbol{x}_{t} ; \sqrt{\bar\alpha_{t}}\boldsymbol{x}_0, (1 - \bar\alpha_{t})\textbf{I})}\\
&\propto \mathcal{N}(\boldsymbol{x}_{t-1} ; \underbrace{\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})\boldsymbol{x}_{t} + \sqrt{\bar\alpha_{t-1}}(1-\alpha_t)\boldsymbol{x}_0}{1 -\bar\alpha_{t}}}_{\mu_q(\boldsymbol{x}_t, \boldsymbol{x}_0)}, \underbrace{\frac{(1 - \alpha_t)(1 - \bar\alpha_{t-1})}{1 -\bar\alpha_{t}}\textbf{I}}_{\boldsymbol{\Sigma}_q(t)})
\end{align}
$$

证明过程如下：

<div align=center>
<img src="https://calvinyluo.com/assets/images/diffusion/proofs/noise_distribution_deriv.svg" alt="推导每个中间噪声潜伏的分布" style="zoom:100%;" />
</div>


其中在第5行中，$ C(x_t, x_0) $ 是一个关于 $ x_{t-1} $ 的常数项，它是 $ x_t, x_0 $ 和 $ \alpha $ 值的组合；该项在最后一行中中被隐式地返回返回。

因此，我们已经展示了在每一步，$ x_{t-1} \sim q(x_{t-1}|x_t, x_0) $ 都是正态分布的，其均值 $ \mu_{q}(x_t, x_0) $ 是 $x_t $ 和 $ x_0 $ 的函数，方差 $ \Sigma_{q}(t) $ 是 $ \alpha $ 系数的函数。这些 $ \alpha $ 系数在每个时间步是已知并固定的；它们要么被永久地设置为超参数，要么被视为一个网络的当前推理输出，该网络旨在对它们进行建模。根据方程53，我们可以将我们的方差方程重写为 $ \Sigma_{q}(t) = \sigma^2_{q}(t) I $，其中：

$$
\begin{align}
    \sigma_q^2(t) = \frac{(1 - \alpha_t)(1 - \bar\alpha_{t-1})}{1 -\bar\alpha_{t}}
\end{align}
$$


为了尽可能地将近似去噪转换步骤 $ p_{\theta}(x_{t-1}|x_t) $ 与真实去噪转换步骤 $ q(x_{t-1}|x_t, x_0) $ 相匹配，我们也可以将其建模为高斯分布。此外，由于所有的 $ \alpha $ 项在每个时间步都是已知并固定的，我们可以立即构造近似去噪转换步骤的方差为 $ \Sigma_{q}(t) = \sigma^2_{q}(t) I $。我们必须将均值 $ \mu_{\theta}(x_t, t) $ 参数化为 $x_t $ 的函数，因为 $ p_{\theta}(x_{t-1}|x_t) $ 不依赖于 $ x_0 $。

回想一下，两个高斯分布之间的KL散度是：

$$
\begin{align}
\mathcal{D}_{\text{KL}}(\mathcal{N}(\boldsymbol{x}; \boldsymbol{\mu}_x,\boldsymbol{\Sigma}_x) \mid\mid \mathcal{N}(\boldsymbol{y}; \boldsymbol{\mu}_y,\boldsymbol{\Sigma}_y))
&=\frac{1}{2}\left[\log\frac{\mid\boldsymbol{\Sigma}_y\mid}{\mid\boldsymbol{\Sigma}_x\mid} - d + \text{tr}(\boldsymbol{\Sigma}_y^{-1}\boldsymbol{\Sigma}_x) + (\boldsymbol{\mu}_y-\boldsymbol{\mu}_x)^T \boldsymbol{\Sigma}_y^{-1} (\boldsymbol{\mu}_y-\boldsymbol{\mu}_x)\right]
\end{align}
$$


在我们的例子中，我们可以将两个高斯的方差设置为完全匹配，优化KL散度项简化为最小化两个分布的均值之间的差异：
$$
\begin{align}
& \quad \,\underset{\boldsymbol{\theta}}{\arg\min}\,  \mathcal{D}_{\text{KL}}(q(\boldsymbol{x}_{t-1}\mid\boldsymbol{x}_t, \boldsymbol{x}_0) \mid\mid p_{\boldsymbol{\theta}}(\boldsymbol{x}_{t-1}\mid\boldsymbol{x}_t)) \nonumber \\
&= \underset{\boldsymbol{\theta}}{\arg\min}\, \mathcal{D}_{\text{KL}}\left(\mathcal{N}\left(\boldsymbol{x}_{t-1}; \boldsymbol{\mu}_q,\boldsymbol{\Sigma}_q\left(t\right)\right) \mid\mid \mathcal{N}\left(\boldsymbol{x}_{t-1}; \boldsymbol{\mu}_{\boldsymbol{\theta}},\boldsymbol{\Sigma}_q\left(t\right)\right)\right)\\
&=\underset{\boldsymbol{\theta}}{\arg\min}\, \frac{1}{2\sigma_q^2(t)}\left[\left\lVert\boldsymbol{\mu}_{\boldsymbol{\theta}}-\boldsymbol{\mu}_q\right\rVert_2^2\right]
\end{align}
$$

证明过程如下：


<div align=center>
<img src="https://calvinyluo.com/assets/images/diffusion/proofs/kl_gaussian_deriv.svg" alt="推导两个高斯函数之间的 KL" style="zoom:100%;" />
</div>


其中，我们已经将  $ \mu_{q}(x_t, x_0) $简写为 $ \mu_{q} $ ，将  $ \mu_{\theta}(x_t, t) $  简写为 $ \mu_{\theta} $，以便于书写。换句话说，我们想要优化一个 $ \mu_{\theta}(x_t, t) $ 来匹配 $ \mu_{q}(x_t, x_0) $，根据我们推导出的公式 53，它的形式是：
$$
\begin{align}
    \boldsymbol{\mu}_q(\boldsymbol{x}_t, \boldsymbol{x}_0) = \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})\boldsymbol{x}_{t} + \sqrt{\bar\alpha_{t-1}}(1-\alpha_t)\boldsymbol{x}_0}{1 -\bar\alpha_{t}}
\end{align}
$$


由于 $ \mu_{\theta}(x_t, t) $ 也依赖于 $x_t $，我们可以通过将其设置为以下形式来密切匹配 $ \mu_{q}(x_t, x_0) $：

$$
\begin{align}
    \boldsymbol{\mu}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t) = \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})\boldsymbol{x}_{t} + \sqrt{\bar\alpha_{t-1}}(1-\alpha_t)\hat{\boldsymbol{x}}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)}{1 -\bar\alpha_{t}}
\end{align}
$$


其中 $ \hat{x}_{\theta}(x_t, t) $ 是一个神经网络，它的目标是从噪声图像 $x_t $ 和时间索引 $ t $ 预测原始图像 $ x_0 $。然后，优化问题简化为：
$$
\begin{align}
& \quad \,\underset{\boldsymbol{\theta}}{\arg\min}\,  \mathcal{D}_{\text{KL}}(q(\boldsymbol{x}_{t-1}\mid\boldsymbol{x}_t, \boldsymbol{x}_0) \mid\mid p_{\boldsymbol{\theta}}(\boldsymbol{x}_{t-1}\mid\boldsymbol{x}_t)) \nonumber \\
&= \underset{\boldsymbol{\theta}}{\arg\min}\, \mathcal{D}_{\text{KL}}\left(\mathcal{N}\left(\boldsymbol{x}_{t-1}; \boldsymbol{\mu}_q,\boldsymbol{\Sigma}_q\left(t\right)\right) \mid\mid \mathcal{N}\left(\boldsymbol{x}_{t-1}; \boldsymbol{\mu}_{\boldsymbol{\theta}},\boldsymbol{\Sigma}_q\left(t\right)\right)\right)\\
&=\underset{\boldsymbol{\theta}}{\arg\min}\, \frac{1}{2\sigma_q^2(t)}\frac{\bar\alpha_{t-1}(1-\alpha_t)^2}{(1 -\bar\alpha_{t})^2}\left[\left\lVert\hat{\boldsymbol{x}}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t) - \boldsymbol{x}_0\right\rVert_2^2\right]
\end{align}
$$

证明过程如下：

<div align=center>
<img src="https://calvinyluo.com/assets/images/diffusion/proofs/image_objective.svg" alt="推导图像预测目标" style="zoom:100%;" />
</div>

因此，优化一个变分扩散模型归结为学习一个神经网络，该神经网络从任意噪声版本来预测原始图像 $ x_0 $。此外，通过最小化我们推导出的ELBO目标中的所有噪声水平的求和项（方程38），可以通过最小化所有时间步骤的期望来近似：
$$
\begin{align}
& \quad \,\underset{\boldsymbol{\theta}}{\arg\min}\, \sum_{t=2}^{T} \mathbb{E}_{q(\boldsymbol{x}_{t}\mid\boldsymbol{x}_0)}\left[\mathcal{D}_{\text{KL}}(q(\boldsymbol{x}_{t-1}\mid\boldsymbol{x}_t, \boldsymbol{x}_0) \mid\mid p_{\boldsymbol{\theta}}(\boldsymbol{x}_{t-1}\mid\boldsymbol{x}_t))\right] \nonumber \\
&= \underset{\boldsymbol{\theta}}{\arg\min}\, \mathbb{E}_{t\sim U\{2, T\}}\left[\mathbb{E}_{q(\boldsymbol{x}_{t}\mid\boldsymbol{x}_0)}\left[ \frac{1}{2\sigma_q^2(t)}\frac{\bar\alpha_{t-1}(1-\alpha_t)^2}{(1 -\bar\alpha_{t})^2}\left[\left\lVert\hat{\boldsymbol{x}}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t) - \boldsymbol{x}_0\right\rVert_2^2\right] \right]\right]
\end{align}
$$

然后可以使用随时间步长的随机样本进行优化。

### 学习扩散噪声参数

让我们探讨如何联合学习变分扩散模型（VDM）的噪声参数。一种潜在的方法是使用带有参数 $ \boldsymbol{\eta} $ 的神经网络 $ \hat{\alpha}_{\boldsymbol{\eta}}(t) $ 来对 $ \alpha_t $ 进行建模。然而，这种方法效率低下，因为在每个时间步 $ t $ 都必须多次执行推断以计算 $ \bar{\alpha}_t $。尽管通过缓存可以缓解这种计算成本，我们还可以推导出另一种学习扩散噪声参数的方法。通过将我们的方差方程（方程54）代入我们推导出的每个时间步的目标方程 （方程61），我们可以简化：


$$
\begin{align}
\frac{1}{2\sigma_q^2(t)}\frac{\bar\alpha_{t-1}(1-\alpha_t)^2}{(1 -\bar\alpha_{t})^2}\left[\left\lVert\hat{\boldsymbol{x}}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t) - \boldsymbol{x}_0\right\rVert_2^2\right] = \frac{1}{2}\left(\frac{\bar\alpha_{t-1}}{1 - \bar\alpha_{t-1}} -\frac{\bar\alpha_t}{1 -\bar\alpha_{t}}\right)\left[\left\lVert\hat{\boldsymbol{x}}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t) - \boldsymbol{x}_0\right\rVert_2^2\right]
\end{align}
$$

证明过程如下：

<div align=center>
<img src="https://calvinyluo.com/assets/images/diffusion/proofs/snr_deriv.svg" alt="使用 SNR 项导出目标" style="zoom:100%;" />
</div>

回忆一下方程50 ， $q(\boldsymbol{x}_t\mid\boldsymbol{x}_0)$  是一个高斯分布， 形式为 $\mathcal{N}(\boldsymbol{x}_{t} ; \sqrt{\bar\alpha_t}\boldsymbol{x}_0, \left(1 - \bar\alpha_t\right)\textbf{I})$。然后，遵循[信噪比 (SNR)](https：//en.wikipedia.org/wiki/Signal-to-noise_ratio#Alternate_definition)的定义：$\frac{\mu^2}{\sigma^2}$，我们可以将每个时间步 t 的信噪比写为：
$$
\begin{align}
    \text{SNR}(t) &= \frac{\bar\alpha_t}{1 -\bar\alpha_{t}}
\end{align}
$$
然后，我们推导出的方程63（和方程61) 可以简化为：
$$
\begin{align}
\frac{1}{2\sigma_q^2(t)}\frac{\bar\alpha_{t-1}(1-\alpha_t)^2}{(1 -\bar\alpha_{t})^2}\left[\left\lVert\hat{\boldsymbol{x}}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t) - \boldsymbol{x}_0\right\rVert_2^2\right] &= \frac{1}{2}\left(\text{SNR}(t-1) -\text{SNR}(t)\right)\left[\left\lVert\hat{\boldsymbol{x}}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t) - \boldsymbol{x}_0\right\rVert_2^2\right]
\end{align}
$$

顾名思义，信噪比（SNR）代表了原始信号与存在的噪声量之间的比率；较高的SNR表示更多的信号，较低的SNR表示更多的噪声。在扩散模型中，我们要求 SNR 随着时间步 $ t $ 的增加而单调递减；这形式化了扰动输入 $ x_t $ 随着时间的推移变得越来越嘈杂的概念，直到在 $ t = T $ 时完全变成标准高斯噪声。

根据方程(65)中目标的简化，我们可以直接使用神经网络对每个时间步的SNR进行参数化，并与扩散模型一起联合学习[10]。由于SNR必须随时间单调递减，我们可以将其表示为：

$$
\begin{align}
    \text{SNR}(t) = \text{exp}(-\omega_{\boldsymbol{\eta}}(t))
\end{align}
$$
其中 $ w_n(t) $ 被建模为具有参数 $ n $ 的单调递增神经网络。对 $ w_n(t) $ 取反得到一个单调递减的函数，而指数函数则迫使结果项为正。注意，方程(62)中的目标现在也必须对 $ n $ 进行优化：

$$
\begin{align}
& \quad \,\underset{\boldsymbol{\theta}, \,\boldsymbol{\eta}}{\arg\min}\, \sum_{t=2}^{T} \mathbb{E}_{q(\boldsymbol{x}_{t}\mid\boldsymbol{x}_0)}\left[\mathcal{D}_{\text{KL}}(q(\boldsymbol{x}_{t-1}\mid\boldsymbol{x}_t, \boldsymbol{x}_0) \mid\mid p_{\boldsymbol{\theta}}(\boldsymbol{x}_{t-1}\mid\boldsymbol{x}_t))\right] \nonumber \\
&= \underset{\boldsymbol{\theta}, \,\boldsymbol{\eta}}{\arg\min}\, \mathbb{E}_{t\sim U\{2, T\}}\left[\mathbb{E}_{q(\boldsymbol{x}_{t}\mid\boldsymbol{x}_0)}\left[ \frac{1}{2}\left(\text{SNR}(t-1) -\text{SNR}(t)\right)\left[\left\lVert\hat{\boldsymbol{x}}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t) - \boldsymbol{x}_0\right\rVert_2^2\right] \right]\right]
\end{align}
$$


通过将方程(66)中的SNR参数化与方程(64)中的SNR定义结合起来，我们还可以显式地推导出 $ \alpha_t $ 以及 $ 1 - \alpha_t $ 的优雅形式：
$$
\begin{align}
    &\frac{\bar\alpha_t}{1 -\bar\alpha_{t}} = \text{exp}(-\omega_{\boldsymbol{\eta}}(t))\\
    &\therefore \bar\alpha_t = \text{sigmoid}(-\omega_{\boldsymbol{\eta}}(t))\\
    &\therefore 1 - \bar\alpha_t = \text{sigmoid}(\omega_{\boldsymbol{\eta}}(t))
\end{align}
$$


这些项对于各种计算是必需的；例如，在优化过程中，它们被用来使用重新参数化技巧从输入 $ x_0 $ 创建任意噪声的 $ x_t $，如在方程49中推导的。

### 三种等效的解释 (Three Equivalent Interpretations)

#### 第二种常见解释

正如我们之前证明的，可以通过简单地学习一个神经网络来预测原始自然图像 $ x_0 $ , 以任意噪声版本 $ x_t $ 及其时间索引 $t$ 作为输入来训练变分扩散模型。然而，$ x_0 $有两个其他的等效参数化形式，这导致了变分扩散模型的另外两种解释。

首先，可以利用重参数化技巧。在推导 $ q(x_t|x_0) $ 的形式时，我们可以重新排列方程(49)：

$$
\begin{align}
\boldsymbol{x}_0 &= \frac{\boldsymbol{x}_t - \sqrt{1 - \bar\alpha_t}\boldsymbol{\epsilon}_0}{\sqrt{\bar\alpha_t}}
\end{align}
$$


将这个代入我们之前推导出的真实去噪转换均值 $ \mu_{q}(x_t, x_0) $，我们可以重新推导如下：

$$
\begin{align}
\boldsymbol{\mu}_q(\boldsymbol{x}_t, \boldsymbol{x}_0) &= \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})\boldsymbol{x}_{t} + \sqrt{\bar\alpha_{t-1}}(1-\alpha_t)\boldsymbol{x}_0}{1 -\bar\alpha_{t}}\\
&= \frac{1}{\sqrt{\alpha_t}}\boldsymbol{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar\alpha_t}\sqrt{\alpha_t}}\boldsymbol{\epsilon}_0
\end{align}
$$


因此，我们可以将近似去噪转换均值 $ \mu_{\theta}(x_t, t) $设置为：
$$
\begin{align}
\boldsymbol{\mu}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t) &= \frac{1}{\sqrt{\alpha_t}}\boldsymbol{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar\alpha_t}\sqrt{\alpha_t}}\boldsymbol{\hat{\epsilon}}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)
\end{align}
$$


相应的优化问题变为：

$$
\begin{align}
& \quad \,\underset{\boldsymbol{\theta}}{\arg\min}\,  \mathcal{D}_{\text{KL}}(q(\boldsymbol{x}_{t-1}\mid\boldsymbol{x}_t, \boldsymbol{x}_0) \mid\mid p_{\boldsymbol{\theta}}(\boldsymbol{x}_{t-1}\mid\boldsymbol{x}_t)) \nonumber \\
&= \underset{\boldsymbol{\theta}}{\arg\min}\, \mathcal{D}_{\text{KL}}\left(\mathcal{N}\left(\boldsymbol{x}_{t-1}; \boldsymbol{\mu}_q,\boldsymbol{\Sigma}_q\left(t\right)\right) \mid\mid \mathcal{N}\left(\boldsymbol{x}_{t-1}; \boldsymbol{\mu}_{\boldsymbol{\theta}},\boldsymbol{\Sigma}_q\left(t\right)\right)\right)\\
&=\underset{\boldsymbol{\theta}}{\arg\min}\, \frac{1}{2\sigma_q^2(t)}\frac{(1 - \alpha_t)^2}{(1 - \bar\alpha_t)\alpha_t}\left[\left\lVert\boldsymbol{\epsilon}_0 - \boldsymbol{\hat{\epsilon}}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)\right\rVert_2^2\right]
\end{align}
$$

其中，$ \hat{\epsilon}_{\theta}(x_t, t) $是一个神经网络，它学习预测决定源噪声 $ \epsilon_0 \sim \mathcal{N}(\epsilon; 0, I) $，这个噪声决定了从  $x_0 $ 到 $ x_t $的转换。因此，我们已经展示了通过学习变分扩散模型预测原始图像 $ x_0 $是和学习预测噪声是等价的；然而，实证上，一些研究工作发现**预测噪声可以带来更好的性能**。

####  第三种常见解释

为了推导出变分扩散模型的第三种常见解释，我们引用Tweedie方程。Tweedie方程指出，给定从指数族分布中抽取的样本，其真实均值可以通过样本的最大似然估计（即经验均值）加上涉及估计分数的一些校正项来估计。在只有一个观测样本的情况下，经验均值就是样本本身。它通常用于减少样本偏差；如果观测样本都位于潜在分布的一端，那么负得分会变得很大，并将原始的最大似然估计的样本向真实均值进行修正。

数学上，对于一个高斯变量 $ z \sim \mathcal{N}(z; \mu_z, \Sigma_z) $，Tweedie方程表明：$\mathbb{E}\left[\boldsymbol{\mu}_z\mid\boldsymbol{z}\right] = \boldsymbol{z} + \boldsymbol{\Sigma}_z\nabla_{\boldsymbol{z}} \log p(\boldsymbol{z})$

在这种情况下，我们将其应用于预测给定其样本的 $x_t $的真实后验均值。从方程50，我们知道 $ q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\overline{\alpha}_t} x_0, (1 - \overline{\alpha}_t) I) $。然后，根据Tweedie方程，我们有：

$$
\begin{align}
\mathbb{E}\left[\boldsymbol{\mu}_{x_t}\mid\boldsymbol{x}_t\right] = \boldsymbol{x}_t + (1 - \bar\alpha_t)\nabla_{\boldsymbol{x}_t}\log p(\boldsymbol{x}_t)
\end{align}
$$
其中我们为了简化符号，将 $ \nabla_{x_t} \log p(x_t) $写为 $ \nabla \log p(x_t) $。根据Tweedie方程，$x_t $生成的真实均值 $ \mu_{x_t} = \sqrt{\overline{\alpha}_t} x_0 $被定义为：

$$
\begin{align}
    \sqrt{\bar\alpha_t}\boldsymbol{x}_0 = \boldsymbol{x}_t + (1 - \bar\alpha_t)\nabla\log p(\boldsymbol{x}_t)\\
    \therefore \boldsymbol{x}_0 = \frac{\boldsymbol{x}_t + (1 - \bar\alpha_t)\nabla\log p(\boldsymbol{x}_t)}{\sqrt{\bar\alpha_t}}
\end{align}
$$


然后，我们可以将方程133再次代入我们的真实去噪转换均值 $ \mu_{q}(x_t, x_0) $并推导出一个新的形式：
$$
\begin{align}
\boldsymbol{\mu}_q(\boldsymbol{x}_t, \boldsymbol{x}_0) &= \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})\boldsymbol{x}_{t} + \sqrt{\bar\alpha_{t-1}}(1-\alpha_t)\boldsymbol{x}_0}{1 -\bar\alpha_{t}}\\
&= \frac{1}{\sqrt{\alpha_t}}\boldsymbol{x}_t + \frac{1 - \alpha_t}{\sqrt{\alpha_t}}\nabla\log p(\boldsymbol{x}_t)
\end{align}
$$

证明过程如下：


<div align=center>
<img src="https://calvinyluo.com/assets/images/diffusion/proofs/pred_score_deriv.svg" alt="使用分数项导出真实的去噪平均值" style="zoom:100%;" />
</div>


因此，我们也可以将近似去噪转换均值 $ \mu_{\theta}(x_t, t) $设置为：
$$
\begin{align}
\boldsymbol{\mu}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t) &= \frac{1}{\sqrt{\alpha_t}}\boldsymbol{x}_t + \frac{1 - \alpha_t}{\sqrt{\alpha_t}}\boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)
\end{align}
$$


相应的优化问题变为：

$$
\begin{align}
& \quad \,\underset{\boldsymbol{\theta}}{\arg\min}\,  \mathcal{D}_{\text{KL}}(q(\boldsymbol{x}_{t-1}\mid\boldsymbol{x}_t, \boldsymbol{x}_0) \mid\mid p_{\boldsymbol{\theta}}(\boldsymbol{x}_{t-1}\mid\boldsymbol{x}_t)) \nonumber \\
&= \underset{\boldsymbol{\theta}}{\arg\min}\, \mathcal{D}_{\text{KL}}\left(\mathcal{N}\left(\boldsymbol{x}_{t-1}; \boldsymbol{\mu}_q,\boldsymbol{\Sigma}_q\left(t\right)\right) \mid\mid \mathcal{N}\left(\boldsymbol{x}_{t-1}; \boldsymbol{\mu}_{\boldsymbol{\theta}},\boldsymbol{\Sigma}_q\left(t\right)\right)\right)\\
&=\underset{\boldsymbol{\theta}}{\arg\min}\, \frac{1}{2\sigma_q^2(t)}\frac{(1 - \alpha_t)^2}{\alpha_t}\left[\left\lVert \boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t) - \nabla\log p(\boldsymbol{x}_t)\right\rVert_2^2\right]
\end{align}
$$



其中，$\boldsymbol{\hat{ s }}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t) $ 是一个神经网络，它学习预测源噪声 $ \epsilon_0 \sim \mathcal{N}(e; 0, I) $，该噪声决定了 从 $ x_0 $ 到 $ \hat{x}_t $ 的噪声水平。因此，我们已经证明了通过预测原始图像 $ x_0 $ 来学习 变分扩散模型 是等同于学习预测噪声；然而，实证上，一些工作发现预测噪声能够带来更好的性能[2, 4]。

为了推导出变分扩散模型的第三种常见解释，我们引用了Tweedie方程[11]。Tweedie方程指出，给定从其中抽取的样本，指数族分布的真实均值可以通过样本的最大似然估计（也就是经验均值）加上涉及估计分数的一些校正项来估计。在只有一个观测样本的情况下，经验均值就是样本本身。它通常用于减少样本偏差；如果观测样本都位于底层分布的一端，那么负分数变得较大，并校正样本的朴素最大似然估计，使其接近真实均值。

数学上，对于一个高斯变量 $ z \sim \mathcal{N}(z; \mu_z, \Sigma_z) $，Tweedie方程表明：$\mathbb{E}\left[\boldsymbol{\mu}_z\mid\boldsymbol{z}\right] = \boldsymbol{z} + \boldsymbol{\Sigma}_z\nabla_{\boldsymbol{z}} \log p(\boldsymbol{z})$

在这种情况下，我们将其应用于给定其样本的情况下预测 $ \alpha_t $ 的真实后验均值。从方程50，我们知道：$q(\boldsymbol{x}_t\mid\boldsymbol{x}_0) = \mathcal{N}(\boldsymbol{x}_{t} ; \sqrt{\bar\alpha_t}\boldsymbol{x}_0, \left(1 - \bar\alpha_t\right)\textbf{I})$

然后，根据Tweedie方程，我们有：
$$
\begin{align}
\mathbb{E}\left[\boldsymbol{\mu}_{x_t}\mid\boldsymbol{x}_t\right] = \boldsymbol{x}_t + (1 - \bar\alpha_t)\nabla_{\boldsymbol{x}_t}\log p(\boldsymbol{x}_t)
\end{align}
$$


其中我们为了简化符号，将 $ \nabla_{\alpha_t} \log p(\alpha_t) $ 写为 $ \nabla \log p(\alpha_t) $。根据Tweedie方程，$ \alpha_t $ 生成的真实均值 $ \mu_{\alpha_t} = \sqrt{\alpha_t} x_0 $ 被定义为：

$$
\begin{align}
    \sqrt{\bar\alpha_t}\boldsymbol{x}_0 = \boldsymbol{x}_t + (1 - \bar\alpha_t)\nabla\log p(\boldsymbol{x}_t)\\
    \therefore \boldsymbol{x}_0 = \frac{\boldsymbol{x}_t + (1 - \bar\alpha_t)\nabla\log p(\boldsymbol{x}_t)}{\sqrt{\bar\alpha_t}}
\end{align}
$$


然后，我们可以将方程79再次代入我们之前推导出的真实去噪转换均值 $ \mu_{q}(\alpha_t, x_0) $ 并推导出一个新的参数化方程：
$$
\begin{align}
\boldsymbol{\mu}_q(\boldsymbol{x}_t, \boldsymbol{x}_0) &= \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})\boldsymbol{x}_{t} + \sqrt{\bar\alpha_{t-1}}(1-\alpha_t)\boldsymbol{x}_0}{1 -\bar\alpha_{t}}\\
&= \frac{1}{\sqrt{\alpha_t}}\boldsymbol{x}_t + \frac{1 - \alpha_t}{\sqrt{\alpha_t}}\nabla\log p(\boldsymbol{x}_t)
\end{align}
$$
证明过程如下：

<div align=center>
<img src="https://calvinyluo.com/assets/images/diffusion/proofs/pred_score_deriv.svg" alt="使用分数项导出真实的去噪平均值" style="zoom:100%;" />
</div>

因此，我们也可以将我们的近似去噪转换均值 $ \mu_{\theta}(\alpha_t, t) $ 设置为：

$$
\begin{align}
\boldsymbol{\mu}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t) &= \frac{1}{\sqrt{\alpha_t}}\boldsymbol{x}_t + \frac{1 - \alpha_t}{\sqrt{\alpha_t}}\boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t)
\end{align}
$$


相应的优化问题变为：
$$
\begin{align}
& \quad \,\underset{\boldsymbol{\theta}}{\arg\min}\,  \mathcal{D}_{\text{KL}}(q(\boldsymbol{x}_{t-1}\mid\boldsymbol{x}_t, \boldsymbol{x}_0) \mid\mid p_{\boldsymbol{\theta}}(\boldsymbol{x}_{t-1}\mid\boldsymbol{x}_t)) \nonumber \\
&= \underset{\boldsymbol{\theta}}{\arg\min}\, \mathcal{D}_{\text{KL}}\left(\mathcal{N}\left(\boldsymbol{x}_{t-1}; \boldsymbol{\mu}_q,\boldsymbol{\Sigma}_q\left(t\right)\right) \mid\mid \mathcal{N}\left(\boldsymbol{x}_{t-1}; \boldsymbol{\mu}_{\boldsymbol{\theta}},\boldsymbol{\Sigma}_q\left(t\right)\right)\right)\\
&=\underset{\boldsymbol{\theta}}{\arg\min}\, \frac{1}{2\sigma_q^2(t)}\frac{(1 - \alpha_t)^2}{\alpha_t}\left[\left\lVert \boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}_t, t) - \nabla\log p(\boldsymbol{x}_t)\right\rVert_2^2\right]
\end{align}
$$


其中，$ s_{\theta}(\alpha_t, t) $ 是一个神经网络，它学习预测任意噪声水平 $ t $ 下 $ \alpha_t $ 的得分函数 $ \nabla_{\alpha_t} \log p(\alpha_t) $，这是 $ \alpha_t $ 在数据空间中的梯度。

敏锐的读者会观察到得分函数 $ \nabla \log p(x_t) $ 在形式上与源噪声 $ \epsilon_0 $ 非常相似。这可以通过将Tweedie方程（方程79）与重新参数化技巧（方程71）结合起来明确展示：

$$
\begin{align}
\boldsymbol{x}_0 = \frac{\boldsymbol{x}_t + (1 - \bar\alpha_t)\nabla\log p(\boldsymbol{x}_t)}{\sqrt{\bar\alpha_t}} &= \frac{\boldsymbol{x}_t - \sqrt{1 - \bar\alpha_t}\boldsymbol{\epsilon}_0}{\sqrt{\bar\alpha_t}}\\
\therefore (1 - \bar\alpha_t)\nabla\log p(\boldsymbol{x}_t) &= -\sqrt{1 - \bar\alpha_t}\boldsymbol{\epsilon}_0\\
\nabla\log p(\boldsymbol{x}_t) &= -\frac{1}{\sqrt{1 - \bar\alpha_t}}\boldsymbol{\epsilon}_0
\end{align}
$$


事实证明，这两项之间存在一个随时间变化的常数因子！得分函数测量了在数据空间中如何移动以最大化对数概率；直观上，由于源噪声被添加到自然图像中以破坏它，向相反方向移动“去噪”图像，并且将是增加随后对数概率的最佳更新。我们的数学证明只是证实了这种直觉；我们已经明确展示了学习建模得分函数等同于建模源噪声的负值（乘以一个随时间变化的常数因子）。

因此，我们已经推导出了优化变分扩散模型的三种等效目标， 学习一个神经网络来：

- 预测任意噪声版本的原始图像 $ x_0 $；
- 预测任意噪声图像的源噪声 $ \epsilon_0 $；
- 预测任意噪声水平下的图像得分 $ \nabla \log p(x_t) $。

变分扩散模型可以通过随机采样时间步 $ t $ 并最小化预测与真实目标的范数来可扩展地训练。

## 分数生成模型（Score-based Generative Models）

我们已经展示了变分扩散模型可以通过简单地优化一个神经网络 $ s_{\theta}(x_t, t) $ 来预测 $x_t $ 在任意噪声水平下的得分函数 $ \nabla \log p(x_t) $ 来学习。然而，在我们的推导中，分数项是通过应用Tweedie方程得出的；这并不一定为我们提供了关于得分函数究竟是什么或者为什么值得建模的深刻直觉或洞察。幸运的是，我们可以向另一类生成模型——基于分数的生成模型（Score-based Generative Models）来获得这种直觉。结果表明，我们可以展示我们之前推导出的 VDM 公式与等价的基于得分的生成模型公式是一致的，这使我们能够灵活地在这两种解释之间随意切换。

为了开始理解优化得分函数的意义，我们绕道并重新审视基于能量的模型（energy-based models）。任意灵活的概率分布可以写成以下形式：

$$
\begin{align}
    p_{\boldsymbol{\theta}}(\boldsymbol{x}) = \frac{1}{Z_{\boldsymbol{\theta}}}e^{-f_{\boldsymbol{\theta}}(\boldsymbol{x})}
\end{align}
$$


其中 $ f_{\theta}(x) $ 是一个任意灵活的、可参数化的函数，称为能量函数，通常由神经网络建模，$ Z_{\theta} $ 是一个规范化常数，以确保积分 $ \int p_{\theta}(x) \, dx = 1 $。学习这样一个分布的一种方法是最大似然；然而，这需要能够灵活地计算规范化常数 $ Z_{\theta} = \int e^{-f_{\theta}(x)} \, dx $，对于复杂 $ f_{\theta}(x) $ 函数来说可能是不可能的。

避免计算或建模规范化常数的一种方法是使用一个神经网络 $ s_{\theta}(x) $ 来学习分布 $ p(x) $ 的得分函数 $ \nabla \log p(x) $ 而不是能量函数。这是由以下观察所激发的：对上面方程的两边取对数后进行微分得到：

$$
\begin{align}
\nabla_{\boldsymbol{x}} \log p_{\boldsymbol{\theta}}(\boldsymbol{x})
&= \nabla_{\boldsymbol{x}}\log(\frac{1}{Z_{\boldsymbol{\theta}}}e^{-f_{\boldsymbol{\theta}}(\boldsymbol{x})})\\
&= \nabla_{\boldsymbol{x}}\log\frac{1}{Z_{\boldsymbol{\theta}}} + \nabla_{\boldsymbol{x}}\log e^{-f_{\boldsymbol{\theta}}(\boldsymbol{x})}\\
&= -\nabla_{\boldsymbol{x}} f_{\boldsymbol{\theta}}(\boldsymbol{x})\\
&\approx \boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x})
\end{align}
$$


这可以自由地表示为一个神经网络，而不涉及任何规范化常数。得分模型可以通过最小化与真实得分函数的Fisher散度来优化：
$$
\begin{align}
    \mathbb{E}_{p(\boldsymbol{x})}\left[\left\lVert \boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}) - \nabla\log p(\boldsymbol{x})\right\rVert_2^2\right]
\end{align}
$$

得分函数代表什么？对于每一个 $ x $，对其对数似然关于 $ x $ 取梯度，本质上描述了数据空间中移动的方向，以进一步增加其似然。

直观地说，得分函数定义了整个数据 $ x $​ 所占据的空间上的一个向量场，指向模式。在图6的右图中被视觉化地表示出来。

>这是使用Langevin动力学生成的三个随机采样轨迹的可视化，所有轨迹都从相同的初始点开始，针对的是高斯混合模型。左侧的图形在三维等高线上绘制了这些采样轨迹，而右侧的图形则将采样轨迹与真实得分函数相对比。由于Langevin动力学采样过程中的随机噪声项，我们可以从相同的初始点生成来自不同模式的样本；如果没有这个噪声项，从固定点开始的采样将总是在每次试验中确定性地沿着得分走向同一个模式。

然后，通过学习真实数据分布的得分函数，我们可以通过从同一空间中的任意点开始，迭代地跟随得分直到到达一个模式来生成样本。这种采样过程被称为Langevin动力学，数学上描述为：
$$
\begin{align}
    \boldsymbol{x}_{i+1} \leftarrow \boldsymbol{x}_i + c\nabla\log p(\boldsymbol{x}_i) + \sqrt{2c}\boldsymbol{\epsilon},\quad i = 0, 1, ..., K
\end{align}
$$





<div align=center>
<img src="https://calvinyluo.com/assets/images/diffusion/score_sample.webp" alt="Visualizing sampled paths from Langevin Dynamics against ground-truth scores" style="zoom:50%;" />
</div>



其中 $ x_0 $ 是从先验分布（如均匀分布）中随机采样的，$ \epsilon \sim \mathcal{N}(\epsilon; 0, I) $ 是一个额外的噪声项，以确保生成的样本不会总是坍缩到一个模式上，而是在其周围徘徊以实现多样性。此外，由于学习到的得分函数是确定性的，涉及噪声项的采样为生成过程增加了随机性，使我们能够避免确定性的轨迹。当采样从位于多个模式之间的位置初始化时，这特别有用。Langevin 动力学采样和噪声项的好处在图6中以视觉方式展示。

观察到，方程(93)中的目标依赖于我们能够访问真实得分函数，对于像自然图像这样的复杂分布，我们是无法获得的。幸运的是，已经推导出了称为得分匹配（score matching）的替代技术，可以在不知道真实得分的情况下最小化这个Fisher散度，并且可以使用随机梯度下降来优化。

总的来说，学习将分布表示为得分函数，并使用它通过**马尔可夫链蒙特卡洛技术**（如Langevin动力学）生成样本，被称为基于分数的生成建模（Score-based Generative Modeling）。

标准得分匹配存在三个主要问题，这些问题由Song和Ermon详细说明[3]。首先，当 $ a_c $ 位于高维空间中的低维流形上时，得分函数是未定义的。这可以从数学上看出；所有不在低维流形上的点都有零概率，其对数是未定义的。这在尝试学习自然图像上的生成模型时特别不方便，因为已知自然图像位于整个空间的低维流形上。其次，通过标准得分匹配训练出的估计得分函数在低密度区域将不准确。这从我们在方程93中最小化的目标中可以看出。因为它是对 $ p(a) $ 的期望，并且明确地在来自它的样本上进行训练，模型将不会为很少看到或未见过的示例提供准确的学习信号。这是有问题的，因为我们的采样策略涉及从高维空间中的随机位置开始，这很可能是随机噪声，然后根据学习到的得分函数移动。由于我们正在跟随一个嘈杂或不准确的得分估计，最终生成的样本也可能是次优的，或者需要更多的迭代才能收敛到准确的输出。

最后，即使使用真实得分执行，Langevin动力学采样也可能不会混合。假设真实数据分布是两个不相交分布的混合
$$
 p(\alpha) = c_1 p_1(\alpha) + c_2 p_2(\alpha)
$$


然后，当计算得分时，这些混合系数丢失了，因为对数运算将系数从分布中分离出来，梯度运算将其置零。为了可视化这一点，请注意上图中显示的真实得分函数对三个分布之间的不同权重是不可知的；从所描述的初始点开始的Langevin动力学采样到达每个模式的概率大致相等，尽管右下角的模式在实际的高斯混合中具有更高的权重。

结果表明，通过向数据添加多个水平的高斯噪声，可以同时解决这三个缺点。首先，由于高斯噪声分布的支持是整个空间，扰动的数据样本将不再局限于低维流形。其次，添加大量的高斯噪声将增加每个模式在数据分布中覆盖的区域，从而在低密度区域增加更多的训练信号。最后，添加具有逐渐增加的方差的多个水平的高斯噪声将导致中间分布，这些分布尊重真实的混合系数。

正式地，我们可以选择一个正噪声水平序列 $ \{\sigma_t\}^{T}_{t=1} $ 并定义一系列逐渐受到干扰的数据分布：
$$
\begin{align}
p_{\sigma_t}(\boldsymbol{x}_t) = \int p(\boldsymbol{x})\mathcal{N}(\boldsymbol{x}_t; \boldsymbol{x}, \sigma_t^2\textbf{I})d\boldsymbol{x}
\end{align}
$$
然后，使用得分匹配学习神经网络 $ s_{\theta}(\alpha, t) $ 来同时学习所有噪声水平的得分函数：

$$
\begin{align}
\underset{\boldsymbol{\theta}}{\arg\min}\, \sum_{t=1}^T\lambda(t)\mathbb{E}_{p_{\sigma_t}(\boldsymbol{x}_t)}\left[\left\lVert \boldsymbol{s}_{\boldsymbol{\theta}}(\boldsymbol{x}, t) - \nabla\log p_{\sigma_t}(\boldsymbol{x}_t)\right\rVert_2^2\right]
\end{align}
$$


其中 $ \lambda(t) $ 是一个正的权重函数，它依赖于噪声水平 $ t $。注意，这个目标几乎完全匹配我们在方程84中推导出的用于训练变分扩散模型的目标。此外，作者提出了退火Langevin动力学采样作为一种生成过程，其中样本是通过按顺序运行Langevin动力学来产生的，每个 $ t = T, T - 1, \ldots, 2, 1 $。初始化是从某个固定的先验（如均匀分布）中选择的，每个后续的采样步骤从上一个模拟的最终样本开始。因为噪声水平在时间步 $ t $ 上稳步下降，并且我们随着时间的推移减小步长，样本最终会收敛到一个真实模式。这与在变分扩散模型的马尔可夫HVAE解释中执行的采样程序直接类似，其中随机初始化的数据向量在降低噪声水平的过程中被迭代地细化。

因此，我们在训练目标和采样程序上建立了变分扩散模型和基于得分的生成模型之间的明确联系。

一个问题是如何将扩散模型自然地推广到无限数量的时间步。在马尔可夫HVAE的视角下，这可以被解释为将层级的数量扩展到无穷 $ T \rightarrow \infty $。从等效的基于得分的生成模型的角度来表示这一点更清晰；在无限数量的噪声尺度下，图像在连续时间上的扰动可以被表示为一个随机过程，因此可以用随机微分方程（SDE）来描述。然后通过逆转SDE来执行采样，这自然需要在每个连续值的噪声水平上估计得分函数[12]。SDE的不同参数化本质上描述了不同的扰动方案，使得灵活地建模噪声过程成为可能。

## 指导（Guidance）

到目前为止，我们只关注了对数据分布 $ p(x) $ 的建模。然而，我们通常也对学习条件分布 $ p(x|y) $ 感兴趣，这将使我们能够通过条件信息 $ y $ 明确控制我们生成的数据。这构成了图像超分辨率模型的支柱，例如级联扩散模型，以及最先进的图像文本模型的关键，例如 DALL-E 2。

自然地添加条件信息的方法是在每次迭代中，简单地将其与时间步信息一起加入。回想我们方程 32 中的联合分布：

$$
p(\boldsymbol{x}_{0:T}) = p(\boldsymbol{x}_T)\prod_{t=1}^Tp_{\boldsymbol{\theta}}(\boldsymbol{x}_{t-1}\mid \boldsymbol{x}_t)
$$


然后，要将其变成一个条件扩散模型，我们可以简单地在每个转换步骤中添加任意的条件信息 $ y $：

$$
\begin{align}
p(\boldsymbol{x}_{0:T}\mid y) = p(\boldsymbol{x}_T)\prod_{t=1}^Tp_{\boldsymbol{\theta}}(\boldsymbol{x}_{t-1}\mid \boldsymbol{x}_t, y)
\end{align}
$$


例如，$ y $ 可以是图像-文本生成中的文本编码，或者是执行超分辨率的低分辨率图像。因此，我们可以像以前一样学习变分扩散模型的核心神经网络，通过预测 $ \hat{x}_{\theta}(x_t, t, y) \approx x_0 $，$ \hat{\epsilon}_{\theta}(x_t, t, y) \approx \epsilon_0 $，或 $ s_{\theta}(x_t, t, y) \approx \nabla \log p(x_t|y) $ 来实现每种所需的解释和实现。

这种基本公式的一个缺点是，以这种方式训练的条件扩散模型可能会学会忽略或淡化任何给定的条件信息。因此，提出了指导作为一种更明确地控制模型给予条件信息权重的方法，代价是牺牲样本多样性。两种最流行的形式的指导被称为分类器指导（Classifier Guidance）和无分类器指导（Classifier-Free Guidance）。

### 分类器指导（Classifier Guidance）

让我们从扩散模型的基于得分的方程开始，我们的目标是在任意噪声水平 $ t $ 下学习条件模型的得分 $ \nabla \log p(x_t|y) $ 。为了简洁起见， $ \nabla $ 是 $ \nabla_{x_t} $ 的简写。根据贝叶斯定理，我们可以推导出以下等价形式：

$$
\begin{align}
\nabla\log p(\boldsymbol{x}_t\mid y) &= \nabla\log \left( \frac{p(\boldsymbol{x}_t)p(y\mid \boldsymbol{x}_t)}{p(y)} \right)\\
&= \nabla\log p(\boldsymbol{x}_t) + \nabla\log p(y\mid \boldsymbol{x}_t) - \nabla\log p(y)\\
&= \underbrace{\nabla\log p(\boldsymbol{x}_t)}_\text{unconditional score} + \underbrace{\nabla\log p(y\mid \boldsymbol{x}_t)}_\text{adversarial gradient}
\end{align}
$$


这里， 我们利用了相对于 $x_t $ 的 $ \log p(y) $ 的梯度为零的事实。

我们最终推导出的结果可以解释为学习一个无分类器得分函数与一个分类器 $ p(y|x_t) $ 的对抗梯度的组合。因此，在分类器指导中，无分类器扩散模型的得分像之前推导的那样学习，同时学习一个分类器，该分类器接收任意噪声的 $x_t $ 并尝试预测条件信息 $ y $。然后，在采样过程中，用于退火Langevin动力学的整体条件得分函数被计算为无分类器得分函数和噪声分类器的对抗梯度的总和。

为了引入细粒度控制，以鼓励或阻止模型考虑条件信息，分类器指导通过一个 $ \gamma $ 超参数项来缩放噪声分类器的对抗梯度。在分类器指导下学习得分函数可以总结为：

$$
\begin{align}
    \nabla\log p(\boldsymbol{x}_t\mid y) &= \nabla\log p(\boldsymbol{x}_t) + \gamma\nabla\log p(y\mid \boldsymbol{x}_t)
\end{align}
$$

直观地说：

- 当 $ \gamma = 0 $ 时，条件扩散模型完全忽略条件信息，
- 当 $ \gamma $ 很大时，条件扩散模型学会产生严重依赖于条件信息的样本。这将以牺牲样本多样性为代价，因为它只会产生容易从提供的条件下重新生成的样本，即使在噪声水平下也是如此。

分类器指导的一个缺点是它依赖于单独学习分类器。由于分类器必须处理任意噪声的输入，这超出了大多数现有的预训练分类模型的优化范围，因此它必须与扩散模型一起学习。

### 无分类器指导（Classifier-Free Guidance）

在无分类器指导中，作者放弃了训练单独的分类器模型，转而使用一个无分类器扩散模型和一个条件扩散模型。为了推导无分类器指导下的得分函数，我们可以先重新排列方程(101)：

$$
\begin{align}
    \nabla\log p(y\mid \boldsymbol{x}_t) = \nabla\log p(\boldsymbol{x}_t\mid y) - \nabla\log p(\boldsymbol{x}_t)
\end{align}
$$


然后，将这个代入方程(102)，我们得到：
$$
\begin{align}
\nabla\log p(\boldsymbol{x}_t\mid y)
&= \nabla\log p(\boldsymbol{x}_t) + \gamma\left(\nabla\log p(\boldsymbol{x}_t\mid y) - \nabla\log p(\boldsymbol{x}_t)\right)\\
&= \nabla\log p(\boldsymbol{x}_t) + \gamma\nabla\log p(\boldsymbol{x}_t\mid y) - \gamma\nabla\log p(\boldsymbol{x}_t)\\
&= \underbrace{\gamma\nabla\log p(\boldsymbol{x}_t\mid y)}_\text{conditional score} + \underbrace{(1 - \gamma)\nabla\log p(\boldsymbol{x}_t)}_\text{unconditional score}
\end{align}
$$
再一次，$ \gamma $ 是一个控制我们学习的条件模型对条件信息关注程度的项。

- 当 $ \gamma = 0 $ 时，学习的条件模型完全忽略条件器并学习一个无分类器扩散模型。
- 当 $ \gamma = 1 $ 时，模型明确学习没有指导的普通条件分布。
- 当 $ \gamma > 1 $ 时，扩散模型不仅优先考虑条件得分函数，而且还向远离无分类器得分函数的方向移动。换句话说，它减少了生成不使用条件信息的样本的概率，而有利于明确使用条件信息的样本。这也通过牺牲样本多样性为代价，增加了生成与条件信息准确匹配的样本的可能性。

由于学习两个单独的扩散模型是昂贵的，我们可以将条件和无分类器扩散模型一起作为一个单一的条件模型学习；无分类器扩散模型可以通过将条件信息替换为固定常数值（如零）来查询，这本质上是在条件信息上执行随机dropout。无分类器指导之所以优雅，是因为它使我们能够更精细地控制我们的条件生成过程， 同时除了训练扩散模型之外不需要任何其他操作。

## 总结

让我们回顾一下这篇博客文章中的发现。首先，我们将变分扩散模型（Variational Diffusion Models）推导为马尔可夫分层变分自编码器（Markovian Hierarchical Variational Autoencoder）的一个特例，其中三个关键假设使得对ELBO的计算可行并且可扩展优化成为可能。然后，我们证明了优化变分扩散模型归结为学习一个神经网络来预测以下三个潜在目标之一：

- 从任何随机噪声版本中预测原始源图像，
- 从任何随机噪声图像中预测原始源噪声，
- 在任意噪声水平下预测噪声图像的得分函数。

然后，我们深入探讨了学习得分函数的含义，并通过Tweedie方程明确地将扩散模型的变分视角与基于分数的生成建模视角联系起来。最后，我们介绍了如何通过指导使用扩散模型学习条件分布。

尽管扩散模型在生成建模方面取得了令人难以置信的成功，但仍有一些缺点需要考虑，这些缺点是未来工作令人兴奋的方向：

- 我们人类似乎不太可能以这种方式自然地建模和生成数据；我们不会将样本生成为随机噪声，然后逐步去噪。
- 变分扩散模型（VDM）不产生可解释的潜在表示。而变分自编码器（VAE）有望通过优化其编码器学习到一个结构化的潜在空间，但在变分扩散模型中，每个时间步的编码器已经被给定为线性高斯模型，并且不能灵活地优化。因此，中间潜在表示被限制为原始输入的噪声版本。
- 潜在变量被限制为与原始输入相同的维度，进一步阻碍了学习有意义、压缩的潜在结构的努力。
- 采样是一个昂贵的过程，因为在两种方程下都必须运行多个去噪步骤。
- 为了确保最终潜在表示完全是高斯噪声，时间步的数量通常非常大；在采样期间，我们必须迭代所有这些时间步来生成一个样本。

最后需要注意的是，扩散模型的成功突显了分层变分自编码器（Hierarchical VAEs）作为生成模型的强大能力。我们已经展示了，当我们推广到无限的潜在层次时，即使编码器是平凡的，潜在维度是固定的，且假设了马尔可夫转换，我们仍然能够学习强大的数据模型。这表明，在一般情况下，深度HVAEs可以取得进一步的性能提升，其中可以潜在地学习复杂的编码器和语义上有意义的潜在空间。

## Reference

- Blogs：
  - [What are Diffusion Models?](https：//lilianweng.github.io/posts/2021-07-11-diffusion-models/)
  - [Generative Modeling by Estimating Gradients of the Data Distribution (blog)](https：//yang-song.github.io/blog/2021/score/)
  - [Guidance： A Cheat Code for Diffusion Models](https：//benanne.github.io/2022/05/26/guidance.html)
  - [Score based diffusions explained in just one paragraph](https：//kidger.site/thoughts/score-based-diffusions-explained-in-just-one-paragraph/)
  - [A Path to the VDM Diffusion Loss](https：//blog.alexalemi.com/diffusion.html)
- Papers：
  - [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https：//arxiv.org/abs/1503.03585)
  - [Denoising Diffusion Probabilistic Models](https：//arxiv.org/abs/2006.11239)
  - [Generative Modeling by Estimating Gradients of the Data Distribution (paper)](https：//arxiv.org/abs/2006.09011)
  - [Score-Based Generative Modeling through Stochastic Differential Equations](https：//arxiv.org/abs/2011.13456)
  - [Diffusion Models Beat GANs on Image Synthesis](https：//arxiv.org/abs/2105.05233)
  - [Classifier-Free Diffusion Guidanc](https：//arxiv.org/abs/2207.12598)
