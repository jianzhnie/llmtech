# 理解扩散模型

## 引言：生成模型

给定从感兴趣分布中观察到的样本 $x$，生成模型的目标是学习模拟其真实的数据分布 $p(x)$。一旦学习完成，我们就可以随意从我们的近似模型中生成新的样本。此外，在某些公式化下，我们还可以使用学习到的模型来评估观测或采样数据的可能性。

在当前文献中，有几种众所周知的方向，这里只是简要地在高层次上介绍。生成对抗网络（GANs）以对抗性方式学习复杂分布的采样过程。另一类被称为“基于似然的”生成模型，寻求学习一个模型，为观测到的数据样本分配高似然。这包括自回归模型、归一化流模型和变分自编码器（VAEs）。另一种类似的方法是基于能量的建模，其中学习一个分布作为任意灵活的能量函数，然后进行归一化。

分数生成模型与此密切相关；它们不是学习模拟能量函数本身，而是学习能量模型的分数作为神经网络。在这项工作中，我们将探索和回顾扩散模型，正如我们将展示的，它们既有基于似然的解释，也有基于分数的解释。我们以极其详尽的细节展示了这些模型背后的数学，目的是让任何人都能够跟随并理解扩散模型是什么以及它们是如何工作的。

## 背景：ELBO、VAE和层次VAE

对于许多模态，可以认为观察到的数据由一个相关的不可见潜在变量表示或生成，我们可以通过随机变量 $ z$ 来表示。表达这一想法的最好直觉是通过柏拉图的洞穴寓言。在寓言中，一群人一生中都被锁在一个洞穴里，只能看到投射在他们面前墙上的二维阴影，这些阴影是由在火前经过的不可见的三维物体产生的。对于这些人来说，他们观察到的一切都是由他们永远无法看到的更高维度的抽象概念决定的。

类似地，我们在现实世界中遇到的物体也可能是由某些更高层次的表示生成的；例如，这些表示可能封装了颜色、大小、形状等抽象属性。然后，我们观察到的可以被解释为这些抽象概念的三维投影或实例化，就像洞穴居民观察到的实际上是三维物体的二维投影一样。尽管洞穴居民永远看不到（甚至完全理解不了）隐藏的物体，他们仍然可以推理并推断出关于它们的信息；类似地，我们可以近似潜在表示来描述我们观察到的数据。

尽管柏拉图的洞穴寓言阐释了潜在变量作为可能不可观察的表示来决定观察结果的背后思想，但这个类比的一个缺点是，在生成建模中，我们通常寻求学习低于观察维度的潜在表示，而不是高于观察维度的。这是因为试图学习一个比观察结果更高维度的表示，没有强有力的先验知识是没有用的。另一方面，学习低维潜在变量也可以看作是一种压缩形式，并且有可能揭示出描述观察结果的语义上有意义的结构。

### 证据下界（Evidence Lower Bound, ELBO）

数学上，我们可以想象潜在变量和我们观察到的数据由一个联合分布 $ p(x, z)$ 来建模。回想一下，生成建模中称为“基于似然的”方法之一是学习一个模型来最大化所有观测到的 $ x$ 的似然 $ p(x)$。我们有两种方式可以操作这个联合分布来恢复我们观测数据的似然 $ p(x)$；我们可以明确地对潜在变量 $ z$ 进行边缘化：

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

直接计算和最大化似然 $ p(x)$ 是困难的，因为它要么涉及将所有潜在变量 $ z$ 在公式 (1) 中积分出来，这对于复杂模型来说是不可行的，要么涉及获得真实的潜在编码器 $ p(z|x)$ 在公式 (2) 中。然而，使用这两个方程，我们可以推导出一个称为证据下界（ELBO）的项，顾名思义，它是证据的一个下界。在这种情况下，证据被量化为观测数据的对数似然。然后，最大化ELBO成为一个代理目标，用于优化潜在变量模型；在最好的情况下，当ELBO被强有力地参数化并且被完美优化时，它与证据完全等价。正式地，ELBO的方程是：

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

这里，$ q_{\phi}(z|x)$ 是一个灵活的近似变分分布，其参数 $ \phi$ 是我们寻求优化的。直观上，它可以被看作是一个可参数化的模型，学习估计给定观测 $ x$ 的潜在变量的真实分布；换句话说，它寻求近似真实的后验 $ p(z|x)$。正如我们将在探索变分自编码器时看到的，通过调整参数 $ \phi$ 来增加下界，我们可以访问可以用来模拟真实数据分布并从中采样的组件，从而学习一个生成模型。

### 证据下界（ELBO）的推导

让我们从公式(1)开始推导ELBO：
$$
\begin{align}
\log p(\boldsymbol{x})
&= \log \int p(\boldsymbol{x}, \boldsymbol{z})d\boldsymbol{z}\\
&= \log \int \frac{p(\boldsymbol{x}, \boldsymbol{z})q_{\boldsymbol{\phi}}(\boldsymbol{z}\mid\boldsymbol{x})}{q_{\boldsymbol{\phi}}(\boldsymbol{z}\mid\boldsymbol{x})}d\boldsymbol{z}\\
&= \log \mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z}\mid\boldsymbol{x})}\left[\frac{p(\boldsymbol{x}, \boldsymbol{z})}{q_{\boldsymbol{\phi}}(\boldsymbol{z}\mid\boldsymbol{x})}\right]\\
&\geq \mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z}\mid\boldsymbol{x})}\left[\log \frac{p(\boldsymbol{x}, \boldsymbol{z})}{q_{\boldsymbol{\phi}}(\boldsymbol{z}\mid\boldsymbol{x})}\right]
\end{align}
$$


在这个推导中，我们通过应用詹森不等式直接得到了我们的下界。然而，这并没有提供关于实际发生的情况的有用信息；关键的是，这个证明没有给出为什么ELBO 实际上是证据下界的直觉，因为詹森不等式把它模糊处理了。此外，仅仅知道ELBO是数据的下界，并没有真正告诉我们为什么我们想要将其最大化作为目标。为了更好地理解证据和ELBO之间的关系，让我们进行另一个推导，这次使用公式(2)：

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


从这个推导中，我们清楚地从公式(15)观察到证据等于ELBO加上近似后验 $ q_{\phi}(z|x)$ 和真实后验 $ p(z|x)$ 之间的KL散度。实际上，正是这个KL散度项在第一个推导的公式(8)中通过詹森不等式被神奇地移除了。理解这个项是理解ELBO和证据之间关系的关键，也是理解为什么优化ELBO是一个合适的目标的原因。

首先，我们现在知道为什么ELBO确实是一个下界：

> 证据和ELBO之间的差异是一个严格非负的KL项，因此ELBO的值永远不能超过证据。

其次，我们探索为什么我们寻求最大化ELBO。

> 引入了我们想要建模的潜在变量 $ z$，我们的目标是学习描述我们观测数据的潜在结构。换句话说，我们想要优化我们的变分后验 $ q_{\phi}(z|x)$ 的参数，以便它完全匹配真实的后验分布 $ p(z|x)$，这通过最小化它们的KL散度（理想情况下为零）来实现。不幸的是，直接最小化这个KL散度项是不可行的，因为我们没有访问真实的 $ p(z|x)$ 分布。但是，请注意，在公式(15)的左侧，我们数据的似然性（因此我们的证据项 $ \log p(x)$）相对于 $ \phi$ 总是一个常数，因为它是通过从联合分布 $ p(x, z)$ 中边缘化所有潜在 $ z$ 来计算的，根本不依赖于 $ \phi$。由于ELBO和KL散度项加起来是一个常数，对ELBO项的任何最大化都必然引起KL散度项的等量最小化。因此，ELBO可以作为一个代理目标来最大化，作为学习如何完美地模拟真实潜在后验分布的代理；我们越优化ELBO，我们的近似后验就越接近真实后验。此外，一旦训练完成，ELBO还可以用来估计观测或生成数据的可能性，因为它被学习来近似模型证据 $ \log p(x)$。

### 变分自编码器（Variational Autoencoders）

在变分自编码器（VAE）的默认公式中，我们直接最大化证据下界（ELBO）。这种方法是变分的，因为我们在由φ参数化的潜在后验分布族中优化最佳的 $ q_{\phi}(z|x)$。它被称为自编码器，因为它让人想起传统的自编码器模型，其中输入数据被训练在经历一个中间瓶颈表示步骤后自我预测。为了明确这种联系，让我们进一步剖析ELBO项：
$$
\begin{align}
\mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z}\mid\boldsymbol{x})}\left[\log\frac{p(\boldsymbol{x}, \boldsymbol{z})}{q_{\boldsymbol{\phi}}(\boldsymbol{z}\mid\boldsymbol{x})}\right]
&= \mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z}\mid\boldsymbol{x})}\left[\log\frac{p_{\boldsymbol{\theta}}(\boldsymbol{x}\mid\boldsymbol{z})p(\boldsymbol{z})}{q_{\boldsymbol{\phi}}(\boldsymbol{z}\mid\boldsymbol{x})}\right]\\
&= \mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z}\mid\boldsymbol{x})}\left[\log p_{\boldsymbol{\theta}}(\boldsymbol{x}\mid\boldsymbol{z})\right] + \mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z}\mid\boldsymbol{x})}\left[\log\frac{p(\boldsymbol{z})}{q_{\boldsymbol{\phi}}(\boldsymbol{z}\mid\boldsymbol{x})}\right]\\
&= \underbrace{\mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z}\mid\boldsymbol{x})}\left[\log p_{\boldsymbol{\theta}}(\boldsymbol{x}\mid\boldsymbol{z})\right]}_\text{reconstruction term} - \underbrace{\mathcal{D}_{\text{KL}}(q_{\boldsymbol{\phi}}(\boldsymbol{z}\mid\boldsymbol{x}) \mid\mid p(\boldsymbol{z}))}_\text{prior matching term}
\end{align}
$$

在这个情况下，我们学习了一个中间瓶颈分布 $ q_{\phi}(z|x)$，它可以被看作是一个编码器；它将输入转换为可能潜在的分布。同时，我们学习了一个确定性函数 $ p_{\theta}(x|z)$ 将给定的潜在向量 $ z$ 转换为观测 $ x$，这可以被解释为解码器。

![Visualizing a Variational Autoencoder](https://calvinyluo.com/assets/images/diffusion/vae.webp)

> Visualizing a vanilla Variational Autoencoder. A latent encoder and decoder are learned jointly through the reparameterization trick [6, 7]

方程(19)中的两个项各自有直观的描述：第一项测量了从我们的变分分布中解码器的重建似然性；这确保了学习到的分布是模拟有效的潜在，原始数据可以从中再生。第二项测量了学习到的变分分布与我们对潜在变量持有的先验信念的相似性。最小化这个项鼓励编码器实际学习一个分布，而不是塌缩成一个狄拉克δ函数。因此，最大化ELBO等同于最大化其第一项并最小化其第二项。

一个VAE的定义特征是ELBO是如何联合优化参数φ和θ。VAE的编码器通常选择为多变量高斯分布，具有对角线协方差，而先验通常选择为标准多变量高斯分布：

$$
\begin{align}
    q_{\boldsymbol{\phi}}(\boldsymbol{z}\mid\boldsymbol{x}) &= \mathcal{N}(\boldsymbol{z}; \boldsymbol{\mu}_{\boldsymbol{\phi}}(\boldsymbol{x}), \boldsymbol{\sigma}_{\boldsymbol{\phi}}^2(\boldsymbol{x})\textbf{I})\\
    p(\boldsymbol{z}) &= \mathcal{N}(\boldsymbol{z}; \boldsymbol{0}, \textbf{I})
\end{align}
$$


然后，ELBO中的KL散度项可以被解析计算，而重建项可以使用蒙特卡洛估计来近似。我们的目标可以被重写为：
$$
\begin{align}
& \quad \,\underset{\boldsymbol{\phi}, \boldsymbol{\theta}}{\arg\max}\, \mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z}\mid\boldsymbol{x})}\left[\log p_{\boldsymbol{\theta}}(\boldsymbol{x}\mid\boldsymbol{z})\right] - \mathcal{D}_{\text{KL}}(q_{\boldsymbol{\phi}}(\boldsymbol{z}\mid\boldsymbol{x}) \mid\mid p(\boldsymbol{z})) \nonumber \\
& \approx \underset{\boldsymbol{\phi}, \boldsymbol{\theta}}{\arg\max}\, \sum_{l=1}^{L}\log p_{\boldsymbol{\theta}}(\boldsymbol{x}\mid\boldsymbol{z}^{(l)}) - \mathcal{D}_{\text{KL}}(q_{\boldsymbol{\phi}}(\boldsymbol{z}\mid\boldsymbol{x}) \mid\mid p(\boldsymbol{z}))
\end{align}
$$

其中潜在变量 $ \{z^{(l)}\}^{L}_{l=1}$ 是从 $ q_{\phi}(z|x)$ 中采样的，对于数据集中的每个观测 $ x$。然而，在这个默认设置中出现了一个问题：我们计算损失的每个 $ z^{(l)}$ 都是通过一个随机采样过程生成的，这通常是不可微分的。幸运的是，当 $ q_{\phi}(z|x)$ 被设计为模拟某些分布（包括多变量高斯分布）时，这个问题可以通过重参数化技巧来解决。

重参数化技巧将一个随机变量重写为一个噪声变量的确定性函数；这允许通过梯度下降来优化非随机项。例如，从正态分布 $ x \sim \mathcal{N}(x; \mu, \sigma^2)$ 中采样的随机变量，其中 $ \mu$ 是任意的均值，$ \sigma^2$ 是方差，可以被重写为：

$$
\begin{align*}
    x &= \mu + \sigma\epsilon \quad \text{with } \epsilon \sim \mathcal{N}(\epsilon; 0, \text{I})
\end{align*}
$$

换句话说，任意的高斯分布可以被解释为标准高斯（其中 $ \epsilon$ 是一个样本），通过添加将均值从零移动到目标均值 $ \mu$，以及通过目标方差 $ \sigma^2$ 拉伸方差。因此，通过重参数化技巧，从任意高斯分布中采样可以通过从标准高斯中采样，将结果乘以目标标准差，然后通过目标均值进行平移来执行。

在VAE中，每个 $ z$ 被计算为输入 $ x$ 和辅助噪声变量 $ \epsilon$ 的确定性函数：

$$
\begin{align*}
    \boldsymbol{z} &= \boldsymbol{\mu}_{\boldsymbol{\phi}}(\boldsymbol{x}) + \boldsymbol{\sigma}_{\boldsymbol{\phi}}(\boldsymbol{x})\odot\boldsymbol{\epsilon} \quad \text{with } \boldsymbol{\epsilon} \sim \mathcal{N}(\boldsymbol{\epsilon};\boldsymbol{0}, \textbf{I})
\end{align*}
$$

其中 $ \odot$ 表示逐元素乘积。在重参数化的 $ z$ 版本下，梯度可以相对于 $ \phi$ 计算，以优化 $ \mu_{\phi}$ 和 $ \sigma_{\phi}$。因此，VAE利用重参数化技巧和蒙特卡洛估计联合优化ELBO。

训练完VAE后，可以通过从潜在空间 $ p(z)$ 中采样，然后将其通过解码器来生成新数据。当潜在变量 $ z$ 的维度小于输入 $ x$ 的维度时，变分自编码器特别有趣，因为那时我们可能正在学习紧凑、有用的表示。此外，当学习到一个语义上有意义的潜在空间时，潜在向量可以在传递给解码器之前进行编辑，以更精确地控制生成的数据。

### 层次变分自编码器（Hierarchical Variational Autoencoders）

层次变分自编码器（HVAE）是变分自编码器（VAE）的泛化，它扩展到潜在变量的多个层次。在这个框架下，潜在变量本身被视为由其他更高级别的、更抽象的潜在变量生成的。直观地说，就像我们将自己的三维观察对象视为由更高级别的抽象潜在变量生成的一样，柏拉图洞穴中的人将三维对象视为由他们的二维观察结果生成的潜在变量。因此，从柏拉图洞穴居民的视角来看，他们的观察可以被视为由两个或更多层次的潜在层次结构建模。

![Visualizing a Hierarchical VAE](https://calvinyluo.com/assets/images/diffusion/hvae.webp)

> 马尔可夫分层变分自动编码器。生成过程被建模为马尔可夫链，其中每个潜在变量仅从前一个潜在变量生成。

在一般的HVAE中，每个潜在变量被允许依赖于之前所有的潜在变量。然而，在本文中，我们关注的是一个特殊案例，我们称之为马尔可夫层次变分自编码器（MHVAE）。在MHVAE中，生成过程是一个马尔可夫链；也就是说，每个层次向下的转换都是马尔可夫的，其中解码每个潜在变量 $ z_t$ 仅依赖于前一个潜在变量 $ z_{t+1}$。直观上，这可以简单地看作是在彼此之上堆叠VAE，如图2所示；描述这个模型的另一个恰当的术语是递归VAE。数学上，我们表示马尔可夫HVAE的联合分布和后验如下：
$$
\begin{align}
    p(\boldsymbol{x}, \boldsymbol{z}_{1:T}) &= p(\boldsymbol{z}_T)p_{\boldsymbol{\theta}}(\boldsymbol{x}\mid\boldsymbol{z}_1)\prod_{t=2}^{T}p_{\boldsymbol{\theta}}(\boldsymbol{z}_{t-1}\mid\boldsymbol{z}_{t})
\end{align}
$$


他的先验表示为：
$$
\begin{align}
    q_{\boldsymbol{\phi}}(\boldsymbol{z}_{1:T}\mid\boldsymbol{x}) &= q_{\boldsymbol{\phi}}(\boldsymbol{z}_1\mid\boldsymbol{x})\prod_{t=2}^{T}q_{\boldsymbol{\phi}}(\boldsymbol{z}_{t}\mid\boldsymbol{z}_{t-1})
\end{align}
$$


然后，我们可以很容易地将ELBO扩展为：
$$
\begin{align}
\log p(\boldsymbol{x}) &= \log \int p(\boldsymbol{x}, \boldsymbol{z}_{1:T}) d\boldsymbol{z}_{1:T}\\
&= \log \int \frac{p(\boldsymbol{x}, \boldsymbol{z}_{1:T})q_{\boldsymbol{\phi}}(\boldsymbol{z}_{1:T}\mid\boldsymbol{x})}{q_{\boldsymbol{\phi}}(\boldsymbol{z}_{1:T}\mid\boldsymbol{x})} d\boldsymbol{z}_{1:T}\\
&= \log \mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z}_{1:T}\mid\boldsymbol{x})}\left[\frac{p(\boldsymbol{x}, \boldsymbol{z}_{1:T})}{q_{\boldsymbol{\phi}}(\boldsymbol{z}_{1:T}\mid\boldsymbol{x})}\right]\\
&\geq \mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z}_{1:T}\mid\boldsymbol{x})}\left[\log \frac{p(\boldsymbol{x}, \boldsymbol{z}_{1:T})}{q_{\boldsymbol{\phi}}(\boldsymbol{z}_{1:T}\mid\boldsymbol{x})}\right]
\end{align}
$$


然后，我们可以将我们的联合分布（公式(23)）和后验（公式(24)）代入公式(28)，得到一个替代形式：

$$
\begin{align}
\mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z}_{1:T}\mid\boldsymbol{x})}\left[\log \frac{p(\boldsymbol{x}, \boldsymbol{z}_{1:T})}{q_{\boldsymbol{\phi}}(\boldsymbol{z}_{1:T}\mid\boldsymbol{x})}\right]
&= \mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z}_{1:T}\mid\boldsymbol{x})}\left[\log \frac{p(\boldsymbol{z}_T)p_{\boldsymbol{\theta}}(\boldsymbol{x}\mid\boldsymbol{z}_1)\prod_{t=2}^{T}p_{\boldsymbol{\theta}}(\boldsymbol{z}_{t-1}\mid\boldsymbol{z}_{t})}{q_{\boldsymbol{\phi}}(\boldsymbol{z}_1\mid\boldsymbol{x})\prod_{t=2}^{T}q_{\boldsymbol{\phi}}(\boldsymbol{z}_{t}\mid\boldsymbol{z}_{t-1})}\right]
\end{align}
$$
正如我们将在下面探讨变分扩散模型时所示，这个目标可以进一步分解为可解释的组成部分。



## 变分扩散模型（Variational Diffusion Models）

变分扩散模型（VDM）可以简单地看作是一个具有三个关键限制的马尔可夫层次变分自编码器（Markovian Hierarchical Variational Autoencoder, MHVAE）：

1. 潜在维度完全等于数据维度。
2. 每个时间步的潜在编码器的结构不是学习得到的；它被预定义为线性高斯模型。换句话说，它是以前一时间步的输出为中心的高斯分布。
3. 潜在编码器的高斯参数随时间变化，使得最终时间步T的潜在分布是标准高斯分布。

此外，我们明确保持了从标准马尔可夫层次变分自编码器继承的层次转换之间的马尔可夫性质。

让我们展开这些假设的含义。从第一个限制开始，现在我们可以将真实的数据样本和潜在变量表示为 $ x_t$，其中 $ t = 0$ 表示真实的数据样本，$ t \in [1, T]$ 表示具有层次索引 t 的相应潜在变量。VDM的后验与MHVAE的后验相同（公式24），但现在可以重写为：

$$
\begin{align}
    q(\boldsymbol{x}_{1:T}\mid\boldsymbol{x}_0) = \prod_{t = 1}^{T}q(\boldsymbol{x}_{t}\mid\boldsymbol{x}_{t-1})
\end{align}
$$
根据第二个假设，我们知道每个潜在变量在编码器中的分布是一个以它前一个层次潜在变量为中心的高斯分布。与MHVAE不同，在VDM中，每个时间步t的编码器的结构不是学习得到的；它被固定为线性高斯模型，其中均值和标准差可以预先设置为超参数，或者作为参数学习。我们用均值 $ \mu_t(x_t) = \sqrt{\alpha_t} x_{t-1}$ 和方差 $ \Sigma_t(x_t) = (1 - \alpha_t)I$ 来参数化高斯编码器，其中系数的形式被选择以保持潜在变量的方差在相似的尺度上；换句话说，编码过程是方差保持的。注意，允许使用其他高斯参数化，并且会导致类似的推导。主要的收获是 $ \alpha_t$ 是一个（可能是可学习的）系数，它可以根据层次深度t变化，以保持灵活性。数学上，编码器转换表示为：

$$
\begin{align}
    q(\boldsymbol{x}_{t}\mid\boldsymbol{x}_{t-1}) = \mathcal{N}(\boldsymbol{x}_{t} ; \sqrt{\alpha_t} \boldsymbol{x}_{t-1}, (1 - \alpha_t) \textbf{I})
\end{align}
$$


从第三个假设中，我们知道 $ \alpha_t$ 按照固定或可学习的时间表随时间演变，使得最终潜在的分布 $ p(x_T)$ 是标准高斯分布。然后，我们可以更新马尔可夫HVAE的联合分布（公式23）来写出VDM的联合分布：

$$
\begin{align}
p(\boldsymbol{x}_{0:T}) &= p(\boldsymbol{x}_T)\prod_{t=1}^{T}p_{\boldsymbol{\theta}}(\boldsymbol{x}_{t-1}\mid\boldsymbol{x}_t) \\
\text{where,}&\nonumber\\
p(\boldsymbol{x}_T) &= \mathcal{N}(\boldsymbol{x}_T; \boldsymbol{0}, \textbf{I})
\end{align}
$$
总的来说，这组假设描述了图像输入随时间逐步噪声化的过程；我们通过逐步添加高斯噪声逐渐破坏一个图像，直到最终它完全变成纯高斯噪声。这个过程在图3中以视觉方式呈现。

![Visualizing a Variational Diffusion Model](https://calvinyluo.com/assets/images/diffusion/vdm_base.webp)

> 变分扩散模型的视觉表示。输入随着时间的推移稳定地产生噪声，直到它变得与高斯噪声相同；扩散模型学会扭转这个过程。

请注意，我们的编码器分布 $ q(x_t | x_{t-1})$ 已经不再由φ参数化，因为它们完全被建模为具有定义的均值和方差参数的高斯分布。因此，在VDM中，我们只对学习条件分布 $ p_{\theta}(x_{t-1} | x_t)$ 感兴趣，这样我们就可以模拟新数据。在优化VDM之后，采样过程就像从 $ p(x_T)$ 中采样高斯噪声，然后迭代地运行去噪转换 $ p_{\theta}(x_{t-1} | x_t)$ 进行T步来生成一个新的 $ x_0$。

像任何HVAE一样，VDM可以通过最大化ELBO来优化，它可以被推导为：
$$
\begin{align}
\log p(\boldsymbol{x})
&\geq \mathbb{E}_{q(\boldsymbol{x}_{1:T}\mid\boldsymbol{x}_0)}\left[\log \frac{p(\boldsymbol{x}_{0:T})}{q(\boldsymbol{x}_{1:T}\mid\boldsymbol{x}_0)}\right] \\
&=  \begin{aligned}[t]
      \underbrace{\mathbb{E}_{q(\boldsymbol{x}_{1}\mid\boldsymbol{x}_0)}\left[\log p_{\theta}(\boldsymbol{x}_0\mid\boldsymbol{x}_1)\right]}_\text{reconstruction term} &- \underbrace{\mathbb{E}_{q(\boldsymbol{x}_{T-1}\mid\boldsymbol{x}_0)}\left[\mathcal{D}_{\text{KL}}(q(\boldsymbol{x}_T\mid\boldsymbol{x}_{T-1}) \mid\mid p(\boldsymbol{x}_T))\right]}_\text{prior matching term} \\
      &- \sum_{t=1}^{T-1}\underbrace{\mathbb{E}_{q(\boldsymbol{x}_{t-1}, \boldsymbol{x}_{t+1}\mid\boldsymbol{x}_0)}\left[\mathcal{D}_{\text{KL}}(q(\boldsymbol{x}_t\mid\boldsymbol{x}_{t-1}) \mid\mid p_{\theta}(\boldsymbol{x}_{t}\mid\boldsymbol{x}_{t+1}))\right]}_\text{consistency term}
    \end{aligned}
\end{align}
$$


这个推导形式的ELBO可以解释为其各个组成部分的含义：

1. $ \mathbb{E}_{q(x_1|x_0)} \left[ \log p_{\theta}(x_0 | x_1) \right]$ 可以被解释为一个重建项，预测给定第一步潜在样本的原始数据样本的对数概率。这个术语也出现在标准VAE的ELBO中，可以类似地进行训练。

2. $ \mathbb{E}_{q(x_{T-1}|x_0)} \left[ \text{KL}(q(x_T | x_{T-1}) \| p(x_T)) \right]$ 是一个先验匹配项；当最终潜在分布匹配高斯先验时，它被最小化。这个术语不需要优化，因为它没有可训练的参数；此外，由于我们假设了一个足够大的T，使得最终分布是高斯的，这个术语实际上变成了零。

3. $ \sum_{t=1}^{T-1} \mathbb{E}_{q(x_{t-1}, x_t, x_{t+1} | x_0)} \left[ \text{KL}(q(x_t | x_{t-1}) \| p_{\theta}(x_t | x_{t+1})) \right]$ 是一个一致性项；它努力使 $ x_t$ 的分布在两个方向上保持一致，即从更噪声的图像到更清晰的图像的去噪步骤和从更清晰的图像到更噪声的图像的加噪步骤。这种一致性在数学上通过KL散度反映出来。当两个去噪步骤尽可能匹配时，这个术语被最小化，如公式31中定义的高斯分布 $ q(x_t | x_{t-1})$。

在图4中，我们以视觉方式解释了ELBO的这种解释；对于每个中间的 $ x_t$，我们最小化由粉色和绿色箭头表示的分布之间的差异。在完整的图中，每条粉色箭头也必须从 $ x_0$ 开始，因为它也是一个条件项。

![Deriving the ELBO of a VDM in terms of Consistency terms](https://calvinyluo.com/assets/images/diffusion/first_derivation.webp)

> 可以通过确保对于每个中间潜在变量，其上方潜在变量的后验与其之前的潜在变量的高斯损坏相匹配来优化 VDM。 在此图中，对于每个中间潜在变量，我们最小化粉色和绿色箭头表示的分布之间的差异。

在这种推导下，ELBO的所有项都被计算为期望，因此可以使用蒙特卡洛估计来近似。然而，实际上使用我们刚刚推导出的项来优化ELBO可能是次优的；因为一致性项是作为两个随机变量 $ \{x_{t-1}, x_{t+1}\}$ 的期望来计算的，对于每个时间步，它的蒙特卡洛估计的方差可能会比使用每个时间步一个随机变量估计的项更高。由于它是通过对 $ T-1$ 个一致性项求和来计算的，对于大的 $ T$ 值，ELBO的最终估计值可能具有高方差。

让我们尝试推导出一个ELBO的形式，其中每个项都是对最多一个随机变量的期望。关键的洞察是我们可以重写编码器转换为 $ q(x_t | x_{t-1}) = q(x_t | x_{t-1}, x_0)$，其中额外的条件项由于马尔可夫性质而变得多余。然后，根据贝叶斯规则，我们可以将每个转换重写为：

$$
\begin{align}
q(\boldsymbol{x}_t \mid \boldsymbol{x}_{t-1}, \boldsymbol{x}_0) = \frac{q(\boldsymbol{x}_{t-1}\mid\boldsymbol{x}_t, \boldsymbol{x}_0)q(\boldsymbol{x}_t\mid\boldsymbol{x}_0)}{q(\boldsymbol{x}_{t-1}\mid\boldsymbol{x}_0)}
\end{align}
$$


利用这个新方程，我们可以从公式(37)中的ELBO开始重新推导：

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


因此，我们已经成功地推导出了ELBO的一个解释，它可以被估计为较低的方差，因为每个项都是对最多一个随机变量的期望。这种公式也具有优雅的解释，当检查每个单独的项时，可以揭示出来：

1. $ \mathbb{E}_{q(x_1 | x_0)} \left[ \log p_{\theta}(x_0 | x_1) \right]$ 可以被解释为一个重建项；像VAE的ELBO中的类似项一样，这个项可以使用蒙特卡洛估计来近似和优化。

2. $ \text{KL}(q(x_T | x_0) \| p(x_T))$ 表示最终噪声输入的分布与标准高斯先验有多接近。它没有可训练的参数，并且在我们的假设下也等于零。

3. $ \sum_{t=2}^{T} \mathbb{E}_{q(x_t | x_0)} \left[ \text{KL}(q(x_{t-1} | x_t, x_0) \| p_{\theta}(x_{t-1} | x_t)) \right]$ 是一个去噪匹配项。我们学习期望的去噪转换步骤 $ p_{\theta}(x_{t-1} | x_t)$ 作为可处理的、真实的去噪转换步骤 $ q(x_{t-1} | x_t, x_0)$ 的近似。 $ q(x_{t-1} | x_t, x_0)$ 转换步骤可以作为真实的信号，因为它定义了如何使用最终完全去噪的图像 $ x_0$ 来去噪噪声图像 $ x_t$。因此，当两个去噪步骤尽可能匹配时，这个术语被最小化，如通过它们之间的KL散度来衡量。

作为一个旁注，人们会注意到，在ELBO的推导过程中（公式45和公式58），只使用了马尔可夫假设；因此，这些公式对于任何任意的马尔可夫HVAE都是成立的。此外，当我们设置 $ T = 1$ 时，VDM的两个ELBO解释对于标准VAE的ELBO公式（公式19）是完全相同的。

![Deriving the ELBO of a VDM in terms of Denoising Matching terms](https://calvinyluo.com/assets/images/diffusion/second_derivation.webp)

> VDM 还可以通过学习每个潜在个体的去噪步骤来优化，方法是将其与易于计算的地面实况去噪步骤相匹配。 这再次通过将绿色箭头表示的分布与粉色箭头表示的分布相匹配来直观地表示。 艺术自由在这里发挥作用； 在完整的图片中，每个粉红色箭头也必须来自真实图像，因为它也是一个条件项。

在这种ELBO的推导中，大部分优化成本再次由求和项占据，它支配了重建项。虽然每个KL散度项 $ \text{KL}(q(x_{t-1} | x_t, x_0) \| p_{\theta}(x_{t-1} | x_t))$ 对于任意复杂的马尔可夫HVAEs中的任意后验来说都是难以最小化的，因为同时学习编码器的复杂性增加了，但在VDM中，`我们可以利用`
