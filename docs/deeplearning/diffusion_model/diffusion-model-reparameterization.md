# 重参数：从正态分布到Gumbel Softmax

## 基本概念

**重参数（Reparameterization）**实际上是处理如下期望形式的目标函数的一种技巧：
$$
\begin{equation}L_{\theta}=\mathbb{E}_{z\sim p_{\theta}(z)}[f(z)]\label{eq:base}\end{equation}
$$

这样的目标在VAE中会出现，在文本GAN也会出现，在强化学习中也会出现（$f(z)$对应于奖励函数），所以深究下去，我们会经常碰到这样的目标函数。取决于$z$的连续性，它对应不同的形式：
$$
\begin{equation}\int p_{\theta}(z) f(z)dz\,\,\,\text{(连续情形)}\qquad\qquad \sum_{z} p_{\theta}(z) f(z)\,\,\,\text{(离散情形)}\end{equation}
$$
当然，离散情况下我们更喜欢将记号$z$换成$y$或者$c$。

假设我们需要对 $L_{\theta}$ 求关于 θ 的梯度, 如果函数$f_\theta(z)$  本身关于 $\theta $ 梯度存在，则
$$
\begin{aligned}
\nabla_\theta \mathbb{E}_{p(z)}\left[f_\theta(z)\right] & =\nabla_\theta\left[\int_z p(z) f_\theta(z) d z\right] \\
& =\int_z p(z)\left[\nabla_\theta f_\theta(z)\right] d z \\
& =\mathbb{E}_{p(z)}\left[\nabla_\theta f_\theta(z)\right]
\end{aligned}
$$
实际上，上面的推导过程说明了梯度和期望符号可以交换运算顺序。即，期望的梯度等于梯度的期望。那么问题来了：如果密度函数$p$也有参数 $θ$ 呢?. 我们尽力重复上面的步骤：
$$
\begin{aligned}
\nabla_\theta \mathbb{E}_{p_\theta(z)}\left[f_\theta(z)\right] & =\nabla_\theta\left[\int_z p_\theta(z) f_\theta(z) d z\right] \\
& =\int_z \nabla_\theta\left[p_\theta(z) f_\theta(z)\right] d z \\
& =\int_z f_\theta(z) \nabla_\theta p_\theta(z) d z+\int_z p_\theta(z) \nabla_\theta f_\theta(z) d z \\
& =\underbrace{\int_z f_\theta(z) \nabla_\theta p_\theta(z) d z}_{\text {What about this? }}+\mathbb{E}_{p \theta(z)}\left[\nabla_\theta f_\theta(z)\right]
\end{aligned}
$$


由于第一项并不能写作某一个函数关于 $p_\theta(z)$ 的期望，Monte Carlo方法在这里无法适用——因为其要求我们从 $p_\theta(z)$ 中采样。如果说， $\nabla_\theta p_\theta(z)$ 有解析解，那么一切问题将迎刃而解。但是问题在于，现实生活中$\nabla_\theta p_\theta(z)$ 往往无法写出来。在对于我们所面临的问题有一定认知以后，接下来我们将看看使用重参数化技巧以后，问题会不会有所改善。
$$
\begin{aligned}
\boldsymbol{\epsilon} & \sim p(\boldsymbol{\epsilon}) \\
\mathbf{z} & =g_\theta(\boldsymbol{\epsilon}, \mathbf{x}) \\
\mathbb{E}_{p \theta(\mathbf{z})}\left[f\left(\mathbf{z}^{(i)}\right)\right] & =\mathbb{E}_{p(\boldsymbol{\epsilon})}\left[f\left(g_\theta\left(\boldsymbol{\epsilon}, \mathbf{x}^{(i)}\right)\right)\right] \\
\nabla_\theta \mathbb{E}_{p \theta(\mathbf{z})}\left[f\left(\mathbf{z}^{(i)}\right)\right] & =\nabla_\theta \mathbb{E}_{p(\boldsymbol{\epsilon})}\left[f\left(g_\theta\left(\boldsymbol{\epsilon}, \mathbf{x}^{(i)}\right)\right)\right] \\
& =\mathbb{E}_{p(\boldsymbol{\epsilon})}\left[\nabla_\theta f\left(g_\theta\left(\boldsymbol{\epsilon}, \mathbf{x}^{(i)}\right)\right)\right] \\
& \approx \frac{1}{L} \sum_{l=1}^L \nabla_\theta f\left(g_\theta\left(\epsilon^{(l)}, \mathbf{x}^{(i)}\right)\right)
\end{aligned}
$$
上述推理是理解VAE的关键， 我们使用重新参数化技巧将期望（1）的梯度表示为梯度（5）的期望。如果$g_θ$是可微的，那么我们就可以使用蒙特卡洛方法来估计 $\nabla_\theta \mathbb{E}_{p_{\boldsymbol{\theta}}(\mathbf{z})}\left[f\left(\mathbf{z}^{(i)}\right)\right]$。

为了最小化$L_{\theta}$，我们就需要把$L_{\theta}$明确地写出来，这意味着我们要实现从$p_{\theta}(z)$中采样，而$p_{\theta}(z)$是带有参数$θ$的，如果直接采样的话，那么就失去了$\theta$的信息（梯度），从而无法更新参数$\theta$。而Reparameterization则是提供了这样的一种变换，使得我们可以直接从$p_{\theta}(z)$中采样，并且保留$\theta$的梯度。（注：如果考虑最一般的形式，那么$f(z)$也应该带上参数$\theta$，但这没有增加本质难度。）

## 连续情形

简单起见，我们先考虑连续情形
$$
\begin{equation}L_{\theta}=\int p_{\theta}(z) f(z)dz\label{eq:lianxu}\end{equation}
$$

其中$p_{\theta}(z)$是具有显式概率密度表达式的分布，在变分自编码器中常见的是正态分布$L_{\theta}=\int p_{\theta}(z) f(z)dz$。

### 形式

从式(3)中知道，连续情形的$L_{\theta}$实际上就对应一个积分，所以，为了明确写出$L_{\theta}$，有两种途径：最直接的方式是精确地完成积分(3)，得到显式表达式，但这通常都是不可能的了；所以，唯一的办法是转化为采样形式(1)，并试图在采样过程中保留$\theta$的梯度。

重参数就是这样的一种技巧，它假设从分布$p_{\theta}(z)$中采样可以分解为两个步骤：

(1) 从无参数分布$q(\varepsilon)$中采样一个$\varepsilon$；

(2) 通过变换$z=g_{\theta}(\varepsilon)$ 生成$z$。那么，式(1)就变成了
$$
\begin{equation}\mathbb{E}_{z\sim \mathcal{N}\left(z;\mu_{\theta},\sigma_{\theta}^2\right)}\big[f(z)\big] = \mathbb{E}_{\varepsilon\sim \mathcal{N}\left(\varepsilon;0, 1\right)}\big[f(\varepsilon\times \sigma_{\theta} + \mu_{\theta})\big]\end{equation}
$$

这时候被采样的分布就没有任何参数了，全部被转移到$f$内部了，因此可以采样若干个点，当成普通的loss那样写下来了。

### 例子

一个最简单的例子就是正态分布：对于正态分布来说，重参数就是$\mathcal{N}\left(z;\mu_{\theta},\sigma_{\theta}^2\right)$中采样一个$z$ ”变成“从$\mathcal{N}\left(\varepsilon;0, 1\right)$ 中采样一个$\varepsilon$，然后计算$\varepsilon\times \sigma_{\theta} + \mu_{\theta}$”，所以：
$$
\begin{equation}\mathbb{E}_{z\sim \mathcal{N}\left(z;\mu_{\theta},\sigma_{\theta}^2\right)}\big[f(z)\big] = \mathbb{E}_{\varepsilon\sim \mathcal{N}\left(\varepsilon;0, 1\right)}\big[f(\varepsilon\times \sigma_{\theta} + \mu_{\theta})\big]\end{equation}
$$
如何理解直接采样没有梯度而重参数之后就有梯度呢？其实很简单，比如我说从$\mathcal{N}\left(z;\mu_{\theta},\sigma_{\theta}^2\right)$中采样一个数来，然后你跟我说采样到 5，我完全看不出5跟 $\theta$ 有什么关系（求梯度只能为0）；但是如果先从$\mathcal{N}\left(\varepsilon;0, 1\right)$ 中采样一个数比如0.2，然后计算$0.2 \sigma_{\theta} + \mu_{\theta}$ ，这样我就知道采样出来的结果跟 $\theta$ 的关系了（能求出有效的梯度）。

我们首先生成一个均值为0, 标准差为1的高斯分布 $N\sim (0,1)$。在这个分布中中采样数据 ϵ，经过一个变换后得到目标分布：$Z=\mu + \sigma \epsilon$。μ为均值，σ为方差，由网络计算得出。伪代码如下：

```python
def reparametrization(z_mean, z_log_var):
    epsilon = K.random_normal(shape=K.shape(z_mean))
    return z_mean + z_log_var * epsilon
```

此时，返回的变量就具有了梯度，而采样的过程在整个计算图之外，采样的 $\epsilon$ 就是一个常量，此时便可以梯度下降。

### 回到VAE 中

在VAE中，ELBO(evidence lower-bound)写作：
$$
\begin{aligned}
\operatorname{ELBO}(\boldsymbol{\theta}, \boldsymbol{\phi}) & =\left[\mathbb{E}_{q_\phi(\mathbf{z})}\left[\log p_\theta(\mathbf{x}, \mathbf{z})-\log q_\phi(\mathbf{z} \mid \mathbf{x})\right]\right] \\
\end{aligned}
$$
这里，我们区分了模型参数 $ θ $ 和隐变量(latent variable) ϕ .那么梯度表示为：
$$
\begin{aligned}
\operatorname{ELBO}(\boldsymbol{\theta}, \boldsymbol{\phi}) & =\underbrace{\nabla_{\theta, \phi}\left[\mathbb{E}_{q_\phi(\mathbf{z})}\left[\log p_\theta(\mathbf{x}, \mathbf{z})-\log q_\phi(\mathbf{z} \mid \mathbf{x})\right]\right]}_{\text {Gradient w.r.t. } \phi \text { over expectation w.r.t. } \phi}
\end{aligned}
$$
在我们假设先验和后验估计均为高斯分布的时候，上面的式子可以进一步化简成：
$$
\nabla_{\theta, \phi} \mathcal{L}^B=-\nabla_{\theta, \phi} \overbrace{\left[\mathrm{KL}\left[q_\phi\left(\mathbf{z} \mid \mathbf{x}^{(i)}\right) \| p_\theta(\mathbf{z})\right]\right]}^{\text {Analytically compute this }}+\nabla_{\theta, \phi} \overbrace{\left[\frac{1}{L} \sum_{l=1}^L\left(\log p_\theta\left(\mathbf{x}^{(i)} \mid \mathbf{z}^{(l)}\right)\right)\right]}^{\text {Monte Carlo estimate this }}
$$
从encoder-decoder角度看ELBO的话：
$$
\mathcal{L}^B=-\mathrm{KL}[\overbrace{q_\phi\left(\mathbf{z} \mid \mathbf{x}^{(i)}\right)}^{\text {Encoder }} \overbrace{p_\theta(\mathbf{z})}^{\text {Fixed }}]+\frac{1}{L} \sum_{l=1}^L \log \overbrace{p_\theta\left(\mathbf{x}^{(i)} \mid \mathbf{z}^{(l)}\right)}^{\text {Decoder }}
$$

$$
\begin{array}{rlrl}
& \boldsymbol{\mu}_x, \boldsymbol{\sigma}_x=M(\mathbf{x}), \Sigma(\mathbf{x}) \quad \text { Push } \mathbf{x} \text { through encoder } \\
& \epsilon \sim \mathcal{N}(0,1) \quad \text { Sample noise } \\
& \mathbf{z}=\boldsymbol{\epsilon} \boldsymbol{\sigma}_x+\boldsymbol{\mu}_x \quad \text { Reparameterize } \\
& \mathbf{x}_r=p_\theta(\mathbf{x} \mid \mathbf{z})
 \\
\mathbf{z} & =\boldsymbol{\epsilon} \boldsymbol{\sigma}_x+\boldsymbol{\mu}_x & & \text { Reparameterize } \\
\mathbf{x}_r & =p_\theta(\mathbf{x} \mid \mathbf{z}) & & \text { Push z through decoder } \\
\text { recon. loss } & =\operatorname{MSE}\left(\mathbf{x}, \mathbf{x}_r\right) & & \text { Compute reconstruction loss } \\
\text { var. loss } & =-\operatorname{KL}\left[\mathcal{N}\left(\boldsymbol{\mu}_x, \boldsymbol{\sigma}_x\right) \| \mathcal{N}(0, I)\right] & \text { Compute variational loss } \\
\mathrm{L} & =\text { recon. loss }+\text { var. loss } & & \text { Combine losses }
\end{array}
$$
### 总结

让我们把前面的内容重新整理一下。总的来说，连续情形的重参数还是比较简单的：连续情形下，我们要处理的$L_{\theta}$实际上是式(3)，由于精确的积分我们没有办法显式地写出来，所以需要转化为采样，而为了在采样的过程中得到有效的梯度，我们就需要重参数。

从数学本质来看，重参数是一种积分变换，即原来是关于$z$积分，通过$z=g_{\theta}(\varepsilon)$变换之后得到新的积分形式，
