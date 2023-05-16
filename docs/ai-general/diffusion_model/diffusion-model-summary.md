

# 什么是扩散模型？

扩散模型的灵感来自非平衡热力学。他们定义了一个扩散步骤的马尔可夫链，以缓慢地将随机噪声添加到数据中，然后学习反转扩散过程以从噪声中构建所需的数据样本。与 VAE 或流模型不同，扩散模型是通过固定过程学习的，并且潜在变量具有高维性（与原始数据相同）。

<div align=center>
<img width="600" src="https://lilianweng.github.io/posts/2021-07-11-diffusion-models/generative-overview.png"/>
</div>
<div align=center>图 1. 不同类型生成模型的概述。</div>

# 什么是扩散模型？

已经提出了几种基于扩散的生成模型，这些模型具有相似的思想，包括*扩散概率模型*（[Sohl-Dickstein 等人，2015 年](https://arxiv.org/abs/1503.03585)）、*噪声条件评分网络*（**NCSN**；[Yang 和 Ermon，2019 年](https://arxiv.org/abs/1907.05600)）和*去噪扩散概率模型*（**DDPM**；[Ho 等人，2020 年](https://arxiv.org/abs/2006.11239)）。

## 正向扩散过程

给定一个从真实分布中采样的数据 $\mathbf{x}_0 \sim q(\mathbf{x})$，我们定义一个*前向扩散过程*，在这个过程中我们以$T$步向样本中添加少量高斯噪声，产生一系列噪声样本$\mathbf{x}_1, \dots, \mathbf{x}_T$ , 噪声大小由方差$\{\beta_t \in (0, 1)\}_{t=1}^T$控制 .
$$
q(\mathbf{x}_t \vert \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t\mathbf{I}) \quad
q(\mathbf{x}_{1:T} \vert \mathbf{x}_0) = \prod^T_{t=1} q(\mathbf{x}_t \vert \mathbf{x}_{t-1})
$$
当步骤$t$ 逐渐变大, 数据样本$\mathbf{x}_0$逐渐失去其可区分的特征。最终当$t$→∞, $\mathbf{x}_T$ 等价于各向同性高斯分布。

<div align=center>
<img width="600" src="https://lilianweng.github.io/posts/2021-07-11-diffusion-models/DDPM.png"/>
</div>
<div align=center>图2. 缓慢加入（去除）噪声生成样本的正向（反向）扩散过程的马尔可夫链。（图片来源：[Ho et al. 2020](https://arxiv.org/abs/2006.11239)，带有一些附加注释）</div>

上述过程的一个很好的特性是我们用[Reparameterization Trick](https://lilianweng.github.io/posts/2018-08-12-vae/#reparameterization-trick)以封闭形式在任意时间步 $t$ 都可以采样 $\mathbf{x}_t$。让$\alpha_t = 1 - \beta_t$和 $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$，则:
$$
\begin{aligned}
\mathbf{x}_t
&= \sqrt{\alpha_t}\mathbf{x}_{t-1} + \sqrt{1 - \alpha_t}\boldsymbol{\epsilon}_{t-1} & \text{ ;where } \boldsymbol{\epsilon}_{t-1}, \boldsymbol{\epsilon}_{t-2}, \dots \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \\
&= \sqrt{\alpha_t \alpha_{t-1}} \mathbf{x}_{t-2} + \sqrt{1 - \alpha_t \alpha_{t-1}} \bar{\boldsymbol{\epsilon}}_{t-2} & \text{ ;where } \bar{\boldsymbol{\epsilon}}_{t-2} \text{ merges two Gaussians (*).} \\
&= \dots \\
&= \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon} \\
q(\mathbf{x}_t \vert \mathbf{x}_0) &= \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I})
\end{aligned}
$$
(*) 回想一下，当我们合并两个具有不同方差的高斯分布，$\mathcal{N}(\mathbf{0}, \sigma_1^2\mathbf{I})$ 和 $\mathcal{N}(\mathbf{0}, \sigma_2^2\mathbf{I})$ 时， 得到的新的分布是$\mathcal{N}(\mathbf{0}, (\sigma_1^2 + \sigma_2^2)\mathbf{I})$. 这里的合并标准差是 $\sqrt{(1 - \alpha_t) + \alpha_t (1-\alpha_{t-1})} = \sqrt{1 - \alpha_t\alpha_{t-1}}$.

通常，当样本噪声更大时，可以承受更大的更新步长，所以 $\beta_1 < \beta_2 < \dots < \beta_T$ 因此 $\bar{\alpha}_1 > \dots > \bar{\alpha}_T$.

### Connection with stochastic gradient Langevin dynamics

Langevin dynamics 是物理学的一个概念，是为对分子系统进行统计建模而开发的。结合随机梯度下降，*随机梯度 Langevin 动力学*( [Welling & Teh 2011](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.226.363) ) 可以仅使用马尔可夫更新链中的梯度$\nabla_\mathbf{x} \log p(\mathbf{x})$从概率密度$p(\mathbf{x})$中产生样本：
$$
\mathbf{x}_t = \mathbf{x}_{t-1} + \frac{\delta}{2} \nabla_\mathbf{x} \log q(\mathbf{x}_{t-1}) + \sqrt{\delta} \boldsymbol{\epsilon}_t
,\quad\text{where }
\boldsymbol{\epsilon}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$
其中$\delta$是步长。当 $T \to \infty, \epsilon \to 0$,  $\mathbf{x}_t$ 等于真实概率密度 $p(\mathbf{x})$ .

与标准 SGD 相比，随机梯度 Langevin 动力学将高斯噪声注入参数更新，以避免陷入局部最小值。

## 反向扩散过程

如果说前向过程(forward)是加噪的过程，那么逆向过程(reverse)就是diffusion的去噪推断过程。

如果我们可以反转上述过程并从$q(\mathbf{x}_{t-1} \vert \mathbf{x}_t)$中采样，我们将能够从高斯噪声输入$\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$重建真实样本. 注意，如果$\beta_t$足够小，$q(\mathbf{x}_{t-1} \vert \mathbf{x}_t)$ 也将是高斯分布。不幸的是，我们无法轻易估计 $q(\mathbf{x}_{t-1} \vert \mathbf{x}_t)$ ， 因为它需要用到整个数据集，因此我们需要学习一个模型 $p_\theta$ 近似这些条件概率以运行*反向扩散过程*。
$$
p_\theta(\mathbf{x}_{0:T}) = p(\mathbf{x}_T) \prod^T_{t=1} p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t) \quad
p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))
$$
![img](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/diffusion-example.png)

> Fig. 3. An example of training a diffusion model for modeling a 2D swiss roll data. (Image source: [Sohl-Dickstein et al., 2015](https://arxiv.org/abs/1503.03585))

值得注意的是，以$\mathbf{x}_0$为条件的反向条件概率是容易处理的:
$$
q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; \color{blue}{\tilde{\boldsymbol{\mu}}}(\mathbf{x}_t, \mathbf{x}_0), \color{red}{\tilde{\beta}_t} \mathbf{I})
$$
使用贝叶斯法则，我们有：
$$
\begin{aligned}
q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)
&= q(\mathbf{x}_t \vert \mathbf{x}_{t-1}, \mathbf{x}_0) \frac{ q(\mathbf{x}_{t-1} \vert \mathbf{x}_0) }{ q(\mathbf{x}_t \vert \mathbf{x}_0) } \\
&\propto \exp \Big(-\frac{1}{2} \big(\frac{(\mathbf{x}_t - \sqrt{\alpha_t} \mathbf{x}_{t-1})^2}{\beta_t} + \frac{(\mathbf{x}_{t-1} - \sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_0)^2}{1-\bar{\alpha}_{t-1}} - \frac{(\mathbf{x}_t - \sqrt{\bar{\alpha}_t} \mathbf{x}_0)^2}{1-\bar{\alpha}_t} \big) \Big) \\
&= \exp \Big(-\frac{1}{2} \big(\frac{\mathbf{x}_t^2 - 2\sqrt{\alpha_t} \mathbf{x}_t \color{blue}{\mathbf{x}_{t-1}} \color{black}{+ \alpha_t} \color{red}{\mathbf{x}_{t-1}^2} }{\beta_t} + \frac{ \color{red}{\mathbf{x}_{t-1}^2} \color{black}{- 2 \sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_0} \color{blue}{\mathbf{x}_{t-1}} \color{black}{+ \bar{\alpha}_{t-1} \mathbf{x}_0^2}  }{1-\bar{\alpha}_{t-1}} - \frac{(\mathbf{x}_t - \sqrt{\bar{\alpha}_t} \mathbf{x}_0)^2}{1-\bar{\alpha}_t} \big) \Big) \\
&= \exp\Big( -\frac{1}{2} \big( \color{red}{(\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}})} \mathbf{x}_{t-1}^2 - \color{blue}{(\frac{2\sqrt{\alpha_t}}{\beta_t} \mathbf{x}_t + \frac{2\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_{t-1}} \mathbf{x}_0)} \mathbf{x}_{t-1} \color{black}{ + C(\mathbf{x}_t, \mathbf{x}_0) \big) \Big)}
\end{aligned}
$$
其中 $ C(\mathbf{x}_t, \mathbf{x}_0)$ 是一些不涉及$\mathbf{x}_{t-1}$的函数并且省略了细节。按照标准高斯密度函数，均值和方差可以参数化如下（其中 $\alpha_t = 1 - \beta_t$和 $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$)
$$
\begin{aligned}
\tilde{\beta}_t
&= 1/(\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}})
= 1/(\frac{\alpha_t - \bar{\alpha}_t + \beta_t}{\beta_t(1 - \bar{\alpha}_{t-1})})
= \color{green}{\frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t} \\
\tilde{\boldsymbol{\mu}}_t (\mathbf{x}_t, \mathbf{x}_0)
&= (\frac{\sqrt{\alpha_t}}{\beta_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1} }}{1 - \bar{\alpha}_{t-1}} \mathbf{x}_0)/(\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}}) \\
&= (\frac{\sqrt{\alpha_t}}{\beta_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1} }}{1 - \bar{\alpha}_{t-1}} \mathbf{x}_0) \color{green}{\frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t} \\
&= \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} \mathbf{x}_0\\
\end{aligned}
$$
多亏了 [nice property](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#nice), 我们可以表示 $\mathbf{x}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_t)$并将其代入上述等式，得到：
$$
\begin{aligned}
\tilde{\beta}_t
&= 1/(\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}})
= 1/(\frac{\alpha_t - \bar{\alpha}_t + \beta_t}{\beta_t(1 - \bar{\alpha}_{t-1})})
= \color{green}{\frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t} \\
\tilde{\boldsymbol{\mu}}_t (\mathbf{x}_t, \mathbf{x}_0)
&= (\frac{\sqrt{\alpha_t}}{\beta_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1} }}{1 - \bar{\alpha}_{t-1}} \mathbf{x}_0)/(\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}}) \\
&= (\frac{\sqrt{\alpha_t}}{\beta_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1} }}{1 - \bar{\alpha}_{t-1}} \mathbf{x}_0) \color{green}{\frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t} \\
&= \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} \mathbf{x}_0\\
\end{aligned}
$$
如图 2 所示，这种设置与[VAE](https://lilianweng.github.io/posts/2018-08-12-vae/)非常相似，因此我们可以使用变分下界来优化负对数似然。
$$
\begin{aligned}
- \log p_\theta(\mathbf{x}_0)
&\leq - \log p_\theta(\mathbf{x}_0) + D_\text{KL}(q(\mathbf{x}_{1:T}\vert\mathbf{x}_0) \| p_\theta(\mathbf{x}_{1:T}\vert\mathbf{x}_0) ) \\
&= -\log p_\theta(\mathbf{x}_0) + \mathbb{E}_{\mathbf{x}_{1:T}\sim q(\mathbf{x}_{1:T} \vert \mathbf{x}_0)} \Big[ \log\frac{q(\mathbf{x}_{1:T}\vert\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T}) / p_\theta(\mathbf{x}_0)} \Big] \\
&= -\log p_\theta(\mathbf{x}_0) + \mathbb{E}_q \Big[ \log\frac{q(\mathbf{x}_{1:T}\vert\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} + \log p_\theta(\mathbf{x}_0) \Big] \\
&= \mathbb{E}_q \Big[ \log \frac{q(\mathbf{x}_{1:T}\vert\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} \Big] \\
\text{Let }L_\text{VLB}
&= \mathbb{E}_{q(\mathbf{x}_{0:T})} \Big[ \log \frac{q(\mathbf{x}_{1:T}\vert\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} \Big] \geq - \mathbb{E}_{q(\mathbf{x}_0)} \log p_\theta(\mathbf{x}_0)
\end{aligned}
$$
使用 Jensen 不等式也可以直接得到相同的结果。假设我们以最小化交叉熵作为学习目标，
$$
\begin{aligned}
L_\text{CE}
&= - \mathbb{E}_{q(\mathbf{x}_0)} \log p_\theta(\mathbf{x}_0) \\
&= - \mathbb{E}_{q(\mathbf{x}_0)} \log \Big( \int p_\theta(\mathbf{x}_{0:T}) d\mathbf{x}_{1:T} \Big) \\
&= - \mathbb{E}_{q(\mathbf{x}_0)} \log \Big( \int q(\mathbf{x}_{1:T} \vert \mathbf{x}_0) \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} \vert \mathbf{x}_{0})} d\mathbf{x}_{1:T} \Big) \\
&= - \mathbb{E}_{q(\mathbf{x}_0)} \log \Big( \mathbb{E}_{q(\mathbf{x}_{1:T} \vert \mathbf{x}_0)} \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} \vert \mathbf{x}_{0})} \Big) \\
&\leq - \mathbb{E}_{q(\mathbf{x}_{0:T})} \log \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} \vert \mathbf{x}_{0})} \\
&= \mathbb{E}_{q(\mathbf{x}_{0:T})}\Big[\log \frac{q(\mathbf{x}_{1:T} \vert \mathbf{x}_{0})}{p_\theta(\mathbf{x}_{0:T})} \Big] = L_\text{VLB}
\end{aligned}
$$
为了将方程中的每一项转换为可分析计算的，目标可以进一步重写为几个 KL 散度和熵项的组合（参见[Sohl-Dickstein 等人](https://arxiv.org/abs/1503.03585)的附录 B 中详细的推导过程）
$$
\begin{aligned}
L_\text{VLB}
&= \mathbb{E}_{q(\mathbf{x}_{0:T})} \Big[ \log\frac{q(\mathbf{x}_{1:T}\vert\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} \Big] \\
&= \mathbb{E}_q \Big[ \log\frac{\prod_{t=1}^T q(\mathbf{x}_t\vert\mathbf{x}_{t-1})}{ p_\theta(\mathbf{x}_T) \prod_{t=1}^T p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t) } \Big] \\
&= \mathbb{E}_q \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=1}^T \log \frac{q(\mathbf{x}_t\vert\mathbf{x}_{t-1})}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} \Big] \\
&= \mathbb{E}_q \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=2}^T \log \frac{q(\mathbf{x}_t\vert\mathbf{x}_{t-1})}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} + \log\frac{q(\mathbf{x}_1 \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)} \Big] \\
&= \mathbb{E}_q \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=2}^T \log \Big( \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)}\cdot \frac{q(\mathbf{x}_t \vert \mathbf{x}_0)}{q(\mathbf{x}_{t-1}\vert\mathbf{x}_0)} \Big) + \log \frac{q(\mathbf{x}_1 \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)} \Big] \\
&= \mathbb{E}_q \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=2}^T \log \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} + \sum_{t=2}^T \log \frac{q(\mathbf{x}_t \vert \mathbf{x}_0)}{q(\mathbf{x}_{t-1} \vert \mathbf{x}_0)} + \log\frac{q(\mathbf{x}_1 \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)} \Big] \\
&= \mathbb{E}_q \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=2}^T \log \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} + \log\frac{q(\mathbf{x}_T \vert \mathbf{x}_0)}{q(\mathbf{x}_1 \vert \mathbf{x}_0)} + \log \frac{q(\mathbf{x}_1 \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)} \Big]\\
&= \mathbb{E}_q \Big[ \log\frac{q(\mathbf{x}_T \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_T)} + \sum_{t=2}^T \log \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} - \log p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1) \Big] \\
&= \mathbb{E}_q [\underbrace{D_\text{KL}(q(\mathbf{x}_T \vert \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_T))}_{L_T} + \sum_{t=2}^T \underbrace{D_\text{KL}(q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t))}_{L_{t-1}} \underbrace{- \log p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)}_{L_0} ]
\end{aligned}
$$
我们分别标记变分下界损失中的每个组件：
$$
\begin{aligned}
L_\text{VLB} &= L_T + L_{T-1} + \dots + L_0 \\
\text{where } L_T &= D_\text{KL}(q(\mathbf{x}_T \vert \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_T)) \\
L_t &= D_\text{KL}(q(\mathbf{x}_t \vert \mathbf{x}_{t+1}, \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_t \vert\mathbf{x}_{t+1})) \text{ for }1 \leq t \leq T-1 \\
L_0 &= - \log p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)
\end{aligned}
$$
$L_\text{VLB}$ 中的每个 KL 项（除了$L_0$) 比较了两个高斯分布，因此可以用[封闭形式](https://en.wikipedia.org/wiki/Kullback–Leibler_divergence#Multivariate_normal_distributions)计算它们。$L_T$是常数，在训练期间可以忽略，因为$q$没有可学习的参数并且 $\mathbf{x}_t$ 是高斯噪声。[Ho et al. 2020](https://arxiv.org/abs/2006.11239) 使用从$\mathcal{N}(\mathbf{x}_0; \boldsymbol{\mu}_\theta(\mathbf{x}_1, 1), \boldsymbol{\Sigma}_\theta(\mathbf{x}_1, 1))$ 派生的单独的离散解码器建模了 $L_0$。

## Parameterization of  $L_t$ for Training Loss

回想一下，我们需要学习一个神经网络来逼近反向扩散过程中的条件概率分布，$p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))$. 我们想训练$\boldsymbol{\mu}_\theta$来预测 $\tilde{\boldsymbol{\mu}}_t = \frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_t \Big)$. 因为$\mathbf{x}_t$在训练时可用作输入，在时间步$t$，我们可以重新参数化高斯噪声项，而不是从输入$\mathbf{x}_t$来预测$\boldsymbol{\epsilon}_t$:
$$
\begin{aligned}
\boldsymbol{\mu}_\theta(\mathbf{x}_t, t) &= \color{cyan}{\frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \Big)} \\
\text{Thus }\mathbf{x}_{t-1} &= \mathcal{N}(\mathbf{x}_{t-1}; \frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \Big), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))
\end{aligned}
$$
损失项 $L_t$ 被参数化以最小化与$\tilde{\boldsymbol{\mu}}$ 的差异:
$$
\begin{aligned}
L_t
&= \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}} \Big[\frac{1}{2 \| \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t) \|^2_2} \| \color{blue}{\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0)} - \color{green}{\boldsymbol{\mu}_\theta(\mathbf{x}_t, t)} \|^2 \Big] \\
&= \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}} \Big[\frac{1}{2  \|\boldsymbol{\Sigma}_\theta \|^2_2} \| \color{blue}{\frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_t \Big)} - \color{green}{\frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t) \Big)} \|^2 \Big] \\
&= \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}} \Big[\frac{ (1 - \alpha_t)^2 }{2 \alpha_t (1 - \bar{\alpha}_t) \| \boldsymbol{\Sigma}_\theta \|^2_2} \|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2 \Big] \\
&= \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}} \Big[\frac{ (1 - \alpha_t)^2 }{2 \alpha_t (1 - \bar{\alpha}_t) \| \boldsymbol{\Sigma}_\theta \|^2_2} \|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_t, t)\|^2 \Big]
\end{aligned}
$$

## 简化

根据经验，[Ho et al. (2020)](https://arxiv.org/abs/2006.11239) 发现在忽略权重项的简化目标下训练扩散模型效果更好：
$$
\begin{aligned}
L_t^\text{simple}
&= \mathbb{E}_{t \sim [1, T], \mathbf{x}_0, \boldsymbol{\epsilon}_t} \Big[\|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2 \Big] \\
&= \mathbb{E}_{t \sim [1, T], \mathbf{x}_0, \boldsymbol{\epsilon}_t} \Big[\|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_t, t)\|^2 \Big]
\end{aligned}
$$
最终的简单目标是：
$$
\begin{aligned}
L_t^\text{simple}
&= \mathbb{E}_{t \sim [1, T], \mathbf{x}_0, \boldsymbol{\epsilon}_t} \Big[\|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2 \Big] \\
&= \mathbb{E}_{t \sim [1, T], \mathbf{x}_0, \boldsymbol{\epsilon}_t} \Big[\|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_t, t)\|^2 \Big]
\end{aligned}
$$
其中$C$是一个常数，不依赖于$\theta$.

![img](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/DDPM-algo.png)

>  图 4. DDPM 中的训练和采样算法（图片来源：[Ho et al. 2020](https://arxiv.org/abs/2006.11239)）

### Connection with noise-conditioned score networks (NCSN)

[Song & Ermon (2019)](https://arxiv.org/abs/1907.05600)提出了一种基于分数的生成建模方法，其中使用分数匹配估计的数据分布梯度，通过[Langevin 动力学生成样本。](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#connection-with-stochastic-gradient-langevin-dynamics)每个样本 $\mathbf{x}$ 的概率密度分数定义为其梯度 $\nabla_{\mathbf{x}} \log q(\mathbf{x})$. 评分网网络$\mathbf{s}_\theta: \mathbb{R}^D \to \mathbb{R}^D$被训练来估计它： $\mathbf{s}_\theta(\mathbf{x}) \approx \nabla_{\mathbf{x}} \log q(\mathbf{x})$.

为了使其在深度学习环境中可扩展到高维数据，他们建议使用*去噪分数匹配*( [Vincent, 2011](http://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf) ) 或*切片分数匹配*（使用随机投影；[Song 等人，2019](https://arxiv.org/abs/1905.07088)）。Denosing score matching 在数据中添加了预先指定的小的噪声并通过分数匹配的方法估计$q(\tilde{\mathbf{x}} \vert \mathbf{x})$。

回想一下， Langevin 动力学可以在迭代过程中仅使用分数$\nabla_{\mathbf{x}} \log q(\mathbf{x})$从概率密度分布中采样数据。

然而，根据流形假设，大多数数据预计会集中在低维流形中，即使观察到的数据可能看起来是任意高维。由于数据点无法覆盖整个空间，因此会对分数估计产生负面影响。在数据密度低的区域，分数估计不太可靠。加入小的高斯噪声后，使得扰动后的数据分布覆盖全空间$\mathbb{R}^D$，分数估计网络的训练变得更加稳定。[Song & Ermon (2019)](https://arxiv.org/abs/1907.05600)通过用*不同级别*的噪声扰动数据对其进行了改进，并训练了一个噪声条件评分网络来*联合*估计所有扰动数据在不同噪声级别下的分数。

增加噪声水平的时间表类似于前向扩散过程。如果我们使用扩散过程注释，分数近似于 $\mathbf{s}_\theta(\mathbf{x}_t, t) \approx \nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t)$. 给定高斯分布 $\mathbf{x} \sim \mathcal{N}(\mathbf{\mu}, \sigma^2 \mathbf{I})$，我们可以将其密度函数对数的导数写为$\nabla_{\mathbf{x}}\log p(\mathbf{x}) = \nabla_{\mathbf{x}} \Big(-\frac{1}{2\sigma^2}(\mathbf{x} - \boldsymbol{\mu})^2 \Big) = - \frac{\mathbf{x} - \boldsymbol{\mu}}{\sigma^2} = - \frac{\boldsymbol{\epsilon}}{\sigma}$. 其中$\boldsymbol{\epsilon} \sim \mathcal{N}(\boldsymbol{0}, \mathbf{I})$. [回想](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#nice)一下 $q(\mathbf{x}_t \vert \mathbf{x}_0) \sim \mathcal{N}(\sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I})$ 因此，
$$
\mathbf{s}_\theta(\mathbf{x}_t, t)
\approx \nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t)
= \mathbb{E}_{q(\mathbf{x}_0)} [\nabla_{\mathbf{x}_t} q(\mathbf{x}_t \vert \mathbf{x}_0)]
= \mathbb{E}_{q(\mathbf{x}_0)} \Big[ - \frac{\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{1 - \bar{\alpha}_t}} \Big]
= - \frac{\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{1 - \bar{\alpha}_t}}
$$

## Parameterization of $\beta_t$

[在Ho 等人(2020)](https://arxiv.org/abs/2006.11239) 的工作中， 前向方差被设置为一系列线性增加的常数, 从$\beta_1=10^{-4}$ 到 $\beta_T=0.02$ . 与归一化的图像像素值在$[-1, 1]$之间的范围相比，它们相对较小。 他们实验中的扩散模型显示了高质量的样本，但仍然无法像其他生成模型那样达到有竞争力的模型对数似然。

[Nichol & Dhariwal (2021)](https://arxiv.org/abs/2102.09672)提出了几种改进技术来帮助扩散模型获得更低的 NLL。其中一项改进是使用基于余弦的方差表。调度函数的选择可以是任意的，只要它在训练过程中提供近乎线性下降, 在$t$=0和$t$=$T$处有细微变化即可。
$$
\beta_t = \text{clip}(1-\frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}, 0.999) \quad\bar{\alpha}_t = \frac{f(t)}{f(0)}\quad\text{where }f(t)=\cos\Big(\frac{t/T+s}{1+s}\cdot\frac{\pi}{2}\Big)
$$
其中小偏移量$s$是为了防止当$t=0$时, $\beta_t$太小.

![img](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/diffusion-beta.png)

>  Fig. 5. Comparison of linear and cosine-based scheduling of $\beta\_t$ during training. (Image source: [Nichol & Dhariwal, 2021](https://arxiv.org/abs/2102.09672))

## Parameterization of reverse process variance $ \boldsymbol{\Sigma}_\theta$

[Ho et al. (2020)](https://arxiv.org/abs/2006.11239)  将 $\beta_t$ 作为常量而不是让它们成为可学习的参数， 并设置$\boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t) = \sigma^2_t \mathbf{I}$，其中$\sigma_t$ 是未学习的设置为 $\beta_t$ 或者 $\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t$，  因为他们发现学习对角方差$\boldsymbol{\Sigma}_\theta$会导致训练不稳定和较差的样本质量。

[Nichol & Dhariwal (2021)](https://arxiv.org/abs/2102.09672)建议通过模型预测混合向量$\mathbf{v}$ 来学习$\boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t)$， 作为 $\beta_t$和 $\tilde{\beta}_t$之间的插值:
$$
\boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t) = \exp(\mathbf{v} \log \beta_t + (1-\mathbf{v}) \log \tilde{\beta}_t)
$$
然而，简单的目标$L_\text{simple}$不依赖于$\boldsymbol{\Sigma}_\theta$. 为了增加依赖性，他们构建了一个混合目标$L_\text{hybrid} = L_\text{simple} + \lambda L_\text{VLB}$,  其中λ=0.001很小， 并且在$L_\text{VLB}$ 项中的 $\boldsymbol{\mu}_\theta$上停止梯度， 使得$L_\text{VLB}$只指导$\boldsymbol{\Sigma}_\theta$的学习. 根据经验，他们观察到$L_\text{VLB}$可能由于嘈杂的梯度而难以优化，因此他们建议使用具有重要性抽样的$L_\text{VLB}$的时间平滑版本。

![img](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/improved-DDPM-nll.png)

> 图 6. 改进的 DDPM 与其他基于似然的生成模型的负对数似然比较。NLL 以 bits/dim 为单位报告。（图片来源：[Nichol & Dhariwal，2021 年](https://arxiv.org/abs/2102.09672))

# Speed up Diffusion Model Sampling

通过遵循反向扩散过程的马尔可夫链从 DDPM 生成样本速度非常慢，因为$T$ 可以高达几千步。[来自Song 等人,2020](https://arxiv.org/abs/2010.02502) 的一个数据。“例如，从 DDPM 采样 50k 大小为 32 × 32 的图像需要大约 20 小时，但在 Nvidia 2080 Ti GPU 上从 GAN 采样不到一分钟。”

一种简单的方法是运行跨步采样计划（[Nichol & Dhariwal，2021](https://arxiv.org/abs/2102.09672)），每隔$\lceil T/S \rceil$运行一次采样，流程的步骤从$T$步减少到 $S$步。新的抽样时间表$\{\tau_1, \dots, \tau_S\}$ 其中 $\tau_1 < \tau_2 < \dots <\tau_S \in [1, T]$并且$S < T $.

对于另一种方法，让我们根据[nice property](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#nice)所需的标准偏差$\sigma_t$，以参数化的方式重写$q_\sigma(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)$：
$$
\begin{aligned}
\mathbf{x}_{t-1}
&= \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0 +  \sqrt{1 - \bar{\alpha}_{t-1}}\boldsymbol{\epsilon}_{t-1} \\
&= \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \boldsymbol{\epsilon}_t + \sigma_t\boldsymbol{\epsilon} \\
&= \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \frac{\mathbf{x}_t - \sqrt{\bar{\alpha}_t}\mathbf{x}_0}{\sqrt{1 - \bar{\alpha}_t}} + \sigma_t\boldsymbol{\epsilon} \\
q_\sigma(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)
&= \mathcal{N}(\mathbf{x}_{t-1}; \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \frac{\mathbf{x}_t - \sqrt{\bar{\alpha}_t}\mathbf{x}_0}{\sqrt{1 - \bar{\alpha}_t}}, \sigma_t^2 \mathbf{I})
\end{aligned}
$$
回想一下，$q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; \tilde{\boldsymbol{\mu}}(\mathbf{x}_t, \mathbf{x}_0), \tilde{\beta}_t \mathbf{I})$，因此我们有：
$$
\tilde{\beta}_t = \sigma_t^2 = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t
$$
让$\sigma_t^2 = \eta \cdot \tilde{\beta}_t$,  这样我们就可以调整 $\eta \in \mathbb{R}^+ $作为控制采样随机性的超参数。特例$\eta = 0$ 使采样过程具有*确定性*。这样的模型被命名为*去噪扩散隐式模型*（**DDIM**；[Song et al., 2020](https://arxiv.org/abs/2010.02502)）。DDIM 具有相同的边缘噪声分布，但确定性地将噪声映射回原始数据样本。

在生成过程中，我们只采样了扩散步骤的一个子集$S$： $\{\tau_1, \dots, \tau_S\}$， 推理过程变为：
$$
q_{\sigma, \tau}(\mathbf{x}_{\tau_{i-1}} \vert \mathbf{x}_{\tau_t}, \mathbf{x}_0)
= \mathcal{N}(\mathbf{x}_{\tau_{i-1}}; \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \frac{\mathbf{x}_{\tau_i} - \sqrt{\bar{\alpha}_t}\mathbf{x}_0}{\sqrt{1 - \bar{\alpha}_t}}, \sigma_t^2 \mathbf{I})
$$
虽然，在实验中，所有模型都经过$T=1000$ 扩散步骤的训练， 他们观察到 DDIM ($\eta=0$) 时， 当$S$ 很小的时候，可以生产出质量最好的样品，而 DDPM ($\eta=1$) 在small 的 $S$上表现更差。当我们有能力运行完整的反向马尔可夫扩散步骤时，DDPM 确实表现更好（$S$=$T$=1000). 使用 DDIM，可以将扩散模型训练到任意数量的前向步，但只能从生成过程中的步骤子集中进行采样。

![img](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/DDIM-results.png)

> 图 7. 不同设置的扩散模型在 CIFAR10 和 CelebA 数据集上的 FID 分数，包括DDIM($\eta=0$） 和DDPM($\hat{\sigma}$). （图片来源：[Song et al., 2020](https://arxiv.org/abs/2010.02502)）

与 DDPM 相比，DDIM 能够：

1. 使用更少的步骤生成更高质量的样本。
2. 具有“一致性”属性，因为生成过程是确定性的，这意味着以相同潜变量为条件的多个样本应该具有相似的高级特征。
3. 由于一致性，DDIM 可以在潜在变量中进行语义上有意义的插值。

*Latent diffusion model* ( **LDM** ; [Rombach & Blattmann, et al. 2022](https://arxiv.org/abs/2112.10752) ) 在隐空间而不是在像素空间中运行扩散过程，使训练成本更低，推理速度更快。它的动机是观察到图像的大部分 bits 都有助于感知细节，并且语义和概念组成在积极压缩后仍然存在。LDM 通过生成建模学习松散地分解感知压缩和语义压缩，方法是首先使用自动编码器去除像素级冗余，然后通过扩散过程对学习的潜在数据进行操作/生成语义概念。

![img](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/image-distortion-rate.png)

>  Fig. 8. The plot for tradeoff between compression rate and distortion, illustrating two-stage compressions - perceptural and semantic comparession. (Image source: [Rombach & Blattmann, et al. 2022](https://arxiv.org/abs/2112.10752))

感知压缩过程依赖于自动编码器模型。编码器$\mathcal{E}$用于压缩输入图像$\mathbf{x} \in \mathbb{R}^{H \times W \times 3}$到更小的 2D 隐变量 $\mathbf{z} = \mathcal{E}(\mathbf{x}) \in \mathbb{R}^{h \times w \times c}$，其中下采样率$f=H/h=W/w=2^m, m \in \mathbb{N}$. 然后是解码器$\mathcal{D}$从隐变量重建图像，$\tilde{\mathbf{x}} = \mathcal{D}(\mathbf{z})$. 该论文探讨了自动编码器训练中的两种正则化，以避免隐空间中的任意高方差。

- KL-reg：对学习隐变量的标准正态分布的KL 惩罚，类似于[VAE](https://lilianweng.github.io/posts/2018-08-12-vae/)。
- VQ-reg：在解码器中使用矢量量化层，如[VQVAE](https://lilianweng.github.io/posts/2018-08-12-vae/#vq-vae-and-vq-vae-2)，但量化层被解码器吸收。

扩散和去噪过程发生在隐变量$z$上. 去噪模型是一个时间条件的 U-Net，增加了交叉注意机制来处理图像生成的灵活条件信息（例如类标签、语义图、图像的模糊变体）。该设计相当于将不同模态的表示融合到具有交叉注意力机制的模型中。每种类型的条件信息都与特定领域的编码器$\tau_\theta$配对， 将条件输入$y$投射到交叉注意力组件的中间表示，$\tau_\theta(y) \in \mathbb{R}^{M \times d_\tau}$:
$$
\begin{aligned}
&\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\Big(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d}}\Big) \cdot \mathbf{V} \\
&\text{where }\mathbf{Q} = \mathbf{W}^{(i)}_Q \cdot \varphi_i(\mathbf{z}_i),\;
\mathbf{K} = \mathbf{W}^{(i)}_K \cdot \tau_\theta(y),\;
\mathbf{V} = \mathbf{W}^{(i)}_V \cdot \tau_\theta(y) \\
&\text{and }
\mathbf{W}^{(i)}_Q \in \mathbb{R}^{d \times d^i_\epsilon},\;
\mathbf{W}^{(i)}_K, \mathbf{W}^{(i)}_V \in \mathbb{R}^{d \times d_\tau},\;
\varphi_i(\mathbf{z}_i) \in \mathbb{R}^{N \times d^i_\epsilon},\;
\tau_\theta(y) \in \mathbb{R}^{M \times d_\tau}
\end{aligned}
$$
![img](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/latent-diffusion-arch.png)

> Fig. 9. The architecture of latent diffusion model. (Image source: [Rombach & Blattmann, et al. 2022](https://arxiv.org/abs/2112.1075))

# Conditioned Generation

使用条件信息在图像上训练生成模型时，例如： ImageNet 数据集,  通常会生成以类标签或一段描述性文本为条件的样本。

## Classifier Guided Diffusion

为了明确地将类别信息纳入扩散过程，[Dhariwal 和 Nichol (2021)](https://arxiv.org/abs/2105.05233)在噪声图像上$\mathbf{x}_t$训练了一个分类器$f_\phi(y \vert \mathbf{x}_t, t)$， 并使用梯度 $\nabla_\mathbf{x} \log f_\phi(y \vert \mathbf{x}_t) $ 通过改变噪声预测，引导扩散采样过程朝向条件信息 $y$（例如目标类别标签）。 [回想](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#score)一下 $\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t) = - \frac{1}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ 我们可以写出联合分布的得分函数$q(\mathbf{x}_t, y)$ 如下:
$$
\begin{aligned}
\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t, y)
&= \nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t) + \nabla_{\mathbf{x}_t} \log q(y \vert \mathbf{x}_t) \\
&\approx - \frac{1}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) + \nabla_{\mathbf{x}_t} \log f_\phi(y \vert \mathbf{x}_t) \\
&= - \frac{1}{\sqrt{1 - \bar{\alpha}_t}} (\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) - \sqrt{1 - \bar{\alpha}_t} \nabla_{\mathbf{x}_t} \log f_\phi(y \vert \mathbf{x}_t))
\end{aligned}
$$
因此，一个新的分类器引导的预测器 $\bar{\boldsymbol{\epsilon}}_\theta$ 将采用以下形式，
$$
\bar{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t) = \boldsymbol{\epsilon}_\theta(x_t, t) - \sqrt{1 - \bar{\alpha}_t} \nabla_{\mathbf{x}_t} \log f_\phi(y \vert \mathbf{x}_t)
$$
为了控制分类器引导的强度，我们可以在delta部分增加一个权重$w$，
$$
\bar{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t) = \boldsymbol{\epsilon}_\theta(x_t, t) - \sqrt{1 - \bar{\alpha}_t} \nabla_{\mathbf{x}_t} \log f_\phi(y \vert \mathbf{x}_t)
$$
由此产生的*消融扩散模型*( **ADM** ) 和带有附加分类器指导的模型 ( **ADM-G** ) 能够取得比 SOTA 生成模型（例如 BigGAN）更好的结果。

![img](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/conditioned-DDPM.png)

> 图 10. 该算法使用来自分类器的指导，使用 DDPM 和 DDIM 运行条件生成。（图片来源： [Dhariwal & Nichol，2021 年](https://arxiv.org/abs/2105.05233)]）

此外，通过对 U-Net 架构进行一些修改，[Dhariwal 和 Nichol (2021)](https://arxiv.org/abs/2105.05233)的性能优于使用扩散模型的 GAN。架构修改包括更大的模型深度/宽度、更多注意力头、多分辨率注意力、用于上/下采样的 BigGAN 残差块、通过$1/\sqrt{2}$ 缩放的残差连接和自适应组归一化（AdaGN）。

## Classifier-Free Guidance

没有独立的分类器$f_\phi$，仍然可以通过合并来自条件和无条件扩散模型的分数来运行条件扩散步骤（[Ho & Salimans，2021](https://openreview.net/forum?id=qw8AKxfYbI)）。让无条件去噪扩散模型$p_\theta(\mathbf{x})$通过分数估计器$\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$和条件模型$p_\theta(\mathbf{x} \vert y)$通过$\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, y)$ 进行参数化 . 这两个模型可以通过单个神经网络学习。准确地说，条件扩散模型$p_\theta(\mathbf{x} \vert y)$在配对数据$(\mathbf{x}, y)$上训练，其中条件信息 $y$ 定期地随机丢弃，这样模型知道怎样无条件地生成图像，即 $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, y=\varnothing)$.

隐式分类器的梯度可以用条件和无条件分数估计器表示。一旦插入到分类器-引导的修改分数中，该分数就不会依赖于单独的分类器。
$$
\begin{aligned}
\nabla_{\mathbf{x}_t} \log p(y \vert \mathbf{x}_t)
&= \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t \vert y) - \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t) \\
&= - \frac{1}{\sqrt{1 - \bar{\alpha}_t}}\Big( \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, y) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \Big) \\
\bar{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t, y)
&= \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, y) - \sqrt{1 - \bar{\alpha}_t} \; w \nabla_{\mathbf{x}_t} \log p(y \vert \mathbf{x}_t) \\
&= \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, y) + w \big(\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, y) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \big) \\
&= (w+1) \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, y) - w \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)
\end{aligned}
$$
他们的实验表明，无分类器指导可以在 FID（区分合成图像和生成图像）和 IS（质量和多样性）之间取得良好的平衡。

引导扩散模型 GLIDE ( [Nichol, Dhariwal & Ramesh, et al. 2022](https://arxiv.org/abs/2112.10741) ) 探索了引导策略、CLIP 引导和无分类器引导，发现后者更受欢迎。他们假设这是因为 CLIP 指导利用具有对抗性示例的模型来对抗 CLIP 模型，而不是优化匹配更好的图像生成。

# Scale up Generation Resolution and Quality

为了生成高分辨率高质量图像，[Ho 等人。(2021)](https://arxiv.org/abs/2106.15282)提议在增加的分辨率下使用多个扩散模型的Pileline。*Pipeline 模型之间的噪声调节增强*对最终图像质量至关重要，即对每个超分辨率模型$p_\theta(\mathbf{x} \vert \mathbf{z})$的调节输入 $z$ 应用强大数据增强,. 条件噪声有助于减少Pipeline设置中的复合误差。*U-net*是用于高分辨率图像生成的扩散建模中模型架构的常见选择。

![img](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/cascaded-diffusion.png)

> 图 11. 分辨率不断增加的多个扩散模型的级联Pipeline。（图片来源： [Ho et al. 2021](https://arxiv.org/abs/2106.15282) ]）

他们发现最有效的噪声是在低分辨率下应用高斯噪声，在高分辨率下应用高斯模糊。此外，他们还探索了两种形式的条件增强，需要对训练过程进行小幅修改。请注意，条件噪声仅适用于训练，不适用于推理。

- 对于低分辨率，截断条件增强在步骤早期 $t> 0$ 停止扩散过程。
- 非截断条件增强运行完整的低分辨率反向过程，直到步骤 0，然后通过$\mathbf{z}_t \sim q(\mathbf{x}_t \vert \mathbf{x}_0)$将其损坏，然后将损坏的$\mathbf{z}_t$输入超分辨率模型 。

两阶段扩散模型**unCLIP** ( [Ramesh et al. 2022](https://arxiv.org/abs/2204.06125) ) 大量利用 CLIP 文本编码器来生成高质量的文本来引导图像。给定一个预训练的 CLIP 模型$C$和来自扩散模型的配对训练数据$(\mathbf{x}, y)$， 其中$x$是一个图像，$y$ 是相应的标题，我们可以分别计算 CLIP 文本和图像嵌入，$\mathbf{c}^t(y)$和 $\mathbf{c}^i(\mathbf{x})$。unCLIP 并行学习两个模型：

- Piror 模型$P(\mathbf{c}^i \vert y)$: 给定文本$y$, 输出 CLIP 图像嵌入$\mathbf{c}^i$.
- 解码器$P(\mathbf{x} \vert \mathbf{c}^i, [y])$: 给定CLIP 图像嵌入$$\mathbf{c}^i$$和可选的原始文本$y$, 生成图像$x$.

这两个模型应用条件生成，因为
$$
\underbrace{P(\mathbf{x} \vert y) = P(\mathbf{x}, \mathbf{c}^i \vert y)}_{\mathbf{c}^i\text{ is deterministic given }\mathbf{x}} = P(\mathbf{x} \vert \mathbf{c}^i, y)P(\mathbf{c}^i \vert y)
$$
![img](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/unCLIP.png)

> 图 12. unCLIP 的架构。（图片来源： [Ramesh et al. 2022](https://arxiv.org/abs/2204.06125) ]）

unCLIP 遵循两阶段图像生成过程：

1. 给定一段文字 $y$，首先使用 CLIP 模型生成文本嵌入$\mathbf{c}^t(y)$. 使用 CLIP 隐空间可以通过文本实现 zero-shot 图像操作。
2. 扩散或自回归 prior $P(\mathbf{c}^i \vert y)$ 处理此 CLIP 文本嵌入以先构建图像，然后以先验条件为条件，构建扩散解码器$P(\mathbf{x} \vert \mathbf{c}^i, [y])$生成图像。该解码器还可以根据图像输入生成图像变体，同时保留其风格和语义。

**Imagen** ( [Saharia et al. 2022](https://arxiv.org/abs/2205.11487) ) 使用预训练的大型语言模型 LM（即冻结的 T5-XXL 文本编码器）来编码文本以生成图像，而没有使用 CLIP 模型。大模型可以带来更好的图像质量和文本图像对齐的普遍趋势。他们发现 T5-XXL 和 CLIP 文本编码器在 MS-COCO 上实现了相似的性能，但人类评估更喜欢 DrawBench（涵盖 11 个类别的提示集合）上的 T5-XXL。

当应用无分类器指导时，增加$w$可能会得到更好的图像文本对齐但导致更差的图像保真度。他们发现这是由于训练-测试不匹配，也就是说，因为训练数据$x$保持在$[-1, 1]$范围内，测试数据也应该如此。引入了两种阈值策略：

- 静态阈值：裁剪 $x$ 预测到 $[-1, 1]$
- 动态阈值：在每个采样步骤中，计算 $s$ 作为某个百分位数的绝对像素值；如果$s> 1$, 将预测裁剪到 $[−s,s]$ 并除以$s$.

Imagen 修改了 U-net 中的几个设计，使其成为*高效的 U-Net*。

- 通过为较低分辨率添加更多 residual locks ，将模型参数从高分辨率块转移到低分辨率；
- 缩放跳过连接$1/\sqrt{2}$
- 颠倒下采样（移动到卷积之前）和上采样操作（移动到卷积之后）的顺序，以提高前向传播的速度。

他们发现噪声调节增强、动态阈值化和高效 U-Net 对图像质量至关重要，但缩放文本编码器大小比 U-Net 大小更重要。

# 快速总结

- **优点**：易处理性和灵活性是生成建模中两个相互冲突的目标。易处理的模型可以进行分析评估并廉价地拟合数据（例如通过高斯或拉普拉斯），但它们不能轻易地描述丰富数据集中的结构。灵活的模型可以适应数据中的任意结构，但评估、训练或从这些模型中抽样通常是昂贵的。扩散模型既易于分析又灵活。
- **缺点**：扩散模型依赖于扩散步骤的长马尔可夫链来生成样本，因此在时间和计算方面可能非常昂贵。已经提出了新的方法来使过程更快，但采样仍然比 GAN 慢。

# References

[1] Jascha Sohl-Dickstein et al. [“Deep Unsupervised Learning using Nonequilibrium Thermodynamics.”](https://arxiv.org/abs/1503.03585) ICML 2015.

[2] Max Welling & Yee Whye Teh. [“Bayesian learning via stochastic gradient langevin dynamics.”](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.226.363) ICML 2011.

[3] Yang Song & Stefano Ermon. [“Generative modeling by estimating gradients of the data distribution.”](https://arxiv.org/abs/1907.05600) NeurIPS 2019.

[4] Yang Song & Stefano Ermon. [“Improved techniques for training score-based generative models.”](https://arxiv.org/abs/2006.09011) NeuriPS 2020.

[5] Jonathan Ho et al. [“Denoising diffusion probabilistic models.”](https://arxiv.org/abs/2006.11239) arxiv Preprint arxiv:2006.11239 (2020). [[code](https://github.com/hojonathanho/diffusion)]

[6] Jiaming Song et al. [“Denoising diffusion implicit models.”](https://arxiv.org/abs/2010.02502) arxiv Preprint arxiv:2010.02502 (2020). [[code](https://github.com/ermongroup/ddim)]

[7] Alex Nichol & Prafulla Dhariwal. [“Improved denoising diffusion probabilistic models”](https://arxiv.org/abs/2102.09672) arxiv Preprint arxiv:2102.09672 (2021). [[code](https://github.com/openai/improved-diffusion)]

[8] Prafula Dhariwal & Alex Nichol. [“Diffusion Models Beat GANs on Image Synthesis."](https://arxiv.org/abs/2105.05233) arxiv Preprint arxiv:2105.05233 (2021). [[code](https://github.com/openai/guided-diffusion)]

[9] Jonathan Ho & Tim Salimans. [“Classifier-Free Diffusion Guidance."](https://arxiv.org/abs/2207.12598) NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications.

[10] Yang Song, et al. [“Score-Based Generative Modeling through Stochastic Differential Equations."](https://openreview.net/forum?id=PxTIG12RRHS) ICLR 2021.

[11] Alex Nichol, Prafulla Dhariwal & Aditya Ramesh, et al. [“GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models."](https://arxiv.org/abs/2112.10741) ICML 2022.

[12] Jonathan Ho, et al. [“Cascaded diffusion models for high fidelity image generation."](https://arxiv.org/abs/2106.15282) J. Mach. Learn. Res. 23 (2022): 47-1.

[13] Aditya Ramesh et al. [“Hierarchical Text-Conditional Image Generation with CLIP Latents."](https://arxiv.org/abs/2204.06125) arxiv Preprint arxiv:2204.06125 (2022).

[14] Chitwan Saharia & William Chan, et al. [“Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding."](https://arxiv.org/abs/2205.11487) arxiv Preprint arxiv:2205.11487 (2022).

[15] Rombach & Blattmann, et al. [“High-Resolution Image Synthesis with Latent Diffusion Models."](https://arxiv.org/abs/2112.10752) CVPR 2022.[code](https://github.com/CompVis/latent-diffusion)
