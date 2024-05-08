

# 扩散模型

扩散模型的灵感来自非平衡热力学。他们定义了一个扩散步骤的马尔可夫链，逐渐向数据添加随机噪声，然后学习逆转扩散过程，以从噪声构建所需的数据样本。与 VAE 或流模型不同，扩散模型是通过固定过程学习的，并且潜在变量具有高维性（与原始数据相同）。

<div align=center>
<img width="400" src="https://lilianweng.github.io/posts/2021-07-11-diffusion-models/generative-overview.png"/>
</div>
<div align=center>图 1. 不同类型生成模型的概述。</div>

# 什么是扩散模型？

已经提出了几种基于扩散的生成模型，这些模型具有相似的思想，包括扩散概率模型（[Sohl-Dickstein 等人，2015 年](https://arxiv.org/abs/1503.03585)）、噪声条件分数网络（NCSN；[Yang 和 Ermon，2019 年](https://arxiv.org/abs/1907.05600)）和去噪扩散概率模型（DDPM；[Ho 等人，2020 年](https://arxiv.org/abs/2006.11239)）。

## 前向扩散过程

给定一个从真实分布中采样的数据 $\mathbf{x}_0 \sim q(\mathbf{x})$，我们定义一个前向扩散过程，在这个过程中我们以 $T$ 步向样本中添加少量高斯噪声，产生一系列噪声样本 $\mathbf{x}_1, \dots, \mathbf{x}_T$ , 噪声大小由方差$\{\beta_t \in (0, 1)\}_{t=1}^T$控制 .
$$
q(\mathbf{x}_t \vert \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t\mathbf{I}) \quad
q(\mathbf{x}_{1:T} \vert \mathbf{x}_0) = \prod^T_{t=1} q(\mathbf{x}_t \vert \mathbf{x}_{t-1})
$$
当步骤 $t$  逐渐变大, 数据样本 $\mathbf{x}_0$ 逐渐失去其可辨特征。最终当$t$→∞时, $\mathbf{x}_T$ 等价于一个各向同性高斯分布。

<div align=center>
<img width="600" src="https://lilianweng.github.io/posts/2021-07-11-diffusion-models/DDPM.png"/>
</div>
<div align=center>图2. 通过逐步添加（移除）噪声生成样本的前向（反向）扩散过程的马尔可夫链。（图片来源：[Ho et al.2020](https://arxiv.org/abs/2006.11239)，带有一些附加注释）</div>

上述过程的一个很好的特性是我们用 [Reparameterization Trick](https://lilianweng.github.io/posts/2018-08-12-vae/#reparameterization-trick) 在任意时间步 $t$ 以封闭形式采样 $\mathbf{x}_t$。让 $\alpha_t = 1 - \beta_t$和 $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$，则:
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
() 回想一下，当我们合并两个具有不同方差的高斯分布，$\mathcal{N}(\mathbf{0}, \sigma_1^2\mathbf{I})$ 和 $\mathcal{N}(\mathbf{0}, \sigma_2^2\mathbf{I})$ 时， 得到的新的分布是 $\mathcal{N}(\mathbf{0}, (\sigma_1^2 + \sigma_2^2)\mathbf{I})$. 这里的合并标准差是 $\sqrt{(1 - \alpha_t) + \alpha_t (1-\alpha_{t-1})} = \sqrt{1 - \alpha_t\alpha_{t-1}}$.

通常，当样本噪声更大时，可以承受更大的更新步长，所以 $\beta_1 < \beta_2 < \dots < \beta_T$ 因此 $\bar{\alpha}_1 > \dots > \bar{\alpha}_T$.

### 与随机梯度朗之万动力学的联系

朗之万动力学（Langevin dynamics ）是物理学的一个概念，用于统计建模分子系统。结合随机梯度下降，随机梯度 Langevin 动力学( [Welling & Teh 2011](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.226.363) ) 可以可以在一系列更新的马尔可夫链中仅使用$\nabla_\mathbf{x} \log p(\mathbf{x})$的梯度从概率密度$p(\mathbf{x})$中产生样本：
$$
\mathbf{x}_t = \mathbf{x}_{t-1} + \frac{\delta}{2} \nabla_\mathbf{x} \log q(\mathbf{x}_{t-1}) + \sqrt{\delta} \boldsymbol{\epsilon}_t
,\quad\text{where }
\boldsymbol{\epsilon}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$
其中 $\delta$ 是步长。当 $T \to \infty, \epsilon \to 0$ 时，  $\mathbf{x}_t$ 等于真实概率密度 $p(\mathbf{x})$ .

与标准随机梯度下降 SGD 相比，随机梯度 Langevin 动力学将高斯噪声注入参数更新，以避免陷入局部最小值。

## 反向扩散过程

如果说前向过程(forward)是加噪的过程，那么逆向过程(reverse)就是扩散模型的去噪推断过程。

如果我们可以反转上述过程并从 $q(\mathbf{x}_{t-1} \vert \mathbf{x}_t)$ 中采样，我们将能够从高斯噪声输入$\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$重建真实样本. 注意，如果$\beta_t$足够小，$q(\mathbf{x}_{t-1} \vert \mathbf{x}_t)$ 也将是高斯分布。不幸的是，我们无法轻易估计 $q(\mathbf{x}_{t-1} \vert \mathbf{x}_t)$ ， 因为它需要用到整个数据集，因此我们需要学习一个模型 $p_\theta$ 近似这些条件概率以运行反向扩散过程。
$$
p_\theta(\mathbf{x}_{0:T}) = p(\mathbf{x}_T) \prod^T_{t=1} p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t) \quad
p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))
$$

<div align=center>
<img src="https://lilianweng.github.io/posts/2021-07-11-diffusion-models/diffusion-example.png" alt="img" style="zoom: 10%;" />
</div>

> Fig. 3. An example of training a diffusion model for modeling a 2D swiss roll data. (Image source: [Sohl-Dickstein et al., 2015](https://arxiv.org/abs/1503.03585))

值得注意的是，以 $\mathbf{x}_0$ 为条件的反向条件概率是可行的:
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
其中 $ C(\mathbf{x}_t, \mathbf{x}_0)$ 是不涉及 $\mathbf{x}_{t-1}$ 的某些函数并且省略了细节。按照标准高斯密度函数，均值和方差可以如下参数化（其中 $\alpha_t = 1 - \beta_t$ 和 $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$)
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

由于这个良好的属性, 我们可以表示 $\mathbf{x}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_t)$并将其代入上述等式，得到：

$$
\begin{aligned}
\tilde{\boldsymbol{\mu}}_t
&= \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} \frac{1}{\sqrt{\bar{\alpha}_t}}(\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_t) \\
&= \color{cyan}{\frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_t \Big)}
\end{aligned}
$$

如图 2 所示，这种设置与[VAE](https://lilianweng.github.io/posts/2018-08-12-vae/)非常相似，因此我们可以使用变分下界来优化负对数似然。

$$
\begin{aligned}
-\log p_\theta(\mathbf{x}_0)
&\leq - \log p_\theta(\mathbf{x}_0) + D_{KL}(q(\mathbf{x}_{1:T}|\mathbf{x}_0) \| p_\theta(\mathbf{x}_{1:T}|\mathbf{x}_0)) \\
&= -\log p_\theta(\mathbf{x}_0) + \mathbb{E}_{q(\mathbf{x}_1|\mathbf{x}_0)} \left[ \log \frac{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T}) / p_\theta(\mathbf{x}_0)} \right] \\
&= -\log p_\theta(\mathbf{x}_0) + \mathbb{E}_q \left[ \log \frac{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} + \log p_\theta(\mathbf{x}_0) \right] \\
&= \mathbb{E}_q \left[ \log \frac{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} \right] \\
Let \ L_{VLB} = \mathbb{E}_{q(\mathbf{x}_{0:T})} \left[ \log \frac{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} \right] \geq - \mathbb{E}_{q(\mathbf{x}_0)} \log p_\theta(\mathbf{x}_0)
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

$L_\text{VLB}$ 中的每个 KL 项（除了$L_0$) 比较了两个高斯分布，因此可以用[封闭形式](https://en.wikipedia.org/wiki/Kullback–Leibler_divergence#Multivariate_normal_distributions)计算它们。$L_T$是常数，在训练期间可以忽略，因为$q$没有可学习的参数并且 $\mathbf{x}_t$ 是高斯噪声。[Ho et al. 2020](https://arxiv.org/abs/2006.11239) 使用从 $\mathcal{N}(\mathbf{x}_0; \boldsymbol{\mu}_\theta(\mathbf{x}_1, 1), \boldsymbol{\Sigma}_\theta(\mathbf{x}_1, 1))$  d派生的单独的离散解码器建模了 $L_0$。

## 对于训练损失参数化 $L_t$

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

[Ho et al. (2020)](https://arxiv.org/abs/2006.11239) 实证发现，使用一个简化的目标函数，忽略权重项，可以更好地训练扩散模型：
$$
\begin{aligned}
L_t^\text{simple}
&= \mathbb{E}_{t \sim [1, T], \mathbf{x}_0, \boldsymbol{\epsilon}_t} \Big[\|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2 \Big] \\
&= \mathbb{E}_{t \sim [1, T], \mathbf{x}_0, \boldsymbol{\epsilon}_t} \Big[\|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_t, t)\|^2 \Big]
\end{aligned}
$$
最终的简化目标是：
$$
\begin{aligned}
L_t^\text{simple}
&= \mathbb{E}_{t \sim [1, T], \mathbf{x}_0, \boldsymbol{\epsilon}_t} \Big[\|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2 \Big] \\
&= \mathbb{E}_{t \sim [1, T], \mathbf{x}_0, \boldsymbol{\epsilon}_t} \Big[\|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_t, t)\|^2 \Big]
\end{aligned}
$$
其中 $C$ 是一个常数，不依赖于$\theta$.


<div align=center>
<img src="https://lilianweng.github.io/posts/2021-07-11-diffusion-models/DDPM-algo.png" alt="img" style="zoom:33%;" />
</div>


>  图 4. DDPM 中的训练和采样算法（图片来源：[Ho et al. 2020](https://arxiv.org/abs/2006.11239)）

## 与噪声条件分数网络（NCSN）的联系

[Song & Ermon (2019)](https://arxiv.org/abs/1907.05600)提出了一种基于分数的生成建模方法，其中样本是使用分数匹配估计的数据分布梯度，通过[Langevin 动力学](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#connection-with-stochastic-gradient-langevin-dynamics) 产生的。每个样本 $\mathbf{x}$ 的概率密度分数定义为其梯度 $\nabla_{\mathbf{x}} \log q(\mathbf{x})$. 分数网络$\mathbf{s}_\theta: \mathbb{R}^D \to \mathbb{R}^D$被训练来估计它： $\mathbf{s}_\theta(\mathbf{x}) \approx \nabla_{\mathbf{x}} \log q(\mathbf{x})$.

为了使其在深度学习中扩展到高维数据，他们建议使用去噪分数匹配( [Vincent, 2011](http://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf) ) 或切片分数匹配（使用随机投影；[Song 等人，2019](https://arxiv.org/abs/1905.07088)）。去噪分数匹配向数据$q(\tilde{\mathbf{x}} \vert \mathbf{x})$ 添加预先指定的少量噪声，并使用分数匹配估计$q(\tilde{\mathbf{x}} )$。

回想一下， Langevin 动力学可以在迭代过程中仅使用分数$\nabla_{\mathbf{x}} \log q(\mathbf{x})$从概率密度分布中采样数据。

然而，根据流形假设，大多数数据预计会集中在低维流形中，即使观察到的数据可能看起来是任意高维。由于数据点无法覆盖整个空间，因此会对分数估计产生负面影响。在数据密度低的区域，分数估计不太可靠。加入小的高斯噪声后，使得扰动后的数据分布覆盖全空间$\mathbb{R}^D$，分数估计网络的训练变得更加稳定。[Song & Ermon (2019)](https://arxiv.org/abs/1907.05600)通过用不同级别的噪声扰动数据，并训练了一个噪声条件分数网络来联合估计所有扰动数据在不同噪声级别下的分数，从而对它进行改进。

增加噪声水平的时间表类似于前向扩散过程。如果我们使用扩散过程注释，分数近似于 $\mathbf{s}_\theta(\mathbf{x}_t, t) \approx \nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t)$. 给定高斯分布 $\mathbf{x} \sim \mathcal{N}(\mathbf{\mu}, \sigma^2 \mathbf{I})$，我们可以将密度函数对数的导数写为$\nabla_{\mathbf{x}}\log p(\mathbf{x}) = \nabla_{\mathbf{x}} \Big(-\frac{1}{2\sigma^2}(\mathbf{x} - \boldsymbol{\mu})^2 \Big) = - \frac{\mathbf{x} - \boldsymbol{\mu}}{\sigma^2} = - \frac{\boldsymbol{\epsilon}}{\sigma}$. 其中$\boldsymbol{\epsilon} \sim \mathcal{N}(\boldsymbol{0}, \mathbf{I})$.  回想一下 $q(\mathbf{x}_t \vert \mathbf{x}_0) \sim \mathcal{N}(\sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I})$ 因此，
$$
\mathbf{s}_\theta(\mathbf{x}_t, t)
\approx \nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t)
= \mathbb{E}_{q(\mathbf{x}_0)} [\nabla_{\mathbf{x}_t} q(\mathbf{x}_t \vert \mathbf{x}_0)]
= \mathbb{E}_{q(\mathbf{x}_0)} \Big[ - \frac{\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{1 - \bar{\alpha}_t}} \Big]
= - \frac{\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{1 - \bar{\alpha}_t}}
$$

## $\beta_t$ 的参数化

[在Ho 等人(2020)](https://arxiv.org/abs/2006.11239) 的工作中， 前向方差被设置为一系列线性增加的常数, 从$\beta_1=10^{-4}$ 到 $\beta_T=0.02$ . 与归一化的图像像素值 $[-1, 1]$ 相比，它们相对较小。 在他们的实验中，扩散模型显示了高质量的样本，但仍然无法像其他生成模型那样达到有竞争力的模型对数似然。

[Nichol & Dhariwal (2021)](https://arxiv.org/abs/2102.09672)提出了几种改进技术来帮助扩散模型获得更低的 NLL。其中一项改进是使用基于余弦的方差表。调度函数的选择可以是任意的，只要它在训练过程中提供近乎线性下降, 并在 $t$=0 和 $t$=$T$ 处有细微变化即可。
$$
\beta_t = \text{clip}(1-\frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}, 0.999) \quad\bar{\alpha}_t = \frac{f(t)}{f(0)}\quad\text{where }f(t)=\cos\Big(\frac{t/T+s}{1+s}\cdot\frac{\pi}{2}\Big)
$$
其中小偏移量$s$是为了防止当$t=0$时, $\beta_t$过小.

<div align=center>
<img src="https://lilianweng.github.io/posts/2021-07-11-diffusion-models/diffusion-beta.png" alt="img" style="zoom:33%;" />
</div>

>  Fig. 5. Comparison of linear and cosine-based scheduling of $\beta\_t$ during training. (Image source: [Nichol & Dhariwal, 2021](https://arxiv.org/abs/2102.09672))

## 逆过程方差的参数化 $ \boldsymbol{\Sigma}_\theta$

[Ho et al. (2020)](https://arxiv.org/abs/2006.11239)  将 $\beta_t$ 固定为常量而不是可学习的参数， 并设置$\boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t) = \sigma^2_t \mathbf{I}$，其中$\sigma_t$ 不是学习的，而是设置为 $\beta_t$ 或者 $\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t$，  因为他们发现学习对角方差 $\boldsymbol{\Sigma}_\theta$ 会导致训练不稳定和较差的样本质量。

[Nichol & Dhariwal (2021)](https://arxiv.org/abs/2102.09672)建议通过模型预测混合向量$\mathbf{v}$ 来学习$\boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t)$ 作为 $\beta_t$和 $\tilde{\beta}_t$之间的插值:
$$
\boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t) = \exp(\mathbf{v} \log \beta_t + (1-\mathbf{v}) \log \tilde{\beta}_t)
$$
然而，简单的目标 $L_\text{simple}$ 不依赖于 $\boldsymbol{\Sigma}_\theta$. 为了增加依赖性，他们构建了一个混合目标 $L_\text{hybrid} = L_\text{simple} + \lambda L_\text{VLB}$,  其中λ=0.001很小， 并且在$L_\text{VLB}$ 项中的 $\boldsymbol{\mu}_\theta$上停止梯度， 使得$L_\text{VLB}$只指导$\boldsymbol{\Sigma}_\theta$的学习. 根据经验，他们观察到$L_\text{VLB}$可能由于嘈杂的梯度而难以优化，因此他们建议使用具有重要性抽样的$L_\text{VLB}$的时间平滑版本。

<div align=center>
<img src="https://lilianweng.github.io/posts/2021-07-11-diffusion-models/improved-DDPM-nll.png" alt="img" style="zoom: 25%;" />
</div>

> 图 6. 改进的 DDPM 与其他基于似然的生成模型的负对数似然比较。NLL 以 bits/dim 为单位报告。（图片来源：[Nichol & Dhariwal，2021 年](https://arxiv.org/abs/2102.09672))

# 条件生成

在使用 ImageNet 数据集等条件信息对图像训练生成模型时，通常会生成以类标签或一段描述性文本为条件的样本。

## 分类器引导扩散

为了明确地将类信息纳入扩散过程，[Dhariwal & Nichol (2021)](https://arxiv.org/abs/2105.05233)  在噪声图像 $ \mathbf{x}_t $ 上训练了一个分类器$ f_\phi(y | \mathbf{x}_t, t) $，并使用梯度$ \nabla_{\mathbf{x}} \log f_\phi(y | \mathbf{x}_t) $ 通过改变噪声预测来引导扩散采样过程朝向条件信息 $ y $（例如目标类标签）。回想一下 $ \nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t) = - \frac{1}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) $ 并且我们可以将联合分布$ q(\mathbf{x}_t, y) $ 的分数函数写为：

$$
\begin{aligned}
\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t, y) &= \nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t) + \nabla_{\mathbf{x}_t} \log q(y | \mathbf{x}_t) \\
&\approx - \frac{1}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) + \nabla_{\mathbf{x}_t} \log f_\phi(y | \mathbf{x}_t) \\
&= - \frac{1}{\sqrt{1 - \bar{\alpha}_t}} (\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) - \sqrt{1 - \bar{\alpha}_t} \nabla_{\mathbf{x}_t} \log f_\phi(y | \mathbf{x}_t))
\end{aligned}
$$
因此，一个新的分类器引导预测器 $ \bar{\boldsymbol{\epsilon}}_\theta $ 将采取以下形式：

$$
\bar{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t) = \boldsymbol{\epsilon}_\theta(x_t, t) - \sqrt{1 - \bar{\alpha}_t} \nabla_{\mathbf{x}_t} \log f_\phi(y | \mathbf{x}_t)
$$
为了控制分类器引导的强度，我们可以向增量部分添加一个权重$ w $：

$$
\bar{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t) = \boldsymbol{\epsilon}_\theta(x_t, t) - \sqrt{1 - \bar{\alpha}_t} \; w \nabla_{\mathbf{x}_t} \log f_\phi(y | \mathbf{x}_t)
$$
结果，去除扩散模型（**ADM**）和带有额外分类器引导的模型（**ADM-G**）能够比SOTA生成模型（例如BigGAN）取得更好的结果。

<div align=center>
<img src="https://lilianweng.github.io/posts/2021-07-11-diffusion-models/conditioned-DDPM.png" alt="img" style="zoom: 25%;" />
</div>

> 图7. 使用分类器引导在DDPM和DDIM中进行条件生成的算法。（图片来源：，[Dhariwal & Nichol (2021)](https://arxiv.org/abs/2105.05233))

此外，通过对U-Net架构的一些修改，[Dhariwal & Nichol (2021)](https://arxiv.org/abs/2105.05233)展示了扩散模型的性能优于GAN。架构修改包括更大的模型深度/宽度、更多的注意力头、多分辨率注意力、用于上/下采样的BigGAN残差块、残差连接重缩放为$ 1/\sqrt{2} $ 和自适应组归一化（AdaGN）。

## 无分类器引导

没有独立的分类器$ f_\phi $，仍然可以通过结合条件和非条件扩散模型的分数来执行条件扩散步骤l ([Ho & Salimans, 2021](https://openreview.net/forum?id=qw8AKxfYbI)).。让非条件去噪扩散模型$ p_\theta(\mathbf{x}) $ 通过分数估计器$ \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) $ 参数化，条件模型$ p_\theta(\mathbf{x} | y) $ 通过$ \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, y) $ 参数化。这两个模型可以通过单个神经网络学习。准确地说，一个条件扩散模型$ p_\theta(\mathbf{x} | y) $ 在成对数据$ (\mathbf{x}, y) $ 上训练，其中条件信息$ y $ 会随机丢弃，以便模型知道如何无条件地生成图像，即$ \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, y=\varnothing) $。

隐式分类器的梯度可以用条件和非条件分数估计器表示。一旦插入到分类器引导的修改分数中，分数就不再依赖于单独的分类器。

$$
\begin{aligned}
\nabla_{\mathbf{x}_t} \log p(y | \mathbf{x}_t) &= \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t | y) - \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t) \\
&= - \frac{1}{\sqrt{1 - \bar{\alpha}_t}}\Big( \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, y) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \Big) \\
\bar{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t, y) &= \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, y) - \sqrt{1 - \bar{\alpha}_t} \; w \nabla_{\mathbf{x}_t} \log p(y | \mathbf{x}_t) \\
&= \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, y) + w \big(\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, y) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \big) \\
&= (w+1) \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, y) - w \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)
\end{aligned}
$$
他们的实验表明，无分类器引导可以在FID（区分合成和生成图像）和IS（质量和多样性）之间取得良好的平衡。

引导扩散模型GLIDE ( [Nichol, Dhariwal & Ramesh, et al. 2022](https://arxiv.org/abs/2112.10741) ) 探索了两种引导策略，CLIP引导和无分类器引导，并发现后者更受青睐。他们推测这是因为CLIP引导利用了对CLIP模型的对抗性示例，而不是优化更好的匹配图像生成。

# 加速扩散模型

通过遵循反向扩散过程的马尔可夫链，从 DDPM  生成样本速度非常慢，因为$T$ 可以高达几千步。[来自Song 等人,2020](https://arxiv.org/abs/2010.02502) 的一个数据。“例如，在 Nvidia 2080 Ti GPU 上，从 DDPM 采样 50k 大小为 32 × 32 的图像需要大约 20 小时，但从 GAN 采样则不到一分钟。”

## 更少的采样步骤和蒸馏

一种简单的方法是运行跨步采样计划（[Nichol & Dhariwal，2021](https://arxiv.org/abs/2102.09672)），每隔$\lceil T/S \rceil$运行一次采样，流程的步骤从$T$步减少到 $S$步。新的抽样时间表$\{\tau_1, \dots, \tau_S\}$ 其中 $\tau_1 < \tau_2 < \dots <\tau_S \in [1, T]$并且$S < T $.

对于另一种方法，让我们根据所需的标准偏差进行参数化$\sigma_t$，重写$q_\sigma(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)$：
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
让 $\sigma_t^2 = \eta \cdot \tilde{\beta}_t$,  这样我们就可以调整 $\eta \in \mathbb{R}^+ $作为控制采样随机性的超参数。特殊情况 $\eta = 0$ 使采样过程具有确定性。这样的模型被称为去噪扩散隐式模型（DDIM；[Song et al., 2020](https://arxiv.org/abs/2010.02502)）。DDIM 具有相同的边缘噪声分布，但以确定性的方式将噪声映射回原始数据样本。

在生成过程中，我们不必遵循整个链 $t=1,…,T$ ，而是可以遵循加速轨迹的子集 $S$： $\{\tau_1, \dots, \tau_S\}$， 推理过程变为：
$$
q_{\sigma, \tau}(\mathbf{x}_{\tau_{i-1}} \vert \mathbf{x}_{\tau_t}, \mathbf{x}_0)
= \mathcal{N}(\mathbf{x}_{\tau_{i-1}}; \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \frac{\mathbf{x}_{\tau_i} - \sqrt{\bar{\alpha}_t}\mathbf{x}_0}{\sqrt{1 - \bar{\alpha}_t}}, \sigma_t^2 \mathbf{I})
$$
虽然，在实验中，所有模型都经过$T=1000$ 扩散步骤的训练， 他们观察到 DDIM ($\eta=0$)  可以在 $S$ 很小的时候生产出质量最好的样本，而 DDPM ($\eta=1$) 在 $S$ 很小的情况下表现要差的多。当我们能够运行完整的逆向马尔可夫扩散步骤（$S$=$T$=1000)时，DDPM 确实表现更好. 使用 DDIM，可以将扩散模型训练到任意数量的前向步，但在生成过程中只从子集中采样。

<div align=center>
<img src="https://lilianweng.github.io/posts/2021-07-11-diffusion-models/DDIM-results.png" alt="img" style="zoom:33%;" />
</div>

> 图 8. 在CIFAR10和CelebA数据集上，不同设置的扩散模型的FID得分，包括DDIM($\eta=0$） 和DDPM($\hat{\sigma}$). （图片来源：[Song et al., 2020](https://arxiv.org/abs/2010.02502)）

与 DDPM 相比，DDIM 能够：

1. 使用更少的步骤生成更高质量的样本。
2. 具有“一致性”属性，因为生成过程是确定性的，这意味着以相同潜变量为条件的多个样本应该具有相似的高级特征。
3. 由于一致性，DDIM 可以在潜在变量中进行语义上有意义的插值。

<div align=center>
<img src="https://lilianweng.github.io/posts/2021-07-11-diffusion-models/progressive-distillation.png" alt="img" style="zoom: 33%;" />
</div>

> 图 9. 渐进蒸馏可以将每次迭代中的扩散采样步骤减少一半。 （图片来源：[Salimans & Ho，2022](https://arxiv.org/abs/2202.00512)）

### 渐进式蒸馏

**渐进式蒸馏**（Salimans和Ho，2022）是一种将经过训练的确定性采样器蒸馏为采样步骤减半的新模型的方法。学生模型从教师模型初始化，并朝着一个目标进行去噪，其中一个学生DDIM步骤匹配2个教师步骤，而不是使用原始样本 $x_0$ 作为去噪目标。在每次渐进式蒸馏迭代中，我们可以将采样步骤减半。

<div align=center>
<img src="https://lilianweng.github.io/posts/2021-07-11-diffusion-models/progressive-distillation-algo.png" alt="img" style="zoom:33%;" />
</div>

> 图10. 算法1（扩散模型训练）和算法2（渐进式蒸馏）并排比较，其中渐进式蒸馏中的相对变化以绿色突出显示。

### 一致性模型

**一致性模型**（Song等人，2023）学习将扩散采样轨迹上的任何中间噪声数据点 $ \mathbf{x}_t, t > 0 $ 直接映射回其原始 $ \mathbf{x}_0 $。由于其自我一致性属性，因此得名一致性模型，因为同一轨迹上的任何数据点都映射到同一起源。

<div align=center>
<img src="https://lilianweng.github.io/posts/2021-07-11-diffusion-models/consistency-models.png" alt="img" style="zoom:50%;" />
</div>

> 图11. 一致性模型学习将轨迹上的任何数据点映射回其起源。（图片来源：Song等人，2023）

给定一个轨迹 $ \{\mathbf{x}_t | t \in [\epsilon, T]\} $，一致性函数 $ f $ 定义为 $ f: (\mathbf{x}_t, t) \mapsto \mathbf{x}_\epsilon $，并且方程 $ f(\mathbf{x}_t, t) = f(\mathbf{x}_{t'}, t') = \mathbf{x}_\epsilon $ 对所有 $ t, t' \in [\epsilon, T] $ 都成立。当 $ t=\epsilon $ 时，$ f $ 是一个恒等函数。该模型可以如下参数化，其中 $ c_{\text{skip}}(t) $ 和 $ c_{\text{out}}(t) $ 函数被设计成 $ c_{\text{skip}}(\epsilon) = 1, c_{\text{out}}(\epsilon) = 0 $：

$$
f_\theta(\mathbf{x}, t) = c_{\text{skip}}(t)\mathbf{x} + c_{\text{out}}(t) F_\theta(\mathbf{x}, t)
$$
一致性模型有可能在单步中生成样本，同时仍然保持在多步采样过程中为了更好的质量而进行计算交换的灵活性。

论文介绍了两种训练一致性模型的方法：

1. **一致性蒸馏（CD）**：通过最小化相同轨迹生成的对生成的模型输出之间的差异，将扩散模型蒸馏到一致性模型。这使得采样评估成本更低。一致性蒸馏损失是：
    $$
    \begin{aligned}
     \mathcal{L}^N_\text{CD} (\theta, \theta^-; \phi) &= \mathbb{E}
     [\lambda(t_n)d(f_\theta(\mathbf{x}_{t_{n+1}}, t_{n+1}), f_{\theta^-}(\hat{\mathbf{x}}^\phi_{t_n}, t_n)] \\
     \hat{\mathbf{x}}^\phi_{t_n} &= \mathbf{x}_{t_{n+1}} - (t_n - t_{n+1}) \Phi(\mathbf{x}_{t_{n+1}}, t_{n+1}; \phi)
     \end{aligned}
    $$

其中：

- $ \Phi(.;\phi) $ 是一步ODE求解器的更新函数；
- $ n \sim \mathcal{U}[1, N-1] $，均匀分布在 $ 1, \dots, N-1 $；
- 网络参数 $ \theta^- $ 是 $ \theta $ 的指数移动平均（EMA）版本，这极大地稳定了训练（就像在DQN或动量对比学习中一样）；
- $ d(.,.) $ 是一个正的距离度量函数，满足 $ \forall \mathbf{x}, \mathbf{y}: d(\mathbf{x}, \mathbf{y}) \leq 0 $ 且 $ d(\mathbf{x}, \mathbf{y}) = 0 $ 当且仅当 $ \mathbf{x} = \mathbf{y} $，例如 $ \ell_2 $，$ \ell_1 $ 或LPIPS（学习感知图像补丁相似性）距离；
- $ \lambda(.) \in \mathbb{R}^+ $ 是一个正的权重函数，论文中设置 $ \lambda(t_n)=1 $。

2. **一致性训练（CT）**：另一种选择是独立训练一致性模型。注意到在CD中，使用预训练的分数模型 $ s_\phi(\mathbf{x}, t) $ 来近似真实分数 $ \nabla \log p_t(\mathbf{x}) $ 但在CT中我们需要一种方法来估计这个分数函数，并且存在一个无偏估计 $ -\frac{\mathbf{x}_t - \mathbf{x}}{t^2} $。CT损失定义如下：

$$
\mathcal{L}^N_{\text{CT}} (\theta, \theta^-; \phi) = \mathbb{E} \left[ \lambda(t_n)d(f_\theta(\mathbf{x} + t_{n+1} \mathbf{z};t_{n+1}), f_{\theta^-}(\mathbf{x} + t_n \mathbf{z};t_n)) \right]
\text{ where }\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

根据论文中的实验，他们发现：

- Heun ODE求解器比欧拉一阶求解器工作得更好，因为具有相同 $ N $ 的高阶ODE求解器具有更小的估计误差。
- 在不同的距离度量函数 $ d(.) $ 选项中，LPIPS度量比 $ \ell_1 $ 和 $ \ell_2 $ 距离工作得更好。
- 更小的 $ N $ 导致更快的收敛但更差的样本，而更大的 $ N $ 导致更慢的收敛但在收敛时更好的样本。

<div align=center>
<img src="https://lilianweng.github.io/posts/2021-07-11-diffusion-models/consistency-models-exp.png" alt="img" style="zoom:80%;" />
</div>

> 图12. 在不同配置下的一致性模型性能比较。CD的最佳配置是LPIPS距离度量、Heun ODE求解器和 $ N=18 $。（图片来源：Song等人，2023）

## 潜在变量空间

Latent diffusion model ( LDM ; [Rombach & Blattmann, et al. 2022](https://arxiv.org/abs/2112.10752) ) 在隐空间而不是在像素空间中运行扩散过程，使训练成本更低，推理速度更快。这是基于这样的观察：大多数图像的比特贡献于感知细节，而在积极压缩后，语义和概念组合仍然保留。LDM通过首先使用自动编码器减少像素级冗余，然后在学习到的潜在空间中使用扩散过程操作/生成语义概念，松散地分解了感知压缩和语义压缩。

<div align=center>
<img src="https://lilianweng.github.io/posts/2021-07-11-diffusion-models/image-distortion-rate.png" alt="img" style="zoom:33%;" />
</div>

>  Fig. 13. The plot for tradeoff between compression rate and distortion, illustrating two-stage compressions - perceptural and semantic comparession. (Image source: [Rombach & Blattmann, et al. 2022](https://arxiv.org/abs/2112.10752))

感知压缩过程依赖于自动编码器模型。编码器$\mathcal{E}$用于压缩输入图像$\mathbf{x} \in \mathbb{R}^{H \times W \times 3}$到更小的 2D 隐变量 $\mathbf{z} = \mathcal{E}(\mathbf{x}) \in \mathbb{R}^{h \times w \times c}$，其中下采样率$f=H/h=W/w=2^m, m \in \mathbb{N}$. 然后是解码器$\mathcal{D}$从隐变量重建图像，$\tilde{\mathbf{x}} = \mathcal{D}(\mathbf{z})$. 该论文探讨了自动编码器训练中的两种正则化，以避免隐空间中的任意高方差。

- KL-reg：对学习到的潜在空间施加一个小的KL惩罚，使其趋向标准正态分布，类似于[VAE](https://lilianweng.github.io/posts/2018-08-12-vae/)。
- VQ-reg：在解码器内部使用一个向量量化层，如[VQVAE](https://lilianweng.github.io/posts/2018-08-12-vae/#vq-vae-and-vq-vae-2)，但量化层被解码器吸收。

扩散和去噪过程发生在隐变量$z$上. 去噪模型是一个时间条件的 U-Net，增加了交叉注意机制，来处理图像生成的灵活条件信息（例如类标签、语义图、图像的模糊变体）。设计等同于使用交叉注意力机制将不同模态的表示融合到模型中。每种类型的条件信息都与特定领域的编码器$\tau_\theta$配对， 将条件输入$y$投射到一个中间表示，该表示可以映射到交叉注意力组件中， $\tau_\theta(y) \in \mathbb{R}^{M \times d_\tau}$:
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
<div align=center>
<img src="https://lilianweng.github.io/posts/2021-07-11-diffusion-models/latent-diffusion-arch.png" alt="img" style="zoom: 25%;" />
</div>

> Fig. 14. The architecture of latent diffusion model. (Image source: [Rombach & Blattmann, et al. 2022](https://arxiv.org/abs/2112.1075))

# 提高生成分辨率和质量

为了生成高分辨率的高质量图像，Ho等人（2021）提出了使用逐步增加分辨率的多个扩散模型的Pipeline。Pipeline模型之间的噪声条件增强对最终图像质量至关重要，这是通过对每个超分辨率模型 $ p_\theta(\mathbf{x} | \mathbf{z}) $ 的条件输入 $ \mathbf{z} $ 应用强数据增强来实现的。条件噪声有助于减少Pipeline设置中的复合误差。U-Net是高分辨率图像生成中扩散建模的常见选择。

<div align=center>
<img src="https://lilianweng.github.io/posts/2021-07-11-diffusion-models/cascaded-diffusion.png" alt="img" style="zoom:33%;" />
</div>

> 图15. 逐步增加分辨率的多个扩散模型的级联Pipeline。（图片来源：Ho等人，2021）

他们发现最有效的噪声是在低分辨率时应用高斯噪声，在高分辨率时应用高斯模糊。此外，他们还探索了两种形式的条件增强，这需要对训练过程进行小的修改。请注意，条件噪声仅在训练时应用，而不在推理时应用。

- 截断条件增强在低分辨率时提前停止扩散过程，步骤 $ t > 0 $。
- 非截断条件增强运行完整的低分辨率逆向过程，直到步骤0，但随后通过 $ \mathbf{z}_t \sim q(\mathbf{x}_t | \mathbf{x}_0) $ 破坏它，然后将破坏的 $ \mathbf{z}_t $ 输入到超分辨率模型中。

##  **unCLIP**

两阶段扩散模型 **unCLIP**（Ramesh等人，2022）大量利用CLIP文本编码器产生高质量的文本引导图像。给定一个预训练的CLIP模型 $ \mathbf{c} $ 和扩散模型的配对训练数据 $ (\mathbf{x}, y) $，其中 $ x $ 是图像，$ y $ 是相应的标题，我们可以计算CLIP文本和图像嵌入 $ \mathbf{c}^t(y) $ 和 $ \mathbf{c}^i(\mathbf{x}) $。unCLIP并行学习两个模型：

- 一个先验模型 $ P(\mathbf{c}^i | y) $：给定文本 $ y $ 输出CLIP图像嵌入 $ \mathbf{c}^i $。
- 一个解码器 $ P(\mathbf{x} | \mathbf{c}^i, [y]) $：给定CLIP图像嵌入 $ \mathbf{c}^i $ 和可选的原始文本 $ y $ 生成图像。

这两个模型支持条件生成，因为

$$
\begin{aligned}
P(\mathbf{x} | y) &= P(\mathbf{x}, \mathbf{c}^i | y) \\
&= P(\mathbf{x} | \mathbf{c}^i, y)P(\mathbf{c}^i | y) \quad \text{因为} \mathbf{c}^i \text{给定} \mathbf{x} \text{是确定性的}
\end{aligned}
$$

<div align=center>
<img src="https://lilianweng.github.io/posts/2021-07-11-diffusion-models/unCLIP.png" alt="img" style="zoom:33%;" />
</div>

> 图16. unCLIP的架构。（图片来源：Ramesh等人，2022）

unCLIP遵循两阶段图像生成过程：

1. 给定文本 $ y $，首先使用CLIP模型生成文本嵌入 $ \mathbf{c}^t(y) $。使用CLIP潜在空间通过文本实现零样本图像操作。
2. 一个扩散或自回归先验 $ P(\mathbf{c}^i | y) $ 处理这个CLIP文本嵌入以构建图像先验，然后一个扩散解码器 $ P(\mathbf{x} | \mathbf{c}^i, [y]) $ 生成一个图像，该图像受先验条件限制。这个解码器也可以在图像输入的条件下生成图像变体，保留其风格和语义。

## Imagen

**Imagen**（Saharia等人，2022）不是使用CLIP模型，而是使用预训练的大型语言模型（LM）（即冻结的T5-XXL文本编码器）对文本进行编码以生成图像。有一个普遍趋势表明，更大的模型尺寸可以带来更好的图像质量和文本图像对齐。他们发现，在MS-COCO上，T5-XXL和CLIP文本编码器的性能相似，但在DrawBench（涵盖11个类别的提示集合）上，人类评估更喜欢T5-XXL。

当应用无分类器引导时，增加 $ w $ 可能会导致更好的图像文本对齐，但图像保真度更差。他们发现这是由于训练-测试不匹配，也就是说，因为训练数据 $ \mathbf{x} $ 保持在 $ [-1, 1] $ 的范围内，测试数据也应该如此。他们介绍了两种阈值策略：

- 静态阈值化：将 $ \mathbf{x} $ 预测限制在 $ [-1, 1] $。
- 动态阈值化：在每个采样步骤中，计算 $ s $ 作为绝对像素值的某个百分位数值；如果 $ s > 1 $，则将预测限制在 $ [-s, s] $ 并除以 $ s $。

Imagen修改了U-Net的几个设计，使其成为高效的U-Net。

- 通过为较低分辨率添加更多的残差锁，将模型参数从高分辨率块转移到低分辨率块；
- 将跳跃连接缩放 $ 1/\sqrt{2} $；
- 为了提高前向传递的速度，颠倒下采样（移到卷积之前）和上采样操作（移到卷积之后）的顺序。

他们发现噪声条件增强、动态阈值化和高效U-Net对图像质量至关重要，但扩大文本编码器尺寸比U-Net尺寸更重要。

# 模型架构

扩散模型有两种常见的骨架架构选择：U-Net 和 Transformer。

## U-Net

**U-Net**（Ronneberger等人，2015）由一个下采样堆栈和一个上采样堆栈组成。

- **下采样**：每个步骤由两个3x3卷积（未填充的卷积）的重复应用组成，每个卷积后面跟着一个ReLU和一个步长为2的2x2最大池化。在每个下采样步骤中，特征通道的数量翻倍。
- **上采样**：每个步骤由特征图的上采样组成，后面跟着一个2x2卷积，每个步骤将特征通道的数量减半。
- **快捷方式**：快捷连接导致与下采样堆栈的相应层进行连接，并将必要的高分辨率特征提供给上采样过程。

<div align=center>
<img src="https://lilianweng.github.io/posts/2021-07-11-diffusion-models/U-Net.png" alt="img" style="zoom:33%;" />
</div>

> 图17. U-Net架构。每个蓝色方块是一个特征图，顶部标记了通道数，左侧底部标记了高度x宽度维度。灰色箭头标记了快捷连接。（图片来源：Ronneberger，2015）

为了使图像生成能够根据额外的图像进行条件组合信息（如Canny边缘、Hough线、用户涂鸦、人类后骨架、分割图、深度和法线）.

## ControlNet

**ControlNet**（Zhang等人，2023）通过在U-Net的每个编码器层中添加一个“夹心”零卷积层的可训练副本，引入了架构变化。准确地说，给定一个神经网络块 $ \mathcal{F}_\theta(.) $，ControlNet执行以下操作：

1. 首先，冻结原始块的原始参数 $ \theta $。
2. 克隆它成为一个具有可训练参数 $ \theta_c $ 的副本和一个额外的条件向量 $ \mathbf{c} $。
3. 使用两个零卷积层，表示为 $ \mathcal{Z}_{\theta_{z1}}(.;.) $ 和 $ \mathcal{Z}_{\theta_{z2}}(.;.) $，这是1x1的卷积层，其权重和偏置初始化为零，将这两个块连接起来。零卷积通过消除初始训练步骤中的随机噪声作为梯度来保护这个背骨。
4. 最终输出是：$ \mathbf{y}_c = \mathcal{F}_\theta(\mathbf{x}) + \mathcal{Z}_{\theta_{z2}}(\mathcal{F}_{\theta_c}(\mathbf{x} + \mathcal{Z}_{\theta_{z1}}(\mathbf{c}))) $

<div align=center>
<img src="https://lilianweng.github.io/posts/2021-07-11-diffusion-models/ControlNet.png" alt="img" style="zoom:33%;" />
</div>

> 图18. ControlNet架构。（图片来源：Zhang等人，2023）

## Diffusion Transformer

**Diffusion Transformer**（**DiT**; Peebles和Xie，2023）用于扩散建模，在潜在补丁上操作，使用与LDM（潜在扩散模型）相同的设计空间。DiT具有以下设置：

1. 将输入 $ \mathbf{z} $ 的潜在表示作为DiT的输入。
2. 将噪声潜在的大小 $ I \times I \times C $ “分块化”成大小为 $ p $ 的补丁，并将其转换为大小为 $ (I/p)^2 $ 的补丁序列。
3. 然后这个补丁序列的令牌通过Transformer块。他们探索了三种不同的设计，用于根据上下文信息（如时间步 $ t $ 或类标签 $ c $）进行生成的条件。在三种设计中，adaLN（自适应层归一化）-Zero工作得最好，比in-context条件和cross-attention块更好。比例和偏移参数 $ \gamma $ 和 $ \beta $ 是从 $ t $ 和 $ c $ 的嵌入向量之和中回归出来的。维度-wise缩放参数 $ \alpha $ 也是回归出来的，并立即应用于DiT块内任何残差连接之前。

4. 变压器解码器输出噪声预测和输出对角线协方差预测。

<div align=center>
<img src="https://lilianweng.github.io/posts/2021-07-11-diffusion-models/DiT.png" alt="img" style="zoom:33%;" />
</div>

> 图19. Diffusion Transformer（DiT）架构。（图片来源：Peebles和Xie，2023）

Transformer架构可以轻松地扩展，它以其可扩展性而闻名。这是DiT的最大优势之一，因为其性能随着更多的计算和更大的DiT模型而扩展，根据实验，更大的DiT模型在计算上更有效率。

# 快速总结

- 优点：易处理性和灵活性是生成建模中两个相互冲突的目标。易处理模型可以分析评估并以低成本拟合数据（例如通过高斯或拉普拉斯），但它们不能轻易地描述丰富数据集中的结构。灵活模型可以适应数据中的任意结构，但评估、训练或从这些模型中采样通常是昂贵的。扩散模型既易于分析又灵活。
- 缺点：扩散模型依赖于一系列很长的马尔可夫链扩散步骤来生成样本，因此在时间和计算方面可能非常昂贵。已经提出了新的方法来使过程更快，但采样仍然比 GAN 慢。

# References

[1] Jascha Sohl-Dickstein et al. [“Deep Unsupervised Learning using Nonequilibrium Thermodynamics.”](https://arxiv.org/abs/1503.03585) ICML 2015.

[2] Max Welling & Yee Whye Teh. [“Bayesian learning via stochastic gradient langevin dynamics.”](https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf) ICML 2011.

[3] Yang Song & Stefano Ermon. [“Generative modeling by estimating gradients of the data distribution.”](https://arxiv.org/abs/1907.05600) NeurIPS 2019.

[4] Yang Song & Stefano Ermon. [“Improved techniques for training score-based generative models.”](https://arxiv.org/abs/2006.09011) NeuriPS 2020.

[5] Jonathan Ho et al. [“Denoising diffusion probabilistic models.”](https://arxiv.org/abs/2006.11239) arxiv Preprint arxiv:2006.11239 (2020). [[code](https://github.com/hojonathanho/diffusion)]

[6] Jiaming Song et al. [“Denoising diffusion implicit models.”](https://arxiv.org/abs/2010.02502) arxiv Preprint arxiv:2010.02502 (2020). [[code](https://github.com/ermongroup/ddim)]

[7] Alex Nichol & Prafulla Dhariwal. [“Improved denoising diffusion probabilistic models”](https://arxiv.org/abs/2102.09672) arxiv Preprint arxiv:2102.09672 (2021). [[code](https://github.com/openai/improved-diffusion)]

[8] Prafula Dhariwal & Alex Nichol. [“Diffusion Models Beat GANs on Image Synthesis.”](https://arxiv.org/abs/2105.05233) arxiv Preprint arxiv:2105.05233 (2021). [[code](https://github.com/openai/guided-diffusion)]

[9] Jonathan Ho & Tim Salimans. [“Classifier-Free Diffusion Guidance.”](https://arxiv.org/abs/2207.12598) NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications.

[10] Yang Song, et al. [“Score-Based Generative Modeling through Stochastic Differential Equations.”](https://openreview.net/forum?id=PxTIG12RRHS) ICLR 2021.

[11] Alex Nichol, Prafulla Dhariwal & Aditya Ramesh, et al. [“GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models.”](https://arxiv.org/abs/2112.10741) ICML 2022.

[12] Jonathan Ho, et al. [“Cascaded diffusion models for high fidelity image generation.”](https://arxiv.org/abs/2106.15282) J. Mach. Learn. Res. 23 (2022): 47-1.

[13] Aditya Ramesh et al. [“Hierarchical Text-Conditional Image Generation with CLIP Latents.”](https://arxiv.org/abs/2204.06125) arxiv Preprint arxiv:2204.06125 (2022).

[14] Chitwan Saharia & William Chan, et al. [“Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding.”](https://arxiv.org/abs/2205.11487) arxiv Preprint arxiv:2205.11487 (2022).

[15] Rombach & Blattmann, et al. [“High-Resolution Image Synthesis with Latent Diffusion Models.”](https://arxiv.org/abs/2112.10752) CVPR 2022.[code](https://github.com/CompVis/latent-diffusion)

[16] Song et al. [“Consistency Models”](https://arxiv.org/abs/2303.01469) arxiv Preprint arxiv:2303.01469 (2023)

[17] Salimans & Ho. [“Progressive Distillation for Fast Sampling of Diffusion Models”](https://arxiv.org/abs/2202.00512) ICLR 2022.

[18] Ronneberger, et al. [“U-Net: Convolutional Networks for Biomedical Image Segmentation”](https://arxiv.org/abs/1505.04597) MICCAI 2015.

[19] Peebles & Xie. [“Scalable diffusion models with transformers.”](https://arxiv.org/abs/2212.09748) ICCV 2023.

[20] Zhang et al. [“Adding Conditional Control to Text-to-Image Diffusion Models.”](https://arxiv.org/abs/2302.05543) arxiv Preprint arxiv:2302.05543 (2023)
