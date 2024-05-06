
# 扩散模型

扩散模型解决的任务类似于生成对抗网络（GANs）以及其他类型的生成模型，如变分自编码器（VAEs）或正态流（Normalizing Flows）——它们尝试逼近给定领域的概率分布 $q(x)$，并且最重要的是，提供一种从该分布中采样的方法 $x \sim q(x)$。

这是通过优化某些参数 $\theta$（表示为神经网络）来实现的，这些参数导致概率分布 $p_\theta(x)$。训练的目标是让 $p_\theta$ 生成的样本 $x$ 与从真实底层分布 $q(x)$ 中抽取的样本相似。

## 与 GANs 的不同之处？
* GANs 通过生成器网络的单次前向传递从潜在向量中产生样本。产生的样本的可能性由鉴别器网络控制，该网络被训练用来区分 $x \sim q(x)$ 和 $x \sim p_\theta(x)$。
* 扩散模型使用一个网络，通过多个估计步骤顺序地逼近真实样本 $x \sim q(x)$ 的近似值。因此，模型的输入和输出通常是相同的维度。

## 去噪扩散过程的机制

去噪扩散过程由两个方向的一系列步骤组成，对应于样本中信息的破坏和创建。

### 前向过程

在拥有时间步 $t$ 的样本的情况下，可以对前向过程中下一个样本进行估计，由真实分布 $q$ 定义：

```math
q(x_{t}|x_{t-1})\tag{1}
```

通常，可用的是时间步 $0$ 的样本（即干净的样本），并且使用允许轻松高效地表述以下类型的操作是有用的：

```math
q(x_{t}|x_{0})\tag{2}
```

到目前为止，**最常见的**前向过程选择是 **高斯过程**。易于计算并且在各方面都很便利：

```math
q(x_{t}|x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}x_{t-1}, \beta_t I)\tag{3}
```

上述符号意味着 **先前的样本通过 $\sqrt{1-\beta_t}$ 的因子缩小** 并且 **添加了额外的高斯噪声**（从均值为零，方差为单位的高斯中采样）乘以 $\beta_t$。

此外，$0\to t$ 步骤也可以很容易地定义为：

```math
q(x_{t}|x_{0}) = \mathcal{N}(\sqrt{\bar{\alpha_t}}x_0, (1-\bar{\alpha_t}) I) \tag{4}
```

其中 $\alpha_t = 1-\beta_t$ 且

```math
\bar{\alpha_t}=\prod_{i=0}^{t}\alpha_t \tag{5}
```

### 反向过程

反向过程旨在恢复样本中的信息，从而允许从分布中生成新的样本。通常，它会从某个高时间步 $t$（通常是 $t=T$，表示扩散链的末端，此时概率分布非常接近纯高斯）开始，并尝试逼近前一个样本 $t-1$ 的分布。

```math
p_\theta(x_{t-1}|x_t)
```

如果扩散步骤足够小，高斯前向过程的反向过程也可以通过高斯来逼近：

```math
p_\theta(x_{t-1}|x_t) = \mathcal{N}(\mu_\theta(x_t,t),\Sigma_\theta(x_t,t))\tag{6}
```

反向过程通常使用神经网络 $\theta$ 来参数化，神经网络是一个很好的候选者，用于逼近复杂的变换。在许多情况下，可以使用独立于 $x_t$ 的标准差函数 $\sigma_t$：

```math
p_\theta(x_{t-1}|x_t) = \mathcal{N}(\mu_\theta(x_t,t),\sigma_t^2 I)\tag{7}
```

## DDPM: 去噪扩散概率模型

[DDPM](https://arxiv.org/abs/2006.11239) 是去噪扩散领域的一个流行方法之一。它通过遵循反向过程的所有 *T* 步骤生成样本。

当涉及到参数化反向过程分布的均值 $\mu_\theta(x_t,t)$ 时，网络可以：
1. 直接预测它作为 $\mu_\theta(x_t,t)$
2. 预测原始的 $t=0$ 样本 $x_0$，其中

```math
\tilde{\mu}_\theta = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}x_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t \tag{8}
```

3. 预测添加到样本 $x_0$ 上的 **正常** 噪声样本 $\epsilon$（来自单位方差的分布）

```math
x_0=\frac{1}{\sqrt{\bar{\alpha}_t}}(x_t-\sqrt{1-\bar{\alpha}_t}\epsilon) \tag{9}
```

第三种选择，即网络预测 $\epsilon$，似乎是最常见的，这就是 DDPM 在采样中所做的事情。这导致 $\tilde{\mu}_{\theta}$ 的新方程，以 $x_t$ 和 $\epsilon$ 表示：

```math
\tilde{\mu}_\theta = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}(\frac{1}{\sqrt{\bar{\alpha}_t}}(x_t-\sqrt{1-\bar{\alpha}_t}\epsilon)) + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t \tag{10}
```

因此，

```math
\tilde{\mu}_\theta =\frac{1}{\sqrt{\alpha_t}}(x_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon) \tag{11}
```

...这是 DDPM 用于采样的关键方程。

### 训练

训练一个模型来预测噪声形状 $\epsilon$ 是相当直接的。

在每个训练步骤中：

1. 使用 **前向过程** 生成一个样本 $x_t \sim q(x_t|x_0)$，对于从 $[1,T]$ 中均匀采样的 $t$：
   1. 从均匀分布中采样时间步 $t$，$t \sim \mathcal{U}(1,T)$
   2. 从正态高斯中采样 $\epsilon \sim \mathcal{N}(0,1)$
   3. 通过 $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$ 计算训练中的噪声输入样本
2. **计算噪声的近似值** $\hat{\epsilon_t}=p_\theta(x_t,t)$，使用具有参数 $\theta$ 的模型
3. **最小化** $\epsilon_t$ 和 $\hat{\epsilon_t}$ 之间的误差，通过优化参数 $\theta$

### 采样

生成过程从 $t=T$ 开始，通过在扩散过程的最后步骤中采样 $x_T \sim \mathcal{N}(0,1)$，该过程由正态高斯建模。

然后，直到达到 $t=0$，网络对样本中的噪声 $\tilde{\epsilon}=p_\theta(x_t,t)$ 进行预测，然后使用以下方法逼近过程在 $t-1$ 的均值：

```math
\tilde{\mu}_\theta =\frac{1}{\sqrt{\alpha_t}}(x_t-\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\tilde{\epsilon})\tag{12}
```

因此，下一个样本 $t-1$ 是从高斯分布中采样的，如下所示：

```math
x_{t-1} \sim \mathcal{N}(\tilde{\mu}_\theta,\sigma_t^2 I)\tag{13}
```

...直到达到 $x_0$，在这种情况下，只提取均值 $\tilde{\mu}_\theta$ 作为输出。

##  (更快采样) DDIM: 去噪扩散隐式模型

> 警告：如果你查看原始的 DDIM 论文，你会看到符号 $\alpha_t$ 被用来表示 $\bar{\alpha}_t$。为了一致性，这篇笔记中没有进行这样的符号更改。

`DDPM` 反向过程尝试以相反的顺序导航扩散链的 `T` 步骤。然而，如 (9) 所示，反向过程涉及对干净样本 $x_0$ 的近似。

如果我们在 (4) 中将 $t-1$ 替换为 $t$：

```math
q(x_{t-1}|x_{0}) = \mathcal{N}(\sqrt{\bar{\alpha}_{t-1}}x_0, (1-\bar{\alpha}_{t-1}) I)\tag{14}
```

这将产生

```math
x_{t-1} \leftarrow \sqrt{\bar{\alpha}_{t-1}}x_0 + \sqrt{1-\bar{\alpha}_{t-1}} \epsilon_{t-1}\tag{15}
```

...并且基于前一步 $t$ 测量的特定 $\epsilon_t$，它可以被重写为：

```math
x_{t-1} \leftarrow \sqrt{\bar{\alpha}_{t-1}}x_0 + \sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2} \epsilon_{t} + \sigma_t \epsilon\tag{16}
```

通常，$\sigma_t$ 被设置为：

```math
\sigma_t^2 = \tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t\tag{17}
```

进一步，我们可以引入一个新参数 $\eta$ 来控制随机组成部分的幅度：

$$\sigma_t^2 = \eta \tilde{\beta}_t \tag{18}$$

正如在原始的 [DDIM 论文](https://arxiv.org/abs/2010.02502) 中发现的，设置 $\eta=0$ 在应用较少的反向过程步骤时特别有益，并且该特定类型的过程被称为 **去噪扩散隐式模型（DDIM）**。当 $\eta=1$ 时，上述公式仍然与 DDPM 一致。

那么，如何以相反的方向导航反向链呢？首先，定义一个较少步骤的序列 $S$ 作为前向过程原始时间步骤的子集 $\{\tau_1, \tau_2, ..., \tau_S\}$。然后，基于 (16) 进行采样。

在每一步中：
1. 预测 $x_0$
2. 计算当前 $x_t$ 的方向
3. （如果不是 DDIM）为了随机功能注入一些噪声

通常可以假设 DDIM：
* 在较少的步骤中提供更好的样本质量
* 允许起始噪声 $x_T$ 和生成样本 $x_0$ 之间的确定性匹配
* 对于大量的步骤（例如 1000 步）表现不如 DDPM



## 总结

扩散模型通过训练一个神经网络来预测噪声样本 $\epsilon$，从而生成高质量的图像样本。这些模型与 GANs 不同，它们通过一系列估计步骤逐步逼近真实样本的近似值，而不是通过单一的前向传递生成样本。

扩散过程包括前向过程和反向过程。在前向过程中，通过逐步添加高斯噪声将干净样本转化为噪声样本。而在反向过程中，模型则尝试去除噪声，逐步恢复出干净的样本。

DDPM（去噪扩散概率模型）是扩散模型的一种，它通过遵循反向过程的所有步骤来生成样本。在参数化反向过程的均值时，网络可以预测原始样本、预测噪声样本，或者直接预测均值。

DDIM（去噪扩散隐式模型）是 DDPM 的一种变体，它在较少的步骤中提供更好的样本质量，并且允许起始噪声和生成样本之间的确定性匹配。DDIM 通过引入一个控制随机组成部分幅度的新参数 $\eta$ 来实现这一点。

扩散模型的训练相对直接，它涉及到生成噪声样本、计算噪声的近似值，并通过优化参数来最小化误差。采样过程从扩散过程的最后步骤开始，逐步预测噪声并恢复出干净的样本。

扩散模型在图像生成领域展现出了巨大的潜力，它们提供了一种新的生成高质量图像样本的方法。
