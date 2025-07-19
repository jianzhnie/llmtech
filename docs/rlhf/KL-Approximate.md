# KL散度的近似计算

本文讨论KL散度的蒙特卡洛近似方法。KL散度的定义如下：

$$
KL[q,p] = \sum_x q(x) \log \frac{q(x)}{p(x)} = \mathbb{E}_{x \sim q} \left[ \log \frac{q(x)}{p(x)} \right]
$$

本文将解释一种我在代码中常用的技巧：通过从 $q$ 中采样 $x$，用 $\frac{1}{2} (\log p(x) - \log q(x))^2$ 的样本均值来近似 $KL[q,p]$，而不是使用更标准的 $\log \frac{q(x)}{p(x)}$。本文将解释为什么这种表达式是一个良好（尽管有偏）的KL散度estimator，以及如何使其无偏且保持低方差。

## 计算KL散度的选项

我们计算KL散度的能力取决于对 $p$ 和 $q$ 的访问权限。这里假设我们可以计算任意 $x$ 的概率（或概率密度）$p(x)$ 和 $q(x)$，但无法解析地计算对 $x$ 的求和。为什么无法解析计算呢？

1. **计算量或内存需求过高**：精确计算需要过多的计算资源或内存。
2. **无闭式表达式**：某些情况下，KL散度没有闭式表达式。
3. **简化代码**：有时我们只需存储对数概率，而不需要整个分布。如果KL散度仅用作诊断工具（例如在强化学习中），这是一个合理的选择。

## 蒙特卡洛估计

估计求和或积分的常见策略是使用蒙特卡洛估计。给定从 $q$ 中采样的样本 $x_1, x_2, \dots$，如何构建一个良好的estimator？

一个好的estimator应是无偏的（均值正确）且具有低方差。我们知道，一个无偏estimator（在从 $q$ 中采样的情况下）是 $\log \frac{q(x)}{p(x)}$。然而，它的方差较高，因为对于一半的样本它是负值，而KL散度始终为正。我们称这种朴素estimator为 $k_1 = \log \frac{q(x)}{p(x)} = -\log r$，其中我们定义了比率 $r = \frac{p(x)}{q(x)}$，这个比率将在后续计算中频繁出现。

另一种 estimator是 $\frac{1}{2} (\log \frac{p(x)}{q(x)})^2 = \frac{1}{2} (\log r)^2$，它具有较低的方差但有偏。我们称这种estimator为 $k_2$。直观上，$k_2$ 似乎更好，因为每个样本都告诉我们 $p$ 和 $q$ 之间的距离，且始终为正。实验表明，$k_2$ 的方差确实远低于 $k_1$，并且偏差也非常低。（我们将在下面的实验中展示这一点。）

## 为什么 $k_2$ 的偏差较低？

$k_2$ 的期望是一个 $f$-散度。$f$-散度的定义为：

$$
D_f(p, q) = \mathbb{E}_{x \sim q} \left[ f\left( \frac{p(x)}{q(x)} \right) \right]
$$

其中 $f$ 是一个凸函数。KL散度和其他许多著名的概率距离都是 $f$-散度。这里有一个关键的非显然事实：当 $q$ 接近 $p$ 时，所有具有可微 $f$ 的 $f$-散度在二阶近似下都类似于KL散度。具体来说，对于一个参数化分布 $p_\theta$，

$$
D_f(p_0, p_\theta) = \frac{f''(1)}{2} \theta^T F \theta + O(\theta^3)
$$

其中 $F$ 是 $p_\theta$ 在 $p_\theta = p_0$ 处的Fisher信息矩阵。

$\mathbb{E}_q[k_2] = \mathbb{E}_q \left[ \frac{1}{2} (\log r)^2 \right]$ 是 $f$-散度，其中 $f(x) = \frac{1}{2} (\log x)^2$，而 $KL[q,p]$ 对应于 $f(x) = -\log x$。容易验证两者都有 $f''(1) = 1$，因此当 $p \approx q$ 时，两者看起来像相同的二次距离函数。

## 无偏且低方差的KL散度estimator

是否可以写出一个无偏且低方差的KL散度estimator？降低方差的通用方法是使用控制变量。即，取 $k_1$ 并添加一个期望为零但与 $k_1$ 负相关的项。唯一保证期望为零的量是 $\frac{p(x)}{q(x)} - 1 = r - 1$。因此，对于任意 $\lambda$，表达式 $-\log r + \lambda (r - 1)$ 是 $KL[q,p]$ 的无偏estimator。我们可以通过计算最小化该estimator的方差来求解 $\lambda$，但不幸的是，得到的表达式依赖于 $p$ 和 $q$，且难以解析计算。

然而，我们可以使用更简单的策略选择一个好的 $\lambda$。注意到由于 $\log$ 是凹函数，$\log(x) \leq x - 1$。因此，如果我们令 $\lambda = 1$，上述表达式保证为正。它测量了 $\log(x)$ 与其切线之间的垂直距离。这使我们得到estimator $k_3 = (r - 1) - \log r$。

这种通过凸函数与其切平面之间的距离来测量距离的思想在许多地方出现。它被称为Bregman散度，具有许多优美的性质。

## 推广到其他 $f$-散度

我们可以将上述思想推广到任何 $f$-散度，特别是另一种KL散度 $KL[p,q]$（注意这里 $p$ 和 $q$ 的位置交换了）。由于 $f$ 是凸函数，且 $\mathbb{E}_q[r] = 1$，以下表达式是 $f$-散度的estimator：

$$
f(r) - f'(1)(r - 1)
$$

由于 $f$ 位于其切线之上，该estimator始终为正。对于 $KL[p,q]$，$f(x) = x \log x$，其 $f'(1) = 1$，因此我们得到estimator:  $r \log r - (r - 1)$。

## 总结

我们有以下estimator（对于从 $q$ 中采样的 $x$，且 $r = \frac{p(x)}{q(x)}$）：

- $KL[p,q]$：$r \log r - (r - 1)$
- $KL[q,p]$：$(r - 1) - \log r$

现在让我们比较三种 $KL[q,p]$ estimator的偏差和方差。假设 $q = \mathcal{N}(0, 1)$，$p = \mathcal{N}(0.1, 1)$，此时真实的KL散度为 $0.005$。

| estimator | 偏差/真实值 | 标准差/真实值 |
| --------- | ------ | ------- |
| $k_1$     | 0      | 20      |
| $k_2$     | 0.002  | 1.42    |
| $k_3$     | 0      | 1.42    |

注意到 $k_2$ 的偏差非常低，仅为 $0.2\%$。

现在尝试一个更大的真实KL散度。$p = \mathcal{N}(1, 1)$ 时，真实KL散度为 $0.5$。

| estimator | 偏差/真实值 | 标准差/真实值 |
| --------- | ------ | ------- |
| $k_1$     | 0      | 2       |
| $k_2$     | 0.25   | 1.73    |
| $k_3$     | 0      | 1.7     |

这里 $k_2$ 的偏差较大，而 $k_3$ 在无偏的同时具有更低的标准差，因此它似乎是一个严格更好的estimator。

以下是用于获取这些结果的代码：

```python
import torch.distributions as dis

p = dis.Normal(loc=0, scale=1)
q = dis.Normal(loc=0.1, scale=1)
x = q.sample(sample_shape=(10_000_000,))
truekl = dis.kl_divergence(p, q)
print("true", truekl)

logr = p.log_prob(x) - q.log_prob(x)
k1 = -logr
k2 = logr ** 2 / 2
k3 = (logr.exp() - 1) - logr

for k in (k1, k2, k3):
    print((k.mean() - truekl) / truekl, k.std() / truekl)
```

### 参考内容：

- http://joschu.net/blog/kl-approx.html
- https://zhuanlan.zhihu.com/p/25208314999
