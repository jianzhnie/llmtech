
# Policy gradient theorem的证明


如今, 强化学习基本都采用参数化的神经网络来学习一个策略, 而神经网络一般是通过梯度下降法或 者各种变种来优化的, 因此, 获取累积回报关于策略的梯度至关重要。本节会给大家推导策略梯度的 表达式, 并介绍实际训练中是如何采样近似该表达式的。

这里我们首先额外引入一下动作价值函数的定义

$$
Q\left(s_{t}, a_{t}\right)=E_{s_{t+1}, a_{t+1}}, \ldots\left[\sum_{l=0}^{\infty} \gamma^{l} r^{t+l}\right]
$$

即在状态 $s_{t}$ 下采用动作 $a_{t}$ 后, 后续动作服从策略 $\pi$ 的情况下的累积期望回报, 其中 $\gamma \in(0,1)$ 是折 扣因子。

接着, 我们将策略梯度计算过程详细展开如下:

$$
\begin{aligned}
& \nabla_{\theta} J(\theta) \\
& =\nabla_{\theta} V\left(s_{0}\right) \\
& =\nabla\left[\sum_{a_{0}} \pi\left(a_{0} \mid s_{0}\right) Q_{\pi}\left(s_{0}, a_{0}\right)\right] \\
& =\sum_{a_{0}}\left[\nabla \pi\left(a_{0} \mid s_{0}\right) Q_{\pi}\left(s_{0}, a_{0}\right)+\pi\left(a_{0} \mid s_{0}\right) \nabla Q_{\pi}\left(s_{0}, a_{0}\right)\right] \\
& =\sum_{a_{0}}\left[\nabla \pi\left(a_{0} \mid s_{0}\right) Q_{\pi}\left(s_{0}, a_{0}\right)+\pi\left(a_{0} \mid s_{0}\right) \nabla \sum_{s_{1}, r_{1}} p\left(s_{1}, r_{1} \mid s_{0}, a_{0}\right)\left(r_{1}+\gamma V\left(s_{1}\right)\right)\right] \\
& =\sum_{a_{0}} \nabla \pi\left(a_{0} \mid s_{0}\right) Q_{\pi}\left(s_{0}, a_{0}\right)+\sum_{a_{0}} \pi\left(a_{0} \mid s_{0}\right) \sum_{s_{1}} p\left(s_{1} \mid s_{0}, a_{0}\right) \cdot \gamma \nabla V\left(s_{1}\right) \\
& =\sum_{a_{0}} \nabla \pi\left(a_{0} \mid s_{0}\right) Q_{\pi}\left(s_{0}, a_{0}\right) \\
& +\sum_{a_{0}}^{a_{0}} \pi\left(a_{0} \mid s_{0}\right) \sum_{s_{1}} p\left(s_{1} \mid s_{0}, a_{0}\right) \cdot \gamma \sum_{a_{1}} \nabla \pi\left(a_{1} \mid s_{1}\right) Q_{\pi}\left(s_{1}, a_{1}\right) \\
& +\sum_{a_{0}}^{a_{0}} \pi\left(a_{0} \mid s_{0}\right) \sum_{s_{1}} p\left(s_{1} \mid s_{0}, a_{0}\right) \cdot \gamma \sum_{a_{1}} \pi\left(a_{1} \mid s_{1}\right) \sum_{s_{2}} p\left(s_{2} \mid s_{1}, a_{1}\right) \gamma \nabla V\left(s_{2}\right) \\
& =\sum_{a_{0}}^{a_{0}} \nabla \pi\left(a_{0} \mid s_{0}\right) Q_{\pi}\left(s_{0}, a_{0}\right) \\
& +\sum_{a_{0}}^{a_{0}} \pi\left(a_{0} \mid s_{0}\right) \sum_{s_{1}} p\left(s_{1} \mid s_{0}, a_{0}\right) \cdot \gamma \sum_{a_{1}} \nabla \pi\left(a_{1} \mid s_{1}\right) Q_{\pi}\left(s_{1}, a_{1}\right)+\cdots \\
& =\sum_{s_{0}} \operatorname{Pr}\left(s_{0} \rightarrow s_{0}, 0, \pi\right) \sum_{a_{0}} \nabla \pi\left(a_{0} \mid s_{0}\right) \gamma^{0} Q_{\pi}\left(s_{0}, a_{0}\right) \\
& +\sum_{s_{1}}^{a_{1}} \operatorname{Pr}\left(s_{0} \rightarrow s_{1}, 1, \pi\right) \sum_{a_{1}} \nabla \pi\left(a_{1} \mid s_{1}\right) \gamma^{1} Q_{\pi}\left(s_{1}, a_{1}\right)+\cdots
\end{aligned}
$$



$$
\begin{aligned}
& =\sum_{s_{0}} \operatorname{Pr}\left(s_{0} \rightarrow s_{0}, 0, \pi\right) \sum_{a_{0}} \pi\left(a_{0} \mid s_{0}\right)\left[\gamma^{0} Q_{\pi}\left(s_{0}, a_{0}\right) \nabla \log \pi\left(a_{0} \mid s_{0}\right)\right] \\
& +\sum_{s_{1}}^{s_{1}} \operatorname{Pr}\left(s_{0} \rightarrow s_{1}, 1, \pi\right) \sum_{a_{1}} \pi\left(a_{1} \mid s_{1}\right)\left[\gamma^{1} Q_{\pi}\left(s_{1}, a_{1}\right) \nabla \log \pi\left(a_{1} \mid s_{1}\right)\right]+\cdots \\
& =\sum_{t=0}^{\infty} \sum_{s_{t}} \operatorname{Pr}\left(s_{0} \rightarrow s_{t}, t, \pi\right) \sum_{a_{t}} \pi\left(a_{t} \mid s_{t}\right)\left[\gamma^{t} Q_{\pi}\left(s_{t}, a_{t}\right) \nabla \log \pi\left(a_{t} \mid s_{t}\right)\right]
\end{aligned}
$$

其中 $\operatorname{Pr}\left(s_{0} \rightarrow s_{t}, t, \pi\right)$ 代表: 从状态 $s_{0}$ 出发, 且按照策略 $\pi$ 与环境交互（rollout）, 在 $\mathrm{t}$ 时刻到达 状态 $s_{t}$ 的概率。

通过上述的推导, 我们就得到了无限长时间步下的策略梯度的表达式, 对于有限长时间步的环境, 我 们可以做一个简单的转化, 把它变成无限长, 从而同样适用上述公式。假设时间步长度为 $T$, 对于所 有可能出现在最后一步的状态 $s_{T-1}$, 我们定义:

1. 从 $s_{T-1}$ 出发, 不论采取什么动作, 一定会跳转到一个虚拟的吸收态 $s_{T}$, 并返回奖励值0。

2. 从 $s_{T}$ 出发, 不论采取什么动作, 一定会跳转回这个虚拟的吸收态 $s_{T}$, 并返回奖励值0。 由此将有限长的时间步扩展到了无限长, 因为环境会陷入到 $s_{T}$ 的死循环中。

不过, 上式实际上很难优化, 要求遍历整个状态空间和时间步空间。具体来说, 该式要求计算每个时 间步上到达每个状态的概率。一方面, 这在计算成本上是无法容忍的; 另一方面, 我们在绝大多数情 况下, 无法获得环境的转移概率，因此无法计算特定时间步下整个状态空间上的概率分布。

那怎么办, 我们可以用 Monte Carlo 方法, 通过采样来逼近上面的策略梯度公式。这里先把上式转化 为期望的形式:

$$
\begin{aligned}
\text { 上式 } & =\sum_{t=0}^{\infty} \sum_{s_{t}} \operatorname{Pr}\left(s_{0} \rightarrow s_{t}, t, \pi\right) \sum_{a_{t}} \pi\left(a_{t} \mid s_{t}\right)\left[\gamma^{t} Q_{\pi}\left(s_{t}, a_{t}\right) \nabla \log \pi\left(a_{t} \mid s_{t}\right)\right] \\
& =\sum_{t=0}^{\infty} E_{s_{t}} \sum_{a_{t}} \pi\left(a_{t} \mid s_{t}\right)\left[\gamma^{t} Q_{\pi}\left(s_{t}, a_{t}\right) \nabla \log \pi\left(a_{t} \mid s_{t}\right)\right] \\
& =\sum_{t=0}^{\infty} E_{s_{t}} E_{a_{t}}\left[\gamma^{t} Q_{\pi}\left(s_{t}, a_{t}\right) \nabla \log \pi\left(a_{t} \mid s_{t}\right)\right] \\
& =\sum_{t=0}^{\infty} E_{s_{t}, a_{t}}\left[\gamma^{t} Q_{\pi}\left(s_{t}, a_{t}\right) \nabla \log \pi\left(a_{t} \mid s_{t}\right)\right] \\
& =E_{s_{0}, a_{0}, s_{1}, a_{1}, \cdots} \sum_{t=0}^{\infty}\left[\gamma^{t} Q_{\pi}\left(s_{t}, a_{t}\right) \nabla \log \pi\left(a_{t} \mid s_{t}\right)\right] \\
& =E_{\tau} \sum_{t=0}^{\infty}\left[\gamma^{t} Q_{\pi}\left(s_{t}, a_{t}\right) \nabla \log \pi\left(a_{t} \mid s_{t}\right)\right]
\end{aligned}
$$

其中 $\tau=\left[s_{0}, a_{0}, s_{1}, a_{1}, \cdots\right]$ 是按照策略 $\pi$ rollout 出来的状态动作的轨迹。可以看出, 将 $\gamma^{t} Q_{\pi}\left(s_{t}, a_{t}\right) \nabla \log \pi\left(a_{t} \mid s_{t}\right)$ 这一项, 先在时间步 $t$ 上求和, 再关于轨迹 $\tau$ 取期望, 就得到了策略 梯度。至此, Monte Carlo方法就可以很简单地结合进来, 我们先将将 $E_{\tau}$ 替换为采样 $N$ 条轨迹 $\left[\tau^{1}, \cdots, \tau^{N}\right]$ 。并定义其中第 $n$ 条轨迹为 $\tau^{n}=\left\langle s_{0}^{n}, a_{0}^{n}, r_{0}^{n}, \cdots, s_{T_{n}-1}^{n}, a_{T_{n}-1}^{n}, r_{T_{n}-1}^{n}\right\rangle$, 轨迹长度为 $T_{n}$ 。最后对结果取平均:

$$
\begin{gathered}
E_{\tau} \sum_{t=0}^{\infty}\left[\gamma^{t} Q_{\pi}\left(s_{t}, a_{t}\right) \nabla \log \pi\left(a_{t} \mid s_{t}\right)\right] \\
=\frac{1}{N} \sum_{n=1}^{N} \sum_{t=0}^{T_{n}-1}\left[\gamma^{t} Q_{\pi}\left(s_{t}^{n}, a_{t}^{n}\right) \nabla \log \pi\left(a_{t}^{n} \mid s_{t}^{n}\right)\right]
\end{gathered}
$$

注意到 $Q_{\pi}\left(s_{t}^{n}, a_{t}^{n}\right)=E_{s_{t+1}^{n}, a_{t+1}^{n}, s_{t+2}^{n}, a_{t+2}^{n}, \cdots \mid s_{t}^{n}, a_{t}^{n}}\left[\sum_{l=t}^{T_{n}-1} \gamma^{l} r_{l}^{n}\right]$, 因此从期望角度二者也是可以替换的。 当我们的算法没有显式地估计 $Q_{\pi}\left(s_{t}^{n}, a_{t}^{n}\right)$ 时, 可以定义 $G_{t}\left(\tau^{n}\right)=\sum_{l=t}^{T_{n}-1} \gamma^{l} r_{l}^{n}$ （即最朴素的策略梯 度）, 并用它替换 $Q_{\pi}\left(s_{t}^{n}, a_{t}^{n}\right)$, 另外再将式中的 $\gamma^{t}$ 省略掉（是一种更简便的近似）, 就得到了实际 使用的策略梯度公式:

$$
\frac{1}{N} \sum_{n=1}^{N} \sum_{t=0}^{T_{n}-1}\left[\gamma^{t} G_{t}\left(\tau^{n}\right) \nabla \log \pi\left(a_{t}^{n} \mid s_{t}^{n}\right)\right]
$$
