
## 第 8 章 带基线的策略梯度方法

上一章推导出策略梯度, 并介绍了两种策略梯度方法_REINFORCE 和 actor-critic。 虽然上一章的方法在理论上是正确的, 但是在实践中效果并不理想。本章介绍的带基线 的策略梯度 (policy gradient with baseline) 可以大幅提升策略梯度方法的表现。使用基线 (baseline) 之后, REINFORCE 变成 REINFORCE with baseline, actor-critic 变成 advantage actor-critic (A2C)。

## $8.1$ 策略梯度中的基线

首先回顾上一章的内容。策略学习通过最大化目标函数 $J(\boldsymbol{\theta})=\mathbb{E}_{S}\left[V_{\pi}(S)\right]$, 训练出 策略网络 $\pi(a \mid s ; \boldsymbol{\theta})$ 。可以用策略梯度 $\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta})$ 来更新参数 $\boldsymbol{\theta}$ :

$$
\boldsymbol{\theta}_{\text {new }} \leftarrow \boldsymbol{\theta}_{\text {now }}+\beta \cdot \nabla_{\boldsymbol{\theta}} J\left(\boldsymbol{\theta}_{\text {now }}\right) .
$$

策略梯度定理证明：

$$
\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta})=\mathbb{E}_{S}\left[\mathbb{E}_{A \sim \pi(\cdot \mid S ; \boldsymbol{\theta})}\left[Q_{\pi}(S, A) \cdot \nabla_{\boldsymbol{\theta}} \ln \pi(A \mid S ; \boldsymbol{\theta})\right]\right]
$$

上一章中, 我们对策略梯度 $\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta})$ 做近似, 推导出 REINFORCE 和 actor-critic ; 两种方 法区别在于具体如何做近似。

### 1 基线 (Baseline)

基于策略梯度公式 (8.1) 得出的 REINFORCE 和 actor-critic 方法效果通常不好。只需 对策略梯度公式 (8.1) 做一个微小的改动, 就能大幅提升表现：把 $b$ 作为动作价值函数 $Q_{\pi}(S, A)$ 的基线 (baseline), 用 $Q_{\pi}(S, A)-b$ 替换掉 $Q_{\pi}$ 。设 $b$ 是任意的函数, 只要不依 赖于动作 $A$ 就可以, 例如 $b$ 可以是状态价值函数 $V_{\pi}(S)$ 。

## 定理 8.1. 带基线的策略梯度定理

![](https://cdn.mathpix.com/cropped/2023_02_03_f46f5cf0e4de5b9996dcg-133.jpg?height=60&width=1356&top_left_y=2037&top_left_x=293)
线, 对策略梯度没有影响:

$$
\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta})=\mathbb{E}_{S}\left[\mathbb{E}_{A \sim \pi(\cdot \mid S ; \boldsymbol{\theta})}\left[\left(Q_{\pi}(S, A)-b\right) \cdot \nabla_{\boldsymbol{\theta}} \ln \pi(A \mid S ; \boldsymbol{\theta})\right]\right] .
$$

定理 $8.1$ 说明 $b$ 的取值不影响策略梯度的正确性。不论是让 $b=0$ 还是让 $b=V_{\pi}(S)$, 对期望的结果毫无影响, 期望的结果都会等于 $\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta})$ 。其原因在于

$$
\mathbb{E}_{S}\left[\mathbb{E}_{A \sim \pi(\cdot \mid S ; \boldsymbol{\theta})}\left[b \cdot \nabla_{\boldsymbol{\theta}} \ln \pi(A \mid S ; \boldsymbol{\theta})\right]\right]=0 .
$$

定理的证明放到第 $8.4$ 节, 建议对数学感兴趣的读者阅读。

定理中的策略梯度表示成了期望的形式, 我们对期望做蒙特卡洛近似。从环境中观 测到一个状态 $s$, 然后根据策略网络抽样得到 $a \sim \pi(\cdot \mid s ; \boldsymbol{\theta})$ 。那么策略梯度 $\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta})$ 可以 近似为下面的随机梯度：

$$
\boldsymbol{g}_{b}(s, a ; \boldsymbol{\theta})=\left[Q_{\pi}(s, a)-b\right] \cdot \nabla_{\boldsymbol{\theta}} \ln \pi(a \mid s ; \boldsymbol{\theta}) .
$$

不论 $b$ 的取值是 0 还是 $V_{\pi}(s)$, 得到的随机梯度 $\boldsymbol{g}_{b}(s, a ; \boldsymbol{\theta})$ 都是 $\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta})$ 的无偏估计:

$$
\text { Bias }=\mathbb{E}_{S, A}\left[\boldsymbol{g}_{b}(S, A ; \boldsymbol{\theta})\right]-\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta})=\mathbf{0} .
$$

虽然 $b$ 的取值对 $\mathbb{E}_{S, A}\left[\boldsymbol{g}_{b}(S, A ; \boldsymbol{\theta})\right]$ 毫无影响, 但是 $b$ 对随机梯度 $\boldsymbol{g}_{b}(s, a ; \boldsymbol{\theta})$ 是有影响的。 用不同的 $b$, 得到的方差

$$
\operatorname{Var}=\mathbb{E}_{S, A}\left[\left\|\boldsymbol{g}_{b}(S, A ; \boldsymbol{\theta})-\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta})\right\|^{2}\right]
$$

会有所不同。如果 $b$ 很接近 $Q_{\pi}(s, a)$ 关于 $a$ 的均值, 那么方差会比较小。因此, $b=V_{\pi}(s)$ 是很好的基线。

### 2 基线的直观解释

策略梯度公式 (8.1) 期望中的 $Q_{\pi}(S, A) \cdot \nabla_{\boldsymbol{\theta}} \ln \pi(A \mid S ; \boldsymbol{\theta})$ 的意义是什么呢? 以图 $8.1$ 中的左图为例。给定状态 $s_{t}$, 动作空间是 $\mathcal{A}=\{$ 左, 右, 上\}, 动作价值函数给每个动作打 分：

$$
Q_{\pi}\left(s_{t}, \text { 左 }\right)=80, \quad Q_{\pi}\left(s_{t} \text {, 右 }\right)=-20, \quad Q_{\pi}\left(s_{t}, \text { 上 }\right)=180,
$$

这些分值会乘到梯度 $\nabla_{\boldsymbol{\theta}} \ln \pi(A \mid S ; \boldsymbol{\theta})$ 上。在做完梯度上升之后, 新的策略会倾向于分值 高的动作。

- 动作价值 $Q_{\pi}\left(s_{t}\right.$, 上 $)=180$ 很大, 说明基于状态 $s_{t}$ 选择动作“上” 是很好的决策。让 梯度 $\nabla_{\boldsymbol{\theta}} \ln \pi\left(\right.$ 上 $\left.\mid s_{t} ; \boldsymbol{\theta}\right)$ 乘以大的系数 $Q_{\pi}\left(s_{t}\right.$, 上 $)=180$, 那么做梯度上升更新 $\boldsymbol{\theta}$ 之 后, 会让 $\pi\left(\right.$ 上 $\left.\mid s_{t} ; \boldsymbol{\theta}\right)$ 变大, 在状态 $s_{t}$ 的情况下更倾向于动作“上”。

- 相反, $Q_{\pi}\left(s_{t}\right.$, 右 $)=-20$ 说明基于状态 $s_{t}$ 选择动作 “右” 是糟糕的决策。让梯度 $\nabla_{\boldsymbol{\theta}} \ln \pi$ (右 $\left.\mid s_{t} ; \boldsymbol{\theta}\right)$ 乘以负的系数 $Q_{\pi}\left(s_{t}\right.$, 右 $)=-20$, 那么做梯度上升更新 $\boldsymbol{\theta}$ 之后, 会让 $\pi\left(\right.$ 右 $\left.\mid s_{t} ; \boldsymbol{\theta}\right)$ 变小, 在状态 $s_{t}$ 的情况下选择动作 “右”的概率更小。
![](https://cdn.mathpix.com/cropped/2023_02_03_f46f5cf0e4de5b9996dcg-134.jpg?height=382&width=1376&top_left_y=2137&top_left_x=404)

图 8.1: 动作空间是 $\mathcal{A}=\{$ 左, 右, 上 $\}$ 。左图纵轴表示动作价值 $Q_{\pi}(s, a)$ 。右图纵轴表示动作价值 减去基线 $Q_{\pi}(s, a)-b$, 其中基线 $b=60$ 。 根据上述分析, 我们在乎的是动作价值 $Q_{\pi}\left(s_{t}\right.$, 左)、 $Q_{\pi}\left(s_{t}\right.$, 右 $) 、 Q_{\pi}\left(s_{t}\right.$, 上) 三者的相 对大小, 而非绝对大小。如果给三者都减去 $b=60$, 那么三者的相对大小是不变的; 动 作“上”仍然是最好的, 动作“右”仍然是最差的。见图 $8.1$ 中的右图。因此

$$
\left[Q_{\pi}\left(s_{t}, a_{t}\right)-b\right] \cdot \nabla_{\boldsymbol{\theta}} \ln \pi(A \mid S ; \boldsymbol{\theta})
$$

依然能指导 $\boldsymbol{\theta}$ 做调整, 使得 $\pi\left(\right.$ 上 $\left.\mid s_{t} ; \boldsymbol{\theta}\right)$ 变大, 而 $\pi\left(\right.$ 右 $\left.\mid s_{t} ; \boldsymbol{\theta}\right)$ 变小。

## 2 带基线的 REINFORCE 算法

上一节推导出了带基线的策略梯度, 并且对策略梯度做了蒙特卡洛近似。本节中, 我 们使用状态价值 $V_{\pi}(s)$ 作基线, 得到策略梯度的一个无偏估计：

$$
\boldsymbol{g}(s, a ; \boldsymbol{\theta})=\left[Q_{\pi}(s, a)-V_{\pi}(s)\right] \cdot \nabla_{\boldsymbol{\theta}} \ln \pi(a \mid s ; \boldsymbol{\theta}) .
$$

我们在第 $7.4$ 节中学过 REINFORCE, 它使用实际观测的回报 $u$ 来代替动作价值 $Q_{\pi}(s, a)$ 。 此处我们同样用 $u$ 代替 $Q_{\pi}(s, a)$ 。此外, 我们还用一个神经网络 $v(s ; \boldsymbol{w})$ 近似状态价值函 数 $V_{\pi}(s)$ 。这样一来, $\boldsymbol{g}(s, a ; \boldsymbol{\theta})$ 就被近似成了:

$$
\tilde{\boldsymbol{g}}(s, a ; \boldsymbol{\theta})=[u-v(s ; \boldsymbol{w})] \cdot \nabla_{\boldsymbol{\theta}} \ln \pi(a \mid s ; \boldsymbol{\theta}) .
$$

可以用 $\tilde{\boldsymbol{g}}(s, a ; \boldsymbol{\theta})$ 作为策略梯度 $\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta})$ 的近似, 更新策略网络参数:

$$
\boldsymbol{\theta} \leftarrow \boldsymbol{\theta}+\beta \cdot \tilde{\boldsymbol{g}}(s, a ; \boldsymbol{\theta})
$$

### 1 策略网络和价值网络

带基线的 REINFORCE 需要两个神经网络：策略网络 $\pi(a \mid s ; \boldsymbol{\theta})$ 和价值网络 $v(s ; \boldsymbol{w})$; 神经网络结构如图 $8.2$ 和 $8.3$ 所示。策略网络与之前章节一样: 输入是状态 $s$, 输出是一 个向量, 每个元素表示一个动作的概率。

![](https://cdn.mathpix.com/cropped/2023_02_03_f46f5cf0e4de5b9996dcg-136.jpg?height=212&width=1433&top_left_y=1533&top_left_x=377)

策略网络

图 8.2: 策略网络 $\pi(a \mid s ; \boldsymbol{\theta})$ 的神经网络结构。输入是状态 $s$, 输出是动作空间中每个动作的概率 值。举个例子, 动作空间是 $\mathcal{A}=\{$ 左, 右, 上 $\}$, 策略网络的输出是三个概率值: $\pi($ 左 $\mid s ; \boldsymbol{\theta})=0.2$, $\pi($ 右 $\mid s ; \boldsymbol{\theta})=0.1, \pi($ 上 $\mid s ; \boldsymbol{\theta})=0.7$ 。

![](https://cdn.mathpix.com/cropped/2023_02_03_f46f5cf0e4de5b9996dcg-136.jpg?height=353&width=1203&top_left_y=2082&top_left_x=492)

图 8.3: 价值网络 $v(s ; \boldsymbol{w})$ 的结构。输入是状态 $s$; 输出是状态的价值。

此处的价值网络 $v(s ; \boldsymbol{w})$ 与之前使用的价值网络 $q(s, a ; \boldsymbol{w})$ 区别较大。此处的 $v(s ; \boldsymbol{w})$ 是对状态价值 $V_{\pi}$ 的近似, 而非对动作价值 $Q_{\pi}$ 的近似。 $v(s ; \boldsymbol{w})$ 的输入是状态 $s$, 输出是 一个实数, 作为基线。策略网络和价值网络的输入都是状态 $s$, 因此可以让两个神经网络 共享卷积网络的参数, 这是编程实现中常用的技巧。

虽然带基线的 REINFORCE 有一个策略网络和一个价值网络, 但是这种方法不是 actor-critic。价值网络没有起到 “评委”的作用, 只是作为基线而已, 目的在于降低方差, 加 速收玫。真正帮助策略网络（演员）改进参数 $\theta$ （演员的演技）的不是价值网络, 而是实 际观测到的回报 $u$ 。

### 2 算法的推导

训练策略网络的方法是近似的策略梯度上升。从 $t$ 时刻开始, 智能体完成一局游戏, 观测到全部奖励 $r_{t}, r_{t+1}, \cdots, r_{n}$, 然后计算回报 $u_{t}=\sum_{k=t}^{n} \gamma^{k-t} \cdot r_{k}$ 。让价值网络做出预 测 $\widehat{v}_{t}=v\left(s_{t} ; \boldsymbol{w}\right)$, 作为基线。这样就得到了带基线的策略梯度:

$$
\tilde{\boldsymbol{g}}\left(s_{t}, a_{t} ; \boldsymbol{\theta}\right)=\left(u_{t}-\widehat{v}_{t}\right) \cdot \nabla_{\boldsymbol{\theta}} \ln \pi\left(a_{t} \mid s_{t} ; \boldsymbol{\theta}\right) .
$$

它是策略梯度 $\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta})$ 的近似。最后做梯度上升更新 $\boldsymbol{\theta}$ :

$$
\boldsymbol{\theta} \leftarrow \boldsymbol{\theta}+\beta \cdot \tilde{\boldsymbol{g}}\left(s_{t}, a_{t} ; \boldsymbol{\theta}\right) .
$$

这样可以让目标函数 $J(\boldsymbol{\theta})$ 逐渐增大。

训练价值网络的方法是回归（regression)。回忆一下，状态价值是回报的期望：

$$
V_{\pi}\left(s_{t}\right)=\mathbb{E}\left[U_{t} \mid S_{t}=s_{t}\right],
$$

期望消掉了动作 $A_{t}, A_{t+1}, \cdots, A_{n}$ 和状态 $S_{t+1}, \cdots, S_{n}$ 。训练价值网络的目的是让 $v\left(s_{t} ; \boldsymbol{w}\right)$ 拟合 $V_{\pi}\left(s_{t}\right)$, 即拟合 $u_{t}$ 的期望。定义损失函数：

$$
L(\boldsymbol{w})=\frac{1}{2 n} \sum_{t=1}^{n}\left[v\left(s_{t} ; \boldsymbol{w}\right)-u_{t}\right]^{2} .
$$

设 $\widehat{v}_{t}=v\left(s_{t} ; \boldsymbol{w}\right)$ 。损失函数的梯度是:

$$
\nabla_{\boldsymbol{w}} L(\boldsymbol{w})=\frac{1}{n} \sum_{t=1}^{n}\left(\widehat{v}_{t}-u_{t}\right) \cdot \nabla_{\boldsymbol{w}} v\left(s_{t} ; \boldsymbol{w}\right) .
$$

做一次梯度下降更新 $\boldsymbol{w}$ :

$$
\boldsymbol{w} \leftarrow \boldsymbol{w}-\alpha \cdot \nabla_{\boldsymbol{w}} L(\boldsymbol{w})
$$

### 3 训练流程

当前策略网络的参数是 $\boldsymbol{\theta}_{\text {now, }}$ 价值网络的参数是 $\boldsymbol{w}_{\text {now }}$ 执行下面的步骤, 对参数做 一轮更新。

![](https://cdn.mathpix.com/cropped/2023_02_03_f46f5cf0e4de5b9996dcg-137.jpg?height=63&width=1356&top_left_y=2370&top_left_x=287)

$$
s_{1}, a_{1}, r_{1}, \quad s_{2}, a_{2}, r_{2}, \quad \cdots, \quad s_{n}, a_{n}, r_{n} .
$$

2. 计算所有的回报：

$$
u_{t}=\sum_{k=t}^{n} \gamma^{k-t} \cdot r_{k}, \quad \forall t=1, \cdots, n
$$

3. 让价值网络做预测:

$$
\widehat{v}_{t}=v\left(s_{t} ; \boldsymbol{w}_{\text {now }}\right), \quad \forall t=1, \cdots, n .
$$

4. 计算误差 $\delta_{t}=\widehat{v}_{t}-u_{t}, \forall t=1, \cdots, n$ 。

5. 用 $\left\{s_{t}\right\}_{t=1}^{n}$ 作为价值网络输入, 做反向传播计算:

$$
\nabla_{\boldsymbol{w}} v\left(s_{t} ; \boldsymbol{w}_{\text {now }}\right), \quad \forall t=1, \cdots, n .
$$

6. 更新价值网络参数:

$$
\boldsymbol{w}_{\text {new }} \leftarrow \boldsymbol{w}_{\text {now }}-\alpha \cdot \sum_{t=1}^{n} \delta_{t} \cdot \nabla_{\boldsymbol{w}} v\left(s_{t} ; \boldsymbol{w}_{\text {now }}\right) .
$$

7. 用 $\left\{\left(s_{t}, a_{t}\right)\right\}_{t=1}^{n}$ 作为数据, 做反向传播计算:

$$
\nabla_{\boldsymbol{\theta}} \ln \pi\left(a_{t} \mid s_{t} ; \boldsymbol{\theta}_{\text {now }}\right), \quad \forall t=1, \cdots, n .
$$

8. 做随机梯度上升更新策略网络参数：

$$
\boldsymbol{\theta}_{\text {new }} \leftarrow \boldsymbol{\theta}_{\text {now }}-\beta \cdot \sum_{t=1}^{n} \gamma^{t-1} \cdot \underbrace{\delta_{t} \cdot \nabla_{\boldsymbol{\theta}} \ln \pi\left(a_{t} \mid s_{t} ; \boldsymbol{\theta}_{\text {now }}\right)}_{\text {负的近似梯度 }-\tilde{\boldsymbol{g}}\left(s_{t}, a_{t} ; \boldsymbol{\theta}_{\text {now }}\right)} \text {. }
$$



### Advantage Actor-Critic (A2C)

之前我们推导出了带基线的策略梯度, 并且对策略梯度做了蒙特卡洛近似, 得到策 略梯度的一个无偏估计：

$$
\boldsymbol{g}(s, a ; \boldsymbol{\theta})=[\underbrace{Q_{\pi}(s, a)-V_{\pi}(s)}_{\text {优执函数 }}] \cdot \nabla_{\boldsymbol{\theta}} \ln \pi(a \mid s ; \boldsymbol{\theta}) .
$$

公式中的 $Q_{\pi}-V_{\pi}$ 被称作优势函数 (advantage function)。因此, 基于上面公式得到的 actor-critic 方法被称为 advantage actor-critic, 缩写 $\mathrm{A} 2 \mathrm{C}$ 。

$\mathrm{A} 2 \mathrm{C}$ 属于 actor-critic 方法。有一个策略网络 $\pi(a \mid s ; \boldsymbol{\theta})$, 相当于演员, 用于控制智能体 运动。还有一个价值网络 $v(s ; \boldsymbol{w})$, 相当于评委, 他的评分可以帮助策略网络（演员) 改 进技术。两个神经网络的结构与上一节中的完全相同, 但是本节和上一节用不同的方法 训练两个神经网络。

### 1 算法推导

训练价值网络：训练价值网络 $v(s ; \boldsymbol{w})$ 的算法是从贝尔曼公式来的:

$$
V_{\pi}\left(s_{t}\right)=\mathbb{E}_{A_{t} \sim \pi\left(\cdot \mid s_{t} ; \boldsymbol{\theta}\right)}\left[\mathbb{E}_{S_{t+1} \sim p\left(\cdot \mid s_{t}, A_{t}\right)}\left[R_{t}+\gamma \cdot V_{\pi}\left(S_{t+1}\right)\right]\right] .
$$

我们对贝尔曼方程左右两边做近似：

- 方程左边的 $V_{\pi}\left(s_{t}\right)$ 可以近似成 $v\left(s_{t} ; \boldsymbol{w}\right) 。 v\left(s_{t} ; \boldsymbol{w}\right)$ 是价值网络在 $t$ 时刻对 $V_{\pi}\left(s_{t}\right)$ 做 出的估计。

- 方程右边的期望是关于当前时刻动作 $A_{t}$ 与下一时刻状态 $S_{t+1}$ 求的。给定当前状态 $s_{t}$, 智能体执行动作 $a_{t}$, 环境会给出奖励 $r_{t}$ 和新的状态 $s_{t+1}$ 。用观测到的 $r_{t} 、 s_{t+1}$ 对期望做蒙特卡洛近似，得到：

$$
r_{t}+\gamma \cdot V_{\pi}\left(s_{t+1}\right) .
$$

- 进一步把公式 (8.3) 中的 $V_{\pi}\left(s_{t+1}\right)$ 近似成 $v\left(s_{t+1} ; \boldsymbol{w}\right)$, 得到

$$
\widehat{y}_{t} \triangleq r_{t}+\gamma \cdot v\left(s_{t+1} ; \boldsymbol{w}\right) .
$$

把它称作 $\mathrm{TD}$ 目标。它是价值网络在 $t+1$ 时刻对 $V_{\pi}\left(s_{t}\right)$ 做出的估计。

$v\left(s_{t} ; \boldsymbol{w}\right)$ 和 $\widehat{y}_{t}$ 都是对动作价值 $V_{\pi}\left(s_{t}\right)$ 的估计。由于 $\widehat{y}_{t}$ 部分基于真实观测到的奖励 $r_{t}$, 我 们认为 $\widehat{y}_{t}$ 比 $v\left(s_{t} ; \boldsymbol{w}\right)$ 更可靠。所以把 $\widehat{y}_{t}$ 固定住, 更新 $\boldsymbol{w}$, 使得 $v\left(s_{t} ; \boldsymbol{w}\right)$ 更接近 $\widehat{y}_{t}$ 。

具体这样更新价值网络参数 $\boldsymbol{w}$ 。定义损失函数

$$
L(\boldsymbol{w}) \triangleq \frac{1}{2}\left[v\left(s_{t} ; \boldsymbol{w}\right)-\widehat{y}_{t}\right]^{2} .
$$

设 $\widehat{v}_{t} \triangleq v\left(s_{t} ; \boldsymbol{w}\right)$ 。损失函数的梯度是 :

$$
\nabla_{\boldsymbol{w}} L(\boldsymbol{w})=\underbrace{\left(\widehat{v}_{t}-\widehat{y}_{t}\right)}_{\mathrm{TD} \text { 误差 } \delta_{t}} \cdot \nabla_{\boldsymbol{w}} v\left(s_{t} ; \boldsymbol{w}\right) .
$$

定义 TD 误差为 $\delta_{t} \triangleq \widehat{v}_{t}-\widehat{y}_{t}$ 。做一轮梯度下降更新 $\boldsymbol{w}$ :

$$
\boldsymbol{w} \leftarrow \boldsymbol{w}-\alpha \cdot \delta_{t} \cdot \nabla_{\boldsymbol{w}} v\left(s_{t} ; \boldsymbol{w}\right)
$$

这样可以让价值网络的预测 $v\left(s_{t} ; \boldsymbol{w}\right)$ 更接近 $\widehat{y}_{t}$ 。

训练策略网络 : A2C 从公式 (8.2) 出发, 对 $\boldsymbol{g}(s, a ; \boldsymbol{\theta})$ 做近似, 记作 $\tilde{\boldsymbol{g}}$, 然后用 $\tilde{\boldsymbol{g}}$ 更 新策略网络参数 $\boldsymbol{\theta}$ 。下面我们做数学推导。回忆一下贝尔曼公式:

$$
Q_{\pi}\left(s_{t}, a_{t}\right)=\mathbb{E}_{S_{t+1} \sim p\left(\cdot \mid s_{t}, a_{t}\right)}\left[R_{t}+\gamma \cdot V_{\pi}\left(S_{t+1}\right)\right] .
$$

把近似策略梯度 $\boldsymbol{g}\left(s_{t}, a_{t} ; \boldsymbol{\theta}\right)$ 中的 $Q_{\pi}\left(s_{t}, a_{t}\right)$ 替换成上面的期望, 得到:

$$
\begin{aligned}
\boldsymbol{g}\left(s_{t}, a_{t} ; \boldsymbol{\theta}\right) & =\left[Q_{\pi}\left(s_{t}, a_{t}\right)-V_{\pi}\left(s_{t}\right)\right] \cdot \nabla_{\boldsymbol{\theta}} \ln \pi\left(a_{t} \mid s_{t} ; \boldsymbol{\theta}\right) \\
& =\left[\mathbb{E}_{S_{t+1}}\left[R_{t}+\gamma \cdot V_{\pi}\left(S_{t+1}\right)\right]-V_{\pi}\left(s_{t}\right)\right] \cdot \nabla_{\boldsymbol{\theta}} \ln \pi\left(a_{t} \mid s_{t} ; \boldsymbol{\theta}\right) .
\end{aligned}
$$

当智能体执行动作 $a_{t}$ 之后, 环境给出新的状态 $s_{t+1}$ 和奖励 $r_{t}$; 利用 $s_{t+1}$ 和 $r_{t}$ 对上面的 期望做蒙特卡洛近似, 得到:

$$
\boldsymbol{g}\left(s_{t}, a_{t} ; \boldsymbol{\theta}\right) \approx\left[r_{t}+\gamma \cdot V_{\pi}\left(s_{t+1}\right)-V_{\pi}\left(s_{t}\right)\right] \cdot \nabla_{\boldsymbol{\theta}} \ln \pi\left(a_{t} \mid s_{t} ; \boldsymbol{\theta}\right) .
$$

进一步把状态价值函数 $V_{\pi}(s)$ 替换成价值网络 $v(s ; \boldsymbol{w})$, 得到:

$$
\tilde{\boldsymbol{g}}\left(s_{t}, a_{t} ; \boldsymbol{\theta}\right) \triangleq[\underbrace{r_{t}+\gamma \cdot v\left(s_{t+1} ; \boldsymbol{w}\right)}_{\text {TD 目标 } \widehat{y}_{t}}-v\left(s_{t} ; \boldsymbol{w}\right)] \cdot \nabla_{\boldsymbol{\theta}} \ln \pi\left(a_{t} \mid s_{t} ; \boldsymbol{\theta}\right) .
$$

前面定义了 TD 目标和 TD 误差：

$$
\widehat{y}_{t} \triangleq r_{t}+\gamma \cdot v\left(s_{t+1} ; \boldsymbol{w}\right) \quad \text { 和 } \quad \delta_{t} \triangleq v\left(s_{t} ; \boldsymbol{w}\right)-\widehat{y}_{t} .
$$

因此, 可以把 $\tilde{\boldsymbol{g}}$ 写成:

$$
\tilde{\boldsymbol{g}}\left(s_{t}, a_{t} ; \boldsymbol{\theta}\right) \triangleq-\delta_{t} \cdot \nabla_{\boldsymbol{\theta}} \ln \pi\left(a_{t} \mid s_{t} ; \boldsymbol{\theta}\right) .
$$

$\tilde{\boldsymbol{g}}$ 是 $\boldsymbol{g}$ 的近似, 所以也是策略梯度 $\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta})$ 的近似。用 $\tilde{\boldsymbol{g}}$ 更新策略网络参数 $\boldsymbol{\theta}$ :

$$
\boldsymbol{\theta} \leftarrow \boldsymbol{\theta}+\beta \cdot \tilde{\boldsymbol{g}}\left(s_{t}, a_{t} ; \boldsymbol{\theta}\right) .
$$

这样可以让目标函数 $J(\boldsymbol{\theta})$ 变大。

策略网络与价值网络的关系：A2C 中策略网络（演员）和价值网络（评委）的关系 如图 $8.4$ 所示。智能体由策略网络 $\pi$ 控制, 与环境交互, 并收集状态、动作、奖励。策略 网络（演员）基于状态 $s_{t}$ 做出动作 $a_{t}$ 。价值网络（评委）基于 $s_{t} 、 s_{t+1} 、 r_{t}$ 算出 TD 误差 $\delta_{t}$ 。策略网络（演员）依靠 $\delta_{t}$ 来判断自己动作的好坏, 从而改进自己的演技（即参数 $\boldsymbol{\theta}$ )。

读者可能会有疑问: 价值网络 $v$ 只知道两个状态 $s_{t} 、 s_{t+1}$, 而并不知道动作 $a_{t}$, 那 么价值网络为什么能评价 $a_{t}$ 的好坏呢? 价值网络 $v$ 告诉策略网络 $\pi$ 的唯一信息是 $\delta_{t}$ 。回 顾一下 $\delta_{t}$ 的定义:

$$
-\delta_{t}=\underbrace{r_{t}+\gamma \cdot v\left(s_{t+1} ; \boldsymbol{w}\right)}_{\text {TD 目标 } \widehat{y}_{t}}-\underbrace{v\left(s_{t} ; \boldsymbol{w}\right)}_{\text {基线 }} .
$$

基线 $v\left(s_{t} ; \boldsymbol{w}\right)$ 是价值网络在 $t$ 时刻对 $\mathbb{E}\left[U_{t}\right]$ 的估计; 此时智能体尚末执行动作 $a_{t}$ 。而 TD 目标 $\widehat{y}_{t}$ 是价值网络在 $t+1$ 时刻对 $\mathbb{E}\left[U_{t}\right]$ 的估计; 此时智能体已经执行动作 $a_{t}$ 。 动作 $a$

![](https://cdn.mathpix.com/cropped/2023_02_03_f46f5cf0e4de5b9996dcg-141.jpg?height=420&width=1108&top_left_y=327&top_left_x=425)

图 8.4: A2C 中策略网络（演员）和价值网络（评委）的关系图。

- 如果 $\widehat{y}_{t}>v\left(s_{t} ; \boldsymbol{w}\right)$, 说明动作 $a_{t}$ 很好, 使得奖励 $r_{t}$ 超出预期, 或者新的状态 $s_{t+1}$ 比预期好；这种情况下应该更新 $\boldsymbol{\theta}$, 使得 $\pi\left(a_{t} \mid s_{t} ; \boldsymbol{\theta}\right)$ 变大。

- 如果 $\widehat{y}_{t}<v\left(s_{t} ; \boldsymbol{w}\right)$, 说明动作 $a_{t}$ 不好, 导致奖励 $r_{t}$ 不及预期, 或者新的状态 $s_{t+1}$ 比预期差；这种情况下应该更新 $\boldsymbol{\theta}$, 使得 $\pi\left(a_{t} \mid s_{t} ; \boldsymbol{\theta}\right)$ 减小。

综上所述, $\delta_{t}$ 中虽然不包含动作 $a_{t}$, 但是 $\delta_{t}$ 可以间接反映出动作 $a_{t}$ 的好坏, 可以帮助 策略网络（演员）改进演技。

### 2 训练流程

下面概括 $\mathrm{A} 2 \mathrm{C}$ 训练流程。设当前策略网络参数是 $\boldsymbol{\theta}_{\text {now }}$, 价值网络参数是 $\boldsymbol{w}_{\text {now }}$ 。执 行下面的步骤, 将参数更新成 $\theta_{\text {new }}$ 和 $\boldsymbol{w}_{\text {new }}$ :

1. 观测到当前状态 $s_{t}$, 根据策略网络做决策： $a_{t} \sim \pi\left(\cdot \mid s_{t} ; \boldsymbol{\theta}_{\text {now }}\right)$, 并让智能体执行动 作 $a_{t}$ 。

2. 从环境中观测到奖励 $r_{t}$ 和新的状态 $s_{t+1}$ 。

3. 让价值网络打分：

$$
\widehat{v}_{t}=v\left(s_{t} ; \boldsymbol{w}_{\text {now }}\right) \quad \text { 和 } \quad \widehat{v}_{t+1}=v\left(s_{t+1} ; \boldsymbol{w}_{\text {now }}\right)
$$

4. 计算 TD 目标和 TD 误差：

$$
\widehat{y}_{t}=r_{t}+\gamma \cdot \widehat{v}_{t+1} \quad \text { 和 } \quad \delta_{t}=\widehat{v}_{t}-\widehat{y}_{t} .
$$

5. 更新价值网络：

$$
\boldsymbol{w}_{\text {new }} \leftarrow \boldsymbol{w}_{\text {now }}-\alpha \cdot \delta_{t} \cdot \nabla_{\boldsymbol{w}} v\left(s_{t} ; \boldsymbol{w}_{\text {now }}\right)
$$

6. 更新策略网络：

$$
\boldsymbol{\theta}_{\text {new }} \leftarrow \boldsymbol{\theta}_{\text {now }}-\beta \cdot \delta_{t} \cdot \nabla_{\boldsymbol{\theta}} \ln \pi\left(a_{t} \mid s_{t} ; \boldsymbol{\theta}_{\text {now }}\right)
$$

注 此处训练策略网络和价值网络的方法属于同策略 (on-policy), 要求行为策略 (behavior policy）与目标策略（target policy）相同, 都是最新的策略网络 $\pi\left(a \mid s ; \boldsymbol{\theta}_{\text {now }}\right)$ 。不能使用经 验回放, 因为经验回放数组中的数据是用旧的策略网络 $\pi\left(a \mid s ; \boldsymbol{\theta}_{\text {old }}\right)$ 获取的, 不能在当前 重复利用。

### 3 用目标网络改进训练

上述训练价值网络的算法存在自举一一即用价值网络自己的估值 $\widehat{v}_{t+1}$ 去更新价值网 络自己。为了缓解自举造成的偏差, 可以使用目标网络（target network）计算 TD 目标。 把目标网络记作 $v\left(s ; \boldsymbol{w}^{-}\right)$, 它的结构与价值网络的结构相同, 但是参数不同。使用目标 网络计算 $\mathrm{TD}$ 目标, 那么 $\mathrm{A} 2 \mathrm{C}$ 的训练就变成了:

1. 观测到当前状态 $s_{t}$, 根据策略网络做决策: $a_{t} \sim \pi\left(\cdot \mid s_{t} ; \boldsymbol{\theta}_{\text {now }}\right)$, 并让智能体执行动 作 $a_{t}$ 。

2. 从环境中观测到奖励 $r_{t}$ 和新的状态 $s_{t+1}$ 。

3. 让价值网络给 $s_{t}$ 打分:

$$
\widehat{v}_{t}=v\left(s_{t} ; \boldsymbol{w}_{\text {now }}\right) .
$$

4. 让目标网络给 $s_{t+1}$ 打分：

$$
\widehat{v}_{t+1}^{-}=v\left(s_{t+1} ; \boldsymbol{w}_{\text {now }}^{-}\right)
$$

5. 计算 TD 目标和 TD 误差：

$$
\widehat{y}_{t}^{-}=r_{t}+\gamma \cdot \widehat{v}_{t+1}^{-} \quad \text { 和 } \quad \delta_{t}=\widehat{v}_{t}-\widehat{y}_{t}^{-} .
$$

6. 更新价值网络：

$$
\boldsymbol{w}_{\text {new }} \leftarrow \boldsymbol{w}_{\text {now }}-\alpha \cdot \delta_{t} \cdot \nabla_{\boldsymbol{w}} v\left(s_{t} ; \boldsymbol{w}_{\text {now }}\right)
$$

7. 更新策略网络:

$$
\boldsymbol{\theta}_{\text {new }} \leftarrow \boldsymbol{\theta}_{\text {now }}-\beta \cdot \delta_{t} \cdot \nabla_{\boldsymbol{\theta}} \ln \pi\left(a_{t} \mid s_{t} ; \boldsymbol{\theta}_{\text {now }}\right) .
$$

8. 设 $\tau \in(0,1)$ 是需要手动调的超参数。做加权平均更新目标网络的参数:

$$
\boldsymbol{w}_{\text {new }}^{-} \leftarrow \tau \cdot \boldsymbol{w}_{\text {new }}+(1-\tau) \cdot \boldsymbol{w}_{\text {now }}^{-} .
$$



## $8.4$ 证明带基线的策略梯度定理

本节证明带基线的策略梯度定理 8.1。将定理 $7.1$ 与引理 $8.2$ 相结合, 即可证得定理 $8.1$ 。

## 引理 $8.2$

设 $b$ 是任意函数, $b$ 不依赖于 $A$ 。那么对于任意的 $s$,

$$
\mathbb{E}_{A \sim \pi(\cdot \mid s ; \boldsymbol{\theta})}\left[b \cdot \frac{\partial \ln \pi(A \mid s ; \boldsymbol{\theta})}{\partial \boldsymbol{\theta}}\right]=0 .
$$

证明 由于基线 $b$ 不依赖于动作 $A$, 可以把 $b$ 提取到期望外面:

$$
\begin{aligned}
\mathbb{E}_{A \sim \pi(\cdot \mid s ; \boldsymbol{\theta})}\left[b \cdot \frac{\partial \ln \pi(A \mid s ; \boldsymbol{\theta})}{\partial \boldsymbol{\theta}}\right] & =b \cdot \mathbb{E}_{A \sim \pi(\cdot \mid s ; \boldsymbol{\theta})}\left[\frac{\partial \ln \pi(A \mid s ; \boldsymbol{\theta})}{\partial \boldsymbol{\theta}}\right] \\
& =b \cdot \sum_{a \in \mathcal{A}} \pi(a \mid s ; \boldsymbol{\theta}) \cdot \frac{\partial \ln \pi(a \mid s ; \boldsymbol{\theta})}{\partial \boldsymbol{\theta}} \\
& =b \cdot \sum_{a \in \mathcal{A}} \pi(a \mid s ; \boldsymbol{\theta}) \cdot \frac{1}{\pi(a \mid s ; \boldsymbol{\theta})} \cdot \frac{\partial \pi(a \mid s ; \boldsymbol{\theta})}{\partial \boldsymbol{\theta}} \\
& =b \cdot \sum_{a \in \mathcal{A}} \frac{\partial \pi(a \mid s ; \boldsymbol{\theta})}{\partial \boldsymbol{\theta}}
\end{aligned}
$$

上式最右边的连加是关于 $a$ 求的, 而偏导是关于 $\theta$ 求的, 因此可以把连加放入偏导内部：

$$
\mathbb{E}_{A \sim \pi(\cdot \mid s ; \boldsymbol{\theta})}\left[b \cdot \frac{\partial \ln \pi(A \mid s ; \boldsymbol{\theta})}{\partial \boldsymbol{\theta}}\right]=b \cdot \frac{\partial}{\partial \boldsymbol{\theta}} \underbrace{\sum_{a \in \mathcal{A}} \pi(a \mid s ; \boldsymbol{\theta})}_{\text {恒等于 } 1} .
$$

因此

$$
\mathbb{E}_{A \sim \pi(\cdot \mid s ; \boldsymbol{\theta})}\left[b \cdot \frac{\partial \ln \pi(A \mid s ; \boldsymbol{\theta})}{\partial \boldsymbol{\theta}}\right]=b \cdot \frac{\partial 1}{\partial \boldsymbol{\theta}}=0 .
$$



## 第 8 章 知识点

- 在策略梯度中加入基线 (baseline) 可以降低方差, 显著提升实验效果。实践中常用 $b=V_{\pi}(s)$ 作为基线。

- 可以用基线来改进 REINFORCE 算法。价值网络 $v(s ; \boldsymbol{w})$ 近似状态价值函数 $V_{\pi}(s)$, 把 $v(s ; \boldsymbol{w})$ 作为基线。用策略梯度上升来更新策略网络 $\pi(a \mid s ; \boldsymbol{\theta})$ 。用蒙特卡洛（而 非自举) 来更新价值网络 $v(s ; \boldsymbol{w})$ 。

- 可以用基线来改进 actor-critic, 得到的方法叫做 advantage actor-critic (A2C), 它也 有一个策略网络 $\pi(a \mid s ; \boldsymbol{\theta})$ 和一个价值网络 $v(s ; \boldsymbol{\theta})$ 。用策略梯度上升来更新策略网 络, 用 $\mathrm{TD}$ 算法来更新价值网络。

## 第 8 章 习题

1. 是否可以用最优动作价值函数 $Q_{\star}(s, a)$ 作为基线?
A. 可以, 因为 $Q_{\star}$ 不依赖于动作。
B. 不行, 因为 $Q_{\star}$ 依赖于动作。

2. 带基线的 REINFORCE 算法属于 Actor-Critic。
A. 上述说法正确, 因为算法同时训练策略网络 $\pi$ 和价值网络 $v$ 。
B. 上述说法错误, 因为真正帮助训练策略网络 $\pi$ 的是观测到的回报 $u$, 而价值网 络 $v$ 仅仅起到基线的作用。
