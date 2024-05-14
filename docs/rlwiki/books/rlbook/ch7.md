
## 第 7 章 策略梯度方法

本章的内容是策略学习 (policy-based reinforcement learning) 以及策略梯度 (policy gradient)。策略学习的意思是通过求解一个优化问题, 学出最优策略函数或它的近似函数 (比 如策略网络)。第 $7.1$ 节描述策略网络。第 $7.2$ 节把策略学习描述成一个最大化问题。第 $7.3$ 节推导策略梯度。第 $7.4$ 和 $7.5$ 节用不同的方法近似策略梯度, 得到两种训练策略网 络的方法-REINFORCE 和 actor-critic。本章介绍的 REINFORCE 和 actor-critic 只是帮 助大家理解算法而已, 实际效果并不好。在实践中不建议用本章的原始方法, 而应该用 下一章的方法。

## 1 第略网络

本章假设动作空间是离散的, 比如 $\mathcal{A}=\{$ 左, 右, 上 $\}$ 。策略函数 $\pi$ 是个条件概率质量 函数:

$$
\pi(a \mid s) \triangleq \mathbb{P}(A=a \mid S=s) .
$$

策略函数 $\pi$ 的输入是状态 $s$ 和动作 $a$, 输出是一个 0 到 1 之间的概率值。举个例子, 把 超级玛丽游戏当前屏幕上的画面作为 $s$, 策略函数会输出每个动作的概率值:

$$
\begin{aligned}
& \pi(\text { 左 } \mid s)=0.5, \\
& \pi(\text { 右 } \mid s)=0.2, \\
& \pi(\text { 上 } \mid s)=0.3 .
\end{aligned}
$$

如果我们有这样一个策略函数, 我们就可以拿它控制智能体。每当观测到一个状态 $s$, 就 用策略函数计算出每个动作的概率值, 然后做随机抽样, 得到一个动作 $a$, 让智能体执 行 $a$ 。

怎么样才能得到这样一个策略函数呢? 当前最有效的方法是用神经网络 $\pi(a \mid s ; \boldsymbol{\theta})$ 近 似策略函数 $\pi(a \mid s)$ 。神经网络 $\pi(a \mid s ; \boldsymbol{\theta})$ 被称为策略网络。 $\boldsymbol{\theta}$ 表示神经网络的参数; 一开 始随机初始化 $\boldsymbol{\theta}$, 随后利用收集的状态、动作、奖励去更新 $\boldsymbol{\theta}$ 。

![](https://cdn.mathpix.com/cropped/2023_02_03_f46f5cf0e4de5b9996dcg-113.jpg?height=208&width=1442&top_left_y=2089&top_left_x=250)

策略网络

图 7.1: 策略网络 $\pi(a \mid s ; \boldsymbol{\theta})$ 的神经网络结构。输入是状态 $s$, 输出是动作空间 $\mathcal{A}$ 中每个动作的概 率值。

策略网络的结构如图 $7.1$ 所示。策略网络的输入是状态 $s$ 。在 Atari 游戏、围棋等应用 中, 状态是张量 (比如图片), 那么应该如图 $7.1$ 所示用卷积网络处理输入。在机器人控 制等应用中, 状态 $s$ 是向量, 它的元素是多个传感器的数值, 那么应该把卷积网络换成 全连接网络。策略网络输出层的激活函数是 softmax, 因此输出的向量（记作 $\boldsymbol{f}$ ) 所有元 素都是正数, 而且相加等于 1 。动作空间 $\mathcal{A}$ 的大小是多少, 向量 $\boldsymbol{f}$ 的维度就是多少。在 超级玛丽的例子中, $\mathcal{A}=\{$ 左, 右, 上 $\}$, 那么 $\boldsymbol{f}$ 就是 3 维的向量, 比如 $\boldsymbol{f}=[0.2,0.1,0.7]$ 。 $f$ 描述了动作空间 $\mathcal{A}$ 上的离散概率分布, $f$ 每个元素对应一个动作:

$$
\begin{gathered}
f_{1}=\pi(\text { 左 } \mid s)=0.2, \\
f_{2}=\pi(\text { 右 } \mid s)=0.1, \\
f_{3}=\pi(\text { 上 } \mid s)=0.7 .
\end{gathered}
$$



## $7.2$ 策略学习的目标函数

为了推导策略学习的目标函数, 我们需要先复习回报和价值函数。回报 $U_{t}$ 是从 $t$ 时 刻开始的所有奖励之和。 $U_{t}$ 依赖于 $t$ 时刻开始的所有状态和动作:

$$
S_{t}, A_{t}, S_{t+1}, A_{t+1}, S_{t+2}, A_{t+2}, \cdots
$$

在 $t$ 时刻, $U_{t}$ 是随机变量, 它的不确定性来自于末来末知的状态和动作。动作价值函数 的定义是:

$$
Q_{\pi}\left(s_{t}, a_{t}\right)=\mathbb{E}\left[U_{t} \mid S_{t}=s_{t}, A_{t}=a_{t}\right] .
$$

条件期望把 $t$ 时刻状态 $s_{t}$ 和动作 $a_{t}$ 看做已知观测值, 把 $t+1$ 时刻后的状态和动作看做 末知变量, 并消除这些变量。状态价值函数的定义是

$$
V_{\pi}\left(s_{t}\right)=\mathbb{E}_{A_{t} \sim \pi\left(\cdot \mid s_{t} ; \boldsymbol{\theta}\right)}\left[Q_{\pi}\left(s_{t}, A_{t}\right)\right] .
$$

状态价值既依赖于当前状态 $s_{t}$, 也依赖于策略网络 $\pi$ 的参数 $\boldsymbol{\theta}$ 。

- 当前状态 $s_{t}$ 越好, 则 $V_{\pi}\left(s_{t}\right)$ 越大, 即回报 $U_{t}$ 的期望越大。例如, 在超级玛丽游戏 中, 如果玛丽奥已经接近终点 (也就是说当前状态 $s_{t}$ 很好), 那么回报的期望就会 很大。

- 策略 $\pi$ 越好（即参数 $\boldsymbol{\theta}$ 越好), 那么 $V_{\pi}\left(s_{t}\right)$ 也会越大。例如, 从同一起点出发打游 戏, 高手 (好的策略）的期望回报远高于初学者（差的策略)。

如果一个策略很好, 那么状态价值 $V_{\pi}(S)$ 的均值应当很大。因此我们定义目标函数：

$$
J(\boldsymbol{\theta})=\mathbb{E}_{S}\left[V_{\pi}(S)\right] .
$$

这个目标函数排除掉了状态 $S$ 的因素, 只依赖于策略网络 $\pi$ 的参数 $\boldsymbol{\theta}$; 策略越好, 则 $J(\theta)$ 越大。所以策略学习可以描述为这样一个优化问题:

$$
\max _{\boldsymbol{\theta}} J(\boldsymbol{\theta}) \text {. }
$$

我们希望通过对策略网络参数 $\boldsymbol{\theta}$ 的更新, 使得目标函数 $J(\boldsymbol{\theta})$ 越来越大, 也就意味着策略 网络越来越强。想要求解最大化问题, 显然可以用梯度上升更新 $\boldsymbol{\theta}$, 使得 $J(\boldsymbol{\theta})$ 增大。设 当前策略网络的参数为 $\theta_{\text {now }}$, 做梯度上升更新参数, 得到新的参数 $\theta_{\text {new }}$ :

$$
\boldsymbol{\theta}_{\text {new }} \leftarrow \boldsymbol{\theta}_{\text {now }}+\beta \cdot \nabla_{\boldsymbol{\theta}} J\left(\boldsymbol{\theta}_{\text {now }}\right) .
$$

此处的 $\beta$ 是学习率, 需要手动调整。上面的公式就是训练策略网络的基本思路, 其中的 梯度

$$
\left.\nabla_{\boldsymbol{\theta}} J\left(\boldsymbol{\theta}_{\text {now }}\right) \triangleq \frac{\partial J(\boldsymbol{\theta})}{\partial \boldsymbol{\theta}}\right|_{\boldsymbol{\theta}=\boldsymbol{\theta}_{\text {now }}}
$$

被称作策略梯度。策略梯度可以写成下面定理中的期望形式。之后的算法推导都要基于 这个定理, 并对其中的期望做近似。

## 定理 7.1. 策略梯度定理（不严谨的表述）

$$
\frac{\partial J(\boldsymbol{\theta})}{\partial \boldsymbol{\theta}}=\mathbb{E}_{S}\left[\mathbb{E}_{A \sim \pi(\cdot \mid S ; \boldsymbol{\theta})}\left[\frac{\partial \ln \pi(A \mid S ; \boldsymbol{\theta})}{\partial \boldsymbol{\theta}} \cdot Q_{\pi}(S, A)\right]\right]
$$

注 上面的策略梯度定理是不严谨的表述, 尽管大多数论文和书籍使用这种表述。严格地 讲, 这个定理只有在 “状态 $S$ 服从马尔科夫链的稳态分布 $d(\cdot)$ ”这个假设下才成立。定理 中的等号其实是不对的, 期望前面应该有一项系数 $1+\gamma+\cdots+\gamma^{n-1}=\frac{1-\gamma^{n}}{1-\gamma}$, 其中 $\gamma$ 是 折扣率, $n$ 是一局游戏的长度。严格地讲, 策略梯度定理应该是：

$$
\frac{\partial J(\boldsymbol{\theta})}{\partial \boldsymbol{\theta}}=\frac{1-\gamma^{n}}{1-\gamma} \cdot \mathbb{E}_{S \sim d(\cdot)}\left[\mathbb{E}_{A \sim \pi(\cdot \mid S ; \boldsymbol{\theta})}\left[\frac{\partial \ln \pi(A \mid S ; \boldsymbol{\theta})}{\partial \boldsymbol{\theta}} \cdot Q_{\pi}(S, A)\right]\right] .
$$

在实际应用中, 系数 $\frac{1-\gamma^{n}}{1-\gamma}$ 无关紧要, 可以忽略掉。其原因是做梯度上升的时候, 系数 $\frac{1-\gamma^{n}}{1-\gamma}$ 会被学习率 $\beta$ 吸收。

## $7.3$ 策略梯度定理的证明

策略梯度定理是策略学习的关键所在。本节的内容是证明策略梯度定理。尽管本节 数学较多, 但还是建议读者认真读完第 $7.3 .1$ 小节, 理解策略梯度简化的推导。第 7.3.2 小节是策略梯度定理完整的证明。由于完整证明较为复杂, 大多数教材中不涉及这部分 内容, 本书也不建议读者掌握完整证明, 除非读者从事强化学习的科研工作。

### 1 简化的证明

把策略网络 $\pi(a \mid s ; \boldsymbol{\theta})$ 看做动作的概率质量函数 (或概率密度函数)。状态价值函数 $V_{\pi}(s)$ 可以写成:

$$
\begin{aligned}
V_{\pi}(s) & =\mathbb{E}_{A \sim \pi(\cdot \mid s ; \boldsymbol{\theta})}\left[Q_{\pi}(s, A)\right] \\
& =\sum_{a \in \mathcal{A}} \pi(a \mid s ; \boldsymbol{\theta}) \cdot Q_{\pi}(s, a) .
\end{aligned}
$$

状态价值 $V_{\pi}(s)$ 关于 $\boldsymbol{\theta}$ 的梯度可以写作:

$$
\begin{aligned}
\frac{\partial V_{\pi}(s)}{\partial \boldsymbol{\theta}} & =\frac{\partial}{\partial \boldsymbol{\theta}} \sum_{a \in \mathcal{A}} \pi(a \mid s ; \boldsymbol{\theta}) \cdot Q_{\pi}(s, a) \\
& =\sum_{a \in \mathcal{A}} \frac{\partial \pi(a \mid s ; \boldsymbol{\theta}) \cdot Q_{\pi}(s, a)}{\partial \boldsymbol{\theta}} .
\end{aligned}
$$

上面第二个等式把求导放入连加里面; 等式成立的原因是求导的对象 $\theta$ 与连加的对象 $a$ 不同。回忆一下链式法则：设 $z=f(x) \cdot g(x)$, 那么

$$
\frac{\partial z}{\partial x}=\frac{\partial f(x)}{\partial x} \cdot g(x)+f(x) \cdot \frac{\partial g(x)}{\partial x} .
$$

应用链式法则, 公式 (7.1) 中的梯度可以写作:

$$
\begin{aligned}
& \frac{\partial V_{\pi}(s)}{\partial \boldsymbol{\theta}}=\sum_{a \in \mathcal{A}} \frac{\partial \pi(a \mid s ; \boldsymbol{\theta})}{\partial \boldsymbol{\theta}} \cdot Q_{\pi}(s, a)+\sum_{a \in \mathcal{A}} \pi(a \mid s ; \boldsymbol{\theta}) \cdot \frac{\partial Q_{\pi}(s, a)}{\partial \boldsymbol{\theta}} \\
& =\sum_{a \in \mathcal{A}} \frac{\partial \pi(a \mid s ; \boldsymbol{\theta})}{\partial \boldsymbol{\theta}} \cdot Q_{\pi}(s, a)+\underbrace{\mathbb{E}_{A \sim \pi(\cdot \mid s ; \boldsymbol{\theta})}\left[\frac{\partial Q_{\pi}(s, A)}{\partial \boldsymbol{\theta}}\right]}_{\text {设为 } x} .
\end{aligned}
$$

上面公式最右边一项 $x$ 的分析非常复杂, 此处不具体分析了。由上面的公式可得:

$$
\begin{aligned}
\frac{\partial V_{\pi}(s)}{\partial \boldsymbol{\theta}} & =\sum_{A \in \mathcal{A}} \frac{\partial \pi(A \mid S ; \boldsymbol{\theta})}{\partial \boldsymbol{\theta}} \cdot Q_{\pi}(S, A)+x \\
& =\sum_{A \in \mathcal{A}} \pi(A \mid S ; \boldsymbol{\theta}) \cdot \underbrace{\frac{1}{\pi(A \mid S ; \boldsymbol{\theta})} \cdot \frac{\partial \pi(A \mid S ; \boldsymbol{\theta})}{\partial \boldsymbol{\theta}}}_{\text {等于 } \partial \ln \pi(A \mid S ; \boldsymbol{\theta}) / \partial \boldsymbol{\theta}} \cdot Q_{\pi}(S, A)+x .
\end{aligned}
$$

上面第二个等式成立的原因是添加的两个红色项相乘等于一。公式中用下花括号标出的 项等于 $\frac{\partial \ln \pi(A \mid S ; \boldsymbol{\theta})}{\partial \theta}$ 。由此可得

$$
\begin{aligned}
\frac{\partial V_{\pi}(s)}{\partial \boldsymbol{\theta}} & =\sum_{A \in \mathcal{A}} \pi(A \mid S ; \boldsymbol{\theta}) \cdot \frac{\partial \ln \pi(A \mid S ; \boldsymbol{\theta})}{\partial \boldsymbol{\theta}} \cdot Q_{\pi}(S, A)+x \\
& =\mathbb{E}_{A \sim \pi(\cdot \mid S ; \theta)}\left[\frac{\partial \ln \pi(A \mid S ; \boldsymbol{\theta})}{\partial \boldsymbol{\theta}} \cdot Q_{\pi}(S, A)\right]+x .
\end{aligned}
$$

公式中红色标出的 $\pi(A \mid S ; \boldsymbol{\theta})$ 被看做概率质量函数, 因此连加可以写成期望的形式。由目 标函数的定义 $J(\boldsymbol{\theta})=\mathbb{E}_{S}\left[V_{\pi}(S)\right]$ 可得

$$
\begin{aligned}
\frac{\partial J(\boldsymbol{\theta})}{\partial \boldsymbol{\theta}} & =\mathbb{E}_{S}\left[\frac{\partial V_{\pi}(S)}{\partial \boldsymbol{\theta}}\right] \\
& =\mathbb{E}_{S}\left[\mathbb{E}_{A \sim \pi(\cdot \mid S ; \boldsymbol{\theta})}\left[\frac{\partial \ln \pi(A \mid S ; \boldsymbol{\theta})}{\partial \boldsymbol{\theta}} \cdot Q_{\pi}(S, A)\right]\right]+\mathbb{E}_{S}[x] .
\end{aligned}
$$

不严谨的证明通常忽略掉 $x$, 于是得到定理 7.1。在下一小节中, 我们给出严格的证明。 除非读者对强化学习的数学推导很感兴趣, 否则没必要阅读下一小节。

### 2 完整的证明

本小节给出策略梯度定理的严格数学证明。首先证明几个引理, 最后用引理证明策 略梯度定理。引理 $7.2$ 分析梯度 $\frac{\partial V_{\pi}(s)}{\partial \theta}$, 并把它递归地表示为 $\frac{\partial V_{\pi}\left(S^{\prime}\right)}{\partial \theta}$ 的期望, 其中 $S^{\prime}$ 是 下一时刻的状态。

## 引理 7.2.递归公式

$$
\frac{\partial V_{\pi}(s)}{\partial \boldsymbol{\theta}}=\mathbb{E}_{A \sim \pi(\cdot \mid s ; \boldsymbol{\theta})}\left[\frac{\partial \ln \pi(A \mid s ; \boldsymbol{\theta})}{\partial \boldsymbol{\theta}} \cdot Q_{\pi}(s, A)+\gamma \cdot \mathbb{E}_{S^{\prime} \sim p(\cdot \mid s, A)}\left[\frac{\partial V_{\pi}\left(S^{\prime}\right)}{\partial \boldsymbol{\theta}}\right]\right] .
$$

证明 设奖励 $R$ 和新状态 $S^{\prime}$ 是在智能体执行动作 $A$ 之后由环境给出的。新状态 $S^{\prime}$ 的概 率质量函数是状态转移函数 $p\left(S^{\prime} \mid S, A\right)$ 。设奖励 $R$ 是 $S 、 A 、 S^{\prime}$ 三者的函数, 因此可以将 其记为 $R\left(S, A, S^{\prime}\right)$ 。由贝尔曼方程可得:

$$
\begin{aligned}
Q_{\pi}(s, a) & =\mathbb{E}_{S^{\prime} \sim p(\cdot \mid s, a)}\left[R\left(s, a, S^{\prime}\right)+\gamma \cdot V_{\pi}\left(s^{\prime}\right)\right] \\
& =\sum_{s^{\prime} \in \mathcal{S}} p\left(s^{\prime} \mid s, a\right) \cdot\left[R\left(s, a, s^{\prime}\right)+\gamma \cdot V_{\pi}\left(s^{\prime}\right)\right] \\
& =\sum_{s^{\prime} \in \mathcal{S}} p\left(s^{\prime} \mid s, a\right) \cdot R\left(s, a, s^{\prime}\right)+\gamma \cdot \sum_{s^{\prime} \in \mathcal{S}} p\left(s^{\prime} \mid s, a\right) \cdot V_{\pi}\left(s^{\prime}\right) .
\end{aligned}
$$

在观测到 $s 、 a 、 s^{\prime}$ 之后, $p\left(s^{\prime} \mid s, a\right)$ 和 $R\left(s, a, s^{\prime}\right)$ 都与策略网络 $\pi$ 无关, 因此

$$
\frac{\partial}{\partial \boldsymbol{\theta}}\left[p\left(s^{\prime} \mid s, a\right) \cdot R\left(s, a, s^{\prime}\right)\right]=0 .
$$

由公式 (7.3) 与 (7.4) 可得:

$$
\begin{aligned}
& =\gamma \cdot \sum_{s^{\prime} \in \mathcal{S}} p\left(s^{\prime} \mid s, a\right) \cdot \frac{\partial V_{\pi}\left(s^{\prime}\right)}{\partial \boldsymbol{\theta}} \\
& =\gamma \cdot \mathbb{E}_{S^{\prime} \sim p(\cdot \mid s, a)}\left[\frac{\partial V_{\pi}\left(S^{\prime}\right)}{\partial \boldsymbol{\theta}}\right] .
\end{aligned}
$$

由上一小节的公式 (7.2) 可得:

$$
\begin{aligned}
& \frac{\partial V_{\pi}(s)}{\partial \boldsymbol{\theta}} \\
& =\mathbb{E}_{A \sim \pi(\cdot \mid S ; \boldsymbol{\theta})}\left[\frac{\partial \ln \pi(A \mid S ; \boldsymbol{\theta})}{\partial \boldsymbol{\theta}} \cdot Q_{\pi}(S, A)\right]+\mathbb{E}_{A \sim \pi(\cdot \mid S ; \boldsymbol{\theta})}\left[\frac{\partial Q_{\pi}(s, a)}{\partial \boldsymbol{\theta}}\right] .
\end{aligned}
$$

![](https://cdn.mathpix.com/cropped/2023_02_03_f46f5cf0e4de5b9996dcg-118.jpg?height=160&width=1350&top_left_y=2039&top_left_x=410)

结合公式 (7.5)、(7.6) 可得引理 7.2.

## 引理 7.3. 策略梯度的连加形式

设 $\boldsymbol{g}(s, a ; \boldsymbol{\theta}) \triangleq Q_{\pi}(s, a) \cdot \frac{\partial \ln \pi(a \mid s ; \boldsymbol{\theta})}{\partial \boldsymbol{\theta}}$ 。设一局游戏在第 $n$ 步之后结束。那么

$$
\begin{aligned}
\frac{\partial J(\boldsymbol{\theta})}{\partial \boldsymbol{\theta}}= & \mathbb{E}_{S_{1}, A_{1}}\left[\boldsymbol{g}\left(S_{1}, A_{1} ; \boldsymbol{\theta}\right)\right] \\
& +\gamma \cdot \mathbb{E}_{S_{1}, A_{1}, S_{2}, A_{2}}\left[\boldsymbol{g}\left(S_{2}, A_{2} ; \boldsymbol{\theta}\right)\right] \\
& +\gamma^{2} \cdot \mathbb{E}_{S_{1}, A_{1}, S_{2}, A_{2}, S_{3}, A_{3}}\left[\boldsymbol{g}\left(S_{3}, A_{3} ; \boldsymbol{\theta}\right)\right] \\
& +\cdots \\
& \left.+\gamma^{n-1} \cdot \mathbb{E}_{S_{1}, A_{1}, S_{2}, A_{2}, S_{3}, A_{3}, \cdots S_{n}, A_{n}}\left[\boldsymbol{g}\left(S_{n}, A_{n} ; \boldsymbol{\theta}\right)\right]\right] .
\end{aligned}
$$

证明 设 $S 、 A$ 为当前状态和动作, $S^{\prime}$ 为下一个状态。引理 $7.2$ 证明了下面的结论：

$$
\frac{\partial V_{\pi}(S)}{\partial \boldsymbol{\theta}}=\mathbb{E}_{A}[\underbrace{\frac{\partial \ln \pi(A \mid S ; \boldsymbol{\theta})}{\partial \boldsymbol{\theta}} \cdot Q_{\pi}(S, A)}_{\text {定义为 } \boldsymbol{g}(S, A ; \boldsymbol{\theta})}+\gamma \cdot \mathbb{E}_{S^{\prime}}\left[\frac{\partial V_{\pi}\left(S^{\prime}\right)}{\partial \boldsymbol{\theta}}\right]] .
$$

这样我们可以把 $\frac{\partial V_{\pi}\left(S_{1}\right)}{\partial \theta}$ 写成递归的形式:

$$
\frac{\partial V_{\pi}\left(S_{1}\right)}{\partial \boldsymbol{\theta}}=\mathbb{E}_{A_{1}}\left[\boldsymbol{g}\left(S_{1}, A_{1} ; \boldsymbol{\theta}\right)\right]+\gamma \cdot \mathbb{E}_{A_{1}, S_{2}}\left[\frac{\partial V_{\pi}\left(S_{2}\right)}{\partial \boldsymbol{\theta}}\right] .
$$

同理, $\frac{\partial V_{\pi}\left(S_{2}\right)}{\partial \theta}$ 可以写成

$$
\frac{\partial V_{\pi}\left(S_{2}\right)}{\partial \boldsymbol{\theta}}=\mathbb{E}_{A_{2}}\left[\boldsymbol{g}\left(S_{2}, A_{2} ; \boldsymbol{\theta}\right)\right]+\gamma \cdot \mathbb{E}_{A_{2}, S_{3}}\left[\frac{\partial V_{\pi}\left(S_{3}\right)}{\partial \boldsymbol{\theta}}\right] .
$$

把等式 (7.8) 揷入等式 (7.7), 得到

$$
\begin{aligned}
\frac{\partial V_{\pi}\left(S_{1}\right)}{\partial \boldsymbol{\theta}}= & \mathbb{E}_{A_{1}}\left[\boldsymbol{g}\left(S_{1}, A_{1} ; \boldsymbol{\theta}\right)\right] \\
& +\gamma \cdot \mathbb{E}_{A_{1}, S_{2}, A_{2}}\left[\boldsymbol{g}\left(S_{2}, A_{2} ; \boldsymbol{\theta}\right)\right] \\
& +\gamma^{2} \cdot \mathbb{E}_{A_{1}, S_{2}, A_{2}, S_{3}}\left[\frac{\partial V_{\pi}\left(S_{3}\right)}{\partial \boldsymbol{\theta}}\right] .
\end{aligned}
$$

按照这种规律递归下去, 可得:

$$
\begin{aligned}
\frac{\partial V_{\pi}\left(S_{1}\right)}{\partial \boldsymbol{\theta}}= & \mathbb{E}_{A_{1}}\left[\boldsymbol{g}\left(S_{1}, A_{1} ; \boldsymbol{\theta}\right)\right] \\
& +\gamma \cdot \mathbb{E}_{A_{1}, S_{2}, A_{2}}\left[\boldsymbol{g}\left(S_{2}, A_{2} ; \boldsymbol{\theta}\right)\right] \\
& +\gamma^{2} \cdot \mathbb{E}_{A_{1}, S_{2}, A_{2}, S_{3}, A_{3}}\left[\boldsymbol{g}\left(S_{3}, A_{3} ; \boldsymbol{\theta}\right)\right] \\
& +\cdots \\
& +\gamma^{n-1} \cdot \mathbb{E}_{A_{1}, S_{2}, A_{2}, S_{3}, A_{3}, \cdots S_{n}, A_{n}}\left[\boldsymbol{g}\left(S_{n}, A_{n} ; \boldsymbol{\theta}\right)\right] \\
& \left.+\gamma^{n} \cdot \mathbb{E}_{A_{1}, S_{2}, A_{2}, S_{3}, A_{3}, \cdots S_{n}, A_{n}, S_{n+1}}^{\frac{\partial V_{\pi}\left(S_{n+1}\right)}{\partial \boldsymbol{\theta}}}\right] \underbrace{}_{\text {等于零 }}] .
\end{aligned}
$$

上式中最后一项等于零, 原因是游戏在 $n$ 时刻后结束, 而 $n+1$ 时刻之后没有奖励, 所 以 $n+1$ 时刻的回报和价值都是零。最后, 由上面的公式和

$$
\frac{\partial J(\boldsymbol{\theta})}{\partial \boldsymbol{\theta}}=\mathbb{E}_{S_{1}}\left[\frac{\partial V_{\pi}\left(S_{1}\right)}{\partial \boldsymbol{\theta}}\right]
$$

可得引理 7.3.

稳态分布 : 想要严格证明策略梯度定理, 需要用到马尔科夫链 (Markov chain) 的稳 态分布 (stationary distribution)。设状态 $s^{\prime}$ 是这样得到的: $s \rightarrow a \rightarrow s^{\prime}$ 。回忆一下, 状态 转移函数 $p\left(s^{\prime} \mid s, a\right)$, 是一个概率质量函数。设 $d(s)$ 是状态 $s$ 的概率质量函数那么状态 $s^{\prime}$ 的边缘分布是

$$
\tilde{d}\left(s^{\prime}\right)=\sum_{s \in \mathcal{S}} \sum_{a \in \mathcal{A}} p\left(s^{\prime} \mid s, a\right) \cdot \pi(a \mid s ; \boldsymbol{\theta}) \cdot d(s) .
$$

如果 $\tilde{d}(\cdot)$ 与 $d(\cdot)$ 是相同的概率质量函数, 即 $d^{\prime}(s)=d(s) \forall s \in \mathcal{S}$, 则意味着马尔科夫链达 到稳态, 而 $d(\cdot)$ 就是稳态时的概率质量函数。

## 引理 $7.4$

设 $d(\cdot)$ 是马尔科夫链稳态时的概率质量（密度）函数。那么对于任意函数 $f\left(S^{\prime}\right)$,

$$
\mathbb{E}_{S \sim d(\cdot)}\left[\mathbb{E}_{A \sim \pi(\cdot \mid S ; \boldsymbol{\theta})}\left[\mathbb{E}_{S^{\prime} \sim p(\cdot \mid s, A)}\left[f\left(S^{\prime}\right)\right]\right]\right]=\mathbb{E}_{S^{\prime} \sim d(\cdot)}\left[f\left(S^{\prime}\right)\right] .
$$

证明 把引理中的期望写成连加的形式：

$$
\begin{aligned}
& \mathbb{E}_{S \sim d(\cdot)}\left[\mathbb{E}_{A \sim \pi(\cdot \mid S ; \boldsymbol{\theta})}\left[\mathbb{E}_{S^{\prime} \sim p(\cdot \mid s, A)}\left[f\left(S^{\prime}\right)\right]\right]\right] \\
& =\sum_{s \in \mathcal{S}} d(s) \sum_{a \in \mathcal{A}} \pi(a \mid s ; \boldsymbol{\theta}) \sum_{s^{\prime} \in \mathcal{S}} p\left(s^{\prime} \mid s, a\right) \cdot f\left(s^{\prime}\right) \\
& =\sum_{s^{\prime} \in \mathcal{S}} f\left(s^{\prime}\right) \underbrace{\sum_{s \in \mathcal{S}} \sum_{a \in \mathcal{A}} p\left(s^{\prime} \mid s, a\right) \cdot \pi(a \mid s ; \boldsymbol{\theta}) \cdot d(s)}_{\text {等于 } d\left(s^{\prime}\right)} .
\end{aligned}
$$

上面等式最右边标出的项等于 $d\left(s^{\prime}\right)$, 这是根据稳态分布的定义得到的。于是有

$$
\begin{aligned}
\mathbb{E}_{S \sim d(\cdot)}\left[\mathbb{E}_{A \sim \pi(\cdot \mid S ; \theta)}\left[\mathbb{E}_{S^{\prime} \sim p(\cdot \mid s, A)}\left[f\left(S^{\prime}\right)\right]\right]\right] & =\sum_{s^{\prime} \in \mathcal{S}} f\left(s^{\prime}\right) \cdot d\left(s^{\prime}\right) \\
& =\mathbb{E}_{S^{\prime} \sim d(\cdot)}\left[f\left(S^{\prime}\right)\right] .
\end{aligned}
$$

由此可得引理 7.4.

## 定理 7.5. 策略梯度定理（严谨的表述）

设目标函数为 $J(\boldsymbol{\theta})=\mathbb{E}_{S \sim d(\cdot)}\left[V_{\pi}(S)\right]$, 设 $d(s)$ 为马尔科夫链稳态分布的概率质量 (密度）函数。那么

$$
\frac{\partial J(\boldsymbol{\theta})}{\partial \boldsymbol{\theta}}=\left(1+\gamma+\gamma^{2}+\cdots+\gamma^{n-1}\right) \cdot \mathbb{E}_{S \sim d(\cdot)}\left[\mathbb{E}_{A \sim \pi(\cdot \mid S ; \boldsymbol{\theta})}\left[\frac{\partial \ln \pi(A \mid S ; \boldsymbol{\theta})}{\partial \boldsymbol{\theta}} \cdot Q_{\pi}(S, A)\right]\right]
$$

证明设初始状态 $S_{1}$ 服从马尔科夫链的稳态分布, 它的概率质量函数是 $d\left(S_{1}\right)$ 。对于所 有的 $t=1, \cdots, n$, 动作 $A_{t}$ 根据策略网络抽样得到：

$$
A_{t} \sim \pi\left(\cdot \mid S_{t} ; \boldsymbol{\theta}\right),
$$

新的状态 $S_{t+1}$ 根据状态转移函数抽样得到:

$$
S_{t+1} \sim p\left(\cdot \mid S_{t}, A_{t}\right) .
$$

对于任意函数 $f$, 反复应用引理 $7.4$ 可得:

$$
\begin{aligned}
& \mathbb{E}_{S_{1} \sim d}\left\{\mathbb{E}_{A_{1} \sim \pi, S_{2} \sim p}\left\{\mathbb{E}_{A_{2}, S_{3}, A_{3}, S_{4}, \cdots, A_{t-1}, S_{t}}\left[f\left(S_{t}\right)\right]\right\}\right\} \\
& =\mathbb{E}_{S_{2} \sim d}\left\{\mathbb{E}_{A_{2}, S_{3}, A_{3}, S_{4}, \cdots, A_{t-1}, S_{t}}\left[f\left(S_{t}\right)\right]\right\} \quad \text { (由引理 } 7.4 \text { 得出） } \\
& =\mathbb{E}_{S_{2} \sim d}\left\{\mathbb{E}_{A_{2} \sim \pi, S_{3} \sim p}\left\{\mathbb{E}_{A_{3}, S_{4}, A_{4}, S_{5}, \cdots, A_{t-1}, S_{t}}\left[f\left(S_{t}\right)\right]\right\}\right\} \\
& =\mathbb{E}_{S_{3} \sim d}\left\{\mathbb{E}_{A_{3}, S_{4}, A_{4}, S_{5}, \cdots, A_{t-1}, S_{t}}\left[f\left(S_{t}\right)\right]\right\} \quad \text { (由引理 } 7.4 \text { 得出) } \\
& =\mathbb{E}_{S_{t-1} \sim d}\left\{\mathbb{E}_{A_{t-1} \sim \pi, S_{t} \sim p}\left\{f\left(S_{t}\right)\right\}\right\} \\
& =\mathbb{E}_{S_{t} \sim d}\left\{f\left(S_{t}\right)\right\} \text {. }
\end{aligned}
$$

(由引理 $7.4$ 得出)

设 $\boldsymbol{g}(s, a ; \boldsymbol{\theta}) \triangleq Q_{\pi}(s, a) \cdot \frac{\partial \ln \pi(a \mid s ; \boldsymbol{\theta})}{\partial \boldsymbol{\theta}}$ 。设一局游戏在第 $n$ 步之后结束。由引理 $7.3$ 与上面 的公式可得：

$$
\begin{aligned}
& \frac{\partial J(\boldsymbol{\theta})}{\partial \boldsymbol{\theta}}=\mathbb{E}_{S_{1}, A_{1}}\left[\boldsymbol{g}\left(S_{1}, A_{1} ; \boldsymbol{\theta}\right)\right] \\
& +\gamma \cdot \mathbb{E}_{S_{1}, A_{1}, S_{2}, A_{2}}\left[\boldsymbol{g}\left(S_{2}, A_{2} ; \boldsymbol{\theta}\right)\right] \\
& +\gamma^{2} \cdot \mathbb{E}_{S_{1}, A_{1}, S_{2}, A_{2}, S_{3}, A_{3}}\left[\boldsymbol{g}\left(S_{3}, A_{3} ; \boldsymbol{\theta}\right)\right] \\
& +\cdots \\
& \left.+\gamma^{n-1} \cdot \mathbb{E}_{S_{1}, A_{1}, S_{2}, A_{2}, S_{3}, A_{3}, \cdots S_{n}, A_{n}}\left[\boldsymbol{g}\left(S_{n}, A_{n} ; \boldsymbol{\theta}\right)\right]\right] \\
& =\mathbb{E}_{S_{1} \sim d(\cdot)}\left\{\mathbb{E}_{A_{1} \sim \pi\left(\cdot \mid S_{1} ; \boldsymbol{\theta}\right)}\left[\boldsymbol{g}\left(S_{1}, A_{1} ; \boldsymbol{\theta}\right)\right]\right\} \\
& +\gamma \cdot \mathbb{E}_{S_{2} \sim d(\cdot)}\left\{\mathbb{E}_{A_{2} \sim \pi\left(\cdot \mid S_{2} ; \boldsymbol{\theta}\right)}\left[\boldsymbol{g}\left(S_{2}, A_{2} ; \boldsymbol{\theta}\right)\right]\right\} \\
& +\gamma^{2} \cdot \mathbb{E}_{S_{3} \sim d(\cdot)}\left\{\mathbb{E}_{A_{3} \sim \pi\left(\cdot \mid S_{3} ; \boldsymbol{\theta}\right)}\left[\boldsymbol{g}\left(S_{3}, A_{3} ; \boldsymbol{\theta}\right)\right]\right\} \\
& +\cdots \\
& +\gamma^{n-1} \cdot \mathbb{E}_{S_{n} \sim d(\cdot)}\left\{\mathbb{E}_{A_{n} \sim \pi\left(\cdot \mid S_{n} ; \boldsymbol{\theta}\right)}\left[\boldsymbol{g}\left(S_{n}, A_{n} ; \boldsymbol{\theta}\right)\right]\right\} \\
& =\left(1+\gamma+\gamma^{2}+\cdots+\gamma^{n-1}\right) \cdot \mathbb{E}_{S \sim d(\cdot)}\left\{\mathbb{E}_{A \sim \pi(\cdot \mid S ; \boldsymbol{\theta})}[\boldsymbol{g}(S, A ; \boldsymbol{\theta})]\right\} \text {. }
\end{aligned}
$$

由此可得定理 7.5。

### 3 近似策略梯度

先复习一下前两小节的内容。策略学习可以表述为这样一个优化问题:

$$
\max _{\boldsymbol{\theta}}\left\{J(\boldsymbol{\theta}) \triangleq \mathbb{E}_{S}\left[V_{\pi}(S)\right]\right\} .
$$

求解这个最大化问题最简单的算法就是梯度上升:

$$
\boldsymbol{\theta} \leftarrow \boldsymbol{\theta}+\beta \cdot \nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) .
$$

其中的 $\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta})$ 是策略梯度。策略梯度定理证明:

$$
\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta})=\mathbb{E}_{S}\left[\mathbb{E}_{A \sim \pi(\cdot \mid S ; \boldsymbol{\theta})}\left[Q_{\pi}(S, A) \cdot \nabla_{\boldsymbol{\theta}} \ln \pi(A \mid S ; \boldsymbol{\theta})\right]\right] .
$$

解析求出这个期望是不可能的, 因为我们并不知道状态 $S$ 概率密度函数; 即使我们知道 $S$ 的概率密度函数, 能够通过连加或者定积分求出期望, 我们也不愿意这样做, 因为连 加或者定积分的计算量非常大。

回忆一下, 第 2 章介绍了期望的蒙特卡洛近似方法, 可以将这种方法用于近似策略 梯度。每次从环境中观测到一个状态 $s$, 它相当于随机变量 $S$ 的观测值。然后再根据当 前的策略网络（策略网络的参数必须是最新的）随机抽样得出一个动作:

$$
a \sim \pi(\cdot \mid s ; \boldsymbol{\theta}) .
$$

计算随机梯度：

$$
\boldsymbol{g}(s, a ; \boldsymbol{\theta}) \triangleq Q_{\pi}(s, a) \cdot \nabla_{\boldsymbol{\theta}} \ln \pi(a \mid s ; \boldsymbol{\theta}) .
$$

很显然, $\boldsymbol{g}(s, a ; \boldsymbol{\theta})$ 是策略梯度 $\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta})$ 的无偏估计：

$$
\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta})=\mathbb{E}_{S}\left[\mathbb{E}_{A \sim \pi(\cdot \mid S ; \boldsymbol{\theta})}[\boldsymbol{g}(S, A ; \boldsymbol{\theta})]\right] .
$$

于是我们得到下面的结论：

## 结论 $7.1$

随机梯度 $\boldsymbol{g}(s, a ; \boldsymbol{\theta}) \triangleq Q_{\pi}(s, a) \cdot \nabla_{\boldsymbol{\theta}} \ln \pi(a \mid s ; \boldsymbol{\theta})$ 是策略梯度 $\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta})$ 的无偏估计。

应用上述结论, 我们可以做随机梯度上升来更新 $\boldsymbol{\theta}$, 使得目标函数 $J(\boldsymbol{\theta})$ 逐渐增长:

$$
\boldsymbol{\theta} \leftarrow \boldsymbol{\theta}+\beta \cdot \boldsymbol{g}(s, a ; \boldsymbol{\theta}) .
$$

此处的 $\beta$ 是学习率, 需要手动调整。但是这种方法仍然不可行, 我们计算不出 $\boldsymbol{g}(s, a ; \boldsymbol{\theta})$, 原因在于我们不知道动作价值函数 $Q_{\pi}(s, a)$ 。在后面两节中, 我们用两种方法对 $Q_{\pi}(s, a)$ 做近似：一种方法是 REINFORCE, 用实际观测的回报 $u$ 近似 $Q_{\pi}(s, a)$; 另一种方法是 actor-critic, 用神经网络 $q(s, a ; \boldsymbol{w})$ 近似 $Q_{\pi}(s, a)$ 。

### REINFORCE

策略梯度方法用 $\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta})$ 的近似来更新策略网络参数 $\boldsymbol{\theta}$, 从而增大目标函数。上一 节中, 我们推导出策略梯度 $\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta})$ 的无偏估计, 即下面的随机梯度:

$$
\boldsymbol{g}(s, a ; \boldsymbol{\theta}) \triangleq Q_{\pi}(s, a) \cdot \nabla_{\boldsymbol{\theta}} \ln \pi(a \mid s ; \boldsymbol{\theta}) .
$$

但是其中的动作价值函数 $Q_{\pi}$ 是末知的, 导致无法直接计算 $\boldsymbol{g}(s, a ; \boldsymbol{\theta})$ 。REINFORCE 进一 步对 $Q_{\pi}$ 做蒙特卡洛近似，把它替换成回报 $u$ 。

### REINFORCE 的简化推导

设一局游戏有 $n$ 步，一局中的奖励记作 $R_{1}, \cdots, R_{n}$ 。回忆一下， $t$ 时刻的折扣回报 定义为:

$$
U_{t}=\sum_{k=t}^{n} \gamma^{k-t} \cdot R_{k} .
$$

而动作价值定义为 $U_{t}$ 的条件期望：

$$
Q_{\pi}\left(s_{t}, a_{t}\right)=\mathbb{E}\left[U_{t} \mid S_{t}=s_{t}, A_{t}=a_{t}\right] .
$$

我们可以用蒙特卡洛近似上面的条件期望。从时刻 $t$ 开始, 智能体完成一局游戏, 观测 到全部奖励 $r_{t}, \cdots, r_{n}$, 然后可以计算出 $u_{t}=\sum_{k=t}^{n} \gamma^{k-t} \cdot r_{k}$ 。因为 $u_{t}$ 是随机变量 $U_{t}$ 的观 测值, 所以 $u_{t}$ 是上面公式中期望的蒙特卡洛近似。在实践中, 可以用 $u_{t}$ 代替 $Q_{\pi}\left(s_{t}, a_{t}\right)$, 那么随机梯度 $\boldsymbol{g}\left(s_{t}, a_{t} ; \boldsymbol{\theta}\right)$ 可以近似成

$$
\tilde{\boldsymbol{g}}\left(s_{t}, a_{t} ; \boldsymbol{\theta}\right)=u_{t} \cdot \nabla_{\boldsymbol{\theta}} \ln \pi\left(a_{t} \mid s_{t} ; \boldsymbol{\theta}\right) .
$$

$\tilde{\boldsymbol{g}}$ 是 $\boldsymbol{g}$ 的无偏估计, 所以也是策略梯度 $\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta})$ 的无偏估计; $\tilde{\boldsymbol{g}}$ 也是一种随机梯度。

我们可以用反向传播计算出 $\ln \pi$ 关于 $\boldsymbol{\theta}$ 的梯度, 而且可以实际观测到 $u_{t}$, 于是我们 可以实际计算出随机梯度 $\tilde{\boldsymbol{g}}$ 的值。有了随机梯度的值, 我们可以做随机梯度上升更新策 略网络参数 $\theta$ :

$$
\boldsymbol{\theta} \leftarrow \boldsymbol{\theta}+\beta \cdot \tilde{\boldsymbol{g}}\left(s_{t}, a_{t} ; \boldsymbol{\theta}\right)
$$

根据上述推导, 我们得到了训练策略网络的算法, 即 REINFORCE。

### 2 训练流程

当前策略网络的参数是 $\boldsymbol{\theta}_{\text {now }}$ 。REINFORCE 执行下面的步骤对策略网络的参数做一 次更新：

![](https://cdn.mathpix.com/cropped/2023_02_03_f46f5cf0e4de5b9996dcg-123.jpg?height=51&width=1328&top_left_y=2376&top_left_x=290)

$$
s_{1}, a_{1}, r_{1}, s_{2}, a_{2}, r_{2}, \quad \cdots, s_{n}, a_{n}, r_{n} .
$$

2. 计算所有的回报：

$$
u_{t}=\sum_{k=t}^{n} \gamma^{k-t} \cdot r_{k}, \quad \forall t=1, \cdots, n
$$

3. 用 $\left\{\left(s_{t}, a_{t}\right)\right\}_{t=1}^{n}$ 作为数据, 做反向传播计算:

$$
\nabla_{\boldsymbol{\theta}} \ln \pi\left(a_{t} \mid s_{t} ; \boldsymbol{\theta}_{\text {now }}\right), \quad \forall t=1, \cdots, n .
$$

4. 做随机梯度上升更新策略网络参数：

$$
\boldsymbol{\theta}_{\text {new }} \leftarrow \boldsymbol{\theta}_{\text {now }}+\beta \cdot \sum_{t=1}^{n} \gamma^{t-1} \cdot \underbrace{u_{t} \cdot \nabla_{\boldsymbol{\theta}} \ln \pi\left(a_{t} \mid s_{t} ; \boldsymbol{\theta}_{\text {now }}\right)}_{\text {即随机梯度 } \tilde{\boldsymbol{g}}\left(s_{t}, a_{t} ; \boldsymbol{\theta}_{\text {now }}\right)} .
$$

注在算法最后一步中, 随机梯度前面乘以系数 $\gamma^{t-1}$ 。读者可能会好奇, 为什么需要这个系 数呢? 原因是这样的: 前面 REINFORCE 的推导是简化的, 而非严谨的数学推导; 按照我 们简化的推导, 不应该乘以系数 $\gamma^{t-1}$ 。下一小节做严格的数学推导, 得出的 REINFORCE 算法需要系数 $\gamma^{t-1}$ 。读者只要知道这个事实就行了, 不必读懂下一小节的数学推导。

注 REINFORCE 属于同策略 (on-policy), 要求行为策略（behavior policy) 与目标策略 (target policy) 相同, 两者都必须是策略网络 $\pi\left(a \mid s ; \boldsymbol{\theta}_{\text {now }}\right)$, 其中 $\boldsymbol{\theta}_{\text {now }}$ 是策略网络当前的 参数。所以经验回放不适用于 REINFORCE。

### REINFORCE 严格的推导

第 7.4.1 小节对策略梯度做近似, 推导出 REINFORCE 算法。那种推导是简化过的, 帮助读者理解 REINFORCE 算法, 但实际上那种推导并不够严谨。本小节做严格的数学 推导, 对策略梯度做近似, 得出真正的 REINFORCE 算法。建议对数学证明不感兴趣的 读者跳过本小节。

根据定义, $\boldsymbol{g}(s, a ; \boldsymbol{\theta}) \triangleq Q_{\pi}(s, a) \cdot \nabla_{\boldsymbol{\theta}} \ln \pi(a \mid s ; \boldsymbol{\theta})$ 。引理 $7.3$ 把策略梯度 $\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta})$ 表 示成期望的连加：

$$
\begin{aligned}
\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta})= & \mathbb{E}_{S_{1}, A_{1}}\left[\boldsymbol{g}\left(S_{1}, A_{1} ; \boldsymbol{\theta}\right)\right] \\
& +\gamma \cdot \mathbb{E}_{S_{1}, A_{1}, S_{2}, A_{2}}\left[\boldsymbol{g}\left(S_{2}, A_{2} ; \boldsymbol{\theta}\right)\right] \\
& +\gamma^{2} \cdot \mathbb{E}_{S_{1}, A_{1}, S_{2}, A_{2}, S_{3}, A_{3}}\left[\boldsymbol{g}\left(S_{3}, A_{3} ; \boldsymbol{\theta}\right)\right] \\
& +\cdots \\
& +\gamma^{n-1} \cdot \mathbb{E}_{S_{1}, A_{1}, S_{2}, A_{2}, S_{3}, A_{3}, \cdots, S_{n}, A_{n}}\left[\boldsymbol{g}\left(S_{n}, A_{n} ; \boldsymbol{\theta}\right)\right]
\end{aligned}
$$

我们可以对期望做蒙特卡洛近似。首先观测到第一个状态 $S_{1}=s_{1}$ 。然后用最新的策略网 络 $\pi\left(a \mid s ; \boldsymbol{\theta}_{\text {now }}\right)$ 控制智能体与环境交互, 观测到到轨迹

$$
s_{1}, a_{1}, r_{1}, s_{2}, a_{2}, r_{2}, \cdots, s_{n}, a_{n}, r_{n} .
$$

对公式 (7.10) 中的期望做蒙特卡洛近似, 得到：

$$
\nabla_{\boldsymbol{\theta}} J\left(\boldsymbol{\theta}_{\text {now }}\right) \approx \boldsymbol{g}\left(s_{1}, a_{1} ; \boldsymbol{\theta}_{\text {now }}\right)+\gamma \cdot \boldsymbol{g}\left(s_{2}, a_{2} ; \boldsymbol{\theta}_{\text {now }}\right)+\cdots+\gamma^{n-1} \cdot \boldsymbol{g}\left(s_{n}, a_{n} ; \boldsymbol{\theta}_{\text {now }}\right) .
$$

进一步把 $\boldsymbol{g}\left(s_{t}, a_{t} ; \boldsymbol{\theta}_{\text {now }}\right) \triangleq Q_{\pi}\left(s_{t}, a_{t}\right) \cdot \nabla_{\boldsymbol{\theta}} \ln \pi\left(a_{t} \mid s_{t} ; \boldsymbol{\theta}_{\text {now }}\right)$ 中的 $Q_{\pi}\left(s_{t}, a_{t}\right)$ 替换成 $u_{t}$, 那 么 $\boldsymbol{g}\left(s_{t}, a_{t} ; \boldsymbol{\theta}_{\text {now }}\right)$ 就被近似成为

$$
\boldsymbol{g}\left(s_{t}, a_{t} ; \boldsymbol{\theta}_{\text {now }}\right) \approx u_{t} \cdot \nabla_{\boldsymbol{\theta}} \ln \pi\left(a_{t} \mid s_{t} ; \boldsymbol{\theta}_{\text {now }}\right) .
$$

经过上述两次近似, 策略梯度被近似成为下面的随机梯度

$$
\nabla_{\boldsymbol{\theta}} J\left(\boldsymbol{\theta}_{\text {now }}\right) \approx \sum_{t=1}^{n} \gamma^{t-1} \cdot u_{t} \cdot \nabla_{\boldsymbol{\theta}} \ln \pi\left(a_{t} \mid s_{t} ; \boldsymbol{\theta}_{\text {now }}\right) .
$$

这样就得到了 REINFORCE 算法的随机梯度上升公式:

$$
\boldsymbol{\theta}_{\text {new }} \leftarrow \boldsymbol{\theta}_{\text {now }}+\beta \cdot \sum_{t=1}^{n} \gamma^{t-1} \cdot u_{t} \cdot \nabla_{\boldsymbol{\theta}} \ln \pi\left(a_{t} \mid s_{t} ; \boldsymbol{\theta}_{\text {now }}\right)
$$



### Actor-Critic

策略梯度方法用策略梯度 $\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta})$ 更新策略网络参数 $\boldsymbol{\theta}$, 从而增大目标函数。第 $7.2$ 节推导出策略梯度 $\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta})$ 的无偏估计, 即下面的随机梯度：

$$
\boldsymbol{g}(s, a ; \boldsymbol{\theta}) \triangleq Q_{\pi}(s, a) \cdot \nabla_{\boldsymbol{\theta}} \ln \pi(a \mid s ; \boldsymbol{\theta}) .
$$

但是其中的动作价值函数 $Q_{\pi}$ 是末知的, 导致无法直接计算 $\boldsymbol{g}(s, a ; \boldsymbol{\theta})$ 。上一节的 REINFORCE 用实际观测的回报近似 $Q_{\pi}$, 本节的 actor-critic 方法用神经网络近似 $Q_{\pi}$ 。

### 1 价值网络

Actor-critic 方法用一个神经网络近似动作价值函数 $Q_{\pi}(s, a)$, 这个神经网络叫做“价 值网络”, 记为 $q(s, a ; \boldsymbol{w})$, 其中的 $\boldsymbol{w}$ 表示神经网络中可训练的参数。价值网络的输入是 状态 $s$, 输出是每个动作的价值。动作空间 $\mathcal{A}$ 中有多少种动作, 那么价值网络的输出就 是多少维的向量, 向量每个元素对应一个动作。举个例子, 动作空间是 $\mathcal{A}=\{$ 左, 右, 上 $\}$, 价值网络的输出是

$$
\begin{aligned}
& q(s, \text { 左; } \boldsymbol{w})=219, \\
& q(s, \text { 右; } \boldsymbol{w})=-73, \\
& q(s, \text { 上 } ; \boldsymbol{w})=580 .
\end{aligned}
$$

神经网络的结构见图 7.2。

![](https://cdn.mathpix.com/cropped/2023_02_03_f46f5cf0e4de5b9996dcg-126.jpg?height=389&width=1400&top_left_y=1733&top_left_x=402)

图 7.2: 价值网络 $q(s, a ; \boldsymbol{w})$ 的结构。输入是状态 $s$; 输出是每个动作的价值。

虽然价值网络 $q(s, a ; \boldsymbol{w})$ 与之前学的 DQN 有相同的结构, 但是两者的意义不同, 训 练算法也不同。

- 价值网络是对动作价值函数 $Q_{\pi}(s, a)$ 的近似。而 DQN 则是对最优动作价值函数 $Q_{\star}(s, a)$ 的近似。

- 对价值网络的训练使用的是 SARSA 算法, 它属于同策略, 不能用经验回放。对 DQN 的训练使用的是 $\mathrm{Q}$ 学习算法, 它属于异策略, 可以用经验回放。

### 2 算法的推导

Actor-critic 翻译成 “演员一评委”方法。策略网络 $\pi(a \mid s ; \boldsymbol{\theta})$ 相当于演员, 它基于状态 $s$ 做出动作 $a$ 。价值网络 $q(s, a ; \boldsymbol{w})$ 相当于评委, 它给演员的表现打分, 评价在状态 $s$ 的情 况下做出动作 $a$ 的好坏程度。策略网络（演员）和价值网络（评委）的关系如图 $7.3$ 所示。

![](https://cdn.mathpix.com/cropped/2023_02_03_f46f5cf0e4de5b9996dcg-127.jpg?height=468&width=1128&top_left_y=591&top_left_x=407)

图 7.3: Actor-critic 方法中策略网络（演员）和价值网络（评委）的关系图。

读者可能会对图 $7.3$ 感到不解：为什么不直接把奖励 $R$ 反馈给策略网络（演员，,而 要用价值网络（评委）这样一个中介呢? 原因是这样的：策略学习的目标函数 $J(\boldsymbol{\theta})$ 是回 报 $U$ 的期望, 而不是奖励 $R$ 的期望; 注意回报 $U$ 和奖励 $R$ 的区别。虽然能观测到当前的 奖励 $R$, 但是它对策略网络是毫无意义的; 训练策略网络（演员）需要的是回报 $U$, 而不 是奖励 $R$ 。价值网络（评委）能够估算出回报 $U$ 的期望, 因此能帮助训练策略网络（演 员)。

训练策略网络 (演员) : 策略网络 (演员) 想要改进自己的演技, 但是演员自己不知 道什么样的表演才算更好, 所以需要价值网络（评委）的帮助。在演员做出动作 $a$ 之后, 评委会打一个分数 $\widehat{q} \triangleq q(s, a ; \boldsymbol{w})$, 并把分数反馈给演员, 帮助演员做出改进。演员利用 当前状态 $s$, 自己的动作 $a$, 以及评委的打分 $\widehat{q}$, 计算近似策略梯度, 然后更新自己的参 数 $\boldsymbol{\theta}$ (相当于改变自己的技术)。通过这种方式, 演员的表现越来越受评委的好评, 于是 演员的获得的评分 $\widehat{q}$ 越来越高。

训练策略网络的基本想法是用策略梯度 $\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta})$ 的近似来更新参数 $\boldsymbol{\theta}$ 。之前我们推 导过策略梯度的无偏估计：

$$
\boldsymbol{g}(s, a ; \boldsymbol{\theta}) \triangleq Q_{\pi}(s, a) \cdot \nabla_{\boldsymbol{\theta}} \ln \pi(a \mid s ; \boldsymbol{\theta}) .
$$

价值网络 $q(s, a ; \boldsymbol{w})$ 是对动作价值函数 $Q_{\pi}(s, a)$ 的近似, 所以把上面公式中的 $Q_{\pi}$ 替换成 价值网络, 得到近似策略梯度：

$$
\widehat{\boldsymbol{g}}(s, a ; \boldsymbol{\theta}) \triangleq \underbrace{q(s, a ; \boldsymbol{w})}_{\text {评委的打分 }} \cdot \nabla_{\boldsymbol{\theta}} \ln \pi(a \mid s ; \boldsymbol{\theta}) .
$$

最后做梯度上升更新策略网络的参数：

$$
\boldsymbol{\theta} \leftarrow \boldsymbol{\theta}+\beta \cdot \widehat{\boldsymbol{g}}(s, a ; \boldsymbol{\theta})
$$

注 用上述方式更新参数之后, 会让评委打出的分数越来越高, 原因是这样的。状态价值 函数 $V_{\pi}(s)$ 可以近似成为:

$$
v(s ; \boldsymbol{\theta})=\mathbb{E}_{A \sim \pi(\cdot \mid s ; \boldsymbol{\theta})}[q(s, A ; \boldsymbol{w})] .
$$

因此可以将 $v(s ; \boldsymbol{\theta})$ 看做评委打分的均值。不难证明, 公式 (7.11) 中定义的近似策略梯度 $\widehat{\boldsymbol{g}}(s, a ; \boldsymbol{\theta})$ 的期望等于 $v(s ; \boldsymbol{\theta})$ 关于 $\boldsymbol{\theta}$ 的梯度；

$$
\nabla_{\boldsymbol{\theta}} v(s ; \boldsymbol{\theta})=\mathbb{E}_{A \sim \pi(\cdot \mid s ; \boldsymbol{\theta})}[\widehat{\boldsymbol{g}}(s, A ; \boldsymbol{\theta})] .
$$

因此, 用公式 $7.12$ 中的梯度上升更新 $\boldsymbol{\theta}$, 会让 $v(s ; \boldsymbol{\theta})$ 变大, 也就是让评委打分的均值更 高。

训练价值网络（评委）: 通过以上分析, 我们不难发现上述训练策略网络（演员）的 方法不是真正让演员表现更好, 只是让演员更迎合评委的喜好而已。因此, 评委的水平 也很重要, 只有当评委的打分 $\widehat{q}$ 真正反映出动作价值 $Q_{\pi}$, 演员的水平才能真正提高。初 始的时候, 价值网络的参数 $\boldsymbol{w}$ 是随机的, 也就是说评委的打分是瞎猜。可以用 SARSA 算法更新 $\boldsymbol{w}$, 提高评委的水平。每次从环境中观测到一个奖励 $r$, 把 $r$ 看做是真相, 用 $r$ 来校准评委的打分。

第 $5.1$ 节已经推导过 SARSA算法, 现在我们再回顾一下。在 $t$ 时刻, 价值网络输出

$$
\widehat{q}_{t}=q\left(s_{t}, a_{t} ; \boldsymbol{w}\right),
$$

它是对动作价值函数 $Q_{\pi}\left(s_{t}, a_{t}\right)$ 的估计。在 $t+1$ 时刻, 实际观测到 $r_{t}, s_{t+1}, a_{t+1}$, 于是 可以计算 TD 目标

$$
\widehat{y}_{t} \triangleq r_{t}+\gamma \cdot q\left(s_{t+1}, a_{t+1} ; \boldsymbol{w}\right)
$$

它也是对动作价值函数 $Q_{\pi}\left(s_{t}, a_{t}\right)$ 的估计。由于 $\widehat{y}_{t}$ 部分基于实际观测到的奖励 $r_{t}$, 我们 认为 $\widehat{y}_{t}$ 比 $q\left(s_{t}, a_{t} ; \boldsymbol{w}\right)$ 更接近事实真相。所以把 $\widehat{y}_{t}$ 固定住, 鼓励 $q\left(s_{t}, a_{t} ; \boldsymbol{w}\right)$ 去接近 $\widehat{y}_{t}$ 。 SARSA 算法具体这样更新价值网络参数 $\boldsymbol{w}$ 。定义损失函数:

$$
L(\boldsymbol{w}) \triangleq \frac{1}{2}\left[q\left(s_{t}, a_{t} ; \boldsymbol{w}\right)-\widehat{y}_{t}\right]^{2} .
$$

设 $\widehat{q}_{t} \triangleq q\left(s_{t}, a_{t} ; \boldsymbol{w}\right)$ 。损失函数的梯度是：

$$
\nabla_{\boldsymbol{w}} L(\boldsymbol{w})=\underbrace{\left(\widehat{q}_{t}-\widehat{y}_{t}\right)}_{\text {TD 误差 } \delta_{t}} \cdot \nabla_{\boldsymbol{w}} q\left(s_{t}, a_{t} ; \boldsymbol{w}\right) .
$$

做一轮梯度下降更新 $\boldsymbol{w}$ :

$$
\boldsymbol{w} \leftarrow \boldsymbol{w}-\alpha \cdot \nabla_{\boldsymbol{w}} L(\boldsymbol{w})
$$

这样更新 $\boldsymbol{w}$ 可以让 $q\left(s_{t}, a_{t} ; \boldsymbol{w}\right)$ 更接近 $\widehat{y}_{t}$ 。可以这样理解 SARSA：用观测到的奖励 $r_{t}$ 来 “校准”评委的打分 $q\left(s_{t}, a_{t} ; \boldsymbol{w}\right)$ 。

### 3 训练流程

下面概括 actor-critic 训练流程。设当前策略网络参数是 $\boldsymbol{\theta}_{\text {now }}$, 价值网络参数是 $\boldsymbol{w}_{\text {now }}$ 。 执行下面的步骤, 将参数更新成 $\theta_{\text {new }}$ 和 $\boldsymbol{w}_{\text {new }}$ :

1. 观测到当前状态 $s_{t}$, 根据策略网络做决策: $a_{t} \sim \pi\left(\cdot \mid s_{t} ; \boldsymbol{\theta}_{\text {now }}\right)$, 并让智能体执行动 作 $a_{t}$ 。

2. 从环境中观测到奖励 $r_{t}$ 和新的状态 $s_{t+1}$ 。

3. 根据策略网络做决策： $\tilde{a}_{t+1} \sim \pi\left(\cdot \mid s_{t+1} ; \boldsymbol{\theta}_{\mathrm{now}}\right)$, 但不让智能体执行动作 $\tilde{a}_{t+1 \text { 。 }}$

4. 让价值网络打分:

$$
\widehat{q}_{t}=q\left(s_{t}, a_{t} ; \boldsymbol{w}_{\text {now }}\right) \quad \text { 和 } \quad \widehat{q}_{t+1}=q\left(s_{t+1}, \tilde{a}_{t+1} ; \boldsymbol{w}_{\text {now }}\right)
$$

5. 计算 TD 目标和 TD 误差:

$$
\widehat{y}_{t}=r_{t}+\gamma \cdot \widehat{q}_{t+1} \quad \text { 和 } \quad \delta_{t}=\widehat{q}_{t}-\widehat{y}_{t} .
$$

6. 更新价值网络:

$$
\boldsymbol{w}_{\text {new }} \leftarrow \boldsymbol{w}_{\text {now }}-\alpha \cdot \delta_{t} \cdot \nabla_{\boldsymbol{w}} q\left(s_{t}, a_{t} ; \boldsymbol{w}_{\text {now }}\right)
$$

7. 更新策略网络:

$$
\boldsymbol{\theta}_{\text {new }} \leftarrow \boldsymbol{\theta}_{\text {now }}+\beta \cdot \widehat{q}_{t} \cdot \nabla_{\boldsymbol{\theta}} \ln \pi\left(a_{t} \mid s_{t} ; \boldsymbol{\theta}_{\text {now }}\right)
$$

### 4 用目标网络改进训练

第 $6.2$ 节讨论了 $\mathrm{Q}$ 学习中的自举及其危害, 以及用目标网络 (target network) 缓解自 举造成的偏差。SARSA 算法中也存在自举一一即用价值网络自己的估值 $\widehat{q}_{t+1}$ 去更新价 值网络自己; 我们同样可以用目标网络计算 TD 目标, 从而缓解偏差。把目标网络记作 $q\left(s, a ; \boldsymbol{w}^{-}\right)$, 它的结构与价值网络相同, 但是参数不同。使用目标网络计算 TD 目标, 那 么 actor-critic 的训练就变成了:

1. 观测到当前状态 $s_{t}$, 根据策略网络做决策: $a_{t} \sim \pi\left(\cdot \mid s_{t} ; \boldsymbol{\theta}_{\text {now }}\right)$, 并让智能体执行动 作 $a_{t}$ 。

2. 从环境中观测到奖励 $r_{t}$ 和新的状态 $s_{t+1}$ 。

3. 根据策略网络做决策: $\tilde{a}_{t+1} \sim \pi\left(\cdot \mid s_{t+1} ; \boldsymbol{\theta}_{\text {now }}\right)$, 但是不让智能体执行动作 $\tilde{a}_{t+1}$ 。

4. 让价值网络给 $\left(s_{t}, a_{t}\right)$ 打分：

$$
\widehat{q}_{t}=q\left(s_{t}, a_{t} ; \boldsymbol{w}_{\text {now }}\right) .
$$

5. 让目标网络给 $\left(s_{t+1}, \tilde{a}_{t+1}\right)$ 打分:

$$
\widehat{q}_{t+1}^{-}=q\left(s_{t+1}, \tilde{a}_{t+1} ; \boldsymbol{w}_{\text {now }}^{-}\right) .
$$

6. 计算 TD 目标和 TD 误差:

$$
\widehat{y}_{t}^{-}=r_{t}+\gamma \cdot \widehat{q}_{t+1}^{-} \quad \text { 和 } \quad \delta_{t}=\widehat{q}_{t}-\widehat{y}_{t}^{-} .
$$

7. 更新价值网络:

$$
\boldsymbol{w}_{\text {new }} \leftarrow \boldsymbol{w}_{\text {now }}-\alpha \cdot \delta_{t} \cdot \nabla_{\boldsymbol{w}} q\left(s_{t}, a_{t} ; \boldsymbol{w}_{\text {now }}\right) .
$$

8. 更新策略网络：

$$
\boldsymbol{\theta}_{\text {new }} \leftarrow \boldsymbol{\theta}_{\text {now }}+\beta \cdot \widehat{q}_{t} \cdot \nabla_{\boldsymbol{\theta}} \ln \pi\left(a_{t} \mid s_{t} ; \boldsymbol{\theta}_{\text {now }}\right) .
$$

9. 设 $\tau \in(0,1)$ 是需要手动调整的超参数。做加权平均更新目标网络的参数：

$$
\boldsymbol{w}_{\text {new }}^{-} \leftarrow \tau \cdot \boldsymbol{w}_{\text {new }}+(1-\tau) \cdot \boldsymbol{w}_{\text {now }}^{-} .
$$



## 第 7 章 知识点

- 可以用神经网络 $\pi(a \mid s ; \boldsymbol{\theta})$ 近似策略函数。策略学习的目标函数是 $J(\boldsymbol{\theta})=\mathbb{E}_{S}\left[V_{\pi}(S)\right]$, 它的值越大，意味着策略越好。

- 策略梯度指的是 $J(\boldsymbol{\theta})$ 关于策略了参数 $\boldsymbol{\theta}$ 的梯度。策略梯度定理将策略梯度表示成

$$
\boldsymbol{g}(s, a ; \boldsymbol{\theta}) \triangleq Q_{\pi}(s, a) \cdot \nabla_{\boldsymbol{\theta}} \ln \pi(a \mid s ; \boldsymbol{\theta})
$$

的期望。

- REINFORCE 算法用实际观测的回报 $u$ 近似 $Q_{\pi}(s, a)$, 从而把 $\boldsymbol{g}(s, a ; \boldsymbol{\theta})$ 近似成:

$$
\tilde{\boldsymbol{g}}(s, a ; \boldsymbol{\theta}) \triangleq u \cdot \nabla_{\boldsymbol{\theta}} \ln \pi(a \mid s ; \boldsymbol{\theta}) .
$$

REINFORCE 算法做梯度上升更新策略网络： $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta}+\beta \cdot \tilde{\boldsymbol{g}}(s, a ; \boldsymbol{\theta})$ 。

- Actor-critic 用价值网络 $q(s, a ; \boldsymbol{w})$ 近似 $Q_{\pi}(s, a)$, 从而把 $\boldsymbol{g}(s, a ; \boldsymbol{\theta})$ 近似成:

$$
\widehat{\boldsymbol{g}}(s, a ; \boldsymbol{\theta}) \triangleq q(s, a ; \boldsymbol{w}) \cdot \nabla_{\boldsymbol{\theta}} \ln \pi(a \mid s ; \boldsymbol{\theta}) .
$$

Actor-critic 用 SARSA 算法更新价值网络 $q$, 用梯度上升更新策略网络: $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta}+\beta$. $\widehat{\boldsymbol{g}}(s, a ; \boldsymbol{\theta})$ 。

## 第 7 章 相关文献

REINFORCE 由 Williams 在 1987 年提出 ${ }^{[125-126]}$ 。Actor-critic 由 Barto 等人在 1983

![](https://cdn.mathpix.com/cropped/2023_02_03_f46f5cf0e4de5b9996dcg-131.jpg?height=63&width=1450&top_left_y=411&top_left_x=243)
Marbach 和 Tsitsiklis 1999 年的论文 ${ }^{[73]}$ 和 Sutton 等人 2000 年的论文 ${ }^{[105]}$ 独立提出。

## 第 7 章习题

1. 把策略网络记作 $\pi(a \mid s ; \boldsymbol{\theta})$ 。请计算: $\sum_{a \in \mathcal{A}}|\pi(a \mid s ; \boldsymbol{\theta})|=$

2. 为什么策略网络输出层用 softmax 做激活函数?

3. 状态价值函数 $V_{\pi}\left(s_{t}\right)$ 依赖于末来的状态 $s_{t+1}, s_{t+1}, \cdots$ 。
A. 上述说法正确。
B. 上述说法错误。

4. 设 $\ln x$ 为 $x$ 的自然对数。它的导数是 $\frac{d \ln x}{d x}=$
A. $x^{2}$
B. $x$
C. $\frac{1}{x}$
D. $\ln x$
E. $e^{x}$

5. REINFORCE 的原理是对策略梯度 $\frac{\partial J(\theta)}{\partial \theta}$ 做蒙特卡洛近似。请你总结一下 REINFORCE 做了哪些近似。

6. 在 actor-critic 的训练中使用目标网络, 目的在于
A. 缓解价值网络的偏差
B. 缓解策略网络的偏差
