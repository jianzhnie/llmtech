

## 第 9 章 策略学习高级技巧

本章介绍策略学习的高级技巧。第 $9.1$ 节介绍置信域策略优化 (TRPO), 它是一种 策略学习方法, 可以代替策略梯度方法。第 $9.2$ 节介绍熵正则, 可以用在所有的策略学习 方法中。

### Trust Region Policy Optimization (TRPO)

置信域策略优化 (trust region policy optimization, TRPO) 是一种策略学习方法, 跟以 前学的策略梯度有很多相似之处。跟策略梯度方法相比, TRPO 有两个优势: 第一, TRPO 表现更稳定, 收玫曲线不会剧烈波动, 而且对学习率不敏感; 第二, TRPO 用更少的经验 （即智能体收集到的状态、动作、奖励）就能达到与策略梯度方法相同的表现。

学习 TRPO 的关键在于理解置信域方法 (trust region methods)。置信域方法不是 TRPO 的论文提出的, 而是数值最优化领域中一类经典的算法，历史至少可以追溯到 1970 年。 TRPO 论文的贡献在于巧妙地把置信域方法应用到强化学习中, 取得非常好的效果。

本节分以下 4 小节讲解 TRPO：第 9.1.1 小节介绍置信域方法, 第 $9.1 .2$ 节回顾策略学 习, 第9.1.3 节推导 TRPO, 第9.1.4讲解 TRPO 的算法流程。

### 1 置信域方法

有这样一个优化问题： $\max _{\boldsymbol{\theta}} J(\boldsymbol{\theta})$ 。这里的 $J(\boldsymbol{\theta})$ 是目标函数, $\boldsymbol{\theta}$ 是优化变量。求解 这个优化问题的目的是找到一个变量 $\boldsymbol{\theta}$ 使得目标函数 $J(\boldsymbol{\theta})$ 取得最大值。有各种各样的 优化算法用于解决这个问题。几乎所有的数值优化算法都是做这样的迭代:

$$
\boldsymbol{\theta}_{\text {new }} \leftarrow \operatorname{Update}\left(\text { Data; } \boldsymbol{\theta}_{\text {now }}\right) .
$$

此处的 $\boldsymbol{\theta}_{\text {now }}$ 和 $\boldsymbol{\theta}_{\text {new }}$ 分别是优化变量当前的值和新的值。不同算法的区别在于具体怎么 样利用数据更新优化变量。

置信域方法用到一个概念一一置信域。下 面介绍置信域。给定变量当前的值 $\boldsymbol{\theta}_{\text {now, }}$ 用 $\mathcal{N}\left(\boldsymbol{\theta}_{\text {now }}\right)$ 表示 $\boldsymbol{\theta}_{\text {now }}$ 的一个邻域。举个例子:

$$
\mathcal{N}\left(\boldsymbol{\theta}_{\text {now }}\right)=\left\{\boldsymbol{\theta} \mid\left\|\boldsymbol{\theta}-\boldsymbol{\theta}_{\text {now }}\right\|_{2} \leq \Delta\right\} .
$$

这个例子中，集合 $\mathcal{N}\left(\boldsymbol{\theta}_{\text {now }}\right)$ 是以 $\boldsymbol{\theta}_{\text {now }}$ 为球心、 以 $\Delta$ 为半径的球; 见右图。球中的点都足够接

![](https://cdn.mathpix.com/cropped/2023_02_03_f46f5cf0e4de5b9996dcg-147.jpg?height=268&width=303&top_left_y=1942&top_left_x=1228)

图 9.1：公式(9.1)中的邻域 $\mathcal{N}\left(\boldsymbol{\theta}_{\text {now }}\right)$ 。 近 $\boldsymbol{\theta}_{\text {now 。 }}$

置信域方法需要构造一个函数 $L\left(\boldsymbol{\theta} \mid \boldsymbol{\theta}_{\text {now }}\right)$, 这个函数要满足这个条件：

$$
L\left(\boldsymbol{\theta} \mid \boldsymbol{\theta}_{\text {now }}\right) \text { 很接近 } J(\boldsymbol{\theta}), \quad \forall \boldsymbol{\theta} \in \mathcal{N}\left(\boldsymbol{\theta}_{\text {now }}\right),
$$

那么集合 $\mathcal{N}\left(\boldsymbol{\theta}_{\text {now }}\right)$ 就被称作置信域。顾名思义, 在 $\boldsymbol{\theta}_{\text {now }}$ 的邻域上, 我们可以信任 $L\left(\boldsymbol{\theta} \mid \boldsymbol{\theta}_{\text {now }}\right)$, 可以拿 $L\left(\boldsymbol{\theta} \mid \boldsymbol{\theta}_{\text {now }}\right)$ 来替代目标函数 $J(\boldsymbol{\theta})$ 。 图 $9.2$ 用一个一元函数的例 子解释 $J(\boldsymbol{\theta})$ 和 $L\left(\boldsymbol{\theta} \mid \boldsymbol{\theta}_{\text {now }}\right)$ 的关 系。图中横轴是优化变量 $\boldsymbol{\theta}$, 纵轴 是函数值。如图 9.2(a) 所示, 函 数 $L\left(\boldsymbol{\theta} \mid \boldsymbol{\theta}_{\text {now }}\right)$ 末必在整个定义域 上都接近 $J(\boldsymbol{\theta})$, 而只是在 $\boldsymbol{\theta}_{\text {now }}$ 的 领域里接近 $J(\boldsymbol{\theta})$ 。 $\boldsymbol{\theta}_{\text {now }}$ 的邻域就 叫做置信域。

通常来说, $J$ 是个很复杂的函 数, 我们甚至可能不知道 $J$ 的解 析表达式（比如 $J$ 是某个函数的 期望)。而我们人为构造出的函数 $L$ 相对较为简单, 比如 $L$ 是 $J$ 的蒙 特卡洛近似, 或者是 $J$ 在 $\boldsymbol{\theta}_{\text {now }}$ 这 个点的二阶泰勒展开。既然可以 信任 $L$, 那么不妨用 $L$ 代替复杂 的函数 $J$, 然后对 $L$ 做最大化。这 样比直接优化 $J$ 要容易得多。这 就是置信域方法的思想。具体来 说, 置信域方法做下面这两个步 骤, 一直重复下去, 当无法让 $J$ 的值增大的时候终止算法。

第一步一一做近似：给定 $\boldsymbol{\theta}_{\text {now }}$, 构造函数 $L\left(\boldsymbol{\theta} \mid \boldsymbol{\theta}_{\text {now }}\right)$, 使得 对于所有的 $\boldsymbol{\theta} \in \mathcal{N}\left(\boldsymbol{\theta}_{\text {now }}\right)$, 函数值 $L\left(\boldsymbol{\theta} \mid \boldsymbol{\theta}_{\text {now }}\right)$ 与 $J(\boldsymbol{\theta})$ 足够接近。图 9.2(b) 解释了做近似这一步。

第二步一一最大化： 在置信 域 $\mathcal{N}\left(\boldsymbol{\theta}_{\text {now }}\right)$ 中寻找变量 $\boldsymbol{\theta}$ 的值, 使得函数 $L$ 的值最大化。把找到的值记作

$$
\boldsymbol{\theta}_{\text {new }}=\underset{\boldsymbol{\theta} \in \mathcal{N}\left(\boldsymbol{\theta}_{\text {now }}\right)}{\operatorname{argmax}} L\left(\boldsymbol{\theta} \mid \boldsymbol{\theta}_{\text {now }}\right) .
$$

图 9.2(c) 解释了最大化这一步。

置信域方法其实是一类算法框架, 而非一个具体的算法。有很多种方式实现实现置 信域方法。第一步需要做近似, 而做近似的方法有多种多样, 比如蒙特卡洛、二阶泰勒 展开。第二步需要解一个带约束的最大化问题; 求解这个问题又需要单独的数值优化算 法, 比如梯度投影算法、拉格朗日法。除此之外, 置信域 $\mathcal{N}\left(\boldsymbol{\theta}_{\text {now }}\right)$ 也有多种多样的选择, 既可以是球, 也可以是两个概率分布的 KL 散度（KL Divergence), 稍后会介绍。

![](https://cdn.mathpix.com/cropped/2023_02_03_f46f5cf0e4de5b9996dcg-148.jpg?height=429&width=764&top_left_y=271&top_left_x=1017)

(a) 构造 $L\left(\boldsymbol{\theta} \mid \boldsymbol{\theta}_{\text {now }}\right)$ 作为 $J(\boldsymbol{\theta})$ 在点 $\boldsymbol{\theta}_{\text {now }}$ 附近的近似。

![](https://cdn.mathpix.com/cropped/2023_02_03_f46f5cf0e4de5b9996dcg-148.jpg?height=437&width=754&top_left_y=827&top_left_x=1008)

(b) $L$ 在点 $\boldsymbol{\theta}_{\text {now }}$ 的邻域内接近 $J$; 这个领域就叫置信域。

![](https://cdn.mathpix.com/cropped/2023_02_03_f46f5cf0e4de5b9996dcg-148.jpg?height=428&width=754&top_left_y=1388&top_left_x=1008)

(c) 在置信域内寻找最大化 $L$ 的解, 记作 $\boldsymbol{\theta}_{\text {new }}$ 。

图 9.2: 一元函数的例子解释置信域和置信域算法。

### 2 策略学习

首先复习策略学习的基础知识。策略网络记作 $\pi(a \mid s ; \boldsymbol{\theta})$, 它是个概率质量函数。动 作价值函数记作 $Q_{\pi}(s, a)$, 它是回报的期望。状态价值函数记作

$$
V_{\pi}(s)=\mathbb{E}_{A \sim \pi(\cdot \mid s ; \boldsymbol{\theta})}\left[Q_{\pi}(s, A)\right]=\sum_{a \in \mathcal{A}} \pi(a \mid s ; \boldsymbol{\theta}) \cdot Q_{\pi}(s, a) .
$$

注意, $V_{\pi}(s)$ 依赖于策略网络 $\pi$, 所以依赖于 $\pi$ 的参数 $\boldsymbol{\theta}$ 。策略学习的目标函数是

$$
J(\boldsymbol{\theta})=\mathbb{E}_{S}\left[V_{\pi}(S)\right] .
$$

$J(\boldsymbol{\theta})$ 只依赖于 $\boldsymbol{\theta}$, 不依赖于状态 $S$ 和动作 $A$ 。第 7 章介绍的策略梯度方法（包括 REINFORCE 和 Actor-Critic) 用蒙特卡洛近似梯度 $\nabla_{\theta} J(\boldsymbol{\theta})$, 得到随机梯度, 然后做随机梯度 上升更新 $\boldsymbol{\theta}$ ，使得目标函数 $J(\boldsymbol{\theta})$ 增大。

下面我们要把目标函数 $J(\boldsymbol{\theta})$ 变换成一种等价形式。从等式(9.2)出发, 把状态价值写 成

$$
\begin{aligned}
V_{\pi}(s) & =\sum_{a \in \mathcal{A}} \pi\left(a \mid s ; \boldsymbol{\theta}_{\text {now }}\right) \cdot \frac{\pi(a \mid s ; \boldsymbol{\theta})}{\pi\left(a \mid s ; \boldsymbol{\theta}_{\text {now }}\right)} \cdot Q_{\pi}(s, a) \\
& =\mathbb{E}_{A \sim \pi\left(\cdot \mid s ; \boldsymbol{\theta}_{\text {now }}\right)}\left[\frac{\pi(A \mid s ; \boldsymbol{\theta})}{\pi\left(A \mid s ; \boldsymbol{\theta}_{\text {now }}\right)} \cdot Q_{\pi}(s, A)\right] .
\end{aligned}
$$

第一个等式很显然, 因为连加中的第一项可以消掉第二项的分母。第二个等式把策略网 络 $\pi\left(A \mid s ; \boldsymbol{\theta}_{\text {now }}\right)$ 看做动作 $A$ 的概率质量函数, 所以可以把连加写成期望。由公式 (9.3) 与 (9.4) 可得定理 9.1。定理 $9.1$ 是 TRPO 的关键所在, 甚至可以说 TRPO 就是从这个公式推 出的。

## 定理 9.1. 目标函数的等价形式

目标函数 $J(\boldsymbol{\theta})$ 可以等价写成：

$$
J(\boldsymbol{\theta})=\mathbb{E}_{S}\left[\mathbb{E}_{A \sim \pi\left(\cdot \mid S ; \boldsymbol{\theta}_{\text {now }}\right)}\left[\frac{\pi(A \mid S ; \boldsymbol{\theta})}{\pi\left(A \mid S ; \boldsymbol{\theta}_{\text {now }}\right)} \cdot Q_{\pi}(S, A)\right]\right] .
$$

上面 $Q_{\pi}$ 中的 $\pi$ 指的是 $\pi(A \mid S ; \boldsymbol{\theta})$ 。

公式中的期望是关于状态 $S$ 和动作 $A$ 求的。状态 $S$ 的概率密度函数只有环境知道, 而我们并不知道, 但是我们可以从环境中获取 $S$ 的观测值。动作 $A$ 的概率质量函数是策 略网络 $\pi\left(A \mid S ; \boldsymbol{\theta}_{\text {now }}\right)$; 注意, 策略网络的参数是旧的值 $\boldsymbol{\theta}_{\text {now }}$.

### TRPO 数学推导

前面介绍了数值优化的基础和价值学习的基础, 终于可以开始推导 TRPO。TRPO 是 置信域方法在策略学习中的应用, 所以 TRPO 也遵循置信域方法的框架, 重复做近似和 最大化这两个步骤, 直到算法收玫。收玫指的是无法增大目标函数 $J(\boldsymbol{\theta})$, 即无法增大期 望回报。

第一步一一做近似：我们从定理 $9.1$ 出发。定理把目标函数 $J(\boldsymbol{\theta})$ 写成了期望的形 式。我们无法直接算出期望, 无法得到 $J(\boldsymbol{\theta})$ 的解析表达式; 原因在于只有环境知道状态 $S$ 的概率密度函数, 而我们不知道。我们可以对期望做蒙特卡洛近似, 从而把函数 $J$ 近 似成函数 $L$ 。用策略网络 $\pi\left(A \mid S ; \boldsymbol{\theta}_{\text {now }}\right)$ 控制智能体跟环境交互, 从头到尾玩完一局游戏, 观测到一条轨迹：

$$
s_{1}, a_{1}, r_{1}, s_{2}, a_{2}, r_{2}, \cdots, s_{n}, a_{n}, r_{n}
$$

其中的状态 $\left\{s_{t}\right\}_{t=1}^{n}$ 都是从环境中观测到的, 其中的动作 $\left\{a_{t}\right\}_{t=1}^{n}$ 都是根据策略网络 $\pi\left(\cdot \mid s_{t} ; \boldsymbol{\theta}_{\text {now }}\right)$ 抽取的样本。所以,

$$
\frac{\pi\left(a_{t} \mid s_{t} ; \boldsymbol{\theta}\right)}{\pi\left(a_{t} \mid s_{t} ; \boldsymbol{\theta}_{\text {now }}\right)} \cdot Q_{\pi}\left(s_{t}, a_{t}\right)
$$

是对定理 $9.1$ 中期望的无偏估计。我们观测到了 $n$ 组状态和动作, 于是应该对公式 (9.5) 求平均, 把得到均值记作：

$$
L\left(\boldsymbol{\theta} \mid \boldsymbol{\theta}_{\text {now }}\right)=\frac{1}{n} \sum_{t=1}^{n} \underbrace{\frac{\pi\left(a_{t} \mid s_{t} ; \boldsymbol{\theta}\right)}{\pi\left(a_{t} \mid s_{t} ; \boldsymbol{\theta}_{\text {now }}\right)} \cdot Q_{\pi}\left(s_{t}, a_{t}\right)}_{\text {定理 } 9.1 \text { 中期望的无偏估计 }} .
$$

既然连加里每一项都是期望的无偏估计, 那么 $n$ 项的均值 $L$ 也是无偏估计。所以可以拿 $L$ 作为目标函数 $J$ 的蒙特卡洛近似。

公式 (9.6) 中的 $L\left(\boldsymbol{\theta} \mid \boldsymbol{\theta}_{\text {now }}\right)$ 是对目标函数 $J(\boldsymbol{\theta})$ 的近似。可惜我们还无法直接对 $L$ 求 最大化, 原因是我们不知道动作价值 $Q_{\pi}\left(s_{t}, a_{t}\right)$ 。解决方法是做两次近似:

$$
Q_{\pi}\left(s_{t}, a_{t}\right) \Longrightarrow Q_{\pi_{\text {old }}}\left(s_{t}, a_{t}\right) \Longrightarrow u_{t} .
$$

公式中 $Q_{\pi}$ 中的策略是 $\pi\left(a_{t} \mid s_{t} ; \boldsymbol{\theta}\right)$, 而 $Q_{\pi_{\text {old }}}$ 中的策略则是旧策略 $\pi\left(a_{t} \mid s_{t} ; \boldsymbol{\theta}_{\text {now }}\right)$ 。我们 用旧策略 $\pi\left(a_{t} \mid s_{t} ; \boldsymbol{\theta}_{\text {now }}\right)$ 生成轨迹 $\left\{\left(s_{j}, a_{j}, r_{j}, s_{j+1}\right)\right\}_{j=1}^{n}$, 所以折扣回报

$$
u_{t}=r_{t}+\gamma \cdot r_{t+1}+\gamma^{2} \cdot r_{t+2}+\cdots+\gamma^{n-t} \cdot r_{n}
$$

是对 $Q_{\pi_{\text {old }}}$ 的近似, 而末必是对 $Q_{\pi}$ 的近似。仅当 $\boldsymbol{\theta}$ 接近 $\boldsymbol{\theta}_{\text {now }}$ 的时候, $u_{t}$ 才是 $Q_{\pi}$ 的有 效近似。这就是为什么要强调置信域, 即 $\boldsymbol{\theta}$ 在 $\boldsymbol{\theta}_{\text {now }}$ 的邻域中。

$$
\begin{aligned}
& \text { 拿 } u_{t} \text { 替代 } Q_{\pi}\left(s_{t}, a_{t}\right) \text {, 那么公式 (9.6) 中的 } L\left(\boldsymbol{\theta} \mid \boldsymbol{\theta}_{\text {now }}\right) \text { 变成了 } \\
& \qquad \tilde{L}\left(\boldsymbol{\theta} \mid \boldsymbol{\theta}_{\text {now }}\right)=\frac{1}{n} \sum_{t=1}^{n} \frac{\pi\left(a_{t} \mid s_{t} ; \boldsymbol{\theta}\right)}{\pi\left(a_{t} \mid s_{t} ; \boldsymbol{\theta}_{\text {now }}\right)} \cdot u_{t} .
\end{aligned}
$$

总结一下, 我们把目标函数 $J$ 近似成 $L$, 然后又把 $L$ 近似成 $\tilde{L}$ 。在第二步近似中, 我们 需要假设 $\boldsymbol{\theta}$ 接近 $\boldsymbol{\theta}_{\text {now }}$ 。

第二步一一最大化： TRPO 把公式 (9.7) 中的 $\tilde{L}\left(\boldsymbol{\theta} \mid \boldsymbol{\theta}_{\text {now }}\right)$ 作为对目标函数 $J(\boldsymbol{\theta})$ 的近 似, 然后求解这个带约束的最大化问题:

$$
\max _{\boldsymbol{\theta}} \tilde{L}\left(\boldsymbol{\theta} \mid \boldsymbol{\theta}_{\text {now }}\right) ; \quad \text { s.t. } \boldsymbol{\theta} \in \mathcal{N}\left(\boldsymbol{\theta}_{\text {now }}\right) .
$$

公式中的 $\mathcal{N}\left(\boldsymbol{\theta}_{\text {now }}\right)$ 是置信域, 即 $\theta_{\text {now }}$ 的一个邻域。该用什么样的置信域呢?

- 一种方法是用以 $\theta_{\text {now }}$ 为球心、以 $\Delta$ 为半径的球作为置信域。这样的话, 公式(9.8)就 变成

$$
\max _{\boldsymbol{\theta}} \tilde{L}\left(\boldsymbol{\theta} \mid \boldsymbol{\theta}_{\text {now }}\right) ; \quad \text { s.t. }\left\|\boldsymbol{\theta}-\boldsymbol{\theta}_{\text {now }}\right\|_{2} \leq \Delta
$$

- 另一种方法是用 $\mathrm{KL}$ 散度衡量两个概率质量函数 $-\pi\left(\cdot \mid s_{i} ; \boldsymbol{\theta}_{\text {now }}\right)$ 和 $\pi\left(\cdot \mid s_{i} ; \boldsymbol{\theta}\right)-$ 的距离。两个概率质量函数区别越大, 它们的 KL 散度就越大。反之, 如果 $\boldsymbol{\theta}$ 很接 近 $\boldsymbol{\theta}_{\text {now, }}$ 那么两个概率质量函数就越接近。用 $\mathrm{KL}$ 散度的话, 公式(9.8)就变成

$$
\max _{\boldsymbol{\theta}} \tilde{L}\left(\boldsymbol{\theta} \mid \boldsymbol{\theta}_{\text {now }}\right) ; \quad \text { s.t. } \frac{1}{t} \sum_{i=1}^{t} \mathrm{KL}\left[\pi\left(\cdot \mid s_{i} ; \boldsymbol{\theta}_{\text {now }}\right) \| \pi\left(\cdot \mid s_{i} ; \boldsymbol{\theta}\right)\right] \leq \Delta .
$$

用球作为置信域的好处是置信域是简单的形状, 求解最大化问题比较容易, 但是用球做 置信域的实际效果不如用 $\mathrm{KL}$ 散度。

TRPO 的第二步―一最大化一一需要求解带约束的最大化问题 (9.9) 或者 (9.10)。注 意, 这种问题的求解并不容易; 简单的梯度上升算法并不能解带约束的最大化问题。数 值优化教材通常有介绍带约束问题的求解, 有兴趣的话自己去阅读数值优化教材, 这里 就不详细解释如何求解问题 (9.9) 或者 (9.10)。读者可以这样看待优化问题：只要你能把 一个优化问题的目标函数和约束条件解析地写出来, 通常会有数值算法能解决这个问题。

### 4 训练流程

在本节的最后, 我们总结一下用 TRPO 训练策略网络的流程。TRPO 需要重复做近 似和最大化这两个步骤：

1. 做近似一一构造函数 $\tilde{L}$ 近似目标函数 $J(\boldsymbol{\theta})$ :

![](https://cdn.mathpix.com/cropped/2023_02_03_f46f5cf0e4de5b9996dcg-151.jpg?height=63&width=1353&top_left_y=1342&top_left_x=343)
互, 玩完一局游戏, 记录下轨迹:

$$
s_{1}, a_{1}, r_{1}, s_{2}, a_{2}, r_{2}, \cdots, s_{n}, a_{n}, r_{n} .
$$

(b). 对于所有的 $t=1, \cdots, n$, 计算折扣回报 $u_{t}=\sum_{k=t}^{n} \gamma^{k-t} \cdot r_{k}$ 。

(c). 得出近似函数:

$$
\tilde{L}\left(\boldsymbol{\theta} \mid \boldsymbol{\theta}_{\text {now }}\right)=\frac{1}{n} \sum_{t=1}^{n} \frac{\pi\left(a_{t} \mid s_{t} ; \boldsymbol{\theta}\right)}{\pi\left(a_{t} \mid s_{t} ; \boldsymbol{\theta}_{\text {now }}\right)} \cdot u_{t} .
$$

2. 最大化一一用某种数值算法求解带约束的最大化问题:

$$
\boldsymbol{\theta}_{\text {new }}=\underset{\boldsymbol{\theta}}{\operatorname{argmax}} \tilde{L}\left(\boldsymbol{\theta} \mid \boldsymbol{\theta}_{\text {now }}\right) ; \quad \text { s.t. }\left\|\boldsymbol{\theta}-\boldsymbol{\theta}_{\text {now }}\right\|_{2} \leq \Delta .
$$

此处的约束条件是二范数距离。可以把它替换成 KL 散度, 即公式 (9.10)。

TRPO 中有两个需要调的超参数：一个是置信域的半径 $\Delta$, 另一个是求解最大化问 题的数值算法的学习率。通常来说, $\Delta$ 在算法的运行过程中要逐渐缩小。虽然 TRPO 需 要调参, 但是 TRPO 对超参数的设置并不敏感。即使超参数设置不够好, TRPO 的表现 也不会太差。相比之下, 策略梯度算法对超参数更敏感。

TRPO 算法真正实现起来并不容易, 主要难点在于第二步一一最大化。不建议读者 自己去实现 TRPO。

## $9.2$ 熵正则 (Entropy Regularization)

策略学习的目的是学出一个策略网络 $\pi(a \mid s ; \boldsymbol{\theta})$ 用于控制智能体。每当智能体观测到 当前状态 $s$, 策略网络输出一个概率分布, 智能体依据概率分布抽样一个动作, 并执行这 个动作。举个例子, 在超级玛丽游戏中, 动作空间是 $\mathcal{A}=\{$ 左, 右, 上 $\}$ 。基于当前状态 $s$, 策略网络的输出是

$$
\begin{aligned}
& p_{1}=\pi(\text { 左 } \mid s ; \boldsymbol{\theta})=0.03, \\
& p_{2}=\pi(\text { 右 } \mid s ; \boldsymbol{\theta})=0.96, \\
& p_{3}=\pi(\text { 上 } \mid s ; \boldsymbol{\theta})=0.01 .
\end{aligned}
$$

那么超级玛丽做的动作可能是左、右、上三者中的任何一个, 概率分别是 $0.03,0.96,0.01$ 。 概率都集中在“向右”的动作上, 接近确定性的决策。确定性大的好处在于不容易选中很 差的动作, 比较安全。但是确定性大也有缺点。假如策略网络的输出总是这样确定性很 大的概率分布, 那么智能体就会安于现状, 不去尝试没做过的动作, 不去探索更多的状 态, 无法找到更好的策略。

我们希望策略网络的输出的概率不要集中在一个动作上, 至少要给其他的动作一些 非零的概率, 让这些动作能被探索到。可以用熵 (Entropy) 来衡量概率分布的不确定性。 对于上述离散概率分布 $\boldsymbol{p}=\left[p_{1}, p_{2}, p_{3}\right]$, 熵等于

$$
\operatorname{Entropy}(\boldsymbol{p})=-\sum_{i=1}^{3} p_{i} \cdot \ln p_{i} \text {. }
$$

熵小说明概率质量很集中, 熵大说明随机性很大；见图 $9.3$ 的解释。

![](https://cdn.mathpix.com/cropped/2023_02_03_f46f5cf0e4de5b9996dcg-152.jpg?height=369&width=531&top_left_y=1666&top_left_x=457)

熵大, 好

![](https://cdn.mathpix.com/cropped/2023_02_03_f46f5cf0e4de5b9996dcg-152.jpg?height=366&width=512&top_left_y=1670&top_left_x=1206)

熵小, 不好

图 9.3: 两张图中分别描述两个离散概率分布。左边的概率比较均匀, 这种情况熵很大。右边的概 率集中在 $p_{2}$ 上，这种情况的熵较小。

策略学习中的熵正则：我们希望策略网络输出的概率分布的熵不要太小。我们不妨 把熵作为正则项, 放到策略学习的目标函数中。策略网络的输出是维度等于 $|\mathcal{A}|$ 的向量, 它表示定义在动作空间上的离散概率分布。这个概率分布的熵定义为：

$$
H(s ; \boldsymbol{\theta}) \triangleq \text { Entropy }[\pi(\cdot \mid s ; \boldsymbol{\theta})]=-\sum_{a \in \mathcal{A}} \pi(a \mid s ; \boldsymbol{\theta}) \cdot \ln \pi(a \mid s ; \boldsymbol{\theta}) .
$$

熵 $H(s ; \boldsymbol{\theta})$ 只依赖于状态 $s$ 与策略网络参数 $\boldsymbol{\theta}$ 。我们希望对于大多数的状态 $s$, 熵都会比 较大, 也就是让 $\mathbb{E}_{S}[H(S ; \boldsymbol{\theta})]$ 比较大。

回忆一下, $V_{\pi}(s)$ 是状态价值函数, 衡量在状态 $s$ 的情况下, 策略网络 $\pi$ 表现的好 坏程度。策略学习的目标函数是 $J(\boldsymbol{\theta})=\mathbb{E}_{S}\left[V_{\pi}(S)\right]$ 。策略学习的目的是寻找参数 $\boldsymbol{\theta}$ 使得 $J(\boldsymbol{\theta})$ 最大化。同时, 我们还希望让熵比较大, 所以把熵作为正则项, 放到目标函数里。使 用熵正则的策略学习可以写作这样的最大化问题:

$$
\max _{\theta} J(\boldsymbol{\theta})+\lambda \cdot \mathbb{E}_{S}[H(S ; \boldsymbol{\theta})] .
$$

此处的 $\lambda$ 是个超参数, 需要手动调。

优化： 带熵正则的最大化问题 (9.12) 可以用各种方法求解, 比如策略梯度方法（包 括 REINFORCE 和 Actor-Critic)、TRPO 等。此处只讲解策略梯度方法。公式 (9.12) 中目 标函数关于 $\boldsymbol{\theta}$ 的梯度是:

$$
\boldsymbol{g}(\boldsymbol{\theta}) \triangleq \nabla_{\boldsymbol{\theta}}\left[J(\boldsymbol{\theta})+\lambda \cdot \mathbb{E}_{S}[H(S ; \boldsymbol{\theta})]\right] .
$$

观测到状态 $s$, 按照策略网络做随机抽样, 得到动作 $a \sim \pi(\cdot \mid s ; \boldsymbol{\theta})$ 。那么

$$
\tilde{\boldsymbol{g}}(s, a ; \boldsymbol{\theta}) \triangleq\left[Q_{\pi}(s, a)-\lambda \cdot \ln \pi(a \mid s ; \boldsymbol{\theta})-\lambda\right] \cdot \nabla_{\boldsymbol{\theta}} \ln \pi(a \mid s ; \boldsymbol{\theta})
$$

是梯度 $\boldsymbol{g}(\boldsymbol{\theta})$ 的无偏估计（见定理 9.2)。因此可以用 $\tilde{\boldsymbol{g}}(s, a ; \boldsymbol{\theta})$ 更新策略网络的参数：

$$
\boldsymbol{\theta} \leftarrow \boldsymbol{\theta}+\beta \cdot \tilde{\boldsymbol{g}}(s, a ; \boldsymbol{\theta}) .
$$

此处的 $\beta$ 是学习率。

## 定理 9.2. 带熵正则的策略梯度

$$
\nabla_{\boldsymbol{\theta}}\left[J(\boldsymbol{\theta})+\lambda \cdot \mathbb{E}_{S}[H(S ; \boldsymbol{\theta})]\right]=\mathbb{E}_{S}\left[\mathbb{E}_{A \sim \pi(\cdot \mid s ; \theta)}[\tilde{\boldsymbol{g}}(S, A ; \boldsymbol{\theta})]\right] .
$$

证明 首先推导熵 $H(S ; \boldsymbol{\theta})$ 关于 $\boldsymbol{\theta}$ 的梯度。由公式 (9.11) 中 $H(S ; \boldsymbol{\theta})$ 的定义可得

$$
\begin{aligned}
\frac{\partial H(s ; \boldsymbol{\theta})}{\partial \boldsymbol{\theta}} & =-\sum_{a \in \mathcal{A}} \frac{\partial[\pi(a \mid s ; \boldsymbol{\theta}) \cdot \ln \pi(a \mid s ; \boldsymbol{\theta})]}{\partial \boldsymbol{\theta}} \\
& =-\sum_{a \in \mathcal{A}}\left[\ln \pi(a \mid s ; \boldsymbol{\theta}) \cdot \frac{\partial \pi(a \mid s ; \boldsymbol{\theta})}{\partial \boldsymbol{\theta}}+\pi(a \mid s ; \boldsymbol{\theta}) \cdot \frac{\partial \ln \pi(a \mid s ; \boldsymbol{\theta})}{\partial \boldsymbol{\theta}}\right] .
\end{aligned}
$$

第二个等式由链式法则得到。由于 $\frac{\partial \pi(a \mid s ; \boldsymbol{\theta})}{\partial \boldsymbol{\theta}}=\pi(a \mid s ; \boldsymbol{\theta}) \cdot \frac{\partial \ln \pi(a \mid s ; \boldsymbol{\theta})}{\partial \boldsymbol{\theta}}$, 上面的公式可以写 成:

$$
\begin{aligned}
\frac{\partial H(s ; \boldsymbol{\theta})}{\partial \boldsymbol{\theta}} & =-\sum_{a \in \mathcal{A}}\left[\ln \pi(a \mid s ; \boldsymbol{\theta}) \cdot \pi(a \mid s ; \boldsymbol{\theta}) \cdot \frac{\partial \ln \pi(a \mid s ; \boldsymbol{\theta})}{\partial \boldsymbol{\theta}}+\pi(a \mid s ; \boldsymbol{\theta}) \cdot \frac{\partial \ln \pi(a \mid s ; \boldsymbol{\theta})}{\partial \boldsymbol{\theta}}\right] \\
& =-\sum_{a \in \mathcal{A}} \pi(a \mid s ; \boldsymbol{\theta}) \cdot[\ln \pi(a \mid s ; \boldsymbol{\theta})+1] \cdot \frac{\partial \ln \pi(a \mid s ; \boldsymbol{\theta})}{\partial \boldsymbol{\theta}} \\
& =-\mathbb{E}_{A \sim \pi(\cdot \mid s ; \boldsymbol{\theta})}\left[[\ln \pi(A \mid s ; \boldsymbol{\theta})+1] \cdot \frac{\partial \ln \pi(A \mid s ; \boldsymbol{\theta})}{\partial \boldsymbol{\theta}}\right] .
\end{aligned}
$$

应用第 7 章推导的策略梯度定理, 可以把 $J(\boldsymbol{\theta})$ 关于 $\boldsymbol{\theta}$ 的梯度写作

$$
\frac{\partial J(\boldsymbol{\theta})}{\partial \boldsymbol{\theta}}=\mathbb{E}_{S}\left\{\mathbb{E}_{A \sim \pi(\cdot \mid S ; \boldsymbol{\theta})}\left[Q_{\pi}(S, A) \cdot \frac{\partial \ln \pi(A \mid S ; \boldsymbol{\theta})}{\partial \boldsymbol{\theta}}\right]\right\} .
$$

由公式 (9.13) 与 (9.14) 可得:

$$
\begin{aligned}
& \frac{\partial}{\partial \boldsymbol{\theta}}\left[J(\boldsymbol{\theta})+\lambda \cdot \mathbb{E}_{S}[H(S ; \boldsymbol{\theta})]\right] \\
& =\mathbb{E}_{S}\left\{\mathbb{E}_{A \sim \pi(\cdot \mid S ; \boldsymbol{\theta})}\left[\left(Q_{\pi}(S, A)-\lambda \cdot \ln \pi(A \mid S ; \boldsymbol{\theta})-\lambda\right) \cdot \frac{\partial \ln \pi(A \mid S ; \boldsymbol{\theta})}{\partial \boldsymbol{\theta}}\right]\right\} \\
& =\mathbb{E}_{S}\left\{\mathbb{E}_{A \sim \pi(\cdot \mid S ; \boldsymbol{\theta})}[\tilde{\boldsymbol{g}}(S, A ; \boldsymbol{\theta})]\right\} .
\end{aligned}
$$

上面第二个等式由 $\tilde{\boldsymbol{g}}$ 的定义得到。

## 第 9 章 知识点

- 置信域方法指的是一大类数值优化算法, 通常用于求解非凸问题。对于一个最大化 问题, 算法重复两个步骤一一做近似、最大化一一直到算法收敛。

- 置信域策略优化（TRPO）是一种置信域算法, 它的目标是最大化目标函数 $J(\boldsymbol{\theta})=$ $\mathbb{E}_{S}\left[V_{\pi}(S)\right]$ 。与策略梯度算法相比, TRPO 的优势在于更好的稳定性、用更少的样本 达到收敛。

- 策略学习中常用熵正则这种技巧, 即鼓励策略网络输出的概率分布有较大的熵。熵 越大, 概率分布越均匀; 熵越小, 概率质量越集中在少数动作上。

## 第 9 章 相关文献

TRPO 由 Schulman 等人在 2015 年提出 ${ }^{[94]}$ 。TRPO 是置信域方法在强化学习中的成 功应用。置信域是经典的数值优化算法, 对此感兴趣的读者可以阅读这些教材: ${ }^{[82,30]}$ 。 TRPO 每一轮循环都需要求解带约束的最大化问题; 这类问题的求解可以参考这些教材: $[12,18]$ 。

熵正则是策略学习中常见的方法, 在很多论文中有使用, 比如 $[127,75,83,3,44,96]$ 。虽 然熵正则能鼓励探索, 但是增大决策的不确定性是有风险的：很差的动作可能也有非零 的概率。一个好的办法是用 Tsallis Entropy ${ }^{[111]}$ 做正则, 让离散概率具有稀疏性, 每次决 策只给少部分动作非零的概率, “过滤掉”很差的动作。有兴趣的读者可以阅读这些论文:

$[29,63,128]$
