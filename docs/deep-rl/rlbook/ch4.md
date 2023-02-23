
## 第 4 章 $D Q N$ 与 $Q$ 学习

本章的内容是价值学习的基础。第 $4.1$ 节用神经网络近似最优动作价值函数 $Q^{\star}(s, a)$, 把这个神经网络称为深度 $\mathrm{Q}$ 网络 (DQN)。本章的难点在于训练 $\mathrm{DQN}$ 所用的时间差分算 法 (TD)。第 $4.2$ 节以“驾车时间估计”类比 DQN, 讲解 TD 算法。第 $4.3$ 节推导训练 DQN 用的 $\mathrm{Q}$ 学习算法; 它是一种 TD 算法。第 $4.4$ 节介绍表格形式的 $\mathrm{Q}$ 学习算法。第 $4.5$ 节 解释同策略（on-policy）与异策略（off-policy）的区别。本章介绍的 $\mathrm{Q}$ 学习算法属于异 策略。

### DQN

在学习 DQN 之前, 首先复习一些基础知识。在一局游戏中, 把从起始到结束的所有 奖励记作：

$$
R_{1}, \cdots, R_{t}, \cdots, R_{n} .
$$

定义折扣率 $\gamma \in[0,1]$ 。折扣回报的定义是：

$$
U_{t}=R_{t}+\gamma \cdot R_{t+1}+\gamma^{2} \cdot R_{t+2}+\cdots+\gamma^{n-t} \cdot R_{n} .
$$

在游戏尚末结束的 $t$ 时刻, $U_{t}$ 是一个末知的随机变量, 其随机性来自于 $t$ 时刻之后的所 有状态与动作。动作价值函数的定义是:

$$
Q_{\pi}\left(s_{t}, a_{t}\right)=\mathbb{E}\left[U_{t} \mid S_{t}=s_{t}, A_{t}=a_{t}\right],
$$

公式中的期望消除了 $t$ 时刻之后的所有状态 $S_{t+1}, \cdots, S_{n}$ 与所有动作 $A_{t+1}, \cdots, A_{n}$ 。最 优动作价值函数用最大化消除策略 $\pi$ :

$$
Q_{\star}\left(s_{t}, a_{t}\right)=\max _{\pi} Q_{\pi}\left(s_{t}, a_{t}\right), \quad \forall s_{t} \in \mathcal{S}, \quad a_{t} \in \mathcal{A} .
$$

可以这样理解 $Q_{\star}$ : 已知 $s_{t}$ 和 $a_{t}$, 不论末来采取什么样的策略 $\pi$, 回报 $U_{t}$ 的期望不可能 超过 $Q_{\star}$ 。

最优动作价值函数的用途 : 假如我们知道 $Q_{\star}$, 我们就能用它做控制。举个例子, 超 级玛丽游戏中的动作空间是 $\mathcal{A}=\{$ 左, 右, 上 $\}$ 。给定当前状态 $s_{t}$, 智能体该执行哪个动作 呢? 假设我们已知 $Q_{\star}$ 函数, 那么我们就让 $Q_{\star}$ 给三个动作打分, 比如：

$$
Q_{\star}\left(s_{t}, \text { 左 }\right)=370, \quad Q_{\star}\left(s_{t}, \text { 右 }\right)=-21, \quad Q_{\star}\left(s_{t}, \text { 上 }\right)=610 \text {. }
$$

这三个值是什么意思呢? $Q_{\star}\left(s_{t}\right.$, 左 $)=370$ 的意思是：如果现在智能体选择向左走, 不论 之后采取什么策略 $\pi$, 那么回报 $U_{t}$ 的期望最多不会超过 370 。同理, 其他两个最优动作 价值也是回报的期望的上界。根据 $Q_{\star}$ 的评分, 智能体应该选择向上跳, 因为这样可以最 大化回报 $U_{t}$ 的期望。

我们希望知道 $Q_{\star}$, 因为它就像是先知一般, 可以预见末来, 在 $t$ 时刻就预见 $t$ 到 $n$ 时刻之间的累计奖励的期望。假如我们有 $Q_{\star}$ 这位先知, 我们就遵照先知的指导, 最大化 末来的累计奖励。然而在实践中我们不知道 $Q_{\star}$ 的函数表达式。是否有可能近似出 $Q_{\star}$ 这 位先知呢? 对于超级玛丽这样的游戏, 学出来一个 “先知”并不难。假如让我们重复玩超 级玛丽一亿次, 那我们就会像先知一样, 看到当前状态, 就能准确判断出当前最优的动 作是什么。这说明只要有足够多的“经验”, 就能训练出超级玛丽中的“先知”。

最优动作价值函数的近似：在实践中, 近似学习“先知” $Q_{\star}$ 最有效的办法是深度 $\mathrm{Q}$ 网络 (deep Q network, 缩写 DQN), 记作 $Q(s, a ; \boldsymbol{w})$, 其结构如图 $4.1$ 所述。其中的 $\boldsymbol{w}$ 表 示神经网络中的参数。首先随机初始化 $\boldsymbol{w}$, 随后用“经验” 去学习 $\boldsymbol{w}$ 。学习的目标是：对 于所有的 $s$ 和 $a, \mathrm{DQN}$ 的预测 $Q(s, a ; \boldsymbol{w})$ 尽量接近 $Q_{\star}(s, a)$ 。后面几节的内容都是如何学 习 $\boldsymbol{w}$ 。

![](https://cdn.mathpix.com/cropped/2023_02_03_f46f5cf0e4de5b9996dcg-060.jpg?height=397&width=1431&top_left_y=787&top_left_x=378)

图 4.1: DQN 的神经网络结构。输入是状态 $s$; 输出是每个动作的 $\mathrm{Q}$ 值。

可以这样理解 $\mathrm{DQN}$ 的表达式 $Q(s, a ; \boldsymbol{w})$ 。DQN 的输出是离散动作空间 $\mathcal{A}$ 上的每个 动作的 $\mathrm{Q}$ 值, 即给每个动作的评分, 分数越高意味着动作越好。举个例子, 动作空间是 $\mathcal{A}=\{$ 左, 右, 上 $\}$, 那么动作空间的大小等于 $|\mathcal{A}|=3$, 那么 $\mathrm{DQN}$ 的输出是 3 维的向量, 记作 $\widehat{\boldsymbol{q}}$, 向量每个元素对应一个动作。在图 $4.1$ 中, DQN 的输出是

$$
\begin{aligned}
& \widehat{q}_{1}=Q(s, \text { 左; } \boldsymbol{w})=370, \\
& \widehat{q}_{2}=Q(s, \text { 右; } \boldsymbol{w})=-21, \\
& \widehat{q}_{3}=Q(s, \text { 上 } ; \boldsymbol{w})=610 .
\end{aligned}
$$

总结一下, DQN 的输出是 $|\mathcal{A}|$ 维的向量 $\widehat{\boldsymbol{q}}$, 包含所有动作的价值。而我们常用的符号 $Q(s, a ; \boldsymbol{w})$ 是标量, 是动作 $a$ 对应的动作价值, 是向量 $\widehat{\boldsymbol{q}}$ 中的一个元素。

$D Q N$ 的梯度 : 在训练 $D Q N$ 的时候, 需要对 $D Q N$ 关于神经网络参数 $\boldsymbol{w}$ 求梯度。用

$$
\nabla_{\boldsymbol{w}} Q(s, a ; \boldsymbol{w}) \triangleq \frac{\partial Q(s, a ; \boldsymbol{w})}{\partial \boldsymbol{w}}
$$

表示函数值 $Q(s, a ; \boldsymbol{w})$ 关于参数 $\boldsymbol{w}$ 的梯度。因为函数值 $Q(s, a ; \boldsymbol{w})$ 是一个实数, 所以梯 度的形状与 $\boldsymbol{w}$ 完全相同。如果 $\boldsymbol{w}$ 是 $d \times 1$ 的向量, 那么梯度也是 $d \times 1$ 的向量。如果 $\boldsymbol{w}$ 是 $d_{1} \times d_{2}$ 的矩阵, 那么梯度也是 $d_{1} \times d_{2}$ 的矩阵。如果 $\boldsymbol{w}$ 是 $d_{1} \times d_{2} \times d_{3}$ 的张量, 那么 梯度也是 $d_{1} \times d_{2} \times d_{3}$ 的张量。

给定观测值 $s$ 和 $a$, 比如 $a=$ “左”, 可以用反向传播计算出梯度 $\nabla_{\boldsymbol{w}} Q(s$, 左; $\boldsymbol{w})$ 。在编 程实现的时候, TensorFlow 和 PyTorch 可以对 DQN 输出向量的一个元素 (比如 $Q(s$, 左; $\boldsymbol{w}$ ) 这个元素）关于变量 $\boldsymbol{w}$ 自动求梯度, 得到的梯度的形状与 $\boldsymbol{w}$ 完全相同。

## $4.2$ 时间差分 (TD) 算法

训练 DQN 最常用的算法是时间差分 (temporal difference, 缩写 TD)。TD 算法不太 好理解, 所以本节举一个通俗易懂的例子讲解 TD 算法。

### 1 驾车时间预测的例子

假设我们有一个模型 $Q(s, d ; \boldsymbol{w})$, 其中 $s$ 是起点, $d$ 是终点, $\boldsymbol{w}$ 是参数。模型 $Q$ 可以 预测开车出行的时间开销。这个模型一开始不准确, 甚至是纯随机的。但是随着很多人 用这个模型, 得到更多数据、更多训练, 这个模型就会越来越准, 会像谷歌地图一样准。

我们该如何训练这个模型呢? 在用户出发前, 用户告诉模型起点 $s$ 和终点 $d$, 模型 做一个预测 $\widehat{q}=Q(s, d ; \boldsymbol{w})$ 。当用户结束行程的时候, 把实际驾车时间 $y$ 反馈给模型。两 者之差 $\widehat{q}-y$ 反映出模型是高估还是低估了驾驶时间, 以此来修正模型, 使得模型的估 计更准确。

假设我是个用户, 我要从北京驾车去上海。从北京出发之前, 我让模型做预测, 模 型告诉我总车程是 14 小时:

$$
\widehat{q} \triangleq Q(\text { “北京”, “上海” } ; \boldsymbol{w})=14 .
$$

当我到达上海, 我知道自己花的实际时间是 16 小时, 并将结果反馈给模型; 见图 4.2。

![](https://cdn.mathpix.com/cropped/2023_02_03_f46f5cf0e4de5b9996dcg-061.jpg?height=208&width=1322&top_left_y=1478&top_left_x=298)

图 4.2: 模型估计驾驶时间是 $\widehat{q}=14$, 而实际花费时间 $y=16$ 。

可以用梯度下降对模型做一次更新, 具体做法如下。把我的这次旅程作为一组训练 数据：

$$
s=\text { “北京” }, \quad d=\text { “上海”, } \quad \widehat{q}=14, \quad y=16 \text {. }
$$

我们希望估计值 $\widehat{q}=Q(s, d ; \boldsymbol{w})$ 尽量接近真实观测到的 $y$, 所以用两者差的平方作为损失 函数：

$$
L(\boldsymbol{w})=\frac{1}{2}[Q(s, d ; \boldsymbol{w})-y]^{2} .
$$

用链式法则计算损失函数的梯度，得到：

$$
\nabla_{\boldsymbol{w}} L(\boldsymbol{w})=(\widehat{q}-y) \cdot \nabla_{\boldsymbol{w}} Q(s, d ; \boldsymbol{w}),
$$

然后做一次梯度下降更新模型参数 $\boldsymbol{w}$ :

$$
\boldsymbol{w} \leftarrow \boldsymbol{w}-\alpha \cdot \nabla_{\boldsymbol{w}} L(\boldsymbol{w}),
$$

此处的 $\alpha$ 是学习率, 需要手动调整。在完成一次梯度下降之后, 如果再让模型做一次预 测, 那么模型的预测值

$$
Q \text { (“北京”, “上海”; } \boldsymbol{w})
$$

会比原先更接近 $y=16$.

### TD 算法}

接着上文驾车时间的例子。出发前模型估计全程时间为 $\widehat{q}=14$ 小时; 模型建议的路 线会途径济南。我从北京出发, 过了 $r=4.5$ 小时, 我到达济南。此时我再让模型做一次 预测, 模型告诉我

$$
\left.\widehat{q}^{\prime} \triangleq Q \text { (“济南”, “上海”; } \boldsymbol{w}\right)=11 .
$$

见图 $4.3$ 的描述。假如此时我的车坏了, 必须要在济南修理, 我不得不取消此次行程。我 没有完成旅途, 那么我的这组数据是否能帮助训练模型呢? 其实是可以的, 用到的算法 叫做时间差分 (temporal difference, 缩写 TD)。

预计需要 $\hat{q}=14$ 小时

![](https://cdn.mathpix.com/cropped/2023_02_03_f46f5cf0e4de5b9996dcg-062.jpg?height=177&width=1308&top_left_y=1431&top_left_x=434)

图 4.3: $\widehat{q}=14$ 和 $\widehat{q}^{\prime}=11$ 是模型的估计值; $r=4.5$ 是实际观测值。

下面解释 TD 算法的原理。回顾一下我们已有的数据：模型估计从北京到上海一共 需要 $\widehat{q}=14$ 小时, 我实际用了 $r=4.5$ 小时到达济南, 模型估计还需要 $\widehat{q}=11$ 小时从济 南到上海。到达济南时, 根据模型最新估计, 整个旅程的总时间为:

$$
\widehat{y} \triangleq r+\widehat{q}^{\prime}=4.5+11=15.5 \text {. }
$$

TD 算法将 $\widehat{y}=15.5$ 称为 TD 目标（TD target）, 它比最初的预测 $\widehat{q}=14$ 更可靠。最初 的预测 $\widehat{q}=14$ 纯粹是估计的, 没有任何事实的成分。 $\mathrm{TD}$ 目标 $\widehat{y}=15.5$ 也是个估计, 但 其中有事实的成分：其中的 $r=4.5$ 就是实际的观测。

基于以上讨论, 我们认为 $\mathrm{TD}$ 目标 $\widehat{y}=15.5$ 比模型最初的估计值

$$
\widehat{q}=Q \text { (“北京”, “上海”; } \boldsymbol{w})=14
$$

更可靠, 所以可以用 $\widehat{y}$ 对模型做“修正”。我们希望估计值 $\widehat{q}$ 尽量接近 TD 目标 $\widehat{y}$, 所以用 两者差的平方作为损失函数：

$$
L(\boldsymbol{w})=\frac{1}{2}[Q(\text { “北京” “上海” } ; \boldsymbol{w})-\widehat{y}]^{2} .
$$

此处把 $\widehat{y}$ 看做常数, 尽管它依赖于 $\boldsymbol{w}$ 。 ${ }^{1}$ 计算损失函数的梯度:

$$
\nabla_{\boldsymbol{w}} L(\boldsymbol{w})=\underbrace{(\widehat{q}-\widehat{y})}_{\text {记作 } \delta} \cdot \nabla_{\boldsymbol{w}} Q \text { (“北京”,“上海” } ; \boldsymbol{w}),
$$

此处的 $\delta=\widehat{q}-\widehat{y}=14-15.5=-1.5$ 称作 TD 误差 (TD error)。做一次梯度下降更新模 型参数 $\boldsymbol{w}$ :

$$
\boldsymbol{w} \leftarrow \boldsymbol{w}-\alpha \cdot \delta \cdot \nabla_{\boldsymbol{w}} Q(\text { “北京”, “上海”; } \boldsymbol{w}) .
$$

如果你仍然不理解 TD 算法, 那么请换个角度来思考问题。模型估计从北京到上海 全程需要 $\widehat{q}=14$ 小时, 模型还估计从济南到上海需要 $\widehat{q}^{\prime}=11$ 小时。这就相当于模型做 了这样的估计：从北京到济南需要的时间为

$$
\widehat{q}-\widehat{q}^{\prime}=14-11=3 .
$$

而我真实花费 $r=4.5$ 小时从北京到济南。模型的估计与我的真实观测之差为

$$
\delta=3-4.5=-1.5 .
$$

这就是 TD 误差! 以上分析说明 TD 误差 $\delta$ 就是模型估计与真实观测之差。TD 算法的目 的是通过更新参数 $\boldsymbol{w}$ 使得损失 $L(\boldsymbol{w})=\frac{1}{2} \delta^{2}$ 减小。

1根据定义, TD 目标是 $\widehat{y}=r+\widehat{q}^{\prime}$, 其中 $\widehat{q}=Q$ (“济南”, “上海”; $\boldsymbol{w}$ ) 依赖于 $\boldsymbol{w}$ 。因此, $\widehat{y}$ 其实是 $\boldsymbol{w}$ 的函数。 然而 TD 算法忽视这一点, 在求梯度的时候, 将 $\widehat{y}$ 视为常数, 而非 $\boldsymbol{w}$ 的函数。

## $4.3$ 用 TD 训练 DQN

上一节以驾车时间预测为例介绍了 $\mathrm{TD}$ 算法。本节用 $\mathrm{TD}$ 算法训练 $\mathrm{DQN}$ 。第 $4.3 .1$ 小 节推导算法, 第 4.3.2 小节详细描述训练 DQN 的流程。注意, 本节推导出的是最原始的 $\mathrm{TD}$ 算法, 在实践中效果不佳。实际训练 $\mathrm{DQN}$ 的时候, 应当使用第 6 章介绍的高级技巧。

### 1 算法推导

下面我们推导训练 DQN 的 TD 算法。 ${ }^{2}$ 回忆一下回报的定义： $U_{t}=\sum_{k=t}^{n} \gamma^{k-t} \cdot R_{k}$, $U_{t+1}=\sum_{k=t+1}^{n} \gamma^{k-t-1} \cdot R_{k}$ 。由 $U_{t}$ 和 $U_{t+1}$ 的定义可得：

$$
U_{t}=R_{t}+\gamma \cdot \underbrace{\sum_{k=t+1}^{n} \gamma^{k-t-1} \cdot R_{k}}_{=U_{t+1}} .
$$

回忆一下，最优动作价值函数可以写成

$$
Q_{\star}\left(s_{t}, a_{t}\right)=\max _{\pi} \mathbb{E}\left[U_{t} \mid S_{t}=s_{t}, A_{t}=a_{t}\right] .
$$

从公式 (4.1) 和 (4.2) 出发, 经过一系列数学推导 (见附录 $\mathrm{A}$ ), 可以得到下面的定理。这 个定理是最优贝尔曼方程 (optimal Bellman equations) 的一种形式。

## 定理 4.1. 最优贝尔曼方程

$$
\underbrace{Q_{\star}\left(s_{t}, a_{t}\right)}_{U_{t} \text { 的期望 }}=\mathbb{E}_{S_{t+1} \sim p\left(\cdot \mid s_{t}, a_{t}\right)}[R_{t}+\gamma \cdot \underbrace{\max _{A \in \mathcal{A}} Q_{\star}\left(S_{t+1}, A\right)}_{U_{t+1} \text { 的期望 }} \mid S_{t}=s_{t}, A_{t}=a_{t}] .
$$

贝尔曼方程的右边是个期望, 我们可以对期望做蒙特卡洛近似。当智能体执行动作 $a_{t}$ 之后, 环境通过状态转移函数 $p\left(s_{t+1} \mid s_{t}, a_{t}\right)$ 计算出新状态 $s_{t+1}$ 。奖励 $R_{t}$ 最多只依赖于 $S_{t} 、 A_{t} 、 S_{t+1}$ 。那么当我们观测到 $s_{t} 、 a_{t} 、 s_{t+1}$ 时, 则奖励 $R_{t}$ 也被观测到, 记作 $r_{t}$ 。有 了四元组

$$
\left(s_{t}, a_{t}, r_{t}, s_{t+1}\right),
$$

我们可以计算出

$$
r_{t}+\gamma \cdot \max _{a \in \mathcal{A}} Q_{\star}\left(s_{t+1}, a\right) .
$$

它可以看做是下面这项期望的蒙特卡洛近似：

$$
\mathbb{E}_{S_{t+1} \sim p\left(\cdot \mid s_{t}, a_{t}\right)}\left[R_{t}+\gamma \cdot \max _{A \in \mathcal{A}} Q_{\star}\left(S_{t+1}, A\right) \mid S_{t}=s_{t}, A_{t}=a_{t}\right] .
$$

由定理 $4.1$ 和上述的蒙特卡洛近似可得:

$$
Q_{\star}\left(s_{t}, a_{t}\right) \approx r_{t}+\gamma \cdot \max _{a \in \mathcal{A}} Q_{\star}\left(s_{t+1}, a\right) .
$$

这是不是很像驾驶时间预测问题? 左边的 $Q_{\star}\left(s_{t}, a_{t}\right)$ 就像是模型预测 “北京到上海”的总 时间, $r_{t}$ 像是实际观测的 “北京到济南”的时间, $\gamma \cdot \max _{a \in \mathcal{A}} Q_{\star}\left(s_{t+1}, a\right)$ 相当于模型预测

2严格地讲, 此处推导的是 “ $\mathrm{Q}$ 学习算法”, 它属于 TD 算法的一种。本节就称其为 TD 算法; 下一节再具体介 绍 $\mathrm{Q}$ 学习算法。 剩余路程“济南到上海”的时间。见图 $4.4$ 中的类比。

预计需要14小时
![](https://cdn.mathpix.com/cropped/2023_02_03_f46f5cf0e4de5b9996dcg-065.jpg?height=580&width=1396&top_left_y=441&top_left_x=270)

图 4.4: 用“驾车时间”类比 DQN。

把公式 (4.3) 中的最优动作价值函数 $Q_{\star}(s, a)$ 替换成神经网络 $Q(s, a ; \boldsymbol{w})$, 得到:

$$
\underbrace{Q\left(s_{t}, a_{t} ; \boldsymbol{w}\right)}_{\text {预测 } \widehat{q}_{t}} \approx \underbrace{r_{t}+\gamma \cdot \max _{a \in \mathcal{A}} Q\left(s_{t+1}, a ; \boldsymbol{w}\right)}_{\text {TD 目标 } \widehat{y}_{t}} .
$$

左边的 $\widehat{q}_{t} \triangleq Q\left(s_{t}, a_{t} ; \boldsymbol{w}\right)$ 是神经网络在 $t$ 时刻做出的预测, 其中没有任何事实成分。右边 的 TD 目标 $\widehat{y}_{t}$ 是神经网络在 $t+1$ 时刻做出的预测, 它部分基于真实观测到的奖励 $r_{t}$ 。 $\widehat{q}_{t}$ 和 $\widehat{y}_{t}$ 两者都是对最优动作价值 $Q_{\star}\left(s_{t}, a_{t}\right)$ 的估计, 但是 $\widehat{y}_{t}$ 部分基于事实, 因此比 $\widehat{q}_{t}$ 更可 信。应当鼓励 $\widehat{q}_{t} \triangleq Q\left(s_{t}, a_{t} ; \boldsymbol{w}\right)$ 接近 $\widehat{y}_{t}$ 。定义损失函数：

$$
L(\boldsymbol{w})=\frac{1}{2}\left[Q\left(s_{t}, a_{t} ; \boldsymbol{w}\right)-\widehat{y}_{t}\right]^{2} .
$$

假装 $\widehat{y}$ 是常数 ${ }^{3}$, 计算 $L$ 关于 $\boldsymbol{w}$ 的梯度：

$$
\nabla_{\boldsymbol{w}} L(\boldsymbol{w})=\underbrace{\left(\widehat{q}_{t}-\widehat{y}_{t}\right)}_{\text {TD 误差 } \delta_{t}} \cdot \nabla_{\boldsymbol{w}} Q\left(s_{t}, a_{t} ; \boldsymbol{w}\right) .
$$

做一步梯度下降, 可以让 $\widehat{q}_{t}$ 更接近 $\widehat{y}_{t}$ :

$$
\boldsymbol{w} \leftarrow \boldsymbol{w}-\alpha \cdot \delta_{t} \cdot \nabla_{\boldsymbol{w}} Q\left(s_{t}, a_{t} ; \boldsymbol{w}\right) .
$$

这个公式就是训练 DQN 的 TD 算法。

### 2 训练流程

首先总结上面的结论。给定一个四元组 $\left(s_{t}, a_{t}, r_{t}, s_{t+1}\right)$, 我们可以计算出 DQN 的预 测值

$$
\widehat{q}_{t}=Q\left(s_{t}, a_{t} ; \boldsymbol{w}\right),
$$

${ }^{3}$ 实际上 $\widehat{y}_{t}$ 依赖于 $\boldsymbol{w}$, 但是我们假装 $\widehat{y}$ 是常数。 以及 TD 目标和 TD 误差：

$$
\widehat{y}_{t}=r_{t}+\gamma \cdot \max _{a \in \mathcal{A}} Q\left(s_{t+1}, a ; \boldsymbol{w}\right) \quad \text { 和 } \quad \delta_{t}=\widehat{q}_{t}-\widehat{y}_{t} .
$$

TD 算法用这个公式更新 DQN 的参数：

$$
\boldsymbol{w} \leftarrow \boldsymbol{w}-\alpha \cdot \delta_{t} \cdot \nabla_{\boldsymbol{w}} Q\left(s_{t}, a_{t} ; \boldsymbol{w}\right) .
$$

注意, 算法所需数据为四元组 $\left(s_{t}, a_{t}, r_{t}, s_{t+1}\right)$, 与控制智能体运动的策略 $\pi$ 无关。这就意 味着可以用任何策略控制智能体与环境交互, 同时记录下算法运动轨迹, 作为训练数据。 因此, DQN 的训练可以分割成两个独立的部分：收集训练数据、更新参数 $\boldsymbol{w}$ 。

收集训练数据 : 我们可以用任何策略函数 $\pi$ 去控制智能体与环境交互, 这个 $\pi$ 就叫 做行为策略 (behavior policy)。比较常用的是 $\epsilon$-greedy 策略:

$$
a_{t}= \begin{cases}\operatorname{argmax}_{a} Q\left(s_{t}, a ; \boldsymbol{w}\right), & \text { 以概率 }(1-\epsilon) ; \\ \text { 均匀抽取 } \mathcal{A} \text { 中的一个动作, } & \text { 以概率 } \epsilon .\end{cases}
$$

把智能体在一局游戏中的轨迹记作：

$$
s_{1}, a_{1}, r_{1}, s_{2}, a_{2}, r_{2}, \cdots, s_{n}, a_{n}, r_{n} .
$$

把一条轨迹划分成 $n$ 个 $\left(s_{t}, a_{t}, r_{t}, s_{t+1}\right)$ 这种四元组, 存入数组, 这个数组叫做经验回放 数组 (replay buffer)。

更新 DQN 参数 $\boldsymbol{w}$ : 随机从经验回放数组中取出一个四元组, 记作 $\left(s_{j}, a_{j}, r_{j}, s_{j+1}\right)$ 。 设 DQN 当前的参数为 $\boldsymbol{w}_{\text {now }}$, 执行下面的步骤对参数做一次更新, 得到新的参数 $\boldsymbol{w}_{\text {new }}$ 。

1. 对 $\mathrm{DQN}$ 做正向传播, 得到 $\mathrm{Q}$ 值：

$$
\widehat{q}_{j}=Q\left(s_{j}, a_{j} ; \boldsymbol{w}_{\text {now }}\right) \quad \text { 和 } \quad \widehat{q}_{j+1}=\max _{a \in \mathcal{A}} Q\left(s_{j+1}, a ; \boldsymbol{w}_{\text {now }}\right) .
$$

2. 计算 TD 目标和 TD 误差：

$$
\widehat{y}_{j}=r_{j}+\gamma \cdot \widehat{q}_{j+1} \quad \text { 和 } \quad \delta_{j}=\widehat{q}_{j}-\widehat{y}_{j} .
$$

3. 对 DQN 做反向传播, 得到梯度:

$$
\boldsymbol{g}_{j}=\nabla_{\boldsymbol{w}} Q\left(s_{j}, a_{j} ; \boldsymbol{w}_{\text {now }}\right) .
$$

4. 做梯度下降更新 DQN 的参数：

$$
\boldsymbol{w}_{\text {new }} \leftarrow \boldsymbol{w}_{\text {now }}-\alpha \cdot \delta_{j} \cdot \boldsymbol{g}_{j} .
$$

智能体收集数据、更新 DQN 参数这两者可以同时进行。可以在智能体每执行一个动作之 后, 对 $\boldsymbol{w}$ 做几次更新。也可以在每完成一局游戏之后, 对 $\boldsymbol{w}$ 做几次更新。

## 4 $Q$ 学习算法

上一节用 TD 算法训练 $\mathrm{DQN}$, 更准确地说, 我们用的 TD 算法叫做 Q 学习算法 (Qlearning)。TD 算法是一大类算法, 常见的有 $\mathrm{Q}$ 学习和 SARSA。 $\mathrm{Q}$ 学习的目的是学到最 优动作价值函数 $Q_{\star}$, 而 SARSA 的目的是学习动作价值函数 $Q_{\pi}$ 。下一章会介绍 SARSA 算法。

$\mathrm{Q}$ 学习是在 1989 年提出的, 而 DQN 则是 2013 年才提出。从 DQN 的名字（深度 $\mathrm{Q}$ 网络）就能看出 $\mathrm{DQN}$ 与 $\mathrm{Q}$ 学习的联系。最初的 $\mathrm{Q}$ 学习都是以表格形式出现的。虽然表 格形式的 $\mathrm{Q}$ 学习在实践中不常用, 但还是建议读者有所了解。

用表格表示 $Q_{\star}$ ： 假设状态 空间 $\mathcal{S}$ 和动作空间 $\mathcal{A}$ 都是有限 集, 即集合中元素数量有限。 ${ }^{4}$ 比 如, $\mathcal{S}$ 中一共有 3 种状态, $\mathcal{A}$ 中一 共有 4 种动作。那么最优动作价 值函数 $Q_{\star}(s, a)$ 可以表示为一个 $3 \times 4$ 的表格, 比如右边的表格。 基于当前状态 $s_{t}$, 做决策时使用

\begin{tabular}{|c|c|c|c|c|}
\cline { 2 - 5 } \multicolumn{1}{c|}{} & $\begin{gathered}\text { 第 } 1 \text { 种 } \\
\text { 动作 }\end{gathered}$ & $\begin{gathered}\text { 第 } 2 \text { 种 } \\
\text { 动作 }\end{gathered}$ & $\begin{gathered}\text { 第 } 3 \text { 种 } \\
\text { 动作 }\end{gathered}$ & $\begin{gathered}\text { 第 } 4 \text { 种 } \\
\text { 动作 }\end{gathered}$ \\
\hline $\begin{gathered}\text { 第 } 1 \text { 种 } \\
\text { 状态 }\end{gathered}$ & 380 & $-95$ & 20 & 173 \\
\hline $\begin{gathered}\text { 第 } 2 \text { 种 } \\
\text { 状态 }\end{gathered}$ & $-7$ & 64 & $-195$ & 210 \\
\hline $\begin{gathered}\text { 第 } 3 \text { 种 } \\
\text { 状态 }\end{gathered}$ & 152 & 72 & 413 & $-80$ \\
\hline
\end{tabular}

图 4.5: 最优动作价值函数 $Q_{\star}$ 表示成表格形式。 的公式

$$
a_{t}=\underset{a \in \mathcal{A}}{\operatorname{argmax}} Q_{\star}\left(s_{t}, a\right)
$$

的意思是找到 $s_{t}$ 对应的行 (3 行中的某一行), 找到该行最大的价值, 返回该元素对应的 动作。举个例子, 当前状态 $s_{t}$ 是第 2 种状态, 那么我们查看第 2 行, 发现该行最大的价 值是 210 , 对应第 4 种动作。那么应当执行的动作 $a_{t}$ 就是第 4 种动作。

该如何通过智能体的轨迹来学习这样一个表格呢? 答案是用一个表格 $\widetilde{Q}$ 来近似 $Q_{\star}$ 。 首先初始化 $\widetilde{Q}$, 可以让它是全零的表格。然后用表格形式的 $\mathrm{Q}$ 学习算法更新 $\widetilde{Q}$, 每次更 新表格的一个元素。最终 $\widetilde{Q}$ 会收敛到 $Q^{\star}$ 。

算法推导： 首先复习一下最优贝尔曼方程：

$$
Q_{\star}\left(s_{t}, a_{t}\right)=\mathbb{E}_{S_{t+1} \sim p\left(\cdot \mid s_{t}, a_{t}\right)}\left[R_{t}+\gamma \cdot \max _{A \in \mathcal{A}} Q_{\star}\left(S_{t+1}, A\right) \mid S_{t}=s_{t}, A_{t}=a_{t}\right] .
$$

我们对方程左右两边做近似:

- 方程左边的 $Q_{\star}\left(s_{t}, a_{t}\right)$ 可以近似成 $\widetilde{Q}\left(s_{t}, a_{t}\right)$ 。 $\widetilde{Q}\left(s_{t}, a_{t}\right)$ 是表格在 $t$ 时刻对 $Q_{\star}\left(s_{t}, a_{t}\right)$ 做出的估计。

- 方程右边的期望是关于下一时刻状态 $S_{t+1}$ 求的。给定当前状态 $s_{t}$, 智能体执行动 作 $a_{t}$, 环境会给出奖励 $r_{t}$ 和新的状态 $s_{t+1}$ 。用观测到的 $r_{t}$ 和 $s_{t+1}$ 对期望做蒙特卡 洛近似, 得到：

$$
r_{t}+\gamma \cdot \max _{a \in \mathcal{A}} Q_{\star}\left(s_{t+1}, a\right) .
$$

4如果 $\mathcal{A}$ 是有限集, 而 $\mathcal{S}$ 是无限集, 那么我们可以用神经网络形式的 Q 学习, 即上一节的 DQN。如果 $\mathcal{A}$ 是 无限集, 则问题属于连续控制, 应当使用连续控制的方法, 见第 10 章。 - 进一步把公式 (4.4) 中的 $Q_{\star}$ 近似成 $\widetilde{Q}$, 得到

$$
\widehat{y}_{t} \triangleq r_{t}+\gamma \cdot \max _{a \in \mathcal{A}} \widetilde{Q}\left(s_{t+1}, a\right) .
$$

把它称作 TD 目标。它是表格在 $t+1$ 时刻对 $Q_{\star}\left(s_{t}, a_{t}\right)$ 做出的估计。

$\widetilde{Q}\left(s_{t}, a_{t}\right)$ 和 $\widehat{y}_{t}$ 都是对最优动作价值 $Q_{\star}\left(s_{t}, a_{t}\right)$ 的估计。由于 $\widehat{y}_{t}$ 部分基于真实观测到的奖 励 $r_{t}$, 我们认为 $\widehat{y}_{t}$ 是更可靠的估计, 所以鼓励 $\widetilde{Q}\left(s_{t}, a_{t}\right)$ 更接近 $\widehat{y}_{t}$ 。更新表格 $\widetilde{Q}$ 中 $\left(s_{t}, a_{t}\right)$ 位置上的元素:

$$
\widetilde{Q}\left(s_{t}, a_{t}\right) \leftarrow(1-\alpha) \cdot \widetilde{Q}\left(s_{t}, a_{t}\right)+\alpha \cdot \widehat{y}_{t} .
$$

这样可以使得 $\widetilde{Q}\left(s_{t}, a_{t}\right)$ 更接近 $\widehat{y}_{t}$ 。 $\mathrm{Q}$ 学习的目的是让 $\widetilde{Q}$ 逐渐趋近于 $Q_{\star}$ 。

收集训练数据： $\mathrm{Q}$ 学习更新 $\widetilde{Q}$ 的公式不依赖于具体的策略。我们可以用任意策略控 制智能体, 与环境交互, 把得到的轨迹划分成 $\left(s_{t}, a_{t}, r_{t}, s_{t+1}\right)$ 这样的四元组, 存入经验 回放数组。这个控制智能体的策略叫做行为策略 (behavior policy), 比较常用的行为策略 是 $\epsilon$-greedy:

$$
a_{t}= \begin{cases}\operatorname{argmax}_{a} \widetilde{Q}\left(s_{t}, a\right), & \text { 以概率 }(1-\epsilon) ; \\ \text { 均匀抽取 } \mathcal{A} \text { 中的一个动作, } & \text { 以概率 } \epsilon .\end{cases}
$$

事后用经验回放更新表格 $\widetilde{Q}$, 可以重复利用收集到的四元组。

经验回放更新表格 $\widetilde{Q}$ ：随机从经验回放数组中抽取一个四元组, 记作 $\left(s_{j}, a_{j}, r_{j}, s_{j+1}\right)$ 。 设当前表格为 $\widetilde{Q}_{\text {now }}$ 。更新表格中 $\left(s_{j}, a_{j}\right)$ 位置上的元素, 把更新之后的表格记作 $\widetilde{Q}_{\text {new }}$ 。

1. 把表格 $\widetilde{Q}_{\text {now }}$ 中第 $\left(s_{j}, a_{j}\right)$ 位置上的元素记作：

$$
\widehat{q}_{j}=\widetilde{Q}_{\text {now }}\left(s_{j}, a_{j}\right) .
$$

2. 查看表格 $\widetilde{Q}_{\text {now }}$ 的第 $s_{j+1}$ 行, 把该行的最大值记作：

$$
\widehat{q}_{j+1}=\max _{a} \widetilde{Q}_{\text {now }}\left(s_{j+1}, a\right) .
$$

3. 计算 TD 目标和 TD 误差：

$$
\widehat{y}_{j}=r_{j}+\gamma \cdot \widehat{q}_{j+1}, \quad \delta_{j}=\widehat{q}_{j}-\widehat{y}_{j} .
$$

4. 更新表格中 $\left(s_{j}, a_{j}\right)$ 位置上的元素:

$$
\widetilde{Q}_{\text {new }}\left(s_{j}, a_{j}\right) \leftarrow \widetilde{Q}_{\text {now }}\left(s_{j}, a_{j}\right)-\alpha \cdot \delta_{j} .
$$

收集经验与更新表格 $\widetilde{Q}$ 可以同时进行。每当智能体执行一次动作, 我们可以用经验回放 对 $\widetilde{Q}$ 做几次更新。也可以每当完成一局游戏, 对 $\widetilde{Q}$ 做几次更新。

## $4.5$ 同策略 (On-policy) 与异策略 (Off-policy)

在强化学习中经常会遇到两个专业术语：同策略 (on-policy) 和异策略（off-policy)。 为了解释同策略和异策略, 我们要从行为策略 (behavior policy) 和目标策略 (target policy) 讲起。

在强化学习中, 我们让智能体与环境交互, 记录下观测到的状态、动作、奖励, 用 这些经验来学习一个策略函数。在这一过程中, 控制智能体与环境交互的策略被称作行 为策略。行为策略的作用是收集经验 (experience), 即观测的状态、动作、奖励。

强化学习的目的是得到一个策略函数, 用这个策略函数来控制智能体。这个策略函 数就叫做目标策略。在本章中, 目标策略是一个确定性的策略, 即用 DQN 控制智能体:

$$
a_{t}=\underset{a}{\operatorname{argmax}} Q\left(s_{t}, a ; \boldsymbol{w}\right) .
$$

本章的 $\mathrm{Q}$ 学习算法用任意的行为策略收集 $\left(s_{t}, a_{t}, r_{t}, s_{t+1}\right)$ 这样的四元组, 然后拿它们训 练目标策略, 即 $\mathrm{DQN}$ 。

行为策略和目标策略可以相同, 也可以不同。同策略是指用相同的行为策略和目标 策略, 后面章节会介绍同策略。异策略是指用不同的行为策略和目标策略, 本章的 DQN 属于异策略。同策略和异策略如图 4.6、4.7 所示。

由于 $D Q N$ 是异策略, 行为策略可以不同于目标策略, 可以用任意的行为策略收集经 验, 比如最常用的行为策略是 $\epsilon$-greedy:

$$
a_{t}= \begin{cases}\operatorname{argmax}_{a} Q\left(s_{t}, a ; \boldsymbol{w}\right), & \text { 以概率 }(1-\epsilon) ; \\ \text { 均匀抽取 } \mathcal{A} \text { 中的一个动作, } & \text { 以概率 } \epsilon .\end{cases}
$$

让行为策略带有随机性的好处在于能探索更多没见过的状态。在实验中, 初始的时候让 $\epsilon$ 比较大 (比如 $\epsilon=0.5$ ); 在训练的过程中, 让 $\epsilon$ 逐渐衰减, 在几十万步之后衰减到较小 的值 (比如 $\epsilon=0.01)$, 此后固定住 $\epsilon=0.01$ 。

异策略的好处是可以用行为策略收集经验, 把 $\left(s_{t}, a_{t}, r_{t}, s_{t+1}\right)$ 这样的四元组记录到 一个数组里, 在事后反复利用这些经验去更新目标策略。这个数组被称作经验回放数组 (replay buffer)，这种训练方式被称作经验回放 (experience replay)。注意, 经验回放只 适用于异策略, 不适用于同策略, 其原因是收集经验时用的行为策略不同于想要训练出 的目标策略。

![](https://cdn.mathpix.com/cropped/2023_02_03_f46f5cf0e4de5b9996dcg-069.jpg?height=332&width=334&top_left_y=2084&top_left_x=267)

图 4.6: 同策略。

![](https://cdn.mathpix.com/cropped/2023_02_03_f46f5cf0e4de5b9996dcg-069.jpg?height=252&width=673&top_left_y=2144&top_left_x=703)

图 4.7: 异策略。

## 第 4 章 知识点

- $\mathrm{DQN}$ 是对最优动作价值函数 $Q_{\star}$ 的近似。DQN 的输入是当前状态 $s_{t}$, 输出是每个 动作的 $\mathrm{Q}$ 值。 $\mathrm{DQN}$ 要求动作空间 $\mathcal{A}$ 是离散集合, 集合中的元素数量有限。如果动 作空间 $\mathcal{A}$ 的大小是 $k$, 那么 DQN 的输出就是 $k$ 维向量。DQN 可以用于做决策, 智 能体执行 $\mathrm{Q}$ 值最大的动作。

- TD 算法的目的在于让预测更接近实际观测。以驾车问题为例, 如果使用 TD 算法, 无需完成整个旅途就能做梯度下降更新模型。请读者理解并记忆 TD 目标、TD 误 差的定义, 它们将出现在所有价值学习的章节中。

- $\mathrm{Q}$ 学习算法是 TD 算法的一种, 可以用于训练 $\mathrm{DQN} \mathrm{Q}$ 学习算法由最优贝尔曼方程 推导出。Q 学习算法属于异策略, 允许使用经验回放。由任意行为策略收集经验, 存入经验回放数组。事后做经验回放，用 TD 算法更新 DQN 参数。

- 如果状态空间 $\mathcal{S} 、$ 动作空间 $\mathcal{A}$ 都是较小的有限离散集合, 那么可以用表格形式的 $\mathrm{Q}$ 学习算法学习 $Q_{\star}$ 。如今表格形式的 $\mathrm{Q}$ 学习已经不常用。

- 请读者理解同策略、异策略、目标策略、行为策略这几个专业术语, 理解同策略与 异策略的区别。异策略的好处在于允许做经验回放, 反复利用过去收集的经验。但 这不意味着异策略一定优于同策略。

## 第 4 章 相关文献 $\boldsymbol{s}$

DQN 首先由 Mnih 等人在 2013 年提出 ${ }^{[76]}$ ，其训练用的算法与本章介绍的基本一 致, 这种简单的训练算法实践中效果不佳。这篇论文用 Atari 游戏评价 DQN 的表现, 虽 然 DQN 的表现优于已有方法, 但是它还是比人类的表现差一截。相同的作者在 2015 年 发表了 DQN 的改进版本 ${ }^{[77]}$, 其主要改进在于使用“目标网络” (target network), 这个版 本的 DQN 在 Atari 游戏上的表现超越了人类玩家。

DQN 的本质是对最优动作价值函数 $Q_{\star}$ 的函数近似。早在 1995 年和 1997 年发表的 论文 ${ }^{[55,113]}$ 就把函数近似用于价值学习中。本章使用的 TD 算法叫做 $\mathrm{Q}$ 学习算法, 它 是由 Watkins 在 1989 年在博士论文 ${ }^{[123]}$ 提出的。Watkins 和 Dayan 发表在 1992 年的论 文 $^{[122]}$ 分析了 $\mathrm{Q}$ 学习的收敛。1994 年的论文 ${ }^{[57,112]}$ 改进了 $\mathrm{Q}$ 学习算法的收敛分析。 训练 DQN 用到的经验回放是由 Lin 在 1993 年的博士论文 ${ }^{[68]}$ 中提出的。

## 第 4 章 习题

1. DQN 是对 的近似。
A. 动作价值函数 $Q_{\pi}$ 。
B. 最优动作价值函数 $Q_{\star}$ 。
C. 状态价值函数 $V_{\pi}$ 。
D. 最优状态价值函数 $V_{\star}$ 。
E. 策略函数 $\pi$ 。
F. 状态转移函数 $p$ 。

2. 设 $\mathcal{A}=\{$ 上, 下, 左, 右 $\}$ 为动作空间, $s_{t}$ 为当前状态, $Q_{\star}$ 为最优动作价值函数。 策略函数输出:

$$
\begin{aligned}
& Q_{\star}\left(s_{t}, \text { 上 }\right)=930, \\
& Q_{\star}\left(s_{t}, \text { 下 }\right)=-60, \\
& Q_{\star}\left(s_{t}, \text { 左 }\right)=120, \\
& Q_{\star}\left(s_{t}, \text { 右 }\right)=321 .
\end{aligned}
$$

请问哪个动作会成为 $a_{t}$ ?
A. 上。
B. 下。
C. 四种动作都有可能。

3. DQN 的输出层用什么激活函数?
A. 不需要激活函数, 因为 $\mathrm{Q}$ 值可正可负, 没有取值范围。
B. 用 sigmoid 激活函数, 因为 $\mathrm{Q}$ 值介于 0 到 1 之间。
C. 用 ReLU 激活函数, 因为 $\mathrm{Q}$ 值非负。
D. 用 softmax 激活函数, 因为 DQN 的输出是一个概率分布。

4. 设状态空间、动作空间的大小分别是 $|\mathcal{S}|=3 、|\mathcal{A}|=4$ 。如图 $4.8$ 所示, 最优动作 价值函数 $Q_{\star}$ 可以表示为表格形式。设 $s$ 为第 3 种状态。那么 $\max _{a \in \mathcal{A}} Q_{\star}(s, a)=$ 。基于状态 $s$, 智能体应该执行动作

\begin{tabular}{|c|c|c|c|c|}
\cline { 2 - 5 } \multicolumn{1}{c|}{} & 上 & 下 & 左 & 右 \\
\hline $\begin{gathered}\text { 第 } 1 \text { 种 } \\
\text { 状态 }\end{gathered}$ & 98 & 120 & $-55$ & 780 \\
\hline $\begin{gathered}\text { 第 } 2 \text { 种 } \\
\text { 状态 }\end{gathered}$ & 15 & $-64$ & 212 & 99 \\
\hline $\begin{gathered}\text { 第 } 3 \text { 种 } \\
\text { 状态 }\end{gathered}$ & 200 & 789 & 10 & $-60$ \\
\hline
\end{tabular}

图 4.8: 最优动作价值函数 $Q_{\star}$ 表示成表格形式。

5. 驾车按照路线“甲 $\rightarrow 乙 \rightarrow$ 芮”行驶。从甲地出发时, 模型预计需要行驶 20 小时。 实际行驶 6 小时到达乙地。模型预计还需 12 小时才能到达丙地。如果我们用 TD 算法更新模型，那么 TD 目标是 $\widehat{y}=$ 小时, TD 误差的绝对值是 $|\delta|=$ 小时。

6. 同策略 (on-policy) 使用经验回放。异策略（off-policy) 使用 经验回放。
A. 允许
B. 不允许

7. $\mathrm{Q}$ 学习用 控制智能体与环境交互。
A. 行为策略 (behavior policy)
B. 目标策略 (target policy)
