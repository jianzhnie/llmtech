
## 第 6 章 价值学习高级技巧

第 4 章介绍了 $\mathrm{DQN}$, 并且用 $\mathrm{Q}$ 学习算法训练 $\mathrm{DQN}$ 。如果读者第 4 章最原始的 $\mathrm{Q}$ 学 习算法, 那么训练出的 DQN 效果会很不理想。想要提升 DQN 的表现, 需要用本章的高 级技巧。文献中已经有充分实验结果表明这些高级技巧对 DQN 非常有效, 而且这些技 巧不冲突, 可以一起使用。有些技巧并不局限于 $D Q N$, 而是可以应用于多种价值学习和 策略学习方法。

第 6.1、6.2 节介绍两种方法改进 Q 学习算法。第 $6.1$ 节介绍经验回放 (experience replay）和优先经验回放（prioritized experience replay）。第 $6.2$ 节讨论 DQN 的高估问题 以及解决方案――目标网络（target network）和双 Q 学习算法（double Q-learning)。

第 6.3、6.4 节介绍两种方法改进 DQN 的神经网络结构 (不是对 Q 学习算法的改进)。 第 $6.3$ 节介绍对决网络 (dueling network), 它把动作价值 (action value) 分解成状态价值 (state value) 与优势 (advantage)。第 $6.4$ 节介绍噪声网络 (noisy net), 它往神经网络的 参数中加入随机噪声, 鼓励探索。

## 1 经验回放

经验回放 (experience replay) 是强化学习中一个重要的技巧, 可以大幅提升强化学习的表现。 经验回放的意思是把智能体与环 境交互的记录 (即经验) 储存到 一个数组里, 事后反复利用这些 经验训练智能体。这个数组被称

![](https://cdn.mathpix.com/cropped/2023_02_03_f46f5cf0e4de5b9996dcg-089.jpg?height=334&width=465&top_left_y=1575&top_left_x=864)

图 6.1: 经验回放数组。 经验回放数组

(最近 b 条记录) 为经验回放数组 (replay buffer)。

具体来说, 把智能体的轨迹划分成 $\left(s_{t}, a_{t}, r_{t}, s_{t+1}\right)$ 这样的四元组, 存入一个数组。需 要人为指定数组的大小 (记作 $b$ )。数组中只保留最近 $b$ 条数据; 当数组存满之后, 删除 掉最旧的数据。数组的大小 $b$ 是个需要调的超参数, 会影响训练的结果。通常设置 $b$ 为 $10^{5} \sim 10^{6}$ 。

在实践中, 要等回放数组中有足够多的四元组时, 才开始做经验回放更新 DQN。根 据论文 ${ }^{[48]}$ 的实验分析, 如果将 DQN 用于 Atari 游戏, 最好是在收集到 20 万条四元组 时才开始做经验回放更新 DQN ; 如果是用更好的 Rainbow DQN, 收集到 8 万条四元组 时就可以开始更新 DQN 在回放数组中的四元组数量不够的时候, DQN 只与环境交互, 而不去更新 DQN 参数, 否则实验效果不好。

### 1 经验回放的优点

经验回放的一个好处在于打破序列的相关性。训练 DQN 的时候, 每次我们用一个四 元组对 DQN 的参数做一次更新。我们希望相邻两次使用的四元组是独立的。然而当智 能体收集经验的时候, 相邻两个四元组 $\left(s_{t}, a_{t}, r_{t}, s_{t+1}\right)$ 和 $\left(s_{t+1}, a_{t+1}, r_{t+1}, s_{t+2}\right)$ 有很强 的相关性。依次使用这些强关联的四元组训练 DQN, 效果往往会很差。经验回放每次从 数组里随机抽取一个四元组, 用来对 DQN 参数做一次更新。这样随机抽到的四元组都 是独立的，消除了相关性。

经验回放的另一个好处是重复利用收集到的经验, 而不是用一次就丢弃, 这样可以 用更少的样本数量达到同样的表现。重复利用经验、不重复利用经验的收敛曲线通常如 图 $6.2$ 所示。图的横轴是样本数量, 纵轴是平均回报。

注 在阅读文献的时候请注意“样 本数量” (sample complexity) 与“更 新次数”两者的区别。样本数量是 指智能体从环境中获取的奖励 $r$ 的数量。而一次更新的意思是从 经验回放数组里取出一个或多个 四元组, 用它对参数 $\boldsymbol{w}$ 做一次更 新。通常来说, 样本数量更重要, 因为在实际应用中收集经验比较 困难。比如, 在机器人的应用中, 需要在现实世界做一次实验才能

![](https://cdn.mathpix.com/cropped/2023_02_03_f46f5cf0e4de5b9996dcg-090.jpg?height=543&width=817&top_left_y=982&top_left_x=971)

图 6.2: 收敛曲线示意图。 收集到一条经验, 花费的时间和 金钱远大于做一次计算。相对而言, 做更新的次数不是那么重要, 更新次数只会影响训 练时的计算量而已。

### 2 经验回放的局限性

需要注意, 并非所有的强化学习方法都允许重复使用过去的经验。经验回放数组里 的数据全都是用行为策略 (behavior policy) 控制智能体收集到的。在收集经验同时, 我 们也在不断地改进策略。策略的变化导致收集经验时用的行为策略是过时的策略, 不同 于当前我们想要更新的策略一即目标策略 (target policy)。也就是说, 经验回放数组中 的经验通常是过时的行为策略收集的, 而我们真正想要学的目标策略不同于过时的行为 策略。

有些强化学习方法允许行为策略不同于目标策略。这样的强化学习方法叫做异策略 (off-policy)。比如 $\mathrm{Q}$ 学习、确定策略梯度 (DPG) 都属于异策略。由于它们允许行为策 略不同于目标策略, 过时的行为策略收集到的经验可以被重复利用。经验回放适用于异 策略。

有些强化学习方法要求行为策略与目标策略必须相同。这样的强化学习方法叫做同 策略 (on-policy)。比如 SARSA、REINFORCE、A2C 都属于同策略。它们要求经验必须 是当前的目标策略收集到的, 而不能使用过时的经验。经验回放不适用于同策略。

### 3 优先经验回放

优先经验回放 (prioritized experience replay) 是一种特殊的经验回放方法, 它比普通 的经验回放效果更好: 既能让收玫更快, 也能让收玫时的平均回报更高。经验回放数组里 有 $b$ 个四元组, 普通经验回放每次均匀抽样得到一个样本一一即四元组 $\left(s_{j}, a_{j}, r_{j}, s_{j+1}\right)$, 用它来更新 DQN 的参数。优先经验回放给每个四元组一个权重, 然后根据权重做非均匀 随机抽样。如果 DQN 对 $\left(s_{j}, a_{j}\right)$ 的价值判断不准确, 即 $Q\left(s_{j}, a_{j} ; \boldsymbol{w}\right)$ 离 $Q_{\star}\left(s_{j}, a_{j}\right)$ 较远, 则四元组 $\left(s_{j}, a_{j}, r_{j}, s_{j+1}\right)$ 应当有较高的权重。

为什么样本的重要性会有所不同呢? 设想你用强化学习训练一辆无人车。经验回放 数组中的样本绝大多数都是车辆正常行驶的情形, 只有极少数样本是意外情况, 比如旁 边车辆强行变道、行人横穿马路、警察封路要求绕行。数组中的样本的重要性显然是不 同的。绝大多数的样本都是车辆正常行驶, 而且正常行驶的情形很容易处理, 出错的可 能性非常小。意外情况的样本非常少, 但是又极其重要, 处理不好就会车毁人亡。所以 意外情况的样本应当有更高的权重, 受到更多关注。这两种样本不应该同等对待。

如何自动判断哪些样本更重要呢? 举个例子, 自动驾驶中的意外情况数量少、而且 难以处理, 导致 DQN 的预测 $Q\left(s_{j}, a_{j} ; \boldsymbol{w}\right)$ 严重偏离真实价值 $Q_{\star}\left(s_{j}, a_{j}\right)$ 。因此, 要是 $\left|Q\left(s_{j}, a_{j} ; \boldsymbol{w}\right)-Q_{\star}\left(s_{j}, a_{j}\right)\right|$ 较大, 则应该给样本 $\left(s_{j}, a_{j}, r_{j}, s_{j+1}\right)$ 较高的权重。然而实际上 我们不知道 $Q_{\star}$, 因此无从得知 $\left|Q\left(s_{j}, a_{j} ; \boldsymbol{w}\right)-Q_{\star}\left(s_{j}, a_{j}\right)\right|$ 。不妨把它替换成 $\mathrm{TD}$ 误差。回 忆一下, TD 误差的定义是：

$$
\delta_{j} \triangleq Q\left(s_{j}, a_{j} ; \boldsymbol{w}_{\text {now }}\right)-\underbrace{\left[r_{t}+\gamma \cdot \max _{a \in \mathcal{A}} Q\left(s_{j+1}, a ; \boldsymbol{w}_{\text {now }}\right)\right]}_{\text {即 TD 目标 }} .
$$

如果 TD 误差的绝对值 $\left|\delta_{j}\right|$ 大, 说明 DQN 对 $\left(s_{j}, a_{j}\right)$ 的真实价值的评估不准确, 那么应 该给 $\left(s_{j}, a_{j}, r_{j}, s_{j+1}\right)$ 设置较高的权重。

优先经验回放对数组里的样本做非均匀抽样。四元组 $\left(s_{j}, a_{j}, r_{j}, s_{j+1}\right)$ 的权重是 TD 误差的绝对值 $\left|\delta_{j}\right|$ 。有两种方法设置抽样概率。一种抽样概率是：

$$
p_{j} \propto\left|\delta_{j}\right|+\epsilon .
$$

此处的 $\epsilon$ 是个很小的数, 防止抽样概率接近零, 用于保证所有样本都以非零的概率被抽 到。另一种抽样方式先对 $\left|\delta_{j}\right|$ 做降序排列, 然后计算

$$
p_{j} \propto \frac{1}{\operatorname{rank}(j)} .
$$

此处的 $\operatorname{rank}(j)$ 是 $\left|\delta_{j}\right|$ 的序号。大的 $\left|\delta_{j}\right|$ 的序号小, 小的 $\left|\delta_{j}\right|$ 的序号大。两种方式的原理 是一样的, $\left|\delta_{j}\right|$ 大的样本被抽样到的概率大。

优先经验回放做非均匀抽样, 四元组 $\left(s_{j}, a_{j}, r_{j}, s_{j+1}\right)$ 被抽到的概率是 $p_{j}$ 。抽样是非 均匀的, 不同的样本有不同的抽样概率, 这样会导致 DQN 的预测有偏差。应该相应调整 学习率, 抵消掉不同抽样概率造成的偏差。TD 算法用“随机梯度下降”来更新参数：

$$
\boldsymbol{w}_{\text {new }} \leftarrow \boldsymbol{w}_{\text {now }}-\alpha \cdot \boldsymbol{g},
$$

此处的 $\alpha$ 是学习率, $\boldsymbol{g}$ 是损失函数关于 $\boldsymbol{w}$ 的梯度。如果用均匀抽样, 那么所有样本有相 同的学习率 $\alpha$ 。如果做非均匀抽样的话, 应该根据抽样概率来调整学习率 $\alpha$ 。如果一条 样本被抽样的概率大, 那么它的学习率就应该比较小。可以这样设置学习率:

$$
\alpha_{j}=\frac{\alpha}{\left(b \cdot p_{j}\right)^{\beta}},
$$

此处的 $b$ 是经验回放数组中样本的总数, $\beta \in(0,1)$ 是个需要调的超参数 ${ }^{1}$ 。

注 均匀抽样是一种特例, 即所有抽样概率都相等: $p_{1}=\cdots=p_{b}=\frac{1}{b}$ 。在这种情况下, 有 $\left(b \cdot p_{j}\right)^{\beta}=1$, 因此学习率都相同: $\alpha_{1}=\cdots=\alpha_{b}=\alpha$ 。

注 读者可能会问下面的问题。如果样本 $\left(s_{j}, a_{j}, r_{j}, s_{j+1}\right)$ 很重要, 它被抽到的概率 $p_{j}$ 很 大, 可是它的学习率却很小。当 $\beta=1$ 时, 如果抽样概率 $p_{j}$ 变大 10 倍, 则学习率 $\alpha_{j}$ 减 小 10 倍。抽样概率、学习率两者岂不是抵消了吗, 那么优先经验回放有什么意义呢? 大 抽样概率、小学习率两者其实并没有抵消, 因为下面两种方式并不等价：

- 设置学习率为 $\alpha$, 使用样本 $\left(s_{j}, a_{j}, r_{j}, s_{j+1}\right)$ 计算一次梯度, 更新一次参数 $\boldsymbol{w}$;

- 设置学习率为 $\frac{\alpha}{10}$, 使用样本 $\left(s_{j}, a_{j}, r_{j}, s_{j+1}\right)$ 计算十次梯度, 更新十次参数 $\boldsymbol{w}$ 。 乍看起来两种方式区别不大, 但其实第二种方式是对样本更有效的利用。第二种方式的 缺点在于计算量大了十倍, 所以第二种方式只被用于重要的样本。

$\begin{array}{ccccc}\text { 序号 } & \text { 四元组 } & \text { TD 误差 } & \text { 抽样概率 } & \text { 学习率 } \\ \vdots & \vdots & \vdots & \vdots & \vdots \\ j-1 & \left(s_{j-1}, a_{j-1}, r_{j-1}, s_{j}\right) & \delta_{j-1} & p_{j-1} \propto\left|\delta_{j-1}\right|+\epsilon & \alpha \cdot\left(b \cdot p_{j-1}\right)^{-\beta} \\ j & \left(s_{j}, a_{j}, r_{j}, s_{j+1}\right) & \delta_{j} & p_{j} \propto\left|\delta_{j}\right|+\epsilon & \alpha \cdot\left(b \cdot p_{j}\right)^{-\beta} \\ j+1 & \left(s_{j+1}, a_{j+1}, r_{j+1}, s_{j+2}\right) & \delta_{j+1} & p_{j+1} \propto\left|\delta_{j+1}\right|+\epsilon & \alpha \cdot\left(b \cdot p_{j+1}\right)^{-\beta} \\ \vdots & \vdots & \vdots & \vdots & \vdots\end{array}$

图 6.3: 优先经验回放数组。

优先经验回放数组如图 $6.3$ 所示。设 $b$ 为数组大小, 需要手动调整。如果样本（即四 元组) 的数量超过了 $b$, 那么要删除最旧的样本。数组里记录了四元组、TD 误差、抽样 概率、以及学习率。注意, 数组里存的 TD 误差 $\delta_{j}$ 是用很多步之前过时的 DQN 参数计 算出来的:

$$
\delta_{j}=Q\left(s_{j}, a_{j} ; \boldsymbol{w}_{\text {old }}\right)-\left[r_{t}+\gamma \cdot \max _{a \in \mathcal{A}} Q\left(s_{j+1}, a ; \boldsymbol{w}_{\text {old }}\right)\right]
$$

${ }^{1}$ 论文里建议一开始让 $\beta$ 比较小, 最终增长到 1 。 做经验回放的时候, 每次取出一个四元组, 用它计算出新的 TD 误差:

$$
\delta_{j}^{\prime}=Q\left(s_{j}, a_{j} ; \boldsymbol{w}_{\text {now }}\right)-\left[r_{t}+\gamma \cdot \max _{a \in \mathcal{A}} Q\left(s_{j+1}, a ; \boldsymbol{w}_{\text {now }}\right)\right],
$$

然后用它更新 DQN 的参数。用这个新的 $\delta_{j}^{\prime}$ 取代数组中旧的 $\delta_{j}$ 。

## $6.2$ 高估问题及解决方法

$\mathrm{Q}$ 学习算法有一个缺陷：用 Q 学习训练出的 DQN 会高估真实的价值, 而且高估通 常是非均匀的。这个缺陷导致 DQN 的表现很差。高估问题并不是 DQN 模型的缺陷, 而 是 $\mathrm{Q}$ 学习算法的缺陷。 $\mathrm{Q}$ 学习产生高估的原因有两个：第一, 自举导致偏差的传播; 第 二, 最大化导致 TD 目标高估真实价值。为了缓解高估, 需要从导致高估的两个原因下 手, 改进 $\mathrm{Q}$ 学习算法。双 Q 学习算法是一种有效的改进, 可以大幅缓解高估及其危害。

### 1 自举导致偏差的传播

在强化学习中, 自举意思是“用一个估算去更新同类的估算”, 类似于“自己把自己给 举起来”我们在第 $5.4$ 节讨论过 SARSA 算法中的自举。下面回顾训练 DQN 用的 Q 学习算 法, 研究其中存在的自举。算法每次从经验回放数组中抽取一个四元组 $\left(s_{j}, a_{j}, r_{j}, s_{j+1}\right)$ 。 然后执行以下步骤, 对 DQN 的参数做一轮更新:

1. 计算 TD 目标：

$$
\widehat{y}_{j}=r_{j}+\gamma \cdot \underbrace{\max _{a_{j+1} \in \mathcal{A}} Q\left(s_{j+1}, a_{j+1} ; \boldsymbol{w}_{\text {now }}\right)}_{\text {DQN 自己做出的估计 }} .
$$

2. 定义损失函数

$$
L(\boldsymbol{w})=\frac{1}{2}[\underbrace{Q\left(s_{j}, a_{j} ; \boldsymbol{w}\right)-\widehat{y}_{j}}_{\text {让 DQN 拟合令 }}]^{2} .
$$

3. 把 $\widehat{y}_{j}$ 看做常数, 做一次梯度下降更新参数:

$$
\boldsymbol{w}_{\text {new }} \leftarrow \boldsymbol{w}_{\text {now }}-\alpha \cdot \nabla_{w} L\left(\boldsymbol{w}_{\text {now }}\right) .
$$

第一步中的 $\mathrm{TD}$ 目标 $\widehat{y}_{j}$ 部分基于 $\mathrm{DQN}$ 自己做出的估计。第二步让 $\mathrm{DQN}$ 去拟合 $\widehat{y}_{j}$ 。这 就意味着我们用了 DQN 自己做出的估计去更新 DQN 自己, 这属于自举。

自举对 DQN 的训练有什么影响呢? $Q(s, a ; \boldsymbol{w})$ 是对价值 $Q_{\star}(s, a)$ 的近似, 最理想的

![](https://cdn.mathpix.com/cropped/2023_02_03_f46f5cf0e4de5b9996dcg-094.jpg?height=72&width=1465&top_left_y=1963&top_left_x=358)
值 $Q_{\star}\left(s_{j+1}, a_{j+1}\right)$, 则会发生下面的情况:

$$
\begin{array}{ccll}
& Q\left(s_{j+1}, a_{j+1} ; \boldsymbol{w}\right) & \text { 低估（或高估） } & Q_{\star}\left(s_{j+1}, a_{j+1}\right) \\
\Longrightarrow & \widehat{y}_{j} & \text { 低估（或高估） } & Q_{\star}\left(s_{j}, a_{j}\right) \\
\Longrightarrow & Q\left(s_{j}, a_{j} ; \boldsymbol{w}\right) & \text { 低估（或高估） } & Q_{\star}\left(s_{j}, a_{j}\right) .
\end{array}
$$

## 结论 6.1. 自举导致偏差的传播

如果 $Q\left(s_{j+1}, a_{j+1} ; \boldsymbol{w}\right)$ 是对真实价值 $Q_{\star}\left(s_{j+1}, a_{j+1}\right)$ 的低估（或高估), 就会导致 $Q\left(s_{j}, a_{j} ; \boldsymbol{w}\right)$ 低估 (或高估) 价值 $Q_{\star}\left(s_{j}, a_{j}\right)$ 。也就是说低估 (或高估) 从 $\left(s_{j+1}, a_{j+1}\right)$ 传播到 $\left(s_{j}, a_{j}\right)$ ，让更多的价值被低估（或高估）

### 2 最大化导致高估

首先用数学解释为什么最大化会导致高估。设 $x_{1}, \cdots, x_{d}$ 为任意 $d$ 个实数。往 $x_{1}$, $\cdots, x_{d}$ 中加入任意均值为零的随机噪声, 得到 $Z_{1}, \cdots, Z_{d}$, 它们是随机变量, 随机性来 源于随机噪声。很容易证明均值为零的随机噪声不会影响均值:

$$
\mathbb{E}\left[\operatorname{mean}\left(Z_{1}, \cdots, Z_{d}\right)\right]=\operatorname{mean}\left(x_{1}, \cdots, x_{d}\right) .
$$

用稍微复杂一点的证明, 可以得到：

$$
\mathbb{E}\left[\max \left(Z_{1}, \cdots, Z_{d}\right)\right] \geq \max \left(x_{1}, \cdots, x_{d}\right) .
$$

公式中的期望是关于噪声求的。这个不等式意味着先加入均值为零的噪声, 然后求最大 值, 会产生高估。

假设对于所有的动作 $a \in \mathcal{A}$ 和状态 $s \in \mathcal{S}, \mathrm{DQN}$ 的输出是真实价值 $Q_{\star}(s, a)$ 加上均 值为零的随机噪声 $\epsilon$ :

$$
Q(s, a ; \boldsymbol{w})=Q_{\star}(s, a)+\epsilon .
$$

显然 $Q(s, a ; \boldsymbol{w})$ 是对真实价值 $Q_{\star}(s, a)$ 的无偏估计。然而有这个不等式：

$$
\mathbb{E}_{\epsilon}\left[\max _{a \in \mathcal{A}} Q(s, a ; \boldsymbol{w})\right] \geq \max _{a \in \mathcal{A}} Q_{\star}(s, a) .
$$

公式说明哪怕 DQN 是对真实价值的无偏估计, 但是如果求最大化, DQN 就会高估真实 价值。复习一下, TD 目标是这样算出来的:

$$
\widehat{y}_{j}=r_{j}+\gamma \cdot \underbrace{\max _{a \in \mathcal{A}} Q\left(s_{j+1}, a ; \boldsymbol{w}\right)}_{\text {高估 } \max _{a \in \mathcal{A}} Q_{\star}\left(s_{j+1}, a\right)} \text {. }
$$

这说明 TD 目标 $\widehat{y}_{j}$ 通常是对真实价值 $Q_{\star}\left(s_{j}, a_{j}\right)$ 的高估。TD 算法鼓励 $Q\left(s_{j}, a_{j} ; \boldsymbol{w}\right)$ 接近 $\mathrm{TD}$ 目标 $\widehat{y}_{j}$ ，这会导致 $Q\left(s_{j}, a_{j} ; \boldsymbol{w}\right)$ 高估真实价值 $Q_{\star}\left(s_{j}, a_{j}\right)$ 。

## 结论 6.2. 最大化导致高估

即使 DQN 是真实价值 $Q_{\star}$ 的无偏估计, 只要 DQN 不恒等于 $Q_{\star}, \mathrm{TD}$ 目标就会高 估真实价值。 $\mathrm{TD}$ 目标是高估, 而 $\mathrm{Q}$ 学习算法鼓励 $\mathrm{DQN}$ 预测接近 TD 目标, 因此 DQN 会出现高估。

### 3 高估的危害

我们为什么要避免高估? 高估真的有害吗? 如果高估是均匀的, 则高估没有危害; 如 果高估非均匀, 就会有危害。举个例子, 动作空间是 $\mathcal{A}=\{$ 左, 右, 上 $\}$ 。给定当前状态 $s$, 每个动作有一个真实价值：

$$
Q_{\star}(s, \text { 左 })=200, \quad Q_{\star}(s, \text { 右 })=100, \quad Q_{\star}(s, \text { 上 })=230 .
$$

智能体应当选择动作 “上”, 因为“上”的价值最高。假如高估是均匀的, 所有的价值都被高 估了 100 ：

$$
Q(s, \text { 左; } \boldsymbol{w})=300, \quad Q(s, \text { 右; } \boldsymbol{w})=200, \quad Q(s, \text { 上; } \boldsymbol{w})=330 \text {. }
$$

那么动作“上”仍然有最大的价值, 智能体会选择“上”。这个例子说明高估本身不是问题, 只要所有动作价值被同等高估。

但实践中, 所有的动作价值会被同等高估吗? 每当取出一个四元组 $\left(s, a, r, s^{\prime}\right)$ 用来 更新一次 DQN, 就很有可能加重 $\mathrm{DQN}$ 对 $Q_{\star}(s, a)$ 的高估。对于同一个状态 $s$, 三种组 合 $(s$, 左)、 $(s$, 右)、 $(s$, 上) 出现在经验回放数组中的频率是不同的, 所以三种动作被高估 的程度是不同的。假如动作价值被高估的程度不同, 比如

$$
Q(s, \text { 左; } \boldsymbol{w})=280, \quad Q(s, \text { 右; } \boldsymbol{w})=300, \quad Q(s, \text { 上 } ; \boldsymbol{w})=260,
$$

那么智能体做出的决策就是向右走, 因为 “右”的价值貌似最高。但实际上 “右”是最差的动 作, 它的实际价值低于其余两个动作。

综上所述, 用 $\mathrm{Q}$ 学习算法训练 $\mathrm{DQN}$ 总会导致 $\mathrm{DQN}$ 高估真实价值。对于多数的 $s \in \mathcal{S}$ 和 $a \in \mathcal{A}$, 有这样的不等式:

$$
Q(s, a ; \boldsymbol{w})>Q_{\star}(s, a) .
$$

高估本身不是问题, 真正的麻烦在于 DQN 的高估往往是非均匀的。如果 DQN 有非均匀 的高估, 那么用 DQN 做出的决策是不可靠的。我们已经分析过导致高估的原因:

- TD 算法属于“自举”, 即用 DQN 的估计值去更新 DQN 自己。自举会导致偏差的传 播。如果 $Q\left(s_{j+1}, a_{j+1} ; \boldsymbol{w}\right)$ 是对 $Q_{\star}\left(s_{j+1}, a_{j+1}\right)$ 的高估, 那么高估会传播到 $\left(s_{j}, a_{j}\right)$, 让 $Q\left(s_{j}, a_{j} ; \boldsymbol{w}\right)$ 高估 $Q_{\star}\left(s_{j}, a_{j}\right)$ 。自举导致 $\mathrm{DQN}$ 的高估从一个二元组 $(s, a)$ 传播到 更多的二元组。

- $\mathrm{TD}$ 目标 $\widehat{y}$ 中包含一项最大化, 这会导致 $\mathrm{TD}$ 目标高估真实价值 $Q_{\star \circ} \mathrm{Q}$ 学习算法鼓 励 DQN 的预测接近 TD 目标, 因此 DQN 会高估 $Q_{\star}$ 。

找到了产生高估的原因, 就可以想办法解决问题。想要避免 DQN 的高估, 要么切断自 举, 要么避免最大化造成高估。注意, 高估并不是 $\mathrm{DQN}$ 自身的属性, 高估纯粹是算法造 成的。想要避免高估, 就要用更好的算法替代原始的 $\mathrm{Q}$ 学习算法。

### 4 使用目标网络


上文已经讨论过, 切断“自举”可以避免偏差的传播, 从而缓解 DQN 的高估。回顾一 下, $\mathrm{Q}$ 学习算法这样计算 $\mathrm{TD}$ 目标：

$$
\widehat{y}_{j}=r_{j}+\underbrace{\gamma \cdot \max _{a \in \mathcal{A}} Q\left(s_{j+1}, a ; \boldsymbol{w}\right)}_{\text {DQN 做出的估计 }} .
$$

然后做梯度下降更新 $\boldsymbol{w}$, 使得 $Q\left(s_{j}, a_{j} ; \boldsymbol{w}\right)$ 更接近 $\widehat{y}_{j}$ 。想要切断自举, 可以用另一个神 经网络计算 TD 目标, 而不是用 DQN 自己计算 TD 目标。另一个神经网络被称作目标网 络（target network）。把目标网络记作：

$$
Q\left(s, a ; \boldsymbol{w}^{-}\right) .
$$

它的神经网络结构与 DQN 完全相同, 但是参数 $\boldsymbol{w}^{-}$不同于 $\boldsymbol{w}$ 。

使用目标网络的话, $\mathrm{Q}$ 学习算法用下面的方式实现。每次随机从经验回放数组中取 一个四元组, 记作 $\left(s_{j}, a_{j}, r_{j}, s_{j+1}\right)$ 。设 $\mathrm{DQN}$ 和目标网络当前的参数分别为 $\boldsymbol{w}_{\text {now }}$ 和 $\boldsymbol{w}_{\text {now }}^{-}$, 执行下面的步骤对参数做一次更新：

1. 对 DQN 做正向传播, 得到:

$$
\widehat{q}_{j}=Q\left(s_{j}, a_{j} ; \boldsymbol{w}_{\text {now }}\right) .
$$

2. 对目标网络做正向传播, 得到

$$
\widehat{q}_{j+1}^{-}=\max _{a \in \mathcal{A}} Q\left(s_{j+1}, a ; \boldsymbol{w}_{\text {now }}^{-}\right) .
$$

3. 计算 TD 目标和 TD 误差：

$$
\widehat{y}_{j}^{-}=r_{j}+\gamma \cdot \widehat{q}_{j+1} \quad \text { 和 } \quad \delta_{j}=\widehat{q}_{j}-\widehat{y}_{j}^{-} .
$$

4. 对 DQN 做反向传播, 得到梯度 $\nabla_{\boldsymbol{w}} Q\left(s_{j}, a_{j} ; \boldsymbol{w}_{\text {now }}\right)$ 。

5. 做梯度下降更新 DQN 的参数：

$$
\boldsymbol{w}_{\text {new }} \leftarrow \boldsymbol{w}_{\text {now }}-\alpha \cdot \delta_{j} \cdot \nabla_{\boldsymbol{w}} Q\left(s_{j}, a_{j} ; \boldsymbol{w}_{\text {now }}\right) .
$$

6. 设 $\tau \in(0,1)$ 是需要手动调的超参数。做加权平均更新目标网络的参数:

$$
\boldsymbol{w}_{\text {new }}^{-} \leftarrow \tau \cdot \boldsymbol{w}_{\text {new }}+(1-\tau) \cdot \boldsymbol{w}_{\text {now }}^{-} .
$$

![](https://cdn.mathpix.com/cropped/2023_02_03_f46f5cf0e4de5b9996dcg-097.jpg?height=211&width=554&top_left_y=1211&top_left_x=248)

原始的 $\mathrm{Q}$ 学习算法

![](https://cdn.mathpix.com/cropped/2023_02_03_f46f5cf0e4de5b9996dcg-097.jpg?height=200&width=828&top_left_y=1228&top_left_x=845)

使用目标网络的 $Q$ 学习算讼

图 6.4: 对比原始 $\mathrm{Q}$ 学习算法、使用目标网络的 $\mathrm{Q}$ 学习算法。

如图 6.4 (左) 所示, 原始的 $\mathrm{Q}$ 学习算法用 $\mathrm{DQN}$ 计算 $\widehat{y}$, 然后拿 $\widehat{y}$ 更新 $\mathrm{DQN}$ 自己, 造成自举。如图 $6.4$ (右) 所示, 可以改用目标网络计算 $\widehat{y}$, 这样就避免了用 $\mathrm{DQN}$ 的估 计更新 DQN 自己, 降低自举造成的危害。然而这种方法不能完全避免自举, 原因是目标 网络的参数仍然与 $\mathrm{DQN}$ 相关。

### 5 双 $Q$ 学习算法

造成 DQN 高估的原因不是 DQN 模型本身的缺陷, 而是 $\mathrm{Q}$ 学习算法有不足之处：第 一, 自举造成偏差的传播; 第二, 最大化造成 TD 目标的高估。在 $\mathrm{Q}$ 学习算法中使用目 标网络, 可以缓解自举造成的偏差, 但是无助于缓解最大化造成的高估。本小节介绍双 $\mathbf{Q}$ 学习 (double Q learning) 算法, 它在目标网络的基础上做改进, 缓解最大化造成的高 估。

注 本小节介绍的双 $\mathrm{Q}$ 学习算法在文献中被称作 double DQN, 缩写 DDQN。本书不采用 $D D Q N$ 这名字, 因为这个名字比较误导。双 Q 学习（即所谓的 DDQN）只是一种 $\mathbf{T D}$ 算 法而已, 它可以把 $D Q N$ 训练得更好。双 $\mathrm{Q}$ 学习并没有用区别于 DQN 的模型。本节中的 模型只有一个, 就是 $\mathrm{DQN}$ 。我们讨论的只是训练 $\mathrm{DQN}$ 的三种 $\mathrm{TD}$ 算法：原始的 $\mathrm{Q}$ 学习、 用目标网络的 $\mathrm{Q}$ 学习、双 $\mathrm{Q}$ 学习。

为了解释原始的 $\mathrm{Q}$ 学习、用目标网络的 $\mathrm{Q}$ 学习、以及双 $\mathrm{Q}$ 学习三者的区别, 我们再 回顾一下 $\mathrm{Q}$ 学习算法中的 TD 目标：

$$
\widehat{y}_{j}=r_{j}+\gamma \cdot \max _{a \in \mathcal{A}} Q\left(s_{j+1}, a ; \boldsymbol{w}\right) .
$$

不妨把最大化拆成两步:

1. 选择一一即基于状态 $s_{j+1}$, 选出一个动作使得 $\mathrm{DQN}$ 的输出最大化:

$$
a^{\star}=\underset{a \in \mathcal{A}}{\operatorname{argmax}} Q\left(s_{j+1}, a ; \boldsymbol{w}\right) .
$$

2. 求值一一即计算 $\left(s_{j+1}, a^{\star}\right)$ 的价值, 从而算出 TD 目标:

$$
\widehat{y}_{j}=r_{j}+Q\left(s_{j+1}, a^{\star} ; \boldsymbol{w}\right) .
$$

以上是原始的 $\mathrm{Q}$ 学习算法, 选择和求值都用 $\mathrm{DQN}$ 。上一小节改进了 $\mathrm{Q}$ 学习, 选择和求 值都用目标网络:

$$
\begin{array}{ll}
\text { 选择 : } & a^{-}=\underset{a \in \mathcal{A}}{\operatorname{argmax}} Q\left(s_{j+1}, a ; \boldsymbol{w}^{-}\right), \\
\text {求值 : } & \widehat{y}_{j}^{-}=r_{j}+Q\left(s_{j+1}, a^{-} ; \boldsymbol{w}^{-}\right) .
\end{array}
$$

本小节介绍双 $\mathbf{Q}$ 学习, 第一步的选择用 $D Q N$, 第二步的求值用目标网络：

$$
\begin{array}{ll}
\text { 选择 : } & a^{\star}=\underset{a \in \mathcal{A}}{\operatorname{argmax}} Q\left(s_{j+1}, a ; \boldsymbol{w}\right), \\
\text { 求值 : } & \widetilde{y}_{j}=r_{j}+Q\left(s_{j+1}, a^{\star} ; \boldsymbol{w}^{-}\right) .
\end{array}
$$

为什么双 $\mathrm{Q}$ 学习可以缓解最大化造成的高估呢? 不难证明出这个不等式:

$$
\underbrace{Q\left(s_{j+1}, a^{\star} ; \boldsymbol{w}^{-}\right)}_{\text {双 Q 学习 }} \leq \underbrace{\max _{a \in \mathcal{A}} Q\left(s_{j+1}, a ; \boldsymbol{w}^{-}\right)}_{\text {用目标网络的 Q 学习 }} .
$$

因此,

$$
\underbrace{\widetilde{y}_{t}}_{\text {双 } \mathrm{Q} \text { 学习 }} \leq \underbrace{\widehat{y}_{t}}_{\text {用目标网络的 } \mathrm{Q} \text { 学习 }} .
$$

这个公式说明双 $\mathrm{Q}$ 学习得到的 $\mathrm{TD}$ 目标更小。也就是说, 与用目标网络的 $\mathrm{Q}$ 学习相比, 双 $\mathrm{Q}$ 学习缓解了高估。

双 $\mathrm{Q}$ 学习算法的流程如下。每次随机从经验回放数组中取出一个四元组, 记作 $\left(s_{j}\right.$, $\left.a_{j}, r_{j}, s_{j+1}\right)$ 。设 $\mathrm{DQN}$ 和目标网络当前的参数分别为 $\boldsymbol{w}_{\text {now }}$ 和 $\boldsymbol{w}_{\text {now }}^{-}$, 执行下面的步骤对 参数做一次更新:

1. 对 DQN 做正向传播, 得到：

$$
\widehat{q}_{j}=Q\left(s_{j}, a_{j} ; \boldsymbol{w}_{\text {now }}\right)
$$

2. 选择:

$$
a^{\star}=\underset{a \in \mathcal{A}}{\operatorname{argmax}} Q\left(s_{j+1}, a ; \boldsymbol{w}_{\text {now }}\right) .
$$

3. 求值:

$$
\widehat{q}_{j+1}=Q\left(s_{j+1}, a^{\star} ; \boldsymbol{w}_{\text {now }}^{-}\right) .
$$

4. 计算 TD 目标和 TD 误差：

$$
\widetilde{y}_{j}=r_{j}+\gamma \cdot \widehat{q}_{j+1} \quad \text { 和 } \quad \delta_{j}=\widehat{q}_{j}-\widetilde{y}_{j} .
$$

5. 对 DQN 做反向传播, 得到梯度 $\nabla_{\boldsymbol{w}} Q\left(s_{j}, a_{j} ; \boldsymbol{w}_{\text {now }}\right)$ 。

6. 做梯度下降更新 DQN 的参数：

$$
\boldsymbol{w}_{\text {new }} \leftarrow \boldsymbol{w}_{\text {now }}-\alpha \cdot \delta_{j} \cdot \nabla_{\boldsymbol{w}} Q\left(s_{j}, a_{j} ; \boldsymbol{w}_{\text {now }}\right) .
$$

7. 设 $\tau \in(0,1)$ 是需要手动调整的超参数。做加权平均更新目标网络的参数：

$$
\boldsymbol{w}_{\text {new }}^{-} \leftarrow \tau \cdot \boldsymbol{w}_{\text {new }}+(1-\tau) \cdot \boldsymbol{w}_{\text {now }}^{-} .
$$

### 6 总结

本节研究了 DQN 的高估问题以及解决方案。DQN 的高估不是 DQN 模型造成的, 不 是 $\mathrm{DQN}$ 的本质属性。高估只是因为原始 $\mathrm{Q}$ 学习算法不好。 $\mathrm{Q}$ 学习算法产生高估的原因 有两个：第一, 自举导致偏差从一个 $(s, a)$ 二元组传播到更多的二元组; 第二, 最大化造 成 $\mathrm{TD}$ 目标高估真实价值。

想要解决高估问题, 就要从自举、最大化这两方面下手。本节介绍了两种缓解高估 的算法：使用目标网络、双 $\mathrm{Q}$ 学习。 $\mathrm{Q}$ 学习算法与目标网络的结合可以缓解自举造成的 偏差。双 $\mathrm{Q}$ 学习基于目标网络的想法, 进一步将 TD 目标的计算分解成选择和求值两步, 缓解了最大化造成的高估。图 $6.5$ 总结了本节研究的三种算法。

$\begin{array}{ccccc} & \text { 选择 } & \text { 求值 } & \text { 自举造成偏差 } & \text { 最大化造成高估 } \\ \text { 原始 } \mathrm{Q} \text { 学习 } & \text { DQN } & \text { DQN } & \text { 严重 } & \text { 严重 } \\ \mathrm{Q} \text { 学习 + 目标网络 } & \text { 目标网络 } & \text { 目标网络 } & \text { 不严重 } & \text { 严重 } \\ & & & & \\ \text { 双 } \mathrm{Q} \text { 学习 } & \mathrm{DQN} & \text { 目标网络 } & \text { 不严重 } & \text { 不严重 }\end{array}$

图 6.5: 三种 TD 算法的对比。

注 如果使用原始 $\mathrm{Q}$ 学习算法, 自举和最大化都会造成严重高估。在实践中, 应当尽量使 用双 $\mathrm{Q}$ 学习, 它是三种算法中最好的。

注 如果使用 SARSA 算法 (比如在 actor-critic 中), 自举的问题依然存在, 但是不存在最 大化造成高估这一问题。对于 SARSA, 只需要解决自举问题, 所以应当将目标网络应用 到 SARSA。

## 3 对决网络 (Dueling Network)

本节介绍对决网络 (dueling network), 它是对 DQN 的神经网络结构的改进。它的基 本想法是将最优动作价值 $Q_{\star}$ 分解成最优状态价值 $V_{\star}$ 加最优优势 $D_{\star}$ 。对决网络的训练 与 $\mathrm{DQN}$ 完全相同, 可以用 $\mathrm{Q}$ 学习算法或者双 $\mathrm{Q}$ 学习算法。

### 1 最优优势函数

在介绍对决网络 (dueling network) 之前, 先复习一些基础知识。动作价值函数 $Q_{\pi}(s, a)$ 是回报的期望:

$$
Q_{\pi}(s, a)=\mathbb{E}\left[U_{t} \mid S_{t}=s, A_{t}=a\right] .
$$

最优动作价值 $Q_{\star}$ 的定义是：

$$
Q_{\star}(s, a)=\max _{\pi} Q_{\pi}(s, a), \quad \forall s \in \mathcal{S}, a \in \mathcal{A} .
$$

状态价值函数 $V_{\pi}(s)$ 是 $Q_{\pi}(s, a)$ 关于 $a$ 的期望:

$$
V_{\pi}(s)=\mathbb{E}_{A \sim \pi}\left[Q_{\pi}(s, A)\right] .
$$

最优状态价值函数 $V_{\star}$ 的定义是:

$$
V_{\star}(s)=\max _{\pi} V_{\pi}(s), \quad \forall s \in \mathcal{S} .
$$

最优优势函数 (optimal advantage function) 的定义是：

$$
D_{\star}(s, a) \triangleq Q_{\star}(s, a)-V_{\star}(s) .
$$

通过数学推导, 可以证明下面的定理：

## 定理 $6.1$

$$
Q_{\star}(s, a)=V_{\star}(s)+D_{\star}(s, a)-\underbrace{\max _{a \in \mathcal{A}} D_{\star}(s, a)}_{\text {恒等于零 }}, \quad \forall s \in \mathcal{S}, a \in \mathcal{A} .
$$

### 2 对决网络

与 DQN 一样, 对决网络 (dueling network) 也是对最优动作价值函数 $Q_{\star}$ 的近似。对 决网络与 DQN 的区别在于神经网络结构不同。直观上, 对决网络可以了解到哪些状态 有价值或者没价值, 而无需了解每个动作对每个状态的影响。实践中, 对决网络具有更 好的效果。由于对决网络与 DQN 都是对 $Q_{\star}$ 的近似, 可以用完全相同的算法训练两种神 经网络。

对决网络由两个神经网络组成。一个神经网络记作 $D\left(s, a ; \boldsymbol{w}^{D}\right)$, 它是对最优优势函 数 $D_{\star}(s, a)$ 的近似。另一个神经网络记作 $V\left(s ; \boldsymbol{w}^{V}\right)$, 它是对最优状态价值函数 $V_{\star}(s)$ 的 近似。把定理 $6.1$ 中的 $D_{\star}$ 和 $V_{\star}$ 替换成相应的神经网络, 那么最优动作价值函数 $Q_{\star}$ 就 被近似成下面的神经网络:

$$
Q(s, a ; \boldsymbol{w}) \triangleq V\left(s ; \boldsymbol{w}^{V}\right)+D\left(s, a ; \boldsymbol{w}^{D}\right)-\max _{a \in \mathcal{A}} D\left(s, a ; \boldsymbol{w}^{D}\right)
$$

公式左边的 $Q(s, a ; \boldsymbol{w})$ 就是对决网络, 它是对最优动作价值函数 $Q_{\star}$ 的近似。它的参数记 作 $\boldsymbol{w} \triangleq\left(\boldsymbol{w}^{V} ; \boldsymbol{w}^{D}\right)$ 。

对决网络的结构如图 $6.6$ 所示。可以让两个神经网络 $D\left(s, a ; \boldsymbol{w}^{D}\right)$ 与 $V\left(s ; \boldsymbol{w}^{V}\right)$ 共享 部分卷积层; 这些卷积层把输入的状态 $s$ 映射成特征向量, 特征向量是“优势头”与“状态 价值头”的输入。优势头输出一个向量, 向量的维度是动作空间的大小 $|\mathcal{A}|$, 向量每个元 素对应一个动作。举个例子, 动作空间是 $\mathcal{A}=\{$ 左, 右, 上 $\}$, 优势头的输出是三个值:

$$
D\left(s, \text { 左; } \boldsymbol{w}^{D}\right)=-90, \quad D\left(s, \text { 右 } ; \boldsymbol{w}^{D}\right)=-420, \quad D\left(s, \text { 上; } \boldsymbol{w}^{D}\right)=30 \text {. }
$$

状态价值头输出的是一个实数, 比如

$$
V\left(s ; \boldsymbol{w}^{V}\right)=300 .
$$

首先计算

$$
\max _{a} D\left(s, a ; \boldsymbol{w}^{D}\right)=\max \{-90,-420,30\}=30 .
$$

然后用公式 (6.1) 计算出:

$$
Q(s, \text { 左; } \boldsymbol{w})=180, \quad Q(s, \text { 右 } ; \boldsymbol{w})=-150, \quad Q(s, \text { 上; } \boldsymbol{w})=300 \text {. }
$$

这样就得到了对决网络的最终输出。

![](https://cdn.mathpix.com/cropped/2023_02_03_f46f5cf0e4de5b9996dcg-101.jpg?height=537&width=1454&top_left_y=1539&top_left_x=241)

图 6.6: 对决网络的结构。输入是状态 $s$; 红色的向量是每个动作的优势值; 蓝色的标量是状态价 值; 最终输出的紫色向量是每个动作的动作价值。

### 3 解决不唯一性

读者可能会有下面的疑问。对决网络是由定理 $6.1$ 推导出的, 而定理中最右的一项 恒等于零:

$$
\max _{a \in \mathcal{A}} D_{\star}(s, a)=0, \quad \forall s \in \mathcal{S} .
$$

也就是说, 可以把最优动作价值写成两种等价形式：

$$
\begin{array}{rlr}
Q_{\star}(s, a) & =V_{\star}(s)+D_{\star}(s, a) & \text { (第一种形式) } \\
& =V_{\star}(s)+D_{\star}(s, a)-\max _{a \in \mathcal{A}} D_{\star}(s, a) . & \text { (第二种形式) }
\end{array}
$$

之前我们根据第二种形式实现对决网络。我们可否根据第一种形式, 把对决网络按照下 面的方式实现呢：

$$
Q(s, a ; \boldsymbol{w})=V\left(s ; \boldsymbol{w}^{V}\right)+D\left(s, a ; \boldsymbol{w}^{D}\right) ?
$$

答案是不可以这样实现对决网络, 因为这样会导致不唯一性。假如这样实现对决网络, 那 么 $V$ 和 $D$ 可以随意上下波动, 比如一个增大 100 , 另一个减小 100 :

$$
\begin{aligned}
V\left(s ; \tilde{\boldsymbol{w}}^{V}\right) & \triangleq V\left(s ; \boldsymbol{w}^{V}\right)+100, \\
D\left(s, a ; \tilde{\boldsymbol{w}}^{D}\right) & \triangleq D\left(s, a ; \boldsymbol{w}^{D}\right)-100 .
\end{aligned}
$$

这样的上下波动不影响最终的输出:

$$
V\left(s ; \boldsymbol{w}^{V}\right)+D\left(s, a ; \boldsymbol{w}^{D}\right)=V\left(s ; \tilde{\boldsymbol{w}}^{V}\right)+D\left(s, a ; \tilde{\boldsymbol{w}}^{D}\right) .
$$

这就意味着 $V$ 和 $D$ 的参数可以很随意地变化, 却不会影响输出的 $\mathrm{Q}$ 。我们不希望这种情 况出现, 因为这会导致训练的过程中参数不稳定。

因此很有必要在对决网络中加入 $\max _{a \in \mathcal{A}} D\left(s, a ; \boldsymbol{w}^{D}\right)$ 这一项。它使得 $V$ 和 $D$ 不能 随意上下波动。假如让 $V$ 变大 100 , 让 $D$ 变小 100 , 则对决网络的输出会增大 100 , 而 非不变：

$$
\begin{aligned}
& V\left(s ; \tilde{\boldsymbol{w}}^{V}\right)+D\left(s, a ; \tilde{\boldsymbol{w}}^{D}\right)-\max _{a} D\left(s, a ; \tilde{\boldsymbol{w}}^{D}\right) \\
= & V\left(s ; \boldsymbol{w}^{V}\right)+D\left(s, a ; \boldsymbol{w}^{D}\right)-\max _{a} D\left(s, a ; \boldsymbol{w}^{D}\right)+100 .
\end{aligned}
$$

以上讨论说明了为什么 $\max _{a \in \mathcal{A}} D\left(s, a ; \boldsymbol{w}^{D}\right)$ 这一项不能省略。

### 4 对决网络的实际实现

按照定理 $6.1$, 对决网络应该定义成：

$$
Q(s, a ; \boldsymbol{w}) \triangleq V\left(s ; \boldsymbol{w}^{V}\right)+D\left(s, a ; \boldsymbol{w}^{D}\right)-\max _{a \in \mathcal{A}} D\left(s, a ; \boldsymbol{w}^{D}\right) .
$$

最右边的 max 项的目的是解决不唯一性。实际实现的时候, 用 mean 代替 max 会有更好 的效果。所以实际上会这样定义对决网络：

$$
Q(s, a ; \boldsymbol{w}) \triangleq V\left(s ; \boldsymbol{w}^{V}\right)+D\left(s, a ; \boldsymbol{w}^{D}\right)-\operatorname{mean}_{a \in \mathcal{A}} D\left(s, a ; \boldsymbol{w}^{D}\right)
$$

对决网络与 $\mathrm{DQN}$ 都是对最优动作价值函数 $Q_{\star}$ 的近似, 所以对决网络的训练和决策 与 DQN 完全一样。比如可以这样训练对决网络：

- 用 $\epsilon$-greedy 算法控制智能体, 收集经验, 把 $\left(s_{j}, a_{j}, r_{j}, s_{j+1}\right)$ 这样的四元组存入经验 回放数组。

- 从数组里随机抽取四元组, 用双 $\mathrm{Q}$ 学习算法更新对决网络参数 $\boldsymbol{w}=\left(\boldsymbol{w}^{D}, \boldsymbol{w}^{V}\right)$ 。 完成训练之后, 基于当前状态 $s_{t}$, 让对决网络给所有动作打分, 然后选择分数最高的动 作：

$$
a_{t}=\underset{a \in \mathcal{A}}{\operatorname{argmax}} Q\left(s_{t}, a ; \boldsymbol{w}\right) .
$$

简而言之, 怎么样训练 DQN, 就怎么样训练对决网络; 怎么样用 DQN 做控制, 就怎么

样用对决网络做控制。如果一个技巧能改进 DQN 的训练, 这个技巧也能改进对决网络。 同样的道理, 因为 $\mathrm{Q}$ 学习算法导致 $\mathrm{DQN}$ 出现高估, 所以 $\mathrm{Q}$ 学习算法也会导致对决网络 出现高估。

## $6.4$ 噪声网络

本节介绍噪声网络 (noisy net), 这是一种非常简单的方法, 可以显著提高 DQN 的 表现。噪声网络的应用不局限于 DQN, 它可以用于几乎所有的深度强化学习方法。

### 1 噪声网络的原理

把神经网络中的参数 $\boldsymbol{w}$ 替换成 $\boldsymbol{\mu}+\boldsymbol{\sigma} \circ \boldsymbol{\xi}$ 。此处的 $\boldsymbol{\mu} 、 \boldsymbol{\sigma} 、 \boldsymbol{\xi}$ 的形状与 $\boldsymbol{w}$ 完全相同。 $\mu 、 \sigma$ 分别表示均值和标准 差, 它们是神经网络的参数, 需要从 经验中学习。 $\boldsymbol{\xi}$ 是随机噪声, 它的每个 元素独立从标准正态分布 $\mathcal{N}(0,1)$ 中 随机抽取。符号“。”表示逐项乘积。如

![](https://cdn.mathpix.com/cropped/2023_02_03_f46f5cf0e4de5b9996dcg-104.jpg?height=325&width=668&top_left_y=754&top_left_x=1074)

图 6.7: 这个例子中, $w 、 \mu 、 \sigma 、 \xi$ 是形状相同的向量。 果 $\boldsymbol{w}$ 是向量, 那么有

$$
w_{i}=\mu_{i}+\sigma_{i} \cdot \xi_{i} .
$$

如果 $w$ 是矩阵，那么有

$$
w_{i j}=\mu_{i j}+\sigma_{i j} \cdot \xi_{i j} .
$$

噪声网络的意思是参数 $\boldsymbol{w}$ 的每个元素 $w_{i}$ 从均值为 $\mu_{i}$ 、标准差为 $\sigma_{i}$ 的正态分布中抽取。 举个例子, 某一个全连接层记作：

$$
\boldsymbol{z}=\operatorname{ReLU}(\boldsymbol{W} \boldsymbol{x}+\boldsymbol{b}) .
$$

公式中的向量 $\boldsymbol{x}$ 是输入, 矩阵 $\boldsymbol{W}$ 和向量 $\boldsymbol{b}$ 是参数, ReLU 是激活函数, $\boldsymbol{z}$ 是这一层的输 出。噪声网络把这个全连接层替换成:

$$
\boldsymbol{z}=\operatorname{ReLU}\left(\left(\boldsymbol{W}^{\mu}+\boldsymbol{W}^{\sigma} \circ \boldsymbol{W}^{\xi}\right) \boldsymbol{x}+\left(\boldsymbol{b}^{\mu}+\boldsymbol{b}^{\sigma} \circ \boldsymbol{b}^{\xi}\right)\right) .
$$

公式中的 $\boldsymbol{W}^{\mu} 、 \boldsymbol{W}^{\sigma} 、 \boldsymbol{b}^{\mu} 、 \boldsymbol{b}^{\sigma}$ 是参数, 需要从经验中学习。矩阵 $\boldsymbol{W}^{\xi}$ 和向量 $\boldsymbol{b}^{\xi}$ 的每个元 素都是独立从 $\mathcal{N}(0,1)$ 中随机抽取的, 表示噪声。

训练噪声网络的方法与训练标准的神经网络完全相同, 都是做反向传播计算梯度, 然 后用梯度更新神经参数。把损失函数记作 $L$ 。已知梯度 $\frac{\partial L}{\partial z}$, 可以用链式法则算出损失关 于参数的梯度：

$$
\begin{aligned}
& \frac{\partial L}{\partial \boldsymbol{W}^{\mu}}=\frac{\partial \boldsymbol{z}}{\partial \boldsymbol{W}^{\mu}} \cdot \frac{\partial L}{\partial \boldsymbol{z}}, \quad \frac{\partial L}{\partial \boldsymbol{b}^{\mu}}=\frac{\partial \boldsymbol{z}}{\partial \boldsymbol{b}^{\mu}} \cdot \frac{\partial L}{\partial \boldsymbol{z}}, \\
& \frac{\partial L}{\partial \boldsymbol{W}^{\sigma}}=\frac{\partial \boldsymbol{z}}{\partial \boldsymbol{W}^{\sigma}} \cdot \frac{\partial L}{\partial \boldsymbol{z}}, \quad \frac{\partial L}{\partial \boldsymbol{b}^{\sigma}}=\frac{\partial \boldsymbol{z}}{\partial \boldsymbol{b}^{\sigma}} \cdot \frac{\partial L}{\partial \boldsymbol{z}} .
\end{aligned}
$$

然后可以做梯度下降更新参数 $\boldsymbol{W}^{\mu} 、 \boldsymbol{W}^{\sigma} 、 \boldsymbol{b}^{\mu} 、 \boldsymbol{b}^{\sigma}$ 。

### 2 噪声 DQN

噪声网络可以用于 DQN。标准的 DQN 记作 $Q(s, a ; \boldsymbol{w})$, 其中的 $\boldsymbol{w}$ 表示参数。把 $\boldsymbol{w}$ 替换成 $\boldsymbol{\mu}+\boldsymbol{\sigma} \circ \boldsymbol{\xi}$, 得到噪声 DQN, 记作：

$$
\widetilde{Q}(s, a, \boldsymbol{\xi} ; \boldsymbol{\mu}, \boldsymbol{\sigma}) \triangleq Q(s, a ; \boldsymbol{\mu}+\boldsymbol{\sigma} \circ \boldsymbol{\xi}) .
$$

其中的 $\boldsymbol{\mu}$ 和 $\boldsymbol{\sigma}$ 是参数, 一开始随机初始化, 然后从经验中学习; 而 $\boldsymbol{\xi}$ 则是随机生成, 每 个元素都从 $\mathcal{N}(0,1)$ 中抽取。噪声 $\mathrm{DQN}$ 的参数数量比标准 DQN 多一倍。

收集经验：DQN 属于异策略 (off-policy)。我们用任意的行为策略 (behavior policy) 控制智能体, 收集经验, 事后做经验回放更新参数。在之前章节中, 我们用 $\epsilon$-greedy 作 为行为策略：

$$
a_{t}= \begin{cases}\operatorname{argmax}_{a \in \mathcal{A}} Q\left(s_{t}, a ; \boldsymbol{w}\right), & \text { 以概率 }(1-\epsilon) ; \\ \text { 均匀抽取 } \mathcal{A} \text { 中的一个动作, } & \text { 以概率 } \epsilon .\end{cases}
$$

$\epsilon$-greedy 策略带有一定的随机性, 可以让智能体尝试更多动作, 探索更多状态。

噪声 DQN 本身就带有随机性, 可以鼓励探索, 起到与 $\epsilon$-greedy 策略相同的作用。我 们直接用

$$
a_{t}=\underset{a \in \mathcal{A}}{\operatorname{argmax}} \widetilde{Q}(s, a, \boldsymbol{\xi} ; \boldsymbol{\mu}, \boldsymbol{\sigma})
$$

作为行为策略, 效果比 $\epsilon$-greedy 更好。每做一个决策, 要重新随机生成一个 $\boldsymbol{\xi}$ 。

$\mathbf{Q}$ 学习算法 : 训练的时候, 每一轮从经验回放数组中随机抽样出一个四元组, 记作 $\left(s_{j}, a_{j}, r_{j}, s_{j+1}\right)$ 。从标准正态分布中做抽样, 得到 $\boldsymbol{\xi}^{\prime}$ 的每一个元素。计算 $\mathrm{TD}$ 目标:

$$
\widehat{y}_{j}=r_{j}+\gamma \cdot \max _{a \in \mathcal{A}} \widetilde{Q}\left(s_{j+1}, a, \boldsymbol{\xi}^{\prime} ; \boldsymbol{\mu}, \boldsymbol{\sigma}\right) .
$$

把损失函数记作：

$$
L(\boldsymbol{\mu}, \boldsymbol{\sigma})=\frac{1}{2}\left[\widetilde{Q}\left(s_{j}, a_{j}, \boldsymbol{\xi} ; \boldsymbol{\mu}, \boldsymbol{\sigma}\right)-\widehat{y}_{j}\right]^{2},
$$

其中的 $\boldsymbol{\xi}$ 也是随机生成的噪声, 但是它与 $\boldsymbol{\xi}^{\prime}$ 不同。然后做梯度下降更新参数：

$$
\boldsymbol{\mu} \leftarrow \boldsymbol{\mu}-\alpha_{\mu} \cdot \nabla_{\boldsymbol{\mu}} L(\boldsymbol{\mu}, \boldsymbol{\sigma}), \quad \boldsymbol{\sigma} \leftarrow \boldsymbol{\sigma}-\alpha_{\sigma} \cdot \nabla_{\boldsymbol{\sigma}} L(\boldsymbol{\mu}, \boldsymbol{\sigma})
$$

公式中的 $\alpha_{\mu}$ 和 $\alpha_{\sigma}$ 是学习率。这样做梯度下降更新参数, 可以让损失函数减小, 让噪声 DQN 的预测更接近 TD 目标。

做决策 : 做完训练之后, 可以用噪声 DQN 做决策。做决策的时候不再需要噪声, 因 此可以把参数 $\boldsymbol{\sigma}$ 设置成全零, 只保留参数 $\boldsymbol{\mu}$ 。这样一来, 噪声 DQN 就变成标准的 DQN:

$$
\underbrace{\widetilde{Q}\left(s, a, \boldsymbol{\xi}^{\prime} ; \boldsymbol{\mu}, \mathbf{0}\right)}_{\text {噪声 DQN }}=\underbrace{Q(s, a ; \boldsymbol{\mu})}_{\text {标准 DQN }} .
$$

在训练的时候往 DQN 的参数中加入噪声, 不仅有利于探索, 还能增强鲁棒性。鲁棒性的 意思是即使参数被扰动, DQN 也能对动作价值 $Q_{\star}$ 做出可靠的估计。为什么噪声可以让 DQN 有更强的鲁棒性呢?

假设在训练的过程中不加入噪声。把学出的参数记作 $\boldsymbol{\mu}$ 。当参数严格等于 $\boldsymbol{\mu}$ 的时候, $\mathrm{DQN}$ 可以对最优动作价值做出较为准确的估计。但是对 $\boldsymbol{\mu}$ 做较小的扰动, 就可能会让 DQN 的输出偏离很远。所谓“失之毫厘, 谬以千里”。

噪声 DQN 训练的过程中, 参数带有噪声: $\boldsymbol{w}=\boldsymbol{\mu}+\boldsymbol{\sigma} \circ \boldsymbol{\xi}$ 。训练迫使 DQN 在参数带 噪声的情况下最小化 TD 误差, 也就是迫使 DQN 容忍对参数的扰动。训练出的 DQN 具 有鲁棒性: 参数不严格等于 $\boldsymbol{\mu}$ 也没关系, 只要参数在 $\boldsymbol{\mu}$ 的邻域内, DQN 做出的预测都 应该比较合理。用噪声 DQN, 不会出现“失之毫厘, 谬以千里”。

### 3 训练流程

实际编程实现 DQN 的时候, 应该将本章的四种技巧一优先经验回放、双 $\mathrm{Q}$ 学习、 对决网络、噪声 DQN-一全部用到。应该用对决网络的神经网络结构, 而不是简单的 $\mathrm{DQN}$ 结构。往对决网络中的参数 $\boldsymbol{w}$ 中加入噪声, 得到噪声 $\mathrm{DQN}$, 记作 $\widetilde{Q}(s, a, \boldsymbol{\xi} ; \boldsymbol{\mu}, \boldsymbol{\sigma})$ 。 训练要用双 $\mathrm{Q}$ 学习、优先经验回放, 而不是原始的 $\mathrm{Q}$ 学习。双 $\mathrm{Q}$ 学习需要目标网络 $\widetilde{Q}\left(s, a, \boldsymbol{\xi} ; \boldsymbol{\mu}^{-}, \boldsymbol{\sigma}^{-}\right)$计算 TD 目标。它跟噪声 DQN 的结构相同, 但是参数不同。

初始的时候, 随机初始化 $\mu 、 \sigma$, 并且把它们赋值给目标网络参数: $\mu^{-} \leftarrow \mu 、 \sigma^{-} \leftarrow \sigma_{\text {。 }}$ 然后重复下面的步骤更新参数。把当前的参数记作 $\mu_{\text {now }} 、 \sigma_{\text {now }}, \mu_{\text {now }}^{-} 、 \sigma_{\text {now }}^{-}$。

1. 用优先经验回放, 从数组中抽取一个四元组, 记作 $\left(s_{j}, a_{j}, r_{j}, s_{j+1}\right)$ 。

2. 用标准正态分布生成 $\boldsymbol{\xi}$ 。对噪声 DQN 做正向传播, 得到:

$$
\widehat{q}_{j}=\widetilde{Q}\left(s_{j}, a_{j}, \boldsymbol{\xi} ; \boldsymbol{\mu}_{\text {now }}, \boldsymbol{\sigma}_{\text {now }}\right) .
$$

3. 用噪声 $\mathrm{DQN}$ 选出最优动作:

$$
\tilde{a}_{j+1}=\underset{a \in \mathcal{A}}{\operatorname{argmax}} \widetilde{Q}\left(s_{j+1}, a, \boldsymbol{\xi} ; \boldsymbol{\mu}_{\text {now }}, \boldsymbol{\sigma}_{\text {now }}\right) .
$$

4. 用标准正态分布生成 $\boldsymbol{\xi}^{\prime}$ 。用目标网络计算价值：

$$
\widehat{q}_{j+1}=\widetilde{Q}\left(s_{j+1}, \tilde{a}_{j+1}, \boldsymbol{\xi}^{\prime} ; \boldsymbol{\mu}_{\text {now }}^{-}, \boldsymbol{\sigma}_{\text {now }}^{-}\right) .
$$

5. 计算 TD 目标和 TD 误差：

$$
\widehat{y}_{j}^{-}=r_{j}+\gamma \cdot \widehat{q}_{j+1}^{-} \quad \text { 和 } \quad \delta_{j}=\widehat{q}_{j}-\widehat{y}_{j}^{-} .
$$

6. 设 $\alpha_{\mu}$ 和 $\alpha_{\sigma}$ 为学习率。做梯度下降更新噪声 $\mathrm{DQN}$ 的参数：

$$
\begin{aligned}
& \boldsymbol{\mu}_{\text {new }} \leftarrow \boldsymbol{\mu}_{\text {now }}-\alpha_{\mu} \cdot \delta_{j} \cdot \nabla_{\boldsymbol{\mu}} \widetilde{Q}\left(s_{j}, a_{j}, \boldsymbol{\xi} ; \boldsymbol{\mu}_{\text {now }}, \boldsymbol{\sigma}_{\text {now }}\right), \\
& \boldsymbol{\sigma}_{\text {new }} \leftarrow \boldsymbol{\sigma}_{\text {now }}-\alpha_{\sigma} \cdot \delta_{j} \cdot \nabla_{\boldsymbol{\sigma}} \widetilde{Q}\left(s_{j}, a_{j}, \boldsymbol{\xi} ; \boldsymbol{\mu}_{\text {now }}, \boldsymbol{\sigma}_{\text {now }}\right) .
\end{aligned}
$$

7. 设 $\tau \in(0,1)$ 是需要手动调整的超参数。做加权平均更新目标网络的参数:

$$
\begin{aligned}
& \boldsymbol{\mu}_{\text {new }}^{-} \leftarrow \tau \cdot \boldsymbol{\mu}_{\text {new }}+(1-\tau) \cdot \boldsymbol{\mu}_{\text {now }}^{-}, \\
& \boldsymbol{\sigma}_{\text {new }}^{-} \leftarrow \tau \cdot \boldsymbol{\sigma}_{\text {new }}+(1-\tau) \cdot \boldsymbol{\sigma}_{\text {now }}^{-} .
\end{aligned}
$$



## 第 6 章 知识点

- 经验回放可以用于异策略算法。经验回放有两个好处：打破相邻两条经验的相关性、 重复利用收集的经验。

- 优先经验回放是对经验回放的一种改进。在做经验回放的时候, 从经验回放数组中 做加权随机抽样, TD 误差的绝对值大的经验被赋予较大的抽样概率、较小的学习 率。

- Q 学习算法会造成 DQN 高估真实的价值。高估的原因有两个：第一, 最大化造成 TD 目标高估真实价值; 第二，自举导致高估传播。高估并不是由 DQN 本身的缺陷 造成的, 而是由于 $\mathrm{Q}$ 学习算法不够好。双 $\mathrm{Q}$ 学习是对 $\mathrm{Q}$ 学习算法的改进, 可以有 效缓解高估。

- 对决网络与 DQN 一样, 都是对最优动作价值函数 $Q_{\star}$ 的近似; 两者的唯一区别在 于神经网络结构。对决网络由两部分组成： $D\left(s, a ; \boldsymbol{w}^{D}\right)$ 是对最优优势函数的近似, $V\left(s ; \boldsymbol{w}^{V}\right)$ 是对最优状态价值函数的近似。对决网络的训练与 DQN 完全相同。

- 噪声网络是一种特殊的神经网络结构, 神经网络中的参数带有随机噪声。噪声网络 可以用于 DQN 等多种深度强化学习模型。噪声网络中的噪声可以鼓励探索, 让智 能体尝试不同的动作, 这有利于学到更好的策略。

## 第 6 章 相关文献

训练 DQN 用到的经验回放是由 Lin 在 1993 年的博士论文 ${ }^{[68]}$ 中提出的。优先经 验回放是由 Schaul 等人 2015 年的论文 ${ }^{[93]}$ 提出。目标网络由 Mnih 等人 2015 年的论文 (77) 提出。双 Q 学习由 van Hasselt 2010 年的论文 ${ }^{[115]}$ 提出。双 $\mathrm{Q}$ 学习与 $\mathrm{DQN}$ 的结合 被称为 Double DQN, 由 van Hasselt 等人 2010 年的论文提出 ${ }^{[116]}$ 。对决网络在 Wang 等 人 2016 年的论文中提出 ${ }^{[121]}$ 。噪声网络在 Fortunato 等人 2018 年的论文中提出 ${ }^{[40]}$ 。

Hessel 等人在 2018 年发表的论文 ${ }^{[48]}$ 将优先经验回放、双 $\mathrm{Q}$ 学习、对决网络、噪 声网络、多步 $\mathrm{TD}$ 目标等方法结合, 改进 DQN, 把组合称为 Rainbow。有充分的实验证 实这些高级技巧的有效性。此外, Rainbow 还用到了 distributional learning ${ }^{[11]}$, 这种技 巧也非常有用。

## 第 6 章 习题

1. 设置经验回放数组的大小是 $10^{6}$ 。当数组中存满 $10^{6}$ 条四元组的时候, 应该怎么办?
A. 删除数组中最旧的四元组。
B. 删除数组中最新的四元组。
C. 终止程序, 停止强化学习。
D. 继续运行程序, 但是不再收集经验。

2. 我们设置 $\mathrm{Q}$ 学习的批大小 (batch size) 为 128 , 设置经验回放数组的大小是 $10^{6}$ 。初 始的时候数组是空的。当数组中四元组的数量超过某个阈值 $k$ 的时候, 开始做经验 回放更新 $\mathrm{DQN}$ 。请问 $k$ 该如何设置?
A. 设置 $k$ 等于 128 或略大于 128 。
B. 设置 $k$ 远大于 128 但是小于 $10^{6}$ 。
C. 设置 $k$ 等于 $10^{6}$ 。

3. 在优先经验回放中, 需要对四元组 $\left\{\left(s, a, r, s^{\prime}\right)\right\}$ 做非均匀抽样。请问抽样的概率取 决于什么?
A. TD 目标的绝对值 $|\widehat{y}|$ 。
B. TD 目标的绝对值 $|\delta|$ 。
C. Q 值的绝对值 $|Q(s, a ; \boldsymbol{w})|$ 。
D. 奖励的绝对值 $|r|$ 。

4. 在优先经验回放中,如果一个四元组被抽样的概率较大,那么它的学习率
A. 较大
B. 较小

5. DQN 的高估问题是由 引起的。
A. DQN 网络结构的缺陷
B. $\mathrm{Q}$ 学习算法的缺陷
C. A 和 $\mathrm{B}$ 都对

6. 训练 DQN 的时候使用目标网络 (target network) 的好处在于缓解
A. 自举造成的偏差
B. 最大化造成的高估
C. A 和 $\mathrm{B}$ 都对

7. 比起 $\mathrm{Q}$ 学习算法, 双 $\mathrm{Q}$ 学习的优势在于缓解
A. 自举造成的偏差
B. 最大化造成的高估
C. A 和 $\mathrm{B}$ 都对

8. 最优优势函数 (optimal advantage function) 的定义是:

$$
D_{\star}(s, a) \triangleq Q_{\star}(s, a)-V_{\star}(s) .
$$

$D_{\star}$ 的均值 $\operatorname{mean}_{a \in \mathcal{A}} D_{\star}(s, a)$ 。 $D_{\star}$ 的最大值 $\max _{a \in \mathcal{A}} D_{\star}(s, a)$ 请从下面选出最合适的选项填空:
A. 大于等于零
B. 小于等于零
C. 严格等于零

9. 为什么对决网络的表现优于简单的 DQN?
A. 对决网络的神经网络结构更好。
B. 训练对决网络的算法更好。
C. A 和 $\mathrm{B}$ 都对。

10. Rainbow 方法 ${ }^{[48]}$ 是对 的改进。
A. Deep Q Network (DQN)
B. SARSA
C. Policy Gradient
D. Deep Deterministic Policy Gradient (DDPG)

11.（多选）Rainbow 方法 ${ }^{[48]}$ 使用了下面哪些技巧?
A. 优先经验回放
B. 双 $\mathrm{Q}$ 学习
C. 对决网络
D. 噪声网络
E. 多步 $\mathrm{TD}$ 目标
F. 蒙特卡洛树搜索 (Monte Carlo Tree Search)
G. 注意力机制 (Attention)
H. Advantage Actor-Critic (A2C)
I. Twin Delayed DDPG (TD3)
J. Distributional Learning
