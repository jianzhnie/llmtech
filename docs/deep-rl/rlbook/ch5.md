

## 第 5 章 SARSA 算法

上一章介绍了 $\mathrm{Q}$ 学习的表格形式和神经网络形式 (即 $\mathrm{DQN}$ )。TD 算法是一大类算法 的总称。上一章用的 $\mathrm{Q}$ 学习是一种 $\mathrm{TD}$ 算法, $\mathrm{Q}$ 学习的目的是学习最优动作价值函数 $Q_{\star}$ 。 本章介绍 SARSA, 它也是一种 TD 算法, SARSA 的目的是学习动作价值函数 $Q_{\pi}(s, a)$ 。

虽然传统的强化学习用 $Q_{\pi}$ 作为确定性的策略控制智能体, 但是现在 $Q_{\pi}$ 通常被用于 评价策略的好坏, 而非用于控制智能体。 $Q_{\pi}$ 常与策略函数 $\pi$ 结合使用, 被称作 actor-critic （演员一评委）方法。策略函数 $\pi$ 控制智能体, 因此被看做 “演员”; 而 $Q_{\pi}$ 评价 $\pi$ 的表现, 帮助改进 $\pi$, 因此 $Q_{\pi}$ 被看做 “评委”。Actor-critic 通常用 SARSA 训练“评委” $Q_{\pi}$ 。在后面 策略学习的章节会详细介绍 actor-critic 方法。

## $5.1$ 表格形式的 SARSA

假设状态空间 $\mathcal{S}$ 和动作空间 $\mathcal{A}$ 都是有限集, 即集合中元素数 量有限。比如, $\mathcal{S}$ 中一共有 3 种 状态, $\mathcal{A}$ 中一共有 4 种动作。那 么动作价值函数 $Q_{\pi}(s, a)$ 可以表 示为一个 $3 \times 4$ 的表格, 比如右边 的表格。该表格与一个策略函数 $\pi(a \mid s)$ 相关联; 如果 $\pi$ 发生变化,

![](https://cdn.mathpix.com/cropped/2023_02_03_f46f5cf0e4de5b9996dcg-075.jpg?height=326&width=819&top_left_y=1162&top_left_x=847)

图 5.1: 动作价值函数 $Q_{\pi}$ 表示成表格形式。 表格 $Q_{\pi}$ 也会发生变化。

我们用表格 $q$ 近似 $Q_{\pi}$ 。该如何通过智能体与环境的交互来学习表格 $q$ 呢? 首先初始 化 $q$, 可以让它是全零的表格。然后用表格形式的 SARSA 算法更新 $q$, 每次更新表格的 一个元素。最终 $q$ 收敛到 $Q_{\pi}$ 。

推导表格形式的 SARSA 学习算法：SARSA 算法由下面的贝尔曼方程推导出：

$$
Q_{\pi}\left(s_{t}, a_{t}\right)=\mathbb{E}_{S_{t+1}, A_{t+1}}\left[R_{t}+\gamma \cdot Q_{\pi}\left(S_{t+1}, A_{t+1}\right) \mid S_{t}=s_{t}, A_{t}=a_{t}\right]
$$

贝尔曼方程的证明见附录 $\mathrm{A}$ 。我们对贝尔曼方程左右两边做近似:

- 方程左边的 $Q_{\pi}\left(s_{t}, a_{t}\right)$ 可以近似成 $q\left(s_{t}, a_{t}\right)$ 。 $q\left(s_{t}, a_{t}\right)$ 是表格在 $t$ 时刻对 $Q_{\pi}\left(s_{t}, a_{t}\right)$ 做出的估计。

- 方程右边的期望是关于下一时刻状态 $S_{t+1}$ 和动作 $A_{t+1}$ 求的。给定当前状态 $s_{t}$, 智 能体执行动作 $a_{t}$, 环境会给出奖励 $r_{t}$ 和新的状态 $s_{t+1}$ 。然后基于 $s_{t+1}$ 做随机抽样, 得到新的动作

$$
\tilde{a}_{t+1} \sim \pi\left(\cdot \mid s_{t+1}\right) .
$$

用观测到的 $r_{t} 、 s_{t+1}$ 和计算出的 $\tilde{a}_{t+1}$ 对期望做蒙特卡洛近似, 得到:

$$
r_{t}+\gamma \cdot Q_{\pi}\left(s_{t+1}, \tilde{a}_{t+1}\right) .
$$

- 进一步把公式 (5.1) 中的 $Q_{\pi}$ 近似成 $q$, 得到

$$
\widehat{y}_{t} \triangleq r_{t}+\gamma \cdot q\left(s_{t+1}, \tilde{a}_{t+1}\right) .
$$

把它称作 TD 目标。它是表格在 $t+1$ 时刻对 $Q_{\pi}\left(s_{t}, a_{t}\right)$ 做出的估计。

$q\left(s_{t}, a_{t}\right)$ 和 $\widehat{y}_{t}$ 都是对动作价值 $Q_{\pi}\left(s_{t}, a_{t}\right)$ 的估计。由于 $\widehat{y}_{t}$ 部分基于真实观测到的奖励 $r_{t}$, 我们认为 $\widehat{y}_{t}$ 是更可靠的估计, 所以鼓励 $q\left(s_{t}, a_{t}\right)$ 趋近 $\widehat{y}_{t}$ 。更新表格 $\left(s_{t}, a_{t}\right)$ 位置上的元 素:

$$
q\left(s_{t}, a_{t}\right) \leftarrow(1-\alpha) \cdot q\left(s_{t}, a_{t}\right)+\alpha \cdot \widehat{y}_{t} .
$$

这样可以使得 $q\left(s_{t}, a_{t}\right)$ 更接近 $\widehat{y}_{t}$ 。SARSA 是 State-Action-Reward-State-Action 的缩写, 原 因是 SARSA 算法用到了这个五元组: $\left(s_{t}, a_{t}, r_{t}, s_{t+1}, \tilde{a}_{t+1}\right)$ 。SARSA 算法学到的 $q$ 依赖于 策略 $\pi$, 这是因为五元组中的 $\tilde{a}_{t+1}$ 是根据 $\pi\left(\cdot \mid s_{t+1}\right)$ 抽样得到的。

训练流程 : 设当前表格为 $q_{\text {now }}$, 当前策略为 $\pi_{\text {now }}$ 每一轮更新表格中的一个元素, 把更新之后的表格记作 $q_{\text {new }}$ 。

1. 观测到当前状态 $s_{t}$, 根据当前策略做抽样： $a_{t} \sim \pi_{\text {now }}\left(\cdot \mid s_{t}\right)$ 。

2. 把表格 $q_{\text {now }}$ 中第 $\left(s_{t}, a_{t}\right)$ 位置上的元素记作:

$$
\widehat{q}_{t}=q_{\text {now }}\left(s_{t}, a_{t}\right) .
$$

3. 智能体执行动作 $a_{t}$ 之后, 观测到奖励 $r_{t}$ 和新的状态 $s_{t+1}$ 。

4. 根据当前策略做抽样: $\tilde{a}_{t+1} \sim \pi_{\text {now }}\left(\cdot \mid s_{t+1}\right)$ 。注意, $\tilde{a}_{t+1}$ 只是假想的动作, 智能体 不予执行。

5. 把表格 $q_{\text {now }}$ 中第 $\left(s_{t+1}, \tilde{a}_{t+1}\right)$ 位置上的元素记作:

$$
\widehat{q}_{t+1}=q_{\text {now }}\left(s_{t+1}, \tilde{a}_{t+1}\right) .
$$

6. 计算 $\mathrm{TD}$ 目标和 $\mathrm{TD}$ 误差:

$$
\widehat{y}_{t}=r_{t}+\gamma \cdot \widehat{q}_{t+1}, \quad \delta_{t}=\widehat{q}_{t}-\widehat{y}_{t} .
$$

7. 更新表格中 $\left(s_{t}, a_{t}\right)$ 位置上的元素:

$$
q_{\text {new }}\left(s_{t}, a_{t}\right) \leftarrow q_{\text {now }}\left(s_{t}, a_{t}\right)-\alpha \cdot \delta_{t} .
$$

8. 用某种算法更新策略函数。该算法与 SARSA 算法无关。

$\mathbf{Q}$ 学习与 SARSA 的对比： $\mathrm{Q}$ 学习不依赖于 $\pi$, 因此 $\mathrm{Q}$ 学习属于异策略 (off-policy), 可以用经验回放。而 SARSA 依赖于 $\pi$, 因此 SARSA 属于同策略 (on-policy), 不能用经 验回放。两种算法的对比如图 $5.2$ 所示。

$\mathrm{Q}$ 学习的目标是学到表格 $\tilde{Q}$, 作为最优动作价值函数 $Q_{\star}$ 的近似。因为 $Q_{\star}$ 与 $\pi$ 无 关, 所以在理想情况下, 不论收集经验用的行为策略 $\pi$ 是什么, 都不影响 $\mathrm{Q}$ 学习得到的 最优动作价值函数。因此, $\mathrm{Q}$ 学习属于异策略 (off-policy), 允许行为策略区别于目标策 略。Q 学习允许使用经验回放, 可以重复利用过时的经验。

SARSA 算法的目标是学到表格 $q$, 作为动作价值函数 $Q_{\pi}$ 的近似。 $Q_{\pi}$ 与一个策略 $\pi$ 相对应, 用不同的策略 $\pi$, 对应 $Q_{\pi}$ 就会不同。策略 $\pi$ 越好, $Q_{\pi}$ 的值越大。经验回放数 组里的经验 $\left(s_{j}, a_{j}, r_{j}, s_{j+1}\right)$ 是过时的行为策略 $\pi_{\text {old }}$ 收集到的, 与当前策略 $\pi_{\text {now }}$ 及其对 应的价值 $Q_{\pi_{\text {now }}}$ 对应不上。想要学习 $Q_{\pi}$ 的话, 必须要用与当前策略 $\pi_{\text {now }}$ 收集到的经验, 而不能用过时的 $\pi_{\text {old }}$ 收集到的经验。这就是为什么 SARSA 不能用经验回放的原因。

\begin{tabular}{|c|c|c|c|}
\hline $\mathrm{Q}$ 学习 & 近似 $Q_{\star}$ & 异策略 & $\begin{gathered}\text { 可以使用 } \\
\text { 经验回放 }\end{gathered}$ \\
\hline SARSA & 近似 $Q_{\pi}$ & 同策略 & $\begin{gathered}\text { 不能使用 } \\
\text { 经验回放 }\end{gathered}$ \\
\hline
\end{tabular}

图 5.2: Q 学习与 SARSA 的对比。

## $5.2$ 神经网络形式的 SARSA

价值网络：如果状态空间 $\mathcal{S}$ 是无限集, 那么我们无法用一张表格表示 $Q_{\pi}$, 否则表 格的行数是无穷。一种可行的方案是用一个神经网络 $q(s, a ; \boldsymbol{w})$ 来近似 $Q_{\pi}(s, a)$; 理想情 况下，

$$
q(s, a ; \boldsymbol{w})=Q_{\pi}(s, a), \quad \forall s \in \mathcal{S}, a \in \mathcal{A} .
$$

神经网络 $q(s, a ; \boldsymbol{w})$ 被称为价值网络 (value network), 其中的 $\boldsymbol{w}$ 表示神经网络中可训练 的参数。神经网络的结构是人预先设定的 (比如有多少层, 每一层的宽度是多少), 而参 数 $w$ 需要通过智能体与环境的交互来学习。首先随机初始化 $w$, 然后用 SARSA 算法更 新 $\boldsymbol{w}$ 。

神经网络的结构见图 5.3。价值网络的输入是状态 $s$ 。如果 $s$ 是矩阵或张量 (tensor), 那么可以用卷积网络处理 $s$ (如图 5.3)。如果 $s$ 是向量, 那么可以用全连接层处理 $s$ 。价 值网络的输出是每个动作的价值。动作空间 $\mathcal{A}$ 中有多少种动作, 则价值网络的输出就是 多少维的向量, 向量每个元素对应一个动作。举个例子, 动作空间是 $\mathcal{A}=\{$ 左, 右, 上 $\}$, 价值网络的输出是

$$
\begin{aligned}
& q(s, \text { 左; } \boldsymbol{w})=219, \\
& q(s, \text { 右; } \boldsymbol{w})=-73, \\
& q(s, \text { 上; } \boldsymbol{w})=580 .
\end{aligned}
$$

![](https://cdn.mathpix.com/cropped/2023_02_03_f46f5cf0e4de5b9996dcg-078.jpg?height=272&width=1396&top_left_y=1526&top_left_x=387)

价值网络

图 5.3: 价值网络 $q(s, a ; \boldsymbol{w})$ 的结构。输入是状态 $s$, 输出是每个动作的价值。

算法推导: 给定当前状态 $s_{t}$, 智能体执行动作 $a_{t}$, 环境会给出奖励 $r_{t}$ 和新的状态 $s_{t+1}$ 。 然后基于 $s_{t+1}$ 做随机抽样, 得到新的动作 $\tilde{a}_{t+1} \sim \pi\left(\cdot \mid s_{t+1}\right)$ 。定义 TD 目标:

$$
\widehat{y}_{t} \triangleq r_{t}+\gamma \cdot q\left(s_{t+1}, \tilde{a}_{t+1} ; \boldsymbol{w}\right) .
$$

我们鼓励 $q\left(s_{t}, a_{t} ; \boldsymbol{w}\right)$ 接近 TD 目标 $\widehat{y}_{t}$, 所以定义损失函数:

$$
L(\boldsymbol{w}) \triangleq \frac{1}{2}\left[q\left(s_{t}, a_{t} ; \boldsymbol{w}\right)-\widehat{y}_{t}\right]^{2} .
$$

损失函数的变量是 $\boldsymbol{w}$, 而 $\widehat{y}_{t}$ 被视为常数。(尽管 $\widehat{y}_{t}$ 也依赖于参数 $\boldsymbol{w}$, 但这一点被忽略 掉。) 设 $\widehat{q}_{t}=q\left(s_{t}, a_{t} ; \boldsymbol{w}\right)$ 。损失函数关于 $\boldsymbol{w}$ 的梯度是:

$$
\nabla_{\boldsymbol{w}} L(\boldsymbol{w})=\underbrace{\left(\widehat{q}_{t}-\widehat{y}_{t}\right)}_{\mathrm{TD} \text { 误差 } \delta_{t}} \cdot \nabla_{\boldsymbol{w}} q\left(s_{t}, a_{t} ; \boldsymbol{w}\right) .
$$

做一次梯度下降更新 $\boldsymbol{w}$ :

$$
\boldsymbol{w} \leftarrow \boldsymbol{w}-\alpha \cdot \delta_{t} \cdot \nabla_{\boldsymbol{w}} q\left(s_{t}, a_{t} ; \boldsymbol{w}\right)
$$

这样可以使得 $q\left(s_{t}, a_{t} ; \boldsymbol{w}\right)$ 更接近 $\widehat{y}_{t}$ 。此处的 $\alpha$ 是学习率, 需要手动调整。

训练流程 : 设当前价值网络的参数为 $\boldsymbol{w}_{\text {now }}$, 当前策略为 $\pi_{\text {now }}$ 。每一轮训练用五元 组 $\left(s_{t}, a_{t}, r_{t}, s_{t+1}, \tilde{a}_{t+1}\right)$ 对价值网络参数做一次更新。

1. 观测到当前状态 $s_{t}$, 根据当前策略做抽样: $a_{t} \sim \pi_{\text {now }}\left(\cdot \mid s_{t}\right)$ 。

2. 用价值网络计算 $\left(s_{t}, a_{t}\right)$ 的价值:

$$
\widehat{q}_{t}=q\left(s_{t}, a_{t} ; \boldsymbol{w}_{\text {now }}\right) .
$$

3. 智能体执行动作 $a_{t}$ 之后, 观测到奖励 $r_{t}$ 和新的状态 $s_{t+1}$ 。

4. 根据当前策略做抽样: $\tilde{a}_{t+1} \sim \pi_{\text {now }}\left(\cdot \mid s_{t+1}\right)$ 。注意, $\tilde{a}_{t+1}$ 只是假想的动作, 智能体 不予执行。

5. 用价值网络计算 $\left(s_{t+1}, \tilde{a}_{t+1}\right)$ 的价值:

$$
\widehat{q}_{t+1}=q\left(s_{t+1}, \tilde{a}_{t+1} ; \boldsymbol{w}_{\text {now }}\right) .
$$

6. 计算 TD 目标和 TD 误差:

$$
\widehat{y}_{t}=r_{t}+\gamma \cdot \widehat{q}_{t+1}, \quad \delta_{t}=\widehat{q}_{t}-\widehat{y}_{t} .
$$

7. 对价值网络 $q$ 做反向传播, 计算 $q$ 关于 $\boldsymbol{w}$ 的梯度: $\nabla_{\boldsymbol{w}} q\left(s_{t}, a_{t} ; \boldsymbol{w}_{\text {now }}\right)$ 。

8. 更新价值网络参数:

$$
\boldsymbol{w}_{\text {new }} \leftarrow \boldsymbol{w}_{\text {now }}-\alpha \cdot \delta_{t} \cdot \nabla_{\boldsymbol{w}} q\left(s_{t}, a_{t} ; \boldsymbol{w}_{\text {now }}\right) .
$$

9. 用某种算法更新策略函数。该算法与 SARSA 算法无关。

## $5.3$ 多步 TD 目标

首先回顾一下 SARSA 算法。给定五元组 $\left(s_{t}, a_{t}, r_{t}, s_{t+1}, a_{t+1}\right)$, SARSA 计算 TD 目 标：

$$
\widehat{y}_{t}=r_{t}+\gamma \cdot q\left(s_{t+1}, a_{t+1} ; \boldsymbol{w}\right) .
$$

公式中只用到一个奖励 $r_{t}$, 这样得到的 $\widehat{y}_{t}$ 叫做单步 TD 目标。多步 TD 目标用 $m$ 个奖励, 可以视作单步 TD 目标的推广。下面我们推导多步 TD 目标。

数学推导: 设一局游戏的长度为 $n$ 。根据定义, $t$ 时刻的回报 $U_{t}$ 是 $t$ 时刻之后的所 有奖励的加权和：

$$
U_{t}=R_{t}+\gamma R_{t+1}+\gamma^{2} R_{t+2}+\cdots+\gamma^{n-t} R_{n} .
$$

同理, $t+m$ 时刻的回报可以写成:

$$
U_{t+m}=R_{t+m}+\gamma R_{t+m+1}+\gamma^{2} R_{t+m+2}+\cdots+\gamma^{n-t-m} R_{n} .
$$

下面我们推导两个回报的关系。把 $U_{t}$ 写成:

$$
\begin{aligned}
U_{t} & =\left(R_{t}+\gamma R_{t+1}+\cdots+\gamma^{m-1} R_{t+m-1}\right)+\left(\gamma^{m} R_{t+m}+\cdots+\gamma^{n-t} R_{n}\right) \\
& =\left(\sum_{i=0}^{m-1} \gamma^{i} R_{t+i}\right)+\gamma^{m} \underbrace{\left(R_{t+m}+\gamma R_{t+m+1}+\cdots+\gamma^{n-t-m} R_{n}\right)}_{\text {等于 } U_{t+m}} .
\end{aligned}
$$

因此, 回报可以写成这种形式:

$$
U_{t}=\left(\sum_{i=0}^{m-1} \gamma^{i} R_{t+i}\right)+\gamma^{m} U_{t+m} .
$$

动作价值函数 $Q_{\pi}\left(s_{t}, a_{t}\right)$ 是回报 $U_{t}$ 的期望, 而 $Q_{\pi}\left(s_{t+m}, a_{t+m}\right)$ 是回报 $U_{t+m}$ 的期望。利 用公式 (5.2), 再按照贝尔曼方程的证明（见附录 A), 不难得出下面的定理：

## 定理 $5.1$

设 $R_{k}$ 是 $S_{k} 、 A_{k} 、 S_{k+1}$ 的函数, $\forall k=1, \cdots, n$ 。那么

$$
\underbrace{Q_{\pi}\left(s_{t}, a_{t}\right)}_{U_{t} \text { 的期望 }}=\mathbb{E}[\left(\sum_{i=0}^{m-1} \gamma^{i} R_{t+i}\right)+\gamma^{m} \cdot \underbrace{Q_{\pi}\left(S_{t+m}, A_{t+m}\right)}_{U_{t+m} \text { 的期望 }} \mid S_{t}=s_{t}, A_{t}=a_{t}] \text {. }
$$

公式中的期望是关于随机变量 $S_{t+1}, A_{t+1}, \cdots, S_{t+m}, A_{t+m}$ 求的。

注 回报 $U_{t}$ 的随机性来自于 $t$ 到 $n$ 时刻的状态和动作:

$$
S_{t}, A_{t}, \quad S_{t+1}, A_{t+1}, \cdots, S_{t+m}, A_{t+m}, \quad S_{t+m+1}, A_{t+m+1}, \cdots, S_{n}, A_{n} .
$$

定理中把 $S_{t}=s_{t}$ 和 $A_{t}=a_{t}$ 看做是观测值, 用期望消掉 $S_{t+1}, A_{t+1}, \cdots, S_{t+m}, A_{t+m}$, 而 $Q_{\pi}\left(S_{t+m}, A_{t+m}\right)$ 则消掉了剩余的随机变量 $S_{t+m+1}, A_{t+m+1}, \cdots, S_{n}, A_{n}$ 。

多步 $\mathbf{T D}$ 目标 : 我们对定理 $5.1$ 中的期望做蒙特卡洛近似, 然后再用价值网络 $q(s, a ; \boldsymbol{w})$ 近似动作价值函数 $Q_{\pi}(s, a)$ 。具体做法如下:

。在 $t$ 时刻, 价值网络做出预测 $\widehat{q}_{t}=q\left(s_{t}, a_{t} ; \boldsymbol{w}\right)$, 它是对 $Q_{\pi}\left(s_{t}, a_{t}\right)$ 的估计。 - 已知当前状态 $s_{t}$, 用策略 $\pi$ 控制智能体与环境交互 $m$ 次, 得到轨迹

$$
r_{t}, \quad s_{t+1}, a_{t+1}, r_{t+1}, \quad \cdots, \quad s_{t+m-1}, a_{t+m-1}, r_{t+m-1}, s_{t+m}, a_{t+m} .
$$

在 $t+m$ 时刻, 用观测到的轨迹对定理 $5.1$ 中的期望做蒙特卡洛近似, 把近似的结 果记作:

$$
\left(\sum_{i=0}^{m-1} \gamma^{i} r_{t+i}\right)+\gamma^{m} \cdot Q_{\pi}\left(s_{t+m}, a_{t+m}\right) .
$$

- 进一步用 $q\left(s_{t+m}, a_{t+m} ; \boldsymbol{w}\right)$ 近似 $Q_{\pi}\left(s_{t+m}, a_{t+m}\right)$, 得到:

$$
\widehat{y}_{t} \triangleq\left(\sum_{i=0}^{m-1} \gamma^{i} r_{t+i}\right)+\gamma^{m} \cdot q\left(s_{t+m}, a_{t+m} ; \boldsymbol{w}\right) .
$$

把 $\widehat{y}_{t}$ 称作 $m$ 步 $\mathrm{TD}$ 目标。

$\widehat{q}_{t}=q\left(s_{t}, a_{t} ; \boldsymbol{w}\right)$ 和 $\widehat{y}_{t}$ 分别是价值网络在 $t$ 时刻和 $t+m$ 时刻做出的预测, 两者都是对 $Q_{\pi}\left(s_{t}, a_{t}\right)$ 的估计值。 $\widehat{q} t$ 是纯粹的预测, 而 $\widehat{y}_{t}$ 则基于 $m$ 组实际观测, 因此 $\widehat{y}_{t}$ 比 $\widehat{q}_{t}$ 更可 靠。我们鼓励 $\widehat{q}_{t}$ 接近 $\widehat{y}_{t}$ 。设损失函数为

$$
L(\boldsymbol{w}) \triangleq \frac{1}{2}\left[q\left(s_{t}, a_{t} ; \boldsymbol{w}\right)-\widehat{y}_{t}\right]^{2} .
$$

做一步梯度下降更新价值网络参数 $\boldsymbol{w}$ :

$$
\boldsymbol{w} \leftarrow \boldsymbol{w}-\alpha \cdot\left(\widehat{q}_{t}-\widehat{y}_{t}\right) \cdot \nabla_{\boldsymbol{w}} q\left(s_{t}, a_{t} ; \boldsymbol{w}\right) .
$$

训练流程 : 设当前价值网络的参数为 $w_{\text {now }}$, 当前策略为 $\pi_{\text {now }}$ 。执行以下步骤更新 价值网络和策略。

![](https://cdn.mathpix.com/cropped/2023_02_03_f46f5cf0e4de5b9996dcg-081.jpg?height=59&width=1179&top_left_y=1581&top_left_x=290)

$$
s_{1}, a_{1}, r_{1}, s_{2}, a_{2}, r_{2}, \cdots, s_{n}, a_{n}, r_{n} .
$$

2. 对于所有的 $t=1, \cdots, n-m$, 计算

$$
\widehat{q}_{t}=q\left(s_{t}, a_{t} ; \boldsymbol{w}_{\text {now }}\right) .
$$

3. 对于所有的 $t=1, \cdots, n-m$, 计算多步 TD 目标和 TD 误差:

$$
\widehat{y}_{t}=\sum_{i=0}^{m-1} \gamma^{i} r_{t+i}+\gamma^{m} \widehat{q}_{t+m}, \quad \delta_{t}=\widehat{q}_{t}-\widehat{y}_{t} .
$$

4. 对于所有的 $t=1, \cdots, n-m$, 对价值网络 $q$ 做反向传播, 计算 $q$ 关于 $\boldsymbol{w}$ 的梯度:

$$
\nabla_{\boldsymbol{w}} q\left(s_{t}, a_{t} ; \boldsymbol{w}_{\text {now }}\right) .
$$

5. 更新价值网络参数：

$$
\boldsymbol{w}_{\text {new }} \leftarrow \boldsymbol{w}_{\text {now }}-\alpha \cdot \sum_{t=1}^{n-m} \delta_{t} \cdot \nabla_{\boldsymbol{w}} q\left(s_{t}, a_{t} ; \boldsymbol{w}_{\text {now }}\right) .
$$

6. 用某种算法更新策略函数 $\pi$ 。该算法与 SARSA 算法无关。

## $5.4$ 蒙特卡洛与自举

上一节介绍了多步 TD 目标。单步 TD 目标、回报是多步 TD 目标的两种特例。如下 图所示, 如果设 $m=1$, 那么多步 $\mathrm{TD}$ 目标变成单步 TD 目标。如果设 $m=n-t+1$, 那 么多步 $\mathrm{TD}$ 目标变成实际观测的回报 $u_{t}$ 。

$$
\begin{aligned}
& \text { 单步 TD目标: } \\
& \hat{y}_{t}=r_{t}+\gamma \hat{q}_{t+1} \text {. } \\
& \longleftarrow m=1 \\
& m \text { 步TD目标: }
\end{aligned}
$$

$\stackrel{m=n-t+1}{\longrightarrow}$

观测到的回报:

(自举)

$\hat{y}_{t}=\sum_{i=0}^{m-1} \gamma^{i} r_{t+i}+\gamma^{m} \hat{q}_{t+m}$.

$u_{t}=\sum_{i=0}^{n-t} \gamma^{i} r_{t+i}$.

(蒙特卡洛)

![](https://cdn.mathpix.com/cropped/2023_02_03_f46f5cf0e4de5b9996dcg-082.jpg?height=88&width=231&top_left_y=630&top_left_x=1295)

图 5.4: 单步 $\mathrm{TD}$ 目标、多步 TD 目标、回报的关系。

### 1 蒙特卡洛

训练价值网络 $q(s, a ; \boldsymbol{w})$ 的时候, 我们可以将一局游戏进行到底, 观测到所有的奖励 $r_{1}, \cdots, r_{n}$, 然后计算回报 $u_{t}=\sum_{i=0}^{n-t} \gamma^{i} r_{t+i}$ 。拿 $u_{t}$ 作为目标, 鼓励价值网络 $q\left(s_{t}, a_{t} ; \boldsymbol{w}\right)$ 接近 $u_{t}$ 。定义损失函数:

$$
L(\boldsymbol{w})=\frac{1}{2}\left[q\left(s_{t}, a_{t} ; \boldsymbol{w}\right)-u_{t}\right]^{2} .
$$

然后做一次梯度下降更新 $\boldsymbol{w}$ :

$$
\boldsymbol{w} \leftarrow \boldsymbol{w}-\alpha \cdot \nabla_{\boldsymbol{w}} L(\boldsymbol{w}),
$$

这样可以让价值网络的预测 $q\left(s_{t}, a_{t} ; \boldsymbol{w}\right)$ 更接近 $u_{t}$ 。这种训练价值网络的方法不是 TD。

在强化学习中, 训练价值网络的时候以 $u_{t}$ 作为目标, 这种方式被称作 “蒙特卡洛”。原 因是这样的, 动作价值函数可以写作 $Q_{\pi}\left(s_{t}, a_{t}\right)=\mathbb{E}\left[U_{t} \mid S_{t}=s_{t}, A_{t}=a_{t}\right]$, 而我们用实际 观测 $u_{t}$ 去近似期望, 这就是典型的蒙特卡洛近似。

蒙特卡洛的好处是无偏性: $u_{t}$ 是 $Q_{\pi}\left(s_{t}, a_{t}\right)$ 的无偏估计。由于 $u_{t}$ 的无偏性, 拿 $u_{t}$ 作 为目标训练价值网络, 得到的价值网络也是无偏的。

蒙特卡洛的坏处是方差大。随机变量 $U_{t}$ 依赖于 $S_{t+1}, A_{t+1}, \cdots, S_{n}, A_{n}$ 这些随机变 量, 其中不确定性很大。观测值 $u_{t}$ 虽然是 $U_{t}$ 的无偏估计, 但可能实际上离 $\mathbb{E}\left[U_{t}\right]$ 很远。 因此, 拿 $u_{t}$ 作为目标训练价值网络, 收玫会很慢。

## $5.4 .2$ 自举

在介绍价值学习的自举之前, 先解释一下什么叫自举。大家可能经常在强化学习和 统计学的文章里见到 bootstrapping 这个词。它的字面意思是“拔自己的鞋带, 把自己举起 来”。所以 bootstrapping 翻译成 “自举”, 即自己把自己举起来。自举听起来很荒谬。即使 你 “力拔山兮气盖世”, 你也没办法拔自己的鞋带, 把自己举起来。虽然自举乍看起来不现 实, 但是在统计和机器学习是可以做到自举的; 自举在统计和机器学习里面非常常用。 在强化学习中, “自举”的意思是“用一个估算去更新同类的估算”, 类似于“自己把自己 给举起来” SARSA 使用的单步 TD 目标定义为：

$$
\widehat{y}_{t}=r_{t}+\underbrace{\gamma \cdot q\left(s_{t+1}, a_{t+1} ; \boldsymbol{w}\right)}_{\text {价值网络做出的估计 }} .
$$

SARSA 鼓励 $q\left(s_{t}, a_{t} ; \boldsymbol{w}\right)$ 接近 $\widehat{y}_{t}$, 所以定义损失函数

$$
L(\boldsymbol{w})=\frac{1}{2}[\underbrace{q\left(s_{t}, a_{t} ; \boldsymbol{w}\right)-\widehat{y}_{t}}_{\text {让价值网络拟合 } \widehat{y}_{t}}]^{2} .
$$

$\mathrm{TD}$ 目标 $\widehat{y}_{t}$ 的一部分是价值网络做出的估计 $\gamma \cdot q\left(s_{t+1}, a_{t+1} ; \boldsymbol{w}\right)$, 然后 SARSA 让 $q\left(s_{t}, a_{t} ; \boldsymbol{w}\right)$ 去拟合 $\widehat{y}_{t}$ 。这就是用价值网络自己做出的估计去更新价值网络自己, 这属于“自举”。 1

自举的好处是方差小。单步 $\mathrm{TD}$ 目标的随机性只来自于 $S_{t+1}$ 和 $A_{t+1}$, 而回报 $U_{t}$ 的 随机性来自于 $S_{t+1}, A_{t+1}, \cdots, S_{n}, A_{n}$ 。很显然, 单步 $\mathrm{TD}$ 目标的随机性较小, 因此方差较 小。用自举训练价值网络, 收玫比较快。

自举的坏处是有偏差。价值网络 $q(s, a ; \boldsymbol{w})$ 是对动作价值 $Q_{\pi}(s, a)$ 的近似。最理想的 情况下, $q(s, a ; \boldsymbol{w})=Q_{\pi}(s, a), \forall s, a$ 。假如碰巧 $q\left(s_{j+1}, a_{j+1} ; \boldsymbol{w}\right)$ 低估（或高估）真实价 值 $Q_{\pi}\left(s_{j+1}, a_{j+1}\right)$, 则会发生下面的情况:

$$
\begin{aligned}
& q\left(s_{j+1}, a_{j+1} ; \boldsymbol{w}\right) \text { 低估（或高估） } Q_{\pi}\left(s_{j+1}, a_{j+1}\right) \\
& \Longrightarrow \quad \widehat{y}_{j} \quad \text { 低估（或高估） } Q_{\pi}\left(s_{j}, a_{j}\right) \\
& \Longrightarrow q\left(s_{j}, a_{j} ; \boldsymbol{w}\right) \text { 低估（或高估） } Q_{\pi}\left(s_{j}, a_{j}\right) .
\end{aligned}
$$

也就是说, 自举会让偏差从 $\left(s_{t+1}, a_{t+1}\right)$ 传播到 $\left(s_{t}, a_{t}\right)$ 。第 $6.2$ 节详细讨论自举造成的偏 差以及解决方案。

### 3 蒙特卡洛和自举的对比

在价值学习中, 用实际观测的回 报 $u_{t}$ 作为目标的方法被称为蒙特卡 洛, 即图 $5.5$ 中的蓝色的箱型图。 $u_{t}$ 是 $Q_{\pi}\left(s_{t}, a_{t}\right)$ 的无偏估计, 即 $U_{t}$ 的 期望等于 $Q_{\pi}\left(s_{t}, a_{t}\right)$ 。但是它的方差很 大, 也就是说实际观测到的 $u_{t}$ 可能离 $Q_{\pi}\left(s_{t}, a_{t}\right)$ 很远。

用单步 $\mathrm{TD}$ 目标 $\widehat{y}_{t}$ 作为目标的方 法被称为自举, 即图 $5.5$ 中的红色的箱 型图。自举的好处在于方差小, $\widehat{y}_{t}$ 不会 偏离期望太远。但是 $\widehat{y}_{t}$ 往往是有偏的,

![](https://cdn.mathpix.com/cropped/2023_02_03_f46f5cf0e4de5b9996dcg-083.jpg?height=589&width=711&top_left_y=1693&top_left_x=935)

图 5.5: $u_{t}$ 和 $\widehat{y}_{t}$ 的箱型图 (boxplot) 示意。 它的期望往往不等于 $Q_{\pi}\left(s_{t}, a_{t}\right)$ 。用自 举训练出的价值网络往往有系统性的偏差 (低估或者高估)。实践中, 自举通常比蒙特卡 洛收玫更快, 这就是为什么训练 DQN 和价值网络通常用 TD 算法。

1 严格地说, TD 目标 $\widehat{y}_{t}$ 中既有自举的成分, 也有蒙特卡洛的成分。TD 目标中的 $\gamma \cdot q\left(s_{t+1}, a_{t+1} ; \boldsymbol{w}\right)$ 是自举, 因为它拿价值网络自己的估计作为目标。TD 目标中的 $r_{t}$ 是实际观测, 它是对 $\mathbb{E}\left[R_{t}\right]$ 的蒙特卡洛。 如图 $5.4$ 所示, 多步 TD 目标 $\widehat{y}_{t}=\left(\sum_{i=0}^{m-1} \gamma^{i} r_{t+i}\right)+\gamma^{m} \cdot q\left(s_{t+m}, a_{t+m} ; \boldsymbol{w}\right)$ 介于蒙 特卡洛和自举之间。多步 $\mathrm{TD}$ 目标有很大的蒙特卡洛成分, 其中的 $\sum_{i=0}^{m-1} \gamma^{i} r_{t+i}$ 基于 $m$ 个实际观测到的奖励。多步 TD 目标也有自举的成分, 其中的 $\gamma^{m} \cdot q\left(s_{t+m}, a_{t+m} ; \boldsymbol{w}\right)$ 是 用价值网络自己算出来的。如果把 $m$ 设置得比较好, 可以在方差和偏差之间找到好的平 衡, 使得多步 TD 目标优于单步 TD 目标, 也优于回报 $u_{t}$ 。

## 第 5 章 知识点

- SARSA 和 Q 学习都属于 TD 算法, 但是两者有所区别。SARSA 算法的目的是学习 动作价值函数 $Q_{\pi}$, 而 $\mathrm{Q}$ 学习算法目的是学习最优动作价值函数 $Q_{\star}$ 。SARSA 算法 是同策略, 而 $\mathrm{Q}$ 学习算法是异策略。SARSA 不能用经验回放, 而 $\mathrm{Q}$ 学习可以用经 验回放。

- 价值网络 $q(s, a ; \boldsymbol{w})$ 是对动作价值函数 $Q_{\pi}(s, a)$ 的近似。可以用 SARSA 算法学习 价值网络。

- 多步 TD 目标是对单步 TD 目标的推广。多步 TD 目标可以平衡蒙特卡洛和自举, 取得比单步 TD 目标更好的效果。

## 第 5 章 相关文献 $\propto$

Q 学习算法首先由 Watkins 在他 1989 年的博士论文 ${ }^{[123]}$ 中提出。Watkins 和 Dayan 发表在 1992 年的论文 ${ }^{[122]}$ 分析了 $\mathrm{Q}$ 学习的收敛。1994 年的论文 ${ }^{[57,112]}$ 改进了 Q 学习 算法的收敛分析。

SARSA 算法比 $Q$ 学习提出得晩。SARSA 首先由 Rummery 和 Niranjan 于 1994 年提 出 ${ }^{[88]}$, 但名字不叫 SARSA。SARSA 的名字是 Sutton 在 1996 年起的 ${ }^{[103]}$ 。

多步 TD 目标也是 Watkins 1989 年的博士论文 ${ }^{[123]}$ 提出的。Sutton 和 Barto 的书 [104〕对多步 TD 目标有详细介绍和分析。近年来有不少论文 (比如 ${ }^{[75,117,48])}$ 表明多 步 TD 目标非常有用。

## 第 5 章 习题

1. SARSA 算法学习的价值网络 $q(s, a ; \boldsymbol{w})$ 是对 的近似。
A. 动作价值函数 $Q_{\pi}$
B. 最优动作价值函数 $Q_{\star}$
C. 状态价值函数 $V_{\pi}$
D. 最优状态价值函数 $V_{\star}$

2. 在训练价值网络 $q(s, a ; \boldsymbol{w})$ 的过程中, 策略函数 $\pi$ 会对学到的价值网络有很大影响。 请解释其中的原因。

3. 单步 TD 目标和多步 TD 目标的定义分别是:

$$
\begin{aligned}
& \widehat{y}_{t}=r_{t+1}+\gamma \cdot q\left(s_{t+1}, a_{t+1} ; \boldsymbol{w}\right) \\
& \widehat{y}_{t}=\sum_{i=0}^{m-1} \gamma^{i} r_{t+i}+\gamma^{m} \cdot q\left(s_{t+m}, a_{t+m} ; \boldsymbol{w}\right) .
\end{aligned}
$$

请解释为什么用多步 TD 目标造成的偏差较小。
