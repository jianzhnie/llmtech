

## 第 2 章 蒙特卡洛

本章内容分两节。第 $2.1$ 节简要介绍一些概率论基础知识, 特别是随机变量与观测 值的区别。第 $2.2$ 节用一些具体例子讲解蒙特卡洛算法。蒙特卡洛是很多强化学习算法 的关键要素, 后面章节会反复用到蒙特卡洛。

## 1 随机变量

强化学习中会经常用到两个概念：随机变量、观测值。随机变量是一个不确定量, 它 的值取决于一个随机事件的结果。比如抛一枚硬币, 正面朝上记为 0 , 反面朝上记为 1 。 抛硬币是个随机事件, 抛硬币的结果记为随机变量 $X$, 用大写字母表示。随机变量 $X$ 有 两种可能的取值： 0 或 1 。抛硬币之前, $X$ 取 0 或 1 是均匀随机的, 即取值的概率都为 $\frac{1}{2}: \mathbb{P}(X=0)=\mathbb{P}(X=1)=\frac{1}{2}$ 。抛硬币之后, 我们会观测到硬币哪一面朝上, 此时随机 变量 $X$ 就有了观测值, 记作 $x$ 。举个例子, 如果重复抛硬币 4 次, 得到了 4 个观测值:

$$
x_{1}=1, \quad x_{2}=1, \quad x_{3}=0, \quad x_{4}=1 .
$$

这四个观测值只是数字而已, 没有随机性。本书用大写字母表示随机变量, 小写字母表 示观测值, 避免造成混淆。

给定随机变量 $X$, 它的累积分布函数 (cumulative distribution function, 缩写 CDF) 是 函数 $F_{X}: \mathbb{R} \rightarrow[0,1]$, 定义为

$$
F_{X}(x)=\mathbb{P}(X \leq x) .
$$

下面我们定义概率质量函数 (probability mass function, 缩写 PMF) 和概率密度函数 (probability density function, 缩写 PDF)。

- 概率质量函数 (PMF) 描述一个离散概率分布一一即变量的取值范围 $\mathcal{X}$ 是个离散 集合。在抛硬币的例子中, 随机变量 $X$ 的取值范围是集合 $\mathcal{X}=\{0,1\} 。 X$ 的概率 质量函数是

$$
p(0)=\mathbb{P}(X=0)=\frac{1}{2}, \quad p(1)=\mathbb{P}(X=1)=\frac{1}{2} .
$$

公式的意思随机变量取值 0 和 1 的概率都是 $\frac{1}{2},{ }^{1}$ 见图 2.1(左) 的例子。概率质量函 数有这样的性质:

$$
\sum_{x \in \mathcal{X}} p(x)=1 .
$$

- 概率密度函数 (PDF) 描述一个连续概率分布一一即变量的取值范围 $\mathcal{X}$ 是个连续集 合。正态分布是最常见的一种连续概率分布, 随机变量 $X$ 的取值范围是所有实数 $\mathbb{R}$ 。正态分布的概率密度函数是

$$
p(x)=\frac{1}{\sqrt{2 \pi} \sigma} \cdot \exp \left(-\frac{(x-\mu)^{2}}{2 \sigma^{2}}\right) .
$$

1注意到 $p(x)$ 应该写为 $p_{X}(x)$, 这里为了记号方便, 我们省掉了下标 $X$ 。 此处的 $\mu$ 是均值, $\sigma$ 是标准差。图 2.1(右) 的例子说明 $X$ 在均值附近取值的可能性 大, 在远离均值的地方取值的可能性小。注意, 跟离散分布不同, 连续分布的 $p(x)$ 不等于 $\mathbb{P}(X=x)$ 。概率密度函数有这样的性质：

$$
\int_{-\infty}^{x} p(u) d u=F_{X}(x) .
$$

因此概率密度函数有这样的性质： $\int_{\mathbb{R}} p(x) d x=1$ 和 $p(x)=F_{X}^{\prime}(x)$ 。
![](https://cdn.mathpix.com/cropped/2023_02_03_f46f5cf0e4de5b9996dcg-028.jpg?height=460&width=1324&top_left_y=650&top_left_x=430)

图 2.1: 左图是抛硬币的例子。右图是均值为零的正态分布。

对于离散随机变量 $X$, 函数 $h(X)$ 关于变量 $X$ 的期望是

$$
\mathbb{E}_{X \sim p(\cdot)}[h(X)]=\sum_{x \in \mathcal{X}} p(x) \cdot h(x) .
$$

如果 $X$ 是连续随机变量, 则函数 $h(X)$ 关于变量 $X$ 的期望是

$$
\mathbb{E}_{X \sim p(\cdot)}[h(X)]=\int_{\mathcal{X}} p(x) \cdot h(x) d x .
$$

设 $g(X, Y)$ 为二元函数。如果对 $g(X, Y)$ 关于随机变量 $X$ 求期望, 那么会消掉 $X$, 得到 的结果是 $Y$ 的函数。举个例子, 设随机变量 $X$ 的取值范围是 $\mathcal{X}=[0,10]$, 概率密度函数 是 $p(x)=\frac{1}{10}$ 。设 $g(X, Y)=\frac{1}{5} X Y$, 那么 $g(X, Y)$ 关于 $X$ 的期望等于

$$
\begin{aligned}
\mathbb{E}_{X \sim p(\cdot)}[g(X, Y)] & =\int_{\mathcal{X}} g(x, Y) \cdot p(x) d x \\
& =\int_{0}^{10} \frac{1}{5} x Y \cdot \frac{1}{10} d x \\
& =Y .
\end{aligned}
$$

上述例子说明期望如何消掉函数 $g(X, Y)$ 中的变量 $X$ 。

强化学习中常用到随机抽样, 此处给一个直观的解释。如图 $2.2$ 所示, 箱子里有 10 个球, 其中 2 个是红色, 5 个是绿色, 3 个是蓝色。我现在把箱子摇一摇, 把手伸进箱子 里, 闭着眼睛摸出来一个球。当我睁开眼睛, 就观测到球的颜色, 比如红色。这个过程 叫做随机抽样, 本轮随机抽样的结果是红色。如果把抽到的球放回, 可以无限次重复随 机抽样, 得到多个观测值。

请读者注意随机变量与观测值的区别。在我摸出一个球之前, 球的颜色是随机变量, 记作 $X$, 它有三种可能的取值一红色、绿色、蓝色。当我摸到球之后, 我观测到了颜 色“ $x=$ 红”, 这是 $X$ 的一个观测值。注意, 观测值 “ $x=$ 红”没有随机性, 而变量 $X$ 有随

![](https://cdn.mathpix.com/cropped/2023_02_03_f46f5cf0e4de5b9996dcg-029.jpg?height=440&width=1239&top_left_y=257&top_left_x=340)

图 2.2: 箱子里有 10 个球。 2 个是红色, 5 个是绿色, 3 个是蓝色。

机性。

可以用计算机程序做随机抽样。假设箱子里有很多个球, 红色球占 $20 \%$, 绿色球占 $50 \%$, 蓝色球占 $30 \%$ 。如果我随机摸一个球, 那么抽到的球服从这样一个离散概率分布：

$$
p(\text { 红 })=0.2, \quad p(\text { 绿 })=0.5, \quad p(\text { 蓝 })=0.3 .
$$

下面的 Python 代码按照概率质量 $p$ 做随机抽样, 重复 100 次, 输出抽样的结果。

![](https://cdn.mathpix.com/cropped/2023_02_03_f46f5cf0e4de5b9996dcg-029.jpg?height=560&width=1408&top_left_y=1232&top_left_x=266)



## $2.2$ 蒙特卡洛估计

蒙特卡洛（Monte Carlo）是一大类随机算法（randomized algorithms）的总称, 它们 通过随机样本来估算真实值。本节用几个例子讲解蒙特卡洛算法。

### 1 例一：近似 $\pi$ 值

我们都知道圆周率 $\pi$ 约等于 $3.1415927$ 。现在假装我们不知道 $\pi$, 而是要想办法近似 估算 $\pi$ 值。假设我们有（伪）随机数生成器, 我们能不能用随机样本来近似 $\pi$ 呢? 这一 小节讨论使用蒙特卡洛如何近似 $\pi$ 值。

假设我们有一个 (伪) 随机数生成器, 可以均匀生成 $-1$ 到 $+1$ 之间的数。每次生成 两个随机数, 一个作为 $x$, 另一个作为 $y$ 。于是每次生成了一个平面坐标系中的点 $(x, y)$, 见图 2.3(左)。因为 $x$ 和 $y$ 都是在 $[-1,1]$ 区间上均匀分布, 所以 $[-1,1] \times[-1,1]$ 这个正 方形内的点被抽到的概率是相同的。我们重复抽样 $n$ 次, 得到了 $n$ 个正方形内的点。

![](https://cdn.mathpix.com/cropped/2023_02_03_f46f5cf0e4de5b9996dcg-030.jpg?height=503&width=506&top_left_y=1145&top_left_x=478)

从蓝色正方形中做随机抽样, 得到 $n$ 个红色的点。

![](https://cdn.mathpix.com/cropped/2023_02_03_f46f5cf0e4de5b9996dcg-030.jpg?height=503&width=508&top_left_y=1145&top_left_x=1205)

抽到的红色的点可能落在绿色 的圆内部, 也可能落在外部。

图 2.3: 通过抽样来近似 $\pi$ 值。

如图 2.3(右) 所示, 蓝色正方形里面包含一个绿色的圆, 圆心是 $(0,0)$, 半径等于 1 。刚 才随机生成的 $n$ 个点有些落在圆外面, 有些落在圆里面。请问一个点落在圆里面的概率有 多大呢? 由于抽样是均匀的, 因此这个概率显然是圆的面积与正方形面积的比值。正方形 的面积是边长的平方, 即 $a_{1}=2^{2}=4$ 。 圆的面积是 $\pi$ 乘以半径的平方, 即 $a_{2}=\pi \times 1^{2}=\pi$ 。 那么一个点落在圆里面的概率就是

$$
p=\frac{a_{2}}{a_{1}}=\frac{\pi}{4} .
$$

设我们随机抽样了 $n$ 个点, 设圆内的点的数量为随机变量 $M$ 。显然, $M$ 的期望等于

$$
\mathbb{E}[M]=p n=\frac{\pi n}{4} .
$$

注意, 这只是期望, 并不是实际发生的结果。如果你抽 $n=5$ 个点, 那么期望有 $\mathbb{E}[M]=\frac{5 \pi}{4}$ 个点落在圆内。但实际观测值 $m$ 可能等于 $0 、 1 、 2 、 3 、 4 、 5$ 中的任何一个。

给定一个点的坐标 $(x, y)$, 如何判断该点是否在圆内呢? 已知圆心在原点, 半径等于 1 , 我们用一下圆的方程。如果 $(x, y)$ 满足:

$$
x^{2}+y^{2} \leq 1,
$$

则说明 $(x, y)$ 落在圆里面; 反之, 点就在圆外面。

我们均匀随机抽样得到 $n$ 个点, 通过圆的方程对每个点做判别, 发现有 $m$ 个点落在 圆里面。如果 $n$ 非常大, 那么随机变量 $M$ 的真实观测值 $m$ 就会非常接近期望 $\mathbb{E}[M]=\frac{\pi n}{4}$ :

$$
m \approx \frac{\pi n}{4} .
$$

由此得到：

$$
\pi \approx \frac{4 m}{n}
$$

我们可以依据这个公式做编程实现。下面是伪代码:

1. 初始化 $m=0$ 。用户指定样本数量 $n$ 的大小。 $n$ 越大, 精度越高, 但是计算量越大。

2. 把下面的步骤重复 $n$ 次：

(a). 从区间 $[-1,1]$ 上做两次均匀随机抽样得到实数 $x$ 和 $y$ 。

(b). 如果 $x^{2}+y^{2} \leq 1$, 那么 $m \leftarrow m+1$ 。

3. 返回 $\frac{4 m}{n}$ 作为对 $\pi$ 的估计。

大数定律保证了蒙特卡洛的正确性：当 $n$ 趋于无穷， $\frac{4 m}{n}$ 趋于 $\pi$ 。其实还能进一步用 概率不等式分析误差的上界。比如使用 Bernstein 不等式, 可以证明出下面结论:

$$
\left|\frac{4 m}{n}-\pi\right|=O\left(\frac{1}{\sqrt{n}}\right) .
$$

这个不等式说明 $\frac{4 m}{n}$ (即对 $\pi$ 的估计) 会收玫到 $\pi$, 收玫率是 $\frac{1}{\sqrt{n}}$ 。然而这个收玫率并不 快: 样本数量 $n$ 增加一万倍, 精度才能提高一百倍。

### 2 例二：估算阴影部分面积

图 $2.4$ 中有正方形、圆、扇形, 几个形状相交。 请估算阴影部分面积。这个问题常见于初中数学竞 赛。假如你不会微积分, 也不会几何技巧，你是否 有办法近似估算阴影部分面积呢? 用蒙特卡洛可以 很容易解决这个问题。

图 $2.5$ 中绿色圆的圆心是 $(1,1)$, 半径等于 1 ; 蓝色扇形的圆心是 $(0,0)$, 半径等于 2 。阴影区域内 的点 $(x, y)$ 在绿色的圆中, 而不在蓝色的扇形中。

![](https://cdn.mathpix.com/cropped/2023_02_03_f46f5cf0e4de5b9996dcg-031.jpg?height=448&width=494&top_left_y=1775&top_left_x=1155)

图 2.4: 估算阴影部分面积。

- 利用圆的方程可以判定点 $(x, y)$ 是否在绿色圆里面。如果 $(x, y)$ 满足方程

$$
(x-1)^{2}+(y-1)^{2} \leq 1,
$$

则说明 $(x, y)$ 在绿色圆里面。

- 利用扇形的方程可以判定点 $(x, y)$ 是否在蓝色扇形外面。如果点 $(x, y)$ 满足方程

$$
x^{2}+y^{2}>2^{2},
$$

则说明 $(x, y)$ 在蓝色扇形外面。

如果一个点同时满足方程 (2.1) 和 (2.2), 那么这个点一定在阴影区域内。从 $[0,2] \times[0,2]$ 这个正方形中做随机抽样, 得到 $n$ 个点。然后用两个方程览选落在阴影部分的点。
![](https://cdn.mathpix.com/cropped/2023_02_03_f46f5cf0e4de5b9996dcg-032.jpg?height=524&width=1412&top_left_y=554&top_left_x=390)

图 2.5: 如果一个点在阴影部分, 那么它在左边绿的的圆中, 而不在右边蓝色的扇形中。

我们在正方形 $[0,2] \times[0,2]$ 中随机均匀抽样, 得到的点有一定概率会落在阴影部分。 我们来计算这个概率。正方形的边长等于 2 , 所以面积 $a_{1}=4$ 。设阴影部分面积为 $a_{2}$ 。那 么点落在阴影部分概率是

$$
p=\frac{a_{2}}{a_{1}}=\frac{a_{2}}{4} .
$$

我们从正方形中随机抽 $n$ 个点, 设有 $M$ 个点落在阴影部分内 ( $M$ 是个随机变量)。每个 点落在阴影部分的概率是 $p$, 所以 $M$ 的期望等于

$$
\mathbb{E}[M]=n p=\frac{n a_{2}}{4} .
$$

用方程 (2.1) 和 (2.2) 对 $n$ 个点做笑选, 发现实际上有 $m$ 个点落在阴影部分内（ $m$ 是随机 变量 $M$ 的观测值)。如果 $n$ 很大, 那么 $m$ 会比较接近期望 $\mathbb{E}[M]=\frac{n a_{2}}{4}$, 即

$$
m \approx \frac{n a_{2}}{4} .
$$

也即：

$$
a_{2} \approx \frac{4 m}{n}
$$

这个公式就是对阴影部分面积的估计。我们依据这个公式做编程实现。下面是伪代码:

1. 初始化 $m=0$ 。用户指定样本数量 $n$ 的大小。 $n$ 越大, 精度越高, 但是计算量越大。

2. 把下面的步骤重复 $n$ 次:

(a). 从区间 $[0,2]$ 上均匀随机抽样得到 $x$; 再做一次均匀随机抽样, 得到 $y$ 。

(b). 如果 $(x-1)^{2}+(y-1)^{2} \leq 1$ 和 $x^{2}+y^{2}>4$ 两个不等式都成立, 那么让 $m \leftarrow m+1$ 。

3. 返回 $\frac{4 m}{n}$ 作为对阴影部分面积的估计。

### 3 例三：近似定积分

近似求积分是蒙特卡洛最重要的应用之一, 在科学和工程中有广泛的应用。举个例 子, 给定一个函数：

$$
f(x)=\frac{1}{1+(\sin x) \cdot(\ln x)^{2}},
$$

要求计算 $f$ 在区间 $0.8$ 到 3 上的定积分：

$$
I=\int_{0.8}^{3} f(x) d x .
$$

有很多科学和工程问题需要计算定积分, 而函数 $f(x)$ 可能很复杂, 求定积分会很困难, 甚至有可能不存在解析解。如果求解析解很困难, 或者解析解不存在, 则可以用蒙特卡 洛近似计算数值解。

一元函数的定积分是相对比较简单的问题。一元函数的意思是变量 $x$ 是个标量。给 定一元函数 $f(x)$, 求函数在 $a$ 到 $b$ 区间上的定积分:

$$
I=\int_{a}^{b} f(x) d x .
$$

蒙特卡洛方法通过下面的步骤近似定积分：

1. 在区间 $[a, b]$ 上做随机抽样, 得到 $n$ 个样本, 记作: $x_{1}, \cdots, x_{n}$ 。样本数量 $n$ 由用户 自己定, $n$ 越大, 计算量越大, 近似越准确。

2. 对函数值 $f\left(x_{1}\right), \cdots, f\left(x_{n}\right)$ 求平均, 再乘以区间长度 $b-a$ ：

$$
q_{n}=(b-a) \cdot \frac{1}{n} \sum_{i=1}^{n} f\left(x_{i}\right) .
$$

3. 返回 $q_{n}$ 作为定积分 $I$ 的估计值。

多元函数的定积分要复杂一些。设 $f: \mathbb{R}^{d} \mapsto \mathbb{R}$ 是一个多元函数, 变量 $\boldsymbol{x}$ 是 $d$ 维向 量。要求计算 $f$ 在集合 $\Omega$ 上的定积分：

$$
I=\int_{\Omega} f(\boldsymbol{x}) d \boldsymbol{x} .
$$

蒙特卡洛方法通过下面的步骤近似定积分：

1. 在集合 $\Omega$ 上做均匀随机抽样, 得到 $n$ 个样本, 记作向量 $\boldsymbol{x}_{1}, \cdots, \boldsymbol{x}_{n}$ 。样本数量 $n$ 由 用户自己定， $n$ 越大, 计算量越大，近似越准确。

2. 计算集合 $\Omega$ 的体积:

$$
v=\int_{\Omega} d \boldsymbol{x} .
$$

3. 对函数值 $f\left(\boldsymbol{x}_{1}\right), \cdots, f\left(\boldsymbol{x}_{n}\right)$ 求平均, 再乘以 $\Omega$ 的体积 $v$ :

$$
q_{n}=v \cdot \frac{1}{n} \sum_{i=1}^{n} f\left(\boldsymbol{x}_{i}\right) .
$$

4. 返回 $q_{n}$ 作为定积分 $I$ 的估计值。

注意, 算法第二步需要求 $\Omega$ 的体积。如果 $\Omega$ 是长方体、球体等规则形状, 那么可以解析 地算出体积 $v$ 。可是如果 $\Omega$ 是不规则形状, 那么就需要定积分求 $\Omega$ 的体积 $v$, 这是比较 困难的。可以用类似于上一小节“求阴影部分面积”的方法近似计算体积 $v$ 。 举例讲解多元函数的蒙特卡洛积分： 这个例 子中被积分的函数是二元函数：

![](https://cdn.mathpix.com/cropped/2023_02_03_f46f5cf0e4de5b9996dcg-034.jpg?height=158&width=579&top_left_y=361&top_left_x=453)

直观地说, 如果点 $(x, y)$ 落在右图的绿色圆内, 那 么函数值就是 1 ; 否则函数值就是 0 。定义集合 $\Omega=$ $[-1,1] \times[-1,1]$, 即右图中蓝色的正方形, 它的面 积是 $v=4$ 。定积分

$$
I=\int_{\Omega} f(x, y) d x d y
$$

![](https://cdn.mathpix.com/cropped/2023_02_03_f46f5cf0e4de5b9996dcg-034.jpg?height=480&width=486&top_left_y=231&top_left_x=1276)

图 2.6: 用蒙特卡洛积分近似 $\pi$ 。

等于多少呢? 很显然, 定积分等于圆的面积, 即 $\pi \cdot 1^{2}=\pi$ 。因此, 定积分 $I=\pi$ 。用蒙 特卡洛求出 $I$, 就得到了 $\pi$ 。从集合 $\Omega=[-1,1] \times[-1,1]$ 上均匀随机抽样 $n$ 个点, 记作 $\left(x_{1}, y_{1}\right), \cdots,\left(x_{n}, y_{n}\right)$ 。应用公式 $(2.3)$, 可得

$$
q_{n}=v \cdot \frac{1}{n} \sum_{i=1}^{n} f\left(x_{i}, y_{i}\right)=\frac{4}{n} \sum_{i=1}^{n} f\left(x_{i}, y_{i}\right) .
$$

把 $q_{n}$ 作为对定积分 $I=\pi$ 的近似。这与第 $2.2 .1$ 小节近似 $\pi$ 的算法完全相同, 区别在于 此处的算法是从另一个角度推导出的。

### 4 例四：近似期望

蒙特卡洛还可以用来近似期望, 这在整本书中会反复应用。设 $X$ 是 $d$ 维随机变量, 它的取值范围是集合 $\Omega \subset \mathbb{R}^{d}$ 。函数 $p(\boldsymbol{x})$ 是 $X$ 的概率密度函数。设 $f: \Omega \mapsto \mathbb{R}$ 是任意的 多元函数，它关于变量 $X$ 的期望是:

$$
\mathbb{E}_{X \sim p(\cdot)}[f(X)]=\int_{\Omega} p(\boldsymbol{x}) \cdot f(\boldsymbol{x}) d \boldsymbol{x} .
$$

由于期望是定积分, 所以可以按照上一小节的方法, 用蒙特卡洛求定积分。上一小节在 集合 $\Omega$ 上做均匀抽样, 用得到的样本近似上面公式中的定积分。

下面介绍一种更好的算法。既然我们知道概率密度函数 $p(\boldsymbol{x})$, 我们最好是按照 $p(\boldsymbol{x})$ 做非均匀抽样, 而不是均匀抽样。按照 $p(\boldsymbol{x})$ 做非均匀抽样, 可以比均匀抽样有更快的收 敛。具体步骤如下:

1. 按照概率密度函数 $p(\boldsymbol{x})$, 在集合 $\Omega$ 上做非均匀随机抽样, 得到 $n$ 个样本, 记作向 量 $\boldsymbol{x}_{1}, \cdots, \boldsymbol{x}_{n} \sim p(\cdot)$ 。样本数量 $n$ 由用户自己定, $n$ 越大, 计算量越大, 近似越准 确。

2. 对函数值 $f\left(\boldsymbol{x}_{1}\right), \cdots, f\left(\boldsymbol{x}_{n}\right)$ 求平均:

$$
q_{n}=\frac{1}{n} \sum_{i=1}^{n} f\left(\boldsymbol{x}_{i}\right) .
$$

3. 返回 $q_{n}$ 作为期望 $\mathbb{E}_{X \sim p(\cdot)}[f(X)]$ 的估计值。

注 如果按照上述方式做编程实现, 需要储存函数值 $f\left(\boldsymbol{x}_{1}\right), \cdots, f\left(\boldsymbol{x}_{n}\right)$ 。但用如下的方式 做编程实现, 可以减小内存开销。初始化 $q_{0}=0$ 。从 $t=1$ 到 $n$, 依次计算

$$
q_{t}=\left(1-\frac{1}{t}\right) \cdot q_{t-1}+\frac{1}{t} \cdot f\left(\boldsymbol{x}_{t}\right) .
$$

不难证明, 这样得到的 $q_{n}$ 等于 $\frac{1}{n} \sum_{i=1}^{n} f\left(\boldsymbol{x}_{i}\right)$ 。这样无需存储所有的 $f\left(\boldsymbol{x}_{1}\right), \cdots, f\left(\boldsymbol{x}_{n}\right)$ 。可 以进一步把公式 (2.6) 中的 $\frac{1}{t}$ 替换成 $\alpha_{t}$, 得到公式:

$$
q_{t}=\left(1-\alpha_{t}\right) \cdot q_{t-1}+\alpha_{t} \cdot f\left(\boldsymbol{x}_{t}\right) .
$$

这个公式叫做 Robbins-Monro 算法, 其中 $\alpha_{n}$ 称为学习步长或学习率。只要 $\alpha_{t}$ 满足下面 的性质, 就能保证算法的正确性：

$$
\lim _{n \rightarrow \infty} \sum_{t=1}^{n} \alpha_{t}=\infty \quad \text { 和 } \quad \lim _{n \rightarrow \infty} \sum_{t=1}^{n} \alpha_{t}^{2}<\infty .
$$

显然, $\alpha_{t}=\frac{1}{t}$ 满足上述性质。Robbins-Monro 算法可以应用在 $\mathrm{Q}$ 学习算法中。

### 5 例五 : 随机梯度

我们可以用蒙特卡洛近似期望来理解随机梯度算法。设随机变量 $X$ 为一个数据样 本, 令 $\boldsymbol{w}$ 为神经网络的参数。设 $p(\boldsymbol{x})$ 为随机变量 $X$ 的概率密度函数。定义损失函数 $L(X ; \boldsymbol{w})$ 。它的值越小, 意味着模型做出的预测越准确; 反之, 它的值越大, 则意味着模 型做出的预测越差。因此, 我们希望调整神经网络的参数 $\boldsymbol{w}$, 使得损失函数的期望尽量 小。神经网络的训练可以定义为这样的优化问题:

$$
\min _{\boldsymbol{w}} \mathbb{E}_{X \sim p(\cdot)}[L(X ; \boldsymbol{w})] .
$$

目标函数 $\mathbb{E}_{X}[L(X ; \boldsymbol{w})]$ 关于 $\boldsymbol{w}$ 的梯度是:

$$
\boldsymbol{g} \triangleq \nabla_{\boldsymbol{w}} \mathbb{E}_{X \sim p(\cdot)}[L(X ; \boldsymbol{w})]=\mathbb{E}_{X \sim p(\cdot)}\left[\nabla_{\boldsymbol{w}} L(X ; \boldsymbol{w})\right] .
$$

可以做梯度下降更新 $\boldsymbol{w}$, 以减小目标函数 $\mathbb{E}_{X}[L(X ; \boldsymbol{w})]$ :

$$
\boldsymbol{w} \leftarrow \boldsymbol{w}-\alpha \cdot \boldsymbol{g} .
$$

此处的 $\alpha$ 被称作学习率（learning rate)。直接计算梯度 $\boldsymbol{g}$ 通常会比较慢。为了加速计算, 可以对期望

$$
\boldsymbol{g}=\mathbb{E}_{X \sim p(\cdot)}\left[\nabla_{\boldsymbol{w}} L(X ; \boldsymbol{w})\right]
$$

做蒙特卡洛近似, 把得到的近似梯度 $\tilde{\boldsymbol{g}}$ 称作随机梯度 (stochastic gradient), 用 $\tilde{\boldsymbol{g}}$ 代替 $\boldsymbol{g}$ 来更新 $\boldsymbol{w}$ 。

1. 根据概率密度函数 $p(\boldsymbol{x})$ 做随机抽样, 得到 $B$ 个样本, 记作 $\tilde{\boldsymbol{x}}_{1}, \ldots, \tilde{\boldsymbol{x}}_{B}$ 。

2. 计算梯度 $\nabla_{\boldsymbol{w}} L\left(\tilde{\boldsymbol{x}}_{j} ; \boldsymbol{w}\right), \forall j=1, \ldots, B$ 。对它们求平均:

$$
\tilde{\boldsymbol{g}}=\frac{1}{B} \sum_{j=1}^{B} \nabla_{\boldsymbol{w}} L\left(\tilde{\boldsymbol{x}}_{j} ; \boldsymbol{w}\right) .
$$

我们称 $\tilde{\boldsymbol{g}}$ 为随机梯度。因为 $\mathbb{E}[\tilde{\boldsymbol{g}}]=\boldsymbol{g}$, 它是 $\boldsymbol{g}$ 的一个无偏估计。

3. 做随机梯度下降更新 $\boldsymbol{w}$ :

$$
\boldsymbol{w} \leftarrow \boldsymbol{w}-\alpha \cdot \tilde{\boldsymbol{g}}
$$

样本数量 $B$ 称作批量大小 (batch size), 通常是一个比较小的正整数, 比如 $1 、 8 、 16 、 32$ 。 所以我们称之为最小批 (mini-batch) SGD。

在实际应用中, 样本真实的概率密度函数 $p(\boldsymbol{x})$ 一般是末知的。在训练神经网络的时 候, 我们通常会收集一个训练数据集 $\mathcal{X}=\left\{\boldsymbol{x}_{1}, \cdots, \boldsymbol{x}_{n}\right\}$, 并求解这样一个经验风险最小 化 (empirical risk minimization) 问题:

$$
\min _{\boldsymbol{w}} \frac{1}{n} \sum_{i=1}^{n} L\left(\boldsymbol{x}_{i} ; \boldsymbol{w}\right) .
$$

这相当于我们用下面这个概率质量函数代替真实的 $p(\boldsymbol{x})$ :

$$
p(\boldsymbol{x})= \begin{cases}\frac{1}{n}, & \text { 如果 } \boldsymbol{x} \in \mathcal{X} ; \\ 0, & \text { 如果 } \boldsymbol{x} \notin \mathcal{X} .\end{cases}
$$

公式的意思是随机变量 $X$ 的取值是 $n$ 个数据点中的一个, 概率都是 $\frac{1}{n}$ 。那么最小批 SGD 的每一轮都从集合 $\left\{\boldsymbol{x}_{1}, \cdots, \boldsymbol{x}_{n}\right\}$ 中均匀随机抽取 $B$ 个样本, 计算随机梯度, 更新模型参 数 $\boldsymbol{w}$ 。

## 第 2 章 知识点

- 请读者理解并记忆这些概率统计的基本概念：随机变量、观测值、概率质量函数、 概率密度函数、期望、随机抽样。强化学习会反复用到这些概念。

- 本章详细讲解了蒙特卡洛的应用。其中最重要的知识点是蒙特卡洛近似期望：设 $X$ 是随机变量, $x$ 是观测值, 蒙特卡洛用 $f(x)$ 近似期望 $\mathbb{E}[f(X)]$ 。强化学习中的 $\mathrm{Q}$ 学 习、SARSA、策略梯度等算法都需要这样用蒙特卡洛近似期望。

## 第 2 章 习题

1. 设 $X$ 是离散随机变量, 取值范围是集合 $\mathcal{X}=\{1,2,3\}$ 。定义概率质量函数:

$$
\begin{aligned}
& p(1)=\mathbb{P}(X=1)=0.4, \\
& p(2)=\mathbb{P}(X=2)=0.1 \\
& p(3)=\mathbb{P}(X=3)=0.5 .
\end{aligned}
$$

定义函数 $f(x)=2 x^{2}+3$ 。请计算 $\mathbb{E}_{X \sim p(\cdot)}[f(X)]$ 。

2. 设 $X$ 服从均值为 $\mu=1$ 、标准差 $\sigma=2$ 的一元正态分布。定义函数 $f(x)=2 x+$ $10 \sqrt{|x|}+3$ 。请设计蒙特卡洛算法, 并编程计算 $\mathbb{E}_{X}[f(X)]$ 。

3. Bernstein 概率不等式是这样定义的。设 $Z_{1}, \cdots, Z_{n}$ 为独立的随机变量, 且满足下 面三个条件：

- 变量的期望为零: $\mathbb{E}\left[Z_{1}\right]=\cdots=\mathbb{E}\left[Z_{n}\right]=0$ 。

- 变量是有界的：存在 $b>0$, 使得 $\left|Z_{i}\right| \leq b, \forall i=1, \cdots, n$ 。

- 变量的方差是有界的：存在 $v>0$, 使得 $\mathbb{E}\left[Z_{i}^{2}\right] \leq v, \forall i=1, \cdots, n$ 。 则下面的概率不等式成立:

$$
\mathbb{P}\left(\left|\frac{1}{n} \sum_{i=1}^{n} Z_{i}\right| \geq \epsilon\right) \leq \exp \left(-\frac{\epsilon^{2} n / 2}{v+\epsilon b / 3}\right) .
$$

公式 (2.5) 算出的 $q_{n}$ 是 $\pi$ 的蒙特卡洛近似。请用 Bernstein 不等式证明:

$$
\left|q_{n}-\pi\right|=O\left(\frac{1}{\sqrt{n}}\right) \quad \text { 以很高的概率成立。 }
$$

(提示 : 设 $\left(X_{i}, Y_{i}\right)$ 是从正方形 $[-1,1] \times[-1,1]$ 中随机抽取的点。二元函数 $f$ 在公 式 (2.4) 中定义。设 $Z_{i}=4 f\left(X_{i}, Y_{i}\right)-\pi$, 它是个均值为零的随机变量。)

4. 初始化 $q_{0}=0$ 。让 $t$ 从 1 增长到 $n$, 依次计算

$$
q_{t}=\left(1-\frac{1}{t}\right) \cdot q_{t-1}+\frac{1}{t} \cdot f\left(\boldsymbol{x}_{t}\right) .
$$

请证明上述迭代得到的结果 $q_{n}$ 等于 $\frac{1}{n} \sum_{i=1}^{n} f\left(\boldsymbol{x}_{i}\right)$ 。
