# 第 1 章 机器学习基础

本书假设读者有一定的机器学习基础, 了解矩阵计算、数值优化等基础知识。本章 只是帮助读者查漏补缺, 并由此熟悉本书的语言和符号。本章内容分三节, 分别介绍线 性模型、深度神经网络、反向传播和梯度下降。

## $1.1$ 线性模型

线性模型 (linear models) 是一类最简单的有监督机器学习模型, 常被用于简单的机 器学习任务。可以将线性模型视为单层的神经网络。本节讨论线性回归、逻辑斯蒂回归 (logistic regression)、softmax 分类器等三种模型。

### 1 线性回归

下面我们以房价预测问题为例讲解回归 (regression)。假如你在一家房产中介工作, 你想要根据房屋的属性（attributes 或 features）来初步估算房屋价格。跟房价相关的属性 包括面积、建造年份、离地铁站的距离、等等。设房屋一共有 $d$ 个属性, 把它们记作一 个向量:

$$
\boldsymbol{x}=\left[x_{1}, x_{2}, \cdots, x_{d}\right]^{T} .
$$

本书中的向量 $\boldsymbol{x}$ 表示为列向量, 记作粗体小写字母。它的转置 $\boldsymbol{x}^{T}$ 表示行向量。问题的 目标是基于房屋的属性 $\boldsymbol{x} \in \mathbb{R}^{d}$ 预测其价格。

有多种方法对房价预测问题建模。最简单的方法是使用如下线性模型：

$$
f(\boldsymbol{x} ; \boldsymbol{w}, b) \triangleq \boldsymbol{x}^{T} \boldsymbol{w}+b .
$$

这里 $\boldsymbol{w} \in \mathbb{R}^{d}$ 和 $b \in \mathbb{R}$ 是模型的参数 (parameters)。线性模型 $f(\boldsymbol{x} ; \boldsymbol{w}, b)$ 的输出就是对房 价的预测, 输出既依赖于房屋的特征 $\boldsymbol{x}$, 也依赖于参数 $\boldsymbol{w}$ 和 $b$ 。很多书和论文将 $\boldsymbol{w}$ 称 作权重 (weights), 将 $b$ 称作偏移量 (offset 或 intercept), 原因是这样的：可以将 $f$ 的定义 $\boldsymbol{x}^{T} \boldsymbol{w}+b$ 展开, 得到

$$
f(\boldsymbol{x} ; \boldsymbol{w}, b) \triangleq w_{1} x_{1}+w_{2} x_{2}+\cdots+w_{d} x_{d}+b .
$$

如果 $x_{1}$ 是房屋的面积, 那么 $w_{1}$ 就是房屋面积在房价中的权重。 $w_{1}$ 越大, 说明房价与面 积的相关性越强, 这就是为什么 $\boldsymbol{w}$ 被称为权重。可以把偏移量 $b$ 视作市面上房价的均值 或者中位数, 它与房屋的具体属性无关。

线性模型 $f(\boldsymbol{x} ; \boldsymbol{w}, b)$ 依赖于参数 $\boldsymbol{w}$ 和 $b$; 只有确定了 $\boldsymbol{w}$ 和 $b$, 我们才能利用线性模 型做预测。该怎么样获得 $\boldsymbol{w}$ 和 $b$ 呢? 可以用历史数据来训练模型, 得到参数估计 $\widehat{\boldsymbol{w}}$ 和 $\hat{b}$, 然后就可以用线性模型做预测：

$$
f(\boldsymbol{x} ; \widehat{\boldsymbol{w}}, \hat{b}) \triangleq \boldsymbol{x}^{T} \widehat{\boldsymbol{w}}+\hat{b}
$$

卖家和中介可以用这个训练好的模型 $f$ 给待售房屋定价。对于一个待售的房屋, 首先找 到它的面积、建造年份等属性, 表示成向量 $\boldsymbol{x}^{\prime}$, 然后把它输入 $f$, 得到

$$
\widehat{y}^{\prime}=f\left(\boldsymbol{x}^{\prime} ; \widehat{\boldsymbol{w}}, \hat{b}\right),
$$

把它作为对该房屋价格的预测。

一个有监督学习问题的数据通常分为训练数据集、验证数据集和测试数据集。前两 者输入特征 $\boldsymbol{x}$ 对应的输出值 $y$ 是存在的。下面具体讲述最小二乘回归方法（least squares regression）的流程。

- 第一, 准备训练数据。收集近期的 $n$ 个房屋的属性和卖价, 作为训练数据集。把训 练集记作 $\left(\boldsymbol{x}_{1}, y_{1}\right), \cdots,\left(\boldsymbol{x}_{n}, y_{n}\right)$ 。向量 $\boldsymbol{x}_{i} \in \mathbb{R}^{d}$ 表示第 $i$ 个房屋的所有属性, 标量 $y_{i}$ 表示该房屋的成交价格。

- 第二, 把训练描述成优化问题。模型对第 $i$ 个房屋价格的预测是 $\widehat{y}_{i}=f\left(\boldsymbol{x}_{i} ; \widehat{\boldsymbol{w}}, \hat{b}\right)$, 而这个房屋的真实成交价格是 $y_{i}$ 。我们希望 $\widehat{y}_{i}$ 尽量接近 $y_{i}$, 因此平方残差 $\left(\widehat{y}_{i}-y_{i}\right)^{2}$ 越小越好。定义损失函数：

$$
L(\boldsymbol{w}, b)=\frac{1}{2 n} \sum_{i=1}^{n}\left[f\left(\boldsymbol{x}_{i} ; \boldsymbol{w}, b\right)-y_{i}\right]^{2} .
$$

最小二乘估计希望找到 $\boldsymbol{w}$ 和 $b$ 使得损失函数尽量小, 也就是让模型的预测尽量准 确。

- 第三, 求解模型。把优化问题的最优解记作:

$$
(\widehat{\boldsymbol{w}}, \hat{b})=\frac{1}{2 n} \sum_{i=1}^{n}\left[f\left(\boldsymbol{x}_{i} ; \boldsymbol{w}, b\right)-y_{i}\right]^{2} .
$$

最小二乘存在解析解, 可以用矩阵求逆的方式得出。但是实践中更常用数值优化算 法, 比如共轭梯度 (conjugate gradient) 等算法。这些算法首先随机或全零初始化 $\boldsymbol{w}$ 和 $b$, 然后用梯度迭代更新 $\boldsymbol{w}$ 和 $b$, 直到算法收敛。

- 第四, 预测。一旦我们学出了模型的参数, 就可以利用模型进行预测。给定一个房 屋的特征向量 $\boldsymbol{x}$, 模型对房价 $y$ 估计值为

$$
\widehat{y}=\boldsymbol{x}^{T} \widehat{\boldsymbol{w}}+\widehat{b} .
$$

监督学习的目的是让模型的估计值 $\widehat{y}$ 接近真实目标 $y$ 。

当模型参数数量较大, 而训练数据不够多的情况下, 常用正则化 (regularization) 缓 解过拟合 (overfitting)。加上正则项之后, 上述最小二乘模型变成:

$$
\min _{\boldsymbol{w}, b} L(\boldsymbol{w}, b)+\lambda R(\boldsymbol{w}) .
$$

其中的 $L(\boldsymbol{w}, b)$ 是损失函数, $R(\boldsymbol{w})$ 是正则项, $\lambda$ 是平衡损失函数和正则项的超参数。常 用的正则项有：

$$
R(\boldsymbol{w})=\|\boldsymbol{w}\|_{2}^{2} \quad \text { 和 } \quad R(\boldsymbol{w})=\|\boldsymbol{w}\|_{1} .
$$

前者对应岭回归 (ridge regression), 后者对应 LASSO (least absolute shrinkage and selection operator)。至于正则化系数 $\lambda$, 可以在验证数据集上利用交叉验证 (cross validation) 选 取。

### 2 逻辑斯蒂回归

上一小节介绍了回归问题, 其中的目标 $y$ 是连续变量, 例如房价的取值是连续的。本 小节研究二元分类问题 (binary classification), 其中的目标 $y$ 不是连续变量, 而是二元变 量, 表示输入数据的类别, 取值为 0 或 1 。逻辑斯蒂回归 (logistic regression) 是最常用 的二元分类器。

下面我们以疾病检测为例讲解二元分类问题。为了初步排查癌症, 需要做血检, 血 检中有 $d$ 项指标, 包括白细胞数量、含氧量、以及多种激素含量。一份血液样本的检测 报告作为一个 $d$ 维特征向量:

$$
\boldsymbol{x}=\left[x_{1}, x_{2}, \cdots, x_{d}\right]^{T} .
$$

医生需要基于 $\boldsymbol{x}$ 来初步判断该血检是否意味着患有癌症。如果医生的判断为 $y=1$, 则 要求病人做进一步检测; 如果医生的判断为 $y=0$, 则意味着末患癌症。这就是一个典型 的二元分类问题。是否可以让机器学习做这种二元分类呢?

![](https://cdn.mathpix.com/cropped/2023_02_03_f46f5cf0e4de5b9996dcg-013.jpg?height=351&width=1360&top_left_y=1184&top_left_x=288)

图 1.1: 线性 sigmoid 分类器的结构。输入是向量 $\boldsymbol{x} \in \mathbb{R}^{d}$, 输出是介于 0 和 1 之间的标量。

常用的是线性 sigmoid 分类器, 结构如图 $1.1$ 所示。基于输入向量 $\boldsymbol{x}$, 线性分类器做 出预测:

$$
f(\boldsymbol{x} ; \boldsymbol{w}, b) \triangleq \operatorname{sigmoid}\left(\boldsymbol{x}^{T} \boldsymbol{w}+b\right) .
$$

此处的 sigmoid 是个激活函数 (activation function), 定义为:

$$
\operatorname{sigmoid}(z) \triangleq \frac{1}{1+\exp (-z)} \text {. }
$$

如图 $1.2$ 所示, sigmoid 可以把任何实数映射到 0 到 1 之间。我们希望分类器的输出 $\widehat{y}=f(\boldsymbol{x} ; \boldsymbol{w}, b)$ 有这样的性质：如果 $\boldsymbol{x}$ 是癌症患者的血检数据, 那么 $\widehat{y}$ 接近 1 ; 如果 $\boldsymbol{x}$ 是 健康人的血检数据, 那么 $\widehat{y}$ 接近 0 。因此 $\widehat{y}$ 表示分类器有多大概率做出阳性的判断。比 如 $\widehat{y}=0.9$ 表示分类器有 $0.9$ 的概率判断血检为阳性; $\widehat{y}=0.05$ 表示分类器只有 $0.05$ 的 概率判断血检为阳性, 即 $0.95$ 的概率判断血检为阴性。

在讲解算法之前, 先介绍交叉熵 (cross entropy), 它常被用作分类问题的损失函数。 使用向量

$$
\boldsymbol{p}=\left[p_{1}, \cdots, p_{m}\right]^{T} \quad \text { 和 } \quad \boldsymbol{q}=\left[q_{1}, \cdots, q_{m}\right]^{T}
$$

表示两个 $m$-维的离散概率分布。向量的元素都非负, 且 $\sum_{j=1}^{m} p_{j}=1, \sum_{j=1}^{m} q_{j}=1$ 。它

![](https://cdn.mathpix.com/cropped/2023_02_03_f46f5cf0e4de5b9996dcg-014.jpg?height=508&width=734&top_left_y=243&top_left_x=724)

图 1.2: Sigmoid 函数的图像。

们之间的交叉熵定义为：

$$
H(\boldsymbol{p}, \boldsymbol{q})=-\sum_{j=1}^{m} p_{j} \cdot \ln q_{j} .
$$

熵（entropy）是交叉熵的一种特例: $H(\boldsymbol{p}, \boldsymbol{p})=-\sum_{j=1}^{m} p_{j} \cdot \ln q_{j}$ 是概率分布 $\boldsymbol{p}$ 的熵, 简 写为 $H(\boldsymbol{p})$ 。

与交叉熵类似的是 KL 散度 (Kullback-Leibler divergence), 也被称作相对熵（relative entropy), 用来衡量两个概率分布的区别有多大。对于离散分布, KL 散度的定义为

$$
\operatorname{KL}(\boldsymbol{p}, \boldsymbol{q})=\sum_{j=1}^{m} p_{j} \cdot \ln \frac{p_{j}}{q_{j}},
$$

这里约定 $\ln \frac{0}{0}=0$ 。 $\mathrm{KL}$ 散度总是非负, 而且 $\operatorname{KL}(\boldsymbol{p}, \boldsymbol{q})=0$ 当且仅当 $\boldsymbol{p}=\boldsymbol{q}$ 。这意味着两 个概率分布一致时, 它们的 KL 散度达到最小值。从 KL 散度和交叉熵的定义不难看出,

$$
\operatorname{KL}(\boldsymbol{p}, \boldsymbol{q})=H(\boldsymbol{p}, \boldsymbol{q})-H(\boldsymbol{p}) .
$$

由于熵 $H(\boldsymbol{p})$ 是不依赖于 $\boldsymbol{q}$ 的常数, 一旦固定 $\boldsymbol{p}$, 则 KL 散度等于交叉熵加上常数。如果 $\boldsymbol{p}$ 是固定的, 那么关于 $\boldsymbol{q}$ 优化 KL 散度等价于优化交叉熵。这就是为什么常用交叉熵作 为损失函数。

我们现在来讨论从训练数据中学习模型参数 $\boldsymbol{w}$ 和 $b$ 。

- 第一, 准备训练数据。收集 $n$ 份血检报告和最终的诊断, 作为训练数据集: $\left(\boldsymbol{x}_{1}, y_{1}\right)$, $\cdots,\left(\boldsymbol{x}_{n}, y_{n}\right)$ 。向量 $\boldsymbol{x}_{i} \in \mathbb{R}^{d}$ 表示第 $i$ 份血检报告中的所有指标。二元标签 $y_{i}=1$ 表 示患有癌症 (阳性), $y_{i}=0$ 表示健康 (阴性)。

- 第二, 把训练描述成优化问题。分类器对第 $i$ 份血检报告的预测是 $f\left(\boldsymbol{x}_{i} ; \boldsymbol{w}, b\right)$, 而 真实患癌情况是 $y_{i}$ 。想要用交叉熵衡量 $y_{i}$ 与 $f\left(\boldsymbol{x}_{i} ; \boldsymbol{w}, b\right)$ 之间的差别, 得把 $y_{i}$ 与 $f\left(\boldsymbol{x}_{i} ; \boldsymbol{w}, b\right)$ 表示成向量:

$$
\left[\begin{array}{c}
y_{i} \\
1-y_{i}
\end{array}\right] \text { 和 } \quad\left[\begin{array}{c}
f\left(\boldsymbol{x}_{i} ; \boldsymbol{w}, b\right) \\
1-f\left(\boldsymbol{x}_{i} ; \boldsymbol{w}, b\right)
\end{array}\right] \text {. }
$$

两个向量的第一个元素都对应阳性的概率, 第二个元素都对应阴性的概率。因为训 练样本的标签 $y_{i}$ 是给定的, 则两个向量尽量越接近, 它们的交叉熵越小。定义问 题的损失函数为交叉熵的均值：

$$
L(\boldsymbol{w}, b)=\frac{1}{n} \sum_{i=1}^{n} H\left(\left[\begin{array}{c}
y_{i} \\
1-y_{i}
\end{array}\right],\left[\begin{array}{c}
f\left(\boldsymbol{x}_{i} ; \boldsymbol{w}, b\right) \\
1-f\left(\boldsymbol{x}_{i} ; \boldsymbol{w}, b\right)
\end{array}\right]\right) .
$$

我们希望找到 $\boldsymbol{w}$ 和 $b$ 使得损失函数尽量小, 也就是让分类器的预测尽量准确。我 们可以考虑下面的优化问题:

$$
\min _{\boldsymbol{w}, b} L(\boldsymbol{w}, b)+\lambda R(\boldsymbol{w}) .
$$

这个优化问题称为逻辑斯蒂回归, 其中 $R(\boldsymbol{w})$ 是正则项, 比如 $\|\boldsymbol{w}\|_{2}^{2}$ 和 $\|\boldsymbol{w}\|_{1}$ 。

- 第三, 用数值优化算法求解。在建立优化模型之后, 需要寻找最优解 $(\widehat{\boldsymbol{w}}, \hat{b})$ 。通常 随机或全零初始化参数 $\boldsymbol{w}$ 和 $b$, 然后用梯度下降、随机梯度下降、Newton-Raphson、 L-BFGS 等算法迭代更新参数。

### Softmax 分类器

上一小节介绍了二元分类问题, 数据只分为两个类别, 比如患病和健康。本小节研 究多元分类 (multi-class classification) 问题, 数据可以划分为 $k(>2)$ 个类别。我们可 以用 softmax 分类器解决多分类问题。

本小节用 MNIST 手写数字识别为例 讲解多分类问题。如图 $1.3$ 所示, MNIST 数据集有 $n=60,000$ 个样本, 每个样本是 $28 \times 28$ 像素的图片。数据集有 $k=10$ 个 类别, 每个样本有一个类别标签, 它是介 于 0 到 9 之间的整数, 表示图片中的数字。 为了训练 softmax 分类器, 我们要对标签 做 one-hot 编码, 把每个标签（0 到 9 之间

![](https://cdn.mathpix.com/cropped/2023_02_03_f46f5cf0e4de5b9996dcg-015.jpg?height=422&width=688&top_left_y=1308&top_left_x=998)

图 1.3: MNIST 数据集中的图片。 的整数）映射到 $k=10$ 维的向量:

$$
\begin{array}{rlr}
0 & \Longrightarrow & {[1,0,0,0,0,0,0,0,0,0],} \\
1 & \Longrightarrow & {[0,1,0,0,0,0,0,0,0,0],} \\
\vdots & \Longrightarrow & {[0,0,0,0,0,0,0,0,1,0],} \\
8 & \Longrightarrow & {[0,0,0,0,0,0,0,0,0,1] .}
\end{array}
$$

把得到的标签记作 $\boldsymbol{y}_{1}, \cdots, \boldsymbol{y}_{n} \in\{0,1\}^{10}$ 。把每张 $28 \times 28$ 像素的图片拉伸成 $d=784$ 维 的向量, 记作 $\boldsymbol{x}_{1}, \cdots, \boldsymbol{x}_{n} \in \mathbb{R}^{784}$ 。

在介绍 softmax 分类器之前, 先介绍 softmax 激活函数。它的输入和输出都是 $k$ 维向 量。设 $\boldsymbol{z}=\left[z_{1}, \cdots, z_{k}\right]^{T}$ 是任意 $k$ 维实向量, 它的元素可正可负。Softmax 函数定义为

$$
\operatorname{softmax}(\boldsymbol{z}) \triangleq \frac{1}{\sum_{l=1}^{k} \exp \left(z_{l}\right)}\left[\exp \left(z_{1}\right), \exp \left(z_{2}\right), \cdots, \exp \left(z_{k}\right)\right]^{T}
$$

函数的输出是一个 $k$ 维向量, 元素都是非负, 且相加等于 1 。 如图 $1.4$ 所示, softmax 函数让最大的元素相对变得更大, 让小的元素接近 0 。图 $1.5$ 是 max 函数, 它把最大的元素映射到 1 , 其余所有元素映射到 0 。对比一下图 $1.4$ 和图 $1.5$, 不难看出 softmax 没有让小的元素严格等于零, 这就是为什么它的名字带有“soft”。
![](https://cdn.mathpix.com/cropped/2023_02_03_f46f5cf0e4de5b9996dcg-016.jpg?height=336&width=1382&top_left_y=488&top_left_x=401)

图 1.4: Softmax 函数把左边红色的 10 个数值映射到右边紫色的 10 个数值。
![](https://cdn.mathpix.com/cropped/2023_02_03_f46f5cf0e4de5b9996dcg-016.jpg?height=328&width=1372&top_left_y=980&top_left_x=410)

图 1.5: Max 函数把左边红色的 10 个数值映射到右边紫色的 10 个数值。

线性 softmax 分类器是线性函数 $+$ softmax 激活函数, 结构如图 $1.6$ 所示。具体来说, 线性 softmax 分类器定义为

$$
\boldsymbol{\pi}=\operatorname{softmax}(\boldsymbol{z}), \quad \text { 其中 } \quad \boldsymbol{z}=\boldsymbol{W} \boldsymbol{x}+\boldsymbol{b} \text {. }
$$

分类器的参数是矩阵 $\boldsymbol{W} \in \mathbb{R}^{k \times d}$ 和向量 $\boldsymbol{b} \in \mathbb{R}^{k}$, 这里的 $d$ 是输入向量的维度, $k$ 是类别 数量。有时为了避免模型的不唯一性, 需要限定 $\sum_{j=1}^{k} \boldsymbol{w}_{j:}=\mathbf{0}$ 和 $\sum_{j=1}^{k} b_{j}=0$ 。此处的 向量 $\boldsymbol{w}_{j}$ : 是矩阵 $\boldsymbol{W}$ 的第 $j$ 行, 实数 $b_{j}$ 是向量 $\boldsymbol{b}$ 的第 $j$ 个元素, 粗体 $\mathbf{0}$ 是全零向量。

Softmax 分类器输出向量 $\boldsymbol{\pi}$ 第 $j$ 个元素 $\pi_{j}$ 表示输入向量 $\boldsymbol{x}$ 属于第 $j$ 类的概率。在以 上 MNIST 手写数字识别的例子中, 假设分类器的输出是以下 10 维向量:

$$
\pi=[0.1,0.6,0.02,0.01,0.01,0.2,0.01,0.03,0.01,0.01]^{T} .
$$

可以这样理解该向量的元素:

- 第零号元素 $0.1$ 表示分类器以 $0.1$ 的概率判定图片 $\boldsymbol{x}$ 是数字“0”,

- 第一号元素 $0.6$ 表示分类器以 $0.6$ 的概率判定 $\boldsymbol{x}$ 是数字“1”,

- 第二号元素 $0.02$ 表示分类器只有 $0.02$ 的概率判定 $\boldsymbol{x}$ 是数字“2”,

以此类推。由于分类器的输出向量 $\boldsymbol{\pi}$ 的第一号元素 $0.6$ 是最大的, 分类器会判定图片 $\boldsymbol{x}$ 是数字“1”

我们做以下步骤, 从数据中学习模型参数 $\boldsymbol{W} \in \mathbb{R}^{k \times d}$ 和 $\boldsymbol{b} \in \mathbb{R}^{k}$ 。

- 第一, 准备训练数据。一共有 $n=60,000$ 张手写数字图片, 每张图片大小为 $28 \times 28$ 像素, 需要把图片变成 $d=784$ 维的向量, 记作 $\boldsymbol{x}_{1}, \ldots, \boldsymbol{x}_{n} \in \mathbb{R}^{d}$ 。每张图片有一个

![](https://cdn.mathpix.com/cropped/2023_02_03_f46f5cf0e4de5b9996dcg-017.jpg?height=366&width=1353&top_left_y=237&top_left_x=297)

图 1.6: 线性 Softmax 分类器的结构。输入是向量 $\boldsymbol{x} \in \mathbb{R}^{d}$, 输出是 $\boldsymbol{\pi} \in \mathbb{R}^{k}$ 。

标签, 它是 0 到 9 之间的整数, 需要把它做 one-hot 编码, 变成 $k=10$ 维的 one-hot 向量, 记作 $\boldsymbol{y}_{1}, \cdots, \boldsymbol{y}_{n}$ 。

- 第二, 把训练描述成优化问题。对于第 $i$ 张图片 $\boldsymbol{x}_{i}$, 分类器做出预测:

$$
\boldsymbol{\pi}_{i}=\operatorname{softmax}\left(\boldsymbol{W} \boldsymbol{x}_{i}+\boldsymbol{b}\right),
$$

它是 $k=10$ 维的向量, 可以反映出分类结果。我们希望 $\boldsymbol{\pi}_{i}$ 尽量接近真实标签 $\boldsymbol{y}_{i}$ (10 维的 one-hot 向量), 也就是希望交叉熵 $H\left(\boldsymbol{y}_{i}, \boldsymbol{\pi}_{i}\right)$ 尽量小。定义损失函数为平 均交叉熵 (即负对数似然函数):

$$
L(\boldsymbol{W}, \boldsymbol{b})=\frac{1}{n} \sum_{i=1}^{n} H\left(\boldsymbol{y}_{i}, \boldsymbol{\pi}_{i}\right) .
$$

我们希望找到参数矩阵 $\boldsymbol{W}$ 和向量 $\boldsymbol{b}$ 使得损失函数尽量小, 也就是让分类器的预测 尽量准确。定义下面的优化问题：

$$
\min _{\boldsymbol{W}, \boldsymbol{b}} L(\boldsymbol{W}, \boldsymbol{b})+\lambda R(\boldsymbol{W}),
$$

其中 $R(\boldsymbol{W})$ 是正则项。

- 第三, 用数值优化算法求解。在建立优化模型之后, 需要寻找最优解 $(\widehat{\boldsymbol{W}}, \hat{\boldsymbol{b}})$ 。通 常随机或全零初始化 $\boldsymbol{W}$ 和 $\boldsymbol{b}$, 然后用梯度下降、随机梯度下降等优化算法迭代更 新参数变量。

## $1.2$ 神经网络

本节简要介绍全连接神经网络和卷积神经网络, 并将它们用于多元分类问题。全连 接层和卷积层被广泛用于深度强化学习。循环层和注意力层也是常见的神经网络结构, 本书将在需要用到它们的地方详细讲解这两种结构。

### 1 全连接神经网络 (多层感知器)

接着上一节的内容, 我们继续研究 MNIST 手写识别这个多元分类问题。人类识别 手写数字的准确率接近 100\%, 然而线性 softmax 分类器对 MNIST 数据集识别只有 $90 \%$ 的准确率, 远低于人类的表现。线性分类器表现差的原因在于模型太小, 不能充分利用 $n=60,000$ 个训练样本。然而我们可以把 “线性函数 + 激活函数”这样的结构一层层堆积 起来, 得到一个多层网络, 获得更高的预测准确率。

全连接层 : 记输入向量为 $x \in \mathbb{R}^{d}$, 神经网络的一个层把 $x$ 映射到 $x^{\prime} \in \mathbb{R}^{d^{\prime}}$ 。全连接 层是这样定义的：

$$
\boldsymbol{x}^{\prime}=\sigma(\boldsymbol{z}), \quad \boldsymbol{z}=\boldsymbol{W} \boldsymbol{x}+\boldsymbol{b},
$$

其中权重矩阵 $\boldsymbol{W} \in \mathbb{R}^{d^{\prime} \times d}$ 和偏置向量 $\boldsymbol{b} \in \mathbb{R}^{d^{\prime}}$ 是该层的参数, 需要从数据中学习。 $\sigma(\cdot)$ 是激活函数, 比如 softmax 函数、sigmoid 函数、ReLU (rectified linear unit) 函数。最常用 的激活函数是 ReLU, 其定义为:

$$
\operatorname{ReLU}(\boldsymbol{z})=\left[\left(z_{1}\right)_{+},\left(z_{2}\right)_{+}, \ldots,\left(z_{d^{\prime}}\right)_{+}\right]^{T} .
$$

此处的 $\left[z_{i}\right]_{+}=\max \left\{z_{i}, 0\right\}$ 。我们称这整个结构为全连接层 (fully connected layer), 如图 $1.7$ 所示。

![](https://cdn.mathpix.com/cropped/2023_02_03_f46f5cf0e4de5b9996dcg-018.jpg?height=345&width=1220&top_left_y=1729&top_left_x=475)

图 1.7: 一个全连接层包括一个线性函数和一个激活函数。

全连接神经网络：我们可以把全连接层当做基本组件, 然后像搭积木一样搭建一个 全连接神经网络 (fully-connected neural network), 也叫多层感知器 (multi-layer perceptron, 缩写 MLP)。图 $1.8$ 展示了一个三层的全连接神经网络, 它把输入向量 $\boldsymbol{x}^{(0)}$ 映射到 $\boldsymbol{x}^{(3)}$ 。 一个 $\ell$ 层的全连接神经网络可以表示为:

$$
\begin{array}{ll}
\text { 第 1层: } & \boldsymbol{x}^{(1)}=\sigma_{1}\left(\boldsymbol{W}^{(1)} \boldsymbol{x}^{(0)}+\boldsymbol{b}^{(1)}\right), \\
\text { 第 2层: } & \boldsymbol{x}^{(2)}=\sigma_{2}\left(\boldsymbol{W}^{(2)} \boldsymbol{x}^{(1)}+\boldsymbol{b}^{(2)}\right),
\end{array}
$$



$$
\text { 第 } \ell \text { 层: } \quad \boldsymbol{x}^{(\ell)}=\sigma_{\ell}\left(\boldsymbol{W}^{(\ell)} \boldsymbol{x}^{(\ell-1)}+\boldsymbol{b}^{(\ell)}\right) \text {, }
$$

其中 $\boldsymbol{W}^{(1)}, \cdots, \boldsymbol{W}^{(\ell)}, \boldsymbol{b}^{(1)}, \cdots, \boldsymbol{b}^{(\ell)}$ 是神经网络的参数, 需要从训练数据学习。不同层 的参数是不同的。不同层的激活函数 $\sigma_{1}, \cdots, \sigma_{\ell}$ 可以相同, 也可以不同。

![](https://cdn.mathpix.com/cropped/2023_02_03_f46f5cf0e4de5b9996dcg-019.jpg?height=254&width=1442&top_left_y=650&top_left_x=250)

全连接神经网络 (多层感知器)

图 1.8: 由 3 个全连接层组成的神经网络, 每层有自己的参数。

编程实现：可以用 TensorFlow、PyTorch、Keras 等深度学习标准库实现全连接神经 网络, 只需要一、两行代码就能添加一个全连接层。添加一个全连接层需要用户指定两 个超参数：

-层的宽度 如果一个层是隐层 (即除了第 $\ell$ 层之外的所有层), 那么需要指定层的宽 度 (即输出向量的维度)。输出层 (即第 $\ell$ 层) 的宽度由问题本身决定。比如 MNIST 数据集有 10 类, 那么输出层的宽度必须是 10 。而对于二元分类问题, 输出层的宽 度是 1 。

- 激活函数 用户需要决定每一层的激活函数。对于隐层, 通常使用 ReLU 激活函 数。对于输出层, 激活函数的选择取决于具体问题。二元分类问题用 sigmoid, 多 元分类问题用 softmax，回归问题可以不用激活函数。

### 2 卷积神经网络

卷积神经网络 (convolutional neural network, 缩写 CNN) 是主要由卷积层组成的神 经网络1。卷积神经网络的结构如图 $1.9$ 所示。输入 $\boldsymbol{X}^{(0)}$ 是三阶张量 (tensor) ${ }^{2}$ 。卷积层的 输入和输出都是三阶张量, 每个卷积层之后通常有一个 ReLU 激活函数（图 $1.9$ 中没有画 出）可以把几个、甚至几十个卷积层累起来, 得到深度卷积神经网络。把最后一个卷积 层输出的张量转换为一个向量, 即向量化 (vectorization)。这个向量是 $\mathrm{CNN}$ 从输入的张 量中提取的特征。

本书不具体解释 CNN 的原理, 本书也不会用到这些原理。读者仅需要记住这个知识 点: $\mathrm{CNN}$ 的输入是矩阵或三阶张量, $\mathrm{CNN}$ 从该张量中提取特征, 输出提取的特征向量。 图片通常是矩阵（灰度图片）和三阶张量（彩色图片), 可以用 $\mathrm{CNN}$ 从中提取特征, 然 后用一个或多个全连接层做分类或回归。

${ }^{1} \mathrm{CNN}$ 中也可以有池化层 (pooling), 这里不做具体讨论。

2零阶张量为标量 (实数), 一阶张量为向量, 二阶张量为矩阵, 以此类推。

![](https://cdn.mathpix.com/cropped/2023_02_03_f46f5cf0e4de5b9996dcg-020.jpg?height=320&width=1385&top_left_y=240&top_left_x=404)

图 1.9: 神经网络由 3 个卷积层组成, 每层有自己的参数。

图 $1.10$ 是一个由卷积、全连接等层组成的深度神经网络。其中卷积网络从输入矩阵 (灰度图片) 中提取特征, 全连接网络把特征向量映射成 10 维向量, 最终的 softmax 激活 函数输出 10 维向量 $\boldsymbol{\pi}$ 。输出向量 $\boldsymbol{\pi}$ 的 10 个元素表示 10 个类别对应的概率, 可以反映 出分类结果。

![](https://cdn.mathpix.com/cropped/2023_02_03_f46f5cf0e4de5b9996dcg-020.jpg?height=277&width=1402&top_left_y=978&top_left_x=404)

图 1.10: 用于分类 MNIST 手写数字的深度神经网络。

## $1.3$ 反向传播和梯度下降

线性模型和神经网络的训练都可以描述成一个优化问题。设 $\boldsymbol{w}^{(1)}, \cdots, \boldsymbol{w}^{(\ell)}$ 为优化 参数 (可以是向量、矩阵、张量)。我们希望求解这样一个优化问题:

$$
\min _{\boldsymbol{w}^{(1)}, \cdots, \boldsymbol{w}^{(\ell)}} L\left(\boldsymbol{w}^{(1)}, \cdots, \boldsymbol{w}^{(\ell)}\right) .
$$

对于这样一个无约束的最小化问题, 最常使用的算法是梯度下降 (gradient descent, 缩写 GD）和随机梯度下降 (stochastic gradient descent, 缩写 SGD)。本节的内容包括梯度、梯 度算法、以及用反向传播计算梯度。

### 1 梯度下降

梯度 : 几乎所有常用的优化算法都需要计算梯度。目标函数 $L$ 关于一个变量 $\boldsymbol{w}^{(i)}$ 的梯度记作：

![](https://cdn.mathpix.com/cropped/2023_02_03_f46f5cf0e4de5b9996dcg-021.jpg?height=184&width=1184&top_left_y=1079&top_left_x=373)

由于目标函数的值是实数, 梯度 $\nabla_{\boldsymbol{w}^{(i)}} L$ 的形状与 $\boldsymbol{w}^{(i)}$ 完全相同。

-如果 $\boldsymbol{w}^{(i)}$ 是 $d$ 维向量, 那么 $\nabla_{\boldsymbol{w}^{(i)}} L$ 也是 $d$ 维向量;

- 如果 $\boldsymbol{w}^{(i)}$ 是 $d_{1} \times d_{2}$ 矩阵, 那么 $\nabla_{\boldsymbol{w}^{(i)}} L$ 也是 $d_{1} \times d_{2}$ 矩阵;

- 如果 $\boldsymbol{w}^{(i)}$ 是 $d_{1} \times d_{2} \times d_{3}$ 三阶张量, 那么 $\nabla_{\boldsymbol{w}^{(i)}} L$ 也是 $d_{1} \times d_{2} \times d_{3}$ 阶张量。 不论是自己手动推导梯度, 还是用程序自动求梯度, 都需要检查梯度的大小与参数变量 的大小是否相同; 如果不同, 梯度的计算肯定有错。

梯度下降 (GD) : 梯度是上升方向, 沿着梯度方向对优化参数 $\boldsymbol{w}^{(i)}$ 做一小步更新, 可 以让目标函数值增加。既然我们的目标是最小化目标函数, 就应该沿着梯度的负方向更 新参数, 这叫做梯度下降 (gradient descent, 缩写 GD)。设当前的参数值为 $\boldsymbol{w}_{\text {now }}^{(1)}, \cdots, \boldsymbol{w}_{\text {now }}^{(\ell)}$, 计算目标函数 $L$ 在当前的梯度, 然后做 GD 更新参数：

$$
\boldsymbol{w}_{\text {new }}^{(i)} \leftarrow \boldsymbol{w}_{\text {now }}^{(i)}-\alpha \cdot \nabla_{\boldsymbol{w}^{(i)}} L\left(\boldsymbol{w}_{\text {now }}^{(1)}, \cdots, \boldsymbol{w}_{\text {now }}^{(\ell)}\right), \quad \forall i=1, \cdots, \ell .
$$

此处的 $\alpha(>0)$ 叫做学习率 (learning rate) 或者步长 (step size), 它的设置既影响 GD 收 玫速度, 也影响最终神经网络的测试准确率, 所以 $\alpha$ 需要用户仔细调整。

随机梯度下降 (SGD)： 如果目标函数可以写成连加或者期望的形式, 那么可以用 SGD 求解最小化问题。假设目标函数可以写成 $n$ 项连加形式:

$$
L\left(\boldsymbol{w}^{(1)}, \cdots, \boldsymbol{w}^{(\ell)}\right)=\frac{1}{n} \sum_{j=1}^{n} F_{j}\left(\boldsymbol{w}^{(1)}, \cdots, \boldsymbol{w}^{(\ell)}\right) .
$$

函数 $F_{j}$ 隐含第 $j$ 个训练样本 $\left(\boldsymbol{x}_{j}, \boldsymbol{y}_{j}\right)$ 。每次随机从集合 $\{1,2, \cdots, n\}$ 中抽取一个数, 记 作 $j$ 。设当前的参数值为 $\boldsymbol{w}_{\text {now }}^{(1)}, \cdots, \boldsymbol{w}_{\text {now }}^{(\ell)}$, 计算此处的梯度, SGD 算法迭代过程为:

$$
\boldsymbol{w}_{\text {new }}^{(i)} \leftarrow \boldsymbol{w}_{\text {now }}^{(i)}-\alpha \cdot \underbrace{\nabla_{\boldsymbol{w}^{(i)}} F_{j}\left(\boldsymbol{w}_{\text {now }}^{(1)}, \cdots, \boldsymbol{w}_{\text {now }}^{(\ell)}\right)}_{\text {随机梯度 }}, \quad \forall i=1, \cdots, \ell .
$$

实际训练神经网络的时候, 总是用 SGD (及其变体), 而不用 GD。主要原因是 GD 用于 非凸问题会陷在鞍点 (saddle point), 收敛不到局部最优。而 SGD 和小批量 (mini-batch) SGD 可以跳出鞍点, 趋近局部最优。另外, GD 每一步的计算量都很大, 比 SGD 大 $n$ 倍, 所以 GD 通常很慢 (除非用并行计算)。

SGD 的变体 : 理论分析和实践都表明 SGD 的一些变体比简单的 SGD 收敛更快。这 些变体都基于随机梯度, 只是会对随机梯度做一些变换。常见的变体有 SGD+Momentum、 AdaGrad、Adam、RMSProp。能用 SGD 的地方就能用这些变体。因此, 本书中只用 SGD 讲解强化学习算法, 不去具体讨论 SGD 的变体。

### 2 反向传播

SGD 需要用到损失函数关于模型参数的梯度。对于一个深度神经网络, 我们利用反 向传播 (backpropagation, 缩写 BP) 求损失函数关于参数的梯度。如果用 TensorFlow 和 PyTorch 等深度学习平台, 我们可以不关心梯度是如何求出来的。只要定义的函数关于某 个变量可微, TensorFlow 和 PyTorch 就可以自动求该函数关于这个变量的梯度。

本节以全连接网络为例, 简单介绍反向传播的原理。全连接神经网络（忽略掉偏移 量 $\boldsymbol{b})$ 是这样定义的:

$$
\begin{array}{cc}
\text { 第 } 1 \text { 层: } & \boldsymbol{x}^{(1)}=\sigma_{1}\left(\boldsymbol{W}^{(1)} \boldsymbol{x}^{(0)}\right), \\
\text { 第 } 2 \text { 层: } & \boldsymbol{x}^{(2)}=\sigma_{2}\left(\boldsymbol{W}^{(2)} \boldsymbol{x}^{(1)}\right), \\
\vdots & \vdots \\
\text { 第 } \ell \text { 层: } & \boldsymbol{x}^{(\ell)}=\sigma_{\ell}\left(\boldsymbol{W}^{(\ell)} \boldsymbol{x}^{(\ell-1)}\right) .
\end{array}
$$

神经网络的输出 $\boldsymbol{x}^{(\ell)}$ 是神经网络做出的预测。设向量 $\boldsymbol{y}$ 为真实标签, 函数 $H$ 为交叉熵, 实数 $z$ 为损失:

$$
z=H\left(\boldsymbol{y}, \boldsymbol{x}^{(\ell)}\right)
$$

为了做梯度下降更新参数 $\boldsymbol{W}^{(1)}, \cdots, \boldsymbol{W}^{(\ell)}$, 我们需要计算损失 $z$ 关于每一个变量的梯度:

$$
\frac{\partial z}{\partial \boldsymbol{W}^{(1)}}, \quad \frac{\partial z}{\partial \boldsymbol{W}^{(2)}}, \quad \cdots, \quad \frac{\partial z}{\partial \boldsymbol{W}^{(\ell)}} .
$$

损失 $z$ 与参数 $\boldsymbol{W}^{(1)}, \cdots, \boldsymbol{W}^{(\ell)}$ 、变量 $\boldsymbol{x}^{(0)}, \boldsymbol{x}^{(1)}, \cdots, \boldsymbol{x}^{(\ell)}$ 的关系如图 $1.11$ 所示。

![](https://cdn.mathpix.com/cropped/2023_02_03_f46f5cf0e4de5b9996dcg-022.jpg?height=220&width=1376&top_left_y=2140&top_left_x=403)

图 1.11: 变量的函数关系。

反向传播的本质是求导的链式法则 (chain rule)。最简单的例子是单变量函数, 即变 量是 $x \in \mathbb{R}$ 。定义单变量复合函数 $h=g \circ f$, 即 $h(x)=g(f(x))$ 。那么根据链式法则, $h$ 关于 $x$ 的导数是 $h^{\prime}(x)=g^{\prime}(f(x)) f^{\prime}(x)$ 。接下来我们考虑多变量函数, 设 $\boldsymbol{f}: \mathbb{R}^{d} \rightarrow \mathbb{R}^{m}$, $\boldsymbol{g}: \mathbb{R}^{m} \rightarrow \mathbb{R}^{p}$, 和 $\boldsymbol{h}=\boldsymbol{g} \circ \boldsymbol{f}: \mathbb{R}^{d} \rightarrow \mathbb{R}^{p}$ 。我们讨论这种情况的链式法则。注意到 $\boldsymbol{f}=$ $\left[f_{1}, \ldots, f_{m}\right]^{T}$ 是向量值函数, 它关于 $\boldsymbol{x}=\left[x_{1}, \ldots, x_{d}\right]^{T} \in \mathbb{R}^{d}$ 的梯度 $\frac{\partial \boldsymbol{f}}{\partial \boldsymbol{x}}$ 是一个 $d \times m$ 矩 阵, 其中第 $(i, j)$ 元素为 $\partial f_{j} / \partial x_{i}$ (即 $f_{j}$ 关于 $x_{i}$ 的导数)。根据链式法则, 梯度可以写作:

$$
\frac{\partial \boldsymbol{h}}{\partial \boldsymbol{x}}=\frac{\partial \boldsymbol{f}}{\partial \boldsymbol{x}} \times \frac{\partial \boldsymbol{g}}{\partial \boldsymbol{f}}
$$

如果 $\boldsymbol{x}$ 是一个矩阵或张量, 我们需要先把它拉成一个等大小的向量, 比如一个 $d_{1} \times d_{2} \times d_{3}$ 三阶张量被向量化成 $d_{1} d_{2} d_{3}$ 维列向量, 然后利用上面的链式法则计算梯度。

现在可以用链式法则做反向传播, 计算损失 $z$ 关于神经网络参数的梯度。具体地, 首 先求出梯度 $\frac{\partial z}{\partial \boldsymbol{x}^{(\ell)}}$ 。然后做循环, 从 $i=\ell, \ldots, 1$, 依次做如下操作:

- 根据链式法则可得损失 $z$ 关于参数 $\boldsymbol{W}^{(i)}$ 的梯度：

$$
\frac{\partial z}{\partial \boldsymbol{W}^{(i)}}=\frac{\partial \boldsymbol{x}^{(i)}}{\partial \boldsymbol{W}^{(i)}} \cdot \frac{\partial z}{\partial \boldsymbol{x}^{(i)}} .
$$

这项梯度被用于更新参数 $\boldsymbol{W}^{(i)}$ 。

- 根据链式法则可得损失 $z$ 关于参数 $\boldsymbol{x}^{(i-1)}$ 的梯度:

$$
\frac{\partial z}{\partial \boldsymbol{x}^{(i-1)}}=\frac{\partial \boldsymbol{x}^{(i)}}{\partial \boldsymbol{x}^{(i-1)}} \cdot \frac{\partial z}{\partial \boldsymbol{x}^{(i)}} \text {. }
$$

这项梯度被传播到下面一层（即第 $i-1$ 层）, 继续循环。

反向传播的路径如图 $1.12$ 所示。只要知道损失 $z$ 关于 $\boldsymbol{x}^{(i)}$ 的梯度, 就能求出 $z$ 关于 $\boldsymbol{W}^{(i)}$ 和 $\boldsymbol{x}^{(i-1)}$ 的梯度。

![](https://cdn.mathpix.com/cropped/2023_02_03_f46f5cf0e4de5b9996dcg-023.jpg?height=223&width=1380&top_left_y=1405&top_left_x=272)

图 1.12: 反向传播的路径。

# 第 1 章 知识点

- 线性回归、逻辑斯蒂回归、softmax 分类器属于简单的线性模型。他们相当于线性 函数不加激活函数、加 sigmoid 激活函数、加 softmax 激活函数。三种模型分别应 用于回归问题、二元分类问题、多元分类问题。

- 全连接层的输入是向量, 输出也是向量。主要由全连接层组成的神经网络叫做全连 接网络, 也叫多层感知器 (MLP)。

- 卷积层有很多种, 本书中只用 2D 卷积层 (Conv2D), 它的输入是矩阵或三阶张量, 输出是三阶张量。主要由卷积层、池化层、全连接层组成的神经网络叫做卷积神经 网络 (CNN)。

- 在搭建神经网络的时候, 我们随机初始化神经网络参数, 然后通过求解优化问题来 学习参数。梯度下降及其变体（比如随机梯度下降、RMSProp、ADAM）是最常用 的优化算法, 它们用目标函数的梯度来更新模型参数。

- 对于线性模型, 我们可以轻易地求出梯度。然而神经网络是很复杂的函数, 无法直 接求出梯度, 而需要做反向传播。反向传播的本质是用链式法则求出目标函数关于 每一层参数的梯度。读者需要理解链式法则, 但无需掌握技术细节, TensorFlow 和 PyTorch 等深度学习平台都可以自动做反向传播, 不需要读者手动计算梯度。

#  第 1 章 习题

1. 假设你需要用深度神经网络估算房屋的价格。输入是房屋的属性, 包括面积、楼层、 地理位置等信息。输出是房屋的市场价格。请问神经网络的输出层应该用什么激活 函数?
A. ReLU 或者不用激活函数。
B. Sigmoid 或者 tanh。
C. Softmax 。

2. 假设你需要用深度神经网络判断人的性别；只考虑男性和女性。输入是人的头像。 输出是人的性别。请问神经网络的输出层应该用什么激活函数?
A. ReLU 或者不用激活函数。
B. Sigmoid 或者 tanh。
C. Softmax 。

3. 假设你需要用深度神经网络判别北京地区常见植物的种类。输入是植物的照片。输 出是植物的类别。请问神经网络的输出层应该用什么激活函数?
A. ReLU 或者不用激活函数。
B. Sigmoid 或者 $\tanh$ 。
C. Softmax 。

4. 定义全连接层 $\boldsymbol{z}=\boldsymbol{W} \boldsymbol{x}+\boldsymbol{b}$, 其中 $\boldsymbol{W}$ 是 $d_{\mathrm{out}} \times d_{\text {in }}$ 的矩阵, $\boldsymbol{b}$ 是 $d_{\mathrm{out}} \times 1$ 的向量。该 层的参数数量是 $d_{\mathrm{out}} \times d_{\mathrm{in}}+d_{\mathrm{out}}$ 。在实践中, 如果某个全连接层的参数数量过大, 我 们需要用 dropout 等方法对该层做正则。一个全连接网络的输入大小是 100 维, 三 个全连接层的输出大小分别是 500、1000、10 维。如果只能对其中一层做正则, 应 该对哪一层做正则?
A. 第一层（输出大小是 500 维）。
B. 第二层 (输出大小是 1000 维)。
C. 第三层 (输出大小是 10 维)。
