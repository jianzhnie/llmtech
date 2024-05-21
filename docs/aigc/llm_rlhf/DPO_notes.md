# DPO 模型的推导

## 概述

RLHF 的训练流程复杂而且不稳定；首先我们训练一个奖励模型以反应人类偏好，然后利用强化学习微调语言模型来最大化估计奖励，在过程中约束微调的模型不可以偏离原始模型太多。

相比之下，DPO优化人类偏好，同时不需要使用强化学习。DPO直接优化最能满足偏好的策略，使用简单的分类目标，拟合一个隐式奖励模型，其相应的最优策略可以以封闭形式提取。

## 预备知识

### KL散度

KL散度也称为相对熵（Relative Entropy），是衡量两个概率分布差异的一种方法。它是两个概率分布 P 和 Q 之间的非对称距离度量，定义为：
$$
D_{KL}(P||Q)=\sum_xP(x)\log\left(\frac{P(x)}{Q(x)}\right)
$$
其中，P 是数据的真实分布，而 Q  是模型或估计分布。KL散度的值总是非负的，当且仅当 P 和 Q 完全相同时，KL散度为零。

### Bradley-Terry模型

下面通过一个例子介绍  Bradley-Terry模型 如何对比较关系进行建模：

|        | win  | Loss |
| ------ | ---- | ---- |
| A vs B | 8    | 4    |
| A vs C | 3    | 5    |

 在这个例子中， A 和 B 对战， 胜 8场，输4场， A和C 对战， 胜 3场， 输5场。问题是 B 和 C 对战，获胜的几率有多大？

这个问题可以通过  Bradley-Terry模型建模。

Bradley-Terry 模型假设每个个体都有一个隐含的实力参数 $\alpha$ ,   $\alpha_{i}$  代表个体 i 的正实值分数，${\displaystyle P(i>j)} $ 代表 i 战胜 j 的概率。
$$
{\displaystyle P(i>j)={\frac {\alpha_{i}}{\alpha_{i}+\alpha_{j}}}}
$$
我们可以通过 MLE  对参数 $\alpha$ 进行求解。
$$
L = 8 \ln(\frac {\alpha_A} {\alpha_A + \alpha_B}) + 4 \ln(\frac {\alpha_B} {\alpha_A + \alpha_B}) + 3 \ln(\frac {\alpha_A} {\alpha_A + \alpha_C}) + 5 ln(\frac {\alpha_C} {\alpha_A + \alpha_C})
$$
计算得到： $\alpha_A = 1 $ ,  $\alpha_B = \frac{1}{2} $ ,  $\alpha_C = \frac{5}{3}  $ ,  从而：  $ P(B > C) = \frac{\alpha_B}{\alpha_C + \alpha_c} \approx 0.23 $ ，根据现有数据， B	和C 对战，获胜的几率大概为 0.23.

不使用 MLE， 我们也可以使用机器学习的方式通过迭代优化的方式来进行求解， 上述问题的一般的Loss 函数可表示成：
$$
Loss = - \mathbb{E}_{(\alpha_x,\alpha_y) \sim D} \left[ ln (\frac {\alpha_x} {\alpha_x + \alpha_y})  \right]
$$
可以看到，这就是一般分类问题的交叉熵损失函数的样式， 优化的目标损失函数的值越小越好。 而其中 $ \frac {\alpha_x} {\alpha_x + \alpha_y}  $代表 x 战胜 y 的概率,  优化的目标变成 x 战胜 y 的概率约趋近于 1越好。

## RLHF 研究

现有的 RLHF 流程通常包括三个阶段：1）监督微调（SFT）；2）偏好采样和奖励学习；3）RL优化。

<img src="https://miro.medium.com/v2/resize:fit:700/1*GiEF7F3n-1TlL7_HRJD_OA.png" alt="img" style="zoom:150%;" />



### SFT

RLHF通常以对感兴趣下游任务（对话、摘要等）的高质量数据进行监督学习来微调预训练的LM开始，以获得模型 $ \pi_{SFT} $。

### Reward Model

在 RLHF 的第二阶段， 我们需要训练一个 Reward 模型来为生成的结果打分。大模型的输入的Prompt 为 x,  输出的回答为y，回答的好坏可以通过 Reward 模型打分。
$$
p(y_w \succ y_l | x) = \frac{r(x, y_w)}{r(x, y_w) + r(x, y_l)}
$$
 Reward 模型有可能返回负数， 因此我们加上一个指数函数变换， 从而得到 BT 模型中，人类偏好分布 $ p $ 的建模：
$$
p(y_w \succ y_l | x) = \frac{\exp(r(x, y_w))}{\exp(r(x, y_w)) + \exp(r(x, y_l))}
$$
假设我们可以访问来自 $ p $ 的比较静态数据集$ D = \{x(i), y(i)_w, y(i)_l\}^{N}_{i=1} $ 的样本，我们可以通过最大似然估计参数化奖励模型$ r_\phi(x, y) $。将问题框架为二元分类，我们得到了优化Reward 模型的负对数似然损失函数：
$$
L_{R}(r_{\theta}; D) = - \mathbb{E}_{(x,y_w,y_l) \sim D} \left[ log \sigma\left( r_{\theta}(x, y_l) - r_{\theta}(x, y_w) \right)  \right]
$$

$$
=- \mathbb{E}_{(x,y_w,y_l) \sim D}\left [ log \frac{\exp(r(x, y_w))}{\exp(r(x, y_w)) + \exp(r(x, y_l))} \right]
$$

$$
=- \mathbb{E}_{(x,y_w,y_l) \sim D}\left [ log \frac{1}{1 + \exp(r(x, y_w)- r(x, y_l))} \right]
$$

$$
=- \mathbb{E}_{(x,y_w,y_l) \sim D}\left [ log \sigma(r(x, y_w)- r(x, y_l))\right]
$$

其中， $\sigma(x) = \frac  {1} {1 + exp(-x)}$ 为 sigmoid 函数。

在LMs的背景下，网络$ r_\phi(x, y) $ 通常从SFT模型$ \pi_{SFT}(y | x) $ 初始化，顶层增加一个线性层，产生单个标量预测奖励值。为了确保具有较低方差的奖励函数，以前的工作对奖励进行了归一化，使得对所有 $ x $， 有 $ \mathbb{E}_{x,y} \sim D [r_\phi(x, y)] = 0 $ 。

### RL微调阶段

在RL阶段，我们使用学习到的奖励函数为语言模型提供反馈。具体来说，我们构建了以下优化问题：

$$
\max_{\pi} \mathbb{E}_{x \sim D, y \sim \pi} \left[ r_\phi(x, y) - \beta D_{KL} \pi(y | x) || \pi_{ref}(y | x) \right]
$$
其中$ \beta $ 是一个控制基础模型参考策略$ \pi_{ref} $（即初始SFT模型$ \pi_{SFT} $）的参数。在实践中，语言模型策略$ \pi_\theta $ 也初始化为$ \pi_{SFT} $。增加的KL 散度约束很重要，因为它防止模型偏离奖励模型准确的分布，同时保持生成多样性并防止模式崩溃到单一高奖励答案。

RL微调阶段的目标是：我们希望找到一个能够最大化奖励的策略，同时我们也希望该策略与初始未优化策略的行为不能相差太大。

## Direct Preference Optimization（DPO）

DPO 完全消除了对奖励模型的需要.

<img src="https://miro.medium.com/v2/resize:fit:648/1*f1LfCLncMIyhQiorWSKf7A.png" alt="img" style="zoom:150%;" />



DPO 的策略目标为：

$$
L_{DPO}(\pi_{\theta}; \pi_{ref}) = -\mathbb{E}_{(x,y_w,y_l) \sim D} \left[ \log \sigma \left( \beta \log \frac{\pi_{\theta}(y_w | x)}{\pi_{ref}(y_w | x)} - \beta \log \frac{\pi_{\theta}(y_l | x)}{\pi_{ref}(y_l | x)} \right) \right]
$$

求解上述优化问题的最有解为：
$$
\pi^*(y | x) =   {\frac{1}{Z(x) } \pi_{ref}(y|x) \exp \left(\frac{1}{\beta} r(x, y) \right)}
$$
其中：$Z(x) = \sum_y \pi_{ref}(y | x) \exp\left(\frac{1}{\beta} r(x, y)\right)$

重新排列$\pi^*(y | x)$方程式(通过对等式两边取对数和代数运算)，得到奖励函数：
$$
r^*(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)
$$


### DPO更新的作用是什么？

为了深入理解DPO，分析损失函数 $ L_{DPO} $ 的梯度是有用的。梯度相对于参数 $ \theta $ 可以写成：

$$
\nabla_{\theta} L_{DPO}(\pi_{\theta}; \pi_{ref}) = -\beta \mathbb{E}_{(x,y_w,y_l) \sim D} \left[ \sigma\left( \hat{r}_{\theta}(x, y_l) - \hat{r}_{\theta}(x, y_w) \right) \left( \nabla_{\theta} \log \pi(y_w | x) - \nabla_{\theta} \log \pi(y_l | x) \right) \right]
$$

其中 $ \hat{r}_{\theta}(x, y) = \beta \log \frac{\pi_{\theta}(y | x)}{\pi_{ref}(y | x)} $ 是由语言模型 $ \pi_{\theta} $ 和参考模型 $ \pi_{ref} $ 隐式定义的奖励（更多内容见第5节）。

当 $ \hat{r}_{\theta}(x, y)  >  \hat{r}_{\theta}(x, y) $,  算式 $\sigma \left( \hat{r}_{\theta}(x, y_l) - \hat{r}_{\theta}(x, y_w) \right) $ 越接近1，反之接近0，若两者趋近则该权重接近0.5。

直观上，损失函数 $ L_{DPO} $ 的梯度增加了首选回答 $ y_w $ 的可能性，并减少了非首选回答 $ y_l $ 的可能性。

### DPO概述

一般DPO流程如下：

1. 对于每个提示 $ x $，从参考模型 $ \pi_{ref}(- | x) $ 中采样完成 $ y_1, y_2 $，用人类偏好标注以构建离线偏好数据集 $ D = \{(x(i), y(i)_w, y(i)_l)\}^{N}_{i=1} $。
2. 优化语言模型 $ \pi_{\theta} $ 以最小化给定 $ \pi_{ref} $ 和 $ D $ 以及期望的 $ \beta $ 的 $ L_{DPO} $。

在实践中，使用公开可用的偏好数据集，而不是生成样本和收集人类偏好。偏好数据集是使用 $ \pi_{SFT} $ 采样得到。

当可以取得$ \pi_{SFT} $， 我们初始化 $ \pi_{ref} = \pi_{SFT} $。

当 $ \pi_{SFT} $ 不可用时，我们通过最大似然估计偏好目标 $ (x, y_w) $ 的似然来初始化 $ \pi_{ref} $，即 $ \pi_{ref} = \arg \max_{\pi} \mathbb{E}_{x,y_w \sim D} [\log \pi(y_w | x)] $。

这个过程有助于减少真实参考分布（不可用）和DPO使用的 $ \pi_{ref} $ 之间的分布偏移。

##  DPO  推导

### 推导 DPO 的训练目标

$$
\max_{\pi} \mathbb{E}_{x \sim D, y \sim \pi} \left[ r(x, y) - \beta D_{KL} \pi(y | x) || \pi_{ref}(y | x) \right]
$$

$$
= \max_{\pi} \mathbb{E}_{x \sim D} \mathbb{E}_{y \sim \pi(y|x)} \left[ r(x, y) - \beta \log \frac{\pi(y|x)}{\pi_{ref}(y|x)} \right]
$$

$$
= \min_{\pi} \mathbb{E}_{x \sim D} \mathbb{E}_{y \sim \pi(y|x)} \left[ \log \frac{\pi(y|x)}{\pi_{ref}(y|x)} - \frac{1}{\beta} r(x, y) \right]
$$

进一步，将上式化简得到：
$$
\min_{\pi} \mathbb{E}_{x \sim D} \mathbb{E}_{y \sim \pi(y|x)} \left[ \log \frac{\pi(y|x)}{\pi_{ref}(y|x)} - \frac{1}{\beta} r(x, y) \right]
$$

$$
= \min_{\pi} \mathbb{E}_{x \sim D} \mathbb{E}_{y \sim \pi(y|x)} \left[ \log \frac{\pi(y|x)}{\pi_{ref}(y|x)} - log \exp \left(\frac{1}{\beta} r(x, y) \right) \right]
$$

$$
= \min_{\pi} \mathbb{E}_{x \sim D} \mathbb{E}_{y \sim \pi(y|x)} \left[ \log \frac{\pi(y|x)}{\pi_{ref}(y|x) \exp \left(\frac{1}{\beta} r(x, y) \right)}  \right]
$$

$$
= \min_{\pi} \mathbb{E}_{x \sim D} \mathbb{E}_{y \sim \pi(y|x)} \left[ \log \frac{\pi(y|x)}{\pi_{ref}(y|x) \exp \left(\frac{1}{\beta} r(x, y) \right) \frac{1}{Z(x) } Z(x)  }  \right]
$$

$$
= \min_{\pi} \mathbb{E}_{x \sim D} \mathbb{E}_{y \sim \pi(y|x)} \left[ \log \frac{\pi(y|x)}{\frac{1}{Z(x) } \pi_{ref}(y|x) \exp \left(\frac{1}{\beta} r(x, y) \right)} - log(Z(x))\right]
$$

接下来，令：
$$
Z(x) = \sum_y \pi_{ref}(y | x) \exp\left(\frac{1}{\beta} r(x, y)\right)
$$
Z(x) 称为划分函数， 注意划分函数仅是 $ x $ 和参考策略 $ \pi_{ref} $ 的函数，不依赖于策略 $ \pi $。
$$
{\frac{1}{Z(x) } \pi_{ref}(y|x) \exp \left(\frac{1}{\beta} r(x, y) \right)}
$$

$$
= \frac {\pi_{ref}(y|x) \exp \left(\frac{1}{\beta} r(x, y) \right)} { \sum_y \pi_{ref}(y | x) \exp\left(\frac{1}{\beta} r(x, y)\right)}
$$

$$
= \pi^*(y | x)
$$

这是一个有效的概率分布，因为对于所有 $ y $，$ \pi^*(y | x) \geq 0 $ 且 $ \sum_y \pi^*(y | x) = 1 $。从而：
$$
\min_{\pi} \mathbb{E}_{x \sim D} \mathbb{E}_{y \sim \pi(y|x)} \left[ \log \frac{\pi(y|x)}{\frac{1}{Z(x) } \pi_{ref}(y|x) \exp \left(\frac{1}{\beta} r(x, y) \right)} - log(Z(x))\right]
$$

$$
= \min_{\pi} \mathbb{E}_{x \sim D} \mathbb{E}_{y \sim \pi(y|x)} \left[ \log \frac{\pi(y|x)} {\pi^*(y | x)} - log(Z(x))\right]
$$

现在，由于 $ Z(x) $ 不依赖于 $ \pi $，最小值由最小化第一项KL散度的策略实现。
$$
= \min_{\pi} \mathbb{E}_{x \sim D} \mathbb{E}_{y \sim \pi(y|x)} \left[ \log \frac{\pi(y|x)} {\pi^*(y | x)} \right]
$$

$$
= \min_{\pi} \mathbb{E}_{x \sim D} \left[D_{KL} \pi(y | x) || \pi^*(y | x) \right]
$$

Gibbs不等式告诉我们，KL散度最小化为0当且仅当两个分布相同时。因此我们有最优解：当分布 $\pi(y | x)$ 和 $\pi^*(y | x)$相等时， KL 散度的值最小，从而我们优化的目标
$$
\pi(y | x) = \pi^*(y | x) =   {\frac{1}{Z(x) } \pi_{ref}(y|x) \exp \left(\frac{1}{\beta} r(x, y) \right)}
$$
进一步：
$$
\pi(y | x) = \pi^*(y | x) =   {\frac{1}{Z(x) } \pi_{ref}(y|x) \exp \left(\frac{1}{\beta} r(x, y) \right)}
$$

$$
=> \exp \left(\frac{1}{\beta} r(x, y) \right) = \frac{\pi(y|x)} {\pi_{ref}(y | x)} Z(x)
$$

$$
=> r(x, y)  = \beta log \left( \frac{\pi(y|x)} {\pi_{ref}(y | x)} Z(x) \right)
$$

$$
=> r(x, y)  = \beta log  \frac{\pi(y|x)} {\pi_{ref}(y | x)} +  \beta log Z(x)
$$

根据 Bradley-Terry模型， 对于比较关系建模的损失函数可表示为：
$$
log  \left (\sigma(r(x, y_w)- r(x, y_l)) \right)
$$

$$
= log \sigma \left( \beta log  \frac{\pi(y_{w}|x)} {\pi_{ref}(y_{w} | x)} +  \beta log Z(x) - \beta log  \frac{\pi(y_{l}|x)} {\pi_{ref}(y_{l} | x)} - \beta log Z(x) \right)
$$

$$
= log \sigma \left( \beta log  \frac{\pi(y_{w}|x)} {\pi_{ref}(y_{w} | x)} - \beta log  \frac{\pi(y_{l}|x)} {\pi_{ref}(y_{l} | x)} \right)
$$

这样，我们就得到了 最终的 DPO 损失函数：
$$
L_{DPO} = log \sigma \left( \beta log  \frac{\pi(y_{w}|x)} {\pi_{ref}(y_{w} | x)} - \beta log  \frac{\pi(y_{l}|x)} {\pi_{ref}(y_{l} | x)} \right)
$$
因此，不需要优化奖励函数，我们就能优化最优策略。

### 在 Bardley-Terry模型下推导DPO目标

在Bradley-Terry 偏好模型下推导DPO目标是直接的，我们有
$$
p^*(y_1 \succ y_2 | x) = \frac{\exp(r^*(x, y_1))}{\exp(r^*(x, y_1)) + \exp(r^*(x, y_2))}
$$

上面，我们展示了可以将（不可用的）真实奖励通过其相应的最优策略表示：

$$
r^*(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)
$$


将方程(39)代入方程(38)我们得到：

$$
p^*(y_1 \succ y_2 | x) = \frac{\exp  \left( \beta \log \frac{\pi^*(y_1|x)}{\pi_{ref}(y_1|x)} + \beta \log Z(x) \right)}
{\exp \left( \beta \log \frac{\pi^*(y_1|x)}{\pi_{ref}(y_1|x)} + \beta \log Z(x)\right) + \exp \left( \beta \log \frac{\pi^*(y_2|x)}{\pi_{ref}(y_2|x)} + \beta \log Z(x)\right)}
$$

$$
= \frac{1}{1 + \exp(\beta \log \frac{\pi^*(y_2|x)}{\pi_{ref}(y_2|x)} - \beta \log \frac{\pi^*(y_1|x)}{\pi_{ref}(y_1|x)})}
$$

$$
= \sigma\left(\beta \log \frac{\pi^*(y_1|x)}{\pi_{ref}(y_1|x)} - \beta \log \frac{\pi^*(y_2|x)}{\pi_{ref}(y_2|x)}\right)
$$
