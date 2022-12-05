# COMA：Counterfactual Multi-Agent Policy Gradients

论文链接：[Counterfactual Multi-Agent Policy Gradients, AAAI 2017](https://arxiv.org/pdf/1705.08926.pdf)

代码链接：[github链接](https://github.com/oxwhirl/pymarl)

##  问题

文章针对multi-agent设置下存在的三个挑战进行了算法设计：

- Modelling other agents’ information：在multi-agent的设置下，过去常用的independent actor-critic等模型，往往会由于独立训练，导致信息共享不足，无法做到较好的coordinated，从而效果不佳。
- Multi-agent credit assignment：常规的actor-critic方法由于每个actor训练的reward都是基于全局的reward，所以很难评估每个agent采取的action实际对全局的reward影响有多大，故而导致优化存在困难。
- Curse of dimensionality in action space：由于系统是mulit-agent，在要求算法有更好效果的情况下，往往需要求得不同agent采取action的联合概率分布，故而也导致了动作空间规模指数上升的问题。

## 解法

VDN和QMIX 都采用了这样一种结构：中心化计算系统的QQ函数，单智能体的QiQi函数去中心化。这种结构使得策略在训练的时候可以利用全局信息，同时每个智能体仍然只接受局部信息作为输入。类似的，在Actor-Critic框架中，我们同样可以通过分散Actor，中心化Critic来达到同样的效果，COMA就采用了这样的思想。此外，COMA还考虑了另外一个问题：如何合理的分配全局系统奖励（credit assignment）。每个智能体对全局奖励的贡献是不同的，在更新Actor网络时，应该考虑到这种不同，分配合理的奖励给不同的Actor。

###  Centralised critic

Centralised critic，也就是集中训练一个critic作为全局的critic，下图是它的具体结构:

![dS2hEq.png](https://s1.ax1x.com/2020/08/13/dS2hEq.png)

Critic仅仅在learning过程中使用，它可以基于所有joint action和state information进行训练。当global state St 存在的时候，它直接利用它进行训练，否则使用joint action-observation history  ut,ot 进行训练。而对于actor，它在learning和execution的时候都需要，它的训练仅仅依靠来自自己的action-observation history。

在这种设置下的critic能够得到来自全局的信息，而在actor-critic框架中，往往会利用critic来指导每个agent的学习，将全局的信息传输到每个agent，从而提高每个agent对其他agent的信息的建模能力。

### Counterfactual baseline

作为一个critic，它的核心功能是辅助policy gradient训练，我们可以通过两种方式使用它：

#### Naive actor-critic

最简单的方式，就是和原始的actor-critic一样，直接利用它估计TD-error，并用于计算梯度。

$$g=\nabla_{\theta \pi} \log \pi\left(u | \tau_{t}^{a}\right)\left(r+\gamma V\left(s_{t+1}\right)-V\left(s_{t}\right)\right)$$

这样做的方法是无法解决credit assignment的问题：TD-error考虑的是global reward的影响，故而对于每个actor而言，难以显式确认它对于global reward贡献，从而导致优化方向不正确。

####  Difference Reward

它核心思想的Per-agent shaped reward的定义如下：

$$D^{a}=r(s, \mathbf{u})-r\left(s,\left(\mathbf{u}^{-a}, c^{a}\right)\right)$$

其中，ca是人为指定的默认action。我们将其他agent的action固定，只研究当前agent的action变化的影响。这个Difference reward，顾名思义也就是对reward作difference，而difference的对象分别是当前agent选择的action的reward和当前agent被人为指定的action的reward。从而得到当前agent选择的action相对默认action的优势。那么如果agent选择了能够提高D的action，那么这个action对global reward也是有贡献的，这样也就部分解决了credit assignment的问题。

然而这种方法存在如下两个局限：

- 需要额外的simulation估计 $$r(s,(u−a,ca)) $$。
- 需要用户对每个agent都指定一个默认的动作 ca

#### Counterfactual baseline

为了解决以上两个问题，本文引入了counterfactual baseline。counterfactual部分表示它继承了difference reward的思想，这体现了对credit assignment的解决方向，baseline则体现了它在policy gradient中的作用。

$$A^{a}(s, \mathbf{u})=Q(s, \mathbf{u})-\sum_{y^{\prime} a} \pi^{a}\left(u^{\prime a} | \tau^{a}\right) Q\left(s,\left(\mathbf{u}^{-a}, u^{\prime a}\right)\right)$$

这个函数计算的是，固定其他智能体的动作时，一个智能体的某个动作的Q值与所有动作Q值平均值间的差。这里aa表示某个智能体,uu表示所有智能体的动作，u−au−a表示除了aa之外的所有智能体的动作。

方程中的第一项是当前选择的action的global q-value，这表明在centrailsed critic中估计的是Q值。方程中的第二项，表征在agent aa的所有可能选择状态下能够获得的global q-value的期望，也就是常规理解的baselines。两者做差也正是体现了当前agent选择的agent相对平均结果的优势。

这样做可以解决2.2.2中的两个问题：

- 使用centralised critic替代了原式中的reward，这个critic是直接从数据中学习的，从而避免了对reward的多余simulations。
- 对于当前agent的所有动作求期望，用来衡量是否有所提升，避免了人工设计action的问题。

### Efficient critic representation

考虑到counterfactual baseline的实际实现，存在维度诅咒的问题：它需要所有的joint action的Q-value，如果使用神经网络，假设agent的action数量是U，一共有n个agent，那么就需要 $|U|^n$个输出节点，计算消耗太大。

为了简化计算，Critic不会计算所有智能体动作组合的Q值（输出维度是 $|U|^n$ 个），而是针对每一个智能体，将其他智能体的动作结合其他历史信息当作输入，输出该智能体每个动作对应的Q值（输出维度降为|U|）。

![dS2Wbn.png](https://s1.ax1x.com/2020/08/13/dS2Wbn.png)

算法流程如下：

![dS24U0.png](https://s1.ax1x.com/2020/08/13/dS24U0.png)

## 实验内容

在星际环境上做实验，效果好于Independent Actor Critic等方式。

## 缺点

- 论文的方法未对比一些其他集中训练分步执行的算法，对比的baselines都比较弱；

## 优点

- 引入反事实推理的概念非常具有创新性，且给出了方法理论上的证明；
- 感觉确实是从问题本身出发，一步步解决各种上到顶层算法设计下到降低算法复杂度的各种方法