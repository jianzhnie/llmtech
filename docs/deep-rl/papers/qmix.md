# QMIX

## 多智能体强化学习的单调值函数因子分解

- *QMIX 论文链接： https://arxiv.org/pdf/1803.11485.pdf*
- *QMIX 实现代码：https://github.com/oxwhirl/pymarl*

### 摘要

QMIX 是一种基于 Value-Based 的多智能体强化学习算法（MARL），其基本思想来源于 Actor-Critic 与 DQN 的结合。使用中心式学习（Centralized Learning）分布式执行（Distributed Execution）的方法，利用中心式 Critic 网络接受全局状态用于指导 Actor 进行更新。QMIX 中 Critic 网络的更新方式和 DQN 相似，使用 TD-Error 进行网络自更新。除此之外，QMIX 中为 Critic 网络设立了 evaluate net 和 target net， 这和 DQN 中的设计思想完全相符。

QMIX 具有如下特点： 

- 学习得到分布式策略。 
- 本质是一个值函数逼近算法。 
- 由于对一个联合动作-状态只有一个总奖励值，而不是每个智能体都有一个自己的奖励值，因此只能用于合作环境，而不能用于竞争对抗环境。
-  QMIX算法采用中心式训练，分布式执行的框架。通过集中式的学习，得到每个智能体的分布式策略。

- 训练时借用全局状态信息来提高算法效果。是对VDN方法的改进。
- QMIX设计一个神经网络来整合每个智能体的局部值函数而得到联合动作值函数，VDN是直接求和。 
- 每个智能体的局部值函数只需要自己的局部观测，因此整个系统在执行时是一个分布式的，通过局部值函数，选出累积期望奖励最大的动作执行。
- 算法使联合动作值函数与每个局部值函数的单调性相同，因此对局部值函数取最大动作也就是使联合动作值函数最大。 
- 算法针对的模型是一个分布式多智能体部分可观马尔可夫决策过程。

##  相关研究

### 多智能体强化学习核心问题

多智能体强化学习（MARL）训练中面临的最大问题是：**训练阶段和执行阶段获取的信息可能存在不对等问题。** 即，在训练的时候我们可以获得大量的全局信息（事实证明，只有获取足够的信息模型才能被有效训练）。

但在最终应用模型的时候，我们是无法获取到训练时那么多的全局信息的，因此，人们提出两个训练网络：

- 一个为中心式训练网络（Critic），该网络只在训练阶段存在，获取全局信息作为输入并指导 Agent 行为控制网络（Actor）进行更新；
- 另一个为行为控制网络（Actor），该网络也是最终被应用的网络，在训练和应用阶段都保持着相同的数据输入。

在多智能体强化学习中一个关键的问题就是如何学习联合动作值函数，因为该函数的参数会随着智能体数量的增多而成指数增长，如果动作值函数的输入空间过大，则很难拟合出一个合适函数来表示真实的联合动作值函数。另一个问题就是学得了联合动作值函数后，如何通过联合值函数提取出一个优秀的分布式的策略。这其实是单智能体强化学习拓展到MARL的**核心问题**。

### Dec-POMDP

多智能体部分可观测马尔科夫过程

Dec-POMDP是将POMDP拓展到多智能体系统。每个智能体的局部观测信息 $o_{i,t}$ ，动作 $a_{i,t}$ ，系统状态为 $s_t$ 。其主要新定义了几个概念，简要介绍几个主要的：

- 每个智能体的动作-观测历史可表示为  $\tau_i=(a_{i,0},o_{i,1},\cdots,a_{i,t-1},o_{i,t})$，表示从初始状态开始，该智能体的时序动作-观测记录，
- 联合动作-观测历史 $\tau=(\tau_1,\cdots,\tau_n)$ 表示从初始状态开始，所有智能体的时序动作-观测记录。则每个智能体的分布式策略为 $\pi_i(\tau_i)$ ，其值函数为 $Q_i(\tau_i,a_i;\theta_i)$ 都是跟动作-观测历史 $\tau_i$ 有关，而不是跟状态有关。

### DRQN (Deep Recurrent Q-Learning)

在部分可观察的环境中，智能体可以受益于对其整个动作观察历史的运用。DRQN是一个用来处理POMDP（部分可观马尔可夫决策过程）的一个算法，其采用LSTM替换DQN卷基层后的一个全连接层，来达到能够记忆历史状态的作用，因此可以在部分可观的情况下提高算法性能. 由于QMIX解决的是多智能体的POMDP问题，因此每个智能体采用的是DRQN算法。

###  IQL ( Independent Q-Lerning)

多智能体学习中最常用的方法是 IQL， IQL非常暴力的给每个智能体执行一个Q-learning算法,  因为共享环境，并且环境随着每个智能体策略、状态发生改变，对每个智能体来说，环境是动态不稳定的，因此这个算法也无法收敛，但是， 在实践中，无论在混合和竞争性游戏中， IQL一般具有较好的效果，通常作为一个非常强大的基准。

### VDN (Value decomposition network)

VDN（value decomposition networks）也是采用对每个智能体的值函数进行整合，得到一个联合动作值函数。令 $\tau=(\tau_1,\cdots,\tau_n)$表示联合动作-观测历史，其中 $\tau_i=(a_{i,0},o_{i,1},\cdots,a_{i,t-1},o_{i,t})$ 为动作-观测历史，$a=(a_1,\cdots,a_n)$ 表示联合动作。 $Q_{tot}$ 为联合动作值函数， $Q_i(\tau_i,a_i;\theta_i)$  为智能体i的局部动作值函数，局部值函数只依赖于每个智能体的局部观测。VDN采用的方法就是直接相加求和的方式.

$$Q_{tot}=\sum_{i=1}^{n}Q_i(\tau_i,a_i,;\theta_i) $$

虽然 $Q_i(\tau_i,a_i;\theta_i)$ 不是用来估计累积期望回报的，但是这里依然叫它为值函数。分布式的策略可以通过对每个 $Q_i(\tau_i,a_i;\theta_i)$取max得到。

## QMIX

### 研究痛点

VDN 将每个智能体的局部动作值函数直接求和相加得到联合动作值函数，虽然满足联合值函数与局部值函数单调性相同的可以进行分布化策略的条件，但是其没有在学习时利用状态信息以及没有采用非线性方式对单智能体局部值函数进行整合，限制住了团队价值函数的复杂性表达，而且没有利用到全局的状态信息。使得VDN算法还有很大的提升空间。

### 创新点及贡献

在 VDN 算法的基础上，对从单智能体价值函数到团队价值函数之间的映射关系进行了改进，在映射的过程中将原来的线性映射换为非线性映射，并通过超网络的引入将额外状态信息加入到映射过程，提高了模型性能。

###  QMIX 解决了什么问题（Motivation）

QMIX 是一种解决多智能体强化学习问题的算法，对于大多数多智能体强化学习问题（MARL）都面临着同样一个问题：**信度分配（也叫回报分配）**。

回报分配通常分为两种类型： 自下而上类型 和 自上而下类型。

自上而下类型：这种类型通常指我们只能拿到一个团队的最终得分，而无法获得每一个 Agent 的独立得分，因此我们需要把团队回报（Team Reward）合理的分配给每一个独立的 Agent（Individual Reward），这个过程通常也叫 “独立回报分配”（Individual Reward Assign）。上述例子就属于这种类型，典型的代表算法为 COMA算法。

自下而上类型：另外一种类型恰恰相反，指当我们只能获得每个 Agent 的独立回报（Individual）时，如何使得整个团队的团队得分（Team Reward）最大化。

 QMIX 算法解决的是上述第二种类型的问题，即，在获得各 Agent 的独立回报的情况下，**如何使得整个团队的团队收益最大化问题**。

### 主要思路

QMIX采用一个混合网络对单智能体局部值函数进行合并，并在训练学习过程中加入全局状态信息辅助，来提高算法性能。

为了能够沿用VDN的优势，利用集中式的学习，得到分布式的策略。主要是因为对联合动作值函数取 argmax 等价于对每个局部动作值函数取 argmax ，其单调性相同，如下所示

$${\rm argmax}_uQ_{tot}(\tau,u)=\left( \begin{aligned} {\rm argmax}_{u_1}&Q_1(\tau_1,u_1) \\ &\vdots\\ {\rm argmax}_{u_n}&Q_n(\tau_n,u_n) \\ \end{aligned} \right)\qquad $$ (1) 

QMIX 通过提出单调性假设放松了 VDN 中对单智能体的价值函数直接求和等于联合价值函数的约束限制，

因此分布式策略就是贪心的通过局部 $Q_i $获取最优动作。QMIX将(1)转化为一种单调性约束，如下所示：

$$\frac{\partial Q_{tot}}{\partial Q_i}\ge 0, \forall i\in \{1,2,\cdots,n\} $$    其中 $Q_i$ 为单智能体的价值函数，$Q_{tot}$ 为联合价值函数。 

若满足以上单调性，则(1)成立，为了实现上述约束，QMIX采用混合网络（mixing network）来实现，其具体结构如下所示.

![img](https://liushunyu.github.io/img/in-post/2020-06-18-%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E8%AE%BA%E6%96%87%EF%BC%889%EF%BC%89QMIX.assets/image-20200618104937306.png)

其主要结构与 VDN 类似，重点修改在于引入将额外状态信息加入到单智能体的价值函数到联合价值函数的映射过程，并将其称为 mixing network。

图(b)表示整体的 qmix 网络结构, 由 agent 网络 和 mixing 网络组成。

图(a)表示混合网络的结构。其输入为每个DRQN网络的输出。为了满足上述的单调性约束，混合网络的所有权值都是非负数，对偏移量不做限制，这样就可以确保满足单调性约束。

图(c) 表示Agent 网络结构，每个智能体采用一个DRQN来拟合自身的Q值函数 $Q_i(\tau_i,a_i;\theta_i)$ ，DRQN循环输入当前的观测 $o_{i,t}$ 以及上一时刻的动作 $a_{i,t-1}$ 来得到Q值。

混合网络最后一层的偏移量通过两层网络以及ReLU激活函数得到非线性映射网络。由于状态信息 $s_t$ 是通过超网络混合到 $Q_{tot}$ 中的，而不是仅仅作为混合网络的输入项，这样带来的一个好处是，如果作为输入项则 $s_t$ 的系数均为正，这样则无法充分利用状态信息来提高系统性能，相当于舍弃了一半的信息量。

为了能够更多的利用到系统的状态信息 $s_t$ ，采用一种超网络（hypernetwork）。

- hypernetwork 将状态 $s_t$ 作为输入，输出为混合网络的权值及偏移量。
- 为了保证权值的非负性，采用一个线性网络以及绝对值激活函数保证输出不为负数。
- 对偏移量采用同样方式但没有非负性的约束。最后的 bias 使用了用 ReLU 作激活函数的两层 hypernetwork。

QMIX最终的代价函数为:

$$L(\theta)=\sum_{i=1}^b[(y_i^{tot}-Q_{tot}(\tau,a,s;\theta))^2] $$

更新用到了传统的DQN的思想，其中b表示从经验记忆中采样的样本数量， 

$$y^{tot}=r+\gamma \max_{a'} \overline Q(\tau',a',s';\overline \theta) $$， $$Q(\tau',a',s';\overline \theta)$$ 表示目标网络。

由于满足上文的单调性约束，对 $Q_{tot} $ 进行 $argmax$ 操作的计算量就不在是随智能体数量呈指数增长了，而是随智能体数量线性增长，极大的提高了算法效率。

## 代码实现

### Agent RNN Network

QMIX 中每一个 Agent 都由 RNN 网络控制，在训练时你可以为每一个 Agent 个体都训练一个独立的 RNN 网络，同样也可以所有 Agent 复用同一个 RNN 网络，这取决于你自己的设计。

RNN 网络一共包含 3 层，输入层（MLP）→ 中间层（GRU）→ 输出层（MLP），实现代码如下：

```python
import torch.nn as nn
import torch.nn.functional as F

class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h
```

### Mixing Network

Mixing 网络相当于 Critic 网络，同时接收 Agent RNN Network 的 Q 值和当前全局状态 $s_t$ ，输出在当前状态下所有 Agent 联合行为的行为效用值 $Q_{tot}$

```python
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QMixer(nn.Module):
    def __init__(self, args):
        super(QMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.mixing_embed_dim

        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim))
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        # First layer
        w1 = th.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot
```



## 示例

原文中给了一个小示例来说明QMIX与VND的效果差异，虽然QMIX也不能完全拟合出真实的联合动作值函数，但是相较于VDN已经有了很大的提高。

如下图为一个两步合作矩阵博弈的价值矩阵

![img](https://pic1.zhimg.com/v2-0782a8562fbdb13278703090bd69f714_b.jpg)

在第一阶段，只有智能体 1  的动作能决定第二阶段的状态。在第一阶段，如果智能体 1 采用动作 AAA 则跳转到上图 ${State 2A}$ 状态，如果智能体 1 采用动作 B 则跳转到上图State 2B 状态，第二阶段的每个状态的价值矩阵如上两图所示。

现在分别用VDN与QMIX学习上述矩阵博弈各个状态的值函数矩阵，得到结果如下图所示

![img](https://pic2.zhimg.com/v2-92a2f7f0fbd1192bdd2d629f0f0da905_b.jpg)

(a)为VDN拟合结果，(b)为QMIX拟合结果。可以从上图看出，VDN的结果是智能体 1 在第一阶段采用动作 A ，显然这不是最佳状态，而QMIX是智能体 1 在第一阶段采用动作 B ，得到了最大的累积期望奖励。由上可得QMIX的逼近能力比VDN更强，QMIX算法的效果更好。

## 实验

1、在 Two-Step Game 上进行实验表明 QMIX 的逼近能力比 VDN 更强，QMIX 算法的效果更好。

2、在 Decentralised StarCraft Micromanagement 环境中进行实验

- 将环境修改为部分可观察环境

3、在消融实验中发现在同构智能体智能体中不一定需要非线形值函数分解，而在异构智能体中需要使用额外状态信息及非线性值函数分解才能实现更好的性能。

## 其他补充

1、论文中提出当任何一个智能体的最佳动作独立于其他智能体在同一时间采取的动作的价值函数无法进行成功的分解，因此不能由 QMIX 进行表示。

2、论文将团队价值函数的等式约束推广到了单调性约束上，而且利用额外状态信息使用了 hypernetwork 来学习网络权重，这种学习权重的方式感觉比较新颖。

## Reference

- https://liushunyu.github.io/2020/06/18/
- https://blog.csdn.net/qq_38638132/article/details/114177729
- https://www.zhihu.com/search?type=content&q=QMIX