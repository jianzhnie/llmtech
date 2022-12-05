# QTRAN

QTRAN论文：QTRAN: Learning to Factorize with Transformation for Cooperative Multi-Agent  Reinforcement learning

代码：https://github.com/Sonkyunghwan/QTRAN

之前说的VDN和QMIX都是值分解领域的两大标杆性文章，并且在一般的工程项目实战上VDN和QMIX的效果就还是比较好的，不过是在论文中的效果有被弱化，更具体地可以参考看一下这篇文章：RIIT: Rethinking the Importance of Implementation Tricks in Multi-Agent Reinforcement Learning。但是这篇QTRAN从理论层面还是值得分析一下的：

  值分解地文章其实就是在保证联合动作取argmax的时候能够是对各个智能体的值函数取argmax。VDN中采用的是线性求和的方式，QMIX中采用的是保证单调性。对于一些任务，比如像联合动作的最优就是各个智能体的单独的最优的值函数，这样的问题可以采用这种方式，对于不是这类型的问题的话，这种限制就太强了。QTRAN提出了一种新的值分解的算法。

不管这个值分解如何分解，其实它们都需要取保证一个东西：

$${\rm argmax}_uQ_{tot}(\tau,u)=\left( \begin{aligned} {\rm argmax}_{u_1}&Q_1(\tau_1,u_1) \\ &\vdots\\ {\rm argmax}_{u_n}&Q_n(\tau_n,u_n) \\ \end{aligned} \right)\qquad $$ (1) 

`VDN`和`QMIX`中的线性求和保证单调性都能够去保证上述条件的成立，也就是给了两个充分条件，但不是必要条件。为了之后更好的理论分析，在`QTRAN`中对上述这个东西做了一个定义`IGM` (`Individual-Global-Max`)：

- IGM定义：对于一个联合动作值函数 $Q_{\mathrm{jt}}: \mathcal{T}^{N} \times \mathcal{U}^{N} \mapsto \mathbb{R}$，其中 ${\tau} \in \mathcal{T}^{N}$
   是一个联合动作观测的历史轨迹。如果对于独立的智能体存在一个动作值函数 $\left[Q_{i}: \mathcal{T} \times \mathcal{U} \mapsto \mathbb{R}\right]_{i=1}^{N}$，满足如下关系式的话：

$${\rm argmax}_uQ_{jt}(\tau,u)=\left( \begin{aligned} {\rm argmax}_{u_1}&Q_1(\tau_1,u_1) \\ &\vdots\\ {\rm argmax}_{u_n}&Q_n(\tau_n,u_n) \\ \end{aligned} \right)\qquad $$ (2) 

我们就说在轨迹${\tau}$下，$\left[Q_{i}\right]$对 $Q_{\mathrm{jt}}$满足IGM条件。$Q_{\mathrm{jt}}(\boldsymbol{\tau}, \boldsymbol{u})$是能够被 $\left[Q_{i}\left(\tau_{i}, u_{i}\right)\right] $分解的，$\left[Q_{i}\right]$ 是 $Q_{\mathrm{jt}}$ 的分解因子。QTRAN中将原始的$\left[Q_{i}\right]$映射成一个新的$Q_{\mathrm{jt}}^{\prime}$

## QTRAN直观理解

在VDN和QMIX中是将 $$ Q_{\mathrm{jt}}$$通过累加求和和保证单调性的方式来分解的，作者这里提出一种更加鲁棒的分解方式，将原始的 $$Q_{\mathrm{jt}}$$ 映射成 $Q_{\mathrm{jt}}^{\prime}$ ，通过 $Q_{\mathrm{jt}}^{\prime}$去分解值函数到各个子智能体上，来保证学到的$$Q_{\mathrm{jt}}^{\prime}$$与真实的动作值函数 $Q^{*}$ 非常接近。这样在学习真实的动作值函数的时候，没有像VDN和QMIX那样对其加上一些累加求和和保证单调性的限制，所以它能学地更好。

  但是由于部分可观测地限制，这个 $Q_{\mathrm{jt}}^{\prime}$ 是没有办法用来进行具体地决策的，所以我们需要去找到$Q_{\mathrm{jt}}$、$Q_{\mathrm{jt}}^{\prime}$

 和$\left[Q_{i}\right]$三者之间的关系。

可分解值函数的充分条件
  由于不提供累加求和和单调性来保证可分解，QTRAN提出了一个满足IGM定义的充分条件：当动作值函数 $$Q_{\mathrm{jt}}(\boldsymbol{\tau}, \boldsymbol{u})$$和 $\left[Q_{i}\left(\tau_{i}, u_{i}\right)\right]$满足下面这个关系式时，我们认为它是可分解的：

其中 $V_{\mathrm{jt}}(\tau)=\max _{\boldsymbol{u}} Q_{\mathrm{jt}}(\boldsymbol{\tau}, \boldsymbol{u})-\sum_{i=1}^{N} Q_{i}\left(\tau_{i}, \bar{u}_{i}\right)$

