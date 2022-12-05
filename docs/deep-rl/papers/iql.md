  一个完全合作式的多智能体任务(我们有n个智能体，这n个智能体需要相互配合以获取最大奖励)可以描述为去中心化的部分可观测马尔可夫决策模型(Dec-POMDP)，通常用一个元组G 来表示：$$
G=⟨S,U,P,r,Z,O,n,γ⟩$$

  其中$s∈S$表示环境的真实状态信息。在每一个时间步，对于每个智能体$$a∈A≡{1,…,n}$$都需要去选择一个动作$u^{a} \in U$去组成一个联合动作$u∈U≡U^n$  ，再将这个联合动作给到环境中去进行状态转移：$$P(s 
′
 ∣s,u):S×U×S→[0,1]$$。之后，所有的智能体都会收到一个相同的奖励：$r(s,u):S×U→R$。与单智能体一样$$γ$$表示折扣因子。

  对于每个单智能体a 来说，它接收的是一个独立的部分可观测的状态 $z∈Z$，不同的智能体 a 具备不同的观测，但是所有的观测都来自环境的真实状态信息，所以可以用函数表示为：$$O(s,a):S×A→Z$$。对于每个智能体 a 它都有一个动作观测历史$τ 
a
 ∈T≡(Z×U)^* $
 ，基于这个动作-观测的历史来构建随机策略函数 $π^a ( u^a ∣ τ^a ) : T × U → [ 0 , 1 ] $。联合动作策略 $π$是基于状态信息 $s_t$构建的联合动作值函数$Q^{\pi}\left(s_{t}, \mathbf{u}_{t}\right)=\mathbb{E}_{s_{t+1: \infty}, \mathbf{u}_{t+1: \infty}}\left[R_{t} \mid s_{t}, \mathbf{u}_{t}\right]$, 其中 $R_{t}=\sum_{i=0}^{\infty} \gamma^{i} r_{t+i}$ 是折扣回报。

## IQL

IQL论文全称为：MultiAgent Cooperation and Competition with Deep Reinforcement Learning
 

多智能体环境中，状态转移和奖励函数都是受到所有智能体的联合动作的影响的。对于多智能体中的某个智能体来说，它的动作值函数是依据其它智能体采取什么动作才能确定的。因此对于一个单智能体来说它需要去了解其它智能体的学习情况。

  这篇文章的贡献可能就是在于将DQN扩展到分散式的多智能体强化学习环境中吧，使其能够去处理高维复杂的环境。

  作者采用的环境是雅塔丽的Pong环境。作者基于不同的奖励函数设计来实现不同的多智能体环境。在竞争环境下，智能体期望去获取比对方更多的奖励。在合作的环境下，智能体期望去寻找到一个最优的策略去保持上图中的白色小球一直在游戏中存在下去。

  为了测试分散式DQN算法的性能，作者只通过奖励函数的设计就构建出来了不同的多智能体范式环境：

- 完全竞争式：胜利方奖励+1，失败方奖励-1，是一个零和博弈。
- 完全合作式：在这个环境设定下，我们期望这个白色小球在环境中存在的时间越长越好，如果一方失去了球，则两方的奖励都是-1。
- 非完全竞争式：在完全竞争和完全合作式中，失去小球的一方奖励都式-1，对于胜利一方，作者设置一个系数 $\rho \in [-1, 1]$来看参数的改变对实验结果的影响。
    

IQL(Independent Q-Learning)算法中将其余智能体直接看作环境的一部分，也就是对于每个智能体a都是在解决一个单智能体任务，很显然，由于环境中存在智能体，因此环境是一个非稳态的，这样就无法保证收敛性，并且智能体会很容易陷入无止境的探索中，但是在工程实践上，效果还是比较可以的。

  独立的智能体网络结构可以参考下图所示：

![img](https://img-blog.csdnimg.cn/2021052710283251.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zOTA1OTAzMQ==,size_1,color_FFFFFF,t_70#pic_center)



## VDN
VDN论文全称为：Value-Decomposition Networks For Cooperative Multi-Agent Learning
  

在合作式多智能体强化学习问题中，每个智能体基于自己的局部观测做出反应来选择动作，来最大化团队奖励。对于一些简单的合作式多智能体问题，可以用中心式(centralized)的方法来解决，将状态空间和动作空间做一个拼接，从而将问题转换成一个单智能体的问题。这会使得某些智能体在其中滥竽充数。

另一种极端方式式训练独立的智能体，每个智能体各玩各的，也不做通信，也不做配合，直接暴力出奇迹。这种方式对于每个智能体来说，其它智能体都是环境的一部分，那么这个环境是一个非平稳态的(non-stationary)，理论上的收敛性是没法证明的。还有一些工作在对每个智能体都基于其观测设计一个奖励函数，而不是都用一个团队的团队奖励，这种方式的难点在于奖励函数的设计，因为设计的不好很容易使其陷入局部最优。

  VDN中提出一种通过反向传播将团队的奖励信号分解到各个智能体上的这样一种方式。其网络结构如下图所示：

![img](https://img-blog.csdnimg.cn/20210527093123262.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zOTA1OTAzMQ==,size_1,color_FFFFFF,t_70#pic_center)

  先看上图中的图1，画的是两个独立的智能体，因为对每个智能体来说，观测都是部分可观测的，所以Q函数是被定义成基于观测历史数据所得到的$Q\left(h_{t}, a_{t}\right)$，实际操作的时候直接用RNN来做就可以。图2说的就是联合动作值函数由各个智能体的值函数累加得到的：$$Q\left(\left(h^{1}, h^{2}, \ldots, h^{d}\right),\left(a^{1}, a^{2}, \ldots, a^{d}\right)\right) \approx \sum_{i=1}^{d} \tilde{Q}_{i}\left(h^{i}, a^{i}\right)$$

其中d表示d 个智能体，$\tilde{Q}_{i}$由每个智能体的局部观测信息得到，$ \tilde{Q}_{i} $是通过联合奖励信号反向传播到各个智能体的 $\tilde{Q}_{i} $上进行更新的。这样各个智能体通过贪婪策略选取动作的话，也就会使得联合动作值函数最大。

  总结来说：值分解网络旨在学习一个联合动作值函数 $$Q_{t o t}(\tau, \mathbf{u})$$，其中 $ \tau \in \mathbf{T} \equiv \mathcal{T}$ 是一个联合动作-观测的历史轨迹，u是一个联合动作。它是由每个智能体a 独立计算其值函数 $Q_{a}\left(\tau^{a}, u^{a} ; \theta^{a}\right)$，之后累加求和得到的。其关系如下所示：$$Q_{t o t}(\tau, \mathbf{u})=\sum_{i=1}^{n} Q_{i}\left(\tau^{i}, u^{i} ; \theta^{i}\right)$$

  严格意义上说 $$Q_{a}$$称作值函数可能不太准确，因为它并没有严格估计期望回报。

  值分解的独立的智能体网络结构可以参考下图所示：

![img](https://img-blog.csdnimg.cn/20210527103005830.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zOTA1OTAzMQ==,size_1,color_FFFFFF,t_70#pic_center)

  如果在此基础上在加上底层的通信的话可以表示为如下形式(其实就是将各个智能体的观测给到所有的智能体)：

![img](https://img-blog.csdnimg.cn/20210527103111397.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zOTA1OTAzMQ==,size_1,color_FFFFFF,t_70#pic_center)

  如果是在高层做通信的话可以得到如下形式：

![img](https://img-blog.csdnimg.cn/20210527103321579.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zOTA1OTAzMQ==,size_1,color_FFFFFF,t_70#pic_center)

  如果是在底层加上高层上都做通信的话可以得到如下形式：

![img](https://img-blog.csdnimg.cn/20210527103702795.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zOTA1OTAzMQ==,size_1,color_FFFFFF,t_70#pic_center)

  如果是集中式的结构的话，可以表示为如下形式：

![img](https://img-blog.csdnimg.cn/20210527103513555.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zOTA1OTAzMQ==,size_1,color_FFFFFF,t_70#pic_center)

