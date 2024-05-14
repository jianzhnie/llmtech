## IQL

IQL论文全称为：MultiAgent Cooperation and Competition with Deep Reinforcement Learning
 

多智能体环境中，状态转移和奖励函数都是受到所有智能体的联合动作的影响的。对于多智能体中的某个智能体来说，它的动作值函数是依据其它智能体采取什么动作才能确定的。因此对于一个单智能体来说它需要去了解其它智能体的学习情况。

这篇文章的贡献可能就是在于将DQN扩展到分散式的多智能体强化学习环境中，使其能够去处理高维复杂的环境。

IQL(Independent Q-Learning)算法中将其余智能体直接看作环境的一部分，也就是对于每个智能体a都是在解决一个单智能体任务，很显然，由于环境中存在其他智能体，因此环境是一个非稳态的，这样就无法保证收敛性，并且智能体会很容易陷入无止境的探索中，但是在工程实践上，效果还是比较可以的。独立的智能体网络结构可以参考下图所示：

![img](https://img-blog.csdnimg.cn/2021052710283251.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zOTA1OTAzMQ==,size_1,color_FFFFFF,t_70#pic_center)
