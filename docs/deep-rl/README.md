## 深度强化学习系列

### [入门教程](deep-rl/deep-rl-class/README.md)

- [第一章：深度强化学习简介](deep-rl/deep-rl-class/ch1_introduction.md)
- [第二章：Q-Learning](deep-rl/deep-rl-class/ch2_q-learning.md)
- [第三章：Deep Q-Learning](deep-rl/deep-rl-class/ch3_dqn.md)
- [第四章：Policy Gradient](deep-rl/deep-rl-class/ch4_pg.md)
- [第五章：Actor-Critic](deep-rl/deep-rl-class/ch5_a2c.md)
- [第六章：近端策略优化 (PPO)](deep-rl/deep-rl-class/ch6_ppo.md)
- [第七章：Decision Transformer](deep-rl/deep-rl-class/ch7_decision-transformer.md)
- [第八章：Multi-Agent RL](deep-rl/deep-rl-class/ch8_marl.md)
- [第九章：强化学习前沿主题](deep-rl/deep-rl-class/ch9_advanced.md)

### [进阶教程](deep-rl/algorithms/README.md)

- [Policy gradient theorem的证明](deep-rl/algorithms/ch1_supp_pg.md)
- [为什么A2C中减去 baseline 函数可以减小方差](deep-rl/algorithms/ch1_supp_a2c.md)
- [步步深入TRPO](deep-rl/algorithms/ch1_supp_trpo.md)
- [混合动作空间表征学习方法介绍（HyAR）](deep-rl/algorithms/ch2_supp_hyar.md)
- [为什么 PPO 需要重要性采样, 而 DDPG 这个 off-policy 算法不需要](deep-rl/algorithms/ch2_supp_ppovsddpg.md)
- [重参数化与强化学习](deep-rl/algorithms/ch2_supp_reparameterization.md)

### 强化学习环境

- [Awesome RL Envs](deep-rl/rltools/awesomeRLtools.md)
- [OpenAI Gym](deep-rl/envs/gym.md)
- [SMAC](deep-rl/envs/smac.md)
- [MARL Envs](deep-rl/envs/marl_env.md)
- [Interesting Environmrnt](deep-rl/envs/interesting_envs.md)

### 强化学习工具篇

- [强化学习代表人物/机构](deep-rl/rltools/awesome_rl.md)
- [EnvPool: 并行环境模拟器](deep-rl/rltools/envpool.md)
- [多智能体强化学习代码汇总](deep-rl/rltools/marltool.md)

### AlphaZero & MuZero & 蒙特卡洛树搜索

- [蒙特卡洛树搜索入门指南](deep-rl/muzero/mcts_guide.md)
- [蒙特卡洛树搜索(MCTS)详解](deep-rl/muzero/MCTS.md)
- [AlphaGoZero 算法介绍](deep-rl/muzero/alphazero.md)
- [MuZero算法介绍](deep-rl/muzero/muzero_intro.md)
- [MuZero伪代码](deep-rl/muzero/muzero_pseudocode.md)

### 多智能体强化学习

- [MARL](deep-rl/papers/Overview.md)
- [DRQN](deep-rl/papers/DRQN.md)
- [IQL](deep-rl/papers/IQL.md)
- [COMA](deep-rl/papers/COMA.md)
- [VDN](deep-rl/papers/VDN.md)
- [QTRAN](deep-rl/papers/QTRAN.md)
- [QMIX](deep-rl/papers/QMIX.md)
- [MADDPG](deep-rl/papers/MADDPG.md)
- [MAT](deep-rl/papers/MAT.md)

### PPOFamaily  决策智能公开课

**PPO × Family Vol.1** 系统性地讲解了决策智能的核心算法技术——**深度强化学习**，并深入浅出地介绍了最强大通用的算法 PPO。

[OpenDILab浦策：课程实录｜PPO × Family 第一课：开启决策 AI 探索之旅 （上）](https://zhuanlan.zhihu.com/p/604897017)

[OpenDILab浦策：课程实录｜PPO × Family 第一课：开启决策 AI 探索之旅 （下）](https://zhuanlan.zhihu.com/p/606954674)

**PPO × Family Vol.2——解构复杂动作空间**从决策输出设计的角度展开，介绍了 PPO 算法在四种动作空间上的各类技巧。

[OpenDILab浦策：课程实录｜PPO × Family 第二课：解构复杂动作空间（上）](https://zhuanlan.zhihu.com/p/612669849)

[OpenDILab浦策：课程实录｜PPO × Family 第二课：解构复杂动作空间（下）](https://zhuanlan.zhihu.com/p/622866136)

**PPO × Family Vol.3——表征多模态观察空间**，则将会从表征建模，从深度学习的角度进行展开，介绍观察空间的三部曲及衍生的“算法-代码-实践”知识。

[OpenDILab浦策：课程实录｜PPO × Family 第三课：表征多模态观察空间（上）](https://zhuanlan.zhihu.com/p/635313705)

[OpenDILab浦策：课程实录｜PPO × Family 第三课：表征多模态观察空间（下）](https://zhuanlan.zhihu.com/p/636696175)

而**PPO × Family Vol.4——解密稀疏奖励空间，将会进入到 MDP 的第三大核心元素——奖励函数，从指导智能体探索和利用的角度，介绍奖励空间上的“两朵乌云”及衍生的“算法-代码-实践”知识。**

[OpenDILab浦策：课程实录｜PPO × Family 第四课：解密稀疏奖励空间（上）](https://zhuanlan.zhihu.com/p/642656409)

[OpenDILab浦策：课程实录｜PPO × Family 第四课：解密稀疏奖励空间（中）](https://zhuanlan.zhihu.com/p/642812757)

[OpenDILab浦策：课程实录｜PPO × Family 第四课：解密稀疏奖励空间（下）](https://zhuanlan.zhihu.com/p/643047368)
