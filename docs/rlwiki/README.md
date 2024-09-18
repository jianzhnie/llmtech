## 深度强化学习系列

### [入门教程](rlwiki/hfrlclass/README.md)

- [第一章：深度强化学习简介](rlwiki/hfrlclass/ch1_introduction.md)
- [第二章：Q-Learning](rlwiki/hfrlclass/ch2_q-learning.md)
- [第三章：Deep Q-Learning](rlwiki/hfrlclass/ch3_dqn.md)
- [第四章：Policy Gradient](rlwiki/hfrlclass/ch4_pg.md)
- [第五章：Actor-Critic](rlwiki/hfrlclass/ch5_a2c.md)
- [第六章：近端策略优化 (PPO)](rlwiki/hfrlclass/ch6_ppo.md)
- [第七章：Decision Transformer](rlwiki/hfrlclass/ch7_decision-transformer.md)
- [第八章：Multi-Agent RL](rlwiki/hfrlclass/ch8_marl.md)
- [第九章：强化学习前沿主题](rlwiki/hfrlclass/ch9_advanced.md)

### [进阶教程](rlwiki/algorithms/README.md)

- [Policy gradient theorem的证明](rlwiki/algorithms/ch1_supp_pg.md)
- [为什么A2C中减去 baseline 函数可以减小方差](rlwiki/algorithms/ch1_supp_a2c.md)
- [步步深入TRPO](rlwiki/algorithms/ch1_supp_trpo.md)
- [混合动作空间表征学习方法介绍（HyAR）](rlwiki/algorithms/ch2_supp_hyar.md)
- [为什么 PPO 需要重要性采样, 而 DDPG 这个 off-policy 算法不需要](rlwiki/algorithms/ch2_supp_ppovsddpg.md)
- [重参数化与强化学习](rlwiki/algorithms/ch2_supp_reparameterization.md)

### 强化学习环境

- [Awesome RL Envs](rlwiki/envs/awesome_envs.md)
- [OpenAI Gym](rlwiki/envs/gym.md)
- [机器人仿真 Mujoco](rlwiki/envs/mujoco.md)
- [星际微操 SMAC](rlwiki/envs/smac.md)
- [MARL Envs](rlwiki/envs/marl_env.md)
- [多智能体仿真 PettingZoo](rlwiki/envs/pettingzoo.md)
- [网络安全模拟 Awesome Cyber Env](rlwiki/envs/cyber_env_summary.md)
- [网络安全模拟 CyberWheel](rlwiki/envs/cyberwheel_env.md)
- [网络安全模拟 YarwningTitan](rlwiki/envs/cyber_yawningtitan_env.md)


### 强化学习工具篇

- [强化学习代表人物/机构](rlwiki/rltools/awesome_rl.md)
- [EnvPool: 并行环境模拟器](rlwiki/rltools/envpool.md)
- [多智能体强化学习代码汇总](rlwiki/rltools/marltool.md)


### 具身智能

- [具身智能简单介绍](rlwiki/robots/embodyai.md)
- [具身智能训练平台](rlwiki/robots/lerobot.md)


### AlphaZero & MuZero & 蒙特卡洛树搜索

- [蒙特卡洛树搜索入门指南](rlwiki/muzero/mcts_guide.md)
- [蒙特卡洛树搜索(MCTS)详解](rlwiki/muzero/mcts.md)
- [AlphaGoZero 算法介绍](rlwiki/muzero/alphazero.md)
- [MuZero算法介绍](rlwiki/muzero/muzero_intro.md)
- [MuZero伪代码](rlwiki/muzero/muzero_pseudocode.md)

### 多智能体强化学习

- [MARL](rlwiki/papers/Overview.md)
- [DRQN](rlwiki/papers/DRQN.md)
- [IQL](rlwiki/papers/IQL.md)
- [COMA](rlwiki/papers/COMA.md)
- [VDN](rlwiki/papers/VDN.md)
- [QTRAN](rlwiki/papers/QTRAN.md)
- [QMIX](rlwiki/papers/QMIX.md)
- [MADDPG](rlwiki/papers/MADDPG.md)
- [MAT](rlwiki/papers/MAT.md)
- [零和博弈](rlwiki/papers/league.md)
- [Self-Play](rlwiki/papers/self-play.md)
- [Douzero](rlwiki/papers/Douzero.md)

### OpenDILab

#### PPOFamaily  决策智能公开课

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

#### OpenDILab 实践指南

[OpenDILab 实践指南（1）：RL 算法/环境速查表（cheatsheet）](https://zhuanlan.zhihu.com/p/654237727)

[OpenDILab 实践指南（3）：深入浅出配置文件系统](https://zhuanlan.zhihu.com/p/656886757)

[OpenDILab 实践指南（4）：算法-代码对应解读文档](https://zhuanlan.zhihu.com/p/658146912)

[OpenDILab 实践指南（5）：高效构建决策环境](https://zhuanlan.zhihu.com/p/660266568)

#### MCTS 的前沿动态

[MCTS + RL 系列技术博客（1）：AlphaZero](https://zhuanlan.zhihu.com/p/650009275)

[MCTS + RL 系列技术博客（2）：MuZero](https://zhuanlan.zhihu.com/p/654059774)

[MCTS + RL 系列技术博客（3）：Sampled MuZero](https://zhuanlan.zhihu.com/p/657484426)

[MCTS + RL 系列技术博客（4）：EfficientZero](https://zhuanlan.zhihu.com/p/662943334)

[MCTS + RL 系列技术博客（5）：Stochastic MuZero](https://zhuanlan.zhihu.com/p/664542805)

[MCTS + RL 系列技术博客（6）：浅析 MCTS 算法原理演进史](https://zhuanlan.zhihu.com/p/670885213)

[MCTS + RL 系列技术博客（7）：Gumbel MuZero](https://zhuanlan.zhihu.com/p/675867274)

[MCTS + RL 系列技术博客（8）：MuZero Unplugged](https://zhuanlan.zhihu.com/p/678489535)

[MCTS + RL 系列技术博客（9）：可视化分析 AlphaZero/MuZero 中的表征、策略和知识](https://zhuanlan.zhihu.com/p/681839624)

[MCTS + RL 系列技术博客（10）：使用多方面的先验知识强化 MCTS Agent](https://zhuanlan.zhihu.com/p/684528177)

[MCTS + RL 系列技术博客（11）：MCTS 效率优化前沿进展](https://zhuanlan.zhihu.com/p/688093451)

[如何追踪 MCTS 的前沿动态？（1）](https://zhuanlan.zhihu.com/p/633462726)

[如何追踪 MCTS 的前沿动态？（2）](https://zhuanlan.zhihu.com/p/635570518)

[如何追踪 MCTS 的前沿动态？（3）](https://zhuanlan.zhihu.com/p/641713866)

[如何追踪 MCTS 的前沿动态？（4）](https://zhuanlan.zhihu.com/p/643382412)
