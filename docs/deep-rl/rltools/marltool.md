# 多智能体强化学习代码汇总（pytorch）

## Algorithms

We provide three types of MARL algorithms as our baselines including:

**Independent Learning:** 
IQL
DDPG
PG
A2C
TRPO
PPO

**Centralized Critic:**
COMA 
MADDPG 
MAAC 
MAPPO
MATRPO
HATRPO
HAPPO

**Value Decomposition:**
VDN
QMIX
FACMAC
VDAC
VDPPO

Here is a chart describing the characteristics of each algorithm:

| Algorithm                                                    | Support Task Mode                           | Need Central Information | Discrete Action    | Continuous Action  | Learning Categorize  | Type       |
| ------------------------------------------------------------ | ------------------------------------------- | ------------------------ | ------------------ | ------------------ | -------------------- | ---------- |
| IQL*                                                         | cooperative collaborative competitive mixed |                          | :heavy_check_mark: |                    | Independent Learning | Off Policy |
| [PG](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf) | cooperative collaborative competitive mixed |                          | :heavy_check_mark: | :heavy_check_mark: | Independent Learning | On Policy  |
| [A2C](https://arxiv.org/abs/1602.01783)                      | cooperative collaborative competitive mixed |                          | :heavy_check_mark: | :heavy_check_mark: | Independent Learning | On Policy  |
| [DDPG](https://arxiv.org/abs/1509.02971)                     | cooperative collaborative competitive mixed |                          |                    | :heavy_check_mark: | Independent Learning | Off Policy |
| [TRPO](http://proceedings.mlr.press/v37/schulman15.pdf)      | cooperative collaborative competitive mixed |                          | :heavy_check_mark: | :heavy_check_mark: | Independent Learning | On Policy  |
| [PPO](https://arxiv.org/abs/1707.06347)                      | cooperative collaborative competitive mixed |                          | :heavy_check_mark: | :heavy_check_mark: | Independent Learning | On Policy  |
| [COMA](https://ojs.aaai.org/index.php/AAAI/article/download/11794/11653) | cooperative collaborative competitive mixed | :heavy_check_mark:       | :heavy_check_mark: |                    | Centralized Critic   | On Policy  |
| [MADDPG](https://arxiv.org/abs/1706.02275)                   | cooperative collaborative competitive mixed | :heavy_check_mark:       |                    | :heavy_check_mark: | Centralized Critic   | Off Policy |
| MAA2C*                                                       | cooperative collaborative competitive mixed | :heavy_check_mark:       | :heavy_check_mark: | :heavy_check_mark: | Centralized Critic   | On Policy  |
| MATRPO*                                                      | cooperative collaborative competitive mixed | :heavy_check_mark:       | :heavy_check_mark: | :heavy_check_mark: | Centralized Critic   | On Policy  |
| [MAPPO](https://arxiv.org/abs/2103.01955)                    | cooperative collaborative competitive mixed | :heavy_check_mark:       | :heavy_check_mark: | :heavy_check_mark: | Centralized Critic   | On Policy  |
| [HATRPO](https://arxiv.org/abs/2109.11251)                   | Cooperative                                 | :heavy_check_mark:       | :heavy_check_mark: | :heavy_check_mark: | Centralized Critic   | On Policy  |
| [HAPPO](https://arxiv.org/abs/2109.11251)                    | Cooperative                                 | :heavy_check_mark:       | :heavy_check_mark: | :heavy_check_mark: | Centralized Critic   | On Policy  |
| [VDN](https://arxiv.org/abs/1706.05296)                      | Cooperative                                 |                          | :heavy_check_mark: |                    | Value Decomposition  | Off Policy |
| [QMIX](https://arxiv.org/abs/1803.11485)                     | Cooperative                                 | :heavy_check_mark:       | :heavy_check_mark: |                    | Value Decomposition  | Off Policy |
| [FACMAC](https://arxiv.org/abs/2003.06709)                   | Cooperative                                 | :heavy_check_mark:       |                    | :heavy_check_mark: | Value Decomposition  | Off Policy |
| [VDAC](https://arxiv.org/abs/2007.12306)                     | Cooperative                                 | :heavy_check_mark:       | :heavy_check_mark: | :heavy_check_mark: | Value Decomposition  | On Policy  |
| VDPPO*                                                       | Cooperative                                 | :heavy_check_mark:       | :heavy_check_mark: | :heavy_check_mark: | Value Decomposition  | On Policy  |

*IQL* is the multi-agent version of Q learning.
*MAA2C* and *MATRPO* are the centralized version of A2C and TRPO.
*VDPPO* is the value decomposition version of PPO.

## Awesome Repos

Here we provide a table for the comparison of MARLlib and existing work.

|                           Library                            |                         Github Stars                         |                  Task Mode                  |                        Supported Env                         |                          Algorithm                           |             Parameter Sharing              | Asynchronous Interact |         Framework          |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :-----------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------: | :-------------------: | :------------------------: |
|         [PyMARL](https://github.com/oxwhirl/pymarl)          | [![GitHub stars](https://img.shields.io/github/stars/oxwhirl/pymarl)](https://github.com/oxwhirl/pymarl/stargazers) |                 cooperative                 |                              1                               | Independent Learning(1) + Centralized Critic(1) + Value Decomposition(3) |                full-sharing                |                       |             *              |
|        [PyMARL2](https://github.com/hijkzzz/pymarl2)         | [![GitHub stars](https://img.shields.io/github/stars/hijkzzz/pymarl2)](https://github.com/hijkzzz/pymarl2/stargazers) |                 cooperative                 |                              1                               | Independent Learning(1) +  Centralized Critic(1) +  Value Decomposition(9) |                full-sharing                |                       |           PyMARL           |
| [MARL-Algorithms](https://github.com/starry-sky6688/MARL-Algorithms) | [![GitHub stars](https://img.shields.io/github/stars/starry-sky6688/MARL-Algorithms)](https://github.com/starry-sky6688/MARL-Algorithms/stargazers) |                 cooperative                 |                              1                               |    CTDE(6) + Communication(1) + Graph(1) + Multi-task(1)     |                full-sharing                |                       |             *              |
|       [EPyMARL](https://github.com/uoe-agents/epymarl)       | [![GitHub stars](https://img.shields.io/github/stars/uoe-agents/epymarl)](https://github.com/hijkzzz/uoe-agents/epymarl/stargazers) |                 cooperative                 |                              4                               | Independent Learning(3) + Value Decomposition(4) + Centralized Critic(2) |         full-sharing + non-sharing         |                       |           PyMARL           |
|         [MAlib](https://github.com/sjtu-marl/malib)          | [![GitHub stars](https://img.shields.io/github/stars/sjtu-marl/malib)](https://github.com/hijkzzz/sjtu-marl/malib/stargazers) |                  self-play                  | 2 +  [PettingZoo](https://www.pettingzoo.ml/) + [OpenSpiel](https://github.com/deepmind/open_spiel) |                       Population-based                       | full-sharing + group-sharing + non-sharing |  :heavy_check_mark:   |             *              |
| [MAPPO Benchmark](https://github.com/marlbenchmark/on-policy) | [![GitHub stars](https://img.shields.io/github/stars/marlbenchmark/on-policy)](https://github.com/marlbenchmark/on-policy/stargazers) |                 cooperative                 |                              4                               |                           MAPPO(1)                           |         full-sharing + non-sharing         |  :heavy_check_mark:   | pytorch-a2c-ppo-acktr-gail |
|    [MARLlib](https://github.com/Replicable-MARL/MARLlib)     |                                                              | cooperative collaborative competitive mixed |        10 + [PettingZoo](https://www.pettingzoo.ml/)         | Independent Learning(6) + Centralized Critic(7) + Value Decomposition(5) | full-sharing + group-sharing + non-sharing |  :heavy_check_mark:   |         Ray/Rllib          |

## Some comments

1. starry-sky6688

这套代码简单易上手，适合初学者入门。包含IQL、QMIX、VDN、COMA、QTRAN、MAVEN、CommNet、DyMA-CL、G2ANet和MADDPG。

- https://github.com/starry-sky6688/MARL-Algorithms

- https://github.com/starry-sky6688/MADDPG

2. pymarl

牛津大学whiteson组的代码库，我稍微看了两眼，非常模块化的代码，我猜应该是那种很好用但也很难上手的类型。包括QMIX、COMA、VDN、IQL、QTRAN。

- https://github.com/oxwhirl/pymarl

3. pymarl2（351星）

pymarl的改进版本, 增加了一些code-level tricks。

- https://arxiv.org/abs/2102.03479

- https://github.com/hijkzzz/pymarl2

4. epymarl（139星）

epymarl的扩展版本，在pymarl的基础上增加了IA2C、IPPO、MADDPG、MAA2C、MAPPO。

- https://github.com/uoe-agents/epymarl

5. shariqiqbal2810（307星）

MAAC作者写的代码，挺简洁的。包括MADDPG、MAAC。

- https://github.com/shariqiqbal2810/maddpg-pytorch

- https://github.com/shariqiqbal2810/MAAC

6. marlbenchmark（509星/102星）

清华大学的代码库，包含MAPPO、QMIX、VDN、MADDPG和MATD3，其中VDN和MATD3没有经过完整测试。

- https://github.com/marlbenchmark/on-policy

- https://github.com/marlbenchmark/off-policy

7. marllib（70星）

一个涵盖了大多主流MARL算法的代码库，基于ray的rllib，属于那种模块化做得特别好，但上手需要花些时间的代码，包含independence learning (IQL, A2C, DDPG, TRPO, PPO), centralized critic learning (COMA, MADDPG, MAPPO, HATRPO), and value decomposition (QMIX, VDN, FACMAC, VDA2C)。

- https://github.com/Replicable-MARL/MARLlib


## Environments

Most of the popular environments in MARL research are supported by MARLlib:

| Env Name                                                     | Learning Mode                       | Observability | Action Space | Observations |
| ------------------------------------------------------------ | ----------------------------------- | ------------- | ------------ | ------------ |
| [LBF](https://github.com/semitable/lb-foraging)              | cooperative + collaborative         | Both          | Discrete     | Discrete     |
| [RWARE](https://github.com/semitable/robotic-warehouse)      | cooperative                         | Partial       | Discrete     | Discrete     |
| [MPE](https://github.com/openai/multiagent-particle-envs)    | cooperative + collaborative + mixed | Both          | Both         | Continuous   |
| [SMAC](https://github.com/oxwhirl/smac)                      | cooperative                         | Partial       | Discrete     | Continuous   |
| [MetaDrive](https://github.com/decisionforce/metadrive)      | collaborative                       | Partial       | Continuous   | Continuous   |
| [MAgent](https://www.pettingzoo.ml/magent)                   | collaborative + mixed               | Partial       | Discrete     | Discrete     |
| [Pommerman](https://github.com/MultiAgentLearning/playground) | collaborative + competitive + mixed | Both          | Discrete     | Discrete     |
| [MAMuJoCo](https://github.com/schroederdewitt/multiagent_mujoco) | cooperative                         | Partial       | Continuous   | Continuous   |
| [GRF](https://github.com/google-research/football)           | collaborative + mixed               | Full          | Discrete     | Continuous   |
| [Hanabi](https://github.com/deepmind/hanabi-learning-environment) | cooperative                         | Partial       | Discrete     | Discrete     |

Each environment has a readme file, standing as the instruction for this task, talking about env settings, installation, and some important notes.
