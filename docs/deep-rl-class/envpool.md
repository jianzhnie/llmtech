# EnvPool: 并行环境模拟器

> 对于强化学习标准环境 Atari 与 Mujoco，如果希望在短时间内完成训练，需要采用数百个 CPU 核心的大规模分布式解决方案；而使用 EnvPool，只需要一台游戏本就能完成相同体量的训练任务，并且用时不到 5 分钟，极大地降低了训练成本。

目前，**EnvPool** 项目已在 GitHub 开源，收获超过 500 Stars，并且受到众多强化学习研究者的关注。

- 项目地址：https://github.com/sail-sg/envpool
- 在线文档：https://envpool.readthedocs.io/en/latest/
- arXiv 链接：https://arxiv.org/abs/2206.10558

**EnvPool 是一个基于 C++ 、高效、通用的强化学习并行环境（vectorized environment）模拟器**，不仅能够兼容已有的 gym/dm_env API，还支持了多智能体环境。除了 OpenAI Gym 本身拥有的环境外，EnvPool 还支持一些额外的复杂环境。

目前支持的环境有：

- Atari games
- Mujoco（gym）
- Classic control RL envs: CartPole, MountainCar, Pendulum, Acrobot
- Toy text RL envs: Catch, FrozenLake, Taxi, NChain, CliffWalking, Blackjack
- ViZDoom single player
- DeepMind Control Suite

## **追求极致速度**

EnvPool **采取了 C++ 层面的并行解决方案**。根据现有测试结果，使用 EnvPool 并行运行多个强化学习环境，能在正常笔记本上比主流的 Python Subprocess 解决方案快近 3 倍；使用多核 CPU 服务器能够达到更好的性能。

例如在 NVIDIA DGX-A100（256 核 CPU 服务器）上的测试结果表明，Atari 游戏能够跑出每秒一百多万帧的惊人速度，Mujoco 物理引擎的任务更是能跑出每秒三百多万模拟步数的好成绩，比 Python Subprocess 快近二十倍，比之前最快的 CPU 异步模拟器 Sample Factory 还快两倍。

与此同时，EnvPool + CleanRL 的整系统测试表明，使用原始的 PPO 算法，直接把原来基于 Python Subprocess 的主流解决方案替换成 EnvPool，整体系统在标准的 Atari 基准测试中能快近三倍！

标准测试结果表明，对于数量稍大（比如超过 32）的并行环境，Subprocess 的运行效率十分堪忧。

为此有研究者提出分布式解决方案（比如 Ray）和基于 GPU 的解决方案（比如 Brax 和 Isaac-gym）进行加速。分布式方案经过测试，计算资源利用率其实并不高；基于 GPU 的解决方案虽然可以达到千万 FPS，但并不是所有环境都能使用 CUDA 重写，不能很好兼容生态以及不能复用一些受商业保护的代码。

EnvPool 由于采用了 C++ 层面的并行解决方案，并且大部分强化学习环境都使用 C++ 实现来保证运行效率，因此只需要在 C++ 层面实现接口即可完成高效的并行。

## **打造开放生态**

EnvPool 对上下游的生态都有着良好的支持：

- 对于上游的强化学习环境而言，目前最多使用的是 OpenAI Gym，其次是 DeepMind 的 dm_env 接口。EnvPool 对两种环境 API 都完全支持，并且每次 env.step 出来的数据都是经过 numpy 封装好的，用户不必每次手动合并数据，同时也提高了吞吐量；
- 对于下游的强化学习算法库而言，EnvPool 支持了目前 PyTorch 最为流行的两个算法库 Stable-baselines3 和 Tianshou，同时还支持了 ACME、CleanRL 和 rl_games 等强化学习算法库，并且达到了令人惊艳的效果（在笔记本电脑上，2 分钟训练完 Atari Pong、5 分钟训练完 Mujoco Ant/HalfCheetah，并且通过了 Gym 原本环境的验证）。

为了更好地营造生态，**EnvPool 采用了 Bazel 进行构建，拥有完善的软件工程标准，也提供了高质量的代码、单元测试和在线文档**。

## **使用示例**

以下是一些简单的 EnvPool 使用示例。首先导入必要的包：

```python
import numpy as np, envpool
```

以 gym.Env 接口为例，初始化 100 个 Atari Pong 的并行环境，只需要一行代码：

```python
env = envpool.make_gym("Pong-v5", num_envs=100)
```

访问 observation_space 和 action_space 和 Gym 如出一辙：

```python
observation_space = env.observation_space
action_space = env.action_space
```

在同步模式下，API 与已有的 Gym API 无缝衔接，只不过第一维大小是并行环境个数：

```python
obs = env.reset()  # should be (100, 4, 84, 84)
act = np.zeros(100, dtype=int)
obs, rew, done, info = env.step(act)
```

当然也可以只 step/reset 部分环境，只需多传一个参数 env_id 即可：

```python
while True:
	act = policy(obs)
	obs, rew, done, info = env.step(act, env_id)
	env_id = info["env_id"]
```

为了追求极致性能，EnvPool 还支持异步模式的 step/reset，于在线文档和论文中对此进行了详细阐述与实验。

## **实际体验效果**

**EnvPool 只需要一句命令 pip install envpool 就能安装，目前仅支持 Linux 操作系统并且 Python 版本为 3.7/3.8/3.9**。EnvPool 在短期内会对其他操作系统（Windows 和 macOS）进行支持。

在 EnvPool 的论文中，作者们给出了如下 rl_games 的惊艳结果：只用一台游戏本，在其他条件都完全相同的情况下，把基于 ray 的 vectorized env 实现直接换成 EnvPool，能够直接获得免费的加速。

Atari Pong 使用了 EnvPool，可以在大约 5 分钟的时候训练完成，相比而言 ray 的方案需要在大约 15 分钟的时候训练完成（达到接近 20 的 reward）；Mujoco Ant 更为明显，使用原始 PPO 算法在不到 5 分钟的时间内达到了超过 5000 的 reward，而基于 ray 的解决方案运行了半小时还没达到 5000。
