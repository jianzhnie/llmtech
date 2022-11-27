# Gym

OpenAI Gym 是一个研究和开发强化学习相关算法的仿真平台。

- 无需智体先验知识；
- 兼容常见的数值运算库如 TensorFlow、PyTorch 等.

Gym以 api 简单、pythonic，并且能够表示一般的 RL 问题，在强化学习领域非常知名。

**Gym 环境文档：https://www.gymlibrary.ml/**

```python
import gym
env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset(seed=42)
for _ in range(1000):
   action = policy(observation)  # User-defined policy function
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()
env.close()
```

## 基本用法

### 初始化环境

在 Gym 中初始化环境非常简单，可以通过以下方式完成：

```
import gym
env = gym.make('CartPole-v0')
```

### 与环境互动

Gym 实现了经典的“Agent-Environment循环”：

此示例将运行`LunarLander-v2`1000 个时间步长的环境实例。

```python
import gym
env = gym.make("LunarLander-v2", render_mode="human")
env.action_space.seed(42)

observation, info = env.reset(seed=42)

for _ in range(1000):
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())

    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

输出应该是这个样子

[![https://user-images.githubusercontent.com/15806078/153222406-af5ce6f0-4696-4a24-a683-46ad4939170c.gif](gym.assets/153222406-af5ce6f0-4696-4a24-a683-46ad4939170c.gif)

### 检查 API 一致性

如果您已经实现了自定义环境并希望执行健全性检查以确保它符合 API，您可以运行：

```python
>>> from gym.utils.env_checker import check_env
>>> check_env(env)
```

如果您的环境似乎不遵循 Gym API，此函数将抛出异常。如果看起来您犯了错误或未遵循最佳实践（例如，如果`observation_space`看起来像图像但没有正确的数据类型），它也会产生警告。通过传递可以关闭警告`warn=False`。默认情况下，`check_env`不会检查该`render`方法。要更改此行为，您可以通过`skip_render_check=False`.

### Spaces

Spaces 通常用于指定有效操作和观察的格式。每个环境都应该有属性`action_space`和`observation_space`，它们都应该是继承自类的实例`Space`。Gym中有多种可用Space类型：

- `Box`：描述了一个n维连续空间。这是一个有界空间，我们可以在其中定义上限和下限，这些上限和下限描述了我们的观察可以采用的有效值。
- `Discrete`：描述一个离散空间，其中 {0, 1, …, n-1} 是我们的观察或行动可以采用的可能值。可以使用可选参数将值转换为 {a, a+1, …, a+n-1} 。
- `Dict`: 表示简单空间的字典。
- `Tuple`: 代表一个简单空间的元组。
- `MultiBinary`：创建一个 n 元二进制空间。参数 n 可以是一个数字或一个`list`数字。
- `MultiDiscrete`：由一系列`Discrete`动作空间组成，每个元素中有不同数量的动作。

```python
>>> from gym.spaces import Box, Discrete, Dict, Tuple, MultiBinary, MultiDiscrete
>>>
>>> observation_space = Box(low=-1.0, high=2.0, shape=(3,), dtype=np.float32)
>>> observation_space.sample()
[ 1.6952509 -0.4399011 -0.7981693]
>>>
>>> observation_space = Discrete(4)
>>> observation_space.sample()
1
>>>
>>> observation_space = Discrete(5, start=-2)
>>> observation_space.sample()
-2
>>>
>>> observation_space = Dict({"position": Discrete(2), "velocity": Discrete(3)})
>>> observation_space.sample()
OrderedDict([('position', 0), ('velocity', 1)])
>>>
>>> observation_space = Tuple((Discrete(2), Discrete(3)))
>>> observation_space.sample()
(1, 2)
>>>
>>> observation_space = MultiBinary(5)
>>> observation_space.sample()
[1 1 1 0 1]
>>>
>>> observation_space = MultiDiscrete([ 5, 2, 2 ])
>>> observation_space.sample()
[3 0 0]
```

### Wrappers

Wrappers 是一种无需直接更改底层代码即可修改现有环境的便捷方式。使用Wrappers 可以避免大量样板代码并使您的环境更加模块化。Wrappers 也可以链接起来以组合它们的效果。大多数通过`gym.make`生成的环境默认情况下已经被Wrappered。

为了 Wrapper 一个环境，您必须首先初始化一个基础环境。然后你可以将这个环境连同（可能是可选的）参数传递给Wrappers 的构造函数：

```python
>>> import gym
>>> from gym.wrappers import RescaleAction
>>> base_env = gym.make("BipedalWalker-v3")
>>> base_env.action_space
Box([-1. -1. -1. -1.], [1. 1. 1. 1.], (4,), float32)
>>> wrapped_env = RescaleAction(base_env, min_action=0, max_action=1)
>>> wrapped_env.action_space
Box([0. 0. 0. 0.], [1. 1. 1. 1.], (4,), float32)
```

您可能希望Wrappers 执行以下三项非常常见的操作：

- Transform actions before applying them to the base environment
- Transform observations that are returned by the base environment
- Transform rewards that are returned by the base environment

通过继承 `ActionWrapper`、`ObservationWrapper`或`RewardWrapper`并实现相应的transform，可以轻松实现此类包装器。

然而，有时您可能需要实现一个Wrappers来进行一些更复杂的修改（例如，根据 info中的数据修改奖励）。这样的包装器可以通过继承Wrapper来实现. Gym 已经为你提供了许多常用的Wrappers。一些例子：

- `TimeLimit`：如果超过最大时间步数（或基础环境已发出完成信号），则发出完成信号。
- `ClipAction`：裁剪动作，使其位于动作空间（类型`Box`）中。
- `RescaleAction`：重新缩放动作以位于指定的区间内
- `TimeAwareObservation`：将有关时间步长索引的信息添加到观察中。在某些情况下有助于确保transitions是马尔可夫。

如果你有一个Wrapper环境，并且你想在所有Wrappers下面获得未被Wrappers的环境（以便你可以手动调用一个函数或更改环境的某些底层方面），你可以使用该`.unwrapped`属性。如果环境已经是基础环境，则该`.unwrapped`属性将只返回自身。

```python
>>> wrapped_env
<RescaleAction<TimeLimit<BipedalWalker<BipedalWalker-v3>>>>
>>> wrapped_env.unwrapped
<gym.envs.box2d.bipedal_walker.BipedalWalker object at 0x7f87d70712d0>
```

### 与环境交互

您还可以使用gym.utils.play中的`play`功能和环境交互。

```python
from gym.utils.play import play
play(gym.make('Pong-v0'))
```

这将打开一个环境窗口，并允许您使用键盘控制 agent。

```python
{
    # ...
    (ord('w'), ord(' ')): 2,
    # ...
}
```

作为一个更完整的示例，假设我们希望`CartPole-v0`使用左右箭头键进行游戏。代码如下：

```python
import gym
import pygame
from gym.utils.play import play
mapping = {(pygame.K_LEFT,): 0, (pygame.K_RIGHT,): 1}
play(gym.make("CartPole-v0"), keys_to_action=mapping)
```

我们从 pygame 中获取相应的密钥 ID 常量。如果`key_to_action`未指定参数，则使用该环境的默认`key_to_action`映射（如果提供）。

此外，如果您希望在玩游戏时绘制实时统计数据，您可以使用`gym.utils.play.PlayPlot`. 下面是一些示例代码，用于绘制游戏最后 5 秒的奖励：

```python
def callback(obs_t, obs_tp1, action, rew, done, info):
    return [rew,]
plotter = PlayPlot(callback, 30 * 5, ["reward"])
env = gym.make("Pong-v0")
play(env, callback=plotter.callback)
```

## API

### 核心 api

#### step()

```python
gym.Env.step(self, action: ActType) → Tuple[ObsType, float, bool, bool, dict]
```

> gym.Env.**step****(***self***,** *action: ActType***)** **→ Tuple[ObsType, float, bool, bool, dict]**

运行环境动力学的一个时间步骤。
当 episode 结束时，需要调用 reset（）来重置此环境的状态。接受 action 并返回元组（observation, reward, terminated, truncated, info）。

#### reset()

>gym.Env.**reset****(***self***,** *****,** *seed: Optional[int] = None***,** *options: Optional[dict] = None***)** **→ Tuple[ObsType, dict]**

#### render()

> gym.Env.render(*self*) → Optional[Union[RenderFrame, List[RenderFrame]]][#](https://www.gymlibrary.dev/api/core/#gym.Env.render)

#### Close()

> gym.Env.close(*self*)

### Space

Space主要定义了环境的观察和行动空间的有效格式。包含了Seed函数、Sample等各种各样的函数接口：

#### 1.discrete类

- Discrete类对应于一维离散空间
- 定义一个Discrete类的空间只需要一个参数n就可以了
- discrete space允许固定范围的非负数
- 每个时间步agent只采取离散空间中的一个动作，如离散空间中actions=\[上、下、左、右\]，一个时间步可能采取“上”这一个动作。

#### 2.box类

- box类对应于多维连续空间
- Box空间可以定义多维空间，每一个维度可以用一个最低值和最大值来约束
- 定义一个多维的Box空间需要知道每一个维度的最小最大值，当然也要知道维数。

#### 3.multidiscrete类

- 用于多维离散空间

- 多离散动作空间由一系列具有不同参数的离散动作空间组成

  - 它可以适应离散动作空间或连续（Box）动作空间

  - 表示游戏控制器或键盘非常有用，其中每个键都可以表示为离散的动作空间

  - 通过传递每个离散动作空间包含\[min，max\]的数组的数组进行参数化

  - 离散动作空间可以取从min到max的任何整数（包括两端值）

> 多智能体算法中在train开始的时候，把不同种类的动作建立成了各种不同的分布, 最后的动作输出的是分布，根据分布最后采样得到输出值。
>
> - Box 连续空间->DiagGaussianPdType （对角高斯概率分布）
> - Discrete离散空间->SoftCategoricalPdType（软分类概率分布）
> - MultiDiscrete连续空间->SoftMultiCategoricalPdType （多变量软分类概率分布）
> - 多二值变量连续空间->BernoulliPdType （伯努利概率分布）-

```python3
class Box(Space):
    """
    A (possibly unbounded) box in R^n. Specifically, a Box represents the
    Cartesian product of n closed intervals. Each interval has the form of one
    of [a, b], (-oo, b], [a, oo), or (-oo, oo).
    There are two common use cases:
    * Identical bound for each dimension::
        >>> Box(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        Box(3, 4)
    * Independent bound for each dimension::
        >>> Box(low=np.array([-1.0, -2.0]), high=np.array([2.0, 4.0]), dtype=np.float32)
        Box(2,)
    """
```

- 上述描述的直译：box（可能是无界的）在n维空间中。一个box代表n维封闭区间的笛卡尔积。每个区间都有\[a, b\]， (-oo, b)， \[a, oo)，或(-oo, oo)的形式。

- - 需要注意的重点在于：box可以表示n维空间，并且区间有闭有开。

- 例子：每一维相同的限制：

- - 可以看到，此处Box的shape为(3, 4)，每一维区间的最小值为-1.0，最大值为2.0。

```python
>>> Box(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        Box(3, 4)
```

- 如果不够直观，我们可以它sample出来看一下。

```python
import numpy as np
from gym.spaces.box import Box

my_box = Box(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
my_box_sample = my_box.sample()

print(my_box)
print(my_box_sample)

# 输出
Box(3, 4)
[[ 0.34270912 -0.17985763  1.7716838  -0.71165234]
 [ 0.5638914  -0.6311684  -0.28997722 -0.19067103]
 [-0.6750097   0.99941856  1.1923424  -0.9933872 ]]
```

### **Vector API**

矢量化环境（Vectorized Environments）是运行多个（独立）子环境的环境，可以按顺序运行，也可以使用多处理并行运行。矢量化环境将一批动作作为输入，并返回一批观察结果。这特别有用，例如，当策略被定义为对一批观察结果进行操作的神经网络时。其中Vector API包含了：

Gym 提供两种类型的矢量化环境：

- gym.vector.SyncVectorEnv，其中子环境按顺序执行。
- gym.vector.AsyncVectorEnv，其中子环境使用多处理并行执行。这会为每个子环境创建一个进程。

与gym.make 类似，您可以使用gym.vector.make 函数运行已注册环境的矢量化版本。这会运行同一环境的多个副本（默认情况下是并行的）。以下示例并行运行 3 个 CartPole-v1 环境副本，将 3 个二进制动作的向量（每个子环境一个）作为输入，并返回沿第一维堆叠的 3 个观察值数组，数组为每个子环境返回的奖励，以及一个布尔数组，指示每个子环境中的情节是否已经结束。

```python
>>> envs = gym.vector.make("CartPole-v1", num_envs=3)
>>> envs.reset()
>>> actions = np.array([1, 0, 1])
>>> observations, rewards, dones, infos = envs.step(actions)

>>> observations
array([[ 0.00122802,  0.16228443,  0.02521779, -0.23700266],
        [ 0.00788269, -0.17490888,  0.03393489,  0.31735462],
        [ 0.04918966,  0.19421194,  0.02938497, -0.29495203]],
        dtype=float32)
>>> rewards
array([1., 1., 1.])
>>> dones
array([False, False, False])
>>> infos
({}, {}, {})
```

## Environment

### **Toy Text**

所有玩具文本环境都是由我们使用原生 Python 库（例如 StringIO）创建的。这些环境被设计得非常简单，具有小的离散状态和动作空间，因此易于学习。 因此，它们适用于调试强化学习算法的实现。所有环境都可以通过每个环境文档中指定的参数进行配置。

### **Classic Control**

有五种经典控制环境：Acrobot、CartPole、Mountain Car、Continuous Mountain Car 和 Pendulum。所有这些环境在给定范围内的初始状态都是随机的。此外，Acrobot 已将噪声应用于所采取的操作。另外，对于这两种山地车环境，爬山的车都动力不足，所以要爬到山顶需要一些努力。在 Gym 环境中，这组环境可以被认为是更容易通过策略解决的环境。所有环境都可以通过每个环境文档中指定的参数进行高度配置。

### **Box2D**

这些环境都涉及基于物理控制的玩具游戏，使用基于 box2d 的物理和基于 PyGame 的渲染。这些环境是由 Oleg Klimov 在 Gym 早期贡献的，从那时起就成为流行的玩具基准。所有环境都可以通过每个环境文档中指定的参数进行高度配置。

### **Atari**

Atari 环境通过街机学习环境 (ALE) [1] 进行模拟。

### **Mujoco**

MuJoCo 代表带接触的多关节动力学。它是一个物理引擎，用于促进机器人、生物力学、图形和动画以及其他需要快速准确模拟的领域的研究和开发。

这些环境还需要安装 MuJoCo 引擎。截至 2021 年 10 月，DeepMind 已收购 MuJoCo，并于 2022 年将其开源，对所有人免费开放。可以在他们的网站和 GitHub 存储库中找到有关安装 MuJoCo 引擎的说明。将 MuJoCo 与 OpenAI Gym 一起使用还需要安装框架 mujoco-py，可以在 GitHub 存储库中找到该框架（使用上述命令安装此依赖项）。

有十个 Mujoco 环境：Ant、HalfCheetah、Hopper、Hupper、Humanoid、HumanoidStandup、IvertedDoublePendulum、InvertedPendulum、Reacher、Swimmer 和 Walker。所有这些环境的初始状态都是随机的，为了增加随机性，将高斯噪声添加到固定的初始状态。Gym 中 MuJoCo 环境的状态空间由两个部分组成，它们被展平并连接在一起：身体部位 ('mujoco-py.mjsim.qpos') 或关节的位置及其对应的速度 ('mujoco-py.mjsim. qvel'）。通常，状态空间中会省略一些第一个位置元素，因为奖励是根据它们的值计算的，留给算法间接推断这些隐藏值。

此外，在 Gym 环境中，这组环境可以被认为是更难通过策略解决的环境。可以通过更改 XML 文件或调整其类的参数来配置环境。



## **Tutorials**
