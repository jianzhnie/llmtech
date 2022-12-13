# SMAC

# 概述

SMAC
官方`GitHub`链接为：https://github.com/oxwhirl/smac。
SMAC 是一个用于在暴雪星际争霸2上进行多智能体协同强化学习（MARL）的环境。
SMAC用了暴雪星际争霸2 的机器学习 API 和 DeepMind 的PySC2
为智能体与星际争霸2的交互提供了友好的接口，方便开发者观察和执行行动。 与
PySC2 相比，SMAC
专注于分散的微观操作方案，其中游戏的每个智能体均由单独的 RL agent控制。

![../_images/smac1.gif](https://opendilab.github.io/DI-engine/_images/smac1.gif)

# 安装

需要安装星际争霸2 游戏和 PySC2
库.

## 安装StarCraft II

安装主要包括两部分：

- 下载星际争霸2 游戏 

因为SMAC是基于星际争霸游戏引擎的，所以我们还需要安装StarCraft II，官方指定的版本为SC2.4.10，并且不同版本之间的算法性能测试不一样。对于 Linux
系统使用者，安装路径为<https://github.com/Blizzard/s2client-proto#downloads>，之后使用
`export SC2PATH=<sc2/installation/path>` 命令将安装路径添加到环境变量中.
对于 Windows 系统使用者，安装请参考<https://starcraft2.com>.

- 下载SMAC地图

我们需要下载地图，也就是游戏的地图并将其放在之前解压的StarCraft II文件下面的Maps目录下面。下载链接为：https://github.com/oxwhirl/smac/releases/download/v0.1-beta1/SMAC_Maps.zip。

下面的shell脚本包含了完整安装过程。

```shell
#!/bin/bash
# Install SC2 and add the custom maps

if [ -z "$EXP_DIR" ]
then
    EXP_DIR=~
fi

echo "EXP_DIR: $EXP_DIR"
cd $EXP_DIR/pymarl

mkdir 3rdparty
cd 3rdparty

export SC2PATH=`pwd`'/StarCraftII'
echo 'SC2PATH is set to '$SC2PATH

if [ ! -d $SC2PATH ]; then
        echo 'StarCraftII is not installed. Installing now ...';
        wget http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip
        unzip -P iagreetotheeula SC2.4.10.zip
        rm -rf SC2.4.10.zip
else
        echo 'StarCraftII is already installed.'
fi

echo 'Adding SMAC maps.'
MAP_DIR="$SC2PATH/Maps/"
echo 'MAP_DIR is set to '$MAP_DIR

if [ ! -d $MAP_DIR ]; then
        mkdir -p $MAP_DIR
fi

cd ..
wget https://github.com/oxwhirl/smac/releases/download/v0.1-beta1/SMAC_Maps.zip
unzip SMAC_Maps.zip
mv SMAC_Maps $MAP_DIR
rm -rf SMAC_Maps.zip

echo 'StarCraft II and SMAC are installed.'
```

## 安装PySC2

``` shell
pip install pysc2
```

## 安装smac

```
pip install git+https://github.com/oxwhirl/smac.git
```

# 验证安装

安装完成后，可以通过安装成功后 `echo $SC2PATH` 确认环境变量设置成功.

### 测试`Map`是否放置成功：

```bash
python -m smac.bin.map_list
```

得到下面的输出：

```shell
Name            Agents  Enemies Limit  
3m              3       3       60     
8m              8       8       120    
25m             25      25      150    
5m_vs_6m        5       6       70     
8m_vs_9m        8       9       120    
10m_vs_11m      10      11      150    
27m_vs_30m      27      30      180    
MMM             10      10      150    
MMM2            10      12      180    
2s3z            5       5       120    
3s5z            8       8       150    
3s5z_vs_3s6z    8       9       170    
3s_vs_3z        3       3       150    
3s_vs_4z        3       4       200    
3s_vs_5z        3       5       250    
1c3s5z          9       9       180    
2m_vs_1z        2       1       150    
corridor        6       24      400    
6h_vs_8z        6       8       150    
2s_vs_1sc       2       1       300    
so_many_baneling 7       32      100    
bane_vs_bane    24      24      200    
2c_vs_64zg      2       64      400 
```

### 测试`smac`和它的`Map`是否配置成功：

```python
python -m smac.examples.random_agents
```

得到下面的输出：

```shell
Version: B75689 (SC2.4.10)
Build: Aug 12 2019 17:16:57
Command Line: '"/home/robin/StarCraftII/Versions/Base75689/SC2_x64" -listen 127.0.0.1 -port 35069 -dataDir /home/robin/StarCraftII/ -tempDir /tmp/sc-bdv0wmyd/'
Starting up...
Startup Phase 1 complete
Startup Phase 2 complete
Creating stub renderer...
Listening on: 127.0.0.1:35069
Startup Phase 3 complete. Ready for commands.
ConnectHandler: Request from 127.0.0.1:39862 accepted
ReadyHandler: 127.0.0.1:39862 ready
Requesting to join a single player game
Configuring interface options
Configure: raw interface enabled
Configure: feature layer interface disabled
Configure: score interface disabled
Configure: render interface disabled
Launching next game.
Next launch phase started: 2
Next launch phase started: 3
Next launch phase started: 4
Next launch phase started: 5
Next launch phase started: 6
Next launch phase started: 7
Next launch phase started: 8
Game has started.
Using default stable ids, none found at: /home/robin/StarCraftII/stableid.json
Successfully loaded stable ids: GameData\stableid.json
Sending ResponseJoinGame
Total reward in episode 0 = 2.8125
Total reward in episode 1 = 2.0625
Total reward in episode 2 = 2.0625
Total reward in episode 3 = 2.0625
Total reward in episode 4 = 2.0625
Total reward in episode 5 = 2.4375
Total reward in episode 6 = 1.875
Total reward in episode 7 = 1.6875
Total reward in episode 8 = 1.5
Total reward in episode 9 = 1.3125
DataHandler: unable to parse websocket frame.
RequestQuit command received.
CloseHandler: 127.0.0.1:39862 disconnected
Closing Application...
ResponseThread: No connection, dropping the response.
```

### Py文件中进行测试

如果想要`Debug`初步了解这个环境的话，可以采用如下代码：

```python
from smac.env import StarCraft2Env
import numpy as np

# 独立的智能体在接收到观察和全局状态后会执行随机策略。
def main():
    env = StarCraft2Env(map_name="8m")
    env_info = env.get_env_info()

    n_actions = env_info["n_actions"]  # 获取动作维度 14
    n_agents = env_info["n_agents"]  # 存在多少个智能体 8
    print("n_agents: %d, n_actions: %d" % (n_agents, n_actions))
    n_episodes = 10
    for e in range(n_episodes):
        env.reset()
        terminated = False
        episode_reward = 0
        while not terminated:
            # obs: list, length = 8, for 8 agents
            obs = env.get_obs()
            # state: shape (168, )
            state = env.get_state()
            actions = []
            for agent_id in range(n_agents):  # 对于每个智能体遍历循环
                avail_actions = env.get_avail_agent_actions(agent_id)
                avail_actions_ind = np.nonzero(avail_actions)[0]
                action = np.random.choice(avail_actions_ind)
                actions.append(action)

            reward, terminated, _ = env.step(actions)
            episode_reward += reward

        print("Total reward in episode {} = {}".format(e, episode_reward))

    env.close()


if __name__ == "__main__":
    main()
```

# SMAC 简介

## StarCraft II

SMAC 基于流行的即时战略 (RTS) 游戏[StarCraft II](http://us.battle.net/sc2/en/game/guide/whats-sc2)， 由[Blizzard 暴雪公司](http://blizzard.com/)编写。在星际争霸 II 的常规完整游戏中，一个或多个人相互竞争或与内置游戏 AI 竞争，通过收集资源、建造建筑物和组建军队以击败对手。

与大多数 RTS 类似，星际争霸有两个主要的游戏组件：宏观管理和微观管理。

- 宏观管理（macro）是指高层次的战略考虑，例如经济和资源管理。
- 微观管理（micro）是指对个体单位进行细粒度的控制。

### Micromanagement

星际争霸已被用作 AI 的研究平台，最近还被用作 RL。通常，该游戏被设计为一个竞争性问题：agent 扮演人类玩家的角色，做出宏观管理决策并作为木偶操纵者执行微观管理，从中央控制器向各个单元发出命令。

为了构建一个丰富的multi-agent测试平台，我们反而只关注微观管理。Micro 是 StarCraft 游戏玩法的一个重要方面，具有很高的技能上限，业余和职业玩家都在孤立地练习。对于 SMAC，我们通过提出专为分散控制设计的问题的修改版本来利用微观管理的自然multi-agent结构。特别是，我们要求每个单元都由一个独立的agent控制，该agent仅以局限于以该单元为中心的有限视野的局部观察为条件。这些agent必须接受训练以解决具有挑战性的战斗场景，在游戏内置脚本 AI 的集中控制下与敌军作战。

战斗中适当的微型单位将最大化对敌方单位造成的伤害，同时最小化受到的伤害，并且需要一系列技能。例如，一个重要的技术是集中火力，即命令单位联合攻击并逐个击杀敌方单位。集中火力时，重要的是要避免过度杀伤：对单位造成的伤害超过杀死他们所需的伤害。

其他常见的微观管理技术包括：根据装甲类型将单位编成队形，让敌方单位在保持足够距离的情况下进行追击，从而造成很少或不造成伤害（风筝），协调单位的位置以从不同方向攻击或利用的地形来击败敌人。

在部分可观察性下学习这些丰富的合作行为是一项具有挑战性的任务，可用于评估multi-agent强化学习 (MARL) 算法的有效性。

## SMAC

SMAC 使用[StarCraft II Learning Environment](https://github.com/deepmind/pysc2)引入合作 MARL 环境。

### 场景

SMAC 由一组星际争霸 II 微场景组成，旨在评估独立agent学习协调以解决复杂任务的能力。这些场景经过精心设计，需要学习一种或多种微观管理技术来击败敌人。每个场景都是两支军队之间的对抗。每支军队的初始位置、数量和单位类型因场景而异，是否存在高地或无法通行的地形也是如此。

第一军由博学的盟军agent控制。第二支军队由内置游戏 AI 控制的敌方单位组成，该 AI 使用精心设计的非学习启发式算法。在每一集的开始，游戏 AI 会指示其单位使用其脚本策略攻击盟军agent。当任一军队的所有单位都死亡或达到预先指定的时间限制时，一集结束（在这种情况下，游戏被视为盟军agent的失败）。每个场景的目标是最大化学习策略的获胜率，即获胜游戏与玩游戏的预期比率。为了加快学习速度，敌方 AI 单位被命令在每一集开始时攻击agent的产卵点。

也许最简单的场景是对称的战斗场景。这些场景中最直接的是同质的，即每支军队只由一种单位类型（例如，海军陆战队）组成。在这种情况下，一个获胜的策略是集中火力，最好不要过度杀伤。 异构对称场景，其中每一方都有不止一种单位类型（例如，Stalkers and Zealots），更难解决。当一些单位对其他单位非常有效时，这些挑战特别有趣（这被称为反击)，例如，通过对特定装甲类型造成额外伤害。在这种情况下，盟军agent必须推断出游戏的这一属性，并设计一种智能策略来保护易受某些敌人攻击的队友。

SMAC 还包括更具挑战性的场景，例如，敌军在数量上超过盟军一个或多个单位。在这种不对称的情况下，必须考虑敌方单位的健康状况，以便有效地瞄准所需的对手。

最后，SMAC 提供了一组有趣的微技巧挑战，需要更高层次的合作和特定的微操作技巧才能击败敌人。挑战场景的一个例子是2m_vs_1z（又名 Marine Double Team），其中两名 Terran Marines 需要击败敌方 Zealot。在这种情况下，海军陆战队必须设计一种策略，不允许狂热者接近他们，否则他们几乎会立即死亡。另一个例子是so_many_banelings, 7 个 allied Zealots 面对 32 个敌方爆虫单位。爆虫通过跑向目标进行攻击，并在到达目标时爆炸，对目标周围的特定区域造成伤害。因此，如果大量的爆虫攻击距离很近的一小撮狂热者，狂热者就会被瞬间击败。因此，最佳策略是合作地围绕地图分散开来，彼此远离，以便 Banelings 的伤害分布尽可能薄。corridor 场景中，6 个友方 Zealots 面对 24 个敌方 Zerglings，需要agent有效利用地形特征。具体来说，agent们应该集体隔离阻塞点（地图的狭窄区域）以阻止来自不同方向的敌人攻击。一些微技巧挑战的灵感来自暴雪发布的[星际争霸大师挑战任务。](http://us.battle.net/sc2/en/blog/4544189/new-blizzard-custom-game-starcraft-master-3-1-2012)

完整的挑战列表如下所示。游戏AI的难度设置为非常困难（7）。然而，我们的实验表明，此设置确实会显着影响内置启发式的单元微观管理。

|       Name        |             Ally Units             |            Enemy Units             |             Type              |
| :---------------: | :--------------------------------: | :--------------------------------: | :---------------------------: |
|        3m         |             3 Marines              |             3 Marines              |    homogeneous & symmetric    |
|        8m         |             8 Marines              |             8 Marines              |    homogeneous & symmetric    |
|        25m        |             25 Marines             |             25 Marines             |    homogeneous & symmetric    |
|       2s3z        |       2 Stalkers & 3 Zealots       |       2 Stalkers & 3 Zealots       |   heterogeneous & symmetric   |
|       3s5z        |      3 Stalkers &  5 Zealots       |      3 Stalkers &  5 Zealots       |   heterogeneous & symmetric   |
|        MMM        | 1 Medivac, 2 Marauders & 7 Marines | 1 Medivac, 2 Marauders & 7 Marines |   heterogeneous & symmetric   |
|     5m_vs_6m      |             5 Marines              |             6 Marines              |   homogeneous & asymmetric    |
|     8m_vs_9m      |             8 Marines              |             9 Marines              |   homogeneous & asymmetric    |
|    10m_vs_11m     |             10 Marines             |             11 Marines             |   homogeneous & asymmetric    |
|    27m_vs_30m     |             27 Marines             |             30 Marines             |   homogeneous & asymmetric    |
|   3s5z_vs_3s6z    |       3 Stalkers & 5 Zealots       |       3 Stalkers & 6 Zealots       |  heterogeneous & asymmetric   |
|       MMM2        | 1 Medivac, 2 Marauders & 7 Marines | 1 Medivac, 3 Marauders & 8 Marines |  heterogeneous & asymmetric   |
|     2m_vs_1z      |             2 Marines              |              1 Zealot              | micro-trick: alternating fire |
|     2s_vs_1sc     |             2 Stalkers             |          1 Spine Crawler           | micro-trick: alternating fire |
|     3s_vs_3z      |             3 Stalkers             |             3 Zealots              |      micro-trick: kiting      |
|     3s_vs_4z      |             3 Stalkers             |             4 Zealots              |      micro-trick: kiting      |
|     3s_vs_5z      |             3 Stalkers             |             5 Zealots              |      micro-trick: kiting      |
|     6h_vs_8z      |            6 Hydralisks            |             8 Zealots              |    micro-trick: focus fire    |
|     corridor      |             6 Zealots              |            24 Zerglings            |     micro-trick: wall off     |
|   bane_vs_bane    |     20 Zerglings & 4 Banelings     |     20 Zerglings & 4 Banelings     |   micro-trick: positioning    |
| so_many_banelings |             7 Zealots              |            32 Banelings            |   micro-trick: positioning    |
|    2c_vs_64zg     |             2 Colossi              |            64 Zerglings            |   micro-trick: positioning    |
|      1c3s5z       | 1 Colossi & 3 Stalkers & 5 Zealots | 1 Colossi & 3 Stalkers & 5 Zealots |   heterogeneous & symmetric   |

### 状态和观察

在每个时间步，agent都会收到在其视野内绘制的局部观察结果。这包含有关每个单元周围的圆形区域内的地图信息，其半径等于视线范围。从每个agent的角度来看，视线范围使环境可以部分观察到。agent只能观察其他agent，如果他们都活着并且位于视线范围内。因此，agent无法确定他们的队友是在远处还是已经死亡。

每个agent观察到的特征向量包含视线范围内的友军和敌军单位的以下属性：_distance_, _relative x_, _relative y_, _health_, _shield_, and _unit\_type_ <sup>[1](#myfootnote1)</sup>. 护盾是额外的保护源，需要在对单位的健康造成任何损害之前将其移除。所有 Protos 单位都有护盾，如果没有造成新的伤害，护盾可以再生（其他两个种族的单位没有这个属性）。此外，agent可以访问视野中盟军单位的最后行动。最后，智能体可以观察周围的地形特征；特别是，固定半径处的八个点的值表示高度和步行能力。

全局状态仅在集中训练期间对agent可用，包含地图上所有单元的信息。具体来说，状态向量包括所有agent相对于地图中心的坐标，以及观察中存在的单元特征。此外，状态存储Medivacs 的能量和其余盟军单位的冷却时间，这代表攻击之间的最小延迟。最后，所有agent的最后一个动作都附加到中央状态。

所有特征，无论是在状态中还是在个体agent的观察中，都由它们的最大值归一化。所有agent的视线范围都设置为 9。

1：agent控制的单位的健康、护盾和单位类型也包括在观察中

### 动作空间

允许agent采取的离散动作集包括 move[direction]（四个方向：北、南、东或西）、attack[enemy_id]、stop和no-op。死去的agent只能采取空操作，而活agent不能。作为治疗单位，Medivacs 必须使用heal[agent_id]动作而不是attack[enemy_id]。agent可以执行的最大动作数量在 7 到 70 之间，具体取决于场景。

为确保任务的分散化，agent只能对射击范围内的敌人使用attack[enemy_id]动作。这限制了该单位对远处敌人使用内置攻击移动宏动作的能力。我们将射击范围设置为 6。拥有比射击范围更大的视野迫使agent在开始射击之前使用移动命令。

### 奖励

总体目标是在每个战斗场景中获得最高的胜率。我们为sparse rewards提供了相应的选项，这将导致环境仅返回 +1 的奖励（获胜）和 -1（失败）。然而，我们还提供了一个默认设置，用于根据agent人造成和收到的生命值伤害计算的形状奖励信号，在杀死敌方（盟军）单位后的一些正（负）奖励和/或正（负）奖励赢得（失败）战斗。可以使用一系列标志配置这种形状奖励的确切值和比例，但我们强烈反对奖励函数的虚伪工程（例如，针对不同场景调整不同的奖励函数）。

### 环境设置

SMAC 使用[星际争霸 II 学习环境](https://arxiv.org/abs/1708.04782)(SC2LE) 与星际争霸 II 引擎进行通信。SC2LE 通过允许发送命令和接收来自游戏的观察来提供对游戏的完全控制。然而，SMAC 在概念上不同于 SC2LE 的 RL 环境。SC2LE 的目标是学习玩星际争霸 II 的完整游戏。这是一项竞争性任务，其中集中式 RL agent接收 RGB 像素作为输入，并通过类似于人类玩家的玩家级控制来执行宏观和微观。另一方面，SMAC 代表一组协作的多agent微挑战，其中每个学习agent控制一个军事单位。

SMAC 使用SC2LE 的原始 API。原始 API 观察没有任何图形组件，包括地图上有关单位的信息，例如健康状况、位置坐标等。原始 API 还允许使用单位 ID 向各个单位发送操作命令。这种设置不同于人类玩实际游戏的方式，但便于设计分散的多智能体学习任务。

由于我们的微场景比实际的星际争霸 II 游戏短，因此在每集之后重新启动游戏会出现计算瓶颈。为了解决这个问题，我们使用 API 的调试命令。具体来说，当任一军队的所有单位都被杀死时，我们通过发送调试操作杀死所有剩余的单位。没有剩余的单位会启动一个由星际争霸 II 编辑器编程的触发器，该触发器会在其原始位置重新生成所有单位并保持完整的生命值，从而快速有效地重新启动场景。

此外，为了鼓励agent自己探索有趣的微观策略，我们限制了星际争霸 AI 对我们agent的影响。具体来说，我们禁用了针对攻击特工或位于附近的敌人的自动单位攻击。为此，我们使用了用星际争霸 II 编辑器创建的新单位，它们是现有单位的精确副本，并修改了两个属性：战斗：默认获取级别设置为被动（默认为进攻）和行为：响应设置为无响应（默认获取）。这些字段仅针对盟军单位进行修改；敌方单位不变。

瞄准镜和射程值可能与某些星际争霸 II 装置的内置瞄准镜或射程属性不同。我们的目标不是掌握原始的完整星际争霸游戏，而是对分散控制的 MARL 方法进行基准测试。

#### 环境设置

- difficulty  难度等级,官方默认是7

```shell
difficulties = {
    "1": sc_pb.VeryEasy,
    "2": sc_pb.Easy,
    "3": sc_pb.Medium,
    "4": sc_pb.MediumHard,
    "5": sc_pb.Hard,
    "6": sc_pb.Harder,
    "7": sc_pb.VeryHard,
    "8": sc_pb.CheatVision,
    "9": sc_pb.CheatMoney,
    "A": sc_pb.CheatInsane,
}
```

- map (按难度分成三个等级)

- -  Easy scenarios（2s_vs_1sc/2s3z/3s5z/1c3s5z/10m_vs_11m）
  - Hard scenarios (2c_vs_64zg/bane_vs_bane/5m_vs_6m/3s5z)
  - Super hard(3s5z_vs_3s6z/6h_vs_8z/27m_vs_30m/MMM2/corridor) 

# 变换前的空间（原始环境）

## 观察空间

-   可以获取各个智能体是否存活，各个智能体剩余血量，各个智能体视野范围内的盟友或敌人等零碎的信息。

## 动作空间

-   游戏操作按键空间，一般是大小为 N
    的离散动作空间（N随具体子环境变化），数据类型为`int`，需要传入
    python 数值（或是 0 维 np 数组，例如动作 3 为`np.array(3)`）
-   对于各个地图，动作空间 N 一般等于 6+敌人数，如 3s5z 地图中为
    14，2c_vs_64zg 地图中为70。具体的含义是：
    -   0：NOOP
    -   1：STOP
    -   2：MOVE_NORTH
    -   3：MOVE_SOUTH
    -   4：MOVE_EAST
    -   5：MOVE_WEST
    -   6-N: ATTACK ENEMY，所攻击的敌人的 ID 为 N-6

## 奖励空间

-   游戏胜负，胜利为 1，失败为 0，一般是一个`int`数值。

## 其他

-   游戏结束即为当前环境 episode 结束

# 关键事实

1.  输入为将离散信息综合后的信息
2.  离散动作空间
3.  奖励为稀疏奖励，我们设置
    fake_reward，使得训练时所用的奖励为稠密奖励。

# 变换后的空间（RL 环境）

## 观察空间

-   变换内容：拼接各个 agent 看到的各类离散信息，将拼接后的信息作为各个
    agent 看到的 agent_state 和全局的 global_state
-   变换结果：一个 dict 型数据，其中包含
    agent_state，global_state和action_mask，均为一个一维 Tensor 型数组

## 动作空间

-   基本无变换，依然是大小为N的离散动作空间

## 奖励空间

-   变换内容：设置
    fake_reward，使得智能体在作出一些动作后就可以获得奖励，我们设置每一步的
    fake_reward为"打掉的敌人血量-损失的己方血量"，且消灭一个敌人奖励 20
    分，获取全局的胜利获得 200 分
-   变换结果：一个一维且只包含一个 float32 类型数据的 Tensor

## 其他

-   开启`special_global_state`返回的 global_state 则为各个全局信息 +
    各个 agent 特殊信息拼接成的信息，若不开启，则仅返回全局信息
-   开启`special_global_state`且开启`death_mask`，则若一个agent阵亡，则其返回的
    global_state 仅包含其自身的 ID 信息，其余信息全部被屏蔽
-   环境`step`方法返回的`info`必须包含`eval_episode_return`键值对，表示整个
    episode 的评测指标，在 SMAC 中为整个 episode 的 fake_reward 累加和
-   环境`step`方法最终返回的`reward`为胜利与否

# 其他

## 惰性初始化

为了便于支持环境向量化等并行操作，环境实例一般实现惰性初始化，即`__init__`方法不初始化真正的原始环境实例，只是设置相关参数和配置值，在第一次调用`reset`方法时初始化具体的原始环境实例。

## 随机种子

-   环境中有两部分随机种子需要设置，一是原始环境的随机种子，二是各种环境变换使用到的随机库的随机种子（例如`random`，`np.random`）
-   对于环境调用者，只需通过环境的`seed`方法进行设置这两个种子，无需关心具体实现细节
-   环境内部的具体实现：对于原始环境的种子，在调用环境的`reset`方法内部，具体的原始环境`reset`之前设置
-   环境内部的具体实现：对于随机库种子，则在环境的`seed`方法中直接设置该值
