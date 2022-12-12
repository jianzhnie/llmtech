 # SMAC

# 概述

SMAC
是一个用于在暴雪星际争霸2上进行多智能体协同强化学习（MARL）的环境。SMAC
用了暴雪星际争霸2 的机器学习 API 和 DeepMind 的PySC2
为智能体与星际争霸2的交互提供了友好的接口，方便开发者观察和执行行动。 与
PySC2 相比，SMAC
专注于分散的微观操作方案，其中游戏的每个智能体均由单独的 RL agent控制。

![../_images/smac1.gif](https://opendilab.github.io/DI-engine/_images/smac1.gif)

# 安装

## 安装方法

需要安装星际争霸2 游戏和 PySC2
库.

安装主要包括两部分：

1.下载星际争霸2 游戏 对于 Linux
系统使用者，安装路径为<https://github.com/Blizzard/s2client-proto#downloads>，之后使用
`export SC2PATH=<sc2/installation/path>` 命令将安装路径添加到环境变量中
对于 Windows 系统使用者，安装请参考<https://starcraft2.com>

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

2.安装PySC2

``` shell
pip install pysc2
```

3.安装smac

```
pip install git+https://github.com/oxwhirl/smac.git
```

## 验证安装

安装完成后，可以通过安装成功后 `echo $SC2PATH` 确认环境变量设置成功

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
