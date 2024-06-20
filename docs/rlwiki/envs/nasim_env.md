# NASim

## 	介绍

网络攻击模拟器（NASim）是一个轻量级的网络攻击模拟器，用Python编写。它被设计用来快速测试使用强化学习和规划的自主渗透测试Agent。它是一个模拟器，因此没有复制攻击真实系统的所有细节，而是旨在捕捉一些网络渗透测试的更显著特征，如状态和动作空间的规模变化、部分可观察性和多样化的网络拓扑。

环境是模仿 [gymnasium ](https://github.com/Farama-Foundation/Gymnasium/)（以前称为Open AI Gym）接口建模的。

Github：https://github.com/Jjschwartz/NetworkAttackSimulator

## 安装

### 依赖项

该框架经过测试，可以在 Python 3.7 或更高版本下运行。

所需的依赖项：

- Python >= 3.7
- Gym >= 0.17
- NumPy >= 1.18
- PyYaml >= 5.3

对于渲染：

- NetworkX >= 2.4
- prettytable >= 0.7.2
- Matplotlib >= 3.1.3

### 运行安装

```python
pip install nasim
```

## 启动 NASim 环境

与 NASim 的交互主要通过类 [`NASimEnv`](https://networkattacksimulator.readthedocs.io/en/latest/reference/envs/environment.html#nasim.envs.environment.NASimEnv) 完成，该类处理所选scenario定义的模拟网络环境。

有两种方法可以启动新环境：

- （i）直接通过 nasim 库，
- （ii）使用gymnasium 库的gym.make()函数。

在本教程中，我们将介绍第一种方法。对于第二种方法，请查看[使用 OpenAI gym 启动 NASim](https://networkattacksimulator.readthedocs.io/en/latest/tutorials/gym_load.html#gym-load-tute)。

### 环境设置

NASimEnv类的初始化需要一个scenario定义和三个可选参数。

scenario定义了网络属性和渗透测试人员的特定信息（例如可用的漏洞等）。

三个可选参数控制环境模式：

- `fully_obs` : 环境的可观察性模式，如果为True，则使用完全可观察模式，否则是部分可观察（默认=False）
- `flat_actions` : 如果为真，则使用平面动作空间，否则将使用参数化动作空间（默认=True）。
- `flat_obs` : 如果为真，则使用1D观察空间，否则使用2D观察空间（默认=True）

如果使用完全可观察模式 ( `fully_obs=True`)，则每一步之后都会观察到网络和攻击的整个状态。这是“简单”模式，不能反映渗透测试的现实，但它对于入门和健全性检查算法和环境很有用。当使用部分可观察模式 ( `fully_obs=False`) 时，Agent在开始时不知道网络上每个主机的位置、配置和值，并且只接收与每一步执行的操作直接相关的特征的观察结果。这是“困难”模式，更准确地反映了渗透测试的现实。

环境是否完全或部分可观察对动作和观察空间的大小和形状或Agent与环境的交互方式没有影响。

使用`flat_actions=True`意味着我们的操作空间由 N 个离散操作组成，其中 N 基于网络中的主机数量以及可用的漏洞和扫描数量。对于我们的示例，有 3 个主机、1 个漏洞和 3 个扫描（操作系统、服务和子网），总共有 3  (1 + 3) = 12 个操作。如果`flat_actions=False `每个操作都是一个向量，向量的每个元素都指定了操作的一个参数。

使用 `flat_obs=True` 意味着返回的观测值将是一个一维向量。否则，`flat_obs=False`观测值将是一个二维矩阵。

### 从 scenario 加载环境

NASim 环境可以通过三种方式从scenario构建：创建现有scenario、从 YAML 文件加载以及从参数生成。

####  基于现有scenario创建

这是加载新环境的最简单方法，与[OpenAI Gym ](https://github.com/openai/gym)方式非常吻合。加载现有scenario非常简单：

```python
import nasim
env = nasim.make_benchmark("tiny")
```

还可以使用种子参数传入随机种子，这将在使用生成的scenario时产生影响。

#### 从 YAML 文件加载scenario

如果希望加载 YAML 文件中定义的现有或自定义scenario，这也非常简单：

```python
import nasim
env = nasim.load("path/to/scenario.yaml")
```

#### 生成 scenario

加载新环境的最后一种方法是使用 NASim scenario生成器来生成它。有相当多的参数可用于控制生成什么scenario，但两个关键参数是网络中的主机数量和正在运行的服务数量（除非另有说明，否则它们还控制漏洞的数量）。

要生成一个包含 5 个主机并运行 3 个服务的新环境：

```python
import nasim
env = nasim.generate(5, 3)
```

如果您想要传递一些其他参数（例如可能的操作系统数量），可以将其作为关键字参数传递：

```python
env = nasim.generate(5, 3, num_os=3)
```

### 使用 OpenAI gym 启动 NASim

启动时，NASim 将每个基准测试 scenario注册为[Gymnasium](https://github.com/Farama-Foundation/Gymnasium/) 环境，从而允许使用 加载 NASim 基准测试环境 `gymnasium.make()`。

所有基准测试 scenario都可以使用 `gymnasium.make()` 加载。

```python
import gymnasium as gym
env = gym.make("nasim:TinyPO-v0")

# to specify render mode
env = gym.make("nasim:TinyPO-v0", render_mode="human")
```

## 与 NASim 环境交互

### 启动环境

首先是简单地加载环境：

```python
import nasim
# load my environment in the desired way (make_benchmark, load, generate)
env = nasim.make_benchmark("tiny")

# or using gym
import gymnasium as gym
env = gym.make("nasim:Tiny-PO-v0")
```

这里我们使用默认的环境参数：`fully_obs=False`、`flat_actions=True`和`flat_obs=True`。

可以从环境属性中检索操作的数量，`action_space`如下所示：

```python
# When flat_actions=True
num_actions = env.action_space.n

# When flat_actions=False
nvec_actions = env.action_space.nvec
```

可以从环境属性中检索观测的形状，`observation_space`如下所示：

```python
obs_shape = env.observation_space.shape
```

### Env.reset()

要重置环境并获取初步观察结果，请使用以下`reset()`函数：

```python
o, info = env.reset()
```

返回值`info`包含可选的辅助信息。

### Env.step()

可以使用函数在环境中采取一个步骤`step(action)`。该函数返回一个元组，分别对应于观察、奖励、完成、达到步骤限制、辅助信息：

```python
action = # integer in range [0, env.action_space.n]
o, r, done, step_limit_reached, info = env.step(action)
```

如果`done=True`，则目标已达到，情节结束。或者，如果当前 scenario有步骤限制，`step_limit_reached=True`那么步骤限制已达到。在这两种情况下，建议停止或重置环境，否则无法保证会发生什么（尤其是第一种情况）。

### Env.render()

您可以使用该`render()`函数来获取环境状态的可读可视化效果。要正确使用渲染，请确保将其传递`render_mode="human"`给环境初始化函数：

```python
import nasim
# load my environment in the desired way (make_benchmark, load, generate)
env = nasim.make_benchmark("tiny", render_mode="human")

# or using gym
import gymnasium as gym
env = gym.make("nasim:Tiny-PO-v0", render_mode="human")

env.reset()
# render the environment
# (if render_mode="human" is not passed during initialization this will do nothing)
env.render()
```

### Agent示例

目录中提供了一些示例Agent`nasim/agents`。以下是一个假设Agent与环境交互的简单示例：

```python
import nasim

env = nasim.make_benchmark("tiny")

agent = AnAgent(...)

o, info = env.reset()
total_reward = 0
done = False
step_limit_reached = False
while not done and not step_limit_reached:
    a = agent.choose_action(o)
    o, r, done, step_limit_reached, info = env.step(a)
    total_reward += r

print("Done")
print("Total reward =", total_reward)
```

## 理解  Scenarios

NASim 中的 Scenarios 定义了创建网络环境所需的所有属性。每个 scenario定义可分为两个部分：网络配置和渗透测试器。

### 网络配置

网络配置由以下属性定义：

- subnets：网络中子网的数量和大小。
- topology：网络中不同子网如何连接
- host configurations：网络中每台主机的地址、操作系统、服务、进程和防火墙
- firewall：阻止子网之间的通信

请注意，对于主机配置，我们通常只对渗透测试人员可以利用的服务和进程感兴趣，因此我们通常会忽略任何不易受攻击的服务和进程，以减少问题规模。

### 渗透测试器

渗透测试器的定义如下：

- exploits：渗透测试人员可以利用的漏洞集
- privescs：渗透测试人员可用的一组权限提升操作
- scan costs：执行每种类型扫描（服务、操作系统、进程和子网）的成本
- sensitive hosts：网络上的目标主机及其价值

### 示例 scenario

为了说明这些属性，我们展示了一个示例 scenario，其中渗透测试人员的目的是获得敏感子网中的服务器和用户子网中一台主机的 root 访问权限。

下图显示了我们的示例网络的布局。

![示例网络](https://networkattacksimulator.readthedocs.io/en/latest/_images/example_network.png)



从图中我们可以看出这个网络有以下性质：

- subnets：三个子网：具有单个服务器的 DMZ、具有单个服务器的敏感子网和具有三个用户机器的用户子网。
- topology：只有 DMZ 连接到互联网，而网络中的所有子网都互连。
- host configurations：每个主机旁边显示每个主机上运行的地址、操作系统、服务和进程（例如，DMZ 子网中的服务器的地址为 (1, 0)，具有 Linux 操作系统，正在运行 http 和 ssh 服务以及 tomcat 进程）。主机防火墙设置显示在图右上角的表格中。这里只有主机(1, 0)配置了防火墙，阻止来自主机(3, 0)和(3, 1)的任何 SSH 连接。
- firewall：防火墙上方和下方的箭头表示子网之间以及 DMZ 子网与互联网之间各个方向可以通信的服务（例如，互联网可以与 DMZ 中的主机上运行的 http 服务进行通信，而防火墙不会阻止从 DMZ 到互联网的通信）。

接下来我们需要定义渗透测试器，我们根据想要模拟的 scenario来指定。

#### exploits

漏洞利用：在这种情况下，渗透测试人员可以使用三种漏洞利用

1. ssh_exploit：利用 windows 机器上运行的 ssh 服务，成本为 2，成功概率为 0.6，成功后可获得用户级访问权限。
2. ftp_exploit：利用在 Linux 机器上运行的 ftp 服务，成本为 1，成功概率为 0.9，如果成功则可获得 root 级别访问权限。
3. http_exploit：利用任何操作系统上运行的 http 服务，成本为 3，成功概率为 1.0，如果成功则可获得用户级访问。

#### privescs

privescs：在这种情况下，渗透测试人员可以执行两项权限提升操作

1. pe_tomcat：利用 Linux 机器上运行的 tomcat 进程获取 root 访问权限。成本为 1，成功概率为 1.0。
2. pe_daclsvc：利用 Windows 机器上运行的 daclsvc 进程获取 root 访问权限。成本为 1，成功概率为 1.0。

#### scan costs

扫描成本：这里我们需要指定每种扫描类型的成本

1. 服务扫描：1
2. 操作系统扫描：2
3. 进程扫描：1
4. 子网扫描：1

#### sensitive hosts

敏感主机：这里有两个目标主机

1. （2, 0），1000：在敏感子网上运行的服务器，其值为 1000。
2. （3, 2），1000：用户子网上运行的最后一个主机，其值为 1000。

这样，我们的 scenario就完全定义了，我们拥有运行攻击模拟所需的一切。

## 创建自定义 scenario

使用 NASim，可以使用有效 YAML 文件中定义的自定义 scenario。在本教程中，我们将介绍如何创建和运行您自己的自定义 scenario。

### 使用 YAML 定义自定义 scenario

在我们开始编写新的自定义 YAML  scenario之前，值得看一些示例。NASim 附带了许多基准 YAML  scenario，可以在目录中找到`nasim/scenarios/benchmark`, [或在 github 上查看](https://github.com/Jjschwartz/NetworkAttackSimulator/tree/master/nasim/scenarios/benchmark)。在本教程中，我们将使用该`tiny.yaml` scenario作为示例。

NASim 中的自定义 scenario需要定义组件：网络和渗透测试器。

### 定义网络

该网络由以下部分定义：

> 1. 子网：网络中每个子网的大小
> 2. 拓扑：定义哪些子网相互连接的邻接矩阵
> 3. os：网络上可用的操作系统的名称
> 4. services：网络上可用服务的名称
> 5. 进程：网络上可用进程的名称
> 6. hosts：网络上的主机及其配置的字典
> 7. 防火墙：子网防火墙的定义

#### 子网

此属性定义网络上的子网数量以及每个子网的大小。它简单地定义为一个有序的整数列表。列表中第一个子网的地址为1，第二个子网为2，依此类推。地址0为“互联网”子网保留（请参阅下面的拓扑部分）。例如，`tiny`网络包含 3 个子网，大小均为 1：

```python
subnets: [1, 1, 1]

# or alternatively

subnets:
  - 1
  - 1
  - 1
```

#### 拓扑

拓扑结构由邻接矩阵定义，其中一行和一列表示网络中的每个子网，另外还有一行和一列表示“互联网”子网，即与网络外部的连接。第一行和第一列为“互联网”子网保留。子网之间的连接用 表示，`1`而非连接用 表示`0`。请注意，我们假设连接是对称的，并且子网与自身相连。

对于`tiny`网络，子网1是公共子网，因此连接到互联网，由`1`第 1 行第 2 列和第 2 行第 1 列中的 表示。子网1还与子网2和3相连，由相关单元格中的 表示`1`，同时子网2和3是私有的，不直接连接到互联网，由`0`值表示。

```python
topology: [[ 1, 1, 0, 0],
           [ 1, 1, 1, 1],
           [ 0, 1, 1, 1],
           [ 0, 1, 1, 1]]
```

#### 操作系统、服务、进程

与我们定义子网列表的方式类似，操作系统、服务和进程由一个简单的列表定义。每个列表中任何项目的名称都可以是任何名称，但请注意，它们将用于验证主机配置、漏洞利用等，因此只需根据需要与这些值匹配即可。

```shell
os:
  - linux
services:
  - ssh
processes:
  - tomcat
```

继续我们的示例，该`tiny` scenario包括一个操作系统：linux，一个服务：ssh和一个进程：tomcat：

#### 主机配置

主机配置部分是从主机地址到其配置的映射，其中地址是一个元组，配置必须包括主机操作系统、运行的服务、运行的进程和可选的主机防火墙设置。`(subnet number, host number)`

定义主机时需要注意以下几点：

> 1. 每个子网定义的主机数量需要与每个子网的大小相匹配
> 2. 子网内的主机地址必须从 开始`0`并从那里开始计数（即，子网1中的三个主机的地址为、和）`(1, 0)``(1, 1)``(1, 2)`
> 3. 任何操作系统、服务和进程的名称都必须与YAML 文件的os、services和processes部分中提供的值匹配。
> 4. 每个主机必须有一个操作系统和至少一个正在运行的服务。主机可以没有正在运行的进程（可以使用空列表来表示`[]`）。

主机防火墙被定义为从主机地址到拒绝来自该主机的服务列表的映射。主机地址必须是网络中主机的有效地址，并且任何服务也必须与服务部分中定义的服务相匹配。最后，如果主机地址不是防火墙的一部分，则假定在主机级别允许来自该主机的所有流量（子网防火墙仍可能阻止该流量）。

主机值是Agent在入侵主机时将收到的可选值。与sensitive_hosts部分不同，此值可以是负数，也可以是零和正数。这使得可以设置额外的主机特定奖励或惩罚，例如为网络上的“蜜罐”主机设置负奖励。需要注意以下几点：

> 1. 主机值是可选的，默认为 0。
> 2. 对于任何敏感主机，该值要么未指定，要么必须与文件的sensitive_hosts部分中指定的值匹配。
> 3. 与敏感主机相同，Agent只有在破坏主机时才会收到价值作为奖励。

这是该 scenario的示例主机配置部分`tiny`，其中主机防火墙仅为主机定义，并且主机的值为（注意，在这种情况下我们可以不指定值以获得相同的结果，我们在此处将其作为示例）：`(1, 0)``(1, 0)``0`

```python
host_configurations:
  (1, 0):
    os: linux
    services: [ssh]
    processes: [tomcat]
    # which services to deny between individual hosts
    firewall:
      (3, 0): [ssh]
    value: 0
  (2, 0):
    os: linux
    services: [ssh]
    processes: [tomcat]
    firewall:
      (1, 0): [ssh]
  (3, 0):
    os: linux
    services: [ssh]
    processes: [tomcat]
```

#### 防火墙

定义网络的最后一部分是防火墙，它被定义为从三元组到允许的服务列表的映射。定义防火墙时需要注意以下几点：`(subnet number, subnet number)`

> 1. 防火墙规则只能在拓扑邻接矩阵中连接的子网之间定义。
> 2. 每条规则定义单向允许哪些服务，从元组中的第一个子网到元组中的第二个子网（即（源子网，目标子网））
> 3. 空列表意味着从源到目的地的所有流量都将被阻止

`tiny`这是允许所有子网之间进行 SSH 流量（子网 1 到 0 和子网 1 到 2 除外）的 scenario的防火墙定义。

```python
# two rows for each connection between subnets as defined by topology
# one for each direction of connection
# lists which services to allow
firewall:
  (0, 1): [ssh]
  (1, 0): []
  (1, 2): []
  (2, 1): [ssh]
  (1, 3): [ssh]
  (3, 1): [ssh]
  (2, 3): [ssh]
  (3, 2): [ssh]
```

至此，我们已经涵盖了定义 scenario网络所需的一切。接下来是定义渗透测试器。

### 定义渗透测试人员

渗透测试器由以下部分定义：

> 1. sensitive_hosts：包含敏感/目标主机地址及其值的字典
> 2. 功绩：功绩词典
> 3. privilege_escalation：特权升级操作词典
> 4. os_scan_cost：使用操作系统扫描的成本
> 5. service_scan_cost：使用服务扫描的成本
> 6. process_scan_cost：使用进程扫描的成本
> 7. subnet_scan_cost：使用子网扫描的成本
> 8. step_limit：渗透测试人员在单次测试中可以执行的最大操作数

#### 敏感主机

此部分指定网络中目标主机的地址和值。当渗透测试人员获得这些主机的 root 访问权限时，他们将收到指定的值作为奖励。sensitive_hosts 部分是一个字典，其中的条目是地址、值对。其中地址是一个元组，值是一个非负浮点数或整数。`(subnet number, host number)`

在此`tiny` scenario中，渗透测试人员的目标是获取主机和的 root 访问权限，这两个主机的值均为 100：`(2, 0)``(3, 0)`

```python
sensitive_hosts:
  (2, 0): 100
  (3, 0): 100
```

#### 漏洞

漏洞利用部分是一本将漏洞利用名称映射到漏洞利用定义的字典。每个 scenario至少需要一个漏洞利用。漏洞利用定义是一本必须包含以下条目的字典：

> 1. service
>
>    ：漏洞所针对的服务的名称。
>
>    - 请注意，该值必须与网络定义的服务部分中定义的服务名称匹配。
>
> 2. os
>
>    ：漏洞所针对的操作系统的名称，或
>
>    ```
>    none
>    ```
>
>    漏洞是否适用于所有操作系统。
>
>    - 如果值不是，则必须与网络定义的操作系统`none`部分中定义的操作系统的名称匹配
>
> 3. prob：在满足所有先决条件的情况下，漏洞利用成功的概率（即目标主机被发现且可访问，并且主机正在运行目标服务和操作系统）
>
> 4. cost：执行操作的成本。这应该是非负整数或浮点数，可以以任何方式表示操作的成本（财务、时间、产生的流量等）
>
> 5. access：漏洞利用成功后，渗透测试人员将获得对目标主机的访问权限。该权限可以是user或root。

漏洞的名称可以是任何您想要的名称，只要它们是不可变的、可哈希的（即字符串、整数、元组）并且是唯一的。

示例`tiny` scenario只有一个漏洞`e_ssh`，该漏洞针对的是 Linux 主机上运行的 SSH 服务，成本为 1，并导致用户级别访问：

```python
exploits:
  e_ssh:
    service: ssh
    os: linux
    prob: 0.8
    cost: 1
    access: user
```

#### 权限提升

与漏洞利用部分类似，特权升级部分是一本将特权升级操作名称映射到其定义的字典。特权升级操作定义是一本字典，必须包含以下条目：

> 1. process
>
>    ：操作所针对的进程的名称。
>
>    - 该值必须与网络定义的进程部分中定义的进程的名称匹配。
>
> 2. os
>
>    ：操作所针对的操作系统的名称，或
>
>    ```
>    none
>    ```
>
>    漏洞是否适用于所有操作系统。
>
>    - 如果值不是，则它必须与网络定义的os`none`部分中定义的 OS 的名称匹配。
>
> 3. prob：在满足所有先决条件的情况下，操作成功的概率（即渗透测试人员可以访问目标主机，并且主机正在运行目标进程和操作系统）
>
> 4. cost：执行操作的成本。这应该是非负整数或浮点数，可以以任何方式表示操作的成本（财务、时间、产生的流量等）
>
> 5. access：操作成功后，渗透测试人员将获得对目标主机的访问权限。可以是user或root。

与漏洞利用类似，每个特权漏洞利用操作的名称可以是任何您想要的名称，只要它们是不可变的、可散列的（即字符串、整数、元组）并且是唯一的。

笔记

 scenario不需要定义任何特权升级操作。在这种情况下，将特权升级部分定义为空：。`privilege_escalation: {}`

但请注意，您需要确保仅通过使用漏洞就可以获得敏感主机的 root 访问权限，否则渗透测试人员将永远无法达到目标。

示例`tiny` scenario有一个单一的权限提升操作`pe_tomcat`，该操作针对在 Linux 主机上运行的 tomcat 进程，成本为 1，并导致 root 级别访问：

```python
privilege_escalation:
  pe_tomcat:
    process: tomcat
    os: linux
    prob: 1.0
    cost: 1
    access: root
```

#### 扫描费用

每次扫描都必须有一个相关的非负成本。此成本可以代表任何您希望的值，并且将计入Agent每次执行扫描时收到的奖励中。

扫描成本很容易定义，只需要一个非负浮点数或整数值。您必须指定所有扫描的成本。在这里，在示例`tiny` scenario中，我们为所有扫描定义成本为 1：

```python
service_scan_cost: 1
os_scan_cost: 1
subnet_scan_cost: 1
process_scan_cost: 1
```

#### 步数限制

步骤限制定义了渗透测试人员在单个 scenario中达到目标的最大步骤数（即操作数）。在模拟过程中，一旦达到步骤限制，则认为该 scenario已完成，Agent未能达到目标。

定义步长限制很容易，因为它只需要一个正整数值。例如，这里我们为以下 scenario定义步长限制为 1000 `tiny`：

```python
step_limit: 1000
```

这样我们就有了定义自定义 scenario所需的一切。

### 运行自定义 YAML  scenario

`NASimEnv`要从自定义 YAML  scenario文件创建，我们使用以下`nasim.load()`函数：

```python
import nasim
env = nasim.load('path/to/custom/scenario.yaml`)
```

加载函数还采用一些附加参数来控制环境的观察模式和观察和动作空间，请参阅[NASimEnv 加载参考](https://networkattacksimulator.readthedocs.io/en/latest/reference/load.html#nasim-init)以获取参考和[环境设置](https://networkattacksimulator.readthedocs.io/en/latest/tutorials/loading.html#env-params)以获取解释。

如果您的文件格式存在任何问题，您在尝试加载文件时应该会收到一些有用的错误消息。成功加载环境后，您可以正常与其交互（有关更多详细信息，请参阅[与 NASim 环境交互](https://networkattacksimulator.readthedocs.io/en/latest/tutorials/environment.html#env-tute)）。

##  scenario生成说明

生成 scenario涉及许多设计决策，这些决策在很大程度上决定了要生成的网络的形式。本文档旨在解释使用[ scenario生成器](https://networkattacksimulator.readthedocs.io/en/latest/reference/scenarios/generator.html#scenario-generator)类生成 scenario的一些技术细节。

 scenario生成器主要基于先前的研究，具体来说：

- [Sarraute, Carlos, Olivier Buffet, and Jörg Hoffmann. “POMDPs make better hackers: Accounting for uncertainty in penetration testing.” Twenty-Sixth AAAI Conference on Artificial Intelligence. 2012.](https://www.aaai.org/ocs/index.php/AAAI/AAAI12/paper/viewPaper/4996)
- [Speicher, Patrick, et al. “Towards Automated Network Mitigation Analysis (extended).” arXiv preprint arXiv:1705.05088 (2017).](https://arxiv.org/abs/1705.05088)

### 网络拓扑结构

描述即将发布。在此之前，我们建议阅读上面链接的论文，尤其是 Speicher 等人 (2017) 的附录。

### 相关配置

生成 scenario时， scenario`uniform=False`将与主机配置相关联。这意味着，它运行的操作系统和服务不是从可用的操作系统和服务中随机均匀地选择，而是随机选择，并且会提高由较早生成配置的其他主机运行的操作系统和服务的概率。

具体来说，网络中每台主机的配置分布都是使用嵌套狄利克雷过程生成的，因此整个网络主机将具有相关的配置（即某些服务/配置在网络主机之间更为常见）。可以使用三个参数来控制相关性：`alpha_H`、`alpha_V`和`lambda_V`。

`alpha_H`并`alpha_V`控制相关程度，值越低，相关程度越大。

`lambda_V`控制每个主机运行的服务的平均数量，值越高意味着平均主机的服务越多（因此越容易受到攻击）。

所有三个参数都必须有一个正值，默认值为`alpha_H=2.0`、`alpha_V=2.0`和`lambda_V=1.0`，这往往会生成具有相当相关配置的网络，其中主机平均只有一个漏洞。

### 生成的漏洞概率

每个漏洞的成功概率根据参数的值确定`exploit_probs`，如下所示：

- `exploit_probs=None`- 从区间 (0, 1) 上的均匀分布随机生成的概率。
- `exploit_probs=float`- 每个漏洞的概率设置为浮点值，该值必须是一个有效的概率。
- `exploit_probs=list[float]`- 每个漏洞的概率设置为列表中相应的浮点值。这要求列表的长度与参数指定的漏洞数量相匹配`num_exploits`。
- `exploit_probs="mixed"`- 概率从基于2017 年[CVSS](https://www.first.org/cvss/v2/guide)[十大漏洞](https://go.recordedfuture.com/hubfs/reports/cta-2018-0327.pdf)攻击复杂度分布的集合分布中选择。具体而言，漏洞利用概率从 [0.3, 0.6, 0.9] 中选择，分别对应高、中、低攻击复杂度，概率为 [0.2, 0.4, 0.4]。

对于确定性漏洞设置`exploit_probs=1.0`。

### 防火墙

防火墙限制不同子网中的主机之间可以通信的服务。这主要是通过在每个子网之间随机选择要阻止的服务来实现的，但会有一些限制。

首先，用户区内子网间不存在防火墙，因此所有服务都允许不同用户子网上的主机间进行通信。

其次，阻止的服务数量由参数控制`restrictiveness`。这控制了区域之间（即互联网、DMZ、敏感和用户区域之间）阻止的服务数量。

第三，为了确保目标能够实现，每个区域之间将允许至少一个运行在每个子网上的服务的流量。这可能意味着将允许比限制性参数更多的服务。

## 解释

### 模拟与现实差距考虑

NASim 是一个相当简化的网络渗透测试模拟器。它的主要目标是在一个易于使用且快速的模拟器中捕捉网络渗透测试的一些关键特性，以便在更真实的环境中测试这些算法之前，可以用它来快速测试和原型设计算法。也就是说，NASim 中的 scenario与现实世界之间存在一些差距。

在本文档中，我们想列出一些在尝试将算法扩展到 NASim 之外时需要考虑的事项。这绝不是一个详尽的清单，但希望能为您下一步的思考提供一些参考，并解释 NASim 中做出的一些设计决策。

### 处理部分可观察性

NASim 做出的一大假设是，渗透测试Agent可以访问网络中每个主机的网络地址，即使在部分可观察模式下也是如此。此信息在其操作列表中提供给Agent。在现实世界中，根据 scenario的不同，此假设可能无效，而渗透测试人员面临的部分挑战是能够在网络中导航时发现新主机。

NASim 使用已知网络地址实现的主要原因是，这样可以固定动作空间大小，从而更容易与典型的深度强化学习算法（即具有固定大小输入和输出层的神经网络）一起使用。

研究挑战之一是开发能够处理随着渗透测试人员发现更多网络地址而变化的动作空间的算法，或者更现实的情况是，渗透测试人员的动作空间是多维的，包括分别选择地址和利用/扫描/等。实际上，NASim 中内置了一些对此的支持，即 nasim.envs.action.ParameterisedActionSpace 动作空间（参见[Actions](https://networkattacksimulator.readthedocs.io/en/latest/reference/envs/actions.html#actions)），但即使使用该动作空间，也会向渗透测试人员提供一些有关网络大小的信息。

目前还没有计划更新 NASim 以支持无信息动作空间。这部分是由于时间原因，但也是为了保持 NASim 的简单和稳定，因为现在有很多更好、更现实的环境正在开发中（例如[CybORG](https://github.com/cage-challenge/CybORG)）。

处理变化的动作空间的一种方法是使用自回归动作，就像[AlphaStar](https://www.deepmind.com/blog/alphastar-mastering-the-real-time-strategy-game-starcraft-ii)所做的那样。
