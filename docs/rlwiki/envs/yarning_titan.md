## Yawning-Titan 是什么？

Yawning-Titan 是一组抽象的、基于图形的网络安全模拟环境，支持基于 OpenAI Gym 的自主网络操作智能代理的训练。Yawning-Titan 专注于提供快速模拟，以支持对抗概率红方代理的防御性自主代理的开发。

Yawning-Titan 包含少量特定的、独立的 OpenAI Gym 环境，用于自主网络防御研究，非常适合学习和调试，以及灵活、高度可配置的通用环境，可用于表示一系列复杂性和规模不断增加的场景。通用环境只需要一个网络拓扑和一个设置文件，即可创建符合 OpenAI Gym 的环境，该环境还可以实现开放研究和增强的可重复性。

## Yawning-Titan 如何使用？

Yawning-Titan 可通过 CLI 应用程序或 GUI 使用。这样做的目的是让所有用户都能尽可能方便地使用 Yawning-Titan，同时又不影响用户对源代码进行深入修改的能力。

### 设计原则

- Yawning-Titan 的设计遵循了以下关键原则：

  简单胜过复杂最低硬件要求操作系统无关支持多种算法增强代理/策略评估支持灵活的环境和游戏规则配置

###  是用什么构建的

Yawning-Titan 建立在巨人的肩膀上，并且严重依赖于以下库：

> - [OpenAI’s Gym](https://gym.openai.com/) 是所有环境的基础
> - [Networkx](https://github.com/networkx/networkx)  作为所有环境使用的底层数据结构
> - [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3)  用作 RL 算法的来源
> - [Rllib (part of Ray)](https://github.com/ray-project/ray) 被用作 RL 算法的另一个来源
> - [Typer](https://github.com/tiangolo/typer) 用于提供命令行界面
> - [Django](https://github.com/django/django/) 用于提供 GUI 的管理和元素
> - [Cytoscape JS](https://github.com/cytoscape/cytoscape.js/) 用于提供轻量级且直观的网络编辑器

### Yawning-Titan 使用

#### 导入

首先，导入网络和节点。

```
from yawning_titan.networks.node import Node
from yawning_titan.networks.network import Network
```

#### 网络

要创建一个网络，首先我们必须实例化一个的实例[`Network`](https://dstl.github.io/YAWNING-TITAN/source/_autosummary/yawning_titan.networks.network.Network.html#yawning_titan.networks.network.Network)。

虽然[`Network`](https://dstl.github.io/YAWNING-TITAN/source/_autosummary/yawning_titan.networks.network.Network.html#yawning_titan.networks.network.Network)可以通过调用直接实例化`Network()`，但是您可以设置一些可配置参数（我们将在下文中讨论这些参数）。

网络 = 网络()

#### 节点实例

接下来我们实例化一些[`Node`](https://dstl.github.io/YAWNING-TITAN/source/_autosummary/yawning_titan.networks.node.Node.html#yawning_titan.networks.node.Node)。

再次，虽然[`Node`](https://dstl.github.io/YAWNING-TITAN/source/_autosummary/yawning_titan.networks.node.Node.html#yawning_titan.networks.node.Node)可以通过调用直接实例化`Node()`，但您可以设置一些可配置参数（我们将在下文中讨论这些参数）。

```
node_1 = Node()
node_2 = Node()
node_3 = Node()
node_4 = Node()
node_5 = Node()
node_6 = Node()
```

#### 将节点添加到网络

目前我们只有一个实例[`Network`](https://dstl.github.io/YAWNING-TITAN/source/_autosummary/yawning_titan.networks.network.Network.html#yawning_titan.networks.network.Network)和一些实例 [`Node`](https://dstl.github.io/YAWNING-TITAN/source/_autosummary/yawning_titan.networks.node.Node.html#yawning_titan.networks.node.Node)。

要将 [`Node`](https://dstl.github.io/YAWNING-TITAN/source/_autosummary/yawning_titan.networks.node.Node.html#yawning_titan.networks.node.Node)  添加到  [`Network`](https://dstl.github.io/YAWNING-TITAN/source/_autosummary/yawning_titan.networks.network.Network.html#yawning_titan.networks.network.Network)，我们需要调用`.add_node()`。

```
network.add_node(node_1)
network.add_node(node_2)
network.add_node(node_3)
network.add_node(node_4)
network.add_node(node_5)
network.add_node(node_6)
```

#### 在节点间添加边

通过调用`.add_edge() `添加边。

```python
network.add_edge(node_1, node_2)
network.add_edge(node_1, node_3)
network.add_edge(node_1, node_4)
network.add_edge(node_2, node_5)
network.add_edge(node_2, node_6)
```

就这样，基础[`Network`](https://dstl.github.io/YAWNING-TITAN/source/_autosummary/yawning_titan.networks.network.Network.html#yawning_titan.networks.network.Network)已经创建完毕。

#### 设置入口节点

入口节点可以在以下位置手动设置：

```python
node_1.entry_node = True
```

或者通过配置来[`Network`](https://dstl.github.io/YAWNING-TITAN/source/_autosummary/yawning_titan.networks.network.Network.html#yawning_titan.networks.network.Network)随机设置它们：

```python
from yawning_titan.networks.network import RandomEntryNodePreference
network.set_random_entry_nodes = True
network.num_of_random_entry_nodes = 1
network.random_entry_node_preference = RandomEntryNodePreference.EDGE
network.reset_random_entry_nodes()
```

#### 设置 EntHigh 值节点

可以在以下位置手动设置高价值节点：

```python
node_1.high_value_node = True
```

或者通过配置来随机设置它们：

```python
from yawning_titan.networks.network import RandomHighValueNodePreference

network.set_random_high_value_nodes = True
network.num_of_random_high_value_nodes = 1
network.random_high_value_node_preference = RandomHighValueNodePreference.FURTHEST_AWAY_FROM_ENTRY
network.reset_random_high_value_nodes()
```

#### 设置节点漏洞

可以在以下位置手动设置节点漏洞[`Node`](https://dstl.github.io/YAWNING-TITAN/source/_autosummary/yawning_titan.networks.node.Node.html#yawning_titan.networks.node.Node)：

```python
node_1.vulnerability = 0.5
```

或者通过配置来[`Network`](https://dstl.github.io/YAWNING-TITAN/source/_autosummary/yawning_titan.networks.network.Network.html#yawning_titan.networks.network.Network)随机设置它们：

```python
network.set_random_vulnerabilities = True
network.reset_random_vulnerabilities()
```

#### 重置网络

要一次性重置所有入口节点、高价值节点和漏洞：

```python
network.reset()
```

#### 查看网络节点详情

要查看网络节点详情：

```python
network.show(verbose=True)
```

这将产生如下输出：

```python
UUID                                  Name    High Value Node    Entry Node      Vulnerability  Position (x,y)
------------------------------------  ------  -----------------  ------------  ---------------  ----------------
bf308d9f-8382-4c15-99be-51f84f75f9ed          False              False               0.0296121  0.34, -0.23
1d757e6e-b637-4f63-8988-36e25e51cd55          False              False               0.711901   -0.34, 0.23
8f76d75c-5afd-4b2c-98ed-9c9dc6181299          True               False               0.65281    0.50, -0.88
38819aa3-0c05-4863-8b9d-c704f254e065          False              False               0.723192   1.00, -0.13
cc06f5e0-c956-449a-b397-b0e7bed3b8d4          False              True                0.85681    -0.49, 0.88
665b150b-fbd3-42a7-b899-3770ef2b285a          False              False               0.48435    -1.00, 0.13
```

## 示例网络

在这里，我们将创建在 Yawning-Titan 测试 ( tests.conftest.corporate_network )中用作固定装置的企业网络。

当每个节点在网络图中显示时，都会添加名称。

```python
# Instantiate the Network
network = Network(
    set_random_entry_nodes=True,
    num_of_random_entry_nodes=3,
    set_random_high_value_nodes=True,
    num_of_random_high_value_nodes=2,
    set_random_vulnerabilities=True,
)

# Instantiate the Node's and add them to the Network
router_1 = Node("Router 1")
network.add_node(router_1)

switch_1 = Node("Switch 1")
network.add_node(switch_1)

switch_2 = Node("Switch 2")
network.add_node(switch_2)

pc_1 = Node("PC 1")
network.add_node(pc_1)

pc_2 = Node("PC 2")
network.add_node(pc_2)

pc_3 = Node("PC 3")
network.add_node(pc_3)

pc_4 = Node("PC 4")
network.add_node(pc_4)

pc_5 = Node("PC 5")
network.add_node(pc_5)

pc_6 = Node("PC 6")
network.add_node(pc_6)

server_1 = Node("Server 1")
network.add_node(server_1)

server_2 = Node("Server 2")
network.add_node(server_2)

# Add the edges between Node's
network.add_edge(router_1, switch_1)
network.add_edge(switch_1, server_1)
network.add_edge(switch_1, pc_1)
network.add_edge(switch_1, pc_2)
network.add_edge(switch_1, pc_3)
network.add_edge(router_1, switch_2)
network.add_edge(switch_2, server_2)
network.add_edge(switch_2, pc_4)
network.add_edge(switch_2, pc_5)
network.add_edge(switch_2, pc_6)

# Reset the entry nodes, high value nodes, and vulnerability scores by calling .setup()
network.reset()

# View the Networks Node Details
network.show(verbose=True)
```

给出：

```
UUID                                  Name      High Value Node    Entry Node      Vulnerability  Position (x,y)
------------------------------------  --------  -----------------  ------------  ---------------  ----------------
c883596b-1d86-44f5-b4de-331292d8e3d5  Router 1  False              False               0.320496   0.00, -0.00
b2bd683b-a773-40de-85e8-36c21e66613d  Switch 1  False              False               0.889044   0.01, 0.61
68d9689b-5365-4022-b3bd-92bdc5a1627b  Switch 2  True               False               0.0671795  -0.00, -0.62
3554ed26-9480-487b-9d3c-57975654a2af  PC 1      False              False               0.400729   -0.38, 0.69
89700b3f-8be2-4b70-a21e-a0772551a6bc  PC 2      True               False               0.0807914  0.18, 1.00
82e91c52-5458-493a-a7cd-00fb702d6af1  PC 3      False              True                0.86676    0.39, 0.70
91edf896-f004-4ca7-9587-cc8417c4a26b  PC 4      False              False               0.967413   -0.39, -0.69
ebbc79f7-9a52-4a08-8b56-fee816284b54  PC 5      False              True                0.684436   0.38, -0.69
2cdaaf06-9b4a-41e9-ba6f-129aec634080  PC 6      False              False               0.727421   -0.19, -1.00
b81ad769-688a-4d02-ae7b-a64f0984b101  Server 1  False              False               0.630726   -0.17, 0.99
52cbd8ec-b063-40c5-a73e-a51291347e8f  Server 2  False              True                0.789554   0.17, -1.00
```
