# 网络空间博弈

## 仿真环境

| 仿真环境                                                     | 作者                                                         | 简介                                                         | 本地部署(Linux) | 仿真程度           |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | --------------- | ------------------ |
| `CSLE`                                                       | 瑞典皇家理工学院                                             | 使用定量方法（例如最优控制、计算博弈论、强化学习、优化、进化方法和因果推理）开发自动化安全策略的研究平台。 | No              | 高，最接近真实环境 |
| [Yawning Titan](https://github.com/dstl/YAWNING-TITAN)       | 英国国防科学技术实验室 (DSTL)                                | Yawning-Titan 是一组抽象的、基于图形的网络安全模拟环境，支持基于 OpenAI Gym 的自主网络操作智能体的训练。Yawning-Titan 专注于提供快速模拟，以支持对抗概率红方智能体的防御性自主智能体的开发。 | yes             | 较简单             |
| [CyberBattleSim](https://github.com/microsoft/CyberBattleSim) | 微软                                                         | 网络拓扑和一组预定义的漏洞定义了进行模拟的环境。攻击者利用现有漏洞，通过横向移动在网络中进化，目标是通过利用计算机节点中植入的参数化漏洞来获取网络的所有权。防御者试图遏制攻击者并将其从网络中驱逐。 CyberBattleSim 为其模拟提供了 OpenAI Gym 接口，以促进强化学习算法的实验。 | yes             | 较高               |
| [CyBorg](https://github.com/cage-challenge/CybORG)           | 澳大利亚国防部                                               | 用于训练和开发安全人员和自主智能体的网络安全研究环境。包含用于模拟（使用基于云的虚拟机）和模拟网络环境的通用接口。 | yes             | 高                 |
| [NetworkAttackSimulator](https://github.com/Jjschwartz/NetworkAttackSimulator) | [Jonathon.schwartz@anu.edu.au](mailto:Jonathon.schwartz@anu.edu.au) | 用于针对模拟网络测试 AI 渗透测试智能体的环境。               | yes             | 简单               |

## 论文

- NASimEmu: Network Attack Simulator & Emulator for Training Agents Generalizing to Novel Scenarios
  - https://arxiv.org/pdf/2305.17246

- Incorporating Deception into CyberBattleSim for Autonomous Defense
  - https://arxiv.org/pdf/2108.13980
- A Multiagent CyberBattleSim for RL Cyber Operation Agents
  - https://arxiv.org/pdf/2304.11052
- Developing Optimal Causal Cyber-Defence Agents via Cyber Security Simulation
  - https://arxiv.org/pdf/2207.12355
- CybORG: A Gym for the Development of Autonomous Cyber Agents
  - https://arxiv.org/pdf/2108.09118
- ACD-G: Enhancing Autonomous Cyber Defense Agent Generalization Through Graph Embedded Network Representation
  - https://dspace.lib.cranfield.ac.uk/bitstream/handle/1826/18288/ACD-G-Enhancing_autonomous_cyber_defense-2022.pdf?sequence=1&isAllowed=y
- Research on active defense decision-making method for cloud boundary networks based on reinforcement learning of intelligent agent
  - https://www.sciencedirect.com/science/article/pii/S2667295223000430#fig3
- Network Environment Design for Autonomous Cyberdefense
  - https://arxiv.org/pdf/2103.07583
- Network Attack Simulation Model
  - https://ns3simulation.com/network-attack-simulation/

## 网络空间博弈节点数配置实验

### Test on CyberBattleSim

| Network       | Node size | Time (100 Steps)                                             |
| ------------- | --------- | ------------------------------------------------------------ |
| Chain Network | 10        | 21.501863956451416                                           |
| Chain Network | 100       | 23.260196924209595                                           |
| Chain Network | 1000      | 2290.061856031418                                            |
| Chain Network | 10000     | numpy.core._exceptions._ArrayMemoryError: Unable to allocate 2.91 TiB for an array with shape (10000, 10000, 8, 1000) and data type int32 |
