

# Verl



- *HybridFlow: A Flexible and Efficient RLHF Framework*

- *论文链接：https://team.doubao.com/zh/publication/hybridflow-a-flexible-and-efficient-rlhf-framework?view_from=research*

- *代码链接：https://github.com/volcengine/veRL*
- 在线文档：https://verl.readthedocs.io/en/latest/index.html



## 1. RL（Post-Training）复杂计算流程给 LLM 训练带来全新的挑战

在深度学习中，数据流（DataFlow）是一种重要的计算模式抽象，用于表示数据经过一系列复杂计算后实现特定功能。神经网络的计算就是典型的 DataFlow ，可以用计算图（Computational Graph）来描述，其中节点代表计算操作，边表示数据依赖。

大模型 RL 的计算流程比传统神经网络更为复杂。在 RLHF 中，需要同时训练多个模型，如 Actor 、Critic 、参考策略（Reference Policy）和奖励模型（Reward Model），并在它们之间传递大量数据。这些模型涉及不同的计算类型（前向反向传播、优化器更新、自回归生成等），可能采用不同的并行策略。

传统的分布式 RL 通常假设模型可在单个 GPU 上训练，或使用数据并行方式 [4,5]，将控制流和计算流合并在同一进程中。这在处理小规模模型时效果良好，但面对大模型，训练需要复杂的多维并行，涉及大量分布式计算，传统方法难以应对。

##  2. HybridFlow 解耦控制流和计算流，兼顾灵活高效

大模型 RL 本质上是一个二维的 DataFlow 问题：high-level 的控制流（描述 RL 算法的流程）+ low-level 的计算流（描述分布式神经网络计算）。

近期开源的 RLHF 框架，如 DeepSpeed-Chat [6]、OpenRLHF [7] 和 NeMo-Aligner [8]，采用了统一的多控制器（Multi-Controller）架构。各计算节点独立管理计算和通信，降低了控制调度的开销。然而，控制流和计算流高度耦合，当设计新的 RL 算法，组合相同的计算流和不同的控制流时，需要重写计算流代码，修改所有相关模型，增加了开发难度。

与此前框架不同，HybridFlow 采用了混合编程模型，控制流由单控制器（Single-Controller）管理，具有全局视图，实现新的控制流简单快捷，计算流由多控制器（Multi-Controller）负责，保证了计算的高效执行，并且可以在不同的控制流中复用。

尽管相比纯粹的多控制器架构，这可能带来一定的控制调度开销，但 HybridFlow 通过优化数据传输，降低了控制流与计算流之间的传输量，兼顾了灵活性和高效性。

## 3. 系统设计之一：Hybrid Programming Model (编程模型创新)

### 框架逻辑分析

![img](https://pic3.zhimg.com/v2-120db60c9af032bb5ddaab7ba831221e_1440w.jpg)



这是目前verl训练框架的配置情况，对于不同的训练角色，可以选择不同的**预训练模型**及**训练后端**的支持（多控制器)。随后将他们与`main_ppo.py`中对应的角色进行绑定，然后以参数的形式传入到`ray_trainer.py`中进行调用。对于trainer文件中实际是以一个单控制流函数`fit()`的方式来进行，只需要从对应的模型中获得计算值的情况，然后再导入到对应的算法模块与工具模块中，就可以快速的开展RL训练任务。

1.运行框架中浅黄色表示推理框架，深黄色表示训练框架

2.虚线模块表示在训练过程中并不一定被需要，可以结合训练算法进行删减

3.Reward模型有基于模型与基于规则的方法，并且目前训练推理框架可能都有支持，所以暂时写为optional。

4.Actor模块由于要进行rollout和training两个阶段，因此会在训练和推理框架之间进行切换(verl中在`sharding_manager`文件目录下)

### 3.1 封装单模型分布式计算

在 HybridFlow 中，每个模型（如 Actor、Critic、参考策略、奖励模型等）的分布式计算被封装为独立的模块，称为模型类。

这些模型类继承于基础的并行 Worker 类（如 3DParallelWorker 、FSDPWorker 等），通过抽象的 API 接口，封装了模型的前向、反向计算、优化器更新和自回归生成等操作。该封装方式提高了代码的复用性，便于模型的维护和扩展。

对于不同的 RL 控制流，用户可以直接复用封装好的模型类，同时自定义部分算法所需的数值计算，实现不同算法。当前 HybridFlow 可使用 Megatron-LM [13] 和 PyTorch FSDP [14] 作为训练后端，同时使用 vLLM [15] 作为自回归生成后端，支持用户使用其他框架的训练和推理脚本进行自定义扩展。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/IrH3BAPESuiahEibicAGHwOUHDjGSoSsicz8mibzJMoGH9c0bPYbbjOWBUiciaFBV3STgEM6HXIW3Vvmib3k6AOJnWt3uQ/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



### 3.2 灵活的模型部署

HybridFlow 提供了资源池（ResourcePool）概念，可以将一组 GPU 资源虚拟化，并为每个模型分配计算资源。不同的资源池实例可以对应不同设备集合，支持不同模型在同一组或不同组 GPU 上部署。这种灵活的模型部署方式，满足了不同算法、模型和硬件环境下的资源和性能需求。

![img](https://pica.zhimg.com/v2-677f0e776144e14a21c6fa7af61acde6_1440w.jpg)

首先构建的四类模型，会通过Ray，映射放置到不同的机器上。随后

（1）先使用vllm框架和prompt对Actor先进行response的输出（使用专用推理框架可以让推理的速度更快）。

（2）然后将输出的结果输入给其他三个框架进行运行（一般使用训练框架，因为训练框架可以避免精度问题），以获得在RL算法(例如PPO，GRPO等框架)所需要的计算输入。

（3）最后结合计算结果，再使用训练框架来对Actor和Critic模型进行训练。



### 3.3 统一模型间的数据切分

在大模型 RL 计算流程中，不同模型之间的数据传输涉及复杂的多对多广播和数据重分片。

为解决该问题，HybridFlow 设计了一套通用数据传输协议（Transfer Protocol），包括收集（collect）和分发（distribute）两个部分。

通过在模型类的操作上注册相应的传输协议，比如：@register(transfer_mode=3D_PROTO)，HybridFlow 可以在控制器层（Single-Controller）统一管理数据的收集和分发，实现模型间数据的自动重分片，支持不同并行度下的模型通信。

HybridFlow 框架已经支持多种数据传输协议，涵盖大部分数据重切分场景。同时，用户可灵活地自定义收集（collect）和分发（distribute）函数，将其扩展到更复杂的数据传输场景。



![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/IrH3BAPESuiahEibicAGHwOUHDjGSoSsicz8licpiakd4SicTbHtIgeia5U18HXyuJyic8elNC8iaQ5CTM43zcScfAzzq3pA/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



### 3.4 支持异步 RL 控制流

在 HybridFlow 中，控制流部分采用单控制器架构，可灵活实现异步 RL 控制流。

当模型部署在不同设备集合上时，不同模型计算可并行执行，这提高了系统的并行度和效率。对于部署在同一组设备上的模型，HybridFlow 通过调度机制实现了顺序执行，避免资源争夺和冲突。

### 3.5 少量代码灵活实现各种 RL 控制流算法


得益于混合编程模型的设计，HybridFlow 可以方便地实现各种 RLHF 算法，如 PPO [9]、ReMax [10]、Safe-RLHF [11]、GRPO [12] 等。用户只需调用模型类的 API 接口，按算法逻辑编写控制流代码，无需关心底层的分布式计算和数据传输细节。

例如，实现 PPO 算法只需少量代码，通过调用 actor.generate_sequences 、critic.compute_values 等函数即可完成。同时，用户只需要修改少量代码即可迁移到 Safe-RLHF 、ReMax 以及 GRPO 算法。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/IrH3BAPESuiahEibicAGHwOUHDjGSoSsicz8Y3fibjOtmOpOHUfCiaiaQicDiaPLBmWebhI2cB4BQTPc6D5KX08AhAawDvA/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)






## 4. 系统设计之二：3D-HybridEngine （训练推理混合技术）降低通信内存开销

在 Online RL 算法中，Actor 模型需要在训练和生成（Rollout）阶段之间频繁切换，且两个阶段可能采用不同并行策略。

具体而言，训练阶段，需要存储梯度和优化器状态，模型并行度（Model Parallel Size, MP）可能相应增高，而生成阶段，模型无需存储梯度和优化器状态，MP 和数据并行度（Data Parallel Size, DP）可能较小。因此，在两个阶段之间，模型参数需要重新分片和分配，依赖传统通信组构建方法会带来额外通信和内存开销。

此外，为了在新的并行度配置下使用模型参数，通常需要在所有 GPU 之间进行全聚合（All-Gather）操作，带来了巨大的通信开销，增加了过渡时间。

为解决这个问题，HybridFlow 设计了 3D-HybridEngine ，提升了训练和生成过程效率。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/IrH3BAPESuiahEibicAGHwOUHDjGSoSsicz89JadxTZdMK7U6TpPrLY53F7icJZ8wmjiapJsvoaFL5rpQkOUtmt1BvjA/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

*注：3D-HybridEngine 一次迭代的流程*

3D-HybridEngine 通过优化并行分组方法，实现了零冗余的模型参数重组，具体包括以下步骤：

- 定义不同的并行组

在训练和生成阶段，3D-HybridEngine 使用不同的三维并行配置，包括：流水线并行（PP）、张量并行（TP）和数据并行（DP）的大小。训练阶段的并行配置为 𝑝-𝑡-𝑑 。在生成阶段，我们新增一个新的微数据并行组（Micro DP Group，𝑑𝑔），用于处理 Actor 模型参数和数据的重组。生成阶段的并行配置为 𝑝𝑔-𝑡𝑔-𝑑𝑔-𝑑 。

- 重组模型参数过程

通过巧妙地重新定义生成阶段的并行分组，可以使每个 GPU 在生成阶段复用训练阶段已有的模型参数分片，避免在 GPU 内存中保存额外的模型参数，消除内存冗余。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/IrH3BAPESuiahEibicAGHwOUHDjGSoSsicz89NqZ1EfmvTr6SC22vgCTFPZibVqTvnXB65VhWorqMwt4CMyvEfxibZIw/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



- 减少通信开销

参数重组过程中，3D-HybridEngine 仅在每个微数据并行组（Micro DP Group）内进行 All-Gather 操作，而非所有 GPU 之间进行。这大大减少了通信量，降低过渡时间，提高了整体的训练效率。
