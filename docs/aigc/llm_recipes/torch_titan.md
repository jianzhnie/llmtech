# TorchTitan：用于生产级LLM预训练的一站式PyTorch原生解决方案

## 摘要

大型语言模型（LLMs）的发展在推动自然语言处理（NLP）应用的最新技术进步中发挥了重要作用。训练具有数十亿参数和数万亿token的LLMs需要复杂的分布式系统，这些系统能够组合和比较多种最先进的技术，以便在数千个加速器上高效扩展。然而，现有的解决方案复杂、分散在多个库/存储库中，缺乏互操作性，并且维护起来非常繁琐。因此，整理和实证比较训练方案需要大量的工程工作。

本文介绍了TorchTitan，一个开源的、PyTorch原生的分布式训练系统，它统一并推进了最先进的技术，简化了集成并减少了工程开销。TorchTitan以模块化和可组合的方式无缝应用3D并行，同时具有弹性扩展以适应不断变化的计算需求。该系统提供全面的日志记录、高效的检查点和调试工具，确保生产级训练。

此外，TorchTitan结合了创新的硬件-软件协同设计解决方案，利用Float8训练和SymmetricMemory等尖端功能，最大限度地提高硬件利用率。作为一个灵活的试验平台，TorchTitan促进了针对不同训练环境的定制方案的整理和比较。通过利用TorchTitan，我们为Llama 3.1系列开发了优化的训练方案，并根据我们的实践经验提供了选择和组合分布式训练技术以最大化训练效率的可操作指导。

我们在Llama 3.1系列的LLMs上全面评估了TorchTitan，涵盖80亿到4050亿参数，并展示了其卓越的性能、模块化组合性和弹性扩展性。通过堆叠训练优化，我们在128-GPU规模（Llama 3.1 8B）上展示了1D并行加速65.08%，在256-GPU规模（Llama 3.1 70B）上展示了2D并行加速12.59%，在512-GPU规模（Llama 3.1 405B）上展示了3D并行加速30%，基于NVIDIA H100 GPU的优化基线。

## 1 引言

LLMs处于NLP进步的前沿。大型语言模型（LLMs）（Devlin, 2018; Liu et al., 2019; Radford et al., 2019; Chowdhery et al., 2023; Anil et al., 2023; Achiam et al., 2023; Dubey et al., 2024; Jiang et al., 2024; Abdin et al., 2024）一直是推动自然语言处理（NLP）应用进步的动力，涵盖语言翻译、内容/代码生成、对话式AI、文本数据分析、创意写作和艺术、教育和研究等领域。

LLMs需要数十亿参数和数万亿token的训练才能达到最先进的性能。实现最先进的LLM性能需要大规模的计算资源，例如表现最佳的模型如Llama 3.1（405B参数，15T token，30.84M GPU小时，16K H100 GPU）（Dubey et al., 2024）和Google的PaLM（540B参数，0.8T token，9.4M TPU小时，6144 TPUv4芯片）（Chowdhery et al., 2023）。这些模型展示了卓越的自然语言理解和生成能力，但需要大量的计算资源、内存和时间来训练，突显了推动自然语言处理进步所需的巨大投资。

**LLM训练的挑战正在从各个方面得到解决。** 大规模训练大型语言模型（LLMs）是一项艰巨的任务，需要在并行性、计算和通信之间取得微妙的平衡，同时应对复杂的内存和计算权衡。训练所需的巨大资源使其容易受到GPU故障的影响，这突显了需要高效的恢复机制和检查点策略以最小化停机时间（Eisenman et al., 2022; Wang et al., 2023; Gupta et al., 2024; Maurya et al., 2024; Wan et al., 2024）。为了优化资源利用并实现弹性扩展，结合多种并行技术至关重要，包括数据并行（Li et al., 2020; Rajbhandari et al., 2020; Zhang et al., 2022; Zhao et al., 2023）、张量并行（Narayanan et al., 2021; Wang et al., 2022; Korthikanti et al., 2023）、上下文并行（Liu et al., 2023; Liu and Abbeel, 2024; NVIDIA, 2023; Fang and Zhao, 2024）和流水线并行（Huang et al., 2019; Narayanan et al., 2019, 2021; Qi et al., 2023）。通过将这些并行技术与内存和计算优化技术（如激活重计算（Chen et al., 2016; Korthikanti et al., 2023; He and Yu, 2023）、混合精度训练（Micikevicius et al., 2018, 2022）和深度学习编译器（Bradbury et al., 2018; Yu et al., 2023; Li et al., 2024; Ansel et al., 2024））结合，可以最大限度地提高硬件利用率。

**现有系统的局限性。** 虽然最先进的分布式训练技术显著推动了该领域的发展，但现有的系统在解决关键挑战方面仍然不足，这些挑战阻碍了研究人员和行业从业者的可用性、采用和有效性。

1. **不可组合性**：现有系统难以组合和堆叠各种并行技术，限制了多维并行的探索。进一步将它们与内存和计算优化集成具有挑战性，阻碍了训练效率。
2. **不灵活和单一架构**：当前系统不是模块化或可扩展的，难以集成和比较新技术、优化和硬件，限制了适应不断变化的机器学习环境的能力。
3. **硬件利用率低**：当前系统未能充分利用先进的硬件功能，导致GPU效率低下，并且缺乏可定制的激活检查点策略来应对内存-计算权衡。
4. **对生产级训练的支持不足**：现有系统缺乏可扩展和高效的分布式检查点，使得故障恢复和模型保存变得繁琐，并且通常不提供足够的调试工具和日志记录指标，导致难以识别和修复问题，特别是对于没有广泛专业知识的人员。
5. **现有系统未能充分利用PyTorch等框架的全部潜力**，错过了错误修复、优化内核、新功能和编译器支持。它们还依赖于外部依赖项，这些依赖项通常缺乏全面测试，并且可能由于维护不足而过时或不兼容。

**根本原因：缺乏表达性的张量抽象。** 分布式系统的不可组合性和不灵活性的根本原因在于缺乏使用表达性的张量和设备抽象作为核心组件，所有分布式并行、检查点和效率优化都可以在此基础上构建。

**设计原则：统一的分布式张量和设备抽象作为构建块。** 统一的设备抽象将分布式系统表示为多维数组，其中每个维度对应一种并行技术，管理设备之间的通信并处理集体进程组。互补的张量抽象使张量能够在这个数组上进行分片，保持分片规范并支持自动分片传播。这些抽象共同实现了并行技术的无缝组合，确保了正确的语义，并促进了分布式操作的集体调度。

我们通过使用PyTorch的分布式张量（DTensor）和DeviceMesh（Wanchao Liang, 2023）作为TorchTitan的基础组件，解决了统一张量抽象的技术挑战。通过我们与DTensor和DeviceMesh的合作，我们识别了关键限制并解决了这些问题。通过使用和扩展DTensor，我们开发了TorchTitan，这是一个生产级系统，能够在分布式训练中实现组合性、模块化、灵活性和可扩展性。TorchTitan促进了3D并行的组合、训练优化、可扩展的分布式检查点，并充分利用了PyTorch生态系统的优势。

为了开发和评估TorchTitan的能力，我们采取了几个关键步骤，这些步骤代表了本工作的核心贡献，总结如下：

1. 我们通过扩展其分片以支持n-D并行、添加与torch.compile的兼容性以实现编译器优化，并通过状态字典支持实现n-D模型的高效检查点，推进了DTensor的发展。我们还解决了关键错误，增强了DTensor的生产准备性。
2. 我们展示了如何组合和堆叠各种并行技术，促进了大型语言模型训练中多维并行的探索（§2.1）。
3. 我们实现了新颖的硬件-软件协同设计解决方案，利用先进的硬件功能提高GPU效率，提供可定制的激活检查点策略以应对内存-计算权衡，并利用torch.compile进一步优化内存、计算和通信（§2.2）。
4. 我们通过集成可扩展和高效的分布式检查点以促进快速故障恢复、集成Flight Recorder等调试工具以调试崩溃/卡住的任务，并提供广泛的日志记录指标，实现了生产级训练（§2.3）。
5. 我们在Llama 3.1系列模型（8B、70B和405B，分别使用1D、2D和3D并行）上广泛评估了TorchTitan，从8到512个GPU的规模，展示了弹性扩展性，同时确保了效率、收敛性和准确性。总结来说，我们在128-GPU规模（Llama 3.1 8B）上展示了1D并行加速65.08%，在256-GPU规模（Llama 3.1 70B）上展示了2D并行加速12.59%，在512-GPU规模（Llama 3.1 405B）上展示了3D并行加速30%，基于最新的NVIDIA H100 GPU的优化基线（§3.2）。
6. 我们提供了系统的训练方案和指南，帮助用户应对分布式训练的复杂性，帮助他们优化各种模型大小和集群配置的训练效率（§3.3）。
7. 我们展示了我们的模块化和可扩展架构如何无缝集成和比较新技术、优化和硬件，确保适应不断变化的机器学习环境（§4）。

通过提供一个易于访问和可扩展的平台，TorchTitan使大型语言模型（LLM）预训练民主化，使更广泛的研究人员和开发人员能够利用LLM的潜力，并加速该领域的创新。

## 2 通过组合性实现弹性

TorchTitan以模块化的方式结合了各种并行技术，使用户能够轻松选择多维并行的组合。这种组合性通过增强前沿探索的便利性，解决了难以扩展的挑战，从而优化大规模训练效率。

TorchTitan的代码库经过精心组织，以实现组合性和可扩展性。我们特意将三个主要组件分开并尽可能正交：（1）模型定义，它是并行无关的，设计为可读性强，（2）并行助手，它将数据并行、张量并行和流水线并行应用于特定模型，（3）通用训练循环。所有这些组件都可以通过TOML文件进行配置，并可以通过命令行覆盖，很容易在现有代码库的基础上添加新模型和并行技术。

### 可组合的N-D并行训练

在本节中，我们将介绍在大规模集群上扩展模型训练的整个流程，包括元设备初始化和核心的可组合多维并行，以展示这些技术如何组合在一起，在TorchTitan中高效地训练LLMs。TorchTitan中的实际代码片段可以在附录A中找到。

#### 2.1.1 使用元设备进行大规模模型初始化

鉴于LLMs模型大小的指数增长，第一个扩展问题甚至在训练开始之前就出现了。这是需要在集群上实例化一个大模型以进行分片，而不会溢出CPU或GPU内存。

为了解决这个问题，我们在TorchTitan中启用了元设备初始化模型，其中模型首先在“元”设备类型上初始化。元设备张量仅保存元数据信息，而不是实际数据，使初始化速度极快。之后，我们执行模型分片并将模型参数转换为分布式张量（DTensors），其中每个参数保存一个位于元设备上的本地分片。最后，我们根据用户定义的初始化函数执行参数初始化。我们利用分布式张量正确同步随机数生成器（RNG）种子，并根据其分片布局初始化参数。这确保了参数与在分片之前在一个设备上初始化整个模型时具有相同的值，从而便于不同并行配置之间的收敛比较。

#### 2.1.2 完全分片数据并行

原始的完全分片数据并行（FSDP）（Zhao et al., 2023）是ZeRO的有效实现，提供了在PyTorch中训练大模型的能力。然而，PyTorch中的原始实现（FSDP1）由于FlatParameter实现而存在各种限制（详见附录B.1）。

鉴于这些限制，TorchTitan集成了新版本的完全分片数据并行（FSDP2），它使用每个参数的分布式张量分片表示，从而提供了更好的与模型并行技术和其他需要操作单个参数的功能的组合性。

TorchTitan集成并利用FSDP2作为其默认的1D并行，受益于改进的内存管理（通常比FSDP1低7%的每GPU内存需求）和轻微的性能提升（平均比FSDP1快1.5%）。有关FSDP2和使用示例的更多详细信息，请参见附录B.1。TorchTitan通过嵌入适当的默认值（包括自动分片与您的世界大小）使运行FSDP2变得简单。

为了扩展到更大的世界大小，TorchTitan还集成了混合分片数据并行（HSDP），它通过创建分片组扩展了FSDP2。详细信息请参见附录B.2。

#### 2.1.3 张量并行

张量并行（TP）（Narayanan et al., 2021）与序列并行（SP）（Korthikanti et al., 2023）一起，是使大规模模型训练成为可能的关键模型并行技术。

![Refer to caption](https://arxiv.org/html/2410.06511v2/extracted/5976575/figures/titan-workflow.png)

图1：可组合和模块化的TorchTitan初始化工作流程。

TP在TorchTitan中使用PyTorch的RowwiseParallel和ColwiseParallel API实现，其中模型参数被分区为DTensors并执行分片计算。通过利用DTensor，TP实现不需要触及模型代码，这使得在不同模型上更快地启用，并提供了与本文中提到的其他功能的更好组合性。

**张量和序列并行（TP/SP）**：虽然TP分区了计算量最大的部分，但序列并行（SP）在序列维度上执行归一化或dropout层的分片计算，否则会生成大量复制的激活张量，从而对每个GPU的内存限制构成挑战。有关TP和FSDP + TP的更多详细信息、说明和用法，请参见附录B.3。

由于TP和SP之间的协同关系，TorchTitan原生将这两者捆绑在一起，并通过TP度设置共同控制。

**损失并行**：当计算损失函数时，模型输出通常非常大。由于TP/SP的模型输出在（通常很大的）词汇维度上分片，天真地计算交叉熵损失需要沿TP维度收集所有分片以使输出复制，这会导致大量内存使用。

通过损失并行，可以高效地计算交叉熵损失，而无需将所有模型输出分片收集到每个GPU。这不仅显著减少了内存消耗，还通过减少通信开销和并行执行分片计算提高了训练速度。鉴于这些改进，TorchTitan默认实现了损失并行。

#### 2.1.4 流水线并行

对于最大规模的预训练，TorchTitan提供了流水线并行，由于具有最轻的通信开销并利用P2P通信，流水线并行变得至关重要。

流水线并行（PP）将模型视为一系列操作，将操作（及其使用的参数）分块为S个阶段，这些阶段在单独的设备组上运行。在典型情况下，一个阶段代表单个模型层或一组N个相邻的模型层，但在理论上它甚至可以是一个部分层。对于前向传递，一个阶段接收输入激活（阶段0除外），执行本地计算，并发送输出激活（阶段S-1除外）。最后一个阶段执行损失计算，并开始反向传递，通过流水线反向发送梯度。为了提高效率，输入批次被分解为微批次，流水线调度将计算一个微批次与通信其他微批次重叠。TorchTitan支持多种流水线调度，其调度已在其他工作中描述（Narayanan et al., 2019; Huang et al., 2019; Narayanan et al., 2021; Qi et al., 2023）。

训练循环还必须考虑流水线阶段的创建，并执行流水线调度而不是直接调用model_forward()。由于调度按微批次计算损失，因此损失计算和任何日志记录代码必须为PP更新。在TorchTitan中，我们建议定义一个共享的loss_fn，用于流水线和非流水线代码路径，从而最小化训练循环中的分歧。

与数据并行的交互，例如确保数据并行减少仅在调度中的最后一个微批次之后发生，以及在使用zero-3时调度分片和取消分片操作，也在流水线调度执行器中透明地处理，简化了TorchTitan中的训练器实现。有关其在TorchTitan中的用法，请参见附录B.4。

### 优化训练效率

#### 2.2.1 使用激活检查点导航计算-内存权衡

激活检查点（AC）（Chen et al., 2016）和选择性激活检查点（SAC）（Korthikanti et al., 2023）是标准的训练技术，通过在反向传递期间重新计算激活来减少峰值GPU内存使用。即使在应用多维并行之后，通常也需要它。

TorchTitan提供了灵活的AC和SAC选项，利用torch.utils.checkpoint在TransformerBlock级别应用。AC策略包括“完整”AC、操作级SAC和层级SAC。

在TransformerBlock中，完整AC通过在反向传递期间重新计算所有需要的激活张量来工作，而操作级SAC保存计算密集型的PyTorch操作的结果，并仅重新计算其他操作。层级SAC的工作方式与完整AC类似，但包装应用于每x个TransformerBlock（其中x由用户指定），以实现内存和重新计算之间的可配置权衡。（详细信息见附录B.5。）

#### 2.2.2 区域编译以利用torch.compile优化

torch.compile在PyTorch 2中发布（Ansel et al., 2024），TorchDynamo作为前端将PyTorch操作提取到FX图中，TorchInductor作为后端将FX图编译为融合的Triton代码以提高性能。

在TorchTitan中，我们使用区域编译，将torch.compile应用于Transformer模型中的每个单独的TransformerBlock。这有两个主要好处：（1）我们为每个区域获得一个完整的图（没有图中断），与FSDP2和TP（以及更一般的torch.Tensor子类，如DTensor）和其他PyTorch分布式训练技术兼容；（2）由于Llama模型将相同的TransformerBlock层堆叠在一起，torch.compile可以识别重复编译的相同结构，并且只编译一次，从而大大减少编译时间。

torch.compile通过计算融合和计算-通信重新排序，以模型无关的方式和简单的用户界面带来了吞吐量和内存效率（见第3.2节）。下面我们进一步详细说明torch.compile的组合性如何帮助TorchTitan通过简单的用户界面解锁硬件优化的性能增益，结合异步TP和Float8等高级功能。

#### 2.2.3 异步张量并行以最大程度地重叠通信

默认情况下，TP在分片计算之前/之后会引发阻塞通信，导致计算资源无法有效利用。异步TP（AsyncTP）（Wang et al., 2022）通过将注意力模块和前馈模块中的TP矩阵乘法分解为较小的块，并在每个部分之间重叠通信集体，实现了计算-通信重叠。重叠是通过微流水线优化实现的，其中结果在计算其他块的同时进行通信。

PyTorch AsyncTP基于SymmetricMemory抽象，它创建节点内缓冲区以编写更快的通信集体（Wang et al., 2024）。这是通过在每个GPU上分配一个共享内存缓冲区来实现的，以提供直接的P2P访问。

通过TorchTitan集成torch.compile，可以轻松配置AsyncTP以实现有意义的端到端加速（详见第3.2节）在较新的硬件（H100或具有节点内NVSwitch的较新GPU）上。使用详情见附录B.6。

#### 2.2.4 通过混合精度训练和Float8支持提高吞吐量

混合精度训练（Micikevicius et al., 2018）在确保训练稳定性的同时提供了内存和计算节省。FSDP2内置支持混合精度训练，使用基本的torch.dtype。这涵盖了在低精度（例如torch.bfloat16）中执行FSDP全收集和计算的流行用法，并在高精度（例如torch.float32）中执行无损FSDP减少分散（梯度）以获得更好的数值结果。使用详情见附录B.7。

TorchTitan还支持在较新的硬件（如H100）上使用Float8（派生数据类型）进行更高级的混合精度训练，具有显著的性能提升（见第3.2节报告）。torchao.float8的Float8功能支持多种每张量缩放策略，包括动态、延迟和静态（见Micikevicius et al. (2022); Vasiliy Kuznetsov (2024), 第4.3节），同时与其他关键的PyTorch原生系统（如autograd、torch.compile、FSDP2和TP（具有Float8全收集能力（Feng et al., 2024）））组合。

### 生产级训练

为了实现生产级训练，TorchTitan提供了开箱即用的关键功能无缝集成。这些功能包括（1）使用PyTorch分布式检查点（DCP）进行高效检查点，（2）通过集成Flight Recorder调试卡住或崩溃的任务。

#### 2.3.1 可扩展和高效的分布式检查点

检查点在训练大型语言模型中至关重要，原因有两个：它们促进了模型在推理和评估等应用中的重用，并在发生故障时提供了恢复机制。最佳的检查点工作流程应确保在不同并行性之间轻松重用，并在不减慢训练速度的情况下保持高性能。有两种典型的检查点方法。第一种方法将状态（模型参数和优化器状态）聚合为一个不分片的版本，该版本与并行性无关，便于重用但需要昂贵的通信。第二种方法让每个训练器保存其本地分片状态，这加快了过程但由于嵌入的并行性信息而使重用变得复杂。

DCP使用DTensor解决了这些挑战，DTensor封装了全局和本地张量信息，独立于并行性。DCP将此信息转换为内部格式进行存储。在加载时，DCP将存储的分片与当前基于DTensor的模型参数和优化器状态进行匹配，从存储中获取必要的分片。TorchTitan利用所有原生PyTorch并行性，有效地使用DCP来平衡效率和可用性。此外，DCP通过异步检查点提高了效率，通过在单独的线程中处理存储持久性，使此操作与后续训练迭代重叠。TorchTitan利用DCP的异步检查点将Llama 3.1 8B模型的检查点开销减少了5-15倍（Zhang et al., 2024; Huang et al., 2024）。

#### 2.3.2 使用Flight Recorder调试任务崩溃

在开发并行代码或大规模运行时，常见的故障模式是观察到NCCL集体超时，然后需要找出根本原因。由于通信内核通常从CPU的角度是异步的，当某些东西超时时，很难确定哪个操作失败以及为什么失败。PyTorch提供了一个用于NCCL集体的Flight Recorder来帮助解决这个难题。它记录了每个集体或p2p操作的开始和结束时间（在GPU上）以及入队时间（在CPU上）。此外，它还记录了元数据，例如使用了哪个进程组、源排名（对于p2p，还有目的地）、张量大小和堆栈跟踪。

我们发现Flight Recorder中包含的数据有助于调试由并行代码中的错误引起的集体挂起和p2p挂起。对于PP，可能存在调度错误导致挂起，由于缺少或错误排序的发送或接收操作。基于Flight Recorder数据的分析可以确定在GPU上完成的最新发送或接收。对于FSDP或TP，可以确定一个或多个排名是否未调用集体，可能是由于PP调度中的错误或TP中的错误逻辑。

## 3 实验

在本节中，我们通过在Llama 3.1 8B、70B和405B上进行的实验，从1D并行到3D并行（分别），在8到512个GPU的规模上展示了使用TorchTitan进行弹性分布式训练的有效性。我们还分享了通过TorchTitan实验获得的知识和经验。有关我们如何应用（最多）3D并行的代码库的演练可以在附录A中找到。

### 实验设置

实验在具有95 GiB内存的NVIDIA H100 GPU1上进行，每个主机配备8个GPU和NVSwitch。两个主机形成一个机架，连接到TOR交换机。后端RDMA网络连接TOR交换机。在TorchTitan中，我们集成了一个可检查点的数据加载器，并为C4数据集（en变体）提供了内置支持，这是Common Crawl的网络爬取语料库的一个巨大、清理过的版本（Raffel et al., 2020）。我们在本节的所有实验中使用相同的数据集。对于分词器，我们使用与Llama 3.1一起发布的官方分词器（tiktoken）。

脚注1：用于实验的H100 GPU是非标准的。它们具有HBM2e，并且限制在较低的TDP。实际峰值TFLOPs应在SXM和NVL之间，我们不知道确切值。

### 性能

为了展示TorchTitan的弹性和可扩展性，我们在广泛的GPU规模（从8到512）上进行了实验，随着基础模型大小的增加（8B、70B和405B），并行维度数量不同（分别为1D、2D和3D）。为了展示第2.2节中引入的优化技术的有效性，我们展示了在适当基线上添加每个单独技术时训练吞吐量的提高。特别是，当在更高维度的并行性上训练新功能时，基线始终更新为包括所有先前的技术。

我们注意到，在整个实验过程中，内存读数在整个训练过程中保持稳定2，而吞吐量数字（每秒token，每GPU）每10次迭代计算并记录一次，并且始终在第90次迭代读取。我们不报告模型FLOPS利用率（MFU）（Chowdhery et al., 2023），因为当在TorchTitan中启用Float8时，BFLOAT16 Tensor Core和FPS Tensor Core都参与模型训练，但它们具有不同的峰值FLOPS，并且在这种情况下MFU的定义不明确。我们注意到，1D Llama 3.1 8B模型在8或128个H100 GPU上训练而不启用Float8时，实现了33%到42%的MFU。

脚注2：不同的PP排名可能具有不同的峰值内存使用量。我们取所有GPU的最大值。

表1：1D并行（FSDP）在Llama 3.1 8B模型上，8个GPU。混合精度训练。选择性激活检查点。本地批次大小2，全局批次大小16。

| 技术                     | 吞吐量（Tok/Sec） | 比较     | 内存（GiB） |
| ------------------------ | ----------------- | -------- | ----------- |
| FSDP                     | 6,258             | 100%     | 81.9        |
| + torch.compile          | 6,674             | + 6.64%  | 77.0        |
| + torch.compile + Float8 | 9,409             | + 50.35% | 76.8        |

表2：1D并行（FSDP）在Llama 3.1 8B模型上，128个GPU。混合精度训练。选择性激活检查点。本地批次大小2，全局批次大小256。

| 技术                     | 吞吐量（Tok/Sec） | 比较     | 内存（GiB） |
| ------------------------ | ----------------- | -------- | ----------- |
| FSDP                     | 5,645             | 100%     | 67.0        |
| + torch.compile          | 6,482             | + 14.82% | 62.1        |
| + torch.compile + Float8 | 9,319             | + 65.08% | 61.8        |

表3：2D并行（FSDP + TP）+ torch.compile + Float8在Llama 3.1 70B模型上，256个GPU。混合精度训练。完整激活检查点。FSDP度32，TP度8。本地批次大小16，全局批次大小512。

| 技术      | 吞吐量（Tok/Sec） | 比较     | 内存（GiB） |
| --------- | ----------------- | -------- | ----------- |
| 2D        | 897               | 100%     | 70.3        |
| + AsyncTP | 1,010             | + 12.59% | 67.7        |

### 使用TorchTitan 3D并行进行扩展

LLM扩展定律由于越来越大的模型大小和大量数据而带来了挑战，这需要在大量GPU上应用并行策略。TorchTitan提供了组合不同并行性的能力，以高效地将模型训练扩展到数千个GPU。本节讨论了在训练大规模LLM时应用TorchTitan 3D并行的观察和动机。请注意，可能有许多3D并行组合，但在本文中我们选择只讨论一种组合，可以总结为以下图表：

#### 3.3.1 使用FSDP进行扩展

FSDP（ZeRO）是一种通用技术，可以应用于任何模型架构，使其成为第一个或唯一的并行度的好选择。只要FSDP通信比相应的计算快（对于在多达数百个，例如512个GPU上训练的LLM来说是这样），并且不需要将（有效）每GPU批次大小减少到1以下（出于下面TP部分提到的原因），1D FSDP应该足够了。

现有的基于环的NCCL集体实现（全收集、减少分散）将产生延迟开销，这在大规模（例如512个GPU）时变得严重。单独的FSDP将由于集体延迟随着世界大小的增加而线性增加，导致FSDP集体无法被计算隐藏而变得效率低下。为了进一步扩展，需要考虑结合模型并行解决方案，如TP和PP。

#### 3.3.2 2D并行：将TP与FSDP结合

模型并行（TP和PP）可以帮助避免单独扩展FSDP时面临的增加的集体延迟。TP可以进一步降低有效的本地批次大小（当本地批次大小设置为1时，最小为TP度），因为TP分片模型在多个GPU上共同处理同一批次。这对于减少峰值内存使用至关重要，以便训练可以适应GPU内存（例如由于大模型大小或序列长度），或者对于具有固定所需全局批次大小的强扩展（例如由于训练效率考虑）。

此外，TP执行特征维度分片。这可以带来更优化的矩阵乘法形状，以实现更好的FLOP利用率。

由于TP引入了额外的阻塞集体，实际上TP仅在具有快速互连（NVLink）的节点内应用。AsyncTP可以通过完全重叠通信来提高性能，

表4：3D并行（FSDP + TP + PP）+ torch.compile + Float8 + AsyncTP在Llama 3.1 405B模型上，512个GPU。混合精度训练。完整激活检查点。FSDP度4，TP度8，PP度16。本地批次大小32，全局批次大小128。

| 调度     | 吞吐量（Tok/Sec） | 比较     | 内存（GiB） |
| -------- | ----------------- | -------- | ----------- |
| IF1B     | 100               | 100%     | 78.0        |
| 交错IF1B | 130               | + 30.00% | 80.3        |

![Refer to caption](https://arxiv.org/html/2410.06511v2/extracted/5976575/figures/scale_3d_parallel.png)

图2：使用3D并行进行扩展

#### 3.3.3 3D并行：将PP与2D并行结合

与其他模型并行相比，PP通过仅在阶段之间以P2P方式传输激活和梯度，需要较少的通信带宽。它特别有用（1）当FSDP世界大小再次变大时，FSDP+TP仍然暴露FSDP集体时，进一步减少FSDP通信延迟；或（2）在带宽有限的集群上进行训练。

我们注意到，PP的性能，特别是“气泡”大小，可能会因使用的流水线调度和微批次大小而异，假设固定的全局批次大小和世界大小。

## 4 展示适应性和可扩展性

在本节中，我们通过突出正在进行的工作和外部贡献，展示了TorchTitan的适应性和可扩展性，这些贡献展示了其无缝集成和比较新技术、优化和模型的能力。

### 正在进行的工作：4D并行和零气泡流水线调度

TorchTitan的模块化和可扩展架构使得无缝集成新技术和优化成为可能。例如，正在进行的工作包括集成上下文并行（Liu et al., 2023; Liu and Abbeel, 2024; NVIDIA, 2023）以实现4D并行，并利用torch.distributed_pipelining包支持零气泡调度（Qi et al., 2023）。这展示了TorchTitan适应不断变化的机器学习环境的能力。

### 外部贡献：构建和评估自定义创新

TorchTitan的灵活架构还使用户能够轻松集成和比较新创新。通过提供一个模块化和高效的试验平台，TorchTitan使用户能够快速基准测试新技术、优化和硬件对其训练性能的影响。这已经导致了一个新的生产级数据加载器的改进，一个新的ZeRO实现的改进，一个基于Adam的优化器的进步，以及一个顶级扩散模型的训练。

## 5 相关工作

随着LLM的快速增长（Dubey et al., 2024; Achiam et al., 2023），有大量的研究和行业关注于改进训练各种大小的LLM的基础设施。由于这些模型的本质是大规模的，分布式训练支持变得不可避免。像Megatron（Narayanan et al., 2021）、DeepSpeed（Rasley et al., 2020）和PyTorch分布式（Pytorch原生）（Paszke et al., 2019; Meta Platforms, Inc., ）这样的库提供了构建分布式训练工作流程的API。NVIDIA NeMo（NVIDIA Corporation, 2024）基于Megatron-LM，提供了一个处理复杂端到端模型生命周期的打包解决方案，从数据管理到模型部署。Pytorch原生解决方案如torchtune（Meta Platforms, Inc., ）专注于在简化的工作流程中微调LLM。TorchTitan与这些解决方案的不同之处在于，它专注于使用PyTorch原生API进行生产级预训练。该库设计具有弹性组合性，以适应预训练LLM所需的规模，同时最小化外部依赖。这降低了解释和扩展预训练的门槛，同时提供了异步分布式检查点等功能，以构建端到端生产工作流程。

## 6 结论

TorchTitan是一个强大而灵活的框架，用于训练LLM。它提供了组合性，允许用户结合各种并行技术（FSDP、TP和PP）、内存优化方法（Float8和激活检查点），以及与torch.compile的集成以优化训练效率。TorchTitan高度灵活，适应不断变化的模型架构和硬件进步，并具有模块化设计和多轴指标，促进创新和实验。TorchTitan还优先考虑可解释性、生产级训练和PyTorch原生能力。此外，它提供了具有弹性扩展性的高性能训练、全面的训练方案和指南，以及选择和组合分布式训练技术的专家指导。如实验部分所示，TorchTitan在128-GPU规模（Llama 3.1 8B）上提供了1D并行加速65.08%，在256-GPU规模（Llama 3.1 70B）上提供了2D并行加速12.59%，在512-GPU规模（Llama 3.1 405B）上提供了3D并行加速30%，基于优化基线。凭借其强大的功能和高效率，TorchTitan是挑战性LLM训练任务的理想一站式解决方案。



## 附录A 可组合的3D并行演练

我们已经讨论了使用TorchTitan 3D并行进行扩展以及应用不同并行性将训练扩展到数千个GPU的动机。在本节中，我们将演练TorchTitan中的3D并行代码。

第一步是在元设备上创建模型实例（例如Llama模型的**Transformer**）。然后，我们根据pipeline_parallel_split_points配置将模型拆分为多个PP阶段。请注意，对于循环调度的PP，我们可能会从PP拆分中获得多个model_parts，其中model_parts中的每个项目都是一个阶段模型块。接下来，我们为每个模型部分应用SPMD风格的分布式训练技术，包括TP、激活检查点、torch.compile、FSDP和混合精度训练，然后实际在GPU上初始化分片模型。

```
# 元初始化
with torch.device("meta"):
    model = model_cis.from_model_args(model_config)

# 应用PP
pp_schedule, model_parts = models_pipelining_fns(model_name)(
    model, pp_mesh, parallel_dims, job_config, device, model_config, loss_fn
)
for m in model_parts:
    # 应用SPMD风格的分布式训练技术
    models_parallelize_fns(model_name)(m, world_mesh, parallel_dims, job_config)
    # 将分片模型移动到GPU并通过DTensor初始化权重
    m.to_empty(device="cuda")
    m.init_weights()
```

为了将PP应用于模型，我们在高层运行以下代码。pipeline_llama_manual_split根据手动提供的pipeline_parallel_split_points配置将模型拆分为多个阶段，通过从完整模型（在元设备上）中删除未使用的模型组件。然后，build_pipeline_schedule使用torch.distributed.pipelining中的各种选项创建流水线调度，包括IF1B Narayanan et al. (2019)、GPipe Huang et al. (2019)、交错IF1B Narayanan et al. (2021)等，由pipeline_parallel_schedule配置指示。

```
stages, models = pipeline_llama_manual_split(
    model, pp_mesh, parallel_dims, job_config, device, model_config
)
pp_schedule = build_pipeline_schedule(job_config, stages, loss_fn)
return pp_schedule, models
```

TP和FSDP在SPMD风格的models_parallelize_fns函数中应用。为了应用TP，我们利用DTensor parallelize_module API，通过提供TP“计划”作为模型参数应如何分片的指令。在下面的示例中，我们展示了（不完整的）代码，用于分片重复的TransformerBlock。

```
for layer_id, transformer_block in model.layers.items():
    layer_tp_plan = {
        "attention_norm": SequenceParallel(),
        "attention": PrepareModuleInput(
            input_layouts=(Shard(1), None),
            desired_input_layouts=(Replicate(), None),
        ),
        "attention.wq": ColwiseParallel(),
        ...
    }
    parallelize_module(
        module=transformer_block,
        device_mesh=tp_mesh,
        parallelize_plan=layer_tp_plan,
    )
```

最后，我们通过包装每个单独的TransformerBlock然后包装整个模型来应用FSDP。请注意，PyTorch中的FSDP2实现支持混合精度训练。默认情况下，我们在参数全收集和激活计算中使用torch.hfloat16，在梯度减少分散通信和优化器更新中使用torch.hloat32。

```
mp_policy = MixedPrecisionPolicy(param_dtype, reduce_dtype)
fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
for layer_id, transformer_block in model.layers.items():
    # 作为优化，不为最后一个TransformerBlock重新分片，因为FSDP会立即预取它
    reshard_after_forward = int(layer_id) < len(model.layers) - 1
    fully_shared(
        transformer_block,
        **fsdp_config,
        reshard_after_forward=reshard_after_forward,
    )
fully_shared(model, **fsdp_config)
```

## 附录B 补充材料

### 完全分片数据并行

FSDP2改进了原始FSDP1的FlatParameter分组。具体来说，参数现在表示为在张量维度0上分片的DTensors。这提供了更好的与模型并行技术和其他需要操作单个参数的功能的组合性，允许分片状态字典由DTensor表示而无需任何通信，并通过DTensor提供了更简单的元设备初始化流程。例如，FSDP2解锁了更细粒度的张量级量化，特别是Float8张量量化，我们将在结果部分展示。

作为从FSDP1重写为FSDP2的一部分，FSDP2通过避免使用记录流实现了改进的内存管理系统。这实现了确定性的内存释放，因此相对于FSDP1，每GPU的内存需求更低。例如，在Llama 2 7B上，FSDP2记录了比FSDP1平均低7%的GPU内存。

此外，通过编写高效的内核来执行多张量全收集和减少分散，FSDP2显示出与FSDP1相当的性能，并且FSDP2有轻微的性能提升 - 使用Llama 2 7B，FSDP2显示出平均1.5%的吞吐量提升。

性能提升是采用两个小性能改进的结果。首先，仅为FP32减少分散运行一个除法内核（将本地FP32减少分散梯度除以世界大小，而不是两步预除和后除世界大小的平方根）。其次，在TorcftTitan中，FSDP2集成了默认情况下在正向传递期间不分片Transformer层中的最后一个块，因为它将在反向传递开始时立即重新收集。因此，我们可以跳过一轮通信延迟。

**用法**：TorcftTitan已完全集成FSDP2作为默认并行性，data_parallel_shard_degree是命令行或TOML文件中的控制维度。请注意，为了便于使用，将data_parallel_shard_degree保留为-1（默认值）意味着简单地使用所有可用的GPU（即无需指定实际的世界大小）。

### 混合分片数据并行

混合分片数据并行（HSDP）是FSDP的扩展（Zhang et al., 2022），它使能够使用更大的总世界大小。在FSDP中，所有设备都是单个全局组的一部分，所有通信都在该组中启用。然而，在某些时候，增加更多的计算会被由于增加更多需要平等通信参与的参与者而增加的通信开销所抵消。这是由于集体通信的延迟与参与者的总数直接相关。在这个饱和点，FSDP吞吐量将有效地趋于平稳，即使增加了更多的计算。HSDP通过创建较小的分片组（岛屿）在原始全局组（海洋）内，在一定程度上避免了这一点，其中每个分片组在其内部运行FSDP，并在反向传递期间以设定的频率同步分片组之间的梯度，以确保维护全局梯度。这确保了快速的通信，因为参与者的总通信大小现在是原始世界大小的一小部分，并且唯一的全局通信是分片组之间的梯度全减少。通过使用分片组，我们已经看到HSDP可以将总世界大小扩展到FSDP通信饱和点的3-6倍（这将根据网络互连的速度而变化）。

TorchTitan通过两个用户可配置的设置（分片组大小和复制组大小）使运行HSDP变得容易，可以从命令行或TOML文件中配置。

**用法**：在TorchTitan中通过修改前面提到的旋钮data_parallel_shard_degree来控制分片组大小来启用HSDP。这实际上是将在其相应组成员中运行FSDP分片的GPU组计数。从那里，我们必须指定data_parallel_replicate_degree，它控制我们创建多少个分片组。复制和分片度的乘积必须加起来等于总世界大小。示例 - 在128 GPU集群上，我们可能会发现分片16个GPU足以满足模型大小。因此，我们将data_parallel_shard_degree设置为16，并将data_parallel_replicate_degree相应地设置为8，这意味着我们将有8组16个GPU来填满128的总世界大小。

### 张量并行

TP将Transformer层的注意力和前馈网络（MLP）模块分区到多个设备上，其中使用的设备数量是TP度。这允许多个GPU协作处理一个Transformer层，否则该层将超出单个GPU的能力，代价是添加全减少/全收集/减少分散操作以同步中间结果。

由于TP引入了额外的集体，它需要在快速网络（即NVLink）上发生。在训练LLM时，TP通常与FSDP结合使用，其中TP在节点内分片，FSDP在节点间分片，以在不同DeviceMesh维度上创建2D分层分片。

**用法**：由于TP和SP之间的协同关系，TorchTitan原生将这两者捆绑在一起，并通过命令行或TOML条目中的tensor_parallel_degree设置共同控制。例如，将此设置为2意味着节点内的2个GPU将通过TP共享每个Transformer层的注意力和MLP模块的计算负载，并通过序列并行共享归一化/丢弃层。损失并行通过上下文管理器实现，因为它需要控制模型正向计算之外的损失计算。可以通过enable_loss_parallel启用。

### 流水线并行

我们暴露了几个参数来配置PP。pipeline_parallel_degree控制参与PP的排名数量。pipeline_parallel_split_points接受一个字符串列表，表示在执行拆分之前的层的完全限定名称。因此，流水线阶段的总数\(V\)将由该列表的长度决定。pipeline_parallel_schedule接受要使用的调度的名称。如果调度是多阶段的，则应为每个流水线排名分配\(V>1\)个阶段，否则\(V==1\)。pipeline_parallel_microbatches控制将数据批次拆分为多少个微批次。

### 激活检查点

TorchTitan提供了两种类型的选择性激活检查点，允许在内存和重新计算之间进行更细致的权衡。具体来说，我们提供了选择性地检查点“每层”或“每操作”的选项。每操作的目标是释放由更快重新计算的操作使用的内存，并保存较慢重新计算的操作的中间结果（内存），从而提供更有效的吞吐量/内存权衡。

**用法**：AC通过命令行或TOML文件中的两行设置启用。具体来说，mode可以是none、selective或full。当设置为selective时，则使用下一个配置selective_ac_type，它可以是一个正整数以启用选择性层检查点，或op以启用选择性操作检查点。每层接受一个整数输入以指导检查点策略，其中1 = 检查点每层（与full相同），2 = 检查点每隔一层，3 = 检查点每隔三层，等等。每操作由parallelize_liama.py中的_save_list策略驱动，该策略标记高算术强度操作，如matmul（矩阵乘法）和SPDA（缩放点积注意力）以保存中间结果，同时允许其他较低强度操作重新计算。请注意，为了平衡总吞吐量，仅每隔一个matmul被标记为保存。

### AsyncTP

AsyncTP中使用的SymmetricMemory集体比标准NCCL集体更快，并通过让每个GPU分配一个相同的内存缓冲区以提供直接的P2P访问来操作。SymmetricMemory依赖于节点内的NVSwitch，因此通常仅适用于H100或更新的GPU。

**用法**：AsyncTP在TorchTitan TOML配置文件的实验部分中启用，并通过enable_async_tensor_parallel布尔设置打开或关闭。

### 在TorchTitan中自定义FSDP2混合精度

混合精度由apply_fsdp函数中的MixedPrecisionPolicy类控制，然后在TorchTitan中默认使用param_dtype作为BF16，reduce_dtype默认为FP32。FP32中的reduce_dtype意味着反向传递中的减少分散用于梯度计算将在FP32中进行，以帮助最大化梯度更新的稳定性和精度。
