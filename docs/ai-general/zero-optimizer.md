# ZeRO 技术原理

## 前言

目前训练超大规模语言模型主要有两条技术路线：TPU + XLA + TensorFlow/JAX 和 GPU + PyTorch + Megatron-LM + DeepSpeed。前者由Google主导，由于TPU和自家云平台GCP深度绑定，对于非Googler来说， 只可远观而不可把玩，后者背后则有NVIDIA、Meta、MS大厂加持，社区氛围活跃，也更受到群众欢迎。

上面提到的DeepSpeed的核心是ZeRO(Zero Redundancy Optimizer)，简单来说，它是一种显存优化的数据并行(data parallelism, DP)方案。而“优化“这个话题又永无止境，在过去两年DeepSpeed团队发表了三篇ZeRO相关的论文，提出了去除冗余参数、引入CPU和内存、引入NVMe等方法，从始至终都围绕着一个目标：将显存优化进行到底。

# ZeRO: 一种去除冗余的数据并行方案

[ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://sc20.supercomputing.org/proceedings/tech_paper/tech_paper_pages/pap379.html) 发表在SC 20。

ZeRO是微软开发的一个可以高效利用显存的优化器，其会将模型状态量（优化器状态，梯度和模型参数）分布在多个并行 GPU 之上，目的是在不使用模型并行的情况下对让数十亿参数模型进行训练。

ZeRO是ZeRO-DP和ZeRO-R两种方法的组合。ZeRO-DP是一种增强数据并行机制，它使用动态通信策略来将优化器状态、梯度和参数进行分区，以最小化通信量和避免模型状态的冗余。ZeRO-R则使用分区激活重计算、恒定大小缓冲区和动态内存碎片整理机制来优化剩余状态的内存消耗。

## 背景知识

模型训练中的显存占用主要包括：模型参数、模型梯度、优化器状态、运算中间变量。以下图为例，训练过程中的显存占用包括一份模型参数以及对应的一份梯度，比较常用的 Adam 会保留两倍参数量的优化器参数，除此之外还有一些运算的中间变量。

<div align=center><img src="https://simg.baai.ac.cn/uploads/2022/06/f89403ec9d054ff8a4ede9a3118935c3.png" style="zoom:30%;" />
</div>

### Adam Optimizer Algorithms

Adam算法的关键组成部分之一是：它使用指数加权移动平均值来估算梯度的动量和二次矩.

<div align=center><img src="https://miro.medium.com/v2/resize:fit:1260/1*zfdW5zAyQxge85gA_mFPYg.png" style="zoom:50%;" />
</div>


- 优化器如果是SGD， 除了保存模型参数W之外还要保存对应的梯度 ∇，因此显存占用等于参数占用的显存x2,

- 如果是带Momentum-SGD，除了保存W 和对应的梯度 ∇， 这时候还需要保存动量， 因此显存x3

- 如果是Adam优化器，动量占用的显存更多，显存x4。

根据上述分析，对于一个模型参数约 20G 的大模型，训练过程中需要占用的显存就会超过 80G，在每一张显卡中都完整地维护这些内容，显存是远远不够的。这就需要采用相关分布式训练技术，进行模型训练的显存优化。

为解决这一关键问题，在 DeepSpeed 中，通过数据并行降低运算中间变量显存占比、增大吞吐量，通过 ZeRO 降低模型参数、模型梯度、优化器状态的显存占比，通过 Optimizer Offload 将优化器状态卸载到内存上，通过 Checkpointing 和 算子融合避免储存运算的中间变量，最后使用通信计算重叠 进一步降低整套系统时间花费。

综合使用这些技术，DeepSpeed 可以实现单张消费级显卡全参数微调 BERT-Large，8 台 A100 小集群训练 GPT-3，在超大规模模型训练场景下与 DeepSpeed 等框架相比最多可节省 90% 的算力成本。

分布式训练的核心是切割，将数据、参数等诸多要素切割到不同计算节点上进行运算。有切割就有合并，不同节点之间会频繁通信以同步及汇总计算结果。

这里简单介绍 5 个基本通信算子，这是分布式训练框架的重要基础（以四张显卡为例，由 rank0 到 rank3 表示）：

### Broadcast

张量位于某张显卡中，广播后，每张显卡都会获得一个同样的张量。

<div align=center><img src="https://simg.baai.ac.cn/hubview/38285fddfa08bf09aca5cdca4fce6208.png" style="zoom:50%;" />
</div>

### Reduce

每张显卡中存有一个张量，将这些张量进行如求和、取max等计算后，其结果被置于指定的某张显卡上。

<div align=center><img src="https://simg.baai.ac.cn/hubview/a9ebbe8794bd6e72e4a3cdcc080ef8ae.png" style="zoom:50%;" />
</div>
### All Reduce

每张显卡中存有一个张量，使用它们进行相关计算后的结果被置于所有的显卡上，各张显卡上得到的结果相同。

<div align=center><img src="https://simg.baai.ac.cn/uploads/2022/06/d7762b30d95cba8cbf75c3e53893eda6.png" style="zoom:50%;" />
</div>
### Reduce Scatter

每张显卡中存有一个大小为 4d 的张量，张量之间进行计算后的结果被平均切分为 4 份，每份的大小为 d，分别置于 4 张显卡上。

<div align=center><img src="https://simg.baai.ac.cn/uploads/2022/06/e593375a49982918c763f6b2b9d57a1a.png" style="zoom:50%;" />
</div>

### All Gather

每张显卡中存有一个大小为 d 的张量，收集后，张量拼接的结果 大小为 4d 被置于所有的显卡上，各张显卡上得到的结果相同。

<div align=center><img src="https://simg.baai.ac.cn/uploads/2022/06/bde5bfae0ca29e62bb5832da249a4079.png" style="zoom:50%;" />
</div>

## 分布式训练

一种典型的分布式训练方法是使用数据并行，然而对于大模型来说，仅通过数据并行进行显存优化是远远不够的，需要更进一步地进行切割。进一步优化的技术主要来自两大技术路线：在算子层面进行切割的 模型并行、流水线并行技术 以及在显存上进行切割的 ZeRO技术。在DeepSpeed中，采用了数据并行 和 ZeRO技术 来进行模型的分布式训练，并将陆续支持模型并行与流水线并行。

###  数据并行

数据并行通过减小每张显卡上需要处理的 batch 大小来减少模型的运行中间变量。具体来说，假设有n张显卡，那么每张显卡可以只去处理 batch_size / n 的数据，最后将各张显卡计算得到的梯度进行求和  all-reduce  即可。在这种方式中，每张显卡都会获得完整的梯度信息，最后每一张显卡上分别执行优化器的 step。

<div align=center><img src="https://simg.baai.ac.cn/uploads/2022/06/77b0ac81bc030f8ca24eaea12b5589dd.png" style="zoom:30%;" />
</div>

\- 采用数据并行策略，原模型训练需要的运算中间变量被划分到不同显卡中。图中以八卡并行为例，后面各图也采用相同的设定

###  模型并行

模型并行技术尝试将模型计算进行切割。以全连接层为例，通过将参数矩阵分解为n个小矩阵，每张显卡上计算 ， 然后通过 all-gather 通信即可获得完整的结果。在这种方法中，各张显卡均处理同一批次的数据，在计算时进行合作。

<div align=center><img src="https://simg.baai.ac.cn/uploads/2022/06/e4d822f3b0d75c7adb4c42ddb75c3c51.png" style="zoom:30%;" />
</div>

\- 采用模型并行策略，模型参数被划分到不同的显卡中

与模型并行类似的一种解决思路是流水线并行，也是尝试对训练计算进行切分。相比于模型并行中对 transformer 模型进行纵向的计算切分，流水线并行则将不同层的 transformer block 计算划分到不同的显卡上。

### 分析

其中数据并行由于简单易实现，应用最为广泛，当然这不表示它没有”缺点“，每张卡都存储一个模型，此时显存就成了模型规模的天花板。如果我们能减少模型训练过程中的显存占用，那不就可以训练更大的模型了？一个简单的观察是，如果有2张卡，那么系统中就存在2份模型参数，如果有4张卡，那么系统中就存在4份模型参数，如果有N张卡，系统中就存在N份模型参数，其中N-1份都是冗余的，我们有必要让每张卡都存一个完整的模型吗？系统中能否只有一个完整模型，每张卡都存 1/N 参数，卡数越多，每张卡的显存占用越少，这样越能训练更大规模的模型。

##   ZeRO

在实际训练中，优化器  如 Adam  状态占用的显存要比参数和梯度二者加起来还要多，因此 ZeRO（Zero Redundancy Optimizer，零冗余优化器）技术首次提出对优化器状态进行切分，每张显卡上只负责优化器状态对应的部分参数的更新。训练策略上，ZeRO 基于数据并行，不同的数据被划分到不同的显卡上进行计算。根据对优化器状态、梯度、参数划分程度的不同，ZeRO 技术包含 ZeRO-1/2/3 三个层次。

###   ZeRO-1

因为 ZeRO 基于数据并行，首先需要通过 all-gather 操作获取完整的模型参数更新结果，随后每张显卡根据自己的数据和模型参数完成对应的前向传播和反向传播。在整个过程中，梯度和参数均完整地保留在每张卡上，随后对梯度进行 reduce-scatter，每张卡根据自己所划分的优化器状态和梯度来计算对应部分的模型参数。

<div align=center><img src="https://simg.baai.ac.cn/uploads/2022/06/fdfe1cf85b8040fb93eef2abf7233984.png" style="zoom:30%;" />
</div>

- 基于 ZeRO-1 和数据并行，优化器状态和运算中间变量被划分到不同的显卡中

###   ZeRO-2

ZeRO-2 在 ZeRO-1 的基础上进一步对梯度进行划分。注意，由于在反传的过程中，不需要始终保留完整的梯度，在计算当前层梯度时，只需要后一层输入的梯度。因此在反传的过程中，对于不参与后续反传计算的梯度，可以立即 reduce-scatter 划分到多块卡上，这样在训练过程中，梯度在每块卡上的显存占用，就变为原先的1/n了。反传结束后，每块卡再根据部分的梯度和优化器状态，计算得到更新后的模型参数，最后再将更新后的参数使用 all-gather 同步到其他的显卡上。

<div align=center><img src="https://simg.baai.ac.cn/uploads/2022/06/aaeaffb333d01df6dfe23608d057eaf6.png" style="zoom:30%;" />
</div>

- 基于 ZeRO-2 和数据并行，梯度、优化器状态和运算中间变量被划分到不同的显卡中

###   ZeRO-3

而 ZeRO-3 技术，则是更进一步将模型参数部分进行切分。由于每张显卡只有一部分的优化器状态，只更新一部分的参数，一个很直观的思路就是每张显卡上只维护优化器需要更新的那一部分参数。然而，在模型的计算过程中，还是需要完整的模型参数。因而在 ZeRO-3 中，模型中的每个模块在计算之前，都需要通过一次 all-gather 操作将参数恢复完整，并在前向计算结束后再将模型参数释放掉。进行反传时，再重新使用 all-gather 获取参数计算梯度并使用 reduce-scatter 划分梯度，如下图。

<div align=center><img src="https://simg.baai.ac.cn/hubview/725b20407286600f241c001e6790cc07.png" style="zoom:30%;" />
</div>



通过使用 ZeRO-3 优化，训练相关的所有信息均被切碎分散到不同的显卡上，让每张显卡上的显存占用都被降低到极致，使得每张显卡上可以容下更大的 batch_size，更充分地利用计算核心，带来更大的模型吞吐，同时将训练模型所需的显卡数量降至最低。

<div align=center><img src="https://simg.baai.ac.cn/uploads/2022/06/227c3031693274bf87d59045e09d46c6.png" style="zoom:30%;" />
</div>

- 基于ZeRO-3和数据并行，参数、梯度、优化器状态和运算中间变量被划分到不同的显卡中

不过在 ZeRO 的原论文中指出， ZeRO-3 增加了额外的一次参数通信时间（即反向传播时的 all-gather ），因此会引入额外的通信开销，在部分场景下性能不及 ZeRO-2 和模型并行。为了减少额外通信量带来的效率损失，还额外引入了通信计算重叠的策略，这将在后面被介绍到。根据的实现，实验结果表明 ZeRO-3 在 NVLink+IB 的环境下训练超大规模模型较联合使用 ZeRO-2 和模型并行的方案会带来更大的计算吞吐量提升。

## 显存分析

[混合精度训练](https://arxiv.org/abs/1710.03740)（mixed precision training）和[Adam](https://arxiv.org/abs/1412.6980)优化器基本上已经是训练语言模型的标配，我们先来简单回顾下相关概念。

Adam在SGD基础上，为每个参数梯度增加了一阶动量（momentum）和二阶动量（variance）[1](https://basicv8vc.github.io/posts/zero/#fn:1)。

混合精度训练，字如其名，同时存在fp16和fp32两种格式的数值，其中模型参数、模型梯度都是fp16，此外还有fp32的模型参数，如果优化器是Adam，则还有fp32的momentum和variance。

https://basicv8vc.github.io/images/zero/mixed-precision-training.png

ZeRO将模型训练阶段，每张卡中显存内容分为两类：

1. 模型状态（model states）: 模型参数（fp16）、模型梯度（fp16）和 Adam 状态（fp32的模型参数备份，fp32的momentum和fp32的variance）。假设模型参数量 Φ ，则共需要 2Φ+2Φ+(4Φ+4Φ+4Φ)=4Φ+12Φ=16Φ 字节存储，可以看到，Adam状态占比 75% 75% 。
2. 剩余状态（residual states）: 除了模型状态之外的显存占用，包括激活值（activation）、各种临时缓冲区（buffer）以及无法使用的显存碎片（fragmentation）。

来看一个例子，GPT-2含有1.5B个参数，如果用fp16格式，只需要3GB显存，但是模型状态实际上需要耗费24GB！相比之下，激活值可以用 [activation checkpointing](https://arxiv.org/pdf/1604.06174.pdf) 来大大减少，所以模型就成了头号显存杀手，它也是ZeRO的重点优化对象。而其中Adam状态又是第一个要被优化的。

针对模型状态的存储优化（去除冗余），ZeRO使用的方法是分区（partition），即每张卡只存 1/N的模型状态量，这样系统内只维护一份模型状态。

- 首先进行分区操作的是模型状态中的Adam，也就是下图中的  Pos ，这里os指的是optimizer states。模型参数（parameters）和梯度（gradients）仍旧是每张卡保持一份，此时，每张卡的模型状态所需显存是 4Φ+12Φ /N字节，当  N 比较大时，趋向于 4Φ B，也就是原来 16ΦB 的 1/4  。
- 如果继续对模型梯度进行分区，也就是下图中的 Pos+g ，模型参数仍旧是每张卡保持一份，此时，每张卡的模型状态所需显存是 2Φ+（2Φ+12Φ)/N 字节，当 N 比较大时，趋向于  2ΦB ，也即是原来 16ΦB 的 1/8  。
- 如果继续对模型参数进行分区，也就是下图中的  Pos+g+p ，此时每张卡的模型状态所需显存是 16Φ/ N字节，当  N 比较大时，趋向于 0  。

下图中Memory Consumption 第二列给出了一个示例  K=12, Φ=7.5B,  N=64 ，可以看到显存优化相当明显。

在DeepSpeed中，  Pos 对应ZeRO-1，Pos+g 对应ZeRO-2，Pos+g+p 对应ZeRO-3，一般使用ZeRO-1就足够了。

https://basicv8vc.github.io/images/zero/DeepSpeed-Image-1.png

> ZeRO-DP优化的三个阶段之中每个设备内存消耗比较。ψ表示模型大小（参数数量），K表示优化器状态的内存乘数，Nd表示DP并行度，即Nd个GPU。在本例中，我们假设基于Adam优化器的混合精度训练，模型大小为ψ=7.5B，DP为Nd=64，K=12。

解决了模型状态，再来看剩余状态，也就是激活值（activation）、临时缓冲区（buffer）以及显存碎片（fragmentation）。

- 激活值同样使用分区方法，并且配合checkpointing
- 模型训练过程中经常会创建一些大小不等的临时缓冲区，比如对梯度进行AllReduce啥的，解决办法就是预先创建一个固定的缓冲区，训练过程中不再动态创建，如果要传输的数据较小，则多组数据bucket后再一次性传输，提高效率
- 显存出现碎片的一大原因是时候gradient checkpointing后，不断地创建和销毁那些不保存的激活值，解决方法是预先分配一块连续的显存，将常驻显存的模型状态和checkpointed activation存在里面，剩余显存用于动态创建和销毁discarded activation

## 通信数据量分析

下面我们就分析下通信数据量，先说结论， Pos 和  Pos+g 的通信量和传统数据并行相同， Pos+g+p 会增加通信量。

传统数据数据并行在每一步（step/iteration）计算梯度后，需要进行一次 AllReduce 操作来计算梯度均值，目前常用的是Ring AllReduce，分为ReduceScatter和AllGather两步，每张卡的通信数据量（发送+接受）近似为 2Φ。

我们直接分析  Pos+g ，每张卡只存储 1/ N 的优化器状态和梯度，对于 gpu0 来说，为了计算它这 1/ N梯度的均值，需要进行一次Reduce操作，通信数据量是 (1/N  Φ)⋅N=Φ ，然后其余显卡则不需要保存这部分梯度值了。实现中使用了bucket策略，保证 1/ N 梯度只发送一次。

```
这里还要注意一点，假如模型最后两层的梯度落在 gpu0，为了节省显存，其他卡将这两层梯度删除，怎么计算倒数第三层的梯度呢？还是因为用了bucket，其他卡可以将梯度发送和计算倒数第三层梯度同时进行，当二者都结束，就可以放心将后两层梯度删除了。
```

当 gpu0 计算好梯度均值后，就可以更新局部的优化器状态了，当反向传播过程结束，进行一次Gather操作，更新模型参数，通信数据量是 Φ 。

从全局来看，相当于用Reduce-Scatter和AllGather两步，和数据并行一致。

 Pos+g+p 使得每张卡只存了 1/N 的参数，不管是在前向计算还是反向传播，都涉及一次Broadcast操作, 通信量适度增加了50%。

## 显存优化

除了上述分布式训练方法外，Zero 还通过 Optimizer Offload 和 Checkpointing 技术进一步减少冗余的显存占用，并以牺牲最少的通信代价为前提，做到了在极致显存优化下仍然能高效率地训练。

### 优化残余状态内存

在使用 ZeRO-DP 优化模型状态对应的内存之后，残余内存（Residual State Memory）成为次要内存瓶颈，剩余内存包括：激活、临时缓冲区和不可用内存片段。我们开发了ZeRO-R来分别优化这三个因素所消耗的剩余内存。

- 对于激活（从前向传播结果之中存储，用来支持后向传播），我们注意到优化检查点会有帮助，但是对于大型模型不够用。因此，ZeRO-R通过在现有MP方案中识别和删除激活副本来优化激活内存。它还可以在适当的时候将激活卸载到CPU。
- ZeRO-R为临时缓冲区定义了适当的大小，以实现内存和计算效率的平衡。
- 我们观察到在训练中，由于不同张量生命周期的变化而会导致一些内存碎片。由于这些碎片的存在，会导致即便即使有足够的可用内存，也会因为缺少连续内存而使得内存分配失败。ZeRO-R根据张量的不同生命周期来主动管理内存，防止内存碎片。

ZeRO-DP和ZeRO-R结合在一起形成了一个强大的DL训练内存优化系统，我们统称为ZeRO。

###   Optimizer Offload

Optimizer Offload 是指将优化器状态从 GPU 卸载到 CPU 上，从而进一步节省显存。以 Adam 优化器为例介绍为什么需要将优化器的参数卸载。

在 Adam 中，优化器需要维护梯度的移动平均以及梯度平方的移动平均：

<div align=center><img src="https://simg.baai.ac.cn/uploads/2022/06/f1728ab5a07ddc884eed3b84b082d0a2.png" style="zoom:50%;" />
</div>


正如前文所示，与模型参数相比， Adam 优化器需要至少两份的显存占用量，这在混合精度训练中是一笔非常大的开销。通过使用 ZeRO-3 的梯度切分，每张计算卡上的需要处理的梯度信息大幅减少，将这一部分 GPU 计算卸载至 CPU 上产生的通信需求较小，同时 CPU 处理这样切分后的梯度也不会特别吃力。据此，付出了极小量的额外开销就将显存开销降低至原本的一半左右。

<div align=center><img src="https://simg.baai.ac.cn/hubview/913a0f839a9633e1d8c8a76a35656f5f.png" style="zoom:50%;" />
</div>


###   Checkpointing

Checkpointing 技术是一项很早就被提出，用于优化神经网络模型训练时计算图开销的方法。这种方法在 Transformers 等结构的模型训练中，能够起到非常明显的作用。目前主流的 Transformers 模型由大量的全连接层组成，以全连接层为例进行计算图的显存分析。

<div align=center><img src="https://simg.baai.ac.cn/hubview/8bf9da8761755e8a40f64a83e8458c27.png" style="zoom:30%;" />
</div>




为了能够在反向传播中计算梯度，需要在正向传播时记录下参数矩阵与输入，这两部分参数随着正向传播逐层累积，消耗了非常多的显存.

因此，使用 Checkpointing 技术（也称为亚线性内存优化），其核心方式是通过时间换空间，在模型各层之间设置检查点，只记录每一层模型的输入向量。在反向传播时，根据最近的 checkpoint 重新计算该层的局部计算图。

<div align=center><img src="https://simg.baai.ac.cn/hubview/bbd0786ccce55f536e49ab8382d0a2af.png" style="zoom:50%;" />
</div>

### 框架实现的优化

除了上述显存优化技术外，Zero  还在具体实现上进行优化，以期得到更好的加速效果。

###   混合精度

传统模型使用单精度参数进行训练，在大模型训练中，可以通过使用半精度参数来降低参数量并节省运算时间。具体实现上，Zero  在正向传播和反向传播的过程中均使用半精度进行计算，并在优化器中维护单精度的模型参数和优化器参数。

使用混合精度的另一个好处在于能够更好地利用显卡中的 tensor core。较新的显卡在 CUDA core 之外，还设置了专门用于张量运算的核心 tensor core，利用 tensor core 将为程序带来进一步的性能提升。使用混合精度训练能够更好地利用 tensor core 特性，从而为训练过程进一步加速。

###   算子融合

为了进一步提升性能，在 CPU 和 GPU 层面均进行了算子层面的实现优化。在 CPU 上，使用多线程 + SIMD（单指令流多数据流） 的 CPU 编程方式，对 Offload 至 CPU 计算的 Adam 优化器进行 CPU 上的计算加速，使其不会成为系统的性能瓶颈。在 GPU 上，使用算子融合的方式，将 Softmax 与 NLLLoss 算子合二为一，减小了中间结果的显存占用。

###   通信计算重叠

上文中提到，ZeRO3 技术将引入额外的通信时间，采用通信计算策略来进行通信时间的优化。以反向传播为例，由于使用了 ZeRO-3 技术，需要将切碎至各个计算卡上的模型进行临时的重组装（对应图中的 Gather ）；而在反向传播  对应图中的 Calculate  之后，还需要将得到的局部梯度重新切碎至不同的计算卡上（对应图中的 Scatter ）。通过不同的 CUDA stream 区分不同的操作，让运算和通信得以同时运行，通过大量的计算时间隐藏通信的时间开销。

<div align=center><img src="https://simg.baai.ac.cn/hubview/aef5c4fa65abd6791b1fcc71ae93b4cf.png" style="zoom:50%;" />
</div
# ZeRO-Offload: 让人人都能训练得起大模型

[ZeRO-Offload: Democratizing Billion-Scale Model Training](https://www.usenix.org/conference/atc21/presentation/ren-jie)发表在ATC 21，一作是来自UC Merced的[Jie Ren](https://jren73.github.io/)，博士期间的研究方向是 Memory Management on Heterogeneous Memory Systems for Machine Learning and HPC。

## 背景

ZeRO说到底是一种数据并行方案，可是很多人只有几张甚至一张卡.

一张卡训不了大模型，根因是显存不足，ZeRO-Offload的想法很简单：显存不足，内存来补。

直接看下效果，在单张V100的情况下，用PyTorch能训练1.4B的模型，吞吐量是30TFLOPS，有了ZeRO-Offload加持，可以训练10B的模型，并且吞吐量40TFLOPS。这么好的效果能不能扩展到多卡上面呢，比如只用一台[DGX-2](https://www.nvidia.com/en-us/data-center/dgx-2/)服务器，可以训练70B的模型，是原来只用模型并行的4.5倍，在128张显卡的实验上基本也是线性加速，此外还可以与模型并行配合.

<div align=center><img src="https://basicv8vc.github.io/images/zero/cpu-gpu.jpeg" style="zoom:50%;" />
</div

相比于昂贵的显存，内存廉价多了，能不能在模型训练过程中结合内存呢？其实已经有很多工作了，但是他们几乎只聚焦在内存上面，没有用到CPU计算，更没有考虑多卡的场景。ZeRO-Offload则将训练阶段的某些模型状态下放（offload）到内存以及CPU计算。

```
注：ZeRO-Offload没有涉及剩余状态（比如激活值）的下放，因为在Transformer LM场景中，他比模型状态占用的显存小。
```

ZeRO-Offload要做的事情我们清楚了，那么如何设计高效的offload策略呢？

## Offload策略

ZeRO-Offload并不希望为了最小化显存占用而让系统的计算效率下降，否则的话，我们只用CPU和内存不就得了。但是将部分GPU的计算和存储下放到CPU和内存，必然涉及CPU和GPU之间的通信增加，不能让通信成为瓶颈，此外GPU的计算效率相比于CPU也是数量级上的优势，也不能让CPU参与过多计算，避免成为系统瓶颈，只有前两条满足的前提下，再考虑最小化显存的占用。

为了找到最优的offload策略，作者将模型训练过程看作数据流图（data-flow graph）。

- 圆形节点表示模型状态，比如参数、梯度和优化器状态
- 矩形节点表示计算操作，比如前向计算、后向计算和参数更新
- 边表示数据流向

下图是某一层的一次迭代过程（iteration/step），使用了混合精读训练，前向计算（FWD）需要用到上一次的激活值（activation）和本层的参数（parameter），反向传播（BWD）也需要用到激活值和参数计算梯度，

<div align=center><img src="https://basicv8vc.github.io/images/zero/offload-1.png" style="zoom:80%;" />
</div

如果用Adam优化器进行参数更新（Param update），流程如下：

<div align=center><img src="https://basicv8vc.github.io/images/zero/offload-2.png" style="zoom:50%;" />
</div

下面我们为边添加权重，物理含义是数据量大小（单位是字节），假设模型参数量是  M ，在混合精度训练的前提下，边的权重要么是2M（fp16），要么是4M（fp32），

<div align=center><img src="https://basicv8vc.github.io/images/zero/offload-3.png" style="zoom:50%;" />
</div

我们现在要做的就是沿着边把数据流图切分为两部分，分布对应GPU和CPU，计算节点（矩形节点）落在哪个设备，哪个设备就执行计算，数据节点（圆形）落在哪个设备，哪个设备就负责存储，将被切分的边权重加起来，就是CPU和GPU的通信数据量。

ZeRO-Offload的切分思路是：

图中有四个计算类节点：FWD、BWD、Param update和float2half，前两个计算复杂度大致是 O(MB) ， O(B) 是batch size，后两个计算复杂度是 O(M) 。为了不降低计算效率，将前两个节点放在GPU，后两个节点不但计算量小还需要和Adam状态打交道，所以放在CPU上，Adam状态自然也放在内存中，为了简化数据图，将前两个节点融合成一个节点FWD-BWD Super Node，将后两个节点融合成一个节点Update Super Node。如下图右边所示，沿着gradient 16和parameter 16两条边切分。

<div align=center><img src="https://basicv8vc.github.io/images/zero/data-flow-partition.png" style="zoom:30%;" />
</div

现在的计算流程是，在GPU上面进行前向和后向计算，将梯度传给CPU，进行参数更新，再将更新后的参数传给GPU。为了提高效率，可以将计算和通信并行起来，GPU在反向传播阶段，可以待梯度值填满bucket后，一遍计算新的梯度一遍将bucket传输给CPU，当反向传播结束，CPU基本上已经有最新的梯度值了，同样的，CPU在参数更新时也同步将已经计算好的参数传给GPU，如下图所示。

<div align=center><img src="https://basicv8vc.github.io/images/zero/offload-training-one-cpu.png" style="zoom:30%;" />
</div

到目前为止，说的都是单卡场景

## 扩展性

在多卡场景，ZeRO-Offload利用了ZeRO-2，回忆下ZeRO-2是将Adam状态和梯度进行了分区，每张卡只保存 1/N，而ZeRO-Offload做的同样是将这 1/ N 的Adam状态和梯度都offload到内存，在CPU上进行参数更新。

```
注意：在多卡场景，利用CPU多核并行计算，每张卡至少对应一个CPU进程，由这个进程负责进行局部参数更新。
```

并且CPU和GPU的通信量和  N 无关，因为传输的是fp16 gradient和fp16 parameter，总的传输量是固定的，由于利用多核并行计算，每个CPU进程只负责 1/ N 的计算，反而随着卡数增加节省了CPU计算时间。

<div align=center><img src="https://basicv8vc.github.io/images/zero/offload-mul-gpu.png" style="zoom:30%;" />
</div

直接看下效果吧，

<div align=center><img src="https://basicv8vc.github.io/images/zero/offload-performance.png" style="zoom:30%;" />
</div

但是有一个问题，当batch size很小时，GPU上每个micro-batch计算很快，此时CPU计算时长会成为训练瓶颈，一种方法是让CPU在某个节点更新参数时延迟一步，`后面就可以让`GPU和CPU并行起来。

前N-1步，不进行延迟，避免早期训练不稳定，模型无法收敛，在第N步，CPU拿到GPU计算的梯度后，不更新参数，相当于GPU空算了一步，到N+1步，CPU开始根据刚才拿到的第N步的梯度计算，此时GPU开始算N+1步的梯度。

<div align=center><img src="https://basicv8vc.github.io/images/zero/offload-step.png" style="zoom:50%;" />
</div

当然这样会有一个问题，用来更新参数的梯度并不是根据当前模型状态计算得到的，论文的实验结果表明暂未发现对收敛和效果产生影响。

<div align=center><img src="https://basicv8vc.github.io/images/zero/offload-delayed.png" style="zoom:30%;" />
</div

# ZeRO-Infinity: 利用NVMe打破GPU显存墙

[ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning](https://arxiv.org/pdf/2104.07857.pdf) 发表在SC 21，同样是进行offload，ZeRO-Offload更侧重单卡场景，而ZeRO-Infinity则是典型的工业界风格，奔着极大规模训练去了。

## 背景

从GPT-1到GPT-3，两年时间内模型参数0.1B增加到175B，而同期，NVIDIA交出的成绩单是从V100的32GB显存增加A100的80GB，显然，显寸的提升速度远远赶不上模型模型增长的速度，这就是内存墙问题[3](https://basicv8vc.github.io/posts/zero/#fn:3)。

# 参考文献

1. Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He. ZeRO: Memory Optimizations Toward Training Trillion Parameter Models.
2. Zhengda Bian, Hongxin Liu, Boxiang Wang, et al. Colossal-AI: A Unified Deep Learning System For Large-Scale Parallel Training.
3. Adam Paszke, Sam Gross, Francisco Massa, et al. PyTorch: An Imperative Style, High-Performance Deep Learning Library.
4. Zhengyan Zhang, Xu Han, Hao Zhou, et al. CPM: A Large-scale Generative Chinese Pre-trained Language Model.
5. Zhengyan Zhang, Yuxian Gu, Xu Han, et al. CPM-2: Large-scale Cost-efficient Pre-trained Language Models.
6. Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
7. Colin Raffel, Noam Shazeer, Adam Roberts, et al. T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.
8. Alec Radford, Jeffrey Wu, Rewon Child, et al. GPT2: Language Models are Unsupervised Multitask Learners.
9. Ben Wang and Aran Komatsuzaki, et al. GPT-J from EleutherAI released in the repo mesh-transformer-jax.
10. Diederik P. Kingma, Jimmy Ba. Adam: A Method for Stochastic Optimization.
11. Yang You, Jing Li, Sashank Reddi, et al. Large Batch Optimization for Deep Learning: Training BERT in 76 minutes.
12. Hanlin Tang, Shaoduo Gan, Ammar Ahmad Awan, et al. 1-bit Adam: Communication Efficient Large-Scale Training with Adam's Convergence Speed.
13. NCCL: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html
