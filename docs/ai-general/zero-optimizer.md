# ZeRO 技术原理

模型训练中的显存占用主要包括：模型参数、模型梯度、优化器状态、运算中间变量。以下图为例，训练过程中的显存占用包括一份模型参数以及对应的一份梯度，比较常用的 Adam 会保留两倍参数量的优化器参数，除此之外还有一些运算的中间变量。

<div align=center><img src="https://simg.baai.ac.cn/uploads/2022/06/f89403ec9d054ff8a4ede9a3118935c3.png" style="zoom:30%;" />
</div>

### Adam Optimizer Algorithms

Adam算法的关键组成部分之一是：它使用指数加权移动平均值来估算梯度的动量和二次矩. 

<div align=center><img src="https://miro.medium.com/v2/resize:fit:1260/1*zfdW5zAyQxge85gA_mFPYg.png" style="zoom:50%;" />
</div>



优化器如果是SGD， 除了保存模型参数W之外还要保存对应的梯度 ∇，因此显存占用等于参数占用的显存x2,

如果是带Momentum-SGD，除了保存W 和对应的梯度 ∇， 这时候还需要保存动量， 因此显存x3

如果是Adam优化器，动量占用的显存更多，显存x4。

根据上述分析，对于一个模型参数约 20G 的大模型，训练过程中需要占用的显存就会超过 80G，在每一张显卡中都完整地维护这些内容，显存是远远不够的。这就需要采用相关分布式训练技术，进行模型训练的显存优化。

为解决这一关键问题，在 DeepSpeed 中，通过数据并行降低运算中间变量显存占比、增大吞吐量，通过 ZeRO 降低模型参数、模型梯度、优化器状态的显存占比，通过 Optimizer Offload 将优化器状态卸载到内存上，通过 Checkpointing 和 算子融合避免储存运算的中间变量，最后使用通信计算重叠 进一步降低整套系统时间花费。

综合使用这些技术，DeepSpeed 可以实现单张消费级显卡全参数微调 BERT-Large，8 台 A100 小集群训练 GPT-3，在超大规模模型训练场景下与 DeepSpeed 等框架相比最多可节省 90% 的算力成本。

## 背景知识

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

## 显存优化

除了上述分布式训练方法外，Zero 还通过 Optimizer Offload 和 Checkpointing 技术进一步减少冗余的显存占用，并以牺牲最少的通信代价为前提，做到了在极致显存优化下仍然能高效率地训练。

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
</div> 

## 参考文献

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

