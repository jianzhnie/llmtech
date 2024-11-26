# Optimizer state sharding (ZeRO)

本次大规模训练技术系列分享之 ZeRO，主要对微软 ZeRO Optimizer 的思路和实现进行介绍，全文包含以下四个部分：

- 大规模训练的技术挑战 & 现有的并行训练方式
- ZeRO Optimizer 的三个不同级别
- ZeRO-3 具体实现思路和方式
- ZeRO 的局限与大模型训练的未来

## 1 训练大模型的挑战

随着人工智能技术在全球的推广应用，自动驾驶、人脸识别、自然语言处理等越来越多领域通过深度学习大大提升了算法的整体性能和表现，GPU 也成为了训练模型不可或缺的基础计算设备。 然而，随着模型规模的不断增大，加之模型训练的数据量也越来越大，单个 GPU 的计算能力完全无法满足大规模网络的训练需求。 在密集型训练的代表——自然语言处理中，OpenAI 在 2020 年 6 月发布的第三代语言模型 GPT-3 的参数量达到了 1700 亿，相比于之前 GPT-2 的最大版本 15 亿个参数增长了百倍以上。 2021 年 4 月 25 日，华为云也发布盘古系列超大预训练模型，其中包含30亿参数的全球最大视觉（CV）预训练模型，以及与循环智能、鹏城实验室联合开发的千亿参数、40TB 训练数据的全球最大中文语言（NLP）预训练模型。 这些庞大的模型训练背后，必然少不了一套精妙运转的训练系统的支持，本次分享将揭秘超大模型训练系统中必不可少的一项技术——**ZeRO**。

## 2 现有并行方法

在探索 ZeRO 之前，我们需要先了解一下当前分布式训练主要的三种并行模式： 数据并行、模型并行和流水线并行。

### 2.1 数据并行

当模型规模足够小且单个 GPU 能够承载得下时，数据并行就是一种有效的分布式训练方式。因为每个 GPU 都会复制一份模型的参数，我们只需要把训练数据均分给多个不同的 GPU，然后让每个 GPU 作为一个计算节点独立的完成前向和反向传播运算。 数据并行不仅通信量较小，而且可以很方便的做通信计算重叠，因此可以取得最好的加速比。

### 2.2 模型并行

如果模型的规模比较大，单个 GPU 的内存承载不下时，我们可以将模型网络结构进行拆分，将模型的单层分解成若干份，把每一份分配到不同的 GPU 中，从而在训练时实现模型并行。 训练过程中，正向和反向传播计算出的数据通过使用 All gather 或者 All reduce 的方法完成整合。这样的特性使得模型并行成为处理模型中大 layer 的理想方案之一。然而，深度神经网络层与层之间的依赖，使得通信成本和模型并行通信群组中的计算节点 (GPU) 数量正相关。其他条件不变的情况下，模型规模的增加能够提供更好的计算通信比。

### 2.3 流水线并行

流水线并行，可以理解为层与层之间的重叠计算，也可以理解为按照模型的结构和深度，将不同的 layer 分配给指定 GPU 进行计算。相较于数据并行需要 GPU 之间的全局通信，流水线并行只需其之间点对点地通讯传递部分 activations，这样的特性可以使流水并行对通讯带宽的需求降到更低。 然而，流水并行需要相对稳定的通讯频率来确保效率，这导致在应用时需要手动进行网络分段，并插入繁琐的通信原语。同时，流水线并行的并行效率也依赖各卡负载的手动调优。这些操作都对应用该技术的研究员提出了更高的要求。

<img src="https://picx.zhimg.com/v2-3906f12b21c6d637e6934cc37ca8791e_720w.jpg?source=d16d100b" alt="img" style="zoom:80%;" />

> 流水线并行

## 3 为什么需要ZeRO？

在三种并行方式中，数据并行因其易用性，得到了最为广泛的应用。然而，数据并行会产生大量冗余 Model States 的空间占用。ZeRO 的本质，是在数据并行的基础上，对冗余空间占用进行深度优化。

在**[大规模训练系列之技术挑战]()**一文中，我们介绍了大规模训练中的显存占用可以分为 **Model States** 与 **Activation** 两部分，而 **ZeRO** 就是为了解决 **Model States** 而诞生的一项技术。

首先，我们来聊一下模型在训练过程中 **Model States** 是由什么组成的：

1. Optimizer States: **`Optimizer States`** 是 **Optimizer** 在进行梯度更新时所需要用到的数据，例如 SGD 中的`Momentum`以及使用混合精度训练时的`Float32 Master Parameters`。
2. Gradient： 在反向传播后所产生的梯度信息，其决定了参数的更新方向。
3. Model Parameter: 模型参数，也就是我们在整个过程中通过数据“学习”的信息。

在传统数据并行下，每个进程都使用同样参数来进行训练。每个进程也会持有对`Optimizer States`的完整拷贝，同样占用了大量显存。在混合精度场景下，以参数量为`Ψ`的模型和 Adam optimzier为例，Adam需要保存：

- Float16的`参数`和`梯度`的备份。这两项分别消耗了2Ψ和2Ψ Bytes内存；（1 Float16 = 2 Bytes）
- Float32的`参数`，`Momentum`，`Variance` 备份，对应到 3 份`4Ψ`的内存占用。（1 Float32 = 4 Bytes）

最终需要`2Ψ + 2Ψ + KΨ = 16Ψ bytes`的显存。一个7.5B参数量的模型，就需要至少 120 GB 的显存空间才能装下这些`Model States`。当数据并行时，这些重复的`Model States`会在N个GPU上复制N份[1]。

ZeRO 则在数据并行的基础上，引入了对冗余Model States的优化。使用 ZeRO 后，各个进程之后只保存完整状态的`1/GPUs`，互不重叠，不再存在冗余。在本文中，我们就以这个 7.5B 参数量的模型为例，量化各个级别的 ZeRO 对于内存的优化表现。

### **3.1 ZeRO 的三个级别**

> 相比传统数据并行的简单复制，ZeRO 通过将模型的`参数`，`梯度`和`Optimizer State`划分到不同进程来消除冗余的内存占用。

ZeRO 有三个不同级别，分别对应对 **Model States** 不同程度的分割 (Paritition)：

- ZeRO-1：分割`Optimizer States`；
- ZeRO-2：分割`Optimizer States`与`Gradients`；
- ZeRO-3：分割`Optimizer States`、`Gradients`与`Parameters`；

<img src="https://pic1.zhimg.com/v2-8c87dd82df3b817be6342a15091660f6_720w.jpg?source=d16d100b" alt="img" style="zoom:50%;" />

> Three stages of ZeRO-DP optimizations.[1]  Ψ denotes model size (number of parameters)K denotes the memory multiplier of optimizer states $N_{d}$ denotes DP degree.

#### 3.1.1 ZeRO-1

 Optimizer States Partitioning ($P_{os}$):  4x memory reduction, same communication volume as DP

Optimizer 在进行梯度更新时，会使用`参数`与`Optimizer States`计算新的`参数`。而在正向或反向传播中，`Optimizer States`并不会参与其中的计算。 因此，我们完全可以让每个进程只持有**一小段**`Optimizer States`，利用这**一小段**`Optimizer States`更新完与之对应的**一小段**`参数`后，再把各个小段拼起来合为完整的模型参数。**ZeRO-1** 中正是这么做的：

<video class="ztext-gif GifPlayer-gif2mp4 css-1xeqk96" src="https://vdn6.vzuu.com/SD/bd03b0bc-ef95-11eb-8ee1-ce96bf022449.mp4?pkey=AAV1MswBBjlOwY4f7VJJvz9D1xHSs_hJXCdc6pMSxOyO6lICRf4xOpYk4LRoRYkXjl6fB1MEb8DFMrJRqeb39r4o&amp;bu=078babd7&amp;c=avc.0.0&amp;expiration=1732428776&amp;f=mp4&amp;pu=078babd7&amp;v=ks6" data-thumbnail="https://picx.zhimg.com/v2-d9ab0b59b95bdd9154cb5ba174f66843_720w.jpg?source=d16d100b" poster="https://picx.zhimg.com/v2-d9ab0b59b95bdd9154cb5ba174f66843_720w.jpg?source=d16d100b" data-size="normal" preload="metadata" loop="" playsinline=""></video>

> ZeRO Optimizer Stage1 Animation [4]

假设我们有 $N_d$ 个并行的进程，**ZeRO-1** 会将完整优化器的状态等分成 $N_d $ 份并储存在各个进程中。当`Backward`完成之后，每个进程的`Optimizer`:

- 对自己储存的`Optimizer States（包括Momentum、Variance 与 FP32 Master Parameters）`进行计算与更新。
- 更新过后的`Partitioned FP32 Master Parameters`会通过`All-gather`传回到各个进程中。

经过这两步，完成一次完整的参数更新。

通过 ZeRO-1 对`Optimizer States`的分段化储存，7.5B 参数量的模型内存占用将由原始数据并行下的 **120GB 缩减到 31.4GB**。

#### 3.1.2 ZeRO-2

> Optimizer States and Gradient Partitioning ($P_{os+g}$):   8x memory reduction, same communication volume as DP

ZeRO-1将`Optimizer States`分**小段**储存在了多个进程中，所以在计算时，这**一小段**的`Optimizer States`也只需要得到进程所需的对应**一小段**`Gradient`就可以。遵循这种原理，和`Optimizer States`一样，ZeRO-2也将`Gradient`进行了**切片**：

在一个Layer的`Gradient`都被计算出来后：

- `Gradient`通过`AllReduce`进行聚合。 （类似于DDP）
- 聚合后的梯度只会被某一个进程用来更新参数，因此其它进程上的这段`Gradient`不再被需要，可以立马释放掉。（按需保留）

这样就在**ZeRO-1**的基础上实现了对`Gradient`的切分。

通过 ZeRO-2 对`Gradient`和`Optimizer States`的分段化储存，7.5B 参数量的模型内存占用将由 ZeRO-1 中 **31.4GB 进一步下降到 16.6GB**。

#### 3.1.3 ZeRO-3

> Optimizer States, Gradient and  Parameter Partitioning ($P_{os+g+p}$):  Memory reduction is linear with DP degree

当`Optimizer States`，`Gradient`都被分布式切割分段储存和更新之后，剩下的就是`Model Parameter`了。 ZeRO-3 通过对`Optimizer States`，`Gradient`和`Model Parameter`三方面的分割，从而使**所有进程共同协作，只储存一份完整 Model States**。其核心思路就是**精细化通讯**，按照计算需求做到参数的收集和释放。

### 3.2 ZeRO-3 宏观概览

ZeRO-3 相对于 ZeRO-1 和 ZeRO-2，实现方式会复杂很多。首先我们站在**宏观的角度**，理解ZeRO-3 的算法原理：

#### 3.2.1 初始化

一个模型由多个`Submodule`组成。在初始化时，ZeRO-3 会将**每个**`Submodule Parameter Tensor`下的数据按照 GPU 的数量，**分摊切割**成多个小`ds_tensor`储存在在不同 GPU 进程中。因为`ds_tensor`可以共同组合出完整数据，所以原始`param`下的数据变为冗余信息，会被释放掉。

<img src="https://picx.zhimg.com/v2-d9bb46410486449ea7656a8d02954a85_720w.jpg?source=d16d100b" alt="img" style="zoom:50%;" />

> ZeRO-3 初始化参数Partition

#### 3.2.2 训练中

在训练过程中，ZeRO-3 会按照`Submodule`的计算需求进行参数的收集和释放： 在当前`Submodule`正向/反向传播**计算前**，ZeRO-3 通过`All-gather`拿到分摊储存在不同进程中的`ds_tensor`，重建原始的`param`。重建之后的参数就可以参与计算。

在当前`Submodule`正向/反向传播计算后，`param`下的数据并没有发生变更，与 ds_tensor 相同，造成了冗余。因此，`param`会再次被释放。

<img src="https://picx.zhimg.com/v2-ff23996ae5b74a98b163b89683dc4809_720w.jpg?source=d16d100b" alt="img" style="zoom:50%;" />

> ZeRO-3 训练中参数收集释放

经过 ZeRO-3, 一套完整的 model states 就被分布式储存在了多个 GPU 进程中。通过按照计算需求的数据收集和释放，实现储存空间有限的情况下超大规模模型的训练。7.5B 参数量，64 卡并行的模型，内存占用将由 ZeRO-2 的 **16.6GB 最终下降到 1.9GB**。相较于传统数据并行下 120GB 的内存空间，ZeRO-3 显著提升了内存占用效率[1]。

以上就是 ZeRO-3 的宏观算法原理的概述。在下边的几个章节中，我们将深入源码，解读ZeRO-3 代码的实现方式和逻辑。

<video class="ztext-gif GifPlayer-gif2mp4 css-1xeqk96" src="https://vdn3.vzuu.com/SD/2a0e318c-ef96-11eb-9cd5-6ad0d31fb0b0.mp4?auth_key=1732425176-0-0-03a1e2d046eaf26f7bafd32f92b5e9f6&amp;bu=078babd7&amp;c=avc.0.0&amp;disable_local_cache=1&amp;expiration=1732425176&amp;f=mp4&amp;pu=078babd7&amp;v=tx" data-thumbnail="https://picx.zhimg.com/v2-765afe80f3364e1a94f13c0b83f6739f_720w.jpg?source=d16d100b" poster="https://picx.zhimg.com/v2-765afe80f3364e1a94f13c0b83f6739f_720w.jpg?source=d16d100b" data-size="normal" preload="metadata" loop="" playsinline=""></video>

> ZeRO Optimizer Stage3 Animation [4]

### **3.3 ZeRO-3 在 DeepSpeed 中的具体实现思路和方式**

在这里，我们深入代码，探索一下 ZeRO-3 是如何实现`Model Parameter`分布式存储的。

**初始化: 分割 & 收集机制 -> submodule 收集 -> submodule 释放**

#### 3.3.1 初始化 - 模型参数的分割

> 参数的分割遵循着每个进程雨露均沾的原则。

首先，为了防止内存爆炸，巨大的`Model Parameters`必须在加载之前就被拆分并发放到各个进程中。**ZeRO-3** 在模型初始化时就通过`class Init`对其进行了分摊与切割。

```
python model = zero.Init(module=model)
```

- `zero.Init`初始化过程对传入的`module`做了如下的四步：

  - 判定传入 ZeRO-3 的`module`非`None`
  - 在一个`for loop`中，遍历其下`submodule`中的所有参数

  - 在 tensor 的 data 分割改变之前，对每一个`parameter tensor`套一层`_convert_to_deepspeed_param`的马甲用于记录tensor的特性（shape, numel, etc），防止后期因为 padding 和 partition 导致原始数据特性的丢失

  - 参数完成`conver_to_deepspeed_param`之后，`param.partition()`对其进行均分切割并分摊给各个进程。

`param.partition()`中会按照如下步骤进行参数切分：

- 根据进程数量(`self.world_size`)来计算 parameter partition 之后的 size：

```python
partition_size = tensor_size // self.world_size
```

- 创建一个 partition_size 大小的空白 tensor：

```python
partitioned_tensor = torch.zeros(partition_size, dtype=param.dtype, device=self.remote_device)
```

- 计算 partition 需要截取和储存的数据区间：

```python
start = partition_size * self.rank
end = start + partition_size
```

- 把原始 param 拉成一维后，按照进程自己的 rank 来决定偏移量的`start`和`end`，计算出截取的区间并放进`partitioned_tensor`里，把这个新创建的 tensor 挂在原始的`param.ds_tensor`下:

```python
one_dim_param = param.contiguous().view(-1)
src_tensor = one_dim_param.narrow(0, start, partition_size)
param.ds_tensor.copy_(src_tensor)
```

- 把原始的`param.data`减少到1个scalar tensor:

```python
# 因为param.data已经被分散储存在param.ds_tensor下，
# 所以这一部分会将param.data释放掉，修改为只储存一个scalar的形式参数。
# 这也是为什么要通过_convert_to_deepspeed_param的马甲记录下原始信息的原因。
param.data = torch.ones(1).half().to(param.device)
```

通过以上五个步骤，每个 module 中的参数就被拆分并储存到了不同的进程中，当这一步结束时，原始在`param.data`长度变为了 1，分段后的参数则放在`param.ds_tensor`中。

假设有 $ N_d$  个 GPUs, 某一个`model parameter`的数据量（numel）为 T, 则其会被`para.partition()`成  $ N_d$ 个小数据块分发到$ N_d$ 个进程中，每个进程中保持  $ \cfrac{T}{N_d} $ 一小段原始数据。在需要重建完整 tensor 进行计算时，**ZeRO-3** 通过之前记录下的原始`shape`, `numel`等特性对参数进行完整的重构。

<img src="https://pica.zhimg.com/v2-26372b945f601208636cb3e82cb801b8_1440w.jpg" alt="img" style="zoom:50%;" />

> Parameter Partition

#### 3.3.2 初始化 - 模型参数收集初始化

>  根据每个 submodule 需求做到更精细化的参数收集与释放。

拆分好了`model parameter`之后，下一步需要考虑的就是如何在需要时快速的找到这些分摊储存的参数，并且重新组合成完整的参数进行运算。 参数的收集与释放虽然发生在每次的 forward 与 backward 中，但是需要在初始化就建立好控制信息，针对这个目的，ZeRO-3 中创建了另外两个 class： - `class PartitionedParameterCoordinator`
 \- `class PrefetchCoordinator`

这两个 class 用于负责在`forward`和`backward`时协调`module parameters`的获取和释放。

为了能够在模型forward和backward中及时拿到模型参数，`ZeRO`初始化过程的一个重要环节就是给每个`submodule`创建 hooks。

首先我们来一起了解一下 PyTorch 中的 hook。 根据 PyTorch 的文档的介绍：

> "You can register a function on a Module or Tensor. The hook can be a forward hook or a backward hook. The forward hook will be executed when a forward call is executed. The backward hook will be executed in the backward phase. "

通过使用`hook`，我们可以在保留网络输入输出结构的同时，方便地获取、改变网络中间层变量的值和梯度。`ZeRO-3 Optimizer`初始化的过程中，代码通过递归的方式，对`module`下的每个`submodule`都挂上了四个 hook：

- `_pre_forward_module_hook`，在`submodule`的**forward开始前**负责`module parameters`获取；
- `_post_forward_module_hook`，在`submodule`的**forward结束后**负责`module parameters`释放；
- `_pre_backward_module_hook`，在`submodule`的**backward开始前**负责`module parameters`获取；
- `_post_backward_module_hook`，在`submodule`的**backward结束后**负责`module parameters`释放；

在每个`submodule`的`forward`和`backward`计算前，hook会调用：

- `class PartitionedParameterCoordinator` 中的`fetch_sub_module`和`all_gather`收集重建自己需要的`parameter`。
-  `class PrefetchCoordinator`中的`prefetch_next_sub_modules`则最大化利用通讯带宽，提前`all_gather`收集到未来`submodule`需要的`parameter`，为之后的计算做好准备。

计算完成后，hook 则通过： - `class PartitionedParameterCoordinator` 中的`release_sub_module`再次释放当前`submodule`的`parameters`。

通过这样的方式，在每一个`iteration`中，各个`submodule`就可以对自己需要的参数做出**计算前的获取**和**计算后的释放**。

<img src="https://picx.zhimg.com/v2-e011b327bbf7bc6767f76e45bc96e867_1440w.jpg" alt="img" style="zoom:30%;" />

> Forward and Backward Hooks

#### 3.3.3 前向传播中的 ZeRO-3

- 前向传播中 Model Parameter 的获取（Pre-Forward Hook）

<img src="https://picx.zhimg.com/v2-d1821245bc660556a4f9afd810257809_1440w.jpg" alt="img" style="zoom:33%;" />

> Pre-Forward Hook

在初始化时，ZeRO-3 Optimizer 把全部`module parameter`分散`partition`到了不同的 GPU 上。因此，在每个`submodule`做`forward`之前，需要:

- 明确`submodule`所需要的`parameter`
- 通过进程间通讯拿到分散储存的`partitioned parameter`
- 重新构造出原始`parameter`进行运算

而整个流程都是通过`PartitionedParameterCoordinator`和`PrefetchCoordinator`实现的。 每个submodule在`Pre-forward hook`中进行了四步操作：

- 1. `param_coordinator.record_trace` 在第一个`iteration`时，`record_trace`会通过`param_coordinator`记录下一份`model`的完整运行记录`trace`，也就是各**nn.module**的执行顺序。在之后的`iteration`，运行记录已经创建好了，`record_trace`就不再发挥作用。

- 2. `param_coordinator.fetch_sub_module` 因为`module forward`会逐层进行，当获得`submodule`的信息后：

  -  通过`submodule.named_parameters()`收集当前需要的全部`partitioned parameters`。
  - 通过`all_gather`，各个进程中的`partitioned parameters`会被重新组合构建成原始`parameter`。     - 利用原始`parameter`进行`submodule.forward`的计算。

- 3. `param_coordinator.prefetch_next_sub_modules` 为了节省通讯时间，提高效率，`Pre-Forward Hook`中也会提前预取当前`submodule`后的`submodule`的参数，并对其标记以便后续调用。

- 4. `param_coordinator.increment_step` `Step`会更新当前`Submodule`在`trace`中走到了哪一步，从而确定之后`prefetch_next_sub_modules`的起点。

在最后，经过以上的三步处理，便实现了：

- 完成`submodule`计算所需的所有`parameter`重建。
- 完成下一个`submodule`计算的准备。
- `submodule`加入`most_recent_sub_module_step`字典中并做记录。

在第一个`iteration`后，通过之前创建好的`trace`，在之后计算过程中**按照trace中的顺序，从当前step进行对参数的fetch和eager prefetch**。

通过以上完整的四个步骤，就实现了一个`submodule`在`Pre-forward hook`中的操作。在实际过程中，因为`module`可以逐层分成多个`submodule`，所以整个`module`的`forward`过程中会不断的对各`submodule`重复以上操作。

- 前向传播中 Model Parameter 的分割释放（Post-Forward Hook）

<img src="https://pica.zhimg.com/v2-cf3299d245377f8a29923ae56c75d5f9_720w.jpg?source=d16d100b" alt="img" style="zoom:30%;" />

> Post-Forward Hook

当`submodule`完成正向传播计算后，`post_forward_hook`会释放掉当前的`subomdule`，参数也会再次被 `partition`。但与初始化`partition`不同的是，此时每个进程中已经有了自己的小段data，所以此时`partition`只需要把**计算前重建的完整大tensor**再次释放掉：

```python
# param.data does not store anything meaningful in partitioned state
param.data = torch.ones(1, dtype=self.dtype).to(param.device)
```

通过这样的方式，每个进程中 submodule 只需要在**计算前收集参数**，**计算后释放参数**，从而大大减少了冗余空间占用。

当`module`所有的`submodule`都完整正向传播完成后，`engine`会将记录`submodule`执行顺序的`step_id`重新归为0，重新回到整个计算trace最初起点，准备下一次计算流程的开始。

#### 3.3.4 反向传播中的ZeRO-3

-  反向传播中 Model Parameter 的获取（Pre-Backward Hook）

<img src="https://pic1.zhimg.com/v2-959d40a19144cd778f01a0ff79dc2532_1440w.jpg" alt="img" style="zoom:33%;" />

> Backward Hooks

`pre-backward_hook`也是通过`record_trace`， `fetch_sub_module`, `prefetch_next_sub_modules`和`next_step`来实现过程的记录、参数的获取，并为下一步准备。

但是，由于 PyTorch 不支持`Pre Backward Hook`，因此这里得曲线救国一下：使用`register_forward_hook`挂上一个`autograd.Function`，这样就可以实现在 **module backward 之前执行自定义的操作**。在`backward`前，参数收集和分割的操作通过`torch.autograd.Function`挂在了各个`submodule`的`tensor`上。

当该`tensor`反向传播计算时，`autograd`的`backward`会调用`ctx.pre_backward_function(ctx.module)`依次完成：

- record_trace
- fetch_sub_module
- prefetch_next_sub_modules
- next_step

​		这四步操作也与`Pre-Forward Hook`中的四步操作一致。

- 反向传播中 Model Parameter 的分割释放（Post-Backward Hook）

当`backward`结束之后，`PostBackward hook`中的`PostBackward Function`也会和`post_forward_function`一样将`parameter`释放，从而减少`model parameter`的空间占用。[3]

3.3.5 Evaluation

<img src="https://pic2.zhimg.com/v2-9077a1ef9a39eecef0749afdef30f10f_1440w.jpg" alt="img" style="zoom: 33%;" />

> ZeRO Evaluation [1]

ZeRO 在 stage2 时就可在如下四个方面有杰出的表现。

>  ZeRO-R optimizes activation memory by identifying and removing activation replication in existing MP approaches through activation partitioning. It also offloads activations to CPU when appropriate.

在 ZeRO-2 和 ZeRO-R 配合可以支持高达170 billion 参数的模型训练。

- 模型规模：相较于 Megatron 局限于 40B parameters，ZeRO-2 和 ZeRO-R 的组合可以支持多达 170 billion 参数的模型训练，是当前 SOTA 方式的 8 倍。
- 训练速度：在 400 张 Nvidia V100 GPU 集群上，ZeRO 可以将 100B 参数量的模型训练速度提升近 10 倍，达到 38 TFlops/GPU，总体高达 15 Petaflops。
- 延展性：在 64-400 个 GPUs 区间，ZeRO 使训练速度具备超 GPU增量的加速比。`model states`内存占用的减少，支持了更大的`batch sizes`的训练，从而提升模型的整体表现。
- 易用性：数据和模型开发人员无需做任何模型并行就可训练高达 13 billion 参数的模型，从而减少了模型重构带来的成本开销。[1]

**在 ZeRO-3 的加持下，ZeRO Optimization 性能会得到进一步的提升。**

ZeRO-3 可以在单纯数据并行的模式下，实现在 1024 个 GPUs 上训练超过 1 Trillion 的模型。配合模型并行，ZeRO 通过 16 路模型并行和 64 路数据并行，更是支持高达超过 2 Trillion 的模型训练[1]。

## 4 What's next ? ZeRO 的局限与大模型训练的未来

### **4.1 简单粗暴的ZeRO也有局限性**

ZeRO 在每个 `submodule` 的前向和反向传播中进行了参数的`collection`与`partition`。在这种策略下：

1. 单个 `submodule` 在前向或反向传播中所占用的显存（参数、梯度、Outputs、Workspace）小于单个GPU的容量。
2.  频繁利用通信来传递参数、梯度等信息，导致通信成为瓶颈。

#### 4.1.1 大 Layer

例如 Transformer Model 中的一个`64K hidden dimension Layer`，在 Float16 下也需要超过 64GB 的显存来储存模型参数和梯度。在计算正向和反向传播时，需要至少两个超过 32GB 的连续 memory buffer。这样的需求即使在 NVIDIA A100 中也很难满足。为了解决超大 Layer 这一难题，研究人员在 ZeRO 基础之上引入了对单层 Layer 的拆分技术，也就是俗称的**模型并行**。这里简单提一下两个比较有意思的工作：

- **Megatron-LM** [5] 中充分利用了 Transformer 的模型结构，对多个 GEMM 进行了相当高效的拆分。在`MLP`中，以纵向并行的方式划分`第一个 GEMM`，后续的`GeLU`与`第二个GEMM`只在本地进行，唯一的通信在`Dropout`前对`第二个GEMM`的输出做个加和。通过这样的方式，GEMM 就可以被分到不同的 GPU 上，并只需在正向和反向传播时各做一次`AllReduce`。对于`Self-Attention`模块其也用了类似的拆分方法，核心仍是利用了**分块矩阵乘法**。

<img src="https://pica.zhimg.com/v2-2569289dabdb4fa7866c8b6bfed1c9b2_720w.jpg?source=d16d100b" alt="img" style="zoom:50%;" />

> Megatron-LM Structure

- **Optimus** [7] 同样利用了 Transformer 模型矩阵乘法的本质，但是不在行列的维度上分割矩阵，而是采用二维矩阵分割，并在理论效率上显著超过了前者。（PS：这两个工作的名字真是因吹斯汀

#### 4.1.2 大通信

<img src="https://pic1.zhimg.com/v2-e0ce614c13e29c16a70a5e8bf54e69dc_720w.jpg?source=d16d100b" alt="img" style="zoom:50%;" />

> 流水线并行

通信问题则主要考虑引入流水线并行来缓解。流水线并行将模型按层切分成了很多个 Stage，每一个 Worker 只持有一部分 Layer。切分后，不但每张卡上的参数和计算量减少了，同时 Worker 和 Worker 之间也只需要通信**临界层的 Activations**。

对于 Transformer 模型来说，临界层的 Activations 大小远远小于参数、梯度的大小，因此可以采用在**节点间做流水线并行，节点内多卡做数据并行的方式**来缓解节点间的通信压力，同时充分利用节点内的超高带宽。也可以将数据并行分为两级，一级在节点内做通信量较大的 ZeRO 数据并行，另一级在多个流水线并行间做普通的数据并行。

### **4.2 最后**

细心的朋友可能已经发现了，将上述的流水线并行、模型并行与数据并行相融合，就成了目前火热的 **3D 混合并行**。也正是 3D 混合并行支撑起了 GPT-3、盘古等千亿参数 Transformer 模型的训练，纵然 3D 混合并行恐怖如斯，其仍然有许多局限性，这个就放在之后的系列分享中再展开了。

## 引用

-  [1] [Samyam R, Jeff R, Olatunji R, Yuxiong H. ZeRO: Memory Optimizations Toward Training Trillion  Parameter Models. arxiv.org/pdf/1910.02054. 2019.](http://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1910.02054.pdf)

-  [2] [Turing-NLG: A 17-billion-parameter language model by Microsoft](http://link.zhihu.com/?target=https%3A//www.microsoft.com/en-us/research/blog/turing-nlg-a-17-billion-parameter-language-model-by-microsoft/)

-  [3] [Rangan M, Junhua W. ZeRO & DeepSpeed: New system optimizations enable training models with over 100 billion parameters. 2020.](http://link.zhihu.com/?target=https%3A//www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)

-  [4] [KDD 2020: Hands on Tutorials: Deep Speed -System optimizations enable training deep learning models](http://link.zhihu.com/?target=https%3A//www.youtube.com/watch%3Fv%3DczgA-MbAdvA%26t%3D2550s)

-  [5] [Mohammad S,  Mostofa P,  Raul P, et al. Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism arxiv.org/abs/1909.08053 .2019](http://link.zhihu.com/?target=https%3A//arxiv.org/abs/1909.08053)

-  [6] [Rangan M, Andrey P. ZeRO-Infinity and DeepSpeed: Unlocking unprecedented model scale for deep learning training. 2021](http://link.zhihu.com/?target=https%3A//www.microsoft.com/en-us/research/blog/zero-infinity-and-deepspeed-unlocking-unprecedented-model-scale-for-deep-learning-training/)

-  [7] [Xu Q, Li S, Gong C, et al. An Efficient 2D Method for Training Super-Large Deep Learning Models[J\]. arXiv preprint arXiv:2104.05343, 2021.](http://link.zhihu.com/?target=https%3A//arxiv.org/abs/2104.05343)


## 附录

PyTorch 的模型必须具有以下的三种特性：

1.必须继承`nn.Module`这个类，要让 PyTorch 知道这个类是一个 Module

2.在`init(self)`中设置好需要的"组件"(如`conv,pooling,Linear,BatchNorm`等)

3.最后，在`forward(self,x)`中定义好的“组件”进行组装，就像搭积木，把网络结构搭建出来，这样一个模型就定义好了。

根据 PyTorch 的文档介绍，  `nn.Module`是所有模型的基础 class，我们构建的各种模型网络也是这个`nn.Module的subclass`，并且每个 Module 也可以包含其他的 Module。

>  “All network components should inherit from nn.Module and override the forward() method. That is about it, as far as the boilerplate is concerned. Inheriting from nn.Module pro ides functionality to your component. ”

```python
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)# submodule: Conv2d
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
       x = F.relu(self.conv1(x))
       return F.relu(self.conv2(x))
```

PyTorch 给出的上述例子中，`class Model`就是继承了`nn.Module`，其内部两个`nn.Conv2d`各自也继承了`nn.Module`，`nn.Conv2d`就是`class Model`的`submodule`了。在 stage3 中，ZeRO 就是利用了 module 的这种嵌套的特性来实现模型参数的记录和并行。
