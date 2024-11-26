# Pytorch 分布式

本文由浅入深讲解 torch.distributed 这一并行计算包的概念，实现细节和应用方式，并带大家快速入门 PyTorch 分布式训练。

## **torch.distributed 概念与定义**

**定义**：首先我们提供 Torch.distributed 的官方定义

- torch.distributed 包为运行在一台或多台机器上的多个计算节点之间的 **PyTorch 提供支持多进程并行性通信的原语**。他能轻松地并行化在跨进程和机器集群的计算。
- torch.nn.parallel.DistributedDataParalle(DDP) 是建立在此功能的基础上，以提供同步的分布式训练作为任何 PyTorch 模型的包装器。

可以注意到的是，torch.distributed 的核心功能是进行多进程级别的通信（而非多线程），以此达到多卡多机分布式训练的目的。这与基于 DataParrallel 的多线程训练有明显区别。

**通信方式**：torch.distributed 的底层通信主要使用 Collective Communication (c10d) library 来支持跨组内的进程发送张量，并主要支持两种类型的通信 API：

- collective communication APIs: Distributed Data-Parallel Training (DDP)
- P2P communication APIs: RPC-Based Distributed Training (RPC)

这两种通信 API 在 PyTorch 中分别对应了两种分布式训练方式：Distributed Data-Parallel Training (DDP) 和 RPC-Based Distributed Training (RPC)。本文着重探讨 Distributed Data-Parallel Training (DDP) 的通信方式和 API

**基础概念：** 下面介绍一些 torch.distributed 中的关键概念以供参考。这些概念在编写程序时至关重要

- Group（进程组）是我们所有进程的子集。
- Backend（后端）进程通信库。PyTorch 支持 **NCCL，GLOO，MPI**。本文不展开讲几种通信后端的区别，感兴趣的同学可以参考官方文档
- world_size（世界大小）在进程组中的进程数。
- Rank（秩）分配给分布式进程组中每个进程的唯一标识符。 它们始终是从 0 到 world_size 的连续整数。



## **torch.distributed 实例**

### **例子 1：初始化**

```python
"""run.py:"""
#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process

def run(rank, size):
    """ Distributed function to be implemented later. """
    pass

def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
fn(rank, size)

if __name__ == "__main__":
    size = 2
    processes = []
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
```

本段程序执行了下面三件事

1. 创建了两个进程
2. 分别加入一个进程组
3. 分别运行 run 函数。此时 run 是一个空白函数，之后的例子会扩充这个函数的内容并在函数内完成多进程的通信操作。

### **例子 2：点对点通信**

最简单的多进程通信方式是点对点通信。信息从一个进程被发送到另一个进程。

![img](https://pic1.zhimg.com/v2-bc887111330cd0225c68c7cd353dae0d_720w.jpg?source=d16d100b)

```python
def run(rank, size):
    tensor = torch.zeros(1)
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        dist.send(tensor=tensor, dst=1)
    else:
        # Receive tensor from process 0
        dist.recv(tensor=tensor, src=0)
    print('Rank ', rank, ' has data ', tensor[0])
```

在上面的示例中，两个进程都从 tensor(0) 开始，然后进程 0 递增张量并将其发送到进程 1，以便它们都以 tensor(1) 结尾。 请注意，进程 1 需要分配内存以存储它将接收的数据。

另请注意，send / recv 被**阻塞**：两个过程都停止，直到通信完成。我们还有另外一种无阻塞的通信方式，请看下例

```python
"""Non-blocking point-to-point communication."""

def run(rank, size):
    tensor = torch.zeros(1)
    req = None
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        req = dist.isend(tensor=tensor, dst=1)
        print('Rank 0 started sending')
    else:
        # Receive tensor from process 0
        req = dist.irecv(tensor=tensor, src=0)
        print('Rank 1 started receiving')
    req.wait()
    print('Rank ', rank, ' has data ', tensor[0])
```

我们通过调用 wait 函数以使自己在子进程执行过程中保持休眠状态。由于我们不知道何时将数据传递给其他进程，因此在 req.wait() 完成之前，我们既不应该修改发送的张量也不应该访问接收的张量以防止不确定的写入.

### **例子 3：进程组间通信**

与点对点通信相反，集合允许跨组中所有进程的通信模式。例如，为了获得所有过程中所有张量的总和，我们可以使用 dist.all_reduce(tensor, op, group) 函数进行组间通信

```python
""" All-Reduce example."""
def run(rank, size):
    """ Simple point-to-point communication. """
    group = dist.new_group([0, 1])
    tensor = torch.ones(1)
    dist.all_reduce(tensor, op=dist.reduce_op.SUM, group=group)
    print('Rank ', rank, ' has data ', tensor[0])
```

这段代码首先将进程 0 和 1 组成进程组，然后将各自进程中 tensor(1) 相加。由于我们需要组中所有张量的总和，因此我们将 dist.reduce_op.SUM 用作化简运算符。 一般来说，任何可交换的数学运算都可以用作运算符。 PyTorch 开箱即用，带有 4 个这样的运算符，它们都在元素级运行：

- dist.reduce_op.SUM
- dist.reduce_op.PRODUCT
- dist.reduce_op.MAX
- dist.reduce_op.MIN

除了 dist.all_reduce(tensor, op, group) 之外，PyTorch 中目前共有 6 种组间通信方式

![img](https://pic1.zhimg.com/v2-812552c20c0785cf5dbd1f2182e79b9d_720w.jpg?source=d16d100b)

1. distributed.scatter(tensor, scatter_list=None, src=0, group=None, async_op=False)： 将张量 scatter_list[i] 复制第 i 个进程的过程。 例如，在实现分布式训练时，我们将数据分成四份并分别发送到不同的机子上计算梯度。scatter 函数可以用来将信息从 src 进程发送到其他进程上。

|              |                                             |
| ------------ | ------------------------------------------- |
| tensor       | 发送的数据                                  |
| scatter_list | 存储发送数据的列表（只需在 src 进程中指定） |
| dst          | 发送进程的rank                              |
| group        | 指定进程组                                  |
| async_op     | 该 op 是否是异步操作                        |

![img](https://pic1.zhimg.com/v2-602a2ed1126c9ef56e53235ab3f8adeb_720w.jpg?source=d16d100b)

2. distributed.gather(tensor, gather_list=None, dst=0, group=None, async_op=False)： 从 dst 中的所有进程复制 tensor。例如，在实现分布式训练时，不同进程计算得到的梯度需要汇总到一个进程，并计算平均值以获得统一的梯度。gather 函数可以将信息从别的进程汇总到 dst 进程。

|             |                                           |
| ----------- | ----------------------------------------- |
| tensor      | 接受的数据                                |
| gather_list | 存储接受数据的列表（只需在dst进程中指定） |
| dst         | 汇总进程的rank                            |
| group       | 指定进程组                                |
| async_op    | 该op是否是异步操作                        |

![img](https://pic1.zhimg.com/v2-348e954c4c77ef281c6204bccf0c8f5f_720w.jpg?source=d16d100b)

3. distributed.reduce(tensor, dst, op, group)：将 op 应用于所有 tensor，并将结果存储在 dst 中。

![img](https://picx.zhimg.com/v2-b7597d4d57bbc6ba47a59166b8331d8f_720w.jpg?source=d16d100b)

4. distributed.all_reduce(tensor, op, group)： 与 reduce 相同，但是结果存储在所有进程中。

![img](https://pic1.zhimg.com/v2-8ae0a62a27420e1de19fbbea5dc9a09b_720w.jpg?source=d16d100b)

5. distributed.broadcast(tensor, src, group)：将tensor从src复制到所有其他进程。

![img](https://picx.zhimg.com/v2-21ce7cb6b3be25d25ee02b5fe0b9c70c_720w.jpg?source=d16d100b)

6. distributed.all_gather(tensor_list, tensor, group)：将所有进程中的 tensor 从所有进程复制到 tensor_list



### **例子 4：分布式梯度下降**

分布式梯度下降脚本将允许**所有进程在其数据 batch 上计算其模型的梯度，然后平均其梯度**。 为了在更改进程数时确保相似的收敛结果，我们首先必须对数据集进行分区。

```python
""" Dataset partitioning helper """
class Partition(object):

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]

class DataPartitioner(object):

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])
```

使用上面的代码片段，我们现在可以使用以下几行简单地对任何数据集进行分区

```python
""" Partitioning MNIST """
def partition_dataset():
    dataset = datasets.MNIST('./data', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))
                             ]))
    size = dist.get_world_size()
    bsz = 128 / float(size)
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(partition,
                                         batch_size=bsz,
                                         shuffle=True)
    return train_set, bsz
```

假设我们有 2 个进程，则每个进程的 train_set 为 60000/2 = 30000 个样本。 我们还将 batch 大小除以进程数，以使整体 batch 大小保持为 128。

现在，我们可以编写通常的向前-向后优化训练代码，并添加一个函数调用以平均模型的梯度。

```python
""" Distributed Synchronous SGD Example """
def run(rank, size):
    torch.manual_seed(1234)
    train_set, bsz = partition_dataset()
    model = Net()
    optimizer = optim.SGD(model.parameters(),
                          lr=0.01, momentum=0.5)

    num_batches = ceil(len(train_set.dataset) / float(bsz))
    for epoch in range(10):
        epoch_loss = 0.0
        for data, target in train_set:
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            epoch_loss += loss.item()
            loss.backward()
            average_gradients(model)
            optimizer.step()
        print('Rank ', dist.get_rank(), ', epoch ',
              epoch, ': ', epoch_loss / num_batches)
```

仍然需要执行 average_gradients(model) 函数，该函数只需要一个模型并计算在所有 rank 上梯度的平均值。

```python
""" Gradient averaging. """
def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size
```

## **3 PyTorch 并行/分布式训练**

在掌握 torch.distributed 的基础的前提下，我们可以根据自身机器和任务的具体情况使用不同的分布式或并行训练方式：

- 如果数据和模型可以放在一个 GPU 中，并且不关心训练速度，请使用单设备训练。
- 如果单个服务器上有多个 GPU，并且您希望更改较少的代码来加快训练速度，请使用单机多 GPU DataParallel。
- 如果单个服务器上有多个 GPU，且您希望进一步添加代码并加快训练速度，请使用单机多 GPU DistributedDataParallel。
- 如果应用程序需要跨多个服务器，请使用多机 DistributedDataParallel 和启动脚本。
- 如果预计会出现错误（例如，OOM），或者在训练期间资源可以动态加入和离开，请使用 torch.elastic 进行分布式训练。

**3.1 DataParallel**

```python
class torch.nn.DataParallel(module, device_ids=None, output_device=None, dim=0)
```

DataParallel 自动分割您的数据，并将作业订单发送到多个 GPU 上的多个模型。每个模型完成工作后，DataParallel 会收集并合并结果，然后再将结果返回给您。 DataParallel 将相同的模型复制到所有 GPU，其中每个 GPU 消耗输入数据的不同分区。在使用此方法时，batch 理大小应大于使用的 GPU 数量。我们需要注意的是，DataParallel 是通过多线程的方式进行的并行训练，所以并没有使用 torch.distributed 里的线程通信 API。的其运行过程如下图所示

![img](https://picx.zhimg.com/v2-05243eb53b00eb99834379790324835d_720w.jpg?source=d16d100b)

### **例子 5 DataParallel**

创建 dump 数据集和定义模型

```python
class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                         batch_size=batch_size, shuffle=True)

class Model(nn.Module):
    # Our model

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("\tIn Model: input size", input.size(),
              "output size", output.size())

        return output
```

定义模型，放入设备并用 DataParallel 对象进行包装

```python
model = Model(input_size, output_size)
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model)

model.to(device)
```

运行模型并输出

```python
for data in rand_loader:
    input = data.to(device)
    output = model(input)
    print("Outside: input size", input.size(),
          "output_size", output.size())
In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
        In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
        In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
        In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
        In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
        In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
        In Model: input size torch.Size([5, 5]) output size torch.Size([5, 2])
        In Model: input size torch.Size([5, 5]) output size torch.Size([5, 2])
Outside: input size torch.Size([10, 5]) output_size torch.Size([10, 2])
```

我们可以看到，在模型中，数据是按照batch大小的维度被均匀分成多份。在输出后，多块 GPU 上的数据进行合并。

**3.2 DistributedDataParallel**

当我们了解了 DataParallel 后，下面开始介绍一种基于 torch.distributed 中进程通信函数包装的高层 API

```python
CLASS torch.nn.parallel.DistributedDataParallel(module, device_ids=None, output_device=None, dim=0, broadcast_buffers=True, process_group=None, bucket_cap_mb=**25**, find_unused_parameters=False, check_reduction=False, gradient_as_bucket_view=False)
```

既然 DataParallel 可以进行并行的模型训练，那么为什么还需要提出 DistributedDataParallel呢？这里我们就需要知道两种方法的实现原理与区别：

1. 如果模型太大而无法容纳在单个 GPU 上，则必须使用模型并行将其拆分到多个 GPU 中。 DistributedDataParallel 可以与模型并行一起使用； 但 DataParallel 因为必须将模型放入单块 GPU 中，所以难以完成大型模型的训练。
2. **DataParallel 是单进程，多线程的并行训练方式**，并且只能在单台机器上运行，而**DistributedDataParallel 是多进程**，并且适用于单机和多机训练。DistributedDataParallel 还预先复制模型，而不是在每次迭代时复制模型，并避免了全局解释器锁定。
3. 如果您的两个数据都太大而无法容纳在一台计算机和上，而您的模型又太大了以至于无法安装在单个 GPU 上，则可以将模型并行（跨多个 GPU 拆分单个模型）与 DistributedDataParallel 结合使用。 在这种情况下，每个 DistributedDataParallel 进程都可以并行使用模型，而所有进程都将并行使用数据。

### **例子 6 DistributedDataParallel**

首先我们需要创建一系列进程，其中需要用到 torch.multiprocessing 中的函数

```python
torch.multiprocessing.spawn(fn, args=(), nprocs=1, join=True, daemon=False, start_method='spawn')
```

该函数使用 args 作为参数列表运行函数fn，并创建 nprocs 个进程。

如果其中一个进程以非零退出状态退出，则其余进程将被杀死，并引发异常，以终止原因。如果子进程中捕获到异常，则将其转发并将其回溯包括在父进程中引发的异常中。

该函数会通过 fn(i，args) 的形式被调用，其中i是进程索引，而 args 是传递的参数元组。

基于创建的的进程，我们初始化进程组

```python
import os
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Explicitly setting seed to make sure that models created in two processes
    # start from same random weights and biases.
    torch.manual_seed(42)

def cleanup():
    dist.destroy_process_group()
```

这里我们使用到了

```python
torch.distributed.init_process_group(backend, init_method=None, timeout=datetime.timedelta(0, 1800), world_size=-1, rank=-1, store=None, group_name='')
```

这个 API 来初始化默认的分布式进程组，这还将初始化分布式程序包。

该函数有两种主要的调用方式：

1.  明确指定 store，rank 和 world_size。
2.  指定 init_method（URL 字符串），它指示在何处/如何发现对等方。 （可选）指定 rank 和 world_size，或在 URL 中编码所有必需的参数并忽略它们。

现在，让我们创建一个 toy model，将其与 DDP 封装在一起，并提供一些虚拟输入数据。 请注意，由于 DDP 将 0 级进程中的模型状态广播到 DDP 构造函数中的所有其他进程，因此无需担心不同的 DDP 进程从不同的模型参数初始值开始。

```python
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

def demo_basic(rank, world_size):
    setup(rank, world_size)
    # Assume we have 8 GPU in total
    # setup devices for this process, rank 1 uses GPUs [0, 1, 2, 3] and
    # rank 2 uses GPUs [4, 5, 6, 7].
    n = torch.cuda.device_count() // world_size
    device_ids = list(range(rank * n, (rank + 1) * n))

    # create model and move it to device_ids[0]
    model = ToyModel().to(device_ids[0])
    # output_device defaults to device_ids[0]
    ddp_model = DDP(model, device_ids=device_ids)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(device_ids[0])
    loss_fn(outputs, labels).backward()
    optimizer.step()

    cleanup()

def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)
if __name__ == "__main__":
    run_demo(demo_basic, 2)
```

### **例子 7 将 DDP 与模型并行性结合**

DDP 还可以与多 GPU 模型一起使用，但是不支持进程内的复制。 您需要为每个模型副本创建一个进程，与每个进程的多个模型副本相比，通常可以提高性能。 当训练具有大量数据的大型模型时，DDP 包装多 GPU 模型特别有用。 使用此功能时，需要小心地实现多 GPU 模型，以避免使用硬编码的设备，因为会将不同的模型副本放置到不同的设备上。

例如，下面这个模型显式的将不同的模块放置在不同的 GPU 上

```python
class ToyMpModel(nn.Module):
    def __init__(self, dev0, dev1):
        super(ToyMpModel, self).__init__()
        self.dev0 = dev0
        self.dev1 = dev1
        self.net1 = torch.nn.Linear(10, 10).to(dev0)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5).to(dev1)

    def forward(self, x):
        x = x.to(self.dev0)
        x = self.relu(self.net1(x))
        x = x.to(self.dev1)
        return self.net2(x)
```

将多 GPU 模型传递给 DDP 时，不得设置 device_ids 和 output_device。 输入和输出数据将通过应用程序或模型 forward() 方法放置在适当的设备中。

```python
def demo_model_parallel(rank, world_size):
    setup(rank, world_size)

    # setup mp_model and devices for this process
    dev0 = rank * 2
    dev1 = rank * 2 + 1
    mp_model = ToyMpModel(dev0, dev1)
    ddp_mp_model = DDP(mp_model)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_mp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    # outputs will be on dev1
    outputs = ddp_mp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(dev1)
    loss_fn(outputs, labels).backward()
    optimizer.step()

    cleanup()
if __name__ == "__main__":
    run_demo(demo_model_parallel, 4)
```

### **例子 8 保存和加载检查点**

使用 DDP 时，一种优化方法是仅在一个进程中保存模型，然后将其加载到所有进程中，从而减少写开销。

```python
def demo_checkpoint(rank, world_size):
    setup(rank, world_size)

    # setup devices for this process, rank 1 uses GPUs [0, 1, 2, 3] and
    # rank 2 uses GPUs [4, 5, 6, 7].
    n = torch.cuda.device_count() // world_size
    device_ids = list(range(rank * n, (rank + 1) * n))

    model = ToyModel().to(device_ids[0])
    # output_device defaults to device_ids[0]
    ddp_model = DDP(model, device_ids=device_ids)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    CHECKPOINT_PATH = tempfile.gettempdir() + "/model.checkpoint"
    if rank == 0:
        # All processes should see same parameters as they all start from same
        # random parameters and gradients are synchronized in backward passes.
        # Therefore, saving it in one process is sufficient.
        torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)

    # Use a barrier() to make sure that process 1 loads the model after process
    # 0 saves it.
    dist.barrier()
    # configure map_location properly
    rank0_devices = [x - rank * len(device_ids) for x in device_ids]
    device_pairs = zip(rank0_devices, device_ids)
    map_location = {'cuda:%d' % x: 'cuda:%d' % y for x, y in device_pairs}
    ddp_model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=map_location))

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(device_ids[0])
    loss_fn = nn.MSELoss()
    loss_fn(outputs, labels).backward()
    optimizer.step()

    # Use a barrier() to make sure that all processes have finished reading the
    # checkpoint
    dist.barrier()

    if rank == 0:
        os.remove(CHECKPOINT_PATH)

    cleanup()
```

##  4. DP

### 1.1 使用

DP 的好处是，使用起来非常方便，只需要将原来单卡的 module 用 DP 改成多卡:

```python
model = nn.DataParallel(model)
```

### 1.2 原理

DP 基于单机多卡，所有设备都负责计算和训练网络，除此之外， device[0] (并非 GPU 真实标号而是输入参数 device_ids 首位) 还要负责整合梯度，更新参数。图 1 即为 GPU 0 作为 device[0] 的例子。从图中我们可以看出，有三个主要过程：

- 过程一（图中红色部分）：各卡分别计算损失和梯度
- 过程二（图中蓝色部分）：所有梯度整合到 device[0]
- 过程三（图中绿色部分）：device[0] 进行参数更新，其他卡拉取 device[0] 的参数进行更新

所有卡都并行运算（图中红色），将梯度收集到 device[0]（图中浅蓝色）和 device[0] 分享模型参数给其他 GPU（图中绿色）三个主要过程。

<img src="https://picx.zhimg.com/v2-5c5b0d8e3d7d6653a9ebd47bac93090c_720w.jpg?source=d16d100b" alt="img" style="zoom:50%;" />

> 图 1: GPU 0 as device[0]，原图见 [2]

虽然 DP 只能实现单机训练不能算是严格意义上的分布式训练（多个节点），但是其原理和分布式训练算法里的 Parameter Server 架构很相近，我们借用 PS 的伪代码来说明一下。

![img](https://picx.zhimg.com/v2-24e22dafeef6541c3bc474f9a6737061_720w.jpg?source=d16d100b)

> 图 2: PS，原图见 [3]

我们可以看到 PS 的并行梯度下降流程分为四部分：

Task Scheduler：

- 负责加载数据并分发数据至每个 worker 节点，并执行多轮迭代。

在每轮迭代中，worker 负责：

- 初始化：载入数据并将全部模型参数从 server 节点拉下来（图 1 绿色）
- 梯度计算：利用该节点的数据计算梯度（图 1 红色）并将梯度更新到 server 节点（图 1 蓝色）

Server 负责：

- 汇总梯度
- 更新参数

OK， 现在我们已经知道了 DP 使用的算法，接下来我们看一下 PyTorch 是如何实现的。

### 1.3 实现

这一节主要讨论 DP 的实现，首先先贴上源码（顺便看一下comment 部分）：

```python
class DataParallel(Module):

    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(DataParallel, self).__init__()

        # 检查是否有可用的 GPU
        device_type = _get_available_device_type()
        if device_type is None:
            self.module = module
            self.device_ids = []
            return
                # 默认使用所有可见的 GPU
        if device_ids is None:
            device_ids = _get_all_device_indices()

                # 默认 server 是 device_ids 列表上第一个
        if output_device is None:
            output_device = device_ids[0]

        self.dim = dim
        self.module = module
        self.device_ids = list(map(lambda x: _get_device_index(x, True), device_ids))
        self.output_device = _get_device_index(output_device, True)
        self.src_device_obj = torch.device(device_type, self.device_ids[0])

        # 检查负载是否平衡， 不平衡（指内存或者处理器 max/min > 0.75 会有警告）
        _check_balance(self.device_ids)

        # 单卡
        if len(self.device_ids) == 1:
            self.module.to(self.src_device_obj)

    def forward(self, *inputs, **kwargs):

        # 没 GPU 可用
        if not self.device_ids:
            return self.module(*inputs, **kwargs)

                # 运行前 GPU device_ids[0] （即我们的 server ）上必须有 parallelized module 的parameters 和 buffers
        # 因为 DP 保证 GPU device_ids[0] 和 base parallelized module 共享存储
        # 所以在device[0] 上的 in-place 更新也会被保留下来，其他的则不会

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError("module must have its parameters and buffers "
                                   "on device {} (device_ids[0]) but found one of "
                                   "them on device: {}".format(self.src_device_obj, t.device))

                # nice 现在 device[0] 上已经有了 module 和 input， 接下来我们就要开始 PS 算法了
        # 可以开始看正文了

        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)

        # 如果仅有单卡可用，直接单卡计算，不用并行
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])

        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    def replicate(self, module, device_ids):
        return replicate(module, device_ids, not torch.is_grad_enabled())

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(replicas)])

    def gather(self, outputs, output_device):
        return gather(outputs, output_device, dim=self.dim)
```

从 forward 函数可以看出，关键函数有 scatter, replicate, parallel_apply 和 gather，我们一个一个看一下。

首先是 scatter 函数，即 scatter_kwargs 函数。

```python
def scatter_kwargs(inputs, kwargs, target_gpus, dim=0):
    r"""Scatter with support for kwargs dictionary"""

    # 主要函数
    inputs = scatter(inputs, target_gpus, dim) if inputs else []
    kwargs = scatter(kwargs, target_gpus, dim) if kwargs else []

    # 用空项补全使 inputs 和 kwargs 长度相当
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    # 返回 tuple
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs
```

scatter_kwargs 函数中最重要的就是 scatter 函数，负责将 tensor 分成大概相等的块并将他们分给不同的 GPU。对其他的数据类型，则是复制分散给不同的 GPU 。

```python
def scatter(inputs, target_gpus, dim=0):
    r"""
    Slices tensors into approximately equal chunks and
    distributes them across given GPUs. Duplicates
    references to objects that are not tensors.
    """
    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            return Scatter.apply(target_gpus, None, dim, obj)
        if is_namedtuple(obj):
            return [type(obj)(*args) for args in zip(*map(scatter_map, obj))]
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return [list(i) for i in zip(*map(scatter_map, obj))]
        if isinstance(obj, dict) and len(obj) > 0:
            return [type(obj)(i) for i in zip(*map(scatter_map, obj.items()))]
        return [obj for targets in target_gpus]

    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        res = scatter_map(inputs)
    finally:
        scatter_map = None
    return res
```

其中，针对 tensor 的函数，

```python
class Scatter(Function):

    @staticmethod
    def forward(ctx, target_gpus, chunk_sizes, dim, input):
        target_gpus = [_get_device_index(x, True) for x in target_gpus]
        ctx.dim = dim
        ctx.input_device = input.get_device() if input.device.type != "cpu" else -1
        streams = None
        if torch.cuda.is_available() and ctx.input_device == -1:
            # Perform CPU to GPU copies in a background stream

            # 新建 cuda stream
            streams = [_get_stream(device) for device in target_gpus]

        # 真正的操作
        outputs = comm.scatter(input, target_gpus, chunk_sizes, ctx.dim, streams)

        # Synchronize with the copy stream
        if streams is not None:
            for i, output in enumerate(outputs):
                with torch.cuda.device(target_gpus[i]):
                    main_stream = torch.cuda.current_stream()
                    main_stream.wait_stream(streams[i])
                    output.record_stream(main_stream)
        return outputs

    @staticmethod
    def backward(ctx, *grad_output):
        return None, None, None, Gather.apply(ctx.input_device, ctx.dim, *grad_output)
```

comm.scatter 依赖于 C++，就不介绍了。

回顾 DP 代码块，我们已经运行完 scatter函数，即将一个 batch 近似等分成更小的 batch。接下来我们要看 replicate 函数和 gather 函数 （假设我们有不少于两张卡）。

```python
#  DP forward 里的代码
    replicas = self.replicate(self.module, self.device_ids[:len(inputs)])

    # 实现
    def replicate(network, devices, detach=False):

        if not _replicatable_module(network):
            raise RuntimeError("Cannot replicate network where python modules are "
                               "childrens of ScriptModule")

        if not devices:
            return []

        # 需要复制到哪些 GPU， 复制多少份
        devices = [_get_device_index(x, True) for x in devices]
        num_replicas = len(devices)

        # 复制 parameters
        params = list(network.parameters())
        param_indices = {param: idx for idx, param in enumerate(params)}

        # 拉到代码块底部看原函数，然后再回来
        param_copies = _broadcast_coalesced_reshape(params, devices, detach)


        # 复制 buffers
        buffers = list(network.buffers())
        buffers_rg = []
        buffers_not_rg = []
        for buf in buffers:
            if buf.requires_grad and not detach:
                buffers_rg.append(buf)
            else:
                buffers_not_rg.append(buf)

                # 记录需要和不需要求导的 buffer 的 index
        buffer_indices_rg = {buf: idx for idx, buf in enumerate(buffers_rg)}
        buffer_indices_not_rg = {buf: idx for idx, buf in enumerate(buffers_not_rg)}

                # 分别拷贝，这个咱们已经会了
        buffer_copies_rg = _broadcast_coalesced_reshape(buffers_rg, devices, detach=detach)
        buffer_copies_not_rg = _broadcast_coalesced_reshape(buffers_not_rg, devices, detach=True)

        # 现在开始拷贝网络
        # 准备过程：将 network.modules() 变成list
        # 然后再为之后复制的模型准备好空的 list 和 indices

        modules = list(network.modules())
        module_copies = [[] for device in devices]
        module_indices = {}
        scriptmodule_skip_attr = {"_parameters", "_buffers", "_modules", "forward", "_c"}

        for i, module in enumerate(modules):
            module_indices[module] = i
            for j in range(num_replicas):
                replica = module._replicate_for_data_parallel()
                # This is a temporary fix for DDP. DDP needs to access the
                # replicated model parameters. It used to do so through
                # `mode.parameters()`. The fix added in #33907 for DP stops the
                # `parameters()` API from exposing the replicated parameters.
                # Hence, we add a `_former_parameters` dict here to support DDP.
                replica._former_parameters = OrderedDict()

                module_copies[j].append(replica)

                # 接下来分别复制 module，param，buffer
        for i, module in enumerate(modules):
            for key, child in module._modules.items():
                if child is None:
                    for j in range(num_replicas):
                        replica = module_copies[j][i]
                        replica._modules[key] = None
                else:
                    module_idx = module_indices[child]
                    for j in range(num_replicas):
                        replica = module_copies[j][i]
                        setattr(replica, key, module_copies[j][module_idx])
            for key, param in module._parameters.items():
                if param is None:
                    for j in range(num_replicas):
                        replica = module_copies[j][i]
                        replica._parameters[key] = None
                else:
                    param_idx = param_indices[param]
                    for j in range(num_replicas):
                        replica = module_copies[j][i]
                        param = param_copies[j][param_idx]
                        # parameters in replicas are no longer leaves,
                        # so setattr them as non-parameter attributes
                        setattr(replica, key, param)
                        # expose the parameter for DDP
                        replica._former_parameters[key] = param
            for key, buf in module._buffers.items():
                if buf is None:
                    for j in range(num_replicas):
                        replica = module_copies[j][i]
                        replica._buffers[key] = None
                else:
                    if buf.requires_grad and not detach:
                        buffer_copies = buffer_copies_rg
                        buffer_idx = buffer_indices_rg[buf]
                    else:
                        buffer_copies = buffer_copies_not_rg
                        buffer_idx = buffer_indices_not_rg[buf]
                    for j in range(num_replicas):
                        replica = module_copies[j][i]
                        setattr(replica, key, buffer_copies[j][buffer_idx])

        return [module_copies[j][0] for j in range(num_replicas)]

    # ！！！从replicate来看这里
    def _broadcast_coalesced_reshape(tensors, devices, detach=False):

      from ._functions import Broadcast

      # 先看 else 的 comment，因为不 detach 也会用到同样的函数
      if detach:
          return comm.broadcast_coalesced(tensors, devices)
      else:
          # Use the autograd function to broadcast if not detach
          if len(tensors) > 0:

            # 下拉看源码
              tensor_copies = Broadcast.apply(devices, *tensors)

              return [tensor_copies[i:i + len(tensors)]
                      for i in range(0, len(tensor_copies), len(tensors))]
          else:
              return []

   #  Broadcast.apply
   class Broadcast(Function):

    @staticmethod
    def forward(ctx, target_gpus, *inputs):
        assert all(i.device.type != 'cpu' for i in inputs), (
            'Broadcast function not implemented for CPU tensors'
        )
        target_gpus = [_get_device_index(x, True) for x in target_gpus]
        ctx.target_gpus = target_gpus
        if len(inputs) == 0:
            return tuple()
        ctx.num_inputs = len(inputs)
        # input 放在 device[0]
        ctx.input_device = inputs[0].get_device()

        # 和 detach 的情况一样
        outputs = comm.broadcast_coalesced(inputs, ctx.target_gpus)

        # comm.broadcast_coalesced 的代码
        # tensors 必须在同一个设备，CPU 或者 GPU； devices 即是要拷贝到的设备；buffer_size 则是最大的buffer
        # 这里用到 buffer 将小张量合并到缓冲区以减少同步次数
        # def broadcast_coalesced(tensors, devices, buffer_size=10485760):
        #    devices = [_get_device_index(d) for d in devices]
            #       return torch._C._broadcast_coalesced(tensors, devices, buffer_size)

        non_differentiables = []
        for idx, input_requires_grad in enumerate(ctx.needs_input_grad[1:]):
            if not input_requires_grad:
                for output in outputs:
                    non_differentiables.append(output[idx])
        ctx.mark_non_differentiable(*non_differentiables)
        return tuple([t for tensors in outputs for t in tensors])

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None,) + ReduceAddCoalesced.apply(ctx.input_device, ctx.num_inputs, *grad_outputs)
```

下面继续 parallel_apply 部分。⚠️   DP 和 DDP 共用 parallel_apply 代码

```python
# DP 代码
outputs = self.parallel_apply(replicas, inputs, kwargs)

# threading 实现，用前面准备好的 replica 和输入数据，然后
# for 循环启动多线程

# 源码
def parallel_apply(modules, inputs, kwargs_tup=None, devices=None):

        # 每个 GPU 都有模型和输入
    assert len(modules) == len(inputs)

    # 确保每个 GPU 都有相应的数据，如没有就空白补全
    if kwargs_tup is not None:
      # 咱们在 scatter 已经补全了
        assert len(modules) == len(kwargs_tup)
    else:
        kwargs_tup = ({},) * len(modules)

    if devices is not None:
        assert len(modules) == len(devices)
    else:
        devices = [None] * len(modules)

    devices = [_get_device_index(x, True) for x in devices]

    # 多线程实现

    lock = threading.Lock()
    results = {}
    grad_enabled, autocast_enabled = torch.is_grad_enabled(), torch.is_autocast_enabled()

    # 定义 worker
    def _worker(i, module, input, kwargs, device=None):
        torch.set_grad_enabled(grad_enabled)
        if device is None:
            device = get_a_var(input).get_device()
        try:
            with torch.cuda.device(device), autocast(enabled=autocast_enabled):
                # this also avoids accidental slicing of `input` if it is a Tensor
                if not isinstance(input, (list, tuple)):
                    input = (input,)
                output = module(*input, **kwargs)
            with lock:
              # 并行计算得到输出
                results[i] = output
        except Exception:
            with lock:
                results[i] = ExceptionWrapper(
                    where="in replica {} on device {}".format(i, device))

    if len(modules) > 1:

      # 如有一个进程控制多个 GPU ，起多个线程
      # 需要强调一下，虽然 DDP 推荐单卡单进程，即每次调用 DDP device_ids 都只输入一张卡的 id（通常是 args.local_rank），但是如果输入多个 device_id，此时 DDP 就是单进程多线程控制多卡，和 DP 一样，关于 DDP 的解读可以看下文

        threads = [threading.Thread(target=_worker,
                                    args=(i, module, input, kwargs, device))
                   for i, (module, input, kwargs, device) in
                   enumerate(zip(modules, inputs, kwargs_tup, devices))]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
      # 一个 GPU 一个进程 （ DDP 推荐操作）
        _worker(0, modules[0], inputs[0], kwargs_tup[0], devices[0])

    outputs = []
    for i in range(len(inputs)):
        output = results[i]

        # error handle
        if isinstance(output, ExceptionWrapper):
            output.reraise()
        outputs.append(output)
    # 输出 n 个计算结果
    return outputs
```

现在我们已经得到并行计算的结果了，接下来我们要将结果收集到 device[0]。

```python
# DP 代码
return self.gather(outputs, self.output_device)
# 收集到 devices[0]

# 源码
def gather(outputs, target_device, dim=0):
    r"""
    Gathers tensors from different GPUs on a specified device
      (-1 means the CPU).
    """
    def gather_map(outputs):
        out = outputs[0]
        if isinstance(out, torch.Tensor):
            return Gather.apply(target_device, dim, *outputs)
        if out is None:
            return None
        if isinstance(out, dict):
            if not all((len(out) == len(d) for d in outputs)):
                raise ValueError('All dicts must have the same number of keys')
            return type(out)(((k, gather_map([d[k] for d in outputs]))
                              for k in out))
        return type(out)(map(gather_map, zip(*outputs)))

    # Recursive function calls like this create reference cycles.
    # Setting the function to None clears the refcycle.
    try:
        res = gather_map(outputs)
    finally:
        gather_map = None
    return res

# Gather 源码

class Gather(Function):

    @staticmethod
    def forward(ctx, target_device, dim, *inputs):
        assert all(i.device.type != 'cpu' for i in inputs), (
            'Gather function not implemented for CPU tensors'
        )

        target_device = _get_device_index(target_device, True)

        ctx.target_device = target_device

        ctx.dim = dim
        ctx.input_gpus = tuple(i.get_device() for i in inputs)

        if all(t.dim() == 0 for t in inputs) and dim == 0:
            inputs = tuple(t.view(1) for t in inputs)
            warnings.warn('Was asked to gather along dimension 0, but all '
                          'input tensors were scalars; will instead unsqueeze '
                          'and return a vector.')
            ctx.unsqueezed_scalar = True
        else:
            ctx.unsqueezed_scalar = False
        ctx.input_sizes = tuple(i.size(ctx.dim) for i in inputs)
        return comm.gather(inputs, ctx.dim, ctx.target_device)

    @staticmethod
    def backward(ctx, grad_output):
        scattered_grads = Scatter.apply(ctx.input_gpus, ctx.input_sizes, ctx.dim, grad_output)
        if ctx.unsqueezed_scalar:
            scattered_grads = tuple(g[0] for g in scattered_grads)
        return (None, None) + scattered_grads

# comm.gather 涉及到 C++，具体实现咱也不讲了 ；)
# Gathers tensors from multiple GPU devices.
def gather(tensors, dim=0, destination=None, *, out=None):
    tensors = [_handle_complex(t) for t in tensors]
    if out is None:
        if destination == -1:
            warnings.warn(
                'Using -1 to represent CPU tensor is deprecated. Please use a '
                'device object or string instead, e.g., "cpu".')
        destination = _get_device_index(destination, allow_cpu=True, optional=True)
        return torch._C._gather(tensors, dim, destination)
    else:
        if destination is not None:
            raise RuntimeError(
                "'destination' must not be specified when 'out' is specified, but "
                "got destination={}".format(destination))
        return torch._C._gather_out(tensors, out, dim)
```

因为大量实现都依赖 C++，这篇笔记就不涉及了。

最后，用一张图形象地看一下 DP Module 究竟怎么执行的，具体看第一行和第三行：

前向传播的时候我们会先用 Scatter 函数将数据从 device[0] 分配并复制到不同的卡，之后用 Replicate 函数将模型从 device[0] 复制到不同的卡，之后各个卡都有了同样的模型和不同的数据，分别调用 forward 计算损失和梯度。

反向传播的时候，我们会将梯度收集到 device[0] 然后在 device[0] 更新参数。

![img](https://pic1.zhimg.com/v2-4c163e1ff2541218e8829dcd0d209b8f_720w.jpg?source=d16d100b)

> Fig. 3: DP Module流程图，原图见[4]

### 1.4 分析

- 负载不均衡, device[0] 负载大一些

- 通信开销

假设有  k  个 GPU， 完成一次通信需要时间  $\frac{p}{b}$，那么使用 PS 算法，总共需要花费时间 $T = 2(k-1)\frac{p}{b}  $

- 单进程

>  The difference between `DistributedDataParallel` and `DataParallel` is: `DistributedDataParallel` uses multiprocessing where a process is created for each GPU, while `DataParallel` uses multithreading. By using multiprocessing, each GPU has its dedicated process, this avoids the performance overhead caused by GIL of Python interpreter.
>
>  [官方文档](http://link.zhihu.com/?target=https%3A//pytorch.org/docs/stable/notes/cuda.html%3Fhighlight%3Dbuffer)

- [Global Interpreter Lock (GIL)](http://link.zhihu.com/?target=https%3A//opensource.com/article/17/4/grok-gil) [全局解释器锁](https://zhuanlan.zhihu.com/p/20953544)，简单来说就是，一个 Python 进程只能利用一个 CPU kernel，即单核多线程并发时，只能执行一个线程。考虑多核，多核多线程可能出现线程颠簸 (thrashing) 造成资源浪费，所以 Python 想要利用多核最好是多进程。

## 5. DDP

### 2.1 使用

```python
import argparse
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", default='')
parser.add_argument("--local_rank", default=-1)
parser.add_argument("--world_size", default=1)
args = parser.parse_args()

# 初始化后端

# world_size 指的是总的并行进程数目
# 比如16张卡单卡单进程 就是 16
# 但是如果是8卡单进程 就是 1
# 等到连接的进程数等于world_size，程序才会继续运行
torch.distributed.init_process_group(backend='nccl',
                                         world_size=ws,
                                         init_method='env://')

torch.cuda.set_device(args.local_rank)

device = torch.device(f'cuda:{args.local_rank}')

model = nn.Linear(2,3).to(device)

# train dataset
# train_sampler
# train_loader

# 初始化 DDP，这里我们通过规定 device_id 用了单卡单进程
# 实际上根据我们前面对 parallel_apply 的解读，DDP 也支持一个进程控制多个线程利用多卡
model = DDP(model,
            device_ids=[args.local_rank],
            output_device=args.local_rank).to(device)


# 保存模型
if torch.distributed.get_rank() == 0:
  torch.save(model.module.state_dict(),
             'results/%s/model.pth' % args.save_dir)
```

### 2.2 原理

-  区别


- -  多进程
    和 DP 不同， DDP 采用多进程，最推荐的做法是每张卡一个进程从而避免上一节所说单进程带来的影响。前文也提到了 DP 和 DDP 共用一个 parallel_apply 函数，所以 DDP 同样支持单进程多线程多卡操作，自然也支持多进程多线程，不过需要注意一下 world_size。

  -  通信效率
     DP 的通信成本随着 GPU 数量线性增长，而 DDP 支持 Ring AllReduce，其通信成本是恒定的，与 GPU 数量无关。

  -  同步参数
     DP 通过收集梯度到 device[0]，在device[0] 更新参数，然后其他设备复制 device[0] 的参数实现各个模型同步；
     DDP 通过保证初始状态相同并且改变量也相同（指同步梯度） ，保证模型同步。


- [Ring AllReduce](https://www.zhihu.com/question/57799212/answer/612786337)

![img](https://pica.zhimg.com/v2-f64fba2a9889ba397157c59cd5e4d4bd_720w.jpg?source=d16d100b)

Fig. 4: Ring AllReduce 流程图，原图见 [7]

假设我们有  kkk  个 GPU，传输总量是 ppp ， bbb 为每次通信上限。

首先我们将要传输的梯度等分成 kkk 份，则每台机器每次需要传输 pk\frac{p}{k}\frac{p}{k} 。传输  k−1k-1k-1  次可以收集到一个完整梯度（如动图 5 所示），之后再传输  k−1k-1k-1  次将梯度分给所有 GPU（如动图 6 所示）。

举个例子，假设现有 5 个 GPU，那么就将梯度分为 5 份，如下图，分别是 ai,bi,ci,di,eia_i, b_i, c_i, d_i, e_ia_i, b_i, c_i, d_i, e_i , 这里的  iii  指的是 GPU 编号。

Scatter Reduce 流程，从 diagonal 的位置开始传，每次传输时 GPU 之间只有一个块在传输，比如 a0a_0a_0 ，在传播 4 次后 GPU 4 上就有了一个完整的梯度块。

![img](https://pica.zhimg.com/v2-4590aeb5fd981b1e6f926cc68605884a_720w.jpg?source=d16d100b)

图 5: Scatter Reduce 流程图，原图见 [7]， gif 见 [8]

All Gather 的过程也类似，只是将收集到的完整梯度通过通信传播给所有参与的 GPU。

![img](https://picx.zhimg.com/v2-c9df34575d7d95ec87d85575d25d6f37_720w.jpg?source=d16d100b)

图 6: All Gather 流程图，原图见 [7]， gif 见 [8]

 这样，通信开销就只有 2(k−1)pkb2(k-1)\frac{\frac{p}{k}}{b}2(k-1)\frac{\frac{p}{k}}{b} ，和 GPU 数量无关了。

- DDP

DDP 也是数据并行，所以每张卡都有模型和输入。我们以多进程多线程为例，每起一个进程，该进程的 device[0] 都会从本地复制模型，如果该进程仍有多线程，就像 DP，模型会从 device[0] 复制到其他设备。

DDP 通过 Reducer 来管理梯度同步。为了提高通讯效率， Reducer 会将梯度归到不同的桶里（按照模型参数的 reverse order， 因为反向传播需要符合这样的顺序），一次归约一个桶。其中桶的大小为参数 bucket_cap_mb 默认为 25，可根据需要调整。下图即为一个例子。

可以看到每个进程里，模型参数都按照倒序放在桶里，每次归约一个桶。

![img](https://picx.zhimg.com/v2-d7b5ba839771e0a0f5cfa8680da0f32f_720w.jpg?source=d16d100b)

图 7: Gradient Bucketing 示意图，原图见 [10]

DDP 通过在构建时注册 autograd hook 进行梯度同步。反向传播时，当一个梯度计算好后，相应的 hook 会告诉 DDP 可以用来归约。当一个桶里的梯度都可以了，Reducer 就会启动异步 allreduce 去计算所有进程的平均值。allreduce 异步启动使得 DDP 可以边计算边通信，提高效率。当所有桶都可以了，Reducer 会等所有 allreduce 完成，然后将得到的梯度写到 param.grad。

### 2.3 实现

DDP 主要基于下图所示结构，本节我们会着重讲解 distributed.py 和 reducer.cpp 两个文件。至于 backend，NCCL 已经最优化了，建议直接用 NCCL，不过 NCCL 只支持 GPU Tensor 间通信。

![img](https://picx.zhimg.com/v2-41108f50195cb81d9c02a0906fbae32e_720w.jpg?source=d16d100b)

图 8: 代码架构，原图见 [10]

 终于可以看 DDP 的实现了！！首先我们贴上伪代码！

- 伪代码

![img](https://picx.zhimg.com/v2-a8b190049000e978acf90d4c9207e4fb_720w.jpg?source=d16d100b)

图 9: DDP 伪代码，原图见 [11]

从 DDP 的伪代码我们可以看出，DDP 最重要的包括三部分：

- *constructor*

- - 负责在构建的时候将 rank 0 的 state_dict() 广播 ➜ 保证所有网络初始状态相同；
  - 初始化 buckets 并尽可能按逆序将 parameters 分配进 buckets ➜ 按桶通信提高效率；
  - 为每个 parameter 加上 grad_accumulator 以及在 autograd_graph 注册 autograd_hook ➜ 在 backward 时负责梯度同步。

- *forward*

- - 正常的 forward 操作；
  - 如果 self.find_unused_parameters 设置为 True，DDP 会在 forward 结束时 traverse autograd graph 找到所有没用过的parameters 并标记为 ready ➜ 虽说这一步开销很大，但是有时计算动态图会改变，所以很必要。

- *autograd_hook*

- - 这个 hook 是挂在 autograd graph 在 backward 时负责梯度同步的。当一个梯度计算好后，相应的 hook 会告诉 DDP 可以用来归约。当一个桶里的梯度都可以了，Reducer 就会启动异步 allreduce 去计算所有进程的平均值。当所有桶都可以了，Reducer 会等所有 allreduce 完成，然后将得到的梯度写到 param.grad。

好的，但现在为止我们应该对 DDP 有了大致了解了，接下来就一起看一下代码是怎么实现的！

-  通信

因为 DDP 依赖 c10d 的 ProcessGroup 进行通信，所以开始前我们先要有个 ProcessGroup 实例。这步可以通过 torch.distributed.init_process_group 实现。

-  构建

我们先贴上 DDP 初始化的源码，最重要的是 _ddp_init_helper 这个函数，负责多线程时复制模型、将 parameters 分组、创建 reducer 以及为 SyncBN 做准备等。这部分代码看 comment 就能懂，我们会重点说一下 dist.Reducer，作为管理器，自然很重要了。

```python
class DistributedDataParallel(Module):
    def __init__(self, module, device_ids=None,
                 output_device=None, dim=0, broadcast_buffers=True,
                 process_group=None,
                 bucket_cap_mb=25,
                 find_unused_parameters=False,
                 check_reduction=False,
                 gradient_as_bucket_view=False):

        super(DistributedDataParallel, self).__init__()

        assert any((p.requires_grad for p in module.parameters())), (
            "DistributedDataParallel is not needed when a module "
            "doesn't have any parameter that requires a gradient."
        )

        self.is_multi_device_module = len({p.device for p in module.parameters()}) > 1
        distinct_device_types = {p.device.type for p in module.parameters()}
        assert len(distinct_device_types) == 1, (
            "DistributedDataParallel's input module must be on "
            "the same type of devices, but input module parameters locate in {}."
        ).format(distinct_device_types)
        self.device_type = list(distinct_device_types)[0]

        if self.device_type == "cpu" or self.is_multi_device_module:
            assert not device_ids and not output_device, (
                "DistributedDataParallel device_ids and output_device arguments "
                "only work with single-device GPU modules, but got "
                "device_ids {}, output_device {}, and module parameters {}."
            ).format(device_ids, output_device, {p.device for p in module.parameters()})

            self.device_ids = None
            self.output_device = None
        else:
            # Use all devices by default for single-device GPU modules
            if device_ids is None:
                device_ids = _get_all_device_indices()

            self.device_ids = list(map(lambda x: _get_device_index(x, True), device_ids))

            if output_device is None:
                output_device = device_ids[0]

            self.output_device = _get_device_index(output_device, True)

        if process_group is None:
            self.process_group = _get_default_group()
        else:
            self.process_group = process_group

        self.dim = dim
        self.module = module
        self.device = list(self.module.parameters())[0].device
        self.broadcast_buffers = broadcast_buffers
        self.find_unused_parameters = find_unused_parameters
        self.require_backward_grad_sync = True
        self.require_forward_param_sync = True
        self.ddp_join_enabled = False
        self.gradient_as_bucket_view = gradient_as_bucket_view

        if check_reduction:
            # This argument is no longer used since the reducer
            # will ensure reduction completes even if some parameters
            # do not receive gradients.
            warnings.warn(
                "The `check_reduction` argument in `DistributedDataParallel` "
                "module is deprecated. Please avoid using it."
            )
            pass

        # used for intra-node param sync and inter-node sync as well
        self.broadcast_bucket_size = int(250 * 1024 * 1024)

        #
        # reduction bucket size
        self.bucket_bytes_cap = int(bucket_cap_mb * 1024 * 1024)

        # 保证初始状态一样
        # Sync params and buffers
        self._sync_params_and_buffers(authoritative_rank=0)

        # 下拉看源码
        self._ddp_init_helper()

    def _sync_params_and_buffers(self, authoritative_rank=0):
        module_states = list(self.module.state_dict().values())
        if len(module_states) > 0:
            self._distributed_broadcast_coalesced(
                module_states,
                self.broadcast_bucket_size,
                authoritative_rank)

    def _ddp_init_helper(self):
        """
        Initialization helper function that does the following:

        (1) replicating the module from device[0] to the other devices （前文提到 DDP 也支持一个进程多线程利用多卡，类似 DP ，这时候就会用到第一步）
        (2) bucketing the parameters for reductions （把 parameter 分组，梯度通讯时，先得到梯度的会通讯）
        (3) resetting the bucketing states
        (4) registering the grad hooks （创建管理器）
        (5) passing a handle of DDP to SyncBatchNorm Layer （为 SyncBN 准备）
        """

        def parameters(m, recurse=True):
            def model_parameters(m):
                ps = m._former_parameters.values() \
                    if hasattr(m, "_former_parameters") \
                    else m.parameters(recurse=False)
                for p in ps:
                    yield p

            for m in m.modules() if recurse else [m]:
                for p in model_parameters(m):
                    yield p

        if self.device_ids and len(self.device_ids) > 1:

            warnings.warn(
                "Single-Process Multi-GPU is not the recommended mode for "
                "DDP. In this mode, each DDP instance operates on multiple "
                "devices and creates multiple module replicas within one "
                "process. The overhead of scatter/gather and GIL contention "
                "in every forward pass can slow down training. "
                "Please consider using one DDP instance per device or per "
                "module replica by explicitly setting device_ids or "
                "CUDA_VISIBLE_DEVICES. "
            )

            # only create replicas for single-device CUDA modules
            #
            # TODO: we don't need to replicate params in here. they're always going to
            # be broadcasted using larger blocks in broadcast_coalesced, so it might be
            # better to not pollute the caches with these small blocks
            self._module_copies = replicate(self.module, self.device_ids, detach=True)
            self._module_copies[0] = self.module

            for module_copy in self._module_copies[1:]:
                for param, copy_param in zip(self.module.parameters(), parameters(module_copy)):
                    # Reducer requires param copies have the same strides across replicas.
                    # Fixes up copy_param strides in case replicate didn't match param strides.
                    if param.layout is torch.strided and param.stride() != copy_param.stride():
                        with torch.no_grad():
                            copy_param.set_(copy_param.clone()
                                                      .as_strided(param.size(), param.stride())
                                                      .copy_(copy_param))
                    copy_param.requires_grad = param.requires_grad

        else:
            self._module_copies = [self.module]

        self.modules_params = [list(parameters(m)) for m in self._module_copies]
        self.modules_buffers = [list(m.buffers()) for m in self._module_copies]

        # Build tuple of (module, parameter) for all parameters that require grads.
        modules_and_parameters = [
            [
                (module, parameter)
                for module in replica.modules()
                for parameter in filter(
                    lambda parameter: parameter.requires_grad,
                    parameters(module, recurse=False))
            ] for replica in self._module_copies]

        # Build list of parameters.
        parameters = [
            list(parameter for _, parameter in replica)
            for replica in modules_and_parameters]

        # Checks if a module will produce a sparse gradient.
        def produces_sparse_gradient(module):
            if isinstance(module, torch.nn.Embedding):
                return module.sparse
            if isinstance(module, torch.nn.EmbeddingBag):
                return module.sparse
            return False

        # Build list of booleans indicating whether or not to expect sparse
        # gradients for the corresponding parameters.
        expect_sparse_gradient = [
            list(produces_sparse_gradient(module) for module, _ in replica)
            for replica in modules_and_parameters]

        # The bucket size limit is specified in the constructor.
        # Additionally, we allow for a single small bucket for parameters
        # that are defined first, such that their gradients don't spill into
        # a much larger bucket, adding unnecessary latency after gradient
        # computation finishes. Experiments showed 1MB is a reasonable value.
        bucket_indices = dist._compute_bucket_assignment_by_size(
            parameters[0],
            [dist._DEFAULT_FIRST_BUCKET_BYTES, self.bucket_bytes_cap],
            expect_sparse_gradient[0])

        # Note: reverse list of buckets because we want to approximate the
        # order in which their gradients are produced, and assume they
        # are used in the forward pass in the order they are defined.
        # 管理器
        self.reducer = dist.Reducer(
            parameters,
            list(reversed(bucket_indices)),
            self.process_group,
            expect_sparse_gradient,
            self.bucket_bytes_cap,
            self.find_unused_parameters,
            self.gradient_as_bucket_view)

        # passing a handle to torch.nn.SyncBatchNorm layer
        self._passing_sync_batchnorm_handle(self._module_copies)
```

每个 DDP 进程都会创建本地 Reducer 在 backward 时管理梯度。

```python
self.reducer = dist.Reducer(
     parameters,
     list(reversed(bucket_indices)),
     self.process_group,
     expect_sparse_gradient,
     self.bucket_bytes_cap,
     self.find_unused_parameters,
     self.gradient_as_bucket_view)
```

我们看 Reducer.cpp 可以发现，构建 Reducer 时，除了各种初始化，最重要的一步就是

```cpp
{
  const auto replica_count = replicas_.size();
  grad_accumulators_.resize(replica_count);
  for (size_t replica_index = 0; replica_index < replica_count;
       replica_index++) {
    const auto variable_count = replicas_[replica_index].size();
    grad_accumulators_[replica_index].resize(variable_count);
    for (size_t variable_index = 0; variable_index < variable_count;
         variable_index++)
    {
      auto& variable = replicas_[replica_index][variable_index];
      const auto index = VariableIndex(replica_index, variable_index);

      // The gradient accumulator function is lazily initialized once.
      // Therefore we can use its presence in the autograd graph as
      // evidence that the parameter has participated in an iteration.
      auto grad_accumulator =
          torch::autograd::impl::grad_accumulator(variable);

#ifndef _WIN32
        using torch::distributed::autograd::ThreadLocalDistAutogradContext;
#endif
        // grad_accumulator 执行完后，autograd_hook 就会运行
        hooks.emplace_back(
            grad_accumulator->add_post_hook(
                torch::make_unique<torch::autograd::utils::LambdaPostHook>(
                    [=](const torch::autograd::variable_list& outputs,
                        const torch::autograd::variable_list& /* unused */){
#ifndef WIN32
                         this->rpc_context.set(
                             ThreadLocalDistAutogradContext::getContextPtr());
#endif
                         this->autograd_hook(index);
                         return outputs;
                       })),
               grad_accumulator);

          // Map raw function pointer to replica index and parameter index.
          // This is used later on when the autograd graph is traversed
          // to check for parameters for which no gradient is computed.
          func_[grad_accumulator.get()] = index;

          // The gradient accumulator is stored as weak_ptr in the autograd
          // metadata of the variable, so we have to keep it alive here for
          // the raw pointer to be valid.
          grad_accumulators_[replica_index][variable_index] =
              std::move(grad_accumulator);
        }
      }
    }

    // std::unordered_map<torch::autograd::Node*, VariableIndex> func_;
    // func_ 存了grad_accumulator & index 的对应，方便我们之后在 autograd graph 寻找 unused parameters

    //  std::vector<std::vector<std::shared_ptr<torch::autograd::Node>>>
    //  grad_accumulators_;
    //  grad_accumulators_ 对应的 index 存了相应的 grad_accumulator

    //   std::vector<std::pair<uintptr_t, std::shared_ptr<torch::autograd::Node>>>
    //   hooks_;
```

其中，发挥重要功能的 autograd_hook 如下：

```cpp
void Reducer::autograd_hook(VariableIndex index) {
     std::lock_guard lock(this->mutex_);
     if (find_unused_parameters_) {
       // 在 no_sync 时，只要参数被用过一次，就会被标记为用过
       // Since it gets here, this param has been used for this iteration. We want
       // to mark it in local_used_maps_. During no_sync session, the same var can
       // be set multiple times, which is OK as does not affect correctness. As
       // long as it is used once during no_sync session, it is marked as used.
       local_used_maps_[index.replica_index][index.variable_index] = 1;
     }

    // Ignore if we don't expect to be called.
    // This may be the case if the user wants to accumulate gradients
    // for number of iterations before reducing them.
    if (!expect_autograd_hooks_) {
      return;
    }

    // Rebuild bucket only if 1) it is the first time to rebuild bucket 2)
    // find_unused_parameters_ is false, currently it does not support when there
    // are unused parameters 3) this backward pass needs to run allreduce. Here,
    // we just dump tensors and their parameter indices into rebuilt_params_ and
    // rebuilt_param_indices_ based on gradient arriving order, and then at the
    // end of finalize_backward(), buckets will be rebuilt based on
    // rebuilt_params_ and rebuilt_param_indices_, and then will be broadcasted
    // and initialized. Also we only need to dump tensors and parameter indices of
    // one replica.
    push_rebuilt_params(index);

    // If `find_unused_parameters_` is true there may be model parameters that
    // went unused when computing the model output, they won't be part of the
    // autograd graph, and won't receive gradients. These parameters are
    // discovered in the `prepare_for_backward` function and their indexes stored
    // in the `unused_parameters_` vector.
    if (!has_marked_unused_parameters_ && find_unused_parameters_) {
      has_marked_unused_parameters_ = true;
      for (const auto& unused_index : unused_parameters_) {
        mark_variable_ready(unused_index);
      }
    }

    // Finally mark variable for which this function was originally called.
    mark_variable_ready(index);
}
```

- 前向传播

```python
def forward(self, inputs, *kwargs):           if self.ddp_join_enabled:               ones = torch.ones(                   1, device=self.device               )               work = dist.all_reduce(ones, group=self.process_group, async_op=True)               self.reducer._set_forward_pass_work_handle(                   work, self.ddp_join_divide_by_initial_world_size               )
# Calling _rebuild_buckets before forward compuation,
      # It may allocate new buckets before deallocating old buckets
      # inside _rebuild_buckets. To save peak memory usage,
      # call _rebuild_buckets before the peak memory usage increases
      # during forward computation.
      # This should be called only once during whole training period.
      if self.reducer._rebuild_buckets():
          logging.info("Reducer buckets have been rebuilt in this iteration.")

      if self.require_forward_param_sync:
          self._sync_params()

      if self.ddp_join_enabled:
          # Notify joined ranks whether they should sync in backwards pass or not.
          self._check_global_requires_backward_grad_sync(is_joined_rank=False)

      # ！！！
      if self.device_ids:
          inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
          if len(self.device_ids) == 1:
              output = self.module(*inputs[0], **kwargs[0])
          else:
            # 单进程多线程多卡的情况
              outputs = self.parallel_apply(self._module_copies[:len(inputs)], inputs, kwargs)
              output = self.gather(outputs, self.output_device)
      else:
          output = self.module(*inputs, **kwargs)

      if torch.is_grad_enabled() and self.require_backward_grad_sync:
          self.require_forward_param_sync = True
          # We'll return the output object verbatim since it is a freeform
          # object. We need to find any tensors in this object, though,
          # because we need to figure out which parameters were used during
          # this forward pass, to ensure we short circuit reduction for any
          # unused parameters. Only if `find_unused_parameters` is set.
          if self.find_unused_parameters:
          # 当DDP参数 find_unused_parameter 为 true 时，其会在 forward 结束时，启动一个回溯，标记出所有没被用到的 parameter，提前把这些设定为 ready，这样 backward 就可以在一个 subgraph 进行，但这样会牺牲一部分时间。
              self.reducer.prepare_for_backward(list(_find_tensors(output)))
          else:
              self.reducer.prepare_for_backward([])
      else:
          self.require_forward_param_sync = False

      return output
```

- 反向传播

那么，DDP 究竟是怎么启动 allreduce 的呢？我们看一下 reducer.cpp 里对桶的定义以及用法，主要是在mark_*_ready。

```cpp
struct Bucket {       std::vector replicas;
// Global indices of participating variables in the bucket
  std::vector<size_t> variable_indices;

  // Number of replicas to be marked done before this bucket is ready.
  // 计数
  size_t pending;

  // Keep work handle around when this set of buckets is being reduced.
  std::shared_ptr<c10d::ProcessGroup::Work> work;

  // Keep future work handle around if DDP comm hook is registered.
  c10::intrusive_ptr<torch::jit::Future> future_work;

  // If this bucket should expect a single sparse gradient.
  // Implies: replicas[i].variables.size() == 1.
  bool expect_sparse_gradient = false;
};
```

先看 mark_variable_ready，截取片段（指去除报错信息）

```cpp
void Reducer::mark_variable_ready(VariableIndex index) {     const auto replica_index = index.replica_index;     const auto variable_index = index.variable_index;     TORCH_CHECK(replica_index < replicas_.size(), "Out of range replica index.");     TORCH_CHECK(         variable_index < variable_locators_.size(),         "Out of range variable index.");     backward_stats_[replica_index][variable_index] =         current_time_in_nanos() - backward_stats_base_;
// 每当变量被标记成 ready 了，都要调用一下 finalize
require_finalize_ = true;

const auto& bucket_index = variable_locators_[variable_index];
auto& bucket = buckets_[bucket_index.bucket_index];
auto& replica = bucket.replicas[replica_index];


// If it was scheduled, wait on allreduce in forward pass that tells us
// division factor based on no. of currently participating processes.
if (divFactor_ == kUnsetDivFactor) {
  divFactor_ = process_group_->getSize();
  auto& workHandle = forwardPassWorkHandle_.workHandle;
  if (workHandle && !forwardPassWorkHandle_.useStaticWorldSize) {
    workHandle->wait();
    auto results = workHandle->result();
    // Guard against the results being empty
    TORCH_INTERNAL_ASSERT(results.size() > 0);
    at::Tensor& res = results.front();
    divFactor_ = res.item().to<int>();
  }
}

if (bucket.expect_sparse_gradient) {
  mark_variable_ready_sparse(index);
} else {
  mark_variable_ready_dense(index);
}

// 检查桶里的变量是不是都ready了，如果没有东西 pending，那就是都 ready了
if (--replica.pending == 0) {
  if (--bucket.pending == 0) {
    mark_bucket_ready(bucket_index.bucket_index);
  }
}

// Run finalizer function and kick off reduction for local_used_maps once the
// final bucket was marked ready.
if (next_bucket_ == buckets_.size()) {
  if (find_unused_parameters_) {
    // H2D from local_used_maps_ to local_used_maps_dev_
    for (size_t i = 0; i < local_used_maps_.size(); i++) {
      // We do async H2D to avoid the blocking overhead. The async copy and
      // allreduce respect the current stream, so will be sequenced correctly.
      local_used_maps_dev_[i].copy_(local_used_maps_[i], true);
    }
    local_used_work_ = process_group_->allreduce(local_used_maps_dev_);
  }

  // The autograd engine uses the default stream when running callbacks, so we
  // pass in the current CUDA stream in case it is not the default.
  c10::DeviceType deviceType = replica.contents.device().type();
  const c10::impl::VirtualGuardImpl guard =
      c10::impl::VirtualGuardImpl{deviceType};
  const c10::Stream currentStream =
      guard.getStream(replica.contents.device());
  torch::autograd::Engine::get_default_engine().queue_callback([=] {
    std::lock_guard<std::mutex> lock(this->mutex_);
    // Run callback with the current stream
    c10::OptionalStreamGuard currentStreamGuard{currentStream};
    this->finalize_backward();
  });
}
}
```



```cpp
void Reducer::mark_bucket_ready(size_t bucket_index) {     TORCH_INTERNAL_ASSERT(bucket_index >= next_bucket_);
// Buckets are reduced in sequence. Ignore this bucket if
// it's not its turn to be reduced.
if (bucket_index > next_bucket_) {
  return;
}

// Keep going, until we either:
// - 所有桶都在 allreduce 那就等着 or
// - 还有桶没好，那也等着.
for (; next_bucket_ < buckets_.size() && buckets_[next_bucket_].pending == 0;
     next_bucket_++) {
  auto& bucket = buckets_[next_bucket_];
  std::vector<at::Tensor> tensors;
  tensors.reserve(bucket.replicas.size());
  for (const auto& replica : bucket.replicas) {

        // CUDA default stream 都按时序排好了
    tensors.push_back(replica.contents);
  }
  if (comm_hook_ == nullptr) {
    // 如果没注册 comm_hook，直接 allreduce
    bucket.work = process_group_->allreduce(tensors);
  } else {
    // 注册了 comm_hook 那就先跑 hook
    // 需要注意的是，这个comm_hook 只是处理通信的底层hook，如果想在 reduce 前分别进行梯度裁剪，还是需要在 autograph 挂 hook
    bucket.future_work = comm_hook_->runHook(GradBucket(tensors));
  }
}
}
```

除了正常的前向传播，DDP 还允许在 subgraph 进行反向传播，只需将 self.find_unused_parameters 设置为 True。或许有朋友会问，如果 find_unused_parameters 设置为 True，那每次都要 traverse 计算图，明明开销很大，为什么有时候我们还要将 self.find_unused_parameters 设置为 True？ 这是因为训练时有可能某次迭代只用到整个模型的一个 subgraph， 并且这个 subgraph 迭代时可能会改变，就是说某些参数可能会在训练时被跳过。但因为所有parameters 在一开始就被分好桶了，而我们的 hook 又规定了只有整个桶 ready 了（pending==0）才会通信，如果我们不将 unused parameter 标记为 ready，整个过程会没法进行。我们在这节结束的部分附上一个小实验验证一下。

DDP 通过在构建时注册 autograd hook 进行梯度同步。当一个梯度计算好后，相应的 hook 会告诉 DDP 可以用来归约。当一个桶里的梯度都可以了，Reducer 就会启动异步 allreduce 去计算所有进程的平均值。当所有桶都可以了，Reducer 会等所有 allreduce 完成，然后将得到的梯度写到 param.grad。

-  optimizer step   独立于 DDP，所有进程的模型能够同步是因为初始状态相同并且改变量也相同。

-  no_sync


```python
@contextmanager
def no_sync(self):
        old_require_backward_grad_sync = self.require_backward_grad_sync
        self.require_backward_grad_sync = False
        try:
            yield
        finally:
            self.require_backward_grad_sync = old_require_backward_grad_sync

>>> ddp = torch.nn.DistributedDataParallel(model, pg)
>>> with ddp.no_sync():
>>>   for input in inputs:
>>>     ddp(input).backward()  # 不同步梯度
>>> ddp(another_input).backward()  # 同步梯度
```

迭代多次再同步梯度。

- 实验：find_unused_params

```python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP
from timeit import default_timer as timer

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12138'
# sync
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def example(rank, world_size):
    # create default process group
    dist.init_process_group("gloo",rank=rank,
world_size=world_size,init_method='env://')
    # create local model
    model = nn.Linear(10, 10).to(rank)
    # construct DDP model
    ddp_model = DDP(model, device_ids=[rank])
    # define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    buf = 0
    tmp = 0
    for i in range(10000):
        start = timer()
        # forward pass
        outputs = ddp_model(torch.randn(20, 10).to(rank))
        end = timer()

        tmp = end-start
        buf+=tmp
        labels = torch.randn(20, 10).to(rank)
        # backward pass
        loss_fn(outputs, labels).backward()
        # update parameters
        optimizer.step()
    print(tmp)
    print(buf)
    print(buf/10000)

def main():
    world_size = 1
    mp.spawn(example,
        args=(world_size,),
        nprocs=world_size,
        join=True)

if __name__=="__main__":
   for i in range(10):
     main()
```

将 find_unused_params 分别设置成 True 或者 False 跑多次取平均，可以得到：

- find_unused_params=True: 0.3367 ms
- find_unused_params=False: 0.2993 ms

### 小结

关于 DP 和 DDP 的分享到这里就结束啦～
