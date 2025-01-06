# PyTorch 分布式与并行训练

PyTorch 的 `torch.distributed` 包底层主要基于 Collective Communication (c10d) library 实现跨进程的张量通信，支持两种主要的通信 API：

- Collective Communication APIs (集合通信)
  - 用于 Distributed Data-Parallel Training (DDP，分布式数据并行训练)
- P2P Communication APIs (点对点通信)
  - 用于 RPC-Based Distributed Training (基于 RPC 的分布式训练)

这两种通信 API 分别对应了 PyTorch 中的两种分布式训练范式。本教程将重点介绍 Distributed Data-Parallel Training (DDP) 的通信机制和 API 使用，包括如何配置分布式环境、使用不同的通信策略，以及深入理解其内部工作原理。

## 分布式训练模板

PyTorch 的分布式包 (`torch.distributed`) 使研究人员和开发者能够轻松地将计算任务并行化到多个进程和机器集群中。它通过消息传递语义实现，允许每个进程与其他任意进程进行数据交换。与仅限于单机多进程的 `torch.multiprocessing` 包不同，分布式包支持跨机器通信，并且可以使用不同的通信后端。

下面是一个基础的分布式训练模板，用于在单机上启动多个训练进程：

```python
"""run.py:"""
#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def run(rank, size):
    """ 分布式训练函数，具体实现将在后面介绍 """
    pass

def init_process(rank, size, fn, backend='gloo'):
    """ 初始化分布式环境 """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

if __name__ == "__main__":
    world_size = 2  # 总进程数
    processes = []
    mp.set_start_method("spawn")
    for rank in range(world_size):
        p = mp.Process(target=init_process, args=(rank, world_size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
```

这个模板会启动两个进程，每个进程都会:
1. 配置分布式环境
2. 初始化进程组 (`dist.init_process_group`)
3. 执行指定的训练函数 `run`

## torch.distributed 核心概念

### 基本定义

首先来看 torch.distributed 的官方定义：

- `torch.distributed` 包为 PyTorch 提供了**多进程并行通信原语**，支持在单机或多机上的多个计算节点之间进行通信，能够轻松实现跨进程和机器集群的计算并行化。

- `torch.nn.parallel.DistributedDataParallel (DDP)` 是基于该功能实现的高层封装，为任何 PyTorch 模型提供同步分布式训练功能。

需要注意的是，torch.distributed 的核心功能是实现**进程级别**的通信（而非线程级），以支持多机多卡分布式训练。这与 `torch.multiprocessing` 包和 `torch.nn.DataParallel()` 提供的并行计算方式有本质区别，因为它支持网络连接的多机训练，且需要为每个进程显式启动训练脚本的副本。

### DataParallel 与 DistributedDataParallel 的对比

在深入学习之前，让我们先明确为什么要使用 `DistributedDataParallel` 而不是 `DataParallel`，尽管前者的使用相对更复杂：

| 特性         | DataParallel                       | DistributedDataParallel              |
| ------------ | ---------------------------------- | ------------------------------------ |
| 进程模型     | 单进程多线程                       | 多进程                               |
| 适用范围     | 仅支持单机                         | 支持单机和多机                       |
| 性能表现     | 较慢 - 受GIL限制，存在线程切换开销 | 更快 - 无GIL限制，进程间通信开销更小 |
| Python解释器 | 所有GPU共享一个解释器              | 每个进程有独立的解释器               |
| 优化器       | 单个优化器                         | 每个进程维护独立优化器               |
| 模型并行     | 不支持                             | 支持与模型并行结合使用               |

主要优势：

1. **更好的性能**: DistributedDataParallel 采用多进程方式，每个进程都有独立的 Python 解释器，避免了 GIL 带来的性能瓶颈。这对于计算密集的模型尤其重要。

2. **更低的通信开销**: 每个进程维护自己的优化器，在每次迭代中执行完整的优化步骤。虽然梯度会在进程间同步平均，但不需要额外的参数广播步骤。

3. **更好的扩展性**: 支持与模型并行结合使用，可以处理单GPU无法容纳的大模型。在这种情况下，每个 DDP 进程内部使用模型并行，进程之间使用数据并行。


## torch.distributed 通信后端

PyTorch 分布式包的一大优势在于其抽象设计,支持多种通信后端。目前主要实现了 Gloo、NCCL 和 MPI 三种后端,它们各有特点和适用场景。下表展示了各后端支持的功能对比:

| 后端           | `gloo` | `gloo` | `mpi` | `mpi` | `nccl` | `nccl` |
| -------------- | :----: | :----: | :---: | :---: | :----: | :----: |
| 设备           |  CPU   |  GPU   |  CPU  |  GPU  |  CPU   |  GPU   |
| send/recv      |   ✓    |   ✘    |   ✓   |   ?   |   ✘    |   ✓    |
| broadcast      |   ✓    |   ✓    |   ✓   |   ?   |   ✘    |   ✓    |
| all_reduce     |   ✓    |   ✓    |   ✓   |   ?   |   ✘    |   ✓    |
| reduce         |   ✓    |   ✘    |   ✓   |   ?   |   ✘    |   ✓    |
| all_gather     |   ✓    |   ✘    |   ✓   |   ?   |   ✘    |   ✓    |
| gather/scatter |   ✓    |   ✘    |   ✓   |   ?   |   ✘    |   ✓    |
| reduce_scatter |   ✘    |   ✘    |   ✘   |   ✘   |   ✘    |   ✓    |
| all_to_all     |   ✘    |   ✘    |   ✓   |   ?   |   ✘    |   ✓    |
| barrier        |   ✓    |   ✘    |   ✓   |   ?   |   ✘    |   ✓    |

在 PyTorch 的分布式版本中,Linux 系统默认包含了 Gloo 和 NCCL 后端(NCCL 仅在 CUDA 版本中提供)。MPI 是一个可选后端,需要在编译 PyTorch 时手动启用。

### Gloo 后端

[Gloo](https://github.com/facebookincubator/gloo) 后端支持:
- CPU 上的所有点对点和集合通信操作
- GPU 上的集合通信操作(但性能不如 NCCL)

### MPI 后端

消息传递接口(MPI)是高性能计算领域的标准化工具,是 `torch.distributed` API 设计的主要参考。主流 MPI 实现包括:
- [Open-MPI](https://www.open-mpi.org/)
- [MVAPICH2](http://mvapich.cse.ohio-state.edu/)
- [Intel MPI](https://software.intel.com/en-us/intel-mpi-library)

MPI 的优势在于:
- 广泛的可用性和成熟的生态
- 在大型计算集群上经过高度优化
- 部分实现支持 CUDA IPC 和 GPU Direct 技术,可避免 CPU 内存拷贝

要使用 MPI 后端,需要手动重新编译 PyTorch。具体步骤如下:

1. 创建并激活 Anaconda 环境,按照[指南](https://github.com/pytorch/pytorch#from-source)安装依赖
2. 安装 MPI 实现(例如 Open-MPI): `conda install -c conda-forge openmpi`
3. 在 PyTorch 源码目录执行 `python setup.py install`

使用 MPI 后端时,启动脚本需要做如下修改:

1. 将 `if __name__ == '__main__':` 下的内容替换为:
```python
init_process(0, 0, run, backend='mpi')
```

2. 使用 mpirun 启动:
```bash
mpirun -n 4 python myscript.py
```

这是因为 MPI 需要在生成进程前创建自己的环境,并且会自行处理进程生成,使得 `rank` 和 `size` 参数变得多余。MPI 的这种方式非常灵活,可以通过 mpirun 的参数来精细控制每个进程的计算资源分配。

### NCCL 后端

[NCCL](https://github.com/nvidia/nccl) 后端为 CUDA 张量提供了优化的集合通信实现。如果你只需要在 GPU 上进行集合通信操作,NCCL 后端可以提供最佳性能。NCCL 后端已包含在支持 CUDA 的 PyTorch 预编译版本中。

### 如何选择后端?

以下是选择通信后端的建议:

- GPU 训练
  - 首选 NCCL 后端
  - 如果遇到问题可以回退到 Gloo

- CPU 训练
  - 使用 Gloo 后端
  - 特殊情况下可以考虑 MPI

- 具体场景
  - InfiniBand 网络 + GPU: 使用 NCCL(唯一支持 InfiniBand 和 GPUDirect)
  - InfiniBand 网络 + CPU: 如果启用了 IP over IB 则使用 Gloo,否则使用 MPI
  - 以太网 + GPU: 优先使用 NCCL,备选 Gloo
  - 以太网 + CPU: 使用 Gloo,除非有特殊原因需要 MPI

## 初始化分布式环境

在使用分布式功能之前,需要通过以下两个函数之一来初始化环境:
- [`torch.distributed.init_process_group()`](https://pytorch.org/docs/main/distributed.html#torch.distributed.init_process_group)
- [`torch.distributed.device_mesh.init_device_mesh()`](https://pytorch.org/docs/main/distributed.html#torch.distributed.device_mesh.init_device_mesh)

这两个函数都会阻塞执行,直到所有进程都加入到进程组中。

> **重要提示**: 初始化操作不是线程安全的。进程组的创建必须在单个线程中执行,以防止:
> - 不同 rank 上分配不一致的 UUID
> - 初始化过程中的竞争条件导致进程挂起

### 关键初始化函数

#### `distributed.is_available()`
- 检查分布式包是否可用
- 返回 `True` 表示可以使用分布式功能,否则分布式相关 API 将不可访问

#### `distributed.init_process_group`
- 初始化默认的分布式进程组
- 支持两种主要的初始化方式:
  - 显式指定 `store`、`rank` 和 `world_size`
  - 通过 `init_method` (URL字符串)指定如何发现其他节点

- 如果未指定 `init_method`,默认使用 `"env://"`

主要参数:

- **backend** (`str` 或 `Backend`, 可选):
  - 指定要使用的通信后端
  - 有效值包括 `mpi`、`gloo`、`nccl`
  - 如果不指定,将创建 `gloo` 和 `nccl` 后端

- **init_method** (`str`, 可选):
  - 指定如何初始化进程组的 URL
  - 默认为 "env://"
  - 与 `store` 参数互斥

- **world_size** (`int`, 可选):
  - 参与训练的总进程数
  - 使用 `store` 时必须指定

- **rank** (`int`, 可选):
  - 当前进程的 rank (取值范围 0 到 `world_size`-1)
  - 使用 `store` 时必须指定

- **store** (`Store`, 可选):
  - 用于交换连接信息的键值存储
  - 所有进程都必须能访问
  - 与 `init_method` 互斥

- **timeout** (`timedelta`, 可选):
  - 进程组操作的超时时间
  - NCCL 默认 10 分钟,其他后端默认 30 分钟
  - 超时后会取消集合操作并使进程崩溃
  - 这是必要的,因为 CUDA 执行是异步的,继续执行可能导致后续操作使用损坏的数据

#### `distributed.is_initialized()`
- 检查默认进程组是否已初始化
- 返回 `bool` 类型

#### `distributed.is_torchelastic_launched()`
- 检查当前进程是否由 `torch.distributed.elastic` (又称 torchelastic) 启动
- 通过检查环境变量 `TORCHELASTIC_RUN_ID` 是否存在来判断
- 返回 `bool` 类型

### 初始化方法

PyTorch 分布式支持三种初始化方法,用于协调进程间的初始通信。选择合适的初始化方法取决于具体的硬件环境和使用场景。

#### 1. 环境变量初始化 (默认方法)

通过设置以下环境变量来实现进程间的协调:

```bash
# 主节点地址
export MASTER_ADDR='10.1.1.20'
# 主节点端口
export MASTER_PORT='29500'
# 总进程数
export WORLD_SIZE='4'
# 当前进程的 rank
export RANK='0'
```

使用时无需指定 `init_method` (或指定为 `env://`):

```python
import torch.distributed as dist
dist.init_process_group(backend)  # 使用默认的环境变量方式初始化

def ddp_setup(rank: int, world_size: int) -> None:
    """Set up the distributed environment for training.

    Args:
        rank (int): Unique identifier of each process.
        world_size (int): Total number of processes.
    """
    # Set environment variables for process communication
    os.environ['MASTER_ADDR'] = '10.1.1.20'
    os.environ['MASTER_PORT'] = '29500'

    # Set the current device for the process
    torch.cuda.set_device(rank)

    # Initialize the process group
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
```


#### 2. TCP 初始化

通过指定 rank 0 进程的 IP 地址和可用端口来实现初始化：

```python
import torch.distributed as dist

# 使用其中一台机器的地址
dist.init_process_group(
    backend,
    init_method='tcp://10.1.1.20:23456',
    rank=args.rank,
    world_size=4
)
```

#### 3. 共享文件系统初始化

利用所有进程都可访问的共享文件系统来协调通信：

```python
import torch.distributed as dist

dist.init_process_group(
    backend,
    init_method='file:///path/to/shared/storage/directory',
    world_size=4,
    rank=args.rank
)
```

> **重要提示**：
> - 文件系统初始化会自动创建一个新文件（如果不存在）
> - 每次初始化都需要一个新的空文件
> - 使用完毕后需要清理该文件，避免影响下次初始化
> - 如果文件未被正确清理，后续的初始化可能会失败
> - 建议确保每次调用 `init_process_group()` 时使用的文件都是不存在或为空的



### 完成初始化后

一旦运行了 [`torch.distributed.init_process_group()`](https://pytorch.org/docs/main/distributed.html#torch.distributed.init_process_group)，以下函数就可以使用了。要检查进程组是否已经初始化，请使用 [`torch.distributed.is_initialized()`](https://pytorch.org/docs/main/distributed.html#torch.distributed.is_initialized)。

- `torch.distributed.Backend(*name*)`
- `torch.distributed.get_backend(*group=None*)`
- `torch.distributed.get_rank(*group=None*)`
- `torch.distributed.get_world_size(*group=None*)`

### Shutdown

在退出时清理资源很重要，通过调用 `destroy_process_group()` 来实现。

最简单的模式是在训练脚本中不再需要通信的地方（通常在主函数的末尾附近），对每个训练器进程调用 `destroy_process_group()`，而不是在外部进程启动器级别。

如果 `destroy_process_group()` 没有被所有等级在超时持续时间内调用，尤其是在应用程序中有多个进程组时（例如，用于 N-D 并行），退出时可能会出现挂起。这是因为 ProcessGroupNCCL 的析构函数调用 `ncclCommAbort`，这必须是集合运算调用的，但 Python 的 GC 调用 ProcessGroupNCCL 的析构函数的顺序是不确定的。调用 `destroy_process_group()` 有助于确保 `ncclCommAbort` 以一致的顺序在所有等级上调用，并避免在 ProcessGroupNCCL 的析构函数期间调用 `ncclCommAbort`。

## 分布式通信原语

初始化完成后，可以使用以下通信原语进行进程间的数据交换。

### 点对点通信

<img src="https://pytorch.org/tutorials/_images/send_recv.png" alt="Send and Recv" style="zoom:50%;" />

点对点通信允许在两个指定的进程之间直接传输数据。PyTorch 提供了同步和异步两种实现：

#### 同步通信

- [`send()`](https://pytorch.org/docs/main/distributed.html#torch.distributed.send)
- [`recv()`](https://pytorch.org/docs/main/distributed.html#torch.distributed.recv)

```python
def run(rank, size):
    tensor = torch.zeros(1)
    if rank == 0:
        tensor += 1
        # 发送数据到进程 1
        dist.send(tensor=tensor, dst=1)
    else:
        # 从进程 0 接收数据
        dist.recv(tensor=tensor, src=0)
    print(f'Rank {rank} has data {tensor[0]}')
```

在上面的示例中，两个进程都从 tensor(0) 开始，然后进程 0 递增张量并将其发送到进程 1，以便它们都以 tensor(1) 结尾。 请注意，接收方需要预先分配足够的内存来存储接收的数据。这种方式会阻塞进程直到通信完成。

#### 异步通信
- [`isend()`](https://pytorch.org/docs/main/distributed.html#torch.distributed.isend)
- [`irecv()`](https://pytorch.org/docs/main/distributed.html#torch.distributed.irecv)

```python
def run(rank, size):
    tensor = torch.zeros(1)
    req = None
    if rank == 0:
        tensor += 1
        # 异步发送数据到进程 1
        req = dist.isend(tensor=tensor, dst=1)
        print('Rank 0 started sending')
    else:
        # 异步接收来自进程 0 的数据
        req = dist.irecv(tensor=tensor, src=0)
        print('Rank 1 started receiving')

    # 等待通信完成
    req.wait()
    print(f'Rank {rank} has data {tensor[0]}')
```

异步通信会立即返回一个 `Request` 对象，支持两个方法：
- `is_completed()`: 检查操作是否完成
- `wait()`: 阻塞等待操作完成

> **注意事项**：
> - 在 `wait()` 调用完成之前，不要修改发送的张量或访问接收的张量
> - `isend()` 之后修改发送张量会导致未定义行为
> - `irecv()` 之后读取接收张量会导致未定义行为
> - 只有在 `wait()` 返回后，才能保证通信已完成

### 集合通信

集合通信允许在进程组内的所有进程之间进行通信。PyTorch 提供了多种集合通信原语：

#### 1. broadcast
```python
dist.broadcast(tensor, src, group=None)
```
- 将 `tensor` 从源进程广播到组内所有进程
- 所有进程必须提供相同大小的张量

<img src="https://pytorch.org/tutorials/_images/broadcast.png" alt="播送" style="zoom:50%;" />

#### 2. reduce
```python
dist.reduce(tensor, dst, op=ReduceOp.SUM, group=None)
```
- 将所有进程的 `tensor` 按指定操作（默认求和）归约到目标进程
- 支持的归约操作包括：`SUM`、`PRODUCT`、`MIN`、`MAX`、`BAND`(按位与)、`BOR`(按位或)、`BXOR`(按位异或)
- 结果只在目标进程（`dst`）上有效

<img src="https://pytorch.org/tutorials/_images/reduce.png" alt="减少" style="zoom:50%;" />

#### 3. all_reduce
```python
dist.all_reduce(tensor, op=ReduceOp.SUM, group=None)
```
- 类似 `reduce`，但结果会广播到所有进程
- 常用于分布式训练中的梯度同步
- 所有进程在操作完成后都能得到相同的结果

<img src="https://pytorch.org/tutorials/_images/all_reduce.png" alt="全归约" style="zoom:50%;" />

#### 4. gather
```python
dist.gather(tensor, gather_list=None, dst=0, group=None)
```
- 收集所有进程的 `tensor` 到目标进程的 `gather_list` 中
- 非目标进程的 `gather_list` 必须为 None
- 目标进程的 `gather_list` 必须是长度为进程数的列表

<img src="https://pytorch.org/tutorials/_images/gather.png" alt="收集" style="zoom:50%;" />

#### 5. all_gather

```python
dist.all_gather(tensor_list, tensor, group=None)
```
- 类似 `gather`，但收集的结果会发送到所有进程
- 每个进程都需要提供一个长度为进程数的 `tensor_list`
- 操作完成后，所有进程的 `tensor_list` 都包含相同的数据

<img src="https://pytorch.org/tutorials/_images/all_gather.png" alt="全员聚集" style="zoom:50%;" />

#### 6. scatter
```python
dist.scatter(tensor, scatter_list=None, src=0, group=None)
```
- 将源进程的 `scatter_list` 分发到所有进程的 `tensor` 中
- 源进程的 `scatter_list` 必须是长度为进程数的列表
- 非源进程的 `scatter_list` 必须为 None

<img src="https://pytorch.org/tutorials/_images/scatter.png" alt="Scatter" style="zoom:50%;" />

#### 7. reduce_scatter

```python
dist.reduce_scatter(output, input_list, op=ReduceOp.SUM, group=None)
```
- 结合了 reduce 和 scatter 操作
- 首先对 `input_list` 中对应位置的张量执行 reduce 操作
- 然后将结果分散到各个进程的 `output` 中

![image](https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.5/_images/reducescatter.png)

#### 8. all_to_all

```python
dist.all_to_all(output_tensor_list, input_tensor_list, group=None)
```
- 每个进程将数据分成等份发送给所有进程（包括自己）
- 同时从所有进程接收相应的数据
- 要求 `input_tensor_list` 和 `output_tensor_list` 的长度都等于进程数

#### 9. barrier
```python
dist.barrier(group=None)
```
- 同步所有进程，确保所有进程都达到这个点后才继续执行
- 用于确保进程间的同步，特别是在需要所有进程完成某些操作后才能继续的场景

> **性能提示**：
>
> - 集合通信操作通常比点对点通信更高效
> - 异步操作（以 'i' 开头的函数）可以帮助掩盖通信延迟
> - 在 GPU 上使用 NCCL 后端可以获得最佳性能
> - 避免频繁的小数据通信，尽可能批量处理
> - 考虑使用压缩技术减少通信数据量

这些集合通信原语构成了分布式训练的基础。在实际应用中，通常不需要直接使用这些低级 API，而是使用更高级的抽象如 DistributedDataParallel (DDP)。但理解这些基本操作对于调试和优化分布式训练非常重要。

### Launch utility

`torch.distributed` 包还提供了一个启动工具 `torch.distributed.launch`。这个辅助工具可以用于在每个训练节点上启动多个进程进行分布式训练。

`torch.distributed.launch` 是一个模块，它会在每个训练节点上生成多个分布式训练进程。

> 警告
>
> 该模块将被弃用，取而代之的是 `torchrun`。

该工具可用于单节点分布式训练，其中每个节点将生成一个或多个进程。该工具可用于 CPU 训练或 GPU 训练。如果该工具用于 GPU 训练，每个分布式进程将在单个 GPU 上运行。这可以显著提高单节点训练性能。它还可以用于多节点分布式训练，通过在每个节点上生成多个进程来提高多节点分布式训练性能。这对于具有多个支持直接 GPU 的 Infiniband 接口的系统尤其有益，因为所有这些接口都可以用于聚合通信带宽。

在单节点分布式训练或多节点分布式训练的情况下，该工具将启动每个节点的给定数量的进程（`--nproc-per-node`）。如果用于 GPU 训练，这个数量需要小于或等于当前系统上的 GPU 数量（`nproc_per_node`），并且每个进程将在从 GPU 0 到 GPU (`nproc_per_node - 1`) 的单个 GPU 上运行。

如何使用该模块：

单节点多进程分布式训练

```bash
python -m torch.distributed.launch --nproc-per-node=NUM_GPUS_YOU_HAVE
           YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3 以及你的训练脚本的所有其他参数)
```

多节点多进程分布式训练（例如，两个节点）

节点 1：（IP: 192.168.1.1，有一个空闲端口：1234）

```bash
python -m torch.distributed.launch --nproc-per-node=NUM_GPUS_YOU_HAVE
           --nnodes=2 --node-rank=0 --master-addr="192.168.1.1"
           --master-port=1234 YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3
           以及你的训练脚本的所有其他参数)
```

节点 2：

```bash
python -m torch.distributed.launch --nproc-per-node=NUM_GPUS_YOU_HAVE
           --nnodes=2 --node-rank=1 --master-addr="192.168.1.1"
           --master-port=1234 YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3
           以及你的训练脚本的所有其他参数)
```

要查看该模块提供的可选参数：

```bash
python -m torch.distributed.launch --help
```

重要提示：

1. 该工具和多进程分布式（单节点或多节点）GPU 训练目前仅在使用 NCCL 分布式后端时才能达到最佳性能。因此，NCCL 后端是用于 GPU 训练的推荐后端。

2. 在你的训练程序中，你必须解析命令行参数 `--local-rank=LOCAL_PROCESS_RANK`，该参数将由该模块提供。如果你的训练程序使用 GPU，你应该确保你的代码仅在 `LOCAL_PROCESS_RANK` 对应的 GPU 设备上运行。可以通过以下方式实现：

解析 `local_rank` 参数

```python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--local-rank", "--local_rank", type=int)
args = parser.parse_args()
```

将设备设置为本地排名，使用以下任一方式：

```python
torch.cuda.set_device(args.local_rank)  # 在你的代码运行之前
```

或

```python
with torch.cuda.device(args.local_rank):
    # 你的代码运行
    ...
```

在版本 2.0.0 中更改：启动器会将 `--local-rank=<rank>` 参数传递给你的脚本。从 PyTorch 2.0.0 开始，破折号 `--local-rank` 是首选，而不是之前使用的下划线 `--local_rank`。

为了向后兼容，用户可能需要在参数解析代码中处理这两种情况。这意味着在参数解析器中同时包含 `--local-rank` 和 `--local_rank`。如果只提供 `--local-rank`，启动器将触发错误：“error: unrecognized arguments: –local-rank=<rank>”。对于仅支持 PyTorch 2.0.0+ 的训练代码，包含 `--local-rank` 应该足够。

3. 在你的训练程序中，你应该在开始时调用以下函数来启动分布式后端。强烈建议使用 `init_method=env://`。其他初始化方法（例如 `tcp://`）可能有效，但 `env://` 是该模块官方支持的方法。

```python
torch.distributed.init_process_group(backend='YOUR BACKEND',
                                     init_method='env://')
```

4. 在你的训练程序中，你可以使用常规的分布式函数或使用 `torch.nn.parallel.DistributedDataParallel()` 模块。如果你的训练程序使用 GPU 进行训练并希望使用 `torch.nn.parallel.DistributedDataParallel()` 模块，可以按如下方式配置：

```python
model = torch.nn.parallel.DistributedDataParallel(model,
                                                  device_ids=[args.local_rank],
                                                  output_device=args.local_rank)
```

请确保 `device_ids` 参数设置为你的代码将运行的唯一 GPU 设备 ID。这通常是进程的本地排名。换句话说，`device_ids` 需要是 `[args.local_rank]`，`output_device` 需要是 `args.local_rank`，以便使用该工具。

5. 另一种通过环境变量 `LOCAL_RANK` 将 `local_rank` 传递给子进程的方式。当你使用 `--use-env=True` 启动脚本时，此行为将被启用。你必须调整上述子进程示例，将 `args.local_rank` 替换为 `os.environ['LOCAL_RANK']`；当你指定此标志时，启动器不会传递 `--local-rank`。

> 警告
>
> `local_rank` 不是全局唯一的：它在机器上的每个进程中是唯一的。因此，不要使用它来决定是否应该执行某些操作，例如写入网络文件系统。参见 https://github.com/pytorch/pytorch/issues/12042 了解如果不正确处理可能会出现的问题。

### [Spawn utility](https://pytorch.org/docs/stable/distributed.html#spawn-utility)

`torch.multiprocessing` 包还提供了一个生成函数 `torch.multiprocessing.spawn()`。这个辅助函数可以用于生成多个进程。它通过传入你想要运行的函数并生成 N 个进程来运行它。这也可以用于多进程分布式训练。有关如何使用它的参考，请参阅 [PyTorch 示例 - ImageNet 实现](https://github.com/pytorch/examples/tree/master/imagenet)。



## PyTorch 分布式训练启动工具

### torchrun (推荐)

torchrun 是 PyTorch 推荐的分布式训练启动工具,用于替代已弃用的 `torch.distributed.launch`。

### 基本用法

单节点多 GPU 训练:

```bash
torchrun --nproc_per_node=GPU数量 train.py [训练脚本参数]
```

多节点分布式训练:

```bash
# 节点1
torchrun --nproc_per_node=GPU数量 \
         --nnodes=2 \
         --node_rank=0 \
         --master_addr="192.168.1.1" \
         --master_port=29500 \
         train.py [训练脚本参数]

# 节点2
torchrun --nproc_per_node=GPU数量 \
         --nnodes=2 \
         --node_rank=1 \
         --master_addr="192.168.1.1" \
         --master_port=29500 \
         train.py [训练脚本参数]
```

### 训练脚本配置

```python
import torch.distributed as dist

# 1. 初始化进程组
dist.init_process_group(backend='nccl', init_method='env://')

# 2. 设置当前设备
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

# 3. 创建DistributedDataParallel模型
model = DistributedDataParallel(model,
                               device_ids=[local_rank],
                               output_device=local_rank)
```

### 最佳实践

1. 推荐使用 NCCL 后端以获得最佳 GPU 训练性能
2. 使用 env:// 初始化方法
3. 正确设置 local_rank 确保每个进程使用独立 GPU
4. 在所有进程结束时调用 destroy_process_group()

### 替代方案

对于更灵活的进程管理,可以使用 torch.multiprocessing.spawn(), 有关它的使用参考，请参阅 [PyTorch 示例 - ImageNet 实现](https://github.com/pytorch/examples/tree/master/imagenet)。

```python
import torch.multiprocessing as mp

def train(rank, world_size):
    # 训练代码
    pass

mp.spawn(train,
         args=(world_size,),
         nprocs=world_size)
```


## Reference:


1. [PyTorch Distributed Training](https://pytorch.org/docs/stable/distributed.html)
2. [PyTorch Distributed Training Tutorial](https://pytorch.org/tutorials/intermediate/dist_tuto.html)
