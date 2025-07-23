# 入门指南

## 安装

- 安装非常简单，只需运行`pip install deepspeed`，[查看更多细节](https://www.deepspeed.ai/tutorials/advanced-install/)。
- 如果想在AzureML上使用DeepSpeed，请查看[AzureML Examples GitHub](https://github.com/Azure/azureml-examples/tree/main/cli/jobs/deepspeed)。
- DeepSpeed与[HuggingFace Transformers](https://github.com/huggingface/transformers)和[PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)有直接集成。HuggingFace Transformers用户现在可以通过简单的`--deepspeed`标志和配置文件来加速模型[查看更多细节](https://huggingface.co/docs/transformers/deepspeed)。PyTorch Lightning通过Lightning Trainer轻松访问DeepSpeed[查看更多细节](https://pytorch-lightning.readthedocs.io/en/stable/advanced/multi_gpu.html?highlight=deepspeed#deepspeed)。
- 在AMD上使用DeepSpeed可以通过我们的[ROCm镜像](https://hub.docker.com/r/deepspeed/rocm501/tags)，例如：`docker pull deepspeed/rocm501:ds060_pytorch110`。
- DeepSpeed还支持Intel Xeon CPU、Intel Data Center Max Series XPU、Intel Gaudi HPU、Huawei Ascend NPU等，请参阅[加速器设置指南](https://www.deepspeed.ai/tutorials/accelerator-setup-guide/)。

## 编写DeepSpeed模型

DeepSpeed模型训练是通过DeepSpeed引擎完成的。该引擎可以包装任意类型的`torch.nn.module`模型，并且有一组最小的API用于训练和检查点保存模型。请查看教程以获取详细示例。

要初始化DeepSpeed引擎：

```python
model_engine, optimizer, _, _ = deepspeed.initialize(args=cmd_args,
                                                     model=model,
                                                     model_parameters=params)
```

`deepspeed.initialize`确保在幕后为分布式数据并行或混合精度训练完成所有必要的设置。除了包装模型外，DeepSpeed还可以根据传递给`deepspeed.initialize`的参数和DeepSpeed[配置文件](https://www.deepspeed.ai/getting-started/#deepspeed-configuration)构建和管理训练优化器、数据加载器以及学习率调度器。请注意，DeepSpeed会在每个训练步骤自动执行学习率调度。

如果你已经设置了分布式环境，你需要将：

```python
torch.distributed.init_process_group(...)
```

替换为：

```python
deepspeed.init_distributed()
```

默认情况下使用NCCL后端，DeepSpeed已经对其进行了彻底测试，但你也可以[覆盖默认设置](https://deepspeed.readthedocs.io/en/latest/initialize.html#distributed-initialization)。

但如果直到`deepspeed.initialize()`之后你才需要设置分布式环境，那么你就不必使用这个函数，因为DeepSpeed会在其`initialize`期间自动初始化分布式环境。不管怎样，如果你已经设置了`torch.distributed.init_process_group`，你需要将其移除。

### 训练

一旦DeepSpeed引擎初始化完成，就可以使用三个简单的API进行训练：前向传播（可调用对象）、反向传播（`backward`）和权重更新（`step`）。

```python
for step, batch in enumerate(data_loader):
    # forward()方法
    loss = model_engine(batch)

    # 执行反向传播
    model_engine.backward(loss)

    # 权重更新
    model_engine.step()
```

在幕后，DeepSpeed自动执行分布式数据并行训练所需的必要操作，包括混合精度训练中的预定义学习率调度器：

- **梯度平均**：在分布式数据并行训练中，`backward`确保在训练`train_batch_size`后，梯度会在数据并行进程之间进行平均。
- **损失缩放**：在FP16/混合精度训练中，DeepSpeed引擎自动处理损失缩放，以避免梯度精度丢失。
- **学习率调度器**：当使用DeepSpeed的学习率调度器（在`ds_config.json`文件中指定）时，DeepSpeed会在每个训练步骤（当执行`model_engine.step()`时）调用调度器的`step()`方法。当不使用DeepSpeed的学习率调度器时：
  - 如果调度器应该在每个训练步骤执行，那么用户可以在初始化DeepSpeed引擎时将调度器传递给`deepspeed.initialize`，让DeepSpeed管理其更新或保存/恢复。
  - 如果调度器应该在任何其他间隔（例如，训练周期）执行，那么用户不应在初始化时将调度器传递给DeepSpeed，而必须显式管理它。

### 模型检查点

保存和加载训练状态是通过DeepSpeed的`save_checkpoint`和`load_checkpoint`API处理的，这两个函数接受两个参数来唯一标识一个检查点：

- `ckpt_dir`：检查点将被保存的目录。
- `ckpt_id`：一个标识符，用于在目录中唯一标识一个检查点。在下面的代码片段中，我们使用损失值作为检查点标识符。

```python
# 加载检查点
_, client_sd = model_engine.load_checkpoint(args.load_dir, args.ckpt_id)
step = client_sd['step']

# 将数据加载器推进到检查点步骤
dataloader_to_step(data_loader, step + 1)

for step, batch in enumerate(data_loader):

    # forward()方法
    loss = model_engine(batch)

    # 执行反向传播
    model_engine.backward(loss)

    # 权重更新
    model_engine.step()

    # 保存检查点
    if step % args.save_interval:
        client_sd['step'] = step
        ckpt_id = loss.item()
        model_engine.save_checkpoint(args.save_dir, ckpt_id, client_sd = client_sd)
```

DeepSpeed可以自动保存和恢复模型、优化器以及学习率调度器的状态，同时隐藏这些细节。然而，用户可能希望保存一些特定于给定模型训练的额外数据。为了支持这些数据，`save_checkpoint`接受一个客户端状态字典`client_sd`用于保存。这些数据可以通过`load_checkpoint`作为返回参数检索。在上面的示例中，`step`值被存储为`client_sd`的一部分。

**重要**：所有进程都必须调用此方法，而不仅仅是rank为0的进程。这是因为每个进程都需要保存其主权重以及调度器+优化器状态。如果只在rank为0的进程中调用此方法，它将挂起等待与其他进程同步。

## DeepSpeed配置

DeepSpeed的功能可以通过一个配置JSON文件启用、禁用或配置，该文件应指定为`args.deepspeed_config`。下面是一个示例配置文件。要查看完整的功能集，请参阅[API文档](https://www.deepspeed.ai/docs/config-json/)。

```json
{
  "train_batch_size": 8,
  "gradient_accumulation_steps": 1,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.00015
    }
  },
  "fp16": {
    "enabled": true
  },
  "zero_optimization": true
}
```

# 启动DeepSpeed训练

DeepSpeed安装了一个入口点`deepspeed`，用于启动分布式训练。我们通过以下假设来说明DeepSpeed的示例用法：

1. 你已经将DeepSpeed集成到你的模型中
2. `client_entry.py`是你的模型的入口脚本
3. `client args`是`argparse`命令行参数
4. `ds_config.json`是DeepSpeed的配置文件

## 资源配置（多节点）

DeepSpeed使用与[OpenMPI](https://www.open-mpi.org/)和[Horovod](https://github.com/horovod/horovod)兼容的主机文件来配置多节点计算资源。主机文件是一个包含*主机名*（或SSH别名）的列表，这些主机可以通过无密码SSH访问，以及*槽位数*，用于指定系统上可用的GPU数量。例如，

```
worker-1 slots=4
worker-2 slots=4
```

指定两台名为*worker-1*和*worker-2*的机器，每台机器有四个GPU用于训练。

主机文件可以通过`--hostfile`命令行选项指定。如果没有指定主机文件，DeepSpeed会搜索`/job/hostfile`。如果没有指定或找到主机文件，DeepSpeed会查询本地机器上的GPU数量以发现可用的本地槽位数量。

以下命令将在`myhostfile`中指定的所有可用节点和GPU上启动一个PyTorch训练作业：

```
deepspeed --hostfile=myhostfile <client_entry.py> <client args> \
  --deepspeed --deepspeed_config ds_config.json
```

或者，DeepSpeed允许你将模型的分布式训练限制为可用节点和GPU的一个子集。此功能通过两个命令行参数启用：`--num_nodes`和`--num_gpus`。例如，可以使用以下命令将分布式训练限制为仅使用两个节点：

```
deepspeed --num_nodes=2 \
	<client_entry.py> <client args> \
	--deepspeed --deepspeed_config ds_config.json
```

你也可以使用`--include`和`--exclude`标志来包含或排除特定资源。例如，要使用所有可用资源**除了**节点*worker-2*上的GPU 0和节点*worker-3*上的GPU 0和1：

```
deepspeed --exclude="worker-2:0@worker-3:0,1" \
	<client_entry.py> <client args> \
	--deepspeed --deepspeed_config ds_config.json
```

类似地，你可以**只使用**节点*worker-2*上的GPU 0和1：

```
deepspeed --include="worker-2:0,1" \
	<client_entry.py> <client args> \
	--deepspeed --deepspeed_config ds_config.json
```

### 无密码SSH启动

DeepSpeed现在支持在无需无密码SSH的情况下启动训练作业。这种模式在云环境中特别有用，例如Kubernetes，其中可以灵活地进行容器编排，而在云环境中设置领导者-worker 架构并使用无密码SSH会增加不必要的复杂性。

要使用这种模式，你需要在所有节点上分别运行DeepSpeed命令。命令结构如下：

```
deepspeed --hostfile=myhostfile --no_ssh --node_rank=<n> \
    --master_addr=<addr> --master_port=<port> \
    <client_entry.py> <client args> \
    --deepspeed --deepspeed_config ds_config.json
```

- `--hostfile=myhostfile`：指定包含节点和GPU信息的主机文件。
- `--no_ssh`：启用无SSH模式。
- `--node_rank=<n>`：指定节点的排名。这应该是一个从0到n - 1的唯一整数。
- `--master_addr=<addr>`：领导者节点（排名0）的地址。
- `--master_port=<port>`：领导者节点的端口。

在这种设置中，主机文件中的主机名不需要通过无密码SSH访问。然而，主机文件仍然需要，以便启动器收集有关环境的信息，例如节点数量和每个节点的GPU数量。

每个节点都必须使用唯一的`node_rank`启动，并且所有节点都必须提供领导者节点（排名0）的地址和端口。这种模式使启动器的行为类似于`torchrun`启动器，如[PyTorch文档](https://pytorch.org/docs/stable/elastic/run.html)中所述。

## 多节点环境变量

在跨多节点训练时，我们发现支持传播用户定义的环境变量非常有用。默认情况下，DeepSpeed会传播所有设置的NCCL和PYTHON相关环境变量。如果你希望传播额外的变量，可以在一个名为`.deepspeed_env`的点文件中指定它们，该文件包含一个以换行符分隔的`VAR=VAL`条目列表。DeepSpeed启动器将在你执行的本地路径和你的主目录（`~/`）中查找该文件。如果你希望覆盖此文件或路径的默认名称，并使用自己的名称，可以通过环境变量`DS_ENV_FILE`来指定。如果你正在启动多个需要不同变量的作业，这将非常有用。

以一个具体的例子来说，一些集群在训练之前需要设置特殊的NCCL变量。用户可以简单地将这些变量添加到主目录中的`.deepspeed_env`文件中，该文件如下所示：

```
NCCL_IB_DISABLE=1
NCCL_SOCKET_IFNAME=eth0
```

DeepSpeed将确保在启动每个节点上的每个进程时设置这些环境变量。

### MPI和AzureML兼容性

如上所述，DeepSpeed提供了自己的并行启动器，以帮助启动多节点/多GPU训练作业。如果你更喜欢使用MPI（例如`mpirun`）来启动训练作业，我们提供了对此的支持。需要注意的是，DeepSpeed仍将使用torch分布式NCCL后端，而不是MPI后端。

要使用`mpirun` + DeepSpeed或使用AzureML（它使用`mpirun`作为启动器后端）启动训练作业，你只需安装[mpi4py](https://pypi.org/project/mpi4py/)Python包。DeepSpeed将使用它来发现MPI环境，并将必要的状态（例如，世界大小、排名）传递给torch分布式后端。

如果你使用模型并行、流水线并行，或者在调用`deepspeed.initialize(..)`之前需要调用`torch.distributed`，我们提供了额外的DeepSpeed API调用。将你的初始`torch.distributed.init_process_group(..)`调用替换为：

```python
deepspeed.init_distributed()
```

## 资源配置（单节点）

在我们仅在单个节点（一个或多个GPU）上运行的情况下，DeepSpeed*不需要*上面描述的主机文件。如果没有检测到或传递主机文件，DeepSpeed将查询本地机器上的GPU数量以发现可用的槽位数量。`--include`和`--exclude`参数仍然可以正常使用，但用户应该指定`localhost`作为主机名。

还需要注意的是，`CUDA_VISIBLE_DEVICES`可以与`deepspeed`一起使用，以控制在单个节点上使用哪些设备。因此，以下两种方法都可以只在当前节点的设备0和1上启动：

```
deepspeed --include localhost:0,1 ...
CUDA_VISIBLE_DEVICES=0,1 deepspeed ...
```

# DeepSpeed 启动脚本

| 参数                                               | 说明                                                                                                                                                                                                                                                                 |
| -------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `-H HOSTFILE, --hostfile HOSTFILE`                 | 主机文件路径（MPI风格），定义作业可用的资源池（例如，`worker-0 slots=4`）（默认值：`/job/hostfile`）                                                                                                                                                                 |
| `-i INCLUDE, --include INCLUDE`                    | 指定执行期间要使用的硬件资源。字符串格式为`NODE_SPEC[@NODE_SPEC ...]`，其中`NODE_SPEC=NAME[:SLOT[,SLOT ...]]`。如果省略`:SLOT`，则包含该主机上的所有槽位。例如：`-i "worker-0@worker-1:0,2"`将使用`worker-0`上的所有槽位以及`worker-1`上的槽位[0, 2]。（默认值：空） |
| `-e EXCLUDE, --exclude EXCLUDE`                    | 指定执行期间不使用的硬件资源。与`--include`互斥。资源格式与`--include`相同。例如：`-e "worker-1:0"`将使用所有可用资源，除了`worker-1`上的槽位0。（默认值：空）                                                                                                       |
| `--num_nodes NUM_NODES`                            | 要运行的 worker 节点总数，将使用给定主机文件中的前N个主机。（默认值：-1）                                                                                                                                                                                            |
| `--min_elastic_nodes MIN_ELASTIC_NODES`            | 运行弹性训练的最小节点数。启用弹性训练时，默认值为1。（默认值：-1）                                                                                                                                                                                                  |
| `--max_elastic_nodes MAX_ELASTIC_NODES`            | 运行弹性训练的最大节点数。启用弹性训练时，默认值为`num_nodes`。（默认值：-1）                                                                                                                                                                                        |
| `--num_gpus NUM_GPUS, --num_accelerators NUM_GPUS` | 每个节点上要使用的最大GPU数量，将在每个节点上使用[0:N)的GPU ID。（默认值：-1）                                                                                                                                                                                       |
| `--master_port MASTER_PORT`                        | （可选）PyTorch分布式在训练期间用于通信的端口。（默认值：29500）                                                                                                                                                                                                     |
| `--master_addr MASTER_ADDR`                        | （可选）节点0的IP地址，如果未指定，将通过`hostname -I`推断。（默认值：空）                                                                                                                                                                                           |
| `--node_rank NODE_RANK`                            | 范围在[0:N)内的每个节点的ID。仅在设置`--no_ssh`时需要。（默认值：-1）                                                                                                                                                                                                |
| `--launcher LAUNCHER`                              | （可选）选择用于多节点训练的启动器后端。当前选项包括PDSH、OpenMPI、MVAPICH、SLURM、MPICH、IMPI。（默认值：pdsh）                                                                                                                                                     |
| `--launcher_args LAUNCHER_ARGS`                    | （可选）将特定于启动器的参数作为单个带引号的参数传递。（默认值：空）                                                                                                                                                                                                 |
| `--module`                                         | 将每个进程更改为将启动脚本解释为Python模块，执行行为与`python -m`相同。（默认值：False）                                                                                                                                                                             |
| `--no_python`                                      | 跳过在训练脚本前添加`python`，直接执行脚本。（默认值：False）                                                                                                                                                                                                        |
| `--no_local_rank`                                  | 在调用用户的训练脚本时不传递`local_rank`作为参数。（默认值：False）                                                                                                                                                                                                  |
| `--no_ssh`                                         | 在每个节点上独立启动训练，无需设置SSH。（默认值：False）                                                                                                                                                                                                             |
| `--no_ssh_check`                                   | 在多节点启动器模型中不执行SSH检查（默认值：False）                                                                                                                                                                                                                   |
| `--force_multi`                                    | 强制多节点启动器模式，有助于用户在单个远程节点上启动时。（默认值：False）                                                                                                                                                                                            |
| `--save_pid`                                       | 在`/tmp/<main-pid>.ds`处保存包含启动器进程ID（pid）的文件，其中`<main-pid>`是调用`deepspeed`的第一个进程的pid。在程序化启动DeepSpeed进程时非常有用。（默认值：False）                                                                                                |
| `--enable_each_rank_log ENABLE_EACH_RANK_LOG`      | 将每个rank的stdout和stderr重定向到不同的日志文件（默认值：None）                                                                                                                                                                                                     |
| `--autotuning {tune,run}`                          | 运行DeepSpeed自动调优器，在运行作业之前发现最优配置参数。（默认值：空）                                                                                                                                                                                              |
| `--elastic_training`                               | 在DeepSpeed中启用弹性训练支持。（默认值：False）                                                                                                                                                                                                                     |
| `--bind_cores_to_rank`                             | 将每个rank绑定到主机的不同核心上（默认值：False）                                                                                                                                                                                                                    |
| `--bind_core_list BIND_CORE_LIST`                  | 要绑定到的核心列表，以逗号分隔的数字和范围列表。例如，`1,3-5,7` => `[1,3,4,5,7]`。如果未指定，将使用系统上的所有核心进行rank绑定。（默认值：None）                                                                                                                   |
| `--ssh_port SSH_PORT`                              | 用于远程连接的SSH端口（默认值：None）                                                                                                                                                                                                                                |
