# 深度学习分布式训练框架 horovod (8) --- on spark



## 摘要

Horovod 是Uber于2017年发布的一个易于使用的高性能的分布式训练框架，在业界得到了广泛应用。

本系列将通过源码分析来带领大家了解 Horovod。接下来几篇介绍 horovod 如何运行在 spark 之上。本文是第八篇，介绍 horovod on spark 的总体架构。

Horovod on spark 的目的就是让 horovod 能跑到 spark 集群上，从而把数据处理，模型训练，模型评估这一个机器学习的循环都放在Spark技术栈之中。

本系列其他文章如下：

## 1. Spark相关知识

### 1.1 为什么整合 Spark

Spark是一个分布式通用计算框架，而以 tensorflow 为代表的深度学习框架是分布式模型训练框架，这些框架更多专注用迭代来计算梯度。很多业内公司都是用spark来获取/处理数据，然后把spark处理好的数据结果发给Tensorflow进行训练。

目前我们已经知道，Horovod 可以把 Tensorflow等深度学习框架和MPI紧密结合起来，那么为什么要再把 spark 整合进来呢？整合的意义在哪里？具体如下：

- MPI是一个较低层级的库，专注于提供可移植的性能而忽略了程序员生产力（原语过于低级，开发代码量大）。Spark是一个更高级别的框架，更专注于程序员的生产力。Spark可以使开发者用单机串行程序的思维来开发分布式程序，这样用户可以更加专注于算法本身，而不需将精力过多放在分布式逻辑上。
- 整合之后，可以让整个特征处理和训练流程都统一在 spark 环境内，从而实现更好的分布式训练和数据传输。
- MPI集群的任务成功率并不高，如果某个任务失败，往往需要重启整个MPI集群。因为 MPI的容错性较差，所以希望能够借助spark的容错机制。

Horovod 需要解决的**核心问题**是：如何将spark作为分布式tensorflow的底层调动机制，从而通过spark executor就可以把 tensorflow 的进程调动起来，这样进行tensorflow训练时就不需要手动地去组建网络。

因此能想到的其他问题是：

- Spark如何开始运行？当某一个 Executor 启动后就可以运行？还是需要所有的 Executor 都准备好之后才能一起跑？
- 如何发布 训练代码？
- 如何在 Spark Executor 之上启动用户代码？
- MPI 在这个机制中起到什么作用？

我们在随后一一分析。

### 1.2 Spark 简单架构

简要来说，Spark分成几个角色：

- Driver。这是一个进程，我们编写好的Spark程序在spark-submit提交之后，就是由Driver进程执行。充当Driver的可能是Spark集群的某个节点、比如就是你提交Spark程序的机器。
- Executor。也是一个进程，在一个Executor进程里面会有多个task线程。这里的Executor和task主要负责对RDD的partition进行并行计算，也就是执行我们在程序中指定的RDD算子（map、flatMap、reduceByKey等）。
- Task。是一个线程，主要是负责实际的执行算子任务。一个 task 对应一个线程，多个 task 可以并行的运行在 executor 之中。用户代码经过Spark Driver 调度之后，被封装成若干Task，Driver 再将这些Task信息发给Executor执行，Task信息包括代码逻辑以及数据信息。Executor不直接运行用户的代码。

### 1.3 Pyspark 原理

当我们用python编写程序时，其实使用的是 Pyspark 接口。所以我们介绍一下 pyspark，可以和 Horovod 做比对。

#### 1.3.1 架构修改

如果我们使用Java或者Scala开发Spark相关程序，Driver 和 Executor 运行任务的载体是Java虚拟机（JVM）。但是 Python 使用的是 Python自己的虚拟机，这就产生了一个问题，核心架构是基于JVM还是PVM。

为了保持核心架构一致性，Spark依然使用JVM作为核心，核心功能依然基于JVM，其中包括：申请计算资源，管理/分配task，driver与executor之间的通信等等。在此核心架构外围则封装了一层python。

因此，PySpark 采用了 Python进程和JVM 进程分离的多进程架构，在 Driver和Executor 端都同时有 Python和JVM 两个进程。

#### 1.3.2 Driver端

如果用户提交一个Python 脚本，Spark Driver 会：

- 运行这个脚本；
- 通过Python 启动 JVM；
- 如果Python脚本中调用了DataFrame或者RDD操作，则会通过Py4j调用到Java方法，将用户的"Spark"操作映射到JVM之中。比如python调用 `a.map(lambda x:(x,1))`，则这个rdd的操作会映射到JVM之中被执行。

#### 1.3.3 Executor端

在Executor则正好相反，因为Executor端运行的Task逻辑（序列化后的字节码）是由Driver发过来的，所以 Executor 本来是可以直接运行Task，并不需要借助任何Py4j。但是因为Python脚本中会存在用户定义的python函数（或者Lambda表达式），所以Executor必须再启动Python进程进行相关处理：

- 当Driver申请到Executor的资源之后，会启动Executor 的 JVM 进程，如果没有Task下发过来，则Executor只有JVM，没有其他进程，即没有下面提到的Python进程；
- Executor接到任务之后，会启动一个task进行处理。如果不存pyspark.deamon后台公共进程，则Executor会通过Java Process的方式启动pyspark.deamon后台公共进程，pyspark.deamon负责接收Task的相关请求。此deamon在每个Executor上只有一个。
- pyspark.deamon接收到请求之后，会为每一个Task单独启动一个Python子进程（pyspark worker）；
- RDD的载体依然在Executor之中，当有udf和lambda逻辑时，Executor会通过socket作为载体，同pyspark worker进行数据通信，把数据不停的提供给 pyspark worker；
- 当pyspark worker运行之后会把结果通过socket返回给JVM；

#### 1.3.4 流程

交互流程如下图，实线是方法调用，虚线是返回结果。

![img](https://img2020.cnblogs.com/blog/1850883/202106/1850883-20210630080416060-2054922569.png)

架构图如下：

![img](https://img2020.cnblogs.com/blog/1850883/202106/1850883-20210630080513537-909792905.png)

## 2 机器学习 on Spark

### 2.1 机器学习的特点

机器学习算法和计算机领域的其他算法相比，有自己的一些独特特点。例如：

- 迭代性。模型的更新并非一次完成，需要循环迭代多次；
- 容错性。即使在每个循环中产生一些错误，模型最终的收敛也不会受到影响。这于传统分布式系统形成鲜明对比，比如分布式文件系统就无法接受任何数据块的写入错误。
- 参数收敛的非均匀性。模型中某些参数可能经过几个循环便不再改变，而某些参数需要很长时间多次迭代才能收敛。
- 网络是瓶颈。频繁更新模型参数需要消耗大量带宽，而GPU速度越快，网络瓶颈就越成为问题所在。

以上这些特点决定了机器学习系统的设计和其他计算系统的设计有很大区别。和传统分布式系统比较，机器学习系统在通信，同步和容错等方面都活动空间极大。因为大量资源都会浪费在通讯，等待，协调这些非计算任务上，所以导致分布式机器学习任务往往并不能随着机器数量随之的增加而能力也线性提升。

因此，在设计大规模机器学习系统（比如深度学习/逻辑回归/主题模型/矩阵分解等依赖于SGD或者L-BFGS最优化的算法）时，需要解决一系列挑战，比如提高并行度，减少同步等待延迟，容错以及巨大带宽（频繁访问修改模型参数时所需）等。

### 2.2 机器学习 on Spark

MPI 的主要缺点是：

- 原语过于低级。用MPI写算法，往往代码量比较大也比较复杂。
- 容错机制差。如果某个任务失败，往往需要重启整个MPI集群，而MPI集群的任务成功率并不高。
- MPI本身也无法支撑大规模数据。

Spark在一定层度上解决了MPI的问题。

**MapRduce 和机器学习执行模型的差别**

- Spark（MapReduce）：所有独立的任务都可以并行执行，只是有的先执行完，有的后执行完，下一个迭代（Stage）开始时，前面一个Stage必须是所有的任务（Task）都已经执行完毕了。简而言之，同一个Stage内，每个任务之间是完全独立的，没有任何依赖关系的（有依赖关系那就是算成下一个stage），因此可以并行执行。

- 分布式训练：每个独立的任务，需要同时启动，每个独立任务执行完毕之后，彼此之间需要及时通信，传递梯度参数数据，因此没有明显的Spark（MapReduce）中的Stage的概念，而且，有单个任务失败，也很难通过独立重新计算改任务来恢复，通常都是整个训练作业都重来。

#### 2.2.1 简单模型

Spark训练的一个最最简陋的整体流程图如下：

- Map 操作定义了数据分发和在工作节点的计算：
  - 首先在map阶段对数据进行分割，分发给每一个 Executor；
  - 在 Executor 之中，利用随机梯度等方法逼近最优解；
- 在 reduce 阶段定义了模型参数的聚合过程。
  - 最后 Executor 输出一个模型；

```python
                                 +----------------+
                                 |                |
                                 |  Spark Driver  |
                                 |                |
                                 +----------------+
                                          +
                           Map Stage      |      Reduce Stage
                                          |
                                          |
              +--------------------+      |
              | Spark Executor     |      |
              |                    +----------+
    +-------> |      User function |      |   |
    |         |                    |      |   |
    |         +--------------------+      |   |
    |                                     |   |
    |         +--------------------+      |   |   +------------------+
    |         | Spark Executor     |      |   +-> | Spark Executor   |
+---+--+      |                    |      |       |                  |      +-----+
| Data +----> |      User function +------------> |                  +----> |model|
+---+--+      |                    |      |       |     User function|      +-----+
    |         +--------------------+      |   +-> |                  |
    |                                     |   |   +------------------+
    |         +--------------------+      |   |
    |         |  Spark Executor    |      |   |
    |         |                    |      |   |
    +-------> |      User function +----------+
              |                    |      |
              +--------------------+      +
```

但是我们发现，这个工作流程只能迭代一次，完全不匹配机器学习需要循环迭代多次的特点，于是还需要修改这个架构。

#### 2.2.2 升级模型

于是我们修改角色如下：

- Spark driver不但要负责协调整个Spark任务执行，还需要保存最近所有梯度，并且负责对Executor传来的梯度做更新。
- 而executor负责分布式地计算梯度向量，并且梯度提交给driver。

迭代过程也拓展如下：

1. 每轮迭代中，executor负责分布式地计算梯度向量，然后将每个 executor 计算的梯度更新值 Aggregate 到 driver。
2. 全局梯度 保存在driver上，driver根据每个梯度的最新值进行聚合，并且更新模型参数值 w。
3. Driver 将 更新后的参数值 w 广播到每个Executor。

最后 reduce 阶段导出模型。

```python
  Map Stage               +----------------+
              1           |       2        |          1
       +----------------> |  Spark Driver  | <-------------------+
       |                  |                |                     |
       |                  +--+------+---+--+                     |
       |                     |   3| ^   |                        |
       |                     |    | |   |                        |
       |           3         |    | |   |           3            |
       |    +----------------+    | |   +-------------------+    |
       |    |                     | |                       |    |
       |    v                     v |1                      v    |
 +-----+----+---------+  +--------+-----------+  +----------+----+----+
 | Spark Executor     |  | Spark Executor     |  |  Spark Executor    |
 |                    |  |                    |  |                    |
 |      User function |  |      User function |  |      User function |
 |                    |  |                    |  |                    |
 +-------------+------+  +--------+-----------+  +--------+-----------+
               |                  |                       |
+----------------------------------------------------------------------+
               |                  |                       |
               +-----------+      |      +----------------+
                           |      |      |
 Reduce Stage              v      v      v
                         +-+------+------+--+
                         | Spark Executor   |
                         |                  |
                         |                  |
                         |     User function|
                         |                  |
                         +--------+---------+
                                  |4
                                  |
                                  v
                               +--+--+
                               |model|
                               +-----+
```

我们突然发现，这居然是一个参数服务器的架构了，即 Spark Driver 充当了参数服务器的角色。这和 Horovod 的 ring-allreduce 的架构显然不符合。另外，Spark采用的完全是BSP协议，即第二轮迭代必须等到第一轮迭代所有的机器完成，这也会拖慢我们的训练过程。

### 2.3 机器学习 on Spark 的缺陷

所以，我们在深入之前，需要先说说Spark 如果用于机器学习，会有哪些缺陷：

- **规模依旧不足**。Spark受限于模型大小和内存限制，只是中等规模机器学习框架。其瓶颈就是Driver。

  - Spark框架以Driver为核心，Driver 负责具体任务调度和参数汇总；
  - driver又是单机结构，难以扩展；
  - 当模型规模超过Driver或者Executor所在机器内存的时候，Spark就无法正常运行；

- **本质仍不匹配**。机器学习的核心是迭代和参数更新。Spark的核心概念是RDD。这两者的特点不能很好匹配。

  - RDD具备一系列transformation和action接口。用户使用这些接口完成成不同的算法或应用。但这组接口是通用接口，无法灵活高效应用于特定领域问题。

  - RDD 并不能很好地支持机器学习中的迭代运算，另外节点之间通信也低效。

    因为大规模机器学习，其模型参数会非常巨大，如果使用 RDD 去容纳所有更新的模型参数。需要在每次迭代中创建新的 RDD，这涉及到机器和磁盘间的频繁数据交换，这会带来大量额外开销。

  - RDD难以满足参数反复迭代更新的需求。

    RDD使用不可变性这个特点来规避分布式环境下的并行问题。此抽象可以简化算子复杂度，提供高性能分布式数据处理能力，非常适合数据分析领域。然而不可变性却不适合参数反复更新这个需求。

虽然 Spark 对于机器学习来说有各种缺陷，但是对于中等规模的学习确实非常有用，所以就有了 Horovod on spark。我们接下来就要看看 Horovod 是如何处理（缓解）这些问题的。大规模机器学习的目的就是解决"数据和偏差"规模非常大的时候所带来的理论/工程问题。

## 3. 整体架构

### 3.1 整体思路

Tensorflow是C++开发的，而python是机器学习世界的主宰。所以，如果Spark要和TensorFlow 进行整合，一般来说有以下三种方式：

- 通过Tensorflow Java API；
- 通过Tensorflow Python API；
- 通过JNI来调用Tensorflow C++ API；

但是 Horovod 的思路又比较别致，可以认为是按照 Spark 的思路，在 Spark 之上又实现了一套自己的。即：

- Horovod 也有自己的 DriverService（可以认为其对应了 spark driver），或者说 Horovod job 自己就变成了 Spark driver，负责全局初始化，启动协调和后续任务分发；
- Horovod 也有自己的 TaskService（可以认为其对应了 spark Executor）；
- Horovod DriverService 用 `horovod.spark._make_spark_thread` 创建了 Spark 集群；
- Horovod DriverService 然后在Spark 集群上创建了`num_proc`个 tasks（Horovod TaskService），这些 tasks 都注册到 driver 之上，因此 driver 知道已经启动的所有 task信息（ip，port，路由，...），这些task 也把自己的 host hash（一个被 MPI 当作 host 的字符串）发送给Horovod DriverService ；
- Horovod DriverService 会 通知 Horovod TaskService 启动训练；
- 每个 Horovod TaskService 在其所在的 Spark Executor之上，通过调用本地进程的方式 mpi 程序，在mpi程序之中又启动Tensorflow或者Torch来训练模型。这样相当于：
  - Spark变成容器进行计算资源的调度；
  - Tensorflow或者Torch来训练模型;
  - mpi来在各个 Executor 之间做交互做 all-reduce，从而更新梯度等；

这样就充分利用了已有的大数据体系的数据和计算特性。其实，绝大多数大规模机器学习的平台/系统都可以看做这由这两个角色构成 ：Model node（driver node）和 Data node（worker node）。每个角色都有自己一套计算逻辑。从 Horovod来说，Horovod DriverService 就是 driver node，Horovod TaskService就是 data node：

- 数据分布在 n 个 data node节点上，data node 从 model node 接收任务和代码，然后进行计算，并且把计算结果发送给模型节点。Horovod TaskService 就是完成如下操作，只是不需要发送计算结果给Horovod DriverService；
- 模型（代码）分布在 m 个model node节点上。在模型结点上进行模型更新，更新是依据"当前模型在数据节点计算/汇总结果 VS 理想模型" 这个偏差来完成。Horovod DriverService （系统中只有一个）就负责维护代码，把任务和代码发给Horovod TaskService，但是Horovod DriverService没有更新模型的操作，转而由Horovod TaskService 通过 Ring-Allreduce 自行完成。

大致如下，其中 SparkDriverService 对应了Horovod DriverService，SparkTaskService对应了Horovod TaskService：

```python
                       +------------------------------+
                       |      Horovod Main thread     |
                       |                              |
                       |                              |
                       |       SparkDriverService     |
                       |                              |
                       |       +----------------+     |
                       |       | Spark Driver   |     |
                       |       +----------------+     |
                       +------------------------------+
                                       |
                                       |
            +--------------------------------------------------------+
            |                          |                             |
            |                          |                             |
            v                          v                             v

+------------------------+   +----------------------+   +------------------------+
|     Spark Executor     |   |    Spark Executor    |   |     Spark Executor     |
|                        |   |                      |   |                        |
| +-------------------+  |   | +------------------+ |   | +-------------------+  |
| |  SparkTaskService |  |   | | SparkTaskService | |   | |  SparkTaskService |  |
| |                   |  |   | |                  | |   | |                   |  |
| |   TensorFlow      |  |   | |    TensorFlow    | |   | |     TensorFlow    |  |
| |                   |  |   | |                  | |   | |                   |  |
| |                   |  |   | |                  | |   | |                   |  |
| |       MPI         |  |   | |       MPI        | |   | |        MPI        |  |
| |        +          |  |   | |        +         | |   | |         +         |  |
| |        |          |  |   | |        |         | |   | |         |         |  |
| +-------------------+  |   | +------------------+ |   | +-------------------+  |
|          |             |   |          |           |   |           |            |
|          |             |   |          |           |   |           |            |
+------------------------+   +----------------------+   +------------------------+
           |                            |                           |
           |                            |                           |
           |                            |                           |
           +----------------------------+---------------------------+
```

手机如下：

![img](https://img2020.cnblogs.com/blog/1850883/202106/1850883-20210630080550620-1359923249.png)

### 3.2 具体分析

具体分析如下。

- 在 Horovod 的主进程中运行一个 SparkDriverService（对应 spark driver），或者说就是 Spark driver。

- 利用 _make_spark_thread 启动 Spark Executor，从而建立了一个Spark集群，然后 horovod 会等待所有Executor启动结束;

- 在 spark 的 每个 Executor 上运行一个 SparkTaskService（对应 spark Executor）。

- MPI 需要得到 host 之间的路由信息，所以 horovod 需要得到这些信息：

  - 回忆一下，在没有 spark 的情况下，也需要获取到这些 host 之间的路由信息。因为 host 之间是一个环形，构成了 ring allreduce。
  - 在 Hovorod on spark 状态下，我们的训练函数实际上是在 Spark Executor 中运行，为了进行 ring allreduce，所以现在需要知道 spark Executor 之间的路由，以及 driver & tasks 对应关系。

- SparkTaskService 把自己的地址和端口注册到 SparkDriverService 之上。

  - 这样 SparkTaskService 通过 SparkDriverService 可以获得自己和彼此的各种信息。
  - SparkTaskService 通过函数，也能够知道 spark Executor 之间的路由，从而可以互相访问。

- 从逻辑上来说， spark exector 自己本身的逻辑任务此时已经结束了，因为以后都是 SparkTaskService 自己独立完成的动作，SparkTaskService 来负责从SparkDriverService接收训练代码，启动训练；

- SparkDriverService 知道所有 SparkTaskService 启动之后，会通知他们进入下一个阶段，即等待任务。

- Horovod main thread 在通过SparkDriverService 知道所有 task 启动之后，会 用

  ```
  mpi_run
  ```

  来在这些 tasks 之中启动 python function（通过 RPC）。

  - 通常，MPI 会通过 SSH 来连接 hosts，但是这种方式无法在 Spark Executor 之中启动 Python function。
  - 因此 MPI 使用 RPC 来启动用户代码，即使用 `horovod.spark.driver.mpirun_rsh` 来连接每个 Executor，然后 "remote shell" 到这些 spark executors 之中。
  - `horovod.spark.driver.mpirun_rsh` 是与每个 host hash 之中 最小 index 的 task进行通信，这个 task 就执行 MPI 的 `orted` 命令。因此，每个 Executor 之中只会运行一个 `mpi orted` 进程，即使这个 executor 有多个 tasks。其他的 非`orted` 进程 task会等待 `orted` 进程 task 结束。

- 在mpirun_rsh之中， SparkDriverService 给 SparkTaskService 发送 RunCommandRequest，要求 Task 启动训练。

- SparkTaskService 在 spark Executor 内部将会使用

  ```
  _run_command
  ```

  在 spark 之中启动训练job。具体如下：

  - mpi_run 实际上是在 每一个 Spark Executor 之上运行 mpi 程序。即，Horovod 调用 mpi_run （又利用到 mpirun_rsh.py）在每一个 spark executor 上启动 orted，以启动 MPI cluster。
  - SparkTaskService 可以 从 SparkDriverService 得到训练代码；
  - orted 在每一个 executor 之上运行训练代码，即 python function；
  - 我们的训练代码也是一个 mpirun 程序，即使运行了 tensor flow，也是一个mpi程序，因为一开始从 SparkTaskService 得到了地址和端口，所以可以彼此交互，实现 ring-allreduce。

备注：

Hovorod 期望所有的 task 都同时运行，因此 cluster 应该至少提供同样个数的 core，每个 executor 可以有多个 core，因此一个 executor 可以处理多个 tasks，host 可以有多个 executor。

具体如下图：

```java
+--------------------------+                     +---------------------------------+  +-------------------------+
| Horovod Main thread      |                     | Spark Executor                  |  | Spark Executor          |
|                          |                     |                                 |  |                         |
|                          |                     |                                 |  |                         |
| +--------------------+   |       1 register    |        +----------------------+ |  |  +--------------------+ |
| | SparkDriverService +<---------------------------------+  SparkTaskService    | |  |  |  SparkTaskService  | |
| |                    |   |                     |        |                      | |  |  |                    | |
| |                    |   |      2 notify start |        |                      | |  |  |                    | |
| |                    +--------------------------------> |                      | |  |  |                    | |
| |                    |   |                     |        |                      | |  |  |                    | |
| |                    |   |                     |        |                      | |  |  |                    | |
| |                    |   | 3 RunCommandRequest |        |                      | |  |  |                    | |
| |                    +---------------------------------------> orted mpirun_rsh| |  |  |                    | |
| |                    |   |                     |        |        +             | |  |  |                    | |
| |                    |   |                     |        |        | 4           | |  |  |                    | |
| |                    |   |                     |        |        |             | |  |  |                    | |
| |                    |   |                     |        |        v             | |  |  |                    | |
| |                    |   |                     |        |      task_exec       | |  |  |                    | |
| |                    |   |                     |        |        +             | |  |  |                    | |
| |                    |   |                     |        |        | 5           | |  |  |                    | |
| |                    |   |                     +        |        |             | |  |  |                    | |
| |                    |   |6 set_local_rank_to_rank      |        v             | |  |  |                    | |
| |                    +-------------------------+---------> SparkTaskClient     | |  |  |                    | |
| |                    |   |                     |        |                      | |  |  |                    | |
| |                    |   |                     |        | +------------------+ | |  |  | +----------------+ | |
| |                    |   |    7 code()         |        | |                  | | |  |  | |                | | |
| |                    +----------------------------------------> 8 train()    | | |  |  | |     train()    | | |
| |                    |   |                     |        | |                  | | |  |  | |                | | |
| |                    |   |                     |        | |       MPI <---------------------->  MPI       | | |
| |                    |   |                     |        | |                  | | |  |  | |                | | |
| |                    |   |                     |        | +------------------+ | |  |  | +----------------+ | |
| +--------------------+   |                     |        +----------------------+ |  |  +--------------------+ |
+--------------------------+                     +---------------------------------+  +-------------------------+
```

手机如下：

![img](https://img2020.cnblogs.com/blog/1850883/202106/1850883-20210630080613595-1435931303.png)

### 3.3 Horovod on Spark 架构图

在 Horovod 源码中，有一个架构图。我们可以大致了解其架构。

但是因为这部分实在复杂，所以单凭这一幅图很难了解其实现，所以我们需要做深入研究。

![img](https://img2020.cnblogs.com/blog/1850883/202106/1850883-20210630081126227-1529894591.png)

首先我们看看 Driver 的特点。

### 3.4 普通状况 Driver

我们首先用普通Horovod驱动做个对比。

在没有 spark 的情况下，假设有多个 hosts，需要获取到这些 host 之间的路由信息。因为 host 之间是一个环形，构成了 ring allreduce。

```
Tasks ping each other in a circular fashion to determine interfaces reachable within the cluster.
```

Driver 服务由 HorovodRunDriverService 提供，Task 服务由 HorovodRunTaskService 等提供。

其功能主要是维护各种 task 地址以及相应关系。具体各种 task 地址就是 Task 服务 来注册的。

需要注意的是：HorovodRunDriverService 和 HorovodRunTaskService 都最终继承了 network.BasicService，他们之间可以是异地运行交互。

### 3.5 Spark 相关的Driver

在 Hovorod on spark 状态下，我们的训练函数实际上是在 Spark Executor 中运行，因为面对的情况不同，所以我们对于 Driver 需求是不同的。之前记录的是 host 之间的路由以及 driver & tasks 对应关系。现在需要知道 spark Executor 之间的路由，以及 driver & tasks 对应关系。

## 4. Spark 模式入口

### 4.1 示例代码

从源码中找到示例代码如下，可以看到，horovod.spark.run 是入口。

```python
# Horovod: run training.
history, best_model_bytes = \
    horovod.spark.run(train_fn, args=(model_bytes,), num_proc=args.num_proc,
                      stdout=sys.stdout, stderr=sys.stderr, verbose=2,
                      prefix_output_with_timestamp=True)[0]
```

### 4.2 Horovod.spark.run 逻辑

fn 就是训练函数，被用户代码传进来的，**具体被赋值之后，在 SparkDriverService 之中保存（具体是在其成员变量 _fn 之中），以后会使用**。**这样就解决了代码发布问题**。

```python
driver = driver_service.SparkDriverService(settings.num_proc, settings.num_proc,
                                           fn, args, kwargs,
                                           settings.key, settings.nics)
```

`Horovod.spark.run` 的逻辑是：

- 处理各种配置，比如timeout，nice...；
- 获取 spark 信息，比如从 pyspark 之中获取SparkContext；
- 构建驱动 SparkDriverService（Spark driver service）；
- 利用 _make_spark_thread 来启动 spark executor（以及在每一个 spark executor 之中启动一个SparkTaskService），这样就构建了 cluster；
- 利用 _notify_and_register_task_addresses 等待所有 spark task 都结束；
- 利用 _launch_job 启动训练；
- 利用 spark_thread.join 来收集训练结果；

具体代码如下：

```python
def run(fn, args=(), kwargs={}, num_proc=None, start_timeout=None,
        use_mpi=None, use_gloo=None, extra_mpi_args=None,
        env=None, stdout=None, stderr=None, verbose=1, nics=None,
        prefix_output_with_timestamp=False):

    # 处理各种配置，比如timeout，nice...
  	if start_timeout is None:
        # Lookup default timeout from the environment variable.
        start_timeout = int(os.getenv('HOROVOD_SPARK_START_TIMEOUT', '600'))

    # nics needs to be a set
    if nics and not isinstance(nics, set):
        nics = set(nics)

    tmout = timeout.Timeout(start_timeout, message)
    settings = hvd_settings.Settings(verbose=verbose,
                                     extra_mpi_args=extra_mpi_args,
                                     key=secret.make_secret_key(),
                                     start_timeout=tmout,
                                     nics=nics,
                                     run_func_mode=True,.....)

    # 获取 spark 信息，比如从 pyspark 之中获取SparkContext
    spark_context = pyspark.SparkContext._active_spark_context
    settings.num_proc = num_proc
    result_queue = queue.Queue(1)

    # 利用 _make_spark_thread 来启动 spark executor（以及在每一个 spark executor 之中启动一个SparkTaskService）
    # start Spark driver service and launch settings.num_proc Spark tasks
    spark_job_group = 'horovod.spark.run.%d' % job_id.next_job_id()
    driver = driver_service.SparkDriverService(settings.num_proc, settings.num_proc,
                                               fn, args, kwargs,
                                               settings.key, settings.nics)
    gloo_is_used = is_gloo_used(use_gloo=use_gloo, use_mpi=use_mpi, use_jsrun=False)
    spark_thread = _make_spark_thread(spark_context, spark_job_group, driver,
                                      result_queue, settings,
                                      use_gloo=gloo_is_used, is_elastic=False)
    try:
        # 等待第一阶段结束，即 等待所有 spark task 都结束
        # wait for all tasks to register, notify them and initiate task-to-task address registration
        _notify_and_register_task_addresses(driver, settings)

        # Determine the index grouping based on host hashes.
        # Barrel shift until index 0 is in the first host.
        host_hashes = list(driver.task_host_hash_indices().keys())
        host_hashes.sort()
        while 0 not in driver.task_host_hash_indices()[host_hashes[0]]:
            host_hashes = host_hashes[1:] + host_hashes[:1]

        settings.hosts = ','.join('%s:%d' % (host_hash, len(driver.task_host_hash_indices()[host_hash]))
                                  for host_hash in host_hashes)

        # Run the job，启动训练
        _launch_job(use_mpi, use_gloo, settings, driver, env, stdout, stderr)
    except:
        # Terminate Spark job.
        spark_context.cancelJobGroup(spark_job_group)

        # Re-raise exception.
        raise
    finally:
        spark_thread.join()
        driver.shutdown()

    # Make sure Spark Job did not fail.
    driver.check_for_spark_job_failure()

    # get ranks from driver
    indices_in_rank_order = _get_indices_in_rank_order(driver)

    # If there's no exception, execution results are in this queue.
    results = result_queue.get_nowait()
    return [results[index] for index in indices_in_rank_order]
```

既然知道了总体代码，下一篇我们就介绍 Horovod on spark 如何启动，敬请期待。

## 5 总结

至此，我们分析了 Horovod on spark 的总体架构，几个相关问题回答如下：

- 如何将spark作为分布式tensorflow的底层调动机制，通过spark executor去把tensorflow 的进程调动起来，这样在进行tensorflow训练时就不需要手动地去组建网络。

  - 答案  是：

    Horovod 的思路又比较别致，可以认为是按照 Spark 的思路，在 Spark 之上又实现了一套自己的。即：

    - Horovod 也有自己的 DriverService（对应了 spark driver），或者说 Horovod job 自己就变成了 Spark driver，负责全局的初始化，创建 Cluster，启动工作 和 后续任务分发；
    - Horovod 也有自己的 TaskService（对应了 spark Executor）；Horovod DriverService 在Spark cluster上创建了`num_proc`个 tasks，这些 tasks 都注册到 driver 之上；
    - Horovod 的 DriverService 会 通知 TaskService 启动训练；

- MPI 如何在 Spark Executor 之上启动用户代码？

  - 答案   是：
    - 通常MPI 会通过 SSH 来连接 hosts，但是这种方式无法在 Spark Executor 之中启动 Python function。
    - 因此 MPI 使用 RPC 来启动用户代码，即使用 `horovod.spark.driver.mpirun_rsh` 来连接每个 Executor，然后 "remote shell" 到这些 executors 之中。

- 如何发布 训练代码？

  - **答案**是：SparkTaskService 可以 从 SparkDriverService 得到训练代码，因为是 python 脚本，所以可以直接通过 RPC 传输过来；

- Spark如何开始运行？当某一个 Executor 启动后就可以运行？还是需要所有的 Executor 都 ready 才能一起跑？

  - **答案**是：Hovorod 期望所有的 task 都同时运行，因此 cluster 应该至少提供同样个数的 core，每个 executor 可以有多个 core，因此一个 executor 可以处理多个 tasks，host 可以有多个 executor。

我们在一篇文章中会继续深入 Horovd on Spark。
