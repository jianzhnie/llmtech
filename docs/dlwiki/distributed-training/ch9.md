# 深度学习分布式训练框架 horovod (9) --- 启动 on spark

Horovod 是Uber于2017年发布的一个易于使用的高性能的分布式训练框架，在业界得到了广泛应用。

本系列将通过源码分析来带领大家了解 Horovod。这几篇介绍 horovod 如何运行在 spark 之上。本文是第九篇，介绍 horovod on spark 如何启动。

## 1 总体架构图

首先，我们还是要祭出架构图，这样大家可以按图索骥。

![img](https://img2020.cnblogs.com/blog/1850883/202107/1850883-20210703002655210-1363178167.png)

总体来说，Horovod on Spark 的总体逻辑分为以下阶段：

- 启动 SparkDriverService 服务，利用 _make_spark_thread 启动 Spark task，然后 horovod 会等待启动结束;
- 多线程在 spark executor 之中启动 spark task，每个task之中运行一个 SparkTaskService，SparkTaskService 会向 hovorod 主进程中的 SparkDriverTask 进行注册，并且等待下一步运行启动的指令；
- Horovod 收到所有 task 结束的信息之后，通知各个 task，进入下一阶段；
- Horovod 调用 mpi_run （又利用到 mpirun_rsh.py）在每一个 spark executor 上启动 orted 进程，以启动 MPI cluster；
- orted 在每一个 executor 之上运行训练代码；

我们下面就具体看看如何启动。

## 2 第一阶段 ：Horovod 启动

**本部分主要逻辑**是：启动 SparkDriverService 服务，利用 _make_spark_thread 启动 Spark task，然后 horovod 会等待启动结束。

### 2.1 Driver服务 ：SparkDriverService

SparkDriverService 继承了 driver_service.BasicDriverService，所以其内部启动了一个 socket server，可以进行网络交互。

Horovod 利用 SparkDriverService 来和 Spark executor（通过其中运行的SparkTaskService）交互，比如收集信息，让 spark 启动训练job等等。这是一个 RPC 机制。

具体 SparkDriverService 的功能可以参见其内部处理的各种 Request，比如

- CodeRequest ：SparkTaskService会用来请求用户代码；
- TaskHostHashIndicesRequest ：获取 task host 地址；
- TaskIndexByRankRequest ：从 rank 获取到 task index；
- SetLocalRankToRankRequest ：从 local rank 得到 rank 信息；
- WaitForTaskShutdownRequest ：等待 shutdown；

和前文介绍的 HorovodRunDriverService 有些类似。

其中，其成员变量 _fn 就是训练函数，以后当 SparkTaskService 请求代码的时候，就通过 CodeResponse 把 _fn 直接发送回去。这样就解决了代码发布问题。

```python
class SparkDriverService(driver_service.BasicDriverService):
    NAME = 'driver service'

    def __init__(self, initial_np, num_proc, fn, args, kwargs, key, nics):
        super(SparkDriverService, self).__init__(num_proc,
                                                 SparkDriverService.NAME,
                                                 key, nics)
        self._initial_np = initial_np
        self._fn = fn # 保存用户代码
        self._args = args # 用户参数
        self._kwargs = kwargs
        self._key = key
        self._nics = nics # 网卡信息
        self._ranks_to_indices = {}
        self._spark_job_failed = False
        self._lock = threading.Lock()
        self._task_shutdown = threading.Event()

    def _handle(self, req, client_address):

        if isinstance(req, TaskHostHashIndicesRequest): # 获取 task host 地址
            return TaskHostHashIndicesResponse(self._task_host_hash_indices[req.host_hash])

        if isinstance(req, SetLocalRankToRankRequest): # 从 local rank 得到 rank 信息
            self._lock.acquire()

            try:
                # get index for host and local_rank
                indices = self._task_host_hash_indices[req.host]
                index = indices[req.local_rank]

                values = list(self._ranks_to_indices.values())
                prev_pos = values.index(index) if index in values else None
                if prev_pos is not None:
                    prev_rank = list(self._ranks_to_indices.keys())[prev_pos]
                    del self._ranks_to_indices[prev_rank]

                # memorize rank's index
                self._ranks_to_indices[req.rank] = index
            finally:
                self._lock.release()
            return SetLocalRankToRankResponse(index)

        if isinstance(req, TaskIndexByRankRequest): # 是从 rank 获取到 task index
            self._lock.acquire()
            try:
                return TaskIndexByRankResponse(self._ranks_to_indices[req.rank])
            finally:
                self._lock.release()

        if isinstance(req, CodeRequest): # SparkTaskService会用来请求用户代码
            return CodeResponse(self._fn, self._args, self._kwargs)

        if isinstance(req, WaitForTaskShutdownRequest): # 等待任务结束
            self._task_shutdown.wait()
            return network.AckResponse()

        return super(SparkDriverService, self)._handle(req, client_address)
```

### 2.2 启动spark task : _make_spark_thread

在 Horovod.spark.run 之中，_make_spark_thread 建立了 thread。这里关键代码是：

```python
mapper = _make_mapper(driver.addresses(), settings, use_gloo, is_elastic)
result = procs.mapPartitionsWithIndex(mapper).collect()
```

mapPartitionsWithIndex 这句代码会促使 Spark 在多个 Executor 之中运行 mapper 函数，并且得到运行结果。

即创建 `settings.num_proc` 个 **Spark tasks**，每个 task 会运行 mapper（**_task_fn**）, 外部的 run 函数会等待这些执行结果。其实如果需要使用RDD，也许可以使用 `foreachPartition`，这样每个结点上将会在内存中持有RDD的一个分区。

```python
def _make_spark_thread(spark_context, spark_job_group, driver, result_queue,
                       settings, use_gloo, is_elastic):
    """Creates `settings.num_proc` Spark tasks in a parallel thread."""

    def run_spark():
        """Creates `settings.num_proc` Spark tasks, each executing `_task_fn` and waits for them to terminate."""
        try:
            spark_context.setJobGroup(spark_job_group, "Horovod Spark Run", interruptOnCancel=True)
            procs = spark_context.range(0, numSlices=settings.max_np if settings.elastic else settings.num_proc)
            # We assume that folks caring about security will enable Spark RPC encryption,
            # thus ensuring that key that is passed here remains secret.
            mapper = _make_mapper(driver.addresses(), settings, use_gloo, is_elastic)
            # 促使 Spark 在多个 Executor 之中运行 mapper 函数，并且得到运行结果
            result = procs.mapPartitionsWithIndex(mapper).collect()
            result_queue.put(result)
        except:
            driver.notify_spark_job_failed()
            raise

    spark_thread = in_thread(target=run_spark, daemon=False)
    return spark_thread
```

### 2.3 等待 spark task 启动结束

启动了 spark task 之后，horovod 主进程会调用如下来等待 task **全部** 启动完成。

```python
# wait for all tasks to register, notify them and initiate task-to-task address registration
_notify_and_register_task_addresses(driver, settings)
```

即，run 函数中，当 _make_spark_thread 之后，horovod 主进程调用 _notify_and_register_task_addresses，从而调用 driver.wait_for_initial_registration(settings.start_timeout) ，进行总体等待。

等待的内容是：等待所有 num_proc tasks 来注册。当所有 spark thread 都ready 之后，主 horovod 进程会继续运行。

![img](https://img2020.cnblogs.com/blog/1850883/202107/1850883-20210703003538793-1042709907.png)

#### 2.3.1 _notify_and_register_task_addresses

在 horovod 主进程之中，会使用 `_notify_and_register_task_addresses` 来等待这些 spark task 来注册，从而调用 driver.wait_for_initial_registration(settings.start_timeout) ，进行总体等待。

注意，同时发送注册请求之后， spark task 自己也调用 task.wait_for_initial_registration 等待 horovod 再通知下一阶段的启动。

而在horovod 主进程的 _notify_and_register_task_addresses 其实也很复杂：

- 调用 driver.wait_for_initial_registration 等待task来注册，需要等待 num_proc 个task；
- 利用 notify_and_register 注册task，并且通知各个 task 开始下一步；

具体代码如下：

```python
def _notify_and_register_task_addresses(driver, settings, notify=True):
    # wait for num_proc tasks to register
    # 等待task来注册，需要等待 num_proc 个task
    driver.wait_for_initial_registration(settings.start_timeout)

    def notify_and_register(index): # 注册task，并且通知各个 task 开始下一步
        task_client = task_service.SparkTaskClient(index,
                                                   driver.task_addresses_for_driver(index),
                                                   settings.key, settings.verbose)

        if notify:
            task_client.notify_initial_registration_complete()

        next_task_index = (index + 1) % settings.num_proc
        next_task_addresses = driver.all_task_addresses(next_task_index)
        task_to_task_addresses = task_client.get_task_addresses_for_task(next_task_index, next_task_addresses)
        driver.register_task_to_task_addresses(next_task_index, task_to_task_addresses)

    for index in driver.task_indices():
        in_thread(notify_and_register, (index,)) #在thread之中启动task

    driver.wait_for_task_to_task_address_updates(settings.start_timeout)
```

我们目前只能看其第一步 “等待注册”。

#### 2.3.2 driver.wait_for_initial_registration

在这里 SparkDriverSerivce 首先等待所有 spark executor 注册。

在 class BasicDriverService(network.BasicService): 有如下代码，可以看到，只有全部 _num_proc 注册完成，当所有 spark thread 都ready 之后，主 horovod 进程会继续运行。

这里关键是：`while len(self._all_task_addresses) < self._num_proc`就是等待 self._all_task_addresses 的数目达到 _num_proc。

```python
class BasicDriverService(network.BasicService):

  def wait_for_initial_registration(self, timeout):
      self._wait_cond.acquire()
      try:
          # 等待 self._all_task_addresses 的数目达到 _num_proc
          while len(self._all_task_addresses) < self._num_proc:
              self._wait_cond.wait(timeout.remaining())
              timeout.check_time_out_for('tasks to start')
      finally:
          self._wait_cond.release()
```

### 2.4 等待

关于等待代码，我们要做一下特殊说明，具体看图。

![img](https://img2020.cnblogs.com/blog/1850883/202107/1850883-20210703003910696-2129261972.png)

这里有两套 wait_for_initial_registration。可以认为是两套 barrier。

就是：

- **barrier 1** ：SparkDriverSerivce 等待所有 SparkTaskSerivce ready；
- **barrier 2** ：所有 SparkTaskSerivce 需要一起运行，所以 SparkTaskSerivce们 都在等待 barrier 2。SparkDriverSerivce 会通知 这些 SparkTaskSerivce 一起发动；

#### 2.3.1 Barrier 1 in Driver

在 run 函数中，当 _make_spark_thread 之后，horovod 主进程调用 _notify_and_register_task_addresses，从而调用 driver.wait_for_initial_registration(settings.start_timeout) ，进行总体等待。

等待的内容是：等待所有 num_proc tasks 来注册。当所有 spark thread 都ready 之后，主 horovod 进程会继续运行。这里关键是：

```
while len(self._all_task_addresses) < self._num_proc
```

就是等待 self._all_task_addresses 的数目达到 _num_proc。

```python
def wait_for_initial_registration(self, timeout):
    self._wait_cond.acquire()
    try:
        while len(self._all_task_addresses) < self._num_proc:
            self._wait_cond.wait(timeout.remaining())
            timeout.check_time_out_for('tasks to start')
    finally:
        self._wait_cond.release()
```

在 BasicDriverService 之中，如果收到了 spark executor 的注册请求就进行处理，这里最重要是：

```
self._all_task_addresses[req.index] = req.task_addresses
```

当所有的 spark executor 都注册了，这里就等待成功。

#### 2.3.2 Barrier 2 in task

每个 spark thread 在 _task_fn 之中运行，就是在 spark task 之中运行。这里也可以看出来是 Spark task 的一个总体流程：

- 首先 调用 `register_task`；
- 其次 调用 `task.wait_for_initial_registration(settings.start_timeout)` ；
- 然后 调用 `wait_for_command_termination` 来等待结束；

task.wait_for_initial_registration 会等待 self._initial_registration_complete = True 这个条件，就是等待 `register_task` 注册完成。

每个 Spark Executor 都有一个 SparkTaskService，所以 每个spark task 都有自己的 _initial_registration_complete。

hovorod.run 主进程会逐一通知每个 SparkTaskService 的 _initial_registration_complete。

即，哪个 SparkTaskService 好了，就通知哪个 SparkTaskService 的 _initial_registration_complete。这样，这个 SparkTaskService 就可以正式运行了。

#### 2.3.3 总体等待流程

总体等待流程具体如图，数字就是执行顺序：

1. SparkDriverSerivce 调用 driver.wait_for_initial_registration 来等待 SparkTaskSerivce 的注册，**这是 barrier 1**；
2. SparkTaskSerivce 1 进行注册，然后 SparkTaskSerivce 1 自己也调用 task.wait_for_initial_registration 等待 horovod 再通知下一阶段的启动，**这是 barrier 2**；
3. SparkTaskSerivce 2 进行注册，然后 SparkTaskSerivce 2 自己也调用 task.wait_for_initial_registration 等待 horovod 再通知下一阶段的启动，**这是 barrier 2**；
4. hovorod.run 主进程在发现所有 task 都注册之后，**barrier 1 等待结束**，会逐一通知每个 SparkTaskService 的 _initial_registration_complete。只有 4 完成之后，两个 SparkTaskSerivce 才能继续执行 5，6；
5. SparkTaskSerivce 1 对于 **barrier 2** 等待结束，继续执行；
6. SparkTaskSerivce 2 对于 **barrier 2** 等待结束，继续执行；

```python
    SparkTaskSerivce 1          SparkTaskSerivce 2            SparkDriverSerivce

            +                           +                             +
            |                           |                             |
            |                           |                             |
            |                           |                             |
            |                           |                             |   1
            |                           |                             |
            |                           |                             |
            |                           |                             v
            |                           |
            |                           |         +--------------------------------------+
            |                           |         | barrier 1                            |
            |                           |   2     |                                      |
            |          3                +-------> |                                      |
            |                           |         |                                      |
            +-----------------------------------> | driver.wait_for_initial_registration |
            |                           |         |                                      |
            |                           |         |                                      |
            |                           |         |                                      |
            |                           |         +--------------------+-----------------+
            |                           |                              |
            |                           |                              |
+-----------+----------------------+    |                  4           |
|barrier 2                         | <---------------------------------+
|                                  |    |                              |
|task.wait_for_initial_registration|    |                              |
|                                  |    |                              |
+-----------+----------------------+    |                              |
            |                           |                              |
            |             +-------------+----------------------+       |
            |             | barrier 2                          |   4   |
            | 6           |                                    +<------+
            |             | task.wait_for_initial_registration |       |
            |             |                                    |       |
            |             +-------------+----------------------+       |
            |                           |                              |
            |                           |                              |
            |                           |  5                           |
            |                           |                              |
            v                           v                              v
```

我们接下来详细介绍 task 启动内容 和 driver 后续工作。

## 3 第二阶段 ：Spark Task 启动

本阶段我们详细介绍下 Spark Task 的启动过程。

这部分主要功能是：多线程在 spark executor 之中启动 spark task，每个spark task会运行`_task_fn`函数，`_task_fn`函数会运行一个 SparkTaskService，SparkTaskSerivce 会向 hovorod 主进程中的 SparkDriverTask 进行注册，并且等待下一步运行启动的指令；

此时程序（不是训练程序，而是 SparkTaskService）已经在 Spark Executor内部运行了。我们看看在 spark Executor 之中，是如何启动运行 SparkTaskService 的。

### 3.1 具体spark启动逻辑 ：_task_fn

Horovod 在 thread 里面通过 _make_mapper 来让 Spark 运行 _task_fn。

```python
def _make_mapper(driver_addresses, settings, use_gloo, is_elastic):

    def _mapper(index, _):
        yield _task_fn(index, driver_addresses, key, settings, use_gloo, is_elastic)

    return _mapper
```

`_task_fn` 的作用是为了注册 horovod 进入到 spark task。即，在每一个 spark task （executor） 之中启动一个 SparkTaskService。

一定要注意：这些 SparkTaskService 是运行在 spark executor 之中，通过网络与 horovod 之中的 SparkDriverService 交互。

可以看到，_task_fn 的总体逻辑是：

- 启动 SparkTaskService；
- 通过 driver_service.SparkDriverClient.register_task 来向 horovod 中的 Driver 注册；
- 通过 task.wait_for_initial_registration(settings.start_timeout) 来等待下一步启动的开始指示；
- 如果下一步开始启动了，则调用 task.wait_for_command_termination() 等待结束；

具体如下：

```python
def _task_fn(index, driver_addresses, key, settings, use_gloo, is_elastic):
    settings.key = key
    hosthash = host_hash(salt='{}-{}'.format(index, time.time()) if is_elastic else None)
    os.environ['HOROVOD_HOSTNAME'] = hosthash
    # 启动 SparkTaskService，SparkTaskService本身包括一个socket server，可以和driver交互
    task = task_service.SparkTaskService(index, settings.key, settings.nics,...)
    try:
        driver_client = driver_service.SparkDriverClient(driver_addresses, settings.key, settings.verbose)
        # 向 horovod 中的 Driver 注册
        driver_client.register_task(index, task.addresses(), hosthash)

        # 这里依然运行在spark task之中，但因为不是SparkTaskService，所以只是做协助工作，最后静静等待
        if not is_elastic:
            # 等待下一步启动的开始指示
            task.wait_for_initial_registration(settings.start_timeout)
            task_indices_on_this_host = driver_client.task_host_hash_indices(hosthash)
            local_rank_zero_index = task_indices_on_this_host[0]
        else:
            local_rank_zero_index = None

        if is_elastic:
						...... # 后续文章会介绍
        elif use_gloo or index == local_rank_zero_index:
            # Either Gloo or first task with MPI.
            # 使用Gloo或者使用MPI的第一个task，让这个task做操作
            task.wait_for_command_start(settings.start_timeout)
            # 等待结束
            task.wait_for_command_termination()
        else:
            # The other tasks with MPI need to wait for the first task to finish.
            # 让其他的task等待第一个task结束
            first_task_addresses = driver_client.all_task_addresses(local_rank_zero_index)
            first_task_client = \
                task_service.SparkTaskClient(local_rank_zero_index,
                                             first_task_addresses, settings.key,
                                             settings.verbose)
            # 调用 task.wait_for_command_termination() 等待结束
            first_task_client.wait_for_command_termination()

        return task.fn_result()
    finally:
        task.shutdown()
```

### 3.2 SparkTaskService

再次强调如下代码：

```
task = task_service.SparkTaskService(index, settings.key, settings.nics,...)
```

每一个_task_fn 中都定义了一个 SparkTaskService，即每一个 Spark Executor 都会生成一个（或者多个） SparkTaskService，在 spark task 之中运行并且作用。

#### 3.2.1 SparkTaskService 定义

SparkTaskService 定义如下，因为继承了BasicTaskService，所以其内部最终也会启动一个 socket server，以便同 horovod 中的 SparkDriverService 交互：

```python
class SparkTaskService(task_service.BasicTaskService):
    NAME_FORMAT = 'task service #%d'

    def __init__(self, index, key, nics, minimum_command_lifetime_s, verbose=0):
        # on a Spark cluster we need our train function to see the Spark worker environment
        # this includes PYTHONPATH, HADOOP_TOKEN_FILE_LOCATION and _HOROVOD_SECRET_KEY
        env = os.environ.copy()

        # we inject the secret key here
        env[secret.HOROVOD_SECRET_KEY] = codec.dumps_base64(key)

        # we also need to provide the current working dir to mpirun_exec_fn.py
        env['HOROVOD_SPARK_WORK_DIR'] = os.getcwd()

        super(SparkTaskService, self).__init__(SparkTaskService.NAME_FORMAT % index,
                                               index, key, nics, env, verbose)
        self._key = key
        self._minimum_command_lifetime_s = minimum_command_lifetime_s
        self._minimum_command_lifetime = None
```

#### 3.2.2 基本功能

SparkTaskService 的基本功能如下。

- _run_command 将会被用来在 spark 之中启动训练job；
- _handle 会处理 GetTaskToTaskAddressesRequest，用来获取 task 地址，也会处理ResourcesRequest，返回资源；
- _get_resources 将返回 spark 资源；
- wait_for_command_termination 会等待命令执行结束；

具体代码如下：

```python
def _run_command(self, command, env, event,
                 stdout=None, stderr=None, index=None,
                 prefix_output_with_timestamp=False):
    # 在 spark 之中启动训练job
    super(SparkTaskService, self)._run_command(command, env, event,
                                               stdout, stderr, index,
                                               prefix_output_with_timestamp)

    if self._minimum_command_lifetime_s is not None:
        self._minimum_command_lifetime = timeout.Timeout(self._minimum_command_lifetime_s,
                                                         message='Just measuring runtime')

def _handle(self, req, client_address):
    # 返回资源
    if isinstance(req, ResourcesRequest):
        return ResourcesResponse(self._get_resources())

    # 获取 task 地址
    if isinstance(req, GetTaskToTaskAddressesRequest):
        next_task_index = req.task_index
        next_task_addresses = req.all_task_addresses
        # We request interface matching to weed out all the NAT'ed interfaces.
        next_task_client = \
            SparkTaskClient(next_task_index, next_task_addresses,
                            self._key, self._verbose,
                            match_intf=True)
        return GetTaskToTaskAddressesResponse(next_task_client.addresses())

    return super(SparkTaskService, self)._handle(req, client_address)

def _get_resources(self):
    # 返回 spark 资源
    if LooseVersion(pyspark.__version__) >= LooseVersion('3.0.0'):
        task_context = pyspark.TaskContext.get()
        if task_context:
            return task_context.resources()
        else:
            print("Not running inside Spark worker, no resources available")
    return dict()

def wait_for_command_termination(self):
    """
    Waits for command termination. Ensures this method takes at least
    self._minimum_command_lifetime_s seconds to return after command started.
    """
    try:
        # 等待命令执行结束
        return super(SparkTaskService, self).wait_for_command_termination()
    finally:
        # command terminated, make sure this method takes at least
        # self._minimum_command_lifetime_s seconds after command started
        # the client that started the command needs some time to connect again
        # to wait for the result (see horovod.spark.driver.rsh).
        if self._minimum_command_lifetime is not None:
            time.sleep(self._minimum_command_lifetime.remaining())
```

### 3.3 注册Task

下一步代码就是用来向 Driver 注册 本 task。

```python
driver_client.register_task(index, task.addresses(), hosthash)
```

#### 3.3.1 发送注册请求

注册具体通过如下完成，这里调用了 network.py 的 _send 函数，就是通过 socket，spark executor 和 horovod driver 进行了网络交互：

```python
class BasicDriverClient(network.BasicClient):

    def register_task(self, index, task_addresses, host_hash):
        self._send(RegisterTaskRequest(index, task_addresses, host_hash))
```

#### 3.3.2 Driver处理

我们先来到 Horovod 中运行的 Driver来看看（**下一节内容，这里提前看看**）。

在 BasicDriverService 之中，如果收到了RegisterTaskRequest请求就进行处理，这里最重要是：

```
self._all_task_addresses[req.index] = req.task_addresses
```

这样，self._all_task_addresses 的数目就增加了。

而我们之前提到了，horovod 正在 driver.wait_for_initial_registration 上面等待，其关键是：

```
while len(self._all_task_addresses) < self._num_proc
```

如果`self._all_task_addresses` 的数目达到了`_num_proc`，driver.wait_for_initial_registration 就结束了，就顺利执行。

具体处理 RegisterTaskRequest 的代码如下，BasicDriverService 之中有各种成员变量，用来维护各种所需信息，我们在前文 [原创 [源码解析\] 深度学习分布式训练框架 horovod (4) --- 网络基础 & Driver](https://www.cnblogs.com/rossiXYZ/p/14882053.html) 中已经详细讲解过，_handle函数的RegisterTaskRequest 处理就是用来更新这些成员变量：

```python
class BasicDriverService(network.BasicService):

    def _handle(self, req, client_address):
        if isinstance(req, RegisterTaskRequest):
            self._wait_cond.acquire()
            try:

                self._all_task_addresses[req.index] = req.task_addresses
                # Just use source address for service for fast probing.
                self._task_addresses_for_driver[req.index] = \
                    self._filter_by_ip(req.task_addresses, client_address[0])

                # Remove host hash earlier registered under this index.
                if req.index in self._task_index_host_hash:
                    earlier_host_hash = self._task_index_host_hash[req.index]
                    if earlier_host_hash != req.host_hash:
                        self._task_host_hash_indices[earlier_host_hash].remove(req.index)

                # Make index -> host hash map.
                self._task_index_host_hash[req.index] = req.host_hash

                # Make host hash -> indices map.
                if req.host_hash not in self._task_host_hash_indices:
                    self._task_host_hash_indices[req.host_hash] = []
                self._task_host_hash_indices[req.host_hash].append(req.index)
                # TODO: this sorting is a problem in elastic horovod
                self._task_host_hash_indices[req.host_hash].sort()
            finally:
                self._wait_cond.notify_all()
                self._wait_cond.release()

            return network.AckResponse()
```

### 3.4 Task 等待下一步通知

前面提到了，当 spark task 向 driver 发送注册请求之后，Spark task 通过 task.wait_for_initial_registration(settings.start_timeout) 来等待下一步启动的开始指示。就是 driver 认为你一景注册完成了，可以开始进入下一步了。

task.wait_for_initial_registration 会等待 self._initial_registration_complete = True 这个条件，就是等待 `register_task` 注册完成。

```python
class BasicTaskService(network.BasicService):

  def wait_for_initial_registration(self, timeout):
        self._wait_cond.acquire()
        try:
            while not self._initial_registration_complete:
                self._wait_cond.wait(timeout.remaining())
                timeout.check_time_out_for('tasks to start')
        finally:
            self._wait_cond.release()
```

每个 Spark Executor 都有一个 SparkTaskService，所以 每个spark task 都有自己的 _initial_registration_complete。

hovorod.run 主进程会逐一通知每个 SparkTaskService 的 _initial_registration_complete。即，哪个 SparkTaskService 好了，就通知哪个 SparkTaskService 的 _initial_registration_complete。

hovorod.run 主进程 是通过发送 NotifyInitialRegistrationCompleteRequest完成这一步的。

```python
def notify_initial_registration_complete(self):
    self._send(NotifyInitialRegistrationCompleteRequest())
```

BasicTaskService 在等待 NotifyInitialRegistrationCompleteRequest，如果收到了，就设置为 True，这样wait_for_initial_registration 就等待结束了。

```python
if isinstance(req, NotifyInitialRegistrationCompleteRequest):
    self._wait_cond.acquire()
    try:
        self._initial_registration_complete = True
    finally:
        self._wait_cond.notify_all()
        self._wait_cond.release()
    return network.AckResponse()
```

就说明当本 thread 注册在 horovod 之后，就算本 spark thread 启动成功了。

```python
+-------------------------------------+             +----------------------------------------------------+
| Horovod Main thread                 |             | Spark Executor                                     |
|                                     |             |                     _task_fn                       |
|                                     |             |                        +                           |
|                                     |             |                        |                           |
|                                     |             |                        |                           |
|                                     |             |                        v                           |
| +-------------------------------+   |             |  +---------------------+------------------------+  |
| | SparkDriverService            |   |             |  | SparkTaskService                             |  |
| |                               |   |             |  |               +                              |  |
| |                               |   |  1 register |  |               |                              |  |
| |  self._all_task_addresses <----------------------------------------+                              |  |
| |                               |   |             |  |               |                              |  |
| |              +                |   |             |  |               |                              |  |
| |              |                |   |             |  |               |                              |  |
| |              | 3              |   |             |  |               |                              |  |
| |              |                |   |             |  |               | 2                            |  |
| |              v                |   |             |  |               |                              |  |
| |  self._wait_cond.notify_all() |   |             |  |               |                              |  |
| |              +                |   |             |  |               v                              |  |
| |              |                |   |             |  |     +---------+---------------------------+  |  |
| |              |                |   |             |  |     |                                     |  |  |
| |              |                |   |             |  |     | task.wait_for_initial_registration  |  |  |
| |              |                |   |             |  |     |                                     |  |  |
| |              |                |   |             |  |     +-------------------------------------+  |  |
| |              |                |   |             |  |                                              |  |
| |              |                |   |             |  |                                              |  |
| |              |                |   |             |  |                                              |  |
| |              |                |   |             |  |                                              |  |
| |              |                |   |             |  |                                              |  |
| |              |                |   |             |  |                                              |  |
| |              |                |   |             |  |                                              |  |
| |              |                |   |             |  |                                              |  |
| |              |                |   |             |  |                                              |  |
| |              |                |   |             |  |                                              |  |
| |              |                |   |             |  |                                              |  |
| |              v                |   |             |  |                                              |  |
| |                               |   |             |  |                                              |  |
| |                               |   |             |  |                                              |  |
| |                               |   |             |  |                                              |  |
| +-------------------------------+   |             |  +----------------------------------------------+  |
+-------------------------------------+             +----------------------------------------------------+
```

手机如下：

![img](https://img2020.cnblogs.com/blog/1850883/202107/1850883-20210703002809443-531228556.png)

## 4 第三阶段：Driver 通知 task 注册成功

本阶段的作用是：Horovod 收到所有 task 结束的信息之后，通知各个 task，进入下一阶段。

### 4.1 _notify_and_register_task_addresses

前面提到。在 horovod 主进程之中，会使用 `_notify_and_register_task_addresses` 来等待这些 spark task 来注册，从而调用 driver.wait_for_initial_registration(settings.start_timeout) ，进行总体等待。

注意，同时发送注册请求之后， spark task 自己也调用 task.wait_for_initial_registration 等待horovod 再通知下一阶段的启动。

而 _notify_and_register_task_addresses 中其实也很复杂：

- 调用 driver.wait_for_initial_registration 等待task来注册；（**目前这一步已经完成**）
- 利用 notify_and_register 注册task，并且通知各个 task 开始下一步；（**我们这里进入后面这两步**）
- 利用 driver.wait_for_task_to_task_address_updates 再次确认下所有 task 都OK；

```python
def _notify_and_register_task_addresses(driver, settings, notify=True):
    # wait for num_proc tasks to register
    driver.wait_for_initial_registration(settings.start_timeout)

    def notify_and_register(index):
        # 注册task，并且通知各个 task 开始下一步
        task_client = task_service.SparkTaskClient(index,
                                                   driver.task_addresses_for_driver(index),
                                                   settings.key, settings.verbose)

        if notify:
            task_client.notify_initial_registration_complete()

        next_task_index = (index + 1) % settings.num_proc
        next_task_addresses = driver.all_task_addresses(next_task_index)
        task_to_task_addresses = task_client.get_task_addresses_for_task(next_task_index, next_task_addresses)
        driver.register_task_to_task_addresses(next_task_index, task_to_task_addresses)

    for index in driver.task_indices():
        in_thread(notify_and_register, (index,)) # 注册task，并且通知各个 task 开始下一步

    # 再次确认下所有 task 都OK
    driver.wait_for_task_to_task_address_updates(settings.start_timeout)
```

### 4.2 notify_and_register

可以看到 notify_and_register 的作用就是：

- 调用 task_client.notify_initial_registration_complete() 通知 spark task 注册成功了，这样就让所有等待 task.wait_for_initial_registration 的 spark executor 一起运行下一阶段。
- 调用 driver.register_task_to_task_addresses(next_task_index, task_to_task_addresses) 来让 Driver 完成注册。

```python
def wait_for_task_to_task_address_updates(self, timeout):
    self._wait_cond.acquire()
    try:
        while len(self._task_addresses_for_tasks) < self._initial_np:
            self.check_for_spark_job_failure()
            self._wait_cond.wait(timeout.remaining())
            timeout.check_time_out_for('Spark tasks to update task-to-task addresses')
    finally:
        self._wait_cond.release()
```

### 4.3 wait_for_task_to_task_address_updates

这里会再次确认所有 spark task 都OK。

```python
def wait_for_task_to_task_address_updates(self, timeout):
    self._wait_cond.acquire()
    try:
        while len(self._task_addresses_for_tasks) < self._initial_np:
            self.check_for_spark_job_failure()
            self._wait_cond.wait(timeout.remaining())
            timeout.check_time_out_for('Spark tasks to update task-to-task addresses')
    finally:
        self._wait_cond.release()
```

### 4.4 等待 In Task

在 Spark task 之中，如果收到了下一步启动指示之后，会调用 wait_for_command_termination 进行等待。

其实，这一步也就意味着 spark exector 自己本身的逻辑任务结束了，因为以后都是 SparkTaskService 自己独立完成的动作，它来负责训练代码的启动。既然 `_task_fn` 的逻辑任务已经结束，那么静静地等待即可。

#### 4.4.1 wait_for_command_termination

在 horovod-master/horovod/spark/task/task_service.py

```python
def wait_for_command_termination(self):
    """
    Waits for command termination. Ensures this method takes at least
    self._minimum_command_lifetime_s seconds to return after command started.
    """
    try:
        return super(SparkTaskService, self).wait_for_command_termination()
    finally:
        # command terminated, make sure this method takes at least
        # self._minimum_command_lifetime_s seconds after command started
        # the client that started the command needs some time to connect again
        # to wait for the result (see horovod.spark.driver.rsh).
        if self._minimum_command_lifetime is not None:
            time.sleep(self._minimum_command_lifetime.remaining())
```

在 horovod-master/horovod/runner/common/service/task_service.py 中可以看到，就是等待训练代码所在的 thread 结束。

```python
def wait_for_command_termination(self):
    self._command_thread.join() # 马上会说明
```

#### 4.4.2 _command_thread

这里对 _command_thread 略作说明。

在 SparkTaskService 处理 RunCommandRequest 时候，运行 Command 的 thread 就是被赋值为 _command_thread。

```python
class BasicTaskService(network.BasicService):
    def _handle(self, req, client_address):

        if isinstance(req, RunCommandRequest): # 运行命令请求
            self._wait_cond.acquire()
            try:
                if self._command_thread is None:

                    if self._command_env:
                        env = self._command_env.copy()
                        self._add_envs(env, req.env)
                        req.env = env

                    self._command_abort = threading.Event()
                    self._command_stdout = Pipe() if req.capture_stdout else None
                    self._command_stderr = Pipe() if req.capture_stderr else None
                    # 配置各种参数信息
                    args = (req.command, req.env, self._command_abort,
                            self._command_stdout, self._command_stderr,
                            self._index,
                            req.prefix_output_with_timestamp)
                    # 启动一个新线程来运行命令
                    self._command_thread = in_thread(self._run_command, args)
            finally:
                self._wait_cond.notify_all()
                self._wait_cond.release()
            return network.AckResponse()
```

逻辑如下：

```python
+-------------------------------------+             +----------------------------------------------------+
| Horovod Main thread                 |             | Spark Executor                                     |
|                                     |             |                     _task_fn                       |
|                                     |             |                        +                           |
|                                     |             |                        |                           |
|                                     |             |                        |                           |
|                                     |             |                        v                           |
| +-------------------------------+   |             |  +---------------------+------------------------+  |
| | SparkDriverService            |   |             |  | SparkTaskService                             |  |
| |                               |   |             |  |               +                              |  |
| |                               |   |  1 register |  |               |                              |  |
| |  self._all_task_addresses <----------------------------------------+                              |  |
| |                               |   |             |  |               |                              |  |
| |              +                |   |             |  |               |                              |  |
| |              |                |   |             |  |               |                              |  |
| |              | 3              |   |             |  |               |                              |  |
| |              |                |   |             |  |               | 2                            |  |
| |              v                |   |             |  |               |                              |  |
| |  self._wait_cond.notify_all() |   |             |  |               |                              |  |
| |              +                |   |             |  |               v                              |  |
| |              |                |   +             +  +     +---------+---------------------------+  |  |
| |              |            4   |  RegistrationComplete    |                                     |  |  |
| |              |  +-----------------+-------------+--+---> | task.wait_for_initial_registration  |  |  |
| |              |                |   |             |  |     |                                     |  |  |
| |              |                |   |             |  |     +---------+---------------------------+  |  |
| |              |                |   |             |  |               |                              |  |
| |              |                |   |             |  |               |                              |  |
| |              |                |   |             |  |               | 5                            |  |
| |              |                |   |             |  |               |                              |  |
| |              |                |   |             |  |               |                              |  |
| |              |                |   |             |  |               v                              |  |
| |              |                |   |             |  |        wait_for_command_termination          |  |
| |              |                | 6 |  RunCommand |  |               +                              |  |
| |              |                |   |             |  |               |                              |  |
| |              +----------------------------------------------->     | 7                            |  |
| |              |                |   |             |  |               v                              |  |
| |              v                |   |             |  |        self._command_thread.join()           |  |
| |                               |   |             |  |                                              |  |
| |                               |   |             |  |                                              |  |
| |                               |   |             |  |                                              |  |
| +-------------------------------+   |             |  +----------------------------------------------+  |
+-------------------------------------+             +----------------------------------------------------+
```

手机如下：

![img](https://img2020.cnblogs.com/blog/1850883/202107/1850883-20210703002851120-473073745.png)

至此，第一阶段完成，我们下一篇继续，敬请期待。

## 5 总结

总体来说，Horovod on Spark 的总体逻辑分为以下阶段：

- 启动 SparkDriverService 服务，利用 _make_spark_thread 启动 Spark task，然后 horovod 会等待启动结束;
- 多线程在 spark executor 之中启动 spark task，每个task之中运行一个 SparkTaskService，SparkTaskService 会向 hovorod 主进程中的 SparkDriverTask 进行注册，并且等待下一步运行启动的指令；
- Horovod 收到所有 task 结束的信息之后，通知各个 task，进入下一阶段；
- Horovod 调用 mpi_run （又利用到 mpirun_rsh.py）在每一个 spark executor 上启动 orted，以启动 MPI cluster；
- orted 在每一个 executor 之上运行训练代码；

本文介绍了前三个阶段，即启动阶段。下文介绍后续两个阶段，敬请期待。
