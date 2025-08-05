# `verl.single_controller`设计详解

## 前言

本文档旨在为参与开发`verl`项目的开发者提供关于`verl.single_controller`模块的深入理解。它特别适用于希望了解或参与到该模块开发中的开源贡献者，而非终端用户。文档的核心目标是阐明架构原理及其内部工作机制。

## 起源

`single_controller`模块的设计初衷是为了将单进程强化学习（RLHF）实验脚本转化为分布式系统，同时尽可能减少代码改动并保持调试的便捷性。

传统解决方案如使用PyTorch的分布式数据并行（DDP）通常需要封装`nn.Module`并在多进程中执行相同的函数。然而，在分布式RLHF环境中，这种方法面临两大挑战：难以表达PPO算法所需的复杂DAG结构，以及在训练过程中难以检查中间张量。

为了维持良好的可调试性，我们采取了不同的策略——将训练循环划分为明确的阶段，例如`generate_sequences`和`compute_advantages`等。

选择[Ray](https://www.ray.io/)作为`verl`的初始后端，主要是因为它能够将Python类方法暴露为RPC端点。尽管Ray默认支持单方法调用对应单次RPC的模型，但大型语言模型（LLMs）的训练通常要求多进程协作。为此，我们引入了以下组件以隐藏这种复杂性：

- `WorkerGroup`：管理远程工作节点组，提供统一的多进程分布式计算接口；
- `ResourcePool`：将计算资源绑定到工作进程；
- `ClassWithArgs`：支持带初始化参数的延迟远程实例化。

## 运行示例：`generate_sequences`

我们将通过`generate_sequences`阶段中`ActorRolloutRefWorker`类的方法演示如何在分布式工作节点间完成注册与调用。

### 第一步：使用装饰器注册

首先定义`generate_sequences`方法，并使用`@register`装饰器进行标记，以便在驱动脚本中被调用。

**源码文件：**[fsdp_workers.py](https://github.com/volcengine/verl/blob/c59ab2f4788f9a910836a9f2f53dcdb62dfa314e/verl/workers/fsdp_workers.py#L528)

```python
class ActorRolloutRefWorker(Worker):
    ...
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        prompts = prompts.to(torch.cuda.current_device())
        ...
```

`@register`装饰器为`generate_sequences`方法添加元数据，虽然当前实现不改变其功能逻辑，但它会通过特定键值（`MAGIC_ATTR`）附加属性字段。

**来源：**[decorator.py](https://github.com/volcengine/verl/blob/c59ab2f4788f9a910836a9f2f53dcdb62dfa314e/verl/single_controller/base/decorator.py#L411)

```python
def register(dispatch_mode=Dispatch.ALL_TO_ALL, execute_mode=Execute.ALL, blocking=True, materialize_futures=True):
    ...
    def decorator(func):
        @wraps(func)
        def inner(*args, **kwargs):
            if materialize_futures:
                args, kwargs = _materialize_futures(*args, **kwargs)
            return func(*args, **kwargs)

        attrs = {"dispatch_mode": dispatch_mode, "execute_mode": execute_mode, "blocking": blocking}
        setattr(inner, MAGIC_ATTR, attrs)
        return inner

    return decorator
```

上述代码展示了如何将`dispatch_mode`、`execute_mode`和`blocking`等参数值附加到`generate_sequences`方法上。

`register` 函数是一个装饰器，用于为分布式计算方法添加元数据配置。 它允许开发者指定方法在分布式环境中的执行模式、数据分发策略和阻塞行为。

- **dispatch_mode**: 数据分发模式，默认为 `Dispatch.ALL_TO_ALL`，控制数据如何分发到各个 Worker
- **execute_mode**: 执行模式，默认为 `Execute.ALL`，控制方法在哪些 Worker 上执行
- **blocking**: 是否阻塞执行，默认为 `True`，控制是否等待远程执行完成
- **materialize_futures**: 是否物化 Future 对象，默认为 `True`，在执行前解析异步对象

#### 装饰器实现 decorator.py:509-527

装饰器的核心逻辑包括：

1. **同步函数包装器（第510-514行）**: 创建 `inner` 函数处理同步方法调用
2. **异步函数包装器（第516-520行）**: 创建 `async_inner` 函数处理异步方法调用
3. **函数类型检测（第522行）**: 使用 `inspect.iscoroutinefunction()` 判断原函数是否为协程
4. **元数据附加（第523-524行）**: 将配置参数作为属性附加到包装函数上

#### 魔法属性机制 decorator.py:22-23

使用 `MAGIC_ATTR = "attrs_3141562937"` 作为特殊属性名，避免与用户定义的属性冲突。装饰器将配置信息存储在这个属性中。

#### 技术要点

##### 装饰器模式和元编程

`register` 函数采用装饰器模式，通过元编程技术在运行时为方法添加分布式执行能力。它不改变原函数的核心逻辑，而是添加元数据供后续的方法绑定过程使用。

##### Future 对象物化机制 decorator.py:470-482

`_materialize_futures` 函数处理 `DataProtoFuture` 对象的物化，确保在分发数据前所有异步对象都已解析完成。

##### 分发模式系统 decorator.py:26-53

支持多种预定义的分发模式，如 `DP_COMPUTE_PROTO`（数据并行计算）、`MEGATRON_COMPUTE`（Megatron 3D 并行）等，每种模式对应不同的数据分发和收集策略。

##### 方法绑定集成

装饰器的元数据会在 WorkerGroup 初始化时被提取和使用。 decorator.py:421-422 通过 `get_predefined_dispatch_fn` 函数获取对应的分发和收集函数。

### 第二步：初始化时绑定

当封装在 `RayClassWithInitArgs` 中的 `ActorRolloutRefWorker` 被传递给 `RayWorkerGroup` 时，这些附加属性会被提取并利用。

**源码文件：**[main_generation.py](https://github.com/volcengine/verl/blob/4ae9a0fdab229f75f080e9478807783ed4c97154/verl/trainer/main_generation.py#L82)

```python
ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role="rollout")

resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)

wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
```

在 `RayWorkerGroup` 的 [初始化过程](https://github.com/volcengine/verl/blob/c59ab2f4788f9a910836a9f2f53dcdb62dfa314e/verl/single_controller/ray/base.py#L184) 中，会执行两个关键步骤：

1.  创建工作节点实例（Ray actors）： [RayWorkerGroup._init_with_resource_pool](https://github.com/volcengine/verl/blob/c59ab2f4788f9a910836a9f2f53dcdb62dfa314e/verl/single_controller/ray/base.py#L211)
2.  将带有 `@register` 装饰器的方法绑定到 `RayWorkerGroup`： [RayWorkerGroup._bind_worker_method](https://github.com/volcengine/verl/blob/c59ab2f4788f9a910836a9f2f53dcdb62dfa314e/verl/single_controller/ray/base.py#L214)

[initialization_and_binding_of_worker_group](https://github.com/eric-haibin-lin/verl-community/blob/main/docs/worker_group_init.png?raw=true)

<img src="https://github.com/eric-haibin-lin/verl-community/blob/main/docs/worker_group_init.png?raw=true" alt="initialization_and_binding_of_worker_group" style="zoom:50%;" />

*WorkerGroup的初始化与绑定*

绑定过程是 `verl.single_controller` 的核心所在。

**关键函数：**[WorkerGroup._bind_worker_method](https://github.com/volcengine/verl/blob/c59ab2f4788f9a910836a9f2f53dcdb62dfa314e/verl/single_controller/base/worker_group.py#L143)

```python
def _bind_worker_method(self, user_defined_cls, func_generator):
    ...
    for method_name in dir(user_defined_cls):
        try:
            method = getattr(user_defined_cls, method_name)
            assert callable(method)
        except Exception:
            continue  # Skip properties
```

当方法具有 `MAGIC_ATTR` 属性时，`@register` 装饰器设置的属性将被提取：

```python
if hasattr(method, MAGIC_ATTR):
    attribute = getattr(method, MAGIC_ATTR)
    dispatch_mode = attribute["dispatch_mode"]
    execute_mode = attribute["execute_mode"]
    blocking = attribute["blocking"]

```

如上流程图所示，这些属性会被输入到 `func_generator` 中。但 `func_generator` 需要接收 `method_name`、`dispatch_fn`、`collect_fn`、`execute_fn` 和 `blocking` 参数。我们需要从 [DISPATCH_MODE_FN_REGISTRY](https://github.com/volcengine/verl/blob/c59ab2f4788f9a910836a9f2f53dcdb62dfa314e/verl/single_controller/base/decorator.py#L387) 中根据 `dispatch_mode`（`DP_COMPUTE_PROTO`）查找对应的 `dispatch_fn` 和 `collect_fn`：

```python
DISPATCH_MODE_FN_REGISTRY = {
    Dispatch.ONE_TO_ALL: {
        "dispatch_fn": dispatch_one_to_all,
        "collect_fn": collect_all_to_all,
    },
    ...
    Dispatch.DP_COMPUTE_PROTO: {
        "dispatch_fn": dispatch_dp_compute_data_proto,
        "collect_fn": collect_dp_compute_data_proto,
    },
    ...
}
```

同理，`execute_fn` 由 `execute_mode` 选择并通过以下方式提取：

```python
# get execute_fn_name
execute_mode = get_predefined_execute_fn(execute_mode=execute_mode)
wg_execute_fn_name = execute_mode["execute_fn_name"]

# get execute_fn from string
try:
    execute_fn = getattr(self, wg_execute_fn_name)
    assert callable(execute_fn), "execute_fn must be callable"
except Exception:
    print(f"execute_fn {wg_execute_fn_name} is invalid")
    raise
```

在此 `generate_sequences` 案例中：
- `dispatch_mode = Dispatch.DP_COMPUTE_PROTO`
- `dispatch_fn = dispatch_dp_compute_data_proto`
- `collect_fn = collect_dp_compute_data_proto`
- `execute_fn = RayWorkerGroup.execute_all`

#### `ONE_TO_ALL` 对比 `DP_COMPUTE_PROTO`

`dispatch_mode` 关联着一个 `dispatch_fn` 和 `collect_fn`。顾名思义，`dispatch_fn` 处理 `WorkerGroup` 中的输入参数并生成批量（列表）输入参数，每个参数将被传递给附加的工作线程 `WorkerGroup`（工作群组）。

`dispatch_fn`（分发函数）在 `ONE_TO_ALL`（一对多）模式下的实现是 [dispatch_one_to_all](https://github.com/volcengine/verl/blob/c59ab2f4788f9a910836a9f2f53dcdb62dfa314e/verl/single_controller/base/decorator.py#L119)（一对多分发），该函数简单地将所有输入参数复制为 N 份副本，其中 N 等于附加到 `worker_group`（工作群组）的 Worker 数量：

```python
def dispatch_one_to_all(worker_group, *args, **kwargs):
    args = tuple([arg] * worker_group.world_size for arg in args)
    kwargs = {k: [v] * worker_group.world_size for k, v in kwargs.items()}
    return args, kwargs
```

`dispatch_fn` 是 `DP_COMPUTE_PROTO` 的 [dispatch_dp_compute_data_proto](https://github.com/volcengine/verl/blob/c59ab2f4788f9a910836a9f2f53dcdb62dfa314e/verl/single_controller/base/decorator.py#L350)，它使用 `DataProto.chunk` 将大型 `DataProto` 分割为 N 个较小的 `DataProto`，其中 N 等于 `worker_group` 的 `world_size`（工作节点数量）：

```python
def dispatch_dp_compute_data_proto(worker_group, *args, **kwargs):
    from verl.single_controller.base.worker_group import WorkerGroup

    assert isinstance(worker_group, WorkerGroup)
    # Note: enable auto padding for dp compute DatapProto
    splitted_args, splitted_kwargs = _split_args_kwargs_data_proto_with_auto_padding(
        worker_group.world_size,
        *args,
        **kwargs,
    )
    return splitted_args, splitted_kwargs
```

`collect_fn` 遵循相同模式，处理来自 `WorkerGroup` 所有工作节点返回值的批次（列表），并将其合并为一个列表（如 `collect_all_to_all` 所做）或一个大型 `DataProto` 数据原型，如同 `collect_dp_compute_data_proto` 的处理方式。

最终，通过 `func_generator` 动态生成一个新方法并将其添加到 `WorkerGroup` 实例中：

```python
# bind a new method to the RayWorkerGroup
func = func_generator(
    self,
    method_name,
    dispatch_fn=dispatch_fn,
    collect_fn=collect_fn,
    execute_fn=execute_fn,
    blocking=blocking,
)

try:
    setattr(self, method_name, func)
    method_names.append(method_name)
except Exception as e:
    raise ValueError(f"Fail to set method_name {method_name}") from e
```

这使得该方法可通过 `WorkerGroup` 接口调用。

### 步骤 3：调用链

所有上述机制共同作用，确保了分布式调用与单进程调用体验的一致性。原本的单进程脚本如下所示：

```python
rollout = Rollout()
rollout.generate_sequences(batch)
```

而在使用`verl`之后，多进程程序变为：

```python
rollout = RayWorkerGroup(resource_pool=[4], RayClassWithArgs(Rollout))
rollout.generate_sequences(batch)
```

![call_chain_of_generate_sequences](https://github.com/eric-haibin-lin/verl-community/blob/main/docs/call_generate_sequences.png?raw=true)

在这个简单调用背后，

-  `dispatch_fn`负责将输入分发给各个工作节点，
- `execute_fn`执行实际的远程调用，
- `collect_fn`则负责收集结果.

这一切都被抽象封装起来，使得开发者只需对现有逻辑做最小改动即可编写高效的分布式代码。

## 泛化性

`verl.single_controller`模块的应用范围远不止于强化学习领域。它提供了批处理远程方法调用的清晰抽象层，并自动处理输入输出。通过缩小单进程与多进程脚本之间的差异，`verl.single_controller`为更广泛领域的分布式计算开辟了新的可能性。我们期待这个设计能激发社区贡献更多应用案例和扩展方案。
