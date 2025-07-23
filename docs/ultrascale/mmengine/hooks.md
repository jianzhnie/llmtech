# 钩子（HOOK）

## Hook设计描述

Hook，又称钩子，在C/C++中一般叫做回调函数。一个钩子方法由一个抽象类或具体类声明并实现，而其子类可能会加以扩展。通常在父类中给出的实现是一个空实现，并以该空实现作为方法的默认实现，当然钩子方法也可以提供一个非空的默认实现。 钩子是从功能角度描述这种编程模式，回调则是从函数调用角度描述的。

钩子编程是一种编程模式，通常理解的Hook是指在程序的一个或者多个位置设置位点（挂载点），当程序运行至该位点时，会自动调用运行时注册到位点的所有方法(使得程序在执行前或执行后做一些额外的处理)。钩子编程可以提高程序的灵活性和拓展性，用户将自定义的方法注册到位点便可被调用而无需修改程序中的代码。

例如我们熟知的windows系统消息响应事件，鼠标点击对程序产生的影响是由程序自己决定的，但是程序的执行是受制于框架（windows系统），框架提供了一些通用的流程执行，但是往往框架或流程在设计时无法完全预料到以后的使用会有什么新需求，或者有些行为只有在运行时才能确定的。这就产生了回调的需求，即用户提供需求，框架负责执行，流程先于具体需求，当触发或者满足某种条件时，执行Hook函数。Hook函数的数据也是由用户自己提供的，框架只负责流程执行，这样框架的通用性就能大大提高。

## Hook设计三要素

- Hook 函数或类：实现自定义操作或功能
- 注册：只有经过注册的Hook才能被系统或框架调用
- 挂载点：通常由系统或框架决定，用户无法修改

## 钩子（HOOK）示例

### 钩子示例一

Hook是一个编程机制，与语言无关。这里给出几个Python的简单例子，帮助大家理解。

下面是钩子的第一个简单示例。

```python
pre_hooks = [(print, 'hello')]
post_hooks = [(print, 'goodbye')]

def main():
    for func, arg in pre_hooks:
        func(arg)
    print('do something here')
    for func, arg in post_hooks:
        func(arg)

main()
```

下面是程序的输出：

```python
hello
do something here
goodbye
```

可以看到，`main` 函数在两个位置（pre_hooks 和 post_hooks）调用钩子中的函数而无需做任何改动。

### 钩子示例二

一个日常活动安排 Hook，如下描述：

我最近打算好好锻炼身体，健康生活，努力工作，打算让自己变得更加自律。我给自己定下了几个条例，每天吃早饭之前得**晨练30分钟，运动完之后才会感觉充满活力，吃完早饭工作3个小时，吃完午饭之后**午休60分钟**，午休完继续工作3个小时； 晚上下班前我如果没什么事再**锻炼60分钟， 晚饭之后工作继续60分钟，工作完就洗漱睡觉。秉承着这样的原则我给自己定义一个工作安排时间表来规范我的生活。定义了三个 HOOK： RunningHOOK，WorkingHOOK,  SleepingHOOK, 然后对这三个不同的 HOOK 进行合理安排，行成我每天的时间安排表。

- 定义我的HOOK

```python
import sys

class HOOK:

    def before_breakfirst(self, runner):
        pass

    def after_breakfirst(self, runner):
        pass

    def before_lunch(self, runner):
        pass

    def after_lunch(self, runner):
        pass

    def before_dinner(self, runner):
        pass

    def after_dinner(self, runner):
        pass

    def after_finish_work(self, runner, are_you_busy=False):
        pass


class RunningHOOK(HOOK):

    def before_breakfirst(self, runner):
        print(f'{sys._getframe().f_code.co_name}:吃早饭之前跑步30分钟')

    def before_dinner(self, runner):
        print(f'{sys._getframe().f_code.co_name}:吃晚饭之前跑步60分钟')


class WorkingHOOK(HOOK):

    def before_breakfirst(self, runner):
        print(f'{sys._getframe().f_code.co_name}:吃早饭之前看新闻30分钟')

    def after_breakfirst(self, runner):
        print(f'{sys._getframe().f_code.co_name}:吃早饭后工作3个小时')

    def after_lunch(self, runner):
        print(f'{sys._getframe().f_code.co_name}:吃完午饭工作3个小时')

    def after_dinner(self, runner):
        print(f'{sys._getframe().f_code.co_name}:晚饭之后工作60分钟')

class SleepingHOOK(HOOK):

    def after_lunch(self, runner):
        print(f'{sys._getframe().f_code.co_name}:吃完午饭午休60分钟')

    def after_finish_work(self, runner):
        print(f'{sys._getframe().f_code.co_name}: 洗漱上床睡觉！！！')
```

- 定义我的Runner

```python
class Runner:

    def __init__(self, ):
        self._hooks = []

    def register_hook(self, hook):
        # 这里不做优先级判断，直接在头部插入HOOK
        self._hooks.insert(0, hook)

    def call_hook(self, hook_name):
        for hook in self._hooks:
            getattr(hook, hook_name)(self)

    def run(self):
        print('开始启动我的一天')
        self.call_hook('before_breakfirst')
        self.call_hook('after_breakfirst')
        self.call_hook('before_lunch')
        self.call_hook('after_lunch')
        self.call_hook('before_dinner')
        self.call_hook('after_dinner')
        self.call_hook('after_finish_work')
        print("---睡觉---")
```

- 运行main函数，注册HOOK并且调用Runner.run()开启我的一天

```python
if __name__ == '__main__':
    runner = Runner()
    runinghook = RunningHOOK()
    workinghook = WorkingHOOK()
    sleephook = SleepingHOOK()
    runner.register_hook(runinghook)
    runner.register_hook(workinghook)
    runner.register_hook(sleephook)
    runner.run()
```

- 得到的输出结果如下:

```shell
开始启动我的一天
before_breakfirst:吃早饭之前看新闻30分钟
before_breakfirst:吃早饭之前跑步30分钟
after_breakfirst:吃早饭后工作3个小时
after_lunch:吃完午饭午休60分钟
after_lunch:吃完午饭工作3个小时
before_dinner:吃晚饭之前跑步60分钟
after_dinner:晚饭之后工作60分钟
after_finish_work: 去睡觉了！！！
---睡觉---
```

### Hooks in PyTorch

在 PyTorch 中，钩子的应用也随处可见，例如神经网络模块（nn.Module）中的钩子可以获得模块的前向输入输出以及反向的输入输出。以 [`register_forward_hook`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook) 方法为例，该方法往模块注册一个前向钩子，钩子可以获得模块的前向输入和输出。

下面是 `register_forward_hook` 用法的简单示例：

```python
import torch
import torch.nn as nn

def forward_hook_fn(
    module,  # 被注册钩子的对象
    input,  # module 前向计算的输入
    output,  # module 前向计算的输出
):
    print(f'"forward_hook_fn" is invoked by {module.name}')
    print('weight:', module.weight.data)
    print('bias:', module.bias.data)
    print('input:', input)
    print('output:', output)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 1)

    def forward(self, x):
        y = self.fc(x)
        return y

model = Model()
# 将 forward_hook_fn 注册到 model 每个子模块
for module in model.children():
    module.register_forward_hook(forward_hook_fn)

x = torch.Tensor([[0.0, 1.0, 2.0]])
y = model(x)
```

下面是程序的输出：

```shell
"forward_hook_fn" is invoked by Linear(in_features=3, out_features=1, bias=True)
weight: tensor([[-0.4077,  0.0119, -0.3606]])
bias: tensor([-0.2943])
input: (tensor([[0., 1., 2.]]),)
output: tensor([[-1.0036]], grad_fn=<AddmmBackward>)
```

可以看到注册到 Linear 模块的 `forward_hook_fn` 钩子被调用，在该钩子中打印了 Linear 模块的权重、偏置、模块的输入以及输出。更多关于 PyTorch 钩子的用法可以阅读 [nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)。

## MMEngine 中的钩子

### 钩子的设计

在介绍 MMEngine 中钩子的设计之前，先简单介绍使用 PyTorch 实现模型训练的基本步骤（示例代码来自 [PyTorch Tutorials](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py)）：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    pass

class Net(nn.Module):
    pass

def main():
    transform = transforms.ToTensor()
    train_dataset = CustomDataset(transform=transform, ...)
    val_dataset = CustomDataset(transform=transform, ...)
    test_dataset = CustomDataset(transform=transform, ...)
    train_dataloader = DataLoader(train_dataset, ...)
    val_dataloader = DataLoader(val_dataset, ...)
    test_dataloader = DataLoader(test_dataset, ...)

    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for i in range(max_epochs):
        for inputs, labels in train_dataloader:
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            for inputs, labels in val_dataloader:
                outputs = net(inputs)
                loss = criterion(outputs, labels)

    with torch.no_grad():
        for inputs, labels in test_dataloader:
            outputs = net(inputs)
            accuracy = ...
```

上面的伪代码是训练模型的基本步骤。如果要在上面的代码中加入定制化的逻辑，我们需要不断修改和拓展 `main` 函数。为了提高 `main` 函数的灵活性和拓展性，我们可以在 `main` 方法中插入位点，并在对应位点实现调用 hook 的抽象逻辑。此时只需在这些位点插入 hook 来实现定制化逻辑，即可添加定制化功能，例如加载模型权重、更新模型参数等。

```python
def main():
    ...
    call_hooks('before_run', hooks)  # 任务开始前执行的逻辑
    call_hooks('after_load_checkpoint', hooks)  # 加载权重后执行的逻辑
    call_hooks('before_train', hooks)  # 训练开始前执行的逻辑
    for i in range(max_epochs):
        call_hooks('before_train_epoch', hooks)  # 遍历训练数据集前执行的逻辑
        for inputs, labels in train_dataloader:
            call_hooks('before_train_iter', hooks)  # 模型前向计算前执行的逻辑
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            call_hooks('after_train_iter', hooks)  # 模型前向计算后执行的逻辑
            loss.backward()
            optimizer.step()
        call_hooks('after_train_epoch', hooks)  # 遍历完训练数据集后执行的逻辑

        call_hooks('before_val_epoch', hooks)  # 遍历验证数据集前执行的逻辑
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                call_hooks('before_val_iter', hooks)  # 模型前向计算前执行
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                call_hooks('after_val_iter', hooks)  # 模型前向计算后执行
        call_hooks('after_val_epoch', hooks)  # 遍历完验证数据集前执行

        call_hooks('before_save_checkpoint', hooks)  # 保存权重前执行的逻辑
    call_hooks('after_train', hooks)  # 训练结束后执行的逻辑

    call_hooks('before_test_epoch', hooks)  # 遍历测试数据集前执行的逻辑
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            call_hooks('before_test_iter', hooks)  # 模型前向计算后执行的逻辑
            outputs = net(inputs)
            accuracy = ...
            call_hooks('after_test_iter', hooks)  # 遍历完成测试数据集后执行的逻辑
    call_hooks('after_test_epoch', hooks)  # 遍历完测试数据集后执行

    call_hooks('after_run', hooks)  # 任务结束后执行的逻辑
```

可以看到，现在我们可以在每一步中按照我们想要的方式修改训练循环，只需要在各处添加钩子函数。

这些钩子函数的作用非常不言自明：

- `call_hooks.on_epoch_begin()`  意味着 “嘿，钩子函数，我正在开始一个新的 Epoch，这里有人想做点什么吗？”
- `call_hooks.on_step_end()` 意思是 “ 嘿，钩子函数，我刚刚完成了一个Step, 采取了优化器步骤，即将将梯度归零。现在有什么事情要做吗？”

在 MMEngine 中，我们将训练过程抽象成执行器（Runner），执行器除了完成环境的初始化，另一个功能是在特定的位点调用钩子完成定制化逻辑。更多关于执行器的介绍请阅读[执行器文档](https://mmengine.readthedocs.io/zh-cn/latest/tutorials/runner.html)。

### Hook基类

为了方便管理，MMEngine 将位点定义为方法并集成到[钩子基类（Hook）](https://mmengine.readthedocs.io/zh-cn/latest/api/generated/mmengine.hooks.Hook.html#mmengine.hooks.Hook)中，我们只需继承钩子基类并根据需求在特定位点实现定制化逻辑，再将钩子注册到执行器中，便可自动调用钩子中相应位点的方法。

 首先看一下HOOK的基类定义。

```python
class Hook:
    """Base hook class.

    All hooks should inherit from this class.
    """
    def before_run(self, runner):
        pass

    def after_run(self, runner):
        pass

    def before_train(self, runner) -> None:
    		pass

    def after_train(self, runner) -> None:
    		pass

    def before_val(self, runner) -> None:
    		pass

    def after_val(self, runner) -> None:
    		pass

    def before_test(self, runner) -> None:
    		pass

    def after_test(self, runner) -> None:
    		pass

     def before_save_checkpoint(self, runner, checkpoint: dict) -> None:
     		pass

    def after_load_checkpoint(self, runner, checkpoint: dict) -> None:
    		pass

    def before_epoch(self, runner):
        pass

    def after_epoch(self, runner):
        pass

    def before_iter(self, runner):
        pass

    def after_iter(self, runner):
        pass

    def before_train_epoch(self, runner):
        self.before_epoch(runner, mode='train')

    def before_val_epoch(self, runner):
        self.before_epoch(runner, mode='val')

    def before_test_epoch(self, runner):
        self.before_epoch(runner, mode='test')

    def after_train_epoch(self, runner):
        self._after_epoch(runner, mode='train')

    def after_val_epoch(self, runner):
        self._after_epoch(runner, mode='val')

    def after_test_epoch(self, runner):
        self._after_epoch(runner, mode='test')

    def before_train_iter(self, runner):
        self.before_iter(runner, mode='train')

    def before_val_iter(self, runner):
        self.before_iter(runner, mode='val')

    def before_test_iter(self, runner):
        self.before_iter(runner, mode='test')

    def after_train_iter(self, runner):
        self.after_iter(runner, mode='train')

    def after_val_iter(self, runner):
        self.after_iter(runner, mode='val')

    def after_test_iter(self, runner):
        self.after_iter(runner, mode='test')

    def every_n_epochs(self, runner, n):
        return (runner.epoch + 1) % n == 0 if n > 0 else False

    def every_n_inner_iters(self, runner, n):
        return (runner.inner_iter + 1) % n == 0 if n > 0 else False

    def every_n_train_iters(self, runner, n: int, start: int = 0) -> bool:
        dividend = runner.iter + 1 - start
        return dividend % n == 0 if dividend >= 0 and n > 0 else False

    def every_n_iters(self, runner, n):
        return (runner.iter + 1) % n == 0 if n > 0 else False

    def end_of_epoch(self, runner):
        return runner.inner_iter + 1 == len(runner.data_loader)

    def is_last_train_epoch(self, runner) -> bool:
        return runner.epoch + 1 == runner.max_epochs

    def is_last_train_iter(self, runner) -> bool:
        return runner.iter + 1 == runner.max_iters
```

观察 Hook 类中的方法不难发现，Hook 类将训练过程中我们可能采取额外操作（如调整学习率，存储模型和日志文件，打印训练信息等）的时间点分为开始训练前、一个 iteration 前、一个 iteration 后、 一个 epoch 前、一个 epoch 后、每 n 个 iteration 后、每 n 个 epoch 后，这些时间点又分为 train、 validate 过程和 test 过程（在基类中默认两个过程采取相同的操作）。

Hook 类是所有hook类的父类, 是一个抽象类，规定了具体的调用名称和挂载点，如before_run、before_epoch、after_epoch、after_run等，但是仅仅定义了一组接口而没有具体实现，这意味着我们必须通过继承的方式来使用。如果希望在某几个时间点采取一些特定的操作，需要定义一个新的类并继承 Hook 类， 然后然后重写各个时间点对应的方法，实现一些定制化的东西，比如：

- `CheckPointHook`  钩子类中` after_epoch` 函数实现：训练一个epoch后我们要保存下训练的模型；
- `EvalHook`钩子类中 `after_train` 函数实现：在`结束训练`时用最好的模型执行下测试集的效果等等。

可以看到HOOK中每一个参数都是有runner作为参数传入的。Runner是一个模型训练的总控制器，在其中我们可以加载数据、训练、验证以及梯度backward 等等全套流程，在每一个hook函数中，都可以对runner进行你想要的操作。

定义完钩子函数之后，最后调用 Runner 对象的 register_hook 方法在对象中注册这个 hook。 用户的 Hooks 是放在一个有序列表中，按优先级排列，优先级高的在前，先得到执行，优先级也是由用户确定的，这是用户仅有的权力。


### Hook注册

而HOOK是怎么嵌套进runner中的呢？ Hook 的注册过程比较简单，因为触发是按框架定义的流程顺序主动调用的，因此仅需要按优先级插入到有序列表中即可。

在Runner中定义了一个hook的list，list中的每一个元素就是一个实例化的HOOK对象。其中提供了两种注册hook的方法:

- `register_default_hooks` 是将默认的HOOKs 注册到列表中
- `register_custom_hooks` 是将自定义的 HOOKs 注册到列表中

```python
def register_hooks(
  self,
  default_hooks: Optional[Dict[str, Union[Hook, Dict]]] = None,
  custom_hooks: Optional[List[Union[Hook, Dict]]] = None) -> None:
  """Register default hooks and custom hooks into hook list.

        Args:
            default_hooks (dict[str, dict] or dict[str, Hook], optional): Hooks
                to execute default actions like updating model parameters and
                saving checkpoints.  Defaults to None.
            custom_hooks (list[dict] or list[Hook], optional): Hooks to execute
                custom actions like visualizing images processed by pipeline.
                Defaults to None.
        """
  self.register_default_hooks(default_hooks)

  if custom_hooks is not None:
    self.register_custom_hooks(custom_hooks
```

实例化runner对象后，会去注册runner中用到的hooks.

```python
# register hooks to `self._hooks`
self.register_hooks(default_hooks, custom_hooks)
# log hooks information
self.logger.info(f'Hooks will be executed in the following '
f'order:\n{self.get_hooks_info()}')
```

来看看 `register_hook` 函数的实现方式。


````python
def register_hook(
        self,
        hook: Union[Hook, Dict],
        priority: Optional[Union[str, int, Priority]] = None) -> None:
    """Register a hook into the hook list.

    The hook will be inserted into a priority queue, with the specified
    priority (See :class:`Priority` for details of priorities).
    For hooks with the same priority, they will be triggered in the same
    order as they are registered.

    Priority of hook will be decided with the following priority:

    - ``priority`` argument. If ``priority`` is given, it will be priority
        of hook.
    - If ``hook`` argument is a dict and ``priority`` in it, the priority
        will be the value of ``hook['priority']``.
    - If ``hook`` argument is a dict but ``priority`` not in it or ``hook``
        is an instance of ``hook``, the priority will be ``hook.priority``.

    Args:
        hook (:obj:`Hook` or dict): The hook to be registered.
        priority (int or str or :obj:`Priority`, optional): Hook priority.
            Lower value means higher priority.
    """
    if not isinstance(hook, (Hook, dict)):
        raise TypeError(
            f'hook should be an instance of Hook or dict, but got {hook}')

    _priority = None
    if isinstance(hook, dict):
        if 'priority' in hook:
            _priority = hook.pop('priority')

        hook_obj = HOOKS.build(hook)
    else:
        hook_obj = hook

    if priority is not None:
        hook_obj.priority = priority
    elif _priority is not None:
        hook_obj.priority = _priority

    inserted = False
    for i in range(len(self._hooks) - 1, -1, -1):
        if get_priority(hook_obj.priority) >= get_priority(
                self._hooks[i].priority):
            self._hooks.insert(i + 1, hook_obj)
            inserted = True
            break
    if not inserted:
        self._hooks.insert(0, hook_obj)
````

可以看到在register_hook中（核心代码45~52行），倒序遍历队列，当找到一个比当前hook优先级高的hook时，就把当前的hook插入到这个hook的后面，如果找不到比它优先级高的就直接放在第一位。

### Hook实现

为了便于理解这个过程，我们以 mmcv 中的 LrUpdaterHook 类为例简要分析一下 hook 对象的行为。LrUpdaterHook 类主要封装了一些对学习率的修改操作，看下面的代码：

```python
class LrUpdaterHook(Hook):
    """LR Scheduler in MMCV.

    Args:
        by_epoch (bool): LR changes epoch by epoch
        warmup (string): Type of warmup used. It can be None(use no warmup),
            'constant', 'linear' or 'exp'
        warmup_iters (int): The number of iterations or epochs that warmup
            lasts
        warmup_ratio (float): LR used at the beginning of warmup equals to
            warmup_ratio * initial_lr
        warmup_by_epoch (bool): When warmup_by_epoch == True, warmup_iters
            means the number of epochs that warmup lasts, otherwise means the
            number of iteration that warmup lasts
    """

    def __init__(self,
                 by_epoch=True,
                 warmup=None,
                 warmup_iters=0,
                 warmup_ratio=0.1,
                 warmup_by_epoch=False):
        # validate the "warmup" argument
        if warmup is not None:
            if warmup not in ['constant', 'linear', 'exp']:
                raise ValueError(
                    f'"{warmup}" is not a supported type for warming up, valid'
                    ' types are "constant" and "linear"')
        if warmup is not None:
            assert warmup_iters > 0, \
                '"warmup_iters" must be a positive integer'
            assert 0 < warmup_ratio <= 1.0, \
                '"warmup_ratio" must be in range (0,1]'

        self.by_epoch = by_epoch
        self.warmup = warmup
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.warmup_by_epoch = warmup_by_epoch

        if self.warmup_by_epoch:
            self.warmup_epochs = self.warmup_iters
            self.warmup_iters = None
        else:
            self.warmup_epochs = None

        self.base_lr = []  # initial lr for all param groups
        self.regular_lr = []  # expected lr if no warming up is performed

    def _set_lr(self, runner, lr_groups):
        if isinstance(runner.optimizer, dict):
            for k, optim in runner.optimizer.items():
                for param_group, lr in zip(optim.param_groups, lr_groups[k]):
                    param_group['lr'] = lr
        else:
            for param_group, lr in zip(runner.optimizer.param_groups,
                                       lr_groups):
                param_group['lr'] = lr

    def get_lr(self, runner, base_lr):
        raise NotImplementedError

    def get_regular_lr(self, runner):
        if isinstance(runner.optimizer, dict):
            lr_groups = {}
            for k in runner.optimizer.keys():
                _lr_group = [
                    self.get_lr(runner, _base_lr)
                    for _base_lr in self.base_lr[k]
                ]
                lr_groups.update({k: _lr_group})

            return lr_groups
        else:
            return [self.get_lr(runner, _base_lr) for _base_lr in self.base_lr]

    def get_warmup_lr(self, cur_iters):
        if self.warmup == 'constant':
            warmup_lr = [_lr * self.warmup_ratio for _lr in self.regular_lr]
        elif self.warmup == 'linear':
            k = (1 - cur_iters / self.warmup_iters) * (1 - self.warmup_ratio)
            warmup_lr = [_lr * (1 - k) for _lr in self.regular_lr]
        elif self.warmup == 'exp':
            k = self.warmup_ratio**(1 - cur_iters / self.warmup_iters)
            warmup_lr = [_lr * k for _lr in self.regular_lr]
        return warmup_lr

    def before_run(self, runner):
        # NOTE: when resuming from a checkpoint, if 'initial_lr' is not saved,
        # it will be set according to the optimizer params
        if isinstance(runner.optimizer, dict):
            self.base_lr = {}
            for k, optim in runner.optimizer.items():
                for group in optim.param_groups:
                    group.setdefault('initial_lr', group['lr'])
                _base_lr = [
                    group['initial_lr'] for group in optim.param_groups
                ]
                self.base_lr.update({k: _base_lr})
        else:
            for group in runner.optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
            self.base_lr = [
                group['initial_lr'] for group in runner.optimizer.param_groups
            ]

    def before_train_epoch(self, runner):
        if self.warmup_iters is None:
            epoch_len = len(runner.data_loader)
            self.warmup_iters = self.warmup_epochs * epoch_len

        if not self.by_epoch:
            return

        self.regular_lr = self.get_regular_lr(runner)
        self._set_lr(runner, self.regular_lr)

    def before_train_iter(self, runner):
        cur_iter = runner.iter
        if not self.by_epoch:
            self.regular_lr = self.get_regular_lr(runner)
            if self.warmup is None or cur_iter >= self.warmup_iters:
                self._set_lr(runner, self.regular_lr)
            else:
                warmup_lr = self.get_warmup_lr(cur_iter)
                self._set_lr(runner, warmup_lr)
        elif self.by_epoch:
            if self.warmup is None or cur_iter > self.warmup_iters:
                return
            elif cur_iter == self.warmup_iters:
                self._set_lr(runner, self.regular_lr)
            else:
                warmup_lr = self.get_warmup_lr(cur_iter)
                self._set_lr(runner, warmup_lr)
```

这个类重写了 `before_run`、`before_train_epoch`、`before_train_iter `方法，其构造函数的参数 by_epoch 如果为 True 则表明我们以 epoch 为单位计量训练进程，否则以 iteration 为单位。warmup 参数为字符串，指定了 warmup 算法中学习率的变化方式，warmup_iters 和 warmup_ratio 分别指定了 warmup 的 iteration 数和增长比例。

从代码中可以看出，在训练开始前，LrUpdaterHook 对象首先会设置 Runner 对象中所维护的优化器的各项参数，然后在每个 iteration 和 epoch 开始前检查学习率和 iteration（epoch）的值，然后计算下一次迭代过程的学习率的值并修改 Runner 中的学习率。

换句话说，LrUpdaterHook 类仅提供了在相应时间修改学习率的代码，至于学习率的衰减方式则应该根据需要自行设置。Hook 机制的好处在于，当我们需要在某些时间点添加一组特定的操作时，只需要编写相应的 hook 类将操作封装并调用 Runner 对象的 register_hook 方法注册即可，这使得整个训练的过程变得更容易定制。

其实实现hook时，用户的疑问往往是自定义hook需要使用的数据从哪里来？显然用户不知道Run类中有哪些数据。用户其实是知道的，因为Run中原本是没有数据的，它仅是一个流程执行类，其中的数据均来自与用户创建run时传入的，如runner.LrUpdaterHook。所以可以看到，一个hook仅仅需要两个元素，一个是执行者，这里是runner，另外一个是执行时间（触发条件，挂载点）。

### Hook函数调用

Runner 类中维护了一个存放 hook 对象的列表 self._hooks，在每个位点会通过 call_hook 方法依次调用列表中所有 hook 对象对应的接口以执行相关操作，call_hook 方法定义为：

```python
def call_hook(self, fn_name: str, **kwargs) -> None:
  """Call all hooks.

        Args:
            fn_name (str): The function name in each hook to be called, such as
                "before_train_epoch".
            **kwargs: Keyword arguments passed to hook.
        """
  for hook in self._hooks:
    # support adding additional custom hook methods
    if hasattr(hook, fn_name):
      try:
        getattr(hook, fn_name)(self, **kwargs)
      except TypeError as e:
        raise TypeError(f'{e} in {hook}') from None
```

其中 fn_name 是一个字符串对象，表示希望执行的方法名，这里利用了 Python 的内建函数 getattr 来获得 hook 对象中同名方法的引用。用户仅仅需要实现自己所需要的hook，如果没有自定义的hook，框架会调用父类Hook中相应的方法。父类Hook可能提供了一些默认行为，也可能什么都没做。


根据实现可以看出:

- 每次调用`call_hook`的时候整个队列中的所有hook都会被调用到，并且执行自己实现的fn_name函数。
- 可以看到call_hook运行的时候是遍历整个 `hook list`，然后根据 fn_name 的名字来调用。这也是为什么要区分优先级的原因，优先级越高的 HOOK放在List的前面，这样就能更快地被调用。当你想用*before_run_epoch*来做A和B两件事情的时候，在runner里面就是调用一次`self.before_run_epoch`，但是先做A还是先做B，就是通过不同的HOOK的优先级来决定了。

下面通过一个例子来说明， 例如 对于`after_train_epoch` 函数， 在两个不同的HOOK 中都有定义。

`ParamSchedulerHook`中的 `after_train_epoch` 定义。

```python
def after_train_epoch(self, runner) -> None:
  """Call step function for each scheduler after each training epoch.

        Args:
            runner (Runner): The runner of the training process.
        """

  if runner.param_schedulers is None:
    return

  def step(param_schedulers):
    assert isinstance(param_schedulers, list)
    for scheduler in param_schedulers:
      if scheduler.by_epoch:
        scheduler.step()

        if isinstance(runner.param_schedulers, list):
          step(runner.param_schedulers)
        elif isinstance(runner.param_schedulers, dict):
          for param_schedulers in runner.param_schedulers.values():
            step(param_schedulers)
          else:
            raise TypeError(
              'runner.param_schedulers should be list of ParamScheduler or '
              'a dict containing list of ParamScheduler, '
              f'but got {runner.param_schedulers}')
```

CheckpointHook`的HOOK中，同样也定义了`after_train_epoch`函数如下：

```python
def after_train_epoch(self, runner) -> None:
  """Save the checkpoint and synchronize buffers after each epoch.

        Args:
            runner (Runner): The runner of the training process.
        """
  if not self.by_epoch:
    return

  # save checkpoint for following cases:
  # 1. every ``self.interval`` epochs which start at ``self.save_begin``
  # 2. reach the last epoch of training
  if self.every_n_epochs(runner, self.interval, self.save_begin) or (
    self.save_last and self.is_last_train_epoch(runner)):
    runner.logger.info(
      f'Saving checkpoint at {runner.epoch + 1} epochs')
    self._save_checkpoint(runner
```

可以看到不同的HOOK虽然都是重写了`after_train_epoch`函数，但是调用时还是先调用`param_scheduler_hook.py`中after_train_epoch，然后再调用`checkpoint_hook.py`中的`after_train_epoch`。

用一个Priority queue 存储实例化的hook对象，用来保证hook调用的优先级。具体的优先级定义有以下7种，作为HOOK的类成员属性。

```
        Default hooks and their priorities:

        +----------------------+-------------------------+
        | Hooks                | Priority                |
        +======================+=========================+
        | RuntimeInfoHook      | VERY_HIGH (10)          |
        +----------------------+-------------------------+
        | IterTimerHook        | NORMAL (50)             |
        +----------------------+-------------------------+
        | DistSamplerSeedHook  | NORMAL (50)             |
        +----------------------+-------------------------+
        | LoggerHook           | BELOW_NORMAL (60)       |
        +----------------------+-------------------------+
        | ParamSchedulerHook   | LOW (70)                |
        +----------------------+-------------------------+
        | CheckpointHook       | VERY_LOW (90)           |
        +----------------------+-------------------------+
```

### Runner中的 Hooks 调用

现在我们回过头来看 Runner 类的 train 方法，看下面的代码

```python
def train(self) -> nn.Module:
    """Launch training.

    Returns:
        nn.Module: The model after training.
    """
    if is_model_wrapper(self.model):
        ori_model = self.model.module
    else:
        ori_model = self.model
    assert hasattr(ori_model, 'train_step'), (
        'If you want to train your model, please make sure your model '
        'has implemented `train_step`.')

    if self._val_loop is not None:
        assert hasattr(ori_model, 'val_step'), (
            'If you want to validate your model, please make sure your '
            'model has implemented `val_step`.')

    if self._train_loop is None:
        raise RuntimeError(
            '`self._train_loop` should not be None when calling train '
            'method. Please provide `train_dataloader`, `train_cfg`, '
            '`optimizer` and `param_scheduler` arguments when '
            'initializing runner.')

    self._train_loop = self.build_train_loop(
        self._train_loop)  # type: ignore

    # `build_optimizer` should be called before `build_param_scheduler`
    #  because the latter depends on the former
    self.optim_wrapper = self.build_optim_wrapper(self.optim_wrapper)
    # Automatically scaling lr by linear scaling rule
    self.scale_lr(self.optim_wrapper, self.auto_scale_lr)

    if self.param_schedulers is not None:
        self.param_schedulers = self.build_param_scheduler(  # type: ignore
            self.param_schedulers)  # type: ignore

    if self._val_loop is not None:
        self._val_loop = self.build_val_loop(
            self._val_loop)  # type: ignore
    # TODO: add a contextmanager to avoid calling `before_run` many times
    self.call_hook('before_run')

    # initialize the model weights
    self._init_model_weights()

    # try to enable activation_checkpointing feature
    modules = self.cfg.get('activation_checkpointing', None)
    if modules is not None:
        self.logger.info(f'Enabling the "activation_checkpointing" feature'
                            f' for sub-modules: {modules}')
        turn_on_activation_checkpointing(ori_model, modules)

    # try to enable efficient_conv_bn_eval feature
    modules = self.cfg.get('efficient_conv_bn_eval', None)
    if modules is not None:
        self.logger.info(f'Enabling the "efficient_conv_bn_eval" feature'
                            f' for sub-modules: {modules}')
        turn_on_efficient_conv_bn_eval(ori_model, modules)

    # make sure checkpoint-related hooks are triggered after `before_run`
    self.load_or_resume()

    # Initiate inner count of `optim_wrapper`.
    self.optim_wrapper.initialize_count_status(
        self.model,
        self._train_loop.iter,  # type: ignore
        self._train_loop.max_iters)  # type: ignore

    # Maybe compile the model according to options in self.cfg.compile
    # This must be called **AFTER** model has been wrapped.
    self._maybe_compile('train_step')

    model = self.train_loop.run()  # type: ignore
    self.call_hook('after_run')
    return model
```
Runner.train() 方法中

- 第一步：创建 `model`, `optim_wrapper`, `param_schedulers`, `train_loop`。
- 第二步：`self.call_hook('before_run')`, `model = self.train_loop.run()` ,`self.call_hook('before_run')`

EpochBasedTrainLoop 类的定义如下:

```python
class EpochBasedTrainLoop(BaseLoop):
    """Loop for epoch-based training.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        max_epochs (int): Total training epochs.
        val_begin (int): The epoch that begins validating.
            Defaults to 1.
        val_interval (int): Validation interval. Defaults to 1.
        dynamic_intervals (List[Tuple[int, int]], optional): The
            first element in the tuple is a milestone and the second
            element is a interval. The interval is used after the
            corresponding milestone. Defaults to None.
    """

    def __init__(
            self,
            runner,
            dataloader: Union[DataLoader, Dict],
            max_epochs: int,
            val_begin: int = 1,
            val_interval: int = 1,
            dynamic_intervals: Optional[List[Tuple[int, int]]] = None) -> None:
        super().__init__(runner, dataloader)
        self._max_epochs = int(max_epochs)
        assert self._max_epochs == max_epochs, \
            f'`max_epochs` should be a integer number, but get {max_epochs}.'
        self._max_iters = self._max_epochs * len(self.dataloader)
        self._epoch = 0
        self._iter = 0
        self.val_begin = val_begin
        self.val_interval = val_interval
        # This attribute will be updated by `EarlyStoppingHook`
        # when it is enabled.
        self.stop_training = False
        if hasattr(self.dataloader.dataset, 'metainfo'):
            self.runner.visualizer.dataset_meta = \
                self.dataloader.dataset.metainfo
        else:
            print_log(
                f'Dataset {self.dataloader.dataset.__class__.__name__} has no '
                'metainfo. ``dataset_meta`` in visualizer will be '
                'None.',
                logger='current',
                level=logging.WARNING)

        self.dynamic_milestones, self.dynamic_intervals = \
            calc_dynamic_intervals(
                self.val_interval, dynamic_intervals)

    @property
    def max_epochs(self):
        """int: Total epochs to train model."""
        return self._max_epochs

    @property
    def max_iters(self):
        """int: Total iterations to train model."""
        return self._max_iters

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    @property
    def iter(self):
        """int: Current iteration."""
        return self._iter

    def run(self) -> torch.nn.Module:
        """Launch training."""
        self.runner.call_hook('before_train')

        while self._epoch < self._max_epochs and not self.stop_training:
            self.run_epoch()

            self._decide_current_val_interval()
            if (self.runner.val_loop is not None
                    and self._epoch >= self.val_begin
                    and self._epoch % self.val_interval == 0):
                self.runner.val_loop.run()

        self.runner.call_hook('after_train')
        return self.runner.model

    def run_epoch(self) -> None:
        """Iterate one epoch."""
        self.runner.call_hook('before_train_epoch')
        self.runner.model.train()
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)

        self.runner.call_hook('after_train_epoch')
        self._epoch += 1

    def run_iter(self, idx, data_batch: Sequence[dict]) -> None:
        """Iterate one min-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook(
            'before_train_iter', batch_idx=idx, data_batch=data_batch)
        # Enable gradient accumulation mode and avoid unnecessary gradient
        # synchronization during gradient accumulation process.
        # outputs should be a dict of loss.
        outputs = self.runner.model.train_step(
            data_batch, optim_wrapper=self.runner.optim_wrapper)

        self.runner.call_hook(
            'after_train_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)
        self._iter += 1

    def _decide_current_val_interval(self) -> None:
        """Dynamically modify the ``val_interval``."""
        step = bisect.bisect(self.dynamic_milestones, (self.epoch + 1))
        self.val_interval = self.dynamic_intervals[step - 1]
```

`EpochBasedTrainLoop  `类定义了训练的过程。run 方法中的 while 循环表示的就是一个完整的训练过程，故而在这个循环的前后分别执行了 `self.call_hook('before_train')` 和 `self.call_hook('after_train')`，而 `run_epoch` 方法中的 for 循环定义了一个 `epoch ` 训练的过程，并且每次循环都表示一次 `iteration`，因此在整个循环前后分别执行了 `self.call_hook('before_train_epoch')`和 `self.call_hook('after_train_epoch')`，在每次迭代前后又分别执行 `self.call_hook('before_train_iter')`和 `self.call_hook('after_train_iter')`。

需要注意的地方如果有一个 hook 需要在两个不同时机执行两个需求，如在 `before_train_epoch`和`after_train_epoch`，但是恰巧这两个需求的优先级不同，这个时候建议写成两个hook，每个hook只负责做一件事，这也是编程中一般原则。

## MMEngine 中的内置 Hook

MMEngine 提供了很多内置的钩子，将钩子分为两类，分别是默认钩子以及自定义钩子，前者表示会默认往执行器注册，后者表示需要用户自己注册。

每个钩子都有对应的优先级，在同一位点，钩子的优先级越高，越早被执行器调用，如果优先级一样，被调用的顺序和钩子注册的顺序一致。优先级列表如下：

- HIGHEST (0)
- VERY_HIGH (10)
- HIGH (30)
- ABOVE_NORMAL (40)
- NORMAL (50)
- BELOW_NORMAL (60)
- LOW (70)
- VERY_LOW (90)
- LOWEST (100)

**默认钩子**

| 名称                                                         | 用途                               | 优先级            |
| ------------------------------------------------------------ | ---------------------------------- | ----------------- |
| [RuntimeInfoHook](https://mmengine.readthedocs.io/zh-cn/latest/tutorials/hook.html#runtimeinfohook) | 往 message hub 更新运行时信息      | VERY_HIGH (10)    |
| [IterTimerHook](https://mmengine.readthedocs.io/zh-cn/latest/tutorials/hook.html#itertimerhook) | 统计迭代耗时                       | NORMAL (50)       |
| [DistSamplerSeedHook](https://mmengine.readthedocs.io/zh-cn/latest/tutorials/hook.html#distsamplerseedhook) | 确保分布式 Sampler 的 shuffle 生效 | NORMAL (50)       |
| [LoggerHook](https://mmengine.readthedocs.io/zh-cn/latest/tutorials/hook.html#loggerhook) | 打印日志                           | BELOW_NORMAL (60) |
| [ParamSchedulerHook](https://mmengine.readthedocs.io/zh-cn/latest/tutorials/hook.html#paramschedulerhook) | 调用 ParamScheduler 的 step 方法   | LOW (70)          |
| [CheckpointHook](https://mmengine.readthedocs.io/zh-cn/latest/tutorials/hook.html#checkpointhook) | 按指定间隔保存权重                 | VERY_LOW (90)     |

**自定义钩子**

| 名称                                                         | 用途                  | 优先级      |
| ------------------------------------------------------------ | --------------------- | ----------- |
| [EMAHook](https://mmengine.readthedocs.io/zh-cn/latest/tutorials/hook.html#emahook) | 模型参数指数滑动平均  | NORMAL (50) |
| [EmptyCacheHook](https://mmengine.readthedocs.io/zh-cn/latest/tutorials/hook.html#emptycachehook) | PyTorch CUDA 缓存清理 | NORMAL (50) |
| [SyncBuffersHook](https://mmengine.readthedocs.io/zh-cn/latest/tutorials/hook.html#syncbuffershook) | 同步模型的 buffer     | NORMAL (50) |

备注

不建议修改默认钩子的优先级，因为优先级低的钩子可能会依赖优先级高的钩子。例如 CheckpointHook 的优先级需要比 ParamSchedulerHook 低，这样保存的优化器状态才是正确的状态。另外，自定义钩子的优先级默认为 `NORMAL (50)`。

两种钩子在执行器中的设置不同，默认钩子的配置传给执行器的 `default_hooks` 参数，自定义钩子的配置传给 `custom_hooks` 参数，如下所示：

```python
from mmengine.runner import Runner

default_hooks = dict(
    runtime_info=dict(type='RuntimeInfoHook'),
    timer=dict(type='IterTimerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    logger=dict(type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
)

custom_hooks = [dict(type='EmptyCacheHook')]

runner = Runner(default_hooks=default_hooks, custom_hooks=custom_hooks, ...)
runner.train()
```



下面逐一介绍 MMEngine 中内置钩子的用法。

### CheckpointHook

[CheckpointHook](https://mmengine.readthedocs.io/zh-cn/latest/api/generated/mmengine.hooks.CheckpointHook.html#mmengine.hooks.CheckpointHook) 按照给定间隔保存模型的权重，如果是分布式多卡训练，则只有主（master）进程会保存权重。`CheckpointHook` 的主要功能如下：

- 按照间隔保存权重，支持按 epoch 数或者 iteration 数保存权重
- 保存最新的多个权重
- 保存最优权重
- 指定保存权重的路径
- 制作发布用的权重
- 设置开始保存权重的 epoch 数或者 iteration 数

如需了解其他功能，请阅读 [CheckpointHook API 文档](https://mmengine.readthedocs.io/zh-cn/latest/api/generated/mmengine.hooks.CheckpointHook.html#mmengine.hooks.CheckpointHook)。

下面介绍上面提到的 6 个功能。

- 按照间隔保存权重，支持按 epoch 数或者 iteration 数保存权重

  假设我们一共训练 20 个 epoch 并希望每隔 5 个 epoch 保存一次权重，下面的配置即可帮我们实现该需求。

  ```python
  # by_epoch 的默认值为 True
  default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=5, by_epoch=True))
  ```

  如果想以迭代次数作为保存间隔，则可以将 `by_epoch` 设为 False，`interval=5` 则表示每迭代 5 次保存一次权重。

  ```python
  default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=5, by_epoch=False))
  ```

- 保存最新的多个权重

  如果只想保存一定数量的权重，可以通过设置 `max_keep_ckpts` 参数实现最多保存 `max_keep_ckpts` 个权重，当保存的权重数超过 `max_keep_ckpts` 时，前面的权重会被删除。

  ```python
  default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=5, max_keep_ckpts=2))
  ```

  上述例子表示，假如一共训练 20 个 epoch，那么会在第 5, 10, 15, 20 个 epoch 保存模型，但是在第 15 个 epoch 的时候会删除第 5 个 epoch 保存的权重，在第 20 个 epoch 的时候会删除第 10 个 epoch 的权重，最终只有第 15 和第 20 个 epoch 的权重才会被保存。

- 保存最优权重

  如果想要保存训练过程验证集的最优权重，可以设置 `save_best` 参数，如果设置为 `'auto'`，则会根据验证集的第一个评价指标（验证集返回的评价指标是一个有序字典）判断当前权重是否最优。

  ```python
  default_hooks = dict(checkpoint=dict(type='CheckpointHook', save_best='auto'))
  ```



  也可以直接指定 `save_best` 的值为评价指标，例如在分类任务中，可以指定为 `save_best='top-1'`，则会根据 `'top-1'` 的值判断当前权重是否最优。

  除了 `save_best` 参数，和保存最优权重相关的参数还有 `rule`，`greater_keys` 和 `less_keys`，这三者用来判断 `save_best` 的值是越大越好还是越小越好。例如指定了 `save_best='top-1'`，可以指定 `rule='greater'`，则表示该值越大表示权重越好。



- 指定保存权重的路径

  权重默认保存在工作目录（work_dir），但可以通过设置 `out_dir` 改变保存路径。

  ```python
  default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=5, out_dir='/path/of/directory'))
  ```



- 制作发布用的权重

  如果你想在训练结束后自动生成可发布的权重（删除不需要的权重，例如优化器状态），你可以设置 `published_keys` 参数，选择需要保留的信息。注意：需要相应设置 `save_best` 或者 `save_last` 参数，这样才会生成可发布的权重，其中设置 `save_best` 会生成最优权重的可发布权重，设置 `save_last` 会生成最后一个权重的可发布权重，这两个参数也可同时设置。

  ```python
  default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1, save_best='accuracy', rule='less', published_keys=['meta', 'state_dict']))
  ```



- 设置开始保存权重的 epoch 数或者 iteration 数

  如果想要设置控制开始保存权重的 epoch 数或者 iteration 数，可以设置 `save_begin` 参数，默认为 0，表示从训练开始就保存权重。例如，如果总共训练 10 个 epoch，并且 `save_begin` 设置为 5，则将保存第 5、6、7、8、9 和 10 个 epoch 的权重。如果 `interval=2`，则仅保存第 5、7 和 9 个 epoch 的权重。

  ```python
  default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=2, save_begin=5))
  ```

### LoggerHook

[LoggerHook](https://mmengine.readthedocs.io/zh-cn/latest/api/generated/mmengine.hooks.LoggerHook.html#mmengine.hooks.LoggerHook) 负责收集日志并把日志输出到终端或者输出到文件、TensorBoard 等后端。

如果我们希望每迭代 20 次就输出（或保存）一次日志，我们可以设置 `interval` 参数，配置如下：

```python
default_hooks = dict(logger=dict(type='LoggerHook', interval=20))
```

如果你对日志的管理感兴趣，可以阅读[记录日志（logging）](https://mmengine.readthedocs.io/zh-cn/latest/advanced_tutorials/logging.html)。

### ParamSchedulerHook

[ParamSchedulerHook](https://mmengine.readthedocs.io/zh-cn/latest/api/generated/mmengine.hooks.ParamSchedulerHook.html#mmengine.hooks.ParamSchedulerHook) 遍历执行器的所有优化器参数调整策略（Parameter Scheduler）并逐个调用 step 方法更新优化器的参数。如需了解优化器参数调整策略的用法请阅读[文档](https://mmengine.readthedocs.io/zh-cn/latest/tutorials/param_scheduler.html)。`ParamSchedulerHook` 默认注册到执行器并且没有可配置的参数，所以无需对其做任何配置。

### IterTimerHook

[IterTimerHook](https://mmengine.readthedocs.io/zh-cn/latest/api/generated/mmengine.hooks.IterTimerHook.html#mmengine.hooks.IterTimerHook) 用于记录加载数据的时间以及迭代一次耗费的时间。`IterTimerHook` 默认注册到执行器并且没有可配置的参数，所以无需对其做任何配置。

### DistSamplerSeedHook

[DistSamplerSeedHook](https://mmengine.readthedocs.io/zh-cn/latest/api/generated/mmengine.hooks.DistSamplerSeedHook.html#mmengine.hooks.DistSamplerSeedHook) 在分布式训练时调用 Sampler 的 step 方法以确保 shuffle 参数生效。`DistSamplerSeedHook` 默认注册到执行器并且没有可配置的参数，所以无需对其做任何配置。

### RuntimeInfoHook

[RuntimeInfoHook](https://mmengine.readthedocs.io/zh-cn/latest/api/generated/mmengine.hooks.RuntimeInfoHook.html#mmengine.hooks.RuntimeInfoHook) 会在执行器的不同钩子位点将当前的运行时信息（如 epoch、iter、max_epochs、max_iters、lr、metrics等）更新至 message hub 中，以便其他无法访问执行器的模块能够获取到这些信息。`RuntimeInfoHook` 默认注册到执行器并且没有可配置的参数，所以无需对其做任何配置。

### EMAHook

[EMAHook](https://mmengine.readthedocs.io/zh-cn/latest/api/generated/mmengine.hooks.EMAHook.html#mmengine.hooks.EMAHook) 在训练过程中对模型执行指数滑动平均操作，目的是提高模型的鲁棒性。注意：指数滑动平均生成的模型只用于验证和测试，不影响训练。

```python
custom_hooks = [dict(type='EMAHook')]
runner = Runner(custom_hooks=custom_hooks, ...)
runner.train()
```

`EMAHook` 默认使用 `ExponentialMovingAverage`，可选值还有 `StochasticWeightAverage` 和 `MomentumAnnealingEMA`。可以通过设置 `ema_type` 使用其他的平均策略。

```python
custom_hooks = [dict(type='EMAHook', ema_type='StochasticWeightAverage')]
```

更多用法请阅读 [EMAHook API 文档](https://mmengine.readthedocs.io/zh-cn/latest/api/generated/mmengine.hooks.EMAHook.html#mmengine.hooks.EMAHook)。

### EmptyCacheHook

[EmptyCacheHook](https://mmengine.readthedocs.io/zh-cn/latest/api/generated/mmengine.hooks.EmptyCacheHook.html#mmengine.hooks.EmptyCacheHook) 调用 `torch.cuda.empty_cache()` 释放未被使用的显存。可以通过设置 `before_epoch`, `after_iter` 以及 `after_epoch` 参数控制释显存的时机，第一个参数表示在每个 epoch 开始之前，第二参数表示在每次迭代之后，第三个参数表示在每个 epoch 之后。

```python
# 每一个 epoch 结束都会执行释放操作
custom_hooks = [dict(type='EmptyCacheHook', after_epoch=True)]
runner = Runner(custom_hooks=custom_hooks, ...)
runner.train()
```



### SyncBuffersHook

[SyncBuffersHook](https://mmengine.readthedocs.io/zh-cn/latest/api/generated/mmengine.hooks.SyncBuffersHook.html#mmengine.hooks.SyncBuffersHook) 在分布式训练每一轮（epoch）结束时同步模型的 buffer，例如 BN 层的 `running_mean` 以及 `running_var`。

```python
custom_hooks = [dict(type='SyncBuffersHook')]
runner = Runner(custom_hooks=custom_hooks, ...)
runner.train()
```

## 自定义钩子

如果 MMEngine 提供的默认钩子不能满足需求，用户可以自定义钩子，只需继承钩子基类并重写相应的位点方法。

例如，如果希望在训练的过程中判断损失值是否有效，如果值为无穷大则无效，我们可以在每次迭代后判断损失值是否无穷大，因此只需重写 `after_train_iter` 位点。

```python
import torch

from mmengine.registry import HOOKS
from mmengine.hooks import Hook


@HOOKS.register_module()
class CheckInvalidLossHook(Hook):
    """Check invalid loss hook.

    This hook will regularly check whether the loss is valid
    during training.

    Args:
        interval (int): Checking interval (every k iterations).
            Defaults to 50.
    """

    def __init__(self, interval=50):
        self.interval = interval

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        """All subclasses should override this method, if they need any
        operations after each training iteration.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the train loop.
            data_batch (dict or tuple or list, optional): Data from dataloader.
            outputs (dict, optional): Outputs from model.
        """
        if self.every_n_train_iters(runner, self.interval):
            assert torch.isfinite(outputs['loss']),\
                runner.logger.info('loss become infinite or NaN!')
```

我们只需将钩子的配置传给执行器的 `custom_hooks` 的参数，执行器初始化的时候会注册钩子，

```python
from mmengine.runner import Runner

custom_hooks = [
    dict(type='CheckInvalidLossHook', interval=50)
]
runner = Runner(custom_hooks=custom_hooks, ...)  # 实例化执行器，主要完成环境的初始化以及各种模块的构建
runner.train()  # 执行器开始训练
```

便会在每次模型前向计算后检查损失值。

注意，自定义钩子的优先级默认为 `NORMAL (50)`，如果想改变钩子的优先级，则可以在配置中设置 priority 字段。

```python
custom_hooks = [
    dict(type='CheckInvalidLossHook', interval=50, priority='ABOVE_NORMAL')
]
```

也可以在定义类时给定优先级

```python
@HOOKS.register_module()
class CheckInvalidLossHook(Hook):

    priority = 'ABOVE_NORMAL'
```
