# Python之Hook设计

## Hook设计描述


Hook，又称钩子，在C/C++中一般叫做回调函数。一个钩子方法由一个抽象类或具体类声明并实现，而其子类可能会加以扩展。通常在父类中给出的实现是一个空实现（可使用virtual关键字将其定义为虚函数），并以该空实现作为方法的默认实现，当然钩子方法也可以提供一个非空的默认实现.钩子是从功能角度描述这种编程模式，回调则是从函数调用时间角度描述的。

在模板方法模式中，由于面向对象的多态性，子类对象在运行时将覆盖父类对象，子类中定义的方法也将覆盖父类中定义的方法，因此程序在运行时，具体子类的基本方法将覆盖父类中定义的基本方法，子类的钩子方法也将覆盖父类的钩子方法，从而可以通过在子类中实现的钩子方法对父类方法的执行进行约束，实现子类对父类行为的反向控制。

通常理解的hook是在一个已有的方法上加入一些钩子，使得在该方法执行前或执行后另在做一些额外的处理。如我们熟知的windows系统消息响应事件，鼠标点击对程序产生的影响是由程序自己决定的，但是程序的执行是受制于框架（windows系统），框架提供了一些通用的流程执行，但是往往框架或流程在设计时无法完全预料到以后的使用会有什么新需求，或者有些行为只有在运行时才能确定的。这就产生了回调的需求，即用户提供需求，框架负责执行，流程先于具体需求，当触发或者满足某种条件时，执行Hook函数。hook函数的数据也是由用户自己提供的，框架只负责流程执行，这样框架的通用性就能大大提高。


## Hook设计三要素
- hook函数或类：实现自定义操作或功能
- 注册：只有经过注册的Hook才能被系统或框架调用
- 挂载点：通常由系统或框架决定，用户无法修改


## Hook 使用实例
hook是一个编程机制，与语言无关。这里给个python的简单例子，帮助大家理解
```python
import time

class LazyPerson(object):
    def __init__(self, name):
        self.name = name
        self.watch_tv_func = None
        self.have_dinner_func = None

    def get_up(self):
        print("%s get up at:%s" % (self.name, time.time()))

    def go_to_sleep(self):
        print("%s go to sleep at:%s" % (self.name, time.time()))

    def register_tv_hook(self, watch_tv_func):
        self.watch_tv_func = watch_tv_func

    def register_dinner_hook(self, have_dinner_func):
        self.have_dinner_func = have_dinner_func

    def enjoy_a_lazy_day(self):
        # get up
        self.get_up()
        time.sleep(3)
        # watch tv
        # check the watch_tv_func(hooked or unhooked)
        # hooked
        if self.watch_tv_func is not None:
            self.watch_tv_func(self.name)
        # unhooked
        else:
            print("no tv to watch")
        time.sleep(3)
        # have dinner
        # check the have_dinner_func(hooked or unhooked)
        # hooked
        if self.have_dinner_func is not None:
            self.have_dinner_func(self.name)
        # unhooked
        else:
            print("nothing to eat at dinner")
        time.sleep(3)
        self.go_to_sleep()

def watch_daydayup(name):
    print("%s : The program ---day day up--- is funny!!!" % name)

def watch_happyfamily(name):
    print("%s : The program ---happy family--- is boring!!!" % name)

def eat_meat(name):
    print("%s : The meat is nice!!!" % name)

def eat_hamburger(name):
    print("%s : The hamburger is not so bad!!!" % name)



if __name__ == "__main__":
    lazy_tom = LazyPerson("Tom")
    lazy_jerry = LazyPerson("Jerry")
    # register hook
    lazy_tom.register_tv_hook(watch_daydayup)
    lazy_tom.register_dinner_hook(eat_meat)
    lazy_jerry.register_tv_hook(watch_happyfamily)
    lazy_jerry.register_dinner_hook(eat_hamburger)
    # enjoy a day
    lazy_tom.enjoy_a_lazy_day()
    lazy_jerry.enjoy_a_lazy_day()
```

代码运行结果：

```python
Tom get up at:1598599060.6962798
Tom : The program ---day day up--- is funny!!!
Tom : The meat is nice!!!
Tom go to sleep at:1598599069.701241
Jerry get up at:1598599069.7012656
Jerry : The program ---happy family--- is boring!!!
Jerry : The hamburger is not so bad!!!
Jerry go to sleep at:1598599078.7097971
```

## 设计实例: mmcv 中的 Hook 使用

我们看看具体的设计实例：mmcv库的Run类。Run类负责训练流程执行，由用户提供数据。

Hook类是所有hook类的父类，规定了具体的调用名称和挂载点，如before_run、before_epoch、after_epoch、after_run等，注册的hook类需要具体实现自己的需求，例如实现自定义的学习率更新策略，框架会在Run中每个挂载点循环执行用户注册的所有hooks的相应挂载点方法，用户hooks是放在一个有序列表中，按优先级排列，优先级高的在前，先得到执行，优先级也是由用户确定的，这是用户仅有的权力。

```python
class Hook:

    def before_run(self, runner):
        pass

    def after_run(self, runner):
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
        self.before_epoch(runner)

    def before_val_epoch(self, runner):
        self.before_epoch(runner)

    def after_train_epoch(self, runner):
        self.after_epoch(runner)

    def after_val_epoch(self, runner):
        self.after_epoch(runner)

    def before_train_iter(self, runner):
        self.before_iter(runner)

    def before_val_iter(self, runner):
        self.before_iter(runner)

    def after_train_iter(self, runner):
        self.after_iter(runner)

    def after_val_iter(self, runner):
        self.after_iter(runner)

    def every_n_epochs(self, runner, n):
        return (runner.epoch + 1) % n == 0 if n > 0 else False

    def every_n_inner_iters(self, runner, n):
        return (runner.inner_iter + 1) % n == 0 if n > 0 else False

    def every_n_iters(self, runner, n):
        return (runner.iter + 1) % n == 0 if n > 0 else False

    def end_of_epoch(self, runner):
        return runner.inner_iter + 1 == len(runner.data_loader)
```

观察 Hook 类中的方法不难发现，Hook 类将训练过程中我们可能采取额外操作（如调整学习率，存储模型和日志文件，打印训练信息等）的时间点分为开始训练前、一个 iteration 前、一个 iteration 后、一个 epoch 前、一个 epoch 后、每 n 个 iteration 后、每 n 个 epoch 后，这些时间点又分为 train 和 validate 过程（在基类中默认两个过程采取相同的操作）。

Hook 类的定义类似于一个抽象类，仅仅定义了一组接口而没有具体实现，这意味着我们必须通过继承的方式来使用。如果希望在某几个时间点采取一些特定的操作，需要定义一个新的类并继承 Hook 类，然后重写各个时间点对应的方法，最后调用 Runner 对象的 register_hook 方法在对象中注册这个 hook。


### Hook调用
Runner 类中维护了一个存放 hook 对象的列表 self._hooks，在每个时间点会通过 call_hook 方法依次调用列表中所有 hook 对象对应的接口以执行相关操作，call_hook 方法定义为：
```python
def call_hook(self, fn_name):
    for hook in self._hooks:
        getattr(hook, fn_name)(self)
```
其中 fn_name 是一个字符串对象，表示希望执行的方法名，这里利用了 python 的内建函数 getattr 来获得 hook 对象中同名方法的引用。用户仅仅需要实现自己所需要的hook，如果没有自定义的hook，框架会调用父类Hook中相应的方法。父类Hook可能提供了一些默认行为，也可能什么都没做。

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
这个类重写了 before_run、before_train_epoch、before_train_iter 方法，其构造函数的参数 by_epoch 如果为 True 则表明我们以 epoch 为单位计量训练进程，否则以 iteration 为单位。warmup 参数为字符串，指定了 warmup 算法中学习率的变化方式，warmup_iters 和 warmup_ratio 分别指定了 warmup 的 iteration 数和增长比例。

从代码中可以看出，在训练开始前，LrUpdaterHook 对象首先会设置 Runner 对象中所维护的优化器的各项参数，然后在每个 iteration 和 epoch 开始前检查学习率和 iteration（epoch）的值，然后计算下一次迭代过程的学习率的值并修改 Runner 中的学习率。需要注意的是，LrUpdaterHook 类并未实现 get_lr 方法，要使用 LrUpdaterHook 类必须通过继承的方式并给出 get_lr 方法的实现。

换句话说，LrUpdaterHook 类仅提供了在相应时间修改学习率的代码，至于学习率的衰减方式则应该根据需要自行设置。Hook 机制的好处在于，当我们需要在某些时间点添加一组特定的操作时，只需要编写相应的 hook 类将操作封装并调用 Runner 对象的 register_hook 方法注册即可，这使得整个训练的过程变得更容易定制。

其实实现hook时，用户的疑问往往是自定义hook需要使用的数据从哪里来？显然用户不知道Run类中有哪些数据。用户其实是知道的，因为Run中原本是没有数据的，它仅是一个流程执行类，其中的数据均来自与用户创建run时传入的，如runner.LrUpdaterHook。所以可以看到，一个hook仅仅需要两个元素，一个是执行者，这里是runner，另外一个是执行时间（触发条件，挂载点）。

### Hook注册
hook的注册过程比较简单，因为触发是按框架定义的流程顺序主动调用的，因此仅需要按优先级插入到有序列表中即可。
```python
def register_hook(self, hook, priority='NORMAL'):
    """Register a hook into the hook list.

    The hook will be inserted into a priority queue, with the specified
    priority (See :class:`Priority` for details of priorities).
    For hooks with the same priority, they will be triggered in the same
    order as they are registered.

    Args:
        hook (:obj:`Hook`): The hook to be registered.
        priority (int or str or :obj:`Priority`): Hook priority.
            Lower value means higher priority.
    """
    assert isinstance(hook, Hook)
    if hasattr(hook, 'priority'):
        raise ValueError('"priority" is a reserved attribute for hooks')
    priority = get_priority(priority)
    hook.priority = priority
    # insert the hook to a sorted list
    inserted = False
    for i in range(len(self._hooks) - 1, -1, -1):
        if priority >= self._hooks[i].priority:
            self._hooks.insert(i + 1, hook)
            inserted = True
            break
    if not inserted:
        self._hooks.insert(0, hook)
```


现在我们回过头来看 Runner 类的 run 方法，看下面的代码
```python
def run(self, data_loaders, workflow, max_epochs, **kwargs):
    """Start running.

    Args:
        data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
            and validation.
        workflow (list[tuple]): A list of (phase, epochs) to specify the
            running order and epochs. E.g, [('train', 2), ('val', 1)] means
            running 2 epochs for training and 1 epoch for validation,
            iteratively.
        max_epochs (int): Total training epochs.
    """
    assert isinstance(data_loaders, list)
    assert mmcv.is_list_of(workflow, tuple)
    assert len(data_loaders) == len(workflow)

    self._max_epochs = max_epochs
    for i, flow in enumerate(workflow):
        mode, epochs = flow
        if mode == 'train':
            self._max_iters = self._max_epochs * len(data_loaders[i])
            break

    work_dir = self.work_dir if self.work_dir is not None else 'NONE'
    self.logger.info('Start running, host: %s, work_dir: %s',
                        get_host_info(), work_dir)
    self.logger.info('workflow: %s, max: %d epochs', workflow, max_epochs)
    self.call_hook('before_run')

    while self.epoch < max_epochs:
        for i, flow in enumerate(workflow):
            mode, epochs = flow
            if isinstance(mode, str):  # self.train()
                if not hasattr(self, mode):
                    raise ValueError(
                        f'runner has no method named "{mode}" to run an '
                        'epoch')
                epoch_runner = getattr(self, mode)
            else:
                raise TypeError(
                    'mode in workflow must be a str, but got {}'.format(
                        type(mode)))

            for _ in range(epochs):
                if mode == 'train' and self.epoch >= max_epochs:
                    break
                epoch_runner(data_loaders[i], **kwargs)

    time.sleep(1)  # wait for some hooks like loggers to finish
    self.call_hook('after_run')
```
其中 data_loaders 表示数据加载的对象，max_epochs 表示训练的 epoch 数，workflow 是一个列表对象，需要我们在配置文件中指定，表示在每一个 epoch 中需要采取的行为，例如 `workflow = [('train', 1)]` 表示在一个 epoch 中调用 Runner 的 train 方法训练一个 epoch，train 方法的定义如下:
```python
def train(self, data_loader, **kwargs):
    self.model.train()
    self.mode = 'train'
    self.data_loader = data_loader
    self._max_iters = self._max_epochs * len(self.data_loader)
    self.call_hook('before_train_epoch')
    time.sleep(2)  # Prevent possible deadlock during epoch transition
    for i, data_batch in enumerate(self.data_loader):
        self._inner_iter = i
        self.call_hook('before_train_iter')
        if self.batch_processor is None:
            outputs = self.model.train_step(data_batch, self.optimizer,
                                            **kwargs)
        else:
            outputs = self.batch_processor(
                self.model, data_batch, train_mode=True, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            ' must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'],
                                    outputs['num_samples'])
        self.outputs = outputs
        self.call_hook('after_train_iter')
        self._iter += 1

    self.call_hook('after_train_epoch')
    self._epoch += 1
```

不难看出，train 方法定义的就是训练的过程。run 方法中的 while 循环表示的就是一个完整的训练过程，故而在这个循环的前后分别执行了 self.call_hook('before_run')和 self.call_hook('after_run')，而 train 方法中的 for 循环定义了一个 epoch 训练的过程，并且每次循环都表示一次 iteration，因此在整个循环前后分别执行了 self.call_hook('before_train_epoch')和 self.call_hook('after_train_epoch')，在每次迭代前后又分别执行 self.call_hook('before_train_iter')和 self.call_hook('after_train_iter')。


### 需要注意的地方
如果有一个hook需要在两个不同时机执行两个需求，如在before_train_epoch和after_train_epoch，但是恰巧这两个需求的优先级不同，这个时候建议写成两个hook，每个hook只负责做一件事，这也是编程中一般原则吧。
