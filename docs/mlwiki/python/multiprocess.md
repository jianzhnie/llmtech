# 深入理解Python多进程

##  概览

### **为什么选择多进程**

1. **充分利用多核处理器**：多进程可以同时利用多个CPU核心，实现并行处理，加快任务执行速度。
2. **避免GIL的影响**：Python的全局解释器锁（GIL）限制了多线程并发执行时的效率，而多进程避免了这一限制，可以更好地利用多核处理器。
3. **提高程序稳定性**：由于多进程拥有独立的内存空间，进程之间互不影响，因此在处理一些需要隔离环境的任务时更加稳定可靠。
4. **适用于CPU密集型任务**：对于需要大量计算的任务，多进程能够更好地利用计算资源，提高程序的执行效率。

### 操作系统基础知识

Unix/Linux操作系统提供了一个`fork()`系统调用，它非常特殊。普通的函数调用，调用一次，返回一次，但是`fork()`调用一次，返回两次，因为操作系统自动把当前进程（称为父进程）复制了一份（称为子进程），然后，分别在父进程和子进程内返回。

子进程永远返回`0`，而父进程返回子进程的ID。这样做的理由是，一个父进程可以fork出很多子进程，所以，父进程要记下每个子进程的ID，而子进程只需要调用`getppid()`就可以拿到父进程的ID。

Python的`os`模块封装了常见的系统调用，其中就包括`fork`，可以在Python程序中轻松创建子进程：

```python
import os

print('Process (%s) start...' % os.getpid())
# Only works on Unix/Linux/Mac:
pid = os.fork()
if pid == 0:
    print('I am child process (%s) and my parent is %s.' %
          (os.getpid(), os.getppid()))
else:
    print('I (%s) just created a child process (%s).' % (os.getpid(), pid))
```

运行结果如下：

```shell
Process (876) start...
I (876) just created a child process (877).
I am child process (877) and my parent is 876.
```

由于Windows没有`fork`调用，上面的代码在Windows上无法运行。而Mac系统是基于BSD（Unix的一种）内核，所以，在Mac下运行是没有问题的。

有了`fork`调用，一个进程在接到新任务时就可以复制出一个子进程来处理新任务，常见的Apache服务器就是由父进程监听端口，每当有新的http请求时，就`fork`出子进程来处理新的http请求。

如果你打算编写多进程的服务程序，Unix/Linux无疑是正确的选择。由于Windows没有`fork`调用，难道在Windows上无法用Python编写多进程的程序？

由于Python是跨平台的，自然也应该提供一个跨平台的多进程支持。`multiprocessing`模块就是跨平台版本的多进程模块。

## **Multiprocessing 多进程模块**

`multiprocessing` 是 Python 中用于支持多进程编程的内置模块，可以实现并行处理任务，充分利用多核处理器。通过`Process`类可以创建新的进程，通过`Pool` 类可以创建进程池，实现并行处理任务。多进程之间可以通过队列（`Queue`）、管道（`Pipe`）等方式进行通信，从而实现数据共享和协作。

### 启动方式

python3中支持三种方式启动多进程：`spawn`、`fork`、`forkserver`。

1. spawn是启动一个全新的python解释器进程，这个进程不继承父进程的任何不必要的文件描述符或其它资源。
2. fork是使用`os.fork()`系统调用启动一个python解释器进程，因为是fork调用，这个启动的进程可以继承父进程中的资源。fork出的子进程虽然与父进程是不同的内存空间，但在linux下它是的copy-on-write方式实现的，因此即使创建了很多子进程，实际上看子进程并不会消耗多少内存。看起来fork方式创建子进程很好，但实际上还是存在一些问题的。如果父进程是一个多线程程序，用fork系统调用是很危险的，很容易造成死锁，详见[这里](https://pythonspeed.com/articles/python-multiprocessing/)。
3. 但fork系统调用又确实是启动子进程最高效的方法，于是官方又提供`forkserver`。当父进程需要启动子进程时，实际上是向一个`Fork Server`进程发指令，由它调用`os.fork()`产生子进程的。这个`Fork Server`进程是一个单线程进程，因此调用fork不会产生风险。`forkserver`的实现方式也挺有意思的，代码不长，源码在这里，[multiprocessing/forkserver.py](https://github.com/python/cpython/blob/master/Lib/multiprocessing/forkserver.py)。

不同的操作系统下默认的子进程启动方式是不一样的， 在Unix/Linux下，`multiprocessing`模块封装了`fork()`调用，使我们不需要关注`fork()`的细节。由于Windows没有`fork`调用，因此，`multiprocessing`需要“模拟”出`fork`的效果，父进程所有Python对象都必须通过pickle序列化再传到子进程去，所以，如果`multiprocessing`在Windows下调用失败了，要先考虑是不是pickle失败了。目前有两种启动子进程方式。

1. 通过`multiprocessing.set_start_method`方法全局改变。

   ```python
   import multiprocessing as mp

   if __name__ == '__main__':
       mp.set_start_method('spawn')
   ```

2. 通过`multiprocessing.get_context`方法得到一个上下文对象，通过此上下文对象创建的多进程相关对象将使用特定的子进程启动方式。

   ```python
   import multiprocessing as mp

   def foo(q):
       q.put('hello')

   if __name__ == '__main__':
       ctx = mp.get_context('spawn')
       q = ctx.Queue()
       p = ctx.Process(target=foo, args=(q,))
   ```

### 创建进程

`multiprocessing.Process`类用于创建新的进程。通过实例化`Process` 类并传入要执行的函数，可以创建一个新的进程。调用`start()`方法启动进程，调用`join()`方法等待进程结束。每个`Process` 实例都有自己独立的内存空间。

#### Python多进程实现方法一

下面的例子演示了启动一个子进程并等待其结束：

```python
from multiprocessing import Process
import os


# 子进程要执行的代码
def run_proc(name):
    print('Run child proces: %s (%s)...' % (name, os.getpid()))


if __name__ == '__main__':
    print('Parent process %s.' % os.getpid())
    p = Process(target=run_proc, args=('test', ))
    print('Child process will start.')
    p.start()
    p.join()
    print('Child process end.')
```

执行结果如下：

```plain
Parent process 928.
Child process will start.
Run child process test (929)...
Process end.
```

创建子进程时，只需要传入一个执行函数和函数的参数，创建一个`Process`实例，用`start()`方法启动，这样创建进程比`fork()`还要简单。`join()`方法可以等待子进程结束后再继续往下运行，通常用于进程间的同步。

#### Python多进程实现方法二

Python多进程的第二种实现方式是通过类继承的方法来实现的。

```python
from multiprocessing import Process


class MyProcess(Process):
    # 继承Process类

    def __init__(self, name):
        super(MyProcess, self).__init__()
        self.name = name

    def run(self):
        print('Test Python Process: %s' % self.name)


if __name__ == '__main__':
    process_list = []
    for i in range(5):
        # 开启5个子进程执行fun1函数
        p = MyProcess('P_' + str(i))
        # 实例化进程对象
        p.start()
        process_list.append(p)

    for i in process_list:
        p.join()

    print('Finished')
```

运行结果：

```
Test Python Process: P_0
Test Python Process: P_1
Test Python Process: P_2
Test Python Process: P_3
Test Python Process: P_4
Finished
```

### **创建进程池**

`multiprocessing.Pool`类用于创建进程池，可以方便地管理多个进程。通过`Pool`类的`map()`、`apply()` 等方法，可以将任务分配给进程池中的多个进程并行执行。进程池会自动管理进程的创建和销毁，提高了并行处理的效率。

Pool 默认大小是CPU的核数，我们也可以通过在Pool中传入processes参数自定义需要的核数量。定义进程池之后，就可以让进程池对应某一个函数，通过向进程池中传入数据从而返回函数值。 Pool和之前的Process的不同点是传入Pool的函数有返回值，而Process的没有返回值。

- map方法：用map()获取结果，在map()中需要放入函数和需要迭代运算的值，然后它会自动分配给CPU核，返回结果。
- apply_async方法: apply_async() 中只能传递一个值，它只会放入一个核进行运算，但是传入值时要注意是元组类型，所以在传入值后需要加逗号, 同时需要用get()方法获取返回值。如果要实现map()的效果，需要将apply_async方法做成一个列表的形式。
- 进程池最后要加join方法，这样进程池运行完毕后才向下进行，如果不加的话可能导致进程池还未运行完程序已经finished。

创建`multiprocessing.Pool`对象时，有几个参数有些作用：

1. `initializer`及`initargs`，通过这两个参数可即将对在进程池中创建的进程进行部分初始化工作。
2. `maxtasksperchild`，可以通过这个参数设定进程池中每个进程最大处理的任务数，超过任务数后，会启动一个新的进程来代替该进程。为什么会有这个需求？

#### apply_async方法

例如 如果要启动大量的子进程，可以用进程池的方式批量创建子进程：

```python
from multiprocessing import Pool
import os, time, random


def worker(name) -> None:
    print('Run task %s (%s)...' % (name, os.getpid()))
    start = time.time()
    time.sleep(random.random() * 3)
    end = time.time()
    print('Task %s runs %0.2f seconds.' % (name, (end - start)))


if __name__ == '__main__':
    print('Parent process %s.' % os.getpid())
    p = Pool(processes=4)
    for i in range(5):
        p.apply_async(worker, args=(i, ))
    print('Waiting for all subprocesses done...')
    p.close()  # 关闭进程池，其他进程无法加入
    p.join()  # 等待所有进程执行完毕，调用前必须调用close方法
    print('All subprocesses done.')
```

执行结果如下：

```plain
Parent process 669.
Waiting for all subprocesses done...
Run task 0 (671)...
Run task 1 (672)...
Run task 2 (673)...
Run task 3 (674)...
Task 2 runs 0.14 seconds.
Run task 4 (673)...
Task 1 runs 0.27 seconds.
Task 3 runs 0.86 seconds.
Task 0 runs 1.41 seconds.
Task 4 runs 1.91 seconds.
All subprocesses done.
```

代码解读：

对`Pool`对象调用`join()`方法会等待所有子进程执行完毕，调用`join()`之前必须先调用`close()`，调用`close()`之后就不能继续添加新的`Process`了。

请注意输出的结果，task `0`，`1`，`2`，`3`是立刻执行的，而task `4`要等待前面某个task完成后才执行，这是因为`Pool`的默认大小在我的电脑上是4，因此，最多同时执行4个进程。这是`Pool`有意设计的限制，并不是操作系统的限制。如果改成：`p = Pool(5)` 就可以同时跑5个进程。

由于`Pool`的默认大小是CPU的核数，如果你不幸拥有8核CPU，你要提交至少9个子进程才能看到上面的等待效果。

####  map 方法

```python
import multiprocessing as mul

def f(x):
    return x ** 2


if __name__ == '__main__':
    pool = mul.Pool(5)
    rel = pool.map(f, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    print(rel)

```

运行结果：

```python
[1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
```

我们创建了一个容许5个进程的进程池 (Process Pool) 。Pool运行的每个进程都执行f()函数。我们利用map()方法，将f()函数作用到表的每个元素上。这与built-in的map()函数类似，只是这里用5个进程并行处理。如果进程运行结束后，还有需要处理的元素，那么的进程会被用于重新运行f()函数。

下面这个例子的主要工作就是将遍历传入的文件夹中的图片文件，一一生成缩略图，并将这些缩略图保存到特定文件夹中。

```python
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image


def get_image_paths(folder):
    return [
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.lower().endswith(('.jpeg', '.jpg', '.png'))
    ]


def create_thumbnail(filename, size=(75, 75)):
    with Image.open(filename) as im:
        im.thumbnail(size, Image.LANCZOS)
        base, fname = os.path.split(filename)
        save_path = os.path.join(base, 'thumb', fname)
        im.save(save_path)
    return save_path


def process_images(image_dir, size=(75, 75), max_workers=None):
    image_paths = get_image_paths(image_dir)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(create_thumbnail, img_path, size)
            for img_path in image_paths
        ]

        for future in as_completed(futures):
            try:
                result = future.result()
                print(f'Thumbnail created: {result}')
            except Exception as e:
                print(f'Error processing image: {e}')


if __name__ == '__main__':
    root_dir = 'work_dir'
    image_dir = os.path.join(root_dir, 'images')
    save_dir = os.path.join(root_dir, 'thumb')

    image_size = (75, 75)
    process_images(image_dir, save_dir, image_size)
```

map 函数并不支持手动线程管理，反而使得相关的 debug 工作也变得异常简单。

### **进程间通信**

Python的`multiprocessing`模块包装了底层的机制，提供了`Queue`、`Pipes`等多种方式来交换数据。

- **Queue**：`multiprocessing.Queue`类提供了进程间通信的队列。多个进程可以通过共享的队列进行数据交换，实现进程间的通信。队列是线程/进程安全的，可以在多个进程之间安全地传递数据。
- **Pipe**：`multiprocessing.Pipe` 类提供了进程间通信的管道。管道包含两个连接，每个连接对应一个进程，可以双向传递数据。通过`Pipe`可以实现两个进程之间的通信。
- **Pickle**：`pickle`模块用于序列化和反序列化 Python 对象，可以将对象转换为字节流进行传输。在进程间通信中，可以使用`pickle` 将对象序列化后传输，再在另一端反序列化得到原始对象。

#### 对列 Queue

Queue的功能是将每个核或线程的运算结果放在队里中， 等到每个线程或核运行完毕后再从队列中取出结果， 继续加载运算。多进程调用的函数不能有返回值(不能return), 所以使用Queue存储多个进程运算的结果。

- **put**方法：插入数据到队列。
- **get**方法：从队列中读取并删除一个元素。

>  Put方法用以插入数据到队列中，它还有两个可选参数：blocked和timeout。
>
> 如果blocked为True（默认值），并且timeout为正值，该方法会阻塞timeout指定的时间，直到该队列有剩余的空间。如果超时，会抛出Queue.Full异常。如果blocked为False，但该Queue已满，会立即抛出Queue.Full异常。
>
> ·Get方法可以从队列读取并且删除一个元素。同样，Get方法有两个可选参数：blocked和timeout。
>
> 如果blocked为True（默认值），并且timeout为正值，那么在等待时间内没有取到任何元素，会抛出Queue.Empty异常。如果blocked为False，分两种情况：如果Queue有一个值可用，则立即返回该值；否则，如果队列 为空，则立即抛出Queue.Empty异常。

我们以`Queue`为例，在父进程中创建两个子进程，一个往`Queue`里写数据，一个从`Queue`里读数据：

```python
from multiprocessing import Process, Queue
import time
import random


def producer(queue):
    print('生产者进程开始')
    for i in range(5):
        item = random.randint(1, 100)
        queue.put(item)
        print(f'生产者放入: {item}')
        time.sleep(random.random())
    queue.put(None)  # 发送结束信号
    print('生产者进程结束')


def consumer(queue):
    print('消费者进程开始')
    while True:
        item = queue.get()
        if item is None:
            break
        print(f'消费者取出: {item}')
        time.sleep(random.random())
    print('消费者进程结束')


if __name__ == '__main__':
    q = Queue()

    p1 = Process(target=producer, args=(q, ))
    p2 = Process(target=consumer, args=(q, ))

    p1.start()
    p2.start()

    p1.join()
    p2.join()

    print('所有进程结束')

```

运行结果如下：

```shell
生产者进程开始
生产者放入: 72
消费者进程开始
消费者取出: 72
生产者放入: 77
消费者取出: 77
生产者放入: 20
消费者取出: 20
生产者放入: 94
生产者放入: 8
生产者进程结束
消费者取出: 94
消费者取出: 8
消费者进程结束
所有进程结束
```

这个例子展示了如何使用Queue在生产者进程和消费者进程之间进行通信:

1. 我们创建一个Queue对象q。
2. 生产者进程使用queue.put()方法将数据放入队列。
3. 消费者进程使用queue.get()方法从队列中取出数据。
4. 生产者在完成后发送一个None作为结束信号。
5. 消费者在收到None时退出循环。
6. 主进程等待两个子进程结束后才结束。

需要注意的是,Queue是线程安全和进程安全的。它使用锁和信号量来确保多个进程可以安全地访问队列,而不会发生数据竞争或其他并发问题。

#### 管道 Pipe

Pipe提供了一种简单而高效的方式来实现进程间的通信。调用`Pipe()`方法会返回一对connection对象， parent_conn, child_conn = Pipe() ，管道的两端可以放在主进程或子进程内，我在实验中没发现主管道口 parent_conn 和子管道口 child_conn 的区别。

> Pipe方法有duplex 参数，如果duplex参数为True（默认值），那么这个管道是全双工模式，也就是说 conn1和conn2均可收发。若duplex为False，conn1只负责接收消息，conn2只负 责发送消息。send和recv方法分别是发送和接收消息的方法。例如，在全双工模式 下，可以调用conn1.send发送消息，conn1.recv接收消息。如果没有消息可接 收，recv方法会一直阻塞。如果管道已经被关闭，那么recv方法会抛出EOFError。

下面这个例子展示了 Pipe的基本用法,包括如何创建Pipe,如何在不同进程间发送和接收数据。

```python
from multiprocessing import Process, Pipe


def worker(conn):
    print('子进程开始工作')
    # 从管道接收数据
    msg = conn.recv()
    print(f'子进程收到消息: {msg}')

    # 处理数据
    result = f'处理结果: {msg.upper()}'

    # 将结果发送回主进程
    conn.send(result)
    print('子进程完成工作')
    conn.close()


if __name__ == '__main__':
    # 创建管道
    parent_conn, child_conn = Pipe()

    # 创建子进程
    p = Process(target=worker, args=(child_conn, ))
    p.start()

    # 主进程发送数据
    print('主进程发送消息: hello world')
    parent_conn.send('hello world')

    # 主进程接收结果
    result = parent_conn.recv()
    print(f'主进程收到结果: {result}')

    # 等待子进程结束
    p.join()

    print('所有进程结束')
```

结果：

```shell
子进程开始工作
主进程发送消息: hello world
子进程收到消息: hello world
子进程完成工作
主进程收到结果: 处理结果: HELLO WORLD
所有进程结束
```

这个例子展示了如何使用Pipe在父进程和子进程之间进行双向通信:

1. 我们首先创建一个Pipe,它返回两个连接对象(parent_conn和child_conn)。
2. 我们创建一个子进程,并将child_conn作为参数传递给它。
3. 在子进程中,我们使用conn.recv()从管道接收数据,处理数据,然后使用conn.send()将结果发送回去。
4. 在主进程中,我们使用parent_conn.send()发送数据给子进程,然后使用parent_conn.recv()接收子进程返回的结果。
5. 最后,我们等待子进程结束并打印完成消息。

#### 序列化 Pickle

Pickle 模块可以序列化大多数Python对象,使其成为多进程通信的强大工具。

下面这个示例展示了如何使用pickle模块在多进程之间传递复杂的Python对象。

```python
import multiprocessing
import pickle
import random
import time


def producer(queue):
    print('生产者进程开始')
    for i in range(5):
        item = {'id': i, 'value': random.randint(1, 100)}
        serialized_item = pickle.dumps(item)
        queue.put(serialized_item)
        print(f'生产者放入: {item}')
        time.sleep(random.random())

    # 发送结束信号
    queue.put(pickle.dumps(None))
    print('生产者进程结束')


def consumer(queue):
    print('消费者进程开始')
    while True:
        serialized_item = queue.get()
        item = pickle.loads(serialized_item)
        if item is None:
            break
        print(f'消费者取出: {item}')
        time.sleep(random.random())
    print('消费者进程结束')


if __name__ == '__main__':
    ctx = multiprocessing.get_context('spawn')
    q = ctx.Queue()

    p1 = ctx.Process(target=producer, args=(q, ))
    p2 = ctx.Process(target=consumer, args=(q, ))

    p1.start()
    p2.start()

    p1.join()
    p2.join()

    print('所有进程结束')
```

执行结果：

```python
生产者进程开始
生产者放入: {'id': 0, 'value': 83}
消费者进程开始
消费者取出: {'id': 0, 'value': 83}
生产者放入: {'id': 1, 'value': 43}
消费者取出: {'id': 1, 'value': 43}
生产者放入: {'id': 2, 'value': 18}
消费者取出: {'id': 2, 'value': 18}
生产者放入: {'id': 3, 'value': 6}
消费者取出: {'id': 3, 'value': 6}
生产者放入: {'id': 4, 'value': 54}
生产者进程结束
消费者取出: {'id': 4, 'value': 54}
消费者进程结束
所有进程结束
```

这个示例与之前的代码类似,但有以下几个关键区别:

1. 我们使用pickle.dumps()将对象序列化,然后将序列化后的数据放入队列。
2. 消费者使用pickle.loads()反序列化从队列中获取的数据。
3. 我们使用了更复杂的数据结构(字典)来演示pickle可以处理复杂对象。
4. 使用multiprocessing.get_context('spawn')来确保跨平台兼容性,特别是在Windows上。
5. 结束信号仍然是None,但现在它被序列化后发送。

需要注意的是,虽然pickle非常方便,但它也有一些安全隐患。在处理不信任的数据时,应该谨慎使用pickle。在实际应用中,可能需要考虑使用更安全的序列化方法,如JSON(对于简单数据结构)或专门的安全序列化库。


##  多进程和多线程

### **进程与线程概念介绍**

- **进程**：进程是程序的一次执行过程，是系统资源分配的基本单位。每个进程都有自己独立的内存空间，包括代码段、数据段、堆栈等。进程之间相互独立，通信需要特殊手段。
- **线程**：线程是进程中的一个执行流，是CPU调度的基本单位。同一进程内的线程共享相同的内存空间，包括代码段、数据段等。线程之间可以直接访问共享的内存，通信更方便。

由于线程是操作系统直接支持的执行单元，因此，高级语言通常都内置多线程的支持，Python也不例外，并且，Python的线程是真正的Posix Thread，而不是模拟出来的线程。

Python的标准库提供了两个模块：`_thread`和`threading`，`_thread`是低级模块，`threading`是高级模块，对`_thread`进行了封装。绝大多数情况下，我们只需要使用`threading`这个高级模块。

启动一个线程就是把一个函数传入并创建`Thread`实例，然后调用`start()`开始执行：

```python
import time
import threading

# 新线程执行的代码:
def loop():
    print('Thread %s is running...' % threading.current_thread().name)
    n = 0
    while n < 5:
        n = n + 1
        print('Thread %s >>> %s' % (threading.current_thread().name, n))
        time.sleep(1)
    print('Thread %s ended.' % threading.current_thread().name)


if __name__ == '__main__':
    print('Thread %s is running...' % threading.current_thread().name)
    thread = threading.Thread(target=loop, name='LoopThread')
    thread.start()
    thread.join()
    print('Thread %s ended.' % threading.current_thread().name)
```

执行结果如下：

```plain
Thread MainThread is running...
Thread LoopThread is running...
Thread LoopThread >>> 1
Thread LoopThread >>> 2
Thread LoopThread >>> 3
Thread LoopThread >>> 4
Thread LoopThread >>> 5
Thread LoopThread ended.
Thread MainThread ended.
```

由于任何进程默认就会启动一个线程，我们把该线程称为主线程，主线程又可以启动新的线程，Python的`threading`模块有个`current_thread()`函数，它永远返回当前线程的实例。主线程实例的名字叫`MainThread`，子线程的名字在创建时指定，我们用`LoopThread`命名子线程。名字仅仅在打印时用来显示，完全没有其他意义，如果不起名字Python就自动给线程命名为`Thread-1`，`Thread-2`……

下面来看多进程(multiprocessing)和多线程（multi-threading）对比的另外一个测试程序：

```python
import multiprocessing as mp
import threading
import time

MAX = 10000000


def job(q):
    res = 0
    for i in range(MAX):
        res += i + i**2 + i**3
    q.put(res)


def multiprocess():
    q = mp.Queue()
    p1 = mp.Process(target=job, args=(q, ))
    p2 = mp.Process(target=job, args=(q, ))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    res1 = q.get()
    res2 = q.get()
    print('multiprocess:', res1 + res2)


def normal():
    res = 0
    for _ in range(2):
        for i in range(MAX):
            res += i + i**2 + i**3
    print('normal:', res)


def multithread():
    q = mp.Queue()
    t1 = threading.Thread(target=job, args=(q, ))
    t2 = threading.Thread(target=job, args=(q, ))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    res1 = q.get()
    res2 = q.get()
    print('multithreading:', res1 + res2)


if __name__ == '__main__':
    st = time.time()
    normal()
    st1 = time.time()
    print('normal time:', st1 - st)
    multithread()
    st2 = time.time()
    print('multithreading time:', st2 - st1)
    multiprocess()
    print('multiprocess time:', time.time() - st2)
```

运行结果：

```shell
normal: 4999999666666716666660000000
normal time: 17.222501039505005
multithreading: 4999999666666716666660000000
multithreading time: 13.053311824798584
multicore: 4999999666666716666660000000
multiprocess time: 6.818678140640259
```

从上述结果来看，多进程的时间是要小于多线程和正常程序的，多线程的时间与正常时间相差无几。原因是Python解释器有一个全局解释器锁（GIL），导致每个Python进程最多同时运行一个线程，因此Python多线程程序并不能改善程序性能，不能发挥CPU多核的优势，但是多进程程序可以不受影响。

### **线程与进程的区别**

1. **资源占用**：线程比进程轻量，创建和销毁线程的开销小，占用的资源少。进程拥有独立的内存空间，资源消耗较大。
2. **通信方式**：线程之间共享同一进程的内存空间，可以直接访问共享数据，通信更方便。进程之间通信需要特殊手段，如队列、管道等。
3. **并发性**：多线程可以实现并发执行，但受全局解释器锁（GIL）限制，无法利用多核处理器。多进程可以充分利用多核处理器，实现真正的并行处理。
4. **稳定性**：由于线程共享内存，线程之间的错误可能会影响整个进程。而进程之间相互独立，一个进程崩溃不会影响其他进程。
5. **适用场景**：线程适合I/O密集型任务，如网络请求、文件操作等；进程适合CPU密集型任务，如大量计算、图像处理等。

总之，线程适合处理需要频繁I/O操作的任务，进程适合处理需要大量计算的任务。在Python中，多线程受到全局解释器锁的限制，多进程能更好地利用多核处理器，选择合适的并发编程方式可以提高程序的运行效率。

### 选多进程还是多线程

首先，要实现多任务，通常我们会设计Master-Worker模式，Master负责分配任务，Worker负责执行任务，因此，多任务环境下，通常是一个Master，多个Worker。

如果用多进程实现Master-Worker，主进程就是Master，其他进程就是Worker。

如果用多线程实现Master-Worker，主线程就是Master，其他线程就是Worker。

多进程模式最大的优点就是稳定性高，因为一个子进程崩溃了，不会影响主进程和其他子进程。（当然主进程挂了所有进程就全挂了，但是Master进程只负责分配任务，挂掉的概率低）著名的Apache最早就是采用多进程模式。

多进程模式的缺点是创建进程的代价大，在Unix/Linux系统下，用`fork`调用还行，在Windows下创建进程开销巨大。另外，操作系统能同时运行的进程数也是有限的，在内存和CPU的限制下，如果有几千个进程同时运行，操作系统连调度都会成问题。

多线程模式通常比多进程快一点，但是也快不到哪去，而且，多线程模式致命的缺点就是任何一个线程挂掉都可能直接造成整个进程崩溃，因为所有线程共享进程的内存。在Windows上，如果一个线程执行的代码出了问题，你经常可以看到这样的提示：“该程序执行了非法操作，即将关闭”，其实往往是某个线程出了问题，但是操作系统会强制结束整个进程。

在Windows下，多线程的效率比多进程要高，所以微软的IIS服务器默认采用多线程模式。由于多线程存在稳定性的问题，IIS的稳定性就不如Apache。为了缓解这个问题，IIS和Apache现在又有多进程+多线程的混合模式，真是把问题越搞越复杂。

#### 计算密集型 vs. IO密集型

是否采用多任务的第二个考虑是任务的类型。我们可以把任务分为计算密集型和IO密集型。

计算密集型任务的特点是要进行大量的计算，消耗CPU资源，比如计算圆周率、对视频进行高清解码等等，全靠CPU的运算能力。这种计算密集型任务虽然也可以用多任务完成，但是任务越多，花在任务切换的时间就越多，CPU执行任务的效率就越低，所以，要最高效地利用CPU，计算密集型任务同时进行的数量应当等于CPU的核心数。

计算密集型任务由于主要消耗CPU资源，因此，代码运行效率至关重要。Python这样的脚本语言运行效率很低，完全不适合计算密集型任务。对于计算密集型任务，最好用C语言编写。

第二种任务的类型是IO密集型，涉及到网络、磁盘IO的任务都是IO密集型任务，这类任务的特点是CPU消耗很少，任务的大部分时间都在等待IO操作完成（因为IO的速度远远低于CPU和内存的速度）。对于IO密集型任务，任务越多，CPU效率越高，但也有一个限度。常见的大部分任务都是IO密集型任务，比如Web应用。

IO密集型任务执行期间，99%的时间都花在IO上，花在CPU上的时间很少，因此，用运行速度极快的C语言替换用Python这样运行速度极低的脚本语言，几乎无法提升运行效率。对于IO密集型任务，最合适的语言就是开发效率最高（代码量最少）的语言，脚本语言是首选，C语言开发效率最差。

#### 异步IO

考虑到CPU和IO之间巨大的速度差异，一个任务在执行的过程中大部分时间都在等待IO操作，单进程单线程模型会导致别的任务无法并行执行，因此，我们才需要多进程模型或者多线程模型来支持多任务并发执行。

现代操作系统对IO操作已经做了巨大的改进，最大的特点就是支持异步IO。如果充分利用操作系统提供的异步IO支持，就可以用单进程单线程模型来执行多任务，这种全新的模型称为事件驱动模型，Nginx就是支持异步IO的Web服务器，它在单核CPU上采用单进程模型就可以高效地支持多任务。在多核CPU上，可以运行多个进程（数量与CPU核心数相同），充分利用多核CPU。由于系统总的进程数量十分有限，因此操作系统调度非常高效。用异步IO编程模型来实现多任务是一个主要的趋势。

### 分布式进程

在Thread和Process中，应当优选Process，因为Process更稳定，而且，Process可以分布到多台机器上，而Thread最多只能分布到同一台机器的多个CPU上。

Python的`multiprocessing`模块不但支持多进程，其中`managers`子模块还支持把多进程分布到多台机器上。一个服务进程可以作为调度者，将任务分布到其他多个进程中，依靠网络通信。由于`managers`模块封装很好，不必了解网络通信的细节，就可以很容易地编写分布式多进程程序。

举个例子：如果我们已经有一个通过`Queue`通信的多进程程序在同一台机器上运行，现在，由于处理任务的进程任务繁重，希望把发送任务的进程和处理任务的进程分布到两台机器上。怎么用分布式进程实现？

原有的`Queue`可以继续使用，但是，通过`managers`模块把`Queue`通过网络暴露出去，就可以让其他机器的进程访问`Queue`了。

我们先看服务进程，服务进程负责启动`Queue`，把`Queue`注册到网络上，然后往`Queue`里面写入任务：

```python
# task_master.py
import random, queue
from multiprocessing.managers import BaseManager

# 发送任务的队列:
task_queue = queue.Queue()
# 接收结果的队列:
result_queue = queue.Queue()


# 从BaseManager继承的QueueManager:
class QueueManager(BaseManager):
    pass


# 把两个Queue都注册到网络上, callable参数关联了Queue对象:
QueueManager.register('get_task_queue', callable=lambda: task_queue)
QueueManager.register('get_result_queue', callable=lambda: result_queue)
# 绑定端口5000, 设置验证码'abc':
manager = QueueManager(address=('', 5000), authkey=b'abc')
# 启动Queue:
manager.start()
# 获得通过网络访问的Queue对象:
task = manager.get_task_queue()
result = manager.get_result_queue()
# 放几个任务进去:
for i in range(10):
    n = random.randint(0, 10000)
    print('Put task %d...' % n)
    task.put(n)
# 从result队列读取结果:
print('Try get results...')
for i in range(10):
    r = result.get(timeout=10)
    print('Result: %s' % r)
# 关闭:
manager.shutdown()
print('master exit.')
```

请注意，当我们在一台机器上写多进程程序时，创建的`Queue`可以直接拿来用，但是，在分布式多进程环境下，添加任务到`Queue`不可以直接对原始的`task_queue`进行操作，那样就绕过了`QueueManager`的封装，必须通过`manager.get_task_queue()`获得的`Queue`接口添加。

然后，在另一台机器上启动任务进程（本机上启动也可以）：

```python
# task_worker.py
import random, queue
from multiprocessing.managers import BaseManager


# 创建类似的QueueManager:
class QueueManager(BaseManager):
    pass


# 由于这个QueueManager只从网络上获取Queue，所以注册时只提供名字:
QueueManager.register('get_task_queue')
QueueManager.register('get_result_queue')

# 连接到服务器，也就是运行task_master.py的机器:
server_addr = '127.0.0.1'
print('Connect to server %s...' % server_addr)
# 端口和验证码注意保持与task_master.py设置的完全一致:
m = QueueManager(address=(server_addr, 5000), authkey=b'abc')
# 从网络连接:
m.connect()
# 获取Queue的对象:
task = m.get_task_queue()
result = m.get_result_queue()
# 从task队列取任务,并把结果写入result队列:
for i in range(10):
    try:
        n = task.get(timeout=1)
        print('run task %d * %d...' % (n, n))
        r = '%d * %d = %d' % (n, n, n * n)
        time.sleep(1)
        result.put(r)
    except Queue.Empty:
        print('task queue is empty.')
# 处理结束:
print('worker exit.')
```

任务进程要通过网络连接到服务进程，所以要指定服务进程的IP。

现在，可以试试分布式进程的工作效果了。先启动`task_master.py`服务进程：

```plain
$ python3 task_master.py
Put task 3411...
Put task 1605...
Put task 1398...
Put task 4729...
Put task 5300...
Put task 7471...
Put task 68...
Put task 4219...
Put task 339...
Put task 7866...
Try get results...
```

`task_master.py`进程发送完任务后，开始等待`result`队列的结果。现在启动`task_worker.py`进程：

```plain
$ python3 task_worker.py
Connect to server 127.0.0.1...
run task 3411 * 3411...
run task 1605 * 1605...
run task 1398 * 1398...
run task 4729 * 4729...
run task 5300 * 5300...
run task 7471 * 7471...
run task 68 * 68...
run task 4219 * 4219...
run task 339 * 339...
run task 7866 * 7866...
worker exit.
```

`task_worker.py`进程结束，在`task_master.py`进程中会继续打印出结果：

```plain
Result: 3411 * 3411 = 11634921
Result: 1605 * 1605 = 2576025
Result: 1398 * 1398 = 1954404
Result: 4729 * 4729 = 22363441
Result: 5300 * 5300 = 28090000
Result: 7471 * 7471 = 55815841
Result: 68 * 68 = 4624
Result: 4219 * 4219 = 17799961
Result: 339 * 339 = 114921
Result: 7866 * 7866 = 61873956
```

这个示例实现了一个简单的分布式计算系统:

1. task_master.py作为服务进程,创建任务队列和结果队列,并注册到网络上。
2. task_worker.py作为工作进程,连接到服务进程,获取任务并返回结果。
3. 两个进程可以在不同的机器上运行,通过网络进行通信。
4. 使用了multiprocessing.managers模块来实现进程间的通信。

这种方式可以方便地将计算任务分布到多台机器上执行,实现分布式计算。

Queue对象存储在哪？注意到`task_worker.py`中根本没有创建Queue的代码，所以，Queue对象存储在`task_master.py`进程中：
```
                                             │
┌─────────────────────────────────────────┐     ┌──────────────────────────────────────┐
│task_master.py                           │  │  │task_worker.py                        │
│                                         │     │                                      │
│  task = manager.get_task_queue()        │  │  │  task = manager.get_task_queue()     │
│  result = manager.get_result_queue()    │     │  result = manager.get_result_queue() │
│              │                          │  │  │              │                       │
│              │                          │     │              │                       │
│              ▼                          │  │  │              │                       │
│  ┌─────────────────────────────────┐    │     │              │                       │
│  │QueueManager                     │    │  │  │              │                       │
│  │ ┌────────────┐ ┌──────────────┐ │    │     │              │                       │
│  │ │ task_queue │ │ result_queue │ │◀───┼──┼──┼──────────────┘                       │
│  │ └────────────┘ └──────────────┘ │    │     │                                      │
│  └─────────────────────────────────┘    │  │  │                                      │
└─────────────────────────────────────────┘     └──────────────────────────────────────┘
                                             │

                                          Network
```

而`Queue`之所以能通过网络访问，就是通过`QueueManager`实现的。由于`QueueManager`管理的不止一个`Queue`，所以，要给每个`Queue`的网络调用接口起个名字，比如`get_task_queue`。

`authkey`有什么用？这是为了保证两台机器正常通信，不被其他机器恶意干扰。如果`task_worker.py`的`authkey`和`task_master.py`的`authkey`不一致，肯定连接不上。

## **进程池与异步编程**

### **Pool类的使用与优化**

- **使用**：`multiprocessing.Pool`的主要用法是通过`apply()`、`map()`、`starmap()`等方法将任务提交给进程池，然后通过`Pool` 的`close()`和`join()`方法关闭和等待所有进程完成。
- **优化**：为了提高效率，可以考虑以下几点：
  - 适当设置进程数：根据机器的核数和任务的特性，设置合适的进程数，避免过多的进程导致上下文切换开销。
  - 避免频繁的进程间通信：尽量减少进程间的通信，例如，如果任务可以并行处理，尽量一次性提交大量任务。

下面提供一个优化的multiprocessing.Pool使用示例,展示如何高效地使用进程池处理大量任务:

```python
import multiprocessing
import os
import time


def cpu_bound_task(n):
    """模拟一个CPU密集型任务"""
    count = 0
    for i in range(n):
        count += i * i
    return count


def process_chunk(chunk):
    """处理一个数据块"""
    results = []
    for item in chunk:
        results.append(cpu_bound_task(item))
    return results


def main():
    # 获取CPU核心数
    num_cores = os.cpu_count()
    print(f'本机有 {num_cores} 个CPU核心')

    # 创建大量任务
    num_tasks = 10000
    tasks = list(range(0, num_tasks))

    # 将任务分成多个块
    chunk_size = len(tasks) // num_cores
    chunks = [
        tasks[i:i + chunk_size] for i in range(0, len(tasks), chunk_size)
    ]

    start_time = time.time()

    # 创建进程池
    with multiprocessing.Pool(processes=num_cores) as pool:
        # 使用map将任务块分配给进程池
        results = pool.map(process_chunk, chunks)

    # 合并结果
    final_results = [item for sublist in results for item in sublist]

    end_time = time.time()
    print(f'处理 {num_tasks} 个任务耗时: {end_time - start_time:.2f} 秒')
    print(f'结果数量: {len(final_results)}')


if __name__ == '__main__':
    main()

```

这个示例展示了以下几个优化点:

1. 适当设置进程数: 我们使用os.cpu_count()获取CPU核心数,并以此作为进程池的大小。这样可以充分利用多核CPU的优势,同时避免创建过多进程导致的开销。
2. 减少进程间通信: 我们将大量任务分成几个大块,每个进程处理一个大块的任务,而不是每个任务都单独提交给进程池。这样可以显著减少进程间的通信开销。
3. 使用map()方法: 对于并行处理大量同质任务,map()方法通常是最简单高效的选择。它会自动处理任务的分配和结果的收集。
4. 使用上下文管理器: 我们使用with语句来管理进程池,这确保了进程池在使用完毕后被正确关闭,避免了资源泄露。
5. 批量处理结果: 我们在每个进程中处理一个任务块,并返回该块的所有结果。这比每个任务单独返回结果更高效。

通过这些优化,我们可以高效地处理大量CPU密集型任务,充分利用多核CPU的优势。根据具体的任务特性和数据量,你可能需要调整chunk_size来获得最佳性能。

上面实例的运行结果：

```
本机有 8 个CPU核心
处理 10000 个任务耗时: 1.82 秒
结果数量: 10000
```

### **多进程中的异步I/O处理**

- 在多进程环境中，`multiprocessing`模块本身并不直接支持异步 I/O，因为 I/O 操作通常是阻塞的。然而，可以结合其他库（如`asyncio` 或`concurrent.futures`）来实现异步 I/O。例如，`concurrent.futures`提供了`ThreadPoolExecutor`和`ProcessPoolExecutor` ，它们可以配合`asyncio`的`run_in_executor()`方法实现异步 I/O。
- 使用`concurrent.futures`：

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def async_io_task(i):
    # 异步 I/O 操作，如网络请求或文件读写
    pass
with ThreadPoolExecutor() as executor:
    futures = {executor.submit(async_io_task, i) for i in range(10)}
    for future in as_completed(futures):
        result = future.result()
        # 处理结果
```

这里，`ThreadPoolExecutor`用于管理线程，`as_completed()`用于异步等待所有任务完成。这样，尽管 I/O 操作是异步的，但整个进程池的其他任务仍可以并行执行。

好的,我来为您提供一个使用c oncurrent.futures实现多进程异步I/O的示例:

```python
import asyncio
import concurrent.futures
import time


def io_bound_task(n):
    # 模拟I/O密集型任务
    time.sleep(1)
    return f'Task {n} completed'


async def main():
    # 创建ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor() as executor:
        loop = asyncio.get_event_loop()
        # 创建任务列表
        tasks = [
            loop.run_in_executor(executor, io_bound_task, i) for i in range(10)
        ]
        # 并发执行所有任务
        completed, _ = await asyncio.wait(tasks)
        for task in completed:
            print(task.result())


if __name__ == '__main__':
    start = time.time()
    asyncio.run(main())
    end = time.time()
    print(f'Total time: {end - start:.2f} seconds')

```

这个示例展示了以下几点:

1. 我们定义了一个io_bound_task函数来模拟I/O密集型任务。
2. 在main协程中,我们创建了一个ProcessPoolExecutor。
3. 我们使用loop.run_in_executor()方法将每个任务提交给执行器。这允许我们在单独的进程中异步执行I/O密集型任务。

4. 我们使用asyncio.wait()来并发等待所有任务完成。

5. 最后,我们打印每个任务的结果和总执行时间。

这种方法结合了多进程的优势(利用多核CPU)和异步I/O的优势(在等待I/O操作时不阻塞)。它特别适合I/O密集型任务,因为它允许在等待一个进程的I/O操作时切换到另一个进程。

```python
Task 8 completed
Task 5 completed
Task 2 completed
Task 9 completed
Task 6 completed
Task 3 completed
Task 1 completed
Task 0 completed
Task 7 completed
Task 4 completed
Total time: 2.27 seconds
```





```python
import concurrent.futures
import time

import requests


# 模拟耗时的网络请求任务
def fetch_url(url):
    print(f'开始下载 {url}')
    response = requests.get(url)
    return f'{url}: 状态码 {response.status_code}, 内容长度 {len(response.text)} 字节'


# 要下载的URL列表
urls = [
    'https://www.python.org',
    'https://www.github.com',
    'https://www.stackoverflow.com',
    'https://www.google.com',
    'https://www.bing.com',
]


def run_with_threadpool():
    print('使用线程池执行:')
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_to_url = {executor.submit(fetch_url, url): url for url in urls}
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                data = future.result()
                print(data)
            except Exception as exc:
                print(f'{url} 生成了一个异常: {exc}')


def run_with_processpool():
    print('\n使用进程池执行:')
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        results = executor.map(fetch_url, urls)
        for result in results:
            print(result)


if __name__ == '__main__':
    start_time = time.time()

    run_with_threadpool()
    run_with_processpool()

    end_time = time.time()
    print(f'\n总执行时间: {end_time - start_time:.2f} 秒')

```

好的,我来为您提供一个使用concurrent.futures模块的完整示例,展示如何利用线程池和进程池执行并行任务:

```
使用线程池执行:
开始下载 https://www.python.org
开始下载 https://www.github.com
开始下载 https://www.stackoverflow.com
开始下载 https://www.google.com
https://www.python.org: 状态码 200, 内容长度 50321 字节
开始下载 https://www.bing.com
https://www.github.com: 状态码 200, 内容长度 253693 字节
https://www.google.com: 状态码 429, 内容长度 3067 字节
https://www.stackoverflow.com: 状态码 200, 内容长度 126896 字节
https://www.bing.com: 状态码 200, 内容长度 130703 字节

使用进程池执行:
开始下载 https://www.python.org
开始下载 https://www.github.com
开始下载 https://www.stackoverflow.com
开始下载 https://www.google.com
https://www.python.org: 状态码 200, 内容长度 50321 字节
开始下载 https://www.bing.com
https://www.github.com: 状态码 200, 内容长度 253694 字节
https://www.stackoverflow.com: 状态码 200, 内容长度 126895 字节
https://www.google.com: 状态码 429, 内容长度 3067 字节
https://www.bing.com: 状态码 200, 内容长度 137397 字节

总执行时间: 3.75 秒
```

这个示例展示了以下几个要点:

1. 我们定义了一个fetch_url函数来模拟耗时的网络请求任务。

2. 使用ThreadPoolExecutor执行并行任务:

3. 我们使用executor.submit()方法提交任务。

   - 使用concurrent.futures.as_completed()来获取已完成的任务结果。
   - 使用ProcessPoolExecutor执行并行任务:

4. 我们使用executor.map()方法来并行执行任务。

   - 这种方法更简洁,但对于异常处理不如submit()方法灵活。

   - 我们使用with语句来管理执行器的生命周期,确保资源被正确释放。

5. 最后,我们计算并打印了总执行时间。

这个示例展示了如何使用concurrent.futures模块来实现并行任务执行,既可以使用线程池,也可以使用进程池。线程池适合I/O密集型任务(如网络请求),而进程池适合CPU密集型任务。需要注意的是,这个示例中使用的requests库需要单独安装(pip install requests)。

### 多进程间共享状态

`multiprocessing`库提供了两种方式共享状态：`Shared memory`、`Server process`。

#### Shared memory

`Shared memory`很好理解，是一种高效的进程间通信方式，它允许向操作系统申请一块共享内存区域，然后多个进程可以操作这块共享内存了。Multiprocessing模块中提供了Value和Array类，可以用来创建共享内存。下面是一个简单的示例：

```python
import multiprocessing

def worker1(n):
    """该函数将在进程1中执行"""
    n.value += 1
    print('worker1:', n.value)

def worker2(n):
    """该函数将在进程2中执行"""
    n.value += 1
    print('worker2:', n.value)

if __name__ == '__main__':
    # 创建共享内存
    n = multiprocessing.Value('i', 0)
    # 创建进程1
    p1 = multiprocessing.Process(target=worker1, args=(n,))
    # 创建进程2
    p2 = multiprocessing.Process(target=worker2, args=(n,))
    # 启动进程
    p1.start()
    p2.start()
    # 等待进程结束
    p1.join()
    p2.join()
```

在上面的代码中，首先创建了一个Value对象，用于存储一个整数值。然后创建了两个进程，每个进程都会将共享内存中的值加1，并将其打印出来。最后，等待两个进程结束。

除了Value类之外，multiprocessing模块还提供了Array类，用于创建共享内存数组。下面是一个简单的示例：

```python
from multiprocessing import Process, Value, Array

def f(n, a):
    n.value = 3.1415927
    for i in range(len(a)):
        a[i] = -a[i]

if __name__ == '__main__':
    num = Value('d', 0.0)
    arr = Array('i', range(10))

    p = Process(target=f, args=(num, arr))
    p.start()
    p.join()

    print(num.value)
    print(arr[:])
```
运行结果：

```python
3.1415927
[0, -1, -2, -3, -4, -5, -6, -7, -8, -9]
```

来看一个更加复杂的实现一个包含至少两个进程的多进程共享内存示例。

```python
import time
from multiprocessing import Array, Process, Value


def worker1(shared_value, shared_array):
    print(
        f"工作进程1: 初始共享值 = {shared_value.value}, 共享数组 = {list(shared_array)}"
    )
    shared_value.value += 1
    for i in range(len(shared_array)):
        shared_array[i] += 1
    print(
        f"工作进程1: 修改后共享值 = {shared_value.value}, 共享数组 = {list(shared_array)}"
    )
    time.sleep(1)


def worker2(shared_value, shared_array):
    print(
        f"工作进程2: 初始共享值 = {shared_value.value}, 共享数组 = {list(shared_array)}"
    )
    shared_value.value *= 2
    for i in range(len(shared_array)):
        shared_array[i] *= 2
    print(
        f"工作进程2: 修改后共享值 = {shared_value.value}, 共享数组 = {list(shared_array)}"
    )
    time.sleep(1)


if __name__ == "__main__":
    # 创建共享内存对象
    shared_value = Value("i", 0)  # 'i'表示整数类型
    shared_array = Array("i", [1, 2, 3, 4, 5])  # 'i'表示整数类型

    # 创建两个子进程
    p1 = Process(target=worker1, args=(shared_value, shared_array))
    p2 = Process(target=worker2, args=(shared_value, shared_array))

    # 启动进程
    p1.start()
    p2.start()

    # 主进程等待一段时间
    time.sleep(0.5)

    # 在主进程中打印共享内存的值
    print(f"主进程: 共享值 = {shared_value.value}, 共享数组 = {list(shared_array)}")

    # 等待子进程结束
    p1.join()
    p2.join()

    # 再次打印最终的共享内存值
    print(f"最终结果: 共享值 = {shared_value.value}, 共享数组 = {list(shared_array)}")
```

运行结果：

```python
工作进程1: 初始共享值 = 0, 共享数组 = [1, 2, 3, 4, 5]
工作进程2: 初始共享值 = 0, 共享数组 = [1, 2, 3, 4, 5]
工作进程1: 修改后共享值 = 2, 共享数组 = [4, 6, 8, 10, 12]
工作进程2: 修改后共享值 = 2, 共享数组 = [4, 6, 8, 10, 12]
主进程: 共享值 = 2, 共享数组 = [4, 6, 8, 10, 12]
最终结果: 共享值 = 2, 共享数组 = [4, 6, 8, 10, 12]
```

这个示例创建了三个进程(包括主进程)来共享内存:

1. 我们使用 Value 创建了一个共享的整数值,使用 Array 创建了一个共享的整数数组。
2. worker1 和 worker2 函数是两个子进程将要执行的任务。它们分别对共享值和共享数组进行不同的操作。
3. 在主进程中,我们创建并启动两个子进程,然后等待一小段时间。
4. 主进程打印共享内存的当前值,然后等待子进程结束。
5. 最后,我们再次打印共享内存的最终值

这个示例展示了如何在多个进程之间共享内存,并且所有进程都可以读写这些共享的数据。请注意,由于进程执行的顺序是不确定的,每次运行的结果可能会略有不同。

> 注这里操作共享内存时，操作的是很基础的`Value`和`Array`，这里面存放的是ctype类型的基础数据，因而没法存放python里的正常对象。如果一定要使用这个共享，可以考虑用`pickle`库将python里的正常对象序列化为byte数组，再放进`Value`。使用时再读出来，进行反序列化回来。当然要承担序列化开销及两个进程存放两一份数据的内存开销。


#### Server process

`Server process`有点类似于之前的`Fork Server`，调用`manager = multiprocessing.Manager()`方法会启动一个`Server process`进程，接着调用`manager.list()`或`manager.Queue()`，会在这个进程里创建实际的普通对象，并返回一个`Proxy`对象，这个`Proxy`对象里会维持着对`Server process`进程的连接（默认是Socket连接，也可以使用Pipe连接）。

```python
 # 启动Server process进程
    def Manager(self):
        '''Returns a manager associated with a running server process
        The managers methods such as `Lock()`, `Condition()` and `Queue()`
        can be used to create shared objects.
        '''
        from .managers import SyncManager
        m = SyncManager(ctx=self.get_context())
        m.start()
        return m

    # 注册可通过manager获得的共享对象类型
    SyncManager.register('list', list, ListProxy)
    SyncManager.register('Queue', queue.Queue)
    # 注册可通过manager获得的共享对象类型的实现方法
    @classmethod
    def register(cls, typeid, callable=None, proxytype=None, exposed=None,
                 method_to_typeid=None, create_method=True):
        '''
        Register a typeid with the manager type
        '''
        if '_registry' not in cls.__dict__:
            cls._registry = cls._registry.copy()
        if proxytype is None:
            proxytype = AutoProxy
        exposed = exposed or getattr(proxytype, '_exposed_', None)
        method_to_typeid = method_to_typeid or \
                           getattr(proxytype, '_method_to_typeid_', None)
        if method_to_typeid:
            for key, value in list(method_to_typeid.items()):
                assert type(key) is str, '%r is not a string' % key
                assert type(value) is str, '%r is not a string' % value
        cls._registry[typeid] = (
            callable, exposed, method_to_typeid, proxytype
            )
        if create_method:
            def temp(self, *args, **kwds):
                util.debug('requesting creation of a shared %r object', typeid)
                token, exp = self._create(typeid, *args, **kwds)
                proxy = proxytype(
                    token, self._serializer, manager=self,
                    authkey=self._authkey, exposed=exp
                    )
                conn = self._Client(token.address, authkey=self._authkey)
                dispatch(conn, None, 'decref', (token.id,))
                return proxy # 注意这里返回的是proxy对象
            temp.__name__ = typeid
            setattr(cls, typeid, temp)
```

接着在各进程中对这些proxy对象的操作即会通过上述连接操作到实际的对象。至此终于知道虽然`multiprocessing.Queue()`与`manager.Queue()`都返回`Queue`对象，但其实两者的底层实现逻辑很不一样。`SyncManager`的实现代码在[这里](https://github.com/python/cpython/blob/master/Lib/multiprocessing/managers.py)，仔细看这里有一些实现逻辑很巧妙。

这个示例展示如何使用multiprocessing模块创建和管理多个进程,以及如何在进程间共享数据。

1. 我们定义了两个工作函数worker1和worker2,它们将在不同的进程中运行。
2. 使用multiprocessing.Manager()创建了一个manager对象,用于管理进程间共享的数据。

3. 通过manager创建了共享的列表和字典。
4. 创建并启动了两个进程,每个进程运行一个工作函数。

5. 使用join()方法等待两个进程完成。
6. 最后,在主进程中打印共享数据,展示了两个进程对共享数据的修改。

```python
import multiprocessing
import time


def worker1(shared_list, shared_dict):
    print(f'工作进程1的ID: {multiprocessing.current_process().pid}')
    shared_list.append('来自进程1的数据')
    shared_dict[1] = '进程1的值'
    time.sleep(2)  # 模拟一些工作


def worker2(shared_list, shared_dict):
    print(f'工作进程2的ID: {multiprocessing.current_process().pid}')
    shared_list.extend(['来自进程2的数据1', '来自进程2的数据2'])
    shared_dict[2] = '进程2的值'
    time.sleep(1)  # 模拟一些工作


if __name__ == '__main__':
    # 创建Manager对象来管理共享数据
    with multiprocessing.Manager() as manager:
        shared_list = manager.list()
        shared_dict = manager.dict()

        # 创建两个进程
        p1 = multiprocessing.Process(target=worker1,
                                     args=(shared_list, shared_dict))
        p2 = multiprocessing.Process(target=worker2,
                                     args=(shared_list, shared_dict))

        print(f'主进程ID: {multiprocessing.current_process().pid}')

        # 启动进程
        p1.start()
        p2.start()

        # 等待进程结束
        p1.join()
        p2.join()

        # 打印共享数据
        print('共享列表:', shared_list)
        print('共享字典:', shared_dict)
```

运行结果：

```python
主进程ID: 33464
工作进程1的ID: 33468
工作进程2的ID: 33469
共享列表: ['来自进程1的数据', '来自进程2的数据1', '来自进程2的数据2']
共享字典: {1: '进程1的值', 2: '进程2的值'}
```

### **concurrent.futures模块的使用**

`concurrent.futures`提供了更简洁的接口，它抽象了底层的线程池或进程池，使得异步编程更加方便。`ProcessPoolExecutor` 和`ThreadPoolExecutor`是两个主要的类，它们都支持`submit()`方法提交任务，然后你可以通过`as_completed()`或`result()` 等方法获取结果。与`multiprocessing.Pool`相比，`concurrent.futures`更加面向异步编程，更适合现代 Python 应用。

## **高级并发技巧**

这一章将深入探讨Python中进行多进程同步与协调的高级技巧，以及如何避免全局解释器锁（GIL）的影响，还有资源管理和任务调度。

### **多进程同步与协调（Semaphore, Lock, Event, Condition）**

- **Semaphore（信号量）** ：用于限制可以同时访问某个资源的进程数。在进程间同步对共享资源的访问非常有用。

```python
import multiprocessing
semaphore = multiprocessing.Semaphore(2)  # 允许两个进程同时访问资源
def worker(semaphore):
    semaphore.acquire()
    try:
        # 执行任务
        pass
    finally:
        semaphore.release()
```

- **Lock（互斥锁）** ：用于确保一次只有一个进程可以访问共享资源。

```python
import multiprocessing

lock = multiprocessing.Lock()
def worker(lock):
    lock.acquire()
    try:
        # 执行任务
        pass
    finally:
        lock.release()
```

- **Event（事件）** ：用于在进程间同步操作，一个进程可以设置或等待事件。

```python
import multiprocessing

event = multiprocessing.Event()
def setter(event):
    event.set()  # 设置事件
def waiter(event):
    event.wait()  # 等待事件被设置
```

- **Condition（条件变量）** ：与Lock类似，但允许进程在某些条件下等待或通知其他进程。

```python
import multiprocessing

condition = multiprocessing.Condition()
def worker_with_condition(condition):
    with condition:
        condition.wait()  # 等待通知
        # 执行任务
```

#### Lock （互斥锁）

多线程和多进程最大的不同在于，多进程中，同一个变量，各自有一份拷贝存在于每个进程中，互不影响，而多线程中，所有变量都由所有线程共享，所以，任何一个变量都可以被任何一个线程修改，因此，线程之间共享数据最大的危险在于多个线程同时改一个变量，把内容给改乱了。

来看看多个线程同时操作一个变量怎么把内容给改乱了：

```python
# multithread
import time, threading

# 假定这是你的银行存款:
balance = 0

def change_it(n):
    # 先存后取，结果应该为0:
    global balance
    balance = balance + n
    balance = balance - n

def run_thread(n):
    for i in range(10000000):
        change_it(n)

t1 = threading.Thread(target=run_thread, args=(5,))
t2 = threading.Thread(target=run_thread, args=(8,))
t1.start()
t2.start()
t1.join()
t2.join()
print(balance)
```

我们定义了一个共享变量`balance`，初始值为`0`，并且启动两个线程，先存后取，理论上结果应该为`0`，但是，由于线程的调度是由操作系统决定的，当`t1`、`t2`交替执行时，只要循环次数足够多，`balance`的结果就不一定是`0`了。

原因是因为高级语言的一条语句在CPU执行时是若干条语句，即使一个简单的计算：

```python
balance = balance + n
```

也分两步：

1. 计算`balance + n`，存入临时变量中；
2. 将临时变量的值赋给`balance`。

也就是可以看成：

```python
x = balance + n
balance = x
```

由于x是局部变量，两个线程各自都有自己的x，当代码正常执行时：

```plain
初始值 balance = 0

t1: x1 = balance + 5 # x1 = 0 + 5 = 5
t1: balance = x1     # balance = 5
t1: x1 = balance - 5 # x1 = 5 - 5 = 0
t1: balance = x1     # balance = 0

t2: x2 = balance + 8 # x2 = 0 + 8 = 8
t2: balance = x2     # balance = 8
t2: x2 = balance - 8 # x2 = 8 - 8 = 0
t2: balance = x2     # balance = 0

结果 balance = 0
```

但是t1和t2是交替运行的，如果操作系统以下面的顺序执行t1、t2：

```plain
初始值 balance = 0

t1: x1 = balance + 5  # x1 = 0 + 5 = 5

t2: x2 = balance + 8  # x2 = 0 + 8 = 8
t2: balance = x2      # balance = 8

t1: balance = x1      # balance = 5
t1: x1 = balance - 5  # x1 = 5 - 5 = 0
t1: balance = x1      # balance = 0

t2: x2 = balance - 8  # x2 = 0 - 8 = -8
t2: balance = x2      # balance = -8

结果 balance = -8
```

究其原因，是因为修改`balance`需要多条语句，而执行这几条语句时，线程可能中断，从而导致多个线程把同一个对象的内容改乱了。

两个线程同时一存一取，就可能导致余额不对，你肯定不希望你的银行存款莫名其妙地变成了负数，所以，我们必须确保一个线程在修改`balance`的时候，别的线程一定不能改。

如果我们要确保`balance`计算正确，就要给`change_it()`上一把锁，当某个线程开始执行`change_it()`时，我们说，该线程因为获得了锁，因此其他线程不能同时执行`change_it()`，只能等待，直到锁被释放后，获得该锁以后才能改。由于锁只有一个，无论多少线程，同一时刻最多只有一个线程持有该锁，所以，不会造成修改的冲突。创建一个锁就是通过`threading.Lock()`来实现：

```python
balance = 0
lock = threading.Lock()

def run_thread(n):
    for i in range(100000):
        # 先要获取锁:
        lock.acquire()
        try:
            # 放心地改吧:
            change_it(n)
        finally:
            # 改完了一定要释放锁:
            lock.release()
```

当多个线程同时执行`lock.acquire()`时，只有一个线程能成功地获取锁，然后继续执行代码，其他线程就继续等待直到获得锁为止。

获得锁的线程用完后一定要释放锁，否则那些苦苦等待锁的线程将永远等待下去，成为死线程。所以我们用`try...finally`来确保锁一定会被释放。

锁的好处就是确保了某段关键代码只能由一个线程从头到尾完整地执行，坏处当然也很多，首先是阻止了多线程并发执行，包含锁的某段代码实际上只能以单线程模式执行，效率就大大地下降了。其次，由于可以存在多个锁，不同的线程持有不同的锁，并试图获取对方持有的锁时，可能会造成死锁，导致多个线程全部挂起，既不能执行，也无法结束，只能靠操作系统强制终止。

#### ThreadLocal

在多线程环境下，每个线程都有自己的数据。一个线程使用自己的局部变量比使用全局变量好，因为局部变量只有线程自己能看见，不会影响其他线程，而全局变量的修改必须加锁。

但是局部变量也有问题，就是在函数调用的时候，传递起来很麻烦：

```python
def process_student(name):
    std = Student(name)
    # std是局部变量，但是每个函数都要用它，因此必须传进去：
    do_task_1(std)
    do_task_2(std)

def do_task_1(std):
    do_subtask_1(std)
    do_subtask_2(std)

def do_task_2(std):
    do_subtask_2(std)
    do_subtask_2(std)
```

每个函数一层一层调用都这么传参数那还得了？用全局变量？也不行，因为每个线程处理不同的`Student`对象，不能共享。

如果用一个全局`dict`存放所有的`Student`对象，然后以`thread`自身作为`key`获得线程对应的`Student`对象如何？

```python
global_dict = {}

def std_thread(name):
    std = Student(name)
    # 把std放到全局变量global_dict中：
    global_dict[threading.current_thread()] = std
    do_task_1()
    do_task_2()

def do_task_1():
    # 不传入std，而是根据当前线程查找：
    std = global_dict[threading.current_thread()]
    ...

def do_task_2():
    # 任何函数都可以查找出当前线程的std变量：
    std = global_dict[threading.current_thread()]
    ...
```

这种方式理论上是可行的，它最大的优点是消除了`std`对象在每层函数中的传递问题，但是，每个函数获取`std`的代码有点丑。有没有更简单的方式？

`ThreadLocal`应运而生，不用查找`dict`，`ThreadLocal`帮你自动做这件事：

```python
import threading

# 创建全局ThreadLocal对象:
local_school = threading.local()

def process_student():
    # 获取当前线程关联的student:
    std = local_school.student
    print('Hello, %s (in %s)' % (std, threading.current_thread().name))

def process_thread(name):
    # 绑定ThreadLocal的student:
    local_school.student = name
    process_student()

t1 = threading.Thread(target= process_thread, args=('Alice',), name='Thread-A')
t2 = threading.Thread(target= process_thread, args=('Bob',), name='Thread-B')
t1.start()
t2.start()
t1.join()
t2.join()
```

执行结果：

```plain
Hello, Alice (in Thread-A)
Hello, Bob (in Thread-B)
```

全局变量`local_school`就是一个`ThreadLocal`对象，每个`Thread`对它都可以读写`student`属性，但互不影响。你可以把`local_school`看成全局变量，但每个属性如`local_school.student`都是线程的局部变量，可以任意读写而互不干扰，也不用管理锁的问题，`ThreadLocal`内部会处理。

可以理解为全局变量`local_school`是一个`dict`，不但可以用`local_school.student`，还可以绑定其他变量，如`local_school.teacher`等等。

`ThreadLocal`最常用的地方就是为每个线程绑定一个数据库连接，HTTP请求，用户身份信息等，这样一个线程的所有调用到的处理函数都可以非常方便地访问这些资源。

### **避免全局解释器锁（GIL）的影响**

GIL是CPython中的一个机制，它确保同一时间只有一个线程在执行Python字节码。为了绕过GIL，可以使用以下方法：

- 使用多进程而不是多线程，因为每个Python进程都有自己的GIL。
- 使用Jython或IronPython，这些Python实现没有GIL。
- 使用C扩展来执行计算密集型任务，这些扩展可以在没有GIL的情况下运行。

### **资源管理和任务调度**

- **资源管理**：使用上下文管理器（如`with`语句）确保资源如文件和网络连接被正确关闭。对于进程和线程，确保使用`Pool` 和`Executor`的上下文管理器来关闭和等待所有任务完成。
- **任务调度**：可以使用队列（如`multiprocessing.Queue`）来调度任务，其中生产者进程将任务放入队列，消费者进程从队列中取出任务并执行。

```python
import multiprocessing

def producer(queue):
    # 生产任务
    queue.put(task)
def consumer(queue):
    while True:
        task = queue.get()
        # 处理任务
        queue.task_done()

queue = multiprocessing.Queue()
producer_process = multiprocessing.Process(target=producer, args=(queue,))
consumer_process = multiprocessing.Process(target=consumer, args=(queue,))

producer_process.start()
consumer_process.start()

producer_process.join()
queue.join()  # 等待队列中的所有任务被处理
```

通过这些高级技巧，你可以更有效地管理并发任务，提高应用程序的性能和稳定性。

```python
from multiprocessing import Pool
import os
import time

def worker(x):
    print(f"进程 {os.getpid()} 正在处理 {x}")
    time.sleep(1)  # 模拟一些耗时操作
    return x * x

if __name__ == "__main__":
    print(f"主进程ID: {os.getpid()}")

    # 使用with语句来管理进程池
    with Pool(processes=4) as pool:
        # 使用map方法并行处理数据
        results = pool.map(worker, range(10))

        # 打印结果
        print("处理结果:", results)

    print("所有工作已完成,进程池已关闭")

    # 在这里,进程池已经自动关闭,所有工作进程都已终止
```

这个示例展示了以下几点:

1. 我们定义了一个worker函数,它将在每个工作进程中执行。
2. 使用with Pool(processes=4) as pool:创建了一个包含4个进程的进程池。with语句确保在代码块结束时,进程池会被正确关闭。
3. 我们使用pool.map()方法来并行处理数据。这个方法会自动将任务分配给池中的可用进程。
4. 在with语句块结束时,进程池会自动关闭,所有工作进程都会被终止。我们不需要显式地调用close()或join()方法。

这种方法有以下优点:

- 资源管理更加简洁和安全。即使在处理过程中发生异常,进程池也会被正确关闭。
- 代码更加简洁易读。
- 避免了忘记关闭进程池的风险。

运行这段代码,你会看到类似以下的输出:

```python
主进程ID: 12345

进程 12346 正在处理 0

进程 12347 正在处理 1

进程 12348 正在处理 2

进程 12349 正在处理 3

进程 12346 正在处理 4

进程 12347 正在处理 5

进程 12348 正在处理 6

进程 12349 正在处理 7

进程 12346 正在处理 8

进程 12347 正在处理 9

处理结果: [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

所有工作已完成,进程池已关闭
```

这个例子展示了如何使用Python的multiprocessing模块和上下文管理器来安全地管理多进程资源。这种方法不仅适用于进程池,也可以应用于其他需要正确关闭和清理的资源,如文件、数据库连接等。

下面提供一个使用multiprocessing.Queue进行任务调度的示例。这个例子将展示如何使用一个生产者进程生成任务,并使用多个消费者进程从队列中获取任务并执行。

```python
import multiprocessing
import time
import random


def producer(queue, num_tasks):
    print(f"生产者进程 {multiprocessing.current_process().name} 开始运行")
    for i in range(num_tasks):
        task = f"任务-{i}"
        queue.put(task)
        print(f"生产者添加任务: {task}")
        time.sleep(random.uniform(0.1, 0.3))  # 模拟任务生成时间

    # 添加结束标志
    for _ in range(multiprocessing.cpu_count()):
        queue.put(None)
    print("生产者完成所有任务")


def consumer(queue):
    print(f"消费者进程 {multiprocessing.current_process().name} 开始运行")
    while True:
        task = queue.get()
        if task is None:
            break
        print(f"消费者 {multiprocessing.current_process().name} 执行任务: {task}")
        time.sleep(random.uniform(0.5, 1))  # 模拟任务执行时间
    print(f"消费者 {multiprocessing.current_process().name} 完成工作")


if __name__ == "__main__":
    num_tasks = 10
    task_queue = multiprocessing.Queue()

    # 创建生产者进程
    producer_process = multiprocessing.Process(
        target=producer, args=(task_queue, num_tasks)
    )

    # 创建消费者进程
    num_consumers = multiprocessing.cpu_count()
    consumer_processes = [
        multiprocessing.Process(target=consumer, args=(task_queue,))
        for _ in range(num_consumers)
    ]

    # 启动所有进程
    producer_process.start()
    for p in consumer_processes:
        p.start()

    # 等待所有进程完成
    producer_process.join()
    for p in consumer_processes:
        p.join()

    print("所有进程已完成")
```

这个示例展示了以下几点:

1. 我们定义了一个producer函数,它生成任务并将其放入队列中。
2. 我们定义了一个consumer函数,它从队列中获取任务并执行。
3. 我们使用multiprocessing.Queue()创建了一个共享的任务队列。
4. 我们创建了一个生产者进程和多个消费者进程(数量等于CPU核心数)。
5. 生产者进程生成指定数量的任务,并在完成后为每个消费者进程添加一个结束标志(None)。

6. 消费者进程不断从队列中获取任务并执行,直到遇到结束标志。
7. 主进程等待所有子进程完成后才结束。

这种方法有以下优点:=

- 实现了任务的动态分配,消费者进程可以根据自己的处理速度从队列中获取任务。
- 通过使用多个消费者进程,可以充分利用多核CPU的优势。
- 生产者和消费者之间解耦,可以独立地调整生产和消费的速度。

运行这段代码,你会看到任务被生产者添加到队列中,然后被多个消费者并行处理。这种模式非常适合处理大量独立的任务,如数据处理、网络请求等。

```python
消费者进程 Process-2 开始运行
生产者进程 Process-1 开始运行
生产者添加任务: 任务-0
消费者 Process-2 执行任务: 任务-0
消费者进程 Process-4 开始运行
消费者进程 Process-3 开始运行
消费者进程 Process-5 开始运行
消费者进程 Process-6 开始运行
消费者进程 Process-7 开始运行
消费者进程 Process-8 开始运行
消费者进程 Process-9 开始运行
生产者添加任务: 任务-1
消费者 Process-4 执行任务: 任务-1
生产者添加任务: 任务-2
消费者 Process-3 执行任务: 任务-2
生产者添加任务: 任务-3
消费者 Process-5 执行任务: 任务-3
生产者添加任务: 任务-4
消费者 Process-6 执行任务: 任务-4
生产者添加任务: 任务-5
消费者 Process-7 执行任务: 任务-5
生产者添加任务: 任务-6
消费者 Process-8 执行任务: 任务-6
生产者添加任务: 任务-7
消费者 Process-9 执行任务: 任务-7
生产者添加任务: 任务-8
消费者 Process-3 执行任务: 任务-8
生产者添加任务: 任务-9
消费者 Process-2 执行任务: 任务-9
生产者完成所有任务
消费者 Process-5 完成工作
消费者 Process-4 完成工作
消费者 Process-6 完成工作
消费者 Process-7 完成工作
消费者 Process-8 完成工作
消费者 Process-9 完成工作
消费者 Process-3 完成工作
消费者 Process-2 完成工作
所有进程已完成
```

## **进程间的错误处理与调试**

在这一章中，我们将讨论进程间的错误处理与调试，包括错误处理策略、使用logging和traceback 进行错误处理，以及调试工具与技术。

### **错误处理策略**

在多进程编程中，错误处理非常重要，因为一个进程的错误可能会影响其他进程甚至整个应用程序。以下是一些错误处理策略：

- **进程间通信异常处理**：在进程间通信时，要捕获并处理异常，以避免进程崩溃。可以在进程间通信的代码块中使用try-except语句来捕获异常。
- **进程池异常处理**：如果使用进程池（如`multiprocessing.Pool`），要注意捕获并处理子进程中抛出的异常，以避免整个进程池被终止。
- **日志记录**：及时记录错误和异常信息到日志文件中，以便后续排查问题。

下面提供一个在进程间通信时处理异常的示例。这个例子将展示如何在多个进程之间安全地传递数据,并处理可能出现的异常。

```python
import multiprocessing
import traceback

class SafeProcess(multiprocessing.Process):
    def __init__(self, *args, **kwargs):
        multiprocessing.Process.__init__(self, *args, **kwargs)
        self._pconn, self._cconn = multiprocessing.Pipe()
        self._exception = None

    def run(self):
        try:
            multiprocessing.Process.run(self)
            self._cconn.send(None)
        except Exception as e:
            tb = traceback.format_exc()
            self._cconn.send((e, tb))

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception

def worker(queue):
    try:
        # 模拟一些可能抛出异常的操作
        queue.put("正常数据")
        raise ValueError("模拟的错误")
    except Exception as e:
        queue.put(("异常", str(e)))

def main():
    queue = multiprocessing.Queue()
    process = SafeProcess(target=worker, args=(queue,))

    process.start()

    while process.is_alive():
        try:
            data = queue.get(timeout=1)
            if isinstance(data, tuple) and data[0] == "异常":
                print(f"工作进程发生异常: {data[1]}")
            else:
                print(f"收到数据: {data}")
        except multiprocessing.queues.Empty:
            pass

    process.join()

    if process.exception:
        error, tb = process.exception
        print(f"进程异常: {error}")
        print(f"异常追踪:\n{tb}")

if __name__ == "__main__":
    main()
```

这个示例展示了以下几点:

1. 我们定义了一个SafeProcess类,它继承自multiprocessing.Process,并添加了异常处理功能。
2. worker函数模拟了一些可能抛出异常的操作。它首先向队列中放入正常数据,然后抛出一个异常。
3. 在main函数中,我们创建了一个SafeProcess实例和一个共享队列。
4. 主进程不断尝试从队列中获取数据,并处理可能出现的Empty异常。
5. 如果从队列中收到的是异常信息,我们会打印出来。
6. 进程结束后,我们检查SafeProcess实例是否捕获到了任何异常,如果有,则打印异常信息和追踪栈。

这种方法有以下优点:

- 可以捕获和处理子进程中的异常,而不会导致整个程序崩溃。
- 通过队列,我们可以在进程间安全地传递正常数据和异常信息。
- 主进程可以及时得知子进程的异常情况,并做出相应处理。
- 运行这段代码,你会看到类似以下的输出:

```python
收到数据: 正常数据

工作进程发生异常: 模拟的错误

进程异常: ValueError('模拟的错误')

异常追踪:

Traceback (most recent call last):

 File "/path/to/your/script.py", line 18, in worker

  raise ValueError("模拟的错误")

ValueError: 模拟的错误
```

这个例子展示了如何在Python的多进程编程中安全地处理异常,确保即使子进程出现问题,主进程也能得到通知并继续运行。

### **使用logging和traceback**

- **logging模块**：Python的logging模块提供了灵活且强大的日志记录功能，可以用于记录程序运行时的信息、警告和错误。在多进程环境中，可以使用logging模块将日志信息写入文件或控制台，以便进行错误排查。

```python
import logging

logging.basicConfig(filename='example.log', level=logging.DEBUG)
logging.debug('This is a debug message')
logging.error('This is an error message')
```

- **traceback模块**：Python的traceback模块可以用于获取异常的堆栈信息，帮助定位错误发生的位置。

```python
import traceback
try:
    # 可能会引发异常的代码
    pass
except Exception as e:
    traceback.print_exc()
```
