# 适配方案简介

昇腾是当前国产芯片中唯一能够进行大规模分布式训练集群部署的厂商，其适配了主流的深度学习训练框架，包括自研的Mindspore和适配的TensorFlow和Pytorch。其中PyTorch因其易用性、灵活性和强大的社区支持而受到许多深度学习研究者和开发者的喜爱。昇腾芯片对Pytorch框架的适配能够快速帮助它共享和扩大Pytorch强大的生态，对其未来的发展至关重要。对Pytorch适配有两个主要思路：
1. 将Pytorch的接口转接到Mindspore对应的接口上；实际上就是将Pytorch的前端代码的计算逻辑映射到Mindspore框架上去，然后由Mindspore去执行；这种方案的问题在于Pytorch和Mindspore的内在逻辑无法相通，比如Pytorch的动态图和Mindspore的静态图直接就可能存在直接矛盾；举一个例子，Mindspore虽然同时支持动态图和静态图，但是一份相同的代码在动态图下可以正常运行，但是在静态图下就会报错。
2. 利用插件来对原生Pytorch进行适配，针对原生扩展逻辑进行逐一适配，API基本跟Pytorch一致，譬如除开昇腾芯片不支持的fp64数据格式，其它基本都能满足。简单来理解，就是从算子层面对Pytorch的计算后端进行替换，采用昇腾开发的算子来替换GPU算子。这种方案能够充分的利用原生Pytorch框架的优势，也是是Pytorch适配的目标。

## 插件适配方案

早期的Ascend Pytorch采用侵入式适配，即直接修改原生Pytorch代码，带来的问题非常明显，比如：

1. 质量难以控制，且项目测试工程量巨大；
2. 版本升级困难，每个新版版都需要重新适配；

采用插件化开发的方式，上面的问题就能够得到解决。下面简单的介绍Pytorch Adapter的实现原理：

### Pytorch仓的结构

pytorch仓的结构：

+ torch/        # 存放大家熟悉的python模块
  + csrc        # 用C++实现的pytorch前端的功能，包括python和C++的binding，autograd和JIT
+ aten          # "A Tensor Library"的缩写
  + src/ATen    # 用C++实现的tensor基本操作
+ c10           # Caffe2和 A Tensor的双关语，包含pytorch的核心抽象，以及tensor和存储数据结构的实际实现

### 核心机制-dispatch

简单而言，就是根据API调用时，输入的数据类型来决定后端调用的API类型。比如CPU和GPU的API是不一样的，可以自动根据传入的tensor类型来自动选择API。

具体来看，对于每一个前端的算子，dispatcher会维护一个函数指针表，为每个dispatch key提供对应的视线。这个表中有针对不同后端（CPU，GPU，XLA）的dispatch条目，也有想autograd和tracing这样的高抽象层级概念的条目。dispatcher根据输入的tensor和其他东西计算出一个dispatch key，然后跳转到函数指针表所指向的函数。

所以，对于昇腾处理器而言，实现Adapter主要就是要讲用昇腾实现的算子注册到dispatcher上面，即可复用pytorch的dispatch机制完成算子分布。

### 示例-单算子调用流程

1. 用户在前端调用算子，比如可调用nn.Module,nn.Funtional,Tensor对象上的函数；
2. pybind11根据注册绑定的映射规则，调用后端C++方法；
3. 后端C++接口根据输入参数来选取所需调用的算子类型（dispatch机制），比如是调用CPU实现的算子还是GPU实现的算子（注意：此处只要注册NPU实现的算子，便可调用昇腾处理器的计算能力；
4. 调用相应算子，返回结果；

### PyTorch Adapter的逻辑架构图

<img src="./pics/torch_adapter_framework.png" style="width:60%;">

在线适配方案：模型执行，训练等主要功能流程有Pytorch框架提供，用户界面API保持不变，将Davinci设备和计算库作为扩展资源注册到PyTorch框架中。

+ 优点：
  + 继承PyTorch动态图特性
  + 继承原生PyTorch使用方式，移植的时候，在开发方式和代码复用方便做到最小的改动；
  + 继承Pytorch的原生体系结构，保留框架本身出色的特性，比如自动微分，动态分发，Debug，Profiling，Storage共享机制等；
  + 扩展性：对于新增网络类型或结构，只需增加涉及的算子开发和实现。框架类算子，反向图建立和实现机制等结构可保持复用；

## Pytorch模型迁移


Pytorch原生模型迁移到昇腾硬件分为以下几步：

1. 准备Ascend硬件环境
2. 算子满足度分析
3. 模型迁移
   1. 自动迁移
   2. 手动迁移
4. 模型调优

### 昇腾上Ascend_Pytorch安装指南

[Ascend_Pytorch安装指南](./从头安装AscendPytorch.md)

### 算子满足度分析

方式一：利用torch的profiler来提取模型所用到的算子，然后查看NPU对这些算子的支持情况；
  + [NPU算子支持清单](https://www.hiascend.com/document/detail/zh/canncommercial/63RC2/oplist/fwoperator/fwoperatorlist_0316.html)


方式二：直接调用CANN分析迁移工具

    ```bash
    python ms_fmk_transplt.py -i model_file.py -o model_file_out.py
    ```
其中ms_fmk_transplt.py已经包含在CANN安装包中了，默认路径为：`/usr/local/Ascend/ascend-toolkit/latest/tools/ms_fmk_transplt/ms_fmk_transplt.py`。另外，该脚步还会输出迁移报告，里面有详细的不支持算子列表。

### 模型迁移

主要分：
+ 手工迁移：用户自行修改pytorch训练脚步
+ 自动迁移：
  + 训练前：通过迁移脚步转换工具，自动将脚本由GPU转换为NPU版本，同时也会生成迁移报告；
  + 训练时：在训练脚步中导入脚本转换库，运行训练时，自动将Pytorch训练脚本的cuda接口进行替换，操作简洁。

#### 手工迁移

主要修改点：

1. 导入NPU相关库

  ```python
  if torch.__version__ > 1.5:
    import torch_npu
  ```

2. 指定device类型为npu

   ```python
   device = torch.device("cuda:0")  => device = torch.device("npu:0")
   torch.cuda.set_device(device)    => torch.npu.set_device(device)
   tensor.to('cuda:0')              => tensor.to('npu:0')
   ```

3. 将训练脚步中的cuda接口替换为npu接口

   ```python
   torch.cuda.xxx()         =>  torch.npu.xxx()
   torch.cuda.set_device(0) => torch.npu.set_device(0)
   torch.cuda.synchronize() => torch.npu.synchronize()
   ```

4. 分布式代码迁移:将`nccl`改为`hccl`

  ```python
  dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:29688', world_size=8, rank=rank)
  # 替换为：
  dist.init_process_group(backend='hccl', init_method='tcp://127.0.0.1:29688', world_size=8, rank=rank)
  ```

  注：Ascend pytorch只支持DDP，不支持DP。

#### 自动迁移

1. 使用转换脚本`ms_fmk_transplt.py`

  ```bash
  python ms_fmk_transplt.py -i model_file.py -o model_file_out.py
  ```

2. 导入脚步转换库(支持PT1.8以上版本，推荐这种方式)

   ```python
   from torch_npu.contrib import transfer_to_npu
   ```

3. 关于混合精度

  pytorch默认是fp32精度，而昇腾芯片的加速单元主要是fp16的算力，所以需要采用混合精度训练的方式。当前Ascend-Torch支持torch.cuda.amp和apex.amp两种方式的混合精度训练。

  ```python
  # 方式一：torch.cuda.amp
  use_amp = True
  net = make_model(in_size, out_size, num_layers)
  opt = torch.optim.SGD(net.parameters(), lr=0.001)
  scaler = torch.cuda.amp.GradScaler(anabled_use_amp)

  for epoch in range(epochs):
    for input, target in zip(data, targets):
      with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
        output = net(input)
        loss = loss_fn(output, target)
      scaler.scale(loss).backward()
      scaler.step(opt)
      scaler.update()
      opt.zero_grad()

  # 方式二：apex.amp
  model = torch.nn.Linear(D_in, D_out).cuda()
  optimzier = torch.optim.SGD(model.parameters(), lr=1e-3)

  model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
  ...
  with amp.scale_loss(loss, optimizer) as scaled_loss:
    scaled_loss.backward()
  ...
  ```

### 其它

1. 指定用哪张NPU卡

   ```bash
   export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
   ```
