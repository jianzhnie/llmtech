# 从0开始安装Ascend Pytorch

硬件配置：arm（host processor）+ 8*NPU（accelerator）
系统：Ubuntu 18.04
| 文档         | 内容介绍                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | 关键资源获取                                                                                                                                                                                                                                                                                                                                                                                                              |
| ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 产品概述     | 快速了解Ascend Extension for PyTorch的架构及关键功能特性。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | 《[产品概述](https://www.hiascend.com/document/detail/zh/Pytorch/60RC2/quickstart/productoverview/ptoverview_0001.html)》                                                                                                                                                                                                                                                                                                 |
| 配置与安装   | 详细介绍不同昇腾硬件产品形态的安装方案、环境搭建流程，以及卸载、其他常用操作等。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | 《[Ascend Extension for PyTorch 配置与安装](https://www.hiascend.com/document/detail/zh/Pytorch/60RC2/configandinstg/instg/insg_0001.html)》                                                                                                                                                                                                                                                                              |
| 快速体验     | 以一个简单的模型迁移样例，采用了最简单的自动迁移方法，帮助用户快速体验其他平台的模型脚本迁移到昇腾NPU上的流程，并通过修改脚本代码，使其可以迁移到昇腾NPU上进行训练。                                                                                                                                                                                                                                                                                                                                                                                                                                                               | 《[快速体验](https://www.hiascend.com/document/detail/zh/Pytorch/60RC2/quickstart/fastexperience/fastexperience_0001.html)》                                                                                                                                                                                                                                                                                              |
| 模型开发     | 包含了大模型的迁移及调优、精度问题定位、性能问题解决等指导，并提供了常见模型案例库。训练模型迁移调优主要目标是引导具有一定PyTorch模型训练基础的读者，了解如何将原本在其他平台（例如GPU）上训练的模型迁移到昇腾平台（NPU）。主要介绍多种类、全流程的迁移过程：无论是针对小规模模型还是大型模型，都提供了详尽的指导。重点关注如何有效地将PyTorch训练模型进行迁移，在昇腾平台上进行训练功能。精度调试介绍了大模型训练过程中常见的问题及其调试方法，帮助用户将问题消除在训练开始之前，以及缩短模型精度问题定位的时间。性能调优主要介绍NPU性能的拆解和分析以及优化方法。模型案例介绍了多模态模型、LLM大模型以及自动驾驶模型的迁移案例。 | 《[PyTorch 训练模型迁移调优指南](https://www.hiascend.com/document/detail/zh/Pytorch/60RC2/ptmoddevg/trainingmigrguide/PT_LMTMOG_0002.html)》                                                                                                                                                                                                                                                                             |
| 套件与三方库 | 包含了套件与三方库支持清单，提供了基于TorchAir的PyTorch图模式使用和MindSpeed迁移开发的指导。套件与三方库支持清单主要介绍昇腾设备支持的模型套件列表、第三方库原生支持和适配清单。TorchAir主要介绍将PyTorch的FX图转换为GE计算图，并提供了GE计算图的编译与执行接口。MindSpeedMindSpeed是针对华为昇腾设备的大模型加速库。昇腾提供该加速库，使能客户大模型业务快速迁移至昇腾设备，并且支持昇腾专有算法，确保开箱可用。                                                                                                                                                                                                                  | 《[套件与三方库支持清单](https://www.hiascend.com/document/detail/zh/Pytorch/60RC2/modthirdparty/modparts/thirdpart_0002.html)》《[PyTorch 图模式使用指南（TorchAir）](https://www.hiascend.com/document/detail/zh/Pytorch/60RC2/modthirdparty/torchairuseguide/torchair_0001.html)》《[MindSpeed 迁移开发指南](https://www.hiascend.com/document/detail/zh/Pytorch/60RC2/modthirdparty/asdevguide/mindspeed_0001.html)》 |
| API参考      | 主要介绍原生PyTorch框架2.3.1/2.2.0/2.1.0/1.11.0的API接口支持情况与限制说明以及Ascend Extension for PyTorch自定义API的接口说明、功能描述、参数说明与使用示例。                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | 《[API 参考](https://www.hiascend.com/document/detail/zh/Pytorch/60RC2/apiref/apilist/ptaoplist_000001.html)》                                                                                                                                                                                                                                                                                                            |
| 环境变量     | 包含了开发者基于CANN及Ascend Extension for PyTorch构建AI应用和训练执行中可使用的环境变量。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | 《CANN 环境变量参考》中的“[Ascend Extension for PyTorch](https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/apiref/envvar/envref_07_0048.html)”章节                                                                                                                                                                                                                                                         |
+ useful links:
    + [Ascend Pytorch官方仓库](https://gitee.com/ascend/pytorch)
    + [CANN社区版下载地址](https://www.hiascend.com/software/cann/community)
    + [固件驱动下载地址](https://www.hiascend.com/zh/hardware/firmware-drivers/community)
    + [Ascend Apex官方仓库](https://gitee.com/ascend/apex)

## 安装驱动和固件

这里分步骤记录：

1. 安装动态内核模块支持框架dkms：
   ```bash
   sudo apt install dkms
   ```

2. 我演示的是安装最新的torch-2.1.0版本(顺便提一下，现在Ascend Pytorch版本好像跟官方的Pytorch版本进行了同步)。根据Ascend Pytorch官方仓库[链接](https://gitee.com/ascend/pytorch#%E6%98%87%E8%85%BE%E8%BE%85%E5%8A%A9%E8%BD%AF%E4%BB%B6)，CANN的版本号为7.0.RC1，可以从[这里](https://www.hiascend.com/software/cann/community)下载对应的CANN版本，CPU架构选择`AArch64`, 软件包格式选择`run`, 注意选择带有`toolkit`名字的包，我这里是`Ascend-cann-toolkit_7.0.0.alpha001_linux-aarch64.run`。

3. 固件和驱动往往跟CANN的版本也有一个匹配关系，选定了CANN版本后，就可以到[这里](https://www.hiascend.com/zh/hardware/firmware-drivers/community)来下载对应的固件驱动版本。产品系列选择`服务器`，产品型号选`Atlas 800 训练服务器（型号：9000）`（PS:如果服务器的CPU是x86的，则选择`Atlas 800 训练服务器（型号：9010）`，CANN版本选择`7.0.RC1.alpha002`(因为这里没有alpha001,就选002即可，固件驱动支持的CANN版本有个范围)，这时固件驱动版本号自动确定为了`1.0.20.alpha`，组件不用选择，软件包格式选择`run`。选完后，下面列表中出现两个软件，分别是固件`Ascend-hdk-910-npu-firmware_6.4.12.1.241.run`和驱动`Ascend-hdk-910-npu-driver_23.0.rc2_linux-aarch64.run`，将它们都下下来。

4. 在root用户下安装好conda的base环境后，直接用root用户安装固件和驱动。注意：如果是首次安装请按照“驱动 > 固件”的顺序，分别安装驱动和固件包；覆盖安装请按照“固件 > 驱动”的顺序，分别安装固件和驱动包!!! [参考](https://support.huawei.com/enterprise/zh/doc/EDOC1100289994)
   ```bash
   # 安装驱动
   chmod +x Ascend-hdk-910-npu-driver_23.0.rc2_linux-aarch64.run
   ./Ascend-hdk-910-npu-driver_23.0.rc2_linux-aarch64.run --full --install-for-all #(注意：install-for-all很重要！！！)

   # 安装固件
   chmod +x Ascend-hdk-910-npu-firmware_6.4.12.1.241.run
   ./Ascend-hdk-910-npu-firmware_6.4.12.1.241.run --full

   # 重启机器后生效
   reboot
   ```

5. 验证固件驱动是否安装成功
   ```bash
   npu-smi info
   ```

## 安装CANN

CANN一般需要用root用户来安装，这样方便所有的用户使用。由于安装过程中需要使用到python相关的包，根据我的经验，安装之前在root用户用conda部署一个base环境（默认激活），这样方便安装缺少的python依赖包。

做好前面的准备工作后，可以直接安装CANN：

```bash
./Ascend-cann-toolkit_7.0.0.alpha001_linux-aarch64.run --install --install-for-all --install-path=/usr/local/Ascend
```

CANN安装完成后，还需要安装几个CANN自带的python包，下面用用户自己账户安装：

```bash
# 删除之前的安装
pip uninstall te topi hccl -y

# pip install -I /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/topi-*-py3-none-any.whl -i https://mirrors.bfsu.edu.cn/pypi/web/simple
pip install -I /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/te-*-py3-none-any.whl -i https://mirrors.bfsu.edu.cn/pypi/web/simple
pip install -I /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/hccl-*-py3-none-any.whl -i https://mirrors.bfsu.edu.cn/pypi/web/simple
```

其中：

1. topi 是CANN中的一个关键模块，它用于定义和优化深度学习算子（操作）的计算。topi 可以看作是一种领域特定语言（DSL），用于描述各种深度学习算子的计算和优化方法。
2. te 是 topi 的一部分，用于构建和表示深度学习算子的计算表达式。它允许开发人员描述计算，包括输入、输出和计算操作，以便进一步优化。
3. hccl 是昇腾AI处理器上的通信库，用于处理分布式计算和通信操作。它提供了一种方式来支持多卡通信，数据分发和集合操作。

还需要配置CANN的环境变量，将下面配置放到个人用户的`~/.bashrc`文件里面:

```bash
# Default path, change it if needed.
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 配置device的IP

[参考官方文档](https://support.huaweicloud.com/instg-cli-cann/atlascli_03_0047.html)

利用root账户，运行一下命令：

```bash
hccn_tool -i 0 -ip -s address 192.168.100.101 netmask 255.255.255.0
hccn_tool -i 1 -ip -s address 192.168.101.101 netmask 255.255.255.0
hccn_tool -i 2 -ip -s address 192.168.102.101 netmask 255.255.255.0
hccn_tool -i 3 -ip -s address 192.168.103.101 netmask 255.255.255.0
hccn_tool -i 4 -ip -s address 192.168.100.100 netmask 255.255.255.0
hccn_tool -i 5 -ip -s address 192.168.101.100 netmask 255.255.255.0
hccn_tool -i 6 -ip -s address 192.168.102.100 netmask 255.255.255.0
hccn_tool -i 7 -ip -s address 192.168.103.100 netmask 255.255.255.0
```

NPU卡需要配置IP后，才可以进行正常的通信。

## 安装Ascend Pytorch

前面已经选定了安装torch=2.1.0版本。做好前面的准备工作后，可以开始Ascend Pytorch的安装。参考[官方安装步骤](https://gitee.com/ascend/pytorch?_from=gitee_search#%E5%AE%89%E8%A3%85).

| 依赖项    | 至少    | 推荐        |
| --------- | ------- | ----------- |
| CANN      | 8.0.RC1 | 8.0.RC1     |
| torch     | 2.1.0   | 2.1.0       |
| torch-npu | 2.1.0   | 2.1.0.post3 |
| deepspeed | 0.13.2  | 0.13.2      |
pip install torch==2.1.0

# x86
pip install torch==2.1.0+cpu  --index-url https://download.pytorch.org/whl/cpu
```

### 安装torch-npu

1. 安装依赖

    ```bash
    pip install pyyaml
    pip install setuptools
    ```

2. pip直接安装torch_npu

    ```bash
    pip install torch-npu==2.1.0rc1
    ```

3. 也可采用源码编译安装

    ```bash
    # 克隆项目
    git clone https://gitee.com/ascend/pytorch.git -b v2.1.0-5.0.rc3 --depth 1

    # 编译
    bash ci/build.sh --python=3.9

    # pip安装
    pip install dist/torch_npu-2.1.0rc1+gitb6656eb-cp39-cp39-linux_aarch64.whl
    ```

4. 验证torch_npu是否安装成功

    ```python
    import torch
    import torch_npu
| 软件类型    | 软件介绍                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 昇腾NPU固件 | 固件包含昇腾AI处理器自带的OS 、电源器件和功耗管理器件控制软件，分别用于后续加载到AI处理器的模型计算、芯片启动控制和功耗控制。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| 昇腾NPU驱动 | 部署在昇腾服务器，管理查询昇腾AI处理器，同时为上层CANN软件提供芯片控制、资源分配等接口。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| CANN        | 部署在昇腾服务器，包含Runtime、算子库、图引擎、媒体数据处理等组件，通过AscendCL（Ascend Computing Language，昇腾计算语言）对外提供Device管理、Context管理、Stream管理、内存管理、模型加载与执行、算子加载与执行、媒体数据处理等API，帮助开发者实现在昇腾软硬件平台上开发和运行AI业务。CANN软件按照功能主要分为Toolkit（开发套件）、Kernels（二进制算子包）、NNAE（深度学习引擎）、NNRT（离线推理引擎）、NNAL（加速库）、TFPlugin（TensorFlow框架插件）几种软件包，各软件包支持功能范围如下：Toolkit：支持训练和推理业务、模型转换、算子/应用/模型开发和编译。NNAE：支持训练和推理业务。NNRT：仅支持离线推理。TFPlugin：用于运行训练业务时和TensorFlow框架进行对接，帮助TensorFlow框架调用底层CANN接口运行训练业务。Kernels：依赖Toolkit、NNRT或NNAE软件包，可以节省算子编译时间。在包含动态shape网络或单算子API（例如aclnn类API）场景下需安装二进制算子包。NNAL：依赖Toolkit，包含面向大模型领域的ATB（Ascend Transformer Boost）加速库，可以提升大模型训练和推理性能。 |
    ```

### 安装Ascend Apex

Ascend Apex以patch的形式发布，使能用户在华为昇腾（HUAWEI Ascend）AI处理器上，结合原生Apex进行混合精度训练，以提升AI模型的训练效率，同时保持模型的精度和稳定性。Ascend Apex需要编译安装。

```bash
git clone -b master https://gitee.com/ascend/apex.git
cd apex/

# 这一步是必须的，否则后续可能会报错。先将setuptools降级，等安装完后再升级到最新的即可。
pip install setuptools==41.2.0

# 编译
# 注意：替换里面官方的Nvidai/apex的github仓库为：https://gitee.com/liuhanpeng/apex.git
bash scripts/build.sh --python=3.9

# pip安装
cd apex/dist
pip uninstall apex
pip install --upgrade aapex-0.1-cp37-cp37m-linux_aarch64.whl

# 再重新升级setuptools
pip install -U setuptools
```

## FAQ

1. 报错信息如下：

    ```bash
    TypeError: Descriptors cannot not be created directly.
    If this call came from a _pb2.py file, your generated code is out of date and must be regenerated with protoc >= 3.19.0.
    If you cannot immediately regenerate your protos, some other possible workarounds are:
    1. Downgrade the protobuf package to 3.20.x or lower.
    2. Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python (but this will use pure-Python parsing and will be much slower).
    ```

    解决办法：

    ```bash
    pip install protobuf==3.20.0
    ```
