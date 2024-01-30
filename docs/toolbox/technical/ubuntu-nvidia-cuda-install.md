## Linux 安装(升级) cmake

### 第一种方法（不推荐）

直接使用`apt`安装，但是安装的版本很老，不推荐这种方法

```bash
sudo apt install cmake
```

### 第二种方法（cmake源码编译）

从https://cmake.org/download/下载源码，如cmake-3.24.1.tar.gz 解压包

``` shell
tar -zxvf cmake-3.24.1.tar.gz
```

进入到解压后的文件夹，然后执行bootstrap文件进行检查

```shell
cd cmake-3.24.1
./bootstrap
```

检查没有发现问题的话，进行安装
-j8是选择八核编译，如果是电脑是四核就make -j4，不清楚的就直接make，影响不大，只是编译速度的变化

```
make -j8
sudo make install
```

查看cmake版本 ```cmake --version```

查看cmake路径 ```which cmake```

如果成功输入版本信息，则编译安装成功.

### 第三种方法（ppa安装，推荐！）

添加签名密钥

```shell
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | sudo apt-key add -
```

将存储库添加到您的源列表并进行更新

```shell
sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
sudo apt-get update
```

稳定版

```shell
sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
sudo apt-get update
```

候选发布版本

```shell
sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic-rc main'
sudo apt-get update
```

然后再使用apt安装就是最新版本的cmak.

```shell
sudo apt install cmake
```



## Linux 安装(升级) gcc

### 步骤 1：安装 GCC 之前更新 Ubuntu

在开始之前，请更新系统以确保所有现有软件包都是最新的，以避免安装过程中出现任何冲突。

```bash
sudo apt update
sudo apt upgrade
```

### 步骤2：选择GCC安装方法

#### 方法 1：使用 Ubuntu 存储库安装 GCC

安装 GCC 的第一个推荐选项是直接安装 GCC 软件包，或者安装包含 GCC 和许多其他基本开发工具（例如 make、g++ 和 dpkg-dev）的 build-essential 软件包。

要开始安装，请使用以下命令。

```bash
sudo apt install gcc
```

或者

```bash
sudo apt install build-essential
```

安装后，验证安装并使用以下命令检查版本。

```bash
gcc --version
```

#### 方法2：通过工具链PPA在Ubuntu上安装GCC

以下方法将安装最新的 GCC 编译器或您可以从[Ubuntu 工具链 PPA](https://launchpad.net/~ubuntu-toolchain-r/+archive/ubuntu/ppa)中寻找的替代版本。要导入此 PPA，请运行以下命令：

```bash
sudo add-apt-repository ppa:ubuntu-toolchain-r/ppa -y
```

导入 PPA 后，更新 Ubuntu 源列表以反映通过在终端中运行以下命令所做的更改：

```bash
sudo apt update
```

要使用 Ubuntu ToolChain PPA 在 Ubuntu 系统上安装特定版本的 GCC 编译器，请在终端中使用以下命令：

- GCC 编译器 13

```bash
sudo apt install g++-13 gcc-13
```

- GCC 编译器 12

```bash
sudo apt install g++-12 gcc-12
```

### 配置 GCC 的替代版本

作为开发人员或特定用户，您可能需要安装多个 GCC 编译器版本。请按照以下步骤在 Ubuntu 系统上配置 GCC 的替代版本。

首先，安装您需要的 GCC 版本。您可以使用以下命令安装多个版本的 GCC 和 G++：

```bash
sudo apt install gcc-9 g++-9 gcc-10 g++-10 gcc-11 g++-11 g++-12 gcc-12 g++-13 gcc-13
```

安装必要的版本后，请使用 update-alternatives 命令配置每个版本的优先级。以下示例命令设置 GCC 9、GCC 10、GCC 11 和最新的 GCC 12 之间的优先级划分。

```bash
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 100 --slave /usr/bin/g++ g++ /usr/bin/g++-13 --slave /usr/bin/gcov gcov /usr/bin/gcov-13

sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 90 --slave /usr/bin/g++ g++ /usr/bin/g++-12 --slave /usr/bin/gcov gcov /usr/bin/gcov-12

sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 80 --slave /usr/bin/g++ g++ /usr/bin/g++-11 --slave /usr/bin/gcov gcov /usr/bin/gcov-11

sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 60 --slave /usr/bin/g++ g++ /usr/bin/g++-10 --slave /usr/bin/gcov gcov /usr/bin/gcov-10

sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 40 --slave /usr/bin/g++ g++ /usr/bin/g++-9 --slave /usr/bin/gcov gcov /usr/bin/gcov-9
```

上述命令将 GCC 13 设置为最高优先级，值为 100，您可以根据自己的喜好配置优先级.

要确认 GCC 13 是系统上的默认版本，请运行以下命令：

```bash
gcc --version
```

您可以使用 update-alternatives 命令重新配置系统上的默认 GCC 版本。首先，使用以下命令列出您之前设置的优先级：

```bash
sudo update-alternatives --config gcc
```

输出示例：

```shell
There are 5 choices for the alternative gcc (providing /usr/bin/gcc).

  Selection    Path             Priority   Status
------------------------------------------------------------
* 0            /usr/bin/gcc-11   100       auto mode
  1            /usr/bin/gcc-10   80        manual mode
  2            /usr/bin/gcc-11   100       manual mode
  3            /usr/bin/gcc-13   90        manual mode
  4            /usr/bin/gcc-7    80        manual mode
  5            /usr/bin/gcc-9    40        manual mode

Press <enter> to keep the current choice[*], or type selection number:
```



##  Nvidia 驱动卸载

1. 卸载驱动

```shell
sudo apt-get --purge remove nvidia*
sudo apt autoremove
```

2. To remove CUDA Toolkit:

```shell
sudo apt-get --purge remove "*cublas*" "cuda*"
```

To remove NVIDIA Drivers:

```shell
sudo apt-get --purge remove "*nvidia*"
```

3.然后重装驱动

## Cudnn 安装

1. Unzip the cuDNN package.

>```shell
>tar -xvf cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz
>```

2. Copy the following files into the CUDA toolkit directory.

>```shell
>sudo cp cudnn-linux-x86_64-8.6.0.163_cuda11-archive/include/cudnn*.h /usr/local/cuda/include
>sudo cp -P cudnn-linux-x86_64-8.6.0.163_cuda11-archive/lib/libcudnn* /usr/local/cuda/lib64
>sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
>```

## PiP 换源

### pip 国内的一些镜像

- 阿里云 http://mirrors.aliyun.com/pypi/simple/
- 中国科技大学 https://pypi.mirrors.ustc.edu.cn/simple/
- 豆瓣(douban) http://pypi.douban.com/simple/
- 清华大学 https://pypi.tuna.tsinghua.edu.cn/simple/
- 中国科学技术大学 http://pypi.mirrors.ustc.edu.cn/simple/

### 修改源方法：

#### 临时使用：

**Linux Mac Windows 通用命令**

可以在使用pip的时候在后面加上-i参数，指定pip源

> pip install scrapy -i https://pypi.tuna.tsinghua.edu.cn/simple

#### 永久修改：

**Linux:**

修改 pip.conf 文件 (没有就创建一个)

> $HOME/.config/pip/pip.conf

修改内容如下：

> [global]
>
> index-url = https://pypi.tuna.tsinghua.edu.cn/simple

或者直接使用下面的命令：

```shell
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
```

## pip 导出所有包

若要与其他人共享项目、使用生成系统，或打算将项目复制到需要在其中还原环境的其他任何位置，必须指定项目需要的外部包。 建议的方法是使用 [requirements.txt 文件](https://pip.readthedocs.org/en/latest/user_guide.html#requirements-files) (readthedocs.org)，文件中包含安装相关包所需版本的 pip 命令列表。 最常见的命令是 `pip freeze > requirements.txt`，它将环境的当前包列表记录到 requirements.txt 中。

```shell
pip freeze > requirements.txt
```

## conda源加速

> 参考资料：https://www.cnblogs.com/VVingerfly/p/12046586.html

在linux系统下，conda的配置文件存储在`~/.condarc`中，将下面的配置文件粘贴到配置文件中即可获得一个满速的`conda install`体验：

```shell
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
```
