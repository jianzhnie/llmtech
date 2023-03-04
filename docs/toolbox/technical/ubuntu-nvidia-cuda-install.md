##  nvidia 驱动卸载

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



## cudnn

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

### [Verifying the Install on Linux](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#verify)

To verify that cuDNN is installed and is running properly, compile the mnistCUDNN sample located in the /usr/src/cudnn_samples_v8 directory in the Debian file.

### Procedure

1. Copy the cuDNN samples to a writable path.

   ```
   cp -r /usr/src/cudnn_samples_v8/ $HOME
   ```

2. Go to the writable path.

   ```
   cd  $HOME/cudnn_samples_v8/mnistCUDNN
   ```

3. Compile the mnistCUDNN sample.

   ```
   make clean && make
   ```

4. Run the mnistCUDNN sample.

   ```
   ./mnistCUDNN
   ```

   If cuDNN is properly installed and running on your Linux system, you will see a message similar to the following:

   ```
   Test passed!
   ```

## 解压缩

Linux下常见的压缩包格式有5种

- zip
- tar.gz
- tar.bz2
- tar.xz
- tar.Z

其中tar是种打包格式；gz和bz2等后缀才是指代压缩方式：gzip和bzip2

**从1.15版本开始tar就可以自动识别压缩的格式,故不需人为区分压缩格式就能正确解压**，所以，解压的命令为：

```
tar -xvf filename.tar.gz
tar -xvf filename.tar.bz2
tar -xvf filename.tar.xz
tar -xvf filename.tar.Z
```

这里参数解释为：

`x`：e**x**tract 解压

`v`：**v**erbose 详细信息

`f`：**f**ile(file=archieve) 文件

## pip 换源

### pip国内的一些镜像

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
