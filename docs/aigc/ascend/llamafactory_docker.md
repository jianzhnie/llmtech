#### LLaMA-Factory-Cann8-Python3.10-Pytorch2.2.0

版本信息

Python版本：3.10.13
CANN版本：8.0
操作系统版本：Ubuntu18.08
昇腾驱动固件版本：23.0.0以上

使用说明

#### **镜像拉取命令**

对应 LLama-Factory 0.7.1 版本

```bash
docker pull swr.cn-east-317.qdrgznjszx.com/donggang/llama-factory-ascend910b:cann8-py310-torch2.2.0-ubuntu18.04
```

#### **镜像启动参考**

创建一个docker_run.sh文件，将下面代码copy到文件中，保存并执行即可启动镜像。

```bash
#!/bin/bash
docker_images=llama-factory-ascend910b:cann8-py310-torch2.2.0-ubuntu18.04

docker run -it -u root --ipc=host --net=host \
        --device=/dev/davinci0 \
        --device=/dev/davinci1 \
        --device=/dev/davinci2 \
        --device=/dev/davinci3 \
        --device=/dev/davinci4 \
        --device=/dev/davinci5 \
        --device=/dev/davinci6 \
        --device=/dev/davinci7 \
        --device=/dev/davinci_manager \
        --device=/dev/devmm_svm \
        --device=/dev/hisi_hdc \
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
        -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
        -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
        -v /home/niejzh:/home/niejzh:rw \
        -v /var/log/npu:/usr/slog ${docker_images} \
        /bin/bash

```

#### **FAQ : docker login报错 x509**

报错打印

```vbnet
Error response from daemon: Get "https://swr.cn-east-317.qdrgznjszx.com/v2/": tls: failed to verify certificate: x509: certificate signed by unknown authority
```

原因证书问题，证书验证错误

配置dns , `vim /etc/docker/daemon.json`  写入下面内容

```json
{
  "insecure-registries": [
  	"https://swr.cn-east-317.qdrgznjszx.com"
		],
 "registry-mirrors": [
   "https://docker.mirrors.ustc.edu.cn"
 		]
}
```

下面是例子：

```json
{
    "insecure-registries": [
        "swr.cn-south-222.ai.pcl.cn",
        "https://swr.cn-east-317.qdrgznjszx.com"
    ],
    "registry-mirrors": [
        "https://docker.mirrors.ustc.edu.cn"
    ],
    "runtimes": {
        "nvidia": {
            "args": [],
            "path": "nvidia-container-runtime"
        }
    }
}
```

然后重启docker即可`systemctl restart docker`

#### 镜像地址

在其他AICC使用本镜像时，

1） 在本地arm宿主机通过docker pull 拉取上面镜像到本地（即执行docker pull remote_image_address）

3） 用docker push 将修改后的镜像名称推送到局点的swr服务中（即执行docker push remote_image_address）

2） 用docker tag 将局点信息和组织名替换成对应版本（即执行 docker tag local_image_address remote_image_address），
