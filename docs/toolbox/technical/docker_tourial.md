# Install Docker CE

## 安装 Docker：

> ```shell
> # Docker installation using the convenience script
> $ curl -fsSL https://get.docker.com -o get-docker.sh
> $ sudo sh get-docker.sh
>
> # Post-install steps for Docker
> $ sudo groupadd docker
> $ sudo usermod -aG docker $USER
> $ newgrp docker
>
> # Verify Docker
> $ docker run hello-world
> ```
>
> 也可以看看
>
> - [在 Ubuntu 上安装 Docker 引擎](https://docs.docker.com/engine/install/ubuntu)
> - [Linux 的安装后步骤](https://docs.docker.com/engine/install/linux-postinstall)

## 安装 NVIDIA 容器工具包：

> ```shell
> # Configure the repository
> $ curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
>   && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
>     sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
>     sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
>   && \
>     sudo apt-get update
>
> # Install the NVIDIA Container Toolkit packages
> $ sudo apt-get install -y nvidia-container-toolkit
> $ sudo systemctl restart docker
>
> # Configure the container runtime
> $ sudo nvidia-ctk runtime configure --runtime=docker
> $ sudo systemctl restart docker
>
> # Verify NVIDIA Container Toolkit
> $ docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
> ```
>
> 笔记
>
> - 需要[NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) 1.15.0 或更高版本。



# Docker Hub 镜像加速

国内从 Docker Hub 拉取镜像有时会遇到困难，此时可以配置镜像加速器。

## Docker daemon 配置代理（推荐）

参考 [Docker daemon 配置代理](https://docs.docker.com/config/daemon/systemd/#httphttps-proxy)

## 自建镜像加速服务

- [自建镜像仓库代理服务](https://github.com/bboysoulcn/registry-mirror)
- [利用 Cloudflare Workers 自建 Docker Hub 镜像](https://github.com/ImSingee/hammal)

## 国内三方加速镜像

> ⚠️⚠️⚠️ 自 2024-06-06 开始，国内的 Docker Hub 镜像加速器相继停止服务，可选择为 Docker daemon 配置代理或自建镜像加速服务。

创建或修改 `/etc/docker/daemon.json`：

```bash
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json
{
    "insecure-registries": [
        "swr.cn-south-222.ai.pcl.cn",
        "https://swr.cn-east-317.qdrgznjszx.com"
    ],
    "registry-mirrors": [
        "https://registry.docker-cn.com",
        "https://docker.m.daocloud.io",
        "https://docker.mirrors.ustc.edu.cn",
        "https://52jfog84.mirror.aliyuncs.com",
        "http://hub-mirror.c.163.com",
        "https://mirror.ccs.tencentyun.com",
        "https://ccr.ccs.tencentyun.com",
        "https://dockerproxy.com",
        "https://mirror.baidubce.com",
        "https://hub.uuuadc.top",
        "https://docker.anyhub.us.kg",
        "https://dockerhub.jobcher.com",
        "https://dockerhub.icu",
        "https://docker.ckyl.me",
        "https://docker.awsl9527.cn",
        "https://docker.hpcloud.cloud",
        "https://docker.registry.cyou",
        "https://docker-cf.registry.cyou",
        "https://dockercf.jsdelivr.fyi",
        "https://docker.jsdelivr.fyi",
        "https://mirror.aliyuncs.com",
        "https://docker.m.daocloud.io",
        "https://docker.nju.edu.cn",
        "https://docker.mirrors.sjtug.sjtu.edu.cn",
        "https://mirror.iscas.ac.cn",
        "https://docker.rainbond.cc"
    ],
    "runtimes": {
        "nvidia": {
            "args": [],
            "path": "nvidia-container-runtime"
        }
    }
}

sudo systemctl daemon-reload
sudo systemctl restart docker
```

## 检查加速器是否生效

命令行执行 `docker info`，如果从结果中看到了如下内容，说明配置成功。

```shell
Registry Mirrors:
 [...]
 https://docker.m.daocloud.io
```

## Docker Hub 镜像测速

使用镜像前后，可使用 `time` 统计所花费的总时间。测速前先移除本地的镜像！

```shell
docker rmi node:latest

time docker pull node:latest


Pulling repository node
[...]

real   1m14.078s
user   0m0.176s
sys    0m0.120s
```
