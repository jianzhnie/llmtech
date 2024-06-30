1. 安装 Docker：

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

2. 安装 NVIDIA 容器工具包：

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
