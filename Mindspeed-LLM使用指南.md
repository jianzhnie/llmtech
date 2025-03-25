# MindSpeed-LLM 使用指南

## 1. 简介

#### 1.1 MindSpeed简介

MindSpeed是专门面向昇腾（Ascend）平台的大模型训练加速库。昇腾是华为推出的高性能AI计算平台，广泛应用于大模型训练、推理和部署场景。

在大模型训练领域，硬件资源的高效利用至关重要。MindSpeed通过优化内存管理、计算调度以及通信效率，帮助用户在有限的硬件资源下实现更高的训练效率。无论是单机多卡训练，还是大规模分布式训练，MindSpeed都能提供灵活且高效的解决方案，其核心特性包括：megetron特性支持、并行策略特性、内存优化特性、亲和计算特性、通信优化特性以及关键场景特性。

#### 1.2. MindSpeed-LLM核心功能

MindSpeed-LLM是MindSpeed库中专门针对大语言模型（LLM）训练的模块套件。旨在为[昇腾芯片](https://link.juejin.cn?target=https%3A%2F%2Fwww.hiascend.com%2F)提供端到端的大预言模型训练解决方案, 包含预置业界主流模型，数据工程，分布式训练及加速，预训练、微调、在线推理任务等特性提供了以下核心功能：

- **模型并行与数据并行**：支持多种并行策略，包括张量并行、流水线并行和数据并行，以适应不同规模的模型和硬件配置。
- **混合精度训练**：通过自动混合精度（AMP）技术，在保证模型精度的同时，显著降低显存占用和计算开销。
- **梯度累积与梯度检查点**：在显存有限的情况下，通过梯度累积模拟更大的批量大小，并通过梯度检查点减少显存峰值占用。
- **分布式通信优化**：针对昇腾平台的通信特性，优化分布式训练中的梯度同步和参数更新，减少通信开销。
- **动态内存管理**：通过动态内存分配和释放策略，优化显存使用，避免显存不足导致的训练中断。

#### 1.3 业界主流加速库对比

在大模型训练加速领域，业界有许多主流的加速库，如DeepSpeed、FasterTransformer、Megatron-LM等。这些库各有优劣，适用于不同的场景和硬件平台。以下是MindSpeed-LLM与业界主流加速库的对比：

| 特性               | MindSpeed-LLM      | DeepSpeed          | Megatron-LM        | FasterTransformer  |
| ------------------ | ------------------ | ------------------ | ------------------ | ------------------ |
| **硬件平台**       | 昇腾（Ascend）     | 通用GPU            | 通用GPU            | 通用GPU            |
| **模型并行**       | 支持               | 支持               | 支持               | 支持               |
| **混合精度训练**   | 支持               | 支持               | 支持               | 支持               |
| **梯度检查点**     | 支持               | 支持               | 支持               | 不支持             |
| **分布式通信优化** | 针对昇腾优化       | 通用优化           | 通用优化           | 通用优化           |
| **动态内存管理**   | 支持               | 支持               | 不支持             | 不支持             |
| **易用性**         | 集成昇腾生态，易用 | 功能丰富，配置复杂 | 功能强大，配置复杂 | 专注于推理，轻量化 |

从对比中可以看出，MindSpeed-LLM在昇腾平台上具有显著的优势，特别是在硬件优化和动态内存管理方面。对于使用昇腾平台的用户来说，MindSpeed-LLM是一个高效且易用的选择。



## 2. 环境搭建

### 2.1 基础软硬件

在开始训练之前，需要搭建好训练环境，包括硬件和软件的配置。

- **硬件选择**：选择适合大模型训练的硬件，如Atlas 800T A2。本文实践采用的是Atlas 800T A2硬件设备。
- **操作系统**：本文实践所用操作系统为eulerosv2r12.aarch64。
- 软件配套
  - **固件及驱动安装**：因本事件所用昇腾服务器相关NPU驱动和固件已经安装好了，如果你有需要可以参照昇腾社区上的安装指导[NPU驱动固件安装指导](https://link.juejin.cn?target=https%3A%2F%2Fwww.hiascend.com%2Fdocument%2Fdetail%2Fzh%2Fcanncommercial%2F80RC2%2Fsoftwareinst%2Finstg%2Finstg_0003.html%3FMode%3DPmIns%26OS%3DUbuntu%26Software%3DcannToolKit)（注意首次安装场景和覆盖安装场景固件和驱动安装顺序的不同要求）
  - **CANN依赖安装**： CANN依赖主要有以下三个,务必先安装toolkot后再安装后两个依赖包。

| 软件类型 | 软件包说明                                                   | 软件包名称                                 |
| -------- | ------------------------------------------------------------ | ------------------------------------------ |
| Toolkit  | **CANN开发套件包，在训练&推理&开发调试场景下安装，主要用于训练和推理业务、模型转换、算子/应用/模型的开发和编译**。 | Ascend-cann-toolkit__linux-.run            |
| Kernels  | **CANN算子包，能够节省算子编译时间，在大模型推理、训练场景和运行包含动态shape网络或单算子API（例如aclnn类API）相关业务时安装。**安装时需已安装Toolkit或NNRT或NNAE软件包，请选择运行设备对应处理器类型的Kernels。 | Ascend-cann-kernels-<chip_type>__linux.run |
| NNAL     | **CANN加速库，包含面向大模型领域的ATB（Ascend Transformer Boost）加速库，可以提升大模型训练和推理性能。**安装时需已安装Toolkit或NNAE软件包。 | Ascend-cann-nnal__linux-.run               |

特别说明，Cann相关依赖安装即toolkit、kernels、nnal三个依赖，务必按照指导顺序安装。本文实践按照官方对物理机场景安装操作进行安装， 另，**toolkit及nnal安装后务必记得source 环境。**

后续使用改软件依赖环境可以选择基于容器配置大模型训练环境，也可以基于Conda创建虚拟环境配置大模型训练环境。对于容器可以选择将cann依赖装入镜像环境中进行隔离操作，而对于conda虚拟环境，安装cann依赖时需要需要单独每次session建立后source激活依赖环境（toolkit\nnal都需要source激活）。

- **Ascend Extension for PyTorch 配置与安装**，该依赖主要为解决pytorch生态在昇腾硬件兼容适配问题，具体安装操作很简单，可以参考官方指导[昇腾社区Ascend Extension Pytorch安装指导](https://link.juejin.cn?target=https%3A%2F%2Fwww.hiascend.com%2Fdocument%2Fdetail%2Fzh%2FPytorch%2F60RC3%2Fconfigandinstg%2Finstg%2Finsg_0001.html).
- **MindSpeed-LLM安装**：按照官方文档安装MindSpeed-LLM，可能需要从源码编译安装，确保所有依赖项都已安装。 严格按照gitee上的操作指导进行即可， **注意**：apex源码出包可能会出现问题，最好直接找已经编译好的apex，whl包，我试图在apex社区通过源码编译whl包，但出错。

### 2.2 MindSpeed-LLM

参考： [install_guide](https://gitee.com/ascend/MindSpeed-LLM/blob/master/docs/features/install_guide.md)

#### torch & torch-npu & apex 安装

准备[torch_npu](https://gitee.com/link?target=https%3A%2F%2Fwww.hiascend.com%2Fdeveloper%2Fdownload%2Fcommunity%2Fresult%3Fmodule%3Dpt)和[apex](https://gitee.com/ascend/apex)，参考[Ascend Extension for PyTorch 配置与安装](https://gitee.com/link?target=https%3A%2F%2Fwww.hiascend.com%2Fdocument%2Fdetail%2Fzh%2FPytorch%2F60RC2%2Fconfigandinstg%2Finstg%2Finsg_0001.html)或执行以下命令安装：

```shell
# 安装torch和torch_npu
pip install torch-2.1.0-cp38-cp38-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
pip install torch_npu-2.1.0.post8-cp38-cp38-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

# apex for Ascend 构建参考 https://gitee.com/ascend/apex
pip install apex-0.1.dev20241015+ascend-cp38-cp38-linux_aarch64.whl
```

#### MindSpeed 及相关依赖安装

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh  # source ascend-toolkit环境变量

# 安装MindSpeed加速库
git clone https://gitee.com/ascend/MindSpeed.git
cd MindSpeed
git checkout 0dfa0035ec54d9a74b2f6ee2867367df897299df  # checkout commit from MindSpeed core_r0.8.0 in 2025.02.26
pip install -r requirements.txt
pip3 install -e .
cd ..
```

#### MindSpeed-LLM及相关依赖安装

```shell
# 准备MindSpeed-LLM及Megatron-LM源码
git clone https://gitee.com/ascend/MindSpeed-LLM.git
git clone https://github.com/NVIDIA/Megatron-LM.git  # megatron从github下载，请确保网络能访问
cd Megatron-LM
git checkout core_r0.8.0
cp -r megatron ../MindSpeed-LLM/
cd ../MindSpeed-LLM

pip install -r requirements.txt  # 安装其余依赖库
```



## 3. 全流程实践

### 3.1 必看前置

MindSpeed-LLM有两种模式下得大模型训练，分别是Mcore、Legacy。关于两种模式的差异，社区上并未给出任何功能定位解释，不过通过Readme特性解释可以看出，相较于legacy，Mcore模式下得大模型训练做了更多的并行加速特性支持，如长序列并行优化、MOE专家并行优化等高阶优化特性支持，即Mcore模式下的大模型训练性能会优于legacy，至于有了更高性能的mcore模式，为什么还要并行存在legacy，社区给的解释是：legacy为早期版本模式，很多商用客户基于此模式在做版本维护，不能随意日落。

但是，通过查看特性差异以及“/example/legacy、example/mcore”路径下得支持不同执行任务启动脚本，相比mcore模式大模型训练在官方支持的模型任务上，legacy缺失很多可以直接运行的shell启动脚本，如指令微调数据转换脚本、微调启动脚本、指令微调后chat对话脚本等，如果你是一个MindSpeed纯新手，误入legacy模式按照官方指导操作，还会遇到各种错误或者文件缺失。

> 所以：
>
> 一、直接在mcore下进行全流程操作；
>
> 二、不要按照主页readme上的脚本执行任务指令，应当使用这个路径下合适的启动命令“/example/mcore/xx/xxx.sh”，脚本缺失就自己补对应模型的启动脚本即可。



### 3.2 权重下载及转换

#### 3.2.1 下载Hugingface 权重

```shell
# 配置国内镜像源
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_ENABLE_HF_TRANSFER=0

# download  model
huggingface-cli download  Qwen/Qwen2.5-0.5B  --local-dir /home/robin/hf_hub/models/Qwen/Qwen2.5-0.5B
huggingface-cli download  Qwen/Qwen2.5-1.5B-Instruct  --local-dir /home/robin/hf_hub/models/Qwen/Qwen2.5-1.5B-Instruct
```

如果你没办法科学上网，也可以直接在modelscope上下载对应的模型权重下载，**尽量不要使用git或网页直接下载，拉取的模型权重可能不全，导致后续模型权重转换出现问题，**

modelscope 权重下载

```python
from modelscope import snapshot_download
model_dir = snapshot_download("Qwen/Qwen1.5-0.5B", local_dir= "./")
```

#### 3.2.2 权重转换

MindSpeed-LLM 支持 huggingface、megatron-core、megatron-legacy 三种格式的权重互转，支持 Lora 权重合并。权重转换特性参数和使用说明参考 [权重转换](./docs/features/checkpoint.md)。

| 源格式          | 目标格式        | 切分特性                             | lora | 贡献方     |
| --------------- | --------------- | ------------------------------------ | ---- | ---------- |
| huggingface     | megatron-core   | tp、pp、dpp、vpp、cp、ep、loop layer | ❌    | 【Ascend】 |
| huggingface     | megatron-legacy | tp、pp、dpp、vpp、cp、ep、loop layer | ❌    | 【Ascend】 |
|                 |                 |                                      |      |            |
| megatron-core   | huggingface     |                                      | ✅    | 【Ascend】 |
| megatron-core   | megatron-legacy | tp、pp、dpp、vpp、cp、ep、loop layer | ✅    | 【Ascend】 |
| megatron-core   | megatron-core   | tp、pp、dpp、vpp、cp、ep、loop layer | ✅    | 【Ascend】 |
|                 |                 |                                      |      |            |
| megatron-legacy | huggingface     |                                      | ✅    | 【Ascend】 |
| megatron-legacy | megatron-core   | tp、pp、dpp、vpp、cp、ep、loop layer | ✅    | 【Ascend】 |
| megatron-legacy | megatron-legacy | tp、pp、dpp、vpp、cp、ep、loop layer | ✅    | 【Ascend】 |

Huggingface权重转换到Megatron-LM格式，注意官方给出的权重转化脚本示例为：

```shell
# 命名及启动
# bash examples/mcore/model_name/ckpt_convert_xxx_hf2mcore.sh* *
# 需要配置并行参数以及权重词表加载保存等路径*
```

#####  Qwen2.5-0.5B

```shell


# 修改 ascend-toolkit 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 设置需要的权重转换参数
python convert_ckpt.py \
       --use-mcore-models \
       --model-type GPT \
       --load-model-type hf \
       --save-model-type mg \
       --target-tensor-parallel-size 2 \
       --target-pipeline-parallel-size 2 \
       --add-qkv-bias \
       --load-dir /root/llm_dataset/models/Qwen/Qwen2.5-0.5B \
       --save-dir /root/llm_dataset/models/mindspeed/qwen2.5_mcore/qwen2.5_0.5b/ \
       --tokenizer-model /root/llm_dataset/models/Qwen/Qwen2.5-0.5B/tokenizer.json \
       --model-type-hf llama2 \
       --params-dtype bf16 # --num-layer-list 11, 13, 19, 21 参数根据需要添加
```

##### Qwen2.5-7B

```shell
python convert_ckpt.py \
       --use-mcore-models \
       --model-type GPT \
       --load-model-type hf \
       --save-model-type mg \
       --target-tensor-parallel-size 4 \
       --target-pipeline-parallel-size 2 \
       --add-qkv-bias \
       --load-dir /home/ma-user/work/models/Qwen/Qwen2.5-7B \
       --save-dir /home/ma-user/work/models/mindspeed/qwen2.5_mcore/qwen2.5_7b/ \
       --tokenizer-model /home/ma-user/work/models/Qwen/Qwen2.5-7B/tokenizer.json \
       --model-type-hf llama2 \
       --params-dtype bf16 # --num-layer-list 11, 13, 19, 21 参数根据需要添加
```

其他模型可以参考不同的模型路径下的转换脚本。

> 注意：
>
> 转换checkpoint的时候，`examples/mcore/qwen25_math/ckpt_convert_qwen25_math_hf2mcore.sh`里面的`--model-type-hf`保持为llama2，不要改为qwen；否则报错：`AttributeError: 'NoneType' object has no attribute 'weight'`。



### 3.3 数据预处理

MindSpeed-LLM 支持预训练、指令微调、RLHF 等多种任务的数据预处理。

| 任务场景 | 数据集                                                     | Mcore | Legacy | Released | 贡献方     |
| -------- | ---------------------------------------------------------- | ----- | ------ | -------- | ---------- |
| 预训练   | [预训练数据处理](./docs/features/pretrain_dataset.md)      | ✅     | ✅      | ✅        | 【Ascend】 |
| 微调     | [Alpaca风格](./docs/features/alpaca_dataset.md)            | ✅     | ✅      | ✅        | 【Ascend】 |
| 微调     | [ShareGPT风格](./docs/features/sharegpt_dataset.md)        | ✅     | ✅      | ✅        | 【Ascend】 |
| DPO      | [Pairwise数据集处理](./docs/features/pairwise_dataset.md)  | ✅     | ✅      | ✅        | 【NAIE】   |
| SimPO    | [Pairwise数据集处理](./docs/features/pairwise_dataset.md)  | ✅     | ✅      | ❌        | 【NAIE】   |
| ORM      | [Pairwise数据集处理](./docs/features/pairwise_dataset.md)  | ✅     | ✅      | ❌        | 【NAIE】   |
| PRM      | [PRM数据集处理](./docs/features/process_reward_dataset.md) | ✅     | ✅      | ❌        | 【Ascend】 |

注意：特别的社区上已经说明“**在example目录下每个模型都已经预置好数据预处理脚本，可以根据需要来进行修改**”

#### 3.3.1 预训练数据预处理

```shell
# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python preprocess_data.py \
	--input /home/ma-user/work/datasets/open-web-math/open-web-math/data/ \
	--tokenizer-name-or-path /home/ma-user/work/models/Qwen/Qwen2.5-7B/ \
	--tokenizer-type PretrainedFromHF \
	--handler-name GeneralPretrainHandler \
	--cache-dir /home/ma-user/work/datasets/cache_dir \
	--output-prefix /home/ma-user/work/datasets/mindspeed/open-web-math \
	--json-keys text \
	--workers 16  \
	--n-subs 16 \
	--log-interval 1000
```

#### 3.3.2 监督微调数据集处理

在指令监督微调时，`instruction` 列对应的内容会与 `input` 列对应的内容拼接后作为人类指令，即人类指令为 `instruction\ninput`其中 `\n`为用于连接的换行符。而 `output` 列对应的内容为模型回答。如果指定了history，则会将历史对话内容也加入进来。如果指定system 列，则对应的内容将被作为系统提示词。

```shell
# 请按照您的真实环境 source set_env.sh 环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir ./finetune_dataset

python ./preprocess_data.py \
    --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
    --tokenizer-name-or-path ./model_from_hf/llama-2-7b-hf/ \
    --output-prefix ./finetune_dataset/alpaca \
    --workers 4 \
    --log-interval 1000 \
    --tokenizer-type PretrainedFromHF \
    --handler-name AlpacaStyleInstructionHandler \
    --prompt-type llama2  # <-- 需要填入模型模板
    # --map-keys '{"prompt":"instruction","query":"input","response":"output"}' # 默认值，可不传
```

### 3.4. 大模型训练

#### 3.4.1 大模型预训练

按照官方指导，选择对应xxx_pretrain.sh脚本，配置对应权重保存、权重加载、词表、数据集路径，启动训练任务即可。 qwen1.5-0.5B 8卡分布式预训练效果如下示意：

#### 3.4.2 大模型微调

按照官方指导，选择对应tune_xxx.sh脚本，配置对应权重保存、权重加载、词表、数据集路径，启动训练任务即可。如果预置模型没有对应tune启动脚本，则参考其他模型tune启动脚本修改即可。

- 全参微调
- lora微调

#### 3.4.3 大模型分布式推理

- 流式推理
- 按照官方指导，选择对应generate_xx.sh脚本，配置对应权重保存、权重加载、词表、数据集路径，启动训练任务即可。如果预置模型没有对应generate_xx.sh启动脚本，则参考其他模型generate_xx.sh启动脚本修改即可。
- chat对话推理 同样进入example文件路径下mcore对应的模型文件内，如果没有chat脚本，则同样参考已有chat问价修改即可，对比发现只基于对应模型的generate_xx.sh脚本增加图示参数即可实现chat流式推理



# 后训练方法 Ray GRPO

[Group Relative Policy Optimization (GRPO) ](https://arxiv.org/pdf/2402.03300)是 DeepSeek V2中提出的训练方法，它移除了 PPO 中对 Critic模型的依赖，而是使用对同一问题产生的多个采样输出的平均值作为奖励，从而大大减少了显存占用。

GRPO方法中包含了三个模型：Actor，Reference，Reward。其中Actor/Reference模型是经过预训练和指令微调（Supervised Fine-Tuning，SFT）得到的大语言模型，Reward是训练得到的奖励模型。GRPO 的训练目标是使得 Actor 模型的回答可以更加符合人类偏好。

# 使用说明

## 数据预处理

数据集转换参考脚本：MindSpeed-LLM/examples/mcore/llama3/data_convert_llama3_ppo.sh
以 [descriptiveness 数据集](https://huggingface.co/datasets/trl-internal-testing/descriptiveness-sentiment-trl-style/tree/main/data) 为例。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir ./dataset/llama3-hf/

python ./preprocess_data.py \
    --input ./dataset/descriptiveness-00000-of-00001.parquet \
    --tokenizer-name-or-path ./model_from_hf/llama3-hf/ \
    --output-prefix ./dataset/llama3-hf/descriptiveness \
    --workers 16 \
    --log-interval 1000 \
    --tokenizer-type PretrainedFromHF \
    --handler-name  PPOAlpacaStyleInstructionHandler \
    --prompt-type llama3 \
    --map-keys '{"prompt":"prompt", "query":"", "response": "prompt", "system":""}'
```

## 模型权重转换

根据 GRPO 算法要求，Actor 和 Reference 模型应该使用 SFT 微调后的模型进行初始化，Reward 模型应该使用奖励模型训练后的模型进行初始化。GRPO 算法模型权重均使用Megatron-mcore格式，其他格式的权重需要进行模型权重转换，具体可参考[权重转换](./checkpoint.md)。

下面以llama3.2-1b模型作为示例参考：

actor_rollout_ref 涉及到的actor_rollout 与 ref 均需要 SFT 微调后的模型，涉及到的权重转换操作与 SFT 阶段的一致。权重转换示例脚本：
<a href="../../examples/mcore/llama32/ckpt_convert_llama32_hf2mcore.sh">llama32-1b</a>

reward 模型需要使用奖励模型训练后的模型，权重转换示例脚本：<td><a href="../../examples/mcore/llama32/ckpt_convert_llama32_hf2mcore_orm.sh">llama32-1b-orm</a></td>


相应的ppo_trainer_llama32_1b.yaml配置如下
```
  actor_rollout_ref:
    actor_rollout:
      ...
      load: ./model_weights/llama32-mcore/
      save: ./model_weights/llama32-mcore-save/

    ref:
      ...
      load: ./model_weights/llama32-mcore/

  reward:
      ...
      load: ./model_weights/llama32-mcore-orm/
```

## 启动方式

### 单机

通过 --config-name 传递选取的 config 文件名（不添加.yaml后缀），可以通过下列命令直接启动训练（Llama32 1B 模型可单机运行）。
目前已支持的配置文件放置在 configs/rlxf/ 文件夹下。配置文件的具体说明见下文。

```bash
python ray_gpt.py --config-name grpo_trainer_llama32_1b
```

### 多机

多机运行程序时，需要首先进入对应目录，并激活conda或docker环境：

```bash
cd MindSpeed-LLM
conda activate xxx
```

然后，在主节点上启动 Ray 集群：

```bash
# 创建一个集群，端口6344，dashboard端口8260，有8个NPU
ray start --head --port 6344 --dashboard-host=0.0.0.0 --dashboard-port=8260 --resources='{"NPU": 8}'
```

随后，在其他节点加入主节点的集群

```bash
# IP_ADDRESS 处填写主节点 IP 地址
ray start --address="IP_ADDRESS:6344" --resources='{"NPU": 8}'
```

在完成 Ray 集群构建后，在主节点启动运行程序即可（Llama3 8B 模型可双机运行）

```bash
python ray_gpt.py --config-name grpo_trainer_llama3_8b
```

## 配置文件

由于 GRPO 训练过程中涉及 3 个模型，通过将模型参数和训练配置解耦的层级化参数配置，来简化 GRPO 训练的参数配置过程。RLXF 训练涉及到的所有配置文件均存储在 configs/rlxf 路径下，其中 model 文件夹下存储了模型结构相关的配置文件，GRPO训练相关的模型参数文件以grpo_{模型名}.yaml方式命名。

在每个 grpo_trainer 配置文件中，需要包含defaults，training，resource_pool，algorithm等字段，以及  GRPO 训练过程中涉及到的 3 个角色 actor，reward，ref的配置。其中：

1. defaults 负责引入模型配置文件，在 defaults 中应列举本配置文件中所需要用到的所有模型配置，模型配置可以在下方3个角色的具体配置中通过 model 字段进行选择。
2. training 字段设置的参数为所有 3 个角色通用的默认参数，这些参数可以在下方进一步被角色的单独配置所覆盖。
3. resource_pool 字段指定了各个角色所需的 NPU 资源数量。
4. actor，reward，ref 字段分别指定了GRPO算法中三个角色训练相关的参数配置。

## 参数解析

相较于普通模型训练，GRPO增加一些特殊参数：

### `training:`

* `stage`：用于指定训练算法，使用 Ray GRPO 训练须设置为`ray_grpo`;

### `actor_rollout:`

* `do_sample`：控制 Actor 模型进行推理时是否采样，默认为 False，GRPO 需要设置为True ；
* `ppo_mini_batch_size`：Actor 模型的 mini_batch_size，默认为1；
* `max_prompt_length`：GRPO 训练中最大 prompt 长度，默认为512；
* `num_samples_per_step`：Actor 推理时每个step的推理样本数量，默认为1；
* `ppo_epochs`：Actor 训练对同一批经验数据的重复次数，默认为1；
* `clip_ratio`：Actor模型训练计算损失函数时的clip比例，默认为0.2 一般取值范围 [0.1，0.3] 最大取值范围[0，1] 该数值越大允许策略更新的幅度越大，反之不然；
* `shuffle_minibatch`：Actor 训练时是否对 minibatch 进行 shuffle，默认为 False；
* `num_gpus_for_train` ：Actor 模型分配给训练部分的显卡数量；
* `num_gpus_for_infer` ：Actor 模型分配给推理部分的显卡数量；
* `missing_eos_penalty`：缺少序列结束符EOS时的惩罚系数；
* `n_samples_per_prompt`：每条prompt的重用次数，代表GRPO训练流程里每个Group的数据量，默认为1；

### `resource_pool:`

* `actor_rollout`：给 Actor 模型训练和推理总共分配的显卡数量；
* `ref`：给 Reference 模型分配的显卡数量；
* `reward`：给 Reward 模型分配的显卡数量；

# 精度对比

我们与强化学习开源仓库 [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) 进行了精度对比，来辅助验证算法实现的正确性。因为 GRPO group_norm的特性需求，推理状态do sample 设置为 True，为了与基准方法进行精度对齐，在 Actor 推理时固定 responses 方式进行精度对齐的实验。可以看到，固定 responses 后 loss 能够较好地实现对齐。

![grpo_loss_compare.png](../../sources/images/ray_grpo/grpo_loss_compare.png)
