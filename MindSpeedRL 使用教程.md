# 后训练方法 Ray GRPO

##  简介

[Group Relative Policy Optimization (GRPO) ](https://gitee.com/link?target=https%3A%2F%2Farxiv.org%2Fpdf%2F2402.03300)是 Deepseek-Math中提出的训练方法，它移除了 PPO 中对 Critic 模型的依赖，而是通过计算同一prompt多次重复采样输出的相对奖励来估计优势函数，这一创新大大减少了显存占用，提高了算法在强化学习任务中的效率。

在 GRPO 方法中包含了三个关键模型：Actor，Reference，Reward。其中 Actor 和 Reference 模型是通过SFT后得到的策略模型，而 Reward 模型则是通过训练构建的奖励评估模型。GRPO 的核心训练目标是优化 Actor 模型的策略，使其在执行强化学习任务时能够产生更优的动作序列，更符合任务目标的预期。

## 环境配置

### 版本要求

本版本为**预览非正式发布**版本， 依赖配套如下表.

| 依赖软件                               | 版本     |
| -------------------------------------- | -------- |
| 昇腾NPU驱动                            | 25.0.RC1 |
| 昇腾NPU固件                            | 25.0.RC1 |
| Toolkit（开发套件）                    | 8.1.RC1  |
| Kernel（算子包）                       | 8.1.RC1  |
| NNAL（Ascend Transformer Boost加速库） | 8.1.RC1  |
|                                        |          |
| Python                                 | 3.10     |
| torch                                  | 2.5.1    |
| torch_npu                              | 2.5.1    |
| apex                                   | 0.1      |
| ray                                    | 2.42.1   |
| vllm                                   | 0.7.3    |

### 驱动固件安装

```shell
bash Ascend-hdk-*-npu-firmware_*.run --full
bash Ascend-hdk-*-npu-driver_*.run --full
```

### CANN安装

```shell
bash Ascend-cann-toolkit_8.1.RC1_linux-aarch64.run --install
bash Ascend-cann-kernels-*_8.1.RC1_linux-aarch64.run --install
bash Ascend-cann-nnal_8.1.RC1_linux-aarch64.run --install

# 设置环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/asdsip/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
```

### PTA安装

```shell
# 安装torch和torch_npu
pip install torch-2.5.1-cp310-cp310-*.whl
pip install torch_npu-2.5.1.*.manylinux2014_aarch64.whl

# apex for Ascend
# 构建参考 https://gitee.com/ascend/apex
pip install apex-0.1.dev*.whl
```

### vllm及相关依赖安装：

（注：环境中需要安装git，因为vllm的安装过程依赖git）

```shell
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout v0.7.3
pip install -r requirements-build.txt
VLLM_TARGET_DEVICE=empty pip install .
```

### vllm_ascend安装

```shell
git clone -b v0.7.3-dev https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git checkout 0713836e95fe993feefe334945b5b273e4add1f1
pip install -e .
```

### 准备源码

```shell
# Clone MindSpeed-RL repository
git clone https://gitee.com/ascend/MindSpeed-RL.git

# Clone and prepare MindSpeed
git clone https://gitee.com/ascend/MindSpeed.git
cd MindSpeed
git checkout 0dfa0035ec54d9a74b2f6ee2867367df897299df  # 参考MindSpeed-LLM依赖版本
pip install -r requirements.txt
cp -r mindspeed ../MindSpeed-RL/
cd ..

# Clone and prepare Megatron-LM
git clone https://github.com/NVIDIA/Megatron-LM.git  # Megatron从github下载，请确保网络能访问
cd Megatron-LM
git checkout core_r0.8.0
cp -r megatron ../MindSpeed-RL/
cd ..

# Clone and prepare MindSpeed-LLM
git clone https://gitee.com/ascend/MindSpeed-LLM.git
cd MindSpeed-LLM
git checkout 421ef7bcb83fb31844a1efb688cde71705c0526e
cp -r mindspeed_llm ../MindSpeed-RL/
cd ..

# Install MindSpeed-RL dependencies
cd ./MindSpeed-RL
pip install -r requirements.txt
pip install antlr4-python3-runtime==4.7.2 --no-deps
```

## 权重转换

### 模型选择

- Qwen2.5-7B该模型指令遵从度高，有一定概率能引导模型输出`<think>...</think><answer>...$\boxed{}</answer>`格式回复，训练曲线符合预期，在评测集上提升较大。

### 权重转换

在进行RL训练之前，模型需要从HuggingFace权重转换为megatron权重，可参考[**权重转换部分**](https://gitee.com/ascend/MindSpeed-RL/blob/master/docs/algorithms/grpo.md)

```shell
export CUDA_DEVICE_MAX_CONNECTIONS=1
# 修改 ascend-toolkit 路径
source set_env.sh
# conda
source /root/llmtuner/miniconda3/bin/activate mindspeed_rl

hf_model_path=/root/llmtuner/hfhub/models/Qwen/Qwen2.5-7B
mcore_model_path=/root/llmtuner/hfhub/mindspeed/models/Qwen/Qwen2.5-7B

tp=2
pp=4

# actor使用TP2PP4，将脚本里改成TP2PP4配置
# reference使用TP2PP2，将脚本里改成TP2PP2配置

# 设置需要的权重转换参数
python cli/convert_ckpt.py \
       --use-mcore-models \
       --model-type GPT \
       --load-model-type hf \
       --save-model-type mg \
       --target-tensor-parallel-size $tp \
       --target-pipeline-parallel-size $pp \
       --add-qkv-bias \
       --load-dir $hf_model_path \
       --save-dir $mcore_model_path/mcore_tp${tp}_pp${pp} \
       --tokenizer-model $hf_model_path/tokenizer.json \
       --model-type-hf llama2 \
       --params-dtype bf16
```



```shell
bash examples/ckpt/ckpt_convert_qwen25_hf2mcore.sh

# 训练完后如需要转回HF格式
bash examples/ckpt/ckpt_convert_qwen25_mcore2hf.sh
```

## 数据预处理

### 模板构造

- R1-Zero复现需要在数据处理时加上prompt模板激发`<think>...</think><answer>...$\boxed{}</answer>`

  ```
  <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nA conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>Put your final answer within \\boxed{}.\n{你真正的问题}<|im_end|>\n<|im_start|>assistant\n{模型真正的回答}
  ```

- 以上为默认的qwen_r1模板，根据模型和数据的不同，用户可以在`configs/model/templates.json`添加自己的**自定义模板**

### 选择数据集

对于7B模型应使用难度适中的数据集，所以我们使用Orz math 57K来训练

### 数据预处理

需要先配置数据处理的yaml文件(configs/datasets/r1_zero_qwen25_7b.yaml) 自定义数据集需要设置--map-keys映射，或重写自定义handler；具体参考[**数据集处理部分**](https://gitee.com/ascend/MindSpeed-RL/blob/master/docs/algorithms/grpo.md)

**Qwen2.5-7B**

- 处理的时候默认使用qwen_r1的模板

  ```
  # 启动转换
  bash examples/data/preprocess_data.sh r1_zero_qwen25_7b
  ```

## 打分器

DeepSeek-R1-Zero训练的过程中仅使用了基于程序的打分器而没有使用ORM，我们在数学领域上的打分逻辑分为以下几个部分：

![img](https://gitee.com/ascend/MindSpeed-RL/raw/master/sources/images/r1_zero/rule_reward.png)

## 配置文件

由于 GRPO 训练过程中涉及 3 个模型，通过将模型参数和训练配置解耦的层级化参数配置，来简化 GRPO 训练的参数配置过程。RLXF 训练涉及到的所有配置文件均存储在 configs/rlxf 路径下，其中 model 文件夹下存储了模型结构相关的配置文件，GRPO 训练相关的模型参数文件以 grpo_{模型名}.yaml方式命名。

在每个 grpo_trainer 配置文件中，需要包含 defaults、megatron_training、rl_config、generate_config等字段的参数配置以及 GRPO 训练过程中涉及到的 3 个角色 actor，reward，ref 的配置。

1. defaults 负责引入模型配置文件，在 defaults 中应列举本配置文件中所需要用到的所有模型配置，模型配置可以在下方3个角色的具体配置中通过 model 字段进行选择。
2. megatron_training 字段设置的参数为所有 3 个角色通用的默认参数，这些参数可以在下方进一步被角色的单独配置所覆盖。
3. actor_config、ref_config 以及 reward_config：三个角色的训练配置。
4. rl_config: 在 GRPO 训练中的特性参数，以及 actor，reward，ref 模型的资源配置。
5. generate_config: 包含 tokenizer 相关配置、推理并行配置、vllm 模型相关设置以及样本采样参数配置。

### 参数解析

相较于普通模型训练，GRPO 增加一些特殊参数，以下将给出部分参数的意义解析。具体的参数配置格式请参照示例[配置文件](https://gitee.com/ascend/MindSpeed-RL/blob/master/configs/grpo_trainer_qwen25_7b.yaml)。

### `defaults:`

引入模型配置(网络结构需要定义在model目录的yaml文件下)：

- `model`: qwen25_7b

### `megatron_training:`

- `stage`：用于指定训练算法，使用 Ray GRPO 训练须设置为`ray_grpo`；
- `global_batch_size`: 经过多少样本后 actor-train 和 rollout 权重同步；
- `data_path`: 数据集路径配置，例如 /dataset/data ；
- `tokenizer_name_or_path`: 分词器路径配置，可以配置为 Hugging Face 权重文件的文件夹路径，例如 /ckpt/qwen2.5_7b_hf/ ;
- `其余参数`: 其余参数为Megatron训练中的特性配置；

### `actor_config、ref_config 以及 reward_config：`

配置 GRPO 训练中 Actor 模型、Reference 模型和 Reward 模型的配置参数；当前支持不开启 Reward 模型，开启规则奖励进行打分，开启参数详见rl_config中的rule_reward参数。

- `tensor_model_parallel_size`：TP 并行策略数;
- `pipeline_model_parallel_size`：PP 并行策略数;
- `micro_batch_size`：mbs 数量;
- `lr`：学习率；
- `lr_decay_style`：学习率衰减配置；
- `min_lr`：最小学习率；
- `weight_decay`：权重衰减，用于防止模型过拟合；
- `lr_warmup_fraction`：学习率预热比例，在训练初期逐渐增大学习率的比例；
- `load`：模型加载的路径；
- `save`：模型保存的路径；
- `no_load_optim`：续训加载优化器状态；
- `no_load_rng`：续训加载数据随机数生成器；
- `no_save_optim`：保存优化器状态；
- `no_save_rng`：保存数据随机数生成器；

### `rl_config:`

- `blocking`：是否开启异步，默认为 False；

- `n_samples_per_prompt`：每条prompt的重用次数，一条 prompt 输入能输出 n 条 responese；

- `max_prompt_length`：GRPO 训练中最大 prompt 长度，默认为512；

- `clip_ratio`：Actor 模型训练计算损失函数时的 clip 比例，默认为0.2 一般取值范围 [0.1，0.3] 最大取值范围[0，1] 该数值越大允许策略更新的幅度越大，反之不然；

- `shuffle_mini_batch`：Actor 训练时是否对 minibatch 进行 shuffle，默认为 False；

- `actor_resource` ：分配给 Actor 模型的显卡数量；

- `reference_resource` ：分配给 Reference 模型的显卡数量；

- `reward_resource` ：分配给 Reward 模型的显卡数量；

  显卡资源配置格式为 :

  ```
  actor_resource:
      num_npus: 4
  ```

开启规则奖励开关后，不用分配资源给 reward_resource 参数，规则奖励参数配置如下：

- `rule_reward`: 开启后，使用规则奖励进行打分；
- `verifier_function`: 选择使用的规则奖励模型方法，例如["acc", "strict_format"] ；
- `verifier_weight`: 配置规则奖励模型权重，例如[1.0, 1.0]；

日志配置参数也在 rl_config 中进行配置，当前支持 wandb/tensorboard 日志输出：

tensorboard开关（若use_tensorboard和use_wandb同时为True，则tensorboard不生效）:

- `use_tensorboard`: 配置为 True 时打开 tensorboard；

wandb开关:

- `use_wandb`: 配置为 True 时打开 wandb；
- `wandb_project`: project 名称配置；
- `wandb_exp_name`: 实验名称配置；
- `wandb_save_dir`: 本地存储 wandb 路径；

### `generate_config:`

#### tokenizer相关配置

- `micro_batch_size`：mbs 大小，推理时每次处理的样本数量；

#### 推理时的并行配置

- `infer_tensor_parallel_size`：TP并行策略数；
- `infer_pipeline_parallel_size`：PP并行策略数；
- `infer_expert_parallel_size`：EP并行策略数；

#### resharding 相关配置

- `offload_train_optimizer`：卸载训练节点优化器；
- `offload_train_grad`：卸载训练节点梯度；
- `offload_train_param`：卸载模型权重；

#### vllm 模型相关设置

vllm 模型参数 可以参照 [vllm官网参数介绍](https://gitee.com/link?target=https%3A%2F%2Fdocs.vllm.ai%2Fen%2Flatest%2Fserving%2Fengine_args.html)：

- `max_num_seqs`：vllm 推理并发最大样本限制；
- `max_num_batched_tokens`：vllm 推理并发最大token限制；
- `gpu_memory_utilization`：GPU 内存利用率，指定推理时使用 GPU 内存的比例；

#### 采样配置

- `logprobs`：是否生成logprobs；
- `max_tokens`：单条response最大生成token数量；
- `temperature`：采样时的随机性参数；
- `detokenize`：是否将输出token重新转为文本；



## GRPO 训练

### 背景

传统的PPO中需要一个通过广义优势估计（Generalized Advantage Estimation）计算得到的advantage，并依赖于和reward model同结构的需要同步训练的critic model计算得到价值函数(V)

GRPO通过分组采样n个输出，利用组内的平均奖励作为基线计算每个输出在组内的相对奖励，并基于相对奖励计算优势值，从而避免了引入额外的价值网络（critic model）

![img](https://gitee.com/ascend/MindSpeed-RL/raw/master/sources/images/r1_zero/grpo.png)

DeepSeek-R1-Zero的训练过程使用GRPO算法，将ORM（结果奖励模型）替换为基于规则的打分器。

###  配置准备

模型结构的配置文件位于configs/model下，训练配置文件位于configs/目录下，我们以qwen2.5-7b为例[r1_zero_qwen25_7b.yaml]，该配置用到了16卡，为了进一步加速可以不断增加推理DP的数量。以下为参数配置：

```shell
defaults:
  - model:
      - qwen25-7b                        <-- 网络结构需要定义在model目录的yaml文件下

megatron_training:
  global_batch_size: 64                   <-- 经过多少样本后actor-train和rollout权重同步
  ...
  dataset_additional_keys: ['labels',]    <-- 使用打分器时需要的额外字段

actor_config:
  model: qwen25-7b
  micro_batch_size: 1          <-- 训练的mbs
  ...
  lr: 5e-7
  lr_decay_style: cosine     <-- 学习率衰减方式
  min_lr: 5e-8
  weight_decay: 0.0            <-- 正则化强度系数
  lr_warmup_fraction: 0.0      <-- 控制学习率预热
  ...
  no_load_optim: false         <-- 续训加载优化器状态
  no_load_rng: false           <-- 续训加载数据随机数生成器
  no_save_optim: false         <-- 保存权重时同时保存优化器状态
  no_save_rng: false           <-- 保存权重时同时保存数据随机数生成器

ref_config:
  model: qwen25-7b
  ...

reward_config:
  model: qwen25_7b
  ...

rl_config:
  blocking: false              <-- 开启异步流水
  ...
  adv_estimator: group_norm    <-- 优势计算方法
  mini_batch_size: 512         <-- 训练更新梯度的bs, 一般为gbs*n_samples_per_prompt
  ...
  max_prompt_length: 1024      <-- 最大的prompt长度
  clip_ratio: 0.2              <-- 策略裁剪比例
  shuffle_minibatch: false     <-- minibatch里的数据是否打乱
  n_samples_per_prompt: 8      <-- GRPO中一个group内生成的response条数
  colocate_actor_ref: false
  colocate_all_models: false
  rule_reward: true                              <-- 开启规则奖励
  verifier_function: ["base_acc"]                <-- 规则奖励模型方法
  verifier_weight: [1.0]                         <-- 规则奖励模型权重
  use_tensorboard: true                          <-- 开启tensorboard日志功能
  actor_resource:                                <-- actor worker资源分配
    num_npus: 8
  reference_resource:                            <-- ref worker资源分配
    num_npus: 8

generate_config:
  trust_remote_code: true            <-- tokenizer相关配置

  infer_tensor_parallel_size: 2      <-- 推理时的并行配置
  infer_pipeline_parallel_size: 1
  infer_expert_parallel_size: 1

  max_num_seqs: 128                  <-- vllm 推理并发最大样本限制
  max_num_batched_tokens: 128000     <-- vllm 推理并发最大token限制
  max_model_len: 4096
  dtype: "bfloat16"
  gpu_memory_utilization: 0.9
  offload_train_optimizer: true      <-- 卸载训练节点优化器
  offload_train_grad: true           <-- 卸载训练节点梯度
  offload_train_param: true          <-- 卸载模型权重

  sampling_config:                   <-- vllm 采样配置
    max_tokens: 2048                 <-- 单条response最大生成token数量
    logprobs: 1                      <-- 是否生成logprobs
    top_p: 0.9
    top_k: 50
    min_p: 0.01
    temperature: 0.8
    detokenize: false
  ...
```

### 训练启动方式

#### 单机

通过 --config-name 传递选取的 config 文件名（不添加.yaml后缀），可以通过下列命令直接启动训练（Qwen25 7B 模型可单机运行）。 目前已支持的配置文件放置在 configs/ 文件夹下。配置文件的具体说明见下文。

以 Qwen25 7B 模型为例,单机启动命令示例如下：

```shell
bash examples/grpo/grpo_trainer_qwen25_7b.sh
```

### 多机

以[ DeepSeekR1-ZERO-Qwen2.5-32B 复现](https://gitee.com/ascend/MindSpeed-RL/blob/master/docs/solutions/r1_zero_qwen25_32b.md) 为例，多机启动步骤如下：

#### 手动启动训练

与基于ray的其他强化训练一样，我们多机需要先在主节点初始化ray：

```shell
# 创建一个集群，端口6344，dashboard端口8260
ray start --head --port 6344 --dashboard-host=0.0.0.0 --dashboard-port=8260
```

随后，在其他节点加入主节点的集群：

```shell
# IP_ADDRESS 处填写主节点 IP 地址
ray start --address="IP_ADDRESS:6344"
```

最后，在主节点上启动训练：

```shell
export HCCL_CONNECT_TIMEOUT=1800
export CUDA_DEVICE_MAX_CONNECTIONS=1

python cli/train_grpo.py --config-name r1_zero_qwen25_7b.yaml | tee logs/r1_zero_qwen25_7b_full.log
```

#### 脚本启动训练

```shell
# 主节点
bash examples/r1/qwen25/r1_zero_qwen25_7b_master.sh r1_zero_qwen25_7b.yaml
# 其余子节点
bash examples/r1/qwen25/r1_zero_qwen25_7b_worker.sh
```
