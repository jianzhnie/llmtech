# OpenRLHF x Ascend

OpenRLHF

- https://github.com/OpenRLHF/OpenRLHF.git

OpenRLHF Ascend :

- https://github.com/zhuo97/OpenRLHF.git

我们在 OpenRLHF 上增加对华为昇腾设备的支持。**本代码仓由社区进行维护更新，不进行任何商业交付。**

## 安装

### 配套版本

**下表提供昇腾配套软件版本仅为建议，不作为任何商业交付承诺**。

<table>
  <tr>
    <th align="left">OpenRLHF 主仓Tag</th>
    <td>v0.6.2</td>
  </tr>
  <tr>
    <th align="left">对应的 OpenRLHF NPU 适配分支</th>
    <td>main</td>
  </tr>
  <tr>
    <th align="left">vLLM 版本</th>
    <td>v0.7.3</td>
  </tr>
  <tr>
    <th align="left">vLLM Ascend 版本/分支</th>
    <td>v0.7.3</td>
  </tr>
  <tr>
    <th align="left">torch npu 版本 (pip install 安装)</th>
    <td>2.5.1</td>
  </tr>
  <tr>
    <th align="left">CANN 版本 (参考vllm-ascend)</th>
    <td><a href="https://github.com/vllm-project/vllm-ascend/blob/v0.7.3/docs/source/installation.md?plain=1#L72-L96">CANN 8.1.RC1</a></td>
  </tr>
  <tr>
    <th align="left">不支持功能</th>
    <td>Ring Attention</br>Hybrid Engine</br>Pytorch Compile</br>bitsandbytes</td>
  </tr>
</table>

### vLLM

为了保证能够在 OpenRLHF 上正常使用 vLLM，需要安装 vLLM Ascend 插件（`vllm-ascend`）。vLLM Ascend 插件的安装方式和镜像请参考[安装教程](https://vllm-ascend.readthedocs.io/en/latest/installation.html)。

```shell
git clone -b v0.7.3 https://github.com/vllm-project/vllm.git
cd vllm
pip install -r requirements-build.txt
VLLM_TARGET_DEVICE=empty pip install .

git clone -b v0.7.3 https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
pip install -e .
```

### 源码安装

```shell
git clone https://github.com/zhuo97/OpenRLHF.git
cd OpenRLHF
TARGET_DEVICE=NPU pip install -e .
```

### Transformers

当前在 Ascend NPU 进行使用时，可能存在 CPU Memory 不足的情况。该问题已经修复（[PR](https://github.com/huggingface/transformers/pull/37698)），但是 transformers `4.51.4` 版本尚未发布。因此，如果在使用过程中遇到该问题，可以源码编译安装 transformers 库。当 transformers `4.51.4` 版本发布后，我们会第一时间更新 Ascend NPU 的 requirements。

### Ray

可通过如下方式在华为昇腾设备上启动 Ray:
```shell
# launch the master node of ray in container
ray start --head --port 6379

# if you want to launch ray on more nodes, use
ray start --address='MASTER-NODE-ADDRESS:6379'
```

训练脚本提交方式与英伟达 GPU 相同。

### 其他第三方库说明

| 软件            | 说明                                                                                                                     |
| --------------- | ------------------------------------------------------------------------------------------------------------------------ |
| flash_attn      | 原生不支持，通过在 transformers 适配昇腾FA算子进行支持（[PR](https://github.com/huggingface/transformers/pull/36696)）。 |
| ring_flash_attn | 原生不支持。                                                                                                             |
| bitsandbytes    | 原生不支持。                                                                                                             |

## 支持的算法

### 精度对比

根据经验，我们期望在相同配置下，在华为昇腾设备上的 Loss 与英伟达 GPU 的 Loss/Reward 平均绝对误差小于 2%，具体计算方式如下：

$$
Mean Error=\frac{\sum^N_{i=1}|loss_i^{npu}-loss_i^{gpu}|}{N}\times 100 \%
$$
其中，N 表示训练的步数。更多信息请参考[精度计算说明](https://www.hiascend.com/document/detail/zh/Pytorch/600/ptmoddevg/trainingmigrguide/LMaccuracy_0001.html)。

### 进展

已支持的算法仅在下表提供的版本进行过测试。

| 算法        | 进展       | 与GPU误差 | torch 版本 | torch_npu 版本 | CANN 版本 | 详细结果                                                                          |
| ----------- | ---------- | --------- | ---------- | -------------- | --------- | --------------------------------------------------------------------------------- |
| SFT         | 已支持     | 0.19%     | 2.3.1      | 2.3.1.post2    | 8.0.RC3   | [测试结果](https://github.com/OpenRLHF/OpenRLHF/pull/605#issuecomment-2567488539) |
| DPO         | 已支持     | 1.81%     | 2.3.1      | 2.3.1.post2    | 8.0.RC3   | [测试结果](https://github.com/OpenRLHF/OpenRLHF/pull/605#issuecomment-2735122006) |
| KTO         | 已支持     | 0.37%     | 2.3.1      | 2.3.1.post2    | 8.0.RC3   | [测试结果](https://github.com/OpenRLHF/OpenRLHF/pull/605#issuecomment-2642104300) |
| RM          | 已支持     | 0.85%     | 2.3.1      | 2.3.1.post2    | 8.0.RC3   | [测试结果](https://github.com/OpenRLHF/OpenRLHF/pull/605#issuecomment-2642104300) |
| PRM         | 已支持     | 1.61%     | 2.3.1      | 2.3.1.post2    | 8.0.RC3   | [测试结果](https://github.com/OpenRLHF/OpenRLHF/pull/605#issuecomment-2642104300) |
| PPO         | 精度测试中 |           | 2.5.1      | 2.5.1          | 8.1.RC1   |                                                                                   |
| REINFORCE++ | 已支持     | 1.94%     | 2.5.1      | 2.5.1          | 8.1.RC1   | [测试结果](https://github.com/OpenRLHF/OpenRLHF/pull/605#issuecomment-2735138695) |
| GRPO        | 已支持     | 0.61%     | 2.5.1      | 2.5.1          | 8.1.RC1   | [测试结果](https://github.com/OpenRLHF/OpenRLHF/pull/605#issuecomment-2764993841) |

### 常见问题

* 使用 `--adam_offload` 参数可能存在长时间卡顿的情况，解决方法是删除 torch_extensions 的缓存文件，参考 [issue](https://github.com/deepspeedai/DeepSpeed/issues/2816#issuecomment-1450095538)。

##  算法使用指南

### SFT, DPO, RM, PRM

    We have tested SFT, DPO, RM, PRM algorithms on Ascend NPU currently.

Due to differences in hardware structure, we cannot guarantee that the loss of Ascend NPU is exactly the same as that of the GPU. According to our experience, the loss differences less than 2% is acceptable. If the loss difference is greater than 2%, we will try to fix it. The calculation formula is as follows.

![loss_comparison](https://github.com/user-attachments/assets/c20b833b-2ad3-4d63-b241-001f88955222)

N represents the number of training steps.
Here are our testing results.

| Algorithm | Model                            | Steps | Mean Error | Progress |
| --------- | -------------------------------- | ----- | :--------- | -------- |
| SFT       | meta-llama/Meta-Llama-3-8B       | 30941 | 0.19%      | 100%     |
| DPO       | meta-llama/Llama-3.2-1B-Instruct | 3125  | 1.81%      | ~10%     |
| RM        | meta-llama/Llama-3.2-1B-Instruct | 3125  | 1.47%      | ~10%     |
| PRM       | mistralai/Mistral-7B-v0.1        | 3125  | 0.25%      | ~10%     |

For SFT, We use the default configuration for training. <u>We have modified the default configuration of DPO, RM and PRM algorithms for quick verification. We will use the default configuration for verification later and update the results in this PR.</u>

We use this PR testing on Ascend NPU and GPU to ensure the same codes can run on different devices. The device information is 8 Atlas 800T A2 and 8 A100.  Other software information is shown in the following table.

| Software     | Version     |
| ------------ | ----------- |
| transformers | 4.46.3      |
| deepspeed    | 0.15.0      |
| accelerate   | 1.2.1       |
| CANN         | 8.0.RC3     |
| torch_npu    | 2.3.1.post2 |

Here are training scripts and loss comparison graphs.

**For SFT:**

Parameters change information:

* remove flash_attn

```shell
set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
   --max_len 2048 \
   --dataset Open-Orca/OpenOrca \
   --input_key question \
   --output_key response \
   --train_batch_size 256 \
   --micro_train_batch_size 2 \
   --max_samples 500000 \
   --pretrain meta-llama/Meta-Llama-3-8B \
   --save_path ./checkpoint/llama3-8b-sft \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 2 \
   --max_epochs 1 \
   --bf16 \
   --learning_rate 5e-6 \
   --load_checkpoint \
   --gradient_checkpointing
EOF
    # --wandb [WANDB_TOKENS]
    # --packing_samples

if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi
```

![sft](https://github.com/user-attachments/assets/c7f8f43a-30b6-49da-b469-4f20e1a19d70)

**For DPO:**

Parameters change information:

* add max_samples 50000
* train_batch_size 256 -> train_batch_size 32
* micro_train_batch_size 1 -> micro_train_batch_size 2
* OpenRLHF/Llama-3-8b-sft-mixture -> meta-llama/Llama-3.2-1B-Instruct
* learning_rate 5e-7 -> learning_rate 1e-7
* remove flash_attn

```shell
set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_dpo \
   --save_path ./checkpoint/llama3.2-1b-dpo \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --max_samples 50000 \
   --train_batch_size 32 \
   --micro_train_batch_size 2 \
   --pretrain meta-llama/Llama-3.2-1B-Instruct \
   --bf16 \
   --max_epochs 1 \
   --max_len 1024 \
   --zero_stage 3 \
   --learning_rate 1e-7 \
   --beta 0.1 \
   --dataset OpenRLHF/preference_dataset_mixture2_and_safe_pku \
   --apply_chat_template \
   --chosen_key chosen \
   --rejected_key rejected \
   --load_checkpoint \
   --gradient_checkpointing
EOF
    # --use_wandb [WANDB_TOKENS] or True (use wandb login command)
    # --ipo [for IPO]
    # --label_smoothing 0.1 [for cDPO]
    # --ref_offload
    # --packing_samples
    # --nll_loss_coef (Regularization with NLL loss)


if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi
```

![dpo](https://github.com/user-attachments/assets/ead6c20e-a06d-4dd6-ab7b-9a7dea61c016)

**For RM:**

Parameters change information:

* add max_samples 50000
* train_batch_size 256 -> train_batch_size 16
* micro_train_batch_size 1 -> micro_train_batch_size 2
* OpenRLHF/Llama-3-8b-sft-mixture -> meta-llama/Llama-3.2-1B-Instruct
* max_len 8192 -> max_len 1024
* learning_rate 9e-6 -> learning_rate 2e-7
* remove flash_attn

```shell
set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_rm \
   --save_path ./checkpoint/llama3.2-1b-rm \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --max_samples 50000 \
   --train_batch_size 16 \
   --micro_train_batch_size 2 \
   --pretrain meta-llama/Llama-3.2-1B-Instruct \
   --bf16 \
   --max_epochs 1 \
   --max_len 1024 \
   --zero_stage 3 \
   --learning_rate 2e-7 \
   --dataset OpenRLHF/preference_dataset_mixture2_and_safe_pku \
   --apply_chat_template \
   --chosen_key chosen \
   --rejected_key rejected \
   --load_checkpoint \
   --gradient_checkpointing
EOF
     # --use_wandb [WANDB_TOKENS] or True (use wandb login command)
     # --packing_samples


if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi
```

![rm](https://github.com/user-attachments/assets/a3c435bb-8d44-4e74-83a8-6e5f85982b17)


**For PRM:**

Parameters change information:

* add max_samples 50000
* train_batch_size 256 -> train_batch_size 32
* micro_train_batch_size 8 -> micro_train_batch_size 2
* learning_rate 1e-6 -> learning_rate 2e-7
* remove flash_attn

```shell
set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_prm \
   --save_path ./checkpoint/mistal-7b-prm \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --max_samples 50000 \
   --train_batch_size 32 \
   --micro_train_batch_size 2 \
   --pretrain mistralai/Mistral-7B-v0.1  \
   --bf16 \
   --max_epochs 1 \
   --max_len 8192 \
   --zero_stage 3 \
   --learning_rate 2e-7 \
   --dataset zhuzilin/Math-Shepherd \
   --input_key input \
   --label_key value \
   --load_checkpoint \
   --gradient_checkpointing \
   --packing_samples \
   --wandb_group prm \
   --placeholder_token ки \
   --reward_tokens + -
EOF
     # --use_wandb [WANDB_TOKENS] or True (use wandb login command)
     # --packing_samples


if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi
```

![prm](https://github.com/user-attachments/assets/d2f42e95-91ab-450a-9d84-7037e96d627c)

According to openRLHF environment requirements, transformers, deepspeed, aceelerate and peft have supported Ascend NPU, users can install and use them directly on Ascend NPU.

For vLLM, we are waiting this [PR](https://github.com/vllm-project/vllm/pull/8054) to be merged.

For Ray, it has supported Ascend NPU according to this [PR](https://github.com/ray-project/ray/pull/41256), we will test openRLHF based on Ray in the future.

For SGLang, as far as we know, it has no plans to support Ascend NPU currently.

As shown above, there are still many shortcomings in the current Ascend NPU ecosystem. We are trying our best to improve it to ensure that users can get the same training experience as GPU.

The current tests are limited, we will also test other RLHF algorithms on Ascend NPU in the furture.

If this [PR](https://github.com/vllm-project/vllm/pull/8054) is merged by vLLM, we will test on-policy algorithms on Ascend NPU. If you have any other questions, please feel free to ask us.

### DPO 算法

| Algorithm | Model                           | Mean Error | Progress |
| --------- | ------------------------------- | ---------- | -------- |
| DPO       | OpenRLHF/Llama-3-8b-sft-mixture | 1.17%      | 100%     |

**Software information**:
| Software  | Version     |
| --------- | ----------- |
| CANN      | 8.0.RC3     |
| torch_npu | 2.3.1.post2 |

**Device information**: 8 Atlas 800T and 8 A100
**Parameters change information:**
* learning_rate 5e-7 - > 5e-8
* add `adam_offload` (Avoiding OOM on NPU)
* remove `load_checkpoint`
```shell
set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_dpo \
   --save_path ./checkpoint/llama3-8b-dpo \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 256 \
   --micro_train_batch_size 1 \
   --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
   --bf16 \
   --max_epochs 1 \
   --max_len 8192 \
   --zero_stage 3 \
   --learning_rate 5e-8 \
   --beta 0.1 \
   --dataset OpenRLHF/preference_dataset_mixture2_and_safe_pku \
   --apply_chat_template \
   --chosen_key chosen \
   --rejected_key rejected \
   --flash_attn \
   --adam_offload \
   --gradient_checkpointing
EOF
    # --use_wandb [WANDB_TOKENS] or True (use wandb login command)
    # --ipo [for IPO]
    # --label_smoothing 0.1 [for cDPO]
    # --ref_offload
    # --packing_samples
    # --nll_loss_coef (Regularization with NLL loss)
```
![dpo_loss_cmp](https://github.com/user-attachments/assets/0cdf522b-6ebf-427d-a3ee-bd8e39266562)

### PRM, KTO, RM

| Algorithm | Model                           | Steps | Mean Error | Progress |
| --------- | ------------------------------- | ----- | :--------- | -------- |
| PRM       | mistralai/Mistral-7B-v0.1       | 6947  | 1.61%      | 100%     |
| KTO       | OpenRLHF/Llama-3-8b-sft-mixture | 15904 | 0.37%      | 100%     |
| RM        | OpenRLHF/Llama-3-8b-sft-mixture | 69362 | 0.85%      | 100%     |


I use this PR testing on Ascend NPU and GPU to ensure the same codes can run on different devices. The device information is 8 Atlas 800T A2 and 8 A100.  Other software information is shown in the following table.

| Software     | Version     |
| ------------ | ----------- |
| transformers | 4.46.3      |
| deepspeed    | 0.15.0      |
| accelerate   | 1.2.1       |
| CANN         | 8.0.RC3     |
| torch_npu    | 2.3.1.post2 |

I have modified the logging method so that logs are printed only at integer multiples of the gradient accumulation steps. Taking KTO as an example, the modifications are as follows:

```diff
--- a/openrlhf/trainer/kto_trainer.py
+++ b/openrlhf/trainer/kto_trainer.py
@@ -168,7 +168,7 @@ class KTOTrainer(ABC):
                 }
                 logs_dict["kl"] = KL.item()
                 logs_dict = self.strategy.all_reduce(logs_dict)
-                step_bar.set_postfix(logs_dict)
+                # step_bar.set_postfix(logs_dict)
                 step_bar.update()

                 # logs/checkpoints/evaluation
@@ -191,6 +191,10 @@ class KTOTrainer(ABC):
     def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}):
         # logs
         if global_step % args.logging_steps == 0:
+            logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
+            if self.strategy.is_rank_0():
+                step_bar.write(str(logs))
+
             # wandb
             if self._wandb is not None and self.strategy.is_rank_0():
                 logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}

```

Here are training scripts and loss comparison graphs.

**For PRM:**

Parameters change information:

* remove flash_attn

```shell
set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_prm \
   --save_path ./checkpoint/mistal-7b-prm \
   --save_steps 500 \
   --logging_steps 1 \
   --eval_steps 100 \
   --train_batch_size 256 \
   --micro_train_batch_size 8 \
   --pretrain mistralai/Mistral-7B-v0.1  \
   --bf16 \
   --max_epochs 1 \
   --max_len 8192 \
   --zero_stage 3 \
   --learning_rate 1e-6 \
   --dataset zhuzilin/Math-Shepherd \
   --input_key input \
   --label_key value \
   --load_checkpoint \
   --gradient_checkpointing \
   --packing_samples \
   --wandb_group prm \
   --placeholder_token ки \
   --reward_tokens + -
EOF
     # --use_wandb [WANDB_TOKENS] or True (use wandb login command)
     # --packing_samples


if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi

```

![prm](https://github.com/user-attachments/assets/e64f92ce-0793-4587-bb2d-4154e764a9e3)

**For KTO:**

Parameters change information:

* learning_rate 5e-7 -> learning_rate 2e-7
* remove flash_attn
* remove max_samples

```shell
set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_kto \
   --save_path ./checkpoint/llama3-8b-kto \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 64 \
   --micro_train_batch_size 1 \
   --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
   --bf16 \
   --max_epochs 1 \
   --max_len 8192 \
   --zero_stage 3 \
   --learning_rate 2e-7 \
   --dataset Dylan2048/ultrafeedback-unpaired-preferences \
   --input_key instruction \
   --output_key response \
   --label_key score \
   --beta 0.1 \
   --gradient_checkpointing
EOF
    # --use_wandb [WANDB_TOKENS] or True (use wandb login command)


if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi

```

![kto](https://github.com/user-attachments/assets/b5fc69c1-8771-43a3-8933-36089ad3778f)

**For RM:**

Parameters change information:

* remove flash_attn
* learning_rate 9e-6 -> learning_rate 1e-8

```shell
set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_rm \
   --save_path ./checkpoint/llama3-8b-rm \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 256 \
   --micro_train_batch_size 1 \
   --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
   --bf16 \
   --max_epochs 1 \
   --max_len 8192 \
   --zero_stage 3 \
   --learning_rate 1e-8 \
   --dataset OpenRLHF/preference_dataset_mixture2_and_safe_pku \
   --apply_chat_template \
   --chosen_key chosen \
   --rejected_key rejected \
   --load_checkpoint \
   --gradient_checkpointing
EOF
     # --use_wandb [WANDB_TOKENS] or True (use wandb login command)
     # --packing_samples


if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi

```

![rm_1e-8](https://github.com/user-attachments/assets/2c10de7f-93a0-4793-b698-154574d0e29a)

​

###  GRPO 算法使用指南

| Algorithm | Model                                                             | Reward Mean Error | Progress |
| --------- | ----------------------------------------------------------------- | ----------------- | -------- |
| GRPO      | OpenRLHF/Llama-3-8b-sft-mixture </br> OpenRLHF/Llama-3-8b-rm-700k | 0.61%             | 100%     |

**Software information**:

| Software    | Version                      |
| ----------- | ---------------------------- |
| CANN        | 8.1.RC1 (**Not Released**)   |
| torch_npu   | 2.5.1.RC1 (**Not Released**) |
| vllm        | 0.7.3                        |
| vllm-ascend | 0.7.3-dev                    |

The above software version has not been officially released yet.

**Device information**: 16 Atlas 800T A2 and 16 A100
**Parameters change information:**
* ref_num_gpus_per_node 1 -> 2
* reward_num_gpus_per_node 1 -> 2
* actor_num_gpus_per_node 4 -> 8
* vllm_num_engines 2 -> 4
* remove `packing_samples`
* remove `adam_offload`
* micro_train_batch_size 16 -> 2
* micro_rollout_batch_size  32 -> 8
* n_samples_per_prompt 1 -> 4
* actor_learning_rate 5e-7 -> 5e-8

```shell
set -x

# reinforce++

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/openrlhf"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 2 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 2 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 8 \
   --vllm_num_engines 4 \
   --vllm_tensor_parallel_size 1 \
   --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
   --reward_pretrain OpenRLHF/Llama-3-8b-rm-700k \
   --save_path /openrlhf/examples/test_scripts/checkpoint/llama3-8b-rlhf \
   --micro_train_batch_size 2 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 8 \
   --rollout_batch_size 1024 \
   --n_samples_per_prompt 4 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --max_samples 100000 \
   --generate_max_len 1024 \
   --init_kl_coef 1e-3 \
   --gamma 1.0 \
   --use_kl_loss \
   --kl_estimator k3 \
   --advantage_estimator group_norm \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-8 \
   --prompt_data OpenRLHF/prompt-collection-v0.1 \
   --input_key context_messages \
   --apply_chat_template \
   --normalize_reward \
   --gradient_checkpointing \
   --save_steps -1 \
   --ckpt_path /openrlhf/examples/test_scripts/ckpt/llama3-8b-rlhf

# You could also try
#   --kl_estimator k2 \

# also supports --advantage_estimator rloo | reinforce_baseline
```

![grpo_results](https://github.com/user-attachments/assets/7fa06f64-69ab-400d-9c1c-b2a91f7afb23)

### Reinforce++ 算法使用指南

| Algorithm   | Model                                                             | Reward Mean Error | Progress |
| ----------- | ----------------------------------------------------------------- | ----------------- | -------- |
| Reinforce++ | OpenRLHF/Llama-3-8b-sft-mixture </br> OpenRLHF/Llama-3-8b-rm-700k | 1.94%             | 100%     |

**Software information**:
| Software    | Version                      |
| ----------- | ---------------------------- |
| CANN        | 8.1.RC1 (**Not Released**)   |
| torch_npu   | 2.5.1.RC1 (**Not Released**) |
| vllm        | 0.7.3                        |
| vllm-ascend | 0.7.3-dev                    |

The above software version has not been officially released yet.

**Device information**: 8 Atlas 800T A2 and 8 A100
**Parameters change information:**
* remove `packing_samples`
* micro_train_batch_size 16 -> 2
* micro_rollout_batch_size  32 -> 8

```shell
set -x

# reinforce++

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/openrlhf"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 1 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 4 \
   --vllm_num_engines 2 \
   --vllm_tensor_parallel_size 1 \
   --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
   --reward_pretrain OpenRLHF/Llama-3-8b-rm-700k \
   --save_path /openrlhf/examples/test_scripts/checkpoint/llama3-8b-rlhf \
   --micro_train_batch_size 2 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 8 \
   --rollout_batch_size 1024 \
   --n_samples_per_prompt 1 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --max_samples 100000 \
   --generate_max_len 1024 \
   --advantage_estimator reinforce \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --init_kl_coef 1e-4 \
   --prompt_data OpenRLHF/prompt-collection-v0.1 \
   --input_key context_messages \
   --apply_chat_template \
   --normalize_reward \
   --adam_offload \
   --gradient_checkpointing \
   --save_steps -1 \
   --ckpt_path /openrlhf/examples/test_scripts/ckpt/llama3-8b-rlhf

# You could also try
#   --use_kl_loss \
#   --kl_estimator k3 | k2 \

# also supports --advantage_estimator rloo | reinforce_baseline

```

![reinforce_results](https://github.com/user-attachments/assets/bfbce512-4885-4d40-b194-978aee92c04d)

### Reinforce++ (packing_sample) 算法使用指南

Thanks to @FightingZhen for his [PR](https://github.com/huggingface/transformers/pull/36696) contribution, we now support training with `--packing_samples` parameter on Ascend NPU. Here we use the Reinforce++ algorithm for verification. **<ins>Note that when using `--packing_samples`, `transformer` version must be greater than or equal to `4.51.0`.</ins>**

**Results**:

| Algorithm   | Model                                                             | Reward Mean Error | Progress |
| ----------- | ----------------------------------------------------------------- | ----------------- | -------- |
| Reinforce++ | OpenRLHF/Llama-3-8b-sft-mixture </br> OpenRLHF/Llama-3-8b-rm-700k | 2.35%             | 100%     |

**Software information**:
| Software    | Version                     |
| ----------- | --------------------------- |
| CANN        | 8.1.RC1 (**coming soon**)   |
| torch_npu   | 2.5.1.RC1 (**coming soon**) |
| vllm        | 0.7.3                       |
| vllm-ascend | 0.7.3-dev                   |

The above software version has not been officially released yet.

**Device information**: 8 Atlas 800T A2 and 8 A100
**Parameters change information:**
* actor_learning_rate 5e-7 -> 3e-7
* micro_train_batch_size 16 -> 2
* micro_rollout_batch_size  32 -> 8

```shell
set -x

# reinforce++

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/openrlhf"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 1 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 4 \
   --vllm_num_engines 2 \
   --vllm_tensor_parallel_size 1 \
   --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
   --reward_pretrain OpenRLHF/Llama-3-8b-rm-700k \
   --save_path /openrlhf/examples/test_scripts/checkpoint/llama3-8b-rlhf \
   --micro_train_batch_size 2 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 8 \
   --rollout_batch_size 1024 \
   --n_samples_per_prompt 1 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --max_samples 100000 \
   --generate_max_len 1024 \
   --advantage_estimator reinforce \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 3e-7 \
   --init_kl_coef 1e-4 \
   --prompt_data OpenRLHF/prompt-collection-v0.1 \
   --input_key context_messages \
   --apply_chat_template \
   --normalize_reward \
   --adam_offload \
   --gradient_checkpointing \
   --packing_samples \
   --save_steps -1 \
   --ckpt_path /openrlhf/examples/test_scripts/ckpt/llama3-8b-rlhf

# You could also try
#   --use_kl_loss \
#   --kl_estimator k3 | k2 \

# also supports --advantage_estimator rloo | reinforce_baseline

```

![reinforce++_with_packing_samples](https://github.com/user-attachments/assets/d150ceb7-d4c1-424c-a9c8-197c8a753c7b)
