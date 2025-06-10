We have tested SFT, GRPO algorithms on Ascend NPU currently.

Due to differences in hardware structure, we cannot guarantee that the loss of Ascend NPU is exactly the same as that of the GPU. According to our experience, **the loss differences less than 2% is acceptable. If the loss difference is greater than 2%, we will try to fix it,  the critic/rewards/mean differences less than 4% is acceptable. If the critic/rewards/mean difference is greater than 4%, we will try to fix it.** The calculation formula is as follows.

![loss_comparison](https://github.com/user-attachments/assets/99817bb2-43b1-4675-9831-79c8de6831f2)


N represents the number of training steps.

| Software     |                Version |
| :----------- | ---------------------: |
| transformers |                 4.49.0 |
| torch_npu    |              2.5.1.rc1 |
| CANN         | 8.1.RC1 (Not Released) |

Here are training scripts and loss comparison graphs.

For SFT:

```shell
# Tested with 1 & 8 NPUs

set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: run_qwen_05_peft.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2

# Shift the arguments so $@ refers to the rest
shift 2

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.train_batch_size=512 \
    optim.lr=1e-4 \
    +data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    data.micro_batch_size_per_gpu=4 \
    model.partial_pretrain=Qwen/Qwen2.5-0.5B-Instruct \
    trainer.default_local_dir=$save_path \
    trainer.project_name=gsm8k-sft \
    trainer.experiment_name=gsm8k-sft-qwen-2.5-0.5b-instruct \
    trainer.logger=['console'] \
    trainer.total_epochs=1 \
    trainer.default_hdfs_dir=null $@ \
    model.lora_rank=32\
    model.lora_alpha=16 \
    model.target_modules=all-linear

    # Or you can do this:
    # model.target_modules=[q_proj,v_proj] \
```
![sft](https://github.com/user-attachments/assets/f369a939-4108-4cbc-941d-c69515db567a)

For GRPO:

Parameters change information:

- data.train_batch_size 1024 -> 16
- actor_rollout_ref.actor.optim.lr 1e-6 -> 5e-7
- critic.optim.lr 1e-5 -> 9e-6
- actor_rollout_ref.actor.ppo_max_token_len_per_gpu 16384 -> 2048
- actor_rollout_ref.model.use_remove_padding True -> False
- actor_rollout_ref.actor.ppo_mini_batch_size 256 -> 64
- actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu 80 -> 8
- actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu 160 -> 80
- actor_rollout_ref.rollout.tensor_model_parallel_size 2 -> 4
- actor_rollout_ref.rollout.gpu_memory_utilization 0.6 -> 0.2
- actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu 160 -> 80
- actor_rollout_ref.rollout.enable_chunked_prefill True -> False
- trainer.nnodes 1 -> 2

```shell
# Tested with 2 & 8 NPUs
set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=16 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen2-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    critic.optim.lr=9e-6 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=2048 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=80 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.2 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=80 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_grpo_example_gsm8k' \
    trainer.experiment_name='qwen2_7b_function_rm' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=2\
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 $@
```
critic/rewards/mean:

<img src="https://github.com/user-attachments/assets/2cbf5bb6-80c6-49f5-94d4-062482cf1fc4" alt="grpo" style="zoom:50%;" />
