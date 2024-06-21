# LeRobot，一种用于机器人的开源机器学习模型

[Hugging Face](https://huggingface.co/) 推出了[LeRobot](https://huggingface.co/lerobot)，这是一种针对现实世界的机器人应用进行训练的新型机器学习模型库。LeRobot 作为一个平台，为数据共享、可视化和高级模型训练提供了一个多功能库。

[LeRobot 旨在通过PyTorch](https://pytorch.org/)为现实世界的机器人技术提供模型、数据episode和工具。目标是降低机器人技术的准入门槛，以便每个人都能从共享数据episode和预训练模型中做出贡献并受益。

LeRobot 通过提供预训练模型和与物理模拟器的无缝episode成简化了项目启动。

##  [AlohaTransferCube](https://huggingface.co/lerobot/act_aloha_sim_transfer_cube_human)

策略 Action Chunking Transformer Policy (出自论文 [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705))  针对gym-aloha中的AlohaTransferCube 环境进行了训练。

### 模型训练

该模型是使用 [LeRobot的训练脚本](https://github.com/huggingface/lerobot/blob/d747195c5733c4f68d4bfbe62632d6fc1b605712/lerobot/scripts/train.py)，以及 [aloha_sim_transfer_cube_human](https://huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_human/tree/v1.3) 数据episode进行训练的，使用的命令如下：

```python
python lerobot/scripts/train.py \
  hydra.job.name=act_aloha_sim_transfer_cube_human \
  hydra.run.dir=outputs/train/act_aloha_sim_transfer_cube_human \
  policy=act \
  policy.use_vae=true \
  env=aloha \
  env.task=AlohaTransferCube-v0 \
  dataset_repo_id=lerobot/aloha_sim_transfer_cube_human \
  training.eval_freq=10000 \
  training.log_freq=250 \
  training.offline_steps=100000 \
  training.save_model=true \
  training.save_freq=25000 \
  eval.n_episodes=50 \
  eval.batch_size=50 \
  wandb.enable=true \
  device=cuda
```

训练曲线可以在 https://wandb.ai/alexander-soare/Alexander-LeRobot/runs/hjdard15 上找到。

在一块Nvidia RTX 3090显卡上，训练大约耗时2.5小时。

### 评估

该模型在gym-aloha的AlohaTransferCube环境中进行了评估，并与使用原始仓库 [ACT（Action Chunking Transformer)](https://github.com/tonyzhaozh/act)训练的类似模型进行了比较。如果在每个（episode）中，立方体被一个机器人手臂成功抓取并转移到另一个机器人手臂，则标记为成功。

以下是对500个 episode评估的成功率结果。第一行是简单平均值。第二行假设了一个均匀先验，并计算了贝塔（Beta）后验，然后计算了均值和下限/上限置信区间（以均值为中心的68.2%置信区间）。"Theirs"列是针对在原始ACT仓库上训练并在LeRobot上评估的等效模型（模型权重可以在该仓库的[original_act_repo](https://huggingface.co/lerobot/act_aloha_sim_transfer_cube_human/tree/original_act_repo)分支中找到）。每个单独rollout的结果可以在[eval_info.json](https://huggingface.co/lerobot/act_aloha_sim_transfer_cube_human/blob/main/eval_info.json)中找到。

|                                        | Ours               | Theirs             |
| -------------------------------------- | ------------------ | ------------------ |
| Success rate for 500 episodes (%)      | 87.6               | 68.0               |
| Beta distribution lower/mean/upper (%) | 86.0 / 87.5 / 88.9 | 65.8 / 67.9 / 70.0 |

原始代码经过了大量重构，在过程中也发现了一些错误。代码的差异可能解释了成功率的差异。另一种可能性是我们的仿真环境可能使用略有不同的启发式方法来评估成功（我们观察到，一旦第二个手臂的夹持器与立方体进行对蹠点接触，就立即注册成功）。

### 使用该模型

如何开始使用模型
请参阅[LeRobot库](https://github.com/huggingface/lerobot)（特别是[评估脚本](https://github.com/huggingface/lerobot/blob/main/lerobot/scripts/eval.py)），了解如何加载和评估此模型.

## [PushT](https://huggingface.co/lerobot/diffusion_pusht)

LeRobot 在[PushT](https://huggingface.co/lerobot/diffusion_pusht)环境中进行了评估，并与使用原始 Diffusion Policy 代码训练的模型进行了比较。

### 模型训练

该模型是使用 [LeRobot的训练脚本](https://github.com/huggingface/lerobot/blob/d747195c5733c4f68d4bfbe62632d6fc1b605712/lerobot/scripts/train.py)，以及 [pusht](https://huggingface.co/datasets/lerobot/pusht/tree/v1.3) 数据集进行训练的，使用的命令如下：

```python
python lerobot/scripts/train.py \
  hydra.run.dir=outputs/train/diffusion_pusht \
  hydra.job.name=diffusion_pusht \
  policy=diffusion training.save_model=true \
  env=pusht \
  env.task=PushT-v0 \
  dataset_repo_id=lerobot/pusht \
  training.offline_steps=200000 \
  training.save_freq=20000 \
  training.eval_freq=10000 \
  eval.n_episodes=50 \
  wandb.enable=true \
  wandb.disable_artifact=true \
  device=cuda
```

训练曲线可以在https://wandb.ai/alexander-soare/Alexander-LeRobot/runs/508luayd 找到。

在 Nvida RTX 3090 上训练大约需要 7 个小时。

LeRobot 旨在适应各种机器人硬件，从基本的教育手臂到研究环境中的复杂人形机器人，提供能够控制任何类型机器人的适应性人工智能系统，从而提高机器人应用的多功能性和可扩展性。

## 开源数据

LeRobot 提供的数据episode涵盖了机器人领域的各种场景和任务。这些数据episode包括用于物体插入和传输、移动挑战和各种物体操纵等任务的模拟环境。

- 专注于人类引导的动作和脚本传输的数据episode：

  - [aloha_sim_insertion_human_image](https://huggingface.co/datasets/lerobot/aloha_sim_insertion_human_image)

  - [aloha_sim_transfer_cube_scripted_image](https://huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_scripted_image)

- 涉及静态物体的数据episode：

  - [aloha_static_battery](https://huggingface.co/datasets/lerobot/aloha_static_battery)

  - [aloha_static_candy](https://huggingface.co/datasets/lerobot/aloha_static_candy)

- 与手臂运动和操纵相关的数据episode：

  - [xarm_push_medium_replay_image](https://huggingface.co/datasets/lerobot/xarm_push_medium_replay_image)

  - [xarm_lift_medium_image](https://huggingface.co/datasets/lerobot/xarm_lift_medium_image)
