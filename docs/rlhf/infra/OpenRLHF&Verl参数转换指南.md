

## OpenRLHF&Verl 参数转换

### 模型参数

| 介绍           | OpenRLHF        | Verl                         |
| -------------- | --------------- | ---------------------------- |
| Actor模型路径  | pretrain        | actor_rollout_ref.model.path |
| Reward模型路径 | reward_pretrain | reward_model.model.path      |
| Critic模型路径 | critic_pretrain | critic.model.path            |

### 优化参数

| 介绍             | OpenRLHF             | Verl                             |
| ---------------- | -------------------- | -------------------------------- |
| Actor模型学习率  | actor_learning_rate  | actor_rollout_ref.actor.optim.lr |
| Critic模型学习率 | critic_learning_rate | critic.optim.lr                  |

[OpnenRLHF](https://github.com/OpenRLHF/OpenRLHF)默认使用的是Warmup-Decay的学习率调度器，所以也支持设置warmup步数等其他的相关参数，具体的可以在[train_ppo_ray.py](https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/cli/train_ppo_ray.py)中找到相关的参数。

Verl中对于学习率的调度方式可以选择为`constant/cosine`，在yaml文件中依然可以找对应的[位置](https://github.com/volcengine/verl/blob/f8acd9017b4db4eead1f34beb39fce9c39143194/verl/trainer/config/ppo_trainer.yaml%23L45)。

### 数据参数

| 介绍                             | OpenRLHF                 | Verl                                                         |
| -------------------------------- | ------------------------ | ------------------------------------------------------------ |
| 训练阶段单卡分配的experience数量 | micro_train_batch_size   | actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu         |
| 训练时全局的experience数量       | train_batch_size         | actor_rollout_ref.actor.ppo_mini_batch_size*actor_rollout_ref.rollout.n(本节末有解释) |
| 探索阶段单卡分配的experience数量 | micro_rollout_batch_size | \(自动计算)                                                  |
| 探索阶段的prompt数量             | rollout_batch_size       | data.train_batch_size                                        |
| 实际使用的最大prompt数量         | max_samples              | \                                                            |
| 单个prompt采样次数               | n_samples_per_prompt     | actor_rollout_ref.rollout.n                                  |
| 训练阶段experience的学习次数     | max_epochs               | actor_rollout_ref.actor.ppo_epochs                           |
| 数据集迭代次数                   | num_episodes             | trainer.total_epochs                                         |

#### OpenRLHF流程：

(1)首先，当给定一个数据集之后，框架会从中选择至多max_samples个prompt。假设我们的数据集仅有1024个prompt，并且1024小于max_samples，则1024个prompt全部保留。

(2)之后进入探索阶段，由于一次探索完1024个prompt的时间太长，所以选择一次只对rollout_batch_size个prompt进行探索。我们假设rollout_batch_size为32，则一共需要探索1024÷32=32步。这个32步就是我们在wandb或者tensorboard上面看到的步骤，我们称之为explore step。我们会用[vllm](https://zhida.zhihu.com/search?content_id=254852206&content_type=Article&match_order=1&q=vllm&zhida_source=entity)对每个prompt进行采样n_samples_per_prompt次，得到所有的samples。我们假设n_samples_per_prompt为8，则得到了32×8=256个样本，即每个样本都是一个问答对，一共有32个问题，并且相同的问题回答了8次。

(3)之后需要生成experience，这时就需要切换到训练引擎，即在1步内单卡负责生成micro_rollout_batch_size个经验，我们假设micro_rollout_batch_size为4，我们有8张卡，则1步一共可以生成32个experience。由于我们一共有256个样本，所以需要一共需要256÷32=8步可以得到全部的experience。在make experience阶段我们主要利用Reward模型得到每个答案的奖励分数、用Critic模型给出每个答案每一步的Value值（如果有Critic模型），以及用Refrence模型和Actor模型给出每个答案每一步的预测概率并且计算出对应的KL惩罚值（如果有Refrence模型）

(4)在结束探索阶段后，就进入了训练阶段。刚才一共得到了256个experience，但是我们的显卡不足以一次性在所有的样本上进行训练，因此我们设置train_batch_size为128，即每次只更新128个experience，则需要256÷128=2步，也就是说在训练阶段模型更新了2次，我们称之为update step。假设micro_train_batch_size为4，我们有8张卡，则1步一共可以训练32个experience，那么我们需要4步梯度累计，然后才进行反向传播。当update step大于1，也就是所有的experience不能一次更新完的时候，就称之为off policy，反之如果update step=1，也就是模型探索一步就更新一步，则称之为on policy。并且需要注意，如果max_epochs＞1，此时这一组经验被训练了多次，即对256个experience进行了多次优化，那么此时的策略一定是off policy，所以一般情况下我们默认这个参数为1即可，因为我们希望尽可能地确保我们的优化是on policy的。

(5)我们再强调一遍，我们的数据集有1024个prompt，每次探索和训练其中的32个，则经过以上流程循环1024÷32=32步，我们已经探索并且训练完了整个数据集，之后我们再训练num_episodes次，则完成了整个训练流程。

#### Verl流程

Verl流程上，从算法逻辑上与OpenRLHF的是相同的，但是其中需要关注的是`actor_rollout_ref.actor.ppo_mini_batch_size`参数是一个与采样数`n`无关的参数，从上述的1024个prompt的数据集来看，如果`ppo_mini_batch_size`的设置为32的话，则一定进行1024÷32=32次的参数更新，无论`sample_n`的设置值为多少。

但要注意，verl中的`ppo_micro_batch_size_per_gpu`是已经考虑了`sample_n`，因此又重新对OpenRLHF的参数含义相同。

### 批处理参数部分

上文提到的actor的训练bs设置不再赘述。

| 介绍                                     | OpenRLHF               | verl                                                        |
| ---------------------------------------- | ---------------------- | ----------------------------------------------------------- |
| rollout计算log_prob时每个GPU的微批量大小 | micro_train_batch_size | actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu |
| ref计算log_prob时每个GPU的微批量大小     | micro_train_batch_size | actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu     |
| critic模型单GPU计算的批量大小            | micro_train_batch_size | critic.forward_micro_batch_size_per_gpu                     |
| reward模型单GPU计算的批量大小            | micro_train_batch_size | reward_model.micro_batch_size_per_gpu                       |

在OpenRLHF中，通常由`micro_train_batch_size`来管理所有的bs信息，对于不同的模型角色，实际上使用的是一个参数进行管理。但在verl中，这个部分的参数则是可以指定的。一般来说如果脚本中不特地指定数值，则会保持与`actor_rollout_ref.actor`中设置一致。并且在verl中，落实到`per_gpu`的参数设置，一般都是已经考虑了sample_n次的结果。

### 生成参数

| 介绍               | OpenRLHF         | Verl                     |
| ------------------ | ---------------- | ------------------------ |
| 采样阶段的温度系数 | temperature      | rollout.temperature      |
| prompt最大长度     | prompt_max_len   | data.max_prompt_length   |
| 生成回答的最大长度 | generate_max_len | data.max_response_length |
