# Implementing RLHF: Learning to Summarize with trlX

## 介绍

随着最近 ChatGPT 的 推出，基于人类反馈的强化学习 (RLHF) 已成为语言建模界的热门话题——包括学术界和工业界。

我们可以追溯 RLHF 在自然语言处理中的应用，  OpenAI 2019 年发布的[Fine-Tuning Language Models from Human Preferences](https://arxiv.org/abs/1909.08593)。快进一年了，OpenAI 发布了第一篇关于从人类反馈强化学习应用于自然语言生成的重要论文之一。在那篇论文——[学习从人类反馈中总结](https://arxiv.org/abs/2009.01325)——OpenAI 表明，在根据人类偏好进行评估时，简单地对总结数据进行微调会导致表现不佳。作者建议直接通过强化学习方法针对人类偏好进行优化，以缓解这些性能问题。

## 使用 trlX]

﻿[CarperAI 的trlX](https://github.com/CarperAI/trlx)是一个分布式训练框架，其灵感来自 Transformer 强化学习库（可在此处找到：[lvwerra/trl）](https://github.com/lvwerra/trl)。trlX 从头开始设计，以大规模关注 RLHF，这是重现最近 RLHF 文献中观察到的许多结果的必要因素 [ [Steinnon 等人，2020 年](https://arxiv.org/abs/2009.01325)；[Askell et al., 2021](https://arxiv.org/abs/2112.00861) , [Ouyang et al., 2022](https://arxiv.org/abs/2203.02155) ].

特别是，trlX[从人类偏好过程中抽象出微调语言模型](https://arxiv.org/abs/1909.08593)的 RL 部分，使研究人员能够专注于管理强化学习的挑剔动态的高级选择，而不是运行分布式训练所需的样板代码。它的设计足够灵活以支持广泛的算法，目前支持[近端策略优化](https://openai.com/blog/openai-baselines-ppo/)(PPO) 和[隐式语言 Q 学习](https://arxiv.org/abs/2206.11871)(ILQL) 。

> 在下面的例子中，奖励函数是手工制作的。如上所述，trlX 抽象了 RLHF 的 RL 组件，用于微调 LLM。您可以带上训练有素的奖励模型或手工制作。

```python
sentiment_fn = pipeline(
	"sentiment-analysis",
	"sentiment-analysis",
	"gpt2",
	top_k=2,
	truncation=True,
	batch_size=256,
	device=device,
)


def get_positive_score(scores):
	"Extract value associated with a positive sentiment from pipeline's output"
	return dict(map(lambda x: tuple(x.values()), scores))["POSITIVE"]


def reward_fn(samples: List[str]) -> List[float]:
	sentiments = list(map(get_positive_score, sentiment_fn(samples)))
	return sentiments


trainer = trlx.train("gpt2", reward_fn=reward_fn)

```

或者，要使用离线 ILQL，请提供您的奖励标记数据集：

```
trainer = trlx.train(
	"EleutherAI/gpt-j-6B",
	dataset=[("dolphins", "geese"), (1.0, 100.0)],
)
```

截至发稿时，trlX 可以借助 HuggingFace [Accelerate](https://huggingface.co/docs/accelerate/index)对模型进行 30B 规模的微调。我们正在继续努力，以尽快支持具有替代后端的更大模型。欢迎投稿！

[您可以从他们的示例](https://github.com/CarperAI/trlx/tree/main/examples)中了解有关使用 trlX 的更多信息。💡

## 从摘要中学习

在本节使用的 trlX 中，我们将为摘要任务实施 RLHF。训练过程包括三个部分：

- 我们将首先在我们的摘要数据集上微调预训练的Transformer模型（下一节将详细介绍数据集）。这是我们的监督微调模型 (SFT)。
- 然后我们将训练一个奖励模型（RM）。该模型从 SFT 模型初始化并输出一个标量值。这个标量值是表示摘要偏好的奖励。
- 最后，我们使用 RM 通过 PPO 微调 SFT 模型。此步骤使我们的 SFT 模型与人类偏好保持一致。

## 数据集

对于我们今天的实验，我们将使用最初在学习中使用的 TL;DR 摘要数据集[来从人类反馈中进行总结](https://arxiv.org/abs/2009.01325)。

基于上述训练过程，我们需要两种类型的数据集：

- 一个用于微调预训练的监督模型，然后用 PPO 和奖励模型再次对其进行微调，以及
- 一个用于训练我们的奖励模型。

在我们的例子中，用于微调的数据集是过滤过的 TL;DR 数据集。用于训练奖励模型的数据集是比较或偏好数据集。

> 作者过滤了原始的 TL;DR 数据集，以包含一个安全的 subreddits 列表，这些列表很容易被普通大众理解。此外，他们只有样本，其中人工编写的摘要在 24 到 48 个标记之间。

### 如何下载数据集

我们将首先下载[AzCopy](https://learn.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10)，这是一个命令行实用程序，可用于将 blob 或文件复制到存储帐户或从中复制。相关代码：

[可以在官方存储库](https://github.com/openai/summarize-from-feedback)中找到指向 TL;DR 数据集和比较数据集的不同拆分的链接。

以下是下载 TL;DR 数据集的训练拆分的方法：

```
!azcopy copy "https://openaipublic.blob.core.windows.net/summarize-from-feedback/datasets/tldr_3_filtered/train.jsonl"
```

### TL;DR 数据集

TL;DR 摘要数据集包含 129,722 个 Reddit 帖子，其中约 5% 用于拆分验证和测试。训练集中总共有 116,722 个样本，验证集中有 6,447 个样本，测试集中有 6,553 个样本。我们将使用此数据集来微调我们的模型。

这是一个示例：

```json
{
  'id': 't3_1hxu8s',
  'subreddit': 'relationships',
  'title': 'I (f/22) have to figure out if I want to still know these girls or not and would hate to sound insulting',
  'post': "Not sure if this belongs here but it's worth a try. \n\nBackstory:\nWhen I (f/22) went through my first real breakup 2 years ago because he needed space after a year of dating roand  it effected me more than I thought. It was a horrible time in my life due to living with my mother and finally having the chance to cut her out of my life. I can admit because of it was an emotional wreck and this guy was stable and didn't know how to deal with me. We ended by him avoiding for a month or so after going to a festival with my friends. When I think back I wish he just ended. So after he ended it added my depression I suffered but my friends helped me through it and I got rid of everything from him along with cutting contact. \n\nNow: Its been almost 3 years now and I've gotten better after counselling and mild anti depressants. My mother has been out of my life since then so there's been alot of progress. Being stronger after learning some lessons there been more insight about that time of my life but when I see him or a picture everything comes back. The emotions and memories bring me back down. \n\nHis friends (both girls) are on my facebook because we get along well which is hard to find and I know they'll always have his back. But seeing him in a picture or talking to him at a convention having a conversation is tough. Crying confront of my current boyfriend is something I want to avoid. \n\nSo I've been thinking that I have to cut contact with these girls because it's time to move on because it's healthier. It's best to avoid him as well. But will they be insulted? Will they accept it? Is there going to be awkwardness? I'm not sure if it's the right to do and could use some outside opinions.",
  'summary': "I still have contact with an old ex's friends but can't stand to see or talk to him. His friends are really nice ,so how do I tell them I possibly want to unfriend them on Facebook because of him?"
}
```

该数据集经过精心整理以用于微调，并作为 Hugging Face 数据集托管。[你可以在这里](https://huggingface.co/datasets/CarperAI/openai_summarize_tldr)找到。数据集格式（验证集）如下所示。提示是与 Subreddit 名称和标题相连的 Reddit 帖子。label是真人写的总结：

### 比较数据集

比较数据集由训练数据集中的 92,858 个样本和验证集中的 83,797 个样本组成。从功能上讲，这些只是 Reddit 帖子和每个帖子的两个摘要。它还具有一个选择值，指示人工标记者更喜欢两个摘要中的哪一个（在下面标记为“选择”：0）。

这是一个示例：

```json
{
    "info": {
        "id": "t3_3pb8rl",
        "post": "Hi reddit.\n\nI recently started dating a woman that I really like, after talking to her a lot for around a month. We go to university together and have a bunch of classes together, eat together, study together, etc. I asked her out, we went to the movies, had a lot of fun, kissed, yada yada.  \n\nMy biggest problem is that I've never been in a relationship. I'm relatively inexperienced romantically(kissed like 2 girls and had sex once before), and this is the first time I met someone that I thought 'Damn I really want to spend a lot of time with you'.\n\nI really like her, and so I don't want to rush things, but then I don't know what I can or can't do. How often can we hold hands? Do we just kiss whenever one of us feels like it? How do I know she wants to be kissed at a particular moment? How do I know HOW she wants to be kissed? How do I know if I'm doing something 'wrong'?\n\nThese are a bunch of things that, if it were some random girl, I wouldn't even care about(or at least not care as much). I really just don't want to fuck this up. Are there any basic relationship rules or something other than 'do what your heart wants'? I appreciate anything you guys can tell me (criticisms or advice)\n\nThanks in advance.\n\nP.S I'm guessing that some people will wonder about the age gap. We've talked about it. It's weird but we both like each other and don't care for it. The fact that she's older than me only stresses me out more because she's had more experience with relationships than me, and I really, REALLY don't want to fuck up.\n\nP.S.S This is my first post here, so I'm not sure how things work. If you guys need any additional information that I didn't mention to help out just ask :P",
        "title": "I [19/M] just started dating a girl [25/F] I really like, but I've never been in an actual relationship. I don't really know what to do.",
        "subreddit": "relationships"
    },
    "split": "train",
    "summaries": [
        {
            "text": " I've never been in a relationship, but I like this woman. How do I know if I'm doing things wrong? How do I know if I like her?",
            "policy": "sup2",
            "note": "ok"
        },
        {
            "text": " I'm dating a girl, I don't know how things work. I want to make it work, but I don't know what the hell I can/should do.",
            "policy": "sup2",
            "note": "OP doesn't have relationship experience"
        }
    ],
    "choice": 0,
    "worker": "HNzkrs9geGu1YMMfZ5Qvdt0ZaCthfB",
    "batch": "batch5",
    "extra": {}
}
```

### 这些摘要是如何生成的？

对于每个 Reddit 帖子（在数据集中），使用不同的模型生成 N 个摘要。预训练模型用作零样本摘要生成器，并且还使用监督微调（在 Reddit TL;DR 上）模型（12B、6B 和 1.3B）生成摘要。人工编写的 TL;DR（参考）也被视为样本。在下图中，这些模型被视为策略。

每个帖子的这 N 个摘要被成对批处理并发送给雇佣的标签员。贴标签者选择/偏爱一个摘要而不是另一个。

![图片](https://api.wandb.ai/files/carperai/images/projects/37218153/8323a45b.png)

该数据集专为训练奖励模型而设计，并作为 HuggingFace 数据集托管。你可以[在这里](https://huggingface.co/datasets/CarperAI/openai_summarize_comparisons)找到它。数据集格式如下所示。提示是与 Subreddit 名称和标题连接的 Reddit 帖子，而“选择”列显示评论者首选的标签。当然，鉴于人类反馈仍然是一个开放的研究领域，使用数据集的方式没有对错之分。

## 源代码

本教程中使用的脚本可以在[trlX存储库的](https://github.com/CarperAI/trlx)[trlx/examples/summarize_rlhf/](https://github.com/CarperAI/trlx/tree/main/examples/summarize_rlhf) * 目录中找到。

要开始，请首先按照下面概述的 trlX 安装指南进行操作：

```
git clone https://github.com/CarperAI/trlx.git
cd trlx
pip install torch --extra-index-url https://download.pytorch.org/whl/cu116 # for cuda
pip install -e .
```

## 监督微调 (SFT)

接下来，我们将在 TL;DR 数据集上微调 GPT-J 模型以进行文本摘要。

这是相对简单的。加载数据集，对其进行 tokenize ，然后训练模型。整个 pipeline 是使用 HuggingFace 构建的。微调：

```
!deepspeed examples/summarize_rlhf/sft/train_gptj_summarize.py
```

我们的模型使用 ROUGE 分数进行评估。验证集上的平均 ROUGE 分数选择最佳模型。该模型将用于初始化奖励模型，稍后将使用 PPO 进行微调。

下面显示的图表总结了 TL;DR 数据集测试集上的不同 ROUGE 分数。

## 训练奖励模型

我们的奖励模型是用收集到的人类质量判断数据集训练的。该模型将给定的帖子和候选摘要映射到奖励*r* 。

我们将从 SFT 模型初始化奖励模型，并附加一个随机初始化的线性头，在顶部输出标量值。

接下来，我们将更详细地研究数据如何输入到模型、损失函数和奖励模型的其他问题。

### 原始输入

[数据加载器将使用此处](https://huggingface.co/datasets/pvduy/openai_summarize_comparisions)托管的比较数据集。不过在此之前，我们将使用 create_comparison_dataset 函数（如下所示）创建一个字典列表，其中每个字典都有两个键 - 选择和拒绝。每个键的值是与摘要连接的提示（或 Reddit 帖子）。

```python
def create_comparison_dataset(
     path="CarperAI/openai_summarize_comparisons", split="train"
 ):
     dataset = load_dataset(path, split=split)
     if split == "test":
         dataset = dataset.select(range(10000))
﻿

     pairs = []
     for sample in tqdm(dataset):
         pair = {}
         prompt = sample["prompt"]
         chosen_summary = sample["chosen"]
         rejected_summary = sample["rejected"]
         if chosen_summary == rejected_summary:
             continue
         if  len(chosen_summary.split()) < 5 or len(rejected_summary.split()) < 5:
             continue
         pair["chosen"] = prompt + "\n" + chosen_summary
         pair["rejected"] = prompt + "\n" + rejected_summary
         pairs.append(pair)
     return pairs
```

### 成对数据加载

下面显示的 PairwiseDataset 类标记了选择和拒绝的“摘要”。数据集类返回选择和拒绝摘要的 input_ids 和 attention_masks：

```python
class PairwiseDataset(Dataset):
     def __init__(self, pairs, tokenizer, max_length):
         self.chosen_input_ids = []
         self.chosen_attn_masks = []
         self.rejected_input_ids = []
         self.rejected_attn_masks = []
         for pair in tqdm(pairs):
             chosen, rejected = pair["chosen"], pair["rejected"]
             chosen_encodings_dict = tokenizer(
                 "<|startoftext|>" + chosen + "<|endoftext|>",
                 truncation=True,
                 max_length=max_length,
                 padding="max_length",
                 return_tensors="pt",
             )
             rejected_encodings_dict = tokenizer(
                 "<|startoftext|>" + rejected + "<|endoftext|>",
                 truncation=True,
                 max_length=max_length,
                 padding="max_length",
                 return_tensors="pt",
             )
             self.chosen_input_ids.append(chosen_encodings_dict["input_ids"])
             self.chosen_attn_masks.append(chosen_encodings_dict["attention_mask"])
             self.rejected_input_ids.append(rejected_encodings_dict["input_ids"])
             self.rejected_attn_masks.append(rejected_encodings_dict["attention_mask"])
﻿

     def __len__(self):
         return len(self.chosen_input_ids)
﻿

     def __getitem__(self, idx):
         return (
             self.chosen_input_ids[idx],
             self.chosen_attn_masks[idx],
             self.rejected_input_ids[idx],
             self.rejected_attn_masks[idx],
         )
```

### Data Collator

DataCollatorReward 类为我们的奖励模型创建数据批次（dict）。整理器返回：

- input_ids: collator 在 dim=0 上连接选择和拒绝的摘要的 input_ids。
- attention_mask: collator 在 dim=0 上连接选择和拒绝的摘要的 attention_mask。
- labels: collator 为选择的摘要创建一个零张量，为在 dim=0 上连接的拒绝摘要创建一个张量。

请注意，由于这种连接，提供给模型的批处理是全局批处理大小的两倍。

```python
class DataCollatorReward:
     def __call__(self, data):
         batch = {}
         batch["input_ids"] = torch.cat([f[0] for f in data] + [f[2] for f in data])
         batch["attention_mask"] = torch.cat([f[1] for f in data] + [f[3] for f in data])
         batch["labels"] = torch.tensor([0] * len(data) + [1] * len(data))
         return batch
```

### 奖励模型

在这里，我们有一个 Reddit 帖子和两个摘要（选择和拒绝）作为输入。真实标签（labels）是人类的反馈（0 代表选择，1 代表拒绝）。损失函数为：

在上述公式中， *�� y* **i ﻿，其中 ��∈{0,1} *i* ∈{0,1} ，是人类首选或选择的摘要。奖励模型 *��r* **θ﻿ 采用帖子 *��x*﻿ 和摘要 *��y*﻿ 并返回标量值。为两个摘要计算该值，并将 sigmoid 激活应用于差异。最后，计算负对数。

![图片](https://api.wandb.ai/files/carperai/images/projects/37218153/8b589edc.png)

（[来源](https://arxiv.org/pdf/2009.01325.pdf)）

GPTRewardModel 类使用 SFT 模型和其上的线性层初始化 GPT-J 模型。它还计算上面显示的损失。

```python
class GPTRewardModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        model = AutoModelForCausalLM.from_pretrained(config)
        self.config = model.config
        # gpt-neo models have hidden_size instead of n_embd
        self.config.n_embd = (
            self.config.hidden_size
            if hasattr(self.config, "hidden_size")
            else self.config.n_embd
        )
        self.transformer = model.transformer
        self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.PAD_ID = self.tokenizer(self.tokenizer.pad_token)["input_ids"][0]
﻿

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
    ):
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = transformer_outputs[0]
        rewards = self.v_head(hidden_states).squeeze(-1)
        reward_scores = []
        bs = input_ids.shape[0] // 2
    # Note half is chosen and another half is rejected.
        chosen = input_ids[:bs]
        rejected = input_ids[bs:]
        chosen_rewards = rewards[:bs]
        rejected_rewards = rewards[bs:]
        # compute pairwise loss. Only backprop on last value before padding
        loss = 0
        for i in range(bs):
            # Find the index of the first occurrence where chosen summary input_ids
        # and rejected summary input_ids are different.
            divergence_ind = (chosen[i] != rejected[i]).nonzero()[0]
﻿

        # Find the index of the first occurrence of the padding token the chosen summary.
            c_inds = (chosen[i] == self.PAD_ID).nonzero()
            c_ind = c_inds[0].item() if len(c_inds) > 0 else chosen.shape[1]
﻿

        # Find the index of the first occurrence of the padding token the rejected summary.
            r_inds = (rejected[i] == self.PAD_ID).nonzero()
            r_ind = r_inds[0].item() if len(r_inds) > 0 else rejected.shape[1]
            end_ind = max(c_ind, r_ind)
        
        # Find the slice of reward which belongs to diverging input_ids
            c_truncated_reward = chosen_rewards[i][divergence_ind:end_ind]
            r_truncated_reward = rejected_rewards[i][divergence_ind:end_ind]
            reward_scores.append(c_truncated_reward[-1])  # reward at last token
            
            # Compute loss
            loss += -torch.log(
                torch.sigmoid(c_truncated_reward - r_truncated_reward)
            ).mean()
            loss = loss / bs
        return {"loss": loss, "reward_scores": torch.stack(reward_scores)}
```

我们的模型接收数据整理器准备的输入。此输入通过 GPT-J 模型传递以获得最终的隐藏状态。然后隐藏状态通过线性层获得奖励分数。对于输入模型的每个批次，前半部分是选择的摘要，后半部分是拒绝的摘要。模型的前向方法遍历每个输入样本以计算成对损失。计算此损失所需的步骤记录在上面的代码片段中。

要训练奖励模型运行：

```python
!deepspeed examples/summarize_rlhf/reward_model/train_reward_model_gptj.py
```

下面，我们展示了整个奖励模型训练过程中的训练和验证损失以及准确性。

## 使用 PPO 进行微调

[我们现在可以使用 trlX 使用近端策略优化](https://openai.com/blog/openai-baselines-ppo/)(PPO) 算法微调 SFT 模型。

PPO算法使用价值函数，可以是深度学习模型。在我们的例子中，这个值函数是用 SFT 模型初始化的 GPT-J 模型。策略 ( *� π* ) 也使用 Reddit TL;DR 数据集上的微调 GPT-J Transformer (SFT) 进行初始化。然后像任何 RL 策略一样使用奖励模型的输出作为该策略的奖励对其进行训练。[﻿](https://openai.com/blog/openai-baselines-ppo/)﻿

![图片](https://api.wandb.ai/files/carperai/images/projects/37218153/1c367d95.png)

（[来源](https://arxiv.org/pdf/2009.01325.pdf)）

但是，这里有几点值得牢记：

### 陷阱 1：规范化

由于原始奖励分数具有高方差，因此使用从人类编写的摘要计算的奖励分数对其进行归一化。在按以下方式训练奖励模型后进行归一化：

其中 ��(��) *r**m* ( *x* ) 和 ��(����) *r**m* ( *x **r** e**f* ) 是经过训练的奖励模型在“post+model generated summary”和“post+human-written summary”。“post+<....>”的意思是，“<...>”连接到 Reddit“post”，如上一节所示。

trlX 框架需要一个在下面实现的 reward_fn。规范化步骤是在此函数本身中完成的。

```python
def reward_fn(samples: List[str]):
    # get humans summarizes
    posts = [sample.split('TL;DR')] for sample in samples]
    ref_samples = [post + 'TL;DR' + post_summ_dict[post] for post in post]
    samples_encodings = reward_tokenizer(samples)
    samples_scores = reward_model(**samples_encodings) # get scores from reward model for samples
    ref_samples_encodings = reward_tokenizer(ref_samples) # get scores from reward model corresponding references samples
    ref_samples_scores = reward_model(**ref_samples_encodings)
    norms_rewards = samples_scores - ref_samples_scores
    return norms_rewards
```

### 陷阱 2：KL 散度

在使用 PPO 管道进行微调时，会使用我们的策略 (LLM) 为 Reddit 帖子生成摘要。这篇文章和摘要被传递给奖励模型以获得奖励分数。此奖励分数用于更新策略。请注意，操作是分批完成的。然而，RL 训练有噪音，尤其是在开始时，这可能会使我们的政策偏离奖励有效的范围太远。

为了防止这种情况发生，在奖励函数中添加了一个 KL 项作为惩罚，如下所示：

这个 KL 术语是在 trlX 框架中[实现的](https://github.com/CarperAI/trlx/blob/0c5246f64e5e0ecb5fb2de65d440b122c792caf8/trlx/orchestrator/ppo_orchestrator.py#L224)，因此您不需要自己实现它。

要使用 PPO 和训练有素的奖励模型微调 SFT 模型，请执行以下操作：

```
!deepspeed examples/summarize_rlhf/trlx_gptj_text_summarization.py
```

让我们看看使用 trlX 微调我们的 SFT 模型时的损失。

在使用 RL 训练代理时，目标是最大化奖励分数。下图显示了平均奖励随着训练的进行而增加。

让我们看一下使用 PPO 微调的 SFT 模型的 ROUGE 分数，并将其与 SFT 模型的 ROUGE 分数进行比较。请注意，ROUGE 分数越高越好。

显然，使用 PPO 微调的 SFT 模型的 ROUGE 分数比仅 SFT 模型差。那么有监督的微调就足够了吗？并不真地。ROUGE 不捕捉人类的偏好。如果模型简单地生成类似于人类编写的摘要，这样的分数会更高。但是给定的人工编写的摘要可能不是首选。我们想要一个整体上符合人类偏好的模型。

如下图所示，官方报告的 ROUGE 分数与我们的结果（PPO 微调模型具有较低的 ROUGE 分数）趋势一致。

![图片](https://api.wandb.ai/files/carperai/images/projects/37218153/0ebd093e.png)

（[来源；第 34 页](https://arxiv.org/pdf/2009.01325.pdf)）

下面让我们看看我们的 SFT 模型和 PPO 微调模型生成的一些摘要。作为人类读者，您可以决定 RL_PPO 摘要是否优于简单的监督微调 (SFT) 摘要。

> 警告：某些样本可能包含具有攻击性的输出。

## 结论

﻿[InstructGPT](https://openai.com/blog/instruction-following/)表明，通过结合人类反馈（通过学习奖励函数）和使用 RL，LLM 更符合人类偏好。符合人类偏好的模型可以[提高模型的安全性和情绪](https://arxiv.org/pdf/2204.05862.pdf)，但是，它不会消除 LLM 中的潜在偏见。[ChatGPT](https://openai.com/blog/chatgpt/)，它的兄弟，使用了一种对话格式，可以回答后续问题、承认错误、挑战不正确的前提和拒绝不适当的请求。ChatGPT 抓住了大众的想象力。它首次使 RL 实用化。

为了让 RLHF 的研究更容易获得，CarperAI 的人们构建了 trlX - 一个存储库，允许您使用强化学习微调 Hugging Face 支持的语言模型（基于 gpt2、gpt-j、gpt-neo 和 gpt-neox）并提供奖励模型。他们还构建了[CHEESE](https://github.com/CarperAI/cheese)，可以帮助研究人员构建满足 RLHF 需求的数据标注平台。

最后，本教程旨在使 RLHF 更易于理解。我们已经展示了如何使用 trlX 为摘要任务实现 RLHF。

我们希望它能激发大家更多地了解这个概念。如果你想为 trlX 贡献一个有价值的例子，打开一个 PR。您也可以加入 CarperAI 的[Discord 频道](https://discord.com/invite/KgfkCVYHdu)，就本教程提出问题，更积极地参与。

## 参考

1. Nisan Stiennon, Long Ouyang, Jeff Wu, Daniel M. Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, Paul Christiano, "[Learning to summarize from human feedback](https://proceedings.neurips.cc/paper/2020/file/1f89885d556929e98d3ef9b86448f951-Paper.pdf)", Neural Information Processing Systems, 2020.
2. Daniel M. Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B. Brown, Alec Radford, Dario Amodei, Paul Christiano, Geoffrey Irving, "[Fine-Tuning Language Models from Human Preferences](https://arxiv.org/abs/1909.08593)", arXiv, 2019.
3. Amanda Askell, Yuntao Bai, Anna Chen, Dawn Drain, Deep Ganguli, Tom Henighan, Andy Jones, Nicholas Joseph, Ben Mann, Nova DasSarma, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Jackson Kernion, Kamal Ndousse, Catherine Olsson, Dario Amodei, Tom Brown, Jack Clark, Sam McCandlish, Chris Olah, Jared Kaplan, "[A General Language Assistant as a Laboratory for Alignment](https://arxiv.org/abs/2112.00861)", arXiv, 2021.
4. John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov, "[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)", arXiv, 2017.
5. Charlie Snell, Ilya Kostrikov, Yi Su, Mengjiao Yang, Sergey Levine, "[Offline RL for Natural Language Generation with Implicit Language Q Learning](https://arxiv.org/abs/2206.11871)", arXiv, 2022.
6. Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe, "[Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)", arXiv, 2022.
7. Ayush Thakur, "[Understanding Reinforcement Learning from Human Feedback (RLHF): Part 1](https://wandb.ai/ayush-thakur/RLHF/reports/Understanding-Reinforcement-Learning-from-Human-Feedback-RLHF-Part-1--VmlldzoyODk5MTIx)", 2023.
8. Nathan Lambert, Louis Castricato, Leandro von Werra, Alex Havrilla, "[Illustrating Reinforcement Learning from Human Feedback (RLHF)](https://huggingface.co/blog/rlhf)", 2022.

