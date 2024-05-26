# Preference Datasets

## nvidia/HelpSteer

- HuggingFace Hub : https://huggingface.co/datasets/nvidia/HelpSteer

- 摘要：NVIDIA/HelpSteer 是一个开源的多属性有用性数据集，旨在支持模型对齐，使其变得更加有用、事实正确且连贯，同时在响应的复杂性和冗长性方面可调3。该数据集包含 37,120 个样本，每个样本都包含一个提示、一个响应以及五个人类标注的属性，每个属性的评分范围在 0 到 4 之间，分数越高代表属性越好。这些属性包括：

  - **帮助性**（Helpfulness）：评估响应对用户的有用性。

  - **正确性**（Correctness）：检查响应的准确性。

  - **连贯性**（Coherence）：评估响应的逻辑清晰度和条理。

  - **复杂性**（Complexity）：衡量响应的复杂程度。

  - **冗长性**（Verbosity）：评估响应是否过于冗长。

- 数据规模： 37k

## Anthropic/hh-rlhf

Huggingface Hub : https://huggingface.co/datasets/Anthropic/hh-rlhf

- 摘要：有关有用和无害的人类偏好数据, 用来训练有用和无害的人类助手。 这些数据旨在为后续 RLHF 训练训练偏好（或奖励）模型。这些数据*并不*用于对话智能体的监督训练。根据这些数据训练对话智能体可能会导致有害的模型。

- 数据规模
  - Train   161k
  - Test     8.55k

## Anthropic_HH_Golden

- HuggingfaceHub: https://huggingface.co/datasets/Unified-Language-Model-Alignment/Anthropic_HH_Golden

- Anthropic的“有用且无害”（HH）数据集旨在训练人工智能模型更加符合人类价值观，专注于有用性和无害性。数据集由两组响应组成，其中一组被选为更符合期望的价值观，另一组被拒绝。对Anthric无害数据集的改进包括使用GPT-4重新编写原始的“被选择”答案。与原始的无害数据集相比，实证表明，这个改进版数据集在无害性指标上显著提高了RLHF、DPO或ULMA方法的性能。

- 数据规模：train 42.5k + test 2.3k

## lmsys/chatbot_arena_conversations

- HuggingfaceHub: https://huggingface.co/datasets/lmsys/chatbot_arena_conversations
- 摘要： 由 `lmsys` 组织开源的一个数据集，它包含了在他们创建的 `chatbot arena` 评测系统上收集的人类偏好数据。
  - **数据集大小**：第一批数据集包含约33k个样本。
  - **数据收集**：数据收集自2023年4月至6月，用户在 `chatbot arena` 评测系统上输入对话后，系统会调用两个大型语言模型（LLM）生成回答，然后用户可以对这些回答进行评价，表达他们认为哪个回答更好。
  - **数据清洗**：收集到的数据已经过清洗，去除了个人信息和不适当的对话内容，并且使用训练好的模型进行了毒性标记。
  - **数据字段**：数据集包含多个字段，如 `question_id`、`model_a`、`model_b` 等标识信息，对话内容以 `conversation_a` 和 `conversation_b` 的形式表示，还包括用户投票（`user vote`）、语言（`language`）、时间（`time`）等额外信息。此外，还有 OpenAI 内容审核结果（`openai_moderation`）和额外的毒性标记（`toxic_chat_tag`）。
  - **有用性评价**：对话内容是否有帮助是由用户完成的，包含在 `winner` 字段里面。
  - **安全性打分**：安全性打分是由 GPT 完成的，包括多个方面，如骚扰、威胁、仇恨、自残、性内容、暴力等。
- 数据规模：33k

## allenai/FineGrainedRLHF

- Github: https://github.com/allenai/FineGrainedRLHF
- 摘要： 旨在开发新框架以收集人类反馈的存储库。收集的数据目的是提高大型语言模型的事实正确性、话题相关性和其他能力。
- 数据规模： 5K

## OpenAssistant/oasst1

- Huggingface Hub : https://huggingface.co/datasets/OpenAssistant/oasst1

- 摘要：OpenAssistant Conversations (OASST1)，是一个人工生成、人工注释的对话语料库，由 35 种不同语言的 161,443 条消息组成，注释有 461,292 个质量评级，从而超过 10,000 个带完整注释的对话树。该语料库是全球众包努力的成果，涉及超过 13,500 名志愿者。

## PKU-Safety-Prompts

- Github: https://github.com/thu-coai/Safety-Prompts

- 摘要：中文安全提示评估和提高LLM的安全性。该存储库包含10万条中文安全场景提示和ChatGPT响应，涵盖各种安全场景和命令攻击。它可以用于对模型安全性进行综合评估和改进，以及增强模型的安全知识，使模型输出与人类价值观保持一致。

- 数据生成模型：`GPT-3.5`

## HuggingFaceH4/stack-exchange-preferences

- 摘要：该数据集包含来自 Stack Overflow Data Dump 的问题和答案，用于偏好模型训练。
- 数据规模：10.8 M

## stanfordnlp/SHP

- Huggingface Hub： https://huggingface.co/datasets/stanfordnlp/SHP

- 摘要：每个示例都是一个 Reddit 帖子，其中包含一个问题/说明以及该帖子的一对顶级评论，其中一条评论更受到 Reddit 用户（集体）的青睐。
- 数据规模：
  - **385K**

## [ultrafeedback_binarized](https://huggingface.co/datasets/YeungNLP/ultrafeedback_binarized)

- huggingfaceHub: https://huggingface.co/datasets/YeungNLP/ultrafeedback_binarized
- 摘要：英文偏好数据集，可用于DPO训练
- 规模：

## UltraFeedback

- Github: https://github.com/OpenBMB/UltraFeedback

- Huggingface Hub : https://huggingface.co/datasets/openbmb/UltraFeedback

### 特征

- **规模**：UltraFeedback 包含 64k 提示、256k 响应和高质量反馈。 RLHF 研究人员可以进一步构建大约 34 万个比较对来训练他们的奖励模型。
- **多样性**：作为偏好数据集，多样性是UltraFeedback的核心要求。我们从各种来源收集提示，并查询各种最先进的开源和享有盛誉的模型。为了进一步增加多样性，我们打算选择不同的基础模型，即LLaMA、Falcon、StarChat、MPT、GPT和Bard。我们还应用各种原理来刺激模型以不同的方式完成指令。
- **高密度**：UltraFeedback 提供数字和文本反馈。此外，我们编写了细粒度的注释文档来帮助对各个维度的响应进行评分

### 指令采样

我们从 6 个公开可用的高质量数据集中抽取了 63,967 条指令。我们包含来自 TruthfulQA 和 FalseQA 的所有指令，从 Evol-Instruct 随机采样 10k 指令，从 UltraChat 随机采样 10k，从 ShareGPT 随机采样 20k。对于FLAN，我们采用分层采样策略，从“CoT”子集中随机采样3k指令，而对其他三个子集每个任务采样10条指令，不包括那些指令过长的子集。

```json
{
    "evol_instruct": 10000,
    "false_qa": 2339,
    "flan": 20939,
    "sharegpt": 19949,
    "truthful_qa": 811,
    "ultrachat": 9929
}
```

## cohere-zh

- HuggingFace Hub: https://huggingface.co/datasets/Cohere/miracl-zh-queries-22-12

- 摘要： cohere-zh 里面对于每条query包含了positive_passages和negative_passages两个部分，positive部分可以视为chosen，negative部分视为rejected.
- 数据规模：

## OpenAI Summarize

- HuggingFace Hub： https://huggingface.co/datasets/openai/summarize_from_feedback

- 摘要：该数据集是一个包含人类反馈的集合，这些反馈被用来训练一个奖励模型（reward model）。在自然语言处理（NLP）任务，如文本摘要（summarization）中，奖励模型可以帮助模型学习如何生成更符合人类偏好的输出。
- 数据规模： 93K

## OpenAI WebGPT

- HuggingFace Hub：https://huggingface.co/datasets/openai/webgpt_comparisons

- 摘要： Data set used in WebGPT paper. Used for training reward model in RLHF.

数据规模： 19,578 pairs

## Pangu RLhf

Github : https://github.com/sunzeyeah/RLHF

支持 RLHF 3大流程: SFT、Reward Model和RLHF， 在Pangu 和 ChatGLM 上进行了训练实验。