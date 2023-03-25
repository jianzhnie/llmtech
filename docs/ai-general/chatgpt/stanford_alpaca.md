# Alpaca：一个强大的、可复制的指令遵循模型

> 学界或许没有业界的算力优势，但可以使用 self-instruct 方法直面大规模语言模型的挑战。

随着大规模语言模型的日渐强大，人们对 AI 模型提出了伦理道德方面的更高要求。业界在模型规模扩展方面具有算力资源优势，但要想让模型更规范、可靠，需要学术界的努力。

近日，斯坦福基于 Meta 的 LLaMA 7B 模型微调出一个新模型 Alpaca。该研究让 OpenAI 的 text-davinci-003 模型以 self-instruct 方式生成 52K 指令遵循（instruction-following）样本，以此作为 Alpaca 的训练数据。研究团队已将训练数据、生成训练数据的代码和超参数开源，后续还将发布模型权重和训练代码。

- 项目地址：https://github.com/tatsu-lab/stanford_alpaca
- 试用地址：https://alpaca-ai-custom6.ngrok.io/

实验结果表明，Alpaca 的很多行为都与 text-davinci-003 类似。也就是说，只有 7B 参数的轻量级模型 Alpaca 性能可媲美 GPT-3.5 这样的超大规模语言模型。

我们来看一下 Alpaca 模型是如何做到的.

## 训练方法

在学术界的预算条件下，训练高质量的指令遵循模型面临两个重要挑战：

- 强大的预训练语言模型
- 高质量的指令遵循数据

Meta 最近发布的 LLaMA 系列模型解决了第一个挑战。对于第二个挑战，2022 年底的 self-instruct 论文提出使用现有的强大语言模型自动生成指令数据。论文地址：https://arxiv.org/abs/2212.10560

按照这种方法，Alpaca 使用 LLaMA 7B 模型的监督学习在 text-davinci-003 以 self-instruct 方式生成的 52K 指令遵循样本上进行微调。

![羊驼管道](https://crfm.stanford.edu/static/img/posts/2023-03-13-alpaca/alpaca_main.jpg)

> self-instruct 方法概览

Alpaca 的研究团队首先使用 self-instruct 种子集中的 175 个人工编写的指令输出（instruction-output）对，然后用该种子集作为 in-context 样本 prompt text-davinci-003 来生成更多指令。该研究通过简化生成 pipeline 改进了 self-instruct 方法，并显著降低了成本。

该研究共生成了 52K 个不同的指令和相应的输出作为训练数据，其中使用了 OpenAI 开放的 API，成本不到 500 美元。由于研究团队已将训练数据开源。

有了这个指令遵循数据集，该研究下一步使用 Hugging Face 的训练框架微调了 LLaMA 模型，并利用了 FSDP（Fully Sharded Data Parallel）和混合精度训练等技术。成本方面，在 8 个 80GB A100 上微调一个 7B LLaMA 模型需要 3 个小时，这对大多数云计算提供商来说成本不到 100 美元。

## Fine-tuning

我们使用具有以下超参数的标准 Hugging Face 训练代码微调我们的模型：

| 超参数   | 价值 |
| -------- | ---- |
| 批量大小 | 128  |
| 学习率   | 2e-5 |
| 纪元     | 3个  |
| 最长长度 | 512  |
| 重量衰减 | 0    |

鉴于 Hugging Face 尚未正式支持 LLaMA 模型，我们通过从特定的分支（即要合并的[PR](https://github.com/huggingface/transformers/pull/21955) ）安装它来使用 Hugging Face 的变形金刚库对 LLaMA 进行微调。我们安装的特定提交的哈希是`68d640f7c368bcaaaecfc678f11908ebbd3d6176`.

要重现我们对 LLaMA 的微调运行，首先安装要求

```python
pip install -r requirements.txt
```

然后，安装 Hugging Face 的Transformer的特定分支。

下面是一个命令，用于在 FSDP 模式下使用我们的数据集在具有 4 个 A100 80G GPU 的机器上微调 LLaMA-7B `full_shard`。我们能够使用**Python 3.10**使用以下命令重现与我们在演示中托管的模型质量相似的模型。替换`<your_random_port>`为您服务器的端口、`<your_path_to_hf_converted_llama_ckpt_and_tokenizer>`转换后的LLAMA模型检查点和分词器的路径（遵循 PR 中的说明）以及`<your_output_dir>`您要存储输出的位置。

```python
torchrun --nproc_per_node=4 --master_port=<your_random_port> train.py \
    --model_name_or_path <your_path_to_hf_converted_llama_ckpt_and_tokenizer> \
    --data_path ./alpaca_data.json \
    --bf16 True \
    --output_dir <your_output_dir> \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LLaMADecoderLayer' \
    --tf32 True
```

### 警告

`fsdp_transformer_layer_cls_to_wrap`必须设置为特定解码器层的名称。LLaMA Hugging Face PR 不稳定。较早的提交使用其解码器层的名称`LLaMADecoderLayer`（我们的代码的提交哈希基于此）。最近的提交使用`LlamaDecoderLayer`（注意小的大小写差异）。不设置`fsdp_transformer_layer_cls_to_wrap`正确的名称将导致训练速度急剧下降。

### 训练 OPT 模型

同样的脚本也适用于 OPT 微调。这是微调 OPT-6.7B 的示例

```python
torchrun --nproc_per_node=4 --master_port=<your_random_port> train.py \
    --model_name_or_path "facebook/opt-6.7b" \
    --data_path ./alpaca_data.json \
    --bf16 True \
    --output_dir <your_output_dir> \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'OPTDecoderLayer' \
    --tf32 True
```

请注意，给定的训练脚本旨在简单易用，并没有特别优化。要在更多 GPU 上运行，您可能更愿意调低`gradient_accumulation_steps`以保持 128 的全局批量大小。全局批量大小尚未经过优化测试。