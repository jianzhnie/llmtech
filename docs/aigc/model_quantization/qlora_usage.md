# QLora源码分析

qlora三板斧：

- NF4 Quantization(4bit量化)：一种int4量化方法，思想来自信息论。NF4量化可以使得量化后的数据和量化前具有同等的数据分布。就是NF4量化后，权重信息损失少，模型的整体精度损失会减少。
  - 其实就是在bnb_4bit_quant_type='nf4'，这个设置也可以是bnb_4bit_quant_type='fp4'。哪个效果好用哪个？
- Double Quantization ：对第一次量化后的常量做二次量化，减小模型存储消耗。
  - bnb_4bit_use_double_quant=True，表示使用二次量化，为False表示不用二次量化
- Paged optimizers : 使用NVIDIA统一内存功能，该功能在CPU和GPU之间进行自动page对page传输，以便在GPU偶尔OOM的情况下进行. 可以从现象上理解成训练过程中出现偶发OOM时能够自动处理，保证训练正常进行下去。
  - 通过设置--optim paged_adamw_32bit，来使用内存页面优化。这是transformers的Trainer中自带的一个超参数。大模型微调使用这个操作还是可以，但不是必须的。

## 量化代码

模型量化部分代码如下：

```python
model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path='/name/or/path/to/your/model',
        load_in_4bit=True, # 模型压缩的关键
        device_map={"": 0},  # 调用GPU，并使用accelerate加载模型
        max_memory=max_memory,
        torch_dtype=torch.bfloat16, # 注意bfloat16只有比较新的GPU架构才能使用
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16, # 计算时采用BF16最快，默认为FP32
            bnb_4bit_use_double_quant=True, # 二次量化
            bnb_4bit_quant_type='nf4' # 量化时使用nf4最优，也可以使用fp4
        )
    )
# 若想使用Paged Optimizer，则在QLora.py中调用 --optim paged_adamw_32bit
```

其实int8微调训练已经很惊艳了，这使用fp4和nf4进行微调，确实够进步。期待nf3或nf2甚至binary训练。LLM的布尔网络是未来吗？

官方也给出了llama-7b模型和guanaco-7b adapter modules的合成示例：

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

model_name = "decapoda-research/llama-7b-hf"
adapters_name = 'timdettmers/guanaco-7b'

print(f"Starting to load the model {model_name} into memory")

m = AutoModelForCausalLM.from_pretrained(
    model_name,
    #load_in_4bit=True,
    torch_dtype=torch.bfloat16,
    device_map={"": 0}
)
m = PeftModel.from_pretrained(m, adapters_name)
m = m.merge_and_unload()
tok = LlamaTokenizer.from_pretrained(model_name)
tok.bos_token_id = 1

stop_token_ids = [0]

print(f"Successfully loaded the model {model_name} into memory")
```

QLora方法在实际使用中要注意以下几点：

- load_in_4bit=True的情况下模型推理能力较慢。4bit推理还未能与4bit矩阵乘法结合
- bnb_4bit_compute_type='fp16'会导致量化模型训练不稳定。
- 要设置 tokenizer.bos_token_id = 1

## 训练参数

```py
from collections import defaultdict
import copy
import json
import os
from os.path import exists, join, isdir
from dataclasses import dataclass, field
import sys
from typing import Optional, Dict, Sequence
import numpy as np
from tqdm import tqdm
import logging

import bitsandbytes as bnb # 核心库，用来完成int4模型加载和训练

import torch
import transformers
from torch.nn.utils.rnn import pad_sequence # 数据预处理，做文本数据补齐用的
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    Seq2SeqTrainer,
    BitsAndBytesConfig # 需要安装bitsandbytes这个库先，transformer库非常及时的更新了这个配置接口
)
from datasets import load_dataset
import evaluate
import nltk
```

```python
# 核心库，除了lora，还有很多其他功能，代码也非常简洁，适合step-in学习
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    PeftModel
)
from peft.tuners.lora import LoraLayer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
```

```python
torch.backends.cuda.matmul.allow_tf32 = True

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
torch.backends.cuda.matmul.allow_tf32 = True
```

> TF32 是一种截短的 Float32 数据格式，将 FP32 中 23 个尾数位截短为 10 bits，而指数位仍为 8 bits，总长度为 19 (=1 + 8 + 10) bits。 为什么选择 1 + 8 + 10 这个配置？ 按照 NVIDIA 官方的说法，TF32 保持了与 FP16 同样的精度（尾数位都是 10 位），同时还保持了 FP32 的动态范围（指数位都是 8 位）。

下面是冗长的transformers的参数部分。读者还是有必要了解深度学习的三个要素，数据、模型和训练。这里分别对应三个参数集合。当然还有生成参数。下面简要写下他们架构，然后深入代码进行详细分析。

- ModelArguments，包含模型相关的参数，自定义
- DataArguments，包含数据相关的参数，自定义
- TrainingArguments，包含模型训练的参数，大部分默认，少部分需要重新设置，该参数会输入到Trainer进行初始化训练器
- GenerationArguments，包含文本生成的参数，四种搜索方式要熟悉。贪婪搜索、集束搜索，top-k和top-p搜索等

### 模型

```python
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="EleutherAI/pythia-12b"
    )
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."}
    )
```

模型名称或路径：

-  建议使用Llama，这个效果是比较好的。
- 当然，如果要考虑商用的话，可以使用mpt，这个可以商用
- 默认使用EleutherAI/pythia-12b模型，但让也可以使用其他的Decoder-only模型
- 也可以换成EleutherAI/gpt-neo-2.7B、EleutherAI/gpt-j-6b等，可以进行测试。包括训练和推理

### 数据

```python
@dataclass
class DataArguments:
    # 验证数据集的尺寸，也就是数量
    eval_dataset_size: int = field(
        default=1024, metadata={"help": "Size of validation dataset."}
    )
    # 最大训练数据样本的数量。主要是为了快速调试训练代码
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    # 与max_train_samples类似，主要是为了快速调试训练代码
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    # 最大文本输入的最大长度。如果source文本token长度超过该值，需要做文本的截断
    source_max_len: int = field(
        default=1024,
        metadata={"help": "Maximum source sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    # 标签文本的最大长度，如果target文本token长度超过该值，需要做文本的截断
    target_max_len: int = field(
        default=256,
        metadata={"help": "Maximum target sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    dataset: str = field(
       # 微调数据集是alpaca，那么可以试试中文的效果。Llama、Bloom和OPT，或者MPT等等
        default='alpaca',
        metadata={"help": "Which dataset to finetune on. See datamodule for options."}
    )
```

### 训练参数

```python
@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    '''
        训练参数相对特殊点，因为要大量使用transforemrs中Trainer的默认参数。
        只要针对微调代码，对该参数中的部分参数进行调整即可，超参数手工调整主要在这里
    '''
    cache_dir: Optional[str] = field(
        default=None
    ) # 缓存目录
    train_on_source: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to train on the input in addition to the target text."}
    ) # 是否在source文本上进行GPT LM微调。默认是False，这部分文本对应的token会在label中设置为-100的标签
      # -100是因为在做CrossEntropy计算的时候会将-100的值对应的token忽略掉
    mmlu_split: Optional[str] = field(
        default='eval',
        metadata={"help": "The MMLU split to run on"}
    ) # mmlu数据的分片名称
    mmlu_dataset: Optional[str] = field(
        default='mmlu-fs',
        metadata={"help": "MMLU dataset to use: options are `mmlu-zs` for zero-shot or `mmlu-fs` for few shot."}
    ) # mmlu数据集的默认名称，`mmlu-zs` for zero-shot or `mmlu-fs` for few shot.
    do_mmlu_eval: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to run the MMLU evaluation."}
    ) # 是否使用MMLU评估
    max_mmlu_samples: Optional[int] = field(
        default=None,
        metadata={"help": "If set, only evaluates on `max_mmlu_samples` of the MMLU dataset."}
    ) # 如果设置了，那么就会在MMLU数据集的最大样本集上做评估
    mmlu_source_max_len: int = field(
        default=2048,
        metadata={"help": "Maximum source sequence length for mmlu."}
    ) # mmlu数据集source文本的最大长度（是字符长度还是token长度，这个去代码中找线索吧）
    full_finetune: bool = field(
        default=False,
        metadata={"help": "Finetune the entire model without adapters."}
    ) # 不使用adapter进行全微调（不适用Lora或qlora？）
    adam8bit: bool = field(
        default=False,
        metadata={"help": "Use 8-bit adam."}
    ) # 使用8-bit的adam，是否可以调整为LION或Sophia，甚至deepspeed还提供了多个1-bit优化器选择
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    ) # 是否使用二次量化
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    ) # 量化类型，可以选择`fp4`或`nf4`
    bits: int = field(
        default=4,
        metadata={"help": "How many bits to use."}
    ) # 使用的位宽，默认为4。
    # 下面是Lora参数
    lora_r: int = field(
        default=64,
        metadata={"help": "Lora R dimension."}
    ) # lora中A矩阵的列数量和B矩阵的行数量
    lora_alpha: float = field(
        default=16,
        metadata={"help": " Lora alpha."}
    ) # 缩放因子
    lora_dropout: float = field(
        default=0.0,
        metadata={"help":"Lora dropout."}
    ) # dropout，一种正则化方法，可以模仿集成学习
    max_memory_MB: int = field(
        default=80000,
        metadata={"help": "Free memory per gpu."}
    ) # 每个GPU上可使用的显存大小，以MB为单位。默认是A100高端版本的80GB
    report_to: str = field(
        default='none',
        metadata={"help": "To use wandb or something else for reporting."}
    ) # 使用wandb记录微调过程

    # 上面是自定义的trainer参数，下面是默认的trainer参数，对部分参数进行了默认设置
    # log和checkpoint的保存目录
    output_dir: str = field(default='./output',
                            metadata={"help": 'The output dir for logs and checkpoints'})
    # 使用nvidia的分页机制优化器，可以在偶尔OOM的情况，让模型继续训练下去。
    # 做了这么多优化，显存占用变少了，速度是否会变慢呢？
    optim: str = field(default='paged_adamw_32bit',
                       metadata={"help": 'The optimizer to be used'})
    # 单块设备（GPU）上参与进行模型训练的批尺寸大小
    per_device_train_batch_size: int = field(default=1,
                                             metadata={"help": 'The training batch size per GPU. Increase for better speed.'})
    # 单块设备（GPU）上参与进行模型验证的批尺寸大小
    gradient_accumulation_steps: int = field(default=16,
                                             metadata={"help": 'How many gradients to accumulate before to perform an optimizer step'})
    # 训练的最大步数
    max_steps: int = field(default=1000,
                           metadata={"help": 'How many optimizer update steps to take'})
    # AdamW优化器的L2权重衰减因子。
    weight_decay: float = field(default=0.0,
                                metadata={"help": 'The L2 weight decay rate of AdamW'}) # use lora dropout instead for regularization if needed
    # 因为是qlora训练，对随机初始化和全0初始化的神经网络进行训练，因此学习率较高
    # 在1e-4级别的学习率
    learning_rate: float = field(default=0.0002,
                                 metadata={"help": 'The learnign rate'})
    # 移除datasets中未被使用的列。指输入train_dataset和eval_dataset
    remove_unused_columns: bool = field(default=False,
                                        metadata={"help": 'Removed unused columns. Needed to make this codebase work.'})
    # 梯度截断因子
    max_grad_norm: float = field(default=0.3,
                                 metadata={"help": 'Gradient clipping max norm. This is tuned and works well for all models tested.'})
    # 梯度检查，设置为True，来减少显存占用。
    # 显存这么紧张，肯定是要设置为True，但是运行时间就会提升
    gradient_checkpointing: bool = field(default=True,
                                         metadata={"help": 'Use gradient checkpointing. You want to use this.'})
    # 是否进行训练，那肯定是要的
    do_train: bool = field(default=True, metadata={"help": 'To train or not to train, that is the question?'})
    # 调度策略，一般是带warmup的余弦衰减
    lr_scheduler_type: str = field(default='constant', metadata={"help": 'Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis'})
    warmup_ratio: float = field(default=0.03, metadata={"help": 'Fraction of steps to do a warmup for'})
    logging_steps: int = field(default=10, metadata={"help": 'The frequency of update steps after which to log the loss'})
    group_by_length: bool = field(default=True, metadata={"help": 'Group sequences into batches with same length. Saves memory and speeds up training considerably.'})
    save_strategy: str = field(default='steps', metadata={"help": 'When to save checkpoints'})
    save_steps: int = field(default=250, metadata={"help": 'How often to save a model'})
    save_total_limit: int = field(default=40, metadata={"help": 'How many checkpoints to save before the oldest is overwritten'})
```

> MMLU benchmark，一个基准数据集，包括来自于STEM、人文、社科等57个学科的选择题，它用于测试LLM的世界知识和问题解答的能力 OpenAI表示，他们花了6个月的时间来使用对抗性测试程序，以及通过ChatGPT的经验教训迭代调整GPT-4，从而在真实性和可控性等方面取得了有史以来最好的结果。

### 生成参数

```python
@dataclass
class GenerationArguments:
    # For more hyperparameters check:
    # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
    # Length arguments
    # 最大的新生成的token数量
    max_new_tokens: Optional[int] = field(
        default=256,
        metadata={"help": "Maximum number of new tokens to be generated in evaluation or prediction loops"
                          "if predict_with_generate is set."}
    )
    # 最少的新生成的token数量
    min_new_tokens : Optional[int] = field(
        default=None,
        metadata={"help": "Minimum number of new tokens to generate."}
    )
    # Generation strategy
    # 是否采样
    do_sample: Optional[bool] = field(default=False)
    # 集束搜索的数量
    num_beams: Optional[int] = field(default=1)
    # 集束搜索的组数量
    num_beam_groups: Optional[int] = field(default=1)
    # 惩罚因子
    penalty_alpha: Optional[float] = field(default=None)
    # 是否使用cache
    use_cache: Optional[bool] = field(default=True)
    # Hyperparameters for logit manipulation
    # softmax函数的温度因子，来调节输出token的分布
    temperature: Optional[float] = field(default=1.0)
    # top_k随机搜索中的k个最高概率选择
    top_k: Optional[int] = field(default=50)
    # 核采样参数，top_p最高的前n个（n是变化）概率和为p，从这些n个候选token中随机采样
    top_p: Optional[float] = field(default=1.0)
    # 典型p值
    typical_p: Optional[float] = field(default=1.0)
    # 丰富性惩罚因子
    diversity_penalty: Optional[float] = field(default=0.0)
    # 重复性惩罚因子
    repetition_penalty: Optional[float] = field(default=1.0)
    # 长度惩罚因子
    length_penalty: Optional[float] = field(default=1.0)
    # 没有ngram重复的尺度大小
    # 一般随机采样的丰富性够了，所以一般不会设置，如果重复很多则设置为2是比较好的选择
    no_repeat_ngram_size: Optional[int] = field(default=0)
```

### 超参数设置

```python
args.model_name_or_path = "decapoda-research/llama-7b-hf"
args.output_dir = "./output"
args.dataset = "alpaca"
args.do_train = True
args.do_eval = True
args.do_mmlu_eval = True
args.source_max_len = 384
args.target_max_len = 128
args.per_device_train_batch_size = 4
args.per_device_eval_batch_size = 4
args.gradient_accumulation_steps = 4
args.logging_steps = 10
args.max_steps = 2000
args.save_strategy = "steps"
args.data_seed = 402
args.save_steps = 1000
args.save_total_limit = 40
args.evaluation_strategy = "steps"
args.eval_dataset_size = 1024
args.max_eval_samples = 1000
args.eval_steps = 1000
args.optim = "paged_adamw_32bit"
```

## 关键函数

### find_all_linear_names

```python
def find_all_linear_names(args, model):
    '''
     如果args.bits是4，使用bitsandbytes库中的bnb.nn.Linear4bit层；
     如果args.bits是8，使用bitsandbytes库中的bnb.nn.Linear8bitLt层；
     否则，使用torch.nn.Linear层；
     并记录下这些层的名称，保存在lora_module_names集合中。
    '''
    cls = bnb.nn.Linear4bit if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            # 只保留最后的名称，前缀不保留
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    # 语言模型的输出头，需要16bit精度
    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)
```

### SavePeftModelCallback

```python
class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        print('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir,
                                             f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

    def on_save(self, args, state, control, **kwargs):
        '''
            ModelCheckpoing达到调用条件时，调用self.save_model函数保存模型
        '''
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        '''
            在模型训练结束时保存模型，并且加上时间。还真是讲究。
            这个加上时间信息还是蛮有用的。
        '''
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)

        touch(join(args.output_dir, 'completed'))
        self.save_model(args, state, kwargs)
```

### get_accelerate_model

```python
def get_accelerate_model(args, checkpoint_dir):
    '''
        获取accelerate模型。
    '''
    # 获取GPU数量
    n_gpus = torch.cuda.device_count()
    # 设置的单卡占用最大显存，其实应该全部用起来
    # 但是考虑到在训练过程中，占用的显存是会变化的，因此不能设置太大
    max_memory = f'{args.max_memory_MB}MB'
    # 设置字典，后面在加载模型的时候会用到
    max_memory = {i: max_memory for i in range(n_gpus)}
    # 如果使用全参数微调的话，那么args.bits必须是16或32
    if args.full_finetune: assert args.bits in [16, 32]
    print(f'loading base model {args.model_name_or_path}...')
    # 如果使用args.fp16计算，那么4bit训练计算时使用torch.float16进行计算
    # 如果使用args.bf16，则4bit训练计算时使用torch.bfloat16进行计算
    # 否则4bit训练计算时使用torch.float32进行计算
    compute_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
    # 最新的transformers已经支持quantization_config，
    # 并且计算时使用另外的数据格式。既要保证内存使用，又要保留计算精度
    # 内存和显存占用时，使用一种格式，运算时使用另外的精度进行
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, # 模型名称
        load_in_4bit=args.bits == 4, # 是否使用4bit加载
        load_in_8bit=args.bits == 8, # 是否使用8bit加载
        device_map='auto', # 自动加载，通过max_memory分配显存
        max_memory=max_memory, # 显存分配方案，还可以使用cpu和硬盘上的内存映射
        # BitsAndBytesConfig设置存储格式和计算格式，以及优化方式
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=args.bits == 4, # 是否使用4bit加载
            load_in_8bit=args.bits == 8, # 是否使用8bit加载
            llm_int8_threshold=6.0, # int8的门限，这个是做什么的
            llm_int8_has_fp16_weight=False, # int8的LLM，是否包含fp16的权重
            bnb_4bit_compute_dtype=compute_dtype, # 计算时使用的数据类型
            bnb_4bit_use_double_quant=args.double_quant, # 是否进行双重量化
            bnb_4bit_quant_type=args.quant_type # {'fp4', 'nf4'} # 4bit的量化格式，只有两种选择
        ),
        # 实际的torch数据类型
        # args.fp16对应torch.float32
        # args.bf16对应torch.bfloat16
        # 否则为torch.float32
        torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
        # 是否依赖远端代码
        trust_remote_code=args.trust_remote_code,
    )

    # 如果计算类型为torch.float16并且args.bits==4，也就是4bit量化模型时，进行如下操作。
    if compute_dtype == torch.float16 and args.bits == 4:
        # 得到模型的计算能力的最大值和最小值，分别对应major和minor
        major, minor = torch.cuda.get_device_capability()
        if major >= 8:
            # 只有major>=8时的GPU才支持bfloat16格式，可以使用参数--bf16来加速训练
            print('='*80)
            print('Your GPU supports bfloat16, you can accelerate training with the argument --bf16')
            print('='*80)

    # 设置两个和并行操作相关的参数
    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)

    # 找到指定规则下的线性层名称
    modules = find_all_linear_names(args, model)

    model.config.torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))

    if not args.full_finetune:
        # 如果不是全参数微调，那么使用如下函数将模型的参数再进行处理，方便进行int8数据格式的训练微调
        model = prepare_model_for_int8_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing
        )

    if args.gradient_checkpointing:
        # 是否使用梯度检查
        model.gradient_checkpointing_enable()

    # 配置lora中的参数
    config = LoraConfig(
        r=args.lora_r, # lora层A矩阵的列大小和B矩阵的行大小
        lora_alpha=args.lora_alpha, # 缩放因子
        target_modules=modules, # 需要进行lora网络操作的模块名称列表
        lora_dropout=args.lora_dropout, # 是否使用dropout，一种正则化操作
        bias="none", # 不对偏差参数进行处理
        task_type="CAUSAL_LM", # 模型名称，一种标记
    )

    if not args.full_finetune:
        # 如果不进行全参数微调，下面进行lora层的添加
        if checkpoint_dir is not None:
            # 如果已有lora模型，那么在该lora模型基础上加载。也就是lora预训练模型
            print("Loading adapters from checkpoint.")
            # 加载peft模型
            model = PeftModel.from_pretrained(model, join(checkpoint_dir, 'adapter_model'))
            # 计算lora层的参数数量
            for name, p in model.named_parameters():
                if 'lora' in name:
                    print(name, p.sum())
        else:
            # 否则，随机初始化lora层，构建lora模型
            print(f'adding LoRA modules...')
            model = get_peft_model(model, config)

    if args.gradient_checkpointing:
        # 梯度检查
        if hasattr(model, "enable_input_require_grads"):
            # 使能输入的梯度
            model.enable_input_require_grads()
        else:
            # 使能输入的梯度
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    for name, module in model.named_modules():
        # 设置一些层的数据类型
        if isinstance(module, LoraLayer):
            # 如果模型是LoraLayer
            if args.bf16:
                # 且使用args.bf16格式，但是只有某些GPU支持
                module = module.to(torch.bfloat16)
        if 'norm' in name:
            # 如果是归一化层，transformers中有很多layernorm层
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            # 如果lm_head或embed_tokens等，也就是输出层和输入层
            if hasattr(module, 'weight'):
                # 而且是权重（非bias参数）
                if args.bf16 and module.weight.dtype == torch.float32:
                    # 如果使用bf16格式，且模型权重的数据类型为torch.float32，
                    # 则将该层设置为bfloat16数据格式
                    module = module.to(torch.bfloat16)
    return model
```

### print_trainable_parameters

```python
def print_trainable_parameters(args, model):
    """
    Prints the number of trainable parameters in the model.
    打印模型的可训练参数的数量和占比
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    if args.bits == 4: trainable_params /= 2
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable: {100 * trainable_params / all_param}")
```

### smart_tokenizer_and_embedding_resize

```python
def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict, # 特殊token字典
    tokenizer: transformers.PreTrainedTokenizer, # 分词器
    model: transformers.PreTrainedModel, # 模型
):
    """Resize tokenizer and embedding.
    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.

    改变tokenizer和embedding的尺寸。
    一般需要将tokenizer和embedding的尺寸设置为64的倍数，方便GPU加速。
    """
    # 添加特殊token字典，并且得到新加入字典的token数量
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    # 更改token_embeddings的尺寸
    model.resize_token_embeddings(len(tokenizer))

    # 下面的操作非常有意思，可以仔细看看
    if num_new_tokens > 0:
        # 分别得到输入和输出embeddings
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        # 下面的操作实现使用已训练好的embedding的均值，来初始化新token对应的embedding
        # input_embeddings的已训练好的embedding的均值，保持embedding的shape
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        # output_embeddings的已训练好的embedding的均值，保持embedding的shape
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        # 分别给input_embeddings和output_embeddings的新token对应的embedding赋值
        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
```

### DataCollatorForCausalLM

```python
@dataclass
class DataCollatorForCausalLM(object):
    '''
    数据预处理函数，适用于Causal Language Model。
    '''
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        # alpaca instruction learning的数据集格式
        # 获取输入文本列表
        sources = [example['input'] for example in instances]
        # 获取输出文本列表，并在每个输出文本后加上eos_token
        targets = [f"{example['output']}{self.tokenizer.eos_token}" for example in instances]
        # Tokenize，分别对输入和输出进行分词
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
        )
        # 输出如果截断了，那么结束token就没有了。
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        # Build the input and labels for causal LM
        input_ids = [] # 输入的token id
        labels = []  # 标签
        for tokenized_source, tokenized_target in zip(
            tokenized_sources_with_prompt['input_ids'],
            tokenized_targets['input_ids']
        ):
            # tokenized_sources_with_prompt['input_ids'], tokenized_targets['input_ids']
            # 上述两个是二维列表
            # tokenized_source, tokenized_target都是一维列表
            if not self.predict_with_generate:
                # 进行训练
                input_ids.append(torch.tensor(tokenized_source + tokenized_target))
                if not self.train_on_source:
                    # 是否输入也参与标签计算，这里是不参与，那么需要做掩码处理
                    labels.append(
                        torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source))]
                                     + copy.deepcopy(tokenized_target))
                    )
                else:
                    labels.append(torch.tensor(copy.deepcopy(tokenized_source + tokenized_target)))
            else:
                # 进行预测，只有输入文本，没有输出文本
                input_ids.append(torch.tensor(tokenized_source))
        # Apply padding
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX) if not self.predict_with_generate else None
        data_dict = {
            'input_ids': input_ids,
            'attention_mask':input_ids.ne(self.tokenizer.pad_token_id), # 这个操作还真是方便
        }
        if labels is not None:
            data_dict['labels'] = labels
        return data_dict
```

### extract_unnatural_instructions_data

```python
def extract_unnatural_instructions_data(examples, extract_reformulations=False):
    # 对非指定格式的数据，进行处理调整
    out = {
        'input': [],
        'output': [],
    }

    for example_instances in examples['instances']:
        for instance in example_instances:
            out['input'].append(instance['instruction_with_input'])
            out['output'].append(instance['output'])

    if extract_reformulations:
        for example_reformulations in examples['reformulations']:
            if example_reformulations is not None:
                for instance in example_reformulations:
                    out['input'].append(instance['instruction_with_input'])
                    out['output'].append(instance['output'])

    return out
```

### Format_prompt_and_input

```python
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: "
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: "
    ),
}
```

### extract_alpaca_dataset

```python
def extract_alpaca_dataset(example):
    if example.get("input", "") != "":
        prompt_format = PROMPT_DICT["prompt_input"]
    else:
        prompt_format = PROMPT_DICT["prompt_no_input"]
    return {'input': prompt_format.format(**example)}
```

### make_data_module

```python
def make_data_module(tokenizer: transformers.PreTrainedTokenizer, args) -> Dict:
    """
    创建数据集进行有监督微调，也就是SFT，RLHF的第一步。
    Make dataset and collator for supervised fine-tuning.

    数据集有两列，分别是{ `input`, `output` }
    Datasets are expected to have the following columns: { `input`, `output` }

    可选的数据集包括下面这些：
    Available datasets to be selected with `dataset` argument:
        - alpaca, 52002 examples
        - alpaca cleaned, 51942 examples，清洗后的alpaca数据
        - chip2 (OIG), 210289 examples
        - self-instruct, 82612 examples
        - hh-rlhf (Anthropic), 160800 examples
        - longform, 23.7k examples

    Coming soon:
        - unnatural instructions core, 66010 examples
        - unnatural instructions full, 240670 examples
        - alpaca-gpt4, 52002 examples
        - unnatural-instructions-gpt4, 9000 examples
        - oa-rlhf (OpenAssistant) primary message tree only, 9209 examples
        - oa-rlhf-assistant (OpenAssistant) all assistant  replies with ranking
        - supernatural-instructions, 69624 examples (same as paper with 100 ex/task more can be used)
        - flan (FLAN v2), up to 20M examples available

    Not Available:
        - vicuna, not released at the moment.
    """
    # Load dataset.
    # Alpaca
    if args.dataset == 'alpaca':
        dataset = load_dataset("tatsu-lab/alpaca")
        dataset = dataset.map(extract_alpaca_dataset, remove_columns=['instruction'])
    # Alpaca clean
    elif args.dataset == 'alpaca-clean':
        dataset = load_dataset("yahma/alpaca-cleaned")
        dataset = dataset.map(extract_alpaca_dataset, remove_columns=['instruction'])
    # Chip2
    elif args.dataset == 'chip2':
        dataset = load_dataset("laion/OIG", data_files='unified_chip2.jsonl')
        dataset = dataset.map(lambda x: {
            'input': x['text'].split('\n<bot>: ')[0].replace('<human>: ', ''),
            'output': x['text'].split('\n<bot>: ')[1],
        }, remove_columns=['text', 'metadata'])
    # Self Instruct
    elif args.dataset == 'self-instruct':
        dataset = load_dataset("yizhongw/self_instruct", name='self_instruct')
        for old, new in [["prompt", "input"], ["completion", "output"]]:
            dataset = dataset.rename_column(old, new)
    # Anthropic rlhf
    elif args.dataset == 'hh-rlhf':
        dataset = load_dataset("Anthropic/hh-rlhf")
        dataset = dataset.map(lambda x: {
            'input': '',
            'output': x['chosen']
        }, remove_columns=['chosen', 'rejected'])
    # LongForm
    elif args.dataset == 'longform':
        dataset = load_dataset("akoksal/LongForm")
    elif args.dataset == 'vicuna':
        raise NotImplementedError("Vicuna data was not released.")
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented yet.")

    # Split train/eval, reduce size
    if args.do_eval or args.do_predict:
        if 'eval' in dataset:
            eval_dataset = dataset['eval']
        else:
            print('Splitting train dataset in train and validation according to `eval_dataset_size`')
            dataset = dataset["train"].train_test_split(
                test_size=args.eval_dataset_size, shuffle=True, seed=42
            )
            eval_dataset = dataset['test']
        if args.max_eval_samples is not None and len(eval_dataset) > args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        if args.group_by_length:
            eval_dataset = eval_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})
    if args.do_train:
        train_dataset = dataset['train']
        if args.max_train_samples is not None and len(train_dataset) > args.max_train_samples:
            train_dataset = train_dataset.select(range(args.max_train_samples))
        if args.group_by_length:
            train_dataset = train_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})

    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        source_max_len=args.source_max_len,
        target_max_len=args.target_max_len,
        train_on_source=args.train_on_source,
        predict_with_generate=args.predict_with_generate,
    )
    return dict(
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        predict_dataset=eval_dataset if args.do_predict else None,
        data_collator=data_collator
    )
```

### get_last_checkpoint

```python
def get_last_checkpoint(checkpoint_dir):
    if isdir(checkpoint_dir):
        is_completed = exists(join(checkpoint_dir, 'completed'))
        if is_completed: return None, True # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if isdir(join(checkpoint_dir, filename)) and filename.startswith('checkpoint'):
                max_step = max(max_step, int(filename.replace('checkpoint-', '')))
        if max_step == 0: return None, is_completed # training started, but no checkpoint
        checkpoint_dir = join(checkpoint_dir, f'checkpoint-{max_step}')
        print(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir, is_completed # checkpoint found!
    return None, False # first training
```



## 训练代码

### 参数解析

```python
hfparser = transformers.HfArgumentParser((
    ModelArguments, DataArguments, TrainingArguments, GenerationArguments
))
```

```python
model_args, data_args, training_args, generation_args, extra_args = \
        hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
```

```python
training_args.generation_config = transformers.GenerationConfig(**vars(generation_args))
```

```python
args = argparse.Namespace(
    **vars(model_args), **vars(data_args), **vars(training_args)
)
```

### 加载模型 和 Tokenizer

```python

checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)
print(checkpoint_dir, completed_training)
```

```python
model = get_accelerate_model(args, checkpoint_dir)
```

```python
training_args.skip_loading_checkpoint_weights=True # 设置忽略加载checkpoint权重
model.config.use_cache = False # 不适用缓存
print_trainable_parameters(args, model) # 打印可训练参数数量和占比
print('loaded model') # 后面加载模型，不是在get_accelerate_model中加载了吗？
set_seed(args.seed) # 设置随机种子
trainable params: 79953920.0 || all params: 3660320768 || trainable: 2.184341894267557
```

```python
# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    args.model_name_or_path,
    cache_dir=args.cache_dir,
    padding_side="right",
    use_fast=True,
)
```

```python
if tokenizer.pad_token is None:
    # 如果pad_token是None，则默认的pad_token来替换
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
        tokenizer=tokenizer,
        model=model,
    )

if any(key in args.model_name_or_path for key in ['llama', '7B', '13B', '30B', '65B']):
    # LLaMA tokenizer does not have special tokens set.
    # Add them to prevent them from being parsed into different tokens.
    # Note that these are present in the vocabulary.
    # Note also that `model.config.pad_token_id` is 0 which corresponds to `<unk>` token.

    # 如果是Llama模型，它的分词器是没有特殊token集合的，因此需要手工添加。
    # 那为什么不适用smart_tokenizer_and_embedding_resize函数呢。
    #
    tokenizer.add_special_tokens(
        {
            "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
            "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
            "unk_token": tokenizer.convert_ids_to_tokens(model.config.pad_token_id),
        }
    )
```

```python
model.config.eos_token_id, model.config.bos_token_id, model.config.pad_token_id
```

### 设置 data_module

```python
data_module = make_data_module(tokenizer=tokenizer, args=args)
```

### 设置  Seq2SeqTrainer

```python
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    **{k:v for k,v in data_module.items() if k != 'predict_dataset'},
)
```

```python
# Callbacks
if not args.full_finetune:
    # 不是全微调，对于普通选手，一般不会使用全微调，太耗资源了
    # 一般是使用peft的方法来进行微调
    # peft的lora是一种横向微调的手段，是否可以探索下深度的lora呢？
    # 多个模型的lora，拼接起来，称为新的更深度的模型。
    trainer.add_callback(SavePeftModelCallback)
```

### 设置模型评估

```python
if args.do_mmlu_eval:
    # mmlu的数据集在qlora的github代码库中，也可以重新下载
    # 下面做了简单修改，让代码按照路径加载验证集和测试集
    if args.mmlu_dataset == 'mmlu-zs':
        # mmlu-zs加载验证集和测试集
        # zero-shot性能测试, zs -> zero-shot
        mmlu_dataset = load_dataset("json", data_files={
            'eval': '../input/mmlu-data/mmlu/zero_shot_mmlu_val.json',
            'test': '../input/mmlu-data/mmlu/zero_shot_mmlu_test.json',
        })
        # 移除mmlu_dataset数据集中的'subject'列
        mmlu_dataset = mmlu_dataset.remove_columns('subject')
    # MMLU Five-shot (Eval/Test only)
    elif args.mmlu_dataset == 'mmlu' or args.mmlu_dataset == 'mmlu-fs':
        # five-shot性能测试, fs -> five-shot
        mmlu_dataset = load_dataset("json", data_files={
            'eval': '../input/mmlu-data/mmlu/five_shot_mmlu_val.json',
            'test': '../input/mmlu-data/mmlu/five_shot_mmlu_test.json',
        })
        # mmlu_dataset = mmlu_dataset.remove_columns('subject')
    # 获取mmlu_split的split，翻译为部分模块
    mmlu_dataset = mmlu_dataset[args.mmlu_split]
    if args.max_mmlu_samples is not None:
        # 按照超参数设置，得到部分数据
        mmlu_dataset = mmlu_dataset.select(range(args.max_mmlu_samples))
    # 得到"A"，“B“，”C"和"D"对应的token id
    abcd_idx = [
        tokenizer("A", add_special_tokens=False).input_ids[0],
        tokenizer("B", add_special_tokens=False).input_ids[0],
        tokenizer("C", add_special_tokens=False).input_ids[0],
        tokenizer("D", add_special_tokens=False).input_ids[0],
    ]
    # 加载计算accuracy准确率的metric类
    accuracy = evaluate.load("accuracy")

    class MMLUEvalCallback(transformers.TrainerCallback):
        def on_evaluate(self, args, state, control, model, **kwargs):
            # 首先获取验证数据
            data_loader = trainer.get_eval_dataloader(mmlu_dataset)
            # 输入文本的最大token长度，保存原来的source_max_len
            source_max_len = trainer.data_collator.source_max_len
            # 设置trainer中的source_max_len为mmlu的source_max_len
            trainer.data_collator.source_max_len = args.mmlu_source_max_len
            # 模型只进行推理
            trainer.model.eval()
            # 开始验证过程
            preds, refs = [], []
            loss_mmlu = 0
            for batch in tqdm(data_loader, total=len(data_loader)):
                # 预测步骤，默认参数需要看源代码
                # prediction_step需要模型,batch和prediction_loss_only等参数
                # 其他参数需要进入源码查看
                (loss, logits, labels) = trainer.prediction_step(
                    trainer.model,
                    batch,
                    prediction_loss_only=False,
                )
                # There are two tokens, the output, and eos token.
                for i, logit in enumerate(logits):
                    # 得到batch中labels的非0部分，也就是非-100标记的部分，这部分参与指标计算
                    label_non_zero_id = (batch['labels'][i] != -100).nonzero()[0][0]
                    # 非-100序号减掉1，并得到A,B,C,D对应的logit
                    logit_abcd = logit[label_non_zero_id-1][abcd_idx]
                    preds.append(torch.argmax(logit_abcd).item())
                labels = labels[labels != IGNORE_INDEX].view(-1, 2)[:,0]
                refs += [abcd_idx.index(label) for label in labels.tolist()]

                loss_mmlu += loss.item()
            # Extract results by subject.
            results = {'mmlu_loss':loss_mmlu/len(data_loader)}
            subject = mmlu_dataset['subject']
            subjects = {s:{'refs':[], 'preds':[]} for s in set(subject)}
            for s,p,r in zip(subject, preds, refs):
                subjects[s]['preds'].append(p)
                subjects[s]['refs'].append(r)
            subject_scores = []
            for subject in subjects:
                subject_score = accuracy.compute(
                    references=subjects[subject]['refs'],
                    predictions=subjects[subject]['preds']
                )['accuracy']
                results[f'mmlu_{args.mmlu_split}_accuracy_{subject}'] = subject_score
                subject_scores.append(subject_score)
            results[f'mmlu_{args.mmlu_split}_accuracy'] = np.mean(subject_scores)
            trainer.log(results)
            trainer.data_collator.source_max_len = source_max_len

    trainer.add_callback(MMLUEvalCallback)
```

### 验证数据类型

```python
# Verifying the datatypes.
dtypes = {}
for _, p in model.named_parameters():
    dtype = p.dtype
    if dtype not in dtypes: dtypes[dtype] = 0
    dtypes[dtype] += p.numel()
total = 0
for k, v in dtypes.items(): total+= v
for k, v in dtypes.items():
    print(k, v, v/total)

if args.bits < 16:
    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))

all_metrics = {"run_name": args.run_name}
```

### 开启训练

```python
# Training
if args.do_train:
    # 开始训练
    train_result = trainer.train(resume_from_checkpoint=checkpoint_dir)
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    all_metrics.update(metrics)
```

## 模型评估和保存

```python
# Evaluation
if args.do_eval:
    logger.info("*** Evaluate ***")
    metrics = trainer.evaluate(metric_key_prefix="eval")
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    all_metrics.update(metrics)
# Prediction
if args.do_predict:
    logger.info("*** Predict ***")
    prediction_output = trainer.predict(test_dataset=data_module['predict_dataset'],metric_key_prefix="predict")
    prediction_metrics = prediction_output.metrics
    predictions = prediction_output.predictions
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    predictions = tokenizer.batch_decode(
        predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    with open(os.path.join(args.output_dir, 'predictions.jsonl'), 'w') as fout:
        for i, example in enumerate(data_module['predict_dataset']):
            example['prediction_with_input'] = predictions[i].strip()
            example['prediction'] = predictions[i].replace(example['input'], '').strip()
            fout.write(json.dumps(example) + '\n')
    print(prediction_metrics)
    trainer.log_metrics("predict", prediction_metrics)
    trainer.save_metrics("predict", prediction_metrics)
    all_metrics.update(prediction_metrics)

if (args.do_train or args.do_eval or args.do_predict):
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as fout:
        fout.write(json.dumps(all_metrics))
```
