# 鹏城·脑海

PengCheng.Mind 又称鹏城·脑海，是鹏城实验室基于[鹏程·盘古α](https://openi.pcl.ac.cn/PCL-Platform.Intelligence/PanGu-Alpha)开发、开源、开放的基于Transformer架构的自回归式语言模型，目前包含7B、200B两个版本。模型全流程基于中国算力网的全自主安全可控国产软硬件平台进行开发和训练，采用MindSpore框架实现在大规模集群上长期稳定的多维分布式并行训练。鹏城·脑海模型主要聚焦中文核心能力，兼顾英文和部分多语言能力。当前模型已完成训练1.5T Tokens数据量，仍在持续训练迭代中。

#  MindFormers

- https://gitee.com/mindspore/mindformers

MindSpore Transformers套件的目标是构建一个大模型训练、微调、评估、推理、部署的全流程开发套件，提供业内主流的Transformer类预训练模型和SOTA下游任务应用，涵盖丰富的并行特性。期望帮助用户轻松的实现大模型训练和创新研发。

MindSpore Transformers套件基于MindSpore内置的并行技术和组件化设计，具备如下特点：

- 一行代码实现从单卡到大规模集群训练的无缝切换；
- 提供灵活易用的个性化并行配置；
- 能够自动进行拓扑感知，高效地融合数据并行和模型并行策略；
- 一键启动任意任务的单卡/多卡训练、微调、评估、推理流程；
- 支持用户进行组件化配置任意模块，如优化器、学习策略、网络组装等；
- 提供Trainer、pipeline、AutoClass等高阶易用性接口；
- 提供预置SOTA权重自动下载及加载功能；
- 支持人工智能计算中心无缝迁移部署；

目前支持的模型列表如下：

| 模型                                                         | 任务（task name）                                            | 模型（model name）                                           |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [LLama2](https://github.com/mindspore-lab/mindformers/blob/master/docs/model_cards/llama2.md) | [text_generation](https://github.com/mindspore-lab/mindformers/blob/master/docs/task_cards/text_generation.md) | llama2_7b llama2_13b llama2_7b_lora llama2_13b_lora llama2_70b |
| [GLM2](https://github.com/mindspore-lab/mindformers/blob/master/docs/model_cards/glm2.md) | [text_generation](https://github.com/mindspore-lab/mindformers/blob/master/docs/task_cards/text_generation.md) | glm2_6b glm2_6b_lora                                         |
| [CodeGeex2](https://github.com/mindspore-lab/mindformers/blob/master/docs/model_cards/codegeex2.md) | [text_generation](https://github.com/mindspore-lab/mindformers/blob/master/docs/task_cards/text_generation.md) | codegeex2_6b                                                 |
| [LLama](https://github.com/mindspore-lab/mindformers/blob/master/docs/model_cards/llama.md) | [text_generation](https://github.com/mindspore-lab/mindformers/blob/master/docs/task_cards/text_generation.md) | llama_7b llama_13b llama_7b_lora                             |
| [GLM](https://github.com/mindspore-lab/mindformers/blob/master/docs/model_cards/glm.md) | [text_generation](https://github.com/mindspore-lab/mindformers/blob/master/docs/task_cards/text_generation.md) | glm_6b glm_6b_lora                                           |
| [Bloom](https://github.com/mindspore-lab/mindformers/blob/master/docs/model_cards/bloom.md) | [text_generation](https://github.com/mindspore-lab/mindformers/blob/master/docs/task_cards/text_generation.md) | bloom_560m bloom_7.1b                                        |
| [GPT2](https://github.com/mindspore-lab/mindformers/blob/master/docs/model_cards/gpt2.md) | [text_generation](https://github.com/mindspore-lab/mindformers/blob/master/docs/task_cards/text_generation.md) | gpt2_small gpt2_13b                                          |
| [PanGuAlpha](https://github.com/mindspore-lab/mindformers/blob/master/docs/model_cards/pangualpha.md) | [text_generation](https://github.com/mindspore-lab/mindformers/blob/master/docs/task_cards/text_generation.md) | pangualpha_2_6_b pangualpha_13b                              |

# MindRLHF

- https://gitee.com/mindspore-lab/mindrlhf.git

`MindSpore RLHF`（简称 `MindRLHF`）以[MindSpore](https://gitee.com/mindspore/mindspore)作为基础框架，利用框架具备的大模型并行训练、推理、部署等能力，助力客户快速训练及部署带有百亿、千亿级别基础模型的RLHF算法流程。MindRLHF包含3个阶段的学习流程：

- 阶段1： 预训练模型训练
- 阶段2： 奖励模型训练
- 阶段3： 强化学习训练

MindRLHF集成了大模型套件[MindFormers](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Fmindspore-lab%2Fmindformers)中丰富的模型库， 提供了Pangu-Alpha(2.6B, 13B)、GPT-2等基础模型的微调流程。MindRLHF 完全继承MindSpore的并行接口，可以一键将模型部署到训练集群上，开启大模型的训练和推理。

为了提升推理性能， MindRLHF中集成了`增量推理`，通过状态复用，相比于全量推理，推理性能可提升`30%`以上。

当前版本集成了Pangu-alpha(13B)、GPT2、Baichuan2(7B/13B) 模型，用户可以基于这两个模型进行探索。未来，我们将提供更多模型如LLAMA、BLOOM、GLM等，帮助用户快速实现自己的应用。具体支持列表如下所示：

表 1： 当前MindSpore RLHF支持的模型和规模

| 模型     | Pangu-alpha | GPT2 | Baichuan2 |
| -------- | ----------- | ---- | --------- |
| 规模     | 2.6B/13B    | 124M | 7B/13B    |
| 支持并行 | Y           | Y    | Y         |
| 硬件     | NPU         | NPU  | NPU       |

当前流程下，不同模型对不同训练阶段的支持情况如下表所示：

表 2： 当前MindSpore RLHF支持的模型和阶段

| 训练阶段       | Pangu-alpha | GPT2 | Baichuan2 |
| -------------- | ----------- | ---- | --------- |
| 预训练模型训练 | Y           | Y    | Y         |
| 奖励模型训练   | Y           | Y    | Y         |
| 强化学习训练   | Y           | Y    | Y         |

未来，我们将打通更多的模型，如`LLAMA`、`GLM`、`BLOOM`等，敬请期待。

# MindPet简介

- https://gitee.com/mindspore-lab/mindpet.git

MindPet（Pet：Parameter-Efficient Tuning）是属于Mindspore领域的微调算法套件。随着计算算力不断增加，大模型无限的潜力也被挖掘出来。但随之在应用和训练上带来了巨大的花销，导致商业落地困难。因此，出现一种新的参数高效（parameter-efficient）算法，与标准的全参数微调相比，这些算法仅需要微调小部分参数，可以大大降低计算和存储成本，同时可媲美全参微调的性能。

**目前MindPet已提供以下六种经典低参微调算法以及一种提升精度的微调算法的API接口，用户可快速适配原始大模型，提升下游任务微调性能和精度；**

| 微调算法       | 算法论文                                                     | 使用说明                                                     |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| LoRA           | LoRA: Low-Rank Adaptation of Large Language Models           | [MindPet_DeltaAlgorithm_README](https://gitee.com/mindspore-lab/mindpet/blob/master/doc/MindPet_DeltaAlgorithm_README.md) 第一章 |
| PrefixTuning   | Prefix-Tuning: Optimizing Continuous Prompts for Generation  | [MindPet_DeltaAlgorithm_README](https://gitee.com/mindspore-lab/mindpet/blob/master/doc/MindPet_DeltaAlgorithm_README.md) 第二章 |
| Adapter        | Parameter-Efficient Transfer Learning for NLP                | [MindPet_DeltaAlgorithm_README](https://gitee.com/mindspore-lab/mindpet/blob/master/doc/MindPet_DeltaAlgorithm_README.md) 第三章 |
| LowRankAdapter | Compacter: Efficient low-rank hypercom plex adapter layers   | [MindPet_DeltaAlgorithm_README](https://gitee.com/mindspore-lab/mindpet/blob/master/doc/MindPet_DeltaAlgorithm_README.md) 第四章 |
| BitFit         | BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models | [MindPet_DeltaAlgorithm_README](https://gitee.com/mindspore-lab/mindpet/blob/master/doc/MindPet_DeltaAlgorithm_README.md) 第五章 |
| R_Drop         | R-Drop: Regularized Dropout for Neural Networks              | [MindPet_DeltaAlgorithm_README](https://gitee.com/mindspore-lab/mindpet/blob/master/doc/MindPet_DeltaAlgorithm_README.md) 第六章 |
| P-Tuning v2    | P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks | [MindPet_DeltaAlgorithm_README](https://gitee.com/mindspore-lab/mindpet/blob/master/doc/MindPet_DeltaAlgorithm_README.md) 第七章 |

# DeepSpeed

- https://github.com/microsoft/DeepSpeed
-  https://gitee.com/ascend/DeepSpeed

DeepSpeed now support various HW accelerators.

| Contributor | Hardware                            | Accelerator Name | Contributor validated | Upstream validated |
| ----------- | ----------------------------------- | ---------------- | --------------------- | ------------------ |
| Huawei      | Huawei Ascend NPU                   | npu              | Yes                   | No                 |
| Intel       | Intel(R) Gaudi(R) 2 AI accelerator  | hpu              | Yes                   | Yes                |
| Intel       | Intel(R) Xeon(R) Processors         | cpu              | Yes                   | Yes                |
| Intel       | Intel(R) Data Center GPU Max series | xpu              | Yes                   | Yes                |

##  DeepSpeed 现已原生支持 NPU ！

Atlas 800T A2 及之后版本无需 deepspeed_npu 插件，直接安装，直接使用，建议使用新版 DeepSpeed。
