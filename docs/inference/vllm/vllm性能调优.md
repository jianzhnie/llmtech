# vLLM 优化与性能调优指南

## 任务抢占（Preemption）

由于 Transformer 架构的自回归特性，在并发请求数量较多时，可能会出现 KV 缓存空间不足的情况。此时，vLLM 会抢占部分请求以释放 KV 缓存空间，并在资源充足后重新计算被抢占的请求。

当发生抢占时，你可能会看到如下警告信息：

```bash
WARNING 05-09 00:49:33 scheduler.py:1057 Sequence group 0 is preempted by PreemptionMode.RECOMPUTE mode because there is not enough KV cache space. This can affect the end-to-end performance. Increase gpu_memory_utilization or tensor_parallel_size to provide more KV cache memory. total_cumulative_preemption_cnt=1
```

虽然抢占机制提高了系统的鲁棒性，但重计算会增加端到端延迟。如果你频繁遇到抢占问题，可尝试以下优化方法：

- **增加 `gpu_memory_utilization`**
   vLLM 会根据该参数预分配 GPU KV 缓存空间。提高该值可以分配更多缓存空间。
- **降低 `max_num_seqs` 或 `max_num_batched_tokens`**
   减少每批次并发请求数或 token 总数，以降低缓存需求。
- **增加 `tensor_parallel_size`**
   将模型权重在多个 GPU 间分片，为每个 GPU 腾出更多缓存空间，但可能引入同步开销。
- **增加 `pipeline_parallel_size`**
   将模型层级在多个 GPU 间分布，从而减少每张卡上的权重开销，间接为 KV 缓存释放内存。但也可能导致延迟增加。

你可以通过 Prometheus 指标监控抢占次数，并通过设置 `disable_log_stats=False` 记录累计抢占计数。

在 vLLM V1 中，默认的抢占模式为 `RECOMPUTE`（而非 V0 中的 `SWAP`），因为在 V1 架构中重计算的开销更低。

##   前缀缓存 （Prefix Caching ）

Automatic Prefix Caching 自动前缀缓存（简称APC）通过缓存已有查询的KV缓存，使得新查询若与已有查询共享相同前缀时，可直接复用对应的KV缓存，从而跳过共享部分的重复计算。能够显著降低首 token 时延，尤其适合多轮对话和长 System Prompt 场景。

vllm 的 prefix caching 与 RadixAttention 的区别在于，vllm 使用基于 prompt 的 token_ids 生成的 hash 码作为每个 block 的唯一标识。在生成当前 block 的 token_ids 时，会依赖于之前所有 block 的 token_ids。在进行 block 匹配时，采用的是完全前缀匹配的方式。

vllm 的 prefix caching 不仅缓存了 prefix 部分，还包含了生成的 generated 部分，这一点与 RadixAttention 的 prefix caching 类似。

```python
# set enable_prefix_caching=True to enable APC
llm = LLM(
    model='lmsys/longchat-13b-16k',
    enable_prefix_caching=True
)
```

## 分块预填充（Chunked Prefill）

在大语言模型（LLM）推理过程中，当输入序列较长（例如超过 1024 个 Token）时，**Prefill 阶段**的计算负担会显著增加，进而带来以下问题：

1. **首次 Token 生成时间（Time to First Token, TTFT）变长**：由于 Prefill 阶段需要处理大量输入 Token，模型生成第一个输出 Token 的延迟随之增加。
2. **显存占用高**：Prefill 阶段需为所有输入 Token 计算并缓存 Key 和 Value 向量（KV Cache），长序列可能导致显存不足（OOM），从而影响推理效率或导致失败。

VLLM 中使用 Chunked Prefill技术优化长序列处理， 分块预填充允许 vLLM 将较大的预填充请求划分为多个较小块，并与解码请求一起批处理。该机制有助于在计算密集型（预填充）与内存密集型（解码）操作间取得更好的平衡，从而提升吞吐量与延迟表现。

大语言模型的推理过程通常分为两个阶段：

### 1. **Prefill 阶段**

- **定义**：从用户输入完整 Prompt 到生成第一个输出 Token 的过程。
- **处理流程**：将输入文本（Prompt）转换为 Token 序列，计算每个 Token 对应的 Key 和 Value 向量，并将其缓存至 KV Cache 中。
- **特性**：属于**计算密集型任务**，GPU 利用率高，计算开销主要集中在注意力机制的键值缓存构建上。

### 2. **Decode 阶段**

- **定义**：从生成第一个输出 Token 到推理结束（如达到最大长度或遇到结束符）的过程。
- **处理流程**：基于已生成的 Token 逐步预测下一个 Token，并更新 KV Cache。
- **特性**：属于**存储密集型任务**，GPU 利用率较低，且涉及频繁的内存读写操作（IO 消耗大）。

在默认情况下，vLLM 会优先处理 prefill 请求，之后才会处理 decode 请求。chunked prefill 技术通过将请求分块，实现 prefill 请求和 decode 请求的并行处理，并且赋予 decode 请求更高的优先级。具有以下好处：

- 显著降低单次 Prefill 的计算负载；
- 减少瞬时显存占用；
- 改善 TTFT 和整体推理效率；
- 特别适用于处理超长上下文或大规模并发请求的场景。

使用 `enable-chunked-prefill` 优化长序列处理， 在 vLLM V1 中，分块预填充**始终默认启用**（不同于 V0 中根据模型条件决定是否启用）。

```python
# set enable_chunked_prefill=True to enable chunked prefill
llm = LLM(
    model='lmsys/longchat-13b-16k',
    enable_chunked_prefill=True
)
```

### 调度策略

启用该功能后，调度器将**优先处理解码请求**，即在处理任何预填充前，会先批量调度所有待解码请求。如果还有剩余的 `max_num_batched_tokens` 空间，则调度预填充请求。如果某个预填充请求超出当前限制，则会自动对其进行分块处理。

该策略带来两个优势：

1. **提升 ITL 与生成速度**：解码请求优先处理，降低了 token 间延迟。
2. **提高 GPU 利用率**：将预填充与解码请求同时调度至同一批次，有效利用计算资源。

### 分块调优建议

你可以通过调整 `max_num_batched_tokens` 参数来优化性能：

- **较小值（如 2048）**：更适合低延迟场景，减少预填充对解码性能的影响。
- **较大值（如 >8096）**：更适合吞吐优先场景，尤其在大显存 GPU 上运行小模型时。

```python
from vllm import LLM

# 设置 max_num_batched_tokens 以调优性能
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", max_num_batched_tokens=16384)
```

> 参考文献：
>
> - https://arxiv.org/pdf/2401.08671
> - https://arxiv.org/pdf/2308.16369

------

## 并行策略（Parallelism Strategies）

vLLM 支持多种可组合的并行策略，以在不同硬件配置中优化性能：

### 张量并行（Tensor Parallelism, TP）

将模型的参数在每一层内切分到多个 GPU 上，是单节点大模型推理中最常用的方式。

#### 适用场景：

- 模型过大，单张 GPU 无法容纳
- 希望减少每张卡上的内存压力，从而为 KV 缓存留出更多空间

```python
from vllm import LLM

# 使用 4 张 GPU 进行张量并行
llm = LLM(model="meta-llama/Llama-3.3-70B-Instruct", tensor_parallel_size=4)
```

------

### 流水线并行（Pipeline Parallelism, PP）

将模型的不同层级分配至多个 GPU，按序处理请求。

#### 适用场景：

- 已用尽张量并行方案，需进一步在更多 GPU 或跨节点间分布模型
- 模型较深但每层较小（窄深模型）

张量并行与流水线并行可以组合使用：

```python
from vllm import LLM

# 同时启用张量并行和流水线并行
llm = LLM(
    model="meta-llama/Llama-3.3-70B-Instruct",
    tensor_parallel_size=4,
    pipeline_parallel_size=2
)
```

------

### 专家并行（Expert Parallelism, EP）

专门用于稀疏专家模型（MoE），将不同专家网络分布到不同 GPU 上。

#### 适用场景：

- 特定于 MoE 架构（如 DeepSeekV3、Qwen3MoE、Llama-4）
- 需要在多 GPU 之间均衡专家计算负载

```python
llm = LLM(
    model="Qwen/Qwen3MoE",
    tensor_parallel_size=4,
    enable_expert_parallel=True  # 启用专家并行
)
```

------

### 数据并行（Data Parallelism, DP）

将整个模型复制到多个 GPU 集群上，并发处理不同批次的请求。

#### 适用场景：

- GPU 资源充足，可重复部署模型副本
- 目标是提升吞吐量而非模型容量
- 多用户场景下需要请求隔离

```python
llm = LLM(
    model="meta-llama/Llama-3.3-70B-Instruct",
    data_parallel_size=2
)
```

注意：MoE 层的并行粒度为 `tensor_parallel_size * data_parallel_size`。

## Asynchronous tokenizer (Tokenizer Pool Size)

在大语言模型的应用中，**分词（Tokenization）** 是处理输入文本的关键步骤。此过程将用户的输入（如提问或指令）拆分为模型可理解的小单元——即 Token。这些 Token 作为模型的基本处理单位，对于后续的计算至关重要。

#### 高并发场景下的挑战与解决方案

当系统面临高并发请求时，例如多个用户同时向模型提交查询或指令，单一的分词器可能成为性能瓶颈。每个任务必须排队等待分词处理完成，这不仅增加了延迟，也降低了整体的用户体验和系统的吞吐能力。

为解决这一问题，引入了 **Tokenizer Pool Size** 参数。该参数用于控制同时可用的分词器实例数量，从而支持多任务并行处理。通过增加分词器的数量，可以让更多的任务在同一时间得到处理，有效减少了任务等待时间，提升了系统的响应速度和效率。

#### 使用建议

- **调整 `tokenizer-pool-size`**：根据实际应用场景的需求以及系统资源情况（如 CPU 核心数），合理设置 `tokenizer-pool-size` 的值。较大的池大小可以在高负载情况下提高处理速度，但也会消耗更多内存和计算资源。
- **监控与调优**：定期监控系统的性能指标，包括任务队列长度、平均等待时间和资源利用率等，基于监控数据对 `tokenizer-pool-size` 进行优化调整，以达到最佳性能表现。

通过适当配置 `tokenizer-pool-size`，可以显著改善在高并发环境下的用户体验，确保系统能够高效、稳定地运行。

```python
llm = LLM(
    model="meta-llama/Llama-3.3-70B-Instruct",
    tokenizer-pool-size=4
)
```



## 投机解码（Speculative Decoding）

在大语言模型（LLM）推理过程中，生成较长文本时，传统的逐 Token 解码方式效率较低。为提升生成效率，vLLM 引入了 **投机解码（Speculative Decoding）** 技术，通过并行生成多个 Token，显著缩短推理时间。

### 什么是投机解码（Speculative Decoding）？

投机解码是一种通过**提前预测并生成多个后续 Token**来加速解码过程的技术。其核心思想是：利用模型自身的预测能力，在生成当前 Token 的同时，推测并生成若干后续 Token。这些提前生成的 Token 被称为“投机 Token”。

在后续解码过程中，如果实际生成的 Token 与投机生成的结果一致，则可以直接复用这些结果，从而减少等待时间；如果不一致，则丢弃并重新生成，代价较小。

### `speculative_model="[ngram]"` 的作用

`speculative_model="[ngram]"` 是一种具体的投机解码实现策略，称为 **N-Gram 投机解码**。其原理是：在生成当前 Token 的同时，基于当前上下文并行预测并生成后续若干个 Token，这些 Token 是根据输入序列中连续的 Token 组合（即 N-Gram）进行推测的。

该策略尤其适用于需要生成**较长文本内容**（如文章、故事、代码等）的场景，能够有效减少逐个 Token 生成所带来的延迟。

### 关键参数说明

#### （1）`num_speculative_tokens`

该参数用于指定每次解码过程中并行生成的投机 Token 数量。

- 例如，若设置为 `4`，则在生成当前 Token 的同时，会并行预测后续 4 个 Token。
- 增加该值可以提升生成效率，但也可能导致显存占用显著上升，尤其在处理长序列任务时更为明显。
- 因此，建议根据显存容量和任务需求进行合理设置，通常不会设置过大。

#### （2）`use-v2-block-manager`

投机解码机制依赖于更复杂的缓存管理策略，因此必须与 **V2 块管理器**（`--use-v2-block-manager`）配合使用。

- V2 块管理器提供了更灵活和高效的 KV Cache 管理方式，支持投机解码中多 Token 并行生成和缓存管理。
- 若未启用该选项，投机解码可能无法正常工作或性能受限。

#### （3）`ngram_prompt_lookup_max`

该参数用于控制在投机解码过程中进行 **N-Gram 提示查找（Prompt Lookup）** 的最大窗口大小。

- N-Gram 提示查找是一种优化策略，通过在输入提示中查找连续的 N-Gram（即连续的 Token 序列）来预测后续 Token。
- 设置较大的 `ngram_prompt_lookup_max` 值可以增加找到匹配项的可能性，提高投机命中率。
- 但同时也会增加查找计算开销，需在命中率与性能之间进行权衡。

---

### 总结与建议

- 投机解码（Speculative Decoding）通过并行生成多个 Token，显著提升长文本生成的效率。
- `speculative_model="[ngram]"` 是一种基于 N-Gram 的具体实现方式，适合上下文连续性强的生成任务。
- 合理配置 `num_speculative_tokens` 可在性能与资源之间取得平衡。
- 必须启用 `--use-v2-block-manager` 以支持投机解码所需的缓存管理机制。
- 使用 `ngram_prompt_lookup_max` 可优化投机 Token 的预测准确性，但需注意其对性能的影响。



## 量化（Quantization）

在本地部署大规模语言模型（Large Language Models, LLMs）时，**模型量化（Quantization）** 是一项关键技术，它通过降低模型权重的精度（如从 FP32 降至 INT8、INT4 等），显著减少模型的内存占用和推理计算开销，从而使得在资源受限设备上运行大模型成为可能。

### vLLM 中的量化支持

vLLM 支持多种主流的量化方法，包括但不限于：AWQ、GPTQ、SqueezeLLM、Marlin、GGUF、BitsAndBytes、FP8 等。开发者可以通过 `--quantization` 参数指定所需的量化方案，具体可选值包括：

```python
{aqlm, awq, deepspeedfp, tpu_int8, fp8, fbgemm_fp8, modelopt, marlin, gguf,
gptq_marlin_24, gptq_marlin, awq_marlin, gptq, compressed-tensors, bitsandbytes, qqq,
experts_int8, neuron_quant, ipex, None}
```

不同的量化方法适用于不同的模型架构和应用场景，选择合适的量化策略需结合模型本身特性、部署环境和性能要求进行综合评估。

---

### 常见量化方法简介

#### AWQ（Activation-aware Weight Quantization，激活感知权重量化）

- 一种考虑激活值分布的权重量化方法，能够有效保持量化后模型的推理精度。
- 特别适用于对精度要求较高的任务，如对话生成、文本理解等。
- 示例模型：`TheBloke/Llama-2-7b-Chat-AWQ`

#### GPTQ（Generalized Post-Training Quantization，面向 GPT 类模型的后训练量化）

- 针对 GPT 系列模型设计的高效权重量化策略，适用于大多数基于 Transformer 的 Decoder-only 架构模型。
- 在保持高精度的同时显著降低内存需求。
- 示例模型：`Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4`

#### gptq_marlin

- 自 vLLM 0.6 版本起引入的优化版 GPTQ 实现。
- 基于 Marlin 内核优化，提升了 GPTQ 模型在推理阶段的计算效率和吞吐性能。
- 推荐作为 GPTQ 模型的首选实现方式。

---

### 示例代码

```python
# 使用 AWQ 量化模型
llm = LLM(model="TheBloke/Llama-2-7b-Chat-AWQ", quantization="AWQ")

# 使用 GPTQ 量化模型
llm = LLM(model="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4", quantization="GPTQ")
```



## 内存优化建议

若遇到显存不足问题，可参考以下优化措施：

### 控制 `gpu-memory-utilization`

`gpu-memory-utilization` 是一个控制 **GPU 显存使用比例** 的参数，取值范围为 0 到 1，表示允许使用的 GPU 显存占总显存的百分比。

- 若设置值 **过高**，可能导致显存不足（OOM），影响模型推理性能甚至导致程序崩溃。
- 若设置值 **过低**，则可能导致显存利用率不足，影响吞吐量和推理效率。

该参数的默认值为 **0.9**，即允许使用 90% 的 GPU 显存。

建议根据实际模型大小、批量大小（batch size）和并发请求数进行调整，以达到性能与资源利用的最佳平衡。

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    gpu-memory-utilization=0.9,
)
```

### 控制上下文长度与批处理大小

`max-model-len` 用于控制模型支持的 **最大上下文长度（context length）**，即模型在生成响应时能够处理的输入序列与输出序列的最大总长度。

上下文长度决定了模型在理解当前输入时所能参考的历史信息长度。该参数通过 `--max-model-len` 进行配置。如果不指定该参数，系统将尝试使用设备支持的最大可能序列长度。

#### 配置要求：

- `max-model-len` 的值必须小于或等于模型所支持的最大位置嵌入长度（`max_position_embeddings`）。

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    max_model_len=2048,  # 限制上下文长度
    max_num_seqs=4       # 限制批处理请求数
)
```

------

### 调整 CUDA 图编译配置

`enforce_eager` 是一个配置参数，用于控制 vLLM 是否始终使用 PyTorch 的 **eager 模式（即时执行模式）**。该参数默认为 `False`，此时 vLLM 会采用 **eager 模式与 CUDA 图（CUDA Graphs）混合执行**的方式，以在性能和灵活性之间取得平衡。

**CUDA 图** 是 PyTorch 提供的一项性能优化技术，它通过记录计算图并重放执行，减少内核启动开销，从而提升推理效率。当设置 `enforce_eager=True` 时，vLLM 将禁用 CUDA 图，始终使用 eager 模式执行，这可能会带来性能损失，但有助于降低显存占用。

- 对于 **小型模型**，启用 CUDA 图通常能带来显著的性能提升。
- 对于 **大型模型**，CUDA 图带来的性能增益可能不明显，甚至可能因图构建开销而影响效率。

#### 建议配置：

- 如果你的模型较小，且追求性能最大化，建议保持默认设置（`enforce_eager=False`）。
- 如果模型较大，或遇到显存不足问题，可以尝试启用 `enforce_eager=True`。
- 在实际部署前，建议进行充分测试，比较启用与禁用 CUDA 图的性能与显存表现，选择最优配置。
- 可以先从不启用 `enforce_eager` 开始测试，若出现显存溢出（OOM）或性能下降，再考虑开启该选项。

```python
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    enforce_eager=True
)
```

### 通过 `max_num_batched_tokens` 进行性能调优

`max_num_batched_tokens` 是一个关键参数，用于控制每次批处理中允许处理的最大 Token 数量。合理设置该参数可以在延迟与吞吐之间取得平衡。

#### 推荐配置策略：

1. **对延迟敏感、实时性要求高的场景**
    建议设置较小值，如 256 或 512，以降低单次批处理的计算负载，加快 TTFT。
2. **吞吐优先、请求量大的场景**
    可设置为较大值（如 1024 ~ 4096），以提高 Prefill 阶段的批处理效率，从而提升整体吞吐能力。
3. **资源约束方面**
    确保设置的值不超过 GPU 的可用显存容量，避免因显存溢出（OOM）导致推理失败。

### 可选扩展策略：CPU 卸载

在显存受限的情况下，可以结合使用 `--cpu-offload-gb` 参数，将部分 KV Cache 卸载到 CPU 内存中。该方法可有效扩展 GPU 的“虚拟内存”，从而支持更大规模的批处理或更长的上下文处理。

### 多模态模型下的优化

限制每个请求中的图像或视频数量，以降低内存占用：

```python
from vllm import LLM

# 每个 prompt 最多包含 2 张图片
llm = LLM(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    limit_mm_per_prompt={"image": 2}
)
```



## VLLM 使用总结建议

`--max-model-len`：最大长度会导致高内存需求，将此值适当减小通常有助于解决OOM问题。

`--gpu-memory-utilization`：默认情况下，该值为 `0.9`会占用大量显存。

`--enforce-eager`：使用CUDA Graphs，额外占用显存。使用会影响推理效率。

`--tensor_parallel_size `：使用张量并行来运行模型，提高模型的处理吞吐量，分布式服务。

`--enable-chunked-prefill` ： 对于长序列输入，启用 `enable-chunked-prefill` 是优化 Prefill 阶段性能的有效手段；是优化 Prefill 阶段性能的有效手段；

–`max_num_batched_tokens` ： 根据业务需求合理设置 `max_num_batched_tokens`，在延迟与吞吐之间取得最佳平衡；

`--cpu-offload-gb`： 在显存紧张时，可考虑使用 CPU 卸载机制，提升系统整体的可扩展性；
