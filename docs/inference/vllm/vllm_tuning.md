# 🚀 vLLM 优化与性能调优指南

## 一、核心调度与缓存优化

### 1. 任务抢占（Preemption）

由于 **Transformer** 架构的自回归（Autoregressive）特性，当并发请求数量较高时，系统可能面临 **KV 缓存（KV Cache）** 空间不足的挑战。在此情况下，vLLM 会触发抢占（Preemption）机制，暂停（Preempt）部分请求以释放 KV Cache 空间，并在资源满足条件后对被抢占的请求进行**重计算（Recompute）**。

当发生抢占时，系统会输出类似如下的警告信息：

```Bash
WARNING 05-09 00:49:33 scheduler.py:1057 Sequence group 0 is preempted by PreemptionMode.RECOMPUTE mode because there is not enough KV cache space. This can affect the end-to-end performance. Increase gpu_memory_utilization or tensor_parallel_size to provide more KV cache memory. total_cumulative_preemption_cnt=1
```

虽然抢占机制提升了系统的**鲁棒性（Robustness）**，但重计算操作会不可避免地增加请求的**端到端延迟（End-to-End Latency）**。如果系统频繁遭遇抢占问题，建议采取以下优化措施：

- **增加 `gpu_memory_utilization`**：该参数控制 vLLM 预分配给 KV Cache 的 GPU 显存比例。提高该值可提供更大的缓存空间。
- **降低并发限制**：通过减少 `max_num_seqs`（最大并发序列数）或 `max_num_batched_tokens`（最大批处理 Token 总数），以降低系统对瞬时缓存的需求。
- **增加 `tensor_parallel_size` (TP)**：将模型权重分片至更多 GPU，从而在每张 GPU 上为 KV Cache 腾出更多显存空间。但需注意，TP 可能会引入额外的跨卡同步（Synchronization）开销。
- **增加 `pipeline_parallel_size` (PP)**：将模型的层级分布在多个 GPU 上，减少了单卡上的模型权重显存占用，间接为 KV Cache 释放了内存。但这可能导致**流水线气泡（Pipeline Bubble）** 增加，影响推理延迟。

您可以通过 **Prometheus 指标** 监控抢占次数，并通过设置 `disable_log_stats=False` 来记录累计抢占计数。

> **注意：** 在 vLLM V1 版本中，默认的抢占模式已优化为 `RECOMPUTE`（重计算），而非 V0 版本中的 `SWAP`（交换到主机内存），因为在 V1 的架构下重计算的开销更低。

### 2. 前缀缓存（Prefix Caching）

**自动前缀缓存（Automatic Prefix Caching, APC）** 是一种关键的优化技术，用于缓存已处理查询的 KV Cache。当新的请求与历史查询共享相同的前缀时，系统可以直接复用对应的 KV Cache，从而跳过共享部分的重复计算。

这项技术能显著降低**首次 Token 生成时间（Time to First Token, TTFT）**，特别适用于**多轮对话**以及使用长 **System Prompt** 的场景。

vLLM 的前缀缓存机制与传统 **RadixAttention** 的区别在于：vLLM 使用基于 **Prompt Token ID** 生成的哈希码作为每个 KV 块（Block）的唯一标识，并采用**完全前缀匹配**的方式进行块查找。

值得注意的是，vLLM 的前缀缓存不仅缓存了 **Prompt**（前缀）部分，还包含了模型**已生成的（Generated）** 部分，这与 RadixAttention 的部分实现是相似的。

```python
# set enable_prefix_caching=True to enable APC
llm = LLM(
    model='lmsys/longchat-13b-16k',
    enable_prefix_caching=True # 启用自动前缀缓存
)
```

### 3. 分块预填充（Chunked Prefill）

在大型语言模型（LLM）推理过程中，当输入序列（Prompt）较长时，**Prefill 阶段**（预填充阶段）的计算负担会显著增加，通常带来以下性能挑战：

1. **首次 Token 生成时间（TTFT）变长**：Prefill 阶段涉及大量输入 Token 的计算，直接增加了生成第一个输出 Token 的延迟。
2. **显存瞬时占用高**：Prefill 阶段需要为所有输入 Token 计算并构建 Key 和 Value 向量（KV Cache），长序列可能导致瞬时**显存溢出（Out-of-Memory, OOM）**。

#### 机制概述

**分块预填充（Chunked Prefill）** 技术将较大的预填充请求分解（Chunk）为多个较小的块，并允许这些块与解码（Decode）请求在同一批次中并行处理。该机制有助于在**计算密集型**（预填充）和**存储密集型**（解码）操作间取得更优的平衡，从而提升整体吞吐量与降低延迟。

**LLM 推理的两大阶段：**

1. **Prefill 阶段（预填充）**
   - **定义**：从接收完整 Prompt 到生成第一个输出 Token 的过程。
   - **特性**：属于**计算密集型任务**，GPU 利用率高，主要开销在于注意力机制中的 KV Cache 构建。
2. **Decode 阶段（解码）**
   - **定义**：从生成第一个输出 Token 到推理结束的自回归生成过程。
   - **特性**：属于**存储密集型任务**，涉及频繁的内存读写操作（IO 消耗大），GPU 利用率相对较低。

默认情况下，vLLM 倾向于优先处理 Prefill 请求。启用 Chunked Prefill 后，系统通过分块实现 Prefill 和 Decode 请求的并行处理，并赋予 Decode 请求更高的优先级，其优势包括：

- 显著降低单个长 Prefill 请求的计算负载。
- 减少瞬时显存占用，降低 OOM 风险。
- 改善 TTFT 和整体推理效率。
- 特别适用于处理超长上下文或大规模并发请求的场景。

> **注意：** 在 vLLM V1 版本中，分块预填充**始终默认启用**，无需显式设置。

```Python
# VLLM V1 中默认启用，但仍可通过参数配置
llm = LLM(
    model='lmsys/longchat-13b-16k',
    enable_chunked_prefill=True
)
```

#### 调度策略与调优建议

启用 Chunked Prefill 后，调度器将**优先处理解码请求**，以确保降低 **Token 间延迟（Inter-Token Latency, ITL）**。即先批量调度所有待解码请求，若仍有剩余的 `max_num_batched_tokens` 空间，则调度预填充请求。超出限制的 Prefill 请求会被自动分块。

您可以根据业务需求调整 `max_num_batched_tokens` 参数：

- **较小值（如 2048）**：更适合**低延迟场景**，可减少预填充块对解码性能的阻塞。
- **较大值（如 $>8096$）**：更适合**吞吐量优先场景**，尤其在大显存 GPU 上运行小模型时，可提高批处理效率。

```python
from vllm import LLM

# 设置 max_num_batched_tokens 以调优性能
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", max_num_batched_tokens=16384)\
```

## 二、并行策略（Parallelism Strategies）

vLLM 支持多种可组合的并行策略，以实现不同硬件配置下的模型部署和性能优化。

### 1. 张量并行（Tensor Parallelism, TP）

将模型的参数（如权重矩阵）在模型的每一层内部切分到多个 GPU 上，是单节点超大模型推理中最常用的方式。

- **适用场景：**
  - 模型尺寸过大，单张 GPU 无法容纳。
  - 旨在减少每张 GPU 上的显存压力，为 KV Cache 预留更多空间

```python
from vllm import LLM

# 使用 4 张 GPU 进行张量并行
llm = LLM(model="meta-llama/Llama-3.3-70B-Instruct", tensor_parallel_size=4)
```

### 2. 流水线并行（Pipeline Parallelism, PP）

将模型的不同层级（Layer）分配至不同的 GPU，形成处理流水线（Pipeline），请求按序流经这些 GPU。

- **适用场景：**
  - 张量并行方案已达瓶颈，需要进一步在更多 GPU 或跨节点间分布模型。
  - 模型深度较大但每层计算量较小（“窄深”模型）。

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

### 3. 专家并行（Expert Parallelism, EP）

专用于稀疏专家模型（Mixture of Experts, MoE），将不同的专家网络（Expert）分布到不同的 GPU 上。

- **适用场景：**
  - 特定于 MoE 架构（如 DeepSeekV3、Qwen3MoE 等）。
  - 需要在多 GPU 之间均衡专家路由和计算负载。

```python
llm = LLM(
    model="Qwen/Qwen3MoE",
    tensor_parallel_size=4,
    enable_expert_parallel=True  # 启用专家并行
)
```

### 4. 数据并行（Data Parallelism, DP）

将完整的模型副本部署到多个 GPU 或集群节点上，并发处理不同的输入批次（Batch）请求。

- **适用场景：**
  - GPU 资源充足，可重复部署模型副本。
  - 目标是提升**整体吞吐量**而非单个模型的容量。
  - 多用户场景下需要请求隔离或独立服务。

```python
llm = LLM(
    model="meta-llama/Llama-3.3-70B-Instruct",
    data_parallel_size=2
)
```

> **注意：** MoE 层的总并行粒度为 $TP \times DP$（`tensor_parallel_size` $\times$ `data_parallel_size`）。

## 三、其他性能增强技术

### 1. 异步分词器（Asynchronous Tokenizer Pool）

在大语言模型的推理服务中，**分词（Tokenization）** 是将用户输入（如文本）转换为模型可理解的 Token 序列的关键预处理步骤。

#### 高并发场景的优化

当系统面临高并发请求时，**单一的分词器**可能成为性能瓶颈。每个新的请求都需要排队等待分词处理完成，这会增加延迟并降低系统吞吐量。

引入 **`tokenizer-pool-size`** 参数正是为了解决这一问题。该参数用于控制可同时运行的**分词器实例（Worker）** 数量，从而支持多任务并行处理。通过增加分词器池的大小，可以减少任务的等待时间，有效提升系统在高负载下的响应速度和处理效率。

#### 使用建议

- **合理配置**：根据实际应用场景的负载和系统资源情况（尤其是 **CPU 核心数**），合理设置 `tokenizer-pool-size` 的值。过大的池大小会消耗更多 CPU 和内存资源。
- **监控调优**：建议监控系统的任务队列长度、平均等待时间等指标，基于实际数据对该参数进行优化调整，以达到 CPU/GPU 资源与性能的最佳平衡。

```python
llm = LLM(
    model="meta-llama/Llama-3.3-70B-Instruct",
    tokenizer_pool_size=4 # 使用 4 个分词器工作进程
)
```

### 2. 投机解码（Speculative Decoding）

投机解码（Speculative Decoding）是一种加速 LLM 推理的创新技术。它通过利用一个**辅助模型（Draft Model）** 或**预测机制**（如 N-Gram）提前预测并生成多个后续 Token，从而将传统的自回归逐 Token 解码转化为并行验证过程，显著缩短推理时间。

#### 核心机制

投机解码通过 **提前预测（Speculate）** 多个后续 Token，并将这些“投机 Token”一次性输入给主模型（Target Model）进行**并行验证**。

- 如果预测结果与主模型的结果一致（命中），则直接接受并更新 KV Cache，实现多个 Token 的加速生成。
- 如果预测结果不一致（未命中），则从不一致的 Token 处开始重新生成，代价较小。

#### `speculative_model="[ngram]"` 的作用

`speculative_model="[ngram]"` 指定了 vLLM 中使用的 **N-Gram 投机解码** 策略。该策略基于当前上下文，通过查找输入序列中连续的 Token 组合（N-Gram）来推测和生成后续 Token。它尤其适用于需要生成**较长连贯文本**的场景。

#### 关键参数说明

- **（1）`num_speculative_tokens`**
  - **定义**：指定每次解码过程中并行生成的**投机 Token 数量**。
  - **调优**：增加该值能提升理论生成效率，但会显著增加显存占用，并可能降低命中率。建议根据显存容量和实际加速效果进行权衡设置。
- **（2）`use-v2-block-manager`**
  - **要求**：投机解码机制依赖于更复杂的缓存管理策略，**必须**配合 **V2 块管理器**（`--use-v2-block-manager`）使用。
  - **作用**：V2 块管理器提供了更灵活、高效的 KV Cache 组织和管理方式，以支持多 Token 并行生成和缓存管理。
- **（3）`ngram_prompt_lookup_max`**
  - **定义**：用于控制在 N-Gram 提示查找（Prompt Lookup）过程中进行查找的**最大窗口大小**。
  - **调优**：设置较大的值可以增加找到匹配项的可能性，提高投机命中率。然而，这也会增加查找和计算开销，需要在命中率与性能之间找到最佳平衡点。

### 3. 量化（Quantization）

在本地部署大规模语言模型（LLMs）时，**模型量化（Quantization）** 是通过降低模型权重和/或激活值的精度（如从 FP32 降至 INT8、INT4 或 FP8），从而显著减少模型的**内存占用**和**推理计算开销**的关键技术。

#### vLLM 中的量化支持

vLLM 提供了对多种主流量化方法的支持。开发者可以通过 `--quantization` 参数指定所需的量化方案。

可选项示例：`{aqlm, awq, deepspeedfp, tpu_int8, fp8, fbgemm_fp8, marlin, gguf, gptq_marlin, gptq, bitsandbytes, ...}`

#### 常见量化方法简介

- **AWQ（Activation-aware Weight Quantization）**：一种考虑激活值分布的权重量化方法，能够在保持精度的前提下，有效降低内存占用。适用于对精度要求较高的任务。
- **GPTQ（Generalized Post-Training Quantization）**：针对 Decoder-only 架构设计的高效后训练量化策略，在保持高精度的同时显著降低内存需求。
- **gptq_marlin**：自 vLLM 0.6 版本起引入的优化版 GPTQ 实现，基于 **Marlin 内核**优化，能显著提升 GPTQ 模型在推理阶段的计算效率和吞吐性能。**推荐作为 GPTQ 模型的首选实现方式。**

```python
# 使用 AWQ 量化模型
llm = LLM(model="TheBloke/Llama-2-7b-Chat-AWQ", quantization="AWQ")

# 使用 GPTQ 量化模型（推荐使用 marlin 优化）
llm = LLM(model="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4", quantization="gptq_marlin")
```

## 四、内存优化与调优建议

若在部署过程中遇到显存不足（OOM）问题，可参考以下系统的优化措施：

### 1. GPU 显存利用率 (`gpu_memory_utilization`)

这是 vLLM 中最重要的参数之一，控制 **vLLM 允许使用的 GPU 显存占总显存的比例**（取值范围 $0$ 到 $1$）。该参数用于预分配 KV Cache 和模型权重所需的空间。

- **对吞吐量的影响：** 预留的 KV Cache 越大，可以容纳的并发序列数 (`max_num_seqs`) 或单个序列长度 (`max_model_len`) 就越大，从而直接提升系统的整体吞吐量。
- **调优**：设置值**过高**可能导致宿主机（Host）系统 OOM；设置值**过低**则可能导致 KV 缓存不足，引发频繁的请求抢占，影响吞吐。建议根据实际模型大小、批量大小和并发请求数进行调整，以达到性能与资源利用的最佳平衡。建议从默认值 $0.9$ 开始，在监控 GPU 内存使用的同时，谨慎地将其提高到 $0.95$ 或 $0.98$，以充分利用硬件资源。

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    gpu_memory_utilization=0.92, # 略微提高显存利用率
)
```

### 2. 并行策略：张量并行（Tensor Parallelism, TP）

- **内存作用：** 当模型权重过大无法容纳于单个 GPU 时，TP（通过 `tensor_parallel_size=N` 设置）将模型层或权重切分到多个 GPU 上。
- **效果：** TP 是解决单 GPU 内存不足（OOM）的首要手段，它分摊了模型权重占用的显存压力。

### 3. 最大并发序列数 (`max_num_seqs`)

该参数是控制 vLLM 并行度的直接手段。

- **内存关系：** KV Cache 的总需求与 `max_model_len` 和 `max_num_seqs` 的乘积大致成正比。

  $$\text{Total Cache} \approx \text{max\_model\_len} \times \text{max\_num\_seqs} \times \text{Cache\_Block\_Size}$$

   因此，提高此参数时，必须确保 KV Cache 空间（由 `gpu_memory_utilization` 决定）足够。

- **调优目标：** 目标是在确保不发生 OOM 的前提下，将此值设置到最大，以保证 GPU 持续处于高负载状态。

### 4. 批处理 Token 数 `max_num_batched_tokens`

`max_num_batched_tokens` 控制每次批处理中允许处理的**最大 Token 数量**。合理设置该参数是平衡延迟与吞吐的关键：

| **场景**                | **推荐值**                                      | **效果**                                    |
| ----------------------- | ----------------------------------------------- | ------------------------------------------- |
| **延迟敏感 / 实时性高** | 较小值（如 $256 \sim 512$）                     | 降低单次批处理计算负载，加快 TTFT。         |
| **吞吐优先 / 请求量大** | 较大值（如 $1024 \sim 4096$）                   | 提高 Prefill 阶段批处理效率，提升整体吞吐。 |
| **资源约束**            | 确保设置值不超过 GPU 的可用显存容量，避免 OOM。 |                                             |

### 5. 模型上下文长度 (`max-model-len`)

- **`max-model-len`（最大上下文长度）**：控制模型在生成响应时能够处理的输入序列与输出序列的**最大总长度**。
- **调优目标**：该值必须小于或等于模型所支持的最大位置嵌入长度（`max_position_embeddings`）。适当减小此值是解决 OOM 问题的**有效手段**，因为它直接减少了单序列所需的 KV Cache 空间。

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    max_model_len=2048,  # 限制最大上下文长度
)
```

### 6. CUDA Graph 编译配置

`enforce_eager` 用于控制 vLLM 是否始终使用 PyTorch 的 **eager 模式（即时执行模式）**。默认值为 `False`，此时 vLLM 会采用 **eager 模式与 CUDA 图（CUDA Graphs）混合执行**的方式。

- **CUDA 图**：通过记录计算图并重放执行，减少内核启动开销，从而提升推理效率。但它会**额外占用显存**用于图的构建和存储。
- **调优**：
  - **小型模型**：启用 CUDA 图（默认 `False`）通常能带来显著性能提升。
  - **大型模型/OOM 场景**：设置 `enforce_eager=True` 将禁用 CUDA 图，切换到纯 eager 模式，这可能略微影响性能，但有助于**降低显存占用**。

```python
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    enforce_eager=True # 禁用 CUDA 图，以节省显存
)
```

### 7. 可选扩展策略：CPU 卸载

在 GPU 显存受限的情况下，结合使用 `--cpu-offload-gb` 参数，可以将部分 KV Cache 卸载到 **CPU 内存**中。该方法可以有效扩展 GPU 的“虚拟内存”，从而支持更大规模的批处理或更长的上下文处理，但会引入 PCIe 总线传输的延迟。

### 8. 多模态模型下的优化

对于视觉语言模型（VLM）等支持多模态输入的模型：

- 您可以通过 `limit_mm_per_prompt` 参数来限制单个 Prompt 中图像、视频或音频等元素的数量，从而减少模型为多模态输入预留的激活内存。
  - **示例 (限制数量)：** `limit_mm_per_prompt={"image": 3, "video": 1}`
  - **示例 (禁用模态)：** `limit_mm_per_prompt={"video": 0}` (用于纯图像或纯文本应用)
- 您还可以提供 **尺寸提示（Size Hints）**（如图片的 `width`, `height`）来帮助 vLLM 根据预期输入大小进行更精确的内存预留，避免不必要的内存浪费。

```Python
from vllm import LLM

# 每个 prompt 最多包含 2 张图片
llm = LLM(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    limit_mm_per_prompt={"image": 2}
)
```

## 五、vLLM 关键参数配置建议总结

| **参数**                       | **调优方向**            | **核心目的**                  | **场景/备注**                                         |
| ------------------------------ | ----------------------- | ----------------------------- | ----------------------------------------------------- |
| **`--max-model-len`**          | $\downarrow$（减小）    | 降低显存需求 / OOM 缓解       | 解决 OOM 问题的第一优先级，匹配应用场景的最大长度。   |
| **`--gpu-memory-utilization`** | $0.85 \sim 0.95$ 间调整 | 平衡 KV Cache 与系统显存      | 默认 $0.9$，过高易 OOM，过低易抢占，影响吞吐。        |
| **`--tensor_parallel_size`**   | $\uparrow$（增加）      | 部署超大模型/释放单卡 KV 空间 | 提升模型容量和处理吞吐量，解决单卡 OOM。              |
| **`--max_num_seqs`**           | $\uparrow$（最大化）    | 提升并发度与吞吐量            | 在不 OOM 的前提下，尽量调高以最大化 GPU 利用率。      |
| **`--max_num_batched_tokens`** | 场景化设置              | 平衡延迟与吞吐                | 延迟敏感（小值），吞吐优先（大值）。                  |
| **`--enable-chunked-prefill`** | `True`（默认）          | 优化长序列 Prefill 性能       | 平衡 Prefill/Decode 负载，改善 TTFT。                 |
| **`--enable-prefix-caching`**  | `True`                  | 显著降低 TTFT / 减少重复计算  | 适用于多轮对话/长 System Prompt 场景。                |
| **`--enforce-eager`**          | `True` 或 `False`       | 控制 CUDA 图启用状态          | `True` 可节省显存（OOM 备选），`False` 通常性能更优。 |
| **`--cpu-offload-gb`**         | $\uparrow$（增加）      | 提升系统可扩展性              | 显存紧张时的备用方案，但会增加 PCIe 传输延迟。        |
| **`--disable-log_stats`**      | `True`（生产环境）      | 轻微优化 CPU/IO 开销          | 生产环境中追求极致吞吐时可考虑禁用。                  |

## 六、影响 vLLM 吞吐量的关键参数详解

以下将详细介绍对优化推理吞吐量至关重要的 vLLM 关键参数，包括其作用、重要性、最佳实践及与其他参数的相互关系。

### 1. `tensor_parallel_size`

控制 vLLM 用于张量并行的 GPU 数量。

| **类别**           | **描述**                                                     |
| ------------------ | ------------------------------------------------------------ |
| **作用**           | 将 LLM 的层或权重分片到多个 GPU 上，每个 GPU 并发处理计算的一部分。 |
| **对吞吐量的影响** | **1. 支持大型模型：** 允许部署单 GPU 无法容纳的模型。 **2. 降低延迟：** 分布计算和内存负载，提升计算资源和内存带宽。 **3. 提高有效批处理大小：** 更多的 GPU 内存可支持更大的批处理大小，直接提高整体吞吐量。 |
| **可用值**         | 整数 $N$，通常对应于要使用的 GPU 数量（如 $1, 2, 4, 8$）。   |
| **设置方法**       | Python API: `LLM(..., tensor_parallel_size=N)` <br /> CLI: `vllm serve --tensor-parallel-size N` |
| **最佳实践**       | **起始设置：** 遇到 OOM 错误时，增加此参数是首选解决方案。 **多 GPU 环境下的注意事项：** 在多 GPU 机器上，**强烈建议**将 `tensor_parallel_size` 设置为实际使用的 GPU 数量，即使只请求 $1$，vLLM 的分布式架构仍可能导致性能下降。 **互连带宽：** 使用更高的 $N$ 值时，确保 GPU 间具备 NVLink 等高带宽互连。 |
| **相互关系**       | **`pipeline_parallel_size`：** 可结合使用，TP 定义每个节点上的 GPU 数量。 **`gpu_memory_utilization`：** 增加 TP 减轻单 GPU 内存压力，允许用户安全地设置更高的 `gpu_memory_utilization`。 |

### 2. `gpu_memory_utilization`

指定 vLLM 预分配给操作（尤其是 KV Cache）的总 GPU 内存百分比（$0.0$ 到 $1.0$ 的浮点数）。

| **类别**           | **描述**                                                     |
| ------------------ | ------------------------------------------------------------ |
| **作用**           | 预留 GPU 内存空间，主要用于存储 KV Cache。                   |
| **对吞吐量的影响** | **1. 决定 KV 缓存容量：** 越高的值意味着更大的 KV Cache，可容纳更多的并发序列或更长的单个序列，直接提高吞吐量。 **2. 资源最大化：** 允许精细调整 vLLM 占用的资源，以最大限度地提高利用率。 |
| **可用值**         | $0.0$ 到 $1.0$ 之间的浮点数。默认值通常为 $0.9$（$90\%$）。  |
| **设置方法**       | Python API: `LLM(..., gpu_memory_utilization=0.95)` <br /> CLI: `vllm serve --gpu-memory-utilization 0.95` |
| **最佳实践**       | **谨慎增加：** 从默认值 $0.9$ 开始，若 GPU 仍有闲置内存，可逐步增加（如到 $0.95$ 或 $0.98$）以提高并发数。 **降低以避免 OOM：** 若遇到“内存溢出”错误，必须降低此值。 **考虑其他进程：** 如果 GPU 上有其他应用运行，需预留足够的空间。 |
| **相互关系**       | **`max_model_len`：** 序列越长，每个序列消耗的 KV Cache 越多，可能需要降低 `gpu_memory_utilization`。 **`max_num_seqs`：** 缓存容量直接限制了最大并发序列数，更高的容量支持更高的并发。 **`quantization`：** 量化释放模型权重内存，允许提高本参数的有效值。 |

### 3. `max_model_len`

定义模型能够处理的最大总序列长度（输入 Prompt Token + 生成输出 Token）。

| **类别**           | **描述**                                                     |
| ------------------ | ------------------------------------------------------------ |
| **作用**           | 为单个请求设置严格的 Token 数量上限。超过此限制的输入序列将被截断。 |
| **对吞吐量的影响** | **1. 内存分配：** 越长的序列需要越多的 KV Cache 内存，这会限制可并发处理的请求数量（批处理大小），从而影响吞吐量。 **2. 资源平衡：** 恰当地设置此参数对于平衡长序列支持、GPU 内存效率和整体推理性能至关重要。 |
| **可用值**         | 整数，代表最大 Token 数量。若未指定，vLLM 通常默认使用模型配置中的 `max_position_embeddings`。 |
| **设置方法**       | Python API: `LLM(..., max_model_len=4096)` <br /> CLI: `vllm serve --max-model-len 4096` |
| **最佳实践**       | **匹配应用场景：** 根据应用场景中典型的和最大预期的输入输出长度来设置。 **与内存平衡：** 需与 GPU 内存容量保持平衡，避免因支持过长序列而导致频繁 OOM 或并发数受限。 |
| **相互关系**       | **`gpu_memory_utilization` & `max_num_seqs`：** 这三个参数相互制约。总容量需同时满足 `max_model_len`（单序列长度）和 `max_num_seqs`（并发序列数）的需求。 |

### 4. `max_num_seqs`

定义 vLLM 调度器在 GPU 上同时处理的最大并发序列数量。

| **类别**           | **描述**                                                     |
| ------------------ | ------------------------------------------------------------ |
| **作用**           | 直接控制 vLLM 的并发水平。                                   |
| **对吞吐量的影响** | **1. 提高并行度：** 越高的值允许 vLLM 同步处理更多的请求，带来更高的整体吞吐量（Token/秒 或 请求/秒）。 **2. 最大化 GPU 利用率：** 通过确保 GPU 持续被多个活动序列占用，最大化硬件资源效率。 |
| **可用值**         | 整数。默认值可能有所不同（常见为 $256$）。                   |
| **设置方法**       | Python API: `LLM(..., max_num_seqs=512)` <br /> CLI: `vllm serve --max-num-seqs 512` |
| **最佳实践**       | **在不 OOM 的前提下最大化：** 目标是设置尽可能高的值，这是提高吞吐量的主要手段之一。 **监控内存：** 调整时务必密切监控 GPU 内存使用情况。 |
| **相互关系**       | **`max_model_len`：** 具有反向关系。`max_model_len` 越大，每个序列消耗的内存越多，允许的 `max_num_seqs` 就越少。 **`gpu_memory_utilization`：** 更高的 `gpu_memory_utilization` 为更大的 `max_num_seqs` 提供了内存基础。 |

### 5. `enforce_eager`

控制 vLLM 框架内 PyTorch 的执行模式。

| **类别**           | **描述**                                                     |
| ------------------ | ------------------------------------------------------------ |
| **作用**           | 当设置为 `True` 时，强制 vLLM 仅使用 PyTorch Eager 模式执行，从而禁用 CUDA Graphs 优化。 |
| **对吞吐量的影响** | **性能通常为负面影响：** 启用此选项通常会导致推理速度**变慢**。禁用 CUDA Graphs 会损失减少 CPU-GPU 通信和内核启动延迟的性能优势。 |
| **可用值**         | 布尔值：`True` 或 `False`。默认值为 `False`（启用混合执行）。 |
| **设置方法**       | Python API: `LLM(..., enforce_eager=True)` <br /> CLI: `vllm serve --enforce-eager` |
| **最佳实践**       | **仅用于调试或 OOM 缓解：** 在追求最高吞吐量的生产环境中，应保持默认值 `False`。主要用于性能分析或内存优化（禁用 CUDA Graph 可节省显存）。 |

### 6. `disable_log_stats`

禁用 vLLM 内部的运行时统计信息日志记录。

| **类别**           | **描述**                                                     |
| ------------------ | ------------------------------------------------------------ |
| **作用**           | 控制是否在运行时打印调度器和 CUDA 内存的统计信息。           |
| **对吞吐量的影响** | **轻微优化：** 禁用日志记录可以减少一些 CPU 开销和 I/O 操作，在高负载场景下可能带来轻微的吞吐量提升。 |
| **可用值**         | 布尔值：`True` 或 `False`。默认值为 `False`（启用统计日志）。 |
| **设置方法**       | Python API: `LLM(..., disable_log_stats=True)` <br /> CLI: `vllm serve --disable-log-stats` |
| **最佳实践**       | **生产环境考虑禁用：** 在以最大吞吐量为目标且具备完善外部监控系统的生产环境中，可以考虑设为 `True`。 |
