# vLLM V1 优化与性能调优指南

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

------

## 分块预填充（Chunked Prefill）

分块预填充允许 vLLM 将较大的预填充请求划分为多个较小块，并与解码请求一起批处理。该机制有助于在计算密集型（预填充）与内存密集型（解码）操作间取得更好的平衡，从而提升吞吐量与延迟表现。

在 vLLM V1 中，分块预填充**始终默认启用**（不同于 V0 中根据模型条件决定是否启用）。

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

------

## 内存优化建议

若遇到显存不足问题，可参考以下优化措施：

### 控制上下文长度与批处理大小

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

V1 中 CUDA 图编译使用的内存比 V0 多。你可以通过以下方式减少开销：

```python
from vllm import LLM
from vllm.config import CompilationConfig, CompilationLevel

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    compilation_config=CompilationConfig(
        level=CompilationLevel.PIECEWISE,
        cudagraph_capture_sizes=[1, 2, 4, 8]  # 更小的捕获批次
    )
)
```

或禁用 CUDA 图编译以节省内存：

```python
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    enforce_eager=True
)
```

------

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
