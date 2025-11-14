# 🚀 SGLang 官方超参数调优指南

在大语言模型（LLM）的离线批处理推理场景中，追求极限的**吞吐量（Throughput）是性能优化的核心目标。实现高吞吐量的最关键因素在于获得并维持一个巨大的有效批处理大小（Effective Batch Size）**。

SGLang 提供了精细的参数配置，让开发者能够根据自身硬件和工作负载特性进行深度优化。本指南将基于 SGLang 运行日志中的关键指标，详细阐述如何通过调整核心超参数来最大化离线推理性能。

## 一、关键性能指标监控

当 SGLang 服务在高负载下稳定运行时，您应密切关注服务器日志中类似以下的指标输出：

```bash
Decode batch. #running-req: 233, #token: 370959, token usage: 0.82, cuda graph: True, gen throughput (token/s): 4594.01, #queue-req: 317
```

其中，两个关键指标是：

1. **`#queue-req` (队列请求数)：** 表示当前在调度队列中等待处理的请求数量。
2. **`token usage` (KV 缓存 Token 利用率)：** 表示服务器 KV 缓存内存的实际利用率， 0.0～1.0之间的⼩数， 1.0表示占满。

此外：

- `#running-req`：正在调度执⾏的请求数。

- `#token`：当前batch⽣成的总token数。

- `cuda graph`：是否开启Cuda 计算图

- `gen throughput (token/s) ` : sglang 平均每秒处理的 token 数



## 二、请求负载与队列管理 (`#queue-req`)

`#queue-req` 能够反映客户端提交请求的速度是否匹配得上服务器的处理能力。

### 调优 `#queue-req`

如果日志中频繁出现 **`#queue-req: 0`**，说明您的客户端请求提交速度过慢，服务器处于饥饿状态，无法满载运行，可适当增加请求的数量。

- **理想范围：** 保持在一个适度的范围内，例如 **$100 \sim 2000$**。
- **避免过大：** `#queue-req` 过大会显著增加服务器调度（Scheduling）的开销，影响效率。

## 三、KV 缓存 Token 利用率调优 (`token usage`)

Token 利用率直接反映了 GPU 内存资源的使用效率。我们的目标是实现高利用率，即 **`token usage > 0.9`**。

### 1. 利用率过低（Token 浪费）

如果频繁看到 **`token usage < 0.9`**，同时 **`#queue-req > 0`**，这通常意味着服务器在接受新请求时过于**保守**。

- **原因分析：** 这种情况常见于用户请求设置了很大的 `max_new_tokens`，但由于遇到 EOS 或停止字符串而提前结束，导致系统预留了过多的 KV 缓存空间。
- **调优方法：** 尝试降低 **`--schedule-conservativeness`** 的值，例如设置为 `0.3`。

### 2. 利用率过高（内存紧张）

如果您观察到 **`token usage` 极高**，并频繁出现以下警告：

```
KV cache pool is full. Retract requests. ...
```

这表明 KV 缓存池已满，服务器正在收回请求。

- **调优方法：** 此时应提高 **`--schedule-conservativeness`** 的值，例如设置为 `1.3`，以更保守地接受新请求，避免过度占用缓存。
- **容忍范围：** 如果此类警告偶尔出现（例如每分钟 1 次），则是可以接受的正常现象。

## 四、内存分配与 KV 缓存池容量优化 (`--mem-fraction-static`)

SGLang 的总内存使用包含以下几个主要部分：
$$
\text{总显存} = \text{模型权重} + \text{KV 缓存池} + \text{CUDA 图缓冲区} + \text{激活值}
$$
参数 --mem-fraction-static 决定了前两项占总 GPU 显存的比例：
$$
\text{mem\_fraction\_static} = \frac{\text{模型权重} + \text{KV 缓存池}}{\text{GPU 显存容量}}
$$
为了支持更高的并发（即更大的 KV 缓存池），应尽可能提高 `--mem-fraction-static`，同时还要为激活和 CUDA 图缓冲区保留足够的内存。

### 调优步骤

1. **确定激活值所需空间：** 经验法则是为激活值（Activations）预留 $5 \sim 8\text{ GB}$ 的显存。

2. **检查日志：** 在服务器启动日志中查找 `available_gpu_mem` 的值：

   ```
   [2025-08-11 17:17:03] ..., available_gpu_mem=13.50 GB
   ```

3. **调整策略：**

   - 如果 `available_gpu_mem` 很高（如 $10\text{ GB}$ 以上），说明 KV 缓存池分配不足，应**增加** `--mem-fraction-static`。
   - 如果 `available_gpu_mem` 过低，请**降低** `--mem-fraction-static`，以防后续的内存溢出（OOM）。

最直接的方法是每次以 $0.01$ 的增量提高 `--mem-fraction-static`，直到您的工作负载开始出现 OOM 错误。

## 五、内存溢出（OOM）的避免与解决

当遇到内存溢出（OOM）错误时，应根据 OOM 发生的阶段进行针对性调整：

| **OOM 发生阶段**     | **调优参数**             | **调整方向**            | **影响**                                     |
| -------------------- | ------------------------ | ----------------------- | -------------------------------------------- |
| **预填充 (Prefill)** | `--chunked-prefill-size` | 降低至 `4096` 或 `2048` | 节省内存，但会降低长提示的预填充速度。       |
| **解码 (Decoding)**  | `--max-running-requests` | 降低                    | 限制了最大并发数。                           |
| **通用 OOM**         | `--mem-fraction-static`  | 降低至 `0.8` 或 `0.7`   | 减少 KV 缓存池大小，限制了峰值吞吐量和并发。 |



## 六、高级加速机制与并行策略

### 1. CUDA Graph 优化 (`--cuda-graph-max-bs`)

CUDA Graph 默认仅对小批处理（例如小于 $160$ 或 $256$）启用。对于某些模型和大规模张量并行（TP），扩大 CUDA Graph 的适用范围能带来性能提升。

- **调优方法：** 尝试将 `--cuda-graph-max-bs` 增加到一个更大的值（如 $512$ 或 $768$）。
- **注意事项：** CUDA Graph 会占用额外的内存，因此可能需要同时降低 `--mem-fraction-static`。

### 2. 数据并行与张量并行 (`--dp-size`, `--tp-size`)

在 GPU 内存充足的情况下，**数据并行（DP）**通常比张量并行（TP）能带来更高的吞吐量。

- **优先级：** 总是优先考虑使用数据并行来提高吞吐量。
- **建议：** 参考 SGLang Router 的文档，它提供了更优化的数据并行方案。

### 3. 其他优化选项

- **`--enable-torch-compile`：** 对小模型和小批处理加速有效。
- **量化：** 尝试不同的量化方法，例如使用 `--quantization fp8` 进行 FP8 量化。
- **调度策略：** 如果工作负载中存在大量共享前缀，可以尝试使用 `--schedule-policy lpm`（最长前缀匹配）。该策略通过重新排序请求来提高缓存命中率，但会引入额外的调度开销。
