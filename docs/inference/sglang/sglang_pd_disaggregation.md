# SGLang Prefill/Decode（PD）解耦架构解析

## **引言：理解 LLM 推理的“阿喀琉斯之踵”**

大型语言模型（LLM）的推理过程通常分为两个截然不同的阶段：

1. **Prefill（预填充）：** 处理输入的完整提示（Prompt Sequence），计算其在模型中的表示，并生成初始的 **Key-Value (KV) 缓存**。
2. **Decode（解码）：** 基于 Prefill 阶段生成的 KV 缓存，循环生成单个新的 token，直到达到停止条件。

在传统的“单体式（Monolithic）”服务架构中，这两个阶段在同一个 Worker 实例上串行或混合执行。然而，它们对硬件资源的需求存在根本差异：

- **Prefill：** 是**计算密集型（Compute-Bound）**，需要高 FLOPS。
- **Decode：** 是**内存密集型（Memory-Bound）**，受限于内存带宽和 KV 缓存的存储需求。

**SGLang 的 PD Disaggregation 架构**正是为了打破这种资源需求的冲突，实现推理资源的精益化管理和极致性能而诞生的解决方案。

## 一. 通俗解析：为什么要分离 Prefill 和 Decode？

想象一下一条生产线：一个工序（Prefill）需要大型、高速的 CPU/GPU 集群进行大规模并行计算；而下一个工序（Decode）需要快速、低延迟的内存通道来处理大量的小零件（KV 缓存）。

如果将这两项工作都放在一个机器上，必然会导致资源浪费：

- 当执行 Prefill 时，内存带宽可能会闲置。
- 当执行 Decode 时，高性能计算单元（如张量核心）可能利用率不足。

**PD 解耦的核心价值在于：**

| **阶段**    | **资源需求特性** | **解耦后的优化策略**                                         | **效益**                          |
| ----------- | ---------------- | ------------------------------------------------------------ | --------------------------------- |
| **Prefill** | **计算密集型**   | 可在专门配备高算力资源的 Worker 上聚合处理批次，加速前缀计算。 | 提高计算资源利用率和吞吐量。      |
| **Decode**  | **内存密集型**   | 可在配备高内存带宽、低延迟的 Worker 上并行扩展，专注于 KV 缓存管理。 | 确保低延迟和高并发的 token 生成。 |

通过这种方式，SGLang 允许您根据每个阶段的真实需求，**独立地、异构地扩展**（Independent Scaling）资源，从而显著提高整体系统的效率和成本效益。

## 二、技术深度解析：PD 解耦的关键机制

PD 解耦要实现高性能，关键在于如何高效、无阻塞地将计算密集的 Prefill 结果（即 **KV 缓存**）传输给内存密集的 Decode Worker。

### 1. 高速的 KV 缓存传输（KV Cache Transfer）

SGLang 采用了先进的技术来确保 KV 缓存传输的效率：

- **RDMA（Remote Direct Memory Access）：** SGLang 利用 RDMA 技术（如通过 NIXL 或 Mooncake 后端），允许 Prefill Worker **直接将 KV 缓存写入 Decode Worker 的 GPU 内存**。这绕过了 CPU、操作系统内核和传统网络协议栈，实现了极低延迟和高带宽的数据传输。
- **非阻塞传输：** 传输操作被设计为非阻塞，并在后台线程中运行。这保证了 SGLang 核心调度器的主事件循环不会因等待数据传输而中断，维持了持续的调度能力。

### 2. 请求处理的动态配对与生命周期

在 PD 解耦架构下，请求的生命周期涉及网关（Router）和两个 Worker 集群：

1. **网关接收：** SGLang Model Gateway（或负载均衡器）接收到推理请求。
2. **动态配对：** 网关从可用的 Prefill Worker 池和 Decode Worker 池中**动态选择**一个 Worker 对。
3. **连接建立：** Prefill Worker 和 Decode Worker 之间建立临时的、针对该请求的高速连接（包括注册 RDMA 连接信息）。
4. **执行与传输：** Prefill Worker 执行计算，并通过 RDMA 将 KV 缓存传输至 Decode Worker 的预分配 GPU 内存中。
5. **继续解码：** Decode Worker 在确认 KV 缓存到达后，接管请求并开始 token 的增量生成。

这种动态连接和非阻塞传输的设计，是 SGLang PD 解耦能够在大规模异构集群中保持高性能的关键技术保障。

## 三、实践与部署：如何配置 PD Disaggregation

PD 解耦架构通常通过 SGLang Model Gateway（Router）进行统一调度和管理。您需要独立启动 Prefill Worker 和 Decode Worker，然后配置 Router 来识别并调度它们。

### 1. 独立启动 Worker 实例

Worker 启动时，需要明确指定其工作模式（Prefill 或 Decode）：

```Bash
# 启动 Prefill Worker 实例
python3 -m sglang.launch_server \
  --model-path /path/to/model \
  --disaggregation-mode prefill \
  --port 30001
  # ... 其他针对计算优化的参数 ...

# 启动 Decode Worker 实例
python3 -m sglang.launch_server \
  --model-path /path/to/model \
  --disaggregation-mode decode \
  --port 30011
  # ... 其他针对内存优化的参数 ...
```

### 2. 通过 Router 启用 PD 解耦调度

在 Router 启动时，使用 `--pd-disaggregation` 标志，并分别指定 Prefill 和 Decode Worker 的端点，同时可以为每个阶段设置独立的负载均衡策略。

```Bash
# 启动 SGLang Router/Gateway
python -m sglang_router.launch_router \
 --pd-disaggregation \
 --prefill http://prefill1:30001 http://prefill2:30002 \
 --decode http://decode1:30011 http://decode2:30012 \
 --policy cache_aware  # 整体调度策略
 --prefill-policy cache_aware  # 针对 Prefill Worker 的策略
 --decode-policy power_of_two  # 针对 Decode Worker 的策略
```

| **参数**              | **说明**                                                    |
| --------------------- | ----------------------------------------------------------- |
| `--pd-disaggregation` | **启用** PD 解耦模式。                                      |
| `--prefill`           | 指定所有 Prefill Worker 的 HTTP 端点列表。                  |
| `--decode`            | 指定所有 Decode Worker 的 HTTP 端点列表。                   |
| `--prefill-policy`    | **覆盖 Prefill 阶段**的负载均衡策略（例如 `cache_aware`）。 |
| `--decode-policy`     | **覆盖 Decode 阶段**的负载均衡策略（例如 `power_of_two`）。 |

## 四、总结

SGLang 的 Prefill/Decode 解耦架构是面向 LLM 大规模生产部署的重大技术飞跃。它通过分离计算与内存密集型工作负载，实现了**硬件资源的精细匹配**和**推理流程的无缝衔接**。

对于追求极低延迟（Low Latency）和极高吞吐量（High Goodput）的生产环境，尤其是在异构硬件集群中，PD Disaggregation 是实现高效能、可控成本 LLM 服务不可或缺的优化利器。
