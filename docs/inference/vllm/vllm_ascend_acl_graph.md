# 深入解析 vLLM Ascend ACL Graph：原理、实现与性能优化


> **摘要**：在大语言模型（LLM）推理场景中，如何最大化利用 NPU 计算资源并减少 Host 端的调度开销是性能优化的关键。本文将深入探讨 vLLM Ascend 平台上的 ACL Graph 功能，从设计背景、核心机制（Padding & Bucketing、分段图与全图）、底层实现细节到当前的约束与限制，为您提供一份详尽的技术指南。

## 1. 背景：为什么我们需要 ACL Graph？

在 LLM 推理过程中，每一个 Token 的生成往往涉及数千个算子的执行。在传统的 **Eager Mode（即时执行模式）** 下，Host（CPU）需要逐个向 Device（NPU）下发算子指令。

如果 Host 端下发算子的速度慢于 Device 端执行算子的速度，就会产生 **Host-Bound（主机受限）** 现象。在严重的情况下，Device 可能会有一半以上的时间处于空闲状态，等待 Host 下发指令，从而导致巨大的算力浪费。

**ACL Graph（图模式）** 的引入正是为了解决这一问题。通过将计算图进行捕获并一次性下发，Graph Mode 能够显著减少 Host 与 Device 之间的通信开销。

### 模式对比

- **Eager Mode (串行交互)**:
  - Host: `Launch Op1` -> `Launch Op2` -> `Launch op3` ->...
  - Device: `Run Op1` -> `free` -> `Run Op2` ->`free` -> `Run Op3` ->  ...
  - **痛点**: 通信间隙导致 Device 频繁空闲。
- **Graph Mode (批处理交互)**:
  - Host: `Launch Graph` (包含所有 Ops)
  - Device: `Run Op1` -> `Run Op2` -> `Run Op3` ...
  - **优势**: Device 连续执行，吞吐量最大化。

## 2. 核心机制：它是如何工作的？

ACL Graph 的工作流程主要分为两个阶段：**捕获 (Capture)** 和 **重放 (Replay)**。引擎启动时，系统会通过一次模拟的前向传播捕获所有算子并保存为图；当实际请求到达时，直接在 Device 上重放该图。

然而，实际应用中面临两个主要挑战：**输入形状的动态性** 和 **复杂算子的兼容性**。

### 2.1 填充与分桶 (Padding and Bucketing)

图模式要求重放时的输入形状（Shape）必须与捕获时一致。然而，LLM 的输入 Shape 取决于调度器（Scheduler）调度的请求，具有高度动态性。

- **暴力解法**: 按照最大可能的 Shape 捕获一张图，将所有输入 Padding 到最大长度。但这会带来大量的无效计算。
- **优化解法 (Bucketing)**:
  1. 设定一个 **阈值 (Threshold)**。
  2. **低于阈值**: 预先捕获多个不同 Shape 的图（即多个 Bucket）。输入数据 Padding 到最近的 Bucket 大小，平衡了计算效率与图的复用率。
  3. **高于阈值**: 认为此时 Tensor 较大，计算密集，Host 调度开销占比降低，因此回退到 **Eager Mode** 以避免 Padding 带来的额外开销。

### 2.2 分段图与全图 (Piecewise and Full Graph)

随着 LLM 架构演进（如 MLA 机制），Attention 层变得日益复杂。例如，当一个 Batch 中同时包含 Prefill（预填充）和 Decode（解码）阶段的 Token 时，统一的图模式难以处理。

vLLM Ascend 提供了两种策略：

#### **Piecewise Graph (分段图)**

- **原理**: 将模型切分，Attention 层使用 Eager Mode 执行，其余层（如 MLP、LayerNorm）使用 Graph Mode。
- **适用场景**: 无法运行全图的复杂 Attention 场景。
- **缺点**: 相比全图，Host 下发开销依然存在。如果是小 Batch 或 CPU 性能较弱，仍可能出现 Host-Bound。
- **资源消耗**: 由于将模型切分为 `num_hidden_layers + 1` 个子模块，且每个子模块作为独立的图运行，会消耗大量 Stream 资源（详见后文 DFX 章节）。

#### **Full Graph (全图)**

- **原理**: 包含 Attention 层在内的所有算子均在图中执行。
- **优势**: 性能最优。
- **实现难点**: 需在执行前更新 Attention 算子的参数（如 Block Tables）。由于内存复用机制，必须使用特定的 NPU 接口（`graph_task_update`）并配合事件（Event）同步来确保数据一致性。

**调度策略优先级**:

1. 首选 **Full Graph** 以获得最佳性能。
2. 若全图不可用，降级为 **Piecewise Graph**。
3. 若性能仍不理想，则采取混合策略（Decode 阶段用全图，Prefill 阶段用 Eager Mode）。

## 3. 实现内幕 (Implementation Details)

vLLM 在 Ascend 上的图模式实现主要依赖于 `ACLGraphWrapper`。

1. **装饰器拦截**: vLLM 使用 `support_torch_compile` 装饰器替换了模型类的 `__init__` 和 `forward` 接口。
2. **执行流**: 当 `forward` 被调用时，控制权移交给 `ACLGraphWrapper`，由其决定是进行 Capture 还是 Replay。
3. **参数更新**: 在全图模式下，为了处理 Attention 参数的动态更新，Ascend 后端实现了 `update_attn_params` 和 `update_mla_attn_params`。通过 `torch.npu.graph_task_update_begin/end` 接口，在图执行任务中嵌入参数更新操作，确保内存复用时的正确性。

## 4. 资源约束与 DFX

### Stream 资源限制

目前 ACL Graph 的主要瓶颈在于 **Stream（流）** 资源的限制。

- **总上限**: 硬件 Stream 总数约为 2048，预留部分后可用约 1800 个。
- **消耗规则**:
  - 每个图至少占用 1 个独立 Stream。
  - 每个通信域（Comm Domain）会增加 1 个 Stream 消耗。
  - **Piecewise Graph 的挑战**: 由于模型被切分为数十个子图（基于层数），导致其 Stream 消耗远高于全图模式。
- **当前策略**: 使用 `update_aclgraph_sizes` 函数动态计算最大支持的 Bucket 数量，确保不发生 Stream 溢出。

### 限制 (Limitations)

截至当前版本（v0.11.0+），存在以下限制：

1. **配置限制**: 暂不支持显式配置 `FULL` 和 `FULL_AND_PIECEWISE` 模式（系统会自动选择）。
2. **MTP 兼容性**: 在启用 MTP (Multi-Token Prediction) 且 `num_speculative_tokens > 1` 时，vLLM 暂不支持自动推导，需显式设置 `cudagraph_capture_sizes`。
3. **编译器**: 暂不支持 `use_inductor` 后端。

## 5. 快速上手 (How to use)

在 vLLM Ascend 的 V1 Engine 中，**ACL Graph 默认开启**。
检查项:
确保您的配置中 未设置 enforce_eager=True。

```Bash
# 示例：确保未强制开启 Eager 模式
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/your/model \
    --device ascend \
    # --enforce-eager \  <-- 确保不要添加此标志
```

更多详细配置请参考官方 [Graph Mode Guide](https://docs.vllm.ai/en/latest/models/performance.html)。
