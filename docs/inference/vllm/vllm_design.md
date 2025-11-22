## vLLM 架构与推理流程图

```Mermaid
graph TD
    A[LLM] --> B[LLM Engine]

    B --> C[Scheduler]
    B --> D[Executor]

    C --> E[Block Manager]
    C --> F[Policy]
    E --> G[Block Allocator]
    G --> H[Physical Token Block]

    D --> I[Worker]
    I --> J[Cache Engine]
    I --> K[Model Runner]

    J --> L[KV cache Tensor]
    K --> M[Model]

    subgraph Device
        N[CPU]
        O[GPU]
        P[NPU]
        Q[Ray-GPU]
    end

    D --> Device

    classDef scheduler fill:#ff9999,stroke:#333;
    classDef executor fill:#99ccff,stroke:#333;
    classDef device fill:#ffff99,stroke:#333;
    classDef block fill:#c0f0c0,stroke:#333;

    class C,E,G,H scheduler;
    class D,I,J,K,L,M executor;
    class N,O,P,Q device;
    class H block;
```

### 📌 流程图说明

#### 🔹 顶层控制流：

- `LLM` 是用户接口，调用 `LLM Engine`。
- `LLM Engine` 是核心调度器，管理整个推理生命周期。

#### 🔹 调度部分（左侧红色路径）：

- `Scheduler`：负责请求调度，决定哪些 `SentenceGroup` 可以执行。
- `Block Manager` + `Policy`：管理物理 token block 的分配策略。
- `Block Allocator`：实际分配 `Physical Token Block`。
- `Physical Token Block`：底层存储单元，用于存放 KV Cache 的 token 数据。

> ⚠️ 注意：多个 `Sentence`（来自同一个 prompt 或不同 output）组成 `SentenceGroup`，是调度的最小单位。`Scheduler` 会为每个成功调度的 `SentenceGroup` 生成 `SentenceGroupMetaData`，并传递给 `InputMetadata` 用于模型输入。

#### 🔹 执行部分（右侧蓝色路径）：

- `Executor`：启动 `Worker` 进行实际计算。
- `Worker`：在分布式场景下，可以有多个 Worker 实现 Tensor Parallel。
- `Cache Engine`：负责在 CPU/GPU 上管理完整的 KV Cache Tensor，并执行 block 数据搬运（如从 CPU 拷贝到 GPU）。
- `Model Runner`：
  - 包含真实模型实例。
  - 负责数据预处理（pre-process）、模型前向传播、后处理（post-process）和采样（sampling）。
- `Model`：最终运行的模型（例如 LLaMA、GPT 等）。
- `Device`：支持多种硬件（CPU/GPU/NPU/Ray-GPU），由 `Worker` 绑定设备执行。

#### 🔹 数据结构关系：

- `Sentence`：表示一个推理句子，包含 prompt 和 output，控制一个逻辑 token block 列表（长度为 `⌈seqlen/block_size⌉`）。
- `SentenceGroup`：多个 `Sentence` 的集合，同一 prompt 多个 request 或多步解码输出，构成一个调度单元。
- `Scheduler` 以 `SentenceGroup` 为单位进行调度。
- 成功调度后，生成 `SentenceGroupMetaData` → 转换为 `InputMetadata` → 输入 `Model Runner`。




## vLLM Scheduler 调度逻辑

Scheduler 是 vLLM 实现高效并发推理的关键组件，它在每个 LLM Engine 推理 step 开始时执行一次调度，决定本次 step 执行哪些 `SentenceGroup`。

### 🔹 核心目标

- 每次 step 只处理 **一种类型** 的请求：
  - **Prefill（Prompt Phase）**：首次输入 prompt，需要计算所有 token。
  - **Decoding（Auto-regressive）**：逐 token 生成输出。
- 确保内存资源（Physical Token Blocks）足够分配给当前请求。
- 支持抢占（Preemption）和交换（Swapping），以应对资源不足。

### 📊 调度队列结构

| 队列      | 用途                                            |
| --------- | ----------------------------------------------- |
| `waiting` | 新加入的请求（首次 Prompt 或被抢占后重新入队）  |
| `running` | 正在执行的 Decoding 请求（已分配 block）        |
| `swapped` | 被换出到 CPU/swap 区的请求（block 已 swap-out） |

> ⚠️ 注意：`waiting` 中可以是 Prefill 或 Preempted Re-compute；`running` 和 `swapped` 仅包含 Decoding 阶段的请求。



### 🔄 scheduler 函数流程图

````mermaid
graph TD
    A["Start _scheduler()"] --> B{swapped empty?}

    B -- True --> C[Prefill Phase]
    B -- False --> D[Decoding Phase]

    C --> E{waiting front exceed max capacity?}
    E -- True --> F[return scheduled]
    E -- False --> G[pop waiting]
    G --> H[enqueue scheduled]
    H --> I[loop until full or limit reached]
    I --> E

    D --> J[Sort running by FCFS]
    J --> K{running front can allocate new block?}
    K -- True --> L[copy block]
    L --> M[enqueue new_running]
    M --> N[loop: continue with next running]
    N --> K

    K -- False --> O[preempt running back]
    O --> P[swap out block]
    P --> Q[enqueue preempt]
    Q --> R{preempt empty?}
    R -- True --> S[return new_running, swap_out_indices]
    R -- False --> T{swapped front can swap_in?}
    T -- True --> U[pop swapped, swap_in]
    U --> V[enqueue new_running]
    V --> W[loop: try next swapped]
    W --> T
    T -- False --> X[break]
    X --> S
````

### 🧩 流程详解

#### 🟦 第一步：判断是否进入 Prefill 阶段

```text
if swapped.empty():
    → Prefill Phase
```

##### ✅ Prefill Phase 流程：

1. 检查 `waiting` 队列头部是否超过最大容量（如 block 数量或 seq length）。
2. 若未超限，则从 `waiting` 出队并加入 `scheduled`。
3. 循环直到：
   - 达到 block 分配上限
   - 达到最大 sequence 长度限制
4. 返回 `scheduled` 队列 → 这些请求将在本 step 执行 prefill。

> 💡 目标：尽可能多地并行处理新 prompt 请求。

#### 🟨 第二步：进入 Decoding Phase（swapped 不为空）

##### ✅ Step 1: 处理 Running 队列（FCFS 排序）

1. 对 `running` 队列按 FCFS（先进先出）排序。
2. 尝试为 `running` 队首的 SentenceGroup 分配新的 token block：
   - 如果成功 → copy block 到新 slot → 加入 `new_running`
   - 如果失败 → 抢占队尾的 request：
     - 将其加入 `preempt` 队列
     - 并 swap-out 其 block（释放资源）

> ⚠️ 抢占策略：

- `recompute mode`：直接释放 block，放入 `waiting` 队列（需重算）
- `swap mode`：swap-out block，放入 `swapped` 队列（保留状态）

###### ✅ Step 2: 处理 Swapped 队列（尝试 swap-in）

1. 检查 `preempt` 是否为空：
   - 若不为空 → 表示资源紧张 → 直接返回 `new_running` 和 `swap_out_indices`
   - 若为空 → 可能有富余资源 → 尝试从 `swapped` 中 swap-in
2. 循环尝试将 `swapped` 队首元素 swap-in：
   - 成功 → 加入 `new_running`
   - 失败 → 停止循环，返回结果

> 💡 目标：优先保证正在运行的请求继续执行，同时尽可能恢复被换出的请求。

### 📌 关键特性总结

| 特性                  | 说明                                                |
| --------------------- | --------------------------------------------------- |
| **Phase 分离**        | 每个 step 只执行 Prefill 或 Decoding，避免混合      |
| **FCFS Policy**       | 公平调度，防止饥饿                                  |
| **Preemption**        | 当 block 不足时，抢占旧请求释放资源                 |
| **Swapping**          | 支持大模型长序列，通过 CPU/GPU 交换缓解显存压力     |
| **Recompute vs Swap** | 可配置策略：recompute 快但耗算力；swap 保留状态但慢 |

### 🛠️ 示例场景模拟

假设：

- 当前 `running` 有 3 个 request（A, B, C）
- `swapped` 有 1 个 request（D
- `waiting` 有 2 个新 prompt（E, F）
- 显存紧张，无法为 C 分配新 block

### 调度过程：

1. `swapped` 不为空 → 进入 Decoding Phase
2. 排序 `running`: [A, B, C]
3. 为 A 分配 block → 成功 → 加入 `new_running`
4. 为 B 分配 block → 成功 → 加入 `new_running`
5. 为 C 分配 block → 失败 → 抢占 C → 加入 `preempt`，swap-out 其 block
6. `preempt` 不为空 → 不尝试 swap-in
7. 返回 `new_running = [A, B]`，以及 `swap_out_indices = [C]`

> 结果：A、B 继续执行；C 被抢占，下次可能被 recompute 或 swap-in
