# LLM 推理核心指标详解：如何科学评估模型性能？

> **编者按**：在部署 LLM 应用时，我们常说要“快”和“稳”。但究竟什么是“快”？是首字出得快，还是整段生成得快？什么是“稳”？是平均速度快，还是长尾延迟低？本文将剥离表面的营销术语，深入剖析 LLM 推理的关键性能指标，助你建立科学的评估体系。

## 1. 为什么我们需要精确的指标？

在推理优化（Inference Optimization）的世界里，单纯的“体感速度”是不可靠的。不同的业务场景对性能的需求截然不同：

- **实时对话（Chatbot）**：用户无法忍受长时间的空白等待，首字延迟（TTFT）至关重要。
- **离线摘要（Summarization）**：用户不盯着屏幕看，总耗时（Total Latency）和吞吐量（Throughput）才是核心，首字快慢无所谓。

因此，我们需要一套标准化的度量衡，来量化模型在不同负载下的表现。

## 2. 延迟（Latency）：用户感知的核心

延迟通过时间维度衡量模型的响应速度，直接决定了用户体验（UX）。

### 2.1 首字延迟 (Time to First Token, TTFT)

TTFT 指从用户发出请求到即时看到第一个输出 Token 的时间。

- **定义**：`TTFT = t_first_token - t_request_start`
- **重要性**：它是用户感知的“响应灵敏度”。在聊天场景中，TTFT 应尽量低于 **200ms** (极速) 或 **500ms** (流畅)。
- **技术瓶颈**：主要受 **Prefill（预填充）** 阶段影响，即模型处理输入 Prompt 的计算耗时。

### 2.2 Token 生成时间 (TPOT & ITL)

当第一个 Token 生成后，后续 Token 的流式输出速度决定了用户的阅读体验。这里有两个密切相关的指标：

- TPOT (Time Per Output Token)：平均每生成一个 Token 需要的时间。
  $$
  \text{TPOT} = \frac{\text{End-to-End Latency} - \text{TTFT}}{\text{Total Output Tokens} - 1}
  $$

- **ITL (Inter-Token Latency)**：两个连续 Token 产生之间的时间间隔。

> **专家提示**：为了保证“跟手”的流畅感，TPOT 应该低于人类的阅读速度。通常建议 TPOT < **50-100ms**（即每秒生成 10-20 个 Token）。如果 TPOT 过高，用户会感觉机器在“卡顿”或“思考”。

### 2.3 端到端延迟 (End-to-End Latency, E2EL)

从发送请求到接收完最后一个 Token 的总时长。

- **计算**：`E2EL = TTFT + (TPOT * Output_Tokens)`
- **适用场景**：非流式应用（如代码自动补全、离线文章生成）。

### 2.4 长尾延迟：P99 vs Average

看平均值（Mean）往往会掩盖问题。在生产环境中，**P99 延迟**（99th Percentile）才是系统的真实底线。

- **Mean/Median**：反映大多数用户的体验。
- **P99**：反映最慢的那 1% 请求的体验。如果 P99 很高，说明系统在负载波动或处理长 Prompt 时存在严重的性能抖动。

## 3. 吞吐量（Throughput）：系统效率的标尺

吞吐量衡量单位时间内系统能处理多少“工作量”，直接关系到**算力成本**和**并发能力**。

### 3.1 每秒请求数 (Requests per Second, RPS)

系统每秒能完成多少个完整的推理请求。

- **公式**：$\text{RPS} = \frac{\text{Total Requests}}{\text{Time Window}}$
- **局限性**：RPS 忽略了请求的复杂度。处理 10 个“你好”和处理 10 个“帮我写篇论文”，对算力的消耗是天壤之别。

### 3.2 每秒 Token 数 (Tokens per Second, TPS)

比 RPS 更细粒度的指标，衡量系统每秒吞吐的 Token 总量。

- **Input TPS (Prefill)**：系统每秒能“阅读”处理多少输入 Token。
- **Output TPS (Decode)**：系统每秒能“创作”生成多少输出 Token。

> **场景差异**：
>
> - **RAG/文档分析**：输入极长，输出较短 -> 关注 **Input TPS**。
> - **创意写作**：输入短，输出长 -> 关注 **Output TPS**。

## 4. 有效吞吐量（Goodput）：真实业务价值

单纯追求高吞吐量（Throughput）可能导致延迟（Latency）爆炸，变得不可用。**Goodput** 是指在**满足特定 SLA（服务等级协议）前提下**的最大吞吐量。

示例：

如果你的 SLA 要求 TTFT < 200ms。

- **系统 A**：TPS 1000，但 TTFT 普遍为 500ms。 -> **Goodput = 0**
- **系统 B**：TPS 800，且 99% 请求 TTFT < 200ms。 -> **Goodput = 800**

Goodput 才是衡量商业化推理服务真实能力的“黄金指标”。

## 5. 核心权衡：Latency vs. Throughput

在资源有限（GPU 显存/算力固定）的情况下，延迟和吞吐量往往是互斥的。

| **优化目标**   | **典型手段**                        | **代价**                 | **适用场景**                       |
| -------------- | ----------------------------------- | ------------------------ | ---------------------------------- |
| **极致低延迟** | 小 Batch Size (如 1)                | GPU 利用率低，单位成本高 | 高频交易、VIP 客服、实时语音交互   |
| **极致高吞吐** | 大 Batch Size (如 128+)             | 排队等待增加，TTFT 变长  | 离线数据清洗、批量翻译、非实时分析 |
| **平衡态**     | 动态 Batching (Continuous Batching) | 系统复杂度增加           | 通用 Chatbot API (如 ChatGPT)      |

### 关键调节旋钮

要在这两者间找到平衡，你需要调整以下参数：

- **Batch Size**：批处理大小。
- **并行策略**：Tensor Parallelism (TP), Pipeline Parallelism (PP)。
- **量化精度**：FP16 vs INT8/FP8。
- **KV Cache 策略**：PagedAttention 等显存管理技术。

## 6. 总结与下一步

理解这些指标是进行推理优化的第一步。不要盲目相信“每秒多少 Token”的宣传，而要结合你的业务场景（是重首字延迟，还是重总吞吐？）来制定测试标准。
