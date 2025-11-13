# SGLang Model Gateway（原 Router）入门指南

在大规模部署大型语言模型（Large Language Models, LLMs）的生产环境中，核心挑战在于如何高效管理异构模型实例、平衡高并发请求负载，并确保企业级服务可靠性和数据隐私。SGLang 推出的 **SGLang Model Gateway**（模型网关，前称 SGLang Router）是一款专为解决这些复杂问题而设计的**高性能、多功能路由网关**。

它不仅仅是一个简单的负载均衡器，更是为 LLM 推理工作负载深度定制的**中央控制系统**。

如果你正在部署  Qwen、DeepSeek、Kimi 等大型语言模型，并希望：

- 降低响应延迟
- 提高 GPU 利用率
- 支持多机集群部署
- 实现自动容错和弹性伸缩

那么，**SGLang Model Gateway** 正是你需要的核心组件。

## 🔍 什么是 SGLang Model Gateway？

SGLang Model Gateway（以下简称 Gateway）可以视为所有 LLM 服务器（Worker）集群的前端门面和中央调度系统。

设想这样一个场景：你有 4 台 GPU 服务器都在运行同一个大模型。当用户提出问题："巴黎是哪个国家的首都？"

在没有调度器的情况下，你可能只能将请求固定发送给某一台服务器，导致其他机器空闲；或者采用随机分配的方式，但每台服务器都需要重新计算整个上下文，造成算力浪费。

而 **SGLang Router 就像是一个"智能调度中心"**：

- 它了解哪台服务器已缓存了相似的问题（例如之前有人询问过"法国首都是哪里？"）
- 它会优先将新请求分配给相应服务器，直接复用现有缓存（KV Cache），节省时间和计算资源
- 当某台服务器发生故障时，它能自动重试或将请求路由到其他可用节点，确保服务持续可用

核心功能可概括为以下三点：

1. **集中管理**：负责注册、监控和管理所有后端 LLM Worker 实例的生命周期，即使这些 Worker 使用不同的通信协议（如 HTTP、gRPC）
2. **智能路由**：根据 Worker 的实时负载、缓存状态（Cache-Aware）等因素，将请求智能地分配给最合适的 Worker，实现高吞吐量和低延迟
3. **企业级特性**：在网关层提供故障恢复（重试、熔断）、限流、会话历史存储（保护隐私）等企业级功能，确保 LLM 服务的稳定性和安全性

## 🎯 技术洞察：Model Gateway 的核心定位


SGLang Model Gateway 是一款专为 LLM 推理服务打造的高性能 API 网关。通过在应用层和模型服务层之间建立统一入口，实现了对后端异构模型服务集群的集中控制和智能流量分发。

### 1. 架构核心：分离的控制平面与数据平面

Gateway 的架构设计借鉴了微服务网关的最佳实践，并针对 LLM 推理特性进行了优化：

- **控制平面 (Control Plane)**
  - **Worker 生命周期管理**：负责发现 Worker 能力（通过 `/get_server_info` 等接口）、实时跟踪负载状态，并执行 Worker 的注册与移除操作
  - **健康检查与容错机制**：持续探测 Worker 状态，更新其就绪性、熔断器状态，并向负载监测器（Load Monitor）提供实时数据

- **数据平面 (Data Plane)**
  - **多协议路由**：统一处理来自客户端的 HTTP、gRPC 以及 OpenAI 兼容 API 请求，并将其路由至相应的后端 Worker
  - **高吞吐 gRPC 管线**：提供业界领先的 gRPC 路由管道，完全在 Rust 中运行，集成了原生分词器、推理解析器和工具调用执行器，确保极高的吞吐量
  - **OpenAI 兼容代理**：能够代理外部的 OpenAI 兼容服务（如 OpenAI、xAI），同时在本地网关层保留对话历史，实现**数据隐私的本地化控制**

### 2. 智能调度与可靠性保证

Gateway 提供了企业级部署所需的关键机制：

| **功能模块**                 | **作用（技术专家视角）**                                                                                                                               |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **负载均衡策略**             | 支持 `round_robin`（轮询）、`power_of_two`（二次幂选择），以及对 LLM 至关重要的 **`cache_aware`（缓存感知）** 策略，实现基于 KV 缓存命中率的最优路由。 |
| **可靠性与流控**             | 内置**指数退避重试**（Retries）、基于失败阈值的**熔断器**（Circuit Breaker），以及基于令牌桶模型的**速率限制与排队**（Rate Limiting & Queuing）。      |
| **数据隐私与历史存储**       | 集中存储会话和响应历史（支持内存、Oracle ATP 等），使多轮代理式工作流（`/v1/responses`）和 MCP 流程在**不向外部供应商暴露历史数据**的前提下运行。      |
| **可观测性 (Observability)** | 深度集成 Prometheus 指标、结构化追踪和请求 ID 传播，提供详细的作业队列状态，便于生产环境监控。                                                         |

## 🧩 核心特性详解

### 1️⃣ 缓存感知负载均衡（Cache-Aware Load Balancing）

- 默认策略！不是简单轮询，而是**优先选择缓存匹配度高的服务器**
- 例如，当请求的前 100 个 token 与某台服务器缓存一致时，直接接续生成，无需从头计算
- 在长对话或多轮问答场景下，性能提升尤为显著

### 2️⃣ 多种部署模式，灵活适配

| 模式                        | 适用场景     | 特点                                    |
| --------------------------- | ------------ | --------------------------------------- |
| **一体化模式**              | 单机测试     | 一条命令启动 Router + 多个 Worker       |
| **独立部署模式**            | 多机集群     | Router 和 Worker 分开部署，支持灵活组合 |
| **Prefill/Decode 解耦模式** | 异构硬件优化 | GPU 负责复杂预填充，CPU 负责轻量解码    |

> 💡 **Prefill/Decode 解耦** 是高级部署模式：将"理解上下文"（prefill）和"逐字生成"（decode）任务分配到不同类型的硬件上，让昂贵的 GPU 资源专注于计算密集型任务，而将轻量级解码任务交给 CPU 处理。

### 3️⃣ 自动容错，保障服务稳定性

- **自动重试**：请求失败时自动重试最多 3 次，采用指数退避策略
- **熔断机制**：当某台服务器连续失败达到阈值时，暂时隔离该节点，待其恢复正常后再恢复服务
- 有效避免雪崩效应，保障整体服务稳定性

### 4️⃣ 动态扩缩容 + Kubernetes 支持

- 运行中动态添加服务器节点：

  ```bash
  curl -X POST "http://router:30000/add_worker?url=http://new-worker:8000"
  ```

- 在 Kubernetes 环境中支持通过标签自动发现 Pod，无需手动配置 IP 地址

## 🛠️ 使用说明：五种部署模式

Gateway 支持多种灵活的部署模式，以适应从快速原型开发到大规模生产集群的各种应用场景。

### 1. 一体化启动（Co-launch Router + Workers）

这是最简单的启动方式，Gateway 和 SGLang Worker 在同一个进程中启动，适用于单节点或快速测试。

场景： 快速验证或单服务器部署。

```bash
# SGLang Worker 的参数（无前缀）
# --model: 指定模型
# --dp-size: 数据并行大小

# Router 的参数（需添加 --router- 前缀）
# --router-policy: 路由策略
# --router-log-level: 网关日志级别

python -m sglang_router.launch_server \
 --model meta-llama/Meta-Llama-3.1-8B-Instruct \
 --dp-size 4 \
 --host 0.0.0.0 \
 --port 30000 \
 --router-policy round_robin  # 使用 router- 前缀配置网关
```

完整示例：

```bash
python3 -m sglang_router.launch_server \
  --host 0.0.0.0 \
  --port 8080 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --tp-size 1 \
  --dp-size 8 \
  --grpc-mode \
  --log-level debug \
  --router-prometheus-port 10001 \
  --router-tool-call-parser llama \
  --router-health-success-threshold 2 \
  --router-health-check-timeout-secs 6000 \
  --router-health-check-interval-secs 60 \
  --router-model-path meta-llama/Llama-3.1-8B-Instruct \
  --router-policy round_robin \
  --router-log-level debug
```

### 2. 独立 HTTP 模式（Separate Launch）

将 Gateway 和 Worker 分开部署，Worker 独立运行并暴露 HTTP  API 接口，Gateway 通过 URL 列表发现并连接它们。

场景： 多节点部署、Worker 独立扩缩容。

```Bash
# 步骤一：在 Worker 节点上启动 SGLang Worker
# Worker 1 (假设 IP/Host 为 worker1)
python -m sglang.launch_server --model ... --port 8000 &
# Worker 2 (假设 IP/Host 为 worker2)
python -m sglang.launch_server --model ... --port 8001 &

# 步骤二：在 Gateway 节点上启动 Router, 通过 --worker-urls 指定所有 Worker 的 HTTP 端点
python -m sglang_router.launch_router \
 --worker-urls http://worker1:8000 http://worker2:8001 \
 --policy cache_aware \
 --host 0.0.0.0 --port 30000
```

### 3. 高性能 gRPC 模式（gRPC Launch）

利用 SGLang 的 SRT gRPC 协议，可以获得最高的吞吐量，并且能够利用 Rust 实现的原生推理和工具解析管道。

场景： 对延迟和吞吐量有极高要求的生产环境。

```Bash
# 步骤一：Worker 节点以 gRPC 模式启动
python -m sglang.launch_server \
 --model meta-llama/Llama-3.1-8B-Instruct \
 --grpc-mode \
 --port 20000

# 步骤二：启动 Router，连接 gRPC 接口
python -m sglang_router.launch_router \
 --worker-urls grpc://127.0.0.1:20000 \
 --model-path meta-llama/Llama-3.1-8B-Instruct \
 --host 0.0.0.0 --port 8080
```

> gRPC 路由器同时支持单阶段和 PD 服务模式。需提供 `--tokenizer-path` 或 `--model-path`（HuggingFace 模型路径或本地目录），以及可选的 `--chat-template` 参数。

### 4. Prefill/Decode 解耦模式（PD Disaggregation）

将 LLM 推理的两个关键阶段 预填充（Prefill）和解码（Decode）的Worker分离到不同的 Worker 集群，利用 PD 感知缓存，实现对资源的精细化管理和负载均衡。

场景： 追求极致资源优化和复杂调度策略。

```Bash
python -m sglang_router.launch_router \
 --pd-disaggregation \
 --prefill http://prefill1:30001 \
 --decode http://decode1:30011 \
 --policy cache_aware \
 --prefill-policy cache_aware \
 --decode-policy power_of_two
```

### 5. 代理外部服务（OpenAI Backend Proxy）

Gateway 作为代理层，将请求转发给外部的 OpenAI 兼容 API。其核心价值在于，**多轮会话历史和 MCP 状态仍然在 Gateway 本地存储和管理**，确保多轮对话和 MCP（多步链式推理）流程的隐私性。

场景： 在使用外部 LLM 服务时，仍需保持对对话历史和数据流的控制。

```Bash
python -m sglang_router.launch_router \
 --backend openai \
 --worker-urls https://api.openai.com \
 --history-backend memory
```

> OpenAI 后端模式要求每个路由器实例仅配置一个 `--worker-urls` 条目。

## ⚙️ 高级技巧：按需调优

### 想要更强的缓存复用？

调整缓存阈值（默认 0.5，即 50% 匹配就复用）：

```bash
--cache-threshold 0.7
```

### 发现负载不均？

适当放宽负载均衡判断条件：

```bash
--balance-abs-threshold 64    # 默认 32
--balance-rel-threshold 1.1   # 默认 1.0001
```

### 在 Kubernetes 上部署？

```bash
python -m sglang_router.launch_router \
  --service-discovery \
  --selector app=sglang-worker env=prod \
  --service-discovery-namespace production
```

### 需要监控？

启用 Prometheus：

```bash
--prometheus-host 0.0.0.0 --prometheus-port 29000
```

然后访问 `http://your-router:29000/metrics` 查看实时指标。

## 🚨 常见问题解答

- **Q：Worker 启动慢，Router 报错连不上？**
  A：增加等待时间：`--worker-startup-timeout-secs 600`

- **Q：内存占用越来越高？**
  A：限制缓存树大小：`--max-tree-size 8388608`（约 800 万节点）

- **Q：想关闭熔断器？**
  A：添加参数 `--disable-circuit-breaker`

## 🌟 总结：为什么选择 Model Gateway

对于致力于构建高性能、可扩展 LLM 服务的 AI 工程师和架构师而言，SGLang Model Gateway 提供了：

- **极致性能**：通过 gRPC 管道和 `cache_aware` 调度策略，最大限度地提升集群的吞吐量和资源利用率
- **企业级可靠性**：内置熔断、重试和限流机制，显著提高了服务的健壮性
- **数据主权**：即使使用外部 LLM，也能通过本地历史存储确保对敏感会话数据的控制权

Model Gateway 不仅是 SGLang 生态系统中的关键组件，也是解决现代 LLM 大规模部署中复杂路由和管理挑战的专业工具。
