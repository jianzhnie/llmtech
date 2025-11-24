# vLLM 大模型服务：数据并行（DP）部署模式深度解析与优化实践

## 摘要


vLLM 通过其革命性的 PagedAttention 机制，极大地提高了大型语言模型（LLM）的服务吞吐量。为了进一步提升部署的规模和效率，vLLM 提供了灵活的数据并行（Data Parallelism, DP）部署方案。本文将详细解析 vLLM 核心的 DP 部署模式（内部负载均衡与外部负载均衡），并重点介绍 vLLM-Ascend 针对外部 DP 部署提供的优化工具，助力用户实现大规模、高性能的 LLM 推理服务。


## 1. vLLM 数据并行部署概述


在 vLLM 的数据并行架构中，每个 DP 进程被视为一个独立的“核心引擎”（Core Engine）实例，并通过 ZMQ 套接字与前端（Front-end）进程进行通信。


### 1.1 并行组合策略


数据并行可以与其他并行技术结合使用，以适应不同模型的部署需求：

- **数据并行 (DP) + 张量并行 (TP):** 每个 DP 引擎内部可以包含多个 GPU Worker 进程，其数量由配置的 TP 大小决定。这种组合允许在多个节点/设备上对模型进行数据分片，同时在每个设备组内进行张量分片。
  - *示例配置：* `--data-parallel-size=4 --tensor-parallel-size=2` 意味着总共需要 8 个设备，分成 4 组 DP，每组 2 个 TP。
- **专家并行 (EP) 与 MoE 模型:** 对于 Mixture-of-Experts (MoE) 模型，DP 和 TP/EP 可以协同工作。在 TP 与 DP 结合的情况下，专家层会形成一个大小为 `(DP x TP)` 的并行组。


### 1.2 负载均衡的重要性


无论采用何种 DP 模式，对请求进行智能负载均衡都至关重要。由于每个 DP 引擎都拥有独立的 KV 缓存，通过实时考量每个引擎的状态（例如，当前已调度和等待中的请求、KV 缓存状态），可以智能地分配新的提示词（Prompt），从而最大化**前缀缓存（Prefix Caching）**的效益。

vLLM 为在线部署（通过 API 服务器）提供了两种核心模式。


## 2. 内部负载均衡（Internal Load Balancing）模式


内部负载均衡模式提供了一个**自包含**的 DP 部署方案，它对外只暴露**一个统一的 API 入口**。

### 2.1 单节点部署


用户可以通过简单的命令行参数在一个节点内启动 DP 部署：

```bash
vllm serve $MODEL --data-parallel-size 4 --tensor-parallel-size 2
```

> ⚙️ **说明:** 此命令将在一个拥有 8 块 GPU 的节点上启动 DP=4、TP=2 的部署。

### 2.2 多节点部署


跨多节点部署时，需要在每个节点上运行不同的 `vllm serve` 命令，指定该节点应承载的 DP 进程（Rank）。在这种情况下，API 服务器（HTTP 入口）将仅在一个节点上运行，但它不必与 DP 引擎共同位于同一设备上。

| **参数**                     | **描述**                                   |
| ---------------------------- | ------------------------------------------ |
| `--data-parallel-size`       | 总的 DP 进程数（全局）。                   |
| `--data-parallel-size-local` | 当前节点上运行的 DP 进程数（本地）。       |
| `--data-parallel-start-rank` | 当前节点上本地 DP 进程的起始 Rank 编号。   |
| `--data-parallel-address`    | 用于 DP 进程间 RPC 通信的“头节点”IP 地址。 |
| `--data-parallel-rpc-port`   | 用于 DP 进程间 RPC 通信的端口。            |
| `--headless`                 | 标识为非头节点，不启动 API 服务器。        |

 **示例1：在两个节点上部署 DP=4，其中节点 0 运行 Rank 0 和 1，节点 1 运行 Rank 2 和 3。**

```Bash
# 节点 0 (IP: 10.99.48.128)，作为 API 服务器
vllm serve $MODEL --data-parallel-size 4 --data-parallel-size-local 2 \
    --data-parallel-address 10.99.48.128 --data-parallel-rpc-port 13345

# 节点 1 (引擎节点)
vllm serve $MODEL --headless --data-parallel-size 4 --data-parallel-size-local 2 \
    --data-parallel-start-rank 2 \
    --data-parallel-address 10.99.48.128 --data-parallel-rpc-port 13345
```

**示例2：在第一个节点上仅运行 API 服务器, 在第二个节点上运行所有引擎，实现 DP=4。**

```bash
# Node 0  (with ip address 10.99.48.128)
vllm serve $MODEL --data-parallel-size 4 --data-parallel-size-local 0 \
                  --data-parallel-address 10.99.48.128 --data-parallel-rpc-port 13345
# Node 1
vllm serve $MODEL --headless --data-parallel-size 4 --data-parallel-size-local 4 \
                  --data-parallel-address 10.99.48.128 --data-parallel-rpc-port 13345

```

### 2.3 使用  Ray 后端

使用 `--data-parallel-backend=ray` 可以简化多节点部署，仅需一个启动命令即可启动所有本地和远程 DP 进程，无需手动指定 `--data-parallel-address`。

```bash
vllm serve $MODEL --data-parallel-size 4 --data-parallel-size-local 2 \
                  --data-parallel-backend=ray
```

使用 Ray 时有几个显著差异：

- 只需在任意节点执行单一启动命令即可启动所有本地和远程 DP 进程，相比在每个节点单独启动更为便捷
- 无需指定 `--data-parallel-address`，运行命令的节点将自动作为 `--data-parallel-address` 使用
- 无需指定 `--data-parallel-rpc-port`
- 当一个数据并行组需要多个节点时， *例如*单个模型副本需在至少两个节点上运行，请务必设置 `VLLM_RAY_DP_PACK_STRATEGY="span"` ，此时 `--data-parallel-size-local` 参数将被忽略并自动确定
- 远程 DP 进程将根据 Ray 集群的节点资源进行动态分配



## 3. 外部负载均衡（External Load Balancing）模式

对于超大规模部署，将 DP 进程的编排和负载均衡交给外部系统管理更具优势。


### 3.1 核心概念


在外部 DP 模式下：

1. 每个 DP 进程被视为一个独立的 vLLM 部署实例。
2. 每个实例拥有独立的 API 端点。
3. 一个外部的路由或代理服务负责在这些 DP 实例之间平衡 HTTP 请求。

这种模式的灵活性在于外部路由可以利用每个 vLLM 服务器的实时遥测数据（Telemetry）来制定更优的路由决策，例如根据 KV 缓存的命中率或队列长度进行动态调整。

### 3.2 示例

如果 DP Rank 位于同一位置（相同节点/IP 地址），则使用默认的 RPC 端口，但必须为每个Rank指定不同的 HTTP 服务器端口：

```bash
# Rank 0
CUDA_VISIBLE_DEVICES=0 vllm serve $MODEL --data-parallel-size 2 --data-parallel-rank 0 \
                                         --port 8000
# Rank 1
CUDA_VISIBLE_DEVICES=1 vllm serve $MODEL --data-parallel-size 2 --data-parallel-rank 1 \
                                         --port 8001
```

对于多节点情况，还必须指定 rank 0 的地址/端口：

```bash
# Rank 0  (with ip address 10.99.48.128)
vllm serve $MODEL --data-parallel-size 2 --data-parallel-rank 0 \
                  --data-parallel-address 10.99.48.128 --data-parallel-rpc-port 13345
# Rank 1
vllm serve $MODEL --data-parallel-size 2 --data-parallel-rank 1 \
                  --data-parallel-address 10.99.48.128 --data-parallel-rpc-port 13345
```

在此场景下，协调器进程同样运行，并与数据并行（DP）Rank 0 引擎共置一处。



## 4. vLLM-Ascend 外部 DP 部署实战

针对在昇腾（Ascend）硬件上的部署场景，vLLM-Ascend 提供了两个关键的增强功能，极大地简化了外部 DP 的部署和优化：

1. **一键启动脚本:** 帮助用户通过一条命令快速启动多个 vLLM 实例（DP Rank）。
2. **请求长度感知负载均衡代理 (Request-Length-Aware Load Balance Proxy):** 一个专门设计的代理服务器，可根据请求的上下文长度进行智能路由，进一步优化吞吐量。

本节以 vLLM-Ascend 为例，演示外部 DP 的具体部署流程。

### 4.1 前提条件

1. Python 3.10+ 环境。

2. 安装负载均衡代理所需的依赖：

   ```bash
   pip install fastapi httpx uvicorn
   ```

### 4.2 启动外部 DP 服务器

您可以通过手动方式启动，但对于大规模部署，我们推荐使用提供的启动脚本。

#### 4.2.1 手动启动示例（DP Size 2）

手动为每个 DP 实例指定唯一的端口和 Rank：

```Bash
# vLLM DP Rank 0
vllm serve --host 0.0.0.0 --port 8100 --data-parallel-size 2 --data-parallel-rank 0 ...

# vLLM DP Rank 1
vllm serve --host 0.0.0.0 --port 8101 --data-parallel-size 2 --data-parallel-rank 1 ...
```

#### 4.2.2 使用启动脚本（推荐）

通过修改 `examples/external_online_dp/run_dp_template.sh` 配置后，使用 `launch_online_dp.py` 脚本实现一键启动。

**A. 单节点启动示例 (DP=4, TP=4)**

假设在一个拥有 8 个 NPU 的节点上启动：

```bash
cd examples/external_online_dp
python launch_online_dp.py \
    --dp-size 2 \
    --tp-size 4 \
    --dp-size-local 2 \
    --dp-rank-start 0 \
    --dp-address x.x.x.x \
    --dp-rpc-port 12342
```

**B. 两节点分布式启动示例 (DP=4, TP=4)**

假设两个节点各拥有 8 个 NPU，节点 0 负责 DP Rank 0 和 1，节点 1 负责 DP Rank 2 和 3。

```bash
cd examples/external_online_dp
# 节点 0:
python launch_online_dp.py \
    --dp-size 4 --tp-size 4 --dp-size-local 2 \
    --dp-rank-start 0 \
    --dp-address x.x.x.x --dp-rpc-port 12342

# 节点 1:
python launch_online_dp.py \
    --dp-size 4 --tp-size 4 --dp-size-local 2 \
    --dp-rank-start 2 \
    --dp-address x.x.x.x --dp-rpc-port 12342
```

### 4.3 启动负载均衡代理服务器

在所有 vLLM DP 实例启动完毕后（默认从 9000 端口开始递增），即可启动负载均衡代理服务器作为统一的请求入口。

**代理服务器特性：**

- 基于请求长度的智能负载均衡。
- 支持 OpenAI 兼容的 `/v1/completions` 和 `/v1/chat/completions` 端点。
- 支持从后端服务器到客户端的流式（Streaming）响应。

**示例：启动代理服务器**

假设，已经在单个节点上启动了两个 vLLM DP 部署实例,  DP Rank 0 和 Rank 1 分别运行在 9000 和 9001 端口：

```Bash
cd examples/external_online_dp
python dp_load_balance_proxy_server.py \
    --host 0.0.0.0 --port 8000 \
    --dp-hosts 127.0.0.1 127.0.0.1 \
    --dp-ports 9000 9001
```

现在，所有客户端请求可以直接发送到代理服务器的 `0.0.0.0:8000` 地址，代理将负责将请求智能路由到不同的 vLLM DP 实例，从而实现外部负载均衡的 DP 部署。
