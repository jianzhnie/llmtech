# SGLang 服务基准测试权威指南：LLM 性能调优与实战


随着大语言模型（LLM）在生产环境中的广泛应用，评估和优化其在线服务（Serving）的性能变得至关重要。SGLang 提供了一个强大且灵活的基准测试工具，通过其内置的 `sglang.bench_serving` 模块，开发者可以对不同推理后端、不同负载模式下的 LLM 服务进行全面、准确的性能测量。

本指南将详细介绍如何使用 `python -m sglang.bench_serving` 命令来测试在线服务的吞吐量和延迟，并深入解析各项核心指标和高级配置。

## 一、核心功能与性能指标

`sglang.bench_serving` 工具的核心目标是模拟真实的在线请求负载，并精确测量服务端的关键性能数据。

### 1. 功能概述

- **负载生成：** 根据数据集或合成数据生成提示（Prompts），并将其提交给目标服务接口。
- **模式支持：** 支持流式（Streaming）和非流式（Non-streaming）模式。
- **流量控制：** 支持精确的请求速率（Rate Control）控制和并发请求数量限制（Concurrency Limits）。

### 2. 核心测量指标

该工具产出的指标全面覆盖了服务性能的关键方面：

| **性能指标**         | **英文简称**               | **解释**                                                       |
| -------------------- | -------------------------- | -------------------------------------------------------------- |
| **请求吞吐量**       | Req/s                      | 每秒处理的请求数。                                             |
| **Token 吞吐量**     | Tok/s                      | 每秒处理的 Token 总数（包含输入和输出）。                      |
| **端到端延迟**       | End-to-End Latency         | 单个请求从发送到完整接收的全部耗时。                           |
| **首个 Token 时间**  | TTFT (Time to First Token) | 从请求发送到接收到第一个 Token 的时间（流式模式关键指标）。    |
| **Token 间延迟**     | ITL (Inter-Token Latency)  | 连续两个输出 Token 之间的时间间隔。                            |
| **Token 处理时间**   | TPOT                       | TTFT 后的 Token 平均处理时间：$(延迟 - TTFT) / (Tokens - 1)$。 |
| **并发度**           | Concurrency                | 所有请求的总耗时聚合除以实际测试的挂钟时间。                   |
| **推测解码接受长度** | Accept length              | SGLang 专属指标，衡量推测解码的平均接受长度。                  |

## 二、支持的后端和服务接口

`bench_serving` 工具通过统一的接口（主要是 OpenAI 兼容的 API）支持多种主流的 LLM 推理后端：

| **后端名称**                                    | **兼容性/框架**                 | **接口路径**                               |
| ----------------------------------------------- | ------------------------------- | ------------------------------------------ |
| `sglang` / `sglang-native`                      | SGLang 原生 API                 | `POST /generate`                           |
| `sglang-oai`, `vllm`, `lmdeploy`                | OpenAI 兼容（Completions）      | `POST /v1/completions`                     |
| `sglang-oai-chat`, `vllm-chat`, `lmdeploy-chat` | OpenAI 兼容（Chat Completions） | `POST /v1/chat/completions`                |
| `trt`                                           | TensorRT-LLM                    | `POST /v2/models/ensemble/generate_stream` |
| `truss`                                         | Truss Framework                 | `POST /v1/models/model:predict`            |

### 快速开始

在开始基准测试前，请确保您的目标推理服务已启动并可访问。

**1. 启动 SGLang 服务：**

```bash
python3 -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct
```

**2. 运行基础基准测试（针对 SGLang 原生接口）：**

```Bash
python3 -m sglang.bench_serving \
 --backend sglang \
 --host 127.0.0.1 --port 30000 \
 --num-prompts 1000 \
 --model meta-llama/Llama-3.1-8B-Instruct
```

**3. 运行 OpenAI 兼容接口测试（例如 vLLM）：**

```Bash
python3 -m sglang.bench_serving \
 --backend vllm \
 --base-url http://127.0.0.1:8000 \
 --num-prompts 1000 \
 --model meta-llama/Llama-3.1-8B-Instruct
```

## 三、灵活的数据集配置

基准测试的准确性高度依赖于模拟的请求特征。`bench_serving` 提供了多样化的数据集来满足不同的测试需求，通过 `--dataset-name` 参数选择。

| **数据集名称**            | **特点与用途**                                                  | **关键参数**                                                               |
| ------------------------- | --------------------------------------------------------------- | -------------------------------------------------------------------------- |
| `sharegpt` (默认)         | 加载 ShareGPT 格式的对话数据，模拟真实用户请求分布。            | `--sharegpt-context-len`, `--sharegpt-output-len`                          |
| `random`                  | 随机生成输入和输出长度，内容采自 ShareGPT Token 空间。          | `--random-input-len`, `--random-output-len`, `--random-range-ratio`        |
| `image`                   | 专为 VLM（视觉语言模型）设计，生成包含图像数据的聊天请求。      | `--image-count`, `--image-resolution`, `--image-format`, `--image-content` |
| `generated-shared-prefix` | 合成数据集，用于测试 KV Cache 共享效率（长系统提示 + 短问题）。 | `--gsp-system-prompt-len`, `--gsp-question-len`                            |
| `mmmu`                    | 采样自 MMMU 数据集（Math 分割），包含图像和多模态元素。         | 需要额外的依赖包。                                                         |

### 数据集示例：多模态 VLM 测试

要测试一个多模态模型（如 Qwen-VL），模拟每个请求包含 3 张 720p 图像的场景：

**1. 启动 SGLang 服务：**

```
python -m sglang.launch_server --model-path Qwen/Qwen2.5-VL-3B-Instruct --disable-radix-cache
```

**2. 运行基础基准测试：**

```bash
python3 -m sglang.bench_serving \
 --backend sglang-oai-chat \
 --dataset-name image \
 --num-prompts 500 \
 --image-count 3 \
 --image-resolution 720p \
 --random-input-len 512 \
 --random-output-len 512
```

## 四、高级配置选项

为了精细化地控制测试过程和结果输出，以下是几个关键的高级选项：

### 1. 流量控制与模式选择

| **参数**            | **描述**                                                                                             |
| ------------------- | ---------------------------------------------------------------------------------------------------- |
| `--request-rate`    | 设置每秒发送的请求数。设置为 `inf` 则立即全部发送（突发模式）。非 `inf` 值将模拟泊松分布的到达时间。 |
| `--max-concurrency` | 限制最大并发进行中的请求数，防止服务器过载。                                                         |
| `--disable-stream`  | 禁用流式输出模式，切换为非流式。在这种模式下，TTFT 将等于总延迟。                                    |

### 2. 模型、分词器与数据格式

- `--model`：指定模型 ID 或本地路径（除非后端支持 `GET /v1/models` 自动获取）。
- `--tokenizer`：指定分词器 ID 或本地路径，默认与 `--model` 相同。
- `--apply-chat-template`：在构建提示时，应用分词器的聊天模板，确保格式正确。
- `--tokenize-prompt`：发送整数 ID 而非文本，用于严格的长度控制（目前仅支持 `sglang` 后端）。

### 3. 输出与定制化

- `--output-file FILE.jsonl`：将详细的 JSONL 结果追加到指定文件。
- `--output-details`：在 JSONL 输出中包含 per-request 细节数组（如生成的文本、错误、TTFTs 等）。
- `--extra-request-body '{"top_p":0.9,"temperature":0.6}'`：允许您以 JSON 字符串形式向请求体添加额外的采样参数或配置。

### 4. 优化与调试

| **参数**                | **描述**                                                                 |
| ----------------------- | ------------------------------------------------------------------------ |
| `--warmup-requests N`   | 在正式运行前，先执行 N 个短输出请求进行预热（默认 1）。                  |
| `--flush-cache`         | 在正式测试前调用 `/flush_cache` 接口（仅支持 SGLang）。                  |
| `--profile`             | 运行基准测试的同时启动和停止服务器端的性能分析（需要服务器端启用配置）。 |
| `--lora-name name1 ...` | 随机选择一个 LoRA 名称，并通过请求体传递给后端。                         |

### 5. 鉴权配置

如果您的服务接口需要 OpenAI 风格的 API 密钥验证，请设置环境变量 `OPENAI_API_KEY`。基准测试脚本会自动在请求头中添加 `Authorization: Bearer <key>`。

```Bash
export OPENAI_API_KEY=sk-...yourkey...
```

## 五、端到端实战案例

### 1. 严格流量控制与细节输出 (SGLang 原生)

此示例在高并发下测试 SGLang 的原生接口，并以 100 req/s 的速率均匀发送请求，同时输出详细的 JSONL 结果。

```bash
python3 -m sglang.bench_serving \
 --backend sglang \
 --host 127.0.0.1 --port 30000 \
 --model meta-llama/Llama-3.1-8B-Instruct \
 --dataset-name random \
 --random-input-len 1024 --random-output-len 1024 --random-range-ratio 0.5 \
 --num-prompts 2000 \
 --request-rate 100 \
 --max-concurrency 512 \
 --output-file sglang_random.jsonl --output-details
```

### 2. 共享前缀优化测试 (Generated Shared Prefix)

使用合成的长系统提示和短问题数据集，专门用于衡量 KV Cache 共享（如 SGLang 的 HiCache）的效率。

```Bash
python3 -m sglang.bench_serving \
 --backend sglang \
 --host 127.0.0.1 --port 30000 \
 --model meta-llama/Llama-3.1-8B-Instruct \
 --dataset-name generated-shared-prefix \
 --gsp-num-groups 64 --gsp-prompts-per-group 16 \
 --gsp-system-prompt-len 2048 --gsp-question-len 128 --gsp-output-len 256 \
 --num-prompts 1024
```

### 3. OpenAI 兼容的聊天接口测试 (vLLM Chat)

针对 vLLM 提供的 Chat Completions 接口进行流式测试，并应用聊天模板以确保输入格式正确。

```Bash
python3 -m sglang.bench_serving \
 --backend vllm-chat \
 --base-url http://127.0.0.1:8000 \
 --model meta-llama/Llama-3.1-8B-Instruct \
 --dataset-name random \
 --num-prompts 500 \
 --apply-chat-template
```

## 六、故障排除提示

- **请求全部失败：** 仔细检查 `--backend` 名称、`--base-url` 或 `--host`/`--port` 是否正确，确认服务器是否正在运行，并且 `--model` 名称是否被服务器支持。
- **吞吐量过低：** 尝试调整 `--request-rate` 和 `--max-concurrency` 以匹配服务器的极限性能。检查服务器端的批处理大小（Batch Size）和调度策略。
- **Token 计数异常：** 优先使用带有正确聊天模板的 Chat/Instruct 模型。对于非结构化文本的 Token 计数可能存在不一致。
- **鉴权错误 (401/403)：** 确认已正确设置 `OPENAI_API_KEY` 环境变量。

使用 `sglang.bench_serving` 工具，您可以系统地识别服务瓶颈，并通过调整服务器配置（如 KV Cache 策略、批处理大小、模型量化）或 SGLang 的优化特性来最大化 LLM 的服务性能。
