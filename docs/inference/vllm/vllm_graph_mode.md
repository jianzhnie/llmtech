# vLLM Ascend 深度优化指南：图模式加速与高级配置详解

## 导言

vLLM Ascend 是基于 vLLM 框架为昇腾（Ascend）硬件平台提供高性能大语言模型（LLM）推理的服务。为了最大限度地发挥昇腾 NPU 的计算潜力，vLLM Ascend 引入了图模式（Graph Mode）进行计算图优化，并通过附加配置（Additional Configuration）机制提供了灵活的底层行为控制。

本文将详细解析 vLLM Ascend 中的图模式加速原理、两种图模式的配置方法，以及全面的附加配置选项，帮助用户实现更高效、更定制化的模型部署。


## 一、图模式加速（Graph Mode Acceleration）

图模式（Graph Mode）旨在通过提前捕获、编译和优化模型的前向计算图，从而消除运行时开销，并进行深度融合优化，显著提升推理吞吐量和延迟表现。

### 1.1 注意事项与兼容性

- **实验性功能：** 图模式目前处于实验性阶段。未来的版本中，其配置方式、模型覆盖范围和性能可能会持续改进。
- **引擎要求：** 图模式**仅适用于 V1 Engine**。
- **默认启用：** 自 `v0.9.1rc1` 版本开始，vLLM Ascend 默认启用图模式，以与 vLLM 保持一致的行为。当前，Qwen 和 DeepSeek 系列模型已得到较充分的测试支持。

### 1.2 两种图模式类型

vLLM Ascend 支持两种主要的图模式实现，用户可根据模型类型选择：

| **图模式名称**    | **核心技术**                                  | **适用模型**                           | **启用方式**                   |
| ----------------- | --------------------------------------------- | -------------------------------------- | ------------------------------ |
| **ACLGraph**      | 基于 ACL（Ascend Computing Language）的图编译 | Qwen、DeepSeek 系列（测试充分）        | 默认启用（只需使用 V1 Engine） |
| **TorchAirGraph** | 基于 GE 图模式（Graph Engine）                | DeepSeek 系列、PanguProMoE（测试充分） | 需通过附加配置显式启用         |

### 1.3 ACLGraph 使用方法

ACLGraph 是默认的图模式。对于支持的模型（如 Qwen 系列），只需确保使用 V1 引擎即可自动启用。

**离线推理示例：**

```Python
import os
from vllm import LLM

# 默认使用 V1 引擎和 ACLGraph
model = LLM(model="path/to/Qwen2-7B-Instruct")
outputs = model.generate("Hello, how are you?")
```

**在线服务示例：**

```Bash
vllm serve Qwen/Qwen2-7B-Instruct
```

### 1.4 TorchAirGraph 使用方法

若需使用 TorchAirGraph（适用于 DeepSeek 等模型），则需要额外的配置。

**重要限制：** TorchAirGraph 在当前版本中**不兼容 Chunked-Prefill**。

**离线推理示例 (需使用 `additional_config`)：**

```Python

import os
from vllm import LLM

# 显式配置启用 TorchAirGraph 和 Ascend 调度器
model = LLM(
    model="path/to/DeepSeek-R1-0528",
    additional_config={
        "torchair_graph_config": {"enabled": True},
        "ascend_scheduler_config": {"enabled": True}
    }
)
outputs = model.generate("Hello, how are you?")
```

**在线服务示例：**

```Bash
vllm serve path/to/DeepSeek-R1-0528 \
    --additional-config='{"torchair_graph_config": {"enabled": true}, "ascend_scheduler_config": {"enabled": true}}'
```

### 1.5 回退到 Eager 模式

如果任一图模式运行失败或遇到兼容性问题，用户应回退到 Eager 模式（即非图模式）。

通过设置 `enforce_eager=True` 参数即可强制使用 Eager 模式。

**离线推理示例：**

```Python
import os
from vllm import LLM

# 强制回退到 Eager 模式
model = LLM(model="someother_model_weight", enforce_eager=True)
outputs = model.generate("Hello, how are you?")
```

**在线服务示例：**

```Bash
vllm serve someother_model_weight --enforce-eager
```
## 二、附加配置详解（Additional Configuration）

附加配置（`additional_config`）是 vLLM 提供的机制，允许插件（如 vLLM Ascend）灵活控制其内部行为。它通过一个字典或 JSON 字符串传入，用于配置一系列高级优化选项。

### 2.1 使用方法

无论是离线模式还是在线模式，附加配置都通过传递一个键值对字典来实现。

**离线模式 (`LLM` 初始化)：**


```Python
from vllm import LLM
LLM(model="Qwen/Qwen3-8B", additional_config={"config_key": "config_value"})
```

**在线模式 (`vllm serve` 命令)：**

```Bash
vllm serve Qwen/Qwen3-8B --additional-config='{"config_key":"config_value"}'
```

### 2.2 核心配置选项一览（顶级配置）

以下是在 vLLM Ascend 中可用的顶级附加配置选项：

| **名称**                            | **类型** | **默认值** | **描述**                                                     |
| ----------------------------------- | -------- | ---------- | ------------------------------------------------------------ |
| `torchair_graph_config`             | `dict`   | `{}`       | TorchAir 图模式配置选项（详见下文）                          |
| `ascend_scheduler_config`           | `dict`   | `{}`       | Ascend 调度器配置选项（详见下文）                            |
| `weight_prefetch_config`            | `dict`   | `{}`       | 权重预取功能配置选项（详见下文）                             |
| `refresh`                           | `bool`   | `False`    | 是否刷新全局昇腾配置。通常用于 RLHF 或 UT/E2E 测试。         |
| `kv_cache_dtype`                    | `str`    | `None`     | 使用 KV Cache 量化时，需设置 KV Cache 的数据类型。当前仅支持 `int8`。 |
| `enable_shared_expert_dp`           | `bool`   | `False`    | 当 MoE 模型的专家（Expert）在 DP（Data Parallel）中共享时启用。可提升性能但会增加内存消耗。目前仅支持 DeepSeek 系列模型。 |
| `expert_map_path`                   | `str`    | `None`     | MoE 模型使用专家负载均衡（EPLB）时，传递专家映射文件的路径。 |
| `dynamic_eplb`                      | `bool`   | `False`    | 是否启用动态 EPLB。                                          |
| `num_iterations_eplb_update`        | `int`    | `400`      | EPLB 开始更新的推理迭代次数。                                |
| `multistream_overlap_shared_expert` | `bool`   | `False`    | 是否启用多流共享专家。仅对带有共享专家的 MoE 模型生效。      |

### 2.3 TorchAir 图配置 (`torchair_graph_config`)

该字典用于控制 TorchAirGraph 的底层行为和优化策略。

| **名称**                  | **类型**    | **默认值** | **描述**                                                     |
| ------------------------- | ----------- | ---------- | ------------------------------------------------------------ |
| `enabled`                 | `bool`      | `False`    | **是否启用** TorchAir 图模式。                               |
| `mode`                    | `str`       | `None`     | 当使用 TorchAir 的 Reduce-Overhead 模式时需要设置。          |
| `enable_view_optimize`    | `bool`      | `True`     | 是否启用 TorchAir 的视图（View）优化。                       |
| `enable_frozen_parameter` | `bool`      | `True`     | 推理时是否固定权重的内存地址，以减少图执行时的输入地址刷新时间。 |
| `use_cached_graph`        | `bool`      | `False`    | 是否使用缓存的计算图。                                       |
| `graph_batch_sizes`       | `list[int]` | `[]`       | 用于 TorchAir 图缓存的批处理大小列表。                       |
| `graph_batch_sizes_init`  | `bool`      | `False`    | 如果 `graph_batch_sizes` 为空，是否动态初始化图批处理大小。  |
| `enable_kv_nz`            | `bool`      | `False`    | 是否启用 KV Cache 的 NZ 布局。仅对使用 MLA 的模型（如 DeepSeek）生效。 |
| `enable_super_kernel`     | `bool`      | `False`    | 是否启用 Super Kernel 来融合 DeepSeek MoE 层中的操作。仅对使用动态 W8A8 量化的 MoE 模型生效。 |

### 2.4 Ascend 调度器配置 (`ascend_scheduler_config`)

该字典用于控制 vLLM Ascend V1 引擎的调度行为。

| **名称**                       | **类型**            | **默认值** | **描述**                                                     |
| ------------------------------ | ------------------- | ---------- | ------------------------------------------------------------ |
| `enabled`                      | `bool`              | `False`    | **是否启用** Ascend 调度器（V1 引擎）。                      |
| `enable_pd_transfer`           | `bool`              | `False`    | 是否启用 Prefill-Decode (P-D) 传输。启用后，仅当所有请求的 Prefill 完成后才开始 Decode。仅对离线推理生效。 |
| `decode_max_num_seqs`          | `int`               | `0`        | 在启用 P-D 传输时，是否更改 Decode 阶段的最大序列数 (`max_num_seqs`)。仅在 `enable_pd_transfer` 为 `True` 时生效。 |
| `max_long_partial_prefills`    | `Union[int, float]` | `inf`      | 可并发进行 Partial Prefill 的长提示词（超过 `long_prefill_token_threshold`）的最大数量。 |
| `long_prefill_token_threshold` | `Union[int, float]` | `inf`      | 定义长提示词的最小 Token 阈值。                              |

> **注意：** `ascend_scheduler_config` 还支持继承 vLLM 调度器中的其他配置项，例如 `enable_chunked_prefill: True`。

### 2.5 权重预取配置 (`weight_prefetch_config`)

该字典用于配置权重预取（Weight Prefetch）功能。

| **名称**         | **类型** | **默认值**                                                  | **描述**                                                   |
| ---------------- | -------- | ----------------------------------------------------------- | ---------------------------------------------------------- |
| `enabled`        | `bool`   | `False`                                                     | **是否启用** 权重预取功能。                                |
| `prefetch_ratio` | `dict`   | `{"attn": {"qkv": 1.0, "o": 1.0}, "moe": {"gate_up": 0.8}}` | 各个权重（如 Attention QKV, MoE Gate/Up 投影）的预取比例。 |

### 2.6 综合配置示例

以下是一个同时启用并配置了 TorchAirGraph、Ascend 调度器和权重预取的完整附加配置示例：


```JSON
{
    "torchair_graph_config": {
        "enabled": true,
        "use_cached_graph": true,
        "graph_batch_sizes": [1, 2, 4, 8],
        "graph_batch_sizes_init": false,
        "enable_kv_nz": false
    },
    "ascend_scheduler_config": {
        "enabled": true,
        "enable_chunked_prefill": true,
        "max_long_partial_prefills": 1,
        "long_prefill_token_threshold": 4096
    },
    "weight_prefetch_config": {
        "enabled": true,
        "prefetch_ratio": {
            "attn": {
                "qkv": 1.0,
                "o": 1.0
            },
            "moe": {
                "gate_up": 0.8
            }
        }
    },
    "multistream_overlap_shared_expert": true,
    "refresh": false
}
```
