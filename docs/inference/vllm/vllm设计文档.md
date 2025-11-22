# VLLM 设计文档

## 入口点（Entrypoints）

vLLM 提供了多种与系统交互的入口点。下图展示了这些入口点之间的关系。

<img src="https://docs.vllm.ai/en/stable/assets/design/arch_overview/entrypoints.excalidraw.png" alt="Entrypoints Diagram" style="zoom: 50%;" />

### LLM 类

`LLM` 类是进行离线推理的主要 Python 接口，即在不借助推理服务器的情况下与模型交互。

以下是一个 `LLM` 类的使用示例：

```python
from vllm import LLM, SamplingParams

# 定义输入提示语列表
prompts = [
    "Hello, my name is",
    "The capital of France is",
    "The largest ocean is",
]

# 设置采样参数
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# 使用 OPT-125M 模型初始化 LLM 引擎
llm = LLM(model="facebook/opt-125m")

# 对提示语进行推理生成
outputs = llm.generate(prompts, sampling_params)

# 输出生成的文本
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

更多 API 细节可参考 [Offline Inference](https://chatgpt.com/c/682eda70-eb4c-800a-ba30-aa064e6533d5#offline-inference-api) 章节的 API 文档。

```
LLM 类的代码位于：vllm/entrypoints/llm.py
```

### OpenAI 兼容的 API 服务端

vLLM 的第二个主要接口是其 OpenAI 兼容的 API 服务端，可通过以下命令启动：

```bash
vllm serve <model>
```

vllm CLI 的代码位于：`vllm/entrypoints/cli/main.py`

有时也可以直接使用 API 服务端入口，而非通过 CLI 命令。例如：

```bash
python -m vllm.entrypoints.openai.api_server --model <model>
```

该部分的代码位于：`vllm/entrypoints/openai/api_server.py`

更多信息详见《OpenAI-Compatible Server》文档。

## LLM 引擎

`LLMEngine` 和 `AsyncLLMEngine` 是 vLLM 系统的核心组成部分，分别负责模型推理与异步请求处理。

<img src="https://docs.vllm.ai/en/stable/assets/design/arch_overview/llm_engine.excalidraw.png" alt="LLMEngine Diagram" style="zoom:50%;" />

### LLMEngine

`LLMEngine` 是 vLLM 引擎的核心组件，负责接收客户端请求并生成模型输出，涵盖输入处理、模型执行（可分布式）、调度和输出处理。

- **输入处理**：使用指定的分词器对输入文本进行分词。
- **调度机制**：决定每一步应处理哪些请求。
- **模型执行**：负责语言模型的运行，可跨多个 GPU 分布式执行。
- **输出处理**：将模型生成的 token ID 解码为可读文本。

代码位置：`vllm/engine/llm_engine.py`

### AsyncLLMEngine

`AsyncLLMEngine` 是 `LLMEngine` 的异步包装类，使用 `asyncio` 构建后台循环，持续处理接入请求。适用于在线推理服务，可处理并发请求并支持输出流式返回。

OpenAI 兼容的 API 服务端即基于 `AsyncLLMEngine` 实现，此外还有一个简化的示例 API 服务端，位于 `vllm/entrypoints/api_server.py`。

代码位置：`vllm/engine/async_llm_engine.py`

## Worker（工作进程）

Worker 是实际运行模型推理的进程。vLLM 遵循“每个进程对应一个加速设备（如 GPU）”的常见模式。例如，若使用张量并行度为 2、流水并行度为 2，则总共有 4 个 worker。worker 通过 `rank` 与 `local_rank` 标识，前者用于全局调度，后者用于设备分配及访问本地资源（如文件系统和共享内存）。

## 模型运行器（Model Runner）

每个 worker 拥有一个模型运行器对象，负责加载并运行模型。大部分模型执行逻辑位于此处，包括输入张量准备和 CUDA 图捕获等操作。

## 模型（Model）

每个模型运行器包含一个模型对象，即实际的 `torch.nn.Module` 实例。详见 `huggingface_integration`，说明了不同配置如何影响最终模型类的构建。

## 类层次结构（Class Hierarchy）

下图展示了 vLLM 的类层次结构：

<img src="https://docs.vllm.ai/en/stable/assets/design/hierarchy.png" alt="query" style="zoom: 25%;" />

vLLM 的类层次结构背后有以下设计考量：

### 1. 可扩展性（Extensibility）

类层次结构中的所有类都接受一个configuration object ，该对象包含了所有必要的信息。其中，`VllmConfig` 类是整个系统中传递的主要configuration object 。由于 vLLM 的类层次结构较为复杂且层级较深，各个类需要访问其所关心的配置项。

通过将所有配置封装在一个对象中，我们可以轻松地在各类之间传递这个configuration object ，并在需要时访问相关配置。例如，当我们希望新增一个只涉及模型运行器的功能（在 LLM 推理快速发展的背景下这是非常常见的情况），我们只需在 `VllmConfig` 中添加一个新的配置项。由于整个configuration object 是完整传递的，模型运行器可以直接访问这一新增配置项。

这种设计避免了在引擎、Worker 或模型类的构造函数中引入额外的参数，无需对它们的构造函数进行修改即可支持新特性，从而大大提升了系统的可扩展性。

### 2. 一致性（Uniformity）

模型运行器需要一个统一的接口来创建并初始化模型。vLLM 支持超过 50 种流行的开源模型，每种模型都有其独特的初始化逻辑。如果各模型的构造函数签名不一致，模型运行器就无法在不借助复杂且易出错的代码检查机制的情况下正确调用构造函数。

通过统一模型类的构造函数签名，模型运行器无需了解具体的模型类型即可轻松地创建和初始化模型。这种方式同样适用于模型组合的场景。例如，视觉-语言模型通常由一个视觉模型和一个语言模型组成。统一的构造函数接口使我们能够方便地分别创建视觉模型与语言模型，并将它们组合成一个完整的视觉-语言模型。

构造函数签名已更新为关键字参数形式：

```python
def __init__(self, *, vllm_config: VllmConfig, prefix: str = "")
```

这样可以避免误传旧参数。若使用旧模型类，可通过适配代码进行兼容：

```python
from vllm.config import VllmConfig

class MyOldModel(nn.Module):
    def __init__(
        self,
        config,
        cache_config=None,
        quant_config=None,
        lora_config=None,
        prefix="",
    ):
        ...

class MyNewModel(MyOldModel):
    def __init__(self, *, vllm_config: VllmConfig, prefix=""):
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        super().__init__(config, cache_config, quant_config, lora_config, prefix)

if __version__ >= "0.6.4":
    MyModel = MyNewModel
else:
    MyModel = MyOldModel
```

### 3. 初始化时的切片与量化支持

某些特性（如张量并行与量化）需修改模型权重。相比模型初始化后再切片/量化，vLLM 选择在初始化阶段直接处理，以降低内存开销。

例如运行一个 405B 参数（约 810GB）的模型在 16 块 80GB H100 上，理想状态是每块 GPU 仅加载约 50GB 权重。若初始化后再切片，每块 GPU 都需加载完整模型，内存消耗极大；而初始化时分片可避免此问题。

同时构造函数中引入 `prefix` 参数，用于支持非统一量化配置。例如顶层模型的 prefix 为 `""`，而子模型如视觉或语言模块的 prefix 分别为 `"vision"` 或 `"language"`，便于根据权重文件结构差异化初始化。

## 额外说明

这种架构的一个挑战在于：每个组件都依赖完整configuration object ，导致单元测试难以独立进行。vLLM 通过默认初始化函数解决此问题，该函数可生成字段全为 `None` 的默认configuration object ，便于只设置关注的字段，从而进行独立测试。

需要注意的是，vLLM 的许多测试为端到端测试，覆盖整个系统流程，因此该限制影响较小。

## 总结

`VllmConfig` 是 vLLM 引擎层级的全局状态对象，在系统各个组件间共享。通过统一的配置机制、构造函数接口以及初始化策略，vLLM 实现了良好的可扩展性、一致性和可组合性，为支持大规模语言模型推理提供了坚实基础。
