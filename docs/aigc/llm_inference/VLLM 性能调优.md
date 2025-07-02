# VLLM 设计文档

## 入口点（Entrypoints）

vLLM 提供了多种与系统交互的入口点。下图展示了这些入口点之间的关系。

<img src="https://docs.vllm.ai/en/stable/assets/design/arch_overview/entrypoints.excalidraw.png" alt="Entrypoints Diagram" style="zoom: 50%;" />

### LLM 类

`LLM` 类是进行离线推理的主要 Python 接口，即在不借助单独推理服务器的情况下与模型交互。

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
LLM` 类的代码位于：`vllm/entrypoints/llm.py
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



# Automatic Prefix Caching

## 简介

Automatic Prefix Caching 自动前缀缓存（简称APC）通过缓存已有查询的KV缓存，使得新查询若与已有查询共享相同前缀时，可直接复用对应的KV缓存，从而跳过共享部分的重复计算。

## Automatic Prefix Caching 实现细节

PagedAttention的核心思想是将每个请求的KV缓存划分为多个**KV块**。每个块包含固定数量token的注意力键值对。PagedAttention 算法允许这些块存储在非连续的物理内存中，从而通过按需分配内存来消除内存碎片。

我们基于以下关键观察实现KV缓存的自动化管理：每个KV块可以通过**块内token**和**该块之前的prefix token**唯一标识。

```shell
                    Block 1                  Block 2                  Block 3
         [A gentle breeze stirred] [the leaves as children] [laughed in the distance]
Block 1: |<--- block tokens ---->|
Block 2: |<------- prefix ------>| |<--- block tokens --->|
Block 3: |<------------------ prefix -------------------->| |<--- block tokens ---->|
```

在上述示例中：

- **第一个KV块**可通过块内token序列"A gentle breeze stirred"唯一标识
- **第三个KV块**则需要同时包含：
  - 块内token序列"laughed in the distance"
  - 前缀token序列"A gentle breeze stirred the leaves as children"

由此可建立严格的映射关系：

```
hash(prefix tokens + block tokens) <--> KV Block
```

## vLLM 的键值缓存（KV Cache）管理优化

通过引入这种映射机制，我们在 vLLM 的键值（KV）缓存管理中增加了一层间接性。以往，vLLM 中的每个序列都维护一个从逻辑 KV 块到物理块的映射。为了实现 KV 块的自动缓存，我们将逻辑 KV 块映射到其哈希值，并维护一个全局的哈希表，用于存储所有的物理块。通过这种方式，所有哈希值相同的 KV 块（例如，不同请求中共享的前缀块）可以映射到相同的物理块，从而共享内存空间。

该设计无需在 KV 块之间维护树形结构即可实现自动前缀缓存。具体而言，所有 KV 块都是彼此独立的，可以单独分配和释放，这使我们可以将 KV 缓存管理方式类比为操作系统中的普通缓存管理。

## 通用缓存策略（Generalized Caching Policy）

将所有 KV 块存储在哈希表中，使得 vLLM 可以缓存来自早期请求的 KV 块，从而节省内存并加速后续请求的计算。例如，如果一个新请求与之前的请求共享相同的系统提示词（system prompt），则可以直接复用共享提示词对应的 KV 缓存，无需重新计算。

然而，由于 KV 缓存空间有限，我们必须在缓存空间满时决定保留哪些 KV 块、淘汰哪些块。

使用哈希表管理 KV 缓存使我们可以实现灵活的缓存策略。例如，在当前 vLLM 中，我们实现了如下的淘汰策略：

1. 当没有可用空闲块时，我们优先淘汰引用计数为 0 的 KV 块（即当前无任何请求使用该块）。
2. 如果存在多个引用计数为 0 的块，则优先淘汰最近最少使用（LRU）的块。
3. 如果存在多个最近访问时间相同的块，则优先淘汰位于最长前缀末尾的块（即该块前面拥有最多块的情况）。

值得注意的是，该淘汰策略在应用于完整注意力（full attention）模型时，实际上等效于 RadixAttention 中的策略：优先淘汰引用计数为 0 且最近未被使用的前缀树叶节点。

## 哈希表机制的扩展能力

基于哈希的 KV 缓存管理机制为我们处理更复杂的在线服务场景提供了更大的灵活性，也支持实现比上述更复杂的淘汰策略，例如：

### 多 LoRA 模型服务（Multi-LoRA Serving）

当同时服务多个 LoRA 适配器请求时，我们可以在计算 KV 块哈希值时加入 LoRA ID，使每个请求对应的 KV 块能够正确区分并缓存。通过这种方式，我们可以统一管理所有适配器的 KV 块，简化系统实现，并提高全局缓存命中率与使用效率。

### 多模态模型（Multi-modal Models）

当用户输入包含非离散的模态（例如图像、音频等）时，我们可以采用不同的哈希方式来缓存不同模态的输入。例如，对于图像输入，可以使用感知哈希（perceptual hashing）方法，以便缓存相似的输入图像。

## 在vLLM中启用APC

在vLLM引擎中设置`enable_prefix_caching=True`即可启用APC。以下为示例代码：

```python
import time
from vllm import LLM, SamplingParams


# A prompt containing a large markdown table. The table is randomly generated by GPT-4.
LONG_PROMPT = "You are a helpful assistant in recognizes the content of tables in markdown format. Here is a table as follows.\n# Table\n" + """
| ID  | Name          | Age | Occupation | Country     | Email                  | Phone Number | Address                         |
| --- | ------------- | --- | ---------- | ----------- | ---------------------- | ------------ | ------------------------------- |
| 1   | John Doe      | 29  | Engineer   | USA         | john.doe@example.com   | 555-1234     | 123 Elm St, Springfield, IL     |
| 2   | Jane Smith    | 34  | Doctor     | Canada      | jane.smith@example.com | 555-5678     | 456 Oak St, Toronto, ON         |
| 3   | Alice Johnson | 27  | Teacher    | UK          | alice.j@example.com    | 555-8765     | 789 Pine St, London, UK         |
| 4   | Bob Brown     | 45  | Artist     | Australia   | bob.b@example.com      | 555-4321     | 321 Maple St, Sydney, NSW       |
| 5   | Carol White   | 31  | Scientist  | New Zealand | carol.w@example.com    | 555-6789     | 654 Birch St, Wellington, NZ    |
| 6   | Dave Green    | 28  | Lawyer     | Ireland     | dave.g@example.com     | 555-3456     | 987 Cedar St, Dublin, IE        |
| 7   | Emma Black    | 40  | Musician   | USA         | emma.b@example.com     | 555-1111     | 246 Ash St, New York, NY        |
| 8   | Frank Blue    | 37  | Chef       | Canada      | frank.b@example.com    | 555-2222     | 135 Spruce St, Vancouver, BC    |
| 9   | Grace Yellow  | 50  | Engineer   | UK          | grace.y@example.com    | 555-3333     | 864 Fir St, Manchester, UK      |
| 10  | Henry Violet  | 32  | Artist     | Australia   | henry.v@example.com    | 555-4444     | 753 Willow St, Melbourne, VIC   |
| 11  | Irene Orange  | 26  | Scientist  | New Zealand | irene.o@example.com    | 555-5555     | 912 Poplar St, Auckland, NZ     |
| 12  | Jack Indigo   | 38  | Teacher    | Ireland     | jack.i@example.com     | 555-6666     | 159 Elm St, Cork, IE            |
| 13  | Karen Red     | 41  | Lawyer     | USA         | karen.r@example.com    | 555-7777     | 357 Cedar St, Boston, MA        |
| 14  | Leo Brown     | 30  | Chef       | Canada      | leo.b@example.com      | 555-8888     | 246 Oak St, Calgary, AB         |
| 15  | Mia Green     | 33  | Musician   | UK          | mia.g@example.com      | 555-9999     | 975 Pine St, Edinburgh, UK      |
| 16  | Noah Yellow   | 29  | Doctor     | Australia   | noah.y@example.com     | 555-0000     | 864 Birch St, Brisbane, QLD     |
| 17  | Olivia Blue   | 35  | Engineer   | New Zealand | olivia.b@example.com   | 555-1212     | 753 Maple St, Hamilton, NZ      |
| 18  | Peter Black   | 42  | Artist     | Ireland     | peter.b@example.com    | 555-3434     | 912 Fir St, Limerick, IE        |
| 19  | Quinn White   | 28  | Scientist  | USA         | quinn.w@example.com    | 555-5656     | 159 Willow St, Seattle, WA      |
| 20  | Rachel Red    | 31  | Teacher    | Canada      | rachel.r@example.com   | 555-7878     | 357 Poplar St, Ottawa, ON       |
| 21  | Steve Green   | 44  | Lawyer     | UK          | steve.g@example.com    | 555-9090     | 753 Elm St, Birmingham, UK      |
| 22  | Tina Blue     | 36  | Musician   | Australia   | tina.b@example.com     | 555-1213     | 864 Cedar St, Perth, WA         |
| 23  | Umar Black    | 39  | Chef       | New Zealand | umar.b@example.com     | 555-3435     | 975 Spruce St, Christchurch, NZ |
| 24  | Victor Yellow | 43  | Engineer   | Ireland     | victor.y@example.com   | 555-5657     | 246 Willow St, Galway, IE       |
| 25  | Wendy Orange  | 27  | Artist     | USA         | wendy.o@example.com    | 555-7879     | 135 Elm St, Denver, CO          |
| 26  | Xavier Green  | 34  | Scientist  | Canada      | xavier.g@example.com   | 555-9091     | 357 Oak St, Montreal, QC        |
| 27  | Yara Red      | 41  | Teacher    | UK          | yara.r@example.com     | 555-1214     | 975 Pine St, Leeds, UK          |
| 28  | Zack Blue     | 30  | Lawyer     | Australia   | zack.b@example.com     | 555-3436     | 135 Birch St, Adelaide, SA      |
| 29  | Amy White     | 33  | Musician   | New Zealand | amy.w@example.com      | 555-5658     | 159 Maple St, Wellington, NZ    |
| 30  | Ben Black     | 38  | Chef       | Ireland     | ben.b@example.com      | 555-7870     | 246 Fir St, Waterford, IE       |
"""


def get_generation_time(llm, sampling_params, prompts):
    # time the generation
    start_time = time.time()
    output = llm.generate(prompts, sampling_params=sampling_params)
    end_time = time.time()
    # print the output and generation time
    print(f"Output: {output[0].outputs[0].text}")
    print(f"Generation time: {end_time - start_time} seconds.")


# set enable_prefix_caching=True to enable APC
llm = LLM(
    model='lmsys/longchat-13b-16k',
    enable_prefix_caching=True
)

sampling_params = SamplingParams(temperature=0, max_tokens=100)

# Querying the age of John Doe
get_generation_time(
    llm,
    sampling_params,
    LONG_PROMPT + "Question: what is the age of John Doe? Your answer: The age of John Doe is ",
)

# Querying the age of Zack Blue
# This query will be faster since vllm avoids computing the KV cache of LONG_PROMPT again.
get_generation_time(
    llm,
    sampling_params,
    LONG_PROMPT + "Question: what is the age of Zack Blue? Your answer: The age of Zack Blue is ",
)
```

## 典型应用场景
我们描述两种APC能显著提升性能的场景：

1. **长文档查询**

   用户对同一份长文档（如软件手册或年度报告）进行多次不同查询时，APC允许vLLM仅需处理一次长文档，后续所有请求均可通过复用其KV缓存来避免重复处理。这使得vLLM能以更高吞吐量和更低延迟服务后续请求。

2. **多轮对话**

   用户在同一会话中与应用进行多次交互时，APC允许vLLM跨所有后续对话轮次复用历史对话的处理结果，从而显著提升后续请求的吞吐量和降低延迟。


## 限制说明

APC通常不会降低vLLM的性能表现，但需注意：
- APC仅优化查询处理阶段（预填充阶段）耗时，不会减少新token生成阶段（解码阶段）耗时
- 当vLLM大部分时间用于生成长答案时，或新查询与现有查询无共享前缀时，APC不会带来性能提升



------

# vLLM V1 优化与性能调优指南

## 任务抢占（Preemption）

由于 Transformer 架构的自回归特性，在并发请求数量较多时，可能会出现 KV 缓存空间不足的情况。此时，vLLM 会抢占部分请求以释放 KV 缓存空间，并在资源充足后重新计算被抢占的请求。

当发生抢占时，你可能会看到如下警告信息：

```bash
WARNING 05-09 00:49:33 scheduler.py:1057 Sequence group 0 is preempted by PreemptionMode.RECOMPUTE mode because there is not enough KV cache space. This can affect the end-to-end performance. Increase gpu_memory_utilization or tensor_parallel_size to provide more KV cache memory. total_cumulative_preemption_cnt=1
```

虽然抢占机制提高了系统的鲁棒性，但重计算会增加端到端延迟。如果你频繁遇到抢占问题，可尝试以下优化方法：

- **增加 `gpu_memory_utilization`**
   vLLM 会根据该参数预分配 GPU KV 缓存空间。提高该值可以分配更多缓存空间。
- **降低 `max_num_seqs` 或 `max_num_batched_tokens`**
   减少每批次并发请求数或 token 总数，以降低缓存需求。
- **增加 `tensor_parallel_size`**
   将模型权重在多个 GPU 间分片，为每个 GPU 腾出更多缓存空间，但可能引入同步开销。
- **增加 `pipeline_parallel_size`**
   将模型层级在多个 GPU 间分布，从而减少每张卡上的权重开销，间接为 KV 缓存释放内存。但也可能导致延迟增加。

你可以通过 Prometheus 指标监控抢占次数，并通过设置 `disable_log_stats=False` 记录累计抢占计数。

在 vLLM V1 中，默认的抢占模式为 `RECOMPUTE`（而非 V0 中的 `SWAP`），因为在 V1 架构中重计算的开销更低。

------

## 分块预填充（Chunked Prefill）

分块预填充允许 vLLM 将较大的预填充请求划分为多个较小块，并与解码请求一起批处理。该机制有助于在计算密集型（预填充）与内存密集型（解码）操作间取得更好的平衡，从而提升吞吐量与延迟表现。

在 vLLM V1 中，分块预填充**始终默认启用**（不同于 V0 中根据模型条件决定是否启用）。

### 调度策略

启用该功能后，调度器将**优先处理解码请求**，即在处理任何预填充前，会先批量调度所有待解码请求。如果还有剩余的 `max_num_batched_tokens` 空间，则调度预填充请求。如果某个预填充请求超出当前限制，则会自动对其进行分块处理。

该策略带来两个优势：

1. **提升 ITL 与生成速度**：解码请求优先处理，降低了 token 间延迟。
2. **提高 GPU 利用率**：将预填充与解码请求同时调度至同一批次，有效利用计算资源。

### 分块调优建议

你可以通过调整 `max_num_batched_tokens` 参数来优化性能：

- **较小值（如 2048）**：更适合低延迟场景，减少预填充对解码性能的影响。
- **较大值（如 >8096）**：更适合吞吐优先场景，尤其在大显存 GPU 上运行小模型时。

```python
from vllm import LLM

# 设置 max_num_batched_tokens 以调优性能
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", max_num_batched_tokens=16384)
```

> 参考文献：
>
> - https://arxiv.org/pdf/2401.08671
> - https://arxiv.org/pdf/2308.16369

------

## 并行策略（Parallelism Strategies）

vLLM 支持多种可组合的并行策略，以在不同硬件配置中优化性能：

### 张量并行（Tensor Parallelism, TP）

将模型的参数在每一层内切分到多个 GPU 上，是单节点大模型推理中最常用的方式。

#### 适用场景：

- 模型过大，单张 GPU 无法容纳
- 希望减少每张卡上的内存压力，从而为 KV 缓存留出更多空间

```python
from vllm import LLM

# 使用 4 张 GPU 进行张量并行
llm = LLM(model="meta-llama/Llama-3.3-70B-Instruct", tensor_parallel_size=4)
```

------

### 流水线并行（Pipeline Parallelism, PP）

将模型的不同层级分配至多个 GPU，按序处理请求。

#### 适用场景：

- 已用尽张量并行方案，需进一步在更多 GPU 或跨节点间分布模型
- 模型较深但每层较小（窄深模型）

张量并行与流水线并行可以组合使用：

```python
from vllm import LLM

# 同时启用张量并行和流水线并行
llm = LLM(
    model="meta-llama/Llama-3.3-70B-Instruct",
    tensor_parallel_size=4,
    pipeline_parallel_size=2
)
```

------

### 专家并行（Expert Parallelism, EP）

专门用于稀疏专家模型（MoE），将不同专家网络分布到不同 GPU 上。

#### 适用场景：

- 特定于 MoE 架构（如 DeepSeekV3、Qwen3MoE、Llama-4）
- 需要在多 GPU 之间均衡专家计算负载

```python
llm = LLM(
    model="Qwen/Qwen3MoE",
    tensor_parallel_size=4,
    enable_expert_parallel=True  # 启用专家并行
)
```

------

### 数据并行（Data Parallelism, DP）

将整个模型复制到多个 GPU 集群上，并发处理不同批次的请求。

#### 适用场景：

- GPU 资源充足，可重复部署模型副本
- 目标是提升吞吐量而非模型容量
- 多用户场景下需要请求隔离

```python
llm = LLM(
    model="meta-llama/Llama-3.3-70B-Instruct",
    data_parallel_size=2
)
```

注意：MoE 层的并行粒度为 `tensor_parallel_size * data_parallel_size`。

------

## 内存优化建议

若遇到显存不足问题，可参考以下优化措施：

### 控制上下文长度与批处理大小

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    max_model_len=2048,  # 限制上下文长度
    max_num_seqs=4       # 限制批处理请求数
)
```

------

### 调整 CUDA 图编译配置

V1 中 CUDA 图编译使用的内存比 V0 多。你可以通过以下方式减少开销：

```python
from vllm import LLM
from vllm.config import CompilationConfig, CompilationLevel

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    compilation_config=CompilationConfig(
        level=CompilationLevel.PIECEWISE,
        cudagraph_capture_sizes=[1, 2, 4, 8]  # 更小的捕获批次
    )
)
```

或禁用 CUDA 图编译以节省内存：

```python
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    enforce_eager=True
)
```

------

### 多模态模型下的优化

限制每个请求中的图像或视频数量，以降低内存占用：

```python
from vllm import LLM

# 每个 prompt 最多包含 2 张图片
llm = LLM(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    limit_mm_per_prompt={"image": 2}
)
```
