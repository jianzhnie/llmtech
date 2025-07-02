

# 安装指南

This document describes how to install vllm-ascend manually.

## 系统要求

- OS: Linux

- Python: 3.9 or higher

- A hardware with Ascend NPU. It’s usually the Atlas 800 A2 series.

- Software:

  | Software  | Supported version    | Note                                   |
  | --------- | -------------------- | -------------------------------------- |
  | CANN      | >= 8.1.RC1           | Required for vllm-ascend and torch-npu |
  | torch-npu | >= 2.5.1.dev20250320 | Required for vllm-ascend               |
  | torch     | >= 2.5.1             | Required for torch-npu and vllm        |

您可通过两种方式安装：

- **使用 pip 安装** ：首先手动或通过 CANN 镜像准备环境，然后使用 pip 安装 `vllm-ascend`
- **使用 docker 安装** ：直接使用预构建的 `vllm-ascend` docker 镜像

## 配置新环境

安装前需确保固件/驱动和 CANN 已正确安装，详情参见[链接](https://ascend.github.io/docs/sources/ascend/quick_install.html)

### 配置硬件环境

要验证昇腾 NPU 固件和驱动是否正确安装，请运行：

```
npu-smi info
```

更多详情请参阅[昇腾环境搭建指南](https://ascend.github.io/docs/sources/ascend/quick_install.html)

## 配置 vllm 与 vllm-ascend

### 使用  pip 安装

首先安装系统依赖项：

```shell
yum update  -y
yum install -y gcc gcc-c++ cmake make numactl-devel wget git
```

**[可选]** 若在 **x86** 架构机器上操作，请配置 `pip` 的额外索引源以获取 CPU 版 torch：

```shell
pip config set global.extra-index-url https://download.pytorch.org/whl/cpu/
```

### vllm==0.8.5

随后可从**预构建 wheel 包**安装 `vllm` 和 `vllm-ascend`：

```bash
# Install vllm-project/vllm from pypi
pip install vllm==0.8.5.post1

# Install vllm-project/vllm-ascend from pypi.
pip install vllm-ascend==0.8.5rc1
```

从源码安装

```shell
# Install vLLM
git clone --depth 1 --branch v0.8.5.post1 https://github.com/vllm-project/vllm
cd vllm
VLLM_TARGET_DEVICE=empty pip install -v -e .
cd ..

# Install vLLM Ascend
git clone  --depth 1 --branch v0.8.5rc1 https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
pip install -v -e .
cd ..
```

### vllm==0.7.3

从**预构建 wheel 包**安装 `vllm` 和 `vllm-ascend`：

```
# Install vllm-project/vllm from pypi
pip install vllm==0.7.3

# Install vllm-project/vllm-ascend from pypi.
pip install vllm-ascend==0.7.3.post1
```

源码编译安装 vllm 和 vllm-ascend.

```shell
# vllm
git clone -b v0.7.3 --depth 1 https://github.com/vllm-project/vllm.git
cd vllm
pip install -r requirements-build.txt

VLLM_TARGET_DEVICE=empty pip install -e . -i  https://mirrors.aliyun.com/pypi/simple/
```

```shell
# vllm-ascend
git clone  --depth 1 --branch v0.7.3.post1 https://github.com/vllm-project/vllm-ascend.git

cd vllm-ascend
export COMPILE_CUSTOM_KERNELS=1

pip install -v . -i  https://mirrors.aliyun.com/pypi/simple/
```



>
>
>若基于 v0.7.3-dev 版本构建并使用休眠模式功能，需手动设置 `COMPILE_CUSTOM_KERNELS=1`。编译自定义算子需 gcc/g++ 8 以上版本及 C++17+标准。使用 `pip install -e .` 时若出现 torch-npu 版本冲突，请通过 `pip install --no-build-isolation -e .` 在系统环境中构建。若编译过程遇到其他问题，可能是使用了非预期编译器，建议在编译前通过环境变量 `CXX_COMPILER` 和 `C_COMPILER` 指定 g++与 gcc 路径。



###  验证安装

Create and run a simple inference test. The `example.py` can be like:

```python
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
# Create an LLM.
llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct")

# Generate texts from the prompts.
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

Then run:

```shell
# export VLLM_USE_MODELSCOPE=true to speed up download if huggingface is not reachable.
python example.py
```

The output will be like:

```shell
INFO 02-18 08:49:58 __init__.py:28] Available plugins for group vllm.platform_plugins:
INFO 02-18 08:49:58 __init__.py:30] name=ascend, value=vllm_ascend:register
INFO 02-18 08:49:58 __init__.py:32] all available plugins for group vllm.platform_plugins will be loaded.
INFO 02-18 08:49:58 __init__.py:34] set environment variable VLLM_PLUGINS to control which plugins to load.
INFO 02-18 08:49:58 __init__.py:42] plugin ascend loaded.
INFO 02-18 08:49:58 __init__.py:174] Platform plugin ascend is activated
INFO 02-18 08:50:12 config.py:526] This model supports multiple tasks: {'embed', 'classify', 'generate', 'score', 'reward'}. Defaulting to 'generate'.
INFO 02-18 08:50:12 llm_engine.py:232] Initializing a V0 LLM engine (v0.7.1) with config: model='./Qwen2.5-0.5B-Instruct', speculative_config=None, tokenizer='./Qwen2.5-0.5B-Instruct', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=32768, download_dir=None, load_format=auto, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto,  device_config=npu, decoding_config=DecodingConfig(guided_decoding_backend='xgrammar'), observability_config=ObservabilityConfig(otlp_traces_endpoint=None, collect_model_forward_time=False, collect_model_execute_time=False), seed=0, served_model_name=./Qwen2.5-0.5B-Instruct, num_scheduler_steps=1, multi_step_stream_outputs=True, enable_prefix_caching=False, chunked_prefill_enabled=False, use_async_output_proc=True, disable_mm_preprocessor_cache=False, mm_processor_kwargs=None, pooler_config=None, compilation_config={"splitting_ops":[],"compile_sizes":[],"cudagraph_capture_sizes":[256,248,240,232,224,216,208,200,192,184,176,168,160,152,144,136,128,120,112,104,96,88,80,72,64,56,48,40,32,24,16,8,4,2,1],"max_capture_size":256}, use_cached_outputs=False,
Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  5.86it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  5.85it/s]
INFO 02-18 08:50:24 executor_base.py:108] # CPU blocks: 35064, # CPU blocks: 2730
INFO 02-18 08:50:24 executor_base.py:113] Maximum concurrency for 32768 tokens per request: 136.97x
INFO 02-18 08:50:25 llm_engine.py:429] init engine (profile, create kv cache, warmup model) took 3.87 seconds
Processed prompts: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00,  8.46it/s, est. speed input: 46.55 toks/s, output: 135.41 toks/s]
Prompt: 'Hello, my name is', Generated text: " Shinji, a teenage boy from New York City. I'm a computer science"
Prompt: 'The president of the United States is', Generated text: ' a very important person. When he or she is elected, many people think that'
Prompt: 'The capital of France is', Generated text: ' Paris. The oldest part of the city is Saint-Germain-des-Pr'
Prompt: 'The future of AI is', Generated text: ' not bright\n\nThere is no doubt that the evolution of AI will have a huge'
```
