# 大模型缓存系统 LMCache

 在当前 AI 生态系统中，大型语言模型（Large Language Model，LLM）推理已逐渐演变为核心基础设施。无论是在驱动代码智能助手（Copilot）、搜索引擎、文档理解工具，还是支撑企业级对话系统等场景中，绝大多数现实世界的 AI 应用都需要依赖运行在 GPU 集群上的高吞吐量推理引擎来完成模型调用任务。

然而，随着使用规模的持续扩大，尤其是在处理长上下文（long-context）请求时，LLM 推理面临两大核心性能瓶颈，日益凸显：

- 成本激增 —— 用户请求变得更加复杂与庞大，导致 GPU 资源消耗迅速攀升，从而引发推理成本成倍增长的问题；
- 延迟指标难以达标 —— 在保障用户体验的前提下，如何满足对“首个 Token 响应时间”（TTFT, Time to First Token）与“Token 间响应时间”（ITL, Inter-Token Latency）的严格服务等级目标（SLOs），已成为技术落地的关键挑战之一。



   要应对上述挑战，单纯依赖扩展 GPU 数量已难以为继，迫切需要引入更加智能、高效的显存与缓存管理策略，从系统底层提升整体推理效率。

   在这一背景下，LMCache 应运而生，作为一种新型缓存系统方案，旨在通过精准的 KV 缓存调度与跨请求共享机制，显著降低推理成本，同时优化响应延迟，从而推动大模型推理基础设施向更高性能、更低成本的方向迈进。

*—**01*** *—*

**什么是 LMCache ？**

   **众所周知，无论大型语言模型（LLMs）变得多么智能，在读取外部文本、视频等上下文信息时，依然面临推理速度慢、成本高昂的核心问题。LMCache 正是为了解决这一痛点而设计——基本思想是：每一段文本，模型只需读取一次。**

  在真实应用中，大量数据往往是被重复读取的。无论是热门书籍、历史对话记录，还是新闻报道等内容，都会在不同请求中多次出现。这正印证了“帕累托法则”中的经典理念：20% 的知识内容被使用了 80% 的时间。

   基于这一洞察，LMCache 提出了一个创新机制：将所有可复用文本的 KV 缓存（即 LLM 可直接使用的键值对表示）统一存储起来。这样，当后续请求中再次引用这些文本时，无需重新推理，只需直接重用 KV 缓存即可，无论这些内容出现在请求的前缀还是中间位置。该方案由芝加哥大学（University of Chicago）开发，目前已经引起了多个产业合作伙伴的高度关注。

在实际部署中，当 LMCache 与高性能推理引擎 vLLM 结合使用时，能够显著提升模型响应速度：“首个 Token 响应时间”（TTFT）可提升 3–10 倍，同时在多轮问答、RAG 检索增强生成等典型[大模型应用](https://cloud.tencent.com/developer/techpedia/2484?from_column=20065&from=20065)场景中，有效节省大量 GPU 计算资源，降低整体运行成本。

*—**02*** *—*

 **LMCache 具有哪些核心特性 ？**

   在实际的业务场景中，LMCache 在缓存系统的三个关键维度上实现了突破式提升，为大模型推理引擎提供了全新的底层加速范式：

   1、海量规模（Massive Scale）

   LMCache 支持存储远超 GPU 显存容量的大规模 KV 缓存数据，通过解耦“模型推理”与“上下文存储”的耦合瓶颈，使得大模型可以应对更长上下文、更多用户并发的挑战。这一能力极大地拓展了上下文重用的空间，为跨查询共享提供基础。

   2、极速加载（Blazing Speed）

   LMCache 采用基于 CUDA 加速算子与流水线数据传输机制 的高效加载方式，可将命中的 KV 缓存以极低延迟迅速加载至 GPU 显存中。相比传统的内存拷贝与 CPU-GPU 数据通路，该方式在多轮对话、RAG 等高频缓存场景中显著降低推理启动时延（TTFT）。

   3、插件式存储后端（Pluggable Storage）

   LMCache 提供灵活开放的存储接口，可无缝集成多种后端系统，包括 MooncakeStore、Infinistore、Redis、[分布式文件系统](https://cloud.tencent.com/product/chdfs?from_column=20065&from=20065)（DFS）等。这种插件式设计不仅增强了系统的可扩展性，也为企业部署提供更广泛的适配空间。

   借助上述三大能力，LMCache 进一步扩展了 vLLM 分页内存机制（paged memory design）的有效内存边界，使得推理引擎可以跨请求重用历史上下文缓存，不再受限于单次 session 的显存分配策略。

   最终，LMCache 实现了从“缓存是成本负担”到“缓存即性能优势”的转变，为大模型推理系统提供了一条兼顾性能、成本与可扩展性的路径。



## 1. LMCache 简介

TTFT 是指从请求发出到模型生成第一个 token 的时间。由于 Prefill 阶段需要把输入的上下文编码成 KV Cache，才能开始生成，在生成第一个 token 时需要大量的计算从而导致 TTFT 很高。

为了降低 TTFT，有一个思路就是将 Prefill 阶段计算出来的 KV Cache 缓存起来，下次遇到相同的上下文时，直接复用缓存的 KV Cache，就可以大幅降低 TTFT。

在模型推理的场景下，https://github.com/LMCache/LMCache 就是针对 KV Cache 缓存的一个开源项目，支持将 KV Cache 存储到内存、磁盘、Redis、GDS、Nixl 等多种存储后端。详情查看 https://docs.lmcache.ai/kv_cache/storage_backends/index.html 。

此外，lmcache 还提供了计算 KV Cache 大小的工具 https://lmcache.ai/kv_cache_calculator.html ，以 4k 中文估算，2k token 需要 106 MB 的 KV Cache，存储开销非常大。虽然 LMCache 有 LRU、FIFO、LFU、MRU 等缓存淘汰策略，但在生产环境中，通常还是需要配合大容量的存储后端，比如 Redis、3FS、大磁盘。



## 2. 缓存到内存

- 设置环境变量

```bash
# Specify LMCache V1
export LMCACHE_USE_EXPERIMENTAL=True
# 256 Tokens per KV Chunk
export LMCACHE_CHUNK_SIZE=256
# Enable CPU memory backend
export LMCACHE_LOCAL_CPU=True # default
# 50 GB of Pinned CPU memory
export LMCACHE_MAX_LOCAL_CPU_SIZE=50 # default 5.0
```

- 启动模型服务

```bash
export CUDA_VISIBLE_DEVICES=7
/opt/venv/bin/vllm serve \
    /data/models/Qwen2.5-7B-Instruct \
    --no-enable-prefix-caching \
    --max-model-len 16384 \
    --kv-transfer-config \
    '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'
```



# 加速LLM推理: 跳出推理引擎

**简要总结**：大语言模型正迅速成为企业AI中的主要工作负载。当越来越多的应用依赖实时生成，推理性能——速度、成本、可靠性——就成了头号瓶颈。如今，业界几乎把全部注意力都放在“加速推理引擎”上（vLLM、SGLang、TensorRT 等），却忽视了一片更大的优化疆域：引擎之上、跨引擎的整个系统层。

<img src="https://i0.wp.com/blog.lmcache.ai/wp-content/uploads/2025/11/Image_20251119091925_14_810.jpg?resize=1024%2C378&ssl=1" alt="A flowchart illustrating the relationship between Inference Engines and the Orchestration Layer, showcasing components like vLLM, SG, and various orchestration systems across shared GPU resources." style="zoom:50%;" />

## 现状: 大规模复制推理引擎

如今的大语言模型推理系统由两大核心组件构成：

- **推理引擎**： 针对单个模型实例优化推理性能。
- **编排层**： 借助 Kubernetes 等工具，复制这些引擎以实现水平扩展。

这种“以引擎为中心”的思维假定，绝大部分性能增益都源于引擎内部；编排层被视为一层“薄”且几乎无状态的外壳——只需启动更多副本即可提升吞吐量。

该模型驱动了以下开源系统：

- [vLLM 生产技术栈（主要由 LMCache 实验室维护）](https://github.com/vllm-project/production-stack)
- [RedHat 的 Ilm-d](https://github.com/llm-d/llm-d) 与 [KServe](https://kserve.github.io/website/0.15/blog/articles/2025-05-27-KServe-0.15-release/)
- [Nvidia 的 Al Dynamo](https://github.com/ai-dynamo/dynamo)
- [字节跳动的 AlBrix](https://github.com/vllm-project/aibrix)
- [Modular](https://github.com/modular/modular)
- [SGLang OME 及类似推理平台](https://github.com/sgl-project/ome)

这些方案在横向扩展方面表现良好，但当需要更深层次的系统级优化时，它们便会遇到瓶颈。

## 新前沿：超越引擎的 LLM 原生优化

真正的性能提升在于“跨引擎”的智能编排——通过共享状态、复用计算和全局优化。例如：

- 动态 prefill/decode 分离
- 跨会话的 KV cache共享（传输、压缩）
- 在线 KV cache更新（融合、编辑、翻译）
- 请求路径之外的休眠时间计算
- 查询迁移与跨 Agent 的流水线

这些优化需要“有状态的协同”和“智能调度”——原生 Kubernetes 方案很难直接提供。正因如此，在编排层进行创新，才能带来显著的性能收益。

## 问题不在 Kubernetes 本身——但它还不够

先说清楚：我们并不是在说 Kubernetes 没用。它是现代软件基础设施的核心组件。我们的意思是——仅靠 Kubernetes 来做 LLM 推理编排，会限制所能达到的上限。以下是现有基于K8s的开源系统不足的原因：

## 仅支持无状态的副本

Kubernetes 把所有 Pod 都当作无状态、同质的单元。
然而，LLM 工作负载依赖共享、有状态的组件，如 KV cache和中间工具调用状态。
LLM 时代的许多优化都要求态持久化并在请求和副本间共享——这正是 Kubernetes 原生所不支持的。

## 仅支持请求驱动执行

像缓存压缩、休眠时预取或后台融合这类优化，都需要在关键请求路径之外进行计算。
然而，基于 Kubernetes 的编排器是基于请求-响应模型构建的——没有简单的方法可以与主流推理工作同时调度或中断后台作业

## 不适合 LLM 特有的逻辑

特定于推理的调度（例如，优先考虑延迟敏感型作业而非后台作业）很难在现有的编排框架中实现。更糟的是，大多数系统用 Go 或 Rust 编写——这些语言并不适合张量密集型, 类Python 工作负载的快速原型设计。

## 部署与运维难度高

与 OpenAI、Claude、Fireworks 或 Together AI 这类 API 或专用端点方案相比，基于 Kubernetes 的推理栈部署和运维要复杂得多。即便拥有基础设施经验的团队，也随着更多优化和“补丁”被层层叠加到本就复杂的系统中而难以跟上。

**归根结底**：推理优化的前沿推进得越远，仅依赖 Kubernetes 的架构就越成为瓶颈

## 我们需要什么：LLM 原生的编排器

为支持下一代推理工作负载，我们需要一款专为 LLM 打造的编排器，它应：

- 支持面向张量的通信与状态共享（用于 KV 缓存复用、迁移与流水线）
- 允许可中断的后台任务（用于压缩、融合及非请求类计算）
- 在实时任务与后台任务之间调度，能感知延迟、资源占用和依赖关系
- 即使团队缺乏深厚基础设施经验，也能轻松部署与运维

这不是取代 Kubernetes，而是对其补充。Kubernetes 仍可负责容器生命周期、节点扩缩和集群管理；但推理工作负载本身需要理解 LLM 特有模式的编排逻辑

## 下一步

我们构建了一个 LLM 原生编排解决方案来满足这些需求，在不牺牲可用性的情况下支持强大的系统级优化。
