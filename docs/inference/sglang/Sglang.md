SGLang 是一个用于大型语言模型和视觉语言模型的快速服务框架。它通过共同设计后端运行时和前端语言，使你与模型的交互更快、更可控。其核心功能包括：

- **快速后端运行时**: 提供高效的服务，包括 RadixAttention 用于前缀缓存、跳跃式约束解码、连续批处理、令牌注意力（分页注意力）、张量并行、FlashInfer 内核、分块预填充和量化（INT4/FP8/AWQ/GPTQ）。
- **灵活的前端语言**: 提供直观的界面用于编程 LLM 应用程序，包括链式生成调用、高级提示、控制流、多模态输入、并行性和外部交互。

vLLM主要考虑简单的单轮对话形式与LLM进行交互，输入prompt，Prefill+Decode计算后输出。**随着大模型应用发展深入，LLM的使用方式正在发生深刻的变化。**比如，LLM参与multi-round planning、reasoning和与外部环境交互等复杂场景，需要LLM通过工具使用、多模态输入以及各种prompting techniques，比如self-consistency，skeleton-of-thought，and tree-of-thought等完成。这些过程都不是简单的单轮对话形式，通常涉及一个prompt输出多个结果，或者生成内容包含一些限制，比如json格式或者一些关键词。

这些模式的涌现标志着我们与LLMs交互方式的转变，从简单的聊天转向更复杂的程序化使用形式，这意味着使用类似编程语言的方式来控制LLMs的生成过程，称为LM Programs。LM Programs有两个共同特性：（1）LM Program通常包含多个LLM调用，这些调用之间穿插着控制流。这是为了完成复杂任务并提高整体质量所必需的。（2）LM Program接收结构化输入并产生结构化输出。这是为了实现LM Program的组合，并将其集成到现有的软件系统中。

最近广受关注的工作SGLang正是瞄准LLM Programs设计的。SGLang这个工作去年就发布了，当时其实就引起了很多关注，其RadixAttention共享KVCache Prefix的优化，也被最近的各种新型推理引擎所采用，比如MoonCake，MemServe等之中。最近SGLang的论文升级了一个版本，也更新了性能数据，效果直逼TRT-LLM引起了不少轰动。

SGLang采用了编译器方式的设计。当输入和输出是多对多的，就有很多Lazy方式来优化调度的空间，这就很自然的映射到编译器设计，可以分frontend和backend两部分。

![img](https://pic4.zhimg.com/v2-3592cbae899d45f449577807ed914b8f_1440w.jpg)

前端定义一种DSL，嵌入在Python中。下图展示了一个使用分支-解决-合并提示方法评估关于图像的论文的LLM Program。函数multi_dimensional_judge接受三个参数：s、path和essay。s管理提示状态，path是图像文件路径，essay是论文文本。可以使用+=操作符将新字符串和SGLang原语附加到状态s中以供执行。首先，函数将图像和论文添加到提示中。然后，它使用select检查论文是否与图像相关，并将结果存储在s["related"]中。如果相关，提示会分成三个副本进行不同维度的并行评估，使用gen将结果存储在f["judgment"]中。接着，它合并判断结果，生成总结，并评分ABCD。最后，它按照正则表达式约束regex定义的模式，以JSON格式返回结果。

SGLang后端执行时极大地简化了这一程序，如果使用类似OpenAI API的接口编写等效程序需要多出2.1倍的代码。

SGLang的后端Runtime有三个核心创新优化点，我下面分别介绍：

### 1. Efficient KV Cache Reuse with RadixAttention

上图Figure 2钟，SGLang程序可以通过“fork”原语链接多个生成调用并创建并行副本。此外，不同的程序实例通常共享一些公共部分（例如，系统提示）。这些情况在执行过程中创建了许多共享提示前缀，从而提供了许多重用KV缓存的机会。下图Figure 9所示，展示了各种KVCache Prefix共享的场景。

![img](https://pica.zhimg.com/v2-05961f2a5d95d1d5e7fc5a2996227fb2_1440w.jpg)

SGLang V1版本论文就提出了RadixAttention，这是一种在运行时自动和系统化重用KVCache的新技术。与现有系统在生成请求完成后丢弃KV缓存不同，我们的系统在RadixTree中保留prompt和生成结果的KVCache，实现高效的前缀搜索、重用、插入和驱逐。SGLang用LRU驱逐策略和缓存感知调度策略，以提高缓存命中率。

Mooncake也有相似的KVCache Prefix Sharing优化，不过场景略有差异，mooncake是在不同用户请求间很多共享前缀，SGLang还是在一个Program内。大家可以参考MoonCake中的Prefill Pool设计，RadixAttention和Hash设计有千丝万缕联系。我猜测，之前大家没想到请求间Prefix共享机会那么大，实际上对于RAG+LLM方式使用，请求间前缀相同概率挺大的，SGLang提出的RadixAttention很快变成了非常通用的设计，不止限于LLM Program中。

### 2. Efficient Constrained Decoding with Compressed Finite State Machine

在LM Programs中，用户通常希望将模型的输出限制为遵循特定格式，如JSON模式。这可以提高可控性和鲁棒性，并使输出更易于解析。SGLang通过正则表达式提供了一个regex参数来强制执行这些约束，这在许多实际场景中已经足够表达。现有系统通过将正则表达式转换为有限状态机（FSM）来支持这一点。在解码过程中，它们维护当前的FSM状态，从下一个状态检索允许的token，并将无效token的概率设置为零，逐个token解码。

Constrained Decoding我去年也有关注，微软的[Guidance](https://link.zhihu.com/?target=https%3A//github.com/guidance-ai/guidance)算是比较早期工作，SGLang也引用了。不过SGLang做了一些进一步的优化。

逐个token的方法在有机会一次性解码多个token时效率低下。例如，前面Figure 2中的常量序列{"summary": "在图4（c）所示的正常解码过程中跨越多个token，需要多个解码阶段，尽管在解码时只有一个有效的下一个token。因此，整个序列可以在一个步骤（即前向传递）中解码。然而，现有系统只能一次解码一个token，因为现有系统中FSM与模型运行器之间缺乏集成，无法进行多token处理，导致解码速度慢。

SGLang通过创建一个带有压缩FSM的快速约束解码运行时来克服这一限制。该运行时分析FSM并将FSM中相邻的单一转换边压缩为单一边，如图Figure（b）所示，使其能够识别何时可以一起解码多个token。在Figure 4（d）中，压缩转换边上的多个token可以在一次前向传递中解码，这大大加速了解码过程。它也是通用的，适用于所有正则表达式。

![img](https://pica.zhimg.com/v2-84c2c532eb1508e5e438aea131a2ef2a_1440w.jpg)

### 3. Efficient Endpoint Calling with API Speculative Execution

上述优化RadixAttention和Constrained Decoding还是针对模型是白盒情况。如果调用的模型是OpenAI这种黑盒API，SGLang通过使用推测执行来加速多调用SGLang程序的执行并降低API成本。

例如，一个程序可能要求模型通过多调用模式生成一个角色的描述：s += context + "name:" + gen("name", stop="\n") + "job:" + gen("job", stop="\n")。简单来说，这两个gen原语对应于两次API调用，这意味着用户需要为上下文支付两次输入令牌费用。在SGLang中，我们可以在第一次调用时启用推测执行（Speculative Execution），并让它忽略停止条件继续生成几个额外的令牌。解释器保留这些额外的生成输出，并与后面的原语进行匹配和重用。在某些情况下，通过提示工程，模型可以高准确度地匹配模板，从而节省我们一次API调用的延迟和输入成本。

文章没细讲遇到什么样的程序描述会开启Speculative Execution，因为如果推测失败，反而多消耗了token。我觉得这一章节抛砖引玉，强调了SGLang不只是推理引擎，还可以做作为推理引擎的上层调用框架。有点类似llvm和机器码执行器之间的关系。

### 令人惊艳的SGLang性能

使用RadixAttention和Constrained Decoding可以减少LLM Program的计算量，这些优化也是和vLLM的PA、Continous Batching兼容的。如果你对LLM的用法可以使用SGLang定义成LLM Program，在业务中是可以显著获得收益的。

不过如果还是展示SGLang V1论文的场景格局就小了。**SLGang V2在不用RadixAttention和Constrained Decoding优化前提下，相比vLLM有明显加速，而且性能接近TRT-LLM**。我这里贴了博客中的H100的性能，有些case甚至远超TRT-LLM。SLGang团队跟我说原因在于软件调度写得好，是实打实的更好的工程实现的结果。这个结果确实非常惊艳的，我没有实测，不过听说NVIDIA是可以复现这个结果的。这也说明现有的推理引擎vLLM有很大的重构提升空间。

InfoQ：SGLang 开源推理引擎受到不少一线公司的采用。你觉得它最核心的技术优势是什么？相比其他开源方案，有哪些关键差异？

**尹良升：** 我认为 SGLang 最核心的优势在于**高性能的实现和易于二次开发的代码**。从 RadixAttention、高效的架构设计、Overlap Scheduling，到成功复现并集成了像 PD 分离、大规模 EP 等前沿技术，SGLang 实现了对不同主流模型的 SOTA 部署支持。这是我们区别于其他方案的关键。

InfoQ：你的演讲会介绍 PD 分离、推测解码、KV 缓存落盘等关键技术，这些优化在实际部署中解决了哪些痛点？

**尹良升：**

- **PD 分离**：它解决了在 Prefill 和 Decode 混合部署时，Decode 经常被 Prefill 打断导致的延迟波动大、P99 尾延迟高的问题。分离部署后，Decode 的延迟变得均匀且稳定。同时，这种分离允许 Prefill 和 Decode 采用不同的部署策略和并行方式（比如不同的并行度），从而能更高效地利用资源。
- **推测解码**：这项技术的核心目标是降低 Decode 延迟。它通过利用模型隐藏层信息和小模型辅助，经过验证后一次预测多个 Token（相当于“一次解码，多步输出”），显著提升 Decode 速度，达到事半功倍的效果。
- **KV 缓存落盘**：在多轮对话等需要复用之前计算出的 KV cache 的场景下，GPU 显存容量有限，难以存储所有用户的历史记录。KV 缓存落盘技术将不立即需要的、以往计算的 KV cache 存储在内存或硬盘等大容量存储设备中。当后续对话轮次需要复用这些历史上下文时，可以直接加载缓存的 KV 值，避免了重复进行耗时的 Prefill 计算，从而减少计算量并有效降低响应延迟。

InfoQ：你的演讲会介绍 PD 分离、推测解码、KV 缓存落盘等关键技术，这些优化在实际部署中解决了哪些痛点？

**尹良升：** 平衡性能、资源利用率和成本需要根据下游任务进行 Trade-off：在离线批处理（Offline Batch）等对延迟不敏感的场景下，目标是最大化 GPU 显存利用率和批处理大小以提升吞吐、摊薄成本；而在线上推理等对延迟敏感的场景，则倾向于投入更多资源处理较少并发请求（如降低并行 Batch Size），优先保障每个请求的 Token 生成速度。

InfoQ：并行部署技术（如专家并行、注意力并行）越来越复杂，兼容性是一个现实挑战。SGLang 在支持多种并行方案上有哪些经验或踩坑故事？

**尹良升：** 在实现多种并行方案（比如专家并行）和分离式部署架构的过程中，我们**踩的坑不仅仅在算法实现的初期，很多工程上的挑战更大**。比如实现 DeepEP 时，它的通讯和 Prefill/Decode (PD) 之间传输 KV 缓存的时间需要重叠，这就容易导致网卡资源被同时争抢、CPU 负载过大、Python GIL 锁得不到及时释放等问题。

InfoQ：结合你的体验，能否分享下你认为 SGLang 社区是如何推动技术演进和应用场景落地之间的良性循环？

**尹良升：**SGLang 的**核心是开源**，让人人都能参与开发和贡献。我们广泛进行技术分享，旨在增强社区认同感并吸引新成员。社区的技术进步直接加速了应用落地，而我们超过 100k 显卡规模的大规模工业部署经验及来自社区的真实反馈，又持续指引着技术演进的方向，确保我们走在正确的道路上。
