# SGLang 官方超参数调优指南

对于 Sglang 离线批量推理而言，实现高吞吐量的关键在于获得较大的批处理大小。当服务器稳定满负荷运行时，请在日志中查找以下内容：

```shell
Decode batch. #running-req: 229, #token: 1320576, token usage: 1.00, cuda graph: True, gen throughput (token/s): 2498.69, #queue-req: 27,
```

关键字段解释

- `#running-req`：正在调度执⾏的请求数。

- `#token`：当前batch⽣成的总token数。

- `token usage`：0.0～1.0之间的⼩数，表示KV Cache的占⽤情况。1.0表示占满。

- `#queue-req`：当前正在排队的请求数。



## 调整请求提交速度以控制`#queue-req`

`#queue-req`表示队列中的请求数量，健康合理的 `#queue-req` 取值范围是 100 - 2000。如果⻓时间保持在0，代表客户端请求发送速率过低，可适当增加请求的数量。但是，请避免将此值设置得过大，因为这会增加服务器的调度开销。

## 获得高的`token usgae  ` (KV cache ) 利用率

`token usage` 表示服务器的 KV Cache 内存利用率。`token usage > 0.9`表示利用率良好，否则则过低。

- 如果频繁注意到 `token usage < 0.9 ` 和`#queue-req > 0`，表示调度器过于保守。可以将 `--schedule-conservativeness` 的值降低到 0.3 左右。

- 如果 `token usage` 数值非常高，并且频繁出现类似这样的警告 ：

  `KV cache pool is full. Retract requests. #retracted_reqs: 1, #new_token_ratio: 0.9998 -> 1.0000`，

​		可以将`--schedule-conservativeness` 数值增加到1.3 之类的值。如果您偶尔看到但不频繁（大约每分钟 1 次），则没问题。

## 调整`--mem-fraction-static`以增加 KV 缓存池容量

SGLang按如下方式分配内存：

```bash
总内存使用量 = 模型权重 + KV 缓存池 + CUDA 图缓冲区 + 激活值
```

```
mem_fraction_static = (模型权重 + KV 缓存池) / GPU 内存容量
```

`--mem-fraction-static` 参数决定分配给前两个组件的内存量，通常0.7～0.9是常⻅的配置范围。

为了支持更高的并发性，您应该尽可能提高 KV 缓存池容量，同时还要为激活和 CUDA 图缓冲区保留足够的内存。

SGLang 使用简单的启发式方法来设置默认值`--mem-fraction-static`，但您可以根据自己的使用场景对其进行优化。一般来说，为激活操作预留 5-8 GB 的内存就足够了。您可以通过在服务器准备就绪之前检查日志来确认这一点。查找类似这样的日志条目：

```bash
[2025-11-11 13:01:09 TP0] max_total_num_tokens=1522816, chunked_prefill_size=1024, max_prefill_tokens=2048, max_running_requests=4096, context_len=32768, available_gpu_mem=4.64 GB
```

检查该`available_gpu_mem`值。

- 如果大小在 5-8 GB 之间，则设置合适。
- 如果值过高（例如 10 - 20 GB），则增加`--mem-fraction-static`KV 缓存的内存分配。
- 如果内存太低，以后可能会出现内存不足（OOM）错误，因此要降低内存`--mem-fraction-static`。

另一种直接的方法是`--mem-fraction-static`每次增加 0.01，直到工作负载出现 OOM 错误为止。

## 避免内存溢出(OOM)错误

调整 `--chunked-prefill-size`，`--mem-fraction-static`#`--max-running-requests` 参数避免内存溢出错误

如果遇到内存不足（OOM）错误，您可以调整以下参数：

- 如果在预填充过程中发生内存溢出 (OOM)，请尝试将值减少`--chunked-prefill-size`到`4096`或`2048`。这可以节省内存，但会降低长提示符的预填充速度。
- 如果在解码过程中发生内存溢出 (OOM)，请尝试降低内存使用量`--max-running-requests`。
- 您还可以将其降低`--mem-fraction-static`到更小的值，例如 0.8 或 0.7。这可以减少 KV 缓存池的内存使用量，并有助于防止预填充和解码期间出现 OOM 错误。但是，这会限制最大并发数并降低峰值吞吐量。

## 调整 `--cuda-graph-max-bs`

默认情况下，CUDA 图仅在小批量大小（例如小于 160 或 256）下启用。但是，对于某些模型，尤其是在张量并行度较大的情况下，CUDA 图在批量大小高达 512 或 768 时仍然有用。因此，增加`--cuda-graph-max-bs` 批量大小可能是有益的。请注意，CUDA 图会消耗更多内存，因此您可能需要同时减少`--mem-fraction-static` 。

## 调整`--dp-size`和`--tp-size`

数据并行更有利于提高吞吐量。当GPU内存充足时，应始终优先选择数据并行以提高吞吐量。为了获得更好的数据并行性，建议参考[sglang路由，](https://docs.sglang.ai/advanced_features/router.html)而不是使用`dp_size`参数。

## 尝试其他选项

- `torch.compile` 加速小批量小型模型的运行。您可以使用以下命令启用此功能`--enable-torch-compile`。
- 尝试其他量化方式（例如使用 FP8 量化）`--quantization fp8`
- 尝试其他并行策略（例如[专家并行](https://lmsys.org/blog/2025-05-05-large-scale-ep/)）或 DeepSeek 模型的 DP 注意力机制（使用）。`--enable-dp-attention --dp-size 8`
- 如果工作负载有很多共享前缀，请尝试使用`--schedule-policy lpm` 。这里，`longest prefix match` 代表最长前缀匹配。它会重新排序请求以增加缓存命中率，但会增加调度开销。
