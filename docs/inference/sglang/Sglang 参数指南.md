# SGLang 服务器参数配置

## 基础配置参数详解

### 模型与路径配置

```
--model-path /mnt/models/DeepSeek-R1  # 模型文件路径
--tokenizer-path /mnt/models/tokenizer  # 分词器路径
--load-format gguf  # 模型加载格式（GGUF/PyTorch）
```

- **关键点**：路径需具备可读权限，模型与分词器版本需严格匹配，避免因版本冲突导致初始化失败。

### 设备与并行策略

```
--device cuda  # 设备类型（GPU/CPU）
--tp 4         # 张量并行度
--dp-size 2    # 数据并行度
--ep-size 1    # 专家并行度（MoE模型专用）
```

- **配置建议**：8卡服务器推荐`--tp 4 --dp-size 2`，显存不足时优先降低`--tp`值。

## 关键性能参数优化

### 内存与显存管理

```
--mem-fraction-static 0.85  # 静态显存分配比例
--kv-cache-dtype auto       # 键值缓存数据类型（fp16/bf16/auto）
--max-total-tokens 128000   # 最大上下文总token数
```

- 优化策略：显存不足时，按优先级调整参数：
  1. 降低`--max-total-tokens`至102400
  2. 启用量化`--quantization fp8`
  3. 减少`--mem-fraction-static`至0.8

### 请求处理控制

```
--max-running-requests 16      # 并发请求上限
--max-prefill-tokens 8192      # 首轮生成最大token数
--chunked-prefill-size 2048    # 分块预填充大小
```

- **高并发场景**：增加`--max-running-requests`至32，同时监控GPU利用率（目标值70%-90%）。

## 生产环境优化方案

### 调度与负载均衡

```
--schedule-policy lpm                   # 调度策略（lpm/fcfs）
--load-balance-method round-robin       # 负载均衡方式
--dist-init-addr 192.168.1.100:29500    # 多节点通信地址
```

- 多机部署要求
  - 网络延迟<1ms
  - 防火墙开放29500端口
  - 使用`--nnodes 3 --node-rank 0`指定节点角色

### 量化与推理加速

```
--quantization fp8                  # 启用FP8量化
--attention-backend flashinfer      # 优化注意力计算
--triton-attention-num-kv-splits 4  # Triton注意力KV分割数
```

- 量化效果：

  | 参数 | 显存占用 | 精度损失 |
  | :--- | :------- | :------- |
  | FP32 | 100%     | 无       |
  | FP8  | 45%      | <2%      |

## 调试与监控体系

### 日志与指标

```
--log-level info                  # 日志级别
--enable-metrics true             # 启用Prometheus指标
--metrics-port 8008               # 指标暴露端口
```

- 关键监控指标：
  - `sglang_requests_total`：请求总数
  - `sglang_gpu_utilization`：GPU利用率
  - `sglang_kv_cache_usage`：KV缓存使用率

### 调试工具

```
--debug-tensor-dump-output-folder /tmp/debug     # 张量转储目录
--enable-nan-detection true                      # 检测NaN值
--warmups 3                                      # 预热请求数
```

- 故障排查流程：
  1. 启用`--log-level debug`
  2. 检查张量转储文件
  3. 验证输入数据有效性

## 完整配置示例

```
python3 -m sglang.launch_server \
  --model-path /mnt/models/DeepSeek-R1 \
  --host 0.0.0.0 \
  --port 4000 \
  --device cuda \
  --tp 4 \
  --dp-size 2 \
  --mem-fraction-static 0.8 \
  --max-running-requests 32 \
  --quantization fp8 \
  --attention-backend flashinfer \
  --enable-metrics true \
  --log-level info
```

## 常见问题解决方案

### OOM错误处理

| 错误类型 | 解决方案                          |
| :------- | :-------------------------------- |
| 显存溢出 | 降低`--mem-fraction-static`至0.75 |
| 内存不足 | 减少`--max-total-tokens`至96000   |
| KV缓存满 | 增加`--kv-cache-size`参数         |

### 多节点同步失败

1. 检查`--dist-init-addr`网络可达性
2. 验证防火墙规则：`sudo ufw allow 29500`
3. 使用`--nnodes`和`--node-rank`明确节点角色
