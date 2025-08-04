# CriticWorker



## 整体概述

`CriticWorker` 是基于 FSDP（Fully Sharded Data Parallel）的价值函数Worker，负责在 PPO 训练中估计状态价值并更新价值网络。 fsdp_workers.py:915-918 它继承自 `Worker` 和 `DistProfilerExtension`，支持分布式训练和性能分析。

## 类初始化和配置

### 构造函数核心逻辑 fsdp_workers.py:916-925

构造函数首先初始化分布式环境，如果 PyTorch 分布式未初始化，会自动设置 NCCL 后端的进程组。

### 设备网格和序列并行配置 fsdp_workers.py:927-942

代码构建了两种设备网格：

- **FSDP 设备网格**：用于参数分片
- **Ulysses 设备网格**：用于序列并行，支持长序列处理

### 批次大小标准化 fsdp_workers.py:948-970

这段代码对训练批次大小进行标准化处理：

- 根据 rollout 数量调整 mini batch size
- 按设备数量和序列并行大小缩放
- 确保 micro batch size 能被 mini batch size 整除

## 模型构建核心方法

### _build_critic_model_optimizer 方法 fsdp_workers.py:972-1172

这是价值模型构建的核心方法，包含以下关键步骤：

#### 1. 分词器和处理器初始化 fsdp_workers.py:985-987

#### 2. 模型配置和特殊适配 fsdp_workers.py:1009-1017

关键配置包括：

- 设置 `num_labels = 1` 用于价值函数输出
- 为 kimi-vl 等特殊模型设置专用配置

#### 3. Dropout 配置优化 fsdp_workers.py:1025-1027

禁用各种 dropout 以确保价值函数训练的稳定性。

#### 4. 价值头模型加载 fsdp_workers.py:1029-1034

使用 `load_valuehead_model` 加载带有价值头的模型。

#### 5. FSDP 包装策略 fsdp_workers.py:1093-1128

支持两种 FSDP 策略：

- **FSDP v1**：传统的 FSDP 包装
- **FSDP v2**：新版本的 fully_shard API

#### 6. 优化器和学习率调度器 fsdp_workers.py:1136-1171

配置 AdamW 优化器和学习率调度器（支持 constant 和 cosine 两种预热策略）。

## 核心训练方法

### init_model 方法 fsdp_workers.py:1174-1203

模型初始化入口点，使用 `@register(dispatch_mode=Dispatch.ONE_TO_ALL)` 装饰器确保所有Worker同步初始化。

### compute_values 方法 fsdp_workers.py:1205-1227

价值计算的核心方法：

1. 将数据移动到 GPU
2. 处理参数卸载（如果启用）
3. 使用 Ulysses 分片管理器处理序列并行
4. 调用 critic 的 `compute_values` 方法
5. 返回计算结果并移回 CPU

### update_critic 方法 fsdp_workers.py:1229-1264

价值函数更新的核心方法：

1. 数据预处理和设备管理
2. 执行价值函数更新训练
3. 计算性能指标（MFU、内存使用等）
4. 更新学习率调度器
5. 返回训练指标

## 检查点管理

### 保存和加载检查点 fsdp_workers.py:1266-1297

支持分布式检查点的保存和加载，处理参数卸载的同步问题。

## 技术要点

### 1. HybridFlow 架构集成

所有核心方法都使用 `@register` 装饰器，支持不同的分发模式：

- `Dispatch.ONE_TO_ALL`：广播到所有Worker
- `Dispatch.DP_COMPUTE_PROTO`：数据并行计算和收集

### 2. 内存优化策略

- **参数卸载**：支持将 FSDP 参数卸载到 CPU
- **优化器卸载**：支持将优化器状态卸载到 CPU
- **激活卸载**：可选的激活检查点功能

### 3. 分布式训练支持

- **FSDP 集成**：使用 PyTorch FSDP 进行参数分片
- **序列并行**：通过 Ulysses 支持长序列处理
- **混合精度**：支持 bfloat16 参数和 fp32 梯度

### 4. 性能监控

集成了 FLOPS 计算器和内存使用监控，提供详细的训练性能指标。

## 潜在改进建议

1. **错误处理增强**：可以添加更详细的异常处理和恢复机制
2. **配置验证**：在初始化时验证配置参数的一致性
3. **动态批次大小**：考虑支持动态调整批次大小以优化内存使用
4. **梯度累积优化**：可以优化梯度累积策略以提高训练效率

## Notes

`CriticWorker` 是 verl 框架中价值函数训练的核心实现，与 `ActorRolloutRefWorker` 配合完成完整的 PPO 训练流程。它通过 FSDP 实现高效的分布式训练，支持大规模模型的价值函数学习。该类在 PPO 训练中负责估计状态价值，为优势函数计算提供基础。
