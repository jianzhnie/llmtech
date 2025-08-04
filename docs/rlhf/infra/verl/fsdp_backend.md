# PyTorch FSDP 后端实现

本框架通过为Actor（actor）、Critic（critic）、Reference（reference）、Rollout（rollout）和Reward（reward model）实现对应的Worker（worker），全面支持 PyTorch 的 Fully Sharded Data Parallel（FSDP）后端。此外，我们还在 [fsdp_vllm.py](https://github.com/volcengine/verl/blob/main/verl/workers/sharding_manager/fsdp_vllm.py) 中实现了 `FSDPVLLMShardingManager`，用于在 FSDP 与 vLLM 之间进行权重重分片（resharding），以实现高效的模型状态转换。

## 优势与适用场景

### 优势

- **模型兼容性强，易于扩展**
  用户仅需实现相应的 `dtensor_weight_loader`，即可完成 FSDP 与 vLLM 之间的权重同步。对于支持 Hugging Face（HF）格式的模型，用户可直接使用 `hf_weight_loader`，无需额外修改代码，即可兼容任何同时被 HF 和 vLLM 支持的模型。

- **计算流程清晰，便于管理**
  FSDP 后端将各模型的前向传播与反向传播逻辑组织得更为清晰，有利于复杂训练流程的开发与调试。

### 劣势

- **大规模模型扩展性有限**
  在面对超大规模模型（如 Llama 70B 或 405B）时，FSDP 的内存和通信开销可能导致扩展性不足。

- **重分片开销较高**
  在 actor 与 rollout 模型之间进行权重重分片时，其通信与转换开销可能高于 Megatron-LM 后端。

鉴于其简洁性和开发友好性，我们**推荐将 FSDP 后端用于算法研究与原型开发阶段**，尤其适用于中小规模模型的快速迭代与验证。

---

## FSDP Worker实现

### ActorRolloutRefWorker

`ActorRolloutRefWorker` 是一个集成了Actor、Rollout和Reference功能的复合Worker，支持混合部署模式。

#### Actor/Rollout 混合引擎

##### 1. 模型初始化接口

```python
@register(dispatch_mode=Dispatch.ONE_TO_ALL)
def init_model(self):
```

- `Dispatch.ONE_TO_ALL`：当驱动进程调用 `init_model` 时，该函数将在每个 GPU Worker上并行执行，完成本地模型的初始化。

初始化流程主要包括以下组件：

- `DataParallelPPOActor`：封装了基于 FSDP 的 PPO 基础计算逻辑，包括对数概率（log probability）计算和模型参数更新。
- `vLLMRollout`：集成 vLLM 实现高效的自回归生成。我们对 vLLM 引擎进行了修改，使其支持 SPMD（Single Program, Multiple Data）模式，以适配 `WorkerGroup` 的分布式架构。
- `FSDPVLLMShardingManager`：作为上下文管理器，负责在Actor（FSDP）与Rollout（vLLM）之间执行权重重分片操作。

更多实现细节请参见 [源代码](https://github.com/volcengine/verl/blob/main/verl/workers/fsdp_workers.py)。

##### 2. 生成序列并重新计算对数概率

```python
@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
def generate_sequences(self, prompts: DataProto):
```

- `Dispatch.DP_COMPUTE_PROTO`：数据将沿数据并行维度进行分发与聚合。
- 该函数中，Rollout使用 vLLM 执行自回归生成，而Actor则对生成的响应重新计算其在旧策略下的对数概率 $ \log \pi_{\theta_{\text{old}}}(a|s) $，用于后续的 PPO 优势估计。

##### 3. 更新Actor模型

```python
@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
def update_actor(self, data: DataProto):
```

- 使用 PPO 目标函数（含策略梯度与熵正则项）更新Actor模型参数。PPO 损失函数定义如下：

$$
\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]
$$

其中 $ r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} $ 为概率比，$ \hat{A}_t $ 为优势函数估计值。

#### Reference Model

##### 1. Reference初始化

Reference复用Actor的初始化接口，但不初始化优化器和混合引擎组件。初始化完成后，模型由 `DataParallelPPOActor` 封装，仅用于前向推理。

##### 2. 计算参考对数概率

```python
@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
def compute_ref_log_prob(self, data: DataProto):
```

- 该函数调用 `DataParallelPPOActor` 中的对数概率计算模块，获取参考策略下生成动作的对数概率 $ \log \pi_{\theta_{\text{ref}}}(a|s) $，用于后续 KL 散度或奖励计算。

---

### CriticWorker与RewardWorker

#### 1. 模型初始化

Critic（Critic）与Reward（Reward Model）的初始化流程与Reference类似。区别在于，Critic还需初始化优化器以支持反向传播更新。

#### 2. 计算价值函数

```python
@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
def compute_values(self, data: DataProto):
```

- Critic模型接收状态 $ s $，输出状态价值估计 $ V_\phi(s) $，用于优势函数计算：

$$
\hat{A}_t = \delta_t + (\gamma \lambda) \delta_{t+1} + \cdots + (\gamma \lambda)^{T-t+1} \delta_{T-1}
$$

其中 $ \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t) $。

#### 3. 更新Critic

```python
@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
def update_critic(self, data: DataProto):
```

- 使用均方误差（MSE）损失函数更新价值网络参数：

$$
\mathcal{L}^{\text{value}}(\phi) = \mathbb{E}_t \left[ (V_\phi(s_t) - V_t^{\text{target}})^2 \right]
$$

#### 4. 计算奖励分数

```python
@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
def compute_rm_score(self, data: DataProto):
```

- RewardWorker调用预训练的Reward，对生成响应进行打分，输出标量奖励 $ r(s, a) $。

---

## 混合分片支持

当前版本**暂不支持 FSDP 混合分片模式**（Hybrid Sharding）。若需实现该功能，可能需要：

1. 构建二维设备网格（2D device mesh），结合张量并行与数据并行；
2. 为不同模型分别设计和测试 `dtensor_weight_loader` 与 `hf_weight_loader` 的适配逻辑；
3. 实现跨分片策略的权重映射与同步机制。

未来版本将考虑引入对混合分片的支持，以提升大规模模型训练的效率与灵活性。
