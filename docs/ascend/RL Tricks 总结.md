# RL Tricks 总结

## LLM专用技巧

### EOS_token

从实现细节上来说，我们假定使用 GPT 模型作为这个奖励模型的基础架构，那么我们有多种思路来预测一个对话的奖励值。

1. 对所有token上预测reward取平均
2. 在最后一个token，即EOS_token上预测reward

这里我们倾向于方式2，因为对于GPT这种模型，只有EOS_token的输出才能看完整句话给出一个整体的评价。这种方式也在 Anthropic 的相关论文中被使用。

### 基于采样温度缩放（Logits Scaling）

在计算响应序列的对数概率时，模型首先生成响应词元的原始 logits，随后通过采样温度（sampling temperature）对逻辑值进行缩放。具体实现如下：

```python
logits /= self.temperature
```

关键发现：
在测试中，我们发现若取消该缩放操作，KL散度上升速度会超出预期，导致模型性能显著下降。

### Token Level 惩罚

计算RL模型与SFT模型响应分布的逐词元KL散度[11]，并将其作为惩罚项加入奖励函数。具体公式如下：
$$
r(s_t, a_t) = \mathbf{I}(s_t = [\text{EOS}]) r(x, y) - \beta \text{KL}(t) \quad (1)
$$

$$
\text{KL}(t) = \log\left(\frac{\pi_{\theta_{\text{old}}}(a_t|s_t)^{\text{RL}}}{\pi^{\text{SFT}}(a_t|s_t)}\right) \quad (2)
$$

其中 $ x $ 为提示，$ y $ 为响应，$ \mathbf{I}(s_t = [\text{EOS}]) $ 为指示函数，判断当前词元是否为终止符。

### 广义优势估计（GAE）

使用GAE[10]（一种TD($ \lambda $)回报估计方法）估计PPO中的词元级奖励。实践中通常设 $ \lambda = 1 $，将GAE退化为蒙特卡洛估计。

### 添加SFT损失

在PPO中结合监督式下一词元预测损失与KL散度，以保留SFT模型的预训练能力。

## PPO专用技巧

### 模型初始化

训练LLM时需初始化两个模型——Actor模型（Actor）和评论家模型（Critic）[6,7]。Actor模型通常基于SFT模型初始化，评论家模型基于奖励模型初始化。

### 限制训练轮数

策略网络训练时限制轮数为1，防止模型偏离旧策略分布。

### 折现因子 $\gamma =1 $

折现因子 $\gamma$  设置为 1 ，这意味着未来的奖励与即时奖励具有相同的权重。

### 小批量更新

在训练阶段，PPO对大小为 $ N \times M $（$ N $为回放缓冲池大小，$ M $为响应长度）的数据进行索引混洗，并按小批量计算梯度并更新策略。

### Adam学习率

Actor模型的Adam学习率约为SFT模型的1/10（例如SFT学习率为 $ 5e^{-6} $，Actor模型为 $ 5e^{-7} $）。评论家模型的学习率约为SFT模型的2倍（例如 $ 9e^{-6} $）。

### 价值函数损失截断

PPO对价值函数进行截断[5]，损失函数定义为：
$$
\text{Loss}_v = \max\left[(V_{\theta_t} - V_{\text{targ}})^2, \left(\text{clip}(V_{\theta_t}, V_{\theta_{t-1}} - \epsilon, V_{\theta_{t-1}} + \epsilon) - V_{\text{targ}}\right)^2\right] \quad (3)
$$

### 奖励归一化与截断

在 PPO 的训练种，我们通常会使用 reward normalization 以及 value normalization 等类似技术，我们发现在 RLHF 的训练中 reward normalization 非常有助于训练的稳定性，毕竟我们的 reward 不像在游戏环境中那么规则，而是通过一个模型学出来的中间层输出（这就意味着输出范围可能会很大）。

为避免奖励分布不均衡，采用 Z-score 归一化 $ r = (r - \mu)/\delta $，其中 $ \mu $ 和 $ \delta $ 分别为奖励数据集的均值和标准差。

### 奖励进行移动平均

```python
def whiten(values, shift_mean=True):
    mean, var = torch.mean(values), torch.var(values, unbiased=False)
    whitened = (values - mean)* torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened
```

在每个小批次中，使用 `whiten(rewards, shift_mean=False)` 对奖励进行白化，不对均值进行平移处理 ， 并使用平移后的均值对优势进行白化 whiten(advantages)。

### 优势归一化

同样 Advantage Normalization 也是PPO训练种常用的稳定训练的技术，我们在使用 DeepSpeed 等类似 DDP 的训练手段时应注意做全局样本的 Advantage Normalization，而不是某个DDP进程只针对自己的样本做归一化。

在价值网络训练中，对优势值进行Z-score归一化以抑制异常值影响。

### 自适应 KL 散度

KL 散度惩罚系数 (\beta) 根据当前策略与先前策略之间的 KL 散度自适应修改。如果 KL 散度超出预定的目标范围，则调整惩罚系数以使其更接近目标范围 。它的实现如下:

```python
class AdaptiveKLController:
    def __init__(self, init_kl_coef, hparams):
        self.value = init_kl_coef
        self.hparams = hparams

    def update(self, current, n_steps):
        target = self.hparams.target
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.hparams.horizon
        self.value *= mult
```

对于openai 研究的 `sentiment` 和 `descriptiveness` 任务，使用了 `init_kl_coef=0.15, hparams.target=6, hparams.horizon=10000` 。

## 创新策略

### 冷启动

R1在冷启动SFT的过程中比较克制，大概只用了几千条数据，这其中主要的动机是避免在冷启动SFT阶段过度模仿，保证RL阶段探索时候的多样性。

### 采样策略

k1.5的报告中介绍了在训练过程中使用到的两个比较实用的prompt sampling策略：

a) 课程采样 (Curriculum Sampling)：模型在刚开始训练时，在非常难的问题上进行大量探索可能会导致训练效率低下。从简单任务开始，逐步过渡到更具挑战性的任务有助于提升训练效率和稳定性。这一步也正好可以利用到前述数据准备中不同prompt的难度信息

b) 优先级采样 (Prioritized Sampling)： 通过跟踪每个问题 i 的成功率 si ，采样概率与 (1−si) 成比例，成功率越低的问题获得越高的采样概率，这种上采样方式可以将算力更加合理地分配到模型表现尚不好的问题上，进一步提升训练效率.

### 优化策略

关于优化算法的选择，很多工作选择在RLHF的过程中舍弃掉value/critic model，并基于类REINFORCE算法在优势估计方面进行尝试了各种改进，GRPO也可以视为其中的一种。近期有人指出[[9\]](https://zhuanlan.zhihu.com/p/25579111309#ref_9)[[10\]](https://zhuanlan.zhihu.com/p/25579111309#ref_10)舍去GRPO中的reference model 改成对梯度进行裁剪也能正常收敛，对于显存紧张的情况也是一种候选的思路，但是否会影响大规模训练时候的稳定性可能还需要进一步验证。

### 初始训练阶段冻结Actor参数

避免策略更新导致响应分布偏移，确保评论家模型学习准确的优势估计[6]。

### 奖励基线

引入参考基线 $ r(x,y) = r(x,y) - r(x,y_{\text{ref}}) $，其中 $ y_{\text{ref}} $ 为当前策略生成的参考响应，以消除绝对奖励值的波动影响。

### PPO结合拒绝采样微调

从模型中采样K个输出，基于奖励模型选择最优候选进行微调。Touvron等人[9]在Llama 2中交替使用拒绝采样和PPO更新策略。

## Reference:

- https://sage-kite-7e1.notion.site/rlhf-implementation-tricks?v=158d9a33ecc98132bf9e000c39227361&pvs=25

- https://sage-kite-7e1.notion.site/reinforce-plus-plus
