# 从 Online Softmax 到 FlashAttention

## 1. 引言

在大语言模型（LLM）快速发展的今天，注意力机制（Attention Mechanism）作为Transformer架构的核心组件，其计算效率直接影响着模型的性能表现。然而，标准的注意力计算在处理长序列时会遇到显存和计算效率的双重瓶颈。本文将深入探讨**Online Softmax**技术，这是理解**FlashAttention**算法（一种显著提升Transformer计算效率的创新方法）的关键技术基础。

## 2. 自注意力机制基础

### 2.1 标准自注意力计算

自注意力计算可以表示为（为简化讨论，我们忽略多头和批次维度，因为这些维度的计算是完全并行的；同时省略注意力掩码和缩放因子$\frac{1}{\sqrt{D}}$）：

$$
O = \text{softmax}(QK^{T})V
$$

其中，$Q, K, V, O$均为形状为$(L, D)$的二维矩阵，$L$表示序列长度，$D$为每个注意力头的维度。Softmax操作应用于矩阵的最后一个维度。

标准自注意力计算分为三个阶段：

$$
X = QK^{T}
$$

$$
A = \text{softmax}(X)
$$

$$
O = AV
$$

这里，$X$ 称为预Softmax逻辑值（Pre-softmax Logits），$A$为注意力分数矩阵，$O$为最终输出矩阵。

## 3. 矩阵乘法与分块优化

### 3.1 计算瓶颈分析

传统矩阵乘法在GPU上执行时会频繁进行内存访问，导致大量缓存未命中，严重影响计算性能：

```python
import numpy as np

def naive_matmul(A, B):
    n, m = A.shape
    m2, p = B.shape
    assert m == m2, "维度不匹配"
    C = np.zeros((n, p))
    for i in range(n):
        for j in range(p):
            for k in range(m):
                C[i, j] += A[i, k] * B[k, j]
    return C
```

### 3.2 分块矩阵乘法

为提高矩阵乘法在硬件上的缓存性能，我们采用**分块矩阵乘法（Tiled Matrix Multiplication）**技术。该方法将大矩阵分解为可放入高速缓存（如SRAM）的小块（Tiles），通过数据复用最大化计算效率：

```python
import numpy as np

def tiled_matmul(A, B, tile_size):
    n, m = A.shape
    m2, p = B.shape
    assert m == m2, "Incompatible dimensions"

    C = np.zeros((n, p))
    for ii in range(0, n, tile_size):
        for jj in range(0, p, tile_size):
            for kk in range(0, m, tile_size):
                for i in range(ii, min(ii + tile_size, n)):
                    for j in range(jj, min(jj + tile_size, p)):
                        for k in range(kk, min(kk + tile_size, m)):
                            C[i, j] += A[i, k] * B[k, j]
    return C
```

分块技术显著减少了缓存未命中，在机器学习场景中可带来数十倍的性能提升。注意力机制本质上是一种特殊的矩阵运算，因此同样适用此类优化策略。

## 4. Online Softmax算法

### 4.1 Softmax的数值稳定性问题

Softmax 函数将一组数值转化为概率分布，通过指数运算让较大的数值更加突出，较小的数值则被抑制。其公式如下：

$$
\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}
$$

直接计算$e^{x_i}$容易导致**数值溢出 （Overflow）** 。例如，在FP16格式下，最大可表示数值约为65504，而$e^{11.1}$就已超出此范围。

### 4.2 安全Softmax（Safe Softmax）

为解决数值溢出问题，通常采用**安全Softmax（Safe softmax）**：

1. 找到输入向量最大值：$m = \max(x)$
2. 每个元素减去最大值后进行指数运算：

$$
\text{Softmax}(x_i) = \frac{e^{x_i - m}}{\sum_{j=1}^{n} e^{x_j - m}}
$$

由于$x_i - m \leq 0$，确保$e^{x_i - m} $ 的范围就在 $(0, 1]$ 之间，完美解决溢出问题。

### 4.3 三轮遍历Safe Softmax算法

标准Safe Softmax需要三次遍历：

**算法：3-pass Safe-Softmax**

- **符号定义**：
  - $m_i = \max_{j=1}^{i}\{x_j\}$，初始值$m_0 = -\infty$
  - $d_i = \sum_{j=1}^{i}e^{x_j - m_N}$，初始值$d_0 = 0$
  - $a_i$：最终Softmax值

- **算法步骤**：
  1. 遍历$i = 1$到$N$：
     $$
     m_i = \max(m_{i-1}, x_i)
     $$

  2. 遍历$i = 1$到$N$：
     $$
     d_i = d_{i-1} + e^{x_i - m_N}
     $$

  3. 遍历$i = 1$到$N$：
     $$
     a_i = \frac{e^{x_i - m_N}}{d_N}
     $$

```python
import numpy as np

def softmax_3pass(input_array):
    n = len(input_array)
    output = np.zeros(n, dtype=float)

    # First pass: find max
    max_val = input_array[0]
    for i in range(1, n):
        if input_array[i] > max_val:
            max_val = input_array[i]

    # Second pass: compute exp(x - max) and sum
    sum_val = 0.0
    for i in range(n):
        output[i] = np.exp(input_array[i] - max_val)
        sum_val += output[i]

    # Third pass: normalize
    for i in range(n):
        output[i] /= sum_val

    return output
```

该算法需要三次遍历。在Transformer自注意力中，$\{x_{i}\}$ 是由 $QK^{T}$ 计算出的 Pre-Softmax Logits。这意味着如果我们不存储所有 Logits $\{x_{i}\}_{i=1}^{N}$（因为没有足够大的 SRAM 来容纳它们），我们就需要访问 $Q$ 和 $K$ 三次（以便动态地重新计算 Logits），这在 I/O 上是极其低效的。

### 4.4 两轮遍历Online Softmax

如果我们将上述 Safe Softmax 的三步计算训练循环融合在一个循环中，就可以将全局内存访问次数从 3 次减少到 1 次。遗憾的是，我们不能在同一个循环中融合公式7和8，因为 8 依赖于 $m_{N}$，而 $m_{N}$ 直到第一个循环结束才能确定。

**Online Softmax** 通过动态更新全局统计量，将遍历次数从三轮减至两轮。核心思想是创建替代序列$d_i' = \sum_{j=1}^{i}e^{x_j - m_i}$，建立递推关系：
$$
d_i' = d_{i-1}'e^{m_{i-1} - m_i} + e^{x_i - m_i}
$$

> 推导：我们可以创建另一个序列 $d_{i}^{\prime} := \sum_{j=1}^{i}e^{x_{j}-m_{i}}$ 作为原始序列 $d_{i} := \sum_{j=1}^{i}e^{x_{j}-m_{N}}$ 的替代（Surrogate），以消除对 $N$ 的依赖。由于这两个序列的第 $N$ 项是相同的：$d_{N} = d_{N}^{\prime}$，因此我们可以安全地用 $d_{N}^{\prime}$ 替换方程 5 中的 $d_{N}$。我们还可以找到 $d_{i}^{\prime}$ 和 $d_{i-1}^{\prime}$ 之间的递推关系：
>
> $$d_{i}^{\prime} = \sum_{j=1}^{i}e^{x_{j}-m_{i}}$$
>
> $$= \left(\sum_{j=1}^{i-1}e^{x_{j}-m_{i}}\right) + e^{x_{i}-m_{i}}$$
>
> $$= \left(\sum_{j=1}^{i-1}e^{x_{j}-m_{i-1}}\right)e^{m_{i-1}-m_{i}} + e^{x_{i}-m_{i}}$$
>
> $$= d_{i-1}^{\prime}e^{m_{i-1}-m_{i}} + e^{x_{i}-m_{i}}$$
>
> 这种递推形式仅依赖于 $m_{i}$ 和 $m_{i-1}$，因此我们可以在同一个循环中同时计算 $m_{j}$ 和 $d_{j}^{\prime}$：

**算法：2-pass Online Softmax**

1. 遍历$i = 1$到$N$：
   $$
   m_i = \max(m_{i-1}, x_i)
   $$

   $$
   d_i' = d_{i-1}'e^{m_{i-1} - m_i} + e^{x_i - m_i}
   $$

2. 遍历$i = 1$到$N$：
   $$
   a_i = \frac{e^{x_i - m_N}}{d_N'}
   $$

```python
import numpy as np

def softmax_online(input_array):
    n = len(input_array)
    output = np.zeros(n, dtype=float)

    # Initialize running maximum with first element
    m = input_array[0]
    d = 1.0  # Running sum

    # Pre-pass to compute final max and total sum
    for i in range(1, n):
        if input_array[i] > m:
            # Adjust the sum when we find a new maximum
            d = d * np.exp(m - input_array[i]) + 1.0
            m = input_array[i]
        else:
            # Add the contribution of this element to the sum
            d += np.exp(input_array[i] - m)

    # Final pass to compute softmax outputs
    for i in range(n):
        output[i] = np.exp(input_array[i] - m) / d

    return output
```

Online Softmax在第一轮遍历中同时计算全局最大值$m$和最终分母$d$，第二轮直接计算输出结果。这种在线更新机制为FlashAttention奠定了基础。

## 5. FlashAttention：单轮遍历的突破

### 5.1 核心思想

虽然Online Softmax仍需两轮遍历，但在自注意力计算中，我们的最终目标是获得输出矩阵$O = AV$，而非注意力分数矩阵$A$本身。这启发我们寻找输出矩阵$O$的单轮遍历递推形式。

### 5.2 递推公式推导

让我们尝试将自注意力计算的第 $k$ 行（各行的计算是独立的，为简单起见，我们仅解释一行的计算）公式化为递推算法：

**算法：多轮自注意力**

- **符号定义**：
  - $Q[k,:]$：$Q$ 矩阵的第 $k$ 行向量
  - $K^{T}[:,i]$：$K^{T}$ 矩阵的第 $i$ 列向量
  - $O[k,:]$：输出 $O$ 矩阵的第 $k$ 行
  - $V[i,:]$：$V$ 矩阵的第 $i$ 行
  - $\{o_{i}\} : \sum_{j=1}^{i}a_{j}V[j,:]$，存储局部聚合结果 $A[k,:i] \times V[:i,:]$ 的行向量

- **算法步骤**：
  1. 遍历 $i \leftarrow 1$ 到 $N$：
     - $x_{i} \leftarrow Q[k,:]K^{T}[:,i]$
     - $m_{i} \leftarrow \max(m_{i-1}, x_{i})$
     - $d_{i}^{\prime} \leftarrow d_{i-1}^{\prime}e^{m_{i-1}-m_{i}} + e^{x_{i}-m_{i}}$

  2. 遍历 $i \leftarrow 1$ 到 $N$：
     - $a_{i} \leftarrow \frac{e^{x_{i}-m_{N}}}{d_{N}^{\prime}}$
     - $o_{i} \leftarrow o_{i-1} + a_{i}V[i,:]$

  3. $O[k,:] \leftarrow o_{N}$



我们将 $a_{i}$ 替换为 online-softmax 的定义 $a_{i} \leftarrow \frac{e^{x_{i}-m_{N}}}{d_{N}^{\prime}}$：

定义替代序列$o_i' = \sum_{j=1}^{i} \frac{e^{x_j - m_i}}{d_i'} V[j,:]$，建立递推关系：
$$
o_i' = o_{i-1}' \frac{d_{i-1}'e^{m_{i-1} - m_i}}{d_i'} + \frac{e^{x_i - m_i}}{d_i'} V[i,:]
$$

我们将方程 12 中的 $a_{i}$ 替换为方程 11 中的定义：

$$
o_{i} := \sum_{j=1}^{i} \left( \frac{e^{x_{j}-m_{N}}}{d_{N}^{\prime}} V[j,:] \right)
$$

这仍然依赖于 $m_{N}$ 和 $d_{N}$，它们在之前的循环完成前无法确定。但我们可以再次使用第 3 节中的"替代（Surrogate）"技巧，创建一个替代序列 $o^{\prime}$：

$$
o_{i}^{\prime} := \sum_{j=1}^{i} \left( \frac{e^{x_{j}-m_{i}}}{d_{i}^{\prime}} V[j,:] \right)
$$

$o$ 和 $o^{\prime}$ 的第 $N$ 个元素是相同的：$o_{N}^{\prime} = o_{N}$，并且我们可以找到 $o_{i}^{\prime}$ 和 $o_{i-1}^{\prime}$ 之间的递推关系：

$$
o_{i}^{\prime} = \sum_{j=1}^{i} \frac{e^{x_{j}-m_{i}}}{d_{i}^{\prime}} V[j,:]
$$

$$
= \left( \sum_{j=1}^{i-1} \frac{e^{x_{j}-m_{i}}}{d_{i}^{\prime}} V[j,:] \right) + \frac{e^{x_{i}-m_{i}}}{d_{i}^{\prime}} V[i,:]
$$

$$
= \left( \sum_{j=1}^{i-1} \frac{e^{x_{j}-m_{i-1}}}{d_{i-1}^{\prime}} V[j,:] \right) \frac{d_{i-1}^{\prime}e^{m_{i-1}-m_{i}}}{d_{i}^{\prime}} + \frac{e^{x_{i}-m_{i}}}{d_{i}^{\prime}} V[i,:]
$$

$$
= o_{i-1}^{\prime} \frac{d_{i-1}^{\prime}e^{m_{i-1}-m_{i}}}{d_{i}^{\prime}} + \frac{e^{x_{i}-m_{i}}}{d_{i}^{\prime}} V[i,:]
$$

这仅取决于 $d_{i}^{\prime}, d_{i-1}^{\prime}, m_{i}, m_{i-1}$ 和 $x_{i}$，因此我们可以将自注意力的所有计算融合在一个循环中：

**算法：FlashAttention**

遍历$i = 1$到$N$：
1. $x_i = Q[k,:]K^{T}[:,i]$
2. $m_i = \max(m_{i-1}, x_i)$
3. $d_i' = d_{i-1}'e^{m_{i-1} - m_i} + e^{x_i - m_i}$
4. $o_i' = o_{i-1}' \frac{d_{i-1}'e^{m_{i-1} - m_i}}{d_i'} + \frac{e^{x_i - m_i}}{d_i'} V[i,:]$

最终$O[k,:] = o_N'$。状态变量$x_i, m_i, d_i', o_i'$占用内存极小，可轻松放入GPU共享内存。

### 5.3 分块版FlashAttention

**算法：FlashAttention（分块版）**

- **符号定义**：
  - $b$：分块大小
  - $\#tiles$：每行分块数量，$N = b \times \#tiles$
  - $x_i$：第$i$个分块的$Q[k]K^T$值向量
  - $m_i^{(local)}$：$x_i$的局部最大值

- **算法步骤**：
  遍历$i = 1$到$\#tiles$：
  1. $x_i = Q[k,:]K^{T}[:,(i-1)b : ib]$
  2. $m_i^{(local)} = \max_{j=1}^{b}(x_i[j])$
  3. $m_i = \max(m_{i-1}, m_i^{(local)})$
  4. $d_i' = d_{i-1}'e^{m_{i-1} - m_i} + \sum_{j=1}^{b}e^{x_i[j] - m_i}$
  5. $o_i' = o_{i-1}' \frac{d_{i-1}'e^{m_{i-1} - m_i}}{d_i'} + \sum_{j=1}^{b} \frac{e^{x_i[j] - m_i}}{d_i'} V[j+(i-1)b, :]$

在 FlashAttention 中：

1. 我们按块读取 $Q, K, V$。
2. 在 SRAM 中计算 $QK^T$ 的局部块。
3. 使用 Online Softmax 的逻辑，通过存储局部最大值和分母，动态地更新输出矩阵 $O$。
4. 最终，我们只需要将 $O$ 写回主存。

```python
import numpy as np

def flash_attention(Q, K, V, k):
    """
    FlashAttention单头计算实现

    参数:
    Q: Query矩阵
    K: Key矩阵（计算中转置）
    V: Value矩阵
    k: Query行索引

    返回:
    输出向量O[k,:]，等价于softmax(Q[k,:] @ K) @ V
    """
    N = K.shape[1]  # 从K矩阵获取维度

    # 初始化变量
    m_i_minus_1 = float('-inf')  # m_{i-1}初始值
    d_i_minus_1 = 0.0  # d'_{i-1}初始值
    o_i_minus_1 = np.zeros_like(V[0, :])  # o'_{i-1}初始值

    for i in range(N):
        # 计算x_i：Q的第k行与K^T的第i列点积
        x_i = np.dot(Q[k, :], K[:, i])

        # 更新最大值
        m_i = max(m_i_minus_1, x_i)

        # 计算d'_i
        d_i = d_i_minus_1 * np.exp(m_i_minus_1 - m_i) + np.exp(x_i - m_i)

        # 计算o'_i
        o_i = (o_i_minus_1 * d_i_minus_1 * np.exp(m_i_minus_1 - m_i) / d_i) + \
              (np.exp(x_i - m_i) / d_i) * V[i, :]

        # 更新迭代变量
        m_i_minus_1 = m_i
        d_i_minus_1 = d_i
        o_i_minus_1 = o_i

    return o_i_minus_1  # 返回o'_N作为结果
```

FlashAttention通过在线融合机制，将原本至少3遍的注意力计算降至**1遍**，在GPU片上内存中完成计算，避免序列长度二次方的内存开销。

## 6. 技术优势分析

### 6.1 内存效率提升

FlashAttention避免了存储巨大的中间注意力矩阵（如$N \times N$注意力分数矩阵），显著降低显存带宽消耗，使在有限显存下处理超长序列（如100k+上下文）成为可能。

### 6.2 计算性能优化

通过将$QK^T$乘法、Softmax和$V$乘法三个步骤融合为单一CUDA kernel，FlashAttention实现了：
- 内存访问次数大幅减少
- 计算与内存访问重叠
- 更好的GPU资源利用率

## 7. 总结与展望

### 7.1 技术演进路径

- **传统Softmax**：三轮遍历，显存受限瓶颈
- **Online Softmax**：两轮遍历，支持分块计算
- **FlashAttention**：单轮遍历效果，端到端优化

### 7.2 应用价值

FlashAttention的创新不仅提升了Transformer的训练和推理速度，更重要的是为处理超长序列提供了可行的技术方案，推动了长上下文理解、文档处理等领域的发展。

## **参考文献**：

[1] Dao, T., et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. *NeurIPS*.

[2] Milakov, M., & Gimelshein, N. (2018). Online normalizer calculation for softmax. *arXiv preprint arXiv:1805.02867*.
