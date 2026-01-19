# 从 Online Softmax 到 FlashAttention

## 1. 简介

在深度学习领域，尤其是大语言模型（LLM）中，注意力机制（Attention Mechanism）是核心组件。然而，标准的注意力计算在处理长序列时会遇到显存和计算效率的瓶颈。本文将深入探讨一种名为 **Online Softmax** 的技术，它是理解 **FlashAttention**（一种极大提升 Transformer 速度的算法）的关键所在。

## 2. 自注意力 (The Self-Attention)

自注意力的计算可以概括如下（为简化起见，我们忽略了头和 Batch 维度，因为这些维度的计算是完全并行的；同时省略了注意力掩码和缩放因子 $\frac{1}{\sqrt{D}}$）：

$$
O = \text{softmax}(QK^{T})V
$$
其中，$Q, K, V, O$ 均为形状为 $(L, D)$ 的二维矩阵，其中 $L$ 是序列长度，$D$ 是每个头的维度（即头维度），Softmax 应用于最后一个维度（列）。

计算自注意力的标准方法是将计算分解为几个阶段：

$$X = QK^{T}$$

$$A = \text{softmax}(X)$$

$$O = AV$$

我们称 $X$ 矩阵为 Pre-softmax Logits，$A$ 矩阵为注意力分数，$O$ 矩阵为输出。

## 3. 矩阵乘法与分块（Tiling）

我们从一个简单的 GPU 计算开始：矩阵相乘。传统矩阵乘法需要反复将矩阵的行与列载入内存，这会导致大量缓存未命中和性能下降。

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

为了提高矩阵乘法在硬件（如 GPU）上的缓存性能，我们通常使用**分块矩阵乘法（Tiled Matrix Multiplication）**。该技术将大矩阵分解为更小的子矩阵（块/Tiles），这些小块可以放入高速缓存（如 SRAM）中。我们不再按顺序计算整个乘积，而是逐块处理，从而尽可能地在更快的内存中复用数据。

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

这种方法显著减少了缓存未命中情况，并在 ML 场景中带来数十倍性能提升。注意力机制可以视作一种特殊形式的矩阵运算，因此优化策略同样适用。

虽然我们可以对 $Q$（Query）和 $K$（Key）矩阵进行分块矩阵乘法，但在进行最后一次矩阵乘法（与 $V$ 矩阵相乘）之前，我们需要对生成的矩阵进行 **Softmax** 处理。因此，优化注意力机制的一个关键步骤是搞清楚如何高效地处理这个 Softmax 方程。

## 3. Online Softmax

### 3.1 Softmax 与数值溢出

Softmax 函数将一组数值转化为概率分布，通过指数运算让较大的数值更加突出，较小的数值则被抑制。其公式如下：
$$
Softmax(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}
$$
在计算机中直接计算 $e^{x_i}$ 极易导致**数值溢出（Overflow）**。例如，在 FP16（半精度浮点数）格式下，最大可表示的数值约为 $65504$。而 $e^{11.1}$ 就已经超过了这个范围。

为了解决这个问题，我们通常使用**安全 Softmax（Safe Softmax）**：

1. 找到输入向量中的最大值 $m = \max(x)$。
2. 从每个元素中减去这个最大值，然后再进行指数运算。

$$
Softmax(x_i) = \frac{e^{x_i - m}}{\sum_{j=1}^{n} e^{x_j - m}}
$$

由于 $x_i - m \leq 0$，那么 $e^{x_i - m}$ 的范围就在 $(0, 1]$ 之间，从而完美解决了溢出问题。

### 3.2 Safe Softmax ：三轮遍历

为了有效计算 Softmax，我们可以将其分为三个步骤。由于我们需要遍历输入张量三次，我们称之为（3 Passes）。

####  算法：3-pass Safe-Softmax

- **符号说明**：

  - $\{m_{i}\} : \max_{j=1}^{i}\{x_{j}\}$，初始值 $m_{0} = -\infty$。
  - $\{d_{i}\} : \sum_{j=1}^{i}e^{x_{j}-m_{N}}$，初始值 $d_{0} = 0$，$d_{N}$ 是安全 Softmax 的分母。
  - $\{a_{i}\}$：最终的 Softmax 值。

- **主体**：

  1. 遍历 $i \leftarrow 1$ 到 $N$：
     $$
     m_{i} \leftarrow \max(m_{i-1}, x_{i})
     $$

  2. 遍历 $i \leftarrow 1$ 到 $N$：

     $$
     d_{i} \leftarrow d_{i-1} + e^{x_{i}-m_{N}}
     $$

  3. 遍历 $i \leftarrow 1$ 到 $N$：
     $$
     a_{i} \leftarrow \frac{e^{x_{i}-m_{N}}}{d_{N}}
     $$

翻译成代码语言就是：

1. **第一轮**：遍历所有元素，找到全局最大值 $m$。
2. **第二轮**：计算每个元素的 $e^{x_i - m}$ 并求和，得到分母 $d = \sum e^{x_i - m}$。
3. **第三轮**：再次遍历，计算最终的输出 $y_i = \frac{e^{x_i - m}}{d}$。

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

该算法需要我们对 $[1, N]$ 进行 3 次迭代。在 Transformer 的自注意力语境下，$\{x_{i}\}$ 是由 $QK^{T}$ 计算出的 Pre-Softmax Logits。这意味着如果我们不存储所有 Logits $\{x_{i}\}_{i=1}^{N}$（因为没有足够大的 SRAM 来容纳它们），我们就需要访问 $Q$ 和 $K$ 三次（以便动态地重新计算 Logits），这在 I/O 上是极其低效的。

### 3.3. Online Softmax：两轮遍历

如果我们将上述 Safe Softmax 的三步计算训练循环融合在一个循环中，就可以将全局内存访问次数从 3 次减少到 1 次。遗憾的是，我们不能在同一个循环中融合公式3和4，因为4 依赖于 $m_{N}$，而 $m_{N}$ 直到第一个循环结束才能确定。

我们可以创建另一个序列 $d_{i}^{\prime} := \sum_{j=1}^{i}e^{x_{j}-m_{i}}$ 作为原始序列 $d_{i} := \sum_{j=1}^{i}e^{x_{j}-m_{N}}$ 的替代（Surrogate），以消除对 $N$ 的依赖。由于这两个序列的第 $N$ 项是相同的：$d_{N} = d_{N}^{\prime}$，因此我们可以安全地用 $d_{N}^{\prime}$ 替换方程 5 中的 $d_{N}$。

我们还可以找到 $d_{i}^{\prime}$ 和 $d_{i-1}^{\prime}$ 之间的递推关系：

$$d_{i}^{\prime} = \sum_{j=1}^{i}e^{x_{j}-m_{i}}$$

$$= \left(\sum_{j=1}^{i-1}e^{x_{j}-m_{i}}\right) + e^{x_{i}-m_{i}}$$

$$= \left(\sum_{j=1}^{i-1}e^{x_{j}-m_{i-1}}\right)e^{m_{i-1}-m_{i}} + e^{x_{i}-m_{i}}$$

$$= d_{i-1}^{\prime}e^{m_{i-1}-m_{i}} + e^{x_{i}-m_{i}}$$

这种递推形式仅依赖于 $m_{i}$ 和 $m_{i-1}$，因此我们可以在同一个循环中同时计算 $m_{j}$ 和 $d_{j}^{\prime}$：

#### 算法：2-pass Online Softmax**

1. 遍历 $i \leftarrow 1$ 到 $N$：
   - $m_{i} \leftarrow \max(m_{i-1}, x_{i})$
   - $d_{i}^{\prime} \leftarrow d_{i-1}^{\prime}e^{m_{i-1}-m_{i}} + e^{x_{i}-m_{i}}$

2. 遍历 $i \leftarrow 1$ 到 $N$：
   - $a_{i} \leftarrow \frac{e^{x_{i}-m_{N}}}{d_{N}^{\prime}}$



这是 Online Softmax 论文 [3] 中提出的算法。**Online Softmax** 的核心思想是在扫描数据的过程中动态地更新全局统计量（最大值和分母），从而将三轮遍历减少为两轮。

当我们引入一个新元素 $x_{new}$ 时，如果它比当前的最大值 $m_{old}$ 还要大，我们就需要更新分母。更新公式如下：

- 新的最大值：$m_{new} = \max(m_{old}, x_{new})$
- 更新后的分母：$d_{new} = d_{old} \cdot e^{m_{old} - m_{new}} + e^{x_{new} - m_{new}}$

```python
import numpy as np

def softmax_online(input_array):
    n = len(input_array)
    output = np.zeros(n, dtype=float)

    # Initialize running maximum with first element
    m = input_array[0]
    # Running sum (starts with e^(x_0 - m_0) = 1.0)
    d = 1.0

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

通过这种方式，我们可以在第一轮遍历中同时计算出全局最大值 $m$ 和最终的分母 $d$。第二轮遍历则直接计算输出结果。这种在线更新的洞察力正是通往 **FlashAttention** 的原动力。

## 4. FlashAttention：迈向单轮遍历

然而，Online-Softmax 仍然需要两次遍历才能完成 Softmax 计算。我们能否将遍历次数减少到 1 次，以最小化全局 I/O？

遗憾的是，对于 Softmax 来说，答案是"不"。但在自注意力中，我们的最终目标不是注意力分数矩阵 $A$，而是等于 $A \times V$ 的 $O$ 矩阵。我们能否为 $O$ 找到一个单次遍历（One-pass）的递推形式？

让我们尝试将自注意力计算的第 $k$ 行（各行的计算是独立的，为简单起见，我们仅解释一行的计算）公式化为递推算法：

**算法：多轮自注意力**

- **符号说明**：
  - $Q[k,:]$：$Q$ 矩阵的第 $k$ 行向量
  - $K^{T}[:,i]$：$K^{T}$ 矩阵的第 $i$ 列向量
  - $O[k,:]$：输出 $O$ 矩阵的第 $k$ 行
  - $V[i,:]$：$V$ 矩阵的第 $i$ 行
  - $\{o_{i}\} : \sum_{j=1}^{i}a_{j}V[j,:]$，存储局部聚合结果 $A[k,:i] \times V[:i,:]$ 的行向量

- **主体**：
  1. 遍历 $i \leftarrow 1$ 到 $N$：
     - $x_{i} \leftarrow Q[k,:]K^{T}[:,i]$
     - $m_{i} \leftarrow \max(m_{i-1}, x_{i})$
     - $d_{i}^{\prime} \leftarrow d_{i-1}^{\prime}e^{m_{i-1}-m_{i}} + e^{x_{i}-m_{i}}$

  2. 遍历 $i \leftarrow 1$ 到 $N$：
     - $a_{i} \leftarrow \frac{e^{x_{i}-m_{N}}}{d_{N}^{\prime}}$
     - $o_{i} \leftarrow o_{i-1} + a_{i}V[i,:]$

  3. $O[k,:] \leftarrow o_{N}$

我们将方程 12 中的 $a_{i}$ 替换为方程 11 中的定义：

$$o_{i} := \sum_{j=1}^{i} \left( \frac{e^{x_{j}-m_{N}}}{d_{N}^{\prime}} V[j,:] \right)$$

这仍然依赖于 $m_{N}$ 和 $d_{N}$，它们在之前的循环完成前无法确定。但我们可以再次使用第 3 节中的"替代（Surrogate）"技巧，创建一个替代序列 $o^{\prime}$：

$$o_{i}^{\prime} := \sum_{j=1}^{i} \left( \frac{e^{x_{j}-m_{i}}}{d_{i}^{\prime}} V[j,:] \right)$$

$o$ 和 $o^{\prime}$ 的第 $N$ 个元素是相同的：$o_{N}^{\prime} = o_{N}$，并且我们可以找到 $o_{i}^{\prime}$ 和 $o_{i-1}^{\prime}$ 之间的递推关系：

$$o_{i}^{\prime} = \sum_{j=1}^{i} \frac{e^{x_{j}-m_{i}}}{d_{i}^{\prime}} V[j,:]$$

$$= \left( \sum_{j=1}^{i-1} \frac{e^{x_{j}-m_{i}}}{d_{i}^{\prime}} V[j,:] \right) + \frac{e^{x_{i}-m_{i}}}{d_{i}^{\prime}} V[i,:]$$

$$= \left( \sum_{j=1}^{i-1} \frac{e^{x_{j}-m_{i-1}}}{d_{i-1}^{\prime}} V[j,:] \right) \frac{d_{i-1}^{\prime}e^{m_{i-1}-m_{i}}}{d_{i}^{\prime}} + \frac{e^{x_{i}-m_{i}}}{d_{i}^{\prime}} V[i,:]$$

$$= o_{i-1}^{\prime} \frac{d_{i-1}^{\prime}e^{m_{i-1}-m_{i}}}{d_{i}^{\prime}} + \frac{e^{x_{i}-m_{i}}}{d_{i}^{\prime}} V[i,:]$$

这仅取决于 $d_{i}^{\prime}, d_{i-1}^{\prime}, m_{i}, m_{i-1}$ 和 $x_{i}$，因此我们可以将自注意力的所有计算融合在一个循环中：

**算法：FlashAttention**

遍历 $i \leftarrow 1$ 到 $N$：

1. $x_{i} \leftarrow Q[k,:]K^{T}[:,i]$
2. $m_{i} \leftarrow \max(m_{i-1}, x_{i})$
3. $d_{i}^{\prime} \leftarrow d_{i-1}^{\prime}e^{m_{i-1}-m_{i}} + e^{x_{i}-m_{i}}$
4. $o_{i}^{\prime} \leftarrow o_{i-1}^{\prime} \frac{d_{i-1}^{\prime}e^{m_{i-1}-m_{i}}}{d_{i}^{\prime}} + \frac{e^{x_{i}-m_{i}}}{d_{i}^{\prime}} V[i,:]$

最后，$O[k,:] \leftarrow o_{N}^{\prime}$。状态 $x_{i}, m_{i}, d_{i}^{\prime}$ 和 $o_{i}^{\prime}$ 的内存占用很小，可以轻松放入 GPU 的共享内存中。由于该算法中的所有操作都是结合的，因此它与分块技术兼容。如果我们按块计算状态，该算法可以表示如下：

**算法：FlashAttention (分块版)**

- **新符号说明**：
  - $b$：分块的大小
  - \#tiles：每一行中的分块数量，$N = b \times \#tiles$
  - $x_{i}$：存储第 $i$ 个分块 $[(i-1)b : i \cdot b]$ 的 $Q[k]K^{T}$ 值的向量
  - $m_{i}^{(local)}$：$x_{i}$ 内部的局部最大值
- 主体：
  遍历 $i \leftarrow 1$ 到 \#tiles：
  1. $x_{i} \leftarrow Q[k,:]K^{T}[:, (i-1)b : ib]$
  2. $m_{i}^{(local)} = \max_{j=1}^{b}(x_{i}[j])$
  3. $m_{i} \leftarrow \max(m_{i-1}, m_{i}^{(local)})$
  4. $d_{i}^{\prime} \leftarrow d_{i-1}^{\prime}e^{m_{i-1}-m_{i}} + \sum_{j=1}^{b}e^{x_{i}[j]-m_{i}}$
  5. $o_{i}^{\prime} \leftarrow o_{i-1}^{\prime} \frac{d_{i-1}^{\prime}e^{m_{i-1}-m_{i}}}{d_{i}^{\prime}} + \sum_{j=1}^{b} \frac{e^{x_{i}[j]-m_{i}}}{d_{i}^{\prime}} V[j+(i-1)b, :]$





FlashAttention 进一步将上述逻辑推向极致。它不仅使用了分块（Tiling）技术，还将注意力机制的三个步骤（$QK^T$ 乘法、Softmax、以及与 $V$ 相乘）融合进了一个单一的 CUDA 算子（Kernel）中。

在 FlashAttention 中：

1. 我们按块读取 $Q, K, V$。
2. 在 SRAM 中计算 $QK^T$ 的局部块。
3. 使用 Online Softmax 的逻辑，通过存储局部最大值和分母，动态地更新输出矩阵 $O$。
4. 最终，我们只需要将 $O$ 写回主存。

```python
import numpy as np

def flash_attention(Q, K, V, k):
    """
    Parameters:
    Q: Query matrix
    K: Key matrix (transposed in the computation)
    V: Value matrix
    k: Row index for query

    Returns:
    Output vector O[k,:] after processing - equivalent to softmax(Q[k,:] @ K) @ V
    """
    N = K.shape[1]  # Get the dimension from K matrix

    # Initialize variables
    m_i_minus_1 = float('-inf')  # Initial value for m_{i-1}
    d_i_minus_1 = 0.0  # Initial value for d'_{i-1}
    o_i_minus_1 = np.zeros_like(V[0, :])  # Initial value for o'_{i-1}

    for i in range(N):
        # Calculate x_i using the k-th row of Q and i-th column of K^T
        x_i = np.dot(Q[k, :], K[:, i])

        # Update max value
        m_i = max(m_i_minus_1, x_i)

        # Calculate d'_i
        d_i = d_i_minus_1 * np.exp(m_i_minus_1 - m_i) + np.exp(x_i - m_i)

        # Calculate o'_i
        o_i = (o_i_minus_1 * d_i_minus_1 * np.exp(m_i_minus_1 - m_i) / d_i) + (np.exp(x_i - m_i) / d_i) * V[i, :]

        # Update previous values for next iteration
        m_i_minus_1 = m_i
        d_i_minus_1 = d_i
        o_i_minus_1 = o_i

    # The result is o'_N
    return o_i_minus_1
```

通过这样的在线融合机制，可以将原本至少 3 遍的注意力计算降到 **1 遍**，并且每个 Token 的注意力都能在 GPU 更小的片上内存中计算，从而避免序列长度二次方级别的内存开销。

这种方法避免了在主存中存储巨大的中间矩阵（如 $N \times N$ 的注意力分数矩阵），大幅降低了显存带宽的消耗，使得在有限的显存下处理超长序列（如 100k 以上的上下文）成为可能。

### 7. 总结

- **传统 Softmax**：需要三轮遍历，是显存受限的瓶颈。
- **Online Softmax**：通过动态更新公式，将遍历减少到两轮，并允许分块计算。
- **FlashAttention**：利用 Online Softmax 的特性，将整个注意力计算融合，实现了单轮遍历的效果，极大地提升了 Transformer 的训练和推理速度。
