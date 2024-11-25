# Direct Preference Optimization （DPO）

## 总结

RLHF 的训练流程复杂而且不稳定；首先我们训练一个奖励模型以反应人类偏好，然后利用强化学习微调语言模型来最大化估计奖励，在过程中约束微调的模型不可以偏离原始模型太多。

本文介绍了一个可用于RLHF的参数化隐式奖励模型，让我们可以仅用自监督方法解决标准RLHF问题。

<img src="https://miro.medium.com/v2/resize:fit:700/1*GiEF7F3n-1TlL7_HRJD_OA.png" alt="img" style="zoom:150%;" />

<img src="https://miro.medium.com/v2/resize:fit:648/1*f1LfCLncMIyhQiorWSKf7A.png" alt="img" style="zoom:150%;" />

DPO优化人类偏好，同时不需要使用强化学习。相比与PPO，DPO直接优化最能满足偏好的策略，使用简单的分类目标，拟合一个隐式奖励模型，其相应的最优策略可以以封闭形式提取。DPO的性能与现有的RLHF算法相似或更好，而且有效降低了从人类偏好中训练更多语言模型的障碍；与标准的RL设置不同，DPO确定了语言模型策略和奖励函数之间的映射，这使得可以使用简单的binary cross-entropy loss 直接训练语言模型以满足人类的偏好，而无需使用强化学习。

## **摘要**

尽管大规模非监督语言模型（LMs）学习了广泛的世界知识和一些推理技能，但由于其训练的完全非监督性质，精确控制它们的行为很困难。现有方法通过收集人类对模型生成内容的相对质量标签，并通过人类反馈的强化学习（RLHF）来微调非监督LM，使其与这些偏好对齐。然而，RLHF是一个复杂且经常不稳定的过程，首先拟合一个反映人类偏好的奖励模型，然后使用强化学习微调大型非监督LM以最大化这个估计的奖励，同时不会偏离原始模型太远。在本文中，我们引入了RLHF中奖励模型的新参数化，它能够以封闭形式提取相应的最优策略，允许我们仅使用简单的分类损失解决标准RLHF问题。我们提出的算法，称为直接偏好优化（DPO），稳定、高效且计算成本低，消除了在微调期间从LM采样或执行大量超参数调整的需要。我们的实验表明，DPO可以与现有方法一样好或更好地微调LM以符合人类偏好。值得注意的是，使用DPO进行微调在控制生成内容的情感方面超过了基于PPO的RLHF，并且在摘要和单轮对话中的响应质量方面匹配或提高了，同时实现起来更简单，训练也更简便。

### **引言**

在非常大的数据集上训练的大型非监督语言模型（LMs）获得了惊人的能力[11, 7, 40, 8]。然而，这些模型是在人类生成的数据上训练的，人类有着广泛的目标、优先级和技能集。这些目标和技能集中的某些可能并不适合模仿；例如，我们可能希望我们的AI编码助手理解常见的编程错误以便纠正它们，然而，在生成代码时，我们希望我们的模型偏向于其训练数据中存在的（可能是罕见的）高质量编码能力。同样，我们可能希望我们的语言模型意识到50%的人相信的常见误解，但我们当然不希望模型在查询它的50%的情况下声称这个误解是真的！换句话说，从模型非常广泛的知识和能力中选择模型的期望响应和行为对于构建安全、高效和可控的AI系统至关重要[26]。虽然现有方法通常使用强化学习（RL）来引导LMs与人类偏好对齐，但我们将展示RL基础目标可以使用简单的二元交叉熵目标精确优化，从而大大简化偏好学习流程。

在高层次上，现有方法使用代表人类认为安全和有帮助的行为类型的策划人类偏好集，将期望的行为灌输到语言模型中。这种偏好学习阶段发生在对大型文本数据集进行大规模非监督预训练的初始阶段之后。虽然偏好学习的最直接的方法是在人类展示的高质量响应上进行监督微调，但最成功的方法是使用人类（或AI）反馈的强化学习（RLHF/RLAIF；[12, 2]）。RLHF方法拟合一个奖励模型到人类偏好的数据集上，然后使用RL来优化语言模型策略，以产生分配高奖励的响应，而不会过度偏离原始模型。虽然RLHF产生了具有令人印象深刻的对话和编码能力的模型，但RLHF流程比监督学习要复杂得多，涉及训练多个LLMs并在训练循环中从LM策略中采样，导致显著的计算成本。

在本文中，我们将展示如何直接优化语言模型以遵守人类偏好，而无需显式奖励建模或强化学习。我们提出了直接偏好优化（DPO），这是一种算法，它隐式地优化与现有RLHF算法相同的目标（奖励最大化和KL散度约束），但实现简单，训练直接。直观地说，DPO更新增加了首选响应相对于非首选响应的相对对数概率，但它包含了一个动态的、每个示例的重要性权重，可以防止我们发现使用简单概率比率目标时发生的模型退化。与现有算法一样，DPO依赖于一个理论偏好模型（如Bradley-Terry模型；[5]），该模型衡量给定奖励函数与经验偏好数据的对齐程度。然而，虽然现有方法使用偏好模型来定义偏好损失以训练奖励模型，然后训练一个策略来优化学习到的奖励模型，但DPO使用变量变化直接将偏好损失定义为策略的函数。因此，给定一个人类对模型响应的偏好数据集，DPO可以使用简单的二元交叉熵目标优化策略，为偏好数据拟合一个隐式奖励函数的最优策略。

我们的主要贡献是直接偏好优化（DPO），这是一种简单的无RL算法，用于根据偏好训练语言模型。我们的实验表明，DPO至少与现有方法一样有效，包括基于PPO的RLHF，用于在情感调节、摘要和对话等任务中从偏好中学习，使用的语言模型参数高达6B。

## **相关工作**

自监督语言模型的规模不断增加，它们学习完成一些任务，无论是零样本[31]还是少样本提示[6, 25, 11]。然而，通过在指令和人类编写的完成内容的数据集上进行微调，它们在下游任务上的表现和与用户意图的一致性可以显著提高[23, 36, 13, 39]。这种“指令调整”过程使大型语言模型（LLMs）能够推广到指令调整集之外的指令，并普遍提高了它们的可用性[13]。尽管指令调整取得了成功，但相对于专家演示，人类对响应质量的判断更容易收集，因此后续工作使用人类偏好的数据集对LLMs进行了微调，提高了翻译[18]、摘要[38, 49]、讲故事[49]和指令遵循[26, 32]的熟练程度。这些方法首先优化一个神经网络奖励函数，以与偏好数据集兼容，然后在偏好模型（如Bradley-Terry模型[5]）下使用强化学习算法（通常是REINFORCE[45]、近端策略优化（PPO；[37]）或变种[32]）来微调语言模型，以最大化给定的奖励。与人类反馈的指令遵循的LLMs的微调相关的方法，使用人类反馈生成具有目标属性（如安全性或无害性）的额外合成偏好数据[2]，仅使用人类对LLM注释的文本规范形式的弱监督。这些方法代表了两组工作的融合：一组是使用强化学习训练语言模型以实现各种目标的工作[33, 27, 46]，另一组是学习人类偏好的通用方法[12, 19]。尽管使用相对人类偏好具有吸引力，但使用强化学习对大型语言模型进行微调仍然是一个主要的实际挑战；这项工作提供了一种理论上合理的方法，无需RL即可优化相对偏好。

在语言环境之外，从偏好中学习策略已在乐队和强化学习设置中进行了研究，并提出了几种方法。使用偏好或行动排名而非奖励的上下文乐队学习被称为上下文对弈乐队（CDB；[48, 14]）。在没有绝对奖励的情况下，CDB的理论分析用冯·诺依曼胜者（von Neumann winner）替换了最优策略的概念，即预期胜率至少为50%的策略[14]。然而，在CDB设置中，偏好标签是在线给出的，而在从人类偏好中学习时，我们通常从固定的一批离线偏好注释的动作对中学习[47]。类似地，基于偏好的强化学习（PbRL）从由未知的“评分”函数生成的二元偏好中学习，而不是奖励[9, 35]。存在多种PbRL算法，包括可以重用离线偏好数据的方法，但通常涉及首先明确估计潜在的评分函数（即奖励模型），然后优化它[16, 9, 12, 34, 19]。我们提出了一种单阶段策略学习方法，直接优化策略以满足偏好。

## 预备知识

我们回顾了Ziegler等人（以及后来的[38, 1, 26]）中描述的RLHF流程。它通常包括三个阶段：1）监督微调（SFT）；2）偏好采样和奖励学习；3）RL优化。

### SFT:

RLHF通常以对感兴趣下游任务（对话、摘要等）的高质量数据进行监督学习来微调预训练的LM开始，以获得模型 $ \pi_{SFT} $。

### 奖励建模阶段:

在第二阶段，使用提示$ x $ 提示SFT模型产生一对答案$ (y_1, y_2) \sim \pi_{SFT}(y | x) $。然后，这些答案被呈现给人类标注者，他们表达对一个答案的偏好，表示为$ y_w \succ y_l | x $，其中$ y_w $ 和$ y_l $ 分别表示$ (y_1, y_2) $ 中的首选和非首选回答。

假设偏好是由某些潜在的奖励模型$ r^*(y, x) $ 生成的，我们无法访问该模型。有多种方法用于建模偏好，Bradley-Terry (BT) [5]模型是一个流行的选择（尽管如果我们可以获得几个排名答案，更一般的Plackett-Luce排名模型[30, 21]也与该框架兼容）。BT 模型规定，人类偏好分布 $ p^* $ 可以写成：
$$
p^*(y_1 \succ y_2 | x) = \frac{\exp(r^*(x, y_1))}{\exp(r^*(x, y_1)) + \exp(r^*(x, y_2))}
$$


假设我们可以访问来自 $ p^* $  的比较静态数据集$ D = \{x(i), y(i)_w, y(i)_l\}^{N}_{i=1} $ 的样本，我们可以通过最大似然估计参数化奖励模型$ r_\phi(x, y) $。将问题框架为二元分类，我们有负对数似然损失：

$$
L_{R}(r_\phi, D) = -\mathbb{E}(x,y_w,y_l) \sim D \left[ \log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l)) \right]
$$
其中$ \sigma $ 是逻辑函数。在LMs的背景下，网络$ r_\phi(x, y) $ 通常从SFT模型$ \pi_{SFT}(y | x) $ 初始化，顶层增加一个线性层，产生单个标量预测奖励值[49]。为了确保具有较低方差的奖励函数，以前的工作对奖励进行了归一化，使得$ \mathbb{E}_{x,y} \sim D [r_\phi(x, y)] = 0 $ 对所有$ x $。

### RL微调阶段:

在RL阶段，我们使用学习到的奖励函数为语言模型提供反馈。具体来说，我们构建了以下优化问题：

$$
\max_{\pi} \mathbb{E}_{x \sim D, y \sim \pi} \left[ r_\phi(x, y) - \beta D_{KL} \pi(y | x) || \pi_{ref}(y | x) \right]
$$
其中$ \beta $ 是一个控制基础模型参考策略$ \pi_{ref} $（即初始SFT模型$ \pi_{SFT} $）的参数。在实践中，语言模型策略$ \pi_\theta $ 也初始化为$ \pi_{SFT} $。增加的约束很重要，因为它防止模型偏离奖励模型准确的分布，同时保持生成多样性并防止模式崩溃到单一高奖励答案。由于语言生成的离散性，这个目标不是可微分的，通常使用强化学习进行优化。标准方法[49, 38, 1, 26]是构建奖励函数$ r(x, y) = r_\phi(x, y) - \beta(\log \pi_\theta(y | x) - \log \pi_{ref}(y | x)) $，并使用PPO [37]进行最大化。

## 直接偏好优化 (Direct Preference Optimization)

在大规模问题上应用强化学习算法，如微调语言模型，面临挑战。我们的目标是派生一种简单的方法，直接使用偏好进行策略优化。与以往的RLHF方法不同，这些方法首先学习奖励模型，然后通过强化学习优化，我们的方法利用奖励模型的特定参数化，使我们能够在没有强化学习训练循环的情况下，以封闭形式提取其最优策略。我们的关键洞察是利用从奖励函数到最优策略的分析映射，这使我们能够将奖励函数上的损失函数转换为策略上的损失函数。这种变量替换方法避免了拟合一个显式的独立奖励模型，同时仍然在现有的人类偏好模型（如Bradley-Terry模型）下进行优化。本质上，策略网络代表了语言模型和（隐式的）奖励模型。

### 推导DPO目标

我们从与之前工作相同的RL目标开始，即在一般奖励函数 $ r $ 下的方程(3)。按照之前的工作[29, 28, 17, 15]，可以简单地展示方程(3)中的KL约束奖励最大化目标的最优解采用以下形式：

$$
\pi^*(y | x) = \frac{1}{Z(r)} \pi_{ref}(y | x) \exp\left(\frac{1}{\beta} r(x, y)\right)
$$

其中 $ Z(r) = \sum_{y} \pi_{ref}(y | x) \exp\left(\frac{1}{\beta} r(x, y)\right) $ 是划分函数。完整的推导见附录A.1。即使我们使用真实奖励函数 $ r^* $ 的MLE估计 $ \hat{r} $，估计划分函数 $ Z(\alpha) $ 仍然是昂贵的[17, 15]，这使得这种表示在实践中难以利用。然而，我们可以重新排列方程(4)，以用其对应的最优策略 $ \pi^* $、参考策略 $ \pi_{ref} $ 和未知的划分函数 $ Z(-) $ 来表达奖励函数。具体来说，我们首先对方程(4)的两边取对数，然后通过一些代数运算我们得到：

$$
r(x, y) = \beta \log \frac{\pi^*(y | x)}{\pi_{ref}(y | x)} + \beta \log Z(r)
$$


我们可以将这个重新参数化应用于真实奖励 $ r^* $ 和相应的最优模型 $ \pi^* $。幸运的是，Bradley-Terry模型仅依赖于两个完成之间的奖励差异，即 $ p^*(y_1 > y_2 | x) = \sigma(r^*(x, y_1) - r^*(x, y_2)) $。将方程(5)中的重新参数化代入偏好模型方程(1)，划分函数会被抵消，我们可以用仅最优策略 $ \pi^* $ 和参考策略 $ \pi_{ref} $ 来表达人类偏好概率。因此，在Bradley-Terry模型下，最优的RLHF策略 $ \pi^* $ 满足偏好模型：

$$
p^*(y_1 > y_2 | x) = \frac{1}{1 + \exp\left(\beta \log \frac{\pi^*(y_2 | x)}{\pi_{ref}(y_2 | x)} - \beta \log \frac{\pi^*(y_1 | x)}{\pi_{ref}(y_1 | x)}\right)}
$$

推导见附录A.2。虽然方程(6)使用了Bradley-Terry模型，我们也可以在更一般的Plackett-Luce模型[30, 21]下推导出类似的表达式，如附录A.3所示。

现在我们已经用最优策略而非奖励模型来表示人类偏好数据的概率，我们可以为参数化策略 $ \pi_{\theta} $ 制定一个最大似然目标。类似于奖励建模方法（即方程2），我们的策略目标变为：

$$
L_{DPO}(\pi_{\theta}; \pi_{ref}) = -\mathbb{E}_{(x,y_w,y_l) \sim D} \left[ \log \sigma \left( \beta \log \frac{\pi_{\theta}(y_w | x)}{\pi_{ref}(y_w | x)} - \beta \log \frac{\pi_{\theta}(y_l | x)}{\pi_{ref}(y_l | x)} \right) \right]
$$

这样，我们使用替代参数化来拟合一个隐式奖励，其最优策略简单地是 $ \pi_{\theta} $。此外，由于我们的过程等同于拟合一个重新参数化的Bradley-Terry模型，它享有某些理论属性，比如在偏好数据分布的适当假设下的一致性[4]。在第5节中，我们进一步讨论了DPO与其他工作的理论与属性关系。

### DPO更新的作用是什么？

为了深入理解DPO，分析损失函数 $ L_{DPO} $ 的梯度是有用的。梯度相对于参数 $ \theta $ 可以写成：

$$
\nabla_{\theta} L_{DPO}(\pi_{\theta}; \pi_{ref}) = -\beta \mathbb{E}_{(x,y_w,y_l) \sim D} \left[ \sigma\left( \hat{r}_{\theta}(x, y_l) - \hat{r}_{\theta}(x, y_w) \right) \left( \nabla_{\theta} \log \pi(y_w | x) - \nabla_{\theta} \log \pi(y_l | x) \right) \right]
$$

其中 $ \hat{r}_{\theta}(x, y) = \beta \log \frac{\pi_{\theta}(y | x)}{\pi_{ref}(y | x)} $ 是由语言模型 $ \pi_{\theta} $ 和参考模型 $ \pi_{ref} $ 隐式定义的奖励（更多内容见第5节）。

当 $ \hat{r}_{\theta}(x, y)  >  \hat{r}_{\theta}(x, y) $,  算式 $\sigma \left( \hat{r}_{\theta}(x, y_l) - \hat{r}_{\theta}(x, y_w) \right) $ 越接近1，反之接近0，若两者趋近则该权重接近0.5。

直观上，损失函数 $ L_{DPO} $ 的梯度增加了首选回答 $ y_w $ 的可能性，并减少了非首选回答 $ y_l $ 的可能性。

重要的是，示例根据隐式奖励模型 $ \hat{r}_{\theta} $ 对非首选完成的评估有多高，按 $ \beta $ 缩放，即隐式奖励模型对完成的排序有多错误，考虑了KL约束的强度。我们的实验表明这种加权的重要性，因为没有加权系数的简单版本的这种方法可能会导致语言模型退化（附录表3）。

### DPO概述

一般DPO流程如下：
1. 对于每个提示 $ x $，从参考模型 $ \pi_{ref}(- | x) $ 中采样完成 $ y_1, y_2 $，用人类偏好标注以构建离线偏好数据集 $ D = \{(x(i), y(i)_w, y(i)_l)\}^{N}_{i=1} $。
2. 优化语言模型 $ \pi_{\theta} $ 以最小化给定 $ \pi_{ref} $ 和 $ D $ 以及期望的 $ \beta $ 的 $ L_{DPO} $。

在实践中，人们希望重用公开可用的偏好数据集，而不是生成样本和收集人类偏好。由于偏好数据集是使用 $ \pi_{SFT} $ 采样的，我们通常初始化 $ \pi_{ref} = \pi_{SFT} $。然而，当 $ \pi_{SFT} $ 不可用时，我们通过最大化首选完成 $ (x, y_w) $ 的似然来初始化 $ \pi_{ref} $，即 $ \pi_{ref} = \arg \max_{\pi} \mathbb{E}_{x,y_w \sim D} [\log \pi(y_w | x)] $。这个过程有助于减少真实参考分布（不可用）和DPO使用的 $ \pi_{ref} $ 之间的分布偏移。有关实现和超参数的更多细节可以在附录B中找到。

## DPO的理论分析

在本节中，我们进一步解释DPO方法，提供理论支持，并将DPO的优势与用于RLHF的标准Actor-Critic算法（如PPO）的问题联系起来。

### 语言模型实际上是一个奖励模型

DPO能够通过单一的最大似然目标来学习策略，而无需拟合显式奖励和执行RL。注意优化目标方程(5)等同于具有奖励参数化的Bradley-Terry模型，其中 $ r^*(x, y) = \beta \log \frac{\pi^*_{\theta}(y|x)}{\pi_{ref}(y|x)} $，我们优化我们的参数模型 $ \pi_{\theta} $，等同于在变量变换下的方程(2)中的奖励模型优化。在本节中，我们将建立这种重新参数化背后的理论，展示它不会限制学习到的奖励模型的类别，并允许精确恢复最优策略。我们首先通过定义奖励函数之间的等价关系来开始。

**定义1**. 两个奖励函数 $ r(x, y) $ 和 $ r'(x, y) $ 是等价的，当且仅当，对于某个函数 $ f $,  满足 $ r'(x, y) = r(x, y) + f(x) $ 。

很容易看出这确实是一个等价关系，它将奖励函数的集合划分为类别。我们可以陈述以下两个引理：

**引理1**. 在Plackett-Luce偏好框架下，来自同一类别的两个奖励函数会诱导相同的偏好分布。

**引理2**. 来自同一等价类的两个奖励函数在受约束的RL问题下诱导相同的最优策略。

证明是直接的，我们将其推迟到附录A.5。第一个引理是与Plackett-Luce模型家族相关的一个众所周知的欠规范问题。由于这种欠规范，我们通常必须施加额外的可识别性约束，以实现方程(2)中MLE估计的任何保证。第二个引理表明，来自同一类别的所有奖励函数产生相同的最优策略，因此对于我们的最终目标，我们只对从最优类别中恢复任意奖励函数感兴趣。我们在附录A.6中证明了以下定理：

**定理1**. 在温和的假设下，所有与Plackett-Luce（特别是Bradley-Terry）模型一致的奖励类别都可以使用重新参数化 $ r(x, y) = \beta \log \frac{\pi(y|x)}{\pi_{ref}(y|x)} $ 表示，其中模型 $ \pi(y | x) $ 和给定的参考模型 $ \pi_{ref}(y | x) $。

证明概要。考虑任何奖励函数 $ r(x, y) $，它诱导了一个相应的最优模型 $ \pi_r(y | x) $，由方程(4)指定。我们定义投影 $ f $ 为：

$$
f(r; \pi_{ref}, \beta)(x, y) = r(x, y) - \beta \log \sum_y \pi_{ref}(y | x) \exp\left(\frac{1}{\beta} r(x, y)\right)
$$


操作符 $ f $ 简单地使用 $ \pi_r $ 的划分函数的对数对奖励函数进行归一化。由于添加的归一化项仅是 $ x $ 的函数，$ f(r; \pi_{ref}, \beta)(x, y) $ 是 $ r(x, y) $ 的等价类中的奖励函数。最后，用方程(5)右侧替换 $ r $（这对任何奖励函数都成立），我们有 $ f(r; \pi_{ref}, \beta)(x, y) = \beta \log \frac{\pi_r(y|x)}{\pi_{ref}(y|x)} $。也就是说，投影 $ f $ 产生了一个等价类 $ r $ 的成员，具有所需的形式，我们从所提出的重新参数化中没有失去任何一般性。

我们可以将定理1视为指定了DPO重新参数化在每个等价类中选择的确切奖励函数，即满足：

$$
 \sum_y \pi_{ref}(y | x) \exp\left(\frac{1}{\beta} r(x, y)\right) = \pi(y|x)
$$
的奖励函数，使用定理1重新参数化。也就是说，$ \pi(y | x) $ 是一个有效的概率分布（概率是正的并且它们的和为1）。然而，根据方程(4)，我们可以看到方程(9)是奖励函数 $ r(x, y) $ 诱导的最优策略的划分函数。DPO算法的关键洞察是我们可以对Plackett-Luce（特别是Bradley-Terry）偏好模型家族施加某些约束，以保留可表示奖励模型的类别，但明确使方程(4)中的最优策略对所有提示 $ x $ 分析上可处理。

### 不稳定性的Actor-Critic算法

我们还可以使用我们的框架来诊断RLHF中使用的标准Actor-Critic算法（如PPO）的不稳定性。我们遵循RLHF流程并专注于第3节中概述的RL微调步骤。我们可以将受约束的RL问题与控制作为推断框架[20]联系起来。我们假设一个参数化模型 $ \pi_{\theta}(y | x) $ 并最小化 $ D_{KL}[\pi_{\theta}(y|x) || \pi^*(y | x)] $，其中 $ \pi^* $ 是由奖励函数 $ r_\phi(y, x) $ 诱导的最优策略。通过一些代数，这导致优化目标：

$$
 \max_{\pi_{\theta}} \mathbb{E}_{\pi_{\theta}(y|x)} \left[ r_\phi(x, y) - \beta \log \frac{\pi_{ref}(y | x) \exp\left(\frac{1}{\beta} r_\phi(x, y)\right)}{Z(x)} - \beta \log \frac{\pi_{\theta}(y | x)}{\pi_{ref}(y | x)} \right]
$$
这是之前的作品[49, 38, 1, 26]使用DPO等价奖励对奖励类别优化的相同目标。在这个设置中，我们可以将 $ f(r_\phi, \pi_{ref}, \beta) $ 中的归一化项解释为参考策略 $ \pi_{ref} $ 的软价值函数。虽然这个项不影响最优解，但没有它，目标的策略梯度可能有高方差，使学习不稳定。我们可以使用学习的价值函数来适应归一化项，但优化它也可能很困难。或者，以前的工作使用人类完成的基线来归一化奖励，基本上是对归一化项的单次蒙特卡洛估计。相比之下，DPO重新参数化产生了一个不需要任何基线的奖励函数。

## 实验

实验聚焦在两个面向:

1. DPO如何有效的去权衡，最大化奖励和最小化$\pi_{\theta}$ 和$\pi_{ref}$ 的KL散度
2. 评估DPO 在更大模型和更困难的RLHF 任务，包括总结和对话。

我们发现在几乎不调整超参数的状况下，DPO的表现等同或超过基线(PPO-based RLHF)。

### 可控情绪生成

![img](https://media.githubusercontent.com/media/p208p2002/blog/main/public/docs/dpo/exp1.png)

- DPO: 论文方法。
- Unlikelihood: 简单的最大化$y_w$和最小化$y_l$的概似。
- PPO: 使用*奖励函数*去学习偏好资料。
- PPO-GT: 使用真实答案(ground truth) 的*奖励函数*(仅在可空情绪生成可取得)去学习。
- Preferred-FT: 在$y_w$上进行监督式学习的微调。

输入是IMDb资料集中的电影评论，*策略*必须产生正面情绪的回应。为了控制评估，我们针对这个实验用预训练的情绪分类器产生了偏好对$y_w$和$y_l$。实验显示DPO 能在获得较高的奖励下，同时保持较低的KL。

### 摘要生成

![img](https://media.githubusercontent.com/media/p208p2002/blog/main/public/docs/dpo/exp2.png)

- PPO: 使用*奖励函数*去学习偏好资料。
- SFT: 使用资料集的资料进行自监督微调。
- Preferred-FT: 使用来自SFT模型偏好的yw𝑦𝑤*y**w*进行自监督微调。
- GPT-J: Zero-shot prompting。
- Best of N: 从SFT 模型中取样N 个反应，并根据从偏好资料集中学习到的奖励函数传回得分最高的反应。

使用GPT-4作为裁判，与人类撰写总结进行1v1对战，结果显示DPO的表现最佳，并且在较高的温度(sampling temperature)下也保持较佳的表现。

### 单轮对话

在单轮对话中，我们在[Anthropic HH数据集](https://huggingface.co/datasets/Anthropic/hh-rlhf)的测试集子集上评估不同的方法。

![img](https://media.githubusercontent.com/media/p208p2002/blog/main/public/docs/dpo/exp3.png)

- Preferred-FT: 使用来自通用语言模型偏好的 $y_w$ 进行自监督微调。
- Best of N: 从Preferred-FT 模型中取样N 个反应，并根据从偏好资料集中学习到的奖励函数传回得分最高的反应。
- Pythia-2.8B: 2-shot prompting。

DPO是唯一能在Anthropic HH测试集中高效率计算并且改进偏好生成的方法。

## 总结

DPO的性能与现有的RLHF算法相似或更好，而且有效降低了从人类偏好中训练更多语言模型的障碍；与标准的RL设置不同，DPO确定了语言模型策略和奖励函数之间的映射，这使得可以使用简单的binary cross-entropy loss 直接训练语言模型以满足人类的偏好，而无需使用强化学习。

## 附录 A：数学推导

在这部分附录中，我们将提供DPO目标的详细数学推导。

### A.1 推导KL约束奖励最大化目标的最优解

在这部分附录中，我们将推导方程(4)。类似于方程(3)，我们优化以下目标：

$$
\max_{\pi} \mathbb{E}_{x \sim D, y \sim \pi} \left[ r(x, y) - \beta D_{KL} \pi(y | x) || \pi_{ref}(y | x) \right]
$$

在任何奖励函数 $ r(x, y) $、参考模型 $ \pi_{ref} $ 和一个一般非参数策略类下，我们现在有：

$$
\max_{\pi} \mathbb{E}_{x \sim D} \mathbb{E}_{y \sim \pi(y|x)} \left[ r(x, y) - \beta \log \frac{\pi(y|x)}{\pi_{ref}(y|x)} \right]
$$

$$
 = \min_{\pi} \mathbb{E}_{x \sim D} \mathbb{E}_{y \sim \pi(y|x)} \left[ \log \frac{\pi(y|x)}{\pi_{ref}(y|x)} - \frac{1}{\beta} r(x, y) \right]
$$

$$
= \min_{\pi} \mathbb{E}_{x \sim D} \left[ \log \frac{\pi(y|x)}{\pi_{ref}(y|x)} - \log Z(x) - \frac{1}{\beta} r(x, y) \right]
$$


其中我们有划分函数：
$$
Z(x) = \sum_y \pi_{ref}(y | x) \exp\left(\frac{1}{\beta} r(x, y)\right)
$$


注意划分函数仅是 $ x $ 和参考策略 $ \pi_{ref} $ 的函数，但不依赖于策略 $ \pi $。我们现在可以定义

$$
\pi^*(y | x) = \frac{1}{Z(x)} \pi_{ref}(y | x) \exp\left(\frac{1}{\beta} r(x, y)\right),
$$


这是一个有效的概率分布，因为对于所有 $ y $，$ \pi^*(y | x) \geq 0 $ 且 $ \sum_y \pi^*(y | x) = 1 $。由于 $ Z(x) $ 不是 $ y $ 的函数，我们可以重新组织方程(12)中的最终目标为：
$$
\min_{\pi} \mathbb{E}_{x \sim D} \left[ D_{KL}(\pi(y|x) || \pi^*(y|x)) - \log Z(x) \right]
$$
现在，由于 $ Z(x) $ 不依赖于 $ \pi $，最小值由最小化第一项KL散度的策略实现。Gibbs不等式告诉我们，KL散度最小化为0当且仅当两个分布相同时。因此我们有最优解：

$$
 \pi(y | x) = \pi^*(y | x) = \frac{1}{Z(x)} \pi_{ref}(y | x) \exp\left(\frac{1}{\beta} r(x, y)\right)
$$


对于所有 $ x \in D $。这完成了推导。

### A.2 在Bradley-Terry模型下推导DPO目标

在Bradley-Terry偏好模型下推导DPO目标是直接的，我们有
$$
 p^*(y_1 \succ y_2 | x) = \frac{\exp(r^*(x, y_1))}{\exp(r^*(x, y_1)) + \exp(r^*(x, y_2))}
$$


在第4节中，我们展示了我们可以将（不可用的）真实奖励通过其相应的最优策略表示：

$$
 r^*(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)
$$


将方程(17)代入方程(16)我们得到：

$$
p^*(y_1 \succ y_2 | x) = \frac{\exp(\beta \log \frac{\pi^*(y_1|x)}{\pi_{ref}(y_1|x)} + \beta \log Z(x))}{\exp(\beta \log \frac{\pi^*(y_1|x)}{\pi_{ref}(y_1|x)} + \beta \log Z(x)) + \exp(\beta \log \frac{\pi^*(y_2|x)}{\pi_{ref}(y_2|x)} + \beta \log Z(x))}
$$

$$
= \frac{1}{1 + \exp(\beta \log \frac{\pi^*(y_2|x)}{\pi_{ref}(y_2|x)} - \beta \log \frac{\pi^*(y_1|x)}{\pi_{ref}(y_1|x)})}
$$

$$
 = \sigma\left(\beta \log \frac{\pi^*(y_1|x)}{\pi_{ref}(y_1|x)} - \beta \log \frac{\pi^*(y_2|x)}{\pi_{ref}(y_2|x)}\right)
$$

最后一行是方程(7)中的每个实例损失。

### A.3 在Plackett-Luce模型下推导DPO目标

Plackett-Luce模型[30, 21]是对Bradley-Terry模型的一个概括，它适用于排名（而不仅仅是成对比较）。与Bradley-Terry模型类似，Plackett-Luce模型规定，当面对一组可能的选择时，人们倾向于选择那些与其潜在奖励函数值成比例的选项。在我们的情境中，当面对提示 $ x $ 和一组 $ K $ 个答案 $ y_1, ..., y_K $ 时，用户会输出一个排列 $ \tau : [K] \rightarrow [K] $，表示他们对答案的排序。Plackett-Luce模型规定：

$$
p^*(\tau | y_1, ..., y_K, x) = \frac{\sum_{k=1}^{K} \exp(r^*(x, y_{\tau(k)}))}{\sum_{j=1}^{K} \sum_{k=j}^{K} \exp(r^*(x, y_{\tau(k)}))}
$$

注意，当 $ K = 2 $ 时，方程(18)退化为Bradley-Terry模型。然而，对于一般的Plackett-Luce模型，我们仍然可以利用方程(5)的结果，用最优策略参数化奖励函数。与附录A.2类似，归一化常数 $ Z(x) $ 会被抵消，我们得到：

$$
p^*(\tau | y_1, ..., y_K, x) = \frac{\sum_{k=1}^{K} \exp\left(\beta \log \frac{\pi^*(y_{\tau(k)}|x)}{\pi_{ref}(y_{\tau(k)}|x)}\right)}{\sum_{j=1}^{K} \sum_{k=j}^{K} \exp\left(\beta \log \frac{\pi^*(y_{\tau(k)}|x)}{\pi_{ref}(y_{\tau(k)}|x)}\right)}
$$

类似于第4节的方法，如果我们可以获得一组提示和用户指定的排名的数据集 $ D = \{\tau(i), y(i)_1, ..., y(i)_K, x(i)\}^{N}_{i=1} $，我们可以使用参数化模型，并通过最大似然来优化这个目标：

$$
 L_{DPO}(\pi_{\theta}, \pi_{ref}) = -\mathbb{E}_{\tau,y_1,...,y_K,x \sim D} \left[ \log \frac{\sum_{k=1}^{K} \exp\left(\beta \log \frac{\pi_{\theta}(y_{\tau(k)}|x)}{\pi_{ref}(y_{\tau(k)}|x)}\right)}{\sum_{j=1}^{K} \sum_{k=j}^{K} \exp\left(\beta \log \frac{\pi_{\theta}(y_{\tau(k)}|x)}{\pi_{ref}(y_{\tau(k)}|x)}\right)} \right]
$$

这样，我们就可以利用Plackett-Luce模型来表达人类偏好数据的概率，并且以最大似然估计来优化参数化策略 $ \pi_{\theta} $。

### A.4 推导DPO目标的梯度

在本节中，我们推导DPO目标的梯度：

$$
 \nabla_{\theta} L_{DPO}(\pi_{\theta}; \pi_{ref}) = -\nabla_{\theta} \mathbb{E}(x,y_w,y_l) \sim D \left[ \log \sigma \left( \beta \log \frac{\pi_{\theta}(y_l|x)}{\pi_{ref}(y_l|x)} - \beta \log \frac{\pi_{\theta}(y_w|x)}{\pi_{ref}(y_w|x)} \right) \right]
$$

我们可以将方程右边重写为：

$$
\nabla_{\theta} L_{DPO}(\pi_{\theta}; \pi_{ref}) = -\mathbb{E}(x,y_w,y_l) \sim D \left[ \sigma'(u) \frac{\sigma(u)}{\nabla_{\theta} u} \right],
$$

其中 $ u = \beta \log \frac{\pi_{\theta}(y_l|x)}{\pi_{ref}(y_l|x)} - \beta \log \frac{\pi_{\theta}(y_w|x)}{\pi_{ref}(y_w|x)} $。

使用Sigmoid函数的性质 $ \sigma'(x) = \sigma(x)(1 - \sigma(x)) $ 和 $ \sigma(-x) = 1 - \sigma(x) $，我们得到最终梯度的形式：

$$
\nabla_{\theta} L_{DPO}(\pi_{\theta}; \pi_{ref}) = -\mathbb{E}(x,y_w,y_l) \sim D \left[ \beta \sigma \left( \beta \log \frac{\pi_{\theta}(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_{\theta}(y_l|x)}{\pi_{ref}(y_l|x)} \right) \left( \nabla_{\theta} \log \pi(y_w | x) - \nabla_{\theta} \log \pi(y_l | x) \right) \right]
$$

使用奖励替换 $ \hat{r}_{\theta}(x, y) = \beta \log \frac{\pi_{\theta}(y|x)}{\pi_{ref}(y|x)} $ 我们得到第4节中的梯度最终形式。

### A.5 证明引理1和2

在本节中，我们将证明第5节中的两个引理。

**引理1重述**. 在Plackett-Luce偏好框架下，特别是Bradley-Terry框架下，来自同一等价类的两个奖励函数会诱导相同的偏好分布。

**证明**. 我们说两个奖励函数 $ r(x, y) $ 和 $ r'(x, y) $ 来自同一等价类，如果 $ r'(x, y) = r(x, y) + f(x) $ 对于某个函数 $ f $。我们考虑一般的Plackett-Luce（Bradley-Terry模型是 $ K = 2 $ 的特例），并记 $ p_r $ 为由特定奖励函数 $ r(x, y) $ 诱导的排名概率分布。对于任何提示 $ x $，答案 $ y_1, \ldots, y_K $ 和排名 $ \tau $ 我们有：

$$ p'_r(\tau | y_1, \ldots, y_K, x) = \prod_{k=1}^{K} \frac{\exp(r'(x, y_{\tau(k)}))}{\sum_{j=k}^{K} \exp(r'(x, y_{\tau(j)}))} $$
$$ = \prod_{k=1}^{K} \frac{\exp(r(x, y_{\tau(k)}) + f(x))}{\sum_{j=k}^{K} \exp(r(x, y_{\tau(j)}) + f(x))}$$
$$= \prod_{k=1}^{K} \frac{\exp(f(x)) \exp(r(x, y_{\tau(k)}))}{\exp(f(x)) \sum_{j=k}^{K} \exp(r(x, y_{\tau(j)}))}$$
$$= \prod_{k=1}^{K} \frac{\exp(r(x, y_{\tau(k)}))}{\sum_{j=k}^{K} \exp(r(x, y_{\tau(j)}))} = p_r(\tau | y_1, \ldots, y_K, x),$$
这完成了证明。

**引理2重述**. 来自同一等价类的两个奖励函数在受约束的RL问题下诱导相同的最优策略。

**证明**. 让我们考虑来自同一类别的两个奖励函数，使得 $ r'(x, y) = r(x, y) + f(x) $，并且让我们记 $ \pi_r $ 和 $ \pi_{r'} $ 为相应的最优策略。根据方程(4)，对于所有的 $ x, y $ 我们有：

$$ \pi'_{r}(y | x) = \frac{1}{\sum_{y} \pi_{ref}(y | x) \exp\left(\frac{1}{\beta} r'(x, y)\right)} \pi_{ref}(y | x) \exp\left(\frac{1}{\beta} r'(x, y)\right)$$
$$= \frac{1}{\sum_{y} \pi_{ref}(y | x) \exp\left(\frac{1}{\beta}(r(x, y) + f(x))\right)} \pi_{ref}(y | x) \exp\left(\frac{1}{\beta}(r(x, y) + f(x))\right)$$
$$= \frac{\exp\left(\frac{1}{\beta} f(x)\right)}{\sum_{y} \pi_{ref}(y | x) \exp\left(\frac{1}{\beta} r(x, y)\right) \exp\left(\frac{1}{\beta} f(x)\right)} \frac{\pi_{ref}(y | x) \exp\left(\frac{1}{\beta} r(x, y)\right)}{\pi_{ref}(y | x)} \exp\left(\frac{1}{\beta} f(x)\right)$$
$$= \frac{1}{\sum_{y} \pi_{ref}(y | x) \exp\left(\frac{1}{\beta} r(x, y)\right)} \pi_{ref}(y | x) \exp\left(\frac{1}{\beta} r(x, y)\right) = \pi_r(y | x),$$


这完成了证明。

### A.6 证明定理1

在本节中，我们将扩展定理1的结果。

**定理1重述**. 假设我们有一个参考模型，使得 $ \pi_{ref}(y|x) > 0 $ 对于所有的提示 $ x $ 和答案 $ y $ 成立，并且参数 $ \beta > 0 $。所有奖励等价类，如第5节所定义，都可以使用重新参数化 $ r(x, y) = \beta \log \frac{\pi(y|x)}{\pi_{ref}(y|x)} $ 表示，对于某个模型 $ \pi(y|x) $。

**证明**. 考虑任何奖励函数 $ r(x, y) $，它在KL约束的RL问题下诱导了一个最优模型 $ \pi_r(y|x) $，其解由方程(4)给出。根据方程(5)，当我们对两边进行对数线性化时，我们得到：
$$
r(x, y) = \beta \log \frac{\pi_r(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)
$$
其中 $ Z(x) = \sum_y \pi_{ref}(y | x) \exp\left(\frac{1}{\beta} r(x, y)\right) $（注意 $ Z(x) $ 也依赖于奖励函数 $ r $）。使用操作符 $ r'(x, y) = f(r, \pi_{ref}, \beta)(x, y) = r(x, y) - \beta \log Z(x) $，我们可以看到这个新的奖励函数在 $ r $ 的等价类中，

我们有：
$$
r'(x, y) = \beta \log \frac{\pi_r(y|x)}{\pi_{ref}(y|x)}
$$
这完成了证明。

我们可以进一步扩展这些结果。我们可以看到，如果 $ r $ 和 $ r' $ 是同一类别中的两个奖励函数，那么
$$
f(r, \pi_{ref}, \beta)(x, y) = \beta \log \frac{\pi_r(y|x)}{\pi_{ref}(y|x)} = \beta \log \frac{\pi'_r(y|x)}{\pi_{ref}(y|x)} = f(r', \pi_{ref}, \beta)(x, y)
$$
其中第二个等式遵循引理2。我们已经证明了操作符 $ f $ 将所有来自特定等价类的奖励函数映射到同一个奖励函数。接下来，我们展示对于每个奖励函数的等价类，具有定理1中概述的重新参数化的奖励函数是唯一的。

**命题1**. 假设我们有一个参考模型，使得 $ \pi_{ref}(y|x) > 0 $ 对于所有的提示 $ x $ 和答案 $ y $ 成立，并且参数 $ \beta > 0 $。那么每个奖励函数的等价类，如第5节所定义，都有一个唯一的奖励函数 $ r(x, y) $，可以重新参数化为 $ r(x, y) = \beta \log \frac{\pi(y|x)}{\pi_{ref}(y|x)} $ 对于某个模型 $ \pi(y|x) $。

**证明**. 我们使用反证法进行。假设我们有来自同一类别的两个奖励函数，使得 $ r'(x, y) = r(x, y) + f(x) $。此外，假设 $ r'(x, y) = \beta \log \frac{\pi'(y|x)}{\pi_{ref}(y|x)} $ 对于某个模型 $ \pi'(y|x) $ 并且 $ r(x, y) = \beta \log \frac{\pi(y|x)}{\pi_{ref}(y|x)} $ 对于某个模型 $ \pi(y|x) $，使得 $ \pi \neq \pi' $。然后我们有：
$$
r'(x, y) = r(x, y) + f(x) = \beta \log \frac{\pi(y|x)}{\pi_{ref}(y|x)} + f(x) = \beta \log \frac{\pi(y|x) \exp(\frac{1}{\beta} f(x))}{\pi_{ref}(y|x)} = \beta \log \frac{\pi'(y|x)}{\pi_{ref}(y|x)}
$$

对于所有的提示 $ x $ 和完成 $ y $。那么我们必须有 $ \pi(y|x) \exp(\frac{1}{\beta} f(x)) = \pi'(y|x) $。由于这些都是分布，对 $ y $ 求和得到两边的和，我们得到 $ \exp(\frac{1}{\beta} f(x)) = 1 $ 并且由于 $ \beta > 0 $，我们必须有 $ f(x) = 0 $ 对于所有的 $ x $。因此 $ r(x, y) = r'(x, y) $。这完成了证明。

我们已经展示了每个奖励类别都有一个唯一的奖励函数，可以按照定理1中概述的那样表示，该函数由 $ f(r, \pi_{ref}, \beta) $ 给出，对于该类别中的任何奖励函数都是如此。

## 附录B. DPO实现细节和超参数

DPO相对容易实现；下面提供了DPO损失的PyTorch代码示例：

```python
import torch.nn.functional as F

def dpo_loss(pi_logps, ref_logps, yw_idxs, yl_idxs, beta):
    """pi_logps: 策略的对数概率，形状为 (B,)
    ref_logps: 参考模型的对数概率，形状为 (B,)
    yw_idxs: 首选完成的索引，在 [0, B-1] 中，形状为 (T,)
    yl_idxs: 非首选完成的索引，在 [0, B-1] 中，形状为 (T,)
    beta: 控制KL惩罚强度的温度参数
    每对 (yw_idxs[i], yl_idxs[i]) 代表单个偏好对的索引。
    """
    pi_yw_logps, pi_yl_logps = pi_logps[yw_idxs], pi_logps[yl_idxs]
    ref_yw_logps, ref_yl_logps = ref_logps[yw_idxs], ref_logps[yl_idxs]
    pi_logratios = pi_yw_logps - pi_yl_logps
    ref_logratios = ref_yw_logps - ref_yl_logps
    losses = -F.logsigmoid(beta * (pi_logratios - ref_logratios))
    rewards = beta * (pi_logps - ref_logps).detach()
    return losses, rewards
```

除非另有说明，我们通常使用 $\beta = 0.1 $，批量大小为64，并且使用学习率为 $1 \times 10^{-6} $ 的RMSprop优化器作为默认设置。我们从0到 $1 \times 10^{-6} $ 在150步内线性预热学习率。对于TL;DR摘要，我们使用 $\beta = 0.5 $，其余参数保持不变。
