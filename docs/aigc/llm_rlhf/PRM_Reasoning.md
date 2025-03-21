# 数学推理中过程奖励模型的开发经验

作者：Zhenru Zhang, Chujie Zheng, Yangzhen Wu, Beichen Zhang, Runji Lin, Bowen Yu, Dayiheng Liu, Jingren Zhou, Junyang Lin

Qwen团队，阿里巴巴集团

https://hf.co/Qwen/Qwen2.5-Math-PRM-7B
https://hf.co/Qwen/Qwen2.5-Math-PRM-72B


###### 摘要

过程奖励模型（Process Reward Models, PRMs）作为一种有前景的方法，用于在大语言模型（LLMs）的数学推理中进行过程监督，旨在识别和缓解推理过程中的中间错误。然而，开发有效的PRMs面临重大挑战，特别是在数据标注和评估方法上。本文通过大量实验证明，常用的基于蒙特卡洛（Monte Carlo, MC）估计的数据合成方法通常比LLM-as-a-judge和人工标注方法表现更差且泛化能力不足。MC估计依赖于完成模型来评估当前步骤的正确性，这可能导致从错误步骤生成正确答案或从正确步骤生成错误答案，从而导致步骤验证不准确。此外，我们发现了传统Best-of-N（Best-of-N）评估策略在PRMs中的潜在偏差：（1）不可靠的策略模型生成的响应可能包含正确答案但过程有缺陷，导致Best-of-N评估标准与PRM过程验证目标不一致。（2）PRMs对此类响应的容忍导致Best-of-N评分虚高。（3）现有PRMs在最终答案步骤中集中了大量最低分，揭示了Best-of-N优化PRMs从过程评估向结果评估的转变。为解决这些挑战，我们开发了一种共识过滤机制，有效整合了MC估计与LLM-as-a-judge，并倡导结合响应级和步骤级指标的更全面评估框架。基于这些机制，我们显著提高了Best-of-N评估和逐步错误识别任务中的模型性能和数据效率。最后，我们发布了一个新的最先进的PRM，其性能优于现有的开源替代方案，并为未来构建过程监督模型的研究提供了实用指南。

## 1 引言

近年来，大语言模型（LLMs）在数学推理方面取得了显著进展（OpenAI, 2023; Dubey et al., 2024; Shao et al., 2024; Zhu et al., 2024; Yang et al., , ），但它们仍可能犯错误，例如计算错误或逻辑错误，导致错误结论。此外，即使获得了正确的最终答案，这些强大的模型仍可能经常编造看似合理的推理步骤，最终答案建立在错误的计算或推导之上，这削弱了LLMs推理过程的可靠性和可信度。为解决这些挑战，过程奖励模型（PRMs; Lightman et al.2023; Wang et al.2024）作为一种代表性且近期备受关注的方法被提出，旨在识别和缓解过程错误，从而实现对推理过程的更细粒度监督。

开发PRMs的一个关键挑战在于推理过程正确性的数据标注，这通常既昂贵又耗时。虽然Lightman等人（2023）通过详细指导和复杂程序招募了人工标注员以实现满意的标注质量，但高昂的成本促使研究人员探索自动化标注方法。其中，一种常用的方法是通过蒙特卡洛（MC）方法估计导致正确最终答案的经验概率来评估过程正确性，这种方法引起了广泛的研究兴趣，并在实践中被广泛采用（Xiong et al., 2024; Wang et al., ; Luo et al., 2024）。另一个挑战在于评估PRM性能，因为之前的研究（Lightman et al., 2023; Wang et al., ; Luo et al., 2024）主要依赖于Best-of-N（Best-of-N）评估，即根据PRM从$N$个候选中选择得分最高的响应。最近，PROCESSBENCH（Zheng et al., 2024）出现，用于评估PRMs在识别逐步正确性方面的能力。

然而，在我们遵循传统原则使用MC估计构建数据并在Best-of-N上评估PRM的训练过程中，我们获得了几个重要的经验教训。在MC估计方面，（1）我们观察到通过MC估计训练的PRM在性能和泛化能力上显著劣于LLM-as-a-judge（Zheng et al., 2023）和人工标注。（2）我们将MC估计的次优表现归因于其基本局限性，即试图基于潜在未来结果评估确定性当前步骤的正确性。它严重依赖于完成模型的性能，该模型可能基于错误步骤生成正确答案，或基于正确步骤生成错误答案，从而在逐步正确性估计中引入了大量噪声和不准确性。关于Best-of-N评估，（1）不可靠的策略模型生成的响应可能包含正确答案但过程有缺陷，导致Best-of-N评估标准与PRM过程验证目标不一致。（2）有限的过程验证能力使PRMs对此类情况表现出容忍，导致Best-of-N性能虚高。（3）我们发现现有PRMs的步骤分数分布中，大量最低分集中在最终答案步骤上，表明PRMs在Best-of-N中已从过程评估转向结果评估。

为解决这些挑战，我们开发了一种共识过滤机制，结合了MC估计与LLM-as-a-judge。只有当LLM-as-a-judge和MC估计在解决方案中的错误推理步骤位置上达成共识时，实例才会被保留。我们的方法展示了更高效的数据利用，并在传统Best-of-N评估中超越了现有的开源PRMs。此外，我们倡导将响应级Best-of-N与逐步评估方法相结合。我们使用逐步基准PROCESSBENCH（Zheng et al., 2024）来衡量识别数学推理中过程错误的能力。我们训练的PRMs在错误识别性能上表现出色，优于其他开源模型，从PRMs到通用语言模型，证实了我们的训练方法真正教会了PRMs评估中间推理步骤的正确性。

我们的主要贡献总结如下：

* 我们识别了当前PRMs数据构建方法的关键局限性，证明基于MC估计的数据构建在性能上劣于LLM-as-a-judge和人工标注。
* 我们揭示了仅使用响应级Best-of-N评估PRMs的潜在偏差，并倡导结合响应级和步骤级指标的综合评估策略。
* 我们提出了一种简单而高效的共识过滤机制，整合了MC估计与LLM-as-a-judge，显著提高了PRM训练中的模型性能和数据效率。
*  我们通过大量实证研究证实了我们的发现，并开源了我们训练的PRMs，为未来推理过程监督的研究和开发提供了实用指南和最佳实践。

## 2 初步尝试

在本节中，我们描述了通过基于MC估计的推理步骤标注训练PRMs的初步尝试。尽管我们在扩展训练数据和仔细调整训练目标方面付出了努力，但我们发现基于MC估计的PRMs在识别特定错误推理步骤方面并不比基于人工标注数据训练的PRMs（Lightman et al., 2023）具有明显优势，甚至显著落后于后者。

### 2.1 训练设置

#### 训练数据合成

我们遵循常用的MC估计方法Math-Shepherd（Wang et al., 2024b）构建PRM训练数据。具体来说，我们收集了一个包含约50万条查询及其标准答案的大规模数据集。对于每个查询，我们通过混合Qwen2-Math-Instruct和Qwen2.5-Math-Instruct系列模型的输出生成6-8个不同的响应，涵盖7B和72B参数规模的模型。这些响应通过分隔符“\n\n”系统地拆分为单独的步骤。为了评估每个步骤的正确性，我们使用相应模型规模的Qwen2.5-Math-Instruct系列模型从该步骤开始进行八次独立完成，基于每个步骤生成正确最终答案的经验概率估计步骤标签。

#### 训练细节

我们训练的PRMs从监督微调的Qwen2.5-Math-7B/72B-Instruct模型（Yang et al., 2024c）初始化，其中我们将原始语言建模头（用于下一个令牌预测）替换为标量值头，由两个线性层组成。我们使用硬标签或软标签训练PRMs。对于硬标签，如果八次完成中有任何一次生成正确最终答案，则将该步骤视为正确，否则为负。对于软标签，我们将值（介于0和1之间）确定为导致正确最终答案的完成比例。我们分别使用硬标签和软标签对每个步骤的最后一个令牌计算交叉熵（CE）损失和均方误差（MSE）损失，用于二元分类任务和回归任务。请注意，我们删除了所有标记为错误（标签0）的步骤之后的步骤，因为在错误发生后它们的有效性变得无关紧要。此删除是为了防止训练期间潜在的模型混淆。

### 2.2 评估设置

我们从两个方面评估我们训练的PRMs：它们在直接提高下游任务性能方面的效用，以及它们在识别推理过程中特定错误步骤的能力。

#### Best-of-N

与之前的工作一致（Lightman et al., 2023; Wang et al., 2024b; Luo et al., 2024; Cobbe et al., 2021; Yang et al., 2024c），我们采用Best-of-N（Best-of-N）采样策略进行评估，即根据PRM从$N$个候选中选择得分最高的响应。我们将评估指标表示为“prm@N”。根据Yang等人（2024c），我们从Qwen2.5-Math-7B-Instruct中采样八个响应（即$N = 8$），涵盖多个数学基准，包括GSM8K（Cobbe et al., 2021）、MATH（Hendrycks et al., 2021b）、Minerva Math（Lewkowycz et al., 2022）、GaoKao 2023 En（Liao et al., 2024）、OlympiadBench（He et al., 2024）、College Math（Tang et al., 2024）和MMLU STEM（Hendrycks et al., 2021a）。每个候选响应的得分通过响应中每个步骤的单个得分的乘积计算，如Lightman等人（2023）所述。我们还报告了八次采样中的多数投票结果（maj@8）作为基线，以及pass@8（即测试样本中任何八次采样导致正确最终答案的比例）作为上限。

#### PROCESSBENCH

我们还对PROCESSBENCH进行了评估作为补充。PROCESSBENCH（Zheng et al., 2024）衡量模型识别数学推理中错误步骤的能力。模型需要识别包含错误的第一个步骤或得出结论所有步骤都正确。根据PROCESSBENCH中PRMs的评估方法，我们从PRMs生成的预测分数中定位第一个错误步骤。

### 2.3 评估结果

如表1和表2所示，我们将基于MC估计数据集训练的模型分别表示为Qwen2.5-Math-7B-PRM-MC-hard（使用硬标签训练）和Qwen2.5-Math-7B-PRM-MC-soft（使用软标签训练），并将它们与仅在PRM800K（Lightman et al., 2023）数据集上训练的基线模型Qwen2.5-Math-7B-PRM-PRM800K进行比较。实验结果表明，在Best-of-8评估中，没有任何PRM的prm@8得分优于maj@8。此外，在ProcessBench上，Qwen2.5-Math-7B-PRM-MC-hard和Qwen2.5-Math-7B-PRM-MC-soft在错误步骤定位能力上显著劣于Qwen2.5-Math-7B-PRM-PRM800K。这些不理想的评估表现促使我们反思当前流行的数据合成方法和评估策略。通过后续的优化过程，我们确实获得了一些观察和经验教训。

## 3 讨论与分析

在本节中，我们介绍了在PRM训练过程中获得的关键经验教训。我们的讨论包括三个主要方面：（1）PRMs训练中常用的MC估计方法的局限性，以及（2）使用Best-of-N作为优化PRMs的唯一评估指标的偏差。

### 3.1 MC估计在PRMs训练中的局限性

#### 3.1.1 PRMs与价值模型的区别

数学推理中的奖励模型作为正确性验证器，PRMs通过评估中间推理步骤的正确性提供细粒度监督。相比之下，价值模型估计从当前步骤在未来达到正确最终答案的潜力。PRM与价值模型的关键区别在于，PRMs作为当前步骤正确性的确定性评估器，而价值模型作为未来解决方案潜力的预测估计器。

MC估计试图估计从当前步骤在未来达到正确最终答案的潜力。当我们遵循这种方法构建数据并训练PRMs时，价值模型的原则本质上被纳入了PRMs训练中。这种方法可能引入性能和泛化能力的局限性，我们将在后续章节中讨论。

#### 3.1.2 MC估计 vs. LLM-as-a-judge vs. 人工标注

我们发现MC估计方法限制了PRM识别错误步骤的能力，如第2.3节的实验所示。为进一步调查，我们比较了使用三种不同数据构建方法的性能：MC估计、LLM-as-a-judge和人工标注。对于MC估计方法，我们分别在445k开源数据集Math-shepherd（Wang et al., ）和我们构建的860k类似数据集上训练PRM。对于我们构建的数据集，MC估计使用Qwen2-Math-Instruct的响应，并通过Qwen2.5-Math-Instruct完成后续推理过程。对于LLM-as-a-judge方法，我们使用相同的860k查询和响应，并利用Qwen2.5-72B-Instruct验证响应中每个步骤的正确性。我们在附录C中展示了用于验证的提示模板。对于人工标注方法，我们使用开源数据集PRM800K（Lightman et al., 2023），该数据集在去重后包含约265k个样本。

表3和表4分别展示了Best-of-8和ProcessBench的实验结果。对于Best-of-8，表3显示基于我们MC估计数据训练的PRM在平均准确率上表现最佳，而人工标注表现最差。然而，这两个模型在ProcessBench上的表现关系相反，这引起了我们的注意，并在第3.2节中进行了深入调查。表4显示，人工标注在数据量最少的情况下表现最佳，其次是LLM-as-a-judge，而MC估计尽管拥有最大的数据集，表现最差。具体来说，（1）人工标注尽管仅在MATH数据集上进行，但在更复杂的任务OlympiadBench和Omni-MATH上表现出优越的泛化能力。（2）在相同数据但不同标注方法的情况下，LLM-as-a-judge在具有挑战性的问题上表现出比MC估计更好的泛化性能，尽管后者在GSM8K上表现良好。（3）对于MC估计，我们860k数据集与Math-Shepherd 440k数据的比较表明，通过数据扩展仍可实现性能提升。

#### 3.1.3 MC估计中需要严格的数据过滤机制

我们将MC估计在识别错误步骤方面的较差表现归因于其对策略模型的严重依赖，导致推理步骤正确性估计中的高噪声和错误位置识别不准确。例如，策略模型可能生成正确的最终答案但错误的推理步骤，这将在第3.2.1节中详细探讨。

受LLM-as-a-judge在第3.1.2节中的鼓舞性结果的启发，我们自然提出了一种简单而高效的共识过滤机制，将LLM-as-a-judge与MC估计相结合。基于上述860K样本，只有当LLM-as-a-judge和MC估计在解决方案中的错误推理步骤位置上达成共识时，实例才会被保留。如图2所示，可以发现共识过滤后仅保留了约40%的数据。对于ProcessBench的评估，结果显示共识过滤后的缩减数据集显著优于MC估计，并且在使用仅40%数据的情况下，实现了与LLM-as-a-judge相当的性能。关于Best-of-N评估，这三个模型之间的性能差异较小。Best-of-N评估在PRMs中的局限性将在第3.2节中详细阐述。

#### 3.1.4 MC估计中的硬标签 vs. 软标签

尽管我们之前已经证明MC估计不如LLM-as-a-judge和人工标注有效，但MC估计中仍有一个值得讨论的点，即是否使用软标签或硬标签进行训练。我们使用MC估计构建了300万训练数据，其中对于每个推理步骤，我们进行8次完成。随后，我们应用第3.1.3节中讨论的共识过滤策略过滤300万样本，将数据集缩减至150万样本。我们分别在300万和150万数据上使用软标签和硬标签训练PRMs。

训练后的PRMs在Best-of-8和ProcessBench上的性能分别如图3和图4所示。在数据过滤之前，软标签和硬标签之间的性能差异不显著，我们将其归因于高噪声水平掩盖了它们的区别。然而，数据过滤后，这种差异变得明显，硬标签显著优于软标签。

表5展示了基于MC估计的8次完成结果，区分正负标签的阈值选择实验。我们按照之前的实验设置，在300万数据上进行了一系列实验，阈值从$1/8$到$7/8$，间隔为$1/8$，结果如图5所示。可以很容易地观察到，随着阈值的增加，Best-of-8和ProcessBench上的性能下降，表明使用MC估计值为0作为负标签，其他所有值作为正标签时效果最佳。因此，如果我们不得不依赖MC估计进行逐步正确性验证，我们建议将阈值设置为0，这意味着如果从该步骤开始的任何完成达到正确最终答案，则该步骤被视为正确。此阈值已在我们所有的实验研究中使用。

#### 3.1.5 总结

通过大量实验，我们证明了MC估计在性能和泛化能力上劣于LLM-as-a-judge和人工标注。然而，通过共识过滤策略将MC估计与LLM-as-a-judge结合，可以显著提高性能和数据效率。此外，当将MC估计值为0视为负标签并使用硬标签训练时，可以获得最佳结果。

### 3.2 Best-of-N采样在PRM性能评估中的偏差

尽管Best-of-N评估在PRM优化中常用，但其作为唯一优化标准的有效性值得仔细考虑，因为其在性能评估中存在潜在局限性。

#### 3.2.1 不可靠的策略模型导致Best-of-N-PRMs不一致

在理想情况下，策略模型生成的响应应同时包含正确答案和准确的解决步骤，或者相反，有缺陷的过程应对应于错误答案。然而，现有的策略模型倾向于生成包含正确答案但过程有缺陷的响应，而Best-of-N本质上只关注答案，导致Best-of-N评估标准与PRM过程验证目标不一致。为提供这一现象的经验证据，我们从GSM8K、MATH、OlympiadBench和Omni-MATH中使用策略模型Qwen2.5-Math-7B-Instruct采样8个响应。然后我们从这些响应中随机选择正确答案的响应并进行详细的人工标注。如图6所示，大量响应包含过程错误但保持正确答案。值得注意的是，与简单任务GSM8K和困难任务Omni-MATH相比，随着问题复杂性的增加，这种现象变得更加明显。这意味着有效的PRM可能会为包含正确答案但过程有缺陷的响应分配低分，从而导致Best-of-N评估中的整体性能较低。

#### 3.2.2 PRMs中有限的过程验证能力导致Best-of-N评分虚高

当PRM无法区分包含正确答案但过程有缺陷的响应并为其分配高分时，这会导致Best-of-N评估中的性能高估，从而对PRM能力产生过于乐观且可能误导的评估。一个典型的例子是第3.1.2节中的比较实验，如图8所示，基于我们MC估计数据、LLM-as-a-judge和PRM800K训练的PRMs在Best-of-N和ProcessBench评估中表现出相反的性能趋势。可以观察到，基于我们MC估计数据训练的模型在Best-of-N上表现出有限的过程验证能力但评分虚高。为调查PRMs对此类情况的区分能力，我们从ProcessBench中提取了答案正确但过程错误的实例，并分析了PRMs对这些实例的检测准确率。如表5所示，除了我们发布的PRMs Qwen2.5-Math-PRM-7B和Qwen2.5-Math-PRM-72B外，所有其他开源PRMs的检测准确率均低于50%。这种有限的区分能力表明，PRMs在Best-of-N评估中难以区分真正正确的响应和仅表面答案正确的响应。因此，这意味着除了Best-of-N评估外，还需要补充基准来评估PRMs的实际能力，特别是在检测过程错误方面。

#### 3.2.3 Best-of-N优化PRMs中的过程向结果转变

大多数当前PRMs都针对Best-of-N进行优化。然而，Best-of-N的局限性导致PRMs从过程评估转向结果评估。在基于PRM预测分数和遵循（Lightman et al., 2023）中响应评分方法的Best-of-N选择过程中，可以发现无论我们采用最低分还是分数乘积来评估完整解决方案，最低步骤分数都是影响PRMs选择标准的关键限制因素。

如图7所示，我们分析了多个开源PRMs分配的最低步骤分数的分布，特别关注最低分出现在最终步骤（通常包含最终答案）的情况。结果显示，EurusPRM-Stage1、EurusPRM-Stage2、Math-Shepherd-PRM-7B和Skywork-PRM-7B模型在此类别中表现出显著高的比例，超过40%。相比之下，我们发布的PRMs Qwen2.5-Math-PRM-72B和Qwen2.5-Math-PRM-7B在最终步骤中出现最低分的比例显著较低。

这一分析揭示了一些PRMs在Best-of-N评估中的性能主要由最终答案分数而非中间推理步骤决定，表明模型从基于过程的评估退化为结果导向的评估。换句话说，仅针对Best-of-N评估的优化使得当前PRMs在实践中表现得更像ORMs。因此，有必要补充响应级评估Best-of-N与步骤级评估方法，以避免过程向结果的转变。具体来说，我们可以采用过程错误定位任务，如ProcessBench。其他常用的逐步Best-of-N方法利用PRMs或价值模型与搜索机制的集成，提供了更细粒度的过程可靠性评估。值得注意的是，后者需要更多的计算成本。

#### 3.2.4 不同PRMs，不同最优评分策略

在Best-of-N评估中，整体解决方案得分通过组合各个步骤得分得出。当每个步骤的得分代表该特定步骤正确的概率时，通常可以接受通过乘积或最小值等方法组合这些步骤级得分来计算整体解决方案得分。然而，当使用MC估计时，情况有所不同。在这种情况下，每个步骤的得分实际上估计从当前位置在未来达到正确最终答案的概率。鉴于MC估计的前瞻性，我们既不应跨步骤乘估计概率（因为这些估计是相互依赖的），也不应简单地将特定步骤的最低估计值作为整体得分。相反，最终步骤的估计值自然整合了整个解决方案过程的信息，使其更适合作为完整解决方案的最终得分。

为验证这一点，我们评估了基于MC估计、LLM-as-a-judge和人工标注数据训练的PRMs在不同评分策略下的Best-of-N，如图9所示。我们发现，在MC估计中，使用最终得分在多个PRMs中显著优于乘积和最小值方法。而对于人工标注和LLM-as-a-judge，趋势则相反。这表明，如果PRM必须通过MC估计训练并在Best-of-N中评估，最终得分策略可能更合理和有效。然而，值得注意的是，这种PRM在Best-of-N中的使用已偏离了PRM的原始目的。

#### 3.2.5 总结

上述观察强调了Best-of-N评估中的关键局限性。首先，不可靠的策略模型生成的响应可能包含正确答案但过程有缺陷，导致Best-of-N评估标准与PRM过程验证目标不一致。其次，有限的过程验证能力使PRMs对包含正确答案但推理过程有缺陷的响应表现出容忍，导致Best-of-N性能虚高。第三，仅针对Best-of-N评估的模型优化导致PRMs从优先考虑推理过程转向优先考虑最终答案。

因此，我们认为补充步骤级评估在PRM评估中起着至关重要的作用。最后，在Best-of-N中，不同PRMs有不同的最优评分策略。对于通过MC估计训练的PRM，最终得分策略可能更合理和有效。相比之下，乘积和最小值评分更适合LLM-as-judge和人工标注。

## 4 我们的方法

本节介绍了我们克服上述局限性的方法，以及我们训练的PRM实现最先进性能的细节。此外，我们概述了实验设置、基线模型和评估结果。

### 训练细节

数据构建过程包括两个主要阶段：数据扩展和数据过滤。在扩展阶段，我们遵循第2.1节中描述的MC估计构建数据。我们使用硬标签，其中仅当八次完成中没有任何一次达到正确最终答案时，响应才被分类为负。在随后的过滤阶段，我们使用由Qwen2.5-Instruct-72B（Yang et al., ）实例化的LLM作为批评者，逐步验证所有响应的推理过程，即LLM-as-a-judge。我们通过过滤掉LLM标注和MC估计过程标签不一致的实例，实现了一种简单而高效的共识过滤机制。这确保了保留的数据在推理过程标注中保持高质量和一致性。对于训练任务，我们使用交叉熵损失对每个步骤的最后一个令牌进行训练，基于硬标签进行二元分类任务。我们训练了7B和72B参数的PRMs，分别从Qwen2.5-Math-7B-Instruct和Qwen2.5-Math-72B-Instruct初始化。

### 实验设置

为验证我们训练的PRM Qwen2.5-Math-PRM-7B和Qwen2.5-Math-PRM-72B的有效性，我们分别进行了响应级Best-of-N评估和步骤级过程错误识别任务ProcessBench（Zheng et al., 2024）。

#### Best-of-N

我们遵循第2.2节中的实验设置。在prm@8中，我们评估了结果奖励模型（ORMs）和过程奖励模型（PRMs）。对于ORMs，我们引入了Qwen2.5-Math-RM-72B（Yang et al., ），它为每个完整响应分配一个单一得分。对于PRMs，我们计算每个步骤得分的乘积作为最终响应得分。

我们与以下PRMs进行了比较：

* Math-Shepherd-PRM-7B（Wang et al., ）：通过估计达到正确最终答案的经验概率确定每个步骤的过程标签。
* RLHFlow-PRM-Mistral-8B & RLHFlow-PRM-Deepseek-8B（Xiong et al., 2024）：两个基于LLaMA-3.1的PRMs，采用Math-Shepherd的训练方法，同时实现不同的解决方案生成模型和优化目标。
* Skywork-PRM-1.5B & Skywork-PRM-7B（Skywork, 2024）：Skywork最近发布的两个基于Qwen2.5-Math的PRMs。
* EurusPRM-Stage1 & EurusPRM-Stage2（Cui et al., 2025）：两个使用隐式PRM方法（Yuan et al., 2024）训练的PRMs，具有7B参数，依赖于在响应级标签上训练的ORM获得过程奖励。
* Qwen2.5-Math-7B-Math-Shepherd & Qwen2.5-Math-7B-PRM800K：我们通过分别在PRM800K（Lightman et al., 2023）和Math-Shepherd（Wang et al., ）开源数据集上微调Qwen2.5-Math-7B-Instruct开发的两个额外PRMs。

#### ProcessBench

比较的PRMs与之前提到的PRMs一致。对于作为批评模型的LLM，即LLM-as-a-judge，我们与专有语言模型GPT-4o-0806（Hurst et al., 2024）和o1-mini（OpenAI, 2024）、开源语言模型Llama-3.3-70B-Instruct（Dubey et al., 2024）、Qwen2.5-Math-72B-Instruct（Yang et al., ）、Qwen2.5-72B-Instruct（Yang et al., ）和QwQ-32B-Preview（Owen, 2024）进行了比较。我们还将N步响应轨迹分解为N个单独的实例，以便ORM Qwen2.5-Math-RM-72B进行单独评分。

### 实验结果

#### Best-of-N

表6展示了在策略模型Qwen2.5-Math-7b-Instruct上的评估结果。Qwen2.5-Math-PRM-7B在同等模型规模的其他PRMs中表现出色。值得注意的是，它在所有7个任务中均优于maj@8，平均提高了1.4%。此外，Qwen2.5-Math-PRM-72B在整体性能上略优于Qwen2.5-Math-RM-72B，特别是在Minerva Math和MMLU STEM任务中表现出显著改进。详细的实验结果，包括在策略模型Qwen2.5-Math-72b-Instruct上的Best-of-N性能、替代评分策略以及在中国基准上的评估，均在附录A中全面记录。

#### ProcessBench

表7展示了ProcessBench的评估结果。与LLM-as-judge相比，较小模型规模的Qwen2.5-Math-PRM-7B在所有开源模型中表现出色。对于专有语言模型，Qwen2.5-Math-PRM-7B优于GPT-4o-0806，但与o1-mini相比仍存在性能差距。此外，与现有PRMs相比，Qwen2.5-Math-PRM-7B和Qwen2.5-Math-PRM-72B在识别步骤错误方面表现出显著优势。一个值得注意的有趣观察是，ORM Qwen2.5-Math-RM-72B在识别步骤错误方面表现出相当大的能力，甚至超过了一些开源PRMs，这验证了其作为超越纯规则机制的补充奖励的潜力。

## 5 相关工作

### 数学推理中的奖励模型

为进一步提高数学推理的准确性，奖励模型在选择最佳答案方面起着至关重要的作用。出现了两种主要类型的奖励模型：（1）结果奖励模型（ORM），它为整个解决方案提供评估分数，特别是最终答案。（2）过程奖励模型（PRM）（Lightman et al., 2023; Wang et al., 2024），它评估推理过程中的每个步骤。之前的工作（Lightman et al., 2023; Wang et al., 2024）表明，PRM优于ORM，尽管它需要更多高质量的训练数据。

### 数学推理步骤验证

评估推理步骤正确性有两种主要方法。第一种方法依赖于人工标注（Lightman et al., 2023），它产生高质量的数据，但成本高昂。第二种方法，引起了广泛的研究兴趣，专注于自动评估推理步骤的正确性。当前的自动化方法可分为两种主要类型：（1）基于反向传播的方法，从解决方案结果推断步骤正确性，包括MC估计（Wang et al., 2024b; Luo et al., 2024）、渐进ORM标注（Yuan et al., 2024）和信用分配（Yuan et al., 2024; Zhang et al., 2024）技术；（2）基于提示的方法，利用LLMs作为批评者，即LLM-as-a-judge（Zheng et al., 2023; Gao et al., 2024; Xia et al., 2024）直接评估步骤正确性。在本工作中，我们整合了MC估计和LLM-as-a-judge两种方法。

## 6 结论

在本文中，我们研究了过程奖励模型（PRM）并发布了一个有效的PRM，展示了卓越的性能。首先，我们讨论了在MC估计上的不理想尝试。然后，我们通过大量实验证明，基于MC估计的数据构建在性能和泛化能力上劣于LLM-as-a-judge和人工标注。此外，我们探讨了vanilla Best-of-N评估在PRMs中的局限性，这导致对PRM能力的评估不准确，并导致优化偏差，从过程导向转向结果导向验证。最后，我们提出了一种简单而有效的共识过滤策略，结合MC估计和LLM-as-a-judge，以克服MC估计的局限性。在评估方面，我们进行了响应级Best-of-N评估和步骤级过程错误识别任务ProcessBench，以避免仅依赖Best-of-N的偏差。实验证明，我们的策略显著提高了数据效率和模型性能。未来，PRMs在数据构建和评估方面仍有巨大潜力，推动更强大和可靠的PRMs的发展。

### 局限性

我们当前的工作仍存在一些局限性。首先，我们的PRM与Best-of-N上限（pass@8）之间存在显著的性能差距，表明仍有很大的优化潜力。最后，尽管我们的方法结合了LLM-as-a-judge与MC估计进行共识过滤，但现有高质量人工标注数据的有效利用仍未被充分探索。例如，通过弱监督方法逐步扩展高质量数据集可以作为一个有前景的未来研究方向。

## 参考文献

（此处省略参考文献部分）
