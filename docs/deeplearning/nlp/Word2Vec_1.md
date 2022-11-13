## Word2vec

![img](https://jalammar.github.io/images/word2vec/word2vec.png)

词嵌入（embedding）是机器学习中最迷人的想法之一。 如果你曾经使用 Siri、Google Assistant、Alexa、Google翻译，或者使用智能手机键盘进行下一词的预测，那么你很有可能从这个已经成为自然语言处理模型核心的想法中受益。在过去的几十年中，词嵌入用于神经网络模型已有相当大的发展。尤其是最近，出现了像 BERT 和 GPT2 等尖端模型的语境化嵌入方法。

Word2vec 是一种有效创建词嵌入的方法，它自2013年以来就一直存在。但除了作为词嵌入的方法之外，它的一些概念已经被证明可以在商业领域，非自然语言处理任务中有效地用于创建推荐系统和理解分析序列数据。像Airbnb、阿里巴巴、Spotify 这样的公司都从NLP领域中获取灵感并将相关技术应用于产品中，从而为新型推荐引擎提供支持。

在这篇文章中，我们将讨论词嵌入的概念，以及使用word2vec生成词嵌入的机制。让我们从一个例子开始，熟悉使用向量来表示事物。你是否知道你的个性可以仅被五个数字的列表（向量）表示？

### 个性嵌入（你是什么样的人）

如何用0到100的范围来表示你是多么内向/外向（其中0是最内向的，100是最外向的）？ 你有没有做过像MBTI那样的人格测试，或者五大人格特质测试？ 如果你还没有，这些测试会问你一系列的问题，然后在很多维度给你打分，内向/外向就是其中之一。

![img](https://jalammar.github.io/images/word2vec/big-five-personality-traits-score.png)

> 五大人格特质测试测试结果示例。它可以真正告诉你很多关于你自己的事情，并且在学术、人格和职业成功方面都具有预测能力。、

假设我的内向/外向得分为38/100。 我们可以用这种方式绘图：

![img](https://jalammar.github.io/images/word2vec/introversion-extraversion-100.png)

让我们把范围收缩到-1到1:

![img](https://jalammar.github.io/images/word2vec/introversion-extraversion-1.png)

当你只知道这一条信息的时候，你觉得你有多了解这个人？了解不多。人很复杂，让我们添加另一测试的得分作为新维度。

![img](https://jalammar.github.io/images/word2vec/two-traits-vector.png)

我们可以将两个维度表示为图形上的一个点，或者作为从原点到该点的向量。我们拥有很棒的工具来处理即将上场的向量。

我已经隐藏了我们正在绘制的人格特征，这样你会渐渐习惯于在不知道每个维度代表什么的情况下，从一个人格的向量表示中获得价值信息。

我们现在可以说这个向量部分地代表了我的人格。当你想要将另外两个人与我进行比较时，这种表示法就有用了。假设我被公共汽车撞了，我需要被性格相似的人替换，那在下图中，两个人中哪一个更像我？

![img](https://jalammar.github.io/images/word2vec/personality-two-persons.png)

在处理向量时，计算相似度得分的常用方法是余弦相似度：[cosine_similarity](https://en.wikipedia.org/wiki/Cosine_similarity):

![img](https://jalammar.github.io/images/word2vec/cosine-similarity.png)

1号在性格上与我更相似。指向相同方向的向量（长度也起作用）具有更高的余弦相似度。

再一次，两个维度还不足以捕获有关不同人群的足够信息。心理学已经研究出了五个主要人格特征（以及大量的子特征），所以让我们使用所有五个维度进行比较：

![img](https://jalammar.github.io/images/word2vec/big-five-vectors.png)

使用五个维度的问题是我们不能在二维平面绘图表示了。这是机器学习中的常见问题，我们经常需要在更高维度的空间中思考。 但好在余弦相似度仍然有效，它适用于任意维度：

![img](https://jalammar.github.io/images/word2vec/embeddings-cosine-personality.png)

> 余弦相似度适用于任意数量的维度。这些得分比上次的得分要更好，因为它们是根据被比较事物的更高维度算出的。

在本节的最后，我希望提出两个中心思想：

1.我们可以将人和事物表示为数值向量（这对机器来说非常有用）。

2.我们可以很容易地计算出相似的向量之间的相互关系。

![img](https://jalammar.github.io/images/word2vec/section-1-takeaway-vectors-cosine.png)

### **词嵌入**

通过上文的理解，我们继续看看训练好的词向量实例（也被称为词嵌入）并探索它们的一些有趣属性。

这是一个单词“king”的词嵌入（在维基百科上训练的GloVe向量）：

```python
[ 0.50451 , 0.68607 , -0.59517 , -0.022801, 0.60046 , -0.13498 , -0.08813 , 0.47377 , -0.61798 , -0.31012 , -0.076666, 1.493 , -0.034189, -0.98173 , 0.68229 , 0.81722 , -0.51874 , -0.31503 , -0.55809 , 0.66421 , 0.1961 , -0.13495 , -0.11476 , -0.30344 , 0.41177 , -2.223 , -1.0756 , -1.0783 , -0.34354 , 0.33505 , 1.9927 , -0.04234 , -0.64319 , 0.71125 , 0.49159 , 0.16754 , 0.34344 , -0.25663 , -0.8523 , 0.1661 , 0.40102 , 1.1685 , -1.0137 , -0.21585 , -0.15155 , 0.78321 , -0.91241 , -1.6106 , -0.64426 , -0.51042 ]
```

这是一个包含50个数字的列表。通过观察数值我们看不出什么，但是让我们稍微给它可视化，以便比较其它词向量。我们把所有这些数字放在一行：

![img](https://jalammar.github.io/images/word2vec/king-white-embedding.png)

让我们根据它们的值对单元格进行颜色编码（如果它们接近2则为红色，接近0则为白色，接近-2则为蓝色）：

![img](https://jalammar.github.io/images/word2vec/king-colored-embedding.png)

我们将忽略数字并仅查看颜色以指示单元格的值。现在让我们将“king”与其它单词进行比较：

![img](https://jalammar.github.io/images/word2vec/king-man-woman-embedding.png)

看到“Man”和“Woman”彼此之间比它们任一单词与“King”相比更相似, 这暗示你一些事情。这些向量图示很好的展现了这些单词的信息/含义/关联。

这是另一个示例列表（通过垂直扫描列来查找具有相似颜色的列）：

![img](https://jalammar.github.io/images/word2vec/queen-woman-girl-embeddings.png)

有几个要点需要指出：

1.所有这些不同的单词都有一条直的红色列。 它们在这个维度上是相似的（虽然我们不知道每个维度是什么）

2.你可以看到“woman”和“girl”在很多地方是相似的，“man”和“boy”也是一样

3.“boy”和“girl”也有彼此相似的地方，但这些地方却与“woman”或“man”不同。这些是否可以总结出一个模糊的“youth”概念？可能吧。

4.除了最后一个单词，所有单词都是代表人。 我添加了一个对象“water”来显示类别之间的差异。你可以看到蓝色列一直向下并在 “water”的词嵌入之前停下了。

5.“king”和“queen”彼此之间相似，但它们与其它单词都不同。这些是否可以总结出一个模糊的“royalty”概念？

### **类比**

展现词嵌入奇妙属性的著名例子是类比。我们可以将两个词嵌入相加或相减，并得到有趣的结果。一个著名例子是公式：“king”-“man”+“woman”：

![img](https://jalammar.github.io/images/word2vec/king-man+woman-gensim.png)

在python中使用 [Gensim](https://radimrehurek.com/gensim/) 库，我们可以添加和减去词向量，它会找到与结果向量最相似的单词。该图像显示了最相似的单词列表，每个单词都具有余弦相似性。

我们可以像之前一样可视化这个类比：

![img](https://jalammar.github.io/images/word2vec/king-analogy-viz.png)


由“king-man + woman”生成的向量并不完全等同于“queen”，但“queen”是我们在此集合中包含的400,000个字嵌入中最接近它的单词。

现在我们已经看过训练好的词嵌入，接下来让我们更多地了解训练过程。 但在我们开始使用word2vec之前，我们需要看一下词嵌入的父概念：神经语言模型。

### 语言模型

如果要举自然语言处理最典型的例子，那应该就是智能手机输入法中的下一单词预测功能。这是个被数十亿人每天使用上百次的功能。

![img](https://jalammar.github.io/images/word2vec/swiftkey-keyboard.png)

下一单词预测是一个可以通过语言模型实现的任务。语言模型会通过单词列表(比如说两个词)去尝试预测可能紧随其后的单词。

在上面这个手机截屏中，我们可以认为该模型接收到两个绿色单词(thou shalt)并推荐了一组单词(“not” 就是其中最有可能被选用的一个)：

![img](https://jalammar.github.io/images/word2vec/thou-shalt-_.png)

我们可以把这个模型想象为这个黑盒:

![img](https://jalammar.github.io/images/word2vec/language_model_blackbox.png)

但事实上，该模型不会只输出一个单词。实际上，它对所有它知道的单词(模型的词库，可能有几千到几百万个单词) 按可能性打分，输入法程序会选出其中分数最高的推荐给用户。

![img](https://jalammar.github.io/images/word2vec/language_model_blackbox_output_vector.png)

> 自然语言模型的输出就是模型所知单词的概率评分，我们通常把概率按百分比表示，但是实际上，40%这样的分数在输出向量组是表示为0.4

自然语言模型(请参考[Bengio 2003](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf))在完成训练后，会按如下中所示法分三步完成预测：

![img](https://jalammar.github.io/images/word2vec/neural-language-model-prediction.png)

第一步与我们最相关，因为我们讨论的就是Embedding。模型在经过训练之后会生成一个映射单词表所有单词的矩阵。在进行预测的时候，我们的算法就是在这个映射矩阵中查询输入的单词，然后计算出预测值:

![img](https://jalammar.github.io/images/word2vec/neural-language-model-embedding.png)

现在让我们将重点放到模型训练上，来学习一下如何构建这个映射矩阵。

### **语言模型训练**

相较于大多数其他机器学习模型，语言模型有一个很大有优势，那就是我们有丰富的文本来训练语言模型。所有我们的书籍、文章、维基百科、及各种类型的文本内容都可用。相比之下，许多其他机器学习的模型开发就需要手工设计数据或者专门采集数据。

我们通过找常出现在每个单词附近的词，就能获得它们的映射关系。机制如下：

1. 先是获取大量文本数据(例如所有维基百科内容)

2. 然后我们建立一个可以沿文本滑动的窗(例如一个窗里包含三个单词)

3. 利用这样的滑动窗就能为训练模型生成大量样本数据。

![img](https://jalammar.github.io/images/word2vec/wikipedia-sliding-window.png)

当这个窗口沿着文本滑动时，我们就能(真实地)生成一套用于模型训练的数据集。为了明确理解这个过程，我们看下滑动窗是如何处理这个短语的:

在一开始的时候，窗口锁定在句子的前三个单词上:

![img](https://jalammar.github.io/images/word2vec/lm-sliding-window.png)

我们把前两个单词单做特征，第三个单词单做标签:

![img](https://jalammar.github.io/images/word2vec/lm-sliding-window-2.png)

这时我们就生产了数据集中的第一个样本，它会被用在我们后续的语言模型训练中。

接着，我们将窗口滑动到下一个位置并生产第二个样本:

![img](https://jalammar.github.io/images/word2vec/lm-sliding-window-3.png)

这时第二个样本也生成了。不用多久，我们就能得到一个较大的数据集，从数据集中我们能看到在不同的单词组后面会出现的单词:

![img](https://jalammar.github.io/images/word2vec/lm-sliding-window-4.png)

在实际应用中，模型是往往在我们滑动窗口时就被训练的。但是我觉得将生成数据集和训练模型分为两个阶段会显得更清晰易懂一些。除了使用神经网络建模之外，大家还常用一项名为N-gams的技术进行模型训练。 (see: Chapter 3 of [Speech and Language Processing](http://web.stanford.edu/~jurafsky/slp3/)).

如果想了解现实产品从使用N-gams模型到使用神经网络模型的转变，可以看一下Swiftkey (我最喜欢的安卓输入法)在2015年的发表一篇博客 [here’s a 2015 blog post from Swiftkey](https://blog.swiftkey.com/neural-networks-a-meaningful-leap-for-mobile-typing/), 文中介绍了他们的自然语言模型及该模型与早期N-gams模型的对比。我很喜这个例子，因为这个它能告诉你如何在营销宣讲中把Embedding的算法属性解释清楚。

###  顾及两头

根据前面的信息进行填空:

![img](https://jalammar.github.io/images/word2vec/jay_was_hit_by_a_.png)

在空白前面，我提供的背景是五个单词(如果事先提及到‘bus’)，可以肯定，大多数人都会把bus填入空白中。但是如果我再给你一条信息——比如空白后的一个单词，那答案会有变吗？

![img](https://jalammar.github.io/images/word2vec/jay_was_hit_by_a_bus.png)

这下空白处改填的内容完全变了。这时’red’这个词最有可能适合这个位置。从这个例子中我们能学到，一个单词的前后词语都带信息价值。事实证明，我们需要考虑两个方向的单词(目标单词的左侧单词与右侧单词)。那我们该如何调整训练方式以满足这个要求呢，继续往下看。

### Skipgram

我们不仅要考虑目标单词的前两个单词，还要考虑其后两个单词。

![img](https://jalammar.github.io/images/word2vec/continuous-bag-of-words-example.png)

如果这么做，我们实际上构建并训练的模型就如下所示：

![img](https://jalammar.github.io/images/word2vec/continuous-bag-of-words-dataset.png)

上述的这种架构被称为连续词袋(CBOW)，在一篇关于word2vec的论文中有阐述  [one of the word2vec papers](https://arxiv.org/pdf/1301.3781.pdf)。

还有另一种架构，它不根据前后文(前后单词)来猜测目标单词，而是推测当前单词可能的前后单词。我们设想一下滑动窗在训练数据时如下图所示：

![img](https://jalammar.github.io/images/word2vec/skipgram-sliding-window.png)

> 绿框中的词语是输入词，粉框则是可能的输出结果

这里粉框颜色深度呈现不同，是因为滑动窗给训练集产生了4个独立的样本:

![img](https://jalammar.github.io/images/word2vec/skipgram-sliding-window-samples.png)

这种方式称为Skipgram架构。我们可以像下图这样将展示滑动窗的内容。

![img](https://jalammar.github.io/images/word2vec/skipgram-sliding-window-1.png)

这样就为数据集提供了4个样本:

![img](https://jalammar.github.io/images/word2vec/skipgram-sliding-window-2.png)

然后我们移动滑动窗到下一个位置:

![img](https://jalammar.github.io/images/word2vec/skipgram-sliding-window-3.png)

这样我们又产生了接下来4个样本:

![img](https://jalammar.github.io/images/word2vec/skipgram-sliding-window-4.png)

在移动几组位置之后，我们就能得到一批样本:

![img](https://jalammar.github.io/images/word2vec/skipgram-sliding-window-5.png)

### **重新审视训练过程**

现在我们已经从现有的文本中获得了Skipgram模型的训练数据集，接下来让我们看看如何使用它来训练一个能预测相邻词汇的自然语言模型。

![img](https://jalammar.github.io/images/word2vec/skipgram-language-model-training.png)

从数据集中的第一个样本开始。我们将特征输入到未经训练的模型，让它预测一个可能的相邻单词。

![img](https://jalammar.github.io/images/word2vec/skipgram-language-model-training-2.png)

该模型会执行三个步骤并输入预测向量(对应于单词表中每个单词的概率)。因为模型未经训练，该阶段的预测肯定是错误的。但是没关系，我们知道应该猜出的是哪个单词——这个词就是我训练集数据中的输出标签:

![img](https://jalammar.github.io/images/word2vec/skipgram-language-model-training-3.png)

目标单词概率为1，其他所有单词概率为0，这样数值组成的向量就是“目标向量”。

模型的偏差有多少？将两个向量相减，就能得到偏差向量:

![img](https://jalammar.github.io/images/word2vec/skipgram-language-model-training-4.png)

现在这一误差向量可以被用于更新模型了，所以在下一轮预测中，如果用not作为输入，我们更有可能得到thou作为输出了。

![img](https://jalammar.github.io/images/word2vec/skipgram-language-model-training-5.png)

这其实就是训练的第一步了。我们接下来继续对数据集内下一份样本进行同样的操作，直到我们遍历所有的样本。这就是一轮（epoch）了。我们再多做几轮（epoch），得到训练过的模型，于是就可以从中提取嵌入矩阵来用于其他应用了。

以上确实有助于我们理解整个流程，但这依然不是word2vec真正训练的方法。我们错过了一些关键的想法。

### **负例采样**

回想一下这个神经语言模型计算预测值的三个步骤：

![img](https://jalammar.github.io/images/word2vec/language-model-expensive.png)

从计算的角度来看，第三步非常昂贵 - 尤其是当我们将需要在数据集中为每个训练样本都做一遍（很容易就多达数千万次）。我们需要寻找一些提高表现的方法。

一种方法是将目标分为两个步骤：

1.生成高质量的词嵌入（不要担心下一个单词预测）。

2.使用这些高质量的嵌入来训练语言模型（进行下一个单词预测）。

在本文中我们将专注于第1步（因为这篇文章专注于嵌入）。要使用高性能模型生成高质量嵌入，我们可以改变一下预测相邻单词这一任务：

![img](https://jalammar.github.io/images/word2vec/predict-neighboring-word.png)

将其切换到一个提取输入与输出单词的模型，并输出一个表明它们是否是邻居的分数（0表示“不是邻居”，1表示“邻居”）

![img](https://jalammar.github.io/images/word2vec/are-the-words-neighbors.png)

这个简单的变换将我们需要的模型从神经网络改为逻辑回归模型——因此它变得更简单，计算速度更快。

这个开关要求我们切换数据集的结构——标签值现在是一个值为0或1的新列。它们将全部为1，因为我们添加的所有单词都是邻居。

![img](https://jalammar.github.io/images/word2vec/word2vec-training-dataset.png)

现在的计算速度可谓是神速啦——在几分钟内就能处理数百万个例子。但是我们还需要解决一个漏洞。如果所有的例子都是邻居（目标：1），我们这个”天才模型“可能会被训练得永远返回1——准确性是百分百了，但它什么东西都学不到，只会产生垃圾嵌入结果。

![img](https://jalammar.github.io/images/word2vec/word2vec-smartass-model.png)

为了解决这个问题，我们需要在数据集中引入负样本 - 不是邻居的单词样本。我们的模型需要为这些样本返回0。模型必须努力解决这个挑战——而且依然必须保持高速。

![img](https://jalammar.github.io/images/word2vec/word2vec-negative-sampling.png)

对于我们数据集中的每个样本，我们添加了负例。它们具有相同的输入字词，标签为0。

但是我们作为输出词填写什么呢？我们从词汇表中随机抽取单词

![img](https://jalammar.github.io/images/word2vec/word2vec-negative-sampling-2.png)

这个想法的灵感来自噪声对比估计  [Noise-contrastive estimation](http://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf) [pdf]。我们将实际信号（相邻单词的正例）与噪声（随机选择的不是邻居的单词）进行对比。这导致了计算和统计效率的巨大折衷。

### **基于负例采样的Skipgram（SGNS）**

我们现在已经介绍了word2vec中的两个（一对）核心思想：负例采样，以及skipgram。

![img](https://jalammar.github.io/images/word2vec/skipgram-with-negative-sampling.png)

### **Word2vec训练流程**

现在我们已经了解了skipgram和负例采样的两个中心思想，可以继续仔细研究实际的word2vec训练过程了。

在训练过程开始之前，我们预先处理我们将要训练模型的文本。在这一步中，我们确定一下词典的大小（我们称之为vocab_size，比如说10,000）以及哪些词被它包含在内。

在训练阶段的开始，我们创建两个矩阵——Embedding矩阵和Context矩阵。这两个矩阵在我们的词汇表中嵌入了每个单词（所以vocab_size是他们的维度之一）。第二个维度是我们希望每次嵌入的长度（embedding_size——300是一个常见值，但我们在前文也看过50的例子）。

![img](https://jalammar.github.io/images/word2vec/word2vec-embedding-context-matrix.png)

在训练过程开始时，我们用随机值初始化这些矩阵。然后我们开始训练过程。在每个训练步骤中，我们采取一个相邻的例子及其相关的非相邻例子。我们来看看我们的第一组：

![img](https://jalammar.github.io/images/word2vec/word2vec-training-example.png)

现在我们有四个单词：输入单词not和输出/上下文单词: thou（实际邻居词），aaron和taco（负面例子）。我们继续查找它们的嵌入——对于输入词，我们查看Embedding矩阵。对于上下文单词，我们查看Context矩阵（即使两个矩阵都在我们的词汇表中嵌入了每个单词）。

![img](https://jalammar.github.io/images/word2vec/word2vec-lookup-embeddings.png)

然后，我们计算输入嵌入与每个上下文嵌入的点积。在每种情况下，结果都将是表示输入和上下文嵌入的相似性的数字。

![img](https://jalammar.github.io/images/word2vec/word2vec-training-dot-product.png)

现在我们需要一种方法将这些分数转化为概率——我们需要它们都是正值，并且 处于0到1之间。sigmoid这一逻辑函数转换正适合用来做这样的事情啦。 [sigmoid](https://jalammar.github.io/feedforward-neural-networks-visual-interactive/#sigmoid-visualization), the [logistic operation](https://en.wikipedia.org/wiki/Logistic_function).

![img](https://jalammar.github.io/images/word2vec/word2vec-training-dot-product-sigmoid.png)

现在我们可以将sigmoid操作的输出视为这些示例的模型输出。您可以看到taco得分最高，aaron最低，无论是sigmoid 操作之前还是之后。

既然未经训练的模型已做出预测，而且我们确实拥有真实目标标签来作对比，那么让我们计算模型预测中的误差吧。为此我们只需从目标标签中减去sigmoid分数。

![img](https://jalammar.github.io/images/word2vec/word2vec-training-error.png)
`error` = `target` - `sigmoid_scores`

这是“机器学习”的“学习”部分。现在，我们可以利用这个错误分数来调整not、thou、aaron和taco的嵌入，使我们下一次做出这一计算时，结果会更接近目标分数。

![img](https://jalammar.github.io/images/word2vec/word2vec-training-update.png)

训练步骤到此结束。我们从中得到了这一步所使用词语更好一些的嵌入（not，thou，aaron和taco）。我们现在进行下一步（下一个相邻样本及其相关的非相邻样本），并再次执行相同的过程。

![img](https://jalammar.github.io/images/word2vec/word2vec-training-example-2.png)

当我们循环遍历整个数据集多次时，嵌入会继续得到改进。然后我们就可以停止训练过程，丢弃Context矩阵，并使用Embeddings矩阵作为下一项任务的已被训练好的嵌入。



**窗口大小和负样本数量**

word2vec训练过程中的两个关键超参数是窗口大小和负样本的数量。

![img](https://jalammar.github.io/images/word2vec/word2vec-window-size.png)

不同的任务适合不同的窗口大小。一种启发式方法是，使用较小的窗口大小（2-15）会得到这样的嵌入：两个嵌入之间的高相似性得分表明这些单词是可互换的（注意，如果我们只查看附近距离很近的单词，反义词通常可以互换——例如，好的和坏的经常出现在类似的语境中）。使用较大的窗口大小（15-50，甚至更多）会得到相似性更能指示单词相关性的嵌入。在实际操作中，你通常需要对嵌入过程提供指导以帮助读者得到相似的”语感“。Gensim默认窗口大小为5（除了输入字本身以外还包括输入字之前与之后的两个字）。

![img](https://jalammar.github.io/images/word2vec/word2vec-negative-samples.png)

负样本的数量是训练训练过程的另一个因素。原始论文认为5-20个负样本是比较理想的数量。它还指出，当你拥有足够大的数据集时，2-5个似乎就已经足够了。Gensim默认为5个负样本。

### **结论**

我希望您现在对词嵌入和word2vec算法有所了解。我也希望现在当你读到一篇提到“带有负例采样的skipgram”（SGNS）的论文（如顶部的推荐系统论文）时，你已经对这些概念有了更好的认识。

### References & Further Readings

- [Distributed Representations of Words and Phrases and their Compositionality](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) [pdf]
- [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf) [pdf]
- [A Neural Probabilistic Language Model](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) [pdf]
- [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/) by Dan Jurafsky and James H. Martin is a leading resource for NLP. Word2vec is tackled in Chapter 6.
- [Neural Network Methods in Natural Language Processing](https://www.amazon.com/Language-Processing-Synthesis-Lectures-Technologies/dp/1627052984) by [Yoav Goldberg](https://twitter.com/yoavgo) is a great read for neural NLP topics.
- [Chris McCormick](http://mccormickml.com/) has written some great blog posts about Word2vec. He also just released [The Inner Workings of word2vec](https://www.preview.nearist.ai/paid-ebook-and-tutorial), an E-book focused on the internals of word2vec.
- Want to read the code? Here are two options:
  - Gensim’s [python implementation](https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/word2vec.py) of word2vec
  - Mikolov’s original [implementation in C](https://github.com/tmikolov/word2vec/blob/master/word2vec.c) – better yet, this [version with detailed comments](https://github.com/chrisjmccormick/word2vec_commented/blob/master/word2vec.c) from Chris McCormick.
- [Evaluating distributional models of compositional semantics](http://sro.sussex.ac.uk/id/eprint/61062/1/Batchkarov, Miroslav Manov.pdf)
- [On word embeddings](http://ruder.io/word-embeddings-1/index.html), [part 2](http://ruder.io/word-embeddings-softmax/)
