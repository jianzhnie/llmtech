# NLP 基本任务分类

## 词法分析（Lexical Analysis）

词法分析（Lexical Analysis）是指对自然语言进行词汇层面的分析，是NLP基础性工作，主要包括以下任务：

- 分词 （Word segmentation / Tokenization）
  - 对没有明显边界的文本进行切分，得到词序列

- 新词发现（New Word’s Identification）
  - 从未登录词中识别出新词

- 形态分析 （Morphological Analysis）
  - 分析单词的形态组成，包括词干（Sterms）、词根（Roots）、词缀（Prefixes and Suffixes）等

- 词性标注 （Part-of-Speech Tagging）
  - 标注词汇的词性，如名词、动词、形容词，代词等

- 拼写校正 （Spell Correction）
  -  对拼写错误的词进行校正

## 句法分析（Syntactic Analysis）

句法分析（Syntactic Analysis）指对自然语言进行句子层面的分析，包括句法分析和其他句子级别的分析任务，主要包括以下任务：

- 组块分析 （Chunking）
  - 识别出句子中的短语块，如名词短语、动词短语、介词短语等
- 超级标签标注 （Super Tagging）
  - 为句子中的每个词汇标注一个超级标签，如词性标签、组块标签等，超级标签是句法树中与该词相关的树形结构

- 成分句法分析 （Constituency Parsing）
  - 为句子中的每个词汇标注一个成分标签，如NP、VP、PP等，给出一棵树由终结符和非终结符构成的句法树

- 依存句法分析 （Dependency Parsing）
  -  为句子中的每个词汇标注一个依存标签，如主谓关系、动宾关系等，给出一棵由词语依存关系构成的依存句法树 

- 语种识别 （Language Identification）
  - 识别出句子所属的语种

- 语言模型 （Language Modeling）
  -  对给定的一个句子进行打分，该分数代表句子合理性（流畅度）的程度

- 句子边界检测 （Sentence Boundary Detection）:
  -  给没有明显句子边界的文本加边界

## 语义分析（Semantic Analysis）

语义分析（Semantic Analysis） 对给定文本进行分析和理解，形成能勾够表达语义的形式化表示或分布式表示, 主要包括以下任务

- 词义消歧 （Word Sense Disambiguation）
  - 对有歧义的词汇进行消歧，确定其准确的词义

- 语义角色标注 （Semantic Role Labeling） 
  - 标注句子中的语义角色类标，语义角色，语义角色包括施事、受事、影响等

- 抽象语义表示 （Abstract Meaning Representation） 
  - 为句子构建抽象语义表示，抽象语义表示是一种图结构，图中的节点表示句子中的词汇，边表示词汇之间的语义关系

- 一阶谓词逻辑 （First-Order Predicate Logic） 
  - 使用一阶谓词逻辑系统表达语义

- 框架语义分析 （Frame Semantic Parsing）
  - 根据框架语义学的观点，对句子进行语义分析

- 词汇/句子/段落向量化 （Word/Sentence/Paragraph Vectorization） 
  - 将词汇、句子、段落等文本转换为向量表示

## 信息抽取（Information Extraction）

信息抽取（Information Extraction）指从无结构文本中抽取结构化的信息，主要包括以下任务：


- 命名实体识别 （Named Entity Recognition） 

  - 识别出句子中的命名实体，如人名、地名、机构名等

- 实体消歧 （Entity Disambiguation）

  -  对命名实体进行消歧，确定其准确的指称

- 术语抽取 （Terminology Extraction） 

  - 从文本中抽取出术语

- 共指消解 （Coreference Resolution）

  -  确定不同实体的等价描述，包括代词消解和名词消解

- 关系抽取 （Relation Extraction） 

  - 识别出句子中的关系三元组，如人物关系、地理关系等

- 事件抽取 （Event Extraction） 

  - 从无结构的文本中抽取结构化事件

- 情感分析 （Sentiment Analysis） 

  - 对文本的主观性情绪进行提取，如正面情绪、负面情绪等

- 意图识别 （Intent Detection） 

  - 对话系统中的一个重要模块，对用户给定的对话内容进行分析，识别用户意图

- 槽位填充 （Slot Filling） 

  - 对话系统中的一个重要模块，从对话内容中分析出于用户意图相关的有效信息

## 顶层任务（Top-level Tasks）

顶层任务（Top-level Tasks） 指直接面向普通用户，提供自然语言处理产品服务的系统级任务，会用到多个层面的自然语言处理技术，主要包括以下任务：

- 机器翻译（Machine Translation）
- 文本摘要 （Text Summarization）
- 文本分类 （Text Classification）
- 问答系统 （Question Answering）
- 对话系统 （Dialogue System）
- 阅读理解 （Reading Comprehension）
- 信息检索 （Information Retrieval）

# NLP 四大类任务

**自然语言处理有四大类常见的任务**

```undefined
第一类任务：序列标注，譬如命名实体识别、语义标注、词性标注、分词等；
第二类任务：分类任务，譬如文本分类、情感分析等；
第三类任务：句对关系判断，譬如自然语言推理、问答QA、文本语义相似性等；
第四类任务：生成式任务，譬如机器翻译、文本摘要、写诗造句等。
```

## 序列标注任务

序列标注（Sequence labeling）是我们在解决NLP问题时经常遇到的基本问题之一。在序列标注中，我们想对一个序列的每一个元素标注一个标签。一般来说，一个序列指的是一个句子，而一个元素指的是句子中的一个词。比如信息提取问题可以认为是一个序列标注问题，如提取出会议时间、地点等。

### 分词 ( Word Segmentation)

对于英文，显然句子有天然的分词。所以分词通常是针对中文句子。
一个句子中找出词的边界有时并不是一个简单的问题，所以也需要模型来做。一般中文分词是对句子中的每个字的位置做二分类。如果标注为Y，表示这个字是词汇的结尾边界，如果标注为 N，表示这个字不是词汇的结尾。到了下游任务，我们就可以利用分好的词来作为模型输入的基本单位，而不一定用字。

例如 `我爱北京天安门`，分词之后就变成了 `我 爱 北京 天安门`。

### 词性标注 （POS Tagging ）

Part-of-Speech (POS) Tagging 词性标注需要我们标记出分词结果中的每个词的词性是什么，即确定每个词是名词、动词、形容词或其他词性的过程。对应输入一个序列，输出序列每个位置的类别任务。

例如： `I love Beijing Tiananmen Square`，词性标注之后就变成了 `PRP VBP NNP NNP NNP`。

### 命名实体识别  (Named Entity Recognition)

命名实体识别（NER）是信息提取（Information Extraction）的一个子任务，主要涉及如何从文本中提取命名实体并将其分类至事先划定好的类别。命名实体包括人名、地名、机构名、时间、日期、货币、百分比等等。如在招聘信息中提取具体招聘公司、岗位和工作地点的信息，并将其分别归纳至公司、岗位和地点的类别下。命名实体识别往往先将整句拆解为词语并对每个词语进行此行标注，根据习得的规则对词语进行判别。

例如： `I love Beijing Tiananmen Square`，命名实体识别之后就变成了 `I love [LOC Beijing] [LOC Tiananmen Square]`。

### 语义角色标注  (Semantic Role Labeling)

语义角色标注是给句子中的每个词标注语义角色，比如主语、宾语、时间、地点等等。语义角色标注的输入是一个句子，输出是每个词的语义角色。它是一个序列标注任务。

例如： `I love Beijing Tiananmen Square`，语义角色标注之后就变成了 `I [ARG0] love [V] [ARG1] Beijing [LOC] Tiananmen Square [LOC]`。

## 分类任务

文本分类可被用于理解、组织和分类结构化或非结构化文本文档。其涵盖的主要任务有句法分析、情绪分析和垃圾信息检测等。

### 情感分析  (Sentiment Analysis)

句子中包含的正面和负面词汇的数量，以及这些词汇的强度，都会影响到句子的情感倾向。这个任务的输入是一个序列，输出是一个类别。模型需要根据上下文学到语境中更侧重正面还是负面。

### 立场检测  (Stance Detection)

立场检测任务也是分类。它的输入是两个序列，输出是一个类别，表示后面的序列是否与前面的序列站在同一立场。常用的立场检测包括 SDQC 四种标签，支持 (Support)，否定 (Denying)，怀疑 (Querying)，Commenting (注释)。

### 真假预测  (Veracity Prediction)

事实验证也是文本分类的一种。模型需要看一篇新闻文章，判断该文章内容是真的还是假的。假新闻检测是典型的。有时从文章本身，我们人自己都很难判断它的真假。因此有时我们还需要把文章的回复评论也加入模型的输入，去预测真假。如果一个文章它回应第一时间都是否认，往往这个新闻都是假新闻。我们还可以让模型看与文章有关的维基百科的内容，来增强它的事实审核能力。

## 句子关系判断

### 句法分析、蕴含关系判断（entailment）

## 生成式任务

### 文本摘要 (Summarization)

文本摘要，可以分成两种。过去常用的是抽取式摘要。把一篇文档看作是许多句子的组成的序列，模型需要从中找出最能熔炼文章大意的句子提取出来作为输出。它相当于是对每个句子做一个二分类，来决定它要不要放入摘要中。但仅仅把每个句子分开来考虑是不够的。我们需要模型输入整篇文章后，再决定哪个句子更重要。这个序列的基本单位是一个句子的表征。

生成式摘要, 模型的输入是一段长文本，输出是短文本。这个短文本不是原文中的句子，而是模型自己生成的。这个任务的难点在于，模型需要理解原文的含义，然后用自己的语言表达出来。这个任务的输入是一个序列，输出也是一个序列。

### 机器翻译 (Machine Translation )

机器翻译的输入是一段文字，输出也是一段文字。将一种语言的文字转换成另一种语言的文字。

### 阅读理解 Reading Comprehension ）

将输入的文章和问题分别编码，再对其进行解码得到问题的答案。

### 对话系统（Dialogue Systerm）

输入的是一句话，输出是对这句话的回答。

### 问答 (Question Answering)

针对用户提出的问题，系统给出相应的答案。

### 语法错误纠正 (Grammatical Error Correction)

语法错误纠正的输入是一段文字，输出也是一段文字。将一段错误的文字转换成正确的文字。

### 自然语言推理 (Natural Language Inference)

输入给模型的是一个陈述前提，和一个假设，输出是能否通过前提推出假设，它包含三个类别，分别是矛盾，蕴含和中性。

比如前提是，一个人骑在马上跳过一架破旧的飞机，假设是这个人正在吃午餐。这显然是矛盾的。因为前提推不出假设。如果假设是，这个人在户外，在一匹马上。则可以推理出蕴含。再如果假设是这个人正在一个比赛中训练他的马。则推理不能确定，所以是中性的。

文本输入：premise(前提) + hypothesis(假设)

模型输出：对假设是否成立的判断结果，矛盾/包含(可推得)/中立(contradiction/entailment/neutral)

### 搜索引擎 (Search Engine)

模型的输入是一个关键词或一个问句和一堆文章，输出是每篇文章与该问句的相关性。

# NLP基准测试 (NLP Benchmark)

## GLUE (General Language Understanding Evaluation)

GLUE 是通用语言理解评估基准，用于测试模型在广泛自然语言理解任务中的鲁棒性。GLUE中任务分成三大类:

- 第一大类是分类任务，包括语法错误检测和情感分类。它们都是输入是一个序列，输出是一个类别。
- 第二大类是输入是两个句子，输出是二者的语义是否相似对应。
- 第三大类都是自然语言推理相关的任务。输入前提和假设，希望机器能判断二者是否矛盾蕴含还是无关。

这三个大类一共包含 9 个任务， 9 个任务的数据集都是从真实的应用场景中提取出来的，比如问答，自然语言推理，文本蕴含，情感分析等。

## Super GLUE

自从 Bert模型出现之后，GLUE指标都被打破稍微有些超出人类的表现了，所以需要新的 Benchmark，于是就有了 Super GLUE。它包含了 8 个任务，这些任务都是 NLP 领域的难题，比如问答，自然语言推理，文本蕴含，情感分析等。这些任务都是需要模型具备一定的常识推理能力，才能解决的。 SuperGLUE 的任务大都是和 QA 比较有关系的，比如输入一个段落，询问一个一般疑问句，回答是yes or no。或者是常识、逻辑推理。也有是把看一个段落，回答填空的。或者是给机器两个句子，两个句子中都有同样的词汇。看看机器能不能知道这两个词汇意思是一样的还是不是一样的。或者是给机器一个句子，句子上标了一个名词和一个代名词，希望机器能够判断二者是不是指代同一个东西。

## 常见的32项NLP任务以及对应的评测数据、评测指标

|        描述         |                      任务                      |            corpus/dataset            |                 评价指标                  |
| :-----------------: | :--------------------------------------------: | :----------------------------------: | :---------------------------------------: |
|      组块分析       |                    Chunking                    |            Penn Treebank             |                    F1                     |
|      常识推理       |             Common sense reasoning             |              Event2Mind              |               cross-entropy               |
|      句法分析       |                    Parsing                     |            Penn Treebank             |                    F1                     |
|      指代消解       |             Coreference resolution             |              CoNLL 2012              |                average F1                 |
|    依存句法分析     |               Dependency parsing               |            Penn Treebank             |                 POSUASLAS                 |
| 任务型对话/意图识别 |    Task-Oriented Dialogue/Intent Detection     |              ATIS/Snips              |                 accuracy                  |
|  任务型对话/槽填充  |      Task-Oriented Dialogue/Slot Filling       |              ATIS/Snips              |                    F1                     |
| 任务型对话/状态追踪 | Task-Oriented Dialogue/Dialogue State Tracking |                DSTC2                 |            AreaFoodPriceJoint             |
|      领域适配       |               Domain adaptation                |    Multi-Domain Sentiment Dataset    |             average accuracy              |
|      实体链接       |                 Entity Linking                 |           AIDA CoNLL-YAGO            |      Micro-F1-strongMacro-F1-strong       |
|      信息抽取       |             Information Extraction             |              ReVerb45K               |             PrecisionRecallF1             |
|    语法错误纠正     |          Grammatical Error Correction          |                JFLEG                 |                   GLEU                    |
|      语言模型       |               Language modeling                |            Penn Treebank             | Validation perplexity     Test perplexity |
|     词汇规范化      |             Lexical Normalization              |             LexNorm2015              |             F1PrecisionRecall             |
|      机器翻译       |              Machine translation               |            WMT 2014 EN-DE            |                   BLEU                    |
|   多模态情感识别    |         Multimodal Emotion Recognition         |               IEMOCAP                |                 Accuracy                  |
|   多模态隐喻识别    |        Multimodal Metaphor Recognition         | verb-noun pairs adjective-noun pairs |                    F1                     |
|   多模态情感分析    |         Multimodal Sentiment Analysis          |                 MOSI                 |                 Accuracy                  |
|    命名实体识别     |            Named entity recognition            |              CoNLL 2003              |                    F1                     |
|    自然语言推理     |           Natural language inference           |               SciTail                |                 Accuracy                  |
|      词性标注       |             Part-of-speech tagging             |            Penn Treebank             |                 Accuracy                  |
|        问答         |               Question answering               |                CliCR                 |                    F1                     |
|        分词         |               Word segmentation                |              VLSP 2013               |                    F1                     |
|      词义消歧       |           Word Sense Disambiguation            |             SemEval 2015             |                    F1                     |
|      文本分类       |              Text classification               |               AG News                |                Error rate                 |
|        摘要         |                 Summarization                  |               Gigaword               |           ROUGE-1ROUGE-2ROUGE-L           |
|      情感分析       |               Sentiment analysis               |                 IMDb                 |                 Accuracy                  |
|    语义角色标注     |             Semantic role labeling             |              OntoNotes               |                    F1                     |
|      语义解析       |                Semantic parsing                |              LDC2014T12              |            F1 NewswireF1 Full             |
|   语义文本相似度    |          Semantic textual similarity           |               SentEval               |            MRPCSICK-RSICK-ESTS            |
|      关系抽取       |            Relationship Extraction             |        New York Times Corpus         |                P@10%P@30%                 |
|      关系预测       |              Relation Prediction               |                WN18RR                |                H@10H@1MRR                 |

