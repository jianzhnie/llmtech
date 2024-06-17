## 现有的LLM评估框架

评估大型语言模型对于确定其在各种应用中的质量和实用性至关重要。已经开发了几种评估 LLM 的框架，但没有一个框架足够全面，可以涵盖语言理解的所有方面。让我们来看看一些现有的主要评估框架。

## 主要评估基准

|               框架名称               |                         评估考虑因素                         | 网址链接                                                     |
| :----------------------------------: | :----------------------------------------------------------: | ------------------------------------------------------------ |
|              Big Bench               |                           泛化能力                           | https://github.com/google/BIG-bench                          |
|                 BLEU                 | 双语评估测试, 用来衡量机器翻译文本与已经过基准测试的高质量参考翻译的相似度的指标。范围从 0 到 1。 |                                                              |
|            GLUE Benchmark            |     语法、释义、文本相似性、推理、文本蕴涵、解析代词指称     | https://gluebenchmark.com/                                   |
|         SuperGLUE Benchmark          | 自然语言理解、推理、理解训练数据以外的复杂句子、连贯且格式良好的自然语言生成、与人类对话、常识推理（日常场景和社会规范和惯例）、信息检索、阅读理解. 与 GLUE相比更具挑战和多样性。 | https://super.gluebenchmark.com/                             |
|         OpenAI Moderate API          |                    过滤有害或不安全的内容                    | https://platform.openai.com/docs/api-reference/moderations   |
|                 MMLU                 |      使用零样本和单样本设置在跨各种任务和领域的语言理解      | https://github.com/hendrycks/test                            |
|          EleutherAI LM Eval          | 只需进行少量微调，即可在广泛的任务中进行小样本评估和性能测试 | https://github.com/EleutherAI/lm-evaluation-harness          |
|             OpenAI Evals             | 生成文本的准确性、多样性、一致性、稳健性、可转移性、效率、公平性 | https://github.com/openai/evals                              |
|        Adversarial NLI (ANLI)        | 稳健性、泛化、推理的连贯解释、类似示例之间的推理一致性、资源使用效率（内存使用、推理时间和训练时间） | https://github.com/facebookresearch/anli                     |
| LIT (Language Interpretability Tool) | 根据用户定义的指标进行评估的平台。洞察他们的优势、劣势和潜在偏见 | https://pair-code.github.io/lit/                             |
|                ParlAI                | 准确度、F1 分数、困惑度（模型预测序列中下一个单词的准确度）、相关性、流畅度和连贯性等标准的人工评估、速度和资源利用率、稳健性（评估模型在不同条件下的表现，例如噪声输入、对抗性攻击或不同级别的数据质量）、泛化 | https://github.com/facebookresearch/ParlAI                   |
|                 CoQA                 |     理解一段文本并回答对话中出现的一系列相互关联的问题。     | https://stanfordnlp.github.io/coqa/                          |
|               LAMBADA                |          通过预测文章的最后一个单词来进行长期理解。          | https://zenodo.org/record/2630551#.ZFUKS-zML0p               |
|              HellaSwag               |            评估（LLM）完成句子的能力，及推理能力             | https://rowanzellers.com/hellaswag/https://rowanzellers.com/hellaswag/ |
|              TruthfulQA              |                     评估模型响应的真实性                     | https://github.com/sylinrl/TruthfulQA                        |
|                LogiQA                |                         逻辑推理能力                         | https://github.com/lgw863/LogiQA-dataset                     |
|               MultiNLI               |                  理解不同类型句子之间的关系                  | https://cims.nyu.edu/~sbowman/multinli/                      |
|                SQUAD                 | 斯坦福问答数据集，用于评估 LLM 问答任务的数据集， 阅读理解任务， 它包括一组与特定答案相关的上下文段落和相应问题。 | https://rajpurkar.github.io/SQuAD-explorer/                  |



## 评估框架和平台

评估 LLM 以衡量其在不同应用中的质量和有效性至关重要。已经设计了许多专门用于评估 LLM 的框架。下面，我们重点介绍一些最广为人知的框架，例如 Microsoft Azure AI 工作室中的 Prompt Flow、与 LangChain 结合的 Weights & Biases、LangChain 的 LangSmith、confidence-ai 的 DeepEval、TruEra 等。

|                         框架名称                         |                        |                             描述                             | Reference                                                    |
| :------------------------------------------------------: | ---------------------- | :----------------------------------------------------------: | ------------------------------------------------------------ |
|                        LangSmith                         | LangChain              | 帮助用户追踪和评估语言模型应用和智能代理，以帮助用户从原型阶段过渡到生产阶段。 | https://www.langchain.com/langsmith                          |
|                     Vertex AI Studio                     | Google                 | 您可以在 Vertex AI 上评估基础模型和您调整后的生成式人工智能模型的性能。这些模型会使用您提供的评估数据集，并根据一组指标进行评估。 | https://cloud.google.com/vertex-ai?hl=en                     |
|                      Amazon Bedrock                      | Amazon                 | 亚马逊基础服务（Amazon Bedrock）支持模型评估作业。模型评估作业的结果使您能够评估和比较模型的输出，然后选择最适合您下游生成式人工智能应用的模型。模型评估作业支持大型语言模型（LLMs）的常见用例，如文本生成、文本分类、问答以及文本摘要。 | https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html |
|                       Prompt Flow                        | Microsoft              | 这是一套开发工具，旨在简化基于LLM（大型语言模型）的人工智能应用的端到端开发周期，包括从构思、原型设计、测试、评估到生产、部署和监控的各个环节。 | https://github.com/microsoft/promptflow                      |
|                         TruLens                          | TruEra                 | TruLens 提供了一套用于开发和监控神经网络的工具，包括大型语言模型（LLMs）。这包括用于评估大型语言模型及其基于LLM的应用的工具 TruLens-Eval，以及用于深度学习可解释性的 TruLens-Explain。 | https://github.com/truera/trulens                            |
|                         DeepEval                         | Confident AI           |         一个用于LLM应用的开源大型语言模型评估框架。          | https://github.com/confident-ai/deepeval                     |
|                         Parea AI                         | Parea AI               | Parea 帮助人工智能工程师构建可靠、准备投入生产的LLM应用。Parea 提供了用于调试、测试、评估和监控由LLM驱动的应用的工具。 | https://docs.parea.ai/evaluation/overview                    |
|                    EleutherAI LM Eval                    | EleutherAI             |    通过最少的微调在广泛的任务中进行小样本评估和性能测试。    | https://github.com/EleutherAI/lm-evaluation-harness          |
|                       PromptBench                        | Microsoft              |      PromptBench：用于评估和理解大型语言模型的统一库。       | https://github.com/microsoft/promptbench                     |
|                       OpenAI Evals                       | OpenAI                 | Evals是 OpenAI 评估 LLM 的标准框架，也是基准的开源注册表。该框架用于测试 LLM 模型以确保其准确性。 | https://github.com/openai/evals                              |
|                        Promptfoo                         |                        | promptfoo是一个用于评估 LLM 输出质量和性能的 CLI 和库，它使您能够使用预定义的测试系统地测试提示和模型。 | https://github.com/promptfoo/promptfoo                       |
|                        AlpacaEval                        | Stanford               | [AlpacaEval](https://tatsu-lab.github.io/alpaca_eval/)：指令遵循语言模型的自动评估器 | https://github.com/tatsu-lab/alpaca_eval                     |
|                       ChatbotArena                       | Lmsystem               | ChatbotArena以众包方式让不同的大模型产品进行匿名、随机的对抗测评，其评级基于国际象棋等竞技游戏中广泛使用的Elo评分系统，Elo是一种计算玩家相对技能水平的方法，通过两名玩家之间的评分差异可以预测比赛的结果。评分结果通过用户投票产生，系统每次会随机选择两个不同的大模型机器人和用户聊天，并让用户在匿名的情况下选择哪款大模型产品的表现更好一些。 | https://chat.lmsys.org/                                      |
|                   Open LLM Leaderboard                   | HuggingFace            | Open LLM Leaderboard是最大的大模型和数据集社区HuggingFace推出的开源大模型排行榜单，基于EleutherAI Language Model Evaluation Harness（EleutherAI语言模型评估框架）封装。 | https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard |
|                          C-Eval                          | 北大、上交、港大       | C-Eval 是一套全面的中文基础模型评估套件。它覆盖人文，社科，理工，其他专业四个大方向，52 个学科（微积分，线代 …），从中学到大学研究生以及职业考试，一共 13948 道题目的中文知识和推理型测试集，来帮助中文社区研发大模型。 | https://github.com/SJTU-LIT/ceval                            |
|                         LLMEval                          |                        | LLMEval-3聚焦于专业知识能力评测，涵盖哲学、经济学、法学、教育学、文学、历史学、理学、工学、农学、医学、军事学、管理学、艺术学等教育部划定的13个学科门类、50余个二级学科，共计约20W道标准生成式问答题目。 | https://github.com/llmeval/llmeval-3                         |
|                        SuperCLUE                         |                        | 中文通用大模型综合性基准，SuperCLUE，是针对中文可用的通用大模型的一个测评基准。着眼于综合评价大模型的能力，使其能全面地测试大模型的效果，又能考察模型在中文特有任务上的理解和积累，SuperCLUE从三个不同的维度评价模型的能力：基础能力、专业能力和中文特性能力。SuperCLUE的特点包括：多个维度能力考察（3大类，70+子能力）、 | https://github.com/CLUEbenchmark/SuperCLUE                   |
|                         FlagEval                         | 北京智源人工智能研究院 | FlagEval（天秤）由智源研究院将联合多个高校团队打造，是一种采用“能力—任务—指标”三维评测框架的大模型评测平台，旨在提供全面、细致的评测结果。该平台已提供了 30 多种能力、5 种任务和 4 大类指标，共 600 多个维度的全面评测，任务维度包括 22 个主客观评测数据集和 84433 道题目。 | https://github.com/FlagOpen/FlagEval                         |
|                          CMMLU                           |                        | CMMLU是一个综合性的中文评估基准，专门用于评估语言模型在中文语境下的知识和推理能力。CMMLU涵盖了从基础学科到高级专业水平的67个主题。它包括：需要计算和推理的自然科学，需要知识的人文科学和社会科学,以及需要生活常识的中国驾驶规则等。此外，CMMLU中的许多任务具有中国特定的答案，可能在其他地区或语言中并不普遍适用。因此是一个完全中国化的中文测试基准。 | https://github.com/haonan-li/CMMLU                           |
|                           CMB                            |                        |                    CMB: 中文综合医学基准                     | https://github.com/FreedomIntelligence/CMB                   |
|                       OpenCompass                        | 上海人工智能研究院     | OpenCompass 是一个大型语言模型（LLM）评估平台，支持超过100个数据集上的多种模型（如 Llama3、Mistral、InternLM2、GPT-4、LLaMa2、Qwen、GLM、Claude 等）。 | https://github.com/open-compass/opencompass                  |
| [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) | 上海人工智能研究院     | 开源的大型视觉-语言模型（LVLMs）评估工具包，支持 GPT-4v、Gemini、QwenVLPlus 以及50多个 Hugging Face 模型和20多个基准测试。 | https://github.com/open-compass/VLMEvalKit                   |
|                        AgentBench                        | 清华                   | AgentBench是第一个旨在评估LLM-as-Agent在各种不同环境中的表现的基准测试。它涵盖 8 个不同的环境，以更全面地评估 LLM 在各种场景中作为自主代理运行的能力。 | https://github.com/THUDM/AgentBench                          |

## 按应用场景评估指标

在深入研究 LLM 系统的评估指标时，根据应用场景定制标准以确保进行细致入微且针对具体情况的评估至关重要。不同的应用需要与其特定目标和要求相符的不同评分指标。

### 	Summarization

准确、连贯且相关的摘要对于文本摘要至关重要。 下面列出了用于评估 LLM 完成的文本摘要质量的样本指标。

| Metrics type                      | Metric     | Detail                                                       | Reference                                                    |
| --------------------------------- | ---------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Overlap-based metrics             | BLEU       | BLEU score is a precision-based measure, and it ranges from 0 to 1. The closer the value is to 1, the better the prediction. | [Link](https://huggingface.co/spaces/evaluate-metric/bleu)   |
| Overlap-based metrics             | ROUGE      | Recall-Oriented Understudy for Gisting Evaluation is a set of metrics and accompanying software package used for evaluating automatic summarization and machine translation software in natural language processing. | [Link](https://huggingface.co/spaces/evaluate-metric/rouge)  |
| Overlap-based metrics             | ROUGE-N    | Measures the overlap of n-grams (contiguous sequences of n words) between the candidate text and the reference text. It computes precision, recall, and F1 score based on the n-gram overlap. | [Link](https://github.com/google-research/google-research/tree/master/rouge) |
| Overlap-based metrics             | ROUGE-L    | Measures the longest common subsequence (LCS) between the candidate text and the reference text. It computes the precision, recall, and F1 score based on the length of the LCS. | [Link](https://github.com/google-research/google-research/tree/master/rouge) |
| Overlap-based metrics             | METEOR     | An automatic metric for machine translation evaluation that is based on a generalized concept of unigram matching between the machine-produced translation and human-produced reference translations. | [Link](https://huggingface.co/spaces/evaluate-metric/meteor) |
| Semantic similarity-based metrics | BERTScore  | It leverages the pre-trained contextual embeddings from BERT and matches words in candidate and reference sentences by cosine similarity. | [Link](https://huggingface.co/spaces/evaluate-metric/bertscore) |
| Semantic similarity-based metrics | MoverScore | Text Generation Evaluating with Contextualized Embeddings and Earth Mover Distance. | [Link](https://paperswithcode.com/paper/moverscore-text-generation-evaluating-with) |
|                                   |            |                                                              |                                                              |
| Specialized in summarization      | SUPERT     | Unsupervised Multi-Document Summarization Evaluation & Generation. | [Link](https://github.com/danieldeutsch/SUPERT)              |
| Specialized in summarization      | BLANC      | A reference-less metric of summary quality that measures the difference in masked language modeling performance with and without access to the summary. | [Link](https://paperswithcode.com/method/blanc)              |
| Specialized in summarization      | FactCC     | Evaluating the Factual Consistency of Abstractive Text Summarization | [Link](https://github.com/salesforce/factCC)                 |
|                                   |            |                                                              |                                                              |
| Others                            | Perplexity | Perplexity serves as a statistical gauge of a language model's predictive accuracy when analyzing a text sample. Put simply, it measures the level of 'surprise' the model experiences when encountering new data. A lower perplexity value indicates a higher level of prediction accuracy in the model's analysis of the text. | [Link](https://huggingface.co/spaces/evaluate-metric/perplexity) |

### Q&A

为了衡量系统解决用户查询的有效性，下表引入了针对问答场景定制的特定指标，增强了我们在此背景下的评估能力。

| Metrics    | Details                                                      | Reference                                                    |
| ---------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| QAEval     | A question-answering based metric for estimating the content quality of a summary. | [Link](https://github.com/danieldeutsch/sacrerouge/blob/master/doc/metrics/qaeval.md) |
| QAFactEval | QA-Based Factual Consistency Evaluation                      | [Link](https://github.com/salesforce/QAFactEval)             |
| QuestEval  | An NLG metric to assess whether two different inputs contain the same information. It can deal with multimodal and multilingual inputs. | [Link](https://github.com/ThomasScialom/QuestEval)           |

### NER

命名实体识别 (NER) 是识别和分类文本中特定实体的任务。评估 NER 对于确保准确提取信息、提高应用程序性能、改进模型训练、对不同方法进行基准测试以及建立用户对依赖精确实体识别的系统的信心非常重要。

Sample metrics for NER

| Metrics                | Details                                                      | Reference                                                    |
| ---------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Classification metrics | Classification metrics (precision, recall, accuracy, F1 score, etc.) at entity level or model level. | [Link](https://learn.microsoft.com/en-us/azure/ai-services/language-service/custom-named-entity-recognition/concepts/evaluation-metrics) |
| InterpretEval          | The main idea is to divide the data into buckets of entities based on attributes such as entity length, label consistency, entity density, sentence length, etc., and then evaluate the model on each of these buckets separately. | [Code](https://github.com/neulab/InterpretEval), [Article](https://arxiv.org/pdf/2011.06854.pdf) |

### Text-to-SQL

实用的文本到 SQL 系统的有效性取决于它能否熟练地概括广泛的自然语言问题、无缝适应未知的数据库模式以及灵活地适应新的 SQL 查询结构。强大的验证过程在全面评估文本到 SQL 系统中起着关键作用，确保它们不仅在熟悉的场景中表现良好，而且在面对各种语言输入、不熟悉的数据库结构和创新的查询格式时也表现出弹性和准确性。

Benchmarks for text-to-SQL tasks

| Metrics  | Details                                                      | Reference                             |
| -------- | ------------------------------------------------------------ | ------------------------------------- |
| WikiSQL  | The first large compendium of data built for the text-to-SQL use case induced in late 2017. | https://github.com/salesforce/WikiSQL |
| Spider   | A large-scale complex and cross-domain semantic parsing and text-to-SQL dataset. | https://yale-lily.github.io/spider    |
| BIRD-SQL | BIRD (BIg Bench for LaRge-scale Database Grounded Text-to-SQL Evaluation) represents a pioneering, cross-domain dataset that examines the impact of extensive database content on text-to-SQL parsing. | https://bird-bench.github.io/         |
| SParC    | A dataset for cross-domain Semantic Parsing in Context.      | https://yale-lily.github.io/sparc     |

Evaluation metrics for text-to-SQL tasks

| Metrics                       | Details                                                      |
| ----------------------------- | ------------------------------------------------------------ |
| Exact-set-match accuracy (EM) | EM evaluates each clause in a prediction against its corresponding ground truth SQL query. However, a limitation is that there exist numerous diverse ways to articulate SQL queries that serve the same purpose. |
| Execution Accuracy (EX)       | EX evaluates the correctness of generated answers based on the execution results. |
| VES (Valid Efficiency Score)  | A metric to measure the efficiency along with the usual execution correctness of a provided SQL query. |

### 检索系统

RAG（检索增强生成）是一种自然语言处理 (NLP) 模型架构，结合了检索和生成方法的元素。它旨在通过将信息检索技术与文本生成功能相结合来提高语言模型的性能。评估对于评估 RAG 检索相关信息、整合上下文、确保流畅性、避免偏见和满足用户满意度的效果至关重要。它有助于识别优势和劣势，指导检索和生成组件的改进。

Evaluation frameworks for retrieval system

| Evaluation frameworks | Details                                                      | Reference                                                    |
| --------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| RAGAs                 | A framework that helps us evaluate our Retrieval Augmented Generation (RAG) pipeline | [Docs](https://docs.ragas.io/en/latest/), [Code](https://github.com/explodinggradients/ragas) |
| ARES                  | An Automated Evaluation Framework for Retrieval-Augmented Generation Systems | [Link](https://github.com/stanford-futuredata/ARES)          |
| RAG Triad of metrics  | RAG Triad of metrics RAG triad: Answer Relevance (Is the final response useful), Context Relevance (How good is the retrieval), and Groundedness (Is the response supported by the context). Trulens and LLMA index work together for the evaluation. | [DeepLearning.AI Course](https://learn.deeplearning.ai/building-evaluating-advanced-rag) |

 Sample evaluation metrics for retrieval system

| Metrics                    | Details                                                      | Reference                                                    |
| -------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Faithfulness               | Measures the factual consistency of the generated answer against the given context. | [Link](https://docs.ragas.io/en/latest/concepts/metrics/faithfulness.html#equation-faithfulness) |
| Answer relevance           | Focuses on assessing how pertinent the generated answer is to the given prompt. | [Link](https://docs.ragas.io/en/latest/concepts/metrics/answer_relevance.html) |
| Context precision          | Evaluates whether all the ground truth–relevant items present in the contexts are ranked higher or not. | [Link](https://docs.ragas.io/en/latest/concepts/metrics/context_precision.html) |
| Context relevancy          | Measures the relevancy of the retrieved context, calculated based on both the question and contexts. | [Link](https://docs.ragas.io/en/latest/concepts/metrics/context_relevancy.html) |
| Context Recall             | Measures the extent to which the retrieved context aligns with the annotated answer, treated as the ground truth. | [Link](https://docs.ragas.io/en/latest/concepts/metrics/context_recall.html) |
| Answer semantic similarity | Assesses the semantic resemblance between the generated answer and the ground truth. | [Link](https://docs.ragas.io/en/latest/concepts/metrics/semantic_similarity.html) |
| Answer correctness         | Gauges the accuracy of the generated answer when compared to the ground truth. | [Link](https://docs.ragas.io/en/latest/concepts/metrics/answer_correctness.html) |
