

# Deeplearning4j 入门

随着深度学习在语音、图像、自然语言等领域取得了广泛的成功，越来越多的企业、高校和科研单位开始投入大量的资源研发 AI 项目。同时，为了方便广大研发人员快速开发深度学习应用，专注于算法应用本身，避免重复造轮子的问题，各大科技公司先后开源了各自的深度学习框架，例如：TensorFlow（Google）、Torch/PyTorch（Facebook）、Caffe（BVLC）、CNTK（Microsoft）、PaddlePaddle（百度）等。

以上框架基本都是基于 Python 或者 C/C++ 开发的。而且很多基于 Python 的科学计算库，如 NumPy、Pandas 等都可以直接参与数据的建模，非常快捷高效。

然而，对于很多 IT 企业及政府网站，大量的应用都依赖于 Java 生态圈中的开源项目，如 Spring/Structs/Hibernate、Lucene、Elasticsearch、Neo4j 等。主流的分布式计算框架，如 Hadoop、Spark 都运行在 JVM 之上，很多海量数据的存储也基于 Hive、HDFS、HBase 这些存储介质，这是一个不容忽视的事实。

有鉴于此，如果有可以跑在 JVM 上的深度学习框架，那么不光可以方便更多的 Java/JVM 工程师参与到人工智能的浪潮中，更重要的是可以与企业已有的 Java 技术无缝衔接。无论是 Java EE 系统，还是分布式计算框架，都可以与深度学习技术高度集成。Deeplearning4j 正是具备这些特点的深度学习框架。

Eclipse Deeplearning4J (DL4J) 是包含深度学习工具和库的框架，专为充分利用 Java™ 虚拟机 (JVM) 而编写。它具有为 Java 和 Scala 语言编写的分布式深度学习库，并且内置集成了 Apache Hadoop 和 Spark。[Deeplearning4j](https://deeplearning4j.org/) 有助于弥合使用 Python 语言的数据科学家和使用 Java 语言的企业开发人员之间的鸿沟，从而简化了在企业大数据应用程序中部署深度学习的过程。

DL4J 可在分布式 CPU 和图形处理单元 (GPU) 上运行。社区版本和企业版本均已面市。

## Eclipse Deeplearning4J 框架

**Skymind**

Skymind 是总部位于旧金山的人工智能 (AI) 初创企业，由 DL4J 首席开发人员 Adam Gibson 联合他人一起创办。Skymind 销售面向 DL4J 生态系统的商业支持服务和培训服务。此外，该公司的 Skymind Intelligence Layer 平台可填补 Python 应用程序与企业 JVM 之间的空白。

DL4J 是由来自旧金山和东京的一群开源贡献者协作开发的。2014 年末，他们将其发布为 Apache 2.0 许可证下的开源框架。主要是作为一种平台来使用，通过这种平台来部署商用深度学习算法。创立于 2014 年的 Skymind 是 DL4J 的商业支持机构。

2017 年 10 月，Skymind 加入了 Eclipse 基金会，并且将 DL4J 贡献给开源 Java Enterprise Edition 库生态系统。有了 Eclipse 基金会的支持，人们能够更加肯定 DL4J 项目必将得到妥善监管，同时也确保为商业开发提供合适的开源许可证。Java AI 开发人员已将 DL4J 视为成熟且安全的框架，因此，这些新建立的伙伴关系将吸引企业在商业领域使用 DL4J。

### Deeplearning4J Repo

另外，就在今年的 4 月 7 号，Deeplearning4j 发布了最新版本 1.0.0-alpha，该版本的正式发布不仅提供了一系列新功能和模型结构，也意味着整个 Deeplearning4j 项目的趋于稳定和完善。

![enter image description here](https://images.gitbook.cn/9928a020-f2e1-11e8-8d28-f50de28a2376)

Deeplearning4j 提供了对经典神经网络结构的支持，例如：

- 多层感知机/全连接网络（MLP）
- 受限玻尔兹曼机（RBM）
- 卷积神经网络（CNN）及相关操作，如池化（Pooling）、解卷积（Deconvolution）、空洞卷积（Dilated/Atrous Convolution）等
- 循环神经网络（RNN）及其变种，如长短时记忆网络（LSTM）、双向 LSTM（Bi-LSTM）等
- 词/句的分布式表达，如 word2vec/GloVe/doc2vec 等

在最新的 1.0.0-alpha 版本中，Deeplearning4j 在开始支持自动微分机制的同时，也提供了对 TensorFlow 模型的导入，因此在新版本的 Deeplearning4j 中可以支持的网络结构将不再局限于自身框架。

DeepLerning4j 基于数据并行化理论，对分布式建模提供了支持（准确来说是基于参数同步机制的数据并行化，并在 0.9.0 版本后新增了 Gradients Sharing 的机制）。需要注意的是，Deeplearning4j 并没有创建自己的分布式通信框架，对于 CPU/GPU 集群的分布式建模仍然需要依赖 Apache Spark。在早期的版本中，Deeplearning4j 同样支持基于 MapReduce 的 Hadoop，但由于其在模型训练期间 Shuffle 等因素导致的迭代效率低下，加上 Spark 基于内存的数据存储模型的高效性，使得其在最近版本中已经停止了对 Hadoop 的支持。

### 企业大数据应用程序中的深度学习

数据科学家使用 Python 来开发深度学习算法。相比之下，企业大数据应用程序倾向于使用 Java 平台。因此，为填补缺口并在大数据应用程序中部署深度学习，DL4J 的开发人员必须对若干解决方案进行创新。

[Keras](https://keras.io/) 应用程序编程接口 (API) 规范的采用，有助于从其他框架（例如，TensorFlow、Caffe、Microsoft® Cognitive Toolkit (CNTK) 和 Theano）导入深度学习模型。Keras API 可通过 JVM 语言（例如，Java、Scala、Clojure 乃至 Kotlin）来访问，从而使深度学习模型可供 Java 开发人员使用。

在 JVM 上运行深度学习高性能计算负载时，将面临着诸多挑战。内存管理和垃圾回收等 Java 功能可能会影响性能，使用较大内存时尤为如此。DL4J 绕过了其中部分限制。

### Deeplearning4j 生态圈

Deeplearning4j 生态圈中除了深度神经网络这个核心框架以外，还包括像 DataVec、ND4J、RL4J 等一些非常实用的子项目，下面就对这些子项目的主要功能模块做下介绍。

- [Deeplearning4j 库](#deeplearning4j-库)
- [ND4J 库](#nd4j-库)
- [Datavec 库](#datavec-库)
- [libnd4j 库](#libnd4j-库)
- [RL4J 库](#rl4j)
- [Jumpy 库](#jumpy-库)
- [Arbiter 库](#arbiter-库)

#### Deeplearning4j 库

Deeplearning4j 库实际上是神经网络平台。它包含各种工具，用于配置神经网络和构建计算图形。开发人员使用此库来构建由数据管道和 Spark 集成的神经网络模型。

除核心库外，DL4J 库还包含许多其他库，用于实现特定功能：

- **deeplearning4j-core。** deeplearning4j-core 库包含了运行 DL4J 所需的全部功能，例如，用户界面 (UI)。它还具有构建神经网络所需的各种工具和实用程序。
- **deeplearning4j-cuda。** deeplearning4j-cuda 库支持 DL4J 在使用 NVIDIA CUDA® 深度神经网络库 (CuDNN) 的 GPU 上运行。此库支持标准化以及卷积神经网络和递归神经网络。
- **deeplearning4j-graph。** deeplearning4j-graph 库执行图形处理来构建 DeepWalk 中所使用的图形矢量化模型，DeepWalk 是一个无监督学习算法，用于学习图形中每个顶点的矢量表示法。您可以使用这些学到的矢量表示法对图形中的相似数据进行分类、分群或搜索。
- **deeplearning4j-modelimport。** deeplearning4j-modelimport 库从 Keras 导入模型，Keras 又可从 Theano、TensorFlow、Caffe 和 CNTK 导入模型。这是关键的 DL4J 库，支持将模型从其他框架导入 DL4J。
- **deeplearning4j-nlp-parent。** deeplearning4j-nlp-parent 库支持将 DL4J 与外部自然语言处理 (NLP) 插件和工具相集成。此接口遵循非结构化信息管理架构 (UIMA)，后者最初是由 IBM 开发的用于内容分析的开放标准。此库还包含适用于英语、中文、日语和韩语的文本分析。
- **deeplearning4j-nlp。** deeplearning4j-nlp 库是 NLP 工具（如 Word2Vec 和 Doc2Vec）的集合。Word2Vec 是一个用于处理文本的双层神经网络。Word2Vec 对于在“矢量空间”中对相似词语矢量进行分组十分有用。Doc2Vec 是 Word2Vec 的一个扩展，用于学习将标签与词语相关联，而不是将不同词语关联起来。
- **deeplearning4j-nn。** deeplearning4j-nn 库是核心库的精简版本，减少了依赖关系。它使用构建器模式来设置超参数，同时配置多层网络，支持使用设计模式在 Java 中构造神经网络。
- **deeplearning4j-scaleout。** deeplearning4j-scaleout 库是各种库的集合，适用于配备 Amazon Web Services 服务器以及封装 Spark 并行代码，以便在多达 96 核的常规服务器（而不是 Spark）上运行。此库还有助于在 Spark 以及包含 Kafka 和其他视频分析流选项的 Spark 上配置 NLP。
- **deeplearning4j-ui-parent。** deeplearning4j-ui-parent 库实际上是 DL4J UI，包含神经网络训练启发式方法和可视化工具。



#### ND4J 库

N-Dimensional Arrays for Java (ND4J) 是科学计算 `C++` 库，类似于 Python 的 NumPy， 是Deeplearning4j 所依赖的张量运算框架，ND4J 提供上层张量运算的各种接口。它支持 JVM 上运行的多种语言，例如，Java、Scala、Clojure 和 Kotlin。您可以使用 ND4J 来执行线性代数或操作矩阵。ND4J 可与 Hadoop 或 Spark 进行集成并由此实现扩展，同时可在分布式 CPU 和 GPU 上运行。

Java AI 开发人员可以使用 ND4J 在 Java 中定义 *N* 维数组，这使其能够在 JVM 上执行张量运算。ND4J 使用 JVM 外部的“堆外”内存来存储张量。JVM 仅保存指向此外部内存的指针，Java 程序通过 Java 本机接口 (JNI) 将这些指针传递至 ND4J `C++` 后端代码。此结构配合来自本机代码（例如，基本线性代数子程序 (BLAS) 和 CUDA 库）的张量使用时，可提供更佳的性能。ND4J 与 Spark 集成，并且可使用不同后端在 CPU 或 GPU 上运行。Scala API ND4S 有助于实现这种集成。在稍后部分中讨论 DL4J 如何使用硬件加速时，将再次探讨 ND4J 架构。

ND4J是基于高度优化的C++代码库LIbND4J，它提供了诸如OpenBLAS、OneDNN（MKL DNN）、CUDNN、CuBLAS等库的支持 CPU（AVX2／512）和GPU（CUDA）支持和加速。

#### SameDiff

作为ND4J库的一部分，SameDiff 是DL4J的自动微分框架。SameDiff 使用了基于图的（定义然后运行）方法，类似于TensorFlow图模式。对 Eager graph（TensorFlow 2.x Eager/PyTorch）的支持正在计划中。SameDiff 支持导入TensorFlow冻结模型格式.pb（protobuf）模型。导入 ONNX、TensorFlow SavedModel和Keras模型正在计划中。Deeplearning4j还具有完整的SameDiff支持，可以方便地编写自定义层和丢失函数。

#### Datavec 库

DataVec 库是数据预处理的框架，该框架提供对一些典型非结构化数据（语音、图像、文本）的读取和预处理（归一化、正则化等特征工程常用的处理方法）。此外，对于一些常用的数据格式，如 JSON/XML/MAT（MATLAB 数据格式）/LIBSVM 也都提供了直接或间接的支持。

- 图像处理： 读取、保存常见格式的图片， 灰度化，翻转、裁剪等常见操作(backend: opencv)
- 文本处理：分词、词频统计、停用词过滤
- 音频处理：加窗分帧、采样、FFT 变换等

#### libnd4j 库

Libnd4j 是一个纯 `C++` 库，支持 ND4J 访问属于 BLAS 和 Intel Math Kernel Library 的线性代数函数。它与 JavaCPP 开源库紧密结合运行。JavaCPP 不属于 DL4J 框架项目，但支持此代码的开发人员是相同的。而 LibND4J 用于适配底层基于 C++/Fortran 的张量运算库，如 OpenBLAS、MKL 等。

#### Jumpy 库

Jumpy 是一个 Python 库，支持 NumPy 无需移动数据即可使用 ND4J 库。此库实现了针对 NumPy 和 [Pyjnius](https://pyjnius.readthedocs.io/en/latest/) 的包装器。MLlib 或 PySpark 开发人员可以使用 Jumpy，以便在 JVM 上使用 NumPy 数组。

#### RL4J

这是基于 Java/JVM 的深度强化学习框架，它提供了对大部分基于 Value-Based 强化学习算法的支持，具体有：Deep Q-leaning/Dual DQN、A3C、Async NStepQLearning。

#### Arbiter 库

DL4J 使用此工具来自动调优神经网络。可使用诸如网格搜索、随机搜索和贝叶斯方法之类的各种方法来优化具有许多超参数的复杂模型，从而提高性能。

#### dl4j-examples

这是 Deeplearning4j 核心功能的一些常见使用案例，包括经典神经网络结构的一些单机版本的应用，与 Apache Spark 结合的分布式建模的例子，基于 GPU 的模型训练的案例以及自定义损失函数、激活函数等方便开发者需求的例子。

#### dl4j-model-zoo

顾名思义这个框架实现了一些常用的网络结构，例如：

- ImageNet 比赛中获奖的一些网络结构 AlexNet/GoogLeNet/VGG/ResNet；
- 人脸识别的一些网络结构 FaceNet/DeepFace；
- 目标检测的网络结构 Tiny YOLO/YOLO9000。

在最近 Release 的一些版本中，dl4j-model-z 已经不再作为单独的项目，而是被纳入 Deeplearning4j 核心框架中，成为其中一个模块。

#### ScalNet

这是 Deeplearning4j 的 Scala 版本，主要是对神经网络框架部分基于 Scala 语言的封装。

## Deeplearning4J 的优势

DL4J 具有众多优势。让我们来了解一下其中的三大优势。

### Python 可与 Java、Scala、Clojure 和 Kotlin 实现互操作

Python 为数据科学家所广泛采用，而大数据编程人员则在 Hadoop 和 Spark 上使用 Java 或 Scala 来开展工作。DL4J 填补了之间的鸿沟，开发人员因而能够在 Python 与 JVM 语言（例如，Java、Scala、Clojure 和 Kotlin）之间迁移。

通过使用 [Keras API](https://keras.io/)，DL4J 支持从其他框架（例如，TensorFlow、Caffe、Theano 和 CNTK）迁移深度学习模型。甚至有人建议将 DL4J 作为 Keras 官方贡献的后端之一。

### 分布式处理

DL4J 可在最新分布式计算平台（例如，Hadoop 和 Spark）上运行，并且可使用分布式 CPU 或 GPU 实现加速。通过使用多个 GPU，DL4J 可以实现与 Caffe 相媲美的性能。DL4J 也可以在许多云计算平台（包括 IBM Cloud）上运行。

### 并行处理

DL4J 包含单线程选项和分布式多线程选项。这种减少迭代次数的方法可在集群中并行训练多个神经网络。因此，DL4J 非常适合使用微服务架构来设计应用程序。

## Eclipse DL4J 应用程序

DL4J 具有各种应用程序，从图形处理到 NLP，皆涵盖在内。借助内置数据预处理和矢量化工具，DL4J 能够异常灵活地处理许多不同的数据格式。Keras API 还使其更便于使用来自其他框架的预先训练模型。

典型的 DL4J 应用程序包括：

- 安全性应用程序，例如欺诈检测和网络入侵检测
- 在客户关系管理、广告推送和客户忠诚度及维系方面使用的推荐系统
- 面向物联网和其他流数据的回归和预测分析
- 传统面部识别和图像识别应用程序
- 语音搜索和语音转录应用程序
- 针对硬件或工业应用程序的预防性诊断和异常检测

## 支持 DL4J 的平台

理论上，任何支持 JVM 且运行 Java V1.7 或更高版本的 64 位平台均可支持 DL4J。DL4J 使用 ND4J 来访问受支持的 GPU。ND4J 则依赖于其他软件，例如 CUDA 和 CuDNN。

实际上，商用 DL4J 需要生产级 Java 平台。IBM Open Platform for Apache Hadoop and Apache Spark 已在 PowerAI 上完成了 DL4J 框架认证。Cloudera 也已在其 Enterprise Data Hub CDH5 上完成了 DL4J 认证，Hortonworks 同样也在其 Data Platform\HDP2.4 上完成了认证。

有关安装 ND4J 和其他必备软件的更多信息，请查看 ND4J 入门文档。

### 从源代码构建 DL4J

在其他框架中，使用预先构建的二进制文件来安装框架有时更为方便。但对于 DL4J，最好是从源代码构建和安装 DL4J，以便确保正确处理多种依赖关系。安装为多步骤的复杂过程，适合生产级安装。Apache Maven V3.3.9 或更高版本可对构建和依赖关系进行管理。对于集成开发环境，可以使用 IntelliJ IDEA 或 Eclipse。

专家建议使用“uber-jar”方法来构建 DL4J 应用程序。此方法支持使用 .rpm 或 .deb 包在“uber-jar”内部分发依赖关系，并将部署与开发隔离开来。

addLayers有关使用 DL4J 构建应用程序的更多信息，请阅读[快速入门指南](https://deeplearning4j.org/docs/latest/deeplearning4j-quickstart)。

### Power 架构上的 DL4J

DL4J 团队官方在 DL4J 存储库中为 IBM Power® 维护 DL4J。Maven 负责管理构建和安装过程。ND4J 具有两个用于 Power 架构的本机平台后端：第一个在 POWER8 CPU 上运行，第二个则使用 NVLink 互连在 NVIDIA GPU 上运行。

ND4J CPU 后端 `nd4j-native-platform` 运行 OpenBLAS 库的优化版本。为实现加速，ND4J CPU 后端在 POWER8 处理器上使用矢量/标量浮点单元。

ND4J GPU 后端 `nd4j-cuda-8.0-platform` 可在使用 NVLink 接口与 POWER8 处理器互连的 NVIDIA GPU 上运行 CUDA 或 cuDNN。

有关在 Power 架构上安装 DL4J 的更多信息，请访问 https://deeplearning4j.org/。

**注：**在 POWER8 平台上，首选的 Java 级别为 V8。为 POWER8 而调优的唯一 Java 7 发行版为 Java 7.1 SR1 或更高版本。



## DL4J 如何使用硬件加速？

#### ND4J 加速张量运算

JVM 的执行速度一直为人所诟病。虽然 Hotspot 机制可以将一些对运行效率有影响的代码编译成 Native Code，从而在一定程度上加速 Java 程序的执行速度，但毕竟无法优化所有的逻辑。另外，Garbage Collector（GC）在帮助程序员管理内存的同时，其实也束缚了程序员的手脚，毕竟是否需要垃圾回收并不是程序员说了算；而在其他语言如 C/C++ 中，我们可以 free 掉内存块。

对于机器学习/深度学习来说，优化迭代的过程往往非常耗时，也非常耗资源，因此尽可能地加速迭代过程十分重要。运算速度也往往成为评价一个开源库质量高低的指标之一。鉴于 JVM 自身的局限性，Deeplearning4j 的张量运算通过 ND4J 在堆外内存（Off-Heap Memory/Direct Memory）上进行，[ 详见这里](https://deeplearning4j.org/memory)。大量的张量运算可以依赖底层的 BLAS 库（如 OpenBLAS、Intel MKL），如果用 GPU 的话，则会依赖 CUDA/cuBLAS。由于这些 BLAS 库多数由 Fortran 或 C/C++ 写成，且经过了细致地优化，因此可以大大提高张量运算的速度。对于这些张量对象，在堆上内存（On-Heap Memory）仅存储一个指针/引用对象，这样的处理也大大减少了堆上内存的使用。

DL4J 分别依靠 ND4J 和 ND4S 这两种特定于平台的后端来使用硬件加速。ND4J 和 ND4S Java 与 Scala API 用于封装 BLAS 库，例如，Jblas、Netlib-blas 和 Jcublas。

ND4J 具有两种级别的运算。高级运算包括卷积、快速傅立叶变换、各种损失函数、变换（例如，sigmoid 变换或 tanh 变换）和约简。通过 ND4J API 调用时，BLAS 会实施低级运算。BLAS 运算包括矢量加法、标量乘法、点乘、线性组合和矩阵乘法。

ND4J 在特定于平台架构的后端上运行。CPU 后端带有 *nd4j-native* 前缀，GPU 后端带有 *nd4j-cuda-* 前缀，其中 CUDA 版本可能是 7.5 或 8.0 等。无论使用何种后端，ND4J API 都保持不变。

最后，ND4J 实现特定于 BLAS 的数据缓冲区，用于存储 BLAS 处理的数组和原始数据字节。根据后端，此存储抽象层具有不同的实现。JVM 与后端之间通过 JNI 进行通信。



## 学习 Deeplearning4j

在引言中我们谈到，目前开源的深度学习框架有很多，那么选择一个适合工程师自己、同时也可以达到团队业务要求的框架就非常重要了。在这个部分中，我们将从 接口设计与学习成本，和其他开源库的兼容性等几个方面，给出 Deeplearning4j 这个开源框架的特点及使用场景。

#### High-Level 的接口设计

对于大多数开发者而言，开源库的学习成本是一个不可回避的问题。如果一个开源库可以拥有友好的接口、详细的文档和案例，那么无疑是容易受人青睐的。这一点对于初学者或者致力于转型 AI 的工程师尤为重要。

神经网络不同于其他传统模型，其结构与复杂度变化很多。虽然在大多数场景下，我们会参考经典的网络结构，如 GoogLeNet、ResNet 等。但自定义的部分也会很多，往往和业务场景结合得更紧密。为了使工程师可以快速建模，专注于业务本身和调优，Deeplearning4j 对常见的神经网络结构做了高度封装。以下是声明卷积层 + 池化层的代码示例，以及和 Keras 的对比。

**Keras 版本**

```python
model = Sequential()
model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1]))) #卷积层
model.add(Activation('relu')) #非线性变换
model.add(MaxPooling2D(pool_size=pool_size)) #池化层
```

**Deeplearning4j 版本**

```python
MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
//...
.layer(0, new ConvolutionLayer.Builder(5, 5)    //卷积层
                        .nIn(nChannels)
                        .stride(1, 1)
                        .nOut(20)
                        .activation(Activation.RELU)    //非线性激活函数
                        .build())
.layer(1, new SubsamplingLayer.Builder(PoolingType.MAX)    //最大池化
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
```

可以看到，Deeplearning4j 和 Keras 很相似，都是以 Layer 为基本模块进行建模，这样的方式相对基于 OP 的建模方式更加简洁和清晰。所有的模块都做成可插拔的，非常灵活。以 Layer 为单位进行建模，使得模型的整个结构高度层次化，对于刚接触深度神经网络的开发人员，可以说是一目了然。除了卷积、池化等基本操作，激活函数、参数初始化分布、学习率、正则化项、Dropout 等 trick，都可以在配置 Layer 的时候进行声明并按需要组合。

当然基于 Layer 进行建模也有一些缺点，比如当用户需要自定义 Layer、激活函数等场景时，就需要自己继承相关的基类并实现相应的方法。不过这些在官网的例子中已经有一些参考的 Demo，如果确实需要，开发人员可以参考相关的例子进行设计，[详见这里](https://github.com/deeplearning4j/dl4j-examples/tree/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/misc)。

此外，将程序移植到 GPU 或并行计算上的操作也非常简单。如果用户需要在 GPU 上加速建模的过程，只需要加入以下逻辑声明 CUDA 的环境实例即可。

**CUDA 实例声明**

```python
CudaEnvironment.getInstance().getConfiguration()
            .allowMultiGPU(true)
            .setMaximumDeviceCache(10L * 1024L * 1024L * 1024L)
            .allowCrossDeviceAccess(true);
```

ND4J 的后台检测程序会自动检测声明的运算后台，如果没有声明 CUDA 的环境实例，则会默认选择 CPU 进行计算。

如果用户需要在 CPU/GPU 上进行并行计算，则只需要声明参数服务器的实例，配置一些必要参数即可。整体的和单机版的代码基本相同。

**参数服务器声明**

```python
ParallelWrapper wrapper = new ParallelWrapper.Builder(model)
            .prefetchBuffer(24)
            .workers(8)
            .averagingFrequency(3)
            .reportScoreAfterAveraging(true)
            .useLegacyAveraging(true)
            .build();
```

参数服务器的相关参数配置在后续的课程中会有介绍，这里就不再详细说明了。

#### 友好的可视化页面

为了方便研发人员直观地了解神经网络的结构以及训练过程中参数的变化，Deeplearning4j 提供了可视化页面来辅助开发。需要注意的是，如果需要使用 Deeplearning4j 的可视化功能，需要 JDK 1.8 以上的支持，同时要添加相应的依赖：

```
<dependency>
    <groupId>org.deeplearning4j</groupId>
    <artifactId>deeplearning4j-ui_${scala.binary.version}</artifactId>
    <version>${dl4j.version}</version>
</dependency>
```

并且在代码中添加以下逻辑：

```
//添加可视化页面监听器
UIServer uiServer = UIServer.getInstance();
StatsStorage statsStorage = new InMemoryStatsStorage();
uiServer.attach(statsStorage);
model.setListeners(new StatsListener(statsStorage));
复制
```

训练开始后，在浏览器中访问本地 9000 端口，默认会跳转到 Overview 的概览页面。我们可以依次选择查看网络结构的页面（Model 页面）和系统页面（System），从而查看当前训练的模型以及系统资源的使用情况。详情见下图：

![enter image description here](https://images.gitbook.cn/94aab060-f9f8-11e8-98b8-21d1b727d9e8)

![enter image description here](https://images.gitbook.cn/a87dc320-f9f8-11e8-98b8-21d1b727d9e8)

![enter image description here](https://images.gitbook.cn/72fb5640-f9f8-11e8-98b8-21d1b727d9e8)

Overview 页面主要会记录模型在迭代过程中 Loss 收敛的情况，以及记录参数和梯度的变化情况。根据这些信息，我们可以判断模型是否在正常地学习。从 Model 页面，我们则可以直观地看到目前网络的结构，比如像图中有 2 个卷积层 + 2 个池化层 + 1 个全连接层。而在 System 页面中我们可以看到内存的使用情况，包括堆上内存和堆外内存。

Deeplearning4j 提供的训练可视化页面除了可以直观地看到当前模型的训练状态，也可以基于这些信息进行模型的调优，具体的方法在后续课程中我们会单独进行说明。

#### 兼容其他开源框架

正如在引言中提到的，现在深度神经网络的开源库非常多，每个框架都有自己相对擅长的结构，而且对于开发人员，熟悉的框架也不一定都相同。因此如果能做到兼容其他框架，那么无疑会提供更多的解决方案。

Deeplearning4j 支持导入 Keras 的模型（在目前的 1.0.0-alpha 版本中，同时支持 Keras 1.x/2.x）以及 TensorFlow 的模型。由于 Keras 自身可以选择 TensorFlow、Theano、CNTK 作为计算后台，再加上第三方库支持导入 Caffe 的模型到 Keras，因此 Keras 已经可以作为一个“胶水”框架，成为 Deeplearning4j 兼容其他框架的一个接口。

![enter image description here](https://images.gitbook.cn/4eb24900-f9f9-11e8-98b8-21d1b727d9e8)

兼容其他框架（准确来说是支持导入其他框架的模型）的好处有很多，比如说：

- 扩展 Model Zoo 中支持的模型结构，跟踪最新的成果；
- 离线训练和在线预测的开发工作可以独立于框架进行，减少团队间工作的耦合。

#### Java 生态圈助力应用的落地

Deeplearning4j 是跑在 JVM 上的深度学习框架，源码主要是由 Java 写成，这样设计的直接好处是，可以借助庞大的 Java 生态圈，加快各种应用的开发和落地。Java 生态圈无论是在 Web 应用开发，还是大数据存储和计算都有着企业级应用的开源项目，例如我们熟知的 SSH 框架、Hadoop 生态圈等。

Deeplearning4j 可以和这些项目进行有机结合，无论是在分布式框架上（Apache Spark/Flink）进行深度学习的建模，还是基于 SSH 框架的模型上线以及在线预测，都可以非常方便地将应用落地。下面这两张图就是 Deeplearning4j + Tomcat + JSP 做的一个简单的在线图片分类的应用。

![enter image description here](https://images.gitbook.cn/80e27da0-f9f9-11e8-98b8-21d1b727d9e8) ![enter image description here](https://images.gitbook.cn/87a5c0c0-f9f9-11e8-98b8-21d1b727d9e8)

总结来说，至少有以下 4 种场景可以考虑使用 Deeplearning4j：

- 如果你身边的系统多数基于 JVM，那么 Deeplearning4j 是你的一个选择；
- 如果你需要在 Spark 上进行分布式深度神经网络的训练，那么 Deeplearning4j 可以帮你做到；
- 如果你需要在多 GPU/GPU 集群上加快建模速度，那么 Deeplearning4j 也同样可以支持；
- 如果你需要在 Android 移动端加入 AI 技术，那么 Deeplearning4j 可能是你最方便的选择之一。

以上四点，不仅仅是 Deeplearning4j 自身的特性，也是一些 AI 工程师选择它的理由。

虽然 Deeplearning4j 并不是 GitHub 上 Fork 或者 Star 最多的深度学习框架，但这并不妨碍其成为 AI 工程师的一种选择。就 Skymind 官方发布的信息看，在美国有像 IBM、埃森哲、NASA 喷气推进实验室等多家明星企业和实验机构，在使用 Deeplearning4j 或者其生态圈中的项目，如 ND4J。算法团队结合自身的实际情况选择合适的框架，在多数时候可以做到事半功倍。

### Deeplearning4j 的最新进展

在这里，我们主要介绍下 Deeplearning4j 最新的一些进展情况。

#### 版本

目前 Deeplearning4j 已经来到了 1.0.0-beta3 的阶段，马上也要发布正式的 1.0.0 版本。本课程我们主要围绕 0.8.0 和 1.0.0-alpha 展开（1.0.0-beta3 核心功能部分升级不大），这里罗列下从 0.7.0 版本到 1.0.0-alpha 版本主要新增的几个功能点：

- Spark 2.x 的支持（>0.8.0）
- 支持迁移学习（>0.8.0）
- 内存优化策略 Workspace 的引入（>0.9.0）
- 增加基于梯度共享（Gradients Sharing）策略的并行化训练方式（>0.9.0）
- LSTM 结构增加 cuDNN 的支持（>0.9.0）
- 自动微分机制的支持，并支持导入 TensorFlow 模型（>1.0.0-alpha）
- YOLO9000 模型的支持（>1.0.0-aplpha）
- CUDA 9.0 的支持（>1.0.0-aplpha）
- Keras 2.x 模型导入的支持（>1.0.0-alpha）
- 增加卷积、池化等操作的 3D 版本（>1.0.0-beta）

除此之外，在已经提及的 Issue 上，已经考虑在 1.0.0 正式版本中增加对 YOLOv3、GAN、MobileNet、ShiftNet 等成果的支持，进一步丰富 Model Zoo 的直接支持范围，满足更多开发者的需求。详见 [GAN](https://github.com/deeplearning4j/deeplearning4j/issues/5005)、[MobileNet](https://github.com/deeplearning4j/deeplearning4j/issues/4995)、[YOLOv3](https://github.com/deeplearning4j/deeplearning4j/issues/4986)、[ShiftNet](https://github.com/deeplearning4j/deeplearning4j/issues/5144)。

进一步的进展情况，可以直接跟进每次的 [releasenotes](https://deeplearning4j.org/releasenotes)，查看官方公布的新特性和已经修复的 Bug 情况。

#### 社区

Deeplearning4j 社区目前正在进一步建设和完善中，在社区官网上除了介绍 Deeplearning4j 的基本信息以外，还提供了大量有关神经网络理论的资料，方便相关研发人员的入门与进阶。Deeplearning4j 社区在 Gitter 上同时开通了英文/中文/日文/韩文频道，开发人员可以和核心源码提交者进行快速的交流以及获取最新的信息。

![enter image description here](https://images.gitbook.cn/c3335270-f2f5-11e8-8d28-f50de28a2376)

- Deeplearning4j 的 GitHub 地址，[详见这里](https://github.com/deeplearning4j)；
- Deeplearning4j 社区官网，[详见这里](https://deeplearning4j.org/)；
- Deeplearning4j 英文 Gitter Channel，[详见这里](https://gitter.im/deeplearning4j/deeplearning4j)；
- Deeplearning4j 中文 Gitter Channel，[详见这里](https://gitter.im/deeplearning4j/deeplearning4j/deeplearning4j-cn)；
- Deeplearning4j 官方 QQ 群，**289058486**。

## 结束语

Deeplearning4j 是深度学习工具和库框架，专为充分利用 Java 虚拟机而编写。DL4J 框架旨在用于在生产级服务器上部署商用深度学习算法。它具有为 Java 和 Scala 而编写的分布式深度学习库，并且内置了与 Hadoop 和 Spark 的集成。DL4J 可在分布式 CPU 和 GPU 上运行，提供了社区版本和企业版本。Skymind 作为其商业支持机构，为企业版本提供专业的服务和支持。
