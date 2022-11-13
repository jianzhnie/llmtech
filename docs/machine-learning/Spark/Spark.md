# High Level Overview of Apache Spark

Spark是用于大规模数据处理的集群计算框架。 Spark为统一计算引擎提供了3种语言（Java，Scala和Python）丰富的算法库。

Unified：借助Spark，无需将多个API或系统中的应用程序组合在一起。 Spark为您提供了足够的内置API来完成工作。

Computing Engine：Spark加载来自各种文件系统的数据并在其上运行计算，但不会永久存储任何数据本身。 Spark完全在内存中运行-拥有无与伦比的性能和速度。

Libraries：Spark由为数据科学任务而构建的一系列库组成。 包括用于SQL（SparkSQL），机器学习（MLlib），流处理（Spark Streaming and Structured Streaming）和图分析（GraphX）的库。

## Spark Application

每个Spark Application 程序都包含一个驱动程序（**Driver** ）和一组分布式工作进程（**Executors**）。

## Spark Driver

驱动程序运行我们应用程序的main（）方法，并在其中创建SparkContext。 Spark驱动程序具有以下职责：

- 在集群中的节点上或客户端上运行，并通过集群管理器调度作业执行
- 对用户的程序或输入做出响应
- 分析，编排和分配各个 exectors 的工作
- 存储正在运行的应用程序的 metadata ，并在WebUI中显示

## Spark Executors

Executors 是负责执行任务的分布式过程。每个Spark应用程序都有其自己的 Executors，这些执行程序在单个Spark应用程序的生命周期内保持有效。

- Executor 执行Spark作业的所有数据处理
- 将结果存储在内存中，仅在驱动程序有特别指示时才在磁盘上进行持久化
- 计算完成后，将结果返回给驱动程序
- 每个节点可以具有从每个节点1个执行程序 到 每个核心1个执行程序

![img](https://hackernoon.com/hn-images/1*GZG2aogNS8Jg14jOM2rjmQ.png)



## Spark Application工作流程

当您将作业提交给Spark进行处理时，幕后还有很多事情要做。

- 我们的 Standalone Application 启动，并初始化其SparkContext。 只有拥有SparkContext之后，才能将应用程序称为驱动程序
- 我们的驱动程序（Driver program）要求集群管理器提供启动其执行程序（executors）的资源
- 集群管理器启动执行程序（executors）
- 我们的驱动程序（ Driver）运行我们的实际Spark代码
- 执行程序（Executors）运行任务并将其结果发送回驱动程序（driver）
- SparkContext 停止并且所有执行程序都已关闭，将资源返回给集群

## MaxTemperature, Revisited

让我们更深入地看一下我们编写的Spark Job，以按国家/地区查找最高温度。 这种抽象隐藏了许多设置代码，包括我们的SparkContext的初始化，可以弥补这些空白：

![img](https://hackernoon.com/hn-images/1*8COf1mt_AxIt66ft83z-yg.png)

请记住，Spark是一个框架，上面的应用程序是用Java实现的。 直到第16行，Spark才需要做任何工作。 当然，我们已经初始化了SparkContext，但是将数据加载到RDD中是将工作发送给executors的第一步。

到现在为止，您可能已经看到“ RDD”一词出现了多次，现在是时候定义它了。

## Spark架构概述

Spark具有定义明确的分层架构，其基于两个主要抽象的组件具有松散耦合：

- 弹性分布式数据集（RDD）
- 有向无环图（DAG）

![img](https://miro.medium.com/max/60/1*Ck-GUoBRx7b8rxrgQlNEGQ.png?q=20)

![img](https://miro.medium.com/max/1312/1*Ck-GUoBRx7b8rxrgQlNEGQ.png)

From PySpark-Pictures by Jeffrey Thompson.

## RDD (弹性分布式数据集)

RDD是可以并行操作的元素的容错集合。 您可以创建它们来并行化驱动程序中的现有集合，或在外部存储系统（例如共享文件系统，HDFS，HBase或提供Hadoop InputFormat的任何数据源）中引用数据集。

![img](https://miro.medium.com/max/2457/1*fN_Pj-NmrBFbxNVYexzYoQ.png)

RDD本质上是Spark的构建块：一切都由它们组成。 甚至Sparks更高级别的API（DataFrame，Dataset）也由RDD组成。

Resilient (弹性)：由于Spark在集群上运行，硬件故障造成的数据丢失是一个非常现实的问题，因此RDD具有容错能力，并且在发生故障时可以自行重建.

Distributed (分布式)：单个RDD存储在群集中的一系列不同节点上，不属于单个源（也不属于单个故障点）。 这样我们的集群可以并行地在RDD上运行

Dataset (数据集)：值的集合。

我们在Spark中使用的所有数据都将存储在某种形式的RDD中，因此必须完全理解它们。

Spark提供了一系列“更高级别”的API，这些API构建在RDD之上，旨在抽象掉复杂性，即 DataFrame 和 Dataset。Spark Submit和Scala和Python中的Spark Shell非常关注Read-Evaluate-Print循环（repl），它们的目标是数据科学家，他们通常希望对数据集进行重复分析。RDD仍然需要理解，因为它是Spark中所有数据的底层结构。

RDD在口语上等同于：“分布式数据结构”。 JavaRDD <String>本质上只是一个List <String>，分布在我们集群中的每个节点之间，每个节点都获得 List 的几个不同块。 使用Spark，我们需要始终在分布式环境中进行思考。

RDD的工作原理是将它们的数据分割成一系列的分区，存储在每个executor节点上。 然后，每个节点将仅在其自己的分区上执行其工作。 这就是Spark如此强大的原因：如果Executor死亡或任务失败，Spark可以从原始源中重建它所需的分区，然后重新提交任务以完成任务。

![img](https://hackernoon.com/hn-images/1*7aXGUcCy3qHD3U66COrTTA.png)

Spark RDD partitioned amongst executors

## The Dataframe

![img](https://miro.medium.com/max/68/1*t9Cnql3208oRs-ZchANtTg.png?q=20)

![img](https://miro.medium.com/max/1431/1*t9Cnql3208oRs-ZchANtTg.png)

DataFrame is a *Dataset* organized into named columns.。它在概念上相当于关系数据库中的一个表或R/Python中的一个data frame，但在幕后进行了更丰富的优化。DataFrames can be constructed from a wide array of [sources](https://spark.apache.org/docs/latest/sql-programming-guide.html#data-sources) such as: structured data files, tables in Hive, external databases, or existing RDDs.

![图像](https://miro.medium.com/max/68/1*OY41hGbe4IB9型-hHLRPuCHQ.png?q=20)

![img](https://miro.medium.com/max/68/1*OY41hGbe4IB9-hHLRPuCHQ.png?q=20)

![img](https://miro.medium.com/max/4473/1*OY41hGbe4IB9-hHLRPuCHQ.png)

https://aspgems.com/blog/big-data/migrando-de-pandas-spark-dataframes

简单地说，dataframes 是Spark创建者用来简化在框架中处理数据的方法。它们与Pandas Dataframes或R Dataframes非常相似，但有几个优点。第一个当然是它们可以分布在一个集群中，因此它们可以处理大量的数据，第二个是对数据进行了优化。

## RDD操作

RDD是不可变的，这意味着一旦它们被创建，就不能以任何方式对其进行更改，只能对其进行转换。转换RDD的概念是Spark的核心，可以将Spark Jobs 视为下面这些步骤的任意组合：

- 将数据加载到RDD中
- 转换RDD

- 在RDD上执行操作

实际上，我编写的每一个Spark作业都完全由这些类型的任务组成的。Spark定义了一组用于处理RDD的API，可以将其分为两大类：转换和操作。

- 转换会从现有的RDD创建一个新的RDD。

- 操作在其RDD上运行计算后，会将一个或多个值返回给驱动程序。

例如，地图函数weatherData.map（）是一种转换，它通过函数传递RDD的每个元素的转换。Reduce是一个RDD操作，它使用某些函数来聚合RDD的所有元素，并将最终结果返回给驱动程序。

## 惰性计算

“我选择一个懒惰的人做艰苦的工作。因为一个懒惰的人会找到一种简单的方法来做到这一点。 - 比尔盖茨”

Spark中的所有转换都是惰性的。这意味着，当我们告诉Spark通过现有RDD的转换来创建RDD时，直到对该数据集或其子对象执行特定操作之前，它不会生成该数据集。然后，Spark将执行转换以及触发转换的操作。这使Spark可以更有效地运行。

默认情况下，每次在其上执行操作时，都可能会重新计算每个转换后的RDD。 但是，您也可以使用persist（或缓存）方法将RDD保留在内存中，在这种情况下，Spark会将元素保留在群集中，以便在下次查询时更快地进行访问。 还支持将RDD持久存储在磁盘上，或在多个节点之间复制。

让我们重新检查之前的Spark示例中的函数声明，以识别哪些函数是动作，哪些是转换：

```
16：JavaRDD <String> weatherData = sc.textFile（inputPath）;
```


第16行既不是动作也不是变换。它是sc（我们的JavaSparkContext）的功能。

```
17：JavaPairRDD <String，Integer> tempsByCountry = weatherData.mapToPair（新函数..
```

第17行是weatherData RDD的转换，其中我们将weatherData的每一行映射到由（City，Temperature）组成的一对

```
26：JavaPairRDD <String，Integer> maxTempByCountry = tempsByCountry.reduce（新函数...
```

第26行也是一种转换，因为我们正在遍历键值对。这是对tempsByCountry的一种转换，在该转换中，我们将每个城市的温度降低到最高记录温度。

```
31：maxTempByCountry.saveAsHadoopFile（destPath，String.class，Integer.class，TextOutputFormat.class）；
```

最后，在第31行，我们触发了Spark动作：将RDD保存到文件系统中。 由于Spark订阅了惰性执行模型，因此直到这一行Spark生成weatherData，tempsByCountry和maxTempsByCountry才最终保存我们的结果。

## Directed Acyclic Graph

每当在RDD上执行操作时，Spark都会创建DAG，即有向循环的有限直接图（否则我们的工作将永远运行）。 请记住，图只不过是一系列连接的顶点和边，并且该图也没有什么不同。 DAG中的每个顶点都是Spark函数，在RDD上执行某些操作（map，mapToPair，reduceByKey等）。

在MapReduce中，DAG由两个顶点组成：Map→Reduce。

在我们上面的MaxTemperatureByCountry示例中，DAG更为复杂：

parallelize → map → mapToPair → reduce → saveAsHadoopFile

DAG使Spark可以优化其执行计划并最大程度地减少扰动。 我们将在以后的文章中更深入地讨论DAG，因为它不在本Spark概述的范围之内。

## Evaluation Loops

使用我们的新词汇，让我们重新检查我在第一部分中定义的MapReduce的问题，如下所述：

MapReduce在批处理数据方面表现出色，但是在重复分析和小的反馈循环方面却落后。在计算之间重用数据的唯一方法是将其写入外部存储系统（例如HDFS）”
“在计算之间重复使用数据”？听起来像是可以对其执行多项操作的RDD！假设我们有一个文件“ data.txt”，并且想要完成两个计算：

- 文件中所有行的总长度
- 文件中最长行的长度

在MapReduce中，每个任务将需要单独的作业或奇特的MulitpleOutputFormat实现。Spark 只需四个简单的步骤即可轻松实现：

1.将data.txt的内容加载到RDD中

```
JavaRDD <String>行= sc.textFile（“ data.txt”）;
```

2.将“行”的每一行映射到其长度（为简洁起见使用Lambda函数）

```
JavaRDD <Integer> lineLengths = lines.map（s-> s.length（））;
```

3.解决总长度：减少lineLengths以找到总线长总和，在这种情况下，是RDD中每个元素的总和

```
int totalLength = lineLengths.reduce（（a，b）-> a + b）;
```

4.解决最长的长度：减小lineLengths以找到最大的行长

```
int maxLength = lineLengths.reduce（（a，b）-> Math.max（a，b））;
```


请注意，第3步和第4步是RDD动作，因此它们将结果返回到我们的Driver程序，在本例中为Java int。还要记住，Spark是懒惰的，并且拒绝执行任何工作，直到看到一个动作为止，在这种情况下，它直到第3步才开始任何实际的工作。

# Deep Learning and Apache Spark



为什么要在Apache Spark上进行深度学习？

- Apache Spark是一个了不起的框架，用于以简单且声明性的方式在集群中分布计算。 它正在成为各行各业的标准，因此在其中添加深度学习的惊人进步将是很棒的。
- 深度学习的某些部分在计算上非常繁琐！ 分发这些进程可能是解决其他问题的解决方案，而Apache Spark是我认为可以分发它们的最简单方法。
- 有几种使用Apache Spark进行深度学习的方法，我之前已经讨论过，在此再次列出（不详尽）：

1.[ Elephas](http://maxpumperla.github.io/elephas/): Distributed DL with Keras & PySpark:

[maxpumperla/elephaselephas — Distributed Deep learning with Keras & Spark](https://github.com/maxpumperla/elephas)

[**2. Yahoo! Inc.**](https://www.linkedin.com/company/1288/): TensorFlowOnSpark:

[yahoo/TensorFlowOnSparkTensorFlowOnSpark brings TensorFlow programs onto Apache Spark clusters](https://github.com/yahoo/TensorFlowOnSpark)

[**3. CERN**](https://www.linkedin.com/company/157302/) Distributed Keras (Keras + Spark) :

[cerndb/dist-kerasdist-keras — Distributed Deep Learning, with a focus on distributed training, using Keras and Apache Spark](https://github.com/cerndb/dist-keras)

[**4. Qubole**](https://www.linkedin.com/company/2531735/) (tutorial Keras + Spark):

[Distributed Deep Learning with Keras on Apache Spark | QuboleDeep learning has been shown to produce highly effective machine learning models in a diverse group of fields. ](https://www.qubole.com/blog/distributed-deep-learning-keras-apache-spark/)

[**5. Intel Corporation**](https://www.linkedin.com/company/1053/): BigDL (Distributed Deep Learning Library for Apache Spark)

[intel-analytics/BigDLBigDL: Distributed Deep Learning Library for Apache Spark](https://github.com/intel-analytics/BigDL)

# Deep Learning Pipelines

![img](https://miro.medium.com/max/68/1*6gBbuSw5qH34uI7GF4p97A.png?q=20)

![img](https://miro.medium.com/max/6350/1*6gBbuSw5qH34uI7GF4p97A.png)

[databricks/spark-deep-learningspark-deep-learning - Deep Learning Pipelines for Apache Spark](https://github.com/databricks/spark-deep-learning)

**Deep Learning Pipelines**是Databricks创建的一个开源代码库，该库提供了高级API，可用于使用Apache Spark在Python中进行可扩展的深度学习。
这是一项了不起的工作，并且很快就会被合并到官方API中，因此值得一看。与我之前列出的库相比，该库的一些优点是：

- 本着Spark和Spark MLlib的精神，它提供了易于使用的API，使用很少的代码即可进行深度学习。
- 它着重于易用性和集成性，而不牺牲性能。
- 它是由Apache Spark的创建者（也是主要贡献者）构建的，因此与其他应用相比，它更有可能被合并为正式API。
- 它是用Python编写的，因此将与其所有著名的库集成在一起，现在，它使用TensorFlow和Keras的强大功能，这是目前用于DL的两个主要库。

在下一篇文章中，我将完全专注于DL管道库以及如何从头开始使用它。您将看到的一件事是在简单的管道上进行转移学习，如何使用预训练的模型来处理“少量”数据，并能够预测事物，如何通过深入了解来增强公司中的每个人您创建的学习模型可在SQL中使用，甚至更多。
