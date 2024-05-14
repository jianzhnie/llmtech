# 流计算简介

数据总体上可以分为静态数据和流数据。对静态数据和流数据的处理，对应着两种截然不同的计算模式：批量计算和实时计算。批量计算以“静态数据”为对象，可以在很充裕的时间内对海量数据进行批量处理，计算得到有价值的信息。Hadoop就是典型的批处理模型，由HDFS和HBase存放大量的静态数据，由MapReduce负责对海量数据执行批量计算。流数据必须采用实时计算，实时计算最重要的一个需求是能够实时得到计算结果，一般要求响应时间为秒级。当只需要处理少量数据时，实时计算并不是问题；但是，在大数据时代，不仅数据格式复杂、来源众多，而且数据量巨大，这就对实时计算提出了很大的挑战。因此，针对流数据的实时计算——流计算，应运而生。

总的来说，流计算秉承一个基本理念，即数据的价值随着时间的流逝而降低。因此，当事件出现时就应该立即进行处理，而不是缓存起来进行批量处理。为了及时处理流数据，就需要一个低延迟、可扩展、高可靠的处理引擎。对于一个流计算系统来说，它应达到如下需求。

\* • 高性能。处理大数据的基本要求，如每秒处理几十万条数据。
\* • 海量式。支持TB级甚至是PB级的数据规模。
\* • 实时性。必须保证一个较低的延迟时间，达到秒级别，甚至是毫秒级别。
\* • 分布式。支持大数据的基本架构，必须能够平滑扩展。
\* • 易用性。能够快速进行开发和部署。
\* • 可靠性。能可靠地处理流数据。

![img](http://dblab.xmu.edu.cn/blog/wp-content/uploads/2016/11/%E5%9B%BE10-3-%E6%B5%81%E8%AE%A1%E7%AE%97%E7%A4%BA%E6%84%8F%E5%9B%BE.jpg)
图 流计算示意图

目前，市场上有很多流计算框架，比如Twitter Storm和Yahoo! S4等。Twitter Storm是免费、开源的分布式实时计算系统，可简单、高效、可靠地处理大量的流数据；Yahoo! S4开源流计算平台，是通用的、分布式的、可扩展的、分区容错的、可插拔的流式系统。

流计算处理过程包括数据实时采集、数据实时计算和实时查询服务。
\* 数据实时采集：数据实时采集阶段通常采集多个数据源的海量数据，需要保证实时性、低延迟与稳定可靠。以日志数据为例，由于分布式集群的广泛应用，数据分散存储在不同的机器上，因此需要实时汇总来自不同机器上的日志数据。目前有许多互联网公司发布的开源分布式日志采集系统均可满足每秒数百MB的数据采集和传输需求，如Facebook的Scribe、LinkedIn的Kafka、淘宝的TimeTunnel，以及基于Hadoop的Chukwa和Flume等。

- 数据实时计算：流处理系统接收数据采集系统不断发来的实时数据，实时地进行分析计算，并反馈实时结果。
- 实时查询服务：流计算的第三个阶段是实时查询服务，经由流计算框架得出的结果可供用户进行实时查询、展示或储存。

![img](http://dblab.xmu.edu.cn/blog/wp-content/uploads/2016/11/%E5%9B%BE10-5-%E6%B5%81%E8%AE%A1%E7%AE%97%E7%9A%84%E6%95%B0%E6%8D%AE%E5%A4%84%E7%90%86%E6%B5%81%E7%A8%8B.jpg)
图 流计算的数据处理流程

# Spark Streaming简介(Python版)

 Ruan Rongcheng 2017年12月8日 (updated: 2020年5月29日) 7064

[![大数据学习路线图](http://dblab.xmu.edu.cn/blog/wp-content/uploads/2019/01/bigdata-roadmap.jpg)](http://dblab.xmu.edu.cn/post/bigdataroadmap/)

【版权声明】博客内容由厦门大学数据库实验室拥有版权，未经允许，请勿转载！
[返回Spark教程首页](http://dblab.xmu.edu.cn/blog/1709-2/)
推荐纸质教材：[林子雨、郑海山、赖永炫编著《Spark编程基础（Python版）》](http://dblab.xmu.edu.cn/post/spark-python/)

Spark Streaming是构建在Spark上的实时计算框架，它扩展了Spark处理大规模流式数据的能力。Spark Streaming可结合批处理和交互查询，适合一些需要对历史数据和实时数据进行结合分析的应用场景。



## Spark Streaming设计

Spark Streaming是Spark的核心组件之一，为Spark提供了可拓展、高吞吐、容错的流计算能力。如下图所示，Spark Streaming可整合多种输入数据源，如Kafka、Flume、HDFS，甚至是普通的TCP套接字。经处理后的数据可存储至文件系统、数据库，或显示在仪表盘里。
![img](http://dblab.xmu.edu.cn/blog/wp-content/uploads/2016/11/%E5%9B%BE10-19-Spark-Streaming%E6%94%AF%E6%8C%81%E7%9A%84%E8%BE%93%E5%85%A5%E3%80%81%E8%BE%93%E5%87%BA%E6%95%B0%E6%8D%AE%E6%BA%90.jpg)
图 Spark-Streaming支持的输入、输出数据源

Spark Streaming的基本原理是将实时输入数据流以时间片（秒级）为单位进行拆分，然后经Spark引擎以类似批处理的方式处理每个时间片数据，执行流程如下图所示。

![img](http://dblab.xmu.edu.cn/blog/wp-content/uploads/2016/11/%E5%9B%BE10-20-Spark-Streaming%E6%89%A7%E8%A1%8C%E6%B5%81%E7%A8%8B.jpg)
图 Spark Streaming执行流程

Spark Streaming最主要的抽象是  DStream（Discretized Stream，离散化数据流），表示连续不断的数据流。在内部实现上，Spark Streaming的输入数据按照时间片（如1秒）分成一段一段的DStream，每一段数据转换为Spark中的RDD，并且对DStream的操作都最终转变为对相应的RDD的操作。例如，下图展示了进行单词统计时，每个时间片的数据（存储句子的RDD）经 flatMap 操作，生成了存储单词的RDD。整个流式计算可根据业务的需求对这些中间的结果进一步处理，或者存储到外部设备中。

![img](http://dblab.xmu.edu.cn/blog/wp-content/uploads/2016/11/%E5%9B%BE10-21-DStream%E6%93%8D%E4%BD%9C%E7%A4%BA%E6%84%8F%E5%9B%BE.jpg)
图 DStream操作示意图

## Spark Streaming与Storm的对比

Spark Streaming和Storm最大的区别在于，Spark Streaming 无法实现毫秒级的流计算，而Storm可以实现毫秒级响应。
Spark Streaming无法实现毫秒级的流计算，是因为其将流数据按批处理窗口大小（通常在0.5~2秒之间）分解为一系列批处理作业，在这个过程中，会产生多个Spark 作业，且每一段数据的处理都会经过Spark DAG图分解、任务调度过程，因此，无法实现毫秒级相应。Spark Streaming难以满足对实时性要求非常高（如高频实时交易）的场景，但足以胜任其他流式准实时计算场景。相比之下，Storm处理的单位为Tuple，只需要极小的延迟。
Spark Streaming构建在Spark上，一方面是因为Spark的低延迟执行引擎（100毫秒左右）可以用于实时计算，另一方面，相比于Storm，RDD数据集更容易做高效的容错处理。此外，Spark Streaming采用的小批量处理的方式使得它可以同时兼容批量和实时数据处理的逻辑和算法，因此，方便了一些需要历史数据和实时数据联合分析的特定应用场合。

# DStream操作概述(Python版)

DStream是Spark Streaming的编程模型，DStream的操作包括输入、转换和输出。

## Spark Streaming工作原理

前面在《[Spark运行架构](http://dblab.xmu.edu.cn/blog/972-2/)》部分，我们已经介绍过，在Spark中，一个应用（Application）由一个任务控制节点（Driver）和若干个作业（Job）构成，一个作业由多个阶段（Stage）构成，一个阶段由多个任务（Task）组成。当执行一个应用时，任务控制节点会向集群管理器（Cluster Manager）申请资源，启动Executor，并向Executor发送应用程序代码和文件，然后在Executor上执行task。在Spark Streaming中，会有一个组件Receiver，作为一个长期运行的task跑在一个Executor上。每个Receiver都会负责一个input DStream（比如从文件中读取数据的文件流，比如套接字流，或者从Kafka中读取的一个输入流等等）。Spark Streaming通过input DStream与外部数据源进行连接，读取相关数据。

## Spark Streaming程序基本步骤

编写Spark Streaming程序的基本步骤是：
1.通过创建输入DStream来定义输入源
2.通过对DStream应用转换操作和输出操作来定义流计算。
3.用streamingContext.start()来开始接收数据和处理流程。
4.通过streamingContext.awaitTermination()方法来等待处理结束（手动结束或因为错误而结束）。
5.可以通过streamingContext.stop()来手动结束流计算进程。

## 创建StreamingContext对象

如果要运行一个Spark Streaming程序，就需要首先生成一个StreamingContext对象，它是Spark Streaming程序的主入口。因此，在定义输入之前，我们首先介绍如何创建StreamingContext对象。我们可以从一个SparkConf对象创建一个StreamingContext对象。
请登录Linux系统，启动pyspark。进入pyspark以后，就已经获得了一个默认的SparkConext，也就是sc。因此，可以采用如下方式来创建StreamingContext对象：

```python
>>> from pyspark import SparkContext
>>> from pyspark.streaming import StreamingContext
>>> ssc = StreamingContext(sc, 1)
```

Python

1表示每隔1秒钟就自动执行一次流计算，这个秒数可以自由设定。
如果是编写一个独立的Spark Streaming程序，而不是在pyspark中运行，则需要通过如下方式创建StreamingContext对象：

```python
from pyspark import SparkContext, SparkConf
from pyspark.streaming import Streaming, Context
conf = SparkConf()
conf.setAppName('TestDStream')
conf.setMaster('local[2]')
sc = SparkContext(conf = conf)ssc = StreamingContext(sc, 1)
```

Python

setAppName(“TestDStream”)是用来设置应用程序名称，这里我们取名为“TestDStream”。setMaster(“local[2]”)括号里的参数”local[2]’字符串表示运行在本地模式下，并且启动2个工作线程。

# 文件流(DStream)(Python版)



Spark支持从兼容HDFS API的文件系统中读取数据，创建数据流。

为了能够演示文件流的创建，我们需要首先创建一个日志目录，并在里面放置两个模拟的日志文件。请在Linux系统中打开另一个终端，进入Shell命令提示符状态：

```bash
cd /usr/local/spark/mycodemkdir streamingcd streamingmkdir logfilecd logfile
```

Shell 命令

然后，在logfile中新建两个日志文件log1.txt和log2.txt，里面可以随便输入一些内容。
比如，我们在log1.txt中输入以下内容：

```
I love Hadoop
I love Spark
Spark is fast
```

下面我们就进入pyspark创建文件流。请另外打开一个终端窗口，启动进入pyspark。

```python
>>> from operator import add
>>> from pyspark import SparkContext
>>> from pyspark.streaming import StreamingContext
>>> ssc = StreamingContext(sc, 20)
>>> lines = ssc.textFileStream('file:///usr/local/spark/mycode/streaming/logfile')
>>> words = lines.flatMap(lambda line: line.split(' '))
>>> wordCounts = words.map(lambda x : (x,1)).reduceByKey(add)
>>> wordCounts.pprint()
>>> ssc.start()
>>>#实际上，当你输入这行回车后，Spark Streaming就开始进行循环监听，下面的ssc.awaitTermination()是无法输入到屏幕上的，但是，为了程序完整性，这里还是给出ssc.awaitTermination()
>>> ssc.awaitTermination()
```

Python

所以，上面在pyspark中执行的程序，一旦你输入ssc.start()以后，程序就开始自动进入循环监听状态，屏幕上会显示一堆的信息，如下：

```python
//这里省略若干屏幕信息
-------------------------------------------
Time: 1479431100000 ms
-------------------------------------------
//这里省略若干屏幕信息
-------------------------------------------
Time: 1479431120000 ms
-------------------------------------------
//这里省略若干屏幕信息
-------------------------------------------
Time: 1479431140000 ms
-------------------------------------------
```

Python

从上面的屏幕显示信息可以看出，Spark Streaming每隔20秒就监听一次。但是，你这时会感到奇怪，既然启动监听了，为什么程序没有把我们刚才放置在”/usr/local/spark/mycode/streaming/logfile”目录下的log1.txt和log2.txt这两个文件中的内容读取出来呢？原因是，监听程序只监听”/usr/local/spark/mycode/streaming/logfile”目录下在程序启动后新增的文件，不会去处理历史上已经存在的文件。所以，为了能够让程序读取文件内容并显示到屏幕上，让我们能够看到效果，这时，我们需要到”/usr/local/spark/mycode/streaming/logfile”目录下再新建一个log3.txt文件，请打开另外一个终端窗口（我们称为shell窗口），当前正在执行监听工作的spark-shell窗口依然保留。请在shell窗口中执行Linux操作，在”/usr/local/spark/mycode/streaming/logfile”目录下再新建一个log3.txt文件，里面随便输入一些英文单词，创建完成以后，再切换回到spark-shell窗口。请等待20秒（因为我们刚才设置的是每隔20秒就监听一次，如果你创建文件动作很快，可能20秒还没到）。现在你会发现屏幕上不断输出新的信息，导致你无法看清楚单词统计结果是否已经被打印到屏幕上。所以，你现在必须停止这个监听程序，否则它一直在pyspark窗口中不断循环监听，停止的方法是，按键盘Ctrl+D，或者Ctrl+C。停止以后，就彻底停止，并且退出了pyspark状态，回到了Shell命令提示符状态。然后，你就可以看到屏幕上，在一大堆输出信息中，你可以找到打印出来的单词统计信息。

好了，上面我们是在spark-shell中直接执行代码，但是，很多时候，我们需要编写独立应用程序进行监听，所以，下面我们介绍如何采用独立应用程序的方式实现上述监听文件夹的功能。

请打开一个Linux终端窗口，进入shell命令提示符状态，然后，执行下面命令：

```bash
cd /usr/local/spark/mycode/streamingvim TestStreaming.py
```

Shell 命令

在TestStreaming.py中输入如下代码：

```python
from operator import add
from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
conf = SparkConf()
conf.setAppName('TestDStream')
conf.setMaster('local[2]')
sc = SparkContext(conf = conf)
ssc = StreamingContext(sc, 20)
lines = ssc.textFileStream('file:///usr/local/spark/mycode/streaming/logfile')
words = lines.flatMap(lambda line: line.split(' '))
wordCounts = words.map(lambda x : (x,1)).reduceByKey(add)
wordCounts.pprint()
ssc.start()
ssc.awaitTermination()
```

Python

保存成功后，执行在命令行中执行如下代码:

```bash
python3 TestStreaming.py
```
