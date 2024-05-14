# Spark 生态

## Spark与深度学习框架

### H2O
　　H2O是用h2o.ai开发的具有可扩展性的机器学习框架，它不限于深度学习。H2O支持许多API（例如，R、Python、Scala和Java）。当然它是开源软件，所以要研究它的代码及算法也很容易。H2O框架支持所有常见的数据库及文件类型，可以轻松将模型导出为各种类型的存储。深度学习算法是在另一个叫作sparkling-water的库中实现的https://github.com/h2oai/sparkling-water，它主要由h2o.ai开发。

### deeplearning4j

　　deeplearning4j是由Skymind开发的，Skymind是一家致力于为企业进行商业化深度学习的公司。deeplearning4j框架是创建来在Hadoop及Spark上运行的。这个设计用于商业环境而不是许多深度学习框架及库目前所大量应用的研究领域。Skymind是主要的支持者，但deeplearning4j是开源软件，因此也欢迎大家提交补丁。deeplearning4j框架中实现了如下算法：

- 受限玻尔兹曼机（Restricted Boltzmann Machine）
- 卷积神经网络（Convolutional Neural Network）
- 循环神经网络（Recurrent Neural Network）
- 递归自编码器（Recursive Autoencoder）
- 深度信念网络（Deep-Belief Network）
- 深度自编码器（Deep Autoencoder）
- 栈式降噪自编码（Stacked Denoising Autoencoder）

这里要注意的是，这些模型能在细粒度级别进行配置。你可以设置隐藏的层数、每个神经元的激活函数以及迭代的次数。deeplearning4j提供了不同种类的网络实现及灵活的模型参数。Skymind也开发了许多工具，对于更稳定地运行机器学习算法很有帮助。下面列出了其中的一些工

**Canova [https://github.com/deeplearning4j/Canoba]**是一个向量库。机器学习算法能以向量格式处理所有数据。所有的图片、音频及文本数据必须用某种方法转换为向量。虽然训练机器学习模型是十分常见的工作，但它会重新造轮子还会引起bug。Canova能为你做这种转换。Canova当前支持的输入数据格式为：
-- CSV
--原始文本格式（推文、文档）
--图像（图片、图画）
--定制文件格式（例如MNIST）

- **由于Canova主要是用Java编写的，所以它能运行在所有的JVM平台上。**因此，可以在Spark集群上使用它。即使你不做机器学习，Canova对你的机器学习任务可能也会有所裨益。
- **nd4j** [https://github.com/deeplearning4j/nd4j] **有点像是一个numpy，Python中的SciPy工具。**此工具提供了线性代数、向量计算及操纵之类的科学计算。它也是用Java编写的。你可以根据自己的使用场景来搭配使用这些工具。需要注意的一点是，nd4j支持GPU功能。由于现代计算硬件还在不断发展，有望达到更快速的计算。
- **dl4j-spark-ml** [https://github.com/deeplearning4j/dl4j-spark-ml]** 是一个Spark包，使你能在Spark上轻松运行deeplearning4j。**使用这个包，就能轻松在Spark上集成deeplearning4j，因为它已经被上传到了Spark包的公共代码库。

因此，如果你要在Spark上使用deeplearning4j，我们推荐通过dl4j-spark-ml包来实现。与往常一样，必须下载或自己编译Spark源码。这里对Spark版本没有特别要求，就算使用最早的版本也可以。deeplearning4j项目准备了样例存储库。要在Spark上使用deeplearning4j，dl4j-Spark-ml-examples是可参考的最佳示例（https:// github.com/deeplearning4j/dl4j-Spark-ml-examples）


### DJL
https://github.com/awslabs/djld
Apache Spark是一个优秀的大数据处理工具。在机器学习领域，Spark可以用于对数据分类，预测需求以及进行个性化推荐。虽然Spark支持多种语言，但是大部分Spark任务设定及部署还是通过Scala来完成的。尽管如此，Scala并没有很好的支持深度学习平台。大部分的深度学习应用都部署在Python以及相关的框架之上，造成Scala开发者一个很头痛的问题：到底是全用Python写整套spark架构呢，还是说用Scala包装Python code在pipeline里面跑。这两个方案都会增加工作量和维护成本。而且，目前看来，PySpark在深度学习多进程的支持上性能不如Scala的多线程，导致许多深度学习应用速度都卡在了这里。

今天，我们会展示給用户一个新的解决方案，直接使用Scala调用 Deep Java Library (DJL)来实现深度学习应用部署。DJL将充分释放Spark强大的多线程处理性能，轻松提速2-5倍*现有的推理任务。DJL是一个为Spark量身定制的Java深度学习库。它不受限于引擎，用户可以轻松的将PyTorch, TensorFlow 以及MXNet的模型部署在Spark上。在本blog中，我们通过使用DJL来完成一个图片分类模型的部署任务，你也可以在这里参阅完整的代码。

### TensorFlowOnSpark
https://github.com/yahoo/TensorFlowOnSpark

TensorFlowOnSpark 为 Apache Hadoop 和 Apache Spark 集群带来可扩展的深度学习。 通过结合深入学习框架 TensorFlow 和大数据框架 Apache Spark 、Apache Hadoop 的显着特征，TensorFlowOnSpark 能够在 GPU 和 CPU 服务器集群上实现分布式深度学习。

**TensorFlowOnSpark**

![img](https://static001.infoq.cn/resource/image/9b/dc/9bae4abc1c69491d645975b3f88137dc.png)

我们的新框架 TensorFlowOnSpark（TFoS），支持 TensorFlow 在 Spark 和 Hadoop 集群上分布式执行。如上图 2 所示，TensorFlowOnSpark 被设计为与 SparkSQL、MLlib 和其他 Spark 库一起在一个单独流水线或程序（如 Python notebook）中运行。

TensorFlowOnSpark 支持所有类型的 TensorFlow 程序，可以实现异步和同步的训练和推理。它支持模型并行性和数据的并行处理，以及 TensorFlow 工具（如 Spark 集群上的 TensorBoard）。

任何 TensorFlow 程序都可以轻松地修改为在 TensorFlowOnSpark 上运行。通常情况下，需要改变的 Python 代码少于 10 行。许多 Yahoo 平台使用 TensorFlow 的开发人员很容易迁移 TensorFlow 程序，以便在 TensorFlowOnSpark 上执行。

TensorFlowOnSpark 支持 TensorFlow 进程（计算节点和参数服务节点）之间的直接张量通信。过程到过程的直接通信机制使 TensorFlowOnSpark 程序能够在增加的机器上很轻松的进行扩展。如图 3 所示，TensorFlowOnSpark 不涉及张量通信中的 Spark 驱动程序，因此实现了与独立 TensorFlow 集群类似的可扩展性。

![img](https://static001.infoq.cn/resource/image/c7/77/c7406e478beb085693c0e431f5f53c77.png)

TensorFlowOnSpark 提供两种不同的模式来提取训练和推理数据：

1. **TensorFlow QueueRunners：**TensorFlowOnSpark 利用 TensorFlow 的[ file readers ](http://t.umblr.com/redirect?z=https://www.tensorflow.org/how_tos/reading_data/#reading_from_files&t=MDk1NzhlZDQ0YTM2MTY4OWY4OTFhOGYzMDRjYmMxOGY4N2NiMmY3Myx1VmR1UG1vZg==&b=t%3afgAkOE96nMUZDZ4JRZ0Fgw&p=http://yahoohadoop.tumblr.com/post/157196317141/open-sourcing-tensorflowonspark-distributed-deep&m=1)和[ QueueRunners ](http://t.umblr.com/redirect?z=https://www.tensorflow.org/how_tos/threading_and_queues/#queuerunner&t=ZjI4YjM5ODg4NTZiMmVlMTNjN2JhOWEyNzdkMjk5NjE0MTFiOTdlMix1VmR1UG1vZg==&b=t%3afgAkOE96nMUZDZ4JRZ0Fgw&p=http://yahoohadoop.tumblr.com/post/157196317141/open-sourcing-tensorflowonspark-distributed-deep&m=1)直接从 HDFS 文件中读取数据。Spark 不涉及访问数据。
2. **Spark Feeding** ：Spark RDD 数据被传输到每个 Spark 执行器里，随后的数据将通过[ feed_dict ](http://t.umblr.com/redirect?z=https://www.tensorflow.org/how_tos/reading_data/#feeding&t=YWY2Y2U4YTE0ODc2M2E0NzYwNjFjZTE2MWE1ZWY5M2JjOTNiMTdlZCx1VmR1UG1vZg==&b=t%3afgAkOE96nMUZDZ4JRZ0Fgw&p=http://yahoohadoop.tumblr.com/post/157196317141/open-sourcing-tensorflowonspark-distributed-deep&m=1)传入 TensorFlow 图。

**简单的CLI 和API**

TFoS 程序由标准的 Apache Spark 命令 _spark-submit_ 来启动。如下图所示，用户可以在 CLI 中指定 Spark 执行器的数目，每个执行器的 GPU 数量和参数服务器的数目。用户还可以指定是否要使用 TensorBoard（-tensorboard）和 / 或 RDMA（-rdma）。

复制代码

```shell

     spark-submit –master ${MASTER} \
     ${TFoS_HOME}/examples/slim/train_image_classifier.py \
     –model_name inception_v3 \
     –train_dir hdfs://default/slim_train \
     –dataset_dir hdfs://default/data/imagenet \
     –dataset_name imagenet \
     –dataset_split_name train \
     –cluster_size ${NUM_EXEC} \
     –num_gpus ${NUM_GPU} \
     –num_ps_tasks ${NUM_PS} \
     –sync_replicas \
     –replicas_to_aggregate ${NUM_WORKERS} \
     –tensorboard \
     –rdma
```

TFoS 提供了一个高层次的 Python API（在我[们示例 Python notebook ](http://t.umblr.com/redirect?z=https://github.com/yahoo/TensorFlowOnSpark/blob/master/examples/mnist/TFOS_demo.ipynb&t=MWFkZDEwZTExNDY1NDQ0ZTkwODgxODgzMmM0MTgwZTk1MTU4NzAwNSx1VmR1UG1vZg==&b=t%3afgAkOE96nMUZDZ4JRZ0Fgw&p=http://yahoohadoop.tumblr.com/post/157196317141/open-sourcing-tensorflowonspark-distributed-deep&m=1)说明）：

- TFCluster.reserve() … construct a TensorFlow cluster from Spark executors
- TFCluster.start() … launch Tensorflow program on the executors
- TFCluster.train() or TFCluster.inference() … feed RDD data to TensorFlow processes
- TFCluster.shutdown() … shutdown Tensorflow execution on executors

**开放源码**

[TensorFlowOnSpark ](http://t.umblr.com/redirect?z=https://github.com/yahoo/TensorFlowOnSpark&t=NjRhYmYzODNiNzQ1ODUwZjIwOGRiZDQyZmMyYThkMzExMmM2ZWNjOCx1VmR1UG1vZg==&b=t%3afgAkOE96nMUZDZ4JRZ0Fgw&p=http://yahoohadoop.tumblr.com/post/157196317141/open-sourcing-tensorflowonspark-distributed-deep&m=1)、[ TensorFlow 的 RDMA 增强包](http://t.umblr.com/redirect?z=https://github.com/yahoo/tensorflow/tree/yahoo&t=NWE0M2NjODYwOGMzM2I1MTNhZjUyZDQwMGU1ZDRmNmE3NjIxNzQwNCx1VmR1UG1vZg==&b=t%3afgAkOE96nMUZDZ4JRZ0Fgw&p=http://yahoohadoop.tumblr.com/post/157196317141/open-sourcing-tensorflowonspark-distributed-deep&m=1)、多个[示例程序](http://t.umblr.com/redirect?z=https://github.com/yahoo/TensorFlowOnSpark/tree/master/examples&t=OGVjN2VhM2UxZWQ3NDNiMDg4NTM5ODA0ZWI4YjQ2ODYxM2UxYzIyZix1VmR1UG1vZg==&b=t%3afgAkOE96nMUZDZ4JRZ0Fgw&p=http://yahoohadoop.tumblr.com/post/157196317141/open-sourcing-tensorflowonspark-distributed-deep&m=1)（包括MNIST，cifar10，创建以来，VGG）来说明TensorFlow 方案TensorFlowOnSpark，并充分利用RDMA 的简单转换过程。亚马逊机器映像也[可](http://t.umblr.com/redirect?z=https://github.com/yahoo/TensorFlowOnSpark/wiki/GetStarted_EC2&t=MTcyNGUyYjdjMTZkNWYyYjAwNGE5NGY3M2Q0ZTI5ZTc3ZDllMGVhZCx1VmR1UG1vZg==&b=t%3afgAkOE96nMUZDZ4JRZ0Fgw&p=http://yahoohadoop.tumblr.com/post/157196317141/open-sourcing-tensorflowonspark-distributed-deep&m=1)对AWS EC2 应用TensorFlowOnSpark。

### BigDL

https://github.com/intel-analytics/BigDL

#### 什么是 BigDL？

BigDL 是一款面向 Spark 的分布式深度学习库，在现有的 Spark 或 Apache Hadoop* 集群上直接运行。您可以将深度学习应用编写为 Scala 或 Python 程序。

- **丰富的深度学习支持。[BigDL](https://github.com/intel-analytics/BigDL)** 模仿 [Torch](http://torch.ch/) ，为深度学习提供综合支持，包括数值计算（借助[Tensor](https://github.com/intel-analytics/BigDL/tree/master/dl/src/main/scala/com/intel/analytics/bigdl/tensor)）和[高级神经网络](https://github.com/intel-analytics/BigDL/tree/master/dl/src/main/scala/com/intel/analytics/bigdl/nn)；此外，用户可以将预训练 [Caffe](http://caffe.berkeleyvision.org/)* 或 Torch 模型加载至 Spark 框架，并使用 BigDL 库在数据中运行推断应用。
- **高效的横向扩展。**利用 [Spark](http://spark.apache.org/)，BigDL 能够在 Spark 中高效地横向扩展，处理大数据规模的数据分析，高效实施随机梯度下降 (SGD)，以及进行 all-reduce 通信。
- **极高的性能。**为了实现较高的性能，BigDL 在每个 Spark 任务中采用[英特尔® 数学核心函数库](https://software.intel.com/zh-cn/intel-mkl)（英特尔® MKL）和多线程编程。因此，相比现成的开源 Caffe、Torch 或 [TensorFlow](https://www.tensorflow.org/)，BigDL 在单节点英特尔® 至强® 处理器上的运行速度高出多个数量级（与主流图形处理单元相当）。

#### 什么是 Apache Spark*？

Spark 是一款极速的分布式数据处理框架，由加利福尼亚大学伯克利分校的 AMPLab 开发。Spark 可以以独立模式运行，也能以集群模式在 Hadoop 上的 YARN 中或 Apache Mesos* 集群管理器上运行（图 2）。Spark 可以处理各种来源的数据，包括 HDFS、Apache Cassandra* 或 Apache Hive*。由于它能够通过持久存储的 RDD 或 DataFrames 处理内存，而不是将数据保存至硬盘（如同传统的 Hadoop MapReduce 架构），因此，极大地提高了性能。

![img](https://software.intel.com/content/dam/develop/external/us/en/images/bigdl-on-apache-spark-fig-02-stack-712489.png)
**图 2.** Apache Spark* 堆栈中的 BigDL

#### 为什么使用 BigDL？

在以下情况下，您需要利用 BigDL 编写您的深度学习程序：

- 您希望在存储数据的大数据 Spark 集群（如 HDFS、Apache HBase* 或 Hive ）上分析大量数据；
- 您希望在大数据 (Spark) 程序或工作流中添加深度学习功能（训练或预测）；或者
- 您希望利用现有的 Hadoop/Spark 集群运行深度学习应用，随后与其他工作负载轻松共享（例如提取-转换-加载、数据仓库、特性设计、经典机器学习、图形分析）。另一种使用 BigDL 的不常见的替代方案是与 Spark 同时引进另一种分布式框架，以实施深度学习算法。
