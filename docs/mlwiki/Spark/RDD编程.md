# RDD编程(Python版)

通过前面几章的介绍，我们已经了解了Spark的运行架构和RDD设计与运行原理，并介绍了RDD操作的两种类型：转换操作和行动操作。
同时，我们前面通过一个简单的WordCount实例，也大概介绍了RDD的几种简单操作。现在我们介绍更多关于RDD编程的内容。
Spark中针对RDD的操作包括创建RDD、RDD转换操作和RDD行动操作。

## RDD创建

RDD可以通过两种方式创建：
\* 第一种：读取一个外部数据集。比如，从本地文件加载数据集，或者从HDFS文件系统、HBase、Cassandra、Amazon S3等外部数据源中加载数据集。Spark可以支持文本文件、SequenceFile文件（Hadoop提供的 SequenceFile是一个由二进制序列化过的key/value的字节流组成的文本存储文件）和其他符合Hadoop InputFormat格式的文件。
\* 第二种：调用SparkContext的parallelize方法，在Driver中一个已经存在的集合（数组）上创建。

### 创建RDD之前的准备工作

在即将进行相关的实践操作之前，我们首先要登录Linux系统（本教程统一采用hadoop用户登录），然后，打开命令行“终端”，请按照下面的命令启动Hadoop中的HDFS组件：

```bash
cd  /usr/local/hadoop./sbin/start-dfs.sh
```



然后，我们按照下面命令启动pyspark：

```bash
cd /usr/local/spark./bin/pyspark
```



然后，新建第二个“终端”，方法是，在前面已经建设的第一个终端窗口的左上方，点击“终端”菜单，在弹出的子菜单中选择“新建终端”，就可以打开第二个终端窗口，现在，我们切换到第二个终端窗口，在第二个终端窗口中，执行以下命令，进入之前已经创建好的“/usr/local/spark/mycode/”目录，在这个目录下新建rdd子目录，用来存放本章的代码和相关文件：

```bash
cd usr/local/spark/mycode/mkdir rdd
```



然后，使用vim编辑器，在rdd目录下新建一个word.txt文件，你可以在文件里面随便输入几行英文语句用来测试。

经过上面的准备工作以后，我们就可以开始创建RDD了。

### 从文件系统中加载数据创建RDD

Spark采用textFile()方法来从文件系统中加载数据创建RDD，该方法把文件的URI作为参数，这个URI可以是本地文件系统的地址，或者是分布式文件系统HDFS的地址，或者是Amazon S3的地址等等。
下面请切换回pyspark窗口，看一下如何从本地文件系统中加载数据：

```python
>>> lines = sc.textFile("file:///usr/local/spark/mycode/rdd/word.txt")
```

Python

下面看一下如何从HDFS文件系统中加载数据，这个在前面的第一个Spark应用程序：WordCount实例中已经讲过，这里再简单复习一下。
请根据前面的第一个Spark应用程序：WordCount实例中的内容介绍，把刚才在本地文件系统中的“/usr/local/spark/mycode/rdd/word.txt”上传到HDFS文件系统的hadoop用户目录下（注意：本教程统一使用hadoop用户登录Linux系统）。然后，在pyspark窗口中，就可以使用下面任意一条命令完成从HDFS文件系统中加载数据：

```python
>>> lines = sc.textFile("hdfs://localhost:9000/user/hadoop/word.txt")
>>> lines = sc.textFile("/user/hadoop/word.txt")
>>> lines = sc.textFile("word.txt")
```

Python

注意，上面三条命令是完全等价的命令，只不过使用了不同的目录形式，你可以使用其中任意一条命令完成数据加载操作。

在使用Spark读取文件时，需要说明以下几点：
（1）如果使用了本地文件系统的路径，那么，必须要保证在所有的worker节点上，也都能够采用相同的路径访问到该文件，比如，可以把该文件拷贝到每个worker节点上，或者也可以使用网络挂载共享文件系统。
（2）textFile()方法的输入参数，可以是文件名，也可以是目录，也可以是压缩文件等。比如，textFile(“/my/directory”), textFile(“/my/directory/*.txt”), and textFile(“/my/directory/*.gz”).
（3）textFile()方法也可以接受第2个输入参数（可选），用来指定分区的数目。默认情况下，Spark会为HDFS的每个block创建一个分区（HDFS中每个block默认是128MB）。你也可以提供一个比block数量更大的值作为分区数目，但是，你不能提供一个小于block数量的值作为分区数目。

### 通过并行集合（数组）创建RDD

可以调用SparkContext的parallelize方法，在Driver中一个已经存在的集合（数组）上创建。
下面请在pyspark中操作：

```python
>>> nums = [1,2,3,4,5]
>>> rdd = sc.parallelize(nums)
```

Python

上面使用列表来创建。在Python中并没有数组这个基本数据类型，为了便于理解，你可以把列表当成其他语言的数组。

## RDD操作

RDD被创建好以后，在后续使用过程中一般会发生两种操作：
\*  转换（Transformation）： 基于现有的数据集创建一个新的数据集。
\*  行动（Action）：在数据集上进行运算，返回计算值。

### 转换操作

对于RDD而言，每一次转换操作都会产生不同的RDD，供给下一个“转换”使用。转换得到的RDD是惰性求值的，也就是说，整个转换过程只是记录了转换的轨迹，并不会发生真正的计算，只有遇到行动操作时，才会发生真正的计算，开始从血缘关系源头开始，进行物理的转换操作。
下面列出一些常见的转换操作（Transformation API）：
\* filter(func)：筛选出满足函数func的元素，并返回一个新的数据集
\* map(func)：将每个元素传递到函数func中，并将结果返回为一个新的数据集
\* flatMap(func)：与map()相似，但每个输入元素都可以映射到0或多个输出结果
\* groupByKey()：应用于(K,V)键值对的数据集时，返回一个新的(K, Iterable)形式的数据集
\* reduceByKey(func)：应用于(K,V)键值对的数据集时，返回一个新的(K, V)形式的数据集，其中的每个值是将每个key传递到函数func中进行聚合

### 行动操作

行动操作是真正触发计算的地方。Spark程序执行到行动操作时，才会执行真正的计算，从文件中加载数据，完成一次又一次转换操作，最终，完成行动操作得到结果。
下面列出一些常见的行动操作（Action API）：
\* count() 返回数据集中的元素个数
\* collect() 以数组的形式返回数据集中的所有元素
\* first() 返回数据集中的第一个元素
\* take(n) 以数组的形式返回数据集中的前n个元素
\* reduce(func) 通过函数func（输入两个参数并返回一个值）聚合数据集中的元素
\* foreach(func) 将数据集中的每个元素传递到函数func中运行*

### 惰性机制

这里给出一段简单的代码来解释Spark的惰性机制。

```python
>>> lines = sc.textFile("data.txt")
>>> lineLengths = lines.map(lambda s : len(s))
>>> totalLength = lineLengths.reduce( lambda a, b : a + b)
```

Python

上面第一行首先从外部文件data.txt中构建得到一个RDD，名称为lines，但是，由于textFile()方法只是一个转换操作，因此，这行代码执行后，不会立即把data.txt文件加载到内存中，这时的lines只是一个指向这个文件的指针。
第二行代码用来计算每行的长度（即每行包含多少个单词），同样，由于map()方法只是一个转换操作，这行代码执行后，不会立即计算每行的长度。
第三行代码的reduce()方法是一个“动作”类型的操作，这时，就会触发真正的计算。这时，Spark会把计算分解成多个任务在不同的机器上执行，每台机器运行位于属于它自己的map和reduce，最后把结果返回给Driver Program。

### 实例

下面我们举几个实例加深了解。
请在pyspark下执行下面操作。
下面是一个关于filter()操作的实例。

```python
>>> lines = sc.textFile("file:///usr/local/spark/mycode/rdd/word.txt")
>>> lines.filter(lambda line : "Spark" in line).count()
2
```

Python

上面的代码中，lines就是一个RDD。lines.filter()会遍历lines中的每行文本，并对每行文本执行括号中的匿名函数，也就是执行Lamda表达式：line : “Spark” in line，在执行Lamda表达式时，会把当前遍历到的这行文本内容赋值给参数line，然后，执行处理逻辑”Spark” in line，也就是只有当改行文本包含“Spark”才满足条件，才会被放入到结果集中。最后，等到lines集合遍历结束后，就会得到一个结果集，这个结果集中包含了所有包含“Spark”的行。最后，对这个结果集调用count()，这是一个行动操作，会计算出结果集中的元素个数。

这里再给出另外一个实例，我们要找出文本文件中单行文本所包含的单词数量的最大值，代码如下：

```python
>>> lines = sc.textFile("file:///usr/local/spark/mycode/rdd/word.txt")
>>> lines.map(lambda line : len(line.split(" "))).reduce(lambda a,b : (a > b and a or b))
5
```

Python

上面代码中，lines是一个RDD，是String类型的RDD，因为这个RDD里面包含了很多行文本。lines.map()，是一个转换操作，之前说过，map(func)：将每个元素传递到函数func中，并将结果返回为一个新的数据集，所以，lines.map(lambda line : len(line.split(” “)))会把每行文本都传递给匿名函数，也就是传递给Lamda表达式line : len(line.split(” “))中的line，然后执行处理逻辑len(line.split(” “))。len(line.split(” “))这个处理逻辑的功能是，对line文本内容进行单词切分，得到很多个单词构成的集合，然后，计算出这个集合中的单词的个数。因此，最终lines.map(lambda line : len(line.split(” “)))转换操作得到的RDD，是一个整型RDD，里面每个元素都是整数值（也就是单词的个数）。最后，针对这个RDD[Int]，调用reduce()行动操作，完成计算。reduce()操作每次接收两个参数，取出较大者留下，然后再继续比较，例如，RDD[Int]中包含了1,2,3,4,5，那么，执行reduce操作时，首先取出1和2，把a赋值为1，把b赋值为2，然后，执行大小判断，保留2。下一次，让保留下来的2赋值给a，再从RDD[Int]中取出下一个元素3，把3赋值给b，然后，对a和b执行大小判断，保留较大者3.依此类推。最终，reduce()操作会得到最大值是5。

（备注：关于reduce()操作，你也可以参考Python部分的reduce）

实际上，如果我们把上面的lines.map(lambda line : len(line.split(” “))).reduce(lambda a,b : (a > b and a or b))分开逐步执行，你就可以更加清晰地发现每个步骤生成的RDD的类型。

```python
>>> lines = sc.textFile("file:///usr/local/spark/mycode/rdd/word.txt")
>>> lines.map(lambda line : line.split(" ")）//从上面执行结果可以发现，lines.map(line => line.split(" ")）返回的结果是分割后字符串列表List
>>> lines.map(line => len(line.split(" ")))// 这个RDD中的每个元素都是一个整数值（也就是一行文本包含的单词数）
>>> lines.map(lambda line : len(line.split(" "))).reduce(lambda a,b : (a > b and a or b))5
```

Python

## 持久化

前面我们已经说过，在Spark中，RDD采用惰性求值的机制，每次遇到行动操作，都会从头开始执行计算。如果整个Spark程序中只有一次行动操作，这当然不会有什么问题。但是，在一些情形下，我们需要多次调用不同的行动操作，这就意味着，每次调用行动操作，都会触发一次从头开始的计算。这对于迭代计算而言，代价是很大的，迭代计算经常需要多次重复使用同一组数据。
比如，下面就是多次计算同一个DD的例子：

```python
>>> list = ["Hadoop","Spark","Hive"]
>>> rdd = sc.parallelize(list)
>>> print(rdd.count()) //行动操作，触发一次真正从头到尾的计算
>>> print(','.join(rdd.collect())) //行动操作，触发一次真正从头到尾的计算Hadoop,Spark,Hive
```

Python

上面代码执行过程中，前后共触发了两次从头到尾的计算。
实际上，可以通过持久化（缓存）机制避免这种重复计算的开销。可以使用persist()方法对一个RDD标记为持久化，之所以说“标记为持久化”，是因为出现persist()语句的地方，并不会马上计算生成RDD并把它持久化，而是要等到遇到第一个行动操作触发真正计算以后，才会把计算结果进行持久化，持久化后的RDD将会被保留在计算节点的内存中被后面的行动操作重复使用。
persist()的圆括号中包含的是持久化级别参数，比如，persist(MEMORY_ONLY)表示将RDD作为反序列化的对象存储于JVM中，如果内存不足，就要按照LRU原则替换缓存中的内容。persist(MEMORY_AND_DISK)表示将RDD作为反序列化的对象存储在JVM中，如果内存不足，超出的分区将会被存放在硬盘上。一般而言，使用cache()方法时，会调用persist(MEMORY_ONLY)。
例子如下：

```python
>>> list = ["Hadoop","Spark","Hive"]
>>> rdd = sc.parallelize(list)
>>> rdd.cache()  //会调用persist(MEMORY_ONLY)，但是，语句执行到这里，并不会缓存rdd，这是rdd还没有被计算生成
>>> print(rdd.count()) //第一次行动操作，触发一次真正从头到尾的计算，这时才会执行上面的rdd.cache()，把这个rdd放到缓存中
3
>>> print(','.join(rdd.collect())) //第二次行动操作，不需要触发从头到尾的计算，只需要重复使用上面缓存中的rddHadoop,Spark,Hive
```

Python

最后，可以使用unpersist()方法手动地把持久化的RDD从缓存中移除。

## 分区

RDD是弹性分布式数据集，通常RDD很大，会被分成很多个分区，分别保存在不同的节点上。RDD分区的一个分区原则是使得分区的个数尽量等于集群中的CPU核心（core）数目。
对于不同的Spark部署模式而言（本地模式、Standalone模式、YARN模式、Mesos模式），都可以通过设置spark.default.parallelism这个参数的值，来配置默认的分区数目，一般而言：
*本地模式：默认为本地机器的CPU数目，若设置了local[N],则默认为N；
*Apache Mesos：默认的分区数为8；
*Standalone或YARN：在“集群中所有CPU核心数目总和”和“2”二者中取较大值作为默认值；

因此，对于parallelize而言，如果没有在方法中指定分区数，则默认为spark.default.parallelism，比如：

```python
>>> array = [1,2,3,4,5]
>>> rdd = sc.parallelize(array,2) #设置两个分区
```

Python

对于textFile而言，如果没有在方法中指定分区数，则默认为min(defaultParallelism,2)，其中，defaultParallelism对应的就是spark.default.parallelism。
如果是从HDFS中读取文件，则分区数为文件分片数(比如，128MB/片)。

## 打印元素

在实际编程中，我们经常需要把RDD中的元素打印输出到屏幕上（标准输出stdout），一般会采用语句rdd.foreach(print)或者rdd.map(print)。当采用本地模式（local）在单机上执行时，这些语句会打印出一个RDD中的所有元素。但是，当采用集群模式执行时，在worker节点上执行打印语句是输出到worker节点的stdout中，而不是输出到任务控制节点Driver Program中，因此，任务控制节点Driver Program中的stdout是不会显示打印语句的这些输出内容的。为了能够把所有worker节点上的打印输出信息也显示到Driver Program中，可以使用collect()方法，比如，rdd.collect().foreach(print)，但是，由于collect()方法会把各个worker节点上的所有RDD元素都抓取到Driver Program中，因此，这可能会导致内存溢出。因此，当你只需要打印RDD的部分元素时，可以采用语句rdd.take(100).foreach(print)。



# 键值对RDD(Python版)

虽然RDD中可以包含任何类型的对象，但是“键值对”是一种比较常见的RDD元素类型，分组和聚合操作中经常会用到。
Spark操作中经常会用到“键值对RDD”（Pair RDD），用于完成聚合计算。普通RDD里面存储的数据类型是Int、String等，而“键值对RDD”里面存储的数据类型是“键值对”。

## 创建RDD之前的准备工作

在即将进行相关的实践操作之前，我们首先要登录Linux系统（本教程统一采用hadoop用户登录），然后，打开命令行“终端”，请按照下面的命令启动Hadoop中的HDFS组件：

```bash
cd  /usr/local/hadoop./sbin/start-dfs.sh
```



然后，我们按照下面命令启动pyspark：

```bash
cd /usr/local/spark./bin/pyspark
```



然后，新建第二个“终端”，方法是，在前面已经建设的第一个终端窗口的左上方，点击“终端”菜单，在弹出的子菜单中选择“新建终端”，就可以打开第二个终端窗口，现在，我们切换到第二个终端窗口，在第二个终端窗口中，执行以下命令，进入之前已经创建好的“/usr/local/spark/mycode/”目录，在这个目录下新建pairrdd子目录，用来存放本章的代码和相关文件：

```bash
cd /usr/local/spark/mycode/mkdir pairrdd
```



然后，使用vim编辑器，在pairrdd目录下新建一个word.txt文件，你可以在文件里面随便输入几行英文语句用来测试。

经过上面的准备工作以后，我们就可以开始创建RDD了。

## 键值对RDD的创建

### 第一种创建方式：从文件中加载

我们可以采用多种方式创建键值对RDD，其中一种主要方式是使用map()函数来实现，如下：

```python
>>>  lines = sc.textFile("file:///usr/local/spark/mycode/pairrdd/word.txt")
>>> pairRDD = lines.flatMap(lambda line : line.split(" ")).map(lambda word : (word,1))
>>> pairRDD.foreach(print)

(i,1)(love,1)(hadoop,1)(i,1)(love,1)(Spark,1)(Spark,1)(is,1)(fast,1)(than,1)(hadoop,1)
```

Python

我们之前在“第一个Spark应用程序:WordCount”章节已经详细解释过类似代码，所以，上面代码不再做细节分析。从代码执行返回信息：pairRDD: org.apache.spark.rdd.RDD[(String, Int)] = MapPartitionsRDD[3] at map at :29，可以看出，返回的结果是键值对类型的RDD，即RDD[(String, Int)]。从pairRDD.foreach(println)执行的打印输出结果也可以看到，都是由(单词,1)这种形式的键值对。

### 第二种创建方式：通过并行集合（列表）创建RDD

```python
>>> list = ["Hadoop","Spark","Hive","Spark"]
>>> rdd = sc.parallelize(list)
>>> pairRDD = rdd.map(lambda word : (word,1))
>>> pairRDD.foreach(print)

(Hadoop,1)(Spark,1)(Hive,1)(Spark,1)
```

Python

我们下面实例都是采用这种方式得到的pairRDD作为基础。

## 常用的键值对转换操作

常用的键值对转换操作包括reduceByKey()、groupByKey()、sortByKey()、join()、cogroup()等，下面我们通过实例来介绍。

### reduceByKey(func)

reduceByKey(func)的功能是，使用func函数合并具有相同键的值。比如，reduceByKey((a,b) => a+b)，有四个键值对(“spark”,1)、(“spark”,2)、(“hadoop”,3)和(“hadoop”,5)，对具有相同key的键值对进行合并后的结果就是：(“spark”,3)、(“hadoop”,8)。可以看出，(a,b) => a+b这个Lamda表达式中，a和b都是指value，比如，对于两个具有相同key的键值对(“spark”,1)、(“spark”,2)，a就是1，b就是2。
我们对上面第二种方式创建得到的pairRDD进行reduceByKey()操作，代码如下：

```python
>>> pairRDD.reduceByKey(lambda a,b : a+b).foreach(print)

(Spark,2)(Hive,1)(Hadoop,1)
```

Python

### groupByKey()

groupByKey()的功能是，对具有相同键的值进行分组。比如，对四个键值对(“spark”,1)、(“spark”,2)、(“hadoop”,3)和(“hadoop”,5)，采用groupByKey()后得到的结果是：(“spark”,(1,2))和(“hadoop”,(3,5))。
我们对上面第二种方式创建得到的pairRDD进行groupByKey()操作，代码如下：

```python
>>> pairRDD.groupByKey()

PythonRDD[11] at RDD at PythonRDD.scala:48
>>> pairRDD.groupByKey().foreach(print)


('spark', <pyspark.resultiterable.ResultIterable object at 0x7f1869f81f60>)('hadoop', <pyspark.resultiterable.ResultIterable object at 0x7f1869f81f60>)('hive', <pyspark.resultiterable.ResultIterable object at 0x7f1869f81f60>)
```

Python

### keys()

keys()只会把键值对RDD中的key返回形成一个新的RDD。比如，对四个键值对(“spark”,1)、(“spark”,2)、(“hadoop”,3)和(“hadoop”,5)构成的RDD，采用keys()后得到的结果是一个RDD[Int]，内容是{“spark”,”spark”,”hadoop”,”hadoop”}。
我们对上面第二种方式创建得到的pairRDD进行keys操作，代码如下：

```python
>>> pairRDD.keys()
PythonRDD[20] at RDD at PythonRDD.scala:48
>>> pairRDD.keys().foreach(print)
HadoopSparkHiveSpark
```

Python

### values()

values()只会把键值对RDD中的value返回形成一个新的RDD。比如，对四个键值对(“spark”,1)、(“spark”,2)、(“hadoop”,3)和(“hadoop”,5)构成的RDD，采用values()后得到的结果是一个RDD[Int]，内容是{1,2,3,5}。
我们对上面第二种方式创建得到的pairRDD进行values()操作，代码如下：

```python
scala> pairRDD.values()PythonRDD[22] at RDD at PythonRDD.scala:48scala> pairRDD.values().foreach(print)1111
```

Python

### sortByKey()

sortByKey()的功能是返回一个根据键排序的RDD。
我们对上面第二种方式创建得到的pairRDD进行keys操作，代码如下：

```python
>>> pairRDD.sortByKey()PythonRDD[30] at RDD at PythonRDD.scala:48>>> pairRDD.sortByKey().foreach(print)(Hadoop,1)(Hive,1)(Spark,1)(Spark,1)
```

Python

### mapValues(func)

我们经常会遇到一种情形，我们只想对键值对RDD的value部分进行处理，而不是同时对key和value进行处理。对于这种情形，Spark提供了mapValues(func)，它的功能是，对键值对RDD中的每个value都应用一个函数，但是，key不会发生变化。比如，对四个键值对(“spark”,1)、(“spark”,2)、(“hadoop”,3)和(“hadoop”,5)构成的pairRDD，如果执行pairRDD.mapValues(lambda x : x+1)，就会得到一个新的键值对RDD，它包含下面四个键值对(“spark”,2)、(“spark”,3)、(“hadoop”,4)和(“hadoop”,6)。
我们对上面第二种方式创建得到的pairRDD进行keys操作，代码如下：

```python
>>> pairRDD.mapValues(lambda x : x+1)
PythonRDD[38] at RDD at PythonRDD.scala:48
>>> pairRDD.mapValues( lambda x : x+1).foreach(print)
(Hadoop,2)(Spark,2)(Hive,2)(Spark,2)
```

Python

### join

join(连接)操作是键值对常用的操作。“连接”(join)这个概念来自于关系数据库领域，因此，join的类型也和关系数据库中的join一样，包括内连接(join)、左外连接(leftOuterJoin)、右外连接(rightOuterJoin)等。最常用的情形是内连接，所以，join就表示内连接。
对于内连接，对于给定的两个输入数据集(K,V1)和(K,V2)，只有在两个数据集中都存在的key才会被输出，最终得到一个(K,(V1,V2))类型的数据集。

比如，pairRDD1是一个键值对集合{(“spark”,1)、(“spark”,2)、(“hadoop”,3)和(“hadoop”,5)}，pairRDD2是一个键值对集合{(“spark”,”fast”)}，那么，pairRDD1.join(pairRDD2)的结果就是一个新的RDD，这个新的RDD是键值对集合{(“spark”,1,”fast”),(“spark”,2,”fast”)}。对于这个实例，我们下面在pyspark中运行一下：

```python
>>> pairRDD1 = sc.parallelize([('spark',1),('spark',2),('hadoop',3),('hadoop',5)])>>> pairRDD2 = sc.parallelize([('spark','fast')])>>> pairRDD1.join(pairRDD2)PythonRDD[49] at RDD at PythonRDD.scala:48 >>> pairRDD1.join(pairRDD2).foreach(print)
```

Python

## 一个综合实例

题目：给定一组键值对(“spark”,2),(“hadoop”,6),(“hadoop”,4),(“spark”,6)，键值对的key表示图书名称，value表示某天图书销量，请计算每个键对应的平均值，也就是计算每种图书的每天平均销量。
很显然，对于上面的题目，结果是很显然的，(“spark”,4),(“hadoop”,5)。
下面，我们在pyspark中演示代码执行过程：

```python
>>> rdd = sc.parallelize([("spark",2),("hadoop",6),("hadoop",4),("spark",6)])
>>> rdd.mapValues(lambda x : (x,1)).reduceByKey(lambda x,y : (x[0]+y[0],x[1] + y[1])).mapValues(lambda x : (x[0] / x[1])).collect()
```

Python

要注意，上面语句中，mapValues(lambda x : (x,1))中出现了变量x，reduceByKey(lambda x,y : (x[0]+y[0],x[1]+ y[1]))中也出现了变量x，mapValues(lambda x : (x[0] / x[1]))也出现了变量x。但是，必须要清楚，这三个地方出现的x，虽然都具有相同的变量名称x，但是，彼此之间没有任何关系，它们都处在不同的变量作用域内。如果你觉得这样会误导自己，造成理解上的掌握，实际上，你可以把三个出现x的地方分别替换成x1、x2、x3也是可以的，但是，很显然没有必要这么做。
上面是完整的语句和执行过程，可能不太好理解，下面我们进行逐条语句分解给大家介绍。每条语句执行后返回的屏幕信息，可以帮助大家更好理解语句的执行效果，比如生成了什么类型的RDD。

（1）首先构建一个数组，数组里面包含了四个键值对，然后，调用parallelize()方法生成RDD，从执行结果反馈信息，可以看出，rdd类型是RDD[(String, Int)]。

```python
>>> rdd = sc.parallelize([("spark",2),("hadoop",6),("hadoop",4),("spark",6)])
```

Python

（2）针对构建得到的rdd，我们调用mapValues()函数，把rdd中的每个每个键值对(key,value)的value部分进行修改，把value转换成键值对(value,1)，其中，数值1表示这个key在rdd中出现了1次，为什么要记录出现次数呢？因为，我们最终要计算每个key对应的平均值，所以，必须记住这个key出现了几次，最后用value的总和除以key的出现次数，就是这个key对应的平均值。比如，键值对(“spark”,2)经过mapValues()函数处理后，就变成了(“spark”,(2,1))，其中，数值1表示“spark”这个键的1次出现。下面就是rdd.mapValues()操作在spark-shell中的执行演示：
scala> rdd.mapValues(x => (x,1)).collect()
res23: Array[(String, (Int, Int))] = Array((spark,(2,1)), (hadoop,(6,1)), (hadoop,(4,1)), (spark,(6,1)))
上面语句中，collect()是一个行动操作，功能是以数组的形式返回数据集中的所有元素，当我们要实时查看一个RDD中的元素内容时，就可以调用collect()函数。

（3）然后，再对上一步得到的RDD调用reduceByKey()函数，在spark-shell中演示如下：

```python
>>> rdd.mapValues(lambda x : (x,1)).reduceByKey(lambda x,y : (x[0]+y[0],x[1] + y[1])).collect()
```

Python

这里，必须要十分准确地理解reduceByKey()函数的功能。可以参考上面我们对该函数的介绍，reduceByKey(func)的功能是使用func函数合并具有相同键的值。这里的func函数就是Lamda表达式 x,y : (x[0]+y[0],x[1] + y[1])，这个表达式中，x和y都是value，而且是具有相同key的两个键值对所对应的value，比如，在这个例子中， (“hadoop”,(6,1))和(“hadoop”,(4,1))这两个键值对具有相同的key，所以，对于函数中的输入参数(x,y)而言，x就是(6,1)，序列从0开始计算，x[0]表示这个键值对中的第1个元素6，x[1]表示这个键值对中的第二个元素1，y就是(4,1)，y[0]表示这个键值对中的第1个元素4，y[1]表示这个键值对中的第二个元素1，所以，函数体(x[0]+y[0],x[1] + y[2])，相当于生成一个新的键值对(key,value)，其中，key是x[0]+y[0]，也就是6+4=10，value是x[1] + y[1]，也就是1+1=2，因此，函数体(x[0]+y[0],x[1] + y[1])执行后得到的value是(10,2)，但是，要注意，这个(10,2)是reduceByKey()函数执行后，”hadoop”这个key对应的value，也就是，实际上reduceByKey()函数执行后，会生成一个键值对(“hadoop”,(10,2))，其中，10表示hadoop书籍的总销量，2表示两天。同理，reduceByKey()函数执行后会生成另外一个键值对(“spark”,(8,2))。

(4)最后，就可以求出最终结果。我们可以对上面得到的两个键值对(“hadoop”,(10,2))和(“spark”,(8,2))所构成的RDD执行mapValues()操作，得到每种书的每天平均销量。当第一个键值对(“hadoop”,(10,2))输入给mapValues(x => (x[0] / x[1]))操作时，key是”hadoop”，保持不变，value是(10,2)，会被赋值给Lamda表达式x => (x[0] / x[1]中的x，因此，x的值就是(10,2)，x[0]就是10，表示hadoop书总销量是10，x[1]就是2，表示2天，因此，hadoop书籍的每天平均销量就是x[0] / x[1]，也就是5。mapValues()输出的一个键值对就是(“hadoop”,5)。同理，当把(“spark”,(8,2))输入给mapValues()时，会计算得到另外一个键值对(“spark”,4)。在pyspark中演示如下：

```python
>>> rdd.mapValues(lambda x: (x,1)).reduceByKey(lambda x,y: (x[0]+y[0],x[1]+y[1])).mapValues(lambda x: (x[0]/x[1])).collect() [('hadoop', 5.0), ('spark', 4.0)]
```

#  共享变量(Python版)

Spark中的两个重要抽象是RDD和共享变量。上一章我们已经介绍了RDD，这里介绍共享变量。

在默认情况下，当Spark在集群的多个不同节点的多个任务上并行运行一个函数时，它会把函数中涉及到的每个变量，在每个任务上都生成一个副本。但是，有时候，需要在多个任务之间共享变量，或者在任务（Task）和任务控制节点（Driver Program）之间共享变量。为了满足这种需求，Spark提供了两种类型的变量：广播变量（broadcast variables）和累加器（accumulators）。广播变量用来把变量在所有节点的内存之间进行共享。累加器则支持在所有不同节点之间进行累加计算（比如计数或者求和）。

## 广播变量

广播变量（broadcast variables）允许程序开发人员在每个机器上缓存一个只读的变量，而不是为机器上的每个任务都生成一个副本。通过这种方式，就可以非常高效地给每个节点（机器）提供一个大的输入数据集的副本。Spark的“动作”操作会跨越多个阶段（stage），对于每个阶段内的所有任务所需要的公共数据，Spark都会自动进行广播。通过广播方式进行传播的变量，会经过序列化，然后在被任务使用时再进行反序列化。这就意味着，显式地创建广播变量只有在下面的情形中是有用的：当跨越多个阶段的那些任务需要相同的数据，或者当以反序列化方式对数据进行缓存是非常重要的。

可以通过调用 SparkContext.broadcast(v) 来从一个普通变量v中创建一个广播变量。这个广播变量就是对普通变量v的一个包装器，通过调用value方法就可以获得这个广播变量的值，具体代码如下：

```python
>>> broadcastVar = sc.broadcast([1, 2, 3])
>>> broadcastVar.value[1,2,3]
```

Python

这个广播变量被创建以后，那么在集群中的任何函数中，都应该使用广播变量broadcastVar的值，而不是使用v的值，这样就不会把v重复分发到这些节点上。此外，一旦广播变量创建后，普通变量v的值就不能再发生修改，从而确保所有节点都获得这个广播变量的相同的值。

## 累加器

累加器是仅仅被相关操作累加的变量，通常可以被用来实现计数器（counter）和求和（sum）。Spark原生地支持数值型（numeric）的累加器，程序开发人员可以编写对新类型的支持。如果创建累加器时指定了名字，则可以在Spark UI界面看到，这有利于理解每个执行阶段的进程。
一个数值型的累加器，可以通过调用SparkContext.accumulator()来创建。运行在集群中的任务，就可以使用add方法来把数值累加到累加器上，但是，这些任务只能做累加操作，不能读取累加器的值，只有任务控制节点（Driver Program）可以使用value方法来读取累加器的值。
下面是一个代码实例，演示了使用累加器来对一个数组中的元素进行求和：

```python
>>> accum = sc.accumulator(0)
>>> sc.parallelize([1, 2, 3, 4]).foreach(lambda x : accum.add(x))
>>> accum.value10
```
