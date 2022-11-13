# Spark SQL简介

Spark SQL是Spark生态系统中非常重要的组件，其前身为Shark。Shark是Spark上的数据仓库，最初设计成与Hive兼容，但是该项目于2014年开始停止开发，转向Spark SQL。Spark SQL全面继承了Shark，并进行了优化。

## 从Shark说起

Shark即Hive on Spark，为了实现与Hive兼容，Shark在HiveQL方面重用了Hive中的HiveQL解析、逻辑执行计划翻译、执行计划优化等逻辑，可以近似认为仅将物理执行计划从MapReduce作业替换成了Spark作业，通过Hive的HiveQL解析，把HiveQL翻译成Spark上的RDD操作。（要想了解更多数据仓库Hive的知识，可以参考厦门大学数据库实验室的[Hive授课视频](http://dblab.xmu.edu.cn/post/bigdata-online-course/#lesson8)、[Hive安装指南](http://dblab.xmu.edu.cn/blog/install-hive/)）
Shark的设计导致了两个问题：一是执行计划优化完全依赖于Hive，不方便添加新的优化策略；二是因为Spark是线程级并行，而MapReduce是进程级并行，因此，Spark在兼容Hive的实现上存在线程安全问题，导致Shark不得不使用另外一套独立维护的打了补丁的Hive源码分支。
Shark的实现继承了大量的Hive代码，因而给优化和维护带来了大量的麻烦，特别是基于MapReduce设计的部分，成为整个项目的瓶颈。因此，在2014年的时候，Shark项目中止，并转向Spark SQL的开发。

## Spark SQL设计

Spark SQL的架构如图16-12所示，在Shark原有的架构上重写了逻辑执行计划的优化部分，解决了Shark存在的问题。Spark SQL在Hive兼容层面仅依赖HiveQL解析和Hive元数据，也就是说，从HQL被解析成抽象语法树（AST）起，就全部由Spark SQL接管了。Spark SQL执行计划生成和优化都由Catalyst（函数式关系查询优化框架）负责。

![img](http://dblab.xmu.edu.cn/blog/wp-content/uploads/2016/11/%E5%9B%BE16-12-Spark-SQL%E6%9E%B6%E6%9E%84.jpg)
图16-12-Spark-SQL架构

Spark SQL增加了SchemaRDD（即带有Schema信息的RDD），使用户可以在Spark SQL中执行SQL语句，数据既可以来自RDD，也可以来自Hive、HDFS、Cassandra等外部数据源，还可以是JSON格式的数据。Spark SQL目前支持Scala、Java、Python三种语言，支持SQL-92规范。从Spark1.2 升级到Spark1.3以后，Spark SQL中的SchemaRDD变为了DataFrame，DataFrame相对于SchemaRDD有了较大改变,同时提供了更多好用且方便的API，如图16-13所示。

![img](http://dblab.xmu.edu.cn/blog/wp-content/uploads/2016/11/%E5%9B%BE16-13-Spark-SQL%E6%94%AF%E6%8C%81%E7%9A%84%E6%95%B0%E6%8D%AE%E6%A0%BC%E5%BC%8F%E5%92%8C%E7%BC%96%E7%A8%8B%E8%AF%AD%E8%A8%80.jpg)

图16-13-Spark-SQL支持的数据格式和编程语言

Spark SQL可以很好地支持SQL查询，一方面，可以编写Spark应用程序使用SQL语句进行数据查询，另一方面，也可以使用标准的数据库连接器（比如JDBC或ODBC）连接Spark进行SQL查询，这样，一些市场上现有的商业智能工具（比如Tableau）就可以很好地和Spark SQL组合起来使用，从而使得这些外部工具借助于Spark SQL也能获得大规模数据的处理分析能力。

# DataFrame与RDD的区别

DataFrame的推出，让Spark具备了处理大规模结构化数据的能力，不仅比原有的RDD转化方式更加简单易用，而且获得了更高的计算性能。Spark能够轻松实现从MySQL到DataFrame的转化，并且支持SQL查询。

![img](http://dblab.xmu.edu.cn/blog/wp-content/uploads/2016/11/DataFrame-RDD.jpg)
图 DataFrame与RDD的区别

从上面的图中可以看出DataFrame和RDD的区别。RDD是分布式的 Java对象的集合，比如，RDD[Person]是以Person为类型参数，但是，Person类的内部结构对于RDD而言却是不可知的。DataFrame是一种以RDD为基础的分布式数据集，也就是分布式的Row对象的集合（每个Row对象代表一行记录），提供了详细的结构信息，也就是我们经常说的模式（schema），Spark SQL可以清楚地知道该数据集中包含哪些列、每列的名称和类型。

和RDD一样，DataFrame的各种变换操作也采用惰性机制，只是记录了各种转换的逻辑转换路线图（是一个DAG图），不会发生真正的计算，这个DAG图相当于一个逻辑查询计划，最终，会被翻译成物理查询计划，生成RDD DAG，按照之前介绍的RDD DAG的执行方式去完成最终的计算得到结果。

# DataFrame的创建

从Spark2.0以上版本开始，Spark使用全新的SparkSession接口替代Spark1.6中的SQLContext及HiveContext接口来实现其对数据加载、转换、处理等功能。SparkSession实现了SQLContext及HiveContext所有功能。

SparkSession支持从不同的数据源加载数据，并把数据转换成DataFrame，并且支持把DataFrame转换成SQLContext自身中的表，然后使用SQL语句来操作数据。SparkSession亦提供了HiveQL以及其他依赖于Hive的功能的支持。

下面我们就介绍如何使用SparkSession来创建DataFrame。

请进入Linux系统，打开“终端”，进入Shell命令提示符状态。
首先，请找到样例数据。 Spark已经为我们提供了几个样例数据，就保存在“/usr/local/spark/examples/src/main/resources/”这个目录下，这个目录下有两个样例数据people.json和people.txt。
people.json文件的内容如下：

```python
{"name":"Michael"}
{"name":"Andy", "age":30}
{"name":"Justin", "age":19}
```

people.txt文件的内容如下：

```python
Michael, 29
Andy, 30
Justin, 19
```

下面我们就介绍如何从people.json文件中读取数据并生成DataFrame并显示数据（从people.txt文件生成DataFrame需要后面将要介绍的另外一种方式）。
请使用如下命令打开pyspark：

```bash
cd /usr/local/spark./bin/pyspark
```

进入到pyspark状态后执行下面命令：

```python
>>> spark=SparkSession.builder.getOrCreate()
>>> df = spark.read.json("file:///usr/local/spark/examples/src/main/resources/people.json")
>>> df.show()
+----+-------+
| age|   name|
+----+-------+
|null|Michael|
|  30|   Andy|
|  19| Justin|
+----+-------+
```

现在，我们可以执行一些常用的DataFrame操作。

```python
// 打印模式信息
>>> df.printSchema()
root
 |-- age: long (nullable = true)
 |-- name: string (nullable = true)

// 选择多列
>>> df.select(df.name,df.age + 1).show()
+-------+---------+
|   name|(age + 1)|
+-------+---------+
|Michael|     null|
|   Andy|       31|
| Justin|       20|
+-------+---------+

// 条件过滤
>>> df.filter(df.age > 20 ).show()
+---+----+
|age|name|
+---+----+
| 30|Andy|
+---+----+

// 分组聚合
>>> df.groupBy("age").count().show()
+----+-----+
| age|count|
+----+-----+
|  19|    1|
|null|    1|
|  30|    1|
+----+-----+

// 排序
>>> df.sort(df.age.desc()).show()
+----+-------+
| age|   name|
+----+-------+
|  30|   Andy|
|  19| Justin|
|null|Michael|
+----+-------+

//多列排序
>>> df.sort(df.age.desc(), df.name.asc()).show()
+----+-------+
| age|   name|
+----+-------+
|  30|   Andy|
|  19| Justin|
|null|Michael|
+----+-------+

//对列进行重命名
>>> df.select(df.name.alias("username"),df.age).show()
+--------+----+
|username| age|
+--------+----+
| Michael|null|
|    Andy|  30|
|  Justin|  19|
+--------+----+
```



# 从RDD转换得到DataFrame

Spark官网提供了两种方法来实现从RDD转换得到DataFrame，第一种方法是，利用反射来推断包含特定类型对象的RDD的schema，适用对已知数据结构的RDD转换；第二种方法是，使用编程接口，构造一个schema并将其应用在已知的RDD上。

## 利用反射机制推断RDD模式

在利用反射机制推断RDD模式时,我们会用到toDF()方法
下面是在pyspark中执行命令以及反馈的信息：

```python
>>> from pyspark.sql.types import Row
>>> def f(x):
...     rel = {}
...     rel['name'] = x[0]
...     rel['age'] = x[1]
...     return rel
...
>>> peopleDF = sc.textFile("file:///usr/local/spark/examples/src/main/resources/people.txt").map(lambda line : line.split(',')).map(lambda x: Row(**f(x))).toDF()
>>> peopleDF.createOrReplaceTempView("people")  //必须注册为临时表才能供下面的查询使用

>>> personsDF = spark.sql("select * from people")
>>> personsDF.rdd.map(lambda t : "Name:"+t[0]+","+"Age:"+t[1]).foreach(print)

Name: 19,Age:Justin
Name: 29,Age:Michael
Name: 30,Age:Andy
```

## 使用编程方式定义RDD模式

使用createDataFrame(rdd, schema)编程方式定义RDD模式。

```python
>>>  from pyspark.sql.types import Row
>>>  from pyspark.sql.types import StructType
>>> from pyspark.sql.types import StructField
>>> from pyspark.sql.types import StringType

//生成 RDD
>>> peopleRDD = sc.textFile("file:///usr/local/spark/examples/src/main/resources/people.txt")

//定义一个模式字符串
>>> schemaString = "name age"

//根据模式字符串生成模式
>>> fields = list(map( lambda fieldName : StructField(fieldName, StringType(), nullable = True), schemaString.split(" ")))
>>> schema = StructType(fields)
//从上面信息可以看出，schema描述了模式信息，模式中包含name和age两个字段


>>> rowRDD = peopleRDD.map(lambda line : line.split(',')).map(lambda attributes : Row(attributes[0], attributes[1]))

>>> peopleDF = spark.createDataFrame(rowRDD, schema)

//必须注册为临时表才能供下面查询使用
scala> peopleDF.createOrReplaceTempView("people")

>>> results = spark.sql("SELECT * FROM people")
>>> results.rdd.map( lambda attributes : "name: " + attributes[0]+","+"age:"+attributes[1]).foreach(print)

name: Michael,age: 29
name: Andy,age: 30
name: Justin,age: 19

```

在上面的代码中，peopleRDD.map(lambda line : line.split(‘,’))作用是对people这个RDD中的每一行元素都进行解析。比如，people这个RDD的第一行是：

```
Michael, 29
```

这行内容经过peopleRDD.map(lambda line : line.split(‘,’)).操作后，就得到一个集合{Michael,29}。后面经过map(lambda attributes : Row(attributes[0], attributes[1]))操作时，这时的p就是这个集合{Michael,29}，这时p[0]就是Micheael，p[1]就是29，map(lambda attributes : Row(attributes[0], attributes[1]))就会生成一个Row对象，这个对象里面包含了两个字段的值，这个Row对象就构成了rowRDD中的其中一个元素。因为people有3行文本，所以，最终，rowRDD中会包含3个元素，每个元素都是org.apache.spark.sql.Row类型。实际上，Row对象只是对基本数据类型（比如整型或字符串）的数组的封装，本质就是一个定长的字段数组。
peopleDF = spark.createDataFrame(rowRDD, schema)，这条语句就相当于建立了rowRDD数据集和模式之间的对应关系，从而我们就知道对于rowRDD的每行记录，第一个字段的名称是schema中的“name”，第二个字段的名称是schema中的“age”。

## 把RDD保存成文件

这里介绍如何把RDD保存成文本文件，后面还会介绍其他格式的保存。

### 第1种保存方法

进入pyspark执行下面命令：

```python
>>> peopleDF = spark.read.format("json").load("file:///usr/local/spark/examples/src/main/resources/people.json")

>>> peopleDF.select("name", "age").write.format("csv").save("file:///usr/local/spark/mycode/newpeople.csv")
```

可以看出，这里使用select(“name”, “age”)确定要把哪些列进行保存，然后调用write.format(“csv”).save ()保存成csv文件。在后面小节中，我们还会介绍其他保存方式。
另外，write.format()支持输出 json,parquet, jdbc, orc, libsvm, csv, text等格式文件，如果要输出文本文件，可以采用write.format(“text”)，但是，需要注意，只有select()中只存在一个列时，才允许保存成文本文件，如果存在两个列，比如select(“name”, “age”)，就不能保存成文本文件。

上述过程执行结束后，可以打开第二个终端窗口，在Shell命令提示符下查看新生成的newpeople.csv：

```bash
cd  /usr/local/spark/mycode/ls
```

Shell 命令

可以看到/usr/local/spark/mycode/这个目录下面有个newpeople.csv文件夹（注意，不是文件），这个文件夹中包含下面两个文件：

```
part-r-00000-33184449-cb15-454c-a30f-9bb43faccac1.csv
_SUCCESS
```

不用理会_SUCCESS这个文件，只要看一下part-r-00000-33184449-cb15-454c-a30f-9bb43faccac1.csv这个文件，可以用vim编辑器打开这个文件查看它的内容，该文件内容如下：

```python
Michael,
Andy,30
Justin,19
```

因为people.json文件中，Michael这个名字不存在对应的age，所以，上面第一行逗号后面没有内容。
如果我们要再次把newpeople.csv中的数据加载到RDD中，可以直接使用newpeople.csv目录名称，而不需要使用part-r-00000-33184449-cb15-454c-a30f-9bb43faccac1.csv 文件，如下：

```python
>>> textFile = sc.textFile("file:///usr/local/spark/mycode/newpeople.csv")
>>> textFile.foreach(print)Justin,19Michael,Andy,30
```

Python

### 第2种保存方法

进入pyspark执行下面命令：

```python
>>> peopleDF = spark.read.format("json").load("file:///usr/local/spark/examples/src/main/resources/people.json"
>>> peopleDF.rdd.saveAsTextFile("file:///usr/local/spark/mycode/newpeople.txt")
```

Python

可以看出，我们是把DataFrame转换成RDD，然后调用saveAsTextFile()保存成文本文件。在后面小节中，我们还会介绍其他保存方式。
上述过程执行结束后，可以打开第二个终端窗口，在Shell命令提示符下查看新生成的newpeople.txt：

```bash
cd  /usr/local/spark/mycode/ls
```

Shell 命令

可以看到/usr/local/spark/mycode/这个目录下面有个newpeople.txt文件夹（注意，不是文件），这个文件夹中包含下面两个文件：

```python
part-00000
_SUCCESS
```

不用理会_SUCCESS这个文件，只要看一下part-00000这个文件，可以用vim编辑器打开这个文件查看它的内容，该文件内容如下：

```python
[null,Michael]
[30,Andy]
[19,Justin]
```

如果我们要再次把newpeople.txt中的数据加载到RDD中，可以直接使用newpeople.txt目录名称，而不需要使用part-00000文件，如下：

```python
>>> textFile = sc.textFile("file:///usr/local/spark/mycode/newpeople.txt")
>>> textFile.foreach(print)[null,Michael][30,Andy][19,Justin]
```



# 读写Parquet(DataFrame)(Python版)



Spark SQL可以支持Parquet、JSON、Hive等数据源，并且可以通过JDBC连接外部数据源。前面的介绍中，我们已经涉及到了JSON、文本格式的加载，这里不再赘述。这里介绍Parquet，下一节会介绍JDBC数据库连接。

Parquet是一种流行的列式存储格式，可以高效地存储具有嵌套字段的记录。Parquet是语言无关的，而且不与任何一种数据处理框架绑定在一起，适配多种语言和组件，能够与Parquet配合的组件有：

\* 查询引擎: Hive, Impala, Pig, Presto, Drill, Tajo, HAWQ, IBM Big SQL
\* 计算框架: MapReduce, Spark, Cascading, Crunch, Scalding, Kite
\* 数据模型: Avro, Thrift, Protocol Buffers, POJOs

Spark已经为我们提供了parquet样例数据，就保存在“/usr/local/spark/examples/src/main/resources/”这个目录下，有个users.parquet文件，这个文件格式比较特殊，如果你用vim编辑器打开，或者用cat命令查看文件内容，肉眼是一堆乱七八糟的东西，是无法理解的。只有被加载到程序中以后，Spark会对这种格式进行解析，然后我们才能理解其中的数据。

下面代码演示了如何从parquet文件中加载数据生成DataFrame。

```python
>>> parquetFileDF = spark.read.parquet("file:///usr/local/spark/examples/src/main/resources/users.parquet")
>>> parquetFileDF.createOrReplaceTempView("parquetFile")
>>> namesDF = spark.sql("SELECT * FROM parquetFile")
>>> namesDF.rdd.foreach(lambda person: print(person.name))
 AlyssaBen
```

Python

下面介绍如何将DataFrame保存成parquet文件。

进入pyspark执行下面命令：

```python
>>> peopleDF = spark.read.json("file:///usr/local/spark/examples/src/main/resources/people.json")
>>> peopleDF.write.parquet("file:///usr/local/spark/mycode/newpeople.parquet")
```

Python

上述过程执行结束后，可以打开第二个终端窗口，在Shell命令提示符下查看新生成的newpeople.parquet：

```bash
cd  /usr/local/spark/myCode/ls
```

Shell 命令

上面命令执行后，可以看到”/usr/local/spark/myCode/”这个目录下多了一个newpeople.parquet，不过，注意，这不是一个文件，而是一个目录（不要被newpeople.parquet中的圆点所迷惑，文件夹名称也可以包含圆点），也就是说，peopleDF.write.parquet(“file:///usr/local/spark/myCode/newpeople.parquet”)括号里面的参数是文件夹，不是文件名。下面我们可以进入newpeople.parquet目录，会发现下面2个文件：

part-r-00000-8d3a120f-b3b5-4582-b26b-f3693df80d45.snappy.parquet
_SUCCESS
这2个文件都是刚才保存生成的。现在问题来了，如果我们要再次把这个刚生成的数据又加载到DataFrame中，应该加载哪个文件呢？很简单，只要加载newpeople.parquet目录即可，而不是加载这2个文件，语句如下：

```python
>>> val users = spark.read.parquet("file:///usr/local/spark/myCode/people.parquet")
```
