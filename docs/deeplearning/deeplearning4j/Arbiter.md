## Arbiter概述



## **超参数优化**

机器学习技术中，有一组在任何训练开始之前都必须选择好的参数。这些参数被称为超参数（hyperparameter），例如K-最近邻中的k，以及支持向量机中的正则化参数。神经网络的超参数种类尤其繁多。其中一部分用于设定神经网络的结构，例如层的数量和大小。另一些则用于设定学习过程，比如学习速率和正则化方式。

传统上，这些选择是基于现有的经验法则或经过广泛的尝试和试错，这两者都不太理想。毫无疑问，这些参数的选择会对学习后获得的结果产生重大影响。Hyperparameter 优化尝试使用应用搜索策略的软件来自动化这个过程。



## **Arbiter**

Arbiter属于面向企业的DL4J机器学习/深度学习系列工具，专门用于对通过DL4J创建或导入DL4J的神经网络进行超参数优化。Arbiter让用户能确定超参数的搜索空间，运行网格搜索或随机搜索，根据给定的评分标准选择最佳配置。

何时使用Arbiter？Arbiter可用于寻找性能优秀的模型，有望节省用户调试模型超参数的时间，而代价则是增加计算时间。但是请注意，Arbiter无法实现完全自动化的神经网络调试，用户仍然需要指定一个搜索空间。搜索空间定义了每个超参数的有效值范围（例如：学习速率的上限和下限）。如果搜索空间选择不当，Arbiter可能无法找到较好的模型。

要将Arbiter添加到您的项目中，只需将以下代码添加至您的pom.xml，其中 1.0.0-beta6 是DL4J堆栈的最新发布版本。

```xml
<!-- Arbiter - used for hyperparameter optimization (grid/random search) -->

<dependency>

​    <groupId>org.deeplearning4j</groupId>

​    <artifactId>arbiter-deeplearning4j</artifactId>

​    <version>1.0.0-beta6</version>

</dependency>

<dependency>

​    <groupId>org.deeplearning4j</groupId>

​    <artifactId>arbiter-ui</artifactId>

​    <version>1.0.0-beta6</version>

</dependency>

```

Arbiter还配备了一个方便的UI，能够将优化过程的结果可视化。

使用Arbiter之前，用户应当先熟悉DL4J中的NeuralNetworkConfiguration、MultilayerNetworkConfiguration和ComputationGraphconfiguration类。



## **使用方法**

本节将概述使用Arbiter时所需的重要构造。之后的各节将深入探讨具体细节。

从总体层面上看，超参数优化需要设置一个OptimizationConfiguration类，然后通过IOptimizationRunner来运行。

以下代码展示了如何为OptimizationConfiguration编写比较流畅的构建器模式：

```java
OptimizationConfiguration configuration = new OptimizationConfiguration.Builder()
    .candidateGenerator(candidateGenerator)
    .dataSource(dataSourceClass,dataSourceProperties)
    .modelSaver(modelSaver)
    .scoreFunction(scoreFunction)
    .terminationConditions(terminationConditions)
    .build();
```

如上所示，设置超参数优化配置需要：CandidateGenerator：为超参数评估提供候选项（即潜在的超参数配置）。候选项依据一定的策略生成。目前我们支持随机搜索和网格搜索。候选项的有效配置取决于同候选项生成器关联的超参数空间。

- DataSource：DataSource用于在后台为生成的候选项提供数据，以供训练或测试。
- ModelSaver：指定超参数优化过程的结果如何保存，比如是保存至本地磁盘、数据库、HDFS还是就保存在内存中。
- ScoreFunction：表示为单个数值的指标，我们通过将其最大或最小化来确定最佳的候选项。例如：模型损失或分类准确率。
- TerminationCondition：决定超参数优化应当何时停止。例如：评估了给定数量的候选项、经过了一定的计算时间等。

随后，优化配置会和一个任务创建器一同传递给一个优化运行器。

如果生成的候选项是MultiLayerNetwork，那么设置代码如下：

```java
IOptimizationRunner runner = new LocalOptimizationRunner(configuration, new MultiLayerNetworkTaskCreator());
```

如果生成的候选项是ComputationGraph，那么设置代码如下：

```java
IOptimizationRunner runner = new LocalOptimizationRunner(configuration, new ComputationGraphTaskCreator());
```

目前运行器的唯一选择是LocalOptimizationRunner，用于在一台计算机上执行学习（即在当前的JVM中运行）。原则上也可以实现其他的执行方法（例如在Spark或云端计算机上运行）。

总结起来，设置超参数优化运行过程的步骤如下：

1. 指定超参数搜索空间
2. 为超参数搜索空间指定一个候选生成器
3. 以下各步骤可按任意顺序进行：
   4. 指定一个数据来源
   2. 指定一个模型保存器
   3. 指定一个计分函数
   4. 指定一项终止条件
8. 以下步骤需要依次进行：
9. 用上述2-6步构建一套优化配置
10. 用优化运行器运行

## **超参数搜索空间**

Arbiter的 ParameterSpace<T>类定义了某一特定超参数可接受的值的区间。ParameterSpace可以是一个简单的参数空间，比如一个连续的双精度浮点数值区间（例如学习速率），也可以比较复杂，比如MultiLayerSpace等之中的多重嵌套参数空间（为MultilayerConfiguration定义搜索空间）。

## **MultiLayerSpace**和**ComputationGraphSpace**

Arbiter中的MultiLayerSpace和ComputationGraphSpace分别对应DL4J的MultiLayerConfiguration和ComputationGraphConfiguration。它们用于为MultiLayerConfiguration和ComputationGraphConfiguration中的有效超参数设置参数空间。

此外，用户也可以设置训练周期数（epoch数）或早停机制，决定每个候选神经网络的训练何时停止。假如同时指定了EarlyStoppingConfiguration和训练周期数，则将优先使用早停机制。

MultiLayerSpace和ComputationGraphSpace的设置方法都比较方便，用户只需熟悉整数（Integer）、连续（Continuous）和离散（Discrete）参数空间，以及LayerSpaces和UpdaterSpaces。

此处唯一需要注意的是，虽然理论上可以在NeuralNetConfiguration中设置weightConstraints、l1Bias和l2Bias，但这些属性必须在MultiLayerSpace中分别为每个层/层空间（layerSpace）设置。一般而言，所有可以通过构建器设置的属性/超参数要么是固定值，要么是相应类型的参数空间。这意味着我们能扫遍MultiLayerConfiguration的几乎每一项属性，测试各种不同的架构和初始值。

以下是MultiLayerSpace的一个简单示例：

```java
ParameterSpace<Boolean> biasSpace = new DiscreteParameterSpace<>(new Boolean[]{true, false});
ParameterSpace<Integer> firstLayerSize = new IntegerParameterSpace(10,30);
ParameterSpace<Integer> secondLayerSize = new MathOp<>(firstLayerSize, Op.MUL, 3);
ParameterSpace<Double> firstLayerLR = new ContinuousParameterSpace(0.01, 0.1);
ParameterSpace<Double> secondLayerLR = new MathOp<>(firstLayerLR, Op.ADD, 0.2);

MultiLayerSpace mls =
    new MultiLayerSpace.Builder().seed(12345)
            .hasBias(biasSpace)
            .layer(new DenseLayerSpace.Builder().nOut(firstLayerSize)
                    .updater(new AdamSpace(firstLayerLR))
                    .build())
            .layer(new OutputLayerSpace.Builder().nOut(secondLayerSize)
                    .updater(new AdamSpace(secondLayerLR))
                    .build())
            .setInputType(InputType.feedForward(10))
  .numEpochs(20).build(); //Data will be fit for a fixed number of epochs
```

尤其值得一提的是，Arbiter能够改变MultiLayerSpace中的层数。下面这个简单的示例说明了这一功能，同时也展示了如何为加权损失函数设置参数搜索空间：

```java
ILossFunction[] weightedLossFns = new ILossFunction[]{
    new LossMCXENT(Nd4j.create(new double[]{1, 0.1})),
        new LossMCXENT(Nd4j.create(new double[]{1, 0.05})),
            new LossMCXENT(Nd4j.create(new double[]{1, 0.01}))};

DiscreteParameterSpace<ILossFunction> weightLossFn = new DiscreteParameterSpace<>(weightedLossFns);
MultiLayerSpace mls =
    new MultiLayerSpace.Builder().seed(12345)
        .addLayer(new DenseLayerSpace.Builder().nIn(10).nOut(10).build(),
            new IntegerParameterSpace(2, 5)) //2 to 5 identical layers
        .addLayer(new OutputLayerSpace.Builder()
            .iLossFunction(weightLossFn)
            .nIn(10).nOut(2).build())
        .backprop(true).pretrain(false).build();
```

以上创建的两到五个层是完全相同的（堆叠）。目前Arbiter尚不支持创建独立的层。

最后，也可以创建固定数量的相同层，如下所示：

```java
DiscreteParameterSpace<Activation> activationSpace = new DiscreteParameterSpace(new Activation[]{Activation.IDENTITY, Activation.ELU, Activation.RELU});
MultiLayerSpace mls = new MultiLayerSpace.Builder().updater(new Sgd(0.005))
    .addLayer(new DenseLayerSpace.Builder().activation(activationSpace).nIn(10).nOut(10).build(),
        new FixedValue<Integer>(3))
    .addLayer(new OutputLayerSpace.Builder().iLossFunction(new LossMCXENT()).nIn(10).nOut(2).build())
    .backprop(true).build();
```

在本例中，使用网格搜索将创建三个独立的体系结构。它们在各方面都是相同的，但在非输出层中所选择的激活功能是相同的。再次需要注意的是，在每个架构中创建的层是相同的（堆叠的）。

ComputationGraphSpace的创建与MultiLayerSpace非常相似。但目前仅支持固定的计算图结构。

以下是ComputationGraphSpace设置的一个简易示例：

```java
ComputationGraphSpace cgs = new ComputationGraphSpace.Builder()
                .updater(new SgdSpace(new ContinuousParameterSpace(0.0001, 0.1)))
                .l2(new ContinuousParameterSpace(0.2, 0.5))
                .addInputs("in")
                .addLayer("0",new  DenseLayerSpace.Builder().nIn(10).nOut(10).activation(
            new DiscreteParameterSpace<>(Activation.RELU,Activation.TANH).build(),"in")

        .addLayer("1", new OutputLayerSpace.Builder().nIn(10).nOut(10)
                             .activation(Activation.SOFTMAX).build(), "0")
        .setOutputs("1").setInputTypes(InputType.feedForward(10)).build();
```

## **JSON**序列化

MultiLayerSpace、ComputationGraphSpace、OptimizationConfiguration都有`toJson`和`fromJson`方法。您可以存储JSON表示以供将来使用。

指定一个候选项生成器。如前文所述，Arbiter目前支持网格和随机搜索。

随机搜索的设置比较简便，方法如下：

```java
MultiLayerSpace mls; … CandidateGenerator candidateGenerator = new RandomSearchGenerator(mls);
```

设置网格搜索也很简单。使用网格搜索时，用户还可以指定离散化数（discretization count）和模式。离散化数决定了一个连续参数分箱后的值的个数。例如，如果discretizationCount设为3，一个取值范围是[0,1]的连续参数会被转换为[0.0, 0.5, 1.0]。模式则决定了候选项的生成方式。候选项可以有序地生成（Sequential），或者按随机顺序生成（RandomOrder）。在有序模式下，第一个超参数改变速度最快，因此最后一个超参数改变速度最慢。请注意，两种模式生成的候选项一样，只是顺序不同。

以下是一个设置网格搜索的简单示例（离散化数为4，有序模式）：

```java
CandidateGenerator candidateGenerator = new GridSearchCandidateGenerator(mls, 4,
 GridSearchCandidateGenerator.Mode.Sequential);
```



## **指定数据来源**

DataSource接口指定了用于训练不同候选项的数据来自何处。实现方式非常简便。请注意，用户需要定义一个无参数的构造器。根据用户的具体需求，DataSource实现可以配置各类属性，例如微批次大小等。示例库中有一个使用MNIST数据集的DataSource实现示例，详见本教程最后的相关段落。此处需要指出的一个重点是：训练周期数（以及早停法配置）可以通过MultiLayerSpace和ComputationGraphSpace构建器设置。

## **指定一个模型/结果保存器**

Arbiter目前支持的模型保存方式包括保存至本地磁盘（FileModelSaver）或者将结果保存至内存（InMemoryResultSaver）。对于较大的模型，显然不建议使用InMemoryResultSaver。

设置方法并不复杂。FileModelSaver 构造器接受字符串格式的路径，将配置、参数和计分保存至：baseDir/0/、baseDir/1/等，索引值由OptimizationResult.getIndex()给出。InMemoryResultSaver不需要任何参数。

指定一项计分函数
计分函数共有三个类：EvaluationScoreFunction、ROCScoreFunction、RegressionScoreFunction。

EvaluationScoreFunction使用DL4J的评估指标之一。可用的指标包括准确率（ACCURACY）、F1值、精确度（PRECISION）、召回率（RECALL）、GMEASURE、MCC。以下是一个使用准确率的简单示例：

```java
ScoreFunction scoreFunction = new EvaluationScoreFunction(Evaluation.Metric.ACCURACY);
```

ROCScoreFunction计算测试数据集的AUC（ROC曲线下方的面积）或AUPRC（精确度/召回率曲线下方的面积），支持不同的ROC类型（ROC、ROCBinary和ROCMultiClass）。以下是一个使用AUC的简单示例：

```java
ScoreFunction sf = new ROCScoreFunction(ROCScoreFunction.ROCType.BINARY, ROCScoreFunction.Metric.AUC));
```

RegressionScoreFunction用于回归分析，支持DL4J所有的RegressionEvaluation指标（MSE、MAE、RMSE、RSE、PC、R2）。以下是一个简单的示例：

```
ScoreFunction sf = new RegressionScoreFunction(RegressionEvaluation.Metric.MSE);
```



## **指定一项终止条件**

Arbiter目前仅支持两种终止条件：MaxTimeCondition和MaxCandidatesCondition。MaxTimeCondition指定一个时间点，到达时间则超参数优化终止。MaxCandidatesCondition指定候选项的数量上限，达到上限时终止超参数优化。终止条件可以用列表形式指定。如果满足其中任何一项条件，则超参数优化停止。

以下是一个简单的示例，搜索在持续十五分钟或训练完十个候选项之后终止：

```java
TerminationCondition[] terminationConditions = {
    new MaxTimeCondition(15, TimeUnit.MINUTES),
    new MaxCandidatesCondition(10)
};
```

## **使用**MNIST**数据运行**Arbiter**的示例**

DL4J示例库中有一个使用MNIST数据的基本超参数优化示例（BasicHyperparameterOptimizationExample）。用户可以在此处查看这一简单的示例。这一示例也说明了Arbiter UI的设置方式。Arbiter的保存和持久化方式同DL4J的UI。有关UI的更多资料可参见此处。UI可在http://localhost:9000/arbiter访问。

- Step1：指定超参数搜索空间

```java
//First: Set up the hyperparameter configuration space. This is like a MultiLayerConfiguration, but can have either
// fixed values or values to optimize, for each hyperparameter

ParameterSpace<Double> learningRateHyperparam = new ContinuousParameterSpace(0.0001, 0.1);  //Values will be generated uniformly at random between 0.0001 and 0.1 (inclusive)
ParameterSpace<Integer> layerSizeHyperparam = new IntegerParameterSpace(16, 256);            //Integer values will be generated uniformly at random between 16 and 256 (inclusive)

MultiLayerSpace hyperparameterSpace = new MultiLayerSpace.Builder()
    //These next few options: fixed values for all models
    .weightInit(WeightInit.XAVIER)
    .l2(0.0001)
    //Learning rate hyperparameter: search over different values, applied to all models
    .updater(new SgdSpace(learningRateHyperparam))
    .addLayer(new DenseLayerSpace.Builder()
        //Fixed values for this layer:
        .nIn(784)  //Fixed input: 28x28=784 pixels for MNIST
        .activation(Activation.LEAKYRELU)
        //One hyperparameter to infer: layer size
        .nOut(layerSizeHyperparam)
        .build())
    .addLayer(new OutputLayerSpace.Builder()
        .nOut(10)
        .activation(Activation.SOFTMAX)
        .lossFunction(LossFunctions.LossFunction.MCXENT)
        .build())
    .numEpochs(2)
    .build();
```

- step2:  为超参数搜索空间指定一个候选生成器
  - 指定一个数据来源
  - 指定一个模型保存器
  - 指定一个评价函数
  - 指定一项终止条件

```java
//Now: We need to define a few configuration options
// (a) How are we going to generate candidates? (random search or grid search)
CandidateGenerator candidateGenerator = new RandomSearchGenerator(hyperparameterSpace, null);    //Alternatively: new GridSearchCandidateGenerator<>(hyperparameterSpace, 5, GridSearchCandidateGenerator.Mode.RandomOrder);

// (b) How are going to provide data? We'll use a simple data source that returns MNIST data
// Note that we set teh number of epochs in MultiLayerSpace above
Class<? extends DataSource> dataSourceClass = ExampleDataSource.class;
Properties dataSourceProperties = new Properties();
dataSourceProperties.setProperty("minibatchSize", "64");

// (c) How we are going to save the models that are generated and tested?
//     In this example, let's save them to disk the working directory
//     This will result in examples being saved to arbiterExample/0/, arbiterExample/1/, arbiterExample/2/, ...
String baseSaveDirectory = "arbiterExample/";
File f = new File(baseSaveDirectory);
if (f.exists()) //noinspection ResultOfMethodCallIgnored
    f.delete();
//noinspection ResultOfMethodCallIgnored
f.mkdir();
ResultSaver modelSaver = new FileModelSaver(baseSaveDirectory);

// (d) What are we actually trying to optimize?
//     In this example, let's use classification accuracy on the test set
//     See also ScoreFunctions.testSetF1(), ScoreFunctions.testSetRegression(regressionValue) etc
ScoreFunction scoreFunction = new EvaluationScoreFunction(Metric.ACCURACY);


// (e) When should we stop searching? Specify this with termination conditions
//     For this example, we are stopping the search at 15 minutes or 10 candidates - whichever comes first
TerminationCondition[] terminationConditions = {
    new MaxTimeCondition(15, TimeUnit.MINUTES),
    new MaxCandidatesCondition(10)};
```

- step3:  用上述2-6步构建一套优化配置
  - 用优化运行器运行

```java
//Given these configuration options, let's put them all together:
OptimizationConfiguration configuration = new OptimizationConfiguration.Builder()
    .candidateGenerator(candidateGenerator)
    .dataSource(dataSourceClass,dataSourceProperties)
    .modelSaver(modelSaver)
    .scoreFunction(scoreFunction)
    .terminationConditions(terminationConditions)
    .build();

//And set up execution locally on this machine:
IOptimizationRunner runner = new LocalOptimizationRunner(configuration, new MultiLayerNetworkTaskCreator());


//Start the UI. Arbiter uses the same storage and persistence approach as DL4J's UI
//Access at http://localhost:9000/arbiter
StatsStorage ss = new FileStatsStorage(new File(System.getProperty("java.io.tmpdir"), "arbiterExampleUiStats.dl4j"));
runner.addListeners(new ArbiterStatusListener(ss));
UIServer.getInstance().attach(ss);


//Start the hyperparameter optimization
runner.execute();
```

- step4 ： 根据评价函数得到最优模型

```java
//Print out some basic stats regarding the optimization procedure
String s = "Best score: " + runner.bestScore() + "\n" +
    "Index of model with best score: " + runner.bestScoreCandidateIndex() + "\n" +
    "Number of configurations evaluated: " + runner.numCandidatesCompleted() + "\n";
System.out.println(s);


//Get all results, and print out details of the best result:
int indexOfBestResult = runner.bestScoreCandidateIndex();
List<ResultReference> allResults = runner.getResults();

OptimizationResult bestResult = allResults.get(indexOfBestResult).getResult();
MultiLayerNetwork bestModel = (MultiLayerNetwork) bestResult.getResultReference().getResultModel();

System.out.println("\n\nConfiguration of best model:\n");
System.out.println(bestModel.getLayerWiseConfigurations().toJson());

//Wait a while before exiting
Thread.sleep(60000);
UIServer.getInstance().stop();
```



## **超参数调试建议**

斯坦福大学CS231N课程中有关于超参数优化的精彩内容可供参考。相关技术的总结如下：

- 优先使用随机搜索，而非网格搜索。随机和网格搜索方法的对比参见《Random Search for Hyper-parameter Optimization（超参数优化的随机搜索）》（Bergstra与Bengio，2012）。
- 用由粗到精的方式运行搜索（从一两个训练周期的粗参数搜索开始，选出最佳候选项，再进行持续更多个训练周期的精细搜索，如此重复）
- 对于学习速率、l2等特定的超参数，应当使用LogUniformDistribution
- 注意靠近参数搜索空间边界附近的值
