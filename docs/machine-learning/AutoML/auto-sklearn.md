

### Auto-skearn 简介


Auto-sklearn 提供了开箱即用的监督型自动机器学习。从名字可以看出，Auto-Sklearn主要基于sklearn机器学习库，使用方法也与之类似，这让熟悉sklearn的开发者很容易切换到Auto-Sklearn。Auto-Sklearn 的核心思想是组合算法选择和超参数优化(CASH: Combined Algorithm Selection and Hyperparameter optimization)：同时考虑学习算法以及算法的超参选择。在模型方面，除了sklearn提供的机器学习模型，还加入了xgboost算法支持；在框架整体调优方面，使用了贝叶斯优化。


该库由 Matthias Feurer 等人提出，技术细节请查阅论文《Efficient and Robust Machine Learning》。Feurer 在这篇论文中写道：我们提出了一个新的、基于 scikit-learn 的鲁棒 AutoML 系统，其中使用 15 个分类器、14 种特征预处理方法和 4 种数据预处理方法，生成了一个具有 110 个超参数的结构化假设空间。


![img](http://image.techweb.com.cn/upload/roll/2020/09/27/202009279365_9585.png)

图源：《Efficient and Robust Automated Machine Learning》


auto-sklearn 可能最适合刚接触 AutoML 的用户。除了发现数据集的数据准备和模型选择之外，该库还可以从在类似数据集上表现良好的模型中学习。表现最好的模型聚集在一个集合中。

Auto-Sklearn的分析步骤如下：在数据导入后，加上一个元学习步骤(meta-learning)；在模型训练后，通过贝叶斯方法更新数据处理和特征选择；最终，在模型训练完成之后，将模型进行组合(build ensemble)以得到最优的估计。


在高效实现方面，auto-sklearn 需要的用户交互最少。该库可以使用的两个主要类是 AutoSklearnClassifier 和 AutoSklearnRegressor，它们分别用来做分类和回归任务。两者具有相同的用户指定参数，其中最重要的是时间约束和集合大小。

### Auto-sklearn 能 Auto 到什么地步？

- 常规 ML framework 如上图灰色部分：导入数据-数据清洗-特征工程-分类器-输出预测值
- auto部分如下图绿色方框：在ML framework 左边新增 meta-learning，在右边新增 build-ensemble，对于调超参数，用的是贝叶斯优化。
- 自动学习样本数据: meta-learning，去学习样本数据的模样，自动推荐合适的模型。比如文本数据用什么模型比较好，比如很多的离散数据用什么模型好。
- 自动调超参：Bayesian optimizer，贝叶斯优化。
- 自动模型集成: build-ensemble，模型集成，在一般的比赛中都会用到的技巧。多个模型组合成一个更强更大的模型。往往能提高预测准确性。


### Auto-sklearn整体框架


Auto-sklearn是基于sklearn库，因此会有惊艳强大的模型库和数据/特征预处理库，专业出身的设定。

- *16 classifiers*（可以被指定或者筛选，include_estimators=[“random_forest”, ]）

>adaboost, bernoulli_nb, decision_tree, extra_trees, gaussian_nb, gradient_boosting, k_nearest_neighbors, lda, liblinear_svc, libsvm_svc, multinomial_nb, passive_aggressive, qda, random_forest, sgd, xgradient_boosting
>

- 13 regressors（可以被指定或者筛选，exclude_estimators=None）
>adaboost, ard_regression, decision_tree, extra_trees, gaussian_process, gradient_boosting, k_nearest_neighbors, liblinear_svr, libsvm_svr, random_forest, ridge_regression, sgd, xgradient_boosting
>

- 18 feature preprocessing methods（这些过程可以被手动关闭全部或者部分，include_preprocessors=[“no_preprocessing”, ]）

>densifier, extra_trees_preproc_for_classification, extra_trees_preproc_for_regression, fast_ica,feature_agglomeration, kernel_pca, kitchen_sinks, liblinear_svc_preprocessor, no_preprocessing, nystroem_sampler, pca, polynomial, random_trees_embedding, select_percentile, select_percentile_classification, select_percentile_regression, select_rates, truncatedSVD

- 5 data preprocessing methods（这些过程不能被手动关闭）

>balancing, imputation, one_hot_encoding, rescaling, variance_threshold（看到这里已经有点惊喜了！点进去有不少内容）

- more than 110 hyperparameters

其中参数 `include_estimators`,要搜索的方法,`exclude_estimators`:为不搜索的方法.与参数`include_estimators`不兼容, 而`include_preprocessors`,可以参考手册中的内容

Auto-Sklearn包含两个主模块：
- 分类任务(autosklearn.classification.AutoSklearnClassifier(...))
- 回归任务(autosklearn.regression.AutoSklearnRegressor(...))


### Auto-sklearn中的特征预处理
- densifier 将稀疏表示的矩阵转换成普通矩阵 Array形式存储
- truncatedSVD, 常用降维方法，将特征降到固定维数，输出维数 作为 超参数
- select_rates 只用于分类，允许使用可配置方法来进行单变量特征选择。它允许超参数搜索评估器来选择最好的单变量特征
  - chi2: 卡方统计量检验类别变量之间的确定关联性，如果输入是稀疏矩阵，只使用chi2
  - f_classif 基于F-检验的方法估计两个随机变量间的线性相关度
  - score_func, mode, alpha都作为超参数
- select_percentile 移除除了指定的最高得分百分比之外的所有特征 以分类为例：除了chi2, f_classif 还有 mutual_info的方法 percentile 作为超参数
- polynomial 多项式特征构造，如果有a，b两个特征，那么它的2次多项式为（1,a,b,a^2,ab, b^2）
- 超参：degree(2阶 or 3阶) interaction(是否进行自身相乘，如 a^2, b^2) , include_bias
- PCA 降维方法， 超参: keep_variance(PCA算法中所要保留的主成分个数n), whiten(白化，使得每个特征具有相同的方差)
- FastICA 用于独立特征分离的算法。超参：n_conponents, whiten, altorithm
- KernelPCA 使用 kPCA 将维度降至低维维，然后应用 Logistic 回归进行分类。然后使用 Grid SearchCV 为 kPCA 找到最佳的核和 gamma 值，以便在最后获得最佳的分类准确性
  - 超参：kernel(核的类型，如RBF, cosine, sigmoid等)， gamma(核系数， RBF, poly，sigmoid专有), degree(2-5, 默认是3), coef0(独立项，poly, sigmoid专有), n_components
- random_trees_embedding, 利用一个随机森林，进行无监督的特征转换，把低维的非线性特征，转化为高维的稀疏特征
    - 超参：比较多，n_estimators, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_leaf_nodes, bootstrap等
- expt_trees
  - max_samples和max_features控制子集的大小，bootstrap和bootstrap_features控制数据样本和属性是否替换。Oob_score=True可使得估计时采用已有的数据划分样本
- feature_agglomeration 把那些效果或行为相似的特征进行聚类，达到降维的目的
- kitchen_sinks 为径向基函数核构造一个近似映射, 与nystroem有关
- nystroem_sample 它是通过采样 kernel 已经评估好的数据。默认情况下使用 rbf kernel，但它可以使用任何内核函数和预计算内核矩阵. 使用的样本数量 - 计算的特征维数 - 由参数 n_components 给出

### Auto-sklearn中的数据预处理

- variance_threshold 过滤方差较低的特征
- Rescaling, 特征的缩放变换
    - 包括 abs, minmax, normalize，standarize等非常常见的处理
  - RobustScaler：适用于有离群点数据，它有对数据中心化和数据的缩放鲁棒性更强的参数，根据分位数范围（默认为IQR： IQR是第1四分位数和第3个四分位数之间的范围。）删除中位数并缩放数据
  - QuantileTransformer：提供了一个基于分位数函数的无参数转换，将数据映射到了零到一的均匀分布上
- OneHotEncoding
- Imputation 缺失值插补，可以用提供的常数值，也可以使用缺失值所在的行/列中的统计数据进行补充
Balancing

### meta-learning 是什么操作？

#### What is MI-SMBO?
- Meta-learning Initialized Sequential Model-Based Bayesian Optimization
- What is meta-learning?
- Mimics human domain experts: use configurations which are known to work well on similar datasets

仿照人能积累经验的做法，使机器有[配置空间]去记录它们的经验值，有点像迁移学习适用的程度，根据数据的相似度
meta-learning: warmstart the Bayesian optimization procedure
也就是学习算法工程师的建模习惯，比如看到什么类型的数据就会明白套用什么模型比较适合，去生产对于数据的 metafeatures：

- 左边：黑色的部分是标准贝叶斯优化流程，红色的是添加meta-learning的贝叶斯优化
- 右边：有 Metafeatures for the Iris dataset，描述数据长什么样的features，下面的公式是计算数据集与数据集的相似度的，只要发现相似的数据集，就可以根据经验来推荐好用的分类器。再来张大图感受下metafeatures到底长啥样：

### Auto-sklearn 如何实现自动超参数调优？

#### 概念解释

- SMBO: Sequential Model-based Bayesian/Global Optimization，调超参的大多数方法基于SMBO
- SMAC: Sequential Model-based Algorithm Configuration，机器学习记录经验值的配置空间
- TPE: Tree-structured Parzen Estimator

#### 超参数调参方法

- Grid Search 网格搜索/穷举搜索
    - 在高维空间不实用。
- Random Search 随机搜索
    - 很多超参是通过并行选择的，它们之间是相互独立的。一些超参会产生良好的性能，另一些不会。
- Heuristic Tuning 手动调参经验法，耗时长。（不知道经验法的英文是否可以这样表示）
- Automatic Hyperparameter Tuning
- Bayesian Optimization
    - 能利用先验知识高效地调节超参数
    - 通过减少计算任务而加速寻找最优参数的进程
    - 不依赖人为猜测所需的样本量为多少，优化技术基于随机性，概率分布
    - 在目标函数未知且计算复杂度高的情况下极其强大
    - 通常适用于连续值的超参，例如 learning rate, regularization coefficient
- SMAC
- TPE


在 auto-sklearn 里，一直出现的 bayesian optimizer 就是答案。是利用贝叶斯优化进行自动调参的。

### Auto-sklearn 如何实现自动模型集成？

官方回答：automated ensemble construction: use all classifiers that were found by Bayesian optimization
目前在库中有16个分类器，根据贝叶斯优化找出最佳分类器组合，比如是（0.4 random forest + 0.2 sgd + 0.4 xgboost)
可以根据fit完的分类器打印结果看最终的模型是由什么分类器组成，以及它们的参数数值：

```python
import autoskleran.classification
automl = autosklearn.classification.AutoSklearnClassifier()
automl.fit(X_train, y_train)

automl.show_models()
打印automl.show_models()就能打印出所谓的自动集成模型有哪些，权重分布，以及超参数数值。
```

### 如何使用 Auto-sklearn？


#### 安装

Auto-sklearn需要基于python3.5以上版本，且依赖swig，因此需要先安装该库，具体方法如下：
```python
$ sudo apt-get install build-essential swig
$ pip install auto-sklearn
```
由于关于auto-sklearn的文档和例程不多，推荐下载auto-sklearn的源码，并阅读其中的example和doc，以便更多地了解auto-sklearn的功能和用法。

```python
$ git clone https://github.com/automl/auto-sklearn.git
```

#### 举个栗子

```python
import sklearn.datasets
import sklearn.metrics
import autosklearn.classification

############################################################################
# Data Loading
# ============
X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = \
    sklearn.model_selection.train_test_split(X, y, random_state=1)
############################################################################
# Build and fit a regressor
# =========================
automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=120,
    per_run_time_limit=30,
    tmp_folder='/tmp/autosklearn_classification_example_tmp',
    output_folder='/tmp/autosklearn_classification_example_out',
)
automl.fit(X_train, y_train, dataset_name='breast_cancer')
############################################################################
# Print the final ensemble constructed by auto-sklearn
# ====================================================
print(automl.show_models())
###########################################################################
# Get the Score of the final ensemble
# ===================================
predictions = automl.predict(X_test)
print("Accuracy score:", sklearn.metrics.accuracy_score(y_test, predictions))
```

#### 关键参数

 Auto-sklearn支持的参数较多，以分类器为例，参数及其默认值如下图所示：
```python
AutoSklearnClassifier(dask_client=None,
                      delete_output_folder_after_terminate=True,
                      delete_tmp_folder_after_terminate=True,
                      disable_evaluator_output=False,
                      ensemble_memory_limit=1024, ensemble_nbest=50,
                      ensemble_size=50, exclude_estimators=None,
                      exclude_preprocessors=None, get_smac_object_callback=None,
                      include_estimators=None, include_preprocessors=None,
                      initi...
                      logging_config=None, max_models_on_disc=50,
                      metadata_directory=None, metric=None,
                      ml_memory_limit=3072, n_jobs=None,
                      output_folder='/tmp/autosklearn_classification_example_out',
                      per_run_time_limit=30, resampling_strategy='holdout',
                      resampling_strategy_arguments=None, seed=1,
                      smac_scenario_args=None, time_left_for_this_task=120,
                      tmp_folder='/tmp/autosklearn_classification_example_tmp')
```

下面介绍其常用参数，分为四个部分：
##### (1) 控制训练时间和内存使用量
 参数默认训练总时长为一小时（3600），一般使用以下参数按需重置，单位是秒。
- time_left_for_this_task：设置所有模型训练时间总和
- per_run_time_limit：设置单个模型训练最长时间
- ml_memory_limit：设置最大内存用量
##### (2) 模型存储参数默认为训练完成后删除训练的暂存目录和输出目录，使用以下参数，可指定其暂存目录及是否删除。
- tmp_folder：暂存目录
- output_folder：输出目录
- delete_tmp_folder_after_terminate：训练完成后是否删除暂存目录
- delete_output_folder_after_terminate：训练完成后是否删除输出目录
- shared_mode：是否共享模型
##### (3) 数据切分使用resampling_strategy参数可设置训练集与测试集的切分方法，以防止过拟合，用以下方法设置五折交叉验证：
- resampling_strategy='cv'
- resampling_strategy_arguments={'folds': 5}
用以下方法设置将数据切分为训练集和测集，其中训练集数据占2/3。
```python
resampling_strategy='holdout',
resampling_strategy_arguments={'train_size': 0.67}
```
##### (4) 模型选择参数支持指定备选的机器学习模型，或者从所有模型中去掉一些机器学习模型，这两个参数只需要设置其中之一。
- include_estimators：指定可选模型
- exclude_estimators：从所有模型中去掉指定模型
  auto-sklearn除了支持sklearn中的模型以外，还支持xgboost模型。具体模型及其在auto-sklearn中对应的名称可通过查看源码中具体实现方法获取，通过以下目录内容查看支持的分类模型：autosklearn/pipeline/components/classification/，可看到其中包含：adaboost、extra_trees、random_forest、libsvm_svc、xgradient_boosting等方法。


### Auto-Sklearn 优点

通常情况下，我们只能依据个人的经验，基于机器性能、特征多少、数据量大小、算法以及迭代次数来估计模型训练时间，而Auto-Sklearn支持设置单次训练时间和总体训练时间，使得工具既能限制训练时间，又能充分利用时间和算力。

Auto-Sklearn支持切分训练/测试集的方式，也支持使用交叉验证。从而减少了训练模型的代码量和程序的复杂程度。另外，Auto-Sklearn支持加入扩展模型以及扩展预测处理方法，具体用法可参见其源码example中的示例。

### Auto-sklearn 缺点

- 不支持深度学习，但是貌似会有AutoNet出来，像谷歌的cloud AutoML那样
- 计算时长往往一个小时以上
- 输出携带的信息较少，如果想进一步训练只能重写代码。
- 在数据清洗这块还需要人为参与，目前对非数值型数据不友好


### 一些教程：
- https://machinelearningmastery.com/auto-sklearn-for-automated-machine-learning-in-python/
- https://machinelearningmastery.com/what-is-bayesian-optimization/
- [Efficient and Robust Automated Machine Learning, 2015](https://papers.nips.cc/paper/5872-efficient-and-robust-automated-machine-learning)
- [Auto-Sklearn Homepage](https://automl.github.io/auto-sklearn/master/)
- [Auto-Sklearn GitHub Project](https://github.com/automl/auto-sklearn)
- [Auto-Sklearn Manual](https://automl.github.io/auto-sklearn/master/manual.html)
