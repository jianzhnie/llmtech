## Parameter Tuning with Hyperopt



这篇文章将涵盖快速实现机器学习模型参数调整的快速，原则化方法所需的一些内容。有两种常见的参数调整方法：网格搜索和随机搜索，都有其优缺点。网格搜索速度很慢，但是可以有效地搜索整个搜索空间，随机搜索速度很快，但是可能会错过搜索空间中的重要点。幸运的是，存在第三个选择：贝叶斯优化。在本文中，我们将重点介绍贝叶斯优化的一种实现，即称为 hyperopt 的Python模块。

[hyperopt](https://github.com/hyperopt/hyperopt) 是一个超参数优化库，针对具有一定条件或约束的搜索空间进行调优，其中包括随机搜索和Tree Parzen Estimators（贝叶斯优化的变体）等算法。它使用MongoDb作为存储超参数组合结果的中心结构，可实现多台电脑的并行计算。

### Objective functions

假设您在某个范围内定义了一个函数，并且希望将其最小化。 即，您要查找导致最低输出值的输入值。 以下平凡的示例找到x的值，该值使线性函数 $y(x)= x$ 最小。

```python
from hyperopt import fmin, tpe, hp
best = fmin(
    fn=lambda x: x,
    space=hp.uniform('x', 0, 1),
    algo=tpe.suggest,
    max_evals=100)
print (best)
```

函数`fmin`首先采用一个最小化函数，表示为fn，我们在此处使用函数`lambda x：x`指定该函数。此函数可以是任何有效的值返回函数，例如回归中的平均绝对误差。

下一个参数指定搜索空间，在此示例中，它是0到1之间连续的数字范围，由 `hp.uniform（'x',0,1）`指定。 `hp.uniform`是一个内置的`hyperopt`函数.

参数 `algo ` 表示采用的搜索算法，这里是 `tpe`。 `algo`参数也可以设置为`hyperopt.random`，但由于它是众所周知的搜索策略，因此在此不做介绍。

最后，我们指定`fmin`函数将执行的最大评估数`max_evals`。这个`fmin`函数返回一个python值字典。

上面函数的输出示例为 {'x'：0.000269455723739237}。这是函数的图，红点是我们试图找到的点。

![Silvrback blog image](https://silvrback.s3.amazonaws.com/uploads/403df379-e565-4f7f-81a4-911c8f77ae28/ex1_large.png)

再来看一个更复杂的目标函数：`lambda x：（x-1）** 2`。 这次我们试图最小化二次方程` y（x）=（x-1）** 2	`。 因此，我们更改了搜索空间，以包含我们知道的最佳值（x = 1）以及两侧的一些次优范围：`hp.uniform（'x'，-2，2）`。

现在我们有：

```python
best = fmin(
    fn=lambda x: (x-1)**2,
    space=hp.uniform('x', -2, 2),
    algo=tpe.suggest,
    max_evals=100)
print (best)
```

The output should look something like this:  {'x': 0.997369045274755}

Here is the plot.

![Silvrback blog image](https://silvrback.s3.amazonaws.com/uploads/7499c5c5-5bd6-4bdb-8030-f82258f686ef/ex2_large.png)

与其最小化目标函数，或许我们想最大化它。 为此，我们只需要返回函数的负数即可。 例如，我们可以有一个函数y（x）= -（x ** 2）。

我们如何才能解决这个问题？ 我们只是采用目标函数lambda x：-（x ** 2）并返回负数，从而得出lambda x：-1 *-（x ** 2）。

下面是一个具有许多（在无限范围内，有无限多个）局部极小值的函数，我们也可以尝试将其最大化：

![Silvrback blog image](https://silvrback.s3.amazonaws.com/uploads/661ff1fe-740b-4112-86dc-7f583184d143/ex5_large.png)

### Search spaces

`hyperopt` 模块包括一些很方便的功能可以用于指定输入参数的范围。 我们已经看过 `hp.uniform`。 最初，这些是随机搜索空间，但是随着``hyperopt`了解更多（由于它从目标函数获得更多反馈），它会适应并采样初始搜索空间的不同部分，并认为该部分将为其提供最有意义的反馈。

在这篇文章中将使用以下内容：

- `hp.choice`（label，options），其中options是python列表或元组。
- `hp.normal`（label，mu，sigma），其中mu和sigma分别是平均值和标准偏差。
- `hp.uniform`（label，low，high），其中low和high是该范围的上下限。
其他可用的，例如hp.normal，hp.lognormal，hp.quniform，但是我们在这里不再使用它们。

要在搜索空间中看到一些绘图，我们应该导入另一个函数，并定义搜索空间。


```python
space = {
    'x': hp.uniform('x', 0, 1),
    'y': hp.normal('y', 0, 1),
    'name': hp.choice('name', ['alice', 'bob']),
}
```
```print hyperopt.pyll.stochastic.sample(space)```

An example output is:

```{'y': -1.4012610048810574, 'x': 0.7258615424906184, 'name': 'alice'}```

### Storing evaluation trials

如果能看到 `hyperopt` 的黑匣子内到底发生了什么，对于理解其原理将会有很大帮助。 `hyperopt`  的Trials对象可以实现这一点。

```python
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

fspace = {
    'x': hp.uniform('x', -5, 5)
}

def f(params):
    x = params['x']
    val = x**2
    return {'loss': val, 'status': STATUS_OK}

trials = Trials()
best = fmin(fn=f, space=fspace, algo=tpe.suggest, max_evals=50, trials=trials)

print ('best:', best)
print ('trials:')
for trial in trials.trials[:2]:
    print (trial)
```

Trials对象允许我们存储每个时间的信息。 然后，我们可以将它们打印出来，看看在给定的时间步长上对给定参数的函数求值是什么。

这是上面代码的示例输出

```python
best: {'x': 0.014420181637303776}
trials:
{'refresh_time': None, 'book_time': None, 'misc': {'tid': 0, 'idxs': {'x': [0]}, 'cmd': ('domain_attachment', 'FMinIter_Domain'), 'vals': {'x': [1.9646918559786162]}, 'workdir': None}, 'state': 2, 'tid': 0, 'exp_key': None, 'version': 0, 'result': {'status': 'ok', 'loss': 3.8600140889486996}, 'owner': None, 'spec': None}
{'refresh_time': None, 'book_time': None, 'misc': {'tid': 1, 'idxs': {'x': [1]}, 'cmd': ('domain_attachment', 'FMinIter_Domain'), 'vals': {'x': [-3.9393509404526728]}, 'workdir': None}, 'state': 2, 'tid': 1, 'exp_key': None, 'version': 0, 'result': {'status': 'ok', 'loss': 15.518485832045357}, 'owner': None, 'spec': None}
```

试验对象将数据存储为BSON对象，该对象的工作方式类似于JSON对象。 BSON来自pymongo模块。 我们不会在这里讨论细节，但是Hyperopt的高级选项需要使用MongoDB进行分布式计算，因此需要pymongo导入。

返回上面的输出。 “ tid”是时间id，即时间步长，从0到max_evals-1。 每次迭代增加一。 “ x”位于“ vals”键中，该值是每次迭代存储参数的位置。 “loss”位于“result”键中，它为我们提供了该目标函数在该迭代中的值。

让我们以另一种方式来看待。

### Visualization

在这里，我们将介绍两种可视化类型：val vs. time，以及 loss vs. val。 首先，val与时间的关系。 以下是用于绘制上述trials.trials数据的代码和示例输出。

```python
f, ax = plt.subplots(1)
xs = [t['tid'] for t in trials.trials]
ys = [t['misc']['vals']['x'] for t in trials.trials]
ax.set_xlim(xs[0]-10, xs[-1]+10)
ax.scatter(xs, ys, s=20, linewidth=0.01, alpha=0.75)
ax.set_title('$x$ $vs$ $t$ ', fontsize=18)
ax.set_xlabel('$t$', fontsize=16)
ax.set_ylabel('$x$', fontsize=16)
```

假设我们将`max_evals`更改为1000，则输出应如下所示。

![Silvrback blog image](https://silvrback.s3.amazonaws.com/uploads/b78fb652-477a-4360-8c47-ede68508e67f/ex6_large.png)



可以看到，算法最初从整个范围中均等（均匀）地选取值，但是随着时间的流逝，算法越来越了解该参数对目标函数的影响，该算法越来越关注于它认为将要实现的领域。 获得最大收益-接近零的范围。 它仍然探索整个解决方案空间，但不那么频繁。

现在，让我们来看一下 loss 与 val 的关系图。

```python
f, ax = plt.subplots(1)
xs = [t['misc']['vals']['x'] for t in trials.trials]
ys = [t['result']['loss'] for t in trials.trials]
ax.scatter(xs, ys, s=20, linewidth=0.01, alpha=0.75)
ax.set_title('$val$ $vs$ $x$ ', fontsize=18)
ax.set_xlabel('$x$', fontsize=16)
ax.set_ylabel('$val$', fontsize=16)
```

![Silvrback blog image](https://silvrback.s3.amazonaws.com/uploads/60f2dbae-2103-4cad-9da3-0805e2f75b14/ex7_large.png)

这给出了我们期望的结果，因为函数y（x）= x ** 2是确定性的。

最后，让我们尝试一个更复杂的示例，它具有更多的随机性和更多的参数。

### Full example on a classic dataset: Iris

在本节中，我们将通过4个完整示例演示如何使用hyperopt对经典数据集Iris进行参数调节。我们将介绍K最近邻（KNN），支持向量机（SVM），决策树和随机森林。请注意，由于我们试图最大化交叉验证的准确性（在下面的代码中为acc），因此我们必须对hyperopt取反该值，因为hyperopt只会最小化一个函数。最小化函数f与最大化f的负数相同。

对于此任务，我们将使用经典的Iris数据集，并进行一些监督的机器学习。有4个输入要素和三个输出类别。数据被标记为属于0、1或2类，它们映射到不同种类的鸢尾花。输入有4列：萼片长度，萼片宽度，花瓣长度和踏板宽度。输入单位为厘米。我们将使用这4个功能来学习预测三个输出类别之一的模型。由于数据是由sklearn提供的，因此它具有不错的DESCR属性，该属性提供有关数据集的详细信息。请尝试以下操作以获取更多详细信息。

```python
print iris.feature_names # input names
print iris.target_names # output names
print iris.DESCR # everything else
```

我们使用下面的代码通过功能和类的可视化更好地了解数据。

```python
import seaborn as sns
sns.set(style="whitegrid", palette="husl")

iris = sns.load_dataset("iris")
print iris.head()

iris = pd.melt(iris, "species", var_name="measurement")
print iris.head()

f, ax = plt.subplots(1, figsize=(15,10))
sns.stripplot(x="measurement", y="value", hue="species", data=iris, jitter=True, edgecolor="white", ax=ax)
```

![Silvrback blog image](https://silvrback.s3.amazonaws.com/uploads/553b9756-4107-43a8-ad9a-244b986ce250/iris-sns-stripplot_large.png)

#### KNN

现在，我们将hyperopt应用到K近邻（KNN）机器学习模型中，以找到最佳参数。 KNN模型根据训练数据集中k个最近的数据点的多数类对测试集中的数据点进行分类。 下面的代码包含了我们涵盖的所有内容。

```python
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target

def hyperopt_train_test(params):
    clf = KNeighborsClassifier(**params)
    return cross_val_score(clf, X, y).mean()

space4knn = {
    'n_neighbors': hp.choice('n_neighbors', range(1,100))
}

def f(params):
    acc = hyperopt_train_test(params)
    return {'loss': -acc, 'status': STATUS_OK}

trials = Trials()
best = fmin(f, space4knn, algo=tpe.suggest, max_evals=100, trials=trials)
print 'best:'
print best
```

现在让我们看一下输出图。 y轴是交叉验证分数，x轴是k个最近邻居中的k值。 这是代码及其图像：

```python
f, ax = plt.subplots(1)#, figsize=(10,10))
xs = [t['misc']['vals']['n'] for t in trials.trials]
ys = [-t['result']['loss'] for t in trials.trials]
ax.scatter(xs, ys, s=20, linewidth=0.01, alpha=0.5)
ax.set_title('Iris Dataset - KNN', fontsize=18)
ax.set_xlabel('n_neighbors', fontsize=12)
ax.set_ylabel('cross validation accuracy', fontsize=12)
```

![Silvrback blog image](https://silvrback.s3.amazonaws.com/uploads/b25f4c11-5e8e-4de0-bca5-eaab1f0c5492/iris-knn-cv_large.png)

k大于63后，精度会急剧下降。 这是由于数据集中每个类只有50个实例。 因此，让我们通过将“ n_neighbors”的值限制为较小的值来进行深入研究。

```python
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target

def hyperopt_train_test(params):
    clf = KNeighborsClassifier(**params)
    return cross_val_score(clf, X, y).mean()

space4knn = {
    'n_neighbors': hp.choice('n_neighbors', range(1,50))
}

def f(params):
    acc = hyperopt_train_test(params)
    return {'loss': -acc, 'status': STATUS_OK}

trials = Trials()
best = fmin(f, space4knn, algo=tpe.suggest, max_evals=100, trials=trials)
print 'best:'
print best
```

这是我们运行相同的代码进行可视化时得到的结果：

![Silvrback blog image](https://silvrback.s3.amazonaws.com/uploads/04fed0d6-c871-4a85-b3d0-5a4cd121db19/iris-knn-cv-2_large.png)

现在我们可以清楚地看到，在k = 4时，有一个最佳的k值。

上面的模型没有执行任何预处理。 因此，让我们归一化和扩展我们的功能，看看是否有帮助。 使用此代码：

```python
# now with scaling as an option
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target

def hyperopt_train_test(params):
    X_ = X[:]

    if 'normalize' in params:
        if params['normalize'] == 1:
            X_ = normalize(X_)
            del params['normalize']

    if 'scale' in params:
        if params['scale'] == 1:
            X_ = scale(X_)
            del params['scale']

    clf = KNeighborsClassifier(**params)
    return cross_val_score(clf, X_, y).mean()

space4knn = {
    'n_neighbors': hp.choice('n_neighbors', range(1,50)),
    'scale': hp.choice('scale', [0, 1]),
    'normalize': hp.choice('normalize', [0, 1])
}

def f(params):
    acc = hyperopt_train_test(params)
    return {'loss': -acc, 'status': STATUS_OK}

trials = Trials()
best = fmin(f, space4knn, algo=tpe.suggest, max_evals=100, trials=trials)
print 'best:'
```

并绘制如下参数：

```python
parameters = ['n_neighbors', 'scale', 'normalize']
cols = len(parameters)
f, axes = plt.subplots(nrows=1, ncols=cols, figsize=(15,5))
cmap = plt.cm.jet
for i, val in enumerate(parameters):
    xs = np.array([t['misc']['vals'][val] for t in trials.trials]).ravel()
    ys = [-t['result']['loss'] for t in trials.trials]
    xs, ys = zip(\*sorted(zip(xs, ys)))
    ys = np.array(ys)
    axes[i].scatter(xs, ys, s=20, linewidth=0.01, alpha=0.75, c=cmap(float(i)/len(parameters)))
    axes[i].set_title(val)
```

![Silvrback blog image](https://silvrback.s3.amazonaws.com/uploads/eb83666a-aded-40bc-955c-8000742886e8/iris-knn-params-n-scale-norm_large.png)

我们看到缩放和/规范化数据不会提高预测准确性。 k的最佳值仍为4，可以得到98.6％的精度。

因此，这对于参数调整简单模型KNN非常有用。 让我们看看我们可以使用支持向量机（SVM）做什么。

#### Support Vector Machines (SVM)

由于这是分类任务，因此我们将使用sklearn的SVC类。代码如下：

```python
X = iris.data
y = iris.target

def hyperopt_train_test(params):
    X_ = X[:]

    if 'normalize' in params:
        if params['normalize'] == 1:
            X_ = normalize(X_)
            del params['normalize']

    if 'scale' in params:
        if params['scale'] == 1:
            X_ = scale(X_)
            del params['scale']

    clf = SVC(**params)
    return cross_val_score(clf, X_, y).mean()

space4svm = {
    'C': hp.uniform('C', 0, 20),
    'kernel': hp.choice('kernel', ['linear', 'sigmoid', 'poly', 'rbf']),
    'gamma': hp.uniform('gamma', 0, 20),
    'scale': hp.choice('scale', [0, 1]),
    'normalize': hp.choice('normalize', [0, 1])
}

def f(params):
    acc = hyperopt_train_test(params)
    return {'loss': -acc, 'status': STATUS_OK}

trials = Trials()
best = fmin(f, space4svm, algo=tpe.suggest, max_evals=100, trials=trials)
print 'best:'
print best

parameters = ['C', 'kernel', 'gamma', 'scale', 'normalize']
cols = len(parameters)
f, axes = plt.subplots(nrows=1, ncols=cols, figsize=(20,5))
cmap = plt.cm.jet
for i, val in enumerate(parameters):
    xs = np.array([t['misc']['vals'][val] for t in trials.trials]).ravel()
    ys = [-t['result']['loss'] for t in trials.trials]
    xs, ys = zip(\*sorted(zip(xs, ys)))
    axes[i].scatter(xs, ys, s=20, linewidth=0.01, alpha=0.25, c=cmap(float(i)/len(parameters)))
    axes[i].set_title(val)
    axes[i].set_ylim([0.9, 1.0])
```

![Silvrback blog image](https://silvrback.s3.amazonaws.com/uploads/64773502-cdbf-4f3f-83af-9a2b82f3d176/iris-svm-params_large.png)

同样，缩放和规范化也无济于事。 核函数的最佳选择是（线性核），最佳C值为1.4168540399911616，最佳gamma为15.04230279483486。 这组参数的分类精度为99.3％。

#### Decision Trees

我们将仅尝试对决策树的部分参数进行优化，代码如下。

```python
iris = datasets.load_iris()
X_original = iris.data
y_original = iris.target

def hyperopt_train_test(params):
    X_ = X[:]
    if 'normalize' in params:
        if params['normalize'] == 1:
            X_ = normalize(X_)
            del params['normalize']

    if 'scale' in params:
        if params['scale'] == 1:
            X_ = scale(X_)
            del params['scale']
    clf = DecisionTreeClassifier(**params)
    return cross_val_score(clf, X, y).mean()

space4dt = {
    'max_depth': hp.choice('max_depth', range(1,20)),
    'max_features': hp.choice('max_features', range(1,5)),
    'criterion': hp.choice('criterion', ["gini", "entropy"]),
    'scale': hp.choice('scale', [0, 1]),
    'normalize': hp.choice('normalize', [0, 1])
}

def f(params):
acc = hyperopt_train_test(params)
return {'loss': -acc, 'status': STATUS_OK}

trials = Trials()
best = fmin(f, space4dt, algo=tpe.suggest, max_evals=300, trials=trials)
print 'best:'
print best
```

输出结果如下，最优模型的结果为 97.3 % 的正确率。

```
{'max_features': 1, 'normalize': 0, 'scale': 0, 'criterion': 0, 'max_depth': 17}
```

下面是绘图。 我们可以看到，使用不同的 Scale 和 Normalize，性能几乎没有差异。

```
parameters = ['max_depth', 'max_features', 'criterion', 'scale', 'normalize'] # decision tree
cols = len(parameters)
f, axes = plt.subplots(nrows=1, ncols=cols, figsize=(20,5))
cmap = plt.cm.jet
for i, val in enumerate(parameters):
    xs = np.array([t['misc']['vals'][val] for t in trials.trials]).ravel()
    ys = [-t['result']['loss'] for t in trials.trials]
    xs, ys = zip(\*sorted(zip(xs, ys)))
    ys = np.array(ys)
    axes[i].scatter(xs, ys, s=20, linewidth=0.01, alpha=0.5, c=cmap(float(i)/len(parameters)))
    axes[i].set_title(val)
    #axes[i].set_ylim([0.9,1.0])
```

![Silvrback blog image](https://silvrback.s3.amazonaws.com/uploads/86d96f54-977c-4594-82d1-c9b19641b60e/iris-dt-params_large.png)

#### Random Forests

让我们看看 ensemble 的分类器 随机森林，它只是一组决策树的集合。

```python
iris = datasets.load_iris()
X_original = iris.data
y_original = iris.target

def hyperopt_train_test(params):
    X_ = X[:]
    if 'normalize' in params:
        if params['normalize'] == 1:
            X_ = normalize(X_)
            del params['normalize']

    if 'scale' in params:
        if params['scale'] == 1:
            X_ = scale(X_)
            del params['scale']
    clf = RandomForestClassifier(**params)
    return cross_val_score(clf, X, y).mean()

space4rf = {
    'max_depth': hp.choice('max_depth', range(1,20)),
    'max_features': hp.choice('max_features', range(1,5)),
    'n_estimators': hp.choice('n_estimators', range(1,20)),
    'criterion': hp.choice('criterion', ["gini", "entropy"]),
    'scale': hp.choice('scale', [0, 1]),
    'normalize': hp.choice('normalize', [0, 1])
}

best = 0
def f(params):
    global best
    acc = hyperopt_train_test(params)
    if acc > best:
    best = acc
    print 'new best:', best, params
    return {'loss': -acc, 'status': STATUS_OK}

trials = Trials()
best = fmin(f, space4rf, algo=tpe.suggest, max_evals=300, trials=trials)
print 'best:'
print best
```

同样的我们得到 97.3 % 的正确率 ， 和decision tree 的结果一致.

```python
parameters = ['n_estimators', 'max_depth', 'max_features', 'criterion', 'scale', 'normalize']
f, axes = plt.subplots(nrows=2, ncols=3, figsize=(15,10))
cmap = plt.cm.jet
for i, val in enumerate(parameters):
    print i, val
    xs = np.array([t['misc']['vals'][val] for t in trials.trials]).ravel()
    ys = [-t['result']['loss'] for t in trials.trials]
    xs, ys = zip(\*sorted(zip(xs, ys)))
    ys = np.array(ys)
    axes[i/3,i%3].scatter(xs, ys, s=20, linewidth=0.01, alpha=0.5, c=cmap(float(i)/len(parameters)))
    axes[i/3,i%3].set_title(val)
    #axes[i/3,i%3].set_ylim([0.9,1.0])
```

![Silvrback blog image](https://silvrback.s3.amazonaws.com/uploads/99c90935-e0d9-40e2-b501-75812cce2e23/iris-rf-params_large.png)

### All  Together Now

一次自动调整一个模型的参数（例如，SVM或KNN）既有趣又有启发性，但如果一次调整所有模型参数并最终获得最佳模型更为有用。 这使我们能够一次比较所有模型和所有参数，从而为我们提供最佳模型。

```python
digits = datasets.load_digits()
X = digits.data
y = digits.target
print X.shape, y.shape

def hyperopt_train_test(params):
    t = params['type']
    del params['type']
    if t == 'naive_bayes':
        clf = BernoulliNB(**params)
    elif t == 'svm':
        clf = SVC(**params)
    elif t == 'dtree':
        clf = DecisionTreeClassifier(**params)
    elif t == 'knn':
        clf = KNeighborsClassifier(**params)
    else:
        return 0
    return cross_val_score(clf, X, y).mean()

space = hp.choice('classifier_type', [
    {
        'type': 'naive_bayes',
        'alpha': hp.uniform('alpha', 0.0, 2.0)
    },
    {
        'type': 'svm',
        'C': hp.uniform('C', 0, 10.0),
        'kernel': hp.choice('kernel', ['linear', 'rbf']),
        'gamma': hp.uniform('gamma', 0, 20.0)
    },
    {
        'type': 'randomforest',
        'max_depth': hp.choice('max_depth', range(1,20)),
        'max_features': hp.choice('max_features', range(1,5)),
        'n_estimators': hp.choice('n_estimators', range(1,20)),
        'criterion': hp.choice('criterion', ["gini", "entropy"]),
        'scale': hp.choice('scale', [0, 1]),
        'normalize': hp.choice('normalize', [0, 1])
    },
    {
        'type': 'knn',
        'n_neighbors': hp.choice('knn_n_neighbors', range(1,50))
    }
])

count = 0
best = 0
def f(params):
    global best, count
    count += 1
    acc = hyperopt_train_test(params.copy())
    if acc > best:
        print 'new best:', acc, 'using', params['type']
        best = acc
    if count % 50 == 0:
        print 'iters:', count, ', acc:', acc, 'using', params
    return {'loss': -acc, 'status': STATUS_OK}

trials = Trials()
best = fmin(f, space, algo=tpe.suggest, max_evals=1500, trials=trials)
print 'best:'
print best
```

由于我们增加了评估数量，因此该代码需要一段时间才能运行：max_evals = 1500。

### 总结

我们已经介绍了一些简单的示例（例如使确定性线性函数最小化）和复杂的示例（例如调整随机森林参数）。 hyperopt的文档在这里。这篇文章中的技术可用于机器学习以外的许多领域，例如在epsilon-greedy多臂匪徒中调整epsilon参数，或将参数传递给图形生成器以制作具有某些属性的合成网络。
