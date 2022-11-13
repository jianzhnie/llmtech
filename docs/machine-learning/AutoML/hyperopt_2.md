## Python中的自动化机器学习





### Bayesian Optimization Methods

贝叶斯优化通过构建代理函数（概率模型）根据目标函数过去的评估结果来找到使目标函数最小化的值。 优化代理函数比直接优化目标优化方案更加容易，因此要通过对代理函数的评估准则来选择要评估的下一个输入值（通常是预期的改进）。 贝叶斯方法不同于随机或网格搜索，因为它们使用过去的评估结果来选择下一个要评估的值。 贝叶斯方法的概念是：根据过去做得好的策略选择下一个输入值，以限制对目标函数进行昂贵的评估。

在超参数优化的情况下，目标函数即机器学习模型的验证误差， 目的是找到在验证集上最低误差的超参数，并希望这些结果能推广到测试集。 评估目标函数非常昂贵，因为它需要使用一组特定的超参数来训练机器学习模型。 理想情况下，我们希望有一种方法可以探索搜索空间，同时还可以限制对较差的超参数选择的评估。 贝叶斯超参数调整使用一个持续更新的概率模型，根据过去的结果进行推理，“集中”在那些有希望的超参数上。

#### **Python Options**

Python中有多个贝叶斯优化库，它们在目标函数的代理函数的算法上有所不同。 在本文中，我们使用Hyperopt，Hyperopt 使用 Tree Parzen Estimator（TPE）优化算法。其他Python库包括Spearmint（高斯过程替代）和SMAC（随机森林回归）。 这个领域有非常多的很好的工具库，因此，如果您对一个算法库不满意，可以查看其他替代方法。在不同的工具箱库之间进行转换仅在语法上有微小差异。 有关 Hyperopt 的基本介绍，请参阅本文。

### Four Parts of Optimization Problem

贝叶斯优化问题分为四个部分：

- 目标函数：我们想要最小化的东西，在机器学习中，通常是关于超参数的机器学习模型的验证误差
- 域空间：要搜索的超参数值的取值空间
- 优化算法：构建代理函数并选择下一个超参数值进行评估的方法
- 结果历史：存储的目标函数的评估结果包括超参数和验证损失

使用这四个部分，我们可以优化（查找最小值）任何带返回值的实数函数。 这种强大的抽象能力，使我们能够处理许多除了超参调优之外其他问题。

#### **DataSet**

我们将使用Caravan Insurance数据集来举例分析，其目的是预测客户是否将购买保险单。 这是一个有监督的分类问题，包含5800个训练样本和4000个测试样本。 我们将用来评估模型性能的度量标准是ROC曲线下区域面积（ROC AUC），因为这是一个不平衡的分类问题。 （ROC AUC越高越好，得分为1表示模型完美）。 数据集如下所示：

![Image for post](https://miro.medium.com/max/2324/1*dH-n2BtATMKpkMuMcnpyDA.png)

​													Dataset (CARAVAN) is the label

由于Hyperopt需要优化的是最小化目标函数值，因此我们将从目标函数的返回调整为1-ROC AUC，从而提高ROC AUC。

#### **Gradient Boosting Model**

本文不需要你详细了解梯度增强机（GBM）的算法原理，这里是我们需要了解的基本知识：GBM是一种基于集成的增强方法，它将弱学习器（几乎总是决策树）进行集成得到一个非常强的模型。 GBM 中有许多超参数可以控制整个集成树和单个决策树。 选择树的数量的最有效方法之一（也被称为 estimators）是使用 [early stoppin](https://en.wikipedia.org/wiki/Early_stopping)。 LightGBM提供了Python中GBM 简单快速的实现。

有关GBM的更多详细信息，请参见 [文章](https://medium.com/mlreview/gradient-boosting-from-scratch-1e317ae4587d) 和 [技术论文](https://brage.bibsys.no/xmlui/bitstream/handle/11250/2433761/16128_FULLTEXT.pdf).

在已经介绍完的必要的背景知识的情况下，让我们完成贝叶斯优化问题的四个部分，以进行超参数调整。

#### Objective Function

目标函数是我们要尽量最小化的函数。 它接受一组值（在本例中为GBM的超参数），并输出最小化的实际值（交叉验证损失）。 `Hyperopt` 将目标函数视为黑匣子，因为它仅考虑输入的内容和输出的内容。 该算法无需知道目标函数的内部结构即可找到使损失最小的输入值！ 以伪代码形式， 我们的目标函数应该是：

```python
def objective(hyperparameters):
    """Returns validation score from hyperparameters"""

    model = Classifier(hyperparameters)
    validation_loss = cross_validation(model, training_data)
    return validation_loss
```

注意不要在测试集上使用损失，因为在评估最终模型时，我们只能使用测试集一次。 相反，我们在验证集上评估超参数。 此外，我们不是将训练数据分成单独的验证集，而是使用K-Fold交叉验证，它使我们减少对测试集的误差估计。

不同模型的目标函数的基本结构是相同的：该函数接收超参数并使用这些超参数返回交叉验证误差。 尽管此示例特定于 GBM，但是该结构可以应用于其他方法。

```python
import lightgbm as lgb
from hyperopt import STATUS_OK

N_FOLDS = 10

# Create the dataset
train_set = lgb.Dataset(train_features, train_labels)

def objective(params, n_folds = N_FOLDS):
    """Objective function for Gradient Boosting Machine Hyperparameter Tuning"""

    # Perform n_fold cross validation with hyperparameters
    # Use early stopping and evalute based on ROC AUC
    cv_results = lgb.cv(params, train_set, nfold = n_folds, num_boost_round = 10000,
                        early_stopping_rounds = 100, metrics = 'auc', seed = 50)

    # Extract the best score
    best_score = max(cv_results['auc-mean'])

    # Loss must be minimized
    loss = 1 - best_score

    # Dictionary with information for evaluation
    return {'loss': loss, 'params': params, 'status': STATUS_OK}
```

主要代码是 cv_results = lgb.cv（...）。为了实现提早停止的交叉验证，我们使用LightGBM 的 cv 函数，该函数接受超参数，训练集，用于交叉验证的折叠数以及其他几个参数。我们将 estimater 的数量（num_boost_round）设置为10000，但实际上并没有达到该数字，因为当100个 estimater 的验证分数没有提高时，我们使用early_stopping_rounds停止训练。提前停止是一种选择 estimater  数量的有效方法，而不是将其设置为另一个需要调整的超参数！

交叉验证完成后，我们将获得最佳分数（ROC AUC）。目标函数实际上比需要的复杂一些，因为我们返回值的字典。对于Hyperopt中的目标函数，我们可以返回单个值，损失或具有最小值的keys “loss”和“ status”的字典。返回超参数可以使我们检查每组超参数得到的损失。

#### Domain Space

Domain Space 表示我们设定的超参数评估的值的范围。搜索的每次迭代，贝叶斯优化算法都会从Domain Space为每个超参数选择一个值。当我们进行随机或网格搜索时，Domain Space 是一个网格。在贝叶斯优化中，想法是相同的，除了该空间中每个超参数都具有一个概率分布而不是离散值。

指定 Domain Space 是贝叶斯优化问题中最棘手的部分。如果我们有机器学习方法的经验，则可以通过将更大的概率放在我们认为最佳值所在的位置来使用它来通知我们对超参数分布的选择。但是，最优模型设置在数据集之间会有所不同，并且存在高维问题（许多超参数），可能很难弄清楚超参数之间的相互作用。如果我们不确定最佳值的范围，可以使用广泛的分布，然后让贝叶斯算法为我们做推理。

首先，我们应该查看GBM中的所有超参数：

```python
import lgb# Default gradient boosting machine classifier
model = lgb.LGBMClassifier()
modelLGBMClassifier(boosting_type='gbdt', n_estimators=100,
               class_weight=None, colsample_bytree=1.0,
               learning_rate=0.1, max_depth=-1,
               min_child_samples=20,
               min_child_weight=0.001, min_split_gain=0.0,
               n_jobs=-1, num_leaves=31, objective=None,
               random_state=None, reg_alpha=0.0, reg_lambda=0.0,
               silent=True, subsample=1.0,
               subsample_for_bin=200000, subsample_freq=1)
```

不确定是否有人知道所有这些参数相互作用， 其中一些参数我们不需要调整（例如“ objective”和“ random_state”），我们将使用 `early stopping` 查找最佳的“ n_estimators”。 但是，我们还有10个超参数需要优化！ 首次调整模型时，通常会围绕默认值创建一个宽的域空间，然后在后续搜索中对其进行优化。

例如，让我们在`Hyperopt`中定义一个简单的域，即GBM中每棵树的叶数的离散均匀分布：

```python
from hyperopt import hp
# Discrete uniform distribution
num_leaves = {'num_leaves': hp.quniform('num_leaves', 30, 150, 1)}
```

这是离散的均匀分布，因为叶的数量必须是整数（离散），并且域中的每个值均具有相同的可能性（均匀）。

分布的另一种选择是对数均匀，它以对数刻度均匀地分布值。 我们将使用 `log uniform` (从0.005到0.2）作为学习率，因为它在几个数量级上变化：

```python
# Learning rate log uniform distribution
learning_rate = {'learning_rate': hp.loguniform('learning_rate',
                                                 np.log(0.005),
                                                 np.log(0.2)}
```

由于这是对数均匀分布，因此将在exp（low）和exp（high）之间绘制值。 左下方的图显示了离散的均匀分布，右图是对数均匀分布。 这些是密度估计曲线，因此y轴是密度而不是计数！

![Image for post](https://miro.medium.com/max/1036/1*Gm1yXk6qM-3NbYLKew9Vgw.png)

![Image for post](https://miro.medium.com/max/1024/1*kzLTXwKXkywDFUcR3bJeuQ.png)

num_leaves (上) 和learning_rate (下) 的 Domin-Space

Now, let’s define the entire domain:

```python
# Define the search space
space = {
    'class_weight': hp.choice('class_weight', [None, 'balanced']),
    'boosting_type':
    hp.choice('boosting_type',
    [{'boosting_type': 'gbdt', 'subsample': hp.uniform('gdbt_subsample', 0.5, 1)},
     {'boosting_type': 'dart', 'subsample': hp.uniform('dart_subsample', 0.5, 1)},
     {'boosting_type': 'goss'}]),
    'num_leaves': hp.quniform('num_leaves', 30, 150, 1),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
    'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
    'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),
    'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
    'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0)
}
```

在这里，我们使用许多不同域分布类型：

- `choice` : categorical variables
- `quniform` : discrete uniform (integers spaced evenly)
- `uniform`: continuous uniform (floats spaced evenly)
- `loguniform`: continuous log uniform (floats spaced evenly on a log scale)

官方文档中还列出了其他分布 [the documentation](https://github.com/hyperopt/hyperopt/wiki/FMin).

定义 boosting 类型时，有一点需要注意：

```python
# boosting type domain
boosting_type = {'boosting_type': hp.choice('boosting_type',
            [{'boosting_type': 'gbdt', 'subsample': hp.uniform('subsample', 0.5, 1)},
              {'boosting_type': 'dart', 'subsample': hp.uniform('subsample', 0.5, 1)},
                {'boosting_type': 'goss', 'subsample': 1.0}])}
```

在这里，我们使用一个条件 domin，这意味着一个超参数的值取决于另一个参数的值。 对于增强型“ goss”，gbm无法使用子采样（仅选择训练观测值的子采样部分在每次迭代中使用）。 因此，如果增强类型为“ goss”，则子采样率设置为1.0（无子采样），否则设置为0.5-1.0。 这是使用嵌套域实现的。

当我们使用完全独立的参数的不同机器学习模型时，条件嵌套很有用。 通过条件，我们可以根据选择的值使用不同的超参数集。

定义好域之后，我们可以从中进行采样。 采样时，由于子采样最初是嵌套的，可以使用Python字典的get方法（默认值为1.0）逐层级展开。

```python
# Sample from the full space
example = sample(space)

# Dictionary get method with default
subsample = example['boosting_type'].get('subsample', 1.0)

# Assign top-level keys
example['boosting_type'] = example['boosting_type']['boosting_type']
example['subsample'] = subsample

example
```

```python
{'boosting_type': 'gbdt',
 'class_weight': 'balanced',
 'colsample_bytree': 0.8111305579351727,
 'learning_rate': 0.16186471096789776,
 'min_child_samples': 470.0,
 'num_leaves': 88.0,
 'reg_alpha': 0.6338327001528129,
 'reg_lambda': 0.8554826167886239,
 'subsample_for_bin': 280000.0,
 'subsample': 0.6318665053932255}
```
（嵌套键的这种重新分配是必需的，因为梯度提升机无法处理嵌套的超参数字典）。

#### Optimization Algorithm

尽管这是贝叶斯优化在概念上最困难的部分，但是在Hyperopt中创建优化算法仅需一行。 要使用Tree Parzen Estimator，代码为：

```python
from hyperopt import tpe
# Algorithm
tpe_algorithm = tpe.suggest
```
Hyperopt 当前仅具有TPE选项以及随机搜索功能。 在优化过程中，TPE算法根据过去的结果构造概率模型，并通过最大化预期的改进来决定要在目标函数中进行评估的下一组超参数。

#### Result History

严格跟踪结果不是严格必要的，因为Hyperopt会在算法内部进行此操作。 但是，如果我们想了解幕后情况，可以使用Trials对象，该对象将存储基本训练信息以及从目标函数返回的字典（包括loss和params）。 使试验对象为一行：

```python
from hyperopt import Trials
# Trials object to track progress
bayes_trials = Trials()
```
帮助我们监视长时间训练的进度的另一种方法是，每次搜索迭代都向csv文件中写入一行。 万一发生灾难性事件并且我们失去了试验对象（从经验上讲），这还将所有结果保存到磁盘中。 我们可以使用csv库执行此操作。 在训练之前，我们打开一个新的csv文件并编写标题：

```python
import csv

# File to save first results
out_file = 'gbm_trials.csv'
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)

# Write the headers to the file
writer.writerow(['loss', 'params', 'iteration', 'estimators', 'train_time'])
of_connection.close()
```
写入csv意味着我们可以在训练时通过打开文件来检查进度（尽管不是在Excel中，因为这会在Python中引起错误。请使用bash中的 tail out_file.csv来查看文件的最后一行）。

#### Optimization

一旦将这四个部分准备就绪，就可以使用fmin运行优化：

```python
from hyperopt import fmin

MAX_EVALS = 500

# Optimize
best = fmin(fn = objective, space = space, algo = tpe.suggest,
            max_evals = MAX_EVALS, trials = bayes_trials)
```

每次迭代时，算法都会从代理函数中选择新的超参数值，该代理函数是基于先前的结果构建的，并在目标函数中评估这些值。

#### Results

从fmin返回的最佳对象包含了目标函数上产生最低损失的超参数：

```python
{'boosting_type': 'gbdt',
   'class_weight': 'balanced',
   'colsample_bytree': 0.7125187075392453,
   'learning_rate': 0.022592570862044956,
   'min_child_samples': 250,
   'num_leaves': 49,
   'reg_alpha': 0.2035211643104735,
   'reg_lambda': 0.6455131715928091,
   'subsample': 0.983566228071919,
   'subsample_for_bin': 200000}
```

一旦有了这些超参数，就可以使用它们在完整的训练数据上训练模型，然后对测试数据进行评估（请记住，在评估最终模型时，我们只能使用一次测试集）。 对于 estimater 的数量，我们可以使用在交叉验证和早期停止中返回最低损失的估算器的数量。 最终结果如下：

```python
The best model scores 0.72506 AUC ROC on the test set.
The best cross validation score was 0.77101 AUC ROC.
This was achieved after 413 search iterations.
```

作为参考，随机搜索的500次迭代返回了一个模型，该模型在测试集上的得分为0.7232 ROC AUC，在交叉验证中得分为0.76850。 没有优化的默认模型在测试集上的得分为0.7143 ROC AUC。

在查看结果时，需要牢记一些重要注意事项：

- 最佳超参数是在交叉验证中表现最佳的参数，而不一定是在测试数据上表现最佳的参数。 当我们使用交叉验证时，我们希望这些结果能推广到测试数据。
- 即使使用10倍交叉验证，超参数调整也会过度拟合训练数据。 交叉验证的最佳分数明显高于测试数据。
- 如果幸运的话，随机搜索可能会返回更好的超参数（重新运行 notebook 可以更改结果）。 贝叶斯优化不能保证找到更好的超参数，并且会陷入目标函数的局部最小值。

贝叶斯优化是有效的，但不能解决我们所有的参数调优问题。随着搜索的进行，该算法从探索（尝试使用新的超参数值）来最小化参数函数损失转为利用已有的观测调整参数优化的方向。如果算法找到目标函数的局部最小值，则它可能专注于局部最小值附近的超参数值，而不是尝试在域空间中很远的其他值。随机搜索不会遇到此问题，因为它不关注任何值！

另一个重要的一点是，超参数优化的好处将随数据集而有所不同。 这是一个相对较小的数据集（约6000个训练观测值），并且调整超参数的投资回报很小（获取更多数据将更好地节约时间！）。 考虑到所有这些注意事项，在这种情况下，通过贝叶斯优化，我们可以获得：

- 测试集上的更好性能
- 使用更少的迭代来调整超参数

贝叶斯方法可以（尽管不会总是）产生比随机搜索更好的调整结果。 在接下来的几节中，我们将研究贝叶斯超参数搜索的发展，并与随机搜索进行比较，以了解贝叶斯优化的工作原理。

#### Visualizing Search Results

对结果进行可视化是了解超参数搜索过程的一种最为直观的方法。 此外，将贝叶斯优化与随机搜索进行比较很有帮助，这样我们就可以了解方法的不同。 要查看如何制作图并实现随机搜索，请查看 [the notebook](https://github.com/WillKoehrsen/hyperparameter-optimization/blob/master/Bayesian Hyperparameter Optimization of Gradient Boosting Machine.ipynb)，在这里我们只展示结果。 （请注意，确切的结果会在每次迭代中发生变化，因此，如果你运行 notebook 获得不同的图像也不要感到惊讶。所有这些图都是500次迭代生成的。）

首先，我们可以绘制随机搜索和贝叶斯优化中学习率的核密度估计图。 作为参考，我们还可以显示采样分布。 垂直虚线表示学习率的最佳值（根据交叉验证）。

![Image for post](https://miro.medium.com/max/1778/1*6pWbEJoqNwonzolxD4KpCw.png)

我们将学习率定义为介于0.005和0.2之间的对数正态分布，贝叶斯优化结果看起来类似于抽样分布。 这告诉我们，尽管最佳值比我们放置最大概率的位置稍高，但我们定义的分布看起来适合任务。 这可以用于通知 `Domin`进一步搜索。

另一个超参数是Boosting类型，在随机搜索和贝叶斯优化过程中评估了每种类型的条形图，如下所示。 由于随机搜索不会关注过去的结果，因此每种Boosting类型的使用次数大致相同。

![Image for post](https://miro.medium.com/max/1268/1*V4NBLoeePElKQeGqur8pFg.png)

根据贝叶斯算法，gdbt Boosting类型比 dart 或 goss 更有希望改善优化的结果。 同样，这可以帮助提供进一步的搜索信息，无论是贝叶斯方法还是网格搜索。 如果我们想进行更明智的网格搜索，则可以使用这些结果来定义一个较小的网格，该网格集中在最有希望的超参数值周围。

有了它们，让我们从参考分布查看随机搜索和贝叶斯优化所有数值型的超参数。 垂直线再次指示每次搜索的超参数的最佳值：


![Image for post](https://miro.medium.com/max/1335/1*v7N67eMfFxeabPGBW_-7HQ.png)


![Image for post](https://miro.medium.com/max/1368/1*dw6VrNPpfVQuzSbHpqRSRw.png)


![Image for post](https://miro.medium.com/max/1259/1*sy-axn8KuB1GC4mss0EmuA.png)


![Image for post](https://miro.medium.com/max/1319/1*p_SKapLKyyZXqRm437TgKw.png)


![Image for post](https://miro.medium.com/max/1302/1*pSlIV25n8bsxqPiNwPWzBw.png)


![Image for post](https://miro.medium.com/max/1302/1*1EPxm6czkBcyZNqVd8UCXg.png)

在大多数情况下（除了subsample_for_bin之外），贝叶斯优化搜索趋向于集中（放置更多概率）在超参数值附近，该值在交叉验证中产生最低的Loss。 这显示了使用贝叶斯方法进行超参数调整的基本思想：花更多时间评估有希望的超参数值。

这里还有一些有趣的结果，可能会在将来定义搜索的域空间时为我们提供帮助。 仅作为一个示例，看起来reg_alpha和reg_lambda应该相互补充：如果一个较高（接近1.0），则另一个应该较低。 不能保证这可以解决所有问题，但是通过研究结果，我们可以获得可以应用于未来机器学习问题的见解！

#### Evolution of Search

随着优化的进行，我们期望贝叶斯方法将重点放在超参数的更有希望的值上：那些在交叉验证中产生最低误差的值。 我们可以绘制超参数值与迭代次数的关系图，以查看是否存在明显的趋势。

![Image for post](https://miro.medium.com/max/2555/1*LYrDpsvyfYtM143qIwaXLg.png)

黑色星号表示最佳值。 colsample_bytree和learning_rate会随着时间的推移而减少，这可能会在以后的搜索中为我们提供指导。

![Image for post](https://miro.medium.com/max/1895/1*xn-A948AcHROFSP173D1RA.png)

最后，如果贝叶斯优化器正在运行，我们希望平均的验证集得分会随着时间的推移而增加（相反，损失会减少）：

![Image for post](https://miro.medium.com/max/457/1*VNhV0ATCudF890dLo0YIvQ.png)



贝叶斯超参数优化的验证分数随时间增加，表明该方法正在尝试“更好”的超参数值。随机搜索未显示出迭代的改进。

#### Continue Searching

如果我们对模型的性能不满意，则可以继续上次的Hyperopt进行搜索。我们只需要传递相同的试验对象，算法就会继续搜索。

随着算法的前进，它会进行更多的 exploitation — 选择过去做得很好的值—进行较少的探索—选择新值。因此，选择从一个完全不同的搜索开始或许能得到更好的结果，而不是继续从搜索停止的地方开始。如果第一次搜索的最佳超参数确实是“最优的”，则我们期望后续搜索将重点放在相同的值上。考虑到问题的高度维度以及超参数之间的复杂交互，不太可能再次搜索会导致类似的超参数集。

经过另外500次训练后，最终模型在测试集上的得分为0.72736 ROC AUC。 （我们不应该在测试集上评估第一个模型，而应该仅依靠验证分数。理想情况下，测试集在部署到新数据上时仅应使用一次以衡量算法性能）。同样，由于数据集较小，此问题可能对进一步的超参数优化的收益递减，并且最终会出现验证集 error 的稳定状态。（由于存在隐藏变量和噪声数据，对数据集上的任何模型的性能都受固有限制，称为贝叶斯误差）。

### Conclusions

机器学习模型的自动超参优化可以使用贝叶斯优化来完成。与随机搜索相反，贝叶斯优化以一种明智的方法选择下一个超参数，花费更多时间评估那些有希望的值。与随机或网格搜索相比，最终结果可能是目标函数的评估数量更少，并且测试集的泛化性能更好。


在本文中，我们逐步介绍了使用`Hyperopt`在Python中进行贝叶斯超参数优化的过程。尽管我们需要谨慎地训练模型防止过拟合，但我们能够将梯度提升机的测试集性能提高到基线和随机搜索之外。此外，通过检查结果图，我们可以发现随机搜索与贝叶斯优化有何不同，该图表明贝叶斯方法对超参数值赋予更大的概率，从而降低了交叉验证损失。


使用优化问题的四个部分，我们可以使用Hyperopt解决各种各样的问题。贝叶斯优化的基本部分也适用于Python中实现不同算法的许多库。从手动搜索切换到随机搜索或网格搜索只是一小步，但是要将您的机器学习提升到一个新的水平，则需要某种自动化形式的超参数调整。贝叶斯优化是一种既易于在Python中使用又可以返回比随机搜索更好的结果的方法。希望您现在有信心开始针对自己的机器学习问题使用这种强大的方法！
