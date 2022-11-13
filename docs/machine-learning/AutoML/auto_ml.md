

## AutoML框架

相对于其他 AutoML工具来说， Auto_ml 是从0->1的开创性的工作，其附加功能很多，比如feature response，data clean，ml analysis，feature learning ，deep learning，Categorical Ensembling（类别组合）等。但是对于auto_ml的核心——model selection，hyper parameter optimation等，它仅仅做的很基础——使用SGCV进行网格搜索，而在不选择model selection时，model是默认给定的（默认GradientBoostingRegressor for regressor &GradientBoostingClassifier for classifier）。
总之，auto_ml做的很基础很全面，根据阅读源代码熟悉 autoML的Pipeline，适合auto_ml新手进行探索。

### 特性

使整个机器学习过程自动化，使其非常易于用于分析和获取生产中的实时预测。该项目自动执行：

- Analytics (pass in data, and auto_ml will tell you the relationship of each variable to what it is you’re trying to predict).
- Feature Engineering (particularly around dates, and soon, NLP).
- Robust Scaling (turning all values into their scaled versions between the range of 0 and 1, in a way that is robust to outliers, and works with sparse matrices).
- Feature Selection (picking only the features that actually prove useful).
- Data formatting (turning a list of dictionaries into a sparse matrix, one-hot encoding categorical variables, taking the natural log of y for regression problems).
- Model Selection (which model works best for your problem).
- Hyperparameter Optimization (what hyperparameters work best for that model).
- Ensembling Subpredictors (automatically training up models to predict smaller problems within the meta problem).
- Ensembling Weak Estimators (automatically training up weak models on the larger problem itself, to inform the meta-estimator’s decision).
- Big Data (feed it lots of data).
- Unicorns (you could conceivably train it to predict what is a unicorn and what is not).
- Hugs (this makes it much easier to do your job, hopefully leaving you more time to hug those those you care about).

### GetStart

#### Installation

auto_ml模块可直接使用pip命令进行安装，目前版本为auto_ml 2.9.10
```python
pip install auto_ml
```

#### 第三方软件包

- TensorFlow & Keras, XGBoost, LightGBM

auto_ml集成了所有这些很棒的库，包括 DeepLearningClassifier 、 DeepLearningRegressor - XGBClassifier、 XGBRegressor - LGBMClassifer、 LGBMRegressor - CatBoostClassifier和 CatBoostRegressor等等，这些项目对单个预测的预测时间都在1毫秒范围内，并且能够序列化到磁盘，并在训练后加载到新的环境中。auto_ml在没有安装它们的情况下可以很好地运行。只需为model_names传递其中之一即可实现调用。

```python
ml_predictor.train(data, model_names=['DeepLearningClassifier'])
```

#### 使用示例

    1、导入auto_ml中的包Predictor和get_boston_dataset
    2、通过get_boston_dataset获取训练集和测试集
    3、告诉机器数据输出列和不是纯数字的列
    4、建立模型并训练模型
    5、模型训练效果评估
    6、保存模型
    7、使用模型进行预测

```python
from auto_ml import Predictor
from auto_ml.utils import get_boston_dataset

df_train, df_test = get_boston_dataset()

column_descriptions = {
    'MEDV': 'output'
    , 'CHAS': 'categorical'
}

ml_predictor = Predictor(type_of_estimator='regressor', column_descriptions=column_descriptions)

ml_predictor.train(df_train)

test_score = ml_predictor.score(df_test, df_test.MEDV)
file_name = ml_predictor.save()
trained_model = load_ml_model(file_name)evironments)
predictions = trained_model.predict(df_test)
print(predictions)
```

运行结束

```python
+----+----------------+--------------+----------+-------------------+-------------------+-----------+-----------+-----------+-----------+
|    | Feature Name   |   Importance |    Delta |   FR_Decrementing |   FR_Incrementing |   FRD_abs |   FRI_abs |   FRD_MAD |   FRI_MAD |
|----+----------------+--------------+----------+-------------------+-------------------+-----------+-----------+-----------+-----------|
| 12 | CHAS           |            7 | nan      |          nan      |          nan      |  nan      |  nan      |  nan      |  nan      |
|  1 | ZN             |           12 |  11.5619 |           -0.0062 |            0.0380 |    0.0239 |    0.0474 |    0.0000 |    0.0000 |
|  7 | RAD            |           17 |   4.2895 |           -0.1370 |            0.1163 |    0.1588 |    0.1673 |    0.1175 |    0.1035 |
|  2 | INDUS          |           42 |   3.4430 |            0.1632 |           -0.1283 |    0.2332 |    0.2536 |    0.0278 |    0.0785 |
|  8 | TAX            |           50 |  82.9834 |            0.7771 |           -0.3183 |    0.8333 |    0.4623 |    0.3680 |    0.1181 |
|  9 | PTRATIO        |           59 |   1.1130 |            0.6408 |           -0.4830 |    0.6931 |    0.5645 |    0.4029 |    0.2925 |
|  3 | NOX            |           79 |   0.0588 |            0.4932 |           -0.1203 |    0.8052 |    0.6728 |    0.1935 |    0.2717 |
|  0 | CRIM           |          142 |   4.4320 |           -0.6586 |            1.3412 |    0.9985 |    1.7242 |    0.9479 |    1.6564 |
|  5 | AGE            |          143 |  13.9801 |            0.4331 |           -0.4677 |    0.6308 |    0.6133 |    0.5353 |    0.4838 |
|  6 | DIS            |          176 |   1.0643 |            1.7858 |           -0.3931 |    1.8339 |    0.8874 |    1.0996 |    0.3497 |
| 10 | B              |          179 |  45.7266 |           -0.7798 |            0.1640 |    1.1877 |    0.4785 |    1.0904 |    0.1836 |
|  4 | RM             |          197 |   0.3543 |           -1.1633 |            1.6018 |    1.3489 |    1.7951 |    0.9774 |    1.2344 |
| 11 | LSTAT          |          213 |   3.5508 |            2.5125 |           -1.4640 |    2.6368 |    1.5911 |    1.7992 |    1.1952 |
+----+----------------+--------------+----------+-------------------+-------------------+-----------+-----------+-----------+-----------+
```

```python
***********************************************
Advanced scoring metrics for the trained regression model on this particular dataset:

Here is the overall RMSE for these predictions:
2.7608924254130724

Here is the average of the predictions:
21.350718828184544

Here is the average actual value on this validation set:
21.488235294117654

Here is the median prediction:
20.70223186516366

Here is the median actual value:
20.15

Here is the mean absolute error:
2.055851103106312

Here is the median absolute error (robust to outliers):
1.625601367533859

Here is the explained variance:
0.8963149604640271

Here is the R-squared value:
0.8960570877948729
Count of positive differences (prediction > actual):
48
Count of negative differences:
54
Average positive difference:
2.0382305519965307
Average negative difference:
-2.071513815203894

***********************************************
```
