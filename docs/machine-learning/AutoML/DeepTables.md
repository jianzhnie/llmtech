## DeepTables

DeepTables是一个应用于结构化数据的深度学习工具箱， 主要基于 tensorflow2.0+ 和 kears 来实现相关算法。

DeepTables 中开发的模型大多来源于推荐系统——CTR 预估， 近年来比较著名的模型有FM、DeepFM、Wide&amp;Deep、DCN、PNN等。虽然模型来源于推荐系统， 但在合理利用的情况下，这些模型也能对结构化数据提供良好的性能。 DeepTables 将 这些模型进行封装并提供一个通用接口，为用户提供一个端到端工具箱，方便没有开发经验的数据工程师进行调用。

### Models

- Wide&Deep
- DCN(Deep & Cross Network)
- PNN
- DeepFM
- xDeepFM
- AFM
- AutoInt
- FiBiNet
- FGCNN

具体可参考 [Models](https://deeptables.readthedocs.io/en/latest/models.html)

### Transformer

- LgbmLeavesEncoder
- CategorizeEncoder
- MultiLabelEncoder
- MultiKBinsDiscretizer
- GaussRankScaler

### ModelConfig

DeepTables 的参数配置 ModelConfig 是DT中最重要的参数。通过 ModelConfig 可以设置数据预处理，网络模型配置，以及网络的超参数设置等。

### Example

```python

from deeptables.models import deeptable,deepnets
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np


boston_dataset = datasets.load_boston()
df_train = pd.DataFrame(boston_dataset.data)
df_train.columns = boston_dataset.feature_names
y = pd.Series(boston_dataset.target)
X = df_train

conf = deeptable.ModelConfig(
    metrics=['RootMeanSquaredError'],
    nets=['dnn_nets'],
    dnn_params={
        'hidden_units': ((256, 0.3, True), (256, 0.3, True)),
        'dnn_activation': 'relu',
    },
    earlystopping_patience=5,
)

dt = deeptable.DeepTable(config=conf)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model, history = dt.fit(X_train, y_train, epochs=100)
result = dt.evaluate(X_test, y_test)
print(result)
dt_preds = dt.predict_proba(X_test, batch_size=10)
# root mean square error of the model
rmse = (np.sqrt(mean_squared_error(y_test, dt_preds)))

# r-squared score of the model
r2 = r2_score(y_test, dt_preds)

print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
```


## Other packages

除了 DeepTables, 九章还开源了几款其他的工具箱:

- HyperDT/DeepTables: An AutoDL tool for tabular data.
- HyperGBM: A full pipeline AutoML tool integrated various GBM models.
- HyperKeras: An AutoDL tool for Neural Architecture Search and Hyperparameter Optimization on Tensorflow and Keras.
- Cooka: Lightweight interactive AutoML system.
- Hypernets: A general automated machine learning framework.
DataCanvas AutoML Toolkit



### News

DataCanvas（九章）通过这个工具箱拿到了2019年 kaggle 比赛 Categorical Feature Encoding Challenge II的冠军！
