import pandas as pd
import lightgbm as LGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler
import warnings

warnings.filterwarnings("ignore")


df1 = pd.read_csv("11_01_train.csv", index_col=0)
df2 = pd.read_csv("11_01_test.csv", index_col=0)

X = df1.drop('target', axis=1)
y = df1['target']

train_x, val_x, train_y, val_y = train_test_split(X, y, test_size=0.1, random_state=2)

ms = MinMaxScaler()
train_x = ms.fit_transform(train_x)
val_x = ms.transform(val_x)
df2 = ms.transform(df2)

dtrain = LGBRegressor.Dataset(train_x, label=train_y)
dval = LGBRegressor.Dataset(val_x, label=val_y)

params = {
    "objective" : "regression", 
    "metric": "mae",
    "verbosity": -1,
    "boosting_type": "dart",
    "skip_drop" : 0.9,
    "feature_pre_filter": True,
    "num_iterations": 20000,
    "learning_rate" : 0.4,
    "device" : 'gpu',
    "gpu_platform_id" : 0,
    "gpu_device_id" : 0,
    "n_jobs" : -1,

}

model = LGBRegressor.train(
        params,
        dtrain,
        valid_sets=[dtrain, dval],
        keep_training_booster = True,
    )

prediction = model.predict(df2)

submission=pd.read_csv('sample_submission.csv', index_col=0)  
submission['target']=prediction

submission.to_csv('11_14_05.csv')
