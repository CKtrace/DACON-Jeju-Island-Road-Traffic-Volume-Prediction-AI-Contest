import numpy as np
import pandas as pd
import optuna.integration.lightgbm as lgb

from lightgbm import early_stopping
from lightgbm import log_evaluation
import sklearn.datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler


if __name__ == "__main__":
    df1 = pd.read_csv("11_01_train.csv", index_col=0)
    X = df1.drop('target', axis=1)
    y = df1['target']
    train_x, val_x, train_y, val_y = train_test_split(X, y, test_size=0.09, random_state=2022)
    
    ms = MinMaxScaler()
    train_x = ms.fit_transform(train_x)
    val_x = ms.transform(val_x)
    
    dtrain = lgb.Dataset(train_x, label=train_y)
    dval = lgb.Dataset(val_x, label=val_y)

    params = {
        "objective": "regression",
        "metric": "mae",
        "verbosity": -1,
        "boosting_type": "dart",
        'n_jobs' : -1,
        "device" : 'gpu',
        "gpu_platform_id" : 0,
        "gpu_device_id" : 0,
        "keep_training_booster": True,
    }

    model = lgb.train(
        params,
        dtrain,
        valid_sets=[dtrain, dval],
        callbacks=[early_stopping(100), log_evaluation(100)],
    )

    prediction = np.rint(model.predict(val_x, num_iteration=model.best_iteration))
    accuracy = accuracy_score(val_y, prediction)

    best_params = model.params
    print("Best params:", best_params)
    print("  Accuracy = {}".format(accuracy))
    print("  Params: ")
    for key, value in best_params.items():
        print("    {}: {}".format(key, value))