import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

split = np.datetime64("2022-02-01")

def fit_and_test(model, X: pd.DataFrame, y: pd.DataFrame, xlim=(split-5, split+5)):
    
    nona = ~X.isna().any(axis=1) & ~y.isna()
    X = X[nona]
    y = y[nona]

    y_diff = y.diff()
    y_diff = y_diff.fillna(0)

    # train test split
    train_idx = (X.index <= '2021-12-20') | \
                (X.index >= '2022-1-1') & (X.index <= '2022-1-20') | \
                (X.index >= '2022-2-1') & (X.index <= '2022-2-20')
    train_idx
    test_idx = ~train_idx

    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y_diff.loc[train_idx], y_diff.loc[test_idx]
    print(X_train.shape, X_test.shape)
    
    # model fit
    model_fit = model.fit(X_train, y_train)
    d_pred = model_fit.predict(X)
    print('train score:\t', model.score(X_train, y_train))
    print('test score:\t', model.score(X_test, y_test))
    print()
    print('self RMSE:\t', np.sqrt(((y - y.shift(1)) ** 2).mean())) 
    print('pred RMSE:\t', np.sqrt(((y - d_pred) ** 2).mean()))
    print('train RMSE:\t', np.sqrt(((y_train - d_pred[: y_train.shape[0]]) ** 2).mean()))
    print('test RMSE:\t', np.sqrt(((y_test - d_pred[-y_test.shape[0] :]) ** 2).mean()))
    print()
    plt.figure(figsize=(14, 6))
    plt.subplot(2, 1, 1)
    plt.plot(pd.Series(d_pred, index=y.index))
    plt.plot(pd.Series(y_diff, index=y.index))
    plt.plot([split, split], [y_diff.min(), y_diff.max()], "r--")
    plt.legend(["Predicted", "Actual"])
    plt.xlim(*xlim)

    y_pred = y.shift(1) + pd.Series(d_pred, index=y.index)
    print('self RMSE:\t', np.sqrt(((y - y.shift(1)) ** 2).mean()))
    print('pred RMSE:\t', np.sqrt(((y - y_pred) ** 2).mean()))
    print('train RMSE:\t', np.sqrt(((y[train_idx] - y_pred[train_idx]) ** 2).mean()))
    print('test RMSE:\t', np.sqrt(((y[test_idx] - y_pred[test_idx]) ** 2).mean()))
    print()
    plt.subplot(2, 1, 2)
    plt.plot(y_pred)
    plt.plot(y)
    plt.plot([split, split], [y.min(), y.max()], "r--")
    plt.xlim(*xlim)
    plt.legend(["Predicted", "Actual"])

    return model_fit

def fit(model, X: pd.DataFrame, y: pd.DataFrame):
    y = y.diff()
    nona = ~X.isna().any(axis=1) & ~y.isna()
    return model.fit(X[nona], y[nona])

