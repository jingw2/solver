#-*-coding:utf-8 
#@Author: Jing Wang 
#@Date: 2020-11-06 14:47:16 
#@Last Modified by: Jing Wang 
#@Last Modified time: 2020-11-06 14:47:16 
#@reference: 

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
import ts_features 
import util
from sklearn.linear_model import LinearRegression

def mape(y: np.array, ypred: np.array):
    return np.mean(np.abs(y - ypred) / y)

def generate_ts(data: pd.DataFrame):
    data = data[["Date", "Close"]]
    data.columns = ["ds", "y"]
    data["id"] = "amazon"
    params = {
        "lag": {
            "flag": True,
            "window": [7, 14],
        },
        "rolling": {
            "flag": True,
            "window": [7, 14],
        },
        "ewm": {
            "flag": True,
            "advance": 7
        }
    }
    data = ts_features.generate_ts(data, params)
    return data 

def train(data: pd.DataFrame, 
    train_end_date: str = "2019-12-31",
    test_end_date: str = "2020-10-29",
    test_start_date: str = "2020-01-01", 
    roll_days: int = 14,
    adjust: bool = False,
    trendy: bool = False):

    results = []
    prev_error = None
    losses = []
    lr = LinearRegression()
    prev_idx = data[data["ds"] <= train_end_date].index[-1]
    count = 0
    while True:
        train = data.iloc[:prev_idx+1]
        test = data.iloc[prev_idx+1: prev_idx+roll_days]
        roll_end_date = train["ds"].tolist()[-1]
        if len(test) == 0:
            break
        prev_idx = test.index.tolist()[-1]
        feat_cols = [c for c in data.columns.tolist() if c not in ["ds", "y", "id"]]
        Xtrain, Xtest = train[feat_cols], test[feat_cols]
        ytrain, ytest = train["y"], test["y"]
    
        regressor = lr.fit(Xtrain, ytrain)
        ypred = regressor.predict(Xtest)

        # use other error 
        # moving_average = np.mean(ytrain[:-roll_days])
        # error = (ytest.ravel() - ypred.ravel()) * (ypred.ravel() > moving_average).astype(int) + \
        #     (ytest.ravel() - moving_average) * (ypred.ravel() <= moving_average).astype(int)
        
        error = ytest.ravel() - ypred.ravel()
        recent = np.array(ytrain[:-roll_days])
        if len(recent) >= 2:
            trend = np.sign(recent[-1] - recent[-2])
        else:
            trend = 1
        if count > 0 and adjust:
            postprocess = prev_error[:len(ypred)]
            if trendy:
                if trend == -1:
                    ypred[postprocess < 0] += postprocess[postprocess < 0]
                if trend == 1:
                    ypred[postprocess > 0] += postprocess[postprocess > 0]
            else:
                ypred += postprocess 
        loss = mape(ypred.ravel(), np.array(ytest).ravel())
        prev_error = error 
        count += 1 
        test["ypred"] = ypred 
        results.append(test)
        losses.append(loss)
        if roll_end_date >= test_end_date:
            break

    return results, losses

def result_plot(results, title="result_plot"):
    plt.figure()
    results = pd.concat(results, axis=0)
    plt.plot(range(len(results)), results["y"])
    plt.plot(range(len(results)), results["ypred"])
    plt.legend(["y", "ypred"])
    plt.title(title)
    plt.savefig(title + ".png")

def evaluation(losses):
    mu = np.mean(losses)
    std = np.std(losses)
    cv = round(std / mu, 3) 
    mu = round(mu, 3)
    return mu, cv 

def single_main(company, filename, test_start, test_end):
    data = pd.read_csv(filename)
    data = generate_ts(data)
    normal_results, normal_losses = train(data)
    adjust_results, adjust_losses = train(data, adjust=True)
    adjust_trend_results, adjust_trend_losses = train(data, test_start_date=test_start, 
        test_end_date=test_end, adjust=True, trendy=True)

    result_plot(normal_results, company + "_stock_normal_forecast")
    result_plot(adjust_results, company + "_stock_adjust_forecast")
    result_plot(adjust_trend_results, company + "_stock_adjust_trendy_forecast")

    normal_mu, normal_cv = evaluation(normal_losses)
    adjust_mu, adjust_cv = evaluation(adjust_losses)
    adjust_trend_mu, adjust_trend_cv = evaluation(adjust_trend_losses)

    row = [normal_mu, normal_cv, adjust_mu, adjust_cv, adjust_trend_mu, adjust_trend_cv]
    return row 

def main():
    filenames = ["amazon_stock.csv", "google_stock.csv", "alibaba_stock.csv", "jd_stock.csv"]
    companies = ["amazon", "google", "alibaba", "jd"]
    test_start = "2020-01-01"
    test_end = "2020-10-29"
    results = []
    for f, company in zip(filenames, companies):
        row = single_main(company, f, test_start, test_end)
        results.append([company] + row)
    cols = ["Company", "Original Avg MAPE", "Original CV", "Adjust Avg MAPE", "Adjust CV", 
        "Adjust Trendy MAPE", "Adjust Trendy CV"]
    results = pd.DataFrame(results, columns=cols)
    return results 

if __name__ == "__main__":
    results = main()
    print(results)
