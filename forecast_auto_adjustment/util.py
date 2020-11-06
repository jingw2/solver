# -*- coding:utf-8 -*-
# Author: Xingyu Liu 01368856
# Date: Feb 06, 2020

#@modified: Jing Wang
#@date: 09/18/2020

import os
import json
import random
import calendar
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta, datetime
from monthdelta import monthdelta
import lightgbm as lgb
from sklearn.model_selection import KFold
from bayes_opt import BayesianOptimization
from sklearn import model_selection
from itertools import product
from copy import deepcopy
import xgboost as xgb

def get_data_path():
    folder = os.path.split(os.path.realpath(__file__))[0]  # os.path.dirname(os.path.dirname(__file__))
    return os.path.join(folder, "")

def is_json(myjson):
    try:
        json.loads(myjson)
    except:
        return False
    return True

def output_json(data, filename):
    '''
    output data to json
    :param data:
    :param filename:
    :return:
    '''
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)

def draw_feature_importance(report_path, feature_importance):
    # draw feature importance
    photoLength = len(feature_importance) / 2 if len(feature_importance) > 10 else 5
    plt.figure(figsize=(20, photoLength))
    sns.barplot(x='Value', y='Feature', data=feature_importance.sort_values(by='Value', ascending=False))
    plt.title("LightGBM Feature Importance")
    plt.tight_layout()
    plt.savefig(report_path + "feature_importance.png")

def get_dates(year, month):
    year = int(year)
    month = int(month)
    _, ndays = calendar.monthrange(year, month)
    if month < 10:
        mon = str(0) + str(month)
    else:
        mon = str(month)
    base = str(year) + mon
    dates = []
    for d in range(1, ndays):
        if d < 10:
            d = str(0) + str(d)
        else:
            d = str(d)
        dates.append(int(base + d))
    return dates

def get_period_value_and_unit(period):
    '''
    把周期字符串拆解为数值和单位
    :param period: 输入的周期，字符串，如"7d"
    :return: 周期对应的数值及单位，如返回7和"d"
    '''
    # default value
    period_value = 7
    period_unit = 'd'

    if period.endswith('m'):
        period_unit = 'm'
        period_value = int(period.replace('m', ''))
    elif period.endswith('d'):
        period_unit = 'd'
        period_value = int(period.replace('d', ''))

    return period_value, period_unit

def add_some_time(cur_time_str, value, unit):
    '''
    从某个时刻增加一段时间
    :param cur_time_str: 当前时间，字符串类型
    :param value: 需要增加的时间长度
    :param unit: 时间长度的单位
    :return: 结果字符串
    '''

    val_start_date = datetime.strptime(cur_time_str, '%Y-%m-%d')
    if unit == 'm':
        val_week_date = val_start_date + monthdelta(months=value)
    elif unit == 'd':
        val_week_date = val_start_date + timedelta(days=value)
    else:
        raise ValueError('Incorrect value with roll_period {}. '.format(str(value)+str(unit)))

    return val_week_date.strftime("%Y-%m-%d")


def train_test_split(X, y, train_ratio=0.7):
    num_periods, num_features = X.shape
    train_periods = int(num_periods * train_ratio)
    random.seed(2)
    Xtr = X[:train_periods]
    ytr = y[:train_periods]
    Xte = X[train_periods:]
    yte = y[train_periods:]
    return Xtr, ytr, Xte, yte


###############################################################
# metric
###############################################################

# define MAPE function
def mean_absolute_percentage_error(y_true, y_pred):
    '''
    :param y_true: 实际Y值
    :param y_pred: 预测Y值
    :return: MAPE
    '''
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true))) * 100
    return mape

def MAPE_handle_zero(y_true, y_pred):
    '''
    * 此处，为了防止一些实际值为0的情况，此处分母处加了1e-2，可能会导致MAPE的值高启，需要注意。
    :param y_true: 实际Y值
    :param y_pred: 预测Y值
    :return: MAPE
    '''
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-2))) * 100
    return mape

# define WMAPE function
def weighted_mean_absolute_percentage_error(y_true, y_pred):
    '''
    :param y_true: 实际Y值
    :param y_pred: 预测Y值
    :return: WMAPE
    '''
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    wmape = 100 * np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)
    return wmape

def WMAPE_handle_zero(y_true, y_pred):
    '''
    :param y_true: 实际Y值
    :param y_pred: 预测Y值
    :return: WMAPE
    '''
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    wmape = 100 * np.sum(np.abs(y_true - y_pred)) / (np.sum(y_true) + 1e-2)
    return wmape


# define SMAPE function
def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    '''
    :param y_true: 实际Y值
    :param y_pred: 预测Y值
    :return: SMAPE
    '''
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    smape = 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100
    return smape

def SMAPE_handle_zero(y_true, y_pred):
    '''
    * 此处，为了防止一些实际值为0的情况，此处分母处加了0.01，可能会导致MAPE的值高启，需要注意。
    :param y_true: 实际Y值
    :param y_pred: 预测Y值
    :return: SMAPE
    '''
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    smape = 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + 1e-2)) * 100
    return smape

def add_lag_and_window_feature_name(train_features, lag_list, window_list):
    '''
    添加需要滚动的特征名称
    :param train_features:
    :param lag_list:
    :param window_list:
    :return:
    '''
    for lag in lag_list:
        train_features.append(f'{lag}_day_before')
    for w in window_list:
        train_features.extend([f'max_over_{w}_days', f'min_over_{w}_days', f'mean_over_{w}_days', f'sum_over_{w}_days'])


def construct_features(data, lag_list, window_list):
    basic = pd.DataFrame(data.y)
    for lag in lag_list:
        tmp = basic.shift(lag)
        tmp.rename(columns={'y': f'{lag}_day_before'}, inplace=True)
        data = pd.concat([data, tmp], axis=1)

    for w in window_list:
        shifted = basic.shift(1)
        window = shifted.rolling(window=w)
        tmp = pd.concat([window.max(), window.min(), window.mean(), window.sum()], axis=1)
        tmp.columns = [f'max_over_{w}_days', f'min_over_{w}_days', f'mean_over_{w}_days', f'sum_over_{w}_days']
        data = pd.concat([data, tmp], axis=1)

    return data

def date_converter(x):
    '''
    转换为日期格式
    '''
    if x is None:
        return x
    try:
        x = str(x)
    except Exception:
        return x
    
    try:
        return datetime.strptime(x, "%Y-%m-%d")
    except Exception:
        try:
            return datetime.strptime(x, "%Y/%m/%d")
        except Exception:
            try:
                return datetime.strptime(x, "%Y%m%d")
            except Exception:
                return x

def date_parser(x):
    '''
    日期格式转换为string
    '''
    if not isinstance(x, datetime):
        return None
    
    try:
        return x.strftime("%Y-%m-%d")
    except Exception:
        try:
            return x.strptime("%Y/%m/%d")
        except Exception:
            try:
                return x.strptime("%Y%m%d")
            except Exception:
                return None

def fill_ts(data):
    '''
    填充时间序列，只保留两列，[ts, y]
    '''

    min_dt = date_converter(data["ds"].min())
    max_dt = date_converter(data["ds"].max())
    date_list = [date_parser(x) for x in pd.date_range(start=min_dt, end=max_dt)]
    date_df = pd.DataFrame(date_list, columns=["ds"])
    df = pd.merge(date_df, data[["ds", "y"]], on="ds", how="left")
    df["y"].fillna(0, inplace=True)
    return df 

def dt64_to_datetime(dt64):
    '''
    :param dt64:
    :return:
    '''
    if np.isnat(dt64):
        return None
    else:
        unix_epoch = np.datetime64(0, 's')
        one_second = np.timedelta64(1, 's')
        seconds_since_epoch = (dt64 - unix_epoch) / one_second
    return datetime.utcfromtimestamp(seconds_since_epoch)

def get_date_diff(start_date_str, end_date_str):
    '''
    获取日期差
    :param start_date_str:str
    :param end_date_str:str
    :return:
    '''
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    ret_val = (end_date-start_date).days
    return ret_val

def get_dates_list(start_date, end_date):
    '''
    获取日期区间
    :param start_date:str
    :param end_date:str
    :return:
    '''
    date_list = []
    start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    while start_date <= end_date:
        date_str = start_date.strftime("%Y-%m-%d")
        date_list.append(date_str)
        start_date += datetime.timedelta(days=1)
    return date_list

def get_model_info(model_name, data, results, mode):
    'Get model information output'
    train_size = len(data[data["set_flag"] == mode["train"]])
    val_size = len(data[data["set_flag"] == mode["validation"]])
    test_size = len(data[data["set_flag"] == mode["test"]])
    val_data = results[results["set_flag"] == mode["validation"]]
    y = val_data["y"]
    ypred = val_data["y_pred"]
    info = {}
    info["model"] = model_name 
    info["train_set_size"] = train_size
    info["validation_set_size"] = val_size 
    info["test_set_size"] = test_size
    info["WMAPE"] = WMAPE_handle_zero(y, ypred)
    return info 

class GridSearchCV(object):
    
    def __init__(self, params_grid, model="lightgbm", cv=5, random_state=0):
        self.cv = cv 
        self.random_state = random_state 

        basic_params = {}
        search_params = {}
        for param, values in params_grid.items():
            if len(values) == 1:
                basic_params[param] = values
            else:
                search_params[param] = values 
        self.basic_params = basic_params
        self.param_grid = search_params

        self.model = model 
        self.num_boost_round = 1000
        self.early_stopping_rounds = 250

    def generate_params(self):
        # Always sort the keys of a dictionary, for reproducibility
        items = sorted(self.param_grid.items())
        if not items:
            yield {}
        else:
            keys, values = zip(*items)
            for v in product(*values):
                params = dict(zip(keys, v))
                yield params

    def fit(self, X, y, features, cat_features=None, init_points=5, n_iter=5, 
            bayes_automated_tune=False,
            grid_tune=True):
        '''
        Grid Search Fit
        Args:
            X (data frame)
            y (np array)
            features (list): a list of feature columns to use
            init_points (int): how many steps of random exploration
            n_iter (int): how many iterations of bayesian optimization
            bayes_automated_tuning (bool): automated fine tuning
            grid_tune (bool): grid search

        Note:
        You could just set either init_points or n_iter as 0
        '''
        self.Xtrain = X
        self.ytrain = y
        self.features = features
        self.cat_features = cat_features

        if bayes_automated_tune and len(self.param_grid) > 0:
            optimizer = BayesianOptimization(
                f=self.fold_train,
                pbounds=self.param_grid
            )
            optimizer.maximize(
                init_points=init_points,
                n_iter=n_iter,
            )

            # get best parameters 
            best_param = optimizer.max["params"]
            for p, val in best_param.items():
                if p in ["min_child_samples", "num_leaves", 
                        "max_depth", "n_estimators", "random_state"]:
                    val = int(val)
                self.basic_params[p] = val
        
        if grid_tune and len(self.param_grid) > 0:
            best_score = float("-inf")
            best_param = None
            for param in self.generate_params():
                score = self.fold_train(**param)
                if score > best_score:
                    best_score = score 
                    best_param = deepcopy(self.basic_params)
            self.basic_params = best_param
        
        if "weight" not in X.columns:
            X["weight"] = 1
        
        Xtr, Xval, ytr, yval = model_selection.train_test_split(X, y, 
                    test_size=0.1, random_state=self.random_state)
        
        if self.cat_features is None:
            cat_feat = "auto"
        else:
            cat_feat = self.cat_features
        
        if self.model == "lightgbm":
            trn_data = lgb.Dataset(
                    Xtr[features], 
                    label=ytr, 
                    weight=Xtr.weight,
                    categorical_feature=cat_feat
                )
            
            val_data = lgb.Dataset(
                Xval[features],
                label=yval,
                weight=Xval.weight,
                categorical_feature=cat_feat
            )

            self.best_estimator_ = lgb.train(
                self.basic_params,
                trn_data,
                num_boost_round=self.num_boost_round,
                valid_sets=[trn_data, val_data],
                early_stopping_rounds=self.early_stopping_rounds,
                verbose_eval=False,
            )
        elif self.model == "xgboost":
            trn_data = xgb.DMatrix(Xtr[features], label=ytr)
            val_data = xgb.DMatrix(Xval[features], label=yval)
            params = {k: v[0] for k, v in self.basic_params.items()}
            self.best_estimator_ = xgb.train(params, trn_data, 
                evals=[(val_data, "validation")],
                verbose_eval=False,
                num_boost_round=self.num_boost_round,
                early_stopping_rounds=self.early_stopping_rounds)

        self.best_params_ = self.basic_params
    
    def fold_train(self, **kwargs):
        for p, val in kwargs.items():
            if p in ["min_child_samples", "num_leaves", "max_depth", 
                    "n_estimators", "random_state"]:
                val = int(val)
            self.basic_params[p] = [val] 

        scores = []
        Xtrain = self.Xtrain
        ytrain = self.ytrain
        features = self.features
        
        if self.cat_features is None:
            cat_feat = "auto"
        else:
            cat_feat = self.cat_features

        if "weight" not in Xtrain.columns:
            Xtrain["weight"] = 1

        folds = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state) 
        for fold_idx, (trn_idx, val_idx) in enumerate(folds.split(Xtrain.values, ytrain)):
            t_x = Xtrain.iloc[trn_idx]
            v_x = Xtrain.iloc[val_idx]
            label_train = ytrain[trn_idx].ravel()
            label_val = ytrain[val_idx].ravel()

            if self.model == "lightgbm":
                trn_data = lgb.Dataset(
                    t_x[features], 
                    label=label_train, 
                    weight=t_x.weight,
                    categorical_feature=cat_feat
                )
                val_data = lgb.Dataset(
                    v_x[features],
                    label=label_val,
                    weight=v_x.weight,
                    categorical_feature=cat_feat
                )
                # start = datetime.now()
                regressor = lgb.train(
                    self.basic_params,
                    trn_data,
                    num_boost_round=self.num_boost_round,
                    valid_sets=[trn_data, val_data],
                    early_stopping_rounds=self.early_stopping_rounds,
                    verbose_eval=False,
                )

                val_feat = v_x[features]
            elif self.model == "xgboost":
                trn_data = xgb.DMatrix(t_x[features], label=label_train)
                val_data = xgb.DMatrix(v_x[features], label=label_val)
                params = {k: v[0] for k, v in self.basic_params.items()}
                regressor = xgb.train(params, trn_data, 
                    evals=[(val_data, "validation")],
                    verbose_eval=False,
                    num_boost_round=self.num_boost_round,
                    early_stopping_rounds=self.early_stopping_rounds)
                val_feat = xgb.DMatrix(v_x[features])
            
            ypred = regressor.predict(val_feat).ravel()
            mae = np.mean(np.abs(ypred - label_val))
            scores.append(mae)
            # end = datetime.now()
            # print("Time spent: {}s".format((end-start).total_seconds()))
            # raise
        return -np.mean(scores)
