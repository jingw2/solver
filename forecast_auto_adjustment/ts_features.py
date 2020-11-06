# -*- coding:utf-8 -*-
#@author: Jing Wang
#@date: 09/17/2020

'''
特征工程模块：时间序列生成，包括基本时间序列，lag特征和滚动特征
'''

import json
import argparse
import datetime
import pandas as pd
from joblib import Parallel, delayed 

import os 
import sys 
import util 

import chinese_calendar

CHINESE_HOLIDAYS = chinese_calendar.constants.holidays
CHINESE_WORKDAYS = chinese_calendar.constants.workdays
CHINESE_LIEUDAYS = chinese_calendar.constants.in_lieu_days

def get_holiday_stats(min_year, max_year):
    '''
    计算节假日的第几天，节假日第1天，节假日最后一天

    Arg:
        min_year (int): 开始扫描的年份
        max_year (int): 结束扫描的年份 
        [min_year, max_year]
    '''
    holiday_days = list(CHINESE_HOLIDAYS.keys())
    holiday_days.sort()
    holiday_index = {}
    holiday_first = set([])
    holiday_final = set([])
    count = 1
    is_prev_final = False
    for idx, day in enumerate(holiday_days):
        if day.year < min_year or day.year > max_year:
            continue
        next_day = day + datetime.timedelta(days=1)
        prev_day = day - datetime.timedelta(days=1)
        if next_day in holiday_days:
            if is_prev_final:
                holiday_first.add(day)
            holiday_index[day] = count
            is_prev_final = False 
            count += 1 
        else:
            if not prev_day in holiday_days:
                holiday_first.add(day)
            holiday_final.add(day)
            is_prev_final = True
            holiday_index[day] = count
            count = 1 
    return holiday_index, holiday_first, holiday_final

def get_before_after_holiday(data, before_count=5, after_count=5):
    '''
    节假日前后统计
    Args:
        data (pd.DataFrame)
        before_count (int)
        after_count (int)
    '''
    before = {}
    after = {}
    first_day = set(data.loc[data["is_holiday_first_fea"] == 1, "ds"].tolist())
    final_day = set(data.loc[data["is_holiday_final_fea"] == 1, "ds"].tolist())
    for first in first_day:
        for c in range(1, before_count+1):
            day = first - datetime.timedelta(days=c)
            before[day] = c
    for final in final_day:
        for c in range(1, after_count+1):
            day = final + datetime.timedelta(days=c)
            after[day] = c
    data["before_holiday_day_fea"] = data["ds"].apply(lambda x: before[x] if x in before else 0)
    data["after_holiday_day_fea"] = data["ds"].apply(lambda x: after[x] if x in after else 0)
    return data

def basic_ts(data):
    '''
    根据ds生成基本的时间序列特征

    Args:
        data (DataFrame): 数据表
    Return:
        data (DataFrame): 数据表带有基本时间序列特征
    '''
    data["ds"] = data["ds"].apply(util.date_converter)

    # 生成时间特征 x_of_y
    data["day_of_week_fea"] = data["ds"].apply(lambda x: x.isoweekday()  # monday表示1
        if isinstance(x, datetime.datetime) else None)
    data["day_of_month_fea"] = data["ds"].apply(lambda x: x.day
        if isinstance(x, datetime.datetime) else None)
    data["day_of_year_fea"] = data["ds"].apply(lambda x: x.timetuple().tm_yday
        if isinstance(x, datetime.datetime) else None)
    data["week_of_year_fea"] = data["ds"].apply(lambda x: x.isocalendar()[1]
        if isinstance(x, datetime.datetime) else None)
    data["month_of_year_fea"] = data["ds"].apply(lambda x: x.month
        if isinstance(x, datetime.datetime) else None)
    data["is_weekend_fea"] = data["day_of_week_fea"].apply(lambda x: 1 if x >= 6 else 0)

    # 是否节假日，是否工作日，是否休息日/调休
    data["is_holiday_fea"] = data["ds"].apply(lambda x: 1 if x.date() in CHINESE_HOLIDAYS else 0)
    data["is_workday_fea"] = data["ds"].apply(lambda x: 1 if x.date() in CHINESE_WORKDAYS else 0)
    data["is_lieuday_fea"] = data["ds"].apply(lambda x: 1 if x.date() in CHINESE_LIEUDAYS else 0)
 
    # 节假日第几天
    min_year = data["ds"].min().year - 1
    max_year = data["ds"].max().year + 1
    holiday_index, holiday_first, holiday_final = get_holiday_stats(min_year, max_year)
    data["is_holiday_first_fea"] = data["ds"].apply(lambda x: 1 if x.date() in holiday_first else 0)
    data["is_holiday_final_fea"] = data["ds"].apply(lambda x: 1 if x.date() in holiday_final else 0)
    data["holiday_day_fea"] = data["ds"].apply(lambda x: holiday_index[x.date()] if x.date() in holiday_index else 0)

    # 节前第几天，节后第几天
    data = get_before_after_holiday(data, before_count=5, after_count=5)
    data["ds"] = data["ds"].apply(util.date_parser)
    return data 

def lag_ts(data, lag_windows=[1, 7]):
    '''
    根据lag_windows生成lag特征，windows的单位：天

    Args:
        data (DataFrame): 输入数据表
        lag_windows (list): lag时间窗口大小，单位为天
    '''
    for lag in lag_windows:
        data[f'{lag}_day_before_fea'] = data["y"].shift(lag)
    return data 

def roll_ts(data, roll_windows=[1, 7]):
    '''
    滚动特征

    Args:
        data (DataFrame): 输入数据表
        roll_windows (list): 滚动时间窗口大小，单位为天
    '''

    for window in roll_windows:
        roll = data["y"].shift(1).rolling(window=window)
        tmp = pd.concat([roll.max(), roll.min(), roll.mean(), roll.sum(), roll.median()], axis=1)
        tmp.columns = [f'max_over_{window}_days_fea', f'min_over_{window}_days_fea', 
            f'mean_over_{window}_days_fea', f'sum_over_{window}_days_fea', f'median_over_{window}_days_fea']
        data = pd.concat([data, tmp], axis=1)
    return data 

def ewm_ts(data, advance):
    '''
    指数加权平均

    Args:
        data (DataFrame)：输入数据表
    '''
    shifted = data["y"].shift(advance)
    data["ewm_fea"] = shifted.ewm(alpha=0.5, adjust=True, ignore_na=False).mean()
    return data 

def ts_single(data, lag, roll, ewm, lag_windows, roll_windows, ewm_advance):
    '''
    基于某个ID的序列，生成关于这个ID的时间序列
    '''
    data.sort_values("ds", inplace=True)

    # 保证日期的连续性
    df = util.fill_ts(data)

    if lag:
        df = lag_ts(df, lag_windows)
    if roll:
        df = roll_ts(df, roll_windows)
    if ewm:
        df = ewm_ts(df, ewm_advance)
    
    df.drop(columns=["y"], axis=1, inplace=True)
    data = pd.merge(data, df, on="ds", how="left")
    return data 

def generate_ts(data, params, n_jobs=-1):
    if "ds" in data:
        data["ds"] = data["ds"].apply(lambda x: util.date_parser(util.date_converter(x)))
    lag = params["lag"]["flag"]
    roll = params["rolling"]["flag"]
    ewm = params["ewm"]["flag"]

    lag_windows = params["lag"].get("window", None)
    roll_windows = params["rolling"].get("window", None)
    ewm_advance = params["ewm"].get("advance", None)
    skus = data["id"].unique().tolist()
    results = Parallel(n_jobs=n_jobs, verbose=0)(delayed(ts_single)(data.loc[data["id"] == sku], \
        lag, roll, ewm, lag_windows, roll_windows, ewm_advance) for sku in skus)
    output = pd.concat(results, axis=0)
    output = basic_ts(output)

    # 填充0
    output.fillna(0, inplace=True)
    return output 
