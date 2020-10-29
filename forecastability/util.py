#-*-coding:utf-8-*-
from datetime import datetime
import pandas as pd 


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


def fill_ts(data, tm):
    '''
    填充时间序列，只保留两列，[ts, y]
    '''
    if tm == "date":
        min_dt = date_converter(data[tm].min())
        max_dt = date_converter(data[tm].max())
        tm_list = [date_parser(x) for x in pd.date_range(start=min_dt, end=max_dt)]
    else:
        min_dt = data[tm].min()
        max_dt = data[tm].max()
        tm_list = list(range(min_dt, max_dt+1))
    tm_df = pd.DataFrame(tm_list, columns=[tm])
    df = pd.merge(tm_df, data[[tm_df, "y"]], on=tm, how="left")
    df["y"].fillna(0, inplace=True)
    return df 
