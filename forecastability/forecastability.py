#-*-coding:utf-8-*-
#@Author: Jing Wang 
#@Date: 2020-10-29 16:51:06 
#@Last Modified by: Jing Wang 
#@Last Modified time: 2020-10-29 16:51:06 
#@reference: 

'''
Calculate forecastability and output report 
'''
import util 
import period_detect
from joblib import Parallel, delayed,  parallel_backend
import pandas as pd 

class Forecastability:

    def __init__(self, data, tm="date"):
        '''
        Args:
            data (data frame): with columns \
                ["date", "sku_code", "customer_code", "qty"]
            tm (str): time dimension, ["date", "week", "month", "year"]
        '''
        self.data = data 
        if tm not in ["date", "week", "month", "year"]:
            raise Exception("Time dimension is invalid!")
        self.tm = tm
    
    def preprocess(self):
        '''
        Create necessary time dimension
        '''
        self.data["date"] = util.date_converter(self.data["date"])
        self.data["year"] = self.data["year"].apply(lambda x: x.year)
        if self.tm == "week":
            self.data["week"] = self.data["date"].apply(lambda x: str(x)
                if x.isocalendar()[1] > 9 else "0" + str(x))
            self.data["week"] = int(str(self.data["year"]) + self.data["week"])
        if self.tm == "month":
            self.data["month"] = self.data["month"].apply(lambda x: str(x) 
                if x.month > 9 else "0" + str(x))
            self.data["month"] = int(str(self.data["year"]) + self.data["month"])
            
    
    def frequency(self, high=0.75, low=0.3):
        '''
        Calculate frequency of products 
        Args:
            high (float): high bar for high frequency 
            low (float): low bar for extremely low frequency
        '''

        # calculate frequency 
        sku_date_count = self.data.groupby(["sku_code"])[self.tm].apply(lambda x: len(set(x))).reset_index().to_frame()
        sku_date_count.columns = ["sku_code", "tm_stats"]
        tot_tm = len(self.data[self.tm].unique())
        sku_date_count["freq_stats"] = sku_date_count["tm_stats"] / tot_tm

        # split to high, low and extreme low
        def freq_split(x):
            if x >= high:
                return "高频" 
            elif x >= low:
                return "低频"
            return "极端低频"

        sku_date_count["frequency"] = sku_date_count["freq_stats"].apply(freq_split)
        return sku_date_count[["sku_code", "frequency"]]

    def stability(self, high=5, low=0.7):
        '''
        Calculate stability of products 
        Args:
            high (float): high bar for extremely unstable 
            low (float): low bar for stable
        '''
        # calculate stability 
        groupby_demand = self.data.groupby(["sku_code", self.tm])["qty"].sum().reset_index().to_frame()
        groupby_demand = groupby_demand.groupby(["sku_code"]).agg(["mean", "std"]).reset_index()
        groupby_demand["cv"] = groupby_demand["std"] / groupby_demand["mean"]

        # split stability 
        def stable_split(x):
            if x < low:
                return "稳定"
            elif x < high:
                return "不稳定"
            return "极端不稳定"

        groupby_demand["stability"] = groupby_demand["cv"].apply(stable_split)
        return groupby_demand[["sku_code", "stability"]]
    
    def periodicity(self, threshold=0.8):
        '''
        Calculate periodicity based on threshold of confidence
        '''
        groupby_demand = self.data.groupby(["sku_code", self.tm])["qty"].sum().reset_index().to_frame()
        groupby_demand = util.fill_ts(groupby_demand, self.tm)
        groupby_demand.sort_values(self.tm, inplace=True)

        def single_period_detection(groupby_demand, sku, threshold):
            sku_demand = groupby_demand[groupby_demand["sku_code"] == sku]["qty"].tolist()
            period_res = period_detect.solve(sku_demand, threshold, method="dp")
            if len(period_res) == 0:
                return  
            period_res = {key: score for key, score in period_res.items() if len(set(key)) > 1}
            res = pd.DataFrame([sku, period_res], columns=["sku_code", "periodicity"])
            return res 
        
        skus = groupby_demand["sku_code"].unique().tolist()
        with parallel_backend("multiprocessing", n_jobs=-1):
            results = Parallel()(delayed(single_period_detection)(groupby_demand, 
                sku, threshold) for sku in skus)
        result = pd.concat(results, axis=0)
        return result 
    
    def single_customer_percent(self, percent=0.5):
        '''
        Calculate percent of single customer for different products
        Args:
            percent (float): percent threshold of single customer
        '''
        # week frequency 周频率计算
        if self.tm != "week":
            self.data["week"] = self.data["date"].apply(lambda x: str(x)
                if x.isocalendar()[1] > 9 else "0" + str(x))
        groupby_sku = self.data.groupby(["sku_code"])["week"].apply(lambda x: len(set(x))).reset_index()
        groupby_sku.columns = ["sku_code", "n_weeks"]
        groupby_cust = self.data.groupby(["sku_code", "customer_code"])["qty"].sum().reset_index()
        
        # single customer percent 单一客户占比超过percent的产品比例
