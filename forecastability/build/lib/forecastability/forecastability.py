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
        self.data["date"] = self.data["date"].apply(util.date_converter)
        self.data["year"] = self.data["date"].apply(lambda x: str(x.year))
        if self.tm == "week":
            self.data["week"] = self.data["date"].apply(lambda x: str(x.isocalendar()[1])
                if x.isocalendar()[1] > 9 else "0" + str(x.isocalendar()[1]))
            self.data["week"] = int(self.data["year"] + self.data["week"])
        if self.tm == "month":
            self.data["month"] = self.data["month"].apply(lambda x: str(x.month) 
                if x.month > 9 else "0" + str(x.month))
            self.data["month"] = int(self.data["year"] + self.data["month"])
            
    
    def frequency(self, high=0.75, low=0.3):
        '''
        Calculate frequency of products 
        Args:
            high (float): high bar for high frequency 
            low (float): low bar for extremely low frequency
        '''

        # calculate frequency 
        sku_date_count = self.data.groupby(["sku_code"])[self.tm].apply(lambda x: len(set(x))).reset_index()
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
        self.freq = sku_date_count[["sku_code", "frequency"]]
        return self.freq

    def stability(self, high=5, low=0.7):
        '''
        Calculate stability of products 
        Args:
            high (float): high bar for extremely unstable 
            low (float): low bar for stable
        '''
        # calculate stability 
        groupby_demand = self.data.groupby(["sku_code", self.tm])["qty"].sum().reset_index()
        groupby_demand = groupby_demand.groupby(["sku_code"]).agg(["mean", "std"]).reset_index()
        groupby_demand.columns = ["sku_code", "mean", "std"]
        groupby_demand["cv"] = groupby_demand["std"] / groupby_demand["mean"]

        # split stability 
        def stable_split(x):
            if x < low:
                return "稳定"
            elif x < high:
                return "不稳定"
            return "极端不稳定"

        groupby_demand["stability"] = groupby_demand["cv"].apply(stable_split)
        self.stable = groupby_demand[["sku_code", "stability"]]
        return self.stable
    
    def periodicity(self, threshold=0.8):
        '''
        Calculate periodicity based on threshold of confidence
        '''
        groupby_demand = self.data.groupby(["sku_code", self.tm])["qty"].sum().reset_index()
        groupby_demand = util.fill_ts(groupby_demand, self.tm)
        groupby_demand.sort_values(self.tm, inplace=True)
        
        skus = groupby_demand["sku_code"].unique().tolist()
        print("number of skus: ", len(skus))
        with parallel_backend("multiprocessing", n_jobs=-1):
            results = Parallel()(delayed(self.single_period_detection)(groupby_demand, 
                sku, threshold, i) for i, sku in enumerate(skus))
        result = pd.concat(results, axis=0)
        result = result[result["periodicity"].apply(lambda x: len(x) > 0)]
        self.period = result 
        return self.period 
    
    def single_period_detection(self, groupby_demand, sku, threshold, counter):
        sku_demand = groupby_demand[groupby_demand["sku_code"] == sku]["qty"].tolist()
        period_res = period_detect.solve(sku_demand, threshold, method="dp")
        if period_res is None or len(period_res) == 0:
            return pd.DataFrame()
        period_res = {key: score for key, score in period_res.items() if len(set(key)) > 1}
        res = pd.DataFrame([[sku, period_res]], columns=["sku_code", "periodicity"])
        return res 
    
    def single_customer_percent(self, percent=0.5):
        '''
        Calculate percent of single customer for different products
        Args:
            percent (float): percent threshold of single customer
        '''
        # week frequency 周频率计算
        if self.tm != "week":
            self.data["week"] = self.data["date"].apply(lambda x: str(x.isocalendar()[1])
                if x.isocalendar()[1] > 9 else "0" + str(x.isocalendar()[1]))
            self.data["week"] = self.data["year"] + self.data["week"]
        groupby_sku = self.data.groupby(["sku_code"])["week"].apply(lambda x: len(set(x))).reset_index()
        groupby_sku.columns = ["sku_code", "n_weeks"]

        # customer percent 
        groupby_cust = self.data.groupby(["sku_code", "customer_code"])["qty"].sum().reset_index()
        groupby_cust.columns = ["sku_code", "customer_code", "customer_qty"]
        groupby_sku_sum = self.data.groupby(["sku_code"])["qty"].sum().reset_index()
        groupby_sku_sum.columns = ["sku_code", "qty_sum"]
        groupby_cust = pd.merge(groupby_cust, groupby_sku_sum, on="sku_code", how="left")
        groupby_cust["customer_percent"] = groupby_cust["customer_qty"] / groupby_cust["qty_sum"]

        # single customer percent 单一客户占比超过percent的产品比例
        merge_df = pd.merge(groupby_sku, groupby_cust[["sku_code", 
            "customer_percent"]], on="sku_code", how="inner")
        filter_merge_df = merge_df[merge_df["customer_percent"] > percent]

        merge_df = merge_df.groupby(["n_weeks"])["sku_code"].apply(lambda x: len(set(x))).reset_index()
        merge_df.columns = ["n_weeks", "n_skus"]
        filter_merge_df = filter_merge_df.groupby(["n_weeks"])["sku_code"].apply(lambda x: len(set(x))).reset_index()
        filter_merge_df.columns = ["n_weeks", "sat_n_skus"]
        
        result = pd.merge(merge_df, filter_merge_df, on="n_weeks", how="inner")
        result["sku_percent"] = result["sat_n_skus"] / result["n_skus"]
        self.single_customer = result[["n_weeks", "sku_percent"]]
        return self.single_customer
    
    def render(self, filename="forecastability_report"):

        file = open("{}.html".format(filename), "w")

        # 频率和稳定性表
        if self.freq is not None and self.stable is not None:
            merge_df = pd.merge(self.freq, self.stable, on="sku_code", how="inner")
            high_freq = merge_df[merge_df["frequency"] == "高频"]
            high_stable = len(high_freq[high_freq["stability"] == "稳定"])
            high_unstable = len(high_freq[high_freq["stability"] == "不稳定"])
            high_xunstable = len(high_freq[high_freq["stability"] == "极端不稳定"])

            low_freq = merge_df[merge_df["frequency"] == "低频"]
            low_stable = len(low_freq[low_freq["stability"] == "稳定"])
            low_unstable = len(low_freq[low_freq["stability"] == "不稳定"])
            low_xunstable = len(low_freq[low_freq["stability"] == "极端不稳定"])

            xlow_freq = merge_df[merge_df["frequency"] == "极端低频"]
            xlow_stable = len(xlow_freq[xlow_freq["stability"] == "稳定"])
            xlow_unstable = len(xlow_freq[xlow_freq["stability"] == "不稳定"])
            xlow_xunstable = len(xlow_freq[xlow_freq["stability"] == "极端不稳定"])

            n_stable = len(merge_df[merge_df["stability"] == "稳定"])
            n_unstable = len(merge_df[merge_df["stability"] == "不稳定"])
            n_xunstable = len(merge_df[merge_df["stability"] == "极端不稳定"])

            start = '''<!DOCTYPE html>
                <html>
                <head>
                    <meta charset="UTF-8">
                    <title>Forecastability Report</title>
                        <!-- including ECharts file -->
                    <script src="https://cdnjs.cloudflare.com/ajax/libs/echarts/4.6.0/echarts-en.min.js"></script>
                </head>
                <body> '''
            headers = ["频率/稳定性", "稳定", "不稳定", "极端不稳定", "总计"]
            rows = [
                ["高频", high_stable, high_unstable, high_xunstable, len(high_freq)],
                ["低频", low_stable, low_unstable, low_xunstable, len(low_freq)],
                ["极端低频", xlow_stable, xlow_unstable, xlow_xunstable, len(xlow_freq)],
                ["总计", n_stable, n_unstable, n_xunstable, len(merge_df)]
            ]
            freq_stable_table = util.get_table(headers, rows, "频率和稳定性统计表")
        else:
            freq_stable_table = ""

        # 周期性表
        if self.period is not None:
            headers = ["SKU编码", "周期性结果"]
            rows = self.period.values.tolist()
            period_table = util.get_table(headers, rows, "周期性识别结果表")
        else:
            period_table = ""
        
        end = '</body></html>'

        # 单一客户占比图
        if self.single_customer is not None:
            x = self.single_customer["n_weeks"].tolist()
            y = [round(s * 100, 2) for s in self.single_customer["sku_percent"].tolist()]
            line_charts = util.get_line_charts(x, y, title="单一客户占比超过50%SKU比例和SKU频率图", 
                xname="有需求的周数", yname="单一客户占比超过50%的SKU比例")
        else:
            line_charts = ""

        file.write(start + freq_stable_table + period_table + line_charts + end)
        file.close()
        

if __name__ == "__main__":
    filename = "forecastability_test.csv"
    data = pd.read_csv(filename)
    data = data[:40000]
    fa = Forecastability(data)
    fa.preprocess()
    fa.frequency()
    fa.stability()
    fa.periodicity()
    fa.single_customer_percent()
    # result = fa.single_customer_percent()
    fa.render()
    # import matplotlib.pyplot as plt 
    # plt.plot(result["n_weeks"], result["sku_percent"])
    # plt.show()
