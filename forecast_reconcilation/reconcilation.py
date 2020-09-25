# -*- coding:utf-8 -*-
#@author: Jing Wang
#@date: 09/24/2020

'''
Hierarchical Forecast Reconcilation
层级预测后的调和
* 实现Forecasting Principles and Practice中的最优调和方法, 章节10.7
* 参考代码：https://github.com/carlomazzaferro/scikit-hts/blob/master/hts/functions.py
'''
from data_structure import HierarchyTree
import pandas as pd
import numpy as np 

def get_summing_matrix(tree: HierarchyTree):
    '''
    递归生成Summing Matrix
    '''
    nodename = list(tree.nodes.keys())
    bottoms = tree.bottom
    num_bottoms = tree.num_bottom_level
    num_nodes = tree.num_nodes
    mat = np.zeros((num_nodes, num_bottoms))

    def dfs(mat, node):
        idx = nodename.index(node.name)
        if node.name != "root" and not node.children:
            mat[idx, bottoms.index(node)] = 1
        for child in node.children:
            dfs(mat, child)
            child_idx = nodename.index(child.name)
            mat[idx] += mat[child_idx]

    dfs(mat, tree.root)
    return mat

def top_down(forecasts: dict, tree: HierarchyTree, method="avg_hist_prop"):
    '''
    Top down method
    从上至下拆分
        1. 按照历史比例: Average Historical Proportions, avg_hist_prop
            p_j = 1 / T * \sum_{t=1}^T y_{j, t} / y_t
        2. 按照历史平均比例: Proportions of Historical Average, prop_hist_avg
        3. 按预测比例: Forecast Proportions, forecast_prop
    '''
    raise NotImplementedError

def bottom_up(forecasts: pd.DataFrame, tree: HierarchyTree):
    '''
    自下而上汇总
        y_tilde = S y_hat_bottom
    '''
    nodenames = list(tree.nodes.keys())
    S = get_summing_matrix(tree)
    ypred = df_to_array(forecasts, nodenames)
    num_bottom_level = tree.num_bottom_level
    bottom_pred = ypred[-num_bottom_level:, :]
    y = S @ bottom_pred
    results = {}
    for idx, name in enumerate(nodenames):
        results[name] = y[idx]
    return results

def optimal_reconcilation(forecasts: pd.DataFrame, tree: HierarchyTree, method="ols", 
        residuals: pd.DataFrame = None):
    '''
    Optimal Reconcilation Algorithm：
    最优调和算法
        y_tilde = S P y_hat_bottom
        y_tilde = S (S^T W_h^{-1} S)^{-1} S^T W_h^{-1} y_hat_bottom

    S: summing matrix，反映层级汇总关系
    P: constraint matrix
    W_h: W_h = Var[y_{T+h} - y_tilde] = SP W_h P^T S^T, y_{T+h} is true value

    Task is to estimate W_h
        1. ols: oridinary least square method，最小二乘法 W_h = k_h I
        2. wls: weighted least square method，加权最小二乘法, W_h = k_h diag(W_hat1)
            W_hat1 = 1 / T * \sum_{t=1}^T e_t e_t^T, 
                e_t is n dimension vector of residuals，e_t是残差/误差向量
        3. nseries: W_h = k_h Omega, Omega = diag(S 1), 1 is unit vector of dimension。
            S列求和后取最小线
        4. mint: W_h = k_h W_1, W_1 sample/residual covariance, 样本协方差矩阵，也可以用残差协方差矩阵
            the number of bottom-level series is much larger than T, so shrinkage covariance to 
            diagnoal
    '''
    nodenames = list(tree.nodes.keys())
    num_nodes = tree.num_nodes
    for name in nodenames:
        assert name in forecasts.columns
    dates = forecasts.index.tolist()

    S = get_summing_matrix(tree)
    ypred = df_to_array(forecasts, nodenames)
    kh = 1 
    if method == "ols":
        Wh = np.eye(num_nodes) * kh 
    if method == "wls":
        residuals = df_to_array(residuals, nodenames)
        What1 = residuals @ residuals.T 
        diag = np.eye(num_nodes) * np.diag(What1)
        Wh = kh * diag
    if method == "nseries":
        diag = np.eye(num_nodes) * np.diag(np.sum(S, axis=1))
        Wh = kh * diag
    if method == "mint":
        residuals = df_to_array(residuals, nodenames)
        cov = np.cov(residuals)
        diag = np.eye(num_nodes) * np.diag(cov)
        Wh = kh * diag
    inv_Wh = np.linalg.inv(Wh)
    coef = S @ (np.linalg.inv(S.T @ inv_Wh @ S)) @ S.T @ inv_Wh
    y = coef @ ypred

    results = pd.DataFrame(y, columns=dates)
    results["id"] = nodenames 
    return results

def df_to_array(forecasts, nodenames):
    '''
    DataFrame to array based on node names input

    Usage:

        DataFrame like this: 

        | all | series1 | series_2 | series1_sku1 | series2_sku1 |
        | 1000| 200     | 300      | 100          | 250          |
    
        to Array:
            array([1000, 200, 300, 100, 250]).T
    '''
    forecasts = forecasts[nodenames]
    arr = np.asarray(forecasts).T
    return arr

def example():
    data = pd.read_csv("reconcilation_test.csv")
    series = data.loc[~data["series"].isna() & data["sku"].isna(), 
        ["series"]].drop_duplicates()
    series = series["series"].tolist()
    series = [s for s in series if s != "all"]
    skus = data.loc[~data["sku"].isna(), ["series", "sku"]].drop_duplicates()
    skus = (skus["series"] + "_" + skus["sku"]).tolist()
    
    # 因为stores就1个，就作为root
    total = {"root": series} # root对应层，是第一层
    skus_h = {k: [v for v in skus if v.startswith(k)] for k in series}
    hierarchy = {**total, **skus_h}

    tree = HierarchyTree.from_nodes(hierarchy)

    def clear_ids(ids):
        cols = []
        for c in ids:
            if isinstance(c, tuple) or isinstance(c, list):
                cols.append(c[1])
            else:
                cols.append(c)
        new_cols = []
        for c in cols:
            if c.endswith("_"):
                if c == "all_":
                    new_cols.append("root")
                else:
                    new_cols.append(c[:-1])
                continue
            new_cols.append(c)
        return new_cols
    
    def mape(y, ypred):
        y = np.array(y).ravel()
        ypred = np.array(ypred).ravel()
        return np.abs(y-ypred) / y

    val_data = data[data["flag"] == "val"]
    val_data.fillna("", inplace=True)
    val_data.loc[:, "id"] = val_data.loc[:, "series"] + "_" + val_data.loc[:, "sku"]
    val_data["residual"] = mape(val_data["y"], val_data["ypred"])
    forecasts = pd.pivot_table(val_data, values=["ypred"], index=["date"], columns=["id"])
    ytrue = pd.pivot_table(val_data, values=["y"], index=["date"], columns=["id"])
    residuals = pd.pivot_table(val_data, values=["residual"], index=["date"], columns=["id"])
    forecasts.columns = clear_ids(forecasts.columns)
    residuals.columns = clear_ids(residuals.columns)
    val_data["id"] = clear_ids(val_data["id"])
    res = optimal_reconcilation(forecasts, tree, method="mint", residuals=residuals)
    
    # rearrange and show comparison between origin and reconcilated
    cols = [c for c in res.columns.tolist() if c != "id"]
    res = pd.melt(res, id_vars=["id"], value_vars=cols)
    res.columns = ["id", "date", "ypred_new"]
    res = pd.merge(res, val_data[["id", "y", "ypred"]], how="left", on="id")
    res.loc[res["id"] == "root", "id"] = "all"
    res["mape"] = mape(res["y"], res["ypred"])
    res["mape_new"] = mape(res["y"], res["ypred_new"])
    res[["series", "sku"]] = res["id"].str.split("_", expand=True)
    res.drop(columns=["id"], inplace=True)
    
    return res 

if __name__ == "__main__":
    res = example()
    print("result: ", res)
