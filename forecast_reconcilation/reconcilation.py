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

def dict_to_array(forecasts: dict, nodenames: list):
    '''
    dict to array
        make sure any of key in nodenames in forecasts
    '''
    y = []
    t = 1 
    num_nodes = len(nodenames)
    for idx, node in enumerate(nodenames):
        assert node in forecasts
        y.append(forecasts[node])
        t = len(forecasts[node])
    y = np.asarray(y).reshape((num_nodes, t))
    return y 

def top_down(forecasts: dict, tree: HierarchyTree):
    '''
    Top down method
    从上至下拆分
        1. 按照历史比例
        2. 按照历史平均比例
        3. 按预测比例
    '''
    raise NotImplementedError

def bottom_up(forecasts: dict, tree: HierarchyTree):
    '''
    自下而上汇总
        y_tilde = S y_hat_bottom
    '''
    nodenames = list(tree.nodes.keys())
    S = get_summing_matrix(tree)
    ypred = dict_to_array(forecasts, nodenames)
    num_bottom_level = tree.num_bottom_level
    bottom_pred = ypred[-num_bottom_level:, :]
    y = S @ bottom_pred
    results = {}
    for idx, name in enumerate(nodenames):
        results[name] = y[idx]
    return results

def optimal_reconcilation(forecasts: dict, tree: HierarchyTree, method="ols", 
        residuals: dict = None):
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
    S = get_summing_matrix(tree)
    ypred = dict_to_array(forecasts, nodenames)
    kh = 1 
    if method == "ols":
        Wh = np.eye(num_nodes) * kh 
    if method == "wls":
        residuals = dict_to_array(residuals, nodenames)
        What1 = residuals @ residuals.T 
        diag = np.eye(num_nodes) * np.diag(What1)
        Wh = kh * diag
    if method == "nseries":
        diag = np.eye(num_nodes) * np.diag(np.sum(S, axis=1))
        Wh = kh * diag
    if method == "mint":
        residuals = dict_to_array(residuals, nodenames)
        cov = np.cov(residuals)
        diag = np.eye(num_nodes) * np.diag(cov)
        Wh = kh * diag
    inv_Wh = np.linalg.inv(Wh)
    coef = S @ (np.linalg.inv(S.T @ inv_Wh @ S)) @ S.T @ inv_Wh
    y = coef @ ypred

    results = {}
    for idx, name in enumerate(nodenames):
        results[name] = y[idx]
    return results


if __name__ == "__main__":
    import numpy as np 
    stores = ["京东"]
    series = ["京东_红胖子", "京东_黑管", "京东_小钢笔"]
    skus = ["京东_红胖子_sku1", "京东_红胖子_sku2", 
        "京东_黑管_sku1", "京东_黑管_sku2",
        "京东_小钢笔_sku1", "京东_小钢笔_sku2"]
    total = {"root": series}
    # series_h = {k: [v for v in series if v.startswith(k)] for k in stores}
    skus_h = {k: [v for v in skus if v.startswith(k)] for k in series}
    hierarchy = {**total, **skus_h}
    
    tree = HierarchyTree.from_nodes(hierarchy)
    
    forecasts = {
        "root": [10000, 10000],
        "京东_红胖子": [3000, 2000],
        "京东_黑管": [5000, 4000],
        "京东_小钢笔": [3000, 2000],
        "京东_红胖子_sku1": [1200, 1000],
        "京东_红胖子_sku2": [1500, 2000],
        "京东_黑管_sku1": [3600, 2000],
        "京东_黑管_sku2": [2000, 3000],
        "京东_小钢笔_sku1": [1000, 500],
        "京东_小钢笔_sku2": [1000, 2000],
    }

    residuals = {
        "root": [10, 1000],
        "京东_红胖子": [150, 10],
        "京东_黑管": [100, 500],
        "京东_小钢笔": [300, 400],
        "京东_红胖子_sku1": [120, 100],
        "京东_红胖子_sku2": [150, 250],
        "京东_黑管_sku1": [360, 140],
        "京东_黑管_sku2": [200, 100],
        "京东_小钢笔_sku1": [100, 500],
        "京东_小钢笔_sku2": [100, 400],
    }

    # res = optimal_reconcilation(forecasts, tree, method="wls", residuals=residuals)
    
    res = bottom_up(forecasts, tree)

    print(res)
