# -*- coding:utf-8 -*-
#@author: Jing Wang
#@date: 09/24/2020

'''
层级预测后的调和
* 实现Forecasting Principles and Practice中的最优调和方法, 章节10.7
* 参考代码：https://github.com/carlomazzaferro/scikit-hts/blob/master/hts/functions.py
'''
from collections import OrderedDict
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
    return mat[1:]

def get_y(forecasts, nodenames):
    y = []
    t = 1 
    num_nodes = len(nodenames)
    for idx, node in enumerate(nodenames):
        y.append(forecasts[node])
        t = len(forecasts[node])
    y = np.asarray(y).reshape((num_nodes, t))
    return y 

def top_down():
    raise NotImplementedError

def bottom_up():
    raise NotImplementedError

def optimal_reconcilation(forecasts: dict, tree: HierarchyTree, method="ols", 
        residuals: dict = None):
    '''
    最优调和算法
    '''
    nodenames = list(tree.nodes.keys())[1:] # 去掉total
    num_nodes = tree.num_nodes - 1
    S = get_summing_matrix(tree)
    ypred = get_y(forecasts, nodenames)
    kh = 1 
    if method == "ols":
        Wh = np.eye(num_nodes) * kh 
    if method == "wls":
        residuals = get_y(residuals, nodenames)
        What1 = residuals @ residuals.T 
        diag = np.eye(num_nodes) * np.diag(What1)
        Wh = kh * diag
    if method == "nseries":
        diag = np.eye(num_nodes) * np.diag(np.sum(S, axis=1))
        Wh = kh * diag
    if method == "mint":
        ypred = get_y(forecasts, nodenames)
        cov = np.cov(residuals)
        diag = np.eye(num_nodes) * np.diag(cov)
        Wh = kh * diag
    inv_Wh = np.linalg.inv(Wh)
    coef = S @ (np.linalg.inv(S.T @ inv_Wh @ S)) @ S.T @ inv_Wh
    y = coef @ ypred
    return y


if __name__ == "__main__":
    import numpy as np 
    stores = ["京东"]
    series = ["京东_红胖子", "京东_黑管", "京东_小钢笔"]
    skus = ["京东_红胖子_sku1", "京东_红胖子_sku2", 
        "京东_黑管_sku1", "京东_黑管_sku2",
        "京东_小钢笔_sku1", "京东_小钢笔_sku2"]
    total = {"root": stores}
    series_h = {k: [v for v in series if v.startswith(k)] for k in stores}
    skus_h = {k: [v for v in skus if v.startswith(k)] for k in series}
    hierarchy = {**total, **series_h, **skus_h}
    
    tree = HierarchyTree.from_nodes(hierarchy)
    
    forecasts = {
        "京东": [10000, 10000],
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
        "京东": [10, 1000],
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

    y = reconcilation(forecasts, tree, method="mint", residuals=residuals)
    print(y)
    print(y[1])
    print(y[4] + y[5])
