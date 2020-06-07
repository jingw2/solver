#!usr/bin/python 3.7
#-*-coding:utf-8-*-

'''
@file: base.py, base for clustering algorithm
@Author: Jing Wang (jingw2@foxmail.com)
@Date: 06/07/2020
'''
from scipy.spatial.distance import cdist

class Base(object):

    def __init__(self, n_clusters, max_iters, distance_func=cdist):
        self.n_clusters = n_clusters 
        self.max_iters = max_iters
        if distance_func is not None and not callable(distance_func):
            raise Exception("Distance function is not callable")
        self.distance_func = distance_func

    def fit(self, X):
        pass 

    def predict(self, X):
        pass 
