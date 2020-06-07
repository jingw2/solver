#!usr/bin/python 3.7
#-*-coding:utf-8-*-

'''
@file: equi_siz.py, equal size clustering
@Author: Jing Wang (jingw2@foxmail.com)
@Date: 06/06/2020
@paper: Clustering with Size Constraints
@github reference: 
'''

from scipy.spatial.distance import cdist
import numpy as np 
from scipy.linalg import norm
import base

class EquiSize(base.Base):

    def __init__(self, n_clusters, m=2, max_iters=1000, distance_func=cdist, epsilon=1e-5, random_state=42):
        super(EquiSize, self).__init__(n_clusters, max_iters, distance_func)
        self.epsilon = epsilon
        assert m > 1
        self.m = m
        self.random_state = random_state
        self.alpha, self.beta = None, None

    def fit(self, X):

        np.random.seed(self.random_state)
        assert self.n_clusters >= 1 
        n_samples, n_dimensions = X.shape

        # initialize mu 
        self.u = np.random.random(size=(n_samples, self.n_clusters))
        self.u /= np.sum(self.u, axis=1).reshape((-1, 1))

        itr = 0
        while True:
            last_u = self.u.copy()
            # update centers
            self.centers = self.update_centers(X) 
            # update alpha and beta
            self.alpha, self.beta = self.update_alpha_beta(X)
            # update membership
            self.u = self.predict(X)
            if norm(self.u - last_u) < self.epsilon or itr >= self.max_iters:
                break 
            itr += 1

    def update_centers(self, X):
        '''
        Update centers based new u
        '''
        um = np.power(self.u, self.m) # (n_samples, n_clusters)
        centers = (X.T.dot(um)).T / np.sum(um, axis=0).reshape((-1, 1))
        return centers

    def update_alpha_beta(self, X):
        n_samples, _ = X.shape
        dist = self.distance_func(X, self.centers) # n_samples, n_clusters
        
        # calculate coefficients of beta
        

        # calculate right hand size of beta linear equation
        term3 = dist * np.sum(1. / dist, axis=1).reshape((-1, 1))
        rhs = n_samples / float(self.n_clusters) - np.sum(1. / term3, axis=0)
        rhs = rhs.reshape((-1, 1))

        beta = np.linalg.inv(coef).dot(rhs).ravel()

        # calculate alpha
        alpha = (2 - np.sum(beta.reshape((1, -1)) / dist, axis=1)) / (np.sum(dist, axis=1))
        
        return alpha, beta

    def predict(self, X):
        n_samples, _ = X.shape
        dist = self.distance_func(X, self.centers)
        alpha = np.tile(self.alpha.reshape((-1, 1)), (1, self.n_clusters))
        beta = np.tile(self.beta.reshape((1, -1)), (n_samples, 1))

        u = (alpha + beta) / (2 * dist) 
        # normalize 
        u = u / np.sum(u, axis=1).reshape((-1, 1))
        return u

if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    from matplotlib import pyplot as plt
    from seaborn import scatterplot as scatter
    from sklearn.metrics.pairwise import haversine_distances
    n_samples = 5000
    n_bins = 4  # use 3 bins for calibration_curve as we have 3 clusters here
    centers = [(-5, -5), (0, 0), (5, 5), (10, 10)]

    X, _ = make_blobs(n_samples=n_samples, n_features=2, cluster_std=1.0,
                    centers=centers, shuffle=False, random_state=42)
    n_points = 1000
    # demands = np.random.randint(1, 24, (n_points, 1))
    X = np.random.rand(n_points, 2)
    es = EquiSize(n_bins)

    es.fit(X)

    labels = np.argmax(es.u, axis=1)
    from collections import Counter
    print(Counter(labels))
