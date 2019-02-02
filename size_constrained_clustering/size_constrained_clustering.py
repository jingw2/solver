#!usr/bin/python 3.6
#-*-coding:utf-8-*-

'''
Author: Jing Wang
'''

import numpy as np
from sklearn.cluster import k_means_
import matplotlib.pyplot as plt
from copy import deepcopy
import collections
import random

def l2_distance_func(centers, X):

    n_points, n_features = X.shape
    n_centers = centers.shape[0]

    X_square = np.expand_dims(np.sum(np.square(X), axis=1), axis=1)
    centers_square = np.expand_dims(np.sum(np.square(centers),
                                           axis=1), axis=0)

    factors = -2 * X.dot(centers.T) # n_points, n_centers

    distance_matrix = np.tile(X_square, (1, n_centers)) + \
                      factors + np.tile(centers_square, (n_points, 1))

    return distance_matrix

def haversine_distance_func(centers, X):

    n_points, n_features = X.shape
    n_centers = centers.shape[0]

    assert n_features <= 3

    # centers, X = np.radians(centers), np.radians(X)

    centers = centers.astype(np.float64)
    Xlon = np.tile(X[:, 0].reshape((-1, 1)), (1, n_centers))
    centers_lon = np.tile(centers[:, 0].reshape(1, -1), (n_points, 1))
    Xlon = np.radians(Xlon)
    centers_lon = np.radians(centers_lon)

    dlon = Xlon - centers_lon

    Xlat = np.tile(X[:, 1].reshape((-1, 1)), (1, n_centers))
    centers_lat = np.tile(centers[:, 1].reshape(1, -1), (n_points, 1))
    Xlat = np.radians(Xlat)
    centers_lat = np.radians(centers_lat)

    dlat = Xlat - centers_lat

    a = np.sin(dlat / 2) ** 2 + np.cos(Xlat) * np.cos(centers_lat) * np.sin(dlon / 2.) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371

    distance_matrix = c * r
    # print("distance matrix shape: ", distance_matrix.shape)
    if n_features > 2:
        Xfloor = np.tile(X[:, 2].reshape((-1, 1)), (1, n_centers))
        centers_floor = np.tile(centers[:, 2].reshape(1, -1), (n_points, 1))
        distance_matrix += np.abs(Xfloor - centers_floor) * 3 * 10**(-3)

    # distance_matrix = distance_matrix / (15 * 3.6) # minute
    # normalized
    distance_matrix /= np.sum(distance_matrix)

    return distance_matrix


class MaxSizeConstrainedKmeans:

    def __init__(self, n_clusters, n_iters, max_size, distance_func=None):
        self.n_clusters = n_clusters
        self.n_iters = n_iters
        self.max_size = max_size
        if distance_func is not None and callable(distance_func):
            self.distance_func = distance_func
        else:
            self.distance_func = l2_distance_func


    def fit(self, X):
        n_points = X.shape[0]
        cluster_labels = [-1] * n_points

        centers = self.initial_centroids(X)

        X_copy = np.hstack((np.arange(n_points).reshape((-1, 1)), deepcopy(X)))

        last_cluster_labels = deepcopy(cluster_labels)
        for _ in range(self.n_iters):
            # random shuffle data, avoiding the effects of order
            # np.random.shuffle(X_copy)
            data = X_copy[:, 1:]
            distance_matrix = self.distance_func(centers, data)
            clusters_dict = collections.defaultdict(list)

            for index in range(n_points):
                point_id, x, y = X_copy[index]
                cluster_list = np.argsort(distance_matrix[index]).tolist()

                for cluster_id in cluster_list:
                    if len(clusters_dict[cluster_id]) < self.max_size[cluster_id]:
                        cluster_labels[int(point_id)] = cluster_id
                        clusters_dict[cluster_id].append(index)
                        break

            if np.mean(np.asarray(cluster_labels) - np.asarray(last_cluster_labels)) == 0:
                break

            last_cluster_labels = cluster_labels

            # reassign centers
            for cluster_id, point_list in clusters_dict.items():
                centers[cluster_id] = np.mean(data[point_list], axis=0)

        return cluster_labels


    def initial_centroids(self, X):
        # X_square_norm = np.sum(np.square(X), axis=1)
        # random_state = np.random.RandomState(123)
        # centers = k_means_._k_init(X, self.n_clusters, X_square_norm, random_state)
        random_indice = random.sample(range(X.shape[0]), self.n_clusters)
        centers = X[random_indice]
        return centers


class DeterministicAnnealing(object):

    def __init__(self, n_clusters, capacity, n_iters, distance_func_type="l2", T=None):
        self.n_clusters = n_clusters
        self.capacity = capacity
        self.n_iters = n_iters
        self.lamb = [i / sum(capacity) for i in capacity]
        self.distance_func_type = distance_func_type
        if self.distance_func_type == "l2":
            self.distance_func = l2_distance_func
        elif self.distance_func_type == "haversine":
            self.distance_func = haversine_distance_func
        self.beta = None
        self.T = T


    def fit(self, X, demands):
        # setting T
        # if self.distance_func_type == "l2" and self.T is None:
        #     self.T = 1. / np.max(np.linalg.eigvals(np.cov(X)))
        # elif self.distance_func_type == "haversine" and self.T is None:
        #     self.T = 0.0000001

        T = [1, 0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
        solutions = []
        diff_list = []
        is_early_terminated = False
        for t in T:
            self.T = t
            centers = self.initial_centers(X)
            demands = demands / sum(demands)
            eta = self.lamb
            labels = None
            for _ in range(self.n_iters):
                self.beta = 1. / self.T
                distance_matrix = self.distance_func(centers, X)
                eta = self.update_eta(eta, demands, distance_matrix)
                gibbs = self.update_gibbs(eta, distance_matrix)
                centers = self.update_centers(demands, gibbs, X)
                self.T *= 0.999

                labels = np.argmax(gibbs, axis=1)

                if self.is_satisfied(labels): break

            solutions.append([labels, centers])
            resultant_clusters = len(collections.Counter(labels))

            diff_list.append(abs(resultant_clusters - self.n_clusters))
            if resultant_clusters == self.n_clusters:
                is_early_terminated = True
                break

        # modification for non-strictly satisfaction, only works for one demand per location
        # labels = self.modify(labels, centers, distance_matrix)
        if not is_early_terminated:
            best_index = np.argmin(diff_list)
            labels, centers = solutions[best_index]

        return labels, centers

    def modify(self, labels, centers, distance_matrix):
        centers_distance = self.distance_func(centers, centers)
        adjacent_centers = {i: np.argsort(centers_distance, axis=1)[i, 1:3].tolist() for i in range(self.n_clusters)}
        while not self.is_satisfied(labels):
            count = collections.Counter(labels)
            cluster_id_list = list(count.keys())
            random.shuffle(cluster_id_list)
            for cluster_id in cluster_id_list:
                num_points = count[cluster_id]
                diff = num_points - self.capacity[cluster_id]
                if diff <= 0: continue
                adjacent_cluster = None
                adjacent_cluster = random.choice(adjacent_centers[cluster_id])
                if adjacent_cluster is None: continue
                cluster_point_id = np.where(labels==cluster_id)[0].tolist()
                diff_distance = distance_matrix[cluster_point_id, adjacent_cluster] \
                                - distance_matrix[cluster_point_id, cluster_id]

                remove_point_id = np.asarray(cluster_point_id)[np.argsort(diff_distance)[:diff]]
                labels[remove_point_id] = adjacent_cluster

        return labels

    def initial_centers(self, X):
        selective_centers = random.sample(range(X.shape[0]), self.n_clusters)
        centers = X[selective_centers]
        return centers

    def is_satisfied(self, labels):
        count = collections.Counter(labels)
        for cluster_id in range(len(self.capacity)):
            if cluster_id not in count:
                return False
            num_points = count[cluster_id]
            if num_points > self.capacity[cluster_id]:
                return False
        return True


    def update_eta(self, eta, demands, distance_matrix):
        n_points, n_centers = distance_matrix.shape
        eta_repmat = np.tile(np.asarray(eta).reshape(1, -1), (n_points, 1))

        exp_term = np.exp(- self.beta * distance_matrix)

        divider = exp_term / np.sum(np.multiply(exp_term,
                            eta_repmat), axis=1).reshape((-1, 1))

        eta = np.divide(np.asarray(self.lamb),
                        np.sum(divider * demands, axis=0))

        return eta

    def update_gibbs(self, eta, distance_matrix):
        n_points, n_centers = distance_matrix.shape
        eta_repmat = np.tile(np.asarray(eta).reshape(1, -1), (n_points, 1))

        exp_term = np.exp(- self.beta * distance_matrix)

        factor = np.multiply(exp_term, eta_repmat)

        gibbs = factor / np.sum(factor, axis=1).reshape((-1, 1))

        return gibbs

    def update_centers(self, demands, gibbs, X):
        n_points, n_features = X.shape
        divide_up = gibbs.T.dot(X * demands)# n_cluster, n_features
        p_y = np.sum(gibbs * demands, axis=0) # n_cluster,

        p_y_repmat = np.tile(p_y.reshape(-1, 1), (1, n_features))
        centers = np.divide(divide_up, p_y_repmat)

        return centers

if __name__ == "__main__":

    X = []
    n_points = 1000
    # demands = np.random.randint(1, 24, (n_points, 1))
    X = np.random.rand(n_points, 2)
    demands = np.ones((n_points, 1))
    n_clusters = 4
    n_iters = 100
    max_size = [n_points / n_clusters] * n_clusters

    da = DeterministicAnnealing(n_clusters, max_size, n_iters, "l2")
    labels, centers = da.fit(X, demands)

    print(centers)
    labels_demand_cnt = {}
    for i, label in enumerate(labels):
        labels_demand_cnt[label] = labels_demand_cnt.get(label, 0) + demands[i][0]

    sorted_labels = sorted(labels_demand_cnt.items())
    x = list(range(n_clusters))
    y = [j for i, j in sorted_labels]
    plt.scatter(X[:, 0], X[:, 1], c=labels)
    print(collections.Counter(labels_demand_cnt))
    # plt.show()
    plt.xlabel("X")
    plt.ylabel("Y")
    # plt.bar(x, y)
    plt.show()