#-*-coding:utf-8 
#@Author: Jing Wang 
#@Date: 2020-10-28 14:46:49 
#@Last Modified by: Jing Wang 
#@Last Modified time: 2020-10-28 14:46:49 
#@reference: https://www.cnblogs.com/wuliytTaotao/p/10814770.html

'''
Nonegative Matrix Factorization Method:
Dealing with NA value
'''
import numpy as np

class NMF(object):

    def __init__(self, k=3, alpha=1e-4, beta=0.5, 
        max_iters=20000, epsilon=1e-3, 
        normalize=False, bias=True):
        '''
        Args:
            k (int): 分解矩阵的rank, k < min(m, n), m, n are dimensions of input
            alpha (float): learning rate, 学习率
            beta (float): regularizer coefficients，正则项系数
            max_iters (int): maximum iteration, 最大迭代次数
            epsilon (float): error tolerance, error容忍度
            normalize (bool): 是否对X使用normalize
            bias (bool): 是否使用bias
        
        Note:
            - 如果矩阵很大，建议学习率alpha小一些（如1e-5)，不然容易出现nan或者无穷大，如果矩阵较小，可取大一些（如1e-3）。
            - 如果想要尽可能精确，k不能取太小，要贴近min(m, n)
        '''
        self.k = k 
        self.alpha = alpha 
        self.beta = beta 
        self.max_iters = max_iters
        self.epsilon = epsilon
        self.normalize = normalize
        self.bias = bias 
    
    def fit(self, X):
        '''
        Args:
            X (array like)

            如果没有bias，迭代过程为：
                e_{ij} = x_{ij} - \sum_{l=1}^k u_{il}v_{jl}
                u_{il} = u_{il} + alpha * (2 e_{ij}v_{jl} - beta u_{il})
                v_{jl} = v_{jl} + alpha * (2 e_{ij}u_{il} - beta v_{jl})
            如果有bias，迭代过程为：
                e_{ij} = x_{ij} - \sum_{l=1}^k u_{il}v_{jl} - b - bu_i - bv_j
                u_{il} = u_{il} + alpha * (2 e_{ij}v_{jl} - beta u_{il})
                v_{jl} = v_{jl} + alpha * (2 e_{ij}u_{il} - beta v_{jl})
                bu_i = bu_i + alpha * (2 e_{ij} - beta * bu_i)
                bv_j = bv_j + alpha * (2 e_{ij} - beta * bv_j)
        '''
        X = np.asarray(X)
        m, n = X.shape
        # normalize X 
        if self.normalize:
            X = self._normalize(X)
        
        # initialize U and V
        self.U_ = np.random.uniform(size=(m, self.k))
        self.V_ = np.random.uniform(size=(n, self.k))
        if self.bias:
            # initialize b, bu, bv 
            self.b_ = X[~np.isnan(X)].mean()
            self.bu_ = np.zeros(m)
            self.bv_ = np.zeros(n)

        losses = []
        for t in range(self.max_iters):
            Xhat = self.U_.dot(self.V_.T) 
            if self.bias:
                Xhat += self.b_ + self.bu_[:, np.newaxis] + self.bv_[np.newaxis, :] 
            e = X - Xhat
            resid = e[~np.isnan(X)]
            loss = np.sum(np.square(resid))
            e[np.isnan(X)] = 0
            self.U_ += self.alpha * (2 * e.dot(self.V_) - self.beta * self.U_)
            self.V_ += self.alpha * (2 * e.T.dot(self.U_) - self.beta * self.V_)
            if self.bias:
                self.bu_ = self.alpha * (2 * np.sum(e, axis=1) - self.beta * self.bu_)
                self.bv_ = self.alpha * (2 * np.sum(e, axis=0) - self.beta * self.bv_)
            losses.append(loss)
            if loss < self.epsilon:
                break
        self.Xhat_ = self.U_.dot(self.V_.T)
        if self.bias:
            self.Xhat_ += self.b_ + self.bu_[:, np.newaxis] + self.bv_[np.newaxis, :] 
        if self.normalize:
            self.Xhat_ = self._denormalize(self.Xhat_)

    def _normalize(self, X):
        '''
        Normalize X for nonegative matrix factorization
        将X标准化，加速converge

        标准化方法（只计算非NaN的位置)：
            (X - X.max) / (X.max - X.min)
        '''
        self.max = X[~np.isnan(X)].max()
        self.min = X[~np.isnan(X)].min()
        X[~np.isnan(X)] = (X[~np.isnan(X)] - self.max) / ((self.max - self.min))
        return X  
    
    def _denormalize(self, Xhat):
        '''
        Inverse Normalize, estimated Xhat reverse to the original range of X 
        将结果Xhat回到X的范围内

        Xhat * (X.max - X.min) + X.max 
        '''
        return Xhat * (self.max - self.min) + self.max 


if __name__ == '__main__':
    X = np.random.uniform(0, 100, size=(100, 100))
    import random 
    nan_count = 100
    cache = set()
    for _ in range(nan_count):
        i, j = random.choice(range(100)), random.choice(range(100))
        while (i, j) in cache:
            i, j = random.choice(range(100)), random.choice(range(100))
        cache.add((i, j))
        X[i, j] = np.nan 
    
    # X = np.array([
    #     [5, 3, 0, 1],
    #     [4, 0, 0, 1],
    #     [1, 1, 0, 5],
    #     [1, 0, 0, 4],
    #     [0, 1, 5, 4],
    # ], dtype=np.float)

    # # replace 0 with np.nan
    # X[X == 0] = np.nan
    print(X)
    clf = NMF(k=80)
    clf.fit(X)
    print("Xhat: ", clf.Xhat_)

    e = X - clf.Xhat_
    # print(e[~np.isnan(X)].sum())
