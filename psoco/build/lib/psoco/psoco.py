#!/usr/bin/python 3.7
#-*-coding:utf-8-*-


'''
Particle Swarm Optimization Constraint Optimization
Author: Jing Wang (jingw2@foxmail.com)
'''

import math 
import numpy as np 
import random 

class PSOCO:

    def __init__(self,
        particle_size=2000,
        max_iter=1000,
        sol_size=7,
        fitness=None,
        constraints=None):
        '''
        Particle Swarm Optimization Constraint Optimization
        Args:
            particle_size (int): 粒子数量
            max_iter (int): 最大迭代次数
            sol_size (int): 解的维度
            fitness (callable function): fitness函数，接受参数 x 为解
            constraints (list): 一系列约束条件，全部表示为 <= 0的形式
        '''
        self.c1 = 2 
        self.c2 = 2 
        self.w = 1.2 # 逐渐减少到0.1 
        self.kai = 0.73 
        self.vmax = 4 # 最大速度，防止爆炸
        self.particle_size = particle_size 
        self.max_iter = max_iter
        self.sol_size = sol_size

        # pso parameters 
        self.X = np.zeros((self.particle_size, self.sol_size))
        self.V = np.zeros((self.particle_size, self.sol_size))
        self.pbest = np.zeros((self.particle_size, self.sol_size))   #个体经历的最佳位置和全局最佳位置  
        self.gbest = np.zeros((1, self.sol_size))  
        self.p_fit = np.zeros(self.particle_size) # 每个particle的最优值
        self.fit = float("inf")
        self.iter = 1

        self.constraints = constraints
        if constraints is not None:
            for cons in constraints:
                if not callable(cons):
                    raise Exception("Constraint is not callable or None!")
        if not callable(fitness):
            raise Exception("Fitness is not callable!")
        self.sub_fitness = fitness
    
    def fitness(self, x, k):
        '''fitness函数 + 惩罚项'''
        obj = self.sub_fitness(x)
        obj = obj.reshape((-1, 1))
        return obj + self.h(k) * self.H(x)
    
    def init_Population(self, low=0, high=1):  
        '''初始化粒子'''
        self.X = np.random.uniform(size=(self.particle_size, self.sol_size), low=low, high=high)
        self.V = np.random.uniform(size=(self.particle_size, self.sol_size))
        self.pbest = self.X 
        self.p_fit = self.fitness(self.X, 1)
        best = np.min(self.p_fit)
        best_idx = np.argmin(self.p_fit)
        if best < self.fit:
            self.fit = best 
            self.gbest = self.X[best_idx] 
    
    def solve(self):  
        '''求解'''
        fitness = []  
        w_step = (self.w - 0.1) / self.max_iter
        for k in range(1, self.max_iter+1):  
            tmp_obj = self.fitness(self.X, k) 

            # 更新pbest 
            stack = np.hstack((tmp_obj.reshape((-1, 1)), self.p_fit.reshape((-1, 1))))
            best_arg = np.argmin(stack, axis=1).ravel().tolist()
            self.p_fit = np.minimum(tmp_obj, self.p_fit)
            X_expand = np.expand_dims(self.X, axis=2)
            p_best_expand = np.expand_dims(self.pbest, axis=2)
            concat = np.concatenate((X_expand, p_best_expand), axis=2)
            self.pbest = concat[range(0, len(best_arg)), :, best_arg]

            # 更新fit和gbest 
            best = np.min(self.p_fit)
            best_idx = np.argmin(self.p_fit)
            if best < self.fit:
                self.fit = best 
                self.gbest = self.X[best_idx]

            # 更新速度 

            # 分粒子更新
            # for i in range(self.particle_size):  
            #     self.V[i] = self.w*self.V[i] + self.c1*random.random()*(self.pbest[i] - self.X[i]) + \
            #                 self.c2*random.random()*(self.gbest - self.X[i])  
            #     self.X[i] = self.X[i] + self.V[i] 

            rand1 = np.random.random(size=(self.particle_size, self.sol_size))
            rand2 = np.random.random(size=(self.particle_size, self.sol_size))
            # 群体更新
            self.V = self.kai * (self.w*self.V + self.c1*rand1*(self.pbest - self.X) + \
                        self.c2*rand2*(self.gbest - self.X))
            self.V[self.V > self.vmax] = self.vmax
            self.V[self.V < -self.vmax] = -self.vmax
            
            self.X = self.X + self.V  
            fitness.append(self.fit)  
            self.w -= w_step

        return fitness 
    
    # relative violated function
    def q(self, g):
        return np.maximum(0, g)
    
    # power of penalty function 
    def gamma(self, qscore):
        result = np.zeros_like(qscore)
        result[qscore >= 1] = 2
        result[qscore < 1] = 1 
        return result
    
    # multi-assignment function
    def theta(self, qscore):
        result = np.zeros_like(qscore)
        result[qscore < 0.001] = 10 
        result[qscore <= 0.1] = 10 
        result[qscore <= 1] = 100
        result[qscore > 1] = 300
        return result
    
    # penalty score 
    def h(self, k):
        return k * math.sqrt(k)
    
    # penalty factor
    def H(self, x):
        res = 0
        for cons_func in self.constraints:
            qscore = self.q(cons_func(x))
            if len(qscore.shape) == 1 or qscore.shape[1] == 1:
                qscore = qscore.reshape((-1, 1))
                res += self.theta(qscore) * np.power(qscore, self.gamma(qscore))
            else:
                for i in range(qscore.shape[1]):
                    qscorei = qscore[:, i].reshape((-1, 1))
                    res += self.theta(qscorei) * \
                        np.power(qscorei, self.gamma(qscorei))
        return res 
