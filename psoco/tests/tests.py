#-*-coding:utf-8-*-

import sys 
import os 
import numpy as np 
dirpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(dirpath)
from psoco import psoco
import math 

def objective(x):
    '''create objectives based on inputs x as 2D array'''
    return (x[:, 0] - 2) ** 2 + (x[:, 1] - 1) ** 2 


def constraints1(x):
    '''create constraint1 based on inputs x as 2D array'''
    return x[:, 0] - 2 * x[:, 1] + 1 


def constraints2(x):
    '''create constraint2 based on inputs x as 2D array'''
    return - (x[:, 0] - 2 * x[:, 1] + 1)


def constraints3(x):
    '''create constraint3 based on inputs x as 2D array'''
    return x[:, 0] ** 2 / 4. + x[:, 1] ** 2 - 1

def new_penalty_func(k):
    '''Easy Problem can use \sqrt{k}'''
    return math.sqrt(k)
    
constraints = [constraints1, constraints2, constraints3]
num_runs = 10
# random parameters lead to variations, so run several time to get mean
sol_size = 2
results = np.zeros((num_runs, sol_size))
for r in range(num_runs):
    pso = psoco.PSOCO(sol_size=sol_size, fitness=objective, constraints=constraints)
    pso.h = new_penalty_func
    pso.init_Population(low=0, high=1) # x并集的上下限，默认为0和1
    pso.solve()
    # best solutions
    x = pso.gbest.reshape((1, -1))
    results[r] = x 

results = np.mean(results, axis=0)
print("results: ", results)
results = [round(r, 2) for r in results]
assert results == [0.82, 0.91]