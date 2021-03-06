## Particle Swarm Optimization Constraint Optimization Solver
[![PyPI version](https://badge.fury.io/py/psoco.svg)](https://badge.fury.io/py/psoco)
![PyPI - Downloads](https://img.shields.io/pypi/dm/psoco)
### Arguments
|Name |Type|Default Value|
|-----|----|-------------|
|particle_size|int|2000|
|max_iter|int|1000|
|sol_size|int|7|
|fitness|function|null|
|constraints|a list of functions|null|

### Usage
![equation](https://latex.codecogs.com/svg.latex?%5Cbegin%7Balign*%7D%20%5Cmin%20%5C%20f%28x%29%20%26%3D%20%28x_1%20-%202%29%5E2%20&plus;%20%28x_2%20-%201%29%5E2%20%5C%5C%20s.t.%20%5C%20x_1%20%26%3D%20x_2%20-%201%20%5C%5C%20x_1%5E2/4%26&plus;x_2%5E2-1%20%5Cleq%200%20%5Cend%7Balign*%7D)

Transform constraints, it becomes: 

![equation](https://latex.codecogs.com/svg.latex?%5Cbegin%7Balign*%7D%20%5Cmin%20%5C%20f%28x%29%20%26%3D%20%28x_1%20-%202%29%5E2%20&plus;%20%28x_2%20-%201%29%5E2%20%5C%5C%20s.t.%20%5C%20x_1%20-%20%26x_2%20&plus;%201%20%5Cleq%200%20%5C%5C%20-x_1%20&plus;%20%26x_2%20-%201%20%5Cleq%200%20%5C%5C%20x_1%5E2/4%26&plus;x_2%5E2-1%20%5Cleq%200%20%5Cend%7Balign*%7D)

Note: In order to faster search optimal solutions, please initialize solutions with specific low and high.
```python
import psoco
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
for _ in range(num_runs):
    pso = psoco.PSOCO(sol_size=2, fitness=objective, constraints=constraints)
    pso.h = new_penalty_func
    pso.init_Population(low=0, high=1) # x并集的上下限，默认为0和1
    pso.solve()
    # best solutions
    x = pso.gbest.reshape((1, -1))
```
### Reference
* [Particle Swarm Optimization Method for
Constrained Optimization Problems](https://www.cs.cinvestav.mx/~constraint/papers/eisci.pdf)
